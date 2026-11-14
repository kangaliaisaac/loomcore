"""The context store: which part of the history a model request carries (spec 3.5).

Everything here is pure. The history is cut into items (see `conversation.exchanges`),
each with its token count measured at the edge, and `pack` and `compact` only choose
among them. A model never sees a `tool_use` without its `tool_result`, because an
item is a whole exchange.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

from loomcore.events import JsonObject

ItemKind = Literal["task", "exchange", "hint"]
CLEARED = "[tool result cleared to save context]"
STRATEGY = "clear_tool_results"  # what `context_compacted` names for `compact`


@dataclass(frozen=True)
class Cleared:
    """An exchange with its tool results replaced by `CLEARED`."""

    messages: tuple[JsonObject, ...]
    tokens: int


@dataclass(frozen=True)
class Item:
    id: str  # "task", "hint:<seq>" or "seq:<seq of the model_call>"
    kind: ItemKind
    order: int  # position in the history; a later item is worth more
    tokens: int  # counted at the edge, never here
    messages: tuple[JsonObject, ...]
    pinned: bool = False
    cleared: Cleared | None = None  # None when there is no tool result to clear


@dataclass(frozen=True)
class Packed:
    selected: tuple[Item, ...]  # in history order
    dropped: tuple[Item, ...]
    tokens: int
    budget: int
    over: bool = False  # the pinned items alone do not fit in the budget
    cleared: tuple[str, ...] = ()  # ids of the items `compact` cleared


def pack(candidates: Sequence[Item], budget: int, pinned: Sequence[Item]) -> Packed:
    """Pinned items are always in. The rest are taken by value per token, a later
    item being worth more, and an item that no longer fits is skipped. Ties are
    broken by order, then id, so the result never depends on the input order.

    Raises `ValueError` if a pinned item is among the candidates, a candidate is
    flagged pinned, or two items share an id: the caller has split them wrongly.
    """
    _check_split(candidates, pinned)
    base = sum(item.tokens for item in pinned)
    if base > budget:
        return Packed(_in_order(pinned), _in_order(candidates), base, budget, over=True)
    left = budget - base
    chosen: list[Item] = []
    for item in _by_value(candidates):
        if item.tokens <= left:
            chosen.append(item)
            left -= item.tokens
    kept = {item.id for item in chosen}
    dropped = [item for item in candidates if item.id not in kept]
    selected = _in_order([*pinned, *chosen])
    return Packed(selected, _in_order(dropped), budget - left, budget)


def compact(packed: Packed, compact_at: float, compact_to: float) -> Packed:
    """At `compact_at` of the budget, clear the tool results of the oldest items
    until the window is down to `compact_to` of it. Pinned items are left alone, and
    so is a window that is already within its threshold."""
    if packed.over or packed.tokens < compact_at * packed.budget:
        return packed
    tokens = packed.tokens
    items = list(packed.selected)
    cleared: list[str] = []
    for k, item in enumerate(items):
        if tokens <= compact_to * packed.budget:
            break
        if item.pinned or item.cleared is None:
            continue
        tokens -= item.tokens - item.cleared.tokens
        items[k] = replace(
            item,
            messages=item.cleared.messages,
            tokens=item.cleared.tokens,
            cleared=None,
        )
        cleared.append(item.id)
    if not cleared:
        return packed  # nothing left to clear: a second call changes nothing
    return replace(packed, selected=tuple(items), tokens=tokens, cleared=tuple(cleared))


def apply_cleared(items: Sequence[Item], cleared: Collection[str]) -> list[Item]:
    """Show the items whose ids were cleared in an earlier window in their cleared
    form, so a window starts from where the last one left off."""
    return [
        replace(
            item,
            messages=item.cleared.messages,
            tokens=item.cleared.tokens,
            cleared=None,
        )
        if item.id in cleared and item.cleared is not None and not item.pinned
        else item
        for item in items
    ]


def _check_split(candidates: Sequence[Item], pinned: Sequence[Item]) -> None:
    ids = [item.id for item in (*candidates, *pinned)]
    if len(set(ids)) != len(ids):
        raise ValueError("two items share an id; each item goes in once")
    if any(item.pinned for item in candidates):
        raise ValueError("a pinned item is among the candidates; pass it as pinned")


def _by_value(items: Sequence[Item]) -> list[Item]:
    """Highest value per token first (exact, not floating point). An item of no
    tokens is not ranked first, but it always fits, so it is always taken."""
    return sorted(items, key=lambda item: (_rank(item), item.id))


def _rank(item: Item) -> tuple[Fraction, int]:
    value = Fraction(item.order + 1, max(item.tokens, 1))
    return -value, -item.order


def _in_order(items: Sequence[Item]) -> tuple[Item, ...]:
    return tuple(sorted(items, key=lambda item: (item.order, item.id)))
