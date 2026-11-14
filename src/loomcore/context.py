"""The context store: which part of the history a model request carries (spec 3.5).

Everything here is pure. The history is cut into items (see `conversation.exchanges`),
each with its token count measured at the edge, and `pack` and `compact` only choose
among them. A model never sees a `tool_use` without its `tool_result`, because an
item is a whole exchange.
"""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

from pydantic import JsonValue

from loomcore.events import JsonObject

ItemKind = Literal["task", "exchange", "hint"]
Counter = Callable[[Sequence[JsonObject]], int]  # tokens in a list of messages
CLEARED = "[tool result cleared to save context]"
PLAN_TOOL = "update_plan"  # the exchange of its latest successful call is pinned
STRATEGY = "clear_tool_results"  # what `context_compacted` names for `compact`


@dataclass(frozen=True)
class Cleared:
    """An exchange with its tool results replaced by `CLEARED`."""

    messages: tuple[JsonObject, ...]
    tokens: int


@dataclass(frozen=True)
class Item:
    id: str  # "task", "hint:<run>" or "seq:<run>:<seq of the model_call>"
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
    cleared: tuple[str, ...] = ()  # ids cleared in this window, by `pack` or `compact`


def pack(candidates: Sequence[Item], budget: int, pinned: Sequence[Item]) -> Packed:
    """Pinned items are always in. If everything together is over the budget, the
    oldest unpinned tool results are cleared first, so nothing is dropped that a
    cleared form would have kept. The rest are taken by value per token, a later
    item being worth more, and an item that no longer fits is skipped. Ties are
    broken by order, then id, so the result never depends on the input order.

    Raises `ValueError` if a pinned item is among the candidates, a candidate is
    flagged pinned, or two items share an id: the caller has split them wrongly.
    """
    _check_split(candidates, pinned)
    base = sum(item.tokens for item in pinned)
    if base > budget:
        return Packed(_in_order(pinned), _in_order(candidates), base, budget, over=True)
    total = base + sum(item.tokens for item in candidates)
    items, cleared = _clear_oldest(_in_order(candidates), total, budget)
    left = budget - base
    chosen: list[Item] = []
    for item in _by_value(items):
        if item.tokens <= left:
            chosen.append(item)
            left -= item.tokens
    kept = {item.id for item in chosen}
    dropped = [item for item in items if item.id not in kept]
    selected = _in_order([*pinned, *chosen])
    shown = tuple(
        item_id for item_id in cleared if item_id in kept
    )  # not a dropped one
    return Packed(selected, _in_order(dropped), budget - left, budget, cleared=shown)


def compact(packed: Packed, compact_at: float, compact_to: float) -> Packed:
    """At `compact_at` of the budget, clear the tool results of the oldest items
    until the window is down to `compact_to` of it. Pinned items are left alone, and
    so is a window that is already within its threshold."""
    if packed.over or packed.tokens < compact_at * packed.budget:
        return packed
    items, cleared = _clear_oldest(
        packed.selected, packed.tokens, compact_to * packed.budget
    )
    if not cleared:
        return packed  # nothing left to clear: a second call changes nothing
    tokens = sum(item.tokens for item in items)
    return replace(
        packed,
        selected=tuple(items),
        tokens=tokens,
        cleared=(*packed.cleared, *cleared),
    )


def _clear_oldest(
    items: Sequence[Item], total: int, target: float
) -> tuple[tuple[Item, ...], tuple[str, ...]]:
    """Replace the tool results of the oldest unpinned items with their cleared form
    until the window (`total` tokens now, `items` and perhaps others) is down to
    `target` or none is left."""
    out: list[Item] = []
    cleared: list[str] = []
    for item in items:
        form = _smaller_form(item)
        if form is not None and total > target:
            total -= item.tokens - form.tokens
            cleared.append(item.id)
            out.append(_as_cleared(item, form))
        else:
            out.append(item)
    return tuple(out), tuple(cleared)


def _smaller_form(item: Item) -> Cleared | None:
    """The cleared form, if the item may be cleared and it is smaller. A formatter
    of the caller's own need not promise that, so the check is here too."""
    form = item.cleared
    usable = not item.pinned and form is not None and form.tokens < item.tokens
    return form if usable else None


def _as_cleared(item: Item, cleared: Cleared) -> Item:
    return replace(item, messages=cleared.messages, tokens=cleared.tokens, cleared=None)


def clear_results(messages: Sequence[JsonObject]) -> tuple[JsonObject, ...]:
    """The messages with the content of every tool result replaced by `CLEARED`
    (the `tool_use_id` and `is_error` stay)."""
    return tuple(_clear_message(message) for message in messages)


def apply_cleared(
    items: Sequence[Item], cleared: Collection[str], count: Counter
) -> list[Item]:
    """Show the items cleared in an earlier window in their cleared form, so a
    window starts from where the last one left off. The messages are blanked here,
    from the log's own text, not read from `Item.cleared`, so what is shown does
    not depend on what a counter said when the item was built."""
    return [
        _shown_cleared(item, count) if item.id in cleared and not item.pinned else item
        for item in items
    ]


def _shown_cleared(item: Item, count: Counter) -> Item:
    blank = clear_results(item.messages)
    return replace(item, messages=blank, tokens=count(blank), cleared=None)


def _clear_message(message: JsonObject) -> JsonObject:
    content = message["content"]
    if not isinstance(content, list):
        return message
    kept: list[JsonValue] = [
        {**block, "content": CLEARED}
        if isinstance(block, dict) and block.get("type") == "tool_result"
        else block
        for block in content
    ]
    return {**message, "content": kept}


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
