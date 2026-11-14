"""The window a model request carries (spec 3.5): the context store applied to a log.

Pure. The driver calls it every turn and a hand-written loop can too. What earlier
windows dropped and cleared comes in as ids, folded from the log, so the window is a
function of the log and of what the counter says now; the ids it returns are what is
newly dropped and newly cleared, for the caller to record.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from loomcore.context import (
    Counter,
    Packed,
    apply_cleared,
    compact,
    pack,
)
from loomcore.conversation import exchanges, render
from loomcore.events import Event, JsonObject


@dataclass(frozen=True)
class Carried:
    """Ids dropped and cleared by earlier windows."""

    dropped: Collection[str] = ()
    cleared: Collection[str] = ()


@dataclass(frozen=True)
class Budget:
    """Tokens left for the history (after the system prompt and the tool schemas),
    and the fractions of it at which compaction starts and stops."""

    tokens: int
    compact_at: float
    compact_to: float


@dataclass(frozen=True)
class Window:
    messages: list[JsonObject]
    packed: Packed | None  # None when there is no budget
    tokens_before: int  # the window's tokens before any clearing of this turn
    newly_dropped: tuple[str, ...]  # to record in `context_packed.dropped`
    newly_cleared: tuple[str, ...]  # to record in `context_compacted.cleared`


def _no_count(messages: Sequence[JsonObject]) -> int:
    return 0


def build_window(
    task: JsonValue,
    events: Sequence[Event],
    carried: Carried,
    budget: Budget | None,
    count: Counter,
) -> Window:
    """The messages for the next request. With no budget it is the whole history. An
    `over` window (the pinned items alone exceed the budget) is sent as it is and
    records nothing: it is one turn's failure, not a decision about the history."""
    if budget is None:  # nothing is packed, so nothing is counted either
        return Window(render(exchanges(task, events, _no_count)), None, 0, (), ())
    items = exchanges(task, events, count)
    kept = [item for item in items if item.id not in carried.dropped]
    shown = apply_cleared(kept, carried.cleared, count)
    before = sum(item.tokens for item in shown)
    pinned = [item for item in shown if item.pinned]
    candidates = [item for item in shown if not item.pinned]
    packed = pack(candidates, budget.tokens, pinned)
    packed = compact(packed, budget.compact_at, budget.compact_to)
    messages = render(packed.selected)
    if packed.over:
        return Window(messages, packed, before, (), ())
    dropped = tuple(item.id for item in packed.dropped)
    return Window(messages, packed, before, dropped, packed.cleared)
