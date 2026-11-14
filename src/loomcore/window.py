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
from loomcore.conversation import exchanges, recount, render
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
    tokens_before: int  # all it considered, before this turn's dropping and clearing
    newly_dropped: tuple[str, ...]  # to record in `context_packed.dropped`
    newly_cleared: tuple[str, ...]  # to record in `context_compacted.cleared`


def build_window(
    task: JsonValue,
    events: Sequence[Event],
    carried: Carried,
    budget: Budget | None,
    count: Counter,
) -> Window:
    """The messages for the next request. With no budget nothing is packed or counted,
    but what earlier windows dropped and cleared stays dropped and cleared: turning
    the budget off does not bring the history back. An `over` window (the pinned
    items alone exceed the budget) is sent unpacked, as earlier windows left it, and records nothing: the
    log still describes what was sent, and the provider's error is the backstop."""
    # Items are built without counting so the dropped ones are never counted.
    items = exchanges(task, events, _no_count)
    kept = [item for item in items if item.id not in carried.dropped]
    if budget is None:
        shown_free = apply_cleared(kept, carried.cleared, _no_count)
        return Window(render(shown_free), None, 0, (), ())
    shown = recount(apply_cleared(kept, carried.cleared, count), count)
    before = sum(item.tokens for item in shown)
    pinned = [item for item in shown if item.pinned]
    candidates = [item for item in shown if not item.pinned]
    packed = pack(candidates, budget.tokens, pinned)
    if packed.over:
        return Window(render(shown), packed, before, (), ())
    packed = compact(packed, budget.compact_at, budget.compact_to)
    dropped = tuple(item.id for item in packed.dropped)
    return Window(render(packed.selected), packed, before, dropped, packed.cleared)


def _no_count(messages: Sequence[JsonObject]) -> int:
    return 0
