"""Replay: the run state as a pure fold over the log (spec sections 1 and 2).

`RunState` holds what the log alone must answer today (is a tool call pending, is
an approval open, is the run paused or finished). The loop adds the rest.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from functools import reduce

from pydantic import JsonValue

from loomcore.events import (
    ApprovalDenied,
    ApprovalGranted,
    ApprovalRequested,
    Event,
    JsonObject,
    ModelCall,
    RunFinished,
    RunForked,
    RunPaused,
    RunResumed,
    RunStarted,
    ToolCompleted,
    TurnStarted,
)
from loomcore.months import month_key


class ForkChainError(Exception):
    """A run_forked event points at an event the log lacks, or loops back.

    Raised like `LogCorruptError`: a log whose fork references are broken is
    corrupt, not a state the fold should quietly reinterpret.
    """


@dataclass(frozen=True)
class RunState:
    event_count: int = 0
    pending_tools: frozenset[str] = frozenset()
    open_approvals: frozenset[str] = frozenset()
    paused: bool = False
    finished: str | None = None
    turns: int = 0  # this run's turn_started events; a fork starts again at 0
    # Own model_call cost by UTC month ("2026-11"): a fork starts empty, because it
    # inherits its parent's events for context but not for spend.
    spend_by_month: tuple[tuple[str, Decimal], ...] = ()
    truncations: int = 0  # consecutive model_calls that stopped on max_tokens
    paused_limit: str | None = None  # the limit of the pause not yet resumed
    # Active time is the closed segments plus the open one (spec section 1, timeout).
    active: timedelta = timedelta(0)
    segment_start: datetime | None = None
    last_event_at: datetime | None = None


def tool_use_ids(message: JsonObject) -> list[str]:
    """The tool_use blocks of a model response that the driver must complete.

    When the response stopped on `max_tokens`, a final tool_use block is cut off
    mid-input: it is discarded and never run, so it is not counted.
    """
    content = message.get("content")
    blocks = (
        [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
    )
    if (
        message.get("stop_reason") == "max_tokens"
        and blocks
        and _is_tool_use(blocks[-1])
    ):
        blocks = blocks[:-1]
    return [
        str(block["id"]) for block in blocks if _is_tool_use(block) and "id" in block
    ]


def _is_tool_use(block: dict[str, JsonValue]) -> bool:
    return block.get("type") == "tool_use"


def apply(state: RunState, event: Event) -> RunState:
    """The next state. An event that names an unknown id changes nothing.

    Events are validated with aware timestamps, so `month_key` cannot raise here.
    """
    state = replace(state, event_count=state.event_count + 1)
    state = _lifecycle(_tools_and_approvals(state, event), event)
    return _clock(_budget(state, event), event)


def _tools_and_approvals(state: RunState, event: Event) -> RunState:
    match event:
        case ModelCall():
            pending = state.pending_tools | set(tool_use_ids(event.message))
            state = replace(state, pending_tools=pending)
        case ToolCompleted():
            pending = state.pending_tools - {event.tool_use_id}
            state = replace(state, pending_tools=pending)
        case ApprovalRequested():
            open_ = state.open_approvals | {event.tool_use_id}
            state = replace(state, open_approvals=open_)
        case ApprovalGranted() | ApprovalDenied():
            open_ = state.open_approvals - {event.tool_use_id}
            state = replace(state, open_approvals=open_)
    return state


def _lifecycle(state: RunState, event: Event) -> RunState:
    match event:
        case TurnStarted():
            state = replace(state, turns=state.turns + 1)
        case RunForked():
            state = replace(state, turns=0)
        case RunPaused():
            state = replace(state, paused=True)
        case RunResumed():
            state = replace(state, paused=False)
        case RunFinished():
            state = replace(state, finished=event.reason)
    return state


def _budget(state: RunState, event: Event) -> RunState:
    match event:
        case ModelCall():
            key = month_key(event.timestamp)
            spend = dict(state.spend_by_month)
            spend[key] = spend.get(key, Decimal(0)) + event.cost_usd
            cut_off = event.message.get("stop_reason") == "max_tokens"
            truncations = state.truncations + 1 if cut_off else 0
            state = replace(
                state, spend_by_month=tuple(spend.items()), truncations=truncations
            )
        case RunForked():
            state = replace(state, spend_by_month=(), truncations=0)
        case RunPaused():
            state = replace(state, paused_limit=event.limit)
        case RunResumed():
            # Fresh attempts only when the pause was for truncation: a crash or an
            # approval resume keeps the streak.
            fresh = state.paused_limit == "max_output_tokens"
            state = replace(
                state,
                paused_limit=None,
                truncations=0 if fresh else state.truncations,
            )
    return state


def _clock(state: RunState, event: Event) -> RunState:
    match event:
        case RunStarted() | RunForked():
            state = replace(state, active=timedelta(0), segment_start=event.timestamp)
        case RunResumed():
            # The earlier segment ended at the last event before this gap.
            closed = state.active + open_segment(state, state.last_event_at)
            state = replace(state, active=closed, segment_start=event.timestamp)
    return replace(state, last_event_at=event.timestamp)


def open_segment(state: RunState, end: datetime | None) -> timedelta:
    """The open segment up to `end`; nothing if no segment has started."""
    if state.segment_start is None or end is None:
        return timedelta(0)
    return end - state.segment_start


def spent(state: RunState) -> Decimal:
    """This run's own spend over every month."""
    return sum((usd for _, usd in state.spend_by_month), Decimal(0))


def spent_in(state: RunState, month: str) -> Decimal:
    return dict(state.spend_by_month).get(month, Decimal(0))


def replay(events: Iterable[Event], initial: RunState | None = None) -> RunState:
    """Fold events into a state; the same events always give the same state."""
    return reduce(apply, events, initial or RunState())


def run_events(
    events: Iterable[Event], run_id: str, upto: int | None = None
) -> list[Event]:
    """The events replay reads for one run, in order.

    A forked run reads its parent's events up to and including `from.seq`, then its
    own; nothing is copied into the fork. `upto` stops at that seq of this run.
    """
    everything = list(events)
    return _chain(everything, run_id, upto, frozenset())


def _chain(
    events: list[Event], run_id: str, upto: int | None, seen: frozenset[str]
) -> list[Event]:
    if run_id in seen:
        raise ForkChainError(f"run {run_id} is forked from itself")
    own = [e for e in events if e.run_id == run_id and (upto is None or e.seq <= upto)]
    if own and isinstance(own[0], RunForked):
        origin = own[0].from_
        parent = _chain(events, origin.run_id, origin.seq, seen | {run_id})
        if not parent or (parent[-1].run_id, parent[-1].seq) != (
            origin.run_id,
            origin.seq,
        ):
            raise ForkChainError(
                f"run {run_id} is forked from {origin.run_id} at {origin.seq}, "
                "which the log does not hold"
            )
        return [*parent, *own]
    return own
