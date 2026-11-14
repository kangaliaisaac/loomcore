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
    Escalation,
    Event,
    JsonObject,
    ModelCall,
    RunFinished,
    RunForked,
    RunPaused,
    RunResumed,
    RunStarted,
    ToolCompleted,
    ToolRequested,
    TurnStarted,
)
from loomcore.finish import NAME as FINISH
from loomcore.finish import result_of
from loomcore.ladder import ASK_RUNG
from loomcore.months import month_key


class ForkChainError(Exception):
    """A run_forked event points at an event the log lacks, or loops back.

    Raised like `LogCorruptError`: a log whose fork references are broken is
    corrupt, not a state the fold should quietly reinterpret.
    """


@dataclass(frozen=True)
class RunState:
    event_count: int = 0
    pending_tools: tuple[str, ...] = ()  # in the order the model asked for them
    # (id, name) of the calls with a tool_requested and no completion.
    requested: tuple[tuple[str, str], ...] = ()
    # (id, wave) of the same calls: where each runs in the turn (see `ToolRequested`).
    call_waves: tuple[tuple[str, int], ...] = ()
    # The escalation ladder (spec section 1): the rung reached, this run's call
    # hashes in order, the (call, hash) of each ladder ask, this turn's rung with the
    # calls that tripped and why, and decisions not yet followed by a completion.
    rung: int = 0
    call_hashes: tuple[str, ...] = ()
    ladder_asks: tuple[tuple[str, str], ...] = ()
    escalated: tuple[int, tuple[str, ...], str] | None = None
    decided: tuple[tuple[str, bool, str], ...] = ()
    # The finish call of the latest model_call, (id, result): a crash after it must
    # end the run, not run the call.
    finishing: tuple[str, JsonValue] | None = None
    # The call a crash resume found that may have run (see `_could_have_started`).
    # A requested call need not have started: the turn's lines are written together
    # and its waves run one after the other, so only a crash makes a call
    # "interrupted".
    interrupted: tuple[str, ...] = ()
    # A model_call that asked for no tool ends with (stop_reason,); until the next
    # model_call a crash resume reads it to know what that call asked for.
    ended_with: tuple[JsonValue, ...] | None = None
    open_approvals: tuple[str, ...] = ()  # in the order they were asked
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


def content_blocks(message: JsonObject) -> list[JsonObject]:
    """The blocks of a model response; none if it has no content list."""
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def tool_use_ids(message: JsonObject) -> list[str]:
    """The tool_use blocks of a model response that the driver must complete.

    When the response stopped on `max_tokens`, a final tool_use block is cut off
    mid-input: it is discarded and never run, so it is not counted.
    """
    blocks = content_blocks(message)
    if (
        message.get("stop_reason") == "max_tokens"
        and blocks
        and _is_tool_use(blocks[-1])
    ):
        blocks = blocks[:-1]
    return [
        str(block["id"]) for block in blocks if _is_tool_use(block) and "id" in block
    ]


def finish_call(message: JsonObject) -> tuple[str, JsonValue] | None:
    """The `finish` call of a model response, as (id, result), if it made one."""
    ids = tool_use_ids(message)
    for block in content_blocks(message):
        if block.get("name") == FINISH and block.get("id") in ids:
            args = block.get("input")
            return str(block["id"]), result_of(args if isinstance(args, dict) else {})
    return None


def _is_tool_use(block: dict[str, JsonValue]) -> bool:
    return block.get("type") == "tool_use"


def apply(state: RunState, event: Event) -> RunState:
    """The next state. An event that names an unknown id changes nothing.

    Events are validated with aware timestamps, so `month_key` cannot raise here.
    """
    state = replace(state, event_count=state.event_count + 1)
    state = _lifecycle(_approvals(_tools(state, event), event), event)
    return _clock(_budget(state, event), event)


def _tools(state: RunState, event: Event) -> RunState:
    match event:
        case ModelCall():
            ids = tool_use_ids(event.message)
            pending = state.pending_tools + tuple(
                i for i in ids if i not in state.pending_tools
            )
            ended = None if ids else (event.message.get("stop_reason"),)
            state = replace(
                state,
                pending_tools=pending,
                ended_with=ended,
                escalated=None,
                decided=(),  # a decision covers its turn, which this call ends
                finishing=finish_call(event.message),
            )
        case ToolRequested():
            if event.tool_use_id in state.pending_tools and (
                event.tool_use_id not in dict(state.requested)
            ):
                requested = (*state.requested, (event.tool_use_id, event.name))
                # No wave in the log: it ran alone, after the calls before it.
                wave = len(state.requested) if event.wave is None else event.wave
                waves = (*state.call_waves, (event.tool_use_id, wave))
                hashes = (*state.call_hashes, event.call_hash)
                state = replace(
                    state, requested=requested, call_waves=waves, call_hashes=hashes
                )
        case ToolCompleted():
            pending = tuple(i for i in state.pending_tools if i != event.tool_use_id)
            requested = tuple(r for r in state.requested if r[0] != event.tool_use_id)
            waves = tuple(w for w in state.call_waves if w[0] != event.tool_use_id)
            interrupted = tuple(i for i in state.interrupted if i != event.tool_use_id)
            state = replace(
                state,
                pending_tools=pending,
                requested=requested,
                call_waves=waves,
                interrupted=interrupted,
            )
        case RunResumed(reason="crash"):
            state = replace(state, interrupted=_could_have_started(state))
    return state


def _could_have_started(state: RunState) -> tuple[str, ...]:
    """The calls a crash may have interrupted: those of the first wave that still
    has a pending call.

    A wave starts when the one before it is complete, and only once every line of
    the turn is written. So none can have started while a line is missing, or while
    the ladder holds the turn for a human with no answer logged; otherwise any call
    of the first wave with a call still pending can have, and no call of a later
    wave. In a log where every call is its own wave that is the first pending call.
    """
    requested = {i for i, _ in state.requested}
    # An ask rung holds the turn; so does stuck, which only completes its calls.
    held = state.escalated is not None and state.escalated[0] >= ASK_RUNG
    denied = {i for i, granted, _ in state.decided if not granted}
    if not state.pending_tools or (held and not state.decided):
        return ()
    if any(i not in requested for i in state.pending_tools):
        return ()
    if denied & set(state.pending_tools):  # their Fails are written before any run
        return ()
    waves = dict(state.call_waves)
    first = min(waves[i] for i in state.pending_tools)
    return tuple(i for i in state.pending_tools if waves[i] == first)


def _approvals(state: RunState, event: Event) -> RunState:
    match event:
        case ApprovalRequested():
            asked = event.tool_use_id
            if asked not in state.open_approvals:
                state = replace(state, open_approvals=(*state.open_approvals, asked))
        case ApprovalGranted() | ApprovalDenied():
            if event.tool_use_id in state.open_approvals:
                state = _decide(state, event)
            open_ = tuple(i for i in state.open_approvals if i != event.tool_use_id)
            state = replace(state, open_approvals=open_)
        case Escalation():
            asks = state.ladder_asks
            if event.rung == ASK_RUNG and event.tool_use_id and event.call_hash:
                asks = (*asks, (event.tool_use_id, event.call_hash))
            seen = (event.rung, tuple(event.tripped), event.reason)
            state = replace(state, rung=event.rung, ladder_asks=asks, escalated=seen)
    return state


def _decide(state: RunState, event: ApprovalGranted | ApprovalDenied) -> RunState:
    """Remember a decision until its calls complete; a grant on a ladder ask also
    forgets the earlier calls with the asked call's hash, so they no longer count."""
    granted = isinstance(event, ApprovalGranted)
    reason = "" if granted else f"denied by {event.by}: {event.note}"
    decided = (*state.decided, (event.tool_use_id, granted, reason))
    hashes = state.call_hashes
    asked = dict(state.ladder_asks).get(event.tool_use_id)
    if granted and asked is not None:
        hashes = tuple("" if h == asked else h for h in hashes)  # keep the window
    return replace(state, decided=decided, call_hashes=hashes)


def _lifecycle(state: RunState, event: Event) -> RunState:
    match event:
        case TurnStarted():
            state = replace(state, turns=state.turns + 1)
        case RunForked():
            # A fork forgets how its parent's last call ended and what a crash found.
            state = replace(
                state,
                turns=0,
                ended_with=None,
                interrupted=(),
                rung=0,
                call_hashes=(),
                ladder_asks=(),
                escalated=None,
                decided=(),
                finishing=None,
            )
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
            state = replace(
                state,
                spend_by_month=(),
                truncations=0,
            )
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
