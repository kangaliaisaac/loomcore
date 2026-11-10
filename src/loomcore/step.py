"""The pure core of the loop: the next state and what to do about it (spec section 1).

`step` does no I/O. The driver appends an event, calls `step`, performs the effects
it returns and appends each result as the next event. The escalation ladder's
judgement needs the limits, so the driver makes it and logs an `escalation` event;
`step` only turns that rung into effects.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from loomcore import ladder
from loomcore.events import (
    ApprovalDenied,
    ApprovalGranted,
    Event,
    ModelCall,
    RunForked,
    RunResumed,
    RunStarted,
    ToolCompleted,
    ToolRequested,
)
from loomcore.replay import RunState, apply, finish_call, tool_use_ids

ReadOnly = Callable[[str], bool]  # by tool name; an unknown tool is not read-only


@dataclass(frozen=True)
class CallModel:
    """Ask the model for the next assistant turn."""


@dataclass(frozen=True)
class RequestCalls:
    """Write the turn's `tool_requested` lines together, then judge them once."""

    tool_use_ids: tuple[str, ...]


@dataclass(frozen=True)
class RunTool:
    tool_use_id: str
    note: str | None = None  # a ladder message added to the result's content


@dataclass(frozen=True)
class Fail:
    """Complete a call with ok=false without running it (a denied approval)."""

    tool_use_id: str
    reason: str


@dataclass(frozen=True)
class AskHuman:
    """Pause for a decision about `tool_use_id`; ask only if no approval is open."""

    tool_use_id: str
    question: str


@dataclass(frozen=True)
class Stop:
    reason: Literal["finished", "end_turn", "refused", "stuck", "error"]
    result: JsonValue = None
    finish_id: str | None = None  # the finish call, completed ok=true


Effect = CallModel | RequestCalls | RunTool | Fail | AskHuman | Stop


# What a model response with no tool call to run asks for, by stop_reason.
_AFTER_STOP: dict[str, Effect] = {
    "end_turn": Stop("end_turn"),
    "stop_sequence": Stop("end_turn"),
    "refusal": Stop("refused"),
    "pause_turn": CallModel(),  # send the response back
    "max_tokens": CallModel(),  # continue once; check_limits pauses the second
}


def _never(_name: str) -> bool:
    return False


def step(
    state: RunState, event: Event, read_only: ReadOnly = _never
) -> tuple[RunState, list[Effect]]:
    """Fold `event` into `state` and say what the driver does next."""
    new = apply(state, event)
    if new.finished is not None or new.paused:
        return new, []
    return new, _effects(state, new, event, read_only)


def _effects(
    old: RunState, state: RunState, event: Event, read_only: ReadOnly
) -> list[Effect]:
    match event:
        case RunStarted() | RunForked() | RunResumed(reason="limit"):
            return [CallModel()]
        case RunResumed(reason="crash"):
            return _after_crash(state, read_only)
        case ModelCall():
            return _after_model_call(event)
        case _:
            return _after_tool_event(old, state, event, read_only)


def _after_tool_event(
    old: RunState, state: RunState, event: Event, read_only: ReadOnly
) -> list[Effect]:
    effects: list[Effect] = []
    match event:
        case ToolRequested() if _all_requested(state):
            # The escalation, if any, was written first, so the turn is judged.
            effects = _proceed(state, read_only)
        case ApprovalGranted() if event.tool_use_id in old.open_approvals:
            effects = [RunTool(i) for i in state.pending_tools]
        case ApprovalDenied() if event.tool_use_id in old.open_approvals:
            reason = {i: r for i, _, r in state.decided}[event.tool_use_id]
            effects = _deny(state, event.tool_use_id, reason, read_only)
        case ToolCompleted() if event.tool_use_id in old.pending_tools and (
            not state.pending_tools
        ):
            effects = [CallModel()]  # the turn's last pending call just finished
    return effects


def _all_requested(state: RunState) -> bool:
    requested = dict(state.requested)
    return bool(state.pending_tools) and all(
        i in requested for i in state.pending_tools
    )


def _settle(state: RunState, read_only: ReadOnly) -> list[Effect]:
    """The turn's calls are requested and judged: run them, asking first about any
    that a crash left possibly run."""
    risky = _interrupted_risky(state, read_only)
    if risky:
        return [AskHuman(risky[0], "run again?" + _risky_note(state, read_only))]
    return [RunTool(i) for i in state.pending_tools]


def _ladder(state: RunState) -> list[Effect]:
    """What the turn's rung asks for (spec section 1, escalation ladder)."""
    if state.escalated is None:
        return []
    rung, tripped, why = state.escalated
    if rung == ladder.STUCK_RUNG:
        return [Stop("stuck")]
    if rung == ladder.ASK_RUNG:
        return [AskHuman(tripped[0], why)]
    note = ladder.NUDGE if rung == ladder.NUDGE_RUNG else ladder.SWITCH
    return [RunTool(i, note if i in tripped else None) for i in state.pending_tools]


def _deny(
    state: RunState, tool_use_id: str, reason: str, read_only: ReadOnly
) -> list[Effect]:
    """A denial covers the asked call and every other call that may have run."""
    denied = {tool_use_id, *_interrupted_risky(state, read_only)}
    failed = [Fail(i, reason) for i in state.pending_tools if i in denied]
    others = [RunTool(i) for i in state.pending_tools if i not in denied]
    return [*failed, *others]


def _interrupted_risky(state: RunState, read_only: ReadOnly) -> list[str]:
    names = dict(state.requested)
    return [i for i in state.interrupted if not read_only(names[i])]


def _after_crash(state: RunState, read_only: ReadOnly) -> list[Effect]:
    """Settle what the log leaves open, as a new process finds it."""
    if (ending := _unwritten_ending(state)) is not None:
        return [ending]
    if state.open_approvals:
        first = state.open_approvals[0]
        note = _risky_note(state, read_only).removeprefix("; ")
        return [AskHuman(first, note)]
    if state.pending_tools:
        return _resume_turn(state, read_only)
    return _between_turns(state)


def _between_turns(state: RunState) -> list[Effect]:
    """No call is pending: do what the last model_call asked, or call the model."""
    if state.ended_with is None:
        return [CallModel()]
    return [_stop_effect(state.ended_with[0])]


def _unwritten_ending(state: RunState) -> Stop | None:
    """The `run_finished` a crash kept from being written, if the turn decided one."""
    if state.finishing is not None:
        tool_use_id, result = state.finishing
        return Stop("finished", result, tool_use_id)
    if state.escalated and state.escalated[0] == ladder.STUCK_RUNG:
        return Stop("stuck")
    return None


def _resume_turn(state: RunState, read_only: ReadOnly) -> list[Effect]:
    """Finish a turn whose calls a crash left pending, in the order the turn runs."""
    requested = dict(state.requested)
    unrequested = tuple(i for i in state.pending_tools if i not in requested)
    if unrequested:
        return [RequestCalls(unrequested)]
    if state.decided:  # the answer was logged; the crash came before the turn ended
        return _after_decision(state, state.decided[0], read_only)
    return _proceed(state, read_only)


def _proceed(state: RunState, read_only: ReadOnly) -> list[Effect]:
    """The turn's calls are requested and judged: apply its rung, or run them."""
    rung = state.escalated[0] if state.escalated else 0
    # A rung that holds the calls back (ask, stop) comes before any question about
    # calls that may have run; a nudge or a switch yields to that question.
    if rung >= ladder.ASK_RUNG or (rung and not _interrupted_risky(state, read_only)):
        return _ladder(state)
    return _settle(state, read_only)


def _after_decision(
    state: RunState, decision: tuple[str, bool, str], read_only: ReadOnly
) -> list[Effect]:
    tool_use_id, granted, reason = decision
    if granted:
        return [RunTool(i) for i in state.pending_tools]
    return _deny(state, tool_use_id, reason, read_only)


def _risky_note(state: RunState, read_only: ReadOnly) -> str:
    names = dict(state.requested)
    risky = _interrupted_risky(state, read_only)
    listed = ", ".join(f"{names[i]} ({i})" for i in risky)
    return f"; may have run before the crash: {listed}" if risky else ""


def _after_model_call(event: ModelCall) -> list[Effect]:
    ids = tool_use_ids(event.message)
    if (finishing := finish_call(event.message)) is not None:
        return [Stop("finished", finishing[1], finishing[0])]
    if ids:
        return [RequestCalls(tuple(ids))]
    return [_stop_effect(event.message.get("stop_reason"))]


def _stop_effect(reason: JsonValue) -> Effect:
    if reason == "tool_use":  # it says tools were called, and none can be run
        return Stop("error", "stop_reason tool_use with no tool call to run")
    known = _AFTER_STOP.get(reason) if isinstance(reason, str) else None
    return known or Stop("error", f"unknown stop_reason: {reason!r}")
