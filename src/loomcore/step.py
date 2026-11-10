"""The pure core of the loop: the next state and what to do about it (spec section 1).

`step` does no I/O. The driver appends an event, calls `step`, performs the effects
it returns and appends each result as the next event. Limits, retries and the
escalation ladder wrap this core; the ladder (f11) adds its own `AskHuman` cases.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from loomcore.events import (
    ApprovalDenied,
    ApprovalGranted,
    Event,
    ModelCall,
    RunForked,
    RunResumed,
    RunStarted,
    ToolCompleted,
)
from loomcore.replay import RunState, apply, tool_use_ids

ReadOnly = Callable[[str], bool]  # by tool name; an unknown tool is not read-only


@dataclass(frozen=True)
class CallModel:
    """Ask the model for the next assistant turn."""


@dataclass(frozen=True)
class RunTool:
    tool_use_id: str


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
    reason: Literal["end_turn", "refused", "error"]
    result: JsonValue = None


Effect = CallModel | RunTool | Fail | AskHuman | Stop


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
    effects: list[Effect] = []
    match event:
        case RunStarted() | RunForked() | RunResumed(reason="limit"):
            effects = [CallModel()]
        case RunResumed(reason="crash"):
            effects = _after_crash(state, read_only)
        case ApprovalGranted() if event.tool_use_id in old.open_approvals:
            effects = [RunTool(i) for i in state.pending_tools]
        case ApprovalDenied() if event.tool_use_id in old.open_approvals:
            effects = _after_denial(state, event, read_only)
        case ModelCall():
            effects = _after_model_call(event)
        case ToolCompleted() if event.tool_use_id in old.pending_tools and (
            not state.pending_tools
        ):
            effects = [CallModel()]  # the turn's last pending call just finished
    return effects


def _after_denial(
    state: RunState, event: ApprovalDenied, read_only: ReadOnly
) -> list[Effect]:
    """A denial covers the asked call and every other call that may have run."""
    reason = f"denied by {event.by}: {event.note}"
    denied = {event.tool_use_id, *_interrupted_risky(state, read_only)}
    failed = [Fail(i, reason) for i in state.pending_tools if i in denied]
    others = [RunTool(i) for i in state.pending_tools if i not in denied]
    return [*failed, *others]


def _interrupted_risky(state: RunState, read_only: ReadOnly) -> list[str]:
    names = dict(state.requested)
    return [i for i in state.interrupted if not read_only(names[i])]


def _after_crash(state: RunState, read_only: ReadOnly) -> list[Effect]:
    """Settle what the log leaves open, as a new process finds it."""
    if state.open_approvals:
        first = state.open_approvals[0]
        return [AskHuman(first, "approval still open" + _risky_note(state, read_only))]
    if state.pending_tools:
        return _rerun(state, read_only)
    if state.ended_with is not None:
        return [_stop_effect(state.ended_with[0])]
    return [CallModel()]


def _rerun(state: RunState, read_only: ReadOnly) -> list[Effect]:
    """A call requested but never completed may have run: a read-only tool is run
    again, any other becomes an `AskHuman` first. The question is about the first
    such call and names every such call of the turn: a grant runs them all, a
    denial fails them all."""
    risky = _interrupted_risky(state, read_only)
    if risky:
        return [AskHuman(risky[0], "run again?" + _risky_note(state, read_only))]
    return [RunTool(i) for i in state.pending_tools]


def _risky_note(state: RunState, read_only: ReadOnly) -> str:
    names = dict(state.requested)
    risky = _interrupted_risky(state, read_only)
    listed = ", ".join(f"{names[i]} ({i})" for i in risky)
    return f"; may have run before the crash: {listed}" if risky else ""


def _after_model_call(event: ModelCall) -> list[Effect]:
    ids = tool_use_ids(event.message)
    if ids:
        return [RunTool(tool_use_id) for tool_use_id in ids]
    return [_stop_effect(event.message.get("stop_reason"))]


def _stop_effect(reason: JsonValue) -> Effect:
    if reason == "tool_use":  # it says tools were called, and none can be run
        return Stop("error", "stop_reason tool_use with no tool call to run")
    known = _AFTER_STOP.get(reason) if isinstance(reason, str) else None
    return known or Stop("error", f"unknown stop_reason: {reason!r}")
