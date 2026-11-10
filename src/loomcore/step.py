"""The pure core of the loop: the next state and what to do about it (spec section 1).

`step` does no I/O. The driver appends an event, calls `step`, performs the effects
it returns and appends each result as the next event. Limits, retries and the
escalation ladder wrap this core; they are separate features.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from loomcore.events import (
    Event,
    ModelCall,
    RunForked,
    RunResumed,
    RunStarted,
    ToolCompleted,
)
from loomcore.replay import RunState, apply, tool_use_ids


@dataclass(frozen=True)
class CallModel:
    """Ask the model for the next assistant turn."""


@dataclass(frozen=True)
class RunTool:
    tool_use_id: str


@dataclass(frozen=True)
class Stop:
    reason: Literal["end_turn", "refused", "error"]
    result: JsonValue = None


Effect = CallModel | RunTool | Stop


# What a model response with no tool call to run asks for, by stop_reason.
_AFTER_STOP: dict[str, Effect] = {
    "end_turn": Stop("end_turn"),
    "stop_sequence": Stop("end_turn"),
    "refusal": Stop("refused"),
    "pause_turn": CallModel(),  # send the response back
    "max_tokens": CallModel(),  # continue once; f09 pauses on the second cut-off
    "tool_use": CallModel(),
}


def step(state: RunState, event: Event) -> tuple[RunState, list[Effect]]:
    """Fold `event` into `state` and say what the driver does next."""
    new = apply(state, event)
    if new.finished is not None or new.paused:
        return new, []
    return new, _effects(new, event)


def _effects(state: RunState, event: Event) -> list[Effect]:
    effects: list[Effect] = []
    match event:
        case RunStarted() | RunForked() | RunResumed(reason="limit"):
            effects = [CallModel()]
        case ModelCall():
            effects = _after_model_call(event)
        case ToolCompleted() if not state.pending_tools:
            effects = [CallModel()]
    return effects


def _after_model_call(event: ModelCall) -> list[Effect]:
    ids = tool_use_ids(event.message)
    if ids:
        return [RunTool(tool_use_id) for tool_use_id in ids]
    reason = event.message.get("stop_reason")
    known = _AFTER_STOP.get(reason) if isinstance(reason, str) else None
    return [known or Stop("error", f"unknown stop_reason: {reason!r}")]
