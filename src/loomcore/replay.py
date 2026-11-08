"""Replay: the run state as a pure fold over the log (spec sections 1 and 2).

`RunState` holds what the log alone must answer today (is a tool call pending, is
an approval open, is the run paused or finished). The loop adds the rest.
"""

from collections.abc import Iterable
from functools import reduce

from pydantic import BaseModel, ConfigDict, JsonValue

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
    ToolCompleted,
)


class ForkChainError(Exception):
    """A run_forked event points at an event the log lacks, or loops back.

    Raised like `LogCorruptError`: a log whose fork references are broken is
    corrupt, not a state the fold should quietly reinterpret.
    """


class RunState(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_count: int = 0
    pending_tools: frozenset[str] = frozenset()
    open_approvals: frozenset[str] = frozenset()
    paused: bool = False
    finished: str | None = None


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
    """The next state. Total: an event that names an unknown id changes nothing."""
    update: dict[str, object] = {"event_count": state.event_count + 1}
    match event:
        case ModelCall():
            update["pending_tools"] = state.pending_tools | set(
                tool_use_ids(event.message)
            )
        case ToolCompleted():
            update["pending_tools"] = state.pending_tools - {event.tool_use_id}
        case ApprovalRequested():
            update["open_approvals"] = state.open_approvals | {event.tool_use_id}
        case ApprovalGranted() | ApprovalDenied():
            update["open_approvals"] = state.open_approvals - {event.tool_use_id}
        case RunPaused():
            update["paused"] = True
        case RunResumed():
            update["paused"] = False
        case RunFinished():
            update["finished"] = event.reason
    return state.model_copy(update=update)


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
