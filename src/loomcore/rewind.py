"""Rewind: which points of a run a fork may start from (spec section 1)."""

from collections.abc import Iterable

from loomcore.events import Event, ForkOrigin
from loomcore.replay import replay, run_events


class ForkRefusedError(Exception):
    """The chosen point is not a clean fork point; the message says why."""


def fork_point_problem(events: Iterable[Event], point: ForkOrigin) -> str | None:
    """Why `point` is not a clean fork point, or None if it is.

    Judged on the state folded up to and including the event at `point.seq`.
    `events` are one session file's: a fork names a run, not a session (spec
    section 1), and run ids are unique within the file.
    """
    chain = run_events(events, point.run_id, upto=point.seq)
    if not any(e.run_id == point.run_id and e.seq == point.seq for e in chain):
        return f"run {point.run_id} has no event at seq {point.seq}"
    state = replay(chain)
    problem = None
    if state.finished is not None:
        problem = f"the run had already finished ({state.finished})"
    elif state.pending_tools:
        problem = f"tool calls pending: {list(state.pending_tools)}"
    elif state.open_approvals:
        problem = f"approvals open: {sorted(state.open_approvals)}"
    elif state.paused:
        problem = "the run is paused and has not been resumed"
    return problem


def ensure_fork_point(events: Iterable[Event], point: ForkOrigin) -> None:
    """Raise `ForkRefusedError` unless `point` is clean.

    A log whose fork references are broken raises `ForkChainError` instead: that
    is corruption, not a refused point.
    """
    if problem := fork_point_problem(events, point):
        raise ForkRefusedError(f"cannot fork {point.run_id} at {point.seq}: {problem}")
