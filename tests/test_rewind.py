"""f06: a fork may start only at a clean point."""

import pytest
from factories import calling, make

from loomcore.events import Event
from loomcore.replay import run_events
from loomcore.rewind import (
    ForkPoint,
    ForkRefusedError,
    ensure_fork_point,
    fork_point_problem,
)


def clean_run() -> list[Event]:
    return [
        make("run_started", 0),
        make("turn_started", 1),
        make("model_call", 2, message=calling("a", "b")),
        make("tool_requested", 3, tool_use_id="a"),
        make("tool_completed", 4, tool_use_id="a"),
        make("tool_requested", 5, tool_use_id="b"),
        make("tool_completed", 6, tool_use_id="b"),
        make("turn_started", 7),
    ]


def problem(events: list[Event], seq: int) -> str | None:
    return fork_point_problem(events, ForkPoint("r1", seq))


def test_points_between_turns_are_clean() -> None:
    log = clean_run()
    assert [problem(log, seq) for seq in (0, 1, 6, 7)] == [None] * 4


@pytest.mark.parametrize("seq", [2, 3, 4, 5])
def test_a_point_with_a_tool_call_pending_is_refused(seq: int) -> None:
    # 2: model_call with no tool_requested yet; 3, 5: requested; 4: one of two done.
    assert "pending" in (problem(clean_run(), seq) or "")


def test_a_cut_off_tool_use_does_not_make_a_point_unclean() -> None:
    log = [
        make("model_call", 0, message=calling("only", "cut", stop_reason="max_tokens"))
    ]
    assert "only" in (problem(log, 0) or "")
    log.append(make("tool_completed", 1, tool_use_id="only"))
    assert problem(log, 1) is None


def test_a_point_with_an_open_approval_is_refused() -> None:
    log = [make("turn_started", 0), make("approval_requested", 1)]
    assert "approvals" in (problem(log, 1) or "")
    log.append(make("approval_granted", 2))
    assert problem(log, 2) is None


def test_a_point_in_an_open_pause_is_refused_until_the_resume() -> None:
    log = [
        make("turn_started", 0),
        make("run_paused", 1, reason="limit", limit="max_turns"),
        make("run_resumed", 2, reason="limit", limits={}),
    ]
    assert "paused" in (problem(log, 1) or "")
    assert problem(log, 2) is None


def test_the_finishing_event_and_later_points_are_refused() -> None:
    log = [make("turn_started", 0), make("run_finished", 1)]
    assert problem(log, 0) is None
    assert "finished" in (problem(log, 1) or "")


def test_a_seq_the_run_does_not_have_is_refused() -> None:
    assert "no event" in (problem(clean_run(), 99) or "")


def test_ensure_raises_with_the_reason() -> None:
    with pytest.raises(ForkRefusedError, match="pending"):
        ensure_fork_point(clean_run(), ForkPoint("r1", 3))
    ensure_fork_point(clean_run(), ForkPoint("r1", 7))


def test_a_fork_sees_the_parent_up_to_from_seq_and_a_second_fork_is_judged_too() -> (
    None
):
    log = clean_run()
    fork = make("run_forked", 0, run="r2", **{"from": {"runId": "r1", "seq": 1}})
    log += [fork, make("model_call", 1, run="r2", message=calling("x"))]
    assert [(e.run_id, e.seq) for e in run_events(log, "r2")] == [
        ("r1", 0),
        ("r1", 1),
        ("r2", 0),
        ("r2", 1),
    ]
    assert fork_point_problem(log, ForkPoint("r2", 0)) is None
    assert "pending" in (fork_point_problem(log, ForkPoint("r2", 1)) or "")
