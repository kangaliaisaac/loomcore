"""f03: replay is a deterministic fold; a fork reads its parent's events."""

import pytest
from factories import calling, make

from loomcore.events import Event
from loomcore.replay import (
    ForkChainError,
    RunState,
    apply,
    replay,
    run_events,
    tool_use_ids,
)


def test_an_empty_log_replays_to_the_initial_state() -> None:
    assert replay([]) == RunState()


def test_the_same_log_always_folds_to_the_same_state() -> None:
    events = [
        make("run_started"),
        make("model_call", 1, message=calling("a", "b")),
        make("tool_completed", 2, tool_use_id="a"),
    ]
    assert replay(events) == replay(list(events)) == replay(events)
    assert replay(events).event_count == 3


def test_tool_calls_are_pending_until_completed() -> None:
    call = make("model_call", 1, message=calling("a", "b"))
    assert replay([call]).pending_tools == {"a", "b"}
    done = make("tool_completed", 2, tool_use_id="a")
    assert replay([call, done]).pending_tools == {"b"}


def test_a_completion_for_an_unknown_call_changes_nothing_but_the_count() -> None:
    state = apply(RunState(), make("tool_completed", tool_use_id="ghost"))
    assert state == RunState(event_count=1)


def test_a_tool_use_cut_off_by_max_tokens_is_never_pending() -> None:
    message = calling("done", "cut", stop_reason="max_tokens")
    assert tool_use_ids(message) == ["done"]
    assert tool_use_ids(calling("done", "cut")) == ["done", "cut"]


def test_tool_use_ids_ignores_text_blocks_and_odd_content() -> None:
    assert tool_use_ids({"content": [{"type": "text", "text": "hi"}]}) == []
    assert tool_use_ids({"content": "plain"}) == []
    assert tool_use_ids({}) == []


def test_approvals_are_open_until_granted_or_denied() -> None:
    asked = make("approval_requested")
    assert replay([asked]).open_approvals == {"t1"}
    assert replay([asked, make("approval_granted", 1)]).open_approvals == frozenset()
    assert replay([asked, make("approval_denied", 1)]).open_approvals == frozenset()


def test_a_pause_lasts_until_the_run_resumes() -> None:
    paused = make("run_paused", reason="limit", limit="max_turns")
    resumed = make("run_resumed", 1, reason="limit", limits={})
    assert replay([paused]).paused
    assert not replay([paused, resumed]).paused


def test_a_finished_run_records_its_reason() -> None:
    assert replay([make("run_finished", reason="stuck")]).finished == "stuck"


def test_a_run_reads_only_its_own_events() -> None:
    events = [
        make("run_started", run="a"),
        make("run_started", run="b"),
        make("turn_started", 1, run="a"),
    ]
    assert [e.seq for e in run_events(events, "a")] == [0, 1]
    assert [e.run_id for e in run_events(events, "b")] == ["b"]


def forked_log() -> list[Event]:
    return [
        make("run_started", 0, run="a"),
        make("turn_started", 1, run="a"),
        make("turn_started", 2, run="a"),
        make("run_forked", 0, run="b", **{"from": {"runId": "a", "seq": 1}}),
        make("turn_started", 1, run="b"),
        make("turn_started", 3, run="a"),
    ]


def test_a_fork_reads_the_parent_up_to_and_including_from_seq_then_its_own() -> None:
    chain = run_events(forked_log(), "b")
    assert [(e.run_id, e.seq) for e in chain] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]


def test_the_parent_is_untouched_by_the_fork() -> None:
    assert [e.seq for e in run_events(forked_log(), "a")] == [0, 1, 2, 3]


def test_a_fork_of_a_fork_follows_the_chain() -> None:
    log = [
        *forked_log(),
        make("run_forked", 0, run="c", **{"from": {"runId": "b", "seq": 0}}),
    ]
    chain = run_events(log, "c")
    assert [(e.run_id, e.seq) for e in chain] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("c", 0),
    ]


def test_a_fork_that_leads_back_to_itself_is_an_error() -> None:
    log = [
        make("run_forked", 0, run="a", **{"from": {"runId": "b", "seq": 0}}),
        make("run_forked", 0, run="b", **{"from": {"runId": "a", "seq": 0}}),
    ]
    with pytest.raises(ForkChainError):
        run_events(log, "a")
