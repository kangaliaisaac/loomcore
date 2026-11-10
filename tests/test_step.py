"""f07: `step` is a pure fold over the log that says what to do next."""

import builtins
import socket
import time
from functools import reduce

import pytest
from factories import calling, make
from hypothesis import given, settings
from strategies import logs

from loomcore.events import Event
from loomcore.replay import RunState, apply, replay, run_events
from loomcore.step import CallModel, Effect, RunTool, Stop, step


def run(*events: Event) -> tuple[RunState, list[Effect]]:
    """Step through `events`; the effects of the last one."""
    state = RunState()
    effects: list[Effect] = []
    for event in events:
        state, effects = step(state, event)
    return state, effects


def model_call(seq: int, stop: str, *tool_ids: str) -> Event:
    return make("model_call", seq, message=calling(*tool_ids, stop_reason=stop))


def test_a_new_run_starts_by_calling_the_model() -> None:
    assert run(make("run_started"))[1] == [CallModel()]


def test_a_fork_starts_by_calling_the_model() -> None:
    origin: dict[str, object] = {"from": {"runId": "r0", "seq": 0}}
    assert run(make("run_forked", 0, "r1", **origin))[1] == [CallModel()]


def test_each_tool_use_becomes_a_run_tool_in_order() -> None:
    effects = run(make("run_started"), model_call(1, "tool_use", "b", "a"))[1]
    assert effects == [RunTool("b"), RunTool("a")]


def test_the_model_is_called_again_only_when_every_tool_has_completed() -> None:
    started = [make("run_started"), model_call(1, "tool_use", "a", "b")]
    first = [*started, make("tool_completed", 2, tool_use_id="a")]
    assert run(*first)[1] == []
    both = [*first, make("tool_completed", 3, tool_use_id="b")]
    assert run(*both)[1] == [CallModel()]


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        ("end_turn", Stop("end_turn")),
        ("stop_sequence", Stop("end_turn")),
        ("refusal", Stop("refused")),
        ("pause_turn", CallModel()),
        ("max_tokens", CallModel()),
    ],
)
def test_a_response_with_no_tool_calls_is_handled_by_its_stop_reason(
    stop: str, expected: Effect
) -> None:
    assert run(make("run_started"), model_call(1, stop))[1] == [expected]


def test_an_unknown_or_missing_stop_reason_stops_with_an_error() -> None:
    odd = run(make("run_started"), model_call(1, "novel"))[1]
    assert odd == [Stop("error", "unknown stop_reason: 'novel'")]
    missing = make("model_call", 1, message={"content": []})
    assert run(make("run_started"), missing)[1] == [
        Stop("error", "unknown stop_reason: None")
    ]


def test_a_cut_off_tool_use_is_not_run_and_the_complete_ones_are() -> None:
    message = calling("a", "b", stop_reason="max_tokens")
    effects = run(make("run_started"), make("model_call", 1, message=message))[1]
    assert effects == [RunTool("a")]


def test_a_response_cut_off_inside_its_only_tool_use_continues() -> None:
    message = calling("a", stop_reason="max_tokens")
    effects = run(make("run_started"), make("model_call", 1, message=message))[1]
    assert effects == [CallModel()]


def test_a_completion_that_finishes_nothing_does_not_call_the_model() -> None:
    log = [
        make("run_started"),
        model_call(1, "tool_use", "a"),
        make("tool_completed", 2, tool_use_id="a"),
    ]
    assert run(*log)[1] == [CallModel()]
    assert run(*log, make("tool_completed", 3, tool_use_id="a"))[1] == []
    assert run(*log, make("tool_completed", 3, tool_use_id="ghost"))[1] == []


def test_a_fork_counts_its_own_turns_not_its_parents() -> None:
    parent = [make("run_started"), make("turn_started", 1), make("turn_started", 2)]
    origin: dict[str, object] = {"from": {"runId": "r1", "seq": 2}}
    fork = [make("run_forked", 0, "r2", **origin), make("turn_started", 1, "r2")]
    assert replay(run_events([*parent, *fork], "r2")).turns == 1


def test_a_paused_or_finished_run_has_nothing_to_do() -> None:
    paused = [make("run_started"), make("run_paused", 1)]
    assert run(*paused)[1] == []
    finished = [make("run_started"), make("run_finished", 1)]
    assert run(*finished)[1] == []


def test_resuming_after_a_limit_pause_calls_the_model() -> None:
    log = [
        make("run_started"),
        make("run_paused", 1),
        make("run_resumed", 2, reason="limit", limits={}),
    ]
    assert run(*log)[1] == [CallModel()]


@pytest.mark.parametrize(
    "kind", ["turn_started", "tool_requested", "guard_decision", "note_written"]
)
def test_bookkeeping_events_ask_for_nothing(kind: str) -> None:
    assert run(make("run_started"), make(kind, 1))[1] == []


def test_turn_started_events_are_counted() -> None:
    state = run(make("run_started"), make("turn_started", 1), make("turn_started", 2))[
        0
    ]
    assert state.turns == 2


@given(log=logs())
@settings(max_examples=60, deadline=None)
def test_stepping_the_log_gives_the_state_the_fold_gives(log: list[Event]) -> None:
    state = RunState()
    for event in log:
        state, _ = step(state, event)
    assert state == replay(log) == reduce(apply, log, RunState())


@given(log=logs())
@settings(max_examples=30, deadline=None)
def test_step_does_no_io(log: list[Event]) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("step touched the world")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in [
            (builtins, "open"),
            (socket, "socket"),
            (time, "time"),
            (time, "sleep"),
        ]:
            patch.setattr(module, name, refuse)
        state = RunState()
        for event in log:
            state, _ = step(state, event)
