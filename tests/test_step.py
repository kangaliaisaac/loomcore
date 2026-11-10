"""f07: `step` is a pure fold over the log that says what to do next."""

import ast
import builtins
import io
import socket
import time
from pathlib import Path

import pytest
from factories import calling, make
from hypothesis import given, settings
from strategies import logs

from loomcore import step as step_module
from loomcore.events import Event
from loomcore.replay import RunState, replay, run_events
from loomcore.step import AskHuman, CallModel, Effect, Fail, RunTool, Stop, step


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


def test_a_tool_use_stop_with_no_tool_call_to_run_is_an_error() -> None:
    effects = run(make("run_started"), model_call(1, "tool_use"))[1]
    assert effects == [Stop("error", "stop_reason tool_use with no tool call to run")]


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
def test_the_state_counts_events_and_a_stopped_run_has_nothing_to_do(
    log: list[Event],
) -> None:
    state = RunState()
    for event in log:
        state, effects = step(state, event)
        if state.finished is not None or state.paused:
            assert effects == []
    assert state.event_count == len(log)


def test_step_imports_nothing_that_touches_the_world() -> None:
    tree = ast.parse(Path(step_module.__file__ or "").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= {
        "collections.abc",
        "dataclasses",
        "typing",
        "pydantic",
        "loomcore.events",
        "loomcore.replay",
    }


@given(log=logs())
@settings(max_examples=30, deadline=None)
def test_step_does_no_io(log: list[Event]) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("step touched the world")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in [
            (builtins, "open"),
            (io, "open"),
            (time, "monotonic"),
            (socket, "socket"),
            (time, "time"),
            (time, "sleep"),
        ]:
            patch.setattr(module, name, refuse)
        state = RunState()
        for event in log:
            state, _ = step(state, event)


def asked(seq: int, tool_use_id: str = "a") -> Event:
    return make("approval_requested", seq, tool_use_id=tool_use_id)


def requested(seq: int, tool_use_id: str, name: str) -> Event:
    return make("tool_requested", seq, tool_use_id=tool_use_id, name=name)


def crash(*events: Event, read_only: frozenset[str] = frozenset()) -> list[Effect]:
    state = RunState()
    for event in events:
        state, _ = step(state, event)
    resumed = make("run_resumed", len(events), reason="crash")
    return step(state, resumed, lambda name: name in read_only)[1]


def test_a_crash_before_the_first_model_call_calls_the_model() -> None:
    assert crash(make("run_started")) == [CallModel()]


def test_a_crash_after_the_last_completion_calls_the_model() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a")]
    log.append(make("tool_completed", 2, tool_use_id="a"))
    assert crash(*log) == [CallModel()]


def test_a_crash_after_a_final_answer_stops_with_its_reason() -> None:
    assert crash(make("run_started"), model_call(1, "end_turn")) == [Stop("end_turn")]
    assert crash(make("run_started"), model_call(1, "max_tokens")) == [CallModel()]


def test_a_crash_runs_calls_never_started_and_reruns_read_only_ones() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a", "b")]
    log.append(requested(2, "a", "search"))
    assert crash(*log, read_only=frozenset({"search"})) == [RunTool("a"), RunTool("b")]


def test_a_crash_asks_before_rerunning_a_call_that_may_have_had_effects() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a", "b")]
    log.append(requested(2, "a", "write_file"))
    [effect] = crash(*log, read_only=frozenset({"search"}))
    assert isinstance(effect, AskHuman)
    assert effect.tool_use_id == "a"


def test_a_crash_with_an_open_approval_pauses_for_it() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a"), asked(2)]
    assert [type(e) for e in crash(*log)] == [AskHuman]


def test_a_grant_runs_every_pending_call_and_a_denial_fails_the_asked_one() -> None:
    base = [make("run_started"), model_call(1, "tool_use", "a", "b"), asked(2)]
    base.append(make("run_paused", 3, reason="awaiting_approval", resume_from=2))
    base.append(make("run_resumed", 4, reason="approval", resume_from=2))
    granted = make("approval_granted", 5, tool_use_id="a")
    assert run(*base, granted)[1] == [RunTool("a"), RunTool("b")]
    denied = make("approval_denied", 5, tool_use_id="a", by="me", note="no")
    assert run(*base, denied)[1] == [Fail("a", "denied by me: no"), RunTool("b")]


def test_a_decision_nobody_asked_for_does_nothing() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a")]
    assert run(*log, make("approval_granted", 2, tool_use_id="a"))[1] == []


def paused_for_approval(*log: Event, reason: str) -> RunState:
    """The state after the log, a pause awaiting approval, and its resume."""
    n = len(log)
    pause = make("run_paused", n, reason="awaiting_approval", resume_from=n - 1)
    resume = make("run_resumed", n + 1, reason=reason, resume_from=n - 1)
    state = RunState()
    for event in [*log, pause, resume]:
        state, _ = step(state, event)
    return state


def denial(state: RunState, seq: int) -> list[Effect]:
    denied = make("approval_denied", seq, tool_use_id="a", by="me", note="no")
    return step(state, denied, lambda name: name == "search")[1]


def test_a_denial_without_a_crash_fails_only_the_call_asked_about() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a", "b", "c")]
    log += [requested(2, "a", "write"), requested(3, "b", "search")]
    log += [requested(4, "c", "write"), asked(5)]  # the turn's lines written together
    state = paused_for_approval(*log, reason="approval")
    assert denial(state, 9) == [
        Fail("a", "denied by me: no"),
        RunTool("b"),
        RunTool("c"),
    ]


def test_after_a_crash_a_denial_fails_every_interrupted_call_with_effects() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a", "b", "c")]
    log += [requested(2, "a", "write"), requested(3, "b", "search")]
    log += [requested(4, "c", "write"), asked(5)]
    state = paused_for_approval(*log, reason="crash")
    assert state.interrupted == ("a", "b", "c")
    why = "denied by me: no"
    assert denial(state, 9) == [Fail("a", why), Fail("c", why), RunTool("b")]


def test_a_call_that_completes_is_no_longer_interrupted() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a", "b")]
    log += [requested(2, "a", "write"), requested(3, "b", "write")]
    log.append(make("tool_completed", 4, tool_use_id="b"))
    state = RunState()
    for event in [*log, make("run_resumed", 5, reason="crash")]:
        state, _ = step(state, event)
    assert state.interrupted == ("a",)


def test_a_crash_right_after_a_fork_from_a_final_answer_calls_the_model() -> None:
    parent = [make("run_started"), model_call(1, "end_turn")]
    origin: dict[str, object] = {"from": {"runId": "r1", "seq": 1}}
    state = RunState()
    for event in [*parent, make("run_forked", 0, "r2", **origin)]:
        state, _ = step(state, event)
    resumed = make("run_resumed", 1, "r2", reason="crash")
    assert step(state, resumed)[1] == [CallModel()]
