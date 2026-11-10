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
from pydantic import JsonValue
from strategies import logs

from loomcore import ladder
from loomcore import step as step_module
from loomcore.events import Event, JsonObject
from loomcore.replay import RunState, apply, replay, run_events
from loomcore.step import (
    AskHuman,
    CallModel,
    Effect,
    Fail,
    RequestCalls,
    RunTool,
    Stop,
    step,
)


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


def test_a_turns_tool_uses_are_requested_together_in_order() -> None:
    effects = run(make("run_started"), model_call(1, "tool_use", "b", "a"))[1]
    assert effects == [RequestCalls(("b", "a"))]


def test_once_every_call_is_requested_they_run_in_order() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "b", "a")]
    log += [requested(2, "b", "search"), requested(3, "a", "search")]
    assert run(*log)[1] == [RunTool("b"), RunTool("a")]
    assert run(*log[:3])[1] == []  # the batch is not complete yet


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
    assert effects == [RequestCalls(("a",))]


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
        "loomcore",
        "loomcore.events",
        "loomcore.finish",
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
    unrequested = crash(*log, read_only=frozenset({"search"}))
    assert unrequested == [RequestCalls(("b",))]  # b never started: request, then run
    log.append(requested(3, "b", "search"))
    both = crash(*log, read_only=frozenset({"search"}))
    assert both == [RunTool("a"), RunTool("b")]


def test_a_crash_asks_before_rerunning_a_call_that_may_have_had_effects() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a", "b")]
    log += [requested(2, "a", "write_file"), requested(3, "b", "search")]
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


def test_the_crash_question_names_every_interrupted_call_with_effects() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a", "b", "c")]
    log += [requested(2, "a", "write"), requested(3, "b", "search")]
    log += [requested(4, "c", "delete")]
    [ask] = crash(*log, read_only=frozenset({"search"}))
    assert isinstance(ask, AskHuman)
    assert "write (a)" in ask.question
    assert "delete (c)" in ask.question
    assert "search" not in ask.question


def test_an_open_approval_does_not_hide_an_interrupted_call_from_the_question() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a", "c")]
    log += [requested(2, "c", "write"), asked(3, "a")]
    [ask] = crash(*log)
    assert isinstance(ask, AskHuman)
    assert (ask.tool_use_id, "write (c)" in ask.question) == ("a", True)


def test_a_tool_requested_for_a_call_nobody_asked_for_changes_nothing() -> None:
    state = apply(RunState(), requested(0, "ghost", "write"))
    assert state.requested == ()


# --- f11: the ladder and finish ------------------------------------------------


def escalation(seq: int, rung: int, *tripped: str) -> Event:
    return make(
        "escalation",
        seq,
        rung=rung,
        reason="repeat",
        tool_use_id=tripped[0],
        call_hash="h",
        tripped=list(tripped),
    )


def judged(rung: int, *tripped: str) -> list[Event]:
    """A turn of calls a, b, c, judged at `rung` (written first), then requested."""
    log = [make("run_started"), model_call(1, "tool_use", "a", "b", "c")]
    log.append(escalation(2, rung, *tripped))
    return [
        *log,
        requested(3, "a", "n"),
        requested(4, "b", "n"),
        requested(5, "c", "n"),
    ]


def test_a_nudge_runs_every_call_and_marks_only_the_ones_that_tripped() -> None:
    effects = run(*judged(1, "a", "c"))[1]
    note = ladder.NUDGE
    assert effects == [RunTool("a", note), RunTool("b"), RunTool("c", note)]


def test_a_switch_runs_every_call_with_the_switch_message() -> None:
    effects = run(*judged(2, "b"))[1]
    assert effects == [RunTool("a"), RunTool("b", ladder.SWITCH), RunTool("c")]


def test_an_ask_holds_every_call_and_asks_about_the_first_that_tripped() -> None:
    assert run(*judged(3, "b", "c"))[1] == [AskHuman("b", "repeat")]


def test_a_stuck_turn_stops() -> None:
    assert run(*judged(4, "a"))[1] == [Stop("stuck")]


def test_the_ladder_never_goes_down_and_a_fork_starts_at_zero() -> None:
    state, _ = run(*judged(3, "a"))
    assert state.rung == 3
    forked = make("run_forked", 0, "r2", **{"from": {"runId": "r1", "seq": 0}})
    state, _ = step(state, forked)
    assert (state.rung, state.call_hashes, state.ladder_asks) == (0, (), ())


def test_a_grant_on_a_ladder_ask_forgets_the_earlier_calls_with_that_hash() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a", "b")]
    log += [escalation(2, 3, "a")]
    log += [make("tool_requested", 3, tool_use_id="a", name="n", call_hash="h")]
    log += [make("tool_requested", 4, tool_use_id="b", name="n", call_hash="x")]
    log += [asked(5, "a")]
    log += [make("run_paused", 6, reason="awaiting_approval", resume_from=5)]
    log += [make("run_resumed", 7, reason="approval", resume_from=5)]
    before, _ = run(*log)
    assert before.call_hashes == ("h", "x")
    after, effects = step(before, make("approval_granted", 8, tool_use_id="a"))
    assert after.call_hashes == ("", "x")  # blanked, so the window does not shift
    assert after.rung == 3  # a human approved the call once, not for good
    assert effects == [RunTool("a"), RunTool("b")]


def finish_call(seq: int, *others: str) -> Event:
    blocks: list[JsonValue] = [
        {"type": "tool_use", "id": "f", "name": "finish", "input": {"result": 42}}
    ]
    blocks += [{"type": "tool_use", "id": i, "name": "n", "input": {}} for i in others]
    message: JsonObject = {"content": blocks, "stop_reason": "tool_use"}
    return make("model_call", seq, message=message)


def test_a_finish_call_ends_the_run_with_its_result() -> None:
    effects = run(make("run_started"), finish_call(1))[1]
    assert effects == [Stop("finished", 42, "f")]


def test_a_finish_call_beside_other_calls_still_ends_the_run() -> None:
    effects = run(make("run_started"), finish_call(1, "a"))[1]
    assert effects == [Stop("finished", 42, "f")]


def test_a_crash_after_a_stuck_rung_finishes_the_rung() -> None:
    assert crash(*judged(4, "a")) == [Stop("stuck")]


def test_a_crash_after_an_ask_rung_asks_again_without_running_anything() -> None:
    effects = crash(*judged(3, "b"))
    assert effects == [AskHuman("b", "repeat")]


def test_a_crash_after_a_nudge_finishes_it_but_asks_first_about_risky_calls() -> None:
    log = judged(1, "a")
    assert crash(*log, read_only=frozenset({"n"})) == [
        RunTool("a", ladder.NUDGE),
        RunTool("b"),
        RunTool("c"),
    ]
    [ask] = crash(*log)  # "n" is not read-only: all three may have run
    assert isinstance(ask, AskHuman)


def decided_then_crash(granted: bool) -> list[Effect]:
    log = [make("run_started"), model_call(1, "tool_use", "a", "b")]
    log += [requested(2, "a", "write"), requested(3, "b", "write"), asked(4, "a")]
    log += [make("run_paused", 5, reason="awaiting_approval", resume_from=4)]
    log += [make("run_resumed", 6, reason="approval", resume_from=4)]
    kind = "approval_granted" if granted else "approval_denied"
    log += [make(kind, 7, tool_use_id="a", by="me", note="no")]
    return crash(*log)


def test_a_grant_logged_before_a_crash_is_settled_not_asked_again() -> None:
    assert decided_then_crash(True) == [RunTool("a"), RunTool("b")]


def test_a_denial_logged_before_a_crash_is_settled_not_asked_again() -> None:
    why = "denied by me: no"
    assert decided_then_crash(False) == [Fail("a", why), Fail("b", why)]


def test_a_crash_after_a_finish_call_ends_the_run_and_does_not_run_anything() -> None:
    log = [make("run_started"), finish_call(1, "a")]
    assert crash(*log) == [Stop("finished", 42, "f")]
    partly = [*log, make("tool_completed", 2, tool_use_id="f")]
    assert crash(*partly) == [Stop("finished", 42, "f")]  # a is still pending


def test_a_crash_after_every_call_was_settled_still_writes_the_ending() -> None:
    log = [make("run_started"), finish_call(1)]
    log.append(make("tool_completed", 2, tool_use_id="f"))
    assert crash(*log) == [Stop("finished", 42, "f")]


def test_a_crash_after_the_stuck_completions_still_stops() -> None:
    log = [make("run_started"), model_call(1, "tool_use", "a"), escalation(2, 4, "a")]
    log += [
        requested(3, "a", "n"),
        make("tool_completed", 4, tool_use_id="a", ok=False),
    ]
    assert crash(*log) == [Stop("stuck")]


def decided_turn(granted: bool, *completed: str) -> list[Event]:
    """An ask about `a` in the turn [a, b], answered, with some calls completed."""
    log = [make("run_started"), model_call(1, "tool_use", "a", "b")]
    log += [
        escalation(2, 3, "a"),
        requested(3, "a", "write"),
        requested(4, "b", "write"),
    ]
    log += [asked(5, "a")]
    log += [make("run_paused", 6, reason="awaiting_approval", resume_from=5)]
    log += [make("run_resumed", 7, reason="approval", resume_from=5)]
    kind = "approval_granted" if granted else "approval_denied"
    log += [make(kind, 8, tool_use_id="a", by="me", note="no")]
    for n, i in enumerate(completed):
        log.append(make("tool_completed", 9 + n, tool_use_id=i))
    return log


def test_a_grant_covers_the_whole_turn_even_after_the_asked_call_completed() -> None:
    assert crash(*decided_turn(True, "a")) == [RunTool("b")]


def test_a_denial_covers_every_interrupted_call_even_after_one_was_failed() -> None:
    why = "denied by me: no"
    assert crash(*decided_turn(False, "a")) == [Fail("b", why)]
