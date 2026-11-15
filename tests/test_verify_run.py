"""f20: a failing verifier prevents `finished` and feeds back; a passing one ends the
run (spec 3.6)."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from factories import make
from fakes import (
    CAPS,
    World,
    completed,
    finish,
    kinds,
    log,
    reply,
    run_started,
    use,
    write_log,
)
from fakes import ref as run_ref

from loomcore.eventlog import read_events
from loomcore.events import (
    ContextPacked,
    Event,
    ForkOrigin,
    JsonObject,
    ToolCompleted,
)
from loomcore.limits import Limits
from loomcore.ports import Task, ToolResult
from loomcore.replay import RunState, replay, run_events
from loomcore.verify import Verdict, VerifierError, critique

TASK = Task("go", CAPS)


@dataclass
class Scripted:
    """Answers in order (the last one repeats) and counts how often it was asked."""

    answers: list[bool]
    calls: int = 0
    seen: list[RunState] = field(default_factory=list)

    def __call__(self, state: RunState) -> Verdict:
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        self.seen.append(state)
        return Verdict(answer, "tests", "" if answer else "2 tests fail")


class Never:
    """A verifier a test expects not to be asked."""

    def __call__(self, state: RunState) -> Verdict:
        raise AssertionError("the verifier must not run here")


def last_user_text(world: World) -> str:
    """The text of the last message sent, tool results included."""
    content = world.requests[-1].messages[-1]["content"]
    if isinstance(content, str):
        return content
    assert isinstance(content, list)
    return "\n".join(
        str(block["content"]) for block in content if isinstance(block, dict)
    )


def test_a_passing_verifier_ends_the_run_and_is_logged_first(tmp_path: Path) -> None:
    world = World(script=[reply("all done")])
    verifier = Scripted([True])
    result = world.driver(tmp_path, verifier=verifier).run(TASK)
    assert (result.status, result.reason) == ("finished", "end_turn")
    assert verifier.calls == 1
    assert log(tmp_path)[-2:] == ["verification_passed", "run_finished"]


def test_a_failing_verifier_prevents_end_turn_and_feeds_back(tmp_path: Path) -> None:
    world = World(script=[reply("done"), reply("really done")])
    verifier = Scripted([False, True])
    result = world.driver(tmp_path, verifier=verifier).run(TASK)
    assert (result.status, result.result) == ("finished", "really done")
    assert critique("tests", "2 tests fail") in last_user_text(world)
    assert log(tmp_path).count("verification_failed") == 1
    assert log(tmp_path)[-2:] == ["verification_passed", "run_finished"]


def test_a_failing_verifier_rejects_a_finish_call_through_its_result(
    tmp_path: Path,
) -> None:
    world = World(script=[finish("x", call_id="f1"), finish("y", call_id="f2")])
    result = world.driver(tmp_path, verifier=Scripted([False, True])).run(TASK)
    assert (result.status, result.result) == ("finished", "y")
    first = completed(tmp_path)["f1"]
    assert (first.ok, first.content) == (False, critique("tests", "2 tests fail"))
    assert completed(tmp_path)["f2"].ok  # the accepted finish call
    assert critique("tests", "2 tests fail") in last_user_text(world)
    assert world.tool_calls == []  # finish is never run as a tool


def test_the_other_calls_beside_a_rejected_finish_are_not_run(tmp_path: Path) -> None:
    world = World(
        script=[finish("x", ("a", "write"), call_id="f1"), finish("y", call_id="f2")]
    )
    world.driver(tmp_path, verifier=Scripted([False, True])).run(TASK)
    assert completed(tmp_path)["a"].ok is False
    assert world.tool_calls == []


def test_a_rejected_finish_completes_its_other_calls_first(tmp_path: Path) -> None:
    """The finish call's completion is what makes resume forget the claim, so it
    is written last, whatever order the model listed the calls in."""
    world = World(
        script=[finish("x", ("a", "write"), call_id="f1"), finish("y", call_id="f2")]
    )
    world.driver(tmp_path, verifier=Scripted([False, True])).run(TASK)
    first = log(tmp_path)
    at = first.index("verification_failed")
    assert first[at - 2 : at + 1] == [
        "tool_completed",
        "tool_completed",
        "verification_failed",
    ]
    order = [
        e.tool_use_id
        for e in read_events(tmp_path / "s1.jsonl").events
        if isinstance(e, ToolCompleted)
    ]
    assert order[:2] == ["a", "f1"]


def test_a_stop_that_is_not_a_claim_is_not_verified(tmp_path: Path) -> None:
    world = World(script=[reply("no", stop="refusal")])
    result = world.driver(tmp_path, verifier=Never()).run(TASK)
    assert (result.status, result.reason) == ("finished", "refused")


def test_the_cap_pauses_the_run_and_a_resume_with_more_continues(
    tmp_path: Path,
) -> None:
    limits = Limits.model_validate(
        {"run_cap": None, "month_cap": None, "max_verifications": 2}
    )
    world = World(script=[reply("a"), reply("b"), reply("c"), reply("d")])
    driver = world.driver(tmp_path, verifier=Scripted([False, False, False, True]))
    paused = driver.run(Task("go", limits))
    assert (paused.status, paused.reason, paused.limit) == (
        "paused",
        "limit",
        "max_verifications",
    )
    assert len(world.requests) == 2  # no third call after the second rejection
    more = limits.model_copy(update={"max_verifications": 4})
    done = driver.resume(paused.ref, limits=more)
    assert (done.status, done.result) == ("finished", "d")


def _started(*rest: Event) -> list[Event]:
    return [
        run_started(),
        make("turn_started", 1, turn=1),
        *rest,
    ]


def test_a_crash_after_verification_failed_calls_the_model_not_the_verifier(
    tmp_path: Path,
) -> None:
    rows = _started(
        make("model_call", 2, message=reply("done")),
        make("verification_failed", 3, verifier="tests", message="2 tests fail"),
    )
    write_log(tmp_path, rows)
    world = World(script=[reply("fixed")])
    verifier = Scripted([True])
    result = world.driver(tmp_path, verifier=verifier).resume(run_ref())
    assert (result.status, result.result) == ("finished", "fixed")
    assert critique("tests", "2 tests fail") in last_user_text(world)
    assert verifier.calls == 1  # for the new claim only


def test_a_crash_after_verification_passed_ends_without_a_second_check(
    tmp_path: Path,
) -> None:
    rows = _started(
        make("model_call", 2, message=reply("done")),
        make("verification_passed", 3, verifier="tests", message=""),
    )
    write_log(tmp_path, rows)
    result = World(script=[]).driver(tmp_path, verifier=Never()).resume(run_ref())
    assert (result.status, result.reason) == ("finished", "end_turn")


def test_a_crash_between_a_rejected_finish_and_its_failure_line_goes_on(
    tmp_path: Path,
) -> None:
    rows = _started(
        make("model_call", 2, message=finish("x", call_id="f1")),
        make(
            "tool_completed",
            3,
            tool_use_id="f1",
            ok=False,
            content=critique("tests", "2 tests fail"),
        ),
    )
    write_log(tmp_path, rows)
    world = World(script=[reply("fixed")])
    result = world.driver(tmp_path, verifier=Scripted([True])).resume(run_ref())
    assert (result.status, result.result) == ("finished", "fixed")
    assert critique("tests", "2 tests fail") in last_user_text(world)


def test_a_crash_before_the_verdict_is_logged_asks_the_verifier_again(
    tmp_path: Path,
) -> None:
    write_log(tmp_path, _started(make("model_call", 2, message=reply("done"))))
    verifier = Scripted([True])
    result = World(script=[]).driver(tmp_path, verifier=verifier).resume(run_ref())
    assert (result.status, verifier.calls) == ("finished", 1)


def test_a_fork_counts_its_own_rejections(tmp_path: Path) -> None:
    """The parent was rejected once before the fork point. A fork with a cap of one
    starts again from none, so it calls the model, is rejected itself and pauses on
    the cap; had it inherited the parent's count it would pause before any call."""
    world = World(
        script=[reply("a"), use(("t1", "search")), reply("b"), reply("c")],
    )
    driver = world.driver(tmp_path, verifier=Scripted([False, True, False]))
    parent = driver.run(TASK)
    assert (parent.status, parent.result) == ("finished", "b")
    first = read_events(tmp_path / "s1.jsonl").events
    assert replay(run_events(first, "r1")).verification_failures == 1
    clean = next(e for e in first if isinstance(e, ToolCompleted))
    one = Limits.model_validate(
        {"run_cap": None, "month_cap": None, "max_verifications": 1}
    )
    forked = driver.fork(ForkOrigin(run_id="r1", seq=clean.seq), limits=one)
    assert (forked.status, forked.limit) == ("paused", "max_verifications")
    events = read_events(tmp_path / "s1.jsonl").events
    assert kinds(events, "r2").count("model_call") == 1
    assert replay(run_events(events, "r2")).verification_failures == 1


def test_the_run_without_a_verifier_ends_as_before(tmp_path: Path) -> None:
    world = World(script=[reply("done")])
    result = world.driver(tmp_path).run(TASK)
    assert result.status == "finished"
    assert "verification_passed" not in log(tmp_path)


def test_a_verifier_sees_the_state_at_the_claim(tmp_path: Path) -> None:
    world = World(script=[reply("done")])
    verifier = Scripted([True])
    world.driver(tmp_path, verifier=verifier).run(TASK)
    assert verifier.seen[0].turns == 1


def test_a_crash_after_the_other_calls_of_a_rejected_finish_judges_the_claim_again(
    tmp_path: Path,
) -> None:
    """The other call is completed before the finish call, so a log cut after it still
    holds the claim and the call never runs."""
    rows = _started(
        make("model_call", 2, message=finish("x", ("a", "write"), call_id="f1")),
        make("tool_completed", 3, tool_use_id="a", ok=False, content="not run"),
    )
    write_log(tmp_path, rows)
    world = World(script=[reply("fixed")])
    verifier = Scripted([False, True])
    result = world.driver(tmp_path, verifier=verifier).resume(run_ref())
    assert (result.status, verifier.calls) == ("finished", 2)
    assert world.tool_calls == []  # "a" never ran


def test_an_empty_reply_that_is_rejected_still_tells_the_model(tmp_path: Path) -> None:
    empty: JsonObject = {"content": [], "stop_reason": "end_turn"}
    world = World(script=[empty, reply("now with text")])
    world.driver(tmp_path, verifier=Scripted([False, True])).run(TASK)
    assert critique("tests", "2 tests fail") in last_user_text(world)


def test_an_empty_reply_after_a_tool_turn_still_gets_its_critique(
    tmp_path: Path,
) -> None:
    empty: JsonObject = {"content": [], "stop_reason": "end_turn"}
    world = World(script=[use(("t1", "search")), empty, reply("now with text")])
    world.driver(tmp_path, verifier=Scripted([False, True])).run(TASK)
    assert critique("tests", "2 tests fail") in last_user_text(world)


class Broken:
    """A verifier that cannot reach a verdict."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def __call__(self, state: RunState) -> Verdict:
        raise self.error


def test_a_verifier_that_cannot_judge_ends_the_run_with_an_error(
    tmp_path: Path,
) -> None:
    world = World(script=[finish("x", ("a", "write"), call_id="f1")])
    verifier = Broken(VerifierError("pytest would not start"))
    result = world.driver(tmp_path, verifier=verifier).run(TASK)
    assert (result.status, result.reason) == ("finished", "error")
    assert log(tmp_path)[-1] == "run_finished"
    assert not {"verification_passed", "verification_failed"} & set(log(tmp_path))
    assert not completed(tmp_path)["f1"].ok
    assert not completed(tmp_path)["a"].ok  # every call is settled
    assert world.tool_calls == []


def test_any_other_error_from_a_verifier_is_a_bug_and_propagates(
    tmp_path: Path,
) -> None:
    world = World(script=[reply("done")])
    with pytest.raises(ZeroDivisionError):
        world.driver(tmp_path, verifier=Broken(ZeroDivisionError())).run(TASK)


def test_the_critique_stays_in_a_window_that_drops_older_exchanges(
    tmp_path: Path,
) -> None:
    ids = [f"t{k}" for k in range(14)]
    world = World(
        script=[reply("early"), *(use((i, "search")) for i in ids), reply("done")],
        tool_results={i: ToolResult(True, f"{i} " + "data " * 80) for i in ids},
        read_only={"search"},
    )
    limits = Limits.model_validate(
        {"run_cap": None, "month_cap": None, "max_turns": 100, "context_budget": 700}
    )
    result = world.driver(tmp_path, verifier=Scripted([False, True])).run(
        Task("go", limits)
    )
    assert result.status == "finished"
    packed = [
        e
        for e in read_events(tmp_path / "s1.jsonl").events
        if isinstance(e, ContextPacked)
    ]
    assert any(e.dropped for e in packed)  # the budget did force drops
    failed = next(
        e
        for e in read_events(tmp_path / "s1.jsonl").events
        if e.kind == "verification_failed"
    )
    key = f"verify:r1:{failed.seq}"
    assert all(key in e.selected for e in packed if e.seq > failed.seq)
    sent = [m["content"] for m in world.requests[-1].messages]
    assert critique("tests", "2 tests fail") in sent


def test_a_crash_after_a_verifier_error_settled_the_finish_call_goes_on(
    tmp_path: Path,
) -> None:
    """The error stop settled the finish call `ok=false` and the process died before
    `run_finished`. Resume treats it like a crash during any error stop: the model
    is called with the settled result and the verifier is asked at its next claim."""
    rows = _started(
        make("model_call", 2, message=finish("x", call_id="f1")),
        make(
            "tool_completed",
            3,
            tool_use_id="f1",
            ok=False,
            content="run ended: error",
        ),
    )
    write_log(tmp_path, rows)
    world = World(script=[reply("fixed")])
    verifier = Scripted([True])
    result = world.driver(tmp_path, verifier=verifier).resume(run_ref())
    assert (result.status, result.result) == ("finished", "fixed")
    assert verifier.calls == 1  # asked once, at the claim after the resume
    assert "run ended: error" in last_user_text(world)
