"""f20: a failing verifier prevents `finished` and feeds back; a passing one ends the
run (spec 3.6)."""

from dataclasses import dataclass, field
from pathlib import Path

from factories import make
from fakes import CAPS, CONFIG, World, finish, kinds, reply
from fakes import ref as run_ref

from loomcore.eventlog import open_writer, read_events
from loomcore.events import Event, ToolCompleted
from loomcore.limits import Limits
from loomcore.ports import Task
from loomcore.replay import RunState, replay
from loomcore.verify import Verdict, critique

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


def log(tmp: Path) -> list[str]:
    return kinds(read_events(tmp / "s1.jsonl").events)


def completed(tmp: Path) -> dict[str, ToolCompleted]:
    return {
        e.tool_use_id: e
        for e in read_events(tmp / "s1.jsonl").events
        if isinstance(e, ToolCompleted)
    }


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
    limits = Limits.model_validate({"run_cap": None, "month_cap": None})
    return [
        make(
            "run_started",
            0,
            task="go",
            config=CONFIG.to_json(),
            limits=limits.to_json(),
            config_digest=CONFIG.digest(),
        ),
        make("turn_started", 1, turn=1),
        *rest,
    ]


def _write(tmp: Path, rows: list[Event]) -> None:
    with open_writer(tmp / "s1.jsonl") as writer:
        for row in rows:
            writer.append(row)


def test_a_crash_after_verification_failed_calls_the_model_not_the_verifier(
    tmp_path: Path,
) -> None:
    rows = _started(
        make("model_call", 2, message=reply("done")),
        make("verification_failed", 3, verifier="tests", message="2 tests fail"),
    )
    _write(tmp_path, rows)
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
    _write(tmp_path, rows)
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
    _write(tmp_path, rows)
    world = World(script=[reply("fixed")])
    result = world.driver(tmp_path, verifier=Scripted([True])).resume(run_ref())
    assert (result.status, result.result) == ("finished", "fixed")
    assert critique("tests", "2 tests fail") in last_user_text(world)


def test_a_crash_before_the_verdict_is_logged_asks_the_verifier_again(
    tmp_path: Path,
) -> None:
    _write(tmp_path, _started(make("model_call", 2, message=reply("done"))))
    verifier = Scripted([True])
    result = World(script=[]).driver(tmp_path, verifier=verifier).resume(run_ref())
    assert (result.status, verifier.calls) == ("finished", 1)


def test_a_fork_counts_its_own_rejections() -> None:
    parent = [
        *_started(
            make("model_call", 2, message=reply("done")),
            make("verification_failed", 3, verifier="tests", message="x"),
        ),
    ]
    assert replay(parent).verification_failures == 1
    fork = make("run_forked", 4, run="r2")
    assert replay([*parent, fork]).verification_failures == 0


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
