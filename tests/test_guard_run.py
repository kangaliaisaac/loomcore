"""f19, the pre stage: a deny stops a call, an ask pauses the run (spec 3.7)."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from factories import calling, make
from fakes import CAPS, CONFIG, Crash, World, kinds, reply, use
from fakes import ref as run_ref

from loomcore.eventlog import open_writer, read_events
from loomcore.events import ApprovalRequested, GuardDecision
from loomcore.guards import Allow, Ask, Deny, Guards, PreDecision, ToolRequest
from loomcore.limits import Limits
from loomcore.loop import Driver
from loomcore.ports import Decision, Task
from loomcore.replay import RunState, unanswered_asks
from loomcore.step import RunTool, step

TASK = Task("go", CAPS)


@dataclass(frozen=True)
class Rule:
    """Says `then` about the calls named in `ids`, allows the rest."""

    ids: tuple[str, ...]
    then: PreDecision

    def check(self, request: ToolRequest, state: RunState) -> PreDecision:
        return self.then if request.tool_use_id in self.ids else Allow()


def driver(world: World, tmp: Path, *rules: Rule) -> Driver:
    return world.driver(tmp, guards=Guards(pre=rules))


def log(tmp: Path) -> list[str]:
    return kinds(read_events(tmp / "s1.jsonl").events)


def test_a_deny_stops_the_call_and_the_model_sees_the_reason(tmp_path: Path) -> None:
    world = World(script=[use(("a", "write"), ("b", "read")), reply("ok")])
    result = driver(world, tmp_path, Rule(("a",), Deny("no writes here"))).run(TASK)
    assert result.status == "finished"
    assert world.tool_calls == ["b"]
    sent = str(world.requests[-1].messages[-1]["content"])
    assert "no writes here" in sent
    decisions = [
        e
        for e in read_events(tmp_path / "s1.jsonl").events
        if isinstance(e, GuardDecision)
    ]
    assert [(e.stage, e.tool_use_id, e.decision) for e in decisions] == [
        ("pre", "a", "deny")
    ]
    assert log(tmp_path).index("guard_decision") < log(tmp_path).index("tool_requested")


def test_an_allow_leaves_no_line(tmp_path: Path) -> None:
    world = World(script=[use(("a", "read")), reply()])
    driver(world, tmp_path, Rule((), Deny("x"))).run(TASK)
    assert "guard_decision" not in log(tmp_path)


def test_an_ask_pauses_with_its_question_and_nothing_runs(tmp_path: Path) -> None:
    world = World(script=[use(("a", "write"), ("b", "read")), reply("ok")])
    paused = driver(world, tmp_path, Rule(("a",), Ask("write a?"))).run(TASK)
    assert (paused.status, paused.reason, paused.question) == (
        "paused",
        "awaiting_approval",
        "write a?",
    )
    assert world.tool_calls == []  # b waits too: the whole turn is held


def test_a_grant_runs_the_turn(tmp_path: Path) -> None:
    world = World(script=[use(("a", "write"), ("b", "read")), reply("ok")])
    drv = driver(world, tmp_path, Rule(("a",), Ask("write a?")))
    paused = drv.run(TASK)
    done = drv.resume(paused.ref, Decision(granted=True, by="me"))
    assert done.status == "finished"
    assert world.tool_calls == ["a", "b"]


def test_a_denied_ask_fails_only_that_call(tmp_path: Path) -> None:
    world = World(script=[use(("a", "write"), ("b", "read")), reply("ok")])
    drv = driver(world, tmp_path, Rule(("a",), Ask("write a?")))
    paused = drv.run(TASK)
    done = drv.resume(paused.ref, Decision(granted=False, by="me", note="no"))
    assert done.status == "finished"
    assert world.tool_calls == ["b"]
    assert "denied by me: no" in str(world.requests[-1].messages[-1]["content"])


def test_two_asks_are_asked_one_after_the_other(tmp_path: Path) -> None:
    world = World(script=[use(("a", "write"), ("b", "write")), reply("ok")])
    drv = driver(
        world,
        tmp_path,
        Rule(("a",), Ask("a?")),
        Rule(("b",), Ask("b?")),
    )
    first = drv.run(TASK)
    assert first.question == "a?"
    second = drv.resume(first.ref, Decision(granted=True, by="me"))
    assert (second.status, second.question) == ("paused", "b?")
    assert world.tool_calls == []  # still nothing: one question is open
    done = drv.resume(second.ref, Decision(granted=True, by="me"))
    assert (done.status, world.tool_calls) == ("finished", ["a", "b"])


def test_a_deny_beside_an_ask_fails_at_once_and_the_ask_still_pauses(
    tmp_path: Path,
) -> None:
    world = World(script=[use(("a", "write"), ("b", "write")), reply("ok")])
    drv = driver(world, tmp_path, Rule(("a",), Deny("never")), Rule(("b",), Ask("b?")))
    paused = drv.run(TASK)
    assert paused.question == "b?"
    assert log(tmp_path).count("tool_completed") == 1  # a, failed before the pause
    done = drv.resume(paused.ref, Decision(granted=True, by="me"))
    assert (done.status, world.tool_calls) == ("finished", ["b"])


def test_a_crash_after_the_grant_runs_the_turn_on_resume(tmp_path: Path) -> None:
    world = World(script=[use(("a", "write"), ("b", "read")), reply("ok")])
    world.crash_on_call = 1  # the first tool call after the grant kills the run
    rule = Rule(("a",), Ask("write a?"))
    paused = driver(world, tmp_path, rule).run(TASK)
    with pytest.raises(Crash):
        driver(world, tmp_path, rule).resume(
            paused.ref, Decision(granted=True, by="me")
        )
    # "a" may have run: the crash rule asks again, and the guard's ask is not repeated
    again = driver(world, tmp_path, rule).resume(run_ref())
    assert again.question is not None
    assert "write (a)" in again.question


def test_a_run_killed_before_the_question_is_logged_asks_it_on_resume(
    tmp_path: Path,
) -> None:
    limits = Limits.model_validate({"run_cap": None, "month_cap": None})
    rows = [
        make(
            "run_started",
            0,
            task="go",
            config=CONFIG.to_json(),
            limits=limits.to_json(),
            config_digest=CONFIG.digest(),
        ),
        make("turn_started", 1, turn=1),
        make("model_call", 2, message=use(("a", "write"))),
        make(
            "guard_decision",
            3,
            stage="pre",
            tool_use_id="a",
            decision="ask",
            reason="write a?",
        ),
    ]
    with open_writer(tmp_path / "s1.jsonl") as writer:
        for row in rows:
            writer.append(row)
    world = World(script=[reply("ok")])
    paused = driver(world, tmp_path, Rule(("a",), Ask("write a?"))).resume(run_ref())
    assert paused.question == "write a?"
    assert world.tool_calls == []
    asked = [
        e
        for e in read_events(tmp_path / "s1.jsonl").events
        if isinstance(e, ApprovalRequested)
    ]
    assert [e.question for e in asked] == ["write a?"]


def test_no_run_tool_is_produced_while_an_ask_is_unanswered() -> None:
    """Fold a turn event by event: every effect list while a question is open holds
    no RunTool, whatever the order the answers come in."""
    ids = ("a", "b", "c")
    rows = [
        make("model_call", 0, message=calling(*ids)),
        make(
            "guard_decision",
            1,
            stage="pre",
            tool_use_id="a",
            decision="ask",
            reason="a?",
        ),
        make(
            "guard_decision",
            2,
            stage="pre",
            tool_use_id="b",
            decision="ask",
            reason="b?",
        ),
        *(
            make(
                "tool_requested",
                3 + n,
                tool_use_id=i,
                name="w",
                input={"q": i},
                call_hash=i,
            )
            for n, i in enumerate(ids)
        ),
        make("approval_requested", 6, tool_use_id="a", question="a?"),
        make("approval_granted", 7, tool_use_id="a", by="me", note=""),
        make("approval_requested", 8, tool_use_id="b", question="b?"),
        make("approval_denied", 9, tool_use_id="b", by="me", note="no"),
    ]
    state = RunState()
    for row in rows:
        state, effects = step(state, row)
        if unanswered_asks(state):
            assert not any(isinstance(e, RunTool) for e in effects)
    assert not unanswered_asks(state)
