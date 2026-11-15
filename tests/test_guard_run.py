"""f19, the pre stage: a deny stops a call, an ask pauses the run (spec 3.7)."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from factories import calling, make
from fakes import CAPS, Crash, World, log, reply, run_started, use, write_log
from fakes import ref as run_ref

from loomcore.eventlog import read_events
from loomcore.events import ApprovalRequested, GuardDecision, JsonObject, ToolCompleted
from loomcore.guards import (
    Allow,
    Ask,
    Deny,
    Guards,
    PostDecision,
    PreDecision,
    ToolRequest,
)
from loomcore.limits import Limits
from loomcore.loop import Driver
from loomcore.ports import Decision, Task, ToolResult
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


def test_a_crash_after_the_grant_asks_run_again_not_the_guard_question(
    tmp_path: Path,
) -> None:
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
        run_started(limits),
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
    write_log(tmp_path, rows)
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
    no RunTool (a grant, then a denial)."""
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


REPEATING = Task(
    "go",
    Limits.model_validate({"run_cap": None, "month_cap": None, "repeat_threshold": 2}),
)


def repeats(count: int) -> list[JsonObject | Exception]:
    return [use((f"c{n}", "search"), q="x") for n in range(1, count + 1)]


def test_a_guard_denied_call_is_failed_not_asked_about_by_the_ladder(
    tmp_path: Path,
) -> None:
    """c4 reaches the ladder's ask rung, but a guard denied it: nobody is asked."""
    world = World(script=repeats(6))
    result = driver(world, tmp_path, Rule(("c4",), Deny("barred"))).run(REPEATING)
    assert (result.status, result.reason) == ("finished", "stuck")
    done = [
        e
        for e in read_events(tmp_path / "s1.jsonl").events
        if isinstance(e, ToolCompleted) and e.tool_use_id == "c4"
    ]
    assert [e.content for e in done] == ["barred"]


def test_the_human_sees_the_guards_question_with_the_ladders(tmp_path: Path) -> None:
    world = World(script=repeats(6))
    rule = Rule(("c4",), Ask("this repeats a search; sure?"))
    paused = driver(world, tmp_path, rule).run(REPEATING)
    assert paused.question is not None
    assert "this repeats a search; sure?" in paused.question
    again = driver(world, tmp_path, rule).resume(
        paused.ref, Decision(granted=True, by="me")
    )
    assert again.status == "finished"  # one answer covers the call: not asked twice


def test_a_guard_denied_call_at_the_stuck_rung_does_not_call_the_model_again(
    tmp_path: Path,
) -> None:
    world = World(script=repeats(6))
    rules = (Rule(("c4",), Deny("barred")), Rule(("c5",), Deny("barred again")))
    result = driver(world, tmp_path, *rules).run(REPEATING)
    assert (result.status, result.reason) == ("finished", "stuck")
    assert len(world.requests) == 5  # nothing after the stuck turn
    last = [
        e
        for e in read_events(tmp_path / "s1.jsonl").events
        if isinstance(e, ToolCompleted) and e.tool_use_id == "c5"
    ]
    assert [e.content for e in last] == ["barred again"]


def test_a_crash_after_a_denied_tripped_call_asks_about_the_other_write() -> None:
    """The ask rung's only tripped call was denied by a guard, so nobody was asked and
    the other call ran unprotected: a crash must still find it interrupted."""
    rows = [
        make("model_call", 0, message=use(("t", "search"), ("w", "write"))),
        make(
            "guard_decision",
            1,
            stage="pre",
            tool_use_id="t",
            decision="deny",
            reason="no",
        ),
        make(
            "escalation",
            2,
            rung=3,
            reason="repeat",
            tool_use_id="t",
            call_hash="h",
            tripped=["t"],
        ),
        make(
            "tool_requested", 3, tool_use_id="t", name="search", input={}, call_hash="h"
        ),
        make(
            "tool_requested", 4, tool_use_id="w", name="write", input={}, call_hash="i"
        ),
        make("tool_completed", 5, tool_use_id="t"),
        make("run_resumed", 6, reason="crash"),
    ]
    state = RunState()
    for row in rows:
        state, _ = step(state, row)
    assert state.interrupted == ("w",)


def test_a_crash_before_the_request_lines_keeps_a_denial_after_them(
    tmp_path: Path,
) -> None:
    rows = [
        run_started(Limits.model_validate({"run_cap": None, "month_cap": None})),
        make("turn_started", 1, turn=1),
        make("model_call", 2, message=use(("a", "write"), ("b", "read"))),
        make(
            "guard_decision",
            3,
            stage="pre",
            tool_use_id="a",
            decision="deny",
            reason="no",
        ),
    ]
    write_log(tmp_path, rows)
    world = World(script=[reply("ok")])
    result = driver(world, tmp_path, Rule(("a",), Deny("no"))).resume(run_ref())
    assert result.status == "finished"
    assert world.tool_calls == ["b"]
    order = [e.kind for e in read_events(tmp_path / "s1.jsonl").events]
    after = order[order.index("guard_decision") + 1 :]
    assert after.index("tool_requested") < after.index("tool_completed")
    assert after.count("guard_decision") == 0  # the verdict was not written twice


def test_a_grant_on_the_second_tripped_call_resets_the_repeat_count(
    tmp_path: Path,
) -> None:
    """The ask rung's first tripped call is guard-denied, so the ladder asks about
    the second; its grant must count as the ladder's answer."""
    script: list[JsonObject | Exception] = [
        use(("c1", "search"), q="x"),
        use(("c2", "search"), q="x"),
        use(("c3", "search"), q="x"),
        use(("c4", "search"), ("c5", "search"), q="x"),
        use(("c6", "search"), q="x"),
        reply("done"),
    ]
    world = World(script=script)
    drv = driver(world, tmp_path, Rule(("c4",), Deny("barred")))
    paused = drv.run(REPEATING)
    assert paused.status == "paused"
    done = drv.resume(paused.ref, Decision(granted=True, by="me"))
    assert (done.status, done.reason) == ("finished", "end_turn")


@dataclass
class Spy:
    """Allows everything and keeps the state it was given for each call."""

    pre_seen: dict[str, RunState] = field(default_factory=dict)
    post_seen: dict[str, RunState] = field(default_factory=dict)

    def check(self, request: ToolRequest, state: RunState) -> PreDecision:
        self.pre_seen[request.tool_use_id] = state
        return Allow()


@dataclass
class PostSpy:
    spy: Spy

    def check(
        self, request: ToolRequest, result: ToolResult, state: RunState
    ) -> PostDecision:
        self.spy.post_seen[request.tool_use_id] = state
        return Allow()


def test_the_state_a_guard_sees_is_the_one_the_spec_names(tmp_path: Path) -> None:
    """Pre guards judge before the turn's escalation and `tool_requested` lines; post
    guards run after both but before any call of the turn has completed (spec 3.7)."""
    world = World(
        script=[
            use(("a", "search"), q="x"),
            use(
                ("b", "search"), ("c", "search"), q="x"
            ),  # b repeats a: the ladder trips
            reply("ok"),
        ],
        read_only={"search"},
    )
    spy = Spy()
    guards = Guards(pre=(spy,), post=(PostSpy(spy),))
    world.driver(tmp_path, guards=guards).run(REPEATING)
    before = spy.pre_seen["b"]
    assert before.rung == 0  # the escalation is written after the verdicts
    assert (
        dict(before.requested) == {}
    )  # a has completed; b and c are not requested yet
    assert before.pending_tools == ("b", "c")
    after = spy.post_seen["b"]
    assert after.rung == 1  # by the time the calls run, the ladder has spoken
    assert dict(after.requested) == {"b": "search", "c": "search"}
    assert after.pending_tools == ("b", "c")  # nothing of this turn has completed
    assert spy.post_seen["c"].pending_tools == ("b", "c")  # not even b, for c
