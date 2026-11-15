"""f19, the post stage: a guard redacts or denies a result before it is logged."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from factories import make
from fakes import CAPS, World, completed, kinds, reply, run_started, use, write_log
from fakes import ref as run_ref

from loomcore import ladder
from loomcore.conversation import call_hash
from loomcore.eventlog import read_events
from loomcore.events import Event, GuardDecision, JsonObject
from loomcore.guards import Allow, Deny, Guards, PostDecision, Redact, ToolRequest
from loomcore.limits import Limits
from loomcore.ports import Task, ToolResult
from loomcore.replay import RunState

TASK = Task("go", CAPS)
SECRET = "sk-live-4242"
LEAKY = ToolResult(True, f"token={SECRET} ok")


@dataclass(frozen=True)
class Scrub:
    """Redacts the secret from any result that holds it."""

    def check(
        self, request: ToolRequest, result: ToolResult, state: RunState
    ) -> PostDecision:
        if SECRET in str(result.content):
            return Redact(str(result.content).replace(SECRET, "[redacted]"), "a key")
        return Allow()


@dataclass(frozen=True)
class Refuse:
    ids: tuple[str, ...]

    def check(
        self, request: ToolRequest, result: ToolResult, state: RunState
    ) -> PostDecision:
        return Deny("too large") if request.tool_use_id in self.ids else Allow()


def run(world: World, tmp: Path, *guards: Scrub | Refuse, task: Task = TASK) -> None:
    world.driver(tmp, guards=Guards(post=guards)).run(task)


def events(tmp: Path) -> tuple[Event, ...]:
    return read_events(tmp / "s1.jsonl").events


def test_a_redacted_secret_is_in_neither_the_log_nor_a_request(tmp_path: Path) -> None:
    world = World(script=[use(("a", "read")), reply("ok")])
    world.tool_results["a"] = LEAKY
    run(world, tmp_path, Scrub())
    assert SECRET.encode() not in (tmp_path / "s1.jsonl").read_bytes()
    assert SECRET not in str([r.messages for r in world.requests])
    assert completed(tmp_path)["a"].content == "token=[redacted] ok"
    log = kinds(events(tmp_path))
    assert log.index("guard_decision") < log.index("tool_completed")
    [decision] = [e for e in events(tmp_path) if isinstance(e, GuardDecision)]
    assert (decision.stage, decision.decision, decision.reason) == (
        "post",
        "redact",
        "a key",
    )


def test_an_allowed_result_leaves_no_line(tmp_path: Path) -> None:
    world = World(script=[use(("a", "read")), reply("ok")])
    run(world, tmp_path, Scrub())
    assert "guard_decision" not in kinds(events(tmp_path))
    assert completed(tmp_path)["a"].content == "result a"


def test_a_post_deny_replaces_the_result_with_its_reason(tmp_path: Path) -> None:
    world = World(script=[use(("a", "read")), reply("ok")])
    world.tool_results["a"] = ToolResult(True, "huge", cached=True)
    run(world, tmp_path, Refuse(("a",)))
    done = completed(tmp_path)["a"]
    assert (done.ok, done.content, done.cached) == (False, "too large", True)
    [decision] = [e for e in events(tmp_path) if isinstance(e, GuardDecision)]
    assert (decision.stage, decision.decision, decision.reason) == (
        "post",
        "deny",
        "too large",
    )
    log = kinds(events(tmp_path))
    assert log.index("guard_decision") < log.index("tool_completed")
    assert "too large" in str(world.requests[-1].messages[-1]["content"])


def test_a_result_flagged_cached_is_redacted_like_any_other(tmp_path: Path) -> None:
    world = World(script=[use(("a", "read")), use(("b", "read")), reply("ok")])
    world.tool_results["a"] = ToolResult(True, f"k={SECRET}", cached=True)
    world.tool_results["b"] = ToolResult(True, f"k={SECRET}", cached=True)
    run(world, tmp_path, Scrub())
    assert [e.content for e in completed(tmp_path).values()] == ["k=[redacted]"] * 2
    assert SECRET.encode() not in (tmp_path / "s1.jsonl").read_bytes()


def test_the_calls_of_a_wave_are_each_screened_before_their_completion(
    tmp_path: Path,
) -> None:
    world = World(script=[use(("a", "read"), ("b", "read")), reply("ok")])
    world.together = {"read"}
    world.tool_results["a"] = LEAKY
    world.tool_results["b"] = LEAKY
    run(world, tmp_path, Scrub())
    assert SECRET.encode() not in (tmp_path / "s1.jsonl").read_bytes()
    log = [
        (e.kind, getattr(e, "tool_use_id", ""))
        for e in events(tmp_path)
        if e.kind in ("guard_decision", "tool_completed")
    ]
    assert log == [
        ("guard_decision", "a"),
        ("tool_completed", "a"),
        ("guard_decision", "b"),
        ("tool_completed", "b"),
    ]


def test_the_ladders_note_is_added_after_the_guards(tmp_path: Path) -> None:
    script: list[JsonObject | Exception] = [
        use(("c1", "search"), q="x"),
        use(("c2", "search"), q="x"),
        reply("ok"),
    ]
    world = World(script=script)
    world.tool_results["x"] = LEAKY  # the fake keys results by `q`
    task = Task(
        "go", Limits.model_validate({**CAPS.model_dump(), "repeat_threshold": 2})
    )
    run(world, tmp_path, Scrub(), task=task)
    nudged = completed(tmp_path)["c2"].content
    assert isinstance(nudged, str)
    assert "token=[redacted] ok" in nudged
    assert ladder.NUDGE in nudged
    assert SECRET.encode() not in (tmp_path / "s1.jsonl").read_bytes()


class GuardBugError(Exception):
    pass


@dataclass(frozen=True)
class Breaks:
    id: str

    def check(
        self, request: ToolRequest, result: ToolResult, state: RunState
    ) -> PostDecision:
        if request.tool_use_id == self.id:
            raise GuardBugError(self.id)
        return Allow()


def test_a_guard_that_raises_in_a_wave_does_not_lose_the_other_calls(
    tmp_path: Path,
) -> None:
    world = World(script=[use(("a", "read"), ("b", "read"), ("c", "read"))])
    world.together = {"read"}
    world.noted_in["c"] = "notes.md"
    with pytest.raises(GuardBugError):
        world.driver(tmp_path, guards=Guards(post=(Breaks("b"),))).run(TASK)
    done = set(completed(tmp_path))
    assert done == {"a", "c"}  # b was interrupted; the others are logged
    assert "note_written" in kinds(events(tmp_path))  # c's notice, after b's failure


def test_a_guard_that_raises_on_a_single_call_logs_its_notices_and_leaves_it_open(
    tmp_path: Path,
) -> None:
    world = World(script=[use(("a", "read")), reply("ok")])
    world.noted_in["a"] = "notes.md"
    world.read_only.add("read")
    with pytest.raises(GuardBugError):
        world.driver(tmp_path, guards=Guards(post=(Breaks("a"),))).run(TASK)
    assert "note_written" in kinds(events(tmp_path))  # what the call caused is logged
    assert "a" not in completed(tmp_path)  # but the call is left interrupted
    result = world.driver(tmp_path).resume(run_ref())  # the guard is fixed: a re-runs
    assert (result.status, result.result) == ("finished", "ok")
    assert completed(tmp_path)["a"].ok


def test_a_redact_then_a_deny_log_both_decisions_and_the_denial_stands(
    tmp_path: Path,
) -> None:
    world = World(script=[use(("a", "read")), reply("ok")])
    world.tool_results["a"] = LEAKY
    run(world, tmp_path, Scrub(), Refuse(("a",)))
    decisions = [e for e in events(tmp_path) if isinstance(e, GuardDecision)]
    assert [(d.stage, d.decision) for d in decisions] == [
        ("post", "redact"),
        ("post", "deny"),
    ]
    done = completed(tmp_path)["a"]
    assert (done.ok, done.content) == (False, "too large")
    assert SECRET.encode() not in (tmp_path / "s1.jsonl").read_bytes()


def crashed_after_a_decision(tmp: Path, tool: str) -> None:
    """The process died after the post guard's decision line, before tool_completed."""
    args: JsonObject = {"q": "a"}
    write_log(
        tmp,
        [
            run_started(),
            make("turn_started", 1, turn=1),
            make("model_call", 2, message=use(("a", tool))),
            make(
                "tool_requested",
                3,
                tool_use_id="a",
                name=tool,
                input=args,
                call_hash=call_hash(tool, args),
            ),
            make(
                "guard_decision",
                4,
                stage="post",
                tool_use_id="a",
                decision="redact",
                reason="a key",
            ),
        ],
    )


def test_a_crash_after_a_post_decision_reruns_a_read_and_writes_it_again(
    tmp_path: Path,
) -> None:
    crashed_after_a_decision(tmp_path, "read")
    world = World(script=[reply("ok")])
    world.tool_results["a"] = LEAKY
    world.read_only.add("read")
    result = world.driver(tmp_path, guards=Guards(post=(Scrub(),))).resume(run_ref())
    assert result.status == "finished"
    assert world.tool_calls == ["a"]  # interrupted, so it ran again
    decisions = [e for e in events(tmp_path) if isinstance(e, GuardDecision)]
    assert len(decisions) == 2  # the one the crash left, and the re-run's
    assert completed(tmp_path)["a"].content == "token=[redacted] ok"
    assert SECRET.encode() not in (tmp_path / "s1.jsonl").read_bytes()


def test_a_crash_after_a_post_decision_asks_before_a_write_runs_again(
    tmp_path: Path,
) -> None:
    crashed_after_a_decision(tmp_path, "write")
    world = World(script=[])
    paused = world.driver(tmp_path, guards=Guards(post=(Scrub(),))).resume(run_ref())
    assert (paused.status, paused.reason) == ("paused", "awaiting_approval")
    assert world.tool_calls == []
    assert "a" not in completed(tmp_path)
