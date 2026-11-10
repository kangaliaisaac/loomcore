"""f08 and f09: a scripted model drives runs to finished, paused and resumed."""

import logging
from decimal import Decimal
from pathlib import Path

import pytest
from fakes import CAPS, CONFIG, COST, Crash, World, failure, kinds, reply, use
from fakes import ref as run_ref

from loomcore.engine import ResumeRefusedError
from loomcore.eventlog import read_events
from loomcore.events import ForkOrigin, JsonObject
from loomcore.limits import Limits
from loomcore.loop import ConfigChangedError, Driver, SpendLogMissingError
from loomcore.ports import Config, Decision, RunResult, Task
from loomcore.rewind import ForkRefusedError
from loomcore.spend import SpendLine, append_spend, read_spend

TASK = Task("find it", CAPS)


def events(tmp: Path, run_id: str = "r1") -> list[str]:
    return kinds(read_events(tmp / "s1.jsonl").events, run_id)


def start(world: World, tmp_path: Path, task: Task = TASK) -> RunResult:
    return world.driver(tmp_path).run(task)


def limits(**changes: object) -> Limits:
    return Limits.model_validate({"run_cap": None, "month_cap": None, **changes})


# --- f08: running to the end --------------------------------------------------


def test_a_scripted_model_drives_a_run_to_finished(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "search")), reply("all done")])
    result = start(world, tmp_path)
    assert (result.status, result.reason, result.result) == (
        "finished",
        "end_turn",
        "all done",
    )
    assert events(tmp_path) == [
        "run_started",
        "turn_started",
        "model_call",
        "tool_requested",
        "tool_completed",
        "turn_started",
        "model_call",
        "run_finished",
    ]


def test_tool_results_go_back_to_the_model(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "search"), ("t2", "search")), reply()])
    start(world, tmp_path)
    last = world.requests[-1].messages
    assert [m["role"] for m in last] == ["user", "assistant", "user"]
    results = last[-1]["content"]
    assert isinstance(results, list)
    ids = [r["tool_use_id"] for r in results if isinstance(r, dict)]
    assert ids == ["t1", "t2"]
    assert world.tool_calls == ["t1", "t2"]


def test_every_model_call_leaves_a_spend_line(tmp_path: Path) -> None:
    start(World(script=[use(("t1", "search")), reply()]), tmp_path)
    lines = read_spend(tmp_path / "spend.jsonl").lines
    assert [(line.run, line.seq, line.usd) for line in lines] == [
        ("s1/r1", 2, COST),
        ("s1/r1", 6, COST),
    ]


@pytest.mark.parametrize(
    ("last", "reason"),
    [
        (reply("no", "refusal"), "refused"),
        (reply("?", "novel"), "error"),
        (reply("x", "tool_use"), "error"),
    ],
)
def test_the_other_ways_a_run_ends(
    tmp_path: Path, last: JsonObject, reason: str
) -> None:
    result = start(World(script=[last]), tmp_path)
    assert (result.status, result.reason) == ("finished", reason)


def test_pending_calls_are_completed_before_the_run_finishes(tmp_path: Path) -> None:
    world = World(script=[failure(400)])
    result = start(world, tmp_path)
    assert result.reason == "error"
    assert events(tmp_path)[-1] == "run_finished"


# --- f09: limits pause and resume ---------------------------------------------


def test_max_turns_pauses_by_name_and_a_resume_with_limits_continues(
    tmp_path: Path,
) -> None:
    world = World(script=[use(("t1", "search")), use(("t2", "search")), reply()])
    driver = world.driver(tmp_path)
    paused = driver.run(Task("go", limits(max_turns=2)))
    assert (paused.status, paused.reason, paused.limit) == (
        "paused",
        "limit",
        "max_turns",
    )
    assert world.tool_calls == ["t1", "t2"]
    again = driver.resume(paused.ref, limits=limits(max_turns=2))
    assert again.limit == "max_turns"  # not raised: the same limit reached again
    done = driver.resume(paused.ref, limits=limits(max_turns=3))
    assert (done.status, done.reason) == ("finished", "end_turn")
    kinds_ = events(tmp_path)
    assert kinds_.count("run_paused") == 2
    assert kinds_.count("turn_started") == 3  # the turn count carried over


def test_run_cap_pauses_before_the_call_that_would_cross_it(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "search")), reply()])
    result = world.driver(tmp_path).run(Task("go", limits(run_cap=0.015)))
    assert (result.reason, result.limit) == ("limit", "run_cap")
    assert len(world.requests) == 1


def test_month_cap_counts_what_other_runs_spent(tmp_path: Path) -> None:
    other = SpendLine(at=World().clock.now, usd=Decimal(100), run="s9/x", seq=0)
    append_spend(tmp_path / "spend.jsonl", other)
    world = World(script=[reply()])
    result = world.driver(tmp_path).run(Task("go", limits(month_cap=100)))
    assert (result.reason, result.limit) == ("limit", "month_cap")
    assert world.requests == []


def test_timeout_pauses_on_active_time(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "search")), reply()], tool_seconds=61)
    result = world.driver(tmp_path).run(Task("go", limits(timeout=60)))
    assert (result.reason, result.limit) == ("limit", "timeout")


def test_two_truncated_calls_pause_on_max_output_tokens_then_resume_fresh(
    tmp_path: Path,
) -> None:
    cut = reply("part", "max_tokens")
    world = World(script=[cut, cut, reply("whole")])
    driver = world.driver(tmp_path)
    paused = driver.run(TASK)
    assert paused.limit == "max_output_tokens"
    assert "Carry on" in str(world.requests[1].messages[-1]["content"])
    done = driver.resume(paused.ref, limits=CAPS)
    assert (done.status, done.result) == ("finished", "whole")


# --- f08: resume --------------------------------------------------------------


def test_a_crash_reruns_a_read_only_tool(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "search")), reply()], crash_in={"t1"})
    world.read_only.add("search")
    driver = world.driver(tmp_path)
    with pytest.raises(Crash):
        driver.run(TASK)
    result = driver.resume(run_ref())
    assert (result.status, world.tool_calls) == ("finished", ["t1"])
    assert "run_resumed" in events(tmp_path)


def test_a_crash_asks_before_rerunning_a_tool_with_effects(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "write")), reply("ok")], crash_in={"t1"})
    driver = world.driver(tmp_path)
    with pytest.raises(Crash):
        driver.run(TASK)
    paused = driver.resume(run_ref())
    assert (paused.status, paused.reason) == ("paused", "awaiting_approval")
    assert world.tool_calls == []
    with pytest.raises(ResumeRefusedError, match="decision"):
        driver.resume(paused.ref)
    done = driver.resume(paused.ref, Decision(granted=True, by="me"))
    assert (done.status, world.tool_calls) == ("finished", ["t1"])
    log = events(tmp_path)
    resumed = len(log) - 1 - log[::-1].index("run_resumed")
    assert log[resumed : resumed + 3] == [
        "run_resumed",
        "approval_granted",
        "tool_completed",
    ]


def test_a_denied_call_fails_and_the_loop_goes_on(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "write")), reply("ok")], crash_in={"t1"})
    driver = world.driver(tmp_path)
    with pytest.raises(Crash):
        driver.run(TASK)
    paused = driver.resume(run_ref())
    done = driver.resume(paused.ref, Decision(granted=False, by="me", note="no"))
    assert done.status == "finished"
    assert world.tool_calls == []
    sent = world.requests[-1].messages[-1]["content"]
    assert "denied by me: no" in str(sent)
    assert "'is_error': True" in str(sent)


def test_resume_refuses_what_it_cannot_continue(tmp_path: Path) -> None:
    world = World(script=[reply()])
    driver = world.driver(tmp_path)
    finished = driver.run(TASK)
    with pytest.raises(ResumeRefusedError, match="finished"):
        driver.resume(finished.ref)
    with pytest.raises(ResumeRefusedError, match="no run"):
        driver.resume(run_ref("nope"))


def test_a_decision_on_a_run_that_is_not_paused_is_refused(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "search"))], crash_in={"t1"})
    driver = world.driver(tmp_path)
    with pytest.raises(Crash):
        driver.run(TASK)
    with pytest.raises(ResumeRefusedError, match="only a paused run"):
        driver.resume(run_ref(), Decision(granted=True, by="me"))


def test_resume_refuses_a_changed_configuration(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "search"))], crash_in={"t1"})
    with pytest.raises(Crash):
        world.driver(tmp_path).run(TASK)
    changed = Config(model="m", system="be long", tools=CONFIG.tools)
    with pytest.raises(ConfigChangedError):
        world.driver(tmp_path, changed).resume(run_ref())


def test_a_crash_then_a_late_resume_does_not_time_out(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "search")), reply()], crash_in={"t1"})
    world.read_only.add("search")
    driver = world.driver(tmp_path)
    with pytest.raises(Crash):
        driver.run(Task("go", limits(timeout=60)))
    world.clock.advance(3 * 3600)
    assert driver.resume(run_ref()).status == "finished"


def test_the_approval_wait_does_not_count_against_the_timeout(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "write")), reply()], crash_in={"t1"})
    driver = world.driver(tmp_path)
    with pytest.raises(Crash):
        driver.run(Task("go", limits(timeout=60)))
    paused = driver.resume(run_ref())
    world.clock.advance(3600)
    assert driver.resume(paused.ref, Decision(True, "me")).status == "finished"


# --- f08: the spend log -------------------------------------------------------


def test_resume_repairs_the_spend_log_and_reports_lines_it_cannot_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    world = World(script=[use(("t1", "search")), reply()], crash_in={"t1"})
    world.read_only.add("search")
    driver = world.driver(tmp_path)
    with pytest.raises(Crash):
        driver.run(TASK)
    spend = tmp_path / "spend.jsonl"
    spend.write_text("not json\n")  # the first call's line is lost, one line is junk
    with caplog.at_level(logging.WARNING):
        driver.resume(run_ref())
    assert "not json" in caplog.text
    assert [line.seq for line in read_spend(spend).lines] == [2, 7]


def test_the_spend_log_path_is_logged_and_a_missing_directory_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        World().driver(tmp_path)
    assert str(tmp_path / "spend.jsonl") in caplog.text
    with pytest.raises(SpendLogMissingError):
        Driver(
            CONFIG, World().ports(), tmp_path / "s1.jsonl", tmp_path / "no" / "s.jsonl"
        )


# --- f08: forks ---------------------------------------------------------------


def test_a_fork_checks_the_point_and_then_writes_run_forked(tmp_path: Path) -> None:
    world = World(script=[use(("t1", "search")), reply("first"), reply("second")])
    driver = world.driver(tmp_path)
    driver.run(TASK)
    log = read_events(tmp_path / "s1.jsonl").events
    clean = next(e for e in log if e.kind == "tool_completed")
    forked = driver.fork(ForkOrigin(run_id="r1", seq=clean.seq), hint="try again")
    assert (forked.ref.run_id, forked.result) == ("r2", "second")
    assert events(tmp_path, "r2")[:3] == ["run_forked", "turn_started", "model_call"]
    sent = world.requests[-1].messages
    assert sent[-1] == {"role": "user", "content": "try again"}


def test_a_fork_at_an_unclean_point_is_refused_and_writes_nothing(
    tmp_path: Path,
) -> None:
    world = World(script=[use(("t1", "search")), reply()])
    driver = world.driver(tmp_path)
    driver.run(TASK)
    before = read_events(tmp_path / "s1.jsonl").events
    asked = next(e for e in before if e.kind == "model_call")
    with pytest.raises(ForkRefusedError):
        driver.fork(ForkOrigin(run_id="r1", seq=asked.seq))
    assert read_events(tmp_path / "s1.jsonl").events == before


# --- f10 in the driver: retries -----------------------------------------------


def test_a_rate_limit_is_retried_after_a_backoff(tmp_path: Path) -> None:
    world = World(script=[failure(429), failure(529), reply()])
    result = start(world, tmp_path)
    assert result.status == "finished"
    assert world.sleeps == [0.25, 0.5]  # random() is 0.5: half of base * 2**attempt


def test_a_client_error_is_not_retried(tmp_path: Path) -> None:
    world = World(script=[failure(400), reply()])
    result = start(world, tmp_path)
    assert (result.reason, world.sleeps) == ("error", [])


def test_a_retry_that_would_outlast_the_timeout_pauses_instead_of_sleeping(
    tmp_path: Path,
) -> None:
    world = World(script=[failure(429, retry_after=3600), reply()])
    result = world.driver(tmp_path).run(Task("go", limits(timeout=60)))
    assert (result.reason, result.limit) == ("limit", "timeout")
    assert world.sleeps == []


def test_a_torn_final_line_is_cut_and_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    world = World(script=[use(("t1", "search")), reply()], crash_in={"t1"})
    world.read_only.add("search")
    driver = world.driver(tmp_path)
    with pytest.raises(Crash):
        driver.run(TASK)
    with (tmp_path / "s1.jsonl").open("ab") as file:
        file.write(b'{"torn":')
    with caplog.at_level(logging.WARNING):
        driver.resume(run_ref())
    assert "torn" in caplog.text
