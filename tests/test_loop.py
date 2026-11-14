"""f08 and f09: a scripted model drives runs to finished, paused and resumed."""

import logging
from decimal import Decimal
from pathlib import Path

import pytest
from factories import make
from fakes import (
    CAPS,
    CONFIG,
    COST,
    Crash,
    ToolBugError,
    World,
    failure,
    finish,
    kinds,
    reply,
    use,
)
from fakes import ref as run_ref

from loomcore import ladder
from loomcore.conversation import FORK_CONTINUE, call_hash
from loomcore.engine import ResumeRefusedError
from loomcore.eventlog import open_writer, read_events
from loomcore.events import (
    Escalation,
    Event,
    ForkOrigin,
    JsonObject,
    NoteWritten,
    RunFinished,
    SandboxDenied,
    ToolCompleted,
)
from loomcore.limits import Limits
from loomcore.loop import (
    ConfigChangedError,
    Driver,
    RunIdTakenError,
    SpendLogMissingError,
)
from loomcore.ports import Config, Decision, RunResult, Task
from loomcore.rewind import ForkRefusedError
from loomcore.sandbox import Refusal
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


def test_a_client_error_finishes_the_run_with_error(tmp_path: Path) -> None:
    result = start(World(script=[failure(400)]), tmp_path)
    assert (result.status, result.reason) == ("finished", "error")
    assert events(tmp_path)[-1] == "run_finished"
    assert "turn_started" not in events(tmp_path)  # no response, no turn


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
    assert paused.question is not None
    assert "write (t1)" in paused.question
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
    assert "8 bytes" in caplog.text  # b'{"torn":' is what the crash left


def test_a_retry_pause_leaves_no_turn_behind_and_resume_continues(
    tmp_path: Path,
) -> None:
    world = World(script=[failure(429, retry_after=3600), reply("fine")])
    driver = world.driver(tmp_path)
    paused = driver.run(Task("go", limits(timeout=60, max_turns=1)))
    assert (paused.limit, "turn_started" in events(tmp_path)) == ("timeout", False)
    done = driver.resume(paused.ref, limits=limits(timeout=7200, max_turns=1))
    assert (done.status, done.result) == ("finished", "fine")
    assert events(tmp_path).count("turn_started") == 1  # max_turns=1 was enough


def test_a_new_run_id_the_session_already_uses_is_refused(tmp_path: Path) -> None:
    start(World(script=[reply()]), tmp_path)
    with pytest.raises(RunIdTakenError):
        start(World(script=[reply()]), tmp_path)  # a fresh World hands out r1 again


def test_run_reports_a_torn_line_another_run_left(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    start(World(script=[reply()]), tmp_path)
    with (tmp_path / "s1.jsonl").open("ab") as file:
        file.write(b'{"torn":')
    world = World(script=[reply()])
    world.ids = 1  # the next run is r2
    with caplog.at_level(logging.WARNING):
        world.driver(tmp_path).run(TASK)
    assert "8 bytes" in caplog.text


# --- f11: the ladder and finish ------------------------------------------------

SWITCHED = Config(model="m", system="try something else", tools=CONFIG.tools)
LADDER_CONFIG = Config(CONFIG.model, CONFIG.system, CONFIG.tools, switch_to=SWITCHED)
REPEATING = Task("go", limits(repeat_threshold=2))


def repeat(count: int) -> list[JsonObject | Exception]:
    """The same call, `count` times, under different tool_use ids."""
    return [use((f"c{n}", "search"), q="x") for n in range(1, count + 1)]


def all_events(tmp: Path) -> tuple[Event, ...]:
    return read_events(tmp / "s1.jsonl").events


def completions(tmp: Path) -> list[ToolCompleted]:
    return [e for e in all_events(tmp) if isinstance(e, ToolCompleted)]


def escalations(tmp: Path) -> list[Escalation]:
    return [e for e in all_events(tmp) if isinstance(e, Escalation)]


def test_finish_ends_the_run_with_its_result_and_is_not_run_as_a_tool(
    tmp_path: Path,
) -> None:
    world = World(script=[finish({"answer": 42})])
    result = start(world, tmp_path)
    assert (result.status, result.reason) == ("finished", "finished")
    assert result.result == {"answer": 42}
    done = completions(tmp_path)
    assert [(e.tool_use_id, e.ok) for e in done] == [("fin", True)]
    assert world.tool_calls == []


def test_a_finish_with_no_result_ends_with_that_not_the_last_words(
    tmp_path: Path,
) -> None:
    world = World(script=[finish(None)])
    result = start(world, tmp_path)
    assert (result.reason, result.result) == ("finished", None)
    assert [e.result for e in all_events(tmp_path) if isinstance(e, RunFinished)] == [
        None
    ]


def test_the_calls_beside_finish_are_settled_without_running(tmp_path: Path) -> None:
    world = World(script=[finish("done", ("a", "search"))])
    result = start(world, tmp_path)
    assert result.reason == "finished"
    ok = {
        e.tool_use_id: e.ok for e in all_events(tmp_path) if e.kind == "tool_completed"
    }
    assert ok == {"fin": True, "a": False}
    assert world.tool_calls == []


def test_the_ladder_climbs_from_a_nudge_to_a_switch_to_an_ask_to_stuck(
    tmp_path: Path,
) -> None:
    world = World(script=repeat(6))
    driver = world.driver(tmp_path, LADDER_CONFIG)
    paused = driver.run(REPEATING)
    assert (paused.status, paused.reason) == ("paused", "awaiting_approval")
    assert paused.question is not None
    assert world.tool_calls == ["x", "x", "x"]  # the asked call has not run
    done = driver.resume(paused.ref, Decision(granted=True, by="me"))
    assert (done.status, done.reason) == ("finished", "stuck")
    assert [e.rung for e in escalations(tmp_path)] == [1, 2, 3, 4]
    completed = completions(tmp_path)
    assert len(completed) == 6  # every tool_use has exactly one completion
    assert [e.ok for e in completed] == [True] * 5 + [False]


def test_a_nudge_and_a_switch_reach_the_model_through_the_tool_results(
    tmp_path: Path,
) -> None:
    world = World(script=repeat(6))
    driver = world.driver(tmp_path, LADDER_CONFIG)
    driver.resume(driver.run(REPEATING).ref, Decision(granted=True, by="me"))
    sent = [str(r.messages[-1]["content"]) for r in world.requests]
    assert ladder.NUDGE in sent[2]  # c2 tripped: the result after it carries it
    assert ladder.NUDGE not in sent[1]  # c1 did not trip
    assert ladder.SWITCH in sent[3]
    assert [r.system for r in world.requests[:4]] == [
        "be brief",
        "be brief",
        "be brief",
        "try something else",  # in force from the turn after the switch
    ]


def test_a_denied_ladder_ask_fails_that_call_and_the_next_trip_is_stuck(
    tmp_path: Path,
) -> None:
    world = World(script=repeat(6))
    driver = world.driver(tmp_path, LADDER_CONFIG)
    paused = driver.run(REPEATING)
    done = driver.resume(paused.ref, Decision(granted=False, by="me", note="no"))
    assert (done.status, done.reason) == ("finished", "stuck")
    assert world.tool_calls == ["x", "x", "x"]
    why = [e.content for e in completions(tmp_path)]
    assert "denied by me: no" in why


def test_a_grant_logged_before_a_crash_does_not_license_a_second_run(
    tmp_path: Path,
) -> None:
    world = World(script=repeat(6), crash_on_call=4)
    driver = world.driver(tmp_path, LADDER_CONFIG)
    paused = driver.run(REPEATING)
    with pytest.raises(Crash):  # the granted call c4 is the 4th tool call
        driver.resume(paused.ref, Decision(granted=True, by="me"))
    with pytest.raises(ConfigChangedError):  # the switch is part of the log
        world.driver(tmp_path).resume(paused.ref)
    again = driver.resume(paused.ref)  # c4 may have run: that is a new question
    assert (again.status, again.reason) == ("paused", "awaiting_approval")
    assert again.question is not None
    assert "may have run before the crash" in again.question
    done = driver.resume(again.ref, Decision(granted=True, by="me"))
    assert (done.status, done.reason) == ("finished", "stuck")
    assert events(tmp_path).count("approval_requested") == 2


def test_a_fork_from_a_final_answer_with_no_hint_asks_the_model_to_carry_on(
    tmp_path: Path,
) -> None:
    world = World(script=[reply("first"), reply("second")])
    driver = world.driver(tmp_path)
    driver.run(TASK)
    call = next(e for e in all_events(tmp_path) if e.kind == "model_call")
    driver.fork(ForkOrigin(run_id="r1", seq=call.seq))
    last = world.requests[-1].messages[-1]
    assert (last["role"], last["content"]) == ("user", FORK_CONTINUE)


def test_resume_on_a_session_that_does_not_exist_creates_nothing(
    tmp_path: Path,
) -> None:
    with pytest.raises(ResumeRefusedError, match="no session file"):
        World().driver(tmp_path).resume(run_ref())
    with pytest.raises(ResumeRefusedError, match="no session file"):
        World().driver(tmp_path).fork(ForkOrigin(run_id="r1", seq=0))
    assert not list(tmp_path.glob("s1.*"))


def test_a_price_that_cannot_be_found_leaves_no_turn_behind(tmp_path: Path) -> None:
    world = World(script=[reply()])
    world.pricing.fail = True
    with pytest.raises(LookupError):
        start(world, tmp_path)
    assert events(tmp_path) == ["run_started"]


def test_a_crash_partway_through_the_batch_judges_the_whole_turn_on_resume(
    tmp_path: Path,
) -> None:
    """The turn [a, b] repeats one call; only a's tool_requested line was written."""
    args: JsonObject = {"q": "x"}
    digest = call_hash("search", args)
    log = [
        make(
            "run_started",
            0,
            task="go",
            config=CONFIG.to_json(),
            limits=limits(repeat_threshold=2).to_json(),
            config_digest=CONFIG.digest(),
        ),
        make("turn_started", 1, turn=1),
        make("model_call", 2, message=use(("a", "search"), ("b", "search"), q="x")),
        make(
            "tool_requested",
            3,
            tool_use_id="a",
            name="search",
            input=args,
            call_hash=digest,
        ),
    ]
    with open_writer(tmp_path / "s1.jsonl") as writer:
        for event in log:
            writer.append(event)
    world = World(script=[reply("done")])
    world.read_only.add("search")
    result = world.driver(tmp_path).resume(run_ref())
    assert (result.status, result.result) == ("finished", "done")
    [escalated] = escalations(tmp_path)
    assert (escalated.rung, escalated.tripped) == (1, ["b"])  # b repeats a
    resumed = events(tmp_path)[4:8]
    assert resumed == ["run_resumed", "escalation", "tool_requested", "tool_completed"]
    nudged = {
        e.tool_use_id: ladder.NUDGE in str(e.content) for e in completions(tmp_path)
    }
    assert nudged == {"a": False, "b": True}


def test_an_open_approval_question_is_not_repeated_after_a_crash(
    tmp_path: Path,
) -> None:
    """The crash came after approval_requested and before run_paused."""
    args: JsonObject = {"q": "a"}
    asked = "run again?; may have run before the crash: write (a)"
    log = [
        make(
            "run_started",
            0,
            task="go",
            config=CONFIG.to_json(),
            limits=limits().to_json(),
            config_digest=CONFIG.digest(),
        ),
        make("turn_started", 1, turn=1),
        make("model_call", 2, message=use(("a", "write"))),
        make(
            "tool_requested",
            3,
            tool_use_id="a",
            name="write",
            input=args,
            call_hash=call_hash("write", args),
        ),
        make("approval_requested", 4, tool_use_id="a", question=asked),
    ]
    with open_writer(tmp_path / "s1.jsonl") as writer:
        for event in log:
            writer.append(event)
    paused = World(script=[]).driver(tmp_path).resume(run_ref())
    assert (paused.status, paused.reason) == ("paused", "awaiting_approval")
    assert paused.question == asked  # not "... (may have run before the crash: ...)"


# --- f14: what a call caused is logged with the call's id ---------------------


def test_a_refusal_and_a_note_are_logged_before_their_call_completes(
    tmp_path: Path,
) -> None:
    world = World(
        script=[use(("t1", "search")), reply()],
        refused_in={"t1": Refusal("read", "/etc", "no")},
        noted_in={"t1": "notes.md"},
    )
    start(world, tmp_path)
    log = read_events(tmp_path / "s1.jsonl").events
    assert [e.kind for e in log if e.run_id == "r1"][3:7] == [
        "tool_requested",
        "sandbox_denied",
        "note_written",
        "tool_completed",
    ]
    denied = next(e for e in log if isinstance(e, SandboxDenied))
    noted = next(e for e in log if isinstance(e, NoteWritten))
    assert (denied.tool_use_id, denied.target) == ("t1", "/etc")
    assert (noted.tool_use_id, noted.path) == ("t1", "notes.md")


def test_a_tool_that_breaks_still_has_its_notices_logged_and_its_bug_raised(
    tmp_path: Path,
) -> None:
    world = World(
        script=[use(("t1", "search")), reply()],
        noted_in={"t1": "notes.md"},
        bug_in={"t1"},
    )
    with pytest.raises(ToolBugError):
        start(world, tmp_path)
    log = read_events(tmp_path / "s1.jsonl").events
    noted = next(e for e in log if isinstance(e, NoteWritten))
    assert noted.tool_use_id == "t1"
    assert not any(isinstance(e, ToolCompleted) for e in log)
