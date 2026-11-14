"""f16, part 2: the driver builds its window through the context store."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fakes import CAPS, CONFIG, Crash, World, fake_count, reply, use
from fakes import ref as run_ref

from loomcore.conversation import build_messages
from loomcore.eventlog import read_events
from loomcore.events import (
    ContextCompacted,
    ContextPacked,
    Event,
    JsonObject,
    TurnStarted,
)
from loomcore.loop import Driver
from loomcore.ports import Config, Task, ToolResult
from loomcore.replay import replay, run_events
from loomcore.window import Budget, Carried, build_window

TURNS = 14
BUDGET = 700


def long_run(**limits: object) -> tuple[World, Task]:
    """Fourteen turns, each a tool call with a long result, then an answer."""
    ids = [f"t{k}" for k in range(TURNS)]
    results = {i: ToolResult(True, f"{i} " + "data " * 80) for i in ids}
    script: list[JsonObject | Exception] = [use((i, "search")) for i in ids]
    world = World(
        script=[*script, reply("done")],
        tool_results=results,
        read_only={"search"},
    )
    caps = {"run_cap": None, "month_cap": None, "max_turns": 100, **limits}
    return world, Task("go", CAPS.model_validate(caps))


def finish(world: World, task: Task, tmp: Path, config: Config = CONFIG) -> list[Event]:
    assert world.driver(tmp, config).run(task).status == "finished"
    return list(read_events(tmp / "s1.jsonl").events)


def context_events(log: list[Event]) -> list[Event]:
    return [e for e in log if isinstance(e, (ContextPacked, ContextCompacted))]


def turn_starts(log: list[Event]) -> list[int]:
    return [k for k, e in enumerate(log) if isinstance(e, TurnStarted)]


def test_with_no_budget_nothing_is_counted_and_requests_are_the_whole_history(
    tmp_path: Path,
) -> None:
    world, task = long_run()

    def refuse(messages: object) -> int:
        raise AssertionError("with no budget nothing is counted")

    ports = replace(world.ports(), count_tokens=refuse)
    driver = Driver(CONFIG, ports, tmp_path / "s1.jsonl", tmp_path / "spend.jsonl")
    assert driver.run(task).status == "finished"
    log = list(read_events(tmp_path / "s1.jsonl").events)
    assert context_events(log) == []
    for request, start in zip(world.requests, turn_starts(log), strict=True):
        assert request.messages == build_messages("go", log[:start])


def test_a_budget_records_what_it_drops_and_clears_and_the_run_finishes(
    tmp_path: Path,
) -> None:
    world, task = long_run(context_budget=BUDGET)
    log = finish(world, task, tmp_path)
    packed = [e for e in log if isinstance(e, ContextPacked)]
    compacted = [e for e in log if isinstance(e, ContextCompacted)]
    assert packed
    assert compacted  # fourteen long results do not fit in 700 tokens
    for dropping in packed:
        assert all(str(i).startswith("seq:r1:") for i in dropping.dropped)
        assert dropping.tokens <= dropping.budget
    for clearing in compacted:
        assert clearing.strategy == "clear_tool_results"
        assert clearing.tokens_after < clearing.tokens_before
        assert clearing.cleared
    state = replay(log)
    assert list(state.dropped) == [i for e in packed for i in e.dropped]
    assert list(state.cleared) == [i for e in compacted for i in e.cleared]


def test_the_task_and_the_latest_exchange_are_in_every_request(tmp_path: Path) -> None:
    world, task = long_run(context_budget=BUDGET)
    log = finish(world, task, tmp_path)
    for k, (request, start) in enumerate(
        zip(world.requests, turn_starts(log), strict=True)
    ):
        assert request.messages[0] == {"role": "user", "content": "go"}
        whole = build_messages("go", log[:start])
        if k:
            assert request.messages[-2:] == whole[-2:]  # never cleared, never dropped


def test_between_two_context_events_each_request_extends_the_one_before(
    tmp_path: Path,
) -> None:
    world, task = long_run(context_budget=BUDGET)
    log = finish(world, task, tmp_path)
    changed = []
    for k, start in enumerate(turn_starts(log)):
        before = turn_starts(log)[k - 1] if k else 0
        changed.append(bool(context_events(log[before:start])))
    assert any(changed[1:])
    assert not all(changed[1:])  # the window is not rewritten every turn
    for k in range(1, len(changed)):
        if not changed[k]:
            earlier, later = world.requests[k - 1].messages, world.requests[k].messages
            assert later[: len(earlier)] == earlier


@pytest.mark.parametrize("crash", [False, True])
def test_the_window_that_was_sent_can_be_rebuilt_from_the_log(
    tmp_path: Path, crash: bool
) -> None:
    world, task = long_run(context_budget=BUDGET)
    if crash:
        world.crash_in = {"t6"}
        driver = world.driver(tmp_path)
        with pytest.raises(Crash):
            driver.run(task)
        assert driver.resume(run_ref()).status == "finished"
    else:
        finish(world, task, tmp_path)
    log = list(read_events(tmp_path / "s1.jsonl").events)
    fixed = fake_count([{"role": "user", "content": CONFIG.system}]) + fake_count(
        [{"role": "user", "content": json.dumps(list(CONFIG.tools), sort_keys=True)}]
    )
    budget = Budget(BUDGET - fixed, task.limits.compact_at, task.limits.compact_to)
    starts = turn_starts(log)
    assert len(world.requests) == len(starts)
    for request, start in zip(world.requests, starts, strict=True):
        chain = run_events(log[:start], "r1")
        state = replay(chain)
        carried = Carried(state.dropped, state.cleared)
        again = build_window("go", chain, carried, budget, fake_count)
        assert again.messages == request.messages
        assert (again.newly_dropped, again.newly_cleared) == ((), ())


def test_the_cache_prefix_of_the_configuration_reaches_the_request(
    tmp_path: Path,
) -> None:
    marked = Config(model="m", system="be brief", tools=CONFIG.tools, cache_prefix=4)
    world, task = long_run()
    (tmp_path / "marked").mkdir()
    finish(world, task, tmp_path / "marked", marked)
    assert {r.cache_prefix for r in world.requests} == {4}
    plain, plain_task = long_run()
    (tmp_path / "plain").mkdir()
    finish(plain, plain_task, tmp_path / "plain")
    assert {r.cache_prefix for r in plain.requests} == {None}


def fixed_tokens() -> int:
    """What the driver counts for the system prompt and the tool schemas."""
    return fake_count([{"role": "user", "content": CONFIG.system}]) + fake_count(
        [{"role": "user", "content": json.dumps(list(CONFIG.tools), sort_keys=True)}]
    )


def test_a_window_that_cannot_fit_its_pinned_items_is_sent_whole_and_records_nothing(
    tmp_path: Path,
) -> None:
    world, task = long_run(context_budget=fixed_tokens() + 5)  # 5 left for history
    log = finish(world, task, tmp_path)
    assert context_events(log) == []
    for request, start in zip(world.requests, turn_starts(log), strict=True):
        assert request.messages == build_messages("go", log[:start])  # nothing cut


def test_a_budget_the_system_prompt_and_tools_already_fill_ends_the_run_with_the_reason(
    tmp_path: Path,
) -> None:
    world, task = long_run(context_budget=fixed_tokens())
    result = world.driver(tmp_path).run(task)
    assert result.reason == "error"
    assert "context_budget" in str(result.result)
    assert "no room" in str(result.result)
    assert world.requests == []  # nothing was sent


def assert_recorded_once(log: list[Event]) -> None:
    """An id is dropped once and cleared once. (A cleared item may be dropped later,
    so the two lists are separate.)"""
    dropped = [i for e in log if isinstance(e, ContextPacked) for i in e.dropped]
    cleared = [i for e in log if isinstance(e, ContextCompacted) for i in e.cleared]
    assert dropped
    assert len(dropped) == len(set(dropped))
    assert len(cleared) == len(set(cleared))


def test_a_crash_before_the_model_call_leaves_each_id_in_exactly_one_context_event(
    tmp_path: Path,
) -> None:
    world, task = long_run(context_budget=BUDGET)
    world.crash_on_model = 10  # context events are written, then the process dies
    driver = world.driver(tmp_path)
    with pytest.raises(Crash):
        driver.run(task)
    crashed = list(read_events(tmp_path / "s1.jsonl").events)
    assert context_events(crashed)  # the window had already been recorded
    assert driver.resume(run_ref()).status == "finished"
    log = list(read_events(tmp_path / "s1.jsonl").events)
    assert_recorded_once(log)


def test_a_limit_pause_after_context_events_resumes_without_recording_them_again(
    tmp_path: Path,
) -> None:
    world, task = long_run(context_budget=BUDGET, max_turns=10)
    driver = world.driver(tmp_path)
    paused = driver.run(task)
    assert (paused.status, paused.reason) == ("paused", "limit")
    assert context_events(list(read_events(tmp_path / "s1.jsonl").events))
    more = task.limits.model_copy(update={"max_turns": 100})
    assert driver.resume(paused.ref, limits=more).status == "finished"
    log = list(read_events(tmp_path / "s1.jsonl").events)
    assert_recorded_once(log)
