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


def test_pinned_items_that_do_not_fit_are_sent_as_they_are_and_record_nothing(
    tmp_path: Path,
) -> None:
    world, task = long_run(context_budget=20)
    log = finish(world, task, tmp_path)
    assert context_events(log) == []
    assert len(world.requests) == TURNS + 1
