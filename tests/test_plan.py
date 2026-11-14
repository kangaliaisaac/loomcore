"""f18: the model keeps a plan, and its latest plan is always in the window."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from factories import calling, make
from fakes import CAPS, World, fake_count, reply

from loomcore import plan
from loomcore.context import PLAN_TOOL, Cleared, Item
from loomcore.conversation import exchanges
from loomcore.eventlog import read_events
from loomcore.events import ContextPacked, Event, JsonObject
from loomcore.loop import Driver
from loomcore.notices import Notices
from loomcore.plan import MAX_PLAN_CHARS, PLAN_FILE, update_plan
from loomcore.ports import Config, Task, ToolResult
from loomcore.tools import ToolRegistry
from loomcore.workspace import FilesystemWorkspace


@pytest.fixture
def notices() -> Notices:
    return Notices()


@pytest.fixture
def workspace(tmp_path: Path, notices: Notices) -> FilesystemWorkspace:
    (tmp_path / "work").mkdir()
    return FilesystemWorkspace(
        tmp_path / "work", on_refusal=notices.refused, on_note=notices.noted
    )


@pytest.fixture
def registry(workspace: FilesystemWorkspace) -> ToolRegistry:
    tools = ToolRegistry(workspace)
    plan.register(tools)
    return tools


@dataclass(frozen=True)
class Scene:
    tmp_path: Path
    workspace: FilesystemWorkspace
    registry: ToolRegistry
    notices: Notices


@pytest.fixture
def scene(
    tmp_path: Path,
    workspace: FilesystemWorkspace,
    registry: ToolRegistry,
    notices: Notices,
) -> Scene:
    return Scene(tmp_path, workspace, registry, notices)


def test_the_tool_is_named_for_the_constant_the_store_pins_by(
    registry: ToolRegistry,
) -> None:
    assert [schema.name for schema in registry.schemas()] == [PLAN_TOOL]
    assert registry.access(PLAN_TOOL).writes == frozenset({PLAN_FILE})
    assert not registry.read_only(PLAN_TOOL)


def test_a_plan_is_written_to_the_file_and_replaces_the_one_before(
    workspace: FilesystemWorkspace,
) -> None:
    update_plan(workspace, "1. read\n2. write")
    assert workspace.read(PLAN_FILE) == "1. read\n2. write\n"
    update_plan(workspace, "  3. done  ")
    assert workspace.read(PLAN_FILE) == "3. done\n"  # replaced, trimmed


@pytest.mark.parametrize("text", ["", "  \n "])
def test_an_empty_plan_is_a_result_the_model_can_fix(
    registry: ToolRegistry, workspace: FilesystemWorkspace, text: str
) -> None:
    result = registry.execute(PLAN_TOOL, {"text": text})
    assert not result.ok
    assert "empty" in str(result.content)
    assert workspace.list() == []  # nothing was written


def test_a_plan_too_long_to_pin_is_refused_with_how_to_fix_it(
    registry: ToolRegistry,
) -> None:
    result = registry.execute(PLAN_TOOL, {"text": "x" * (MAX_PLAN_CHARS + 1)})
    assert not result.ok
    assert str(MAX_PLAN_CHARS) in str(result.content)
    assert "say it shorter" in str(result.content)
    assert registry.execute(PLAN_TOOL, {"text": "x" * MAX_PLAN_CHARS}).ok


def plan_call(tool_use_id: str, text: str) -> JsonObject:
    return {
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": PLAN_TOOL,
                "input": {"text": text},
            }
        ],
        "stop_reason": "tool_use",
    }


def plan_turn(seq: int, tool_use_id: str, ok: bool = True) -> list[Event]:
    return [
        make("model_call", seq, message=plan_call(tool_use_id, "the plan")),
        make("tool_completed", seq + 1, tool_use_id=tool_use_id, ok=ok),
    ]


def plain_turn(seq: int, tool_use_id: str) -> list[Event]:
    return [
        make("model_call", seq, message=calling(tool_use_id)),
        make("tool_completed", seq + 1, tool_use_id=tool_use_id),
    ]


def pinned_exchanges(log: list[Event]) -> list[str]:
    items = exchanges("go", log, fake_count)
    return [item.id for item in items if item.pinned and item.kind == "exchange"]


def test_the_exchange_of_the_latest_successful_plan_is_pinned() -> None:
    log = [
        *plan_turn(1, "p1"),
        *plain_turn(3, "a"),
        *plan_turn(5, "p2"),
        *plain_turn(7, "b"),
    ]
    # the latest exchange (b) and the latest plan (p2); the first plan lost its pin
    assert pinned_exchanges(log) == ["seq:r1:5", "seq:r1:7"]


def test_a_plan_update_that_failed_is_not_pinned() -> None:
    log = [*plan_turn(1, "p1"), *plan_turn(3, "p2", ok=False), *plain_turn(5, "a")]
    assert pinned_exchanges(log) == ["seq:r1:1", "seq:r1:5"]  # p1 is still the plan


def test_a_run_that_never_calls_the_tool_pins_only_the_latest_exchange() -> None:
    assert pinned_exchanges([*plain_turn(1, "a"), *plain_turn(3, "b")]) == ["seq:r1:3"]


BUDGET = 600


def long_tool_call(tool_use_id: str) -> JsonObject:
    return {
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": "search",
                "input": {"q": tool_use_id},
            }
        ],
        "stop_reason": "tool_use",
    }


def run_with_plan(scene: Scene, *, sibling: bool = False) -> tuple[World, list[str]]:
    """Twelve long tool turns around two plans, under a small context budget. With
    `sibling`, the first plan is written in the same turn as a very long result."""
    ids = [f"t{k}" for k in range(12)]
    first = plan_call("plan1", "STEP ONE then STEP TWO. " + "detail " * 60)
    if sibling:
        first["content"] = [*first["content"], *long_tool_call("big")["content"]]  # type: ignore[misc]
    script: list[JsonObject | Exception] = [
        first,
        *[long_tool_call(i) for i in ids[:6]],
        plan_call("plan2", "NEW PLAN: finish"),
        *[long_tool_call(i) for i in ids[6:]],
        reply("done"),
    ]
    results = {i: ToolResult(True, f"{i} " + "data " * 80) for i in [*ids, "big"]}
    results["big"] = ToolResult(True, "BIGRESULT " * 400)
    world = World(script=script, tool_results=results, notices=scene.notices)

    def route(name: str, args: JsonObject) -> ToolResult:
        if name == PLAN_TOOL:
            return scene.registry.execute(name, args)
        return world.tool(name, args)

    ports = replace(
        world.ports(),
        tool=route,
        read_only=lambda name: name == "search",
        access=lambda name: (
            scene.registry.access(name) if name == PLAN_TOOL else world.access(name)
        ),
    )
    tools = tuple(schema.to_json() for schema in scene.registry.schemas())
    config = Config(model="m", system="be brief", tools=tools)
    limits = CAPS.model_copy(update={"context_budget": BUDGET, "max_turns": 100})
    session, spend = scene.tmp_path / "s1.jsonl", scene.tmp_path / "spend.jsonl"
    assert (
        Driver(config, ports, session, spend).run(Task("go", limits)).status
        == "finished"
    )
    assert scene.workspace.read(PLAN_FILE) == "NEW PLAN: finish\n"
    return world, [json.dumps(request.messages) for request in world.requests]


def test_the_plan_stays_in_every_request_while_the_history_around_it_is_dropped(
    scene: Scene,
) -> None:
    _, texts = run_with_plan(scene)
    assert all(
        "STEP ONE" in text for text in texts[1:8]
    )  # the plan while it is current
    assert all("NEW PLAN" in text for text in texts[8:])  # then the second replaces it
    assert "t0 data" not in texts[-1]  # the history around it really was thinned
    log = list(read_events(scene.tmp_path / "s1.jsonl").events)
    assert any(isinstance(e, ContextPacked) for e in log)


def test_without_the_pin_the_same_run_loses_the_plan(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check that the test above can fail: take the pin away and the first plan
    is dropped with the rest of the old history."""
    monkeypatch.setattr("loomcore.conversation.PLAN_TOOL", "not_the_plan_tool")
    _, texts = run_with_plan(scene)
    assert not all("STEP ONE" in text for text in texts[1:8])


def test_a_long_result_beside_the_plan_is_cleared_at_once_and_the_plan_stays(
    scene: Scene, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        _, texts = run_with_plan(scene, sibling=True)
    assert "BIGRESULT" in texts[1]  # shown in full while it is the latest exchange
    assert all("BIGRESULT" not in text for text in texts[2:])  # cleared from then on
    assert all("STEP ONE" in text for text in texts[1:8])  # the plan never left
    # only the request that carries that latest exchange whole is over its budget
    assert caplog.text.count("over its budget") == 1


def test_without_clearing_the_pinned_plan_exchange_the_long_result_stays_and_overruns(
    scene: Scene, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check that the test above can fail: forbid clearing a pinned item."""

    def never_pinned(item: Item) -> Cleared | None:
        form = item.cleared
        usable = not item.pinned and form is not None and form.tokens < item.tokens
        return form if usable else None

    monkeypatch.setattr("loomcore.context._smaller_form", never_pinned)
    with caplog.at_level("WARNING"):
        _, texts = run_with_plan(scene, sibling=True)
    assert any("BIGRESULT" in text for text in texts[2:8])
    assert caplog.text.count("over its budget") > 1
