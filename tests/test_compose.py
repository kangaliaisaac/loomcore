"""The tool side of the ports is built from one `Notices` (spec 3.2)."""

from dataclasses import replace
from pathlib import Path

import pytest
from fakes import CAPS, World, reply

from loomcore.compose import tool_kit
from loomcore.eventlog import read_events
from loomcore.events import JsonObject, SandboxDenied
from loomcore.loop import Driver, Settings
from loomcore.ports import Config, Task
from loomcore.workspace import PathRefusedError, Workspace


def read_file(workspace: Workspace, path: str) -> str:
    """Read a file in the workspace."""
    return workspace.read(path)


def test_the_workspace_reports_to_the_notices_the_kit_hands_to_the_driver(
    tmp_path: Path,
) -> None:
    (tmp_path / "work").mkdir()
    kit = tool_kit(tmp_path / "work")
    with kit.notices.collecting() as bucket, pytest.raises(PathRefusedError):
        kit.workspace.read("../outside")
    assert [n.kind for n in bucket.snapshot()] == ["sandbox_denied"]
    assert kit.tool_ports()["notices"] is kit.notices


def test_a_refusal_inside_a_driven_tool_call_is_logged_with_the_call(
    tmp_path: Path,
) -> None:
    (tmp_path / "work").mkdir()
    kit = tool_kit(tmp_path / "work")
    kit.registry.register(read_file, read_only=True)
    tools = tuple(schema.to_json() for schema in kit.registry.schemas())
    call: JsonObject = {
        "content": [
            {
                "type": "tool_use",
                "id": "a",
                "name": "read_file",
                "input": {"path": "../outside"},
            }
        ],
        "stop_reason": "tool_use",
    }
    world = World(script=[call, reply("done")])
    ports = replace(world.ports(), **kit.tool_ports())
    config = Config(model="m", system="be brief", tools=tools)
    settings = Settings(tmp_path / "s1.jsonl", tmp_path / "spend.jsonl")
    result = Driver(config, ports, settings).run(Task("go", CAPS))
    assert result.status == "finished"
    denied = [
        e
        for e in read_events(tmp_path / "s1.jsonl").events
        if isinstance(e, SandboxDenied)
    ]
    assert [d.tool_use_id for d in denied] == ["a"]  # not a warning and a lost event


def test_the_registry_ports_the_kit_hands_out_agree_about_read_only(
    tmp_path: Path,
) -> None:
    (tmp_path / "work").mkdir()
    kit = tool_kit(tmp_path / "work", timeout=5)
    kit.registry.register(read_file, read_only=True)
    ports = kit.tool_ports()
    assert ports["read_only"]("read_file") is True
    assert ports["access"]("read_file").read_only is True
