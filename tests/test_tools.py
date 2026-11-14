"""f14: typed functions become tools; invalid input is a fixable result."""

import threading
from contextvars import copy_context
from pathlib import Path

import pytest
from pydantic import BaseModel

from loomcore.notices import Notices
from loomcore.ports import Notice, ToolCrashedError, ToolResult
from loomcore.sandbox import CommandRefusedError, Refusal
from loomcore.tools import ToolError, ToolRegistry
from loomcore.workspace import FilesystemWorkspace, PathRefusedError, Workspace


class Point(BaseModel):
    x: int
    y: int


class Registry:
    """A registry on a real workspace, with its notices wired as the callbacks."""

    def __init__(self, root: Path) -> None:
        root.mkdir()
        self.workspace_root = root
        self.notices = Notices()
        self.workspace = FilesystemWorkspace(
            root, on_refusal=self.notices.refused, on_note=self.notices.noted
        )
        self.registry = ToolRegistry(self.workspace, self.notices)


@pytest.fixture
def tools(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "work")


def distance(origin: Point, targets: list[Point], scale: float = 1.0) -> float:
    """How far the targets are from the origin."""
    return scale * sum(abs(t.x - origin.x) + abs(t.y - origin.y) for t in targets)


def read_file(workspace: Workspace, path: str) -> str:
    """Read a file in the workspace."""
    return workspace.read(path)


def write_note(workspace: Workspace, text: str) -> str:
    """Append to the notes file."""
    workspace.write("notes.md", text)
    return "saved"


def test_schemas_come_from_the_hints_including_nested_types(tools: Registry) -> None:
    tools.registry.register(distance, read_only=True)
    (schema,) = tools.registry.schemas()
    assert schema.name == "distance"
    assert schema.description == "How far the targets are from the origin."
    properties = schema.input_schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {"origin", "targets", "scale"}
    assert schema.input_schema["required"] == ["origin", "targets"]
    assert set(schema.input_schema["$defs"]) == {"Point"}  # type: ignore[arg-type]
    assert "title" not in schema.input_schema
    assert schema.to_json()["input_schema"] == schema.input_schema


def test_the_workspace_is_passed_in_and_is_not_in_the_schema(tools: Registry) -> None:
    tools.registry.register(read_file, read_only=True)
    tools.workspace.write("a.txt", "hello")
    (schema,) = tools.registry.schemas()
    properties = schema.input_schema["properties"]
    assert isinstance(properties, dict)
    assert list(properties) == ["path"]
    assert tools.registry.execute("read_file", {"path": "a.txt"}).content == "hello"


def test_nested_input_is_validated_and_used(tools: Registry) -> None:
    tools.registry.register(distance)
    result = tools.registry.execute(
        "distance",
        {"origin": {"x": 0, "y": 0}, "targets": [{"x": 1, "y": 2}], "scale": 2},
    )
    assert (result.ok, result.content) == (True, 6.0)


@pytest.mark.parametrize(
    ("args", "mention"),
    [
        ({"origin": {"x": 0}, "targets": []}, "origin.y"),
        ({"origin": {"x": 0, "y": 0}}, "targets"),
        ({"origin": {"x": 0, "y": 0}, "targets": [], "color": "red"}, "color"),
    ],
)
def test_invalid_input_is_not_ok_and_says_what_to_fix(
    tools: Registry, args: dict[str, object], mention: str
) -> None:
    tools.registry.register(distance)
    result = tools.registry.execute("distance", args)  # type: ignore[arg-type]
    assert not result.ok
    assert isinstance(result.content, str)
    assert mention in result.content
    assert "call it again" in result.content


def test_an_unknown_tool_names_the_ones_there_are(tools: Registry) -> None:
    tools.registry.register(distance)
    result = tools.registry.execute("nope", {})
    assert not result.ok
    assert "distance" in str(result.content)
    assert tools.registry.read_only("nope") is False


def test_a_tool_error_is_a_failed_result_with_its_own_message(tools: Registry) -> None:
    def pick(name: str) -> str:
        """Picks a thing by name."""
        raise ToolError(f"there is no {name!r}; try 'a' or 'b'")

    tools.registry.register(pick)
    result = tools.registry.execute("pick", {"name": "z"})
    assert (result.ok, result.content) == (
        False,
        "pick failed: there is no 'z'; try 'a' or 'b'",
    )


def test_a_return_value_that_is_not_json_is_a_failed_result(tools: Registry) -> None:
    def odd() -> object:
        """Returns a thing that is not JSON."""
        return object()

    tools.registry.register(odd)
    assert "not JSON" in str(tools.registry.execute("odd", {}).content)


def test_a_bug_in_a_tool_is_not_swallowed_and_leaves_no_bucket_behind(
    tools: Registry,
) -> None:
    def boom() -> str:
        """Fails in a way nobody planned for."""
        raise RuntimeError("no luck")

    tools.registry.register(boom)
    tools.registry.register(write_note)
    with pytest.raises(ToolCrashedError, match="no luck"):
        tools.registry.execute("boom", {})
    # the next call collects only its own notices
    assert [
        n.kind for n in tools.registry.execute("write_note", {"text": "x"}).notices
    ] == ["note_written"]


def test_registering_checks_the_tool(tools: Registry) -> None:
    def undocumented(a: int) -> int:
        return a

    def unhinted(a) -> int:  # type: ignore[no-untyped-def]  # noqa: ANN001
        """No hint."""
        return 1

    with pytest.raises(ValueError, match="docstring"):
        tools.registry.register(undocumented)
    with pytest.raises(TypeError, match="type hint"):
        tools.registry.register(unhinted)
    with pytest.raises(ValueError, match="cacheable needs read_only"):
        tools.registry.register(distance, cacheable=True)
    tools.registry.register(distance)
    with pytest.raises(ValueError, match="already registered"):
        tools.registry.register(distance)


def test_read_only_is_what_the_tool_declared(tools: Registry) -> None:
    tools.registry.register(read_file, read_only=True)
    tools.registry.register(write_note)
    assert tools.registry.read_only("read_file") is True
    assert tools.registry.read_only("write_note") is False


def test_a_refusal_becomes_a_sandbox_denied_notice(tools: Registry) -> None:
    tools.registry.register(read_file, read_only=True)
    result = tools.registry.execute("read_file", {"path": "../secret"})
    assert not result.ok
    (notice,) = result.notices
    assert notice.kind == "sandbox_denied"
    assert notice.fields["action"] == "read"
    assert notice.fields["target"] == "../secret"


def test_a_note_becomes_a_note_written_notice(tools: Registry) -> None:
    tools.registry.register(write_note)
    result = tools.registry.execute("write_note", {"text": "x"})
    assert result.notices == (Notice("note_written", {"path": "notes.md"}),)


def test_notices_land_with_the_call_that_caused_them(tools: Registry) -> None:
    tools.registry.register(read_file, read_only=True)
    tools.registry.register(write_note)
    refused = tools.registry.execute("read_file", {"path": "../x"})
    noted = tools.registry.execute("write_note", {"text": "x"})
    assert [n.kind for n in refused.notices] == ["sandbox_denied"]
    assert [n.kind for n in noted.notices] == ["note_written"]


def test_a_refusal_outside_a_tool_call_is_not_collected(tools: Registry) -> None:
    with pytest.raises(PathRefusedError):
        tools.workspace.read("../x")
    tools.registry.register(write_note)
    assert [
        n.kind for n in tools.registry.execute("write_note", {"text": "x"}).notices
    ] == ["note_written"]


def test_a_worker_thread_under_a_copied_context_reports_to_its_call(
    tools: Registry,
) -> None:
    with tools.notices.collecting() as bucket:
        context = copy_context()
        worker = threading.Thread(
            target=context.run, args=(tools.notices.noted, "notes.md")
        )
        worker.start()
        worker.join()
        bare = threading.Thread(target=tools.notices.noted, args=("lost.md",))
        bare.start()
        bare.join()
    assert bucket == [Notice("note_written", {"path": "notes.md"})]


def test_a_tool_named_finish_is_refused(tools: Registry) -> None:
    def finish() -> str:
        """Ends the run."""
        return "done"

    with pytest.raises(ValueError, match="reserved"):
        tools.registry.register(finish)


def test_a_missing_file_does_not_put_the_host_path_in_the_result(
    tools: Registry, tmp_path: Path
) -> None:
    tools.registry.register(read_file, read_only=True)
    result = tools.registry.execute("read_file", {"path": "gone.txt"})
    assert not result.ok
    assert "No such file" in str(result.content)
    assert str(tmp_path) not in str(result.content)


def test_a_sandbox_refusal_reported_by_a_tool_becomes_a_notice(tools: Registry) -> None:
    def run_it() -> str:
        """Stands in for a tool whose sandbox refuses the command."""
        tools.notices.refused(Refusal("exec", "/bin/sh", "not an allowed command"))
        raise CommandRefusedError(Refusal("exec", "/bin/sh", "not an allowed command"))

    tools.registry.register(run_it)
    result = tools.registry.execute("run_it", {})
    assert not result.ok
    assert result.notices == (
        Notice(
            "sandbox_denied",
            {"action": "exec", "target": "/bin/sh", "reason": "not an allowed command"},
        ),
    )


def test_a_nested_call_does_not_take_the_outer_calls_notices(tmp_path: Path) -> None:
    outer, inner = Registry(tmp_path / "a"), Registry(tmp_path / "b")
    seen: list[ToolResult] = []

    def poke() -> str:
        """Writes the outer workspace's notes while the inner call runs."""
        outer.workspace.write("notes.md", "n")
        return "ok"

    def delegate() -> str:
        """A tool that runs another registry's tool."""
        seen.append(inner.registry.execute("poke", {}))
        return "ok"

    inner.registry.register(poke)
    outer.registry.register(delegate)
    result = outer.registry.execute("delegate", {})
    assert seen[0].notices == ()  # the note belongs to the outer call
    assert [n.kind for n in result.notices] == ["note_written"]


def test_a_binary_file_is_a_failed_result_not_a_crash(tools: Registry) -> None:
    tools.registry.register(read_file, read_only=True)
    (tools.workspace_root / "blob.bin").write_bytes(b"\xff\xfe\x00")
    result = tools.registry.execute("read_file", {"path": "blob.bin"})
    assert not result.ok
    assert "not valid UTF-8" in str(result.content)


def test_a_crashing_tool_still_hands_over_what_it_had_caused(tools: Registry) -> None:
    def half_then_bug() -> str:
        """Writes a note, then breaks."""
        tools.workspace.write("notes.md", "n")
        raise RuntimeError("no luck")

    tools.registry.register(half_then_bug)
    with pytest.raises(ToolCrashedError) as crashed:
        tools.registry.execute("half_then_bug", {})
    assert isinstance(crashed.value.error, RuntimeError)
    assert [n.kind for n in crashed.value.notices] == ["note_written"]
