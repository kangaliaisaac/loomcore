"""f15: the cache, the generation and the timeout."""

import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fakes import CAPS, CONFIG, World, reply, use

from loomcore.eventlog import read_events
from loomcore.events import ToolCompleted
from loomcore.loop import Driver
from loomcore.notices import Notices
from loomcore.ports import Task
from loomcore.tools import ToolRegistry
from loomcore.workspace import FilesystemWorkspace


@pytest.fixture
def workspace(tmp_path: Path) -> FilesystemWorkspace:
    (tmp_path / "work").mkdir()
    return FilesystemWorkspace(tmp_path / "work", on_refusal=lambda _: None)


class Counter:
    """Tools that count their runs."""

    def __init__(self) -> None:
        self.runs: list[str] = []

    def look(self, key: str) -> str:
        """Looks a key up."""
        self.runs.append(key)
        return f"value of {key}"

    def change(self, key: str) -> str:
        """Changes something."""
        self.runs.append(f"change {key}")
        return "changed"

    def fail(self, key: str) -> str:
        """Always fails in a way the model can fix."""
        self.runs.append(key)
        raise OSError(2, "No such file or directory", key)


def registry(
    workspace: FilesystemWorkspace, counter: Counter, **options: float | None
) -> ToolRegistry:
    tools = ToolRegistry(workspace, **options)  # type: ignore[arg-type]
    tools.register(counter.look, read_only=True, cacheable=True)
    tools.register(counter.change, writes=("k",))
    tools.register(counter.fail, read_only=True, cacheable=True)
    return tools


def test_a_repeated_read_is_served_from_the_cache(
    workspace: FilesystemWorkspace,
) -> None:
    counter = Counter()
    tools = registry(workspace, counter)
    first = tools.execute("look", {"key": "a"})
    second = tools.execute("look", {"key": "a"})
    assert (first.cached, second.cached) == (False, True)
    assert second.content == first.content
    assert counter.runs == ["a"]
    assert not tools.execute("look", {"key": "b"}).cached  # another input


def test_a_write_makes_a_later_read_run_again(workspace: FilesystemWorkspace) -> None:
    counter = Counter()
    tools = registry(workspace, counter)
    tools.execute("look", {"key": "a"})
    before = workspace.generation
    tools.execute("change", {"key": "a"})
    assert workspace.generation == before + 1
    assert not tools.execute("look", {"key": "a"}).cached
    assert counter.runs == ["a", "change a", "a"]


def test_the_least_recently_used_result_goes_first(
    workspace: FilesystemWorkspace,
) -> None:
    counter = Counter()
    tools = registry(workspace, counter, cache_size=2)
    for key in ("a", "b", "a"):  # the second a is a hit and makes b the oldest
        tools.execute("look", {"key": key})
    tools.execute("look", {"key": "c"})  # evicts b
    assert tools.execute("look", {"key": "a"}).cached
    assert not tools.execute("look", {"key": "b"}).cached
    assert counter.runs == ["a", "b", "c", "b"]


def test_a_tool_that_is_not_cacheable_runs_every_time(
    workspace: FilesystemWorkspace,
) -> None:
    counter = Counter()
    tools = ToolRegistry(workspace)
    tools.register(counter.look, read_only=True)
    tools.execute("look", {"key": "a"})
    assert not tools.execute("look", {"key": "a"}).cached
    assert counter.runs == ["a", "a"]


def test_a_failed_result_is_not_cached(workspace: FilesystemWorkspace) -> None:
    counter = Counter()
    tools = registry(workspace, counter)
    assert not tools.execute("fail", {"key": "x"}).ok
    assert not tools.execute("fail", {"key": "x"}).cached
    assert counter.runs == ["x", "x"]


def wait_for_generation(workspace: FilesystemWorkspace, generation: int) -> None:
    deadline = time.monotonic() + 5
    while workspace.generation < generation and time.monotonic() < deadline:
        time.sleep(0.01)


def test_a_call_past_the_timeout_is_abandoned_and_its_late_write_still_invalidates(
    workspace: FilesystemWorkspace,
) -> None:
    release = threading.Event()
    counter = Counter()

    def slow_write(key: str) -> str:
        """Writes, eventually."""
        release.wait()
        return key

    tools = ToolRegistry(workspace, timeout=0.05)
    tools.register(slow_write, writes=("k",))
    tools.register(counter.look, read_only=True, cacheable=True)
    try:
        result = tools.execute("slow_write", {"key": "a"})
        assert not result.ok
        assert "timed out" in str(result.content)
        assert workspace.generation == 1  # bumped when abandoned
        # While the writer may still write, a read is neither served nor stored.
        assert not tools.execute("look", {"key": "a"}).cached
        assert not tools.execute("look", {"key": "a"}).cached
        assert counter.runs == ["a", "a"]
    finally:
        release.set()  # an abandoned thread must not outlive a failing test
    wait_for_generation(workspace, 2)  # bumped again when it finally returned
    assert workspace.generation == 2
    assert not tools.execute("look", {"key": "a"}).cached  # a new generation
    assert tools.execute("look", {"key": "a"}).cached  # and cached again
    assert counter.runs == ["a", "a", "a"]


def test_a_read_only_call_past_the_timeout_does_not_bump(
    workspace: FilesystemWorkspace,
) -> None:
    release = threading.Event()

    def slow_read(key: str) -> str:
        """Reads, eventually."""
        release.wait()
        return key

    tools = ToolRegistry(workspace, timeout=0.05)
    tools.register(slow_read, read_only=True)
    try:
        assert not tools.execute("slow_read", {"key": "a"}).ok
    finally:
        release.set()
    assert workspace.generation == 0


def test_a_timeout_error_raised_by_the_tool_is_not_the_registrys_timeout(
    workspace: FilesystemWorkspace,
) -> None:
    def flaky(key: str) -> str:
        """Raises the built-in timeout error itself."""
        raise TimeoutError(key)

    tools = ToolRegistry(workspace, timeout=5)
    tools.register(flaky, read_only=True)
    result = tools.execute("flaky", {"key": "a"})
    assert not result.ok
    assert "abandoned" not in str(result.content)  # it ended; nothing is running on


def test_no_timeout_runs_the_call_in_the_callers_thread(
    workspace: FilesystemWorkspace,
) -> None:
    seen: list[str] = []

    def where(key: str) -> str:
        """Records its thread."""
        seen.append(threading.current_thread().name)
        return key

    tools = ToolRegistry(workspace, timeout=None)
    tools.register(where, read_only=True)
    tools.execute("where", {"key": "a"})
    assert seen == [threading.current_thread().name]


def test_a_call_that_outlasts_the_timeout_runs_on_a_daemon_thread(
    workspace: FilesystemWorkspace,
) -> None:
    daemons: list[bool] = []

    def where(key: str) -> str:
        """Records whether its thread is a daemon."""
        daemons.append(threading.current_thread().daemon)
        return key

    tools = ToolRegistry(workspace, timeout=5)
    tools.register(where, read_only=True)
    tools.execute("where", {"key": "a"})
    assert daemons == [True]  # so a call that never returns cannot hold the exit


def test_a_caller_that_changes_a_cached_result_does_not_change_later_hits(
    workspace: FilesystemWorkspace,
) -> None:
    def rows(key: str) -> list[str]:
        """Returns a list."""
        return [key]

    tools = ToolRegistry(workspace)
    tools.register(rows, read_only=True, cacheable=True)
    first = tools.execute("rows", {"key": "a"})
    assert isinstance(first.content, list)
    first.content.append("changed")
    second = tools.execute("rows", {"key": "a"})
    assert isinstance(second.content, list)
    second.content.append("changed too")
    assert tools.execute("rows", {"key": "a"}).content == ["a"]


def test_a_late_notice_from_an_abandoned_call_is_warned_about_not_given_to_another(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "w").mkdir()
    notices = Notices()
    workspace = FilesystemWorkspace(
        tmp_path / "w", on_refusal=notices.refused, on_note=notices.noted
    )
    release = threading.Event()

    def slow_note(text: str) -> str:
        """Writes a note, eventually."""
        release.wait()
        workspace.write("notes.md", text)
        return "saved"

    tools = ToolRegistry(workspace, timeout=0.05)
    tools.register(slow_note)
    try:
        with notices.collecting() as first:
            assert not tools.execute("slow_note", {"text": "x"}).ok
        with notices.collecting() as second, caplog.at_level("WARNING"):
            release.set()
            wait_for_generation(workspace, 3)  # its write, the abandon, the return
    finally:
        release.set()
    assert (list(first), list(second)) == ([], [])
    assert "arrived after its tool call ended" in caplog.text


def test_a_cache_hit_reaches_the_log_as_cached(
    tmp_path: Path, workspace: FilesystemWorkspace
) -> None:
    runs: list[str] = []

    def look(q: str) -> str:
        """Looks a key up."""
        runs.append(q)
        return f"value of {q}"

    tools = ToolRegistry(workspace)
    tools.register(look, read_only=True, cacheable=True)
    world = World(script=[use(("t1", "look"), ("t2", "look"), q="same"), reply()])
    ports = replace(world.ports(), tool=tools.execute, read_only=tools.read_only)
    driver = Driver(CONFIG, ports, tmp_path / "s1.jsonl", tmp_path / "spend.jsonl")
    driver.run(Task("find it", CAPS))
    log = read_events(tmp_path / "s1.jsonl").events
    done = [e for e in log if isinstance(e, ToolCompleted)]
    assert [(e.tool_use_id, e.cached) for e in done] == [("t1", False), ("t2", True)]
    assert runs == ["same"]
