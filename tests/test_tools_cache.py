"""f15: the cache, the generation and the timeout."""

import threading
import time
from pathlib import Path

import pytest

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


def test_a_call_past_the_timeout_is_abandoned_and_its_late_write_still_bumps(
    workspace: FilesystemWorkspace,
) -> None:
    release = threading.Event()

    def slow_write(key: str) -> str:
        """Writes, eventually."""
        release.wait()
        return key

    tools = ToolRegistry(workspace, timeout=0.05)
    tools.register(slow_write, writes=("k",))
    try:
        result = tools.execute("slow_write", {"key": "a"})
        assert not result.ok
        assert "timed out" in str(result.content)
        assert workspace.generation == 1  # bumped when abandoned
    finally:
        release.set()  # an abandoned thread must not outlive a failing test
    deadline = time.monotonic() + 5
    while workspace.generation < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert workspace.generation == 2  # and again when it finally returned


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


def test_access_is_what_the_tool_declared(workspace: FilesystemWorkspace) -> None:
    counter = Counter()
    tools = registry(workspace, counter)
    assert tools.access("change").writes == frozenset({"k"})
    assert tools.access("look").read_only
    assert tools.access("nope").read_only  # an unknown tool touches nothing
