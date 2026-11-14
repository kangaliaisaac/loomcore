"""f15, part 2: the driver runs a turn's independent calls together."""

import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import TracebackType

import pytest
from factories import make
from fakes import CAPS, CONFIG, ToolBugError, World, reply, use
from fakes import ref as run_ref
from hypothesis import given, settings
from hypothesis import strategies as st

from loomcore.conversation import call_hash
from loomcore.engine import MAX_TOGETHER
from loomcore.eventlog import open_writer, read_events
from loomcore.events import Event, JsonObject, NoteWritten, ToolCompleted, ToolRequested
from loomcore.loop import Driver
from loomcore.ports import RunResult, Task, ToolResult
from loomcore.schedule import Access, conflict

TASK = Task("find it", CAPS)
Tool = Callable[[str, JsonObject], ToolResult]


class Lanes:
    """Records when each call started and ended, and lets calls wait for each other."""

    def __init__(self, together: int = 2) -> None:
        self.lock = threading.Lock()
        self.marks: list[str] = []
        self.meet = threading.Barrier(together, timeout=5)

    def mark(self, text: str) -> None:
        with self.lock:
            self.marks.append(text)

    def tool(self, name: str, args: JsonObject) -> ToolResult:
        call = str(args["q"])
        self.mark(f"start {call}")
        if name == "scan":
            self.meet.wait()  # breaks after 5 s unless the other scan is in flight
        self.mark(f"end {call}")
        return ToolResult(True, f"result {call}")


def driver(world: World, tmp: Path, tool: Tool) -> Driver:
    ports = replace(world.ports(), tool=tool)
    return Driver(CONFIG, ports, tmp / "s1.jsonl", tmp / "spend.jsonl")


def log(tmp: Path) -> tuple[Event, ...]:
    return read_events(tmp / "s1.jsonl").events


def kinds(events: tuple[Event, ...]) -> list[str]:
    return [e.kind for e in events]


def test_independent_calls_are_in_flight_at_the_same_time(tmp_path: Path) -> None:
    world = World(
        script=[use(("t1", "scan"), ("t2", "scan")), reply()], together={"scan"}
    )
    lanes = Lanes()
    result = driver(world, tmp_path, lanes.tool).run(TASK)
    assert result.status == "finished"  # a Barrier that broke would have been a bug
    assert sorted(lanes.marks[:2]) == ["start t1", "start t2"]  # both begin first
    waves = [e.wave for e in log(tmp_path) if isinstance(e, ToolRequested)]
    assert waves == [0, 0]


def test_the_log_is_written_in_the_order_the_model_emitted_the_calls(
    tmp_path: Path,
) -> None:
    world = World(
        script=[use(("t1", "scan"), ("t2", "scan")), reply()], together={"scan"}
    )
    driver(world, tmp_path, Lanes().tool).run(TASK)
    done = [e.tool_use_id for e in log(tmp_path) if isinstance(e, ToolCompleted)]
    assert done == ["t1", "t2"]  # whichever thread finished first


def test_a_conflicting_call_waits_for_the_one_before_it_and_holds_the_next_back(
    tmp_path: Path,
) -> None:
    world = World(
        script=[use(("t1", "write"), ("t2", "scan"), ("t3", "scan")), reply()],
        together={"scan"},
    )
    lanes = Lanes()
    driver(world, tmp_path, lanes.tool).run(TASK)
    marks = lanes.marks
    assert marks[:2] == ["start t1", "end t1"]  # the write ran alone, first
    assert sorted(marks[2:4]) == ["start t2", "start t3"]  # then the scans together
    waves = [e.wave for e in log(tmp_path) if isinstance(e, ToolRequested)]
    assert waves == [0, 1, 1]
    assert kinds(log(tmp_path)).count("tool_completed") == 3


def test_what_each_call_caused_is_logged_with_its_own_id_before_its_completion(
    tmp_path: Path,
) -> None:
    world = World(
        script=[use(("t1", "scan"), ("t2", "scan")), reply()], together={"scan"}
    )
    meet = threading.Barrier(2, timeout=5)

    def tool(name: str, args: JsonObject) -> ToolResult:
        call = str(args["q"])
        world.notices.noted(f"{call}.md")
        meet.wait()
        return ToolResult(True, call)

    driver(world, tmp_path, tool).run(TASK)
    events = [e for e in log(tmp_path) if e.kind in ("note_written", "tool_completed")]
    assert [(e.kind, e.tool_use_id) for e in events if hasattr(e, "tool_use_id")] == [
        ("note_written", "t1"),
        ("tool_completed", "t1"),
        ("note_written", "t2"),
        ("tool_completed", "t2"),
    ]
    paths = {e.tool_use_id: e.path for e in events if isinstance(e, NoteWritten)}
    assert paths == {"t1": "t1.md", "t2": "t2.md"}


def test_a_bug_in_one_call_of_a_wave_logs_the_others_and_resumes(
    tmp_path: Path,
) -> None:
    world = World(
        script=[use(("t1", "scan"), ("t2", "scan")), reply("done")],
        together={"scan"},
        read_only={"scan"},
    )
    broken = {"t2"}
    gate = {"meet": threading.Barrier(2, timeout=5)}

    def tool(name: str, args: JsonObject) -> ToolResult:
        call = str(args["q"])
        world.notices.noted(f"{call}.md")
        gate["meet"].wait()
        if call in broken:
            raise ToolBugError(call)
        return ToolResult(True, call)

    first = driver(world, tmp_path, tool)
    with pytest.raises(ToolBugError):
        first.run(TASK)
    events = log(tmp_path)
    done = [e.tool_use_id for e in events if isinstance(e, ToolCompleted)]
    assert done == ["t1"]  # t2 broke before it could complete
    notes = [e.tool_use_id for e in events if isinstance(e, NoteWritten)]
    assert notes == ["t1", "t2"]  # both calls' notices were still written
    broken.clear()
    gate["meet"] = threading.Barrier(1, timeout=5)  # only t2 runs again
    result: RunResult = first.resume(run_ref())
    assert result.status == "finished"
    assert [e.tool_use_id for e in log(tmp_path) if isinstance(e, ToolCompleted)] == [
        "t1",
        "t2",
    ]


DECLARED = {
    "read_x": Access(True, reads=frozenset({"x"})),
    "write_x": Access(False, writes=frozenset({"x"})),
    "read_y": Access(True, reads=frozenset({"y"})),
    "write_y": Access(False, writes=frozenset({"y"})),
    "anything": Access(False),  # may write anything
}


@settings(deadline=None, max_examples=40)
@given(st.lists(st.sampled_from(sorted(DECLARED)), min_size=1, max_size=6))
def test_any_mix_of_tools_runs_in_waves_and_logs_in_wave_then_emission_order(
    names: list[str],
) -> None:
    ids = [f"t{k}" for k in range(len(names))]
    world = World(script=[use(*zip(ids, names, strict=True)), reply()])
    world.declared.update(DECLARED)
    lanes = Lanes(together=1)  # a lone barrier party: nothing waits
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        driver(world, tmp, lanes.tool).run(TASK)
        events = log(tmp)
    wave_of = {e.tool_use_id: e.wave for e in events if isinstance(e, ToolRequested)}
    done = [e.tool_use_id for e in events if isinstance(e, ToolCompleted)]
    # waves ascending, and inside a wave the order the model emitted the calls
    assert done == sorted(ids, key=lambda i: (wave_of[i] or 0, ids.index(i)))
    for i, first in enumerate(ids):
        for j in range(i + 1, len(ids)):
            if conflict(DECLARED[names[i]], DECLARED[names[j]]):
                assert (wave_of[first] or 0) < (wave_of[ids[j]] or 0)
                ended = lanes.marks.index(f"end {first}")
                started = lanes.marks.index(f"start {ids[j]}")
                assert ended < started  # a conflicting pair never overlaps


def test_lines_a_crash_left_unwritten_go_after_every_wave_already_in_the_log(
    tmp_path: Path,
) -> None:
    args: JsonObject = {"q": "a"}
    log_events = [
        make(
            "run_started",
            0,
            task="go",
            config=CONFIG.to_json(),
            limits=CAPS.to_json(),
            config_digest=CONFIG.digest(),
        ),
        make("turn_started", 1, turn=1),
        make(
            "model_call",
            2,
            message=use(("a", "other"), ("b", "scan"), ("c", "scan")),
        ),
        make(
            "tool_requested",
            3,
            tool_use_id="a",
            name="other",
            input=args,
            call_hash=call_hash("other", args),
            wave=5,  # whatever the old process worked out
        ),
    ]
    with open_writer(tmp_path / "s1.jsonl") as writer:
        for event in log_events:
            writer.append(event)
    world = World(script=[reply("done")], together={"scan"})
    result = driver(world, tmp_path, Lanes().tool).resume(run_ref())
    assert result.status == "finished"
    waves = {
        e.tool_use_id: e.wave for e in log(tmp_path) if isinstance(e, ToolRequested)
    }
    assert waves == {"a": 5, "b": 6, "c": 6}


def test_a_second_bug_in_a_wave_is_warned_about_while_the_first_is_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    world = World(
        script=[use(("t1", "scan"), ("t2", "scan")), reply()], together={"scan"}
    )
    meet = threading.Barrier(2, timeout=5)

    def tool(name: str, args: JsonObject) -> ToolResult:
        meet.wait()
        raise ToolBugError(str(args["q"]))

    with caplog.at_level("WARNING"), pytest.raises(ToolBugError, match="t1"):
        driver(world, tmp_path, tool).run(TASK)
    assert "t2 also failed" in caplog.text


def test_no_more_than_the_cap_of_a_wave_runs_at_once(tmp_path: Path) -> None:
    ids = [f"t{k}" for k in range(MAX_TOGETHER + 4)]
    world = World(script=[use(*((i, "scan") for i in ids)), reply()], together={"scan"})
    lock = threading.Lock()
    running = {"now": 0, "most": 0}

    def tool(name: str, args: JsonObject) -> ToolResult:
        with lock:
            running["now"] += 1
            running["most"] = max(running["most"], running["now"])
        time.sleep(0.02)
        with lock:
            running["now"] -= 1
        return ToolResult(True, str(args["q"]))

    result = driver(world, tmp_path, tool).run(TASK)
    assert result.status == "finished"
    done = [e.tool_use_id for e in log(tmp_path) if isinstance(e, ToolCompleted)]
    assert done == ids
    assert 1 < running["most"] <= MAX_TOGETHER


def test_a_signal_while_the_driver_waits_for_a_wave_still_logs_what_the_calls_caused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Interrupted(ThreadPoolExecutor):
        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            super().__exit__(exc_type, exc, traceback)  # the calls finish...
            raise KeyboardInterrupt  # ...then the signal arrives

    world = World(
        script=[use(("t1", "scan"), ("t2", "scan")), reply()], together={"scan"}
    )
    meet = threading.Barrier(2, timeout=5)

    def tool(name: str, args: JsonObject) -> ToolResult:
        world.notices.noted(f"{args['q']}.md")
        meet.wait()
        return ToolResult(True, str(args["q"]))

    monkeypatch.setattr("loomcore.engine.ThreadPoolExecutor", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        driver(world, tmp_path, tool).run(TASK)
    events = log(tmp_path)
    assert [e.tool_use_id for e in events if isinstance(e, NoteWritten)] == ["t1", "t2"]
    assert not any(isinstance(e, ToolCompleted) for e in events)
