"""f04: random appends, snapshots and replays agree; a cut file stays safe."""

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from strategies import events_for, logs

from loomcore.eventlog import open_writer, read_events
from loomcore.events import Event, dump_event
from loomcore.replay import RunState, replay
from loomcore.snapshot import Snapshot, replay_from, take_snapshot


class LogMachine(RuleBasedStateMachine):
    """Drives a real file and an in-memory model with the same operations."""

    def __init__(self) -> None:
        super().__init__()
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "s1.jsonl"
        self.model: list[Event] = []
        self.snapshot: Snapshot | None = None

    def teardown(self) -> None:
        self._dir.cleanup()

    @rule(data=st.data(), count=st.integers(min_value=1, max_value=4))
    def append(self, data: st.DataObject, count: int) -> None:
        with open_writer(self.path) as writer:
            for _ in range(count):
                event = data.draw(events_for("r1", len(self.model)))
                writer.append(event)
                self.model.append(event)

    @rule()
    def snapshot_now(self) -> None:
        self.snapshot = take_snapshot(self.model)

    @rule(garbage=st.binary(max_size=30).filter(lambda b: b"\n" not in b))
    @precondition(lambda self: bool(self.model))
    def crash_mid_write(self, garbage: bytes) -> None:
        with self.path.open("ab") as file:
            file.write(garbage)

    @invariant()
    def file_matches_the_model(self) -> None:
        if self.model or self.path.exists():
            assert read_events(self.path).events == tuple(self.model)

    @invariant()
    def replay_from_a_snapshot_agrees_with_the_fold(self) -> None:
        if self.snapshot is not None:
            assert replay_from(self.snapshot, self.model) == replay(self.model)


TestLogMachine = LogMachine.TestCase
TestLogMachine.settings = settings(
    max_examples=40, stateful_step_count=15, deadline=None
)


@given(log=logs(), cut=st.data())
@settings(max_examples=80, deadline=None)
def test_cutting_the_file_at_any_byte_replays_to_an_earlier_state(
    log: list[Event], cut: st.DataObject
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "s1.jsonl"
        lines = [dump_event(event).encode() + b"\n" for event in log]
        whole = b"".join(lines)
        point = cut.draw(st.integers(min_value=0, max_value=len(whole)))
        path.write_bytes(whole[:point])

        result = read_events(path)

        survived = sum(1 for end in _line_ends(lines) if end <= point)
        assert result.events == tuple(log[:survived])
        assert replay(result.events) == replay(log[:survived])
        assert (result.torn_tail is not None) == (point not in {0, *_line_ends(lines)})

        # and the log is usable again: the next append lands on a clean line
        with open_writer(path) as writer:
            for event in log[survived:]:
                writer.append(event)
        assert read_events(path).events == tuple(log)


def _line_ends(lines: list[bytes]) -> list[int]:
    ends, total = [], 0
    for line in lines:
        total += len(line)
        ends.append(total)
    return ends


@given(log=logs())
@settings(max_examples=60, deadline=None)
def test_snapshots_and_replays_at_every_cut_agree_with_the_fold(
    log: list[Event],
) -> None:
    full = replay(log)
    assert full.event_count == len(log)
    for cut in range(len(log) + 1):
        assert replay_from(take_snapshot(log[:cut]), log) == full
        assert replay(log[cut:], replay(log[:cut])) == full
    assert replay([]) == RunState()
