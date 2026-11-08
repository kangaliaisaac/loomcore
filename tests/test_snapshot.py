"""f04 (part): snapshots are atomic and replay can start from one."""

from pathlib import Path

import pytest
from factories import calling, make

from loomcore.replay import replay
from loomcore.snapshot import (
    StaleSnapshotError,
    read_snapshot,
    replay_from,
    take_snapshot,
    write_snapshot,
)

CHAIN = [
    make("run_started"),
    make("model_call", 1, message=calling("a")),
    make("approval_requested", 2),
    make("tool_completed", 3, tool_use_id="a"),
    make("run_paused", 4),
]


def test_replaying_from_a_snapshot_equals_replaying_everything() -> None:
    for cut in range(len(CHAIN) + 1):
        snapshot = take_snapshot(CHAIN[:cut])
        assert replay_from(snapshot, CHAIN) == replay(CHAIN)


def test_a_snapshot_survives_the_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "s1.snapshot.json"
    snapshot = take_snapshot(CHAIN)
    write_snapshot(path, snapshot)
    assert read_snapshot(path) == snapshot
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s1.snapshot.json"]


def test_writing_a_snapshot_replaces_the_old_one_whole(tmp_path: Path) -> None:
    path = tmp_path / "s1.snapshot.json"
    write_snapshot(path, take_snapshot(CHAIN[:2]))
    write_snapshot(path, take_snapshot(CHAIN))
    assert read_snapshot(path).applied == len(CHAIN)


def test_a_snapshot_ahead_of_the_log_is_stale() -> None:
    with pytest.raises(StaleSnapshotError):
        replay_from(take_snapshot(CHAIN), CHAIN[:2])
