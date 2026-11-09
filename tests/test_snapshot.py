"""f04 (part): snapshots are atomic and replay can start from one."""

import os
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


def test_a_snapshot_from_another_chain_of_the_same_length_is_stale() -> None:
    other = [*CHAIN[:-1], make("run_paused", 4, run="r2")]
    with pytest.raises(StaleSnapshotError, match="different chain"):
        replay_from(take_snapshot(CHAIN), other)


def test_writing_a_snapshot_syncs_the_file_and_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        synced.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    write_snapshot(tmp_path / "s1.snapshot.json", take_snapshot(CHAIN))
    assert len(synced) == 2  # the temporary file, then the directory


def test_a_snapshot_whose_last_event_changed_content_is_stale() -> None:
    regrown = [*CHAIN[:-1], make("run_paused", 4, resume_from=2)]
    with pytest.raises(StaleSnapshotError, match="different chain"):
        replay_from(take_snapshot(CHAIN), regrown)


def test_a_failed_snapshot_write_leaves_the_old_file_and_no_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "s1.snapshot.json"
    write_snapshot(path, take_snapshot(CHAIN[:2]))

    def failing_fsync(descriptor: int) -> None:
        raise OSError(5, "I/O error")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="I/O error"):
        write_snapshot(path, take_snapshot(CHAIN))
    monkeypatch.undo()
    assert read_snapshot(path).applied == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s1.snapshot.json"]


def test_snapshots_of_paths_that_differ_only_in_suffix_use_separate_temporaries(
    tmp_path: Path,
) -> None:
    write_snapshot(tmp_path / "s1.a", take_snapshot(CHAIN[:2]))
    write_snapshot(tmp_path / "s1.b", take_snapshot(CHAIN))
    assert read_snapshot(tmp_path / "s1.a").applied == 2
    assert read_snapshot(tmp_path / "s1.b").applied == len(CHAIN)


def test_a_snapshot_written_for_an_older_run_state_is_stale() -> None:
    older = take_snapshot(CHAIN).model_copy(update={"state_fields": ("event_count",)})
    with pytest.raises(StaleSnapshotError, match="older RunState"):
        replay_from(older, CHAIN)


def test_an_unreadable_snapshot_file_is_stale_not_a_validation_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s1.snapshot.json"
    path.write_text('{"applied": 1, "surprise": true}')
    with pytest.raises(StaleSnapshotError, match="discard it"):
        read_snapshot(path)
