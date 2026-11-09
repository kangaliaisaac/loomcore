"""Snapshots of the replayed state, so a long log need not be folded from zero."""

import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from loomcore.events import Event
from loomcore.replay import RunState, replay


class StaleSnapshotError(Exception):
    """The snapshot does not belong to this chain; discard it and replay."""


class Snapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    applied: int  # how many events of the run's chain the state already folds in
    tip: str  # "<runId>/<seq>" of the last of them, to tell a different chain
    state: RunState


def _tip(chain: Sequence[Event], applied: int) -> str:
    return f"{chain[applied - 1].run_id}/{chain[applied - 1].seq}" if applied else ""


def take_snapshot(chain: Sequence[Event]) -> Snapshot:
    return Snapshot(
        applied=len(chain), tip=_tip(chain, len(chain)), state=replay(chain)
    )


def replay_from(snapshot: Snapshot, chain: Sequence[Event]) -> RunState:
    """The state of `chain`, folding only the events the snapshot has not seen."""
    if snapshot.applied > len(chain):
        raise StaleSnapshotError(
            f"snapshot covers {snapshot.applied} events but the log has {len(chain)}"
        )
    if _tip(chain, snapshot.applied) != snapshot.tip:
        raise StaleSnapshotError(
            f"snapshot was taken on a different chain (its last event was "
            f"{snapshot.tip}); discard it and replay"
        )
    return replay(chain[snapshot.applied :], snapshot.state)


def write_snapshot(path: Path, snapshot: Snapshot) -> None:
    """Write beside the target and move into place, so a reader never sees half."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        file.write(snapshot.model_dump_json().encode())
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)  # else a power loss can lose the rename


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_snapshot(path: Path) -> Snapshot:
    return Snapshot.model_validate_json(path.read_bytes())
