"""Snapshots of the replayed state, so a long log need not be folded from zero."""

import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from loomcore.events import Event
from loomcore.replay import RunState, replay


class StaleSnapshotError(Exception):
    """The snapshot covers more events than the log has; discard it and replay."""


class Snapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    applied: int  # how many events of the run's chain the state already folds in
    state: RunState


def take_snapshot(chain: Sequence[Event]) -> Snapshot:
    return Snapshot(applied=len(chain), state=replay(chain))


def replay_from(snapshot: Snapshot, chain: Sequence[Event]) -> RunState:
    """The state of `chain`, folding only the events the snapshot has not seen."""
    if snapshot.applied > len(chain):
        raise StaleSnapshotError(
            f"snapshot covers {snapshot.applied} events but the log has {len(chain)}"
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


def read_snapshot(path: Path) -> Snapshot:
    return Snapshot.model_validate_json(path.read_bytes())
