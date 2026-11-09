"""Snapshots of the replayed state, so a long log need not be folded from zero."""

import hashlib
import os
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from loomcore.events import Event, dump_event
from loomcore.replay import RunState, replay


class StaleSnapshotError(Exception):
    """The snapshot does not belong to this chain; discard it and replay."""


class Snapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    applied: int  # how many events of the run's chain the state already folds in
    tip: str  # sha256 of the last of them, to tell a chain that regrew differently
    state_fields: tuple[str, ...]  # RunState's fields at writing; a new one stales it
    state: RunState


def _tip(chain: Sequence[Event], applied: int) -> str:
    if not applied:
        return ""
    return hashlib.sha256(dump_event(chain[applied - 1]).encode()).hexdigest()


def _state_fields() -> tuple[str, ...]:
    return tuple(f.name for f in fields(RunState))


def take_snapshot(chain: Sequence[Event]) -> Snapshot:
    return Snapshot(
        applied=len(chain),
        tip=_tip(chain, len(chain)),
        state_fields=_state_fields(),
        state=replay(chain),
    )


def replay_from(snapshot: Snapshot, chain: Sequence[Event]) -> RunState:
    """The state of `chain`, folding only the events the snapshot has not seen."""
    if snapshot.applied > len(chain):
        raise StaleSnapshotError(
            f"snapshot covers {snapshot.applied} events but the log has {len(chain)}"
        )
    if snapshot.state_fields != _state_fields():
        raise StaleSnapshotError(
            "snapshot was written for an older RunState; discard it and replay"
        )
    if _tip(chain, snapshot.applied) != snapshot.tip:
        raise StaleSnapshotError(
            "snapshot was taken on a different chain (its last event differs); "
            "discard it and replay"
        )
    return replay(chain[snapshot.applied :], snapshot.state)


def write_snapshot(path: Path, snapshot: Snapshot) -> None:
    """Write beside the target and move into place, so a reader never sees half."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as file:
            file.write(snapshot.model_dump_json().encode())
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)  # else a power loss can lose the rename


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_snapshot(path: Path) -> Snapshot:
    try:
        return Snapshot.model_validate_json(path.read_bytes())
    except ValidationError as error:
        raise StaleSnapshotError(
            f"{path} is not a snapshot this version reads; discard it and replay"
        ) from error
