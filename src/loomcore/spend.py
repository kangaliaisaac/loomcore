"""The spend log: one line per model call, shared by every run (spec section 2).

Each record is a single `os.write` on an `O_APPEND` descriptor, so concurrent runs
need no lock, and a leading newline keeps a torn record from absorbing the next.
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from loomcore.events import Event, ModelCall, Usd


class SpendLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    at: AwareDatetime
    usd: Usd
    run: str  # "<sessionId>/<runId>"
    note: str = ""


@dataclass(frozen=True)
class SpendRead:
    lines: tuple[SpendLine, ...]
    skipped: tuple[str, ...]  # lines that did not parse, reported and ignored


def run_key(session_id: str, run_id: str) -> str:
    return f"{session_id}/{run_id}"


def spend_line(call: ModelCall) -> SpendLine:
    return SpendLine(
        at=call.timestamp,
        usd=call.cost_usd,
        run=run_key(call.session_id, call.run_id),
        note=f"model_call seq {call.seq}",
    )


def append_spend(path: Path, line: SpendLine) -> None:
    record = ("\n" + line.model_dump_json() + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if os.write(descriptor, record) != len(record):
            raise OSError(f"short write to {path}; the record may be torn")
    finally:
        os.close(descriptor)


def read_spend(path: Path) -> SpendRead:
    lines: list[SpendLine] = []
    skipped: list[str] = []
    for raw in path.read_text(errors="replace").split("\n"):
        if not raw.strip():
            continue
        try:
            lines.append(SpendLine.model_validate_json(raw))
        except ValidationError:
            skipped.append(raw)
    return SpendRead(tuple(lines), tuple(skipped))


def month_to_date(lines: Iterable[SpendLine], run: str, now: datetime) -> Decimal:
    """Spend in the UTC month of `now`, leaving out this run's own lines."""
    return sum(
        (
            line.usd
            for line in lines
            if line.run != run and _month(line.at) == _month(now)
        ),
        Decimal(0),
    )


def missing_spend_lines(
    events: Iterable[Event], lines: Iterable[SpendLine], run: str
) -> list[SpendLine]:
    """The run's model calls that have no spend line, each dated at its call.

    Non-zero after a crash between a model_call and its spend line. Dating the
    line at the call keeps a call made in one month out of the next month's total.
    """
    recorded = {line.note for line in lines if line.run == run}
    return [
        line
        for line in (
            spend_line(e)
            for e in events
            if isinstance(e, ModelCall) and run_key(e.session_id, e.run_id) == run
        )
        if line.note not in recorded
    ]


def repair_spend(path: Path, events: Iterable[Event], run: str) -> None:
    """On resume, write the lines a crash lost, so the run's lines match its calls.

    The caller holds the session lock: two unlocked resumes would both write them.
    """
    existing = read_spend(path).lines if path.exists() else ()
    for line in missing_spend_lines(events, existing, run):
        append_spend(path, line)


def _month(moment: datetime) -> tuple[int, int]:
    utc = moment.astimezone(UTC)
    return utc.year, utc.month
