"""The spend log: one line per model call, shared by every run (spec section 2).

Each record is a single `os.write` on an `O_APPEND` descriptor, so concurrent runs
need no lock, and a leading newline keeps a torn record from absorbing the next.
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from loomcore.events import Event, ModelCall, Usd
from loomcore.replay import month_key


class SpendLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    at: AwareDatetime
    usd: Usd
    run: str  # "<sessionId>/<runId>"
    seq: int = Field(ge=0)  # the model_call's seq in that run: what repair matches on
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
        seq=call.seq,
        note="model_call",
    )


def append_spend(path: Path, line: SpendLine) -> None:
    record = ("\n" + line.model_dump_json() + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if os.write(descriptor, record) != len(record):
            raise OSError(f"short write to {path}; the record may be torn")
    finally:
        os.close(descriptor)


def read_spend(path: Path) -> SpendRead:
    """Every parseable line; a missing file (no run has spent yet) reads as empty."""
    lines: list[SpendLine] = []
    skipped: list[str] = []
    if not path.exists():
        return SpendRead((), ())
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
            if line.run != run and month_key(line.at) == month_key(now)
        ),
        Decimal(0),
    )


@dataclass(frozen=True)
class MonthSpend:
    usd: Decimal
    skipped: tuple[str, ...]  # lines the total could not count; the caller reports them


def month_spend(spend: SpendRead, run: str, now: datetime) -> MonthSpend:
    """`month_to_date` that hands back the unparseable lines with the total.

    Taking the whole `SpendRead` means the driver cannot read the total and forget
    that some lines were not counted.
    """
    return MonthSpend(month_to_date(spend.lines, run, now), spend.skipped)


def missing_spend_lines(
    events: Iterable[Event], lines: Iterable[SpendLine], run: str
) -> list[SpendLine]:
    """The run's model calls that have no spend line, each dated at its call.

    Non-zero after a crash between a model_call and its spend line. Dating the
    line at the call keeps a call made in one month out of the next month's total.
    """
    recorded = {line.seq for line in lines if line.run == run}
    return [
        line
        for line in (
            spend_line(e)
            for e in events
            if isinstance(e, ModelCall) and run_key(e.session_id, e.run_id) == run
        )
        if line.seq not in recorded
    ]


def repair_spend(path: Path, events: Iterable[Event], run: str) -> tuple[str, ...]:
    """On resume, write the lines a crash lost, so the run's lines match its calls.

    Returns the log's unparseable lines, which the caller must surface. The caller
    holds the session lock: two unlocked resumes would both write the lines.
    """
    existing = read_spend(path)
    for line in missing_spend_lines(events, existing.lines, run):
        append_spend(path, line)
    return existing.skipped
