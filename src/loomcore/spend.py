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
        os.write(descriptor, record)
    finally:
        os.close(descriptor)


def read_spend(path: Path) -> SpendRead:
    lines: list[SpendLine] = []
    skipped: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
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


def spend_shortfall(
    events: Iterable[Event], lines: Iterable[SpendLine], run: str
) -> Decimal:
    """What the run's model calls cost beyond what its spend lines record.

    Non-zero after a crash between a model_call and its spend line.
    """
    calls = (
        e
        for e in events
        if isinstance(e, ModelCall) and run_key(e.session_id, e.run_id) == run
    )
    logged = sum((call.cost_usd for call in calls), Decimal(0))
    recorded = sum((line.usd for line in lines if line.run == run), Decimal(0))
    return logged - recorded


def repair_spend(path: Path, events: Iterable[Event], run: str, now: datetime) -> None:
    """On resume, write the one line that makes the run's lines match its calls."""
    existing = read_spend(path).lines if path.exists() else ()
    shortfall = spend_shortfall(events, existing, run)
    if shortfall > 0:
        append_spend(
            path, SpendLine(at=now, usd=shortfall, run=run, note="resume repair")
        )


def _month(moment: datetime) -> tuple[int, int]:
    utc = moment.astimezone(UTC)
    return utc.year, utc.month
