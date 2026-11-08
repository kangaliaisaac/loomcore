"""f05: the spend log, one line per model call."""

import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from factories import make

from loomcore.events import ModelCall
from loomcore.spend import (
    SpendLine,
    append_spend,
    month_to_date,
    read_spend,
    repair_spend,
    run_key,
    spend_line,
    spend_shortfall,
)

NOW = datetime(2026, 11, 15, tzinfo=UTC)


def line(usd: str, run: str = "s1/r1", at: datetime = NOW) -> SpendLine:
    return SpendLine(at=at, usd=Decimal(usd), run=run, note="")


def test_a_model_call_becomes_one_line_keyed_by_session_and_run() -> None:
    call = make("model_call", 4, cost_usd=0.0129)
    assert isinstance(call, ModelCall)
    result = spend_line(call)
    assert result.run == "s1/r1"
    assert result.usd == Decimal("0.0129")
    assert result.at == call.timestamp


def test_lines_round_trip_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    append_spend(path, line("0.5"))
    append_spend(path, line("0.25", run="s1/r2"))
    result = read_spend(path)
    assert [x.usd for x in result.lines] == [Decimal("0.5"), Decimal("0.25")]
    assert result.skipped == ()


def test_a_record_starts_and_ends_with_a_newline(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    append_spend(path, line("1"))
    raw = path.read_bytes()
    assert raw.startswith(b"\n")
    assert raw.endswith(b"\n")
    assert set(json.loads(raw)) == {"at", "usd", "run", "note"}


def test_a_torn_append_never_swallows_the_next_line(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    append_spend(path, line("1"))
    with path.open("ab") as file:
        file.write(b'{"at": "2026-11-15T00:00')  # a crash mid-record
    append_spend(path, line("2", run="s1/r2"))
    result = read_spend(path)
    assert [x.usd for x in result.lines] == [Decimal("1"), Decimal("2")]
    assert result.skipped == ('{"at": "2026-11-15T00:00',)


def test_blank_lines_are_ignored_without_being_reported(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    append_spend(path, line("1"))
    append_spend(path, line("2"))
    assert read_spend(path).skipped == ()
    assert len(read_spend(path).lines) == 2


def test_month_to_date_counts_this_utc_month_and_leaves_out_the_runs_own_lines() -> (
    None
):
    lines = [
        line("1", run="s1/r1"),  # this run: excluded
        line("2", run="s1/r2"),
        line("4", run="s2/r1"),
        line(
            "8", run="s1/r2", at=datetime(2026, 10, 31, 23, 59, tzinfo=UTC)
        ),  # last month
    ]
    assert month_to_date(lines, "s1/r1", NOW) == Decimal("6")


def test_the_month_is_judged_in_utc() -> None:
    late_in_utc = datetime(
        2026, 11, 1, 1, tzinfo=timezone(timedelta(hours=5))
    )  # Oct 31 UTC
    assert month_to_date([line("3", run="x/y", at=late_in_utc)], "s1/r1", NOW) == 0


def calls(*costs: float) -> list[ModelCall]:
    made = [make("model_call", i, cost_usd=c) for i, c in enumerate(costs)]
    return [m for m in made if isinstance(m, ModelCall)]


def test_no_shortfall_when_every_call_has_its_line() -> None:
    events = calls(0.5, 0.25)
    assert spend_shortfall(events, [spend_line(c) for c in events], "s1/r1") == 0


def test_the_shortfall_is_the_missing_calls_cost() -> None:
    events = calls(0.5, 0.25)
    assert spend_shortfall(events, [spend_line(events[0])], "s1/r1") == Decimal("0.25")


def test_other_runs_calls_and_lines_do_not_count_toward_the_shortfall() -> None:
    mine = calls(0.5)
    other = make("model_call", 0, run="r2", cost_usd=9)
    lines = [line("100", run="s1/r2")]
    assert spend_shortfall([*mine, other], lines, "s1/r1") == Decimal("0.5")


def test_resume_writes_the_missing_line_once(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    events = calls(0.5, 0.25)
    append_spend(path, spend_line(events[0]))  # the crash came before the second
    repair_spend(path, events, "s1/r1", NOW)
    repair_spend(path, events, "s1/r1", NOW)  # a second resume finds nothing to do
    written = read_spend(path).lines
    assert [x.usd for x in written] == [Decimal("0.5"), Decimal("0.25")]
    assert written[-1].note == "resume repair"
    assert month_to_date(written, "other/run", NOW) == Decimal("0.75")


def test_repair_creates_the_log_when_a_crash_came_before_the_first_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spend.jsonl"
    repair_spend(path, calls(0.5), "s1/r1", NOW)
    assert [x.usd for x in read_spend(path).lines] == [Decimal("0.5")]


def test_run_key_joins_session_and_run() -> None:
    assert run_key("s1", "r1") == "s1/r1"
