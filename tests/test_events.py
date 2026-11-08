"""f01: every kind round-trips to JSON and back; unknown fields are rejected."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, get_args

import pytest
from factories import PAYLOADS, make
from pydantic import ValidationError

from loomcore.events import Event, ModelCall, dump_event, parse_event


def test_every_kind_in_the_spec_has_a_sample() -> None:
    kinds = {
        get_args(m.model_fields["kind"].annotation)[0]
        for m in get_args(get_args(Event)[0])
    }
    assert kinds == set(PAYLOADS)
    assert len(kinds) == 20


@pytest.mark.parametrize("kind", PAYLOADS)
def test_kind_round_trips_through_json(kind: str) -> None:
    event = make(kind)
    assert parse_event(dump_event(event)) == event
    assert json.loads(dump_event(event))["kind"] == kind


@pytest.mark.parametrize("kind", PAYLOADS)
def test_unknown_fields_are_rejected(kind: str) -> None:
    line = json.loads(dump_event(make(kind)))
    line["surprise"] = 1
    with pytest.raises(ValidationError):
        parse_event(json.dumps(line))


def test_unknown_kind_is_rejected() -> None:
    line = json.loads(dump_event(make("turn_started")))
    line["kind"] = "turn_ended"
    with pytest.raises(ValidationError):
        parse_event(json.dumps(line))


def test_envelope_uses_the_spec_field_names_and_millisecond_utc() -> None:
    line = json.loads(dump_event(make("turn_started", seq=2)))
    assert line["schema"] == "loomcore.event.v1"
    assert line["sessionId"] == "s1"
    assert line["runId"] == "r1"
    assert line["timestamp"] == "2026-11-03T08:00:02.000Z"


def test_a_timestamp_without_a_zone_is_rejected() -> None:
    line = json.loads(dump_event(make("turn_started")))
    line["timestamp"] = "2026-11-03T08:00:00.000"
    with pytest.raises(ValidationError):
        parse_event(json.dumps(line))


def test_cost_is_a_json_number_and_a_decimal_in_memory() -> None:
    event = make("model_call", cost_usd=0.0129)
    assert json.loads(dump_event(event))["cost_usd"] == 0.0129
    parsed = parse_event(dump_event(event))
    assert isinstance(parsed, ModelCall)
    assert str(parsed.cost_usd) == "0.0129"


def test_a_fork_is_written_with_the_key_from() -> None:
    line = json.loads(dump_event(make("run_forked", hint="try again")))
    assert line["from"] == {"runId": "r1", "seq": 0}


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("guard_decision", {"stage": "post", "decision": "ask"}),
        ("guard_decision", {"stage": "pre", "decision": "redact"}),
        ("run_paused", {"reason": "limit"}),
        ("run_paused", {"reason": "awaiting_approval", "limit": "max_turns"}),
        ("run_resumed", {"reason": "limit"}),
    ],
)
def test_illegal_combinations_cannot_be_built(
    kind: str, payload: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        make(kind, **payload)


def test_legal_limit_pause_and_resume_build() -> None:
    make("run_paused", reason="limit", limit="max_turns")
    make("run_resumed", reason="limit", limits={"max_turns": 60})


def test_an_event_built_with_microseconds_equals_its_read_back() -> None:
    now = datetime(2026, 11, 3, 8, 0, 0, 123456, tzinfo=UTC)
    event = make("turn_started", timestamp=now.isoformat())
    assert event.timestamp.microsecond == 123000
    assert parse_event(dump_event(event)) == event


def test_money_stays_a_decimal_in_python_mode() -> None:
    dumped = make("model_call", cost_usd=0.5).model_dump()
    assert dumped["cost_usd"] == Decimal("0.5")
