"""Spans, derived from the event log and never written (spec 3.8).

A run is a trace: a `run` span, a `turn` span for each `turn_started`, and under a
turn a span for its model call and for each tool call. Attribute names follow the
OpenTelemetry `gen_ai.*` conventions where one exists, and `loomcore.*` otherwise.
Ids come from the log (session, run, seq or tool_use_id), so the same log always
gives the same spans.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from pydantic import JsonValue

from loomcore.events import (
    Event,
    JsonObject,
    ModelCall,
    RunFinished,
    ToolCompleted,
    ToolRequested,
    TurnStarted,
)
from loomcore.observability.cost import Tokens, find_usage

Kind = Literal["run", "turn", "model_call", "tool_call"]
_RANK: dict[Kind, int] = {"run": 0, "turn": 1, "model_call": 2, "tool_call": 2}


@dataclass(frozen=True)
class Span:
    id: str
    parent: str | None
    kind: Kind
    name: str
    start: datetime
    end: datetime
    attributes: dict[str, JsonValue]

    @property
    def duration_ms(self) -> int:
        return round((self.end - self.start).total_seconds() * 1000)


def spans(events: Iterable[Event]) -> list[Span]:
    """Every run's spans, runs in the order they first appear, each run's spans by
    start time (a parent before its children)."""
    runs: dict[str, list[Event]] = {}
    for event in events:
        runs.setdefault(event.run_id, []).append(event)
    return [span for log in runs.values() for span in _run_spans(log)]


def _id(base: str, kind: Kind, key: object) -> str:
    return f"{base}/{kind}:{key}"


def _run_spans(log: list[Event]) -> list[Span]:
    first, last = log[0], log[-1]
    base = f"{first.session_id}/{first.run_id}"
    finished = last.reason if isinstance(last, RunFinished) else None
    run = Span(
        _id(base, "run", first.run_id),
        None,
        "run",
        "run",
        first.timestamp,
        last.timestamp,
        {"loomcore.run.id": first.run_id, "loomcore.run.reason": finished},
    )
    found = [run, *_turns(base, run.id, log)]
    turn = run.id
    requested: dict[str, ToolRequested] = {}
    for event in log:
        match event:
            case TurnStarted():
                turn = _id(base, "turn", event.seq)
            case ModelCall():
                found.append(_model_span(base, turn, event))
            case ToolRequested():
                requested[event.tool_use_id] = event
            case ToolCompleted() if event.tool_use_id in requested:
                found.append(
                    _tool_span(base, turn, requested[event.tool_use_id], event)
                )
    return sorted(found, key=lambda s: (s.start, _RANK[s.kind]))


def _turns(base: str, parent: str, log: list[Event]) -> list[Span]:
    """A turn lasts from its start to the next turn's start, or the run's last event.
    The driver writes `turn_started` after the model has answered, just before the
    `model_call`, so a turn starts when its model call did, if that is earlier."""
    starts = [e for e in log if isinstance(e, TurnStarted)]
    begins = [_turn_start(event, log) for event in starts]
    ends = [*begins[1:], log[-1].timestamp]
    return [
        Span(
            _id(base, "turn", event.seq),
            parent,
            "turn",
            f"turn {event.turn}",
            begin,
            end,
            {"loomcore.turn": event.turn},
        )
        for event, begin, end in zip(starts, begins, ends, strict=True)
    ]


def _turn_start(turn: TurnStarted, log: list[Event]) -> datetime:
    """The earlier of the turn's own event and its model call's start."""
    call = next((e for e in log if isinstance(e, ModelCall) and e.seq > turn.seq), None)
    if call is None or any(
        isinstance(e, TurnStarted) and turn.seq < e.seq < call.seq for e in log
    ):
        return turn.timestamp
    return min(
        turn.timestamp, call.timestamp - timedelta(milliseconds=call.duration_ms)
    )


def _model_span(base: str, parent: str, call: ModelCall) -> Span:
    """The call ends at its event and started `duration_ms` before."""
    start = call.timestamp - timedelta(milliseconds=call.duration_ms)
    message = call.message
    found = find_usage(message)
    tokens = found.tokens if found else None
    attributes: dict[str, JsonValue] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.response.model": message.get("model"),
        "gen_ai.response.id": message.get("id"),
        "gen_ai.response.finish_reasons": [message.get("stop_reason")],
        "loomcore.cost_usd": format(call.cost_usd, "f"),  # a string: no float price
        "loomcore.pricing_version": call.pricing_version,
    }
    if tokens is not None:
        attributes |= _token_attributes(tokens)
    return Span(
        _id(base, "model_call", call.seq),
        parent,
        "model_call",
        f"chat {message.get('model') or ''}".strip(),
        start,
        call.timestamp,
        attributes,
    )


def _token_attributes(tokens: Tokens) -> JsonObject:
    return {
        "gen_ai.usage.input_tokens": tokens.prompt,
        "gen_ai.usage.output_tokens": tokens.output,
        "gen_ai.usage.cache_read.input_tokens": tokens.cache_read,
        "gen_ai.usage.cache_creation.input_tokens": (
            tokens.cache_write_5m + tokens.cache_write_1h
        ),
    }


def _tool_span(
    base: str, parent: str, asked: ToolRequested, done: ToolCompleted
) -> Span:
    """The call ends at its completion. Its `duration_ms` is the clock the tool's
    port measured; a call that logs none (0) falls back to the time between the
    request and the completion."""
    start = (
        done.timestamp - timedelta(milliseconds=done.duration_ms)
        if done.duration_ms
        else asked.timestamp
    )
    return Span(
        _id(base, "tool_call", done.tool_use_id),
        parent,
        "tool_call",
        f"execute_tool {asked.name}",
        start,
        done.timestamp,
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": asked.name,
            "gen_ai.tool.call.id": done.tool_use_id,
            "loomcore.tool.ok": done.ok,
            "loomcore.tool.cached": done.cached,
        },
    )
