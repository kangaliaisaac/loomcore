"""Spans, derived from the event log and never written (spec 3.8).

A run is a trace: a `run` span, a `turn` span for each `turn_started`, and under a
turn a span for its model call and for each tool call. Attribute names follow the
OpenTelemetry `gen_ai.*` conventions where one exists, and `loomcore.*` otherwise.
Ids come from the log (session, run, seq or tool_use_id), so the same log always
gives the same spans.
"""

import logging
from collections.abc import Iterable, Iterator
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
from loomcore.replay import finish_call

_LOG = logging.getLogger(__name__)
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
    found = [
        run,
        *_turns(base, run.id, log),
        *(
            _model_span(base, turn, event)
            for turn, event in _in_turns(base, run.id, log)
            if isinstance(event, ModelCall)
        ),
        *_tool_spans(base, run.id, log),
    ]
    return sorted(found, key=lambda s: (s.start, _RANK[s.kind]))


def _in_turns(base: str, run_id: str, log: list[Event]) -> Iterator[tuple[str, Event]]:
    """Each event with the id of the turn it belongs to (the run, before the first)."""
    turn = run_id
    for event in log:
        if isinstance(event, TurnStarted):
            turn = _id(base, "turn", event.seq)
        yield turn, event


def _tool_spans(base: str, run_id: str, log: list[Event]) -> list[Span]:
    """A span for each call that was requested: it ends at its completion, or, for a
    call with none (a crash mid-tool, or a run paused for approval), at the run's
    last event, marked open. A
    completion with no request has no span (spec 3.8); one that looks like a call
    that ran is reported."""
    asked: dict[str, tuple[str, ToolRequested]] = {}
    found: list[Span] = []
    orphans: list[ToolCompleted] = []
    for turn, event in _in_turns(base, run_id, log):
        if isinstance(event, ToolRequested):
            asked[event.tool_use_id] = (turn, event)
        elif isinstance(event, ToolCompleted):
            hit = asked.pop(event.tool_use_id, None)
            if hit is None:
                orphans.append(event)
            else:
                found.append(_tool_span(base, hit[0], hit[1], event))
    found += [_open_tool_span(base, t, e, log[-1].timestamp) for t, e in asked.values()]
    _report_orphans(orphans, log)
    return found


def _report_orphans(orphans: list[ToolCompleted], log: list[Event]) -> None:
    """A completion with no request is by design for the `finish` call and for a call
    that did not run (`ok=false`, no time spent); any other is a damaged log. This is
    the one place the core logs, so a damaged log is not silent (spec 3.8); a real call
    that failed in under a millisecond looks like "did not run" and is not reported."""
    if not orphans:
        return
    finishing = {
        found[0]
        for event in log
        if isinstance(event, ModelCall) and (found := finish_call(event.message))
    }
    odd = [
        done.tool_use_id
        for done in orphans
        if done.tool_use_id not in finishing and (done.ok or done.duration_ms > 0)
    ]
    if odd:
        _LOG.warning(
            "tool_completed with no tool_requested, so no span: %s", ", ".join(odd)
        )


def _turns(base: str, parent: str, log: list[Event]) -> list[Span]:
    """A turn lasts from its start to the next turn's start, or the run's last event.
    The driver logs `turn_started` after the model has answered, just before the
    `model_call`, so a turn starts when its model call did, if that is earlier."""
    starts = [e for e in log if isinstance(e, TurnStarted)]
    begins = _turn_starts(log)
    ends = [*(begins[e.seq] for e in starts[1:]), log[-1].timestamp]
    return [
        Span(
            _id(base, "turn", event.seq),
            parent,
            "turn",
            f"turn {event.turn}",
            begins[event.seq],
            end,
            {"loomcore.turn": event.turn},
        )
        for event, end in zip(starts, ends, strict=True)
    ]


def _turn_starts(log: list[Event]) -> dict[int, datetime]:
    """Each turn's start by its `turn_started` seq, in one pass: the earlier of its
    own event and the start of the model call that follows it (if no other turn
    began first)."""
    begins: dict[int, datetime] = {}
    waiting: TurnStarted | None = None
    for event in log:
        if isinstance(event, TurnStarted):
            begins[event.seq] = event.timestamp
            waiting = event
        elif isinstance(event, ModelCall) and waiting is not None:
            call_start = event.timestamp - timedelta(milliseconds=event.duration_ms)
            begins[waiting.seq] = min(begins[waiting.seq], call_start)
            waiting = None
    return begins


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
        # A string, normalised: the log stores the cost as a float, so an event has
        # "2.50" in memory and "2.5" read back, and a span must not tell them apart.
        "loomcore.cost_usd": format(call.cost_usd.normalize(), "f"),
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


def _open_tool_span(
    base: str, parent: str, asked: ToolRequested, end: datetime
) -> Span:
    """A call that was requested and not completed when the log ended (a crash
    mid-tool, or a run paused for an approval): it is open until the last event."""
    return Span(
        _id(base, "tool_call", asked.tool_use_id),
        parent,
        "tool_call",
        f"execute_tool {asked.name}",
        asked.timestamp,
        end,
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": asked.name,
            "gen_ai.tool.call.id": asked.tool_use_id,
            "loomcore.tool.ok": None,
            "loomcore.tool.cached": None,
            "loomcore.tool.open": True,
        },
    )
