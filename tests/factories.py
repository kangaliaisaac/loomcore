"""Build events for tests from just the payload that matters."""

import json
from datetime import UTC, datetime, timedelta

from pydantic import JsonValue

from loomcore.events import Event, JsonObject, parse_event

SESSION = "s1"
START = datetime(2026, 11, 3, 8, 0, tzinfo=UTC)

# The smallest valid payload of every kind.
PAYLOADS: dict[str, JsonObject] = {
    "run_started": {"task": "t", "config": {}, "limits": {}, "config_digest": "d"},
    "run_forked": {"from": {"runId": "r1", "seq": 0}},
    "turn_started": {"turn": 1},
    "model_call": {
        "type": "assistant",
        "message": {"content": []},
        "duration_ms": 5,
        "cost_usd": 0.01,
        "pricing_version": "2026-10",
    },
    "tool_requested": {"tool_use_id": "t1", "name": "n", "input": {}, "call_hash": "h"},
    "tool_completed": {
        "tool_use_id": "t1",
        "ok": True,
        "content": "ok",
        "duration_ms": 1,
        "cached": False,
    },
    "guard_decision": {
        "stage": "pre",
        "tool_use_id": "t1",
        "decision": "allow",
        "reason": "r",
    },
    "approval_requested": {"tool_use_id": "t1", "question": "ok?"},
    "approval_granted": {"tool_use_id": "t1", "by": "me", "note": ""},
    "approval_denied": {"tool_use_id": "t1", "by": "me", "note": "no"},
    "sandbox_denied": {"action": "read", "target": "/etc", "reason": "outside"},
    "context_packed": {"selected": [], "dropped": [], "tokens": 1, "budget": 2},
    "context_compacted": {"strategy": "s", "tokens_before": 2, "tokens_after": 1},
    "verification_passed": {"verifier": "v", "message": "m"},
    "verification_failed": {"verifier": "v", "message": "m"},
    "escalation": {"rung": 1, "reason": "repeat"},
    "note_written": {"path": "notes.md"},
    "run_paused": {"reason": "awaiting_approval", "resume_from": 3},
    "run_resumed": {"reason": "approval", "resume_from": 3},
    "run_finished": {"reason": "finished", "result": "done"},
}


def make(kind: str, seq: int = 0, run: str = "r1", **payload: object) -> Event:
    line = {
        "schema": "loomcore.event.v1",
        "seq": seq,
        "timestamp": (START + timedelta(seconds=seq)).isoformat(
            timespec="milliseconds"
        ),
        "sessionId": SESSION,
        "runId": run,
        "kind": kind,
        **PAYLOADS[kind],
        **payload,
    }
    return parse_event(json.dumps(line))


def calling(*tool_use_ids: str, stop_reason: str = "tool_use") -> JsonObject:
    """A model_call `message` that asks for the given tools."""
    blocks: list[JsonValue] = [
        {"type": "tool_use", "id": i, "name": "n", "input": {}} for i in tool_use_ids
    ]
    return {"content": blocks, "stop_reason": stop_reason}
