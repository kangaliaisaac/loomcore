"""The messages a model call sends, rebuilt from the log (spec section 1)."""

import hashlib
import json
from collections.abc import Sequence

from pydantic import JsonValue

from loomcore.events import Event, JsonObject, ModelCall, RunForked, ToolCompleted
from loomcore.replay import content_blocks, tool_use_ids

CONTINUE = "Your last reply was cut off. Carry on from where it stopped."


def call_hash(name: str, args: JsonObject) -> str:
    """Identifies a repeated call (spec section 1, two identifiers)."""
    canonical = json.dumps(
        {"tool": name, "input": args}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def tool_call(events: Sequence[Event], tool_use_id: str) -> tuple[str, JsonObject]:
    """The name and input the model gave a `tool_use` block."""
    for event in reversed(events):
        if isinstance(event, ModelCall):
            for block in content_blocks(event.message):
                if block.get("id") == tool_use_id:
                    args = block.get("input")
                    return str(block["name"]), args if isinstance(args, dict) else {}
    raise KeyError(f"no tool_use block {tool_use_id} in the log")


def final_text(message: JsonObject) -> str:
    """The text blocks of a response, joined."""
    texts = (
        b.get("text", "") for b in content_blocks(message) if b.get("type") == "text"
    )
    return "".join(str(text) for text in texts)


def build_messages(task: JsonValue, events: Sequence[Event]) -> list[JsonObject]:
    """The conversation so far: the task, then each response and its tool results."""
    done = {e.tool_use_id: e for e in events if isinstance(e, ToolCompleted)}
    messages: list[JsonObject] = [{"role": "user", "content": task}]
    for event in events:
        if isinstance(event, ModelCall):
            if sent := _sent(event.message):  # an empty turn would be rejected
                messages.append({"role": "assistant", "content": sent})
            if reply := _reply(event.message, done):
                messages.append(reply)
        elif isinstance(event, RunForked) and event.hint:
            messages.append({"role": "user", "content": event.hint})
    return messages


def _sent(message: JsonObject) -> list[JsonValue]:
    """The response's blocks, without a tool_use that max_tokens cut off mid-input."""
    kept = set(tool_use_ids(message))
    return [
        b
        for b in content_blocks(message)
        if b.get("type") != "tool_use" or b.get("id") in kept
    ]


def _reply(message: JsonObject, done: dict[str, ToolCompleted]) -> JsonObject | None:
    results: list[JsonValue] = [
        _result(done[i]) for i in tool_use_ids(message) if i in done
    ]
    if message.get("stop_reason") == "max_tokens":
        results.append({"type": "text", "text": CONTINUE})
    return {"role": "user", "content": results} if results else None


def _result(completed: ToolCompleted) -> JsonObject:
    content = completed.content
    return {
        "type": "tool_result",
        "tool_use_id": completed.tool_use_id,
        "content": content if isinstance(content, str) else json.dumps(content),
        "is_error": not completed.ok,
    }
