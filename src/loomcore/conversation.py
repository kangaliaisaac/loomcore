"""The messages a model call sends, rebuilt from the log (spec section 1)."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace

from pydantic import JsonValue

from loomcore.context import (
    PLAN_TOOL,
    Cleared,
    Counter,
    Item,
    ItemKind,
    clear_results,
)
from loomcore.events import Event, JsonObject, ModelCall, RunForked, ToolCompleted
from loomcore.replay import content_blocks, tool_use_ids

FORK_CONTINUE = "Carry on with the task."
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
    return render(exchanges(task, events, lambda _: 0))


def render(items: Sequence[Item]) -> list[JsonObject]:
    """The messages of the items, in order."""
    return [message for item in items for message in item.messages]


def exchanges(task: JsonValue, events: Sequence[Event], count: Counter) -> list[Item]:
    """The history as items: the task, each response with its tool results as one
    exchange, and a fork's hint. `count` measures an item's messages at the edge, so
    the edge can count the shape the provider receives. The task, any hint, the latest
    exchange and the latest exchange with a successful `update_plan` call (the plan the
    model keeps) are pinned."""
    done = {e.tool_use_id: e for e in events if isinstance(e, ToolCompleted)}
    items = [_item("task", "task", 0, [{"role": "user", "content": task}], count)]
    last = "user"
    plan_at: int | None = None
    for event in events:
        messages: list[JsonObject] = []
        kind: ItemKind
        if isinstance(event, ModelCall):
            if sent := _sent(event.message):  # an empty turn would be rejected
                messages.append({"role": "assistant", "content": sent})
            if reply := _reply(event.message, done):
                messages.append(reply)
            kind, key = "exchange", f"seq:{event.run_id}:{event.seq}"
            if _updates_plan(event.message, done):
                plan_at = len(items)  # the index this exchange is appended at
        elif isinstance(event, RunForked):
            messages = _after_fork(last, event)
            kind, key = "hint", f"hint:{event.run_id}"
        if messages:
            last = str(messages[-1]["role"])
            items.append(_item(key, kind, len(items), messages, count))
    return _pin_latest(items, plan_at)


def _item(
    key: str,
    kind: ItemKind,
    order: int,
    messages: list[JsonObject],
    count: Counter,
) -> Item:
    """The task and a hint are pinned; the latest exchange is pinned afterwards."""
    tokens = count(messages)
    return Item(
        key,
        kind,
        order,
        tokens,
        tuple(messages),
        pinned=kind != "exchange",
        cleared=_cleared(messages, tokens, count),
    )


def recount(items: Sequence[Item], count: Counter) -> list[Item]:
    """The items with their tokens, and the cleared form's, counted. `exchanges` can
    build them without counting; the caller counts only the ones it keeps."""
    return [
        replace(
            item,
            tokens=(tokens := count(item.messages)),
            cleared=_cleared(list(item.messages), tokens, count),
        )
        for item in items
    ]


def _cleared(messages: list[JsonObject], tokens: int, count: Counter) -> Cleared | None:
    """The exchange with its results cleared, or None when that would not make it
    smaller (no result, or one shorter than the placeholder)."""
    blank = clear_results(messages)
    if blank == tuple(messages):
        return None  # no tool result to clear, so nothing to count
    cleared = count(blank)
    return None if cleared >= tokens else Cleared(blank, cleared)


def _pin_latest(items: list[Item], plan_at: int | None) -> list[Item]:
    latest = max(
        (k for k, item in enumerate(items) if item.kind == "exchange"), default=None
    )
    if plan_at is not None:  # kept for good, but its results may still be cleared
        items[plan_at] = replace(items[plan_at], pinned=True, may_clear=True)
    if latest is not None:  # the latest exchange is shown in full
        items[latest] = replace(items[latest], pinned=True, may_clear=False)
    return items


def _updates_plan(message: JsonObject, done: dict[str, ToolCompleted]) -> bool:
    """Whether the response made an `update_plan` call that succeeded (a failed one
    left the plan as it was)."""
    ids = set(tool_use_ids(message))
    return any(
        block.get("type") == "tool_use"
        and block.get("name") == PLAN_TOOL
        and block.get("id") in ids
        and (result := done.get(str(block.get("id")))) is not None
        and result.ok
        for block in content_blocks(message)
    )


def _after_fork(last: str, event: RunForked) -> list[JsonObject]:
    """A hint is the next user message; with none, a fork from a final answer would
    end the request on the assistant's turn, so ask it to carry on."""
    if event.hint:
        return [{"role": "user", "content": event.hint}]
    if last == "assistant":
        return [{"role": "user", "content": FORK_CONTINUE}]
    return []


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
