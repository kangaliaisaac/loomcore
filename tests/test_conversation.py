"""The messages a call sends are a pure function of the log."""

import json
from collections.abc import Sequence

from factories import calling, make

from loomcore.context import CLEARED, pack
from loomcore.conversation import (
    CONTINUE,
    FORK_CONTINUE,
    build_messages,
    call_hash,
    exchanges,
    final_text,
    render,
)
from loomcore.events import JsonObject
from loomcore.replay import run_events


def size(messages: Sequence[JsonObject]) -> int:
    """A stand-in counter for the edge's: the serialised length."""
    return len(json.dumps(messages))


def blocks(message: JsonObject) -> list[JsonObject]:
    content = message["content"]
    assert isinstance(content, list)
    return [block for block in content if isinstance(block, dict)]


def test_the_task_then_each_response_and_its_results() -> None:
    log = [
        make("model_call", 1, message=calling("a")),
        make("tool_completed", 2, tool_use_id="a", content="found", ok=True),
    ]
    messages = build_messages("go", log)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[-1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "a",
            "content": "found",
            "is_error": False,
        }
    ]


def test_a_failed_call_is_marked_and_non_text_content_is_json() -> None:
    log = [
        make("model_call", 1, message=calling("a")),
        make("tool_completed", 2, tool_use_id="a", content={"n": 1}, ok=False),
    ]
    [block] = blocks(build_messages("go", log)[-1])
    assert block["is_error"] is True
    assert block["content"] == '{"n": 1}'


def test_a_tool_use_cut_off_by_max_tokens_is_not_sent_back() -> None:
    message = calling("done", "cut", stop_reason="max_tokens")
    log = [
        make("model_call", 1, message=message),
        make("tool_completed", 2, tool_use_id="done"),
    ]
    assistant, reply = build_messages("go", log)[1:]
    assert [b["id"] for b in blocks(assistant)] == ["done"]
    assert blocks(reply)[-1] == {"type": "text", "text": CONTINUE}


def test_a_truncated_reply_with_no_tools_is_followed_by_the_instruction() -> None:
    cut = {"content": [{"type": "text", "text": "par"}], "stop_reason": "max_tokens"}
    messages = build_messages("go", [make("model_call", 1, message=cut)])
    assert messages[-1] == {
        "role": "user",
        "content": [{"type": "text", "text": CONTINUE}],
    }


def test_a_forks_hint_is_the_next_user_message() -> None:
    forked = make("run_forked", 0, "r2", hint="try the other index")
    assert build_messages("go", [forked])[-1]["content"] == "try the other index"


def test_the_call_hash_ignores_key_order() -> None:
    assert call_hash("n", {"a": 1, "b": 2}) == call_hash("n", {"b": 2, "a": 1})
    assert call_hash("n", {"a": 1}) != call_hash("m", {"a": 1})


def test_final_text_joins_the_text_blocks() -> None:
    message: JsonObject = {
        "content": [
            {"type": "text", "text": "a"},
            {"type": "tool_use", "id": "x", "name": "n", "input": {}},
            {"type": "text", "text": "b"},
        ]
    }
    assert final_text(message) == "ab"


def test_a_response_cut_off_in_its_only_tool_use_is_not_sent_empty() -> None:
    message = calling("cut", stop_reason="max_tokens")
    messages = build_messages("go", [make("model_call", 1, message=message)])
    assert [m["role"] for m in messages] == ["user", "user"]
    assert blocks(messages[-1]) == [{"type": "text", "text": CONTINUE}]


def test_the_history_is_cut_into_exchanges_with_the_latest_pinned() -> None:
    log = [
        make("model_call", 1, message=calling("a")),
        make("tool_completed", 2, tool_use_id="a", content="found", ok=True),
        make("model_call", 3, message=calling("b")),
        make("tool_completed", 4, tool_use_id="b", content="more", ok=True),
    ]
    items = exchanges("go", log, size)
    assert [(i.id, i.kind, i.pinned) for i in items] == [
        ("task", "task", True),
        ("seq:r1:1", "exchange", False),
        ("seq:r1:3", "exchange", True),
    ]
    assert [i.order for i in items] == [0, 1, 2]
    assert render(items) == build_messages("go", log)
    assert all(i.tokens > 0 for i in items)


def test_an_exchange_can_be_shown_with_its_results_cleared() -> None:
    log = [
        make("model_call", 1, message=calling("a")),
        make("tool_completed", 2, tool_use_id="a", content="found " * 40, ok=False),
    ]
    _, exchange = exchanges("go", log, size)
    assert exchange.cleared is not None
    [block] = blocks(exchange.cleared.messages[1])
    assert block["content"] == CLEARED
    assert block["is_error"] is True  # what happened stays, what was returned goes
    assert block["tool_use_id"] == "a"
    assert exchange.cleared.tokens < exchange.tokens
    assert (
        exchanges("go", [make("model_call", 1, message=cut_text())], size)[1].cleared
        is None
    )


def cut_text() -> JsonObject:
    return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}


def test_dropping_a_whole_exchange_keeps_the_roles_alternating() -> None:
    log = [
        make("model_call", 1, message=calling("a")),
        make("tool_completed", 2, tool_use_id="a"),
        make("model_call", 3, message=calling("b")),
        make("tool_completed", 4, tool_use_id="b"),
        make("model_call", 5, message=calling("c")),
        make("tool_completed", 6, tool_use_id="c"),
    ]
    task, first, second, third = exchanges("go", log, size)
    roles = [m["role"] for m in render([task, second, third])]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert first.id == "seq:r1:1"


def test_a_fork_hint_is_pinned_and_a_fork_from_an_answer_asks_to_carry_on() -> None:
    answer = {"content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn"}
    origin: dict[str, object] = {"from": {"runId": "r0", "seq": 1}}
    log = [
        make("model_call", 1, message=answer),
        make("run_forked", 2, "r1", **origin),
        make("run_forked", 3, "r2", hint="try b", **origin),
    ]
    items = exchanges("go", log, size)
    hints = [i for i in items if i.kind == "hint"]
    assert [i.pinned for i in hints] == [True, True]
    assert [i.messages[0]["content"] for i in hints] == [FORK_CONTINUE, "try b"]


def test_item_ids_stay_unique_in_a_fork_of_a_fork_because_seq_restarts_per_run() -> (
    None
):
    answer = {"content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn"}
    log = [
        make("run_started", 0, "r1"),
        make("model_call", 1, "r1", message=answer),
        make("run_forked", 0, "r2", hint="b", **{"from": {"runId": "r1", "seq": 1}}),
        make("model_call", 1, "r2", message=answer),
        make("run_forked", 0, "r3", hint="c", **{"from": {"runId": "r2", "seq": 1}}),
        make("model_call", 1, "r3", message=answer),
    ]
    items = exchanges("go", run_events(log, "r3"), size)
    names = [i.id for i in items]
    assert len(set(names)) == len(names)
    assert names == [
        "task",
        "seq:r1:1",
        "hint:r2",
        "seq:r2:1",
        "hint:r3",
        "seq:r3:1",
    ]
    pinned = [i for i in items if i.pinned]
    pack([i for i in items if not i.pinned], 10_000, pinned)  # the first window works
