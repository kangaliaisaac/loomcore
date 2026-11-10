"""The messages a call sends are a pure function of the log."""

from factories import calling, make

from loomcore.conversation import CONTINUE, build_messages, call_hash, final_text
from loomcore.events import JsonObject


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
