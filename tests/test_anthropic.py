"""The Anthropic model port, against a real SDK client on a mock transport."""

import json
import subprocess
import sys
from collections.abc import Callable
from decimal import Decimal

import anthropic
import httpx2
import pytest

import loomcore.anthropic as port
from loomcore.events import JsonObject
from loomcore.ports import ModelCallError, ModelRequest
from loomcore.retry import GiveUp, decide
from loomcore.runtime import system_ports

MESSAGE: JsonObject = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3},
}
USER: JsonObject = {"role": "user", "content": "hello"}
TOOL: JsonObject = {"name": "t", "description": "d", "input_schema": {"type": "object"}}


def request(
    cache_prefix: int | None = None, tools: tuple[JsonObject, ...] = ()
) -> ModelRequest:
    return ModelRequest(
        model="claude-haiku-5-5",
        system="stable rest",
        messages=[USER],
        tools=tools,
        max_tokens=64,
        cache_prefix=cache_prefix,
    )


def client_on(
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def replying(
    status: int = 200, headers: dict[str, str] | None = None
) -> tuple[anthropic.Anthropic, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def handler(sent: httpx2.Request) -> httpx2.Response:
        seen.append(sent)
        body = MESSAGE if status == 200 else {"type": "error", "error": {}}
        return httpx2.Response(status, json=body, headers=headers)

    return client_on(handler), seen


def test_a_call_sends_the_request_and_returns_the_response_as_json() -> None:
    client, seen = replying()
    response = port.model(client)(request())
    sent = json.loads(seen[0].content)
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["max_tokens"] == 64
    assert sent["system"] == "stable rest"
    assert sent["messages"] == [USER]
    assert "tools" not in sent
    assert response["content"] == [{"type": "text", "text": "hi"}]
    assert response["usage"] == MESSAGE["usage"]
    assert "stop_sequence" not in response  # None fields are dropped


def test_every_call_carries_a_timeout() -> None:
    client, seen = replying()
    port.model(client)(request())
    port.model(client, timeout=12.5)(request())
    assert seen[0].extensions["timeout"]["read"] == 300.0
    assert seen[1].extensions["timeout"]["read"] == 12.5


def test_tools_are_sent_when_there_are_some() -> None:
    client, seen = replying()
    port.model(client)(request(tools=(TOOL,)))
    assert json.loads(seen[0].content)["tools"] == [TOOL]


def test_the_cache_marker_goes_after_the_stable_prefix() -> None:
    client, seen = replying()
    port.model(client)(request(cache_prefix=6))
    assert json.loads(seen[0].content)["system"] == [
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": " rest"},
    ]


def test_a_prefix_that_is_the_whole_prompt_has_no_empty_block() -> None:
    client, seen = replying()
    port.model(client)(request(cache_prefix=len("stable rest")))
    assert len(json.loads(seen[0].content)["system"]) == 1


@pytest.mark.parametrize(
    ("headers", "retry_after"),
    [
        ({"retry-after": "3"}, 3.0),
        ({}, None),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({"retry-after": "inf"}, None),
        ({"retry-after": "nan"}, None),
        ({"retry-after": "-1"}, None),
    ],
)
def test_an_http_error_carries_its_status_and_retry_after(
    headers: dict[str, str], retry_after: float | None
) -> None:
    client, seen = replying(429, headers)
    with pytest.raises(ModelCallError) as caught:
        port.model(client)(request())
    assert (caught.value.error.kind, caught.value.error.status) == ("http", 429)
    assert caught.value.error.retry_after == retry_after
    assert len(seen) == 1  # the client does not retry: the driver's policy does


def test_an_http_error_keeps_the_providers_message_and_its_cause() -> None:
    def refuse(sent: httpx2.Request) -> httpx2.Response:
        error = {"type": "invalid_request_error", "message": "prompt is too long"}
        return httpx2.Response(400, json={"type": "error", "error": error})

    with pytest.raises(ModelCallError) as caught:
        port.model(client_on(refuse))(request())
    assert caught.value.error.detail == "prompt is too long"
    assert isinstance(caught.value.__cause__, anthropic.APIStatusError)
    verdict = decide(0, caught.value.error, lambda: 0.5)
    assert verdict == GiveUp("not retryable: HTTP 400: prompt is too long")


def test_a_reply_that_is_not_the_apis_error_json_falls_back_to_the_sdks_text() -> None:
    def refuse(sent: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(502, text="bad gateway")

    with pytest.raises(ModelCallError) as caught:
        port.model(client_on(refuse))(request())
    assert "bad gateway" in caught.value.error.detail


def test_a_long_message_is_cut() -> None:
    def refuse(sent: httpx2.Request) -> httpx2.Response:
        error = {"type": "invalid_request_error", "message": "x" * 1000}
        return httpx2.Response(400, json={"type": "error", "error": error})

    with pytest.raises(ModelCallError) as caught:
        port.model(client_on(refuse))(request())
    assert len(caught.value.error.detail) == 300


@pytest.mark.parametrize(
    ("raised", "kind"),
    [(httpx2.ConnectError, "connection"), (httpx2.ReadTimeout, "timeout")],
)
def test_a_transport_failure_is_a_connection_or_timeout_error(
    raised: type[httpx2.TransportError], kind: str
) -> None:
    def fail(sent: httpx2.Request) -> httpx2.Response:
        raise raised("down", request=sent)

    with pytest.raises(ModelCallError) as caught:
        port.model(client_on(fail))(request())
    assert caught.value.error.kind == kind


def test_a_client_that_retries_is_made_not_to() -> None:
    seen: list[httpx2.Request] = []

    def overloaded(sent: httpx2.Request) -> httpx2.Response:
        seen.append(sent)
        return httpx2.Response(429, json={"type": "error", "error": {}})

    retrying = anthropic.Anthropic(
        api_key="test",
        max_retries=2,
        http_client=httpx2.Client(transport=httpx2.MockTransport(overloaded)),
    )
    with pytest.raises(ModelCallError):
        port.model(retrying)(request())
    assert len(seen) == 1
    assert retrying.max_retries == 2  # the caller's own client is left as it was


def test_the_default_client_is_built_without_arguments_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    seen: list[httpx2.Request] = []

    def overloaded(sent: httpx2.Request) -> httpx2.Response:
        seen.append(sent)
        return httpx2.Response(429, json={"type": "error", "error": {}})

    real = anthropic.Anthropic

    def fake() -> anthropic.Anthropic:
        return real(
            http_client=httpx2.Client(transport=httpx2.MockTransport(overloaded))
        )

    monkeypatch.setattr(anthropic, "Anthropic", fake)
    with pytest.raises(ModelCallError):
        port.model()(request())
    assert len(seen) == 1


def test_a_null_in_a_tool_input_is_kept() -> None:
    reply: JsonObject = {
        **MESSAGE,
        "content": [
            {"type": "tool_use", "id": "t1", "name": "t", "input": {"a": None, "b": 1}}
        ],
        "stop_reason": "tool_use",
    }
    client = client_on(lambda sent: httpx2.Response(200, json=reply))
    content = port.model(client)(request())["content"]
    assert content == [
        {"type": "tool_use", "id": "t1", "name": "t", "input": {"a": None, "b": 1}}
    ]


def test_the_core_does_not_import_the_sdk() -> None:
    code = (
        "import importlib, pkgutil, sys, loomcore\n"
        "for m in pkgutil.walk_packages(loomcore.__path__, 'loomcore.'):\n"
        "    if m.name != 'loomcore.anthropic':\n"
        "        importlib.import_module(m.name)\n"
        "sys.exit('anthropic' in sys.modules)"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], check=False)
    assert done.returncode == 0


def test_the_response_prices_from_its_usage() -> None:
    client, _ = replying()
    cost, _version = system_ports()["pricing"].cost(port.model(client)(request()))
    assert cost == Decimal("0.00000353")  # Haiku: 10 in, 5 out, 3 cached
