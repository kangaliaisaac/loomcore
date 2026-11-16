"""The model port on the Anthropic SDK (`pip install loomcore[anthropic]`).

The core never imports this module or the SDK: it sees only the `ModelRequest ->
response` function `model` returns.
"""

import math
from collections.abc import Callable, Mapping
from typing import Any

import anthropic

from loomcore.events import JsonObject
from loomcore.ports import ModelCallError, ModelRequest
from loomcore.retry import CallError

_DETAIL_CHARS = 300  # the provider's message is kept to this much in a stop reason


def model(
    client: anthropic.Anthropic | None = None, *, timeout: float | None = None
) -> Callable[[ModelRequest], JsonObject]:
    """A model port that calls the Messages API. The default client reads
    `ANTHROPIC_API_KEY`. The client's own retries are switched off, whichever client
    is passed: the driver's retry policy is the only layer, and a second one would
    multiply it and sleep past the run's timeout check. A `timeout` in seconds is
    sent only when given: the SDK's own, worked out from `max_tokens`, otherwise
    applies, and so does its refusal of a request too long to send without
    streaming."""
    api = (client or anthropic.Anthropic()).with_options(max_retries=0)

    def call(request: ModelRequest) -> JsonObject:
        try:
            message = api.messages.create(**_params(request, timeout))
        except anthropic.APITimeoutError as error:
            raise ModelCallError(CallError("timeout")) from error
        except anthropic.APIConnectionError as error:
            raise ModelCallError(CallError("connection")) from error
        except anthropic.APIStatusError as error:
            retry_after = _retry_after(error.response.headers)
            failed = CallError("http", error.status_code, retry_after, _detail(error))
            raise ModelCallError(failed) from error
        response: JsonObject = message.model_dump(mode="json", exclude_none=True)
        return response

    return call


def _params(request: ModelRequest, timeout: float | None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "system": _system(request),
        "messages": request.messages,
    }
    if request.tools:
        params["tools"] = list(request.tools)
    if timeout is not None:
        params["timeout"] = timeout
    return params


def _system(request: ModelRequest) -> str | list[Mapping[str, Any]]:
    """The system prompt, with the cache marker after its stable prefix if set."""
    if request.cache_prefix is None:
        return request.system
    stable = request.system[: request.cache_prefix]
    rest = request.system[request.cache_prefix :]
    blocks: list[Mapping[str, Any]] = [
        {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}}
    ]
    if rest:
        blocks.append({"type": "text", "text": rest})
    return blocks


def _detail(error: anthropic.APIStatusError) -> str:
    """The provider's own message, else the SDK's, cut to a stop reason's length."""
    body = error.body
    inner = body.get("error") if isinstance(body, dict) else None
    text = inner.get("message") if isinstance(inner, dict) else None
    return (text if isinstance(text, str) else error.message)[:_DETAIL_CHARS]


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds to wait: `retry-after-ms` if the provider sent it, else `retry-after`
    (its HTTP-date form is not read). `x-should-retry` is not read."""
    in_ms = _number(headers.get("retry-after-ms"))
    return in_ms / 1000 if in_ms is not None else _number(headers.get("retry-after"))


def _number(header: str | None) -> float | None:
    """A finite, non-negative number, else None."""
    try:
        number = None if header is None else float(header)
    except ValueError:
        return None
    return (
        number if number is not None and math.isfinite(number) and number >= 0 else None
    )
