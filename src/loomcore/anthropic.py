"""The model port on the Anthropic SDK (`pip install loomcore[anthropic]`).

The core never imports this module or the SDK: it sees only the `ModelRequest ->
response` function `model` returns.
"""

from collections.abc import Callable, Mapping
from typing import Any

import anthropic

from loomcore.events import JsonObject
from loomcore.ports import ModelCallError, ModelRequest
from loomcore.retry import CallError


def model(
    client: anthropic.Anthropic | None = None,
) -> Callable[[ModelRequest], JsonObject]:
    """A model port that calls the Messages API. The default client reads
    `ANTHROPIC_API_KEY` and does not retry: the driver's retry policy does, so a
    client's own retries would multiply it. A client passed in should do the same
    (`max_retries=0`)."""
    api = client or anthropic.Anthropic(max_retries=0)

    def call(request: ModelRequest) -> JsonObject:
        try:
            message = api.messages.create(**_params(request))
        except anthropic.APITimeoutError:
            raise ModelCallError(CallError("timeout")) from None
        except anthropic.APIConnectionError:
            raise ModelCallError(CallError("connection")) from None
        except anthropic.APIStatusError as error:
            status = error.status_code
            retry_after = _seconds(error.response.headers.get("retry-after"))
            raise ModelCallError(CallError("http", status, retry_after)) from None
        response: JsonObject = message.model_dump(mode="json", exclude_none=True)
        return response

    return call


def _params(request: ModelRequest) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "system": _system(request),
        "messages": request.messages,
    }
    if request.tools:
        params["tools"] = list(request.tools)
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


def _seconds(header: str | None) -> float | None:
    """`retry-after` as seconds; the HTTP-date form is not read."""
    try:
        return None if header is None else float(header)
    except ValueError:
        return None
