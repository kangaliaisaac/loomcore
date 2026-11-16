"""The ports that need nothing from the caller: the real clocks, the jitter, the
sleep, a run id, the token counter and the price list."""

import json
import random
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TypedDict

from loomcore.context import Counter
from loomcore.events import JsonObject
from loomcore.observability.cost import TablePricing, vendored_table
from loomcore.ports import ModelRequest, Pricing
from loomcore.retry import Random

# Characters per token, rounded to err high: the context budget and the cost ceiling
# then err towards stopping early. What a run is billed is read from the response's
# `usage`, never from this count.
_CHARS_PER_TOKEN = 3


class SystemPorts(TypedDict):
    """The `Ports` fields `system_ports` fills; pass them with `**`."""

    pricing: Pricing
    count_tokens: Counter
    clock: Callable[[], datetime]
    monotonic: Callable[[], float]
    random: Random
    sleep: Callable[[float], None]
    new_run_id: Callable[[], str]


def estimate_tokens(messages: Sequence[JsonObject]) -> int:
    """An offline count of the tokens in a list of messages, from their serialised
    length. It is an estimate: the spec's "counted at the edge" is met by a caller
    that passes an exact counter in its place."""
    return len(json.dumps(messages, sort_keys=True)) // _CHARS_PER_TOKEN + 1


def estimate_request(request: ModelRequest) -> int:
    return estimate_tokens(
        [{"system": request.system}, *request.messages, {"tools": list(request.tools)}]
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def system_ports() -> SystemPorts:
    """The real clock, `time.monotonic`, `random.random`, `time.sleep`, a random run
    id, `estimate_tokens` and the vendored price table priced on `estimate_request`."""
    return SystemPorts(
        pricing=TablePricing(vendored_table(), estimate_request),
        count_tokens=estimate_tokens,
        clock=utc_now,
        monotonic=time.monotonic,
        random=random.random,
        sleep=time.sleep,
        new_run_id=new_run_id,
    )
