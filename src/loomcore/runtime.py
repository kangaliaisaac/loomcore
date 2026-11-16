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

# Characters per token. Three is high for prose and low for dense text such as hashes
# and ids, so the context budget's count is a heuristic. Two is what a hash costs, so
# the pre-call cost ceiling, which must not undercount, uses it. What a run is billed
# is read from the response's `usage`, never from either count.
_CHARS_PER_TOKEN = 3
_PRICING_CHARS_PER_TOKEN = 2


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
    length. It is a heuristic (see `_CHARS_PER_TOKEN`); a caller that relies on the
    context budget passes an exact counter in its place."""
    return _estimate(messages, _CHARS_PER_TOKEN)


def estimate_request(request: ModelRequest) -> int:
    """The count the pre-call cost ceiling uses: two characters a token, so it does
    not undercount a hash or an id. A caller who needs a tight ceiling passes an exact
    counter to `TablePricing`."""
    messages: list[JsonObject] = [
        {"system": request.system},
        *request.messages,
        {"tools": list(request.tools)},
    ]
    return _estimate(messages, _PRICING_CHARS_PER_TOKEN)


def _estimate(messages: Sequence[JsonObject], chars_per_token: int) -> int:
    return len(json.dumps(messages, sort_keys=True)) // chars_per_token + 1


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def system_ports(
    *, count_tokens: Counter, count_request: Callable[[ModelRequest], int]
) -> SystemPorts:
    """The real clock, `time.monotonic`, `random.random`, `time.sleep`, a random run
    id, `count_tokens` as given, and the vendored price table priced on
    `count_request`. The two counters have no default, so a spend cap is never made
    soft by leaving something out: pass `estimate_tokens` and `estimate_request` for
    an offline heuristic (the cost ceiling is then not a guaranteed bound), or exact
    counters where a cap must hold (spec section 1)."""
    return SystemPorts(
        pricing=TablePricing(vendored_table(), count_request),
        count_tokens=count_tokens,
        clock=utc_now,
        monotonic=time.monotonic,
        random=random.random,
        sleep=time.sleep,
        new_run_id=new_run_id,
    )
