"""Tokens to dollars (spec 3.8, "Tokens and cost").

Token counts are the source of truth; a price is always derived from them with the
vendored table, so a log can be priced again years later. Money is `Decimal` and is
never rounded here.
"""

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import cache
from importlib import resources

from pydantic import BaseModel, ConfigDict, Field

from loomcore.events import Event, JsonObject, ModelCall
from loomcore.ports import ModelRequest

PER_TOKENS = Decimal(1_000_000)
_DATED = re.compile(r"-\d{8}$")


class UnknownModelError(LookupError):
    """The table has no price for this model: add it to `pricing.json`, with a new
    `version`, before running against it."""


class Rates(BaseModel):
    """USD per million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: Decimal
    output: Decimal
    cache_read: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal


class LongContextRates(Rates):
    """What a request pays when its whole prompt is longer than `over` tokens."""

    over: int = Field(gt=0)


class ModelPrice(Rates):
    long_context: LongContextRates | None = None

    def for_prompt(self, prompt_tokens: int) -> Rates:
        """The rates one request pays, by the length of its prompt."""
        tier = self.long_context
        return tier if tier and prompt_tokens > tier.over else self


class PricingTable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    source: str
    models: dict[str, ModelPrice]

    def price_of(self, model: str) -> ModelPrice:
        """The model's prices. The table is keyed by undated ids; a dated snapshot
        of one (`claude-sonnet-5-5-20261101`) is priced as the model it snapshots."""
        for key in (model, _DATED.sub("", model)):
            if key in self.models:
                return self.models[key]
        raise UnknownModelError(
            f"no price for {model!r} in pricing table {self.version}"
        )


@cache
def vendored_table() -> PricingTable:
    """The table shipped with the library (`pricing.json`)."""
    text = (resources.files(__package__) / "pricing.json").read_text("utf-8")
    return PricingTable.model_validate(json.loads(text))


@dataclass(frozen=True)
class Tokens:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0

    @property
    def prompt(self) -> int:
        """Everything the request sent, cached or not: what a long-context tier
        is judged on."""
        return self.input + self.cache_read + self.cache_write_5m + self.cache_write_1h

    def __add__(self, other: Tokens) -> Tokens:
        return Tokens(
            self.input + other.input,
            self.output + other.output,
            self.cache_read + other.cache_read,
            self.cache_write_5m + other.cache_write_5m,
            self.cache_write_1h + other.cache_write_1h,
        )


@dataclass(frozen=True)
class Usage:
    """What one response cost in tokens, and the model that served it."""

    model: str
    tokens: Tokens


def _count(block: JsonObject, key: str, *, nullable: bool = False) -> int:
    """A token count, a `ValueError` when it is missing or is not a whole number (a
    malformed block must not price as free). The API documents the cache counts as
    `integer | null` and leaves them out when there is no caching, so those
    (`nullable`) read absent or null as 0; the input and output counts are required."""
    if key not in block:
        if nullable:
            return 0
        raise ValueError(f"usage.{key} is missing, so the call cannot be priced")
    value = block[key]
    if value is None and nullable:
        return 0
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError(f"usage.{key} is {value!r}, not a count of tokens")


def usage_of(message: JsonObject) -> Usage:
    """The `model` and `usage` of a Messages API response; a `ValueError` when it
    has none (the engine only sees real responses, so `TablePricing.cost` raises)."""
    found = find_usage(message)
    if found is None:
        raise ValueError("the response has no `model` and `usage` to price")
    return found


def find_usage(message: JsonObject) -> Usage | None:
    """The same, or None for a response with no `model` and `usage` (a scripted
    fake's, in a log). A response that has them but malformed still raises. A
    response with no cache breakdown counts its cache writes as five-minute ones."""
    model, usage = message.get("model"), message.get("usage")
    if not isinstance(model, str) or not isinstance(usage, dict):
        return None
    split = usage.get("cache_creation")
    if isinstance(split, dict):
        writes = (
            _count(split, "ephemeral_5m_input_tokens", nullable=True),
            _count(split, "ephemeral_1h_input_tokens", nullable=True),
        )
    else:
        writes = (_count(usage, "cache_creation_input_tokens", nullable=True), 0)
    return Usage(
        model,
        Tokens(
            _count(usage, "input_tokens"),
            _count(usage, "output_tokens"),
            _count(usage, "cache_read_input_tokens", nullable=True),
            *writes,
        ),
    )


def price(usage: Usage, table: PricingTable) -> Decimal:
    """The dollars one response cost: each kind of token at its own rate."""
    rates = table.price_of(usage.model).for_prompt(usage.tokens.prompt)
    t = usage.tokens
    total = (
        t.input * rates.input
        + t.output * rates.output
        + t.cache_read * rates.cache_read
        + t.cache_write_5m * rates.cache_write_5m
        + t.cache_write_1h * rates.cache_write_1h
    )
    return total / PER_TOKENS


@dataclass(frozen=True)
class RunCost:
    calls: int  # the calls priced
    tokens: Tokens
    usd: Decimal
    unpriced: int = 0  # calls whose response had no `usage` (a scripted fake's)


def tally(events: Iterable[Event], table: PricingTable) -> dict[str, RunCost]:
    """Each run's tokens and price, from its `model_call` events alone (the logged
    `cost_usd` is advisory and ignored). A fork's calls are its own."""
    runs: dict[str, RunCost] = {}
    for event in events:
        if not isinstance(event, ModelCall):
            continue
        before = runs.get(event.run_id, RunCost(0, Tokens(), Decimal(0)))
        usage = find_usage(event.message)
        if usage is None:
            runs[event.run_id] = replace(before, unpriced=before.unpriced + 1)
            continue
        runs[event.run_id] = RunCost(
            before.calls + 1,
            before.tokens + usage.tokens,
            before.usd + price(usage, table),
            before.unpriced,
        )
    return runs


@dataclass(frozen=True)
class TablePricing:
    """The driver's `Pricing` port, backed by the table. `count_input` is the
    edge's token count of a request (spec 3.5): this class does not estimate; the
    default ports' `count_input` does, and says so (spec 1, Real ports)."""

    table: PricingTable
    count_input: Callable[[ModelRequest], int]

    def worst_case(self, request: ModelRequest) -> Decimal:
        """The known input plus `max_tokens` of output, at the model's rates. A
        request that sets a cache prefix may write it to the cache, which costs more
        than plain input, so its input is priced at the dearest of the three."""
        known = self.count_input(request)
        rates = self.table.price_of(request.model).for_prompt(known)
        per_input = rates.input
        if request.cache_prefix is not None:
            per_input = max(rates.input, rates.cache_write_5m, rates.cache_write_1h)
        return (known * per_input + request.max_tokens * rates.output) / PER_TOKENS

    def cost(self, message: JsonObject) -> tuple[Decimal, str]:
        return price(usage_of(message), self.table), self.table.version
