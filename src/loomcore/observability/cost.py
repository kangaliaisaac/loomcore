"""Tokens to dollars (spec 3.8, "Tokens and cost").

Token counts are the source of truth; a price is always derived from them with the
vendored table, so a log can be priced again years later. Money is `Decimal` and is
never rounded here.
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from importlib import resources

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from loomcore.events import Event, JsonObject, ModelCall
from loomcore.ports import ModelRequest

PER_TOKENS = Decimal(1_000_000)


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
        try:
            return self.models[model]
        except KeyError:
            raise UnknownModelError(
                f"no price for {model!r} in pricing table {self.version}"
            ) from None


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


def _count(block: JsonObject, key: str) -> int:
    value = block.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def usage_of(message: JsonObject) -> Usage:
    """The `model` and `usage` of a Messages API response. A response with no
    cache breakdown counts its cache writes as the default five-minute ones."""
    model, usage = message.get("model"), message.get("usage")
    if not isinstance(model, str) or not isinstance(usage, dict):
        raise ValueError("the response has no `model` and `usage` to price")
    split = usage.get("cache_creation")
    if isinstance(split, dict):
        writes = (
            _count(split, "ephemeral_5m_input_tokens"),
            _count(split, "ephemeral_1h_input_tokens"),
        )
    else:
        writes = (_count(usage, "cache_creation_input_tokens"), 0)
    return Usage(
        model,
        Tokens(
            _count(usage, "input_tokens"),
            _count(usage, "output_tokens"),
            _count(usage, "cache_read_input_tokens"),
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
    calls: int
    tokens: Tokens
    usd: Decimal


def tally(events: Iterable[Event], table: PricingTable) -> dict[str, RunCost]:
    """Each run's tokens and price, from its `model_call` events alone (the logged
    `cost_usd` is advisory and ignored). A fork's calls are its own."""
    runs: dict[str, RunCost] = {}
    for event in events:
        if isinstance(event, ModelCall):
            usage = usage_of(event.message)
            before = runs.get(event.run_id, RunCost(0, Tokens(), Decimal(0)))
            runs[event.run_id] = RunCost(
                before.calls + 1,
                before.tokens + usage.tokens,
                before.usd + price(usage, table),
            )
    return runs


@dataclass(frozen=True)
class TablePricing:
    """The driver's `Pricing` port, backed by the table. `count_input` is the
    edge's token count of a request (spec 3.5): it is never estimated here."""

    table: PricingTable
    count_input: Callable[[ModelRequest], int]

    def worst_case(self, request: ModelRequest) -> Decimal:
        """The known input plus `max_tokens` of output, at the model's rates."""
        known = self.count_input(request)
        rates = self.table.price_of(request.model).for_prompt(known)
        return (known * rates.input + request.max_tokens * rates.output) / PER_TOKENS

    def cost(self, message: JsonObject) -> tuple[Decimal, str]:
        return price(usage_of(message), self.table), self.table.version


def tokens_of(message: JsonValue) -> Tokens | None:
    """The tokens of a response if it has a usage block, else None (a log written
    by a fake model has none)."""
    if not isinstance(message, dict):
        return None
    try:
        return usage_of(message).tokens
    except ValueError:
        return None
