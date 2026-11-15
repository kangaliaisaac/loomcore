"""f21: spans from a recorded log are stable; pricing from tokens matches
hand-computed values; cache read and write tokens are priced separately (spec 3.8)."""

from decimal import Decimal
from pathlib import Path

import pytest
from factories import make
from pydantic import JsonValue

from loomcore.eventlog import open_writer, read_events
from loomcore.events import Event, JsonObject
from loomcore.observability.cost import (
    PricingTable,
    TablePricing,
    Tokens,
    UnknownModelError,
    Usage,
    price,
    tally,
    usage_of,
    vendored_table,
)
from loomcore.observability.trace import spans
from loomcore.ports import ModelRequest

TABLE = vendored_table()
D = Decimal


def usage(model: str = "claude-sonnet-5-5", /, **tokens: int) -> Usage:
    return Usage(model, Tokens(**tokens))


def response(model: str = "claude-sonnet-5-5", /, **fields: JsonValue) -> JsonObject:
    used: dict[str, JsonValue] = {"input_tokens": 1840, "output_tokens": 312, **fields}
    return {
        "id": "msg_1",
        "model": model,
        "stop_reason": "end_turn",
        "content": [],
        "usage": used,
    }


# --- the vendored table -------------------------------------------------------


def test_the_vendored_table_is_the_published_one() -> None:
    sonnet = TABLE.price_of("claude-sonnet-5-5")
    assert (sonnet.input, sonnet.output) == (D(2), D(10))
    assert (sonnet.cache_read, sonnet.cache_write_5m, sonnet.cache_write_1h) == (
        D("0.10"),
        D("2.50"),
        D(4),
    )
    opus = TABLE.price_of("claude-opus-5-5")
    assert (opus.input, opus.output, opus.cache_read) == (D(4), D(20), D("0.20"))
    fable = TABLE.price_of("claude-fable-5-2")
    assert (fable.input, fable.output, fable.cache_write_1h) == (D(8), D(40), D(16))
    assert TABLE.version == "2026-11-15"


def test_a_model_the_table_does_not_list_is_an_error_that_says_what_to_do() -> None:
    with pytest.raises(UnknownModelError, match="pricing table"):
        price(usage("claude-unknown-1", input=1), TABLE)


# --- pricing from tokens ------------------------------------------------------


def test_a_call_is_priced_from_its_tokens_by_hand() -> None:
    # 1840 in at $2, 312 out at $10, 9200 cache reads at $0.10 per million tokens:
    # 3680 + 3120 + 920 = 7720 millionths of a dollar.
    got = price(usage(input=1840, output=312, cache_read=9200), TABLE)
    assert got == D("0.00772")
    opus = usage("claude-opus-5-5", input=1840, output=312, cache_read=9200)
    assert price(opus, TABLE) == D("0.01544")  # 7360 + 6240 + 1840


@pytest.mark.parametrize(
    ("field", "dollars"),
    [
        ("input", "2"),
        ("output", "10"),
        ("cache_read", "0.10"),
        ("cache_write_5m", "2.50"),
        ("cache_write_1h", "4"),
    ],
)
def test_each_kind_of_token_has_its_own_price(field: str, dollars: str) -> None:
    one_million = usage(**{field: 1_000_000})
    assert price(one_million, TABLE) == D(dollars)


def test_cache_reads_and_writes_are_not_priced_as_input() -> None:
    plain = price(usage(input=10_000), TABLE)
    cached = price(usage(cache_read=10_000), TABLE)
    written = price(usage(cache_write_5m=10_000), TABLE)
    longer = price(usage(cache_write_1h=10_000), TABLE)
    assert cached < plain < written < longer
    assert (plain, cached, written, longer) == (
        D("0.02"),
        D("0.001"),
        D("0.025"),
        D("0.04"),
    )


def test_haiku_pays_the_long_context_rates_over_100k_prompt_tokens() -> None:
    haiku = "claude-haiku-5-5"
    assert price(usage(haiku, input=100_000), TABLE) == D("0.01")
    # One token over: the whole request moves up to the long-context rates.
    assert price(usage(haiku, input=100_001), TABLE) == D("0.0500005")
    # Cached tokens count towards the prompt that decides the tier.
    over = usage(haiku, input=1, cache_read=100_000)
    assert price(over, TABLE) == D("0.5") / 1_000_000 + D("0.05") * D("0.1")
    # A model with no long-context tier never moves up.
    big = price(usage(input=500_000), TABLE)
    assert big == D(1)


# --- reading usage ------------------------------------------------------------


def test_usage_reads_the_cache_breakdown() -> None:
    message = response(
        cache_read_input_tokens=9200,
        cache_creation_input_tokens=300,
        cache_creation={
            "ephemeral_5m_input_tokens": 100,
            "ephemeral_1h_input_tokens": 200,
        },
    )
    assert usage_of(message) == usage(
        input=1840, output=312, cache_read=9200, cache_write_5m=100, cache_write_1h=200
    )


def test_cache_writes_with_no_breakdown_are_five_minute_writes() -> None:
    message = response(cache_creation_input_tokens=300)
    assert usage_of(message).tokens.cache_write_5m == 300


def test_a_response_with_no_usage_cannot_be_priced() -> None:
    with pytest.raises(ValueError, match="usage"):
        usage_of({"model": "claude-sonnet-5-5", "content": []})


# --- per run, and the driver's port -------------------------------------------


def call(seq: int, run: str = "r1", **fields: JsonValue) -> Event:
    return make(
        "model_call",
        seq,
        run=run,
        message=response(**fields),
        cost_usd=99,  # advisory: a report re-prices from tokens
    )


def test_a_log_is_priced_per_run_from_tokens_alone() -> None:
    events = [
        call(1),
        call(2, cache_read_input_tokens=9200),
        call(3, run="r2"),
    ]
    costs = tally(events, TABLE)
    first = D("0.00368") + D("0.00312")  # 1840 in, 312 out
    assert costs["r1"].calls == 2
    assert costs["r1"].tokens == Tokens(3680, 624, 9200)
    assert costs["r1"].usd == first * 2 + D("0.00092")
    assert (costs["r2"].calls, costs["r2"].usd) == (1, first)


def test_the_table_prices_for_the_driver_and_says_which_version() -> None:
    pricing = TablePricing(TABLE, count_input=lambda _: 1000)
    got, version = pricing.cost(response(cache_read_input_tokens=9200))
    assert (got, version) == (D("0.00772"), TABLE.version)
    request = ModelRequest("claude-sonnet-5-5", "s", [], (), max_tokens=8192)
    # 1000 known input tokens at $2 and 8192 output tokens at $10, per million.
    assert pricing.worst_case(request) == D("0.083920")
    with pytest.raises(UnknownModelError):
        pricing.worst_case(ModelRequest("nope", "s", [], (), max_tokens=1))


def test_the_table_loads_from_json_with_exact_decimals() -> None:
    text = TABLE.model_dump_json()
    assert PricingTable.model_validate_json(text) == TABLE


# --- spans --------------------------------------------------------------------


def recorded() -> list[Event]:
    return [
        make("run_started", 0),
        make("turn_started", 1, turn=1),
        make(
            "model_call",
            2,
            message=response(cache_read_input_tokens=9200),
            duration_ms=500,
            cost_usd=0.00772,
            pricing_version="2026-11-15",
        ),
        make("tool_requested", 3, tool_use_id="t1", name="search"),
        make("tool_completed", 4, tool_use_id="t1", duration_ms=1500),
        make("turn_started", 5, turn=2),
        make("model_call", 6, message=response(), duration_ms=250),
        make("run_finished", 7),
    ]


def shape(log: list[Event]) -> list[tuple[str, str | None, str, int]]:
    return [(s.id, s.parent, s.name, s.duration_ms) for s in spans(log)]


def test_spans_of_a_recorded_log_are_stable() -> None:
    run = "s1/r1/run:r1"
    assert shape(recorded()) == [
        (run, None, "run", 7000),
        ("s1/r1/turn:1", run, "turn 1", 4000),
        ("s1/r1/model_call:2", "s1/r1/turn:1", "chat claude-sonnet-5-5", 500),
        ("s1/r1/tool_call:t1", "s1/r1/turn:1", "execute_tool search", 1500),
        ("s1/r1/turn:5", run, "turn 2", 2000),
        ("s1/r1/model_call:6", "s1/r1/turn:5", "chat claude-sonnet-5-5", 250),
    ]
    assert spans(recorded()) == spans(recorded())


def test_spans_of_a_log_read_back_from_disk_match(tmp_path: Path) -> None:
    path = tmp_path / "s1.jsonl"
    with open_writer(path) as writer:
        for event in recorded():
            writer.append(event)
    assert spans(read_events(path).events) == spans(recorded())


def test_a_model_span_carries_the_gen_ai_attributes_and_its_tokens() -> None:
    model = next(s for s in spans(recorded()) if s.kind == "model_call")
    assert model.attributes == {
        "gen_ai.operation.name": "chat",
        "gen_ai.response.model": "claude-sonnet-5-5",
        "gen_ai.response.id": "msg_1",
        "gen_ai.response.finish_reasons": ["end_turn"],
        "gen_ai.usage.input_tokens": 1840 + 9200,
        "gen_ai.usage.output_tokens": 312,
        "gen_ai.usage.cache_read.input_tokens": 9200,
        "gen_ai.usage.cache_creation.input_tokens": 0,
        "loomcore.cost_usd": 0.00772,
        "loomcore.pricing_version": "2026-11-15",
    }
    tool = next(s for s in spans(recorded()) if s.kind == "tool_call")
    assert tool.attributes["gen_ai.tool.name"] == "search"
    assert tool.attributes["gen_ai.tool.call.id"] == "t1"
    assert tool.attributes["loomcore.tool.ok"] is True


def test_a_call_with_no_logged_duration_falls_back_to_the_timestamps() -> None:
    log = [
        make("run_started", 0),
        make("turn_started", 1, turn=1),
        make("tool_requested", 2, tool_use_id="t1", name="search"),
        make("tool_completed", 5, tool_use_id="t1", duration_ms=0),
    ]
    tool = next(s for s in spans(log) if s.kind == "tool_call")
    assert tool.duration_ms == 3000


def test_a_run_with_no_finish_is_open_and_a_fork_is_its_own_trace() -> None:
    log = [
        make("run_started", 0),
        make("turn_started", 1, turn=1),
        make("run_forked", 0, run="r2", **{"from": {"runId": "r1", "seq": 1}}),
        make("turn_started", 1, run="r2", turn=1),
    ]
    runs = [s for s in spans(log) if s.kind == "run"]
    assert [s.id for s in runs] == ["s1/r1/run:r1", "s1/r2/run:r2"]
    assert runs[0].attributes["loomcore.run.reason"] is None


def test_a_log_with_no_events_has_no_spans() -> None:
    assert spans([]) == []
