"""The ports that need nothing from the caller."""

from collections.abc import Sequence
from dataclasses import fields
from datetime import timedelta
from decimal import Decimal

import pytest

from loomcore.events import JsonObject
from loomcore.observability.cost import TablePricing, vendored_table
from loomcore.ports import ModelRequest, Ports
from loomcore.runtime import (
    SystemPorts,
    estimate_request,
    estimate_tokens,
    system_ports,
)

MESSAGE: JsonObject = {"role": "user", "content": "hello"}


def estimated() -> SystemPorts:
    return system_ports(count_tokens=estimate_tokens, count_request=estimate_request)


def test_system_ports_are_the_ports_a_caller_has_no_say_in() -> None:
    from_the_caller = {"model", "tool", "read_only", "access", "notices"}
    assert set(SystemPorts.__annotations__) | from_the_caller == {
        f.name for f in fields(Ports)
    }
    assert set(estimated()) == set(SystemPorts.__annotations__)


def test_the_counters_have_no_default_and_the_ones_given_are_the_ones_used() -> None:
    with pytest.raises(TypeError):
        system_ports()  # type: ignore[call-arg]

    def count(messages: Sequence[JsonObject]) -> int:
        return 7

    def count_request(request: ModelRequest) -> int:
        return 11

    ports = system_ports(count_tokens=count, count_request=count_request)
    pricing = ports["pricing"]
    assert ports["count_tokens"] is count
    assert isinstance(pricing, TablePricing)
    assert pricing.count_input is count_request


def test_a_longer_history_counts_more_tokens() -> None:
    longer: list[JsonObject] = [MESSAGE, {"role": "assistant", "content": "x" * 300}]
    assert 0 < estimate_tokens([MESSAGE]) < estimate_tokens(longer)


def test_the_estimate_is_high_for_prose_and_low_for_dense_text() -> None:
    prose = "The quarterly report shows revenue grew while costs held steady. " * 20
    assert estimate_tokens([{"role": "user", "content": prose}]) > len(prose) // 4
    # A hash is about two characters a token, so the heuristic undercounts it; the
    # spec says so and tells a caller who relies on the budget to pass its own.
    digest = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    assert (
        estimate_tokens([{"role": "user", "content": digest * 20}])
        < len(digest) * 20 // 2
    )


def test_a_request_counts_its_system_prompt_messages_and_tools() -> None:
    def request(system: str, tools: tuple[JsonObject, ...]) -> ModelRequest:
        return ModelRequest("m", system, [MESSAGE], tools, max_tokens=8)

    base = estimate_request(request("s", ()))
    assert estimate_request(request("s" * 99, ())) > base
    assert estimate_request(request("s", ({"name": "t" * 99},))) > base


def test_the_clock_is_utc_and_moves_with_the_monotonic_clock() -> None:
    ports = estimated()
    first, later = ports["clock"](), ports["clock"]()
    assert first.utcoffset() == timedelta(0)
    assert first <= later
    assert ports["monotonic"]() <= ports["monotonic"]()


def test_run_ids_differ_and_the_jitter_is_a_fraction() -> None:
    ports = estimated()
    assert ports["new_run_id"]() != ports["new_run_id"]()
    assert 0 <= ports["random"]() < 1


def test_pricing_is_the_vendored_table_on_the_pessimistic_estimate() -> None:
    pricing = estimated()["pricing"]
    assert isinstance(pricing, TablePricing)
    assert pricing.count_input is estimate_request
    assert pricing.table.version == vendored_table().version
    request = ModelRequest("claude-haiku-5-5", "", [MESSAGE], (), max_tokens=100)
    known = estimate_request(request)
    # Haiku: $0.10 in and $0.50 out per million tokens.
    assert (
        pricing.worst_case(request)
        == (known * Decimal("0.10") + 100 * Decimal("0.50")) / 10**6
    )


def test_the_cost_ceiling_counts_two_characters_a_token() -> None:
    digest = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    request = ModelRequest(
        "m", "s", [{"role": "user", "content": digest * 20}], (), max_tokens=8
    )
    # At least a hash's length over two: the choice of two is an assumption about
    # the tokenizer (spec 1), which this pins and does not measure.
    assert estimate_request(request) >= len(digest) * 20 // 2
