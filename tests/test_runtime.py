"""The ports that need nothing from the caller."""

from dataclasses import fields
from datetime import timedelta

from loomcore.events import JsonObject
from loomcore.observability.cost import TablePricing
from loomcore.ports import ModelRequest, Ports
from loomcore.runtime import (
    SystemPorts,
    estimate_request,
    estimate_tokens,
    system_ports,
)

MESSAGE: JsonObject = {"role": "user", "content": "hello"}


def test_system_ports_are_the_ports_a_caller_has_no_say_in() -> None:
    from_the_caller = {"model", "tool", "read_only", "access", "notices"}
    assert set(SystemPorts.__annotations__) | from_the_caller == {
        f.name for f in fields(Ports)
    }
    assert set(system_ports()) == set(SystemPorts.__annotations__)


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
    ports = system_ports()
    first, later = ports["clock"](), ports["clock"]()
    assert first.utcoffset() == timedelta(0)
    assert first <= later
    assert ports["monotonic"]() <= ports["monotonic"]()


def test_run_ids_differ_and_the_jitter_is_a_fraction() -> None:
    ports = system_ports()
    assert ports["new_run_id"]() != ports["new_run_id"]()
    assert 0 <= ports["random"]() < 1


def test_pricing_is_the_vendored_table_on_the_estimate() -> None:
    pricing = system_ports()["pricing"]
    assert isinstance(pricing, TablePricing)
    request = ModelRequest("claude-haiku-5-5", "s", [MESSAGE], (), max_tokens=100)
    assert pricing.worst_case(request) > 0
