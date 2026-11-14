"""A configuration's digest covers what the model saw, and only what is set."""

import pytest
from fakes import CONFIG

from loomcore.ports import Config


def test_a_configuration_without_a_cache_prefix_keeps_its_digest() -> None:
    assert "cache_prefix" not in CONFIG.to_json()
    # the digest of the shape that existed before the field: it must not move
    same = Config(model="m", system="be brief", tools=({"name": "search"},))
    assert same.digest() == CONFIG.digest()


def test_a_cache_prefix_is_in_the_log_and_in_the_digest_when_set() -> None:
    marked = Config(model="m", system="be brief", tools=CONFIG.tools, cache_prefix=4)
    assert marked.to_json()["cache_prefix"] == 4
    assert marked.digest() != CONFIG.digest()


def test_a_cache_prefix_must_lie_within_the_system_prompt() -> None:
    for bad in (0, -1, 9):  # the system prompt "be brief" is 8 characters
        with pytest.raises(ValueError, match="cache_prefix"):
            Config(model="m", system="be brief", cache_prefix=bad)
    assert Config(model="m", system="be brief", cache_prefix=8).cache_prefix == 8
