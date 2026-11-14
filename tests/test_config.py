"""A configuration's digest covers what the model saw, and only what is set."""

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
