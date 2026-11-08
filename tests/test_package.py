"""The package imports and reports its version."""

import loomcore


def test_package_has_a_version() -> None:
    assert loomcore.__version__ == "0.0.0"
