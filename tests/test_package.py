"""The package imports."""

import importlib
import subprocess
import sys
import types

import pytest

import loomcore


def test_package_imports() -> None:
    assert loomcore.__doc__


def test_every_public_name_is_there_and_none_hides_a_module() -> None:
    for name in loomcore.__all__:
        assert hasattr(loomcore, name), name
    for module in ("loomcore.replay", "loomcore.prompt", "loomcore.loop"):
        assert isinstance(importlib.import_module(module), types.ModuleType)
    assert isinstance(loomcore.finish, types.ModuleType)  # a module, by design


def test_the_names_listed_are_the_names_served() -> None:
    assert set(loomcore.__all__) == set(loomcore._EXPORTS)
    assert len(loomcore.__all__) == len(loomcore._EXPORTS)


def test_an_unknown_name_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        loomcore.nope  # noqa: B018


def test_importing_the_package_loads_no_driver_until_a_name_is_used() -> None:
    code = (
        "import sys, loomcore, loomcore.tools\n"
        "assert 'loomcore.loop' not in sys.modules\n"
        "loomcore.Driver\n"
        "assert 'loomcore.loop' in sys.modules"
    )
    assert (
        subprocess.run([sys.executable, "-I", "-c", code], check=False).returncode == 0
    )
