"""The package imports."""

import ast
import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

import loomcore


def test_package_imports() -> None:
    assert loomcore.__doc__


def test_each_name_is_the_object_it_stands_for() -> None:
    # Import every module first: once a submodule is loaded Python sets it as an
    # attribute of the package, and `__getattr__` is no longer asked for that name.
    for module, _ in loomcore._EXPORTS.values():
        importlib.import_module(module)
    for name, (module, attribute) in loomcore._EXPORTS.items():
        loaded = importlib.import_module(module)
        expected = loaded if attribute is None else getattr(loaded, attribute)
        assert getattr(loomcore, name) is expected, name
    assert isinstance(loomcore.finish, types.ModuleType)  # a module, by design


def test_the_names_a_type_checker_sees_are_the_names_served() -> None:
    source = Path(loomcore.__file__ or "").read_text(encoding="utf-8")
    block = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
    )
    seen: dict[str, tuple[str, str | None]] = {}
    for node in block.body:
        assert isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names:
            module = (
                f"{node.module}.{alias.name}"
                if node.module == "loomcore"
                else node.module
            )
            attribute = None if node.module == "loomcore" else alias.name
            seen[alias.asname or alias.name] = (module, attribute)
    assert seen == loomcore._EXPORTS


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
