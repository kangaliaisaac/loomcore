"""The committed charts are what `make charts` draws, from the bench's numbers."""

import re
import tomllib
from pathlib import Path

from charts import (
    DARK,
    LIGHT,
    MEASURED_FONT,
    SANS,
    eviction_chart,
    eviction_score,
    files,
    rounded_points,
)

ROOT = Path(__file__).parent.parent


def texts(svg: bytes) -> list[str]:
    return re.findall(r">([^<>]+)</text>", svg.decode())


def test_the_committed_charts_are_what_the_script_draws() -> None:
    for path, svg in files().items():
        assert path.read_bytes() == svg, (
            f"{path.name} is out of date: run `make charts`"
        )


def test_a_chart_has_a_title_labelled_axes_and_its_values() -> None:
    result = eviction_score()
    shown = texts(eviction_chart(result, LIGHT))
    assert f"Faults on {result.avoidable} repeat requests" in shown
    assert "Faults (count)" in shown
    assert {str(result.policy), str(result.furthest)} <= set(shown)
    assert {"Context store's rule", "Furthest in future"} <= set(shown)
    assert any(line.endswith("Lower is better.") for line in shown)


def test_the_light_and_dark_charts_say_the_same_in_their_own_colours() -> None:
    result = eviction_score()
    light, dark = eviction_chart(result, LIGHT), eviction_chart(result, DARK)
    assert texts(light) == texts(dark)
    assert LIGHT.surface.encode() in light
    assert DARK.surface.encode() in dark
    assert DARK.surface.encode() not in light
    assert LIGHT.surface.encode() not in dark


def test_the_charts_ask_for_the_system_sans_and_not_only_the_measuring_font() -> None:
    for svg in files().values():
        assert SANS in svg
        assert MEASURED_FONT not in svg


def test_a_short_or_empty_bar_stays_above_the_baseline_and_inside_its_width() -> None:
    for top in (0.0, 3.0, 201.0):
        vertices = rounded_points(0, 1, top, 0.6, 7.7)
        assert min(y for _, y in vertices) >= 0.0
        assert max(y for _, y in vertices) <= top
        assert all(0.0 <= x <= 1.0 for x, _ in vertices)


def test_the_readme_shows_both_charts_with_their_numbers_and_version() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    result = eviction_score()
    for path in files():
        assert f"docs/charts/{path.name}" in readme
    alt = (
        f"Bar chart: on {result.avoidable} repeat requests, the context store's rule "
        f"faults {result.policy} times and furthest in future {result.furthest} times."
    )
    assert alt in readme
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert f"`loomcore` {version}" in readme, "the chart caption names an older version"
