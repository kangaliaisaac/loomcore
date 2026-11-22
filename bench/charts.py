"""Charts drawn from the repo's own data: `make charts` writes `docs/charts/`.

One chart so far: the eviction bench's seeded workload, the context store's rule
against furthest in future (see `eviction.py`). Each chart is drawn twice, for a
light and a dark page, and the README shows the pair with a `<picture>`. A test
regenerates them and fails when a committed file differs, so a chart cannot drift
from the numbers.

The SVG is deterministic on purpose: a bundled font for layout, no date, a fixed
hash salt, and the layout set by hand. The drawing is plain: thin marks with a
rounded data end, a hairline grid, text in ink colours, and one accent colour
against a gray for the baseline.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.style
from eviction import SEEDED_BUDGET, Score, score, seeded_workload
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as DrawPath

OUT = Path(__file__).parent.parent / "docs" / "charts"
SIZE = (7.2, 3.8)  # inches at 72 dpi, so one drawing unit is one point
BAR = 18.0  # bar thickness in points: about 24 CSS pixels when the SVG is shown
ROUND = 4.0  # radius of the rounded data end, in points
# The layout is measured with the bundled DejaVu Sans, which is wider than the system
# sans the file asks the viewer for, so text never runs past its box.
MEASURED_FONT = b"font-family: 'DejaVu Sans'"
SANS = b"font-family: system-ui, -apple-system, 'Segoe UI', 'DejaVu Sans', sans-serif"


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str  # titles and values
    ink2: str  # subtitle
    muted: str  # axis text
    grid: str
    base: str  # the baseline
    accent: str  # the series the chart is about
    context: str  # the series it is compared with


LIGHT = Theme(
    "light", "#fcfcfb", "#0b0b0b", "#52514e", "#6b6a65", "#e1e0d9", "#c3c2b7",
    "#2a78d6", "#898781",
)  # fmt: skip
DARK = Theme(
    "dark", "#1a1a19", "#f0efec", "#c3c2b7", "#a8a79f", "#2c2c2a", "#383835",
    "#3987e5", "#898781",
)  # fmt: skip
THEMES = (LIGHT, DARK)


def eviction_score() -> Score:
    return score(seeded_workload(), SEEDED_BUDGET)


def rounded_top(x0: float, x1: float, top: float, rx: float, ry: float) -> DrawPath:
    """A bar from 0 to `top`: square at the baseline, rounded at the data end."""
    move, line, curve, close = (
        DrawPath.MOVETO,
        DrawPath.LINETO,
        DrawPath.CURVE3,
        DrawPath.CLOSEPOLY,
    )
    points = [
        ((x0, 0.0), move),
        ((x0, top - ry), line),
        ((x0, top), curve),
        ((x0 + rx, top), curve),
        ((x1 - rx, top), line),
        ((x1, top), curve),
        ((x1, top - ry), curve),
        ((x1, 0.0), line),
        ((x0, 0.0), close),
    ]
    return DrawPath([p for p, _ in points], [c for _, c in points])


def draw_bars(ax: Axes, theme: Theme, values: Sequence[int]) -> None:
    """Two bars: the first is the series the chart is about, the second the baseline."""
    box = ax.get_window_extent()
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    per_x = box.width / (x_hi - x_lo)  # points per data unit
    per_y = box.height / (y_hi - y_lo)
    colours = (theme.accent, theme.context)
    for i, (value, colour) in enumerate(zip(values, colours, strict=True)):
        half = BAR / per_x / 2
        path = rounded_top(i - half, i + half, value, ROUND / per_x, ROUND / per_y)
        ax.add_patch(PathPatch(path, facecolor=colour, edgecolor="none", zorder=3))
        ax.text(
            i, value + 4 / per_y, f"{value}", ha="center", va="bottom",
            color=theme.ink, fontsize=11, fontweight="bold", zorder=4,
        )  # fmt: skip


def style_axes(ax: Axes, theme: Theme, top: int) -> None:
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, top, 50))
    ax.set_xticks([0, 1], ["Context store's rule", "Furthest in future"])
    ax.set_ylabel("Faults (count)", color=theme.muted, fontsize=9)
    ax.tick_params(axis="y", colors=theme.muted, labelsize=9, length=0)
    ax.tick_params(axis="x", colors=theme.ink2, labelsize=10, length=0, pad=8)
    ax.grid(axis="y", color=theme.grid, linewidth=0.8, linestyle="solid", zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(theme.surface)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.base)
    ax.spines["bottom"].set_linewidth(0.8)


def eviction_chart(result: Score, theme: Theme) -> bytes:
    """The chart as SVG bytes."""
    title = f"Faults on {result.avoidable} repeat requests"
    sub = (
        "The context store's rule against furthest in future, the baseline.\n"
        f"Seeded workload: {result.requests} requests, a {SEEDED_BUDGET:,}-token "
        "budget. Lower is better."
    )
    # Start from matplotlib's defaults, not from the reader's matplotlibrc, so the
    # bytes are the same on every machine.
    with (
        mpl.style.context("default"),
        mpl.rc_context(
            {
                "svg.fonttype": "none",
                "svg.hashsalt": "loomcore",
                "font.family": "DejaVu Sans",
            }
        ),
    ):
        fig = Figure(figsize=SIZE, dpi=72, facecolor=theme.surface)
        ax = fig.add_axes((0.1, 0.2, 0.87, 0.5))
        style_axes(ax, theme, result.avoidable)
        draw_bars(ax, theme, (result.policy, result.furthest))
        fig.text(0.02, 0.95, title, color=theme.ink, fontsize=13, va="top",
                 fontweight="bold")  # fmt: skip
        fig.text(0.02, 0.865, sub, color=theme.ink2, fontsize=8.5, va="top")
        buf = BytesIO()
        fig.savefig(
            buf, format="svg", facecolor=theme.surface,
            metadata={"Date": None, "Creator": "loomcore make charts", "Title": title},
        )  # fmt: skip
    return buf.getvalue().replace(MEASURED_FONT, SANS)


def files() -> dict[Path, bytes]:
    """Every chart file `make charts` writes, by path."""
    result = eviction_score()
    return {OUT / f"eviction-{t.name}.svg": eviction_chart(result, t) for t in THEMES}


def main(argv: Sequence[str]) -> None:
    if argv:
        raise SystemExit("usage: python bench/charts.py (no arguments)")
    OUT.mkdir(parents=True, exist_ok=True)
    for path, svg in files().items():
        path.write_bytes(svg)
        print(f"wrote {path.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main(sys.argv[1:])
