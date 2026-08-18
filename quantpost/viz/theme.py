"""One chart style for every post, in light and dark.

The palette below is a validated categorical set: the slot **order** is the
colour-blind-safety mechanism, not decoration, so `series_colors()` hands them
out in fixed order and never cycles. Two consequences enforced in code:

* `SCATTER_MAX = 3` — for forms where every pair of series can appear adjacent
  (scatter, bubble, small multiples), only the first three slots clear the
  all-pairs separation floors. Past three, fold into "Other" or facet.
* `CATEGORICAL_MAX = 8` — a ninth series is never a generated hue.

Aqua, yellow and magenta sit below 3:1 contrast on the light surface, so charts
using those slots must carry direct labels or a table (the "relief rule"); the
helpers here default to direct labels for that reason.

Dark mode is a separately chosen set of steps for the dark surface, not an
inverted light palette.
"""

from __future__ import annotations

import pathlib as _pathlib
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt

LIGHT_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DARK_SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

CATEGORICAL_MAX = 8
SCATTER_MAX = 3           # all-pairs forms: only the first three slots validate

SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                   "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                   "#184f95", "#104281", "#0d366b"]

STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}


@dataclass(frozen=True)
class Mode:
    name: str
    surface: str
    page: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    diverging_mid: str



# --- the one bundled face -----------------------------------------------------
# Everything in this module uses the system sans. `charts.sketch_card` is the sole
# exception, and it is bundled rather than assumed so the rendered card is identical
# wherever the repo is checked out. Missing font degrades to the system sans.
_FONT_DIR = _pathlib.Path(__file__).with_name("fonts")
SKETCH_FONT_FILE = _FONT_DIR / "PatrickHand-Regular.ttf"
_sketch_family: list[str] | None = None


def sketch_family() -> list[str]:
    """Font family list for the hand-drawn card, most specific first.

    Registers the bundled face with matplotlib on first call. Returns a list whose
    last entry is always a system fallback, so the caller never has to branch on
    whether the font is present.
    """
    global _sketch_family
    if _sketch_family is None:
        fallback = ["DejaVu Sans"]
        if SKETCH_FONT_FILE.exists():
            try:
                _fm.fontManager.addfont(str(SKETCH_FONT_FILE))
                name = _fm.FontProperties(fname=str(SKETCH_FONT_FILE)).get_name()
                _sketch_family = [name] + fallback
            except Exception:
                _sketch_family = fallback
        else:
            _sketch_family = fallback
    return list(_sketch_family)


def sketch_font_available() -> bool:
    """Whether the drawn face actually loaded, for tests and diagnostics."""
    return len(sketch_family()) > 1


LIGHT = Mode("light", "#fcfcfb", "#f9f9f7", "#0b0b0b", "#52514e", "#898781",
             "#e1e0d9", "#c3c2b7", tuple(LIGHT_SERIES), "#f0efec")
DARK = Mode("dark", "#1a1a19", "#0d0d0d", "#ffffff", "#c3c2b7", "#898781",
            "#2c2c2a", "#383835", tuple(DARK_SERIES), "#383835")

MODES = {"light": LIGHT, "dark": DARK}


def series_colors(n: int, mode: str = "light", *, all_pairs: bool = False) -> list[str]:
    """First `n` categorical colours in fixed slot order.

    `all_pairs=True` for scatter/bubble/small-multiple forms, where the cap is 3.
    """
    m = MODES[mode]
    cap = SCATTER_MAX if all_pairs else CATEGORICAL_MAX
    if n > cap:
        raise ValueError(
            f"{n} series exceeds the validated cap of {cap} for "
            f"{'all-pairs' if all_pairs else 'adjacent-pairs'} forms. Fold the "
            "tail into 'Other', or facet into small multiples.")
    return list(m.series[:n])


def sequential(n: int) -> list[str]:
    """`n` steps of the single-hue blue ramp, light to dark."""
    if n < 2:
        return [SEQUENTIAL_BLUE[7]]
    idx = [round(i * (len(SEQUENTIAL_BLUE) - 1) / (n - 1)) for i in range(n)]
    return [SEQUENTIAL_BLUE[i] for i in idx]


def sequential_cmap(mode: str = "light"):
    """Matplotlib colormap for continuous magnitude (heatmaps, spacetime plots)."""
    from matplotlib.colors import LinearSegmentedColormap
    stops = SEQUENTIAL_BLUE if mode == "light" else SEQUENTIAL_BLUE[::-1]
    return LinearSegmentedColormap.from_list(f"qp_seq_{mode}", stops)


def diverging_cmap(mode: str = "light"):
    """Blue <-> red with a neutral grey midpoint. Never a rainbow."""
    from matplotlib.colors import LinearSegmentedColormap
    m = MODES[mode]
    return LinearSegmentedColormap.from_list(
        f"qp_div_{mode}", ["#104281", m.series[0], m.diverging_mid,
                           m.series[7], "#8f1f1f"])


def apply(mode: str = "light", *, figsize: tuple[float, float] = (7.2, 4.0),
          dpi: int = 200) -> Mode:
    """Install the theme into matplotlib rcParams. Returns the Mode."""
    m = MODES[mode]
    mpl.rcParams.update({
        "figure.figsize": figsize,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "figure.facecolor": m.surface,
        "savefig.facecolor": m.surface,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
        "axes.facecolor": m.surface,
        "axes.edgecolor": m.axis,
        "axes.linewidth": 0.8,
        "axes.labelcolor": m.ink_secondary,
        "axes.titlecolor": m.ink,
        "axes.titlesize": 11.5,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.prop_cycle": mpl.cycler(color=list(m.series)),
        "grid.color": m.grid,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "xtick.color": m.muted,
        "ytick.color": m.muted,
        "xtick.labelcolor": m.ink_secondary,
        "ytick.labelcolor": m.ink_secondary,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 0.0,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "legend.fontsize": 9.0,
        "legend.labelcolor": m.ink_secondary,
        "legend.handlelength": 1.4,
        "legend.borderpad": 0.0,
        "legend.columnspacing": 1.4,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "font.size": 9.5,
        "text.color": m.ink,
        "axes.unicode_minus": False,
    })
    return m


def _axes_frac_height_in_points(ax, renderer) -> float:
    return ax.get_window_extent(renderer).height * 72.0 / ax.figure.dpi


def finish(ax, *, title: str | None = None, subtitle: str | None = None,
           source: str | None = None, mode: str = "light",
           legend: bool = True, legend_ncol: int | None = None,
           wrap_subtitle: int = 96) -> None:
    """Stack title, subtitle, legend and source without collisions.

    Offsets are *measured*, not guessed. Every artist above the plot is drawn,
    its extent read back in axes coordinates, and the next one placed above it —
    because a hard-coded `1.02` works for one chart and overlaps on the next as
    soon as the legend wraps to two rows or the tick labels get taller. Same for
    the source note, which is placed below the axes' tight bounding box so it
    clears the tick labels and the x-label whatever they happen to be.

    The legend sits above the plot rather than to the right, so the plot keeps
    its full width in a narrow blog column, and it is always present for >= 2
    series — identity is never carried by colour alone.
    """
    import textwrap

    m = MODES[mode]
    fig = ax.figure

    handles, labels = ax.get_legend_handles_labels()
    leg = None
    if legend and len(labels) >= 2:
        leg = ax.legend(handles, labels, loc="lower left",
                        bbox_to_anchor=(0.0, 1.012),
                        ncol=legend_ncol or min(len(labels), 4))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    top = 1.012
    if leg is not None:
        bb = leg.get_window_extent(renderer).transformed(inv)
        top = max(top, bb.y1)

    sub_artist = None
    if subtitle:
        text = "\n".join(textwrap.wrap(subtitle, wrap_subtitle)) \
            if wrap_subtitle else subtitle
        sub_artist = ax.text(0.0, top + 0.022, text, transform=ax.transAxes,
                             ha="left", va="bottom", fontsize=9.0,
                             color=m.ink_secondary, linespacing=1.35)
        fig.canvas.draw()
        top = sub_artist.get_window_extent(renderer).transformed(inv).y1

    if title:
        pad_pts = max((top - 1.0), 0.0) * _axes_frac_height_in_points(
            ax, renderer) + 7.0
        ax.set_title(title, color=m.ink, pad=pad_pts)

    if source:
        fig.canvas.draw()
        tight = ax.get_tightbbox(renderer).transformed(inv)
        ax.text(0.0, tight.y0 - 0.045, source, transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color=m.muted)


def save(fig, path, *, mode: str = "light", close: bool = True) -> str:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, facecolor=MODES[mode].surface)
    if close:
        plt.close(fig)
    return str(p)
