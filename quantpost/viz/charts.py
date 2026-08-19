"""Chart builders. Each one picks the form from the data's job, not from taste.

Every function returns `(fig, ax)` so a post can tweak, and each takes `alt` to
force you to write alt text — a chart with no alt text is a chart half of your
readers cannot read.

The dual-axis chart is deliberately absent. Two measures on different scales get
two panels (`small_multiples`) or a common index (`indexed_lines`). There is no
`twinx` in this module and there should not be one in your post.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from . import theme


@dataclass
class Figure:
    """A rendered figure plus everything the renderer needs to caption it."""
    path: str
    alt: str
    caption: str
    title: str = ""
    mode: str = "light"

    def markdown(self, base: str = "") -> str:
        """Reference the image, resolving the path for the target.

        `base` empty means a **Hugo page bundle**: the image sits beside
        `index.md`, so the reference is the bare filename. Emitting the build
        path here would embed an absolute local path in the published page, which
        renders as a broken image and stays invisible until someone loads the
        site. A non-empty `base` is an absolute URL prefix, for Medium.
        """
        name = self.path.replace("\\", "/").split("/")[-1]
        src = f"{base.rstrip('/')}/{name}" if base else name
        out = f"![{self.alt}]({src})"
        if self.caption:
            out += f"\n\n*{self.caption}*"
        return out


def _ends(ax, x, y, color, label, mode):
    """Direct-label the last point. Replaces reading a legend for line charts,
    and satisfies the relief rule for the low-contrast slots."""
    m = theme.MODES[mode]
    finite = np.isfinite(np.asarray(y, float))
    if not finite.any():
        return
    i = np.nonzero(finite)[0][-1]
    ax.plot([x[i]], [y[i]], marker="o", ms=4.5, color=color,
            markeredgecolor=m.surface, markeredgewidth=1.6, zorder=5)
    ax.annotate(f" {label}", (x[i], y[i]), textcoords="offset points",
                xytext=(6, 0), va="center", fontsize=8.5,
                color=m.ink_secondary, annotation_clip=False)


def lines(
    frame,
    *,
    title: str = "",
    subtitle: str = "",
    ylabel: str = "",
    xlabel: str = "",
    source: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    direct_labels: bool = True,
    highlight: str | None = None,
    logy: bool = False,
    logx: bool = False,
    ylim: tuple[float, float] | None = None,
    invert_x: bool = False,
    decorate=None,
    figsize: tuple[float, float] = (7.2, 4.0),
    path: str | None = None,
):
    """Change over time, one line per column. Never two y-axes.

    `decorate(fig, ax)` runs **before** the titles and source note are placed.
    Anything that changes the axes — a scale, a limit, an axis label, a reference
    line — has to happen first, because the source note is positioned from the
    axes' measured tight bounding box. Setting an x-label after the fact puts the
    label underneath the note.
    """
    theme.apply(mode, figsize=figsize)
    cols = list(frame.columns)
    colors = theme.series_colors(len(cols), mode)
    fig, ax = plt.subplots()
    x = frame.index
    for c, col in zip(cols, colors):
        emphasised = highlight is None or c == highlight
        ax.plot(x, frame[c].to_numpy(dtype=float), color=col, label=str(c),
                lw=2.0 if emphasised else 1.2,
                alpha=1.0 if emphasised else 0.45, zorder=3 if emphasised else 2)
    if direct_labels and len(cols) <= 4:
        for c, col in zip(cols, colors):
            _ends(ax, x, frame[c].to_numpy(dtype=float), col, str(c), mode)
        ax.margins(x=0.06)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    if invert_x:
        ax.invert_xaxis()
    if decorate is not None:
        decorate(fig, ax)
    theme.finish(ax, title=title, subtitle=subtitle, source=source, mode=mode,
                 legend=not (direct_labels and len(cols) <= 4))
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or title, caption, title, mode), (fig, ax)
    return fig, ax


def indexed_lines(frame, *, base_date=None, base_value: float = 100.0, **kw):
    """Two measures of different scale, made comparable by rebasing.

    This is the correct answer to the urge to add a second y-axis.
    """
    f = frame.dropna(how="all").copy()
    if base_date is None:
        first = f.apply(lambda s: s.first_valid_index())
        base_date = max(v for v in first if v is not None)
    base = f.loc[base_date]
    out = f.divide(base) * base_value
    kw.setdefault("ylabel", f"index, {base_date} = {base_value:g}")
    return lines(out, **kw)


def prediction_vs_truth(
    t,
    truth,
    predictions: dict[str, np.ndarray],
    *,
    title: str = "",
    subtitle: str = "",
    ylabel: str = "",
    xlabel: str = "",
    source: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    divergence_at: float | None = None,
    figsize: tuple[float, float] = (7.2, 4.0),
    path: str | None = None,
):
    """Truth as a thicker recessive band, each model as a thin line.

    Truth gets the muted ink rather than a categorical slot, so all eight slots
    stay available for models and the eye reads "reference" not "series 1".
    """
    m = theme.apply(mode, figsize=figsize)
    fig, ax = plt.subplots()
    ax.plot(t, np.asarray(truth, float), color=m.muted, lw=3.0, alpha=0.55,
            label="truth", zorder=2, solid_capstyle="round")
    colors = theme.series_colors(len(predictions), mode)
    for (name, p), col in zip(predictions.items(), colors):
        ax.plot(t, np.asarray(p, float), color=col, lw=1.8, label=name, zorder=3)
    if divergence_at is not None:
        ax.axvline(divergence_at, color=m.axis, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.annotate("forecast diverges", (divergence_at, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(5, -12), textcoords="offset points",
                    fontsize=8.0, color=m.muted)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    theme.finish(ax, title=title, subtitle=subtitle, source=source, mode=mode)
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or title, caption, title, mode), (fig, ax)
    return fig, ax


def small_multiples(
    panels: dict,
    *,
    title: str = "",
    subtitle: str = "",
    source: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    ncols: int = 2,
    sharex: bool = True,
    sharey: bool = False,
    figsize: tuple[float, float] | None = None,
    path: str | None = None,
):
    """One small panel per key. The right answer whenever series counts grow past
    what one frame can hold legibly, and the right answer to mixed units."""
    m = theme.apply(mode)
    n = len(panels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (7.2, 2.1 * nrows + 0.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex,
                             sharey=sharey, squeeze=False)
    colors = theme.series_colors(min(n, theme.CATEGORICAL_MAX), mode) \
        if n <= theme.CATEGORICAL_MAX else [m.series[0]] * n
    for i, (name, series) in enumerate(panels.items()):
        ax = axes[i // ncols][i % ncols]
        s = series
        x = getattr(s, "index", np.arange(len(s)))
        ax.plot(x, np.asarray(s, float), color=colors[i], lw=1.8)
        ax.set_title(str(name), fontsize=9.5, color=m.ink, loc="left", pad=6)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # Lay the figure out first, then stack the headings above the measured top of
    # the panel grid. Reserving a fixed fraction with `rect` collides with the
    # per-panel titles as soon as the grid shape changes.
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def fig_bbox(artists):
        boxes = [a.get_tightbbox(renderer) for a in artists
                 if a.get_visible()]
        boxes = [b for b in boxes if b is not None]
        from matplotlib.transforms import Bbox
        return Bbox.union(boxes).transformed(inv)

    grid = fig_bbox([a for row in axes for a in row])
    y = grid.y1 + 0.018
    if subtitle:
        import textwrap
        t = fig.text(0.0, y, "\n".join(textwrap.wrap(subtitle, 104)),
                     ha="left", va="bottom", fontsize=9.0,
                     color=m.ink_secondary, linespacing=1.35)
        fig.canvas.draw()
        y = t.get_tightbbox(renderer).transformed(inv).y1 + 0.014
    if title:
        fig.text(0.0, y, title, ha="left", va="bottom", fontsize=11.5,
                 color=m.ink, weight="medium")
    if source:
        fig.text(0.0, grid.y0 - 0.035, source, ha="left", va="top",
                 fontsize=7.5, color=m.muted)
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or title, caption, title, mode), (fig, axes)
    return fig, axes


def spacetime(
    field,
    *,
    x=None,
    t=None,
    title: str = "",
    subtitle: str = "",
    source: str = "",
    alt: str = "",
    caption: str = "",
    cbar_label: str = "",
    mode: str = "light",
    diverging: bool = True,
    symmetric: bool = True,
    figsize: tuple[float, float] = (7.2, 3.4),
    path: str | None = None,
):
    """Space-time heatmap for a PDE field. Diverging when the field is signed
    (a signed field on a sequential ramp hides the zero crossing)."""
    m = theme.apply(mode, figsize=figsize)
    u = np.asarray(field, float)
    fig, ax = plt.subplots()
    cmap = theme.diverging_cmap(mode) if diverging else theme.sequential_cmap(mode)
    vmax = np.nanmax(np.abs(u)) if symmetric and diverging else np.nanmax(u)
    vmin = -vmax if symmetric and diverging else np.nanmin(u)
    extent = None
    if x is not None and t is not None:
        extent = (float(np.min(x)), float(np.max(x)), float(np.min(t)),
                  float(np.max(t)))
    im = ax.imshow(u, aspect="auto", origin="lower", cmap=cmap, vmin=vmin,
                   vmax=vmax, extent=extent, interpolation="nearest")
    ax.grid(False)
    ax.set_xlabel("space $x$")
    ax.set_ylabel("time $t$")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=m.muted)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8.5, color=m.ink_secondary)
    theme.finish(ax, title=title, subtitle=subtitle, source=source, mode=mode,
                 legend=False)
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or title, caption, title, mode), (fig, ax)
    return fig, ax


def ranked_bars(
    labels,
    values,
    *,
    errors=None,
    title: str = "",
    subtitle: str = "",
    xlabel: str = "",
    source: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    signed: bool = False,
    sort: str = "auto",
    value_fmt: str = ",.3g",
    figsize: tuple[float, float] | None = None,
    path: str | None = None,
):
    """Horizontal ranked bars — the form for magnitude comparison across a
    handful of named things (attribution, importance, model scores).

    Signed values use the diverging pair so sign is visible without reading the
    axis; unsigned use one hue, because varying hue across bars of the same
    measure encodes nothing.

    `sort` controls the order. `"auto"` sorts signed data by magnitude and unsigned
    data by value, which is the right default for attribution — the biggest
    contributor first regardless of direction. Pass `"value"` when the *grouping* by
    sign is the finding rather than the ranking: sorting a mixed set by magnitude
    interleaves positives and negatives, and a caption that then says "the four
    positive ones" is describing a chart the reader is not looking at. `"none"`
    keeps the caller's order.

    `value_fmt` sets the label format. The default of three significant figures
    reads badly when the values span an order of magnitude — "-2.4" above "-0.0833"
    looks like carelessness rather than precision — so pass an explicit format when
    the bars are all the same kind of quantity.
    """
    if sort not in ("auto", "value", "magnitude", "none"):
        raise ValueError(f"unknown sort {sort!r}")
    m = theme.apply(mode)
    labels = list(labels)
    values = np.asarray(values, float)
    if sort == "none":
        order = np.arange(len(values))
    elif sort == "value" or (sort == "auto" and not signed):
        order = np.argsort(values)
    else:
        order = np.argsort(np.abs(values))
    labels = [labels[i] for i in order]
    values = values[order]
    # matplotlib accepts either a 1-D symmetric error or a (2, n) [lower, upper]
    # pair; both have to be reordered along the *bar* axis, not naively.
    errs = None
    if errors is not None:
        e = np.asarray(errors, float)
        errs = e[:, order] if e.ndim == 2 else e[order]
    figsize = figsize or (7.2, max(2.0, 0.34 * len(labels) + 1.2))
    fig, ax = plt.subplots(figsize=figsize)
    if signed:
        colors = [m.series[7] if v < 0 else m.series[0] for v in values]
    else:
        colors = [m.series[0]] * len(values)
    ax.barh(range(len(values)), values, color=colors, height=0.66,
            xerr=errs, error_kw={"ecolor": m.muted, "elinewidth": 1.0,
                                 "capsize": 0})
    ax.set_yticks(range(len(values)), labels, fontsize=8.5)
    ax.grid(False)
    ax.xaxis.grid(True, color=m.grid, lw=0.6)
    ax.set_axisbelow(True)
    if signed:
        ax.axvline(0.0, color=m.axis, lw=1.0)
    # Value labels clear the error bar, not just the bar end — otherwise the
    # number lands on top of the whisker exactly when the spread matters most.
    span = float(np.nanmax(np.abs(values))) or 1.0
    if errs is None:
        reach = np.zeros(len(values))
    elif errs.ndim == 2:
        reach = np.where(values >= 0, errs[1], errs[0])
    else:
        reach = errs
    for i, v in enumerate(values):
        off = (0.022 * span + float(reach[i])) * (1 if v >= 0 else -1)
        ax.annotate(f"{v:{value_fmt}}", (v + off, i), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.0,
                    color=m.ink_secondary)
    ax.margins(x=0.20)
    if xlabel:
        ax.set_xlabel(xlabel)
    theme.finish(ax, title=title, subtitle=subtitle, source=source, mode=mode,
                 legend=False)
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or title, caption, title, mode), (fig, ax)
    return fig, ax


def histogram(
    values,
    *,
    bins: int = 30,
    overlay: dict | None = None,
    mark: dict | None = None,
    series_label: str = "observed",
    title: str = "",
    subtitle: str = "",
    xlabel: str = "",
    ylabel: str = "probability density",
    source: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    figsize: tuple[float, float] = (7.2, 4.0),
    path: str | None = None,
):
    """Distribution of one quantity, optionally against a theoretical curve.

    `overlay` is `{label: (x, y)}` for a reference density — the whole point of a
    histogram in an analysis post is usually "is this spread more than chance
    predicts", and that question needs the chance curve drawn on top of it.
    `mark` is `{label: x}` for vertical annotations, e.g. where the winner landed.

    Density, not counts, so the overlay can be a probability density and the two
    are directly comparable without a second axis.
    """
    m = theme.apply(mode, figsize=figsize)
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    fig, ax = plt.subplots()
    counts, edges = np.histogram(v, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # Nameable, because "observed" is wrong whenever the overlay is also observed
    # — a size-biased version of the same data, say, rather than a theory curve.
    ax.bar(centres, counts, width=(edges[1] - edges[0]) * 0.92,
           color=m.series[0], label=series_label)
    for (label, (ox, oy)), col in zip((overlay or {}).items(),
                                      theme.series_colors(
                                          max(len(overlay or {}), 1), mode)[1:]
                                      or [m.series[1]]):
        ax.plot(ox, oy, color=col, lw=2.0, label=label)
    for label, x in (mark or {}).items():
        ax.axvline(x, color=m.series[7], lw=1.4)
        ax.annotate(label, (x, 0.97), xycoords=("data", "axes fraction"),
                    xytext=(6, 0), textcoords="offset points", ha="left",
                    va="top", fontsize=8.5, color=m.series[7])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    theme.finish(ax, title=title, subtitle=subtitle, source=source, mode=mode)
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or title, caption, title, mode), (fig, ax)
    return fig, ax


def table_image(
    rows,
    *,
    header,
    title: str = "",
    subtitle: str = "",
    source: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    bold_cols: tuple[int, ...] = (),
    bold_cells: set | None = None,
    align: str = "",
    row_height: float = 0.42,
    min_width: float = 4.0,
    path: str | None = None,
):
    """Render a small table as an image, in the same style as the charts.

    This exists because **Medium has no table support and strips table markup on
    paste** — a markdown table arrives as a run of plain text with the pipes gone.
    The alternatives are a GitHub Gist embed (selectable text, but a second place
    to keep the numbers in sync) or a picture. A picture rendered from the same
    data that produced the post keeps one source of truth, and matches the
    figures instead of looking like a screenshot of something else.

    Column widths are **measured**, not counted. A character-count model got this
    wrong the first time it met a table whose header was a long word over a short
    number — "unevenness" over "0.00" — and rendered "unevennessaverage gap" as one
    run of text, because proportional type does not care how many characters you
    used. Every cell is drawn, its rendered extent read back, and the figure width
    then set from the total, so the table is exactly as wide as its contents need.

    `align` is one character per column: "l" or "r" (default: first column left,
    the rest right, which is what a label-plus-numbers table wants).
    `bold_cols` emphasises whole columns; `bold_cells` takes `(row, col)` index
    pairs when only some cells in a column carry the finding — bolding a whole
    column when one of its values is the boring control tells the reader the
    control matters too.

    Keep it to a handful of rows. A table that needs scrolling is not a figure —
    it belongs in the repo as a CSV, with the headline numbers in the prose.
    """
    m = theme.apply(mode)
    body = [[str(c) for c in r] for r in rows]
    head = [str(c) for c in header]
    n_col = len(head)
    if any(len(r) != n_col for r in body):
        raise ValueError("every row must have the same number of cells as header")
    align = align or ("l" + "r" * (n_col - 1))
    if len(align) != n_col:
        raise ValueError(f"align needs {n_col} characters, got {align!r}")

    n_row = len(body)
    fig_h = row_height * (n_row + 1) + 1.1
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_row + 1)
    ax.grid(False)

    y_head = n_row + 0.45
    cells = []          # (artist, row_or_None, col)
    for j, text in enumerate(head):
        cells.append((ax.text(0.0, y_head, text, ha="left", va="center",
                              fontsize=9.0, color=m.ink_secondary), None, j))
    for i, row in enumerate(body):
        y = n_row - 0.5 - i
        for j, text in enumerate(row):
            strong = j in bold_cols or (i, j) in (bold_cells or set())
            cells.append((ax.text(0.0, y, text, ha="left", va="center",
                                  fontsize=9.5, color=m.ink,
                                  fontweight="bold" if strong else "normal"),
                          i, j))

    # Measure, then size the figure to what the type actually needs.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    col_px = [0.0] * n_col
    for artist, _i, j in cells:
        w = artist.get_window_extent(renderer).width
        col_px[j] = max(col_px[j], w)
    gap_px = 0.20 * fig.dpi
    need_px = sum(col_px) + gap_px * (n_col - 1)
    frac = ax.get_position().width
    fig.set_size_inches(max(min_width, need_px / (frac * fig.dpi)), fig_h)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_px = ax.get_window_extent(renderer).width

    edges = [0.0]
    for w in col_px:
        edges.append(edges[-1] + (w + gap_px) / axes_px)
    right = edges[-1] - gap_px / axes_px          # where the content actually ends

    for artist, _i, j in cells:
        if align[j] == "l":
            artist.set_position((edges[j], artist.get_position()[1]))
            artist.set_ha("left")
        else:
            artist.set_position((edges[j + 1] - gap_px / axes_px,
                                 artist.get_position()[1]))
            artist.set_ha("right")

    # Rule under the header, hairlines between rows: the minimum a table needs.
    ax.plot([0, right], [n_row, n_row], color=m.axis, lw=1.0, clip_on=False)
    for i in range(1, n_row):
        y = n_row - i
        ax.plot([0, right], [y, y], color=m.grid, lw=0.6, clip_on=False)

    theme.finish(ax, title=title, subtitle=subtitle, source=source, mode=mode,
                 legend=False)
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or title, caption, title, mode), (fig, ax)
    return fig, ax


def coefficient_matrix(
    matrix,
    *,
    row_labels,
    col_labels,
    title: str = "",
    subtitle: str = "",
    source: str = "",
    alt: str = "",
    caption: str = "",
    cbar_label: str = "",
    mode: str = "light",
    annotate: bool = True,
    annotate_threshold: float = 0.02,
    figsize: tuple[float, float] | None = None,
    path: str | None = None,
):
    """Signed coefficient matrix as a diverging heatmap.

    Diverging with a neutral midpoint, because zero is meaningful here: the
    *pattern* of which cells are non-zero is the result, and a sequential ramp
    would make small negative and small positive coefficients look alike.
    Cells below `annotate_threshold` in absolute value are left unlabelled so the
    sparsity reads as sparsity rather than as a wall of zeros.
    """
    m = theme.apply(mode)
    M = np.asarray(matrix, float)
    rows, cols = list(row_labels), list(col_labels)
    figsize = figsize or (max(7.2, 0.62 * len(cols) + 2.6),
                          0.52 * len(rows) + 2.3)
    fig, ax = plt.subplots(figsize=figsize)
    vmax = float(np.nanmax(np.abs(M))) or 1.0
    im = ax.imshow(M, cmap=theme.diverging_cmap(mode), vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    # Rotate only when the labels actually need it — rotated short labels are
    # harder to read than horizontal ones, not easier.
    rot = 0 if max(len(str(c)) for c in cols) <= 5 else 45
    ax.set_xticks(range(len(cols)), cols, fontsize=8.5, rotation=rot,
                  ha="center" if rot == 0 else "right",
                  rotation_mode=None if rot == 0 else "anchor")
    ax.set_yticks(range(len(rows)), rows, fontsize=9.0)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    # 2px surface gap between cells, per the mark spec for adjacent fills.
    ax.grid(which="minor", color=m.surface, linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isfinite(v) or abs(v) < annotate_threshold * vmax:
                    continue
                strong = abs(v) > 0.55 * vmax
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color=m.surface if strong else m.ink)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=m.muted)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8.5, color=m.ink_secondary)
    theme.finish(ax, title=title, subtitle=subtitle, source=source, mode=mode,
                 legend=False)
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or title, caption, title, mode), (fig, ax)
    return fig, ax


def sensitivity_surface(
    matrix,
    *,
    xticks,
    yticks,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    subtitle: str = "",
    source: str = "",
    alt: str = "",
    caption: str = "",
    cbar_label: str = "",
    mode: str = "light",
    mark_best: bool = True,
    lower_is_better: bool = True,
    figsize: tuple[float, float] = (6.4, 4.2),
    path: str | None = None,
):
    """Hyperparameter grid as a heatmap, with the optimum ringed.

    Showing the surface instead of the winning number is the difference between
    "we used rho = 0.9" and "anything in 0.7-1.1 works, and here is the cliff".
    """
    m = theme.apply(mode, figsize=figsize)
    M = np.asarray(matrix, float)
    fig, ax = plt.subplots()
    # Always light -> dark, so dark always means "large value". Reversing the ramp
    # for a lower-is-better metric would make "dark = good" in one figure and
    # "light = good" in the next; the ring marks the optimum instead.
    im = ax.imshow(M, aspect="auto", origin="lower",
                   cmap=theme.sequential_cmap("light"),
                   interpolation="nearest")
    ax.set_xticks(range(len(xticks)), [f"{v:g}" for v in xticks], fontsize=8.5)
    ax.set_yticks(range(len(yticks)), [f"{v:g}" for v in yticks], fontsize=8.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(False)
    if mark_best and np.isfinite(M).any():
        idx = np.nanargmin(M) if lower_is_better else np.nanargmax(M)
        r, c = np.unravel_index(idx, M.shape)
        ax.plot([c], [r], marker="o", ms=11, mfc="none",
                markeredgecolor=m.ink, markeredgewidth=1.8)
        ax.annotate(f"best {M[r, c]:.3g}", (c, r), xytext=(10, 8),
                    textcoords="offset points", fontsize=8.0, color=m.ink)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=m.muted)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8.5, color=m.ink_secondary)
    theme.finish(ax, title=title, subtitle=subtitle, source=source, mode=mode,
                 legend=False)
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or title, caption, title, mode), (fig, ax)
    return fig, ax


def error_growth(
    t,
    curves: dict[str, np.ndarray],
    *,
    threshold: float | None = None,
    title: str = "",
    subtitle: str = "",
    xlabel: str = "Lyapunov times",
    ylabel: str = "normalised error",
    source: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    logy: bool = True,
    ylim: tuple[float, float] | None = None,
    decorate=None,
    figsize: tuple[float, float] = (7.2, 4.0),
    path: str | None = None,
):
    """Forecast error against a Lyapunov-scaled clock — the honest way to show
    a chaotic-forecast result, because the x-axis is comparable across systems."""
    m = theme.apply(mode, figsize=figsize)
    fig, ax = plt.subplots()
    colors = theme.series_colors(len(curves), mode)
    for (name, c), col in zip(curves.items(), colors):
        ax.plot(t[:len(c)], np.asarray(c, float), color=col, lw=1.9, label=name)
    if threshold is not None:
        ax.axhline(threshold, color=m.axis, lw=1.0, ls=(0, (4, 3)))
        ax.annotate(f"VPT threshold {threshold:g}", (0.005, threshold),
                    xycoords=("axes fraction", "data"), xytext=(0, 5),
                    textcoords="offset points", fontsize=8.0, color=m.muted)
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if decorate is not None:
        decorate(fig, ax)
    theme.finish(ax, title=title, subtitle=subtitle, source=source, mode=mode)
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or title, caption, title, mode), (fig, ax)
    return fig, ax


def phase_portrait(
    xy: dict[str, np.ndarray],
    *,
    title: str = "",
    subtitle: str = "",
    xlabel: str = "", ylabel: str = "",
    source: str = "", alt: str = "", caption: str = "",
    mode: str = "light",
    figsize: tuple[float, float] = (5.2, 4.6),
    path: str | None = None,
):
    """Attractor projections. All-pairs form, so the cap is three series."""
    m = theme.apply(mode, figsize=figsize)
    colors = theme.series_colors(len(xy), mode, all_pairs=True)
    fig, ax = plt.subplots()
    for (name, arr), col in zip(xy.items(), colors):
        a = np.asarray(arr, float)
        ax.plot(a[:, 0], a[:, 1], color=col, lw=0.7, alpha=0.85, label=name)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, color=m.grid, lw=0.6)
    ax.set_axisbelow(True)
    theme.finish(ax, title=title, subtitle=subtitle, source=source, mode=mode)
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or title, caption, title, mode), (fig, ax)
    return fig, ax


CARD_SIZE = (8.0, 4.2)          # 1600 x 840 at dpi 200
CARD_DPI = 200
CARD_SAFE = 0.08                # crop-safe margin, top and bottom


def _card_base(mode: str, headline: str, footer: str = "",
               headline_size: float = 17.5):
    """The frame every preview card shares: geometry, headline, footer.

    One helper rather than five copies, because the numbers in here are the ones
    that drift: Medium's preview crops wider than this card's 1.9:1 and takes the
    difference off both edges, so everything lives inside an 8% margin top and
    bottom. A card whose headline sat at 0.94 in one function and 0.86 in another
    would lose its ascenders in exactly one of them, and only in the preview.

    Returns `(mode, figure, axes)` with the axes spanning the whole card in 0..1
    coordinates and no spines, ready to draw type on.
    """
    m = theme.apply(mode, figsize=CARD_SIZE, dpi=CARD_DPI)
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if footer:
        ax.text(0.965, 1.0 - CARD_SAFE - 0.02, footer, ha="right", va="top",
                fontsize=10.5, color=m.muted)
    if headline:
        # Measured, then shrunk to fit. A headline set at a fixed size fits on one
        # card and runs into the footer on the next, and the difference is one word.
        # The limit is 0.86 of the width because the footer owns the top right.
        art = ax.text(0.055, 1.0 - CARD_SAFE - 0.06, headline, ha="left",
                      va="top", fontsize=headline_size, color=m.ink,
                      fontweight="medium")
        for _ in range(12):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            width = art.get_window_extent(renderer).transformed(
                ax.transAxes.inverted()).width
            if width <= 0.86 or art.get_fontsize() <= 12.0:
                break
            art.set_fontsize(art.get_fontsize() - 0.6)
    return m, fig, ax


def _card_plot_axes(fig, bottom: float = 0.10, height: float = 0.44):
    """A bare plotting panel inside a card: no spines, no ticks, no labels.

    Cards that show a shape rather than a number still must not show an axis. At
    preview size a tick label is three pixels tall and reads as dirt.
    """
    ax = fig.add_axes((0.055, bottom, 0.89, height))
    ax.set_axis_off()
    return ax


def _finish_card(fig, ax, path, headline, alt, caption, mode):
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or headline, caption, headline, mode), (fig, ax)
    return fig, ax


def comparison_card(
    *,
    headline: str,
    items,
    note: str = "",
    emphasis: int | None = None,
    footer: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    path: str | None = None,
):
    """Two or three numbers side by side, for a finding that *is* a comparison.

    Use when the post's result is "this was promised and that was delivered", or
    "the brochure says X and you get Y". The whole card is type: at preview size a
    pair of large numbers with a hairline between them survives scaling better than
    any chart of the same pair, and it says the thing in one glance.

    `items` is `[(value, label), ...]`; `emphasis` indexes the one to colour as the
    bad news. Two or three items only — four large numbers is a table, and a table
    at this size is unreadable.
    """
    import textwrap

    items = list(items)
    if not 2 <= len(items) <= 3:
        raise ValueError("comparison_card takes two or three items")
    m, fig, ax = _card_base(mode, headline, footer)

    size = 60 if len(items) == 2 else 46
    edges = np.linspace(0.055, 0.945, len(items) + 1)
    column = float(edges[1] - edges[0])
    values = []
    labels = []
    for i, (value, label) in enumerate(items):
        cx = 0.5 * (edges[i] + edges[i + 1])
        colour = m.series[7] if emphasis == i else m.ink
        values.append(ax.text(cx, 0.50, str(value), ha="center", va="center",
                              fontsize=size, color=colour, fontweight="bold"))
        labels.append(ax.text(cx, 0.30, textwrap.fill(str(label), 26),
                              ha="center", va="top", fontsize=12.0,
                              color=m.ink_secondary, linespacing=1.35))
        if i:
            ax.plot([edges[i], edges[i]], [0.26, 0.62], color=m.grid, lw=1.2)

    # Shrink the values together until the widest fits its column. "10^30 yrs" at
    # 60pt is twice as wide as "42%" and ran straight into its neighbour; scaling
    # them *uniformly* matters, because two numbers set at different sizes read as
    # one being more important than the other.
    for _ in range(24):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transAxes.inverted()
        widest = max(v.get_window_extent(renderer).transformed(inv).width
                     for v in values)
        if widest <= column * 0.88 or values[0].get_fontsize() <= 22.0:
            break
        for v in values:
            v.set_fontsize(v.get_fontsize() - 1.5)
    if note:
        # Placed under the *measured* bottom of the labels, not at a fixed height.
        # A label that wraps to two lines grows downwards into a note pinned to the
        # safe margin, and the collision only appears for some label lengths.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transAxes.inverted()
        floor = min(label.get_window_extent(renderer).transformed(inv).y0
                    for label in labels) if labels else 0.30
        ax.text(0.055, max(floor - 0.03, CARD_SAFE), textwrap.fill(note, 96),
                ha="left", va="top", fontsize=11.0, color=m.muted,
                linespacing=1.35)
    return _finish_card(fig, ax, path, headline, alt, caption, mode)


def distribution_card(
    values,
    *,
    headline: str,
    mark: float | None = None,
    mark_label: str = "",
    note: str = "",
    bins: int = 46,
    footer: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    path: str | None = None,
):
    """A distribution drawn large, with one value marked, for a tail finding.

    Use when the result is "look where this one landed": the shape carries the
    argument, so the shape gets the space instead of a number. Deliberately has no
    big statistic — the marked label is the number, and putting both on the card
    makes it a busier version of `social_card` rather than a different card.
    """
    import textwrap

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values to plot")
    m, fig, ax = _card_base(mode, headline, footer)
    if note:
        ax.text(0.055, 0.72, textwrap.fill(note, 62), ha="left", va="top",
                fontsize=12.0, color=m.ink_secondary, linespacing=1.35)

    panel = _card_plot_axes(fig, bottom=CARD_SAFE + 0.02, height=0.44)
    counts, edges = np.histogram(v, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    panel.bar(centres, counts, width=(edges[1] - edges[0]) * 0.9,
              color=m.series[0], alpha=0.85, lw=0)
    if mark is not None:
        panel.axvline(mark, color=m.series[7], lw=2.4)
        if mark_label:
            panel.annotate(mark_label, (mark, 1.0),
                           xycoords=("data", "axes fraction"),
                           xytext=(8, -2), textcoords="offset points",
                           ha="left", va="top", fontsize=12.5,
                           color=m.series[7], fontweight="medium")
        lo, hi = float(min(centres.min(), mark)), float(max(centres.max(), mark))
        pad = 0.06 * (hi - lo)
        panel.set_xlim(lo - pad, hi + pad * 3)
    panel.set_ylim(0, counts.max() * 1.18)
    return _finish_card(fig, ax, path, headline, alt, caption, mode)


def series_card(
    y,
    *,
    headline: str,
    mark_index: int | None = None,
    mark_label: str = "",
    note: str = "",
    footer: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    path: str | None = None,
):
    """One series drawn large with a single moment annotated.

    Use when the finding is an event in time — a spike, a cliff, a break. The
    annotated point does the work a big number would do in `social_card`, and it
    does it in place, which is the reason to prefer this layout when the *timing*
    is the point.
    """
    import textwrap

    arr = np.asarray(y, float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("series_card needs a 1-D series of at least two points")
    m, fig, ax = _card_base(mode, headline, footer)
    if note:
        ax.text(0.055, 0.72, textwrap.fill(note, 62), ha="left", va="top",
                fontsize=12.0, color=m.ink_secondary, linespacing=1.35)

    panel = _card_plot_axes(fig, bottom=CARD_SAFE + 0.02, height=0.46)
    x = np.arange(arr.size)
    panel.plot(x, arr, color=m.series[0], lw=1.8)
    panel.fill_between(x, arr.min(), arr, color=m.series[0], alpha=0.14, lw=0)
    if mark_index is not None:
        i = int(np.clip(mark_index, 0, arr.size - 1))
        panel.plot([i], [arr[i]], marker="o", ms=11, color=m.series[7],
                   markeredgecolor=m.surface, markeredgewidth=2.0, zorder=5)
        if mark_label:
            # Label away from the edge it is nearest, so a spike at either end of
            # the series does not push its own annotation off the card.
            left = i < arr.size * 0.5
            panel.annotate(mark_label, (i, arr[i]),
                           xytext=(14 if left else -14, 0),
                           textcoords="offset points",
                           ha="left" if left else "right", va="center",
                           fontsize=12.5, color=m.series[7],
                           fontweight="medium")
    span = float(np.ptp(arr)) or 1.0
    panel.set_ylim(arr.min() - 0.08 * span, arr.max() + 0.14 * span)
    return _finish_card(fig, ax, path, headline, alt, caption, mode)


def bar_card(
    *,
    headline: str,
    items,
    note: str = "",
    emphasis: int | None = None,
    footer: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    path: str | None = None,
):
    """A few labelled horizontal bars, for a finding that is a ranking or a ramp.

    Use when the result is "these four cases differ, and by this much". Bars read
    at any size, which is more than can be said for a line chart's axis, and the
    value labels mean the card does not need a legend or a scale.

    `items` is `[(label, value, value_text), ...]`, in the order to draw them top
    to bottom; `emphasis` indexes the bar to colour as the headline case.
    """
    import textwrap

    items = list(items)
    if not 2 <= len(items) <= 5:
        raise ValueError("bar_card takes two to five bars")
    m, fig, ax = _card_base(mode, headline, footer)
    if note:
        ax.text(0.055, CARD_SAFE + 0.02, textwrap.fill(note, 96), ha="left",
                va="bottom", fontsize=11.0, color=m.muted, linespacing=1.35)

    top, bottom = 0.68, CARD_SAFE + (0.10 if note else 0.03)
    n = len(items)
    step = (top - bottom) / n
    biggest = max(abs(float(v)) for _l, v, _t in items) or 1.0
    x0, x1 = 0.30, 0.86            # bar track, leaving room for labels and values
    for i, (label, value, value_text) in enumerate(items):
        y = top - step * (i + 0.5)
        w = (x1 - x0) * abs(float(value)) / biggest
        colour = m.series[7] if emphasis == i else m.series[0]
        ax.add_patch(plt.Rectangle((x0, y - step * 0.28), w, step * 0.56,
                                   color=colour, alpha=0.9, lw=0))
        ax.text(x0 - 0.02, y, str(label), ha="right", va="center", fontsize=12.5,
                color=m.ink_secondary)
        ax.text(x0 + w + 0.015, y, str(value_text), ha="left", va="center",
                fontsize=15.0, color=m.ink, fontweight="medium")
    return _finish_card(fig, ax, path, headline, alt, caption, mode)


def sketch_card(
    *,
    headline: str,
    items,
    sketch=None,
    note: str = "",
    footer: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    path: str | None = None,
):
    """A hand-drawn card: two numbers beside a small diagram of the mechanism.

    The other cards in this family are pure type, which is the right default: at
    preview size type survives scaling and a chart does not. This one exists for the
    case where the finding is a *shape* that type cannot carry — a forecast that
    tracks something until it doesn't, a curve that forks — and where a schematic of
    the shape earns the space a third number would have taken.

    The wobble comes from matplotlib's xkcd path filter plus the bundled Patrick
    Hand face. Both are cosmetic and both degrade: without the font the card renders
    in the system sans and reads as a slightly loose ordinary card. What the style is
    *not* is a licence to draw a fake chart — `sketch` gets no axes, no ticks and no
    numbers, precisely so nobody reads a value off it. It is a diagram, and the
    numbers beside it are the measurement.

    `sketch(panel, m)` receives a bare axes and the mode, and should draw the
    diagram and set its own limits. Two items only: three large numbers plus a
    drawing is a busier version of `comparison_card` rather than a different card.
    """
    import textwrap

    items = list(items)
    if len(items) != 2:
        raise ValueError("sketch_card takes exactly two items; use "
                         "comparison_card for three")
    m = theme.LIGHT if mode == "light" else theme.DARK

    with plt.xkcd(scale=0.9, length=110, randomness=2):
        plt.rcParams["font.family"] = theme.sketch_family()
        fig = plt.figure(figsize=CARD_SIZE, dpi=CARD_DPI)
        fig.patch.set_facecolor(m.surface)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Headline shrinks to fit rather than wrapping into the drawing.
        size = 20.0
        txt = ax.text(0.055, 0.94, headline, ha="left", va="top", fontsize=size,
                      color=m.ink)
        fig.canvas.draw()
        while size > 12.0:
            w = txt.get_window_extent(fig.canvas.get_renderer()).width
            if w <= 0.86 * fig.get_size_inches()[0] * CARD_DPI:
                break
            size -= 0.6
            txt.set_fontsize(size)
            fig.canvas.draw()
        if footer:
            ax.text(0.945, 0.965, footer, ha="right", va="top", fontsize=12,
                    color=m.muted)

        if sketch is not None:
            panel = fig.add_axes([0.05, 0.30, 0.47, 0.46])
            panel.patch.set_visible(False)
            for spine in panel.spines.values():
                spine.set_visible(False)
            panel.set_xticks([])
            panel.set_yticks([])
            sketch(panel, m)

        xs = (0.655, 0.895) if sketch is not None else (0.28, 0.72)
        for (value, label), x in zip(items, xs):
            ax.text(x, 0.56, str(value), ha="center", va="center", fontsize=40,
                    color=m.ink)
            ax.text(x, 0.40, textwrap.fill(str(label), 18), ha="center", va="top",
                    fontsize=11.5, color=m.ink_secondary, linespacing=1.3)
        mid = 0.5 * (xs[0] + xs[1])
        ax.plot([mid, mid], [0.42, 0.64], color=m.grid, lw=1.6)

        if note:
            # Wrapped at 80 rather than the 62 the sans cards use: the drawn face
            # is wider per character, and at 74 this note left an orphan word on a
            # third line.
            ax.text(0.055, 0.16, textwrap.fill(note, 80), ha="left", va="top",
                    fontsize=12.5, color=m.ink_secondary, linespacing=1.35)

        if path:
            theme.save(fig, path, mode=mode, close=False)
            out = Figure(path, alt or headline, caption, headline, mode)
            plt.close(fig)
            return out, (fig, ax)
        return fig, ax


def strip_card(
    *,
    headline: str,
    panels,
    note: str = "",
    footer: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    path: str | None = None,
):
    """A hand-drawn strip: two or three panels, each with a drawing and a number.

    `sketch_card` draws one diagram of a mechanism. This draws a *sequence* — the
    comic-strip form, where the point is that panel two follows panel one and
    undercuts it. Use it when the finding has a setup and a punchline, which is more
    often than it sounds: "the rumour moved it, the announcement did nothing" and
    "the forecast held, then the regime changed" are both two-panel jokes with a
    number under each frame.

    `panels` is `[(draw, value, label), ...]`, where `draw(panel, m)` receives a bare
    axes with no ticks or spines and the mode. Each panel gets a light frame, because
    a strip without frames reads as one confused picture rather than a sequence.

    Same constraints as the other drawn card: no axes, no values inside the drawings,
    nothing a reader could mistake for a measurement. The numbers under the frames
    are the measurement; the frames are the story.
    """
    import textwrap

    panels = list(panels)
    if not 2 <= len(panels) <= 3:
        raise ValueError("a strip takes two or three panels")
    m = theme.LIGHT if mode == "light" else theme.DARK

    with plt.xkcd(scale=0.9, length=110, randomness=2):
        plt.rcParams["font.family"] = theme.sketch_family()
        fig = plt.figure(figsize=CARD_SIZE, dpi=CARD_DPI)
        fig.patch.set_facecolor(m.surface)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        size = 19.0
        txt = ax.text(0.055, 0.95, headline, ha="left", va="top", fontsize=size,
                      color=m.ink)
        fig.canvas.draw()
        while size > 12.0:
            w = txt.get_window_extent(fig.canvas.get_renderer()).width
            if w <= 0.86 * fig.get_size_inches()[0] * CARD_DPI:
                break
            size -= 0.6
            txt.set_fontsize(size)
            fig.canvas.draw()
        if footer:
            ax.text(0.945, 0.975, footer, ha="right", va="top", fontsize=12,
                    color=m.muted)

        n = len(panels)
        left, right, gap = 0.055, 0.945, 0.035
        width = ((right - left) - gap * (n - 1)) / n
        bottom, height = 0.40, 0.36
        for i, (draw, value, label) in enumerate(panels):
            x0 = left + i * (width + gap)
            panel = fig.add_axes([x0, bottom, width, height])
            panel.set_xticks([])
            panel.set_yticks([])
            panel.patch.set_facecolor(m.page)
            for spine in panel.spines.values():
                spine.set_visible(True)
                spine.set_color(m.grid)
                spine.set_linewidth(1.4)
            draw(panel, m)
            ax.text(x0 + width / 2, bottom - 0.06, str(value), ha="center",
                    va="top", fontsize=34, color=m.ink,
                    transform=ax.transAxes)
            ax.text(x0 + width / 2, bottom - 0.20, textwrap.fill(str(label), 24),
                    ha="center", va="top", fontsize=11.5, color=m.ink_secondary,
                    linespacing=1.3, transform=ax.transAxes)

        if note:
            ax.text(0.055, 0.115, textwrap.fill(note, 92), ha="left", va="top",
                    fontsize=12, color=m.ink_secondary, linespacing=1.35)

        if path:
            theme.save(fig, path, mode=mode, close=False)
            out = Figure(path, alt or headline, caption, headline, mode)
            plt.close(fig)
            return out, (fig, ax)
        return fig, ax


def social_card(
    *,
    headline: str,
    stat: str,
    stat_label: str = "",
    supporting: tuple[tuple[str, str], ...] = (),
    silhouette: tuple = (),
    mark: float | None = None,
    footer: str = "",
    alt: str = "",
    caption: str = "",
    mode: str = "light",
    path: str | None = None,
):
    """A 1600x840 preview image for the places that crop and shrink a figure.

    Medium's story preview, an OpenGraph card and a Slack unfurl all show the
    post's first image at a fraction of its size, and an analysis chart loses at
    that size: the axis labels, the legend and the annotation that carry its
    meaning become illegible, so a reader sees a vaguely blue rectangle. This
    renders the *finding* instead — one headline, one large number, at most three
    supporting figures — over a stripped silhouette of the real data, so the card
    is still made of the result rather than being decoration.

    `silhouette` is `(x, y)` for a filled shape across the bottom, `mark` an x
    position (in `silhouette`'s own units) for a vertical rule on it; a mark
    outside the data range is clamped to the edge rather than silently dropped.
    Both optional: the type alone works.

    Keep `stat_label` under ~60 characters and each supporting label under ~32;
    longer strings wrap, and a card that needs three wrapped lines is a card
    doing the body text's job.

    2:1 aspect and 1600px wide clears the ~1500px Medium and OpenGraph both want.
    """
    import textwrap

    m, fig, ax = _card_base(mode, headline, footer)

    # The data, reduced to a shape in the bottom quarter at low contrast: it is a
    # texture behind the type, and anything a reader could measure back out of it
    # would be a lie at this size.
    band = 0.24
    if len(silhouette) == 2:
        sx, sy = (np.asarray(v, float) for v in silhouette)
        if sx.size and sy.size and np.ptp(sx) > 0:
            gx = (sx - sx.min()) / np.ptp(sx)
            gy = sy / sy.max() if sy.max() > 0 else sy
            ax.fill_between(gx, 0.0, 0.01 + band * gy, color=m.series[0],
                            alpha=0.16, lw=0)
            ax.plot(gx, 0.01 + band * gy, color=m.series[0], lw=1.6, alpha=0.5)
            if mark is not None:
                mx = float(np.clip((mark - sx.min()) / np.ptp(sx), 0.0, 1.0))
                # Height follows the silhouette at that x, plus a little headroom,
                # and never rises into the type band. A fixed-height rule collides
                # with the supporting column whenever the mark lands under it —
                # which is exactly where an interesting mark tends to land, since
                # the tails are where the outliers are.
                local = float(np.interp(mx, gx, 0.01 + band * gy))
                ax.plot([mx, mx], [0.0, min(local + 0.07, band + 0.02)],
                        color=m.series[7], lw=2.2)

    ax.text(0.055, 0.63, stat, ha="left", va="center", fontsize=64,
            color=m.ink, fontweight="bold")
    if stat_label:
        ax.text(0.058, 0.44, textwrap.fill(stat_label, 46), ha="left", va="top",
                fontsize=12.5, color=m.ink_secondary, linespacing=1.4)

    # Supporting figures right-aligned in a column, so the eye lands on the big
    # number first and the qualifiers second.
    for i, (label, value) in enumerate(supporting[:3]):
        y = 0.74 - 0.20 * i
        ax.text(0.965, y, value, ha="right", va="center", fontsize=21,
                color=m.ink, fontweight="medium")
        ax.text(0.965, y - 0.06, textwrap.fill(label, 32), ha="right", va="top",
                fontsize=10.5, color=m.ink_secondary, linespacing=1.35)
    if path:
        theme.save(fig, path, mode=mode, close=False)
        return Figure(path, alt or headline, caption, headline, mode), (fig, ax)
    return fig, ax
