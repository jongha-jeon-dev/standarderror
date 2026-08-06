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
    figsize: tuple[float, float] | None = None,
    path: str | None = None,
):
    """Horizontal ranked bars — the form for magnitude comparison across a
    handful of named things (attribution, importance, model scores).

    Signed values use the diverging pair so sign is visible without reading the
    axis; unsigned use one hue, because varying hue across bars of the same
    measure encodes nothing.
    """
    m = theme.apply(mode)
    labels = list(labels)
    values = np.asarray(values, float)
    order = np.argsort(np.abs(values) if signed else values)
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
        ax.annotate(f"{v:,.3g}", (v + off, i), va="center",
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
    ax.bar(centres, counts, width=(edges[1] - edges[0]) * 0.92,
           color=m.series[0], label="observed")
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
    col_width: tuple[float, ...] | None = None,
    row_height: float = 0.42,
    path: str | None = None,
):
    """Render a small table as an image, in the same style as the charts.

    This exists because **Medium has no table support and strips table markup on
    paste** — a markdown table arrives as a run of plain text with the pipes gone.
    The alternatives are a GitHub Gist embed (selectable text, but a second place
    to keep the numbers in sync) or a picture. A picture rendered from the same
    data that produced the post keeps one source of truth, and matches the
    figures instead of looking like a screenshot of something else.

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

    # Column widths from the longest cell, so numbers do not collide with labels.
    widths = col_width or tuple(
        max(len(head[j]), *(len(r[j]) for r in body)) for j in range(n_col))
    total = sum(widths)
    edges = [0.0]
    for w in widths:
        edges.append(edges[-1] + w / total)

    n_row = len(body)
    fig_h = row_height * (n_row + 1) + 1.1
    fig, ax = plt.subplots(figsize=(min(7.2, 1.6 + 0.085 * total), fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_row + 1)
    ax.grid(False)

    def cell_x(j: int) -> tuple[float, str]:
        pad = 0.012
        if align[j] == "l":
            return edges[j] + pad, "left"
        return edges[j + 1] - pad, "right"

    y_head = n_row + 0.45
    for j, text in enumerate(head):
        x, ha = cell_x(j)
        ax.text(x, y_head, text, ha=ha, va="center", fontsize=9.0,
                color=m.ink_secondary)
    # Rule under the header, hairlines between rows: the minimum a table needs.
    ax.plot([0, 1], [n_row, n_row], color=m.axis, lw=1.0, clip_on=False)
    for i, row in enumerate(body):
        y = n_row - 0.5 - i
        if i:
            ax.plot([0, 1], [y + 0.5, y + 0.5], color=m.grid, lw=0.6,
                    clip_on=False)
        for j, text in enumerate(row):
            x, ha = cell_x(j)
            strong = j in bold_cols or (i, j) in (bold_cells or set())
            ax.text(x, y, text, ha=ha, va="center", fontsize=9.5,
                    color=m.ink, fontweight="bold" if strong else "normal")

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
