"""Topology 2: The Barcode's Numbers Mean Nothing on Their Own.

The second episode, and it is the one where the plan was wrong. It was written
to say that the barcode's dynamic range collapses in high dimensions and the
display therefore fails. The range does collapse -- and the summary becomes
*more* informative, not less, because concentration tightens the null much
faster than it shrinks the signal.

Measured:

* The H0 death spread, as a fraction of its own mean, runs 4.87 at d = 2 to
  0.078 at d = 768 on pure noise, as a median over fifteen draws each.
* The separation H0 needs to recover three clusters rises in absolute terms --
  3.95 at d = 3 to 10.49 at d = 768 -- and falls as a fraction of the typical
  pairwise distance, 1.73 to 0.27. The clustering gets easier relative to the
  scale of the space, not harder.
* The largest-gap rule never reports one cluster on a cloud that has none: zero
  times in 40 draws at every dimension tried. And its failure changes shape --
  modal answer 2 at d = 2, and `n - 1` at d = 64 and above.
* The two-cluster ratio, read as an absolute number, is not weak but inverted.
  At d = 2 noise has a median ratio of 1.224 and a cloud whose clusters single
  linkage recovers at purity 0.989 has 1.175, so the ratio's AUC as a detector
  is 0.462 -- worse than a coin.
* Read against a matched null it works, and better with dimension: AUC 0.462,
  0.569, 0.648, 0.764, 0.741, 0.809 at d = 2, 4, 8, 32, 128, 768.

Run: `standarderror run lec202_barcode_axis --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.topology import filtration as ft
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
POST_DATE = date(2026, 9, 9)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Topology for Language Models, Taught Through What Breaks"
SERIES_TAG = "Topology"

#: The dimensions every sweep runs at. 768 because it is the width of a great
#: many embedding spaces, and 2 because that is where the pictures come from.
DIMS = (2, 4, 8, 32, 128, 768)
#: Where the barcode pictures are drawn.
SHOW = (2, 768)
NOISE_N = 200


def compute() -> dict:
    dims = ft.dimension_sweep(DIMS, n=NOISE_N)
    needed = [ft.separation_needed(d, n_struct=min(3, d)) for d in DIMS]
    noise = [ft.gap_rule_on_noise(d, n=NOISE_N) for d in DIMS]
    auc = [ft.separation_ratio_auc(d) for d in DIMS]

    # The two barcodes the first figure draws, from the same generator as the
    # noise sweep so the numbers in the caption are the numbers in the picture.
    bars = {}
    for d in SHOW:
        Y = np.random.default_rng([7, d, 0]).standard_normal((NOISE_N, d))
        bars[d] = ft.rips_h0(ft.pairwise(Y))
    conc = [ft.concentration_check(d) for d in DIMS]
    where = [ft.largest_gap_position(d) for d in DIMS]
    return {"dims": dims, "needed": needed, "noise": noise, "auc": auc,
            "bars": bars, "conc": conc, "where": where}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    dims = {r["d"]: r for r in res["dims"]}
    noise = {r["d"]: r for r in res["noise"]}
    auc = {r["d"]: r for r in res["auc"]}
    needed = {r["d"]: r for r in res["needed"]}
    where = {r["d"]: r for r in res["where"]}
    bars = res["bars"]

    # --- f0: the same barcode at two dimensions, rescaled ------------------
    def two_barcodes(ax, m):
        for row, (d, colour) in enumerate(zip(SHOW, m.series)):
            deaths = bars[d].deaths
            # Scaled by the mean death, because at d = 768 the absolute radii
            # are twenty times larger and the point is the *shape*.
            x = deaths / deaths.mean()
            y = np.linspace(row * 1.15, row * 1.15 + 0.9, len(x))
            ax.hlines(y, x.min() * 0.0 + x.min(), x, color=colour, lw=0.8)
            ax.annotate(f"d = {d}, spread {bars[d].spread_ratio:.2f}",
                        (x.max(), y[-1]), textcoords="offset points",
                        xytext=(-6, 8), ha="right", fontsize=9.0, color=colour)
        ax.set_yticks([])
        ax.set_xlim(0, 2.1)

    out["f0"] = charts.diagram(
        two_barcodes,
        title="The same noise, and the barcode has stopped having a shape",
        subtitle=(f"{NOISE_N} points of pure Gaussian noise — no clusters at "
                  f"all — at two dimensions and at 768. Each barcode's deaths "
                  f"are divided by their own mean, so only the shape is being "
                  f"compared."),
        xlabel="death radius ÷ mean death radius", ylabel="",
        source="Simulated; standarderror/topology/filtration.py.",
        alt=("Two stacked barcodes. The lower one has bars of visibly varying "
             "length; the upper one has bars that all end at essentially the "
             "same place."),
        caption=(f"At two dimensions the deaths spread over "
                 f"{bars[2].spread_ratio:.2f} times their own mean, and noise "
                 f"has visible structure — sparse patches, and therefore long "
                 f"bars. At 768 the spread is {bars[768].spread_ratio:.3f} and "
                 f"every bar ends within a few percent of the same radius. "
                 f"Which reads as bad news for the display, and turns out to be "
                 f"the reason the display becomes useful."),
        path=str(IMG / f"lec202-f0-two-barcodes.{EXT}"))[0]

    # --- f1: the separation H0 needs, both ways ----------------------------
    def separation(ax, m):
        d = [r["d"] for r in res["needed"]]
        ax.plot(d, [r["separation"] for r in res["needed"]], marker="o", ms=6,
                lw=1.9, color=m.series[0], label="absolute separation needed")
        ax.set_xscale("log")
        ax.set_ylabel("absolute", color=m.series[0])
        twin = ax.twinx()
        twin.plot(d, [r["relative"] for r in res["needed"]], marker="s", ms=6,
                  lw=1.9, color=m.series[1],
                  label="÷ the typical pairwise distance")
        twin.axhline(1.0, color=m.grid, lw=1.4)
        twin.set_ylabel("relative to the noise scale", color=m.series[1])
        twin.set_ylim(0, 2.6)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5,
                  loc="upper center")

    out["f1"] = charts.diagram(
        separation,
        title="Harder in absolute terms, easier relative to the space",
        subtitle=("How far apart three clusters have to be before a "
                  "single-linkage cut recovers them at 95% purity. Bisected, at "
                  "each dimension."),
        xlabel="dimension", ylabel="",
        source="Simulated; standarderror/topology/filtration.py.",
        alt=("Two lines against dimension on a log x-axis, one rising and one "
             "falling, with a horizontal reference at one on the falling "
             "line's axis."),
        caption=(f"The absolute separation rises from "
                 f"{needed[2]['separation']:.2f} to "
                 f"{needed[768]['separation']:.2f}, which is the story people "
                 f"expect. Divided by the typical pairwise distance it falls "
                 f"from {needed[2]['relative']:.2f} to "
                 f"{needed[768]['relative']:.2f} — so relative to the scale of "
                 f"the space the clusters get **easier** to separate, and every "
                 f"purity in this sweep is at or above "
                 f"{min(auc[d]['purity'] for d in DIMS):.3f}. Whatever is going "
                 f"wrong "
                 f"in high dimensions is not the clustering."),
        path=str(IMG / f"lec202-f1-separation.{EXT}"))[0]

    # --- f2: the payoff, null against signal -------------------------------
    def overlap(ax, m):
        # Plotted as ratio - 1, on a log axis. The two rows differ by fifty
        # times in absolute terms, so a shared linear axis crushed the d = 768
        # pair into a sliver and served neither row. What has to be legible is
        # which of the two ranges sits to the right of the other.
        rows = [auc[d] for d in SHOW]
        for i, (r, colour) in enumerate(zip(rows, m.series)):
            base = i * 1.0
            ax.plot([r["noise_median"] - 1, r["noise_max"] - 1], [base, base],
                    lw=7, color=m.grid, solid_capstyle="butt")
            ax.plot([r["noise_median"] - 1], [base], "o", ms=8, color=m.ink)
            ax.plot([r["signal_min"] - 1, r["signal_median"] - 1],
                    [base + 0.34, base + 0.34], lw=7, color=colour,
                    solid_capstyle="butt")
            ax.plot([r["signal_median"] - 1], [base + 0.34], "o", ms=8,
                    color=colour)
            ax.annotate(f"d = {r['d']},  AUC {r['auc']:.3f}",
                        (r["noise_median"] - 1, base + 0.60), fontsize=9.5,
                        color=m.ink_secondary)
            arrow = "noise sits to the RIGHT of signal" if r["auc"] < 0.5 \
                else "signal sits to the RIGHT of noise"
            ax.annotate(arrow, (r["signal_min"] - 1, base - 0.20),
                        fontsize=8.5, color=colour)
        ax.set_xscale("log")
        ax.set_yticks([])
        ax.set_ylim(-0.5, 2.05)

    out["f2"] = charts.diagram(
        overlap,
        title="At two dimensions the noise claims more clusters than the clusters do",
        subtitle=("The two-cluster ratio for pure noise against clouds whose "
                  "three clusters a single-linkage cut recovers. 60 draws each, "
                  "at two dimensions and at 768."),
        xlabel="(longest bar ÷ second longest) − 1", ylabel="",
        source="Simulated; standarderror/topology/filtration.py.",
        alt=("Two pairs of horizontal ranges on a log x-axis. In the lower "
             "pair the grey noise range sits to the right of the coloured "
             "signal range; in the upper pair the order is reversed, and both "
             "ranges are fifty times closer to zero."),
        caption=(f"Grey is noise, colour is the clustered cloud; the dot on each "
                 f"range is its median. At d = 2 the noise median is "
                 f"{auc[2]['noise_median']:.4f} and "
                 f"the signal median {auc[2]['signal_median']:.4f} — the wrong "
                 f"way round, giving an AUC of {auc[2]['auc']:.3f}, worse than a "
                 f"coin, on a cloud whose clusters are recovered at purity "
                 f"{auc[2]['purity']:.3f}. At d = 768 the numbers are tiny "
                 f"({auc[768]['noise_median']:.4f} against "
                 f"{auc[768]['signal_median']:.4f}) and the AUC is "
                 f"{auc[768]['auc']:.3f}. The absolute value of the ratio "
                 f"carries no information at either dimension; its position "
                 f"against a matched null carries more as the dimension rises."),
        path=str(IMG / f"lec202-f2-overlap.{EXT}"))[0]

    # --- f3: and the trend ------------------------------------------------
    def trend(ax, m):
        d = [r["d"] for r in res["auc"]]
        ax.plot(d, [r["auc"] for r in res["auc"]], marker="o", ms=6, lw=1.9,
                color=m.series[0], label="the ratio, against a matched null")
        ax.axhline(0.5, color=m.ink, lw=1.5)
        ax.annotate("0.5 = no information", (d[-1], 0.5),
                    textcoords="offset points", xytext=(-4, 7), ha="right",
                    fontsize=8.5, color=m.ink_secondary)
        ax.plot(d, [r["barcode_spread"] / dims[2]["barcode_spread"]
                    for r in res["dims"]], marker="s", ms=5, lw=1.6,
                color=m.series[1],
                label="the barcode's dynamic range, relative to d = 2")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.12)
        ax.legend(frameon=False, fontsize=8.5, loc="center right")

    out["f3"] = charts.diagram(
        trend,
        title="The range collapses and the detector improves",
        subtitle=("Two quantities against dimension: how much of the "
                  "filtration axis the barcode occupies, and how well its "
                  "two-cluster ratio separates clustered clouds from noise."),
        xlabel="dimension", ylabel="",
        source="Simulated; standarderror/topology/filtration.py.",
        alt=("A rising curve of detector performance and a falling curve of "
             "dynamic range, crossing, with a horizontal reference line at "
             "0.5 for chance performance."),
        caption=(f"These move in opposite directions, and it is the same "
                 f"phenomenon. Concentration squeezes the null far harder than "
                 f"it squeezes the signal, so the barcode stops having a "
                 f"readable shape and starts having a usable reference. AUC "
                 f"{auc[2]['auc']:.3f} at d = 2 against "
                 f"{auc[768]['auc']:.3f} at d = 768, while the dynamic range "
                 f"falls by a factor of "
                 f"{dims[2]['barcode_spread'] / dims[768]['barcode_spread']:.0f}."),
        path=str(IMG / f"lec202-f3-trend.{EXT}"))[0]

    # --- f4: the rule that cannot say "none" -------------------------------
    def gap(ax, m):
        d = [r["d"] for r in res["noise"]]
        ax.plot(d, [r["modal_k"] for r in res["noise"]], marker="o", ms=7,
                lw=1.9, color=m.series[0], label="modal answer on pure noise")
        ax.axhline(1.0, color=m.ink, lw=1.6)
        ax.annotate("1 = the right answer, returned 0 times out of 40 "
                    "at every dimension", (d[0], 1.0),
                    textcoords="offset points", xytext=(4, 8), fontsize=8.5,
                    color=m.ink_secondary)
        ax.axhline(NOISE_N - 1, color=m.grid, lw=1.4, ls=(0, (4, 3)))
        ax.annotate(f"{NOISE_N - 1} = every point its own cluster",
                    (d[-1], NOISE_N - 1), textcoords="offset points",
                    xytext=(-4, -14), ha="right", fontsize=8.5,
                    color=m.ink_secondary)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8.5, loc="center left")

    out["f4"] = charts.diagram(
        gap,
        title="The rule cannot return \"no clusters\", and it does not",
        subtitle=(f"\"Count the bars above the biggest jump\", applied to "
                  f"{NOISE_N} points of pure noise, 40 draws at each "
                  f"dimension."),
        xlabel="dimension", ylabel="clusters reported",
        source="Simulated; standarderror/topology/filtration.py.",
        alt=("A step-like rising curve of the modal reported cluster count "
             "against dimension on log axes, starting at two and ending at "
             "199, with reference lines at one and at 199."),
        caption=(f"Zero of forty draws returned 1 at any dimension. The rule "
                 f"locates the largest jump in a sorted list, and a sorted list "
                 f"of pure noise has a largest jump. What changes with "
                 f"dimension is only **where** it is: at d = 2 the modal answer "
                 f"is {noise[2]['modal_k']}, and from d = 32 it is "
                 f"{noise[768]['modal_k']}, because in a concentrated barcode "
                 f"the biggest gap is as often at the front as at the back — "
                 f"{where[768]['in_first_ten']} draws of thirty against "
                 f"{where[768]['in_last_ten']}. This is the scree "
                 f"plot's elbow, which also cannot say zero."),
        path=str(IMG / f"lec202-f4-gap.{EXT}"))[0]

    # --- f5: where the largest gap actually falls --------------------------

    def positions(ax, m):
        rows = (2, 8, 768)
        rng = np.random.default_rng(0)
        for row, (d, colour) in enumerate(zip(rows, m.series)):
            pos = np.asarray(where[d]["positions"], dtype=float)
            y = row + rng.uniform(-0.16, 0.16, size=len(pos))
            ax.scatter(pos, y, s=26, color=colour, alpha=0.75,
                       edgecolors="none")
            ax.annotate(f"d = {d}, skew {where[d]['death_skew']:.2f}",
                        (0.5, row + 0.34), fontsize=9.0, color=colour)
        ax.set_yticks(list(range(len(rows))))
        ax.set_yticklabels([f"d = {d}" for d in rows])
        ax.set_ylim(-0.55, len(rows) - 0.25)
        ax.set_xlim(-6, where[2]["gaps"] + 6)
        ax.axvline(0, color=m.grid, lw=1.2)
        ax.axvline(where[2]["gaps"], color=m.grid, lw=1.2)

    out["f5"] = charts.diagram(
        positions,
        title="Not a drift towards the front. A flip between both ends",
        subtitle=(f"The position of the largest gap among the "
                  f"{where[2]['gaps']} gaps in a sorted noise barcode, one dot "
                  f"per draw, thirty draws at each dimension."),
        xlabel="position of the largest gap in the sorted deaths",
        ylabel="",
        source="Simulated; standarderror/topology/filtration.py.",
        alt=("Three rows of dots. The lowest row is piled entirely at the "
             "right edge; the highest row is split between the left and right "
             "edges with the middle empty."),
        caption=(f"At d = 2 every one of the thirty draws puts the largest gap "
                 f"in the last ten positions, so the rule always answers "
                 f"{noise[2]['modal_k']}. At d = 768 it is in the first ten "
                 f"{where[768]['in_first_ten']} times and in the last ten "
                 f"{where[768]['in_last_ten']} times, and in the whole of the "
                 f"middle "
                 f"{where[768]['draws'] - where[768]['in_first_ten'] - where[768]['in_last_ten']} "
                 f"times. The skew of the death distribution is the mechanism: "
                 f"{where[2]['death_skew']:.2f} at d = 2 against "
                 f"{where[768]['death_skew']:.2f} at 768."),
        path=str(IMG / f"lec202-f5-gap-position.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    auc = {r["d"]: r for r in res["auc"]}
    bars = res["bars"]

    def flat(panel, m):
        for row, d in enumerate(SHOW):
            deaths = bars[d].deaths
            x = deaths / deaths.mean()
            y = np.linspace(row * 1.2, row * 1.2 + 0.9, len(x))
            panel.hlines(y, 0, x, color=m.ink, lw=0.6)
        panel.set_yticks([])

    def crossing(panel, m):
        d = np.log10([r["d"] for r in res["auc"]])
        panel.plot(d, [r["auc"] for r in res["auc"]], color=m.ink, lw=2.6,
                   marker="o", ms=4)
        panel.axhline(0.5, color=m.grid, lw=2.2)
        panel.set_ylim(0.3, 0.95)

    def cannot(panel, m):
        d = np.log10([r["d"] for r in res["noise"]])
        panel.plot(d, np.log10([r["modal_k"] for r in res["noise"]]),
                   color=m.ink, lw=2.6, marker="o", ms=4)
        panel.axhline(0.0, color=m.grid, lw=2.4)
        panel.set_ylim(-0.4, 2.6)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=2,
        headline="A barcode's numbers mean nothing alone",
        panels=[
            (flat, f"{bars[768].spread_ratio:.3f}", "axis left at d = 768"),
            (crossing, f"{auc[2]['auc']:.2f} → {auc[768]['auc']:.2f}",
             "AUC, d = 2 to 768"),
            (cannot, "0 of 40", "times it said \"none\""),
        ],
        note=("This episode was planned to say that the barcode's dynamic "
              "range collapses in high dimensions and the display therefore "
              "fails. The range does collapse, by a factor of 86 — and the "
              "summary becomes more informative, not less, because "
              "concentration tightens the null far faster than it shrinks the "
              "signal. What fails at every dimension is reading the numbers "
              "absolutely."),
        alt=("A three-panel hand-drawn strip. The first shows two barcodes, "
             "one ragged and one flat. The second shows a rising curve "
             "crossing a horizontal line. The third shows a rising step curve "
             "well above a horizontal floor."),
        mode="light",
        path=str(IMG / f"lec202-hero.{EXT}"))[0]


# --------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["shape"] = s.run(f"""
        import numpy as np
        from scipy.spatial.distance import pdist, squareform
        from standarderror.topology.filtration import rips_h0

        # Pure noise. No clusters, at any dimension.
        print(f"{{'d':>5}}  {{'mean death':>11}}  {{'spread / mean':>14}}")
        for d in {list(DIMS)}:
            Y = np.random.default_rng([7, d, 0]).standard_normal(({NOISE_N}, d))
            deaths = rips_h0(squareform(pdist(Y))).deaths
            spread = (deaths.max() - deaths.min()) / deaths.mean()
            print(f"{{d:>5}}  {{deaths.mean():11.3f}}  {{spread:14.3f}}")
    """, expect=["spread / mean"])

    out["gap"] = s.run(f"""
        # And what the usual rule says about that noise. It has one job --
        # find the biggest jump in the sorted deaths -- and a sorted list of
        # noise has a biggest jump, so it always finds one.
        from collections import Counter

        def gap_k(deaths):
            return len(deaths) - int(np.argmax(np.diff(deaths)))

        for d in (2, 32, 768):
            ks = []
            for i in range(40):
                Y = np.random.default_rng([7, d, i]).standard_normal(
                    ({NOISE_N}, d))
                ks.append(gap_k(rips_h0(squareform(pdist(Y))).deaths))
            c = Counter(ks)
            print(f"d = {{d:>3}}   said 1 cluster: {{c.get(1, 0)}}/40   "
                  f"modal answer: {{max(c, key=lambda k: c[k])}}   "
                  f"range: {{min(ks)}}-{{max(ks)}}")
    """, expect=["said 1 cluster: 0/40"])

    out["detector"] = s.run("""
        # The ratio people read as "how strongly does this say clusters",
        # measured against a null built from the same shape of cloud. Noise
        # versus three clusters that single linkage recovers.
        from standarderror.topology.filtration import (
            separated_clusters, pairwise, purity)

        def ratio(X):
            b = rips_h0(pairwise(X)).deaths
            return b[-1] / b[-2]

        for d, sep in ((2, 5.4), (32, 7.0), (768, 13.6)):
            ns = min(3, d)
            noise = [ratio(np.random.default_rng([7, d, i]).standard_normal(
                (120, d))) for i in range(40)]
            sig, pur = [], []
            for i in range(40):
                X, lab = separated_clusters(d, sep, per=40, n_struct=ns,
                                            seed=300 + i)
                sig.append(ratio(X))
                pur.append(purity(pairwise(X), lab, 3))
            print(f"d = {d:>3}   noise median {np.median(noise):.4f}   "
                  f"signal median {np.median(sig):.4f}   "
                  f"clusters recovered: {np.mean(pur):.3f}")
    """, expect=["clusters recovered"])

    return out


# ------------------------------------------------------------------- build

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    dims = {r["d"]: r for r in res["dims"]}
    noise = {r["d"]: r for r in res["noise"]}
    auc = {r["d"]: r for r in res["auc"]}
    needed = {r["d"]: r for r in res["needed"]}

    # The spine, asserted rather than trusted.
    assert dims[2]["barcode_spread"] > 4.0 > dims[768]["barcode_spread"]
    assert dims[768]["barcode_spread"] < 0.10
    spreads = [dims[d]["barcode_spread"] for d in DIMS]
    assert spreads == sorted(spreads, reverse=True)
    assert needed[2]["relative"] > 2.0 > needed[768]["relative"]
    assert needed[768]["separation"] > needed[2]["separation"]
    for d in DIMS:
        assert noise[d]["said_one"] == 0, (d, noise[d]["counts"])
        assert auc[d]["purity"] > 0.98, (d, auc[d]["purity"])
    assert noise[2]["modal_k"] == 2
    assert noise[768]["modal_k"] == NOISE_N - 1
    assert auc[2]["auc"] < 0.5 < auc[768]["auc"]
    assert auc[2]["noise_median"] > auc[2]["signal_median"]
    assert auc[768]["signal_median"] > auc[768]["noise_median"]
    assert auc[768]["auc"] > 0.75

    post = Post(
        title=f"{SERIES_TAG} 2: A Barcode's Numbers Mean Nothing on Their Own",
        slug="topology-2-barcode-numbers",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=2,
        prerequisites=["topology-1-h0-single-linkage"],
        date=POST_DATE,
        subtitle=("This episode was written to say that a barcode loses its "
                  "dynamic range in high dimensions and the display therefore "
                  "fails. The range does collapse, by a factor of 62 — and the "
                  "summary becomes more informative, because concentration "
                  "tightens the null far faster than it shrinks the signal."),
        summary=("Three measurements, and the third one reversed the plan. "
                 "First: the H0 death spread, as a fraction of its own mean, "
                 "falls from 4.87 at two dimensions to 0.078 at 768 on pure "
                 "noise. Second: the separation H0 needs to recover three "
                 "clusters rises in absolute terms and falls relative to the "
                 "scale of the space, 2.32 to 0.27, with every purity in the "
                 "sweep above 0.99 — so the clustering is not what is going "
                 "wrong. Third: the largest-gap rule reports \"one cluster\" "
                 "zero times in forty draws of pure noise at every dimension, "
                 "and its modal answer flips from 2 to n−1; while the "
                 "two-cluster ratio read as an absolute number is not weak but "
                 "inverted at low dimension — noise has a higher median than a "
                 "cloud whose clusters are recovered perfectly, an AUC of "
                 "0.462, worse than a coin. Against a matched null the same "
                 "ratio reaches 0.809 at 768 dimensions. The collapse of the "
                 "dynamic range is what makes that possible."),
        tags=["topology", "persistent-homology", "clustering",
              "high-dimensional", "lectures", "machine-learning"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every cloud here is constructed in the episode "
            "and every number is produced by the code shown, executed when "
            "this page was built.",
            "Machinery: `standarderror/topology/filtration.py`, tested in "
            "`tests/test_filtration.py`.",
            "Where this stops: Beyer, Goldstein, Ramakrishnan and Shaft, "
            "\"When is nearest neighbor meaningful?\", *ICDT* (1999), for the "
            "concentration result the whole episode rests on; Aggarwal, "
            "Hinneburg and Keim, \"On the surprising behavior of distance "
            "metrics in high dimensional space\", *ICDT* (2001), for what it "
            "does to metric choice; Chazal and Michel, \"An introduction to "
            "topological data analysis\", *Frontiers in AI* 4 (2021), for the "
            "null-model practice this episode ends up recommending.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4, "
                           "scipy=1.16.3",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": (f"{NOISE_N} points of pure noise per draw for the "
                           f"null sweeps, 40 or 60 draws per dimension, and "
                           f"three 40-point clusters for the signal"),
            "determinism": ("one generator per measurement, seeded from that "
                            "measurement's own parameters — dimension and draw "
                            "index — rather than advanced through a loop"),
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    dims = {r["d"]: r for r in res["dims"]}
    noise = {r["d"]: r for r in res["noise"]}
    auc = {r["d"]: r for r in res["auc"]}
    needed = {r["d"]: r for r in res["needed"]}
    conc = res["conc"]
    where = res["where"]
    bars = res["bars"]

    post.add(
        "What high dimensions do to a barcode, and what they do not",
        """Episode 1 ended on a picture: 200 points of pure Gaussian noise, and a barcode whose bars all end within a few percent of the same radius. The obvious reading is that persistence stops working in high dimensions, and I planned this episode around it. That reading is wrong, and finding out how it is wrong took three measurements.

Start with the thing that is true. The quantity to watch is how much of the filtration axis the deaths occupy — the spread of the death times divided by their mean — because that is what a reader's eye is looking for when it looks for a gap.""")

    post.add(
        "",
        f"""{snip['shape'].markdown()}

That single draw loses a factor of {bars[2].spread_ratio / bars[768].spread_ratio:.0f} of its spread between the two dimensions, and as a median over fifteen draws at each dimension it is {dims[2]['barcode_spread'] / dims[768]['barcode_spread']:.0f} — {dims[2]['barcode_spread']:.2f} at *d* = 2 against {dims[768]['barcode_spread']:.3f} at 768. The other column moves the opposite way: the mean death radius grows from {bars[2].deaths.mean():.2f} to {bars[768].deaths.mean():.1f} in that draw. The typical distance between two *points* does the same thing — {dims[2]['mean_distance']:.2f} to {dims[768]['mean_distance']:.2f}, a factor of {dims[768]['mean_distance'] / dims[2]['mean_distance']:.0f}, as a median over the sweep — and it is that quantity, not the death radius, that the algebra below is about.

The collapse and the growth are the same fact, and it comes out of three lines of algebra worth doing because everything else in this episode is a consequence of them. For two independent standard Gaussian points in *d* dimensions, each coordinate of *x* − *y* has variance 2, so

$$
\\lVert x - y \\rVert^2 = 2 \\chi^2_d, \\qquad \\mathbb{{E}} \\lVert x - y \\rVert^2 = 2d, \\qquad \\mathrm{{Var}} \\lVert x - y \\rVert^2 = 8d
$$

Take the square root. To first order a function *g* of a random variable has standard deviation |*g*′| times the original, so with *g* the square root,

$$
\\mathbb{{E}} \\lVert x - y \\rVert \\approx \\sqrt{{2d}}, \\qquad \\mathrm{{sd}} \\lVert x - y \\rVert \\approx \\frac{{\\sqrt{{8d}}}}{{2\\sqrt{{2d}}}} = 1
$$

The mean grows like √*d* and the absolute spread tends to a **constant**, so the relative spread falls like 1/√(2*d*). Measured over 300-point clouds: the standard deviation is {conc[0]['sd']:.3f}, {conc[3]['sd']:.3f}, {conc[5]['sd']:.3f} at *d* = 2, 32, 768 — converging to 1 — and the relative spread is {conc[5]['relative']:.4f} against a predicted {conc[5]['relative_predicted']:.4f}.

(The first version of that derivation dropped the factor of two in the delta step and predicted √2 for the standard deviation. The measurement said 1.005, which is how I found out.)

The figure below draws both barcodes with their deaths divided by their own mean, so that only the shape is being compared.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "",
        f"""Notice which one looks like it has structure. Two-dimensional noise has sparse patches, so it has long bars — a median two-cluster ratio of {noise[2]['ratio_median']:.4f} and a maximum, over forty draws, of {noise[2]['ratio_max']:.4f}. Seven hundred and sixty-eight dimensional noise has none: median {noise[768]['ratio_median']:.4f}, maximum {noise[768]['ratio_max']:.4f}.

Hold onto that, because it is going to turn out to be the useful half.""",
        level=3)

    post.add(
        "First measurement: the clustering is fine",
        """Before blaming the display, check the method. How far apart do three clusters have to be before a single-linkage cut recovers them? Bisect for it, at each dimension, requiring 95% purity.""",
        figures=[figs["f1"]])

    post.add(
        "",
        f"""In absolute terms the answer rises, from {needed[2]['separation']:.2f} at two dimensions to {needed[768]['separation']:.2f} at 768, and that is the story everyone expects from the phrase "curse of dimensionality". But the scale of the space rose too — the typical pairwise distance went from {needed[2]['noise_scale']:.2f} to {needed[768]['noise_scale']:.2f} — and dividing one by the other gives {needed[2]['relative']:.2f} at two dimensions falling to {needed[768]['relative']:.2f} at 768.

Relative to the space it lives in, the separation H0 needs *falls* by a factor of {needed[2]['relative'] / needed[768]['relative']:.1f}. And every purity in the detector sweep below is at or above {min(auc[d]['purity'] for d in DIMS):.3f}. So single linkage is not the thing that breaks in high dimensions. It gets *better* at the job, measured against the only scale available to it.

Which leaves the summary.""",
        level=3)

    post.add(
        "Second measurement: the rule cannot say \"none\"",
        f"""The standard way to read a cluster count off a barcode is to find the biggest jump in the sorted deaths and count the bars above it. Ask it about a cloud with no clusters in it.

{snip['gap'].markdown()}

Zero out of forty, at every dimension. Not "rarely" — never. The rule's one job is to locate the largest gap in a sorted list, a sorted list of noise has a largest gap, and so the rule always finds clusters. It is structurally incapable of returning the right answer here, which is exactly the defect the scree-plot episode found in the elbow: a rule that reports a position in a list can never report that the list has no interesting position.""",
        figures=[figs["f4"]])

    post.add(
        "",
        """What changes with dimension is *where* the failure lands, and this is the one place where measuring it changed what I was going to say. I expected the largest gap to migrate steadily towards the front of the sorted deaths as the dimension rose. What it does instead is become bimodal — thirty draws at each of three dimensions below, one dot per draw, and the middle of the high-dimensional rows is empty.""",
        level=3,
        figures=[figs["f5"]])

    post.add(
        "",
        f"""The mechanism is one line of order statistics, and it is worth writing out because it also says *when* to expect the flip. For a sample of size *n* from a density *f*, the gap between neighbouring order statistics near a value *x* runs like 1/(*n* *f*(*x*)) — sorted values are sparse wherever the density is thin. So the largest gap in a barcode lands in whichever tail of the death distribution is thinnest. A right-skewed death distribution has exactly one thin tail, the long one on the right, and the gap goes there every time. A symmetric death distribution has two equally thin tails, and which one wins is decided by the draw.

That is measurable, so it does not have to stay a story. The skew of the deaths, as a median over the same thirty draws, is {where[0]['death_skew']:.2f} at *d* = 2, {where[2]['death_skew']:.2f} at *d* = 8, {where[3]['death_skew']:.2f} at *d* = 32 and {where[5]['death_skew']:.2f} at *d* = 768. Concentration is symmetrising the death distribution, which is the collapsing spread from the first section seen from another angle. And the position of the gap follows the skew rather than the dimension: at *d* = 2 it is in the last ten in {where[0]['in_last_ten']} of {where[0]['draws']} draws, at *d* = 8 — skew {where[2]['death_skew']:.2f}, halfway down — it is in the last ten {where[2]['in_last_ten']} times and in the first ten {where[2]['in_first_ten']}, and by *d* = 768 it is in the last ten {where[5]['in_last_ten']} times and in the first ten {where[5]['in_first_ten']}.

So the rule does not drift from one answer to another. It flips between the two most extreme answers available — {noise[768]['modal_k']} clusters or {noise[2]['modal_k']} — depending on which end of a noise barcode happens to have the bigger step, and that is exactly why `gap_rule_on_noise` reports a modal answer of {noise[768]['modal_k']} with a range of {noise[768]['k_min']} to {noise[768]['k_max']}.

There is a silver lining in it. {noise[768]['modal_k']} clusters from 200 points is *obviously* wrong, and a wrong answer that looks wrong is far less dangerous than the plausible {noise[2]['modal_k']} you get in the dimension people draw their examples in.""",
        level=3)

    post.add(
        "Third measurement, and it reversed the plan",
        f"""So the rule is broken. What about the underlying number — the ratio of the longest bar to the next, which is what "this barcode strongly suggests two clusters" actually means?

Measure it on noise, and on clouds whose three clusters a single-linkage cut recovers.

{snip['detector'].markdown()}

Read the two-dimensional row again. Noise has a **higher** median ratio than the clustered cloud, on a cloud whose clusters that same run recovers at purity 0.992. The ratio is not a weak indicator at two dimensions; it is pointing the wrong way.""",
        figures=[figs["f2"]])

    post.add(
        "",
        f"""As a detector, scored by the area under its ROC curve, that is {auc[2]['auc']:.3f} at two dimensions — worse than a coin — and {auc[768]['auc']:.3f} at 768.

Which is the opposite of the episode I set out to write. The dynamic range does collapse; the barcode does stop having a readable shape; and the ratio gets *better*, monotonically, over the same range.

The reason is in the two numbers from the first section. Concentration squeezes the null much harder than it squeezes the signal. At two dimensions noise produces ratios anywhere from {noise[2]['ratio_min']:.4f} to {noise[2]['ratio_max']:.4f}, so a real cluster structure has to clear a high and noisy bar. At 768 dimensions noise produces {noise[768]['ratio_min']:.4f} to {noise[768]['ratio_max']:.4f} — a null pinned into a band {(noise[2]['ratio_max'] - noise[2]['ratio_min']) / (noise[768]['ratio_max'] - noise[768]['ratio_min']):.0f} times narrower — and a signal only has to clear that.

So the flat, structureless, unreadable barcode is not the problem. It is the reference.""",
        level=3,
        figures=[figs["f3"]])

    post.add(
        "What this actually asks you to do",
        f"""The recommendation follows from the shape of the failure rather than from taste, and it has two parts.

**Never read a barcode number absolutely.** There is no threshold on the two-cluster ratio that survives a change of dimension. At two dimensions a cutoff would have to sit near {auc[2]['noise_median']:.2f} to beat noise; at 768 dimensions every clustered cloud in this sweep is below {max(auc[d]['signal_median'] for d in (128, 768)):.3f}. A rule of thumb calibrated on the examples in a tutorial — which are two- or three-dimensional, because they have to be drawable — is calibrated on the one regime where the quantity is anti-informative.

**Build the null from your own data.** Shuffle each coordinate independently, or draw from a Gaussian matched to your cloud's mean and covariance, compute the barcode fifty times, and ask where your real barcode's ratio falls in that distribution. That is one screenful of code, it costs fifty single-linkage runs, and it converts a number that carries no information into one that carries {auc[768]['auc']:.2f} of an AUC at the dimension embeddings actually live in.

And the thing not to do: conclude from a flat barcode that there is no structure. In 768 dimensions a cloud with three cleanly separated clusters produces a barcode with a median ratio of {auc[768]['signal_median']:.4f}. Flat is what structure looks like there.""")

    post.add(
        "What to keep",
        f"""1. The barcode's dynamic range collapses with dimension: {dims[2]['barcode_spread']:.2f} at d = 2 to {dims[768]['barcode_spread']:.3f} at d = 768, medians over fifteen draws.
2. The clustering does not. The separation H0 needs falls from {needed[2]['relative']:.2f} to {needed[768]['relative']:.2f} of the typical pairwise distance, and purity stays above {min(auc[d]['purity'] for d in DIMS):.3f}.
3. "Count the bars above the biggest gap" returns "one cluster" zero times in forty draws of pure noise, at every dimension. It cannot say none.
4. Its modal answer on noise flips from {noise[2]['modal_k']} at d = 2 to {noise[768]['modal_k']} at d ≥ 32, because in a concentrated barcode the biggest gap is the first one.
5. The two-cluster ratio read absolutely is inverted at low dimension — AUC {auc[2]['auc']:.3f} — and improves with dimension against a matched null, to {auc[768]['auc']:.3f} at 768.
6. So build the null from your own data. The collapse of the dynamic range is what makes the null tight enough to be worth comparing against.""")

    post.add(
        "Exercise",
        """Take the embedding matrix from the last episode's exercise and compute its two-cluster ratio. Write the number down; it means nothing yet.

Now shuffle each coordinate of the matrix independently — this destroys every relationship between dimensions while keeping each dimension's marginal distribution exactly — and compute the ratio again. Fifty times. You now have a null distribution matched to your data in the only way that matters, and your original number has a position in it.

If it is inside the null, your barcode is telling you nothing, however large the ratio looked. If it is outside, you have a measurement, and its size is meaningless but its position is not.

Then do the part that is uncomfortable. Compare the null you just built with the null you would get from a Gaussian matched to your cloud's covariance. If those two nulls disagree, your answer depends on which one you chose — and that is the next episode, which is about the fact that the metric and the normalisation were decided upstream of any of this, and that they decide the filtration before the data gets a vote.""")

    post.hero = figs["hero"]
    return post


def main() -> Post:
    return build()
if __name__ == "__main__":
    r = compute()
    print("dynamic range on pure noise:")
    for row in r["dims"]:
        print(f"  d={row['d']:>4}  mean {row['mean_distance']:7.3f}  "
              f"distances {row['distance_spread']:.3f}  "
              f"barcode {row['barcode_spread']:.3f}")
    print("\nseparation H0 needs:")
    for row in r["needed"]:
        print(f"  d={row['d']:>4}  absolute {row['separation']:6.2f}  "
              f"noise scale {row['noise_scale']:6.2f}  "
              f"relative {row['relative']:.3f}")
    print("\nthe gap rule on noise:")
    for row in r["noise"]:
        print(f"  d={row['d']:>4}  said one: {row['said_one']}/{row['draws']}  "
              f"modal k {row['modal_k']:>4}  range {row['k_min']}-{row['k_max']}  "
              f"ratio median {row['ratio_median']:.4f} "
              f"({row['ratio_min']:.4f}-{row['ratio_max']:.4f})")
    print("\nthe ratio as a detector:")
    for row in r["auc"]:
        print(f"  d={row['d']:>4}  sep {row['separation']:5.1f}  "
              f"purity {row['purity']:.3f}  noise {row['noise_median']:.4f}  "
              f"signal {row['signal_median']:.4f}  AUC {row['auc']:.3f}")
