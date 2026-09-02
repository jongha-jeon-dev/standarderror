"""Linear Algebra 4: PCA When Two Eigenvalues Are Equal.

The claim: "we interpret the second component as ..." is a statement about an
axis, and an axis only exists when its eigenvalue is separated from its
neighbours. Everything else in the episode is that sentence made checkable.

Three constructions, each with a spectrum known in closed form before any sample
is drawn:

* `equicorrelation(5, 0.4)` has one eigenvalue of 2.6 and four of 0.6 -- an exact
  tie. Rotating the tied eigenvectors by any angle reconstructs the same matrix,
  so the axes carry no information at all.
* `block_pairs([0.80, 0.78, 0.30])` has eigenvalues 1.80, 1.78, 1.30, 0.70, 0.22,
  0.20. The two *largest* are 0.02 apart, so the component carrying the most
  variance is the one least determined -- which is the opposite of how components
  are usually triaged.
* A sweep over the top gap, holding the rest of the spectrum fixed, so the
  relationship between gap and movement is measured rather than argued.

And the finding that makes it an episode of this series rather than a note: the
check a practitioner would run -- bootstrap the rows, watch the loadings -- is
blind to the problem, because it centres on the sample's own arbitrary axis. It
reports about 15 degrees of movement where the axis is actually 41 degrees from
the truth.

Run: `standarderror run lec004_equal_eigenvalues --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.linalg import spectral as sp
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
POST_DATE = date(2026, 8, 31)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Linear Algebra for Data Science, Taught Through What Breaks"
SERIES_TAG = "Linear Algebra"

#: The exact tie. Five variables, every pair correlated 0.4: eigenvalues 2.6 and
#: 0.6 four times over.
EQUI_P = 5
EQUI_RHO = 0.40
ROTATIONS = (7.0, 45.0, 123.4)

#: The near tie. Within-pair correlations, so the spectrum is 1 +- each of these:
#: 1.80, 1.78, 1.30, 0.70, 0.22, 0.20. The top gap is 0.02 by construction.
BLOCKS = (0.80, 0.78, 0.30)
LABELS = ("A1", "A2", "B1", "B2", "C1", "C2")
N = 400
REPS = 400

#: The sweep. `base - gap` is the second pair's correlation, so the top gap is
#: exactly the requested one and the rest of the spectrum does not move.
#:
#: The third correlation is 0.05 rather than the 0.30 used above, and the sweep
#: stops at 0.20, because both of the obvious choices create a *second* near-tie
#: at the wide end: with 0.30 and a gap of 0.40 the spectrum is 1.8, 1.4, 1.3,
#: ... and the 0.1 between the second and third eigenvalues is a new tie the
#: curve would be measuring instead of the one being swept.
SWEEP_OTHER = 0.05
SWEEP_GAPS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
SWEEP_REPS = 200

#: Draws used to measure how strongly the sampling noise couples two tied
#: directions -- the numerator of the first-order perturbation term.
COUPLING_REPS = 400

#: The bootstrap comparison: how many independent samples, and how many
#: resamples of each.
BOOT_SAMPLES = 30
BOOT_REPS = 200

SEED = 21


def _draw(n: int, C: np.ndarray, rng) -> np.ndarray:
    return rng.standard_normal((n, C.shape[0])) @ np.linalg.cholesky(C).T


def compute() -> dict:
    out: dict = {}

    # --- the exact tie ---------------------------------------------------
    E = sp.equicorrelation(EQUI_P, EQUI_RHO)
    tied = sp.spectrum(E)
    rebuilt = []
    for deg in ROTATIONS:
        V = sp.rotate_within(tied.vectors, 1, 2, deg)
        rebuilt.append({"degrees": deg,
                        "max_abs_error": float(np.abs(
                            V @ np.diag(tied.values) @ V.T - E).max())})
    out["tie"] = {"matrix": E, "spectrum": tied, "rotations": rebuilt,
                  "multiplicity": EQUI_P - 1}

    # --- the near tie, measured ------------------------------------------
    C = sp.block_pairs(BLOCKS)
    truth_near = sp.spectrum(C)
    rng = np.random.default_rng(SEED)
    near = sp.bootstrap_components(C, n=N, reps=REPS, rng=rng)
    out["near"] = {"matrix": C, "result": near,
                   "table": sp.stability_table(near)}

    # --- the gap, swept ---------------------------------------------------
    out["sweep"] = sp.gap_sweep(SWEEP_GAPS, base=BLOCKS[0], other=SWEEP_OTHER,
                                n=N, reps=SWEEP_REPS,
                                rng=np.random.default_rng(SEED + 1))

    # --- why two pairs with the same gap do not move the same amount ------
    # First-order perturbation theory writes the movement of eigenvector j as a
    # sum of (v_k' E v_j) / (lam_j - lam_k) over the other k. Both tied pairs
    # here have the same denominator, so anything left over is the numerator.
    V, lam = truth_near.vectors, truth_near.values
    r = np.random.default_rng(SEED + 2)
    coupling = {0: [], 4: []}
    for _ in range(COUPLING_REPS):
        E = sp.sample_correlation(N, C, rng=r) - C
        for j in coupling:
            coupling[j].append(abs(V[:, j + 1] @ E @ V[:, j]))
    out["coupling"] = [
        {"pair": f"PC{j + 1}-PC{j + 2}",
         "gap": float(lam[j] - lam[j + 1]),
         "coupling": float(np.median(coupling[j])),
         "first_order": float(np.degrees(np.arctan(
             np.median(coupling[j]) / (lam[j] - lam[j + 1])))),
         "measured": float(near["median_angle"][j])}
        for j in sorted(coupling)]

    # --- and the check that cannot see it ---------------------------------
    truth = truth_near
    reported, actual, swap = [], [], []
    for i in range(BOOT_SAMPLES):
        r = np.random.default_rng(SEED + 100 + i)
        X = _draw(N, C, r)
        b = sp.row_bootstrap(X, reps=BOOT_REPS, rng=r)
        reported.append(float(b["median_angle"][0]))
        swap.append(float(b["swap_rate"][0]))
        actual.append(sp.vector_angle(sp.spectrum(np.corrcoef(X.T)).vectors[:, 0],
                                      truth.vectors[:, 0]))
    out["bootstrap"] = {
        "reported": np.array(reported), "actual": np.array(actual),
        "swap_rate": np.array(swap),
        "median_reported": float(np.median(reported)),
        "median_actual": float(np.median(actual)),
        "understatement": float(np.median(actual) / np.median(reported)),
        # Per-sample rather than median-of-medians: the point of the last figure
        # is that this is not a bias with a correction factor.
        "ratio_lo": float((np.array(actual) / np.array(reported)).min()),
        "ratio_hi": float((np.array(actual) / np.array(reported)).max()),
        "understated_count": int((np.array(actual) > np.array(reported)).sum()),
        "samples": BOOT_SAMPLES, "reps": BOOT_REPS,
    }
    return out


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    out: dict = {}
    tie, near, sweep, boot = res["tie"], res["near"], res["sweep"], res["bootstrap"]
    table = near["table"]

    # --- f0: every axis in a tied plane is the right answer ----------------
    def plane(ax, m):
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(t), np.sin(t), color=m.grid, lw=1.4)
        for k, deg in enumerate((0.0, 33.0, 66.0)):
            a = np.radians(deg)
            colour = m.series[k % len(m.series)]
            for sign in (1, -1):
                ax.annotate("", xy=(sign * np.cos(a), sign * np.sin(a)),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.0,
                                            shrinkA=0, shrinkB=0))
                ax.annotate("", xy=(-sign * np.sin(a), sign * np.cos(a)),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.0,
                                            shrinkA=0, shrinkB=0, alpha=0.55))
        ax.text(0.0, -1.28, "every pair of arrows is a correct set of eigenvectors",
                ha="center", va="top", fontsize=9, color=m.ink_secondary)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.25)

    out["f0"] = charts.diagram(
        plane,
        title="When two eigenvalues are equal there is no second component",
        subtitle=(f"The eigenspace of the repeated eigenvalue of a "
                  f"{EQUI_P}-variable equicorrelation matrix. Three of the "
                  f"infinitely many orthonormal bases for it are drawn."),
        source="A diagram, not a measurement.",
        ticks=False, equal=True,
        alt=("A circle with three pairs of perpendicular arrows through its "
             "centre at different rotations, all equally valid."),
        caption=(f"Rotating the tied eigenvectors by "
                 f"{tie['rotations'][-1]['degrees']:.1f} degrees and rebuilding "
                 f"the matrix changes it by "
                 f"{tie['rotations'][-1]['max_abs_error']:.0e}. The plane is a "
                 f"fact about the data; the arrows in it are a fact about "
                 f"LAPACK."),
        path=str(IMG / f"lec04-f0-plane.{EXT}"))[0]

    # --- t1: the table the episode is built on -----------------------------
    rows = []
    for r in table:
        rows.append([f"PC{r['component']}", f"{r['eigenvalue']:.2f}",
                     f"{r['variance_share']:.1%}", f"PC{r['neighbour']}",
                     f"{r['neighbour_gap']:.2f}", f"{r['swap_rate']:.0%}",
                     f"{r['median_angle']:.1f}°"])
    out["t1"] = charts.table_image(
        rows,
        header=["", "eigenvalue", "share of variance", "nearest", "gap",
                "swaps with it", "axis moves"],
        title="The share of variance does not order the stability",
        subtitle=(f"{REPS} fresh samples of {N} rows from a correlation matrix "
                  f"whose eigenvalues are known exactly. Read the third column "
                  f"against the last."),
        source="Simulated; standarderror/linalg/spectral.py.",
        bold_cols=(2, 6),
        alt=("Table of six principal components with variance share, gap to the "
             "nearest eigenvalue, swap rate and median angle moved."),
        caption=(f"PC1 carries the most variance of any component and moves "
                 f"{table[0]['median_angle']:.0f} degrees; PC3 carries "
                 f"{table[2]['variance_share']:.0%} and moves "
                 f"{table[2]['median_angle']:.0f}. The column that predicts the "
                 f"last one is the gap, not the share."),
        path=str(IMG / f"lec04-t1-stability.{EXT}"))[0]

    # --- f1: the gap, swept ------------------------------------------------
    out["f1"] = charts.lines(
        pd.DataFrame(
            {"the first axis": [p["median_angle"] for p in sweep],
             "the plane the first two axes span":
                 [p["median_plane_angle"] for p in sweep]},
            index=[p["gap"] for p in sweep]),
        title="Closing the gap moves the axis and leaves the plane alone",
        subtitle=(f"Top eigenvalue gap set by construction, the rest of the "
                  f"spectrum held fixed. {SWEEP_REPS} samples of {N} rows per "
                  f"point."),
        xlabel="gap between the top two eigenvalues",
        ylabel="median degrees moved across samples",
        source="Simulated; standarderror/linalg/spectral.py.",
        logx=True,
        alt=("Two curves against gap on a log axis: one rising steeply as the "
             "gap closes, the other flat across the whole range."),
        caption=("Two quantities from the same decomposition, on the same axis, "
                 "in the same units. One of them is a property of the data and "
                 "the other is not."),
        path=str(IMG / f"lec04-f1-gap.{EXT}"))[0]

    # --- f2: ordered by variance, and not ordered by anything else ---------
    out["f2"] = charts.ranked_bars(
        # One space, not two: SVG and HTML collapse runs of whitespace, so a
        # double space here renders as one anyway and only makes the label
        # differ between the PNG and the SVG that goes to Notion.
        [f"PC{r['component']} ({r['variance_share']:.1%})" for r in table],
        [r["median_angle"] for r in table],
        title="Ordered by share of variance, largest first",
        subtitle=("If the share of variance told you which components to trust, "
                  "these bars would get longer as you read down."),
        xlabel="median degrees the axis moves across samples",
        source="Simulated; standarderror/linalg/spectral.py.",
        sort="none", value_fmt=".1f",
        alt=("Horizontal bars for six components in order of variance "
             "explained; the first two are much longer than the third and "
             "fourth, and the last two are long again."),
        caption=("They do not. The two longest bars belong to the two largest "
                 "components, because those two are 0.02 apart; the short bars "
                 "in the middle are the ones with room around them."),
        path=str(IMG / f"lec04-f2-share.{EXT}"))[0]

    # --- f4: and the check cannot see it -----------------------------------
    def understated(ax, m):
        hi = float(max(boot["actual"].max(), boot["reported"].max())) * 1.08
        ax.plot([0, hi], [0, hi], color=m.muted, lw=1.6, ls=(0, (4, 3)),
                label="where the two would agree")
        ax.scatter(boot["reported"], boot["actual"], s=34,
                   color=m.series[0], zorder=3, label="one sample")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)

    out["f4"] = charts.diagram(
        understated,
        title="The bootstrap centres on the answer you happened to get",
        subtitle=(f"{boot['samples']} independent samples of {N} rows. For each, "
                  f"the median movement over {boot['reps']} resamples of its own "
                  f"rows against the angle its first axis really sits at."),
        xlabel="degrees the row bootstrap reports",
        ylabel="degrees the axis actually is from the truth",
        source="Simulated; standarderror/linalg/spectral.py.",
        alt=(f"A scatter of {boot['samples']} points, most of them well above "
             f"the diagonal line and a few below it."),
        caption=(f"Median {boot['median_actual']:.0f} degrees against "
                 f"{boot['median_reported']:.0f} reported, and "
                 f"{boot['understated_count']} of "
                 f"{boot['samples']} samples above the line. Not a bias you "
                 f"could correct for — the per-sample ratio runs from "
                 f"{boot['ratio_lo']:.1f} to {boot['ratio_hi']:.1f} — but a "
                 f"check whose answer is unrelated to the question."),
        path=str(IMG / f"lec04-f4-bootstrap.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    """The cover: no axes, the wrong triage, and the check that is blind."""
    table = res["near"]["table"]
    boot = res["bootstrap"]

    def spun(panel, m):
        t = np.linspace(0, 2 * np.pi, 200)
        panel.plot(0.5 + 0.34 * np.cos(t), 0.5 + 0.42 * np.sin(t),
                   color=m.grid, lw=1.6)
        for deg in (0.0, 40.0, 80.0):
            a = np.radians(deg)
            panel.plot([0.5 - 0.32 * np.cos(a), 0.5 + 0.32 * np.cos(a)],
                       [0.5 - 0.40 * np.sin(a), 0.5 + 0.40 * np.sin(a)],
                       color=m.ink, lw=2.0)
        panel.set_xlim(0, 1); panel.set_ylim(0, 1)

    def bars(panel, m):
        share = [r["variance_share"] for r in table]
        angle = [r["median_angle"] for r in table]
        x = np.arange(len(share))
        panel.bar(x - 0.2, np.array(share) / max(share), width=0.38, color=m.grid)
        panel.bar(x + 0.2, np.array(angle) / max(angle), width=0.38, color=m.ink)
        panel.set_xlim(-0.7, len(share) - 0.3); panel.set_ylim(0, 1.15)

    def two_rulers(panel, m):
        panel.barh([0.66], [0.82], height=0.20, color=m.ink, left=0.09)
        panel.barh([0.30], [0.29], height=0.20, color=m.grid, left=0.09)
        panel.set_xlim(0, 1); panel.set_ylim(0, 1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=4,
        headline="The biggest component is the least determined",
        panels=[
            (spun, f"{res['tie']['rotations'][-1]['max_abs_error']:.0e}",
             "cost of spinning them"),
            (bars, f"{table[0]['median_angle']:.0f}°", "PC1 moves this far"),
            (two_rulers, f"{boot['understatement']:.1f}×",
             "what the check misses"),
        ],
        note=("Two eigenvalues 0.02 apart, on a matrix whose spectrum is known "
              "exactly. The axes of a tied pair can be rotated to any angle "
              "without changing the matrix they came from, the component "
              "carrying the most variance is the one that moves furthest, and "
              "the resampling check a reader would run reports a third of the "
              "movement that is actually there."),
        alt=("A three-panel hand-drawn strip. The first frame shows an ellipse "
             "crossed by three differently angled lines through its centre, "
             "marked with a number near machine precision. The second shows six "
             "pairs of bars where the tall grey bars do not line up with the "
             "tall dark ones, marked with an angle. The third shows a long dark "
             "bar above a much shorter grey one."),
        mode="light",
        path=str(IMG / f"lec04-hero.{EXT}"))[0]


# ---------------------------------------------------------------- the post

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["tie"] = s.run(f"""
        import numpy as np

        # {EQUI_P} variables, every pair correlated {EQUI_RHO}. The spectrum is
        # known in closed form: 1 + (p-1)*rho once, and 1 - rho {EQUI_P - 1} times.
        p, rho = {EQUI_P}, {EQUI_RHO}
        R = (1 - rho) * np.eye(p) + rho * np.ones((p, p))
        vals, V = np.linalg.eigh(R)
        print(f"eigenvalues  {{np.sort(vals)[::-1].round(6)}}")

        # Rotate two of the tied eigenvectors into each other by any angle at all
        # and rebuild the matrix from the rotated basis.
        t = np.radians(123.4)
        W = V.copy()
        W[:, 1] = np.cos(t) * V[:, 1] + np.sin(t) * V[:, 2]
        W[:, 2] = -np.sin(t) * V[:, 1] + np.cos(t) * V[:, 2]

        print(f"still orthonormal   {{np.abs(W.T @ W - np.eye(p)).max():.1e}}")
        print(f"rebuilds the matrix {{np.abs(W @ np.diag(vals) @ W.T - R).max():.1e}}")
    """, expect=["eigenvalues", "rebuilds the matrix"])

    out["gap"] = s.run(f"""
        # Three independent pairs of variables, correlated {BLOCKS[0]}, {BLOCKS[1]}
        # and {BLOCKS[2]} within each pair. The eigenvalues are 1 +- each of those,
        # so the top gap is {BLOCKS[0] - BLOCKS[1]:.2f} by construction rather than by luck.
        C = np.eye(6)
        for i, c in enumerate({list(BLOCKS)}):
            C[2 * i, 2 * i + 1] = C[2 * i + 1, 2 * i] = c

        vals = np.sort(np.linalg.eigvalsh(C))[::-1]
        share = vals / vals.sum()
        print("        eigenvalue  share  gap to next")
        for k, v in enumerate(vals):
            gap = f"{{v - vals[k + 1]:.2f}}" if k + 1 < len(vals) else "   -"
            print(f"  PC{{k + 1}}   {{v:9.2f}}  {{share[k]:5.1%}}      {{gap}}")
    """, expect=["PC1", "PC6"])

    out["boot"] = s.run(f"""
        # One sample, and the check a reader would actually run on it.
        rng = np.random.default_rng({SEED + 100})
        X = rng.standard_normal(({N}, 6)) @ np.linalg.cholesky(C).T

        def first_axis(M):
            return np.linalg.eigh(np.corrcoef(M.T))[1][:, -1]

        def angle(u, v):
            c = abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
            return np.degrees(np.arccos(min(c, 1.0)))

        truth = np.linalg.eigh(C)[1][:, -1]
        mine = first_axis(X)

        moves = [angle(first_axis(X[rng.integers(0, {N}, {N})]), mine)
                 for _ in range({BOOT_REPS})]

        print(f"the bootstrap says PC1 moves  {{np.median(moves):.1f}} degrees")
        print(f"PC1 is actually this far off  {{angle(mine, truth):.1f}} degrees")
    """, expect=["the bootstrap says", "actually this far off"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    tie, near, sweep, boot = res["tie"], res["near"], res["sweep"], res["bootstrap"]
    table = near["table"]
    pc1, pc3 = table[0], table[2]
    tight, loose = sweep[0], sweep[-1]

    # The spine, asserted rather than trusted. Any of these failing means a
    # sentence below has become false and the post must not publish.
    assert tie["rotations"][-1]["max_abs_error"] < 1e-12, tie["rotations"]
    assert pc1["variance_share"] > pc3["variance_share"], table
    assert pc1["median_angle"] > 3 * pc3["median_angle"], table
    assert pc1["median_plane_angle"] < pc1["median_angle"] / 3, table
    assert boot["understatement"] > 2.0, boot
    assert tight["median_angle"] > 2 * loose["median_angle"], sweep

    post = Post(
        title=f"{SERIES_TAG} 4: PCA When Two Eigenvalues Are Equal",
        slug="linear-algebra-4-equal-eigenvalues",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=4,
        prerequisites=["linear-algebra-3-positive-definite"],
        date=POST_DATE,
        subtitle=("A correlation matrix whose two largest eigenvalues differ by "
                  "0.02, where the component carrying the most variance is the "
                  "one that moves 42 degrees between samples — and the "
                  "resampling check reports a third of it."),
        summary=("Every applied account of PCA triages components by the share "
                 "of variance they carry, and stability does not work that way: "
                 "an eigenvector is determined by the distance from its "
                 "eigenvalue to the nearest other one, so the largest component "
                 "in a decomposition can be the least trustworthy axis in it. "
                 "When two eigenvalues are exactly equal the axes stop existing "
                 "altogether — any rotation of them rebuilds the same matrix — "
                 "and when they are merely close, the axes swing by tens of "
                 "degrees while the plane they span holds still. The bootstrap "
                 "everybody runs cannot see this, because it centres on the "
                 "arbitrary answer the sample happened to give."),
        tags=["linear-algebra", "pca", "dimensionality-reduction",
              "lectures", "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every correlation matrix here is written down in "
            "the episode and every number is produced by the code shown, "
            "executed when this page was built.",
            "Machinery: `standarderror/linalg/spectral.py`, tested in "
            "`tests/test_spectral.py`.",
            "Where this stops, and who does it properly: Davis and Kahan, \"The "
            "rotation of eigenvectors by a perturbation. III\", *SIAM J. Numer. "
            "Anal.* 7 (1970); Yu, Wang and Samworth, \"A useful variant of the "
            "Davis-Kahan theorem for statisticians\", *Biometrika* 102 (2015); "
            "Anderson, *An Introduction to Multivariate Statistical Analysis*, "
            "chapter 11, on the distribution of sample eigenvectors.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": (f"{REPS} samples of {N} rows for the table, "
                           f"{SWEEP_REPS} per point in the gap sweep, and "
                           f"{BOOT_SAMPLES} independent samples each bootstrapped "
                           f"{BOOT_REPS} times for the last figure"),
            "determinism": (f"one seed, {SEED}, and every draw derived from it; "
                            f"the correlation matrices themselves are exact"),
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    tie, near, sweep, boot = res["tie"], res["near"], res["sweep"], res["bootstrap"]
    table = near["table"]
    pc1, pc2, pc3, pc4 = table[0], table[1], table[2], table[3]
    tight, loose = sweep[0], sweep[-1]
    rot = tie["rotations"][-1]
    top, bot = res["coupling"][0], res["coupling"][1]

    post.add(
        "Last episode's exercise",
        """The exercise was: take a dataset with six or more numeric columns, bootstrap the rows, and count how often each adjacent pair of eigenvalues swaps order. Then look at the loadings of the pair that swaps most.

Most people expect the swapping to happen at the bottom, among the small components nobody trusts anyway. It happens wherever two eigenvalues are close, and that can be anywhere — including at the very top. The pair that swaps is not the pair with the smallest eigenvalues. It is the pair with the smallest *gap*.

And then the second half of the exercise, which is the part that matters. When two components swap places between one resample and the next, the sentence "the second component represents ..." has been written about a direction the data does not determine. The plane those two components span is a real feature of the data. Which pair of perpendicular arrows inside that plane your library handed back is not.""")

    post.add(
        "Two eigenvalues that are exactly equal",
        f"""Start with the case where this is not a matter of degree. Take {EQUI_P} variables and correlate every pair equally, at {EQUI_RHO}. This matrix has a spectrum you can write down without computing anything: one eigenvalue of {tie['spectrum'].values[0]:.1f}, and {tie['multiplicity']} eigenvalues of {tie['spectrum'].values[1]:.1f}.

Not approximately {tie['spectrum'].values[1]:.1f}. Exactly, {tie['multiplicity']} times over.

{snip['tie'].markdown()}

Read the last line. The rotated basis is still orthonormal, it still consists of eigenvectors of *R*, and rebuilding the matrix from it returns the original to {rot['max_abs_error']:.0e} — machine precision. The rotation was by {rot['degrees']:.1f} degrees, and it could have been by any other number.

So there is no second component here, and no third or fourth either. There is a one-dimensional eigenspace for {tie['spectrum'].values[0]:.1f} and a {tie['multiplicity']}-dimensional eigenspace for {tie['spectrum'].values[1]:.1f}, and *within* that second eigenspace every orthonormal basis is as correct as every other. The particular arrows `eigh` returned are a fact about the algorithm LAPACK implements. Any interpretation of them is an interpretation of LAPACK.""",
        figures=[figs["f0"]])

    post.add(
        "Nearly equal is the same problem, measured in degrees",
        f"""An exact tie is a mathematical statement, and sampled data never produces one. What sampled data produces constantly is a near tie, and the near tie inherits the problem in proportion.

Here is a correlation matrix built so its spectrum is known before any sample is drawn. Three independent pairs of variables, correlated {BLOCKS[0]}, {BLOCKS[1]} and {BLOCKS[2]} within each pair; a matrix like that is block diagonal, and its eigenvalues are exactly 1 ± each of those three numbers.

{snip['gap'].markdown()}

The top two eigenvalues are {BLOCKS[0] - BLOCKS[1]:.2f} apart. They carry {pc1['variance_share']:.1%} and {pc2['variance_share']:.1%} of the variance — the two largest components in the decomposition, the two any analysis would keep and name.

Now draw {N} rows from this population {REPS} times and ask, for each component, how far its axis lands from the population's.""",
        figures=[figs["t1"]])

    post.add(
        "",
        f"""Read the table by comparing its third column with its last. **PC1 carries the most variance of anything in the matrix and its axis moves {pc1['median_angle']:.0f} degrees.** PC3 carries {pc3['variance_share']:.0%} — nine points less — and moves {pc3['median_angle']:.1f}. PC4 carries {pc4['variance_share']:.0%} and moves {pc4['median_angle']:.1f}.

The share of variance does not order the stability. It does not even correlate with it. What orders it is the column in between: the distance from each eigenvalue to the nearest other one.

That is not a coincidence of this matrix, it is a theorem. The Davis-Kahan *sin θ* theorem bounds how far an eigenvector can move when the matrix is perturbed by *E*, and in the form Yu, Wang and Samworth state for statisticians it reads

$$
\\sin \\theta  \\le  \\frac{{2^{{3/2}}  \\lVert E \\rVert_{{\\mathrm{{op}}}}}}{{\\min(\\lambda_{{j-1}} - \\lambda_j,  \\quad \\lambda_j - \\lambda_{{j+1}})}}
$$

The numerator is how much noise there is. The denominator is the gap. Nothing in that expression is the share of variance.""",
        level=3)

    post.add(
        "The gap is the parameter, so set it",
        f"""The construction above is worth the two lines it costs, because it makes the gap a *dial*. Change the second pair's correlation and the top gap changes by exactly that amount, while the rest of the spectrum stays where it is. So the relationship between gap and movement can be measured rather than argued about.

At a gap of {tight['gap']}, the first axis lands a median of {tight['median_angle']:.0f} degrees from the truth and swaps places with the second {tight['swap_rate']:.0%} of the time. At {loose['gap']}, it moves {loose['median_angle']:.0f} degrees and never swaps. And the plane those two axes span — the two-dimensional subspace, rather than either arrow in it — sits at {tight['median_plane_angle']:.0f} to {loose['median_plane_angle']:.0f} degrees across the entire range, unmoved by the thing that moves the axis by a factor of three.""",
        figures=[figs["f1"]])

    post.add(
        "",
        """That flat line is the usable part of the answer, and it is flat for a reason worth stating. The angle between two *subspaces* — the principal angle — does not depend on which basis you chose for either one. Rotate the columns among themselves and it does not move. That is precisely the invariance a single eigenvector does not have, and it is why "these two variables and those two load on a common plane" survives resampling when "PC2 is the size factor" does not.

So a decomposition with a near-tied pair is not uninformative. It is informative about a plane and silent about the axes in it, and the usual reporting format has no way to say that.""",
        level=3)

    post.add(
        "What the share of variance is actually for",
        f"""It is worth being fair to the share of variance, because it is not a useless number — it is a number answering a different question.

The share tells you how much of the total variance you lose by truncating after *k* components. That is a statement about the *subspace* spanned by the first *k*, and it is correct: keep the top four here and you have kept {sum(r['variance_share'] for r in table[:4]):.0%} of the variance, whatever basis those four are expressed in. Reconstruction error, compression, how many components to retain — all subspace questions, all correctly answered by the eigenvalues.

The moment the question becomes *what does this one component mean*, the object under discussion changes from a subspace to an axis, and the number that governed the first question stops applying.""",
        figures=[figs["f2"]])

    post.add(
        "And the check you would run cannot see it",
        f"""There is an obvious diagnostic here and it is the one everybody reaches for: bootstrap the rows, recompute the components, and look at how much the loadings move. If they barely move, the component is stable.

Run it on one sample of this matrix.

{snip['boot'].markdown()}

Both numbers are about the first component of the same sample, and they disagree by a factor of three. Across {boot['samples']} independent samples the median is {boot['median_actual']:.0f} degrees actual against {boot['median_reported']:.0f} reported, and the bootstrap comes in low on {boot['understated_count']} of the {boot['samples']}.""",
        figures=[figs["f4"]])

    post.add(
        "",
        """The reason is structural rather than a matter of too few replicates. The bootstrap measures the spread of the estimator *around the estimate*, and the estimate here is one arbitrary direction in a plane the data does not resolve. Every resample lands near the same arbitrary direction, because every resample is mostly the same rows. The spread is genuinely small. It is small around the wrong centre.

This is the third episode in a row where the standard check is blind to the standard failure, and the three are the same shape. Episode one: the residual is at machine precision while the answer is wrong by 302 percent, because a backward-stable solver guarantees the residual. Episode two: the orthogonality check reads 6e-16 for a fit wrong by 321 percent, because the normal equations *are* the orthogonality condition. Here: the bootstrap reports a small movement, because it is measuring movement relative to the answer whose arbitrariness is the problem.

The pattern is worth naming. **A diagnostic computed from the same object it is auditing will confirm that object.** The checks that work in all three episodes are the ones computed from the matrix before the estimate exists — the condition number, the singular values, and here the eigenvalue gap.""",
        level=3)

    post.add(
        "The gap is the denominator, not the whole story",
        f"""One number in that table has been quietly inconsistent with the account so far, and it is worth stopping on because the honest version of the claim is narrower than the slogan.

PC1 and PC2 are {top['gap']:.2f} apart and the axis moves {top['measured']:.0f} degrees. PC5 and PC6 are also {bot['gap']:.2f} apart — the same gap, in the same matrix, from the same samples — and they move {bot['measured']:.0f}. Half as far, on an identical denominator.

First-order perturbation theory says where the rest of it went. The movement of the *j*-th eigenvector under a perturbation *E* is a sum over the other eigenvectors,

$$
\\delta v_j  \\approx  \\sum_{{k \\ne j}} \\frac{{v_k^{{\\top}} E  v_j}}{{\\lambda_j - \\lambda_k}}  v_k
$$

and the gap is only the denominator. The numerator is how strongly the noise you actually got connects those two particular directions, and it is not a constant: measured across {COUPLING_REPS} samples it is {top['coupling']:.4f} for the top pair and {bot['coupling']:.4f} for the bottom one — a factor of {top['coupling'] / bot['coupling']:.0f}, inside one matrix.

Put those together and the first-order estimate for the bottom pair is {bot['first_order']:.0f} degrees against {bot['measured']:.0f} measured, which is about as well as a first-order approximation ever does. For the top pair it predicts {top['first_order']:.0f} against {top['measured']:.0f}, and the overshoot is the theory announcing its own failure: when the coupling is {top['coupling'] / top['gap']:.1f} times the gap, "small perturbation" has stopped being true and a linearisation is the wrong tool.

This is also why the Davis-Kahan bound quoted earlier is so loose. Its numerator is the operator norm of *E* — the worst this perturbation could do to *any* direction — where what governs one eigenvector is what the perturbation does to that direction specifically. At every gap in the sweep above, the bound evaluates to 90 degrees: the axis could be anywhere in the half-space. That is a true statement, and it is the correct one for a worst case. It is also not a measurement, which is why this episode reports the angle rather than the bound.""")

    post.add(
        "What to take away, and what is still hiding",
        """Four things, in the order you would use them.

**Print the gaps, not just the eigenvalues.** One line beside your scree plot: `np.diff(np.sort(vals)[::-1])`. A component whose gap to its neighbour is small compared with your sampling noise has an arbitrary axis, and you now know that before you have written a sentence about it.

**Interpret subspaces when the axes are tied.** "These four variables share a two-dimensional structure" is supportable here; "PC1 is the size factor and PC2 is the shape factor" is not. If you need named axes inside a tied plane, rotate deliberately — varimax and its relatives exist for exactly this — and say that you did, because the rotation is then your modelling choice rather than your library's default.

**Do not read the share of variance as a stability ranking.** It answers how much you lose by truncating, which is a question about the subspace you keep, and it is correct about that.

**And do not let a bootstrap of the loadings reassure you.** It centres on the answer you already have.

One practical trap that is *not* this problem and looks exactly like it. An eigenvector is defined up to sign, and `eigh` makes no promise about which one it hands back, so two runs on nearly identical matrices routinely differ by a factor of -1 on some columns. Compare loadings across resamples without fixing that first and you will see enormous instability that is entirely a sign convention. Align each vector to a reference before you measure anything — one line, `v * np.sign(v @ reference)` — and then whatever movement is left is the real thing.

One thing this episode has assumed throughout. Every matrix here has been a *correlation* matrix, so every variable arrived on the same scale and the eigenvalues were comparable by construction. Take that away — run PCA on raw columns where one is a duration in seconds and another a probability — and episode one's warning about units returns with interest: the leading component is then whichever column has the largest numbers, and the gaps are an artefact of your unit choices. The standard fix is to standardise, which is to say, to use the correlation matrix. The other standard fix is ridge, which does something to the spectrum that looks like a shrinkage and is really a shift, and which spends degrees of freedom you are not told about. Next episode.

*Exercise.* Take the block matrix from this episode and standardise nothing — instead multiply the first variable by 1,000, as a change of units would. Recompute the eigenvalues of the *covariance* matrix and find the new gaps. How many components does the scree plot now suggest keeping, and how much of that answer is about the data? The answer is at the top of episode five.""")

    post.hero = figs["hero"]
    return post


def main() -> Post:
    post = build()
    problems = post.audit()
    print(f"words: {post.word_count()}")
    print("audit:", "clean" if not problems else "")
    for p in problems:
        print("  -", p)
    return post


if __name__ == "__main__":
    main()
