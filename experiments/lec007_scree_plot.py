"""Linear Algebra 7: The Scree Plot Lies.

Eckart and Young settled which rank-k matrix is closest to yours: your own
truncated SVD, with an error equal to the tail of the spectrum. The theorem is
exact and says nothing about which k. The scree plot is what fills that gap, and
it is not a rank estimator -- it is a picture with no reference distribution in
it.

Measured, on matrices whose rank is known before the noise is added:

* At a comfortable signal every rule agrees and the elbow is right in 100 of 100
  draws. The scree plot is not wrong in the easy case, and saying so is what
  makes the rest of this credible.
* Just below that, Gavish and Donoho's optimal hard threshold reports zero
  components where three exist and the elbow finds all three. Not a defect: it
  minimises the reconstruction error, and a direction barely above the noise
  contributes more noise than signal to a reconstruction. Two different
  questions, and only one of them is "how many directions are real".
* Below about half the noise edge, nothing recovers the rank -- the information
  is not in the matrix. That is the honest limit and it belongs in the episode.
* And on a matrix with *no structure at all*, the calibrated rules return zero
  and the elbow returns an answer, spread over most of the range available. The
  elbow has no way to say "there is nothing here".

Run: `standarderror run lec007_scree_plot --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.linalg import rank as rk
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
POST_DATE = date(2026, 9, 1)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Linear Algebra for Data Science, Taught Through What Breaks"
SERIES_TAG = "Linear Algebra"

N = 400
P = 20
RANK = 3
SIGMA = 1.0

#: Episode six's exercise: the same rank, at three noise levels and two sizes.
EXERCISE = (("n = 400, σ = 1", 400, 1.0), ("n = 400, σ = 2", 400, 2.0),
            ("n = 100, σ = 1", 100, 1.0))
EXERCISE_STRENGTH = 0.5

#: And the same signal held fixed while the noise doubles.
FIXED_SIGNAL_MULTIPLE = 1.5

#: Signal strength as a multiple of the noise edge. 0.0 is a matrix with no
#: structure in it at all, which is the case the episode turns on.
MULTIPLES = (1.5, 1.0, 0.8, 0.65, 0.5, 0.35, 0.2, 0.0)
SWEEP_REPS = 60
NOISE_REPS = 300
SEED = 51


def compute() -> dict:
    out: dict = {}
    edge = rk.noise_edge(N, P, SIGMA)
    out["edge"] = edge
    out["threshold"] = rk.optimal_threshold(N, P, SIGMA)
    out["n"], out["p"], out["rank"], out["sigma"] = N, P, RANK, SIGMA

    # --- episode six's exercise --------------------------------------------
    rows = []
    for label, n, sigma in EXERCISE:
        e = rk.noise_edge(n, P, sigma)
        sv = np.full(RANK, EXERCISE_STRENGTH * e)
        rng = np.random.default_rng(SEED)
        got = []
        spectra = []
        for _ in range(SWEEP_REPS):
            Y = rk.low_rank_plus_noise(n, P, singular_values=sv, sigma=sigma,
                                       rng=rng)
            s = np.linalg.svd(Y, compute_uv=False)
            got.append(rk.elbow(s))
            spectra.append(s)
        got = np.array(got, dtype=float)
        rows.append({"label": label, "n": n, "sigma": sigma, "edge": e,
                     "spectrum": spectra[0], "first_elbow": rk.elbow(spectra[0]),
                     "elbow_median": float(np.median(got)),
                     "elbow_exact": float((got == RANK).mean()),
                     "spread": float(np.quantile(got, 0.9)
                                     - np.quantile(got, 0.1))})
    out["exercise"] = rows

    # The other half of the exercise. Above, sigma and the signal move together,
    # so the matrix at sigma = 2 is *exactly* twice the matrix at sigma = 1 and
    # the elbow, which reads a shape, cannot tell them apart -- an identity, not
    # a finding. Doubling the noise with the signal held where it was is the
    # version that changes something, and what it changes is the ratio.
    fixed = []
    for sigma in (SIGMA, 2 * SIGMA):
        sv = np.full(RANK, FIXED_SIGNAL_MULTIPLE * rk.noise_edge(N, P, SIGMA))
        e = rk.noise_edge(N, P, sigma)
        rng = np.random.default_rng(SEED + 7)
        got, gd = [], []
        tau = rk.optimal_threshold(N, P, sigma)
        for _ in range(SWEEP_REPS):
            s = np.linalg.svd(rk.low_rank_plus_noise(
                N, P, singular_values=sv, sigma=sigma, rng=rng),
                compute_uv=False)
            got.append(rk.elbow(s))
            gd.append(int((s > tau).sum()))
        got = np.array(got, dtype=float)
        fixed.append({"sigma": sigma, "ratio": float(sv[0] / e),
                      "elbow_exact": float((got == RANK).mean()),
                      "gd_exact": float((np.array(gd) == RANK).mean()),
                      "spread": float(np.quantile(got, 0.9)
                                      - np.quantile(got, 0.1))})
    out["exercise_fixed"] = fixed

    # --- Eckart-Young, checked in the form the episode states it ------------
    rng = np.random.default_rng(SEED + 1)
    Y = rk.low_rank_plus_noise(N, P, singular_values=np.full(RANK, 1.5 * edge),
                               sigma=SIGMA, rng=rng)
    out["eckart_young"] = [rk.eckart_young_check(Y, k, trials=200, rng=rng)
                           for k in (1, RANK, 8)]
    out["example_spectrum"] = np.linalg.svd(Y, compute_uv=False)

    # --- the four rules, against signal strength ---------------------------
    out["sweep"] = rk.strength_sweep(MULTIPLES, n=N, p=P, rank=RANK,
                                     sigma=SIGMA, reps=SWEEP_REPS,
                                     rng=np.random.default_rng(SEED + 2))

    # --- what the elbow says about a matrix with nothing in it -------------
    rng = np.random.default_rng(SEED + 3)
    answers = []
    for _ in range(NOISE_REPS):
        Z = rng.normal(0.0, SIGMA, (N, P))
        answers.append(rk.elbow(np.linalg.svd(Z, compute_uv=False)))
    answers = np.array(answers, dtype=float)
    # The single draw the permutation snippet prints, recomputed here so the
    # prose can name the number the reader is looking at rather than a different
    # one from the same distribution.
    demo_rng = np.random.default_rng(SEED + 11)
    out["demo_elbow"] = rk.elbow(np.linalg.svd(
        demo_rng.normal(0.0, SIGMA, (N, P)), compute_uv=False))

    out["pure_noise"] = {
        "answers": answers, "reps": NOISE_REPS,
        "distinct": int(len(np.unique(answers))),
        "modal": int(np.bincount(answers.astype(int)).argmax()),
        "at_maximum": float((answers == P - 1).mean()),
        "spread": float(np.quantile(answers, 0.9) - np.quantile(answers, 0.1)),
    }
    return out


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    out: dict = {}
    ex, sweep, pn = res["exercise"], res["sweep"], res["pure_noise"]
    edge, tau = res["edge"], res["threshold"]

    # --- f0: the same curve three times, once the edge is divided out ------
    out["f0"] = charts.lines(
        pd.DataFrame({e["label"]: e["spectrum"] / e["edge"] for e in ex},
                     index=np.arange(1, res["p"] + 1)),
        title="At half the noise edge there is no elbow to find",
        subtitle=(f"Singular values of a rank-{res['rank']} matrix plus noise, "
                  f"the signal placed at {EXERCISE_STRENGTH} times each "
                  f"matrix's own noise edge σ(√n + √p) and every spectrum "
                  f"divided by that edge."),
        xlabel="index", ylabel="singular value ÷ noise edge",
        source="Simulated; standarderror/linalg/rank.py.",
        decorate=lambda fig, ax: ax.set_xticks(
            [1] + list(range(5, res["p"] + 1, 5))),
        alt=(f"Two visible curves falling smoothly from about 1 to below 0.7 "
             f"and 0.4, with no step anywhere. The two {ex[0]['n']}-row cases "
             f"lie exactly on top of each other."),
        caption=(f"The rank-{res['rank']} structure is in all three of these and "
                 f"none of them shows it: the first three values are "
                 f"indistinguishable from the fourth. Two of the curves are the "
                 f"same line, exactly — doubling σ doubles the edge, so the "
                 f"matrix is simply scaled. Reading the elbow off these gives "
                 f"{ex[0]['first_elbow']} for the {ex[0]['n']}-row draws and "
                 f"{ex[2]['first_elbow']} for the {ex[2]['n']}-row one."),
        path=str(IMG / f"lec07-f0-rescaled.{EXT}"))[0]

    # --- f1: one spectrum, and every rule drawn on it ----------------------
    s = res["example_spectrum"]

    def marked(ax, m):
        ax.plot(np.arange(1, len(s) + 1), s, marker="o", ms=4,
                color=m.series[0], lw=1.8, label="singular values")
        ax.axhline(edge, color=m.series[1], lw=1.6, ls=(0, (5, 3)),
                   label=f"noise edge σ(√n + √p) = {edge:.1f}")
        ax.axhline(tau, color=m.series[2], lw=1.6, ls=(0, (2, 2)),
                   label=f"Gavish–Donoho threshold = {tau:.1f}")
        k = rk.elbow(s)
        ax.axvline(k + 0.5, color=m.muted, lw=1.4, ls=(0, (1, 2)),
                   label=f"the elbow, here at {k}")
        ax.set_yscale("log")
        ax.set_xticks([1] + list(range(5, len(s) + 1, 5)))

    out["f1"] = charts.diagram(
        marked,
        title="The thresholds are lines on the chart the elbow is eyeballed from",
        subtitle=(f"One rank-{res['rank']} matrix at 1.5 times the noise edge, "
                  f"with each rule drawn where it cuts."),
        xlabel="index", ylabel="singular value, log scale",
        source="Simulated; standarderror/linalg/rank.py.",
        alt=("A falling spectrum on a log axis with two horizontal threshold "
             "lines and one vertical line marking the elbow."),
        caption=("Two of these are derived from the noise model and one is a "
                 "judgement about a shape. In this easy case all three agree, "
                 "which is why the practice survives."),
        path=str(IMG / f"lec07-f1-thresholds.{EXT}"))[0]

    # --- t1: what each rule gets right, and how widely it misses ------------
    rows = []
    for r in sweep:
        rows.append([
            f"{r['multiple']:.2f}×" if r["multiple"] else "no signal",
            f"{r['elbow_exact']:.0%}",
            f"{r['elbow_spread']:.0f}",
            f"{r['noise_edge_exact']:.0%}",
            f"{r['optimal_threshold_exact']:.0%}",
            f"{r['parallel_exact']:.0%}",
        ])
    out["t1"] = charts.table_image(
        rows,
        header=["signal, as a multiple of the edge", "elbow right",
                "elbow's spread", "noise edge right", "Gavish–Donoho right",
                "permutation right"],
        title="Four rules, and the disagreement is about which loss",
        subtitle=(f"Share of {SWEEP_REPS} draws in which each rule returns the "
                  f"rank that is actually there — {res['rank']} everywhere "
                  f"except the last row, where nothing was added and the right "
                  f"answer is none. A {res['n']}×{res['p']} matrix; the "
                  f"spread is the 10th-to-90th percentile of the elbow."),
        source="Simulated; standarderror/linalg/rank.py.",
        bold_cols=(1, 2),
        alt=("Table of five columns showing each rule's accuracy falling as the "
             "signal weakens, with the elbow's spread widening."),
        caption=("Read the last row first: on a matrix with nothing in it the "
                 "three calibrated rules mostly return no components, which is "
                 "correct, and the elbow never does. Then read the third row: "
                 "Gavish–Donoho is wrong where the elbow is right, and it is "
                 "wrong on purpose."),
        path=str(IMG / f"lec07-t1-rules.{EXT}"))[0]

    # --- f2: the elbow, asked about a matrix with nothing in it ------------
    # A bar per integer answer rather than a histogram: the quantity is a count
    # of components, so binned density on a half-unit offset misrepresents it.
    counts = np.bincount(pn["answers"].astype(int), minlength=res["p"])
    share = counts / counts.sum()

    def discrete(ax, m):
        ks = np.arange(len(share))
        ax.bar(ks, share, color=m.series[0], width=0.78)
        ax.axvline(0, color=m.series[1], lw=1.6)
        ax.set_ylim(0, share.max() * 1.22)
        ax.annotate("the right answer, never returned", (0, share.max()),
                    textcoords="offset points", xytext=(7, 14), fontsize=8.5,
                    color=m.series[1], va="center")
        ax.set_xticks([0, 1] + list(range(5, res["p"], 5)))
        ax.set_xlim(-1.0, res["p"] - 0.4)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    out["f2"] = charts.diagram(
        discrete,
        title="What the elbow says about a matrix with no structure in it",
        subtitle=(f"{pn['reps']} matrices of pure Gaussian noise, "
                  f"{res['n']}×{res['p']}. The true answer is zero every time."),
        xlabel="rank the elbow reports", ylabel="share of draws",
        source="Simulated; standarderror/linalg/rank.py.",
        alt=("A ragged bar chart covering nearly the whole range of possible "
             "ranks, tallest at one, with a second peak at the far right and no "
             "bar at all at zero."),
        caption=(f"{pn['distinct']} different answers, {pn['at_maximum']:.0%} of "
                 f"them the largest value available. Zero is not among them, "
                 f"because the rule has no way to express it: it is defined as "
                 f"the position of the largest gap, and a spectrum always has "
                 f"one."),
        path=str(IMG / f"lec07-f2-pure-noise.{EXT}"))[0]

    # --- f3: accuracy against signal, all four rules ------------------------
    # The no-signal row is left out here: its accuracy is measured against a
    # different truth, so plotting it on the same axis would draw a jump that is
    # a change of question rather than a change of behaviour. It is in the table.
    with_signal = [r for r in sweep if r["multiple"] > 0]
    out["f3"] = charts.lines(
        pd.DataFrame(
            {"the elbow": [r["elbow_exact"] for r in with_signal],
             "count above the noise edge": [r["noise_edge_exact"]
                                            for r in with_signal],
             "Gavish–Donoho threshold": [r["optimal_threshold_exact"]
                                         for r in with_signal],
             "permutation reference": [r["parallel_exact"]
                                       for r in with_signal]},
            index=[r["multiple"] for r in with_signal]),
        title="Every rule fails, and they fail in different places",
        subtitle=(f"Share of {SWEEP_REPS} draws returning the true rank, against "
                  f"signal strength as a multiple of the noise edge."),
        xlabel="signal ÷ noise edge", ylabel="share of draws with the rank right",
        source="Simulated; standarderror/linalg/rank.py.",
        ylim=(-0.03, 1.03), invert_x=True,
        alt=("Four curves falling as the signal weakens and crossing each "
             "other in the middle of the range: three reach zero and one "
             "flattens out near a tenth."),
        caption=("Below about half the noise edge they collapse together, and "
                 "that is not a failure of the rules: the phase transition is "
                 "real and the information is not in the matrix. Above it, which "
                 "rule wins depends entirely on what you were trying to do. The "
                 "no-signal case is in the table rather than here, because its "
                 "accuracy is measured against a different answer."),
        path=str(IMG / f"lec07-f3-accuracy.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    pn, sweep = res["pure_noise"], res["sweep"]
    strong = sweep[0]
    ey = res["eckart_young"][1]

    def ragged(panel, m):
        counts = np.bincount(pn["answers"].astype(int), minlength=res["p"])
        panel.bar(np.arange(len(counts)), counts, color=m.ink, width=0.75)
        panel.set_xlim(-0.7, len(counts) - 0.3)

    def spectrum_with_line(panel, m):
        s = res["example_spectrum"]
        panel.plot(np.arange(len(s)), s / s[0], color=m.ink, lw=2.2)
        panel.axhline(res["edge"] / s[0], color=m.grid, lw=2.0)
        panel.set_ylim(0, 1.1)

    def two_curves(panel, m):
        x = np.linspace(0, 1, 40)
        panel.plot(x, 1 / (1 + np.exp(-9 * (x - 0.42))), color=m.ink, lw=2.4)
        panel.plot(x, 1 / (1 + np.exp(-9 * (x - 0.68))), color=m.grid, lw=2.2)
        panel.set_xlim(0, 1); panel.set_ylim(-0.05, 1.05)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=7,
        headline="The elbow has no way to say there is nothing there",
        panels=[
            (ragged, f"{pn['distinct']}", "answers, on pure noise"),
            (spectrum_with_line, f"{ey['truncation_error']:.1f}",
             "error Eckart-Young gives"),
            (two_curves, f"{strong['elbow_exact']:.0%}", "and it is right here"),
        ],
        note=("Eckart and Young settle which rank-k matrix is closest to yours "
              "and are silent about k. The scree plot fills that gap with a "
              "judgement about a shape, and on a matrix of pure noise it returns "
              "an answer — one of eighteen different ones, a fifth of them the "
              "largest available. The calibrated rules return zero, which is "
              "correct. Where the signal is strong every rule agrees, which is "
              "why the practice survives."),
        alt=("A three-panel hand-drawn strip. The first shows a ragged bar chart "
             "spread across its whole width. The second shows a falling curve "
             "crossed by a horizontal line. The third shows two S-curves offset "
             "from each other."),
        mode="light",
        path=str(IMG / f"lec07-hero.{EXT}"))[0]


# --------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["exercise"] = s.run(f"""
        import numpy as np

        # Episode six's exercise, made runnable. A rank-{RANK} matrix in {P}
        # columns, the signal at half the noise edge, and the elbow read off
        # {SWEEP_REPS} times rather than once.
        def edge(n, p, sigma):
            return sigma * (np.sqrt(n) + np.sqrt(p))

        def elbow(s):                       # "pick the elbow", made precise
            return int(np.argmax(-np.diff(s))) + 1

        def draw(n, p, rank, strength, sigma, rng):
            sv = np.full(rank, strength * edge(n, p, sigma))
            U = np.linalg.qr(rng.standard_normal((n, rank)))[0]
            V = np.linalg.qr(rng.standard_normal((p, rank)))[0]
            Y = U @ np.diag(sv) @ V.T + rng.normal(0, sigma, (n, p))
            return np.linalg.svd(Y, compute_uv=False)

        cases = [{"".join(chr(10) + " " * 12 + f"({lbl!r}, {n}, {sg})," for lbl, n, sg in EXERCISE)}
        ]

        for label, n, sigma in cases:
            rng = np.random.default_rng({SEED})
            got = np.array([elbow(draw(n, {P}, {RANK}, {EXERCISE_STRENGTH}, sigma, rng))
                            for _ in range({SWEEP_REPS})])
            exact = (got == {RANK}).mean()
            spread = np.quantile(got, 0.9) - np.quantile(got, 0.1)
            print(f"{{label:<12}} elbow = {RANK} in {{exact:.0%}} of draws, "
                  f"10th-90th spread {{spread:.1f}}")
    """, expect=["elbow ="])

    out["identity"] = s.run(f"""
        # Why two of those three lines are identical to the digit: with the
        # signal placed at a multiple of the edge, and the edge proportional to
        # sigma, the whole matrix scales. The elbow reads a shape.
        sv = np.full({RANK}, {EXERCISE_STRENGTH} * edge({N}, {P}, 1.0))
        rng = np.random.default_rng({SEED})
        U = np.linalg.qr(rng.standard_normal(({N}, {RANK})))[0]
        V = np.linalg.qr(rng.standard_normal(({P}, {RANK})))[0]
        E = rng.normal(0, 1.0, ({N}, {P}))

        Y1 = U @ np.diag(sv) @ V.T + E
        Y2 = U @ np.diag(2 * sv) @ V.T + 2 * E
        print(f"largest difference between Y2 and 2*Y1: {{abs(Y2 - 2*Y1).max():.1e}}")
    """, expect=["largest difference"])

    out["thresholds"] = s.run(f"""
        # The two calibrated cuts, in full. Neither needs your data.
        def gavish_donoho_lambda(beta):
            return np.sqrt(2 * (beta + 1) + 8 * beta
                           / ((beta + 1) + np.sqrt(beta**2 + 14*beta + 1)))

        n, p, sigma = {N}, {P}, {SIGMA}
        print(f"noise edge  sigma(sqrt n + sqrt p) = {{edge(n, p, sigma):.2f}}")
        print(f"Gavish-Donoho  lambda(p/n) sigma sqrt n = "
              f"{{gavish_donoho_lambda(p/n) * sigma * np.sqrt(n):.2f}}")
        print(f"lambda(1) = 4/sqrt(3) = {{gavish_donoho_lambda(1.0):.4f}} "
              f"vs {{4/np.sqrt(3):.4f}}")
    """, expect=["noise edge", "Gavish-Donoho"])

    out["permute"] = s.run(f"""
        # The reference distribution the scree plot is missing, built out of your
        # own matrix: permute each column on its own, which keeps every marginal
        # and destroys every relationship between columns.
        def parallel_analysis(Y, rng, reps=30, level=0.95):
            s = np.linalg.svd(Y, compute_uv=False)
            tops = [np.linalg.svd(np.column_stack([rng.permutation(c)
                                                   for c in Y.T]),
                                  compute_uv=False)[0] for _ in range(reps)]
            return int((s > np.quantile(tops, level)).sum())

        rng = np.random.default_rng({SEED} + 11)
        Z = rng.normal(0, {SIGMA}, ({N}, {P}))          # nothing in it at all
        print(f"on pure noise: elbow says {{elbow(np.linalg.svd(Z, compute_uv=False))}}, "
              f"permutation says {{parallel_analysis(Z, rng)}}")
    """, expect=["on pure noise"])

    return out


# ------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    ex, fixed = res["exercise"], res["exercise_fixed"]
    sweep, pn = res["sweep"], res["pure_noise"]
    strong = sweep[0]
    quiet = sweep[-1]

    # The spine, asserted rather than trusted.
    assert quiet["truth"] == 0, quiet
    assert quiet["elbow_over"] == 1.0, quiet
    assert quiet["optimal_threshold_exact"] == 1.0, quiet
    assert all(strong[f"{r}_exact"] == 1.0 for r in
               ("elbow", "noise_edge", "optimal_threshold", "parallel")), strong
    assert all(e["never_beaten"] for e in res["eckart_young"]), res["eckart_young"]
    assert ex[0]["elbow_exact"] == ex[1]["elbow_exact"], ex
    assert fixed[1]["gd_exact"] == 0.0 and fixed[1]["elbow_exact"] > 0.9, fixed

    post = Post(
        title=f"{SERIES_TAG} 7: The Scree Plot Lies",
        slug="linear-algebra-7-scree-plot",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=7,
        prerequisites=["linear-algebra-6-one-row"],
        date=POST_DATE,
        subtitle=("Asked about a matrix of pure noise, the elbow returns "
                  f"{pn['distinct']} different answers across "
                  f"{pn['reps']} draws and never once returns none — while three "
                  "calibrated rules get it right. And the rule that is provably "
                  "optimal reports zero components where three exist."),
        summary=("Eckart and Young settled which rank-k matrix is closest to "
                 "yours — your own truncated SVD — and said nothing about which "
                 "k. The scree plot fills that gap with a judgement about a "
                 "shape and no reference distribution, and on a matrix with no "
                 "structure in it that judgement returns a number every time. "
                 "Against three calibrated alternatives on matrices whose rank "
                 "is known in advance: every rule is right when the signal is "
                 "comfortable, which is why the practice survives; the provably "
                 "optimal threshold is wrong where the elbow is right, on "
                 "purpose, because it minimises reconstruction error and not "
                 "rank; and below about half the noise edge nothing works at "
                 "all, which is a property of the matrix rather than of the "
                 "rules. Choosing a rank is a choice of loss function, and the "
                 "elbow corresponds to none."),
        tags=["linear-algebra", "svd", "dimensionality-reduction", "pca",
              "lectures", "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every matrix here is constructed with its rank "
            "fixed before the noise is added, and every number is produced by "
            "the code shown, executed when this page was built.",
            "Machinery: `standarderror/linalg/rank.py`, tested in "
            "`tests/test_rank.py`.",
            "Where this stops: Eckart and Young, \"The approximation of one "
            "matrix by another of lower rank\", *Psychometrika* 1 (1936); "
            "Gavish and Donoho, \"The optimal hard threshold for singular "
            "values is 4/√3\", *IEEE Transactions on Information Theory* 60 "
            "(2014); Horn, \"A rationale and test for the number of factors in "
            "factor analysis\", *Psychometrika* 30 (1965); Baik, Ben Arous and "
            "Péché, \"Phase transition of the largest eigenvalue for non-null "
            "complex sample covariance matrices\", *Annals of Probability* 33 "
            "(2005), for the transition below which the signal is undetectable "
            "in principle.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": (f"{N}×{P} matrices of known rank {RANK}; "
                           f"{SWEEP_REPS} draws per point in the sweep, "
                           f"{NOISE_REPS} for the pure-noise histogram"),
            "determinism": f"one seed, {SEED}, and every draw derived from it",
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    ex, fixed = res["exercise"], res["exercise_fixed"]
    sweep, pn = res["sweep"], res["pure_noise"]
    strong, quiet = sweep[0], sweep[-1]
    gd_dies = next(r for r in sweep if r["multiple"] == 0.8)
    perm_dies = next(r for r in sweep if r["multiple"] == 0.65)
    half = next(r for r in sweep if r["multiple"] == 0.5)
    ey1, ey3, ey8 = res["eckart_young"]

    post.add(
        "Last episode's exercise",
        f"""The exercise was: take a matrix with a genuine rank-{res['rank']} structure in {res['p']} columns plus noise, pick the elbow of its spectrum, then double the noise and halve the rows and see how much the elbow moves.

Here it is, run {SWEEP_REPS} times per case instead of once, because a rule you apply by eye to one chart is still a rule and it still has a sampling distribution.

{snip['exercise'].markdown()}

Two things in that output, and the second is the more interesting one.

The elbow is right {ex[0]['elbow_exact']:.0%} of the time, and when it is wrong it is wrong by a lot: the middle 80% of its answers span {ex[0]['spread']:.0f} of the {res['p'] - 1} values available. That is the same rule, on the same generating process, reading the same chart type — and the answer is close to arbitrary.

And the σ = 1 and σ = 2 lines agree to the digit, in both columns. That is not a robustness finding. It is an identity, and seeing why is worth more than the number.""",
        figures=[figs["f0"]])

    post.add(
        "",
        f"""The signal in that experiment was placed at {EXERCISE_STRENGTH} times the noise edge, and the noise edge is proportional to σ. So doubling σ doubles the noise *and* doubles the signal, which means the second matrix is exactly twice the first one:

{snip['identity'].markdown()}

Not "close to twice" — the same matrix, scaled. And the elbow reads a shape, so it cannot possibly see a difference. Nor can any of the other rules in this episode.

Which means the version of the exercise that changes something is the one where the noise doubles and the signal stays where it was. Do that, and the ratio to the edge halves — from {fixed[0]['ratio']:.2f} to {fixed[1]['ratio']:.2f} — and here is what moves:

The elbow goes from {fixed[0]['elbow_exact']:.0%} right to {fixed[1]['elbow_exact']:.0%} right. Barely anything. The Gavish–Donoho threshold, which is the *provably optimal* rule in this episode, goes from {fixed[0]['gd_exact']:.0%} to {fixed[1]['gd_exact']:.0%}.

So the honest answer to "how much does the elbow move" is: less than you would expect, and less than the rule with a theorem behind it. Halving the rows, meanwhile, does change the shape — {res['p']} columns in {ex[2]['n']} rows is a different aspect ratio, not a rescaling — and the elbow's spread narrows from {ex[0]['spread']:.1f} to {ex[2]['spread']:.1f} while its accuracy barely moves.

That is the shape of the whole episode. The rules disagree, the ranking among them flips depending on where you are, and underneath the accuracy differences is something more basic: they are not all answering the same question.""",
        level=3)

    post.add(
        "What is actually settled",
        f"""One thing about low-rank approximation is completely settled, and it is worth stating precisely because it is so often stretched into covering the part that is not.

Eckart and Young, 1936. Among all matrices of rank *k*, the one closest to *Y* in Frobenius norm is *Y*'s own truncated SVD, and the error is the tail of the spectrum:

$$
\\min_{{\\mathrm{{rank}}(B) = k}} \\lVert Y - B \\rVert_F  =  \\sqrt{{\\sum_{{j > k}} s_j^2}}
$$

Both halves are exact. The minimiser is *U*ₖ diag(*s*ₖ) *V*ₖᵗ and the value is the square root of the sum of the discarded squared singular values — no approximation, no asymptotics, no assumption about where *Y* came from. It is why every low-rank method in use is a truncated factorisation, and it is one of the few results in this series that never breaks.

Checked in the form just stated, on a matrix built for this episode, against {ey3['trials']} competitors per *k* — each one a random *k*-dimensional column space with the least-squares-optimal coefficients for that space, so the competition is the best matrix with *that* subspace rather than a strawman:

At *k* = 1 the truncation error is {ey1['truncation_error']:.4f} and the closed form gives {ey1['predicted_error']:.4f}. At *k* = {ey3['k']}: {ey3['truncation_error']:.4f} against {ey3['predicted_error']:.4f}. At *k* = 8: {ey8['truncation_error']:.4f} against {ey8['predicted_error']:.4f}. Never beaten, {ey1['trials'] * 3} attempts, closest margin {ey1['smallest_margin']:.2f}.

And now the part the theorem is silent about. It holds for **every** *k*. It ranks nothing. Ask it which *k* to use and it answers, correctly, that larger *k* gives smaller error — all the way to *k* = min(*n*, *p*), where the error is zero and you have reconstructed the noise.""")

    post.add(
        "Four ways to choose k, and only one of them is a picture",
        f"""So the choice of *k* has to come from somewhere else. Four candidates, and the first is the one everybody actually uses.

**The elbow.** Plot the singular values and pick where the curve bends. To compare it against anything it has to be made precise enough to execute, so: the position of the largest drop, `np.argmax(-np.diff(s)) + 1`. Any other formalisation — largest ratio, largest second difference, largest curvature — has the same property, which is that it is a statement about the shape of one spectrum with nothing to compare it to.

**The noise edge.** Count the singular values above σ(√*n* + √*p*), the almost-sure limit of the largest singular value of a pure-noise matrix of that size.

**Gavish–Donoho.** Count above λ(*p*/*n*) σ √*n*, with λ(1) = 4/√3. Derived, not chosen: it minimises the asymptotic mean squared error of the reconstruction.

**A permutation reference.** Permute each column independently — which preserves every marginal distribution exactly and destroys every relationship between columns — and count the singular values above the permuted spectrum's upper quantile. A reference distribution built out of your own matrix, which is precisely what the scree plot lacks.

{snip['thresholds'].markdown()}

Two of those numbers are lines you can draw on the chart you were going to eyeball anyway.""",
        figures=[figs["f1"]])

    post.add(
        "Where they part company",
        f"""{SWEEP_REPS} draws at each of {len(sweep)} signal strengths, on {res['n']}×{res['p']} matrices whose rank is {res['rank']} before the noise is added. The last row has no signal in it at all, so the rank to recover there is zero.""",
        figures=[figs["t1"]])

    post.add(
        "",
        f"""Start where the practice lives. At {strong['multiple']:.1f} times the noise edge every rule returns {res['rank']} in every one of {SWEEP_REPS} draws, and the elbow's spread is {strong['elbow_spread']:.0f} — it does not merely get the answer right, it gets the same answer every time. **The scree plot is not wrong in the easy case.** That is why the practice survives, it is why the criticism sounds like pedantry to anyone who has only used it on clean data, and any argument that skips this is not credible.

Now walk down. At {gd_dies['multiple']:.1f}× the edge, the elbow is right {gd_dies['elbow_exact']:.0%} of the time and the naive edge count is right {gd_dies['noise_edge_exact']:.0%} of the time — and Gavish–Donoho, the rule with the theorem, is right **{gd_dies['optimal_threshold_exact']:.0%}** of the time. It reports zero components on a matrix with three.

That is not a bug, and it is the single most useful thing in this episode.""",
        figures=[figs["f3"]])

    post.add(
        "The optimal rule is optimal for something else",
        """Gavish–Donoho minimises the mean squared error of the reconstruction. Consider a direction whose true signal is just barely above the noise floor. Keeping it adds its real signal to your reconstruction, and it also adds the noise that came with it, and near the floor the second is larger than the first. So dropping a **real** direction makes the reconstruction **better**. The threshold is doing exactly what it was derived to do.

The two questions are:

*How many directions can I keep before adding them hurts my reconstruction?* — Gavish–Donoho, and it is the right answer.

*How many directions in this matrix are not noise?* — a different question, with a different answer, and the optimal threshold is systematically conservative about it.

Confusing the two is the most common mistake in this area, and it is invited by the word "optimal". If you are denoising an image, compressing a matrix, or filling in missing entries, use the threshold. If you are asking how many factors, how many communities, how many latent dimensions — how many *things are there* — the threshold is answering a question you did not ask, and a permutation reference is answering the one you did.""")

    post.add(
        "And then the floor drops out",
        f"""Keep walking down. At {perm_dies['multiple']:.2f}× the edge the permutation reference collapses to {perm_dies['parallel_exact']:.0%} while the naive edge count is still right {perm_dies['noise_edge_exact']:.0%} of the time — the ranking among the rules has now flipped twice. At {half['multiple']:.1f}× nothing works: elbow {half['elbow_exact']:.0%}, edge {half['noise_edge_exact']:.0%}, threshold {half['optimal_threshold_exact']:.0%}, permutation {half['parallel_exact']:.0%}.

This one is not the rules' fault. Below a critical signal-to-noise ratio the leading singular vector of the observed matrix carries **no** information about the true one — the phase transition of Baik, Ben Arous and Péché — and no procedure recovers what is not there. A rule that appeared to work in this regime would be reporting an artefact of its own construction.

Which is the useful form of the result: the reason to compute σ(√*n* + √*p*) is not to threshold with it, it is to find out **which regime you are in** before you interpret anything. Signal comfortably above the edge: any rule, and use the cheap one. Signal near the edge: the rules disagree and the disagreement is about your loss function, so decide which one you have. Signal below half the edge: stop, because the question is unanswerable from this matrix and the honest deliverable is a sample-size calculation rather than a rank.""")

    post.add(
        "What only the elbow cannot do",
        f"""One row of that table is left, and it is the thesis.

Give the rules a matrix of pure Gaussian noise — no structure, nothing to find, the correct answer is *none*.

{snip['permute'].markdown()}

The permutation reference says zero. So does Gavish–Donoho, in {quiet['optimal_threshold_exact']:.0%} of {SWEEP_REPS} draws. The naive edge count says zero {quiet['noise_edge_exact']:.0%} of the time — it inherits its own false-positive rate, because the edge is where the largest singular value *concentrates* rather than a ceiling it cannot cross, and a finite sample lands above it in {1 - quiet['noise_edge_exact']:.0%} of these draws.

The elbow says {res['demo_elbow']} on that draw. On another it says {pn['modal']}, which is its most common answer, and on another {int(np.max(pn['answers']))}, which is the largest value there is — {pn['distinct']} distinct answers across {pn['reps']} draws, {pn['at_maximum']:.0%} of them that maximum, with a 10th-to-90th spread of {pn['spread']:.0f} out of {res['p'] - 1}.

Zero never appears, and it cannot: the elbow is defined as the position of the largest gap in a spectrum, and every spectrum has a largest gap. **The rule has no way to express "there is nothing here."** It is not that it is inaccurate on noise. It is that "no components" is outside its range.""",
        figures=[figs["f2"]])

    post.add(
        "",
        f"""There is one more asymmetry in that table worth naming, because it runs the wrong way. Every calibrated rule degrades towards reporting **nothing** as the signal weakens: at a fifth of the edge all three of them return zero components, and the edge count's accuracy is down to {sweep[-2]['noise_edge_exact']:.0%}. Under-counting is the safe failure — you lose real structure and you know you might have.

The elbow degrades towards **over**-counting, and it gets worse as the signal gets weaker. It returns more than {res['rank']} components in {half['elbow_over']:.0%} of draws at half the edge, {sweep[-2]['elbow_over']:.0%} at a fifth of it, and {quiet['elbow_over']:.0%} of the time on a matrix with nothing in it. The one rule with no reference distribution is also the one whose errors point towards finding structure, and it is the one applied by eye, once, by someone who wants there to be structure.""",
        level=3)

    post.add(
        "What to take away",
        """**Compute σ(√n + √p) before you look at the scree plot.** One line, no data needed beyond the shape and a noise estimate, and it tells you which of the three regimes above you are in. That is worth more than any single rank estimate.

**Draw your thresholds on the plot instead of eyeballing it.** If you are going to look at a spectrum, put the two lines on it. They cost nothing and they turn a judgement into a comparison.

**Pick your rule from your loss.** Reconstructing, denoising or compressing: Gavish–Donoho, and expect it to discard weak real directions on purpose. Counting how many things are there: permute your own columns and use that as the reference. There is no rule that is right for both, because they are different questions.

**Never let the elbow tell you the answer is not zero.** It cannot say zero. If the possibility that your matrix has no low-rank structure matters to your conclusion — and in factor analysis, community detection and "how many regimes are there" it is usually the whole question — the elbow is the one tool that structurally cannot deliver it.

**Report the rank as a decision, not a measurement.** "We kept 3 components" invites the reader to assume 3 was discovered. "We kept the 3 components above a permutation reference at the 95th percentile; the fourth was below it" is the same sentence with its loss function attached.

Which closes the columns half of this series. Seven episodes on *X*: what conditioning is, what it does to a fit, what a penalty buys, what one row can do, and now how many directions are in there. Every one of them has been a linear model — a fit obtained by solving a linear system once, in closed form, where the answer either exists or the matrix tells you why not.

The last episode gives that up. Logistic regression has no closed form, so the coefficients are found by iterating, and the iteration is Newton's method on the log-likelihood dressed up as a weighted least squares problem — the same *X*ᵗ*WX* you already know, with the weights recomputed each pass. Which means every conditioning problem in this series is still there, plus a new one that is worse: the fit can fail to exist at all, the software will not tell you, and the coefficient it reports will look like the most significant result in your table. Next episode, and the last.

*Exercise.* Take any dataset you have used PCA on and kept some number of components from. Compute σ(√*n* + √*p*) for it, using the smallest singular value as a rough noise scale, and see where your chosen cut sits relative to that line. Then permute each column independently, twenty times, and record the largest singular value each time. How many of your components are above the 95th percentile of those twenty numbers? If the answer differs from the number you kept, the difference is your loss function, and it is worth being able to name it.""")

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
