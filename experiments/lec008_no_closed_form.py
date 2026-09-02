"""Linear Algebra 8: When There Is No Closed Form.

The last episode of the series, and the first fit in it without a formula.

Logistic regression's likelihood equations are non-linear in the coefficients, so
the fit is iterated. What is iterated is the machinery of the previous seven
episodes: Newton's method on the log-likelihood is *exactly* a weighted least
squares problem, solved again each pass with weights `p(1-p)` from the current
fit. So every conditioning problem in this series is still here -- and one more,
which is that `W` is produced by the fit rather than brought by the data, and as
the fit sharpens `W` goes to zero.

Measured:

* On an ordinary design it converges in six passes, and the error squares each
  pass. Quadratic convergence, which is what "this is Newton's method" means.
* When a hyperplane separates the classes the maximum does not exist. The fit
  does not fail; it returns a number, and the number is a function of the
  iteration limit. The coefficient grows by *exactly one per iteration* and
  `cond(X'WX)` by *exactly a factor of e*, until the library's weight floor
  pins them.
* The reported standard error under saturation is `1/sqrt(k * floor)`: a
  function of a category size and a constant somebody else chose. It contains
  no information from the data.
* And the case that reaches production is not complete separation, which a
  linear program detects before you fit. It is a dummy true for 4% of rows with
  outcome 1 on all of them: the other coefficients are unaffected to three
  decimal places, the table looks healthy, and one coefficient has no estimate.

Run: `standarderror run lec008_no_closed_form --publish`
"""

from __future__ import annotations

import math
import os
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.linalg import irls as ir
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

#: Episode seven's exercise, worked on a matrix whose rank is known: four
#: factors straddling the noise edge, which is where the rules disagree.
EX_N, EX_P = 300, 15
EX_STRENGTHS = (2.2, 1.4, 0.95, 0.72)
EX_SEED = 88

#: The ordinary logistic design, for the case that is supposed to work.
GOOD_N = 4000
GOOD_BETA = (-0.5, 1.0, -0.8)
GOOD_SEED = 7

#: Newton against a fixed-step gradient ascent, on a smaller well-posed problem.
CONV_N = 600
CONV_STEPS = 12
CONV_LR = 0.1
CONV_SEED = 5

#: Complete separation: one column separates the classes outright.
SEP_N = 200
SEP_SEED = 3

#: Partial separation: a dummy true for a small share of rows, outcome 1 on all
#: of them. Nothing about this design is unusual.
QUIET_N = 1000
QUIET_SHARE = 0.04
QUIET_SEED = 11
STOPS = (5, 10, 25, 50, 100)
LONG_RUN = 300
FLOORS = (1e-8, 1e-10, 1e-12)
ALPHAS = (0.0, 0.1, 1.0, 10.0)

SEED = 61


def compute() -> dict:
    out: dict = {}

    # --- episode seven's exercise, on a matrix whose rank is known ----------
    rng = np.random.default_rng(EX_SEED)
    edge0 = rk.noise_edge(EX_N, EX_P, 1.0)
    Y = rk.low_rank_plus_noise(
        EX_N, EX_P, singular_values=np.array(EX_STRENGTHS) * edge0, sigma=1.0,
        rng=rng)
    s = np.linalg.svd(Y, compute_uv=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    # The noise scale from the smallest singular value, as the exercise said:
    # the lower edge of the noise bulk is sigma(sqrt n - sqrt p).
    sigma_hat = float(s[-1] / (np.sqrt(EX_N) - np.sqrt(EX_P)))
    edge = rk.noise_edge(EX_N, EX_P, sigma_hat)
    tau = rk.optimal_threshold(EX_N, EX_P, sigma_hat)
    out["exercise"] = {
        "n": EX_N, "p": EX_P, "truth": len(EX_STRENGTHS), "spectrum": s,
        "cumulative": cum, "sigma_hat": sigma_hat, "edge": float(edge),
        "threshold": float(tau),
        "elbow": rk.elbow(s),
        "above_edge": int((s > edge).sum()),
        "above_threshold": int((s > tau).sum()),
        "permutation": rk.parallel_analysis(Y, rng=np.random.default_rng(2),
                                            reps=40),
        "eighty": int(np.searchsorted(cum, 0.80) + 1),
        "ninety": int(np.searchsorted(cum, 0.90) + 1),
    }

    # --- the ordinary case --------------------------------------------------
    Xg, yg = ir.well_posed_design(GOOD_N, rng=np.random.default_rng(GOOD_SEED),
                                 beta=GOOD_BETA)
    good = ir.irls(Xg, yg)
    out["good"] = {
        "fit": good, "truth": GOOD_BETA, "n": GOOD_N,
        "rate": float(yg.mean()),
        "separated": ir.separation_lp(Xg, yg)["separated"],
    }

    # --- it is Newton's method ----------------------------------------------
    Xc, yc = ir.well_posed_design(CONV_N, rng=np.random.default_rng(CONV_SEED))
    out["convergence"] = ir.newton_vs_gradient(Xc, yc, steps=CONV_STEPS,
                                               lr=CONV_LR)

    # --- complete separation ------------------------------------------------
    Xs, ys = ir.separable_design(SEP_N, rng=np.random.default_rng(SEP_SEED))
    out["complete"] = {
        "lp": ir.separation_lp(Xs, ys),
        "sweep": ir.iteration_sweep(Xs, ys, STOPS),
        "n": SEP_N,
    }

    # --- partial separation, which is the one that ships --------------------
    Xq, yq, k = ir.quiet_separation_design(
        QUIET_N, rng=np.random.default_rng(QUIET_SEED), share=QUIET_SHARE)
    names = ["intercept", "x1", "x2", "rare category"]
    fits = {m: ir.irls(Xq, yq, max_iter=m) for m in STOPS}
    long = ir.irls(Xq, yq, max_iter=LONG_RUN)
    out["quiet"] = {
        "n": QUIET_N, "k": k, "names": names, "rate": float(yq.mean()),
        "lp_says": ir.separation_lp(Xq, yq)["separated"],
        "empty_cells": ir.empty_cell_check(Xq, yq, names=names),
        "fits": fits, "long": long,
        "increment": float(np.median(np.diff(long.path)[8:20])),
        # The pass at which the weight floor takes over: the last one whose step
        # is still the constant 1. After it the growth is logarithmic, and
        # reading that bend as convergence is the mistake this episode is about.
        "straight_until": int(np.max(np.nonzero(
            np.abs(np.diff(long.path) - 1.0) < 1e-3)[0]) + 1),
        "condition_ratio": float(np.median(long.weighted_condition[6:15]
                                           / long.weighted_condition[5:14])),
        "tail_increment": float(np.diff(long.path)[-1]),
        "floors": ir.floor_determined_error(Xq, yq, index=3, floors=FLOORS),
        "ridge": ir.ridge_sweep(Xq, yq, ALPHAS, index=3, max_iter=100),
    }
    return out


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    out: dict = {}
    ex, conv, quiet = res["exercise"], res["convergence"], res["quiet"]
    long = quiet["long"]

    # --- f0: last episode's exercise, worked --------------------------------
    def marked(ax, m):
        s = ex["spectrum"]
        ax.plot(np.arange(1, len(s) + 1), s, marker="o", ms=4,
                color=m.series[0], lw=1.8, label="singular values")
        ax.axhline(ex["edge"], color=m.series[1], lw=1.6, ls=(0, (5, 3)),
                   label=f"noise edge, σ̂ = {ex['sigma_hat']:.2f}")
        ax.axhline(ex["threshold"], color=m.series[2], lw=1.6, ls=(0, (2, 2)),
                   label="Gavish–Donoho threshold")
        ax.axvline(ex["truth"] + 0.5, color=m.ink, lw=1.8,
                   label=f"the truth, {ex['truth']}")
        ax.set_xticks([1] + list(range(5, len(s) + 1, 5)))
        # Each rule's verdict, marked where it cuts. Drawn as short ticks off the
        # bottom axis rather than as full lines, because five full-height rules
        # on one chart is a chart nobody reads.
        lo, hi = ax.get_ylim()
        # The row for each label is set by hand rather than cycled: three of the
        # five verdicts land within two positions of each other, so a cycle puts
        # two of them on the same line.
        for k, label, row in ((ex["elbow"], "elbow", 0),
                              (ex["above_threshold"], "G–D", 1),
                              (ex["permutation"], "permutation", 2),
                              (ex["above_edge"], "noise edge", 3),
                              (ex["eighty"], "80% of variance", 0)):
            ax.annotate(f"{label}: {k}", (k + 0.5, lo),
                        textcoords="offset points", xytext=(3, 6 + 11 * row),
                        fontsize=8.0, color=m.ink_secondary)
            ax.plot([k + 0.5, k + 0.5], [lo, lo + (hi - lo) * 0.05],
                    color=m.ink_secondary, lw=1.2)
        ax.set_ylim(lo, hi)

    out["f0"] = charts.diagram(
        marked,
        title="Five rules, five answers, one of them right",
        subtitle=(f"A {ex['n']}×{ex['p']} matrix built with {ex['truth']} "
                  f"factors, two of them close to the noise edge. Last "
                  f"episode's exercise, on a matrix whose answer is known."),
        xlabel="index", ylabel="singular value",
        source="Simulated; standarderror/linalg/rank.py.",
        alt=("A falling spectrum with two horizontal threshold lines, one solid "
             "vertical line, and five short ticks along the bottom axis "
             "labelled with each rule's verdict."),
        caption=(f"The elbow says {ex['elbow']}, Gavish–Donoho "
                 f"{ex['above_threshold']}, a permutation reference "
                 f"{ex['permutation']}, the noise edge {ex['above_edge']}, and "
                 f"the convention everybody actually uses — keep 80% of the "
                 f"variance — says {ex['eighty']}. The truth is "
                 f"{ex['truth']}. Note which one is off by a factor of two, and "
                 f"which one is not on this chart at all in most papers."),
        path=str(IMG / f"lec08-f0-exercise.{EXT}"))[0]

    # --- f1: quadratic convergence -----------------------------------------
    # After the fifth pass the distance is exactly zero, which a log axis cannot
    # draw. Rather than floor it at some small number -- a drawn value that is
    # not the measured one -- the Newton line stops where the measurements do,
    # and the subtitle says why.
    newton = np.asarray(conv["newton"], dtype=float)
    zeros = np.nonzero(newton == 0.0)[0]
    stop = int(zeros[0]) if len(zeros) else len(newton)
    drawn = np.full(len(newton), np.nan)
    drawn[:stop] = newton[:stop]
    out["f1"] = charts.lines(
        pd.DataFrame({"Newton (IRLS)": drawn,
                      f"gradient ascent, step {CONV_LR}": conv["gradient"]},
                     index=np.arange(1, len(newton) + 1)),
        title="This is Newton's method, and it converges like one",
        subtitle=(f"Distance to the maximum likelihood estimate, "
                  f"{CONV_N} rows, {len(GOOD_BETA)} coefficients. Log scale, so "
                  f"the Newton line stops at pass {stop}: from pass {stop + 1} "
                  f"the distance is exactly zero and a log axis has nowhere to "
                  f"put it."),
        xlabel="iteration", ylabel="distance to the MLE",
        logy=True,
        source="Simulated; standarderror/linalg/irls.py.",
        alt=("Two curves on a log axis: one plunging almost vertically to the "
             "bottom of the chart within five steps and then running flat along "
             "it, one nearly flat at the top."),
        caption=("Each Newton error is roughly the square of the last, which is "
                 "what makes the reweighting worth doing. The gradient line is "
                 "a fixed step size and therefore a weak opponent — it is here "
                 "to show what having the Hessian buys, not to argue against "
                 "first-order methods."),
        path=str(IMG / f"lec08-f1-convergence.{EXT}"))[0]

    # --- t1: the coefficient table, at five stopping points -----------------
    rows = []
    for m in STOPS:
        f = quiet["fits"][m]
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.abs(f.beta) / f.standard_errors
        rows.append([str(m),
                     f"{f.beta[1]:+.3f}", f"{f.beta[2]:+.3f}",
                     f"{f.beta[3]:+.2f}", f"{np.exp(f.beta[3]):.1e}",
                     f"{f.standard_errors[3]:,.3g}",
                     f"{z[3]:.2f}" if z[3] >= 0.01 else "< 0.01"])
    out["t1"] = charts.table_image(
        rows,
        header=["iterations allowed", "x1", "x2", "rare category",
                "its odds ratio", "its std. error", "its z"],
        title="The same data, the same code, five different answers",
        subtitle=(f"Logistic regression on {quiet['n']} rows. A dummy true for "
                  f"{quiet['k']} of them, with outcome 1 on every one. Nothing "
                  f"else about the design is unusual."),
        source="Simulated; standarderror/linalg/irls.py.",
        bold_cols=(3, 6),
        alt=("Table of seven columns. The first two coefficient columns are "
             "identical down every row; the rare-category column grows from "
             "about 7 to about 28 and its z falls from 2 to nearly zero."),
        caption=("x1 and x2 do not move at all — to three decimal places, "
                 "across a twentyfold change in the iteration limit. The rare "
                 "category's coefficient never stops growing, and its z "
                 "statistic crosses 2 on the way down. At five iterations it "
                 "is a significant finding. At twenty-five it is nothing. "
                 "There is no fact of the matter."),
        path=str(IMG / f"lec08-t1-stops.{EXT}"))[0]

    # --- f2: what each coefficient does, iteration by iteration -------------
    beta_paths = []
    Xq, yq, _ = ir.quiet_separation_design(
        QUIET_N, rng=np.random.default_rng(QUIET_SEED), share=QUIET_SHARE)
    for m in range(1, 41):
        beta_paths.append(ir.irls(Xq, yq, max_iter=m).beta)
    beta_paths = np.asarray(beta_paths)
    out["f2"] = charts.lines(
        pd.DataFrame({name: beta_paths[:, j]
                      for j, name in enumerate(quiet["names"])},
                     index=np.arange(1, len(beta_paths) + 1)),
        title="Three coefficients converge and one leaves",
        subtitle="Every coefficient of the same fit, against iteration count.",
        xlabel="iteration", ylabel="coefficient",
        source="Simulated; standarderror/linalg/irls.py.",
        alt=("Three nearly flat lines close to zero and one rising steeply, "
             "straight for two thirds of the range and then bending towards "
             "the horizontal without flattening."),
        caption=(f"The straight part is the tell, and it is straight for a "
                 f"reason: Newton's step on a likelihood with no maximum "
                 f"settles at a constant, and here that constant is "
                 f"{quiet['increment']:.3f} per pass, which makes the "
                 f"coefficient a restatement of the iteration count. It stays "
                 f"straight to pass "
                 f"{quiet['straight_until']}, where the library's weight floor "
                 f"takes over and the growth turns logarithmic. That bend is "
                 f"what gets read as convergence. It is not: at pass "
                 f"{LONG_RUN} the step is still "
                 f"{quiet['tail_increment']:.4f} and still positive."),
        path=str(IMG / f"lec08-f2-paths.{EXT}"))[0]

    # --- f3: the weighted design going singular, at a rate of e -------------
    # The first 60 passes. Beyond them the line is flat and the chart is width
    # spent on the weight floor.
    c = long.weighted_condition[:60]
    out["f3"] = charts.lines(
        pd.DataFrame({"cond(XᵗWX)": c}, index=np.arange(1, len(c) + 1)),
        title="The design was fine. The fit made it singular",
        subtitle=(f"Condition number of the weighted cross-product at each of "
                  f"the first {len(c)} passes, same fit. The unweighted design "
                  f"has a condition number of {np.linalg.cond(Xq):.1f}. "
                  f"Log scale."),
        xlabel="iteration", ylabel="cond(XᵗWX)", logy=True,
        source="Simulated; standarderror/linalg/irls.py.",
        alt=("A straight rising line on a log axis that flattens abruptly and "
             "stays flat."),
        caption=(f"A straight line on a log axis is geometric growth, and the "
                 f"ratio is {quiet['condition_ratio']:.3f} per pass — e, which "
                 f"follows from the coefficient rising by 1 each pass and a "
                 f"saturated row's weight going like exp(−|η|). The flat part "
                 f"is not convergence; the paragraph below says what it is."),
        path=str(IMG / f"lec08-f3-condition.{EXT}"))[0]

    # --- t2: the penalty, and what it costs ---------------------------------
    rows = []
    for r in quiet["ridge"]:
        rows.append([f"{r['alpha']:.1f}", "yes" if r["converged"] else "no",
                     str(r["iterations"]), f"{r['coefficient']:+.2f}",
                     f"{r['standard_error']:,.2f}",
                     f"{r['log_likelihood']:,.1f}"])
    out["t2"] = charts.table_image(
        rows,
        header=["penalty α", "converged", "iterations", "rare category",
                "its std. error", "log-likelihood"],
        title="Episode five's penalty, on ill-conditioning the fit created",
        subtitle=("The same separated design, refit with a ridge penalty on the "
                  "weighted normal equations."),
        source="Simulated; standarderror/linalg/irls.py.",
        bold_cols=(1, 4),
        alt=("Table of six columns: the first row does not converge and has a "
             "standard error in the thousands; the rest converge with standard "
             "errors under one."),
        caption=("Any penalty at all makes the maximum exist, so the fit "
                 "converges in single digits and the standard error becomes a "
                 "number about the data again. What it costs is in the last "
                 "column."),
        path=str(IMG / f"lec08-t2-ridge.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    quiet = res["quiet"]
    f5 = quiet["fits"][5]
    z5 = abs(f5.beta[3]) / f5.standard_errors[3]
    long = quiet["long"]

    def three_flat_one_gone(panel, m):
        x = np.arange(1, 41)
        for lvl in (0.5, -0.3, -0.55):
            panel.plot(x, np.full_like(x, lvl, dtype=float), color=m.grid,
                       lw=2.2)
        panel.plot(x, 0.12 * x, color=m.ink, lw=2.6)
        panel.set_ylim(-1.0, 3.0)           # the rising line leaves the frame

    def weights_collapsing(panel, m):
        x = np.linspace(0, 1, 60)
        panel.plot(x, np.exp(-7.5 * x) + 0.01, color=m.ink, lw=2.6)
        panel.axhline(0.01, color=m.grid, lw=2.2)
        panel.set_ylim(-0.05, 1.1)

    def two_zs(panel, m):
        panel.bar([0, 1], [z5, 0.07], color=[m.ink, m.ink], width=0.3)
        panel.axhline(1.96, color=m.grid, lw=2.4)
        panel.set_xlim(-0.7, 1.7)
        panel.set_ylim(0, z5 * 1.25)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=8,
        headline="The coefficient was the iteration count",
        panels=[
            (three_flat_one_gone, f"{long.path[-1]:.0f}", "and still climbing"),
            (weights_collapsing, f"{long.min_weight:.0e}",
             "smallest weight left"),
            (two_zs, f"{z5:.2f}", "z, if you stop at 5"),
        ],
        note=("Logistic regression has no closed form, so it is iterated — and "
              "the iteration is the weighted least squares of the previous seven "
              "episodes, with weights p(1-p) the fit produces itself. When a "
              "dummy true for 4% of rows has outcome 1 on all of them, the "
              "maximum does not exist. Those weights go to zero, the other "
              "coefficients are unaffected to three decimals, nothing raises, "
              "and the coefficient with no estimate is significant or not "
              "depending on the iteration limit."),
        alt=("A three-panel hand-drawn strip. The first shows three flat lines "
             "and one rising off the top of its frame. The second shows a curve "
             "decaying onto a horizontal floor. The third shows a tall bar and "
             "a very short one on either side of a horizontal reference line."),
        mode="light",
        path=str(IMG / f"lec08-hero.{EXT}"))[0]


# --------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["irls"] = s.run(f"""
        import numpy as np

        def sigmoid(eta):
            # Split at zero so neither branch overflows. It matters below.
            out = np.empty_like(eta)
            pos = eta >= 0
            out[pos] = 1 / (1 + np.exp(-eta[pos]))
            e = np.exp(eta[~pos])
            out[~pos] = e / (1 + e)
            return out

        def irls(X, y, max_iter=50, tol=1e-10, floor=1e-10):
            beta = np.zeros(X.shape[1])
            for it in range(1, max_iter + 1):
                eta = X @ beta
                mu = sigmoid(eta)
                w = np.maximum(mu * (1 - mu), floor)     # the weights
                z = eta + (y - mu) / w                   # the working response
                sw = np.sqrt(w)
                step = np.linalg.lstsq(X * sw[:, None], sw * z, rcond=None)[0]
                if np.max(np.abs(step - beta)) < tol:
                    return step, it, True
                beta = step
            return beta, max_iter, False

        # An ordinary design: {GOOD_N} rows, an intercept and two columns.
        rng = np.random.default_rng({GOOD_SEED})
        n = {GOOD_N}
        X = np.column_stack([np.ones(n), rng.standard_normal(n),
                             rng.standard_normal(n)])
        truth = np.array({list(GOOD_BETA)})
        y = (rng.random(n) < sigmoid(X @ truth)).astype(float)

        beta, iters, ok = irls(X, y)
        print(f"converged: {{ok}} in {{iters}} passes")
        print("true  ", np.round(truth, 3))
        print("fitted", np.round(beta, 3))
    """, expect=["converged:", "fitted"])

    out["quiet"] = s.run(f"""
        # Now change one thing. A dummy true for {res['quiet']['k']} rows out of
        # {QUIET_N}, and every row where it is true has outcome 1. Nothing else
        # about this design is unusual, and nothing about it is rare in practice.
        rng = np.random.default_rng({QUIET_SEED})
        n = {QUIET_N}
        k = max(round(n * {QUIET_SHARE}), 2)
        d = np.zeros(n); d[rng.choice(n, k, replace=False)] = 1.0
        x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
        y = (rng.random(n) < sigmoid(-0.3 + 0.8 * x1 - 0.5 * x2)).astype(float)
        y[d > 0.5] = 1.0
        X = np.column_stack([np.ones(n), x1, x2, d])

        for m in (5, 25, 100):
            beta, iters, ok = irls(X, y, max_iter=m)
            print(f"max_iter={{m:>4}}  converged={{str(ok):<5}}  "
                  f"x1={{beta[1]:+.3f}}  x2={{beta[2]:+.3f}}  "
                  f"dummy={{beta[3]:+.2f}}")
    """, expect=["max_iter="])

    out["detect"] = s.run("""
        # The check that finds it, and it is not a convergence check. A 2x2 table
        # with an empty cell: one level of a binary column with only one outcome.
        for j in range(X.shape[1]):
            col = X[:, j]
            levels = np.unique(col)
            if len(levels) != 2:
                continue
            for lv in levels:
                rows = col == lv
                if rows.any() and len(np.unique(y[rows])) == 1:
                    print(f"column {j}: {int(rows.sum())} rows at value {lv:g}, "
                          f"all with y = {y[rows][0]:g} "
                          f"-> this coefficient has no maximum")
    """, expect=["no maximum"])

    out["floor"] = s.run(f"""
        # And the standard error the table reports for it. Every weight in that
        # category is pinned at the library's floor, so X'WX contributes
        # k * floor in that direction and the inverse gives 1/sqrt(k * floor).
        for floor in {list(FLOORS)!r}:
            beta, _, _ = irls(X, y, max_iter=100, floor=floor)
            mu = sigmoid(X @ beta)
            w = np.maximum(mu * (1 - mu), floor)
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * w[:, None]))))
            print(f"floor={{floor:.0e}}  coef={{beta[3]:6.2f}}  "
                  f"reported s.e.={{se[3]:11,.2f}}  "
                  f"1/sqrt(k*floor)={{1/np.sqrt(k*floor):11,.2f}}")
    """, expect=["reported s.e."])

    out["default"] = s.run("""
        # The number that decided the p-value, read out of the library rather
        # than remembered. scikit-learn is a dependency of this project, so this
        # runs when the page is built.
        import inspect
        import sklearn
        from sklearn.linear_model import LogisticRegression

        default = inspect.signature(LogisticRegression).parameters["max_iter"]
        print(f"scikit-learn {sklearn.__version__}: "
              f"LogisticRegression(max_iter={default.default})")
    """, expect=["max_iter="])

    return out


# ------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    good, quiet = res["good"], res["quiet"]
    f5, f100 = quiet["fits"][5], quiet["fits"][100]

    # The spine, asserted rather than trusted.
    assert good["fit"].converged and good["fit"].iterations <= 8, good["fit"]
    assert not good["separated"], good
    assert not quiet["lp_says"], "the LP must NOT fire on the quiet case"
    assert len(quiet["empty_cells"]) == 1, quiet["empty_cells"]
    assert abs(quiet["increment"] - 1.0) < 2e-3, quiet["increment"]
    assert abs(quiet["condition_ratio"] - np.e) < 2e-3, quiet["condition_ratio"]
    assert quiet["tail_increment"] > 0 and not quiet["long"].converged
    assert np.allclose(f5.beta[:3], f100.beta[:3], atol=1e-3), (f5.beta, f100.beta)
    for r in quiet["floors"]:
        assert abs(r["standard_error"] / r["closed_form"] - 1) < 1e-6, r
    assert quiet["ridge"][0]["converged"] is False
    assert all(r["converged"] for r in quiet["ridge"][1:])

    post = Post(
        title=f"{SERIES_TAG} 8: When There Is No Closed Form",
        slug="linear-algebra-8-no-closed-form",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=8,
        prerequisites=["linear-algebra-7-scree-plot"],
        date=POST_DATE,
        subtitle=("A logistic regression where one coefficient's value is the "
                  "iteration limit, its standard error is a constant the library "
                  "author chose, and its p-value crosses 0.05 on the way down — "
                  "while every other coefficient in the table is correct to "
                  "three decimal places."),
        summary=("Logistic regression has no closed form, so it is iterated, and "
                 "the iteration is exactly the weighted least squares of the "
                 "previous seven episodes: Newton's method on the log-likelihood "
                 "is a reweighted normal-equations solve, and it converges "
                 "quadratically when there is something to converge to. When a "
                 "category is perfectly predictive there is not. The maximum "
                 "likelihood estimate does not exist, no software raises, and "
                 "the number returned is the iteration count — measured here "
                 "growing by exactly 1 per pass while the weighted design's "
                 "condition number grows by exactly a factor of e. The "
                 "textbook test for this detects complete separation, which is "
                 "not the case that ships; the case that ships is a rare dummy "
                 "with one outcome, and it is caught by looking for an empty "
                 "cell rather than by watching the optimiser."),
        tags=["linear-algebra", "logistic-regression", "glm", "separation",
              "lectures", "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every design here is constructed in the episode "
            "and every number is produced by the code shown, executed when this "
            "page was built.",
            "Machinery: `standarderror/linalg/irls.py`, tested in "
            "`tests/test_irls.py`.",
            "Where this stops: Nelder and Wedderburn, \"Generalized linear "
            "models\", *JRSS A* 135 (1972), for the IRLS formulation; Albert and "
            "Anderson, \"On the existence of maximum likelihood estimates in "
            "logistic regression models\", *Biometrika* 71 (1984), for the "
            "existence conditions; Firth, \"Bias reduction of maximum "
            "likelihood estimates\", *Biometrika* 80 (1993), and Heinze and "
            "Schemper, \"A solution to the problem of separation in logistic "
            "regression\", *Statistics in Medicine* 21 (2002), for what to do "
            "about it.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": (f"{GOOD_N} rows for the well-posed fit, {QUIET_N} for "
                           f"the separated one, {SEP_N} for the completely "
                           f"separated one; no observation is placed by hand"),
            "determinism": "one seed per design, each stated in the code shown",
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    ex, good, quiet = res["exercise"], res["good"], res["quiet"]
    # Counted from the block the reader is looking at, so "N lines" cannot drift
    # away from the code if the code is edited.
    body = snip["irls"].code.split("def irls(")[1].split("\n\n")[0]
    fitter_lines = len([ln for ln in body.split("\n")[1:] if ln.strip()])
    detect_lines = len([ln for ln in snip["detect"].code.split("\n")
                        if ln.strip() and not ln.strip().startswith("#")])
    f5, f10, f100 = quiet["fits"][5], quiet["fits"][10], quiet["fits"][100]
    z5 = abs(f5.beta[3]) / f5.standard_errors[3]
    z10 = abs(f10.beta[3]) / f10.standard_errors[3]
    z100 = abs(f100.beta[3]) / f100.standard_errors[3]
    complete = res["complete"]["sweep"]
    cell = quiet["empty_cells"][0]
    ridge = quiet["ridge"]
    newton = res["convergence"]["newton"]
    grad = res["convergence"]["gradient"]
    # Two-sided normal p-value for that z, computed rather than recalled.
    p5 = math.erfc(z5 / math.sqrt(2.0))

    post.add(
        "Last episode's exercise",
        f"""The exercise was to take a matrix you had run PCA on, compute σ(√*n* + √*p*) using the smallest singular value as a noise scale, build a permutation reference, and compare both against the number of components you had kept.

Your matrix will give its own answer. Here is the same procedure on one where the answer is known before the noise goes in: {ex['n']}×{ex['p']}, {ex['truth']} factors, two of them placed close to the noise edge — which is where episode seven said the rules stop agreeing.

The noise scale first, because the exercise glossed it. The smallest singular value of a noise matrix concentrates at σ(√*n* − √*p*), the *lower* edge of the bulk, so dividing by that recovers σ. Here it gives σ̂ = {ex['sigma_hat']:.3f}, against a true σ of 1. Close enough to use.

Then the five verdicts.""",
        figures=[figs["f0"]])

    post.add(
        "",
        f"""The elbow says {ex['elbow']}. Gavish–Donoho says {ex['above_threshold']}, and so does the permutation reference, which is episode seven's phase-transition result arriving on schedule: two of the four factors are below the edge and are not recoverable. Counting above the noise edge says {ex['above_edge']}, which is correct, and is correct partly by luck — that rule over-counts, and here the over-count lands on the truth.

And the convention that gets used more than all four of these combined — keep enough components to explain 80% of the variance — says {ex['eighty']}. More than double the truth. At 90% it says {ex['ninety']}, which is {ex['ninety']}/{ex['p']} of the columns you started with.

That rule is worth one more sentence, because it is the default in most software and most papers. It has no noise model in it at all. It asks how much of the sum of squared singular values you have captured, and in a {ex['n']}×{ex['p']} matrix the noise contributes a large and predictable share of that sum — so "80% of the variance" is mostly a statement about the shape of your matrix, and it will hand you a large number of components whether or not any of them mean anything.

Which is a good place to end the columns half of this series, and start the last episode.""",
        level=3)

    post.add(
        "Every fit so far had a formula. This one does not",
        f"""Seven episodes, every one of them about *X*ᵗ*X* — its condition number, its inverse, its spectrum, its diagonal, its rank. And every fit obtained by solving a linear system once. The answer existed, or the matrix told you why not.

Logistic regression breaks that. The log-likelihood is

$$
\\ell(\\beta)  =  \\sum_i \\left[ y_i  x_i^{{\\top}}\\beta - \\log\\left(1 + e^{{x_i^{{\\top}}\\beta}}\\right) \\right]
$$

and setting its gradient to zero gives *X*ᵗ(*y* − *p*) = 0 where *p* depends on β. Non-linear. No formula.

So it is iterated. And the iteration is not a new piece of machinery — it is the old one, applied repeatedly. Newton's method needs the Hessian, which here is −*X*ᵗ*WX* with *W* = diag(*p*(1 − *p*)), and one Newton step written out is

$$
\\beta_{{\\text{{new}}}}  =  \\left(X^{{\\top}} W X\\right)^{{-1}} X^{{\\top}} W z, \\qquad z  =  X\\beta + W^{{-1}}(y - p)
$$

which is a weighted least squares fit of a *working response* `z` on the same *X*. That is the whole algorithm. Iteratively reweighted least squares, and it is Newton's method wearing episode one's clothes.

{snip['irls'].markdown()}

{fitter_lines} lines, {good['fit'].iterations} passes, and the coefficients come back. Note what is not in there: no learning rate, no schedule, no tolerance to tune beyond a machine-epsilon stopping rule. Newton's method does not need any of that, and the reason is the rate.""",
        figures=[figs["f1"]])

    post.add(
        "",
        f"""Quadratic convergence, on a chart: {newton[0]:.3f}, then {newton[1]:.4f}, then {newton[2]:.2e}, then {newton[3]:.1e}, then {newton[4]:.1e} — and at the pass after that, exactly zero, because the step returns the same coefficients bit for bit. Each of those is about the square of the one before ({newton[3]:.1e} squared is {newton[3] ** 2:.1e}, against a measured {newton[4]:.1e}), which is the definition of quadratic convergence. It is why nobody fits a GLM with gradient descent. The comparison line is a fixed step of {CONV_LR} and is therefore a weak opponent by construction — after {CONV_STEPS} steps it is still {grad[-1]:.2f} away — but the point is not that first-order methods fail. It is that when you have the Hessian and it is a cross-product matrix you can invert, you should use it.

Everything above is the case that works. Now the one that does not.""",
        level=3)

    post.add(
        "When the maximum does not exist",
        f"""Change one thing about the design. Add a dummy variable that is true for {cell['rows']} rows out of {quiet['n']}, and let every one of those rows have outcome 1.

That is not a contrived design. It is a rare category — a product, a branch, a diagnosis code, a fraud flag — that happens to be perfectly predictive in the sample you have. Nothing about it looks wrong in a data audit: the column has {cell['rows']} ones, the outcome rate overall is {quiet['rate']:.0%}, no value is missing, no column is a duplicate of another.

{snip['quiet'].markdown()}

Three things in that output.

`converged=False` every time, including at 100 passes. The coefficient on the dummy is {f5.beta[3]:.2f} at five passes, {f100.beta[3]:.2f} at a hundred, and it was still rising when I stopped it at {LONG_RUN}.

And `x1` and `x2` do not move. Not "barely move" — identical to three decimal places across a twentyfold change in the iteration limit. Whatever is wrong is confined to one coefficient, and the rest of the table is exactly as trustworthy as it would be without the problem.

That is what makes this the dangerous version. There is a textbook case — *complete* separation, where a hyperplane separates the classes outright — and it is easy to detect and easy to notice, because every coefficient blows up together. On the completely separated design in the code, the largest coefficient runs {complete[0]['largest_coefficient']:.0f} → {complete[2]['largest_coefficient']:.0f} → {complete[-1]['largest_coefficient']:.0f} as the limit is raised, and every fitted probability is at 0 or 1. Nobody ships that table.

This one is not that. There is no separating hyperplane here — I checked, with a linear program, and the answer is no, because the rows where the dummy is false contain both outcomes. The standard test for the standard problem does not fire.""")

    post.add(
        "Why it never stops, and at what rate",
        f"""The likelihood has no maximum in that direction. Making the dummy's coefficient larger always makes the fit better, because the {cell['rows']} rows it applies to are all 1 and pushing their fitted probability closer to 1 always raises the likelihood. There is a supremum and it is never attained. The MLE does not exist — that is a statement about the model and the data, not about the optimiser.

What the optimiser does about it is measurable, and it is tidier than I expected.""",
        figures=[figs["f2"]])

    post.add(
        "",
        f"""Newton's step in that direction settles at a **constant**, and the constant is {quiet['increment']:.3f}. So the coefficient after *t* passes is *t* plus a constant, until something intervenes. The coefficient *is* the iteration count.

And the second measurement follows from the first. A row whose fitted probability is near 1 has weight *p*(1 − *p*) ≈ e^(−|η|), and |η| is rising by 1 each pass, so every weight in that category divides by e each pass, and so does the smallest eigenvalue of *X*ᵗ*WX*.""",
        figures=[figs["f3"]])

    post.add(
        "",
        f"""The unweighted design has a condition number of about 5. The weighted one reaches 5×10¹⁰, on the same data, because the weights are produced by the fit rather than brought by the data. Episode two spent an entire episode on designs whose conditioning was bad on arrival; this one starts perfect and is destroyed by the fitting procedure.

The flat part of that line is not convergence. It is the weight floor — every library applies one, to stop *X*ᵗ*WX* becoming exactly singular — and past pass {quiet['straight_until']} the floor is what is being reported rather than the data. The coefficient keeps moving there: at pass {LONG_RUN} the step is still {quiet['tail_increment']:.4f}, and still positive. A coefficient path that flattens out is the single most convincing false signal in this whole episode, because flattening is what convergence looks like.""",
        level=3)

    post.add(
        "The standard error is not about your data",
        f"""Which brings us to the number that decides whether any of this reaches a conclusion. The reported standard error is the square root of a diagonal entry of (*X*ᵗ*WX*)⁻¹, and under saturation every weight in that category is pinned at the floor. So the arithmetic is not subtle:

{snip['floor'].markdown()}

The reported standard error equals 1/√(*k* × floor) to every digit printed, where *k* is the number of rows in the category and the floor is a constant chosen by whoever wrote the library. Change the floor by two orders of magnitude and the standard error changes by one. **There is no information from the data in that number.**

Now put it next to the coefficient, and read down the table.""",
        figures=[figs["t1"]])

    post.add(
        "",
        f"""At five passes: coefficient {f5.beta[3]:+.2f}, standard error {f5.standard_errors[3]:.2f}, z = {z5:.2f}. That is *p* = {p5:.3f}. A significant finding, with an odds ratio of {np.exp(f5.beta[3]):.0f}, ready to write up.

At ten passes: z = {z10:.2f}. At a hundred: z = {z100:.4f}. Nothing at all.

Same data. Same code. Same model. The p-value is a function of `max_iter`, and `max_iter` is a default nobody looked at:

{snip['default'].markdown()}

Libraries pick different numbers, and they pair them with different stopping criteria, so the same model on the same data can land on either side of 0.05 depending on which package you imported. None of those numbers is a statement about your data.

The direction is worth noticing too, because it is the opposite of the intuition. Under-iterating makes separation look *significant*; iterating properly makes it look like noise. So the failure is not "the software crashed" or even "the coefficient is huge" — it is a plausible odds ratio with a plausible p-value that came out of a fit with no answer in it.""",
        level=3)

    post.add(
        "What to do instead",
        f"""**Check for empty cells before you fit.** Not convergence warnings — those come too late and are routinely suppressed. For every categorical level and every binary column, cross-tabulate against the outcome and look for a zero:

{snip['detect'].markdown()}

{detect_lines} lines, and no model in them. It found the problem in this design, and the linear-programming test for complete separation did not, because complete separation is the wrong thing to test for.

**Check whether the fit converged.** Some APIs return a flag; `scikit-learn` raises a `ConvergenceWarning` and exposes `n_iter_`, which you compare against `max_iter` yourself. Either way it is one line, and almost no analysis pipeline has it. A coefficient from a fit that hit its iteration limit is not an estimate.

**Then choose a fix, knowingly.** There are three, and they are different claims:

*Penalise.* Add a ridge term to the weighted normal equations, which is episode five applied to ill-conditioning the fit created rather than ill-conditioning the design brought.""",
        figures=[figs["t2"]])

    post.add(
        "",
        f"""Any penalty at all makes the maximum exist, and the cost is visible: at α = {ridge[1]['alpha']} the coefficient is {ridge[1]['coefficient']:.2f} with a standard error of {ridge[1]['standard_error']:.2f} — a real one — and the log-likelihood falls from {ridge[0]['log_likelihood']:.1f} to {ridge[1]['log_likelihood']:.1f}. Half a unit of likelihood for an estimate that exists. But α is now a choice you have to defend, and the coefficient depends on it: {ridge[1]['coefficient']:.2f} at α = {ridge[1]['alpha']}, {ridge[3]['coefficient']:.2f} at α = {ridge[3]['alpha']:.0f}.

*Use Firth's penalty instead.* It adds ½ log det(*X*ᵗ*WX*) to the log-likelihood, which guarantees finite estimates and — unlike a ridge term — removes the first-order bias rather than adding shrinkage you have to justify. It is the standard answer in the literature for exactly this case, and it has one parameter fewer than a choice of α.

*Or report the category, not a coefficient.* "All {cell['rows']} rows in this category had the outcome" is a complete and honest description of what the data contains. It is a stronger statement than any odds ratio, and it does not require a model that has no maximum.

The one option that is not available is the one that happens by default: fit it, read the coefficient table, and move on.""",
        level=3)

    post.add(
        "The series, in one page",
        """Eight episodes, and one idea underneath all of them: the linear algebra is not a preliminary to the statistics. It is where the statistics either works or quietly does not.

**Episode 1.** A residual at machine precision does not mean the answer is right. The condition number is the error bar on a solve, and it is a property of the matrix you can compute before you trust anything the solver returns.

**Episode 2.** The closed form in every textbook, β̂ = (*X*ᵗ*X*)⁻¹*X*ᵗ*y*, is a correct formula and a bad algorithm: κ(*X*ᵗ*X*) = κ(*X*)², so forming the cross-product throws away half your digits. QR and SVD do not, and they are one line each.

**Episode 3.** A covariance matrix estimated entry by entry is not guaranteed to be a covariance matrix. Three individually defensible correlations produced a portfolio with a variance of −0.11, because positive definiteness is a joint property and pairwise estimation does not know about it.

**Episode 4.** Two nearly equal eigenvalues make their eigenvectors undetermined. What governs the stability of a principal component is its distance to its *neighbour*, not the share of variance it explains, and a bootstrap will understate the problem because it centres on its own arbitrary axis.

**Episode 5.** Ridge regression is a per-direction multiplier `s²/(s² + α)` on the singular values. That is the whole mechanism. It also explains why every VIF can read 1.00 on a design with a condition number near a billion, and why a nominal 95% interval covered 34%.

**Episode 6.** The leverages sum to *p*, so a single row can take a whole unit of it, and a row at leverage 1 has a residual of exactly zero — invisible to every diagnostic built on residuals. Leverage is not influence; influence needs *y*.

**Episode 7.** Eckart and Young settled which rank-*k* matrix is closest to yours and said nothing about *k*. Choosing a rank is choosing a loss function, and the elbow corresponds to none — it cannot even return zero.

**Episode 8.** When there is no closed form the fit is iterated, the iteration is still weighted least squares, and the weights are now something the fit produces. So the conditioning can be created by the procedure, and the failure mode is a coefficient whose value is the number of passes you allowed.

The thread: in each case the number the software returned was fine, and the meaning attached to it was not. None of these are bugs. Every one of them is a piece of linear algebra that the statistical vocabulary — significance, variance explained, influence, convergence — does not have a word for.

*Thank you for reading the series.* The code for all eight episodes is in the repository, tests included; the tests are where the findings actually live, because each one is a claim that was measured before it was written down, and several of them replaced a first draft that was wrong.""")

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
