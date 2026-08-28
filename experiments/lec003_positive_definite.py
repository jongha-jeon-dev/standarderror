"""Linear Algebra 3 — your covariance matrix is not positive definite.

Third episode. Opens with the answer to episode two's exercise, then the
failure: a correlation matrix assembled one entry at a time, every entry inside
[-1, 1], every entry a defensible estimate, and the matrix as a whole claiming
that a particular portfolio has negative variance.

What breaks
-----------
Pairwise-deleted correlation on a panel where one variable has a shorter history
than the others. lambda_min = -0.11, so the eigenvector for it is a portfolio
whose variance the matrix reports as negative, and an optimiser handed that
matrix has no minimum: the variance falls as the square of the position size.

The theory
----------
w'Sw is the variance of w'x, so positive semi-definiteness is not a technical
condition -- it is the statement that no combination of the variables has
negative variance. Correlations are cosines of angles between centred data
vectors, so feasibility is the triangle inequality on those angles, and
det(R) >= 0 for a 3x3 turns into an interval for the third correlation:
rho_ab*rho_ac +/- sqrt((1-rho_ab^2)(1-rho_ac^2)), which is exactly
cos(alpha -/+ beta). A complete-case matrix is Z'Z/(n-1) -- a Gram matrix, the
same object that squared the condition number in episode two -- and so cannot
violate any of this. A pairwise matrix is not a Gram matrix of anything.

The two lessons that were measured rather than expected
------------------------------------------------------
1. Under MCAR, pairwise deletion almost never produces an infeasible matrix at
   realistic n: at p = 10 and half the values missing it is 100% at n = 40 and
   0% by n = 400. The folk story ("pairwise deletion gives non-PSD matrices") is
   describing a small-sample artefact. The mechanism that actually bites is
   heterogeneous overlap -- entries estimated on different subsamples -- and
   that one does not vanish: lambda_min sits at -0.15 from n = 450 to
   n = 14,400.
2. The standard repair moves the wrong entries. Higham's nearest correlation
   matrix changes the correctly-estimated rho_ac by 0.078 and the
   wrongly-estimated rho_ab by 0.046, while rho_ab's actual error is 0.81. The
   repair is an order of magnitude too small and aimed in the wrong place, and
   the matrix it returns passes every check.

Run: `standarderror run lec003_positive_definite --publish`
"""

from __future__ import annotations

import os
from datetime import date

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import standarderror as se
from standarderror.linalg import covariance as cv
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
POST_DATE = date(2026, 8, 27)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Linear Algebra for Data Science, Taught Through What Breaks"
SERIES_TAG = "Linear Algebra"

#: The head-size failure: three correlations anybody might write down.
TOY = (0.90, 0.90, -0.90)

#: The panel. Three variables; the third has no history in the first regime,
#: and the correlation structure differs between the two -- which is what makes
#: the pairwise entries estimates of different populations.
LONG_REGIME = cv.correlation3(0.35, 0.0, 0.0)
SHORT_REGIME = cv.correlation3(-0.60, 0.90, -0.50)
N_SHORT = 250
LONG_RATIO = 8
PANEL_SEED = 11
LABELS = ("A", "B", "C")

#: The MCAR sweep. p matters more than the missing rate, so it is the thing
#: varied; the rate is held at a deliberately brutal one half.
MCAR_SIZES = (40, 60, 100, 200, 400, 800)
MCAR_WIDTHS = (3, 6, 10)
MCAR_RHO = 0.30
MCAR_RATE = 0.50
MCAR_REPS = 200

#: The heterogeneous-overlap sweep, over the short window's length.
REGIME_SIZES = (50, 100, 200, 400, 800, 1600)

LEVERAGES = (1.0, 2.0, 5.0, 10.0)


def _panel(n_short: int = N_SHORT) -> np.ndarray:
    return cv.two_regime_panel(
        LONG_RATIO * n_short, n_short,
        long_correlation=LONG_REGIME, short_correlation=SHORT_REGIME,
        rng=np.random.default_rng(PANEL_SEED))


def compute() -> dict:
    out: dict = {}

    # --- the toy failure ---------------------------------------------------
    toy = cv.correlation3(*TOY)
    out["toy"] = {"matrix": toy, "report": cv.psd_report(toy),
                  "band": cv.feasible_band(TOY[0], TOY[1]),
                  "angles": cv.correlation_angles(*TOY)}

    # --- the panel ---------------------------------------------------------
    X = _panel()
    pw = cv.pairwise_correlation(X)
    cc = cv.complete_case_correlation(X)
    C = pw["matrix"]
    report = cv.psd_report(C)
    out["panel"] = {
        "pairwise": C, "complete": cc["matrix"],
        "n_used": pw["n_used"], "max_overlap": pw["max_overlap"],
        "min_overlap": pw["min_overlap"], "complete_n": cc["n_used"],
        "rows_dropped": cc["rows_dropped"], "rows": len(X),
        "report": report,
        "complete_min_eigenvalue": float(
            np.linalg.eigvalsh(cc["matrix"])[0]),
        "band": cv.feasible_band(C[0, 1], C[0, 2]),
        "angles": cv.correlation_angles(C[0, 1], C[0, 2], C[1, 2]),
        "leverage": cv.leverage_table(C, report.worst_weights,
                                      leverages=LEVERAGES),
    }

    # --- mechanism one: sampling noise ------------------------------------
    mcar = {}
    for p in MCAR_WIDTHS:
        R = np.full((p, p), MCAR_RHO)
        np.fill_diagonal(R, 1.0)
        mcar[p] = cv.negative_rate(MCAR_SIZES, correlation=R,
                                   missing_rate=MCAR_RATE, reps=MCAR_REPS,
                                   seed=3)
    out["mcar"] = mcar

    # --- mechanism two: heterogeneous overlap ------------------------------
    out["regime"] = cv.regime_limit(
        REGIME_SIZES, long_correlation=LONG_REGIME,
        short_correlation=SHORT_REGIME, ratio=LONG_RATIO, seed=PANEL_SEED)

    # --- the repair --------------------------------------------------------
    higham = cv.nearest_correlation(C)
    clipped = cv.clip_to_psd(C)
    out["repair"] = {
        "higham": higham,
        "clipped": clipped,
        "cost": cv.repair_cost(C, higham["matrix"], labels=LABELS),
        "clip_frobenius": float(np.linalg.norm(clipped - C, "fro")),
        # What was actually wrong: the entry estimated on the long window,
        # against the same entry estimated on the window the other two use.
        "true_error_ab": float(C[0, 1] - cc["matrix"][0, 1]),
        "repaired_min_eigenvalue": float(
            np.linalg.eigvalsh(higham["matrix"])[0]),
    }
    return out


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    out = {}
    pan = res["panel"]
    C = pan["pairwise"]
    ang = pan["angles"]

    # --- f0: correlations are angles, and angles obey a triangle inequality
    def draw_angles(ax, m):
        # Unit vectors in the plane. A along the x-axis, B at the angle its
        # correlation with A implies, and C at *both* of the positions its
        # correlation with A allows -- those two positions bracket every angle
        # B-C that can exist, which is the whole argument.
        def unit(deg):
            r = np.radians(deg)
            return np.array([np.cos(r), np.sin(r)])

        a, b = ang["angle_ab"], ang["angle_ac"]
        arrows = (
            (unit(0.0), "A", m.series[0], 2.4, "left", "center"),
            (unit(a), "B", m.series[2], 2.4, "left", "center"),
            (unit(b), "C, one way up", m.series[3], 2.0, "left", "center"),
            (unit(-b), "C, or the other", m.series[3], 2.0, "left", "center"),
        )
        for vec, label, colour, width, ha, va in arrows:
            ax.annotate("", xy=vec, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", lw=width,
                                        color=colour, shrinkA=0, shrinkB=0))
            ax.annotate(label, vec, xytext=(8 if ha == "left" else 0,
                                            6 if va == "bottom" else 0),
                        textcoords="offset points", ha=ha, va=va,
                        fontsize=10, color=colour)
        # The two extreme angles between B and C. Labelled with the angle only:
        # a longer label here runs into the arrowheads, and the correlations the
        # two arcs correspond to belong in the caption.
        # `at` is where the label goes, chosen rather than taken as the arc's
        # midpoint: the inner arc's midpoint is exactly where C sits.
        for radius, lo, hi, at in ((0.58, b, a, 0.5), (0.40, -b, a, 0.14)):
            th = np.radians(np.linspace(lo, hi, 80))
            ax.plot(radius * np.cos(th), radius * np.sin(th), color="0.45",
                    lw=1.0, ls=(0, (3, 3)))
            place = np.radians(lo + at * (hi - lo))
            ax.annotate(f"{hi - lo:.0f}°",
                        ((radius + 0.07) * np.cos(place),
                         (radius + 0.07) * np.sin(place)),
                        fontsize=9.5, color="0.30", ha="center", va="center")
        ax.plot([0], [0], "o", color=m.ink, ms=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Annotation arrows do not register with the autoscaler and equal aspect
        # derives one limit from the other, so an explicit set_xlim here silently
        # crops whichever vector is tallest. Seed the data limits with the
        # geometry instead and let the aspect do the rest.
        tips = np.array([v for v, *_ in arrows] + [[0.0, 0.0], [1.45, 0.0]])
        ax.plot(tips[:, 0], tips[:, 1], lw=0, alpha=0.0)
        ax.margins(0.10)

    out["f0"] = charts.diagram(
        draw_angles,
        title="A correlation is an angle, and angles cannot be argued with",
        subtitle=(f"Three unit vectors. ρ_AB = {C[0, 1]:.2f} fixes the angle "
                  f"A–B at {ang['angle_ab']:.0f}°, and ρ_AC = {C[0, 2]:.2f} "
                  f"fixes A–C at {ang['angle_ac']:.0f}° — up or down, which is "
                  "the only freedom left."),
        equal=True, ticks=False, figsize=(7.2, 5.0),
        source="A diagram, not a measurement.",
        alt=("Four arrows from a common origin representing three variables, "
             "with the third drawn in both of the positions it is allowed and "
             "dashed arcs marking the smallest and largest angle the remaining "
             "pair can have."),
        caption=(f"The two arcs are the whole feasible set. C is as close to B "
                 f"as it can get at {ang['lower']:.0f}° (ρ_BC = "
                 f"{pan['band'][1]:.2f}) and as far as it can get at "
                 f"{ang['upper']:.0f}° (ρ_BC = {pan['band'][0]:.2f}), and "
                 f"there is no third option. The panel below reports ρ_BC = "
                 f"{C[1, 2]:.2f}, which needs {ang['angle_bc']:.0f}° — "
                 f"{ang['slack']:.0f}° more than the plane has room for, and "
                 f"no number of extra dimensions creates it."),
        path=str(IMG / f"lec03-f0-angles.{EXT}"))[0]

    # --- f1: the feasible band, and the point outside it -------------------
    def draw_band(ax, m):
        grid = np.linspace(-1.0, 1.0, 401)
        a = C[0, 1]
        half = np.sqrt(np.clip((1 - a * a) * (1 - grid ** 2), 0.0, None))
        lo, hi = a * grid - half, a * grid + half
        ax.fill_between(grid, lo, hi, color=m.series[0], alpha=0.16, lw=0)
        ax.plot(grid, lo, color=m.series[0], lw=1.4)
        ax.plot(grid, hi, color=m.series[0], lw=1.4)
        ax.plot([C[0, 2]], [C[1, 2]], "o", color=m.series[3], ms=8, zorder=5)
        ax.annotate("what the\npanel reports", (C[0, 2], C[1, 2]),
                    xytext=(12, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=9, color=m.series[3])
        ax.annotate("every correlation matrix\nlives in here",
                    (-0.35, 0.0), fontsize=9, color=m.series[0], ha="center",
                    va="center")
        ax.axhline(0.0, color="0.85", lw=0.8)
        ax.axvline(0.0, color="0.85", lw=0.8)
        ax.set_xlim(-1.02, 1.34)
        ax.set_ylim(-1.08, 1.08)

    out["f1"] = charts.diagram(
        draw_band,
        title="A correlation is not a free parameter",
        subtitle=(f"With ρ_AB fixed at {C[0, 1]:.2f}, the shaded region is "
                  "every pair (ρ_AC, ρ_BC) that a correlation matrix can have. "
                  "Outside it there is no joint distribution at all."),
        xlabel="ρ_AC",
        ylabel="ρ_BC",
        ticks=True,
        source="Simulated; standarderror/linalg/covariance.py.",
        alt=("A lens-shaped shaded region inside the square from minus one to "
             "one, with a marked point outside the region at the bottom "
             "right."),
        caption=("The region pinches to nothing at ρ_AC = ±1, which is the "
                 "sensible limit: if A and C are the same variable then ρ_BC "
                 "is already determined. The marked point is the panel's "
                 "estimate, and it is not in the region — no data set on any "
                 "three variables could have produced those three numbers "
                 "together."),
        path=str(IMG / f"lec03-f1-band.{EXT}"))[0]

    # --- f2: mechanism one, and how fast it goes away ----------------------
    mcar = pd.DataFrame(
        {f"{p} variables": [r["rate"] for r in res["mcar"][p]]
         for p in MCAR_WIDTHS},
        index=pd.Index(list(MCAR_SIZES), name="rows in the panel"))

    def as_percent(fig, ax):
        ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1.0, decimals=0))

    out["f2"] = charts.lines(
        mcar,
        title="Missing at random: the problem is real and it is small-sample",
        subtitle=(f"Share of {MCAR_REPS} simulated panels whose "
                  f"pairwise-deleted correlation matrix has a negative "
                  f"eigenvalue. Every value missing independently with "
                  f"probability {MCAR_RATE:.0%}, true correlation "
                  f"{MCAR_RHO:.1f} everywhere."),
        xlabel="rows in the panel",
        ylabel="share of panels that are infeasible",
        source="Simulated; standarderror/linalg/covariance.py.",
        decorate=as_percent, direct_labels=False,
        alt=("Three falling curves, all reaching zero, with the widest panel "
             "taking the longest to get there."),
        caption=("More variables means more constraints to satisfy at once, so "
                 "width is what drives this rather than the missing rate. But "
                 "every curve reaches zero: under missing-at-random the "
                 "pairwise estimator is consistent, and the infeasibility is "
                 "sampling noise that more rows remove."),
        path=str(IMG / f"lec03-f2-mcar.{EXT}"))[0]

    # --- f3: mechanism two, which does not go away -------------------------
    reg = pd.DataFrame(
        {"pairwise deletion": [r["min_eigenvalue"] for r in res["regime"]],
         "complete cases only": [r["min_eigenvalue_complete"]
                                 for r in res["regime"]]},
        index=pd.Index([r["n_total"] for r in res["regime"]],
                       name="rows in the panel"))

    def zero(fig, ax):
        ax.axhline(0.0, color="0.45", lw=1.0, ls=(0, (4, 3)))
        ax.annotate("below this line the matrix is not a covariance matrix",
                    (reg.index[0], 0.0), xytext=(2, -4),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=8.5, color="0.35")

    out["f3"] = charts.lines(
        reg,
        title="Unequal histories: the problem is real and more data is no help",
        subtitle=("Smallest eigenvalue of the correlation matrix of the same "
                  "three variables, estimated two ways on the same panel, as "
                  "the panel grows. One variable is observed only in the last "
                  "ninth of it."),
        xlabel="rows in the panel",
        ylabel="smallest eigenvalue",
        source="Simulated; standarderror/linalg/covariance.py.",
        logx=True, decorate=zero, direct_labels=False,
        alt=("Two flat lines against sample size on a log axis, one just below "
             "zero and one just above, neither converging towards the other."),
        caption=("Thirty-two times the data, and the pairwise line has not "
                 "moved. It is not noise: the entries are estimates of "
                 "different populations, so growing the sample sharpens the "
                 "contradiction instead of resolving it. The complete-case "
                 "line cannot go below zero at any sample size, for a reason "
                 "that is one line of algebra."),
        path=str(IMG / f"lec03-f3-regimes.{EXT}"))[0]

    # --- f4: what the repair moved, against what was wrong -----------------
    cost = res["repair"]["cost"]
    labels = [f"ρ_{e['pair'].replace('-', '')} moved by the repair"
              for e in cost["entries"]]
    values = [e["change"] for e in cost["entries"]]
    labels.append("ρ_AB's actual error")
    values.append(res["repair"]["true_error_ab"])

    out["f4"] = charts.ranked_bars(
        labels, values,
        title="The repair is aimed at the wrong entry, by an order of magnitude",
        subtitle=("What Higham's nearest correlation matrix changed, entry by "
                  "entry, against the difference between ρ_AB estimated on the "
                  "long window and the same correlation estimated on the "
                  "window the other two entries use."),
        xlabel="change in the correlation",
        source="Simulated; standarderror/linalg/covariance.py.",
        signed=True, sort="none", value_fmt=",.3f",
        alt=("Four signed horizontal bars, three small and one about ten times "
             "longer in the negative direction."),
        caption=("The repair spreads a correction of "
                 f"{cost['max_change']:.2f} at most across all three entries, "
                 "and takes the most out of ρ_AC — which was estimated "
                 "correctly, on one window, and is merely the most extreme "
                 "number in the matrix. The entry that was actually wrong is "
                 f"wrong by {abs(res['repair']['true_error_ab']):.2f}. "
                 "Afterwards the matrix passes every check."),
        path=str(IMG / f"lec03-f4-repair.{EXT}"))[0]

    # --- t1: the negative variance, priced --------------------------------
    rows = []
    for r in pan["leverage"]:
        rows.append([f"{r['leverage']:.0f}×",
                     f"{r['gross_exposure']:.2f}",
                     f"{r['variance']:+.3f}",
                     "not a number" if np.isnan(r["reported_sd"])
                     else f"{r['reported_sd']:.3f}"])
    out["t1"] = charts.table_image(
        rows,
        header=["position size", "gross exposure", "variance the matrix "
                "reports", "standard deviation"],
        title="A risk model with no minimum",
        subtitle=("The portfolio is the eigenvector for the negative "
                  "eigenvalue: short A, long B and C. Variance scales with the "
                  "square of the position, so scaling up reduces it without "
                  "limit."),
        source="Simulated; standarderror/linalg/covariance.py.",
        bold_cols=(2,),
        alt=("Table of position size against reported variance, the variance "
             "growing more negative and the standard deviation column "
             "undefined throughout."),
        caption=("This is why an infeasible covariance matrix is not a rounding "
                 "problem. A mean-variance optimiser handed this matrix does "
                 "not return a bad answer; it returns whatever the position "
                 "limits are, because it has found a direction where risk is "
                 "free and unbounded."),
        path=str(IMG / f"lec03-t1-leverage.{EXT}"))[0]
    return out


# ---------------------------------------------------------------- the post

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["toy"] = s.run(f"""
        import numpy as np

        # Three correlations, each one perfectly ordinary on its own.
        R = np.array([[ 1.0, {TOY[0]:.1f}, {TOY[1]:.1f}],
                      [{TOY[0]:.1f},  1.0, {TOY[2]:.1f}],
                      [{TOY[1]:.1f}, {TOY[2]:.1f},  1.0]])

        w = np.linalg.eigh(R)[1][:, 0]          # the smallest eigenvalue's vector
        print(f"eigenvalues      {{np.linalg.eigvalsh(R).round(3)}}")
        print(f"weights          {{w.round(3)}}")
        print(f"variance of w'x  {{w @ R @ w:.3f}}")
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError as e:
            print(f"cholesky         {{e}}")
    """, expect=["variance of w'x  -0.", "cholesky"])

    out["panel"] = s.run(f"""
        # A panel of three variables. C does not exist for the first eight
        # ninths of it -- it listed late, or the field was added to the form
        # late -- and the correlation structure of the last stretch is not the
        # structure of the earlier one. Nothing here is missing at random.
        def draw(n, R, rng):
            return rng.standard_normal((n, 3)) @ np.linalg.cholesky(R).T

        early = np.array([[1.0, {LONG_REGIME[0, 1]:.2f}, 0.0],
                          [{LONG_REGIME[0, 1]:.2f}, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
        late = np.array([[1.0, {SHORT_REGIME[0, 1]:.2f}, {SHORT_REGIME[0, 2]:.2f}],
                         [{SHORT_REGIME[0, 1]:.2f}, 1.0, {SHORT_REGIME[1, 2]:.2f}],
                         [{SHORT_REGIME[0, 2]:.2f}, {SHORT_REGIME[1, 2]:.2f}, 1.0]])

        rng = np.random.default_rng({PANEL_SEED})
        X1 = draw({LONG_RATIO * N_SHORT}, early, rng); X1[:, 2] = np.nan
        X2 = draw({N_SHORT}, late, rng)
        X = np.vstack([X1, X2])

        C = np.eye(3)
        for i in range(3):                       # each entry from whatever rows
            for j in range(i + 1, 3):            # have both variables
                ok = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
                C[i, j] = C[j, i] = np.corrcoef(X[ok, i], X[ok, j])[0, 1]
                print(f"rho_{{'ABC'[i]}}{{'ABC'[j]}} = {{C[i, j]:+.3f}} "
                      f"from {{ok.sum():5d}} rows")
        print(f"smallest eigenvalue, pairwise       {{np.linalg.eigvalsh(C)[0]:+.4f}}")

        rows = ~np.isnan(X).any(axis=1)          # the same data, one subsample
        D = np.corrcoef(X[rows].T)
        print(f"smallest eigenvalue, complete cases {{np.linalg.eigvalsh(D)[0]:+.4f}}"
              f"   ({{rows.sum()}} rows)")
        print(f"rho_AB: pairwise {{C[0, 1]:+.3f}}, complete-case {{D[0, 1]:+.3f}}")
    """, expect=["smallest eigenvalue, pairwise       -0.1",
                 "rho_AB: pairwise"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    toy, pan, rep = res["toy"], res["panel"], res["repair"]
    C, D = pan["pairwise"], pan["complete"]
    ang = pan["angles"]
    lo, hi = pan["band"]
    mcar_wide = {r["n"]: r for r in res["mcar"][max(MCAR_WIDTHS)]}
    mcar_narrow = {r["n"]: r for r in res["mcar"][min(MCAR_WIDTHS)]}
    reg_first, reg_last = res["regime"][0], res["regime"][-1]
    # The first panel size at which the widest case never fails, quoted rather
    # than eyeballed off the chart.
    wide_clean = next(r["n"] for r in res["mcar"][max(MCAR_WIDTHS)]
                      if r["rate"] == 0.0)

    # The spine, asserted rather than trusted.
    assert toy["report"].min_eigenvalue < -0.5, toy["report"]
    assert pan["report"].min_eigenvalue < -0.1, pan["report"]
    assert pan["complete_min_eigenvalue"] > 0, pan
    assert not (lo <= C[1, 2] <= hi), (lo, hi, C[1, 2])
    assert mcar_wide[max(MCAR_SIZES)]["rate"] == 0.0, mcar_wide
    assert reg_last["min_eigenvalue"] < -0.1, reg_last
    assert abs(rep["true_error_ab"]) > 8 * rep["cost"]["max_change"], rep

    post = Post(
        title=f"{SERIES_TAG} 3: Your Covariance Matrix Is Not Positive Definite",
        slug="linear-algebra-3-positive-definite",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=3,
        prerequisites=["linear-algebra-2-least-squares-three-ways"],
        date=POST_DATE,
        subtitle=("Three correlations, each one a defensible estimate, each one "
                  "inside [-1, 1] — and together a claim that a particular "
                  "portfolio has a variance of minus 0.11."),
        summary=("Positive semi-definiteness is usually presented as a "
                 "technical condition on a matrix. It is not: w'Sw is the "
                 "variance of the portfolio w'x, so a negative eigenvalue is a "
                 "combination of your variables whose variance the matrix "
                 "reports as negative, and the eigenvector names it. This "
                 "episode builds such a matrix out of ordinary estimates, "
                 "derives the constraint that was violated — which turns out "
                 "to be the triangle inequality on angles — separates the "
                 "sampling-noise version of the failure from the version that "
                 "more data makes worse, and measures what the standard repair "
                 "actually changes. It changes the wrong entry, by a factor of "
                 "ten too little, and afterwards the matrix passes every "
                 "check."),
        tags=["linear-algebra", "covariance", "missing-data", "lectures",
              "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=1900, max_words=2800,
        # Every comparison here is between two estimators on simulated data
        # whose true correlation structure is set by the script. There is no
        # predictive claim and so no baseline to beat.
        requires_baseline=False,
        data_sources=[
            "No external data. Every panel is simulated by the code shown, "
            "from correlation matrices written down in the episode, and "
            "executed when this page was built.",
            "Machinery: `standarderror/linalg/covariance.py`, tested in "
            "`tests/test_covariance.py`.",
            "Where this stops, and who does it properly: Higham, \"Computing "
            "the nearest correlation matrix — a problem from finance\", *IMA "
            "J. Numer. Anal.* 22 (2002); Little and Rubin, *Statistical "
            "Analysis with Missing Data*, chapter 3.",
        ],
        reproducibility={
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "standarderror")),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "simulation": (f"{MCAR_REPS} replications per point in the "
                           f"missing-at-random sweep; the two-regime panel is "
                           f"a single draw at seed {PANEL_SEED}, and its "
                           f"smallest eigenvalue is reported across sample "
                           f"sizes rather than averaged"),
        },
    )

    # ------------------------------------------------------------------ 0
    post.add(
        "Last episode's exercise",
        f"""The exercise was: take a dataset with missing values, compute its
correlation matrix twice — once dropping every row with any missing entry, once
computing each pair from whatever rows have both variables — and find the
smallest eigenvalue of each. One of them can come out negative. Which, and why?

**Pairwise deletion can; complete cases cannot**, and the reason is one line of
last episode's algebra. On the rows where everything is observed you have a
single matrix *Z* — centred, scaled — and the correlation matrix is
*Z*ᵗ*Z*/(*n*−1). That is a Gram matrix, the same object whose condition number
we spent last episode complaining about, and for any weights *w*

$$w^{{\\top}} Z^{{\\top}} Z w = \\lVert Z w \\rVert^{{2}} \\;\\ge\\; 0$$

A squared length. There is no arrangement of data that makes it negative, so a
complete-case correlation matrix is feasible by construction rather than by
luck. A pairwise matrix is not *Z*ᵗ*Z* for any *Z* at all — each entry comes
from a different set of rows — and nothing in its construction forces the
entries to be consistent with one another.

That is the answer, and it leaves the two questions that make it an episode. How
often does it actually happen? And when it does, what should you do — because
there is a standard repair, and it is worth finding out what it repairs.""")

    # ------------------------------------------------------------------ 1
    post.add(
        "Three numbers that cannot all be true",
        f"""First the failure at a size that fits in your head, with no missing
data and no estimation at all. Three variables. A and B move together,
ρ = {TOY[0]:.1f}. A and C move together too, ρ = {TOY[1]:.1f}. And B and C move
*opposite* each other, ρ = {TOY[2]:.1f}.

Every one of those is an ordinary number. Two of them are things people say
about data all the time. Put them in a matrix.""")

    post.add(
        "",
        f"""{snip['toy'].markdown()}

An eigenvalue of {toy['report'].min_eigenvalue:.1f}. The weights are
{toy['report'].worst_weights[0]:.3f}, {toy['report'].worst_weights[1]:.3f},
{toy['report'].worst_weights[2]:.3f} — short A, long B and C in equal size —
and the matrix says that combination has a variance of
{toy['report'].worst_variance:.1f}.

Not a small variance. Not an unstable estimate of a variance. A negative one,
which is not a number any variance can be, because a variance is an average of
squares. So the three correlations are not three slightly-inconsistent
measurements of something. They are a description of a joint distribution that
does not exist.""", level=3)

    # ------------------------------------------------------------------ 2
    post.add(
        "What positive semi-definiteness actually says",
        f"""That is worth slowing down on, because "positive semi-definite"
usually arrives as a condition a matrix has to satisfy, with no account of why
anybody would care.

Take any weights *w* and form the combination *w*ᵗ*x* — a portfolio, an index, a
factor score, a difference between two of your variables. Its variance is

$$\\operatorname{{Var}}(w^{{\\top}} x) = \\sum_i \\sum_j w_i w_j
\\operatorname{{Cov}}(x_i, x_j) = w^{{\\top}} S w$$

That is not a definition being introduced; it is the same expansion you would do
by hand for two variables, written for *n*. And since the left-hand side is a
variance, the right-hand side cannot be negative — **for every *w* at once**.
That requirement, quantified over all *w*, *is* positive semi-definiteness. It is
not a property the matrix ought to have for the linear algebra to be tidy. It is
the statement that the matrix describes something real.

The eigenvalues are how you check it without trying every *w*. If *S**v* = *λ**v*
with ‖*v*‖ = 1, then

$$v^{{\\top}} S v = v^{{\\top}} (\\lambda v) = \\lambda$$

so each eigenvalue is literally the variance of one particular portfolio — the
one its eigenvector describes. The smallest eigenvalue is the smallest variance
any portfolio can have, and if it is negative, its eigenvector hands you the
combination that proves the matrix is fiction. That is why the code above prints
the eigenvector and not just the eigenvalue: the eigenvector tells you *which
variables are in conflict*, which is the only part of the diagnosis you can act
on.""")

    post.add(
        "",
        f"""And the practical consequence is worse than a wrong number, because
*w*ᵗ*S**w* is quadratic in *w*. Double the position and the variance
quadruples — including when it is negative.""",
        figures=[figs["t1"]], level=3)

    post.add(
        "",
        f"""One aside on how to test for this, since the code above did it two
ways. `np.linalg.cholesky` raises if and only if the matrix is not positive
*definite*, it costs about a third of what an eigendecomposition does, and it is
what a well-written library calls before it trusts a covariance matrix. But note
the word: *definite*, not *semi-definite*. A genuinely rank-deficient covariance
matrix — two variables that are the same variable, or more variables than
observations — is a perfectly real covariance matrix with a zero eigenvalue, and
Cholesky refuses it. So a Cholesky failure is a reason to look, and the
eigenvalues are what you look at. The distinction matters in the same way
episode one's did: *how negative* is a question with an answer, and *is it
singular* is not.""", level=3)

    # ------------------------------------------------------------------ 3
    post.add(
        "A correlation is an angle",
        f"""So which of the three numbers was wrong? None of them, individually
— and that is the point. What was wrong was the combination, and there is an
exact statement of what combinations are allowed.

Start from the geometry, because it makes the answer obvious before any algebra.
Centre and scale each variable, so each is a vector of length one in *n*
dimensions. Then the correlation between two of them is their inner product,
which is the cosine of the angle between them:

$$\\rho_{{xy}} = \\frac{{\\langle x, y \\rangle}}{{\\lVert x \\rVert \\lVert y
\\rVert}} = \\cos \\theta$$

Correlation 1 is a zero-degree angle, correlation 0 is a right angle,
correlation −1 is a hundred and eighty degrees. And now the constraint writes
itself: if A is {ang['angle_ab']:.0f}° from B, and A is {ang['angle_ac']:.0f}°
from C, then C cannot be anywhere it likes relative to B. It can be as close to
B as the difference of those two angles, {ang['lower']:.0f}°, or as far as their
sum, {ang['upper']:.0f}°, and nothing in between is ruled out — those are the
endpoints of an interval, not two options. This is the triangle inequality,
on angles rather than on distances.""",
        figures=[figs["f0"]])

    post.add(
        "",
        f"""The algebra says the same thing and is worth having, because it
generalises past three variables where the picture stops. A symmetric matrix is
positive semi-definite only if every determinant you can form by deleting the
same rows and columns comes out non-negative. That direction is the one we need
here — it is necessary, and the sufficient version wants all of those minors
rather than only the leading ones. Take the whole determinant of the 3 × 3
correlation matrix, which multiplies out to

$$\\det R = 1 + 2 \\rho_{{ab}} \\rho_{{ac}} \\rho_{{bc}} - \\rho_{{ab}}^{{2}} -
\\rho_{{ac}}^{{2}} - \\rho_{{bc}}^{{2}} \\;\\ge\\; 0$$

and read it as a quadratic in the one correlation we want to solve for:

$$-\\rho_{{bc}}^{{2}} + 2 \\rho_{{ab}} \\rho_{{ac}} \\, \\rho_{{bc}} + \\left(1 -
\\rho_{{ab}}^{{2}} - \\rho_{{ac}}^{{2}}\\right) \\;\\ge\\; 0$$

A downward parabola, so the feasible set is the closed interval between its two
roots, and the quadratic formula gives them directly:

$$\\rho_{{bc}} \\in \\left[\\; \\rho_{{ab}} \\rho_{{ac}} - \\sqrt{{(1 -
\\rho_{{ab}}^{{2}})(1 - \\rho_{{ac}}^{{2}})}}, \\;\\; \\rho_{{ab}} \\rho_{{ac}}
+ \\sqrt{{(1 - \\rho_{{ab}}^{{2}})(1 - \\rho_{{ac}}^{{2}})}} \\;\\right]$$

Now substitute *ρ*_ab_ = cos *α* and *ρ*_ac_ = cos *β*. The square root becomes
sin *α* sin *β*, and the two endpoints are cos *α* cos *β* ∓ sin *α* sin *β* —
which are cos(*α* + *β*) and cos(*α* − *β*). The determinant condition and the
triangle inequality are not two facts that happen to agree. They are the same
fact, written once in coordinates and once in angles.

For the toy matrix: ρ_AB = ρ_AC = {TOY[0]:.1f} puts the feasible interval for
the third correlation at [{toy['band'][0]:.2f}, {toy['band'][1]:.2f}]. Not
[−1, 1] — the third correlation is forced to be *strongly positive*, and
{TOY[2]:.1f} is not merely outside the interval, it is at the opposite end of
the scale. Two strong positive correlations do not leave room for a negative
one.""", level=3)

    post.add(
        "",
        f"""It is worth looking at the whole feasible region rather than one
interval, because the shape of it is the useful intuition: a correlation is not
a parameter you get to choose, it is a parameter the other correlations have
already spent.""",
        figures=[figs["f1"]], level=3)

    # ------------------------------------------------------------------ 4
    post.add(
        "So how does anyone build a matrix like that?",
        f"""Nobody types in three contradictory correlations on purpose. They
arrive one entry at a time, each from a defensible calculation, and the standard
way that happens is missing data.

Here is a panel of three variables where C simply does not exist for most of the
history — it listed late, the field was added to the form late, the question was
added to the survey late. Each correlation is computed from the rows that have
both variables in it, which is the obvious thing to do and is what
`pandas.DataFrame.corr` does by default.""")

    post.add(
        "",
        f"""{snip['panel'].markdown()}

Read the row counts, because they are the diagnosis. ρ_AB was estimated on
{pan['max_overlap']:,} rows and the other two on {pan['min_overlap']}. The
pairwise matrix has a smallest eigenvalue of
{pan['report'].min_eigenvalue:+.4f}; the complete-case matrix, on the same
panel, has {pan['complete_min_eigenvalue']:+.4f} and is perfectly well behaved.

And the entry that broke it is the one with the most data. ρ_AB is
{C[0, 1]:+.3f} on the long window and {D[0, 1]:+.3f} on the {pan['complete_n']}
rows the other two entries use — the correlation between A and B is not the same
in the two regimes, so the long-window estimate is a fine estimate of something,
and it is not the same something the other two entries are estimates of. Feed it
to the geometry of the previous section: with ρ_AB = {C[0, 1]:.2f} and
ρ_AC = {C[0, 2]:.2f}, the angle between B and C has to be between
{ang['lower']:.0f}° and {ang['upper']:.0f}°. The reported ρ_BC = {C[1, 2]:.2f}
needs {ang['angle_bc']:.0f}°. It is {ang['slack']:.0f} degrees short of
possible, and using nine times as much data for one entry is exactly what put it
there.""", level=3)

    post.add(
        "",
        f"""Now the part I expected to go the other way. The folk version of
this is "pairwise deletion gives you non-positive-definite matrices", stated
about missing data in general. So: how often, if the missingness is the benign
kind — every value dropped independently, no relationship to anything?""")

    post.add(
        "",
        f"""Hardly ever, at any sample size you would work with. At three
variables and half the values thrown away it happens in
{mcar_narrow[min(MCAR_SIZES)]['rate']:.0%} of panels of
{min(MCAR_SIZES)} rows and never again. Ten variables is much worse at
{min(MCAR_SIZES)} rows — {mcar_wide[min(MCAR_SIZES)]['rate']:.0%}, because
there are forty-five pairwise constraints to satisfy simultaneously instead of
three — and it is gone by {wide_clean} rows all the same. Width drives it, and
sample size cures it, which together say the thing worth knowing: under
missing-at-random the pairwise estimator is *consistent*, so infeasibility is
sampling noise, and noise is what more data removes.""",
        figures=[figs["f2"]], level=3)

    post.add(
        "",
        f"""The unequal-histories version does not behave like that at all.""",
        figures=[figs["f3"]], level=3)

    post.add(
        "",
        f"""From {reg_first['n_total']:,} rows to {reg_last['n_total']:,} — a
factor of {reg_last['n_total'] / reg_first['n_total']:.0f} — the smallest
eigenvalue goes from {reg_first['min_eigenvalue']:+.3f} to
{reg_last['min_eigenvalue']:+.3f}. It does not improve, because there is nothing
for it to converge to: the entries are consistent estimates of the correlations
of *different populations*, and a larger sample estimates each of them more
precisely. More data sharpens the contradiction.

Which flips the diagnostic value of the whole thing. A negative eigenvalue in a
small, wide, randomly-incomplete dataset is a nuisance and you should shrink or
regularise your way past it. A negative eigenvalue in a large dataset is
*information*: it is telling you that your entries are not describing the same
population, and the overlap counts will usually show you where.""", level=3)

    # ------------------------------------------------------------------ 5
    post.add(
        "The repair, and what it repairs",
        f"""There is a standard fix, and it is a genuinely nice piece of
mathematics. You want the closest matrix to yours that is a correlation matrix,
in the sense of minimising the sum of squared entry differences. Two constraints
— positive semi-definite, and unit diagonal — and projecting onto either one
breaks the other, so Higham's method alternates between them with a correction
term that keeps it from settling on the wrong point. It converged here in
{rep['higham']['iterations']} iterations and gives a matrix whose smallest
eigenvalue is {abs(rep['repaired_min_eigenvalue']):.0e} — zero, up to the
arithmetic.

The cheap version — take the eigendecomposition, set the negative eigenvalues to
zero, rebuild, then divide through to restore the unit diagonal — is about one
percent further away here in Frobenius norm
({rep['clip_frobenius']:.4f} against {rep['cost']['frobenius']:.4f}). That last
renormalisation is not optional: zeroing an eigenvalue changes the diagonal, and
skipping it leaves you with variances nobody measured.

Both give you a matrix that passes every check. Here is what they changed.""",
        figures=[figs["f4"]])

    post.add(
        "",
        f"""The repair moved three correlations by at most
{rep['cost']['max_change']:.3f}, and took the most out of ρ_AC — which was
estimated correctly, on one window, and is guilty only of being the most extreme
number in the matrix. The entry that was actually wrong, ρ_AB, moved by
{abs(rep['cost']['entries'][0]['change']):.3f}, and it was wrong by
{abs(rep['true_error_ab']):.2f}.

That is not a criticism of the algorithm, which solves exactly the problem it
states. It is a statement about what the problem is. "Find the nearest feasible
matrix" treats infeasibility as damage distributed over the whole matrix,
which is right when the cause is floating point or a small-sample wobble, and
wrong when one entry is an estimate of a different population. In the second
case the repair is a floor over a hole: the matrix now passes Cholesky, the
optimiser now returns a finite answer, and the answer is still built on
ρ_AB = {rep['higham']['matrix'][0, 1]:+.3f} when the number consistent with the
rest of the matrix was {D[0, 1]:+.3f}.

And the negative eigenvalue was the only evidence you had.""", level=3)

    # ------------------------------------------------------------------ 6
    post.add(
        "What to take away, and what is still hiding",
        f"""Four things.

**Read a negative eigenvalue as a sentence, not as a number.** Its eigenvector
is a portfolio, its value is that portfolio's variance, and "this combination of
my variables has negative variance" tells you where to look. `eigh` returns both
and the vector is the useful half.

**Check the overlap counts before the eigenvalues.** If the entries of your
covariance matrix were computed on different numbers of rows, they were computed
on different samples, and consistency between them is a hope rather than a
property. This is the same failure whether it arrives as pairwise deletion, as a
correlation borrowed from a longer history, or as a stress-test overlay somebody
typed in by hand.

**Distinguish the noise case from the bias case, and the test is sample size.**
If the smallest eigenvalue moves towards zero as you add rows, it was noise and
a repair is honest. If it sits still, no repair is honest, and the fix is to
estimate every entry on one sample even when that means a much smaller one.

**Correlations are not free parameters.** Any scenario written down entry by
entry — a stress test, an elicited prior, a hand-adjusted risk model — needs
checking against the feasible region, and the feasible region is much smaller
than [−1, 1]^(p choose 2). Two strong correlations determine the third to within
a narrow interval, and at higher *p* the constraints compound.

And one thing this episode has quietly leaned on. Every diagnosis above read the
*smallest* eigenvalue and its eigenvector as if that pairing were a stable,
interpretable object. For the smallest eigenvalue of a broken matrix it is,
because it is a long way from its neighbours. It is not in general. When two
eigenvalues are close, their eigenvectors are not individually determined at all
— only the plane they span is — and every principal component analysis that
names its second component is relying on a separation nobody checked. Next
episode.

*Exercise.* Take a dataset with six or more numeric columns and compute the
eigenvalues of its correlation matrix. Bootstrap the rows two hundred times and
recompute. For each pair of adjacent eigenvalues, count how often the two swap
order across the bootstrap. Then look at the loadings of the two components that
swap most, and ask what a sentence beginning "the second component represents"
would have meant. The answer is at the top of episode four.""")

    return post


def main() -> Post:
    post = build()
    problems = post.audit()
    print(f"words: {post.word_count()}")
    print("audit:", "clean" if not problems else "")
    for p in problems:
        print("  -", p)
    return post
