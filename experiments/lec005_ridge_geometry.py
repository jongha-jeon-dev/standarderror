"""Linear Algebra 5: What Ridge Does to the Geometry.

Three claims, each measured on a design whose spectrum is known:

* The two standard collinearity diagnostics disagree in both directions. A design
  of mutually uncorrelated columns on different scales has every VIF at 1.00 --
  the floor -- and a condition number near 1e9. A design of genuine near-
  duplicates has a VIF of 861 and a condition number of 73. Neither is wrong;
  they answer different questions, and only one of them is about the solve.
* Ridge is an eigenvalue shift and nothing else. In the basis where the design is
  diagonal it multiplies direction i by s^2/(s^2 + alpha), so it shrinks hardest
  exactly where the data saw least. Cross-validation picks alpha = 2.6 here and
  spends 4.1 of 9 parameters.
* The interval you would report is exact where the data is strong and worthless
  where it is weak. Nominal 95 percent intervals cover 95 percent when the truth
  lies along the leading singular direction and 43 percent when it lies along the
  weakest, with a bias of up to 7 standard errors -- and nothing in the output
  distinguishes the two cases.

Run: `standarderror run lec005_ridge_geometry --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.linalg import ridge as rg
from standarderror.linalg import spectral as sp
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
POST_DATE = date(2026, 9, 1)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Linear Algebra for Data Science, Taught Through What Breaks"
SERIES_TAG = "Linear Algebra"

#: Episode four's exercise: its block matrix, with one variable's units changed.
EXERCISE_BLOCKS = (0.80, 0.78, 0.30)
UNIT_CHANGE = 1000.0

#: The design VIF cannot see, and the one it over-reads.
N = 600
STRENGTH = 0.9995
P_COLLINEAR = 8

#: The ridge grid, and the truth placed in two different parts of the spectrum.
ALPHAS = np.logspace(-3, 4, 40)
COVER_ALPHAS = (0.1, 1.0, 2.57, 10.0, 100.0)
COVER_REPS = 600
BETA_SCALE = 3.0
SIGMA = 1.0
SEED = 31


def compute() -> dict:
    out: dict = {}

    # --- the answer to episode four's exercise -----------------------------
    C4 = sp.block_pairs(EXERCISE_BLOCKS)
    d = np.ones(C4.shape[0])
    d[0] = UNIT_CHANGE
    S4 = np.outer(d, d) * C4
    out["exercise"] = {
        "correlation": sp.spectrum(C4),
        "covariance": sp.spectrum(S4),
        "scale": UNIT_CHANGE,
    }

    # --- the two diagnostics ----------------------------------------------
    units = rg.units_design(N, rng=np.random.default_rng(SEED))
    coll = rg.collinear_design(N, rng=np.random.default_rng(SEED + 1),
                               p=P_COLLINEAR, strength=STRENGTH)
    out["designs"] = [
        {"name": "uncorrelated columns, different units", "X": units,
         "vif": rg.vif(units), "kappa": float(rg.condition_indices(units)[-1])},
        {"name": "near-duplicate columns, one scale", "X": coll,
         "vif": rg.vif(coll), "kappa": float(rg.condition_indices(coll)[-1])},
    ]

    # --- the penalty is in the units of each coefficient -------------------
    # ||beta||^2 adds up numbers that are not commensurable unless the columns
    # are. Measured as the weight the penalty effectively puts on each column,
    # relative to one of them.
    r2 = np.random.default_rng(SEED + 5)
    beta_u = np.array([1.0, 1e-3, 5.0, 1e-8])
    y_u = units @ beta_u + r2.normal(0.0, 1.0, N)
    sd = units.std(0)
    sd[0] = 1.0
    out["scaling"] = {
        "names": ("intercept", "duration in seconds", "a probability",
                  "an amount of money"),
        "sd": sd,
        "relative_penalty": (sd[2] / sd) ** 2,
        "raw": rg.ridge_fit(units, y_u, 1.0).coefficients,
        "standardised": rg.ridge_fit(units / sd, y_u, 1.0).coefficients / sd,
        "truth": beta_u,
    }

    # --- ridge on the design that is genuinely near-singular ---------------
    p = coll.shape[1]
    rng = np.random.default_rng(SEED + 2)
    beta_true = np.ones(p)
    y = coll @ beta_true + rng.normal(0.0, SIGMA, N)
    s = np.linalg.svd(coll, compute_uv=False)
    cv = rg.cross_validated_alpha(coll, y, ALPHAS, folds=5, rng=rng)
    out["ridge"] = {
        "X": coll, "p": p, "singular_values": s, "cv": cv,
        "df_curve": [rg.effective_df(s, a) for a in ALPHAS],
        "fit": rg.ridge_fit(coll, y, cv["alpha"]),
        "ols_df": float(p),
    }

    # --- episode two's deferred question: rcond, answered with regularisation
    out["hard_soft"] = rg.hard_against_soft(coll, y, beta_true)
    out["real_directions"] = int((s > 0.1 * s.max()).sum())

    # --- where the truth sits decides whether the interval means anything --
    Vt = np.linalg.svd(coll, full_matrices=False)[2]
    cases = [("truth along the strongest direction", Vt[0] * BETA_SCALE),
             ("truth along the weakest direction", Vt[-1] * BETA_SCALE)]
    # The alpha cross-validation actually chose belongs on the curve, or the
    # figure shows the effect everywhere except at the setting a reader would use.
    grid = sorted(set(COVER_ALPHAS) | {round(cv["alpha"], 3)})
    out["coverage_alphas"] = grid
    out["coverage"] = []
    for name, beta in cases:
        row = {"name": name, "by_alpha": []}
        for a in grid:
            c = rg.coverage(coll, beta, alpha=a, sigma=SIGMA, reps=COVER_REPS,
                            rng=np.random.default_rng(SEED + 3))
            row["by_alpha"].append({
                "alpha": float(a),
                "median_coverage": float(np.median(c["coverage"])),
                "min_coverage": float(c["coverage"].min()),
                "max_bias_se": float(np.abs(c["mean_bias"]
                                            / c["standard_error"]).max()),
            })
        out["coverage"].append(row)
    return out


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    out: dict = {}
    designs, ridge = res["designs"], res["ridge"]
    s, cv = ridge["singular_values"], ridge["cv"]

    # --- f0: what ridge does, as one curve per alpha -----------------------
    grid = np.logspace(-1.2, 2.0, 200)
    out["f0"] = charts.lines(
        pd.DataFrame({f"α = {a:g}": rg.shrinkage(grid, a)
                      for a in (0.1, 1.0, 10.0, 100.0)}, index=grid),
        title="Ridge keeps the directions the data measured and discards the rest",
        subtitle=("The factor s²/(s² + α) that ridge multiplies each direction "
                  "by, against that direction's singular value."),
        xlabel="singular value of the design in that direction",
        ylabel="fraction of the coefficient that survives",
        source="A property of the estimator, not a measurement.",
        logx=True, ylim=(0.0, 1.05),
        alt=("Four S-shaped curves rising from zero to one, each shifted right "
             "as alpha grows."),
        caption=("Not a uniform shrinkage of the coefficient vector. Each curve "
                 "is flat at one over the well-measured directions and falls off "
                 "a cliff below a threshold that α sets — so the penalty lands "
                 "almost entirely on the directions the design saw least."),
        path=str(IMG / f"lec05-f0-shrinkage.{EXT}"))[0]

    # --- t1: the two diagnostics, side by side -----------------------------
    rows = []
    for d in designs:
        digits = np.log10(d["kappa"] ** 2)
        rows.append([d["name"], f"{d['X'].shape[1]}",
                     f"{d['vif'].max():.2f}",
                     "fine" if d["vif"].max() < 10 else "alarming",
                     f"{d['kappa']:.3g}", f"{digits:.1f}"])
    out["t1"] = charts.table_image(
        rows,
        header=["design", "columns", "largest VIF", "VIF's verdict",
                "κ(X)", "digits κ(XᵗX) costs"],
        title="Two collinearity diagnostics, disagreeing in both directions",
        subtitle=("The usual rule of thumb is that a VIF above 10 is a problem. "
                  "Read its verdict against the last column."),
        source="Simulated; standarderror/linalg/ridge.py.",
        bold_cols=(2, 4),
        alt=("Two-row table comparing largest VIF and condition number for two "
             "designs, where the diagnostics disagree in opposite directions."),
        caption=("Neither statistic is wrong. The VIF asks how much the other "
                 "columns inflate one coefficient's variance, which is scale-free "
                 "and ignores the intercept; the condition number asks how much "
                 "the solve can amplify any error at all. Only the second is "
                 "about the arithmetic you are about to do."),
        path=str(IMG / f"lec05-t1-diagnostics.{EXT}"))[0]

    # --- f1: the parameters you actually spent -----------------------------
    def mark_cv(fig, ax):
        ax.axvline(cv["alpha"], color="0.45", lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"cross-validation chose α = {cv['alpha']:.2f},\n"
                    f"which spends {cv['effective_df']:.1f} of {ridge['p']}",
                    (cv["alpha"], 0.97), xycoords=("data", "axes fraction"),
                    xytext=(7, -4), textcoords="offset points",
                    fontsize=8.5, color="0.35", va="top")

    out["f1"] = charts.lines(
        pd.DataFrame({"parameters the fit actually spends": ridge["df_curve"]},
                     index=ALPHAS),
        title="The degrees of freedom are not p, and nothing prints them",
        subtitle=(f"Σ s²/(s² + α) — the trace of the ridge hat matrix — for the "
                  f"{ridge['p']}-column design, against the penalty."),
        xlabel="α",
        ylabel="effective degrees of freedom",
        source="Simulated; standarderror/linalg/ridge.py.",
        logx=True, decorate=mark_cv, direct_labels=False,
        alt=("A curve falling from nine to near zero as alpha grows, with a "
             "dashed vertical line at the cross-validated alpha."),
        caption=(f"At α = 0 this is p. At the α cross-validation chose it is "
                 f"{cv['effective_df']:.2f}, so the fit spent under half the "
                 f"parameters the output reports — and every standard error, "
                 f"AIC and residual degree-of-freedom count beside it assumed "
                 f"{ridge['p']}."),
        path=str(IMG / f"lec05-f1-df.{EXT}"))[0]

    # --- f2: the coefficient, direction by direction -----------------------
    fit = ridge["fit"]
    out["f2"] = charts.ranked_bars(
        [f"direction {i + 1}  (s = {v:.2f})" for i, v in enumerate(s)],
        list(fit.shrinkage),
        title="What survived, in the basis where the design is diagonal",
        subtitle=(f"The multiplier applied to each direction at the "
                  f"cross-validated α = {cv['alpha']:.2f}."),
        xlabel="fraction of the least-squares coefficient that remains",
        source="Simulated; standarderror/linalg/ridge.py.",
        sort="none", value_fmt=".3f",
        alt=("Nine horizontal bars: the first three near one, the remaining six "
             "very short."),
        caption=("Three directions pass through almost untouched and six are cut "
                 "to under a tenth. The fit that comes out is a three-parameter "
                 "fit wearing nine coefficients."),
        path=str(IMG / f"lec05-f2-directions.{EXT}"))[0]

    # --- f3: and whether the interval means anything -----------------------
    out["f3"] = charts.lines(
        pd.DataFrame(
            {c["name"]: [b["median_coverage"] for b in c["by_alpha"]]
             for c in res["coverage"]},
            index=res["coverage_alphas"]),
        title="The same interval formula, exact here and worthless there",
        subtitle=(f"Coverage of nominal 95 percent intervals over "
                  f"{COVER_REPS} draws, with the truth placed in two different "
                  f"parts of the same design's spectrum."),
        xlabel="α",
        ylabel="share of intervals containing the truth",
        source="Simulated; standarderror/linalg/ridge.py.",
        logx=True, ylim=(0.0, 1.02),
        decorate=lambda fig, ax: ax.axhline(0.95, color="0.45", lw=1.2,
                                            ls=(0, (4, 3)), zorder=1),
        alt=("Two curves against alpha: one flat near 0.95, the other falling to "
             "zero."),
        caption=("The variance formula is the exact one and is identical in both "
                 "cases; what differs is the bias, which no variance formula "
                 "contains. Which curve you are on depends on where the truth "
                 "sits relative to your design — which is the thing you do not "
                 "know."),
        path=str(IMG / f"lec05-f3-coverage.{EXT}"))[0]

    # --- f4: the cliff and the ramp, at matched cost -----------------------
    hs = res["hard_soft"]
    out["f4"] = charts.lines(
        pd.DataFrame(
            {"truncated SVD, keeping k directions whole":
                 [r["hard_error"] for r in hs],
             "ridge at the α that spends k parameters":
                 [r["soft_error"] for r in hs]},
            index=[r["rank"] for r in hs]),
        title="Two answers to \u2018how nearly collinear is too collinear\u2019",
        subtitle=("Distance from the true coefficients, for a hard truncation at "
                  "each rank and for the ridge penalty spending the same number "
                  "of parameters."),
        xlabel="parameters spent",
        ylabel="‖β̂ − β‖",
        source="Simulated; standarderror/linalg/ridge.py.",
        alt=("Two U-shaped curves lying almost on top of each other, both "
             "reaching their minimum at the same point."),
        caption=(f"Both bottom out at {res['real_directions']}, which is how "
                 f"many directions this design actually has. Near that point the "
                 f"two agree to within a few percent; they part company only "
                 f"where the budget is far from what the data supports, and "
                 f"there neither is a good answer."),
        path=str(IMG / f"lec05-f4-hard-soft.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    ridge, cv = res["ridge"], res["ridge"]["cv"]
    designs = res["designs"]
    weak = res["coverage"][1]["by_alpha"]
    worst = min(b["median_coverage"] for b in weak)

    def two_verdicts(panel, m):
        panel.barh([0.68], [0.08], height=0.22, color=m.grid, left=0.08)
        panel.barh([0.30], [0.84], height=0.22, color=m.ink, left=0.08)
        panel.set_xlim(0, 1); panel.set_ylim(0, 1)

    def cliff(panel, m):
        x = np.linspace(-1.4, 1.6, 120)
        panel.plot(x, 1 / (1 + 10.0 ** (-3.2 * x)), color=m.ink, lw=2.4)
        panel.set_xlim(-1.5, 1.7); panel.set_ylim(-0.08, 1.08)

    def falling(panel, m):
        x = np.linspace(0, 1, 60)
        panel.plot([0, 1], [0.9, 0.9], color=m.grid, lw=2.2)
        panel.plot(x, 0.9 / (1 + np.exp(9 * (x - 0.45))), color=m.ink, lw=2.4)
        panel.set_xlim(0, 1); panel.set_ylim(0, 1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=5,
        headline="Ridge spends parameters your output does not count",
        panels=[
            (two_verdicts, f"{designs[0]['vif'].max():.2f}",
             "VIF on a hopeless design"),
            (cliff, f"{cv['effective_df']:.1f} of {ridge['p']}",
             "parameters spent"),
            (falling, f"{worst:.0%}", "a 95% interval covers"),
        ],
        note=("The variance inflation factor is at its floor on a design with a "
              "condition number near 1e9, because it is scale-free and cannot "
              "see the intercept. Ridge then shrinks each direction by "
              "s²/(s² + α), so the fit cross-validation picks spends under half "
              "the parameters reported beside it — and the intervals hold "
              "exactly where the data is strong and collapse where it is not."),
        alt=("A three-panel hand-drawn strip. The first shows a very short bar "
             "above a long one. The second shows an S-curve falling off a cliff. "
             "The third shows a flat line and a curve dropping away beneath it."),
        mode="light",
        path=str(IMG / f"lec05-hero.{EXT}"))[0]


# ---------------------------------------------------------------- the post

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["units"] = s.run(f"""
        import numpy as np

        # Episode four's block matrix, and one variable measured in different
        # units. Nothing about the data changed; one column is now in grams
        # rather than kilograms.
        C = np.eye(6)
        for i, c in enumerate({list(EXERCISE_BLOCKS)}):
            C[2 * i, 2 * i + 1] = C[2 * i + 1, 2 * i] = c

        d = np.ones(6); d[0] = {UNIT_CHANGE:.0f}
        S = np.outer(d, d) * C                 # the covariance, in the new units

        for name, M in (("correlation", C), ("covariance", S)):
            v = np.sort(np.linalg.eigvalsh(M))[::-1]
            print(f"{{name:12s}} first component carries {{v[0] / v.sum():8.4%}}"
                  f"   top gap {{v[0] - v[1]:12.2f}}")
    """, expect=["correlation", "covariance"])

    out["vif"] = s.run(f"""
        # Four columns nobody would call collinear: an intercept, a duration in
        # seconds, a probability, and an amount of money. All mutually
        # uncorrelated, so every variance inflation factor is at its floor.
        rng = np.random.default_rng({SEED})
        X = np.column_stack([np.ones({N}), rng.normal(3600, 600, {N}),
                             rng.normal(0.30, 0.10, {N}),
                             rng.normal(5e7, 1e7, {N})])

        def vif(X):
            "1 / (1 - R^2) from regressing each predictor on the others."
            Z, out = X[:, 1:], []
            for j in range(Z.shape[1]):
                A = np.column_stack([np.ones(len(Z)), np.delete(Z, j, axis=1)])
                resid = Z[:, j] - A @ np.linalg.lstsq(A, Z[:, j], rcond=None)[0]
                r2 = 1 - resid.var() / Z[:, j].var()
                out.append(1 / (1 - r2))
            return np.array(out)

        sv = np.linalg.svd(X, compute_uv=False)
        print(f"VIFs                    {{vif(X).round(4)}}")
        print(f"largest                 {{vif(X).max():.4f}}   "
              f"(rule of thumb: above 10 is a problem)")
        print(f"condition number        {{sv.max() / sv.min():.2e}}")
        print(f"digits kappa(X'X) costs {{2 * np.log10(sv.max() / sv.min()):.1f}} of 15.7")
    """, expect=["largest", "condition number", "digits"])

    out["ridge"] = s.run("""
        # Ridge, written the way it actually works: one multiplier per direction.
        U, sv, Vt = np.linalg.svd(X, full_matrices=False)
        for alpha in (0.0, 1.0, 100.0):
            f = sv**2 / (sv**2 + alpha)
            print(f"alpha {alpha:6.0f}   keeps {np.round(f, 4)}   "
                  f"df {f.sum():.3f}")
    """, expect=["alpha", "df"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    ex, designs, ridge = res["exercise"], res["designs"], res["ridge"]
    cv = ridge["cv"]
    strong, weak = res["coverage"]

    # The spine, asserted rather than trusted.
    assert ex["covariance"].variance_share[0] > 0.999, ex
    assert designs[0]["vif"].max() < 1.05 and designs[0]["kappa"] > 1e7, designs[0]
    assert designs[1]["vif"].max() > 100 and designs[1]["kappa"] < 500, designs[1]
    assert cv["effective_df"] < ridge["p"] / 2, cv
    assert min(b["median_coverage"] for b in weak["by_alpha"]) < 0.5, weak
    assert min(b["median_coverage"] for b in strong["by_alpha"]) > 0.9, strong

    post = Post(
        title=f"{SERIES_TAG} 5: What Ridge Does to the Geometry",
        slug="linear-algebra-5-ridge-geometry",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=5,
        prerequisites=["linear-algebra-4-equal-eigenvalues"],
        date=POST_DATE,
        subtitle=("Every variance inflation factor at 1.00 on a design with a "
                  "condition number near a billion, a cross-validated ridge fit "
                  "that spends 3.6 of its 9 parameters, and a nominal 95 percent "
                  "interval that covers 34 percent."),
        summary=("Ridge is introduced as a penalty and works as an operation on "
                 "the spectrum: in the basis where the design is diagonal it "
                 "multiplies each direction by s²/(s² + α) and does nothing "
                 "else, which is why it shrinks hardest exactly where the data "
                 "saw least. Three consequences the usual output hides — the "
                 "standard collinearity diagnostic is at its floor on a design "
                 "with no correct digits, the fit spends far fewer parameters "
                 "than it reports, and the interval you would quote is exact "
                 "where the evidence is strong and worthless where it is thin, "
                 "with nothing in the printout to tell you which."),
        tags=["linear-algebra", "ridge-regression", "regularisation",
              "lectures", "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every design matrix here is constructed in the "
            "episode and every number is produced by the code shown, executed "
            "when this page was built.",
            "Machinery: `standarderror/linalg/ridge.py`, tested in "
            "`tests/test_ridge.py`.",
            "Where this stops: Hoerl and Kennard, \"Ridge regression: biased "
            "estimation for nonorthogonal problems\", *Technometrics* 12 (1970); "
            "Belsley, Kuh and Welsch, *Regression Diagnostics* (1980), chapter 3; "
            "Hastie, Tibshirani and Friedman, *The Elements of Statistical "
            "Learning*, section 3.4.1.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": (f"{N} rows per design, {COVER_REPS} draws per coverage "
                           f"point, five contiguous cross-validation blocks"),
            "determinism": f"one seed, {SEED}, and every draw derived from it",
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    ex, designs, ridge = res["exercise"], res["designs"], res["ridge"]
    cv, s = ridge["cv"], ridge["singular_values"]
    strong, weak = res["coverage"]
    units, coll = designs
    at_cv = [b for b in weak["by_alpha"]
             if abs(b["alpha"] - round(cv["alpha"], 3)) < 1e-9][0]
    strong_at_cv = [b for b in strong["by_alpha"]
                    if abs(b["alpha"] - round(cv["alpha"], 3)) < 1e-9][0]
    kept = int((ridge["fit"].shrinkage > 0.5).sum())
    sc = res["scaling"]

    post.add(
        "Last episode's exercise",
        f"""The exercise was: take the block matrix from episode four, multiply one variable by 1,000 as a change of units would, and look at the eigenvalues of the *covariance* matrix instead of the correlation matrix.

{snip['units'].markdown()}

The first component now carries {ex['covariance'].variance_share[0]:.4%} of the variance, and the scree plot has one bar. Keep one component and you have kept everything — which is true, and is a fact about the column being in grams.

Notice what else went. The whole of episode four was about two eigenvalues {ex['correlation'].gaps[0]:.2f} apart. In the new units the top gap is {ex['covariance'].gaps[0]:,.0f}. The near-tie did not get resolved by better data; it got hidden by an arbitrary rescaling, along with everything else in the spectrum.

That is why PCA is run on the correlation matrix, and "standardise your columns first" is the one piece of advice everybody gives. But it raises the question this episode is about. If a change of units can move the spectrum that far, what does the spectrum have to do with the model — and what is a method like ridge, which also modifies the spectrum, actually doing to it?""")

    post.add(
        "First, the diagnostic that cannot see any of this",
        f"""Before ridge, the thing ridge is usually reached for. Collinearity has a standard diagnostic — the variance inflation factor, `1/(1 - R²ⱼ)` from regressing each predictor on the others — and a standard threshold: above 10, worry.

Here is episode two's exercise design again. An intercept, a duration in seconds, a probability, and an amount of money, all mutually uncorrelated.

{snip['vif'].markdown()}

Every VIF is at {units['vif'].max():.4f}. Not "acceptable" — that is the smallest number the statistic can take, and it is what it returns when the columns are perfectly uncorrelated, which they are. Meanwhile the condition number is {units['kappa']:.2e}, and by episode one's accounting the normal equations for this design consume more digits than a double has.

The VIF is not broken. It is answering a different question, and answering it correctly: *how much do the other columns inflate this coefficient's variance*. That question is deliberately invariant to how each column is scaled, and it excludes the intercept. Both choices are defensible, and both are exactly why it cannot see a problem made of scales and means.

The reverse also happens.""",
        figures=[figs["t1"]])

    post.add(
        "",
        f"""The second design is eight columns that really are near-duplicates of two underlying directions. Its largest VIF is {coll['vif'].max():.0f} — eighty times the threshold — and its condition number is {coll['kappa']:.0f}, which costs about {2 * np.log10(coll['kappa']):.1f} digits out of 15.7. Uncomfortable, entirely survivable.

So the two statistics disagree in both directions, and it is not that one of them is unreliable. A VIF of {coll['vif'].max():.0f} is a true statement that one coefficient's variance is inflated eight hundred-fold, which matters enormously if that coefficient is your result and not at all if you only want predictions. A condition number of {units['kappa']:.1e} is a true statement that the arithmetic has no digits left, which matters whatever you wanted. **Run both. They cost one line each and they are not substitutes.**""",
        level=3)

    post.add(
        "What ridge actually does",
        f"""Ridge is introduced as a penalty: minimise ‖*y* − *Xβ*‖² + *α*‖*β*‖². That is correct and it explains nothing, because the mechanism is one substitution.

Write the design by its singular value decomposition, *X* = *UΣV*ᵗ. Then the ridge solution is

$$
\\hat\\beta_{{\\alpha}} = V \\, \\mathrm{{diag}}\\!\\left(\\frac{{s_i}}{{s_i^{{2}} + \\alpha}}\\right) U^{{\\top}} y
$$

and comparing it with least squares, which is the same expression with 1/*sᵢ* in the middle, the entire difference is a multiplier per direction:

$$
\\frac{{s_i^{{2}}}}{{s_i^{{2}} + \\alpha}}
$$

That number is between 0 and 1, it is close to 1 whenever *sᵢ*² ≫ *α*, and it falls towards 0 when *sᵢ*² ≪ *α*. **Ridge does not shrink the coefficient vector. It shrinks the directions, by different amounts, and the amount depends on how well the design measured each one.**

{snip['ridge'].markdown()}

That is the whole of it, and it is why the method works: the directions it destroys are the ones carrying almost no information, and the price of a small bias there buys a large reduction in variance. It is also why the price is invisible, because nothing in a coefficient table is indexed by direction.""",
        figures=[figs["f0"]])

    post.add(
        "The parameters you spent, and the ones your output reports",
        f"""Add those multipliers up and you get the trace of the ridge hat matrix, which is the number of parameters the fit actually used:

$$
\\mathrm{{df}}(\\alpha) \\;=\\; \\sum_i \\frac{{s_i^{{2}}}}{{s_i^{{2}} + \\alpha}}
$$

At *α* = 0 it is *p*. As *α* grows it falls smoothly, and there is nothing discrete about it — a direction can count as 0.3 of a parameter.

On the near-duplicate design, cross-validation picks *α* = {cv['alpha']:.2f}, and at that penalty the fit spends **{cv['effective_df']:.2f} of its {ridge['p']} parameters**.""",
        figures=[figs["f1"]])

    post.add(
        "",
        f"""Look at where it went. {kept} of the {ridge['p']} directions pass through essentially untouched; the remaining {ridge['p'] - kept} are cut to under half, most of them to under a tenth. The object that comes out is a {kept}-parameter fit wearing {ridge['p']} coefficients.

Now consider everything printed beside it. A residual degree-of-freedom count of *n* − {ridge['p']}. A standard error using that count. An AIC or BIC with a penalty of {ridge['p']} parameters. An adjusted *R*² correcting for {ridge['p']}. Every one of those is wrong by the same factor, and every one of them is wrong in the optimistic direction, because {cv['effective_df']:.2f} < {ridge['p']}.

The fix is not difficult — `df(α)` is four lines from the singular values, and it belongs wherever *p* currently sits. The difficulty is that no library prints it next to the coefficients.""",
        figures=[figs["f2"]],
        level=3)

    post.add(
        "Which is the question episode two left open",
        f"""Episode two ended on a rank-deficient design and a threshold called `rcond`: the singular value below which `lstsq` treats a direction as numerically zero and discards it. It was described there as a modelling decision disguised as a numerical tolerance, because it decides *how nearly collinear is too collinear*, and the question was deferred to here.

Here is the answer. Truncating the SVD at rank *k* and ridge at a penalty *α* are the same decision, taken with a cliff and with a ramp. Truncation multiplies each direction by 1 or by 0. Ridge multiplies it by *sᵢ*²/(*sᵢ*² + *α*), which is 1 and 0 with a slope in between. Put them on the same axis — parameters spent — and they can be compared, because `df(α)` is exactly what makes a penalty and a rank commensurable.

Both bottom out at {res['real_directions']} parameters, which is the number of directions this design has: {res['real_directions']} singular values above {s[res['real_directions'] - 1]:.0f} and {ridge['p'] - res['real_directions']} below {s[res['real_directions']]:.2f}. Near that point the two are within a few percent of each other. They diverge only at the ends, where the budget is far from what the data supports and neither answer is any good.

So the choice between them is not a choice about how much to regularise — that is the same number either way — but about whether you want a hard rank or a soft one. A rank is easier to report and defend; a ramp is differentiable and does not put a discontinuity in your cross-validation curve.""",
        figures=[figs["f4"]])

    post.add(
        "And whether the interval means anything",
        f"""There is an exact variance formula for the ridge estimator. With *W* = (*X*ᵗ*X* + *αI*)⁻¹ it is *σ*²*W X*ᵗ*X W*, it is not an approximation, and simulating from the design reproduces it to within a few percent. So an interval built from it should be fine.

It is fine, sometimes. Put the truth along the design's strongest direction and nominal 95 percent intervals cover {strong_at_cv['median_coverage']:.0%} at the cross-validated *α*. Put the same-sized truth along the weakest direction — same design, same formula, same *α* — and they cover {at_cv['median_coverage']:.0%}, with a bias of {at_cv['max_bias_se']:.1f} standard errors on the worst coefficient.""",
        figures=[figs["f3"]])

    post.add(
        "",
        """The variance formula is not at fault, and this is the part worth sitting with. It describes the spread of the estimator around **its own expectation**, and ridge's expectation is deliberately not the truth — that is what "biased estimator" means, and the bias is the term no variance formula contains. Where the data is strong the shrinkage is negligible and so is the bias; where it is weak the shrinkage is nearly total and the estimate is pulled to zero regardless of where the truth was.

Which case you are in depends on how your *β* sits relative to your design's singular directions, and that is not something you can look up. It is the quantity you were trying to estimate.

This is the fourth episode where the reported check is silent about the actual failure, and it is the first one where the check is not simply blind. The residual in episode one, the orthogonality condition in episode two, the bootstrap in episode four — those are wrong in every case. This one is *right most of the time*, which is worse, because a diagnostic that fails loudly gets fixed and one that fails occasionally gets trusted.""",
        level=3)

    post.add(
        "One more thing the opening exercise was about",
        f"""The penalty is ‖*β*‖², which adds up the squares of the coefficients. Coefficients are in the units of one over their column, so that sum is adding numbers that are not commensurable unless the columns are.

Take the four-column design from the top of this episode. Its duration column has a standard deviation of about {sc['sd'][1]:,.0f}, its probability column about {sc['sd'][2]:.2f}, and its money column about {sc['sd'][3]:,.3g}. Since the penalty presses on each coefficient in proportion to the square of that column's scale, the duration is penalised {sc['relative_penalty'][1]:.0e} as hard as the probability and the money {sc['relative_penalty'][3]:.0e} as hard — which is to say, not at all. Its coefficient is around 1e-8 because its column is around 1e7, and squaring a number that small contributes nothing to a sum containing a coefficient of order 1. So on this design, ridge is a penalty on the probability column and the intercept, and a rounding error everywhere else.

That is not a subtlety, it is the same units problem the opening exercise made about the eigenvalues, arriving in the penalty instead of in the spectrum. Standardise the columns before you regularise, do not penalise the intercept, and put the coefficients back in their original units afterwards if anybody has to read them.""",
        figures=[])

    post.add(
        "What to take away, and what is still hiding",
        """Four things.

**Run the condition number alongside the VIF.** They answer different questions and disagree in both directions. One line each.

**Read ridge as a per-direction multiplier.** `s**2 / (s**2 + alpha)` from the singular values tells you exactly what the penalty did, which no coefficient table can.

**Report `df(α)`, not `p`.** `sum(s**2 / (s**2 + alpha))`. Then put it wherever *p* was: residual degrees of freedom, AIC, adjusted *R*².

**And treat a ridge interval as a statement about prediction, not about a coefficient.** Where the design is strong it means what it says. Where it is weak the estimate is shrunk towards zero by construction and the interval follows it there.

One thing this episode has quietly assumed. Every statement above has been about *directions* — properties of the columns, of *X*ᵗ*X*, of the spectrum. Nothing has depended on any particular **row**. That is a reasonable assumption when no row is special, and the assumption fails more easily than it sounds: the diagonal of the projection matrix *X*(*X*ᵗ*X*)⁻¹*X*ᵗ has to sum to *p*, so its average entry is *p*/*n* — and there is nothing stopping a single entry from being 1. A row with leverage 1 has a residual of exactly zero, contributes nothing to any residual-based diagnostic, and owns its own fitted value completely. Next episode.

*Exercise.* Build a design with an intercept, two ordinary columns, and a dummy variable that is 1 for exactly one row out of a thousand. Compute the diagonal of the hat matrix. What is the leverage of that row, what is its residual, and what happens to Cook's distance — which divides by the residual? Then delete the row and refit, and see which coefficient moves. The answer is at the top of episode six.""")

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
