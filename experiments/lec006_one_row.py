"""Linear Algebra 6: One Row Can Own the Fit.

Episodes one to five were about columns. This one is about a row, and it rests on
one identity and one consequence.

The identity: the leverages sum to `p`, always, so the average is `p/n` and no
data can change that. The consequence: nothing stops a single entry being 1, and
a row at leverage 1 has a residual of exactly zero -- so it vanishes from every
diagnostic built on residuals, including the ones named after influence.

The clean case is the rare dummy episode two set aside as "not a conditioning
problem". A category with `k` members gives its rows leverage of at least `1/k`;
a category with one member gives leverage exactly 1, a residual at machine zero,
Cook's distance 0/0, and a coefficient that does not survive deleting the row.

Then the part that stops this being a story about pathological designs: leverage
is not influence. A row 60 times above average leverage can move a coefficient by
a tenth of a standard error. Influence needs the row to be far out in x *and* off
the line in y, and when both arrive -- a lognormal column and ordinary noise are
enough -- one observation moves a slope by six standard errors.

Run: `standarderror run lec006_one_row --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.linalg import leverage as lv
from standarderror.linalg import ridge as rg
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
POST_DATE = date(2026, 9, 1)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Linear Algebra for Data Science, Taught Through What Breaks"
SERIES_TAG = "Linear Algebra"

#: Episode five's exercise: a thousand rows and a dummy true for one of them.
N = 1000
EXTRA = 2
TRUE_BETA = (1.0, 2.0, -1.0, 3.0)

#: The category-size sweep, to show 1/k rather than assert it.
K_SWEEP = (1, 2, 3, 5, 10, 25, 50, 100, 200)

#: The design where leverage arrives naturally: a skewed column, of the kind an
#: income, a firm size or a transaction value actually is.
SKEW_N = 500
MILD_SIGMA = 1.6
MILD_SEED = 103
HEAVY_SIGMA = 2.4
HEAVY_SEED = 201
SKEW_BETA = (1.0, 0.5, -0.3)
Y_ERRORS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

#: Two rows in one category, one of them badly wrong: the masking case.
MASK_ROWS = (100, 700)
MASK_ERROR = 6.0

#: Episode five's penalty, applied to a category of one.
POOL_ALPHAS = (0.0, 0.5, 2.0, 10.0)

SEED = 41


def _skew_design(seed: int, sigma: float):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(0.0, sigma, SKEW_N)
    X = np.column_stack([np.ones(SKEW_N), x, rng.standard_normal(SKEW_N)])
    beta = np.array(SKEW_BETA)
    return X, X @ beta + rng.normal(0.0, 1.0, SKEW_N), beta


def compute() -> dict:
    out: dict = {}

    # --- episode five's exercise -------------------------------------------
    rng = np.random.default_rng(SEED)
    X = lv.rare_dummy_design(N, 1, rng=rng, extra=EXTRA)
    y = X @ np.array(TRUE_BETA) + rng.normal(0.0, 1.0, N)
    rep = lv.leverage_report(X)
    i = int(rep.saturated[0])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    out["saturated"] = {
        "X": X, "y": y, "row": i, "report": rep,
        "leverage": float(rep.h[i]),
        "mean_leverage": rep.mean,
        "ratio": rep.max_ratio,
        "residual": float(y[i] - X[i] @ beta),
        "typical_residual": float(np.median(np.abs(y - X @ beta))),
        "cook": float(lv.cook_distance(X, y)[i]),
        "dfbeta": lv.dfbeta(X, y)[i],
        "deletion": lv.deletion_refit(X, y, i),
    }

    # --- one over k --------------------------------------------------------
    out["sweep"] = lv.leverage_sweep(K_SWEEP, n=N,
                                     rng=np.random.default_rng(SEED + 1),
                                     extra=EXTRA)
    out["p"] = X.shape[1]

    # --- masking: two rows in one category, only one of them wrong ---------
    r2 = np.random.default_rng(SEED)
    dm = np.zeros(N)
    dm[list(MASK_ROWS)] = 1.0
    Xm2 = np.column_stack([np.ones(N), r2.standard_normal(N),
                           r2.standard_normal(N), dm])
    ym2 = Xm2 @ np.array(TRUE_BETA) + r2.normal(0.0, 1.0, N)
    ym2[MASK_ROWS[0]] += MASK_ERROR
    full = np.linalg.lstsq(Xm2, ym2, rcond=None)[0]
    drops = {}
    for label, rows in (("guilty", [MASK_ROWS[0]]), ("innocent", [MASK_ROWS[1]]),
                        ("both", list(MASK_ROWS))):
        keep = np.ones(N, bool)
        keep[rows] = False
        drops[label] = float(np.linalg.lstsq(Xm2[keep], ym2[keep],
                                             rcond=None)[0][3])
    out["mask"] = {
        "leverage": lv.hat_diagonal(Xm2)[list(MASK_ROWS)],
        "dfbeta": np.abs(lv.dfbeta(Xm2, ym2)[list(MASK_ROWS), 3]),
        "cook": lv.cook_distance(Xm2, ym2)[list(MASK_ROWS)],
        "full": float(full[3]), "drops": drops, "truth": TRUE_BETA[3],
        "error": MASK_ERROR,
    }
    out["mask"]["cook_gap"] = float(abs(out["mask"]["cook"][0]
                                       - out["mask"]["cook"][1]))
    out["mask"]["dfbeta_gap"] = float(abs(out["mask"]["dfbeta"][0]
                                          - out["mask"]["dfbeta"][1]))

    # --- and what episode five's penalty does to a saturated row -----------
    r3 = np.random.default_rng(SEED)
    Xp = lv.rare_dummy_design(N, 1, rng=r3, extra=EXTRA)
    yp = Xp @ np.array(TRUE_BETA) + r3.normal(0.0, 1.0, N)
    ip = int(np.argmax(lv.hat_diagonal(Xp)))
    out["pooling"] = [
        {"alpha": float(a),
         "coefficient": float(rg.ridge_fit(Xp, yp, a).coefficients[3]),
         "residual": float(yp[ip] - Xp[ip] @ rg.ridge_fit(Xp, yp, a).coefficients)}
        for a in POOL_ALPHAS]

    # --- leverage is not influence -----------------------------------------
    Xm, ym, bm = _skew_design(MILD_SEED, MILD_SIGMA)
    hm = lv.hat_diagonal(Xm)
    hi = int(np.argmax(hm))
    mid = int(np.argsort(hm)[len(hm) // 2])
    ladder = []
    for err in Y_ERRORS:
        row = {"error": float(err)}
        for name, j in (("high", hi), ("typical", mid)):
            yy = ym.copy()
            yy[j] = Xm[j] @ bm + err
            row[name] = float(abs(lv.dfbeta(Xm, yy)[j, 1]))
        ladder.append(row)
    out["mild"] = {
        "X": Xm, "y": ym, "h": hm, "high": hi, "typical": mid,
        "high_leverage": float(hm[hi]), "high_ratio": float(hm[hi] / (3 / SKEW_N)),
        "typical_leverage": float(hm[mid]),
        "as_drawn": float(abs(lv.dfbeta(Xm, ym)[hi, 1])),
        "dfbeta": np.abs(lv.dfbeta(Xm, ym)[:, 1]),
        "ladder": ladder,
    }

    # --- and when both arrive together -------------------------------------
    Xh, yh, _ = _skew_design(HEAVY_SEED, HEAVY_SIGMA)
    hh = lv.hat_diagonal(Xh)
    j = int(np.argmax(hh))
    out["heavy"] = {
        "leverage": float(hh[j]), "ratio": float(hh[j] / (3 / SKEW_N)),
        "dfbeta": float(abs(lv.dfbeta(Xh, yh)[j, 1])),
        "cook": float(lv.cook_distance(Xh, yh)[j]),
        "x": float(Xh[j, 1]), "x_median": float(np.median(Xh[:, 1])),
        "row": j,
    }
    return out


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    out: dict = {}
    sat, sweep, mild = res["saturated"], res["sweep"], res["mild"]
    rep = sat["report"]

    # --- f0: the identity, and the one row that breaks the average ---------
    def spike(ax, m):
        ax.plot(np.arange(len(rep.h)), rep.h, lw=0.9, color=m.series[0],
                label="leverage of each row")
        ax.axhline(rep.mean, color=m.muted, lw=1.6, ls=(0, (4, 3)),
                   label=f"p/n = {rep.mean:g}, and this is an identity")
        ax.annotate(f"h = {sat['leverage']:.0f}", (sat["row"], sat["leverage"]),
                    xytext=(-18, -14), textcoords="offset points",
                    fontsize=9, color=m.ink)
        ax.set_ylim(-0.05, 1.1)

    out["f0"] = charts.diagram(
        spike,
        title="The leverages always sum to p, and one row can take a whole unit",
        subtitle=(f"Diagonal of the projection matrix for {res['p']} columns and "
                  f"{rep.n} rows, one of which carries a dummy true only for it."),
        xlabel="row", ylabel="leverage",
        source="Simulated; standarderror/linalg/leverage.py.",
        alt=("A flat line of tiny values just above zero with a single spike "
             "reaching one, and a dashed reference line near zero."),
        caption=(f"The dashed line is not an average of anything measured — "
                 f"trace(H) = p, so the mean is {rep.mean:g} whatever the data. "
                 f"The spike is {rep.max_ratio:.0f} times it, and every other "
                 f"row is pushed below the mean to pay for it."),
        path=str(IMG / f"lec06-f0-spike.{EXT}"))[0]

    # --- f1: one over k ----------------------------------------------------
    out["f1"] = charts.lines(
        pd.DataFrame({"measured leverage of the category's rows":
                          [r["leverage"] for r in sweep],
                      "1/k": [r["one_over_k"] for r in sweep]},
                     index=[r["k"] for r in sweep]),
        title="A category with k members hands its rows 1/k of the fit",
        subtitle=(f"Leverage of the rows carrying a dummy variable, against how "
                  f"many rows carry it, in a design of {res['p']} columns and "
                  f"{rep.n} rows."),
        xlabel="rows in the category (k)", ylabel="leverage",
        source="Simulated; standarderror/linalg/leverage.py.",
        logx=True, logy=True,
        alt=("Two curves falling together on log axes, separating slightly at "
             "the right-hand end."),
        caption=("Exactly 1/k from the dummy, plus about (p−2)/n from the other "
                 "columns — which is why the two curves part company only once "
                 "1/k has fallen to the same order as p/n. A category of one is "
                 "the left-hand end of an ordinary curve, not a special case."),
        path=str(IMG / f"lec06-f1-one-over-k.{EXT}"))[0]

    # --- t1: what every diagnostic reports about the most influential row ---
    rep = sat["report"]
    d = sat["deletion"]
    rows = [
        ["leverage", f"{sat['leverage']:.6f}", f"{rep.max_ratio:.0f}× the mean"],
        ["residual", f"{sat['residual']:.1e}",
         f"typical row: {sat['typical_residual']:.2f}"],
        ["Cook's distance", "undefined (0/0)", "reported as 0 by most software"],
        ["DFBETA", "undefined", "deletion is not a perturbation here"],
        ["rank of X", f"{d['rank_before']}", f"{d['rank_after']} without this row"],
    ]
    out["t1"] = charts.table_image(
        rows,
        header=["measure", "value at that row", "for comparison"],
        title="Every influence measure is silent about the most influential row",
        subtitle=("The single observation carrying a dummy variable that is true "
                  "only for it, in a design of 1,000 rows."),
        source="Simulated; standarderror/linalg/leverage.py.",
        bold_cols=(1,),
        alt=("Five-row table of leverage, residual, Cook's distance, DFBETA and "
             "rank, three of which are undefined or zero."),
        caption=("The residual is zero because the fit passes exactly through "
                 "the point, and every measure built on residuals inherits that "
                 "zero. The last row is what actually happens if you take the "
                 "advice to drop the influential observation."),
        path=str(IMG / f"lec06-t1-silent.{EXT}"))[0]

    # --- f2: leverage is not influence -------------------------------------
    def scatter(ax, m):
        ax.scatter(mild["h"], mild["dfbeta"], s=18, color=m.series[0],
                   alpha=0.75, label="one row")
        ax.axvline(2 * 3 / SKEW_N, color=m.muted, lw=1.4, ls=(0, (4, 3)),
                   label="the usual 2p/n threshold for 'high leverage'")
        ax.annotate("60× the mean leverage,\nand it moves the slope by "
                    f"{mild['as_drawn']:.2f} SE",
                    (mild["high_leverage"], mild["as_drawn"]),
                    xytext=(-150, 34), textcoords="offset points", fontsize=8.5,
                    color=m.ink,
                    arrowprops=dict(arrowstyle="-", color=m.muted, lw=1.0))

    out["f2"] = charts.diagram(
        scatter,
        title="Leverage says a row could matter, not that it does",
        subtitle=(f"Every row of a {SKEW_N}-row design with one skewed column: "
                  f"its leverage against how far deleting it moves the slope."),
        xlabel="leverage", ylabel="|DFBETA| on the slope, in standard errors",
        source="Simulated; standarderror/linalg/leverage.py.",
        alt=("A scatter where the right-most points, at the highest leverage, "
             "are not the highest points."),
        caption=("The rows furthest to the right are not the rows furthest up. "
                 "Leverage is a property of x alone; influence needs the "
                 "observation to be unusual in x **and** off the line in y, and "
                 "most high-leverage rows are neither surprising nor wrong."),
        path=str(IMG / f"lec06-f2-not-influence.{EXT}"))[0]

    # --- f3: the same error, at two leverages ------------------------------
    out["f3"] = charts.lines(
        pd.DataFrame(
            {f"a row at leverage {mild['high_leverage']:.2f}":
                 [r["high"] for r in mild["ladder"]],
             f"a row at leverage {mild['typical_leverage']:.4f}":
                 [r["typical"] for r in mild["ladder"]]},
            index=[r["error"] for r in mild["ladder"]]),
        title="The same discrepancy, priced at two positions",
        subtitle=("One observation moved off the true line by the amount on the "
                  "x-axis, and what that does to the slope, at high and at "
                  "typical leverage."),
        xlabel="how far that observation sits off the line, in noise standard deviations",
        ylabel="|DFBETA| on the slope, in standard errors",
        source="Simulated; standarderror/linalg/leverage.py.",
        logy=True,
        alt=("Two rising lines on a log y-axis, separated by more than two "
             "orders of magnitude."),
        caption=(f"At the top of the range the discrepancy is the same size in "
                 f"both cases and its effect differs by a factor of "
                 f"{mild['ladder'][-1]['high'] / mild['ladder'][-1]['typical']:.0f}. "
                 f"A 3σ outlier at typical leverage is a rounding error; at high "
                 f"leverage it is most of a conclusion."),
        path=str(IMG / f"lec06-f3-ladder.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    sat, heavy = res["saturated"], res["heavy"]
    rep = sat["report"]

    def spike(panel, m):
        h = rep.h
        panel.plot(np.arange(len(h)), np.clip(h, 0, 1), lw=1.0, color=m.ink)
        panel.axhline(rep.mean, color=m.grid, lw=1.6)
        panel.set_ylim(-0.05, 1.12)

    def flat_zero(panel, m):
        x = np.linspace(0.06, 0.94, 40)
        rng = np.random.default_rng(3)
        panel.plot([0.02, 0.98], [0.5, 0.5], color=m.grid, lw=1.6)
        panel.plot(x, 0.5 + rng.normal(0, 0.11, x.size), color=m.grid, lw=1.4)
        panel.plot([0.5], [0.5], marker="o", ms=9, color=m.ink)
        panel.set_xlim(0, 1); panel.set_ylim(0, 1)

    def two_slopes(panel, m):
        x = np.linspace(0.05, 0.95, 20)
        panel.plot(x, 0.30 + 0.28 * x, color=m.grid, lw=2.0)
        panel.plot(x, 0.30 + 0.62 * x, color=m.ink, lw=2.4)
        panel.plot([0.93], [0.30 + 0.62 * 0.93], marker="o", ms=8, color=m.ink)
        panel.set_xlim(0, 1); panel.set_ylim(0, 1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=6,
        headline="A residual of zero from the row that owns the fit",
        panels=[
            (spike, f"{rep.max_ratio:.0f}×", "one row's share"),
            (flat_zero, f"{sat['residual']:.0e}", "its residual"),
            (two_slopes, f"{heavy['dfbeta']:.1f} SE", "one row moves a slope"),
        ],
        note=("The leverages of a design always sum to the number of columns, so "
              "the average is p/n and nothing can change it — but a single row "
              "can take a whole unit. At leverage 1 the fit passes exactly "
              "through the observation, its residual is zero, and Cook's "
              "distance is 0/0. Leverage alone is not influence; when a skewed "
              "column and ordinary noise put both together, one observation "
              "carries six standard errors of a slope."),
        alt=("A three-panel hand-drawn strip. The first shows a flat line near "
             "zero with one tall spike. The second shows a scatter about a line "
             "with one point sitting exactly on it. The third shows two lines "
             "from a common origin pulled apart by a single distant point."),
        mode="light",
        path=str(IMG / f"lec06-hero.{EXT}"))[0]


# ---------------------------------------------------------------- the post

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["exercise"] = s.run(f"""
        import numpy as np

        # Episode five's exercise. An intercept, two ordinary columns, and a
        # dummy that is true for exactly one row out of {N}.
        rng = np.random.default_rng({SEED})
        n = {N}
        d = np.zeros(n); d[rng.integers(n)] = 1.0
        X = np.column_stack([np.ones(n), rng.standard_normal(n),
                             rng.standard_normal(n), d])
        y = X @ np.array({list(TRUE_BETA)}) + rng.normal(0, 1.0, n)

        # The hat diagonal, without forming an inverse -- episode one's rule.
        Q = np.linalg.qr(X)[0]
        h = np.einsum("ij,ij->i", Q, Q)
        i = int(np.argmax(h))

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta

        print(f"leverages sum to        {{h.sum():.6f}}   (p = {{X.shape[1]}})")
        print(f"mean leverage           {{h.mean():.6f}}   (p/n)")
        print(f"leverage of row {{i}}     {{h[i]:.6f}}")
        print(f"its residual            {{resid[i]:+.2e}}")
        print(f"a typical residual      {{np.median(abs(resid)):+.2f}}")
    """, expect=["leverages sum to", "its residual"])

    out["delete"] = s.run("""
        # The usual advice: drop the influential observation and refit.
        keep = np.ones(len(y), bool); keep[i] = False
        print(f"rank of X            {np.linalg.matrix_rank(X)}")
        print(f"rank without row {i} {np.linalg.matrix_rank(X[keep])}")

        # Cook's distance has the residual on top and (1 - h) underneath.
        s2 = (resid**2).sum() / (len(y) - X.shape[1])
        print(f"Cook numerator       {resid[i]**2 * h[i]:.3e}")
        print(f"Cook denominator     {X.shape[1] * s2 * (1 - h[i])**2:.3e}")
    """, expect=["rank of X", "Cook numerator"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    sat, mild, heavy = res["saturated"], res["mild"], res["heavy"]

    # The spine, asserted rather than trusted.
    assert abs(sat["leverage"] - 1.0) < 1e-9, sat
    assert abs(sat["residual"]) < 1e-8, sat
    assert np.isnan(sat["cook"]), sat
    assert sat["deletion"]["lost_rank"], sat["deletion"]
    assert mild["high_ratio"] > 20 and mild["as_drawn"] < 1.0, mild
    assert heavy["dfbeta"] > 3.0, heavy

    post = Post(
        title=f"{SERIES_TAG} 6: One Row Can Own the Fit",
        slug="linear-algebra-6-one-row",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=6,
        prerequisites=["linear-algebra-5-ridge-geometry"],
        date=POST_DATE,
        subtitle=("A single observation with a residual of exactly zero, a "
                  "Cook's distance of 0/0, and a coefficient that stops existing "
                  "when you delete it — and, separately, one row moving a slope "
                  "by six standard errors."),
        summary=("The diagonal of the projection matrix always sums to the "
                 "number of columns, so average leverage is p/n and no data can "
                 "change it — but nothing stops one row taking a whole unit. A "
                 "row at leverage 1 has a residual of exactly zero, which means "
                 "every diagnostic built on residuals reports nothing about the "
                 "most influential observation in the design, and a category "
                 "with one member produces exactly that. The second half is the "
                 "distinction that matters in ordinary data: leverage says a row "
                 "could matter and influence needs it to be unusual in x and "
                 "wrong in y, and the same discrepancy is worth four hundred "
                 "times more at one position than another."),
        tags=["linear-algebra", "regression-diagnostics", "leverage",
              "lectures", "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every design here is constructed in the episode "
            "and every number is produced by the code shown, executed when this "
            "page was built.",
            "Machinery: `standarderror/linalg/leverage.py`, tested in "
            "`tests/test_leverage.py`.",
            "Where this stops: Belsley, Kuh and Welsch, *Regression Diagnostics* "
            "(1980), chapters 2 and 3; Cook and Weisberg, *Residuals and "
            "Influence in Regression* (1982); Hoaglin and Welsch, \"The hat "
            "matrix in regression and ANOVA\", *The American Statistician* 32 "
            "(1978).",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": (f"{N} rows for the dummy designs, {SKEW_N} for the "
                           f"skewed ones; no observation is placed by hand except "
                           f"where the text says a discrepancy was introduced"),
            "determinism": f"one seed, {SEED}, and every draw derived from it",
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    sat, sweep, mild, heavy = res["saturated"], res["sweep"], res["mild"], res["heavy"]
    rep = sat["report"]
    d = sat["deletion"]
    top = mild["ladder"][-1]
    ratio_3sd = top["high"] / top["typical"]

    post.add(
        "Last episode's exercise",
        f"""The exercise was: build a design with an intercept, two ordinary columns, and a dummy variable true for exactly one row in a thousand. Then look at the leverage of that row, its residual, and Cook's distance.

{snip['exercise'].markdown()}

Three numbers, and each is worth its own sentence.

The leverages sum to {res['p']}, which is the number of columns. That is not a property of this data — it is an identity, and it is the whole reason the rest of the episode works.

The leverage of that one row is {sat['leverage']:.0f}. Not "high": one, exactly, which is the largest value the quantity can take.

And its residual is {sat['residual']:.0e}, against a typical residual of {sat['typical_residual']:.2f} elsewhere in the same fit. The line passes exactly through the point.""")

    post.add(
        "Why the sum is fixed, and what that costs",
        f"""The fitted values are *Xβ̂* = *Hy* where *H* = *X*(*X*ᵗ*X*)⁻¹*X*ᵗ is the projection onto the column space, and *hᵢ* — the *i*-th diagonal entry — is how much of the *i*-th fitted value comes from the *i*-th observation. Its trace is

$$
\\mathrm{{tr}}(H) \\;=\\; \\mathrm{{tr}}\\!\\left( (X^{{\\top}}X)^{{-1}} X^{{\\top}}X \\right) \\;=\\; p
$$

using only that the trace does not care about the order of a product. So the leverages sum to *p* no matter what the data looks like, the mean is *p*/*n* = {rep.mean:g} here, and every threshold you have seen — 2*p*/*n*, 3*p*/*n* — is a multiple of a quantity you already know before collecting anything.

The consequence is the part worth noticing. It is a *fixed budget*. A row at leverage {sat['leverage']:.0f} has taken a whole unit out of a total of {res['p']}, and every other row is pushed below the mean to pay for it.""",
        figures=[figs["f0"]])

    post.add(
        "A category with one member",
        f"""That design was not a contrivance. A dummy variable true for *k* rows gives each of those rows leverage of at least 1/*k*, and the reason is almost tautological: within that category, the model has one free parameter and *k* observations to spend it on.

At *k* = {sweep[1]['k']} it is {sweep[1]['leverage']:.3f}. At {sweep[4]['k']} it is {sweep[4]['leverage']:.3f}. At {sweep[-1]['k']} it is {sweep[-1]['leverage']:.4f}, which is where the other columns' contribution starts to be the larger part.

And at *k* = 1 it is exactly 1, because the model has a parameter whose only evidence is that row.""",
        figures=[figs["f1"]])

    post.add(
        "",
        """This is the rare dummy episode two set aside. There, standardising a design with a 1 percent dummy left the condition number at 1.08, and the conclusion was that a rare category is not a conditioning problem. It is not. Nothing about *X*ᵗ*X* is ill-behaved; the design is perfectly well conditioned and the solve is exact. The problem is entirely on the other axis of the matrix, and it does not show up in any spectrum.

It is also not rare. Any categorical variable with a long tail — a country field, a product code, a diagnosis, a merchant ID — produces categories of size one by the dozen, and one-hot encoding turns each of them into a column whose only evidence is a single row.""",
        level=3)

    post.add(
        "Where every diagnostic goes quiet",
        f"""Now run the standard influence checks on the most influential observation in the design.

{snip['delete'].markdown()}

Cook's distance is *eᵢ*² *hᵢ* / (*p s*² (1 − *hᵢ*)²). At *hᵢ* = 1 the numerator has the squared residual in it, which is zero, and the denominator has (1 − *hᵢ*)², which is also zero. The statistic is undefined. What software prints depends on which underflows first, and in practice that is usually the numerator — so the most influential row in your design is reported as having no influence at all.

DFBETA is worse in an interesting way. Its closed form divides by (1 − *hᵢ*), so it is undefined too, but the reason is substantive rather than numerical: **deleting this row is not a perturbation of the fit.** Delete it and the dummy column becomes all zeros, the rank drops from {d['rank_before']} to {d['rank_after']}, and the coefficient does not move to a different value — it stops existing.""",
        figures=[figs["t1"]])

    post.add(
        "Two members, and they hide each other",
        f"""One more thing the extreme case makes visible before we leave it. Give the category {len(res['mask']['leverage'])} members instead of one, and put a {res['mask']['error']:.0f}σ error into exactly one of them.

Both rows now sit at leverage {res['mask']['leverage'][0]:.4f} — equal to fifteen decimal places, because leverage never looks at *y* and the two rows are interchangeable in *X*. Their Cook's distances agree to {abs(int(np.log10(res['mask']['cook_gap']))) - 1} decimals as well, {res['mask']['cook'][0]:.4f} against {res['mask']['cook'][1]:.4f}. Their DFBETAs differ by {res['mask']['dfbeta_gap']:.1e} — {res['mask']['dfbeta'][0]:.3f} against {res['mask']['dfbeta'][1]:.3f}, a gap of {100 * res['mask']['dfbeta_gap'] / res['mask']['dfbeta'][0]:.2f} percent.

Only one of them is wrong, and that is what the diagnostics have to say about which.

The diagnostics cannot separate them, and it is not a limitation of the arithmetic. Deleting either row leaves one observation to determine the coefficient, so either deletion moves it by the full distance between the two. They are symmetric partners in the same parameter, and a measure defined by "what happens if I remove this row" cannot tell a guilty row from the innocent one it is paired with.

What the deletions actually give is worth reading. The fit with both rows puts the coefficient at {res['mask']['full']:+.2f} against a truth of {res['mask']['truth']:+.1f}. Drop the wrong row and it goes to {res['mask']['drops']['guilty']:+.2f}, which is nearly right. Drop the innocent one and it goes to {res['mask']['drops']['innocent']:+.2f}, which is further from the truth than where you started. Drop both and it is {res['mask']['drops']['both']:+.1f}, because there is nothing left to estimate it from. One of those three is the correct action and nothing in the diagnostics tells you which.""")

    post.add(
        "But leverage is not influence",
        f"""So far this has been about an extreme case that makes the mechanism visible. The version you will actually meet is quieter, and the quiet version is where the mistake gets made — because the natural response to all of the above is to sort by leverage and look at the top.

Here is a {SKEW_N}-row design with one skewed column, the kind an income or a firm size or a transaction value actually is. No outlier is placed by hand; the column is lognormal and the noise is ordinary.

Its highest-leverage row sits at {mild['high_leverage']:.3f}, which is {mild['high_ratio']:.0f} times the mean and far beyond any threshold. Deleting it moves the slope by {mild['as_drawn']:.2f} standard errors.

That is not a failure of the leverage measure. It is what leverage means. *hᵢ* is computed from *X* alone — it never sees *y* — so it can only tell you that an observation is in a position to matter. Whether it does depends on whether it also disagrees with the rest of the data, and most unusual rows do not.""",
        figures=[figs["f2"]])

    post.add(
        "",
        f"""The two combine multiplicatively, which is what makes the position dangerous rather than merely unusual. The change in a coefficient from deleting row *i* is proportional to *eᵢ* /(1 − *hᵢ*): the residual, amplified by how much of its own fitted value the row supplied.

Take the same design and move one observation off the true line by a fixed amount, first at high leverage and then at typical leverage.

At a discrepancy of 3σ — a size you would see a few times in {SKEW_N} rows without anything being wrong — the high-leverage row moves the slope by {top['high']:.2f} standard errors and the ordinary row moves it by {top['typical']:.4f}. The same error, at two positions, worth **{ratio_3sd:.0f} times as much** at one as at the other.""",
        figures=[figs["f3"]],
        level=3)

    post.add(
        "",
        f"""And when both arrive together without anyone arranging it: a lognormal column with a heavier tail, ordinary noise, no hand-placed outlier. One row lands at leverage {heavy['leverage']:.3f} — {heavy['ratio']:.0f} times the mean — with an *x* of {heavy['x']:,.0f} against a median of {heavy['x_median']:.2f}. It moves the slope by **{heavy['dfbeta']:.2f} standard errors**.

A coefficient reported as six standard errors from zero, resting on one observation out of {SKEW_N}. Every number in that sentence came out of a simulation with no adversary in it.""",
        level=3)

    post.add(
        "Which is what episode five's penalty is for",
        f"""There is a fix for the saturated case, and it is the previous episode.

A category with one member is a direction the design barely measured — that is precisely what a singular value near zero means, and it is the direction ridge shrinks hardest. Apply episode five's penalty to the design with the category of one and watch both numbers move.

{chr(10).join(f"At α = {r['alpha']:.1f} the coefficient is {r['coefficient']:+.2f} and the residual at that row is {r['residual']:+.2f}." for r in res['pooling'])}

At α = 0 the coefficient is {res['pooling'][0]['coefficient']:+.2f}, resting entirely on one observation, and the residual is zero because the fit interpolates it. As the penalty grows the coefficient is pulled towards zero — and, more usefully, **the residual stops being zero**. The row no longer owns its own fitted value, so it re-enters every diagnostic that residuals feed.

That is what partial pooling is, arriving from the linear algebra rather than from the hierarchical-model literature. "This category has one member, so shrink its effect towards the overall mean" and "this direction has a small singular value, so shrink its coefficient" are the same instruction. Episode five described the mechanism; this is the case where you can see what it buys.""")

    post.add(
        "What to take away, and what is still hiding",
        """Four things.

**Compute the hat diagonal. It is one line and no inverse.** `Q = np.linalg.qr(X)[0]` and then `np.einsum("ij,ij->i", Q, Q)`. Compare it against *p*/*n*, which you already know.

**Count your categories before you encode them.** Any level with a handful of members hands those rows leverage near 1/*k*, and a level with one member hands it leverage 1. Pool the tail, or use a partial-pooling model, or accept that the coefficient is a restatement of one row — but decide, rather than discover.

**Never read a zero residual as a good fit.** It is the signature of leverage 1, and it takes Cook's distance and everything else built on residuals down with it.

**Regularise the rare levels rather than deleting their rows.** A penalty un-saturates the leverage, which puts the observation back into the diagnostics instead of removing it from the data.

**And do not sort by leverage.** Leverage is a property of *X* and influence needs *y* as well. Sort by DFBETA for the coefficient you care about, or by Cook's distance if you want a single number — but check the leverages for saturation first, because that is the case those measures cannot see.

One thing this episode assumed without saying so. Every fit here kept all *p* columns, and the question of *how many* to keep has been answered by fiat every time it came up: episode five compared a truncated SVD against ridge at matched degrees of freedom, and simply swept the rank rather than choosing one. That choice has a name and a theorem — Eckart and Young settled which rank-*k* matrix is closest to yours, and the answer is the obvious one — but the theorem says nothing about *which k*, and the scree plot everybody uses to decide has no more authority than the elbow somebody thinks they see in it. Next episode.

*Exercise.* Take a data matrix with a genuine low-rank structure plus noise — say rank 3 in 20 columns — and compute its singular values. Plot them and pick the elbow. Now double the noise and plot again, then halve the number of rows and plot again. How much does the elbow move, and is it moving with the rank or with something else? The answer is at the top of episode seven.""")

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
