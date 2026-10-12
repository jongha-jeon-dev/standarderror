"""ML 15: A Forest Cannot Predict Beyond the Largest Value It Has Seen.

The fifteenth episode of *Machine Learning, Taught Through What Breaks*,
closing Arc III. "Tree ensembles can approximate any function." Inside the
range of the training data, yes. Measured on a simulated trend and on the
diabetes data:

* Trained on x in [0, 1] with y rising as 2x, a random forest and histogram
  boosting are the best models inside the range (test MSE 0.12, against 0.51
  for linear regression) and the worst beyond it: at x in [1.5, 2] their
  error is 22 times larger, because their predictions stop at the largest y
  they saw.
* Fitting the trend linearly and the forest to the residuals is best
  everywhere: 0.11 inside and 0.12 beyond.
* On the diabetes data, a forest beats linear regression on a random split
  and loses to it by 40% on the patients with the highest bmi when trained
  without them.

Run: `standarderror run ml115_extrapolation --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import trees as tr
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 12)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

MODELS = ("forest", "boosting", "linear", "trend + forest")


def compute() -> dict:
    return {"ex": tr.extrapolation(), "diab": tr.diabetes_extrapolation()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    ex = res["ex"]
    cx, cv = ex["curve_x"], ex["curve"]

    def curves(ax, m):
        ax.axvspan(0, 1, color=m.grid, alpha=0.35, lw=0)
        ax.text(0.5, 4.3, "training range", ha="center", fontsize=9,
                color=m.ink)
        ax.plot(cx, cv["truth"], lw=1.6, ls="--", color=m.ink,
                label="true trend, 2x")
        for c, k in zip(m.series, MODELS):
            ax.plot(cx, cv[k], lw=2.2, color=c, label=k)
        ax.set_xlim(0, 2)
        ax.set_ylim(-0.5, 4.6)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    last = ex["rows"][-1]
    out["f0"] = charts.diagram(
        curves,
        title="Past the training range, the trees go flat",
        subtitle=("Predictions along x with the other inputs at zero, for "
                  "models trained on 500 rows with x between 0 and 1. The "
                  "truth keeps rising."),
        xlabel="x",
        ylabel="prediction",
        source="Simulated; standarderror/ml/trees.py.",
        alt=("A dashed rising line for the truth; the forest and boosting "
             "curves follow it to x = 1 and then run flat; the linear and "
             "trend-plus-forest curves keep rising with it."),
        caption=(f"The forest never predicts above "
                 f"{last['forest_max']:.2f}; the training labels peaked at "
                 f"{ex['train_y_max']:.2f}. At x between 1.5 and 2 the truth "
                 f"averages {last['truth_mean']:.2f}."),
        path=str(IMG / f"ml115-f0-curves.{EXT}"))[0]

    def bands(ax, m):
        rows = ex["rows"]
        xs = np.arange(len(rows))
        w = 0.2
        for i, (c, k) in enumerate(zip(m.series, MODELS)):
            ax.bar(xs + (i - 1.5) * w, [r[k] for r in rows], w, color=c,
                   label=k)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{a:g}–{b:g}" + (" (inside)" if b <= 1 else "")
                            for a, b in (r["band"] for r in rows)])
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    first = ex["rows"][0]
    out["f1"] = charts.diagram(
        bands,
        title="Best inside, worst outside",
        subtitle=("Test mean squared error on 2,000 new rows in each band of "
                  "x. Training data covered x from 0 to 1."),
        xlabel="band of x in the test rows",
        ylabel="test MSE (log scale)",
        source="Simulated; standarderror/ml/trees.py.",
        alt=("Grouped bars on a log scale: forest and boosting lowest in the "
             "training band and rising steeply in each band beyond; linear "
             "constant; trend plus forest lowest everywhere."),
        caption=(f"Inside the range the forest's error is {first['forest']:.2f} "
                 f"and the linear model's {first['linear']:.2f}. Between 1.5 "
                 f"and 2 the forest's is **{last['forest']:.2f}**. The hybrid "
                 f"stays at {last['trend + forest']:.2f}."),
        path=str(IMG / f"ml115-f1-bands.{EXT}"))[0]

    d = res["diab"]
    rows = []
    for split, tag in (("random", "random split"),
                       ("beyond", f"train bmi ≤ {d['quantile']:.0%} quantile, "
                                  f"test above")):
        for mdl in ("forest", "linear"):
            r = d[split][mdl]
            rows.append([tag if mdl == "forest" else "", mdl,
                         f"{r['mse']:,.0f}", f"{r['bias']:+.1f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["test set", "model", "test MSE", "mean error"],
        title="On real data the gap is smaller and the ranking flips",
        subtitle=(f"Diabetes progression, {d['test_rows']} test patients in "
                  f"each split. Mean error is mean prediction minus mean "
                  f"outcome."),
        source="Diabetes data (scikit-learn); standarderror/ml/trees.py.",
        alt=("A four-row table: on a random split the forest has the lower "
             "error; on the highest-bmi patients the linear model does, and "
             "the forest under-predicts by more."),
        caption=("The forest wins the random split and loses the "
                 "extrapolation, and its misses on the high-bmi patients are "
                 "systematically low."),
        bold_cells={(0, 2), (3, 2)},
        align="llrr",
        path=str(IMG / f"ml115-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    ex = res["ex"]
    cx, cv = ex["curve_x"], ex["curve"]
    last, first = ex["rows"][-1], ex["rows"][0]

    def flat(panel, m):
        panel.plot(cx, cv["truth"], lw=2.0, ls="--", color=m.ink)
        panel.plot(cx, cv["forest"], lw=2.8, color=m.series[0])

    def ratio(panel, m):
        panel.bar([0, 1], [first["forest"], last["forest"]],
                  color=[m.series[2], m.series[0]])

    def hyb(panel, m):
        panel.plot(cx, cv["truth"], lw=2.0, ls="--", color=m.ink)
        panel.plot(cx, cv["trend + forest"], lw=2.8, color=m.series[3])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=15,
        headline="Trees answer only within what they have seen",
        panels=[(flat, f"{last['forest_max']:.2f}", "forest's ceiling"),
                (ratio, f"{last['forest'] / first['forest']:.0f}x",
                 "error beyond the range"),
                (hyb, f"{last['trend + forest']:.2f}", "trend + forest error")],
        note=("A forest predicts averages of training labels, so it cannot "
              "go above the largest one it saw. Inside its range it beats a "
              "linear model; past it, it is the worst model measured. "
              "Fitting the trend first and the trees to what is left fixes "
              "both."),
        alt=("Three hand-drawn frames: a rising dashed line with a solid "
             "line that goes flat; a short bar and a tall bar; a solid line "
             "following the dashed one all the way."),
        path=str(IMG / f"ml115-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["bands"] = s.run("""
        from standarderror.ml import trees as tr

        # train: 500 rows, x in [0, 1], y = 2x + sin(2z) + noise
        ex = tr.extrapolation()
        for r in ex["rows"]:
            lo, hi = r["band"]
            print(f"x in [{lo:4.2f}, {hi:4.2f}]  " + "  ".join(
                f"{k} {r[k]:.3f}" for k in
                ("forest", "boosting", "linear", "trend + forest")))
        """)
    out["ceiling"] = s.run("""
        print(f"largest training label {ex['train_y_max']:.2f}; "
              f"largest forest prediction at x in [1.5, 2]: "
              f"{ex['rows'][-1]['forest_max']:.2f}")
        """)
    out["diab"] = s.run("""
        d = tr.diabetes_extrapolation(quantile=0.9)
        for split in ("random", "beyond"):
            print(split, "  ".join(f"{m} MSE {d[split][m]['mse']:,.0f} "
                                   f"(mean error {d[split][m]['bias']:+.1f})"
                                   for m in ("forest", "linear")))
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    ex, d = res["ex"], res["diab"]
    first, last = ex["rows"][0], ex["rows"][-1]
    ratio = last["forest"] / first["forest"]
    dr, db = d["random"], d["beyond"]
    loss = db["forest"]["mse"] / db["linear"]["mse"] - 1

    # The spine, asserted rather than trusted.
    assert first["forest"] < first["linear"] and first["boosting"] < \
        first["linear"]
    assert last["forest"] > 4 * last["linear"] and ratio > 15
    assert all(a["forest"] < b["forest"] for a, b in
               zip(ex["rows"], ex["rows"][1:]))
    assert max(r["trend + forest"] for r in ex["rows"]) < first["forest"] + 0.01
    assert last["forest_max"] <= ex["train_y_max"] + 1e-9
    assert dr["forest"]["mse"] < dr["linear"]["mse"]
    assert db["forest"]["mse"] > 1.3 * db["linear"]["mse"]
    assert db["forest"]["bias"] < db["linear"]["bias"] < 0

    post = Post(
        title=(f"{SERIES_TAG} 15: A Forest Cannot Predict Beyond the Largest "
               f"Value It Has Seen"),
        slug="ml-15-extrapolation",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=15,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Tree ensembles are said to approximate any function. "
                  "Inside the training range they do; outside it their "
                  "predictions stop at the largest label they saw, and the "
                  "best model becomes the worst."),
        summary=(
            f"Trained on x between 0 and 1 with y rising as 2x, a random "
            f"forest has test error {first['forest']:.2f} inside the range, "
            f"against {first['linear']:.2f} for linear regression, and "
            f"{last['forest']:.2f} between 1.5 and 2 — {ratio:.0f} times "
            f"worse — because it never predicts above {last['forest_max']:.2f}."
            f" Histogram boosting does the same. Fitting the trend linearly "
            f"and a forest to the residuals gives {first['trend + forest']:.2f}"
            f" inside and {last['trend + forest']:.2f} beyond. On the "
            f"diabetes data a forest beats linear regression on a random "
            f"split and loses to it by {loss:.0%} on the highest-bmi "
            f"patients when trained without them."),
        tags=["machine-learning", "data-science", "statistics",
              "random-forest", "extrapolation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: x uniform on the stated range, three further "
            "standard-normal inputs, y = 2x + sin(2z) + 0.3 N(0, 1); 500 "
            "training rows on [0, 1], 2,000 test rows per band.",
            "Diabetes data of Efron et al. (2004), as bundled with "
            "scikit-learn: trained on the patients at or below the 90th "
            "percentile of bmi, tested on the 45 above; a random split of "
            "the same sizes as control.",
            "Machinery: `standarderror/ml/trees.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's RandomForestRegressor, "
            "HistGradientBoostingRegressor and LinearRegression.",
            "Where this stops: Hastie, Tibshirani and Friedman, *The "
            "Elements of Statistical Learning* (2009), section 9.2, on trees "
            "as piecewise-constant fits; Zhang et al., \"Regression-enhanced "
            "random forests\" (2019), for the trend-plus-forest construction.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("training rows from seed 0, test rows from "
                            "seed 1; models with random_state 0; the "
                            "diabetes control split from seed 0"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Universal, inside the box",
        f"""Trees and the ensembles built from them come with a reassuring theorem attached: with enough trees and enough depth they can approximate any reasonable function. The theorem has a domain. A tree's prediction in a leaf is the mean of the training labels that landed there, and a forest's prediction is an average of those. No average of numbers can be larger than the largest of them. Whatever a forest predicts, it predicts within the range of the labels it was trained on.

Inside the region the training data cover, that costs nothing. Outside it, it is decisive. A simulation shows how much: an outcome that rises as 2x plus a nonlinear effect of a second input, 500 training rows with x between 0 and 1, and test rows in bands of x inside and beyond that range.

{snip['bands'].markdown()}

Inside the range the trees are the best models measured: {first['forest']:.3f} for the forest and {first['boosting']:.3f} for boosting, against {first['linear']:.3f} for a linear regression that cannot represent the sine term. In the first band past the edge the forest's error is already {ex['rows'][1]['forest'] / first['forest']:.1f} times larger. Between 1.5 and 2 it is **{last['forest']:.2f}, {ratio:.0f} times its error inside**, and the linear model, unchanged at {last['linear']:.3f}, is {last['forest'] / last['linear']:.1f} times better.""",
        figures=[figs["f0"]])

    post.add(
        "A ceiling, not a mistake",
        f"""The figure makes the mechanism plain. Along x, the forest and the boosted trees follow the trend to the edge of the training data and then run flat:

{snip['ceiling'].markdown()}

Nothing went wrong in fitting. Every split threshold on x lies inside [0, 1], so every test row with x above 1 falls into the same leaves as the training rows with the largest x, and gets their average. The model has no representation of "and it keeps rising"; it has a step function that ends.

Boosting is no exception, despite being a sum rather than an average. Each of its trees is still piecewise constant in x with thresholds inside the training range, so the sum is constant beyond it too.

The fix keeps the trees' strengths and removes the ceiling: fit the trend with a model that can extrapolate it — here a linear regression on x alone — and fit the forest to what is left. The residuals have no trend, so the forest's flat continuation is the right continuation. That hybrid has error {first['trend + forest']:.3f} inside the range and {last['trend + forest']:.3f} between 1.5 and 2: best everywhere. It works because the shape of the trend was known. When it is not, the honest statement is that no model has evidence about what happens past the data, and the trees at least make that visible by refusing to guess.""",
        figures=[figs["f1"]])

    post.add(
        "On real data: smaller, and still there",
        f"""Real extrapolation is rarely so clean. On the diabetes data, train on the patients with bmi at or below its 90th percentile and test on the {d['test_rows']} above it; compare with a random split of the same sizes:

{snip['diab'].markdown()}

On the random split the forest wins, {dr['forest']['mse']:,.0f} against {dr['linear']['mse']:,.0f}. On the high-bmi patients it loses by **{loss:.0%}**, and its errors lean one way: it under-predicts their disease progression by {-db['forest']['bias']:.0f} on average, against {-db['linear']['bias']:.0f} for the linear model. The relationship between bmi and progression keeps rising past the training range, and the forest, as on the simulation, stops.

The effect is much smaller than in the simulation, because bmi is one of ten inputs and the outcome is noisy. That is the realistic version: not a dramatic failure but a quiet, systematic bias on exactly the cases at the edge — the heaviest patients, the largest loans, the newest dates — that are often the ones a model is deployed to judge.""",
        figures=[figs["f2"]])

    post.add(
        "How to see it coming",
        f"""The ceiling is easy to detect before it costs anything, because it is a property of the training data, not of the model's internals. Three checks cover most cases.

Compare each input's range in the data the model will score with its range in the training data. A row whose value of x is beyond the largest training value is a row the trees will treat as if it sat at the edge. In the simulation, every row in the last band is such a row.

Compare the model's predictions with the training labels. A forest's predictions are confined to the label range by construction; if the outcome you expect is outside it — next year's sales, a larger loan, an older patient — the model cannot produce it, and its predictions will pile up just under the old maximum. Here they stop at {last['forest_max']:.2f}, under a training maximum of {ex['train_y_max']:.2f}.

Score the model on a split by the drifting variable, not only a random one. The diabetes table is that check: the random split rated the forest the better model; the split by bmi reversed the ranking.""")

    post.add(
        "Where this breaks",
        """**Time is the common case.** A forest given a date or a time index as a feature is extrapolating on every future prediction. Trends, inflation and growth all live past the last training date; a forest on raw levels will predict last year's level. Differencing or detrending first is the same fix as above.

**Extrapolation in one feature is not extrapolation in all.** A row can be inside the range of every feature separately and still far from any training row, in a corner of the joint space no one sampled. Trees are flat there too, and no per-feature range check will catch it.

**Linear models extrapolate confidently, not correctly.** The linear model here was right beyond the range because the truth was linear in x. Had the trend bent, it would have extrapolated the wrong line with equal conviction. The trees' ceiling is a known, bounded failure; the line's is not.

**One simulation, one real split.** The size of the ratio depends on how far past the edge one tests and how steep the trend is. The ceiling does not.""")

    post.add(
        "What to keep",
        f"""1. A tree's prediction is an average of training labels, so a forest cannot predict beyond the range of the labels it saw. This forest's ceiling was {last['forest_max']:.2f}.

2. Inside the training range the trees were the best models measured ({first['forest']:.2f} against {first['linear']:.2f} for linear); between 1.5 and 2 the forest's error was {ratio:.0f} times larger.

3. Fitting the trend first and the forest to the residuals was best everywhere: {first['trend + forest']:.2f} inside, {last['trend + forest']:.2f} beyond.

4. On diabetes, a forest that won a random split lost by {loss:.0%} on the highest-bmi patients and under-predicted them systematically.

5. Before deploying a tree ensemble, check where its inputs will fall relative to the training data — especially any input that drifts, such as time.""")

    post.add(
        "Exercise",
        """Sort your training data by the feature most likely to drift — a date, a size, a price — and train on the first 80%, test on the last 20%. Compare your tree ensemble with a linear model on that split and on a random split of the same sizes. If the ensemble's advantage shrinks or reverses, plot its predictions against that feature on the test rows and look for the flat line.

Then try the hybrid: regress the outcome on the drifting feature alone, fit the ensemble to the residuals, and add the two back together.""")

    post.add(
        "Next",
        """Arc III ends here: the instability of a single tree, the bias of its importance measure, the forest's free error estimate, boosting's long tail and the ceiling on its predictions. Arc IV turns to reading what a fitted model learned, starting with the importance measure episode 12 recommended — permutation importance — when two features carry the same information.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
