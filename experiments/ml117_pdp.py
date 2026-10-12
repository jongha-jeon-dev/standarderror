"""ML 17: Partial Dependence Asks the Model About Rows That Do Not Exist.

The seventeenth episode of *Machine Learning, Taught Through What Breaks*.
"A partial dependence plot shows a feature's effect." Measured where the
effect is known: y = x1 + 2 x2^2, with x2 a noisy copy of x1.

* With independent inputs, partial dependence and accumulated local effects
  (ALE) both recover the true effect of x1, slope 1.
* As the correlation rises the PDP's slope falls -- 0.78, 0.44, 0.30 for a
  forest at correlations 0.69, 0.94 and 0.985 -- while ALE stays between
  0.93 and 0.96.
* The PDP is not wrong about the model. It averages the model's predictions
  over pairs (x1, x2) far from any training row, where a tree ensemble
  predicts something flat and arbitrary.

Run: `standarderror run ml117_pdp --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import reading as rd
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 12)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"sweep": rd.pdp_vs_ale(), "curves": rd.pdp_curves()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    c = res["curves"]

    def curves(ax, m):
        ax.plot(c["grid"], c["truth"], lw=2.0, ls="--", color=m.ink,
                label="true effect of x1 (slope 1)")
        ax.plot(c["grid"], c["pdp"], lw=2.6, marker="o", ms=4,
                color=m.series[1], label="partial dependence")
        ax.plot(c["grid"], c["ale"], lw=2.6, marker="s", ms=4,
                color=m.series[0], label="accumulated local effects")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    hi = [r for r in res["sweep"] if r["noise"] == 0.05 and
          r["model"] == "forest"][0]
    out["f0"] = charts.diagram(
        curves,
        title="Two readings of one forest, one of them flat",
        subtitle=("Effect of x1 on a random forest's prediction, centred. "
                  "Truth: y = x1 + 2 x2^2, with x2 = x1 + 0.05 N(0, 1), "
                  "correlation 0.985. One dataset of 1,000 rows."),
        xlabel="x1",
        ylabel="effect on prediction (centred)",
        source="Simulated; standarderror/ml/reading.py.",
        alt=("A dashed diagonal for the truth; the ALE curve follows it "
             "closely; the partial dependence curve rises far more "
             "gently."),
        caption=(f"Averaged over three datasets, the PDP's slope is "
                 f"**{hi['pdp_slope']:.2f}** and the ALE's "
                 f"{hi['ale_slope']:.2f}, against a true slope of 1."),
        path=str(IMG / f"ml117-f0-curves.{EXT}"))[0]

    def support(ax, m):
        X = c["X"]
        ax.scatter(X[:, 0], X[:, 1], s=5, color=m.series[0], alpha=0.5,
                   label="training rows")
        g = np.random.default_rng(0)
        idx = g.choice(len(X), 300, replace=False)
        for v in (0.1, 0.9):
            ax.scatter(np.full(300, v), X[idx, 1], s=6, color=m.series[1],
                       alpha=0.6, label="rows the PDP evaluates at x1 = 0.1 and 0.9"
                       if v == 0.1 else None)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.25, 1.25)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    out["f1"] = charts.diagram(
        support,
        title="Where partial dependence asks its questions",
        subtitle=("Training rows, and the rows a PDP builds by setting x1 to "
                  "0.1 or 0.9 and keeping each row's own x2."),
        xlabel="x1",
        ylabel="x2",
        source="Simulated; standarderror/ml/reading.py.",
        alt=("A narrow diagonal band of training points, and two vertical "
             "lines of evaluated points at x1 = 0.1 and 0.9 that run far "
             "outside the band."),
        caption=("To compute the PDP at x1 = 0.1, the model is asked about "
                 "rows with x1 = 0.1 and x2 near 1. No training row looks "
                 "like that, and the forest's answer there is whatever its "
                 "nearest leaves say."),
        path=str(IMG / f"ml117-f1-support.{EXT}"))[0]

    rows = []
    for r in res["sweep"]:
        rows.append(["independent" if r["noise"] is None else
                     f"{r['corr']:.3f}", r["model"],
                     f"{r['pdp_slope']:.2f}", f"{r['ale_slope']:.2f}",
                     f"{r['pdp_rmse']:.3f}", f"{r['ale_rmse']:.3f}",
                     f"{r['pdp_rmse2']:.3f}", f"{r['ale_rmse2']:.3f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["corr(x1, x2)", "model", "PDP slope", "ALE slope",
                "PDP error, x1", "ALE error, x1", "PDP error, x2",
                "ALE error, x2"],
        title="The truth is known; one method finds it",
        subtitle=("True slope of x1 is 1. Errors are RMS distances from the "
                  "true centred effect over the grid. Means of three "
                  "datasets of 1,000 rows."),
        source="Simulated; standarderror/ml/reading.py.",
        alt=("An eight-row table: with independent inputs both methods are "
             "accurate; as correlation rises the PDP's slope falls towards "
             "0.3 and its error grows tenfold, while ALE stays close."),
        caption=("Both methods are exact descriptions of something. ALE "
                 "describes the model where the data are; partial "
                 "dependence describes it everywhere, mostly where they are "
                 "not."),
        align="llrrrrrr",
        path=str(IMG / f"ml117-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res, hi)
    return out


def _hero(res: dict, hi: dict):
    c = res["curves"]
    fr = [r for r in res["sweep"] if r["model"] == "forest"]

    def cv(panel, m):
        panel.plot(c["grid"], c["truth"], lw=2.0, ls="--", color=m.ink)
        panel.plot(c["grid"], c["pdp"], lw=2.8, color=m.series[1])

    def al(panel, m):
        panel.plot(c["grid"], c["truth"], lw=2.0, ls="--", color=m.ink)
        panel.plot(c["grid"], c["ale"], lw=2.8, color=m.series[0])

    def sl(panel, m):
        panel.plot(range(len(fr)), [r["pdp_slope"] for r in fr], lw=2.8,
                   color=m.series[1])
        panel.plot(range(len(fr)), [r["ale_slope"] for r in fr], lw=2.8,
                   color=m.series[0])
        panel.set_ylim(0, 1.2)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=17,
        headline="Partial dependence reads the model off the map",
        panels=[(cv, f"{hi['pdp_slope']:.2f}", "PDP slope (truth 1)"),
                (al, f"{hi['ale_slope']:.2f}", "ALE slope"),
                (sl, "0.985", "correlation of x1, x2")],
        note=("With correlated inputs, a partial dependence plot averages "
              "the model's predictions over combinations that never occur. "
              "On a simulation with a known effect it shows a third of it. "
              "Accumulated local effects, which only compare nearby rows, "
              "recover it."),
        alt=("Three hand-drawn frames: a shallow line under a dashed "
             "diagonal; a line on top of the diagonal; one falling line and "
             "one flat line."),
        path=str(IMG / f"ml117-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["sweep"] = s.run("""
        from standarderror.ml import reading as rd

        # y = x1 + 2 x2^2; x2 = x1 + noise * N(0, 1) (None: independent)
        for r in rd.pdp_vs_ale(noises=(None, 0.3, 0.1, 0.05)):
            if r["model"] == "forest":
                print(f"corr {r['corr']:5.2f}  PDP slope {r['pdp_slope']:.2f}"
                      f"  ALE slope {r['ale_slope']:.2f}   (truth 1.00)")
        """)
    out["ale"] = s.run("""
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor

        X, y = rd.correlated_pair(noise=0.05, seed=0)
        f = RandomForestRegressor(300, min_samples_leaf=5, random_state=0)
        f.fit(X, y)
        edges = np.quantile(X[:, 0], np.linspace(0, 1, 21))   # 20 bins of x1
        steps = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            rows = X[(X[:, 0] >= lo) & (X[:, 0] <= hi)]        # rows really there
            a, b = rows.copy(), rows.copy()
            a[:, 0], b[:, 0] = lo, hi                         # small move only
            steps.append(np.mean(f.predict(b) - f.predict(a)))
        print(f"x1 rises {edges[-1] - edges[0]:.2f}; ALE says the prediction "
              f"rises {sum(steps):.2f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    sw = res["sweep"]
    fr = [r for r in sw if r["model"] == "forest"]
    bo = [r for r in sw if r["model"] == "boosting"]
    ind, hi = fr[0], fr[-1]

    # The spine, asserted rather than trusted.
    for rows in (fr, bo):
        assert abs(rows[0]["pdp_slope"] - 1) < 0.05
        assert abs(rows[0]["ale_slope"] - 1) < 0.05
        assert all(a["pdp_slope"] > b["pdp_slope"] for a, b in
                   zip(rows, rows[1:]))
        assert all(abs(r["ale_slope"] - 1) < 0.1 for r in rows)
        assert rows[-1]["pdp_rmse"] > 3 * rows[-1]["ale_rmse"]
        assert rows[-1]["pdp_rmse2"] > 2.5 * rows[-1]["ale_rmse2"]
    assert hi["pdp_slope"] < 0.35 and hi["corr"] > 0.98

    post = Post(
        title=(f"{SERIES_TAG} 17: Partial Dependence Asks the Model About "
               f"Rows That Do Not Exist"),
        slug="ml-17-pdp",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=17,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A partial dependence plot is read as a feature's effect. "
                  "With correlated inputs it is an average over "
                  "combinations that never occur, and on a simulation with "
                  "a known effect it shows a third of it."),
        summary=(
            f"With y = x1 + 2 x2² and x2 a noisy copy of x1, the true effect "
            f"of x1 is a straight line of slope 1. With independent inputs, "
            f"the partial dependence plot of a random forest has slope "
            f"{ind['pdp_slope']:.2f} and accumulated local effects (ALE) "
            f"{ind['ale_slope']:.2f}. At a correlation of {hi['corr']:.3f} "
            f"the PDP's slope is {hi['pdp_slope']:.2f} and the ALE's "
            f"{hi['ale_slope']:.2f}; the PDP's error against the true effect "
            f"is {hi['pdp_rmse'] / hi['ale_rmse']:.0f} times the ALE's. "
            f"Gradient boosting shows the same pattern. The PDP averages "
            f"the model's predictions over pairs of inputs that never occur "
            f"together in the data."),
        tags=["machine-learning", "data-science", "statistics",
              "interpretability", "partial-dependence", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: x1 uniform on [0, 1]; x2 = x1 + s N(0, 1) for s = "
            "0.3, 0.1, 0.05, or an independent uniform; y = x1 + 2 x2² + 0.1 "
            "N(0, 1); 1,000 rows; three datasets per setting.",
            "Machinery: `standarderror/ml/reading.py` (partial dependence "
            "and ALE implemented directly), tested in `tests/test_ml.py`; "
            "scikit-learn's RandomForestRegressor (300 trees, leaves of 5) "
            "and HistGradientBoostingRegressor.",
            "Where this stops: Friedman, \"Greedy function approximation\", "
            "*Annals of Statistics* (2001), for partial dependence; Apley "
            "and Zhu, \"Visualizing the effects of predictor variables in "
            "black box supervised learning models\", *JRSS B* (2020), for "
            "ALE; Hooker, Mentch and Zhou, \"Unrestricted permutation forces "
            "extrapolation\", *Statistics and Computing* (2021).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("datasets from seeds 0 to 2; models seeded with "
                            "the dataset seed; a grid of 19 points from 0.05 "
                            "to 0.95; ALE on 20 quantile bins"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "What the plot computes",
        f"""Episode 16 measured how much a model *needs* a feature. The next question is *how* the prediction depends on it, and the standard picture is the partial dependence plot (PDP), introduced by Friedman alongside gradient boosting. For each value v on a grid, set the feature to v in every row of the data, leave everything else as it is, predict, and average. Plot the average against v.

The construction is clean, and it has a hidden assumption. Setting x1 to v *in every row* creates rows that pair v with every row's own value of the other features. If x1 and x2 are independent, those rows look like the data. If they are correlated, many of them look like nothing in the data at all, and the PDP averages the model's predictions there.

A simulation with a known answer shows what that costs. Let y = x1 + 2 x2² plus noise, with x2 a noisy copy of x1. The effect of x1, holding x2 fixed, is a straight line of slope 1. Fit a random forest, and compare its PDP for x1 with *accumulated local effects* (ALE), a method built to avoid exactly this problem:

{snip['sweep'].markdown()}

With independent inputs both methods find the line: slopes {ind['pdp_slope']:.2f} and {ind['ale_slope']:.2f}. As the correlation rises, the PDP's slope falls steadily, to **{hi['pdp_slope']:.2f} at a correlation of {hi['corr']:.3f}**. ALE stays within a few hundredths of 1 throughout.""",
        figures=[figs["f0"]])

    post.add(
        "Where the questions are asked",
        f"""The scatter of training rows is a narrow diagonal band: x2 is never far from x1. To compute the PDP at x1 = 0.1, every row has its x1 set to 0.1 and keeps its own x2, which ranges from 0 to 1. Most of those rows sit far off the band, in regions where the forest saw no data.

A forest's prediction in such a region is not a guess about the function; it is whatever the nearest leaves on the band say. For a row with x1 = 0.1 and x2 = 0.9, the splits on x2 dominate — they carry most of the signal, since 2x2² varies more than x1 — and send it to leaves trained on rows with both inputs near 0.9. Its prediction is close to what the forest predicts at x1 = 0.9. Averaging such rows flattens the PDP: the curve says x1 barely matters because most of the rows it averages over are answered by x2.

ALE avoids this by never leaving the band. It splits x1 into narrow bins, and within each bin it moves only the rows that are actually there from the bin's lower edge to its upper edge, recording the average change in prediction. Those rows stay close to the data. Accumulating the changes across bins traces the effect:

{snip['ale'].markdown()}

The truth is a rise equal to x1's, and ALE recovers most of it; the shortfall comes from the end bins, where a forest's predictions flatten for want of data beyond them. The result describes the model where it was trained, and on this simulation that is close to the truth.""",
        figures=[figs["f1"]])

    post.add(
        "Not wrong, but about the wrong thing",
        f"""It would be easy to say the PDP is wrong. It is not; it is an exact description of the forest's average prediction over a particular set of artificial rows. The problem is that the reader takes it as a description of the data, and the two come apart precisely when features are correlated — which in real tables, with incomes and house prices, ages and years of experience, doses and body weights, is the rule.

The error is not small. At the highest correlation the PDP's RMS distance from the true effect of x1 is {hi['pdp_rmse']:.3f}, against {hi['ale_rmse']:.3f} for ALE, and for x2, whose true effect is the curve 2x2², it is {hi['pdp_rmse2']:.3f} against {hi['ale_rmse2']:.3f}. Gradient boosting behaves the same way: its PDP slope falls to {bo[-1]['pdp_slope']:.2f}. Both models predict well on the band; both have arbitrary behaviour off it; the PDP reports the arbitrary part.

ALE has a cost too. It needs enough rows per bin to estimate a local difference, it cannot show what the model does off the data (sometimes that is the question), and with exactly collinear features no method can separate their effects, because the data contain no rows in which one moves without the other.""",
        figures=[figs["f2"]])

    post.add(
        "A one-line check",
        f"""Whether a PDP can be trusted is visible before drawing it. Take the feature and the one it is most correlated with, scatter them, and draw the vertical lines of rows the PDP will build at the lowest and highest grid values. The share of those rows inside the cloud is the share of the curve that describes the data; the rest describes the model somewhere else.

In the simulation at a correlation of {hi['corr']:.3f}, the rows built at x1 = 0.1 are almost all far above the band, and the rows built at x1 = 0.9 almost all far below it. Nearly the whole PDP is computed off the data, which is why its slope is {hi['pdp_slope']:.2f}. At a correlation of {fr[1]['corr']:.2f} the band is wide enough to hold a good share of them, and the PDP's slope, {fr[1]['pdp_slope']:.2f}, is correspondingly closer to the truth. The check costs one plot, and it says in advance which of the two curves to believe.""")

    post.add(
        "Where this breaks",
        """**Individual conditional expectation curves inherit the problem.** ICE plots show each row's curve instead of their average, which reveals heterogeneity but evaluates the same off-band rows.

**Two features is the easy case.** In ten dimensions, "off the data" is the normal state of a modified row, and checking a 2-D scatter will not reveal it. ALE's local moves are the safer default whenever features are correlated at all.

**The truth here is additive.** With an interaction between x1 and x2, the "effect of x1" depends on x2 and no single curve captures it. ALE's second-order plots and the functional-ANOVA decomposition address that; one-dimensional plots do not.

**Three datasets per setting.** The slopes quoted move by a few hundredths between seeds; the ordering never changes.""")

    post.add(
        "What to keep",
        f"""1. A partial dependence plot sets a feature to each value in every row and averages the predictions, including over combinations the data never contain.

2. With y = x1 + 2x2² and x1, x2 correlated at {hi['corr']:.3f}, a forest's PDP showed a slope of {hi['pdp_slope']:.2f} for an effect of slope 1. ALE showed {hi['ale_slope']:.2f}.

3. With independent inputs both methods were within a few hundredths of the truth. The PDP's failure grows with the correlation.

4. The PDP is an accurate picture of the model in regions with no data; ALE is a picture of the model where the data are.

5. Check the correlation of a feature with the others before reading its PDP. If it is high, use ALE.""")

    post.add(
        "Exercise",
        """Pick a feature whose partial dependence plot you have shown someone. Scatter it against the feature it is most correlated with and draw, on the same axes, the vertical lines of rows the PDP evaluates at the lowest and highest grid values. How many of those rows lie inside the cloud?

Then compute ALE for the same feature — twenty quantile bins are enough — and overlay the two curves. Where they disagree, the PDP was reporting the model off the data.""")

    post.add(
        "Next",
        """Episode 18 moves from effects averaged over the data to a single prediction: SHAP values, which split one prediction among the features. They come with an axiom-backed uniqueness theorem — and with a choice of reference distribution that the theorem does not make for you.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
