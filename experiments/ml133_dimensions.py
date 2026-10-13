"""ML 33: The Curse of Dimensionality Counts the Dimensions the Data Use.

The thirty-third episode of *Machine Learning, Taught Through What Breaks*.
"In high dimensions, nearest neighbours stop meaning anything." Measured on
uniform points and on the digits:

* For uniform points the nearest neighbour's distance approaches everyone
  else's: its ratio to the median distance rises from 0.03 in two
  dimensions to 0.94 in 1,024.
* The digits have 64 pixels and a ratio of 0.34, the value uniform data
  have in 8 dimensions. Five-nearest-neighbours scores 0.986 on them.
* Appending uniform noise columns is what breaks it: with 1,024 of them the
  ratio is 0.94 and nearest neighbours fall to 0.659, while logistic
  regression, which can weight columns, falls to 0.871.

Run: `standarderror run ml133_dimensions --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import around
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    c = around.contrast()
    nz = around.neighbours_noise()
    dims = np.array([r["d"] for r in c], float)
    ratios = np.array([r["ratio"] for r in c])
    # the uniform dimension with the digits' ratio, by log-linear interpolation
    eq = float(np.exp(np.interp(nz[0]["ratio"], ratios, np.log(dims))))
    return {"c": c, "n": nz, "equiv": eq, "fix": around.neighbours_repair()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    c, n = res["c"], res["n"]

    def ratio(ax, m):
        ax.plot([r["d"] for r in c], [r["ratio"] for r in c], marker="o",
                color=m.series[0], label="uniform points in d dimensions")
        ax.plot([r["d"] for r in n], [r["ratio"] for r in n], marker="D",
                color=m.series[1],
                label="digits (64 pixels) plus uniform noise columns")
        ax.annotate("digits alone", (n[0]["d"], n[0]["ratio"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8.5,
                    color=m.ink)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    out["f0"] = charts.diagram(
        ratio,
        title="When the nearest point is no nearer than the rest",
        subtitle=("Median over query points of the nearest neighbour's "
                  "distance divided by the median distance to all points. "
                  "Near 0: the nearest neighbour is special. Near 1: it is "
                  "not."),
        xlabel="number of columns (log scale)",
        ylabel="nearest / median distance",
        figsize=(7.4, 4.6),
        source=("Simulated and UCI digits (scikit-learn); "
                "standarderror/ml/around.py."),
        alt=("A rising curve for uniform points from near 0 at one dimension "
             "to about 0.94 at 1,024; the digits sit well below it at 64 "
             "columns and climb towards it as noise columns are added."),
        caption=(f"The digits' 64 pixels have the ratio uniform data have in "
                 f"about **{res['equiv']:.0f}** dimensions. Noise columns move "
                 f"them up the curve."),
        path=str(IMG / f"ml133-f0-ratio.{EXT}"))[0]

    def acc(ax, m):
        ax.plot([r["d"] for r in n], [r["knn"] for r in n], marker="o",
                color=m.series[1], label="5-nearest neighbours")
        ax.plot([r["d"] for r in n], [r["logistic"] for r in n], marker="o",
                color=m.series[2], label="logistic regression")
        ax.set_xscale("log")
        ax.set_ylim(0.6, 1.0)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    out["f1"] = charts.diagram(
        acc,
        title="Noise columns hurt the method that weighs them equally",
        subtitle=("Five-fold accuracy on the digits as uniform noise columns "
                  "on the pixels' scale are appended: 0, 16, 64, 256 and "
                  "1,024 of them."),
        xlabel="total columns (log scale)",
        ylabel="cross-validated accuracy",
        figsize=(7.2, 4.5),
        source="UCI digits (scikit-learn); standarderror/ml/around.py.",
        alt=("Two falling curves: nearest neighbours start higher and fall "
             "steeply past a few hundred columns; logistic regression starts "
             "a little lower and falls much less."),
        caption=(f"With 1,024 noise columns nearest neighbours score "
                 f"**{n[-1]['knn']:.3f}** and logistic regression "
                 f"{n[-1]['logistic']:.3f}."),
        path=str(IMG / f"ml133-f1-accuracy.{EXT}"))[0]

    rows = [[f"uniform, d = {r['d']:,}", f"{r['ratio']:.3f}",
             f"{r['spread']:.2f}", "", ""] for r in c]
    rows += [[("digits" if r["extra"] == 0 else
               f"digits + {r['extra']:,} noise"), f"{r['ratio']:.3f}",
              f"{r['spread']:.2f}", f"{r['knn']:.3f}", f"{r['logistic']:.3f}"]
             for r in n]
    out["f2"] = charts.table_image(
        rows,
        header=["points", "nearest / median", "(farthest − nearest) / nearest",
                "5-NN accuracy", "logistic accuracy"],
        title="Distance contrast, nominal and real",
        subtitle=("Uniform points: 1,000 in [0, 1]^d, 200 queries. Digits: "
                  "pixels scaled to [0, 1], 300 queries against the other "
                  "1,796 images."),
        source=("Simulated and UCI digits (scikit-learn); "
                "standarderror/ml/around.py."),
        alt=("A thirteen-row table of distance ratios for uniform data in "
             "growing dimension and for the digits with growing numbers of "
             "noise columns, with accuracies for the digits."),
        caption=("The digits sit between uniform data in 4 and 16 dimensions. "
                 "The noise columns carry them to where 1,024 uniform "
                 "dimensions are."),
        bold_cells={(len(c), 1), (len(rows) - 1, 3)},
        align="lrrrr",
        path=str(IMG / f"ml133-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    c, n = res["c"], res["n"]

    def uni(panel, m):
        panel.plot(np.log([r["d"] for r in c]), [r["ratio"] for r in c],
                   color=m.series[0], lw=2)
        panel.set_ylim(0, 1)

    def dig(panel, m):
        panel.plot(np.log([r["d"] for r in c]), [r["ratio"] for r in c],
                   color=m.series[0], lw=1)
        panel.scatter([np.log(64)], [n[0]["ratio"]], color=m.series[1], s=40)
        panel.set_ylim(0, 1)

    def noise(panel, m):
        panel.plot([r["knn"] for r in n], color=m.series[1], lw=2)
        panel.plot([r["logistic"] for r in n], color=m.series[2], lw=2)
        panel.set_ylim(0.6, 1.0)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=33,
        headline="Dimensions the data use",
        panels=[(uni, f"{c[-1]['ratio']:.2f}", "uniform, 1,024 dims"),
                (dig, f"≈ {res['equiv']:.0f} dims", "64 digit pixels act like"),
                (noise, f"{n[-1]['knn']:.3f}", "5-NN, +1,024 noise cols")],
        note=("Distances do concentrate in high dimensions, for data that "
              "fill them. Real data rarely do; irrelevant columns do, and "
              "they are what breaks nearest neighbours."),
        alt=("Three hand-drawn frames: a rising curve; the same curve with a "
             "dot well below it; two falling lines, one steeper."),
        path=str(IMG / f"ml133-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["uniform"] = s.run("""
        from standarderror.ml import around

        for r in around.contrast():          # 1,000 uniform points per d
            print(f"d = {r['d']:5d}: nearest / median distance {r['ratio']:.3f}, "
                  f"(farthest - nearest) / nearest {r['spread']:.2f}")
        """)
    out["fix"] = s.run("""
        r = around.neighbours_repair()   # 1,024 noise columns, keep 64 by F
        print(f"5-NN on the {r['keep']} selected columns: {r['knn']:.3f}; "
              f"real pixels among them: {r['real_share']:.0%}")
        """)
    out["digits"] = s.run("""
        for r in around.neighbours_noise():  # digits plus uniform noise columns
            print(f"{r['d']:5d} columns: nearest / median {r['ratio']:.3f}, "
                  f"5-NN {r['knn']:.3f}, logistic {r['logistic']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    c, n, eq, fix = res["c"], res["n"], res["equiv"], res["fix"]
    cd = {r["d"]: r for r in c}
    d0, dmax = n[0], n[-1]

    # The spine, asserted rather than trusted.
    assert all(a["ratio"] < b["ratio"] for a, b in zip(c, c[1:]))
    assert cd[2]["ratio"] < 0.05 and cd[1024]["ratio"] > 0.9
    assert cd[2]["spread"] > 50 and cd[1024]["spread"] < 0.15
    assert 6 < eq < 10 and abs(d0["ratio"] - cd[8]["ratio"]) < 0.02
    assert d0["knn"] > 0.98 and d0["knn"] > d0["logistic"]
    assert all(a["knn"] > b["knn"] for a, b in zip(n, n[1:]))
    assert dmax["knn"] < 0.7 and dmax["logistic"] > 0.85
    assert dmax["ratio"] > 0.9
    assert fix["knn"] > 0.97 and 0.8 < fix["real_share"] < 1.0
    assert (d0["knn"] - dmax["knn"]) > 2.5 * (d0["logistic"] - dmax["logistic"])

    post = Post(
        title=(f"{SERIES_TAG} 33: The Curse of Dimensionality Counts the "
               f"Dimensions the Data Use"),
        slug="ml-33-dimensions",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=33,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Distances concentrate in high dimensions, for points that "
                  "fill them. The digits' 64 pixels behave like 8 uniform "
                  "dimensions, and nearest neighbours work on them until noise "
                  "columns are added."),
        summary=(
            f"For uniform points, the nearest neighbour's distance over the "
            f"median distance rose from {cd[2]['ratio']:.2f} in two dimensions "
            f"to {cd[1024]['ratio']:.2f} in 1,024: in high dimensions the "
            f"nearest point is barely nearer than the rest. The 1,797 digits "
            f"have 64 pixels and a ratio of {d0['ratio']:.2f}, the value "
            f"uniform data have in about {eq:.0f} dimensions, and "
            f"five-nearest-neighbours scores {d0['knn']:.3f} on them. "
            f"Appending 1,024 uniform noise columns raised the ratio to "
            f"{dmax['ratio']:.2f} and dropped nearest neighbours to "
            f"{dmax['knn']:.3f}, while logistic regression fell from "
            f"{d0['logistic']:.3f} to {dmax['logistic']:.3f}."),
        tags=["machine-learning", "data-science", "nearest-neighbours",
              "curse-of-dimensionality", "distance", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: 1,000 points and 200 queries uniform on [0, 1]^d for "
            "d from 1 to 1,024; seed 0.",
            "Optical Recognition of Handwritten Digits (UCI), as bundled with "
            "scikit-learn, CC BY 4.0; pixels divided by 16, with 0 to 1,024 "
            "uniform [0, 1] columns appended (seed 0).",
            "Machinery: `standarderror/ml/around.py`, tested in "
            "`tests/test_ml.py`; Euclidean distances, scikit-learn's "
            "KNeighborsClassifier (k = 5) and LogisticRegression, five-fold "
            "stratified cross-validation.",
            "Where this stops: Beyer, Goldstein, Ramakrishnan and Shaft, "
            "\"When is 'nearest neighbor' meaningful?\", *ICDT* (1999); "
            "Levina and Bickel, \"Maximum likelihood estimation of intrinsic "
            "dimension\", *NeurIPS* (2004).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("uniform points and noise columns from seed 0; "
                            "shuffled folds with random_state 0"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The nearest of many equals",
        f"""Episode 32 clustered by distance and found that the clusters depended on what distance could see. Nearest-neighbour methods depend on distance even more directly: to classify a point, find the training points closest to it and take their vote. The textbook warning attached to them is the curse of dimensionality, usually stated as the sentence under test: *in high dimensions, nearest neighbours stop meaning anything*.

The warning has a precise form. Beyer and colleagues showed that for many distributions, as the number of dimensions grows, the distance to the nearest point and the distance to the farthest become nearly equal, so "nearest" no longer singles anything out. For points spread uniformly over a cube it is easy to see:

{snip['uniform'].markdown()}

In two dimensions the nearest of 1,000 points is at **{cd[2]['ratio']:.2f}** of the median distance, and the farthest is {cd[2]['spread']:.0f} times farther than the nearest. In 1,024 dimensions the nearest is at **{cd[1024]['ratio']:.2f}** of the median, and the farthest only {cd[1024]['spread']:.0%} farther. Every point is about equally far from every other. A vote among the "nearest" five is a vote among five points chosen almost at random.

So the sentence is true of uniform points. The question is whether it is true of data.""",
        figures=[figs["f0"]])

    post.add(
        "Sixty-four pixels, eight dimensions",
        f"""The digits are 8 × 8 images, 64 numbers each: by the column count, a high-dimensional data set. Here is the same ratio for them, and what happens when uniform noise columns are appended to each image:

{snip['digits'].markdown()}

On the digits alone the nearest image is at **{d0['ratio']:.2f}** of the median distance. Uniform points reach that ratio at about **{eq:.0f}** dimensions, not 64. And five-nearest-neighbours, the method the curse is supposed to defeat, scores {d0['knn']:.3f}, higher than logistic regression's {d0['logistic']:.3f}.

The reason is that the 64 pixels are not 64 independent directions. Strokes are continuous, so neighbouring pixels move together; most of the 8 × 8 grid is blank for most digits; and the images of one digit vary in a few ways (slant, thickness, size) rather than in 64. The points occupy a thin, curved region of the 64-dimensional cube, and distances within that region behave like distances in a much smaller space. Levina and Bickel call the dimension of that region the intrinsic dimension and estimate it from exactly this kind of nearest-neighbour statistic.

The other measure agrees. The gap between a digit's farthest and nearest neighbours, as a multiple of the nearest distance, is {d0['spread']:.1f}; for uniform points it is {cd[8]['spread']:.1f} in 8 dimensions and {cd[64]['spread']:.2f} in 64. By either measure the digits look like a space of a handful of dimensions, not of sixty-four.

What matters for the curse, then, is not how many columns the data have but how many directions they actually spread in.""")

    post.add(
        "What breaks it: columns that carry nothing",
        f"""The noise columns show where the curse does bite. Each added column is uniform on the same 0-to-1 scale as the pixels and unrelated to the digit. Sixteen of them barely matter. With 256, the ratio is {n[3]['ratio']:.2f} and nearest neighbours fall to {n[3]['knn']:.3f}. With 1,024, the ratio is **{dmax['ratio']:.2f}**, about what 1,024 uniform dimensions give, and nearest neighbours score **{dmax['knn']:.3f}**.

The noise columns fill their dimensions, which the pixels do not, and Euclidean distance weighs every column equally. Once the noise columns outnumber the pixels sixteen to one, the distance between two images is mostly the distance between their noise, and the nearest neighbours are nearest in noise.

Logistic regression suffers far less, falling from {d0['logistic']:.3f} to {dmax['logistic']:.3f}. It learns a weight for every column and can give the noise columns weights near zero, which a plain distance cannot. The drop for nearest neighbours is {(d0['knn'] - dmax['knn']) / (d0['logistic'] - dmax['logistic']):.1f} times the drop for logistic regression. The practical curse is therefore a curse on methods that treat every column as equally informative, in data with many uninformative columns.""",
        figures=[figs["f1"], figs["f2"]])

    post.add(
        "The repair is to stop counting the noise",
        f"""If the trouble is columns that carry nothing, the repair is to stop letting them into the distance. The bluntest version keeps the 64 columns whose values differ most between digit classes, by an analysis-of-variance F statistic, chosen inside each training fold so the test images play no part in the choice:

{snip['fix'].markdown()}

With the 1,024 noise columns still present in the data, nearest neighbours on the 64 kept columns score **{fix['knn']:.3f}**, against {dmax['knn']:.3f} without selection and {d0['knn']:.3f} on the clean digits. The selection is not perfect: {fix['real_share']:.0%} of the columns it keeps are real pixels, and the rest are noise columns that happened to differ between classes by chance. It is enough. Distance is useful again as soon as most of it is computed on columns that carry information.

That is also the honest reading of the curse. It is not a property of a number of columns, which a selection step can change without changing the problem. It is a property of how the information in the data is spread across them. Many columns, few of them informative, is the hard case. Many columns, all of them moving together with the thing being predicted, is the easy one, and the digits are much closer to it.""")

    post.add(
        "Where this breaks",
        """**Uniform noise is the worst kind.** Real irrelevant features are often correlated with each other or nearly constant, and fill fewer dimensions than independent uniform columns; they hurt less.

**Euclidean distance, unweighted.** Scaling or selecting features, learning a metric, or using a representation from a trained model all change what "near" means, and are the standard repairs. None was applied here, by design.

**One data set with low intrinsic dimension.** Data that genuinely spread in many directions, such as some text or genomic data, can be nearly as hard as the uniform case. The point is that the column count does not tell you which case you are in; the distance ratio, or an intrinsic-dimension estimate, does.

**k = 5, one split scheme.** Other choices of k move the numbers a little and not the pattern.""")

    post.add(
        "What to keep",
        f"""1. For points that fill their space, distances concentrate: the nearest/median ratio rose from {cd[2]['ratio']:.2f} in 2 uniform dimensions to {cd[1024]['ratio']:.2f} in 1,024.

2. Real data rarely fill their columns. The digits' 64 pixels had the ratio of about {eq:.0f} uniform dimensions, and five-nearest-neighbours scored {d0['knn']:.3f}.

3. Irrelevant columns do fill their dimensions. With 1,024 noise columns nearest neighbours fell to {dmax['knn']:.3f}; logistic regression, which weights columns, fell to {dmax['logistic']:.3f}.

4. Before blaming dimension, measure the nearest/median distance ratio of your data, and remove or down-weight columns that carry nothing.""")

    post.add(
        "Exercise",
        """For a data set you use with a distance-based method, take 200 rows and compute, for each, its nearest distance over its median distance to the rest. Then compute the same for uniform data with your number of columns, and with a few smaller numbers.

Find the uniform dimension that matches your data's ratio. If it is far below your column count, the curse is not your problem; if it is close, check how many of your columns actually carry information about the target.""")

    post.add(
        "Next",
        """Distances decide which neighbours vote; probabilities decide how much a prediction should be trusted. Episode 34 measures the standard repair for untrustworthy probabilities, recalibration, and asks how much data it needs and what kind of trust it buys.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
