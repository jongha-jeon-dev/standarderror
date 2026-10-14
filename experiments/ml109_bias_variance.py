"""ML 9: The Bias-Variance U Belongs to a Knob, Not to Complexity.

The ninth episode of *Machine Learning, Taught Through What Breaks*. "Test
error is U-shaped in model complexity: bias falls, variance rises." Measured
exactly on a simulated regression, where every term can be computed:

* For k-nearest neighbours the decomposition is textbook. Bias squared and
  variance plus the noise add up to the measured test error, the variance
  falls and the bias rises with k, and the error has a U with its bottom
  at k = 8.
* For a random forest, adding trees has no U. Bias stays where one tree put
  it, variance falls monotonically, and it levels off at a floor set by the
  correlation between trees (about 0.46 here), as the formula for an
  average of correlated predictors predicts (ESL eq. 15.1). More trees
  never hurt.
* "Complexity" is not one axis. A knob that trades bias for variance draws
  a U; a knob that only averages draws a descending curve with a floor.

Run: `standarderror run ml109_bias_variance --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import linear as ln
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 8)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    knn = ln.knn_sweep()
    forest = ln.forest_sweep()
    return {"knn": knn, "forest": forest, "rho": ln.forest_rho(forest)}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    knn, forest, rho = res["knn"], res["forest"], res["rho"]
    best = min(knn, key=lambda r: r["mse"])

    def u(ax, m):
        xs = list(range(len(knn)))
        ax.plot(xs, [r["mse"] for r in knn], marker="o", ms=6, lw=2.6,
                color=m.ink, label="test error, measured")
        ax.plot(xs, [r["total"] for r in knn], ls=":", lw=2.2,
                color="#9a9a9a", label="bias² + variance + noise")
        ax.plot(xs, [r["variance"] for r in knn], marker="s", ms=5, lw=2.2,
                color=m.series[0], label="variance")
        ax.plot(xs, [r["bias2"] for r in knn], marker="^", ms=5, lw=2.2,
                color=m.series[1], label="bias²")
        ax.axhline(knn[0]["noise"], ls="--", lw=1.2, color=m.series[2],
                   label="noise (irreducible)")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(r["k"]) for r in knn])
        ax.set_ylim(0, 0.34)
        ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)

    out["f0"] = charts.diagram(
        u,
        title="For k-nearest neighbours, the textbook U",
        subtitle=("Test error and its three parts for k-NN regression, "
                  "computed over 300 training sets of 60 points. Smaller k "
                  "is the more flexible model."),
        xlabel="neighbours averaged, k",
        ylabel="squared error, averaged over x",
        source="Simulated; standarderror/ml/linear.py.",
        alt=("Variance falling and bias squared rising with k; their sum "
             "plus the noise traces the measured test error, a U with its "
             "minimum at k = 8."),
        caption=(f"The three parts add up to the measured error to within "
                 f"{max(abs(r['total'] - r['mse']) for r in knn):.3f}. The "
                 f"bottom of the U is k = **{best['k']}**, test error "
                 f"{best['mse']:.3f} against a noise floor of "
                 f"{best['noise']:.2f}."),
        path=str(IMG / f"ml109-f0-knn.{EXT}"))[0]

    def trees(ax, m):
        ts = [r["trees"] for r in forest]
        s2 = forest[0]["variance"]
        grid = np.geomspace(1, ts[-1], 100)
        ax.plot(grid, rho * s2 + (1 - rho) * s2 / grid, lw=2.0, ls=":",
                color="#9a9a9a", label=f"ρσ² + (1 − ρ)σ²/T, ρ = {rho:.2f}")
        ax.plot(ts, [r["variance"] for r in forest], marker="s", ms=6,
                lw=2.4, color=m.series[0], label="variance")
        ax.plot(ts, [r["bias2"] for r in forest], marker="^", ms=6, lw=2.4,
                color=m.series[1], label="bias²")
        ax.axhline(rho * s2, lw=1.0, ls="--", color=m.series[0])
        ax.set_xscale("log")
        ax.set_xticks(ts)
        ax.set_xticklabels([str(t) for t in ts])
        ax.minorticks_off()
        ax.set_ylim(0, 0.18)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    out["f1"] = charts.diagram(
        trees,
        title="For a forest's size, no U at all",
        subtitle=("Bias squared and variance of a random forest as trees are "
                  "added, 100 training sets of 60 points. The dotted curve "
                  "is Breiman's formula fitted from one tree and 100."),
        xlabel="trees in the forest (log scale)",
        ylabel="squared error, averaged over x",
        source="Simulated; standarderror/ml/linear.py.",
        alt=("Variance falling steeply from one tree and levelling off at a "
             "dashed floor; bias squared flat near zero throughout."),
        caption=(f"Bias stays near zero. Variance falls from "
                 f"{forest[0]['variance']:.3f} with one tree to "
                 f"{forest[-1]['variance']:.3f} with 100 and would not go "
                 f"below **{rho * forest[0]['variance']:.3f}** with any "
                 f"number: averaging removes only the part of the variance "
                 f"the trees do not share."),
        path=str(IMG / f"ml109-f1-trees.{EXT}"))[0]

    rows = []
    for r in knn:
        rows.append([f"k-NN, k = {r['k']}", f"{r['bias2']:.4f}",
                     f"{r['variance']:.4f}", f"{r['noise']:.2f}",
                     f"{r['total']:.4f}", f"{r['mse']:.4f}"])
    for r in forest:
        rows.append([f"forest, {r['trees']} tree" + "s" * (r["trees"] > 1),
                     f"{r['bias2']:.4f}", f"{r['variance']:.4f}",
                     f"{r['noise']:.2f}", f"{r['total']:.4f}",
                     f"{r['mse']:.4f}"])
    bold = [(knn.index(best), 5)]
    out["f2"] = charts.table_image(
        rows,
        header=["model", "bias²", "variance", "noise", "sum",
                "test error"],
        title="The decomposition, term by term",
        subtitle=("Target sin(2πx) + x/2 on [0, 1], noise sd 0.4, 60 "
                  "training points. Terms averaged over 200 test points."),
        source="Simulated; standarderror/ml/linear.py.",
        alt=("An eleven-row table of bias squared, variance, noise, their "
             "sum and the measured test error, for six values of k and five "
             "forest sizes."),
        caption=("The sum column and the measured column agree row by row: "
                 "the decomposition is an identity in expectation, and the "
                 "small gaps are simulation error. What differs between the "
                 "two knobs is which terms move."),
        bold_cells=bold,
        align="lrrrrr",
        path=str(IMG / f"ml109-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res, best)
    return out


def _hero(res: dict, best: dict):
    knn, forest, rho = res["knn"], res["forest"], res["rho"]

    def u(panel, m):
        panel.plot(range(len(knn)), [r["mse"] for r in knn], lw=2.8,
                   color=m.series[0])

    def flat(panel, m):
        panel.plot(range(len(forest)), [r["variance"] for r in forest],
                   lw=2.8, color=m.series[1])
        panel.axhline(rho * forest[0]["variance"], lw=1.4, ls="--",
                      color=m.series[1])

    def bars(panel, m):
        panel.bar([0, 1], [forest[0]["variance"], forest[-1]["variance"]],
                  color=[m.series[2], m.series[0]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=9,
        headline="One knob draws a U; another only averages",
        panels=[(u, f"k = {best['k']}", "bottom of the k-NN U"),
                (flat, f"ρ = {rho:.2f}", "floor under more trees"),
                (bars, f"{forest[0]['variance'] / forest[-1]['variance']:.1f}x",
                 "variance, 1 to 100 trees")],
        note=("Bias and variance can be computed exactly on simulated data. "
              "The number of neighbours trades one for the other and test "
              "error has a U. The number of trees only averages: variance "
              "falls to a floor set by how alike the trees are, and more "
              "trees never hurt."),
        alt=("Three hand-drawn frames: a U-shaped curve; a falling curve "
             "levelling at a dashed line; a tall bar beside a short one."),
        path=str(IMG / f"ml109-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["knn"] = s.run("""
        from standarderror.ml import linear as ln

        # 300 training sets of 60 points; terms averaged over 200 test x
        for r in ln.knn_sweep(ks=(1, 2, 4, 8, 16, 32)):
            print(f"k {r['k']:>2}  bias2 {r['bias2']:.4f}  var {r['variance']:.4f}"
                  f"  noise {r['noise']:.2f}  sum {r['total']:.4f}"
                  f"  measured {r['mse']:.4f}")
        """)
    out["forest"] = s.run("""
        sweep = ln.forest_sweep(trees=(1, 2, 5, 25, 100))
        for r in sweep:
            print(f"trees {r['trees']:>3}  bias2 {r['bias2']:.4f}"
                  f"  var {r['variance']:.4f}  measured {r['mse']:.4f}")
        print(f"implied correlation between trees: {ln.forest_rho(sweep):.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    knn, forest, rho = res["knn"], res["forest"], res["rho"]
    best = min(knn, key=lambda r: r["mse"])
    k1, k32 = knn[0], knn[-1]
    f1, f100 = forest[0], forest[-1]
    floor = rho * f1["variance"]
    gap = max(abs(r["total"] - r["mse"]) for r in knn + forest)
    knn_gap = max(abs(r["total"] - r["mse"]) for r in knn)

    # The spine, asserted rather than trusted.
    assert gap < 0.01
    v = [r["variance"] for r in knn]
    b = [r["bias2"] for r in knn]
    assert all(x > y for x, y in zip(v, v[1:]))
    assert b[-1] > b[-2] > b[-3] > b[0]
    assert best["k"] not in (knn[0]["k"], knn[-1]["k"])
    fv = [r["variance"] for r in forest]
    assert all(x > y for x, y in zip(fv, fv[1:]))
    fm = [r["mse"] for r in forest]
    assert all(x > y for x, y in zip(fm, fm[1:]))
    assert max(r["bias2"] for r in forest) < 0.005
    assert 0.3 < rho < 0.6 and f100["variance"] > floor
    assert f100["mse"] > best["mse"]
    assert max(r["bias2"] for r in forest) - f100["bias2"] < 2 * f1["variance"] / 100

    post = Post(
        title=(f"{SERIES_TAG} 9: The Bias–Variance U Belongs to a Knob, "
               f"Not to Complexity"),
        slug="ml-9-bias-variance",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=9,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Test error is said to be U-shaped in model complexity. "
                  "Computed exactly, it is for the number of neighbours and "
                  "is not for the number of trees: one knob trades bias for "
                  "variance, the other only averages."),
        summary=(
            f"On a simulated regression where bias, variance and noise can "
            f"be computed exactly, the three add up to the measured test "
            f"error to within {gap:.3f}. For k-nearest neighbours the "
            f"variance falls from {k1['variance']:.3f} at k = 1 to "
            f"{k32['variance']:.3f} at k = 32 while bias squared rises from "
            f"{k1['bias2']:.4f} to {k32['bias2']:.3f}; test error is a U with "
            f"its bottom at k = {best['k']}. For a random forest, adding "
            f"trees keeps bias near zero (within simulation error) and lowers variance "
            f"monotonically, from {f1['variance']:.3f} to "
            f"{f100['variance']:.3f}, towards a floor of {floor:.3f} set by "
            f"a between-tree correlation of {rho:.2f}. There is no U."),
        tags=["machine-learning", "data-science", "statistics",
              "bias-variance", "random-forest", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: x uniform on [0, 1], y = sin(2πx) + x/2 plus normal "
            "noise of sd 0.4; 60 training points; 300 training sets for "
            "k-NN and 100 for the forests; terms averaged over 200 evenly "
            "spaced test points.",
            "Machinery: `standarderror/ml/linear.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's KNeighborsRegressor and "
            "RandomForestRegressor at their defaults.",
            "Where this stops: Geman, Bienenstock and Doursat, \"Neural "
            "networks and the bias/variance dilemma\", *Neural Computation* "
            "(1992); Breiman, \"Random forests\", *Machine Learning* (2001), "
            "whose bound on a forest's error by the mean correlation between "
            "its trees is the infinite-forest form of the same idea; Hastie, "
            "Tibshirani and Friedman, *The Elements of Statistical "
            "Learning* (2009), sections 7.3 and 15.2, for the decomposition "
            "and the variance of an average of correlated trees.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("training sets from seeds 0 to 299; noisy test "
                            "labels from seed 999; forests with "
                            "random_state 0"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A diagram worth checking",
        f"""Few pictures in machine learning are drawn as often as this one: test error against model complexity, a falling bias curve, a rising variance curve, and their sum making a U whose bottom is the model to choose. It is usually drawn, not measured.

On simulated data it can be measured exactly. Fix a target function, draw many training sets from it, fit the model to each and look at its predictions at fixed test points. The average prediction's distance from the truth is the **bias**; the spread of predictions across training sets is the **variance**; the noise in the labels is the floor nobody can beat. For squared error the three add up, exactly, to the expected test error.

The target here is sin(2πx) + x/2 on [0, 1] with noise of standard deviation 0.4, and each training set has 60 points. The first knob is the number of neighbours a k-nearest-neighbours regression averages.

{snip['knn'].markdown()}

This is the textbook. At k = 1 the model copies the nearest training label, noise and all: variance {k1['variance']:.3f}, bias almost nothing. At k = 32 it averages over half the data: variance {k32['variance']:.3f}, bias squared {k32['bias2']:.3f}. The sum tracks the measured error to within {knn_gap:.3f}, and the error has a clean U with its bottom at **k = {best['k']}**, where it is {best['mse']:.3f} against the noise floor of {best['noise']:.2f}.""",
        figures=[figs["f0"]])

    post.add(
        "A knob with no U",
        f"""The second knob is the number of trees in a random forest. More trees is "more model" by any count of parameters, and the U predicts that past some point it should overfit.

{snip['forest'].markdown()}

It does not. Bias squared is below {max(r['bias2'] for r in forest):.4f} at every size; its slight fall from one tree to a hundred is about the size of the simulation's own error in estimating it, which is the variance divided by the number of training sets ({f1['variance'] / 100:.4f} for one tree). Variance falls at every step, from {f1['variance']:.3f} with one tree to {f100['variance']:.3f} with a hundred, and the measured error falls with it. More trees never hurt.

The variance does not fall to zero either. The textbook formula for an average of correlated predictors gives the reason (Hastie, Tibshirani and Friedman, section 15.2; Breiman's 2001 bound for an infinite forest has the same shape): the average of T trees, each with variance σ² and pairwise correlation ρ, has variance ρσ² + (1 − ρ)σ²/T. Averaging removes the part the trees do not share and leaves the part they do. Fitting ρ from one tree and from a hundred gives **ρ = {rho:.2f}**: whatever the number of trees, this forest's variance cannot go below {floor:.3f}. The dotted curve in the figure is that formula, and the intermediate forest sizes sit on it.

None of this makes the forest a good model of this target. At a hundred trees its test error is {f100['mse']:.3f}, worse than k-NN at its best ({best['mse']:.3f}). Fully grown trees on one input copy their training labels almost as k = 1 does, and averaging a hundred of them removes only the unshared half of that variance. What would help is a knob on each tree — a minimum number of points per leaf, a maximum depth — which, like k, should trade bias for variance and draw a U; that was not measured here.

To lower the floor one must make the trees *less alike* — fewer features per split, smaller bootstrap samples — and that is a knob that does trade bias for variance, because decorrelated trees are individually worse.""",
        figures=[figs["f1"]])

    post.add(
        "What the U is about",
        """The two sweeps disagree because their knobs do different things. The number of neighbours changes *what each prediction is*: a wider average is smoother and less faithful to the target. That is a trade, and a trade has an optimum. The number of trees changes only *how many independent-ish copies of the same kind of prediction are averaged*. Averaging cannot add bias, so there is nothing to trade.

So "test error is U-shaped in complexity" is a claim about a particular knob, not about complexity as such. Depth of a single tree, the penalty of a ridge regression, the degree of a polynomial and k in k-NN all trade bias for variance, and below the point where the model interpolates its training data each typically draws a U; only k was measured here. Ensemble size, the number of bootstrap draws and the number of random restarts averaged do not. In practice the U is found by cross-validation, which estimates the sum and never its parts; the decomposition explains the shape of the curve but is not how anyone tunes a model on real data. The next episode adds a third kind: the number of features in a model that interpolates its training data, where the curve goes up, then down again.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        """**Squared error only.** The clean additive split into bias, variance and noise belongs to squared error. For 0-1 loss there are several competing decompositions, and in none of them does variance simply add.

**One target, one sample size.** Theory says that with more training points the k-NN U should move to larger k, and with a rougher target to smaller k; neither was measured here. The forest's floor depends on the target too. The shapes, not the numbers, are the finding.

**The decomposition needs many training sets.** On real data there is one. The bootstrap gives a rough stand-in, but it estimates variance from resamples that overlap heavily, and bias cannot be computed at all without the true function.

**Not the whole story past interpolation.** A 1-nearest-neighbour fit reproduces its training labels exactly, a fully grown tree those of its bootstrap sample, and both sit at the high-variance end here. Episode 10 shows a family of interpolating models whose error falls again as they grow, which the U has no room for.

**Forest defaults.** scikit-learn's regression forest considers every feature at each split by default. With one input feature here that makes no difference, but on wide data the setting changes ρ, and with it the floor.""")

    post.add(
        "What to keep",
        f"""1. For squared error, bias squared, variance and noise add up exactly to the expected test error; here they matched the measured error to within {gap:.3f}.

2. k in k-NN trades bias for variance, and test error had a U with its bottom at k = {best['k']}.

3. Adding trees to a random forest kept bias near zero, moving only within simulation error, and lowered variance every time, from {f1['variance']:.3f} to {f100['variance']:.3f}. No U.

4. The variance of an average of correlated models has a floor of ρσ²; for this forest ρ = {rho:.2f}. Lower it by decorrelating the members, not by adding more.

5. Before drawing a U, ask whether the knob changes the predictor or only averages copies of it.""")

    post.add(
        "Exercise",
        """Pick a model you tune and a knob you tune it with. On a simulated version of your problem — same number of rows and features, a target you choose — run the decomposition: 100 training sets, predictions at fixed test points, bias squared and variance averaged over them. Does your knob draw a U, or only a descending curve?

Then do the same for the number of members in your ensemble, and estimate ρ from one member and many. If your ensemble is already near ρσ², adding members is wasted compute.""")

    post.add(
        "Next",
        """Episode 10 takes the U past the point where the model fits its training data exactly. With random features and the minimum-norm fit, test error rises to a peak where the number of features equals the number of rows — and then falls again. Whether the second descent is a free lunch is the measurement.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
