"""ML 31: Principal Components Keep Variance, Not Relevance.

The thirty-first episode of *Machine Learning, Taught Through What Breaks*,
opening Arc VII. "The first principal components keep the information that
matters." Measured on a simulation, breast cancer and diabetes:

* On raw breast-cancer measurements the first component holds 98.2% of the
  variance and is essentially one feature, worst area, because area is
  measured in the largest units. The 95% rule keeps one component; on
  standardised features it keeps ten.
* On diabetes the fourth component, with 9.6% of the variance, explains
  12.8% of the target's variance, more than the second and third together.
  Regression on three components reaches R^2 0.349, on four 0.479; partial
  least squares reaches 0.479 with two.
* Variance is a property of the inputs and their units. Relevance is a
  property of the target, which PCA never sees.

Run: `standarderror run ml131_pca --publish`
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
    return {"sim": around.pca_simulation(), "bc": around.pca_breast_cancer(),
            "dia": around.pca_diabetes()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    d = res["dia"]
    x = np.arange(1, len(d["share"]) + 1)

    def bars(ax, m):
        w = 0.38
        ax.bar(x - w / 2, d["share"], w, color=m.series[0],
               label="share of the features' variance")
        ax.bar(x + w / 2, d["r2"], w, color=m.series[1],
               label="share of the target's variance it explains")
        ax.set_xticks(x)
        ax.set_xticklabels([f"PC{i}" for i in x])
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    out["f0"] = charts.diagram(
        bars,
        title="The components in order of variance, not usefulness",
        subtitle=("Diabetes, ten standardised clinical features: each "
                  "principal component's share of the feature variance, and "
                  "the squared correlation of its score with disease "
                  "progression."),
        ylabel="share",
        figsize=(7.4, 4.6),
        source="Diabetes (scikit-learn); standarderror/ml/around.py.",
        alt=("Paired bars for ten components: the blue variance bars fall "
             "steadily; the orange target bars are tall for PC1, small for "
             "PC2 and PC3, and tall again for PC4."),
        caption=(f"PC4 holds {d['share'][3]:.1%} of the variance and explains "
                 f"**{d['r2'][3]:.1%}** of the target; PC2 and PC3 together "
                 f"hold {d['share'][1] + d['share'][2]:.1%} and explain "
                 f"{d['r2'][1] + d['r2'][2]:.1%}."),
        path=str(IMG / f"ml131-f0-components.{EXT}"))[0]

    def curves(ax, m):
        ax.plot(d["ks"], d["pcr"], marker="o", color=m.series[0],
                label="regression on the first k principal components")
        ax.plot(d["ks"], d["pls"], marker="o", color=m.series[2],
                label="partial least squares, k components")
        ax.set_ylim(0.25, 0.52)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    out["f1"] = charts.diagram(
        curves,
        title="Components chosen with the target in view",
        subtitle=("Ten-fold cross-validated R² on diabetes for k components, "
                  "from PCA (unsupervised) and from partial least squares "
                  "(which chooses directions by their covariance with the "
                  "target)."),
        xlabel="number of components, k",
        ylabel="cross-validated R²",
        figsize=(7.2, 4.5),
        source="Diabetes (scikit-learn); standarderror/ml/around.py.",
        alt=("The PCA curve stays near 0.3 for one to three components and "
             "jumps at four; the PLS curve is near 0.48 from two components "
             "on."),
        caption=(f"Partial least squares reaches R² {d['pls'][1]:.3f} with two "
                 f"components; PCA needs four ({d['pcr'][3]:.3f})."),
        path=str(IMG / f"ml131-f1-pls.{EXT}"))[0]

    b = res["bc"]
    rows = []
    for name, key in (("raw measurements", "raw"),
                      ("standardised", "std")):
        r = b[key]
        rows.append([name, f"{r['share'][0]:.1%}", r["top"], str(r["k95"]),
                     str(r["k99"])] + [f"{a:.3f}" for a in r["acc"]])
    out["f2"] = charts.table_image(
        rows,
        header=["features", "variance in PC1", "PC1's largest loading",
                "k for 95%", "k for 99%"] + [f"accuracy, k={k}" for k in b["ks"]],
        title="The units choose the components",
        subtitle=("Breast cancer, 30 features: PCA before logistic "
                  "regression, five-fold cross-validated accuracy for k "
                  "components, and the number of components the 95% and 99% "
                  "variance rules would keep."),
        source=("Breast Cancer Wisconsin (scikit-learn); "
                "standarderror/ml/around.py."),
        alt=("A two-row table: on raw features the first component holds 98% "
             "of the variance, is dominated by worst area, and the variance "
             "rules keep one or two components; standardised, they keep 10 "
             "and 17, and accuracy is higher at every k."),
        caption=("Nothing about the tumours changed between the rows, only "
                 "the units in which their variance is counted."),
        bold_cells={(0, 1), (0, 3), (1, 3)},
        align="lrlrrrrrrrr",
        path=str(IMG / f"ml131-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    d, b, s = res["dia"], res["bc"], res["sim"]

    def sim(panel, m):
        panel.plot(range(1, 11), s["acc"], color=m.series[1], lw=2)
        panel.set_ylim(0.4, 1.0)

    def raw(panel, m):
        panel.bar(range(3), b["raw"]["share"], color=m.series[0])

    def dia(panel, m):
        panel.bar(range(1, 11), d["r2"], color=m.series[2])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=31,
        headline="Variance is not relevance",
        panels=[(sim, f"{s['acc'][-1]:.3f}", "only with the last PC"),
                (raw, f"{b['raw']['share'][0]:.1%}", "variance in raw PC1"),
                (dia, f"{d['r2'][3]:.1%}", "explained by PC4")],
        note=("PCA ranks directions by how much the inputs vary along them. "
              "Whether a direction predicts anything is a question it never "
              "asks."),
        alt=("Three hand-drawn frames: a flat line that jumps at the end; one "
             "tall bar beside two tiny ones; ten bars, the fourth tall."),
        path=str(IMG / f"ml131-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["sim"] = s.run("""
        from standarderror.ml import around

        sim = around.pca_simulation()  # label depends on the smallest feature
        print("accuracy with k components: " + " ".join(
            f"{a:.2f}" for a in sim["acc"]))
        print(f"variance in the last component: {sim['share'][-1]:.2%}")
        """)
    out["bc"] = s.run("""
        bc = around.pca_breast_cancer()
        for key in ("raw", "std"):
            r = bc[key]
            print(f"{key}: PC1 holds {r['share'][0]:.1%}, led by {r['top']!r}; "
                  f"95% rule keeps {r['k95']}, 99% keeps {r['k99']}")
        """)
    out["dia"] = s.run("""
        d = around.pca_diabetes()
        for j, (v, r2) in enumerate(zip(d["share"], d["r2"]), 1):
            print(f"PC{j:<2d} variance {v:5.1%}   target explained {r2:5.1%}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    s, b, d = res["sim"], res["bc"], res["dia"]
    raw, std = b["raw"], b["std"]
    ks = d["ks"]
    pcr = dict(zip(ks, d["pcr"]))
    pls = dict(zip(ks, d["pls"]))
    bks = dict(zip(b["ks"], range(len(b["ks"]))))

    # The spine, asserted rather than trusted.
    assert max(s["acc"][:9]) < 0.55 and s["acc"][9] > 0.9
    assert s["share"][9] < 0.001
    assert raw["share"][0] > 0.98 and raw["top"] == "worst area"
    assert raw["k95"] == 1 and raw["k99"] == 2
    assert std["k95"] == 10 and std["k99"] == 17
    assert all(std["acc"][i] >= raw["acc"][i] for i in range(len(b["ks"])))
    assert max(std["acc"]) - raw["acc"][bks[1]] > 0.07
    assert d["r2"][3] > d["r2"][1] + d["r2"][2]
    assert d["share"][3] < d["share"][1] and d["share"][3] < d["share"][2]
    assert pcr[3] < 0.36 and pcr[4] > 0.47
    assert pls[2] > 0.47 and pls[2] - pcr[2] > 0.1

    post = Post(
        title=f"{SERIES_TAG} 31: Principal Components Keep Variance, Not Relevance",
        slug="ml-31-pca",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=31,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("PCA keeps the directions along which the inputs vary most. "
                  "On real data that can be a choice of units, and the "
                  "direction that predicts the target can come fourth."),
        summary=(
            f"On raw breast-cancer measurements the first principal "
            f"component holds {raw['share'][0]:.1%} of the variance and is "
            f"led by {raw['top']}, the feature with the largest units; the "
            f"95% variance rule keeps {raw['k95']} component, against "
            f"{std['k95']} on standardised features. On diabetes the fourth "
            f"component, with {d['share'][3]:.1%} of the variance, explains "
            f"{d['r2'][3]:.1%} of the target's, more than the second and "
            f"third together; regression on three components reaches R² "
            f"{pcr[3]:.3f} and on four {pcr[4]:.3f}, while partial least "
            f"squares reaches {pls[2]:.3f} with two. PCA ranks directions by "
            f"variance, and never sees the target."),
        tags=["machine-learning", "data-science", "statistics", "pca",
              "dimensionality-reduction", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, and the "
            "diabetes data of Efron et al. (2004), both as bundled with "
            "scikit-learn; the diabetes features come mean-centred and "
            "scaled.",
            "Simulated: ten independent normal features with standard "
            "deviations 10 down to 0.3, the label set by the last; 2,000 rows.",
            "Machinery: `standarderror/ml/around.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's PCA, LogisticRegression, "
            "LinearRegression and PLSRegression, five- and ten-fold "
            "cross-validation with PCA fitted inside the folds.",
            "Where this stops: Jolliffe, \"A note on the use of principal "
            "components in regression\", *Journal of the Royal Statistical "
            "Society C* (1982); Hastie, Tibshirani and Friedman, *The "
            "Elements of Statistical Learning*, section 3.5.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("simulation seed 0; shuffled folds with "
                            "random_state 0"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The step before the model",
        f"""Thirty episodes measured models: how they are scored, fitted, read and trained. Arc VII turns to the things around a model that are applied as routine and rarely measured: the reduction of the inputs before it, the clusters found without it, the distances it relies on, the probabilities it reports and the search that chose it. The first is principal component analysis, and the sentence that justifies running it before a model: *the first principal components keep the information that matters*.

PCA finds the direction along which the inputs vary most, then the direction of most remaining variance at right angles to it, and so on. Keeping the first few and dropping the rest is the standard way to make a wide data set narrow. Its justification is the word "information": the components that vary most are taken to carry most of it.

The extreme case shows what is being assumed. Ten independent features, with standard deviations from 10 down to 0.3, and a label that depends only on the last and smallest:

{snip['sim'].markdown()}

With nine components out of ten, logistic regression is at chance. With the tenth, the one holding {s['share'][-1]:.2%} of the variance, it scores **{s['acc'][-1]:.3f}**. This is constructed, and it answers the question in principle: PCA does not know which direction predicts the label, because it never sees the label. Whether that matters on real data is a separate question.""")

    post.add(
        "The units choose the first component",
        f"""On the breast-cancer data, thirty measurements of cell nuclei:

{snip['bc'].markdown()}

On the raw measurements the first component holds **{raw['share'][0]:.1%}** of the variance and is led by {raw['top']}. That is not a discovery about tumours. Area is measured in square pixels and runs into the thousands, while smoothness and concavity are fractions near zero, so area's variance is millions of times larger and PCA, which counts variance in the units it is given, puts it first. The 95% variance rule keeps {raw['k95']} component: in effect, the size of the largest nucleus.

Standardise the features and the same rule keeps {std['k95']} components, the first holding {std['share'][0]:.1%}. The table below shows what the choice does downstream. With one raw component, logistic regression scores {raw['acc'][bks[1]]:.3f}; with standardised components it reaches {max(std['acc']):.3f}. Standardising is usually recommended before PCA for exactly this reason, and it is worth seeing that it is not a technicality. It is the decision about what "variance" means, and the components follow it.""",
        figures=[figs["f2"]])

    post.add(
        "The fourth component knows the most",
        f"""Diabetes comes with its ten features already centred and scaled, so units cannot be the problem. Each component's share of the feature variance, and the share of the target's variance it explains on its own:

{snip['dia'].markdown()}

The first component explains {d['r2'][0]:.1%} of disease progression. The second and third, which hold {d['share'][1] + d['share'][2]:.1%} of the feature variance between them, explain {d['r2'][1] + d['r2'][2]:.1%}. The fourth, smaller than either, explains **{d['r2'][3]:.1%}**. A pipeline that keeps "the first three components" keeps two that are nearly useless and drops the one that is not.

The second figure makes it a modelling choice. Regression on the first k components reaches a cross-validated R² of {pcr[3]:.3f} with three and **{pcr[4]:.3f}** with four. Partial least squares, which chooses its directions by their covariance with the target instead of by variance alone, reaches {pls[2]:.3f} with two. Jolliffe made the point in 1982 with examples from the literature: components with small variance can be the ones that predict, and dropping them by variance alone is a decision taken without looking at the target.""",
        figures=[figs["f0"], figs["f1"]])

    post.add(
        "Choosing k by the target instead",
        f"""The repair inside PCA's own terms is to treat the number of components as a setting to tune, scored on held-out data, instead of a share of variance to reach. On diabetes that repair works, and it is revealing what it chooses. Cross-validated R² is {pcr[1]:.3f} with one component, {pcr[3]:.3f} with three, {pcr[4]:.3f} with four, {pcr[5]:.3f} with five and {pcr[10]:.3f} with all ten. A search over k lands at four or more, within {pcr[10] - pcr[4]:.3f} of using every component.

So once the target is allowed to choose, it keeps at least four components, and the reduction buys little: four do almost as well as ten. What the target adds is the knowledge of where to stop. A rule that looked only at variance, and stopped where the second and third components had brought the total near two-thirds, would have stopped one component short of the one that matters. On breast cancer the same search on standardised components finds its best at {b['ks'][int(np.argmax(std['acc']))]} components, scoring {max(std['acc']):.3f}, where the 99% variance rule would have kept {std['k99']}. The two rules need not agree, because they answer different questions: one asks how much of the inputs is kept, the other how much of the prediction.""")

    post.add(
        "What PCA is good for",
        """None of this makes PCA a bad tool. It makes it a tool for a different job. When the goal is to describe the inputs (to compress images, to plot a high-dimensional data set, to remove near-collinear redundancy) variance is the right criterion, and the leading components are the right ones to keep. When the inputs happen to vary most along the directions that matter, PCA before a model costs little; the point is that nothing guarantees it.

When the goal is prediction, the number of components is a hyperparameter like any other: choose it by cross-validated performance, not by a share of variance, and fit the PCA inside the folds. And if the reduction itself should serve the prediction, a supervised method (partial least squares, or simply a regularised model on all the features) uses the target to decide what to keep.""")

    post.add(
        "Where this breaks",
        """**Linear structure only.** PCA, PLS and the models after them are linear here. A target that depends on a nonlinear function of the inputs can be invisible to every linear component.

**Small data sets.** Thirty features and ten features. With hundreds of features and few rows, the variance ranking itself is noisy, and the case for some reduction before the model is stronger, whichever kind.

**One target per data set.** A component that does not predict this target can predict another. PCA's neutrality about the target is a cost for one prediction task and a benefit when the reduced features serve many.

**The simulation is extreme by design.** It shows the mechanism, not its frequency. The diabetes result is the measure of how often it bites: once in four components, on a standard teaching data set.""")

    post.add(
        "What to keep",
        f"""1. PCA ranks directions by the variance of the inputs, in their units. On raw breast-cancer data the first component held {raw['share'][0]:.1%} of the variance and was led by {raw['top']}; the 95% rule kept {raw['k95']} component raw and {std['k95']} standardised.

2. Variance is not relevance. On diabetes, the fourth component explained {d['r2'][3]:.1%} of the target, more than the second and third together.

3. Regression on three components reached R² {pcr[3]:.3f}, on four {pcr[4]:.3f}; partial least squares reached {pls[2]:.3f} with two.

4. Before a model, choose the number of components by cross-validated performance, standardise deliberately, and consider a method that sees the target.""")

    post.add(
        "Exercise",
        """Take a data set you have reduced with PCA before modelling. For each component, compute its share of the input variance and its correlation with the target, and list them side by side.

If a component outside the ones you kept has a larger correlation than one inside, refit with the number of components chosen by cross-validation and compare. Then try the same model on all the features with a regularisation penalty, which often makes the reduction unnecessary.""")

    post.add(
        "Next",
        """PCA summarises the inputs without the target. Clustering goes further and invents a target: groups the data are said to fall into. Episode 32 asks whether k-means finds the clusters that are there, and whether the usual scores can tell when there are none.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
