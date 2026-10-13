"""ML 38: A Missing Value Is Information, Until the Process Behind It Changes.

The thirty-eighth episode of *Machine Learning, Taught Through What
Breaks*. "Missing values are a nuisance: fill them in as accurately as you
can and model the rest." Measured on the breast-cancer data with 30% of
cells removed by three mechanisms:

* Missing at random, the sentence held: iterative imputation recovered the
  values best (error 0.44 sd against the mean's 1.01) and gave the best
  model (log loss 0.091 against 0.113); a missingness indicator added
  nothing.
* When large values went missing, the indicator mattered more than the
  imputation: mean imputation plus an indicator reached log loss 0.121,
  iterative imputation 0.143, though it filled the values in better.
* When benign tumours' cells went missing nine times as often, the
  indicator model scored 0.999, above the complete data's 0.980. Scored
  after the missingness became random, it fell to 0.817; the iterative
  imputation model held 0.972.

Run: `standarderror run ml138_missing --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import rows
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

METHODS = ("mean", "mean + indicator", "5-NN imputation",
           "iterative imputation", "boosted trees, native")
KINDS = ("random", "large values", "by diagnosis", "policy change")


def compute() -> dict:
    return {"m": rows.missing_study()}


# ------------------------------------------------------------------ figures

def _bars(ax, m, d):
    w = 0.16
    for j, meth in enumerate(METHODS):
        x = np.arange(len(KINDS)) + (j - 2) * w
        ax.bar(x, [d[k][meth]["acc"] for k in KINDS], width=w,
               color=m.series[j + 1] if j < 4 else m.ink, label=meth)
    ax.axhline(d["complete"]["acc"], color=m.grid, ls=":", lw=1.2)
    ax.set_xticks(np.arange(len(KINDS)))
    ax.set_xticklabels(["at random", "large values", "by diagnosis",
                        "diagnosis, then\nat random in test"], fontsize=8.5)
    ax.set_ylim(0.78, 1.005)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)


def figures(res: dict) -> dict:
    out: dict = {}
    d = res["m"]

    out["f0"] = charts.diagram(
        lambda ax, m: _bars(ax, m, d),
        title="Which way of handling a gap wins depends on why it is there",
        subtitle=("Breast cancer with 30% of cells missing; five-fold "
                  "cross-validated accuracy over 5 masks. Dotted: the "
                  "complete data. Last group: trained with missingness by "
                  "diagnosis, tested with missingness at random."),
        xlabel="",
        ylabel="accuracy",
        figsize=(7.8, 4.8),
        source=("Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, via "
                "scikit-learn; standarderror/ml/rows.py."),
        alt=("Grouped bars. At random, iterative imputation is highest. "
             "With large values missing, mean imputation is far lowest. By "
             "diagnosis, the indicator bar reaches almost 1, above the "
             "dotted complete-data line; in the last group it drops to "
             "about 0.82."),
        caption=(f"Trained where benign cells went missing more often, the "
                 f"indicator model scored {d['by diagnosis']['mean + indicator']['acc']:.3f}; "
                 f"tested after the missingness became random, "
                 f"**{d['policy change']['mean + indicator']['acc']:.3f}**."),
        path=str(IMG / f"ml138-f0-accuracy.{EXT}"))[0]

    body = []
    for kind in KINDS:
        for meth in METHODS:
            r = d[kind][meth]
            rm = d.get(kind, {}).get("_rmse", {}).get(
                "mean" if meth in ("mean", "mean + indicator") else meth)
            body.append([kind if meth == "mean" else "", meth,
                         f"{r['nll']:.3f}", f"{r['acc']:.3f}",
                         "" if rm is None or meth.startswith("boosted")
                         else f"{rm:.2f}"])
    out["f1"] = charts.table_image(
        body,
        header=["missing", "handled by", "log loss", "accuracy",
                "imputation error (sd)"],
        title="Filling in well and predicting well are different scores",
        subtitle=(f"Logistic regression after each imputation; boosted trees "
                  f"with their own handling of missing values. Complete "
                  f"data: log loss {d['complete']['nll']:.3f}, accuracy "
                  f"{d['complete']['acc']:.3f}."),
        source=("Breast Cancer Wisconsin (Diagnostic), CC BY 4.0; "
                "standarderror/ml/rows.py."),
        alt=("A twenty-row table, five methods under four conditions, with "
             "log loss, accuracy and imputation error for each."),
        caption=("Imputation error is the root mean squared error on the "
                 "missing cells, in standard deviations of each feature; "
                 "the policy-change rows reuse the models trained 'by "
                 "diagnosis'."),
        align="llrrr",
        path=str(IMG / f"ml138-f1-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    d = res["m"]

    def bars(kind, keys):
        def f(panel, m):
            panel.bar(range(len(keys)), [d[kind][k]["acc"] for k in keys],
                      color=[m.series[1], m.series[2], m.series[3]][:len(keys)])
            panel.set_ylim(0.8, 1.0)
        return f

    return charts.lecture_hero(
        series=SERIES_TAG, episode=38,
        headline="What a gap says",
        panels=[(bars("random", ("mean", "iterative imputation")),
                 f"{d['random']['iterative imputation']['nll']:.3f}",
                 "best imputer, log loss"),
                (bars("by diagnosis", ("iterative imputation", "mean + indicator")),
                 f"{d['by diagnosis']['mean + indicator']['acc']:.3f}",
                 "indicator, by diagnosis"),
                (bars("policy change", ("iterative imputation", "mean + indicator")),
                 f"{d['policy change']['mean + indicator']['acc']:.3f}",
                 "after the policy changed")],
        note=("Whether a value is missing can say more than the value "
              "would have. A model that learns that is learning the process "
              "that recorded the data, and it breaks when the process does."),
        alt=("Three hand-drawn frames of bars: a taller second bar; a second "
             "bar near the top; a second bar that has dropped."),
        path=str(IMG / f"ml138-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["random"] = s.run("""
        from standarderror.ml import rows

        m = rows.missing_study()          # 30% of cells, 5 masks
        print(f"complete data: log loss {m['complete']['nll']:.3f}")
        for kind in ("random", "large values"):
            print(f"missing: {kind}")
            for meth, r in m[kind].items():
                if not meth.startswith("_"):
                    print(f"   {meth:22s} log loss {r['nll']:.3f}, "
                          f"accuracy {r['acc']:.3f}")
        """)
    out["diag"] = s.run("""
        for kind in ("by diagnosis", "policy change"):
            print(f"missing: {kind}")
            for meth, r in m[kind].items():
                if not meth.startswith("_"):
                    print(f"   {meth:22s} log loss {r['nll']:.3f}, "
                          f"accuracy {r['acc']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    d = res["m"]
    c = d["complete"]
    R, L, D, P = (d[k] for k in KINDS)
    rr, rl = R["_rmse"], L["_rmse"]
    it, ind, mean, knn, hgb = ("iterative imputation", "mean + indicator",
                               "mean", "5-NN imputation",
                               "boosted trees, native")

    # The spine, asserted rather than trusted.
    assert rr["iterative imputation"] < rr["5-NN imputation"] < rr["mean"]
    assert R[it]["nll"] < R[knn]["nll"] and R[it]["nll"] < R[mean]["nll"]
    assert R[ind]["nll"] > R[mean]["nll"]
    assert R[knn]["nll"] > R[mean]["nll"] and rr["5-NN imputation"] < rr["mean"]
    assert 0.015 < R[knn]["nll"] - R[it]["nll"] < 0.035
    assert c["nll"] < R[it]["nll"]
    assert L[mean]["acc"] < 0.89 and L[mean]["nll"] > 2 * L[ind]["nll"]
    assert L[ind]["nll"] < L[it]["nll"] and rl["iterative imputation"] < rl["mean"]
    assert L[it]["nll"] < L[hgb]["nll"] < L[mean]["nll"]
    assert D[ind]["acc"] > 0.995 and D[ind]["acc"] > c["acc"]
    assert D[ind]["nll"] < c["nll"] / 5
    assert D[hgb]["acc"] > c["acc"] and D[knn]["acc"] > c["acc"]
    assert P[ind]["acc"] < 0.83 and P[hgb]["acc"] < 0.90
    assert abs(P[it]["acc"] - R[it]["acc"]) < 0.005
    assert P[it]["acc"] > P[ind]["acc"] + 0.15

    post = Post(
        title=(f"{SERIES_TAG} 38: A Missing Value Is Information, Until the "
               f"Process Behind It Changes"),
        slug="ml-38-missing",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=38,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Imputation treats a gap as a lost number. Sometimes the "
                  "gap is a record of a decision, worth more than the number, "
                  "and the model that uses it depends on the decision staying "
                  "the same."),
        summary=(
            f"The breast-cancer data with 30% of cells removed three ways. "
            f"At random, the imputation that recovered the values best "
            f"(iterative, error {rr['iterative imputation']:.2f} sd) gave "
            f"the best model, log loss {R[it]['nll']:.3f} against "
            f"{R[mean]['nll']:.3f} for the mean. When large values went "
            f"missing, mean imputation plus a missingness indicator reached "
            f"{L[ind]['nll']:.3f}, better than iterative imputation's "
            f"{L[it]['nll']:.3f}. When benign tumours' cells went missing "
            f"more often, the indicator model scored {D[ind]['acc']:.3f}, "
            f"above the complete data's {c['acc']:.3f}, and "
            f"{P[ind]['acc']:.3f} once the test rows' gaps were random."),
        tags=["machine-learning", "data-science", "statistics",
              "missing-data", "imputation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, as bundled "
            "with scikit-learn; 569 tumours, 30 features. 30% of cells "
            "removed: independently at random; with probability rising in "
            "the cell's standardised value (logistic in 2z − 1.2, scaled to "
            "30%); or with probability nine times higher for benign "
            "tumours. Five masks (seeds 0 to 4), test-time random masks "
            "from seeds 100 to 104.",
            "Machinery: `standarderror/ml/rows.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's SimpleImputer (with and "
            "without add_indicator), KNNImputer, IterativeImputer (10 "
            "rounds), StandardScaler and LogisticRegression (C = 0.3), and "
            "HistGradientBoostingClassifier (200 rounds, rate 0.05) on the "
            "raw gaps; five stratified folds, imputers fitted inside them.",
            "Where this stops: Rubin, \"Inference and missing data\", "
            "*Biometrika* (1976); Josse, Prost, Scornet and Varoquaux, \"On "
            "the consistency of supervised learning with missing values\" "
            "(2019); Agniel, Kohane and Weber, \"Biases in electronic health "
            "record data due to processes within the healthcare system\", "
            "*BMJ* (2018).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("masks and folds from seeds 0 to 4; about five "
                            "minutes of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Gaps in the table",
        f"""Episodes 36 and 37 were about rows that resemble each other because of where they came from. Missing values are about rows that are incomplete because of where they came from. A blood test not ordered, a survey question skipped, a sensor that dropped out: each leaves a gap, and the standard advice treats every gap the same way, as the sentence under test puts it: *missing values are a nuisance; fill them in as accurately as you can and model the rest*.

Rubin's classification from 1976 says when that is safe. Values missing completely at random, independently of everything, are a pure loss of data. Values missing at random given the observed data can be filled in from the rest. Values whose absence depends on the value itself, or on the outcome, are missing not at random, and the gap carries information.

The breast-cancer data are complete, so the gaps can be made to order. In each of three ways, 30% of the cells are removed, and five strategies are compared: fill each gap with the column mean; the same plus an indicator column per feature saying whether it was missing; fill from the five most similar tumours; fill by iterative regression on the other features; and boosted trees, which handle a missing value directly by learning which side of each split it should go. Every imputer is fitted inside the cross-validation folds.

{snip['random'].markdown()}

Missing completely at random, the sentence mostly holds. Iterative imputation recovers the removed values best, with a root mean squared error of **{rr['iterative imputation']:.2f}** standard deviations against {rr['mean']:.2f} for the column mean, and it gives the best model: log loss **{R[it]['nll']:.3f}**, against {R[mean]['nll']:.3f} for the mean and {c['nll']:.3f} for the complete data. The thirty features are strongly correlated, so most of a missing value can be read off the others. The indicator columns add nothing, because there is nothing in a random gap to indicate; they cost a little, {R[ind]['nll']:.3f}. The ordering is not perfect, though: five-nearest-neighbour imputation recovers the values much better than the mean ({rr['5-NN imputation']:.2f} sd against {rr['mean']:.2f}) and still gives a slightly worse model ({R[knn]['nll']:.3f} against {R[mean]['nll']:.3f}).""",
        figures=[figs["f0"]])

    post.add(
        "When the gap knows the value",
        f"""In the second setting the chance that a cell goes missing rises with its value: in every feature, the larger measurements are the ones that tend to be absent. The figures in the second half of the block above change sharply. Mean imputation now replaces exactly the large values with an average one, systematically, and the model falls to accuracy **{L[mean]['acc']:.3f}** and log loss {L[mean]['nll']:.3f}.

Iterative imputation does better, {L[it]['nll']:.3f}, but it can only fill a gap with what the observed features predict, and the observed features were selected by the same mechanism. The simplest fix wins: mean imputation plus an indicator, log loss **{L[ind]['nll']:.3f}**, even though, as an imputation, it is the worst of the three ({rl['mean']:.2f} sd against {rl['iterative imputation']:.2f}). The indicator says "this value was probably large", and logistic regression gives that its own weight. Boosted trees, which learn a direction for missing values at each split, reach {L[hgb]['nll']:.3f}, between iterative imputation and the plain mean.

That is the first half of the sentence failing: filling in accurately and predicting well are different objectives. Josse and colleagues showed that when the gaps are not informative, even imputing a constant is consistent for a learner flexible enough to recover the pattern from where the constant appears, and recommended letting trees treat missingness as its own branch, which also captures informative gaps. Logistic regression is not that flexible, so here the indicator has to be handed to it explicitly; a careful imputation that hides the pattern loses it.""",
        figures=[figs["f1"]])

    post.add(
        "When the gap knows the answer",
        f"""The third mechanism is the one hospitals produce. A cell is missing nine times as often for a benign tumour as for a malignant one, the way a follow-up test is ordered mainly when something looks wrong. Now the gap depends on the outcome itself:

{snip['diag'].markdown()}

With an indicator, logistic regression scores **{D[ind]['acc']:.3f}**, with log loss {D[ind]['nll']:.3f}: better than the same model on the complete data, {c['acc']:.3f}, because the pattern of missing tests says almost everything about the diagnosis. Boosted trees find the same thing on their own ({D[hgb]['acc']:.3f}), and five-nearest-neighbour imputation also beats the complete data ({D[knn]['acc']:.3f}), presumably because the neighbours it borrows from are chosen using the cells that happened to be observed. Agniel, Kohane and Weber found the real version in hospital records: for most of the laboratory tests they studied, the timing of the test predicted three-year survival better than its result.

The model has learned the clinicians' behaviour, which is a legitimate signal for as long as the clinicians behave the same way. The last block shows what happens when they do not. The same trained models are scored on tumours whose cells are missing at random, as if the testing policy had changed: every test now ordered, or lost, regardless of suspicion. The indicator model falls to **{P[ind]['acc']:.3f}**, the boosted trees to {P[hgb]['acc']:.3f}. Iterative imputation, which never looked at the pattern, scores {P[it]['acc']:.3f}, the same as it scored with random gaps in training.""")

    post.add(
        "Choosing a strategy",
        """The three settings give three different winners, and no strategy is safe by default. If the gaps are close to random, impute well and leave the pattern out. If the gaps depend on the missing values, give the model the pattern, through indicators or a learner that handles missing values natively. If the gaps depend on a decision made by people or systems upstream, the pattern will probably be the strongest feature in the model, and its value lasts only as long as the decision rule.

Which case applies cannot be read from the data alone; missing-not-at-random is, by definition, not identifiable from what was observed. It has to come from knowing how the data were collected. What can be measured is the dependence on the pattern: fit the model with and without indicators, and if the indicators add a great deal, find out why the values are missing before trusting the gain.""")

    post.add(
        "Where this breaks",
        """**Artificial mechanisms.** The three masks are clean versions of what real data mix together. Real missingness is usually partly random, partly value-dependent and partly decided by someone, in proportions nobody wrote down.

**Thirty correlated features.** Breast-cancer measurements are redundant, which is why iterative imputation does so well at random. With weakly related features, every imputation is closer to the mean, and the differences shrink.

**One rate.** Thirty percent of cells, everywhere. Real gaps concentrate in a few columns and a few rows, and dropping such columns or rows is sometimes the better choice.

**Small data.** With 569 tumours, the differences between strategies at random are about two hundredths of log loss; under the informative mechanisms they are several times larger, which is the point.""")

    post.add(
        "What to keep",
        f"""1. Missing at random, the best imputation gave the best model: iterative imputation, error {rr['iterative imputation']:.2f} sd, log loss {R[it]['nll']:.3f}; the mean, {rr['mean']:.2f} sd, {R[mean]['nll']:.3f}. Indicators added nothing.

2. When large values went missing, the indicator mattered more than the imputation: {L[ind]['nll']:.3f} with mean imputation and indicators, {L[it]['nll']:.3f} with iterative imputation alone.

3. When the gaps followed the diagnosis, the indicator model scored {D[ind]['acc']:.3f}, above the complete data, and {P[ind]['acc']:.3f} once the gaps became random.

4. Ask why each value is missing before choosing a strategy, and treat a large gain from missingness indicators as a dependence on the process that collected the data.""")

    post.add(
        "Exercise",
        """Take a model trained on data with missing values. Fit it twice, once with imputation alone and once with missingness indicators added, and compare the cross-validated scores.

If the indicators help substantially, list the features whose missingness matters most and, for each, find out who or what decides whether it is recorded. Then score the indicator model on data where those gaps are redistributed at random, which is what a change in that decision would look like.""")

    post.add(
        "Next",
        """A missing value is one way the data-collection process leaves its fingerprint. A missing label is a stronger one. Episode 39 trains a model on the outcomes of applicants who were accepted, the only ones whose outcomes anyone saw, and asks what it can say about the applicants who were turned away.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
