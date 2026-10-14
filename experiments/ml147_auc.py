"""ML 47: The Model With the Best AUC Was Never the Best When Misses Cost Ten Times More.

The forty-seventh episode of *Machine Learning, Taught Through What
Breaks*. "The model with the higher AUC makes better decisions." Measured
with five models on episode 4's simulated task, ten training sets each:

* A random forest had the highest AUC in all ten (0.754) and the lowest
  cost in all ten when both errors cost the same.
* When a miss cost ten times a false alarm, it was the cheapest model in
  none: boosted trees were cheapest in seven and logistic regression, last
  by AUC at 0.677, in three. At twenty times, logistic regression and the
  boosted trees split the ten.
* The decision at those costs is which cases to clear, and among the 2%
  of cases each model rated safest, the forest's held 5.6% positives,
  twice the logistic model's 2.8%.

Run: `standarderror run ml147_auc --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import decide
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

MODELS = ("logistic", "naive Bayes", "15-NN", "forest", "boosted trees")


def compute() -> dict:
    return {"a": decide.auc_and_decisions()}


# ------------------------------------------------------------------ figures

def _wins(ax, m, a):
    cols = ["auc", "cost1", "cost2", "cost5", "cost10", "cost20",
            "top0.01", "top0.05", "top0.2"]
    labels = ["AUC", "cost r=1", "r=2", "r=5", "r=10", "r=20",
              "top 1%", "top 5%", "top 20%"]
    bottom = np.zeros(len(cols))
    for j, k in enumerate(MODELS):
        v = np.array([a["wins"][c][k] for c in cols])
        ax.bar(range(len(cols)), v, bottom=bottom, color=m.series[j % 4 + 1]
               if j < 4 else m.ink, label=k)
        bottom += v
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=5,
              bbox_to_anchor=(0.5, -0.12))


def figures(res: dict) -> dict:
    out: dict = {}
    a = res["a"]
    mean = a["mean"]

    out["f0"] = charts.diagram(
        lambda ax, m: _wins(ax, m, a),
        title="Which model wins depends on the decision",
        subtitle=("Share of 10 training sets in which each model was best, "
                  "by AUC, by cost at each cost ratio r (a miss costs r false "
                  "alarms), and by precision among the top 1%, 5% and 20% of "
                  "cases."),
        xlabel="",
        ylabel="share of training sets best",
        figsize=(7.8, 4.8),
        source="Simulated (episode 4's task); standarderror/ml/decide.py.",
        alt=("Stacked bars. The forest wins every training set by AUC and "
             "at r = 1, and none at r = 10 or 20, where boosted trees and "
             "logistic regression share the wins."),
        caption=(f"Best by AUC in 10 of 10 training sets, the forest was the "
                 f"cheapest model at r = 10 in **"
                 f"{a['wins']['cost10']['forest'] * 10:.0f}**; logistic "
                 f"regression, last by AUC, in "
                 f"{a['wins']['cost10']['logistic'] * 10:.0f}."),
        path=str(IMG / f"ml147-f0-wins.{EXT}"))[0]

    body = [[k, f"{mean[k]['auc']:.3f}", f"{mean[k]['cost1']:.3f}",
             f"{mean[k]['cost10']:.3f}", f"{mean[k]['cost20']:.3f}",
             f"{mean[k]['top0.01']:.3f}", f"{mean[k]['bottom0.02']:.3f}",
             f"{mean[k]['bottom0.05']:.3f}"] for k in MODELS]
    body.append(["true probabilities", f"{a['bayes']['auc']:.3f}",
                 f"{a['bayes']['cost1']:.3f}", f"{a['bayes']['cost10']:.3f}",
                 f"{a['bayes']['cost20']:.3f}", "", "", ""])
    out["f1"] = charts.table_image(
        body,
        header=["model", "AUC", "cost, r=1", "cost, r=10", "cost, r=20",
                "positives in top 1%", "positives in safest 2%",
                "positives in safest 5%"],
        title="One ranking summarised, three decisions made",
        subtitle=("Means over 10 training sets of 2,000 rows, 50,000 test "
                  "rows. Cost per case at the best threshold for each ratio; "
                  "flagging every case costs about 0.497."),
        source="Simulated (episode 4's task); standarderror/ml/decide.py.",
        alt=("A six-row table: the forest has the best AUC and the lowest "
             "cost at r = 1; logistic regression and boosted trees have the "
             "fewest positives among the 2% of cases rated safest and the lowest "
             "costs at r = 10 and 20."),
        caption=("'Safest' is the share of cases a model scores lowest; the "
                 "fraction of positives among them is what clearing them "
                 "costs."),
        align="lrrrrrrr",
        path=str(IMG / f"ml147-f1-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    a = res["a"]
    mean = a["mean"]

    def p1(panel, m):
        panel.bar(range(5), [mean[k]["auc"] for k in MODELS],
                  color=[m.series[1], m.series[2], m.series[3], m.series[4], m.ink])
        panel.set_ylim(0.65, 0.76)

    def p2(panel, m):
        panel.bar(range(5), [mean[k]["cost10"] for k in MODELS],
                  color=[m.series[1], m.series[2], m.series[3], m.series[4], m.ink])
        panel.set_ylim(0.47, 0.492)

    def p3(panel, m):
        panel.bar([0, 1], [mean["forest"]["bottom0.02"],
                           mean["logistic"]["bottom0.02"]],
                  color=[m.series[4], m.series[1]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=47,
        headline="One number, many decisions",
        panels=[(p1, f"{mean['forest']['auc']:.3f}", "forest AUC, the best"),
                (p2, f"{a['wins']['cost10']['forest'] * 10:.0f} of 10",
                 "forest best at r = 10"),
                (p3, f"{mean['forest']['bottom0.02']:.1%}",
                 "forest's safest 2%")],
        note=("AUC averages a model's ranking over every threshold. A "
              "decision uses one, and the model that ranks best on average "
              "can rank worst where the decision is made."),
        alt=("Three hand-drawn frames of bars: five rising to a tallest "
             "fourth; five with the fourth among the tallest; two bars, the "
             "first twice the second."),
        path=str(IMG / f"ml147-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["main"] = s.run("""
        from standarderror.ml import decide

        a = decide.auc_and_decisions()   # 10 training sets of 2,000 rows
        for k, v in a["mean"].items():
            print(f"{k:14s} AUC {v['auc']:.3f} | best cost at r=1 "
                  f"{v['cost1']:.3f}, r=10 {v['cost10']:.3f}, r=20 {v['cost20']:.3f}")
        for q in ("auc", "cost1", "cost10", "cost20"):
            w = {k: v for k, v in a["wins"][q].items() if v}
            print(f"best by {q}: {w}")
        """)
    out["tail"] = s.run("""
        for k, v in a["mean"].items():
            print(f"{k:14s} positives among its safest 2%: "
                  f"{v['bottom0.02']:.1%}, safest 5%: {v['bottom0.05']:.1%}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    a = res["a"]
    mean, wins, bayes = a["mean"], a["wins"], a["bayes"]
    fo, lg, bt = mean["forest"], mean["logistic"], mean["boosted trees"]

    # The spine, asserted rather than trusted.
    assert wins["auc"]["forest"] == 1.0 and wins["cost1"]["forest"] == 1.0
    assert min(mean, key=lambda k: mean[k]["auc"]) == "logistic"
    assert wins["cost10"]["forest"] == 0 and wins["cost20"]["forest"] == 0
    assert wins["cost10"]["boosted trees"] == 0.7 and wins["cost10"]["logistic"] == 0.3
    assert wins["cost20"]["logistic"] == 0.5 and wins["cost20"]["boosted trees"] == 0.5
    assert lg["cost10"] < fo["cost10"] and lg["cost20"] < fo["cost20"]
    assert fo["bottom0.02"] > 1.9 * lg["bottom0.02"]
    assert abs(lg["bottom0.02"] - bt["bottom0.02"]) < 0.002
    assert wins["top0.05"]["forest"] + wins["top0.05"]["boosted trees"] == 1.0
    assert wins["pauc_sens"]["logistic"] == 0
    assert all(v["cost10"] < 0.497 for v in mean.values())
    nb = mean["naive Bayes"]
    assert nb["auc"] > lg["auc"] + 0.02 and nb["top0.01"] > lg["top0.01"]
    assert nb["bottom0.02"] == max(v["bottom0.02"] for v in mean.values())
    assert nb["cost10"] == max(v["cost10"] for v in mean.values())
    assert wins["cost2"]["forest"] == 0.9 and wins["cost5"]["forest"] == 0.2
    assert wins["cost5"]["boosted trees"] == 0.7 and wins["cost5"]["logistic"] == 0.1

    post = Post(
        title=(f"{SERIES_TAG} 47: The Model With the Best AUC Was Never the "
               f"Best When Misses Cost Ten Times More"),
        slug="ml-47-auc-decisions",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=47,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("AUC summarises a ranking over every threshold at once. A "
                  "decision uses one threshold, and the model that ranks best "
                  "on average is not always the one that ranks best there."),
        summary=(
            f"Five models on a simulated task, ten training sets each. A "
            f"random forest had the highest AUC every time ({fo['auc']:.3f}) "
            f"and the lowest cost every time when both errors cost the same. "
            f"When a miss cost ten times a false alarm it was the cheapest "
            f"model in none of the ten: boosted trees in "
            f"{wins['cost10']['boosted trees'] * 10:.0f} and logistic "
            f"regression, last by AUC at {lg['auc']:.3f}, in "
            f"{wins['cost10']['logistic'] * 10:.0f}. Among the 2% of cases "
            f"each model rated safest, the forest's held "
            f"{fo['bottom0.02']:.1%} positives and the logistic model's "
            f"{lg['bottom0.02']:.1%}."),
        tags=["machine-learning", "data-science", "statistics",
              "model-evaluation", "roc-auc", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: episode 4's two-class Gaussian task in 10 "
            "dimensions; ten training sets of 2,000 rows (seeds 10 to 19), "
            "50,000 test rows with exact posterior probabilities.",
            "Machinery: `standarderror/ml/decide.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's LogisticRegression, "
            "GaussianNB, KNeighborsClassifier (k = 15), "
            "RandomForestClassifier (300 trees, leaves of 5), "
            "HistGradientBoostingClassifier (300 rounds), roc_auc_score "
            "(with max_fpr for the partial AUC).",
            "Where this stops: Hand, \"Measuring classifier performance: a "
            "coherent alternative to the area under the ROC curve\", "
            "*Machine Learning* (2009); Provost and Fawcett, \"Robust "
            "classification for imprecise environments\", *Machine Learning* "
            "(2001); McClish, \"Analyzing a portion of the ROC curve\", "
            "*Medical Decision Making* (1989).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("training seeds 10 to 19, model random_state 0; "
                            "about a minute of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Choosing the model by one number",
        f"""Episode 46 set the threshold for one model from the costs of its mistakes. Before that, a model has usually been chosen, and the usual basis is a single number that ignores thresholds altogether: the area under the ROC curve, the probability that a random positive is scored above a random negative. Its appeal is that it does not depend on the operating point, and the inference drawn from it is the sentence under test: *the model with the higher AUC makes better decisions*.

Five models on episode 4's task (logistic regression, naive Bayes, fifteen-nearest-neighbours, a random forest and boosted trees) were trained on ten different sets of 2,000 rows. Each was scored by AUC, and by the cost of its best decision at several cost ratios, with each model given its own best threshold, so that only the ranking it produces is being compared.

{snip['main'].markdown()}

When both errors cost the same, AUC chose correctly: the forest had the highest AUC in all ten training sets and the lowest cost in all ten. When a miss cost ten times a false alarm, the forest was the cheapest model in **none** of them. The boosted trees were cheapest in {wins['cost10']['boosted trees'] * 10:.0f}, and logistic regression, whose mean AUC of **{lg['auc']:.3f}** was the lowest of the five by a wide margin, in {wins['cost10']['logistic'] * 10:.0f}. At twenty times, logistic regression and the boosted trees split the ten.""",
        figures=[figs["f0"]])

    post.add(
        "Where the decision is made",
        f"""Episode 46 showed that when misses are expensive the decision is to flag nearly everything, and the model's job shrinks to finding the few cases safe enough to clear. What matters is then the ranking at the very bottom: how pure the cases a model rates safest really are.

{snip['tail'].markdown()}

Among the 2% of cases each model rated safest, the forest's held **{fo['bottom0.02']:.1%}** positives, and logistic regression's {lg['bottom0.02']:.1%}, half as many. The boosted trees matched logistic regression there ({bt['bottom0.02']:.1%}). The forest's leaves average at least five training rows each, which plausibly blunts its scores at the extremes, where the linear model's scores keep going; the measurement shows the effect, not that cause.

The AUC is an average over the whole ranking, and these regions are a small part of it. A model can lead on average and trail in the corner the decision uses. Hand made the stronger point in 2009: comparing AUCs is equivalent to comparing expected costs under a distribution of cost ratios that depends on the classifier itself, so two models' AUCs are averages over different, implicit beliefs about the costs. Using the AUC to choose between them assumes something about the decision that nobody wrote down.""",
        figures=[figs["f1"]])

    post.add(
        "Two models, two ends",
        f"""Naive Bayes shows the same thing from the other side. Its AUC, {mean['naive Bayes']['auc']:.3f}, was clearly above logistic regression's {lg['auc']:.3f}, and among the 1% of cases it rated riskiest it found {mean['naive Bayes']['top0.01']:.1%} positives against logistic regression's {lg['top0.01']:.1%}. At the safe end it was the worst of the five: {mean['naive Bayes']['bottom0.02']:.1%} positives among its safest 2%, and the highest cost of all at r = 10 ({mean['naive Bayes']['cost10']:.3f}). Two models can each be better than the other, at opposite ends of the same ranking, and the AUC adds the two ends together.

The switch is not abrupt. As the cost ratio rises from 1 to 20, the forest's wins go from {wins['cost1']['forest'] * 10:.0f} in ten to {wins['cost2']['forest'] * 10:.0f}, {wins['cost5']['forest'] * 10:.0f} and then none, while the boosted trees' go from none to {wins['cost5']['boosted trees'] * 10:.0f} at r = 5, and logistic regression enters at r = 5 and takes half the wins at r = 20. Each ratio corresponds to a different point on the ROC curve, and the curves cross somewhere along the way. The training-set-to-training-set noise matters too: the AUCs differed by more than their standard deviations across training sets, while at r = 10 the costs of the leading models differed by about one standard deviation, which is why the wins are split rather than unanimous.""")

    post.add(
        "Better summaries, and the real one",
        f"""The obvious repair is a partial AUC over the region that matters, which McClish proposed in 1989. Restricted to the high-sensitivity end, false-negative rates of 10% or less, it chose the forest in {wins['pauc_sens']['forest'] * 10:.0f} training sets and the boosted trees in {wins['pauc_sens']['boosted trees'] * 10:.0f}, and never logistic regression, plausibly because the decisions at r = 10 and 20 sit further out still, clearing only a few percent of cases. A partial AUC is only as good as the guess about where the operating point is.

The direct summary is the one the decision uses: the cost at the operating point, or the precision in the top k when capacity is fixed. At the top end the AUC's ranking roughly held: among the 5% of cases each model rated riskiest, the forest or the boosted trees, the top two by AUC, had the highest precision in every training set, the forest alone in {wins['top0.05']['forest'] * 10:.0f}. Which end matters is a fact about the use, not about the model.

The differences in cost at r = 10 are small, up to about a hundredth per case, because, as episode 46 found, even the true probabilities save only about a tenth of the cost of flagging everyone ({bayes['cost10']:.3f} at their best threshold here, against about 0.497). Small differences in a small saving are still the differences that decide which model to deploy.""")

    post.add(
        "Where this breaks",
        """**One task with overlapping classes.** On tasks where models separate the classes well, their ROC curves may cross less often; that was not measured here.

**Each model at its own best threshold.** Real deployments choose the threshold on validation data, as in episode 46, which adds noise to every cost and can change the winner in close cases.

**Ten training sets.** The counts of wins are out of ten; the forest's zero at r = 10 and its ten by AUC are the robust part.

**Accuracy-like decisions only.** Decisions that depend on calibrated probabilities, not just the ranking, add calibration to the comparison, as episode 46 measured.""")

    post.add(
        "What to keep",
        f"""1. The forest had the highest AUC in all ten training sets and the lowest cost when both errors cost the same.

2. When a miss cost ten times a false alarm it was cheapest in none; logistic regression, last by AUC, was cheapest in {wins['cost10']['logistic'] * 10:.0f} and at twenty times in {wins['cost20']['logistic'] * 10:.0f}.

3. The decision at those costs depends on the cases a model rates safest, where the forest put {fo['bottom0.02']:.1%} positives and logistic regression {lg['bottom0.02']:.1%}.

4. Choose a model by its cost at the operating point you will use, or by precision at the capacity you have, and use the AUC only as a first filter.""")

    post.add(
        "Exercise",
        """Take two models you compared by AUC and plot both ROC curves on the same axes. Mark the point each would operate at, from the costs or the capacity of the decision.

If the curves cross, compute each model's cost or precision at that point and see whether the AUC's choice survives. Then sort each model's scores and compare the positive rate among the 2% it rates safest and the 2% it rates riskiest.""")

    post.add(
        "Next",
        """Episodes 46 and 47 used a model's ranking to decide whom to flag. A different decision asks what to change: which feature, if moved, would move the outcome. Episode 48 compares what a model relies on with what intervening would actually do.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
