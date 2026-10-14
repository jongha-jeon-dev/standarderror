"""ML 46: The Right Threshold Is a Cost Ratio, If the Probabilities Are Calibrated.

The forty-sixth episode of *Machine Learning, Taught Through What
Breaks*, opening Arc X (from prediction to decision). "A classifier
predicts the positive class when its probability is above 0.5." Measured
on episode 4's simulated task, where the true posterior is known, with a
miss costing r times a false alarm:

* At r = 10, flagging at 0.5 cost 1.4 to 2.0 per case, 2.9 to 4.0 times
  what flagging every case costs (0.497).
* The cost-derived threshold 1/(1 + r) came within 0.003 of the best
  threshold in hindsight for a recalibrated model; for overconfident
  boosted trees it cost 0.762, worse than flagging everyone.
* A threshold tuned on 1,000 fresh rows rescued the boosted trees, and at
  r = 20 was worse than flagging everyone for three of five models. Even
  the true posterior beat flagging everyone by only 11% at r = 10.

Run: `standarderror run ml146_threshold --publish`
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


def compute() -> dict:
    return {"t": decide.threshold_costs()}


# ------------------------------------------------------------------ figures

def _lines(ax, m, t):
    rs = t["ratios"]
    ax.plot(rs, [t["flag_all"][r] for r in rs], color=m.grid, ls="--", lw=1.5,
            label="flag every case")
    ax.plot(rs, [t["bayes"][r] for r in rs], color=m.ink, lw=2,
            label="true probabilities, derived threshold")
    spec = (("boosted trees", "half", m.series[1], ":", "boosted trees at 0.5"),
            ("boosted trees", "derived", m.series[1], "-",
             "boosted trees, derived threshold"),
            ("recalibrated boosted trees", "derived", m.series[2], "-",
             "recalibrated, derived threshold"),
            ("logistic", "derived", m.series[3], "-", "logistic, derived threshold"))
    for model, key, col, ls, lab in spec:
        ax.plot(rs, [t["models"][model][r][key] for r in rs], color=col, ls=ls,
                marker="o", label=lab)
    ax.set_xscale("log")
    ax.set_xticks(rs)
    ax.set_xticklabels([str(r) for r in rs])
    ax.set_ylim(0.2, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def figures(res: dict) -> dict:
    out: dict = {}
    t = res["t"]
    bt, rc = t["models"]["boosted trees"], t["models"]["recalibrated boosted trees"]

    out["f0"] = charts.diagram(
        lambda ax, m: _lines(ax, m, t),
        title="What a decision costs, by how much more a miss costs",
        subtitle=("Cost per case on 50,000 test rows when a missed positive "
                  "costs r and a false alarm 1; models trained on 2,000 rows, "
                  "means over 5 training sets. The 0.5 line for boosted "
                  "trees leaves the chart above r = 5."),
        xlabel="cost of a miss relative to a false alarm (log scale)",
        ylabel="cost per case",
        figsize=(7.6, 4.8),
        source="Simulated (episode 4's task); standarderror/ml/decide.py.",
        alt=("Lines of cost against the cost ratio. Flagging every case is "
             "flat near 0.5. With the derived threshold the recalibrated and "
             "logistic models stay below it, closing in at high r; for raw "
             "boosted trees it climbs far above it."),
        caption=(f"At r = 10 the derived threshold cost the recalibrated "
                 f"model {rc[10]['derived']:.3f} and the raw boosted trees "
                 f"**{bt[10]['derived']:.3f}**; flagging every case costs "
                 f"{t['flag_all'][10]:.3f}."),
        path=str(IMG / f"ml146-f0-cost.{EXT}"))[0]

    body = []
    for name, v in t["models"].items():
        for r in (1, 10, 20):
            x = v[r]
            body.append([name if r == 1 else "", f"{r}", f"{x['half']:.3f}",
                         f"{x['derived']:.3f}", f"{x['tuned']:.3f}",
                         f"{x['tuned_t']:.3f}", f"{x['best']:.3f}"])
    out["f1"] = charts.table_image(
        body,
        header=["model", "r", "at 0.5", "at 1/(1+r)", "tuned on 1,000",
                "tuned threshold", "best in hindsight"],
        title="Three ways to choose a threshold",
        subtitle=(f"Cost per case, means over 5 training sets. Flagging every "
                  f"case costs {t['flag_all'][1]:.3f} at any r; the true "
                  f"probabilities cost {t['bayes'][1]:.3f}, "
                  f"{t['bayes'][10]:.3f} and {t['bayes'][20]:.3f} at r = 1, "
                  f"10 and 20."),
        source="Simulated (episode 4's task); standarderror/ml/decide.py.",
        alt=("A fifteen-row table, five models at three cost ratios, with "
             "the cost at four thresholds and the tuned threshold's value."),
        caption=("'Tuned' is the threshold that was cheapest on 1,000 fresh "
                 "rows; 'best in hindsight' is the cheapest on the test rows "
                 "themselves, which no one gets to choose."),
        align="lrrrrrr",
        path=str(IMG / f"ml146-f1-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    t = res["t"]
    bt, rc = t["models"]["boosted trees"], t["models"]["recalibrated boosted trees"]

    def p1(panel, m):
        panel.bar([0, 1], [t["models"]["logistic"][10]["half"], t["flag_all"][10]],
                  color=[m.series[1], m.grid])

    def p2(panel, m):
        panel.bar([0, 1, 2], [bt[10]["derived"], rc[10]["derived"],
                              t["flag_all"][10]],
                  color=[m.series[1], m.series[2], m.grid])

    def p3(panel, m):
        panel.bar([0, 1], [t["flag_all"][10], t["bayes"][10]],
                  color=[m.grid, m.ink])
        panel.set_ylim(0.3, 0.52)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=46,
        headline="Where to cut",
        panels=[(p1, f"{t['models']['logistic'][10]['half']:.2f}",
                 "cost at 0.5, r = 10"),
                (p2, f"{bt[10]['derived']:.3f}", "overconfident, 1/(1+r)"),
                (p3, f"{1 - t['bayes'][10] / t['flag_all'][10]:.0%}",
                 "true probabilities saved")],
        note=("A threshold is a price: where the probability crosses the "
              "ratio of costs. Using it needs probabilities that mean what "
              "they say, and the saving can be small even then."),
        alt=("Three hand-drawn frames: a tall bar and a short one; one tall "
             "bar and two short; two bars of similar height."),
        path=str(IMG / f"ml146-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["half"] = s.run("""
        from standarderror.ml import decide

        t = decide.threshold_costs()   # 5 training sets of 2,000 rows
        print(f"flag every case: {t['flag_all'][10]:.3f} per case")
        print(f"true probabilities: {t['bayes'][10]:.3f} per case")
        for name, v in t["models"].items():
            r = v[10]                   # a miss costs 10 false alarms
            print(f"{name:22s} at 0.5 {r['half']:.3f}, at 1/11 "
                  f"{r['derived']:.3f}, tuned {r['tuned']:.3f}")
        """)
    out["why"] = s.run("""
        b = t["models"]["boosted trees"][10]
        c = t["models"]["recalibrated boosted trees"][10]
        print(f"probabilities below 0.01: raw {b['below_001']:.1%}, "
              f"recalibrated {c['below_001']:.1%}")
        print(f"flagged at 1/11: raw {b['flag_rate']:.1%}, "
              f"recalibrated {c['flag_rate']:.1%}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    t = res["t"]
    M = t["models"]
    bt, rc, lg = M["boosted trees"], M["recalibrated boosted trees"], M["logistic"]
    fa, by = t["flag_all"], t["bayes"]
    half10 = [v[10]["half"] for v in M.values()]
    worse20 = [k for k, v in M.items() if v[20]["tuned"] > fa[20]]

    # The spine, asserted rather than trusted.
    assert all(v[1]["half"] == v[1]["derived"] for v in M.values())
    assert 1.4 < min(half10) and max(half10) < 2.01 and min(half10) > 2.8 * fa[10]
    assert abs(rc[10]["derived"] - rc[10]["best"]) < 0.003
    assert bt[10]["derived"] > 1.5 * fa[10] and bt[20]["derived"] > 1.85 * fa[20]
    assert bt[10]["below_001"] > 0.1 and rc[10]["below_001"] < 0.02
    assert bt[10]["tuned"] < fa[10] and bt[10]["tuned_t"] < 0.005
    assert len(worse20) == 3
    assert 0.11 < 1 - by[10] / fa[10] < 0.13
    assert all(v[10]["best"] < fa[10] for v in M.values())
    assert lg[10]["derived"] - lg[10]["best"] < 0.015

    post = Post(
        title=(f"{SERIES_TAG} 46: The Right Threshold Is a Cost Ratio, If the "
               f"Probabilities Are Calibrated"),
        slug="ml-46-threshold",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=46,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("0.5 is the right threshold only when both mistakes cost "
                  "the same. The threshold that matches the costs needs "
                  "calibrated probabilities, and even the perfect model may "
                  "save little."),
        summary=(
            f"Episode 4's simulated task with a miss costing r times a false "
            f"alarm. At r = 10, flagging at 0.5 cost {min(half10):.2f} to "
            f"{max(half10):.2f} per case, against {fa[10]:.3f} for flagging "
            f"every case. The cost-derived threshold 1/(1 + r) came within "
            f"0.003 of the best threshold in hindsight for recalibrated "
            f"boosted trees ({rc[10]['derived']:.3f}); for the raw, "
            f"overconfident trees it cost {bt[10]['derived']:.3f}, worse than "
            f"flagging everyone. A threshold tuned on 1,000 fresh rows "
            f"rescued them ({bt[10]['tuned']:.3f}), and at r = 20 was worse "
            f"than flagging everyone for three of five models. The true "
            f"probabilities beat flagging everyone by "
            f"{1 - by[10] / fa[10]:.0%} at r = 10."),
        tags=["machine-learning", "data-science", "statistics",
              "decision-theory", "calibration", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: episode 4's two-class Gaussian task in 10 "
            "dimensions with 50,000 test rows and their exact posterior "
            "probabilities; models trained on 2,000 rows (seeds 10 to 14), "
            "thresholds tuned on 1,000 fresh rows (seeds 100 to 104).",
            "Machinery: `standarderror/ml/decide.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's LogisticRegression, "
            "GaussianNB, RandomForestClassifier (300 trees, leaves of 5), "
            "HistGradientBoostingClassifier (300 rounds) and "
            "CalibratedClassifierCV (isotonic, five folds).",
            "Where this stops: Elkan, \"The foundations of cost-sensitive "
            "learning\", *IJCAI* (2001); Provost and Fawcett, \"Robust "
            "classification for imprecise environments\", *Machine Learning* "
            "(2001); episodes 5 and 34 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("seeds as listed; about a minute of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "From a probability to an action",
        f"""Forty-five episodes measured models and the numbers they produce. Arc X follows those numbers one step further, to the decisions made with them: which cases to flag, which model to deploy, which lever to pull, whom to treat, and what happens when the decisions feed the next round of data. The first step is the simplest and the most often skipped. A classifier outputs a probability; something turns it into a yes or no. The default, built into most classifiers' `predict` method, is the sentence under test: *a classifier predicts the positive class when its probability is above 0.5*.

Episode 5 found that rebalancing classes is a roundabout way of moving that threshold. This episode asks where it should be. The answer is standard decision theory, set out for classifiers by Elkan in 2001: if a missed positive costs r times as much as a false alarm, flag a case when its probability of being positive exceeds 1/(1 + r). At r = 1 that is 0.5. At r = 10, a miss being ten times as costly as a false alarm, it is 1/11.

Episode 4's simulated task makes the best possible decision known, because the true probability of every test case is known. Models are trained on 2,000 rows, and each threshold is scored by its cost per case on 50,000 test rows:

{snip['half'].markdown()}

At r = 10, every model flagging at 0.5 costs between **{min(half10):.2f}** and {max(half10):.2f} per case. The trivial policy of flagging every case costs {fa[10]:.3f}, the share of negatives. A model used with the default threshold is roughly three to four times worse ({min(half10) / fa[10]:.1f} to {max(half10) / fa[10]:.1f} times) than not using a model at all.""",
        figures=[figs["f0"]])

    post.add(
        "The derived threshold needs honest probabilities",
        f"""Moving the threshold to 1/11 fixes most of it. Logistic regression costs {lg[10]['derived']:.3f}, within {lg[10]['derived'] - lg[10]['best']:.3f} of the best threshold in hindsight; boosted trees recalibrated by isotonic regression cost **{rc[10]['derived']:.3f}**, within {rc[10]['derived'] - rc[10]['best']:.3f}.

The raw boosted trees are the exception: at 1/11 they cost **{bt[10]['derived']:.3f}**, worse than flagging everyone. The formula assumes that a probability of 0.05 means a 5% chance, and episode 14 found that boosting overfits its probabilities long before its accuracy:

{snip['why'].markdown()}

The raw model sends {bt[10]['below_001']:.0%} of cases below 0.01, against {rc[10]['below_001']:.1%} once recalibrated, so at 1/11 it clears cases that are not nearly as safe as it claims and pays ten for every positive among them. The threshold derived from the costs is exactly as good as the calibration of the probabilities it is applied to, which is the practical reason episode 34's recalibration matters even when accuracy does not move.""",
        figures=[figs["f1"]])

    post.add(
        "Tuning the threshold instead",
        f"""The alternative to trusting the probabilities is to choose the threshold by its cost on held-out data. Tuned on 1,000 fresh rows, the boosted trees' threshold settled at {bt[10]['tuned_t']:.4f}, far below 1/11, and their cost fell to {bt[10]['tuned']:.3f}: tuning absorbs miscalibration, because it only needs the order of the scores.

It has its own cost. The cheapest threshold on 1,000 rows plausibly depends on the handful of rare, expensive misses among them, which makes it noisy at high r. At r = 20 the tuned threshold was worse than flagging everyone for three of the five models ({'; '.join(worse20)}). Tuning and calibration need data for the same reason; tuning needs it at the operating point, where the expensive errors are rare.""")

    post.add(
        "What the derived threshold asks for",
        f"""It is worth seeing what following the costs looks like in practice, because it can be uncomfortable. At r = 10 the derived threshold flagged {lg[10]['flag_rate']:.1%} of cases for logistic regression and {rc[10]['flag_rate']:.1%} for the recalibrated trees. A process built around 0.5, which flags about half, would be asked to flag nearly everything, and its cost per case would fall from {lg[10]['half']:.2f} to {lg[10]['derived']:.3f} as a result.

The raw boosted trees, flagging only {bt[10]['flag_rate']:.1%} at the same threshold, look more selective, and that apparent selectivity is a symptom of their overconfidence: many of the cases they clear are not as safe as their probabilities claim, which is why that threshold cost them {bt[10]['derived']:.3f}. A flag rate is therefore a useful check in its own right. If the cost-derived threshold flags far fewer cases than the costs imply it should, the probabilities are probably too extreme, and the remedy is recalibration before thresholding, not a different threshold.

At r = 1 all of this collapses back to the familiar case: the derived threshold is 0.5, and the five models' costs at it ranged from {min(v[1]['half'] for v in M.values()):.3f} to {max(v[1]['half'] for v in M.values()):.3f}, against {by[1]:.3f} for the true probabilities.""")

    post.add(
        "How much a model is worth here",
        f"""The flat line in the first figure is a useful discipline. At r = 10 the best a model could possibly do on this task, using the true probabilities with the derived threshold, is {by[10]:.3f} per case. Flagging everyone costs {fa[10]:.3f}. The perfect model saves **{1 - by[10] / fa[10]:.0%}**, and the trained models less than half of that.

That is not a defect of the models. When misses are expensive, the decision is to flag almost everything, and a model can only help by finding cases safe enough to clear; on a task where the classes overlap heavily, there are few. Provost and Fawcett's ROC convex hull makes the same point geometrically: for each cost ratio there is an operating point, and the value of a classifier is measured there, against the trivial policies at the corners, not by its accuracy at 0.5.""")

    post.add(
        "Where this breaks",
        """**Known, fixed costs.** Real costs are uncertain, different for different cases and sometimes asymmetric in ways a single ratio cannot express. The arithmetic still applies per case if the costs are known per case.

**Balanced classes.** The task has half positives. With rare positives the derived threshold stays at 1/(1 + r), but flagging nobody becomes the cheap trivial policy to beat.

**One task with heavy overlap.** On a task where the classes separate well, models would likely save much more than 11%; that was not measured here.

**Five training sets.** Costs at r = 20 are dominated by rare misses and are noisy; the counts of models beaten by flagging everyone could change with other training sets.""")

    post.add(
        "What to keep",
        f"""1. With a miss ten times as costly as a false alarm, the 0.5 threshold cost {min(half10):.2f} to {max(half10):.2f} per case, roughly three to four times flagging every case ({fa[10]:.3f}).

2. The threshold 1/(1 + r) came within 0.003 of the best for recalibrated boosted trees and cost the raw, overconfident ones {bt[10]['derived']:.3f}.

3. Tuning the threshold on held-out rows works without calibration but needs data at the operating point; at r = 20 it lost to flagging everyone for three of five models.

4. Choose the threshold from the costs, use it only on calibrated probabilities, and compare the result with the trivial policies before claiming the model is worth deploying.""")

    post.add(
        "Exercise",
        """For a classifier you use, write down what a missed positive and a false alarm each cost, even roughly, and compute 1/(1 + r). Check a calibration curve near that threshold, not overall.

Then compute the cost per case of your current threshold, of 1/(1 + r), and of the two trivial policies, flagging everything and flagging nothing. If the model does not clearly beat the better trivial policy, its value lies elsewhere than in this decision.""")

    post.add(
        "Next",
        """A threshold is chosen for one model. Choosing between models is usually done earlier, by a single number, and the most common one is the AUC. Episode 47 asks whether the model with the best AUC makes the best decisions.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
