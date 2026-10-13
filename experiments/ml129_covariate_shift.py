"""ML 29: Covariate Shift Charges for Misspecification.

The twenty-ninth episode of *Machine Learning, Taught Through What Breaks*.
"Covariate shift costs accuracy." Measured on a one-dimensional simulation
where only the inputs move and P(y | x) stays fixed, so the best possible
accuracy hardly changes:

* A flexible model (logistic regression on cubic splines) stays within about
  a point of the Bayes rule until the test inputs sit two standard deviations
  away, where it loses under three.
* A linear logistic model, misspecified, falls from 0.816 to 0.536.
* Importance weighting moves the linear model to the best linear model for
  the test inputs, which is what it promises. That model has lower log loss
  but not always higher accuracy, and at a shift of 2 the weights leave about
  6% of the training rows' worth of information.

Run: `standarderror run ml129_covariate_shift --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import nets
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    rows = nets.covariate_shift()
    return {"rows": rows, "by": {r["shift"]: r for r in rows}}


def _sig(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    rows, by = res["rows"], res["by"]
    r2 = by[2.0]

    def setup(ax, m):
        x = np.linspace(-3.5, 5.0, 400)
        ax.plot(x, _sig(3 * np.sin(1.5 * x)), color=m.ink, lw=1.6,
                label="P(y = 1 | x), the same in both")
        pdf = lambda v, mu: np.exp(-(v - mu) ** 2 / 2) / 2.5066  # noqa: E731
        ax.fill_between(x, 0, pdf(x, 0.0), color=m.series[0], alpha=0.25,
                        label="training inputs, N(0, 1)")
        ax.fill_between(x, 0, pdf(x, 2.0), color=m.series[1], alpha=0.25,
                        label="test inputs, N(2, 1)")
        b, a = r2["coef_linear"]
        ax.plot(x, _sig(b * x + a), color=m.series[0], lw=1.4, ls="--",
                label="linear fit, unweighted")
        b, a = r2["coef_linear_iw"]
        ax.plot(x, _sig(b * x + a), color=m.series[1], lw=1.4, ls="--",
                label="linear fit, importance-weighted")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    out["f0"] = charts.diagram(
        setup,
        title="The rule stays, the inputs move",
        subtitle=("Training inputs from N(0, 1), test inputs from N(2, 1); the "
                  "probability of the label given x is the same wavy curve in "
                  "both. Two linear logistic fits, averaged over 20 training "
                  "sets of 500."),
        xlabel="x",
        ylabel="probability / density",
        figsize=(7.6, 4.8),
        source="Simulated; standarderror/ml/nets.py.",
        alt=("A sine-like probability curve over x; a blue density centred at "
             "0 and an orange one at 2; a rising dashed line fitted to the "
             "training region and a falling one fitted to the test region."),
        caption=(f"Unweighted, the line fits the rising stretch where the "
                 f"training data are and keeps rising where the test data "
                 f"are; on N(2, 1) it is right on {r2['acc_linear']:.1%} of "
                 f"rows. Weighted, it fits the falling stretch "
                 f"({r2['acc_linear_iw']:.1%})."),
        path=str(IMG / f"ml129-f0-setup.{EXT}"))[0]

    shifts = [r["shift"] for r in rows]

    def lines(ax, m):
        ax.plot(shifts, [r["bayes"] for r in rows], color=m.ink, lw=1.4,
                ls=":", label="Bayes rule")
        for key, lab, c, ls in (
                ("flex", "flexible", 2, "-"), ("flex_iw", "flexible, weighted",
                                               2, "--"),
                ("linear", "linear", 0, "-"), ("linear_iw", "linear, weighted",
                                               1, "--")):
            ax.plot(shifts, [r[f"acc_{key}"] for r in rows], marker="o",
                    ms=4, color=m.series[c], ls=ls, label=lab)
        ax.set_ylim(0.5, 0.86)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    out["f1"] = charts.diagram(
        lines,
        title="The shift costs the misspecified model",
        subtitle=("Test accuracy against the shift in the mean of the inputs, "
                  "average of 20 training sets of 500; 20,000 test rows per "
                  "shift."),
        xlabel="shift of the test inputs (training standard deviations)",
        ylabel="test accuracy",
        figsize=(7.4, 4.6),
        source="Simulated; standarderror/ml/nets.py.",
        alt=("The Bayes line is flat near 0.84; the flexible model tracks it, "
             "dipping slightly at the largest shift; the linear model falls "
             "steeply; the weighted linear model dips then recovers."),
        caption=(f"The best possible accuracy barely moves "
                 f"({min(r['bayes'] for r in rows):.3f} to "
                 f"{max(r['bayes'] for r in rows):.3f}). The flexible model "
                 f"loses {r2['bayes'] - r2['acc_flex']:.3f} at a shift of 2; "
                 f"the linear one loses "
                 f"{r2['bayes'] - r2['acc_linear']:.3f}."),
        path=str(IMG / f"ml129-f1-lines.{EXT}"))[0]

    trs = []
    for r in rows:
        trs.append([f"{r['shift']:.1f}", f"{r['bayes']:.3f}",
                    f"{r['acc_flex']:.3f}", f"{r['acc_linear']:.3f}",
                    f"{r['acc_linear_iw']:.3f}", f"{r['oracle']['acc']:.3f}",
                    f"{r['nll_linear']:.3f}", f"{r['nll_linear_iw']:.3f}",
                    f"{r['oracle']['nll']:.3f}", f"{r['ess']:.0%}"])
    out["f2"] = charts.table_image(
        trs,
        header=["shift", "Bayes", "flexible", "linear", "linear, weighted",
                "best linear", "log loss linear", "log loss weighted",
                "log loss best", "effective n"],
        title="What importance weighting buys a linear model",
        subtitle=("'Best linear' is a linear logistic model fitted to 200,000 "
                  "rows from the test distribution: the target the weights "
                  "aim at. Effective n is (Σw)² / Σw² as a share of 500."),
        source="Simulated; standarderror/ml/nets.py.",
        alt=("A five-row table: weighted log loss is below unweighted at "
             "every shift; weighted accuracy is below unweighted at shifts "
             "0.5 and 1 and above it at 1.5 and 2; effective n falls to a "
             "few percent."),
        caption=("The weighted model tracks the best linear model in both "
                 "accuracy and log loss. The best linear model at a shift of 1 "
                 "is a worse classifier than the unweighted one."),
        bold_cells={(2, 4), (2, 5), (4, 9)},
        align="rrrrrrrrrr",
        path=str(IMG / f"ml129-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    rows, r2 = res["rows"], res["by"][2.0]

    def flex(panel, m):
        panel.plot([r["acc_flex"] for r in rows], color=m.series[2], lw=2)
        panel.set_ylim(0.5, 0.86)

    def lin(panel, m):
        panel.plot([r["acc_linear"] for r in rows], color=m.series[0], lw=2)
        panel.set_ylim(0.5, 0.86)

    def ess(panel, m):
        panel.bar(range(len(rows)), [r["ess"] for r in rows],
                  color=m.series[1])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=29,
        headline="The shift bills the wrong model",
        panels=[(flex, f"{r2['acc_flex']:.3f}", "flexible, shift 2"),
                (lin, f"{r2['acc_linear']:.3f}", "linear, shift 2"),
                (ess, f"{r2['ess']:.0%}", "effective n, weighted")],
        note=("Move the inputs two standard deviations and keep the rule "
              "that labels them: a flexible model barely notices, a linear "
              "one falls to near chance. Importance weights refit the line "
              "for the new inputs from a small share of the data."),
        alt=("Three hand-drawn frames: a nearly flat line; a steeply falling "
             "line; five bars shrinking to almost nothing."),
        path=str(IMG / f"ml129-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["shift"] = s.run("""
        from standarderror.ml import nets

        rows = nets.covariate_shift()        # 500 training rows, 20 repetitions
        for r in rows:
            print(f"shift {r['shift']:.1f}  Bayes {r['bayes']:.3f}  "
                  f"flexible {r['acc_flex']:.3f}  linear {r['acc_linear']:.3f}")
        """)
    out["iw"] = s.run("""
        for r in rows:
            print(f"shift {r['shift']:.1f}  accuracy {r['acc_linear']:.3f} -> "
                  f"{r['acc_linear_iw']:.3f}  log loss {r['nll_linear']:.3f} -> "
                  f"{r['nll_linear_iw']:.3f}  effective n {r['ess']:.0%}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    rows, by = res["rows"], res["by"]
    r0, r1, r15, r2 = by[0.0], by[1.0], by[1.5], by[2.0]
    bayes_lo = min(r["bayes"] for r in rows)
    bayes_hi = max(r["bayes"] for r in rows)
    flex_gap = max(r["bayes"] - r["acc_flex"] for r in rows if r["shift"] <= 1.5)

    # The spine, asserted rather than trusted.
    assert bayes_hi - bayes_lo < 0.01
    assert flex_gap < 0.02
    assert 0.015 < r2["bayes"] - r2["acc_flex"] < 0.04
    assert r0["acc_linear"] > 0.8 and r2["acc_linear"] < 0.6
    assert all(a["acc_linear"] > b["acc_linear"]
               for a, b in zip(rows, rows[1:]))
    assert all(r["nll_linear_iw"] < r["nll_linear"] for r in rows[1:])
    assert r1["acc_linear_iw"] < r1["acc_linear"] - 0.05
    assert by[0.5]["acc_linear_iw"] < by[0.5]["acc_linear"]
    assert r15["acc_linear_iw"] > r15["acc_linear"]
    assert abs(r1["oracle"]["coef"]) < 0.2
    assert r2["acc_linear_iw"] > r2["acc_linear"] + 0.2
    assert r1["oracle"]["acc"] < r1["acc_linear"]
    assert all(abs(r["acc_linear_iw"] - r["oracle"]["acc"]) < 0.05 and
               r["nll_linear_iw"] - r["oracle"]["nll"] < 0.04 for r in rows)
    assert r2["ess"] < 0.08 and r1["ess"] < 0.4
    assert r2["acc_flex_iw"] > r2["acc_flex"]
    assert r2["sd_flex_iw"] < r2["sd_flex"] / 2
    assert r2["ece_linear"] > 0.4 and r2["ece_flex"] < 0.1

    post = Post(
        title=f"{SERIES_TAG} 29: Covariate Shift Charges for Misspecification",
        slug="ml-29-covariate-shift",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=29,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("When only the inputs move, the best possible accuracy need "
                  "not change. Whether a model's accuracy does depends on how "
                  "wrong the model was all along."),
        summary=(
            f"In a simulation where the test inputs move up to two standard "
            f"deviations while P(y | x) stays fixed, the Bayes accuracy stays "
            f"between {bayes_lo:.3f} and {bayes_hi:.3f}. A flexible spline "
            f"logistic model stays within {flex_gap:.3f} of it up to a shift "
            f"of 1.5 and loses {r2['bayes'] - r2['acc_flex']:.3f} at 2. A "
            f"linear logistic model falls from {r0['acc_linear']:.3f} to "
            f"{r2['acc_linear']:.3f}. Importance weighting recovers the best "
            f"linear model for the test inputs: lower log loss at every "
            f"shift, but lower accuracy at a shift of 1 "
            f"({r1['acc_linear_iw']:.3f} against {r1['acc_linear']:.3f}), and "
            f"at a shift of 2 the weights keep an effective "
            f"{r2['ess']:.0%} of the sample."),
        tags=["machine-learning", "data-science", "statistics",
              "distribution-shift", "importance-weighting", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: x ~ N(0, 1) in training and N(shift, 1) in test, "
            "P(y = 1 | x) = sigmoid(3 sin(1.5 x)) in both; 500 training "
            "rows, 20 repetitions, 20,000 test rows per shift.",
            "Machinery: `standarderror/ml/nets.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's LogisticRegression, alone and "
            "on SplineTransformer features (8 knots, cubic, linear "
            "extrapolation, C = 10).",
            "Where this stops: Shimodaira, \"Improving predictive inference "
            "under covariate shift by weighting the log-likelihood "
            "function\", *Journal of Statistical Planning and Inference* "
            "(2000); Sugiyama, Krauledat and Müller, \"Covariate shift "
            "adaptation by importance weighted cross validation\", *JMLR* "
            "(2007); Kong, \"A note on importance sampling using "
            "standardized weights\" (1992), for the effective sample size.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("training sets from seeds 0 to 19, test rows "
                            "from seed 999, the best-linear reference from "
                            "seed 7"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Only the inputs move",
        f"""Episode 28 ended with a model that met inputs it had not been shown. That is the normal condition of a deployed model: the population it scores drifts away from the one it was trained on. The textbook name for the simplest version is *covariate shift*. The distribution of the inputs changes; the rule that turns inputs into outcomes, P(y | x), does not. The sentence under test is the one usually attached to it: *covariate shift costs accuracy*.

It sounds obvious, but the setting makes it odd. If P(y | x) is unchanged, the best possible classifier is unchanged too: the Bayes rule labels each x the same way in training and in test. A model that had learned the rule would carry on being right. So whatever the shift costs is not the shift's doing alone.

The simulation keeps everything visible. The input x is one number, drawn from N(0, 1) in training and from N(s, 1) in test, for shifts s from 0 to 2. The label is 1 with probability sigmoid(3 sin(1.5x)), the same wavy curve everywhere. Two models are fitted to 500 training rows, twenty times over: a linear logistic regression, which cannot follow the wave, and a logistic regression on cubic spline features, which can.

{snip['shift'].markdown()}

The Bayes accuracy stays between **{bayes_lo:.3f}** and **{bayes_hi:.3f}** at every shift: the task does not get harder. The flexible model stays within {flex_gap:.3f} of it up to a shift of 1.5. The linear model loses more at every step and ends at **{r2['acc_linear']:.3f}**, not far above a coin.""",
        figures=[figs["f0"]])

    post.add(
        "Why the line breaks and the spline does not",
        f"""The figure above shows what the linear model did. Most training inputs sit between −1 and 1, where the wave climbs from its trough to its peak, so the best line through them rises. In the test region around x = 2 the wave has turned over and is falling. The line, still rising, predicts class 1 where class 0 is now the more likely label.

Nothing about the line was wrong *for the training distribution*. It was the best linear approximation to the wave over N(0, 1), and the best linear approximation depends on where the inputs are. When the inputs move, the best line moves, and the fitted one does not. That is the whole mechanism: covariate shift hurts a model by exposing the places where it was approximating, and a misspecified model approximates everywhere.

The spline model follows the wave where it has data, so its errors in the test region are mostly the irreducible ones. It does lose something at the largest shift, {r2['bayes'] - r2['acc_flex']:.3f} of accuracy at s = 2, because the right tail of the test distribution reaches x = 4 and beyond, where the training data have almost no points and the spline extrapolates linearly. Shift far enough and every model is extrapolating. The difference is how early the cost starts. For the line it starts at the first step.

Calibration says the same thing more loudly. The linear model's expected calibration error at s = 2 is {r2['ece_linear']:.2f}: its probabilities are confidently pointed the wrong way. The spline's is {r2['ece_flex']:.2f}.""",
        figures=[figs["f1"]])

    post.add(
        "Importance weighting does what it says",
        f"""The standard remedy is to weight each training row by how much more likely its x is under the test distribution than under the training one, w(x) = p_test(x) / p_train(x). Here the ratio is known exactly; in practice it has to be estimated, which is a second source of error this simulation leaves out. Weighted, the training loss becomes an unbiased estimate of the test loss, and the fitted model aims at the best model *for the test inputs*.

{snip['iw'].markdown()}

The weighted log loss is lower than the unweighted at every shift, and the table below shows it tracks a linear model fitted to 200,000 rows drawn from the test distribution itself, the target the weights aim at. At s = 2 the weights turn the line around, and accuracy goes from {r2['acc_linear']:.3f} to {r2['acc_linear_iw']:.3f}.

At s = 1 accuracy goes the other way, from {r1['acc_linear']:.3f} to {r1['acc_linear_iw']:.3f}. This is not the weights failing. The best linear model for N(1, 1) inputs, by log loss, is a nearly flat line (it scores {r1['oracle']['acc']:.3f} itself), because over that range the wave rises and falls about equally. Log loss rewards a line that hedges near 0.5; accuracy rewards one that commits. Importance weighting fixes the loss it is given, and when the model is misspecified, the loss-optimal and the accuracy-optimal models are different models.""",
        figures=[figs["f2"]])

    post.add(
        "The price of the weights",
        f"""Weights are not free. A weighted sample of 500 rows carries less information than 500 rows, and Kong's effective sample size, (Σw)² / Σw², measures how much less. At a shift of 1 it is {r1['ess']:.0%} of the sample. At 1.5 it is {r15['ess']:.0%}, and at 2 it is **{r2['ess']:.0%}**: a few rows in the training set's right tail carry most of the weight, and the fit is effectively made from about {r2['ess'] * 500:.0f} of them.

That is why the weighted linear model's accuracy varies more from one training set to the next: its standard deviation over the twenty repetitions is {r2['sd_linear_iw']:.3f} at s = 2 against {r2['sd_linear']:.3f} unweighted. The weights buy a smaller bias with a larger variance, and the trade gets worse the further the shift goes. When the test inputs lie where the training data have none, there is nothing to reweight.

For the flexible model the weights make little difference in the region it can already follow, and a small one at s = 2: {r2['acc_flex']:.3f} unweighted, {r2['acc_flex_iw']:.3f} weighted, with the spread across repetitions falling from {r2['sd_flex']:.3f} to {r2['sd_flex_iw']:.3f}. That is the extrapolating tail again: weighting puts more of the fit's attention on the few rows near the test region, which plausibly steadies what the splines do just beyond them. The mechanism was not isolated here.""")

    post.add(
        "Where this breaks",
        """**The weights were exact.** Both densities were known. In practice the ratio is estimated, with a classifier trained to tell training from test inputs or a density-ratio method, and estimated weights add their own variance, worst exactly where the true weights are largest.

**Only the inputs moved.** If P(y | x) changes as well, the setting is concept shift and none of this applies: the Bayes rule itself moves, and no reweighting of old labels can learn the new one. In real drift the two usually come together, and telling them apart needs labelled data from the new population.

**One dimension.** In one dimension the flexible model sees the whole range it will be tested on, up to the tail. In many dimensions a shift can move the inputs to combinations no training row came near, and every model extrapolates there. The lesson that misspecification is what the shift charges for still holds; a flexible model just runs out of support sooner.

**A periodic rule.** The wave was chosen so a line would fail. Where the true rule is close to linear over both ranges, a linear model loses little to the shift and weighting has nothing to correct.""")

    post.add(
        "What to keep",
        f"""1. Under covariate shift the best possible accuracy need not change. Here the Bayes accuracy stayed between {bayes_lo:.3f} and {bayes_hi:.3f} while the inputs moved two standard deviations.

2. What the shift costs is the model's misspecification, exposed. A flexible model stayed within {flex_gap:.3f} of Bayes up to a shift of 1.5; a linear one fell from {r0['acc_linear']:.3f} to {r2['acc_linear']:.3f}.

3. Importance weighting refits the model for the test inputs. It lowered log loss at every shift and tracked the best linear model for the test distribution, which at a shift of 1 was a worse classifier ({r1['acc_linear_iw']:.3f} against {r1['acc_linear']:.3f}).

4. The weights shrink the sample: at a shift of 2 the effective sample was {r2['ess']:.0%} of 500 rows.

5. Before reweighting, check whether a more flexible model makes the shift harmless, and check the effective sample size of the weights before trusting what they produce.""")

    post.add(
        "Exercise",
        """Take a model in production and split its recent scoring data by some input that has drifted: newer customers, a new region, a new device. Train a classifier to tell training inputs from recent ones, and turn its probabilities into weights.

Compute the effective sample size of those weights. If it is a few percent, the recent population sits where your training data are thin, and no weighting will rescue the model there. Then refit with a more flexible model and see whether its accuracy on the recent rows (once labels arrive) falls less than the current one's did.""")

    post.add(
        "Next",
        """One episode remains. Every network in this arc was small enough to train many times, which is what let each claim be measured across runs. The last episode climbs a ladder of model sizes and tests the sentence that justifies training very large ones: that loss falls as a power law of model size, so small models predict what large ones will do.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
