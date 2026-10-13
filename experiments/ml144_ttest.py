"""ML 44: Cross-Validation Folds Are Not Independent Samples, and the t-Test Assumes They Are.

The forty-fourth episode of *Machine Learning, Taught Through What
Breaks*. "A paired t-test on the cross-validation fold scores tells you
whether one model is better than another." Measured with two algorithms
that have exactly the same expected accuracy by construction (a decision
tree, and the same tree on rotated inputs of a rotation-invariant task),
on 400 data sets of 300 rows:

* The paired t-test on ten folds declared a difference at the 5% level in
  14% of data sets; on ten repetitions of ten folds, treated as 100
  independent scores, in 58%.
* Nadeau and Bengio's corrected resampled t-test rejected 5.0% of the time,
  Dietterich's 5x2cv test 4.2% and McNemar's test on one split 4.8%.
* On a given data set either tree could truly be better, by more than two
  points in 22% of them; the ten-fold test's rejections pointed the right
  way two times in three.

Run: `standarderror run ml144_ttest --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import combine
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

TESTS = ("10-fold t", "10x10 t", "10x10 corrected", "5x2cv", "McNemar")
LABELS = {"10-fold t": "paired t, 10 folds",
          "10x10 t": "paired t, 10 x 10 folds",
          "10x10 corrected": "corrected t, 10 x 10",
          "5x2cv": "5x2cv t",
          "McNemar": "McNemar, one split"}


def compute() -> dict:
    return {"null": combine.ttest_study(), "alt": combine.ttest_study(depth_b=2)}


# ------------------------------------------------------------------ figures

def _bars(ax, m, null, alt):
    x = np.arange(len(TESTS))
    w = 0.38
    ax.bar(x - w / 2, [null["reject"][t] for t in TESTS], width=w,
           yerr=[2 * null["reject_se"][t] for t in TESTS], color=m.series[1],
           label="equal algorithms (should be 0.05)")
    ax.bar(x + w / 2, [alt["reject"][t] for t in TESTS], width=w,
           yerr=[2 * alt["reject_se"][t] for t in TESTS], color=m.series[2],
           label=f"one truly better by {alt['true_mean']:.3f}")
    ax.axhline(0.05, color=m.ink, ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t] for t in TESTS], fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")


def figures(res: dict) -> dict:
    out: dict = {}
    null, alt = res["null"], res["alt"]

    out["f0"] = charts.diagram(
        lambda ax, m: _bars(ax, m, null, alt),
        title="How often five tests find a difference",
        subtitle=("Rejection rate at the 5% level over 400 data sets of 300 "
                  "rows; bars: two standard errors. Left bars: two "
                  "algorithms with exactly equal expected accuracy. Right "
                  "bars: one algorithm genuinely worse."),
        xlabel="",
        ylabel="share of data sets with p < 0.05",
        figsize=(7.8, 4.6),
        source="Simulated; standarderror/ml/combine.py.",
        alt=("Paired bars for five tests. Under equal algorithms the 10-fold "
             "t-test rejects about 14% and the 10 x 10 t-test about 58%; the "
             "three other tests sit near the dotted 5% line. With a real "
             "difference, the 10 x 10 corrected test rejects most often "
             "among the honest tests."),
        caption=(f"With nothing to find, the paired t-test on ten folds "
                 f"found something in **{null['reject']['10-fold t']:.1%}** "
                 f"of data sets; the corrected test in "
                 f"{null['reject']['10x10 corrected']:.1%}."),
        path=str(IMG / f"ml144-f0-tests.{EXT}"))[0]

    body = [[LABELS[t], f"{null['reject'][t]:.3f}", f"± {null['reject_se'][t]:.3f}",
             f"{alt['reject'][t]:.3f}", f"± {alt['reject_se'][t]:.3f}"]
            for t in TESTS]
    out["f1"] = charts.table_image(
        body,
        header=["test", "equal: rejected", "s.e.", "unequal: rejected", "s.e."],
        title="Size and power of five model-comparison tests",
        subtitle=(f"400 data sets each. Equal: two depth-4 trees, one on "
                  f"rotated inputs (true difference {null['true_mean']:+.4f} "
                  f"on average, sd {null['true_sd']:.3f} across data sets). "
                  f"Unequal: the rotated tree limited to depth 2 (true "
                  f"difference {alt['true_mean']:+.3f})."),
        source="Simulated; standarderror/ml/combine.py.",
        alt=("A five-row table of rejection rates with standard errors, "
             "under equal and unequal algorithms."),
        caption=("A test's size is its rejection rate when there is nothing "
                 "to find; its power, when there is. The naive tests buy "
                 "power with size they do not have."),
        align="lrrrr",
        path=str(IMG / f"ml144-f1-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    null, alt = res["null"], res["alt"]

    def p(keys, src):
        def f(panel, m):
            panel.bar(range(len(keys)), [src["reject"][k] for k in keys],
                      color=[m.series[1], m.series[2], m.series[3]][:len(keys)])
            panel.axhline(0.05, color=m.ink, lw=1)
            panel.set_ylim(0, 0.65)
        return f

    return charts.lecture_hero(
        series=SERIES_TAG, episode=44,
        headline="Folds are not samples",
        panels=[(p(("10-fold t", "10x10 t"), null),
                 f"{null['reject']['10x10 t']:.0%}", "false alarms, 10 x 10 t"),
                (p(("10x10 corrected", "5x2cv", "McNemar"), null),
                 f"{null['reject']['10x10 corrected']:.1%}", "corrected t, size"),
                (p(("10x10 corrected", "5x2cv", "McNemar"), alt),
                 f"{alt['reject']['10x10 corrected']:.0%}", "corrected t, power")],
        note=("Fold scores share training rows, so they agree with each "
              "other more than independent samples would. A test that "
              "ignores that sees certainty in an echo."),
        alt=("Three hand-drawn frames of bars against a low line: one tall "
             "and one very tall bar; three bars at the line; three bars "
             "above it."),
        path=str(IMG / f"ml144-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["null"] = s.run("""
        from standarderror.ml import combine

        r = combine.ttest_study()      # 400 data sets, equal algorithms
        print(f"true difference: mean {r['true_mean']:+.4f}, "
              f"sd across data sets {r['true_sd']:.4f}")
        for test, rate in r["reject"].items():
            print(f"{test:16s} rejects {rate:.3f} "
                  f"(+/- {r['reject_se'][test]:.3f})")
        """)
    out["alt"] = s.run("""
        a = combine.ttest_study(depth_b=2)   # B genuinely worse
        print(f"true difference: mean {a['true_mean']:+.4f}")
        for test, rate in a["reject"].items():
            print(f"{test:16s} rejects {rate:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    null, alt = res["null"], res["alt"]
    rn, ra = null["reject"], alt["reject"]

    # The spine, asserted rather than trusted.
    assert abs(null["true_mean"]) < 0.002 and null["true_sd"] > 0.015
    assert null["true_abs_above_002"] > 0.2
    assert 0.11 < rn["10-fold t"] < 0.18 and rn["10x10 t"] > 0.5
    for t in ("10x10 corrected", "5x2cv", "McNemar"):
        assert abs(rn[t] - 0.05) < 2.5 * null["reject_se"][t]
    assert 0.6 < null["naive_sign_right"] < 0.75
    assert null["naive_reject_big_true"] > null["naive_reject_small_true"]
    assert alt["true_mean"] > 0.03
    assert ra["10x10 corrected"] > max(ra["5x2cv"], ra["McNemar"]) + 0.1
    assert ra["10x10 t"] > ra["10-fold t"] > ra["10x10 corrected"]
    assert ra["5x2cv"] < 0.1 and ra["McNemar"] < 0.1

    post = Post(
        title=(f"{SERIES_TAG} 44: Cross-Validation Folds Are Not Independent "
               f"Samples, and the t-Test Assumes They Are"),
        slug="ml-44-model-tests",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=44,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Two algorithms built to be exactly equal, compared 400 "
                  "times. The most common test found a difference one time "
                  "in seven, the most thorough-looking one more often than "
                  "not, and three others were honest."),
        summary=(
            f"Two algorithms with exactly equal expected accuracy, by "
            f"symmetry, compared on 400 data sets of 300 rows. The paired "
            f"t-test on ten cross-validation folds declared a difference at "
            f"the 5% level in {rn['10-fold t']:.1%} of data sets; on ten "
            f"repetitions of ten folds, in {rn['10x10 t']:.1%}. Nadeau and "
            f"Bengio's corrected resampled t-test rejected "
            f"{rn['10x10 corrected']:.1%} of the time, Dietterich's 5x2cv "
            f"test {rn['5x2cv']:.1%} and McNemar's test {rn['McNemar']:.1%}. "
            f"With one algorithm genuinely worse by {alt['true_mean']:.3f}, "
            f"the corrected test detected it in {ra['10x10 corrected']:.0%} "
            f"of data sets and the other two honest tests in under 10%."),
        tags=["machine-learning", "data-science", "statistics",
              "hypothesis-testing", "cross-validation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: two classes centred at the origin in four "
            "dimensions with standard deviations 1 and 1.6, so every "
            "rotation leaves the task unchanged; 400 data sets of 300 rows "
            "(seeds 0 to 399), 50,000 fresh rows for the true differences "
            "(seed 999).",
            "Machinery: `standarderror/ml/combine.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's DecisionTreeClassifier "
            "(depth 4; depth 2 for the unequal case), a fixed random "
            "rotation from scipy's special_ortho_group (seed 0), KFold; "
            "scipy's t and exact binomial distributions.",
            "Where this stops: Dietterich, \"Approximate statistical tests "
            "for comparing supervised classification learning algorithms\", "
            "*Neural Computation* (1998); Nadeau and Bengio, \"Inference for "
            "the generalization error\", *Machine Learning* (2003); "
            "Bouckaert and Frank, \"Evaluating the replicability of "
            "significance tests for comparing learning algorithms\", *PAKDD* "
            "(2004); Bengio and Grandvalet, \"No unbiased estimator of the "
            "variance of K-fold cross-validation\", *JMLR* (2004).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("data seeds 0 to 399; fold shuffles derived from "
                            "the data seed; about eight minutes of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Is the difference significant?",
        f"""Episodes 41 to 43 combined models. The other half of Arc IX compares them, and the comparison usually ends with a statistical test. Episode 1 compared two classifiers on one test set with McNemar's test. When data are scarce the comparison is made with cross-validation instead: both models are scored on the same folds, the ten differences are fed to a paired t-test, and a small p-value is reported as evidence that one model is better. The sentence under test is the justification: *a paired t-test on the cross-validation fold scores tells you whether one model is better than another*.

Testing a test needs a case where the answer is known to be "no difference", and two real algorithms almost never have exactly equal accuracy. Symmetry provides one. The task has two classes centred at the same point, one more spread out than the other, so its distribution looks the same from every direction. Algorithm A is a depth-4 decision tree. Algorithm B is the same tree fitted after a fixed random rotation of the inputs. Trees split along the axes, so on any one data set the two trees are different models and either can be better; over data sets, since a rotation changes nothing about the task, their expected accuracies are exactly equal at every training size.

Four hundred data sets of 300 rows each, and five tests of "A and B are equally accurate" on each:

{snip['null'].markdown()}

A test at the 5% level should reject about 5% of the time when there is nothing to find. The paired t-test on ten folds rejected **{rn['10-fold t']:.1%}** of the time, nearly three times its nominal level. Repeating the ten-fold split ten times with different shuffles and testing all 100 differences, which looks more thorough, rejected **{rn['10x10 t']:.1%}** of the time.""",
        figures=[figs["f0"]])

    post.add(
        "What the t-test assumes",
        f"""The paired t-test assumes the ten differences are independent draws. They are not. Any two training sets in ten-fold cross-validation share eight-ninths of their rows, so the ten models are close relatives and their differences move together. The spread of the ten differences understates how much the mean difference would vary from one data set to another, and the t statistic, which divides by that spread, comes out too large. Bengio and Grandvalet proved in 2004 that no unbiased estimator of that variance can be built from a single run of K-fold cross-validation.

Repeating the split makes it worse rather than better. The hundred differences from ten repetitions all come from the same 300 rows: the repetitions average away the noise of the fold assignment, but they cannot average away the luck of the data set, and the test treats a hundred echoes of one data set as a hundred independent observations, with 99 degrees of freedom.

There is a second reason, and it concerns the question rather than the arithmetic. On a particular data set the two trees are genuinely different; measured on 50,000 fresh rows, the tree trained on that data set's rows differed from the rotated tree by more than two points in {null['true_abs_above_002']:.0%} of the data sets. Cross-validation on a single data set can at best answer "which is better *here*?". The ten-fold test's rejections pointed the same way as that data set's true difference only {null['naive_sign_right']:.0%} of the time, so they did not answer that question well either. Dietterich made the distinction in 1998: comparing algorithms means asking about their accuracy over the data sets one might have drawn, and only the variation between data sets can answer it.""")

    post.add(
        "Three honest tests",
        f"""The other three tests are built around that variation, and all three held their level: **{rn['10x10 corrected']:.1%}** for Nadeau and Bengio's corrected resampled t-test, {rn['5x2cv']:.1%} for Dietterich's 5x2cv test and {rn['McNemar']:.1%} for McNemar's test on a single two-thirds/one-third split, each within two standard errors of 5%.

The corrected test keeps the hundred differences but inflates their variance by a term for the overlap between training and test sets, the ratio of test-fold size to training size; Bouckaert and Frank recommended it with ten repetitions of ten folds for exactly this reason. The 5x2cv test uses five repetitions of two-fold cross-validation, whose training sets do not overlap within a repetition, and a conservative statistic. McNemar's test sidesteps fold dependence by using one split and the disagreements between the two models on its test rows.

Holding the level is half of a test's job. The other half is power: detecting a difference when there is one. With B limited to depth 2, genuinely worse by {alt['true_mean']:.3f} on average:

{snip['alt'].markdown()}

The corrected test detected the difference in **{ra['10x10 corrected']:.0%}** of data sets; 5x2cv and McNemar in {ra['5x2cv']:.0%} and {ra['McNemar']:.0%}. Both spend their honesty by using less of the data per decision: 5x2cv trains on half the rows, McNemar tests on a third. The naive tests reject more often here too, but that is not power; it is the same overconfidence that made them reject when there was nothing to find.""",
        figures=[figs["f1"]])

    post.add(
        "Where this breaks",
        """**One task, small data.** Three hundred rows and shallow trees, where data-set-to-data-set variation is large compared with the differences between algorithms. With many more rows the variation between data sets shrinks; whether the naive tests' excess shrinks with it was not measured here.

**Unstable learners.** Decision trees change a lot when the data change, which makes the dependence between folds matter. Stable learners such as regularised linear models change less from fold to fold, and the naive tests' excess may be smaller for them; that was not measured here.

**One comparison.** Comparing many models at once, or one model across many data sets, raises further problems: multiple testing, and, for many data sets, the rank-based tests Demšar recommended in 2006.

**Significance is not size.** None of these tests says whether a difference is large enough to matter, which episode 1's error bars address more directly.""")

    post.add(
        "What to keep",
        f"""1. With two algorithms of exactly equal expected accuracy, the paired t-test on ten folds declared a difference in {rn['10-fold t']:.1%} of data sets, and on ten repetitions of ten folds in {rn['10x10 t']:.1%}.

2. Fold scores share training rows, and repeated folds share the data set; neither carries the variation between data sets that the comparison is about.

3. The corrected resampled t-test, 5x2cv and McNemar's test held their 5% level; of the three, the corrected test detected a genuine difference of {alt['true_mean']:.3f} most often ({ra['10x10 corrected']:.0%}).

4. Compare algorithms with the corrected resampled t-test on repeated cross-validation, or McNemar's test on a large held-out set, and report the size of the difference alongside its p-value.""")

    post.add(
        "Exercise",
        """Take two models you compared with cross-validation. Recompute the comparison three ways: the paired t-test on the fold differences, the same with ten repetitions of the folds, and the corrected resampled t-test, which multiplies the variance of the differences by (1/J + n_test/n_train) for J differences.

Then shuffle the labels and rerun all three, twenty times. Count how often each finds a "significant" difference between two models that both know nothing.""")

    post.add(
        "Next",
        """Every test here used each data set once. Episode 45, the last of Arc IX, measures what happens to a test set that is consulted again and again, by a competitor, a careful-looking analysis and an ordinary analyst, without ever being trained on.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
