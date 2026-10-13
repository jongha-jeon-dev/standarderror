"""ML 40: Target Encoding Fitted on the Training Rows Still Leaks, Into Training.

The fortieth episode of *Machine Learning, Taught Through What Breaks*,
closing Arc VIII. "Fit target encoding on the training rows only, inside
the folds, and it cannot leak." Measured on a simulated task with five
numeric features and one category of 10 to 4,000 levels, 4,000 training
rows and 20,000 fresh test rows:

* With a category that carries no signal and 1,000 levels (four rows per
  level), encoding each level by its mean label on the training rows took
  boosted trees from AUC 0.763 without the column to 0.640, and log loss
  from 0.594 to 1.306; logistic regression fell from 0.791 to 0.690.
* On the rows it was fitted on, the encoded noise column alone ranked the
  labels with AUC 0.774; on fresh rows, 0.498.
* Shrinking towards the overall mean helped logistic regression and not
  the trees at one row per level. Cross-fitting, each training row encoded
  from the other folds, matched dropping the column to within 0.002 of AUC on noise, and
  kept the signal when the category had one.

Run: `standarderror run ml140_target_encoding --publish`
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

ENCODINGS = ("dropped", "mean per level", "shrunk, m = 20", "cross-fitted")


def compute() -> dict:
    t = rows.target_encoding()
    return {"t": t, "k": {(r["effect"], r["levels"], r["encoding"], r["model"]): r
                          for r in t},
            "col": rows.encoded_column_auc()}


# ------------------------------------------------------------------ figures

def _lines(ax, m, k, effect):
    levels = (10, 100, 1000, 4000)
    for j, enc in enumerate(ENCODINGS):
        col = m.ink if enc == "dropped" else m.series[j]
        for model, ls in (("boosted trees", "-"), ("logistic", "--")):
            ax.plot(levels, [k[(effect, K, enc, model)]["auc"] for K in levels],
                    color=col, ls=ls, marker="o", ms=4,
                    label=f"{enc}, {model}")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)


def figures(res: dict) -> dict:
    out: dict = {}
    k = res["k"]

    out["f0"] = charts.diagram(
        lambda ax, m: _lines(ax, m, k, 0.0),
        title="A category that means nothing, encoded four ways",
        subtitle=("Test AUC on 20,000 fresh rows as the category's levels go "
                  "from 10 to 4,000 over 4,000 training rows; the category "
                  "has no effect. Solid: boosted trees; dashed: logistic "
                  "regression. Means over 5 data sets."),
        xlabel="levels of the category (log scale)",
        ylabel="test AUC",
        figsize=(7.6, 4.8),
        source="Simulated; standarderror/ml/rows.py.",
        alt=("Lines for the dropped and cross-fitted encodings stay flat; "
             "lines for the per-level mean fall steeply beyond 100 levels; "
             "the shrunk encoding falls steeply for the trees and far less for "
             "logistic regression."),
        caption=(f"With 1,000 levels the per-level mean took boosted trees "
                 f"from {k[(0.0, 1000, 'dropped', 'boosted trees')]['auc']:.3f} "
                 f"to **{k[(0.0, 1000, 'mean per level', 'boosted trees')]['auc']:.3f}**; "
                 f"cross-fitting left them at "
                 f"{k[(0.0, 1000, 'cross-fitted', 'boosted trees')]['auc']:.3f}."),
        path=str(IMG / f"ml140-f0-noise.{EXT}"))[0]

    out["f1"] = charts.diagram(
        lambda ax, m: _lines(ax, m, k, 1.0),
        title="A category that matters, encoded the same four ways",
        subtitle=("The same, with level effects drawn from a normal "
                  "distribution with sd 1 on the log-odds."),
        xlabel="levels of the category (log scale)",
        ylabel="test AUC",
        figsize=(7.6, 4.8),
        source="Simulated; standarderror/ml/rows.py.",
        alt=("All encodings beat dropping the column at 10 and 100 levels; "
             "at 1,000 and 4,000 levels the per-level mean falls below "
             "dropping it, and cross-fitting stays above."),
        caption=(f"At 1,000 levels cross-fitting kept part of the signal "
                 f"(logistic AUC {k[(1.0, 1000, 'cross-fitted', 'logistic')]['auc']:.3f} "
                 f"against {k[(1.0, 1000, 'dropped', 'logistic')]['auc']:.3f} "
                 f"dropped); the per-level mean turned it into a loss "
                 f"({k[(1.0, 1000, 'mean per level', 'logistic')]['auc']:.3f})."),
        path=str(IMG / f"ml140-f1-signal.{EXT}"))[0]

    body = []
    for effect in (0.0, 1.0):
        for K in (100, 1000, 4000):
            for enc in ENCODINGS:
                lg, tr = k[(effect, K, enc, "logistic")], k[(effect, K, enc, "boosted trees")]
                body.append(["none" if effect == 0 else "sd 1",
                             f"{K:,}", f"{4000 / K:g}", enc,
                             f"{lg['auc']:.3f}", f"{tr['auc']:.3f}",
                             f"{tr['nll']:.3f}"])
    out["f2"] = charts.table_image(
        body,
        header=["category effect", "levels", "rows per level", "encoding",
                "logistic AUC", "trees AUC", "trees log loss"],
        title="Every encoding fitted on the training rows only",
        subtitle=("Test scores on 20,000 fresh rows; 4,000 training rows; "
                  "means over 5 data sets."),
        source="Simulated; standarderror/ml/rows.py.",
        alt=("A twenty-four-row table. Without a category effect, the "
             "per-level mean is far below 'dropped' from 1,000 levels on; "
             "cross-fitted is within 0.002 of 'dropped'. With an effect, "
             "cross-fitted is best or close to best at every size."),
        caption=("None of these is a leak in the usual sense: the test rows "
                 "never touched the encoding. The damage is done to the "
                 "training rows."),
        align="lrrlrrr",
        path=str(IMG / f"ml140-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    k, col = res["k"], res["col"]

    def p1(panel, m):
        panel.bar([0, 1], [col["mean_train"], col["mean_test"]],
                  color=[m.series[1], m.ink])
        panel.set_ylim(0.4, 0.8)

    def p2(panel, m):
        L = (10, 100, 1000, 4000)
        panel.plot([k[(0.0, K, "mean per level", "boosted trees")]["auc"] for K in L],
                   color=m.series[1], lw=2)
        panel.plot([k[(0.0, K, "cross-fitted", "boosted trees")]["auc"] for K in L],
                   color=m.series[3], lw=2)

    def p3(panel, m):
        panel.bar([0, 1, 2], [k[(1.0, 1000, e, "logistic")]["auc"] for e in
                              ("mean per level", "dropped", "cross-fitted")],
                  color=[m.series[1], m.ink, m.series[3]])
        panel.set_ylim(0.7, 0.8)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=40,
        headline="Encoded by its own label",
        panels=[(p1, f"{col['mean_train']:.3f}", "noise column, train AUC"),
                (p2, f"{k[(0.0, 4000, 'mean per level', 'boosted trees')]['auc']:.3f}",
                 "trees, 4,000 levels"),
                (p3, f"{k[(1.0, 1000, 'cross-fitted', 'logistic')]['auc']:.3f}",
                 "cross-fitted, signal")],
        note=("A level's mean label, computed from a training row's own "
              "label, tells the model that row's answer. Inside the folds "
              "keeps it from the test set, not from the training set."),
        alt=("Three hand-drawn frames: a tall bar and a short one; a "
             "falling line under a flat one; three bars, the last highest."),
        path=str(IMG / f"ml140-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["noise"] = s.run("""
        from standarderror.ml import rows

        t = rows.target_encoding()     # 4,000 training rows, 5 data sets
        for r in t:
            if r["effect"] == 0 and r["model"] == "boosted trees":
                print(f"{r['levels']:5d} levels, {r['encoding']:15s} "
                      f"AUC {r['auc']:.3f}, log loss {r['nll']:.3f}")
        """)
    out["col"] = s.run("""
        c = rows.encoded_column_auc()  # the noise column alone, 1,000 levels
        print(f"mean per level: AUC {c['mean_train']:.3f} on its own "
              f"training rows, {c['mean_test']:.3f} on fresh rows")
        print(f"cross-fitted:   AUC {c['cross_train']:.3f} on its own "
              f"training rows, {c['cross_test']:.3f} on fresh rows")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    k, col = res["k"], res["col"]

    def a(e, K, enc, m="boosted trees"):
        return k[(e, K, enc, m)]

    T, L = "boosted trees", "logistic"
    drop_t, drop_l = a(0.0, 1000, "dropped"), a(0.0, 1000, "dropped", L)
    mean_t, mean_l = a(0.0, 1000, "mean per level"), a(0.0, 1000, "mean per level", L)

    # The spine, asserted rather than trusted.
    for K in (10, 100, 1000, 4000):
        for m in (T, L):
            assert abs(a(0.0, K, "cross-fitted", m)["auc"]
                       - a(0.0, K, "dropped", m)["auc"]) < 0.002
    assert drop_t["auc"] - mean_t["auc"] > 0.12 and mean_t["nll"] > 2 * drop_t["nll"]
    assert drop_l["auc"] - mean_l["auc"] > 0.09
    assert a(0.0, 4000, "mean per level")["auc"] < 0.59
    assert a(0.0, 4000, "mean per level")["nll"] > 4 * a(0.0, 4000, "dropped")["nll"]
    assert abs(a(0.0, 10, "mean per level")["auc"] - a(0.0, 10, "dropped")["auc"]) < 0.002
    assert a(0.0, 100, "dropped")["auc"] - a(0.0, 100, "mean per level")["auc"] > 0.005
    assert a(0.0, 4000, "shrunk, m = 20", L)["auc"] > 0.76
    assert a(0.0, 4000, "shrunk, m = 20")["auc"] < 0.6
    assert col["mean_train"] > 0.77 and abs(col["mean_test"] - 0.5) < 0.005
    assert col["cross_train"] < 0.5 and abs(col["cross_test"] - 0.5) < 0.005
    for K in (10, 100):
        for enc in ENCODINGS[1:]:
            assert a(1.0, K, enc, L)["auc"] > a(1.0, K, "dropped", L)["auc"] + 0.04
    for K in (1000, 4000):
        for m in (T, L):
            assert a(1.0, K, "cross-fitted", m)["auc"] > a(1.0, K, "dropped", m)["auc"]
            assert a(1.0, K, "mean per level", m)["auc"] < a(1.0, K, "dropped", m)["auc"]
    assert a(1.0, 1000, "cross-fitted", L)["auc"] - a(1.0, 1000, "dropped", L)["auc"] > 0.015

    post = Post(
        title=(f"{SERIES_TAG} 40: Target Encoding Fitted on the Training Rows "
               f"Still Leaks, Into Training"),
        slug="ml-40-target-encoding",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=40,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Fitting an encoding only on training rows protects the "
                  "test set. It does not protect the training rows from "
                  "their own labels, and with many rare levels that is "
                  "enough to wreck the model."),
        summary=(
            f"A simulated task with five numeric features and a category of "
            f"10 to 4,000 levels, 4,000 training rows, every encoding fitted "
            f"on the training rows only. With a category carrying no signal "
            f"and 1,000 levels, encoding each level by its mean label took "
            f"boosted trees from test AUC {drop_t['auc']:.3f} without the "
            f"column to {mean_t['auc']:.3f}, and log loss from "
            f"{drop_t['nll']:.3f} to {mean_t['nll']:.3f}; on its own training "
            f"rows the encoded noise column alone had AUC "
            f"{col['mean_train']:.3f}. Cross-fitting matched dropping the "
            f"column to within 0.002 of AUC on noise, and with a real effect at 1,000 levels "
            f"reached logistic AUC "
            f"{a(1.0, 1000, 'cross-fitted', L)['auc']:.3f} against "
            f"{a(1.0, 1000, 'dropped', L)['auc']:.3f} dropped."),
        tags=["machine-learning", "data-science", "statistics",
              "feature-engineering", "target-encoding", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: five standard-normal features with log-odds x_1 + "
            "0.7 x_2 − 0.5 x_3, plus a category of K equally likely levels "
            "whose effects are 0 or normal with sd 1. 4,000 training rows "
            "(seeds 0 to 4), 20,000 test rows sharing the level effects "
            "(seeds 100 to 104).",
            "Machinery: `standarderror/ml/rows.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's LogisticRegression, "
            "HistGradientBoostingClassifier and TargetEncoder (five shuffled "
            "folds, its default shrinkage); the per-level mean and its "
            "shrunk version (20 pseudo-rows at the overall mean) written "
            "out.",
            "Where this stops: Micci-Barreca, \"A preprocessing scheme for "
            "high-cardinality categorical attributes\", *SIGKDD Explorations* "
            "(2001); Prokhorenkova et al., \"CatBoost: unbiased boosting "
            "with categorical features\", *NeurIPS* (2018); Pargent et al., "
            "\"Regularized target encoding outperforms traditional methods "
            "in supervised machine learning with high cardinality "
            "features\", *Computational Statistics* (2022); episode 3 of "
            "this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("seeds as listed; TargetEncoder folds with "
                            "random_state 0; about a minute of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A feature made from the label",
        f"""Arc VIII has followed the process behind the rows: time (episode 36), people (37), gaps (38) and decisions (39). It ends with a feature built from the outcome itself. A category with many levels, such as a postcode, a merchant, a product or a user, is awkward for most models: one-hot encoding makes thousands of columns, and trees split on arbitrary level codes. Target encoding replaces each level with the mean outcome of the training rows at that level, one numeric column that carries what the category says about the target.

Episode 3 measured the obvious mistake, computing those means on all rows including the test fold, and found that a 200-level noise column fitted that way inflated the score. The standard advice that follows is the sentence under test: *fit the target encoding on the training rows only, inside the folds, and it cannot leak*.

The test here never touches the encoding: it is fitted on 4,000 training rows and scored on 20,000 fresh rows. Five numeric features carry the real signal, and a category of 10 to 4,000 levels is added. In the first experiment the category carries nothing.

{snip['noise'].markdown()}

With 10 levels, 400 rows each, encoding the category changes nothing. With 1,000 levels, four training rows per level, boosted trees given the encoded column score a test AUC of **{mean_t['auc']:.3f}**, against {drop_t['auc']:.3f} without it, and their log loss doubles, to {mean_t['nll']:.3f}. With 4,000 levels they reach {a(0.0, 4000, 'mean per level')['auc']:.3f}. Logistic regression falls too, from {drop_l['auc']:.3f} to {mean_l['auc']:.3f} at 1,000 levels. Nothing leaked into the test set, and the models are genuinely worse.""",
        figures=[figs["f0"]])

    post.add(
        "The leak is into the training rows",
        f"""A level with four training rows has a mean label of 0, 0.25, 0.5, 0.75 or 1, and each of those four rows contributed its own label to it. For a training row, the encoded value is partly its own answer. The encoded column alone shows how much:

{snip['col'].markdown()}

On the rows it was computed from, the encoded noise column ranks the labels with an AUC of **{col['mean_train']:.3f}**, better than many real features. On fresh rows it is {col['mean_test']:.3f}, a coin. The model is trained on rows where this column is a strong predictor, learns to rely on it at the expense of the five real features, and then meets test rows where it is noise. That is overfitting, done by the preprocessing instead of the model, and cross-validation inside the folds measures it correctly, as a bad model, without preventing it.

It is the same mechanism as episode 3's leak moved one step: there, the encoding saw the test rows' labels; here, it sees the training rows' own labels, and the model trained on them cannot tell an honest feature from a recording of the answer.""",
        figures=[figs["f2"]])

    post.add(
        "Two repairs",
        f"""The first repair is shrinkage: pull each level's mean towards the overall mean in proportion to how few rows it has, here as if each level had 20 extra rows at the overall rate. It helps logistic regression a great deal, to {a(0.0, 4000, 'shrunk, m = 20', L)['auc']:.3f} at 4,000 levels, probably because shrinkage pulls the extreme values of rare levels, 0 and 1 for a level seen once, towards the middle, and the default L2 penalty then keeps the model from scaling the column back up. It does not help the trees ({a(0.0, 4000, 'shrunk, m = 20')['auc']:.3f}): within levels of the same size, shrinkage keeps the values ordered by the training rows' own labels, and trees split on order, not on scale.

The second is cross-fitting, which is what scikit-learn's TargetEncoder and CatBoost's ordered statistics both do in different ways. Each training row is encoded by means computed from the other folds, so its own label never enters its own value; test rows get means from all training rows. On the noise category it matched dropping the column at every number of levels, to within 0.002 of AUC for both models, and the encoded column's training AUC fell to {col['cross_train']:.3f}, slightly below a coin, since a leave-out mean is pulled away from the row it leaves out.""")

    post.add(
        "When the category carries signal",
        f"""Dropping the column is safe only when it is noise, which is not why anyone encodes it. In the second experiment each level shifts the log-odds by an amount drawn with standard deviation 1. With 10 or 100 levels, every encoding works and gains four to six points of AUC for logistic regression over dropping the column. With 1,000 levels the per-level mean turns the signal into a loss, {a(1.0, 1000, 'mean per level', L)['auc']:.3f} against {a(1.0, 1000, 'dropped', L)['auc']:.3f} for dropping it, while cross-fitting keeps a gain, **{a(1.0, 1000, 'cross-fitted', L)['auc']:.3f}**. At 4,000 levels the gain is small ({a(1.0, 4000, 'cross-fitted', L)['auc']:.3f} against {a(1.0, 4000, 'dropped', L)['auc']:.3f}), because one row per level says little about any level; but cross-fitting is the only encoding that is not worse than dropping the column for either model.

Pargent and colleagues compared encodings on real high-cardinality data sets and reached a compatible conclusion: regularised target encoding, fitted with an internal cross-validation, was the most reliable choice across the learners they tried.""",
        figures=[figs["f1"]])

    post.add(
        "Where this breaks",
        """**Equally likely levels.** Real categories have a few common levels and a long tail of rare ones. The common levels encode safely and the tail does the damage, so the harm is concentrated where there is least data, which also makes it harder to see in an overall score.

**One sample size.** 4,000 training rows. What matters is rows per level, so the same damage should appear at 40,000 levels with 40,000 rows.

**Default settings.** TargetEncoder's shrinkage and the boosted trees' regularisation were left at their defaults. Stronger regularisation in the trees reduces their reliance on any single column, including this one.

**Binary outcome.** The same mechanism applies to regression targets and to multi-class targets, which encode one column per class.""")

    post.add(
        "What to keep",
        f"""1. Fitting the encoding on the training rows only protects the test set and nothing else. With 1,000 noise levels, boosted trees fell from AUC {drop_t['auc']:.3f} to {mean_t['auc']:.3f}.

2. The training rows leak into their own encoding: the noise column alone ranked its training labels at AUC {col['mean_train']:.3f}, and fresh labels at {col['mean_test']:.3f}.

3. Shrinkage largely rescued logistic regression and not trees at one row per level; cross-fitting matched dropping the column on noise and kept the signal when there was one.

4. Encode a high-cardinality category with out-of-fold means on the training rows, and check how many rows the rarest levels have before trusting the column.""")

    post.add(
        "Exercise",
        f"""Take a model that uses a target-encoded category. Compute the encoded column's AUC (or correlation with the target) on the training rows and on held-out rows.

If the training value is much higher, the column is partly a recording of the training labels; here the gap was {col['mean_train'] - col['mean_test']:.2f} of AUC for a column with no signal at all. Replace the encoding with a cross-fitted one, refit, and compare held-out scores; then try dropping the column, which is the baseline any encoding has to beat.""")

    post.add(
        "Next",
        """Arc VIII ends here, at forty episodes. Where the rows came from decided what the scores meant: shuffled folds fine until something unseen changed slowly, people counted as rows, gaps that recorded decisions, outcomes seen only for the approved, and a feature built from the labels it was meant to predict. Arc IX turns from single models to combinations and comparisons, starting with the most common combination of all: an average.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
