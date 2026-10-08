"""ML 2: Cross-Validation Estimates the Error of a Model You Did Not Train.

The second episode of *Machine Learning, Taught Through What Breaks*. K-fold
CV is usually described as estimating how well the model you fit will do.
On simulated linear regression, where that error is known exactly for every
dataset, the correlation between the CV estimate and it is about zero over
2,000 datasets; on the digits, against a large fixed test set, it is
negative. CV estimates the *average* error of the procedure over training
sets of that size, and the naive interval around it does not cover even that
at the nominal rate.

Measured:

* Linear model, n = 100, p = 20: corr(CV, err_xy) = -0.01; CV's own spread
  is more than twice the spread of what it is trying to track.
* Naive 90% CV intervals cover err_xy 80% of the time and the average error
  83%.
* Digits, 300 training images, 797-image fixed test set: corr = -0.21.
* A held-out set answers the conditional question -- positive correlation,
  near-nominal coverage -- at the price of a noisier estimate and a model
  trained on less data.

Run: `standarderror run ml102_cv_target --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import evaluation as ev
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 8)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def _corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - b) ** 2)))


def compute() -> dict:
    lin = ev.linear_cv_study()
    dig = ev.digits_cv_study()
    rows = [
        {"estimate": "10-fold CV", "target": "error of the fitted model",
         "corr": _corr(lin["cv"], lin["err_xy"]),
         "rmse": _rmse(lin["cv"], lin["err_xy"]),
         "cover": ev.coverage(lin["cv"], lin["cv_se"], lin["err_xy"])},
        {"estimate": "10-fold CV", "target": "average error, any 100 rows",
         "corr": float("nan"),
         "rmse": _rmse(lin["cv"], lin["err"]),
         "cover": ev.coverage(lin["cv"], lin["cv_se"], lin["err"])},
        {"estimate": "100 fresh held-out rows",
         "target": "error of the fitted model",
         "corr": _corr(lin["holdout"], lin["err_xy"]),
         "rmse": _rmse(lin["holdout"], lin["err_xy"]),
         "cover": ev.coverage(lin["holdout"], lin["holdout_se"],
                              lin["err_xy"])},
        {"estimate": "20 of the 100, held out",
         "target": "error of the model fit on 80",
         "corr": _corr(lin["split_holdout"], lin["split_err_xy"]),
         "rmse": _rmse(lin["split_holdout"], lin["split_err_xy"]),
         "cover": ev.coverage(lin["split_holdout"], lin["split_se"],
                              lin["split_err_xy"])},
    ]
    return {"lin": lin, "dig": dig, "rows": rows}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    lin, dig = res["lin"], res["dig"]

    def scatter(ax, m):
        ax.scatter(lin["cv"], lin["err_xy"], s=7, alpha=0.35,
                   color=m.series[0], linewidths=0,
                   label=f"{lin['reps']:,} simulated datasets")
        lo = min(lin["cv"].min(), lin["err_xy"].min())
        hi = max(lin["cv"].max(), lin["err_xy"].max())
        ax.plot([lo, hi], [lo, hi], lw=1.4, ls="--", color=m.ink_secondary,
                label="estimate = truth")
        ax.axhline(lin["err"], lw=1.4, color=m.series[2],
                   label=f"average error over datasets, {lin['err']:.3f}")
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    c = res["rows"][0]["corr"]
    out["f0"] = charts.diagram(
        scatter,
        title="The CV estimate does not track the error it accompanies",
        subtitle=(f"Least squares, n = {lin['n']}, p = {lin['p']}. Each dot "
                  f"is one dataset: its 10-fold CV estimate against the "
                  f"exact test error of the model fitted to it."),
        xlabel="10-fold CV estimate of squared error",
        ylabel="true squared error of the fitted model",
        source="Simulated; standarderror/ml/evaluation.py.",
        alt=("A wide horizontal cloud of dots with no slope, far from the "
             "dashed diagonal, centred on a horizontal line."),
        caption=(f"Correlation **{c:+.2f}**. The CV estimates spread "
                 f"{lin['cv'].std():.2f} wide around a truth that varies by "
                 f"only {lin['err_xy'].std():.2f}: they are tracking the "
                 f"horizontal line, the average, not the dot's own height."),
        path=str(IMG / f"ml102-f0-linear.{EXT}"))[0]

    def digits_plot(ax, m):
        ax.scatter(dig["cv"], dig["true"], s=14, alpha=0.6,
                   color=m.series[1], linewidths=0)
        k, b = np.polyfit(dig["cv"], dig["true"], 1)
        xs = np.linspace(dig["cv"].min(), dig["cv"].max(), 10)
        ax.plot(xs, k * xs + b, lw=2.0, color=m.series[0],
                label=f"least-squares line, correlation {dig['corr']:+.2f}")
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    out["f1"] = charts.diagram(
        digits_plot,
        title="On real digits the relationship runs backwards",
        subtitle=(f"{dig['reps']} training sets of {dig['train']} digits. "
                  f"x: their 10-fold CV accuracy. y: the fitted model's "
                  f"accuracy on {dig['test']} fixed test digits it never "
                  f"saw."),
        xlabel="10-fold CV accuracy",
        ylabel="accuracy on the fixed test set",
        source="Measured; UCI digits; standarderror/ml/evaluation.py.",
        alt=("A cloud of dots with a gently falling fitted line."),
        caption=(f"Correlation **{dig['corr']:+.2f}**. The training sets "
                 f"that cross-validate best produce models that do slightly "
                 f"*worse* on new digits."),
        path=str(IMG / f"ml102-f1-digits.{EXT}"))[0]

    rows_t = [[r["estimate"], r["target"],
               "-" if np.isnan(r["corr"]) else f"{r['corr']:+.2f}",
               f"{r['rmse']:.3f}", f"{r['cover']:.0%}"]
              for r in res["rows"]]
    out["f2"] = charts.table_image(
        rows_t,
        header=["estimate", "of what", "correlation", "RMSE",
                "naive 90% interval covers"],
        title="Two estimates, two targets",
        subtitle=(f"The same {lin['reps']:,} simulated datasets of "
                  f"{lin['n']} rows. The interval is estimate +- 1.645 "
                  f"standard errors of the per-row losses."),
        source="Simulated; standarderror/ml/evaluation.py.",
        alt=("A four-row table comparing CV and held-out estimates by "
             "correlation with their target, error, and interval "
             "coverage."),
        caption=("CV is closer to the average than to the fitted model. A "
                 "held-out set tracks the fitted model, noisily, and its "
                 "interval covers closer to the nominal rate."),
        align="llrrr",
        path=str(IMG / f"ml102-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    lin, dig = res["lin"], res["dig"]

    def cloud(panel, m):
        panel.scatter(lin["cv"][:300], lin["err_xy"][:300], s=6,
                      color=m.series[0], linewidths=0)

    def back(panel, m):
        panel.scatter(dig["cv"], dig["true"], s=8, color=m.series[1],
                      linewidths=0)

    def cover(panel, m):
        vals = [r["cover"] for r in res["rows"]]
        panel.bar(range(len(vals)), vals, color=m.series[2])
        panel.plot([-0.5, len(vals) - 0.5], [0.9, 0.9], lw=2.0,
                   color=m.ink)
        panel.set_ylim(0.6, 1.0)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=2,
        headline="Cross-validation estimates an average, not your model",
        panels=[(cloud, f"r = {res['rows'][0]['corr']:+.2f}",
                 "CV vs your model's error"),
                (back, f"r = {dig['corr']:+.2f}", "on real digits"),
                (cover, f"{res['rows'][0]['cover']:.0%}",
                 "a 90% CV interval covers")],
        note=("Measured where the true error of each fitted model is known, "
              "the cross-validation estimate is uncorrelated with it. CV "
              "estimates how well models trained on data like yours do on "
              "average; a held-out set is what speaks to the one you "
              "actually fit."),
        alt=("Three hand-drawn frames: a flat cloud of dots; a cloud "
             "drifting down; and four bars below a line."),
        path=str(IMG / f"ml102-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["linear"] = s.run("""
        import numpy as np
        from standarderror.ml import evaluation as ev

        # 2,000 datasets: 100 rows, 20 features, least squares.
        lin = ev.linear_cv_study(n=100, p=20, reps=2000)
        cv, err_xy = lin["cv"], lin["err_xy"]
        print(f"mean CV estimate      {cv.mean():.3f}   SD {cv.std():.3f}")
        print(f"mean true error       {err_xy.mean():.3f}   SD {err_xy.std():.3f}")
        print(f"correlation           {np.corrcoef(cv, err_xy)[0, 1]:+.3f}")
        """)
    out["cover"] = s.run("""
        print(f"90% CV interval covers the fitted model's error "
              f"{ev.coverage(cv, lin['cv_se'], err_xy):.1%}")
        print(f"90% CV interval covers the average error        "
              f"{ev.coverage(cv, lin['cv_se'], lin['err']):.1%}")
        """)
    out["digits"] = s.run("""
        dig = ev.digits_cv_study(train=300, test=797, reps=200)
        print(f"CV accuracy    mean {dig['cv'].mean():.4f}  SD {dig['cv'].std():.4f}")
        print(f"test accuracy  mean {dig['true'].mean():.4f}  SD {dig['true'].std():.4f}")
        print(f"correlation    {dig['corr']:+.3f}")
        """)
    out["holdout"] = s.run("""
        ho, ho_err = lin["split_holdout"], lin["split_err_xy"]
        print(f"fit on 80, score on 20: correlation "
              f"{np.corrcoef(ho, ho_err)[0, 1]:+.3f}   SD {ho.std():.3f}   "
              f"mean true error {ho_err.mean():.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    lin, dig, rows = res["lin"], res["dig"], res["rows"]
    cvx, cva, fresh, split = rows
    ratio = lin["cv"].std() / lin["err_xy"].std()

    # The spine, asserted rather than trusted.
    assert abs(cvx["corr"]) < 0.06
    assert ratio > 2
    assert cvx["cover"] < 0.85 and cva["cover"] < 0.87
    assert dig["corr"] < -0.1
    assert fresh["corr"] > 0.35 and split["corr"] > 0.15
    assert fresh["cover"] > cvx["cover"]
    assert cva["rmse"] < cvx["rmse"]
    assert lin["split_err_xy"].mean() > lin["err"]

    post = Post(
        title=(f"{SERIES_TAG} 2: Cross-Validation Estimates the Error of a "
               f"Model You Did Not Train"),
        slug="ml-2-cv-target",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=2,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("K-fold CV is described as estimating how well your model "
                  "will do. Where that is known exactly, the estimate is "
                  "uncorrelated with it. What CV estimates is how models "
                  "like yours do on average."),
        summary=(
            f"On {lin['reps']:,} simulated least-squares datasets of "
            f"{lin['n']} rows, where the exact test error of each fitted "
            f"model is known, the correlation between the 10-fold CV "
            f"estimate and that error is {cvx['corr']:+.2f}. The CV "
            f"estimates spread {ratio:.1f} times as wide as the errors they "
            f"accompany, and their naive 90% intervals cover the fitted "
            f"model's error {cvx['cover']:.0%} of the time - and the average "
            f"error, CV's real target, {cva['cover']:.0%}. On the UCI digits, "
            f"against a fixed 797-image test set, the correlation is "
            f"{dig['corr']:+.2f}. A held-out set does track the fitted "
            f"model ({fresh['corr']:+.2f} with 100 fresh rows), at the price "
            f"of noise and of training on less data."),
        tags=["machine-learning", "data-science", "statistics",
              "cross-validation", "model-evaluation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: standard-normal features, coefficients 0.5, noise SD "
            "1, so each fitted model's expected squared error is exactly "
            "1 + |b - beta|^2.",
            "UCI Optical Recognition of Handwritten Digits, as bundled with "
            "scikit-learn (`load_digits`), CC BY 4.0.",
            "Machinery: `standarderror/ml/evaluation.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Bates, Hastie and Tibshirani, "
            "\"Cross-validation: what does it estimate and how well does it "
            "do it?\", *JASA* (2023), which proves the linear-model result "
            "and proposes nested CV for honest intervals; Hastie, "
            "Tibshirani and Friedman, *The Elements of Statistical Learning* "
            "(2009), section 7.12, which first reported the weak "
            "correlation.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("one seeded generator for the 2,000 simulated "
                            "datasets; the digits test set and training "
                            "draws from another, with fold shuffles seeded "
                            "by replicate"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Two different questions",
        f"""Episode 1 ended where most advice about test-set noise ends: one split is noisy, so cross-validate. K-fold CV uses every row for testing once, averages, and is described — in textbooks, in library documentation, in nearly every report that quotes it — as an estimate of how well *your* model will perform on new data.

There are two quantities that phrase could mean. One is the error of the model you actually fitted, on this data: call it err_xy, because it depends on the particular X and y you were given. The other is the average of that error over all the training sets of the same size you might have been given: call it err. A model fitted to an unlucky sample has err_xy above err; a lucky one, below. The sentence is about err_xy. The question is which one CV estimates.

That is hard to check on real data, because err_xy is never observed. So start where it can be computed exactly. Least squares on 100 rows of 20 standard-normal features: the expected squared error of a fitted coefficient vector b on a fresh row is 1 + |b - β|², no estimation involved.

{snip['linear'].markdown()}

The correlation between the CV estimate and the error of the model it accompanies is **{cvx['corr']:+.3f}**. Across 2,000 datasets, knowing the CV estimate tells you nothing about whether this particular fit is better or worse than usual.""",
        figures=[figs["f0"]])

    post.add(
        "What it tracks instead",
        f"""The figure shows why. The true errors vary by an SD of {lin['err_xy'].std():.3f} from dataset to dataset; the CV estimates by {lin['cv'].std():.3f}, {ratio:.1f} times as much. CV's own noise, from which rows landed in which fold and how hard they happened to be, swamps the variation it is supposed to detect. What survives the averaging is the level — the CV estimates centre on {lin['cv'].mean():.3f}, against an average true error of {lin['err']:.3f}, slightly high because each fold's model trained on 90 rows rather than 100.

So CV is an estimate of err, the average over training sets, and a reasonable one: its RMSE against err is {cva['rmse']:.3f}, against {cvx['rmse']:.3f} for err_xy. That is useful — it is the right quantity for choosing between *procedures*, since a procedure is what you would rerun on new data — but it is not the sentence.

The usual interval around it does not cover either target at its nominal rate.

{snip['cover'].markdown()}

The standard error used there treats the {lin['n']} per-row losses as independent, and they are not: every fold's model was trained on most of the other folds' rows, so the losses are correlated, and the interval is too narrow for err and pointed at the wrong target for err_xy. Bates, Hastie and Tibshirani prove the linear-model version of this and propose nested CV to get the width right; the point here is the one before that, about what the centre is.""")

    post.add(
        "On real digits it runs backwards",
        f"""A simulation is only as good as its assumptions, so the same question on the UCI digits. Split once into a fixed test set of {dig['test']} images and a training pool. Draw {dig['reps']} training sets of {dig['train']} images from the pool; for each, cross-validate logistic regression, fit it to all {dig['train']}, and score that fit on the fixed test set — large enough to stand in for its true accuracy, and disjoint from every training draw.

{snip['digits'].markdown()}

The correlation is **{dig['corr']:+.2f}**: negative. Training sets that cross-validate well produce models that do slightly worse on new digits. The mechanism is the same averaging seen from the other side. A training set that happens to contain more awkward, atypical digits scores poorly in CV, because its held-out folds are full of them, and produces a model that has seen more awkward digits and handles the test set's better. CV rewards the easy sample, not the good model.

The fixed, disjoint test set matters. An earlier version of this experiment scored each fit on all the digits it had not trained on, which makes the test set the complement of the training set: a draw that took the hard digits left an easier test set behind, which exaggerates the effect — that version gave -0.28. With the test set fixed in advance it is {dig['corr']:+.2f}: smaller, and still there.""",
        figures=[figs["f1"]])

    post.add(
        "What answers the question you asked",
        f"""If the question really is how well *this* model will do, the instrument is a held-out set the model never saw. Scoring each simulated fit on 100 fresh rows, the correlation with its true error is {fresh['corr']:+.2f} and the naive interval covers {fresh['cover']:.0%} of the time.

That comparison is unfair to CV, though: it spends 100 extra rows. On the same budget — fit on 80 of the 100 rows, score on the other 20 —

{snip['holdout'].markdown()}

the held-out estimate still tracks its own model, at {split['corr']:+.2f}, and pays twice for it: the estimate is noisy, and the model being estimated was fit on less data, so its average true error is {lin['split_err_xy'].mean():.3f} instead of {lin['err']:.3f}.

That is the real choice. Cross-validation gives a stable estimate of how a *procedure* performs on data like yours, which is what you want when choosing between procedures. A held-out set gives a noisy estimate of how one *fitted model* performs, which is what you want when deciding whether to ship it. Most reports use the first and describe it as the second.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        f"""**The zero correlation is for this model and this size.** Bates, Hastie and Tibshirani show it holds for linear models broadly; for flexible models and very large n the conditional and average errors move closer together and the distinction matters less. At {lin['n']} rows and {lin['p']} features it matters a lot.

**The digits test set is a proxy for the truth.** It is a sample of {dig['test']} images, so each model's score on it is its true accuracy plus an error of order 0.008 — shared in part between models, since they face the same images, but not identical. The sign survived the switch to a fixed test set; the size should be read loosely.

**Squared error is skewed.** Part of the interval undercoverage, for every estimator in the table, comes from using a normal interval for a mean of skewed losses. The comparison between rows is fair; the absolute coverages are a little worse than a better interval would give.""")

    post.add(
        "What to keep",
        f"""1. "The error of your model" can mean the error of the model you fitted (err_xy) or the average error of models fitted to data like yours (err). They are different numbers.

2. On simulated least squares, the 10-fold CV estimate's correlation with err_xy is {cvx['corr']:+.2f}. CV estimates err.

3. CV's naive 90% interval covers err_xy {cvx['cover']:.0%} of the time and err {cva['cover']:.0%}: the width is wrong because the per-row losses are not independent.

4. On real digits the correlation is {dig['corr']:+.2f}: easy training samples cross-validate well and generalise slightly worse.

5. To evaluate a model you will ship, hold data out. To compare procedures, cross-validate. Say which one you did.""")

    post.add(
        "Exercise",
        """Find a report — your own is best — that quotes a cross-validated score as the expected performance of a deployed model. Rewrite the sentence so it says what was measured: the average performance of the training procedure on samples of that size. If the deployment decision depended on the specific fitted model being good, check whether a held-out score exists, and if not, what it would cost to get one.

Then, if you have a model where two cross-validated scores differ by less than their fold-to-fold spread, ask which question you were trying to answer. For choosing a procedure, the CV comparison is the right one, and episode 1's pairing applies to it. For choosing a fitted model, it is not the right comparison at all.""")

    post.add(
        "Next",
        """Episode 3 takes the rule everyone learns right after cross-validation — fit every preprocessing step inside the folds — and measures seven ways of breaking it. Some leak a third of the accuracy scale. Some leak nothing at all.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
