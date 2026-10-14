"""ML 6: On Separable Data, Your Logistic Coefficients Are Your Stopping Rule.

The sixth episode of *Machine Learning, Taught Through What Breaks*, opening
arc II. "Logistic regression finds the maximum-likelihood coefficients" is
true when they exist. When a hyperplane separates the classes the likelihood
keeps rising as the coefficients grow, and there is no maximum.

Measured:

* All 50 stratified half splits of the Wisconsin breast cancer data are
  linearly separable on their 30 standardised features.
* Unpenalised logistic regression in scikit-learn returns coefficients whose
  size is set by the stopping tolerance: norm 2.4 at tol 1e-2, 399 at the
  default 1e-4, 1,289 at 1e-10. Test log loss goes from 0.08 to 1.5.
* Gradient descent's weight norm grows by a fixed amount per tenfold more
  steps -- logarithmically, for ever -- while its direction creeps towards
  the max-margin direction of the problem it solves, with the intercept
  counted in the norm: about 6 degrees from scikit-learn's hard-margin SVM,
  whose intercept is free. Test accuracy rises along the way and test log
  loss, after a minimum, rises too.

Run: `standarderror run ml106_separable --publish`
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
    gd = ln.gd_path()
    return {"frac": ln.separable_fraction(), "tol": ln.tolerance_sweep(),
            "c": ln.c_path(), "gd": gd,
            "decade": ln.norm_per_decade(gd["rows"]),
            "aug": ln.augmented_margin_angle(pair=gd["pair"])}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    rows = res["gd"]["rows"]
    t = [r["step"] for r in rows]

    def norm(ax, m):
        ax.plot(t, [r["norm"] for r in rows], lw=2.6, color=m.series[0],
                label="weight norm")
        ax.set_xscale("log")
        ax2 = ax.twinx()
        ax2.plot(t, [r["angle"] for r in rows], lw=2.2, ls="--",
                 color=m.series[1],
                 label="angle to scikit-learn's hard-margin SVM")
        ax2.set_ylabel("degrees", color=m.ink_secondary)
        ax2.spines["top"].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.4,
                  loc="upper center")

    g = res["gd"]
    out["f0"] = charts.diagram(
        norm,
        title="The coefficients never stop growing",
        subtitle=(f"Gradient descent on the unpenalised logistic loss, "
                  f"digits {g['pair'][0]} vs {g['pair'][1]}, {g['train']} "
                  f"separable training images. Step size 1, from zero."),
        xlabel="gradient-descent steps",
        ylabel="norm of the weights",
        source="Measured; standarderror/ml/linear.py.",
        alt=("A weight norm rising as a straight line against log steps, "
             "and a dashed angle falling slowly."),
        caption=(f"The norm grows by **{res['decade']:.1f} per tenfold more "
                 f"steps** - logarithmically, without limit. The angle to "
                 f"scikit-learn's hard-margin SVM is "
                 f"{rows[-1]['angle']:.1f} degrees after "
                 f"{rows[-1]['step']:,} steps, heading for "
                 f"{res['aug']:.1f}, where the max-margin direction of the "
                 f"problem gradient descent solves lies."),
        path=str(IMG / f"ml106-f0-norm.{EXT}"))[0]

    def test(ax, m):
        ax.plot(t, [r["test_log_loss"] for r in rows], lw=2.6,
                color=m.series[2], label="test log loss")
        ax.set_xscale("log")
        ax2 = ax.twinx()
        ax2.plot(t, [r["test_accuracy"] for r in rows], lw=2.2, ls="--",
                 color=m.series[0], label="test accuracy")
        ax2.axhline(g["svm_test_accuracy"], lw=1, ls=":", color=m.ink)
        ax2.set_ylabel("accuracy", color=m.ink_secondary)
        ax2.spines["top"].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.4,
                  loc="center right")

    best = min(rows, key=lambda r: r["test_log_loss"])
    out["f1"] = charts.diagram(
        test,
        title="Accuracy keeps improving while log loss turns",
        subtitle=(f"The same run, scored on {g['test']} held-out images. "
                  f"Dotted: the hard-margin SVM's test accuracy, "
                  f"{g['svm_test_accuracy']:.3f}."),
        xlabel="gradient-descent steps",
        ylabel="test log loss",
        source="Measured; standarderror/ml/linear.py.",
        alt=("A test log loss that falls to a minimum and then rises, and a "
             "dashed test accuracy that climbs in steps towards a dotted "
             "line."),
        caption=(f"Test log loss is lowest, {best['test_log_loss']:.4f}, at "
                 f"step {best['step']:,} and **{rows[-1]['test_log_loss']:.4f}"
                 f"** at {rows[-1]['step']:,}. Accuracy climbs to "
                 f"{rows[-1]['test_accuracy']:.3f} as the direction nears "
                 f"the max-margin one."),
        path=str(IMG / f"ml106-f1-test.{EXT}"))[0]

    rows_t = [[f"{r['tol']:.0e}", f"{r['iterations']}", f"{r['norm']:,.1f}",
               f"{r['certain']:.0%}", f"{r['test_accuracy']:.3f}",
               f"{r['test_log_loss']:.3f}"] for r in res["tol"]]
    c1 = next(r for r in res["c"] if r["C"] == 1.0)
    rows_t.append(["C = 1 (default L2)", "-", f"{c1['norm']:.1f}", "-",
                   f"{c1['test_accuracy']:.3f}",
                   f"{c1['test_log_loss']:.3f}"])
    bold = {(1, c) for c in range(6)}
    out["f2"] = charts.table_image(
        rows_t,
        header=["stopping tolerance", "iterations", "coefficient norm",
                "training rows at p within 1e-6 of 0 or 1", "test accuracy",
                "test log loss"],
        title="Unpenalised logistic regression, five answers",
        subtitle=("scikit-learn, penalty=None, lbfgs, the same separable "
                  "half of the breast cancer data. The bold row is the "
                  "library's default tolerance."),
        source="Measured; standarderror/ml/linear.py.",
        alt=("A six-row table: coefficient norm from 2 to over 1,000 as the "
             "tolerance tightens, test log loss from 0.08 to 1.6, and the "
             "penalised default at the bottom with the best log loss."),
        caption=("Every row is the same model fitted to the same rows. The "
                 "only thing that differs is when the optimiser was told to "
                 "stop."),
        bold_cells=bold, align="lrrrrr",
        path=str(IMG / f"ml106-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    rows, tol = res["gd"]["rows"], res["tol"]

    def grow(panel, m):
        panel.plot(np.log([r["step"] for r in rows]),
                   [r["norm"] for r in rows], lw=2.8, color=m.series[0])

    def tols(panel, m):
        panel.plot(range(len(tol)), np.log([r["norm"] for r in tol]),
                   lw=2.8, color=m.series[1])

    def ll(panel, m):
        panel.plot(np.log([r["step"] for r in rows]),
                   [r["test_log_loss"] for r in rows], lw=2.8,
                   color=m.series[2])

    d = next(r for r in tol if r["tol"] == 1e-4)
    return charts.lecture_hero(
        series=SERIES_TAG, episode=6,
        headline="Separable data has no maximum-likelihood fit",
        panels=[(grow, f"+{res['decade']:.1f}", "norm per tenfold steps"),
                (tols, f"{tol[0]['norm']:.0f} -> {tol[-1]['norm']:,.0f}",
                 "norm, by stopping rule"),
                (ll, f"{d['test_log_loss']:.2f}", "test log loss at default")],
        note=(f"Every half split of the breast cancer data is linearly "
              f"separable, so the unpenalised logistic fit has no maximum. "
              f"What the library returns is set by when it stopped, and at "
              f"the default tolerance its test log loss is "
              f"{d['test_log_loss'] / tol[0]['test_log_loss']:.0f} times "
              f"the early-stopped one's."),
        alt=("Three hand-drawn frames: a line rising for ever; a line rising "
             "across five settings; and a curve that dips and rises."),
        path=str(IMG / f"ml106-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["sep"] = s.run("""
        from standarderror.ml import linear as ln

        print(f"half splits separable: {ln.separable_fraction(splits=50):.0%}")
        """)
    out["tol"] = s.run("""
        for r in ln.tolerance_sweep():
            print(f"tol {r['tol']:.0e}  {r['iterations']:>3} iterations  "
                  f"norm {r['norm']:8.1f}  test log loss {r['test_log_loss']:.3f}")
        """)
    out["gd"] = s.run("""
        g = ln.gd_path(steps=100_000)
        for r in g["rows"][::6]:
            print(f"step {r['step']:>6,}  norm {r['norm']:6.2f}  "
                  f"angle {r['angle']:5.1f}  train loss {r['train_loss']:.1e}")
        print(f"growth per tenfold steps: {ln.norm_per_decade(g['rows']):.2f}")
        """)
    out["c"] = s.run("""
        for r in ln.c_path():
            print(f"C {r['C']:8.0e}  norm {r['norm']:7.2f}  "
                  f"test accuracy {r['test_accuracy']:.3f}  "
                  f"test log loss {r['test_log_loss']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    tol, cp, g = res["tol"], res["c"], res["gd"]
    rows = g["rows"]
    d = next(r for r in tol if r["tol"] == 1e-4)
    first, last = tol[0], tol[-1]
    c1 = next(r for r in cp if r["C"] == 1.0)
    bestc = min(cp, key=lambda r: r["test_log_loss"])
    best = min(rows, key=lambda r: r["test_log_loss"])

    # The spine, asserted rather than trusted.
    assert res["frac"] == 1.0 and g["separable"]
    assert all(a["norm"] < b["norm"] for a, b in zip(tol, tol[1:]))
    assert last["norm"] > 100 * first["norm"]
    assert d["test_log_loss"] > 10 * first["test_log_loss"]
    assert d["test_accuracy"] < first["test_accuracy"]
    assert all(a["norm"] < b["norm"] for a, b in zip(cp, cp[1:]))
    assert bestc["C"] <= 10
    assert 2.0 < res["decade"] < 5.0
    assert rows[-1]["angle"] < rows[0]["angle"] / 4
    # The limit of GD's direction is the augmented problem's max-margin
    # direction, measurably off the free-intercept SVM, and GD has not
    # reached it yet.
    aug = res["aug"]
    assert 3.0 < aug < rows[-1]["angle"]
    assert rows[-1]["test_log_loss"] > 1.2 * best["test_log_loss"]
    assert rows[-1]["test_accuracy"] >= best["test_accuracy"]
    assert rows[-1]["train_loss"] < 1e-4
    assert c1["test_log_loss"] < min(r["test_log_loss"] for r in tol)

    post = Post(
        title=(f"{SERIES_TAG} 6: On Separable Data, Your Logistic "
               f"Coefficients Are Your Stopping Rule"),
        slug="ml-6-separable",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=6,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Logistic regression is taught as finding the "
                  "maximum-likelihood coefficients. When a hyperplane "
                  "separates the classes there is no maximum, and what comes "
                  "back is decided by when the optimiser stopped."),
        summary=(
            f"Every one of 50 stratified half splits of the Wisconsin breast "
            f"cancer data is linearly separable on its 30 features, so the "
            f"unpenalised logistic likelihood has no maximum. scikit-learn "
            f"returns coefficients of norm {first['norm']:.1f} at a stopping "
            f"tolerance of 1e-2, {d['norm']:.0f} at its default 1e-4 and "
            f"{last['norm']:,.0f} at 1e-10, with test log loss going from "
            f"{first['test_log_loss']:.2f} to {d['test_log_loss']:.2f}. "
            f"Gradient descent's weight norm grows {res['decade']:.1f} per "
            f"tenfold more steps, for ever, while its direction creeps "
            f"towards a max-margin limit - {rows[0]['angle']:.0f} degrees "
            f"from scikit-learn's hard-margin SVM at the start, "
            f"{rows[-1]['angle']:.1f} after 100,000 steps, heading for "
            f"{aug:.1f}, because gradient descent counts the intercept in "
            f"the norm. Test accuracy improves along the way; test log loss "
            f"turns and rises."),
        tags=["machine-learning", "data-science", "statistics",
              "logistic-regression", "optimisation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Breast Cancer Wisconsin (Diagnostic), as bundled with "
            "scikit-learn, CC BY 4.0; stratified half splits, standardised "
            "on the training half.",
            "UCI Optical Recognition of Handwritten Digits, as bundled with "
            "scikit-learn, CC BY 4.0: the 1s and 8s as a binary problem.",
            "Machinery: `standarderror/ml/linear.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Albert and Anderson, \"On the existence of "
            "maximum likelihood estimates in logistic regression models\", "
            "*Biometrika* (1984); Soudry, Hoffer, Nacson, Gunasekar and "
            "Srebro, \"The implicit bias of gradient descent on separable "
            "data\", *JMLR* (2018), for the logarithmic norm and the "
            "max-margin limit.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("split seeds 0 to 49; gradient descent is "
                            "deterministic from zero with step size 1"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A maximum that is not there",
        f"""Arc I was about what a score means. Arc II takes the models people reach for first and pushes them past the conditions the textbook states, starting with the most common classifier there is. Logistic regression is taught as maximum likelihood: write down the probability of the observed labels as a function of the coefficients, and the fitted model is the coefficient vector that maximises it.

That sentence assumes the maximum exists. It does not when a hyperplane separates the two classes in the training data. Take any separating direction w: every training row is on the right side, so every term in the likelihood improves as w is scaled up, and keeps improving for ever. The likelihood approaches 1 and never reaches it. Albert and Anderson worked this out in 1984.

That sounds like an edge case for tiny datasets. It is not an edge case on one of the most used teaching datasets in machine learning.

{snip['sep'].markdown()}

Every stratified half of the Wisconsin breast cancer data — 284 tumours, 30 measurements — is perfectly separated by some hyperplane. With 30 features this informative that is the expected situation, not a fluke — and it is the informativeness, not the count of features per row, that does it.

Cover counted the count's part in 1965. For points in general position in d dimensions with random labels, a separating hyperplane exists with high probability while there are fewer than about 2d of them, and with high probability does not beyond, the switch growing sharper as d grows. Here d is 31 with the intercept and there are 284 rows, far past 2d, so random labels would not be separable. Real labels are not random: the classes differ systematically, and a modest signal in each of 30 features adds up to a margin. The better a feature set predicts, the more likely the textbook estimator is not to exist. That is backwards from the intuition that maximum likelihood fails on hard problems; it fails on easy ones.""")

    post.add(
        "What the library hands back",
        f"""So what does a library return when asked for a maximum that does not exist? Unpenalised logistic regression in scikit-learn, the same training half, five stopping tolerances:

{snip['tol'].markdown()}

The coefficient norm runs from {first['norm']:.1f} to {last['norm']:,.0f}: a factor of {last['norm'] / first['norm']:.0f}, set entirely by when the optimiser was told it was close enough. At the library's default tolerance the model puts {d['certain']:.0%} of training rows within one in a million of certainty, and on held-out tumours its log loss is {d['test_log_loss']:.2f} against {first['test_log_loss']:.2f} for the loosest tolerance — a log loss that size, from a model that is right {d['test_accuracy']:.1%} of the time, means its mistakes are made with near-total confidence. Even its accuracy is lower, {d['test_accuracy']:.3f} against {first['test_accuracy']:.3f}: the loose tolerance stopped early, and early stopping is a regulariser.

Most people never see this, because scikit-learn's default is not unpenalised. It applies an L2 penalty with C = 1, which makes the optimum exist; the bottom row of the table below is that model, and it is the best of the six on log loss. The sentence is true of what the library does by default and false of what the textbook describes.""",
        figures=[figs["f2"]])

    post.add(
        "Where gradient descent goes instead",
        f"""Gradient descent on the unpenalised loss, from zero, does something definite even though the maximum does not exist. On the 1s and 8s of the digits, whose training half is also separable:

{snip['gd'].markdown()}

The training loss falls towards zero and the weight norm grows by about {res['decade']:.1f} for every tenfold increase in steps — logarithmically, the rate Soudry and colleagues proved. It never stops. What does converge is the *direction*. The angle to scikit-learn's hard-margin SVM falls from {rows[0]['angle']:.0f} degrees to {rows[-1]['angle']:.1f}, but it will not reach zero. This gradient descent treats the intercept as one more weight, on a column of ones, so the margin it maximises is that of the augmented problem, in which the intercept counts towards the norm; the SVM leaves its intercept free. On these digits the two max-margin directions are {aug:.1f} degrees apart, and that is where the angle is heading, slowly. Unpenalised logistic regression trained by gradient descent is, in the limit, a hard-margin SVM for the problem as the optimiser poses it — a fact about the optimiser, not about the likelihood.

The test set shows the two faces of that. Accuracy improves as the direction approaches the max-margin one, from {rows[0]['test_accuracy']:.3f} to {rows[-1]['test_accuracy']:.3f}. Log loss reaches its minimum, {best['test_log_loss']:.4f}, at step {best['step']:,}, and then rises to {rows[-1]['test_log_loss']:.4f}: the direction gets better, and the confidence gets worse faster.""",
        figures=[figs["f0"], figs["f1"]])

    post.add(
        "A penalty makes it a question with an answer",
        f"""Any L2 penalty restores a maximum. The coefficient that comes back then depends on C, which is at least a quantity someone chose on purpose:

{snip['c'].markdown()}

The norm grows with C without limit, roughly logarithmically again, because the unpenalised problem underneath has no ceiling. Test log loss is best at C = {bestc['C']:g} and degrades steadily beyond it. Choosing C by cross-validation, as is standard, is choosing how far along that unbounded path to stop — the same decision the tolerance made silently, made out loud.

The two curves also say something about how to read a fitted coefficient. Under separation a coefficient's size is not a measurement of an effect; it is a measurement of how long the optimiser ran, or how weak the penalty was. Its sign and its ratio to the other coefficients carry information — that is the direction, and it converges. Its magnitude, and anything computed from it, such as an odds ratio, does not.""")

    post.add(
        "Where this breaks",
        """**Separation is a property of the training sample.** A population can overlap while every sample of it you can afford is separable; the breast cancer data is not separable in principle, only in every half of it measured here. Whether the unpenalised maximum exists can change with one added row.

**Quasi-complete separation behaves the same way for some coefficients.** When a hyperplane separates the classes except for ties on it, the estimates of some coefficients diverge and the rest are fine. Software warnings for it are inconsistent.

**Standard errors are the first casualty.** Statistical packages that report Wald standard errors for logistic coefficients report enormous ones under separation, and the p-values beside them are meaningless. Firth's penalised likelihood is the usual repair when coefficients, not predictions, are the goal.

**The digits path uses one pair.** The logarithmic growth and the max-margin limit are theorems; the step at which log loss turns depends on the data and the step size.""")

    post.add(
        "What to keep",
        f"""1. The maximum-likelihood logistic fit does not exist when a hyperplane separates the training classes. All 50 half splits of the breast cancer data are separable.

2. Unpenalised, the coefficients a library returns are set by its stopping rule: norm {first['norm']:.1f} to {last['norm']:,.0f} across tolerances, test log loss {first['test_log_loss']:.2f} to {last['test_log_loss']:.2f}.

3. Gradient descent's norm grows like log t ({res['decade']:.1f} per decade here) and its direction converges to the max-margin direction of the problem it solves — with the intercept in the norm, {aug:.1f} degrees from scikit-learn's hard-margin SVM here.

4. Along that path test accuracy can improve while test log loss gets worse: the direction improves, the confidence overshoots.

5. A penalty gives the question an answer. Pick its strength deliberately; the alternative is that a tolerance picks it.""")

    post.add(
        "Exercise",
        """Check whether your training data is separable: fit a linear SVM with a very large C and look at the training accuracy. If it is 100%, any unpenalised logistic coefficients you have reported were a property of your optimiser's settings. Refit at two tolerances; if the coefficients move by more than the precision you reported them to, they were never estimates.

Then look at the probabilities. If a large share of training rows sit within a hair of 0 or 1, the model's confidence on new data is not to be trusted even if its accuracy is, and a penalty — or calibration on held-out data — is the repair.""")

    post.add(
        "Next",
        """Episode 7 stays with penalties and asks what they are a penalty on. A lasso charges for coefficient size, and coefficient size depends on the units of the feature it belongs to. Change a feature's units and you change whether the lasso keeps it.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
