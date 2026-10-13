"""ML 35: The Best Cross-Validated Score Is the Search's Best Luck.

The thirty-fifth episode of *Machine Learning, Taught Through What Breaks*,
closing Arc VII. "The best cross-validated score estimates how well the
tuned model will do." Measured with grid searches of an RBF support vector
machine on 200 rows, twenty times per grid, against the selected model's
accuracy on 20,000 fresh rows:

* Untuned, cross-validation was slightly pessimistic, by 0.013. Searching
  64 configurations turned it optimistic by 0.023, and on labels that were
  pure noise the best of 64 cross-validated scores averaged 0.575 for models
  whose true accuracy was 0.500.
* The searched model was no better on fresh data than the untuned default
  (0.647 against 0.659), though its best score was 0.023 higher.
* Nested cross-validation, which tunes inside each outer fold, came within
  two to three standard errors of the truth in both settings, and removed
  about two-thirds of the optimism on the noise labels.

Run: `standarderror run ml135_tuning --publish`
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
    t = around.tuning_optimism()
    n = around.tuning_optimism(null=True)
    return {"t": t, "n": n, "tk": {r["size"]: r for r in t},
            "nk": {r["size"]: r for r in n}}


# ------------------------------------------------------------------ figures

def _lines(ax, m, rows, ref=None):
    x = [r["size"] for r in rows]
    ax.plot(x, [r["best_cv"] for r in rows], marker="o", color=m.series[1],
            label="best cross-validated score")
    ax.plot(x, [r["nested"] for r in rows], marker="o", color=m.series[2],
            label="nested cross-validation")
    ax.plot(x, [r["true"] for r in rows], marker="o", color=m.ink, lw=2,
            label="selected model on 20,000 fresh rows")
    if ref is not None:
        ax.axhline(ref, color=m.grid, ls=":", lw=1)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def figures(res: dict) -> dict:
    out: dict = {}
    t, n, tk, nk = res["t"], res["n"], res["tk"], res["nk"]

    out["f0"] = charts.diagram(
        lambda ax, m: _lines(ax, m, t),
        title="The more you search, the more the best score flatters",
        subtitle=("RBF support vector machine on 200 rows of episode 4's task, "
                  "grids of 1 to 64 configurations, five-fold "
                  "cross-validation; averages over 20 training sets."),
        xlabel="configurations searched (log scale)",
        ylabel="accuracy",
        figsize=(7.4, 4.6),
        source="Simulated (episode 4's task); standarderror/ml/around.py.",
        alt=("Three lines over grid size: the best cross-validated score "
             "rises above the selected model's true accuracy as the grid "
             "grows; nested cross-validation stays a little below the truth."),
        caption=(f"With 64 configurations the best score is "
                 f"{tk[64]['best_cv']:.3f}; the model it selects scores "
                 f"**{tk[64]['true']:.3f}** on fresh rows, and the untuned "
                 f"default {tk[1]['true']:.3f}."),
        path=str(IMG / f"ml135-f0-task.{EXT}"))[0]

    out["f1"] = charts.diagram(
        lambda ax, m: _lines(ax, m, n, ref=0.5),
        title="On pure noise, the search finds signal",
        subtitle=("The same searches on 200 rows whose labels are coin flips "
                  "independent of the ten features. Every model's true "
                  "accuracy is 0.5 (dotted)."),
        xlabel="configurations searched (log scale)",
        ylabel="accuracy",
        figsize=(7.4, 4.6),
        source="Simulated; standarderror/ml/around.py.",
        alt=("The best cross-validated score climbs from about 0.52 to 0.575 "
             "as the grid grows; the true accuracy stays at 0.5; nested "
             "cross-validation stays near 0.52."),
        caption=(f"The best of 64 cross-validated scores averaged "
                 f"**{nk[64]['best_cv']:.3f}** for models that are right half "
                 f"the time."),
        path=str(IMG / f"ml135-f1-null.{EXT}"))[0]

    rows = []
    for label, rr in (("task", t), ("noise", n)):
        for r in rr:
            rows.append([label, f"{r['size']}", f"{r['best_cv']:.3f}",
                         f"{r['true']:.3f}", f"{r['optimism']:+.3f}",
                         f"± {r['optimism_se']:.3f}", f"{r['nested']:.3f}",
                         f"{r['oracle']:.3f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["labels", "grid", "best CV score", "selected model, true",
                "optimism", "s.e.", "nested CV", "best in grid, true"],
        title="What the search reported, and what it delivered",
        subtitle=("Means over 20 training sets of 200 rows. 'Best in grid' is "
                  "the highest fresh-data accuracy any configuration in the "
                  "grid reaches when trained on the same rows."),
        source=("Simulated (episode 4's task, and noise labels); "
                "standarderror/ml/around.py."),
        alt=("An eight-row table: optimism is slightly negative for one or "
             "four configurations on the task and positive for 16 and 64; on "
             "noise it grows from 0.018 to 0.076."),
        caption=("Optimism is the best CV score minus the selected model's "
                 "true accuracy, with its standard error over the 20 "
                 "repetitions."),
        bold_cells={(3, 4), (7, 2), (7, 4)},
        align="lrrrrrrr",
        path=str(IMG / f"ml135-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    t, n, tk = res["t"], res["n"], res["tk"]

    def task(panel, m):
        panel.plot([r["best_cv"] for r in t], color=m.series[1], lw=2)
        panel.plot([r["true"] for r in t], color=m.ink, lw=2)

    def noise(panel, m):
        panel.plot([r["best_cv"] for r in n], color=m.series[1], lw=2)
        panel.axhline(0.5, color=m.ink, lw=1)
        panel.set_ylim(0.45, 0.6)

    def default(panel, m):
        panel.bar([0, 1], [tk[1]["true"], tk[64]["true"]],
                  color=[m.series[2], m.series[1]])
        panel.set_ylim(0.6, 0.68)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=35,
        headline="The search's best luck",
        panels=[(task, f"{tk[64]['optimism']:+.3f}", "optimism, 64 configs"),
                (noise, f"{res['nk'][64]['best_cv']:.3f}", "best CV on noise"),
                (default, f"{tk[1]['true']:.3f}", "untuned default, true")],
        note=("The highest of many noisy scores is high partly because it "
              "was lucky. The more configurations a search tries, the more "
              "its best score reports luck instead of skill."),
        alt=("Three hand-drawn frames: two lines drawing apart; a line rising "
             "above a level one; two bars, the first slightly taller."),
        path=str(IMG / f"ml135-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["task"] = s.run("""
        from standarderror.ml import around

        for r in around.tuning_optimism():   # 200 rows, 20 repetitions
            print(f"{r['size']:3d} configurations: best CV {r['best_cv']:.3f}, "
                  f"true {r['true']:.3f}, nested {r['nested']:.3f}")
        """)
    out["null"] = s.run("""
        for r in around.tuning_optimism(null=True):   # labels are coin flips
            print(f"{r['size']:3d} configurations: best CV {r['best_cv']:.3f}, "
                  f"true {r['true']:.3f}, nested {r['nested']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    tk, nk = res["tk"], res["nk"]
    t1, t16, t64 = tk[1], tk[16], tk[64]
    n1, n64 = nk[1], nk[64]

    # The spine, asserted rather than trusted.
    assert t1["optimism"] < 0 and t64["optimism"] > 3 * t64["optimism_se"]
    assert t16["optimism"] > 0 and t64["optimism"] > t16["optimism"]
    assert t64["true"] < t1["true"]
    assert tk[4]["oracle"] < t1["true"] and tk[4]["true"] < t1["true"]
    assert t64["oracle"] - t64["true"] > 0.015
    assert all(abs(r["nested_gap"]) < 3 * r["nested_gap_se"] + 0.002
               for r in (t16, t64))
    assert abs(n64["true"] - 0.5) < 0.005 and n64["best_cv"] > 0.56
    assert all(a["best_cv"] < b["best_cv"] for a, b in zip(res["n"], res["n"][1:]))
    assert n64["optimism"] > 0.07 and n64["nested"] < 0.535
    assert n64["optimism"] > 2.5 * n64["nested_gap"]
    assert (n64["best_cv"] - nk[16]["best_cv"]) < (nk[4]["best_cv"] - n1["best_cv"])

    post = Post(
        title=(f"{SERIES_TAG} 35: The Best Cross-Validated Score Is the "
               f"Search's Best Luck"),
        slug="ml-35-tuning",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=35,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Cross-validation estimates one model fairly. The best of "
                  "many cross-validated scores is the maximum of noisy "
                  "estimates, and it overstates the model it picks."),
        summary=(
            f"Grid searches of an RBF support vector machine on 200 rows of a "
            f"simulated task, twenty times per grid size. Untuned, the "
            f"cross-validated score was {-t1['optimism']:.3f} below the model's "
            f"accuracy on 20,000 fresh rows. With 64 configurations the best "
            f"score was {t64['optimism']:.3f} above it ({t64['best_cv']:.3f} "
            f"against {t64['true']:.3f}), and the selected model did no better "
            f"on fresh data than the untuned default ({t1['true']:.3f}). On "
            f"labels that were pure noise, the best of 64 scores averaged "
            f"{n64['best_cv']:.3f} for models whose true accuracy was "
            f"{n64['true']:.3f}. Nested cross-validation gave "
            f"{t64['nested']:.3f} and {n64['nested']:.3f}."),
        tags=["machine-learning", "data-science", "statistics",
              "cross-validation", "hyperparameter-tuning", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: episode 4's two-class Gaussian task in 10 dimensions "
            "(Bayes error 0.268), 200 training rows per repetition (seeds "
            "1,000 to 1,019) and 20,000 fixed test rows; and the same with "
            "ten standard-normal features and coin-flip labels (seeds 0 to "
            "19).",
            "Machinery: `standarderror/ml/around.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's SVC (RBF kernel) and "
            "GridSearchCV with five stratified folds; grids log-spaced in C "
            "from 0.1 to 1,000 and gamma from 0.001 to 3.2 (the one-cell "
            "grid is C = 1, gamma = 0.1); nested estimates with five outer "
            "folds.",
            "Where this stops: Varma and Simon, \"Bias in error estimation "
            "when using cross-validation for model selection\", *BMC "
            "Bioinformatics* (2006); Cawley and Talbot, \"On over-fitting in "
            "model selection and subsequent selection bias in performance "
            "evaluation\", *JMLR* (2010); episode 2 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("data seeds as listed; fold shuffles from the "
                            "repetition number; about half an hour of CPU in "
                            "all"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The score that chose the model",
        f"""Every model in this series was chosen by something: a learning rate, a depth, a regularisation strength, a number of components or clusters. Episode 34 ended with a correction fitted on held-out rows; this episode measures the most common fitting-on-held-out-rows of all, the hyperparameter search. A grid of configurations is cross-validated, the one with the best score is kept, and the best score is reported as the estimate of how well the model will do. That last step is the sentence under test: *the best cross-validated score estimates how well the tuned model will do*.

Episode 2 found that cross-validation estimates the average error of a procedure, not of one fitted model. Here the question is narrower and has a sharper answer. A support vector machine with an RBF kernel has two settings, C and gamma. On 200 rows of episode 4's simulated task, grids of 1, 4, 16 and 64 configurations are each searched by five-fold cross-validation. The winner is refitted on all 200 rows and scored on 20,000 fresh rows, which is its true accuracy to within a few thousandths. The whole thing is repeated on twenty training sets.

{snip['task'].markdown()}

With one configuration, the untuned default, there is no search, and the cross-validated score is {t1['best_cv']:.3f} against a true {t1['true']:.3f}: slightly *pessimistic*, as cross-validation usually is, because each fold trains on four-fifths of the rows. With 64 configurations the best score is **{t64['best_cv']:.3f}** and the selected model's true accuracy **{t64['true']:.3f}**: the estimate is now optimistic by {t64['optimism']:.3f}, with a standard error of {t64['optimism_se']:.3f}.""",
        figures=[figs["f0"]])

    post.add(
        "Why the best score is high",
        f"""Each cross-validated score is an estimate with noise, and on 200 rows the noise is several points of accuracy. The search keeps the highest. Among sixteen or sixty-four configurations of similar real quality, the highest score belongs, more often than not, to the one whose folds happened to suit it. Its score is high partly because the configuration is good and partly because it was lucky, and only the first part comes with it to new data. Varma and Simon showed the bias on gene-expression classifiers; Cawley and Talbot named it over-fitting in model selection.

Pure noise makes the luck visible. The same searches on 200 rows whose labels are coin flips, unrelated to the features, where every model's true accuracy is 0.5:

{snip['null'].markdown()}

With no search the score is {n1['best_cv']:.3f}. With 64 configurations the best score averages **{n64['best_cv']:.3f}**, and the models it picks score {n64['true']:.3f} on fresh rows. A report of "57.5% accuracy, cross-validated" on this data would describe a classifier that knows nothing.

The growth with grid size is also informative. Going from one configuration to four raised the best noise score by {nk[4]['best_cv'] - n1['best_cv']:.3f}; going from sixteen to sixty-four, four times as many again, raised it by only {n64['best_cv'] - nk[16]['best_cv']:.3f}. Neighbouring cells of an SVM grid fit nearly the same function and make nearly the same mistakes on the same folds, so they are not independent draws of luck. The optimism depends on how many genuinely different models a search compares, which suggests that searches across model families, or over many unrelated hyperparameters, would inflate the best score faster than a fine grid over two.""",
        figures=[figs["f1"]])

    post.add(
        "What the search bought",
        f"""The search was supposed to find a better model, and the true accuracies say whether it did. The best configuration anywhere in the 64-cell grid, trained on the same 200 rows, would have scored {t64['oracle']:.3f} on fresh data. The search picked one that scored {t64['true']:.3f}, {t64['oracle'] - t64['true']:.3f} short of it, because the noise that inflated the winner's score also misled the choice. And the untuned default, with no search at all, scored **{t1['true']:.3f}**, better on average than the tuned model.

The four-cell grid is the odd point in the first figure: its cells sit at the corners of the range, every one of them worse than the default, so both its best score and its chosen model fall. A search can only choose among the configurations it is given.

That is not a general verdict against tuning. On this task the default happens to sit near the best region of the grid, and with more rows the scores are less noisy and the search chooses better. It is a verdict against reading the best score as evidence: the 0.023 by which the tuned model's score beat the default's untuned score was luck, and on fresh data the order reversed.""",
        figures=[figs["f2"]])

    post.add(
        "Nested cross-validation",
        f"""The standard repair keeps the search out of the estimate. Nested cross-validation splits the rows into outer folds; within each outer training set it runs the whole search, by inner cross-validation, and scores the chosen configuration on the outer fold it never saw. The search is part of the procedure being evaluated, so its luck is evaluated too.

On the task, nested cross-validation estimated {t64['nested']:.3f} for the 64-configuration search, {abs(t64['nested_gap']):.3f} below the truth with a standard error of {t64['nested_gap_se']:.3f}: pessimistic by about as much as plain cross-validation of the untuned model, and for the same reason. On the noise labels it estimated {n64['nested']:.3f}, against a best score of {n64['best_cv']:.3f} and a truth of {n64['true']:.3f}. Not perfect at 200 rows, but it removed about two-thirds of the optimism.

The price is computation: five outer folds each repeat the full search, so 64 configurations cost five times as many fits. The alternative, when data allow, is simpler: hold out a test set that the search never touches, and report the selected model's score on it once.""")

    post.add(
        "Where this breaks",
        """**Small data on purpose.** At 200 rows the cross-validation noise is large, and so is the optimism. With thousands of rows both shrink, though the optimism still grows with the number of configurations compared.

**A smooth grid.** Neighbouring configurations of an SVM behave alike, so 64 grid cells are fewer than 64 independent chances. Random search over many hyperparameters, or comparisons across model families, offer more independent chances and more optimism.

**One model family.** Selecting among different algorithms by their best cross-validated scores adds a second layer of selection on top of this one.

**Accuracy only.** The same arithmetic applies to any score that is maximised over a search, including AUC and log loss.""")

    post.add(
        "What to keep",
        f"""1. Cross-validation of one model was slightly pessimistic here ({t1['optimism']:+.3f}). The best of 64 cross-validated scores was optimistic by {t64['optimism']:.3f}.

2. On labels that were pure noise, the best of 64 scores averaged {n64['best_cv']:.3f}; the selected models scored {n64['true']:.3f} on fresh data.

3. The search's winner scored {t64['true']:.3f} on fresh rows, {t64['oracle'] - t64['true']:.3f} below the best configuration in its grid and below the untuned default's {t1['true']:.3f}.

4. Report the performance of a tuned model from nested cross-validation or from a test set the search never saw, and count the configurations tried: the more there were, the less the best score means.""")

    post.add(
        "Exercise",
        """Take a model you tuned and record how many configurations the search tried and the best cross-validated score. Then shuffle the labels, rerun the identical search, and record its best score.

The second number is what your search reports when there is nothing to find. If it is not far below the first, most of your reported score is the search's luck. Then estimate the tuned model by nested cross-validation and compare.""")

    post.add(
        "Next",
        """Arc VII ends here, at thirty-five episodes. The steps around a model turned out to need measuring as much as the model: components that keep variance and not relevance, clusters that are partitions of space, distances that depend on which columns carry information, calibration that holds on average, and scores that report the luck of the search that produced them. Arc VIII goes one step further back, to the rows themselves and the process that produced them, starting with episode 36 and the most familiar rule about rows that come in order: never shuffle a time series.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
