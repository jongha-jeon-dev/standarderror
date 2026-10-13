"""ML 42: Stacking Learns to Pick the Best Model, Given Honest Predictions.

The forty-second episode of *Machine Learning, Taught Through What
Breaks*. "Stacking learns the best way to combine models." Measured on
episode 4's simulated task, four base models trained on 100 to 3,000 rows
and a logistic meta-model fitted on their predicted probabilities:

* Fitted on the base models' in-sample predictions, the meta-model put its
  largest weight on the model that fitted the training rows best: 1-NN,
  whose accuracy it then reproduced exactly at every size (0.594 at 1,000
  rows, against the SVM's 0.702); without 1-NN in the pool, on the forest,
  with log loss nearly twice the SVM's.
* Fitted on out-of-fold predictions, it put most of its weight on the SVM
  and came within 0.003 of it from 300 rows on, never more than 0.001
  above. At 100 rows it scored 0.589, below the best member (0.602).
* A plain average beat every member at 100 rows when the pool held no
  certain model; from 300 rows on, the weak members pulled it below the
  SVM.

Run: `standarderror run ml142_stacking --publish`
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


def compute() -> dict:
    a = combine.stacking_study(with_1nn=True)
    b = combine.stacking_study(with_1nn=False)
    return {"a": a, "b": b, "ka": {r["n"]: r for r in a},
            "kb": {r["n"]: r for r in b}}


# ------------------------------------------------------------------ figures

def _lines(ax, m, rows_):
    n = [r["n"] for r in rows_]
    spec = (("SVM", m.grid, "-", "SVM (best member)"),
            ("1-NN", m.grid, ":", "1-NN"),
            ("stack, in-sample", m.series[1], "-", "stack on in-sample predictions"),
            ("stack, out-of-fold", m.series[2], "-", "stack on out-of-fold predictions"),
            ("average", m.series[3], "--", "plain average"))
    for key, col, ls, lab in spec:
        ax.plot(n, [r["acc"][key] for r in rows_], color=col, ls=ls, marker="o",
                lw=2 if key.startswith("stack") else 1.5, label=lab)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def figures(res: dict) -> dict:
    out: dict = {}
    a, b, ka, kb = res["a"], res["b"], res["ka"], res["kb"]

    out["f0"] = charts.diagram(
        lambda ax, m: _lines(ax, m, a),
        title="Stacking follows whichever predictions it is shown",
        subtitle=("Logistic regression, SVM, random forest and 1-NN on "
                  "episode 4's task; a logistic meta-model on their "
                  "probabilities. Accuracy on 20,000 test rows."),
        xlabel="training rows (log scale)",
        ylabel="accuracy",
        figsize=(7.4, 4.6),
        source="Simulated (episode 4's task); standarderror/ml/combine.py.",
        alt=("The in-sample stack's line lies exactly on the 1-NN line, "
             "lowest of all; the out-of-fold stack's line lies on the SVM's "
             "from 300 rows on; the average lies between."),
        caption=(f"On in-sample predictions the stack became 1-NN "
                 f"({ka[1000]['acc']['stack, in-sample']:.3f} at 1,000 rows); "
                 f"on out-of-fold ones, the SVM "
                 f"(**{ka[1000]['acc']['stack, out-of-fold']:.3f}** against "
                 f"{ka[1000]['acc']['SVM']:.3f})."),
        path=str(IMG / f"ml142-f0-stack.{EXT}"))[0]

    body = []
    for label, rr in (("with 1-NN", ka[1000]), ("with 25-NN", kb[1000])):
        for fit in ("in-sample", "out-of-fold"):
            w = rr["weights"][fit]
            body.append([label if fit == "in-sample" else "", fit,
                         *[f"{v:+.2f}" for v in w.values()],
                         f"{rr['acc'][f'stack, {fit}']:.3f}"])
    names = list(ka[1000]["weights"]["in-sample"])
    names_b = list(kb[1000]["weights"]["in-sample"])
    out["f1"] = charts.table_image(
        body,
        header=["pool", "meta-model fitted on", *[f"weight: {n}" for n in names[:3]],
                f"weight: {names[3]} / {names_b[3]}", "stack accuracy"],
        title="What the meta-model learned to trust",
        subtitle=("Coefficients of the logistic meta-model on each base "
                  "model's predicted probability, 1,000 training rows, mean "
                  "over 6 data sets; base accuracies: logistic "
                  f"{ka[1000]['acc']['logistic']:.3f}, SVM "
                  f"{ka[1000]['acc']['SVM']:.3f}, forest "
                  f"{ka[1000]['acc']['forest']:.3f}, 1-NN "
                  f"{ka[1000]['acc']['1-NN']:.3f}, 25-NN "
                  f"{kb[1000]['acc']['25-NN']:.3f}."),
        source="Simulated (episode 4's task); standarderror/ml/combine.py.",
        alt=("A four-row table of meta-model weights. In-sample fits put "
             "weight 13 on 1-NN or 28 on the forest; out-of-fold fits put "
             "about 4 on the SVM and little elsewhere."),
        caption=("In-sample, each base model is judged on rows it has "
                 "memorised to a different degree, and the meta-model "
                 "rewards memory."),
        align="llrrrrr",
        path=str(IMG / f"ml142-f1-weights.{EXT}"))[0]

    body = []
    for label, rr in (("with 1-NN", a), ("with 25-NN", b)):
        for r in rr:
            bm = r["best"]
            body.append([label if r is rr[0] else "", f"{r['n']:,}",
                         f"{bm} {r['acc'][bm]:.3f}",
                         f"{r['acc']['stack, in-sample']:.3f}",
                         f"{r['acc']['stack, out-of-fold']:.3f}",
                         f"{r['acc']['average']:.3f}",
                         f"{r['nll']['stack, out-of-fold']:.3f}",
                         f"{r['nll'][bm]:.3f}"])
    out["f2"] = charts.table_image(
        body,
        header=["pool", "rows", "best member", "stack, in-sample",
                "stack, out-of-fold", "average", "OOF stack log loss",
                "best member log loss"],
        title="Stacking against its best member and a plain average",
        subtitle=("Accuracy on 20,000 test rows; means over 20, 10, 6 and 3 "
                  "data sets at 100, 300, 1,000 and 3,000 rows."),
        source="Simulated (episode 4's task); standarderror/ml/combine.py.",
        alt=("An eight-row table. The out-of-fold stack is within a "
             "0.003 of the best member from 300 rows on and below it "
             "at 100; the average beats every member only at 100 rows "
             "without 1-NN."),
        caption=("The best member is the one with the highest test "
                 "accuracy at that size, which a practitioner would have to "
                 "find by validation."),
        align="lrlrrrrr",
        path=str(IMG / f"ml142-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    ka, kb = res["ka"], res["kb"]

    def p1(panel, m):
        w = ka[1000]["weights"]["in-sample"]
        panel.bar(range(4), list(w.values()),
                  color=[m.series[1], m.series[2], m.series[3], m.ink])

    def p2(panel, m):
        w = ka[1000]["weights"]["out-of-fold"]
        panel.bar(range(4), list(w.values()),
                  color=[m.series[1], m.series[2], m.series[3], m.ink])

    def p3(panel, m):
        n = [100, 300, 1000, 3000]
        panel.plot([kb[x]["acc"]["average"] for x in n], color=m.series[3], lw=2)
        panel.plot([kb[x]["acc"]["stack, out-of-fold"] for x in n],
                   color=m.series[2], lw=2)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=42,
        headline="A model of models",
        panels=[(p1, f"{ka[1000]['weights']['in-sample']['1-NN']:+.1f}",
                 "1-NN weight, in-sample"),
                (p2, f"{ka[1000]['weights']['out-of-fold']['SVM']:+.1f}",
                 "SVM weight, out-of-fold"),
                (p3, f"{kb[100]['acc']['average']:.3f}", "plain average, 100 rows")],
        note=("A stack trusts whichever model looks best on the predictions "
              "it is fitted to. Shown honest predictions, it mostly picks "
              "the best member; shown in-sample ones, the best memoriser."),
        alt=("Three hand-drawn frames: one tall bar among short ones; a "
             "different tall bar; two crossing lines."),
        path=str(IMG / f"ml142-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["a"] = s.run("""
        from standarderror.ml import combine

        for r in combine.stacking_study(with_1nn=True):
            acc = r["acc"]
            print(f"{r['n']:5d} rows: SVM {acc['SVM']:.3f}, 1-NN {acc['1-NN']:.3f}"
                  f" | stacks {acc['stack, in-sample']:.3f} (in-sample), "
                  f"{acc['stack, out-of-fold']:.3f} (OOF)")
        """)
    out["b"] = s.run("""
        for r in combine.stacking_study(with_1nn=False):   # 25-NN instead
            acc = r["acc"]
            best = r["best"]
            print(f"{r['n']:5d} rows: best member {best} {acc[best]:.3f}"
                  f" | out-of-fold stack {acc['stack, out-of-fold']:.3f}, "
                  f"average {acc['average']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    a, b, ka, kb = res["a"], res["b"], res["ka"], res["kb"]
    members_a = ("logistic", "SVM", "forest", "1-NN")
    members_b = ("logistic", "SVM", "forest", "25-NN")

    # The spine, asserted rather than trusted.
    for r in a:
        assert abs(r["acc"]["stack, in-sample"] - r["acc"]["1-NN"]) < 1e-12
        w = r["weights"]["in-sample"]
        assert max(w, key=w.get) == "1-NN" and w["1-NN"] > 12
    for r in b:
        w = r["weights"]["in-sample"]
        assert max(w, key=w.get) == "forest" and w["forest"] > 27
        assert r["nll"]["stack, in-sample"] > 1.6 * max(r["nll"][k] for k in members_b)
        assert r["nll"]["stack, in-sample"] > 1.85 * r["nll"]["SVM"]
    for rr, mem in ((a, members_a), (b, members_b)):
        for r in rr:
            best = max(r["acc"][k] for k in mem)
            assert r["acc"]["stack, out-of-fold"] <= best + 0.001
            if r["n"] >= 300:
                assert best - r["acc"]["stack, out-of-fold"] < 0.0035
                w = r["weights"]["out-of-fold"]
                assert max(w, key=w.get) == "SVM" and w["SVM"] > 3.5
    assert ka[100]["best"] == "forest" and ka[100]["acc"]["stack, out-of-fold"] < 0.592
    assert kb[100]["acc"]["average"] > max(kb[100]["acc"][k] for k in members_b)
    assert kb[100]["acc"]["average"] > kb[100]["acc"]["stack, out-of-fold"] + 0.015
    for r in b[1:]:
        assert r["acc"]["average"] < r["acc"]["SVM"]
    for r in a:
        assert r["acc"]["average"] < max(r["acc"][k] for k in members_a)

    o1, s1 = ka[1000]["acc"]["stack, out-of-fold"], ka[1000]["acc"]["SVM"]

    post = Post(
        title=(f"{SERIES_TAG} 42: Stacking Learns to Pick the Best Model, "
               f"Given Honest Predictions"),
        slug="ml-42-stacking",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=42,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A meta-model trusts whichever base model looks best on "
                  "the predictions it is fitted to. On in-sample "
                  "predictions that is the best memoriser; on honest ones "
                  "it is usually just the best model."),
        summary=(
            f"Four base models on a simulated task, 100 to 3,000 training "
            f"rows, and a logistic meta-model on their probabilities. Fitted "
            f"on in-sample predictions it gave 1-NN a weight of "
            f"{ka[1000]['weights']['in-sample']['1-NN']:.0f} and reproduced "
            f"its accuracy exactly ({ka[1000]['acc']['1-NN']:.3f} at 1,000 "
            f"rows, against the SVM's {s1:.3f}). Fitted on out-of-fold "
            f"predictions it put its weight on the SVM and matched it "
            f"({o1:.3f}), from 300 rows on never more than 0.001 above the "
            f"best member; at 100 rows it scored "
            f"{ka[100]['acc']['stack, out-of-fold']:.3f} against "
            f"{ka[100]['acc']['forest']:.3f}. A plain average beat every "
            f"member at 100 rows when no member was certain, and trailed the "
            f"SVM from 300 on."),
        tags=["machine-learning", "data-science", "statistics", "ensembles",
              "stacking", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: episode 4's two-class Gaussian task in 10 "
            "dimensions, Bayes error 0.268; training sets of 100, 300, 1,000 "
            "and 3,000 rows (20, 10, 6 and 3 data sets, seeds from 0); "
            "20,000 fixed test rows.",
            "Machinery: `standarderror/ml/combine.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's LogisticRegression, SVC "
            "(RBF, Platt probabilities), RandomForestClassifier (200 trees), "
            "KNeighborsClassifier (k = 1 or 25); meta-model "
            "LogisticRegression (C = 10,000) on the base probabilities; "
            "out-of-fold predictions from five stratified folds.",
            "Where this stops: Wolpert, \"Stacked generalization\", *Neural "
            "Networks* (1992); Breiman, \"Stacked regressions\", *Machine "
            "Learning* (1996); Smith and Wallis, \"A simple explanation of "
            "the forecast combination puzzle\", *Oxford Bulletin of "
            "Economics and Statistics* (2009).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("data seeds from 0, fold shuffles from the data "
                            "seed, model random_state 0; about six minutes "
                            "of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Let the data choose the weights",
        f"""Episode 41 ended with an average dragged down by a member that was too weak or too sure of itself. The obvious repair is to stop weighting members equally and let the data decide how much each one counts. Wolpert called the idea stacked generalisation in 1992: fit a second model, the meta-model, whose inputs are the base models' predictions and whose target is the label. It is now a standard ingredient of competition-winning pipelines, sold with the sentence under test: *stacking learns the best way to combine models*.

Four base models on episode 4's simulated task, whose Bayes boundary is curved: logistic regression, an RBF support vector machine, a random forest and 1-nearest-neighbour. A logistic meta-model is fitted on their predicted probabilities in two ways. The quick way uses each base model's predictions on the rows it was trained on. The careful way uses out-of-fold predictions: each training row's prediction comes from a copy of the base model trained without it.

{snip['a'].markdown()}

On in-sample predictions, the stack's accuracy equals 1-NN's at every size, to the last digit: **{ka[1000]['acc']['stack, in-sample']:.3f}** at 1,000 rows, the worst of the four models, while the SVM scores {s1:.3f}. On out-of-fold predictions the stack scores **{o1:.3f}**, the SVM's accuracy.""",
        figures=[figs["f0"]])

    post.add(
        "What the meta-model trusted",
        f"""The meta-model's weights say why. On in-sample predictions, 1-NN is right about every training row, because each row is its own nearest neighbour, and the forest, whose trees are grown deep, is nearly as perfect. The meta-model sees one input that predicts the label without error and gives it a weight of **{ka[1000]['weights']['in-sample']['1-NN']:+.1f}**, then reproduces it on test rows where it is the weakest model. Breiman made the point for regression in 1996: the level-one data must come from predictions the base models did not fit, or the combination rewards overfitting.

Replace 1-NN with 25-NN, which cannot memorise its training rows, and the same mistake picks the next best memoriser: the in-sample meta-model gives the forest a weight of {kb[1000]['weights']['in-sample']['forest']:+.1f}, and its log loss, {kb[1000]['nll']['stack, in-sample']:.3f} at 1,000 rows, is nearly twice the SVM's {kb[1000]['nll']['SVM']:.3f} and worse than any member's. The meta-model is not learning which model is good; it is learning which model has seen the answers.

On out-of-fold predictions, the weights are {', '.join(f'{k} {v:+.2f}' for k, v in ka[1000]['weights']['out-of-fold'].items())}. Almost all the weight is on the SVM, and 1-NN, whose out-of-fold predictions are as noisy as its test predictions, gets nothing.""",
        figures=[figs["f1"]])

    post.add(
        "Honest stacking is mostly selection",
        f"""Out-of-fold stacking works, in the sense that it avoids the trap. Whether it *combines* is another question:

{snip['b'].markdown()}

From 300 rows up, in both pools, the out-of-fold stack is within 0.0035 of the best member and never more than 0.001 above it. The meta-model has learned to pick the SVM, which a validation score would also have done. At 100 rows it is worse than the best member: {ka[100]['acc']['stack, out-of-fold']:.3f}, against the forest's {ka[100]['acc']['forest']:.3f}, though a practitioner would have had to identify the forest by validation, with its own noise. With 100 rows the out-of-fold predictions are noisy and the meta-model's four weights are estimated from the same 100 rows, so the stack pays for estimating the combination.

The plain average shows the other side of that trade. In the pool without 1-NN, at 100 rows, it scores **{kb[100]['acc']['average']:.3f}**, above every member and {kb[100]['acc']['average'] - kb[100]['acc']['stack, out-of-fold']:.3f} above the stack: equal weights cost nothing to estimate. From 300 rows on, logistic regression and 25-NN fall well behind the SVM, and equal weights let them drag the average below it ({kb[1000]['acc']['average']:.3f} against {kb[1000]['acc']['SVM']:.3f} at 1,000 rows). Forecasters know this as the forecast combination puzzle: a simple average of forecasts often beats the weights estimated to combine them optimally, and Smith and Wallis traced it to the estimation error in the weights, exactly the cost visible at 100 rows here.""",
        figures=[figs["f2"]])

    post.add(
        "When stacking earns its place",
        """Stacking improves on its best member when the members are good in different places: one model better on one region of the inputs, another elsewhere, and enough rows to learn where. On this task the SVM is best almost everywhere, so the right combination is close to "use the SVM", and the stack found it. On tasks where diverse models of similar quality each capture part of the structure, as in many competition data sets, out-of-fold stacking can beat every member; the measurement here says what to compare it against before believing that it has.

The comparison has two baselines, not one. The stack should beat the best member chosen by validation, or it is an expensive way of choosing; and it should beat the plain average, or its weights are costing more to estimate than they gain.

Honest stacking also has a price that is easy to forget. Every base model is fitted once per fold and once more on all the rows, six fits instead of one with five folds, and the stack's own score needs data that played no part in building it. Here that was 20,000 separate test rows. A stack scored on the same folds that produced its inputs would carry episode 35's optimism on top of everything measured above.""")

    post.add(
        "Where this breaks",
        """**One task with one dominant model.** The SVM's lead from 300 rows on is large, which makes selection the right answer. With members of equal quality and complementary errors, the out-of-fold stack has more to gain and the average is harder to beat.

**A linear meta-model on probabilities.** Meta-models on log-odds, with non-negativity constraints on the weights (as Breiman recommended), or with the original features as extra inputs, behave differently; the in-sample trap is the same for all of them.

**Few repetitions at large sizes.** Three data sets at 3,000 rows; differences of a thousandth there are within noise. The in-sample stack's exact match to 1-NN is not.

**Accuracy and log loss only.** A stack tuned for AUC or for a cost-weighted metric would choose differently, by the same mechanism.""")

    post.add(
        "What to keep",
        f"""1. Fitted on in-sample predictions, the meta-model gave its largest weight to the model that had memorised the most: 1-NN (weight {ka[1000]['weights']['in-sample']['1-NN']:+.1f}), whose accuracy it reproduced exactly, or the forest when 1-NN was absent.

2. Fitted on out-of-fold predictions it chose the SVM and matched it to within 0.0035 from 300 rows on, never more than 0.001 above it; at 100 rows it lost {ka[100]['acc']['forest'] - ka[100]['acc']['stack, out-of-fold']:.3f} to the best member.

3. A plain average beat every member at 100 rows when no member was certain, and trailed the best member once the weak ones fell behind.

4. Fit stacks on out-of-fold predictions only, and keep a stack only if it beats both the best single model and the plain average on data neither has seen.""")

    post.add(
        "Exercise",
        """Take a stacked model and print the meta-model's weights. Refit the meta-model on in-sample base predictions instead of out-of-fold ones and print them again.

Then score three things on a held-out set: the stack, the base model with the best validation score, and a plain average of the base models. If the stack's weights are concentrated on one model, compare it with that model alone; if it is not better than the average, its weights are not worth their estimation error.""")

    post.add(
        "Next",
        """Stacking and averaging both end in the same question: is this model better than that one? Episode 43 looks at a combination inside a single model, naive Bayes adding up evidence feature by feature, before episode 44 turns to the tests that compare models.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
