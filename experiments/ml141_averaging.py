"""ML 41: An Average of Models Beats the Average Model, Not the Best One.

The forty-first episode of *Machine Learning, Taught Through What Breaks*,
opening Arc IX (combining and comparing models). "Averaging several models
gives a better model than any of them." Measured on the digits, averaging
the predicted probabilities of every combination of six classifiers, and
on the diabetes data with four regressors:

* 11 of the 57 combinations beat their best member's accuracy; 8 fell
  below their members' mean accuracy.
* Averaged with a single decision tree, whose probabilities are all 0 or 1,
  the SVM's probabilities lost every disagreement: the average scored
  exactly the tree's 0.825, against the SVM's 0.977.
* What is guaranteed is weaker. Log loss of the average was at or below
  the members' mean in all 57 combinations; for squared error the
  ensemble's error equals the mean member's minus their spread, to 1e-12.
  On diabetes that still left the average of four regressors worse than
  ridge regression alone.

Run: `standarderror run ml141_averaging --publish`
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
    t = combine.averaging_table()
    return {"members": combine.averaging_members(), "t": t,
            "pairs": {f"{a}+{b}": combine.averaging_pair(a, b)
                      for a, b in (("SVM", "tree"), ("SVM", "calibrated tree"),
                                   ("SVM", "5-NN"), ("SVM", "naive Bayes"))},
            "amb": combine.ambiguity()}


# ------------------------------------------------------------------ figures

def _scatter(ax, m, t):
    groups = (("with the tree", lambda c: "tree" in c, m.series[1]),
              ("with naive Bayes, no tree",
               lambda c: "naive Bayes" in c and "tree" not in c, m.series[2]),
              ("neither", lambda c: "tree" not in c and "naive Bayes" not in c,
               m.series[3]))
    for lab, f, col in groups:
        rr = [r for r in t if f(r["members"])]
        ax.scatter([r["best_acc"] for r in rr], [r["acc"] for r in rr], s=30,
                   color=col, label=f"{lab} ({len(rr)})", zorder=3)
    ax.plot([0.8, 1.0], [0.8, 1.0], color=m.ink, lw=1)
    ax.set_xlim(0.82, 0.99)
    ax.set_ylim(0.8, 1.0)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")


def _amb(ax, m, a):
    names = list(a["members"])
    vals = [a["members"][n] for n in names] + [a["mean_member"], a["ensemble"]]
    labels = names + ["mean member", "average of four"]
    cols = [m.series[1]] * len(names) + [m.grid, m.ink]
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.annotate("", xy=(len(vals) - 1, a["ensemble"]),
                xytext=(len(vals) - 1, a["mean_member"]),
                arrowprops={"arrowstyle": "->", "color": m.series[2], "lw": 1.6})
    ax.text(len(vals) - 1.45, (a["ensemble"] + a["mean_member"]) / 2,
            f"spread {a['ambiguity']:.0f}", fontsize=8.5, ha="right",
            color=m.series[2])


def figures(res: dict) -> dict:
    out: dict = {}
    t, mem, pairs, a = res["t"], res["members"], res["pairs"], res["amb"]

    out["f0"] = charts.diagram(
        lambda ax, m: _scatter(ax, m, t),
        title="An average is rarely better than its best member",
        subtitle=("Every combination of two to six of six classifiers on the "
                  "digits, probabilities averaged; accuracy on held-out "
                  "halves, means over 5 splits. Above the line: the average "
                  "beat its best member."),
        xlabel="accuracy of the best member in the combination",
        ylabel="accuracy of the averaged probabilities",
        figsize=(7.2, 4.8),
        source=("Optical Recognition of Handwritten Digits (UCI), CC BY 4.0, "
                "via scikit-learn; standarderror/ml/combine.py."),
        alt=("Scatter of 57 combinations. Combinations containing the "
             "decision tree range from about 0.83 to 0.98, all below the "
             "line but one; those with naive Bayes but no tree sit around 0.85 to "
             "0.98; those with neither sit near or slightly above the line."),
        caption=(f"One of the 31 combinations containing the decision tree "
                 f"beat its best member, pairing it with naive Bayes; "
                 f"{sum(r['acc'] > r['best_acc'] for r in t)} of 57 "
                 f"combinations did."),
        path=str(IMG / f"ml141-f0-combos.{EXT}"))[0]

    out["f1"] = charts.diagram(
        lambda ax, m: _amb(ax, m, a),
        title="The guarantee: better than the mean member, by the spread",
        subtitle=("Diabetes, four regressors trained on half the patients, "
                  "squared error on the other half. The average's error is "
                  "the members' mean error minus their mean squared spread "
                  "around the average, exactly."),
        xlabel="",
        ylabel="mean squared error",
        figsize=(7.4, 4.6),
        source=("Diabetes data (Efron et al., 2004) via scikit-learn; "
                "standarderror/ml/combine.py."),
        alt=("Bars for ridge, 10-NN, forest and tree, the tree much higher; "
             "a bar for the mean member and a lower bar for the average, "
             "which is still above ridge."),
        caption=(f"{a['mean_member']:.0f} − {a['ambiguity']:.0f} = "
                 f"{a['ensemble']:.0f}, to within {a['identity_gap']:.0e}; "
                 f"ridge alone scored **{a['members']['ridge']:.0f}**."),
        path=str(IMG / f"ml141-f1-ambiguity.{EXT}"))[0]

    body = [[k, f"{v['acc']:.4f}", f"{v['nll']:.3f}", f"{v['certain']:.0%}"]
            for k, v in mem.items()]
    body += [[f"average: {k.replace('+', ' + ')}", f"{v['acc']:.4f}", "",
              f"sides with 2nd: {v['with_b']:.0%}"] for k, v in pairs.items()]
    out["f2"] = charts.table_image(
        body,
        header=["model", "accuracy", "log loss", "certain / disagreements"],
        title="Who wins the disagreements",
        subtitle=("Members on the digits, and four two-model averages. For "
                  "members, the share of digits given probability exactly 1; "
                  "for averages, the share of the two members' disagreements "
                  "that the average resolves in favour of the second."),
        source=("Optical Recognition of Handwritten Digits (UCI), CC BY 4.0; "
                "standarderror/ml/combine.py."),
        alt=("An eleven-row table: seven members with accuracy, log loss "
             "and share of certain predictions, then four averages with the "
             "share of disagreements won by the second model."),
        caption=("The tree is certain about every digit, so paired with a "
                 "model that is almost never certain it wins every "
                 "disagreement. Calibrating it removes that, but the pair "
                 "still trails the SVM alone."),
        align="lrrr",
        path=str(IMG / f"ml141-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    mem, pairs, a, t = res["members"], res["pairs"], res["amb"], res["t"]

    def p1(panel, m):
        panel.bar([0, 1, 2], [mem["SVM"]["acc"], mem["tree"]["acc"],
                              pairs["SVM+tree"]["acc"]],
                  color=[m.series[1], m.series[2], m.ink])
        panel.set_ylim(0.75, 1.0)

    def p2(panel, m):
        panel.scatter([r["best_acc"] for r in t], [r["acc"] for r in t],
                      s=12, color=m.series[1])
        panel.plot([0.95, 0.99], [0.95, 0.99], color=m.ink, lw=1)

    def p3(panel, m):
        panel.bar([0, 1, 2], [a["members"]["ridge"], a["mean_member"],
                              a["ensemble"]],
                  color=[m.series[1], m.grid, m.ink])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=41,
        headline="Averaging is not voting",
        panels=[(p1, f"{pairs['SVM+tree']['acc']:.3f}", "SVM averaged with tree"),
                (p2, f"{sum(r['acc'] > r['best_acc'] for r in t)} of 57",
                 "beat their best member"),
                (p3, f"{a['identity_gap']:.0e}", "ambiguity identity gap")],
        note=("For log loss or squared error, an average is guaranteed to "
              "beat its average member, not its best one. And a member "
              "certain about everything can take over every decision."),
        alt=("Three hand-drawn frames: three bars, the last equal to the "
             "second; scattered dots mostly under a line; three bars, the "
             "middle one tallest."),
        path=str(IMG / f"ml141-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["table"] = s.run("""
        from standarderror.ml import combine

        t = combine.averaging_table()   # 57 combinations, 5 half-splits
        best = sum(r["acc"] > r["best_acc"] for r in t)
        below = sum(r["acc"] < r["mean_acc"] for r in t)
        print(f"beat their best member: {best} of {len(t)}")
        print(f"below their members' mean accuracy: {below} of {len(t)}")
        top = max(t, key=lambda r: r["acc"])
        print(f"best combination: {' + '.join(top['members'])}, "
              f"{top['acc']:.4f} against its best member's {top['best_acc']:.4f}")
        """)
    out["pair"] = s.run("""
        p = combine.averaging_pair("SVM", "tree")
        print(f"SVM + tree: accuracy {p['acc']:.4f}")
        print(f"on {p['disagree']} disagreements the average sided with "
              f"the tree {p['with_b']:.0%} of the time")
        """)
    out["amb"] = s.run("""
        a = combine.ambiguity()          # diabetes, four regressors
        print(f"mean member {a['mean_member']:.1f} - spread "
              f"{a['ambiguity']:.1f} = average {a['ensemble']:.1f} "
              f"(gap {a['identity_gap']:.0e})")
        print("members: " + ", ".join(f"{k} {v:.0f}"
                                      for k, v in a["members"].items()))
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    t, mem, pairs, a = res["t"], res["members"], res["pairs"], res["amb"]
    n_best = sum(r["acc"] > r["best_acc"] for r in t)
    n_below = sum(r["acc"] < r["mean_acc"] for r in t)
    n_nll_best = sum(r["nll"] < r["best_nll"] for r in t)
    top = max(t, key=lambda r: r["acc"])
    st, sc, sk, sn = (pairs[k] for k in ("SVM+tree", "SVM+calibrated tree",
                                         "SVM+5-NN", "SVM+naive Bayes"))
    with_tree = [r for r in t if "tree" in r["members"]]

    # The spine, asserted rather than trusted.
    assert len(t) == 57 and n_best == 11 and n_below == 8
    assert all(r["nll"] <= r["mean_nll"] + 1e-12 for r in t)
    assert n_nll_best < 20
    assert len(with_tree) == 31
    tree_wins = [r for r in with_tree if r["acc"] > r["best_acc"]]
    assert len(tree_wins) == 1
    assert set(tree_wins[0]["members"]) == {"naive Bayes", "tree"}
    assert all(abs(r["acc"] - mem["tree"]["acc"]) < 1e-12 for r in t
               if set(r["members"]) in ({"logistic", "tree"}, {"SVM", "tree"},
                                        {"forest", "tree"}))
    assert mem["tree"]["certain"] == 1.0 and st["with_b"] == 1.0
    assert sn["with_b"] > 0.8 and mem["naive Bayes"]["certain"] > 0.5
    assert mem["calibrated tree"]["certain"] == 0
    assert 0 < mem["forest"]["certain"] < 0.01 and mem["SVM"]["certain"] == 0
    assert mem["tree"]["acc"] < sc["acc"] < mem["SVM"]["acc"]
    assert sk["acc"] > max(mem["SVM"]["acc"], mem["5-NN"]["acc"])
    assert set(top["members"]) == {"SVM", "5-NN", "forest"}
    assert top["acc"] - top["best_acc"] < 0.01
    assert a["identity_gap"] < 1e-9
    assert a["ensemble"] < a["mean_member"]
    assert a["members"]["ridge"] < a["ensemble"]

    post = Post(
        title=(f"{SERIES_TAG} 41: An Average of Models Beats the Average "
               f"Model, Not the Best One"),
        slug="ml-41-averaging",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=41,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Averaging carries a guarantee, and it is weaker than the "
                  "one usually quoted. When one member is certain about "
                  "everything, averaging probabilities hands it every "
                  "decision."),
        summary=(
            f"Six classifiers on the digits, and the averaged probabilities "
            f"of all 57 combinations of two or more. {n_best} combinations "
            f"beat their best member's accuracy and {n_below} fell below "
            f"their members' mean. Averaged with a single decision tree, "
            f"whose probabilities are all 0 or 1, an SVM scoring "
            f"{mem['SVM']['acc']:.3f} lost every disagreement and the "
            f"average scored the tree's {mem['tree']['acc']:.3f}. The "
            f"guarantee that does hold is weaker: the average's log loss "
            f"was at or below the members' mean every time, and on diabetes "
            f"the average of four regressors improved on their mean error "
            f"by exactly their spread, {a['ambiguity']:.0f}, and still lost "
            f"to ridge regression alone."),
        tags=["machine-learning", "data-science", "statistics", "ensembles",
              "model-averaging", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Optical Recognition of Handwritten Digits (UCI), as bundled "
            "with scikit-learn, CC BY 4.0; 1,797 images, five stratified "
            "half splits (seeds 0 to 4).",
            "Diabetes data of Efron et al. (2004), as bundled with "
            "scikit-learn; one half split (seed 0).",
            "Machinery: `standarderror/ml/combine.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's LogisticRegression (C = "
            "0.1), SVC (RBF, Platt probabilities), KNeighborsClassifier "
            "(k = 5), RandomForestClassifier (300 trees), GaussianNB, "
            "DecisionTreeClassifier, CalibratedClassifierCV (isotonic, five "
            "folds); RidgeCV, KNeighborsRegressor (k = 10), "
            "RandomForestRegressor, DecisionTreeRegressor.",
            "Where this stops: Krogh and Vedelsby, \"Neural network "
            "ensembles, cross validation, and active learning\", *NeurIPS* "
            "(1995); Dietterich, \"Ensemble methods in machine learning\" "
            "(2000); Kuncheva, *Combining Pattern Classifiers* (2004).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("split seeds 0 to 4 (digits) and 0 (diabetes); "
                            "model random_state 0; under a minute of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The free lunch",
        f"""Forty episodes measured single models and the data under them. Arc IX turns to what is done with several models: combining them, stacking them, and deciding which is better. The most common combination is the simplest. Train a few different models, average their predicted probabilities, and report the average, on the strength of a sentence that sounds like a theorem: *averaging several models gives a better model than any of them*.

Six classifiers were trained on half the digits and scored on the other half, five times: regularised logistic regression, a support vector machine, five-nearest-neighbours, a random forest, Gaussian naive Bayes and a single decision tree. Their accuracies range from {mem['tree']['acc']:.3f} for the tree to {mem['5-NN']['acc']:.3f} for nearest neighbours. Every combination of two or more, 57 in all, was averaged.

{snip['table'].markdown()}

Only **{n_best}** of the 57 averages beat the best model in them. The best average, of the SVM, nearest neighbours and the forest, gained {top['acc'] - top['best_acc']:.4f} over nearest neighbours alone, about half a point. And {n_below} averages did worse than the *mean* of their members, which an average is supposed to beat by construction.""",
        figures=[figs["f0"]])

    post.add(
        "Averaging is not voting",
        f"""The worst averages contain the decision tree, and the worst of those are pairs. Averaged with logistic regression, the SVM or the forest, the tree's accuracy, {mem['tree']['acc']:.3f}, is exactly what the pair scored. The reason is in what a single fully grown tree predicts: each leaf holds one class, so its predicted probability is 1 for one digit and 0 for the rest, on {mem['tree']['certain']:.0%} of the digits, while those three models are almost never that certain about anything (the forest on {mem['forest']['certain']:.1%} of digits).

{snip['pair'].markdown()}

Average a probability of 1 with one other model's probabilities and the result is at least one half for the tree's class, which no other class can beat unless the other model is just as certain. On all {st['disagree']} digits where the SVM and the tree disagreed, the average sided with the tree. In larger averages the tree's certain class is guaranteed only a share of one over the number of members, and confident agreement among the others can outweigh it, so the averages recover, but only one of the {len(with_tree)} combinations containing the tree beat its best member: the tree with naive Bayes, the two weakest models of the six. The average weights each member by its confidence, and the tree's confidence is unconditional. Naive Bayes, certain about {mem['naive Bayes']['certain']:.0%} of digits, does the same to a lesser degree: averaged with the SVM it won {sn['with_b']:.0%} of their disagreements, and the pair scored {sn['acc']:.3f}.

Calibrating the tree first removes the certainty: an isotonic recalibration by internal cross-validation leaves it certain about no digit, and the SVM-plus-tree average rises to {sc['acc']:.3f}. That is far better, and still below the SVM alone ({mem['SVM']['acc']:.3f}): the tree is a weaker model, and averaging with a weaker model is a cost unless it is wrong on different digits from the stronger one.""",
        figures=[figs["f2"]])

    post.add(
        "What is guaranteed",
        f"""There is a real theorem behind the sentence, and it says less. For squared error, Krogh and Vedelsby showed in 1995 that the error of an equally weighted average is exactly the members' mean error minus their spread around the average, which they called the ambiguity. Since the spread cannot be negative, the average is never worse than the average member, and it improves on the average member by as much as the members disagree. On the diabetes data, with four regressors:

{snip['amb'].markdown()}

The identity holds to twelve decimal places. The average improves on the mean member by **{a['ambiguity']:.0f}**, all of it the members' disagreement. And it is still worse than ridge regression alone, {a['members']['ridge']:.0f} against {a['ensemble']:.0f}: the single tree's error was so large that the mean member started far behind the best one, and the spread did not close the gap.

The same kind of guarantee holds for log loss, because log loss is convex in the probabilities: the averaged probabilities' log loss was at or below the members' mean in all 57 combinations. It was below the best member's in {n_nll_best}. Neither guarantee mentions the best member, and accuracy, which is not convex, has no such guarantee at all; that is how {n_below} combinations fell below even their members' mean.""",
        figures=[figs["f1"]])

    post.add(
        "When averaging pays",
        f"""The combinations that did beat their best member share two properties: their members are of similar quality, and they are wrong on different digits. The SVM and nearest neighbours, {mem['SVM']['acc']:.3f} and {mem['5-NN']['acc']:.3f} alone, disagreed on {sk['disagree']} digits across the five splits and averaged to {sk['acc']:.4f}, better than either. Dietterich's account of why ensembles work rests on exactly these two conditions, accuracy and diversity, and random forests are built to meet them: many trees of equal quality, made different on purpose.

So the useful version of the sentence is conditional. An average of models that are about equally good and differently wrong is usually better than each of them; an average that includes a much weaker or a much more confident model is usually worse than the best. Calibrating the members first stops confidence from deciding the vote, and weighting them by validated performance stops a weak model from diluting a strong one, which is where episode 42 picks up.""")

    post.add(
        "Where this breaks",
        """**One data set where a single model is excellent.** Nearest neighbours alone scores 0.98 on the digits, which leaves little room for any combination to gain. Harder tasks with more balanced members may leave more room; that was not measured here.

**Unweighted averages.** Every member counted equally. Weighted averages and stacking are the obvious refinements, and their own failure modes are the next episode.

**A deliberately bad member.** A fully grown single tree is not something anyone would put in an ensemble on purpose; it is here because its certainty makes the mechanism visible. Any overconfident member, such as an unregularised boosted model late in training, does the same in a milder form.

**Five splits.** The counts of combinations above and below their best member come from one set of five half splits and could shift with others; the tree's capture of every pair it joined with a less certain model follows from its probabilities and would not.""")

    post.add(
        "What to keep",
        f"""1. Of 57 averaged combinations of six classifiers on the digits, {n_best} beat their best member and {n_below} fell below their members' mean accuracy.

2. Averaging probabilities weights each model by its confidence. A tree certain about every digit won all {st['disagree']} disagreements with the SVM, the pair scored {mem['tree']['acc']:.3f}, and only one of the {len(with_tree)} combinations containing the tree, the one pairing it with naive Bayes, beat its best member.

3. The guarantee is against the mean member: for squared error, the average's error is the mean member's minus their spread, exactly; on diabetes that still left it behind ridge regression alone.

4. Average models of similar quality that make different mistakes, calibrate them first, and compare the average with the best single model before adopting it.""")

    post.add(
        "Exercise",
        """Take an ensemble you use and score each member alone on held-out data, alongside the ensemble. Then compute, for every pair of members, the share of rows on which they disagree, and the share of predictions each makes with probability above 0.99.

If the ensemble is not better than its best member, look for a member that is much weaker or much more confident than the rest, and rescore the ensemble without it.""")

    post.add(
        "Next",
        """The natural fix for a bad member is to let the data choose the weights. Episode 42 fits those weights with a second model, stacking, and measures what it learns from in-sample predictions, from out-of-fold ones, and how often it beats picking the best member.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
