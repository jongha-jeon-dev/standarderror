"""ML 11: Resample the Rows and the Tree Tells a Different Story.

The eleventh episode of *Machine Learning, Taught Through What Breaks*, opening
Arc III. "A decision tree is interpretable: you can read off what it learned."
Measured on 200 bootstrap resamples of a breast-cancer training set:

* The tree's accuracy is stable: 0.914 on average, standard deviation 0.015.
* Its root split is not: the full-data tree's root feature is the root in
  only half the resamples, and four other features take turns. Their best
  single splits are within 5% of each other in impurity reduction.
* The top of the tree -- the features in its first three levels -- comes
  out as 194 different sets in 200 resamples, and not one of them is the
  full-data tree's set.
* And the trees disagree on the diagnosis of 36% of test tumours.

Run: `standarderror run ml111_tree_stability --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import trees as tr
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 12)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"boot": tr.tree_bootstrap()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    b = res["boot"]
    roots = b["roots"]
    gains = {c["feature"]: c["relative"] for c in b["candidates"]}

    def rootbar(ax, m):
        names = [r[0] for r in roots][::-1]
        vals = [r[1] / b["boots"] for r in roots][::-1]
        cols = [m.series[0] if n == b["full_root"] else m.series[1]
                for n in names]
        ax.barh(range(len(names)), vals, color=cols, height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([f"{n}  ({gains[n]:.0%} of best gain)"
                            if n in gains else n for n in names])
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9,
                    color=m.ink)
        ax.set_xlim(0, 0.62)

    out["f0"] = charts.diagram(
        rootbar,
        title="Five features take turns at the root",
        subtitle=(f"Root split of a depth-3 tree refitted to "
                  f"{b['boots']} bootstrap resamples of the training rows. "
                  f"In brackets, each feature's best single split on the "
                  f"full training set, as a share of the winner's."),
        xlabel="share of resamples in which it is the root",
        ylabel="",
        figsize=(7.6, 4.2),
        source="Breast Cancer Wisconsin (scikit-learn); "
               "standarderror/ml/trees.py.",
        alt=("Horizontal bars: worst perimeter is the root in half the "
             "resamples, worst concave points in about 30%, and three other "
             "size and shape features in the rest."),
        caption=(f"The full-data tree splits on {b['full_root']} first. A "
                 f"resample agrees **{b['root_same']:.0%}** of the time. The "
                 f"five contenders' best splits are within "
                 f"{1 - min(gains.values()):.0%} of one another, so which "
                 f"one wins is decided by which rows were drawn."),
        path=str(IMG / f"ml111-f0-roots.{EXT}"))[0]

    def rows(ax, m):
        share = b["row_share"]
        order = np.argsort(share)
        x = np.arange(len(share))
        unanimous = (share == 0) | (share == 1)
        ax.scatter(x, share[order], s=12,
                   c=[m.series[1] if not u else m.grid
                      for u in unanimous[order]])
        ax.axhline(0.5, lw=1, ls="--", color=m.ink)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-3, len(x) + 2)

    out["f1"] = charts.diagram(
        rows,
        title="One tumour, two hundred trees, several answers",
        subtitle=(f"For each of {b['test_rows']} test tumours, the share of "
                  f"the {b['boots']} bootstrap trees that call it benign, "
                  f"sorted. Grey: all trees agree. Coloured: at least one "
                  f"disagrees."),
        xlabel="test tumours, sorted",
        ylabel="share of trees saying benign",
        source="Breast Cancer Wisconsin (scikit-learn); "
               "standarderror/ml/trees.py.",
        alt=("Dots sorted from 0 to 1: grey runs at each end where all trees "
             "agree and a coloured middle third where they do not."),
        caption=(f"**{b['any_disagree']:.0%}** of test tumours get a "
                 f"different diagnosis from at least one of the trees; any "
                 f"two trees disagree on {b['pair_disagreement']:.1%} of "
                 f"them. Each tree, read on its own, explains its answer "
                 f"with confidence."),
        path=str(IMG / f"ml111-f1-rows.{EXT}"))[0]

    acc = np.array(b["accuracy"])
    rows_t = [
        ["test accuracy", f"{acc.mean():.3f} ± {acc.std():.3f}",
         "stable"],
        ["root feature same as full-data tree", f"{b['root_same']:.0%}",
         "a coin flip"],
        ["root threshold, when the feature matches (5th-95th pct)",
         f"{b['threshold'][0]:.0f} to {b['threshold'][2]:.0f}",
         "moves 10%"],
        ["distinct feature sets in the top three levels",
         f"{b['distinct_sets']} of {b['boots']}", "almost never repeats"],
        ["test rows where some tree disagrees", f"{b['any_disagree']:.0%}",
         "a third"],
        ["disagreement between two random trees",
         f"{b['pair_disagreement']:.1%}", "about the error rate"],
    ]
    out["f2"] = charts.table_image(
        rows_t,
        header=["what was measured", "across 200 resamples", ""],
        title="Stable score, unstable story",
        subtitle=("Depth-3 classification trees on bootstrap resamples of "
                  "398 training tumours, scored on the same 171 held out."),
        source="Breast Cancer Wisconsin (scikit-learn); "
               "standarderror/ml/trees.py.",
        alt=("A six-row table contrasting a stable test accuracy with an "
             "unstable root feature, threshold, top-level feature set and "
             "per-row prediction."),
        caption=("Everything a reader would read off the tree moves; the "
                 "number a model selection would look at does not."),
        align="lrl",
        path=str(IMG / f"ml111-f2-table.{EXT}"))[0]

    out["hero"] = _hero(b, acc)
    return out


def _hero(b: dict, acc):

    def root(panel, m):
        panel.bar(range(len(b["roots"])),
                  [r[1] for r in b["roots"]],
                  color=[m.series[0]] + [m.series[1]] * (len(b["roots"]) - 1))

    def flat(panel, m):
        panel.hist(acc, bins=12, color=m.series[2])
        panel.set_xlim(0.8, 1.0)

    def rows(panel, m):
        s = np.sort(b["row_share"])
        panel.plot(range(len(s)), s, lw=2.8, color=m.series[1])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=11,
        headline="Same accuracy, a different tree every time",
        panels=[(root, f"{b['root_same']:.0%}", "same root as full fit"),
                (flat, f"± {acc.std():.3f}", "spread in accuracy"),
                (rows, f"{b['any_disagree']:.0%}", "test rows in dispute")],
        note=("Refit a small decision tree on bootstrap resamples of the "
              "same breast-cancer data. Its accuracy barely moves; its root "
              "split, the features near the top and the diagnosis of a "
              "third of the test tumours do."),
        alt=("Three hand-drawn frames: one tall bar among shorter ones; a "
             "narrow histogram; a sorted S-shaped curve."),
        path=str(IMG / f"ml111-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["boot"] = s.run("""
        from standarderror.ml import trees as tr

        b = tr.tree_bootstrap(boots=200, depth=3)
        print(f"full-data root: {b['full_root']}")
        for name, k in b["roots"]:
            print(f"  root in {k:3d} of 200 resamples: {name}")
        acc = b["accuracy"]
        print(f"test accuracy {sum(acc) / len(acc):.3f}, sd {b['accuracy_sd']:.3f}")
        """)
    out["cands"] = s.run("""
        for c in b["candidates"]:
            print(f"{c['feature']:22s} best split gain {c['gain']:.4f}"
                  f"  ({c['relative']:.1%} of the best)")
        """)
    out["rows"] = s.run("""
        print(f"distinct top-3-level feature sets: {b['distinct_sets']} of 200")
        print(f"same set as the full-data tree: {b['full_set_same']} of 200")
        print(f"test rows where some tree disagrees: {b['any_disagree']:.1%}")
        print(f"two trees disagree on, averaged over pairs: "
              f"{b['pair_disagreement']:.1%}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    b = res["boot"]
    acc = np.array(b["accuracy"])
    gains = [c["relative"] for c in b["candidates"]]
    err = 1 - acc.mean()
    second = b["roots"][1]
    cut = b["full_threshold"]
    worst_roots = [n for n, _ in b["roots"] if n.startswith("worst ")]
    mean_roots = [n for n, _ in b["roots"] if n.startswith("mean ")]

    # How closely the contenders track one another, on all 569 tumours.
    from sklearn.datasets import load_breast_cancer
    bcd = load_breast_cancer()
    fn = [str(n) for n in bcd.feature_names]

    def corr(a: str, c: str) -> float:
        return float(np.corrcoef(bcd.data[:, fn.index(a)],
                                 bcd.data[:, fn.index(c)])[0, 1])

    r_size = min(corr("worst perimeter", "worst radius"),
                 corr("worst perimeter", "worst area"))
    r_conc = corr("worst perimeter", "worst concave points")

    # The spine, asserted rather than trusted.
    assert len(worst_roots) == 4 and mean_roots == ["mean concave points"]
    assert b["full_set_same"] == 0 and b["modal_set_count"] <= 3
    assert b["grown_leaves"] < 30 and b["grown_leaves"] > 2 ** b["depth"]
    assert r_size > 0.95 and 0.7 < r_conc < r_size
    assert acc.std() < 0.02
    assert b["root_same"] <= 0.55 and len(b["roots"]) >= 4
    assert min(gains) > 0.94
    assert b["distinct_sets"] > 0.9 * b["boots"]
    assert b["any_disagree"] > 0.3
    assert 0.5 * err < b["pair_disagreement"] < 1.5 * err
    assert sum(k for _, k in b["roots"]) == b["boots"] and len(b["roots"]) == 5
    assert all(("perimeter" in n or "radius" in n or "area" in n
                or "concave points" in n) for n, _ in b["roots"])

    post = Post(
        title=(f"{SERIES_TAG} 11: Resample the Rows and the Tree Tells a "
               f"Different Story"),
        slug="ml-11-tree-stability",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=11,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A decision tree is valued because it can be read. Refit "
                  "on bootstrap resamples of the same data, its accuracy "
                  "holds still and almost everything a reader would read "
                  "off it moves."),
        summary=(
            f"Two hundred depth-3 trees fitted to bootstrap resamples of a "
            f"breast-cancer training set score {acc.mean():.3f} on the same "
            f"held-out tumours, with a standard deviation of {acc.std():.3f}. "
            f"Their structure is another matter. The full-data tree's root "
            f"feature, {b['full_root']}, is the root in "
            f"{b['root_same']:.0%} of resamples; four other features take "
            f"the rest, and all five have best splits within "
            f"{1 - min(gains):.0%} of each other. The {b['boots']} resamples "
            f"put {b['distinct_sets']} different sets of features in the top "
            f"three levels, none of them the full-data tree's (episode 66 "
            f"found 3 of 200 in another split), and "
            f"{b['any_disagree']:.0%} of test tumours get a different "
            f"diagnosis from at least one tree."),
        tags=["machine-learning", "data-science", "statistics",
              "decision-trees", "interpretability", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Breast Cancer Wisconsin (Diagnostic), as bundled with "
            "scikit-learn, CC BY 4.0; one stratified 70/30 split; 200 "
            "bootstrap resamples of the 398 training rows.",
            "Machinery: `standarderror/ml/trees.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's DecisionTreeClassifier with "
            "max_depth=3.",
            "Where this stops: Breiman, \"Bagging predictors\", *Machine "
            "Learning* (1996), on unstable procedures; Breiman, \"Statistical "
            "modeling: the two cultures\", *Statistical Science* (2001), for "
            "the Rashomon effect; Hothorn, Hornik and Zeileis, \"Unbiased "
            "recursive partitioning\", *JCGS* (2006), for split selection "
            "with a significance test.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("split seed 0; bootstrap resamples from one "
                            "generator with seed 0; trees with "
                            "random_state 0"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The case for trees",
        f"""Arc II pushed linear models past their textbook conditions. Arc III does the same for trees and the ensembles built from them, and starts with the reason a single tree is chosen at all: it can be read. A depth-3 tree is at most eight rules a clinician could check by hand — *if the largest nuclei have a perimeter above about {cut:.0f}, call it suspicious and look further down the tree* — and that readability is often the whole argument for using one instead of something more accurate.

Reading a tree as an account of the data assumes the tree is a property of the data. A different sample from the same population should give roughly the same tree. That is checkable: draw bootstrap resamples of the training rows, refit, and compare.

The data are the Wisconsin breast-cancer measurements: 398 training tumours, 171 held out, 30 features describing cell nuclei. The tree is depth 3, the size people actually draw on a slide.

{snip['boot'].markdown()}

The accuracy is what a model-selection report would show, and it is steady: {acc.mean():.3f}, give or take {acc.std():.3f}. The root is not. The tree fitted to all 398 rows splits first on {b['full_root']}; a resample agrees **{b['root_same']:.0%}** of the time. {second[0].capitalize()} wins {second[1]} times, and three more features take the rest.""",
        figures=[figs["f0"]])

    post.add(
        "Why the root moves",
        f"""A tree is grown greedily: at each node it tries every feature and threshold and keeps the single split that reduces impurity most. When several splits are nearly as good, the winner is decided by the details of the sample. Here is how close the contest at the root is on the full training set:

{snip['cands'].markdown()}

All five are within {1 - min(gains):.0%} of the best. That is not a coincidence of this dataset; they measure overlapping things. Perimeter, radius and area of the largest nuclei are almost deterministic functions of one another — across all 569 tumours worst perimeter correlates with worst radius and worst area at {r_size:.2f} or more — and concavity goes with size, less tightly: worst perimeter and worst concave points correlate at {r_conc:.2f}. With contenders that close, a resample that shifts a handful of borderline tumours can plausibly change which one edges ahead.

The instability then propagates. Every split below the root is chosen on the rows the root sent down, so a different root means different subsets and a different contest at every node beneath. The threshold moves too: even when the root feature matches, its cut point ranges from {b['threshold'][0]:.0f} to {b['threshold'][2]:.0f} between the 5th and 95th percentiles of resamples.

{snip['rows'].markdown()}

The {b['boots']} resamples put {b['distinct_sets']} different sets of features in the first three levels. The most common set turns up only {b['modal_set_count']} times, and the full-data tree's own set — the one a reader would be shown — turns up {b['full_set_same']} times. A reader who took any one of these trees as *the* structure of the data would be reading one draw from a wide distribution.""",
        figures=[figs["f1"]])

    post.add(
        "Different rules, different diagnoses",
        f"""Structure could move while predictions stayed put; two trees can encode the same decision with different rules. They do not. Two of these trees disagree on {b['pair_disagreement']:.1%} of test tumours on average (over every pair among the first sixty), about the size of their own error rate ({err:.1%}), and across all two hundred, **{b['any_disagree']:.0%} of tumours get a different diagnosis from at least one tree**.

Those tumours are not random. They are the ones near the boundary, where several rules of similar quality draw the line in slightly different places. That is also where a diagnosis matters most, and where a tree's printed path — "malignant because worst perimeter > {cut:.0f}" — is least an account of the data and most an account of the sample.

The accuracy hides all this because it is an average over tumours, most of which are easy. It is the same lesson as episode 1's error bar, one level down: the number is stable because it averages over exactly the cases where the model is not.""",
        figures=[figs["f2"]])

    post.add(
        "What does survive",
        f"""Not everything moves. Every one of the {b['boots']} roots is one of {len(b['roots'])} features, and all of them measure either the size or the concavity of the nuclei: {len(worst_roots)} describe the largest, most irregular nuclei, and the fifth, {mean_roots[0]}, the concavity averaged over all of them. A statement at that level — *the first thing that separates these tumours is how large and how concave their nuclei are* — holds in every resample. A statement at the level of the printed tree — *worst perimeter, cut at {cut:.0f}* — holds in about half as far as the feature goes, and less often to the unit, since the cut itself ranges from {b['threshold'][0]:.0f} to {b['threshold'][2]:.0f}.

That suggests how to present a tree honestly. Report the set of features that compete for each of the top splits and how often each wins, not the single winner; quote thresholds as ranges across resamples; and say which statements about the data are stable and which are a property of one fit. It is more work than printing the tree, and it is what the tree's readability was supposed to buy in the first place: a claim about the data that a reader can check.

It also says what not to do. A tree is not evidence that its root feature is *the* most important variable, and a feature absent from the tree is not evidence that it is irrelevant: in half these resamples {b['full_root']} is not the root, and in {1 - b['root_in_top']:.0%} it does not appear anywhere in the top three levels.""")

    post.add(
        "Where this breaks",
        f"""**Deeper trees are unlikely to be better.** Depth 3 is the readable case. Grown without a depth limit, a tree on these 398 rows has {b['grown_leaves']} leaves and depth {b['grown_depth']}. Those were not refitted on the resamples here, but every split below the top is chosen on the rows the splits above it selected, so they can be expected to inherit the instability above them.

**Some data have one dominant split.** When one feature separates the classes far better than any other, the root is stable and the instability starts further down. The size of the gap between the best and second-best split at the root is the thing to check, and on data with many correlated measurements it is usually small.

**Bagging is the fix for prediction, not for reading.** Averaging many such trees should give predictions that move far less between resamples than any one tree's — Breiman's point in 1996, not measured here. It also gives up the single readable tree. Episodes 12 and 16 measure what is left of the reading once trees are averaged.

**Unbiased split selection is unlikely to settle the contest.** Conditional inference trees choose the root by a significance test instead of a raw impurity decrease. That corrects a bias episode 12 measures, but with five candidates within {1 - min(gains):.0%} of one another the resample would still be expected to decide; conditional inference trees were not fitted here.""")

    post.add(
        "What to keep",
        f"""1. Two hundred bootstrap refits of a depth-3 tree scored {acc.mean():.3f} ± {acc.std():.3f}: the accuracy is a property of the data.

2. The root feature matched the full-data tree's in {b['root_same']:.0%} of them, and the top three levels came out as {b['distinct_sets']} different feature sets in {b['boots']} resamples, none of them the full-data tree's: the structure is a property of the sample.

3. The root moves because the candidates are close. Here the best five splits were within {1 - min(gains):.0%} of one another.

4. {b['any_disagree']:.0%} of test tumours got a different diagnosis from at least one tree. Instability in the rules is instability in the predictions near the boundary.

5. Before reading a tree, refit it on a few dozen bootstrap resamples and report how often its root and its top-level features recur.""")

    post.add(
        "Exercise",
        """Take a tree you have shown to someone as an explanation. Refit it on 50 bootstrap resamples of its training data and count how often its root feature, its root threshold (to the precision you quoted) and its second-level features reappear. Then compute, for your held-out rows, the share of resampled trees that agree with the original tree's prediction.

If the root survives in fewer than four resamples out of five, say so when you present the tree — or present the set of competing roots instead of one.""")

    post.add(
        "Next",
        """Episode 12 takes the forest that averaging produces and asks the question people put to it in place of reading a tree: which features does it rely on? The default answer, impurity importance, is measured on data where the right answer is known to be nothing at all.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
