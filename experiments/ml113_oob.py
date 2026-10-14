"""ML 13: Out-of-Bag Error Is Free, Pessimistic When Small and Blind to Copies.

The thirteenth episode of *Machine Learning, Taught Through What Breaks*.
"Out-of-bag error is free cross-validation." Measured against 5-fold
cross-validation and a held-out half on breast cancer and digits:

* With 500 trees it is as good as cross-validation and costs one fit
  instead of six: within half a point of the held-out error on both
  datasets.
* With few trees it is badly pessimistic: at 5 trees, 2.7 and 2.8 times the
  held-out error. Each row's OOB prediction comes from the roughly 37% of
  trees that did not see it -- a smaller forest than the one being
  evaluated -- and about a tenth of rows get no vote at all, which
  scikit-learn scores as votes for the first class. Those rows supply
  almost half of the breast-cancer figure and a third of the digits one.
* It assumes the rows are independent. With every digit image duplicated in
  the training data, OOB error is exactly 0 while the held-out error is
  2.7%.

Run: `standarderror run ml113_oob --publish`
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
    return {"bc": tr.oob_study(data="breast cancer"),
            "dg": tr.oob_study(data="digits", reps=5),
            "nv_bc": tr.oob_novote(data="breast cancer", trees=5, reps=10),
            "nv_dg": tr.oob_novote(data="digits", trees=5, reps=5),
            "dup": tr.oob_duplicates()}


def _novote(T: int) -> float:
    """Share of rows in bag for every one of T trees: (1 - e^-1)^T."""
    return (1 - np.exp(-1)) ** T


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}

    def curves(ax, m):
        for c, key, lab in ((m.series[0], "bc", "breast cancer"),
                            (m.series[1], "dg", "digits")):
            rows = res[key]
            T = [r["trees"] for r in rows]
            ax.plot(T, [r["oob"] for r in rows], lw=2.4, marker="o", ms=5,
                    color=c, label=f"{lab}: out-of-bag")
            ax.plot(T, [r["cv"] for r in rows], lw=1.8, marker="s", ms=4,
                    ls=":", color=c, label=f"{lab}: 5-fold CV")
            ax.plot(T, [r["test"] for r in rows], lw=1.8, marker="^", ms=4,
                    ls="--", color=c, label=f"{lab}: held-out half")
        ax.set_xscale("log")
        ticks = [r["trees"] for r in res["bc"]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(0, 0.32)
        ax.legend(frameon=False, fontsize=8.0, loc="upper right", ncol=2)

    b5, d5 = res["bc"][0], res["dg"][0]
    bl, dl = res["bc"][-1], res["dg"][-1]
    out["f0"] = charts.diagram(
        curves,
        title="Out-of-bag error converges from above",
        subtitle=("Random-forest error estimated three ways, against the "
                  "number of trees. Breast cancer: 10 half splits; digits: "
                  "5. Cross-validation and out-of-bag use only the training "
                  "half."),
        xlabel="trees in the forest (log scale)",
        ylabel="estimated error",
        source="Breast Cancer Wisconsin and UCI digits (scikit-learn); "
               "standarderror/ml/trees.py.",
        alt=("Solid out-of-bag lines start far above the dashed held-out "
             "and dotted cross-validation lines at 5 trees, close in on them "
             "by 100 trees and meet them by 500."),
        caption=(f"At 5 trees the out-of-bag error is "
                 f"**{b5['oob'] / b5['test']:.1f}x** the held-out error on "
                 f"breast cancer and {d5['oob'] / d5['test']:.1f}x on digits. "
                 f"At 500 it is within {abs(bl['oob'] - bl['test']):.3f} and "
                 f"{abs(dl['oob'] - dl['test']):.3f}."),
        path=str(IMG / f"ml113-f0-curves.{EXT}"))[0]

    d = res["dup"]

    def dup(ax, m):
        labels = ["independent rows", "every row duplicated"]
        x = np.arange(2)
        w = 0.36
        ax.bar(x - w / 2, [d["oob"], d["oob_dup"]], w, color=m.series[0],
               label="out-of-bag error")
        ax.bar(x + w / 2, [d["test"], d["test_dup"]], w, color=m.series[2],
               label="held-out error")
        for xi, v in zip(x - w / 2, [d["oob"], d["oob_dup"]]):
            ax.text(xi, v + 0.0008, f"{v:.3f}", ha="center", fontsize=9,
                    color=m.ink)
        for xi, v in zip(x + w / 2, [d["test"], d["test_dup"]]):
            ax.text(xi, v + 0.0008, f"{v:.3f}", ha="center", fontsize=9,
                    color=m.ink)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 0.04)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    out["f1"] = charts.diagram(
        dup,
        title="Duplicate the rows and out-of-bag error vanishes",
        subtitle=(f"Digits, {d['trees']}-tree forests, {d['reps']} half "
                  f"splits. On the right every training image appears twice; "
                  f"the held-out half has no duplicates."),
        xlabel="",
        ylabel="error",
        source="UCI digits (scikit-learn); standarderror/ml/trees.py.",
        alt=("Paired bars: with independent rows both errors are near "
             "0.03; with duplicated rows the out-of-bag bar is zero and the "
             "held-out bar is unchanged."),
        caption=(f"Whenever a row is out of bag its copy is usually in it, "
                 f"so the forest is scored on what it memorised: OOB error "
                 f"**{d['oob_dup']:.3f}**, held-out {d['test_dup']:.3f}. The "
                 f"model did not get better; the estimate broke."),
        path=str(IMG / f"ml113-f1-dup.{EXT}"))[0]

    rows = []
    for key, lab in (("bc", "breast cancer"), ("dg", "digits")):
        for r in res[key]:
            rows.append([lab if r["trees"] == 5 else "", f"{r['trees']}",
                         f"{r['oob']:.3f}", f"{r['cv']:.3f}",
                         f"{r['test']:.3f}", f"{_novote(r['trees']):.1%}"])
    out["f2"] = charts.table_image(
        rows,
        header=["data", "trees", "out-of-bag", "5-fold CV", "held-out",
                "rows with no OOB vote"],
        title="Three estimates of one error",
        subtitle=("Means over splits. The last column is (1 - 1/e)^T, the "
                  "expected share of training rows that every tree saw."),
        source="Breast Cancer Wisconsin and UCI digits (scikit-learn); "
               "standarderror/ml/trees.py.",
        alt=("A ten-row table: out-of-bag error far above the others at 5 "
             "and 10 trees and close to them from 100 trees on."),
        caption=("At 500 trees, out-of-bag and five-fold cross-validation are "
                 "both within about half a point of the held-out error on "
                 "both datasets; at 100 trees that holds on breast cancer, "
                 "while on digits both are still about a point high. "
                 "Out-of-bag costs a sixth of the fitting."),
        align="lrrrrr",
        path=str(IMG / f"ml113-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    b = res["bc"]
    d = res["dup"]

    def conv(panel, m):
        panel.plot(range(len(b)), [r["oob"] for r in b], lw=2.8,
                   color=m.series[0])
        panel.plot(range(len(b)), [r["test"] for r in b], lw=2.0, ls="--",
                   color=m.ink)

    def big(panel, m):
        panel.bar([0, 1], [b[-1]["oob"], b[-1]["test"]],
                  color=[m.series[0], m.series[2]])

    def dup(panel, m):
        panel.bar([0, 1], [d["oob_dup"] + 0.0005, d["test_dup"]],
                  color=[m.series[0], m.series[2]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=13,
        headline="Free, honest at scale, and blind to copies",
        panels=[(conv, f"{b[0]['oob'] / b[0]['test']:.1f}x",
                 "OOB vs held-out, 5 trees"),
                (big, f"{b[-1]['oob']:.3f}", "OOB at 500 trees"),
                (dup, f"{d['oob_dup']:.3f}", "OOB with duplicate rows")],
        note=("A forest's out-of-bag error needs no held-out data. With "
              "hundreds of trees it matches a held-out set as well as "
              "cross-validation does. With a handful it overstates the error "
              "almost threefold, and when rows are copies of one another it "
              "reports zero."),
        alt=("Three hand-drawn frames: a solid curve falling onto a dashed "
             "one; two bars of equal height; one empty bar beside a short "
             "one."),
        path=str(IMG / f"ml113-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["bc"] = s.run("""
        from standarderror.ml import trees as tr

        # breast cancer, 10 half splits; OOB and CV use the training half only
        for r in tr.oob_study(data="breast cancer", trees=(5, 10, 25, 100, 500)):
            print(f"{r['trees']:>3} trees  out-of-bag {r['oob']:.3f}"
                  f"  5-fold CV {r['cv']:.3f}  held-out {r['test']:.3f}")
        """)
    out["share"] = s.run("""
        # share of all trees for which a given row is out of bag
        print(" ".join(f"{T} trees: {tr.oob_coverage(T):.3f}" for T in (5, 100)))
        """)
    out["novote"] = s.run("""
        # 5 trees: how much OOB error comes from rows that got no OOB vote
        for data, reps in (("breast cancer", 10), ("digits", 5)):
            v = tr.oob_novote(data=data, trees=5, reps=reps)
            print(f"{data:13s} OOB {v['oob']:.3f}  no-vote rows "
                  f"{v['from_novote']:.3f} ({v['novote_share']:.0%} of rows)"
                  f"  voted rows' error {v['voted_error']:.3f}")
        """)
    out["dup"] = s.run("""
        d = tr.oob_duplicates()     # digits; each training image twice
        print(f"independent rows: OOB {d['oob']:.3f}, held-out {d['test']:.3f}")
        print(f"duplicated rows:  OOB {d['oob_dup']:.3f}, "
              f"held-out {d['test_dup']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    bc, dg, d = res["bc"], res["dg"], res["dup"]
    b5, d5, bl, dl = bc[0], dg[0], bc[-1], dg[-1]
    b100, d100 = bc[3], dg[3]
    nb, nd = res["nv_bc"], res["nv_dg"]
    fb, fd = nb["from_novote"] / nb["oob"], nd["from_novote"] / nd["oob"]

    # The spine, asserted rather than trusted.
    assert b100["trees"] == 100 and d100["trees"] == 100
    assert abs(b100["oob"] - b100["test"]) < 0.005
    assert abs(b100["cv"] - b100["test"]) < 0.005
    assert 0.008 < d100["oob"] - d100["test"] < 0.015
    assert 0.008 < d100["cv"] - d100["test"] < 0.015
    # the no-vote split reproduces oob_study's 5-tree numbers exactly
    assert abs(nb["oob"] - b5["oob"]) < 1e-12 and abs(nd["oob"] - d5["oob"]) < 1e-12
    assert 0.4 < fb < 0.5 and 0.3 < fd < 0.4
    for v, r in ((nb, b5), (nd, d5)):
        assert r["test"] < v["voted_error"] < v["oob"]
        assert 0.08 < v["novote_share"] < 0.12
    assert nb["voted_error"] < 1.8 * b5["test"]
    assert _novote(25) < 1e-3
    for rows in (bc, dg):
        assert rows[0]["oob"] > 2.2 * rows[0]["test"]
        assert rows[0]["oob"] > rows[0]["cv"] > rows[0]["test"]
        assert all(a["oob"] > b["oob"] for a, b in zip(rows, rows[1:]))
        big = rows[-1]
        assert abs(big["oob"] - big["test"]) < 0.006
        assert abs(big["cv"] - big["test"]) < 0.007
    assert abs(dl["oob"] - dl["test"]) < abs(dl["cv"] - dl["test"])
    assert d["oob_dup"] == 0.0 and d["test_dup"] > 0.02
    assert abs(d["test_dup"] - d["test"]) < 0.005

    post = Post(
        title=(f"{SERIES_TAG} 13: Out-of-Bag Error Is Free, Pessimistic When "
               f"Small and Blind to Copies"),
        slug="ml-13-oob",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=13,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A random forest estimates its own test error from the "
                  "rows each tree did not see. With enough trees that is as "
                  "good as cross-validation, for a sixth of the work. With "
                  "few trees, or rows that are not independent, it is not."),
        summary=(
            f"Out-of-bag error against five-fold cross-validation and a "
            f"held-out half. With 500 trees it lands within "
            f"{abs(bl['oob'] - bl['test']):.3f} of the held-out error on "
            f"breast cancer and {abs(dl['oob'] - dl['test']):.3f} on digits, "
            f"about as close as cross-validation, from a single fit. With 5 trees "
            f"it is {b5['oob'] / b5['test']:.1f} and "
            f"{d5['oob'] / d5['test']:.1f} times the held-out error, for two "
            f"reasons: each row is judged by the third of the forest that did "
            f"not see it, and the {nb['novote_share']:.0%} of rows that every "
            f"tree saw get no vote, which scikit-learn scores as a vote for "
            f"the first class. Those rows supply {fb:.0%} of the breast-cancer "
            f"figure. With every digit image duplicated in the training data, "
            f"out-of-bag error is exactly {d['oob_dup']:.0f} while the "
            f"held-out error is {d['test_dup']:.3f}."),
        tags=["machine-learning", "data-science", "statistics",
              "random-forest", "cross-validation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Breast Cancer Wisconsin (Diagnostic) and UCI Optical "
            "Recognition of Handwritten Digits, as bundled with "
            "scikit-learn, CC BY 4.0; stratified half splits, 10 for breast "
            "cancer and 5 for digits.",
            "Machinery: `standarderror/ml/trees.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's RandomForestClassifier with "
            "oob_score=True and default settings otherwise.",
            "Where this stops: Breiman, \"Out-of-bag estimation\", technical "
            "report (1996); Breiman, \"Random forests\", *Machine Learning* "
            "(2001); Janitza and Hornung, \"On the overestimation of random "
            "forest's out-of-bag error\", *PLOS ONE* (2018).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("split seeds 0 to 9 (digits 0 to 4); forests and "
                            "fold assignments seeded with the split seed"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "An error estimate in the box",
        f"""A random forest grows each tree on a bootstrap sample of the training rows. A bootstrap sample of n rows leaves out each row with probability (1 − 1/n)ⁿ, about 1/e or 37%. So every training row has a set of trees that never saw it, and those trees can predict it as if it were new. Collect those predictions for every row and you have the *out-of-bag* (OOB) error: an estimate of test error that needs no held-out data and no refitting. Breiman called it free, and in scikit-learn it is one argument.

"Free cross-validation" is a claim that can be checked directly. Here are the OOB error, five-fold cross-validation on the same training half, and the error on the untouched other half, for forests of increasing size:

{snip['bc'].markdown()}

At 500 trees the claim holds: OOB {bl['oob']:.3f}, cross-validation {bl['cv']:.3f}, held-out {bl['test']:.3f}. Both estimates are within half a point of the held-out number — cross-validation a little closer here, out-of-bag closer on digits ({dl['oob']:.3f} and {dl['cv']:.3f} against {dl['test']:.3f}) — and cross-validation took five extra forests to get there.

At 5 trees it fails badly: **OOB {b5['oob']:.3f} against a held-out {b5['test']:.3f}**, {b5['oob'] / b5['test']:.1f} times too high. Cross-validation, at {b5['cv']:.3f}, is also pessimistic but by far less.""",
        figures=[figs["f0"]])

    post.add(
        "Why small forests look worse than they are",
        f"""The OOB prediction for a row is a vote among the trees that did not see it, not among all of them:

{snip['share'].markdown()}

A row's OOB verdict comes from about 37% of the forest. With 500 trees that is around 185, plenty for a stable vote, so the estimate describes a forest almost as good as the real one. With 5 trees it is one or two on average, and a vote of one or two trees can be expected to be worse than a forest of five. In that sense the OOB error at T trees describes a forest of roughly T/e trees, which matters exactly when the forest is small enough for the difference to matter.

Some rows are worse off still. The chance that a row was drawn into every one of T bootstrap samples is about (1 − 1/e)^T: {_novote(5):.0%} of rows at 5 trees, {_novote(10):.1%} at 10. Those rows have no OOB prediction at all, and scikit-learn does not drop them. It warns, gives them an all-zero vote, and the predicted class becomes the first one — malignant in the breast-cancer data, the digit 0 in the digits. Every such row from another class is counted as an error. How much of the 5-tree number that bookkeeping accounts for can be measured directly:

{snip['novote'].markdown()}

On breast cancer, {nb['from_novote']:.3f} of the {nb['oob']:.3f} — {fb:.0%} of the out-of-bag error — comes from rows that received no vote. Among the rows that did get one the error is {nb['voted_error']:.3f}, {nb['voted_error'] / b5['test']:.1f} times the held-out {b5['test']:.3f}. On digits the no-vote rows supply {fd:.0%} of the {nd['oob']:.3f}, and the voted rows still err at {nd['voted_error']:.3f}, {nd['voted_error'] / d5['test']:.1f} times the held-out {d5['test']:.3f}. So the 5-tree pessimism is part artefact and part a genuinely smaller voting forest. The artefact disappears once every row has a vote — by 25 trees the expected share without one is far below one row in a thousand — and the smaller-forest effect fades more slowly.

Pessimism is the safe direction for an error estimate, and it fades as trees are added: at 100 trees out-of-bag error is {b100['oob']:.3f} against a held-out {b100['test']:.3f} on breast cancer, still {d100['oob']:.3f} against {d100['test']:.3f} on digits, and by 500 it is gone on both. It still matters in two places: when OOB error is used to choose the number of trees, where it argues for more trees than are needed; and when it is used to compare a forest against a model scored by honest cross-validation, where a small forest is handicapped. Janitza and Hornung (2018) document a further, subtler pessimism in OOB error for small samples, which these runs were not designed to isolate.""",
        figures=[figs["f2"]])

    post.add(
        "When rows are not independent",
        f"""The OOB argument rests on one assumption: a tree that did not see a row has learned nothing specific to it. That is true when rows are independent draws. It is false when the training data contain copies — augmented images, repeated measurements of one patient, the same customer on many rows, a careless merge that duplicated a table.

Duplicate every digit image in the training half and refit:

{snip['dup'].markdown()}

Out-of-bag error is **exactly {d['oob_dup']:.0f}**. When one copy of an image is out of a tree's bootstrap sample, the other copy is in it about 63% of the time (1 − 1/e), so most of the trees voting on a row have seen its twin, and a fully grown tree that has seen a row's twin predicts that row correctly. The "unseen" row was seen. The held-out error is unchanged at {d['test_dup']:.3f}: the model is no better, only its own estimate of itself is broken.

This is the same failure episode 3 measured for cross-validation with duplicated rows, and it has the same remedy, with a cost: the rows must be grouped and the groups held out together. A forest's built-in OOB estimate has no notion of groups. With exact copies, as here, it says nothing at all; with looser groups — repeated visits, related images — it should be optimistic to a lesser degree, which was not measured here. Either way, grouped cross-validation, or a held-out set split by group, has to replace it.""",
        figures=[figs["f1"]])

    post.add(
        "Using it well",
        f"""None of this makes out-of-bag error less useful; it says when to use it. The cost argument is real: for a forest of a few hundred trees, OOB error comes with the fit, while five-fold cross-validation fits five more forests. On large data that is the difference between an estimate and no estimate.

Three habits keep it honest. Grow the forest first and read the OOB error last: the curve above settles by 100 trees on breast cancer and only by 500 on digits, and before it settles the number describes a weaker model than the one you have. Keep a held-out set anyway for the final, reported number, because the OOB error of the model you chose after looking at several OOB errors is no longer an independent estimate. And before trusting it at all, ask whether two rows of the training data can be the same thing in disguise. If they can, the bootstrap's notion of "unseen" is wrong, and so is every number built on it, as the duplicated digits show with an error of exactly {d['oob_dup']:.0f}.""")

    post.add(
        "Where this breaks",
        """**OOB is not free for tuning.** Each OOB error is an honest estimate for one forest. Choosing the best of many hyperparameter settings by their OOB errors reuses the same rows to select and to score, and the winner's OOB error is optimistic, as any validation score reused for selection is.

**Gradient boosting has no equivalent.** Its trees are fitted in sequence, each to the residuals of all the earlier ones, so no tree is independent of a row it did not sample. Subsampled boosting reports an "out-of-bag improvement", which is a different and much rougher quantity.

**Two datasets, one model setting.** Both datasets are easy and both forests use scikit-learn's defaults. The crossover to an honest estimate came between 100 and 500 trees here; it plausibly depends on how much a single tree's error exceeds the forest's.

**Other libraries treat the no-vote rows differently.** The artefact measured above is scikit-learn's: a library that drops rows without an OOB vote, or reports them separately, would show less pessimism at 5 trees, and the same smaller-forest effect.

**Duplicates are the extreme case.** Near-duplicates, such as augmented copies with small shifts or repeated visits by one patient, should leak less than exact copies but in the same direction; only exact copies were measured here.""")

    post.add(
        "What to keep",
        f"""1. With 500 trees, out-of-bag error was within {max(abs(bl['oob'] - bl['test']), abs(dl['oob'] - dl['test'])):.3f} of the held-out error on both datasets, about as close as five-fold cross-validation, for one fit instead of six.

2. With 5 trees it was {b5['oob'] / b5['test']:.1f} and {d5['oob'] / d5['test']:.1f} times the held-out error. Each row's OOB vote comes from about 37% of the forest, so a small forest is judged as an even smaller one; and scikit-learn scores the rows with no vote as the first class, which supplied {fb:.0%} and {fd:.0%} of those errors.

3. Use OOB error once the forest has a few hundred trees, and do not compare a small forest's OOB error with another model's cross-validated one.

4. With duplicated rows, OOB error was {d['oob_dup']:.0f} and the held-out error {d['test_dup']:.3f}. With exact copies OOB error is not an estimate of anything; on grouped data, expect it to be optimistic.

5. Selecting hyperparameters by OOB error makes the winner's OOB error optimistic, like any reused validation score.""")

    post.add(
        "Exercise",
        """Fit your forest with `oob_score=True` at 10, 50 and 500 trees and compare each OOB error with the error on a held-out set. Note where the two meet.

Then look for copies: count exact duplicate rows in your training data, and if there is a grouping variable — a patient, a customer, a device — refit with grouped cross-validation and compare with the OOB error. If the grouped estimate is noticeably higher, the OOB number you have been reporting was measuring memory.""")

    post.add(
        "Next",
        """Episode 14 moves from bagging to boosting and to the warning that comes with it: run boosting for long enough and it will overfit. Whether it does depends on which number you watch.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
