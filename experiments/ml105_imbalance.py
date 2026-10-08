"""ML 5: Resampling Moves the Threshold and Breaks the Probabilities.

The fifth episode of *Machine Learning, Taught Through What Breaks*, closing
the first arc. "Resample until the classes are balanced" is the first fix
taught for a rare positive class. Measured at 2% prevalence against class
weights and against moving the plain model's threshold:

* Logistic regression: oversampling, SMOTE and class weights leave the
  ranking where it was (AUC within 0.002, slightly *lower*, consistently);
  undersampling costs a little more.
* They inflate the probabilities tenfold and log loss fivefold; the prior
  correction undoes it.
* What they do to the decisions -- recall at the 0.5 cut from 0.19 to 0.83
  -- the plain model does with a threshold of about 0.02, at the same
  precision.
* Boosted trees: resampling and weights lower the AUC by about 0.01 and
  SMOTE loses a seventh of the average precision; the plain model with a
  moved threshold beats every resampled one at its own recall.

Run: `standarderror run ml105_imbalance --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import imbalance as im
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 8)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def _one_replicate() -> dict:
    """The predictions of one training set, for the figures."""
    task = im.ImbalancedTask()
    Xt, yt = task.draw(200_000, seed=99)
    X, y = task.draw(5000, seed=0)
    plain = im.fit_predict("logistic", "plain", X, y, Xt, seed=0)
    over = im.fit_predict("logistic", "oversample", X, y, Xt, seed=0)
    return {"y": yt, "plain": plain, "over": over,
            "corrected": im.prior_correct(over, 0.5, float(y.mean()))}


def compute() -> dict:
    return {"lo": im.compare(model="logistic", reps=20),
            "bo": im.compare(model="boosted trees", reps=10),
            "dg": im.digits_eights(),
            "one": _one_replicate()}


# ------------------------------------------------------------------ figures

def _pr(p, y):
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    k = np.arange(1, len(p) + 1)
    return tp / y.sum(), tp / k


def figures(res: dict) -> dict:
    out: dict = {}
    one = res["one"]
    y = one["y"]

    def pr(ax, m):
        for key, c, lab in (("plain", m.series[0], "plain"),
                            ("over", m.series[1], "oversampled")):
            r, p = _pr(one[key], y)
            step = max(1, len(r) // 4000)
            ax.plot(r[::step], p[::step], lw=2.2, color=c,
                    label=f"{lab}, every threshold")
        for key, c, lab in (("plain", m.series[0], "plain at 0.5"),
                            ("over", m.series[1], "oversampled at 0.5")):
            h = im.at_threshold(one[key], y, 0.5)
            ax.plot([h["recall"]], [h["precision"]], marker="o", ms=10,
                    color=c, markeredgecolor=m.ink, ls="", label=lab)
        t = im.threshold_for_recall(one["plain"], y, im.at_threshold(
            one["over"], y, 0.5)["recall"])
        h = im.at_threshold(one["plain"], y, t)
        ax.plot([h["recall"]], [h["precision"]], marker="s", ms=8,
                color=m.series[0], markeredgecolor=m.ink, ls="",
                label=f"plain at {t:.3f}")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8.2, loc="upper right")

    lo = res["lo"]
    mt = lo["matched"]["oversample"]
    out["f0"] = charts.diagram(
        pr,
        title="Oversampling moves along the curve, not the curve",
        subtitle=("Precision against recall on 200,000 test rows at 2% "
                  "prevalence, for logistic regression trained plain and "
                  "trained on an oversampled copy of the same data."),
        xlabel="recall",
        ylabel="precision",
        source="Simulated; standarderror/ml/imbalance.py.",
        alt=("Two precision-recall curves lying on top of each other, with "
             "the oversampled model's default operating point far to the "
             "right of the plain model's and a square marking the plain "
             "model at a lower threshold on the same spot."),
        caption=(f"The curves are the same. The default cut puts them at "
                 f"different points, and the plain model gets to the "
                 f"oversampled one's point with a threshold of "
                 f"**{mt['plain_threshold']:.3f}** instead of 0.5."),
        path=str(IMG / f"ml105-f0-pr.{EXT}"))[0]

    def reliability(ax, m):
        edges = np.r_[0, np.geomspace(0.002, 1, 16)]
        for key, c, lab in (("plain", m.series[0], "plain"),
                            ("over", m.series[1], "oversampled"),
                            ("corrected", m.series[2],
                             "oversampled, prior-corrected")):
            p = one[key]
            xs, ys = [], []
            for lo_, hi_ in zip(edges[:-1], edges[1:]):
                mask = (p >= lo_) & (p < hi_)
                if mask.sum() >= 200:
                    xs.append(p[mask].mean())
                    ys.append(y[mask].mean())
            ax.plot(xs, ys, marker="o", ms=5, lw=2.0, color=c, label=lab)
        ax.plot([0.002, 1], [0.002, 1], lw=1.2, ls="--",
                color=m.ink_secondary)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    s = lo["summary"]
    out["f1"] = charts.diagram(
        reliability,
        title="The probabilities are what resampling breaks",
        subtitle=("Observed positive rate against predicted probability, in "
                  "bins, on log axes. On the dashed line a predicted 10% "
                  "comes true 10% of the time."),
        xlabel="predicted probability",
        ylabel="observed positive rate",
        source="Simulated; standarderror/ml/imbalance.py.",
        alt=("A plain-model line on the diagonal, an oversampled line far "
             "below it, and a corrected line back on the diagonal."),
        caption=(f"Oversampled, a predicted probability is about ten times "
                 f"the real rate: mean prediction {s['oversample']['mean_p']:.3f} "
                 f"against a prevalence of {lo['prevalence']:.2f}. "
                 f"**One line of algebra** puts it back."),
        path=str(IMG / f"ml105-f1-calibration.{EXT}"))[0]

    rows_t = []
    for mdl, R in (("logistic", res["lo"]), ("boosted trees", res["bo"])):
        for k in im.METHODS:
            v = R["summary"][k]
            rows_t.append([mdl if k == "plain" else "", k,
                           f"{v['auc']:.4f}", f"{v['ap']:.3f}",
                           f"{v['precision_r80']:.3f}",
                           f"{v['mean_p']:.3f}", f"{v['log_loss']:.3f}",
                           f"{v['recall_half']:.2f}"])
    bold = {(i, c) for i, r in enumerate(rows_t) for c in range(8)
            if r[1] == "plain"}
    out["f2"] = charts.table_image(
        rows_t,
        header=["model", "training data", "ROC AUC", "avg precision",
                "precision at 80% recall", "mean prob.", "log loss",
                "recall at 0.5"],
        title="Ranking, probabilities, decisions",
        subtitle=(f"2% prevalence, {lo['n']:,} training rows, 200,000 test "
                  f"rows; means over {lo['reps']} training sets (logistic) "
                  f"and {res['bo']['reps']} (boosted trees)."),
        source="Simulated; standarderror/ml/imbalance.py.",
        alt=("A ten-row table: ranking columns that never improve on the "
             "plain rows, mean probabilities ten times the prevalence for "
             "resampled logistic regression, and recall at 0.5 that rises "
             "with resampling."),
        caption=("No column on the left improves on the **plain** row. The "
                 "right-hand column, the one resampling is used for, is a "
                 "threshold."),
        bold_cells=bold, align="llrrrrrr",
        path=str(IMG / f"ml105-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    one, lo, bo = res["one"], res["lo"], res["bo"]
    y = one["y"]

    def same(panel, m):
        for key, c in (("plain", 0), ("over", 1)):
            r, p = _pr(one[key], y)
            step = max(1, len(r) // 300)
            panel.plot(r[::step], p[::step], lw=2.6, color=m.series[c])

    def calib(panel, m):
        xs = np.linspace(0, 1, 30)
        panel.plot(xs, xs, lw=2.6, color=m.series[0])
        panel.plot(xs, xs ** 3, lw=2.6, color=m.series[1])

    def trees(panel, m):
        vals = [bo["summary"][k]["auc"] for k in im.METHODS]
        panel.bar(range(len(vals)), vals, color=m.series[2])
        panel.set_ylim(min(vals) - 0.01, max(vals) + 0.003)

    t = lo["matched"]["oversample"]["plain_threshold"]
    ratio = lo["summary"]["oversample"]["mean_p"] / lo["prevalence"]
    return charts.lecture_hero(
        series=SERIES_TAG, episode=5,
        headline="Resampling moves the threshold and breaks the probabilities",
        panels=[(same, f"t = {t:.3f}", "a threshold does it"),
                (calib, f"{ratio:.0f}x", "inflated probabilities"),
                (trees, f"{bo['paired']['SMOTE']['auc']['mean']:+.3f}",
                 "AUC, SMOTE on trees")],
        note=("At 2% prevalence, oversampling, SMOTE and class weights leave "
              "a logistic model's ranking unchanged, multiply its "
              "probabilities tenfold, and change its decisions only as much "
              "as moving the threshold would. On boosted trees they make the "
              "ranking worse."),
        alt=("Three hand-drawn frames: two curves on top of each other; a "
             "diagonal and a curve sagging below it; and five bars of which "
             "the first is tallest."),
        path=str(IMG / f"ml105-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["compare"] = s.run("""
        from standarderror.ml import imbalance as im

        lo = im.compare(model="logistic", reps=20)   # 2% positive
        for m, v in lo["summary"].items():
            print(f"{m:13s} AUC {v['auc']:.4f}  mean prob {v['mean_p']:.3f}  "
                  f"log loss {v['log_loss']:.3f}  recall@0.5 {v['recall_half']:.2f}")
        """)
    out["paired"] = s.run("""
        for m, d in lo["paired"].items():
            a, p = d["auc"], d["precision_r80"]
            print(f"{m:13s} AUC {a['mean']:+.4f} +- {a['se']:.4f}   "
                  f"precision at 80% recall {p['mean']:+.4f} +- {p['se']:.4f}")
        """)
    out["matched"] = s.run("""
        for m in ("oversample", "SMOTE", "class weights"):
            v = lo["matched"][m]
            print(f"{m:13s} recall {v['recall']:.3f}: precision "
                  f"{v['method_precision']:.3f} at 0.5, plain "
                  f"{v['plain_precision']:.3f} at {v['plain_threshold']:.3f}")
        """)
    out["correct"] = s.run("""
        c = lo["corrected"]
        print(f"oversampled, prior-corrected: mean prob {c['mean_p']:.4f}  "
              f"log loss {c['log_loss']:.4f}  AUC {c['auc']:.4f}")
        """)
    out["trees"] = s.run("""
        bo = im.compare(model="boosted trees", reps=10)
        for m, d in bo["paired"].items():
            print(f"{m:13s} AUC {d['auc']['mean']:+.4f} +- {d['auc']['se']:.4f}"
                  f"   avg precision {d['ap']['mean']:+.4f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    lo, bo, dg = res["lo"], res["bo"], res["dg"]
    s, pr_, mt, c = lo["summary"], lo["paired"], lo["matched"], lo["corrected"]
    bp, bm = bo["paired"], bo["matched"]
    ds = dg["summary"]
    prev = lo["prevalence"]
    ratio = s["oversample"]["mean_p"] / prev
    ll = s["oversample"]["log_loss"] / s["plain"]["log_loss"]
    resampled = ("oversample", "SMOTE", "class weights")

    # The spine, asserted rather than trusted.
    for m in resampled:
        assert -0.005 < pr_[m]["auc"]["mean"] < 0
        assert pr_[m]["precision_r80"]["mean"] < 0
        assert mt[m]["plain_precision"] >= mt[m]["method_precision"] - 0.003
        assert s[m]["recall_half"] > 0.75 and s[m]["mean_p"] > 5 * prev
    assert abs(s["plain"]["mean_p"] - prev) < 0.003
    assert pr_["undersample"]["auc"]["mean"] < pr_["oversample"]["auc"]["mean"]
    assert abs(c["log_loss"] / s["plain"]["log_loss"] - 1) < 0.05
    assert ll > 4
    for m in resampled:
        assert bp[m]["auc"]["mean"] + 2 * bp[m]["auc"]["se"] < 0
    assert bm["SMOTE"]["plain_precision"] > bm["SMOTE"]["method_precision"]
    assert all(ds[m]["auc"] <= ds["plain"]["auc"] for m in im.METHODS)
    assert bo["corrected"]["log_loss"] > bo["summary"]["oversample"]["log_loss"]
    assert bp["undersample"]["ap"]["mean"] < -0.03
    assert all(ds[m]["precision_r80"] < ds["plain"]["precision_r80"]
               for m in im.METHODS if m != "plain")

    post = Post(
        title=(f"{SERIES_TAG} 5: Resampling Moves the Threshold and Breaks "
               f"the Probabilities"),
        slug="ml-5-imbalance",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=5,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("'Balance the classes before training' is the first fix "
                  "taught for a rare positive class. Measured, it never "
                  "improved the ranking, multiplied the probabilities "
                  "tenfold, and did nothing to the decisions that a "
                  "threshold does not."),
        summary=(
            f"At {prev:.0%} prevalence, logistic regression trained on an "
            f"oversampled, SMOTE-augmented or class-weighted copy of the "
            f"data ranks the test set no better than the plain model - "
            f"AUC {pr_['oversample']['auc']['mean']:+.4f} for oversampling, "
            f"small and consistently negative - while its mean predicted "
            f"probability goes from {s['plain']['mean_p']:.3f} to "
            f"{s['oversample']['mean_p']:.3f} and its log loss rises "
            f"{ll:.1f}-fold. The recall it gains at the 0.5 cut, "
            f"{s['plain']['recall_half']:.2f} to "
            f"{s['oversample']['recall_half']:.2f}, the plain model gets "
            f"with a threshold of {mt['oversample']['plain_threshold']:.3f}, "
            f"at {mt['oversample']['plain_precision']:.3f} precision against "
            f"{mt['oversample']['method_precision']:.3f}. On boosted trees "
            f"the corrections lower the AUC by about 0.01, and SMOTE costs "
            f"{-bp['SMOTE']['ap']['mean']:.3f} of average precision. The "
            f"digits agree."),
        tags=["machine-learning", "data-science", "statistics",
              "class-imbalance", "calibration", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: 8 standard-normal features, positives at 2% shifted "
            "by 1.2 on three features and spread 1.8 times wider on a "
            "fourth; 5,000 training rows per replicate, 200,000 test rows "
            "at the same prevalence.",
            "UCI Optical Recognition of Handwritten Digits, as bundled with "
            "scikit-learn (`load_digits`), CC BY 4.0, recast as 'is it an "
            "8?' at about 10% positive.",
            "Machinery: `standarderror/ml/imbalance.py`, including a "
            "from-scratch SMOTE, tested in `tests/test_ml.py`.",
            "Where this stops: Chawla et al., \"SMOTE: synthetic minority "
            "over-sampling technique\", *JAIR* (2002); Elkan, \"The "
            "foundations of cost-sensitive learning\", *IJCAI* (2001); "
            "Saerens, Latinne and Decaestecker, \"Adjusting the outputs of a "
            "classifier to new a priori probabilities\", *Neural "
            "Computation* (2002); van den Goorbergh et al., \"The harm of "
            "class imbalance corrections for risk prediction models\", "
            "*JAMIA* (2022).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("training sets from seeds 0 to 19, the test set "
                            "from seed 99; every method within a replicate "
                            "sees the same training set, so differences are "
                            "paired"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The fix taught first",
        f"""When the positive class is rare — fraud, a disease, a defect — a classifier trained on the data as it comes predicts the negative class almost everywhere, and at the default 0.5 cut it catches few positives. The first fix most courses teach is to balance the classes before training: duplicate the minority rows (oversampling), synthesise new ones between them (SMOTE), throw away majority rows (undersampling), or weight the minority class up in the loss. The sentence behind all four is that *imbalance is a problem in the data, and resampling fixes it.*

To see what each actually changes, it helps to separate three things a classifier produces. Its **ranking**: which cases it thinks more likely positive than others, measured by ROC AUC, average precision, or precision at a fixed recall. Its **probabilities**: whether a predicted 10% comes true 10% of the time. And its **decisions**: which cases get flagged at whatever threshold is applied.

A simulated task makes all three measurable at once: positives at {prev:.0%}, 5,000 training rows — about 100 positives — and a 200,000-row test set at the same prevalence. Logistic regression first, the five versions trained on the same rows, twenty times over.

{snip['compare'].markdown()}

The first column barely moves. The second and third move a lot. The last column, the one resampling is used for, moves from {s['plain']['recall_half']:.2f} to {min(s[m]['recall_half'] for m in im.METHODS if m != 'plain'):.2f} or more.""")

    post.add(
        "The ranking: no better, slightly worse",
        f"""Because every version trains on the same rows in each replicate, the differences from the plain model are paired, and their standard errors are small enough to see the sign.

{snip['paired'].markdown()}

Oversampling, SMOTE and class weights each cost about {abs(pr_['oversample']['auc']['mean']):.4f} of AUC — tiny, and consistently negative. None of them improved the ranking in the measurement, and precision at 80% recall fell slightly too. Undersampling, which discards roughly 98% of the majority rows, costs about three times as much.

That is what the algebra predicts for logistic regression. Duplicating positives, or weighting them, mostly shifts the intercept: every log-odds goes up by roughly the same amount, and a shift that applies to every row cannot reorder them. What little reordering there is comes from the slopes being fitted to a reweighted sample, and it does not help.""",
        figures=[figs["f0"]])

    post.add(
        "The decisions: a threshold does the same job",
        f"""If the ranking did not change, the jump in recall must have come from somewhere else, and it did: shifting every log-odds up is the same as lowering the threshold. The test is to take the plain model and move its threshold until it flags as many positives as each corrected model does at 0.5.

{snip['matched'].markdown()}

At the same recall the plain model is as precise as every corrected one, or slightly more. The oversampled model's 0.5 corresponds to a plain-model threshold of about {mt['oversample']['plain_threshold']:.3f} — almost exactly the prevalence, which is what a cost argument says it should be when a missed positive is weighted like 49 false alarms.

That is the real content of "fixing" imbalance: choosing a different trade-off between missed positives and false alarms. It is a decision about costs, and a threshold expresses it directly, can be changed after training, and does not touch the model.""")

    post.add(
        "The probabilities: what resampling breaks",
        f"""The cost of doing it by resampling shows up in the probabilities. The plain model's mean prediction is {s['plain']['mean_p']:.3f} against a true rate of {prev:.2f}: it is calibrated. The oversampled model's is {s['oversample']['mean_p']:.3f}, about {ratio:.0f} times too high, and its log loss is {ll:.1f} times worse. It was trained on a world where positives are as common as negatives, and it reports probabilities for that world.

Anyone who reads those numbers as probabilities — to rank by expected cost, to set a threshold by a business rule, to combine with another model, to tell a patient a risk — is reading the wrong world. Van den Goorbergh and colleagues found exactly this in clinical risk models: imbalance corrections improved no discrimination measure and produced strongly overestimated risks.

The damage is reversible for logistic regression, because it is mostly an intercept: multiply each predicted odds by the ratio of the true prior odds to the training prior odds.

{snip['correct'].markdown()}

Mean prediction and log loss are back to the plain model's. Which is a long way round to arrive where the plain model started.""",
        figures=[figs["f1"]])

    post.add(
        "For trees it is not even neutral",
        f"""Logistic regression can only shift and tilt a hyperplane, so resampling has little room to do harm. A flexible model has room. The same comparison with boosted trees, ten training sets:

{snip['trees'].markdown()}

Oversampling, SMOTE and class weights each lower the AUC by about 0.01, well outside their standard errors, and SMOTE loses {-bp['SMOTE']['ap']['mean']:.3f} of average precision. Undersampling barely moves the AUC ({bp['undersample']['auc']['mean']:+.4f}, inside its standard error) but loses {-bp['undersample']['ap']['mean']:.3f} of average precision, which is where a rare-class problem is decided. Duplicated positives are fitted exactly; SMOTE's synthetic positives fill in the space *between* real ones, which is not where positives are, and the trees learn that geometry. At its own operating point the SMOTE model reaches {bm['SMOTE']['method_precision']:.3f} precision; the plain model with its threshold moved to the same recall reaches {bm['SMOTE']['plain_precision']:.3f}.

The prior correction does not rescue the trees either: their probabilities were not inflated by a clean intercept shift but distorted by memorising the duplicated rows, and the correction applied to them makes the log loss worse, not better.

The digits — "is this an 8?", about {dg['prevalence']:.0%} positive, logistic regression over 30 splits — agree with the simulation. The plain model has the highest AUC ({ds['plain']['auc']:.4f}) and the highest precision at 80% recall ({ds['plain']['precision_r80']:.3f}), against {ds['oversample']['precision_r80']:.3f} oversampled and {ds['undersample']['precision_r80']:.3f} undersampled.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        """**Severe imbalance with tiny counts.** With about 100 positives the plain model is well estimated. With ten, every method is noisy, and some regularisation from reweighting can help a badly overfitted model; the fix there is regularisation or more positives, not balance.

**Models that ignore the threshold.** Some pipelines take the argmax and have no threshold to move — a deep network's softmax inside a larger system, say. Then reweighting is the only lever available, and it is the threshold in disguise; it should still be corrected before the probabilities are used.

**Losses that are not proper.** The comparison assumes the model is trained on log loss. Trained to maximise accuracy directly, a model on imbalanced data can genuinely learn to ignore the minority class, and reweighting changes what it learns.

**One simulated task.** The tree result depends on how much the trees can memorise; with stronger regularisation the damage shrinks. The direction — no gain in ranking — held in all three settings measured.""")

    post.add(
        "What to keep",
        f"""1. Separate ranking, probabilities and decisions. Imbalance corrections change the last two and, at best, leave the first alone.

2. On logistic regression at {prev:.0%} prevalence, oversampling, SMOTE and class weights changed AUC by about {pr_['oversample']['auc']['mean']:+.4f}; undersampling by {pr_['undersample']['auc']['mean']:+.4f}.

3. The recall they gain at 0.5 the plain model gets with a threshold near the prevalence ({mt['oversample']['plain_threshold']:.3f}), at the same precision or better.

4. They inflate predicted probabilities about {ratio:.0f}-fold. For a linear model the prior correction undoes it.

5. On boosted trees they made the ranking worse: about -0.01 AUC, and {-bp['SMOTE']['ap']['mean']:.3f} of average precision for SMOTE.

6. Imbalance is not a defect in the data. It is a cost question, and a threshold is where costs belong.""")

    post.add(
        "Exercise",
        """Find a model in your stack trained on resampled or reweighted data. Train the same model on the data as it comes, and compare the two on the same test rows: AUC, average precision, and precision at the recall you actually operate at. If the plain model matches or beats the corrected one at that recall, the correction was a threshold; replace it with one.

Then look at what consumes the corrected model's probabilities. If anything treats them as probabilities — a dashboard, an expected-cost calculation, another model — check its mean prediction against the true rate. If it is several times too high, either apply the prior correction or retrain without the resampling.""")

    post.add(
        "Next",
        """That closes the first arc, on what a score means. Arc II turns to linear models pushed past their textbook conditions, starting with the most common classifier there is: logistic regression on data that a straight line separates perfectly, where the maximum-likelihood coefficients the textbook promises do not exist.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
