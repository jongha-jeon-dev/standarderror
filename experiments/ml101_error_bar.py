"""ML 1: Your Test Score Has an Error Bar, and It Is Wider Than Your Gain.

The first episode of *Machine Learning, Taught Through What Breaks*. A test
accuracy is an estimate from a finite sample and carries a standard error;
on the 360 test images of a 20% split of the UCI digits that error is about
0.7 points, the same size as the gaps between three ordinary classifiers.

Measured:

* One split: logistic regression, an RBF SVM and 3-NN, separated by
  fractions of a point to two points, with binomial error bars that overlap.
* 200 repeated splits reproduce the binomial spread -- but a control on a
  fixed correctness vector shows that what repeated splits of one pool
  measure is test-set noise with a finite-population correction, not the
  training-set variation they are usually run to expose.
* Comparing on the same rows is what sharpens a comparison. With every image
  scored once through 10-fold CV, 3-NN beats the SVM by 0.6 points and
  McNemar's exact test gives p = 0.06; resolving that gap needs about twice
  as many images as the dataset has.

Run: `standarderror run ml101_error_bar --publish`
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
SPLITS = 200
NAMES = ("logistic", "RBF SVM", "3-NN")


def compute() -> dict:
    X, y = ev.digits()
    sp = ev.split_scores(X, y, splits=SPLITS)
    first = {k: float(v[0]) for k, v in sp["accuracy"].items()}
    cvc = ev.cv_correct(X, y)
    pairs = {(a, b): ev.paired(cvc[a], cvc[b])
             for a, b in (("3-NN", "RBF SVM"), ("RBF SVM", "logistic"),
                          ("3-NN", "logistic"))}
    flips = {}
    acc = sp["accuracy"]
    for a, b in pairs:
        d = acc[a] - acc[b]
        flips[(a, b)] = float(np.mean(np.sign(d) == -np.sign(d.mean())))
    n, pool = sp["n_test"], sp["pool"]
    control = {k: ev.subset_sd(cvc[k], n) for k in NAMES}
    return {"X": X, "y": y, "splits": sp, "first": first, "cv": cvc,
            "pairs": pairs, "flips": flips, "control": control,
            "n": n, "pool": pool,
            "binom": {k: ev.binomial_se(float(acc[k].mean()), n)
                      for k in NAMES},
            "pool_se": {k: ev.pool_se(float(acc[k].mean()), n, pool)
                        for k in NAMES}}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    acc = res["splits"]["accuracy"]

    def spread(ax, m):
        rng = np.random.default_rng(0)
        for i, k in enumerate(NAMES):
            a = acc[k]
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(a)), a, s=9,
                       alpha=0.45, color=m.series[i], linewidths=0)
            mu, b = a.mean(), res["binom"][k]
            ax.errorbar([i + 0.32], [mu], yerr=[[1.96 * b], [1.96 * b]],
                        fmt="o", ms=6, capsize=5, lw=2, color=m.ink)
            f = res["first"][k]
            ax.plot([i - 0.25, i + 0.25], [f, f], lw=2.2, color=m.series[i])
        ax.set_xticks(range(len(NAMES)))
        ax.set_xticklabels(NAMES)
        ax.set_xlim(-0.6, len(NAMES) - 0.3)

    s = res["splits"]
    out["f0"] = charts.diagram(
        spread,
        title="Three models, two hundred test sets",
        subtitle=(f"Accuracy on the digits test split ({res['n']} of "
                  f"{res['pool']:,} images) for {s['splits']} random "
                  f"stratified splits. Bars: one test set's 95% binomial "
                  f"interval. Lines: the first split."),
        xlabel="",
        ylabel="test accuracy",
        source="Measured; standarderror/ml/evaluation.py.",
        alt=("Three overlapping clouds of dots for three classifiers, each "
             "about two points tall, with error bars of the same height."),
        caption=(f"The clouds overlap. One test set's 95% interval is "
                 f"about **+-{1.96 * res['binom']['RBF SVM']:.3f}**, wider "
                 f"than the gap between the SVM and 3-NN, which swap places "
                 f"in {res['flips'][('3-NN', 'RBF SVM')]:.0%} of splits."),
        path=str(IMG / f"ml101-f0-spread.{EXT}"))[0]

    p = res["pairs"][("3-NN", "RBF SVM")]
    diffs = np.linspace(0.002, 0.03, 80)

    def needed(ax, m):
        for disc, c, lab in ((p["discordance"], m.series[0],
                              f"{p['discordance']:.1%} of rows disagree "
                              f"(3-NN vs SVM)"),
                             (0.05, m.series[1], "5% disagree"),
                             (0.10, m.series[2], "10% disagree")):
            ax.plot(diffs * 100, [ev.rows_needed(d, disc) for d in diffs],
                    lw=2.4, color=c, label=lab)
        ax.axhline(res["pool"], lw=1.2, ls="--", color=m.ink_secondary)
        ax.annotate(f"all of digits, {res['pool']:,} images",
                    (diffs[-1] * 100, res["pool"] * 1.12), ha="right",
                    fontsize=8.6, color=m.ink_secondary)
        ax.plot([abs(p["difference"]) * 100],
                [ev.rows_needed(abs(p["difference"]), p["discordance"])],
                marker="o", ms=8, color=m.series[0], markeredgecolor=m.ink)
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8.4, loc="upper right")

    out["f1"] = charts.diagram(
        needed,
        title="How many test rows a paired comparison needs",
        subtitle=("Rows to detect an accuracy difference at 5% with 80% "
                  "power, by McNemar's test. Only rows where the two models "
                  "disagree carry information."),
        xlabel="accuracy difference to detect (points)",
        ylabel="test rows needed",
        source="Normal approximation to McNemar's test; "
               "standarderror/ml/evaluation.py.",
        alt=("Three falling curves on a log scale; a dot for the measured "
             "3-NN versus SVM gap sits above a dashed line marking the size "
             "of the dataset."),
        caption=(f"The measured gap, {abs(p['difference']):.1%} with "
                 f"{p['discordance']:.1%} of rows in disagreement, needs "
                 f"**{ev.rows_needed(abs(p['difference']), p['discordance']):,}"
                 f" rows**. The whole dataset is {res['pool']:,}."),
        path=str(IMG / f"ml101-f1-needed.{EXT}"))[0]

    rows_t = []
    for (a, b), r in res["pairs"].items():
        rows_t.append([f"{a} - {b}", f"{r['difference'] * 100:+.2f}",
                       f"{r['only_a']} / {r['only_b']}",
                       f"{r['unpaired_se'] * 100:.2f}",
                       f"{r['paired_se'] * 100:.2f}", f"{r['p_value']:.3f}",
                       f"{res['flips'][(a, b)]:.0%}"])
    out["f2"] = charts.table_image(
        rows_t,
        header=["comparison", "gap (pts)", "only A / only B right",
                "unpaired SE", "paired SE", "McNemar p",
                "splits it flips"],
        title="Every image scored once, by every model",
        subtitle=(f"10-fold cross-validated predictions on all "
                  f"{res['pool']:,} digits, the same folds for every model. "
                  f"Last column from the {s['splits']} random splits."),
        source="Measured; standarderror/ml/evaluation.py.",
        alt=("A three-row table of pairwise comparisons with their "
             "disagreement counts, standard errors and p-values."),
        caption=("Pairing cuts the standard error because two good models "
                 "fail on mostly the same images. It is still not enough "
                 "to separate the top two."),
        align="lrrrrrr",
        path=str(IMG / f"ml101-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    acc = res["splits"]["accuracy"]
    p = res["pairs"][("3-NN", "RBF SVM")]

    def clouds(panel, m):
        rng = np.random.default_rng(1)
        for i, k in enumerate(NAMES):
            panel.scatter(i + rng.uniform(-0.2, 0.2, 60), acc[k][:60], s=8,
                          color=m.series[i], linewidths=0)

    def bell(panel, m):
        xs = np.linspace(-4, 4, 80)
        panel.plot(xs, np.exp(-xs ** 2 / 2), lw=2.8, color=m.series[0])
        panel.plot([0.6, 0.6], [0, 1], lw=2.0, color=m.series[2])

    def curve(panel, m):
        d = np.linspace(0.003, 0.03, 40)
        panel.plot(d, [np.log(ev.rows_needed(x, p["discordance"]))
                       for x in d], lw=2.8, color=m.series[0])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=1,
        headline="A test score is an estimate, with an error bar",
        panels=[(clouds, f"+-{1.96 * res['binom']['RBF SVM'] * 100:.1f} pts",
                 "one test set, 95%"),
                (bell, f"p = {p['p_value']:.2f}", "3-NN vs SVM, paired"),
                (curve, f"{ev.rows_needed(abs(p['difference']), p['discordance']):,}",
                 "rows to settle it")],
        note=("The top two of three classifiers on 360 test digits differ "
              "by less than one test set's error bar. Scoring them on the "
              "same rows is the one move that sharpens the comparison, and "
              "on this dataset even that cannot separate them."),
        alt=("Three hand-drawn frames: overlapping clouds of dots; a bell "
             "curve with a line inside it; and a falling curve."),
        path=str(IMG / f"ml101-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["one"] = s.run("""
        from sklearn.model_selection import train_test_split
        from standarderror.ml import evaluation as ev

        X, y = ev.digits()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                              random_state=0, stratify=y)
        for name, make in ev.models().items():
            acc = (make().fit(Xtr, ytr).predict(Xte) == yte).mean()
            print(f"{name:9s} {acc:.4f}  +- {1.96 * ev.binomial_se(acc, len(yte)):.4f}")
        """)
    out["repeat"] = s.run("""
        sp = ev.split_scores(X, y, splits=200)
        cvc = ev.cv_correct(X, y)
        for name, acc in sp["accuracy"].items():
            p = acc.mean()
            print(f"{name:9s} SD {acc.std(ddof=1):.4f}  "
                  f"binomial {ev.binomial_se(p, 360):.4f}  "
                  f"pool {ev.pool_se(p, 360, len(y)):.4f}  "
                  f"fixed model {ev.subset_sd(cvc[name], 360):.4f}")
        """)
    out["paired"] = s.run("""
        r = ev.paired(cvc["3-NN"], cvc["RBF SVM"])
        print(f"3-NN - SVM on all {r['n']:,} images: {r['difference']:+.4f}")
        print(f"right only for 3-NN: {r['only_a']}   only for SVM: {r['only_b']}")
        print(f"SE unpaired {r['unpaired_se']:.4f}   paired {r['paired_se']:.4f}"
              f"   McNemar p = {r['p_value']:.3f}")
        print(f"rows needed: {ev.rows_needed(abs(r['difference']), r['discordance']):,}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    acc, first, b = res["splits"]["accuracy"], res["first"], res["binom"]
    n, pool = res["n"], res["pool"]
    p = res["pairs"][("3-NN", "RBF SVM")]
    pl = res["pairs"][("RBF SVM", "logistic")]
    need = ev.rows_needed(abs(p["difference"]), p["discordance"])
    sd = {k: float(acc[k].std(ddof=1)) for k in NAMES}
    ctl = res["control"]
    flip = res["flips"][("3-NN", "RBF SVM")]
    gain0 = first["RBF SVM"] - first["logistic"]
    gap0 = first["3-NN"] - first["RBF SVM"]
    tie = ("ties 3-NN exactly" if abs(gap0) < 1e-12 else
           f"{'trails' if gap0 > 0 else 'beats'} 3-NN by "
           f"{abs(gap0) * 100:.1f}")
    train_share = {k: 1 - ctl[k] ** 2 / sd[k] ** 2 for k in NAMES}

    # The spine, asserted rather than trusted.
    assert all(abs(sd[k] / res["pool_se"][k] - 1) < 0.2 for k in NAMES)
    assert all(abs(sd[k] / ctl[k] - 1) < 0.25 for k in NAMES)
    assert all(sd[k] < b[k] for k in NAMES)
    assert 0.05 < p["p_value"] < 0.2 and p["difference"] > 0
    assert need > pool
    assert pl["p_value"] < 0.001
    assert 0.05 < flip < 0.4
    assert p["paired_se"] < p["unpaired_se"]

    post = Post(
        title=(f"{SERIES_TAG} 1: Your Test Score Has an Error Bar, and It Is "
               f"Wider Than Your Gain"),
        slug="ml-1-error-bar",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=1,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A test accuracy is a proportion measured on a finite "
                  "sample. On 360 test digits its error bar is as wide as "
                  "the gaps between three good classifiers, and repeating "
                  "the split does not measure what it seems to."),
        summary=(
            f"On a 20% test split of the UCI digits, {n} images, logistic "
            f"regression, an RBF SVM and 3-NN score "
            f"{first['logistic']:.3f}, {first['RBF SVM']:.3f} and "
            f"{first['3-NN']:.3f}, and one test set's 95% binomial interval "
            f"is about +-{1.96 * b['RBF SVM']:.3f}. Over {SPLITS} random "
            f"splits the SVM and 3-NN swap places {flip:.0%} of the time. "
            f"The repeated splits' spread matches test-set noise with a "
            f"finite-population correction - it measures the pool, not the "
            f"training sets. Scored on the same rows, every image once, 3-NN "
            f"beats the SVM by {p['difference'] * 100:.1f} points with "
            f"McNemar p = {p['p_value']:.2f}; settling a gap that size "
            f"needs about {need:,} images, and the dataset has {pool:,}."),
        tags=["machine-learning", "data-science", "statistics",
              "model-evaluation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "UCI Optical Recognition of Handwritten Digits, as bundled with "
            "scikit-learn (`load_digits`): 1,797 8x8 images, CC BY 4.0. "
            "Alpaydin and Kaynak (1998), UCI Machine Learning Repository.",
            "Machinery: `standarderror/ml/evaluation.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: McNemar, \"Note on the sampling error of the "
            "difference between correlated proportions or percentages\", "
            "*Psychometrika* (1947); Dietterich, \"Approximate statistical "
            "tests for comparing supervised classification learning "
            "algorithms\", *Neural Computation* (1998), which compares "
            "these tests and warns against resampled t-tests.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("splits use random_state 0 to 199, stratified; "
                            "the 10-fold predictions use one fixed shuffle "
                            "shared by every model"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "One split, three models",
        f"""This series takes one sentence from the standard machine-learning curriculum per episode and measures where it stops being true. Thirty of them, from evaluation through linear models, trees, explanation, training networks and what generalisation looks like in them. The first sentence is the one every other episode leans on: *the model with the higher test score is the better model.*

The rules are the same throughout. Every number is measured by code in the repository and pinned by a test, so a claim that drifts fails the build. The data is either bundled with scikit-learn or simulated with a known answer, so anyone can rebuild it offline. And when a measurement contradicts the sentence I set out to illustrate — it happened twice in the last series — the episode says so rather than quietly changing the question.

Here is the most ordinary experiment there is. Three classifiers on the UCI handwritten digits, one stratified 80/20 split.

{snip['one'].markdown()}

Read the usual way, the SVM beats logistic regression by {gain0 * 100:.1f} points and {tie}. The numbers after the +- are the 95% binomial interval for each score: a test accuracy is a proportion measured on {n} images, and its standard error is sqrt(p(1 - p)/n). At 98% that is {b['RBF SVM']:.4f}, so the interval is about {1.96 * b['RBF SVM'] * 100:.1f} points either way — wider than the gap between the top two models, and not much narrower than the gap between the top and the bottom.""")

    post.add(
        "Repeat the split, and check what you measured",
        f"""The usual response to "one split is noisy" is to repeat it. Two hundred stratified splits:

{snip['repeat'].markdown()}

The columns are the standard deviation of accuracy over the {SPLITS} splits, the binomial standard error of one {n}-image test set, the same with a finite-population correction, and a control explained below. The spread over splits, {sd['RBF SVM']:.4f} for the SVM, is close to the binomial standard error of {b['RBF SVM']:.4f} — slightly *below* it, for every model. That is the first sign the repetition is measuring something narrower than it looks.

Two hundred splits of one pool of {pool:,} images do not draw two hundred fresh test sets. They draw {n}-image subsets of the same images, so the finite-population correction applies: the variance of a subset mean is smaller by (N - n)/(N - 1). With it, the predicted spread is {res['pool_se']['RBF SVM']:.4f}. The last column is the control that settles what the spread is made of: fix each image's right-or-wrong from one cross-validated run, so the model never changes, and draw random {n}-image subsets. That spread, {ctl['RBF SVM']:.4f} for the SVM, is the pure test-set noise with the training set held still, and it accounts for most of the repeated-split spread. The share of variance left over for retraining on different data is {train_share['logistic']:.0%} for logistic regression, {train_share['RBF SVM']:.0%} for the SVM and {train_share['3-NN']:.0%} for 3-NN — a negative share meaning the control alone is already a little wider than the measured spread.

So repeating the split re-measures the test-set noise of *this* pool, slightly understated, and almost none of the variation from training on different data. It does not tell you how the score would move on a new sample of digits; the binomial standard error of one test set is the better answer to that. What the repetition does show plainly is the ranking: the SVM and 3-NN trade places in {flip:.0%} of the {SPLITS} splits.""",
        figures=[figs["f0"]])

    post.add(
        "Compare on the same rows",
        f"""The error bars above treat each model's score as an independent measurement, and they are not. Two good classifiers fail on mostly the same hard images, so on a shared test set their errors are correlated, and the difference between them is less noisy than either score. A paired comparison uses that. Only the images where the models *disagree* carry information about which is better; McNemar's test is the exact test on those.

To use every image, score each one once with 10-fold cross-validation, on the same folds for all three models.

{snip['paired'].markdown()}

On {p['n']:,} images, 3-NN is right on {p['only_a']} that the SVM gets wrong, and the SVM on {p['only_b']} that 3-NN gets wrong. Pairing takes the standard error of the difference from {p['unpaired_se']:.4f} to {p['paired_se']:.4f}: when only a fraction d of rows disagree, the per-row difference is zero almost everywhere, and its variance is close to d rather than the sum of the two models' error variances. The gap is {p['difference'] * 100:.1f} points and McNemar's p is {p['p_value']:.2f}: suggestive, not settled, on every image the dataset has.

How many would settle it? With {p['discordance']:.1%} of images in disagreement, detecting a {abs(p['difference']) * 100:.1f}-point difference at the 5% level with 80% power takes about **{need:,} images** — {need / pool:.1f} times the dataset. The SVM's {pl['difference'] * 100:.1f}-point lead over logistic regression, by contrast, is decisive (p < 0.001). Some comparisons this data can answer. The one at the top of the leaderboard is not one of them.""",
        figures=[figs["f1"], figs["f2"]])

    post.add(
        "Where this breaks",
        """**The binomial error bar assumes independent test rows.** If test rows share a source — several digits from one writer, several frames of one video, several sentences of one document — the effective sample is smaller and the error bar wider. The uncertainty series measured that effect for conformal coverage; it applies to accuracy unchanged.

**The pool-corrected spread is not wrong, it is a different quantity.** If the question is how much this particular dataset's split could move the score, the repeated splits answer it. If the question is how the model would do on new data, they understate it, and they say almost nothing about the training-set component.

**The required-rows formula is an approximation.** It is the normal approximation to McNemar's test with the disagreement rate taken as known. At small disagreement counts it is optimistic; it is meant for planning, not for testing.

**Cross-validated predictions are not quite independent either.** Each image is predicted by a model trained on 90% of the others, and those models overlap. Dietterich's 1998 comparison of tests is the standard reference for what that does to type I error; McNemar on a single split is the conservative choice it recommends.""")

    post.add(
        "What to keep",
        f"""1. A test accuracy is a proportion with standard error sqrt(p(1 - p)/n). On {n} images at 98%, the 95% interval is about +-{1.96 * b['RBF SVM'] * 100:.1f} points.

2. Repeated random splits of one dataset reproduce that spread with a finite-population correction. They measure the pool's test-set noise, not training-set variation and not the spread on new data.

3. Compare models on the same rows. Only disagreements count, and pairing shrinks the standard error of a difference ({p['unpaired_se']:.4f} to {p['paired_se']:.4f} here).

4. 3-NN beats the RBF SVM on digits by {p['difference'] * 100:.1f} points with p = {p['p_value']:.2f}. Settling it would take about {need:,} images; the dataset has {pool:,}.

5. Before reporting an improvement, compute how many test rows it would take to detect it. If the answer is more than you have, the improvement is not yet a result.""")

    post.add(
        "Exercise",
        """Take the last model comparison you reported and redo it as a paired comparison: the same test rows, a count of the rows where only one model was right, and McNemar's p. Then plug the disagreement rate and the gap into the rows-needed formula. If the answer is larger than your test set, write the gap down as "not resolved on this data" and keep the comparison open.

Then check whether your test rows are independent. If they share writers, users, sessions or documents, the binomial error bar is a floor, not an estimate.""")

    post.add(
        "Next",
        """Episode 2 turns to the standard answer to "one split is noisy": cross-validation. It is usually described as estimating the error of the model you fit. Measured on data where that error is known exactly, it is barely correlated with it.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
