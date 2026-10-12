"""ML 14: Boosting Overfits Its Probabilities Long Before Its Accuracy.

The fourteenth episode of *Machine Learning, Taught Through What Breaks*.
"Boosting overfits if you run it long enough." Measured on the breast-cancer
data over ten half splits, to 2,000 rounds:

* With clean labels the test error never turns up: at a learning rate of
  0.1 it falls to 0.046 and stays there.
* The test log loss turns early. It is lowest after about 46 rounds (0.161)
  and 2.6 times worse at 2,000. The model keeps getting the same answers
  right while growing ever more certain about the ones it gets wrong.
* Shrinkage does not prevent this; it slows it. Each tenfold cut in the
  learning rate moves the best round about tenfold later and leaves the
  best log loss where it was.
* With 10% of training labels flipped, the error turns too, but by less than
  two points; the log loss more than triples.

Run: `standarderror run ml114_boosting --publish`
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

RATES = (1.0, 0.1, 0.01)


def compute() -> dict:
    return {(flip, lr): tr.boost_curves(lr=lr, flip=flip)
            for flip in (0.0, 0.1) for lr in RATES}


# ------------------------------------------------------------------ figures

def _axis(ax, rounds):
    ax.set_xscale("log")
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"{r:,}" for r in rounds])
    ax.minorticks_off()


def figures(res: dict) -> dict:
    out: dict = {}
    rounds = res[(0.0, 0.1)]["rounds"]

    def ll(ax, m):
        for c, lr in zip(m.series, RATES):
            r = res[(0.0, lr)]
            ax.plot(rounds, r["log_loss"], marker="o", ms=4, lw=2.2, color=c,
                    label=f"learning rate {lr:g}")
            i = int(np.argmin(r["log_loss"]))
            ax.scatter([rounds[i]], [r["log_loss"][i]], s=90, facecolor="none",
                       edgecolor=c, lw=2)
        _axis(ax, rounds)
        ax.set_ylim(0, 0.7)
        ax.legend(frameon=False, fontsize=8.5, loc="upper center")

    c1 = res[(0.0, 0.1)]
    i1 = int(np.argmin(c1["log_loss"]))
    out["f0"] = charts.diagram(
        ll,
        title="The probabilities overfit early",
        subtitle=("Test log loss of gradient-boosted trees (depth 3) against "
                  "the number of rounds, clean labels, mean of 10 half "
                  "splits of the breast-cancer data. Circles mark each "
                  "curve's best checkpoint."),
        xlabel="boosting rounds (log scale)",
        ylabel="test log loss",
        source="Breast Cancer Wisconsin (scikit-learn); "
               "standarderror/ml/trees.py.",
        alt=("Three U-shaped curves, shifted right as the learning rate "
             "falls, each reaching about the same minimum before rising."),
        caption=(f"At learning rate 0.1 the log loss bottoms at "
                 f"{c1['log_loss'][i1]:.3f} after {rounds[i1]} rounds and "
                 f"reaches **{c1['log_loss'][-1]:.3f}** by 2,000. A smaller "
                 f"rate reaches the same bottom later; it does not lower it."),
        path=str(IMG / f"ml114-f0-logloss.{EXT}"))[0]

    def err(ax, m):
        for c, lr in zip(m.series, RATES):
            for flip, ls in ((0.0, "-"), (0.1, "--")):
                r = res[(flip, lr)]
                ax.plot(rounds[3:], r["error"][3:], ls=ls, marker="o", ms=4,
                        lw=2.0, color=c,
                        label=f"rate {lr:g}, "
                              f"{'clean' if flip == 0 else '10% flipped'}")
        _axis(ax, rounds[3:])
        ax.set_ylim(0, 0.16)
        ax.legend(frameon=False, fontsize=8.0, loc="upper right", ncol=2)

    n1 = res[(0.1, 0.1)]
    out["f1"] = charts.diagram(
        err,
        title="The error barely moves",
        subtitle=("Test misclassification rate of the same models from 30 "
                  "rounds on; dashed lines were trained with 10% of labels "
                  "flipped at random."),
        xlabel="boosting rounds (log scale)",
        ylabel="test error",
        source="Breast Cancer Wisconsin (scikit-learn); "
               "standarderror/ml/trees.py.",
        alt=("Solid lines falling and flattening near 0.04 to 0.05; dashed "
             "lines higher, near 0.07 to 0.10, rising slightly at the "
             "longest runs."),
        caption=(f"With clean labels the error at 2,000 rounds is the lowest "
                 f"it has been. With flipped labels it rises from "
                 f"{n1['error'].min():.3f} to {n1['error'][-1]:.3f} — under "
                 f"two points, while the log loss "
                 f"{n1['log_loss'][-1] / n1['log_loss'].min():.1f}-folds."),
        path=str(IMG / f"ml114-f1-error.{EXT}"))[0]

    rows = []
    for flip in (0.0, 0.1):
        for lr in RATES:
            r = res[(flip, lr)]
            i = int(np.argmin(r["log_loss"]))
            rows.append([f"{flip:.0%}", f"{lr:g}",
                         f"{r['best_round_median']:g}",
                         f"{r['log_loss'][i]:.3f}",
                         f"{r['log_loss'][-1]:.3f}",
                         f"{r['error'].min():.3f}", f"{r['error'][-1]:.3f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["labels flipped", "learning rate", "best round (median)",
                "best log loss", "log loss at 2,000", "best error",
                "error at 2,000"],
        title="Same data, two verdicts on 'overfitting'",
        subtitle=("Means over 10 half splits; the best round is the median "
                  "over splits of the round with the lowest test log loss."),
        source="Breast Cancer Wisconsin (scikit-learn); "
               "standarderror/ml/trees.py.",
        alt=("A six-row table: the best round grows about tenfold per "
             "tenfold drop in learning rate; log loss at 2,000 rounds is "
             "far above its best, error at 2,000 rounds close to its best."),
        caption=("Read the last two columns and boosting barely overfits. "
                 "Read the two before them and it overfits badly. Both are "
                 "true of the same models."),
        align="rrrrrrr",
        path=str(IMG / f"ml114-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res, rounds)
    return out


def _hero(res: dict, rounds):
    c = res[(0.0, 0.1)]

    def ll(panel, m):
        panel.plot(range(len(rounds)), c["log_loss"], lw=2.8,
                   color=m.series[1])

    def er(panel, m):
        panel.plot(range(len(rounds)), c["error"], lw=2.8, color=m.series[0])
        panel.set_ylim(0, 0.4)

    def shift(panel, m):
        panel.bar(range(3), [np.log10(res[(0.0, lr)]["best_round_median"])
                             + 0.3 for lr in RATES], color=m.series[2])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=14,
        headline="Accuracy holds; confidence runs away",
        panels=[(ll, f"{c['log_loss'][-1] / c['log_loss'].min():.1f}x",
                 "log loss, best to 2,000"),
                (er, f"{c['error'][-1]:.3f}", "error at 2,000 rounds"),
                (shift, "10x later", "per 10x less shrinkage")],
        note=("Run gradient boosting for 2,000 rounds on clean labels and "
              "its test error never turns up. Its test log loss turns after "
              "a few dozen. Shrinkage postpones the turn without lowering "
              "the bottom: early stopping has to watch the metric that "
              "matters."),
        alt=("Three hand-drawn frames: a U-shaped curve; a curve falling and "
             "flattening; three bars rising in steps."),
        path=str(IMG / f"ml114-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["clean"] = s.run("""
        from standarderror.ml import trees as tr

        # depth-3 trees, learning rate 0.1, 10 half splits, clean labels
        c = tr.boost_curves(lr=0.1, flip=0.0)
        for r, e, ll in zip(c["rounds"], c["error"], c["log_loss"]):
            print(f"round {r:>5,}  test error {e:.4f}  test log loss {ll:.3f}")
        """)
    out["noisy"] = s.run("""
        c = tr.boost_curves(lr=0.1, flip=0.1)   # 10% of labels flipped
        for r, e, ll in list(zip(c["rounds"], c["error"], c["log_loss"]))[3:]:
            print(f"round {r:>5,}  test error {e:.4f}  test log loss {ll:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    rounds = list(res[(0.0, 0.1)]["rounds"])
    c = {lr: res[(0.0, lr)] for lr in RATES}
    n = {lr: res[(0.1, lr)] for lr in RATES}
    c1 = c[0.1]
    best = {lr: c[lr]["log_loss"].min() for lr in RATES}
    br = {lr: c[lr]["best_round_median"] for lr in RATES}
    worse = c1["log_loss"][-1] / best[0.1]

    # The spine, asserted rather than trusted.
    for lr in (0.1, 0.01):
        e = c[lr]["error"]
        assert e[-1] <= e[3:].min() + 0.003          # error never turns up
        assert c[lr]["log_loss"][-1] > 1.3 * best[lr]
    assert worse > 2.0
    assert c1["error"][rounds.index(30)] > c1["error"][-1]
    assert br[1.0] < br[0.1] < br[0.01]
    assert 4 < br[0.1] / br[1.0] and 4 < br[0.01] / br[0.1] < 25
    assert abs(best[0.1] - best[0.01]) < 0.01 and best[1.0] > best[0.1]
    nn = n[0.1]
    assert nn["error"][-1] - nn["error"].min() < 0.02
    assert nn["log_loss"][-1] > 3 * nn["log_loss"].min()
    assert all(n[lr]["error"].min() > c[lr]["error"].min() for lr in RATES)

    post = Post(
        title=(f"{SERIES_TAG} 14: Boosting Overfits Its Probabilities Long "
               f"Before Its Accuracy"),
        slug="ml-14-boosting",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=14,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Whether boosting overfits depends on what you measure. "
                  "Over 2,000 rounds its test error never turns up; its test "
                  "log loss turns after a few dozen, and shrinkage only "
                  "postpones the turn."),
        summary=(
            f"Gradient-boosted trees on the breast-cancer data, ten half "
            f"splits, 2,000 rounds. With clean labels and a learning rate of "
            f"0.1 the test error falls to {c1['error'][-1]:.3f} and never "
            f"rises. The test log loss is lowest after about {br[0.1]:g} "
            f"rounds ({best[0.1]:.3f}) and {worse:.1f} times worse at 2,000. "
            f"Each tenfold cut in the learning rate moves the best round "
            f"roughly tenfold later ({br[1.0]:g}, {br[0.1]:g}, {br[0.01]:g}) "
            f"without lowering the best log loss. With 10% of training labels "
            f"flipped the error rises by {nn['error'][-1] - nn['error'].min():.3f}"
            f" and the log loss {nn['log_loss'][-1] / nn['log_loss'].min():.1f}"
            f"-fold."),
        tags=["machine-learning", "data-science", "statistics",
              "gradient-boosting", "calibration", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Breast Cancer Wisconsin (Diagnostic), as bundled with "
            "scikit-learn, CC BY 4.0; 10 stratified half splits; for the "
            "noisy runs, 10% of training labels flipped at random, test "
            "labels untouched.",
            "Machinery: `standarderror/ml/trees.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's GradientBoostingClassifier "
            "with depth-3 trees and staged predictions.",
            "Where this stops: Friedman, \"Greedy function approximation: a "
            "gradient boosting machine\", *Annals of Statistics* (2001), for "
            "shrinkage; Mease and Wyner, \"Evidence contrary to the "
            "statistical view of boosting\", *JMLR* (2008); Niculescu-Mizil "
            "and Caruana, \"Predicting good probabilities with supervised "
            "learning\", *ICML* (2005); Schapire, Freund, Bartlett and Lee, "
            "\"Boosting the margin\", *Annals of Statistics* (1998).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("splits from seeds 0 to 9, label flips from "
                            "seeds 1,000 to 1,009, boosting with "
                            "random_state equal to the split seed"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Run it long enough",
        f"""Every new tree in gradient boosting is fitted to what the current model still gets wrong. Run that long enough and the model will fit the training data perfectly, noise included, so the standard warning is that boosting overfits if left running and the number of rounds must be tuned or stopped early.

Whether it overfits depends on what "fits" means. A classifier makes two kinds of output: a decision, scored by the error rate, and a probability, scored by log loss. Here is one boosted model, depth-3 trees at the common learning rate of 0.1, scored both ways on held-out tumours after each checkpoint, averaged over ten half splits:

{snip['clean'].markdown()}

By the error rate there is nothing to stop: it falls to {c1['error'][-1]:.3f} and is still at its lowest after 2,000 rounds. By log loss the model peaked long ago. The log loss is lowest at about **{br[0.1]:g} rounds ({best[0.1]:.3f})** and is {c1['log_loss'][-1]:.3f} by the end — {worse:.1f} times worse.

The two numbers disagree because they look at different things. The error counts which side of 0.5 each prediction falls. Log loss charges for confidence: a tumour predicted at 0.999 benign that turns out malignant costs almost seven nats. As the rounds go on, the model pushes every training probability towards 0 or 1. On test tumours it mostly keeps the same answers, so the error stands still, but the few it gets wrong it now gets wrong with near certainty.""",
        figures=[figs["f0"]])

    post.add(
        "Shrinkage slows it; it does not stop it",
        f"""The usual remedy is shrinkage: scale each tree's contribution by a small learning rate so the model approaches the training data slowly. It works in the sense Friedman intended — the curves above are smoother and their bottoms wider — but look at where the bottoms are:

- learning rate 1: best log loss {best[1.0]:.3f}, after about {br[1.0]:g} rounds;
- learning rate 0.1: best {best[0.1]:.3f}, after about {br[0.1]:g};
- learning rate 0.01: best {best[0.01]:.3f}, after about {br[0.01]:g}.

Each tenfold cut in the learning rate moves the best round about tenfold later, and below 1 it does not lower the best log loss at all: {best[0.1]:.3f} against {best[0.01]:.3f}. A smaller learning rate buys a wider window in which stopping is safe and costs proportionally more trees. It does not make running forever safe. At 2,000 rounds the learning-rate-0.01 model is already past its best, at {c[0.01]['log_loss'][-1]:.3f}.

That is why early stopping is part of every serious boosting library, and why what it monitors matters. A validation set scored by accuracy, here, would never trigger it, and the model shipped would be the one with the worst probabilities of any checkpoint after the first few dozen.""",
        figures=[figs["f1"]])

    post.add(
        "What a stopping rule picks",
        f"""Early stopping takes the round at which a validation metric is best. Which metric it watches decides which model you ship. On the clean runs at learning rate 0.1, watching the error rate picks the last checkpoint, 2,000 rounds, because the error is still at its lowest there. That model's test log loss is {c1['log_loss'][-1]:.3f}. Watching log loss picks a model from around round {br[0.1]:g}, with test log loss {best[0.1]:.3f} and a test error of {c1['error'][rounds.index(30)]:.3f} at the nearby checkpoint of 30 rounds — {c1['error'][rounds.index(30)] - c1['error'][-1]:.3f} worse than the long run.

So the trade is real but lopsided: about a point of accuracy against a {worse:.1f}-fold difference in the quality of the probabilities. If the output is a decision at a fixed threshold, the long run is the better model. If the output is a probability — and most deployed classifiers are read as risks, ranked, or combined with costs — the short run is better by a wide margin, and only a stopping rule on log loss will find it.

Why does the error keep improving while confidence runs away? Schapire and colleagues' margin explanation is the usual answer: later rounds mostly push correctly classified training points further from the boundary, which changes probabilities a great deal and decisions very little. On test points near the boundary the boundary itself barely moves; on test points the model gets wrong, the pushed-out probabilities make each mistake more expensive.""")

    post.add(
        "Noisy labels",
        f"""Clean labels flatter the error rate: with nothing to memorise but the truth, fitting the training data harder cannot cost many decisions. Flip 10% of the training labels at random and boosting has real noise to chase:

{snip['noisy'].markdown()}

Now the error turns up as well, from {nn['error'].min():.3f} at its best to {nn['error'][-1]:.3f} — less than two points over nearly two thousand extra rounds. The log loss goes from {nn['log_loss'].min():.3f} to **{nn['log_loss'][-1]:.3f}**, more than three times worse. The flipped training labels are fitted as faithfully as the true ones, and every test tumour that resembles a flipped training tumour inherits a confident wrong answer.

Mease and Wyner made the accuracy half of this point in 2008: boosted classifiers often keep improving in error long after the statistical view says they should be overfitting. The probability half is the one that matters whenever the output is used as a risk — a credit default probability, a diagnosis likelihood, a click-through rate — because there the model's confidence is the product.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        """**One dataset.** Breast cancer is easy, with a test error under 5%. On harder data, where many test rows sit near the boundary, error turns up sooner and further; the ordering — log loss first, error later — is what generalises.

**Probabilities can be repaired afterwards.** Calibrating an overfit booster on held-out data (Platt scaling, isotonic regression) recovers much of the lost log loss, as Niculescu-Mizil and Caruana showed. It needs a held-out set, and it repairs the symptom, not the training.

**Checkpoints, not every round.** The curves are sampled at eight round counts. The best round quoted is computed from every round, per split, and its median is reported.

**Other boosters regularise more by default.** Libraries with leaf-weight penalties, row subsampling and minimum leaf sizes reach the same place more slowly. scikit-learn's classic implementation, with its defaults, is the plainest version of the algorithm.""")

    post.add(
        "What to keep",
        f"""1. With clean labels, test error of gradient boosting never turned up in 2,000 rounds: {c1['error'][-1]:.3f} at the end.

2. Test log loss was lowest after about {br[0.1]:g} rounds and {worse:.1f} times worse at 2,000. The model's decisions held; its confidence ran away.

3. Each tenfold cut in the learning rate moved the best round about tenfold later and left the best log loss where it was. Shrinkage widens the window; it does not remove the need to stop.

4. With 10% label noise the error rose by under two points and the log loss more than tripled.

5. Stop boosting on the metric the output will be judged by. If the probabilities are used, that is log loss or a proper score, not accuracy.""")

    post.add(
        "Exercise",
        """Take a boosted model you have in use and plot its staged validation error and staged validation log loss against the number of rounds on the same axes. Mark the round your early stopping chose and the round each curve is lowest.

If the two curves bottom out far apart, find out which one your stopping rule watched. If the model's probabilities are used downstream, refit with stopping on log loss and compare the calibration of the two models on held-out data.""")

    post.add(
        "Next",
        """Episode 15 closes Arc III with the one thing no number of trees or rounds can fix: what a tree ensemble predicts outside the range of the data it was trained on.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
