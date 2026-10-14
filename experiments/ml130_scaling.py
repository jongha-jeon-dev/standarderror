"""ML 30: A Power Law Fitted on Small Models Cannot See the Floor.

The thirtieth episode of *Machine Learning, Taught Through What Breaks*,
closing Arc VI. "Scaling laws predict performance." Measured on a ladder of
eight character-level transformers trained for the same 2,000 steps (4.1
million characters, sampled from the same million-character Shakespeare
text):

* A power law fitted to the five smallest models' validation loss predicts
  the sixth, 2.2 times the size of the fifth, to within 0.1%.
* It predicts the seventh and eighth too optimistically, by 3.4% and 8.2%
  (the fitted models scatter about the line by up to 3%): the losses bend
  away from the line as the number of characters seen per parameter falls
  from about a thousand to two. A law fitted to the three smallest hits the
  largest only because two errors cancel.
* Fitting a floor to the five small models finds none. The floor the large
  models run into is consistent with the fixed data and steps (the budget
  was not varied), and the small models never reach it.

Run: `standarderror run ml130_scaling --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import scaling as sc
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

K = 5          # models the law is fitted on


def compute() -> dict:
    r = sc.results()
    runs = r["runs"]
    N = np.array([x["non_embedding"] for x in runs], float)
    V = np.array([x["final_val"] for x in runs])
    T = np.array([x["final_train"] for x in runs])
    pv = sc.fit(N[:K], V[:K], floor=False)
    pt = sc.fit(N[:K], T[:K], floor=False)
    pf = sc.fit(N[:K], V[:K])
    pall = sc.fit(N, V)
    pred = sc.predict(pv, N)
    err = V / pred - 1
    # the same law fitted to the k smallest, for several k: its error on
    # every model of the ladder
    profiles = {}
    for k in (3, 5, 7):
        p = sc.fit(N[:k], V[:k], floor=False)
        profiles[k] = {"b": p["b"], "err": V / sc.predict(p, N) - 1}
    return {"r": r, "runs": runs, "N": N, "V": V, "T": T, "pv": pv,
            "pt": pt, "pf": pf, "pall": pall, "pred": pred, "err": err,
            "tpe": r["tokens"] / N, "profiles": profiles,
            "hours": sum(x["seconds"] for x in runs) / 3600}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    N, V, T, pv, pt = res["N"], res["V"], res["T"], res["pv"], res["pt"]

    def law(ax, m):
        x = np.geomspace(N[0] * 0.8, N[-1] * 1.25, 100)
        ax.plot(x, sc.predict(pv, x), color=m.series[0], lw=1.2, ls="--",
                label="power law fitted to the five smallest, validation")
        ax.plot(x, sc.predict(pt, x), color=m.series[1], lw=1.2, ls="--",
                label="the same, training loss")
        ax.scatter(N[:K], V[:K], color=m.series[0], s=30, zorder=3,
                   label="validation loss, fitted")
        ax.scatter(N[K:], V[K:], color=m.series[0], s=50, marker="D",
                   facecolors="none", zorder=3,
                   label="validation loss, not fitted")
        ax.scatter(N, T, color=m.series[1], s=18, zorder=3,
                   label="training loss")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8, loc="lower left")

    i_last = len(N) - 1
    out["f0"] = charts.diagram(
        law,
        title="The line the small models draw",
        subtitle=("Final loss of eight character transformers against their "
                  "non-embedding parameters, all trained for 2,000 steps on "
                  "the same text. The law L = a N^-b is fitted to the five "
                  "smallest."),
        xlabel="non-embedding parameters (log scale)",
        ylabel="loss, nats per character (log scale)",
        figsize=(7.4, 4.8),
        source=("tinyshakespeare (public domain); "
                "standarderror/ml/scaling.py, data/ml_scaling."),
        alt=("Points on log-log axes falling along a dashed straight line for "
             "the five smallest models; the next point sits on the line and "
             "the two largest sit above it."),
        caption=(f"The sixth model lands on the line ({res['err'][K]:+.1%}). "
                 f"The largest, {N[-1] / N[K - 1]:.0f} times the fifth, is "
                 f"{res['err'][i_last]:+.1%} off."),
        path=str(IMG / f"ml130-f0-law.{EXT}"))[0]

    prof = res["profiles"]

    def profile(ax, m):
        for c, (k, q) in enumerate(sorted(prof.items())):
            ax.plot(N, q["err"], color=m.series[c], lw=1.4,
                    label=f"fitted to the {k} smallest")
            ax.scatter(N[:k], q["err"][:k], color=m.series[c], s=18)
            ax.scatter(N[k:], q["err"][k:], color=m.series[c], s=40,
                       marker="D", facecolors="none")
        ax.axhline(0, color=m.ink, lw=0.8)
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    out["f1"] = charts.diagram(
        profile,
        title="Three fits, three stories",
        subtitle=("Measured validation loss over the law's value, minus one, "
                  "for the law fitted to the 3, 5 or 7 smallest models. Dots "
                  "were fitted; open diamonds were predicted."),
        xlabel="non-embedding parameters (log scale)",
        ylabel="measured / predicted − 1",
        figsize=(7.4, 4.6),
        source=("tinyshakespeare (public domain); "
                "standarderror/ml/scaling.py."),
        alt=("Three lines of residuals: the three-model fit dips below zero "
             "through the middle of the ladder and returns to zero at the "
             "end; the five- and seven-model fits stay near zero and then "
             "rise above it for the largest models."),
        caption=(f"Fitted to three models, the law is too pessimistic for the "
                 f"middle of the ladder and lands on the largest by "
                 f"cancellation ({prof[3]['err'][-1]:+.1%}). Fitted to seven, "
                 f"it misses the eighth by {prof[7]['err'][-1]:+.1%}."),
        path=str(IMG / f"ml130-f1-profiles.{EXT}"))[0]

    rows = []
    for i, x in enumerate(res["runs"]):
        rows.append([f"{x['width']} x {x['layers']}", f"{x['params']:,}",
                     f"{int(N[i]):,}", f"{res['tpe'][i]:,.0f}",
                     f"{T[i]:.3f}", f"{V[i]:.3f}",
                     f"{res['pred'][i]:.3f}",
                     ("fitted " if i < K else "") + f"{res['err'][i]:+.1%}"])
    out["f2"] = charts.table_image(
        rows,
        header=["width x layers", "parameters", "non-embedding",
                "tokens per parameter", "training loss", "validation loss",
                "law's value", "error"],
        title="Eight models, one budget",
        subtitle=("Each model: 2,000 AdamW steps of 32 x 64 characters "
                  "(4.1 million tokens), warm-up and cosine decay, learning "
                  "rate 3e-3, one seed. Losses on fixed batches of the "
                  "training and validation text."),
        source=("tinyshakespeare (public domain); "
                "standarderror/ml/scaling.py, data/ml_scaling."),
        alt=("An eight-row table: tokens per parameter fall from about 950 "
             "to about 2; the law's error is small for the first six models "
             "and grows for the last two."),
        caption=("The first five rows fix the law; the last three test it. "
                 "Tokens per parameter is the number of training characters "
                 "seen per non-embedding parameter."),
        bold_cells={(len(rows) - 1, 7), (len(rows) - 2, 7)},
        align="lrrrrrrr",
        path=str(IMG / f"ml130-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    N, V, pv = res["N"], res["V"], res["pv"]

    def fit(panel, m):
        panel.plot(np.log(N), np.log(sc.predict(pv, N)), color=m.series[0],
                   lw=1.5, ls="--")
        panel.scatter(np.log(N[:K]), np.log(V[:K]), color=m.series[0], s=20)

    def hit(panel, m):
        panel.plot(np.log(N), np.log(sc.predict(pv, N)), color=m.series[0],
                   lw=1.5, ls="--")
        panel.scatter(np.log(N[:K + 1]), np.log(V[:K + 1]),
                      color=m.series[2], s=20)

    def miss(panel, m):
        panel.plot(np.log(N), np.log(sc.predict(pv, N)), color=m.series[0],
                   lw=1.5, ls="--")
        panel.scatter(np.log(N), np.log(V), color=m.series[1], s=20)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=30,
        headline="The law cannot see the floor",
        panels=[(fit, f"b = {pv['b']:.3f}", "fitted on five models"),
                (hit, f"{res['err'][K]:+.1%}", "next size, predicted"),
                (miss, f"{res['err'][-1]:+.1%}", "largest, predicted")],
        note=("A power law fitted to small models predicts the next size "
              "almost exactly and the largest too optimistically: the curve "
              "bends, as data and steps per parameter run out, where the "
              "small models never go."),
        alt=("Three hand-drawn frames: points on a dashed line; one more "
             "point on the line; the last points rising above it."),
        path=str(IMG / f"ml130-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["ladder"] = s.run("""
        from standarderror.ml import scaling as sc

        r = sc.results()     # written by scripts/train_scaling.py
        for run in r["runs"]:
            print(f"{run['width']:3d} wide x {run['layers']} layers: "
                  f"{run['non_embedding']:7,d} parameters, training loss "
                  f"{run['final_train']:.3f}, validation {run['final_val']:.3f}")
        """)
    out["fit"] = s.run(f"""
        small = r["runs"][:{K}]
        law = sc.fit([x["non_embedding"] for x in small],
                     [x["final_val"] for x in small], floor=False)
        print(f"L = {{law['a']:.3f}} N^-{{law['b']:.4f}}")
        for run in r["runs"][{K}:]:
            guess = sc.predict(law, run["non_embedding"])
            print(f"{{run['non_embedding']:7,d}} parameters: predicted "
                  f"{{guess:.3f}}, measured {{run['final_val']:.3f}}")
        """)
    out["floor"] = s.run(f"""
        floor = sc.fit([x["non_embedding"] for x in small],
                       [x["final_val"] for x in small])        # a N^-b + c
        everyone = sc.fit([x["non_embedding"] for x in r["runs"]],
                          [x["final_val"] for x in r["runs"]])
        print(f"floor fitted to the {K} smallest: c = {{floor['c']:.3f}}")
        print(f"floor fitted to all {{len(r['runs'])}}:     c = {{everyone['c']:.3f}}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    N, V, T, err, pv, pt, pf, pall = (res["N"], res["V"], res["T"],
                                      res["err"], res["pv"], res["pt"],
                                      res["pf"], res["pall"])
    tpe, prof = res["tpe"], res["profiles"]
    gap = V - T
    errt = T / sc.predict(pt, N) - 1
    n = len(N)

    # The spine, asserted rather than trusted.
    assert n == 8 and res["r"]["steps"] == 2000
    assert abs(err[K]) < 0.005
    assert err[K + 1] > 0.02 and err[K + 2] > err[K + 1]
    assert errt[K + 1] > 0.02 and errt[K + 2] > errt[K + 1]
    assert pf["c"] < 0.05 and pall["c"] > 0.3
    assert abs(prof[3]["err"][-1]) < 0.01 and min(prof[3]["err"][3:7]) < -0.04
    assert prof[7]["err"][-1] > 0.05
    assert abs(prof[5]["b"] - prof[3]["b"]) > 0.01
    assert tpe[0] > 900 and tpe[-1] < 3
    assert all(V[i] > V[i + 1] for i in range(n - 1))
    assert gap[-1] < 0.3
    # The fitted models' own scatter about the line: the seventh's miss is
    # comparable to it, only the eighth's clearly exceeds it.
    in_max = float(np.max(np.abs(err[:K])))
    assert 0.025 < in_max < 0.035
    assert err[K + 1] < 1.5 * in_max and err[K + 2] > 2 * in_max
    # Only the largest model has more non-embedding parameters than the
    # roughly one million training characters.
    assert N[-1] > 1.1e6 and N[-2] < 0.9e6
    # The five-model floor sits at the fit's lower bound of zero.
    assert pf["c"] < 5e-4

    post = Post(
        title=(f"{SERIES_TAG} 30: A Power Law Fitted on Small Models Cannot "
               f"See the Floor"),
        slug="ml-30-scaling",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=30,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Eight character transformers, one budget of data and "
                  "steps. The law fitted to the five smallest predicts the "
                  "next size almost exactly and the largest too "
                  "optimistically."),
        summary=(
            f"Eight character-level transformers from {int(N[0]):,} to "
            f"{int(N[-1]):,} non-embedding parameters were trained for the same "
            f"2,000 steps on the same million characters of Shakespeare (4.1 "
            f"million characters seen). A power "
            f"law L = {pv['a']:.2f} N^-{pv['b']:.3f} fitted to the five "
            f"smallest models' validation loss predicted the sixth to "
            f"{err[K]:+.2%}, the seventh to {err[K + 1]:+.1%} and the eighth "
            f"to {err[K + 2]:+.1%}, each measured loss higher than predicted. "
            f"Training loss bent the same way, so the miss is not mainly "
            f"overfitting. A floor fitted to the five small models came out "
            f"at {pf['c']:.2f}; fitted to all eight, at {pall['c']:.2f}. The "
            f"bend is consistent with the fixed data and training budget, "
            f"which was not varied, and the small models never come near "
            f"it."),
        tags=["machine-learning", "deep-learning", "scaling-laws",
              "language-models", "extrapolation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "tinyshakespeare (Karpathy's char-rnn collection of Shakespeare, "
            "public domain), fetched and checked against a SHA-256 digest by "
            "`standarderror/llm/tiny.py`, not committed; the first 90% for "
            "training, the last 10% for validation.",
            "Machinery: `standarderror/ml/scaling.py` and "
            "`scripts/train_scaling.py`, whose curves are committed in "
            "`data/ml_scaling/results.json`; tested in `tests/test_ml.py`. "
            "Pre-norm transformers with 4 heads, context 64, widths 16 to "
            "192 and 1 to 4 layers; AdamW, learning rate 3e-3 with warm-up "
            "and cosine decay, weight decay 0.1, batches of 32, 2,000 steps, "
            "seed 0.",
            "Where this stops: Kaplan et al., \"Scaling laws for neural "
            "language models\" (2020); Hoffmann et al., \"Training "
            "compute-optimal large language models\", *NeurIPS* (2022); "
            "episode 4 of this series, on learning curves fitted early.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time against the committed "
                            "training curves; the values the prose quotes "
                            "are pinned"),
            "determinism": ("one seed per model, CPU training on two "
                            "threads; the committed curves record each "
                            "model's training time"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The bet behind large models",
        f"""Episode 29 measured what a model loses when its inputs move away from its training data. The last episode of Arc VI measures a more ambitious extrapolation: from small models to large ones. The sentence under test is the one that justifies building large language models at all: *scaling laws predict performance*. Kaplan and colleagues found in 2020 that a language model's loss falls as a power of its parameter count, L = a N^(-b), over many orders of magnitude, so that a line drawn through cheap small models says what an expensive large one will achieve before it is trained.

The ladder here is eight character-level transformers trained on Shakespeare, from {int(N[0]):,} to {int(N[-1]):,} parameters outside the embeddings (the count Kaplan's law uses). Every one gets the same recipe: 2,000 steps of AdamW with the same schedule over the same million-character training text, 4.1 million characters in all, about four passes. The ladder took {res['hours']:.1f} hours on two CPU threads, so its results are committed to the repository and read back here:

{snip['ladder'].markdown()}

Validation loss falls with every step up the ladder, from {V[0]:.3f} to {V[-1]:.3f} nats per character.""")

    post.add(
        "Five models, one line",
        f"""Fit the law to the five smallest models, then ask it about the three it has not seen:

{snip['fit'].markdown()}

The exponent is {pv['b']:.3f}. The sixth model, {N[K] / N[K - 1]:.1f} times the size of the fifth, comes in at **{err[K]:+.2%}** from the prediction: the law did exactly what it is famous for. The seventh, {N[K + 1] / N[K - 1]:.1f} times the fifth, is {err[K + 1]:+.1%} off; the eighth, {N[K + 2] / N[K - 1]:.0f} times, is **{err[K + 2]:+.1%}** off. Every miss is in the same direction: the large models are worse than the line promised.

A more cautious analyst would fit on fewer models and check on more, and might be reassured. Fitted to the three smallest models, the law's exponent is {prof[3]['b']:.3f}, and its prediction for the largest model, {N[-1] / N[2]:.0f} times the third, is off by only **{prof[3]['err'][-1]:+.1%}**. But the second figure shows how it got there. For the four models in between, the three-model law was too *pessimistic*, by as much as {-min(prof[3]['err'][3:7]):.1%}: its slope was too shallow for the middle of the ladder, and the bend at the top happened to cancel it. Fitted to seven models, the law misses the eighth, only {N[-1] / N[-2]:.1f} times the seventh, by {prof[7]['err'][-1]:+.1%}.

One correct prediction is not evidence that the law holds. The pattern of residuals is. The three-model fit's form a U, below the line through the middle of the ladder and back at the top; the five- and seven-model fits scatter by a few percent about the line through the fitted models (up to {in_max:.1%} for the five) and then rise sharply above it at the top. No straight line on log-log axes can follow that bend. The scatter matters too: the seventh model's {err[K + 1]:+.1%} is comparable to it, and only the eighth's {err[K + 2]:+.1%} clearly exceeds it.""",
        figures=[figs["f0"], figs["f1"]])

    post.add(
        "Not mainly overfitting",
        f"""The obvious suspect is overfitting. The largest model has more non-embedding parameters, {int(N[-1]):,}, than the training text has characters to constrain them, about a million, and the gap between training and validation loss does widen up the ladder, from {gap[0]:.3f} for the smallest model to {gap[-1]:.3f} for the largest.

But the same law fitted to training loss misses in the same direction: the seventh and eighth models' training losses are {errt[K + 1]:+.1%} and {errt[K + 2]:+.1%} above what the five smallest predict. A model that was merely overfitting would do *better* than the line on the data it trained on. These do worse on both. They are not learning as much per parameter as the small ones did, on any data.

The table suggests why. The number of training characters seen per non-embedding parameter falls from {tpe[0]:,.0f} for the smallest model to **{tpe[-1]:.1f}** for the largest. Hoffmann and colleagues found that a compute-efficient language model sees on the order of twenty tokens per parameter; their tokens are subwords and these are characters, so the comparison is only rough. The small models here are trained far past that point; the two largest, at {tpe[-2]:.1f} and {tpe[-1]:.1f}, have been given the same budget and stopped well short of it. Kaplan's law in N describes models that are not limited by data or training; the ladder's largest rungs are plausibly limited by both, though the budget was never varied to show it.""",
        figures=[figs["f2"]])

    post.add(
        "The floor the small models cannot see",
        f"""The standard repair, which episode 4 tried on learning curves, is to fit a floor: L = a N^(-b) + c, where c is the loss no model size can get below with this data and this budget. Fitted to the five small models and to all eight:

{snip['floor'].markdown()}

From the five small models the fitted floor is {pf['c']:.3f}, the lowest value the fit allows: in their range the curve is still a straight line on log-log axes, and there is nothing to estimate a floor from. With all eight, the floor is {pall['c']:.2f}, and the curve bends towards it. The information that the curve bends lives only in the models that have already bent, which are exactly the models the law was supposed to make unnecessary.

That is episode 4's finding, met again at a larger scale. There a learning curve fitted early was optimistic in all nine fits, and adding a floor fixed only the curves that had already bent. Here the law fitted to five small models was optimistic about every model past the sixth, and the floor could not be seen from them. The arithmetic is the same: a power law extrapolated past the range that determined it assumes nothing new happens there, and in a fixed budget of data and steps, something eventually does.""")

    post.add(
        "Where this breaks",
        """**One budget for every model.** The laws used to plan large models are fitted to models that are each trained with enough data and compute for their size, or along a compute-optimal frontier. This ladder holds the budget fixed on purpose, which is the setting in which a law in N alone is misspecified. A ladder with data growing alongside the models would be expected to bend later, or not at all in this range; that was not run.

**Small models, one seed each.** From four thousand to two million non-embedding parameters, each trained once. The run-to-run spread was not measured; the sixth model's agreement with the line, to a fraction of a percent, may be partly luck, and so may the seventh's miss, which is about the size of the scatter of the fitted models. The eighth's miss is more than twice that scatter, which seed noise of that size would be unlikely to produce.

**One learning rate.** Every model used 3e-3. Larger models often want smaller rates, and part of the shortfall at the top of the ladder may be a rate that suits the small models better.

**Character-level Shakespeare.** A million characters of one author is a narrow distribution with a low ceiling on what any model can learn from it. Kaplan's laws were measured on billions of tokens of web text, where the floor is far away.""")

    post.add(
        "What to keep",
        f"""1. A power law fitted to five small models predicted the next size to {err[K]:+.2%}. Scaling laws do predict, over a short reach.

2. Over a longer reach they were optimistic: {err[K + 1]:+.1%} for a model {N[K + 1] / N[K - 1]:.1f} times the largest fitted, {err[K + 2]:+.1%} for one {N[K + 2] / N[K - 1]:.0f} times. A law fitted to three models hit the largest by cancelling two errors, and one fitted to seven missed it by {prof[7]['err'][-1]:+.1%}: judge a law by its residuals, not by one hit.

3. Training loss missed the same way, so the cause was not mainly overfitting. With a fixed budget, characters per parameter fell from {tpe[0]:,.0f} to {tpe[-1]:.1f}, consistent with the largest models being limited by data and steps rather than size (the budget itself was not varied).

4. A floor fitted to the small models was {pf['c']:.3f}; to all eight, {pall['c']:.2f}. Extrapolate a fitted law only as far as the conditions that produced it still hold.""")

    post.add(
        "Exercise",
        """Take three or four sizes of a model you can train cheaply, each with the same data and the same number of steps, and fit L = a N^-b to their validation loss. Before training the next size up, write down the prediction.

Then train it, and a size beyond. If the second misses by more than the first, compute the training tokens per parameter for each model: the point where the curve leaves the line is usually where that number stops being large.""")

    post.add(
        "Next",
        """Arc VI ends here, and with it the episodes about models themselves. Thirty sentences were measured; a few survived as stated, most survived only with a condition attached, and some did not survive at all. Each condition is a place where a number that looks like a property of a model turned out to be a property of the model together with its data, its training and the question asked of it. Arc VII turns to the routine steps around a model, starting with the one most often applied before it: reducing the inputs with principal components.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
