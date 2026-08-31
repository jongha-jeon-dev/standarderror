"""exp003 — A light post: the 99% R² that means nothing.

Backlog E5. Short, no code blocks, no equations, one idea: a forecast of a
provably unpredictable series scores R² = 0.99, so R² on a slow-moving series
cannot tell skill from arithmetic. MASE against the naive forecast can.

The twist that keeps it honest: "just difference it and report R² on the change"
is *also* wrong, and there is a clean counterexample.

Run: `standarderror run exp003_persistence --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.dynamics import ode
from standarderror.models import baselines, metrics
from standarderror.render import Post
from standarderror.viz import charts, theme

#: Pinned so a rebuild cannot silently re-date a published post.
#: `Post.date` defaults to today, which is correct exactly once.
POST_DATE = date(2026, 8, 6)

IMG = se.SETTINGS.build_dir / "img"
# PNG for the site and for Medium; SVG when a target wants inline vector text
# (Notion accepts SVG as an inline attachment, which avoids needing a public URL).
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed
N = 1500


def make_series(kind: str, n: int = N, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "coin-flip random walk":
        return np.cumsum(rng.standard_normal(n))
    if kind == "slow drifting series":
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.99 * x[t - 1] + rng.standard_normal()
        return x
    if kind == "chaotic but deterministic":
        return ode.lorenz63(n_steps=n, dt=0.02, transient=40.0).x[:, 0]
    if kind == "pure noise":
        return rng.standard_normal(n)
    raise KeyError(kind)


def _aligned_one_step(x: np.ndarray, n_tr: int, n_lags: int) -> dict:
    """One-step-ahead forecasts of x[n_tr:], the model and the naive rule scored
    against *identical* targets.

    Index arithmetic written out because getting it wrong is the whole hazard here.
    `LinearAR.predict_teacher_forced(U)` with `n_lags = s` returns one prediction
    per window of `s` consecutive rows of `U`, and window j predicts the value that
    comes *after* `U[j+s-1]`. Feeding `U = x[n_tr-s : -1]` therefore yields
    predictions of exactly `x[n_tr], x[n_tr+1], ...`.

    Trimming from the end instead — comparing against `x[n_tr+1:]` — shifts every
    prediction one step and inflates the error. On a random walk it turned MASE
    0.98 into 1.39, which reads as "the model is 40% worse than doing nothing"
    rather than "the model does nothing", and it is a completely different claim.
    """
    col = x[:, None]
    model = baselines.LinearAR(n_lags=n_lags).fit(col[:n_tr - 1], col[1:n_tr])
    pred = model.predict_teacher_forced(col[n_tr - n_lags:-1]).ravel()
    truth = x[n_tr:]
    naive = x[n_tr - 1:-1]
    if not (len(pred) == len(truth) == len(naive)):
        raise AssertionError(
            f"alignment: pred {len(pred)}, truth {len(truth)}, naive {len(naive)}")
    return {"truth": truth, "pred": pred, "naive": naive}


def score(x: np.ndarray) -> dict:
    """Score a 4-lag linear model and the naive forecast on one series.

    Fitted on the first half, scored on the second, and the naive scale for MASE
    comes from the training half only — the point of the post would collapse if the
    evaluation leaked.
    """
    n_tr = len(x) // 2
    lv = _aligned_one_step(x, n_tr, 4)
    dv = _aligned_one_step(np.diff(x), len(np.diff(x)) // 2, 4)

    mase_naive = metrics.mase(lv["truth"], lv["naive"], x[:n_tr])
    # Free alignment check: the naive forecast's MASE is, by definition, the ratio
    # of mean absolute step size on test to that on train. For any series whose
    # step size is roughly stable it must sit near 1. A value far from 1 means the
    # targets and the predictor are not lined up.
    if not 0.5 < mase_naive < 2.0:
        raise AssertionError(
            f"naive MASE {mase_naive:.3f} is implausible — check the alignment")

    mase_model = metrics.mase(lv["truth"], lv["pred"], x[:n_tr])
    return {
        # Ratio, not raw MASE. Both share the training-set scale factor, so
        # dividing cancels it and 1.0 means *exactly* "the same error as assuming
        # nothing changes" — which is the sentence the reader needs.
        "vs_naive": float(mase_model / mase_naive),
        "r2_level_model": metrics.r2(lv["truth"], lv["pred"]),
        "r2_level_naive": metrics.r2(lv["truth"], lv["naive"]),
        "r2_change_model": metrics.r2(dv["truth"], dv["pred"]),
        "mase_model": mase_model,
        "mase_naive": mase_naive,
        "truth": lv["truth"], "pred": lv["pred"], "n_train": n_tr,
        "dtruth": dv["truth"], "dpred": dv["pred"],
    }


def differencing_trap() -> dict:
    """Difference a series that did not need it, and skill appears from nowhere.

    Taking first differences of white noise creates an MA(1) with autocorrelation
    exactly −0.5, which a linear model predicts happily. The R² on the differenced
    series looks like a result. Reconstruct the level and it buys nothing.
    """
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.standard_normal(n)
    d = np.diff(x)
    lag1 = float(np.corrcoef(d[1:], d[:-1])[0, 1])

    n_tr = len(d) // 2
    dv = _aligned_one_step(d, n_tr, 4)
    # d[t] = x[t+1] - x[t], so a forecast of d[t] plus the known x[t] is a forecast
    # of x[t+1]. dv["truth"] covers d[n_tr:], hence levels x[n_tr+1:].
    k = len(dv["pred"])
    level_pred = x[n_tr:n_tr + k] + dv["pred"]
    level_true = x[n_tr + 1:n_tr + 1 + k]
    if len(level_true) != k:
        raise AssertionError("level reconstruction misaligned")
    return {
        "lag1_autocorr": lag1,
        "r2_on_differences": metrics.r2(dv["truth"], dv["pred"]),
        "r2_reconstructed_level": metrics.r2(level_true, level_pred),
        "mase_reconstructed": metrics.mase(level_true, level_pred, x[:n_tr]),
        "mase_naive": metrics.mase(level_true, x[n_tr:n_tr + k], x[:n_tr]),
    }


def figures(res: dict, trap: dict) -> dict:
    src = "Simulated series; code and seed in the repo."
    figs = {}
    rw = res["coin-flip random walk"]

    # F1 — the illusion. Show only a window, because at 750 points the two lines
    # are one line and the reader cannot see that there are two.
    w = slice(0, 220)
    frame = pd.DataFrame(
        {"what actually happened": rw["truth"][w],
         "what the model predicted": rw["pred"][w]})
    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False,
        title="A forecast of something nobody can forecast",
        subtitle=(f"This series is cumulative coin flips. There is nothing in it "
                  f"to predict. The model scores R² = "
                  f"{rw['r2_level_model']:.4f}."),
        ylabel="value", xlabel="step", source=src,
        alt=("Line chart with two nearly identical lines, one for the actual "
             "series and one for the model's forecast, tracking each other so "
             "closely they are hard to tell apart."),
        caption=("Fig 1. Two lines. You can barely see that there are two. If "
                 "someone showed you this chart and an R² of 0.99, you would "
                 "believe the model worked — and the series is a coin flip."),
        path=str(IMG / f"e5-f1-illusion.{EXT}"))
    figs["illusion"] = fig_meta

    # F2 — the same forecast, on the quantity that was actually up for grabs.
    dframe = pd.DataFrame(
        {"what actually happened": rw["dtruth"][:220],
         "what the model predicted": rw["dpred"][:220]})
    fig_meta, _ = charts.lines(
        dframe, mode="light", direct_labels=False,
        title="The same forecast, showing the part it had to get right",
        subtitle=(f"Now plotting the step-to-step change instead of the level. "
                  f"R² = {rw['r2_change_model']:.4f} — the model has learned "
                  "nothing, and always guesses roughly zero."),
        ylabel="change from previous step", xlabel="step", source=src,
        alt=("Line chart where one line jumps up and down erratically while the "
             "other stays almost flat near zero, showing the forecast captures "
             "none of the variation."),
        caption=("Fig 2. The forecast is the flat line. All of that jagged "
                 "movement is what the model was supposed to predict, and it "
                 "predicted none of it. Same model, same data, same day — only "
                 "the axis changed."),
        path=str(IMG / f"e5-f2-truth.{EXT}"))
    figs["truth"] = fig_meta

    # F3 — one number that is not fooled.
    names = list(res)
    vals = [res[k]["vs_naive"] for k in names]

    def mark_one(_fig, ax):
        m = theme.LIGHT
        ax.axvline(1.0, color=m.series[7], lw=1.4)
        # Anchored in axes fraction on y: a data-space y computed from the bar
        # count lands outside the limits and gets silently clipped.
        ax.annotate("1.0 = exactly as good as\nassuming nothing changes",
                    (1.0, 0.08), xycoords=("data", "axes fraction"),
                    xytext=(7, 0), textcoords="offset points", fontsize=8.0,
                    color=m.series[7], va="bottom", ha="left")

    fig, ax = charts.ranked_bars(
        names, vals, mode="light",
        title="The number that is not fooled",
        subtitle=("The model's average error divided by the lazy forecast's. "
                  "Below 1, the model added something. At 1, it did not."),
        xlabel="error relative to \u201cassume no change\u201d", source=src)
    mark_one(fig, ax)
    theme.save(fig, str(IMG / f"e5-f3-mase.{EXT}"), mode="light")
    figs["mase"] = charts.Figure(
        str(IMG / f"e5-f3-mase.{EXT}"),
        alt=("Horizontal bar chart of forecast error relative to a naive forecast "
             "for four kinds of series, with a vertical reference line at 1.0. Two "
             "bars sit essentially on the line, one is well below it, and one is "
             "very close to zero."),
        caption=(f"Fig 3. Same model, four kinds of series. On the random walk it "
                 f"lands at {res['coin-flip random walk']['vs_naive']:.3f} and on "
                 f"the drifting series {res['slow drifting series']['vs_naive']:.3f} "
                 f"— indistinguishable from doing nothing. On the deterministic "
                 f"chaotic series it reaches "
                 f"{res['chaotic but deterministic']['vs_naive']:.3f}, about "
                 f"{1 / res['chaotic but deterministic']['vs_naive']:.0f} times "
                 "more accurate than laziness. That is what real skill looks like, "
                 "and R\u00b2 could not tell any of these apart."),
        title="mase")

    # F4 — the same table as an image, for Medium, which strips table markup.
    order = ["coin-flip random walk", "slow drifting series",
             "chaotic but deterministic", "pure noise"]
    rows = [[k, f"{res[k]['r2_level_model']:.4f}", f"{res[k]['vs_naive']:.3f}"]
            for k in order]
    fig_meta, _ = charts.table_image(
        rows,
        header=["series", "R\u00b2 on the level",
                "error vs \u201cassume no change\u201d"],
        # Bold only the three rows the argument turns on; bolding the control too
        # would suggest it is part of the finding.
        bold_cells={(0, 2), (1, 2), (2, 2)},
        title="Two scores, same data, opposite conclusions",
        subtitle=("The first column cannot tell the first three rows apart. The "
                  "second column can."),
        source=src, mode="light",
        alt=("Table of four series. R\u00b2 on the level is 0.9934, 0.9874, "
             "1.0000 and -0.0034; error relative to assuming no change is 1.003, "
             "1.010, 0.019 and 0.711."),
        caption=("Table 1. R\u00b2 says the first three are equally good models. "
                 "The comparison against laziness says two of them are worthless "
                 "and one is excellent."),
        path=str(IMG / f"e5-t1-scores.{EXT}"))
    figs["table"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    kinds = ("coin-flip random walk", "slow drifting series",
             "chaotic but deterministic", "pure noise")
    res = {k: score(make_series(k)) for k in kinds}
    trap = differencing_trap()
    figs = figures(res, trap)

    rw = res["coin-flip random walk"]
    slow = res["slow drifting series"]
    chaos = res["chaotic but deterministic"]
    noise = res["pure noise"]

    post = Post(
        title="Your Forecast Is Probably Just Repeating Yesterday",
        slug="your-forecast-is-probably-just-repeating-yesterday",
        date=POST_DATE,
        subtitle=("A model scores 99% on a series of coin flips. Here is why, and "
                  "the one number that would have caught it"),
        summary=(f"I built a forecast for a series that is provably impossible to "
                 f"forecast — cumulative coin flips — and it scored R² = "
                 f"{rw['r2_level_model']:.2f}. The chart looks superb. The model "
                 f"knows nothing. This happens constantly, it is not a trick, and "
                 "one substitution fixes it."),
        tags=["data-science", "statistics", "machine-learning", "forecasting",
              "analytics"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=850, max_words=1600,
        # Medium strips table markup on paste; the crosspost swaps in this image.
        table_figures=[figs["table"]],
        data_sources=[
            "Four simulated series (random walk, slow autoregressive drift, "
            "Lorenz-63, white noise). No external data; every number is "
            "reproducible from the repo with a fixed seed.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "standarderror")),
            "protocol": "fit on the first half, score on the second; the naive "
                        "scale for MASE comes from the training half only",
            "series length": N,
        },
    )

    post.add("A chart that should make you suspicious", f"""
Here is a forecast. The predicted line sits on top of what actually happened so
closely that you have to look twice to see there are two lines at all. The standard
accuracy score, R², comes out at **{rw['r2_level_model']:.4f}**. In most rooms, that
chart ends the conversation.

The series is cumulative coin flips.

I generated it by adding a random number to the previous value, over and over. There
is no pattern in it. There cannot be a pattern in it — I wrote the code that made
it, and the next step is independent of everything that came before. Any forecast of
tomorrow's value is worthless by construction.

And yet the score is 0.99, and the chart is beautiful. Something is clearly wrong,
and it is not the model. It is the question we asked it.
""".strip(), figures=[figs["illusion"]])

    post.add("The model was asked an easy question", f"""
R² measures how much of the *variation* in a series your forecast explains. On a
series like this one, almost all of the variation is the fact that the value today
is nowhere near where it was two hundred steps ago. It wandered. Explaining that
wandering is trivial: today's value is an excellent guess for tomorrow's, because
one step is small compared to how far the series has travelled.

So the model scored 0.99 for knowing that Tuesday resembles Monday.

Watch what happens when I plot exactly the same forecast a different way. Instead of
the level, I show the *change* from one step to the next — which is the only thing
that was ever genuinely unknown. The forecast becomes a flat line at roughly zero,
while reality jumps around it. R² on the change: **{rw['r2_change_model']:.4f}**.
Zero, or a hair below it.

Both charts show the same numbers from the same model. One says 0.99 and one says
0.00. The second one is the honest one, because it is scored on the quantity that
was actually at stake.

This is not a contrived example. Any slow-moving series behaves this way: prices,
temperatures, subscriber counts, queue lengths, sensor readings, portfolio values.
I ran the same model on a smoothly drifting series and it scored
{slow['r2_level_model']:.4f} on the level and {slow['r2_change_model']:.4f} on the
change. Same illusion, and this time the series does have some real structure — the
score just isn't measuring it.
""".strip(), figures=[figs["truth"]])

    post.add("The one number that is not fooled", f"""
There is a simple fix, and it is older than any of the models people worry about.

Before you report anything, make the laziest possible forecast: **assume nothing
changes.** Tomorrow equals today. Then check whether your model beat it.

That comparison has a standard name, MASE, and it is just your model's average error
divided by the lazy forecast's average error. Below 1, you added something. At 1, you
did not. Above 1, you made things worse.

Here is the same model on four different series:

| series | R² on the level | error vs "assume no change" |
|---|---|---|
| coin-flip random walk | {rw['r2_level_model']:.4f} | **{rw['vs_naive']:.3f}** |
| slow drifting series | {slow['r2_level_model']:.4f} | **{slow['vs_naive']:.3f}** |
| chaotic but deterministic | {chaos['r2_level_model']:.4f} | **{chaos['vs_naive']:.3f}** |
| pure noise | {noise['r2_level_model']:.4f} | {noise['vs_naive']:.3f} |

The first column is nearly identical for the first three rows. The second column is
not remotely identical, and it is telling you something true. On the random walk the
model lands at {rw['vs_naive']:.3f} — laziness, to three decimal places. On the
chaotic series it reaches {chaos['vs_naive']:.3f}, roughly
{1 / chaos['vs_naive']:.0f} times more accurate than the lazy forecast. That is a
model that genuinely learned the system it was shown.

Two scores, same data, opposite conclusions. Only one of them changes when the model
actually gets better.
""".strip(), figures=[figs["mase"]])

    post.add("Where this gets more interesting than a rule of thumb", f"""
The obvious lesson would be "always score on the change instead of the level". That
lesson is wrong, and it is worth seeing why, because the wrong version of this advice
is doing damage too.

Take the pure noise series — genuinely random, nothing to predict, and unlike the
random walk it does not wander anywhere. Subtract each value from the previous one
and something strange appears: the differences are now correlated, with a lag-1
correlation of **{trap['lag1_autocorr']:.2f}**. That is not a discovery about the
data. It is an artefact of subtracting, and it is exactly −0.5 in theory.

A model fitted to those differences scores R² =
**{trap['r2_on_differences']:.3f}**. Looks like a finding. Reconstruct the actual
level from it and the R² is {trap['r2_reconstructed_level']:.3f} — negative, meaning
worse than just guessing the long-run average.

So differencing is not a safety measure either. Differencing a series that did not
need it manufactures structure that looks predictable and is not.

The other direction is worth knowing too. Laziness is not always a strong opponent.
On that same noise series, "assume nothing changes" scores R² =
{noise['r2_level_naive']:.3f} — far worse than useless, because yesterday's noise
tells you nothing about today's. Persistence is a brutal baseline on smooth series
and a terrible one on jumpy series, which is precisely why comparing against it is
informative instead of ceremonial.

What survives all of this is narrower and duller than a rule about differencing:
**compare your model to the lazy forecast, on the quantity you actually care
about.** Not on a transformed version that happens to score better.
""".strip())

    post.add("Three questions", """
If someone shows you a forecast — a vendor, a colleague, a paper, your own notebook
from last month — these three questions do most of the work:
**1. What does the lazy forecast score?** If nobody computed it, the number you were
shown is uninterpretable. This is not a hostile question; it takes one line of code
and it is the first thing a careful analyst runs.

**2. Is the score measured on the thing you care about?** Predicting a level you
already almost know is not the same as predicting a change you do not.

**3. Was the comparison made on data the model had never seen?** Everything above
assumed a clean split. Without one, all of these numbers get better and none of them
get truer.

None of this is a criticism of machine learning, and none of it needs a complicated
model to go wrong. My example was a four-parameter linear fit. The failure was
entirely in the scoring, which is where a surprising share of forecasting failures
live — not in the model, but in the sentence describing how well it did.

Next in this series, the same discipline applied to a much stronger claim: a method
that comes with a mathematical *guarantee* about how often it will be right. I will
show it missing that guarantee badly, and explain exactly which assumption it needed
and did not have.
""".strip())

    return post


if __name__ == "__main__":
    p = build()
    print(p.title, "|", p.word_count(), "words |", len(p.figures), "figures")
    for issue in p.audit():
        print("  audit:", issue)
