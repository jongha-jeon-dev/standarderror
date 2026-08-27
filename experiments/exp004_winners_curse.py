"""exp004 — I trained thousands of models on a coin flip and the best looked good.

Backlog: new, Track B, pairs with E5. E5 said "your score is meaningless because
nobody computed the baseline". This one says "your score is meaningless because you
picked the maximum of many".

The design is chosen so the answer is knowable in advance. The target is a
**randomly permuted, exactly balanced** label sequence and the features are pure
noise, so no model can do better than chance, and each model's validation accuracy
is (very nearly) a draw from Binomial(n_val, 1/2). That makes the best-of-N score
predictable in closed form from order statistics — and the point of the post is that
the observed winner sits on that prediction.

Two design notes, both learned the hard way:

1. **Exact balance matters.** A first attempt used i.i.d. coin flips and the class
   balance drifted between splits, so the models inherited a majority-class bias and
   the mean validation accuracy came out at 48.2% instead of 50%. That is a nuisance,
   not a finding, and it obscured the effect being demonstrated.

2. **The order statistics are computed exactly, from the binomial CDF, not from the
   normal approximation.** The table asks for the number of trials needed to reach
   60% on 900 observations, which is a six-sigma event; the normal tail is wrong
   there by a factor of ~1.5 (1.01e9 trials against 6.56e8). Since the whole post
   turns on "the formula predicted it", the formula has to be the right one.

   E[max of N draws] uses the exact discrete identity
   `E[M] = sum_h h (F(h)^N - F(h-1)^N)`, and the table inverts the same function by
   bisection, so Fig 3's curve and Table 1 are literally the same function read in
   opposite directions.

Run: `standarderror run exp004_winners_curse --publish`
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

import standarderror as se
from standarderror.render import Post
from standarderror.uq import multiplicity
from standarderror.viz import charts, theme

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed

N_MODELS = 2000
N_FEATURES = 60
K_PER_MODEL = 4
N_TRAIN, N_VAL, N_TEST = 1200, 900, 900
N_TOTAL = N_TRAIN + N_VAL + N_TEST

ALPHA = 0.05


def make_world(seed: int = 11) -> dict:
    """A target nobody can predict, and features that know nothing about it.

    The label is an exactly balanced sequence, randomly permuted. "Exactly
    balanced" removes base-rate drift between splits; "randomly permuted" is what
    makes it unpredictable. Both matter, and neither is a trick — it is the
    cleanest possible stand-in for a fair coin.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(N_TOTAL, dtype=int)
    y[: N_TOTAL // 2] = 1
    rng.shuffle(y)
    x = rng.standard_normal((N_TOTAL, N_FEATURES))
    tr = slice(0, N_TRAIN)
    va = slice(N_TRAIN, N_TRAIN + N_VAL)
    te = slice(N_TRAIN + N_VAL, N_TOTAL)
    return {"x": x, "y": y, "tr": tr, "va": va, "te": te,
            "base_rate": {"train": float(y[tr].mean()),
                          "validation": float(y[va].mean()),
                          "test": float(y[te].mean())}}


def run_search(world: dict, n_models: int = N_MODELS, seed: int = 5) -> dict:
    """Fit `n_models` candidates, each on a random handful of the noise features."""
    rng = np.random.default_rng(seed)
    x, y = world["x"], world["y"]
    tr, va, te = world["tr"], world["va"], world["te"]

    val_acc = np.empty(n_models)
    test_acc = np.empty(n_models)
    val_hits = np.empty((n_models, N_VAL), dtype=bool)
    test_hits = np.empty((n_models, N_TEST), dtype=bool)
    for i in range(n_models):
        cols = rng.choice(N_FEATURES, size=K_PER_MODEL, replace=False)
        model = LogisticRegression(max_iter=400).fit(x[tr][:, cols], y[tr])
        vp = model.predict(x[va][:, cols])
        tp = model.predict(x[te][:, cols])
        val_hits[i] = vp == y[va]
        test_hits[i] = tp == y[te]
        val_acc[i] = val_hits[i].mean()
        test_acc[i] = test_hits[i].mean()
    return {"val_acc": val_acc, "test_acc": test_acc,
            "val_hits": val_hits, "test_hits": test_hits}


def expected_max(n_models: int) -> float:
    """Expected best accuracy over `n_models` chance-level models, exactly.

    Thin wrapper on `standarderror.uq.multiplicity` so the figures and the table read
    cleanly; the arithmetic, the reason it is exact rather than normal-approximate,
    and the Monte Carlo test all live in the package.
    """
    return multiplicity.expected_max_accuracy(n_models, N_VAL)


def analyse(search: dict) -> dict:
    """The winner, what chance predicted, and what the winner did next."""
    va, te = search["val_acc"], search["test_acc"]
    n = len(va)
    winner = int(np.argmax(va))
    sd = float(np.sqrt(0.25 / N_VAL))   # sd of one model's accuracy under chance

    running_max = np.maximum.accumulate(va)
    grid = np.unique(np.geomspace(1, n, 40).astype(int))
    curve = np.array([expected_max(int(k)) for k in grid])
    # How well the formula tracked the run. Quote this rather than "they lie on top
    # of each other", which is an assertion about a picture. N < 10 is excluded
    # because the expected maximum of a handful of draws is a weak statement about
    # any single realisation of it — the claim is about the trend, not the first step.
    dev = np.abs(running_max[grid - 1] - curve) * 100
    tail = grid >= 10

    # The winner's p-value if you forget you ran a search, and after correcting.
    hits = int(round(va[winner] * N_VAL))
    p_face = float(stats.binom.sf(hits - 1, N_VAL, 0.5))

    thresh, attained = multiplicity.significance_threshold(N_VAL, ALPHA)
    order = np.argsort(va)
    corr = float(np.corrcoef(va, te)[0, 1])
    top5 = order[-5:][::-1]
    # Net "correct minus wrong" on each side of the selection boundary, in calls.
    # This is what Fig 2 plots, so the prose has to quote these and not eyeball it.
    net_left = 2 * search["val_hits"][top5].sum(axis=1) - N_VAL
    net_right = 2 * search["test_hits"][top5].sum(axis=1) - N_TEST
    # Peak excursion after selection, measured from each curve's own level at the
    # boundary. The *net* change understates what the chart shows — the curves run
    # up 40 calls and back — and a caption quoting the net while the reader is
    # looking at the peak is the figure-text mismatch the checklist warns about.
    walk = np.cumsum(2 * search["test_hits"][top5].astype(int) - 1, axis=1)
    return {
        "winner": winner,
        "val_best": float(va[winner]),
        "test_of_best": float(te[winner]),
        "val_mean": float(va.mean()),
        "test_mean": float(te.mean()),
        "val_sd_observed": float(va.std()),
        "val_sd_theory": sd,
        "expected_max": expected_max(n),
        "expected_max_curve": (grid, curve),
        "running_max": running_max,
        "grid": grid,
        "curve_max_dev_pp": float(dev[tail].max()),
        "curve_end_gap_pp": float((running_max[-1] - curve[-1]) * 100),
        "p_face": p_face,
        "p_bonferroni": float(min(1.0, p_face * n)),
        "sig_threshold": thresh,
        "sig_attained": attained,
        "n_individually_significant": int((va >= thresh).sum()),
        "n_expected_significant": attained * n,
        "corr_val_test": corr,
        "corr_r2_pct": 100.0 * corr ** 2,
        "corr_se": float(1.0 / np.sqrt(n - 3)),
        "top5_test_mean": float(te[order[-5:]].mean()),
        "top5_val": va[top5],
        "top5_net_left": net_left,
        "top5_net_right": net_right,
        "top5_left_mean": float(net_left.mean()),
        "top5_right_mean": float(net_right.mean()),
        "top5_right_best": int(net_right.max()),
        "top5_right_worst": int(net_right.min()),
        "top5_peak_up": int(walk.max()),
        "top5_peak_down": int(walk.min()),
        "walk_sd": float(np.sqrt(N_TEST)),
        "n_models": n,
        "trials_for": {p: multiplicity.trials_to_reach(p, N_VAL)
                       for p in (0.53, 0.55, 0.57, 0.60)},
    }


def figures(search: dict, res: dict) -> dict:
    src = "Simulated: a randomly permuted balanced label, pure-noise features."
    figs = {}
    va = search["val_acc"]

    # F1 — the spread of "model quality" is chance, and here is the winner.
    lo, hi = va.min() - 0.005, va.max() + 0.008
    xs = np.linspace(lo, hi, 400)
    pdf = stats.norm.pdf(xs, 0.5, res["val_sd_theory"])
    fig_meta, _ = charts.histogram(
        va, bins=34, overlay={"what pure chance predicts": (xs, pdf)},
        mark={f"the winner: {res['val_best'] * 100:.1f}%": res["val_best"]},
        title=f"{res['n_models']:,} models, and every difference between them is luck",
        subtitle=("Validation accuracy of each candidate. The curve is the "
                  "distribution you get from coin flipping, drawn — not fitted."),
        xlabel="validation accuracy", source=src, mode="light",
        alt=("Histogram of validation accuracy for two thousand models, centred on "
             "50 percent, with a bell curve for pure chance lying on top of the "
             "bars almost exactly, and a marked line at the best model's score in "
             "the right tail."),
        caption=("Fig 1. The bell curve is not fitted to the bars — it is what "
                 "coin flipping predicts, drawn on top. The models differ from "
                 "each other by exactly as much as luck says they should. The "
                 "winner is in the right tail, which is what a right tail is for."),
        path=str(IMG / f"e6-f1-spread.{EXT}"))
    figs["spread"] = fig_meta

    # F2 — the backtest picture: top five, across the boundary.
    top5 = np.argsort(va)[-5:][::-1]
    curves = {}
    for rank, i in enumerate(top5, start=1):
        pnl = np.concatenate([2 * search["val_hits"][i] - 1,
                              2 * search["test_hits"][i] - 1])
        curves[f"#{rank} on validation"] = pd.Series(np.cumsum(pnl))
    frame = pd.DataFrame(curves)

    # A muted line at the average tally on the boundary. Without it the right half
    # reads as "still going up" — the curves do wander upward for a while — and the
    # eye cannot see that they end where they started. With it, the wandering is
    # visibly wandering *around a level*, which is the actual finding.
    level = res["top5_left_mean"]

    def mark_split(_fig, ax):
        m = theme.LIGHT
        # Unlabelled on purpose: every in-axes position for a label collides with
        # one of five wandering curves, so the subtitle carries the explanation.
        ax.axhline(level, color=m.muted, lw=1.2, ls=(0, (5, 3)))
        ax.axvline(N_VAL, color=m.series[7], lw=1.4)
        ax.annotate("selection ends here\nnothing after this\nwas used to choose",
                    (N_VAL, 0.04), xycoords=("data", "axes fraction"),
                    xytext=(8, 0), textcoords="offset points", ha="left",
                    va="bottom", fontsize=8.5, color=m.series[7])
        ax.axhline(0.0, color=m.axis, lw=1.0)

    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False, decorate=mark_split,
        title="The five best models, before and after they were chosen",
        subtitle=("Cumulative correct-minus-wrong calls. Left of the red line is "
                  "the data used to pick them, right of it data they had never "
                  "seen. The dashed line is where they stood when they were "
                  "picked."),
        ylabel="cumulative correct − wrong", xlabel="observation", source=src,
        alt=("Line chart of five cumulative score curves. All five climb steadily "
             "for the first nine hundred observations to around plus eighty-five, "
             "then after a marked vertical line they wander up and down around a "
             "dashed horizontal reference at that level and end near it."),
        caption=(f"Fig 2. Five confident climbs, then a random walk. After the "
                 f"line they run as far as {res['top5_peak_up']:+d} calls above "
                 f"the level they were picked at and {abs(res['top5_peak_down'])} "
                 f"below it, and end {res['top5_right_mean']:+.1f} on average — "
                 f"{res['top5_test_mean'] * 100:.1f}% out of sample. Every "
                 "backtest has a left half; ask what the right half looks like."),
        path=str(IMG / f"e6-f2-curves.{EXT}"))
    figs["curves"] = fig_meta

    # F3 — the winner's score as a function of how many you tried.
    grid, curve = res["expected_max_curve"]
    frame = pd.DataFrame(
        {"what I actually got": pd.Series(res["running_max"][grid - 1] * 100,
                                          index=grid),
         "what luck predicts": pd.Series(curve * 100, index=grid)})
    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False, logx=True,
        title="Your best score is a function of how many you tried",
        subtitle=("Best validation accuracy so far, against the number of models "
                  "tried. The prediction is a formula, with nothing fitted to the "
                  "data."),
        ylabel="best accuracy so far (%)", xlabel="models tried (log scale)",
        source=src,
        alt=("Two rising curves against the number of models tried on a log axis. "
             "The observed best-so-far line is a staircase starting at 52 percent; "
             "the smooth theoretical prediction from order statistics starts at 50. "
             "They cross each other several times and both end near 55.5 percent."),
        caption=("Fig 3. The line is order statistics — the exact expected maximum "
                 "of N coin-flip draws — with nothing fitted. If your search sits "
                 "on this line, your best model is indistinguishable from your "
                 "search budget."),
        path=str(IMG / f"e6-f3-bestofn.{EXT}"))
    figs["bestofn"] = fig_meta

    # T1 — the table, as an image, because Medium strips table markup.
    t = res["trials_for"]
    # round(), not int(): 0.57 * 100 is 56.99999999999999 in binary floating point,
    # and int() truncates it to 56. The table said "56%" for a whole draft.
    rows = [[f"{round(p * 100)}%", f"{t[p]:,}"] for p in (0.53, 0.55, 0.57, 0.60)]
    fig_meta, _ = charts.table_image(
        rows, header=["for your best score to reach", "models you need to try"],
        title="The price of any headline number",
        subtitle=f"On a coin flip, with {N_VAL} observations to select on.",
        source=src, mode="light", bold_cols=(1,),
        alt=("Table of four rows: to reach a best score of 53% you need "
             f"{t[0.53]:,} models, 55% needs {t[0.55]:,}, 57% needs "
             f"{t[0.57]:,}, and 60% needs {t[0.60]:,}."),
        caption=(f"Table 1. A 55% hit rate is {t[0.55]:,} attempts away from "
                 "nothing at all. This is why the number of things you tried is "
                 "part of the result, and why not reporting it is not a neutral "
                 "omission."),
        path=str(IMG / f"e6-t1-trials.{EXT}"))
    figs["table"] = fig_meta

    # HERO — not part of the post body. A *distribution* card rather than the
    # default stat card: this post's finding is where one value landed relative to
    # every other one, and that is a shape, not a number. Putting a big 55.3% on it
    # as well would make it a busier version of the same card every other post uses.
    fig_meta, _ = charts.distribution_card(
        va * 100,
        headline="I trained 2,000 models to predict a coin flip.",
        mark=res["val_best"] * 100,
        mark_label=f"the best one: {res['val_best'] * 100:.1f}%",
        note=("Every model's score, and where the winner landed. There was no "
              "signal in the data at all — order statistics predicted "
              f"{res['expected_max'] * 100:.1f}% before I ran anything."),
        footer="The Standard Error", mode="light",
        alt=(f"A histogram of 2,000 model scores centred on 50 percent, with a red "
             f"line marking the best score of "
             f"{res['val_best'] * 100:.1f} percent out in the right tail."),
        caption="",
        path=str(IMG / f"e6-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    world = make_world()
    search = run_search(world)
    res = analyse(search)
    figs = figures(search, res)

    t = res["trials_for"]
    post = Post(
        title="I Trained 2,000 Models on a Coin Flip and the Best One Looked Great",
        slug="i-trained-2000-models-on-a-coin-flip",
        subtitle=(f"A {res['val_best'] * 100:.1f}% hit rate, a p-value of "
                  f"{res['p_face']:.4f}, and not one shred of signal in the data"),
        summary=(f"I generated a target that cannot be predicted, gave "
                 f"{N_FEATURES} pure-noise variables to "
                 f"{res['n_models']:,} models, and kept the best one. It called "
                 f"{res['val_best'] * 100:.1f}% correctly, which taken at face "
                 f"value is significant at p = {res['p_face']:.4f}. Out of sample "
                 f"it got {res['test_of_best'] * 100:.1f}%. The interesting part "
                 "is that its winning score was predictable before I started — "
                 "from the number of models alone."),
        tags=["data-science", "statistics", "machine-learning", "quantitative-finance",
              "analytics"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=1000, max_words=1800,
        table_figures=[figs["table"]],
        data_sources=[
            "Fully simulated: a randomly permuted, exactly balanced binary label "
            "and 60 independent standard-normal features. No external data; every "
            "number is reproducible from the repo with a fixed seed.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "scipy", "scikit-learn", "standarderror")),
            "design": f"{res['n_models']:,} logistic regressions, each on "
                      f"{K_PER_MODEL} of {N_FEATURES} noise features",
            "splits": f"{N_TRAIN} train / {N_VAL} validation / {N_TEST} test, "
                      "contiguous and disjoint",
            "class balance (train/validation/test)": ", ".join(
                f"{k} {v:.3f}" for k, v in world["base_rate"].items()),
            "chance sd of one model's accuracy":
                f"{res['val_sd_theory'] * 100:.2f}pp (observed across models: "
                f"{res['val_sd_observed'] * 100:.2f}pp)",
            "order statistics": "exact, from the Binomial(900, 1/2) CDF — not the "
                                "normal approximation, which is off by ~1.5x in "
                                "the tail the trials table lives in",
        },
    )

    post.add("A result I can guarantee is fake", f"""
I built a dataset where I know the answer in advance, because I wrote it.

The thing to predict is a sequence of ones and zeros — exactly half of each,
shuffled at random. Think of it as a coin flip, except I have also removed the
nuisance of one side coming up slightly more often. The inputs are
{N_FEATURES} columns of random numbers, generated independently of the target. No
column knows anything about the answer. Nothing can be learned here, and I can say
that with certainty rather than as a hypothesis.

Then I did what everybody does. I picked {K_PER_MODEL} of those {N_FEATURES}
columns at random, fitted a small model, and measured how often it called the
answer correctly on data it had not been trained on. Then I did it again with a
different {K_PER_MODEL}. {res['n_models']:,} times.

The best of those {res['n_models']:,} models called
**{res['val_best'] * 100:.1f}%** of them correctly.

If I show you only that model — and why would I show you the other
{res['n_models'] - 1:,} — it looks like a finding. Take its
{res['val_best'] * 100:.1f}% at face value and ask whether a coin could do that
well by accident: the answer is **p = {res['p_face']:.4f}**. Less than one in a
thousand. In most contexts that ends the argument.

There is no signal. I am certain of it. So where did the
{res['val_best'] * 100:.1f}% come from?
""".strip())

    post.add("Nothing went wrong — that is the problem", f"""
Here is the distribution of all {res['n_models']:,} scores.

The bell curve drawn on top of it is **not fitted to the data.** It is the
distribution you get from flipping a coin {N_VAL} times, which is exactly what each
of these models is doing. Its width is fixed by arithmetic:
{res['val_sd_theory'] * 100:.2f} percentage points. The width I actually measured
across the {res['n_models']:,} models is {res['val_sd_observed'] * 100:.2f}
percentage points.

So all of the apparent variation in model quality — the reason one model looks
better than another, the reason there is anything to choose between them — is
sampling noise, and theory got its size right to better than one part in a hundred.

And once you accept that, the winner stops being mysterious. If you draw
{res['n_models']:,} numbers from a bell curve, one of them is going to be near the
right edge. That is not a discovery about the model. It is a fact about drawing
{res['n_models']:,} numbers.

The scale of it is easy to underestimate. On its own, a model needs
{res['sig_threshold'] * 100:.1f}% — that is
{(res['sig_threshold'] - 0.5) * 100:.1f} percentage points above chance — to clear
a one-sided 5% test on {N_VAL} observations. Out of my {res['n_models']:,} models,
**{res['n_individually_significant']}** cleared that bar. The number you should
expect, if nothing whatsoever is going on, is about
{res['n_expected_significant']:.0f}. I did not find
{res['n_individually_significant']} promising signals. I found the ordinary,
predictable output of asking the same question {res['n_models']:,} times.
""".strip(), figures=[figs["spread"]])

    post.add("The winner's score was knowable before I started", f"""
This is the part I find satisfying, and it is why I think the demonstration is
worth more than the warning.

"The largest of N draws from a known distribution" is a solved problem. It is
called an order statistic, and here the distribution is known exactly — the number
of correct calls in {N_VAL} coin flips — so no simulation and no approximation is
needed. For {res['n_models']:,} models the arithmetic predicts a best score of
**{res['expected_max'] * 100:.1f}%**.

I measured **{res['val_best'] * 100:.1f}%**.

Plot the best-so-far score against the number of models tried and the formula tracks
the experiment across three orders of magnitude — never more than
{res['curve_max_dev_pp']:.1f} percentage points apart once more than ten models are
in, and {abs(res['curve_end_gap_pp']):.1f} percentage points apart at the end.
Nothing is fitted. Your best model's score is a readout of your search budget.

Which means the arithmetic runs in the other direction too, and this is the table I
would pin above every desk where backtests happen. On a coin flip, with {N_VAL}
observations to select on:

| for your best score to reach | models you need to try |
|---|---|
| 53% | {t[0.53]:,} |
| 55% | {t[0.55]:,} |
| 57% | {t[0.57]:,} |
| 60% | {t[0.60]:,} |

A 55% hit rate — the kind of number that gets a strategy funded — is
{t[0.55]:,} attempts away from nothing whatsoever. And "attempts" is broader than
it sounds. Every feature you added and dropped, every window length you tried,
every threshold you nudged, every time you re-ran it after a bad result: those are
all draws. Nobody counts them, and the counting is the whole ballgame.
""".strip(), figures=[figs["bestofn"]])

    post.add("What it looks like when you only see the left half", f"""
Take the five models that scored best and plot their running tally of correct calls
minus wrong ones, straight through the boundary between the data I used to pick
them and the data I did not.

For the first {N_VAL} observations, five clean upward slopes, ending on average
{res['top5_left_mean']:+.0f} calls to the good. Then the line — and after it they
stop climbing and start wandering. Not flatly. At their best moment after selection
one of them was {res['top5_peak_up']} calls above where I picked it; at its worst
another was {abs(res['top5_peak_down'])} below. That is not a surprise either: a
coin-flip tally over {N_TEST} steps has a standard deviation of
{res['walk_sd']:.0f} calls, so swings of this size are the default, not an event.
What none of them does is keep making progress. Over the whole out-of-sample stretch
one ended {res['top5_right_best']} calls up and another
{abs(res['top5_right_worst'])} down, and the five together ended
{res['top5_right_mean']:+.1f} calls from where I picked them — an out-of-sample
**{res['top5_test_mean'] * 100:.1f}%**, with the single best one, my champion at
p = {res['p_face']:.4f}, managing **{res['test_of_best'] * 100:.1f}%**.

Which is the trap in the picture. Had I shown you only the stretch from the line to
that peak and called it a live track record, you would have seen a strategy up
{res['top5_peak_up']} calls with nothing behind it at all.

Across all {res['n_models']:,} models the correlation between validation accuracy
and test accuracy is **{res['corr_val_test']:+.3f}**, against a standard error of
{res['corr_se']:.3f}. It accounts for {res['corr_r2_pct']:.2f}% of the variation in
test accuracy, which is a polite way of saying that the score you selected on
carries no information about the score you care about. That is exactly right: there
was nothing for the validation score to be informative *about*.

That left-half shape is worth memorising, because it is what every overfitted
backtest looks like, and it is indistinguishable by eye from a real one. The
difference is not in the picture. It is in whether the right half exists, and
whether anybody looked at it before deciding.
""".strip(), figures=[figs["curves"]])

    post.add("Where this simulation is kinder than reality", f"""
Two places where my setup is not quite the clean textbook case, and one of them
matters in the direction you would not guess.

My {res['n_models']:,} models are not independent draws. They share a training set
and their feature subsets overlap, so the effective number of independent tries is
somewhat below {res['n_models']:,} — which is why the observed best,
{res['val_best'] * 100:.1f}%, sits a little *under* the
{res['expected_max'] * 100:.1f}% that independence predicts rather than scattered
either side of it. In real research the dependence is far stronger: variants of one
idea are highly correlated. That cuts both ways. It means the naive multiplicity
correction is too harsh, and it means the count of trials you should report is not
simply the number of notebook cells you ran.

The gentler simplification is that my target is exchangeable noise. Financial
series are not: they have autocorrelation, regime changes and drifting volatility,
all of which give a search *more* to latch onto, not less. The 900-observation
validation set here is also generous by the standards of a strategy tested on
monthly data, and a smaller selection set widens the chance distribution, which
pushes every number in Table 1 down. Fifty-five percent gets cheaper.
""".strip())

    post.add("The honest fixes, and their costs", f"""
None of this is new. The winner's curse, data snooping, multiple comparisons,
backtest overfitting — the statistics have been understood for decades, and the
finance literature in particular has been shouting about it since at least White's
Reality Check in 2000. What tends to be missing is not the warning but the
demonstration, so here is what to do about it, with the catch attached to each.

**Count your trials and say the number.** The single highest-value habit. Not just
model variants — every window, threshold, feature set and re-run. The count is part
of the result, and a paper or a pitch that omits it is not reporting a weaker
result, it is reporting an uninterpretable one.

**Correct for the count.** Multiplying your p-value by the number of trials
(Bonferroni) is the crude version: my winner's {res['p_face']:.4f} becomes
{res['p_bonferroni']:.2f}, which is the correct verdict. It is also far too
conservative when the trials are correlated, as model variants always are. The
finance-specific tools — White's Reality Check, the Superior Predictive Ability
test, deflated Sharpe ratios — exist precisely to handle correlated searches, and
they are worth the afternoon.

**Hold out a set you touch exactly once.** Powerful and fragile: it works right up
until the result disappoints you and you go back for another look. At that point it
has silently become a validation set and you no longer have a test set.

**Prefer fewer, motivated candidates.** Ten hypotheses you can each explain beat
ten thousand from a grid search, not because grid search is wrong but because the
correction you owe scales with the count and the explanation does not.

The uncomfortable version of all this: a model that survives a search of
{res['n_models']:,} needs to clear a much higher bar than the same model would if it
were your first idea, and that bar depends on something invisible in the final
notebook — how many things you tried. Any result presented without that number is
missing the denominator.

Next in this series, a method that comes with a mathematical *guarantee* about how
often it will be right, and what happens when the one assumption behind that
guarantee quietly fails. I will show a nominal 90% interval realising under 60%.
""".strip())

    return post


if __name__ == "__main__":
    p = build()
    print(p.title, "|", p.word_count(), "words |", len(p.figures), "figures")
    for issue in p.audit():
        print("  audit:", issue)
