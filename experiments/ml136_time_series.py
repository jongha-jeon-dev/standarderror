"""ML 36: Shuffling a Time Series Is Safe Until the Errors Remember.

The thirty-sixth episode of *Machine Learning, Taught Through What Breaks*,
opening Arc VIII (where the rows come from). "Never shuffle a time series
for cross-validation; always split it forward in time." Measured on seven
simulated settings, twenty series each, against the error on the 250 steps
after the training window:

* On a stationary autoregression with its lags as features, shuffled
  five-fold cross-validation was within 3% of the future error for linear
  regression, nearest neighbours and a forest, as Bergmeir, Hyndman and Koo
  showed it should be.
* When the residuals the features cannot explain are persistent, shuffled
  folds were 12-13% optimistic at persistence 0.9 and 20-22% at 0.99; with a
  drifting level, 19% for a forest and 25% with a time index.
* Forward splitting was not the safe default it is sold as: 18-20%
  pessimistic at persistence 0.99 and 11-14% for the drifting forests.
  Contiguous blocks were the closest scheme in five of the seven hard rows,
  including the most persistent.

Run: `standarderror run ml136_time_series --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import rows
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

SCHEMES = ("shuffled", "blocked", "forward")


def compute() -> dict:
    t = rows.time_series_table()
    key = {(r["setting"], r["model"]): r for r in t}
    return {"t": t, "k": key, "w": rows.window_correlation(),
            "far": rows.far_future()}


# ------------------------------------------------------------------ figures

def _dots(ax, m, table):
    labels = [f"{r['setting']} ({r['model']})" for r in table]
    y = np.arange(len(table))[::-1]
    for j, s in enumerate(SCHEMES):
        ax.scatter([100 * r["rel"][s] for r in table], y, s=34,
                   color=m.series[j + 1], label=f"{s} folds", zorder=3)
    ax.axvline(0, color=m.ink, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")


def figures(res: dict) -> dict:
    out: dict = {}
    t, k = res["t"], res["k"]

    out["f0"] = charts.diagram(
        lambda ax, m: _dots(ax, m, t),
        title="Which folds told the truth about the next 250 steps",
        subtitle=("Cross-validated MSE on 1,000 steps, relative to the MSE of "
                  "the refitted model on the 250 steps that follow; means "
                  "over 20 simulated series per row."),
        xlabel="estimate relative to the future error (%)",
        ylabel="",
        figsize=(8.2, 5.6),
        source="Simulated; standarderror/ml/rows.py.",
        alt=("Dot plot, one row per setting and model, three dots per row. "
             "In the first six rows all dots sit near zero. In the "
             "most persistent and drifting rows the shuffled dot is far to "
             "the left; the forward dot is to the right for the forests."),
        caption=(f"Shuffled folds were within 3% on the autoregressions and "
                 f"**{100 * abs(k[('residuals, persistence 0.99', 'forest')]['rel']['shuffled']):.0f}%** "
                 f"optimistic for a forest when the residuals had persistence "
                 f"0.99."),
        path=str(IMG / f"ml136-f0-folds.{EXT}"))[0]

    body = [[r["setting"], r["model"], f"{r['shuffled']:.3f}",
             f"{r['blocked']:.3f}", f"{r['forward']:.3f}",
             f"{r['future']:.3f}"] for r in t]
    out["f1"] = charts.table_image(
        body,
        header=["setting", "model", "shuffled", "blocked", "forward",
                "next 250 steps"],
        title="Mean squared errors, estimated and realised",
        subtitle=("Five-fold cross-validation on the first 1,000 steps three "
                  "ways, and the model refitted on all 1,000 scored on the "
                  "next 250. Means over 20 series."),
        source="Simulated; standarderror/ml/rows.py.",
        alt=("A thirteen-row table of mean squared errors. In the "
             "autoregressive rows the four numbers agree to a few percent; "
             "in the persistent and drifting rows the shuffled column is the "
             "lowest."),
        caption=("The right-hand column is the target. No scheme hit it in "
                 "every row; shuffling missed it only where something the "
                 "model could not see changed slowly."),
        align="llrrrr",
        path=str(IMG / f"ml136-f1-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    k = res["k"]
    ar = k[("AR(2), two lags", "forest")]
    pe = k[("residuals, persistence 0.99", "forest")]
    dr = k[("drifting level, time index", "forest")]

    def bars(row):
        def f(panel, m):
            panel.bar(range(4), [row[s] for s in (*SCHEMES, "future")],
                      color=[m.series[1], m.series[2], m.series[3], m.ink])
            lo = min(row[s] for s in (*SCHEMES, "future"))
            panel.set_ylim(0.8 * lo, None)
        return f

    return charts.lecture_hero(
        series=SERIES_TAG, episode=36,
        headline="Folds and the future",
        panels=[(bars(ar), f"{100 * ar['rel']['shuffled']:+.0f}%",
                 "shuffled, AR(2)"),
                (bars(pe), f"{100 * pe['rel']['shuffled']:+.0f}%",
                 "shuffled, 0.99 errors"),
                (bars(dr), f"{100 * dr['rel']['forward']:+.0f}%",
                 "forward, drifting")],
        note=("Shuffled folds are fine when the features carry the series' "
              "memory. They flatter the model when something it cannot see "
              "changes slowly, and forward folds overcorrect."),
        alt=("Three hand-drawn frames of four bars: equal bars; a short "
             "first bar; a tall third bar."),
        path=str(IMG / f"ml136-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["table"] = s.run("""
        from standarderror.ml import rows

        for r in rows.time_series_table():   # 20 series per setting
            e = r["rel"]
            print(f"{r['setting']:28s} {r['model']:7s} shuffled "
                  f"{e['shuffled']:+.1%}  blocked {e['blocked']:+.1%}  "
                  f"forward {e['forward']:+.1%}")
        """)
    out["window"] = s.run("""
        w = rows.window_correlation()   # x and u independent by construction
        for phi, c in w.items():
            print(f"persistence {phi}: typical correlation of x and u "
                  f"in a 1,000-step window {c:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    k, w = res["k"], res["w"]
    far = res["far"]["far"]
    ar = [k[("AR(2), two lags", m)] for m in ("linear", "5-NN", "forest")]
    one = k[("AR(2), one lag", "linear")]
    p0 = [k[("residuals, persistence 0", m)] for m in ("5-NN", "forest")]
    p9 = [k[("residuals, persistence 0.9", m)] for m in ("5-NN", "forest")]
    p99 = [k[("residuals, persistence 0.99", m)] for m in ("5-NN", "forest")]
    dl = k[("drifting level", "linear")]
    df = k[("drifting level", "forest")]
    dt = k[("drifting level, time index", "forest")]

    def pct(x):
        return f"{100 * abs(x):.0f}%"

    # The spine, asserted rather than trusted.
    assert all(abs(r["rel"]["shuffled"]) < 0.035 for r in ar + [one] + p0)
    assert all(-0.14 < r["rel"]["shuffled"] < -0.11 for r in p9)
    assert all(-0.23 < r["rel"]["shuffled"] < -0.19 for r in p99)
    assert all(r["rel"]["forward"] > 0.17 and r["rel"]["blocked"] > 0.09
               for r in p99)
    assert -0.21 < df["rel"]["shuffled"] < -0.18
    assert -0.26 < dt["rel"]["shuffled"] < -0.24
    assert dt["rel"]["shuffled"] < df["rel"]["shuffled"]
    assert abs(dt["rel"]["blocked"]) < 0.01 and abs(df["rel"]["blocked"]) < 0.05
    assert df["rel"]["forward"] > 0.10 and dt["rel"]["forward"] > 0.13
    assert all(r["rel"]["shuffled"] < 0 for r in (dl, df, dt))
    assert all(abs(r["rel"]["blocked"]) < abs(r["rel"]["shuffled"])
               for r in p9 + [df, dt, dl])
    hard = p9 + p99 + [dl, df, dt]
    closest = [min(("shuffled", "blocked", "forward"),
                   key=lambda q, r=r: abs(r["rel"][q])) for r in hard]
    assert closest.count("blocked") == 5
    assert abs(p9[0]["rel"]["blocked"]) < abs(p9[0]["rel"]["forward"])
    assert abs(p9[1]["rel"]["forward"]) < abs(p9[1]["rel"]["blocked"])
    assert abs(dl["rel"]["forward"]) < abs(dl["rel"]["blocked"])
    assert 0.17 < p99[1]["rel"]["forward"] < p99[0]["rel"]["forward"] < 0.21
    assert abs(res["far"]["near"] - p99[1]["future"]) < 1e-9
    assert far < p99[1]["future"] < p99[1]["blocked"]
    assert w[0.0] < 0.035 and 0.10 < w[0.9] < 0.12 and 0.17 < w[0.99] < 0.19

    post = Post(
        title=(f"{SERIES_TAG} 36: Shuffling a Time Series Is Safe Until the "
               f"Errors Remember"),
        slug="ml-36-time-series",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=36,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("The rule against shuffled folds is right for a reason "
                  "that is narrower than time: something the model cannot "
                  "see changing slowly. Forward splitting is not the cure "
                  "it is sold as."),
        summary=(
            f"Seven simulated time-series settings, twenty series each, "
            f"cross-validated three ways on 1,000 steps and compared with "
            f"the error on the next 250. On a stationary autoregression with "
            f"its lags as features, shuffled five-fold cross-validation was "
            f"within 3% of the future error for linear regression, nearest "
            f"neighbours and a forest. When the residuals had persistence "
            f"0.9 it was {pct(p9[1]['rel']['shuffled'])} optimistic for a "
            f"forest, at 0.99 {pct(p99[1]['rel']['shuffled'])}, and with a "
            f"drifting level and a time index "
            f"{pct(dt['rel']['shuffled'])}. Forward splitting was "
            f"{pct(dt['rel']['forward'])} pessimistic in that last setting; "
            f"contiguous blocks were within 1%."),
        tags=["machine-learning", "data-science", "statistics",
              "cross-validation", "time-series", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: an AR(2) process y_t = 0.5 y_{t-1} + 0.3 y_{t-2} + "
            "e_t, modelled from its last two lags or its last lag; the same "
            "around a random-walk level (step sd 0.15), with and without a "
            "time-index feature; and y_t = sin(2 x_t) + x_t + 0.7 u_t with x "
            "and u independent unit-variance AR(1) processes, x of "
            "persistence 0.95 and u of persistence 0, 0.9 or 0.99. 1,250 "
            "steps per series after a burn-in, seeds 0 to 19.",
            "Machinery: `standarderror/ml/rows.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's LinearRegression, "
            "KNeighborsRegressor (k = 5), RandomForestRegressor (50 trees, "
            "leaves of at least 5), KFold with and without shuffling, and "
            "TimeSeriesSplit, five folds each.",
            "Where this stops: Bergmeir and Benítez, \"On the use of "
            "cross-validation for time series predictor evaluation\", "
            "*Information Sciences* (2012); Bergmeir, Hyndman and Koo, \"A "
            "note on the validity of cross-validation for evaluating "
            "autoregressive time series prediction\", *Computational "
            "Statistics & Data Analysis* (2018); Roberts et al., "
            "\"Cross-validation strategies for data with temporal, spatial, "
            "hierarchical, or phylogenetic structure\", *Ecography* (2017).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("series seeds 0 to 19, window-correlation seeds "
                            "10,000 to 10,199; shuffled folds from the series "
                            "seed; about five minutes of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The rule everyone learns",
        f"""Arc VII measured the routine steps around a model. Arc VIII goes one step further back, to the rows themselves: where they came from, and what the process that produced them lets a model learn and a score report. Episode 3 found that duplicated rows straddling folds can leak; this arc measures the less obvious cases, starting with the most familiar rule about rows that come in order: *never shuffle a time series for cross-validation; always split it forward in time*.

The reasoning behind the rule is that neighbouring observations in time resemble each other, so a shuffled fold puts a test row's neighbours in the training set, and the model is graded on rows it has nearly seen. Forward splitting, training on the past and testing on what follows, is meant to reproduce how the model will be used.

Both halves are measurable. Seven simulated settings, twenty series each: the model is cross-validated on the first 1,000 steps with shuffled folds, with contiguous blocks, and with forward splits (cut into six stretches: train on the first, test on the second; train on two, test on the third; and so on), then refitted on all 1,000 steps and scored on the 250 that follow. That last number is what every estimate is trying to predict.

{snip['table'].markdown()}

The first rows are the result the rule says should not happen. On a stationary autoregression, with its last two values as the features, shuffled folds were within **{pct(ar[0]['rel']['shuffled'])}** of the future error for linear regression, {pct(ar[1]['rel']['shuffled'])} for five-nearest-neighbours and {pct(ar[2]['rel']['shuffled'])} for a forest. With only one lag, so that the linear model is missing part of the dependence, shuffled folds were still within {pct(one['rel']['shuffled'])}.""",
        figures=[figs["f0"]])

    post.add(
        "Why shuffling was fine",
        """Bergmeir, Hyndman and Koo proved the reason in 2018. When the features are the series' own lags, each row carries the history it needs: the row for step t contains steps t − 1 and t − 2, and once those are given, the error at t is new noise, unrelated to the errors of the neighbouring rows. A shuffled fold does place the rows for t − 1 and t + 1 in training, but their errors tell the model nothing about the error at t. Formally, cross-validation needs the errors of different rows to be uncorrelated given the features, not the rows themselves to be independent, and an autoregression with enough lags provides that.

The memory of the series is in the features, so it is not in the errors. The rule is a heuristic for a narrower condition, and when the condition holds, shuffling costs nothing and uses the data more evenly than any forward scheme can.""")

    post.add(
        "Errors with memory",
        f"""The condition fails when something that affects the target changes slowly and the features do not record it. The third group of settings makes that explicit. The target depends on an input x through a smooth curve, plus a residual u that the model never sees, and u is an AR(1) process whose persistence is set to 0, 0.9 or 0.99. With persistence 0, shuffled folds were within {pct(p0[1]['rel']['shuffled'])} for the forest. At 0.9 they were **{pct(p9[1]['rel']['shuffled'])}** optimistic, and at 0.99, **{pct(p99[1]['rel']['shuffled'])}**.

A likely mechanism, which the measurement supports without isolating, is that over any one window of 1,000 steps two independent slow processes look correlated:

{snip['window'].markdown()}

Over all time x and u are unrelated. Within a window, the typical correlation is {w[0.0]:.3f} when u has no memory and **{w[0.99]:.3f}** when it has a lot, and a flexible model learns that window's accidental relationship between x and the target as if it were part of the curve. Shuffled folds test it on rows from the same window, where the accident still holds, and reward it. The next 250 steps bring a different accident.

A drifting level does the same thing in a stronger form, since a random walk never forgets. A forest given the last two values was {pct(df['rel']['shuffled'])} optimistic under shuffled folds; given a time index as well, which lets it learn the level at each moment and interpolate between neighbouring rows, {pct(dt['rel']['shuffled'])}. Even linear regression, which cannot interpolate, was {pct(dl['rel']['shuffled'])} optimistic: the future level is not where it was in the training window.""",
        figures=[figs["f1"]])

    post.add(
        "Forward is not automatically right",
        f"""The usual cure is forward splitting, and it has a cost the rule does not mention. Its first fold trains on a sixth of the window, its second on two sixths, and so on up to five: on average the models it scores are trained on half the data the final model will have. Where more data helps, forward splitting is pessimistic, and the measurement shows it. With the drifting level and a time index it overstated the forest's future error by **{pct(dt['rel']['forward'])}**, against {pct(dt['rel']['shuffled'])} understated by shuffling. With persistence 0.99 in the residuals it overstated by {pct(p99[1]['rel']['forward'])}.

Contiguous blocks without shuffling, which train on four fifths of the window and test on the remaining block, past or future, were the closest scheme in five of the seven hard rows: within {pct(dt['rel']['blocked'])} for the drifting forest with a time index and within {pct(p9[0]['rel']['blocked'])} for nearest neighbours at persistence 0.9, where forward splits were off by {pct(p9[0]['rel']['forward'])}. Forward splits were closer for the forest at 0.9 and for linear regression on the drifting level. They were not perfect either. At persistence 0.99 every scheme missed, blocks by {pct(p99[1]['rel']['blocked'])} high and forward splits by {pct(p99[1]['rel']['forward'])} high, and the realised error was not unusually low: scored 1,000 steps further on instead of immediately after the window, the same forests averaged {far:.3f} against {p99[1]['future']:.3f}. A likely reason, which this measurement does not isolate, is that a residual this persistent leaves only a few independent stretches in 1,000 steps, so every fold that trains on fewer rows than the final model fits a noticeably worse curve. With that much memory, no split of one window estimates the future well.

Bergmeir and Benítez reached a similar conclusion on real series in 2012: blocked cross-validation was the most reliable of the schemes they compared, and the purely forward ones wasted data.""")

    post.add(
        "Where this breaks",
        """**Simulated series with known structure.** Real series have seasonality, structural breaks and regime changes, each a slow component of the kind that makes shuffling optimistic. The settings here isolate the mechanism; they do not say how much of it a given real series has.

**One horizon.** The future error was measured over the 250 steps after training, refitting once. A model refitted every step, or scored at a longer horizon, has a different target, and the comparison would shift.

**No gaps between folds.** Blocked schemes are often run with a gap of a few steps on each side of the test block, to stop lagged features from reaching across the boundary. With two lags the boundary effect is small; with long lag windows it is not.

**Means over twenty series.** Every estimate here is an average. On a single series the spread of each scheme is wide, and a single forward split, the most common choice in practice, is the noisiest of all.""")

    post.add(
        "What to keep",
        f"""1. Shuffled folds are valid when the features carry the series' memory. On a stationary autoregression with its lags as features they were within 3% of the future error for all three models.

2. They are optimistic when something unseen changes slowly: {pct(p9[1]['rel']['shuffled'])} at residual persistence 0.9, {pct(p99[1]['rel']['shuffled'])} at 0.99, {pct(dt['rel']['shuffled'])} with a drifting level and a time index.

3. Forward splitting trains on less data and was {pct(df['rel']['forward'])} to {pct(p99[0]['rel']['forward'])} pessimistic for the drifting forests and at persistence 0.99. Contiguous blocks were the closest scheme in five of the seven hard rows.

4. Check the residuals of a shuffled-fold model for autocorrelation. If they have none, shuffling is fine; if they do, use blocks, and expect any estimate to be rough when the residuals barely move.""")

    post.add(
        "Exercise",
        """Take a time-series model you cross-validated and compute the autocorrelation of its out-of-fold residuals at lags 1, 10 and 100, in time order.

If they are near zero, rerun with shuffled five-fold cross-validation and compare it with your forward estimate: the shuffled number is likely the more precise of the two. If they are not, compare shuffled folds with contiguous blocks; the gap between them is a direct measure of how much the slow component is flattering the shuffled score.""")

    post.add(
        "Next",
        """Time is one way rows come in groups that resemble each other. People are another: patients measured repeatedly, customers with many transactions, writers with many samples. Episode 37 asks what two thousand rows from twenty people are worth, and what row-level cross-validation says they are worth.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
