"""exp011 — the memory cycle is forecastable; the shortage is a different question.

Backlog: Track A (forecasting and uncertainty) crossed with Track B. Requested as
"a post on semiconductor supply and demand", with the brief that it should carry a
mathematical concept and a predictive-ML component rather than being commentary.

The industry is in a memory shortage as this is written, and the question being
asked out loud is how long it lasts — years, in most of the statements. That is a
forecast at a multi-year horizon, and this post asks what kind of question it is.

The structure
-------------
Supply responds to price with a lag, inventory integrates the imbalance, and price
responds to inventory. In log deviations that reduces to one linear delay
difference equation whose characteristic polynomial is exact:

    z^L - (2-d) z^{L-1} + (1-d) z^{L-2} + kappa*theta * sum_k g_k z^{L-L_k} = 0

Three results come out of it, and only the first is the one I expected.

1. **With no cost anchor (d = 0) the system is unstable at every gain and every
   delay.** The left-hand side becomes `(z-1)^2`, a double integrator, and delayed
   proportional feedback on a double integrator has no stable configuration. So the
   thing that keeps a memory market from diverging is not moderation in capacity
   planning; it is that cost per bit falls on a learning curve and price is pulled
   toward it.

2. **The cycle's period is set by the fast loop and is completely insensitive to
   the fab construction delay.** Sweeping the utilisation-and-inventory delay from
   1 to 12 months moves the realised period from 27 to 53 months. Sweeping the
   capacity construction delay from 36 to 96 months moves it by one sampling bin —
   nothing. The 4-to-6-year lead time that dominates every discussion of chip
   supply leaves no signature in the price series at all. And the period is not a
   fixed multiple of the delay either: it depends on loop gain, so cycle length
   cannot be read back as a lead time.

3. **Saturation turns the unstable mode into a limit cycle, and limit cycles are
   forecastable.** This is where the ML half starts and where my expectation was
   wrong. I set out to measure a forecast-horizon wall. There is no wall: a ridge
   regression on twelve monthly lags of price alone predicts the cycle to a
   normalised RMSE of 0.26 at a twelve-month horizon against 2.01 for persistence,
   and holds under 0.5 out to the 144-month cap. A 400-unit echo state network does
   no better. Shock size from 1.2% to 8% a month barely changes the answer.

What breaks it
--------------
One thing, and it is not dynamical. Step the *trend* rate of demand growth from
15% to 35% a year — the shape of an AI-capex regime shift, and roughly the gap
between published 2026 bit-supply growth of about 16% and demand growth in the
mid-thirties — and the identical models fall from a 144-month horizon to three
months, and from four times better than persistence to four times worse. Handing
them the true demand series does not rescue it: nothing in a training window
contains a trend that has not happened yet.

So a claim about 2030 is a demand-trend claim wearing a cycle's clothes, and the
two have completely different error properties. The post says that and stops; which
forecasts are right is not something this arithmetic can settle, and no company's
outlook is evaluated here.

Everything is closed form or fixed-seed simulation calibrated to published scalars.
No price series is used or redistributed, and there are no investment implications
anywhere in it.

Run: `quantpost run exp011_chip_cycle_horizon --publish`
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import quantpost as qp
from quantpost.dynamics import delay
from quantpost.models import ESN, ESNConfig
from quantpost.render import Post
from quantpost.viz import charts, theme

IMG = qp.SETTINGS.build_dir / "img"
EXT = os.environ.get("QUANTPOST_FIG_EXT", "png")
SEED = qp.SETTINGS.seed

# --- published facts, every one sourced in `data_sources` ----------------------
FAB_MONTHS = (48, 72)            # groundbreaking to volume production
FAB_PHASES = {"design and permitting": (6, 12), "shell construction": (12, 18),
              "cleanroom commissioning": (9, 12), "tool install and qual": (12, 15),
              "yield ramp": (6, 12)}
TOOL_LEAD_MONTHS = 18            # specialised etch and deposition, 2026
CYCLE_HIST_MONTHS = (18, 30)     # historical memory cycle duration
CYCLE_NOW_MONTHS = 36            # current cycle, from the mid-2023 trough
BIT_SUPPLY_GROWTH_2026 = 16.0    # %/yr, capped
BIT_DEMAND_GROWTH_2026 = 35.0    # %/yr, "mid-thirties"
TRADE_RATIO = 3.0                # wafers per bit of stacked product vs planar
HBM_WAFER_SHARE = 23.0           # % of DRAM wafers, July 2026
HBM_REVENUE_SHARE = 16.0         # % of memory revenue
INV_SUPPLIER_WEEKS = (3, 5)
INV_CHANNEL_WEEKS = (7, 9)
INV_WARNING_WEEKS = (8, 10)
MODULE_LEAD_WEEKS = (30, 40)
UTIL_ASSUMED = 95.0              # % fab utilisation in industry models
CYCLE_TIME_CAGR = 14.8           # %/yr growth in fab cycle time since 2020
PRICE_Q1_26 = (90, 95)           # % QoQ contract price change
PRICE_Q2_26 = (58, 63)
PRICE_Q3_26 = (13, 18)

# --- model configuration ------------------------------------------------------
# The fast delay is the one the post measures against; three months is the round
# number for "utilisation and inventory respond within a quarter", and the sweep
# covers one to twelve so nothing rests on the choice.
FAST_DELAY = 3
SLOW_DELAY = 60                  # midpoint of the published 48-72 month range
N_MONTHS = 720
BREAK_AT = 480
BREAK_GROWTH = BIT_DEMAND_GROWTH_2026 / 100.0
BASE_GROWTH = 0.15
FAST_SWEEP = (1, 2, 3, 4, 6, 9, 12)
SLOW_SWEEP = (36, 48, 60, 72, 96)
SHOCK_SWEEP = (0.012, 0.02, 0.03, 0.05, 0.08)
MAX_H = 144
MIX_YEARS_MODEL = 12.0           # adoption period used in the simulation
MIX_YEARS_ACTUAL = 4.0           # roughly what the stacked-memory shift took
N_LAGS = 12
THRESHOLD = 0.5                  # normalised RMSE counting as "still forecasting"
# Columns of the stacked state, in order.
STATE_NAMES = ("log price", "inventory weeks", "utilisation", "mix",
               "log capacity", "log demand")
VIEWS = {
    "price only": (0,),
    "price and inventory": (0, 1),
    "the producer's own state": (0, 1, 2, 3, 4),
    "and the true demand series": (0, 1, 2, 3, 4, 5),
}
# The multi-channel reservoir needs a smaller input scaling to work at all; see
# `forecast_curve`. Reported rather than hidden.
ESN_KW = dict(n_reservoir=400, spectral_radius=0.9, sparsity=0.05,
              input_scaling=0.15, leak_rate=0.5, ridge=1e-6, seed=2)


def state_matrix(run: delay.CycleRun) -> np.ndarray:
    """The six channels, in `STATE_NAMES` order."""
    return np.column_stack([np.log(run.price), run.inventory, run.utilisation,
                            run.mix, np.log(run.capacity), np.log(run.demand)])


def _lagged(z: np.ndarray, n_lags: int) -> np.ndarray:
    t = len(z)
    return np.concatenate([z[n_lags - 1 - j:t - j] for j in range(n_lags)], axis=1)


def forecast_curve(run: delay.CycleRun, cols, *, train_end: int, test: slice,
                   kind: str = "lags", max_h: int = MAX_H) -> dict:
    """Normalised RMSE against forecast horizon, by direct multi-horizon fit.

    One ridge regression per horizon `h`, mapping features at `t` to log price at
    `t + h`. Direct rather than closed-loop on purpose: an iterated rollout of a
    multi-channel model is unstable in ways that have nothing to do with the
    question, and a horizon measured off a diverging rollout measures the rollout.

    Everything is standardised on the *training* window only, so the reported error
    is in units of the pre-break series' standard deviation and the post-break
    numbers are directly comparable to the pre-break ones.
    """
    u = state_matrix(run)[:, list(cols)]
    mu, sg = u[:train_end].mean(0), u[:train_end].std(0) + 1e-12
    z = (u - mu) / sg
    y = np.log(run.price)
    y = (y - y[:train_end].mean()) / (y[:train_end].std() + 1e-12)

    if kind == "esn":
        model = ESN(config=ESNConfig(**ESN_KW))
        model._build(z.shape[1])
        feats, offset = model.harvest(z), 0
    elif kind == "lags":
        feats, offset = _lagged(z, N_LAGS), N_LAGS - 1
    else:
        raise ValueError(f"unknown feature kind {kind!r}")

    hs, model_err, naive_err = [], [], []
    for h in range(1, max_h + 1):
        t = np.arange(offset, len(y) - h)
        x_all, y_all = feats[t - offset], y[t + h]
        is_train = t < train_end - h
        is_test = (t >= test.start) & (t < test.stop)
        if is_train.sum() < 100 or is_test.sum() < 6:
            break
        a, b = x_all[is_train], y_all[is_train]
        w = np.linalg.lstsq(a.T @ a + 1e-4 * np.eye(a.shape[1]), a.T @ b,
                            rcond=None)[0]
        hs.append(h)
        model_err.append(float(np.sqrt(np.mean((x_all[is_test] @ w
                                               - y_all[is_test]) ** 2))))
        # Persistence: the last observed log price, which is what "no forecast"
        # gets you and the only baseline that matters on a slow-moving series.
        naive_err.append(float(np.sqrt(np.mean((z[t[is_test], 0]
                                               - y_all[is_test]) ** 2))))
    hs = np.asarray(hs)
    model_err = np.asarray(model_err)
    naive_err = np.asarray(naive_err)
    ok = hs[model_err < THRESHOLD]
    return {"h": hs, "nrmse": model_err, "persistence": naive_err,
            "horizon": int(ok.max()) if len(ok) else 0,
            "capped": bool(len(ok) and ok.max() == hs.max())}


def analyse() -> dict:
    base = delay.simulate(n_months=N_MONTHS, fast_delay=FAST_DELAY,
                          slow_delay=SLOW_DELAY, seed=11)
    broken = delay.simulate(n_months=N_MONTHS, fast_delay=FAST_DELAY,
                            slow_delay=SLOW_DELAY,
                            regime=(BREAK_AT, BREAK_GROWTH), seed=11)
    assert np.allclose(base.price[:BREAK_AT], broken.price[:BREAK_AT]), (
        "the two runs must be identical before the break, or the comparison is "
        "between two different histories rather than one history and its sequel")

    linear = delay.model_from_simulation(fast_delay=FAST_DELAY,
                                        slow_delay=SLOW_DELAY,
                                        trade_ratio=TRADE_RATIO,
                                        mix_target=HBM_WAFER_SHARE / 100.0)
    dom = delay.dominant_mode(linear)
    undamped = delay.CycleModel(linear.channels, kappa=linear.kappa,
                                theta=linear.theta, decay=0.0)
    undamped_max = float(max(abs(delay.characteristic_roots(undamped))))

    # Which delay sets the period?
    fast_sweep, slow_sweep = {}, {}
    for L in FAST_SWEEP:
        run = delay.simulate(n_months=900, fast_delay=L, slow_delay=SLOW_DELAY,
                             seed=3)
        lin = delay.dominant_mode(
            delay.model_from_simulation(fast_delay=L, slow_delay=SLOW_DELAY))
        fast_sweep[L] = {"realised": delay.realised_period(run.price),
                         "linear": lin["period_months"],
                         "growth": lin["growth_per_step"]}
    for S in SLOW_SWEEP:
        run = delay.simulate(n_months=900, fast_delay=FAST_DELAY, slow_delay=S,
                             seed=3)
        slow_sweep[S] = {"realised": delay.realised_period(run.price)}

    shock_sweep = {}
    for sd in SHOCK_SWEEP:
        run = delay.simulate(n_months=900, fast_delay=FAST_DELAY,
                             slow_delay=SLOW_DELAY, shock_sd=sd, seed=3)
        window = [run.price[i:i + 36].max() / run.price[i:i + 36].min()
                  for i in range(0, len(run.price) - 36, 12)]
        shock_sweep[sd] = {"period": delay.realised_period(run.price),
                           "peak_to_trough": float(np.median(window))}

    # The forecasting experiment. Trained on the pre-break window in both cases,
    # tested on the same calendar months, so the only difference between the two
    # columns of the table is whether the regime changed.
    train_end = BREAK_AT - 24
    test = slice(BREAK_AT + 6, BREAK_AT + 120)
    curves = {}
    for label, cols in VIEWS.items():
        for kind in ("lags", "esn"):
            for tag, run in (("steady", base), ("break", broken)):
                curves[(label, kind, tag)] = forecast_curve(
                    run, cols, train_end=train_end, test=test, kind=kind)

    price_only_steady = curves[("price only", "lags", "steady")]
    price_only_break = curves[("price only", "lags", "break")]
    drag_model = delay.mix_drag(trade_ratio=TRADE_RATIO, mix_start=0.02,
                               mix_target=HBM_WAFER_SHARE / 100.0,
                               mix_years=MIX_YEARS_MODEL)
    drag_actual = delay.mix_drag(trade_ratio=TRADE_RATIO, mix_start=0.02,
                                mix_target=HBM_WAFER_SHARE / 100.0,
                                mix_years=MIX_YEARS_ACTUAL)
    return {
        "drag_model": drag_model, "drag_actual": drag_actual,
        "base": base, "broken": broken, "linear": linear, "dominant": dom,
        "undamped_max_root": undamped_max,
        "fast_sweep": fast_sweep, "slow_sweep": slow_sweep,
        "shock_sweep": shock_sweep, "curves": curves,
        "train_end": train_end, "test": test,
        "realised_base": delay.realised_period(base.price),
        "slow_spread": (max(v["realised"] for v in slow_sweep.values())
                        - min(v["realised"] for v in slow_sweep.values())),
        "fast_spread": (max(v["realised"] for v in fast_sweep.values())
                        - min(v["realised"] for v in fast_sweep.values())),
        "steady_horizon": price_only_steady["horizon"],
        "break_horizon": price_only_break["horizon"],
        "steady_nrmse12": float(price_only_steady["nrmse"][11]),
        "break_nrmse12": float(price_only_break["nrmse"][11]),
        "steady_persistence12": float(price_only_steady["persistence"][11]),
        "break_persistence12": float(price_only_break["persistence"][11]),
        "critical_gain_fast": delay.critical_gain(FAST_DELAY),
        "critical_gain_slow": delay.critical_gain(SLOW_DELAY),
    }


def figures(res: dict) -> dict:
    src_model = ("Closed-form roots of the linearised delay system plus fixed-seed "
                 "nonlinear simulation, calibrated to the published figures cited "
                 "below. No price series is used.")
    figs = {}
    fs, ss = res["fast_sweep"], res["slow_sweep"]

    # F1 — which delay sets the period. The whole point of the first half.
    grid = sorted(set(FAST_SWEEP) | set(SLOW_SWEEP))
    frame = pd.DataFrame(
        {"inventory and utilisation delay":
            [fs[L]["realised"] if L in fs else np.nan for L in grid],
         "capacity construction delay":
            [ss[L]["realised"] if L in ss else np.nan for L in grid]},
        index=grid)

    def band(_fig, ax):
        m = theme.LIGHT
        ax.axhspan(CYCLE_HIST_MONTHS[0], CYCLE_NOW_MONTHS, color=m.series[0],
                   alpha=0.10, lw=0)
        ax.annotate(f"observed memory cycles: "
                    f"{CYCLE_HIST_MONTHS[0]}-{CYCLE_HIST_MONTHS[1]} months "
                    f"historically,\n{CYCLE_NOW_MONTHS} months in the current one",
                    (0.98, CYCLE_HIST_MONTHS[0]),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 8), textcoords="offset points", ha="right",
                    va="bottom", fontsize=8.5, color=m.muted, linespacing=1.4)
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, ScalarFormatter
        ax.xaxis.set_major_locator(FixedLocator(grid))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(FixedLocator([]))

    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False, decorate=band,
        title="Only one of the two delays sets the cycle",
        subtitle=("Realised period of the simulated price series, against the "
                  "length of each feedback delay in turn, with the other held "
                  "fixed. Each point is a 900-month run."),
        ylabel="realised cycle period (months)",
        xlabel="length of the delay being varied (months, log scale)",
        source=src_model,
        alt=("Two lines. The inventory-and-utilisation delay line rises steadily "
             "from about 27 to 53 months as that delay goes from 1 to 12 months. "
             "The capacity-construction line is flat at about 34 months across "
             "delays from 36 to 96 months."),
        caption=(f"Fig 1. Stretching the fast loop from one month to twelve moves "
                 f"the cycle by {res['fast_spread']:.0f} months. Stretching fab "
                 f"construction from three years to eight moves it by "
                 f"{res['slow_spread']:.1f} — one sampling bin, which is to say "
                 f"nothing. The lead time that dominates every discussion of chip "
                 f"supply is the one with no signature in the price series."),
        path=str(IMG / f"a3-f1-which-delay.{EXT}"))
    figs["delay"] = fig_meta

    # F2 — the forecasting result.
    steady = res["curves"][("price only", "lags", "steady")]
    broke = res["curves"][("price only", "lags", "break")]
    n = min(len(steady["h"]), len(broke["h"]))
    curve_frame = pd.DataFrame(
        {"steady demand trend": steady["nrmse"][:n],
         "after a demand-trend break": broke["nrmse"][:n],
         "persistence, steady trend": steady["persistence"][:n]},
        index=steady["h"][:n])

    def mark_threshold(_fig, ax):
        m = theme.LIGHT
        ax.axhline(THRESHOLD, color=m.muted, lw=1.2, ls=(0, (5, 3)))
        # No in-plot label: every region next to this line is crossed by one of the
        # three curves at some horizon, and the subtitle carries the meaning.

    fig_meta, _ = charts.lines(
        curve_frame, mode="light", direct_labels=False, decorate=mark_threshold,
        logy=True,
        title="The same model, the same features, two regimes",
        subtitle=(f"Out-of-sample error of a ridge regression on twelve monthly "
                  f"lags of price alone, against forecast horizon. Both are trained "
                  f"on the identical pre-break window and tested on the same "
                  f"months; the dashed line is the {THRESHOLD:g}-sd threshold this "
                  f"post calls 'still forecasting'."),
        ylabel="RMSE / standard deviation of the training series (log scale)",
        xlabel="forecast horizon (months)", source=src_model,
        alt=("Three curves on a log scale. The steady-trend error stays near 0.1 "
             "across the whole 144-month horizon; the post-break error sits above "
             "1 everywhere, above the persistence baseline; persistence rises from "
             "near zero and flattens around 2."),
        caption=(f"Fig 2. On a steady demand trend the cycle is not merely "
                 f"forecastable, it is easy: normalised error "
                 f"{res['steady_nrmse12']:.2f} at a twelve-month horizon against "
                 f"{res['steady_persistence12']:.2f} for persistence, and under "
                 f"the threshold out to the {MAX_H}-month limit of the experiment. "
                 f"Step the demand trend once and the identical model reaches "
                 f"{res['break_horizon']} months and is *worse* than persistence "
                 f"at twelve ({res['break_nrmse12']:.2f} against "
                 f"{res['break_persistence12']:.2f}). Nothing about the model "
                 f"changed."),
        path=str(IMG / f"a3-f2-horizon.{EXT}"))
    figs["horizon"] = fig_meta

    # F3 — the observability control, which is the answer to "get better data".
    labels, values = [], []
    for label in VIEWS:
        labels.append(label)
        values.append(res["curves"][(label, "lags", "break")]["horizon"])
    fig_meta, _ = charts.ranked_bars(
        labels, values, mode="light", value_fmt=".0f",
        title="More of the state does not buy back the horizon",
        subtitle=("Forecast horizon across a demand-trend break, by how much of "
                  "the system the model is allowed to see. Ridge on twelve "
                  "monthly lags in every case."),
        xlabel="forecast horizon across the break (months)", source=src_model,
        alt=("Four short horizontal bars, all between zero and four months: "
             + ", ".join(f"{k} {v}" for k, v in zip(labels, values)) + "."),
        caption=(f"Fig 3. Bars are sorted by length, not by how much the model is "
                 f"shown. The 'true demand series' row hands it the producer's own "
                 f"utilisation, product mix and capacity *plus* demand itself — data "
                 f"nobody outside the industry has — and the horizon is "
                 f"{res['curves'][('and the true demand series', 'lags', 'break')]['horizon']} "
                 f"months against {res['break_horizon']} for price alone. The limit "
                 f"is not what you can see. A training window cannot contain a trend "
                 f"that has not happened yet, and no feature fixes that."),
        path=str(IMG / f"a3-f3-observability.{EXT}"))
    figs["observability"] = fig_meta

    # T1 — the horizon table, as an image because Medium has no tables.
    fig_meta, _ = charts.table_image(
        table_rows(res), header=TABLE_HEADER,
        title="Horizon, by what the model sees and which regime it is in",
        subtitle=(f"Months at which normalised error crosses {THRESHOLD:g}. "
                  f"'{MAX_H}+' means the experiment ran out of horizon before the "
                  f"model ran out of skill."),
        source=src_model, mode="light", bold_cols=(2, 3), align="llrr",
        alt=("Table of forecast horizons by feature set and model class. Seven of "
             "the eight steady-trend rows reach 142 months or more and one "
             "reservoir row collapses to 8; every post-break row is between zero "
             "and three months."),
        caption=("Table 1. Read down the last two columns: the regime decides "
                 "everything, and neither the model class nor the feature set "
                 "rescues a single row. The one steady-trend row that is not near "
                 "the cap is a reservoir that failed to train on six standardised "
                 "channels at a single fixed configuration — a fragility of the "
                 "method, discussed in the text, not a fact about chips."),
        path=str(IMG / f"a3-t1-horizons.{EXT}"))
    figs["table"] = fig_meta

    # HERO — the finding is two numbers for one model.
    fig_meta, _ = charts.comparison_card(
        headline="How far ahead can you forecast a chip cycle?",
        items=[(f"{MAX_H}+ mo", "steady demand trend"),
               (f"{res['break_horizon']} mo", "after one trend break")],
        note=("Same model, same features, same training window. A saturated "
              "capacity cycle is a limit cycle and limit cycles are easy. What is "
              "hard is not the cycle."),
        footer="quantpost", mode="light",
        alt=(f"Card comparing a forecast horizon of over {MAX_H} months on a "
             f"steady demand trend against {res['break_horizon']} months after a "
             "single step change in the demand growth rate."),
        caption="",
        path=str(IMG / f"a3-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


TABLE_HEADER = ["what the model sees", "model", "steady trend", "after a break"]


def table_rows(res: dict) -> list[list[str]]:
    """The horizon table, formatted once for both the image and the body."""
    def fmt(c: dict) -> str:
        return f"{c['horizon']}+" if c["capped"] else f"{c['horizon']}"

    rows = []
    for label in VIEWS:
        for kind, pretty in (("lags", "ridge on 12 lags"), ("esn", "reservoir, 400")):
            rows.append([label, pretty,
                         fmt(res["curves"][(label, kind, "steady")]),
                         fmt(res["curves"][(label, kind, "break")])])
    return rows


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    res = analyse()
    figs = figures(res)
    dom = res["dominant"]
    fs, ss, sh = res["fast_sweep"], res["slow_sweep"], res["shock_sweep"]
    table_body = "\n".join("| " + " | ".join(r) + " |" for r in table_rows(res))

    post = Post(
        title="The Chip Cycle Is Forecastable. The Shortage Is a Different Question.",
        slug="the-chip-cycle-is-forecastable",
        subtitle=("A delayed feedback loop, an echo state network, and a result I "
                  "had the wrong way round"),
        summary=(
            f"Memory is short, and the statements being made about it run to years. "
            f"So I built the loop that generates the cycle — supply responds to "
            f"price with a lag, inventory integrates the imbalance, price responds "
            f"to inventory — and tried to measure how far ahead it can be "
            f"forecast. Three things came out, and the one I expected was wrong. "
            f"The cycle's period is set by inventory and utilisation, which move "
            f"in months, and is completely insensitive to the four-to-six-year fab "
            f"construction delay that dominates every discussion of chip supply. "
            f"Shifting a fifth of wafers onto memory that consumes three times as "
            f"much of them per gigabyte costs "
            f"{res['drag_model']['extra_wafers_pct']:.0f}% more wafers to deliver "
            f"the bits that already existed. "
            f"Saturation makes the cycle a limit cycle, so it is not hard to "
            f"forecast at all: a ridge regression on twelve monthly lags of price "
            f"holds skill out to the {MAX_H}-month limit of the experiment, and a "
            f"reservoir does no better. And one change breaks it — a single step "
            f"in the demand growth *trend* takes the same model to "
            f"{res['break_horizon']} months and makes it worse than doing nothing. "
            f"Which means a multi-year claim about a shortage is a demand-trend "
            f"claim, not a cycle claim, and the two fail in completely different "
            f"ways."),
        tags=["forecasting", "machine-learning", "semiconductors", "dynamical-systems",
              "data-science"],
        author=qp.SETTINGS.author,
        code_url=qp.SETTINGS.code_repo_url,
        min_words=1600, max_words=2500,
        table_figures=[figs["table"]],
        data_sources=[
            "A leading-edge fab takes roughly 48-72 months from groundbreaking to "
            "volume production, split as design and permitting 6-12 months, shell "
            "construction 12-18, cleanroom commissioning 9-12, tool install and "
            "qualification 12-15 and yield ramp 6-12; specialised etch and "
            "deposition tool lead times in 2026 often exceed 18 months; fab "
            "utilisation above 90% extends standard logic lead times from 16 weeks "
            "to 34-40 — SupplyICs, 'Semiconductor Fab Construction Timeline and "
            "Capacity Analysis', 2026, "
            "<https://supplyics.com/insights/supply-chain/"
            "semiconductor-fab-construction-timeline-2026/>.",
            "Memory cycles have historically run 18-30 months; the cycle running "
            "from the mid-2023 trough had reached about 36 months by July 2026. "
            "2026 bit supply growth capped near 16% against demand growth in the "
            "mid-thirties; HBM consuming about 23% of DRAM wafers with demand up "
            "70% year on year; supplier inventories 3-5 weeks and channel "
            "inventories 7-9 against a historical warning threshold of 8-10; "
            "standard DRAM module lead times 30-40+ weeks; contract prices +90-95% "
            "quarter on quarter in Q1 2026, +58-63% in Q2 and a forecast +13-18% "
            "in Q3 — Luminix, 'DRAM Cycle Mid-2026 Update', July 2026, "
            "<https://www.useluminix.com/reports/industry-analysis/"
            "dram-cycle-position-analysis-peak-timing-indicators>.",
            "Each gigabyte of HBM consumes roughly three times the wafer capacity "
            "of DDR5 — Tom's Hardware, 19 December 2025, "
            "<https://www.tomshardware.com/pc-components/ram/hbm-is-eating-your-ram>.",
            "HBM at about 16% of total memory revenue, with the wafer trade ratio "
            "falling toward 1.5 as it approaches 25%; fab cycle times growing at a "
            "14.8% compound annual rate since 2020; equipment spending per wafer "
            "area up over 150% since 2020; industry models assuming 15% annual DRAM "
            "bit growth and 95% utilisation — Semiconductor Engineering, 'From "
            "Latency To Reaction: Simulating The Next Wafer Demand Inflection', "
            "<https://semiengineering.com/"
            "from-latency-to-reaction-simulating-the-next-wafer-demand-inflection/>.",
            "The ongoing memory shortage and its framing as multi-year — "
            "'2024-present global memory supply shortage', Wikipedia, "
            "<https://en.wikipedia.org/wiki/"
            "2024%E2%80%93present_global_memory_supply_shortage>.",
            "No price series is used or redistributed. Every figure above is a "
            "published scalar; the series in this post are generated by the model "
            "described in it, at a fixed seed.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in qp.environment().items()
                if k in ("python", "numpy", "scipy", "quantpost")),
            "characteristic_polynomial":
                "z^L - (2-d) z^(L-1) + (1-d) z^(L-2) + kappa*theta*sum_k g_k "
                "z^(L-L_k), roots taken exactly; d is the reversion of price "
                "toward long-run cost",
            "undamped": (f"at d = 0 the polynomial's largest root is "
                         f"{res['undamped_max_root']:.4f} > 1, and it exceeds one "
                         f"at every positive gain and delay: a double integrator "
                         f"under delayed proportional feedback has no stable "
                         f"configuration"),
            "dominant_mode": (f"period {dom['period_months']:.1f} months, "
                              f"|z| = {dom['growth_per_step']:.4f}; the nonlinear "
                              f"run's realised period is "
                              f"{res['realised_base']:.1f} months, so saturation "
                              f"lengthens the cycle by about "
                              f"{100 * (res['realised_base'] / dom['period_months'] - 1):.0f}%"),
            "mix_drag": (
                f"moving {100 * 0.02:.0f}% to {HBM_WAFER_SHARE:.0f}% of wafers onto "
                f"product consuming {TRADE_RATIO:.0f}x the wafer per bit takes "
                f"{res['drag_model']['extra_wafers_pct']:.1f}% more wafers for the "
                f"same bits: {res['drag_model']['annual_drag_pct']:.2f}%/yr over the "
                f"model's {MIX_YEARS_MODEL:.0f}-year adoption, "
                f"{res['drag_actual']['annual_drag_pct']:.2f}%/yr over the "
                f"{MIX_YEARS_ACTUAL:.0f} years it actually took"),
            "which_delay": (f"realised period moves "
                            f"{res['fast_spread']:.0f} months across a 1-12 month "
                            f"fast delay and {res['slow_spread']:.1f} months "
                            f"across a 36-96 month capacity delay"),
            "forecast": (f"direct multi-horizon ridge, one fit per horizon, "
                         f"trained on months 0-{res['train_end']} and tested on "
                         f"{res['test'].start}-{res['test'].stop}; horizon is the "
                         f"largest h with RMSE below {THRESHOLD:g} training "
                         f"standard deviations"),
            "reservoir": (f"400 units, {ESN_KW}; the multi-channel views need the "
                          f"small input scaling to work at all, which is reported "
                          f"in the post rather than tuned away quietly"),
        },
    )

    post.add("A question about a shortage, and what kind of question it is", f"""
Memory is short. Contract prices rose **{PRICE_Q1_26[0]}-{PRICE_Q1_26[1]}%** quarter
on quarter in the first quarter of 2026, **{PRICE_Q2_26[0]}-{PRICE_Q2_26[1]}%** in
the second and a forecast {PRICE_Q3_26[0]}-{PRICE_Q3_26[1]}% in the third. Bit
supply growth for the year is capped near **{BIT_SUPPLY_GROWTH_2026:.0f}%** against
demand growth in the mid-thirties. Standard module lead times run
{MODULE_LEAD_WEEKS[0]}-{MODULE_LEAD_WEEKS[1]}+ weeks. High-bandwidth memory takes
about **{HBM_WAFER_SHARE:.0f}% of DRAM wafers** while producing about
{HBM_REVENUE_SHARE:.0f}% of memory revenue, because a gigabyte of it consumes
roughly **{TRADE_RATIO:.0f} times** the wafer capacity of a gigabyte of DDR5.

The question everyone is asking is how long this lasts, and the answers being
offered are measured in years. That is a forecast at a multi-year horizon. This
post is about what kind of object such a forecast is — not about whose is right,
which is not something arithmetic settles.

I went in expecting to find a horizon wall: a point past which the cycle is
unforecastable because it is a nonlinear feedback system, the way a weather
forecast dies at two weeks. I found the opposite, and the opposite is more useful.
""".strip())

    post.add("The loop, written down", f"""
Strip the industry to its feedback structure. Supply responds to price, but with a
lag — nobody raises utilisation or commissions a fab on today's number. Inventory
accumulates whatever imbalance is left over. Price responds to inventory.

In log deviations from a long-run path, with `p` price and `i` inventory relative to
its target:

**i_t = i_(t-1) + theta · sum_k g_k · p_(t - L_k)**

**p_t = (1 - d) · p_(t-1) - kappa · i_t**

Each `k` is a channel with its own gain `g_k` and delay `L_k`; `d` is the pull of
price toward long-run cost. Eliminate inventory and one linear delay difference
equation is left, whose characteristic polynomial is exact:

**z^L - (2-d) z^(L-1) + (1-d) z^(L-2) + kappa·theta · sum_k g_k · z^(L-L_k) = 0**

The first thing worth extracting is what happens at **d = 0**, no cost anchor. The
leading terms collapse to `(z-1)^2` — a double integrator — and a double integrator
under delayed proportional feedback is unstable at *every* positive gain and *every*
delay. The largest root at the configuration used here is
{res['undamped_max_root']:.3f}, and there is no setting of the knobs that brings it
under one.

That is not a modelling nuisance. It says the thing that stops a memory market
diverging is not restraint in capacity planning. It is that cost per bit falls on a
learning curve and price is dragged toward it. Remove the learning curve and the
arithmetic has no equilibrium to offer.

One level effect belongs here rather than in the dynamics, because the two are easy
to conflate. Stacked memory consumes about **{TRADE_RATIO:.0f} times** the wafer per
gigabyte, and it has gone from negligible to about **{HBM_WAFER_SHARE:.0f}% of DRAM
wafers**. Producing the *same* bits across that shift therefore takes
**{res['drag_model']['extra_wafers_pct']:.0f}% more wafers** — the ratio of
1 + (r-1)m at the two mix levels, and nothing more than that. Spread over the twelve
years the simulation gives it, that is
{res['drag_model']['annual_drag_pct']:.1f}% a year of capacity growth spent standing
still. The real shift took closer to four years, which puts the drag at
**{res['drag_actual']['annual_drag_pct']:.1f}% a year** against bit demand growth
that industry models put at 15%. More than half of all capacity growth, going to
deliver the bits that already existed.

With the anchor in, the dominant mode has a period of
**{dom['period_months']:.0f} months** and grows at
{100 * (dom['growth_per_step'] - 1):.1f}% a month — unstable, but bounded once
utilisation hits its ceiling and its floor. The nonlinear simulation settles into a
limit cycle of **{res['realised_base']:.0f} months**, against
{CYCLE_HIST_MONTHS[0]}-{CYCLE_HIST_MONTHS[1]} months for historical memory cycles
and about {CYCLE_NOW_MONTHS} months for the one running now.
""".strip())

    post.add("Only one of the two delays matters", f"""
There are two delays in the loop and they differ by more than an order of
magnitude. Utilisation and inventory move within a quarter. A leading-edge fab takes
**{FAB_MONTHS[0]}-{FAB_MONTHS[1]} months** from groundbreaking to volume — six to
twelve months of permitting, twelve to eighteen of shell, nine to twelve of
cleanroom commissioning, twelve to fifteen of tool install and qualification, then
six to twelve of yield ramp, with specialised tools alone on
{TOOL_LEAD_MONTHS}-month lead times.

The naive expectation, and mine, is that the long delay sets the cycle. It does not.
Vary the fast delay from one month to twelve and the realised period moves
{fs[1]['realised']:.0f} → {fs[12]['realised']:.0f} months. Vary the capacity
construction delay from {SLOW_SWEEP[0]} months to {SLOW_SWEEP[-1]} — three years to
eight — and the period moves **{res['slow_spread']:.1f} months**, which at this
sampling resolution is nothing.

The construction lead time everyone discusses leaves no signature in the price
series at all. It is a real constraint on how much capacity arrives and when, and
it is dynamically invisible in the thing people read the cycle off.

One more piece of naive intuition to discard while we are here. The period is not a
fixed multiple of the delay: it depends on loop gain too, and in this family it
ranges from roughly three to fifteen times the delay depending on how strongly
utilisation responds. So "the cycle is about two years because fabs take about two
years" is wrong twice — wrong about which delay, and wrong to think a period can be
read back as a lead time at all.
""".strip(), figures=[figs["delay"]])

    post.add("The cycle turns out to be the easy part", f"""
Here is where I expected the wall and did not get one.

The unstable mode plus saturation is a **limit cycle**, and limit cycles are
forecastable. Fit a ridge regression to twelve monthly lags of log price — nothing
else, no inventory, no capacity, no industry knowledge — and predict *h* months
ahead, one fit per horizon, out of sample. Normalised error at twelve months:
**{res['steady_nrmse12']:.2f}** standard deviations, against
{res['steady_persistence12']:.2f} for persistence. It stays under the threshold all
the way to {MAX_H} months, which is where I stopped, not where it failed.

Fig 2 contains a free consistency check I did not plan. The persistence baseline's
error collapses almost to zero at horizons near 34, 68, 102 and 136 months — because
one full cycle later, doing nothing is accidentally right. Those dips are spaced by
the cycle period, measured by a baseline that knows nothing about the model, which is
about as independent a confirmation of the {res['realised_base']:.0f}-month figure as
this setup can produce. It is also a warning about persistence as a baseline on any
periodic series: at the wrong horizon it looks unbeatable.

A 400-unit echo state network does no better. That is the honest headline for the
machine-learning half: on this series the nonlinearity buys nothing, because a
saturated limit cycle is smooth and nearly periodic and a linear map on enough lags
represents it perfectly well.

The reservoir was in fact the *fragile* option, and it is worth saying how rather
than leaving it in a footnote. At the default input scaling it failed outright on
every view with more than one channel; dropping the scaling to 0.15 fixed three of
them, and on the six-channel view it still reaches only **8 months** where the ridge
regression reaches {MAX_H}. I did not tune per view, because tuning a model per
feature set and then comparing across feature sets measures the tuning. The honest
summary is that a reservoir on heterogeneous standardised inputs needs care that a
ridge on lags does not, and that on this problem the care buys nothing.

The shock size barely matters either. Raising the monthly demand shock from 1.2% to
8% moves the period from {sh[0.012]['period']:.0f} to {sh[0.08]['period']:.0f}
months and the median three-year peak-to-trough from
{sh[0.012]['peak_to_trough']:.2f}x to {sh[0.08]['peak_to_trough']:.2f}x. The cycle
is not noise-driven. It is structural, and structure is what models are good at.

So if the endogenous cycle is this forecastable, why is anyone uncertain?
""".strip())

    post.add("One thing breaks it, and it is not the dynamics", f"""
Change the **trend**, not the cycle.

Step the rate of demand growth once, from 15% a year to
{100 * BREAK_GROWTH:.0f}% — the shape of an AI-capex regime shift, and roughly the
gap between this year's published bit supply growth of about
{BIT_SUPPLY_GROWTH_2026:.0f}% and demand growth in the mid-thirties. Everything
else is identical: same model, same features, same training window, same test
months. The two simulated histories are the same series up to the break.

The horizon goes from {MAX_H}+ months to **{res['break_horizon']} months**.

And it is worse than that, in a way worth dwelling on. At a twelve-month horizon
the post-break error is **{res['break_nrmse12']:.2f}** standard deviations against
**{res['break_persistence12']:.2f}** for persistence. The model is not merely
uninformative; it is several times *worse than not forecasting at all*. A model that
has learned a cycle confidently extrapolates it, and a confident extrapolation of
the wrong regime is worse than an honest shrug.

Then the control that decides the interpretation. Give the model more to look at:
inventory, then the producer's own utilisation, product mix and capacity, then the
*true demand series itself* — data nobody outside the industry has. Horizons across
the break, in months:

| {" | ".join(TABLE_HEADER)} |
|---|---|---|---|
{table_body}

Read the last two columns. The model class does almost nothing. The feature set does
almost nothing. A training window cannot contain a trend that has not happened yet,
and no amount of state fixes that, because the missing information is not about the
present.
""".strip(), figures=[figs["horizon"], figs["observability"]])

    post.add("What that makes a multi-year claim", f"""
Put the two halves together and the useful statement is a decomposition rather than
a forecast.

A forecast of memory prices contains two objects with completely different error
properties. The **cycle** is endogenous, structural, and easy — a linear model on a
year of monthly history tracks it as far out as I bothered to measure. The **trend**
is exogenous, and every large error in the experiment above came from it.

So when a statement runs to years, it is almost entirely a claim about the demand
trend, whatever the cycle language around it. That is worth knowing because the two
invite different scrutiny. A cycle claim can be checked against structure: what is
the loop gain, which delay dominates, where is inventory relative to the
{INV_WARNING_WEEKS[0]}-{INV_WARNING_WEEKS[1]} week threshold that historically
flagged a turn. A trend claim cannot be checked that way at all. It rests on how
much compute gets built, and no property of the cycle has anything to say about it.

There is a second thing the model is clear about and I want to state carefully,
because it cuts against the natural reading of the first half. The capacity delay
being invisible in the *price series* does not make it unimportant — it makes it
unmeasurable from prices. Capacity committed today arrives four to six years out, so
the supply side of any 2030 statement is largely already determined and the demand
side is almost entirely not. That asymmetry is the reason multi-year forecasts of
this market disagree so much, and it is structural rather than a failure of anyone's
model.

I have no view here on how long the current shortage runs, and this post contains no
opinion about any company, any share price, or any investment.
""".strip())

    post.add("Where this is a caricature", f"""
**One product, one price, one region.** Real memory is several products with
different substitution elasticities, sold under contracts of different lengths, made
by a handful of producers whose capacity decisions are strategic rather than
mechanical responses to a price. A game between three producers is a different model
and would plausibly produce longer cycles than mine.

**The nonlinear period runs long.** The linearised dominant mode says
{dom['period_months']:.0f} months and the simulation delivers
{res['realised_base']:.0f}, about
{100 * (res['realised_base'] / dom['period_months'] - 1):.0f}% longer, because
saturation holds the system at its bounds for part of each swing. So the closed-form
analysis gets the *scaling* right — which delay matters, and in which direction —
and the level out by about
{100 * (res['realised_base'] / dom['period_months'] - 1):.0f}%. Every period quoted
here is the simulated one for that reason.

**A single step in the trend is the friendliest possible break.** Real regime
changes are gradual, partly anticipated, and sometimes reversed. Gradual ones would
be easier to forecast than my step and anticipated ones much easier, so the
three-month horizon is a lower bound on what a real forecaster faces from a real
break — but the direction of the finding does not depend on the shape.

**Calibrated, not estimated.** The gains, the price response and the inventory
target are chosen so that the simulated cycle period, amplitude and inventory range
sit in the published ranges. That is calibration against a handful of scalars, not
estimation against a series, and a different calibration inside the same ranges
would move the numbers. What it would not move is the comparison, because both
columns of Table 1 come from the same calibration.

**And the horizon measure is a threshold on a curve.** I called it a horizon when
normalised error crosses {THRESHOLD:g}. Move the threshold and every number moves;
the ratio between the two regimes barely does, which is the only thing the post
leans on.
""".strip())
    return post
