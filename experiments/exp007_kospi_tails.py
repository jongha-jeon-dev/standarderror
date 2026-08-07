"""exp007 — the KOSPI's record day, and the assumption that actually broke.

Backlog: Track B, the finance-and-markets entry. Topical: on 31 July 2026 the KOSPI
rose 17.91% in a single day, the largest one-day gain in its history, four sessions
after a circuit breaker halted trading and inside the second-worst month the index
has ever had.

The argument, in one line: everyone reaches for **fat tails** to explain a day like
that, and fat tails are the smaller half of the problem. The assumption that failed
harder is **independence**, and it fails in a way that no choice of marginal
distribution can repair.

Three claims, each computed rather than asserted:

1. A +17.91% day is impossible under a Gaussian at any plausible calm-regime
   volatility — 12 to 18 standard deviations, return periods with 30-plus zeros.
   Student-t innovations fix that: the same move becomes a once-in-decades event.
   This part is closed-form, and the point of it is that fat tails are *easy*.
2. Under **any** i.i.d. model — Gaussian, Student-t, anything — the best and worst
   day of a year are adjacent with probability exactly 2/n, where n is the number of
   sessions. For 2026 to the end of July that is about 1.4%. The record gain landed
   next to the crash anyway, and so did the previous record, in October 2008. An
   i.i.d. fat-tailed model gets the *size* of the move right and the *timing*
   wrong, which is the half that costs money.
3. What clustering does to two things practitioners say. "Stay invested or you miss
   the ten best days" is true and is the same statement as "the ten best days are
   inside the crashes" — measured here on a simulated clustering process. And a VaR
   model can pass Kupiec's count test while failing Christoffersen's independence
   test on the same data, which is the failure mode that matters and the one a
   coverage number cannot see.

Data note. The published figures below are *quoted facts with sources*, not a
redistributed price series: there is no KOSPI history in this repo and none is
needed, because every calculation here takes the reported move as its input and
does mathematics to it. Everything else is simulated with a fixed seed.

Run: `quantpost run exp007_kospi_tails --publish`
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

import quantpost as qp
from quantpost.models import metrics
from quantpost.render import Post
from quantpost.viz import charts, theme

IMG = qp.SETTINGS.build_dir / "img"
EXT = os.environ.get("QUANTPOST_FIG_EXT", "png")
SEED = qp.SETTINGS.seed

# --- published facts, each with a source in `data_sources` ---------------------
RECORD_DAY = 17.91          # % — KOSPI, 31 July 2026, largest one-day gain ever
RECORD_POINTS = 1001.89     # index points on the day
RECORD_CLOSE = 6595.45
PREV_RECORD = 11.95         # % — 30 October 2008, the previous record
JULY_MONTH = -22.4          # % — July 2026, second-worst month on record
WORST_MONTH = -27.2         # % — October 1997
PEAK = 9385.0               # June 2026 peak
TROUGH = 5520.0             # July 2026 low
CIRCUIT_BREAKERS_2026 = 8   # by 28 July, the KOSPI's threshold being an 8% fall
SESSIONS_PER_YEAR = 252

# Calm-regime daily volatilities to sweep. Not estimated here — see the post's
# limitations section. The KOSPI's long-run daily standard deviation sits inside
# this range, and the conclusion is the same everywhere in it.
SIGMAS = (1.0, 1.25, 1.5, 2.0)
DFS = (3, 4, 6)             # Student-t degrees of freedom

# Simulated clustering process (GARCH(1,1) with t innovations).
N_SIM = 6000
OMEGA, ARCH, GARCH_B, SIM_DF = 0.02, 0.10, 0.88, 5.0
VAR_ALPHA = 0.01


def years(log10_years: float) -> str:
    """Powers of ten past 10,000 years, plain numbers below it.

    Shared by the rendered table and the markdown table in the body. They have to
    agree: the Hugo page shows the markdown, Medium and Notion show the image, and
    a reader comparing the two should not find different numbers.
    """
    if log10_years >= 4:
        return f"10^{log10_years:.0f}"
    return f"{10 ** log10_years:,.0f}"


def trading_sessions_2026() -> dict:
    """Sessions from 2 January to 31 July 2026, and why the count barely matters.

    Business days, ignoring Korean market holidays, so the true figure is a little
    lower. Both are reported in the post: the probability that matters scales as
    1/n, so a 10-session error moves it by well under a percentage point and the
    argument does not depend on the calendar.
    """
    n_bus = int(np.busday_count("2026-01-02", "2026-08-01"))
    return {"business_days": n_bus, "with_holidays": n_bus - 10}


def gaussian_return_period(move_pct: float, sigma_pct: float) -> dict:
    """How often a Gaussian with this daily sigma produces a move this large.

    Worked in logs throughout: the probabilities here underflow float64 by many
    orders of magnitude, and a return period of `inf` would hide the whole point,
    which is *how* impossible the move is.
    """
    z = move_pct / sigma_pct
    log10_p = float(stats.norm.logsf(z) / np.log(10.0))
    return {"z": float(z), "log10_p": log10_p,
            "log10_years": -log10_p - np.log10(SESSIONS_PER_YEAR)}


def student_return_period(move_pct: float, sigma_pct: float, df: float) -> dict:
    """Same, for Student-t innovations rescaled to the same standard deviation.

    The scaling matters: a raw t has variance df/(df-2), so comparing a t and a
    normal at the same `sigma` requires dividing by sqrt(df/(df-2)). Skipping that
    step flatters the t by giving it a larger spread as well as fatter tails, and
    then the comparison is not about tail shape at all.
    """
    if df <= 2:
        raise ValueError("t with df <= 2 has no finite variance to match")
    scale = sigma_pct / np.sqrt(df / (df - 2.0))
    log10_p = float(stats.t.logsf(move_pct / scale, df) / np.log(10.0))
    return {"log10_p": log10_p,
            "log10_years": -log10_p - np.log10(SESSIONS_PER_YEAR)}


def adjacency_probability(n: int, k: int = 1) -> float:
    """P(the year's best and worst day are within k sessions) under any i.i.d. model.

    For continuous i.i.d. returns every ordering of the n days is equally likely, so
    the positions of the maximum and the minimum are a uniformly random *ordered
    pair of distinct* positions. Counting the pairs with |i - j| <= k gives
    (2kn - k(k+1)) / (n(n-1)), which for k=1 collapses to a memorable 2/n.

    Nothing about the marginal distribution appears in that expression. That is the
    point of the whole post: swapping a normal for a t changes how big the moves
    are and changes nothing at all about when they arrive.
    """
    if n < 2 or k < 1:
        raise ValueError("need n >= 2 and k >= 1")
    k = min(k, n - 1)
    return (2 * k * n - k * (k + 1)) / (n * (n - 1))


def garch_series(n: int = N_SIM, seed: int = 5) -> dict:
    """GARCH(1,1) with Student-t shocks: fat tails *and* volatility clustering.

    Persistence is arch + garch = 0.98, which is the usual estimate on equity
    indices; that is what makes big days arrive next to big days.
    """
    rng = np.random.default_rng(seed)
    z = stats.t.rvs(SIM_DF, size=n, random_state=rng)
    z /= np.sqrt(SIM_DF / (SIM_DF - 2.0))          # unit variance shocks
    r = np.empty(n)
    h = np.empty(n)
    h[0] = OMEGA / max(1e-12, 1.0 - ARCH - GARCH_B)
    for t in range(n):
        if t:
            h[t] = OMEGA + ARCH * r[t - 1] ** 2 + GARCH_B * h[t - 1]
        r[t] = np.sqrt(h[t]) * z[t]
    return {"r": r, "h": h, "sigma_uncond": float(np.std(r))}


def clustering_evidence(sim: dict) -> dict:
    """Does a big up day follow a big down day more often than chance?

    The unconditional rate of a top-1% day is 1% by construction, so the
    conditional rates below are directly comparable to it without any modelling.
    """
    r = sim["r"]
    hi = np.quantile(r, 0.99)
    lo = np.quantile(r, 0.01)
    top = r >= hi
    bottom = r <= lo
    prev_decile = np.clip(
        np.searchsorted(np.quantile(r[:-1], np.linspace(0, 1, 11))[1:-1], r[:-1]),
        0, 9)
    rate_by_decile = [float(top[1:][prev_decile == d].mean()) for d in range(10)]
    return {
        "p_top_given_bottom": float(top[1:][bottom[:-1]].mean()),
        "p_top_uncond": float(top.mean()),
        "p_bottom_given_bottom": float(bottom[1:][bottom[:-1]].mean()),
        "lift": float(top[1:][bottom[:-1]].mean() / top.mean()),
        "lift_decile": float(rate_by_decile[0] / top.mean()),
        "rate_by_decile": rate_by_decile,
        "hi": float(hi), "lo": float(lo),
    }


def best_worst_days(sim: dict, n_days: int = 10) -> dict:
    """The "missing the ten best days" statistic, and its unspoken mirror image.

    Both are computed the way the marketing chart does it — compound the series
    with those days removed — plus where the best days actually sat: their average
    drawdown, which is the part the chart never shows.
    """
    r = sim["r"] / 100.0
    order = np.argsort(r)
    full = float(np.prod(1.0 + r))
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    drawdown = curve / peak - 1.0

    miss_best, miss_worst = [], []
    for k in range(0, n_days + 1):
        keep = np.ones(len(r), bool)
        keep[order[-k:] if k else []] = False
        miss_best.append(float(np.prod(1.0 + r[keep])))
        keep = np.ones(len(r), bool)
        keep[order[:k] if k else []] = False
        miss_worst.append(float(np.prod(1.0 + r[keep])))
    best_idx = order[-n_days:]
    return {"full": full, "miss_best": miss_best, "miss_worst": miss_worst,
            "n_days": n_days,
            "best_days_mean_drawdown": float(np.mean(drawdown[best_idx])),
            "share_of_best_in_drawdown": float(np.mean(
                drawdown[best_idx] < -0.10)),
            "years": len(r) / SESSIONS_PER_YEAR}


def _judge(breaches: np.ndarray) -> dict:
    """Both backtests plus the run length, for one VaR series."""
    kupiec = metrics.kupiec_pof(breaches, VAR_ALPHA)
    christ = metrics.christoffersen_independence(breaches)
    tr = christ.get("transitions", {})
    hazard = (tr.get("n11", 0) / max(1, tr.get("n10", 0) + tr.get("n11", 0)))
    return {"n_test": int(len(breaches)), "breaches": int(breaches.sum()),
            "rate": float(breaches.mean()), "kupiec": kupiec,
            "christoffersen": christ, "hazard": float(hazard),
            "worst_cluster": int(max(
                (len(list(g)) for k, g in itertools.groupby(breaches) if k),
                default=0))}


def var_backtest(sim: dict) -> dict:
    """Two static VaR models, one of which has the fat-tail fix already applied.

    The Gaussian model fails *both* tests, which is not the interesting result —
    it fails the count test because its tail is too thin, and everyone knows that.
    The second model takes the empirical 1% quantile of the same in-sample window,
    so the fat tail is handled non-parametrically and the count comes out right.
    It still has no idea when the breaches will arrive, and that is the point:
    the fat-tail fix repairs the number of exceptions and leaves the clustering
    exactly where it was.
    """
    r = sim["r"]
    half = len(r) // 2
    train, test = r[:half], r[half:]
    sigma = float(np.std(train))
    gauss_var = float(stats.norm.ppf(VAR_ALPHA) * sigma)
    emp_var = float(np.quantile(train, VAR_ALPHA))
    return {"sigma_in_sample": sigma, "gauss_var": gauss_var,
            "emp_var": emp_var,
            "gaussian": _judge(test < gauss_var),
            "empirical": _judge(test < emp_var)}


def analyse() -> dict:
    sessions = trading_sessions_2026()
    n = sessions["business_days"]
    sim = garch_series()
    cluster = clustering_evidence(sim)
    bw = best_worst_days(sim)
    vb = var_backtest(sim)

    gauss = {s: gaussian_return_period(RECORD_DAY, s) for s in SIGMAS}
    tdist = {(s, df): student_return_period(RECORD_DAY, s, df)
             for s in SIGMAS for df in DFS}

    # The monthly move, under the same Gaussian assumption scaled by root-time.
    month_sessions = 21
    gauss_month = {s: gaussian_return_period(
        abs(JULY_MONTH), s * np.sqrt(month_sessions)) for s in SIGMAS}

    # Monte Carlo check of the exact adjacency formula. A closed form that nobody
    # verified is a closed form that is probably wrong.
    rng = np.random.default_rng(17)
    draws = rng.random((200_000, n))
    gaps = np.abs(draws.argmax(axis=1) - draws.argmin(axis=1))
    return {
        "sessions": sessions,
        "n": n,
        "gauss": gauss,
        "gauss_month": gauss_month,
        "t": tdist,
        "adj_1": adjacency_probability(n, 1),
        "adj_3": adjacency_probability(n, 3),
        "adj_1_holidays": adjacency_probability(sessions["with_holidays"], 1),
        "adj_1_mc": float(np.mean(gaps == 1)),
        "adj_3_mc": float(np.mean(gaps <= 3)),
        "peak_to_trough": 100.0 * (TROUGH / PEAK - 1.0),
        "sim": sim,
        "cluster": cluster,
        "bw": bw,
        "var": vb,
    }


def figures(res: dict) -> dict:
    # Two source lines, because two of these figures are closed form and two are
    # simulated, and a note that claims both for all four is wrong on all four.
    src = ("Closed form from the reported move: Gaussian and variance-matched "
           "Student-t survival functions.")
    src_sim = ("Simulated: GARCH(1,1) with Student-t shocks, "
               f"{N_SIM:,} sessions, fixed seed.")
    figs = {}

    # F1 — the return period of the record day under each assumption.
    idx = list(SIGMAS)
    frame = pd.DataFrame({
        "Gaussian": pd.Series(
            [res["gauss"][s]["log10_years"] for s in SIGMAS], index=idx),
        **{f"Student-t, df={df}": pd.Series(
            [res["t"][(s, df)]["log10_years"] for s in SIGMAS], index=idx)
           for df in DFS},
    })

    def mark_ages(_fig, ax):
        m = theme.LIGHT
        for y, label in ((np.log10(1.4e10), "age of the universe"),
                         (np.log10(46), "the index's whole history")):
            ax.axhline(y, color=m.muted, lw=1.2, ls=(0, (5, 3)))
            ax.annotate(label, (0.99, y), xycoords=("axes fraction", "data"),
                        xytext=(0, 5), textcoords="offset points", ha="right",
                        va="bottom", fontsize=8.5, color=m.muted)

    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False, decorate=mark_ages,
        title=f"How often a +{RECORD_DAY}% day happens, by what you assume",
        subtitle=("Return period of the 31 July 2026 move, in years, as a power of "
                  "ten. Closed form from the assumed daily volatility on the "
                  "x axis — nothing is fitted."),
        ylabel="return period: log₁₀(years)",
        xlabel="assumed calm-regime daily volatility (%)", source=src,
        alt=(f"Four falling lines against assumed daily volatility. The Gaussian "
             f"line runs from 10 to the "
             f"{res['gauss'][SIGMAS[0]]['log10_years']:.0f} years down to 10 to "
             f"the {res['gauss'][SIGMAS[-1]]['log10_years']:.0f}, and the three "
             f"Student-t lines sit "
             f"tens of orders of magnitude below it, near or under the dashed "
             f"reference lines for the age of the universe and the length of the "
             f"index's own history."),
        caption=("Fig 1. The vertical axis is a power of ten, so the gap between "
                 "the Gaussian line and the Student-t lines is tens of orders of "
                 "magnitude. This is the easy half of the problem: choosing a "
                 "fat-tailed marginal moves a day like this from impossible to "
                 "merely rare, and costs nothing but a parameter."),
        path=str(IMG / f"b2-f1-return-period.{EXT}"))
    figs["period"] = fig_meta

    # F2 — clustering: the rate of a big up day, by yesterday's decile.
    rate = [100 * v for v in res["cluster"]["rate_by_decile"]]
    frame = pd.DataFrame({"chance of a top-1% up day today":
                          pd.Series(rate, index=list(range(1, 11)))})

    def mark_uncond(_fig, ax):
        m = theme.LIGHT
        ax.axhline(100 * res["cluster"]["p_top_uncond"], color=m.muted, lw=1.4,
                   ls=(0, (5, 3)))
        ax.set_xticks(list(range(1, 11)))

    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False, decorate=mark_uncond,
        title="The best days are the ones that follow the worst days",
        subtitle=("Simulated clustering process. Chance that today is a top-1% "
                  "day, grouped by which decile yesterday's return fell in; the "
                  "dashed line is the unconditional 1%."),
        ylabel="chance of a top-1% day today (%)",
        xlabel="yesterday's return, by decile (1 = worst)", source=src_sim,
        alt=(f"A U-shaped line: after a bottom-decile day the chance of a top-1% "
             f"day today is {res['cluster']['rate_by_decile'][0] * 100:.1f} "
             f"percent against the unconditional 1 percent dashed line, it falls "
             f"below the line for the middle deciles, and rises to "
             f"{res['cluster']['rate_by_decile'][-1] * 100:.1f} percent for the "
             f"top decile."),
        caption=(f"Fig 2. After a bottom-decile day the chance of a top-1% day is "
                 f"{res['cluster']['lift_decile']:.1f} times its unconditional "
                 f"rate; condition on a bottom-*1%* day instead and it is "
                 f"{res['cluster']['lift']:.0f} times. The "
                 "curve is U-shaped, not sloped: what predicts a big move is "
                 "another big move, in either direction. No i.i.d. model, however "
                 "fat its tails, can produce this shape."),
        path=str(IMG / f"b2-f2-clustering.{EXT}"))
    figs["clustering"] = fig_meta

    # F3 — the "ten best days" chart, with its mirror image next to it.
    bw = res["bw"]
    k = list(range(0, bw["n_days"] + 1))
    frame = pd.DataFrame({
        "if you miss the best days": pd.Series(
            [100 * (v / bw["full"] - 1) for v in bw["miss_best"]], index=k),
        "if you miss the worst days": pd.Series(
            [100 * (v / bw["full"] - 1) for v in bw["miss_worst"]], index=k),
    })
    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False,
        title="Both halves of the argument, on the same axes for once",
        subtitle=(f"Simulated {bw['years']:.0f} years. Final wealth relative to "
                  "staying fully invested, as the k largest and k smallest days "
                  "are removed."),
        ylabel="change in final wealth (%)", xlabel="days removed (k)",
        source=src_sim,
        alt=("Two lines from zero: removing the best days drops final wealth "
             "steeply into large negative values, removing the worst days raises "
             "it by a comparable amount. The two are near mirror images."),
        caption=(f"Fig 3. The left-hand argument is always shown and the "
                 f"right-hand one almost never is, and they are comparable in "
                 f"size — the one nobody draws being the larger. "
                 f"The honest reading is not \"stay invested\": it is that "
                 f"{res['bw']['share_of_best_in_drawdown'] * 100:.0f}% of those "
                 f"best days happen while the index is more than 10% below its "
                 "peak, so capturing them and sitting through the crash are one "
                 "decision, not two."),
        path=str(IMG / f"b2-f3-best-worst.{EXT}"))
    figs["bestworst"] = fig_meta

    # HERO — a *comparison* card. The post's hook is two models disagreeing by
    # thirty orders of magnitude about the same day, which is two numbers and no
    # chart; and the note carries the twist, which is that neither of them explains
    # the timing. Drawing the simulated GARCH series here would have been the
    # obvious choice and the dishonest one — at preview size a reader would take it
    # for the KOSPI.
    fig_meta, _ = charts.comparison_card(
        headline=f"A +{RECORD_DAY}% day. How often each model says that happens.",
        items=[(f"10^{res['gauss'][1.5]['log10_years']:.0f} yrs",
                "a Gaussian, at 1.5% daily volatility"),
               (f"{10 ** res['t'][(1.5, 4)]['log10_years']:,.0f} yrs",
                "the same, with fat tails")],
        emphasis=None,
        note=("And neither of them explains the part that mattered: the record "
              "gain landed next to the crash, which under any i.i.d. model — fat "
              f"tails included — has a probability of {res['adj_1'] * 100:.1f}%."),
        footer="quantpost", mode="light",
        alt=(f"Card comparing two return periods for a +{RECORD_DAY}% day: ten to "
             f"the {res['gauss'][1.5]['log10_years']:.0f} years under a Gaussian "
             f"against {10 ** res['t'][(1.5, 4)]['log10_years']:,.0f} years with "
             "fat tails."),
        caption="",
        path=str(IMG / f"b2-hero.{EXT}"))
    figs["hero"] = fig_meta

    # T1 — return periods, as an image, because Medium strips table markup.
    rows = []
    for s in SIGMAS:
        rows.append([f"{s:.2f}%", f"{res['gauss'][s]['z']:.1f}σ",
                     years(res['gauss'][s]['log10_years']),
                     years(res['t'][(s, 4)]['log10_years']),
                     years(res['t'][(s, 3)]['log10_years'])])
    fig_meta, _ = charts.table_image(
        rows, header=["assumed daily vol", "the move, in σ", "Gaussian",
                      "Student-t, df=4", "Student-t, df=3"],
        title=f"Return period of a +{RECORD_DAY}% day, in years",
        subtitle=("Closed form, not simulated. Note which column needs an "
                  "exponent."),
        source=src, mode="light", bold_cols=(2,),
        alt=(f"Table: at assumed daily volatilities from 1.00% to 2.00% the move "
             f"is between {res['gauss'][2.0]['z']:.0f} and "
             f"{res['gauss'][1.0]['z']:.0f} standard deviations. Gaussian return "
             f"periods run from ten to the "
             f"{res['gauss'][2.0]['log10_years']:.0f} up to ten to the "
             f"{res['gauss'][1.0]['log10_years']:.0f} years; the Student-t columns "
             f"are between about {10 ** res['t'][(2.0, 3)]['log10_years']:.0f} and "
             f"{10 ** res['t'][(1.0, 4)]['log10_years']:.0f} years."),
        caption=("Table 1. Every Gaussian figure here is longer than the age of "
                 "the universe. Every Student-t figure is a number you can write "
                 "down without an exponent. That is the whole case for fat tails "
                 "— and it is not the interesting part."),
        path=str(IMG / f"b2-t1-periods.{EXT}"))
    figs["table"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    res = analyse()
    figs = figures(res)
    g15 = res["gauss"][1.5]
    t4 = res["t"][(1.5, 4)]
    cl = res["cluster"]
    bw = res["bw"]
    gv = res["var"]["gaussian"]
    ev = res["var"]["empirical"]

    post = Post(
        title="The KOSPI's Record Day Was Not a Fat-Tail Problem",
        slug="the-kospi-record-day-was-not-a-fat-tail-problem",
        subtitle=("A +17.91% session, a 12-sigma move, and the assumption that "
                  "actually broke"),
        summary=(f"On 31 July 2026 the KOSPI rose {RECORD_DAY}% in a single "
                 f"session, the largest one-day gain in its history, days after a "
                 f"circuit breaker halted trading and inside the second-worst "
                 f"month the index has ever had. Under a Gaussian at 1.5% daily "
                 f"volatility that is an {g15['z']:.1f}-sigma move with a return "
                 f"period of 10^{g15['log10_years']:.0f} years. Fat tails fix "
                 f"that in one line. What they do not fix is that the best day "
                 f"landed next to the worst — which under any i.i.d. model has a "
                 f"probability of exactly 2/n, or {res['adj_1'] * 100:.1f}% for "
                 "this year."),
        tags=["investing", "quantitative-finance", "risk-management", "statistics",
              "data-science"],
        author=qp.SETTINGS.author,
        code_url=qp.SETTINGS.code_repo_url,
        min_words=1500, max_words=2400,
        table_figures=[figs["table"]],
        data_sources=[
            "KOSPI figures are quoted from published reports, not from a "
            "redistributed price series — this repo holds no index history and "
            "needs none, because every calculation takes the reported move as its "
            "input. 31 July 2026: +17.91% (+1,001.89 points) to 6,595.45, the "
            "largest one-day gain on record, surpassing +11.95% on 30 October "
            "2008 — Seoul Economic Daily, <https://en.sedaily.com/finance/2026/08/"
            "03/escaping-the-rollercoaster-kospi-index-recovers-6600-eyes> and "
            "TradingKey, <https://www.tradingkey.com/analysis/stocks/us-stocks/"
            "262067341-kospi-surged-17-9-percent-largest-single-day-gain-history-"
            "july-31-2026-tradingkey>.",
            "28 July 2026: the year's eighth circuit breaker, the KOSPI's "
            "threshold being an 8% fall, with halts on consecutive sessions for "
            "the first time — Seoul Economic Daily, <https://en.sedaily.com/"
            "finance/2026/07/28/breaking-news-kospi-triggers-circuit-breaker-8th-"
            "this-year>.",
            "July 2026 monthly return -22.4%, second only to October 1997's "
            "-27.2%; June peak near 9,385 and July low near 5,520 — TradingKey, "
            "<https://www.tradingkey.com/analysis/stocks/us-stocks/262067341-"
            "kospi-surged-17-9-percent-largest-single-day-gain-history-july-31-"
            "2026-tradingkey>.",
            "Everything else is simulated with a fixed seed: a GARCH(1,1) process "
            "with Student-t shocks, reproducible from the repo.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in qp.environment().items()
                if k in ("python", "numpy", "scipy", "quantpost")),
            "closed forms": "Gaussian and rescaled Student-t survival functions "
                            "evaluated in logs (the probabilities underflow "
                            "float64 by tens of orders of magnitude); adjacency "
                            "probability (2kn - k(k+1)) / (n(n-1))",
            "adjacency check": f"exact {res['adj_1']:.5f} against 200,000 Monte "
                               f"Carlo draws {res['adj_1_mc']:.5f}",
            "sessions": f"{res['n']} business days from 2026-01-02 to 2026-07-31 "
                        f"(ignoring Korean market holidays; at "
                        f"{res['sessions']['with_holidays']} sessions the "
                        f"adjacency probability is "
                        f"{res['adj_1_holidays'] * 100:.2f}% instead of "
                        f"{res['adj_1'] * 100:.2f}%)",
            "simulation": f"GARCH(1,1), omega={OMEGA}, arch={ARCH}, "
                          f"garch={GARCH_B}, t({SIM_DF:.0f}) shocks, "
                          f"{N_SIM:,} sessions "
                          f"({N_SIM / SESSIONS_PER_YEAR:.0f} years), "
                          f"unconditional daily sd "
                          f"{res['sim']['sigma_uncond']:.2f}%",
            "VaR backtest": f"two static VaR models at {VAR_ALPHA:.0%} — a "
                            f"Gaussian quantile and the empirical 1st percentile "
                            f"— both estimated on the first "
                            f"{N_SIM // 2:,} sessions and tested on the next "
                            f"{N_SIM // 2:,}",
        },
    )

    post.add("A day that no risk model contains", f"""
On 31 July 2026 the KOSPI closed up **{RECORD_DAY}%** — {RECORD_POINTS:,.2f} points,
to {RECORD_CLOSE:,.2f}. It is the largest one-day gain in the index's history,
ahead of the {PREV_RECORD}% it managed on 30 October 2008. It happened days after
trading was halted by a circuit breaker for the eighth time this year, and inside a
month that fell {abs(JULY_MONTH)}% — the second-worst month the index has ever had,
behind October 1997.

Take that {RECORD_DAY}% and ask the question a risk system asks. If daily returns
were Gaussian with a calm-regime standard deviation of 1.5% — a typical figure for a
large equity index outside a crisis — the move is **{g15['z']:.1f} standard
deviations**. The Gaussian probability of that is 10^{g15['log10_p']:.0f}, which
works out to a return period of **10^{g15['log10_years']:.0f} years**. The universe
is about 10^10 years old.

So the model is wrong, and everybody already knows the first answer: **fat tails**.
Replace the normal with a Student-t and the same move stops being absurd. At four
degrees of freedom, matched to the same 1.5% volatility, the return period falls
from 10^{g15['log10_years']:.0f} years to about
**{10 ** t4['log10_years']:,.0f} years** — from unimaginable to something a
long-lived institution has already lived through.

That fix is real and it is one line of code. It is also the smaller half of the
problem, and this post is about the larger half.
""".strip(), figures=[figs["period"]])

    table_body = "\n".join(
        f"| {sig:.2f}% | {res['gauss'][sig]['z']:.1f}σ | "
        f"{years(res['gauss'][sig]['log10_years'])} | "
        f"{years(res['t'][(sig, 4)]['log10_years'])} | "
        f"{years(res['t'][(sig, 3)]['log10_years'])} |" for sig in SIGMAS)
    post.add("Fat tails are the easy half", f"""
The table below is the whole fat-tail argument, and I want it out of the way early
because it is not in dispute. Each row assumes a different calm-regime daily
volatility and reports the return period of a +{RECORD_DAY}% day under three
distributions.

| assumed daily vol | the move, in σ | Gaussian | Student-t, df=4 | Student-t, df=3 |
|---|---|---|---|---|
{table_body}

Two things to notice. The Gaussian column is absurd everywhere in the range, so no
amount of arguing about the right volatility rescues it — the failure is the shape,
not the parameter. And — this one surprised me — the *monthly*
figure is not extreme at all. Scale 1.5% daily by the square root of 21 sessions and
{abs(JULY_MONTH)}% in a month is {res['gauss_month'][1.5]['z']:.1f} standard
deviations, a Gaussian return period of about
{10 ** res['gauss_month'][1.5]['log10_years']:.0f} years. Nothing to report.

Which is its own small lesson about reporting frequency. Aggregate to months and the
central limit theorem quietly does its work: the sum of twenty-one wild days looks
almost well-behaved, and a monthly risk report would have shown a bad-but-ordinary
month where the daily data was screaming. If your tail statistics are computed on
monthly data, they are not tail statistics.

Fine. Use a Student-t, or an extreme-value tail, or a jump. All of them are
improvements, all of them are standard, and every risk textbook written since 1963
says so. Now here is what none of them do.
""".strip())

    post.add("The best day of the century, one session after the worst", f"""
The KOSPI's record gain did not arrive on a quiet Tuesday. It arrived in the middle
of the crash, in the same week as a trading halt, immediately after the index had
fallen roughly {abs(res['peak_to_trough']):.0f}% from its June peak. The previous
record — that {PREV_RECORD}% in October 2008 — also arrived in the middle of a
crash. Both of the two largest one-day gains in this index's history happened while
it was falling apart.

That pattern has a probability attached, and it is worth computing, because the
computation does not depend on any distribution at all.

Suppose daily returns are i.i.d. — drawn independently from the same distribution,
whatever it is. Then every ordering of this year's returns is equally likely, so
the position of the best day and the position of the worst day are just a random
ordered pair of distinct days. Counting the pairs that are neighbours gives a
memorably simple answer: **2/n**, where n is the number of sessions. For 2026 up to
the end of July, {res['n']} business days, that is
**{res['adj_1'] * 100:.1f}%**. Widen it to within three sessions and it is
{res['adj_3'] * 100:.1f}%. (I checked the formula against 200,000 simulated years:
{res['adj_1_mc'] * 100:.2f}% against the exact {res['adj_1'] * 100:.2f}%.)

Notice what is *not* in that expression. No volatility. No degrees of freedom. No
tail index. **Switching from a normal to a Student-t changes how large the moves
are and changes nothing whatsoever about when they arrive.** You can fit the
fattest tail in the literature and your model will still put the best day of the
decade on a random calendar date, uncorrelated with the worst.

Real markets do not work that way, and the mechanism is not mysterious: volatility
clusters. A shock raises tomorrow's expected volatility, which raises the chance of
a large move tomorrow, in *either* direction. Fit that and the pattern appears. In
a simulated series with clustering and nothing else — a plain GARCH(1,1) with
Student-t shocks — the chance that today is a top-1% *up* day, given that yesterday
was a bottom-1% *down* day, is {cl['p_top_given_bottom'] * 100:.1f}%, against an
unconditional {cl['p_top_uncond'] * 100:.1f}%. That is
{cl['lift']:.0f} times the base rate, with no skew, no asymmetry and no narrative
attached — just persistence in the variance. Group by yesterday's decile instead and
the shape is a U, not a slope: the bottom decile gives
{cl['rate_by_decile'][0] * 100:.1f}% and the top decile
{cl['rate_by_decile'][-1] * 100:.1f}%, while the six middle deciles sit at or below
the unconditional rate. What predicts a big move is another big move, in either
direction.
""".strip(), figures=[figs["clustering"]])

    post.add("Which quietly demolishes an argument you have heard", f"""
"Stay invested. Miss the ten best days of the last twenty years and you lose most
of your return." That chart is in every fund brochure, and it is arithmetically
true. It is also half of a sentence.

On my simulated {bw['years']:.0f} years: removing the ten best days costs
**{abs(100 * (bw['miss_best'][-1] / bw['full'] - 1)):.0f}%** of final wealth.
Removing the ten worst days *adds*
**{100 * (bw['miss_worst'][-1] / bw['full'] - 1):.0f}%**. In compounding terms those
are {abs(np.log(bw['miss_best'][-1] / bw['full'])):.2f} and
{np.log(bw['miss_worst'][-1] / bw['full']):.2f} in logs — the same order of
magnitude, the mirror argument if anything the larger one, and only one of the two
lines has ever been drawn for a retail investor.

The clustering result tells you why you cannot have one without the other.
**{bw['share_of_best_in_drawdown'] * 100:.0f}% of those ten best days occur while
the index is more than 10% below its previous peak**, with an average drawdown of
{abs(bw['best_days_mean_drawdown']) * 100:.0f}% at the moment they happen. The best
days are not scattered through the good times. They are inside the crashes,
because that is where the volatility is. 31 July 2026 is the cleanest possible
illustration: to have captured the largest single-day gain in the index's history,
you had to be holding through a week that included two consecutive circuit breakers.

So "stay invested to capture the best days" is not a claim about upside. It is a
claim about being able to tolerate the downside, and it should be argued on those
terms.
""".strip(), figures=[figs["bestworst"]])

    post.add("How a risk model fails the test it passes", f"""
This is the part that matters if you own a model rather than a portfolio, and it is
why I care more about clustering than about tails.

On the same simulated series I built two static value-at-risk models at
{VAR_ALPHA:.0%}, both estimated on the first half and both judged on the second, and
ran the two standard backtests on each.

The first is the naive one: a Gaussian VaR from the in-sample standard deviation. It
fails **Kupiec's proportion-of-failures test**, which asks only whether the *number*
of breaches is right — {gv['rate'] * 100:.2f}% observed against
{VAR_ALPHA:.0%} expected, {gv['breaches']} breaches where
{VAR_ALPHA * gv['n_test']:.0f} were expected, p = {gv['kupiec']['p_value']:.1e}. No
surprise: thin tail, too many exceptions. This is the failure the fat-tail
literature exists to fix.

So I fixed it, in the cheapest possible way: take the empirical 1st percentile of the
in-sample returns instead of a Gaussian quantile. No distributional assumption at
all, the fat tail handled non-parametrically. And it works — on the count:

- breaches {ev['rate'] * 100:.2f}% against {VAR_ALPHA:.0%} expected,
  **Kupiec p = {ev['kupiec']['p_value']:.2f}**. Passes comfortably.
- **Christoffersen's independence test**, which asks whether the breaches arrive
  spread out or together: **p = {ev['christoffersen']['p_value']:.1e}**. Fails
  outright.
- The probability of a breach tomorrow given a breach today is
  {ev['hazard'] * 100:.0f}%, against the {VAR_ALPHA:.0%} the model implies. The
  longest run of consecutive breaches is {ev['worst_cluster']}.

Same data, same model, two tests, opposite verdicts — and the fat-tail fix moved the
first verdict without touching the second. A capital buffer sized for
{VAR_ALPHA * ev['n_test']:.0f} breaches spread across
{ev['n_test'] / SESSIONS_PER_YEAR:.0f} years is not sized for
{ev['worst_cluster']} of them in {ev['worst_cluster']} consecutive days. The count
was never the risk. The arrival pattern was.

The remedy is not exotic — let the volatility move (GARCH, EWMA, a realised
estimator, anything conditional) and report expected shortfall next to the quantile
so the *size* of a breach counts and not just the fact of it. But the diagnostic
matters more than the model: **run the independence test, and put its p-value beside
the coverage number.** A dashboard reporting "VaR exceptions: {ev['breaches']},
expected {VAR_ALPHA * ev['n_test']:.0f}" is reporting the test the model passes.
""".strip())

    post.add("Where this argument is thinner than it looks", f"""
Four things I am not claiming.

**I did not estimate the volatility.** The repo holds no KOSPI price series — the
figures here are quoted from published reports, so the calm-regime volatility is an
assumption I swept across a range rather than a number I measured. That is why
Table 1 has four rows instead of one. The conclusion is stable across the range, but
"σ was really 3% by late July" is a legitimate objection, and the honest answer is
that a conditional model would have *known* that, which is the post's point rather
than a hole in it.

**One index, one episode.** Two record gains inside two crashes is a pattern with
two observations in it. The clustering evidence in Figs 2 and 3 is simulated, not
sampled from the KOSPI: it shows that a standard clustering model reproduces the
pattern, not that the KOSPI's parameters are the ones I chose.

**GARCH is not the truth either.** It reproduces volatility clustering and fat
tails, and it still assumes a fixed structure that no market has agreed to. It has
no jumps, no regime changes, and a symmetric response to good and bad news that
equity indices are known to violate.

**2/n assumes continuous i.i.d. returns.** With ties, or with a deterministic
calendar effect, the counting changes slightly. It does not change by anything that
matters at n ≈ {res['n']}: at {res['sessions']['with_holidays']} sessions the
figure is {res['adj_1_holidays'] * 100:.2f}% rather than
{res['adj_1'] * 100:.2f}%.

What survives all four is narrow and, I think, useful: the arithmetic that makes a
+{RECORD_DAY}% day impossible is about the shape of the distribution, and the
arithmetic that makes it land next to the crash is about dependence. The first is
what everyone reaches for. The second is where the money is.

Next in this series, back to dynamics: what a neural network is doing when it looks
like it has learned physics, and how to tell that apart from having memorised a
trajectory.
""".strip())

    return post


if __name__ == "__main__":
    p = build()
    print(p.title, "|", p.word_count(), "words |", len(p.figures), "figures")
    for issue in p.audit():
        print("  audit:", issue)
