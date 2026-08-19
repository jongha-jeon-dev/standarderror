"""exp012 — a paper about when news is priced in, and what its numbers are worth.

Backlog: Track A, and the series' first paper walkthrough. Requested directly:
summarise Kargarzadeh et al., "Buy the Rumor, Sell the News: When Is News Priced
In?" (arXiv:2608.14014, 14 August 2026), and make it readable.

A summary alone would be thin here, so the post does the summary properly and then
adds the one thing a summary cannot: the arithmetic that turns the paper's effect
sizes into units a reader can judge. The paper is careful to make no trading claim.
This post supplies the translation and then shows why declining to make the claim
was the right call.

What the paper measures
-----------------------
4.57 million articles, roughly 3,000 US stocks, 2023-2026. A large language model
labels a sample; a compact classifier is distilled from it by active learning and
labels the rest with 17 event tags and 5 attributes. Articles are clustered into
stories so a first report is distinguished from follow-up coverage. Beta-adjusted
abnormal returns are measured around 1.68 million stock-day events, against 364,405
neutral-sentiment events as a placebo.

Three findings:

1. The news-aligned move is mostly **before** the headline. Pooled over signed
   events the cumulative abnormal return reaches **+0.58%** by the close of
   publication day and stands at **+0.20%** twenty days later — a ratio of 2.8, and
   two thirds of the move handed back. For rumour-flagged events the rumour day
   delivers +0.36% and the later confirmation delivers +0.01%.
2. **Numbers drift, stories revert.** Placebo-adjusted, over days +6 to +20:
   capital returns +0.35%, earnings +0.22%, guidance +0.13%, analyst actions +0.10%;
   macro commentary -0.34%, launches -0.18%, leadership -0.18%, competition -0.17%.
3. News has **width** as well as direction: volatility rises ahead of publication
   and falls once the news is out, because publication resolves uncertainty.

What this post adds
-------------------
One number does the work, and it is not the paper's: S&P's dispersion dashboard put
average pairwise correlation among S&P 500 constituents at **0.05** on 31 July 2026,
with implied constituent volatility at 44.42. That number lands twice, in opposite
directions:

* At rho = 0.05 the systematic share of a single stock's variance is tiny, so
  beta-adjusting an abnormal return removes only about 2.5% of its volatility. The
  noise a 22-basis-point effect sits in is about **10.6% over fifteen trading
  days**. Hence a per-event Sharpe of 0.021, and hence the scale of the study: each
  tag's reported p-value implies a minimum sample, and analyst actions at +0.10% and
  p = 0.012 need **over 70,000 events**. You cannot find these effects with
  thousands of articles. The paper needed millions, and the arithmetic says so
  independently.
* The same rho caps what the effects are worth. Holding n positions with pairwise
  residual correlation rho gives n / (1 + (n-1) rho) independent ones, which
  converges to **1/rho**. At 0.05 that ceiling is twenty, 90% reached by 171
  names, and annualised Sharpe pins near **0.38** however many more you add — 0.20
  once a 10-basis-point round trip is taken out. At rho = 0 the same edge would give
  1.9 at five hundred names and keep climbing.

And the cost statement, which needs no citation because it is an identity: the
round-trip cost that kills a gross edge *is* the gross edge. Every tag in the
paper's table dies below 35 basis points, and the median one below 18.

So the effects are real, carefully measured, and small — and the reason to know
their size is not to trade them but to know how much of a forecast they can carry.

Nothing here re-analyses the paper's data, which is not public. Every input is a
published scalar, quoted with its source. No investment advice, and no claim about
any company.

Run: `quantpost run exp012_priced_in --publish`
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import quantpost as qp
from quantpost.render import Post
from quantpost.uq import evidence
from quantpost.viz import charts, theme

IMG = qp.SETTINGS.build_dir / "img"
EXT = os.environ.get("QUANTPOST_FIG_EXT", "png")
SEED = qp.SETTINGS.seed

# --- the paper's reported figures, sourced in `data_sources` --------------------
PAPER = "Kargarzadeh, Khaledian, Parvini, Ghatak and Khaledian (arXiv:2608.14014)"
N_ARTICLES = 4_570_000
N_STOCKS = 3_000
N_EVENTS = 1_680_000
N_PLACEBO = 364_405
N_TAGS = 17
N_ATTRS = 5
YEARS = "2023-2026"
CAR_AT_PUBLICATION = 0.58        # %, cumulative from day -5 to the close of day 0
CAR_AT_20 = 0.20                 # %, cumulative from day -5 to day +20
CAR_RATIO = 2.8                  # as the paper reports it
# Rumour-flagged events later followed by a non-rumour report within 60 days.
RUMOUR = {"the rumour day": 0.36, "the wait": -0.09,
          "the confirmation day": 0.01, "days +6 to +20": -0.06}
FOLLOWUP_INFLATION = 1.0 / 3.0   # the paper's own estimate of anticipation inflation
# Placebo-adjusted drift over days +6 to +20, with the reported p-value.
TAGS = {
    "capital returns": (0.35, 0.001), "earnings results": (0.22, 0.001),
    "guidance / outlook": (0.13, 0.046), "analyst action": (0.10, 0.012),
    "macro through stock": (-0.34, 0.001), "product launch": (-0.18, 0.022),
    "leadership": (-0.18, 0.016), "competition": (-0.17, 0.026),
}
QUANTIFIED = ("capital returns", "earnings results", "guidance / outlook",
              "analyst action")
BOUNDED_P = {0.001}              # reported as "< 0.001", so |z| is a lower bound

# --- the outside number this post turns on -------------------------------------
VIXEQ = 44.42                    # S&P 500 implied constituent volatility, 31 Jul 2026
PAIRWISE_RHO = 0.05              # average pairwise correlation, same date
DISPERSION = 41.42               # S&P 500 Dispersion Index, same date
DISPERSION_HIGH = 47.51          # its all-time high, 21 July 2026
VIX_CLOSE = 15.99
WINDOW_DAYS = 15                 # days +6 to +20 inclusive
TRADING_YEAR = 252

RHO_SWEEP = (0.0, 0.02, 0.05)
N_SWEEP = tuple(int(round(x)) for x in np.unique(np.round(
    np.geomspace(5, 3000, 22))))
COST_SWEEP = (0.0, 0.05, 0.10, 0.20)   # % round trip


def event_noise(vixeq: float = VIXEQ, rho: float = PAIRWISE_RHO,
                days: int = WINDOW_DAYS) -> dict:
    """Volatility a single event's abnormal return sits in, over the drift window.

    Two steps, and the second is the interesting one. Scale the annualised implied
    constituent volatility to the window; then remove the systematic part, which at
    an average pairwise correlation of `rho` is a share `rho` of the variance. At
    rho = 0.05 that removal is worth almost nothing — which is the first of the two
    places this number lands.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    total = vixeq * np.sqrt(days / TRADING_YEAR)
    idio_share = np.sqrt(1.0 - rho)
    return {"annual_total": vixeq,
            "annual_idio": vixeq * idio_share,
            "window_total": float(total),
            "window_idio": float(total * idio_share),
            "beta_adjustment_saves_pct": float(100.0 * (1.0 - idio_share))}


def analyse() -> dict:
    noise = event_noise()
    sigma = noise["window_idio"]
    periods = TRADING_YEAR / WINDOW_DAYS

    per_tag = {}
    for tag, (drift, p) in TAGS.items():
        s = evidence.per_observation_sharpe(drift, sigma)
        per_tag[tag] = {
            "drift": drift, "p": p, "bounded": p in BOUNDED_P,
            "per_event_sharpe": s,
            "implied_n": evidence.implied_n(drift, sigma, p),
            "breakeven_bp": 100.0 * evidence.breakeven_cost(drift),
            "quantified": tag in QUANTIFIED,
        }

    # Sharpe against breadth, at several residual correlations. The point of the
    # sweep is the ceiling, not any single curve.
    headline = TAGS["earnings results"][0]
    s_head = evidence.per_observation_sharpe(headline, sigma)
    curves = {}
    for rho in RHO_SWEEP:
        label = ("uncorrelated residuals" if rho == 0
                 else f"residual correlation {rho:g}")
        curves[label] = [
            evidence.annualised_sharpe(s_head, n, rho, periods_per_year=periods)
            for n in N_SWEEP]

    ceilings = {rho: (float("inf") if rho == 0 else 1.0 / rho)
                for rho in RHO_SWEEP}
    at_500 = {rho: evidence.annualised_sharpe(s_head, 500, rho,
                                              periods_per_year=periods)
              for rho in RHO_SWEEP}
    # How quickly the ceiling binds: effective breadth as a fraction of its limit.
    bind = {}
    for rho in RHO_SWEEP:
        if rho == 0:
            continue
        n_90 = next(n for n in range(1, 100_000)
                    if evidence.effective_independent(n, rho) >= 0.9 / rho)
        bind[rho] = n_90

    net = {}
    for cost in COST_SWEEP:
        kept = max(0.0, headline - cost)
        net[cost] = {
            "net_drift": kept,
            "sharpe": evidence.annualised_sharpe(
                evidence.per_observation_sharpe(kept, sigma), 500,
                PAIRWISE_RHO, periods_per_year=periods),
        }

    quantified_mean = float(np.mean([v["drift"] for k, v in per_tag.items()
                                     if v["quantified"]]))
    story_mean = float(np.mean([v["drift"] for k, v in per_tag.items()
                                if not v["quantified"]]))
    return {
        "noise": noise, "sigma": sigma, "periods": periods, "per_tag": per_tag,
        "curves": curves, "ceilings": ceilings, "at_500": at_500, "bind": bind,
        "net": net, "s_head": s_head,
        "quantified_mean": quantified_mean, "story_mean": story_mean,
        "given_back_pct": 100.0 * (CAR_AT_PUBLICATION - CAR_AT_20)
        / CAR_AT_PUBLICATION,
        "max_implied_n": max(v["implied_n"] for v in per_tag.values()),
        "max_implied_tag": max(per_tag, key=lambda k: per_tag[k]["implied_n"]),
        "median_breakeven_bp": float(np.median(
            [v["breakeven_bp"] for v in per_tag.values()])),
        "max_breakeven_bp": max(v["breakeven_bp"] for v in per_tag.values()),
        "events_per_tag": N_EVENTS / N_TAGS,
        "rumour_total": sum(RUMOUR.values()),
    }


TABLE_HEADER = ["event tag", "drift, +6 to +20", "p", "events the p needs",
                "dies at"]


def table_rows(res: dict) -> list[list[str]]:
    """The tag table, formatted once for the image and the markdown body."""
    rows = []
    for tag, v in sorted(res["per_tag"].items(), key=lambda kv: -kv[1]["drift"]):
        rows.append([
            tag,
            f"{v['drift']:+.2f}%",
            ("<0.001" if v["bounded"] else f"{v['p']:.3f}"),
            (f">{v['implied_n']:,.0f}" if v["bounded"] else f"{v['implied_n']:,.0f}"),
            f"{v['breakeven_bp']:.0f} bp",
        ])
    return rows


def figures(res: dict) -> dict:
    src_paper = (f"Effect sizes from {PAPER}. Volatility and correlation from "
                 f"S&P Dow Jones Indices, 31 July 2026.")
    figs = {}

    # F1 — the rumour decomposition. Four published numbers that tell the story on
    # their own, which is why this is the chart rather than a CAR path: the paper
    # reports two points on that path, and drawing a curve through two points would
    # be inventing the shape.
    # sort="none" because these are four consecutive stretches of time, not a
    # ranking: sorting a timeline puts the confirmation day before the wait that
    # preceded it. Reversed because barh puts the first row at the bottom, and a
    # timeline should read downwards like the sentence describing it.
    labels = list(RUMOUR.keys())[::-1]
    fig_meta, _ = charts.ranked_bars(
        labels, [RUMOUR[k] for k in labels], mode="light", signed=True,
        sort="none", value_fmt="+.2f",
        title="The rumour day gets the move; the confirmation gets nothing",
        subtitle=("Abnormal return in the rumour's direction, for rumour-flagged "
                  "events later followed by a non-rumour report within 60 days."),
        xlabel="abnormal return in the rumour's direction (%)",
        source=src_paper,
        alt=("Four horizontal bars: the rumour day +0.36%, the wait -0.09%, the "
             "confirmation day +0.01%, days +6 to +20 -0.06%."),
        caption=(f"Fig 1. This is the saying, measured. The rumour day carries "
                 f"**{RUMOUR['the rumour day']:+.2f}%**; the day the story is "
                 f"actually confirmed carries "
                 f"**{RUMOUR['the confirmation day']:+.2f}%** — inside rounding of "
                 f"nothing. Everything after the rumour, added up, is "
                 f"{res['rumour_total'] - RUMOUR['the rumour day']:+.2f}%."),
        path=str(IMG / f"a4-f1-rumour.{EXT}"))
    figs["rumour"] = fig_meta

    # F2 — the tag table as a chart, because the sign split is the finding.
    order = sorted(res["per_tag"], key=lambda k: -res["per_tag"][k]["drift"])
    fig_meta, _ = charts.ranked_bars(
        order, [res["per_tag"][k]["drift"] for k in order], mode="light",
        signed=True, sort="value", value_fmt="+.2f",
        title="Numbers keep drifting. Stories give it back.",
        subtitle=("Placebo-adjusted abnormal return over days +6 to +20 after "
                  "publication, by event type. Positive means the price kept "
                  "moving the way the news pointed."),
        xlabel="drift over days +6 to +20 (%)", source=src_paper,
        alt=("Eight horizontal bars split by sign. Capital returns, earnings, "
             "guidance and analyst actions are positive, from +0.10% to +0.35%; "
             "macro commentary, launches, leadership and competition are negative, "
             "from -0.17% to -0.34%."),
        caption=(f"Fig 2. The four positive tags are the ones carrying a number — "
                 f"a dividend, an earnings figure, a guidance range, a target "
                 f"price — and they average "
                 f"{res['quantified_mean']:+.2f}%. The four negative ones are "
                 f"stories, and they average {res['story_mean']:+.2f}%. Same "
                 f"market, same window, opposite sign."),
        path=str(IMG / f"a4-f2-tags.{EXT}"))
    figs["tags"] = fig_meta

    # F3 — the part the paper does not claim and this post supplies.
    frame = pd.DataFrame(res["curves"], index=list(N_SWEEP))

    def mark_ceiling(_fig, ax):
        m = theme.LIGHT
        for rho in RHO_SWEEP:
            if rho == 0:
                continue
            sr = evidence.annualised_sharpe(res["s_head"], 10 ** 6, rho,
                                            periods_per_year=res["periods"])
            ax.axhline(sr, color=m.muted, lw=1.0, ls=(0, (2, 3)))
        ax.annotate("each dashed line is that correlation's ceiling:\n"
                    "1/rho independent bets, however many names you hold",
                    (0.98, 0.02), xycoords="axes fraction", ha="right",
                    va="bottom", fontsize=8.5, color=m.muted, linespacing=1.4)
        # A log axis defaults to labelling one decade, and the whole chart is a
        # comparison of values inside that decade.
        from matplotlib.ticker import FixedLocator, ScalarFormatter
        ax.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 1.0, 2.0, 4.0]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(FixedLocator([]))

    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False, decorate=mark_ceiling,
        logx=True, logy=True,
        title="Breadth stops helping almost immediately",
        subtitle=(f"Annualised Sharpe from the earnings drift of "
                  f"{TAGS['earnings results'][0]:+.2f}% over fifteen days, against "
                  f"the number of positions held at once, at {len(RHO_SWEEP)} "
                  f"residual correlations. Before costs."),
        ylabel="annualised Sharpe ratio (log scale)",
        xlabel="positions held simultaneously (log scale)", source=src_paper,
        alt=("Four rising curves on log axes. The uncorrelated one keeps climbing "
             "past a Sharpe of 3; the three correlated ones flatten early, the "
             "0.05 curve pinning just under 0.4 from a few hundred positions "
             "onward."),
        caption=(f"Fig 3. The paper makes no trading claim and this is why it was "
                 f"right not to. With uncorrelated residuals the earnings drift "
                 f"reaches a Sharpe of {res['at_500'][0.0]:.1f} at five hundred "
                 f"names and keeps going. At the 0.05 average pairwise correlation "
                 f"S&P reported for July 2026 it pins at "
                 f"{res['at_500'][0.05]:.2f}, and it gets there by "
                 f"{res['bind'][0.05]} names — past which breadth is free and "
                 f"worthless. All of this is before the costs in Table 1."),
        path=str(IMG / f"a4-f3-breadth.{EXT}"))
    figs["breadth"] = fig_meta

    fig_meta, _ = charts.table_image(
        table_rows(res), header=TABLE_HEADER,
        title="What each of the paper's effects needs, and what kills it",
        subtitle=(f"Drift and p-value from the paper. The last two columns are this "
                  f"post's arithmetic against a {res['sigma']:.1f}% fifteen-day "
                  f"idiosyncratic volatility."),
        source=src_paper, mode="light", bold_cols=(1, 4), align="lrrrr",
        alt=("Table of eight event tags with drift, p-value, the minimum number of "
             "events consistent with that p-value, and the round-trip cost at "
             "which the gross edge is consumed. Implied event counts run from about "
             "10,000 to over 70,000; break-even costs from 10 to 35 basis points."),
        caption=("Table 1. The fourth column is a free consistency check on someone "
                 "else's result: a reported p-value implies a minimum sample, and "
                 "every one of these is comfortably inside what 1.68 million events "
                 "across seventeen tags provides. The fifth is the round-trip cost "
                 "at which each edge is exactly gone."),
        path=str(IMG / f"a4-t1-tags.{EXT}"))
    figs["table"] = fig_meta

    # HERO — a two-panel strip, because the finding *is* a setup and a punchline:
    # the whisper moves the price, the announcement does not. Panels carry no axes
    # and no values, so nothing in them can be read as a measurement.
    def whisper(panel, m):
        # A small speech bubble, and a price line that steps up under it.
        from matplotlib.patches import FancyBboxPatch, Polygon
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 6)
        panel.add_patch(FancyBboxPatch((0.7, 3.5), 5.0, 1.7,
                                       boxstyle="round,pad=0.18,rounding_size=0.5",
                                       fc=m.surface, ec=m.ink, lw=2.0))
        panel.add_patch(Polygon([[2.0, 3.5], [2.5, 2.5], [3.1, 3.5]],
                                closed=True, fc=m.surface, ec=m.ink, lw=2.0))
        panel.text(3.2, 4.35, "psst... they\nmight buy it", ha="center",
                   va="center", fontsize=12.5, color=m.ink, linespacing=1.2)
        x = np.linspace(6.2, 9.5, 60)
        y = 1.2 + 2.6 / (1.0 + np.exp(-(x - 7.6) * 2.2))
        panel.plot(x, y, color=m.series[0], lw=3.0, solid_capstyle="round")
        panel.plot([0.7, 5.9], [1.2, 1.2], color=m.muted, lw=1.6,
                   ls=(0, (4, 3)))

    def announcement(panel, m):
        # A front page shouting the same news, and a price line that does nothing.
        from matplotlib.patches import Rectangle
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 6)
        panel.add_patch(Rectangle((0.7, 2.3), 5.0, 3.0, fc=m.surface, ec=m.ink,
                                  lw=2.2))
        panel.add_patch(Rectangle((1.05, 4.35), 4.3, 0.72, fc=m.ink, ec="none"))
        # path_effects=[] because xkcd mode puts a white stroke around every text
        # object, which turns white-on-black lettering into mush.
        panel.text(3.2, 4.71, "IT IS OFFICIAL", ha="center", va="center",
                   fontsize=12.5, color=m.surface, path_effects=[])
        for k, w in enumerate((4.3, 3.9, 4.1, 3.4)):
            panel.plot([1.05, 1.05 + w], [3.95 - k * 0.42] * 2, color=m.muted,
                       lw=1.8, solid_capstyle="round")
        x = np.linspace(6.2, 9.5, 60)
        panel.plot(x, np.full_like(x, 3.8), color=m.series[1], lw=3.0,
                   solid_capstyle="round")

    fig_meta, _ = charts.strip_card(
        headline="Where is the price move when news is published?",
        panels=[(whisper, f"{RUMOUR['the rumour day']:+.2f}%", "the rumour day"),
                (announcement, f"{RUMOUR['the confirmation day']:+.2f}%",
                 "the day it is confirmed")],
        note=(f"Abnormal return in the news direction, across "
              f"{N_EVENTS / 1e6:.2f} million stock-day events. Pooled over every "
              f"signed event, {res['given_back_pct']:.0f}% of the move is handed "
              f"back after publication."),
        footer="quantpost", mode="light",
        alt=("A two-panel hand-drawn strip. In the first a small speech bubble "
             "whispers that someone might buy it, and a price line steps up: "
             f"{RUMOUR['the rumour day']:+.2f}%. In the second a newspaper front "
             "page announces it officially and the price line is flat: "
             f"{RUMOUR['the confirmation day']:+.2f}%."),
        caption="",
        path=str(IMG / f"a4-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    res = analyse()
    figs = figures(res)
    pt = res["per_tag"]
    noise = res["noise"]
    table_body = "\n".join("| " + " | ".join(r) + " |" for r in table_rows(res))

    post = Post(
        title="Two Thirds of the Move Happens Before the Headline",
        slug="two-thirds-of-the-move-before-the-headline",
        subtitle=("A paper measured 'buy the rumour, sell the news' on 4.57 million "
                  "articles, and the sayings mostly hold"),
        summary=(
            f"'It's already priced in' and 'buy the rumour, sell the news' are two "
            f"of the oldest things anyone says about markets, and both put the "
            f"price move before the headline rather than after it. A paper posted "
            f"this month tested them on **{N_ARTICLES / 1e6:.2f} million** news "
            f"articles across about {N_STOCKS:,} US stocks. Pooled over 1.68 "
            f"million events the news-aligned move reaches "
            f"+{CAR_AT_PUBLICATION:.2f}% by the closing bell on publication day and "
            f"+{CAR_AT_20:.2f}% twenty days later — two thirds handed back. For "
            f"rumours, the rumour day carries the whole move and the confirmation "
            f"carries {RUMOUR['the confirmation day']:+.2f}%. Underneath that, "
            f"quantified news keeps drifting and story-driven news reverses. This "
            f"post walks through the study and then does the one thing it "
            f"deliberately does not: works out what effects of this size are worth, "
            f"which turns on a single number neither of us measured."),
        tags=["machine-learning", "quantitative-finance", "market-microstructure",
              "nlp", "data-science"],
        author=qp.SETTINGS.author,
        code_url=qp.SETTINGS.code_repo_url,
        min_words=1600, max_words=2400,
        table_figures=[figs["table"]],
        data_sources=[
            "Alireza Kargarzadeh, Nariman Khaledian, Navid Parvini, Sid Ghatak and "
            "Arman Khaledian, 'Buy the Rumor, Sell the News: When Is News Priced "
            "In?', arXiv:2608.14014, 14 August 2026 (cs.AI, cs.LG, q-fin.ST) — "
            "4.57 million articles on about 3,000 US stocks over 2023-2026; 17 "
            "event tags and 5 attributes assigned by a compact classifier distilled "
            "from an LLM teacher by active learning; 1.68 million stock-day events "
            "with 364,405 neutral-sentiment events as a placebo; pooled cumulative "
            "abnormal return +0.58% by the close of publication day against +0.20% "
            "at day +20, a ratio of 2.8; rumour day +0.36%, the wait -0.09%, "
            "confirmation day +0.01%, days +6 to +20 -0.06%; placebo-adjusted drift "
            "over days +6 to +20 of +0.35% for capital returns, +0.22% earnings, "
            "+0.13% guidance, +0.10% analyst actions, -0.34% macro, -0.18% "
            "launches, -0.18% leadership and -0.17% competition. "
            "<https://arxiv.org/abs/2608.14014>.",
            "Average pairwise correlation among S&P 500 constituents of 0.05, "
            "implied constituent volatility (VIXEQ) of 44.42, the S&P 500 "
            "Dispersion Index at 41.42 after an all-time high of 47.51 on 21 July, "
            "and the VIX closing at 15.99 — all as of 31 July 2026, S&P Dow Jones "
            "Indices dispersion, volatility and correlation dashboard, "
            "<https://www.spglobal.com/spdji/en/documents/performance-reports/"
            "dashboard-dispersion-volatility-correlation.pdf>.",
            "No price series, article text or per-event data is used or "
            "redistributed, and the paper's data is not public. Every input above "
            "is a published scalar; everything else in this post is arithmetic on "
            "those scalars.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in qp.environment().items()
                if k in ("python", "numpy", "scipy", "quantpost")),
            "noise": (f"annualised implied constituent volatility {VIXEQ} scaled to "
                      f"a {WINDOW_DAYS}-day window is "
                      f"{noise['window_total']:.2f}%; removing the systematic share "
                      f"at rho = {PAIRWISE_RHO} leaves "
                      f"{noise['window_idio']:.2f}%, i.e. beta-adjustment removes "
                      f"{noise['beta_adjustment_saves_pct']:.1f}% of the volatility"),
            "implied_n": ("n = (z * sigma / effect)^2 from the reported p-value; "
                          "for p reported as an upper bound the result is a lower "
                          "bound on n and is shown with a > sign"),
            "breadth": ("effective independent positions n / (1 + (n-1) rho), which "
                        "converges to 1/rho; annualised Sharpe is the per-event "
                        "Sharpe times the square root of effective breadth times "
                        f"the square root of {res['periods']:.1f} non-overlapping "
                        "windows a year"),
            "ceiling": (f"at rho = {PAIRWISE_RHO} the ceiling is "
                        f"{res['ceilings'][PAIRWISE_RHO]:.0f} independent bets and "
                        f"is 90% reached by {res['bind'][PAIRWISE_RHO]} positions"),
            "not_reanalysis": ("the paper's data is not public; nothing here "
                               "recomputes its effects, and the effect sizes are "
                               "taken as reported"),
        },
    )

    post.add("Two sayings, and someone finally measured them", f"""
Markets have a pair of proverbs that say almost the same thing. **It's already
priced in**, meaning that by the time you read the headline the move has happened.
And **buy the rumour, sell the news**, meaning the money is made on the whisper and
given back on the confirmation. Both put the price move *before* publication, which
is a testable claim and an awkward one to test, because you need to know what every
article was about.

A paper posted to arXiv this month does it at a scale that makes the question
answerable: **{N_ARTICLES / 1e6:.2f} million financial news articles** covering
roughly **{N_STOCKS:,} US stocks** over {YEARS}. The sayings mostly hold. What is
more interesting is the structure underneath them, which is not folklore at all.
""".strip())

    post.add("How you label four and a half million articles", f"""
The measurement problem is the labelling problem. An event study needs to know what
happened, and "what happened" lives in prose.

The approach is the one that has quietly become standard for this kind of work, and
it is worth naming because it is the reusable part. A large language model acts as a
**teacher**, labelling a sample. A compact classifier is then **distilled** from
those labels through **active learning** — the model asks for labels where it is
least certain — and that small model labels the rest. You get an LLM's judgement at
a small model's cost per article, which at 4.57 million articles is the difference
between a study and a budget request.

Each article comes out with one of **{N_TAGS} event tags** and **{N_ATTRS}
attributes**. The tags are what you would expect and a few you might not: earnings
results, guidance, capital returns, analyst actions, M&A, legal and regulatory,
financing and dilution, insider and ownership, leadership, operations and supply,
product launch, competition, partnership, macro commentary through a single stock,
price commentary, other corporate, and promotional content. The attributes cut
across them — whether the news was **scheduled**, **forward-looking**, from a
**primary source**, **quantified**, or a **rumour**.

Two design choices matter more than the classifier.

First, articles are **clustered into stories**, so a first report is separated from
the follow-up coverage of the same event. Without that step the tenth article about
an acquisition looks like a tenth event, and the "anticipation" you measure is
partly just the press repeating itself. The paper's own estimate is that follow-up
coverage inflates measured anticipation by roughly **a third** — which is a large
enough correction to be the difference between a finding and an artefact.

Second, the comparison is against a **placebo**: {N_PLACEBO:,} neutral-sentiment
news events, used to measure what a stock does around a news day that carries no
direction at all. Every drift number below is net of that baseline. This is the step
that separates "stocks move around news" from "stocks move *in the direction of*
news", and it is the reason the results are worth reading.

That machinery produces **{N_EVENTS / 1e6:.2f} million stock-day events** with
beta-adjusted abnormal returns around each.
""".strip())

    post.add("Finding one: the move is mostly over before you read it", f"""
Take every signed event, orient each one so that positive means "in the direction
the news pointed", and add up abnormal returns from five days before publication.

By the closing bell on publication day the running total is
**+{CAR_AT_PUBLICATION:.2f}%**. Twenty trading days later it is
**+{CAR_AT_20:.2f}%**.

The ratio is {CAR_RATIO}, and the plainer way to say it is that
**{res['given_back_pct']:.0f}% of the news-aligned move is handed back** after the
news is out. The proverb is not quite that nothing happens after publication —
something does, and it points backwards.

The rumour version is sharper, and it is the cleanest result in the paper. Take
events flagged as rumours that were later followed by a non-rumour report within
sixty days, so the same story is caught twice. The rumour day delivers
**{RUMOUR['the rumour day']:+.2f}%**. The wait between rumour and confirmation:
{RUMOUR['the wait']:+.2f}%. The confirmation day itself, when the thing is actually
announced: **{RUMOUR['the confirmation day']:+.2f}%**. The following month:
{RUMOUR['days +6 to +20']:+.2f}%.

One basis point on the day the news is confirmed. Buy the rumour, sell the news.
""".strip(), figures=[figs["rumour"]])

    post.add("Finding two: numbers drift, stories reverse", f"""
The pooled number hides the good part. Split by event type and the sign of the
post-publication drift splits with it.

Over days +6 to +20, placebo-adjusted. The first three columns are the paper's; the
last two are mine and are explained two sections down — ignore them for now.

| {" | ".join(TABLE_HEADER)} |
|---|---|---|---|---|
{table_body}

The four positive rows all carry a **number** — a dividend, an earnings figure, a
guidance range, a price target. They average
{res['quantified_mean']:+.2f}% and they keep drifting the way the news pointed for
weeks after it is public. The four negative rows are **stories** — a launch, a
change of leadership, a competitive threat, a macro argument routed through one
stock. They average {res['story_mean']:+.2f}% and give the move back.

The first half of that is not new: post-earnings announcement drift has been in the
literature since the 1960s and has survived every attempt to explain it away. What
the taxonomy adds is the other half and the contrast. **The market underreacts to
things it can put in a spreadsheet and overreacts to things it has to interpret**,
and the same fifteen-day window measures both, in opposite directions, on the same
stocks.

If you want one mechanism for it, the attribute list already contains the candidate:
*quantified*. A number can be plugged into a model slowly, by many people, over
weeks. A story is priced by whoever finds it most exciting, immediately.
""".strip(), figures=[figs["tags"]])

    post.add("Finding three: news has a width", f"""
The third result is the one I would have missed, and it is about the second moment
rather than the first.

Publicity **raises volatility before** the publication day and volatility **falls
once the news is out**. Not because the news was calming, but because publication
resolves uncertainty: before it, the distribution of what might be announced is
wide; after it, there is only the announcement.

Anyone who has held an option through an earnings date has paid for this. It also
means a news-conditioned model has two things to predict, and the second one is the
better behaved: the direction of the move is nearly gone by the time you can read
about it, while the *width* is predictable ahead of a scheduled event and shrinks on
a known date.
""".strip())

    post.add("What effects of this size are worth", f"""
Here is the part the paper deliberately does not do, and where I can add something:
put these numbers in units you can judge.

Everything turns on one figure neither the paper nor I measured. On 31 July 2026
S&P's dispersion dashboard put **average pairwise correlation** among S&P 500
constituents at **{PAIRWISE_RHO}**, with implied constituent volatility at
{VIXEQ} and the dispersion index at {DISPERSION} — just off an all-time high of
{DISPERSION_HIGH}. That single number lands twice, in opposite directions, and the
two landings are the whole of this section.

**First landing: it explains the size of the study.** At a pairwise correlation of
{PAIRWISE_RHO}, the systematic share of a single stock's variance is negligible, so
beta-adjusting an abnormal return removes only
**{noise['beta_adjustment_saves_pct']:.1f}%** of its volatility. Scale {VIXEQ}%
annual to fifteen trading days and a single event's abnormal return sits in about
**{res['sigma']:.1f}%** of noise. Against that, a {TAGS['earnings results'][0]:+.2f}%
earnings drift is a per-event Sharpe of **{res['s_head']:.3f}**.

Which lets you check the paper from outside. A reported p-value implies a minimum
sample: `n = (z sigma / effect)^2`. Earnings at p < 0.001 needs at least
**{pt['earnings results']['implied_n']:,.0f}** events. Analyst actions, at
{TAGS['analyst action'][0]:+.2f}% and p = {TAGS['analyst action'][1]}, need over
**{pt['analyst action']['implied_n']:,.0f}**. Every one of those is comfortably
inside what {N_EVENTS / 1e6:.2f} million events across {N_TAGS} tags supplies — about
{res['events_per_tag']:,.0f} per tag on average, though tag sizes are certainly very
unequal and the paper does not publish the split, so this is a sanity check rather
than a verification. The p-values are consistent with the design, and the arithmetic
says independently why the study needed millions of articles rather than thousands.
At these effect sizes there was no cheaper way to find them.

**Second landing: the same number caps what they are worth.** Hold `n` positions
whose residual returns have pairwise correlation `rho` and you have
`n / (1 + (n-1) rho)` independent ones — the identity that turns an index of five
hundred names into a handful of bets. It converges to **1/rho**. So at
{PAIRWISE_RHO} the ceiling is **{res['ceilings'][PAIRWISE_RHO]:.0f} independent
bets**, reached by about {res['bind'][PAIRWISE_RHO]} names, and the earnings drift
pins at an annualised Sharpe of **{res['at_500'][PAIRWISE_RHO]:.2f}** however many
more you add. With genuinely uncorrelated residuals the same edge would be
{res['at_500'][0.0]:.1f} at five hundred names and still climbing.

I should be careful about which correlation that is. The {PAIRWISE_RHO} figure is
the correlation of *raw* returns; residual correlation after beta-adjustment is
lower, and I do not know it. That is why Fig 3 sweeps it. The exact part is the
shape: whatever the residual correlation turns out to be, breadth buys you `1/rho`
and stops, and it stops early.

**And the cost line needs no citation at all**, because it is an identity: the
round-trip cost that consumes a gross edge *is* the gross edge. Every tag in Table 1
dies below **{res['max_breakeven_bp']:.0f} basis points** of round-trip cost, and
the median one below **{res['median_breakeven_bp']:.0f}**. Take a ten-basis-point
round trip out of the earnings drift and the {res['at_500'][PAIRWISE_RHO]:.2f} Sharpe
becomes {res['net'][0.10]['sharpe']:.2f}; at twenty it becomes
{res['net'][0.20]['sharpe']:.2f}.
""".strip(), figures=[figs["breadth"]])

    post.add("So what is it for", f"""
Reading the last section back, it sounds like a demolition, and it is the opposite.

The paper does not claim a trading strategy. It says, in its own words, that it is a
descriptive account of where the news-aligned move sits in event time and not a
causal claim about news moving prices. The arithmetic above is what happens when you
try to make the stronger claim anyway, and it fails for reasons that have nothing to
do with the measurement being wrong. The effects are real, carefully separated from
a placebo, and small.

What they are actually for is the last line of the abstract, which is easy to skim
past: the paper ships **a table of measured drift for each event tag, usable as a
prior in news-conditioned forecasting models**. That is the deliverable. A
{TAGS['capital returns'][0]:+.2f}% prior on the fifteen-day drift after a
capital-returns story is not a strategy, but it is a considerably better starting
point than zero, and it is the kind of thing that is worth having precisely because
it is too small to trade on its own and therefore unlikely to be arbitraged away by
someone reading the same paper.

The honest summary is that this is a measurement paper, the measurement is good, and
the numbers are small enough that knowing their size is the point.
""".strip())

    post.add("Where to be careful", f"""
**The paper's own caution, which deserves repeating.** Pre-publication drift mixes
genuine anticipation with reporting on moves that had already happened — a story
written *because* the stock moved will look like the stock moving before the story.
There is no way to fully separate those two from article timestamps, and the paper
says so rather than claiming the anticipation is all information leakage.

**My noise number is an implied volatility, not a realised one.** {VIXEQ}% is what
options were pricing on one day at the end of a month when dispersion had just set a
record. Realised single-stock volatility over {YEARS} was lower on average, which
would *shrink* my sigma and *raise* every Sharpe and lower every implied sample
size. The direction of that error is against my conclusion, so the conclusion is not
resting on the choice — but the specific numbers would move.

**One date for the correlation.** {PAIRWISE_RHO} is a single reading, and an unusual
one: correlation that low is a dispersion regime, not a normal one. In a crisis it
runs five to ten times higher, which lowers the {res['ceilings'][PAIRWISE_RHO]:.0f}
bet ceiling further. Again the direction is against the trading reading, not for it.

**Fifteen days is not a holding period.** I treated days +6 to +20 as a
non-overlapping window and got {res['periods']:.1f} of them a year. A real
implementation would overlap positions, which changes the arithmetic in a way that
depends entirely on the overlap structure and cannot be done from published scalars.

**And I have not re-analysed anything.** The paper's data is not public. Every
effect size here is taken as reported; what I have added is arithmetic on top of
those numbers plus one outside volatility. If the effects are wrong, everything in
my section is wrong in exactly the same direction.
""".strip())
    return post
