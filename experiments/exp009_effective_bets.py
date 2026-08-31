"""exp009 — how many stocks is an index fund actually a bet on?

Backlog: Track B, the second markets post, and deliberately not about any one
country: index concentration is a global argument and the S&P 500 is where the
numbers are published.

The setup everyone argues about: the ten largest S&P 500 companies are around 41%
of the index against roughly 19% in 1990, and the phrase "you are not diversified
any more" follows. That claim has a formula behind it, and computing the formula
gives an answer that is unkind to both sides.

Everything turns on one identity. For weights `w` and an equicorrelation matrix
`rho*11' + (1-rho)I`, portfolio variance relative to a single stock is

    rho + (1 - rho) * HHI            where HHI = sum(w_i^2)

so the number of *independent* stocks that would deliver the same variance
reduction is

    N_bets = 1 / (rho + (1 - rho) * HHI)

Two limits fall out of that expression, and the post is the distance between them:

* At `rho = 0` it is `1 / HHI` — the Herfindahl effective number of holdings, the
  number the concentration debate is actually about.
* At any realistic correlation the `rho` term dominates. The S&P's concentration
  takes the weight-based figure from 503 names down to at most 57; it takes the
  *bets* figure from 2.85 to 2.77 at a typical correlation of 0.35 — 0.08 of a bet,
  under 3%.

  And the size of that effect depends on the regime in a direction worth stating:
  concentration costs 0.83 bets (10%) at the unusually low correlation of early
  2026, 0.08 (2.8%) at a typical 0.35, and 0.01 (0.6%) in a crisis at 0.70. It
  matters most when diversification is working anyway and vanishes when you need
  it, which is the opposite of how a risk factor is supposed to behave.

Two things this is careful about:

1. **The weight-based number is computed as a bound, not an estimate.** Published
   sources give the top-10 weights; the other 493 are not public in convenient
   form. HHI is *minimised* when the remaining weight is spread perfectly evenly,
   so `1/HHI` computed that way is an upper bound on the effective number of
   holdings — the real figure is smaller. Saying "at most 57" is therefore exact,
   where "about 57" would be an estimate of something I did not measure.
2. **Correlation is swept, not assumed.** The Cboe 1-month implied correlation
   index sat near 10 in late January 2026 and just above 15 at February's close,
   which is unusually low; crisis regimes run three to five times that. The
   conclusion is stated across the whole range because it holds across the whole
   range.

Run: `standarderror run exp009_effective_bets --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.render import Post
from standarderror.viz import charts, theme

#: Pinned so a rebuild cannot silently re-date a published post.
#: `Post.date` defaults to today, which is correct exactly once.
POST_DATE = date(2026, 8, 14)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed

# --- published facts, sourced in `data_sources` --------------------------------
# S&P 500 top-10 index weights, March 2026.
TOP10 = {
    "NVIDIA": 7.08, "Apple": 6.19, "Microsoft": 4.97, "Amazon": 3.75,
    "Alphabet A": 3.06, "Alphabet C": 2.85, "Meta": 2.67, "Broadcom": 2.57,
    "Tesla": 2.44, "Berkshire Hathaway": 1.76,
}
N_CONSTITUENTS = 503
# Top-10 share of index weight at four dates, from RBC Wealth Management.
HISTORY = {"1990": 19.0, "end 2000": 23.0, "end 2015": 19.0, "end 2025": 41.0}
TOP10_WEIGHT_2025 = 41.0            # % of index weight, end 2025
TOP10_EARNINGS_2025 = 32.0          # % of expected index earnings, same names
# Other published totals for the same quantity, to show the answer is insensitive.
ALT_TOTALS = {"March 2026 (itemised)": 37.34, "end 2025": 41.0, "July 2026": 43.0}
COR_LOW, COR_HIGH = 0.10, 0.15      # Cboe COR1M, late Jan and end Feb 2026

RHOS = np.round(np.arange(0.02, 0.81, 0.02), 3)
ANCHORS = {"calm (Cboe, early 2026)": 0.12, "long-run typical": 0.35,
           "a crisis": 0.70}
# Short forms for the chart: the full keys read well in prose and collide with the
# next dashed line when drawn at 8.5pt.
ANCHOR_LABELS = {"calm (Cboe, early 2026)": "early 2026 (calm)",
                 "long-run typical": "long-run typical", "a crisis": "a crisis"}
N_SIM = 500_000                     # Monte Carlo draws for the verification
SIM_BATCH = 20_000                  # drawn in batches: 500k x 503 will not fit


def herfindahl(weights: np.ndarray) -> float:
    """Sum of squared weights. Weights must sum to one."""
    w = np.asarray(weights, float)
    if not np.isclose(w.sum(), 1.0, atol=1e-9):
        raise ValueError(f"weights sum to {w.sum()}, not 1")
    return float(np.sum(w ** 2))


def bound_weights(top10_total_pct: float, shape: np.ndarray | None = None,
                  n: int = N_CONSTITUENTS) -> np.ndarray:
    """The weight vector that *minimises* HHI given a known top-10 total.

    The published figure is the top ten's share; the tail is not public in a form
    anyone can cite. Spreading the remaining weight perfectly evenly is the most
    diversified the tail could possibly be, so the HHI it produces is a floor and
    the effective number of holdings it produces is a ceiling. That distinction is
    the difference between a claim I can defend and a guess.
    """
    if not 0 < top10_total_pct < 100:
        raise ValueError("top-10 total must be a percentage in (0, 100)")
    shape = (np.array(list(TOP10.values()), float) if shape is None
             else np.asarray(shape, float))
    top = shape / shape.sum() * (top10_total_pct / 100.0)
    rest = np.full(n - len(top), (1.0 - top.sum()) / (n - len(top)))
    return np.concatenate([top, rest])


def effective_bets(hhi: float, rho: float) -> float:
    """Independent stocks equivalent to this portfolio, at correlation `rho`.

    Portfolio variance under equicorrelation is `rho + (1 - rho) * HHI` times a
    single stock's, so its reciprocal is the number of *uncorrelated* holdings
    that would reduce variance by the same amount. At `rho = 0` this is `1/HHI`,
    the familiar effective number of holdings; the point of the post is how little
    of that survives any realistic `rho`.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    if not 0.0 < hhi <= 1.0:
        raise ValueError("HHI must lie in (0, 1]")
    return 1.0 / (rho + (1.0 - rho) * hhi)


def simulate_variance_ratio(weights: np.ndarray, rho: float, n_draws: int,
                            seed: int) -> float:
    """Monte Carlo check of the closed form, via an explicit one-factor model.

    Each stock is `sqrt(rho) * f + sqrt(1 - rho) * e_i` with unit variances, which
    reproduces the equicorrelation matrix without ever building a 503x503 object.
    If this disagrees with the algebra, the algebra is what gets fixed.
    """
    rng = np.random.default_rng(seed)
    n = len(weights)
    # Accumulated in batches. The full draw is 500,000 x 503 doubles — two
    # gigabytes — and the first version of this quietly used 40,000 instead, which
    # left a 1.4% sampling error and made a "verification" that could not
    # distinguish the closed form from a formula 1% wrong.
    s_port = s_port2 = s_one = s_one2 = 0.0
    done = 0
    while done < n_draws:
        k = min(SIM_BATCH, n_draws - done)
        f = rng.standard_normal(k)
        e = rng.standard_normal((k, n))
        r = np.sqrt(rho) * f[:, None] + np.sqrt(1.0 - rho) * e
        port = r @ weights
        s_port += port.sum(); s_port2 += (port ** 2).sum()
        s_one += r[:, 0].sum(); s_one2 += (r[:, 0] ** 2).sum()
        done += k
    var_port = s_port2 / n_draws - (s_port / n_draws) ** 2
    var_one = s_one2 / n_draws - (s_one / n_draws) ** 2
    return float(var_port / var_one)


def analyse() -> dict:
    w_cap = bound_weights(sum(TOP10.values()))
    hhi_cap = herfindahl(w_cap)
    hhi_equal = 1.0 / N_CONSTITUENTS

    curves = {
        "the index, at its published weights": [
            effective_bets(hhi_cap, r) for r in RHOS],
        "503 stocks, perfectly equally weighted": [
            effective_bets(hhi_equal, r) for r in RHOS],
    }
    # Verification at a handful of correlations, against an explicit factor model.
    checks = {}
    for i, rho in enumerate((0.05, 0.12, 0.35, 0.70)):
        sim = simulate_variance_ratio(w_cap, rho, N_SIM, seed=100 + i)
        exact = 1.0 / effective_bets(hhi_cap, rho)
        checks[rho] = {"simulated": 1.0 / sim, "closed_form": 1.0 / exact,
                       "rel_err_pct": 100.0 * abs(sim - exact) / exact}

    history = {}
    for label, tot in HISTORY.items():
        h = herfindahl(bound_weights(tot))
        history[label] = {"top10_pct": tot, "n_eff_max": 1.0 / h,
                          "bets_at_typical": effective_bets(h, 0.35)}
    alt = {label: 1.0 / herfindahl(bound_weights(tot))
           for label, tot in ALT_TOTALS.items()}

    anchors = {name: {"rho": r,
                      "cap": effective_bets(hhi_cap, r),
                      "equal": effective_bets(hhi_equal, r)}
               for name, r in ANCHORS.items()}
    return {
        "w_cap": w_cap, "hhi_cap": hhi_cap, "hhi_equal": hhi_equal,
        "n_eff_max": 1.0 / hhi_cap, "n_eff_equal": 1.0 / hhi_equal,
        "curves": curves, "checks": checks, "history": history, "alt": alt,
        "anchors": anchors,
        "max_check_err": max(c["rel_err_pct"] for c in checks.values()),
        "top10_total": sum(TOP10.values()),
        # The whole argument in one number: what concentration costs in bets.
        "cost_of_concentration": {
            name: a["equal"] - a["cap"] for name, a in anchors.items()},
    }


def figures(res: dict) -> dict:
    src = ("Closed form from published index weights; the correlation axis is "
           "swept, not estimated.")
    figs = {}
    a = res["anchors"]

    # F1 — the whole post. Two portfolios that differ enormously by weight and
    # barely at all by risk.
    frame = pd.DataFrame(res["curves"], index=RHOS)

    def mark_anchors(_fig, ax):
        m = theme.LIGHT
        for i, (name, v) in enumerate(ANCHORS.items()):
            ax.axvline(v, color=m.muted, lw=1.2, ls=(0, (5, 3)))
            ax.annotate(ANCHOR_LABELS[name], (v, 0.97 - 0.09 * i),
                        xycoords=("data", "axes fraction"),
                        xytext=(6, 0), textcoords="offset points", ha="left",
                        va="top", fontsize=8.5, color=m.muted)
        # Readable ticks on the log axis. The default gives a single decade label,
        # and the whole point of this chart is reading values off it.
        from matplotlib.ticker import FixedLocator, ScalarFormatter
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10, 20, 40]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(FixedLocator([]))

    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False, decorate=mark_anchors, logy=True,
        title="Two very different portfolios, one nearly identical answer",
        subtitle=("Number of independent stocks that would give the same variance "
                  "reduction, against average pairwise correlation. The dashed "
                  "lines mark correlation regimes, not forecasts."),
        ylabel="equivalent independent stocks (log scale)",
        xlabel="average pairwise correlation", source=src,
        alt=(f"Two curves falling steeply as correlation rises, from about "
             f"{res['curves']['503 stocks, perfectly equally weighted'][0]:.0f} at "
             f"very low correlation to under two at 0.8. The cap-weighted and "
             f"equal-weighted curves are almost indistinguishable across the whole "
             "range."),
        caption=(f"Fig 2. The two lines are the S&P at its actual weights and a "
                 f"perfectly equally weighted 503-stock portfolio — the most "
                 f"diversified thing you could build from the same names. At a "
                 f"typical correlation of {ANCHORS['long-run typical']} they are "
                 f"{a['long-run typical']['cap']:.2f} and "
                 f"{a['long-run typical']['equal']:.2f} independent stocks. The "
                 "entire concentration debate is the gap between those two "
                 "numbers."),
        path=str(IMG / f"b4-f1-bets-vs-rho.{EXT}"))
    figs["bets"] = fig_meta

    # F2 — the concentration story itself, which is real and worth showing.
    labels = list(HISTORY.keys())
    fig_meta, _ = charts.ranked_bars(
        labels, [HISTORY[k] for k in labels], mode="light", value_fmt=".0f",
        title="The concentration everyone is arguing about is real",
        subtitle=("Share of S&P 500 index weight held by its ten largest "
                  "companies, at four dates."),
        xlabel="top-10 share of index weight (%)",
        source="RBC Wealth Management, January 2026.",
        alt=("Four horizontal bars of the top-10 share of S&P 500 weight: "
             + ", ".join(f"{k} {v:.0f}%" for k, v in HISTORY.items()) + "."),
        caption=(f"Fig 1. This is not in dispute and it is not small: the ten "
                 f"largest names went from about a fifth of the index to about "
                 f"two fifths, which takes the effective number of *holdings* "
                 f"from at most {res['history']['1990']['n_eff_max']:.0f} to at "
                 f"most {res['history']['end 2025']['n_eff_max']:.0f}. What the "
                 f"same change does to the number of independent bets, at a "
                 f"typical correlation, is "
                 f"{res['history']['1990']['bets_at_typical']:.2f} to "
                 f"{res['history']['end 2025']['bets_at_typical']:.2f}."),
        path=str(IMG / f"b4-f2-history.{EXT}"))
    figs["history"] = fig_meta

    # T1 — the answer under every definition, as an image for Medium.
    rows = [
        ["companies you own", f"{N_CONSTITUENTS}", "the fund's holdings page"],
        ["effective holdings, by weight", f"at most {res['n_eff_max']:.0f}",
         "1 / HHI, tail assumed perfectly even"],
    ]
    for name, v in res["anchors"].items():
        rows.append([f"independent bets, {name}", f"{v['cap']:.2f}",
                     f"1 / (rho + (1-rho)·HHI), rho = {v['rho']}"])
    fig_meta, _ = charts.table_image(
        rows, header=["what you are counting", "how many", "how it is computed"],
        title="How many stocks do you own? Five answers, all correct",
        subtitle="S&P 500 at published March 2026 weights.",
        source=src, mode="light", bold_cols=(1,), align="lrl",
        alt=("Table: 503 companies owned; at most 57 effective holdings by weight; "
             "and between about 1.4 and 8 independent bets depending on the "
             "correlation regime."),
        caption=("Table 1. Every row answers a different question and only the "
                 "last three are about risk. The first row is the one that "
                 "appears in marketing material."),
        path=str(IMG / f"b4-t1-answers.{EXT}"))
    figs["table"] = fig_meta

    # HERO — a comparison card: the finding is three numbers for one portfolio.
    fig_meta, _ = charts.comparison_card(
        headline="How many stocks is an S&P 500 fund a bet on?",
        items=[(f"{N_CONSTITUENTS}", "companies you own"),
               (f"{res['n_eff_max']:.0f}", "effective holdings"),
               (f"{a['long-run typical']['cap']:.1f}", "independent bets")],
        emphasis=2,
        note=(f"All three are correct. The middle one is by weight, the right-hand "
              f"one is by risk at a typical correlation — and thirty-five years of "
              f"rising concentration moved that one from "
              f"{res['history']['1990']['bets_at_typical']:.2f} to "
              f"{res['history']['end 2025']['bets_at_typical']:.2f}."),
        footer="The Standard Error", mode="light",
        alt=(f"Card comparing {N_CONSTITUENTS} companies owned, at most "
             f"{res['n_eff_max']:.0f} effective holdings by weight, and "
             f"{a['long-run typical']['cap']:.1f} independent bets at typical "
             "correlation."),
        caption="",
        path=str(IMG / f"b4-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    res = analyse()
    figs = figures(res)
    a = res["anchors"]
    typical = a["long-run typical"]
    calm, crisis = a["calm (Cboe, early 2026)"], a["a crisis"]

    post = Post(
        title="Your Index Fund Owns 503 Companies and Is About Three Bets",
        slug="your-index-fund-owns-503-companies-and-is-three-bets",
        date=POST_DATE,
        subtitle=("Concentration has a formula, and running it is unkind to both "
                  "sides of the argument"),
        summary=(f"The ten largest S&P 500 companies are around 41% of the index "
                 f"against roughly 19% in 1990, and the usual conclusion is that "
                 f"index investors are no longer diversified. There is a formula "
                 f"for that. It says the effective number of *holdings* has indeed "
                 f"collapsed — {N_CONSTITUENTS} companies, at most "
                 f"{res['n_eff_max']:.0f} of them in effect. It also says the "
                 f"number of independent *bets* is "
                 f"{typical['cap']:.2f} at a typical correlation, that a perfectly "
                 f"equally weighted portfolio of the same 503 names would be "
                 f"{typical['equal']:.2f} — so the whole concentration argument is "
                 f"worth {typical['equal'] - typical['cap']:.2f} of one bet at "
                 "that correlation, and less than that in a crisis."),
        tags=["investing", "quantitative-finance", "risk-management", "statistics",
              "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=1400, max_words=2200,
        table_figures=[figs["table"]],
        data_sources=[
            "S&P 500 top-10 index weights, March 2026 (NVIDIA 7.08%, Apple 6.19%, "
            "Microsoft 4.97%, Amazon 3.75%, Alphabet A 3.06%, Alphabet C 2.85%, "
            "Meta 2.67%, Broadcom 2.57%, Tesla 2.44%, Berkshire Hathaway 1.76%; "
            "503 constituents) — Westmount Fundamentals, "
            "<https://westmountfundamentals.com/sp500-top-10-holdings-weight-2026>.",
            "Top-10 share of index weight of roughly 19% in 1990, 23% at end-2000, "
            "19% at end-2015 and nearly 41% at end-2025, and the observation that "
            "the top 10 were about 41% of weight against about 32% of expected "
            "earnings — RBC Wealth Management, 22 January 2026, "
            "<https://www.rbcwealthmanagement.com/en-us/insights/"
            "the-great-narrowing-sp-500-concentration>.",
            "Cboe 1-Month Implied Correlation Index (COR1M) near 10 in late "
            "January 2026 and just above 15 at February's close — Cboe Index "
            "Insights, February 2026, "
            "<https://www.cboe.com/insights/posts/index-insights-february-2026>.",
            "No price series is used or redistributed. Every figure in this post "
            "is either quoted above or computed from those weights in closed "
            "form, with a Monte Carlo check.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "standarderror")),
            "identity": "portfolio variance under equicorrelation is "
                        "rho + (1-rho)·HHI relative to one stock, so the "
                        "equivalent number of independent stocks is its reciprocal",
            "bound": f"the tail beyond the top ten is assumed perfectly evenly "
                     f"spread, which minimises HHI; every effective-holdings "
                     f"figure here is therefore an upper bound "
                     f"({res['n_eff_max']:.1f} at the itemised March 2026 weights)",
            "insensitivity": ", ".join(
                f"{k}: at most {v:.0f}" for k, v in res["alt"].items()),
            "verification": f"one-factor Monte Carlo, {N_SIM:,} draws at rho in "
                            f"(0.05, 0.12, 0.35, 0.70); largest disagreement with "
                            f"the closed form {res['max_check_err']:.2f}%",
        },
    )

    post.add("A question with five correct answers", f"""
Open the holdings page of an S&P 500 fund and you will be told you own
**{N_CONSTITUENTS} companies**. That is true, and it is the least informative true
thing available.

The ten largest of those companies are about **{HISTORY['end 2025']:.0f}%** of the
index by weight. In 1990 the figure was roughly {HISTORY['1990']:.0f}%; at the end
of 2000, near the top of the dot-com bubble, about {HISTORY['end 2000']:.0f}%; at
the end of 2015, back to {HISTORY['end 2015']:.0f}%. The concentration is real, it
is unprecedented in the modern era, and the conclusion drawn from it — *you are not
diversified any more* — is repeated constantly.

That conclusion is testable. Diversification is not a mood; it is a statement about
variance, and there is a formula. Running it gives an answer that will annoy
everybody, including me, because I expected it to support the concentration
argument and it mostly does not.
""".strip(), figures=[figs["history"]])

    post.add("Answer one: how many holdings, in effect", f"""
The standard tool is the Herfindahl index — add up the squares of the weights —
and its reciprocal, which is the **effective number of holdings**. An equally
weighted portfolio of 100 stocks scores 100. A portfolio with half its money in one
name scores below 4 however many names are in the tail.

I only have the top ten weights, because those are what gets published. That turns
out not to matter, for a reason worth stating: HHI is *minimised* when the
remaining weight is spread perfectly evenly. So if I assume the other 493 companies
are all exactly the same size — the most diversified the tail could conceivably be
— I get a floor on HHI and therefore a **ceiling** on the effective number of
holdings. Not an estimate. A bound.

The bound is **{res['n_eff_max']:.0f}**.

{N_CONSTITUENTS} companies; at most {res['n_eff_max']:.0f} of them in effect, and
really fewer. The concentration critics are right about this part, and it is not
close. Nor does the number depend on whose concentration figure you use: the same
calculation gives at most {res['alt']['end 2025']:.0f} at the 41% figure and at
most {res['alt']['July 2026']:.0f} at the 43% one that circulated in July. Run it
on the 1990 weighting and the ceiling was
{res['history']['1990']['n_eff_max']:.0f}.

So on the weight measure the effective portfolio has shrunk by a factor of
{res['history']['1990']['n_eff_max'] / res['history']['end 2025']['n_eff_max']:.1f}
in thirty-five years — from at most
{res['history']['1990']['n_eff_max']:.0f} names in effect to at most
{res['history']['end 2025']['n_eff_max']:.0f}. That is a large change and the
critics own it.

Now the part that changes the conclusion.
""".strip())

    post.add("Answer two: how many bets", f"""
Effective holdings answers "how spread out is my money". It does not answer "how
many different things am I exposed to", and those are the same question only if the
holdings are independent. Stocks are not independent. They share an economy.

Put a correlation into the arithmetic and it collapses to one line. If every pair
of stocks has correlation *rho*, a portfolio's variance relative to a single
stock's is **rho + (1 - rho) × HHI**, where HHI is the sum of squared weights —
and the number of genuinely independent stocks that would achieve the same variance
reduction is the reciprocal of that. It is worth staring at for a second, because
the two terms are the whole argument. The second term is concentration — everything
the debate is about. The first term is correlation, and it does not care about
weights at all.

At a long-run typical average pairwise correlation of
{typical['rho']}, the S&P 500 at its published weights is
**{typical['cap']:.2f} independent stocks**.

A perfectly equally weighted portfolio of the same {N_CONSTITUENTS} companies — the
most diversified object that can be built from these names, the thing the
concentration critics are implicitly asking for — is
**{typical['equal']:.2f}**.

The difference is **{typical['equal'] - typical['cap']:.2f} of one bet** — under
{100 * (typical['equal'] - typical['cap']) / typical['equal']:.0f}% of the total.

Everything written about index concentration in the past two years, priced in the
units that matter for portfolio variance, comes to about a tenth of a stock at that
correlation. The reason is visible in the formula: HHI here is
{res['hhi_cap']:.4f}, and once *rho* is anything but tiny the first term dwarfs it
whatever you do to the weights.

One honest qualification, because the size of the effect is not constant and it
moves in an awkward direction. Concentration costs
{res['cost_of_concentration']['calm (Cboe, early 2026)']:.2f} of a bet at the
unusually low correlation of early 2026 —
{100 * res['cost_of_concentration']['calm (Cboe, early 2026)'] / calm['equal']:.0f}%
of the total — {typical['equal'] - typical['cap']:.2f} at a typical
{typical['rho']}, and {res['cost_of_concentration']['a crisis']:.2f} in a crisis.
So concentration matters most when correlation is low, which is when
diversification is working anyway, and matters least when correlation is high,
which is when you need it. Whatever else it is, it is not a risk that shows up when
risk shows up.
""".strip(), figures=[figs["bets"]])

    # The same rows as the rendered table image, from the same numbers: Hugo shows
    # this markdown and Medium and Notion get the image substituted in.
    table_rows = [
        ("companies you own", f"{N_CONSTITUENTS}", "the fund's holdings page"),
        ("effective holdings, by weight", f"at most {res['n_eff_max']:.0f}",
         "1 / HHI, tail assumed perfectly even"),
    ] + [(f"independent bets, {name}", f"{v['cap']:.2f}",
          f"1 / (rho + (1-rho)·HHI), rho = {v['rho']}")
         for name, v in res["anchors"].items()]
    table_body = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in table_rows)
    post.add("Which correlation, though", f"""
The number that actually moves the answer is the one nobody argues about, and it
moves it enormously.

Correlation is not a constant. Cboe publishes an implied correlation index derived
from S&P options, and in early 2026 it was **unusually low** — near 10 in late
January and just above 15 at February's close, on a scale where those figures mean
correlations of about {COR_LOW} and {COR_HIGH}. In a crisis the same measure runs
three to five times higher.

Take those regimes in turn, for the actual index:

- **calm, at {calm['rho']}** — {calm['cap']:.1f} independent stocks
- **long-run typical, at {typical['rho']}** — {typical['cap']:.1f}
- **a crisis, at {crisis['rho']}** — {crisis['cap']:.1f}

Or all five answers together, which is the table I would put next to any fund's
holdings count:

| what you are counting | how many | how it is computed |
|---|---|---|
{table_body}

Between the calm regime and the crisis regime your diversification falls by a
factor of {calm['cap'] / crisis['cap']:.0f}, and it does so without a single weight
changing. That is the thing worth being alarmed about, and I have never seen it in
a fund fact sheet.

It also has an unpleasant timing property, which is the honest reason to care.
Correlation rises when markets fall. Diversification is therefore at its weakest at
exactly the moment it is supposed to be helping, and no rebalancing schedule fixes
that, because the problem is not your weights.

(The closed form is easy to get wrong, so I checked it against an explicit
one-factor Monte Carlo with {N_SIM:,} draws at four different correlations. Largest
disagreement: {res['max_check_err']:.2f}%.)
""".strip())

    post.add("Where the concentration worriers are still right", f"""
Having spent four sections deflating the argument, here is the strongest version of
it, because the diversification framing is not the only framing.

**Composition, not count.** Concentration has not changed how many bets you hold;
it has changed *what the single dominant bet is*. When the top ten are two fifths
of the index and most of them sell the same thing to each other, the common factor
you are exposed to stops being "the economy" and becomes something narrower. My
formula is completely blind to that. It counts bets; it has nothing to say about
what they are on, and a portfolio of {typical['cap']:.1f} independent bets on
different things is not the same object as {typical['cap']:.1f} bets on one supply
chain.

**Valuation, not variance.** The most concrete number in the RBC analysis is not a
weight: the top ten were around {TOP10_WEIGHT_2025:.0f}% of the index's weight and
about {TOP10_EARNINGS_2025:.0f}% of its expected earnings. That is a statement about what you are paying, not about how
spread out you are, and my arithmetic has no opinion on it whatsoever.

**And the counterfactual is not free.** "Diversify away from the concentration"
means holding something other than the market portfolio, which is an active bet
with its own concentration — equal weighting is a systematic tilt toward smaller
companies, rebalanced monthly, with turnover and tax to match. That may be a good
idea. It is not the neutral choice, and it should be argued on its own terms rather
than as a correction to an arithmetic problem it barely affects.

I am not going to tell you what to hold, and this post contains no view about
whether any of these companies are worth what they cost. What it contains is a
denominator.
""".strip())

    post.add("Where this is a caricature", """
**Equicorrelation is a cartoon.** One number for every pair is wrong: semiconductor
firms move together far more than a semiconductor firm and a utility. A realistic
block structure would put the effective number of bets somewhat *above* my figure
in calm regimes and somewhat below in crises, because real correlation matrices
have one dominant factor and several smaller ones. The direction of the
post's conclusion is unaffected, since the dominant factor is what is doing the
work either way.

**There is more than one definition of "effective bets".** I used the
variance-ratio one because it answers a question an investor actually has. Measures
built from the eigenvalues of the correlation matrix — Meucci's entropy-based
number of bets, or the participation ratio — give different figures for the same
portfolio, typically larger. They are answering a different question, and anyone
quoting one should say which.

**The weights are a snapshot.** March 2026, one index, one country. The tail is a
bound rather than a measurement.

**And an amusing measurement subtlety:** Alphabet appears twice in the top ten,
because its two share classes are separate index constituents. Whether you treat
that as one company or two changes the "top ten" total by roughly the weight of the
eleventh name — which is a decent reminder that even the number everyone is arguing
about is definition-dependent before you have done any mathematics at all.

Next in this series: what a neural network is doing when it looks like it has
learned physics, and how to tell that apart from having memorised a trajectory.
""".strip())

    return post


if __name__ == "__main__":
    p = build()
    print(p.title, "|", p.word_count(), "words |", len(p.figures), "figures")
    for issue in p.audit():
        print("  audit:", issue)
