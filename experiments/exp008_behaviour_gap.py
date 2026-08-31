"""exp008 — individuals sold a record amount on the biggest up day in history.

Backlog: Track B, the markets entry the reader asked for, and the sequel exp007 set
up. exp007 showed that the best days arrive next to the worst because volatility
clusters. This one set out to price that — and found that the pricing does not work
the way the setup implied.

The hook is a published pair of records from the same session: on 31 July 2026 —
the KOSPI's largest one-day gain ever, +17.91% — individuals net sold a record
8.2543 trillion won and foreigners net bought a record 7.2197 trillion won.

The measurement is the gap between a **time-weighted** return (what the index did,
which every investor in it shares) and a **money-weighted** return (the IRR of what
an investor's cash actually did). They differ only because of *when* money moved,
so the gap is arithmetic, not psychology.

**The hypothesis I started with was wrong, and the control is what caught it.**
I expected the cost of a drawdown rule to be made of volatility clustering: the best
days sit inside the crashes, so exit on a drawdown and you are absent for the
rebound. The control says no. Sweep GARCH persistence with the unconditional
variance pinned and the timing-specific penalty *shrinks* as clustering rises, from
-0.76pp at independence to -0.37pp at equity-index persistence. Clustering is not
the mechanism.

What the decomposition actually finds:

* Of the panic rule's -2.40pp a year, about -2.04pp is simply **being out of a
  rising market** — reproducible by an investor absent for the same number of days
  at random times. Five sixths of the cost is forgone drift.
* The remaining -0.36pp is **timing**, and it comes from the rule's own re-entry
  condition rather than from clustering: exiting at 15% below the peak and returning
  at 5% below it means the index must rise about 11.8% before you are allowed back,
  so the rule is absent for a rally *by construction*. Measured: the index gains
  13.7% during a panic absence against 11.2% during a matched random absence.
* "Missing the best days" is a small channel here. The panic rule holds 44.6% of the
  market's top-1% days against 49.2% for the matched random absentee — under five
  points of difference.

Three more things this is careful about:

1. **The matched counterfactual.** Comparing a drawdown rule with buy-and-hold
   conflates "out of the market" with "out at the wrong moment". The comparison that
   separates them is an investor absent for the same number of days, in episodes of
   the same lengths, at random times.
2. **Cash earns something.** Sitting out at a 0% cash rate overstates the cost of
   panicking. The headline uses 0% and the sensitivity to a realistic cash rate is
   reported next to it.
3. **What the flow data cannot say.** Aggregate net selling on a +17.91% day is not
   evidence about sentiment: forced margin liquidation, resting limit orders and
   index rebalancing all produce the same aggregate. The simulation is a statement
   about a *rule*, not an attribution of one to Korean retail investors, and the
   post says so.

Run: `standarderror run exp008_behaviour_gap --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.dynamics import sde
from standarderror.render import Post
from standarderror.viz import charts, theme

#: Pinned so a rebuild cannot silently re-date a published post.
#: `Post.date` defaults to today, which is correct exactly once.
POST_DATE = date(2026, 8, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed

# --- published facts, sourced in `data_sources` --------------------------------
RECORD_DAY = 17.91          # % — KOSPI, 31 July 2026
RETAIL_NET_SELL = 8.2543    # trillion won, individuals, 31 July, a record
FOREIGN_NET_BUY = 7.2197    # trillion won, foreigners, same session, a record
RETAIL_WEEK_SELL = 6.5437   # trillion won, individuals, week to 31 July

SESSIONS_PER_YEAR = 252
MONTH = 21                  # sessions per contribution period
N_YEARS = 20
N_SIM = N_YEARS * SESSIONS_PER_YEAR
N_HISTORIES = 500

# GARCH(1,1). Unconditional variance is held fixed at UNCOND_SD^2 while
# persistence is swept, so the only thing changing is the *arrival pattern*.
UNCOND_SD = 1.1             # % per day, in log terms
DRIFT_ANNUAL = 0.07         # the index's compounded annual drift
ARCH_SHARE = 0.10           # of the persistence budget that is the ARCH term
PERSISTENCES = (0.0, 0.50, 0.80, 0.90, 0.95, 0.98)
BASE_PERSISTENCE = 0.98
SHOCK_DF = 5.0

DRAWDOWN_EXIT = 0.15        # the panic rule's trigger
DRAWDOWN_REENTRY = 0.05     # and when it comes back
CASH_RATE = 0.0             # headline; sensitivity reported separately
CASH_RATE_ALT = 0.03


def market(persistence: float = BASE_PERSISTENCE, seed: int = 5) -> np.ndarray:
    """Daily *simple* percentage returns: a rising market with clustered volatility.

    Two things this does that a bare GARCH call does not.

    `omega` is solved from the target variance, so changing persistence changes when
    volatility arrives and not how much of it there is. Sweeping persistence without
    that would confound clustering with a louder market, and a louder market alone
    would widen every gap in this post.

    And the GARCH path is treated as *log* returns and converted, which is both the
    conventional reading and a correctness requirement: taken as simple returns, a
    persistent t-shocked path reaches -100% every few hundred simulated years, the
    compounded product goes negative, and annualising a negative total in Python
    returns a **complex number** that numpy then casts back to a float with a
    warning. One history in five hundred did exactly that in the first draft of
    this experiment, and the resulting mean was quietly wrong rather than loudly
    broken. `simple_from_log` also carries the drift: a raw GARCH market does not
    rise, and a behaviour-gap study on a market with no expected return is measuring
    something else.
    """
    arch = ARCH_SHARE * persistence
    beta = persistence - arch
    omega = UNCOND_SD ** 2 * (1.0 - persistence) or UNCOND_SD ** 2
    log_r = sde.garch11(N_SIM, omega=omega, arch=arch, beta=beta, df=SHOCK_DF,
                        seed=seed).data["r"]
    r = sde.simple_from_log(log_r, drift_annual=DRIFT_ANNUAL,
                            periods_per_year=SESSIONS_PER_YEAR)
    if np.any(r <= -100.0) or not np.all(np.isfinite(r)):
        raise AssertionError("impossible daily return survived the conversion")
    return r


# --- the investor rules -------------------------------------------------------
#
# Each rule holds the *same asset* and differs only in when its money moves, which
# is the point: the time-weighted return is identical for all of them.

def _month_returns(r: np.ndarray) -> np.ndarray:
    """Compounded return of each complete month, in decimals."""
    n_months = len(r) // MONTH
    blocks = (1.0 + r[:n_months * MONTH] / 100.0).reshape(n_months, MONTH)
    return blocks.prod(axis=1) - 1.0


def matched_absence(r: np.ndarray, episodes, seed: int) -> np.ndarray:
    """Absence intervals of the same lengths as a rule's, placed at random.

    The counterfactual that makes the decomposition possible. A drawdown rule pays
    two separate costs: it is out of a market that rises, and it is out at *exactly*
    the moments the rebound happens. Comparing it with holding conflates the two.
    Comparing it with an investor who is absent for the same number of days, in
    episodes of the same lengths, but at random times, isolates the second.

    Returns a boolean "in cash" mask.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(len(r), dtype=bool)
    for length in sorted(episodes, reverse=True):
        for _ in range(200):
            start = int(rng.integers(0, max(1, len(r) - length)))
            if not out[start:start + length].any():
                out[start:start + length] = True
                break
    return out


def run_rule(r: np.ndarray, rule: str, cash_rate: float = CASH_RATE,
             absence: np.ndarray | None = None) -> dict:
    """Simulate one rule's account, day by day, and return its cash-flow stream.

    Contributions are one unit per month. Everything is expressed in units so no
    currency is implied; the returns below are rates and do not depend on the
    scale.
    """
    n = len(r)
    if rule == "unlucky timing" and absence is None:
        raise ValueError("the matched-absence rule needs an `absence` mask")
    monthly = _month_returns(r)
    daily_cash = (1.0 + cash_rate) ** (1.0 / SESSIONS_PER_YEAR) - 1.0

    units = 0.0          # invested wealth
    cash = 0.0           # sidelined wealth
    episode_returns: list[float] = []
    level_at_exit = 1.0
    flows = np.zeros(n)
    peak = 1.0
    level = 1.0
    out = False
    days_out = 0
    held = np.zeros(n, dtype=bool)

    for t in range(n):
        # Contribution at the start of each month, sized by the rule.
        if t % MONTH == 0:
            m = t // MONTH
            last = monthly[m - 1] if m >= 1 else 0.0
            if rule in ("steady", "panics", "unlucky timing"):
                flow = 1.0
            elif rule == "chases":
                flow = 2.0 if last > 0.03 else (-1.0 if last < -0.03 else 1.0)
            elif rule == "buys dips":
                flow = 2.0 if last < -0.03 else (-1.0 if last > 0.03 else 1.0)
            else:
                raise ValueError(f"unknown rule {rule!r}")
            if flow >= 0:
                units += flow
                flows[t] += flow
            else:
                # A withdrawal cannot exceed the account.
                taken = min(-flow, units)
                units -= taken
                flows[t] -= taken

        # The panic rule moves the whole account to cash on a deep drawdown and
        # comes back when the index has recovered most of it. No cash flows here:
        # the money never leaves the investor, so the IRR must not see it.
        if rule == "panics":
            if not out and level < peak * (1.0 - DRAWDOWN_EXIT):
                cash += units
                units = 0.0
                out = True
                level_at_exit = level
            elif out and level > peak * (1.0 - DRAWDOWN_REENTRY):
                units += cash
                cash = 0.0
                out = False
                episode_returns.append(level / level_at_exit - 1.0)
        elif rule == "unlucky timing":
            want_out = bool(absence[t])
            if want_out and not out:
                cash += units
                units = 0.0
                out = True
                level_at_exit = level
            elif not want_out and out:
                units += cash
                cash = 0.0
                out = False
                episode_returns.append(level / level_at_exit - 1.0)

        held[t] = not out and units > 0
        step = r[t] / 100.0
        units *= 1.0 + step
        cash *= 1.0 + daily_cash
        days_out += int(out)
        level *= 1.0 + step
        peak = max(peak, level)

    final = units + cash
    return {"flows": flows, "final": final, "days_out": days_out, "held": held,
            "episodes": _episodes(held, r),
            # What the index did while this rule was in cash. The mechanism behind
            # any timing cost has to show up here or it is not the mechanism.
            "episode_returns": episode_returns,
            "money_weighted": sde.money_weighted_return(
                flows, final, periods_per_year=SESSIONS_PER_YEAR)}


def _episodes(held: np.ndarray, r: np.ndarray) -> list[int]:
    """Lengths of the stretches spent in cash, for the matched counterfactual."""
    import itertools
    return [len(list(g)) for k, g in itertools.groupby(~held) if k]


def time_weighted(r: np.ndarray) -> float:
    """The index's own annualised return — the number every fund reports."""
    total = float(np.prod(1.0 + r / 100.0))
    if total <= 0.0:
        raise AssertionError(
            f"compounded total {total} is not positive; annualising it would "
            "return a complex number (see market() for how that happened once)")
    return total ** (SESSIONS_PER_YEAR / len(r)) - 1.0


RULES = ("steady", "chases", "buys dips", "unlucky timing", "panics")
LABELS = {"steady": "contributes the same every month",
          "chases": "adds after good months, withdraws after bad ones",
          "buys dips": "adds after bad months, withdraws after good ones",
          "unlucky timing": "out of the market for the same days, at random times",
          "panics": "sells everything on a 15% drawdown, returns at 5%"}
SHORT = {"steady": "steady", "chases": "chases the rally",
         "buys dips": "buys the dips", "unlucky timing": "absent at random",
         "panics": "panic sells"}


def one_history(seed: int, persistence: float = BASE_PERSISTENCE,
                cash_rate: float = CASH_RATE) -> dict:
    r = market(persistence, seed=seed)
    twr = time_weighted(r)
    top = r >= np.quantile(r, 0.99)
    out = {"twr": twr, "n_top": int(top.sum())}
    # The panic rule runs first: its absence episodes define the matched control.
    panic = run_rule(r, "panics", cash_rate=cash_rate)
    absence = matched_absence(r, panic["episodes"], seed=seed + 7_000_000)
    for rule in RULES:
        res = (panic if rule == "panics"
               else run_rule(r, rule, cash_rate=cash_rate, absence=absence))
        out[rule] = {
            "mwr": res["money_weighted"],
            "gap_pp": 100.0 * (res["money_weighted"] - twr),
            "days_out": res["days_out"],
            "top_days_held": float(res["held"][top].mean()),
            "missed_move": (float(np.mean(res["episode_returns"]))
                            if res["episode_returns"] else 0.0),
            "n_episodes": len(res["episode_returns"]),
        }
    return out


def analyse() -> dict:
    base = one_history(5)
    histories = [one_history(1000 + i) for i in range(N_HISTORIES)]

    def across(rule: str, key: str) -> np.ndarray:
        return np.array([h[rule][key] for h in histories])

    summary = {}
    for rule in RULES:
        gaps = across(rule, "gap_pp")
        summary[rule] = {
            "gap_mean": float(gaps.mean()),
            "gap_sd": float(gaps.std()),
            "gap_median": float(np.median(gaps)),
            "share_negative": float((gaps < 0).mean()),
            "days_out": float(across(rule, "days_out").mean()),
            "top_days_held": float(across(rule, "top_days_held").mean()),
            "missed_move": float(across(rule, "missed_move").mean()),
            "n_episodes": float(across(rule, "n_episodes").mean()),
            "gaps": gaps,
        }

    # The control and the mechanism in one sweep: persistence changes the arrival
    # pattern with the unconditional variance pinned.
    sweep = {}
    for p in PERSISTENCES:
        hs = [one_history(2000 + i, persistence=p) for i in range(120)]
        sweep[p] = {rule: float(np.mean([h[rule]["gap_pp"] for h in hs]))
                    for rule in RULES}

    # Cash is not free. Re-run the panic rule with a realistic cash rate.
    alt = [one_history(1000 + i, cash_rate=CASH_RATE_ALT)
           for i in range(120)]
    alt_gap = float(np.mean([h["panics"]["gap_pp"] for h in alt]))

    iid_gaps = np.array([one_history(3000 + i, persistence=0.0)["panics"]["gap_pp"]
                         for i in range(N_HISTORIES)])
    return {
        "base": base,
        "summary": summary,
        "sweep": sweep,
        "alt_gap": alt_gap,
        "iid_gaps": iid_gaps,
        "n_histories": N_HISTORIES,
        "twr_mean": float(np.mean([h["twr"] for h in histories])),
    }


def figures(res: dict) -> dict:
    src = (f"Simulated: GARCH(1,1) daily returns, persistence "
           f"{BASE_PERSISTENCE}, Student-t({SHOCK_DF:.0f}) shocks, "
           f"{N_YEARS} years x {res['n_histories']} histories, fixed seed.")
    figs = {}
    sm = res["summary"]

    # F1 — the gap by rule. Signed bars: the sign is the finding.
    order = sorted(RULES, key=lambda k: sm[k]["gap_mean"])
    fig_meta, _ = charts.ranked_bars(
        [SHORT[k] for k in order], [sm[k]["gap_mean"] for k in order],
        errors=[sm[k]["gap_sd"] / np.sqrt(res["n_histories"]) for k in order],
        signed=True, mode="light", value_fmt="+.2f",
        title="Five investors, one asset, five different returns",
        subtitle=("Annualised money-weighted return minus the index's "
                  "time-weighted return, averaged over "
                  f"{res['n_histories']} simulated 20-year histories. Bars are "
                  "percentage points; whiskers are standard errors."),
        xlabel="gap against the index (percentage points a year)", source=src,
        alt=(f"Horizontal bars of the return gap for five rules. Panic selling is "
             f"the worst at {sm['panics']['gap_mean']:+.2f} points and being "
             f"absent at random is close behind at "
             f"{sm['unlucky timing']['gap_mean']:+.2f}; the three rules that only "
             "vary their contributions sit within a third of a point of zero."),
        caption=(f"Fig 1. Every rule holds the same index and sees the same "
                 f"{res['twr_mean'] * 100:.1f}% a year; only the timing of the "
                 f"money differs. Note the two groups: the contribution tilts are "
                 f"worth a fraction of a point, while the exit rule costs "
                 f"{abs(sm['panics']['gap_mean']):.2f} points. And note the "
                 f"matched control at "
                 f"{sm['unlucky timing']['gap_mean']:+.2f} — an investor absent "
                 "for the same days at random times, which is most of the way "
                 "there."),
        path=str(IMG / f"b3-f1-gap-by-rule.{EXT}"))
    figs["gap"] = fig_meta

    # F2 — the decomposition against persistence. This figure is the refutation:
    # if clustering were the mechanism the gap between the two lines would widen to
    # the right, and it narrows.
    frame = pd.DataFrame({
        "sells the drawdowns": pd.Series(
            [res["sweep"][p]["panics"] for p in PERSISTENCES],
            index=list(PERSISTENCES)),
        "absent the same days, at random": pd.Series(
            [res["sweep"][p]["unlucky timing"] for p in PERSISTENCES],
            index=list(PERSISTENCES)),
        "the difference: timing alone": pd.Series(
            [res["sweep"][p]["panics"] - res["sweep"][p]["unlucky timing"]
             for p in PERSISTENCES], index=list(PERSISTENCES)),
    })

    def mark_zero(_fig, ax):
        m = theme.LIGHT
        ax.axhline(0.0, color=m.axis, lw=1.2)
        # Below the zero line: above it is where the legend sits.
        ax.annotate("independent returns: no clustering at all", (0.0, 0.0),
                    xycoords=("data", "data"), xytext=(6, -7),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=8.5, color=m.muted)

    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False, decorate=mark_zero,
        title="The clustering explanation, and why I had to drop it",
        subtitle=("The same rules against markets with identical average "
                  "volatility and rising persistence. If clustering caused the "
                  "timing penalty, the difference line would drop away to the "
                  "right. It does the opposite."),
        ylabel="gap against the index (pp a year)",
        xlabel="GARCH persistence (arch + beta)", source=src,
        alt=("Three lines against GARCH persistence. The panic rule and the matched "
             "random absence both sit near minus two points and barely move; their "
             "difference starts near minus 0.8 at zero persistence and rises "
             "towards minus 0.4 at 0.98."),
        caption=(f"Fig 2. Both cost lines are nearly flat in persistence, so the "
                 f"bulk of the penalty is not about clustering at all — it is "
                 f"forgone drift. And the timing-only difference goes the *wrong* "
                 f"way for my hypothesis: "
                 f"{res['sweep'][0.0]['panics'] - res['sweep'][0.0]['unlucky timing']:+.2f} "
                 f"points with independent returns against "
                 f"{res['sweep'][BASE_PERSISTENCE]['panics'] - res['sweep'][BASE_PERSISTENCE]['unlucky timing']:+.2f} "
                 "at equity-index persistence. Clustering makes this rule's timing "
                 "slightly *less* costly."),
        path=str(IMG / f"b3-f2-persistence.{EXT}"))
    figs["mechanism"] = fig_meta

    # F3 — is it systematic or is it luck?
    gaps = sm["panics"]["gaps"]
    iid = res["iid_gaps"]
    lo = float(min(gaps.min(), iid.min())) - 0.2
    hi = float(max(gaps.max(), iid.max())) + 0.2
    counts, edges = np.histogram(iid, bins=34, range=(lo, hi), density=True)
    fig_meta, _ = charts.histogram(
        gaps, bins=34,
        series_label="clustered market (persistence 0.98)",
        overlay={"the same rule, independent returns":
                 (0.5 * (edges[:-1] + edges[1:]), counts)},
        mark={"break even": 0.0},
        title="It loses in four histories out of five, and worse without clustering",
        subtitle=(f"Distribution of the panic rule's gap across "
                  f"{res['n_histories']} simulated histories, against the same "
                  "rule in a market with the same volatility and no clustering."),
        xlabel="gap against the index (percentage points a year)", source=src,
        mode="light",
        alt=("Two distributions of the return gap, both sitting mostly left of "
             "zero. The no-clustering curve is shifted slightly further left than "
             "the clustered histogram rather than being centred on zero."),
        caption=(f"Fig 3. The distribution is shifted, not merely wide — the gap "
                 f"is negative in "
                 f"{sm['panics']['share_negative'] * 100:.0f}% of histories, so "
                 f"this is systematic and not bad luck. The overlaid curve is the "
                 f"same rule with clustering switched off, and it sits "
                 f"*further* left, at a mean of "
                 f"{res['iid_gaps'].mean():+.2f} against "
                 f"{sm['panics']['gap_mean']:+.2f}. That is the figure that killed "
                 "the explanation I came in with."),
        path=str(IMG / f"b3-f3-distribution.{EXT}"))
    figs["distribution"] = fig_meta

    # T1 — the summary table, as an image, because Medium strips table markup.
    rows = [[SHORT[k], f"{res['twr_mean'] * 100 + sm[k]['gap_mean']:.1f}%",
             f"{sm[k]['gap_mean']:+.2f}pp",
             f"{sm[k]['days_out'] / SESSIONS_PER_YEAR:.1f}",
             f"{sm[k]['top_days_held'] * 100:.0f}%",
             (f"{sm[k]['missed_move'] * 100:+.1f}%"
              if sm[k]['missed_move'] else "—")] for k in RULES]
    fig_meta, _ = charts.table_image(
        rows, header=["investor", "what they earned", "gap", "years in cash",
                      "best days held", "index move while out"],
        title=f"The index returned {res['twr_mean'] * 100:.1f}% a year for every "
              f"row of this table",
        subtitle=(f"Annualised, averaged over {res['n_histories']} simulated "
                  "20-year histories. Cash earns nothing here; see the text."),
        source=src, mode="light", bold_cols=(2,),
        alt=("Table of five rules with what each earned, the gap against the "
             "index, years spent in cash, the share of top-1% days held, and how "
             "much the index rose while each was out of it."),
        caption=("Table 1. The last column is the mechanism. Both absentees miss "
                 "about a decade, but the drawdown rule is out for a bigger index "
                 "move — because its re-entry rule will not let it back until the "
                 "index has risen roughly 12% from where it sold."),
        path=str(IMG / f"b3-t1-summary.{EXT}"))
    figs["table"] = fig_meta

    # HERO — a comparison card: this post's finding is one asset and two returns,
    # which is a comparison and nothing else.
    fig_meta, _ = charts.comparison_card(
        headline="Same index, same twenty years, two different returns.",
        items=[(f"{res['twr_mean'] * 100:.1f}%", "what the index did"),
               (f"{res['twr_mean'] * 100 + sm['panics']['gap_mean']:.1f}%",
                "what the investor who sold every 15% drawdown got")],
        emphasis=1,
        note=(f"Annualised, over {res['n_histories']} simulated histories. Five "
              f"sixths of the difference is not missed rebounds — it is "
              f"{sm['panics']['days_out'] / SESSIONS_PER_YEAR:.0f} of the 20 years "
              "spent in cash."),
        footer="The Standard Error", mode="light",
        alt=(f"Card comparing the index's {res['twr_mean'] * 100:.1f}% annualised "
             f"return with the "
             f"{res['twr_mean'] * 100 + sm['panics']['gap_mean']:.1f}% earned by "
             "an investor who sold on deep drawdowns."),
        caption="",
        path=str(IMG / f"b3-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    res = analyse()
    figs = figures(res)
    sm = res["summary"]
    panic, chase, dips, steady = (sm["panics"], sm["chases"], sm["buys dips"],
                                  sm["steady"])
    unlucky = sm["unlucky timing"]
    twr = res["twr_mean"] * 100

    post = Post(
        title="Individuals Sold a Record Amount on the Best Day in Market History",
        slug="individuals-sold-a-record-amount-on-the-best-day",
        date=POST_DATE,
        subtitle=("The gap between what an index returns and what its investors "
                  "earn is arithmetic — and I was wrong about where it comes from"),
        summary=(f"On 31 July 2026 the KOSPI rose {RECORD_DAY}%, the largest "
                 f"one-day gain in its history, and Korean individuals net sold a "
                 f"record {RETAIL_NET_SELL} trillion won into it. So I priced the "
                 f"rule: selling every {DRAWDOWN_EXIT:.0%} drawdown cost "
                 f"{abs(panic['gap_mean']):.2f} percentage points a year in my "
                 f"simulations. I expected the cost to be missed rebounds, and a "
                 f"matched control says it is not — five sixths of it is simply "
                 f"being out of a rising market for "
                 f"{panic['days_out'] / SESSIONS_PER_YEAR:.0f} of 20 years, and "
                 "the rest comes from a re-entry rule that will not buy back until "
                 "the index has risen 12%."),
        tags=["investing", "quantitative-finance", "statistics", "data-science",
              "behavioral-economics"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=1500, max_words=2400,
        table_figures=[figs["table"]],
        data_sources=[
            "Flow and index figures are quoted from published reports, not from a "
            "redistributed price series. 31 July 2026: the KOSPI rose 17.91% "
            "(+1,001.89 points) to 6,595.45, its largest one-day gain on record, "
            "while individuals net sold a record 8.2543 trillion won and "
            "foreigners net bought a record 7.2197 trillion won; individuals net "
            "sold 6.5437 trillion won over the week — Seoul Economic Daily, "
            "<https://en.sedaily.com/finance/2026/08/03/escaping-the-rollercoaster"
            "-kospi-index-recovers-6600-eyes>.",
            "Everything measured in this post is simulated with a fixed seed: "
            "GARCH(1,1) daily returns with Student-t shocks, 20-year histories, "
            "reproducible from the repo. No claim is estimated from Korean flow "
            "data.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "scipy", "standarderror")),
            "market": f"GARCH(1,1) log returns with t({SHOCK_DF:.0f}) shocks "
                      f"converted to simple returns and given a "
                      f"{DRIFT_ANNUAL:.0%} annual drift, unconditional "
                      f"daily sd fixed at {UNCOND_SD}% while persistence is swept "
                      f"over {PERSISTENCES} (omega solved from the target "
                      "variance, so only the arrival pattern changes)",
            "histories": f"{N_YEARS} years x {res['n_histories']} independent "
                         f"histories for the headline figures; 120 per point for "
                         "the persistence sweep",
            "rules": "one unit contributed per 21-session month; 'chases' doubles "
                     "after a month above +3% and withdraws a unit below -3%; "
                     "'buys dips' mirrors it; 'panics' liquidates on a "
                     f"{DRAWDOWN_EXIT:.0%} drawdown and re-enters at "
                     f"{DRAWDOWN_REENTRY:.0%}",
            "money-weighted return": "IRR by bisection on the terminal-value "
                                     "identity, annualised; a rule's own "
                                     "reallocation between the index and cash is "
                                     "not a cash flow, because the money never "
                                     "leaves the investor",
            "cash rate": f"{CASH_RATE:.0%} in the headline figures; at "
                         f"{CASH_RATE_ALT:.0%} the panic rule's gap is "
                         f"{res['alt_gap']:+.2f}pp instead of "
                         f"{panic['gap_mean']:+.2f}pp",
        },
    )

    post.add("Two records, one session", f"""
On 31 July 2026 the KOSPI rose **{RECORD_DAY}%** — the largest one-day gain in the
index's history. In the same session, Korean individuals net sold
**{RETAIL_NET_SELL} trillion won**, their largest daily net sale on record, and
foreign investors net bought **{FOREIGN_NET_BUY} trillion won**, theirs. Over the
week individuals net sold {RETAIL_WEEK_SELL} trillion.

I want to be careful about what that does and does not show, because the tempting
reading is a story about panic and the data cannot support it. An aggregate net
sale is the sum of forced margin liquidations, resting limit orders, index
rebalancing and deliberate decisions, and the four are indistinguishable in a
single number. Somebody being *liquidated* on the best day of the decade is a
different phenomenon from somebody *choosing* to sell into it, and you cannot tell
them apart from the tape.

What is worth doing instead is pricing the *rule*. Forget who did it and ask the
mechanical question: **if an investor reduces exposure after losses and restores it
after gains, what does that cost, and where does the cost come from?** That is
computable, it needs nobody's flow data, and the second half of the question turned
out to have a different answer from the one I was expecting when I started.
""".strip())

    post.add("The two returns of the same asset", f"""
Start with the distinction that makes this measurable.

The number a fund reports is a **time-weighted** return: compound the index's daily
moves and annualise. Every investor holding that index sees the same figure, and it
is indifferent to when anybody's money arrived.

The number an investor *earned* is a **money-weighted** return — the internal rate
of return of their actual cash flows. It is the rate that, applied to each
contribution for as long as that contribution was invested, reproduces the final
account value.

For a single lump sum held to the end the two are identical. As soon as money moves
they separate, and the difference is nothing but the timing of the flows. There is
no term in that arithmetic for how anybody felt.

So I built five investors in a simulated market — 20 years of daily returns with
volatility clustering and a {DRIFT_ANNUAL:.0%} drift, run
{res['n_histories']} times over — who hold the same index and differ only in when
their money moves:

- **steady** contributes one unit a month, forever, and never sells
- **chases the rally** doubles up after a month better than +3% and withdraws after
  one worse than -3%
- **buys the dips** does the exact opposite
- **panic sells** contributes steadily but liquidates the whole account when the
  index is {DRAWDOWN_EXIT:.0%} below its peak, returning when it has recovered to
  within {DRAWDOWN_REENTRY:.0%}
- **absent at random** is the control that makes this post work, and I will come to
  it in a moment

The index earned **{twr:.1f}% a year** for all of them.
""".strip())

    post.add("Exits matter. Contribution timing barely does.", f"""
The three rules that only vary their *contributions* land within a third of a point
of the index: **chases the rally** at {chase['gap_mean']:+.2f} percentage points a
year, **buys the dips** at {dips['gap_mean']:+.2f}, **steady** at
{steady['gap_mean']:+.2f}.

That surprised me and it is worth sitting with, because it is the opposite of how
these arguments usually go. Doubling your monthly contribution after good months
and withdrawing after bad ones — for twenty years — cost about
{abs(chase['gap_mean']):.2f} of a percentage point. The reason is unglamorous: after
a decade of contributions the monthly flow is small relative to the account, so
tilting it moves almost nothing. If your worry is that you are bad at timing your
monthly transfer, the arithmetic says stop worrying.

**Panic selling** is a different animal: {panic['gap_mean']:+.2f} points a year.
Same asset, same index, same {twr:.1f}%. It is negative in
{panic['share_negative'] * 100:.0f}% of the {res['n_histories']} histories, so it is
not bad luck.

And before any interpretation, the number that explains most of it: that rule spent
an average of **{panic['days_out'] / SESSIONS_PER_YEAR:.1f} of its 20 years in
cash**. A 15%-drawdown trigger with a 5% re-entry is not a mild risk control. In a
market with this much volatility it is a decision to be out of the market more than
half the time, and I did not appreciate that until I measured it.
""".strip(), figures=[figs["gap"]])

    post.add("I expected the rebound. It was mostly the drift.", f"""
Here is the hypothesis I started with, straight out of the previous post in this
series: the best days sit inside the crashes, so a rule that exits on a drawdown is
absent precisely for the rebound, and *that* is what it costs.

To test it I need a counterfactual that separates "out of the market" from "out at
the wrong moment". Comparing the panic rule with buy-and-hold cannot do that,
because it changes both at once. So: **absent at random** — an investor out of the
market for the same number of days, in absence episodes of the same lengths, placed
at random times. Same contributions, same cash rate, same everything except *when*
the absences fall.

That investor's gap is **{unlucky['gap_mean']:+.2f} points a year**, against the
panic rule's {panic['gap_mean']:+.2f}.

So the decomposition is:

- **{abs(unlucky['gap_mean']):.2f} of the {abs(panic['gap_mean']):.2f} points** —
  about five sixths — is simply being out of a rising market for a decade. Nothing
  clever, no timing, no rebound. Forgone drift.
- **{abs(panic['gap_mean'] - unlucky['gap_mean']):.2f} points** is attributable to
  *when* the absences fell.

The big number is the boring one. "I sold and sat in cash for
{panic['days_out'] / SESSIONS_PER_YEAR:.0f} of twenty years" explains most of the
damage before any behavioural story starts, and a story about missing rebounds was
doing work that plain arithmetic had already done.

The "missing the best days" channel is smaller than I expected too. The panic rule
held {panic['top_days_held'] * 100:.0f}% of the market's top-1% days; the random
absentee held {unlucky['top_days_held'] * 100:.0f}%. Under five percentage points
apart, for the same time out.
""".strip())

    def _row(k: str) -> str:
        moved = (f"{sm[k]['missed_move'] * 100:+.1f}%"
                 if sm[k]["missed_move"] else "—")
        return (f"| {SHORT[k]} | {twr + sm[k]['gap_mean']:.1f}% | "
                f"{sm[k]['gap_mean']:+.2f}pp | "
                f"{sm[k]['days_out'] / SESSIONS_PER_YEAR:.1f} | "
                f"{sm[k]['top_days_held'] * 100:.0f}% | {moved} |")

    # The same rows as the rendered table image, from the same numbers: the Hugo
    # page shows this markdown and Medium and Notion get the image substituted in.
    table_body = "\n".join(_row(k) for k in RULES)
    post.add("And the timing part is not clustering either", f"""
That leaves {abs(panic['gap_mean'] - unlucky['gap_mean']):.2f} points of genuine
timing cost, which I still expected to be made of volatility clustering. It is not,
and the control is unambiguous about it.

I re-ran everything against markets with the **same average volatility** and rising
persistence, from zero — independent draws — up to the 0.98 typically estimated on
equity indices. Holding the unconditional variance fixed matters: if raising
persistence also made the market louder, the louder market alone would widen every
gap.

If clustering were the mechanism, the timing-only gap would grow with persistence.
It shrinks:
**{res['sweep'][0.0]['panics'] - res['sweep'][0.0]['unlucky timing']:+.2f} points
with independent returns**, narrowing to
{res['sweep'][BASE_PERSISTENCE]['panics'] - res['sweep'][BASE_PERSISTENCE]['unlucky timing']:+.2f}
at equity-index persistence. Clustering makes this rule's timing slightly *less*
costly, and the distribution in Fig 3 says the same thing from the other side: the
no-clustering version of the panic rule averages
{res['iid_gaps'].mean():+.2f} points against {panic['gap_mean']:+.2f} with
clustering. Worse, not better.

The actual mechanism is in the rule's own definition, and here is the table where
it shows up:

| investor | what they earned | gap | years in cash | best days held | index move while out |
|---|---|---|---|---|---|
{table_body}

**While the panic rule was in cash the index rose
{panic['missed_move'] * 100:.1f}% per absence; while the random absentee was in
cash it rose {unlucky['missed_move'] * 100:.1f}%.** Of course it did. Exiting at
{DRAWDOWN_EXIT:.0%} below the peak and re-entering at {DRAWDOWN_REENTRY:.0%} below
it means the index has to climb about
{100 * ((1 - DRAWDOWN_REENTRY) / (1 - DRAWDOWN_EXIT) - 1):.1f}% before the rule is
allowed back in. The rally is not something the rule unluckily misses. **It is the
re-entry condition.** You wrote a rule that will not let you own the asset until it
has gone up, and then you were absent while it went up.

None of that contradicts the earlier post — the best days really do cluster inside
the crashes. It says that for *this* rule, that effect is a minor term next to two
larger ones: the drift you are not earning, and a re-entry threshold that
guarantees you buy back higher than you sold.
""".strip(), figures=[figs["mechanism"], figs["distribution"]])

    post.add("Where I am overstating it", f"""
Four places, and the first one moves the number a lot.

**Cash earns nothing in my headline figures.** That is the objection I would raise
first, and it matters more than I expected: at a {CASH_RATE_ALT:.0%} cash rate the
panic rule's gap is **{res['alt_gap']:+.2f} points instead of
{panic['gap_mean']:+.2f}**. More than half the penalty is the yield you would
actually have earned on the sidelines. The ranking survives and the mechanism
survives, but anyone quoting "panic selling costs 2.4 points" — including this post's
own chart — is quoting a zero-interest world.

**My thresholds are aggressive.** Exit at 15% below the peak, re-enter at 5%: that
combination is out of the market
{panic['days_out'] / SESSIONS_PER_YEAR:.1f} years in 20 here. A wider re-entry band
would spend less time in cash and forgo less drift, and the timing term would shrink
with it, because the re-entry gap *is* the timing term. Someone should sweep those
two thresholds; I have not, and the honest read of my figure is that it prices one
particular rule rather than "selling drawdowns" in general.

**GARCH is not the market.** It reproduces clustering and fat tails, which is what
the argument needed, and it has no jumps, no regime changes and a symmetric response
to good and bad news that equity indices measurably violate. The asymmetry would
matter here: real volatility rises more after falls, which lengthens absences.

**And the Korean flow numbers are a hook, not evidence.** I have estimated nothing
from them. What individuals as a group earned in 2026 is a question for account-level
data, which I do not have and almost nobody outside a regulator does.
""".strip())

    post.add("What I would actually take from this", f"""
**A drawdown rule is a market-timing strategy, and it should be backtested like
one.** "I sell when the index is 15% down" has an entry rule, an exit rule and a
measurable cost, and most of that cost is knowable before you look at a single
rebound: it is the drift you will not earn while you are out, times how long the
rule keeps you out. Compute *that* first. If the answer is "out half the time", the
rest of the analysis is a rounding error.

**Look at your re-entry condition before your exit condition.** The exit is the part
people agonise over and the re-entry is where the money goes. Any rule that requires
a recovery before it buys back has written "buy higher than I sold" into itself. If
you want a rule, put the re-entry on a calendar, not on a level.

**Ask which return a fund is quoting you.** Time-weighted is the industry standard
and it is the right number for judging the *manager*. Money-weighted is the right
number for judging your own *outcome*, and for a fund with volatile flows the two
can differ by more than the manager's entire claimed edge. Both are legitimate; only
one is about you.

**And keep the counterfactual matched.** The reason this post has a finding rather
than a moral is one extra simulated investor: absent for the same days, at random
times. Without it I would have written the story I expected — clustering, rebounds,
discipline — and the numbers would have looked like they agreed with me, because
{abs(panic['gap_mean']):.2f} points is a big number and big numbers are persuasive
even when their explanation is wrong.

Next in this series: the KOSPI has more than 800 listed companies, and I want to
find out how many of them a "diversified" index position is actually a bet on. That
one has a formula too, and the answer is a much smaller number than 800.
""".strip())

    return post


if __name__ == "__main__":
    p = build()
    print(p.title, "|", p.word_count(), "words |", len(p.figures), "figures")
    for issue in p.audit():
        print("  audit:", issue)
