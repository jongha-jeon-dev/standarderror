"""exp010 — Korea's jeonse deposit, priced as the junior tranche it is.

Backlog: Track C, numbers on society, and the third markets post. Requested
directly; the reason it belongs here is that jeonse has an exact core.

Under jeonse a tenant hands the landlord a lump sum — commonly half to four
fifths of what the home is worth — pays no monthly rent for two years, and gets
the sum back at the end. Every English explainer describes this as an
interest-free loan from tenant to landlord, which is right and is where they stop.
The loan has a security structure, and the structure is the whole story.

At the end of the term the tenant receives

    min(D, max(0, lambda * V_T - M))

for deposit `D`, senior mortgage `M`, terminal home value `V_T` and liquidation
ratio `lambda`. That is a put spread on the home struck between `M` and `M + D`:
the tenant holds the **first-loss tranche** of a single-property loan, and the
attachment point is

    V* = (M + D) / lambda

which contains no volatility, no drift and no model. It is arithmetic on two
published numbers, and it is the post.

Run with `M = 0` — no mortgage at all, the friendliest possible landlord — and
two Korean cohorts sit at opposite ends of the same instrument:

* A **Seoul apartment** in January 2026. Jeonse-to-price ratio 50.92%, an
  all-time low; apartments at auction clear at 101% of appraisal. The tenant is
  breached at a **49.6% fall**.
* A **Seoul villa** in December 2022. Ratio 78.6%; villas at auction clear at 79%.
  Breached at a **0.5% fall**.

Same contract, same country, 49 points of cushion against half a point. Two things
worth being careful about, because an earlier draft of this got both wrong:

* That gap is **two** published ratios moving, not one. Of the 49.1 points, the
  deposit ratio (50.92% to 78.6%) accounts for 27.4 and the auction clearing ratio
  (101% to 79%) for 21.7. Villas are riskier because the deposit is bigger *and*
  because a court gets less for them, in roughly equal measure.
* The trigger itself is **linear** in the deposit ratio — slope -1/lambda, which is
  why Fig 1 is three straight lines. Quoting the two cohorts' ratio, "a factor of
  98", was doubly wrong: a ratio of a linear function's values measures no curvature,
  and it is large only because the villa cohort sits near the zero crossing. The
  convexity in this post is real but it lives in the *loss*, not the trigger.

Only the probability of breach needs a model, and the model is deliberately thin: a lognormal terminal value, volatility swept across
the range Giacoletti (2021) measures for individual homes rather than indices,
integrated in closed form and checked against Monte Carlo.

Three controls, because the post's claim is mechanical and mechanical claims can
be switched off:

1. **Volatility to zero.** It passes everywhere except at the ratio that was
   actually being written. The Dec-2022 villa cohort sits 0.51% from its attachment
   point, so even at sigma = 0.2%/yr — two orders of magnitude below any measured
   housing market — the breach probability is 3.6%, and at sigma = 2% it is 43%.
   Half a percent is under two standard deviations of almost nothing.
2. **Liquidation haircut to one.** Holding the ratio at 78.6% and paying full
   value at auction cuts the expected loss by a factor of ten. So the villa
   problem was a *liquidation* problem before it was a market-risk problem, which
   is not how it was reported.
3. **The senior mortgage.** Every headline figure here assumes none, which makes
   them a floor on the tenant's risk rather than an estimate of it.

And one out-of-sample check I did not expect to get. The model says the Dec-2022
villa cohort was losing about 3.3% of deposit a year while HUG charged about 0.13%
a year to guarantee it — a factor of 25. Independently: HUG collected 3,525 억원
of premiums between 2020 and August 2024 and paid 9조 4,189억원 of subrogation
over 2020-2024, recovering 24%, for a net 7조 731억원 — a realised loss ratio of
20. Two unrelated routes to the same order of magnitude.

Nothing here redistributes a price series. Every input is a published scalar,
quoted with its source in `data_sources`, and everything else is closed form or
fixed-seed simulation.

Run: `quantpost run exp010_jeonse_tranche --publish`
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import quantpost as qp
from quantpost.credit import tranche
from quantpost.render import Post
from quantpost.viz import charts, theme

IMG = qp.SETTINGS.build_dir / "img"
EXT = os.environ.get("QUANTPOST_FIG_EXT", "png")
SEED = qp.SETTINGS.seed

TERM_YEARS = 2.0                     # the statutory jeonse term

# --- published facts, every one sourced in `data_sources` ----------------------
# Jeonse-to-sale-price ratios (전세가율), as percentages of the home's value.
RATIO_SEOUL_APT = 50.92              # KB, January 2026 — an all-time low
RATIO_SEOUL_APT_PREV = 50.87         # KB, May 2023 — the record it broke
RATIO_SEOUL_VILLA_2022 = 78.6        # REB, December 2022
RATIO_SEOUL_VILLA_2023 = 68.5        # REB, December 2023
RATIO_SEOUL_VILLA_2024 = 65.4        # REB, December 2024
RATIO_NATIONWIDE = 65.0              # KB composite, August 2025
RATIO_SEOUL_ALL = 57.0               # KB composite, August 2025
# Auction clearing ratios (낙찰가율): winning bid over appraised value.
AUCTION_SEOUL_APT = 101.0            # GG Auction, July 2026
AUCTION_NATIONWIDE_APT = 85.4        # GG Auction, July 2026
AUCTION_SEOUL_VILLA = 79.0           # GG Auction, Seoul villas
AUCTION_VILLA_TAIL = 41.0            # villas sold after four failed rounds
# The in-kind coupon and the alternatives.
CONVERSION_NATIONWIDE = 6.62         # REB officetel conversion rate, June 2026
CONVERSION_SEOUL = 6.06              # REB, Seoul, June 2026 — a record
BASE_RATE = 2.75                     # BOK Base Rate after 16 July 2026
COFIX = 3.05                         # June 2026
HUG_PREMIUM = 0.128                  # %/yr, mid-band of the published schedule
HUG_PREMIUM_BAND = (0.097, 0.211)
STATUTORY_SPREAD = 2.0               # 주택임대차보호법: cap is base rate + 2pp,
STATUTORY_CAP_ABS = 10.0             # or 10%, whichever is lower
# HUG's realised experience, 2020-2024 (억원).
HUG_INCIDENTS = 110_441
HUG_INCIDENT_COUNT = 50_941
HUG_SUBROGATION = 94_189
HUG_SUBROGATION_COUNT = 43_631
HUG_RECOVERED = 23_458
HUG_PREMIUM_INCOME = 3_525           # 2020 - August 2024
HUG_SUPPLY = 993_914                 # cumulative guarantee supply, 2020-2023
HUG_BY_TYPE = {"apartment": (13_251, 32_685), "villa (다세대)": (24_870, 51_960),
               "officetel": (10_648, 21_802)}
# 2026 market context.
SEOUL_MONTHLY_SHARE = 54.1           # % of Seoul apartment rental deals, June 2026
SEOUL_JEONSE_SHARE = 45.9
SEOUL_LISTINGS_FALL = 32.8           # % y/y fall in apartment jeonse listings

# --- model parameters, swept rather than asserted ------------------------------
# Giacoletti (2021) measures 10-18%/yr idiosyncratic volatility for individual
# homes against 4-5%/yr for the metropolitan index. The tenant owns exposure to
# one house, so the individual figure is the relevant one; the index figure is
# carried through as the optimistic bound.
SIGMAS = {"metro index (4-5%)": 0.05, "individual home, low (10-12%)": 0.12,
          "individual home, high (18%)": 0.18}
SIGMA_BASE = 0.12
RATIO_GRID = np.round(np.arange(0.35, 0.951, 0.005), 4)
N_SIM = 400_000                      # Monte Carlo check of the closed form


def attachment(ratio_pct: float, auction_pct: float,
               mortgage_pct: float = 0.0) -> float:
    """Attachment point, in the percentage units this post quotes everything in.

    Thin wrapper over `quantpost.credit.attachment_point`. The library function is
    unit-agnostic; this one fixes the convention — deposit, mortgage and auction
    ratio all as percentages of the home's sale price — so the post cannot mix
    fractions and percentages by accident.
    """
    return tranche.attachment_point(ratio_pct / 100.0, auction_pct / 100.0,
                                    mortgage_pct / 100.0)


def required_fall(ratio_pct: float, auction_pct: float,
                  mortgage_pct: float = 0.0) -> float:
    """Percentage fall in the home's value that first leaves the tenant short.

    Negative means the tenant is *already* short at today's price.
    """
    return tranche.required_fall(ratio_pct / 100.0, auction_pct / 100.0,
                                 mortgage_pct / 100.0)


def expected_loss(ratio_pct: float, auction_pct: float, sigma: float, *,
                  mortgage_pct: float = 0.0, drift: float = 0.0,
                  term: float = TERM_YEARS) -> dict:
    """Expected shortfall on the deposit, per year, in closed form."""
    return tranche.expected_shortfall_rate(
        ratio_pct / 100.0, auction_pct / 100.0, sigma,
        senior=mortgage_pct / 100.0, drift=drift, term=term)


def simulate_loss(ratio_pct: float, auction_pct: float, sigma: float, *,
                  mortgage_pct: float = 0.0, drift: float = 0.0,
                  term: float = TERM_YEARS, n_draws: int = N_SIM,
                  seed: int = 11) -> dict:
    """Monte Carlo check of the integral above."""
    return tranche.simulate_shortfall_rate(
        ratio_pct / 100.0, auction_pct / 100.0, sigma,
        senior=mortgage_pct / 100.0, drift=drift, term=term,
        n_draws=n_draws, seed=seed)


def spread_earned(conversion_pct: float = CONVERSION_NATIONWIDE,
                  alternative_pct: float = BASE_RATE,
                  premium_pct: float = HUG_PREMIUM) -> float:
    """What the tenant is paid, per year, for carrying the tranche.

    The coupon is paid in kind: the tenant occupies a home and pays no rent, and
    the market's price for that is the jeonse-to-monthly-rent conversion rate. Net
    of the return the same money would have earned elsewhere, and of the premium
    on the state guarantee, what is left is compensation for credit risk and
    nothing else.
    """
    return conversion_pct - alternative_pct - premium_pct


def rollover_gap(ratio_now_pct: float, ratio_then_pct: float,
                 price_change_pct: float = 0.0) -> float:
    """Cash the landlord must produce at renewal, as a percentage of the deposit.

    The refund is `D`; the incoming tenant pays `ratio_now * V_T`. The gap is what
    the landlord funds from somewhere else. This is *linear* in the price move and
    it bites at moves of a few percent, where the tranche loss needs tens of
    percent — which is why Korea had a mass refund crisis without anything
    resembling a housing crash.
    """
    d_then = ratio_then_pct / 100.0
    d_now = ratio_now_pct / 100.0
    v = 1.0 + price_change_pct / 100.0
    return 100.0 * max(0.0, d_then - d_now * v) / d_then


def analyse() -> dict:
    spread = spread_earned()

    cohorts = {
        "Seoul apartment, Jan 2026": dict(ratio_pct=RATIO_SEOUL_APT,
                                         auction_pct=AUCTION_SEOUL_APT),
        "Seoul, all housing, 2025": dict(ratio_pct=RATIO_SEOUL_ALL,
                                         auction_pct=AUCTION_SEOUL_APT),
        "nationwide, all housing, 2025": dict(ratio_pct=RATIO_NATIONWIDE,
                                              auction_pct=AUCTION_NATIONWIDE_APT),
        "Seoul villa, Dec 2024": dict(ratio_pct=RATIO_SEOUL_VILLA_2024,
                                      auction_pct=AUCTION_SEOUL_VILLA),
        "Seoul villa, Dec 2022": dict(ratio_pct=RATIO_SEOUL_VILLA_2022,
                                      auction_pct=AUCTION_SEOUL_VILLA),
    }
    table = {}
    for name, kw in cohorts.items():
        row = expected_loss(sigma=SIGMA_BASE, **kw)
        row["coverage"] = (spread / row["loss_per_year_pct"]
                           if row["loss_per_year_pct"] > 1e-12 else np.inf)
        row.update(kw)
        table[name] = row

    # Curves: the exact trigger, and the modelled probability of hitting it.
    triggers = pd.DataFrame(
        {f"{label} ({a:.0f}% at auction)":
            [required_fall(100 * r, a) for r in RATIO_GRID]
         for label, a in (("Seoul apartments", AUCTION_SEOUL_APT),
                          ("apartments, nationwide", AUCTION_NATIONWIDE_APT),
                          ("Seoul villas", AUCTION_SEOUL_VILLA))},
        index=100 * RATIO_GRID)
    losses = pd.DataFrame(
        {label: [expected_loss(100 * r, AUCTION_SEOUL_VILLA, s)["loss_per_year_pct"]
                 for r in RATIO_GRID]
         for label, s in SIGMAS.items()},
        index=100 * RATIO_GRID)

    # Control 1 — volatility to zero, at the ratio that was actually written.
    vol_control = {s: expected_loss(RATIO_SEOUL_VILLA_2022, AUCTION_SEOUL_VILLA, s)
                   for s in (0.002, 0.02, 0.05, 0.12, 0.18)}
    # Control 2 — the liquidation haircut to nothing, ratio held fixed.
    haircut_control = {
        a: expected_loss(RATIO_SEOUL_VILLA_2022, a, SIGMA_BASE)
        for a in (100.0, 95.0, 90.0, 85.0, AUCTION_SEOUL_VILLA)}
    # Control 3 — a senior mortgage, which every headline figure here excludes.
    mortgage_control = {
        m: expected_loss(RATIO_SEOUL_APT, AUCTION_SEOUL_APT, SIGMA_BASE,
                         mortgage_pct=m) for m in (0.0, 10.0, 20.0, 30.0, 40.0)}

    # Verification.
    checks = {}
    for i, (name, kw) in enumerate(cohorts.items()):
        # The trigger is computed twice — once by the exact function the figures use,
        # once inside the loss integral the table uses — and a refactor onto the
        # library silently gave the two different units, so Fig 1 shipped with a
        # y-axis running to -12,000% while the table stayed correct. Two code paths
        # for one quantity need an assertion, not a review.
        if abs(required_fall(**kw) - table[name]["required_fall_pct"]) > 1e-9:
            raise AssertionError(
                f"{name}: exact trigger {required_fall(**kw)} disagrees with the "
                f"modelled one {table[name]['required_fall_pct']}")
        sim = simulate_loss(sigma=SIGMA_BASE, seed=200 + i, **kw)
        exact = table[name]
        ref = max(exact["loss_per_year_pct"], 1e-4)
        checks[name] = {
            "exact": exact["loss_per_year_pct"], "simulated": sim["loss_per_year_pct"],
            "abs_err_pp": abs(sim["loss_per_year_pct"] - exact["loss_per_year_pct"]),
            "rel_err_pct": 100.0 * abs(sim["loss_per_year_pct"]
                                       - exact["loss_per_year_pct"]) / ref}

    # Which of the two published ratios moves the trigger, and by how much. An
    # earlier draft quoted the *ratio* of the two cohorts' triggers — 49.58 over
    # 0.51, "a factor of 98" — and called it evidence of non-linearity. Both halves
    # of that were wrong. The trigger is exactly linear in the deposit ratio, with
    # slope -1/lambda, so a ratio of two of its values measures no curvature; and it
    # is only large because the villa cohort sits near the zero crossing, which makes
    # the figure an artefact of proximity to zero rather than a fact about housing. A
    # *difference* in points is the honest statistic, and it has to be split between
    # the two inputs, which moved by comparable amounts.
    apt_kw = cohorts["Seoul apartment, Jan 2026"]
    villa_kw = cohorts["Seoul villa, Dec 2022"]
    apt_fall, villa_fall = required_fall(**apt_kw), required_fall(**villa_kw)
    ratio_only = required_fall(villa_kw["ratio_pct"], apt_kw["auction_pct"])
    auction_only = required_fall(apt_kw["ratio_pct"], villa_kw["auction_pct"])
    decompose = {
        "apt_fall_pct": apt_fall, "villa_fall_pct": villa_fall,
        "total_pp": apt_fall - villa_fall,
        "deposit_ratio_only_pct": ratio_only,
        "auction_ratio_only_pct": auction_only,
        "deposit_ratio_pp": apt_fall - ratio_only,
        "auction_ratio_pp": ratio_only - villa_fall,
        # The slope is the same at every ratio, which is the point.
        "slope_per_point": 100.0 / villa_kw["auction_pct"],
    }

    # The rollover cash call — arithmetic on two published ratios.
    gaps = {
        "ratio to 65.4%, prices flat": rollover_gap(
            RATIO_SEOUL_VILLA_2024, RATIO_SEOUL_VILLA_2022, 0.0),
        "ratio to 65.4%, prices -10%": rollover_gap(
            RATIO_SEOUL_VILLA_2024, RATIO_SEOUL_VILLA_2022, -10.0),
        "ratio held at 78.6%, prices -10%": rollover_gap(
            RATIO_SEOUL_VILLA_2022, RATIO_SEOUL_VILLA_2022, -10.0),
        "ratio held at 78.6%, prices -20%": rollover_gap(
            RATIO_SEOUL_VILLA_2022, RATIO_SEOUL_VILLA_2022, -20.0),
    }

    # HUG's realised loss ratio, and what the model says the same cohort cost.
    hug_net = HUG_SUBROGATION - HUG_RECOVERED
    hug = {
        "recovery_pct": 100.0 * HUG_RECOVERED / HUG_SUBROGATION,
        "net_loss": hug_net,
        "loss_ratio": hug_net / HUG_PREMIUM_INCOME,
        "gross_ratio": HUG_SUBROGATION / HUG_PREMIUM_INCOME,
        "subrogation_pct_of_supply": 100.0 * HUG_SUBROGATION / HUG_SUPPLY,
        "premium_pct_of_supply": 100.0 * HUG_PREMIUM_INCOME / HUG_SUPPLY,
        "villa_share_of_incidents": 100.0 * HUG_BY_TYPE["villa (다세대)"][0]
        / HUG_INCIDENT_COUNT,
        "villa_over_apartment": (HUG_BY_TYPE["villa (다세대)"][0]
                                 / HUG_BY_TYPE["apartment"][0]),
        "mean_incident": HUG_INCIDENTS / HUG_INCIDENT_COUNT,   # 억원 per incident
    }
    model_mispricing = (table["Seoul villa, Dec 2022"]["loss_per_year_pct"]
                        / HUG_PREMIUM)

    # Coverage is a ratio with a near-zero denominator in the safe cohorts, so
    # quoting it to five figures at one volatility would be fake precision. Recompute
    # it at the top of the swept range, and quote that.
    robust = {}
    for name, kw in cohorts.items():
        hi = expected_loss(kw["ratio_pct"], kw["auction_pct"], max(SIGMAS.values()))
        robust[name] = {
            "loss_per_year_pct": hi["loss_per_year_pct"],
            "p_breach_pct": hi["p_breach_pct"],
            "coverage": (spread / hi["loss_per_year_pct"]
                         if hi["loss_per_year_pct"] > 1e-12 else np.inf)}

    # What default rate the spread would cover, if the tenant were a bond desk.
    lgd = 1.0 - hug["recovery_pct"] / 100.0
    breakeven_default = spread / (100.0 * lgd)

    return {
        "spread": spread, "table": table, "triggers": triggers, "losses": losses,
        "vol_control": vol_control, "haircut_control": haircut_control,
        "mortgage_control": mortgage_control, "checks": checks, "gaps": gaps,
        "hug": hug, "model_mispricing": model_mispricing, "lgd": lgd,
        "robust": robust,
        "haircut_gap_pp": 100.0 - AUCTION_SEOUL_VILLA,
        "haircut_factor": (haircut_control[AUCTION_SEOUL_VILLA]["loss_per_year_pct"]
                           / haircut_control[100.0]["loss_per_year_pct"]),
        "breakeven_default_pct": 100.0 * breakeven_default,
        "statutory_cap": min(STATUTORY_CAP_ABS, BASE_RATE + STATUTORY_SPREAD),
        "max_check_err_pp": max(c["abs_err_pp"] for c in checks.values()),
        "decomposition": decompose,
    }


TABLE_HEADER = ["cohort", "deposit ratio", "auction ratio", "fall needed",
                "P(breach)", "spread covers"]


def table_rows(res: dict) -> list[list[str]]:
    """The cohort table, formatted once.

    Both the rendered image (for Medium and Notion, which have no tables) and the
    markdown table in the body come from here. exp007 shipped a post where those two
    were formatted separately and disagreed on a column, so Hugo and Medium showed
    different numbers for the same quantity.
    """
    rows = []
    for name, r in res["table"].items():
        fall = r["required_fall_pct"]
        cover = r["coverage"]
        rows.append([
            name, f"{r['ratio_pct']:.1f}%", f"{r['auction_pct']:.0f}%",
            (f"{fall:.1f}%" if fall >= 0 else "already short"),
            (f"{r['p_breach_pct']:.1f}%" if r["p_breach_pct"] >= 0.1
             else "<0.1%"),
            (">1,000x" if cover > 1000 else
             f"{cover:,.0f}x" if cover >= 10 else f"{cover:.1f}x")])
    return rows


def figures(res: dict) -> dict:
    src_exact = ("Arithmetic on published jeonse-to-price ratios and auction "
                 "clearing ratios. No price series is used.")
    src_model = ("Closed-form expected loss under a lognormal terminal value; "
                 "volatility swept, not estimated. No price series is used.")
    figs = {}
    t = res["table"]
    apt = t["Seoul apartment, Jan 2026"]
    villa22 = t["Seoul villa, Dec 2022"]
    villa24 = t["Seoul villa, Dec 2024"]

    # F1 — the trigger. Exact, and the whole argument.
    def mark_cohorts(_fig, ax):
        m = theme.LIGHT
        ax.axhline(0.0, color=m.muted, lw=1.2, ls=(0, (5, 3)))
        # Rotated in-plot labels ran straight through the zero-line annotation and
        # through the villa curve. Horizontal labels, one per row at the top, in the
        # same pattern exp009 used for its correlation regimes.
        for i, (ratio, label) in enumerate((
                (RATIO_SEOUL_APT, "Seoul apartments, Jan 2026"),
                (RATIO_SEOUL_VILLA_2024, "Seoul villas, Dec 2024"),
                (RATIO_SEOUL_VILLA_2022, "Seoul villas, Dec 2022"))):
            ax.axvline(ratio, color=m.muted, lw=1.0, ls=(0, (2, 3)))
            ax.annotate(label, (ratio, 0.975 - 0.085 * i),
                        xycoords=("data", "axes fraction"),
                        xytext=(5, 0), textcoords="offset points", ha="left",
                        va="top", fontsize=8.5, color=m.muted)
        ax.annotate("below zero the deposit already exceeds the auction proceeds",
                    (0.015, 0.05), xycoords="axes fraction", ha="left", va="bottom",
                    fontsize=8.5, color=m.muted)

    fig_meta, _ = charts.lines(
        res["triggers"], mode="light", direct_labels=False, decorate=mark_cohorts,
        title="How far the house has to fall before the tenant loses money",
        subtitle=("Deposit as the only claim on the property — no mortgage ahead "
                  "of it, which makes every line here the friendliest case. Each "
                  "curve is one type of property's auction clearing ratio."),
        ylabel="fall in the home's value needed to breach (%)",
        xlabel="jeonse deposit as % of the home's sale price", source=src_exact,
        alt=("Three downward-sloping lines. At a 51% deposit ratio the required "
             "fall is around 50%; the Seoul-villa line reaches zero at a 79% "
             "ratio and goes negative beyond it, meaning the deposit exceeds what "
             "the property would fetch at auction before prices move at all."),
        caption=(f"Fig 1. No model in this chart — it is "
                 f"(deposit + mortgage) / auction ratio, and nothing else. A Seoul "
                 f"apartment tenant in January 2026 needs a "
                 f"**{apt['required_fall_pct']:.0f}% fall** before a single won is "
                 f"at risk. A Seoul villa tenant in December 2022 needed "
                 f"**{villa22['required_fall_pct']:.1f}%**. The lines are straight "
                 f"because the trigger is linear in the deposit ratio, so read the "
                 f"vertical gap in points, not as a ratio — and note that the two "
                 f"cohorts differ in the auction ratio as well as the deposit ratio, "
                 f"by {res['decomposition']['auction_ratio_pp']:.1f} points of the "
                 f"{res['decomposition']['total_pp']:.0f}."),
        path=str(IMG / f"c4-f1-trigger.{EXT}"))
    figs["trigger"] = fig_meta

    # F2 — the loss, which is where volatility enters and where it explodes.
    def mark_spread(_fig, ax):
        m = theme.LIGHT
        ax.axhline(res["spread"], color=m.series[7], lw=1.6, ls=(0, (5, 3)))
        ax.annotate(f"what the tenant is paid: {res['spread']:.2f}%/yr",
                    (0.015, res["spread"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points", ha="left",
                    va="bottom", fontsize=8.5, color=m.series[7])
        ax.axhline(HUG_PREMIUM, color=m.muted, lw=1.2, ls=(0, (2, 3)))
        # Right of centre and below the line is the only region no curve enters:
        # every curve is far above 0.128 by the time the axis is half over.
        ax.annotate(f"state guarantee premium: {HUG_PREMIUM:.3f}%/yr",
                    (0.985, HUG_PREMIUM), xycoords=("axes fraction", "data"),
                    xytext=(0, -6), textcoords="offset points", ha="right",
                    va="top", fontsize=8.5, color=m.muted)

    fig_meta, _ = charts.lines(
        res["losses"], mode="light", direct_labels=False, decorate=mark_spread,
        logy=True, ylim=(1e-3, 20.0),
        title="The loss is an option, so it does not rise in a straight line",
        subtitle=("Expected shortfall on a Seoul villa deposit, per year, against "
                  "the deposit ratio. Three volatilities: the metropolitan index, "
                  "and the low and high ends of what individual homes measure."),
        ylabel="expected loss on the deposit (%/yr, log scale)",
        xlabel="jeonse deposit as % of the home's sale price", source=src_model,
        alt=("Three curves rising steeply and then flattening as the deposit ratio "
             "increases, on a log scale spanning five orders of magnitude. Two "
             "horizontal reference lines mark the spread the tenant earns and the "
             "much lower state guarantee premium."),
        caption=(f"Fig 2. Between a 55% and an 80% deposit ratio the expected loss "
                 f"rises by roughly four orders of magnitude. Nothing about the "
                 f"contract changed across that range; the tenant simply moved "
                 f"from a tranche that is far out of the money to one that is at "
                 f"the money. The two reference lines are what the tenant receives "
                 f"for carrying it ({res['spread']:.2f}%/yr) and what the state "
                 f"charged to guarantee it ({HUG_PREMIUM:.3f}%/yr)."),
        path=str(IMG / f"c4-f2-loss.{EXT}"))
    figs["loss"] = fig_meta

    # F3 — the other mechanism, which is linear and is what actually happened.
    labels = list(res["gaps"].keys())
    fig_meta, _ = charts.ranked_bars(
        labels, [res["gaps"][k] for k in labels], mode="light", value_fmt=".1f",
        title="The refund gap arrives long before the collateral gap",
        subtitle=("Cash a landlord must find at renewal, as a percentage of the "
                  "deposit being returned. Seoul villas, rolling a December 2022 "
                  "contract at December 2024 ratios."),
        xlabel="cash the landlord must produce (% of the deposit)",
        source=src_exact,
        alt=("Four horizontal bars: " + ", ".join(
            f"{k} {v:.1f}%" for k, v in res["gaps"].items()) + "."),
        caption=(f"Fig 3. The 'prices flat' bar contains no price move at all. Seoul villa "
                 f"deposit ratios fell from {RATIO_SEOUL_VILLA_2022}% to "
                 f"{RATIO_SEOUL_VILLA_2024}% between December 2022 and December "
                 f"2024, so a landlord refunding an unchanged house had to produce "
                 f"{res['gaps']['ratio to 65.4%, prices flat']:.1f}% of the "
                 f"deposit in cash. That is a funding problem, it is linear, and it "
                 f"starts at price moves an order of magnitude smaller than the "
                 f"ones Fig 1 is about."),
        path=str(IMG / f"c4-f3-rollover.{EXT}"))
    figs["rollover"] = fig_meta

    # T1 — the five cohorts, as an image because Medium has no tables.
    fig_meta, _ = charts.table_image(
        table_rows(res), header=TABLE_HEADER,
        title="One contract, five cohorts, five completely different risks",
        subtitle=(f"No senior mortgage assumed, so every figure is a floor. Breach "
                  f"probability and coverage at {100 * SIGMA_BASE:.0f}%/yr "
                  f"volatility over a two-year term."),
        source=src_model, mode="light", bold_cols=(3, 5), align="lrrrrr",
        alt=("Table of five Korean rental cohorts. A Seoul apartment in January "
             "2026 needs a 50% fall to breach and the spread covers the expected "
             "loss more than a thousand times over; a Seoul villa in December 2022 "
             "needed a fall of half a percent and the spread covers the loss about "
             "once."),
        caption=("Table 1. Every column but the last two is either published or "
                 "arithmetic on published numbers. The last column is the point: "
                 "the same in-kind coupon is enormous compensation in one row and "
                 "not compensation at all in another."),
        path=str(IMG / f"c4-t1-cohorts.{EXT}"))
    figs["table"] = fig_meta

    # HERO — the finding is a comparison of two triggers.
    fig_meta, _ = charts.comparison_card(
        headline="How far must a Korean home fall before the tenant's deposit is at risk?",
        items=[(f"{apt['required_fall_pct']:.0f}%", "Seoul apartment, Jan 2026"),
               (f"{villa22['required_fall_pct']:.1f}%", "Seoul villa, Dec 2022")],
        emphasis=1,
        note=("Same contract, same city, same law. The deposit is the first-loss "
              "tranche of a single-property loan, and its attachment point is one "
              "published ratio divided by another."),
        footer="quantpost", mode="light",
        alt=(f"Card comparing the price fall needed to put a deposit at risk: "
             f"{apt['required_fall_pct']:.0f}% for a Seoul apartment in January "
             f"2026 against {villa22['required_fall_pct']:.1f}% for a Seoul villa "
             f"in December 2022."),
        caption="",
        path=str(IMG / f"c4-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    res = analyse()
    figs = figures(res)
    t = res["table"]
    apt = t["Seoul apartment, Jan 2026"]
    villa22 = t["Seoul villa, Dec 2022"]
    villa24 = t["Seoul villa, Dec 2024"]
    nation = t["nationwide, all housing, 2025"]
    hug = res["hug"]
    vc, hc, mc = res["vol_control"], res["haircut_control"], res["mortgage_control"]
    table_body = "\n".join("| " + " | ".join(r) + " |" for r in table_rows(res))

    post = Post(
        title="Some Korean Renters Needed a 50% Crash to Lose Money. Others Needed 0.5%.",
        slug="korean-renters-first-loss-tranche",
        subtitle=("Jeonse is a loan from tenant to landlord, and the security "
                  "structure decides everything"),
        summary=(
            f"Under Korea's jeonse system a tenant hands the landlord a lump sum "
            f"worth half to four fifths of the home, pays no monthly rent for two "
            f"years, and gets the sum back at the end. Everyone calls this an "
            f"interest-free loan and stops there. It is a loan with a security "
            f"structure: the tenant holds the first-loss tranche of a "
            f"single-property mortgage, and the attachment point is one published "
            f"ratio divided by another. A Seoul apartment tenant in January 2026 "
            f"needs a {apt['required_fall_pct']:.0f}% fall in the home's value "
            f"before a single won is at risk. A Seoul villa tenant in December "
            f"2022 needed {villa22['required_fall_pct']:.1f}%. The compensation "
            f"is the same in both cases: about {res['spread']:.1f}% a year of "
            f"free housing, net of what the money would otherwise earn. In one "
            f"row that covers the expected loss more than a thousand times over; in "
            f"the other it covers it once."),
        tags=["investing", "risk-management", "housing", "korea", "quantitative-finance"],
        author=qp.SETTINGS.author,
        code_url=qp.SETTINGS.code_repo_url,
        min_words=1500, max_words=2400,
        table_figures=[figs["table"]],
        data_sources=[
            "Jeonse-to-sale-price ratio (전세가율) for Seoul apartments of 50.92% "
            "in January 2026, an all-time low since the series began in April 2013, "
            "breaking 50.87% of May 2023; Gangnam 37.7%, Songpa 39.4%, Yongsan "
            "39.7%, Seocho 41.6% — KB부동산 monthly housing time series via "
            "한국경제TV, 27 January 2026, "
            "<https://www.wowtv.co.kr/NewsCenter/News/Read?articleId=A202601270233>.",
            "Seoul villa (연립·다세대) jeonse-to-price ratio of 78.6% in December "
            "2022, 68.5% in December 2023 and 65.4% in December 2024 — 한국부동산원 "
            "임대차시장 사이렌 via 한경비즈니스, 27 January 2025, "
            "<https://magazine.hankyung.com/business/article/202501271517b>.",
            "Composite jeonse-to-price ratios of 65% nationwide, 57% for Seoul, 65% "
            "for Gyeonggi and 68% for Incheon as of August 2025 — KB부동산, quoted "
            "in iM증권, 'From jeonse to monthly rent', 8 September 2025, "
            "<https://www.imfnsec.com/upload/R_E09/2025/09/%5B08074049%5D_251607.pdf>.",
            "Auction clearing ratios (낙찰가율) for July 2026: nationwide "
            "apartments 85.4%, a 16-month low, and Seoul apartments 101.0%, a "
            "fourth consecutive month above 100%; Incheon 80.4%, Busan 78.8% — "
            "지지옥션 July 2026 auction report via 뉴스핌, 6 August 2026, "
            "<https://www.newspim.com/news/view/20260806000537>.",
            "Seoul villa auction clearing ratio of 79% against 102% for Seoul "
            "apartments, with non-redevelopment villas routinely failing three or "
            "four rounds and selling at 41-50% of appraised value — 지지옥션 via "
            "디지털타임스, <https://www.dt.co.kr/article/12030827>.",
            "Jeonse-to-monthly-rent conversion rate (전월세전환율) of 6.06% for "
            "Seoul in June 2026, the highest since the series began in January "
            "2018, and 6.62% nationwide; COFIX at 3.05% in June 2026 — "
            "한국부동산원 via 뉴데일리, 20 July 2026, "
            "<https://biz.newdaily.co.kr/site/data/html/2026/07/20/2026072000203.html>.",
            "Bank of Korea Base Rate raised 25bp to 2.75% on 16 July 2026 — Bank "
            "of Korea monetary policy decision, "
            "<https://www.bok.or.kr/eng/bbs/E0000634/view.do?nttId=11062944>.",
            "HUG 전세보증금반환보증 premium schedule of 0.097%-0.211% a year "
            "depending on term, building type and debt ratio, with an 80% debt "
            "ratio limit and apartment appraisal capped at 140% of market value — "
            "주택도시보증공사, "
            "<https://www.khug.or.kr/hug/web/ig/dr/igdr000001.jsp>.",
            "HUG guarantee incidents of 11조 441억원 across 50,941 cases in "
            "2020-2024, subrogation of 9조 4,189억원 across 43,631 cases, "
            "recoveries of 2조 3,458억원 for a 24% recovery rate; by building type "
            "apartments 13,251 cases / 3조 2,685억원, 다세대 24,870 cases / 5조 "
            "1,960억원, officetels 10,648 cases / 2조 1,802억원 — 뉴시스, 22 "
            "October 2025, <https://www.newsis.com/view/NISX20251022_0003373232>.",
            "HUG cumulative guarantee supply of 99조 3,914억원 for 2020-2023 and "
            "premium income of 3,525억원 from January 2020 to August 2024, "
            "described as 0.35% of the outstanding guarantee balance — 뉴스토마토, "
            "<https://www.newstomato.com/readnews.aspx?no=1241569>.",
            "Monthly rent at 54.1% of Seoul apartment rental transactions in June "
            "2026 against 45.9% for jeonse (8,819 against 7,477 deals) — Seoul "
            "Metropolitan Government via 파이낸셜뉴스, 22 July 2026, "
            "<https://www.fnnews.com/news/202607220822238116>.",
            "Seoul apartment jeonse listings down 32.8% year on year to 17,116 "
            "from 25,943, and villa jeonse prices up 0.44% in April 2026, the "
            "largest monthly rise in 12 years and 7 months — 한국부동산원 via "
            "헤럴드경제, <https://biz.heraldcorp.com/article/10763783>.",
            "Government announcement of 14 July 2026 on a public trust ('안심신탁') "
            "to hold jeonse deposits instead of landlords, with mechanism, "
            "participation and guarantee terms all still undecided — 한국경제, "
            "<https://www.hankyung.com/article/202607217859O>.",
            "Idiosyncratic volatility of individual house capital gains of roughly "
            "10-18% a year against 4-5% for metropolitan indices — Marco "
            "Giacoletti, 'Idiosyncratic Risk in Housing Markets', Review of "
            "Financial Studies 34(8), 2021, 3695-3741, "
            "<https://academic.oup.com/rfs/article-abstract/34/8/3695/6187964>.",
            "No price series is used or redistributed. Every input above is a "
            "published scalar; everything else in this post is closed form or "
            "fixed-seed simulation.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in qp.environment().items()
                if k in ("python", "numpy", "scipy", "quantpost")),
            "attachment": "the tenant receives min(D, max(0, lambda·V_T - M)), so "
                          "the tranche attaches at (M + D)/lambda — no volatility, "
                          "drift or horizon enters it",
            "bound": "every headline figure sets M = 0, i.e. assumes no mortgage "
                     "ranks ahead of the deposit, which makes the tenant's risk a "
                     "floor rather than an estimate",
            "loss": "expected shortfall integrated in closed form against a "
                    "lognormal terminal value over a two-year term, zero drift, "
                    f"volatility swept over {list(SIGMAS.values())}",
            "verification": (f"Monte Carlo, {N_SIM:,} draws per cohort; largest "
                             f"disagreement with the closed form "
                             f"{res['max_check_err_pp']:.4f} percentage points; the "
                             f"exact trigger and the modelled one are asserted equal "
                             f"at every cohort, which is what catches a unit slip "
                             f"between the two code paths"),
            "decomposition": (
                f"of the {res['decomposition']['total_pp']:.1f}pp gap between the "
                f"January 2026 Seoul apartment and December 2022 Seoul villa "
                f"triggers, the deposit ratio contributes "
                f"{res['decomposition']['deposit_ratio_pp']:.1f}pp and the auction "
                f"clearing ratio {res['decomposition']['auction_ratio_pp']:.1f}pp"),
            "cross_check": (f"the model's mispricing factor for the December 2022 "
                            f"Seoul villa cohort is {res['model_mispricing']:.0f}x; "
                            f"HUG's realised net loss ratio over 2020-2024 is "
                            f"{hug['loss_ratio']:.0f}x"),
        },
    )

    post.add("A loan everybody describes and nobody prices", f"""
Korea has a rental contract that exists nowhere else at scale. Under **jeonse** a
tenant hands the landlord a lump sum — commonly half to four fifths of what the home
is worth — lives there two years paying **no monthly rent at all**, and receives the
whole sum back at the end.

Every English explanation arrives at the same sentence: an interest-free loan from
tenant to landlord. Correct, and it is where they stop — which is a shame, because a
loan has a security structure and this one's decides everything.

Here is the tenant's payoff at the end of the term, with `D` the deposit, `M` any
mortgage ranking ahead of it, `V_T` the home's terminal value and `lambda` the fraction
of appraised value it fetches at a forced sale:

**min(D, max(0, lambda · V_T − M))**

Anyone who has looked at a securitisation has seen that expression. It is a
**first-loss tranche**. The tenant is not a customer paying rent but the most junior
creditor of a single-property loan, for an amount that is typically their entire net
worth. The rest of this post prices it.
""".strip())

    post.add("The one number that matters is arithmetic", f"""
Tranches have an **attachment point**: the collateral value at which they start taking
losses. Rearranging the payoff gives it immediately:

**V\\* = (M + D) / lambda**

Stare at that, because of what is *not* in it. No volatility, no expected return, no
horizon, no model — one published number divided by another, and Korea publishes both
monthly: 전세가율, deposit over sale price, and 낙찰가율, winning bid over appraised
value, from the court auction statistics.

Take the friendliest case, `M = 0` — the deposit is the only claim. Two cohorts from
one city then sit at opposite ends of the same contract.

**A Seoul apartment, January 2026.** Deposit ratio {RATIO_SEOUL_APT}%, an all-time
low since 2013 — not because deposits fell but because sale prices ran away from them.
Seoul apartments cleared {AUCTION_SEOUL_APT:.0f}% of appraisal at auction in July 2026,
a fourth straight month above par. Attachment point {apt['attachment']:.3f}: the home
must fall **{apt['required_fall_pct']:.1f}%** before one won is at risk.

**A Seoul villa, December 2022.** Villas — 연립·다세대, the low-rise walk-ups for
people priced out of apartments — had a ratio of {RATIO_SEOUL_VILLA_2022}%. Seoul
villas clear {AUCTION_SEOUL_VILLA:.0f}%, and the ones nobody wants go for
{AUCTION_VILLA_TAIL:.0f}% after four failed rounds. Attachment point
{villa22['attachment']:.3f}: the home must fall
**{villa22['required_fall_pct']:.1f}%**.

Half of one percent against fifty. Same contract, same city, same statute, same
courts.

Two things about that gap, because the obvious way to describe it is wrong twice
over. First, it is **two** ratios moving, not one: of the
{res['decomposition']['total_pp']:.0f} points, the deposit ratio accounts for
{res['decomposition']['deposit_ratio_pp']:.1f} and the auction clearing ratio for
{res['decomposition']['auction_ratio_pp']:.1f}. Move the deposit ratio alone and the
cushion goes to {res['decomposition']['deposit_ratio_only_pct']:.1f}%, not to half a
percent. Villas are worse because the deposit is bigger *and* because a court gets
less for them, in roughly equal measure.

Second, resist dividing. Fifty over a half is "a factor of a hundred" and it means
nothing: the trigger is exactly linear in the deposit ratio — every point of ratio
costs {res['decomposition']['slope_per_point']:.2f} points of cushion, at every ratio,
which is why Fig 1 is three straight lines — so a ratio of two of its values is large
only because one sits near zero. The honest statistic is the difference in points.
There *is* real convexity here; it is in the next section, and it is about the loss.

None of which makes this clever. It is division, worth writing down because nobody
does it and because no number matters more to a prospective tenant.
""".strip(), figures=[figs["trigger"]])

    post.add("Now the part that needs a model", f"""
Knowing the trigger is not knowing the risk. For that you need the chance of reaching
it, and there a model must be admitted. Mine is thin: lognormal terminal value, zero
drift, two-year term, one volatility parameter.

That parameter is the interesting choice. House price *indices* are quiet — Giacoletti
puts metropolitan index volatility at 4-5% a year. But the tenant does not own an index;
the tenant owns one building, and the same paper measures idiosyncratic volatility for
individual homes at **10-18% a year**. Pricing this off an index understates it three-
to four-fold.

So I swept it. Table 1 has every cohort; the two ends are what matter. At
{100 * SIGMA_BASE:.0f}% a year over a two-year term, the January 2026 Seoul apartment
breaches with probability {apt['p_breach_pct']:.2f}% and loses
{apt['loss_per_year_pct']:.4f}% of the deposit a year; the December 2022 Seoul villa
breaches with probability **{villa22['p_breach_pct']:.0f}%** and loses
**{villa22['loss_per_year_pct']:.2f}%**. Four thousandths of a basis point to three
hundred basis points — four orders of magnitude.

*This* is the non-linearity, and note where it is not. The trigger in Fig 1 is a
straight line; the loss in Fig 2 spans four orders of magnitude over the same range.
Convexity is a property of the option, not of the arithmetic — far out of the money a
tranche is nearly free, at the money nearly worthless, with no gentle middle. Which is
why "the deposit ratio crept up a bit" is not a mild sentence.

Every cohort, with the first three columns published and the last two modelled:

| {" | ".join(TABLE_HEADER)} |
|---|---|---|---|---|---|
{table_body}

(The integral is easy to get wrong, so every cohort is checked against a
{N_SIM:,}-draw Monte Carlo. Largest disagreement:
{res['max_check_err_pp']:.4f} percentage points.)
""".strip(), figures=[figs["loss"]])

    post.add("What the tenant is paid for carrying it", f"""
A junior creditor should be compensated, and this one is — in kind. The tenant lives
rent-free, and Korea publishes the market price of that swap: the 전월세전환율, the
rate at which a deposit converts into monthly rent. In June 2026 it was
**{CONVERSION_SEOUL}%** for Seoul and **{CONVERSION_NATIONWIDE}%** nationwide, a
record.

Subtract what the money would otherwise earn — the Base Rate is {BASE_RATE}% after
July's hike — and the {HUG_PREMIUM:.3f}% guarantee premium. What is left is
compensation for credit risk and nothing else:

**{res['spread']:.2f}% a year.**

(For contrast, the Housing Lease Protection Act caps this rate at the Base Rate plus
{STATUTORY_SPREAD:.0f} points — **{res['statutory_cap']:.2f}%** — in a formula that
never mentions the security structure. It binds only on conversions mid-tenancy, so
nothing is being broken.)

Set that against the expected losses. For the January 2026 Seoul apartment it covers
the loss **more than a thousand times over** — still
{res['robust']['Seoul apartment, Jan 2026']['coverage']:,.0f} times at the top of the
volatility range. For the December 2022 Seoul villa it covers the loss
**{villa22['coverage']:.1f} times**,
and {res['robust']['Seoul villa, Dec 2022']['coverage']:.1f} times at 18% — not at
all, since a first-loss tranche needs a risk premium *on top of* its expected loss.

A bond desk would say it this way. Given a loss-given-default of
{100 * res['lgd']:.0f}% — HUG recovers {hug['recovery_pct']:.1f}% on claims it takes
over — a spread of {res['spread']:.2f}% a year fairly compensates an annual default
probability of about **{res['breakeven_default_pct']:.1f}%**. That is a single-B
credit. Apartment tenants in Seoul are being paid single-B spreads to hold
something that, at a 51% attachment point, is closer to investment grade; villa
tenants in 2022 were paid the same spread to hold something well below it.

Which is the whole problem. The price was never wrong. **One price was quoted for two
completely different instruments.**
""".strip())

    post.add("Three ways to switch the mechanism off", f"""
A mechanism claim should come with a way to switch it off. This one has three.

**Take volatility to zero.** If prices never move, an out-of-the-money tranche never
breaches: the loss in Fig 2 is all option value. Except at the
December 2022 villa ratio, where the control cannot bite, because that cohort sat
{villa22['required_fall_pct']:.2f}% from its attachment point. Take volatility to
**0.2% a year** — two orders of magnitude below any housing market ever measured —
and the breach probability is still **{vc[0.002]['p_breach_pct']:.1f}%**, since half
a percent is under two standard deviations even then. At 2% it is
{vc[0.02]['p_breach_pct']:.0f}%, at 12% {vc[0.12]['p_breach_pct']:.0f}%, at 18%
{vc[0.18]['p_breach_pct']:.0f}%. So the control passes everywhere except at the ratio
actually being written, and there it says what the model cannot: the risk came not from
volatility but from signing half a percent from the edge.

**Take the liquidation haircut away.** Hold the ratio at {RATIO_SEOUL_VILLA_2022}%
and let the property sell at full appraised value. Expected loss falls from
{hc[AUCTION_SEOUL_VILLA]['loss_per_year_pct']:.2f}% a year to
{hc[100.0]['loss_per_year_pct']:.2f}% — a factor of
{res['haircut_factor']:.0f}.
This is the control that reallocates blame. The villa deposit crisis was reported as
a story about falling prices; most of the loss here is not the price falling but the
{res['haircut_gap_pp']:.0f}-point gap between a villa's appraisal and what a court
gets for it. Those call for different policies, and Korea has mostly debated the
first.

**Put a mortgage back in.** Every headline number above assumes none, which is why
they are floors. A senior lien takes the January 2026 Seoul apartment's
{apt['required_fall_pct']:.0f}% cushion to {mc[20.0]['required_fall_pct']:.0f}% at a
20% mortgage and {mc[40.0]['required_fall_pct']:.0f}% at 40%, with expected loss going
from {mc[0.0]['loss_per_year_pct']:.4f}% a year to
{mc[40.0]['loss_per_year_pct']:.2f}%. HUG capped the debt ratio at 80%.
""".strip())

    post.add("The crisis that arrived through the other door", f"""
Here is where my own model needs correcting, and the correction is the most useful
paragraph in the post.

Korea had a mass deposit-refund crisis in 2023 and 2024 without anything deserving the
word crash. For villas that fits Fig 1 — half a percent. But apartments also produced
tens of thousands of incidents, needing falls of tens of percent that never happened. A
collateral model cannot explain those, and I should not pretend it does.

There is a second mechanism, it is **linear**, and it fires first. When a contract rolls
over the landlord refunds `D` and collects a new deposit set by *today's* ratio and
price. The gap is cash he must find elsewhere.

Do that arithmetic on the published villa ratios with no price move at all. They went
from {RATIO_SEOUL_VILLA_2022}% in December 2022 to {RATIO_SEOUL_VILLA_2024}% in
December 2024, so a landlord refunding a completely unchanged house had to produce
**{res['gaps']['ratio to 65.4%, prices flat']:.1f}% of the deposit in cash**; with a
10% price fall, {res['gaps']['ratio to 65.4%, prices -10%']:.1f}%.

That is the shape of what happened. The collateral gap needs a large price move and is
an option; the funding gap needs no price move, is a straight line, and Korea generated
an enormous quantity of it. Whether it becomes a tenant's loss turns on something no
model here contains: whether the landlord had other money.

It is also why the falling ratio is not simply good news. Tenants repriced the tranche
by demanding a lower attachment point — the correct response — and every tenant who
did made every incumbent landlord's refund harder.
""".strip(), figures=[figs["rollover"]])

    post.add("An accidental out-of-sample test", f"""
I did not plan this next number, and it is why I trust the rest.

My model says the December 2022 Seoul villa cohort lost about
{villa22['loss_per_year_pct']:.2f}% of deposit a year. HUG — the state guarantor —
charged {HUG_PREMIUM:.3f}% a year to insure it: **{res['model_mispricing']:.0f} times
too cheap.**

Now the same question with no model involved. Over 2020-2024 HUG paid **9조
4,189억원** of subrogation on incidents totalling **11조 441억원** across
{HUG_INCIDENT_COUNT:,} cases and recovered **2조 3,458억원** — a
{hug['recovery_pct']:.1f}% recovery rate — for a net loss of
**{hug['net_loss'] / 10_000:.1f}조원**. Premiums over roughly the same window:
**3,525억원.**

A realised net loss ratio of **{hug['loss_ratio']:.0f}x premium.**

Two unrelated routes — a lognormal integral on two published ratios, and a public
guarantor's audited cash flows — reaching the same order of magnitude. The agreement
is partly luck: HUG's book is not made of December 2022 Seoul villas, its recovery is
recovery *from landlords* rather than from property liens, and a guarantee also covers
fraud, which my model does not price. I would not defend the two agreeing to within a
factor of two. But the direction and magnitude are not in question, and neither
calculation knew about the other.

The building-type split says it a third time: {HUG_BY_TYPE['villa (다세대)'][0]:,}
of the {HUG_INCIDENT_COUNT:,} incidents were villas against
{HUG_BY_TYPE['apartment'][0]:,} for apartments — {hug['villa_over_apartment']:.1f} to
one, in a country with far more apartments. That is where the attachment points were.
""".strip())

    post.add("What this means now", f"""
Jeonse is disappearing while I write this. In June 2026 monthly rent took
**{SEOUL_MONTHLY_SHARE}%** of Seoul apartment rental deals against
{SEOUL_JEONSE_SHARE}% for jeonse — overtaking the deposit system in the country's
central market — and on 14 July 2026 the government floated a public trust to hold
deposits instead of landlords.

That proposal is a clean statement of the problem in tranche terms: if a public body
holds the deposit and pays the landlord a yield, the attachment point stops existing.
It also means the landlord no longer receives a lump sum, which was the only reason to
offer jeonse — so the honest description is not "jeonse made safe" but "jeonse ended,
with a transition period". Whether that is good policy my arithmetic cannot say.

What it can say is smaller and more useful. Anyone signing a jeonse contract can
compute their own attachment point first, from numbers the state publishes for free:
deposit over sale price, plus any registered mortgage, divided by the auction clearing
ratio for that building type in that district. Ten seconds, for the only figure that
matters. Nobody puts it on the contract, and there is no reason they could not.
""".strip())

    post.add("Where this is a caricature", f"""
**Lognormal, zero drift, one volatility.** Real house prices are autocorrelated and
skewed, volatility clusters, and two-year windows in Korea have been anything but
drift-free. A jump or regime model would fatten the left tail and make every loss figure
larger — the direction that does not rescue the conclusion.

**Volatility does not scale the way I made it scale.** Giacoletti's other finding is
that idiosyncratic house risk barely grows with holding period while index risk does,
so my √T scaling understates one-year and overstates five-year risk. Small over the
two-year term; not to be pushed further unfixed.

**The auction ratio is not the tenant's recovery.** 낙찰가율 is the winning bid over
*appraised* value, and appraisals are stale, contested and — the villa fraud cases
turned on this — sometimes inflated on purpose. Real recovery also loses court costs,
arrears and any tax lien outranking the tenant. Every figure here is optimistic on
that axis.

**And the tenant's real problem is one I did not model at all.** A bond desk holding
this tranche would hold a hundred. A household holds one, funded with everything it
has, and cannot diversify, hedge or sell it. Expected loss is the least of it: what
matters is a {100 * res['lgd']:.0f}% loss of net worth in a single event, and no spread
I can compute makes that a reasonable position. That is an argument about position
sizing rather than pricing, and it is the strongest case against the instrument.
""".strip())
    return post
