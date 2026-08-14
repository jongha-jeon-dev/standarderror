"""exp010 — the safety margin on a Korean jeonse deposit, and how to compute it.

Backlog: Track C, numbers on society. Requested directly; it belongs here because
jeonse has an exactly computable core that nobody puts in front of tenants.

Under jeonse a tenant hands the landlord a lump sum — commonly half to four fifths
of what the home is worth — pays no monthly rent for two years, and receives the sum
back at the end. It is a secured loan, and secured loans have a **safety margin**:
how far the collateral can fall before the loan stops being fully covered. The
tenant's claim pays

    min(D, max(0, lambda * V_T - M))

for deposit `D`, mortgage `M` ranking ahead of it, terminal home value `V_T` and
liquidation ratio `lambda`, so it is fully covered above

    V* = (M + D) / lambda

which contains no volatility, no drift and no model. Two published Korean figures
divided, and the number appears on no contract. That is the post.

Two worked cases with `M = 0`:

* **Seoul apartments** at January 2026 ratios. Deposit ratio 50.92%, an all-time
  low; apartments clear 101% of appraisal at auction. Margin **49.6%**.
* **Seoul villas** at December 2022 ratios. Ratio 78.6%; villas clear 79%. Margin
  **0.51%**.

Two things an earlier draft of this got wrong, both worth keeping written down:

* That gap is **two** ratios moving, not one. Of the 49.1 points, the deposit ratio
  accounts for 27.4 and the auction clearing ratio for 21.7. Building type matters
  through both channels in roughly equal measure.
* The margin is **linear** in the deposit ratio — slope -1/lambda, which is why Fig 1
  is three straight lines. Quoting the two cases' ratio, "a factor of 98", was doubly
  wrong: a ratio of a linear function's values measures no curvature, and it is large
  only because one case sits near zero. The convexity here is real but it lives in
  the *loss*, not the margin.

An earlier draft also carried the analysis into a verdict — on the pricing of the
public deposit-return guarantee, and on the instrument itself. That was further than
this topic warrants from me: millions of households hold most of their net worth in
one of these deposits right now, and a post that reads as an argument against the
system is not what the arithmetic supports. The rewrite keeps every piece of the
analysis and moves the conclusion to what a reader can compute before signing.
Guarantee statistics stay, for scale, with the reasons they cannot be compared
against premium income.

Only the probability of breach needs a model, and the model is deliberately thin: a
lognormal terminal value, volatility swept across the range Giacoletti (2021)
measures for individual homes rather than indices, integrated in closed form and
checked against Monte Carlo.

Three controls, because a mechanical claim should be switchable:

1. **Volatility to zero.** Passes everywhere except at the December 2022 villa
   ratio, which sits 0.51% from its threshold: even at sigma = 0.2%/yr — two orders
   of magnitude below any measured housing market — the breach probability is 3.6%,
   and at sigma = 2% it is 43%. At that margin, volatility is not what decides the
   outcome.
2. **Liquidation haircut to one.** Holding the ratio at 78.6% and paying full
   appraised value cuts expected loss by a factor of ten, which moves the emphasis
   from market risk to appraisal and liquidation.
3. **The senior mortgage.** Every headline figure assumes none, so every margin here
   is an upper bound.

And the mechanism the margin misses: refunds became hard through **rollover**, not
collateral. Seoul villa deposit ratios fell from 78.6% to 65.4% between December 2022
and December 2024, so a landlord returning a deposit on a completely unchanged home
had to find 16.8% of it in cash. That path is linear, needs no price move, and is why
apartment cases with margins of tens of points still saw refund problems.

Nothing here redistributes a price series. Every input is a published scalar quoted
with its source in `data_sources`; everything else is closed form or fixed-seed
simulation.

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
        title="How far the home can fall before the deposit stops being covered",
        subtitle=("Deposit as the only claim on the property — no mortgage ahead "
                  "of it, which makes every line here the friendliest case. Each "
                  "curve is one type of property's auction clearing ratio."),
        ylabel="fall the deposit can absorb (%)",
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
        title="The margin is linear. The loss is not.",
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
        title="One contract, five cases, five different margins",
        subtitle=(f"No senior mortgage assumed, so every figure is a floor. Breach "
                  f"probability and coverage at {100 * SIGMA_BASE:.0f}%/yr "
                  f"volatility over a two-year term. Margins are upper bounds."),
        source=src_model, mode="light", bold_cols=(3, 5), align="lrrrrr",
        alt=("Table of five Korean rental cohorts. A Seoul apartment in January "
             "2026 needs a 50% fall to breach and the spread covers the expected "
             "loss more than a thousand times over; a Seoul villa in December 2022 "
             "needed a fall of half a percent and the spread covers the loss about "
             "once."),
        caption=("Table 1. The first three columns are published; the fourth is "
                 "arithmetic on them; the last two need a volatility assumption. "
                 "Read the margins as bounds — no senior mortgage is assumed."),
        path=str(IMG / f"c4-t1-cohorts.{EXT}"))
    figs["table"] = fig_meta

    # HERO — the finding is a comparison of two triggers.
    fig_meta, _ = charts.comparison_card(
        headline="How far can the home fall before a jeonse deposit stops being covered?",
        items=[(f"{apt['required_fall_pct']:.0f}%", "Seoul apartments, Jan 2026"),
               (f"{villa22['required_fall_pct']:.1f}%", "Seoul villas, Dec 2022")],
        note=("The deposit plus any registered mortgage, divided by what that kind of "
              "property fetches at a court auction. Both figures are published "
              "monthly. Neither is on the contract."),
        footer="quantpost", mode="light",
        alt=(f"Card comparing the price fall a deposit can absorb: "
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
        title="The Jeonse Number That Is Not on the Contract",
        slug="the-jeonse-number-not-on-the-contract",
        subtitle=("A deposit's safety margin is two published ratios divided — and "
                  "the interesting part is what that arithmetic cannot tell you"),
        summary=(
            f"Under Korea's jeonse system a tenant hands the landlord a lump sum "
            f"worth half to four fifths of the home, pays no monthly rent for two "
            f"years, and receives it back at the end. It is a secured loan, and "
            f"secured loans have a safety margin you can compute: how far the home "
            f"can fall before the deposit stops being fully covered. It is the "
            f"deposit plus any registered mortgage, divided by what that kind of "
            f"property fetches at a court auction — two figures Korea publishes "
            f"monthly, and a number that appears on no contract. For a Seoul "
            f"apartment at January 2026 ratios it is about "
            f"{apt['required_fall_pct']:.0f}%. For a Seoul villa at December 2022 "
            f"ratios it was under a point. This post works out how to compute it, "
            f"how much of it is arithmetic and how much is modelling, and the one "
            f"mechanism it misses entirely."),
        tags=["housing", "korea", "risk-management", "quantitative-finance",
              "data-science"],
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
            "guarantee_experience": (
                f"HUG's 2020-2024 figures are quoted for scale only: "
                f"{HUG_INCIDENT_COUNT:,} incidents, "
                f"{hug['recovery_pct']:.1f}% recovered on claims taken over. The "
                f"pool is not the cohorts modelled here, recovery from landlords is "
                f"not lien recovery, and fraud is not priced above, so no comparison "
                f"against premium income is drawn"),
        },
    )

    post.add("A contract with a number missing", f"""
Korea has a rental arrangement that exists nowhere else at scale. Under **jeonse**,
a tenant hands the landlord a lump sum — commonly half to four fifths of what the
home is worth — lives there for two years paying **no monthly rent at all**, and
receives the whole sum back at the end.

It is easy to call this strange and much harder to call it bad. For decades it did
several useful things at once: it turned savings into housing without a mortgage, it
forced saving in a way monthly rent does not, and it was the standard rung between
renting and owning. Millions of households hold most of their net worth in one right
now.

It is also, unavoidably, a **secured loan** from tenant to landlord — and secured
loans have a quantity any lender asks for before signing: the **safety margin**, how
far the collateral can fall before the loan stops being fully covered. Korea publishes
both numbers you need to compute it. It appears on no contract and in no listing.

This post works it out: what the margin is, how much is arithmetic and how much is
modelling, and the one mechanism it misses entirely.
""".strip())

    post.add("The margin is two published ratios divided", f"""
Write `D` for the deposit, `M` for any mortgage registered ahead of it, `V_T` for
what the home is worth when the lease ends, and `lambda` for the fraction of
appraised value that kind of property fetches if it has to be sold at a court
auction. The tenant's claim pays

**min(D, max(0, lambda · V_T − M))**

which is fully covered as long as the home is worth at least

**V\* = (M + D) / lambda**

Note what is *not* in that expression: no volatility, no expected return, no horizon,
no model. The deposit ratio is 전세가율 — deposit over sale price. The liquidation
ratio is 낙찰가율 — winning bid over appraised value, published monthly by building
type and district.

Two worked examples, with `M = 0` so nothing else is in the way:

**Seoul apartments at January 2026 ratios.** Deposit ratio {RATIO_SEOUL_APT}% — an
all-time low since the series began in 2013, not because deposits fell but because
sale prices ran ahead of them. Seoul apartments cleared
{AUCTION_SEOUL_APT:.0f}% of appraisal at auction in July 2026, a fourth straight
month above par. Margin: **{apt['required_fall_pct']:.1f}%**.

**Seoul villas at December 2022 ratios.** Villas — 연립·다세대, much of the
affordable rental stock — had a deposit ratio of {RATIO_SEOUL_VILLA_2022}%, and Seoul
villas clear {AUCTION_SEOUL_VILLA:.0f}% at auction. Margin:
**{villa22['required_fall_pct']:.1f}%**. By December 2024 the ratio had come down to
{RATIO_SEOUL_VILLA_2024}%, which puts the same calculation at
{villa24['required_fall_pct']:.1f}%.

Two things about the distance between those, because the obvious way to describe it
is wrong twice over.

First, it is **two** ratios moving, not one. Of the
{res['decomposition']['total_pp']:.0f} points, the deposit ratio accounts for
{res['decomposition']['deposit_ratio_pp']:.1f} and the auction clearing ratio for
{res['decomposition']['auction_ratio_pp']:.1f}. Move the deposit ratio alone and the
margin goes to {res['decomposition']['deposit_ratio_only_pct']:.1f}%, not to under a
point. Building type matters through both channels, in roughly equal measure.

Second, resist dividing. Fifty over a half is "a factor of a hundred" and means nothing
here: the margin is exactly **linear** in the deposit ratio — every point of ratio costs
{res['decomposition']['slope_per_point']:.2f} points of margin, at every ratio, which is
why Fig 1 is three straight lines — so a ratio of two of its values is large only because
one sits near zero. The honest statistic is the difference in points.

None of which makes this clever. It is division, worth doing because the answer is
specific to your building and nobody hands it to you.
""".strip(), figures=[figs["trigger"]])

    post.add("The margin is not the risk", f"""
Knowing how far the home can fall is not the same as knowing how likely that is,
and for the second question a model has to be admitted. Mine is thin on purpose:
lognormal terminal value, zero drift, two-year term, one volatility parameter.

That parameter is the interesting choice. House price *indices* are quiet —
Giacoletti's work on repeat sales puts metropolitan index volatility at 4-5% a year.
But a tenant is exposed to one building, not an index, and the same paper measures
idiosyncratic volatility for individual homes at **10-18% a year**. Reading this risk
off an index number understates it three- to fourfold, which is probably the most
transferable point here.

At {100 * SIGMA_BASE:.0f}% a year over a two-year term, the January 2026 Seoul
apartment case breaches its margin with probability {apt['p_breach_pct']:.2f}% and
loses {apt['loss_per_year_pct']:.4f}% of the deposit a year in expectation; the
December 2022 Seoul villa case, {villa22['p_breach_pct']:.0f}% and
{villa22['loss_per_year_pct']:.2f}%.

Here the two halves of the calculation behave completely differently, and it is worth
being precise. **The margin is linear. The loss is convex.** Fig 1 is a straight line;
Fig 2 spans four orders of magnitude across the same range. That convexity is a
property of the payoff, not of my parameters: a claim far below its threshold is nearly
riskless, one sitting at its threshold is nearly all risk, with no gentle middle. Which
is why "the deposit ratio went up a few points" is not a mild sentence, even though the
margin moved only a few points too.

Every case, side by side — the first three columns published or arithmetic, the last
two modelled:

| {" | ".join(TABLE_HEADER)} |
|---|---|---|---|---|---|
{table_body}

(The integral is easy to get wrong, so every row is checked against a
{N_SIM:,}-draw Monte Carlo. Largest disagreement:
{res['max_check_err_pp']:.4f} percentage points. The margin is also computed twice
by different code paths and asserted equal, which is how I caught a unit slip that
had already shipped one broken chart.)
""".strip(), figures=[figs["loss"]])

    post.add("What the deposit earns", f"""
The tenant is not lending for nothing. The return is paid in kind — a home occupied
rent-free — and Korea publishes the market price of that swap: the 전월세전환율, the
rate at which a deposit converts into monthly rent. In
June 2026 it was **{CONVERSION_SEOUL}%** for Seoul and **{CONVERSION_NATIONWIDE}%**
nationwide, a record for the series.

Net of what the money would otherwise earn — the Base Rate is {BASE_RATE}% after
July's increase — that is roughly **{res['spread'] + HUG_PREMIUM:.1f}% a year**, and
about {res['spread']:.1f}% after a deposit-return guarantee premium of
{HUG_PREMIUM:.3f}%. For readers who think in credit terms: at the loss rates above,
that is a comfortable multiple of expected loss in the apartment case and roughly
one times it in the December 2022 villa case.

The observation I would draw is narrow, and I want to keep it narrow. The conversion
rate is essentially **one national number**; the margin is **building-specific**, and
across Table 1 it runs from fifty points to under one. That is not evidence anyone was
cheated — deposit ratios are negotiated, tenants have preferences over building and
location, and much besides risk goes into what a home rents for. It does mean the
price is not where you look to find out how safe a deposit is. The margin is, and the
margin is free.

As for what margin is reasonable, the market has a rough convention: analysts describe
deposit ratios of **60-70%** as the band where jeonse and sale prices sit in stable
balance. Table 1 lets you turn any ratio into a breach probability at a volatility you
choose, which is more use than a band.
""".strip())

    post.add("Three ways to check the arithmetic", f"""
A claim built on a mechanism should come with ways to switch the mechanism off.
This one has three, and the second changed what I thought the answer was.

**Take volatility to zero.** If prices never move, a claim below its threshold never
breaches, so the loss in Fig 2 is all option value. The exception is the December 2022
villa ratio, where the control cannot bite: that case sat
{villa22['required_fall_pct']:.2f}% from its threshold, and at a volatility of
**0.2% a year** — two orders of magnitude below any housing market ever measured —
the breach probability is still **{vc[0.002]['p_breach_pct']:.1f}%**, because half a
point is under two standard deviations even then. At 2% it is
{vc[0.02]['p_breach_pct']:.0f}%, at 12% {vc[0.12]['p_breach_pct']:.0f}%, at 18%
{vc[0.18]['p_breach_pct']:.0f}%. So the control passes everywhere except at the
ratio that mattered most, and there it says something the model cannot: at that
margin, volatility is not what determines the outcome.

**Take the liquidation haircut away.** Hold the deposit ratio at
{RATIO_SEOUL_VILLA_2022}% and let the property sell at full appraised value instead
of {AUCTION_SEOUL_VILLA:.0f}%. Expected loss falls from
{hc[AUCTION_SEOUL_VILLA]['loss_per_year_pct']:.2f}% a year to
{hc[100.0]['loss_per_year_pct']:.2f}% — a factor of {res['haircut_factor']:.0f}.
I did not expect that, and it moves the emphasis. Most of the modelled loss is not
the price falling; it is the {res['haircut_gap_pp']:.0f}-point gap between what a
villa is appraised at and what a court realises for it. Appraisal and liquidation
are a different problem from market risk, and the second is the one that gets
discussed.

**Put a mortgage back in.** Every figure above assumes none, which makes them upper
bounds rather than estimates. A senior lien takes the January 2026 Seoul apartment case
from {apt['required_fall_pct']:.0f}% to {mc[20.0]['required_fall_pct']:.0f}% at a 20%
mortgage and {mc[40.0]['required_fall_pct']:.0f}% at 40% — the single biggest reason to
run this on your own building rather than a district average. The registered mortgage is
on the 등기부등본, and it is yours to look up.
""".strip())

    post.add("The mechanism the margin misses", f"""
Now the part where the arithmetic above is not enough, which is the most useful section
here.

Korea had a great deal of difficulty with deposit refunds in 2023 and 2024 without
anything deserving the word crash. For villas that fits Fig 1 — the margin was under a
point. But apartments, whose margins were tens of points, also saw many refund
problems, and the collateral arithmetic cannot explain those. I should not pretend it
does.

There is a second mechanism, it is **linear**, and it arrives first. When a lease
rolls over the landlord returns `D` and collects a new deposit set by *today's*
ratio and *today's* price. Any difference is cash that has to come from somewhere
else, and it has nothing to do with whether the collateral covers the claim.

Do that arithmetic on the published villa ratios with no price move at all. They went
from {RATIO_SEOUL_VILLA_2022}% to {RATIO_SEOUL_VILLA_2024}% between December 2022 and
December 2024, so a landlord returning a deposit on a completely unchanged home had to
produce **{res['gaps']['ratio to 65.4%, prices flat']:.1f}% of it in cash**; with a 10%
price fall, {res['gaps']['ratio to 65.4%, prices -10%']:.1f}%.

That is the shape of it. The collateral gap needs a large price move and is convex;
the funding gap needs no price move, is a straight line, and there was a great deal of
it. Whether a funding gap becomes a tenant's loss depends on something no model here
contains — whether the landlord could raise the difference — and in most cases they
could.

It also cuts against reading the falling ratio as unambiguously good. Asking for a
larger margin is the right response to a convex risk, and every tenant who did made
the incumbent landlord's refund arithmetic harder. Both are true at once, which is
usually a sign a single number is being asked to carry a judgement it cannot.
""".strip(), figures=[figs["rollover"]])

    post.add("Where this is a caricature", f"""
**Lognormal, zero drift, one volatility.** Real house prices are autocorrelated and
skewed, volatility clusters, and two-year windows in Korea have been anything but
drift-free in either direction. A jump or regime model would fatten the left tail and
make every loss figure larger, which is the direction that does not rescue the
conclusion.

**Volatility does not scale the way I made it scale.** Giacoletti's other finding is
that idiosyncratic house risk barely grows with holding period while index risk does,
so my √T scaling understates one-year and overstates five-year risk. Small over the
two-year term; not to be pushed further unfixed.

**The auction ratio is not the tenant's recovery.** 낙찰가율 is the winning bid over
*appraised* value, and appraisals are stale, contested and occasionally wrong. Real
recovery also loses court costs, arrears and any tax lien that outranks the tenant,
and it takes time. Every figure here is optimistic on that axis, which is another
reason to treat the margin as a bound rather than a promise.

**The loss numbers are the least useful part.** An investor holding this kind of claim
would hold a hundred and care about expected loss, because a portfolio average is what
a portfolio delivers. A household holds one, and an average is not what it experiences.
That is a fact about position size rather than pricing, and no expected-loss figure
captures it — which is exactly why the number I would want before signing is the margin
in Fig 1, not anything in Fig 2. It is exact, it is specific to the building, and it
answers the question a household actually has.

**Guarantee statistics, for scale only.** HUG's published figures: 11조 441억원 of
deposit-return guarantee incidents across {HUG_INCIDENT_COUNT:,} cases over 2020-2024,
{HUG_BY_TYPE['villa (다세대)'][0]:,} of them 다세대 against
{HUG_BY_TYPE['apartment'][0]:,} apartments, {hug['recovery_pct']:.0f}% recovered on
claims taken over. Setting that against premium income produces a large ratio and an
earlier draft made something of it, but the pool is not the cohorts modelled here,
recovery from landlords is not lien recovery, and a guarantee covers fraud, which
nothing above prices. A sizing fact, not a verdict.
""".strip())

    post.add("The ten-second version", f"""
If you take one thing from this, take the recipe.

**(deposit + any registered mortgage) ÷ the auction clearing ratio for that building
type and district.** Subtract from one. That is how much the home can lose before your
deposit stops being fully covered.

Every input is public and free. The deposit is on your contract, the registered
mortgage on the 등기부등본, the auction clearing ratio published monthly by building
type and district. And the answer differs enormously between two buildings that would
look identical in a listing, which is the whole reason it is worth the thirty seconds.

Context, stated flatly because it is not my place to draw a conclusion from it. Jeonse's
share is falling: in June 2026 monthly rent took {SEOUL_MONTHLY_SHARE}% of Seoul
apartment rental transactions against {SEOUL_JEONSE_SHARE}% for jeonse, and listings
were down {SEOUL_LISTINGS_FALL}% year on year. In July 2026 the government said it
would develop a proposal for a public body to hold deposits rather than landlords, with
mechanism, participation and guarantee terms all undecided.

Whether any of that is good policy is a question my arithmetic has no standing to
answer. What the arithmetic can do is put one specific, checkable number in front of
someone before they sign — and there is no reason that number could not simply be
printed on the contract.
""".strip())

    return post
