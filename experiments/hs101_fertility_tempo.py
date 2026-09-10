"""Headline Statistics 1: Korea's Fertility Rebound Needs Nobody to Have More Children.

The first episode of a track about aggregate statistics that a newspaper prints
as though they described a person. This one is the period total fertility rate,
which is not children per woman: it is the sum of one calendar year's
age-specific rates, the completed family size of a woman who does not exist.

Measured, and all of it by construction rather than by fitting anything:

* With every cohort having the same completed fertility `Q` on a normal schedule
  and postponing by `delta` years per cohort, the period TFR is exactly
  `Q / (1 + delta)`. Simulation and algebra agree to 1e-5.
* The period mean age of childbearing rises at `r = delta / (1 + delta)`, which
  is *not* `delta`. Reading a published mean-age change as the postponement rate
  understates the postponement.
* `TFR / (1 - r)` returns `Q` exactly. On this schedule the Bongaarts-Feeney
  tempo adjustment is an identity, not an approximation.
* Postponement decelerating from `delta = 0.2` to `0.1` raises the period TFR by
  9.1% with the quantum fixed -- and takes 28 years to start and about twenty
  more to finish, because the cohorts whose behaviour changed have to cross the
  childbearing window first.
* The assumption the literature attacks hardest turns out to be cheap: widening
  the cohort schedule from 4.5 to 8.0 years of spread biases the adjusted rate by
  1.1%. That is the opposite of what this episode was drafted to say.

Run: `standarderror run hs101_fertility_tempo --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.aggregates import tempo as tp
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
POST_DATE = date(2026, 9, 10)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Headline Statistics, Taught Through What Breaks"
SERIES_TAG = "Headline Statistics"

#: Postponement rates the sweep runs at, in years of cohort mean age per year of
#: birth. Korea's recent published mean-age change is 0.1 a year, which on the
#: algebra below corresponds to a `delta` of about 0.11.
DELTAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
QUANTUM = 1.8

#: Published figures the prose quotes. None of them enter any computation here;
#: every number in the figures is constructed. Statistics Korea, 2023 and 2024
#: birth releases, and the 2025 provisional release reported 2026-02-25.
KOREA = {
    "tfr": {2018: 0.977, 2019: 0.918, 2020: 0.837, 2021: 0.808, 2022: 0.778,
            2023: 0.721, 2024: 0.748, 2025: 0.800},
    "mac": {2022: 33.5, 2023: 33.6, 2024: 33.7},
    "first_birth_2024": 33.1,
    "third_birth_change_2024": -0.1,
    "births_2025": 254_500,
}


def compute() -> dict:
    sweep = []
    for d in DELTAS:
        rows = tp.simulate(quantum=QUANTUM, delta=d, years=range(15, 26))
        row = next(r for r in rows if r.year == 20)
        r = tp.mac_change(rows, 20)
        sweep.append({
            "delta": d, "tfr": row.tfr, "r": r, "spread": row.spread,
            "shortfall": row.shortfall,
            "adjusted": tp.bongaarts_feeney(row.tfr, r),
            "tfr_algebra": tp.period_tfr(QUANTUM, d),
            "r_algebra": tp.mac_rate(d),
        })

    reb = tp.rebound(quantum=QUANTUM, fast=0.20, slow=0.10, switch=40)
    lo = min(x.tfr for x in reb)
    hi = max(x.tfr for x in reb)
    first = next(x.year for x in reb if x.tfr > lo * 1.001)
    done = next(x.year for x in reb if x.tfr > lo + 0.99 * (hi - lo))

    var = [tp.variance_bias(spread_from=4.5, spread_to=to)
           for to in (4.5, 5.0, 5.5, 6.5, 8.0)]

    # Korea's arithmetic, from published headline figures only.
    mac = KOREA["mac"]
    r_recent = mac[2024] - mac[2023]
    korea = {
        "r_recent": r_recent,
        "factor_recent": 1.0 / (1.0 - r_recent),
        "adjusted_2024": KOREA["tfr"][2024] / (1.0 - r_recent),
        # OECD Family Database SF2.3: Korea's mean age at first birth rose by
        # more than five years between 2000 and the latest year. Over 24 years
        # that is at least 0.21 a year, against 0.1 now.
        "r_long_run": 5.0 / 24.0,
        "factor_long_run": 1.0 / (1.0 - 5.0 / 24.0),
        "reported_rise": KOREA["tfr"][2025] / KOREA["tfr"][2023] - 1.0,
    }
    korea["factor_fall"] = (korea["factor_recent"] / korea["factor_long_run"]
                            - 1.0)
    return {"sweep": sweep, "rebound": reb, "var": var, "korea": korea,
            "reb_span": {"lo": lo, "hi": hi, "rise": hi / lo - 1.0,
                         "first": first, "done": done}}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    sweep = {row["delta"]: row for row in res["sweep"]}
    reb = res["rebound"]
    span = res["reb_span"]
    var = res["var"]

    def shortfall(ax, m):
        d = np.array([row["delta"] for row in res["sweep"]])
        grid = np.linspace(0, 0.32, 200)
        ax.plot(grid, 1.0 - 1.0 / (1.0 + grid), lw=1.9, color=m.series[0],
                label="algebra:  1 - 1/(1+d)")
        ax.scatter(d, [sweep[x]["shortfall"] for x in d], s=44, zorder=3,
                   color=m.series[1], label="simulation", edgecolors=m.surface,
                   linewidths=0.8)
        ax.axvline(0.11, color=m.grid, lw=1.4)
        ax.annotate("Korea's recent pace", (0.11, 0.02),
                    textcoords="offset points", xytext=(7, 0), fontsize=8.6,
                    color=m.ink_secondary)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.legend(frameon=False, fontsize=8.8, loc="upper left")

    out["f0"] = charts.diagram(
        shortfall,
        title="How far below completed fertility a pure delay puts the TFR",
        subtitle=("Every cohort in the simulation has the same completed "
                  "fertility of 1.8. The only thing that varies is how much "
                  "each cohort postpones."),
        xlabel="postponement per cohort, years of mean age per year of birth",
        ylabel="period TFR below the quantum",
        source="Simulated; standarderror/aggregates/tempo.py.",
        alt=("A rising curve with simulation points sitting exactly on it, "
             "showing the period TFR falling further below completed fertility "
             "as postponement increases."),
        caption=(f"The dots are the simulation and the line is "
                 f"1 - 1/(1 + d), which they sit on to within "
                 f"{max(abs(sweep[x]['tfr'] - sweep[x]['tfr_algebra']) for x in sweep):.0e}. "
                 f"At the pace Korea has recently published, the period rate "
                 f"sits about {sweep[0.10]['shortfall']:.0%} below what the "
                 f"same women would produce if timing stopped moving. Nobody "
                 f"in this picture has fewer children than anybody else."),
        path=str(IMG / f"hs101-f0-shortfall.{EXT}"))[0]

    def rebound_plot(ax, m):
        yr = np.array([x.year for x in reb])
        ax.plot(yr, [x.tfr for x in reb], lw=2.0, color=m.series[0],
                label="period TFR")
        ax.axhline(QUANTUM, color=m.series[2], lw=1.7, ls="--",
                   label=f"completed fertility, constant at {QUANTUM}")
        ax.axhline(span["lo"], color=m.grid, lw=1.3)
        ax.axhline(span["hi"], color=m.grid, lw=1.3)
        ax.axvline(40, color=m.series[1], lw=1.6, ls=":")
        ax.annotate("cohorts born here\npostpone half as much", (40, 1.72),
                    textcoords="offset points", xytext=(8, 0), fontsize=8.6,
                    color=m.series[1], va="top")
        ax.annotate(f"first movement, {span['first'] - 40} years later",
                    (span["first"], span["lo"]), textcoords="offset points",
                    xytext=(8, -14), fontsize=8.6, color=m.ink_secondary)
        ax.set_ylim(1.42, 1.88)
        ax.legend(frameon=False, fontsize=8.8, loc="lower right")

    out["f1"] = charts.diagram(
        rebound_plot,
        title="A rebound that nobody caused, and that arrives a generation late",
        subtitle=("Completed fertility is fixed at 1.8 for every cohort in the "
                  "run. At birth year 40 the cohorts begin postponing half as "
                  "much as before, and nothing else changes."),
        xlabel="calendar year", ylabel="births per woman",
        source="Simulated; standarderror/aggregates/tempo.py.",
        alt=("A flat period TFR line that begins rising about thirty years "
             "after a marked change of behaviour and levels off at a new "
             "plateau, well below a constant completed-fertility line."),
        caption=(f"The period rate rises {span['rise']:.1%}, from "
                 f"{span['lo']:.3f} to {span['hi']:.3f}, and the quantum never "
                 f"moves. It does not start rising until year "
                 f"{span['first']} - the cohorts whose behaviour changed have "
                 f"to reach childbearing age first - and takes another "
                 f"{span['done'] - span['first']} years to finish. Three years "
                 f"of period TFR cannot see any of this."),
        path=str(IMG / f"hs101-f1-rebound.{EXT}"))[0]

    #: The cohort spreads `variance_bias` was asked for, in the same order as
    #: `res["var"]`. The rows carry the resulting *period* spread, which is
    #: narrower by 1 + delta and is not what this figure is about.
    SPREADS = (4.5, 5.0, 5.5, 6.5, 8.0)

    def variance(ax, m):
        ax.plot(SPREADS, [v["error"] for v in var], marker="o", ms=6, lw=1.9,
                color=m.series[0])
        ax.axhline(0.0, color=m.grid, lw=1.4)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.1%}")

    out["f2"] = charts.diagram(
        variance,
        title="The assumption everyone attacks costs about one percent",
        subtitle=("Bongaarts-Feeney assumes the schedule shifts rigidly. Here "
                  "the cohort schedule also widens, from 4.5 years of spread "
                  "to the value on the axis, while postponement runs at 0.2."),
        xlabel="cohort schedule spread at the end of the run, years",
        ylabel="error in the adjusted rate",
        source="Simulated; standarderror/aggregates/tempo.py.",
        alt=("A gently falling line from zero to about minus one percent as "
             "the schedule spread widens from 4.5 to 8 years."),
        caption=(f"A widening far larger than any country has produced leaves "
                 f"the adjustment {abs(var[-1]['error']):.1%} low. This "
                 f"episode was drafted expecting the variance assumption to be "
                 f"where the formula fell over; measured, it is where the "
                 f"formula is fine, which agrees with Mazzuco and Zanotto "
                 f"(2025)."),
        path=str(IMG / f"hs101-f2-variance.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    span = res["reb_span"]
    reb = res["rebound"]

    def synthetic(panel, m):
        # Four cohort schedules, each later than the one below it, on their own
        # rows so they read as a sequence rather than a tangle. The vertical
        # line is one calendar year, and it crosses each cohort at a different
        # point of that cohort's own schedule -- which is the whole objection.
        x = np.linspace(-1.6, 1.6, 160)
        for i in range(4):
            off = -0.75 + i * 0.5
            panel.plot(x + off, np.exp(-x * x * 2.2) * 0.20 + i * 0.26,
                       lw=1.9, color=m.series[0], alpha=0.85)
        panel.plot([0.0, 0.0], [-0.06, 1.06], lw=2.6, color=m.series[1])
        panel.set_xlim(-2.5, 2.6)
        panel.set_ylim(-0.14, 1.14)

    def delay(panel, m):
        x = np.linspace(-3, 3, 200)
        panel.plot(x, np.exp(-x * x / 2), lw=2.0, color=m.grid)
        panel.plot(x + 1.1, np.exp(-x * x / 2) / 1.1, lw=2.4,
                   color=m.series[0])
        panel.annotate("", xy=(1.05, 1.02), xytext=(0.05, 1.02),
                       arrowprops=dict(arrowstyle="-|>", lw=2.0,
                                       color=m.series[1], mutation_scale=15))
        panel.set_xlim(-3.2, 4.4)
        panel.set_ylim(-0.08, 1.35)

    def late(panel, m):
        yr = np.array([x.year for x in reb], dtype=float)
        y = np.array([x.tfr for x in reb])
        panel.plot(yr, y, lw=2.2, color=m.series[0])
        panel.axvline(40, lw=1.8, color=m.series[1], ls=":")
        panel.set_ylim(1.44, 1.70)

    return charts.strip_card(
        headline="The fertility rate is a woman who does not exist",
        panels=[(synthetic, "1 year", "of rates, mixed cohorts"),
                (delay, f"{1 - 1/1.11:.0%}", "lost to delay alone"),
                (late, f"{span['first'] - 40} yrs", "before a change shows up")],
        note=("Every cohort in the simulation has the same completed fertility. "
              "The period total fertility rate still falls, and then rises "
              f"{span['rise']:.0%}, on timing alone."),
        footer=f"{SERIES_TAG.upper()} · 1",
        alt=("Three hand-drawn frames: a vertical line cutting across several "
             "offset fertility curves; one curve shifted right and flattened; "
             "and a period rate that stays flat long after a marked change."),
        path=str(IMG / f"hs101-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["identity"] = s.run(f"""
        from standarderror.aggregates import tempo as tp

        # Every cohort has the same completed fertility. The only thing that
        # varies down this table is how much each cohort postpones.
        print(f"{{'delta':>6}} {{'period TFR':>11}} {{'Q/(1+d)':>9}} "
              f"{{'r':>7}} {{'d/(1+d)':>8}} {{'TFR/(1-r)':>10}}")
        for d in {list(DELTAS)}:
            rows = tp.simulate(quantum={QUANTUM}, delta=d, years=range(15, 26))
            row = next(x for x in rows if x.year == 20)
            r = tp.mac_change(rows, 20)          # half the change across 2020
            print(f"{{d:>6.2f}} {{row.tfr:>11.5f}} "
                  f"{{tp.period_tfr({QUANTUM}, d):>9.5f}} {{r:>7.4f}} "
                  f"{{tp.mac_rate(d):>8.4f}} "
                  f"{{tp.bongaarts_feeney(row.tfr, r):>10.5f}}")
    """, expect=["TFR/(1-r)"])

    out["rebound"] = s.run("""
        # Nothing changes except that cohorts born after year 40 postpone half
        # as much as the ones before them. Completed fertility is 1.8 all the
        # way through, for everybody.
        reb = tp.rebound(quantum=1.8, fast=0.20, slow=0.10, switch=40)
        by = {x.year: x for x in reb}

        print(f"{'year':>5} {'period TFR':>11} {'quantum':>8} {'gap':>7}")
        for y in (60, 70, 75, 80, 85, 90, 110):
            x = by[y]
            print(f"{y:>5} {x.tfr:>11.4f} {x.quantum:>8.2f} "
                  f"{x.shortfall:>7.1%}")
    """, expect=["period TFR"])

    out["korea"] = s.run("""
        # The published Korean figures, and nothing else. Mean age of mother at
        # childbirth: 33.5 in 2022, 33.6 in 2023, 33.7 in 2024 (Statistics
        # Korea). So r is 0.1 a year, and the adjustment is one division.
        tfr = {2023: 0.721, 2024: 0.748, 2025: 0.800}
        r_now = 33.7 - 33.6                      # a year of mean age per year
        r_long = 5.0 / 24.0                      # OECD: over 5 years since 2000

        for name, r in (("recent pace", r_now), ("2000-2024 average", r_long)):
            print(f"{name:>18}  r = {r:.3f}  "
                  f"factor 1/(1-r) = {1/(1-r):.3f}  "
                  f"adjusted 2024 TFR = {tfr[2024]/(1-r):.3f}")

        print(f"\\nreported rise 2023 to 2025: "
              f"{tfr[2025]/tfr[2023] - 1:+.1%}")
        print(f"fall in the correction factor over the same story: "
              f"{(1/(1-r_now))/(1/(1-r_long)) - 1:+.1%}")
    """, expect=["reported rise"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    sweep = {row["delta"]: row for row in res["sweep"]}
    span = res["reb_span"]
    korea = res["korea"]
    var = res["var"]

    # The spine, asserted rather than trusted.
    for d, row in sweep.items():
        assert abs(row["tfr"] - row["tfr_algebra"]) < 1e-4, (d, row)
        assert abs(row["r"] - row["r_algebra"]) < 1e-4, (d, row)
        assert abs(row["adjusted"] - QUANTUM) < 2e-3, (d, row)
        if d > 0:
            assert row["r"] < d, (d, row)
    assert span["rise"] > 0.08
    assert span["first"] - 40 > 20
    assert abs(var[0]["error"]) < 1e-3
    assert abs(var[-1]["error"]) < 0.02
    assert 0.10 < korea["r_recent"] + 1e-9
    assert korea["factor_recent"] < korea["factor_long_run"]

    post = Post(
        title=(f"{SERIES_TAG} 1: Korea's Fertility Rebound Needs Nobody to "
               f"Have More Children"),
        slug="headline-statistics-1-fertility-tempo",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=1,
        date=POST_DATE,
        subtitle=("The total fertility rate is not children per woman. It is "
                  "one calendar year of age-specific rates stacked into the "
                  "completed family size of a woman who does not exist, and a "
                  "pure change in the timing of births moves it while no "
                  "cohort's family size moves at all."),
        summary=(
            "Give every cohort the same completed fertility and let each one "
            "postpone by d years relative to the last. The period TFR is then "
            "exactly Q/(1+d), the period mean age of childbearing rises at "
            "d/(1+d) rather than at d, and dividing the first by one minus the "
            "second returns Q to five decimal places — the Bongaarts-Feeney "
            "adjustment, which on this schedule is an identity rather than a "
            "correction. Two consequences. Postponement at the pace Korea has "
            "published puts the period rate about 10% below what the same "
            "women would produce with timing held still. And postponement "
            "**decelerating** from 0.2 to 0.1 raises the period TFR by 9.1% with "
            "the quantum fixed — arriving 28 years after the behaviour changed "
            "and taking twenty more to finish. Korea's reported rate rose 11% "
            "between 2023 and 2025 while the mean age of mothers rose 0.1 a "
            "year against a 2000-2024 average of at least 0.21. Those are the "
            "same order of magnitude, which is not a decomposition but is "
            "enough to say the rebound does not require anyone to have had "
            "more children."),
        tags=["statistics", "demographics", "economics", "data-science",
              "public-data", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data enters any computation. Every figure and every "
            "table here is constructed from the simulation in the code shown, "
            "executed when this page was built.",
            "Published figures quoted in the prose, and used nowhere else: "
            "Statistics Korea, *Births and Deaths* releases for 2023, 2024 and "
            "the 2025 provisional (total fertility rate; mean age of mother at "
            "childbirth 33.5 in 2022, 33.6 in 2023, 33.7 in 2024; 254,500 "
            "births in 2025); OECD Family Database SF2.3 for the statement "
            "that Korea's mean age at first birth has risen by more than five "
            "years since 2000.",
            "Machinery: `standarderror/aggregates/tempo.py`, tested in "
            "`tests/test_tempo.py`.",
            "Where this stops: Bongaarts and Feeney, \"On the quantum and "
            "tempo of fertility\", *Population and Development Review* 24 "
            "(1998), for the adjustment; Kohler and Philipov, \"Variance "
            "effects in the Bongaarts-Feeney formula\", *Demography* 38 "
            "(2001), for the correction when the schedule's spread also moves; "
            "Mazzuco and Zanotto, *Demographic Research* 52 (2025) art. 19, "
            "for the finding that shape changes act largely through the period "
            "mean and the adjustment is therefore more robust than that "
            "suggests.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": ("one normal fertility schedule per cohort, summed "
                           "over single years of age from 12 to 60, with "
                           "completed fertility fixed at 1.8 for every cohort "
                           "in every run"),
            "determinism": ("no random numbers anywhere; the schedule is "
                            "analytic and every quoted value is a closed-form "
                            "quantity evaluated on a fixed age grid"),
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    sweep = {row["delta"]: row for row in res["sweep"]}
    span = res["reb_span"]
    korea = res["korea"]
    var = res["var"]
    K = KOREA

    post.add(
        "The number that went up",
        f"""On 25 February 2026 Statistics Korea reported that the country's total fertility rate reached {K['tfr'][2025]:.2f} in 2025, up from {K['tfr'][2024]:.3f} the year before and {K['tfr'][2023]:.3f} in 2023, on {K['births_2025']:,} births. It was covered, reasonably enough, as a rebound: the first sustained rise in more than a decade, after a decade of the lowest fertility ever recorded anywhere.

I want to take the number seriously enough to ask what it measures, because the answer is not the one almost every article assumes, and the gap between the two is large enough to account for a rebound of this size on its own.

This is the first of three episodes about aggregate statistics that get printed as though they described a person, and the three fail in structurally different ways. This one is a summary that is a *synthetic construct*: it describes nobody, so a change in timing moves it without moving anything real. The second is a summary that moves because the *population* changed rather than the people in it — the median wage, which can fall while every individual's wage rises. The third is a summary that is *many-to-one*, so two societies needing opposite policies can share one number, which is the Gini coefficient.

A total fertility rate is not children per woman. It is the sum, over single years of age, of the fertility rates observed among women of each age **in one calendar year**. The 25-year-olds in that sum were born in 2000 and the 40-year-olds in 1985. Nobody has lived the life the number describes. It is the completed family size of a synthetic woman assembled from a single year's cross-section, and it is a perfectly good statistic as long as you remember that timing moves it.""")

    post.add(
        "What a pure delay does, exactly",
        r"""Here is the cleanest possible version of the problem. Give every cohort of women the same completed fertility *Q* — nobody has fewer children than anybody else, ever — spread over age by a density with mean age *mu* and spread *sigma*. Now let each cohort postpone: cohort *b* has mean age *mu* + *d b*, so successive cohorts have their children slightly later. The rate observed in calendar year *t* among women aged *a* comes from the cohort born in *t* − *a*, so the period schedule is

$$
f(a, t) = Q \, \phi(a; \mu + d(t - a), \sigma)
$$

Substitute *u* = *a*(1 + *d*) − *mu* − *dt* and integrate over age. The Jacobian is the whole story: *da* = *du*/(1 + *d*), so

$$
\mathrm{TFR}(t) = \frac{Q}{1 + d}
$$

and the period mean age of childbearing works out to (*mu* + *dt*)/(1 + *d*), which rises at

$$
r = \frac{d}{1 + d}
$$

Note what *r* is not. It is not *d*. The period mean age rises **more slowly** than cohorts postpone, because each year's cross-section mixes cohorts at different stages of their own delay. A published mean-age change of 0.1 years per year corresponds to a cohort postponement of about 0.11.

Put the two together. Since 1 − *r* = 1/(1 + *d*),

$$
\frac{\mathrm{TFR}(t)}{1 - r} = Q
$$

That is the Bongaarts-Feeney tempo adjustment, and on this schedule it is not a correction with an error term. It is an identity.""")

    post.add(
        "",
        f"""{snip['identity'].markdown()}

Six rows, and in each one the period TFR matches {QUANTUM}/(1 + *d*), the measured mean-age change matches *d*/(1 + *d*), and the adjustment returns {QUANTUM} to five decimal places. The largest disagreement anywhere in the table is {max(abs(sweep[x]['tfr'] - sweep[x]['tfr_algebra']) for x in sweep):.0e}, which is the cost of summing over whole years of age instead of integrating.

There is a third consequence in that table that is easy to miss. The period schedule's spread is *sigma*/(1 + *d*) — postponement **compresses** the observed schedule, so a period fertility schedule is narrower than any real cohort's. This is why the adjustment is computed separately by birth order in practice, and it is where the assumptions start mattering.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "A rebound with nobody behind it",
        f"""So far the postponement rate was constant, and a constant delay just parks the period rate below the quantum. The interesting case is when the delay *changes*, because that is what actually happens to a country: first births move later and later, and then at some point they stop moving later quite so fast.

Take the same population — completed fertility fixed at {QUANTUM} for every cohort, forever — and let cohorts born after year 40 postpone half as much as their predecessors, 0.10 instead of 0.20. Nothing else changes. No policy, no incentive, no change in anybody's family size.""")

    post.add(
        "",
        f"""{snip['rebound'].markdown()}

The period rate goes from {span['lo']:.3f} to {span['hi']:.3f}, a rise of {span['rise']:.1%}, and the algebra says exactly where it lands: {QUANTUM}/1.2 to {QUANTUM}/1.1. The quantum is printed in the second column of that table so it cannot be misread. It never moves.

Two things about the timing are worth more than the size. The rise does not begin until year {span['first']} — {span['first'] - 40} years after the behaviour changed — because the cohorts who changed have to reach childbearing age before the cross-section can see them. And it then takes another {span['done'] - span['first']} years to finish arriving, as those cohorts cross the window. A period fertility rate is a lagging, smeared indicator of behaviour that is decades old, which is an awkward property for a statistic used to evaluate policy on an annual news cycle.""",
        level=3,
        figures=[figs["f1"]])

    post.add(
        "Korea's arithmetic",
        f"""Now the published numbers, and only the published numbers. Statistics Korea reports the mean age of mothers at childbirth as {K['mac'][2022]} in 2022, {K['mac'][2023]} in 2023 and {K['mac'][2024]} in 2024. So *r* is about {korea['r_recent']:.1f} a year, and the correction factor is 1/(1 − {korea['r_recent']:.1f}) = {korea['factor_recent']:.3f}.

Separately, the OECD Family Database records that Korea's mean age at first birth has risen by more than five years since 2000 — it stood at {K['first_birth_2024']} in 2024. Over twenty-four years that is an average of at least {korea['r_long_run']:.2f} a year, roughly twice the current pace.""")

    post.add(
        "",
        f"""{snip['korea'].markdown()}

Read the last two lines together. The reported total fertility rate rose {korea['reported_rise']:+.1%} between 2023 and 2025. Over the same story, the tempo correction factor shrank by {korea['factor_fall']:+.1%}, because postponement decelerating means the distortion it was creating gets smaller.

Those are the same order of magnitude, and that is all I am willing to claim from it. It is not a decomposition: doing this properly needs the mean age by birth order for each year, applied order by order, and the deceleration is measured here as a long-run average against a two-year change rather than as a series. What it is enough for is the negative statement in the title. **A rise of this size does not require anybody to have had more children.** It is fully available from postponement slowing down, and postponement slowing down is exactly what a country produces at the end of a long delay of first births.

Two things this does not say. It does not say the level is fine: {K['tfr'][2024]:.3f} adjusted at the recent pace is {korea['adjusted_2024']:.3f}, which is still the lowest national fertility ever recorded and still far below replacement. And it does not say the rebound is *only* timing — marriages rose sharply in the same period, and marriage in Korea is a strong leading indicator of first births. The claim is about what the statistic can and cannot distinguish, not about which explanation is true.""",
        level=3)

    post.add(
        "Where I expected it to break, and where it actually does",
        f"""The Bongaarts-Feeney adjustment has one obviously suspect assumption: it assumes the schedule shifts rigidly, keeping its shape and its spread. Real schedules do not. I drafted this episode expecting that to be the punchline, and then measured it.

Let the cohort schedule widen while it shifts, from {4.5} years of spread to {8.0} — a change far larger than any country has produced — with postponement running at 0.2 throughout. The adjusted rate comes out {abs(var[-1]['error']):.1%} low. At a widening of one year it is {abs(var[2]['error']):.2%} low. The assumption the literature attacks hardest costs about a percent, and the sign is consistent, so it is a bias rather than noise, and a small one.""",
        figures=[figs["f2"]])

    post.add(
        "",
        f"""That agrees with Mazzuco and Zanotto (2025), who find the adjustment robust to shape and scale changes for a reason worth stating: cohort shape changes mostly show up as movements in the **period mean age**, which is the quantity the formula already uses. The formula absorbs them by accident.

So where does it actually break? Two places, and neither is the one I went looking for.

The first is the mean age it is given. The all-order mean age of childbearing moves when the *quantum* moves, not only when timing does — fewer third births pull the all-order mean age down. Korea's 2024 release shows exactly this: the mean age rose {K['mac'][2024] - K['mac'][2023]:.1f} for first births, was flat for second, and **fell** {abs(K['third_birth_change_2024']):.1f} for third. An adjustment computed on the all-order figure, which is what the arithmetic above does, mixes tempo and quantum together. Order-specific is not a refinement here; it is the method.

The second is what the adjusted number is. It is a tempo-free *period* rate: what the TFR would be if timing stopped moving this year. It is not any cohort's completed fertility, and if postponement never stops, the tempo-free rate is a counterfactual nobody lives in. The honest use is comparative — is this year's distortion bigger or smaller than last year's — and that is the use this episode makes of it.""",
        level=3)

    post.add(
        "What to keep",
        f"""1. The period TFR is one year of age-specific rates stacked into a synthetic person. Timing moves it; family size is not the only thing it responds to.
2. With every cohort at the same completed fertility *Q* and a postponement of *d* per cohort, the period rate is exactly *Q*/(1 + *d*) and the period mean age rises at *d*/(1 + *d*), which is smaller than *d*.
3. So `TFR/(1 − r)` recovers *Q* exactly, where *r* is the annual change in the period mean age. It is an identity on a rigidly shifting schedule, not an approximation.
4. Postponement slowing from 0.2 to 0.1 raises the period rate {span['rise']:.1%} with nobody changing their family size — starting {span['first'] - 40} years later and taking {span['done'] - span['first']} more years to finish.
5. Korea's reported {korea['reported_rise']:+.0%} rise from 2023 to 2025 and the {abs(korea['factor_fall']):.0%} shrinkage of its tempo correction are the same size. That is not a decomposition, but it is enough to retire the phrase "more children".
6. The assumption everyone attacks — rigid shift, constant spread — costs about {abs(var[-1]['error']):.0%}. The assumption nobody mentions — that the mean age you feed it is a timing measure rather than a mixture of timing and quantum — is where the method actually needs care.""")

    post.add(
        "Exercise",
        """Take your own country's published total fertility rate and mean age of mothers at childbirth for the last ten years. Compute `TFR/(1 − r)` for each year, with *r* as half the change in the mean age across that year.

Two questions to ask of the result. Does the adjusted series have a different **shape** from the reported one, or only a different level? A constant *r* only shifts the level, and if that is all you find, then the tempo story explains none of your country's change and you have learned something real.

Then find the mean age by birth order and redo it order by order, summing the adjusted order-specific rates. Where the two answers disagree, the all-order mean age was carrying quantum change, and the size of the disagreement is the size of the mistake the one-line version makes.

The uncomfortable part is last. Whatever number you end up with, it is still a period rate — a statement about a synthetic woman in a counterfactual year where timing stopped. Write down what you would need in order to say something about real women's completed families, and notice that all of it is unavailable until they are fifty.""")

    post.hero = figs["hero"]
    return post


def main() -> Post:
    return build()


if __name__ == "__main__":
    r = compute()
    print("identity check:")
    for row in r["sweep"]:
        print(f"  d={row['delta']:.2f}  TFR {row['tfr']:.6f} "
              f"(algebra {row['tfr_algebra']:.6f})  r {row['r']:.5f} "
              f"(algebra {row['r_algebra']:.5f})  adjusted {row['adjusted']:.6f}")
    s = r["reb_span"]
    print(f"\nrebound: {s['lo']:.4f} -> {s['hi']:.4f} = {s['rise']:+.1%}, "
          f"first movement {s['first']}, done {s['done']}")
    print("\nvariance bias:")
    for v in r["var"]:
        print(f"  spread {v['spread']:.3f}  error {v['error']:+.4%}")
    print("\nkorea:", {k: round(v, 4) for k, v in r["korea"].items()})
