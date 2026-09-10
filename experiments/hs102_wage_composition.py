"""Headline Statistics 2: Everyone's Wage Rose and the Median Fell.

The second episode, on a summary that moves because the *population* changed
rather than the people in it. Episode 1's total fertility rate described nobody;
this one describes somebody, just not the same somebody twice.

Measured, and every threshold has a closed form to check it against:

* Give `n` incumbents a universal raise of `g` and add `m` entrants below the
  incumbent median. The median slides `m / 2` ranks, order statistics near the
  median are spaced `1 / (n f(M))` apart, so the median falls once the entry
  share passes `s* = 2 g M f(M)`. For a lognormal that is `0.798 g / sigma`:
  4.0% at `g = 3%` and `sigma = 0.6`, bisected at 3.87%.
* `s*` *falls* as `sigma` rises, so a more unequal distribution is more fragile
  to composition, not less.
* The median flips at about 60% of the entry share the mean needs, at every
  spread tried -- 0.61, 0.63, 0.58, 0.56, 0.53. I expected the other ordering.
* Run the sign the other way and it reproduces 2020. Losing the lowest-paid
  9.8% while everyone still employed gets 2.5% prints a median growth of
  10.4%, which is what US median usual weekly earnings printed in the quarter
  the country lost 20.5 million jobs.

Run: `standarderror run hs102_wage_composition --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.aggregates import composition as cp
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 9, 10)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Headline Statistics, Taught Through What Breaks"
SERIES_TAG = "Headline Statistics"

SIGMAS = (0.3, 0.4, 0.6, 0.8, 1.0)
RAISE = 0.03
#: Underlying pay growth for the exit scenario. 2.5% is an ordinary year for US
#: wages and is deliberately unremarkable: the point is what the print does, not
#: what the raise was.
UNDERLYING = 0.025
SHARES = (0.0, 0.05, 0.10, 0.15, 0.20)
#: Net hiring shares for the ordinary-year case. US employment grows by about a
#: percent or a percent and a half in a normal year, so this brackets it.
HIRING = (0.005, 0.010, 0.015, 0.020, 0.030)

#: Published figures the prose quotes. None of them enter any computation.
PUBLISHED = {
    "jobs_lost_april_2020": 20_500_000,
    #: Atlanta Fed macroblog, 10 November 2021.
    "median_weekly_published": 0.104,
    "tracker_removed_pp": 8.0,
    "establishment_spike_pp": 4.5,
    "establishment_ex_leisure_pp": 2.6,
    #: Dallas Fed, 9 February 2021, on a CPS decomposition.
    "cps_spike_pp": 7.0,
    "cps_composition_pp": 5.3,
}


def compute() -> dict:
    sweep = cp.fragility_sweep(SIGMAS, g=RAISE)
    exits = [cp.exit_effect(share=s, g=UNDERLYING) for s in SHARES]
    ordinary = [cp.entry_effect(share=s, g=RAISE) for s in HIRING]
    implied = cp.share_implying(PUBLISHED["median_weekly_published"],
                               g=UNDERLYING)

    # The individual-level picture the first figure draws: a sample of workers,
    # every one of whom gains, against a median that does not.
    rng = np.random.default_rng(11)
    n = 200_000
    w = rng.lognormal(0.0, cp.SIGMA, n)
    med_before = float(np.median(w))
    raised = w * (1.0 + RAISE)
    pool = rng.lognormal(0.0, cp.SIGMA, 6 * n)
    low = pool[pool < med_before]
    share = sweep[2]["median_share"] * 1.6      # comfortably past the flip
    combined = np.concatenate([raised, low[:int(share * n)]])
    picture = {
        "share": float(share),
        "median_before": med_before,
        "median_raised": float(np.median(raised)),
        "median_after": float(np.median(combined)),
        "sample_before": rng.choice(w, 40, replace=False),
        "worst_individual_gain": float(RAISE),
    }
    picture["median_change"] = picture["median_after"] / med_before - 1.0
    return {"sweep": sweep, "exits": exits, "implied": implied,
            "ordinary": ordinary, "picture": picture}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    sweep = {row["sigma"]: row for row in res["sweep"]}
    pic = res["picture"]
    exits = {row["share"]: row for row in res["exits"]}
    implied = res["implied"]

    def everyone_up(ax, m):
        # Plot the *change*, not the level. On a wage axis a 3% raise is a
        # displacement too small to see, which is itself the reason this
        # failure is invisible in practice -- but it makes for a useless
        # picture, so the axis here is the change each party experienced.
        rng = np.random.default_rng(5)
        y = 1.0 + rng.uniform(-0.20, 0.20, len(pic["sample_before"]))
        ax.scatter(np.full(len(y), RAISE), y, s=46, color=m.series[2],
                   alpha=0.8, edgecolors=m.surface, linewidths=0.7)
        ax.scatter([pic["median_change"]], [0.0], s=220, color=m.series[1],
                   edgecolors=m.surface, linewidths=1.0, zorder=3)
        ax.axvline(0.0, color=m.ink_secondary, lw=1.5)
        ax.annotate(f"every one of the forty workers, {RAISE:+.0%}",
                    (RAISE, 1.30), ha="center", fontsize=9.2,
                    color=m.series[2])
        ax.annotate(f"the median wage, {pic['median_change']:+.1%}",
                    (pic["median_change"], -0.32), ha="center", fontsize=9.2,
                    color=m.series[1])
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["the statistic", "the people"], fontsize=9.5)
        ax.set_ylim(-0.75, 1.62)
        ax.set_xlim(-0.028, 0.045)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")

    out["f0"] = charts.diagram(
        everyone_up,
        title="Every worker to the right of zero, and the statistic to the left",
        subtitle=(f"Forty workers sampled from the population, each with the "
                  f"same universal raise of {RAISE:.0%}, against what the "
                  f"median did once entrants worth {pic['share']:.0%} of the "
                  f"workforce were hired below it."),
        xlabel="change over the period", ylabel="",
        source="Simulated; standarderror/aggregates/composition.py.",
        alt=("A dot plot of changes. Forty green dots cluster at plus three "
             "percent on the upper row; one large orange dot sits at minus "
             "one point four percent on the lower row, across a vertical zero "
             "line."),
        caption=(f"There is no dispersion in the upper row to look at: the "
                 f"raise is universal, so every worker is at exactly "
                 f"{RAISE:+.0%}. The median prints "
                 f"{pic['median_change']:+.1%}. Note also how small a real "
                 f"raise is on a wage axis - {RAISE:.0%} is a displacement you "
                 f"could not see if this figure plotted levels, which is part "
                 f"of why the composition term goes unnoticed."),
        figsize=(7.4, 3.9),
        path=str(IMG / f"hs102-f0-everyone-up.{EXT}"))[0]

    def thresholds(ax, m):
        grid = np.linspace(0.25, 1.05, 200)
        ax.plot(grid, [cp.median_threshold(RAISE, s) for s in grid], lw=1.9,
                color=m.series[0], label="median, algebra:  0.798 g / sigma")
        ax.scatter(SIGMAS, [sweep[s]["median_share"] for s in SIGMAS], s=44,
                   zorder=3, color=m.series[0], edgecolors=m.surface,
                   linewidths=0.8, label="median, bisected")
        ax.plot(SIGMAS, [sweep[s]["mean_share"] for s in SIGMAS], marker="s",
                ms=6, lw=1.9, color=m.series[1], label="mean, bisected")
        ax.axvline(cp.SIGMA, color=m.grid, lw=1.4)
        ax.annotate("US log hourly wages,\nroughly", (cp.SIGMA, 0.125),
                    textcoords="offset points", xytext=(8, 0), fontsize=8.6,
                    color=m.ink_secondary, va="top")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_ylim(0, 0.145)
        ax.legend(frameon=False, fontsize=8.6, loc="upper right")

    out["f1"] = charts.diagram(
        thresholds,
        title="The median is the more fragile of the two, at every spread",
        subtitle=(f"The entry share of below-median workers that exactly "
                  f"cancels a universal raise of {RAISE:.0%}. Bisected against "
                  f"lognormal wage populations of 400,000."),
        xlabel="spread of log wages (sigma)",
        ylabel="entry share that hides the raise",
        source="Simulated; standarderror/aggregates/composition.py.",
        alt=("Two falling curves against the spread of log wages, the median's "
             "below the mean's throughout, with simulation points sitting on "
             "the median's algebraic line."),
        caption=(f"Two things here are backwards. The thresholds **fall** as "
                 f"the wage distribution widens, so a more unequal country is "
                 f"more fragile to composition, not less - a wide "
                 f"distribution is thin at its median, so the same slide in "
                 f"rank travels further in money. And the median sits below "
                 f"the mean everywhere, at "
                 f"{min(sweep[s]['ratio'] for s in SIGMAS):.2f} to "
                 f"{max(sweep[s]['ratio'] for s in SIGMAS):.2f} of the mean's "
                 f"threshold. I expected that ordering the other way round."),
        path=str(IMG / f"hs102-f1-thresholds.{EXT}"))[0]

    def other_sign(ax, m):
        s = np.array(SHARES)
        ax.plot(s, [exits[x]["published_median_growth"] for x in s], marker="o",
                ms=6, lw=2.0, color=m.series[1],
                label="what a published median prints")
        ax.axhline(UNDERLYING, color=m.series[2], lw=1.8, ls="--",
                   label=f"what everybody still employed got, "
                         f"{UNDERLYING:.1%}")
        ax.axhline(PUBLISHED["median_weekly_published"], color=m.grid, lw=1.5)
        ax.annotate(f"US median usual weekly earnings, "
                    f"{PUBLISHED['median_weekly_published']:.1%}",
                    (0.0, PUBLISHED["median_weekly_published"]),
                    textcoords="offset points", xytext=(4, 6), fontsize=8.6,
                    color=m.ink_secondary)
        ax.axvline(implied["share"], color=m.grid, lw=1.3, ls=":")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
        ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    out["f2"] = charts.diagram(
        other_sign,
        title="Run the sign backwards and a recession prints a pay rise",
        subtitle=(f"The lowest-paid share on the axis loses their jobs; "
                  f"everybody still employed gets {UNDERLYING:.1%}. The line "
                  f"is what a published median would then report."),
        xlabel="share of the lowest-paid who lose their jobs",
        ylabel="median wage growth as printed",
        source="Simulated; standarderror/aggregates/composition.py.",
        alt=("A rising line of published median growth against the job-loss "
             "share, crossing a horizontal reference at ten percent, well "
             "above a flat dashed line at two and a half percent."),
        caption=(f"Nobody's pay growth is anything other than "
                 f"{UNDERLYING:.1%} anywhere on this line. Losing the "
                 f"lowest-paid {implied['share']:.1%} prints "
                 f"{implied['published_median_growth']:.1%}, which is what US "
                 f"median usual weekly earnings printed in the second quarter "
                 f"of 2020. The dotted rule marks that share; the country lost "
                 f"{PUBLISHED['jobs_lost_april_2020'] / 1e6:.1f} million jobs "
                 f"in the single month of April 2020, concentrated in its "
                 f"lowest-paid industries."),
        path=str(IMG / f"hs102-f2-other-sign.{EXT}"))[0]

    # --- f3: the wedge an ordinary year takes out ---------------------------
    ordinary = {row["share"]: row for row in res["ordinary"]}

    def wedge(ax, m):
        h = np.array(HIRING)
        printed = np.array([ordinary[x]["printed"] for x in h])
        ax.fill_between(h, printed, RAISE, color=m.series[1], alpha=0.18,
                        label="taken by composition")
        ax.plot(h, printed, marker="o", ms=6, lw=2.0, color=m.series[1],
                label="what the median prints")
        ax.axhline(RAISE, color=m.series[2], lw=1.8, ls="--",
                   label=f"what every worker got, {RAISE:.0%}")
        ax.axvline(0.015, color=m.grid, lw=1.4)
        ax.annotate("an ordinary year\nof US net hiring", (0.015, 0.004),
                    textcoords="offset points", xytext=(8, 0), fontsize=8.6,
                    color=m.ink_secondary)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
        ax.set_ylim(0, RAISE * 1.18)
        ax.legend(frameon=False, fontsize=8.6, loc="lower left")

    out["f3"] = charts.diagram(
        wedge,
        title="An ordinary year of hiring takes a third of an ordinary raise",
        subtitle=(f"Every worker gets {RAISE:.0%}. The shaded wedge is the part "
                  f"of it that never reaches the printed median, as a function "
                  f"of how many below-median workers were hired."),
        xlabel="net hiring below the median, as a share of the workforce",
        ylabel="wage growth",
        source="Simulated; standarderror/aggregates/composition.py.",
        alt=("A falling line of printed wage growth under a flat dashed line "
             "at three percent, with the gap between them shaded."),
        caption=(f"At {HIRING[2]:.1%} of net hiring the print reads "
                 f"{ordinary[0.015]['printed']:.2%} against a raise of "
                 f"{RAISE:.0%} - a wedge of "
                 f"{ordinary[0.015]['swallowed']:.2f} points, or "
                 f"{ordinary[0.015]['swallowed_fraction']:.0%}. That is larger "
                 f"than real wage growth in a good year, and it arrives with "
                 f"no headline attached, because the printed number looks "
                 f"exactly like a wage series is supposed to look."),
        path=str(IMG / f"hs102-f3-wedge.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    sweep = {row["sigma"]: row for row in res["sweep"]}
    implied = res["implied"]

    def arrows(panel, m):
        for i in range(7):
            y = i * 0.14
            panel.annotate("", xy=(0.62, y), xytext=(0.18, y),
                           arrowprops=dict(arrowstyle="-|>", lw=2.0,
                                           color=m.series[2],
                                           mutation_scale=11))
        panel.annotate("", xy=(0.22, 1.02), xytext=(0.78, 1.02),
                       arrowprops=dict(arrowstyle="-|>", lw=3.2,
                                       color=m.series[1], mutation_scale=17))
        panel.set_xlim(0.05, 0.95)
        panel.set_ylim(-0.1, 1.16)

    def queue(panel, m):
        # The middle of the queue, and a new arrival that moves who stands there.
        x = np.arange(9)
        panel.scatter(x, np.zeros(9), s=64, color=m.grid)
        panel.scatter([4], [0], s=110, color=m.series[0])
        panel.scatter([-1.2, -0.6], [0, 0], s=64, color=m.series[1])
        panel.scatter([3], [0], s=110, color=m.series[1], zorder=3)
        panel.annotate("", xy=(3.1, 0.42), xytext=(4.0, 0.42),
                       arrowprops=dict(arrowstyle="-|>", lw=2.2,
                                       color=m.series[1], mutation_scale=13))
        panel.set_xlim(-2.0, 9.0)
        panel.set_ylim(-0.7, 0.9)

    def printed(panel, m):
        panel.bar([0, 1], [implied["published_median_growth"], UNDERLYING],
                  color=[m.series[1], m.series[2]], width=0.5)
        panel.set_xlim(-0.6, 1.6)
        panel.set_ylim(0, 0.125)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=2,
        headline="A wage statistic compares two populations, not two payslips",
        panels=[(arrows, f"{RAISE:+.0%}", "every worker, and yet"),
                (queue, f"{sweep[0.6]['median_share']:.0%}", "of new hires is enough"),
                (printed, f"{implied['published_median_growth']:.0%}",
                 f"printed on {UNDERLYING:.1%} real")],
        note=("Nobody in the simulation is paid less than before. The median "
              "still moves the wrong way, and running the same arithmetic "
              "backwards is why wages appeared to jump in 2020."),
        alt=("Three hand-drawn frames: short arrows all pointing right under "
             "one long arrow pointing left; a queue whose middle person "
             "changes when two arrivals join the front; and two bars, a tall "
             "printed figure against a short real one."),
        path=str(IMG / f"hs102-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["three"] = s.run("""
        import numpy as np

        before = np.array([10.0, 20.0, 30.0])
        raised = before * 1.1              # everybody, ten percent, no exceptions
        after = np.concatenate([raised, [5.0, 5.0]])   # two people are hired

        print("wages before     ", before, " median", np.median(before))
        print("after the raise  ", raised, " median", np.median(raised))
        print("after two hires  ", after, " median", np.median(after))
        print("did anyone lose? ", bool(np.any(raised < before)))
    """, expect=["did anyone lose?"])

    out["threshold"] = s.run(f"""
        from standarderror.aggregates import composition as cp

        # The entry share that exactly cancels a universal raise, bisected
        # against a population of 400,000, next to the closed form
        # s* = 2 g M f(M) = 0.798 g / sigma.
        print(f"{{'sigma':>6}} {{'median s*':>10}} {{'algebra':>9}} "
              f"{{'mean s*':>9}} {{'ratio':>7}}")
        for row in cp.fragility_sweep({list(SIGMAS)}, g={RAISE}):
            print(f"{{row['sigma']:>6.1f}} {{row['median_share']:>10.2%}} "
                  f"{{row['median_predicted']:>9.2%}} "
                  f"{{row['mean_share']:>9.2%}} {{row['ratio']:>7.2f}}")
    """, expect=["ratio"])

    out["exit"] = s.run(f"""
        # Now the 2020 sign: the lowest-paid lose their jobs instead, and
        # everybody still employed gets {UNDERLYING:.1%}. Nothing else changes.
        print(f"{{'jobs lost':>10}} {{'median prints':>14}} "
              f"{{'mean prints':>12}} {{'anyone actually got':>20}}")
        for share in {list(SHARES)}:
            r = cp.exit_effect(share=share, g={UNDERLYING})
            print(f"{{share:>10.0%}} {{r['published_median_growth']:>14.2%}} "
                  f"{{r['published_mean_growth']:>12.2%}} "
                  f"{{r['matched_growth']:>20.1%}}")
    """, expect=["anyone actually got"])

    out["ordinary"] = s.run(f"""
        # And the boring case, which runs every year rather than once a
        # century: net hiring of a percent or two, concentrated below the
        # median, against a universal raise of {RAISE:.0%}.
        print(f"{{'below-median hires':>19}} {{'the print':>10}} "
              f"{{'the raise':>10}} {{'swallowed':>10}}")
        for share in {list(HIRING)}:
            r = cp.entry_effect(share=share, g={RAISE})
            print(f"{{share:>19.1%}} {{r['printed']:>10.2%}} "
                  f"{{r['matched']:>10.1%}} {{r['swallowed_fraction']:>10.0%}}")
    """, expect=["swallowed"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    sweep = {row["sigma"]: row for row in res["sweep"]}
    exits = {row["share"]: row for row in res["exits"]}
    implied = res["implied"]

    # The spine, asserted rather than trusted.
    for row in sweep.values():
        assert abs(row["median_share"] / row["median_predicted"] - 1) < 0.05
        assert row["median_share"] < row["mean_share"], row
        assert 0.4 < row["ratio"] < 0.8, row
    shares = [sweep[s]["median_share"] for s in SIGMAS]
    assert shares == sorted(shares, reverse=True), shares
    assert res["picture"]["median_change"] < 0.0
    # With nobody laid off the print is the raise, which is the control that
    # says the exit machinery is not adding anything of its own.
    assert abs(exits[0.0]["published_median_growth"] - UNDERLYING) < 1e-3
    ordinary = {row["share"]: row for row in res["ordinary"]}
    for row in ordinary.values():
        assert 0.0 < row["swallowed"] < RAISE, row
    swallowed = [ordinary[s]["swallowed_fraction"] for s in HIRING]
    assert swallowed == sorted(swallowed), swallowed
    growth = [exits[s]["published_median_growth"] for s in SHARES]
    assert growth == sorted(growth), growth
    assert abs(implied["published_median_growth"]
               - PUBLISHED["median_weekly_published"]) < 1e-3
    assert 0.05 < implied["share"] < 0.15

    post = Post(
        title=f"{SERIES_TAG} 2: Everyone's Wage Rose and the Median Fell",
        slug="headline-statistics-2-wage-composition",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=2,
        prerequisites=["headline-statistics-1-fertility-tempo"],
        date=POST_DATE,
        subtitle=("A wage statistic compares two populations, not two "
                  "payslips. Once hiring and firing happen between the two "
                  "prints, the difference carries a term that has nothing to "
                  "do with anybody's pay - and it is large enough to reverse "
                  "the sign."),
        summary=(
            "Give every worker a universal raise of g and hire entrants below "
            "the median. The median slides m/2 ranks, order statistics near "
            "the median are spaced 1/(n f(M)) apart, so the median falls once "
            "the entry share passes 2 g M f(M) - which for a lognormal is "
            "0.798 g/sigma, or 4.0% at a 3% raise and the spread of US log "
            "wages. Bisected: 3.87%. Two results were backwards from what I "
            "expected. The threshold falls as inequality rises, so a wider "
            "wage distribution is more fragile to composition rather than "
            "less. And the median flips at about 60% of the entry share the "
            "mean needs, at every spread tried - the median is robust to "
            "outliers, which is not what composition is. Run the sign the "
            "other way and it reproduces 2020: losing the lowest-paid 9.8% "
            "while everybody still employed gets 2.5% prints a median growth "
            "of 10.4%, and 10.4% is what US median usual weekly earnings "
            "printed in the quarter the country lost 20.5 million jobs."),
        tags=["statistics", "economics", "labor-economics", "data-science",
              "public-data", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data enters any computation. Every figure and every "
            "table here is constructed from the simulation in the code shown, "
            "executed when this page was built.",
            "Published figures quoted in the prose, and used nowhere else: "
            "US Bureau of Labor Statistics, for the 20.5 million jobs lost in "
            "April 2020; Federal Reserve Bank of Atlanta, \"Compositional "
            "distortions to a measure of wage growth during the pandemic\", "
            "*macroblog* (10 November 2021), for the 10.4% published median "
            "usual weekly earnings growth, the nearly 8 percentage points the "
            "matched-individual Wage Growth Tracker removes, and the "
            "establishment-survey spike of 4.5 points falling to 2.6 with "
            "leisure and hospitality excluded; Federal Reserve Bank of "
            "Dallas, \"Pandemic pushed the U.S. into recession ... and hourly "
            "wages rose?\" (9 February 2021), for the CPS decomposition in "
            "which the composition term is 5.3 of 7 percentage points.",
            "Machinery: `standarderror/aggregates/composition.py`, tested in "
            "`tests/test_composition.py`.",
            "Where this stops: the fix in the last section - matching "
            "individuals across the two periods - removes the arithmetic "
            "problem and replaces the question. A matched sample can only "
            "contain people who held a job in both periods, so in a quarter "
            "when twenty million people stopped working it answers something "
            "much narrower than \"what happened to wages\".",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": ("lognormal wage populations of 200,000 to 400,000 in "
                           "units of their own median, with the raise applied "
                           "to every incumbent without exception"),
            "determinism": ("one generator per measurement, seeded from that "
                            "measurement's own parameters - spread and raise - "
                            "rather than advanced through a loop"),
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    sweep = {row["sigma"]: row for row in res["sweep"]}
    implied = res["implied"]
    P = PUBLISHED

    post.add(
        "A pay rise in the worst month on record",
        f"""In April 2020 the United States lost {P['jobs_lost_april_2020'] / 1e6:.1f} million jobs in a single month. In the quarter that contained it, published median usual weekly earnings grew {P['median_weekly_published']:.1%} against the year before.

Both of those are correct. Neither is a mistake in the data, and no revision has taken them back. They sit together because a wage statistic does not compare two payslips. It compares two populations, and once people are hired and laid off between the prints, the difference contains a term that has nothing to do with anybody's pay.

Episode 1 was about a summary that describes nobody. This one is about a summary that describes somebody — just not the same somebody twice.""")

    post.add(
        "The version that fits in three numbers",
        """Before writing any algebra, here is the whole failure at a scale you can check by eye.""")

    post.add(
        "",
        f"""{snip['three'].markdown()}

Three workers on 10, 20 and 30. Everybody gets ten percent, so nobody is worse off — the last line of that output is the check. Two people are then hired at 5. The median goes from 20 to 11.

That is not a rounding artefact or a small-sample curiosity. It is the mechanism, and the rest of this episode is about how much churn it takes at realistic scale.

It also has a name, which is worth having because it makes the failure recognisable elsewhere. This is Simpson's paradox with the groups left implicit: incumbents and entrants are two groups, each of which moved up or stayed put, while the pooled summary moved down. The same shape produces a hospital whose survival rate falls as every ward improves, a company whose average deal size shrinks in the year every salesperson closes bigger, and a model whose accuracy drops on a benchmark it got better at, because the benchmark grew a harder section. Wherever a pooled number is reported over two periods and the pool was allowed to change, this term is present and unlabelled.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "How much churn it takes, exactly",
        r"""Sort the *n* incumbents. The median is the one at rank (*n* + 1)/2. Give everybody a raise of *g*, then hire *m* entrants who all earn less than the old median. In the combined list the first *m* places belong to entrants, so the new median is the incumbent at rank

$$
\frac{n + m + 1}{2} - m = \frac{n - m + 1}{2}
$$

The median has slid *m*/2 ranks **down** the incumbent list. How far is that in money? Consecutive order statistics of a sample from a density *f* are spaced about 1/(*n f*(*M*)) apart near the median — the same order-statistic spacing that decided where a barcode's largest gap falls two series ago. So the slide costs

$$
\frac{m}{2 n f(M)}
$$

while the raises are worth *gM*. Setting them equal gives the entry share at which a universal raise stops showing up at all:

$$
s^{*} = 2 g M f(M)
$$

*Mf*(*M*) is dimensionless. For a lognormal it is 1/(*sigma*√(2π)), so""")

    post.add(
        "",
        rf"""$$
s^{{*}} = \frac{{0.798\, g}}{{\sigma}}
$$

At a three percent raise and a log-wage spread of 0.6, roughly what US hourly wages have, that is **{cp.median_threshold(RAISE, cp.SIGMA):.1%}**. Four percent of new hires below the median is enough to hide a universal three percent raise completely.""",
        level=3)

    post.add(
        "",
        f"""{snip['threshold'].markdown()}

The bisected shares sit on the closed form to within {max(abs(sweep[s]['median_share'] / sweep[s]['median_predicted'] - 1) for s in SIGMAS):.0%}. Two things in that table are backwards from what I expected, and both are in the last two columns.

**The threshold falls as inequality rises.** At a log-wage spread of {SIGMAS[0]} it takes {sweep[SIGMAS[0]]['median_share']:.1%} of new hires to hide the raise; at {SIGMAS[-1]} it takes {sweep[SIGMAS[-1]]['median_share']:.1%}. A more unequal country is *more* fragile to composition, not less, and the reason is in the formula: a wide distribution is thin at its median, so the same slide in rank travels further in money.

**The median breaks before the mean, every time.** The ratio column runs {sweep[SIGMAS[0]]['ratio']:.2f}, {sweep[SIGMAS[1]]['ratio']:.2f}, {sweep[0.6]['ratio']:.2f}, {sweep[SIGMAS[3]]['ratio']:.2f}, {sweep[SIGMAS[4]]['ratio']:.2f} — the median flips at a bit over half the entry share the mean needs. I had assumed the opposite, on the usual grounds that the median is the robust one. It is robust to *outliers*. Composition is not an outlier problem: it moves the median by moving who is standing in the middle, and the mean at least has the decency to weight a new arrival by only 1/*n*.""",
        level=3,
        figures=[figs["f1"]])

    post.add(
        "The boring version runs every year",
        f"""Everything so far has been calibrated to make a point: a raise that vanishes completely needs {cp.median_threshold(RAISE, cp.SIGMA):.1%} of new hires, which is a lot of churn for one year. That framing undersells the problem, because the interesting case is not the raise vanishing. It is the raise being *quietly reduced*, every year, by an amount nobody reports.

Net employment growth in an ordinary American year is around one to one and a half percent, and hiring skews below the median because entry-level jobs are entry-level. So put an ordinary amount of hiring against an ordinary raise.""")

    post.add(
        "",
        f"""{snip['ordinary'].markdown()}

Net hiring of {HIRING[2]:.1%} — a completely unremarkable year — swallows {res['ordinary'][2]['swallowed']:.2f} percentage points of a {RAISE:.0%} raise, which is {res['ordinary'][2]['swallowed_fraction']:.0%} of it. The print reads {res['ordinary'][2]['printed']:.2%}.

Put that beside the thing it gets compared against. Real wage growth in a good year is under a point, so a composition term of {res['ordinary'][2]['swallowed']:.2f} points is not a correction to the story; it is larger than the story. And unlike 2020 it produces no headline, no Federal Reserve blog post and no correction, because {res['ordinary'][2]['printed']:.2%} looks exactly like what a wage series is supposed to look like.

The sign of this one is worth holding onto too. In an expansion, composition **understates** wage growth, because you are hiring at the bottom. In a downturn it **overstates** it, because you are firing at the bottom. So the composition term is procyclical in employment and countercyclical in the printed wage — which means the measured series is systematically flatter than the truth in both directions, and a reader who compares a boom's wage print with a bust's is comparing two numbers whose errors point opposite ways.""",
        level=3,
        figures=[figs["f3"]])

    post.add(
        "The same arithmetic, run backwards, is 2020",
        f"""Everything above hires people. Fire them instead and every sign reverses: remove workers from the bottom and the median slides *up* the remaining list, printing a pay rise that nobody received.

So take a population where everybody still employed gets an ordinary {UNDERLYING:.1%}, and delete the lowest-paid share of it.""")

    post.add(
        "",
        f"""{snip['exit'].markdown()}

Read the last column first: it is {UNDERLYING:.1%} on every row, by construction. Nobody in any of those scenarios received anything other than {UNDERLYING:.1%}. The first column is the only thing that changes.

Losing the lowest-paid {implied['share']:.1%} prints a median growth of {implied['published_median_growth']:.1%}. That is the figure the United States printed for median usual weekly earnings in the second quarter of 2020, and April 2020 alone removed {P['jobs_lost_april_2020'] / 1e6:.1f} million jobs concentrated in the country's lowest-paid industries.

I want to be exact about what that is and is not. It is not a decomposition of the real series — I have not touched the microdata, and the simulation's wage distribution is a lognormal rather than the American one. It is a demonstration that a job-loss profile of roughly the observed size and shape reproduces the printed number *with no pay growth anywhere near it*, which is enough to make the printed number unusable as evidence about pay.""",
        level=3,
        figures=[figs["f2"]])

    post.add(
        "What the people who own the statistic did about it",
        f"""This is not a discovery and the institutions that publish these numbers said so at the time.

The Federal Reserve Bank of Dallas decomposed the spike on CPS data and put the composition term at **{P['cps_composition_pp']} of {P['cps_spike_pp']} percentage points** — three quarters of the whole move. The Federal Reserve Bank of Atlanta reported that the establishment survey's {P['establishment_spike_pp']}-point jump between February and April 2020 falls to {P['establishment_ex_leisure_pp']} points once leisure and hospitality are excluded, which is the same statement in a cruder form. And its Wage Growth Tracker, which exists for exactly this reason, removed **nearly {P['tracker_removed_pp']:.0f} percentage points** from the published {P['median_weekly_published']:.1%}.

The Tracker's method is the fix and it is one line long: restrict the sample to people who were employed in *both* periods, and compute the median of their individual wage changes. That is not a smarter estimator of the same quantity. It is a different quantity — the median change of a person, rather than the change of a median — and only the first one is what a reader means by "wages went up".""")

    post.add(
        "Where the fix stops",
        f"""Matching individuals removes the arithmetic problem and replaces it with a sampling one, and the replacement is not free.

A matched sample can only contain people who held a job in both periods. In a quarter when {P['jobs_lost_april_2020'] / 1e6:.1f} million people stopped working, that sample is not the workforce; it is the part of the workforce that survived, and survival was not random with respect to pay. So the matched number is an honest answer to a narrower question: what happened to the pay of people who kept working. If you want to know what happened to *earnings* in the economy, the people who went to zero are the story, and no wage statistic that conditions on being employed can see them.

Which leaves the reader with two numbers and no single one that means what they wanted. That is the actual situation, and the useful move is to say which question you are asking before picking the statistic, rather than picking the statistic and inheriting whichever question it happens to answer.""")

    post.add(
        "What to keep",
        f"""1. A wage statistic compares two populations. The difference is the pay change plus a composition term, and the composition term has no upper bound.
2. Hire entrants below the median and it slides *m*/2 ranks, which costs *m*/(2*n f*(*M*)). A universal raise of *g* disappears once the entry share passes 2*gMf*(*M*), or 0.798*g*/*sigma* on a lognormal — **{cp.median_threshold(RAISE, cp.SIGMA):.1%}** at a {RAISE:.0%} raise and the spread of US wages.
3. That threshold **falls** as the wage distribution widens. More unequal means more fragile.
4. The median is the more fragile of the two, at {min(sweep[s]['ratio'] for s in SIGMAS):.2f} to {max(sweep[s]['ratio'] for s in SIGMAS):.2f} of the mean's threshold. Robust to outliers is not robust to composition.
5. In an ordinary year, net hiring of {HIRING[2]:.1%} below the median swallows {res['ordinary'][2]['swallowed']:.2f} points of a {RAISE:.0%} raise — {res['ordinary'][2]['swallowed_fraction']:.0%} of it, and more than a good year's real wage growth. The sign flips with the cycle, so booms understate and busts overstate.
6. Reverse the sign and losing the lowest-paid {implied['share']:.1%} prints {implied['published_median_growth']:.1%} on {UNDERLYING:.1%} of real growth. The Dallas Fed measured the real thing at {P['cps_composition_pp']} of {P['cps_spike_pp']} points.
7. The fix is to match individuals, and it costs you the people who lost their jobs. Choose the question first.""")

    post.add(
        "Exercise",
        """Find your country's published median or average wage series and its employment series, quarterly, over 2019 to 2021. Plot the wage growth against the change in employment over the same quarters.

If the two are negatively related — wage growth printing high exactly when employment fell — you have found the composition term in your own national statistics, without any microdata. That correlation should not exist if the statistic measured pay.

Then do the harder half. Pick the sharpest quarter and ask what job-loss profile would be needed to produce the whole of that quarter's wage print at zero real pay growth, using `s* = 0.798 g / sigma` and a *sigma* estimated from any published wage decile table. If the answer is smaller than the job losses that actually happened, the print carries no information about pay at all, and you have established that with two published series and one division.

The uncomfortable part: go back and find the commentary written about that quarter's wage number at the time. Some of it will have been written by people who knew all of the above.""")

    post.hero = figs["hero"]
    return post


def main() -> Post:
    return build()


if __name__ == "__main__":
    r = compute()
    print("fragility sweep:")
    for row in r["sweep"]:
        print(f"  sigma {row['sigma']:.1f}  median {row['median_share']:.2%} "
              f"(algebra {row['median_predicted']:.2%})  "
              f"mean {row['mean_share']:.2%}  ratio {row['ratio']:.2f}")
    print("\nexit effect:")
    for row in r["exits"]:
        print(f"  {row['share']:>5.0%} gone -> median "
              f"{row['published_median_growth']:+.2%}  mean "
              f"{row['published_mean_growth']:+.2%}  matched "
              f"{row['matched_growth']:+.1%}")
    i = r["implied"]
    print(f"\nshare implying {i['target']:.1%}: {i['share']:.2%}")
    p = r["picture"]
    print(f"picture: entry share {p['share']:.1%}, median "
          f"{p['median_change']:+.2%}")
