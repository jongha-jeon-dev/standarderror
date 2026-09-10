"""Headline Statistics 3: Two Countries, One Gini, Opposite Policies.

The third episode, on a summary that is many-to-one onto distributions. Episode
1's total fertility rate described nobody; episode 2's median wage described
somebody, just not twice; this one describes something real and does not say
which.

Measured:

* One lognormal population, two distortions, each tuned to the same Gini. The
  poorest tenth losing 90% of its income and the richest hundredth having its
  income multiplied by 2.51 both land on 0.339749, agreeing exactly.
* And the Gini is not the only thing that cannot separate them. p90/p10,
  p50/p10, p90/p50 and the poverty headcount are identical too, because each
  distortion lives inside a tail. Five headline numbers agree while the poorest
  tenth holds 9.2 times more income in one world than the other.
* Why: `dG/dx_k = 2k/(n^2 mu) - c` exactly, so the sensitivity to a unit of
  income is linear in the recipient's *rank* and depends on nothing else about
  them. Measured R-squared of 1.0000000000 and a slope matching the closed form
  to six figures, over ranks whose incomes differ sixfold.
* The bound that follows: the poorest tenth holds 3.3% of income here, so
  destroying all of it moves the coefficient 0.041 -- about what tripling the
  top percentile does.
* What does separate them: a tail share, or a poverty measure that counts depth.
  The gap index is 0.092 against 0.022, a factor of 4.2, on an identical
  headcount.

Run: `standarderror run hs103_gini_shape --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.aggregates import dispersion as dp
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 9, 10)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Headline Statistics, Taught Through What Breaks"
SERIES_TAG = "Headline Statistics"

#: The collapse that fixes the pair. The runaway multiple is solved for.
COLLAPSE = 0.90
SENS_N = 4001

#: Published figures the prose quotes. Nothing here enters a computation, and
#: the two columns come from different sources on different definitions, which
#: the episode says out loud rather than in a footnote.
PUBLISHED = {
    "austria": {"gini": 31.2, "bottom10": 1.89, "top10": 34.19},
    "cyprus": {"gini": 31.8, "bottom10": 2.46, "top10": 33.45},
    "year": 2022,
}

LABELS = [
    ("gini", "Gini coefficient", "{:.6f}"),
    ("p90_p10", "p90 / p10", "{:.3f}"),
    ("p50_p10", "p50 / p10", "{:.3f}"),
    ("p90_p50", "p90 / p50", "{:.3f}"),
    ("headcount", "below half the median", "{:.2%}"),
    ("bottom10_share", "bottom 10% income share", "{:.2%}"),
    ("top1_share", "top 1% income share", "{:.2%}"),
    ("gap_index", "poverty gap index", "{:.4f}"),
]


def compute() -> dict:
    pair = dp.matched_pair(a=COLLAPSE)
    low = dp.describe(pair["collapse"])
    high = dp.describe(pair["runaway"])
    agree = dp.agreement(low, high)
    sens = dp.rank_sensitivity(dp.population(n=SENS_N))
    bottom = dp.destroying_the_bottom()
    top = [{"multiple": 1.0 + b,
            "gini": dp.gini(dp.runaway(pair["base"], b))}
           for b in (0.5, 1.0, 2.0)]
    return {"pair": pair, "low": low, "high": high, "agree": agree,
            "sens": sens, "bottom": bottom, "top": top}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    pair = res["pair"]
    agree = res["agree"]
    sens = res["sens"]

    def curves(ax, m):
        for name, colour, label in (
                ("base", m.grid, f"before, Gini {pair['base_gini']:.3f}"),
                ("collapse", m.series[0],
                 f"collapse at the bottom, Gini {pair['gini']:.3f}"),
                ("runaway", m.series[1],
                 f"runaway at the top, Gini {pair['gini']:.3f}")):
            p, curve = dp.lorenz(pair[name])
            ax.plot(p, curve, lw=2.0 if name != "base" else 1.6, color=colour,
                    ls="--" if name == "base" else "-", label=label)
        ax.plot([0, 1], [0, 1], lw=1.2, color=m.muted)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    out["f0"] = charts.diagram(
        curves,
        title="Two Lorenz curves enclosing the same area, in different places",
        subtitle=("The Gini is twice the area between the curve and the "
                  "diagonal. Both distorted worlds enclose the same area; only "
                  "where they depart from the original differs."),
        xlabel="cumulative share of the population",
        ylabel="cumulative share of income",
        source="Simulated; standarderror/aggregates/dispersion.py.",
        alt=("Two Lorenz curves that separate from the baseline at opposite "
             "ends of the population axis, one sagging at the left and one at "
             "the right, with the same enclosed area."),
        caption=(f"The blue curve flattens along the bottom tenth, which lost "
                 f"{COLLAPSE:.0%} of its income, and then runs parallel to the "
                 f"baseline. The orange one bends only at the far right, where "
                 f"the richest hundredth was multiplied by "
                 f"{1 + pair['b']:.2f} - and sits fractionally below the "
                 f"baseline everywhere else, because raising the top raised "
                 f"the total and so lowered everyone else's **share**. Twice "
                 f"the area between either curve and the diagonal is "
                 f"{pair['gini']:.6f}, for both, exactly. The coefficient is "
                 f"an area, and an area does not record where it was."),
        equal=True, figsize=(6.6, 5.4),
        path=str(IMG / f"hs103-f0-lorenz.{EXT}"))[0]

    rows = []
    for key, label, fmt in LABELS:
        row = agree[key]
        rows.append([label, fmt.format(row["collapse"]),
                     fmt.format(row["runaway"]),
                     "same" if row["same"] else f"x{row['ratio']:.2f}"])
    bold = {(i, 3) for i, (key, _, _) in enumerate(LABELS)
            if not agree[key]["same"]}

    out["f1"] = charts.table_image(
        rows,
        header=["statistic", "collapse at the bottom", "runaway at the top",
                "agree?"],
        title="Five of these cannot tell the two worlds apart",
        subtitle=("The same eight statistics computed on both populations. "
                  "The last column is a comparison at a relative tolerance of "
                  "one part in a thousand, not a judgement."),
        source="Simulated; standarderror/aggregates/dispersion.py.",
        alt=("A table of eight inequality and poverty statistics for two "
             "populations. The first five columns of values are identical; the "
             "last three differ by factors of 9.2, 2.3 and 0.24."),
        caption=(f"The percentile ratios agree because each distortion lives "
                 f"**inside** a tail and never reaches the tenth or ninetieth "
                 f"percentile. The poverty headcount agrees because both "
                 f"worlds leave the same people under the line. What separates "
                 f"them is a tail share - the poorest tenth holds "
                 f"{agree['bottom10_share']['ratio']:.1f} times more in one "
                 f"world - or a poverty measure that counts depth instead of "
                 f"heads."),
        bold_cells=bold,
        path=str(IMG / f"hs103-f1-agreement.{EXT}"))[0]

    def sensitivity(ax, m):
        ranks = np.array(sens["ranks"], dtype=float)
        vals = np.array(sens["sensitivity"], dtype=float)
        ax.scatter(ranks, vals, s=26, color=m.series[0], zorder=3,
                   label="measured by central differences")
        fit = np.polyfit(ranks, vals, 1)
        ax.plot(ranks, np.polyval(fit, ranks), lw=1.7, color=m.series[1],
                label=f"slope {sens['slope']:.3e}, algebra "
                      f"{sens['predicted_slope']:.3e}")
        ax.axhline(0.0, color=m.grid, lw=1.3)
        twin = ax.twinx()
        twin.plot(ranks, np.array(sens["incomes"]), lw=1.6, ls=":",
                  color=m.series[2])
        twin.set_ylabel("income at that rank", color=m.series[2])
        ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    out["f2"] = charts.diagram(
        sensitivity,
        title="What a unit of income does to the Gini, by the rank that gets it",
        subtitle=(f"Central differences on a population of {SENS_N}, with a "
                  f"perturbation "
                  f"{1 / sens['step_over_spacing']:,.0f} times smaller than a "
                  f"typical gap between neighbouring incomes."),
        xlabel="rank of the recipient", ylabel="change in the Gini per unit",
        source="Simulated; standarderror/aggregates/dispersion.py.",
        alt=("A straight rising line of Gini sensitivity against the "
             "recipient's rank, crossing zero once, with a dotted curve of "
             "income at that rank on a second axis."),
        caption=(f"A straight line, R-squared "
                 f"{sens['r_squared']:.10f}, with a slope matching "
                 f"2/(n²μ) to a ratio of {sens['slope_ratio']:.6f}. The dotted "
                 f"curve is what those people actually earn, and it varies by "
                 f"a factor of {sens['income_range']:.1f} across the same "
                 f"range. The coefficient prices a unit of income by the "
                 f"recipient's place in the queue and by nothing else about "
                 f"them."),
        path=str(IMG / f"hs103-f2-sensitivity.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    pair = res["pair"]
    agree = res["agree"]
    bottom = res["bottom"]

    def two_curves(panel, m):
        for name, colour in (("collapse", m.series[0]),
                             ("runaway", m.series[1])):
            p, curve = dp.lorenz(pair[name], points=60)
            panel.plot(p, curve, lw=2.4, color=colour)
        panel.plot([0, 1], [0, 1], lw=1.4, color=m.grid)
        panel.set_xlim(-0.02, 1.02)
        panel.set_ylim(-0.02, 1.02)

    def two_worlds(panel, m):
        # Ten figures; in one world the first one has nothing.
        for i in range(10):
            panel.plot([i, i], [0.15, 0.95], lw=2.2,
                       color=m.grid if i else m.series[0])
            panel.scatter([i], [1.12], s=52,
                          color=m.grid if i else m.series[0])
        panel.scatter([9.3], [1.55], s=150, color=m.series[1])
        panel.set_xlim(-1.0, 10.6)
        panel.set_ylim(-0.2, 1.95)

    def barcode(panel, m):
        panel.barh([1, 0], [agree["bottom10_share"]["runaway"],
                            agree["bottom10_share"]["collapse"]],
                   color=[m.series[1], m.series[0]], height=0.45)
        panel.set_xlim(0, 0.037)
        panel.set_ylim(-0.6, 1.6)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=3,
        headline="One coefficient, two worlds, opposite policies",
        panels=[(two_curves, f"{pair['gini']:.3f}", "the same Gini, exactly"),
                (two_worlds, "5 of 8", "headline numbers agree"),
                (barcode, f"{agree['bottom10_share']['ratio']:.1f}x",
                 "gap in the poorest tenth")],
        note=(f"The poorest tenth of this population holds "
              f"{bottom['income_share_held']:.1%} of the income, so taking all "
              f"of it moves the coefficient by "
              f"{bottom['gini_change']:.3f}. A summary weighted by income "
              f"share cannot be sensitive to a group that has none."),
        alt=("Three hand-drawn frames: two Lorenz curves over a diagonal; ten "
             "figures of whom one is marked and a large dot above the last; "
             "and two bars of very different length."),
        path=str(IMG / f"hs103-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["pair"] = s.run(f"""
        from standarderror.aggregates import dispersion as dp

        # One population. Two distortions. The collapse is fixed at 90% and the
        # runaway multiple is solved for, so that both land on the same Gini.
        pair = dp.matched_pair(a={COLLAPSE})

        print(f"base population        Gini {{pair['base_gini']:.6f}}")
        print(f"poorest tenth loses {{pair['a']:.0%}}  "
              f"Gini {{dp.gini(pair['collapse']):.6f}}")
        print(f"richest 1% x {{1 + pair['b']:.3f}}       "
              f"Gini {{dp.gini(pair['runaway']):.6f}}")
        print(f"they differ by         {{pair['disagreement']:.1e}}")
    """, expect=["they differ by"])

    out["agree"] = s.run("""
        # Every headline statistic anyone prints about a distribution, on both.
        low, high = dp.describe(pair["collapse"]), dp.describe(pair["runaway"])

        print(f"{'statistic':>24} {'collapse':>11} {'runaway':>11} {'':>7}")
        for k, v in dp.agreement(low, high).items():
            mark = "same" if v["same"] else f"x{v['ratio']:.2f}"
            print(f"{k:>24} {v['collapse']:>11.5f} {v['runaway']:>11.5f} "
                  f"{mark:>7}")
    """, expect=["statistic"])

    out["sens"] = s.run(f"""
        # And why. dG/dx_k = 2k/(n^2 mu) - c, so the sensitivity to a unit of
        # income is linear in the recipient's rank and in nothing else.
        s = dp.rank_sensitivity(dp.population(n={SENS_N}))

        print(f"fitted slope     {{s['slope']:.6e}}")
        print(f"2 / (n^2 mu)     {{s['predicted_slope']:.6e}}")
        print(f"ratio            {{s['slope_ratio']:.6f}}")
        print(f"R^2 vs rank      {{s['r_squared']:.10f}}")
        print(f"income spread over the same ranks  "
              f"{{s['income_range']:.1f}}x")
    """, expect=["R^2 vs rank"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    pair, agree = res["pair"], res["agree"]
    sens, bottom = res["sens"], res["bottom"]

    # The spine, asserted rather than trusted.
    assert pair["disagreement"] < 1e-12, pair["disagreement"]
    same = {k for k, v in agree.items() if v["same"]}
    assert same == {"gini", "p90_p10", "p50_p10", "p90_p50",
                    "headcount"}, same
    assert agree["bottom10_share"]["ratio"] > 5.0
    assert agree["gap_index"]["ratio"] < 0.4
    assert sens["r_squared"] > 0.999999
    assert abs(sens["slope_ratio"] - 1.0) < 1e-4
    assert sens["income_range"] > 4.0
    assert bottom["income_share_held"] < 0.05
    assert 0.03 < bottom["gini_change"] < 0.06

    post = Post(
        title=f"{SERIES_TAG} 3: Two Countries, One Gini, Opposite Policies",
        slug="headline-statistics-3-gini-shape",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=3,
        prerequisites=["headline-statistics-1-fertility-tempo",
                       "headline-statistics-2-wage-composition"],
        # Declared rather than reworded. Every comparison in this episode is
        # between two constructed populations whose true properties are known
        # by design; there is no prediction, so there is nothing for a naive
        # or chance-level baseline to be a baseline *of*. The auto-detect fires
        # on "the coefficient cannot separate them", which is an identity
        # rather than a performance claim.
        requires_baseline=False,
        date=POST_DATE,
        subtitle=("A scalar cannot carry a shape. The Gini is many-to-one onto "
                  "distributions, and so are the percentile ratios and the "
                  "poverty headcount - so two worlds needing opposite policies "
                  "can agree on five headline numbers at once."),
        summary=(
            "Take one lognormal population and distort it twice: the poorest "
            "tenth loses 90% of its income, or the richest hundredth has its "
            "income multiplied by 2.51. Tuned to the same Gini, both land on "
            "0.339749 exactly. And the coefficient is not the only thing that "
            "cannot separate them - p90/p10, p50/p10, p90/p50 and the poverty "
            "headcount are identical too, because each distortion lives inside "
            "a tail. Five headline numbers agree while the poorest tenth holds "
            "9.2 times more income in one world than the other. The reason is "
            "exact: dG/dx_k = 2k/(n² mu) - c, so a unit of income is priced "
            "by the recipient's rank and by nothing else about them - measured "
            "R-squared 1.0000000000 against rank, slope matching to six "
            "figures, over ranks whose incomes differ sixfold. Which bounds "
            "what the coefficient could ever have said about the bottom: the "
            "poorest tenth holds 3.3% of income here, so destroying all of it "
            "moves the Gini by 0.041, about what tripling the top percentile "
            "does. What does separate the two worlds is a tail share, or a "
            "poverty measure that counts depth: the gap index is 0.092 against "
            "0.022 on an identical headcount."),
        tags=["statistics", "economics", "inequality", "data-science",
              "public-data", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data enters any computation. Every figure and every "
            "table here is constructed from the simulation in the code shown, "
            "executed when this page was built.",
            "Published figures quoted in the prose, and used nowhere else: "
            "World Bank Gini index values of 31.2 for Austria and 31.8 for "
            "Cyprus (2022), alongside World Inequality Database pre-tax income "
            "shares of 1.89% and 2.46% for the bottom decile, as tabulated on "
            "Wikipedia's list of countries by income inequality. Those two "
            "columns are different sources on different definitions - the "
            "Gini is disposable-income based and the shares are pre-tax - so "
            "the pair is an illustration of the ordering disagreeing, not a "
            "measurement of it.",
            "Machinery: `standarderror/aggregates/dispersion.py`, tested in "
            "`tests/test_dispersion.py`, including agreement with the "
            "mean-absolute-difference form and with the Lorenz integral.",
            "Where this stops: nothing here says the Gini is wrong or should "
            "not be published. A scalar cannot carry a shape, every scalar has "
            "a blind spot somewhere, and the useful response is to know which "
            "one rather than to look for a better scalar. Sen, *On Economic "
            "Inequality* (1973), for the axioms the poverty headcount fails; "
            "Foster, Greer and Thorbecke, *Econometrica* 52 (1984), for the "
            "gap index; Atkinson, *Journal of Economic Theory* 2 (1970), for "
            "what choosing an inequality measure commits you to.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4, "
                           "scipy=1.16.3",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": (f"one lognormal population of {dp.N:,} incomes with "
                           f"a log spread of {dp.SIGMA}, and a smaller one of "
                           f"{SENS_N:,} for the sensitivity measurement, which "
                           f"needs a perturbation resolvable against the "
                           f"coefficient"),
            "determinism": ("no random numbers in any published figure beyond "
                            "the single seeded population; the runaway "
                            "multiple is solved by bisection to a tolerance of "
                            "1e-12"),
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    pair, agree = res["pair"], res["agree"]
    sens, bottom, top = res["sens"], res["bottom"], res["top"]
    A, C = PUBLISHED["austria"], PUBLISHED["cyprus"]

    post.add(
        "Two numbers that disagree about which country is worse",
        f"""Austria's Gini index was {A['gini']} in {PUBLISHED['year']} and Cyprus's was {C['gini']}. On the coefficient, Cyprus is the more unequal of the two. Its poorest tenth holds {C['bottom10']}% of income against Austria's {A['bottom10']}% — thirty percent more.

Those two columns come from different sources on different definitions, so treat the pair as an illustration rather than a measurement: the Gini figures are World Bank and disposable-income based, and the decile shares are World Inequality Database pre-tax. What the pair illustrates is that the ordering can disagree with itself, and this episode is about why it must.

It is worth being clear about what the coefficient is asked to do, because that is where the strain comes from. A Gini is used for two jobs at once. The first is to rank countries or years — is this place more unequal than that one, is it getting worse — and for that you need a scalar, which is exactly why the measure exists. The second is to justify a response: a rising coefficient is read as a case for redistribution, and a falling one as evidence that something worked. The first job needs only an ordering. The second needs the number to know **where** the inequality is, because the policy that helps a collapsed bottom and the policy that addresses a runaway top are different policies with different budgets, and one of them does nothing for the other.

A scalar can do the first job. This episode is about the fact that it cannot do the second, and about how far apart two worlds sharing one coefficient can be.

Episode 1 was a summary that describes nobody. Episode 2 was a summary that describes somebody, just not the same somebody twice. This one describes something real, and does not say what.""")

    post.add(
        "One population, two distortions, one coefficient",
        f"""Take a single lognormal income distribution and break it twice, in opposite places.

**Collapse.** The poorest tenth loses {COLLAPSE:.0%} of its income. Nobody else is touched.

**Runaway.** The richest hundredth has its income multiplied. Nobody else is touched.

Fix the collapse and solve for the multiple that lands on the same Gini.""")

    post.add(
        "",
        f"""{snip['pair'].markdown()}

The poorest tenth losing {COLLAPSE:.0%} of everything and the richest hundredth multiplying by {1 + pair['b']:.2f} are, to the Gini coefficient, the same event. Not approximately: the two agree to {pair['disagreement']:.1e}, which is zero at this precision because the multiple was solved for.

That much is a construction and it proves nothing on its own — any one-parameter family hits any reachable target. The interesting part is what *else* fails to separate them.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "",
        f"""{snip['agree'].markdown()}

Five of the eight agree. The Gini by construction; the three percentile ratios because each distortion lives **inside** a tail and never reaches the tenth or the ninetieth percentile; and the poverty headcount because both worlds leave exactly the same people below half the median — the collapse pushes them further under it without pushing anyone new across.

The three that see it are a bottom-decile share ({agree['bottom10_share']['ratio']:.1f} times larger in the runaway world), a top-percentile share ({agree['top1_share']['ratio']:.1f} times larger), and the poverty **gap** index, which is {agree['gap_index']['collapse']:.3f} against {agree['gap_index']['runaway']:.3f}. The people under the line are {agree['mean_shortfall']['collapse']:.0%} below it in one world and {agree['mean_shortfall']['runaway']:.0%} below it in the other, and only a measure of depth notices.

So the recommendation cannot be "use the decile ratio instead". It has to be a tail share or the whole curve.""",
        level=3,
        figures=[figs["f1"]])

    post.add(
        "Why the coefficient cannot see it",
        r"""This is not an accident of the construction. Write the Gini in its rank-weighted form, on sorted incomes:

$$
G = \frac{2 \sum_i i x_i}{n^2 \mu} - \frac{n + 1}{n}
$$

and differentiate with respect to one person's income. The first term gives 2*k*/(*n*²*mu*); the second is where *mu* itself moves, and it contributes the same amount whichever *k* you perturb. So

$$
\frac{\partial G}{\partial x_k} = \frac{2k}{n^2 \mu} - c
$$

with *c* independent of *k*. **The sensitivity is linear in the recipient's rank, with the same slope everywhere, and depends on nothing else about them** — not on their income, and not on what the money would do for them.""")

    post.add(
        "",
        f"""{snip['sens'].markdown()}

An R-squared of {sens['r_squared']:.10f} against rank, and a fitted slope matching 2/(*n*²*mu*) to a ratio of {sens['slope_ratio']:.6f}. Over the same ranks, what those people earn varies by a factor of {sens['income_range']:.1f}.

A unit of income handed to someone at the fifth percentile and the same unit handed to someone at the ninetieth move the coefficient by amounts that differ only through their ranks. The first one changes a life and the second is a rounding error in a portfolio, and the Gini prices them by their positions in a queue.

A note on how not to measure this, because I did it the wrong way first. The obvious experiment is to move a fixed sum some number of ranks down the distribution and watch the coefficient. That measures something else: a sum worth one percent of the mean is about three thousand times a typical gap between neighbouring incomes at this sample size, so it vaults the recipient over thousands of people and the rank distance in the formula is not the rank distance you set. The measurement came out **non-monotone in the distance**, which is what sent me back to the algebra. The perturbation has to be small against the local spacing — the figure above uses one {1 / sens['step_over_spacing']:,.0f} times smaller — and then the identity is exact.""",
        level=3,
        figures=[figs["f2"]])

    post.add(
        "What that bounds",
        f"""The rank-linearity has a consequence worth stating on its own, because it puts a ceiling on what the coefficient could ever have told you about the poor.

In this population the poorest tenth holds {bottom['income_share_held']:.1%} of total income. So there is only {bottom['income_share_held']:.1%} of income down there for any redistribution to move, and taking **all** of it — one person in ten with nothing at all — moves the Gini from {bottom['gini_before']:.3f} to {bottom['gini_after']:.3f}, a change of {bottom['gini_change']:.3f}.

For comparison, multiplying the top percentile's income by {top[2]['multiple']:.0f} moves it {top[2]['gini'] - bottom['gini_before']:.3f}. Complete destitution for a tenth of the population and a tripling at the very top are, on this measure, the same size of event.

This is not a defect that a correction fixes. A summary weighted by income share cannot be sensitive to a group that holds almost none of it, and the poorest decile of any unequal country holds almost none of it. If the bottom is what you care about, the coefficient was never going to be the instrument.""")

    post.add(
        "What to do instead, and what it costs",
        """There is no better scalar. That is the actual finding, and it is worth being blunt about because the literature contains a long shelf of proposed replacements and every one of them is a scalar.

Atkinson's measures pick an inequality-aversion parameter and are explicit that the parameter is a value judgement rather than a measurement. The Theil index is decomposable between groups, which is genuinely useful and does not make it one-to-one. The Palma ratio deliberately looks only at the top decile over the bottom four, which fixes this episode's example and breaks on a different one. Every scalar throws away a shape; choosing one chooses which shape you are willing to lose.

So the practical rule is two lines long. **Publish the Lorenz curve, or a small number of quantile shares, next to any coefficient** — three numbers, the bottom decile's share, the top decile's and the top percentile's, would have separated the two worlds in this episode and take one line of a table. And **say which end of the distribution your question is about before choosing the measure**, because the measure decides which end it can see, and it decides it silently.

The cost of the honest version is that you no longer get a single number to rank countries by, which is exactly what a coefficient is used for. That is not a solvable tension. It is the price of the shape being real.""")

    post.add(
        "What to keep",
        f"""1. The Gini is many-to-one onto distributions. The poorest tenth losing {COLLAPSE:.0%} and the richest hundredth multiplying by {1 + pair['b']:.2f} land on the same coefficient exactly.
2. So do p90/p10, p50/p10, p90/p50 and the poverty headcount — five headline numbers agreeing while the poorest tenth holds {agree['bottom10_share']['ratio']:.1f} times more income in one world.
3. Because `dG/dx_k = 2k/(n²μ) − c`: a unit of income is priced by the recipient's rank, linearly, and by nothing else. R-squared {sens['r_squared']:.6f}, over ranks whose incomes differ {sens['income_range']:.1f}-fold.
4. Which bounds the coefficient's reach at the bottom. The poorest tenth holds {bottom['income_share_held']:.1%} of income, so destroying all of it moves the Gini {bottom['gini_change']:.3f} — about what tripling the top percentile does.
5. The poverty **headcount** has the same defect in miniature: it cannot see the poor getting poorer, because nobody crosses the line. The gap index can, at {agree['gap_index']['collapse']:.3f} against {agree['gap_index']['runaway']:.3f}.
6. There is no better scalar, only a different blind spot. Publish a curve or a few shares beside the coefficient, and decide which end of the distribution the question is about first.""")

    post.add(
        "What the three episodes have in common",
        """This closes the track, and the three failures turn out to be one question asked three ways.

Episode 1's total fertility rate was a **synthetic construct**: it described nobody, so a change in the timing of births moved it while no cohort's family size moved at all. Episode 2's median wage described somebody, but not the same somebody twice — the **population changed** between the prints, and the difference carried a term with no upper bound and no label. This episode's Gini describes something real and complete, and is **many-to-one**: it cannot say which of two opposite worlds produced it.

Those are three distinct mechanisms, and none of them is a data-quality problem. The numbers were correct in all three cases, published by competent institutions, with no revisions pending. What went wrong each time was the step from the number to the sentence a reader forms about it.

Which gives the question worth carrying away, and it is not about demography or wages or inequality. For any summary you rely on: **what are two states of the world this number cannot tell apart, and would I act differently in them?** In episode 1 the two states were "families got smaller" and "births moved later". In episode 2 they were "pay fell" and "the workforce changed". Here they are "the bottom collapsed" and "the top ran away". Each pair took under an hour to construct, and in each pair the two states call for different budgets.

If a summary you use every week survives that question, it is doing its job. If it does not, the honest move is not to stop using it — it is to publish the companion that breaks the tie, which in all three episodes was a small and cheap thing: a mean age by birth order, a matched-individual median, three quantile shares.""")

    post.add(
        "Exercise",
        """Take your own country's published income decile shares — most statistical offices publish them, and they are one table. Compute the Gini from them, then construct a second set of decile shares with the same Gini and a bottom decile half as large. You will not need optimisation; a two-parameter adjustment of the top and bottom deciles has enough freedom.

Now write one sentence describing each of the two countries you have just produced, and notice that the sentences call for different budgets.

Then the part that generalises past income. Find a scalar summary you rely on in your own work — a single accuracy figure, a Sharpe ratio, an average latency, one AUC — and ask the same question of it: what are two states of the world that it cannot tell apart, and would you act differently in them? If you can construct such a pair in ten minutes, the number needs a companion, and this episode is really about that rather than about the Gini.""")

    post.hero = figs["hero"]
    return post


def main() -> Post:
    return build()


if __name__ == "__main__":
    r = compute()
    p = r["pair"]
    print(f"base {p['base_gini']:.6f} -> both {p['gini']:.6f} "
          f"(a={p['a']:.2f}, x{1+p['b']:.3f}), differ {p['disagreement']:.1e}")
    for k, v in r["agree"].items():
        mark = "same" if v["same"] else f"x{v['ratio']:.2f}"
        print(f"  {k:>16} {v['collapse']:>11.5f} {v['runaway']:>11.5f} {mark}")
    s = r["sens"]
    print(f"\nsensitivity: slope {s['slope']:.4e} vs {s['predicted_slope']:.4e} "
          f"(ratio {s['slope_ratio']:.6f}), R2 {s['r_squared']:.10f}")
    print("bottom:", {k: round(v, 5) for k, v in r["bottom"].items()})
