"""ML 49: Target the Effect, Not the Risk.

The forty-ninth episode of *Machine Learning, Taught Through What
Breaks*. "Give the intervention to the people the model says are most at
risk." Measured on a simulated retention offer with a randomised trial
behind it, a budget for 20% of customers, and churn prevented computed
from the true probabilities:

* Offering it to the 20% with the highest predicted churn prevented about
  21 churns per 1,000 customers, about the same as offering it at random (22).
* Ranking by the predicted reduction in churn, the difference of two
  models fitted to the trial's arms, prevented 39 to 50 per 1,000 depending
  on the model and the trial's size; the best possible was 55.
* Among the highest-risk fifth, 11% of customers were made more likely to
  churn by the offer, against 6% overall.

Run: `standarderror run ml149_uplift --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import decide
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

STRATS = ("random", "highest risk (boosted)", "highest risk (logistic)",
          "largest effect (boosted)", "largest effect (logistic)", "true effect")


def compute() -> dict:
    u = decide.uplift_targeting()
    return {"u": u, "k": {r["n"]: r for r in u["rows"]},
            "bud": decide.uplift_budgets()}


# ------------------------------------------------------------------ figures

def _bars(ax, m, k):
    x = np.arange(len(STRATS))
    w = 0.38
    for j, n in enumerate((2000, 20_000)):
        ax.bar(x + (j - 0.5) * w, [k[n][s] for s in STRATS], width=w,
               yerr=[k[n]["sd"][s] for s in STRATS],
               color=m.series[j + 1], label=f"trial of {n:,} customers")
    ax.set_xticks(x)
    ax.set_xticklabels(["random", "risk,\nboosted", "risk,\nlogistic",
                        "effect,\nboosted", "effect,\nlogistic", "true\neffect"],
                       fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def figures(res: dict) -> dict:
    out: dict = {}
    u, k = res["u"], res["k"]

    out["f0"] = charts.diagram(
        lambda ax, m: _bars(ax, m, k),
        title="Whom to make the offer to",
        subtitle=("Churns prevented per 1,000 customers by offering a "
                  "retention deal to 20% of 100,000, chosen six ways from a "
                  "randomised trial of 2,000 or 20,000; means and standard "
                  "deviations over 5 trials."),
        xlabel="",
        ylabel="churns prevented per 1,000 customers",
        figsize=(7.6, 4.6),
        source="Simulated; standarderror/ml/decide.py.",
        alt=("Bars: random and highest-risk targeting both prevent about 21 "
             "churns per 1,000; targeting by predicted effect prevents 39 to "
             "50; the true best is 55."),
        caption=(f"Targeting the highest predicted risk prevented "
                 f"{k[20_000]['highest risk (boosted)']:.1f} per 1,000; "
                 f"random targeting {k[20_000]['random']:.1f}; the predicted "
                 f"effect **{k[20_000]['largest effect (logistic)']:.1f}**."),
        path=str(IMG / f"ml149-f0-targeting.{EXT}"))[0]

    body = [[s, f"{k[2000][s]:.1f}", f"± {k[2000]['sd'][s]:.1f}",
             f"{k[20_000][s]:.1f}", f"± {k[20_000]['sd'][s]:.1f}"] for s in STRATS]
    out["f1"] = charts.table_image(
        body,
        header=["offer to the top 20% by", "trial of 2,000", "sd",
                "trial of 20,000", "sd"],
        title="Churn prevented, by targeting rule",
        subtitle=(f"Per 1,000 customers, from the true probabilities. Without "
                  f"any offer {u['base_churn']:.1%} churn; offering it to "
                  f"everyone would prevent {u['mean_gain'] * 1000:.0f} per "
                  f"1,000 and make {u['harmed_share']:.1%} of customers more "
                  f"likely to leave."),
        source="Simulated; standarderror/ml/decide.py.",
        alt=("A six-row table of churns prevented per 1,000 customers, for "
             "two trial sizes, with standard deviations."),
        caption=("'Risk' ranks by a model of the trial's untreated arm. "
                 "'Effect' ranks by the difference between models of the "
                 "untreated and treated arms."),
        align="lrrrr",
        path=str(IMG / f"ml149-f1-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    u, k = res["u"], res["k"]
    r = k[20_000]

    def p1(panel, m):
        panel.bar([0, 1], [r["random"], r["highest risk (boosted)"]],
                  color=[m.grid, m.series[1]])
        panel.set_ylim(0, 60)

    def p2(panel, m):
        panel.bar([0, 1, 2], [r["highest risk (boosted)"],
                              r["largest effect (logistic)"], r["true effect"]],
                  color=[m.series[1], m.series[2], m.ink])
        panel.set_ylim(0, 60)

    def p3(panel, m):
        panel.bar([0, 1], [u["harmed_share"], u["top_risk_harmed"]],
                  color=[m.grid, m.series[1]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=49,
        headline="Risk is not effect",
        panels=[(p1, f"{r['highest risk (boosted)']:.0f}", "prevented, by risk"),
                (p2, f"{r['largest effect (logistic)']:.0f}", "prevented, by effect"),
                (p3, f"{u['top_risk_harmed']:.0%}", "harmed among top risk")],
        note=("The people most likely to leave are not the people an offer "
              "would keep. A model of the outcome ranks the first; only a "
              "model of the difference the offer makes ranks the second."),
        alt=("Three hand-drawn frames of bars: two equal bars; three "
             "rising bars; a short bar and a taller one."),
        path=str(IMG / f"ml149-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["main"] = s.run("""
        from standarderror.ml import decide

        u = decide.uplift_targeting()   # budget: 20% of customers
        for row in u["rows"]:
            print(f"trial of {row['n']:,} customers:")
            for k, v in row.items():
                if k not in ("n", "sd"):
                    print(f"   {k:26s} {v:5.1f} churns prevented per 1,000")
        """)
    out["harm"] = s.run("""
        print("made more likely to churn by the offer:")
        print(f"   {u['harmed_share']:.1%} of all customers, "
              f"{u['top_risk_harmed']:.1%} of the highest-risk fifth")
        print(f"mean reduction in churn probability: all customers "
              f"{u['mean_gain']:.3f}, highest-risk fifth {u['top_risk_mean_gain']:.3f}")
        """)
    out["bud"] = s.run("""
        for b, v in decide.uplift_budgets().items():   # trial of 20,000
            print(f"offer to {b:4.0%}: by risk {v['risk']:5.1f}, by effect "
                  f"{v['effect']:5.1f}, at random {v['random']:5.1f}, "
                  f"best {v['true']:5.1f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    u, k, bud = res["u"], res["k"], res["bud"]
    a, b = k[2000], k[20_000]
    risk = [r[s] for r in (a, b) for s in ("highest risk (boosted)",
                                            "highest risk (logistic)")]
    eff = [r[s] for r in (a, b) for s in ("largest effect (boosted)",
                                           "largest effect (logistic)")]

    # The spine, asserted rather than trusted.
    assert all(abs(x - a["random"]) < 2.5 for x in risk)
    assert 20 < min(risk) and max(risk) < 22.5
    assert b["highest risk (logistic)"] < b["random"]
    assert 38 < min(eff) and max(eff) < 51 and b["true effect"] > 55
    assert all(r["largest effect (logistic)"] > r["largest effect (boosted)"]
               for r in (a, b))
    assert all(b[s] > a[s] for s in ("largest effect (boosted)",
                                     "largest effect (logistic)"))
    assert u["top_risk_mean_gain"] < u["mean_gain"]
    assert u["top_risk_harmed"] > 1.7 * u["harmed_share"]
    assert abs(a["random"] - 0.2 * u["mean_gain"] * 1000) < 0.2
    assert bud[0.05]["risk"] < bud[0.05]["random"] / 3
    assert bud[0.1]["risk"] < bud[0.1]["random"] / 2
    assert bud[0.5]["risk"] > bud[0.5]["random"] + 15
    assert all(bud[x]["effect"] > bud[x]["risk"] for x in (0.05, 0.1, 0.2, 0.5))
    assert abs(bud[1.0]["risk"] - bud[1.0]["true"]) < 1e-9

    post = Post(
        title=f"{SERIES_TAG} 49: Target the Effect, Not the Risk",
        slug="ml-49-uplift",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=49,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A model of who will churn is not a model of whom an offer "
                  "will keep. Targeting the highest risk did no better than "
                  "chance; targeting the predicted effect did about twice as well."),
        summary=(
            f"A simulated retention offer, a randomised trial behind it and "
            f"a budget for 20% of customers. Offering it to the highest "
            f"predicted churners prevented {min(risk):.1f} to "
            f"{max(risk):.1f} churns per 1,000 customers, about what random "
            f"targeting prevented ({a['random']:.1f}). Ranking by the "
            f"predicted reduction in churn, the difference of two models "
            f"fitted to the trial's arms, prevented {min(eff):.1f} to "
            f"{max(eff):.1f}; the best possible was {b['true effect']:.1f}. "
            f"Among the highest-risk fifth, {u['top_risk_harmed']:.0%} of "
            f"customers were made more likely to leave by the offer, against "
            f"{u['harmed_share']:.0%} overall."),
        tags=["machine-learning", "data-science", "statistics",
              "causal-inference", "uplift-modeling", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: five standard-normal features; churn without the "
            "offer logistic in −1 + 1.5 x_1 + x_2; the offer adds "
            "−1.6 exp(−x_1²) to the log-odds, plus +0.8 for customers with "
            "x_3 > 1. Randomised trials of 2,000 and 20,000 customers, half "
            "offered (seeds 0 to 4); a target population of 100,000 (seed "
            "999).",
            "Machinery: `standarderror/ml/decide.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's "
            "HistGradientBoostingClassifier (200 rounds, rate 0.05) and "
            "LogisticRegression, one model per arm (a T-learner).",
            "Where this stops: Radcliffe and Surry, \"Real-world uplift "
            "modelling with significance-based uplift trees\" (2011); "
            "Künzel, Sekhon, Bickel and Yu, \"Metalearners for estimating "
            "heterogeneous treatment effects using machine learning\", "
            "*PNAS* (2019); Athey and Imbens, \"Recursive partitioning for "
            "heterogeneous causal effects\", *PNAS* (2016).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("seeds as listed; about half a minute of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The people most at risk",
        f"""Episode 48 found that the features a model leans on need not be levers. When the lever is known, a treatment, an offer, an intervention, the remaining decision is whom to apply it to, and the model asked to decide is usually a risk model: who is most likely to churn, default, be readmitted. The sentence under test is the obvious policy that follows: *give the intervention to the people the model says are most at risk*.

The simulation is a retention offer. Without it, {u['base_churn']:.1%} of customers churn, and the chance rises with two features. The offer's effect differs from person to person. It helps most for customers of middling risk; it does little for those almost certain to stay, who will stay anyway, and little for those almost certain to leave, whom one offer will not keep. And for customers with a high value of a third feature whose risk is far from the middle, about one customer in fifteen, it backfires: being contacted makes them more likely to go. Radcliffe and Surry's names for the four groups are the persuadables, the sure things, the lost causes and the do-not-disturbs, also called sleeping dogs.

A randomised trial offers the deal to half of 2,000 or 20,000 customers. From it, models choose which 20% of a new population of 100,000 to offer it to, and the churn prevented is computed from the true probabilities:

{snip['main'].markdown()}

Offering the deal to the fifth with the highest predicted churn prevented between **{min(risk):.1f}** and {max(risk):.1f} churns per 1,000 customers. Choosing the fifth at random prevented {a['random']:.1f}. With the logistic risk model and the large trial, the risk-targeted offer did slightly worse than random.""",
        figures=[figs["f0"]])

    post.add(
        "Why risk is the wrong ranking",
        f"""The offer's value for a customer is the difference between two probabilities, churn without it and churn with it. A risk model estimates the first. High risk is correlated with a large difference only if the offer works best for the people most likely to leave, and here it does not:

{snip['harm'].markdown()}

The highest-risk fifth holds more of the customers the offer harms. On average the offer reduced their churn probability by {u['top_risk_mean_gain']:.3f}, less than its average over all customers, {u['mean_gain']:.3f}, and {u['top_risk_harmed']:.1%} of them were made more likely to leave, against {u['harmed_share']:.1%} overall. Ranking by risk spends the budget on customers moved slightly less than average, and harmed more often.""",
        figures=[figs["f1"]])

    post.add(
        "Ranking by the effect",
        f"""The trial makes a better ranking possible. Fit one model to the customers who were offered the deal and another to those who were not, and rank new customers by the difference between their two predictions: Künzel and colleagues call this the T-learner, the simplest of several meta-learners for heterogeneous effects. It prevented **{b['largest effect (logistic)']:.1f}** churns per 1,000 with logistic models and the large trial, more than twice the risk ranking, against {b['true effect']:.1f} for ranking by the true effect.

Estimating a difference is harder than estimating a level: the noise of two separate fits adds up in their difference. With the smaller trial the effect ranking fell to {a['largest effect (logistic)']:.1f} with logistic models, varying from trial to trial with a standard deviation of {a['sd']['largest effect (logistic)']:.1f}, and to {a['largest effect (boosted)']:.1f} with boosted trees. The logistic T-learner beat the boosted one at both trial sizes ({b['largest effect (logistic)']:.1f} against {b['largest effect (boosted)']:.1f} with the large trial), even though the true effect is not linear in the features; a plausible reason, not isolated here, is that two simpler models leave less noise in their difference than two flexible ones. Neither came closer than {b['true effect'] - b['largest effect (logistic)']:.1f} churns per 1,000 to ranking by the true effect. Methods that model the difference directly, such as Athey and Imbens's causal trees, exist for this reason. And the whole approach rests on the trial: without random assignment, the difference between offered and not-offered customers would mix the offer's effect with the reasons they were offered it, which is episodes 39 and 48 at once.""")

    post.add(
        "How the budget changes the answer",
        f"""All of this was at one budget, a fifth of the customers. The ranking matters most when the budget is small and stops mattering when it covers everyone, so the same comparison was run at five budgets, with logistic models and the large trial:

{snip['bud'].markdown()}

With only 5% of customers reachable, offering the deal to the highest predicted risk prevented **{bud[0.05]['risk']:.1f}** churns per 1,000, against {bud[0.05]['random']:.1f} for a random 5% and {bud[0.05]['effect']:.1f} for the effect ranking: the riskiest twentieth gained on average only {bud[0.05]['risk'] / 50:.3f} in churn probability from the offer, against {u['mean_gain']:.3f} across all customers, and the risk ranking did several times worse than chance. At half the customers the risk ranking finally beat random targeting ({bud[0.5]['risk']:.1f} against {bud[0.5]['random']:.1f}), because a list that long reaches far enough down the risk scale to include the middling customers the offer helps. Offering it to everyone prevents {bud[1.0]['true']:.1f} whatever the ranking.

So the cost of ranking by risk is largest exactly when targeting matters most, under a tight budget, and it is a cost relative to doing nothing clever at all.""")

    post.add(
        "Where this breaks",
        """**A simulated effect with a clear shape.** The offer's effect depends on risk in a way that makes risk targeting fail. Where the effect is largest for the highest-risk people, risk targeting is close to right, and a trial is the only way to know which case holds.

**Known true probabilities.** Churn prevented was computed from the simulation's probabilities. In practice the value of a targeting rule has to be estimated from another trial, with its own noise.

**Budgets measured on one trial size.** The budget sweep used the large trial and logistic models; with a smaller trial the effect ranking is noisier at every budget.

**One-shot.** The offer is made once. Repeated offers, and customers who learn to expect them, are a different problem.""")

    post.add(
        "What to keep",
        f"""1. Offering the deal to the highest predicted churners prevented {min(risk):.1f} to {max(risk):.1f} churns per 1,000 customers, about the same as random targeting ({a['random']:.1f}).

2. The highest-risk fifth held more customers the offer could not help or would harm: {u['top_risk_harmed']:.0%} of them were made more likely to leave, against {u['harmed_share']:.0%} overall.

3. Ranking by the predicted difference the offer makes, from models of a randomised trial's two arms, prevented up to {max(eff):.1f}, against a best possible {b['true effect']:.1f}; it needed the larger trial to get there.

4. Target an intervention by its estimated effect, which needs a randomised comparison, and measure the targeting rule against random targeting before trusting it.""")

    post.add(
        "Exercise",
        """For an intervention targeted by a risk score, find whether a randomised comparison exists, even a small holdout of people who qualified and were not treated.

Split the scored population into fifths by risk and, within each, compare outcomes between treated and untreated. If the treatment's effect is not largest in the top fifth, the risk score is choosing the wrong people, and a model of the difference is worth building. Report every targeting rule as churn prevented per thousand customers next to what random targeting of the same number would prevent, so the comparison that matters is always in view.""")

    post.add(
        "Next",
        """Every decision in this arc changes the data the next model will learn from: the flagged cases, the offered customers, the changed levers. Episode 50, the last of the arc, follows a decision rule that learns only from what its own decisions let it see.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
