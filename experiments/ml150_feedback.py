"""ML 50: A Rule That Chooses Its Own Data Learns Its Own Choices.

The fiftieth episode of *Machine Learning, Taught Through What Breaks*,
closing Arc X. "A model retrained on the data it collects corrects itself
over time." Measured on a simulated inspection programme: two districts,
ten inspections a day for a year, and violations found only where
inspectors go:

* Sending inspectors where more violations had been found sent every one
  of them to a single district by the year's end, in every one of 500
  runs; with the other district's true rate 20% higher, the wrong district
  in 46% of runs.
* Splitting visits in proportion to violations found settled near the
  truth only by chance: district A's final share ranged from 0.07 to 0.75 across the
  middle 80% of runs, and was within 0.1 of the true 0.45 in 21%.
* Dividing violations by visits made fixed it: within 0.1 in every run.
  Sending all visits to the higher rate per visit still ended on the
  worse district in 30% of runs; keeping 10% of visits for exploration cut
  that to 8%.

Run: `standarderror run ml150_feedback --publish`
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

RULES = ("greedy", "proportional", "per visit", "greedy per visit",
         "greedy per visit, explore")
LABELS = {"greedy": "most found so far",
          "proportional": "in proportion to found",
          "per visit": "in proportion to found per visit",
          "greedy per visit": "most found per visit",
          "greedy per visit, explore": "most per visit, 10% explore"}


def compute() -> dict:
    eq = decide.feedback_loop()
    uneq = decide.feedback_loop(rates=(0.10, 0.12))
    return {"eq": eq, "un": uneq}


# ------------------------------------------------------------------ figures

def _spread(ax, m, d):
    y = np.arange(len(RULES))[::-1]
    for yy, r in zip(y, RULES):
        v = d[r]
        ax.plot([v["q10"], v["q90"]], [yy, yy], color=m.series[1], lw=5,
                solid_capstyle="butt")
        ax.scatter([v["mean"]], [yy], color=m.ink, zorder=3, s=30)
    ax.axvline(d["true_share"], color=m.series[2], ls="--", lw=1.5)
    ax.text(d["true_share"] + 0.01, y[-1] - 0.45, "true share", fontsize=8.5,
            color=m.series[2])
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[r] for r in RULES], fontsize=8.5)
    ax.set_xlim(-0.02, 1.02)


def figures(res: dict) -> dict:
    out: dict = {}
    eq, un = res["eq"], res["un"]

    out["f0"] = charts.diagram(
        lambda ax, m: _spread(ax, m, un),
        title="Where the inspectors end up after a year",
        subtitle=("District A's share of the day's inspections on day 365, "
                  "over 500 simulated years; bars span the 10th to 90th "
                  "percentile, dots the mean. A's true violation rate is "
                  "0.10 per visit, B's 0.12."),
        xlabel="district A's share of inspections on the last day",
        ylabel="",
        figsize=(7.6, 4.4),
        source="Simulated; standarderror/ml/decide.py.",
        alt=("Horizontal bars. The two rules that send every visit to the "
             "leader span 0 to 1; splitting by violations found spans 0.07 "
             "to 0.75; splitting by violations per visit is a narrow bar at "
             "the true share of 0.45; with exploration, a point at 0.05."),
        caption=(f"Splitting visits by violations found left A's share "
                 f"between {un['proportional']['q10']:.2f} and "
                 f"{un['proportional']['q90']:.2f} in the middle 80% of runs; "
                 f"dividing by visits made, between "
                 f"{un['per visit']['q10']:.2f} and "
                 f"**{un['per visit']['q90']:.2f}**."),
        path=str(IMG / f"ml150-f0-spread.{EXT}"))[0]

    body = []
    for r in RULES:
        body.append([LABELS[r], f"{eq[r]['locked']:.0%}", f"{eq[r]['within_10']:.0%}",
                     f"{un[r]['locked']:.0%}", f"{un[r]['within_10']:.0%}",
                     f"{un[r]['mostly_a']:.0%}"])
    out["f1"] = charts.table_image(
        body,
        header=["visits allocated by", "equal: 95%+ in one", "equal: within 0.1",
                "B higher: 95%+ in one", "B higher: within 0.1",
                "B higher: mostly to A"],
        title="Five allocation rules, two worlds",
        subtitle=("Share of 500 simulated years ending with at least 95% of "
                  "visits in one district, with A's share within 0.1 of its "
                  "true share, and with most visits in A, the district with "
                  "the lower rate."),
        source="Simulated; standarderror/ml/decide.py.",
        alt=("A five-row table. The two rules that send every visit to the "
             "leader end with all visits in one district in every run; the per-visit rule is "
             "within 0.1 of the truth in every run; exploration cuts the "
             "share of runs ending mostly in the worse district."),
        caption=("'Equal': both districts have rate 0.10. 'B higher': A 0.10, "
                 "B 0.12, so the true share for A, in proportion to rates, "
                 f"is {un['true_share']:.2f}."),
        align="lrrrrr",
        path=str(IMG / f"ml150-f1-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    un = res["un"]

    def p1(panel, m):
        panel.bar([0, 1], [1 - un["greedy"]["mostly_a"], un["greedy"]["mostly_a"]],
                  color=[m.series[2], m.series[1]])

    def p2(panel, m):
        for k, r in enumerate(("proportional", "per visit")):
            v = un[r]
            panel.plot([v["q10"], v["q90"]], [k, k], color=m.series[k + 1], lw=6)
        panel.set_xlim(0, 1)
        panel.set_ylim(-0.5, 1.5)

    def p3(panel, m):
        panel.bar([0, 1], [un["greedy per visit"]["mostly_a"],
                           un["greedy per visit, explore"]["mostly_a"]],
                  color=[m.series[1], m.series[2]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=50,
        headline="Data it chose",
        panels=[(p1, f"{un['greedy']['mostly_a']:.0%}", "locked on worse district"),
                (p2, f"{un['proportional']['within_10']:.0%}", "near truth, by counts"),
                (p3, f"{un['greedy per visit, explore']['mostly_a']:.0%}",
                 "wrong, with exploration")],
        note=("A rule that decides where to look, and learns from what it "
              "finds there, learns about its own decisions. It needs to "
              "divide by its effort, and to keep looking where it thinks "
              "nothing is."),
        alt=("Three hand-drawn frames: two bars of similar height; a long "
             "line below a short one; a tall bar and a short one."),
        path=str(IMG / f"ml150-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["eq"] = s.run("""
        from standarderror.ml import decide

        eq = decide.feedback_loop()                   # equal rates, 0.10
        for rule in ("greedy", "proportional", "per visit"):
            v = eq[rule]
            print(f"{rule:12s} A's final share: 10th-90th percentile "
                  f"{v['q10']:.2f}-{v['q90']:.2f}, all in one district "
                  f"{v['locked']:.0%}")
        """)
    out["un"] = s.run("""
        un = decide.feedback_loop(rates=(0.10, 0.12))  # B's rate 20% higher
        for rule, v in un.items():
            if rule != "true_share":
                print(f"{rule:26s} mostly in A (worse): {v['mostly_a']:.0%}, "
                      f"within 0.1 of truth: {v['within_10']:.0%}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    eq, un = res["eq"], res["un"]
    g, p, pv = un["greedy"], un["proportional"], un["per visit"]
    gv, ge = un["greedy per visit"], un["greedy per visit, explore"]

    # The spine, asserted rather than trusted.
    assert eq["greedy"]["locked"] == 1.0 and g["locked"] == 1.0
    assert 0.4 < g["mostly_a"] < 0.5
    assert p["q10"] < 0.1 and p["q90"] > 0.7 and p["within_10"] < 0.25
    assert eq["proportional"]["within_10"] < 0.25
    assert pv["within_10"] == 1.0 and eq["per visit"]["within_10"] == 1.0
    assert abs(pv["mean"] - un["true_share"]) < 0.005
    assert 0.25 < gv["mostly_a"] < 0.35 and ge["mostly_a"] < 0.1
    assert p["mean"] < un["true_share"] - 0.08

    post = Post(
        title=(f"{SERIES_TAG} 50: A Rule That Chooses Its Own Data Learns Its "
               f"Own Choices"),
        slug="ml-50-feedback",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=50,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("When a rule decides where to look and learns only from "
                  "what it finds there, more data does not correct it. It "
                  "has to account for its own effort, and keep looking where "
                  "it expects nothing."),
        summary=(
            f"Two simulated districts, ten inspections a day for a year, "
            f"violations found only where inspectors go; district B's true "
            f"rate 20% above A's. Sending inspectors where more violations "
            f"had been found put every visit in one district by the year's "
            f"end in all 500 runs, the lower-rate district in "
            f"{g['mostly_a']:.0%}. Splitting visits in proportion to "
            f"violations found left A's share between {p['q10']:.2f} and "
            f"{p['q90']:.2f} in the middle 80% of runs, within 0.1 of the true {un['true_share']:.2f} in "
            f"{p['within_10']:.0%} of runs. Dividing by visits made put it "
            f"within 0.1 in every run. Sending all visits to the higher rate "
            f"per visit ended mostly in the worse district in "
            f"{gv['mostly_a']:.0%} of runs; with 10% of visits kept for "
            f"exploration, {ge['mostly_a']:.0%}."),
        tags=["machine-learning", "data-science", "statistics",
              "feedback-loops", "decision-making", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: two districts with violation rates per inspection of "
            "0.10 and 0.10, or 0.10 and 0.12; ten inspections a day for 365 "
            "days, split binomially by the rule's share; every rule starts "
            "from one violation found and one visit made in each district. "
            "500 runs per rule and world (seed 0).",
            "Machinery: `standarderror/ml/decide.py`, tested in "
            "`tests/test_ml.py`; numpy only.",
            "Where this stops: Ensign, Friedler, Neville, Scheidegger and "
            "Venkatasubramanian, \"Runaway feedback loops in predictive "
            "policing\", *FAT** (2018); Lum and Isaac, \"To predict and "
            "serve?\", *Significance* (2016); Sutton and Barto, "
            "*Reinforcement Learning: An Introduction* (2018), on "
            "exploration; episode 39 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("seed 0 per world; about thirty seconds of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Looking where you looked before",
        f"""Episode 39 trained a model on outcomes that existed only because an earlier decision allowed them. This arc's decisions all do that: the cases flagged, the levers pulled and the customers offered a deal produce the data the next model is trained on. The comfortable assumption is that the loop corrects itself, the sentence under test: *a model retrained on the data it collects corrects itself over time*.

The setting is deliberately small. An inspection programme has ten inspectors a day and two districts. Each inspection finds a violation with a probability that is the district's true rate, and a violation is recorded only where an inspector goes. Each morning a rule splits the day's ten visits using everything recorded so far. In the first world the two districts are identical, both at 0.10 per visit. In the second, district B's rate is 0.12, 20% higher. Each rule runs for a year, 500 times.

{snip['eq'].markdown()}

The most natural rule sends the day's inspectors to the district with more violations on record. In every one of 500 years, by the last day every inspector was going to one district, and which one was decided by the luck of the first day or two: in the identical world it was A in {eq['greedy']['mostly_a']:.0%} of years. A district that falls behind is not visited, so it cannot catch up. Lum and Isaac had shown two years earlier, by applying a predictive-policing algorithm to Oakland's drug-arrest records, how such a system would keep returning to the same neighbourhoods.""",
        figures=[figs["f0"]])

    post.add(
        "Proportional is not neutral",
        f"""The softer rule splits visits in proportion to violations found. It rarely sends everyone to one place, and it still does not learn the truth. Every violation found earns its district more visits, which find more violations: it is a Pólya urn, whose long-run proportions settle at a random value fixed by early luck. Ensign and colleagues used exactly this urn to describe a runaway feedback loop in predictive policing in 2018. In the identical world, A's final share ranged from {eq['proportional']['q10']:.2f} to {eq['proportional']['q90']:.2f} across the middle 80% of years and was within 0.1 of an even split in only {eq['proportional']['within_10']:.0%}.

{snip['un'].markdown()}

When B's rate is really higher, the proportional rule does notice, and overdoes it: A's share averaged {p['mean']:.2f}, against the {un['true_share']:.2f} that matches the two rates, because each find by the higher-rate district earns it more visits faster; under this rule its share keeps growing the longer the programme runs, so the year-end figure depends on the horizon. Either way, the record of where violations were found has become a record of where inspectors were sent.""",
        figures=[figs["f1"]])

    post.add(
        "Dividing by the effort",
        f"""The repair for the counting rules is one division. Split visits in proportion to violations found *per visit made*, an estimate of each district's rate rather than a count of its finds, and the loop stops feeding itself: A's final share was within 0.1 of its true share in **every** run, in both worlds, averaging {pv['mean']:.3f} against {un['true_share']:.3f}. More visits to a district now produce more evidence about its rate, not a larger rate. In the identical world the same rule ended at an average share of {eq['per visit']['mean']:.3f}, against an even split of 0.5.

The division is enough here because this rule never stops visiting either district, so both estimates keep improving. A rule that sends all visits to the district with the higher rate per visit, which is what an organisation trying to find as many violations as possible would do, can stop: once a district falls behind it is no longer visited and its estimate freezes. That rule ended the year with most visits in the worse district in **{gv['mostly_a']:.0%}** of runs. Keeping 10% of visits split evenly, so that every district is always being measured, cut that to {ge['mostly_a']:.0%}: this is the exploration that reinforcement learning has long treated as a price of learning from one's own actions.""")

    post.add(
        "What the arc measured",
        """Arc X followed predictions into decisions. A threshold of 0.5 assumed costs nobody had, and the cost-derived threshold assumed calibration (episode 46). The model with the best AUC was the wrong choice when misses were expensive (47). The features a model relied on were consequences and fellow travellers of the outcome, not levers (48). The people most at risk were helped no more than people chosen at random (49). And a rule that learned from its own decisions learned its own decisions back (50).

In each case the model did what it was built to do, and the decision assumed it had been built for something else. The repairs were small and specific: a cost ratio and a calibration check, an evaluation at the operating point, a causal adjustment or an experiment, a model of the difference a treatment makes, a division by effort and a share of decisions kept independent of the record. None of them is available from the model alone.""")

    post.add(
        "Where this breaks",
        """**Two districts, fixed rates.** Real programmes have many units whose rates change, partly in response to the inspections, which adds a second loop: deterrence lowers the rate where inspectors go, and the record then says those places improved.

**Violations found perfectly when present.** Inspections that miss some violations, or find more where inspectors expect them, bias even the per-visit estimate.

**Simple rules.** A real system retrains a model on features, not a ratio on counts; the mechanism is the same whenever the model's training data come from where its own predictions sent people.

**One year.** Lock-in under the greedy rules is decided early, but under the proportional rule with unequal rates the year-end share depends on the horizon, and the effect of exploration over much longer horizons was not measured.""")

    post.add(
        "What to keep",
        f"""1. Sending inspectors where the most violations had been found put every visit in one district by year's end in every run, the district with the lower true rate in {g['mostly_a']:.0%} of runs.

2. Splitting visits in proportion to violations found settled at random: within 0.1 of the truth in {p['within_10']:.0%} of runs.

3. Dividing by visits made put the split within 0.1 of the truth in every run; when the rule could starve a district, 10% exploration cut wrong lock-ins from {gv['mostly_a']:.0%} to {ge['mostly_a']:.0%}.

4. When a model's decisions decide its data, estimate rates per unit of effort, keep a share of decisions independent of the model, and treat a stable record as evidence about the rule before treating it as evidence about the world.""")

    post.add(
        "Exercise",
        """For a model that decides where to look, whom to review or what to show, find the training data that came from places, people or items it selected, and count how much came from anywhere else.

If almost none did, compare the model's estimates for the units it rarely selects with any independent measurement of them, such as an audit, a survey or a random sample. A loop that has locked in shows up as units the model is confident about and nobody has looked at.""")

    post.add(
        "Next",
        """Arc X ends here, at fifty episodes. The series began with a test score's error bar and ends with a rule that learns from what its own decisions show it; between them, every sentence was measured, and the ones that survived are listed with the ones that did not on the syllabus page.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
