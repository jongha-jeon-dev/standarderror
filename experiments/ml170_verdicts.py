"""ML 70: Sixty-Nine Sentences, Measured, and One Held as Written.

The seventieth and last episode of *Machine Learning, Taught Through What
Breaks*. Sixty-nine episodes each took one sentence the standard
curriculum states as fact and measured where it stopped being true. Sorted:

* one held as written (episode 3: fit preprocessing inside the folds);
* thirty held only under conditions the sentence does not state;
* thirty-eight failed as stated in the setting measured.

The episode also counts the build's own checks (the assert statements
pinning each episode's numbers) and recalls what happened when twelve of
the series' single-draw verdicts were rerun (episode 66).

Run: `standarderror run ml170_verdicts --publish`
"""

from __future__ import annotations

import os
import platform
from collections import Counter
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import course
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 27)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

ARCS = [
    ("I", "What a score means", range(1, 6)),
    ("II", "Linear models, pushed", range(6, 11)),
    ("III", "Trees and ensembles", range(11, 16)),
    ("IV", "Reading what a model learned", range(16, 21)),
    ("V", "Training a network", range(21, 26)),
    ("VI", "Generalisation in a network", range(26, 31)),
    ("VII", "Around the model", range(31, 36)),
    ("VIII", "Where the rows come from", range(36, 41)),
    ("IX", "Combining and comparing", range(41, 46)),
    ("X", "From prediction to decision", range(46, 51)),
    ("XI", "Grading a language model", range(51, 56)),
    ("XII", "Without labels", range(56, 61)),
    ("XIII", "After the model ships", range(61, 66)),
    ("XIV", "The course, measured", range(66, 70)),
]

# The author's reading of each episode's own verdict, by the rule in the
# post: "failed" if the sentence was false in the setting the episode
# measured; "conditions" if it held in some measured settings and not in
# others, or held for one half of a two-part sentence; "held" if it held
# in every setting measured.
FAILED = {1, 2, 4, 5, 7, 8, 11, 16, 19, 21, 22, 24, 26, 27, 28, 31, 35, 37,
          40, 41, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 60, 61,
          63, 67, 69}
HELD = {3}
VERDICTS = {e: "failed" if e in FAILED else "held" if e in HELD
            else "held under conditions" for e in range(1, 70)}
KINDS = ("failed", "held under conditions", "held")


def compute() -> dict:
    return {"census": course.assert_census()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    cen = res["census"]

    def grid(ax, m):
        from matplotlib.patches import Rectangle
        col = {"failed": m.series[1], "held under conditions": m.series[0],
               "held": m.series[2]}
        for r, (num, name, eps) in enumerate(ARCS):
            y = len(ARCS) - 1 - r
            for j, e in enumerate(eps):
                ax.add_patch(Rectangle(
                    (j + 0.05, y + 0.08), 0.9, 0.84, color=col[VERDICTS[e]],
                    lw=0))
                ax.text(j + 0.5, y + 0.5, str(e), ha="center", va="center",
                        fontsize=7.5, color="white")
            ax.text(-0.15, y + 0.5, f"{num}  {name}", ha="right", va="center",
                    fontsize=7.6, color=m.ink)
        for k in KINDS:
            ax.bar([0], [0], color=col[k], label=k)
        ax.set_xlim(-4.6, 5.1)
        ax.set_ylim(0, len(ARCS))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(fontsize=7.6, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.01), ncol=3)

    n = Counter(VERDICTS.values())
    out["f0"] = charts.diagram(
        grid,
        title="Sixty-nine sentences, by arc and by what measurement did to them",
        subtitle=(f"{n['failed']} failed as stated, "
                  f"{n['held under conditions']} held only under conditions "
                  f"the sentence does not state, {n['held']} held as written."),
        figsize=(7.4, 6.4), ticks=False,
        source="The episodes of this series; the classification is the author's.",
        alt=("A grid of 69 numbered squares, five per arc, coloured by "
             "verdict. Failed and conditional verdicts are spread across "
             "every arc; one square, episode 3, held."),
        caption=("Episode 70, this one, tests no sentence of its own."),
        path=str(IMG / f"ml170-f0-verdicts.{EXT}"))[0]

    per = cen["per_episode"]

    def census(ax, m):
        e = sorted(per)
        ax.bar(e, [per[k] for k in e], color=m.series[0], width=0.8)
        ax.axhline(cen["median"], color=m.ink, lw=0.9, ls=":")
        ax.text(70.5, cen["median"] + 0.4, f"median {cen['median']:.0f}",
                ha="right", fontsize=8, color=m.ink)
        ax.set_xlim(0, 71)

    out["f1"] = charts.diagram(
        census,
        title="The checks behind the numbers: assert statements per episode",
        subtitle=(f"{cen['asserts']:,} assertions across {cen['episodes']} "
                  "episode builds. Each pins a number the prose quotes or a "
                  "claim it makes; the build fails if any stops holding."),
        xlabel="episode", ylabel="assert statements", figsize=(7.8, 3.8),
        source="experiments/ml1*.py in this repository, counted at build time.",
        alt=("Seventy bars between about 6 and 21, with a dotted median line "
             "at 13."),
        caption=("A count of statements, not of claims: one assert can pin "
                 "several values, and some claims are pinned by more than one."),
        path=str(IMG / f"ml170-f1-census.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    n = Counter(VERDICTS.values())
    cen = res["census"]

    def p1(panel, m):
        panel.bar(range(3), [n[k] for k in KINDS],
                  color=[m.series[1], m.series[0], m.series[2]])

    def p2(panel, m):
        per = cen["per_episode"]
        panel.bar(sorted(per), [per[k] for k in sorted(per)], color=m.series[0])

    def p3(panel, m):
        panel.bar(range(3), [7, 2, 3], color=[m.series[2], m.series[0],
                                              m.series[1]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=70,
        headline="Seventy episodes, measured",
        panels=[(p1, f"{n['held']} of 69", "held as written"),
                (p2, f"{cen['asserts']:,}", "pinned checks"),
                (p3, "9 of 12", "held on rerun")],
        note=("Most of the curriculum's sentences are true somewhere. What the "
              "series measured is where, and the answer is usually narrower "
              "than the sentence."),
        alt=("Three hand-drawn frames: three bars, the first two tall and "
             "the third tiny; seventy small bars; three bars."),
        path=str(IMG / f"ml170-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["census"] = s.run("""
        from standarderror.ml import course

        c = course.assert_census()       # counts `assert` lines in every build
        print(c["episodes"], "episodes,", c["asserts"], "assert statements")
        print("median per episode:", c["median"], " most:", c["most"])
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    cen = res["census"]
    n = Counter(VERDICTS.values())
    by_arc = {num: Counter(VERDICTS[e] for e in eps) for num, _, eps in ARCS}
    fail_heavy = [num for num, c in by_arc.items() if c["failed"] >= 4]

    # The spine, asserted rather than trusted.
    assert sum(len(eps) for *_, eps in ARCS) == 69 == len(VERDICTS)
    assert (n["failed"], n["held under conditions"], n["held"]) == (38, 30, 1)
    assert cen["episodes"] == 70 and cen["asserts"] > 925
    assert cen["median"] == 13 and cen["most"][1] >= 19
    assert fail_heavy == ["I", "X", "XI"]
    assert min(by_arc, key=lambda a: by_arc[a]["failed"]) == "III"
    assert by_arc["III"]["failed"] == 1 and by_arc["X"]["failed"] == 5
    assert by_arc["XIII"]["failed"] == 2 and by_arc["XIV"]["failed"] == 2

    post = Post(
        title=f"{SERIES_TAG} 70: Sixty-Nine Sentences, Measured, and One Held as Written",
        slug="ml-70-verdicts",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=70,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("The series' last episode sorts what measurement did to each "
                  "sentence it tested. Thirty-eight failed as stated, thirty "
                  "held only under conditions, and one held as written."),
        summary=(
            f"Sixty-nine episodes each measured one sentence the curriculum "
            f"states as fact. By the author's reading of each episode's "
            f"verdict, {n['failed']} failed as stated in the setting "
            f"measured, {n['held under conditions']} held only under "
            f"conditions the sentence does not state, and {n['held']} held as "
            f"written: fit every preprocessing step inside the folds. The "
            f"builds behind the {cen['episodes']} episodes carry "
            f"{cen['asserts']:,} assert statements, a median of "
            f"{cen['median']:.0f} per episode. Of twelve verdicts that rested "
            f"on one random draw, seven held on five new seeds, two on four, "
            f"and three did not hold as written and were revised."),
        tags=["machine-learning", "evaluation", "reproducibility",
              "statistics", "curriculum", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "The 69 earlier episodes of this series and their builds, "
            "`experiments/ml101_*.py` to `experiments/ml169_*.py`; the "
            "verdicts are the author's classification of each episode's "
            "own conclusion, by the rule stated in the post, and are listed "
            "in `experiments/ml170_verdicts.py`.",
            "Machinery: `standarderror/ml/course.py` (`assert_census`), "
            "tested in `tests/test_ml.py`.",
            "Where this stops: the series index, which gives each sentence "
            "and its measurement in one row; episode 66 for the reruns.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "a count of files; under a second",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Sixty-nine sentences",
        f"""This series took one sentence per episode from the standard machine-learning curriculum, the kind of sentence a textbook, a course or a senior colleague states as fact, and measured where it stopped being true. The sentences ran from how a score is read (*the model with the higher test score is better*) through models, training, the rows, decisions and language models, to a model in production and the series itself (*a verdict measured once will hold on the next draw*). This last episode tests no new sentence. It sorts what happened to the sixty-nine, and counts the checks that kept the numbers honest.

Every number in an episode came from code that ran when the page was built, and the numbers the prose quotes are pinned by assertions: if the code stops producing it, the build fails and the page does not publish. Those assertions can be counted.

{snip['census'].markdown()}

The {cen['asserts']:,} statements are the series' working evidence. They guard against a number going stale when the code or a library changes; episode 66 showed what they do not guard against, a number that was lucky.""",
        figures=[figs["f1"]])

    post.add(
        "What measurement did to the sentences",
        f"""Sorting a verdict needs a rule, and this is the one used: a sentence **failed** if it was false in the setting the episode measured; it **held under conditions** if it was true in some measured settings and false in others, or true for one half of a two-part sentence; it **held** if it was true in every setting measured. The classification is the author's reading of each episode's own conclusion, and some calls are close; the list is in the episode's code for anyone who would sort differently.

By that rule, {n['failed']} sentences failed as stated, {n['held under conditions']} held under conditions, and {n['held']} held as written. The one that held was episode 3's: *fit every preprocessing step inside the folds*. Even that came with a measured scale, since violations ranged from nothing on real data with strong features to coin flips scored at 0.88 on noise, but no measured setting made the rule wrong.

The failures were spread across the course. Every arc had at least one, and three had four or more of five: what a score means (I), turning predictions into decisions (X, all five) and grading a language model (XI). Those are the arcs whose sentences are rules for reading a number or acting on it: the higher score is better, the threshold is 0.5, the benchmark measures the model. Such a rule compresses a judgement into a default, and a default is what a measurement in a new setting tends to break. The arc with the fewest failures was trees and ensembles (III), with one: its sentences, about forests, boosting and out-of-bag error, were mostly true within limits the curriculum seldom states.""",
        figures=[figs["f0"]])

    post.add(
        "What the broken sentences had in common",
        """Read together, the failures and the conditional verdicts fall into a few families, and each family is a single habit worth unlearning.

**A score answers a narrower question than the one asked of it.** Cross-validation estimates a procedure, not the fitted model (episode 2). A test score has an error bar wider than most gains (1). The best of a search is the search's luck (35, 67). A benchmark score includes its prompt and its distractors (53, 54). An AUC ranks; it does not decide (47).

**A method reports on the model or its assumptions, not on the data.** Importance measures what a model uses, not what matters (16, 48). Partial dependence and SHAP answer questions about rows that do not exist or a background someone chose (17, 18). The BIC counts Gaussians and t-SNE keeps neighbours (56, 57).

**The rows are not what they seem.** Rows from the same people, applicants who were refused, a category's own labels and a time series' remembered errors (episodes 36, 37, 39 and 40) all make the data say more than they hold, and episode 68 showed that more rows of the same kind did not fix the first two.

**A prediction is not a decision.** The threshold, the cost of a mistake, who is helped by an intervention, and what a deployed rule does to the data it later learns from (46 to 50, 63) are not in the predictions, and in episode 69 the threshold mattered more than the choice of model.""")

    post.add(
        "What the series got wrong about itself",
        """The series held itself to the same standard, and did not fully meet it. Episode 66 reran the twelve verdicts that had rested on a single random draw, with five new seeds each. Seven held every time and two held four times in five, where a threshold written into the prose was tighter than the evidence. Three did not hold as written: a heuristic radius that missed in the published draw and hit in five others (episode 58), a share described as "most" that was 33% to 53% on other splits (18), and an anomaly score's AUC that crossed the round threshold its assertion wrote down (59). Those episodes, and the syllabus rows that summarise them, now say so.

Episode 65 took four of the series' simulated findings to real data and found the directions replicated while the sizes often did not, and episode 67 showed how a sequence of defensible choices can produce a number nobody chose. Both apply to this series too: its sentences were chosen by one author, and its settings were chosen to make each effect visible.""")

    post.add(
        "Where this breaks",
        """**The sentences were chosen to break.** Each was picked because it is stated as fact and could be tested on a laptop; the count of failures says how often such sentences stop short, not how often the curriculum is wrong.

**One author's classification.** The line between "failed" and "held under conditions" is a judgement. A different reader would move a handful of episodes, and the code lists every call so that they can.

**Small data and simulations.** Most episodes used data sets of hundreds to a few thousand rows, or simulations built to show an effect. Episodes 65 and 68 measured how far such findings travel; the answer was: in direction, usually; in size, not always.

**Assertions pin what was measured.** They cannot catch a mistake in the code that produced the number, only a change in it.""")

    post.add(
        "What to keep",
        f"""1. Of sixty-nine sentences, {n['failed']} failed as stated, {n['held under conditions']} held under conditions and {n['held']} held as written. Most of the curriculum is true somewhere; the work is finding where.

2. The failures clustered in three arcs: how a score is read, how a prediction becomes a decision, and how a language model is graded.

3. {cen['asserts']:,} assertions kept the numbers from going stale, and five extra seeds showed three of twelve single-draw verdicts were luckier than the words chosen for them.

4. Before acting on a sentence about machine learning, ask what setting it was measured in, and measure it in yours.""")

    post.add(
        "Exercise",
        """Pick a sentence your team repeats about its models: a rule about thresholds, features, retraining or evaluation. Write it down exactly as people say it.

Design the smallest measurement that could show it false in your setting, run it, and classify the result by this episode's rule. If it held, write down the conditions you tested; if it held only under some, rewrite the sentence so that it says which.""")

    post.add(
        "The end of the series",
        """This is the last episode of *Machine Learning, Taught Through What Breaks*. The series index lists all seventy, each with its sentence and what measurement did to it, and the code behind every number is in the repository. The checklist in episode 69 is the shortest way to use it on a project; the rest is there for the day one of its sentences matters.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
