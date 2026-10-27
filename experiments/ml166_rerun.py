"""ML 66: The Series, Rerun With Five More Seeds.

The sixty-sixth episode of *Machine Learning, Taught Through What Breaks*,
opening Arc XIV. "A verdict measured once, and pinned so the build fails if
it drifts, will hold on the next draw." Twelve verdicts from this series
that rested on a single random draw, rerun with the episodes' own code and
five new seeds:

* Seven held on all five new seeds, with room to spare.
* Two held on four of five: in one split, 3 of 200 bootstrap trees did
  reproduce the full-data tree's top levels (episode 11 said none), and in
  one simulation the tenth principal component scored 0.889, under the 0.9
  the episode wrote down (31).
* Three did not hold as written. The k-distance knee fell inside the
  half-moons' good radii in all five new draws, scoring 0.80 to 0.96
  instead of 0.57 (58). A benign background changed the top feature for
  33-53% of tumours, not "most" (18). Grouped anomalies' AUC was 0.19 to
  0.24, not always under 0.2, though always far below chance (59).

Run: `standarderror run ml166_rerun --publish`
"""

from __future__ import annotations

import os
import platform
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


def compute() -> dict:
    return {"claims": course.rerun_claims()}


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    cl = res["claims"]
    body = []
    for c in cl:
        vals = [r["value"] for r in c["runs"][1:]]
        body.append([str(c["episode"]), c["verdict"], c["value_is"],
                     _fmt(c["runs"][0]["value"]),
                     f"{_fmt(min(vals))} to {_fmt(max(vals))}",
                     f"{c['held']} of 5"])
    out["f0"] = charts.table_image(
        body,
        header=["ep.", "verdict as pinned", "value", "original seed",
                "five new seeds", "held"],
        title="Twelve single-draw verdicts, five more draws each",
        subtitle=("Each episode's own function, rerun with seeds 1 to 5 in "
                  "place of 0. 'Held' counts the new seeds on which every "
                  "condition the episode asserted was still true."),
        source="standarderror/ml/course.py, calling each episode's module.",
        alt=("A twelve-row table. Seven verdicts held on five of five seeds, "
             "two on four, one on two, one on one and one on none."),
        caption="Values are in each episode's own units; the 'value' column says which.",
        align="rllrrr",
        path=str(IMG / f"ml166-f0-table.{EXT}"))[0]

    def held(ax, m):
        eps = [str(c["episode"]) for c in cl]
        h = [c["held"] for c in cl]
        col = [m.series[0] if v == 5 else m.series[2] if v >= 4 else m.series[1]
               for v in h]
        ax.bar(range(len(eps)), h, color=col)
        ax.set_xticks(range(len(eps)))
        ax.set_xticklabels(eps)
        ax.set_yticks(range(6))
        ax.set_ylim(0, 5.4)

    n5 = sum(c["held"] == 5 for c in cl)
    out["f1"] = charts.diagram(
        held,
        title="How many of five new seeds each verdict survived",
        subtitle=("By episode number. Dark: all five; middle: four; light: "
                  "fewer."),
        xlabel="episode", ylabel="new seeds on which it held",
        figsize=(7.2, 3.8),
        source="standarderror/ml/course.py.",
        alt=("Twelve bars: seven at five, two at four, and bars at two, one "
             "and zero for episodes 59, 18 and 58."),
        caption=f"**{n5} of 12** held on every new seed.",
        path=str(IMG / f"ml166-f1-held.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    cl = {c["episode"]: c for c in res["claims"]}
    n5 = sum(c["held"] == 5 for c in res["claims"])

    def p1(panel, m):
        panel.bar(range(12), [c["held"] for c in res["claims"]],
                  color=m.series[0])

    def p2(panel, m):
        v = [r["value"] for r in cl[58]["runs"]]
        panel.bar(range(6), v, color=[m.series[1]] + [m.series[0]] * 5)
        panel.set_ylim(0, 1)

    def p3(panel, m):
        v = [r["value"] for r in cl[18]["runs"]]
        panel.bar(range(6), v, color=[m.series[1]] + [m.series[0]] * 5)
        panel.axhline(0.5, color=m.ink, lw=1, ls=":")
        panel.set_ylim(0, 0.7)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=66,
        headline="Five more seeds",
        panels=[(p1, f"{n5} of 12", "held on every new seed"),
                (p2, "0 of 5", "knee outside, ep. 58"),
                (p3, "1 of 5", "'most' tumours, ep. 18")],
        note=("A verdict drawn once is as sure as the distance between its "
              "number and the word the prose chose. Five of twelve words "
              "were closer than they looked."),
        alt=("Three hand-drawn frames: twelve bars mostly at full height; a "
             "short bar then five tall ones; a bar above a line then five "
             "mostly below."),
        path=str(IMG / f"ml166-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["one"] = s.run("""
        from standarderror.ml import course

        ep, verdict, what, check = next(c for c in course.CLAIMS if c[0] == 58)
        print(verdict)
        for seed in range(3):
            r = check(seed)
            print(f"seed {seed}: {what} {r['value']:.3f}, held: {r['held']}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    cl = {c["episode"]: c for c in res["claims"]}
    allheld = [e for e, c in cl.items() if c["held"] == 5]
    nearly = [e for e, c in cl.items() if c["held"] == 4]
    failed = [e for e, c in cl.items() if c["held"] < 4]

    def vals(e):
        return [r["value"] for r in cl[e]["runs"][1:]]

    # The spine, asserted rather than trusted.
    assert all(c["original_held"] for c in cl.values()) and len(cl) == 12
    assert allheld == [3, 5, 33, 40, 48, 56, 57]
    assert nearly == [11, 31] and failed == [18, 58, 59]
    assert cl[58]["held"] == 0 and min(vals(58)) > 0.75
    assert cl[18]["held"] <= 1 and 0.3 < min(vals(18)) and max(vals(18)) < 0.55
    assert max(vals(59)) < 0.25 and cl[59]["held"] <= 2
    assert sum(v > 0.2 for v in vals(59)) == 3
    assert min(vals(31)) > 0.88 and min(vals(48)) > 1.6
    assert all(v == 9 for v in vals(57)) and max(abs(v - 1) for v in vals(56)) < 0.02

    post = Post(
        title=f"{SERIES_TAG} 66: The Series, Rerun With Five More Seeds",
        slug="ml-66-rerun",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=66,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Twelve of this series' verdicts rested on one random "
                  "draw. Rerun with five more, seven held every time, two "
                  "nearly, and three did not hold as written."),
        summary=(
            f"Twelve verdicts from episodes 3 to 59 that came from a single "
            f"random draw, each rerun with the episode's own code on seeds 1 "
            f"to 5. Seven held on all five. Two held on four: in one split "
            f"3 of 200 bootstrap trees reproduced the full tree's top levels "
            f"where episode 11 said none did, and in one simulation the only "
            f"informative principal component scored {min(vals(31)):.3f} "
            f"against episode 31's threshold of 0.9. Three did not hold as "
            f"written: the k-distance knee scored {min(vals(58)):.2f} to "
            f"{max(vals(58)):.2f} on the half-moons instead of 0.57 "
            f"(episode 58); a benign background changed the top feature for "
            f"{min(vals(18)):.0%} to {max(vals(18)):.0%} of tumours, not most "
            f"(18); grouped anomalies' AUC was {min(vals(59)):.2f} to "
            f"{max(vals(59)):.2f}, not always under the 0.2 written down, "
            f"though far below chance (59). Those three episodes now say so."),
        tags=["machine-learning", "reproducibility", "robustness",
              "statistics", "replication", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Each claim's data are those of its episode: breast cancer, the "
            "digits and the diabetes data via scikit-learn, and the "
            "episodes' simulations; seed 0 reproduces the published value.",
            "Machinery: `standarderror/ml/course.py` (`CLAIMS`, "
            "`rerun_claims`), calling the functions in `leakage.py`, "
            "`imbalance.py`, `trees.py`, `reading.py`, `around.py`, "
            "`rows.py`, `decide.py` and `unsup.py`; tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Bouthillier, Laurent and Vincent, "
            "\"Unreproducible research is reproducible\", *ICML* (2019); "
            "Gelman and Loken, \"The garden of forking paths\" (2013); "
            "episodes 1, 20 and 51 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; about six minutes of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Turning the method on itself",
        f"""Every episode of this series ends its build with a list of assertions: the numbers the prose quotes, and the qualitative claims it makes, checked against the measurement so that the build fails if either drifts. That guards against a stale number. It does not guard against a lucky one. Many verdicts were averaged over repeated draws, but some came from a single split, a single simulated data set or a single random initialisation, and the assertion pinned what that one draw happened to show. The sentence under test is the series' own working assumption: *a verdict measured once, and pinned so the build fails if it drifts, will hold on the next draw*.

The last arc turns the series' methods on the series. This episode reruns it; the next four measure how choices compound, which problems more data cure, what a whole project looks like when every check is run, and what, in the end, survived.

Twelve verdicts qualified: each rested on one draw, and each episode's function takes a seed. Every one was rerun with seeds 1 to 5, using the episode's own code and the conditions its assertions state. Seed 0 reproduces every published value, so the comparison is only between draws.

{snip['one'].markdown()}""")

    post.add(
        "What held",
        f"""Seven verdicts held on all five new seeds: episodes {', '.join(str(e) for e in allheld)}. Some held with the same number to three decimals, such as the BIC choosing {vals(57)[0]} components for three squares every time and t-SNE drawing a cluster ten times wider at {min(vals(56)):.3f} to {max(vals(56)):.3f} times the narrow one's size. Others moved and stayed far from their threshold: dropping the test result inflated the treatment's apparent effect {min(vals(48)):.2f} to {max(vals(48)):.2f}-fold against a claim of more than 1.6, and a training-fitted encoding ranked its own labels at {min(vals(40)):.3f} to {max(vals(40)):.3f}.

Two held on four of five. Episode 11 said that none of 200 bootstrap trees reproduced the full-data tree's top three levels; in one of the new splits, 3 did. Episode 31 said only the tenth principal component predicted, at accuracy above 0.9; in one new simulation it scored {min(vals(31)):.3f}. Both points survive in substance: three trees in 200 is still almost none, and {min(vals(31)):.3f} is still the only component that predicts. The thresholds written into the prose were tighter than the evidence.""",
        figures=[figs["f0"]])

    post.add(
        "What did not hold as written",
        f"""Three verdicts did not survive in the form the episodes stated.

Episode 58 found that the radius at the knee of the k-distance plot fell just below the half-moons' good radii and scored 0.57. On five other draws of the moons it fell inside them every time and scored {min(vals(58)):.2f} to {max(vals(58)):.2f}. The published miss was this one draw's: the knee sits close to the lower edge of the window, and which side it lands on depends on the noise. The episode's broader point, that the heuristic picked a wrong radius on the digits, stands; the moons no longer illustrate it.

Episode 18 found that explaining tumours against benign ones instead of all of them changed the top feature for 58% of tumours, and wrote "most". On five other splits the share was {min(vals(18)):.0%} to {max(vals(18)):.0%}. A large share, but "most" was this split's.

Episode 59 asserted that grouped anomalies scored an AUC below 0.2 on the k-th-neighbour distance whenever k was smaller than the group. On the new seeds the worst AUC ranged from {min(vals(59)):.2f} to {max(vals(59)):.2f}: always far below chance, the episode's real claim, but above the 0.2 its assertion wrote down in three of five.

All three episodes have been revised to say so, with the range from this rerun, and the syllabus rows with them.""",
        figures=[figs["f1"]])

    post.add(
        "What the failures have in common",
        """The verdicts that broke were not the surprising ones. Each was a correct observation whose number sat close to the word the prose attached to it: a knee just below a window, a share just above a half, an AUC just under a round threshold. Bouthillier, Laurent and Vincent made the same observation about deep-learning papers in 2019: results reported from one seed often fail to reproduce under another, not because the method is wrong but because the margin was smaller than the run-to-run variation. Episode 20 measured that variation for networks; this episode measured it for the series' own claims.

The verdicts that held had one of two protections: they were averages over many draws inside the episode, or the effect was far larger than any plausible variation, such as nine components for three squares or a tenfold size difference drawn as none. A single draw is enough evidence for a large effect and not enough for a precise number, and a pinned assertion is only as good as the margin between the measured value and the threshold it checks.

The cost of finding this was small. The twelve reruns took about six minutes of computer time, less than writing any one of the episodes they checked, and they changed three of them. The practical version is a habit rather than a procedure: whenever a number is about to become a word, such as most, none, always or below a round threshold, rerun the calculation with a few seeds first and see whether the word survives the range.""")

    post.add(
        "Where this breaks",
        """**Twelve of many.** These were the verdicts that rested on one draw and whose functions take a seed. Verdicts averaged inside their episode were not rerun, on the reasoning that their spread was already measured; some may still depend on choices other than the seed.

**Five new seeds.** Enough to catch a verdict that fails often, not one that fails one time in twenty.

**Seeds change different things.** For some functions the seed draws a new data set, for others only a new split of the same data or a new model initialisation; real-data claims cannot be given new data this way.

**The same author checking the same code.** A rerun cannot find a mistake in the code itself, only in how much one draw was trusted.""")

    post.add(
        "What to keep",
        f"""1. Of twelve single-draw verdicts, {len(allheld)} held on five of five new seeds and {len(nearly)} on four of five, where a threshold in the prose was tighter than the evidence.

2. Three did not hold as written: a heuristic's radius that missed in one draw and hit in five (episode 58), a share that was "most" in one split and {min(vals(18)):.0%} to {max(vals(18)):.0%} in others (18), and an AUC threshold crossed in three of five reruns (59).

3. The broken verdicts were observations close to the words chosen for them; the ones that held were averages or very large effects.

4. Before writing a number into a claim, rerun it with a few seeds. Write down the range, and choose words the range supports.""")

    post.add(
        "Exercise",
        """Take a result you have reported from a single run: a model comparison, a feature ranking, a tuned setting. Rerun it with three other seeds, changing the data split and the model's randomness.

If the conclusion survives, report the range with it. If it does not, the original was a draw, and the sentence about it needs rewriting before anyone else reads it.""")

    post.add(
        "Next",
        """A rerun varies the seed and nothing else. A real analysis varies much more: which features, which model, which split, which subgroup, each chosen in good faith after a look at the results. Episode 67 makes five such choices in a row on data with no signal at all.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
