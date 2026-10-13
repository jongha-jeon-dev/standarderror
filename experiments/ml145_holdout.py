"""ML 45: A Test Set Answers Honestly Only the First Time.

The forty-fifth episode of *Machine Learning, Taught Through What
Breaks*, closing Arc IX. "A held-out test set gives an unbiased estimate,
however often you look at it, as long as you never train on it." Measured
three ways, from a deliberate attack to an ordinary analyst:

* On a public leaderboard of 1,000 coin-flip labels, keeping the random
  guesses that scored above 0.5 and submitting their majority vote scored
  0.713 after 1,000 submissions; the same vote scored 0.497 on the private
  labels.
* Selecting noise features that agreed with the label on both the training
  set and the holdout gave 0.687 on the holdout and 0.498 on fresh data.
* Greedy feature selection guided by holdout accuracy, never training on
  the holdout, reached 0.690 on it after 20 steps, above the 0.676 that the
  true model scores; fresh accuracy peaked at 0.579.

Run: `standarderror run ml145_holdout --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import combine
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"atk": combine.leaderboard_attack(),
            "dw": combine.noise_selection_holdout(),
            "gh": combine.greedy_holdout(by="holdout"),
            "gc": combine.greedy_holdout(by="cv")}


# ------------------------------------------------------------------ figures

def _attack(ax, m, atk):
    q = [r["queries"] for r in atk]
    ax.plot(q, [r["public"] for r in atk], marker="o", color=m.series[1],
            label="public leaderboard (queried)")
    ax.plot(q, [r["private"] for r in atk], marker="o", color=m.ink,
            label="private labels (never queried)")
    ax.axhline(0.5, color=m.grid, ls=":", lw=1)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def _greedy(ax, m, gh, gc):
    s = np.arange(1, len(gh["holdout"]) + 1)
    ax.plot(s, gh["holdout"], color=m.series[1], lw=2,
            label="holdout-guided: holdout score")
    ax.plot(s, gh["fresh"], color=m.series[1], ls="--",
            label="holdout-guided: fresh rows")
    ax.plot(s, gc["holdout"], color=m.series[2], lw=2,
            label="CV-guided: holdout score")
    ax.plot(s, gc["fresh"], color=m.series[2], ls="--",
            label="CV-guided: fresh rows")
    ax.axhline(gh["oracle"], color=m.ink, ls=":", lw=1.2)
    ax.text(len(s), gh["oracle"] + 0.003, "the 10 real features", ha="right",
            fontsize=8, color=m.ink)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def figures(res: dict) -> dict:
    out: dict = {}
    atk, dw, gh, gc = res["atk"], res["dw"], res["gh"], res["gc"]
    a = {r["queries"]: r for r in atk}

    out["f0"] = charts.diagram(
        lambda ax, m: _attack(ax, m, atk),
        title="Climbing a leaderboard of coin flips",
        subtitle=("1,000 public and 1,000 private labels, all coin flips. "
                  "Submit k random guesses, keep those scoring above 0.5 on "
                  "the public labels, submit their majority vote; means over "
                  "50 leaderboards."),
        xlabel="random submissions scored on the public labels (log scale)",
        ylabel="accuracy of the final vote",
        figsize=(7.2, 4.4),
        source="Simulated; standarderror/ml/combine.py.",
        alt=("The public score rises from 0.50 to 0.84 as submissions grow "
             "from 1 to 3,000; the private score stays at 0.50."),
        caption=(f"After 1,000 submissions the vote scored "
                 f"**{a[1000]['public']:.3f}** on the public labels and "
                 f"{a[1000]['private']:.3f} on the private ones."),
        path=str(IMG / f"ml145-f0-leaderboard.{EXT}"))[0]

    out["f1"] = charts.diagram(
        lambda ax, m: _greedy(ax, m, gh, gc),
        title="An analyst who checks the holdout after every step",
        subtitle=("Forward selection for logistic regression, 10 real "
                  "features among 100, 300 training and 300 holdout rows; "
                  "each step adds the feature that most improves the holdout "
                  "score, or five-fold CV on the training rows. 5 data sets."),
        xlabel="features selected",
        ylabel="accuracy",
        figsize=(7.4, 4.6),
        source="Simulated; standarderror/ml/combine.py.",
        alt=("The holdout-guided holdout score climbs steadily to about "
             "0.69, above a dotted line at 0.676; its fresh-row accuracy "
             "levels off near 0.58. The CV-guided lines stay close "
             "together."),
        caption=(f"After 20 holdout-guided steps the holdout said "
                 f"**{gh['holdout'][-1]:.3f}**; fresh rows said "
                 f"{gh['fresh'][-1]:.3f}, and a model on the 10 real features "
                 f"scores {gh['oracle']:.3f}."),
        path=str(IMG / f"ml145-f1-greedy.{EXT}"))[0]

    body = [[f"{r['k']}", f"{r['kept']:.0f}", f"{r['holdout']:.3f}",
             f"{r['fresh']:.3f}"] for r in dw]
    out["f2"] = charts.table_image(
        body,
        header=["features asked for", "features kept", "holdout accuracy",
                "fresh accuracy"],
        title="Noise that agreed with the holdout",
        subtitle=("2,000 noise features, coin-flip labels, 1,000 rows each "
                  "for training, holdout and fresh data. Keep features whose "
                  "correlation with the label passes 1/√n with the same sign "
                  "on training and holdout; vote with the strongest k. 20 "
                  "repetitions."),
        source="Simulated; standarderror/ml/combine.py.",
        alt=("A four-row table: holdout accuracy rises from 0.56 to 0.69 "
             "as more selected features vote; fresh accuracy stays at 0.50."),
        caption=("The holdout was never trained on. It was asked which "
                 "features to keep, and its answer is now part of the model."),
        align="rrrr",
        path=str(IMG / f"ml145-f2-noise.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    atk, dw, gh = res["atk"], res["dw"], res["gh"]

    def p1(panel, m):
        panel.plot([r["public"] for r in atk], color=m.series[1], lw=2)
        panel.plot([r["private"] for r in atk], color=m.ink, lw=2)

    def p2(panel, m):
        panel.bar([0, 1], [dw[-1]["holdout"], dw[-1]["fresh"]],
                  color=[m.series[1], m.ink])
        panel.set_ylim(0.4, 0.75)

    def p3(panel, m):
        panel.plot(gh["holdout"], color=m.series[1], lw=2)
        panel.plot(gh["fresh"], color=m.ink, lw=2)

    a = {r["queries"]: r for r in atk}
    return charts.lecture_hero(
        series=SERIES_TAG, episode=45,
        headline="Asked once",
        panels=[(p1, f"{a[1000]['public']:.3f}", "coin-flip leaderboard"),
                (p2, f"{dw[-1]['holdout']:.3f}", "noise kept by holdout"),
                (p3, f"{gh['holdout'][-1]:.3f}", "holdout after 20 checks")],
        note=("Every look at a test set that changes what you do next "
              "spends a little of its honesty. Enough looks, and it reports "
              "your choices back to you."),
        alt=("Three hand-drawn frames: a rising line over a flat one; a "
             "tall bar and a short one; two lines drawing apart."),
        path=str(IMG / f"ml145-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["atk"] = s.run("""
        from standarderror.ml import combine

        for r in combine.leaderboard_attack():   # 50 leaderboards
            print(f"{r['queries']:5d} submissions: public {r['public']:.3f}, "
                  f"private {r['private']:.3f}")
        """)
    out["greedy"] = s.run("""
        h = combine.greedy_holdout(by="holdout")
        c = combine.greedy_holdout(by="cv")
        print("features | holdout-guided: holdout / fresh | CV-guided: same")
        for step in (0, 4, 9, 19):
            print(f"{step + 1:8d} | {h['holdout'][step]:.3f} / "
                  f"{h['fresh'][step]:.3f}{'':18s} | {c['holdout'][step]:.3f}"
                  f" / {c['fresh'][step]:.3f}")
        print(f"real features among 20 chosen: holdout-guided "
              f"{h['real_share'][19]:.0%}, CV-guided {c['real_share'][19]:.0%}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    atk, dw, gh, gc = res["atk"], res["dw"], res["gh"], res["gc"]
    a = {r["queries"]: r for r in atk}
    d = {r["k"]: r for r in dw}
    peak = int(np.argmax(gh["fresh"]))
    gap_h = gh["holdout"][-1] - gh["fresh"][-1]
    gap_c = gc["holdout"][-1] - gc["fresh"][-1]

    # The spine, asserted rather than trusted.
    assert all(abs(r["private"] - 0.5) < 0.005 for r in atk)
    assert all(x["public"] < y["public"] for x, y in zip(atk, atk[1:]))
    assert a[1000]["public"] > 0.71 and a[3000]["public"] > 0.83
    assert all(abs(r["fresh"] - 0.5) < 0.008 for r in dw)
    assert d[100]["holdout"] > 0.68 and d[10]["holdout"] > 0.55
    assert gh["holdout"][-1] > gh["oracle"] > gh["fresh"][-1] + 0.09
    assert all(x <= y + 0.0015 for x, y in zip(gh["holdout"], gh["holdout"][1:]))
    assert peak <= 5 and max(gh["fresh"]) < 0.58
    assert gh["real_share"][-1] < 0.15
    assert gap_h > 0.1 and abs(gap_c) < 0.005
    assert gc["real_share"][-1] < 0.2 and abs(gc["fresh"][-1] - gh["fresh"][-1]) < 0.01

    post = Post(
        title=f"{SERIES_TAG} 45: A Test Set Answers Honestly Only the First Time",
        slug="ml-45-holdout",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=45,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A test set is not spent by training on it. It is spent by "
                  "asking it questions whose answers change what you do "
                  "next, and it can be spent without touching a single "
                  "weight."),
        summary=(
            f"Three ways to wear out a holdout without training on it. On a "
            f"leaderboard of 1,000 coin-flip labels, the majority vote of "
            f"random guesses that scored above 0.5 reached "
            f"{a[1000]['public']:.3f} after 1,000 submissions and "
            f"{a[1000]['private']:.3f} on the private labels. Noise features "
            f"kept because they agreed with the holdout gave "
            f"{d[100]['holdout']:.3f} on it and {d[100]['fresh']:.3f} on "
            f"fresh rows. An analyst adding, at each step, the feature that "
            f"most improved holdout accuracy saw {gh['holdout'][-1]:.3f} "
            f"after 20 steps, above the {gh['oracle']:.3f} the true model "
            f"scores; fresh accuracy was {gh['fresh'][-1]:.3f}."),
        tags=["machine-learning", "data-science", "statistics",
              "model-evaluation", "adaptive-data-analysis", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated leaderboards: 1,000 public and 1,000 private "
            "coin-flip labels, random 0/1 submissions; 50 leaderboards "
            "(seeds 0 to 49).",
            "Simulated noise selection: 2,000 standard-normal features and "
            "±1 coin-flip labels, 1,000 rows each for training, holdout and "
            "fresh data; 20 repetitions (seeds 0 to 19).",
            "Simulated forward selection: 100 standard-normal features, the "
            "first 10 with weight 0.35 on the log-odds; 300 training rows, "
            "300 holdout rows, 20,000 fresh rows; 5 data sets (seeds 0 to "
            "4).",
            "Machinery: `standarderror/ml/combine.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's LogisticRegression and "
            "five-fold stratified cross-validation.",
            "Where this stops: Dwork et al., \"The reusable holdout: "
            "preserving validity in adaptive data analysis\", *Science* "
            "(2015); Blum and Hardt, \"The Ladder: a reliable leaderboard "
            "for machine learning competitions\", *ICML* (2015); Recht et "
            "al., \"Do ImageNet classifiers generalize to ImageNet?\", "
            "*ICML* (2019).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("seeds as listed; about ten minutes of CPU, "
                            "most of it the cross-validated selection"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The rule about test sets",
        f"""Episode 1 put an error bar on a test score. Episode 35 found that the best of many cross-validated scores reports the luck of the search. Episode 44 asked which tests can tell two models apart. Each assumed the test set was used once. In practice it rarely is: a model is scored, adjusted, scored again; a competition's public leaderboard is consulted thousands of times. The reassurance that makes this feel safe is the sentence under test: *a held-out test set gives an unbiased estimate, however often you look at it, as long as you never train on it*.

The cleanest refutation is Blum and Hardt's, from 2015. A leaderboard holds 1,000 public labels and 1,000 private ones, and every label is a coin flip, so no submission can genuinely beat 0.5. The attacker submits random guesses, notes which scored above 0.5 on the public labels, and submits the majority vote of those:

{snip['atk'].markdown()}

After 1,000 submissions the vote scores **{a[1000]['public']:.3f}** on the public labels and {a[1000]['private']:.3f} on the private ones. Nothing was trained on anything. Each submission asked the public labels one question, "is this guess better than chance?", and each answer leaked a little about them. The vote collects those leaks.""",
        figures=[figs["f0"]])

    post.add(
        "Without an attacker",
        f"""Nobody does that on purpose. Dwork and colleagues showed in 2015 that an ordinary-looking analysis does it by accident. Their example, scaled down here: 2,000 features of pure noise and coin-flip labels. Keep the features whose correlation with the label is noticeably positive or negative on the training set *and* agrees in sign on the holdout, which sounds like a careful way of keeping only features that replicate. Then classify by a vote of the strongest ones.

Asked for 100 such features (about {d[100]['kept']:.0f} pass the filter on average), the classifier scores **{d[100]['holdout']:.3f}** on the holdout. On fresh data it scores {d[100]['fresh']:.3f}. The holdout was never trained on; it was used to decide which features to keep, and the features it kept are the ones that agree with its particular noise. Its score is now a measurement of how well the model fits the holdout, which is what it was chosen for.""",
        figures=[figs["f2"]])

    post.add(
        "An analyst, step by step",
        f"""The third case is the most ordinary. A data set with 100 features, ten of them weakly informative; 300 rows to train on and 300 held out. The analyst builds a logistic model by forward selection, adding at each step the feature that most improves accuracy on the holdout, a natural thing to do with a holdout one is told not to train on. For comparison, a second analyst chooses each step by five-fold cross-validation on the training rows and leaves the holdout alone until the end.

{snip['greedy'].markdown()}

The first analyst's holdout score rises at almost every step, to **{gh['holdout'][-1]:.3f}** with 20 features. That is above {gh['oracle']:.3f}, what a model given exactly the ten real features scores on fresh data, the best this procedure could honestly reach. On fresh rows the same models peaked at {max(gh['fresh']):.3f} after {peak + 1} features and ended at {gh['fresh'][-1]:.3f}, and only {gh['real_share'][-1]:.0%} of the 20 chosen features were real. The holdout overstated the final model by {gap_h:.3f}.

The second analyst's holdout, consulted only to report the result, said {gc['holdout'][-1]:.3f} for a model that scored {gc['fresh'][-1]:.3f} on fresh rows, within {abs(gap_c):.3f}. Cross-validation inside the training rows is subject to episode 35's luck of the search, and that selection picked noise too (only {gc['real_share'][-1]:.0%} of its 20 features were real), but its luck was spent on the training rows, not on the set used to report the score, so the report stayed honest about the model it got.""",
        figures=[figs["f1"]])

    post.add(
        "Spending a test set",
        """The common thread is that a test set's honesty is a budget, and it is spent by adaptivity: any decision made after looking at a test score, whether to keep a feature, a model, a hyperparameter or a submission, makes later scores on the same set depend on its particular noise. Training on it spends the budget at once. Looking at it spends a little per look, and the amount depends on how much each look changes what happens next.

Two kinds of remedy follow. The procedural one is the oldest: keep a final test set that is used once, after all decisions are made, and accept that its score, with episode 1's error bar, is the report. Everything before it, including any "validation" or "public" set, is part of training in the sense that matters. The algorithmic one, from the same authors, limits what each look reveals: Blum and Hardt's Ladder reports a new leaderboard score only when a submission beats the previous best by a margin, and Dwork and colleagues' Thresholdout answers queries with the training score unless the holdout disagrees by more than a noisy threshold. Both slow the leak, at the price of less precise answers.

Benchmarks raise the same question at the scale of a field. Recht and colleagues rebuilt ImageNet's test set following the original procedure and found that published models lost 11 to 14 points of accuracy on it while keeping their order, and they attributed the loss mainly to the new images being harder rather than to years of adaptive reuse. With large test sets and many genuinely different models, the leak can be slow; the three cases above show where it is fast.""")

    post.add(
        "Where this breaks",
        """**Adversarial by construction.** The leaderboard attack and the noise selection are the worst cases, with nothing real to find. With strong real signal, the honest part of each score dominates and the adaptive part matters less.

**Small holdouts.** 300 to 1,000 rows. In Blum and Hardt's analysis the gain from such attacks shrinks roughly as one over the square root of the holdout's size, so large test sets tolerate more looks; they do not tolerate unlimited ones.

**One analyst, one strategy.** Real model development mixes holdout-guided decisions with others, and counting how many decisions a test set has informed is rarely possible after the fact, which is the practical argument for a set used once.

**Greedy selection is crude.** A more careful analyst would regularise and stop earlier; the mechanism, deciding by a score and then reporting that score, is the same.""")

    post.add(
        "What to keep",
        f"""1. Coin-flip labels, never trained on: the majority vote of random guesses kept for beating 0.5 scored {a[1000]['public']:.3f} on the public labels after 1,000 submissions and {a[1000]['private']:.3f} on the private ones.

2. Features kept because they replicated on the holdout gave {d[100]['holdout']:.3f} there and {d[100]['fresh']:.3f} on fresh data.

3. Forward selection guided by the holdout reported {gh['holdout'][-1]:.3f} after 20 steps, above the true model's {gh['oracle']:.3f}; fresh accuracy was {gh['fresh'][-1]:.3f}. Guided by cross-validation instead, the holdout matched fresh rows to within {abs(gap_c):.3f}.

4. Treat every test score that informed a decision as part of training. Keep one test set for one final report, and count the decisions made with anything else.""")

    post.add(
        "Exercise",
        """List every decision made in your last modelling project after looking at a score on the validation or test set: features, models, hyperparameters, thresholds, early stopping, which runs to keep. Mark the ones made on the set you finally reported.

If any are marked, the reported score is optimistic by an unknown amount. Hold out a fresh sample, even a small one, score the final model on it once, and compare; the difference is a measurement of how much of the test set's honesty was spent.""")

    post.add(
        "Next",
        """Arc IX ends here, at forty-five episodes. Combining and comparing turned out to need the same care as fitting: an average that hands its decisions to its most confident member, a stack that learns from memorised predictions unless shown honest ones, naive Bayes counting the same evidence twice, tests that see a difference between equal algorithms one time in seven or more, and a test set that stays honest only as long as nobody acts on what it says.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
