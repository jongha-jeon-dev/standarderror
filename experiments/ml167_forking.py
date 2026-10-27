"""ML 67: Five Defensible Choices Turned Coin Flips Into 89%.

The sixty-seventh episode of *Machine Learning, Taught Through What
Breaks*, in Arc XIV. "If every choice in an analysis is defensible, so is
its result." Measured by making five ordinary choices in a row, each the one
that scored best, on labels that are coin flips:

* An honest cross-validated plan scored 0.484. Selecting features on all
  rows first raised the reported score to 0.746; the best of four models,
  ten fold splits and three feature counts took it to 0.805; reporting the
  better of two subgroups, to 0.888. The final pipeline scored 0.501 on
  5,000 fresh rows.
* No single step added more than 0.26, and four of the five added less
  than 0.09; each would pass a reviewer's look on its own.
* On breast cancer, where the signal is real, the same walk reported 0.977
  and delivered 0.965 on fresh tumours, against 0.975 for the honest plan.

Run: `standarderror run ml167_forking --publish`
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


def compute() -> dict:
    return {"noise": course.forking_paths(), "real": course.forking_real()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    nz = res["noise"]

    def walk(ax, m):
        x = np.arange(len(nz["steps"]))
        ax.fill_between(x, nz["reported_min"], nz["reported_max"],
                        color=m.series[0], alpha=0.18, lw=0)
        ax.plot(x, nz["reported"], color=m.series[0], marker="o",
                label="reported (cross-validated)")
        ax.axhline(nz["fresh"], color=m.series[1], lw=2,
                   label="final pipeline on fresh rows")
        ax.axhline(0.5, color=m.ink, ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" on all rows", "\non all rows")
                            .replace("best of ", "best of\n")
                            for s in nz["steps"]], fontsize=7.5)
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=8.5, loc="upper left", frameon=False)

    out["f0"] = charts.diagram(
        walk,
        title="Five choices, each the one that scored best, on coin flips",
        subtitle=("Reported cross-validated accuracy after each step, on 200 "
                  "rows of 1,000 noise features and random labels; band: "
                  "lowest to highest of 20 data sets. The line: the final "
                  "pipeline on 5,000 fresh rows."),
        xlabel="", ylabel="accuracy",
        figsize=(7.8, 4.5),
        source="Simulated; standarderror/ml/course.py.",
        alt=("A line climbing from 0.48 to 0.89 across a plan and five choices, with a "
             "band widening around it; a flat line at 0.50 for fresh data."),
        caption=(f"Reported after five choices: **{nz['reported'][-1]:.3f}** on "
                 f"average. Delivered: {nz['fresh']:.3f}."),
        path=str(IMG / f"ml167-f0-walk.{EXT}"))[0]

    rl = res["real"]
    body = []
    for i, s in enumerate(nz["steps"]):
        add = nz["reported"][i] - nz["reported"][i - 1] if i else 0.0
        body.append([str(i), s, f"{nz['reported'][i]:.3f}",
                     "-" if i == 0 else f"{add:+.3f}",
                     f"{rl['reported'][i]:.3f}"])
    body.append(["", "fresh rows, final pipeline", f"{nz['fresh']:.3f}", "",
                 f"{rl['fresh']:.3f}"])
    body.append(["", "fresh rows, honest plan", f"{nz['honest_fresh']:.3f}", "",
                 f"{rl['honest_fresh']:.3f}"])
    out["f1"] = charts.table_image(
        body,
        header=["step", "choice", "noise: reported", "added", "breast cancer: reported"],
        title="What each choice added",
        subtitle=("Means over 20 data sets of noise (200 rows) and 20 splits "
                  "of breast cancer (300 tumours to analyse, 269 fresh)."),
        source="Simulated and Breast Cancer Wisconsin via scikit-learn; "
               "standarderror/ml/course.py.",
        alt=("A table of six rows. On noise the reported score climbs by "
             "0.26, then smaller steps, to 0.888; fresh accuracy is 0.50. On "
             "breast cancer the reported score climbs slightly and the "
             "chosen pipeline does worse on fresh tumours than the honest "
             "plan."),
        caption="Each step keeps the previous steps' choices and picks the best option of its own.",
        align="rlrrr",
        path=str(IMG / f"ml167-f1-table.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    nz, rl = res["noise"], res["real"]

    def p1(panel, m):
        panel.plot(range(6), nz["reported"], color=m.series[0], lw=2.5,
                   marker="o")
        panel.set_ylim(0.4, 0.95)

    def p2(panel, m):
        panel.bar([0, 1], [nz["reported"][-1], nz["fresh"]],
                  color=[m.series[0], m.series[1]])
        panel.set_ylim(0, 1)

    def p3(panel, m):
        panel.bar([0, 1], [rl["fresh"], rl["honest_fresh"]],
                  color=[m.series[1], m.series[2]])
        panel.set_ylim(0.9, 1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=67,
        headline="Forking paths",
        panels=[(p1, f"{nz['reported'][-1]:.0%}", "reported, coin flips"),
                (p2, f"{nz['fresh']:.0%}", "delivered"),
                (p3, f"{rl['fresh']:.3f}", "real data, forked path")],
        note=("Each choice was small and reasonable. Together, made after "
              "looking at the results, they reported a model where there was "
              "nothing to model."),
        alt=("Three hand-drawn frames: a rising line; a tall bar beside a "
             "half-height one; two nearly equal bars."),
        path=str(IMG / f"ml167-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["walk"] = s.run("""
        import numpy as np
        from standarderror.ml import course

        g = np.random.default_rng(0)
        X, y = g.normal(size=(200, 1000)), g.integers(0, 2, 200)    # no signal
        Xf, yf = g.normal(size=(5000, 1000)), g.integers(0, 2, 5000)
        sub, subf = (g.normal(size=200) > 0) * 1, (g.normal(size=5000) > 0) * 1
        w = course._forked(X, y, Xf, yf, sub, subf)
        for step, score in zip(course.STEPS, w["reported"]):
            print(f"{step:30s} {score:.3f}")
        print("choices:", w["choices"], " fresh rows:", round(w["fresh"], 3))
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    nz, rl = res["noise"], res["real"]
    r = nz["reported"]
    adds = [b - a for a, b in zip(r, r[1:])]
    models = Counter(w["choices"]["model"] for w in nz["runs"])

    # The spine, asserted rather than trusted.
    assert abs(r[0] - 0.5) < 0.03 and abs(nz["fresh"] - 0.5) < 0.01
    assert r == sorted(r) and 0.85 < r[-1] < 0.92
    assert 0.2 < adds[0] < 0.3 and sum(a < 0.09 for a in adds[1:]) == 4
    assert min(nz["reported_min"][1:]) > 0.6 and nz["reported_max"][-1] > 0.9
    assert abs(nz["honest_fresh"] - 0.5) < 0.01
    assert rl["reported"][-1] > rl["fresh"] + 0.008
    assert rl["honest_fresh"] > rl["fresh"]
    assert len(models) >= 3

    post = Post(
        title=f"{SERIES_TAG} 67: Five Defensible Choices Turned Coin Flips Into 89%",
        slug="ml-67-forking-paths",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=67,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("On labels that were coin flips, five ordinary analysis "
                  "choices, each made after looking at the scores, reported "
                  "89% accuracy. None of them would have looked wrong on its "
                  "own."),
        summary=(
            f"200 rows of 1,000 noise features with random labels, 20 data "
            f"sets. An honest cross-validated plan scored {r[0]:.3f}. Five "
            f"choices, each the option that scored best, took the reported "
            f"score to {r[1]:.3f} (features selected on all rows), "
            f"{r[2]:.3f} (best of four models), {r[3]:.3f} (best of ten fold "
            f"splits), {r[4]:.3f} (best of three feature counts) and "
            f"{r[5]:.3f} (the better of two subgroups); the final pipeline "
            f"scored {nz['fresh']:.3f} on 5,000 fresh rows. On breast cancer "
            f"the same walk reported {rl['reported'][-1]:.3f} and delivered "
            f"{rl['fresh']:.3f}, less than the honest plan's "
            f"{rl['honest_fresh']:.3f}."),
        tags=["machine-learning", "reproducibility", "p-hacking",
              "evaluation", "statistics", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated (seeds 0-19): 200 rows, 1,000 standard Gaussian "
            "features, labels by coin flip, a subgroup defined by the sign "
            "of one more noise column; 5,000 fresh rows of the same.",
            "Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, via "
            "scikit-learn: 300 tumours to analyse and 269 fresh, 20 random "
            "splits; subgroups above and below the median texture.",
            "Machinery: scikit-learn pipelines (standardisation, F-test "
            "feature selection, logistic regression, RBF SVM, 15-NN, random "
            "forest) and 5-fold stratified CV; "
            "`standarderror/ml/course.py`, tested in `tests/test_ml.py`.",
            "Where this stops: Gelman and Loken, \"The garden of forking "
            "paths\" (2013); Simmons, Nelson and Simonsohn, \"False-positive "
            "psychology\", *Psychological Science* (2011); Cawley and Talbot, "
            "\"On over-fitting in model selection and subsequent selection "
            "bias in performance evaluation\", *JMLR* (2010); episodes 3, "
            "35, 45 and 66 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; about two minutes of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "One small choice at a time",
        f"""Single mistakes are the easy part of this series to teach: fit the scaler inside the folds, do not tune on the test set, do not trust the best of many scores. A real analysis rarely makes one mistake. It makes a sequence of choices, each reasonable, each taken after seeing how the last one turned out, and the series' last arc asks what such a sequence adds up to. Gelman and Loken called the space of such choices the garden of forking paths: an analyst need not try everything for the result to depend on which branch the data happened to favour. The sentence under test is the defence every analyst would make: *if every choice in an analysis is defensible, so is its result*.

The data are built to have nothing in them: 200 rows, 1,000 features of pure noise, labels by coin flip, so the truth is 50% for every model. The analysis starts with an honest plan, fixed in advance: standardise, select the 20 features most associated with the label inside each fold, fit logistic regression, report five-fold cross-validated accuracy. Then five choices are made one after another, each keeping the previous ones and taking whichever option scores best: select the features once on all the rows, because it is simpler; try four model families; try ten random fold assignments; try 10, 20 or 50 features; and report the subgroup, defined by one more column, in which the model works better. Every step is something a careful person has done. In a single run the last step can lower the score, because it picks the better of two halves of the data rather than the better of a half and the whole.

{snip['walk'].markdown()}""")

    post.add(
        "What it adds up to",
        f"""Averaged over 20 such data sets, the honest plan reported {r[0]:.3f}, as it should. Selecting features on all rows, the leak of episode 3, added {adds[0]:+.3f} by itself. Each later step added a little: {adds[1]:+.3f} for the best model, {adds[2]:+.3f} for the best fold split, {adds[3]:+.3f} for the best feature count and {adds[4]:+.3f} for the better subgroup. The final report averaged **{r[-1]:.3f}**, between {nz['reported_min'][-1]:.3f} and {nz['reported_max'][-1]:.3f} across the 20 data sets. The pipeline the walk ended with scored {nz['fresh']:.3f} on 5,000 fresh rows.

No step after the first added more than {max(adds[1:]):.3f}, and each has a respectable justification: model choice is what cross-validation is for, the fold split is arbitrary anyway, the number of features is a hyperparameter, and a subgroup effect is a finding. What makes them dangerous is not any one of them but the order: each was taken because of the score it produced, so each spent a little of the data's ability to tell signal from noise, and the reported score is the maximum of a search nobody wrote down. Simmons, Nelson and Simonsohn showed in 2011 that four such freedoms in a psychology experiment can push the false-positive rate from 5% to over 60%; Cawley and Talbot made the machine-learning version of the argument for model selection in 2010.

Which model won depended on the data set: across the 20, the walk chose {', '.join(f'{k} ' + ('once' if v == 1 else f'{v} times') for k, v in models.most_common())}. On noise there is no right answer for the search to find, so it finds whichever wrong one fits this sample best.""",
        figures=[figs["f0"]])

    post.add(
        "On real data the damage is quieter",
        f"""On breast cancer, where the signal is strong, the same walk looks harmless. The honest plan reported {rl['reported'][0]:.3f}; the five choices took the report to {rl['reported'][-1]:.3f}, a gain nobody would question. On the held-out tumours in the subgroup the walk chose, tumours that played no part in the analysis, the walk's final pipeline scored {rl['fresh']:.3f} and the honest plan, scored on the same tumours, **{rl['honest_fresh']:.3f}**. The search did not merely inflate the report; it chose a worse model than the plan it replaced, presumably because every choice was made on small differences in a small sample, and with a strong signal those differences are mostly noise.

That is the more common situation, and the harder one to see. On noise the reported 89% is implausible enough to invite a second look; on real data a reported 97.7% is entirely plausible, and only data that played no part in any choice can show that it is not real.

The honest plan cost nothing in either setting. Fixed before any score was seen, it delivered {rl['honest_fresh']:.3f} on the held-out tumours, {rl['honest_fresh'] - rl['fresh']:.3f} more than the searched pipeline, and on noise its fresh score was {nz['honest_fresh']:.3f}, as close to a coin as its report. A plan written in advance is often described as a constraint that leaves accuracy on the table; here the table was empty, and the search only moved the report away from the truth. Where a search does find a better model, the way to keep it honest is the one episode 35 measured for tuning: run the whole search, every choice included, inside an outer cross-validation loop, or score its single winner once on data held out from all of it.

The spread matters as much as the average. In the luckiest of the 20 noise data sets the walk reported {nz['reported_max'][-1]:.3f}, and in the least lucky {nz['reported_min'][-1]:.3f}; both would pass most reviews, and neither reviewer could tell from the number alone which kind of data set they were looking at.""",
        figures=[figs["f1"]])

    post.add(
        "Where this breaks",
        """**A greedy walk.** Each step took the best option given the previous ones. A real analyst explores less systematically and sometimes backtracks, which can add more optimism or less.

**Pure noise as the main case.** With no signal every gain is spurious, so the walk measures optimism at its largest. The breast-cancer version shows the other end; most problems sit between.

**Five particular choices.** Others are as common: dropping outliers by residual, transforming the target, choosing the metric. Any choice made after seeing a score belongs on the list.

**Selection on all rows at step one.** Without that leak the later steps would presumably add less, because each would be judged by honest cross-validation; the search would still inflate the maximum, as episode 35 measured for tuning. That variant was not run.""")

    post.add(
        "What to keep",
        f"""1. On coin-flip labels, an honest plan reported {r[0]:.3f}; five choices, each taken because it scored best, reported {r[-1]:.3f}; the result delivered {nz['fresh']:.3f}.

2. After the first leak, no single step added more than {max(adds[1:]):.3f}. The optimism was in the sequence, not in any step.

3. On breast cancer the walk reported {rl['reported'][-1]:.3f} and delivered {rl['fresh']:.3f}, below the honest plan's {rl['honest_fresh']:.3f}.

4. Write the analysis plan before looking at scores, count every choice made after looking, and keep a test set that no choice ever touched; report its score, not the search's.""")

    post.add(
        "Exercise",
        """For your last reported result, list every choice made after the first score was seen: features dropped or added, models tried, settings changed, rows excluded, subgroups examined, metrics switched.

Then count. If the list is longer than two, the reported number is the best of a search, and the honest number is the one from data that never influenced the list. If no such data exist, the next project should set them aside first.""")

    post.add(
        "Next",
        """Several of the problems in this walk shrink when there are more data and some do not. Episode 68 measures seven of the series' problems at growing sample sizes, and sorts them by what more data does to each.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
