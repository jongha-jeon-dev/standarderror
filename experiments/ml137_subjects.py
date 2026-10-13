"""ML 37: Rows From the Same People Are Not New Data.

The thirty-seventh episode of *Machine Learning, Taught Through What
Breaks*. "More rows means more data, and cross-validation over the rows
measures how the model will do." Measured on a simulation of subjects who
each contribute many rows, with a signature in their features and a
propensity in their outcome that do not carry over to new subjects:

* With 2,000 rows from 2,000 subjects, row-level and subject-level
  cross-validation agreed with accuracy on new subjects to a point. With
  the same 2,000 rows from 20 subjects, row-level cross-validation scored
  five-nearest-neighbours at 0.825 and it scored 0.685 on new people;
  row-level scores were higher with fewer subjects while the truth fell.
* Grouping the folds by subject repaired it, slightly pessimistically at
  20 subjects.
* 6,400 rows from 20 subjects trained a logistic model about as well as
  100 rows from 100 subjects, and worse than 400 rows from 400.

Run: `standarderror run ml137_subjects --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import rows
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    cv = rows.subject_cv()
    gr = rows.subject_growth()
    return {"cv": cv, "gr": gr, "same": rows.same_subject_neighbours(),
            "k": {(r["subjects"], r["model"]): r for r in cv},
            "g": {(r["rows"], r["how"]): r for r in gr}}


# ------------------------------------------------------------------ figures

def _cv_lines(ax, m, cv):
    S = sorted({r["subjects"] for r in cv})
    for j, model in enumerate(("5-NN", "forest")):
        rr = {r["subjects"]: r for r in cv if r["model"] == model}
        c = m.series[j + 1]
        ax.plot(S, [rr[s]["row"] for s in S], color=c, ls="--", marker="o",
                label=f"{model}: row-level CV")
        ax.plot(S, [rr[s]["subject"] for s in S], color=c, ls=":", marker="s",
                label=f"{model}: subject-level CV")
        ax.plot(S, [rr[s]["new"] for s in S], color=c, lw=2.2, marker="o",
                label=f"{model}: new subjects")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)


def _growth(ax, m, gr):
    N = sorted({r["rows"] for r in gr})
    for j, model in enumerate(("logistic", "forest")):
        for how, ls in (("new subjects", "-"), ("20 subjects", "--")):
            rr = {r["rows"]: r for r in gr if r["how"] == how}
            ax.plot(N, [rr[n][model] for n in N], color=m.series[j + 1], ls=ls,
                    marker="o", label=f"{model}, {'one row per subject' if how == 'new subjects' else 'from 20 subjects'}")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def figures(res: dict) -> dict:
    out: dict = {}
    cv, gr, k, g = res["cv"], res["gr"], res["k"], res["g"]

    out["f0"] = charts.diagram(
        lambda ax, m: _cv_lines(ax, m, cv),
        title="Row-level CV climbs as the people behind the rows disappear",
        subtitle=("2,000 rows split among fewer and fewer subjects. "
                  "Five-fold CV by rows and by subjects, against accuracy on "
                  "10,000 rows from 2,000 new subjects; means over 6 data "
                  "sets."),
        xlabel="subjects contributing the 2,000 rows (log scale, reversed)",
        ylabel="accuracy",
        figsize=(7.6, 4.8),
        source="Simulated; standarderror/ml/rows.py.",
        alt=("Dashed lines for row-level cross-validation climb as the "
             "number of subjects falls from 2,000 to 20; solid lines for "
             "accuracy on new subjects fall; dotted subject-level lines "
             "follow the solid ones."),
        caption=(f"From 20 subjects, row-level CV scored five-nearest-"
                 f"neighbours at {k[(20, '5-NN')]['row']:.3f}; on new people "
                 f"it scored **{k[(20, '5-NN')]['new']:.3f}**."),
        path=str(IMG / f"ml137-f0-cv.{EXT}"))[0]

    out["f1"] = charts.diagram(
        lambda ax, m: _growth(ax, m, gr),
        title="More rows from the same people buy little",
        subtitle=("Accuracy on new subjects as the training data grow from "
                  "100 to 6,400 rows: one row from each new subject (solid), "
                  "or more rows from the same 20 subjects (dashed)."),
        xlabel="training rows (log scale)",
        ylabel="accuracy on new subjects",
        figsize=(7.4, 4.6),
        source="Simulated; standarderror/ml/rows.py.",
        alt=("Solid lines rise with the number of rows and level off near "
             "0.81 for logistic regression and 0.80 for the forest; dashed "
             "lines stay below them at every size."),
        caption=(f"6,400 rows from 20 subjects: logistic regression "
                 f"{g[(6400, '20 subjects')]['logistic']:.3f}. 400 rows from "
                 f"400 subjects: **{g[(400, 'new subjects')]['logistic']:.3f}**."),
        path=str(IMG / f"ml137-f1-growth.{EXT}"))[0]

    body = [[f"{r['subjects']:,}", f"{r['per']}", r["model"], f"{r['row']:.3f}",
             f"{r['subject']:.3f}", f"{r['new']:.3f}",
             f"{r['row'] - r['new']:+.3f}"] for r in cv]
    out["f2"] = charts.table_image(
        body,
        header=["subjects", "rows each", "model", "row-level CV",
                "subject-level CV", "new subjects", "row CV minus truth"],
        title="Two thousand rows, five ways to collect them",
        subtitle=("Five-fold cross-validation with rows shuffled into folds, "
                  "or with each subject's rows kept in one fold; accuracy on "
                  "new subjects is the target. Means over 6 data sets."),
        source="Simulated; standarderror/ml/rows.py.",
        alt=("A fifteen-row table. With one row per subject the three "
             "accuracy columns agree; with 20 subjects row-level CV is "
             "4 to 14 points above accuracy on new subjects."),
        caption=("Row-level cross-validation measures how well the model "
                 "knows the people it has met. Subject-level "
                 "cross-validation measures how well it does on new ones."),
        align="rrlrrrr",
        path=str(IMG / f"ml137-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    k, g = res["k"], res["g"]

    def cvp(panel, m):
        S = [2000, 400, 100, 40, 20]
        panel.plot([k[(s, "5-NN")]["row"] for s in S], color=m.series[1], lw=2)
        panel.plot([k[(s, "5-NN")]["new"] for s in S], color=m.ink, lw=2)

    def grp(panel, m):
        panel.bar([0, 1], [k[(20, "5-NN")]["subject"], k[(20, "5-NN")]["new"]],
                  color=[m.series[2], m.ink])
        panel.set_ylim(0.6, 0.72)

    def grow(panel, m):
        N = [100, 400, 1600, 6400]
        panel.plot([g[(n, "new subjects")]["logistic"] for n in N],
                   color=m.series[1], lw=2)
        panel.plot([g[(n, "20 subjects")]["logistic"] for n in N],
                   color=m.series[2], lw=2, ls="--")

    return charts.lecture_hero(
        series=SERIES_TAG, episode=37,
        headline="Rows are not people",
        panels=[(cvp, f"{k[(20, '5-NN')]['row'] - k[(20, '5-NN')]['new']:+.3f}",
                 "row CV over truth"),
                (grp, f"{k[(20, '5-NN')]['subject']:.3f}", "subject-level CV"),
                (grow, f"{g[(6400, '20 subjects')]['logistic']:.3f}",
                 "6,400 rows, 20 people")],
        note=("Rows from the same person share what makes that person "
              "different. Folds that split a person reward learning the "
              "person; only new people tell you what carries over."),
        alt=("Three hand-drawn frames: two lines parting; two bars of "
             "different heights; two rising lines, one below the other."),
        path=str(IMG / f"ml137-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["cv"] = s.run("""
        from standarderror.ml import rows

        for r in rows.subject_cv():   # 2,000 rows each time
            if r["model"] == "5-NN":
                print(f"{r['subjects']:5d} subjects x {r['per']:3d} rows: "
                      f"row CV {r['row']:.3f}, subject CV {r['subject']:.3f}, "
                      f"new people {r['new']:.3f}")
        """)
    out["growth"] = s.run("""
        for r in rows.subject_growth():
            print(f"{r['rows']:5d} rows, {r['how']:12s}: logistic "
                  f"{r['logistic']:.3f}, forest {r['forest']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    k, g = res["k"], res["g"]
    same = res["same"]
    kn1, kn20 = k[(2000, "5-NN")], k[(20, "5-NN")]
    f1, f20 = k[(2000, "forest")], k[(20, "forest")]
    l1, l20 = k[(2000, "logistic")], k[(20, "logistic")]
    g_l = {(n, h): g[(n, h)]["logistic"] for (n, h) in g}
    g_f = {(n, h): g[(n, h)]["forest"] for (n, h) in g}
    knn = [k[(s, "5-NN")] for s in (2000, 400, 100, 40, 20)]

    # The spine, asserted rather than trusted.
    assert all(abs(r["row"] - r["new"]) < 0.01 and abs(r["subject"] - r["new"]) < 0.01
               for r in (kn1, f1, l1))
    assert knn[0]["row"] < knn[1]["row"] < min(r["row"] for r in knn[2:])
    assert min(r["row"] for r in knn[2:]) > 0.81
    assert all(a["new"] > b["new"] for a, b in zip(knn, knn[1:]))
    assert kn20["row"] - kn20["new"] > 0.13 and f20["row"] - f20["new"] > 0.07
    assert l20["row"] - l20["new"] > 0.03
    assert same > 0.95
    assert f20["row"] == max(k[(S, "forest")]["row"] for S in (2000, 400, 100, 40, 20))
    assert all(0 < k[(400, m)]["subject"] - k[(400, m)]["new"] < 0.015
               for m in ("logistic", "5-NN", "forest"))
    assert all(0 < r["new"] - r["subject"] < 0.035 for r in (kn20, f20, l20))
    assert g_l[(6400, "20 subjects")] < g_l[(400, "new subjects")]
    assert abs(g_l[(6400, "20 subjects")] - g_l[(100, "new subjects")]) < 0.005
    assert g_f[(6400, "20 subjects")] < g_f[(400, "new subjects")]
    assert all(g_l[(n, "20 subjects")] < g_l[(n, "new subjects")]
               and g_f[(n, "20 subjects")] < g_f[(n, "new subjects")]
               for n in (100, 400, 1600, 6400))
    assert g_l[(6400, "20 subjects")] - g_l[(100, "20 subjects")] < 0.02

    post = Post(
        title=f"{SERIES_TAG} 37: Rows From the Same People Are Not New Data",
        slug="ml-37-subjects",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=37,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("When subjects contribute many rows each, row-level "
                  "cross-validation measures how well the model knows the "
                  "people it has met, and extra rows from the same people "
                  "are worth little."),
        summary=(
            f"2,000 simulated rows from 2,000, 400, 100, 40 or 20 subjects, "
            f"each subject with a feature signature and an outcome "
            f"propensity that do not carry over. With 2,000 subjects, "
            f"row-level and subject-level cross-validation matched accuracy "
            f"on new people to within a point. With 20 subjects, row-level "
            f"cross-validation scored five-nearest-neighbours at "
            f"{kn20['row']:.3f} against {kn20['new']:.3f} on new people, and "
            f"a forest at {f20['row']:.3f} against {f20['new']:.3f}; "
            f"subject-level folds gave {kn20['subject']:.3f} and "
            f"{f20['subject']:.3f}. 6,400 rows from 20 subjects trained a "
            f"logistic model to {g_l[(6400, '20 subjects')]:.3f}, about what "
            f"100 rows from 100 subjects gave."),
        tags=["machine-learning", "data-science", "statistics",
              "cross-validation", "grouped-data", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: each subject has a signature in ten features "
            "(normal, sd 2) added to every one of its rows, and a propensity "
            "(normal, sd 2) added to every row's log-odds; the part of the "
            "outcome that carries over to new subjects is 1.5 x_1 + x_2. "
            "Six data sets per design (seeds 0 to 5); new subjects: 2,000 "
            "of them with 5 rows each (seeds 1,000 to 1,005).",
            "Machinery: `standarderror/ml/rows.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's LogisticRegression, "
            "KNeighborsClassifier (k = 5), RandomForestClassifier (50 "
            "trees), KFold with shuffling and GroupKFold, five folds each.",
            "Where this stops: Kish, *Survey Sampling* (1965) on the design "
            "effect; Saeb et al., \"The need to approximate the use-case in "
            "clinical machine learning\", *GigaScience* (2017); Roberts et "
            "al., *Ecography* (2017); episode 3 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("seeds as listed; shuffled folds from the data "
                            "set's seed; about two minutes of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Twenty people, a hundred rows each",
        f"""Episode 36 found that rows close in time are a problem only when they share something the model cannot see. People are the other common source of rows that share something: a patient measured at every visit, a customer with a year of transactions, a speaker with a hundred recordings, a writer with a page of digits. Episode 3 measured exact duplicates straddling folds and noted that near-duplicates, such as the same patient twice, leak by the same mechanism with less force. This episode measures that case, and the sentence that makes it easy to miss: *more rows means more data, and cross-validation over the rows measures how the model will do*.

The simulation gives each subject two things that belong to them alone. One is a signature: an offset added to all ten of their features, so their rows sit together in feature space. The other is a propensity: an offset added to the log-odds of all their outcomes, so some people are simply more often positive than others. Neither tells you anything about a new person. What does carry over is a relationship between two features and the outcome, the same for everyone. Each design has 2,000 rows; what changes is how many people they come from.

{snip['cv'].markdown()}

With one row from each of 2,000 people, row-level cross-validation and the truth agree: **{kn1['row']:.3f}** against {kn1['new']:.3f}. As the same 2,000 rows come from fewer people, the two move in opposite directions. Row-level cross-validation rises, to above 0.81 from 100 people down and **{kn20['row']:.3f}** with 20, while accuracy on new people falls at every step, to **{kn20['new']:.3f}**. Read through row-level folds, the data sets built from 100 people or fewer look like the best of the five, and the one from 20 people scores the forest highest of all.""",
        figures=[figs["f0"]])

    post.add(
        "What the folds were measuring",
        f"""A row-level fold holds out one-fifth of each person's rows and trains on the other four-fifths. A held-out row's nearest neighbours in feature space are almost all rows from the same person: with 20 people, {same:.1%} of the five nearest training rows, because the signature puts them together, and those rows carry the same propensity. Five-nearest-neighbours answers by asking the person's other rows what usually happens to this person, which is an excellent way to predict a person it has already met and no way at all to predict a new one.

The forest finds the same shortcut through its splits: {f20['row']:.3f} under row-level folds, {f20['new']:.3f} on new people. Logistic regression, which can only draw one plane through the feature space, has less room to memorise people, and its gap is smaller but not zero: {l20['row']:.3f} against {l20['new']:.3f}. A plausible reason, not isolated here, is that with 20 people a few of them can by chance have signatures that line up with their propensities, and a linear model fits that accident too.

The repair is to make the folds match the question. Subject-level cross-validation keeps every person's rows in a single fold, so each held-out row belongs to someone the model has never seen. It scored the nearest-neighbour model at **{kn20['subject']:.3f}** and the forest at {f20['subject']:.3f}, slightly below the truth, likely in part because each fold trains on 16 people instead of 20. That pessimism is a feature of very few groups: with 400 subjects the same folds were a little optimistic, by up to {max(k[(400, m)]['subject'] - k[(400, m)]['new'] for m in ('logistic', '5-NN', 'forest')):.3f}.""",
        figures=[figs["f2"]])

    post.add(
        "What more rows buy",
        f"""The second half of the sentence, that more rows mean more data, is measured directly by growing the training set two ways: one row from each of N new people, or N/20 rows from each of the same 20 people.

{snip['growth'].markdown()}

One row per person keeps improving: logistic regression goes from {g_l[(100, 'new subjects')]:.3f} at 100 rows to {g_l[(6400, 'new subjects')]:.3f} at 6,400. Twenty people with more and more rows each barely move: from {g_l[(100, '20 subjects')]:.3f} to **{g_l[(6400, '20 subjects')]:.3f}**, about what 100 rows from 100 people gave ({g_l[(100, 'new subjects')]:.3f}) and below what 400 rows from 400 people gave ({g_l[(400, 'new subjects')]:.3f}). For the forest, 6,400 rows from 20 people reached {g_f[(6400, '20 subjects')]:.3f}, again below 400 rows from 400 people ({g_f[(400, 'new subjects')]:.3f}).

Survey statisticians have a name for this: the design effect. Kish's formula says that m rows from each cluster, with intra-cluster correlation ρ, are worth n / (1 + (m − 1)ρ) independent rows. When the thing that varies between people is large compared with what varies within them, as it is here, extra rows from the same person add almost nothing about what is true of people in general. They add a great deal about that person, which is what row-level folds reward.""",
        figures=[figs["f1"]])

    post.add(
        "Which question is being asked",
        """Row-level cross-validation is not wrong in itself; it answers a different question. If the model will only ever be used on the people it was trained on, such as predicting a known customer's next purchase or a known patient's next reading, then rows from the same person are exactly the relevant test, and subject-level folds would be too pessimistic. Saeb and colleagues made the point for clinical models built from smartphone data: the right split is the one that imitates the use. The mistake is to answer one question and report the answer to the other.

The same logic covers any grouping the deployment will not share: hospitals when a model moves to a new hospital, schools, devices, plots of land in ecology, and the writers of handwritten digits. Roberts and colleagues reviewed it for temporal, spatial and phylogenetic groupings and reached the same rule: block the folds by whatever the new data will be new in.""")

    post.add(
        "Where this breaks",
        """**Strong subject effects, by construction.** The signature and the propensity both have a standard deviation of 2, large enough that a person's rows are easy to recognise. Where between-person differences are small, the gap between the two kinds of cross-validation shrinks towards the one-row-per-person case, where it vanishes.

**Effects that do not transfer at all.** Here nothing about a person helps with a new one. In practice some person-level traits are measured, and a model that uses them generalises better; the problem is the part that is not measured.

**Twenty subjects is few.** Subject-level cross-validation with 20 groups trains on 16 of them per fold and its estimates are noisy from data set to data set. With very few subjects, leave-one-subject-out folds and an honest statement of the uncertainty are the best available.

**Balanced subjects.** Every subject contributed the same number of rows. With a few heavy contributors, row-level scores are dominated by them, and the distortion can be worse.""")

    post.add(
        "What to keep",
        f"""1. Rows from the same person share what makes that person different. With 2,000 rows from 20 people, row-level cross-validation scored five-nearest-neighbours at {kn20['row']:.3f}; on new people it scored {kn20['new']:.3f}.

2. As the same rows came from fewer people, row-level scores went up, from {kn1['row']:.3f} to above 0.81, while true accuracy fell at every step, so row-level folds favoured exactly the data sets that generalise worst.

3. With 20 subjects, subject-level folds matched the truth to within {max(r['new'] - r['subject'] for r in (kn20, f20, l20)):.3f}, slightly pessimistic.

4. 6,400 rows from 20 people trained logistic regression to {g_l[(6400, '20 subjects')]:.3f}, about what 100 rows from 100 people gave. Count people, not rows, and split the folds by whatever the deployment will be new in.""")

    post.add(
        "Exercise",
        """For a data set with an identifier for the source of each row (person, account, device, site), count the rows per identifier and compute the intra-class correlation of your target within identifiers.

Then cross-validate your model twice, once with shuffled rows and once with folds grouped by the identifier. The difference between the two scores is how much of your model's apparent skill is recognising sources it has already seen; plug the correlation into Kish's formula to see how many independent rows you really have.""")

    post.add(
        "Next",
        """Time and people decide which rows resemble each other. Episode 38 turns to rows that are incomplete, and asks whether a missing value is noise to be filled in or information the process that produced the data left behind.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
