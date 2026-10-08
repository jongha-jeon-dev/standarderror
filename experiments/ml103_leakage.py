"""ML 3: Not Every Leak Leaks, and the Ones That Do Can Be Measured.

The third episode of *Machine Learning, Taught Through What Breaks*. "Fit
every preprocessing step inside the cross-validation folds" is right, and is
usually taught as if every violation were equally bad. Six violations,
measured against their honest versions, range from nothing to a third of
the accuracy scale.

Measured:

* Standardising or mean-imputing on all rows: no measurable leak.
* Selecting 20 of 5,000 pure-noise features on all rows turns coin-flip
  labels into 0.88 cross-validated accuracy; inside the folds, 0.52. The
  leak grows with the number of candidates and is zero when there is nothing
  to choose.
* Selecting 5 of 1,030 features on the real breast-cancer data, also on all
  rows: no leak, because the real features win the selection anyway.
* Target-encoding a 200-level noise column on all rows: +2.3 points.
* Duplicated rows straddling folds: +0.9 points for logistic regression,
  +3.7 for 1-NN.

Run: `standarderror run ml103_leakage --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import leakage as lk
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 8)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"noise": lk.noise_selection(reps=50),
            "sweep": lk.selection_sweep(),
            "table": lk.leak_table(reps=20)}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    sw = res["sweep"]
    ps = [r["candidates"] for r in sw]

    def sweep(ax, m):
        ax.plot(ps, [r["leaky"] for r in sw], marker="o", ms=6, lw=2.4,
                color=m.series[1], label="selection fitted on all rows")
        ax.fill_between(ps, [r["leaky"] - r["leaky_sd"] for r in sw],
                        [r["leaky"] + r["leaky_sd"] for r in sw],
                        color=m.series[1], alpha=0.15, linewidth=0)
        ax.plot(ps, [r["honest"] for r in sw], marker="s", ms=6, lw=2.4,
                color=m.series[0], label="selection inside each fold")
        ax.axhline(0.5, lw=1, color=m.ink_secondary)
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    n = res["noise"]
    out["f0"] = charts.diagram(
        sweep,
        title="A selection step finds signal in coin flips",
        subtitle=(f"{n['n']} rows, coin-flip labels, 20 features chosen by "
                  f"F-test from a pool of pure noise, then 5-fold "
                  f"cross-validated logistic regression. Band: one SD over "
                  f"30 datasets."),
        xlabel="pure-noise candidate features",
        ylabel="cross-validated accuracy",
        source="Simulated; standarderror/ml/leakage.py.",
        alt=("A rising line from 0.5 to near 0.9 as the number of noise "
             "candidates grows, and a flat line at 0.5."),
        caption=(f"With 20 candidates there is nothing to choose and nothing "
                 f"leaks. With {ps[-1]:,} the leaky estimate is "
                 f"**{sw[-1]['leaky']:.2f}** for labels that are coin flips."),
        path=str(IMG / f"ml103-f0-sweep.{EXT}"))[0]

    tab = res["table"]
    names = [r["leak"] for r in tab]

    def gaps(ax, m):
        ys = np.arange(len(tab))[::-1]
        for yv, r in zip(ys, tab):
            ax.barh(yv, r["gap"] * 100, height=0.6,
                    color=m.series[1] if r["uses_label"] else m.series[0])
            ax.annotate(f"{r['gap'] * 100:+.2f}",
                        (max(r["gap"] * 100, 0) + 0.08, yv), va="center",
                        fontsize=8.6, color=m.ink)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=8.6)
        ax.axvline(0, lw=1, color=m.ink_secondary)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=m.series[1],
                                 label="the step looks at the label"),
                           Patch(color=m.series[0], label="it does not")],
                  frameon=False, fontsize=8.4, loc="upper right")

    out["f1"] = charts.diagram(
        gaps,
        title="Six leaks on one dataset, measured",
        subtitle=("Cross-validated accuracy with the step fitted on all rows, "
                  "minus the same with it fitted inside the folds. Breast "
                  "cancer data, 5-fold CV, 20 fold assignments."),
        xlabel="leak (accuracy points)",
        ylabel="",
        figsize=(7.6, 4.4),
        source="Measured; standarderror/ml/leakage.py.",
        alt=("Horizontal bars: three at or near zero for standardising, "
             "imputing and feature selection, and three clearly positive "
             "for target encoding and the two duplicated-row cases."),
        caption=("Neither colour predicts the size. Looking at the label is "
                 "necessary for a selection leak and not sufficient; "
                 "duplicates leak without looking at it at all."),
        path=str(IMG / f"ml103-f1-leaks.{EXT}"))[0]

    rows_t = [[r["leak"], "yes" if r["uses_label"] else "no",
               f"{r['honest']:.4f}", f"{r['leaky']:.4f}",
               f"{r['gap'] * 100:+.2f}"] for r in tab]
    rows_t.insert(0, [f"select 20 of {n['p']:,} noise features "
                      f"(coin-flip labels)", "yes",
                      f"{n['honest'].mean():.4f}", f"{n['leaky'].mean():.4f}",
                      f"{(n['leaky'].mean() - n['honest'].mean()) * 100:+.1f}"])
    out["f2"] = charts.table_image(
        rows_t,
        header=["fitted on all rows", "uses label", "honest CV",
                "leaky CV", "leak (pts)"],
        title="The same rule, broken seven ways",
        subtitle=("First row: simulated noise, 100 rows, 50 datasets. The "
                  "rest: Wisconsin breast cancer, 569 rows."),
        source="Measured; standarderror/ml/leakage.py.",
        alt=("A seven-row table of leaky and honest cross-validated "
             "accuracies; the first row's leak is about 35 points, the rest "
             "between zero and four."),
        caption=("The ranking is the lesson: a leak is the room a step has "
                 "to fit the test rows, not whether it touched them."),
        bold_cells={(0, c) for c in range(5)}, align="lcrrr",
        path=str(IMG / f"ml103-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    sw, tab, n = res["sweep"], res["table"], res["noise"]

    def rise(panel, m):
        panel.plot(np.log([r["candidates"] for r in sw]),
                   [r["leaky"] for r in sw], lw=2.8, color=m.series[1])
        panel.plot(np.log([r["candidates"] for r in sw]),
                   [r["honest"] for r in sw], lw=2.4, color=m.series[0])

    def flat(panel, m):
        vals = [r["gap"] for r in tab[:2]]
        panel.bar([0, 1], vals, color=m.series[0])
        panel.set_ylim(-0.04, 0.04)
        panel.plot([-0.5, 1.5], [0, 0], lw=1.6, color=m.ink)

    def dup(panel, m):
        vals = [r["gap"] for r in tab[-2:]]
        panel.bar([0, 1], vals, color=m.series[0])
        panel.set_ylim(0, 0.045)

    knn = tab[-1]["gap"]
    return charts.lecture_hero(
        series=SERIES_TAG, episode=3,
        headline="A leak is room to fit the test rows",
        panels=[(rise, f"{n['leaky'].mean():.2f}",
                 "CV score on coin flips"),
                (flat, f"{max(abs(tab[0]['gap']), abs(tab[1]['gap'])) * 100:.2f} pts",
                 "scaling, imputing"),
                (dup, f"+{knn * 100:.1f} pts", "duplicates, 1-NN")],
        note=("Fitting a step on all rows before cross-validating leaks "
              "anything from nothing to a third of the accuracy scale. "
              "Standardising leaks nothing; selecting from noise leaks the "
              "most; duplicates leak in proportion to how local the model "
              "is."),
        alt=("Three hand-drawn frames: a rising line over a flat one; two "
             "tiny bars on a zero line; and two bars of different height."),
        path=str(IMG / f"ml103-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["noise"] = s.run("""
        from standarderror.ml import leakage as lk

        # 100 rows, 5,000 noise features, coin-flip labels, 50 datasets.
        r = lk.noise_selection(n=100, p=5000, k=20, reps=50)
        print(f"select on all rows, then CV:  {r['leaky'].mean():.3f}"
              f"   (lowest {r['leaky'].min():.2f})")
        print(f"select inside each fold:      {r['honest'].mean():.3f}"
              f"   (highest {r['honest'].max():.2f})")
        """)
    out["table"] = s.run("""
        for row in lk.leak_table(reps=20):
            print(f"{row['leak']:40s} honest {row['honest']:.4f}  "
                  f"leaky {row['leaky']:.4f}  {row['gap'] * 100:+.2f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    n, sw = res["noise"], res["sweep"]
    t = {r["leak"]: r for r in res["table"]}
    std = t["standardise on all rows"]
    imp = t["impute means on all rows"]
    te = t["target-encode a noise ID on all rows"]
    sel = t["select 5 of 1,030 features on all rows"]
    dl = t["duplicate rows across folds (logistic)"]
    dk = t["duplicate rows across folds (1-NN)"]
    noise_gap = n["leaky"].mean() - n["honest"].mean()
    p_cv = 0.8

    # The spine, asserted rather than trusted.
    assert noise_gap > 0.3 and abs(n["honest"].mean() - 0.5) < 0.05
    assert all(a["leaky"] <= b["leaky"] + 0.02 for a, b in zip(sw, sw[1:]))
    assert abs(sw[0]["leaky"] - sw[0]["honest"]) < 1e-12
    assert abs(std["gap"]) < 0.003 and abs(imp["gap"]) < 0.003
    assert abs(sel["gap"]) < 0.003
    assert te["gap"] > 0.015 and dk["gap"] > 3 * dl["gap"] > 0
    expected_knn = p_cv * 1.0 + (1 - p_cv) * dk["honest"]

    post = Post(
        title=(f"{SERIES_TAG} 3: Not Every Leak Leaks, and the Ones That Do "
               f"Can Be Measured"),
        slug="ml-3-leakage",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=3,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("'Fit every preprocessing step inside the folds' is "
                  "right, and taught as if every violation were equally "
                  "bad. Measured, seven of them range from nothing to a "
                  "third of the accuracy scale."),
        summary=(
            f"Fit a step on all the rows, then cross-validate, and compare "
            f"with the same step fitted inside the folds. Standardising "
            f"leaks {std['gap'] * 100:+.2f} points on the breast cancer "
            f"data and mean imputation {imp['gap'] * 100:+.2f}: nothing. "
            f"Selecting 20 of 5,000 pure-noise features turns coin-flip "
            f"labels into {n['leaky'].mean():.2f} cross-validated accuracy "
            f"against {n['honest'].mean():.2f} done honestly, and the leak "
            f"grows with the number of candidates - yet selecting 5 of "
            f"1,030 features on the real data leaks "
            f"{sel['gap'] * 100:+.2f}, because the real features win "
            f"anyway. Target-encoding a noise ID leaks "
            f"{te['gap'] * 100:+.1f} points; duplicated rows leak "
            f"{dl['gap'] * 100:+.1f} for logistic regression and "
            f"{dk['gap'] * 100:+.1f} for 1-NN."),
        tags=["machine-learning", "data-science", "statistics",
              "data-leakage", "cross-validation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Breast Cancer Wisconsin (Diagnostic), as bundled with "
            "scikit-learn (`load_breast_cancer`): 569 rows, 30 features, "
            "CC BY 4.0. Wolberg, Street and Mangasarian (1995), UCI Machine "
            "Learning Repository.",
            "Simulated: standard-normal noise features and fair coin-flip "
            "labels, so the true accuracy of any model is 0.5.",
            "Machinery: `standarderror/ml/leakage.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Ambroise and McLachlan, \"Selection bias in "
            "gene extraction on the basis of microarray gene-expression "
            "data\", *PNAS* (2002); Kaufman, Rosset, Perlich and Stitelman, "
            "\"Leakage in data mining: formulation, detection, and "
            "avoidance\", *ACM TKDD* (2012); Kapoor and Narayanan, "
            "\"Leakage and the reproducibility crisis in machine-learning-"
            "based science\", *Patterns* (2023).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("noise datasets from one seeded generator; every "
                            "leaky and honest pair uses the same 5-fold "
                            "assignment, seeded by replicate"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "One rule, taught as one rule",
        f"""Episode 2 ended on what cross-validation estimates. This one is about the most common way to make it estimate something else. Every course teaches the rule: anything fitted to data — a scaler, an imputer, a feature selector, an encoder — must be fitted inside each training fold and applied to the held-out fold, never fitted on all the rows first. Violations are called leakage, and Kapoor and Narayanan found them in hundreds of published papers across seventeen fields.

The rule is right. What the usual teaching leaves out is that violations differ in size by orders of magnitude, and the size is predictable. That matters in practice, because the expensive question is never "did we leak?" but "how much of the reported number is leak?"

So here are seven violations, each measured against its honest version on the same folds. Start with the famous one. A hundred rows, 5,000 features of pure noise, labels from a fair coin. Pick the 20 features most associated with the label, then cross-validate logistic regression on them.

{snip['noise'].markdown()}

Selected on all rows, the coin flips cross-validate at **{n['leaky'].mean():.3f}**; no dataset of the fifty came in below {n['leaky'].min():.2f}. Selected inside each fold, {n['honest'].mean():.3f} — which is what labels from a coin deserve. That is Ambroise and McLachlan's result from microarray studies in 2002, reproduced in a few lines.""")

    post.add(
        "The leak is the room to choose",
        f"""Why so large? The selection step looked at all 100 labels, including the ones each fold would later hold out, and with 5,000 candidates it can always find 20 columns that happen to line up with them. The cross-validation then faithfully measures how well those columns predict labels they were chosen to predict.

That suggests the size of the leak is set by how much choice the step has, and it is. Sweep the number of noise candidates the 20 are chosen from.

The leak is exactly zero with 20 candidates — there is nothing to choose — and climbs steadily: {sw[1]['leaky']:.2f} with {sw[1]['candidates']}, {sw[2]['leaky']:.2f} with {sw[2]['candidates']}, {sw[3]['leaky']:.2f} with {sw[3]['candidates']:,}, {sw[4]['leaky']:.2f} with {sw[4]['candidates']:,}. The honest line never moves off 0.5.""",
        figures=[figs["f0"]])

    post.add(
        "Six more, on real data",
        f"""On the Wisconsin breast cancer data, 569 rows and 30 real features, here are six more violations, each scored with its honest version on 20 fold assignments.

{snip['table'].markdown()}

**Standardising and imputing on all rows leak nothing measurable**: {std['gap'] * 100:+.2f} and {imp['gap'] * 100:+.2f} points. Neither looks at the label, and a mean and a standard deviation estimated from 569 rows instead of 455 differ by too little to move a decision boundary. These are the violations most often flagged in code review, and they are the least consequential.

**Selecting features on all rows leaked nothing either** — {sel['gap'] * 100:+.2f} points — even though the selection looked at every label and chose 5 of 1,030 columns, 1,000 of them pure noise. The real features are so much stronger than any noise column's chance alignment that the selection picks them whether or not it sees the held-out labels. The noise experiment above leaked because there was nothing real to find. Using the label is necessary for this kind of leak and not sufficient: what leaks is the room to fit chance.

**Target-encoding a noise ID leaks {te['gap'] * 100:+.1f} points.** A column of 200 random categories, each replaced by its mean label, carries no information; encoded on all rows, each category's mean includes the held-out rows' own labels, and the model learns to read them back.

**Duplicated rows leak without looking at the label at all**: {dl['gap'] * 100:+.1f} points for logistic regression, {dk['gap'] * 100:+.1f} for 1-NN. Copy every row once and split at random, and most copies land in a different fold from their twin. A smooth model gains a little from having seen its test row; a nearest-neighbour model *is* a lookup of its test row. Its leaky score is almost exactly what that predicts: with about {p_cv:.0%} of twins in another fold and the rest scored honestly, {p_cv:.0%} × 1 + {1 - p_cv:.0%} × {dk['honest']:.3f} = {expected_knn:.3f}, against {dk['leaky']:.3f} measured.""",
        figures=[figs["f1"]])

    post.add(
        "A rule for sizing a leak",
        """The measurements support one rule of thumb, sharper than "fit everything inside the folds": **a step leaks to the extent that it gives the model room to fit the test rows.** That room comes from two places.

One is choice informed by the labels — selection, encoding, tuning — and its size is set by how many options the step has relative to how much real signal there is to find. Plenty of real signal and the choice is made the same way with or without the held-out labels; pure noise and every candidate is a chance to fit them.

The other is the test row itself, or a near-copy of it, being present in training. Its size is set by how local the model is: logistic regression barely notices a duplicate, 1-NN returns it.

Steps that do neither — estimating a mean, a scale, a missing-value fill from a few hundred rows — do not leak in any amount worth measuring. Fit them inside the folds anyway, because it costs nothing; but when auditing a result, start with the steps that can choose.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        """**Small data makes the harmless steps less harmless.** At 569 rows a scaler fitted on 455 or 569 rows is the same scaler. With 30 rows and a heavy-tailed feature it would not be, and standardising could leak a little through one extreme row.

**Real duplicates are rarely exact.** Near-duplicates — augmented images, resampled time series, the same patient twice — leak by the same mechanism with less force, and are much harder to find. The 1-NN number is the upper end of the range.

**Tuning is the leak not measured here.** Choosing hyperparameters on the same cross-validation you report is label-informed choice, so by the rule above its size depends on how many configurations were tried and how much they differ. The earlier post "I Trained 2,000 Models on a Coin Flip and the Best One Looked Great" measures that one: the best of many configurations looks good on noise for the same reason the selected features do.

**One dataset.** The breast-cancer features are strong, which is why selection leaked nothing on them. On a dataset with weak signal and many features — the setting where selection is used most — it will look much more like the noise experiment.""")

    post.add(
        "What to keep",
        f"""1. Fitting a step on all rows before cross-validating can leak anything from nothing to a third of the accuracy scale. Measure it: run the honest version on the same folds and subtract.

2. Standardising and mean imputation on all rows leaked {std['gap'] * 100:+.2f} and {imp['gap'] * 100:+.2f} points here. Not worth an alarm; still worth fixing.

3. Feature selection on all rows turned coin flips into {n['leaky'].mean():.2f} accuracy, growing with the number of candidates — and leaked {sel['gap'] * 100:+.2f} on real data where the real features dominate.

4. Target-encoding on all rows leaked {te['gap'] * 100:+.1f} points from a column of pure noise.

5. Duplicates leak without touching the label, {dl['gap'] * 100:+.1f} points for logistic regression and {dk['gap'] * 100:+.1f} for 1-NN.

6. A step leaks to the extent it gives the model room to fit the test rows: label-informed choice, or the test row itself in training.""")

    post.add(
        "Exercise",
        """Take a pipeline you have shipped and list every step that is fitted to data. Mark each one: does it look at the label, and how many options does it choose between? Does any row, or a near-copy of a row, appear on both sides of your splits?

Then measure the one you are least sure of: run the cross-validation with that step fitted on all rows and again fitted inside the folds, on the same fold assignment, and subtract. If the difference is within the error bar from episode 1, you have a style problem. If it is not, the number you reported was partly the leak.""")

    post.add(
        "Next",
        """Episode 4 asks a planning question every project faces before it has a test score at all: will more data help? Learning curves are the standard tool, and the standard extrapolation from them is a power law. Fitted on small samples and checked against what the larger samples actually deliver, it is tested rather than assumed.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
