"""ML 16: Permutation Importance Measures the Model, Not What the Data Need.

The sixteenth episode of *Machine Learning, Taught Through What Breaks*,
opening Arc IV. "Permutation importance measures how much a model needs a
feature." Measured on the diabetes data with a random forest:

* Shuffling bmi costs 0.168 of test R^2. Refitting without bmi costs 0.043:
  the forest relies on bmi, but a forest refitted without it loses about a
  quarter as much (on average over three splits that differ widely).
* Add an exact copy of bmi and its permutation importance falls to 0.017,
  the copy's is 0.021, and their sum is a quarter of what bmi had alone.
  Dropping bmi and keeping the copy costs nothing.
* A permutation invents rows. Shuffle s1, which is correlated 0.90 with s2,
  and 29% of the shuffled rows fall outside the region that holds 99% of the
  real (s1, s2) pairs.

Run: `standarderror run ml116_permutation --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import reading as rd
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 12)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"copy": rd.copy_importance(), "off": rd.off_support(),
            "spread": rd.split_spread()}


def _label(rho) -> str:
    return "no copy" if rho is None else (
        "exact copy" if rho == 1.0 else f"copy, corr {rho:g}")


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    cp = res["copy"]

    def bars(ax, m):
        x = np.arange(len(cp))
        w = 0.2
        ax.bar(x - 1.5 * w, [r["feature"] for r in cp], w, color=m.series[0],
               label="permute bmi")
        ax.bar(x - 0.5 * w, [r["copy"] for r in cp], w, color=m.series[1],
               label="permute the copy")
        ax.bar(x + 0.5 * w, [r["drop_feature"] for r in cp], w,
               color=m.series[2], label="refit without bmi")
        ax.bar(x + 1.5 * w, [r["drop_both"] for r in cp], w,
               color=m.series[3], label="refit without bmi and copy")
        ax.set_xticks(x)
        ax.set_xticklabels([_label(r["rho"]) for r in cp])
        ax.set_ylim(0, 0.2)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    base, ex = cp[0], cp[-1]
    out["f0"] = charts.diagram(
        bars,
        title="Four answers to 'how much does the model need bmi?'",
        subtitle=("Random forest on the diabetes data, with and without a "
                  "copy of bmi added. Loss of test R^2 when a column is "
                  "shuffled, or when the model is refitted without it. Mean "
                  "of three splits."),
        xlabel="",
        ylabel="loss of test R^2",
        source="Diabetes data (scikit-learn); standarderror/ml/reading.py.",
        alt=("Grouped bars: without a copy, permuting bmi costs far more "
             "than refitting without it; with an exact copy, permuting "
             "either column costs little and refitting without both costs "
             "as much as refitting without bmi did alone."),
        caption=(f"Alone, shuffling bmi costs **{base['feature']:.3f}** and "
                 f"refitting without it {base['drop_feature']:.3f}. With an "
                 f"exact copy, shuffling either costs about 0.02; removing "
                 f"bmi and keeping the copy costs nothing, and removing both "
                 f"costs {ex['drop_both']:.3f}."),
        path=str(IMG / f"ml116-f0-copies.{EXT}"))[0]

    off = res["off"]

    def scatter(ax, m):
        P, Q = off["pairs"], off["permuted"]
        ax.scatter(Q[:, 0], Q[:, 1], s=6, color=m.series[1], alpha=0.5,
                   label="after shuffling s1")
        ax.scatter(P[:, 0], P[:, 1], s=6, color=m.series[0], alpha=0.6,
                   label="real patients")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    out["f1"] = charts.diagram(
        scatter,
        title="A permutation invents patients",
        subtitle=(f"s1 against s2 for the 442 patients (correlation "
                  f"{off['corr']:.2f}), and the same pairs after s1 is "
                  f"shuffled, as permutation importance does."),
        xlabel="s1 (total serum cholesterol, standardised)",
        ylabel="s2 (LDL, standardised)",
        source="Diabetes data (scikit-learn); standarderror/ml/reading.py.",
        alt=("A tight diagonal cloud of real points and a round cloud of "
             "shuffled points spreading far off the diagonal."),
        caption=(f"**{off['outside']:.0%}** of shuffled rows lie outside the "
                 f"ellipse that holds {off['level']:.0%} of real pairs. The "
                 f"model is scored on what it predicts for them."),
        path=str(IMG / f"ml116-f1-offsupport.{EXT}"))[0]

    rows = []
    for r in cp:
        rows.append([_label(r["rho"]), f"{r['r2']:.3f}", f"{r['feature']:.3f}",
                     f"{r['copy']:.3f}" if r["rho"] is not None else "—",
                     f"{r['drop_feature']:.3f}", f"{r['drop_both']:.3f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["setting", "test R^2", "permute bmi", "permute copy",
                "refit without bmi", "refit without both"],
        title="Reliance is not necessity",
        subtitle=("Diabetes, random forest (300 trees, leaves of 5), means "
                  "of three 70/30 splits; permutation importance from 20 "
                  "shuffles."),
        source="Diabetes data (scikit-learn); standarderror/ml/reading.py.",
        alt=("A four-row table: test R^2 unchanged by adding a copy; "
             "permutation importance of bmi falling from 0.17 to 0.02 as "
             "the copy improves; refitting without both near 0.04 "
             "throughout."),
        caption=("The last column barely moves: the information in bmi is "
                 "worth about 0.04 of R² in every setting. The permutation "
                 "columns move by a factor of ten."),
        align="lrrrrr",
        path=str(IMG / f"ml116-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    cp, off = res["copy"], res["off"]

    def gap(panel, m):
        panel.bar([0, 1], [cp[0]["feature"], cp[0]["drop_feature"]],
                  color=[m.series[0], m.series[2]])

    def copy(panel, m):
        panel.bar([0, 1, 2], [cp[0]["feature"], cp[-1]["feature"],
                              cp[-1]["copy"]],
                  color=[m.series[0], m.series[0], m.series[1]])

    def cloud(panel, m):
        P, Q = off["pairs"], off["permuted"]
        panel.scatter(Q[:, 0], Q[:, 1], s=3, color=m.series[1])
        panel.scatter(P[:, 0], P[:, 1], s=3, color=m.series[0])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=16,
        headline="Shuffling a column asks the wrong question",
        panels=[(gap, f"{cp[0]['feature'] / cp[0]['drop_feature']:.0f}x",
                 "permute vs refit, bmi"),
                (copy, f"{cp[-1]['feature']:.3f}", "bmi with an exact copy"),
                (cloud, f"{off['outside']:.0%}", "shuffled rows off data")],
        note=("Permutation importance measures how much one fitted model "
              "leans on a column, scored on rows the shuffle made up. It "
              "can be four times what the data need, and a copy of the "
              "feature can make both look unimportant."),
        alt=("Three hand-drawn frames: a tall bar beside a short one; one "
             "tall bar and two short ones; a narrow cloud inside a wide "
             "one."),
        path=str(IMG / f"ml116-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["copy"] = s.run("""
        from standarderror.ml import reading as rd

        # random forest on diabetes; loss of test R^2, mean of 3 splits
        for r in rd.copy_importance(rhos=(None, 0.8, 0.95, 1.0)):
            print(f"{str(r['rho']):>4}  permute bmi {r['feature']:.3f}"
                  f"  permute copy {r['copy']:+.3f}"
                  f"  refit w/o bmi {r['drop_feature']:.3f}"
                  f"  w/o both {r['drop_both']:.3f}")
        """)
    out["off"] = s.run("""
        o = rd.off_support(a="s1", b="s2", level=0.99)
        print(f"corr(s1, s2) = {o['corr']:.2f}")
        print(f"after shuffling s1, {o['outside']:.1%} of rows fall outside "
              f"the region holding 99% of real pairs")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    cp, off, sp = res["copy"], res["off"], res["spread"]
    base, c8, c95, ex = cp
    ratio = base["feature"] / base["drop_feature"]

    # The spine, asserted rather than trusted.
    assert base["feature"] > 3 * base["drop_feature"]
    assert ex["feature"] < 0.2 * base["feature"]
    assert ex["feature"] + ex["copy"] < 0.35 * base["feature"]
    assert all(abs(r["r2"] - base["r2"]) < 0.02 for r in cp)
    assert max(r["drop_feature"] for r in cp[1:]) < 0.01
    assert all(abs(r["drop_both"] - base["drop_feature"]) < 0.01 for r in cp)
    assert c8["feature"] > c95["feature"] > ex["feature"]
    # at 0.8 the copy gets nothing; at 0.95 a little, the original still dominant
    assert abs(c8["copy"]) < 0.005 and c8["copy"] < c95["copy"] < 0.2 * c95["feature"]
    assert 0.4 < base["feature"] / base["r2"] < 0.6     # shuffling bmi halves R^2
    assert abs(ex["feature"] - ex["copy"]) < 0.5 * (ex["feature"] + ex["copy"])
    assert off["outside"] > 20 * off["expected"] and off["corr"] > 0.85
    assert all(r["permute"] > r["drop"] for r in sp)
    assert max(r["drop"] for r in sp) - min(r["drop"] for r in sp) > 0.1

    post = Post(
        title=(f"{SERIES_TAG} 16: Permutation Importance Measures the Model, "
               f"Not What the Data Need"),
        slug="ml-16-permutation",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=16,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Shuffle a column and see how much worse the model does: "
                  "it is the importance episode 12 recommended. It answers "
                  "a narrower question than it seems to, on rows the "
                  "shuffle invented."),
        summary=(
            f"A random forest on the diabetes data. Shuffling bmi costs "
            f"{base['feature']:.3f} of test R²; refitting the forest without "
            f"bmi costs {base['drop_feature']:.3f}, {ratio:.0f} times less. "
            f"Add an exact copy of bmi and the permutation importances are "
            f"{ex['feature']:.3f} for bmi and {ex['copy']:.3f} for the copy "
            f"— together a quarter of what bmi had alone — while dropping "
            f"bmi and keeping the copy costs nothing and dropping both costs "
            f"{ex['drop_both']:.3f}, as before. Shuffling s1, correlated "
            f"{off['corr']:.2f} with s2, puts {off['outside']:.0%} of rows "
            f"outside the region that holds 99% of real (s1, s2) pairs."),
        tags=["machine-learning", "data-science", "statistics",
              "interpretability", "feature-importance", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Diabetes data of Efron et al. (2004), as bundled with "
            "scikit-learn: 442 patients, 10 baseline variables; three 70/30 "
            "splits; a copy of bmi added as rho·bmi + sqrt(1 - rho²)·noise "
            "with the noise scaled to bmi's spread.",
            "Machinery: `standarderror/ml/reading.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's RandomForestRegressor (300 "
            "trees, leaves of 5) and permutation_importance (20 shuffles).",
            "Where this stops: Breiman, \"Random forests\", *Machine "
            "Learning* (2001); Strobl et al., \"Conditional variable "
            "importance for random forests\", *BMC Bioinformatics* (2008); "
            "Hooker, Mentch and Zhou, \"Unrestricted permutation forces "
            "extrapolation\", *Statistics and Computing* (2021).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("splits from seeds 0 to 2; copies' noise from "
                            "seeds 100 to 102; forests seeded with the split "
                            "seed; permutations with seed 0"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Three questions with one name",
        f"""Arc III ended with forests that predict well and resist being read. Arc IV is about reading them anyway, starting with the importance measure episode 12 recommended over impurity importance: shuffle one column of a held-out set, score the model again, and call the drop in performance the feature's importance.

That is often described as "how much the model needs the feature". There are at least three different questions hiding in that phrase. How much does *this fitted model* lean on the column? How much would *a model refitted without it* lose? How much *information* does the column carry that nothing else does? Permutation importance answers only the first, and it answers it on rows the shuffle invented.

The diabetes data and a random forest show the gap. bmi matters to this forest: shuffling it, as the first row below shows, removes about half of its test R² of {base['r2']:.2f}. Add a second column that copies bmi with a chosen correlation, and measure four things:

{snip['copy'].markdown()}

Start with the first row, no copy. Shuffling bmi costs **{base['feature']:.3f}** of test R². Refitting the forest without bmi costs {base['drop_feature']:.3f}, {ratio:.0f} times less. Both are correct; they measure different things. The fitted forest has organised itself around bmi and is badly hurt when bmi is scrambled. A forest grown without bmi loses only a little — presumably because it finds much of the same information in the other measurements, though which ones carry it was not measured here.""",
        figures=[figs["f0"]])

    post.add(
        "A copy hides the original",
        f"""Now add the copy. With an exact duplicate, the two columns offer identical splits, so which one a node uses comes down to tie-breaking, and the forest's reliance is presumably divided between them. Shuffle one and the splits on the other still work. Permutation importance falls to {ex['feature']:.3f} for bmi and {ex['copy']:.3f} for its copy. Their sum, {ex['feature'] + ex['copy']:.3f}, is a quarter of what bmi had alone: permutation importance is not even additive over copies.

Refitting tells the same story differently. Remove bmi and keep its copy, and nothing is lost ({ex['drop_feature']:.3f}). Remove both and the loss is {ex['drop_both']:.3f}, the same as removing bmi when there was no copy. That last column is stable across every row of the table — averaged over the same three splits, the information bmi carries is worth about {base['drop_feature']:.2f} of R² however it is presented — while the permutation numbers move by a factor of ten.

Real tables are full of near-copies: total cholesterol and LDL, height and weight, a price and the same price in another currency, a date and the age computed from it. A pair close enough can split its permutation importance between its members and push both down a ranking. How close is close enough is in the table: at a correlation of 0.8 the forest still leaned almost entirely on the original, and the copy's importance was {c8['copy']:+.3f}; at 0.95 the copy took {c95['copy']:.3f} against bmi's {c95['feature']:.3f}; only the exact copy split it evenly. Episode 12's breast-cancer run had three real measurements with permutation importance at or below zero, which is what this effect would produce, though that run did not check the cause.""",
        figures=[figs["f2"]])

    post.add(
        "Rows that do not exist",
        f"""There is a second problem, and it does not need a copy. Shuffling a column breaks its relationship with every other column. When features are correlated, the shuffled rows are combinations no patient has.

{snip['off'].markdown()}

In the data, s1 and s2 — total cholesterol and LDL — move together closely. After s1 is shuffled, **{off['outside']:.0%} of rows lie outside the region holding 99% of real pairs**, where only 1% would by chance. A patient with very high total cholesterol and very low LDL is not a patient anyone measured. The forest's prediction there is whatever its nearest leaves say, and permutation importance scores the forest on those predictions.

So part of what permutation importance measures is how badly the model extrapolates off the data, which is a property of the model's behaviour in a region it was never trained on. Hooker, Mentch and Zhou (2021) make this the central objection; conditional permutation, which shuffles a feature only among rows with similar values of the others, keeps the rows on the data at the cost of more computation and more choices.""",
        figures=[figs["f1"]])

    post.add(
        "Asking a better question",
        f"""If permutation importance answers the narrowest of the three questions, the fix is to ask a wider one deliberately, and the table already contains the answer to it. Removing bmi and its copy together and refitting cost {ex['drop_both']:.3f} of R², within a hundredth of the {base['drop_feature']:.3f} that removing bmi cost when it had no copy. Measured as a group, the information is stable: duplicated, nearly duplicated or alone, it is worth the same.

That suggests a working procedure. Cluster the features by correlation first — a simple threshold on the absolute correlation is enough to start. Report importance for each cluster, by permuting all its members together or by refitting without the whole cluster. Within a cluster, say which member the model happened to lean on, and say that a different fit could lean on another. Permuting a whole cluster should also leave fewer impossible rows behind, since its members move together and keep their relationship to one another; what is broken is only their relationship to everything else. That was not measured here.

None of this makes importance a causal statement. A group of features that predicts well may be a proxy for something else entirely, as episode 19 will show with a single pixel. It does make the ranking a property of the information in the data rather than of which column the forest's random splits happened to favour.""")

    post.add(
        "Where this breaks",
        f"""**Refitting has its own question.** Drop-column importance measures what the *data* lose without a feature, given the model class. It is expensive — one refit per feature — and it assigns zero to every member of a group of near-copies, which is equally misleading in the other direction. The honest unit is often the group.

**R² on 133 test patients is noisy.** The numbers are averages of three splits, and single splits differ a lot: test R² ranged from {min(r['r2'] for r in sp):.2f} to {max(r['r2'] for r in sp):.2f}, and the loss from refitting without bmi from {min(r['drop'] for r in sp):+.2f} to {max(r['drop'] for r in sp):+.2f}. The direction held in every split — permuting bmi always cost more than refitting without it — but the size of the drop-column numbers, here and in the table, is uncertain to about a tenth of R².

**Other models distribute reliance differently.** A linear model with an exact copy has no unique coefficients at all, and neither does a lasso: any split of the coefficient between the two columns with the same sign has the same penalty, so the answer depends on the solver. With near-copies, episode 8 found the lasso's choice between the two varying from fit to fit. Permutation importance inherits whatever the fitting procedure did.

**A copy is the extreme case.** At correlation 0.8 the forest still prefers the original and the copy's permutation importance is near zero; at 0.95 the copy starts to take a share, and the original still dominates. Most real pairs sit somewhere in between, and so does the distortion.""")

    post.add(
        "What to keep",
        f"""1. Permutation importance measures how much one fitted model leans on a column, not how much the outcome depends on it or how much the data would lose without it.

2. On diabetes, shuffling bmi cost {base['feature']:.3f} of R²; refitting without it cost {base['drop_feature']:.3f}.

3. With an exact copy, the two columns' permutation importances summed to {ex['feature'] + ex['copy']:.3f}, a quarter of bmi's alone. Exact copies hide each other; at a correlation of 0.8 the effect had not yet appeared.

4. Shuffling a correlated feature invents rows: {off['outside']:.0%} of shuffled (s1, s2) pairs lay outside the 99% region of real ones.

5. Group correlated features before computing importance, and report what is lost when the group is removed, not each member's share.""")

    post.add(
        "Exercise",
        """Compute permutation importance for your model and, for its top five features, the importance when each is removed and the model refitted. Where the two disagree by more than a factor of two, list the features most correlated with the one in question.

Then pick your most strongly correlated pair, shuffle one of them in a copy of your test set, and plot the pair before and after. Count the shuffled rows that lie outside the real cloud. That is the share of the importance score computed on data no one has seen.""")

    post.add(
        "Next",
        """Episode 17 moves from how much a feature matters to how the prediction depends on it, and to the partial dependence plot — which builds its curve out of exactly the kind of invented rows measured here.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
