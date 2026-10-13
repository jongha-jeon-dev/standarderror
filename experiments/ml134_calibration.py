"""ML 34: Recalibration Fixes the Average, Given Enough Rows.

The thirty-fourth episode of *Machine Learning, Taught Through What Breaks*.
"Platt scaling or isotonic regression fixes a model's probabilities."
Measured on episode 4's simulated task, with calibration sets from 50 to
5,000 rows, twenty times each, and on a simulation with two groups:

* Boosted trees were overconfident (calibration error 0.157, log loss
  0.748). Platt scaling on 50 rows brought log loss to 0.612; isotonic
  regression on 50 rows made it worse, 0.761, and needed about 100 rows to
  help at all.
* A naive Bayes model was already close to calibrated (error 0.023).
  Recalibrating it on 500 rows or fewer made it worse by either method;
  isotonic regression made it worse even on 5,000.
* A model calibrated to within 0.005 overall was off by 0.19 in a group
  holding 30% of the rows. Recalibrating its scores cannot fix that; giving
  it the group can.

Run: `standarderror run ml134_calibration --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import around
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"c": around.calibration_size(), "g": around.subgroup_calibration()}


def _first_helpful(row, key):
    for r in row[key]:
        if r["nll"] < row["raw"]["nll"]:
            return r["n"]
    return None


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    c = res["c"]

    def curves(ax, m):
        for i, (name, row) in enumerate(c.items()):
            ns = [r["n"] for r in row["platt"]]
            ax.plot(ns, [r["nll"] / row["raw"]["nll"] for r in row["platt"]],
                    marker="o", ms=4, color=m.series[2 * i],
                    label=f"{name}, Platt")
            ax.plot(ns, [r["nll"] / row["raw"]["nll"] for r in row["isotonic"]],
                    marker="o", ms=4, ls="--", color=m.series[2 * i + 1],
                    label=f"{name}, isotonic")
        ax.axhline(1.0, color=m.ink, lw=1)
        ax.text(300, 0.975, "uncalibrated = 1", fontsize=8.5, color=m.ink)
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    gb, nb = c["boosted trees"], c["naive Bayes"]
    out["f0"] = charts.diagram(
        curves,
        title="Recalibration needs rows",
        subtitle=("Test log loss after recalibrating on n fresh rows, as a "
                  "multiple of the uncalibrated model's. Average of 20 "
                  "calibration sets per size; 20,000 test rows."),
        xlabel="calibration rows, n (log scale)",
        ylabel="log loss relative to uncalibrated",
        figsize=(7.4, 4.6),
        source="Simulated (episode 4's task); standarderror/ml/around.py.",
        alt=("Four falling curves. For boosted trees both methods end well "
             "below 1, Platt below 1 from the start and isotonic only from "
             "about 100 rows. For naive Bayes both start above 1 and approach "
             "it from above."),
        caption=(f"Boosted trees: Platt helps from 50 rows, isotonic from "
                 f"{_first_helpful(gb, 'isotonic')}. Naive Bayes, already "
                 f"calibrated: Platt helps only from "
                 f"{_first_helpful(nb, 'platt')} rows, isotonic not at all "
                 f"up to 5,000."),
        path=str(IMG / f"ml134-f0-size.{EXT}"))[0]

    g = res["g"]

    def groups(ax, m):
        labels = ["all rows", "group 0 (70%)", "group 1 (30%)"]
        x = np.arange(3)
        w = 0.38
        blind = [g["blind"]["all"], g["blind"][0], g["blind"][1]]
        aware = [g["aware"]["all"], g["aware"][0], g["aware"][1]]
        ax.bar(x - w / 2, blind, w, color=m.series[1],
               label="model without the group")
        ax.bar(x + w / 2, aware, w, color=m.series[2],
               label="model with the group")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    out["f1"] = charts.diagram(
        groups,
        title="Calibrated on average, wrong for a third",
        subtitle=("Expected calibration error (ten bins) overall and within "
                  "each group, for logistic models trained on 5,000 rows and "
                  "tested on 100,000."),
        ylabel="expected calibration error",
        figsize=(7.2, 4.5),
        source="Simulated; standarderror/ml/around.py.",
        alt=("Paired bars: overall both models are near zero; within the two "
             "groups the model without the group has large errors, the larger "
             "in the smaller group; the model with the group stays near zero."),
        caption=(f"Overall calibration error {g['blind']['all']:.3f}; in the "
                 f"smaller group, **{g['blind'][1]:.3f}**. The group's true "
                 f"rate is {g['groups'][1]['rate']:.2f}, the model's mean "
                 f"prediction {g['groups'][1]['blind_mean']:.2f}."),
        path=str(IMG / f"ml134-f1-groups.{EXT}"))[0]

    rows = []
    for name, row in c.items():
        rows.append([name, "none", "—", f"{row['raw']['nll']:.3f}",
                     f"{row['raw']['ece']:.3f}", f"{row['raw']['auc']:.4f}"])
        for key, lab in (("platt", "Platt"), ("isotonic", "isotonic")):
            for r in row[key]:
                if r["n"] in (50, 500, 5000):
                    rows.append([name, lab, f"{r['n']:,}", f"{r['nll']:.3f}",
                                 f"{r['ece']:.3f}", f"{r['auc']:.4f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["model", "recalibration", "rows", "log loss",
                "calibration error", "AUC"],
        title="What recalibration changes, and what it cannot",
        subtitle=("Means over 20 calibration sets per size. Platt scaling is "
                  "monotone, so it never changes the ranking; isotonic "
                  "regression creates ties and can."),
        source="Simulated (episode 4's task); standarderror/ml/around.py.",
        alt=("A fourteen-row table: for boosted trees log loss falls with "
             "either method given enough rows; for naive Bayes it rises at 50 "
             "rows; AUC is unchanged by Platt and lowered by isotonic."),
        caption=("Platt scaling left AUC exactly as it was. Isotonic "
                 "regression lowered it, most on small calibration sets."),
        align="llrrrr",
        path=str(IMG / f"ml134-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    c, g = res["c"], res["g"]
    gb = c["boosted trees"]

    def iso(panel, m):
        panel.plot([r["nll"] for r in gb["isotonic"]], color=m.series[1], lw=2)
        panel.axhline(gb["raw"]["nll"], color=m.ink, lw=1)

    def nb(panel, m):
        row = c["naive Bayes"]
        panel.plot([r["nll"] for r in row["platt"]], color=m.series[0], lw=2)
        panel.axhline(row["raw"]["nll"], color=m.ink, lw=1)

    def grp(panel, m):
        panel.bar([0, 1, 2], [g["blind"]["all"], g["blind"][0], g["blind"][1]],
                  color=m.series[1])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=34,
        headline="Calibration is an average",
        panels=[(iso, f"{gb['isotonic'][0]['nll']:.3f}",
                 "isotonic, 50 rows"),
                (nb, "worse", "recalibrating NB, <500"),
                (grp, f"{g['blind'][1]:.2f}", "error in one group")],
        note=("Recalibration fits a correction from data, and a small "
              "calibration set fits it badly. What it fixes is the average "
              "over all rows, not the probabilities for any one group."),
        alt=("Three hand-drawn frames: a curve starting above a level line "
             "and falling below it; a curve approaching a level line from "
             "above; three bars, the first tiny."),
        path=str(IMG / f"ml134-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["size"] = s.run("""
        from standarderror.ml import around

        c = around.calibration_size()        # 20 calibration sets per size
        for name, row in c.items():
            print(f"{name}: uncalibrated log loss {row['raw']['nll']:.3f}, "
                  f"calibration error {row['raw']['ece']:.3f}")
            for p, i in zip(row["platt"], row["isotonic"]):
                print(f"   {p['n']:5d} rows   Platt {p['nll']:.3f}   "
                      f"isotonic {i['nll']:.3f}")
        """)
    out["groups"] = s.run("""
        g = around.subgroup_calibration()
        print(f"calibration error, all rows: {g['blind']['all']:.3f}")
        for grp in g["groups"]:
            k = grp["group"]
            print(f"group {k} ({grp['share']:.0%} of rows): true rate "
                  f"{grp['rate']:.3f}, mean prediction {grp['blind_mean']:.3f}, "
                  f"error {g['blind'][k]:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    c, g = res["c"], res["g"]
    gb, nb = c["boosted trees"], c["naive Bayes"]
    gbp = {r["n"]: r for r in gb["platt"]}
    gbi = {r["n"]: r for r in gb["isotonic"]}
    nbp = {r["n"]: r for r in nb["platt"]}
    nbi = {r["n"]: r for r in nb["isotonic"]}
    iso_help = _first_helpful(gb, "isotonic")
    nb_help = _first_helpful(nb, "platt")

    # The spine, asserted rather than trusted.
    assert gb["raw"]["ece"] > 0.12 and nb["raw"]["ece"] < 0.03
    assert gbp[50]["nll"] < gb["raw"]["nll"] - 0.1
    assert gbi[50]["nll"] > gb["raw"]["nll"] and iso_help == 100
    assert all(gbp[n]["nll"] < gbi[n]["nll"] for n in gbp)
    assert all(nbp[n]["nll"] > nb["raw"]["nll"] for n in (50, 100, 200))
    assert nb_help == 1000 and _first_helpful(nb, "isotonic") is None
    assert all(abs(r["auc"] - row["raw"]["auc"]) < 1e-9
               for row in c.values() for r in row["platt"])
    assert all(r["auc"] < row["raw"]["auc"] for row in c.values()
               for r in row["isotonic"])
    assert g["blind"]["all"] < 0.006 and g["blind"][1] > 0.15
    assert nbp[50]["ece"] > 3 * nb["raw"]["ece"]
    assert gbp[50]["ece"] < gb["raw"]["ece"] / 1.5
    assert g["blind"][0] > 0.05 and max(g["aware"][0], g["aware"][1]) < 0.01

    post = Post(
        title=(f"{SERIES_TAG} 34: Recalibration Fixes the Average, Given "
               f"Enough Rows"),
        slug="ml-34-calibration",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=34,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Platt scaling and isotonic regression learn a correction "
                  "from held-out rows. With few rows they can make a model "
                  "worse, and with many they fix the average, not each "
                  "group."),
        summary=(
            f"Boosted trees on a simulated task were overconfident "
            f"(calibration error {gb['raw']['ece']:.3f}, log loss "
            f"{gb['raw']['nll']:.3f}). Platt scaling on 50 calibration rows "
            f"cut log loss to {gbp[50]['nll']:.3f}; isotonic regression on 50 "
            f"rows raised it to {gbi[50]['nll']:.3f} and needed {iso_help} to "
            f"help. A naive Bayes model, already close to calibrated "
            f"(error {nb['raw']['ece']:.3f}), was made worse by Platt scaling "
            f"on fewer than {nb_help} rows and by isotonic regression on up to "
            f"5,000. Platt scaling never changed AUC; isotonic always lowered "
            f"it. A model calibrated to {g['blind']['all']:.3f} overall was "
            f"off by {g['blind'][1]:.2f} in a group holding 30% of the rows."),
        tags=["machine-learning", "data-science", "statistics",
              "calibration", "probability", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: episode 4's two-class Gaussian task in 10 dimensions; "
            "models trained on 2,000 rows, calibration sets of 50 to 5,000 "
            "fresh rows (seeds 100 to 119), 20,000 test rows.",
            "Simulated: two groups (70% and 30%) with base logits −0.6 and "
            "+1.0 and a shared slope on one feature; 5,000 training rows, "
            "100,000 test rows.",
            "Machinery: `standarderror/ml/around.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's GaussianNB, "
            "HistGradientBoostingClassifier (300 rounds), LogisticRegression "
            "for Platt scaling on the logit, IsotonicRegression; expected "
            "calibration error over ten equal-width bins.",
            "Where this stops: Platt, \"Probabilistic outputs for support "
            "vector machines\" (1999); Zadrozny and Elkan, \"Transforming "
            "classifier scores into accurate multiclass probability "
            "estimates\", *KDD* (2002); Niculescu-Mizil and Caruana, "
            "\"Predicting good probabilities with supervised learning\", "
            "*ICML* (2005); Hébert-Johnson et al., \"Multicalibration\", "
            "*ICML* (2018).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("training rows seed 1, calibration sets seeds "
                            "100 to 119, test rows seed 99; group simulation "
                            "seeds 0 and 1"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A second model for the first one's probabilities",
        f"""Episode 33 found that distance means less than the column count suggests, and sometimes more. Probabilities are the other number a model hands over to be trusted. Episodes 14 and 25 met models whose probabilities were too confident: boosting that overfitted its log loss, dropout's weight scaling that dropped its doubts. The standard repair is recalibration, and the sentence under test is how it is usually presented: *Platt scaling or isotonic regression fixes a model's probabilities*.

Both methods fit a second, small model on held-out rows that maps the first model's score to a probability. Platt scaling fits a logistic curve to the score's logit: two parameters. Isotonic regression fits any non-decreasing step function: as many parameters as the data allow. Both need rows the first model did not train on, and the question is how many.

On episode 4's simulated task, two models trained on 2,000 rows are recalibrated on fresh sets of 50 to 5,000 rows, twenty times at each size, and scored on 20,000 test rows:

{snip['size'].markdown()}

The boosted trees start badly calibrated, with an expected calibration error of {gb['raw']['ece']:.3f}. Platt scaling on just 50 rows takes their log loss from {gb['raw']['nll']:.3f} to **{gbp[50]['nll']:.3f}**. Isotonic regression on the same 50 rows makes it worse, **{gbi[50]['nll']:.3f}**, and only starts to help at {iso_help} rows. At every size Platt scaling does better than isotonic.""",
        figures=[figs["f0"]])

    post.add(
        "When there is nothing to fix",
        f"""The naive Bayes model starts nearly calibrated, with an error of {nb['raw']['ece']:.3f}. Recalibrating it can only help a little, and estimating the correction costs something. On 50 rows Platt scaling raises its log loss from {nb['raw']['nll']:.3f} to {nbp[50]['nll']:.3f}; it needs {nb_help} rows before it improves on doing nothing, and then by {nb['raw']['nll'] - nbp[nb_help]['nll']:.4f}. Isotonic regression is worse than doing nothing at every size up to 5,000 rows, where it still costs {nbi[5000]['nll'] - nb['raw']['nll']:.4f}.

The reason is the usual one for any fitted model: variance. A correction estimated from 50 rows is noisy, and applying a noisy correction to probabilities that were already right moves them away from right. Isotonic regression, with its many free steps, is noisier than Platt's two parameters, which is why Niculescu-Mizil and Caruana recommended Platt scaling for small calibration sets and isotonic only when there are thousands of rows. The measurement agrees and adds a point that is easy to miss: recalibration is not free even when it is unnecessary, so check the calibration first.

Recalibration also touches ranking. Platt scaling is a monotone map, and the AUC after it is identical to the uncalibrated AUC at every size. Isotonic regression maps ranges of scores to the same value, creating ties, and lowered AUC in every one of the {len(nb['isotonic'])} sizes for both models: to {gbi[50]['auc']:.4f} from {gb['raw']['auc']:.4f} for the boosted trees on 50 rows.""",
        figures=[figs["f2"]])

    post.add(
        "The number people check",
        f"""Log loss is the honest score for probabilities, but the number usually reported is the calibration error: the gap between predicted and observed rates, averaged over bins of the prediction. It tells the same story with larger swings.

For the boosted trees, calibration error falls from {gb['raw']['ece']:.3f} uncalibrated to {gbp[50]['ece']:.3f} after Platt scaling on 50 rows and {gbp[5000]['ece']:.3f} on 5,000; isotonic regression gets to {gbi[50]['ece']:.3f} and {gbi[5000]['ece']:.3f}. For the naive Bayes model the direction reverses at the small end: from {nb['raw']['ece']:.3f} uncalibrated to **{nbp[50]['ece']:.3f}** after Platt scaling on 50 rows, {nbp[50]['ece'] / nb['raw']['ece']:.1f} times worse, and {nbi[50]['ece']:.3f} after isotonic regression.

There is a structural reason the error cannot reach zero on a finite calibration set. The correction is estimated from the calibration rows' noisy labels, so with n rows the fitted map is off by roughly one over the square root of n in each bin it has to fill. Two parameters spread that noise thinly; a step function with a step wherever the data allow concentrates it. The calibration set has to be large compared with the precision the probabilities are expected to have, and a few hundred rows buy precision of a few hundredths at best.""")

    post.add(
        "Calibrated on average",
        f"""The deeper limit has nothing to do with sample size. A calibration curve compares predicted and observed rates across all rows; calibration in that sense is a property of the average. Two groups with different base rates, a model that sees a shared feature but not the group:

{snip['groups'].markdown()}

Over all rows the model is calibrated to **{g['blind']['all']:.3f}**: when it says 0.6, about 60% of rows are positive. Within the smaller group, 30% of the rows, its error is **{g['blind'][1]:.3f}**: it predicts {g['groups'][1]['blind_mean']:.2f} on average for a group whose rate is {g['groups'][1]['rate']:.2f}, and in the larger group it predicts {g['groups'][0]['blind_mean']:.2f} for a rate of {g['groups'][0]['rate']:.2f}. The errors cancel in the overall average.

No recalibration of the model's score can fix this, because the score is the same function of the feature in both groups and any monotone correction moves both groups' predictions the same way. What fixes it is information: the same logistic model given the group is calibrated to {g['aware'][0]:.3f} and {g['aware'][1]:.3f} within the groups. Hébert-Johnson and colleagues call calibration that holds within every identifiable subgroup multicalibration, and it is a property of the model and its inputs, not of a post-hoc correction.""",
        figures=[figs["f1"]])

    post.add(
        "Where this breaks",
        """**One task, two models.** The row counts at which recalibration starts to help depend on how badly calibrated the model is and how noisy the task is. The order (Platt before isotonic on small sets, check before correcting) is what transfers.

**Calibration sets drawn fresh.** In practice the calibration rows are often a slice of the training data, or the validation set already used for tuning, which adds its own optimism.

**Expected calibration error is binned.** Ten equal-width bins; with other binning the numbers move a little. Log loss, which needs no bins, tells the same story here.

**Two groups, known.** Real subgroups are many, overlapping and often unrecorded. Calibration within the groups one can name is a lower bound on the problem.""")

    post.add(
        "What to keep",
        f"""1. Recalibration is a fitted model, with variance. On 50 rows Platt scaling cut an overconfident model's log loss from {gb['raw']['nll']:.3f} to {gbp[50]['nll']:.3f}; isotonic regression raised it to {gbi[50]['nll']:.3f}.

2. Recalibrating a model that is already calibrated makes it worse until the calibration set is large: {nb_help} rows for Platt scaling here, more than 5,000 for isotonic.

3. Platt scaling leaves the ranking untouched; isotonic regression lowered AUC at every size.

4. Calibration is an average. A model calibrated to {g['blind']['all']:.3f} overall was off by {g['blind'][1]:.2f} in a third of the rows. Check calibration within the groups that matter, and fix it with information, not with a correction of the score.""")

    post.add(
        "Exercise",
        """Take a classifier whose probabilities you use, and a held-out set. Plot its calibration curve overall, then separately for each value of a variable that matters to the decision (region, product, age band) that the model may or may not use.

If the overall curve is close to the diagonal and a group's is not, recalibrating the score will not help that group. Then try Platt scaling on a quarter of your held-out rows and check, on the rest, whether log loss improved at all.""")

    post.add(
        "Next",
        """Every model in these thirty-four episodes was chosen somehow: a learning rate, a depth, a number of components. Episode 35, the last of Arc VII, measures what choosing does to the score that justified the choice.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
