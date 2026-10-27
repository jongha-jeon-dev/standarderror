"""ML 63: Calibrated in Every Group, Unequal in Every Error Rate.

The sixty-third episode of *Machine Learning, Taught Through What Breaks*,
in Arc XIII. "A model that is calibrated within every group treats the
groups the same." Measured on a simulation in which two groups differ in
their base rates and the outcome depends on a latent risk alone:

* A model calibrated to within 0.004 in both groups flagged 10.6% of group
  A's negatives and 26.9% of group B's, and missed 60.5% of A's positives
  against 35.3% of B's, at one shared threshold.
* Leaving the group out did not make the groups' error rates equal: the
  false positive rates moved to 13.7% and 22.9%, the model's score still
  told the groups apart (AUC 0.63), and it lost its calibration within
  groups (+0.028 and -0.020). A proxy for the group put most of it back.
* Moving group A's threshold to equalise false positive rates equalised the
  miss rates too here, cost 1.8 points of accuracy, and left a flag in
  group A right 52% of the time against 67% in group B.

Run: `standarderror run ml163_fairness --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import deploy
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 27)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

MODELS = ["with group", "with a proxy", "without either",
          "equal false positive rates"]


def compute() -> dict:
    return {"fair": deploy.fairness_study()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    f = res["fair"]

    def rates(ax, m):
        metrics = [("fpr", "false positive rate"), ("fnr", "miss rate"),
                   ("ppv", "positive predictive value"),
                   ("calibration_gap", "calibration gap")]
        x = np.arange(len(metrics))
        for k, (grp, off) in enumerate((("A", -0.18), ("B", 0.18))):
            v = [f["with group"][grp][key] for key, _ in metrics]
            ax.bar(x + off, v, 0.34, color=m.series[k], label=f"group {grp}")
        ax.axhline(0, color=m.ink, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in metrics], fontsize=8.5)
        ax.set_ylim(-0.05, 0.75)
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")

    g = f["with group"]
    out["f0"] = charts.diagram(
        rates,
        title="Calibrated in both groups, different in every error rate",
        subtitle=("One logistic model with the group as an input, one "
                  "threshold of 0.5. Group A's base rate is 0.31, group B's "
                  "0.46; 40,000 test rows."),
        xlabel="", ylabel="rate",
        figsize=(7.4, 4.3),
        source="Simulated; standarderror/ml/deploy.py.",
        alt=("Pairs of bars for four quantities. The calibration gaps are "
             "both near zero; the false positive rates, miss rates and "
             "positive predictive values differ between the groups."),
        caption=(f"False positive rates {g['A']['fpr']:.3f} and "
                 f"{g['B']['fpr']:.3f}; miss rates {g['A']['fnr']:.3f} and "
                 f"**{g['B']['fnr']:.3f}**."),
        path=str(IMG / f"ml163-f0-rates.{EXT}"))[0]

    body = []
    for name in MODELS:
        v = f[name]
        body.append([name, f"{v['accuracy']:.3f}", f"{v['group_auc']:.3f}",
                     f"{v['A']['fpr']:.3f} / {v['B']['fpr']:.3f}",
                     f"{v['A']['fnr']:.3f} / {v['B']['fnr']:.3f}",
                     f"{v['A']['ppv']:.3f} / {v['B']['ppv']:.3f}",
                     f"{v['A']['calibration_gap']:+.3f} / "
                     f"{v['B']['calibration_gap']:+.3f}"])
    out["f1"] = charts.table_image(
        body,
        header=["model", "accuracy", "score's AUC for group",
                "false positive A / B", "miss A / B", "precision A / B",
                "calibration gap A / B"],
        title="Four ways to treat two groups",
        subtitle=("Logistic models with the group, with a noisy proxy for it, "
                  "and with neither; and the first with group A's threshold "
                  "moved to equalise false positive rates. Calibration gap: "
                  "mean prediction minus outcome rate in the group."),
        source="Simulated; standarderror/ml/deploy.py.",
        alt=("A four-row table. Removing the group narrows but does not "
             "close the error-rate gaps and breaks calibration within groups; "
             "equalising false positive rates opens a precision gap."),
        caption="'Score's AUC for group': how well the model's score alone tells the groups apart.",
        align="lrrrrrr",
        path=str(IMG / f"ml163-f1-table.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    f = res["fair"]
    g, w, e = f["with group"], f["without either"], f["equal false positive rates"]

    def p1(panel, m):
        panel.bar([0, 1], [g["A"]["fpr"], g["B"]["fpr"]],
                  color=[m.series[0], m.series[1]])
        panel.set_ylim(0, 0.35)

    def p2(panel, m):
        panel.bar([0, 1], [abs(g["A"]["calibration_gap"]),
                           abs(w["A"]["calibration_gap"])],
                  color=[m.series[2], m.series[3]])

    def p3(panel, m):
        panel.bar([0, 1], [e["A"]["ppv"], e["B"]["ppv"]],
                  color=[m.series[0], m.series[1]])
        panel.set_ylim(0, 0.8)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=63,
        headline="Which equality",
        panels=[(p1, f"{g['B']['fpr'] / g['A']['fpr']:.1f}x",
                 "false positives, B vs A"),
                (p2, f"{w['A']['calibration_gap']:+.3f}", "calibration, no group"),
                (p3, f"{e['A']['ppv']:.2f}", "precision A, rates equal")],
        note=("When groups differ in base rates, equal precision and equal "
              "error rates cannot both hold. Choosing between them is a "
              "decision about whom errors fall on, not a technical fix."),
        alt=("Three hand-drawn frames: a short bar and a tall one; a tiny bar "
             "and a taller one; a shorter bar and a taller one."),
        path=str(IMG / f"ml163-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["world"] = s.run("""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from standarderror.ml import deploy

        rng = np.random.default_rng(0)
        g, x1, proxy, y = deploy.fairness_world(40_000, rng)
        G, X1, P, Y = deploy.fairness_world(40_000, rng)
        m = LogisticRegression().fit(np.c_[x1, g], y)      # group as an input
        p = m.predict_proba(np.c_[X1, G])[:, 1]
        for k, name in ((0, "A"), (1, "B")):
            s = G == k
            print(f"group {name}: outcome rate {Y[s].mean():.3f}, mean "
                  f"prediction {p[s].mean():.3f}, false positive rate "
                  f"{np.mean(p[s & (Y == 0)] > 0.5):.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    f = res["fair"]
    g, px, w, e = (f["with group"], f["with a proxy"], f["without either"],
                   f["equal false positive rates"])
    A, B = g["A"], g["B"]

    # The spine, asserted rather than trusted.
    assert abs(A["calibration_gap"]) < 0.006 and abs(B["calibration_gap"]) < 0.006
    assert B["fpr"] > 2 * A["fpr"] and A["fnr"] > B["fnr"] + 0.2
    assert B["base_rate"] - A["base_rate"] > 0.12
    assert w["B"]["fpr"] - w["A"]["fpr"] > 0.07
    assert w["A"]["calibration_gap"] > 0.02 and w["B"]["calibration_gap"] < -0.015
    assert 0.6 < w["group_auc"] < px["group_auc"] < g["group_auc"]
    assert abs(px["accuracy"] - g["accuracy"]) < 0.005
    assert abs(e["A"]["fpr"] - e["B"]["fpr"]) < 0.002
    assert abs(e["A"]["fnr"] - e["B"]["fnr"]) < 0.01
    assert e["B"]["ppv"] - e["A"]["ppv"] > 0.12
    assert 0.01 < g["accuracy"] - e["accuracy"] < 0.03
    assert B["flagged"] - A["flagged"] > 0.2 and w["B"]["flagged"] > w["A"]["flagged"]
    assert e["B"]["flagged"] > e["A"]["flagged"]

    post = Post(
        title=(f"{SERIES_TAG} 63: Calibrated in Every Group, Unequal in Every "
               f"Error Rate"),
        slug="ml-63-fairness",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=63,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A model whose probabilities were right in both groups made "
                  "false alarms 2.5 times as often in one of them. Leaving "
                  "the group out did not fix it, and equalising the error "
                  "rates made a flag mean different things in each group."),
        summary=(
            f"Two simulated groups with outcome rates of {A['base_rate']:.2f} "
            f"and {B['base_rate']:.2f}, an outcome that depends on a latent "
            f"risk alone. A logistic model with the group as an input was "
            f"calibrated to within {max(abs(A['calibration_gap']), abs(B['calibration_gap'])):.3f} "
            f"in both, and at one threshold its false positive rates were "
            f"{A['fpr']:.3f} and {B['fpr']:.3f} and its miss rates "
            f"{A['fnr']:.3f} and {B['fnr']:.3f}. Without the group, false "
            f"positive rates were {w['A']['fpr']:.3f} and {w['B']['fpr']:.3f}, "
            f"calibration within groups was off by {w['A']['calibration_gap']:+.3f} "
            f"and {w['B']['calibration_gap']:+.3f}, and the score still told "
            f"the groups apart with an AUC of {w['group_auc']:.2f}. Equalising "
            f"false positive rates with a separate threshold cost "
            f"{g['accuracy'] - e['accuracy']:.3f} of accuracy and left a "
            f"flagged person in group A truly positive {e['A']['ppv']:.0%} of "
            f"the time against {e['B']['ppv']:.0%} in group B."),
        tags=["machine-learning", "fairness", "calibration",
              "algorithmic-fairness", "decision-making", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated (seed 0): two equal groups; latent risk z standard "
            "Gaussian, 0.6 lower on average in group A; outcome logistic in "
            "1.6z - 0.2; measured feature z + noise (sd 0.7); proxy = group + "
            "noise (sd 0.5). 40,000 training and 40,000 test rows.",
            "Machinery: scikit-learn's LogisticRegression; "
            "`standarderror/ml/deploy.py`, tested in `tests/test_ml.py`.",
            "Where this stops: Kleinberg, Mullainathan and Raghavan, "
            "\"Inherent trade-offs in the fair determination of risk "
            "scores\", *ITCS* (2017); Chouldechova, \"Fair prediction with "
            "disparate impact\", *Big Data* (2017); Hardt, Price and Srebro, "
            "\"Equality of opportunity in supervised learning\", *NeurIPS* "
            "(2016); Dwork, Hardt, Pitassi, Reingold and Zemel, \"Fairness "
            "through awareness\", *ITCS* (2012); Kusner, Loftus, Russell and "
            "Silva, \"Counterfactual fairness\", *NeurIPS* (2017); Corbett-Davies and Goel, "
            "\"The measure and mismeasure of fairness\" (2018); Angwin, Larson, "
            "Mattu and Kirchner, \"Machine bias\", *ProPublica* (2016); "
            "episodes 34 "
            "and 46 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; a few seconds of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "What treating groups the same could mean",
        f"""A deployed model makes decisions about people, and people come in groups that the law and the public care about. The first fairness check most teams run is calibration within each group: does a predicted risk of 30% mean 30% for everyone? Episode 34 showed a model calibrated overall can fail it in a group. A model that passes it seems to have earned the sentence under test: *a model that is calibrated within every group treats the groups the same*.

The simulation is built so that nobody could call the model biased in its construction. Two groups of equal size differ in one thing: group A's latent risk is lower on average, so its outcome rate is {A['base_rate']:.2f} against group B's {B['base_rate']:.2f}. The outcome depends on that latent risk alone, never on the group. The model sees a noisy measurement of the risk and, in its first version, the group.

{snip['world'].markdown()}

The model's probabilities were right in both groups, to within {max(abs(A['calibration_gap']), abs(B['calibration_gap'])):.3f}. Now apply one threshold, 0.5, to everybody.""")

    post.add(
        "Calibrated, and unequal in every error rate",
        f"""Among people whose outcome was negative, the model flagged {A['fpr']:.1%} in group A and **{B['fpr']:.1%}** in group B: a member of B with nothing wrong was {B['fpr'] / A['fpr']:.1f} times as likely to be flagged. Among people whose outcome was positive, it missed {A['fnr']:.1%} in A and {B['fnr']:.1%} in B. A flag was right {A['ppv']:.1%} of the time in A and {B['ppv']:.1%} in B.

Nothing is wrong with the model. Group B's people have higher risk on average, so more of B's negatives sit near the threshold, where a noisy measurement pushes some across it. Two theorems from 2016 say this is unavoidable when base rates differ and prediction is imperfect. Kleinberg, Mullainathan and Raghavan proved that a score cannot be calibrated within groups and also give each group's negatives, and each group's positives, the same average score. Chouldechova proved the counterpart for decisions: equal precision, equal false positive rates and equal miss rates cannot all hold. The dispute over the COMPAS recidivism scores in 2016 was this theorem in public, with one side measuring calibration and the other error rates, both correctly.""",
        figures=[figs["f0"]])

    post.add(
        "Leaving the group out",
        f"""The most common response is to remove the group from the model's inputs. Without it, the false positive rates were {w['A']['fpr']:.1%} and {w['B']['fpr']:.1%}, and the miss rates {w['A']['fnr']:.1%} and {w['B']['fnr']:.1%}: closer, not equal. The model's score still separated the groups, with an AUC of {w['group_auc']:.2f} for telling them apart, because the measured risk differs between groups and the score is built from it.

And the model lost the property it started with. Without the group, it predicted {w['A']['calibration_gap']:+.3f} too high on average in group A and {w['B']['calibration_gap']:+.3f} in group B: the same measured value means a different expected risk in each group, because the groups' risks are distributed differently, and a model that cannot see the group averages over them. Leaving the group out is what Kusner and colleagues called fairness through unawareness; Dwork and colleagues had argued in 2012 that it fails for a different reason, that other features encode the group, which is what a proxy does. With a noisy proxy for the group in place of the group, such as a postcode, the score's AUC for the group came back to {px['group_auc']:.2f} and the calibration gaps shrank to {px['A']['calibration_gap']:+.3f} and {px['B']['calibration_gap']:+.3f}: the model rebuilt much of what was removed, at an accuracy of {px['accuracy']:.3f} against {g['accuracy']:.3f}.""")

    post.add(
        "Equalising the error rates",
        f"""The other response is to equalise the error rates directly, as Hardt, Price and Srebro proposed: keep the score and choose each group's threshold separately. Lowering group A's threshold from 0.5 to {e['threshold_A']:.3f} brought its false positive rate to {e['A']['fpr']:.3f}, matching B's {e['B']['fpr']:.3f} (the threshold was chosen on these same rows, so the match is by construction); in this world the miss rates came out equal as well, {e['A']['fnr']:.3f} and {e['B']['fnr']:.3f}.

The price was in two places. Accuracy fell from {g['accuracy']:.3f} to {e['accuracy']:.3f}, because A's threshold no longer sits where its errors are fewest. And the meaning of a flag now depends on the group: a flagged member of A was truly positive {e['A']['ppv']:.1%} of the time, a flagged member of B {e['B']['ppv']:.1%}. The two groups get equal error rates and unequal precision, where the first model gave them equal calibration and unequal error rates. A third definition, equal shares flagged in each group, was further away than either: the first model flagged {A['flagged']:.1%} of group A and {B['flagged']:.1%} of group B, the model without the group {w['A']['flagged']:.1%} and {w['B']['flagged']:.1%}, and the equal-error-rate thresholds {e['A']['flagged']:.1%} and {e['B']['flagged']:.1%}. Equal flag rates with unequal base rates would mean flagging fewer of group B's positives or more of group A's negatives than the score supports, a third distribution of errors with its own advocates. Each of the three rules measured here satisfied at most one of the three definitions, counting equal false positive and miss rates together as one; Chouldechova's theorem says that, with unequal base rates, no rule can have equal precision and equal error rates together, and the data say only what each choice costs.

Episode 46 put the threshold where costs said it should be; here the costs of errors falling on different groups are part of the decision, and they are not a quantity the data can supply.""",
        figures=[figs["f1"]])

    post.add(
        "Where this breaks",
        """**A clean world.** Here the outcome depends on risk alone and the labels are accurate. In practice the labels often come from past decisions (episode 39), the measured risk can be measured worse in one group, and the group can affect the outcome through channels that are themselves unjust; each changes which definition is defensible.

**Two groups, one threshold rule.** Real deployments have many groups, intersections of them, and decisions that are not one threshold.

**Simulation only.** scikit-learn's bundled data have no protected attribute to test on; the theorem does not need data, but the sizes of the gaps do.

**Fairness metrics chosen for the demonstration.** Calibration, error rates and precision are three of many definitions; Corbett-Davies and Goel review others, and the trade-offs among them follow the same arithmetic.""")

    post.add(
        "What to keep",
        f"""1. A model calibrated to within {max(abs(A['calibration_gap']), abs(B['calibration_gap'])):.3f} in both groups made false alarms on {A['fpr']:.1%} and {B['fpr']:.1%} of their negatives at one threshold. With different base rates, that is arithmetic, not bias in the model.

2. Removing the group narrowed the gaps without closing them, broke calibration within groups, and a proxy restored much of what was removed.

3. Equalising false positive rates cost {g['accuracy'] - e['accuracy']:.3f} of accuracy and made a flag {e['A']['ppv']:.0%} reliable in one group and {e['B']['ppv']:.0%} in the other.

4. Report calibration and error rates by group together. Decide which inequality to accept with the people affected, and write the choice down, because no model can make it disappear.""")

    post.add(
        "Exercise",
        """For a model that makes decisions about people, compute by group: the outcome rate, the mean predicted probability, the false positive rate, the miss rate and the precision at the threshold in use.

If the outcome rates differ, at least one of the other comparisons will show a gap. Name which one your organisation has decided to equalise, and who decided; if the answer is nobody, the default threshold decided for you.""")

    post.add(
        "Next",
        """A fairness audit is a snapshot, and the world the model serves keeps moving. Episode 64 asks how a model should be retrained as it does: on all the history it has, or only the recent part, and what it costs when the labels arrive late.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
