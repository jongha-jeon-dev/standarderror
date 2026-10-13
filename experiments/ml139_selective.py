"""ML 39: The Applicants You Refused Are the Data You Do Not Have.

The thirty-ninth episode of *Machine Learning, Taught Through What
Breaks*. "A model trained on the outcomes you observed predicts the
outcomes you did not." Measured on simulated loan applicants, half
accepted by an old policy, outcomes seen only for the accepted:

* When the old policy used only the features the model sees and the model
  was correctly specified, the accepted-only model predicted a 0.453
  default rate for the refused, who defaulted at 0.432.
* When the old policy also used what a loan officer saw and the model does
  not, the same model predicted 0.137 for a group that defaulted at 0.486,
  with its ranking intact (AUC 0.769 against 0.770).
* Validating on accepted applicants understated AUC in one world (0.673
  against 0.770) and overstated it in another (0.627 against 0.559).
* A model trained only on the 5% approved at random predicted 0.389,
  matching a model trained on everyone's outcomes (0.394).

Run: `standarderror run ml139_selective --publish`
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

WORLDS = ((False, False), (True, False), (False, True), (True, True))


def _wname(curved, officer):
    return (("curved risk" if curved else "straight risk") + ", "
            + ("officer sees more" if officer else "policy sees features only"))


def compute() -> dict:
    s = rows.selective_labels()
    return {"s": s, "k": {(r["curved"], r["officer"], r["model"]): r for r in s}}


# ------------------------------------------------------------------ figures

def _bars(ax, m, k):
    w = 0.2
    keys = (("refused_rate", "actual default rate", m.ink),
            ("refused_pred", "trained on the accepted", m.series[1]),
            ("explore_refused_pred", "trained on 5% approved at random",
             m.series[2]),
            ("oracle_refused_pred", "trained on everyone (unavailable)",
             m.series[3]))
    for j, (key, lab, col) in enumerate(keys):
        x = np.arange(4) + (j - 1.5) * w
        ax.bar(x, [k[(c, o, "logistic")][key] for c, o in WORLDS], width=w,
               color=col, label=lab)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["straight risk,\nfeatures only", "curved risk,\nfeatures only",
                        "straight risk,\nofficer sees more",
                        "curved risk,\nofficer sees more"], fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(0, 0.75)


def figures(res: dict) -> dict:
    out: dict = {}
    k = res["k"]
    so = k[(False, True, "logistic")]

    out["f0"] = charts.diagram(
        lambda ax, m: _bars(ax, m, k),
        title="What the model expects of the applicants it never saw repay",
        subtitle=("Mean predicted default rate among refused applicants, by "
                  "logistic models trained three ways, against the rate at "
                  "which they actually default; 10 cohorts of 20,000."),
        xlabel="",
        ylabel="default rate among the refused",
        figsize=(7.8, 4.8),
        source="Simulated; standarderror/ml/rows.py.",
        alt=("Grouped bars for four worlds. In the first, all bars are "
             "near 0.43. In the straight-risk world where the officer saw "
             "more, the accepted-only bar is near 0.14 while the actual "
             "rate is about 0.49 and the other two bars about 0.39."),
        caption=(f"When the officer saw more, the accepted-only model "
                 f"expected **{so['refused_pred']:.3f}** of the refused to "
                 f"default; {so['refused_rate']:.3f} did."),
        path=str(IMG / f"ml139-f0-refused.{EXT}"))[0]

    body = []
    for c, o in WORLDS:
        for model in ("logistic", "boosted trees"):
            r = k[(c, o, model)]
            body.append([_wname(c, o) if model == "logistic" else "", model,
                         f"{r['refused_rate']:.3f}", f"{r['refused_pred']:.3f}",
                         f"{r['oracle_refused_pred']:.3f}",
                         f"{r['auc_accepted']:.3f}", f"{r['auc_all']:.3f}",
                         f"{r['auc_oracle']:.3f}"])
    out["f1"] = charts.table_image(
        body,
        header=["world", "model", "refused: actual", "refused: predicted",
                "everyone-trained", "AUC on accepted", "AUC on all",
                "everyone-trained AUC"],
        title="Trained on the accepted, judged on everyone",
        subtitle=("Default rates among the refused half, and AUC measured "
                  "the usual way (accepted applicants only) and on the whole "
                  "cohort; means over 10 cohorts."),
        source="Simulated; standarderror/ml/rows.py.",
        alt=("An eight-row table, two models in each of four worlds, with "
             "predicted and actual default rates among the refused and three "
             "AUCs."),
        caption=("'Everyone-trained' is the same model fitted on every "
                 "applicant's outcome, which no lender has: the ceiling for "
                 "a model that sees these features."),
        align="llrrrrrr",
        path=str(IMG / f"ml139-f1-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    k = res["k"]
    so = k[(False, True, "logistic")]
    sf = k[(False, False, "logistic")]

    def p1(panel, m):
        panel.bar([0, 1], [sf["refused_rate"], sf["refused_pred"]],
                  color=[m.ink, m.series[1]])
        panel.set_ylim(0, 0.6)

    def p2(panel, m):
        panel.bar([0, 1], [so["refused_rate"], so["refused_pred"]],
                  color=[m.ink, m.series[1]])
        panel.set_ylim(0, 0.6)

    def p3(panel, m):
        panel.bar([0, 1, 2], [so["refused_pred"], so["explore_refused_pred"],
                              so["oracle_refused_pred"]],
                  color=[m.series[1], m.series[2], m.series[3]])
        panel.set_ylim(0, 0.6)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=39,
        headline="The refused half",
        panels=[(p1, f"{sf['refused_pred']:.3f}", "features-only policy"),
                (p2, f"{so['refused_pred']:.3f}", "officer saw more"),
                (p3, f"{so['explore_refused_pred']:.3f}", "5% approved at random")],
        note=("A model trained on approved applicants learns about the "
              "approved. When the approval used information the model "
              "lacks, it cannot know what it is missing."),
        alt=("Three hand-drawn frames of bars: two equal bars; a tall bar "
             "and a short one; three bars, the last two similar."),
        path=str(IMG / f"ml139-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["main"] = s.run("""
        from standarderror.ml import rows

        for r in rows.selective_labels():     # 10 cohorts of 20,000
            if r["model"] == "logistic":
                print(f"curved={r['curved']!s:5} officer={r['officer']!s:5} "
                      f"refused default {r['refused_rate']:.3f}, predicted "
                      f"{r['refused_pred']:.3f}; AUC {r['auc_all']:.3f}")
        """)
    out["explore"] = s.run("""
        for r in rows.selective_labels():
            if r["model"] == "logistic":
                print(f"curved={r['curved']!s:5} officer={r['officer']!s:5} "
                      f"5% random only: {r['explore_refused_pred']:.3f}; "
                      f"recalibrated: {r['recal_refused_pred']:.3f}; "
                      f"everyone: {r['oracle_refused_pred']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    k = res["k"]
    sf, cf = k[(False, False, "logistic")], k[(True, False, "logistic")]
    so, co = k[(False, True, "logistic")], k[(True, True, "logistic")]
    hsf = k[(False, False, "boosted trees")]

    # The spine, asserted rather than trusted.
    assert abs(sf["refused_pred"] - sf["refused_rate"]) < 0.025
    assert abs(sf["auc_all"] - sf["auc_oracle"]) < 0.003
    assert hsf["refused_pred"] < hsf["refused_rate"] - 0.08
    assert hsf["auc_all"] < hsf["auc_oracle"] - 0.05
    assert cf["refused_pred"] < cf["refused_rate"] / 2
    assert cf["auc_all"] < 0.57 and cf["auc_oracle"] > 0.74
    assert so["refused_pred"] < 0.15 and so["refused_rate"] > 0.48
    assert so["refused_pred"] < so["refused_rate"] / 3
    assert abs(so["auc_all"] - so["auc_oracle"]) < 0.003
    assert so["nll"] - so["nll_oracle"] > 0.13
    assert so["refused_rate"] - so["oracle_refused_pred"] > 0.08
    assert sf["auc_accepted"] < sf["auc_all"] - 0.09
    assert cf["auc_accepted"] > cf["auc_all"] + 0.06
    assert co["auc_all"] < 0.5
    for r in (sf, so, cf, co):
        assert abs(r["explore_refused_pred"] - r["oracle_refused_pred"]) < 0.01
        assert r["explore_nll"] - r["nll_oracle"] < 0.004
    assert abs(so["recal_refused_pred"] - so["oracle_refused_pred"]) < 0.01
    assert cf["recal_refused_pred"] < cf["oracle_refused_pred"] - 0.1
    assert 950 < sf["explore_rows"] < 1050

    post = Post(
        title=(f"{SERIES_TAG} 39: The Applicants You Refused Are the Data "
               f"You Do Not Have"),
        slug="ml-39-selective-labels",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=39,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Outcomes are recorded only for the cases someone approved. "
                  "Whether a model trained on them can judge the rest depends "
                  "on what the approver knew that the model does not."),
        summary=(
            f"Simulated applicants, half accepted by an old policy, "
            f"outcomes observed only for the accepted. When the policy used "
            f"only the model's features, a correctly specified model "
            f"predicted a default rate of {sf['refused_pred']:.3f} for the "
            f"refused, who defaulted at {sf['refused_rate']:.3f}; boosted "
            f"trees predicted {hsf['refused_pred']:.3f}. When a loan officer "
            f"also used information the model lacks, the model predicted "
            f"{so['refused_pred']:.3f} for a group that defaulted at "
            f"{so['refused_rate']:.3f}, with its AUC unchanged. A model "
            f"trained on the 5% of applicants approved at random predicted "
            f"{so['explore_refused_pred']:.3f}, as well as one trained on "
            f"everyone ({so['oracle_refused_pred']:.3f})."),
        tags=["machine-learning", "data-science", "statistics",
              "selection-bias", "credit-risk", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: applicants with two observed features and one "
            "unobserved, all standard normal; default log-odds −1.5 + 1.2 "
            "x_1 + 0.8 x_2 + 1.2 u, plus 0.8 x_1² in the curved worlds. The "
            "old policy accepts the half with the lowest score 1.2 x_1 + "
            "0.8 x_2 + noise, plus 1.2 u when the officer sees more. "
            "Cohorts of 20,000 (seeds 0 to 9), scored on fresh cohorts "
            "(seeds 100 to 109); 5% approved at random from the training "
            "cohort.",
            "Machinery: `standarderror/ml/rows.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's LogisticRegression and "
            "HistGradientBoostingClassifier; recalibration refits the "
            "intercept and slope of the accepted-only model's log-odds on "
            "the randomly approved rows.",
            "Where this stops: Heckman, \"Sample selection bias as a "
            "specification error\", *Econometrica* (1979); Lakkaraju et al., "
            "\"The selective labels problem\", *KDD* (2017); Hand and Henley, "
            "\"Can reject inference ever work?\", *IMA Journal of Mathematics "
            "Applied in Business and Industry* (1993); episode 29 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("cohort seeds as listed; about three minutes of "
                            "CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Outcomes only for the approved",
        f"""Episode 38 found that a missing value can record a decision. A missing label always does. A lender sees whether a loan was repaid only if it made the loan; a hospital sees whether a treatment worked only for patients who got it; a platform sees whether a recommendation was clicked only if it was shown. The next model is trained on those outcomes and then applied to everyone, including the kind of case the last decision turned away. The sentence that makes this feel routine: *a model trained on the outcomes you observed predicts the outcomes you did not*.

The simulation is a lender. Each applicant has two features the model sees, and a third, u, that it does not: something a loan officer reading the file might pick up. The chance of default depends on all three. An old policy accepted the half of each cohort with the lowest risk score, and outcomes exist only for them. In two of four worlds the old policy's score used only the two features. In the other two, the officer's judgement of u also went into it. In two of the worlds risk is straight in the features; in two it curves. A logistic model and boosted trees are trained on the accepted half and applied to a fresh cohort, accepted and refused alike.

{snip['main'].markdown()}

In the simplest world the sentence holds. The policy used only what the model sees, risk is straight, and logistic regression predicts a default rate of **{sf['refused_pred']:.3f}** among the refused, who default at {sf['refused_rate']:.3f}. Its AUC on the whole cohort, {sf['auc_all']:.3f}, is the same as that of a model trained on everyone's outcomes.""",
        figures=[figs["f0"]])

    post.add(
        "Selection on what the model sees",
        f"""That world is episode 29's covariate shift in another form. Selecting on the features changes which inputs the model sees, not the relationship between inputs and outcome, so a correctly specified model learns the right relationship from the accepted half and carries it to the refused half. Episode 29 found that covariate shift charges only for misspecification, and so does this.

The charge appears as soon as the model cannot extend what it learned. Boosted trees, trained on the same accepted half, predicted **{hsf['refused_pred']:.3f}** for the refused: trees cannot predict beyond the range they were trained on, as episode 15 measured, and the refused sit beyond it. In the curved world, logistic regression is the misspecified model, and it predicted {cf['refused_pred']:.3f} for a group that defaulted at {cf['refused_rate']:.3f}. Its AUC on the whole cohort fell to **{cf['auc_all']:.3f}**, against {cf['auc_oracle']:.3f} for the same model trained on everyone: probably because a straight line fitted to the accepted region of a curved relationship does not extend to the refused region.""")

    post.add(
        "Selection on what the model cannot see",
        f"""When the officer's judgement went into the old decision, the accepted are not only the applicants with good features but the ones with good u, among applicants with similar features. In the training data, then, applicants with mediocre features repaid well, because those who were accepted had something else in their favour. The model learns that mediocre features are safe.

Logistic regression predicted a default rate of **{so['refused_pred']:.3f}** for the refused. They defaulted at **{so['refused_rate']:.3f}**. Its ranking survived, an AUC of {so['auc_all']:.3f} against {so['auc_oracle']:.3f} for a model trained on everyone, because the selection shifted the level of its predictions more than their order; its log loss, {so['nll']:.3f} against {so['nll_oracle']:.3f}, records the damage. A lender who used its probabilities to price risk would price the refused applicants at less than a third of their real default rate.

There are two losses here, and only one can be repaired. Even the model trained on every applicant's outcome, which no lender has, predicted {so['oracle_refused_pred']:.3f} for the refused, below their {so['refused_rate']:.3f}: the refused were selected partly for a u that the features do not contain, so no model of the features can know they are worse. That is episode 34's subgroup miscalibration, produced by the decision itself. The gap from {so['refused_pred']:.3f} to {so['oracle_refused_pred']:.3f} is the selection bias Heckman described in 1979, and that part can be fixed.""",
        figures=[figs["f1"]])

    post.add(
        "Validation on the accepted",
        f"""The same selection undermines the usual check. A model like this is validated on held-out accepted applicants, because they are the only ones with outcomes. In the straight world with a features-only policy, the AUC measured that way was **{sf['auc_accepted']:.3f}**, against {sf['auc_all']:.3f} on the whole cohort: the accepted are a narrower, more similar group, and ranking within them is harder. In the curved world it went the other way, {cf['auc_accepted']:.3f} on the accepted against {cf['auc_all']:.3f} on everyone, because the model fits the accepted region and fails outside it. In the curved world with the officer, the model ranked the whole cohort worse than chance, {co['auc_all']:.3f}, while its validation on the accepted reported {co['auc_accepted']:.3f}. Validation on the accepted can err in either direction, and nothing in it says which.""")

    post.add(
        "Buying the missing outcomes",
        f"""The repair that works in every world is to stop selecting for some of the data. Suppose 5% of applicants, about {sf['explore_rows']:.0f} per cohort, had been approved at random, regardless of score and officer:

{snip['explore'].markdown()}

A logistic model trained on those thousand or so randomly approved applicants alone predicted {so['explore_refused_pred']:.3f} for the refused in the officer world and {cf['explore_refused_pred']:.3f} in the curved one, matching the model trained on everyone to within 0.01 in all four worlds. Recalibrating the accepted-only model on the random approvals, refitting only its intercept and slope, worked when the shape was right ({so['recal_refused_pred']:.3f}) and not when it was wrong ({cf['recal_refused_pred']:.3f} in the curved world, against {cf['oracle_refused_pred']:.3f}).

Approving some applicants at random has a cost, paid in defaults among applicants who would have been refused. It is the only source of outcomes from the refused region, and the alternatives, collectively called reject inference, have to assume what the refused would have done. Hand and Henley's verdict in 1993 was that reject inference cannot work reliably unless assumptions or information from outside the accepted data are brought in; the exploration sample is that information, measured rather than assumed.""")

    post.add(
        "Where this breaks",
        """**One officer, one unobserved variable.** Real decisions use many unrecorded signals, some correlated with the recorded features. The size of the bias depends on how much the decision used that the model cannot see, which is rarely known.

**Outcomes are clean.** A loan's outcome takes years and depends on the terms offered, which the policy also chose. The same structure holds for treatments and recommendations, with their own versions of these complications.

**Random approvals of 5%.** Fewer would give noisier estimates; in practice exploration is often stratified, approving at random only near the decision boundary, which costs less and reaches less of the refused region.

**Static.** Each new model sets the next policy, which selects the next training data. Feedback over several rounds can compound the bias measured here, and is not simulated.""")

    post.add(
        "What to keep",
        f"""1. Selection on the features the model sees costs only misspecification: a correctly specified model predicted {sf['refused_pred']:.3f} for refused applicants who defaulted at {sf['refused_rate']:.3f}; boosted trees, which cannot extrapolate, {hsf['refused_pred']:.3f}.

2. Selection on what the model cannot see miscalibrates it without warning: {so['refused_pred']:.3f} predicted for a group defaulting at {so['refused_rate']:.3f}, with its AUC intact.

3. AUC on accepted applicants was {sf['auc_accepted']:.3f} against {sf['auc_all']:.3f} on everyone in one world and {cf['auc_accepted']:.3f} against {cf['auc_all']:.3f} in another.

4. A small randomly approved sample recovered what a model trained on everyone would know. Find out what information the past decision used, and buy some outcomes from outside its approvals.""")

    post.add(
        "Exercise",
        """For a model trained on outcomes that exist only because of an earlier decision, write down every input to that decision and mark the ones your model does not see.

Then compare the model's mean prediction with the observed outcome rate in the slice of approved cases nearest the decision boundary. If the earlier decision used information you lack, that slice is where the gap first shows, and its size bounds how wrong the model may be beyond the boundary.""")

    post.add(
        "Next",
        """Arc VIII ends with a feature built from the labels themselves. Episode 40 encodes a category with thousands of levels by its mean outcome, fitted carefully on the training rows only, and asks whether "inside the folds" is enough to make it safe.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
