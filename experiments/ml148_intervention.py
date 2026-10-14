"""ML 48: What a Model Relies On Is Not What to Change.

The forty-eighth episode of *Machine Learning, Taught Through What
Breaks*. "The features a model relies on most are the levers that move the
outcome." Measured on a small simulated causal system, where the effect of
changing each feature can be computed by simulating the change:

* A test result produced by the outcome itself carried almost all of the
  models' permutation importance (0.43 to 0.45 of AUC), and moving it by one
  standard deviation in the model's input raised the predicted risk by
  0.31. Changing it in the system changes nothing.
* A habit that shares a cause with the outcome lowered the predicted risk
  by 0.025 to 0.029 per standard deviation; its true effect is zero.
* The one real lever, a treatment that raises risk by 0.085 per standard
  deviation, was moved 0.024 by the boosted model and 0.026 by the
  logistic one. A regression on the treatment and the common cause
  recovered 0.089; one that simply dropped the test result gave 0.144.

Run: `standarderror run ml148_intervention --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import decide
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"i": decide.importance_vs_effect()}


# ------------------------------------------------------------------ figures

def _bars(ax, m, i):
    names = i["names"]
    x = np.arange(len(names))
    w = 0.27
    ax.bar(x - w, [i["models"]["boosted trees"]["whatif"][n] for n in names],
           width=w, color=m.series[1], label="boosted trees: change the input")
    ax.bar(x, [i["models"]["logistic"]["whatif"][n] for n in names], width=w,
           color=m.series[2], label="logistic: change the input")
    ax.bar(x + w, [i["true"][n] for n in names], width=w, color=m.ink,
           label="the system: change the feature")
    ax.axhline(0, color=m.grid, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def figures(res: dict) -> dict:
    out: dict = {}
    i = res["i"]
    bt = i["models"]["boosted trees"]

    out["f0"] = charts.diagram(
        lambda ax, m: _bars(ax, m, i),
        title="Moving a feature in the model, and in the world",
        subtitle=("Change in the probability of the outcome when each "
                  "feature is raised by one standard deviation: in the "
                  "fitted model's input (coloured), and in the simulated "
                  "system that generated the data (black). 20,000 rows."),
        xlabel="",
        ylabel="change in probability of the outcome",
        figsize=(7.6, 4.6),
        source="Simulated causal system; standarderror/ml/decide.py.",
        alt=("Grouped bars. For the test result the models predict a rise "
             "of about 0.31 and the system shows none. For the habit the "
             "models predict a small fall and the system none. For the "
             "treatment the models predict about 0.025 and the system 0.085."),
        caption=(f"The test result moves the model's prediction by "
                 f"{bt['whatif']['x3 test result']:.2f} and the outcome by "
                 f"nothing; the treatment moves the outcome by "
                 f"**{i['true']['x2 treatment']:.3f}** and the model by "
                 f"{bt['whatif']['x2 treatment']:.3f}."),
        path=str(IMG / f"ml148-f0-effects.{EXT}"))[0]

    body = []
    for n in i["names"]:
        body.append([n, f"{bt['importance'][n]:.4f}",
                     f"{i['models']['logistic']['importance'][n]:.4f}",
                     f"{bt['whatif'][n]:+.3f}",
                     f"{i['models']['logistic']['whatif'][n]:+.3f}",
                     f"{i['true'][n]:+.3f}"])
    body.append(["x2, adjusting for the common cause", "", "", "",
                 f"{i['adjusted_x2']:+.3f}", f"{i['true']['x2 treatment']:+.3f}"])
    body.append(["x2, dropping the test result", "", "", "",
                 f"{i['without_x3_x2']:+.3f}", f"{i['true']['x2 treatment']:+.3f}"])
    out["f1"] = charts.table_image(
        body,
        header=["feature", "importance, trees", "importance, logistic",
                "what-if, trees", "what-if, logistic", "true effect"],
        title="Reliance, prediction and effect",
        subtitle=("Permutation importance is the fall in AUC on fresh rows "
                  "when the feature is shuffled. What-if and true effect are "
                  "changes in the probability of the outcome for a "
                  "one-standard-deviation rise."),
        source="Simulated causal system; standarderror/ml/decide.py.",
        alt=("A six-row table: the test result has by far the largest "
             "importance and what-if and no true effect; the treatment has "
             "small importance, a small what-if and the only true effect."),
        caption=("The last two rows are logistic regressions chosen with the "
                 "causal structure in mind: one given the common cause, one "
                 "simply without the downstream test."),
        align="lrrrrr",
        path=str(IMG / f"ml148-f1-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    i = res["i"]
    bt = i["models"]["boosted trees"]

    def p1(panel, m):
        panel.bar(range(4), [bt["importance"][n] for n in i["names"]],
                  color=[m.series[1], m.series[2], m.series[3], m.grid])

    def p2(panel, m):
        panel.bar(range(4), [i["true"][n] for n in i["names"]],
                  color=[m.series[1], m.series[2], m.series[3], m.grid])
        panel.set_ylim(-0.01, 0.1)

    def p3(panel, m):
        panel.bar([0, 1, 2], [bt["whatif"]["x2 treatment"], i["adjusted_x2"],
                              i["true"]["x2 treatment"]],
                  color=[m.series[1], m.series[2], m.ink])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=48,
        headline="Levers and signals",
        panels=[(p1, f"{bt['importance']['x3 test result']:.2f}",
                 "test result importance"),
                (p2, f"{i['true']['x3 test result']:.0f}", "test result effect"),
                (p3, f"{i['adjusted_x2']:.3f}", "treatment, adjusted")],
        note=("A predictive model uses whatever carries information about "
              "the outcome, including its consequences and its fellow "
              "travellers. Which features would change it is a different "
              "question, answered by the causal structure."),
        alt=("Three hand-drawn frames of bars: one tall bar among short "
             "ones; one bar among empty places; three bars, the last two "
             "similar."),
        path=str(IMG / f"ml148-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["main"] = s.run("""
        from standarderror.ml import decide

        r = decide.importance_vs_effect()   # 20,000 simulated rows
        m = r["models"]["boosted trees"]
        for n in r["names"]:
            print(f"{n:15s} importance {m['importance'][n]:.4f}, "
                  f"what-if {m['whatif'][n]:+.3f}, true effect {r['true'][n]:+.3f}")
        """)
    out["adj"] = s.run("""
        print(f"treatment alone, no other features:   {r['alone_x2']:+.3f}")
        print(f"treatment, logistic on all four:      "
              f"{r['models']['logistic']['whatif']['x2 treatment']:+.3f}")
        print(f"treatment, given the common cause:    {r['adjusted_x2']:+.3f}")
        print(f"treatment, without the test result:   {r['without_x3_x2']:+.3f}")
        print(f"treatment, true:                      "
              f"{r['true']['x2 treatment']:+.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    i = res["i"]
    bt, lg, tr = i["models"]["boosted trees"], i["models"]["logistic"], i["true"]
    x1, x2, x3 = "x1 habit", "x2 treatment", "x3 test result"

    # The spine, asserted rather than trusted.
    for m in (bt, lg):
        assert m["importance"][x3] > 0.4
        assert m["importance"][x3] > 100 * max(m["importance"][x1], m["importance"][x2])
        assert m["whatif"][x3] > 0.3 and m["whatif"][x1] < -0.02
        assert 0.02 < m["whatif"][x2] < 0.03
    assert tr[x1] == 0 and tr[x3] == 0 and 0.08 < tr[x2] < 0.09
    assert abs(i["adjusted_x2"] - tr[x2]) < 0.006
    assert i["without_x3_x2"] > 1.6 * tr[x2]
    assert tr[x2] > 3 * bt["whatif"][x2]
    assert i["alone_x2"] > 2.2 * tr[x2]

    post = Post(
        title=f"{SERIES_TAG} 48: What a Model Relies On Is Not What to Change",
        slug="ml-48-intervention",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=48,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Importance and what-if analyses describe the model. In a "
                  "system with a common cause and a downstream signal, the "
                  "feature the model relied on most did nothing when changed, "
                  "and the real lever ranked no higher than a feature with no "
                  "effect."),
        summary=(
            f"A simulated system with an unmeasured common cause, a habit "
            f"it drives, a treatment that raises the outcome's risk, and a "
            f"test result produced by the outcome. Boosted trees put "
            f"{bt['importance'][x3]:.2f} of permutation importance on the "
            f"test result, and raising it by one standard deviation in the "
            f"model's input raised the predicted risk by {bt['whatif'][x3]:.2f}; "
            f"changing it in the system changes nothing. The habit lowered "
            f"the predicted risk by {abs(bt['whatif'][x1]):.3f} and has no "
            f"effect. The treatment, whose true effect is "
            f"{tr[x2]:.3f}, moved the prediction by {bt['whatif'][x2]:.3f}. "
            f"A regression on the treatment and the common cause recovered "
            f"{i['adjusted_x2']:.3f}."),
        tags=["machine-learning", "data-science", "statistics",
              "causal-inference", "interpretability", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: U standard normal (unmeasured); habit x1 = −0.8 U + "
            "0.6 e; treatment x2 = 0.6 U + 0.8 e; outcome latent 1.2 U + "
            "0.5 x2 + logistic noise, outcome = latent > 0; test result x3 = "
            "latent + 0.8 e; x4 pure noise. 20,000 training rows (seed 0), "
            "20,000 fresh rows (seed 1), interventions simulated on 200,000 "
            "rows (seed 2).",
            "Machinery: `standarderror/ml/decide.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's HistGradientBoostingClassifier, "
            "LogisticRegression and permutation_importance (AUC, 5 repeats).",
            "Where this stops: Pearl, *Causality* (2009); Hernán and Robins, "
            "*Causal Inference: What If* (2020); Zhao and Hastie, \"Causal "
            "interpretations of black-box models\", *Journal of Business & "
            "Economic Statistics* (2021); episodes 16 to 18 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("seeds 0, 1 and 2; seconds of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "From what predicts to what to change",
        f"""Episodes 46 and 47 acted on a model's ranking: flag these cases, clear those. The next kind of decision acts on the world: change something so that the outcome changes. The model is asked which features matter, through permutation importance (episode 16), partial dependence (episode 17) or a what-if tool that moves one input and watches the prediction, and the answer is read as a list of levers. The sentence under test: *the features a model relies on most are the levers that move the outcome*.

Episodes 16 to 18 found that those tools describe the model, not the data. This episode goes one step further and asks whether they describe the world. That needs a world whose causal structure is known, so here is a small one. An unmeasured trait U raises the risk of an outcome. U also lowers a habit, x1, which has no effect of its own. U raises the level of a treatment, x2, and the treatment genuinely raises the risk. A test result, x3, is produced by the outcome's own underlying state, the way a symptom or a lab value follows a condition. x4 is noise.

A boosted model and a logistic regression are trained on 20,000 rows of the four observed features. For each feature, three numbers: how much the model relies on it (permutation importance), what the model predicts when the feature is raised by one standard deviation in its input, and what actually happens to the outcome when the feature is raised by one standard deviation in the simulated system.

{snip['main'].markdown()}

The test result carries almost all of the model's reliance: shuffling it costs **{bt['importance'][x3]:.2f}** of AUC, and raising it in the input raises the predicted risk by {bt['whatif'][x3]:.2f}. In the system, raising the test result changes nothing, because it is a consequence of the outcome, not a cause.""",
        figures=[figs["f0"]])

    post.add(
        "Three kinds of wrong",
        f"""Each feature is wrong in a different way, and each way is one of the standard patterns of causal reasoning.

The test result is downstream of the outcome. It is the best predictor because it is partly a measurement of the outcome, and an intervention on it, which is like treating the thermometer, does nothing. A model that predicts well will typically lean on such variables when they are available, which is exactly what it should do as a predictor.

The habit is a fellow traveller. It is associated with lower risk only because the unmeasured trait drives both, so the model, rightly for prediction, treats it as evidence of lower risk: {bt['whatif'][x1]:+.3f} per standard deviation. Encouraging the habit would not move the outcome at all.

The treatment is the one real lever, and the model nearly hides it. Its importance is {bt['importance'][x2]:.4f}, indistinguishable from the habit's, and the model's what-if puts its effect at {bt['whatif'][x2]:+.3f}, under a third of the true **{tr[x2]:+.3f}**. A likely reason is that the test result already records most of what the treatment did to the outcome's underlying state, so given the test result the treatment adds little predictive information and the model has little reason to use it.""",
        figures=[figs["f1"]])

    post.add(
        "What recovers the lever",
        f"""The effect is recoverable from the same data, by a model built for the question rather than for prediction:

{snip['adj'].markdown()}

Given the common cause and the treatment only, a logistic regression's what-if is {i['adjusted_x2']:+.3f}, within {abs(i['adjusted_x2'] - tr[x2]):.3f} of the truth: adjusting for the confounder and leaving out the variable downstream of the outcome is the textbook recipe, and here it works. The naive alternative of looking at the treatment on its own is far worse: a logistic regression on the treatment alone puts its effect at {i['alone_x2']:+.3f}, {i['alone_x2'] / tr[x2]:.1f} times the truth, because people with more of the treatment also carry more of the unmeasured trait that raises the risk. The same data gave four answers for one lever, from under a third of the truth to more than twice it, depending only on which other features were in the model. Simply dropping the test result is not enough either: the habit is then only a partial stand-in for the unmeasured trait, and the treatment's apparent effect, {i['without_x3_x2']:+.3f}, overstates the truth by {i['without_x3_x2'] / tr[x2] - 1:.0%}.

The recipe needed something no predictive pipeline supplies: a measurement of the common cause, and the knowledge of which variables come before the outcome and which after. Zhao and Hastie made the conditions precise for black-box models: a partial dependence plot can be read causally when the features held fixed block every back-door path and include nothing downstream. Those conditions are about the world, and no inspection of the model can check them.

The column labelled true effect is itself what an experiment would report. It was computed by raising each feature by one standard deviation for 200,000 simulated people, everything else left to follow from the change, and comparing their outcomes with an untouched population: a randomised trial with perfect compliance, run in software. When the structure is unknown, running that trial for real, on the treatment someone actually proposes to change, is the measurement that needs no assumption about which features to include, which is why the uplift models of the next episode start from one.""")

    post.add(
        "Where this breaks",
        """**A system written down in advance.** The causal structure here is known because it was chosen. In practice it is a hypothesis, and the adjusted estimate is only as good as the assumption that U is the only common cause.

**The common cause was measurable.** U was handed to the adjusted model. When it cannot be measured, adjustment is impossible, and the options are randomised experiments, instruments or bounds.

**Linear effects.** The treatment's effect is linear in the log-odds. Effects that differ between people, the subject of the next episode, need more than one number.

**Four features.** Real models have hundreds, and the downstream ones are often hard to recognise: a variable recorded after the outcome began, or produced by the same process that records it.""")

    post.add(
        "What to keep",
        f"""1. The feature the models relied on most, a consequence of the outcome, carried {bt['importance'][x3]:.2f} of AUC and has no effect when changed.

2. A fellow traveller of the outcome lowered the predicted risk ({bt['whatif'][x1]:+.3f}) and has no effect; the real lever ranked no higher than it in importance, with a what-if under a third of its true effect.

3. A regression on the treatment and the common cause recovered {i['adjusted_x2']:+.3f} against a true {tr[x2]:+.3f}; dropping the downstream feature alone gave {i['without_x3_x2']:+.3f}.

4. Read importance and what-if analyses as statements about the model. Questions about what to change need the causal structure, and a model built for that question, or an experiment.""")

    post.add(
        "Exercise",
        """List the features your model relies on most and, for each, write down whether it is recorded before or after the outcome could have influenced it, and what common causes it might share with the outcome.

For any feature someone proposes to change, find out whether the change has ever been made deliberately, in an experiment or a policy change, and compare what happened with what your model's what-if predicts.""")

    post.add(
        "Next",
        """When the lever is a treatment that can be given to some people and not others, the decision is whom to give it to. Episode 49 compares the usual choice, the people at highest risk, with the people the treatment would actually help.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
