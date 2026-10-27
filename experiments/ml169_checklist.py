"""ML 69: One Project, Every Check, and the Threshold Mattered More Than the Model.

The sixty-ninth episode of *Machine Learning, Taught Through What
Breaks*, in Arc XIV. One project, breast-cancer diagnosis with a missed
malignancy costing ten false alarms, run through every check the series
recommends, in the order a project meets them.

* Cross-validation preferred logistic regression (0.985 against boosted
  trees' 0.957); the corrected test put the difference at p = 0.11. The
  test set preferred boosted trees (0.965 against 0.953) on two
  disagreements, McNemar p = 0.5. Neither accuracy could choose.
* The threshold could. Moving logistic regression's threshold from 0.5 to
  1/11, as the cost ratio says, cut missed malignancies from 4 to 1 and
  the cost per case from 0.257 to 0.140. At their own best thresholds the
  three models cost 0.140 to 0.175.
* Boosted trees missed the same three malignancies at either threshold:
  it had given them probabilities below 1/11.

Run: `standarderror run ml169_checklist --publish`
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

MODELS = ("logistic", "boosted trees", "recalibrated boosted")


def compute() -> dict:
    return {"c": course.checklist_run()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    c = res["c"]
    tl, tb = c["test_logistic"], c["test_boosted trees"]
    cl = c["cost"]["logistic"]
    g = c["groups"]
    body = [
        ["1", "baseline", f"majority class {c['majority']:.3f}",
         "every model clears it"],
        ["2", "compare by CV, corrected (ep. 44)",
         f"{c['cv']['logistic']:.3f} vs {c['cv']['boosted trees']:.3f}, "
         f"p {c['p_corrected']:.2f}", "no winner"],
        ["3", "tune, nested (ep. 35)",
         f"SVM {c['svm_best_cv']:.3f} reported, {c['svm_nested']:.3f} nested",
         "tuning adds nothing"],
        ["4", "test once, paired (ep. 1, 45)",
         f"{tl['acc']:.3f} vs {tb['acc']:.3f}, "
         f"{c['discordant'][0]}:{c['discordant'][1]} disagreements",
         "no winner"],
        ["5", "probabilities (ep. 14, 34)",
         f"{c['extreme']['boosted trees']:.0%} of boosted below 1% or above 99%",
         "boosted too sure"],
        ["6", "threshold from cost (ep. 46, 47)",
         f"misses {c['missed_at_half']} to {c['missed_at_ratio']}, cost "
         f"{cl['at_half']:.3f} to {cl['at_ratio']:.3f}", "the decision"],
        ["7", "subgroups (ep. 34, 63)",
         f"misses {g[0]['missed']}, {g[1]['missed']}, {g[2]['missed']} by size third",
         "middle third"],
        ["8", "stability (ep. 11, 20)",
         f"{c['flip_rows']} of {c['n_test']} flags change on resampling",
         "a review queue"],
        ["9", "shift (ep. 29, 61)",
         f"{c['shift_alarms']} of 30 features alarm on large tumours",
         "labels needed"],
    ]
    out["f0"] = charts.table_image(
        body, header=["", "check (from)", "result", "what it changed"],
        title="One project, nine checks, in the order a project meets them",
        subtitle=(f"Breast cancer, malignant as positive; {c['n_train']} "
                  f"tumours to build, {c['n_test']} held out; a missed "
                  "malignancy costs ten false alarms."),
        source="Breast Cancer Wisconsin (Diagnostic), via scikit-learn.",
        alt=("A nine-row table of checks, results and what each changed. "
             "The comparisons found no winner; the threshold changed the "
             "decision."),
        caption=("Two checks chose nothing, which is a result; the threshold "
                 "was the check that changed what the project would do."),
        align="rlll", bold_cells={(5, 3)},
        path=str(IMG / f"ml169-f0-checks.{EXT}"))[0]

    def bars(ax, m):
        w = 0.26
        for j, (key, lab) in enumerate((("at_half", "threshold 0.5"),
                                        ("at_ratio", "threshold 1/11"),
                                        ("best", "best threshold"))):
            v = [c["cost"][k][key] for k in MODELS]
            ax.bar(np.arange(3) + (j - 1) * w, v, w, color=m.series[j],
                   label=lab)
            for i, x in enumerate(v):
                ax.text(i + (j - 1) * w, x + 0.004, f"{x:.3f}", ha="center",
                        fontsize=7.2, color=m.ink)
        ax.set_xticks(range(3))
        ax.set_xticklabels(MODELS)
        ax.set_ylim(0, 0.3)
        ax.legend(fontsize=8, frameon=False, loc="upper right")

    out["f1"] = charts.diagram(
        bars,
        title="The threshold moved the cost more than the choice of model did",
        subtitle=("Cost per held-out tumour (a miss 10, a false alarm 1) for "
                  "three models at three thresholds. Flagging every tumour "
                  f"would cost {c['flag_all']:.3f}."),
        ylabel="cost per tumour", figsize=(7.6, 4.4),
        source="Breast Cancer Wisconsin (Diagnostic), via scikit-learn.",
        alt=("Grouped bars. Logistic regression falls from 0.257 at 0.5 to "
             "0.140 at 1/11; both boosted models rise at 1/11."),
        caption=("'Best threshold' is chosen on the test rows themselves, so "
                 "it is a bound, not a result."),
        path=str(IMG / f"ml169-f1-costs.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    c = res["c"]
    best = [c["cost"][k]["best"] for k in MODELS]

    def p1(panel, m):
        panel.bar([0, 1], [c["missed_at_half"], c["missed_at_ratio"]],
                  color=[m.series[1], m.series[0]])

    def p2(panel, m):
        panel.bar(range(3), best, color=m.series[2])

    def p3(panel, m):
        panel.bar([0, 1], [c["n_test"] - c["flip_rows"], c["flip_rows"]],
                  color=[m.series[3], m.series[1]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=69,
        headline="One project, every check",
        panels=[(p1, f"{c['missed_at_half']} to {c['missed_at_ratio']}",
                 "misses, 0.5 to 1/11"),
                (p2, f"{min(best):.2f}-{max(best):.2f}", "cost, any model"),
                (p3, f"{c['flip_rows']}", "flags that resample")],
        note=("Neither cross-validation nor the test set could choose between "
              "the models. Choosing the threshold from the cost of a miss "
              "changed what the project would do."),
        alt=("Three hand-drawn frames: a tall bar beside a short one; three "
             "near-equal bars; a tall bar beside a small one."),
        path=str(IMG / f"ml169-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["decide"] = s.run("""
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from standarderror.ml import decide

        X, y = load_breast_cancer(return_X_y=True)
        y = 1 - y                                          # malignant = 1
        Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=0)
        models = {
            "logistic": make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=5000)),
            "boosted trees": HistGradientBoostingClassifier(max_iter=200,
                                                            random_state=0),
        }
        for name, m in models.items():
            p = m.fit(Xa, ya).predict_proba(Xt)[:, 1]
            print(f"{name:13s} accuracy {((p >= 0.5) == yt).mean():.3f}  "
                  f"cost at 0.5 {decide._cost(p, yt, 0.5, 10):.3f}  "
                  f"at 1/11 {decide._cost(p, yt, 1 / 11, 10):.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    c = res["c"]
    tl, tb = c["test_logistic"], c["test_boosted trees"]
    C, K = c["cost"], c["counts"]
    cl, cb, cr = C["logistic"], C["boosted trees"], C["recalibrated boosted"]
    kl, kb, kr = K["logistic"], K["boosted trees"], K["recalibrated boosted"]
    best = [C[k]["best"] for k in MODELS]
    g = c["groups"]
    benign = c["n_test"] - sum(x["malignant"] for x in g)
    shift_benign = c["shift_rows"] - c["shift_malignant"]
    rest_false = kl["false_ratio"] - c["shift_false"]

    # The spine, asserted rather than trusted.
    assert (c["n_train"], c["n_test"]) == (398, 171)
    assert c["cv"]["logistic"] > c["cv"]["boosted trees"]
    assert c["p_naive"] < 0.05 < c["p_corrected"]
    assert tb["acc"] > tl["acc"] and c["mcnemar_p"] > 0.3
    assert tl["lo"] < tb["acc"] < tl["hi"]
    assert c["svm_test"] < tl["acc"] and abs(c["svm_best_cv"] - c["svm_nested"]) < 0.01
    assert c["brier"]["boosted trees"] < c["brier"]["logistic"]
    assert c["extreme"]["boosted trees"] > 0.9 and 0.6 < c["extreme"]["logistic"] < 0.75
    assert (c["missed_at_half"], c["missed_at_ratio"]) == (4, 1)
    assert cl["at_ratio"] == cl["best"] == min(best)
    assert cl["at_half"] - cl["at_ratio"] > 3 * (max(best) - min(best))
    assert cb["at_ratio"] > cb["at_half"] and cr["at_ratio"] > cr["at_half"]
    assert kb["missed_half"] == kb["missed_ratio"] == 3
    assert kr["missed_ratio"] == 2 and kr["false_ratio"] > kb["false_ratio"] + 10
    assert [x["missed"] for x in g] == [0, 1, 0]
    assert [x["malignant"] for x in g] == [1, 13, 50]
    assert [x["false_alarms"] for x in g] == [3, 7, 4]
    assert c["flip_rows"] == 23 and c["shift_alarms"] == 15
    assert (c["shift_missed"], c["shift_false"]) == (1, 9) and rest_false == 5

    post = Post(
        title=(f"{SERIES_TAG} 69: One Project, Every Check, and the Threshold "
               "Mattered More Than the Model"),
        slug="ml-69-checklist",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=69,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Breast-cancer diagnosis, run through every check the series "
                  "recommends. Neither accuracy could choose a model; the cost "
                  "of a miss chose the threshold, and that cut missed "
                  "malignancies from four to one."),
        summary=(
            f"Breast cancer, {c['n_train']} tumours to build and {c['n_test']} "
            f"held out, a missed malignancy costing ten false alarms. "
            f"Cross-validation preferred logistic regression "
            f"({c['cv']['logistic']:.3f} against boosted trees' "
            f"{c['cv']['boosted trees']:.3f}, corrected p "
            f"{c['p_corrected']:.2f}); the test set preferred boosted trees "
            f"({tb['acc']:.3f} against {tl['acc']:.3f}, McNemar p "
            f"{c['mcnemar_p']:.1f}). Moving logistic regression's threshold "
            f"from 0.5 to 1/11 cut missed malignancies from "
            f"{c['missed_at_half']} to {c['missed_at_ratio']} and the cost "
            f"from {cl['at_half']:.3f} to {cl['at_ratio']:.3f}; at their best "
            f"thresholds the three models cost {min(best):.3f} to "
            f"{max(best):.3f}. Boosted trees missed the same "
            f"{kb['missed_ratio']} malignancies at either threshold. "
            f"{c['flip_rows']} of {c['n_test']} flags changed under "
            f"resampling."),
        tags=["machine-learning", "evaluation", "decision-making",
              "calibration", "checklist", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, via "
            "scikit-learn: 569 tumours, malignant as the positive class, one "
            "stratified 70/30 split (seed 0).",
            "Models: standardised logistic regression; histogram gradient "
            "boosting (200 rounds); the same boosting with Platt scaling "
            "(5-fold); an RBF SVM tuned over episode 35's 64 settings, with "
            "nested 5-fold CV. Comparison by 10-fold CV with the Nadeau and "
            "Bengio correction; McNemar's exact test on the held-out "
            "disagreements; 50 bootstrap refits for stability; two-sample "
            "Kolmogorov-Smirnov tests at 0.05/30 for the shift check.",
            "Machinery: `standarderror/ml/course.py` and "
            "`standarderror/ml/decide.py`, tested in `tests/test_ml.py`.",
            "Where this stops: Nadeau and Bengio, \"Inference for the "
            "generalization error\", *Machine Learning* (2003); Elkan, \"The "
            "foundations of cost-sensitive learning\", *IJCAI* (2001); "
            "episodes 1, 11, 14, 20, 29, 34, 35, 44 to 47, 61 and 63 of this "
            "series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; about fifteen seconds of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Every check, in order",
        f"""Sixty-eight episodes measured problems one at a time, each in the setting that shows it best. A project meets them all at once, in an order set by the work rather than by the syllabus, and the checks compete for a practitioner's attention. This episode runs one project through all of them and asks which ones changed what the project would actually do. The sentence under test is the one that most model-building effort silently assumes: *the main decision in a project is which model to use*.

The project is the series' oldest data set, breast-cancer diagnosis on 569 tumours, with malignant as the positive class and a cost attached: a missed malignancy costs ten times a false alarm, the ratio episodes 46 and 47 used. Of the tumours, {c['n_train']} are used to build and {c['n_test']} are held out until the end. Two candidates stand in for the usual shortlist: logistic regression, standardised, and gradient-boosted trees, with a tuned support-vector machine as the model a search would offer.

{snip['decide'].markdown()}""")

    post.add(
        "Choosing a model: two accuracies, opposite answers",
        f"""The baseline comes first: always calling a tumour benign would be right {c['majority']:.1%} of the time, and every candidate clears it easily. Ten-fold cross-validation on the {c['n_train']} building rows then preferred logistic regression, {c['cv']['logistic']:.3f} against {c['cv']['boosted trees']:.3f}. The naive paired t-test on the folds called the difference significant at p = {c['p_naive']:.3f}; the corrected test that episode 44 recommends, which accounts for folds sharing most of their training rows, put it at p = {c['p_corrected']:.2f}. The difference was not established.

Tuning the support-vector machine over 64 settings reported {c['svm_best_cv']:.3f}; nested cross-validation, episode 35's honest version, estimated {c['svm_nested']:.3f}. The search's optimism was small here, and the tuned model scored {c['svm_test']:.3f} on the held-out tumours, the lowest of the three, so it bought nothing.

The held-out tumours, consulted once, as episode 45 insists, then preferred the other model: boosted trees {tb['acc']:.3f}, logistic regression {tl['acc']:.3f}, with a 95% interval of [{tl['lo']:.3f}, {tl['hi']:.3f}] for the latter that contains the former. The two models disagreed on only {sum(c['discordant'])} tumours, both in boosted trees' favour, and McNemar's test on two disagreements gives p = {c['mcnemar_p']:.1f}. Two honest measurements of accuracy pointed in opposite directions, and neither could choose. That is a result, not a failure: it says the choice of model, measured by accuracy, is not where this project's outcome will be decided.""",
        figures=[figs["f0"]])

    post.add(
        "Choosing a threshold: the check that changed the decision",
        f"""Accuracy treats a missed malignancy and a false alarm as equal, and they are not. Episode 46 derived the threshold for a miss that costs ten false alarms: flag a tumour when its probability of malignancy is at least 1/11, provided the probabilities are calibrated. For logistic regression that one change cut the missed malignancies from {kl['missed_half']} to {kl['missed_ratio']}, raised the false alarms from {kl['false_half']} to {kl['false_ratio']}, and cut the cost per tumour from {cl['at_half']:.3f} to **{cl['at_ratio']:.3f}**, which was also the lowest cost any threshold could reach on these tumours.

For boosted trees the same rule made things worse: {cb['at_half']:.3f} at 0.5, {cb['at_ratio']:.3f} at 1/11. The reason is the calibration check. {c['extreme']['boosted trees']:.0%} of boosted trees' probabilities were below 1% or above 99%, against {c['extreme']['logistic']:.0%} of logistic regression's, and the boosted model missed the same {kb['missed_ratio']} malignancies at either threshold, because it had given all three a probability below 1/11. A lower threshold could only add false alarms. Its Brier score, {c['brier']['boosted trees']:.4f} against logistic regression's {c['brier']['logistic']:.4f}, was the better of the two; a good average score did not mean usable probabilities at the point where the decision was made, which is episode 14's lesson in a new setting.

Recalibrating the boosted model with Platt scaling, episode 34's repair, caught one of the three missed malignancies at the cost of {kr['false_ratio'] - kb['false_ratio']} more false alarms, and cost {cr['at_ratio']:.3f} at 1/11. On these tumours, the repair cost more than it saved.

Put together: at their own best thresholds, which are chosen on the test rows and so flatter all three, the models cost between {min(best):.3f} and {max(best):.3f} per tumour. For the logistic model, the threshold alone moved the cost by {cl['at_half'] - cl['at_ratio']:.3f}, more than three times the spread between models. The sentence under test was wrong for this project: the main decision was the threshold, and the model mattered mainly through whether its probabilities could support one.""",
        figures=[figs["f1"]])

    post.add(
        "After the decision: who, how stable, and for how long",
        f"""Three checks remained, and each produced a qualification rather than a change. By tumour size, the smallest third held {g[0]['malignant']} malignant tumour and the model missed none; the largest held {g[2]['malignant']} and it missed none; the single miss was in the middle third, which held {g[1]['malignant']} malignant tumours and drew {g[1]['false_alarms']} of the {kl['false_ratio']} false alarms. A subgroup check on 171 tumours cannot establish a rate, but it can show where the hard cases sit, and here they sit in the middle.

Refitting the logistic model on 50 bootstrap resamples of the building rows changed the flag for {c['flip_rows']} of the {c['n_test']} held-out tumours ({c['flips']:.1%}), the instability of episodes 11 and 20. Those {c['flip_rows']} tumours are where the model's answer depends on which patients it happened to learn from, and a natural use of the result is a review queue: the flags that resampling cannot agree on go to a person.

Finally, a shift check of the kind episode 61 measured: the {c['shift_rows']} held-out tumours above the building rows' median size differed from the building rows on {c['shift_alarms']} of 30 features. The alarm said the inputs differed, as it should for a subset chosen by size. It could not say whether the model was worse there. With labels, it was mixed: one malignancy missed among {c['shift_malignant']}, and {c['shift_false']} false alarms among the {shift_benign} benign tumours, against {rest_false} among the {benign - shift_benign} benign tumours below the median.""")

    post.add(
        "The checklist",
        """Each arc of the series reduces to one question to ask before trusting a number. In the order a project meets them:

1. **What a score means.** What is the score's error bar, and was every step fitted inside the folds?
2. **Linear models.** Are the features on scales the penalty respects, and could the classes be separable?
3. **Trees and ensembles.** Would a resample tell a different story, and are the probabilities overfitted though the accuracy is not?
4. **Reading a model.** Compared with what, and does the explanation ask about rows that exist?
5. **Training a network.** Was each setting chosen on validation data, across more than one seed?
6. **Generalisation.** Do new inputs look like the training inputs, and could an extrapolation have seen the floor?
7. **Around the model.** Is each routine step, reduction, clustering, calibration, search, answering the question being asked?
8. **The rows.** Are the rows independent, and who is missing from them?
9. **Comparing models.** Is the comparison paired and corrected, and has the test set been consulted before?
10. **Decisions.** What does each kind of mistake cost, and are the probabilities calibrated where the threshold sits?
11. **Grading a language model.** How many questions, which prompt, and who is the judge?
12. **Without labels.** What would the method report on data with no structure?
13. **After shipping.** How will you know the model is wrong, and how long until the labels arrive?
14. **The whole analysis.** Does the result hold on another seed, and how many choices were made after the first score was seen?""")

    post.add(
        "Where this breaks",
        f"""**One split.** The project used one 70/30 split, as a project would. Episode 66's reruns showed how often single-draw results move; the costs here differ between models by a few tumours, and another split could reorder them. The threshold's effect, {round((cl['at_half'] - cl['at_ratio']) * c['n_test'])} cost units across the held-out tumours, is larger than that.

**An easy data set.** Breast cancer is nearly separable. On harder problems the choice of model can matter more than it did here.

**A known cost ratio.** The threshold's value depends on the ratio of ten, which in practice is a judgement, and is worth stating and varying.

**Best thresholds on the test rows.** Those costs are bounds, used only to compare spreads; the project's own result is the 1/11 threshold fixed in advance.""")

    post.add(
        "What to keep",
        f"""1. Cross-validation preferred logistic regression ({c['cv']['logistic']:.3f} against {c['cv']['boosted trees']:.3f}, corrected p {c['p_corrected']:.2f}); the test set preferred boosted trees ({tb['acc']:.3f} against {tl['acc']:.3f}, McNemar p {c['mcnemar_p']:.1f}). Neither could choose.

2. A threshold from the cost of a miss cut logistic regression's missed malignancies from {c['missed_at_half']} to {c['missed_at_ratio']} and its cost from {cl['at_half']:.3f} to {cl['at_ratio']:.3f}; the models' best costs spanned {min(best):.3f} to {max(best):.3f}.

3. Boosted trees, with the better Brier score, missed the same {kb['missed_ratio']} malignancies at both thresholds, because its probabilities were too extreme for a threshold to reach them.

4. Choose the threshold before you choose the model, and choose the model whose probabilities can support the threshold.""")

    post.add(
        "Exercise",
        """Take a project you have finished and rerun it in this episode's order: baseline, corrected comparison, nested tuning, one look at the test set, calibration, a threshold from costs you write down, subgroups, resampled flags and a shift check.

Mark each check by whether it changed the decision, qualified it or did nothing. If the comparison of models did nothing and the threshold did, the time spent choosing between models was spent in the wrong place.""")

    post.add(
        "Next",
        """Episode 70 closes the series: all sixty-nine sentences, what measurement did to each, and how many of the series' own numbers survived being checked twice.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
