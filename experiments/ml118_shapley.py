"""ML 18: A SHAP Value Is an Answer to "Compared With What?".

The eighteenth episode of *Machine Learning, Taught Through What Breaks*.
"SHAP values attribute a prediction to its features." They do, exactly and
uniquely -- relative to a background distribution someone has to choose.
Computed exactly, by enumerating all 32 coalitions, for a five-feature
boosted model on the breast-cancer data:

* The values always add up: to the prediction minus the background's mean
  prediction, to 1e-14.
* Change the background from all training tumours to benign ones only and
  the feature with the largest attribution changes for 58% of test tumours;
  15% of all attributions change sign.
* Against the feature means -- a common shortcut -- 19% of attributions
  change sign, though the base value moves by only 0.36.

Run: `standarderror run ml118_shapley --publish`
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

BGS = ("all training rows", "benign only", "malignant only", "feature means")


def compute() -> dict:
    return {"bg": rd.background_study(), "lin": rd.linear_shapley_check()}


def _example(bg: dict) -> int:
    ref = bg["phi"]["all training rows"]
    alt = bg["phi"]["benign only"]
    diff = np.argmax(np.abs(ref), 1) != np.argmax(np.abs(alt), 1)
    flips = (np.sign(ref) != np.sign(alt)).sum(1)
    score = diff * 10 + flips
    return int(np.argmax(score))


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    bg = res["bg"]
    feats = bg["features"]
    i = _example(bg)

    def one(ax, m):
        y = np.arange(len(feats))[::-1]
        h = 0.2
        for k, (c, name) in enumerate(zip(m.series, BGS)):
            ax.barh(y + (1.5 - k) * h, bg["phi"][name][i], h, color=c,
                    label=f"against {name}")
        ax.axvline(0, lw=1, color=m.ink)
        ax.set_yticks(y)
        ax.set_yticklabels(feats)
        ax.legend(frameon=False, fontsize=8.0, loc="lower right")

    out["f0"] = charts.diagram(
        one,
        title="One tumour, one model, four explanations",
        subtitle=("Exact Shapley values for one held-out tumour, in log-odds "
                  "of benign, measured against four background "
                  "distributions. The model and the prediction are the same "
                  "in every row."),
        xlabel="contribution to log-odds of benign",
        ylabel="",
        figsize=(7.6, 4.6),
        source="Breast Cancer Wisconsin (scikit-learn); "
               "standarderror/ml/reading.py.",
        alt=("Grouped horizontal bars for five features: the bars for the "
             "same feature differ in length and in some cases in sign across "
             "the four backgrounds."),
        caption=("Each set of four bars explains the same number. They "
                 "disagree because each answers a different question: why "
                 "this tumour differs from the average tumour, from a benign "
                 "one, from a malignant one, or from an average-valued one."),
        path=str(IMG / f"ml118-f0-one.{EXT}"))[0]

    cmp = bg["compare"]

    def rates(ax, m):
        names = BGS[1:]
        x = np.arange(len(names))
        w = 0.36
        ax.bar(x - w / 2, [cmp[n]["top_differs"] for n in names], w,
               color=m.series[0], label="largest attribution is a different "
                                        "feature")
        ax.bar(x + w / 2, [cmp[n]["sign_flips"] for n in names], w,
               color=m.series[1], label="attribution changes sign")
        for xi, n in zip(x, names):
            ax.text(xi - w / 2, cmp[n]["top_differs"] + 0.01,
                    f"{cmp[n]['top_differs']:.0%}", ha="center", fontsize=9)
            ax.text(xi + w / 2, cmp[n]["sign_flips"] + 0.01,
                    f"{cmp[n]['sign_flips']:.0%}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"against {n}" for n in names])
        ax.set_ylim(0, 0.7)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    out["f1"] = charts.diagram(
        rates,
        title="Change the reference, change the story",
        subtitle=(f"For {bg['rows']} held-out tumours, how the explanation "
                  f"against each background differs from the one against all "
                  f"training tumours."),
        xlabel="",
        ylabel="share of tumours (or of attributions)",
        source="Breast Cancer Wisconsin (scikit-learn); "
               "standarderror/ml/reading.py.",
        alt=("Paired bars for three alternative backgrounds: the benign-only "
             "background changes the top feature for over half the tumours; "
             "all three flip the sign of 15 to 19% of attributions."),
        caption=(f"Against benign tumours only, the most important feature "
                 f"changes for **{cmp['benign only']['top_differs']:.0%}** of "
                 f"tumours. Even the feature-means shortcut flips "
                 f"{cmp['feature means']['sign_flips']:.0%} of signs."),
        path=str(IMG / f"ml118-f1-rates.{EXT}"))[0]

    rows = []
    for n in BGS:
        c = cmp[n]
        rows.append([n, f"{c['base']:+.2f}",
                     *[f"{v:.2f}" for v in c["mean_abs"]],
                     f"{c['top_differs']:.0%}" if n != BGS[0] else "—"])
    out["f2"] = charts.table_image(
        rows,
        header=["background", "base value", *[f.replace("worst ", "w. ")
                                               .replace("mean ", "m. ")
                                               for f in feats],
                "top feature changes"],
        title="What each background is",
        subtitle=("Base value: the model's mean log-odds of benign over the "
                  "background. Feature columns: mean absolute Shapley value "
                  "over the test tumours."),
        source="Breast Cancer Wisconsin (scikit-learn); "
               "standarderror/ml/reading.py.",
        alt=(f"A four-row table of base values and mean absolute "
             f"attributions: base values range from "
             f"{cmp['malignant only']['base']:+.1f} to "
             f"{cmp['benign only']['base']:+.1f}."),
        caption=(f"The base value is the question's 'compared with'. A "
                 f"background of benign tumours puts it at "
                 f"{cmp['benign only']['base']:+.1f}, so every explanation "
                 f"becomes an account of why a tumour is less benign than "
                 f"those. The ranking by mean size is the same in every row."),
        align="lrrrrrrr",
        path=str(IMG / f"ml118-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res, i)
    return out


def _hero(res: dict, i: int):
    bg = res["bg"]
    cmp = bg["compare"]

    def a(panel, m):
        panel.barh(range(5), bg["phi"]["all training rows"][i][::-1],
                   color=m.series[0])

    def b(panel, m):
        panel.barh(range(5), bg["phi"]["benign only"][i][::-1],
                   color=m.series[1])

    def c(panel, m):
        panel.bar([0, 1, 2], [cmp[n]["sign_flips"] for n in BGS[1:]],
                  color=m.series[2])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=18,
        headline="Same model, same tumour, different explanation",
        panels=[(a, "average tumour", "compared with"),
                (b, f"{cmp['benign only']['top_differs']:.0%}",
                 "top feature changes"),
                (c, f"{cmp['feature means']['sign_flips']:.0%}",
                 "signs flip, vs means")],
        note=("Shapley values are the unique fair split of a prediction, "
              "relative to a reference distribution. The theorem does not "
              "pick the reference. Computed exactly, swapping it changes "
              "which feature leads for over half the tumours."),
        alt=("Three hand-drawn frames: one set of horizontal bars; a "
             "differently shaped set; three short bars of similar height."),
        path=str(IMG / f"ml118-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["exact"] = s.run("""
        import inspect
        from standarderror.ml import reading as rd

        # interventional Shapley values by enumerating all 2^p coalitions
        src = inspect.getsource(rd.exact_shapley).split("\\n")
        print("\\n".join(src[6:22]))
        """)
    out["check"] = s.run("""
        print(f"linear model, exact values vs beta*(x - mean): "
              f"max gap {rd.linear_shapley_check():.1e}")
        bg = rd.background_study()
        print(f"boosted model, 57 tumours x 4 backgrounds: "
              f"max |sum(phi) - (f(x) - E f)| = {bg['efficiency_error']:.1e}")
        """)
    out["cmp"] = s.run("""
        for name, c in bg["compare"].items():
            print(f"{name:18s} base {c['base']:+6.2f}  top feature changes "
                  f"{c['top_differs']:4.0%}  signs flip {c['sign_flips']:4.0%}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    bg = res["bg"]
    cmp = bg["compare"]
    ben, mal, mea = (cmp["benign only"], cmp["malignant only"],
                     cmp["feature means"])
    ref = cmp["all training rows"]
    # how far the global summary (mean |phi|) moves across backgrounds
    rel = max(float(np.max(np.abs(c["mean_abs"] / ref["mean_abs"] - 1)))
              for c in cmp.values())

    # the correlation behind "tumours that do not exist", on all 569 tumours
    from sklearn.datasets import load_breast_cancer
    bcd = load_breast_cancer()
    fn = [str(n) for n in bcd.feature_names]
    r_rc = float(np.corrcoef(bcd.data[:, fn.index("worst radius")],
                             bcd.data[:, fn.index("worst concave points")])[0, 1])

    # The spine, asserted rather than trusted.
    assert 0.3 < rel < 0.7 and 0.7 < r_rc < 0.9
    assert bg["efficiency_error"] < 1e-9 and res["lin"] < 1e-9
    assert ben["top_differs"] > 0.5 and mal["top_differs"] > 0.2
    assert all(c["sign_flips"] > 0.1 for c in (ben, mal, mea))
    assert abs(mea["base"] - ref["base"]) < 0.5 and mea["sign_flips"] > 0.15
    assert ben["base"] > 6 and mal["base"] < -6
    order = np.argsort(-ref["mean_abs"])
    assert all((np.argsort(-c["mean_abs"]) == order).all() for c in cmp.values())

    post = Post(
        title=(f"{SERIES_TAG} 18: A SHAP Value Is an Answer to \"Compared "
               f"With What?\""),
        slug="ml-18-shapley",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=18,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Shapley values split a prediction among its features "
                  "fairly and uniquely — relative to a reference that the "
                  "theory leaves to you. Computed exactly, the choice "
                  "changes the story for most predictions."),
        summary=(
            f"Exact Shapley values, by enumerating all 32 coalitions, for a "
            f"five-feature gradient-boosted model of the breast-cancer data. "
            f"Against all training tumours as background, they add up to "
            f"each prediction minus the mean to within "
            f"{bg['efficiency_error']:.0e}. Change the background to benign "
            f"tumours only and the feature with the largest attribution "
            f"changes for {ben['top_differs']:.0%} of the {bg['rows']} test "
            f"tumours, and {ben['sign_flips']:.0%} of attributions change "
            f"sign. Against the feature means, a common shortcut whose base "
            f"value is almost the same, {mea['sign_flips']:.0%} of "
            f"attributions change sign."),
        tags=["machine-learning", "data-science", "statistics",
              "interpretability", "shap", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Breast Cancer Wisconsin (Diagnostic), as bundled with "
            "scikit-learn, CC BY 4.0: five features (worst radius, worst "
            "concave points, worst texture, mean smoothness, worst "
            "symmetry); one stratified 70/30 split; every third test tumour "
            "explained.",
            "Machinery: `standarderror/ml/reading.py` (exact interventional "
            "Shapley values by enumeration; no SHAP library), tested in "
            "`tests/test_ml.py`; scikit-learn's "
            "HistGradientBoostingClassifier, explained in log-odds.",
            "Where this stops: Shapley, \"A value for n-person games\" "
            "(1953); Lundberg and Lee, \"A unified approach to interpreting "
            "model predictions\", *NeurIPS* (2017); Janzing, Minorics and "
            "Bloebaum, \"Feature relevance quantification in explainable AI: "
            "a causal problem\", *AISTATS* (2020); Sundararajan and Najmi, "
            "\"The many Shapley values for model explanation\", *ICML* "
            "(2020).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("split seed 0; boosting with random_state 0; "
                            "Shapley values are computed exactly, with no "
                            "sampling"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A unique answer to an unasked question",
        f"""Episodes 16 and 17 dealt with importance and effects averaged over a dataset. SHAP values work on one prediction at a time: they split the gap between this prediction and a baseline into one contribution per feature. Their appeal is a theorem. Shapley showed in 1953 that there is exactly one way to share a total among players that is efficient, symmetric, ignores players who contribute nothing and adds up across games. Lundberg and Lee brought it to model explanation in 2017, and "the unique fair attribution" has been its selling point since.

The theorem is about a game, and a model is not one until someone defines what a coalition of features is worth. The usual definition, the *interventional* one, is: fix the coalition's features at this row's values, draw the rest from a background dataset, and average the model's predictions. The theorem then gives a unique answer — for that background. The background is the question's "compared with what?", and the theorem has nothing to say about it.

With five features there are only 32 coalitions, so the values can be computed exactly, with no sampling or approximation:

{snip['exact'].markdown()}

Two checks that it is right. For a linear model the answer is known in closed form, coefficient times distance from the background mean. And for any model, the values must add up to the prediction minus the background's mean prediction:

{snip['check'].markdown()}""",
        figures=[figs["f0"]])

    post.add(
        "Four backgrounds, four explanations",
        f"""The model is gradient boosting on five breast-cancer features, explained in log-odds of benign. Four backgrounds, each defensible: all training tumours (the default in most tutorials); benign tumours only (*why is this one less benign than a typical benign one?*); malignant only; and a single row of feature means, a shortcut many implementations offer because it is fast.

{snip['cmp'].markdown()}

Against benign tumours only, the feature with the largest attribution is a different feature for **{ben['top_differs']:.0%}** of the tumours explained, and {ben['sign_flips']:.0%} of all individual attributions change sign. Against malignant tumours, {mal['top_differs']:.0%} and {mal['sign_flips']:.0%}. (Episode 66 reran this comparison on five other train-test splits: the benign background changed the top feature for 33% to 53% of tumours, so "most" belongs to this split; "a large share" is what survived.)

The feature-means background is the most instructive. Its base value, {mea['base']:+.2f}, is close to the all-rows one, {ref['base']:+.2f}, so it looks like a harmless approximation. It flips the sign of **{mea['sign_flips']:.0%}** of attributions — more than either class-specific background. A boosted model is far from linear, and the model's prediction *at* the average tumour is not the average of its predictions *over* tumours. Every coalition is evaluated with the missing features fixed at one artificial point instead of spread across real ones.

None of these explanations is wrong. Each is the unique fair split of a different difference. A clinician asking why a tumour was flagged probably means "compared with tumours we would not flag", which is closest to the benign background, and not to the population average that the default answers.""",
        figures=[figs["f1"]])

    post.add(
        "Off the data again",
        f"""There is a second choice hidden in the interventional definition. Filling in missing features by drawing them from the background *independently* of the features in the coalition builds rows that combine this tumour's worst radius with another tumour's worst concave points. In this data those two features are correlated at {r_rc:.2f}, so many of those rows are tumours that do not exist, the same problem episodes 16 and 17 met in permutation importance and partial dependence.

The alternative, *conditional* Shapley values, draws the missing features from their distribution given the coalition's values, keeping the rows realistic. It changes the answers again, and it has a cost beyond computation: under the conditional definition a feature the model never uses can receive credit, because it is correlated with one the model does use. Janzing and colleagues argue that the interventional version is the right one for asking what the *model* does; Sundararajan and Najmi catalogue how the variants disagree. The choice between them is a second "compared with what?" that the uniqueness theorem leaves open.

What the theorem does guarantee is useful: given a model, a value function and a background, the split is unique and adds up. The table below shows how far apart the backgrounds put the starting point. The skill is in stating the question, and the explanation is only as meaningful as the stated reference.""",
        figures=[figs["f2"]])

    post.add(
        "Choosing the background on purpose",
        f"""If the background is a choice, it should be made like one. Three questions decide it.

*Who is asking, and compared with whom?* A patient asking why their tumour was flagged is comparing it with tumours that were not; a regulator asking why the model behaves as it does across a population is comparing with the population. The first calls for a background of unflagged cases, the second for the whole training set. Here those two choices disagreed on the leading feature for {ben['top_differs']:.0%} of tumours.

*Is the background made of real rows?* A single row of feature means is cheap and not a tumour anyone has; it flipped {mea['sign_flips']:.0%} of signs. A sample of real rows, even a few hundred, costs more and answers a question about real cases.

*Is the background stated?* An explanation without its reference is like an effect size without its baseline. A sentence of the form "this feature added so much to the log-odds, relative to the average benign tumour" can be checked and argued with; "this feature was the most important" cannot.

A useful habit is to show two backgrounds side by side whenever an explanation matters. Where they agree, the explanation is robust to the choice; where they disagree, as for more than half the tumours here, the disagreement is itself the honest answer.""")

    post.add(
        "Where this breaks",
        f"""**Five features is a toy for enumeration.** Exact computation costs 2^p model evaluations per background row. Real explainers sample coalitions or exploit tree structure; their answers approximate the same quantities and inherit the same dependence on the background.

**Log-odds, not probabilities.** Attributions computed on predicted probabilities instead of log-odds are different numbers, and libraries differ in which they report. Everything here is in log-odds; the dependence on the background is part of the definition on either scale, but its size was measured on this one only.

**One model, one split.** A different model fitted to the same data could rank the features differently for the same tumour even with the background fixed. That was not measured here; episode 20 shows that models of near-equal accuracy can disagree on the predictions themselves, which is a reason to expect it.

**Global summaries average the problem away.** Mean absolute Shapley values, the usual "SHAP importance" bar chart, rank the five features in the same order under all four backgrounds, while their sizes move by up to {rel:.0%} and the leading feature of individual explanations changes for up to {ben['top_differs']:.0%} of tumours. That stability of the ranking is real and says nothing about any one prediction.""")

    post.add(
        "What to keep",
        f"""1. Shapley values are a unique fair split of the difference between a prediction and a reference. The reference is a choice, and the theorem does not make it.

2. Computed exactly, the values added up to within {bg['efficiency_error']:.0e}, and switching the background from all tumours to benign ones changed the top feature for {ben['top_differs']:.0%} of tumours.

3. The feature-means shortcut kept the base value almost unchanged and still flipped {mea['sign_flips']:.0%} of attribution signs.

4. Interventional and conditional definitions answer different questions; with correlated features they disagree too.

5. State the background in every explanation you show: "compared with the average benign tumour, this one's concavity added …".""")

    post.add(
        "Exercise",
        """Take one prediction your model made that someone questioned. Compute its SHAP values against your default background, then against a background of rows from the class or group the person had in mind as "normal". Compare the top three features in each.

If they differ, the explanation you gave answered a question the person did not ask. Write down, in one sentence, what the reference was for the explanation you would now give.""")

    post.add(
        "Next",
        """Episode 19 asks whether any of this matters when the model is simply accurate: a shortcut is planted in the digits, and a model that uses it scores better on the test set than one that does not.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
