"""ML 20: Same Accuracy, Different Answers.

The twentieth episode of *Machine Learning, Taught Through What Breaks*,
closing Arc IV. "Two models with the same accuracy make the same
predictions." Measured on a hard simulated task (Bayes error 0.276) and on
the breast-cancer data:

* Five neural networks differing only in their random seed score between
  0.670 and 0.680 on 5,000 test rows. Two of them disagree on 22.5% of the
  rows, and 43% of rows get both labels from some seed.
* QDA and an SVM, 0.716 and 0.714, disagree on 9.4%.
* On breast cancer, where error is about 2%, the models within one point of
  the best disagree on 1.4% of tumours.
* Disagreement is bounded by the sum of the two error rates, not by the
  difference in accuracy. Accuracy can only promise agreement when it is
  high.

Run: `standarderror run ml120_rashomon --publish`
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
    return {"g": rd.rashomon(task="gaussian"),
            "bc": rd.rashomon(task="breast cancer")}


def _groups(res: dict) -> list[dict]:
    g, bc = res["g"], res["bc"]
    out = []
    for task, r, tag in (("hard task", g, "Bayes error 0.28"),
                         ("breast cancer", bc, "error about 0.02")):
        for name, st in (("network seeds", r["families"]["MLP seed"]),
                         ("forest seeds", r["families"]["forest seed"]),
                         ("all within 1 point of best", r["good"])):
            acc = st["accuracy"]
            errs = [1 - a for a in acc]
            out.append({"task": task, "tag": tag, "group": name,
                        "n": len(acc), "acc_lo": min(acc), "acc_hi": max(acc),
                        "ambiguity": st["ambiguity"],
                        "max_pair": st["max_pair"],
                        "bound": max(p[3] for p in st["pairs"]),
                        "min_bound": max(abs(a - b) for a in errs
                                         for b in errs),
                        "prob_median": st["prob_range_median"],
                        "prob_p90": st["prob_range_p90"]})
    return out


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    G = _groups(res)

    def bars(ax, m):
        labels = [f"{r['task']}\n{r['group']}" for r in G]
        y = np.arange(len(G))[::-1]
        h = 0.36
        ax.barh(y + h / 2, [r["acc_hi"] - r["acc_lo"] for r in G], h,
                color=m.series[2], label="spread in accuracy")
        ax.barh(y - h / 2, [r["max_pair"] for r in G], h, color=m.series[1],
                label="largest disagreement between two models")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    mlp = G[0]
    out["f0"] = charts.diagram(
        bars,
        title="A point of accuracy, a fifth of the answers",
        subtitle=("For groups of models with near-equal test accuracy: the "
                  "range of their accuracies, and the largest share of test "
                  "rows on which two of them disagree."),
        xlabel="share of test rows",
        ylabel="",
        figsize=(7.6, 4.8),
        source="Simulated (episode 4's task) and Breast Cancer Wisconsin "
               "(scikit-learn); standarderror/ml/reading.py.",
        alt=("Paired horizontal bars for six groups: on the hard task, the "
             "disagreement bars dwarf the accuracy-spread bars; on breast "
             "cancer both are small."),
        caption=(f"Five networks differing only in their seed span "
                 f"{mlp['acc_hi'] - mlp['acc_lo']:.3f} in accuracy and "
                 f"disagree on up to **{mlp['max_pair']:.1%}** of rows."),
        path=str(IMG / f"ml120-f0-bars.{EXT}"))[0]

    q0, q1 = res["g"]["q_pair"]

    def scatter(ax, m):
        ax.scatter(q0[:2000], q1[:2000], s=5, alpha=0.4, color=m.series[0])
        ax.plot([0, 1], [0, 1], lw=1, ls="--", color=m.ink)
        ax.axhline(0.5, lw=0.8, color=m.grid)
        ax.axvline(0.5, lw=0.8, color=m.grid)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    acc = res["g"]["accuracy"]
    a0, a1 = acc["MLP seed 0"], acc["MLP seed 1"]
    dis = float(np.mean((q0 > 0.5) != (q1 > 0.5)))
    out["f1"] = charts.diagram(
        scatter,
        title="Two seeds, one architecture, two risk scores",
        subtitle=(f"Predicted probability of class 1 from two networks "
                  f"identical except for their random seed, for 2,000 of the "
                  f"test rows. Accuracies {a0:.3f} and {a1:.3f}."),
        xlabel="network, seed 0",
        ylabel="network, seed 1",
        equal=True,
        figsize=(6.0, 6.0),
        source="Simulated; standarderror/ml/reading.py.",
        alt=("A wide cloud of points around the diagonal, with many points "
             "in the off-diagonal quadrants where one network says above "
             "0.5 and the other below."),
        caption=(f"The off-diagonal quadrants hold the {dis:.1%} of rows on "
                 f"which the two disagree. Many rows the first network puts "
                 f"near 0.2 the second puts near 0.8."),
        path=str(IMG / f"ml120-f1-scatter.{EXT}"))[0]

    rows = []
    for r in G:
        rows.append([r["task"], r["group"], f"{r['n']}",
                     f"{r['acc_lo']:.3f}–{r['acc_hi']:.3f}",
                     f"{r['max_pair']:.1%}", f"{r['bound']:.1%}",
                     f"{r['ambiguity']:.1%}", f"{r['prob_median']:.2f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["data", "models", "n", "test accuracy", "largest pairwise "
                "disagreement", "bound (e1 + e2)", "rows with both labels",
                "median spread in probability"],
        title="What equal accuracy promises",
        subtitle=("Disagreement between two models can be as large as the "
                  "sum of their error rates. 5,000 test rows for the hard "
                  "task, 285 for breast cancer."),
        source="Simulated and Breast Cancer Wisconsin (scikit-learn); "
               "standarderror/ml/reading.py.",
        alt=("A six-row table: on the hard task, disagreement of 5 to 23% "
             "and up to 43% of rows receiving both labels; on breast cancer, "
             "under 3% throughout."),
        caption=("On the hard task the networks use a third of the room the "
                 "bound allows. On breast cancer there is almost no room."),
        bold_cells={(0, 4), (0, 6)},
        align="llrrrrrr",
        path=str(IMG / f"ml120-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res, G)
    return out


def _hero(res: dict, G):
    q0, q1 = res["g"]["q_pair"]
    mlp, bcg = G[0], G[5]

    def acc(panel, m):
        a = res["g"]["families"]["MLP seed"]["accuracy"]
        panel.bar(range(len(a)), [v - 0.6 for v in a], color=m.series[2])
        panel.set_ylim(0, 0.12)

    def sc(panel, m):
        panel.scatter(q0[:600], q1[:600], s=4, color=m.series[0])

    def easy(panel, m):
        panel.bar([0, 1], [mlp["max_pair"], bcg["max_pair"]],
                  color=[m.series[1], m.series[3]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=20,
        headline="Equal accuracy, different answers",
        panels=[(acc, f"{mlp['acc_hi'] - mlp['acc_lo']:.3f}",
                 "accuracy spread, 5 seeds"),
                (sc, f"{mlp['max_pair']:.1%}", "rows two seeds dispute"),
                (easy, f"{bcg['max_pair']:.1%}", "on an easy task")],
        note=("Five networks that differ only in their random seed score "
              "within a point of each other on a hard task and give "
              "different answers on a fifth of the rows. Accuracy limits "
              "how often models can disagree only when errors are rare."),
        alt=("Three hand-drawn frames: five bars of equal height; a broad "
             "cloud of points; a tall bar beside a tiny one."),
        path=str(IMG / f"ml120-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["hard"] = s.run("""
        from standarderror.ml import reading as rd

        r = rd.rashomon(task="gaussian")     # episode 4's task, 2,000 rows
        f = r["families"]["MLP seed"]         # same network, seeds 0-4
        print("accuracy by seed: " + " ".join(f"{a:.3f}" for a in f["accuracy"]))
        print(f"largest disagreement between two seeds: {f['max_pair']:.1%}")
        print(f"rows given both labels by some seed:    {f['ambiguity']:.1%}")
        """)
    out["zoo"] = s.run("""
        for k, v in sorted(r["accuracy"].items(), key=lambda kv: -kv[1])[:4]:
            print(f"{k:16s} {v:.3f}")
        g = r["good"]
        print(f"within 1 point of best: {g['models']}, disagree on "
              f"{g['max_pair']:.1%}")
        """)
    out["easy"] = s.run("""
        b = rd.rashomon(task="breast cancer")
        g = b["good"]
        print(f"{len(g['models'])} models within 1 point of {b['best']:.3f}; "
              f"largest disagreement {g['max_pair']:.1%}, "
              f"rows with both labels {g['ambiguity']:.1%}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    G = _groups(res)
    mlp, rf, good, bmlp, brf, bgood = G
    g = res["g"]

    # The spine, asserted rather than trusted.
    assert mlp["acc_hi"] - mlp["acc_lo"] < 0.015
    assert mlp["max_pair"] > 10 * (mlp["acc_hi"] - mlp["acc_lo"])
    assert mlp["max_pair"] > 0.15 and mlp["ambiguity"] > 0.35
    assert good["n"] >= 2 and good["max_pair"] > 0.05
    assert all(r["max_pair"] <= r["bound"] + 1e-12 for r in G)
    assert all(r["max_pair"] >= r["min_bound"] - 1e-12 for r in G)
    assert bgood["max_pair"] < 0.03 and bgood["n"] >= 3
    assert rf["max_pair"] < mlp["max_pair"]
    assert set(good_names := res["g"]["good"]["models"]) == {"QDA", "SVM"}, \
        good_names

    post = Post(
        title=f"{SERIES_TAG} 20: Same Accuracy, Different Answers",
        slug="ml-20-rashomon",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=20,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Two models with the same test accuracy are usually "
                  "treated as interchangeable. On a hard task, networks that "
                  "differ only in their seed give different answers on a "
                  "fifth of the rows."),
        summary=(
            f"On a simulated task with Bayes error {g['bayes']:.3f}, five "
            f"neural networks that differ only in their random seed score "
            f"between {mlp['acc_lo']:.3f} and {mlp['acc_hi']:.3f} on 5,000 "
            f"test rows. Two of them disagree on {mlp['max_pair']:.1%} of "
            f"the rows, and {mlp['ambiguity']:.0%} of rows get both labels "
            f"from some seed. QDA and an SVM, the two best models, disagree "
            f"on {good['max_pair']:.1%}. On breast cancer, where errors are "
            f"rare, models within a point of the best disagree on "
            f"{bgood['max_pair']:.1%}. Two models can disagree on as many "
            f"rows as the sum of their error rates: equal accuracy promises "
            f"agreement only when both are nearly always right."),
        tags=["machine-learning", "data-science", "statistics",
              "model-selection", "predictive-multiplicity", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: episode 4's two-class Gaussian task in 10 "
            "dimensions with unequal covariances, Bayes error computed from "
            "the true densities; 2,000 training rows, 5,000 test rows.",
            "Breast Cancer Wisconsin (Diagnostic), as bundled with "
            "scikit-learn, CC BY 4.0; one stratified half split.",
            "Machinery: `standarderror/ml/reading.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's logistic regression, QDA, "
            "SVC, HistGradientBoosting, 25-nearest neighbours, random "
            "forests and MLPClassifier (one hidden layer of 64, five seeds "
            "each for forests and networks).",
            "Where this stops: Breiman, \"Statistical modeling: the two "
            "cultures\", *Statistical Science* (2001); Marx, Calmon and "
            "Ustun, \"Predictive multiplicity in classification\", *ICML* "
            "(2020); D'Amour et al., \"Underspecification presents "
            "challenges for credibility in modern machine learning\", *JMLR* "
            "(2022).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("training rows from seed 0, test rows from seed "
                            "99; forests and networks with seeds 0 to 4; "
                            "other models with random_state 0"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Interchangeable by the numbers",
        f"""Episode 19 showed a test score that could not tell a shortcut from the task. This episode asks something more modest: when two models have the same test score, are they the same model, for practical purposes? Model selection usually acts as if they were. Of two models within noise of each other, pick either.

Breiman called the phenomenon the *Rashomon effect*: many different models fit the same data equally well. Episode 11 met it in a single tree's structure. Here it is measured where it matters most, in the predictions.

The task is episode 4's: two overlapping Gaussian classes in ten dimensions, Bayes error {g['bayes']:.3f}, so even the best possible classifier is wrong on more than a quarter of rows. Train the same small neural network five times, changing nothing but the random seed:

{snip['hard'].markdown()}

The accuracies span **{mlp['acc_hi'] - mlp['acc_lo']:.3f}** — a single point. By any significance test they are the same model. Two of them disagree on **{mlp['max_pair']:.1%}** of the rows, and across the five, {mlp['ambiguity']:.0%} of test rows receive both labels from some seed. For more than two in five cases, which answer a person gets depends on a random number chosen at training time.""",
        figures=[figs["f0"]])

    post.add(
        "Why accuracy cannot see it",
        f"""Accuracy counts how many rows a model gets right, not which. Two models with error rates e₁ and e₂ can disagree on as few rows as |e₁ − e₂| — when one's mistakes are a subset of the other's — and on as many as e₁ + e₂, when their mistakes never overlap. Equal accuracy pins down only the first bound. The second is set by how often the models are wrong.

Here each network is wrong on about a third of the rows, so two of them could in principle disagree on two-thirds. They use a third of that room. The probabilities show how differently they see the same row. The scatter plot below puts one seed's predicted probability against another's. A row the first network calls 0.2 the second may call 0.8. Neither is miscalibrated on average; both are equally accurate; they have simply drawn different boundaries through a region where the data cannot decide between them.

Different model families do the same. Among all fifteen models, the best two are QDA and an SVM:

{snip['zoo'].markdown()}

They differ by {good['acc_hi'] - good['acc_lo']:.3f} in accuracy and disagree on {good['max_pair']:.1%} of rows. Forests trained with different seeds disagree less ({rf['max_pair']:.1%} at most) because averaging 300 trees damps the effect of the seed — the same averaging that stabilised episode 11's trees.""",
        figures=[figs["f1"]])

    post.add(
        "When errors are rare",
        f"""The bound also says when this does not matter. On the breast-cancer data, where good models are wrong on two or three tumours in a hundred:

{snip['easy'].markdown()}

There is no room to disagree. When both models are almost always right they must almost always agree, and here {bgood['n']} models within a point of the best disagree on at most {bgood['max_pair']:.1%} of tumours.

So "same accuracy, same predictions" is a statement about easy problems. On hard ones — credit default, readmission, churn, anything with a large irreducible error — models of equal accuracy are different models, and the choice between them is a choice about *who* gets which answer. That is the setting Marx, Calmon and Ustun call predictive multiplicity, and D'Amour and colleagues call underspecification: the training procedure and the test score jointly fail to pin down the predictions, and something arbitrary — a seed, an architecture, a library default — fills the gap.""",
        figures=[figs["f2"]])

    post.add(
        "What to do with a disputed row",
        f"""Disagreement among equally good models is information, not only a nuisance. A row on which five seeds split is a row on which the data do not determine the answer, and the people behind those rows deserve to know that their case is close.

Three practical responses follow. Measure it: train a handful of seeds or a handful of near-equal models and report the share of rows on which they disagree, alongside the accuracy. Here that share — {mlp['ambiguity']:.0%} of rows for five seeds — says far more about the reliability of an individual prediction than the accuracy's spread of {mlp['acc_hi'] - mlp['acc_lo']:.3f}. Average: an ensemble of the near-equal models gives one prediction that no longer depends on which seed was drawn, and its probabilities on the disputed rows move towards the middle, which is the truthful position. Escalate: for decisions about individuals, route the disputed rows to a slower process — a second model, a human review, a request for more information — instead of letting a random number decide.

What does not work is picking one model by a tie-break on the test score, or by whichever happened to be trained first. The models are tied within noise, and whichever one is picked, a different fifth of the people get the other answer — and nobody downstream can tell which fifth.""")

    post.add(
        "Where this breaks",
        """**Ensembling reduces it.** Averaging the five networks would give one set of predictions that changes less with any one seed, as the forests show. It does not choose between the boundaries; it blurs them, and the blurred probabilities for disputed rows sit near 0.5, which is honest.

**One hard task, simulated.** The networks are small and trained to convergence with light regularisation. Larger networks, more data or stronger regularisation all change the amount of disagreement; none removes the bound.

**Disagreement is not error.** On rows where two models disagree, one of them is right. The point is not that both are bad but that the test score cannot say which to trust on those rows, and the person affected cannot either.

**The 1-point tolerance is a choice.** With a tolerance of zero, no two models qualify; with five points, many more do and the disagreement grows. One point is roughly the noise in an accuracy measured on 5,000 rows.""")

    post.add(
        "What to keep",
        f"""1. Accuracy says how many rows a model gets right, not which. Two models can disagree on as many rows as the sum of their error rates.

2. On a task with Bayes error {g['bayes']:.2f}, five networks differing only in seed spanned {mlp['acc_hi'] - mlp['acc_lo']:.3f} in accuracy and disagreed on up to {mlp['max_pair']:.1%} of rows; {mlp['ambiguity']:.0%} of rows got both labels from some seed.

3. The two best model families, {good['acc_hi'] - good['acc_lo']:.3f} apart in accuracy, disagreed on {good['max_pair']:.1%}.

4. On breast cancer, with errors near 2%, near-equal models disagreed on {bgood['max_pair']:.1%}: rare errors leave no room.

5. On hard problems, report how much near-equal models disagree, and for decisions about individuals, consider averaging them or flagging the disputed rows.""")

    post.add(
        "Exercise",
        """Retrain your model five times with different seeds, changing nothing else, and score each on your test set. Then count the test rows on which at least two of the five disagree, and the largest disagreement between any pair.

If that count is more than a few percent, look at who those rows are. They are the cases where your model's answer is an accident of training, and they are often the cases where the decision matters most.""")

    post.add(
        "Next",
        """Arc IV ends here. Importance, effects, attributions, shortcuts and the multiplicity of equally good models all said the same thing from different directions: what a model learned is not pinned down by how well it scores. Arc V turns to training a network, starting with what happens before the first step: the claim that initialisation only needs to break symmetry.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
