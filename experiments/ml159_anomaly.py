"""ML 59: An Anomaly Score Ranks; the Contamination Setting Only Cuts.

The fifty-ninth episode of *Machine Learning, Taught Through What Breaks*,
in Arc XII. "An anomaly detector finds the anomalies in your data."
Measured with isolation forests and local outlier factors on simulations
with planted anomalies, and on breast-cancer data with malignant cases
hidden among benign ones:

* The contamination setting flagged exactly that share of the data whatever
  the true rate; the "auto" setting flagged 10.5% to 13.4% whether 0.5%,
  2% or 10% were anomalies. The ranking underneath had an AUC of 0.99 or
  more in every case.
* Fifty anomalies packed tightly together scored as normal or more than
  normal (AUC 0.17 to 0.58) under every local method whose neighbourhood was
  smaller than the group, and perfectly once it was larger.
* With anomalies different in 2 features out of 200, the isolation forest's
  AUC fell from 1.00 to 0.58 on average over six draws (0.50 to 0.68);
  given the two features alone, 0.997.
* Eight malignant tumours among 357 benign ones were ranked with an AUC of
  0.85 to 0.99 across 20 draws; flagging 5% caught 71% of them on average.

Run: `standarderror run ml159_anomaly --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import unsup
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 14)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

SETTINGS = ("0.01", "0.05", "0.1", "auto")


def compute() -> dict:
    return {"cont": unsup.anomaly_contamination(),
            "mask": unsup.anomaly_masking(),
            "dims": unsup.anomaly_dimensions(),
            "bc": unsup.anomaly_breast_cancer()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    body = []
    for r in res["cont"]:
        for c in SETTINGS:
            v = r[c]
            body.append([f"{r['rate']:.1%}", c, f"{v['flagged']:.1%}",
                         f"{v['precision']:.2f}", f"{v['recall']:.2f}",
                         f"{r['auc']:.3f}"])
    out["f0"] = charts.table_image(
        body,
        header=["true anomaly rate", "contamination", "flagged",
                "precision", "recall", "AUC of the score"],
        title="The setting decides how many are flagged",
        subtitle=("An isolation forest on 5,000 points in five dimensions, "
                  "anomalies shifted 2.5 standard deviations in every "
                  "coordinate. The AUC is of the score itself and does not "
                  "depend on the setting."),
        source="Simulated; standarderror/ml/unsup.py.",
        alt=("A twelve-row table. The flagged share equals the contamination "
             "setting in every row; 'auto' flags 10.5% to 13.4%. The AUC is "
             "0.99 or more throughout."),
        caption="Precision and recall are against the planted anomalies.",
        align="rlrrrr",
        path=str(IMG / f"ml159-f0-contamination.{EXT}"))[0]

    mk = res["mask"]

    def masking(ax, m):
        ks = [r["k"] for r in mk["rows"]]
        ax.plot(ks, [r["lof"] for r in mk["rows"]], marker="o",
                color=m.series[0], label="local outlier factor")
        ax.plot(ks, [r["knn"] for r in mk["rows"]], marker="s",
                color=m.series[1], label="distance to k-th neighbour")
        ax.axhline(mk["iforest"], color=m.series[2], ls="--",
                   label="isolation forest")
        ax.axvline(mk["group"], color=m.ink, ls=":", lw=1)
        ax.text(mk["group"] * 1.05, 0.05, "group size", fontsize=8)
        ax.axhline(0.5, color=m.grid, lw=1)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="center left", frameon=False)

    k20 = next(r for r in mk["rows"] if r["k"] == 20)
    out["f1"] = charts.diagram(
        masking,
        title="A crowd of anomalies looks normal from inside",
        subtitle=("2,000 normal points and a tight group of 50 anomalies "
                  "far away. AUC of each score for finding the group, as "
                  "the neighbourhood size k grows."),
        xlabel="neighbourhood size k (log scale)", ylabel="AUC",
        figsize=(7.4, 4.3),
        source="Simulated; standarderror/ml/unsup.py.",
        alt=("Two curves near or below 0.5 for k of 5 and 20; the distance "
             "score reaches 1 at k = 50 and the outlier factor at 100; a "
             "flat line at 1 for the isolation forest."),
        caption=(f"At k = 20 the distance score's AUC was "
                 f"**{k20['knn']:.2f}**: the anomalies looked more normal "
                 f"than the normal points."),
        path=str(IMG / f"ml159-f1-masking.{EXT}"))[0]

    body = []
    for d in res["dims"]:
        body.append([str(d["dim"]), f"{d['iforest']:.3f}", f"{d['lof']:.3f}",
                     f"{d['iforest_two']:.3f}"])
    out["f2"] = charts.table_image(
        body,
        header=["features", "isolation forest", "LOF (k = 20)",
                "isolation forest, the 2 relevant features"],
        title="Two features that matter, among many that do not",
        subtitle=("40 anomalies shifted by 3.5 in two features among 2,000 "
                  "normal points; every other feature is noise for both. AUC "
                  "for finding the anomalies."),
        source="Simulated; standarderror/ml/unsup.py.",
        alt=("A four-row table. The isolation forest falls from 1.00 to "
             "about 0.58 on average; LOF rises then falls; on the two relevant "
             "features alone "
             "the forest stays near 1."),
        caption=("Means over six draws of data and forest; with 40 anomalies, "
                 "an AUC near 0.5 has a standard error of about 0.05."),
        align="rrrr",
        path=str(IMG / f"ml159-f2-dims.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    cont, mk, dims = res["cont"], res["mask"], res["dims"]

    def p1(panel, m):
        panel.bar([0, 1, 2], [r["auto"]["flagged"] for r in cont],
                  color=[m.series[0], m.series[1], m.series[2]])

    def p2(panel, m):
        panel.plot([r["k"] for r in mk["rows"]], [r["knn"] for r in mk["rows"]],
                   color=m.series[1], lw=2.5, marker="o")
        panel.set_xscale("log")
        panel.set_ylim(0, 1.05)

    def p3(panel, m):
        panel.plot([d["dim"] for d in dims], [d["iforest"] for d in dims],
                   color=m.series[0], lw=2.5, marker="o")
        panel.set_xscale("log")
        panel.set_ylim(0, 1.05)

    k20 = next(r for r in mk["rows"] if r["k"] == 20)
    return charts.lecture_hero(
        series=SERIES_TAG, episode=59,
        headline="Ranks, then a cut",
        panels=[(p1, f"{min(r['auto']['flagged'] for r in cont):.1%}+",
                 "flagged by 'auto'"),
                (p2, f"{k20['knn']:.2f}", "AUC, grouped anomalies"),
                (p3, f"{dims[-1]['iforest']:.2f}", "mean AUC, 2 of 200")],
        note=("An anomaly detector ranks points by how unusual they look in "
              "the features and neighbourhood it was given. How many to "
              "flag is your decision, and what counts as unusual is its "
              "assumption."),
        alt=("Three hand-drawn frames: three bars of similar height; a line "
             "that starts low and jumps; a line that falls."),
        path=str(IMG / f"ml159-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["cont"] = s.run("""
        from standarderror.ml import unsup

        for r in unsup.anomaly_contamination(rates=(0.005, 0.1),
                                             settings=(0.05, "auto")):
            for c in ("0.05", "auto"):
                v = r[c]
                print(f"true {r['rate']:5.1%}  contamination={c:4s}  flagged "
                      f"{v['flagged']:.3f}  precision {v['precision']:.2f}")
        """)
    out["mask"] = s.run("""
        m = unsup.anomaly_masking(ks=(20, 100))      # a group of 50 anomalies
        for r in m["rows"]:
            print(f"k={r['k']:3d}  LOF AUC {r['lof']:.2f}  k-NN distance AUC "
                  f"{r['knn']:.2f}")
        print(f"isolation forest AUC {m['iforest']:.2f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    cont, mk, dims, bc = res["cont"], res["mask"], res["dims"], res["bc"]
    auto = [r["auto"]["flagged"] for r in cont]
    small = [r for r in mk["rows"] if r["k"] < mk["group"]]
    big = [r for r in mk["rows"] if r["k"] > mk["group"]]
    lo = {r["rate"]: r for r in cont}

    # The spine, asserted rather than trusted.
    assert all(abs(r[c]["flagged"] - float(c)) < 1e-3 for r in cont
               for c in SETTINGS[:3])
    assert 0.10 < min(auto) and max(auto) < 0.14
    assert all(r["auc"] > 0.99 for r in cont)
    assert lo[0.005]["0.05"]["precision"] < 0.11
    assert lo[0.1]["0.01"]["recall"] < 0.11 and lo[0.1]["0.01"]["precision"] == 1
    assert all(r["lof"] < 0.6 for r in small) and all(r["knn"] < 0.2
                                                      for r in small)
    assert all(r["lof"] == 1 and r["knn"] == 1 for r in big)
    assert mk["iforest"] > 0.99
    assert dims[0]["iforest"] > 0.99 and dims[-1]["iforest"] < 0.65
    assert dims[-1]["iforest_min"] < 0.52 and dims[-1]["iforest_max"] < 0.7
    assert dims[0]["lof"] < dims[2]["lof"] > dims[3]["lof"]
    assert all(d["iforest_two"] > 0.99 for d in dims)
    assert dims[-1]["lof"] > dims[-1]["iforest"] + 0.1
    assert 0.9 < bc["iforest_mean"] and 0.85 < bc["iforest_min"] + 0.01
    assert 0.65 < bc["recall_5pct_mean"] < 0.75 and bc["recall_5pct_min"] < 0.4

    post = Post(
        title=(f"{SERIES_TAG} 59: An Anomaly Score Ranks; the Contamination "
               f"Setting Only Cuts"),
        slug="ml-59-anomaly",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=59,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("An isolation forest ranked planted anomalies almost "
                  "perfectly and then flagged whatever share it was told to. "
                  "Grouped anomalies and irrelevant features changed the "
                  "ranking itself."),
        summary=(
            f"An isolation forest on simulated data with 0.5%, 2% or 10% "
            f"planted anomalies ranked them with AUCs of 0.99 or more, and "
            f"flagged exactly the share set by its contamination parameter; "
            f"the 'auto' setting flagged {min(auto):.1%} to {max(auto):.1%} "
            f"whatever the truth. Fifty anomalies packed together scored "
            f"AUCs of {min(r['knn'] for r in small):.2f} to "
            f"{max(r['lof'] for r in small):.2f} under local methods with "
            f"neighbourhoods smaller than the group, and 1.00 with larger "
            f"ones. Anomalous in 2 features of 200, they were found with an "
            f"average AUC of {dims[-1]['iforest']:.2f}, against "
            f"{dims[-1]['iforest_two']:.3f} on the two features alone. "
            f"Eight malignant tumours among 357 benign ones were ranked with "
            f"AUCs of {bc['iforest_min']:.2f} to {bc['iforest_max']:.2f}; "
            f"flagging 5% caught {bc['recall_5pct_mean']:.0%} of them on "
            f"average."),
        tags=["machine-learning", "unsupervised-learning",
              "anomaly-detection", "isolation-forest", "outliers",
              "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated (seed 0): standard Gaussian normal data; anomalies "
            "shifted by 2.5 in every coordinate with random signs (5,000 "
            "points, 5 dimensions), packed in a group of 50 with standard "
            "deviation 0.1 at (5, 5) beside 2,000 normal points, or shifted "
            "by 3.5 in two of 2 to 200 dimensions (40 among 2,000).",
            "Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, via "
            "scikit-learn: the 357 benign tumours with 8 malignant ones "
            "drawn at random (20 draws), features standardised.",
            "Machinery: scikit-learn's IsolationForest (100 trees) and "
            "LocalOutlierFactor; `standarderror/ml/unsup.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Liu, Ting and Zhou, \"Isolation forest\", "
            "*ICDM* (2008); Breunig, Kriegel, Ng and Sander, \"LOF: "
            "identifying density-based local outliers\", *SIGMOD* (2000); "
            "Zimek, Schubert and Kriegel, \"A survey on unsupervised outlier "
            "detection in high-dimensional numerical data\", *Statistical "
            "Analysis and Data Mining* (2012); episodes 33, 46 and 58 of "
            "this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; about a minute of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Finding what you did not know to look for",
        """Anomaly detection is the unsupervised task with the most practical pull: fraud nobody has labelled yet, a failing machine, a data-entry error among a million rows. With no labels there is no model of the anomaly, only of normality, and a point is flagged for not fitting it. The isolation forest of Liu, Ting and Zhou isolates points with random splits and scores those that take few splits to isolate; the local outlier factor of Breunig and colleagues compares a point's local density with its neighbours'. Both are a line of scikit-learn, and the output is a flag on each row, which invites the sentence under test: *an anomaly detector finds the anomalies in your data*.

The test plants anomalies whose number and nature are known, and asks three questions separately: how many points get flagged, how well the score ranks the planted ones, and what in the data changes that ranking. The last question is replicated on breast-cancer data, with malignant tumours hidden among benign ones as the anomalies to find.""")

    post.add(
        "The setting decides how many",
        f"""scikit-learn's detectors take a `contamination` parameter, and it reads like an estimate of how much of the data is anomalous. It is not. It sets the threshold so that exactly that share of the training points is flagged.

{snip['cont'].markdown()}

With 0.5% of the points truly anomalous, a setting of 5% flagged 5% and gave a precision of {lo[0.005]['0.05']['precision']:.2f}: nine flags in ten were normal points. With 10% truly anomalous, a setting of 1% gave perfect precision and caught {lo[0.1]['0.01']['recall']:.0%} of them. The default, 'auto', applies a fixed threshold to the isolation forest's score, and here it flagged {min(auto):.1%} to {max(auto):.1%} of the points whether the truth was 0.5%, 2% or 10%. Underneath every one of those settings the score ranked the planted anomalies with an AUC of 0.99 or more.

So the detector's real output is a ranking, and the flag is a cut through it that someone chose. Episode 46 made the same point for classifiers: the threshold is a decision about costs, not a property of the model, and here there is not even a probability to anchor it. Choosing it needs either a labelled sample or a budget, the number of cases someone can investigate.""",
        figures=[figs["f0"]])

    post.add(
        "A crowd of anomalies looks normal from inside",
        f"""The ranking depends on what the score compares a point with. Fifty anomalies, all from one fault, packed tightly together far from 2,000 normal points: a local method that looks at a point's k nearest neighbours sees, for any k up to 49, only other anomalies, densely packed.

{snip['mask'].markdown()}

With k of 5 or 20, the local outlier factor scored the group with an AUC between {min(r['lof'] for r in small):.2f} and {max(r['lof'] for r in small):.2f}, barely better than a coin, and the plain distance to the k-th neighbour between {min(r['knn'] for r in small):.2f} and {max(r['knn'] for r in small):.2f}: by that score the anomalies looked *more* normal than the normal points, because they were more tightly packed. At k = 50 the distance score found them perfectly and the outlier factor still did not ({mk_rows(mk)[50]['lof']:.2f}); at k = 100 both did. The isolation forest, which subsamples and isolates rather than comparing neighbours, scored {mk['iforest']:.2f}. This is masking, an old problem in outlier statistics: one anomaly is conspicuous, many similar ones are a cluster. A fault that produces many identical bad records is exactly that case.""",
        figures=[figs["f1"]])

    post.add(
        "Two features that matter among many that do not",
        f"""An anomaly is usually unusual in a few features, not all of them. Here 40 anomalies were shifted by 3.5 standard deviations in two features, and the data had 2, 10, 50 or 200 features in all. Averaged over six draws of the data and the forest, the isolation forest's AUC went from {dims[0]['iforest']:.2f} to {dims[1]['iforest']:.2f}, {dims[2]['iforest']:.2f} and **{dims[3]['iforest']:.2f}**, with single draws at 200 features between {dims[3]['iforest_min']:.2f} and {dims[3]['iforest_max']:.2f}: little or no better than chance. Each of its random splits chooses a feature at random, so with two relevant features among 200 almost every split is spent on noise; the control below supports that explanation without measuring it directly. Given only the two relevant features, it scored {dims[-1]['iforest_two']:.3f}.

The local outlier factor behaved differently, rising from {dims[0]['lof']:.2f} with two features to {dims[2]['lof']:.2f} with 50 and falling to {dims[3]['lof']:.2f} with 200. The fall is the dilution episode 33 measured for nearest neighbours: distances in many dimensions still carry the two shifted features, but less and less. The low start is probably the masking of the previous section: in two dimensions the 40 anomalies form a little cluster of their own, and LOF looks at 20 neighbours. Extra noise features spread them apart. Zimek, Schubert and Kriegel's survey of outlier detection in high dimensions makes the general case: irrelevant attributes hide anomalies that exist in a subspace, and which method suffers most depends on how it uses distances. The practical lesson is older than either method: the detector can only find what the features make unusual, so the choice of features is most of the model.""",
        figures=[figs["f2"]])

    post.add(
        "Malignant among benign",
        f"""The replication hides eight malignant tumours among the 357 benign ones in the breast-cancer data, with all 30 features standardised, and repeats the draw 20 times. The isolation forest ranked the eight with an AUC of {bc['iforest_mean']:.2f} on average, between {bc['iforest_min']:.2f} and {bc['iforest_max']:.2f}; the local outlier factor, {bc['lof_mean']:.2f}. Malignant tumours are larger and more irregular, so in these features they are unusual, and an unsupervised score finds most of them.

Flagging 5% of the 365 tumours, about eighteen, caught {bc['recall_5pct_mean']:.0%} of the malignant ones on average, but between {bc['recall_5pct_min']:.0%} and {bc['recall_5pct_max']:.0%} depending on which eight were drawn and on the forest's seed, with a precision of {bc['precision_5pct_mean']:.2f}. That spread is the cost of having no labels: which malignant cases look unusual, and which benign ones do, decides the result, and nothing in the procedure says which it will be on a given day.""")

    post.add(
        "Where this breaks",
        """**Planted anomalies.** The simulated anomalies were shifted versions of normal points, the easiest kind to define. Real anomalies include points that are individually ordinary and unusual only in combination or in sequence, which these detectors do not look for.

**Scores on the training data.** Every score here was computed on the data the detector was fitted to, which is how the tools are commonly used. Fitting on clean data and scoring new data would presumably remove some masking, since the group would not be among the neighbours, and none of the feature problem; neither was measured.

**Two detectors, default settings.** Other methods, such as one-class support vector machines or autoencoders, use distances or reconstructions differently and fail differently; the masking and irrelevant-feature effects were measured for these two.

**Breast-cancer features are good ones.** Its 30 features were chosen by people who knew what malignancy looks like. Features chosen without that knowledge would be closer to the 200-feature simulation.""")

    post.add(
        "What to keep",
        f"""1. The contamination parameter flagged exactly its own value; 'auto' flagged {min(auto):.1%} to {max(auto):.1%} whatever the true rate. The ranking underneath scored an AUC of 0.99 or more throughout.

2. Fifty grouped anomalies looked normal to local scores until the neighbourhood was larger than the group; at k = 20 the distance score's AUC was {k20_knn(mk):.2f}.

3. Anomalous in 2 features of 200, the isolation forest found them with an average AUC of {dims[-1]['iforest']:.2f}; on the two features, {dims[-1]['iforest_two']:.3f}.

4. Treat an anomaly score as a ranking under assumptions about features and neighbourhoods. Choose how many to flag from a budget or a labelled sample, and check what the top of the ranking actually contains.""")

    post.add(
        "Exercise",
        """Take the rows your detector flagged most recently and the next 50 below the cut. Look at both groups by hand, or have someone who knows the data do it.

If the two groups look alike, the cut is arbitrary and the ranking is all there is. If the flagged group is dominated by one kind of row, check whether many similar rows are sitting just as far from normal, and whether your neighbourhood size could see them.""")

    post.add(
        "Next",
        """Every method in this arc has asked for a number before it started: a perplexity, a number of components, a radius, a contamination. The most basic is the number of clusters itself. Episode 60 tests two ways of letting the data choose it, by resampling and by comparison with data that has no clusters.""")

    return post


def mk_rows(mk):
    return {r["k"]: r for r in mk["rows"]}


def k20_knn(mk):
    return next(r for r in mk["rows"] if r["k"] == 20)["knn"]


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
