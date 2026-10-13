"""ML 32: k-Means Finds k Pieces Whether or Not There Are Clusters.

The thirty-second episode of *Machine Learning, Taught Through What Breaks*.
"k-means finds the clusters in the data." Measured on four simulated shapes
with known groupings, ten draws each, and on the digits:

* On four well-separated round blobs, k-means recovers them (adjusted Rand
  index 1.00) and the silhouette picks k = 4 every time.
* On uniform noise with no clusters at all, the silhouette picks k = 4 every
  time, with a score of 0.40, higher than any k reaches on the digits, which
  do have ten classes.
* On one large group and two small ones, the silhouette picks 8 to 10 and
  k-means at the true k scores ARI 0.09. On two long parallel groups it
  picks the true k = 2 and cuts across them, ARI 0.00.

Run: `standarderror run ml132_kmeans --publish`
"""

from __future__ import annotations

import os
import platform
from collections import Counter
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

SHAPES = ("separated", "uniform", "unequal", "elongated")
LABEL = {"separated": "four separated blobs", "uniform": "uniform noise",
         "unequal": "one large group, two small", "elongated": "two long groups"}


def compute() -> dict:
    s = around.kmeans_shapes()
    d = around.kmeans_digits()
    curves = {name: around._scan(around._shapes(name, 0)[0], range(2, 11))
              for name in SHAPES}
    return {"s": s, "d": d, "curves": curves}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    s = res["s"]

    def panels(ax, m):
        ax.axis("off")
        for i, name in enumerate(SHAPES):
            sub = ax.inset_axes([(i % 2) * 0.52, (1 - i // 2) * 0.52,
                                 0.46, 0.44])
            X = np.array(s[name]["example"]["X"])
            lab = np.array(s[name]["example"]["labels"])
            for c in range(lab.max() + 1):
                sub.scatter(*X[lab == c].T, s=3, alpha=0.6,
                            color=m.series[c % len(m.series)])
            sub.set_xticks([])
            sub.set_yticks([])
            k = s[name]["true_k"] if name != "uniform" else 4
            sub.set_title(f"{LABEL[name]}, k = {k}", fontsize=8.5,
                          color=m.ink)
            if name == "elongated":
                sub.set_aspect("equal")

    out["f0"] = charts.diagram(
        panels,
        title="What k-means draws",
        subtitle=("One draw of each simulated shape, coloured by k-means at "
                  "the true number of groups (k = 4 for the noise, the "
                  "silhouette's choice)."),
        figsize=(7.4, 6.6),
        ticks=False,
        source="Simulated; standarderror/ml/around.py.",
        alt=("Four scatter panels: four blobs coloured cleanly; a uniform "
             "square cut into four quadrants; a large blob cut into slices "
             "while two small blobs share a colour; two long horizontal bands "
             "cut vertically into left and right halves."),
        caption=("Every panel is cut into convex pieces of similar spread. "
                 "That is right for the blobs and wrong for the other three, "
                 "including the square that has no groups at all."),
        path=str(IMG / f"ml132-f0-shapes.{EXT}"))[0]

    curves, d = res["curves"], res["d"]

    def sil(ax, m):
        for i, name in enumerate(SHAPES):
            c = curves[name]
            ax.plot([t["k"] for t in c], [t["silhouette"] for t in c],
                    marker="o", ms=3.5, color=m.series[i], label=LABEL[name])
        ax.plot([t["k"] for t in d["scan"]], [t["silhouette"] for t in d["scan"]],
                marker="o", ms=3.5, color=m.ink, ls="--",
                label="digits (ten real classes)")
        ax.set_ylim(0, 0.85)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    out["f1"] = charts.diagram(
        sil,
        title="The silhouette scores noise above the digits",
        subtitle=("Mean silhouette of k-means at each k, for one draw of each "
                  "shape and for the 1,797 digits. Higher is read as better "
                  "separated clusters."),
        xlabel="number of clusters, k",
        ylabel="mean silhouette",
        figsize=(7.4, 4.6),
        source=("Simulated and UCI digits (scikit-learn); "
                "standarderror/ml/around.py."),
        alt=("Curves of silhouette against k: the separated blobs peak high "
             "at 4; uniform noise sits around 0.4; the digits curve is lowest, "
             "under 0.2 at every k."),
        caption=(f"Uniform noise scores {s['uniform']['silhouette']:.2f} at its "
                 f"best k; the digits never pass "
                 f"{max(t['silhouette'] for t in d['scan']):.2f}."),
        path=str(IMG / f"ml132-f1-silhouette.{EXT}"))[0]

    rows = []
    for name in SHAPES:
        r = s[name]
        c = Counter(r["picks"])
        picks = (f"{min(r['picks'])} to {max(r['picks'])}"
                 if len(c) > 1 else f"{r['picks'][0]} (all)")
        rows.append([LABEL[name], str(r["true_k"]) if name != "uniform"
                     else "none", picks, f"{r['silhouette']:.2f}",
                     f"{r['ari']:.2f}" if r["ari"] is not None else "—"])
    best = max(d["scan"], key=lambda t: t["silhouette"])
    rows.append(["digits", "10", f"{best['k']}", f"{best['silhouette']:.2f}",
                 f"{d['ari10']:.2f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["data", "true groups", "silhouette picks k", "best silhouette",
                "agreement at the true k (ARI)"],
        title="What the score says, and what k-means found",
        subtitle=("Simulated shapes: ten fresh draws of 1,000 points each, "
                  "k from 2 to 10. Digits: k from 2 to 15. ARI is 1 for "
                  "perfect recovery and 0 for chance."),
        source=("Simulated and UCI digits (scikit-learn); "
                "standarderror/ml/around.py."),
        alt=("A five-row table: separated blobs recovered; uniform noise "
             "given four clusters with a fairly high silhouette; the unequal "
             "and elongated shapes recovered at chance level."),
        caption=("The silhouette was right about k in two cases out of four, "
                 "and k-means was right about the groups in one."),
        bold_cells={(1, 3), (2, 4), (3, 4)},
        align="lrrrr",
        path=str(IMG / f"ml132-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    s, d = res["s"], res["d"]

    def noise(panel, m):
        X = np.array(s["uniform"]["example"]["X"])
        lab = np.array(s["uniform"]["example"]["labels"])
        for c in range(4):
            panel.scatter(*X[lab == c][:150].T, s=4, color=m.series[c])

    def long(panel, m):
        X = np.array(s["elongated"]["example"]["X"])
        lab = np.array(s["elongated"]["example"]["labels"])
        for c in range(2):
            panel.scatter(*X[lab == c][:200].T, s=4, color=m.series[c])

    def sil(panel, m):
        panel.bar([0, 1], [s["uniform"]["silhouette"],
                           max(t["silhouette"] for t in d["scan"])],
                  color=[m.series[1], m.series[0]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=32,
        headline="k pieces, clusters or not",
        panels=[(noise, f"{s['uniform']['silhouette']:.2f}",
                 "silhouette of pure noise"),
                (long, f"ARI {s['elongated']['ari']:.2f}", "two long groups"),
                (sil, f"{max(t['silhouette'] for t in d['scan']):.2f}",
                 "best for the digits")],
        note=("k-means always returns k convex pieces of similar spread. The "
              "silhouette rates those pieces, not whether the data have "
              "groups, and scores structureless noise above the digits."),
        alt=("Three hand-drawn frames: a square of dots in four colours; two "
             "long bands split left and right; a taller bar beside a shorter "
             "one."),
        path=str(IMG / f"ml132-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["shapes"] = s.run("""
        from standarderror.ml import around

        r = around.kmeans_shapes()      # ten draws of each shape
        for name, v in r.items():
            ari = "  -  " if v["ari"] is None else f"{v['ari']:.3f}"
            print(f"{name:10s} silhouette picks k = {sorted(set(v['picks']))}, "
                  f"best silhouette {v['silhouette']:.3f}, ARI at true k {ari}")
        """)
    out["digits"] = s.run("""
        d = around.kmeans_digits()
        best = max(d["scan"], key=lambda t: t["silhouette"])
        print(f"digits: silhouette peaks at k = {best['k']} "
              f"({best['silhouette']:.3f}); ARI at k = 10: {d['ari10']:.3f}")
        print("one start per seed, ARI: " + " ".join(
            f"{x['ari']:.2f}" for x in d["single"]))
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    s, d = res["s"], res["d"]
    sep, uni, une, elo = (s[n] for n in SHAPES)
    dbest = max(d["scan"], key=lambda t: t["silhouette"])
    single = [x["ari"] for x in d["single"]]
    by_inertia = sorted(d["single"], key=lambda x: x["inertia"])
    best_ari = max(d["single"], key=lambda x: x["ari"])
    rank_best = by_inertia.index(best_ari) + 1

    # The spine, asserted rather than trusted.
    assert set(sep["picks"]) == {4} and sep["ari"] > 0.99
    assert set(uni["picks"]) == {4} and uni["silhouette"] > 0.38
    assert uni["silhouette"] > 2 * dbest["silhouette"]
    assert min(une["picks"]) >= 8 and une["ari"] < 0.15
    assert set(elo["picks"]) == {2} and elo["ari"] < 0.02
    assert dbest["k"] in (9, 10) and dbest["silhouette"] < 0.2
    assert 0.6 < d["ari10"] < 0.7
    assert max(single) - min(single) > 0.1
    assert rank_best >= 5 and best_ari["ari"] > by_inertia[0]["ari"]

    post = Post(
        title=(f"{SERIES_TAG} 32: k-Means Finds k Pieces Whether or Not There "
               f"Are Clusters"),
        slug="ml-32-kmeans",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=32,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("k-means always returns k convex pieces of similar spread, "
                  "and the silhouette score rates the pieces. On uniform "
                  "noise it rates them higher than it rates the digits."),
        summary=(
            f"Over ten draws of four simulated shapes, k-means recovered four "
            f"separated round blobs (adjusted Rand index {sep['ari']:.2f}) and "
            f"the silhouette chose k = 4 each time. On uniform noise, with no "
            f"clusters, the silhouette also chose k = 4 each time, with a score "
            f"of {uni['silhouette']:.2f}; on the 1,797 digits, which have ten "
            f"classes, it never passed {dbest['silhouette']:.2f}. On one large "
            f"group and two small ones it chose {min(une['picks'])} to "
            f"{max(une['picks'])} clusters and k-means at the true k scored "
            f"ARI {une['ari']:.2f}; on two long parallel groups it chose the "
            f"true k and k-means cut across them, ARI {elo['ari']:.2f}. On the "
            f"digits, single starts ranged from ARI {min(single):.2f} to "
            f"{max(single):.2f}."),
        tags=["machine-learning", "data-science", "clustering", "k-means",
              "unsupervised-learning", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated two-dimensional shapes, 1,000 points each, ten draws "
            "(seeds 0 to 9): four unit-variance blobs 8 apart; uniform on the "
            "unit square; 900 points with sd 1.5 and two groups of 50 with sd "
            "0.3; two groups of 500 stretched 12 times along one axis, 3 "
            "apart along the other.",
            "Optical Recognition of Handwritten Digits (UCI), as bundled with "
            "scikit-learn, CC BY 4.0; 1,797 images, raw pixels.",
            "Machinery: `standarderror/ml/around.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's KMeans (ten starts unless "
            "stated), silhouette_score and adjusted_rand_score.",
            "Where this stops: MacQueen, \"Some methods for classification "
            "and analysis of multivariate observations\" (1967); Rousseeuw, "
            "\"Silhouettes\", *Journal of Computational and Applied "
            "Mathematics* (1987); Tibshirani, Walther and Hastie, \"Estimating "
            "the number of clusters in a data set via the gap statistic\", "
            "*JRSS B* (2001).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("data seeds 0 to 9; KMeans random_state 0, or "
                            "0 to 9 for the single-start runs"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Groups the data were never labelled with",
        f"""Episode 31 reduced the inputs without looking at a target. Clustering goes further and invents one: it assigns each row to a group, and the groups are then named, profiled and acted on as if they had been found in the data. The most used algorithm is k-means, and the sentence under test is how it is usually described: *k-means finds the clusters in the data*.

k-means does one thing exactly. Given k, it places k centres so that the sum of squared distances from each point to its nearest centre is as small as it can find, and assigns each point to that centre. Whether the result is "the clusters" depends on whether the data contain groups of the kind that criterion rewards. The number k has to be chosen too, and the usual tool is the silhouette, which compares each point's distance to its own cluster with its distance to the next nearest, averaged over points: 1 for tight, well-separated clusters, 0 for points on a boundary.

Four simulated shapes, each drawn ten times, put both to the test. The first has what k-means looks for: four round, equally spread blobs, well apart. The second has nothing: points uniform on a square. The third has a large diffuse group and two small tight ones. The fourth has two long parallel groups.

{snip['shapes'].markdown()}

On the separated blobs everything works: the silhouette chooses k = 4 in all ten draws, and k-means recovers the blobs with an adjusted Rand index of **{sep['ari']:.2f}**, where 1 is perfect and 0 is chance.""",
        figures=[figs["f0"]])

    post.add(
        "A score for noise",
        f"""On uniform noise there is nothing to find, and both tools find something anyway. k-means cuts the square into k pieces, as it cuts anything. The silhouette chooses k = 4 in all ten draws, with a mean score of **{uni['silhouette']:.2f}**: four quadrants of a square are compact and their boundaries are short, so by its arithmetic they are decent clusters.

The comparison that matters is with real data that do have classes:

{snip['digits'].markdown()}

On the digits, ten classes written by different people, the silhouette peaks at k = {dbest['k']} with a score of **{dbest['silhouette']:.2f}**, less than half what it gives the noise. The number of clusters it suggests is about right, and the score itself would be read as weak structure. A practitioner comparing silhouettes across data sets would conclude the square is more clustered than the handwriting.

A silhouette value means nothing without a reference for what it would be with no clusters. That is the idea behind Tibshirani, Walther and Hastie's gap statistic: compare the clustering's fit with the fit on data drawn uniformly over the same range, and claim clusters only when the real data beat the reference by more than its noise. Here, the reference is the second row of the table, and it scores {uni['silhouette']:.2f}.""",
        figures=[figs["f1"]])

    post.add(
        "Clusters of the wrong shape",
        f"""The last two shapes have real groups, of shapes k-means does not draw. With one large diffuse group and two small tight ones, k-means at the true k = 3 scores ARI **{une['ari']:.2f}**. The large group contributes most of the squared distance, so minimising it pays to spend centres on slicing that group, and the first figure shows the result: the large group cut up, the two small ones not given a centre each. The silhouette makes it worse, choosing between {min(une['picks'])} and {max(une['picks'])} clusters across the ten draws.

With two long parallel groups, the silhouette chooses the correct k = 2 in every draw, and k-means scores ARI **{elo['ari']:.2f}**. Each group is stretched twelve times as far along its length as across it, and the two lie only 3 apart, so the partition k-means settles on cuts both groups in half across their length instead of separating them. The answer has the right number of clusters and the wrong clusters.

In both cases the algorithm did what it promises. Its promise is convex regions of similar spread around k centres. When the true groups are not shaped like that, k-means returns its own shapes instead, and nothing in its output says so.""",
        figures=[figs["f2"]])

    post.add(
        "And the starting point",
        f"""k-means finds a local minimum, and which one depends on where the centres start. On the digits, ten runs with a single random start each gave ARIs from {min(single):.2f} to {max(single):.2f}. scikit-learn's default of several starts, keeping the one with the smallest within-cluster sum of squares, gave {d['ari10']:.2f}. The groups a single run reports are partly an accident of its seed, which is the clustering version of episode 20's networks that differed only in their seed.

Keeping the start with the smallest sum of squares is the right rule for k-means' own objective, and it is worth seeing how loosely that objective tracks the digits' real classes. Ranked by sum of squares, the best of the ten single starts agreed with the true digits at ARI {by_inertia[0]['ari']:.2f}; the start that agreed best, at {best_ari['ari']:.2f}, ranked {rank_best} of 10 by sum of squares. A lower sum of squares is a better k-means solution, not a better account of the handwriting.""")

    post.add(
        "Where this breaks",
        """**Two dimensions.** The shapes were drawn in the plane so they could be seen. In many dimensions the same failures happen and cannot be seen, which is the argument for checking clusters against a null reference instead of by eye.

**Other algorithms draw other shapes.** Gaussian mixtures allow ellipses of different sizes, and density-based methods such as DBSCAN are designed to follow groups of any shape and leave sparse regions unclustered. They were not run here, and each has its own assumptions and its own failures.

**Raw pixels for the digits.** Clustering the digits after scaling or on learned features would change both the silhouette and the ARI. The comparison with the uniform square is about what the score means, not about the best possible clustering of handwriting.

**Silhouette only.** The elbow of the within-cluster sum of squares, the other common rule, has the same blind spot: the sum falls with k on any data, including noise.""")

    post.add(
        "What to keep",
        f"""1. k-means returns k convex pieces of similar spread on any data. It recovered separated round blobs (ARI {sep['ari']:.2f}) and nothing else here.

2. The silhouette rates the pieces, not the presence of groups. Uniform noise scored {uni['silhouette']:.2f} at its chosen k; the digits never passed {dbest['silhouette']:.2f}.

3. Groups of the wrong shape are cut wrongly: ARI {une['ari']:.2f} for one large group and two small, {elo['ari']:.2f} for two long ones, with the silhouette picking {min(une['picks'])} to {max(une['picks'])} clusters for the first.

4. Compare any clustering score with its value on structureless data of the same range, use several starts, and check whether the cluster shapes the algorithm assumes are plausible for the data.""")

    post.add(
        "Exercise",
        """Take a clustering you have used and rerun it on a reference data set: each column drawn uniformly between that column's minimum and maximum, independently of the others. Compute the same quality score on both.

If the reference scores close to the real data, the clusters you reported are mostly the algorithm's partition of space, not groups in the data. Then rerun k-means with ten different single starts and count how many rows change cluster.""")

    post.add(
        "Next",
        """Clustering and nearest-neighbour methods both rest on distances. Episode 33 asks what happens to distance as the number of dimensions grows, and whether the curse of dimensionality is about the number of columns or about something else.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
