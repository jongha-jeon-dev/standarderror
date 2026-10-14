"""ML 60: The Most Stable k Is Not the Number of Clusters.

The sixtieth episode of *Machine Learning, Taught Through What Breaks*,
closing Arc XII. "The right number of clusters is the one whose clustering
is most stable under resampling." Measured with k-means on four simulated
shapes whose number of groups is known, and on the digits, against the gap
statistic:

* On one elongated Gaussian, which has no clusters, the most stable k was 2
  (stability 0.92); on a uniform square, 4 (0.94). Stability cannot say
  "one".
* On three separated blobs it chose 3 (1.00). On two pairs of blobs it
  chose 2 (1.00) over the four blobs (0.99).
* The gap statistic said one cluster for both clusterless shapes and three
  for the blobs; on the two pairs its rule said 2 and its maximum 4. On the
  digits stability peaked at 10 with one resampling seed and at 9 with two
  others, and the gap's rule found no stopping point up to 15.

Run: `standarderror run ml160_stability --publish`
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


def compute() -> dict:
    return {"shapes": unsup.stability_shapes(),
            "digits": unsup.stability_digits()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    sh = res["shapes"]

    def curves(ax, m):
        for i, (name, v) in enumerate(sh.items()):
            ks = sorted(v["stability"])
            ax.plot(ks, [v["stability"][k] for k in ks], marker="o",
                    color=m.series[i],
                    label=f"{name} (groups: {v['truth']})")
        ax.set_ylim(0.3, 1.03)
        ax.legend(fontsize=8, loc="lower left", frameon=False)

    el = sh["one elongated Gaussian"]
    out["f0"] = charts.diagram(
        curves,
        title="Stability, shape by shape",
        subtitle=("Mean adjusted Rand index between k-means clusterings of "
                  "two random 80% subsamples, on the points they share; 20 "
                  "pairs per k. 1,000 points per shape."),
        xlabel="k", ylabel="stability",
        figsize=(7.4, 4.4),
        source="Simulated; standarderror/ml/unsup.py.",
        alt=("Four curves. The elongated Gaussian peaks at k = 2, the "
             "uniform square at 4, the three blobs at 3 with 1.0, and the "
             "two pairs at 2 and 4, both near 1."),
        caption=(f"One elongated Gaussian, no clusters: k = 2 was stable "
                 f"**{el['stability'][2]:.2f}**."),
        path=str(IMG / f"ml160-f0-stability.{EXT}"))[0]

    dg = res["digits"]
    body = []
    for name, v in sh.items():
        st = v["stability"]
        body.append([name, str(v["truth"]), str(v["stable_k"]),
                     f"{st[v['stable_k']]:.3f}", str(v["gap_rule"]),
                     str(v["gap_argmax"])])
    body.append(["digits", "10 classes", str(dg["stable_k"]),
                 f"{dg['stability'][dg['stable_k']]:.3f}",
                 f"{dg['gap_rule']} (largest tried)",
                 f"{dg['gap_argmax']} (largest tried)"])
    out["f1"] = charts.table_image(
        body,
        header=["data", "groups", "most stable k", "its stability",
                "gap: rule", "gap: maximum"],
        title="Two ways to let the data choose k",
        subtitle=("Most stable k from 2 to 8 (digits: 2 to 15). Gap "
                  "statistic from 1 to 8 (digits: 1 to 15) against uniform "
                  "reference data in a box aligned with the principal "
                  "components; 'rule' is Tibshirani and colleagues' "
                  "smallest k within one standard error of the next."),
        source=("Simulated; UCI digits via scikit-learn; "
                "standarderror/ml/unsup.py."),
        alt=("A five-row table. Stability never says 1; the gap says 1 for "
             "the two clusterless shapes. On the digits stability says 10 "
             "and the gap the largest k tried."),
        caption="Stability at k = 1 is trivially perfect and so uninformative: it can never choose 'no clusters'.",
        align="lrrrll",
        path=str(IMG / f"ml160-f1-table.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    sh, dg = res["shapes"], res["digits"]
    el, un = sh["one elongated Gaussian"], sh["uniform square"]

    def p1(panel, m):
        ks = sorted(el["stability"])
        panel.plot(ks, [el["stability"][k] for k in ks], color=m.series[0],
                   lw=2.5, marker="o")
        panel.set_ylim(0.3, 1.02)

    def p2(panel, m):
        ks = sorted(un["stability"])
        panel.plot(ks, [un["stability"][k] for k in ks], color=m.series[1],
                   lw=2.5, marker="o")
        panel.set_ylim(0.3, 1.02)

    def p3(panel, m):
        ks = sorted(dg["stability"])
        panel.plot(ks, [dg["stability"][k] for k in ks], color=m.series[2],
                   lw=2.5, marker="o")
        panel.set_ylim(0.2, 1.0)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=60,
        headline="Stable is not true",
        panels=[(p1, "k = 2", "most stable, no clusters"),
                (p2, "k = 4", "most stable, uniform"),
                (p3, f"k = {min(dg['stable_ks'])}-{max(dg['stable_ks'])}",
                 "most stable, digits")],
        note=("A clustering is stable when its best answer is unique, "
              "which is a property of the method on the data, not a sign "
              "that the clusters exist."),
        alt=("Three hand-drawn curves: one falling from k = 2, one peaking "
             "at 4, one with several peaks, highest at 10."),
        path=str(IMG / f"ml160-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["stab"] = s.run("""
        import numpy as np
        from standarderror.ml import unsup

        X = np.random.default_rng(0).normal(size=(1000, 2)) * [3, 1]
        for k in (2, 3, 5):                 # one elongated Gaussian, no clusters
            print(f"k={k}  stability {unsup.stability(X, k):.3f}")
        g = unsup.gap(X)
        print("gap statistic chooses", g["rule"])
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    sh, dg = res["shapes"], res["digits"]
    el, un = sh["one elongated Gaussian"], sh["uniform square"]
    tb, tp = sh["three blobs"], sh["two pairs of blobs"]
    dst = dg["stability"]
    near10 = max(abs(dst[9] - dst[10]), abs(dst[11] - dst[10]))
    gaps = [r["gap"] for r in dg["gap"]]

    # The spine, asserted rather than trusted.
    assert el["stable_k"] == 2 and el["stability"][2] > 0.9
    assert un["stable_k"] == 4 and un["stability"][4] > 0.9
    assert tb["stable_k"] == 3 and tb["stability"][3] > 0.99
    assert tp["stable_k"] == 2 and tp["stability"][4] > 0.98
    assert el["gap_rule"] == 1 and un["gap_rule"] == 1 and tb["gap_rule"] == 3
    assert tp["gap_rule"] == 2 and tp["gap_argmax"] == 4
    assert dg["stable_k"] == 10 and near10 < 0.02
    assert sorted(dg["stable_ks"]) == [9, 9, 10]
    assert el["stability"][6] < el["stability"][7]
    assert dst[2] < 0.3
    assert un["stability"][8] < un["stability"][4]
    assert un["stability"][8] > un["stability"][7]
    assert gaps == sorted(gaps) and dg["gap_rule"] == 15

    post = Post(
        title=f"{SERIES_TAG} 60: The Most Stable k Is Not the Number of Clusters",
        slug="ml-60-stability",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=60,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Resampling found k-means' two-way split of a single "
                  "Gaussian as reproducible as three real clusters. "
                  "Stability measures whether the method has one best "
                  "answer, and cannot say that there are no clusters."),
        summary=(
            f"k-means stability, the agreement between clusterings of two "
            f"random 80% subsamples, chose k = 2 for one elongated Gaussian "
            f"(stability {el['stability'][2]:.2f}) and k = 4 for a uniform "
            f"square ({un['stability'][4]:.2f}), neither of which has "
            f"clusters. It chose 3 for three separated blobs "
            f"({tb['stability'][3]:.2f}) and 2 for two pairs of blobs "
            f"({tp['stability'][2]:.2f}, against {tp['stability'][4]:.2f} at "
            f"4). The gap statistic said one cluster for both clusterless "
            f"shapes and three for the blobs. On the digits stability peaked "
            f"at k = {', '.join(str(k) for k in dg['stable_ks'])} with three "
            f"resampling seeds, and the gap statistic's rule found no "
            f"stopping point up to 15."),
        tags=["machine-learning", "unsupervised-learning", "clustering",
              "model-selection", "resampling", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated (seed 0), 1,000 points each: a Gaussian with standard "
            "deviations 3 and 1; a unit uniform square; three unit Gaussian "
            "blobs at (0,0), (6,0) and (3,5); four unit blobs at x = 0, 4, "
            "20 and 24.",
            "UCI optical digits (1,797 images, 64 pixels, CC BY 4.0) via "
            "scikit-learn.",
            "Machinery: scikit-learn's KMeans (three starts); stability as "
            "the mean ARI between clusterings of two random 80% subsamples "
            "on their shared points (20 pairs; 10 for the digits); the gap "
            "statistic with 10 uniform reference sets (5 for the digits); "
            "`standarderror/ml/unsup.py`, tested in `tests/test_ml.py`.",
            "Where this stops: Ben-Hur, Elisseeff and Guyon, \"A stability "
            "based method for discovering structure in clustered data\", "
            "*Pacific Symposium on Biocomputing* (2002); Ben-David, von "
            "Luxburg and Pál, \"A sober look at clustering stability\", "
            "*COLT* (2006); von Luxburg, \"Clustering stability: an "
            "overview\", *Foundations and Trends in Machine Learning* "
            "(2010); Tibshirani, Walther and Hastie, \"Estimating the number "
            "of clusters in a data set via the gap statistic\", *JRSS B* "
            "(2001); episodes 32 and 57 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; about a minute and a half of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "If it is real, it should come back",
        f"""Episode 32 found the silhouette choosing four clusters in uniform noise, and episode 57 found the BIC choosing as many Gaussians as a cluster's shape could pay for. A third idea for choosing k sounds more like science than either: a real structure should reproduce. Cluster two random subsamples of the data, compare the two clusterings on the points they share, and repeat; the right k is the one whose answer comes back most consistently. Ben-Hur, Elisseeff and Guyon proposed a version of this in 2002 (their rule looks at the whole distribution of agreements and picks the largest k still concentrated near 1; the version tested here takes the k with the highest mean agreement), and resampling stability is now a widely used check, so the sentence under test is the rule itself: *the right number of clusters is the one whose clustering is most stable under resampling*.

The test is the one this arc has used throughout: shapes with a known number of groups, including none, and the digits. The comparison is the gap statistic of Tibshirani, Walther and Hastie, which asks how much better the clustering fits than it would fit data drawn uniformly over the same region, and which, unlike stability, can answer k = 1.

{snip['stab'].markdown()}""")

    post.add(
        "Stable without clusters",
        f"""On one elongated Gaussian, a single cloud three times longer than it is wide, k = 2 had a stability of **{el['stability'][2]:.2f}**: two random subsamples split the cloud at almost the same place nearly every time. Stability was {el['stability'][3]:.2f} at k = 3 and {el['stability'][4]:.2f} at 4, then dropped, so the highest-mean rule chose 2; Ben-Hur and colleagues' own rule would probably have accepted 3 or 4, which is no better. On a uniform square the most stable k was 4, at {un['stability'][4]:.2f}, the same four that the silhouette chose in episode 32. Neither shape has any clusters.

Ben-David, von Luxburg and Pál explained why in 2006. Stability measures whether the clustering objective has a unique best answer on the data's distribution. An elongated cloud has a unique best two-way cut, across its long axis, and a square has a unique best four-way cut, into quadrants, so both are perfectly reproducible with enough data. Where the objective has several equally good answers, as it would for a perfectly round cloud, stability is low whether or not clusters exist. Reproducibility is a property of the method on the data; it is not evidence that the groups are real. And since stability needs at least two clusters to compare, it can never answer that there is only one.""",
        figures=[figs["f0"]])

    post.add(
        "Where clusters exist",
        f"""On three well-separated blobs stability did its job: k = 3 reproduced perfectly ({tb['stability'][3]:.2f}), and every other k less so. On two pairs of blobs, pairs far apart and the two blobs within each pair close, k = 2 reproduced perfectly ({tp['stability'][2]:.2f}) and k = 4 almost as well ({tp['stability'][4]:.2f}). Both answers are right at different scales, and the rule picked the coarser one because merging the close blobs left nothing to disagree about. A hierarchy is not a mistake, but a single "most stable k" hides it.

The gap statistic, which compares the within-cluster spread with what the same k gives on uniform reference data, chose one cluster for the elongated Gaussian and for the square, and three for the blobs. On the two pairs its stopping rule, the smallest k whose gap is within one standard error of the next, chose 2, and its largest gap was at 4. Of the two methods only the gap could say "no clusters", because only it compares the data with data that has none.""",
        figures=[figs["f1"]])

    post.add(
        "The digits",
        f"""On the 1,797 digits, k-means stability rose from {dst[2]:.2f} at k = 2 to a peak of {dst[10]:.3f} at k = 10, the number of classes, with k = 9 and 11 within {near10:.3f} of it. It is a fragile success. With two other resampling seeds the peak moved to k = 9 ({dg['runs'][1][9]:.3f} and {dg['runs'][2][9]:.3f}); three neighbouring answers are indistinguishable on this evidence, and episode 32 found k-means at k = 10 agreeing with the classes only partly.

The gap statistic, from 1 to 15, rose at every step by more than the next step's standard error, so its rule found no stopping point and fell back to 15, the largest k tried (at the last step by a margin of about 0.001, with five reference sets). It would presumably have gone on: in 64 dimensions, against uniform reference data in a box, every extra centre plausibly explains structure that uniform data lack, whether that structure is ten digit classes or the many ways each digit is written. The gap was built to detect clustering against a null of none, and on data this far from uniform it detects only that.""")

    post.add(
        "What the arc measured",
        """Arc XII took the labels away. A t-SNE picture kept each point's neighbours and drew cluster sizes and gaps that were not in the data (episode 56). The BIC counted the Gaussians a cluster's shape required, more of them the more data it saw (57). DBSCAN found any shape at one density and none when the densities differed sixfold or more, and its standard radius heuristic missed (58). An anomaly detector ranked well and flagged whatever share it was told to; local scores failed for grouped anomalies, and the isolation forest for anomalies hidden in a few of many features (59). And the most reproducible clustering was reproducible because its answer was unique, not because the clusters existed (60).

Each method answered a narrower question than the one it is used for: who is near whom, how many Gaussians, how dense, how unusual in these features, how unique the optimum. The common repair was the one the series has relied on since episode 1, data with a known answer: simulate structure whose truth you know, including the case of no structure at all, and see whether the method can tell.""")

    post.add(
        "Where this breaks",
        """**k-means only.** Stability was measured for k-means, whose objective has unique optima on these shapes. Other algorithms have other objectives and other kinds of stability; the argument of Ben-David and colleagues applies to any algorithm whose answer converges to a unique optimum.

**One version of each method.** Stability has variants that compare to the full-data clustering, use bootstrap samples or normalise by a null; the gap statistic has variants in its reference distribution. Some of them can answer k = 1; this one cannot.

**Two-dimensional shapes.** Four shapes in two dimensions isolate the mechanism; whether higher-dimensional data tend toward unique or ambiguous optima depends on the data, and the digits are one example.

**k from 2 to 8.** On the uniform square stability rose again from k = 7 to k = 8 while staying below its value at 4; a wider range was not tried, and could in principle find a more stable k for any shape.""")

    post.add(
        "What to keep",
        f"""1. Stability chose k = 2 for one elongated Gaussian ({el['stability'][2]:.2f}) and k = 4 for a uniform square ({un['stability'][4]:.2f}); neither has clusters, and stability cannot answer one.

2. It chose correctly for three separated blobs and chose the coarser of two right answers for two pairs of blobs.

3. The gap statistic said one cluster where there were none and three where there were three, and on the digits found no stopping point up to 15; stability chose 10 or 9 there, depending on the resampling seed.

4. Test any rule for choosing k on data with no clusters first. Report reproducibility as evidence about the method's answer, and compare against a null with no structure before claiming the structure exists.""")

    post.add(
        "Exercise",
        """Take a clustering you have reported and fit the same method, with the same k, to data with no clusters but the same spread: shuffle each column independently, which keeps every feature's distribution and destroys the structure between them.

If the clustering of the shuffled data is about as stable, or scores about as well on your validity index, as the real one, the index was measuring the method, not the data.""")

    post.add(
        "Next",
        """Arc XII ends here, at sixty episodes. The plan continues with the gaps a practitioner meets after deployment: data that drift, intervals for regression predictions, fairness trade-offs and the replication of results on real data.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
