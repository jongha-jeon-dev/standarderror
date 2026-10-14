"""ML 56: A t-SNE Picture Keeps the Neighbours and Loses the Map.

The fifty-sixth episode of *Machine Learning, Taught Through What Breaks*,
opening Arc XII. "A t-SNE plot shows the data's clusters, how big they are
and how far apart." Measured on simulated clusters with known sizes and
distances, and on the digits:

* Two clusters, one ten times wider than the other, came out the same size
  in the picture (radius ratio 0.99 to 1.01 at perplexities 5, 30 and 100).
* Three clusters with one gap ten times another came out with gaps in a
  ratio of 0.98 to 1.45.
* On the digits t-SNE kept 59% of each point's ten nearest neighbours, a
  two-component PCA 12%. With random initialisation it ranked the distances
  between digit classes less faithfully than PCA (0.65-0.75 against 0.81)
  and two seeds agreed on them as little as 0.50; with PCA initialisation,
  scikit-learn's default, every seed gave 0.82 and the same layout.
* Clustering the picture recovered the digits better (ARI 0.79-0.84) than
  clustering the 64 pixels with k-means (0.67) or HDBSCAN (0.17).

Run: `standarderror run ml156_tsne --publish`
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
    return {"sizes": unsup.tsne_sizes(), "dist": unsup.tsne_distances(),
            "digits": unsup.tsne_digits()}


# ------------------------------------------------------------------ figures

def _pair(ax, m, X, Y, n, title_x):
    """Data (first two coordinates) on the left, picture on the right."""
    def norm(Z):
        Z = Z - Z.mean(0)
        return Z / np.abs(Z).max()
    a, b = norm(X), norm(Y)
    for Z, off in ((a, -1.2), (b, 1.2)):
        ax.scatter(Z[:n, 0] + off, Z[:n, 1], s=3, color=m.series[0],
                   alpha=0.6)
        ax.scatter(Z[n:, 0] + off, Z[n:, 1], s=3, color=m.series[1],
                   alpha=0.6)
    ax.text(-1.2, 1.15, title_x, ha="center", fontsize=9)
    ax.text(1.2, 1.15, "t-SNE, perplexity 30", ha="center", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-1.2, 1.3)


def figures(res: dict) -> dict:
    out: dict = {}
    s30 = next(r for r in res["sizes"] if r["perplexity"] == 30)
    out["f0"] = charts.diagram(
        lambda ax, m: _pair(ax, m, s30["X"], s30["Y"], len(s30["X"]) // 2,
                            "the data, first two coordinates"),
        title="Ten times wider, drawn the same size",
        subtitle=("Two Gaussian clusters in ten dimensions, 500 points each, "
                  "one with ten times the spread of the other. Left: the "
                  "first two coordinates, both panels rescaled to fit. Right: "
                  "t-SNE."),
        xlabel="", ylabel="",
        figsize=(7.6, 3.9), ticks=False,
        source="Simulated; standarderror/ml/unsup.py.",
        alt=("Left, a tiny dense cluster beside a large diffuse one. Right, "
             "two round clusters of the same size."),
        caption=(f"Radius of the wide cluster over the narrow one: "
                 f"{s30['data_ratio']:.1f} in the data, "
                 f"**{s30['embedded_ratio']:.2f}** in the picture."),
        path=str(IMG / f"ml156-f0-sizes.{EXT}"))[0]

    body = []
    for s, d in zip(res["sizes"], res["dist"]):
        body.append([str(s["perplexity"]), f"{s['data_ratio']:.1f}",
                     f"{s['embedded_ratio']:.2f}", f"{d['data_ratio']:.1f}",
                     f"{d['embedded_ratio']:.2f}"])
    out["f1"] = charts.table_image(
        body,
        header=["perplexity", "size ratio, data", "size ratio, picture",
                "gap ratio, data", "gap ratio, picture"],
        title="What the picture says about size and distance",
        subtitle=("Size: radius of a cluster with ten times the spread over "
                  "the narrow one's. Gap: the distance from B to C over the "
                  "distance from A to B, for three clusters on a line."),
        source="Simulated; standarderror/ml/unsup.py.",
        alt=("A three-row table. The data's ratios are about 10; the "
             "picture's are all close to 1."),
        caption="Every cluster has 300 or 500 points in ten dimensions; one t-SNE run per cell.",
        align="rrrrr",
        path=str(IMG / f"ml156-f1-ratios.{EXT}"))[0]

    dg = res["digits"]
    body = [["PCA, two components", f"{dg['pca']['knn']:.3f}",
             f"{dg['pca']['global']:.3f}", "-", "-"]]
    for t in dg["tsne"]:
        body.append([f"t-SNE, random start, seed {t['seed']}",
                     f"{t['knn']:.3f}", f"{t['global']:.3f}",
                     f"{t['hdbscan_ari']:.3f}", str(t["hdbscan_clusters"])])
    pi = dg["pca_init"]
    body.append(["t-SNE, PCA start, seeds 0-4",
                 f"{min(t['knn'] for t in pi):.3f}-{max(t['knn'] for t in pi):.3f}",
                 f"{min(t['global'] for t in pi):.3f}-"
                 f"{max(t['global'] for t in pi):.3f}", "-", "-"])
    out["f2"] = charts.table_image(
        body,
        header=["picture", "10 neighbours kept", "class distances, rank corr.",
                "HDBSCAN on it, ARI", "clusters found"],
        title="The digits: local truth, global guesswork",
        subtitle=("Share of each digit's ten nearest neighbours (in 64 "
                  "pixels) still among its ten nearest in the picture; "
                  "Spearman correlation of the 45 distances between class "
                  "centroids, picture against pixels; and HDBSCAN run on "
                  "the picture."),
        source="UCI digits via scikit-learn; standarderror/ml/unsup.py.",
        alt=("A seven-row table. t-SNE keeps about 0.59 of neighbours against "
             "PCA's 0.12; started at random it ranks class distances worse "
             "than PCA, started from PCA as well as PCA."),
        caption=(f"HDBSCAN on the 64 pixels: ARI {dg['hdbscan_raw_ari']:.2f}, "
                 f"with {dg['hdbscan_raw_noise']:.0%} of digits called noise; "
                 f"k-means on the pixels: {dg['kmeans_raw_ari']:.2f}."),
        align="lrrrr",
        path=str(IMG / f"ml156-f2-digits.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    s30 = next(r for r in res["sizes"] if r["perplexity"] == 30)
    d30 = next(r for r in res["dist"] if r["perplexity"] == 30)
    dg = res["digits"]

    def p1(panel, m):
        panel.bar([0, 1], [s30["data_ratio"], s30["embedded_ratio"]],
                  color=[m.series[0], m.series[1]])

    def p2(panel, m):
        panel.bar([0, 1], [d30["data_ratio"], d30["embedded_ratio"]],
                  color=[m.series[0], m.series[1]])

    def p3(panel, m):
        panel.bar([0, 1], [dg["pca"]["knn"],
                           np.mean([t["knn"] for t in dg["tsne"]])],
                  color=[m.series[2], m.series[1]])
        panel.set_ylim(0, 1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=56,
        headline="Neighbours, not a map",
        panels=[(p1, f"{s30['embedded_ratio']:.2f}", "size ratio, was 10"),
                (p2, f"{d30['embedded_ratio']:.2f}",
                 f"gap ratio, was {d30['data_ratio']:.1f}"),
                (p3, f"{np.mean([t['knn'] for t in dg['tsne']]):.2f}",
                 "neighbours kept")],
        note=("t-SNE keeps who is next to whom. How big a cluster looks and "
              "how far it sits from the next are not measurements of the "
              "data."),
        alt=("Three hand-drawn frames: a tall bar beside a short one, twice; "
             "a short bar beside a taller one."),
        path=str(IMG / f"ml156-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["sizes"] = s.run("""
        from standarderror.ml import unsup

        for r in unsup.tsne_sizes(perplexities=(30,)):   # one wide, one narrow
            print(f"radius ratio: data {r['data_ratio']:.2f}, "
                  f"t-SNE {r['embedded_ratio']:.2f}")
        for r in unsup.tsne_distances(perplexities=(30,)):  # gaps 6 and 60
            print(f"gap ratio:    data {r['data_ratio']:.2f}, "
                  f"t-SNE {r['embedded_ratio']:.2f}")
        """)
    out["digits"] = s.run("""
        d = unsup.tsne_digits(seeds=(0,))
        t = d["tsne"][0]
        print(f"neighbours kept: PCA {d['pca']['knn']:.3f}, t-SNE {t['knn']:.3f}")
        print(f"class-distance rank corr.: PCA {d['pca']['global']:.3f}, "
              f"t-SNE {t['global']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    sz, ds, dg = res["sizes"], res["dist"], res["digits"]
    s_er = [r["embedded_ratio"] for r in sz]
    d_er = [r["embedded_ratio"] for r in ds]
    knn = [t["knn"] for t in dg["tsne"]]
    glob = [t["global"] for t in dg["tsne"]]
    hari = [t["hdbscan_ari"] for t in dg["tsne"]]
    ncl = [t["hdbscan_clusters"] for t in dg["tsne"]]
    bs = dg["between_seeds"]
    pg = [t["global"] for t in dg["pca_init"]]
    pk = [t["knn"] for t in dg["pca_init"]]
    pb = dg["pca_init_between"]

    # The spine, asserted rather than trusted.
    assert all(9.5 < r["data_ratio"] < 10.5 for r in sz)
    assert all(0.95 < v < 1.05 for v in s_er)
    assert all(9.5 < r["data_ratio"] < 10.2 for r in ds)
    assert max(d_er) < 1.6 and min(d_er) < 1.0
    assert dg["pca"]["knn"] < 0.15 and min(knn) > 0.55
    assert dg["pca"]["global"] > max(glob) + 0.05
    assert min(bs) < 0.55 and max(bs) > 0.9
    assert min(hari) > 0.75 and dg["kmeans_raw_ari"] < min(hari) - 0.05
    assert dg["hdbscan_raw_ari"] < 0.2 and dg["hdbscan_raw_noise"] > 0.5
    assert set(ncl) <= {10, 11}
    assert min(pg) > dg["pca"]["global"] - 0.01 and min(pb) > 0.99
    assert min(pk) > 0.55
    assert 9 < sz[0]["pca_ratio"] < 11 and 9 < ds[0]["pca_ratio"] < 10.5

    post = Post(
        title=(f"{SERIES_TAG} 56: A t-SNE Picture Keeps the Neighbours and "
               f"Loses the Map"),
        slug="ml-56-tsne",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=56,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A cluster ten times wider than another came out the same "
                  "size, and a gap ten times wider came out about the same "
                  "width. What survives the picture is who sits next to "
                  "whom."),
        summary=(
            f"Two Gaussian clusters in ten dimensions, one ten times as wide "
            f"as the other, came out of t-SNE with a radius ratio between "
            f"{min(s_er):.2f} and {max(s_er):.2f} at perplexities 5, 30 and "
            f"100; three clusters with one gap ten times the other came out "
            f"with a gap ratio between {min(d_er):.2f} and {max(d_er):.2f}. "
            f"On the 1,797 digits, t-SNE kept {np.mean(knn):.0%} of each "
            f"point's ten nearest neighbours against {dg['pca']['knn']:.0%} "
            f"for two principal components, but ranked the distances "
            f"between digit classes less faithfully ({min(glob):.2f} to "
            f"{max(glob):.2f} against {dg['pca']['global']:.2f}) when started at "
            f"random, with seeds agreeing on them as little as {min(bs):.2f}; "
            f"started from PCA, every seed scored {min(pg):.2f}. Clustering the picture with HDBSCAN recovered "
            f"the classes with ARI {min(hari):.2f} to {max(hari):.2f}."),
        tags=["machine-learning", "unsupervised-learning", "t-sne",
              "visualization", "clustering", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: Gaussian clusters in ten dimensions (seed 0); two "
            "of 500 points with standard deviations 1 and 10, and three of "
            "300 with unit spread and centres 0, 6 and 66 along one axis.",
            "UCI optical digits (1,797 8x8 images, CC BY 4.0) via "
            "scikit-learn.",
            "Machinery: scikit-learn's TSNE (Barnes-Hut, PCA initialisation "
            "for the simulations; for the digits, random and then PCA "
            "initialisation, seeds 0-4 each) and HDBSCAN (minimum cluster size 25); "
            "`standarderror/ml/unsup.py`, tested in `tests/test_ml.py`.",
            "Where this stops: van der Maaten and Hinton, \"Visualizing data "
            "using t-SNE\", *JMLR* (2008); Wattenberg, Viégas and Johnson, "
            "\"How to use t-SNE effectively\", *Distill* (2016); Kobak and "
            "Berens, \"The art of using t-SNE for single-cell "
            "transcriptomics\", *Nature Communications* (2019); episodes 31 "
            "and 32 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; about two minutes of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A picture with no right answer to check",
        f"""Every arc so far has had something to check a model against: a label, a known parameter, an exact sum. This arc drops the labels. Clustering, mixture models, density methods and anomaly scores are fitted with no right answer in the data, which makes their output easy to believe and hard to test. The way to test them is the one the series has used throughout: simulate data whose structure is known, then replicate on real data where some of it is.

The most believed output of all is a picture. t-SNE, introduced by van der Maaten and Hinton in 2008, turns high-dimensional points into a two-dimensional scatter in which similar points sit together, and its plots of cells, documents and embeddings are read every day as maps: these groups exist, this one is bigger, those two are close. That reading is the sentence under test: *a t-SNE plot shows the data's clusters, how big they are and how far apart*.

Wattenberg, Viégas and Johnson warned in 2016, with interactive examples, that cluster sizes and the distances between clusters in a t-SNE plot may mean nothing. Here the warning gets numbers: clusters whose sizes and distances are known, and a ratio to compare before and after.

{snip['sizes'].markdown()}""")

    post.add(
        "Ten times wider, drawn the same size",
        f"""Two Gaussian clusters of 500 points in ten dimensions, far apart, one with ten times the other's standard deviation. In the data the wide cluster's radius is {sz[0]['data_ratio']:.1f} times the narrow one's. In the t-SNE picture it was {min(s_er):.2f} to {max(s_er):.2f} times, at perplexities of 5, 30 and 100.

That is not a malfunction. t-SNE gives each point its own bandwidth, chosen so that the point has the same effective number of neighbours, the perplexity, wherever it is. A point in a sparse cluster gets a wide bandwidth and a point in a dense one a narrow bandwidth, so after the conversion every cluster looks equally dense. The algorithm equalises density by design, and the size of a cluster in the picture is not a statement about how spread out its points are.""",
        figures=[figs["f0"]])

    post.add(
        "Gaps are not distances",
        f"""Three clusters on a line: A, then B six units away, then C sixty units beyond B, so the B-C gap is {ds[0]['data_ratio']:.1f} times the A-B gap in the data. In the picture the ratio was {ds[0]['embedded_ratio']:.2f} at perplexity 5, {ds[1]['embedded_ratio']:.2f} at 30 and {ds[2]['embedded_ratio']:.2f} at 100. A cluster that is ten times farther away sat about as far away as a near neighbour.

The objective explains it. t-SNE matches the probabilities that points are each other's neighbours, and two clusters that share no neighbours at all have essentially zero probability of being neighbours whether they are 6 units apart or 60; nothing in the loss rewards placing them in proportion. Larger perplexities let more distant points count as neighbours, which is presumably why the ratio grew with perplexity, and why Kobak and Berens recommend large perplexities and a PCA initialisation when the global layout matters.""",
        figures=[figs["f1"]])

    post.add(
        "A linear picture keeps both",
        f"""The control is the plainest two-dimensional picture there is: the first two principal components, a rotation and a projection. On the same two clusters it drew the wide one {sz[0]['pca_ratio']:.1f} times the radius of the narrow one, and on the three clusters on a line it drew the far gap {ds[0]['pca_ratio']:.1f} times the near one. Both are close to the {sz[0]['data_ratio']:.1f} and {ds[0]['data_ratio']:.1f} of the data: a projection onto the directions of greatest variance keeps what lies along them, and here the gaps and most of the wide cluster's spread do.

So the distortions belong to t-SNE, not to squeezing ten dimensions into two. The price of a linear picture is that it keeps variance, not neighbourhoods (on the digits below it kept 12% of each point's ten nearest neighbours), and, as episode 31 found for what matters to a prediction, when the structure that matters is not along the directions of greatest variance it draws the wrong thing faithfully. The two methods fail in opposite places, which is the argument for looking at both before believing either.""")

    post.add(
        "What the picture keeps",
        f"""The digits test the other half of the claim, the part t-SNE is built for. For each of the 1,797 images, take its ten nearest neighbours in the 64 pixel dimensions and ask how many are still among its ten nearest in the picture. Two principal components, the linear picture from episode 31, kept {dg['pca']['knn']:.0%}. t-SNE kept {min(knn):.1%} to {max(knn):.1%} across five seeds. Locally, it is far better.

{snip['digits'].markdown()}

Globally it depends on how the picture starts. Take the ten class centroids and the 45 distances between them, in pixels and in the picture, and correlate their ranks. PCA scored {dg['pca']['global']:.2f}. These five t-SNE runs started from random positions, and scored {min(glob):.2f} to {max(glob):.2f}; two of them that differed only in the seed agreed with each other on those 45 distances with rank correlations from {min(bs):.2f} to {max(bs):.2f}. With random starts, which digit classes sit next to each other is partly a property of the seed.

Started instead from the first two principal components, scikit-learn's default and what Kobak and Berens recommend, every one of the five seeds scored {min(pg):.3f} to {max(pg):.3f} against the pixels, kept {np.mean(pk):.1%} of neighbours, and the five pictures agreed with each other at {min(pb):.2f} or more. The PCA start carries the linear picture's global layout into t-SNE, and t-SNE then rearranges the neighbourhoods inside it. The layout of a t-SNE picture is only as trustworthy as its initialisation.

Clustering the picture, a common shortcut, worked on the digits: HDBSCAN on the randomly started t-SNE coordinates recovered the classes with adjusted Rand index {min(hari):.2f} to {max(hari):.2f} (counting HDBSCAN's noise points as one more group, scikit-learn's convention), finding 10 or 11 clusters, where k-means on the raw pixels scored {dg['kmeans_raw_ari']:.2f} and HDBSCAN on the raw pixels {dg['hdbscan_raw_ari']:.2f}, calling {dg['hdbscan_raw_noise']:.0%} of the images noise. The digits have well-separated classes whose neighbourhoods t-SNE preserves; the same shortcut inherits the picture's equal sizes and arbitrary gaps wherever those matter.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        """**Gaussian clusters in ten dimensions.** The simulations isolate one property each. Real data mix them, and with continuous structure, a trajectory rather than groups, t-SNE can also cut one group into pieces, which was not measured here.

**t-SNE only.** UMAP is now used as often and has a different objective; it is often said to keep more global structure, but its sizes and gaps are not distances either. It was not measured here.

**One run per cell in the simulations.** The size ratios were within 0.05 of 1 at every perplexity, which is far from 10 by any standard of run-to-run noise; the gap ratios moved more with perplexity and would move with the seed.

**Digits are easy.** Well-separated classes are the best case for clustering a picture; overlapping classes are where the picture's gaps mislead most.""")

    post.add(
        "What to keep",
        f"""1. A cluster with ten times the spread of another came out {min(s_er):.2f} to {max(s_er):.2f} times its size; a gap {ds[0]['data_ratio']:.1f} times another came out {min(d_er):.2f} to {max(d_er):.2f} times as wide.

2. On the digits t-SNE kept {np.mean(knn):.0%} of each point's ten nearest neighbours, against {dg['pca']['knn']:.0%} for two principal components.

3. Started at random, it ranked the distances between classes less faithfully than PCA and two seeds agreed on them as little as {min(bs):.2f}; started from PCA, every seed matched PCA's {dg['pca']['global']:.2f} and the layouts agreed.

4. Read a t-SNE plot for who sits next to whom. Measure sizes and distances in the data, start from PCA, and check any layout claim across seeds and perplexities.""")

    post.add(
        "Exercise",
        """Take a t-SNE or UMAP plot you have shown someone and the statement it supported: these two groups are close, this group is diffuse, this one is small. Compute that quantity in the original space: the distance between group centroids, the spread within each group, the count.

Then rerun the embedding with three seeds and two perplexities. A statement that survives both checks was about the data.""")

    post.add(
        "Next",
        """A picture invites someone to circle the groups. A Gaussian mixture does the circling with a likelihood and comes with a criterion, the BIC, for how many groups to draw. Episode 57 asks what that criterion counts.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
