"""ML 58: DBSCAN's Radius Is a Density, and It Gets Only One.

The fifty-eighth episode of *Machine Learning, Taught Through What Breaks*,
in Arc XII. "DBSCAN finds clusters of any shape without being told how
many." Measured on two half-moons, on clusters of different densities, and
on the digits:

* On the half-moons DBSCAN scored 0.95 or more (ARI; k-means 0.25) for
  radii from 0.071 to 0.206, and 1.00 from 0.116; the radius read off the
  k-distance plot's knee was 0.061, and scored 0.57 (in five other draws
  of the moons, 0.80 to 0.96).
* Beside two tight clusters, a third whose spread grew from equal to eight
  times theirs left fewer and fewer of 150 radii scoring at least 0.9, and
  none at eightfold in any of six draws; HDBSCAN, which needs no radius,
  scored 0.82 to 0.89 there.
* On the digits the best radius, chosen with the labels, scored 0.71; the
  knee's radius put 96% of the images in one cluster and scored 0.00.

Run: `standarderror run ml158_dbscan --publish`
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
    return {"moons": unsup.dbscan_moons(), "dens": unsup.dbscan_densities(),
            "digits": unsup.dbscan_digits()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    dens = res["dens"]

    def curves(ax, m):
        for k, d in enumerate(d for d in dens if round(d["ratio"]) in (1, 4, 8)):
            ax.plot([r[0] for r in d["curve"]], [r[1] for r in d["curve"]],
                    color=m.series[k],
                    label=f"wide cluster's spread {d['ratio']:.0f}x the tight ones'")
        ax.axhline(0.9, color=m.ink, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="lower left", frameon=False)

    worst = dens[-1]
    out["f0"] = charts.diagram(
        curves,
        title="The radius that works for one density fails another",
        subtitle=("DBSCAN (minimum 5 points) on two tight clusters beside a "
                  "third of growing spread: agreement with the three "
                  "clusters, noise points counted as singletons, across 150 "
                  "radii. Dotted: 0.9."),
        xlabel="radius eps (log scale)",
        ylabel="adjusted Rand index",
        figsize=(7.6, 4.4),
        source="Simulated; standarderror/ml/unsup.py.",
        alt=("Three curves against radius. With equal spreads a wide band of "
             "radii scores near 1; as the third cluster widens the band "
             "shrinks, and at eightfold no radius reaches 0.9."),
        caption=(f"At eightfold spread the best radius scored "
                 f"{worst['best']:.2f}, calling **{worst['wide_noise']:.0%}** "
                 f"of the wide cluster noise."),
        path=str(IMG / f"ml158-f0-radius.{EXT}"))[0]

    body = []
    for d in dens:
        body.append([f"{d['ratio']:.0f}x", str(d["good_radii"]),
                     f"{d['best']:.3f}", f"{d['best_eps']:.3f}",
                     f"{d['wide_noise']:.0%}", f"{d['hdbscan']:.3f}",
                     f"{d['hdbscan_noise']:.1%}"])
    out["f1"] = charts.table_image(
        body,
        header=["wide spread", "radii >= 0.9", "best DBSCAN", "its radius",
                "wide cluster as noise", "HDBSCAN", "HDBSCAN noise"],
        title="One radius, three clusters, growing density contrast",
        subtitle=("Out of 150 radii from 0.02 to 3 (log-spaced): how many "
                  "scored at least 0.9, the best score and what that radius "
                  "did to the wide cluster; HDBSCAN with minimum cluster "
                  "size 25 for comparison."),
        source="Simulated; standarderror/ml/unsup.py.",
        alt=("A five-row table. The count of good radii falls to 0; HDBSCAN "
             "stays above 0.85."),
        caption="Scores are adjusted Rand indices with every noise point its own singleton.",
        align="rrrrrrr",
        path=str(IMG / f"ml158-f1-table.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    mo, dens, dg = res["moons"], res["dens"], res["digits"]

    def p1(panel, m):
        panel.bar([0, 1], [mo["kmeans_ari"], mo["best"]],
                  color=[m.series[0], m.series[1]])
        panel.set_ylim(0, 1.05)

    def p2(panel, m):
        panel.plot([d["ratio"] for d in dens], [d["good_radii"] for d in dens],
                   color=m.series[2], lw=2.5, marker="o")

    def p3(panel, m):
        panel.bar([0, 1], [dg["best"], dg["knee_score"]],
                  color=[m.series[1], m.series[3]])
        panel.text(1, 0.03, f"{dg['knee_score']:.2f}", ha="center",
                   fontsize=8)
        panel.set_ylim(0, 1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=58,
        headline="One radius",
        panels=[(p1, f"{mo['best']:.2f}", "moons, at the right eps"),
                (p2, f"{dens[-1]['good_radii']}", "good radii, 8x spread"),
                (p3, f"{dg['knee_score']:.2f}", "digits, knee's radius")],
        note=("DBSCAN finds any shape at one density. The radius is that "
              "density, and it is a choice nothing in the data makes for "
              "you."),
        alt=("Three hand-drawn frames: a short bar and a full one; a falling "
             "line; a tall bar and an empty one."),
        path=str(IMG / f"ml158-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["moons"] = s.run("""
        from standarderror.ml import unsup

        m = unsup.dbscan_moons()            # 1,000 points, two half-moons
        lo, hi = m["window"]
        print(f"best {m['best']:.2f}; radii scoring 0.9+: {lo:.3f} to {hi:.3f}")
        print(f"knee radius {m['knee']:.3f} scores {m['knee_score']:.2f}; "
              f"k-means scores {m['kmeans_ari']:.2f}")
        """)
    out["dens"] = s.run("""
        for d in unsup.dbscan_densities(spreads=(0.2, 1.6)):
            print(f"spread {d['ratio']:.0f}x: {d['good_radii']:3d} good radii, "
                  f"best {d['best']:.3f}, HDBSCAN {d['hdbscan']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    mo, dens, dg = res["moons"], res["dens"], res["digits"]
    good = [d["good_radii"] for d in dens]
    hd = [d["hdbscan"] for d in dens]
    worst = dens[-1]
    six = dens[-2]
    w = mo["window"]
    pf = mo["perfect"]

    # The spine, asserted rather than trusted.
    assert mo["best"] == 1.0 and mo["kmeans_ari"] < 0.3
    assert w[1] / w[0] < 3.5 and mo["window_min_score"] > 0.95
    assert pf[1] / pf[0] < 2.0
    assert mo["knee"] < w[0] and mo["knee_score"] < 0.6
    lows = [mo["windows"][k][0] for k in (3, 10, 20, 40)]
    assert lows == sorted(lows) and mo["windows"][3] == w
    assert mo["default_clusters"] == 1 and mo["default_score"] < 0.01
    assert good == sorted(good, reverse=True) and good[0] >= 30
    assert good[-1] == 0 and max(six["reps_good_radii"] + [good[-2]]) <= 1
    assert max(worst["reps_good_radii"]) == 0
    assert worst["best"] < 0.75 and worst["wide_noise"] > 0.5
    assert max(worst["reps_best"]) < 0.9
    assert min(hd) > 0.85 and min(worst["reps_hdbscan"]) > 0.8
    assert max(worst["reps_hdbscan"]) < 0.9
    assert 0.65 < dg["best"] < 0.75 and dg["best_clusters"] >= 15
    assert dg["knee_score"] < 0.01 and dg["knee_largest"] > 0.9
    assert 0.1 < dg["knee"] / dg["best_eps"] - 1 < 0.3
    assert dg["hdbscan"] < 0.45 and dg["hdbscan_noise"] > 0.5

    post = Post(
        title=(f"{SERIES_TAG} 58: DBSCAN's Radius Is a Density, and It Gets "
               f"Only One"),
        slug="ml-58-dbscan",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=58,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("DBSCAN found two interlocking half-moons that k-means "
                  "could not, at the right radius. When clusters differ in "
                  "density, no radius is right for all of them, and the "
                  "usual way of choosing it missed even the moons."),
        summary=(
            f"On two half-moons DBSCAN scored an adjusted Rand index of 0.95 "
            f"or more for radii between {w[0]:.3f} and {w[1]:.3f}, and 1.00 "
            f"from {pf[0]:.3f}, where k-means scored "
            f"{mo['kmeans_ari']:.2f}; the radius at the knee of the "
            f"k-distance plot, {mo['knee']:.3f}, scored "
            f"{mo['knee_score']:.2f} in this draw and 0.80 to 0.96 in five "
            f"others. Beside two tight clusters, a third whose "
            f"spread grew from equal to eight times theirs left "
            f"{', '.join(str(g) for g in good)} of 150 radii scoring at "
            f"least 0.9; at eight times the spread, no radius reached 0.9 in "
            f"any of six draws, and the best one in the first called "
            f"{worst['wide_noise']:.0%} of the wide cluster noise. HDBSCAN, "
            f"with no radius, scored {min(worst['reps_hdbscan'] + [worst['hdbscan']]):.2f} "
            f"to {max(worst['reps_hdbscan'] + [worst['hdbscan']]):.2f} there. "
            f"On the digits the best radius, chosen with the labels, scored "
            f"{dg['best']:.2f}, and the knee's radius "
            f"{dg['knee_score']:.2f}."),
        tags=["machine-learning", "unsupervised-learning", "clustering",
              "dbscan", "density", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: scikit-learn's make_moons (1,000 points, noise "
            "0.06, seed 0); two Gaussian clusters of 300 points with "
            "standard deviation 0.2, 1.4 apart, beside a third of 300 points "
            "at distance 6 with standard deviation 0.2 to 1.6 (one random "
            "stream per spread and seed; seed 0 in the table, seeds 1-5 for "
            "the ranges).",
            "UCI optical digits (1,797 images, 64 pixels, CC BY 4.0) via "
            "scikit-learn.",
            "Machinery: scikit-learn's DBSCAN (min_samples 5, counting the "
            "point itself) and HDBSCAN (minimum cluster size 25); the knee "
            "is the sorted fourth-nearest-other-point distance farthest below "
            "the chord; scores are adjusted Rand indices with noise points as "
            "singletons; 'best' radii are chosen with the true labels, which "
            "no user has; `standarderror/ml/unsup.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Ester, Kriegel, Sander and Xu, \"A "
            "density-based algorithm for discovering clusters in large "
            "spatial databases with noise\", *KDD* (1996); Schubert, Sander, "
            "Ester, Kriegel and Xu, \"DBSCAN revisited, revisited\", *ACM "
            "TODS* (2017); Campello, Moulavi and Sander, \"Density-based "
            "clustering based on hierarchical density estimates\", *PAKDD* "
            "(2013); episodes 32, 33 and 56 of this series.",
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
        "Clusters as dense regions",
        """k-means cuts the data into k convex pieces, which is why episode 32 found it cutting two parallel groups crosswise. Density-based clustering starts from a different definition: a cluster is a region where points are packed closely, separated from other such regions by sparse space. DBSCAN, introduced by Ester and colleagues in 1996 and widely used since, makes the definition operational with two settings. A point with at least `min_samples` points, itself included, within radius `eps` is a core point; core points within `eps` of each other share a cluster; anything not reachable from a core point is noise.

Its reputation follows from the definition, and is the sentence under test: *DBSCAN finds clusters of any shape without being told how many*. It does not take k. It does take a radius, and the question is how much the radius decides.

Throughout, agreement with the true groups is the adjusted Rand index with every noise point counted as its own singleton, so that calling a whole cluster noise costs what it should. (Counted as one more cluster, noise can earn a high score by being a cluster's worth of discarded points.) The "best" radius for any data set below is chosen with the true labels, which a user never has; it is the most DBSCAN can do, not what it will do.""")

    post.add(
        "Any shape, at the right radius",
        f"""Two interlocking half-moons are the textbook case, and the textbook is right.

{snip['moons'].markdown()}

k-means, asked for two clusters, cut the moons in half (ARI {mo['kmeans_ari']:.2f}). DBSCAN scored 0.95 or more at every radius between {w[0]:.3f} and {w[1]:.3f}, and separated the moons perfectly from {pf[0]:.3f} to {pf[1]:.3f}. Outside that window, smaller radii shatter each moon into fragments and noise, and larger ones join the two moons where they come close.

The good window is about a factor of {w[1] / w[0]:.1f} wide, the perfect one {pf[1] / pf[0]:.1f}, and something has to find it. Ester and colleagues suggested sorting every point's distance to its k-th nearest neighbour and reading the radius off the knee of that curve, where the distances start to climb. Found as the point farthest below the chord of the sorted curve, the knee was at {mo['knee']:.3f}, just below the window, and that radius scored {mo['knee_score']:.2f}. That miss belonged to this draw of the moons: episode 66 reran it on five other draws, and the knee fell inside the good window every time, scoring 0.80 to 0.96. The heuristic is a reasonable start on data like the moons and, as the digits below show, can still land in the wrong place.

The other setting moves the window more than it widens it. With `min_samples` at 3 or 5 the good radii ran from {mo['windows'][3][0]:.3f} to {mo['windows'][3][1]:.3f}; at 10, from {mo['windows'][10][0]:.3f}; at 20, from {mo['windows'][20][0]:.3f} to {mo['windows'][20][1]:.3f}; at 40, from {mo['windows'][40][0]:.3f} to {mo['windows'][40][1]:.3f}. Requiring more neighbours raises the density a core point needs, so the same moons need a larger radius to reach it: the two settings act roughly as one density threshold written two ways, and fixing either leaves the other to be tuned. scikit-learn's defaults, a radius of 0.5 with 5 points, put both moons into one cluster and scored {mo['default_score']:.2f}; the default radius is a number, not a property of anyone's data.

Schubert and colleagues, revisiting DBSCAN in 2017, give rules of thumb for checking a result: if the largest cluster holds more than 20 to 50% of the clustered points, reduce the radius or use a hierarchical method such as HDBSCAN; noise should usually be between 1 and 30%; and `min_samples` should be about twice the number of dimensions.""")

    post.add(
        "One radius for every density",
        f"""A single radius is a single density threshold, and real clusters differ in density. Two tight clusters, standard deviation 0.2 and 1.4 apart, sit beside a third whose spread grows from 0.2 to 1.6, eight times theirs, which in two dimensions makes it about 64 times less dense. A radius small enough to keep the tight pair apart is too small to hold the wide cluster together; one large enough for the wide cluster merges the pair.

{snip['dens'].markdown()}

With equal spreads, {good[0]} of the 150 radii tried scored at least 0.9. As the third cluster widened, {', '.join(str(g) for g in good[1:])}. Five further draws of the data gave {min(six['reps_good_radii'])} or {max(six['reps_good_radii'])} good radii at sixfold spread and none at eightfold. At eightfold spread the best radius in the first draw scored {worst['best']:.2f}, keeping the tight pair apart by declaring **{worst['wide_noise']:.0%}** of the wide cluster noise, {worst['overall_noise']:.0%} of all points; the other draws' best radii scored {min(worst['reps_best']):.2f} to {max(worst['reps_best']):.2f}. The number of clusters was never the problem; a single density was. Schubert and colleagues' noise rule would have accepted that answer, since a quarter of the points is inside their range.

HDBSCAN, from Campello, Moulavi and Sander in 2013, builds DBSCAN's clusters at every radius at once and keeps those that persist, so it needs a minimum cluster size instead of a radius. It scored {min(hd):.2f} to {max(hd):.2f} across the five spreads in the first draw, and {min(worst['reps_hdbscan']):.2f} to {max(worst['reps_hdbscan']):.2f} at eightfold spread in the other five: better than any single radius at large contrasts, and not perfect either.""",
        figures=[figs["f0"], figs["f1"]])

    post.add(
        "The digits",
        f"""In the digits' 64 pixel dimensions, the best of 60 radii between 5 and 40, chosen with the labels, scored {dg['best']:.2f}, at a radius of {dg['best_eps']:.1f}, by producing {dg['best_clusters']} clusters and calling {dg['best_noise']:.0%} of the images noise: the ten classes broken into denser and sparser pieces. The knee of the k-distance curve was at {dg['knee']:.1f}, and at that radius DBSCAN put {dg['knee_largest']:.0%} of the images into a single cluster and scored **{dg['knee_score']:.2f}**: the radius that looked natural on the curve was just past the point where the classes merge. The two radii differ by {(dg['knee'] / dg['best_eps'] - 1):.0%}, a margin that a reading of a sorted-distance curve is unlikely to resolve reliably. Schubert and colleagues' rules would have caught the knee's answer, a largest cluster far above half the data, but their suggested `min_samples` for 64 dimensions is 128, not 5, so this test departs from their guidance.

HDBSCAN did not rescue it here. It scored {dg['hdbscan']:.2f} and called {dg['hdbscan_noise']:.0%} of the digits noise. (Episode 56 quoted 0.17 for the same clustering because it counted noise as one more cluster; counting noise points as singletons, as here, gives the higher figure.) Episode 33 found that the digits' nearest neighbours behave like those of uniform data in only about eight dimensions, so concentration of distances is not the obvious culprit; why density estimates fail on the raw pixels was not measured. Episode 56 found that the same HDBSCAN on a t-SNE picture of the digits recovered them well, because the picture had already turned neighbourhoods into dense, separated blobs.""")

    post.add(
        "Where this breaks",
        """**Two dimensions, Gaussian clusters.** The density-contrast problem is cleanest in two dimensions; how it changes in more dimensions was not measured beyond the digits.

**One `min_samples`.** Five is scikit-learn's default; Ester and colleagues suggested four for two dimensions. Raising it smooths the density estimate and moves the good window; it does not give one radius two densities.

**One knee rule.** The knee was found by a simple geometric rule. People reading the curve by eye will pick other points; the moons' window was narrow enough that a small difference mattered.

**HDBSCAN's own setting.** Its minimum cluster size plays the part the radius played, with a gentler effect here. It was not swept.""")

    post.add(
        "What to keep",
        f"""1. At the right radius DBSCAN separated two half-moons where k-means scored {mo['kmeans_ari']:.2f}; radii scoring 0.95 or more spanned a factor of {w[1] / w[0]:.1f}, and in this draw the k-distance knee fell just outside them (in five other draws, inside).

2. With one cluster eight times the spread of its neighbours, no radius scored 0.9 in any of six draws; the best in the first called {worst['wide_noise']:.0%} of the wide cluster noise. HDBSCAN scored {min(worst['reps_hdbscan'] + [worst['hdbscan']]):.2f} to {max(worst['reps_hdbscan'] + [worst['hdbscan']]):.2f}.

3. On the digits the knee's radius put {dg['knee_largest']:.0%} of the images in one cluster and scored {dg['knee_score']:.2f}; the best radius, chosen with the labels, {dg['best']:.2f}.

4. Treat `eps` as the density you declare a cluster to have. Sweep it and look at what changes, check the largest cluster and the noise share, and when clusters may differ in density use a method that does not need one radius.""")

    post.add(
        "Exercise",
        """Run DBSCAN on your data at the radius you used, at half of it and at double, and record the number of clusters, the size of the largest and the share of points called noise each time.

If any of them changes a lot, the clustering is a statement about that radius. Look at which clusters split or merge first: they are where the densities differ.""")

    post.add(
        "Next",
        """DBSCAN calls the points in no dense region noise. Anomaly detectors make that the goal: score every point by how little it belongs, and flag the ones that belong least. Episode 59 asks what decides which points those are.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
