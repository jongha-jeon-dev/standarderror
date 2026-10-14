"""ML 57: The BIC Counts Gaussians, Not Clusters.

The fifty-seventh episode of *Machine Learning, Taught Through What Breaks*,
in Arc XII. "The BIC tells you how many clusters a Gaussian mixture should
have." Measured on single clusters of known shape and growing size, on
three uniform squares, and on the digits:

* For one Gaussian cluster the BIC chose one component at every sample
  size. For one uniform square it chose 2, 7, 14 and 18 as the sample grew
  from 200 to 20,000 points; for one heavy-tailed cluster, 2 to 9. ICL, which
  charges for overlapping components, chose one almost every time.
* Three separated uniform squares got nine components from the BIC (ARI
  0.42 against the squares) and three from ICL.
* On the digits reduced to ten principal components, the BIC chose 11
  components with full covariances and 28 with diagonal ones: the count
  depended on the shape the components were allowed.
* With scikit-learn's default k-means start, 90% of single EM runs on
  fifteen clusters reached the likelihood of a start at the true centres;
  with random starts, none did.

Run: `standarderror run ml157_bic --publish`
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

SHAPES = {"gaussian": "one Gaussian", "uniform": "one uniform square",
          "t3": "one Student t, 3 d.f."}


def compute() -> dict:
    return {"growth": unsup.bic_growth(), "squares": unsup.bic_squares(),
            "digits": unsup.bic_digits(), "em": unsup.em_starts()}


# ------------------------------------------------------------------ figures

def _growth(ax, m, g):
    for k, (shape, lab) in enumerate(SHAPES.items()):
        ns = [r["n"] for r in g[shape]]
        ax.plot(ns, [r["bic"] for r in g[shape]], color=m.series[k],
                marker="o", label=f"{lab}: BIC")
        ax.plot(ns, [r["icl"] for r in g[shape]], color=m.series[k],
                marker="x", ls=":", label=f"{lab}: ICL")
    ax.set_xscale("log")
    ax.set_xticks([r["n"] for r in g["gaussian"]])
    ax.set_xticklabels([f"{r['n']:,}" for r in g["gaussian"]])
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)


def figures(res: dict) -> dict:
    out: dict = {}
    g = res["growth"]
    u = g["uniform"]
    out["f0"] = charts.diagram(
        lambda ax, m: _growth(ax, m, g),
        title="One cluster, more data, more components",
        subtitle=("Number of Gaussian components chosen by the BIC and by "
                  "ICL (1 to 20 tried, full covariances), for a single "
                  "two-dimensional cluster of three shapes, as the number "
                  "of points grows."),
        xlabel="points in the cluster (log scale)",
        ylabel="components chosen",
        figsize=(7.6, 4.5),
        source="Simulated; standarderror/ml/unsup.py.",
        alt=("Three solid lines. The Gaussian stays at 1; the uniform square "
             "and the heavy-tailed cluster climb with the number of points. "
             "The dotted ICL lines stay at 1 or 2."),
        caption=(f"One uniform square: the BIC chose {u[0]['bic']} components "
                 f"at {u[0]['n']} points and **{u[-1]['bic']}** at "
                 f"{u[-1]['n']:,}."),
        path=str(IMG / f"ml157-f0-growth.{EXT}"))[0]

    sq, dg = res["squares"], res["digits"]
    body = [["three uniform squares", "full", str(sq["bic"]),
             f"{sq['bic_model_ari']:.2f}", str(sq["icl"]), "3"],
            ["digits, 10 PCs", "full", str(dg["full"]["bic"]),
             f"{dg['full']['ari_at_bic']:.2f}", str(dg["full"]["icl"]), "10"],
            ["digits, 10 PCs", "diagonal", str(dg["diag"]["bic"]),
             f"{dg['diag']['ari_at_bic']:.2f}", str(dg["diag"]["icl"]), "10"]]
    out["f1"] = charts.table_image(
        body,
        header=["data", "covariances", "BIC's k", "ARI at BIC's k",
                "ICL's k", "groups"],
        title="Components chosen against the groups that exist",
        subtitle=("Gaussian mixtures fitted for each k (squares: 1 to 10; "
                  "digits: 2 to 30); the adjusted Rand index of "
                  "the BIC's model against the true groups."),
        source=("Simulated; UCI digits via scikit-learn; "
                "standarderror/ml/unsup.py."),
        alt=("A three-row table. The BIC and ICL disagree on the squares and "
             "agree on the digits, where the count depends on the covariance "
             "shape."),
        caption=(f"With diagonal covariances and k fixed at 10, the digits "
                 f"model scored ARI {dg['diag']['ari_at_10']:.2f}."),
        align="llrrrr",
        path=str(IMG / f"ml157-f1-table.{EXT}"))[0]

    em = res["em"]
    body = []
    for init, lab in (("kmeans", "k-means (default)"),
                      ("k-means++", "k-means++ centres"),
                      ("random_from_data", "random data points"),
                      ("random", "random responsibilities")):
        e = em[init]
        body.append([lab, f"{e['reached']:.0%}", f"{e['ari_median']:.2f}",
                     f"{e['ari_min']:.2f}"])
    body.append(["true centres", "-", f"{em['truth_start']['ari']:.2f}", "-"])
    out["f2"] = charts.table_image(
        body,
        header=["EM started from", "reached the truth's likelihood",
                "median ARI", "worst ARI"],
        title="Fifteen clusters, sixty single starts each",
        subtitle=("Share of single EM runs whose final log-likelihood per "
                  "point came within 0.005 of a run started at the true "
                  "centres, and agreement with the true clusters."),
        source="Simulated; standarderror/ml/unsup.py.",
        alt=("A five-row table. The k-means starts reach the reference in "
             "about 90% of runs; random starts in 5% or none."),
        caption="scikit-learn's GaussianMixture fits one start unless n_init is raised.",
        align="lrrr",
        path=str(IMG / f"ml157-f2-em.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    u = res["growth"]["uniform"]
    sq, dg = res["squares"], res["digits"]

    def p1(panel, m):
        panel.plot([r["n"] for r in u], [r["bic"] for r in u],
                   color=m.series[1], lw=2.5)
        panel.set_xscale("log")

    def p2(panel, m):
        panel.bar([0, 1, 2], [3, sq["bic"], sq["icl"]],
                  color=[m.series[0], m.series[1], m.series[2]])

    def p3(panel, m):
        panel.bar([0, 1], [dg["full"]["bic"], dg["diag"]["bic"]],
                  color=[m.series[0], m.series[1]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=57,
        headline="Counting Gaussians",
        panels=[(p1, f"{u[-1]['bic']}", "components, one square"),
                (p2, f"{sq['bic']}", "for three squares"),
                (p3, f"{dg['full']['bic']} or {dg['diag']['bic']}",
                 "digits, by covariance")],
        note=("The BIC answers how many Gaussians fit the data, which is a "
              "question about shape. With enough data, any cluster that is "
              "not Gaussian gets several."),
        alt=("Three hand-drawn frames: a rising line; a short bar, a tall "
             "one and a short one; a short bar and a tall one."),
        path=str(IMG / f"ml157-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["square"] = s.run("""
        import numpy as np
        from standarderror.ml import unsup

        rng = np.random.default_rng(0)
        for n in (200, 1000):
            X = rng.random((n, 2))                 # one uniform square
            rows = unsup._bic_icl(X, range(1, 11))
            best = min(rows, key=lambda r: r["bic"])
            print(f"n={n:5d}  BIC picks k={best['k']}  "
                  f"(BIC at k=1 minus best: "
                  f"{rows[0]['bic'] - best['bic']:.1f})")
        """)
    out["em"] = s.run("""
        em = unsup.em_starts(runs=10)
        for init in ("kmeans", "random"):
            print(f"{init:7s} reached {em[init]['reached']:.0%}, "
                  f"median ARI {em[init]['ari_median']:.2f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    g, sq, dg, em = res["growth"], res["squares"], res["digits"], res["em"]
    u, t3, ga = g["uniform"], g["t3"], g["gaussian"]
    ub, tb = [r["bic"] for r in u], [r["bic"] for r in t3]
    icl_all = [r["icl"] for s in g.values() for r in s]

    # The spine, asserted rather than trusted.
    assert all(r["bic"] == 1 and r["icl"] == 1 for r in ga)
    assert ub == sorted(ub) and ub[0] <= 2 and ub[-1] >= 15
    assert max(ub) < 20
    assert u[1]["margin"] > 10 and u[3]["margin"] > 100 and t3[3]["margin"] > 10
    assert all(r["margin"] == 0 for r in ga)
    assert tb == sorted(tb) and tb[-1] >= 8
    assert all(r["icl"] == 1 for r in u) and max(icl_all) <= 2
    assert sq["bic"] >= 8 and sq["icl"] == 3 and sq["bic_model_ari"] < 0.5
    assert 10 <= dg["full"]["bic"] <= 13 and dg["diag"]["bic"] >= 24
    assert dg["full"]["icl"] == dg["full"]["bic"]
    assert dg["diag"]["icl"] == dg["diag"]["bic"]
    assert all(r["runner_up_gap"] > 10 for r in u[1:] + t3)
    assert u[0]["runner_up_gap"] < 10
    assert dg["diag"]["ari_at_bic"] < dg["diag"]["ari_at_10"] - 0.1
    assert em["kmeans"]["reached"] >= 0.85 and em["random"]["reached"] == 0
    assert em["random"]["ari_median"] < 0.3 < 0.8 < em["kmeans"]["ari_median"]
    assert em["kmeans"]["ari_median"] < em["truth_start"]["ari"]

    post = Post(
        title=f"{SERIES_TAG} 57: The BIC Counts Gaussians, Not Clusters",
        slug="ml-57-bic",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=57,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Given one cluster shaped like a square, the BIC asked for "
                  "more Gaussian components the more data it saw, 18 at "
                  "20,000 points. It was answering a question about shape "
                  "and sample size, not about clusters."),
        summary=(
            f"Gaussian mixtures fitted to a single two-dimensional cluster: "
            f"for a Gaussian one, the BIC chose one component at every size "
            f"from 200 to 20,000 points; for a uniform square it chose "
            f"{', '.join(str(v) for v in ub)}; for a heavy-tailed cluster "
            f"{', '.join(str(v) for v in tb)}. ICL, which charges for "
            f"overlapping components, chose one or two throughout. Three "
            f"separated squares got {sq['bic']} components from the BIC "
            f"(ARI {sq['bic_model_ari']:.2f}) and {sq['icl']} from ICL. On the "
            f"digits reduced to ten principal components the BIC chose "
            f"{dg['full']['bic']} components with full covariances and "
            f"{dg['diag']['bic']} with diagonal ones. With k-means starts, "
            f"{em['kmeans']['reached']:.0%} of single EM runs reached the "
            f"likelihood of a start at the truth; with random starts, none."),
        tags=["machine-learning", "unsupervised-learning", "clustering",
              "gaussian-mixture", "model-selection", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated (seed 0): single two-dimensional clusters (standard "
            "Gaussian, unit uniform square, Student t with 3 degrees of "
            "freedom) of 200 to 20,000 points; three unit squares of 500 "
            "points at (0,0), (2,0) and (1,2); fifteen unit-variance "
            "Gaussian clusters of 100 points with centres uniform on "
            "[-20, 20]^2.",
            "UCI optical digits (1,797 images, CC BY 4.0) via scikit-learn, "
            "reduced to ten principal components.",
            "Machinery: scikit-learn's GaussianMixture (three starts per k "
            "for the simulations, two for the digits, unless stated); "
            "`standarderror/ml/unsup.py`, tested in `tests/test_ml.py`.",
            "Where this stops: Schwarz, \"Estimating the dimension of a "
            "model\", *Annals of Statistics* (1978); Fraley and Raftery, "
            "\"Model-based clustering, discriminant analysis, and density "
            "estimation\", *JASA* (2002); Biernacki, Celeux and Govaert, "
            "\"Assessing a mixture model for clustering with the integrated "
            "completed likelihood\", *IEEE TPAMI* (2000); Baudry, Raftery, "
            "Celeux, Lo and Gottardo, \"Combining mixture components for "
            "clustering\", *JCGS* (2010); Kass and Raftery, \"Bayes factors\", "
            "*JASA* (1995); episode 32 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; about four minutes of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A criterion for the number of groups",
        f"""Episode 32 found that k-means returns k pieces whatever the data, and that the silhouette, asked to choose k, chose four for uniform noise. A Gaussian mixture looks like the principled answer. It is a probability model: the data are drawn from k Gaussian components, EM fits their means, covariances and weights by maximum likelihood, and the Bayesian information criterion (Schwarz, 1978) trades the likelihood against the number of parameters to choose k. Model-based clustering, as Fraley and Raftery set it out, made BIC the standard way to pick the number of components, and the practice is to read that number as the number of clusters: *the BIC tells you how many clusters a Gaussian mixture should have*.

The test needs clusters whose number is known and whose shape is not Gaussian, because real clusters seldom are. The simplest is one cluster: a uniform square, or a heavy-tailed blob, with a Gaussian blob as the control. The number of clusters is one in every case. The question is what the BIC says as the data grow.

{snip['square'].markdown()}""")

    post.add(
        "More data, more components",
        f"""For the Gaussian blob the BIC chose one component at every size, from 200 points to 20,000. For the uniform square it chose {ub[0]} at 200 points, {ub[1]} at 1,000, {ub[2]} at 5,000 and **{ub[3]}** at 20,000. For a Student t with three degrees of freedom, a single cluster with heavy tails, it chose {', '.join(str(v) for v in tb)}.

None of this is a bug in the BIC. A square is not a mixture of any finite number of Gaussians, so there is no true k for the BIC to find: the flat top and sharp edges are better described by a patchwork of components, and the evidence for each extra patch grows in proportion to the number of points while the BIC's penalty grows only with its logarithm. With enough data, every departure from Gaussian shape earns more components, without limit. The criterion counts Gaussians, and the number it reports for a non-Gaussian cluster depends on the cluster's shape and on how much data there is.

From 1,000 points on these were not close calls. A difference in BIC approximates twice the log Bayes factor between two models, and Kass and Raftery's scale calls a difference above 10 very strong evidence. For the square, the chosen k beat the next-best k by {u[1]['runner_up_gap']:.0f} units at 1,000 points, {u[2]['runner_up_gap']:.0f} at 5,000 and {u[3]['runner_up_gap']:.0f} at 20,000, and beat a single component by {u[1]['margin']:,.0f}, {u[2]['margin']:,.0f} and {u[3]['margin']:,.0f}. For the heavy-tailed blob the next-best gaps were {', '.join(f"{r['runner_up_gap']:.0f}" for r in t3)}. At 200 points the square's choice was not decisive: k = 2 beat k = 1 by {u[0]['runner_up_gap']:.1f}, and a different draw of 200 points, in the code above, chose another k. An analyst who checks that the BIC's choice is decisive, with a few thousand points, will find that it is, and will be no closer to the number of clusters.

ICL, proposed by Biernacki, Celeux and Govaert in 2000, adds a charge for how uncertain the assignment of points to components is. Components that tile one cluster overlap heavily and are expensive; components on separate clusters are not. ICL chose one component for the square at every size, and one or two for the heavy-tailed blob.""",
        figures=[figs["f0"]])

    post.add(
        "Three squares, nine components",
        f"""With three separated squares of 500 points each, the BIC chose {sq['bic']} components and its model agreed with the squares at an adjusted Rand index of {sq['bic_model_ari']:.2f}: each square carved into pieces. ICL chose {sq['icl']}. A common repair, from Baudry and colleagues in 2010, keeps the BIC's components and merges them hierarchically into clusters, which separates the two questions the BIC was being asked to answer at once: how to describe the density, and how many groups it has.

The digits show the same dependence in real data. Reduced to ten principal components and given full covariance matrices, the BIC, tried at every k from 2 to 30, chose {dg['full']['bic']} components, one more than the ten classes, with an ARI of {dg['full']['ari_at_bic']:.2f} (at k = 10, {dg['full']['ari_at_10']:.2f}). Restricted to diagonal covariances, components that cannot tilt, it chose {dg['diag']['bic']}, with an ARI of {dg['diag']['ari_at_bic']:.2f}; the same diagonal model at k = 10 scored {dg['diag']['ari_at_10']:.2f}. Neither number is the number of digit classes in any deeper sense than the other. Each is the number of components of the allowed shape that the data can pay for.""",
        figures=[figs["f1"]])

    post.add(
        "The start does much of the work",
        f"""Every k on these curves came from EM, which climbs to a local maximum of the likelihood from wherever it starts, and scikit-learn's GaussianMixture runs one start unless told otherwise. On fifteen overlapping clusters, with a start at the true centres as the reference, the default start from k-means reached the reference's likelihood (within 0.005 per point) in {em['kmeans']['reached']:.0%} of sixty single runs, with a median ARI of {em['kmeans']['ari_median']:.2f} against {em['truth_start']['ari']:.2f} for the reference. Starting from randomly chosen data points reached it in {em['random_from_data']['reached']:.0%}, with a median ARI of {em['random_from_data']['ari_median']:.2f} and a worst run at {em['random_from_data']['ari_min']:.2f}, where the worst k-means start still scored {em['kmeans']['ari_min']:.2f}; starting from random responsibilities, in **none**, with a median ARI of {em['random']['ari_median']:.2f}.

{snip['em'].markdown()}

(The code runs ten starts of each kind, the first ten of the sixty, so its shares are coarser.)

So a mixture's "maximum likelihood" is the maximum near where k-means put it. That is usually a good place, and it means a BIC curve is a curve of local optima: a k that EM fits badly looks worse than it is. Several starts per k, and checking that the best likelihood repeats, is cheap insurance.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        """**Two-dimensional shapes.** In more dimensions a non-Gaussian cluster can depart from Gaussian shape in more ways, so the number of components the BIC can justify plausibly grows faster; that was not measured beyond the digits.

**ICL is not a cure.** It penalises overlap, so it under-counts clusters that genuinely overlap, which is where mixture models are most useful. On the digits it chose the same k as the BIC with both covariance shapes.

**k up to 20, three starts each.** More starts can only raise each k's likelihood, which could move the chosen k by a component or two; that was not measured.

**Other mixtures.** Mixtures of t or skewed distributions describe heavy-tailed or asymmetric clusters with fewer components; they move the problem to whatever shape they assume.""")

    post.add(
        "What to keep",
        f"""1. For one cluster that was not Gaussian, the BIC chose more components the more data it had: {ub[0]} to {ub[-1]} for a square as the points grew from 200 to 20,000. For a Gaussian cluster it chose one throughout.

2. Three separated squares got {sq['bic']} components from the BIC and {sq['icl']} from ICL; the digits got {dg['full']['bic']} or {dg['diag']['bic']} depending on the covariance shape allowed.

3. With one start per fit, an initialisation from random responsibilities never reached the likelihood that k-means initialisation reached {em['kmeans']['reached']:.0%} of the time.

4. Read the BIC's k as the number of components of the allowed shape. For a number of clusters, use a criterion that charges for overlap or merge components afterwards, and fit several starts per k.""")

    post.add(
        "Exercise",
        """Fit your mixture model at the k the BIC chose, then subsample your data to a quarter and choose k again.

If the chosen k falls with the sample, the extra components were describing shape that only a large sample can afford to describe. Look at which components overlap most and ask whether they are one group.""")

    post.add(
        "Next",
        """A mixture assumes a shape. Density-based clustering assumes none: DBSCAN calls a cluster any region where points are packed closely enough, and claims to find clusters of any shape without being told how many. Episode 58 asks what "closely enough" decides.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
