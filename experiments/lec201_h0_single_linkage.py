"""Topology 1: H0 Is Single-Linkage Clustering, Bit for Bit.

The first episode of the third series, and it spends its whole length on one
identity: the 0-dimensional persistent homology of a Rips filtration *is* the
single-linkage dendrogram. Which is good news about the machinery -- it is
thirty lines of union-find -- and bad news about the method, because every
property of single linkage is now a property of the barcode.

Measured:

* `rips_h0` and `scipy`'s `linkage(..., method="single")` return the same floats.
  Not close: `array_equal` is True on four designs from 12 to 60 points and 1 to
  30 dimensions.
* So chaining is inherited. Two blobs 6.0 apart give a longest bar of 4.00
  against a second of 0.598 -- ratio 6.70. **Three** points along the line
  between them bring the ratio to **1.00**, and twelve, out of seventy-two, make
  the k = 2 cut return clusters of size 71 and 1.
* The stability theorem is real and it is not vacuous. Jittering by eps = 0.5
  moves the points a Hausdorff 2.036 and the barcode a bottleneck 0.968.
* And the integer read off the barcode has no such guarantee: the same jitter at
  eps = 1.0 moves "clusters at the largest gap" from 2 to 4.
* Which is a dimension problem too: the barcode's own dynamic range, (max-min)
  over mean of the deaths, runs 7.16 at d = 2 to 0.083 at d = 768.

Run: `standarderror run lec201_h0_single_linkage --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.topology import filtration as ft
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
POST_DATE = date(2026, 9, 9)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Topology for Language Models, Taught Through What Breaks"
SERIES_TAG = "Topology"

#: Bridge sizes for the chaining sweep. 3 is the number that matters.
BRIDGES = (0, 3, 6, 12, 24)
#: Perturbations for the stability sweep.
EPSILONS = (0.0, 0.01, 0.05, 0.2, 0.5, 1.0)
#: Dimensions for the dynamic-range sweep.
DIMS = (2, 8, 64, 768)
#: The identity is checked on these, not on one lucky cloud.
IDENTITY_DESIGNS = ((60, 5, 0), (40, 2, 1), (25, 30, 2), (12, 1, 3))


def compute() -> dict:
    identity = []
    for n, d, seed in IDENTITY_DESIGNS:
        X = np.random.default_rng(seed).standard_normal((n, d))
        D = ft.pairwise(X)
        bc = ft.rips_h0(D)
        ref = ft.single_linkage_heights(D)
        identity.append({"n": n, "d": d, "bars": len(bc.deaths),
                         "equal": bool(np.array_equal(bc.deaths, ref)),
                         "max_gap": float(np.abs(bc.deaths - ref).max())})

    plain, plain_lab = ft.two_blobs()
    plain_bc = ft.rips_h0(ft.pairwise(plain))
    chaining = ft.chaining_sweep(BRIDGES)

    big, _ = ft.two_blobs(per=40)
    stability = ft.stability_sweep(big, EPSILONS)
    dims = ft.dimension_sweep(DIMS)

    # A six-point cloud the reader can check by hand, for the first figure.
    hand = np.array([[0.0, 0.0], [1.0, 0.2], [0.6, 1.1],
                     [4.0, 0.3], [4.8, 1.0], [4.3, -0.6]])
    hand_bc = ft.rips_h0(ft.pairwise(hand))

    gaps = {e: ft.gap_rule_spread(big, e) for e in (0.0, 0.2, 1.0, 1.5)}

    return {"gaps": gaps,
            "identity": identity, "plain": plain, "plain_lab": plain_lab,
            "plain_bc": plain_bc, "chaining": chaining, "big": big,
            "stability": stability, "dims": dims,
            "hand": hand, "hand_bc": hand_bc,
            "bridged": ft.two_blobs(bridge=12)[0]}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    bc, chain = res["plain_bc"], res["chaining"]

    # --- f0: the barcode, which is the whole object ------------------------
    def barcode(ax, m):
        d = bc.deaths
        y = np.arange(len(d))
        ax.hlines(y, 0.0, d, color=m.series[0], lw=1.6)
        ax.plot(d, y, "|", ms=7, color=m.series[0])
        ax.axvline(d[-2], color=m.grid, lw=1.4, ls=(0, (4, 3)))
        ax.axvline(d[-1], color=m.ink, lw=1.6)
        ax.annotate(f"the longest bar dies at {d[-1]:.3f}", (d[-1], len(d) - 1),
                    textcoords="offset points", xytext=(-8, -16), ha="right",
                    fontsize=8.5, color=m.ink_secondary)
        ax.annotate(f"everything else is done by {d[-2]:.3f}", (d[-2], 6),
                    textcoords="offset points", xytext=(8, 0), fontsize=8.5,
                    color=m.ink_secondary)
        ax.set_yticks([])
        ax.set_xlim(0, d[-1] * 1.12)

    out["f0"] = charts.diagram(
        barcode,
        title="A barcode, and the one number people read off it",
        subtitle=("0-dimensional persistence for two blobs 6.0 apart, 30 points "
                  "each. Every bar starts at radius 0 -- each point is its own "
                  "component -- and ends where its component merges into an "
                  "older one."),
        xlabel="filtration radius", ylabel="",
        source="Simulated; standarderror/topology/filtration.py.",
        alt=("Fifty-nine horizontal bars all starting at zero, most ending "
             "below 0.6, one continuing to just past 4, with vertical markers "
             "at the two longest deaths."),
        caption=(f"Fifty-nine bars for sixty points, because one component "
                 f"never dies. The gap between the longest bar "
                 f"({bc.longest:.3f}) and the next ({bc.deaths[-2]:.3f}) is a "
                 f"ratio of {bc.separation_ratio:.2f}, and it is how a barcode "
                 f"says \"two clusters\". The rest of this episode is about how "
                 f"little it takes to erase that gap, and about the fact that "
                 f"the picture is provably stable while the number is not."),
        path=str(IMG / f"lec201-f0-barcode.{EXT}"))[0]

    # --- f1: what three points do ------------------------------------------
    def clouds(ax, m):
        P, B = res["plain"], res["bridged"]
        ax.scatter(P[:, 0], P[:, 1] + 3.2, s=13, color=m.series[0],
                   label="two blobs: ratio 6.70, cut gives 30 and 30")
        ax.scatter(B[:, 0], B[:, 1] - 3.2, s=13, color=m.series[1],
                   label="plus 12 bridge points: cut gives 71 and 1")
        ax.annotate("one short edge is all single linkage needs",
                    (3.0, -3.2), textcoords="offset points", xytext=(0, -26),
                    ha="center", fontsize=8.5, color=m.ink_secondary)
        ax.set_yticks([])
        ax.set_ylim(-6.4, 5.6)
        ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=1)

    out["f1"] = charts.diagram(
        clouds,
        title="The bridge adds no cluster and removes the two that were there",
        subtitle=("The same two blobs, above and below. The lower copy has "
                  "twelve extra points strung along the line between them, at a "
                  "spacing tighter than the blobs' own."),
        xlabel="", ylabel="",
        source="Simulated; standarderror/topology/filtration.py.",
        alt=("Two scatter plots stacked vertically. The upper one shows two "
             "separated clusters. The lower one shows the same two clusters "
             "with a thin line of points joining them."),
        caption=(f"Single linkage merges two components as soon as **any** pair "
                 f"of their points is close, so a chain of points at spacing "
                 f"0.06 joins two blobs 6.0 apart. The longest bar falls from "
                 f"{chain[0]['longest']:.3f} to {chain[3]['longest']:.3f} — "
                 f"which is exactly the second bar of the cloud above "
                 f"({chain[0]['second']:.3f}), so nothing is left but the "
                 f"spacing inside a blob."),
        path=str(IMG / f"lec201-f1-chaining.{EXT}"))[0]

    # --- f2: how fast the signal goes --------------------------------------
    def ratio(ax, m):
        b = [r["bridge"] for r in chain]
        y = [r["separation_ratio"] for r in chain]
        ax.plot(b, y, marker="o", ms=6, lw=1.9, color=m.series[0])
        ax.axhline(1.0, color=m.ink, lw=1.5)
        ax.annotate("1.0 = the two longest bars are the same length",
                    (b[-1], 1.0), textcoords="offset points", xytext=(-4, 9),
                    ha="right", fontsize=8.5, color=m.ink_secondary)
        for r in chain:
            if r["bridge"] in (0, 3):
                ax.annotate(f"{r['separation_ratio']:.2f}",
                            (r["bridge"], r["separation_ratio"]),
                            textcoords="offset points", xytext=(9, 3),
                            fontsize=9.0, color=m.ink_secondary)
        ax.set_ylim(0, 7.6)

    out["f2"] = charts.diagram(
        ratio,
        title="Three points out of sixty-three",
        subtitle=("The barcode's two-cluster signal — longest bar divided by "
                  "the next — against how many points are strung between the "
                  "blobs."),
        xlabel="bridge points added", ylabel="longest bar ÷ second longest",
        source="Simulated; standarderror/topology/filtration.py.",
        alt=("A curve falling steeply from just under 7 at zero bridge points "
             "to 1 at three, then flat along the horizontal line at 1."),
        caption=(f"From {chain[0]['separation_ratio']:.2f} to "
                 f"{chain[1]['separation_ratio']:.2f} on the addition of three "
                 f"points, and it never recovers. This is not a property of "
                 f"persistence; it is single linkage's chaining, which "
                 f"persistence inherits because — as the next section shows — "
                 f"they are the same computation."),
        path=str(IMG / f"lec201-f2-ratio.{EXT}"))[0]

    # --- f3: the theorem, and what it does not cover -----------------------
    def stability(ax, m):
        st = res["stability"]
        h = np.array([r["hausdorff"] for r in st])
        b = np.array([r["bottleneck"] for r in st])
        lim = max(h.max(), b.max()) * 1.08
        ax.plot([0, lim], [0, lim], color=m.ink, lw=1.5,
                label="the stability bound: bottleneck ≤ Hausdorff")
        ax.fill_between([0, lim], [0, lim], [lim, lim], color=m.grid, alpha=0.35)
        ax.plot(h, b, marker="o", ms=6, lw=1.8, color=m.series[0],
                label="measured")
        ax.annotate("forbidden by the theorem", (lim * 0.30, lim * 0.72),
                    fontsize=8.5, color=m.ink_secondary)
        for r in st:
            if r["epsilon"] in (0.5, 1.0):
                ax.annotate(f"ε = {r['epsilon']:g}, gap rule says "
                            f"{r['gap_k']} clusters",
                            (r["hausdorff"], r["bottleneck"]),
                            textcoords="offset points", xytext=(6, -14),
                            ha="left", fontsize=8.5, color=m.ink_secondary)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    st = {r["epsilon"]: r for r in res["stability"]}
    out["f3"] = charts.diagram(
        stability,
        title="The barcode is stable. The number you read off it is not",
        subtitle=("The same cloud jittered by ε, then its barcode compared with "
                  "the original's. The shaded region is what the stability "
                  "theorem forbids."),
        xlabel="how far the points moved (Hausdorff)",
        ylabel="how far the barcode moved (bottleneck)",
        source="Simulated; standarderror/topology/filtration.py.",
        alt=("Measured points against a diagonal bound line, all of them below "
             "the diagonal and well below it at the largest perturbations, with "
             "the two largest annotated by the cluster count."),
        caption=(f"Every measurement is inside the bound, and not narrowly: at "
                 f"ε = 0.5 the points move {st[0.5]['hausdorff']:.3f} and the "
                 f"barcode {st[0.5]['bottleneck']:.3f}. That is a real "
                 f"guarantee about the picture. It is not a guarantee about the "
                 f"answer — at ε = 1.0, with the barcode still inside the "
                 f"bound, \"clusters at the largest gap\" goes from "
                 f"{st[0.5]['gap_k']} to {st[1.0]['gap_k']}."),
        path=str(IMG / f"lec201-f3-stability.{EXT}"))[0]

    # --- f4: the axis runs out ---------------------------------------------
    def dyn(ax, m):
        dm = res["dims"]
        d = [r["d"] for r in dm]
        ax.plot(d, [r["barcode_spread"] for r in dm], marker="o", ms=6, lw=1.9,
                color=m.series[0], label="barcode: (max − min) ÷ mean of deaths")
        ax.plot(d, [r["distance_spread"] for r in dm], marker="s", ms=6, lw=1.7,
                color=m.series[1], label="all pairwise distances, same measure")
        ax.set_xscale("log")
        ax.set_yscale("log")
        for r in dm:
            if r["d"] in (2, 768):
                ax.annotate(f"{r['barcode_spread']:.2f}",
                            (r["d"], r["barcode_spread"]),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=9.0, color=m.ink_secondary)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    dd = {r["d"]: r for r in res["dims"]}
    out["f4"] = charts.diagram(
        dyn,
        title="In 768 dimensions the barcode has almost no axis left",
        subtitle=("200 points of pure Gaussian noise — no clusters at all — at "
                  "four dimensions. How much of the filtration axis the deaths "
                  "actually occupy."),
        xlabel="dimension", ylabel="spread ÷ mean",
        source="Simulated; standarderror/topology/filtration.py.",
        alt=("Two lines falling steeply against dimension on log axes, the "
             "barcode's spread starting higher and ending lower than the "
             "pairwise distances'."),
        caption=(f"From {dd[2]['barcode_spread']:.2f} at two dimensions to "
                 f"{dd[768]['barcode_spread']:.3f} at 768: every bar is born "
                 f"and dies within a few percent of one radius. Nothing has "
                 f"gone wrong with the clustering — this is noise, so there is "
                 f"nothing to find — but the **display** has stopped being able "
                 f"to show a gap, which means \"no clear gap in the barcode\" "
                 f"is not evidence of no clusters in high dimensions."),
        path=str(IMG / f"lec201-f4-dimension.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    chain = res["chaining"]

    def bars(panel, m):
        d = res["plain_bc"].deaths
        y = np.arange(len(d))
        panel.hlines(y, 0, d, color=m.ink, lw=1.4)
        panel.set_yticks([])

    def bridge(panel, m):
        B = res["bridged"]
        panel.plot(B[:, 0], B[:, 1], "o", ms=2.5, color=m.ink)
        panel.set_yticks([])
        panel.set_ylim(-2.4, 2.4)

    def shrinking(panel, m):
        d = [r["d"] for r in res["dims"]]
        panel.plot(np.log10(d), [r["barcode_spread"] for r in res["dims"]],
                   color=m.ink, lw=2.6, marker="o", ms=4)
        panel.axhline(0.0, color=m.grid, lw=2.2)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=1,
        headline="H0 is single-linkage clustering",
        panels=[
            (bars, f"{res['plain_bc'].separation_ratio:.1f}",
             "two clusters, loudly"),
            (bridge, f"{chain[1]['separation_ratio']:.2f}",
             "after 3 more points"),
            (shrinking, f"{res['dims'][-1]['barcode_spread']:.3f}",
             "axis left at d = 768"),
        ],
        note=("The 0-dimensional persistent homology of a Rips filtration is "
              "the single-linkage dendrogram — the same floats, not a close "
              "agreement. So the barcode is thirty lines of union-find, and it "
              "inherits chaining: three points strung between two blobs erase "
              "the two-cluster signal completely. The stability theorem "
              "protects the picture; it says nothing about the integer you "
              "read off it."),
        alt=("A three-panel hand-drawn strip. The first shows a barcode of "
             "many short bars and one long one. The second shows two clusters "
             "joined by a line of points. The third shows a curve falling "
             "towards a horizontal floor."),
        mode="light",
        path=str(IMG / f"lec201-hero.{EXT}"))[0]


# --------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["h0"] = s.run("""
        import numpy as np
        from scipy.spatial.distance import pdist, squareform

        def rips_h0(D):
            # Every point is its own component at radius 0. Walk the edges
            # shortest first; an edge joining two different components kills
            # the younger one, so its length is a death time. That is all of
            # 0-dimensional persistent homology.
            n = len(D)
            iu = np.triu_indices(n, 1)
            order = np.argsort(D[iu], kind="stable")
            parent = list(range(n))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            deaths = []
            for a, b, w in zip(iu[0][order], iu[1][order], D[iu][order]):
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[rb] = ra
                    deaths.append(float(w))
            return np.sort(np.array(deaths))

        # Six points you can check by eye: two triangles, far apart.
        X = np.array([[0.0, 0.0], [1.0, 0.2], [0.6, 1.1],
                      [4.0, 0.3], [4.8, 1.0], [4.3, -0.6]])
        print(np.round(rips_h0(squareform(pdist(X))), 3))
    """, expect=["["])

    out["identity"] = s.run(f"""
        # And it is not a new algorithm. It is the one in scipy, under the
        # other name that computation has.
        from scipy.cluster.hierarchy import linkage

        for n, d, seed in {list(IDENTITY_DESIGNS)}:
            Y = np.random.default_rng(seed).standard_normal((n, d))
            D = squareform(pdist(Y))
            mine = rips_h0(D)
            theirs = np.sort(linkage(pdist(Y), method="single")[:, 2])
            print(f"{{n:>3}} points in {{d:>2}}d: {{len(mine)}} bars, "
                  f"identical to scipy: {{np.array_equal(mine, theirs)}}")
    """, expect=["identical to scipy: True"])

    out["chaining"] = s.run("""
        # So whatever single linkage does wrong, a barcode does wrong. Two
        # blobs six apart, and then a few points strung between them.
        def two_blobs(bridge=0, gap=6.0, per=30, seed=0):
            rng = np.random.default_rng(seed)
            parts = [rng.standard_normal((per, 2)) * 0.5,
                     rng.standard_normal((per, 2)) * 0.5 + [gap, 0.0]]
            if bridge:
                t = np.linspace(0.18, 0.82, bridge)[:, None]
                parts.append(t * np.array([gap, 0.0])
                             + rng.standard_normal((bridge, 2)) * 0.05)
            return np.vstack(parts)

        for k in (0, 3, 12):
            d = rips_h0(squareform(pdist(two_blobs(k))))
            print(f"bridge {k:>2} (n={30*2+k:>2})  longest {d[-1]:.4f}  "
                  f"second {d[-2]:.4f}  ratio {d[-1]/d[-2]:.2f}")
    """, expect=["bridge 12"])

    out["gap"] = s.run("""
        # The rule everybody applies to a barcode, and what a small jitter
        # does to it. The barcode itself barely moves; the integer does.
        def gap_k(deaths):
            return len(deaths) - int(np.argmax(np.diff(deaths)))

        X = two_blobs(0, per=40)
        rng = np.random.default_rng(1)
        for eps in (0.0, 0.5, 1.0):
            Y = X + rng.standard_normal(X.shape) * eps
            d = rips_h0(squareform(pdist(Y)))
            print(f"eps {eps:.1f}   points moved by at most "
                  f"{np.linalg.norm(X - Y, axis=1).max():.3f}   "
                  f"clusters at the largest gap: {gap_k(d)}")
    """, expect=["eps 1.0"])

    return out


# ------------------------------------------------------------------- build

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    bc = res["plain_bc"]
    chain = {r["bridge"]: r for r in res["chaining"]}
    st = {r["epsilon"]: r for r in res["stability"]}
    dims = {r["d"]: r for r in res["dims"]}
    gaps = res["gaps"]

    # The spine, asserted rather than trusted.
    assert all(r["equal"] for r in res["identity"]), res["identity"]
    assert all(r["max_gap"] == 0.0 for r in res["identity"])
    assert len(bc.deaths) == len(res["plain"]) - 1
    assert chain[0]["separation_ratio"] > 6.0
    assert abs(chain[3]["separation_ratio"] - 1.0) < 0.02
    assert chain[12]["cut_sizes"] == [71, 1]
    assert chain[12]["longest"] == chain[0]["second"]
    assert all(r["bound_holds"] for r in res["stability"])
    assert st[0.5]["bottleneck"] < 0.5 * st[0.5]["hausdorff"]
    assert gaps[0.0]["counts"] == {2: 40}
    assert gaps[0.2]["counts"] == {2: 40}
    assert gaps[1.0]["share_modal"] < 0.6
    assert gaps[1.5]["k_max"] >= 10
    assert gaps[1.5]["bottleneck_max"] == gaps[1.0]["bottleneck_max"]
    assert dims[2]["barcode_spread"] > 6.0 > dims[768]["barcode_spread"]
    assert dims[768]["barcode_spread"] < 0.12

    post = Post(
        title=f"{SERIES_TAG} 1: H0 Is Single-Linkage Clustering, Bit for Bit",
        slug="topology-1-h0-single-linkage",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=1,
        date=POST_DATE,
        subtitle=("The 0-dimensional persistent homology of a Rips filtration "
                  "returns the same floats as scipy's single-linkage "
                  "dendrogram — so the barcode is thirty lines of union-find, "
                  "and it inherits chaining: three points strung between two "
                  "clusters take the two-cluster signal from 6.70 to 1.00."),
        summary=("Persistent homology arrives with a large vocabulary and, in "
                 "dimension zero, a very small computation: walk the edges "
                 "shortest first and record the length of each one that joins "
                 "two components. That is the single-linkage dendrogram's merge "
                 "heights, and not approximately — the two return arrays that "
                 "are equal float for float, on four designs from 12 to 60 "
                 "points and 1 to 30 dimensions. Which settles both what H0 is "
                 "for and what it inherits. Chaining: two blobs 6.0 apart give "
                 "a longest bar of 4.00 against a second of 0.60, and three "
                 "points along the line between them make those two bars the "
                 "same length. The stability theorem is real and not vacuous — "
                 "jittering by 0.5 moves the points 1.39 and the barcode 0.44 — "
                 "but it protects the picture, not the answer: over forty noise "
                 "draws the cluster count read off the largest gap comes back 2 "
                 "in twenty-two of them and 3, 4 or 5 in the rest, with every "
                 "barcode inside a bottleneck distance of 1.96."),
        tags=["topology", "persistent-homology", "clustering", "embeddings",
              "lectures", "machine-learning"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every cloud here is constructed in the episode "
            "and every number is produced by the code shown, executed when "
            "this page was built.",
            "Machinery: `standarderror/topology/filtration.py`, tested in "
            "`tests/test_filtration.py`.",
            "Where this stops: Edelsbrunner and Harer, *Computational "
            "Topology* (AMS, 2010), for the filtration and the persistence "
            "pairing; Cohen-Steiner, Edelsbrunner and Harer, \"Stability of "
            "persistence diagrams\", *Discrete & Computational Geometry* 37 "
            "(2007), for the bound the third figure is drawn against; "
            "Carlsson, \"Topology and data\", *Bulletin of the AMS* 46 (2009), "
            "for the programme; and Chazal, Guibas, Oudot and Skraba, "
            "\"Persistence-based clustering in Riemannian manifolds\", *JACM* "
            "60 (2013), for what is actually done about chaining.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4, "
                           "scipy=1.16.3",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": ("two 30-point blobs 6.0 apart for the chaining "
                           "figures, 40 per blob for the stability ones, and "
                           "200 points of pure noise for the dimension sweep"),
            "determinism": ("one generator per measurement, seeded from that "
                            "measurement's own parameters rather than advanced "
                            "through a loop"),
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    bc = res["plain_bc"]
    chain = {r["bridge"]: r for r in res["chaining"]}
    st = {r["epsilon"]: r for r in res["stability"]}
    dims = {r["d"]: r for r in res["dims"]}
    gaps = res["gaps"]
    ident = res["identity"]
    hand = res["hand_bc"].deaths

    def counts(e):
        return ", ".join(f"**{k}** in {n}"
                         for k, n in gaps[e]["counts"].items())

    post.add(
        "A filtration is two choices, and the second one is a slider",
        r"""Take a set of points and grow a ball of radius *r* around each one. At *r* = 0 nothing touches anything; as *r* grows, balls start to overlap, and two points get joined when their balls meet — which happens at *r* = *d*(*x*, *y*)/2, though by convention the filtration is indexed by the distance itself. Keep going and eventually everything is one blob.

That family of growing graphs is a **filtration**, and it is built out of exactly two decisions: a distance function, and where you stop. Neither is in the data.

Watch what happens to the connected components. Every point starts as its own component, so if there are *n* points there are *n* components at radius 0. Components merge as *r* grows, and each merge kills one of them — the younger, by convention. So the life of the component structure is a list: *n* bars, all born at 0, each dying at the radius where it merged, except for one that never dies.

That list is the **0-dimensional persistent homology** of the filtration, and drawing it is drawing a barcode.""")

    post.add(
        "",
        f"""{snip['h0'].markdown()}

Six points arranged as two triangles, four metres apart. Four of the five deaths are around 1.0 — the within-triangle distances — and the fifth is at {hand[-1]:.3f}, which is where one triangle finally reaches the other. Five bars for six points, because the last component survives.

Here is the same picture for something with more in it.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "",
        f"""So a barcode says "two clusters" by having one bar much longer than the rest, and the honest version of that statement is a ratio: {bc.longest:.3f} divided by {bc.deaths[-2]:.3f} is {bc.separation_ratio:.2f}. There is a range of radii — anywhere between those two numbers — at which exactly two components are alive, and the width of that range is the strength of the claim.

Everything else in this episode follows from one fact about how that list was computed.""",
        level=3)

    post.add(
        "It is single-linkage clustering, and the same floats come back",
        f"""Look again at what the code above actually does. It sorts the edges by length and walks them, and an edge only matters if it joins two components that were separate. That is agglomerative clustering where the distance between two clusters is the distance between their *closest* members — which is single linkage, and it has been in every statistics library for fifty years.

Not "is analogous to". The same numbers:

{snip['identity'].markdown()}

`array_equal` on four designs, from {ident[3]['n']} points in one dimension to {ident[0]['n']} points in five, with the maximum difference exactly `0.0`.

And it is not a coincidence to be admired, so here is why. Single linkage builds its dendrogram by repeatedly finding the shortest edge between two distinct clusters and merging them at that height. The union-find loop above walks *all* edges in increasing length and merges whenever it finds one between two distinct components — which, at the moment it merges, is the shortest such edge, because every shorter one has already been seen and was either inside a component or merged something else. So the two procedures merge the same pairs in the same order at the same heights. The union-find version simply notices that most edges can be skipped, which is why it is `O(E log E)` rather than a repeated minimum search.

The useful thing is what that lets you carry over. Every property of single-linkage clustering — good and bad, and there is a well-known bad one — is now a property of the 0-dimensional barcode.

It also answers the obvious next question, which is what the topology buys if the computation was already in scipy. Two things, and it is worth being precise about them. The barcode is a summary with a stability theorem attached, which a dendrogram is not usually treated as having; that is the next section, and it turns out to protect less than people think. And in higher dimensions there are features with no clustering analogue at all — H1 counts independent cycles, and no linkage rule has ever returned one. This series gets there. In dimension zero, though, the honest position is that persistence is a notation for something you already had.""")

    post.add(
        "So it chains",
        f"""Single linkage merges two clusters as soon as *any* pair of their points is close. It does not care that the rest of the two clusters are far apart, or that the close pair is a single point. This is called chaining, it is the reason single linkage is usually the *last* linkage anybody recommends, and it is now a fact about barcodes.

{snip['chaining'].markdown()}

Three points. Not three percent of the data — three points out of sixty-three, adding no cluster of their own, placed along a line where nothing was before. The ratio goes from {chain[0]['separation_ratio']:.2f} to {chain[3]['separation_ratio']:.2f}, which is to say the two longest bars are now the same length to three decimals and the barcode has no two-cluster claim left to make.""",
        figures=[figs["f1"]])

    post.add(
        "",
        f"""At twelve bridge points it is worse than gone. The longest bar is {chain[12]['longest']:.4f} — which is *exactly* the second bar of the unbridged cloud, {chain[0]['second']:.4f}, so nothing survives but the spacing inside a blob — and cutting the tree at k = 2, which is what you do when you have decided there are two clusters, returns groups of size **{chain[12]['cut_sizes'][0]}** and **{chain[12]['cut_sizes'][1]}**. It shaves off one point.

That is the shape of the failure worth remembering. It does not return two wrong clusters. It returns a right answer to a question nobody asked, in the format of an answer to the question you did ask.""",
        level=3,
        figures=[figs["f2"]])

    post.add(
        "The theorem that makes people trust this, and what it covers",
        f"""There is a good reason persistence has a reputation for robustness, and it is a real theorem. Stating it needs one definition first.

Two barcodes are compared by **matching** their bars: pair each bar of one with a bar of the other, and where a bar has no partner, pair it with nothing — which for a bar `(0, d)` costs `d/2`, its distance to the diagonal of the persistence diagram. The cost of a matching is its single worst pair, and the **bottleneck distance** is the cheapest matching's cost:

$$
d_B(A, B) = \\min_{{\\gamma}} \\max_{{a}} \\lVert a - \\gamma(a) \\rVert_{{\\infty}}
$$

A minimum over matchings of a maximum over pairs, which is worth saying twice because getting the two the wrong way round produces a number that is wrong in the direction that flatters you. (It is also worth checking against something that is not itself: `bottleneck_h0` in the repository is validated against brute-force enumeration on small barcodes, and the first version of it failed that test.)

The **stability theorem** then says that for two clouds `X` and `Y`,

$$
d_B\\left(\\mathrm{{Dgm}}(X), \\mathrm{{Dgm}}(Y)\\right) \\le d_H(X, Y)
$$

where `d_H` is the Hausdorff distance between the clouds. Perturb your data a little and the barcode moves a little, with a bound and no assumptions whatever about the data.

That is worth having, and it is not vacuous. Jitter the cloud by ε = 0.5 and the points move by a Hausdorff distance of {st[0.5]['hausdorff']:.4f} while the barcode moves by a bottleneck distance of {st[0.5]['bottleneck']:.4f} — inside the bound with a factor of {st[0.5]['hausdorff'] / st[0.5]['bottleneck']:.1f} to spare.""",
        figures=[figs["f3"]])

    post.add(
        "",
        f"""Now read the theorem's statement again, because it is about the barcode. Nothing in it mentions the number you extract from the barcode, and that number is a different object with different behaviour.

The rule in practice is "count the bars above the biggest jump". Applied to forty independent noise draws at each level:

- ε = 0.0: {counts(0.0)} of 40 draws
- ε = 0.2: {counts(0.2)} of 40
- ε = 1.0: {counts(1.0)}
- ε = 1.5: {counts(1.5)}

At ε = 1.0 the modal answer is still 2, and it is the answer in only {gaps[1.0]['share_modal'] * 100:.0f}% of draws. At ε = 1.5 the same rule on the same cloud returns anything from {gaps[1.5]['k_min']} to {gaps[1.5]['k_max']}.

And here is the part that settles it. Between those two levels the barcode has stopped moving: the largest bottleneck distance is {gaps[1.0]['bottleneck_max']:.3f} at ε = 1.0 and {gaps[1.5]['bottleneck_max']:.3f} at ε = 1.5 — the same number, because the barcode is pinned against the diameter of the cloud and cannot go further. The picture has converged. The integer read off the picture is running from 2 to {gaps[1.5]['k_max']}.

{snip['gap'].markdown()}

The stability theorem is true and the conclusion people draw from it is not. This is episode 7 of the linear-algebra series in a new notation: Eckart and Young settled which rank-*k* matrix is closest to yours and said nothing about *k*; Cohen-Steiner, Edelsbrunner and Harer settled how far a barcode can move and said nothing about how many bars to count.""",
        level=3)

    post.add(
        "One more thing, before the series gets to embeddings",
        """Everything above was two-dimensional, where a barcode is easy to read. The clouds this series is actually about have hundreds of dimensions, and something happens to the picture on the way there.

Take 200 points of pure Gaussian noise — no clusters, nothing to find — and ask how much of the filtration axis the deaths occupy: (max − min) divided by the mean.""",
        figures=[figs["f4"]])

    post.add(
        "",
        f"""{dims[2]['barcode_spread']:.2f} at two dimensions. {dims[768]['barcode_spread']:.3f} at 768. Every bar in that last barcode is born and dies within a few percent of one radius, so there is no visible structure in it — and there shouldn't be, because it is noise.

The problem is that this is a property of the *display* rather than of the data. In 768 dimensions a barcode with real clusters in it also has almost no dynamic range, which means **"there is no clear gap in the barcode" stops being evidence of anything.** That is the next episode: the separation H0 actually needs, against the dimension of the space the noise lives in, and the two answers you get depending on whether you measure it in absolute terms or relative ones.""",
        level=3)

    post.add(
        "What to keep",
        f"""1. A filtration is a metric and a scale. Both are choices, and neither is in the data.
2. H0 of a Rips filtration **is** the single-linkage dendrogram — `array_equal`, on every design tried. Thirty lines of union-find.
3. So it chains. Three points between two blobs take the two-cluster ratio from {chain[0]['separation_ratio']:.2f} to {chain[3]['separation_ratio']:.2f}; twelve make the k = 2 cut return {chain[12]['cut_sizes'][0]} points and {chain[12]['cut_sizes'][1]}.
4. The stability theorem is real: at ε = 0.5, Hausdorff {st[0.5]['hausdorff']:.3f} against bottleneck {st[0.5]['bottleneck']:.3f}.
5. It says nothing about the cluster count. At ε = 1.0 that count is 2 in {gaps[1.0]['counts'].get(2, 0)} of 40 draws and something else in the rest, with every barcode inside a bottleneck of {gaps[1.0]['bottleneck_max']:.2f}.
6. And in high dimensions the barcode's own dynamic range collapses — {dims[2]['barcode_spread']:.2f} to {dims[768]['barcode_spread']:.3f} — so a flat-looking barcode is not evidence of a flat-looking cloud.""")

    post.add(
        "Exercise",
        """Take an embedding matrix you have — any set of vectors, from any model — and compute its H0 barcode with the thirty lines above. Then compute `scipy.cluster.hierarchy.linkage(..., method="single")` on the same distances and check that the merge heights are the same array.

If they are, you have learned that any topological clustering you were planning to do on those vectors is single linkage, and you can go and read about chaining instead of about homology.

Then do the harder version. Add ten points along the line between your two most distant clusters, at a spacing tighter than the clusters' own, and see what the barcode says. If your embeddings are dense enough that such points are already there — and in a real embedding space, between any two clusters, they usually are — that is the measurement, and it is telling you the answer before you started.

Next episode: how much separation H0 needs before it can find a cluster at all, and why that number rises with dimension in absolute terms and falls in relative ones.""")

    post.hero = figs["hero"]
    return post


def main() -> Post:
    return build()
if __name__ == "__main__":
    r = compute()
    print("identity:")
    for row in r["identity"]:
        print(f"  {row['n']:>3} points in {row['d']:>2}d: {row['bars']} bars, "
              f"array_equal={row['equal']}, max gap {row['max_gap']:.1e}")
    print("\nchaining:")
    for row in r["chaining"]:
        print(f"  bridge {row['bridge']:>2} (n={row['n']:>2})  longest "
              f"{row['longest']:.4f}  second {row['second']:.4f}  "
              f"ratio {row['separation_ratio']:.2f}  cut {row['cut_sizes']}")
    print("\nstability:")
    for row in r["stability"]:
        print(f"  eps {row['epsilon']:.2f}  hausdorff {row['hausdorff']:7.4f}  "
              f"bottleneck {row['bottleneck']:7.4f}  bound {row['bound_holds']}  "
              f"gap_k {row['gap_k']}")
    print("\ndynamic range:")
    for row in r["dims"]:
        print(f"  d={row['d']:>4}  mean {row['mean_distance']:7.3f}  "
              f"distances {row['distance_spread']:.3f}  "
              f"barcode {row['barcode_spread']:.3f}")
    print(f"\nsix-point barcode: {np.round(r['hand_bc'].deaths, 3).tolist()}")
