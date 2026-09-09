"""The Rips filtration in dimension zero, and what it is the same thing as.

Grow a ball of radius `r` around every point and connect two points when their
balls touch, which happens at `r = d(x, y) / 2`; by convention the filtration is
indexed by `d(x, y)` itself. Every point is its own connected component at
`r = 0` and components merge as `r` grows, so the 0-dimensional homology is a
list of components, each born at 0 and dying when it merges into an older one.

**That list of deaths is the single-linkage dendrogram's merge heights.** Not
approximately: `rips_h0` and `scipy`'s `linkage(..., method="single")` return
the same floats, and `test_filtration.py` asserts bit equality on 60 points in
five dimensions. Which settles what H0 is for, and also what it inherits:

* **chaining.** Two blobs 6.0 apart give a longest bar of 4.00 against a second
  of 0.598 -- a ratio of 6.7, an unmistakable two-cluster signal. Add *three*
  points along the line between them and the ratio is 1.00 (1.8898 against
  1.8878). Add twelve, out of seventy-two, and cutting the tree at k = 2 returns
  clusters of size 71 and 1: it splits off one point rather than the two blobs.
* **the stability theorem, which is real.** `bottleneck_h0` between a cloud's
  barcode and a jittered copy's stays under the Hausdorff distance between the
  clouds in every measurement here, and well under: at `eps = 0.5`, Hausdorff
  1.39 against bottleneck 0.44.
* **and the integer, which is not.** Over forty noise draws at `eps = 1.0` the
  "number of clusters at the largest gap" comes back 2 twenty-two times, 3
  twelve times, and 4 or 5 six times -- while every one of those barcodes is
  within a bottleneck distance of 1.96.
  The summary is stable; the decision extracted from it is a different object
  with no such guarantee -- which is episode 7 of the linear-algebra series
  arriving in a new notation.

References: Edelsbrunner and Harer, *Computational Topology* (AMS, 2010), for
the filtration and the persistence pairing; Cohen-Steiner, Edelsbrunner and
Harer, "Stability of persistence diagrams", *Discrete & Computational Geometry*
37 (2007), for the bound `bottleneck_h0` is checked against; Carlsson,
"Topology and data", *Bulletin of the AMS* 46 (2009), for the programme; and
Chazal, Guibas, Oudot and Skraba, "Persistence-based clustering in Riemannian
manifolds", *JACM* 60 (2013), for what is done about chaining.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

#: Normalisations that each define a different filtration on the same points.
#: Episode 3 is about the fact that this is a list of four answers, not of four
#: spellings of one answer.
NORMALISATIONS = ("raw", "centred", "l2", "whitened", "cosine")


def normalise(X, kind: str = "raw") -> np.ndarray:
    """Apply one of `NORMALISATIONS` to a point cloud.

    `cosine` is handled by `pairwise` instead, because it changes the metric
    rather than the points; asking for it here is a mistake worth an exception.
    """
    X = np.asarray(X, dtype=float)
    if kind == "raw":
        return X
    if kind == "centred":
        return X - X.mean(0)
    if kind == "l2":
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n
    if kind == "whitened":
        Y = X - X.mean(0)
        s = Y.std(0)
        s[s == 0] = 1.0
        return Y / s
    if kind == "cosine":
        raise ValueError("cosine is a metric, not a normalisation; "
                         "pass it to `pairwise` as `metric`")
    raise ValueError(f"unknown normalisation {kind!r}; "
                     f"expected one of {sorted(set(NORMALISATIONS))}")


def pairwise(X, *, metric: str = "euclidean", normalisation: str = "raw"):
    """The distance matrix a filtration is built from -- the whole choice.

    Two arguments, five values each on one side, and the episode's point is that
    they are not stylistic. `metric="cosine"` on an off-centre cloud has a
    baseline near 1 and almost no spread; the same points centred first do not.
    """
    Y = normalise(X, normalisation)
    return squareform(pdist(Y, metric=metric))


@dataclass(frozen=True)
class Barcode:
    """0-dimensional persistence: every bar is born at 0, so only deaths vary.

    The last death is `inf` mathematically -- one component survives forever --
    and is dropped here rather than stored, because carrying an `inf` through
    every summary below buys nothing and breaks the arithmetic.
    """

    deaths: np.ndarray

    @property
    def n_points(self) -> int:
        return len(self.deaths) + 1

    @property
    def longest(self) -> float:
        """The longest *finite* bar."""
        return float(self.deaths[-1])

    @property
    def separation_ratio(self) -> float:
        """Longest finite bar over the next one: how loudly the cloud claims to
        have exactly two clusters."""
        if len(self.deaths) < 2:
            return float("nan")
        return float(self.deaths[-1] / self.deaths[-2])

    @property
    def spread_ratio(self) -> float:
        """`(max - min) / mean` over the deaths: how much of the filtration axis
        the barcode actually occupies. This is the quantity that collapses with
        dimension -- 7.16 at d = 2 against 0.083 at d = 768 -- and it is a
        property of the display rather than of the clustering."""
        d = self.deaths
        return float((d.max() - d.min()) / d.mean())

    def gap_k(self) -> int:
        """The number of clusters "the biggest jump in the barcode" implies.

        Included because it is what people do, and because it is the object the
        stability theorem says nothing about.
        """
        if len(self.deaths) < 2:
            return 1
        return len(self.deaths) - int(np.argmax(np.diff(self.deaths)))

    def alive_at(self, radius: float) -> int:
        """Components still distinct at this filtration value."""
        return 1 + int(np.count_nonzero(self.deaths > float(radius)))


def rips_h0(D) -> Barcode:
    """0-dimensional persistent homology of the Rips filtration, by union-find.

    Thirty lines, and they are the whole of H0. Edges are considered in order of
    length; an edge that joins two distinct components kills the younger one, so
    its length is a death time. Edges inside a component change nothing, which
    is why this is `O(E log E)` and not a matrix reduction.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("rips_h0 wants a square distance matrix")
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    # `stable` so that ties resolve by index rather than arbitrarily, which is
    # what makes the comparison against scipy reproducible rather than lucky.
    order = np.argsort(D[iu], kind="stable")
    left, right, length = iu[0][order], iu[1][order], D[iu][order]

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    deaths = []
    for a, b, w in zip(left, right, length):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
            deaths.append(float(w))
    return Barcode(deaths=np.sort(np.asarray(deaths, dtype=float)))


def single_linkage_heights(D) -> np.ndarray:
    """`scipy`'s single-linkage merge heights, for comparison with `rips_h0`."""
    D = np.asarray(D, dtype=float)
    return np.sort(linkage(squareform(D, checks=False), method="single")[:, 2])


def cut_at(D, k: int) -> np.ndarray:
    """Cut the single-linkage tree into `k` clusters. Labels are 1-based."""
    return fcluster(linkage(squareform(np.asarray(D, dtype=float), checks=False),
                            method="single"), int(k), criterion="maxclust")


def cluster_sizes(D, k: int) -> list[int]:
    """Descending sizes of the `k` clusters, which is where chaining shows: a
    bridged pair of blobs cut at k = 2 returns `[71, 1]`."""
    lab = cut_at(D, k)
    return sorted(np.bincount(lab)[1:].tolist(), reverse=True)


def purity(D, labels, k: int) -> float:
    """Share of points whose cluster's majority label is their own."""
    labels = np.asarray(labels)
    got = cut_at(D, k)
    total = sum(np.bincount(labels[got == g]).max() for g in np.unique(got))
    return float(total / len(labels))


def bottleneck_h0(a: Barcode, b: Barcode) -> float:
    """Bottleneck distance between two 0-dimensional barcodes.

    Every bar is `(0, death)`, so a matching pairs death times and the cost of
    leaving a bar out is its distance to the diagonal, `death / 2`. The
    bottleneck is the *minimum over matchings of the maximum* cost, and that is
    not what a min-cost assignment returns: the first version of this function
    called `linear_sum_assignment` once and read off the largest chosen cost,
    which minimises the sum instead and disagreed with a brute-force matching on
    three of six random inputs.

    So it is done the standard way. Every candidate answer is one of the finitely
    many pairwise costs, so binary-search that sorted list and ask at each step
    whether a perfect matching exists using only pairs at or below the threshold
    -- which `linear_sum_assignment` answers as a feasibility question rather
    than an optimisation one.
    """
    d1, d2 = np.asarray(a.deaths, dtype=float), np.asarray(b.deaths, dtype=float)
    n, m = len(d1), len(d2)
    if n == 0 and m == 0:
        return 0.0
    big = np.inf
    # A square cost matrix over bars-of-`a` plus diagonal-slots-for-`b`, and
    # bars-of-`b` plus diagonal-slots-for-`a`. Matching a bar to its own
    # diagonal slot costs death/2; the diagonal-to-diagonal block is free.
    size = n + m
    cost = np.full((size, size), big)
    cost[:n, :m] = np.abs(d1[:, None] - d2[None, :])
    for i in range(n):
        cost[i, m + i] = d1[i] / 2.0
    for j in range(m):
        cost[n + j, j] = d2[j] / 2.0
    cost[n:, m:] = 0.0

    def feasible(threshold: float) -> bool:
        mask = np.where(cost <= threshold + 1e-15, 0.0, 1.0)
        rows, cols = linear_sum_assignment(mask)
        return bool(mask[rows, cols].sum() == 0.0)

    candidates = np.unique(cost[np.isfinite(cost)])
    lo, hi = 0, len(candidates) - 1
    if not feasible(float(candidates[hi])):
        raise RuntimeError("no matching exists, which cannot happen here")
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(float(candidates[mid])):
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo])


def hausdorff(X, Y) -> float:
    """Hausdorff distance between two clouds *in correspondence*, which is what
    the stability theorem's right-hand side is when `Y` is a jittered `X`."""
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("this form of the bound wants a point-for-point copy")
    return float(np.linalg.norm(X - Y, axis=1).max())


# ------------------------------------------------------------------ designs

def two_blobs(*, bridge: int = 0, gap: float = 6.0, per: int = 30,
              spread: float = 0.5, bridge_jitter: float = 0.05,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two blobs, optionally joined by a thin bridge of points.

    The bridge is the whole demonstration: it adds no new cluster and it removes
    the old ones, because single linkage only ever needs one short edge.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((per, 2)) * spread
    b = rng.standard_normal((per, 2)) * spread + np.array([gap, 0.0])
    parts, labels = [a, b], [np.zeros(per, int), np.ones(per, int)]
    if bridge:
        t = np.linspace(0.18, 0.82, int(bridge))[:, None]
        parts.append(t * np.array([gap, 0.0])
                     + rng.standard_normal((int(bridge), 2)) * bridge_jitter)
        labels.append(np.full(int(bridge), 2))
    return np.vstack(parts), np.concatenate(labels)


def separated_clusters(d: int, sep: float, *, per: int = 40, groups: int = 3,
                       n_struct: int = 3, seed: int = 0):
    """`groups` isotropic unit-variance clusters, separated along `n_struct`
    axes only. Episode 2's design: the separation that H0 needs, against the
    dimension of the space the noise lives in."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((groups * per, d))
    labels = np.repeat(np.arange(groups), per)
    centres = np.linspace(-1.0, 1.0, groups)
    for g in range(groups):
        X[labels == g, :n_struct] += float(sep) * centres[g]
    return X, labels


def narrow_cone(d: int = 64, *, per: int = 40, groups: int = 3, sep: float = 1.0,
                anisotropy: float = 60.0, cone: float = 6.0, seed: int = 0):
    """A cloud with the two properties embedding spaces are reported to have:
    most of the variance in a few directions, and every vector in a narrow cone
    around a common mean. Episode 3's design."""
    rng = np.random.default_rng(seed)
    scale = np.geomspace(anisotropy, 1.0 / anisotropy, d)
    X = rng.standard_normal((groups * per, d)) * scale
    labels = np.repeat(np.arange(groups), per)
    centres = np.linspace(-1.0, 1.0, groups)
    for g in range(groups):
        X[labels == g, -3:] += float(sep) * centres[g] * scale[-3:]
    return X + float(cone) * np.abs(rng.standard_normal(d)) * scale, labels


def chaining_sweep(bridges=(0, 3, 6, 12, 24), **kw) -> list[dict]:
    """One row per bridge size: what the barcode says and what the cut returns."""
    out = []
    for k in bridges:
        X, _ = two_blobs(bridge=int(k), **kw)
        D = pairwise(X)
        bc = rips_h0(D)
        out.append({"bridge": int(k), "n": len(X), "longest": bc.longest,
                    "second": float(bc.deaths[-2]),
                    "separation_ratio": bc.separation_ratio,
                    "cut_sizes": cluster_sizes(D, 2)})
    return out


def stability_sweep(X, epsilons=(0.0, 0.01, 0.05, 0.2, 0.5, 1.0), *,
                    seed: int = 1) -> list[dict]:
    """Jitter the cloud and compare how far the barcode moved with how far the
    points did, plus what the gap rule says at each step.

    Each row draws from its own generator, seeded by `(seed, eps)`. The first
    version advanced one generator through the loop, which made every row a
    function of how many *other* epsilons were in the list -- so the same
    `eps = 1.0` gave a different perturbation in a three-row sweep than in a
    six-row one, and the article's snippet disagreed with the article's figure.
    A measurement should not depend on what else you measured.
    """
    X = np.asarray(X, dtype=float)
    base = rips_h0(pairwise(X))
    out = []
    for eps in epsilons:
        rng = np.random.default_rng([int(seed), int(round(float(eps) * 1e6))])
        Y = X + rng.standard_normal(X.shape) * float(eps)
        got = rips_h0(pairwise(Y))
        bn = bottleneck_h0(base, got)
        hd = hausdorff(X, Y)
        out.append({"epsilon": float(eps), "hausdorff": hd, "bottleneck": bn,
                    "bound_holds": bool(bn <= hd + 1e-9),
                    "gap_k": got.gap_k(), "spread_ratio": got.spread_ratio})
    return out


def dimension_sweep(dims=(2, 8, 64, 768), *, n: int = 200,
                    seed: int = 0) -> list[dict]:
    """What dimension does to the barcode's dynamic range, on pure noise."""
    rng = np.random.default_rng(seed)
    out = []
    for d in dims:
        Y = rng.standard_normal((int(n), int(d)))
        dv = pdist(Y)
        bc = rips_h0(squareform(dv))
        out.append({"d": int(d), "mean_distance": float(dv.mean()),
                    "distance_spread": float((dv.max() - dv.min()) / dv.mean()),
                    "barcode_spread": bc.spread_ratio})
    return out


def separation_needed(d: int, *, target: float = 0.95, seeds: int = 3,
                      lo: float = 0.5, hi: float = 60.0, iterations: int = 24,
                      **kw) -> dict:
    """Bisect for the separation at which H0 recovers the clusters.

    Reported two ways on purpose. In absolute terms it *rises* with dimension,
    which is the expected story; as a fraction of the typical pairwise distance
    it *falls*, which is not, and the two together are why the barcode's loss of
    dynamic range is the thing to worry about rather than the clustering.
    """
    def ok(sep: float) -> bool:
        scores = [purity(pairwise(*[separated_clusters(d, sep, seed=s, **kw)[0]]),
                         separated_clusters(d, sep, seed=s, **kw)[1], 3)
                  for s in range(int(seeds))]
        return float(np.mean(scores)) > float(target)

    if ok(lo):
        raise ValueError("the lower end of the bracket already succeeds")
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    noise, _ = separated_clusters(d, 0.0, **kw)
    scale = float(pdist(noise).mean())
    return {"d": int(d), "separation": float(hi), "noise_scale": scale,
            "relative": float(hi / scale)}


def gap_rule_spread(X, epsilon: float, *, draws: int = 40, base_seed: int = 1000):
    """What the gap rule answers across independent noise draws at one `epsilon`.

    The stability theorem bounds how far the *barcode* moves. This is what
    happens to the integer somebody reads off it, and the two behave nothing
    alike. On `two_blobs(per=40)` over 40 draws:

        eps    bottleneck      k, with counts
        0.0    0.000           2 x40
        0.2    0.084-0.778     2 x40
        0.5    0.475-1.958     2 x39, 3 x1
        0.8    0.963-1.958     2 x26, 3 x12, 4 x1, 5 x1
        1.0    1.162-1.958     2 x22, 3 x12, 4 x3, 5 x3
        1.5    0.949-1.958     2 x21, 3 x6, 4 x6, 6 x1, 7 x2, 8 x2, 10 x1, 11 x1

    The bottleneck distance saturates at 1.958 and stops moving; the integer
    keeps going, out to 11. So "how many clusters does the barcode show" is not
    a stable functional of the cloud, and nothing in the stability theorem ever
    said it was.
    """
    X = np.asarray(X, dtype=float)
    base = rips_h0(pairwise(X))
    counts: dict[int, int] = {}
    bottlenecks = []
    for i in range(int(draws)):
        rng = np.random.default_rng([int(base_seed) + i,
                                     int(round(float(epsilon) * 1e6))])
        Y = X + rng.standard_normal(X.shape) * float(epsilon)
        bc = rips_h0(pairwise(Y))
        k = bc.gap_k()
        counts[k] = counts.get(k, 0) + 1
        bottlenecks.append(bottleneck_h0(base, bc))
    return {"epsilon": float(epsilon), "draws": int(draws),
            "counts": dict(sorted(counts.items())),
            "modal_k": max(counts, key=lambda k: counts[k]),
            "share_modal": max(counts.values()) / int(draws),
            "k_min": min(counts), "k_max": max(counts),
            "bottleneck_min": float(min(bottlenecks)),
            "bottleneck_max": float(max(bottlenecks))}
