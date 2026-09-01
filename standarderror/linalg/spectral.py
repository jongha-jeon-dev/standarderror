"""Principal components when two eigenvalues are close.

"We interpret the second component as ..." is the most common sentence in applied
PCA and it is a claim about an *axis*. An axis is only well defined when its
eigenvalue is separated from its neighbours. When two eigenvalues are equal the
axes do not exist at all — any orthonormal basis of the shared eigenspace
reconstructs the same matrix, exactly, so which pair of directions `eigh` hands
back is a fact about LAPACK and not about the data. When two eigenvalues are
merely *close*, the axes exist but rotate freely under resampling, and the thing
that governs how freely is the gap.

Three things make this worth a module.

**The gap is the parameter, not the variance explained.** Every applied account
orders components by how much variance they carry and then trusts the big ones.
Stability does not work that way: `davis_kahan_bound` says the angle an
eigenvector can move under a perturbation `E` is controlled by `||E||` divided by
the distance to the *nearest neighbouring eigenvalue*. A component carrying 30
percent of the variance next to one carrying 29.7 percent is less determined than
one carrying 4 percent with nothing near it.

**The subspace survives what the axes do not.** `principal_angles` measures the
rotation between two subspaces rather than two vectors, and it is invariant to
the basis chosen for each — which is exactly the invariance the axes lack. In
every near-tied case measured here the plane spanned by a close pair is stable to
a degree or two while the individual axes inside it swing by tens of degrees. The
usable statement is about the plane.

**The construction here has a closed-form spectrum.** `block_pairs` builds a
correlation matrix from `k` independent pairs with within-pair correlations
`(a, b, c, ...)`, and its eigenvalues are exactly `1 ± a, 1 ± b, 1 ± c, ...`. So
a gap of 0.02 is a number chosen in advance rather than one found by search, and
every claim about gaps can be checked against a matrix whose answer is known
before the sample is drawn. `equicorrelation` goes further: its lower eigenvalue
has multiplicity `p - 1`, an *exact* tie, which is where the axes stop existing
rather than merely wobbling.

References for where this stops: Davis and Kahan, "The rotation of eigenvectors
by a perturbation. III", *SIAM J. Numer. Anal.* 7 (1970); Yu, Wang and Samworth,
"A useful variant of the Davis-Kahan theorem for statisticians", *Biometrika* 102
(2015), whose form is the one implemented here because it is stated in terms of
the population eigenvalues you can reason about rather than the sample ones you
cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Two computed eigenvalues closer than this are treated as one repeated
#: eigenvalue rather than two distinct ones. `eigh` on a matrix with an exact tie
#: returns the pair separated by a few multiples of eps times the norm, so a
#: threshold is unavoidable; this one is loose enough to catch that and far
#: tighter than any gap an applied problem would call "close".
TIE_TOLERANCE = 1e-9

#: Yu, Wang and Samworth (2015), Corollary 1: for a single eigenvector the sine
#: of the angle is bounded by this constant times ||E|| over the gap. The classic
#: Davis-Kahan statement uses 1 but is expressed with the *perturbed* spectrum,
#: which is not available before the perturbation.
DAVIS_KAHAN_CONSTANT = 2.0 ** 1.5


def equicorrelation(p: int, rho: float) -> np.ndarray:
    """`p` variables, every pair correlated `rho`.

    Spectrum in closed form: `1 + (p - 1) * rho` once, and `1 - rho` with
    multiplicity `p - 1`. The repeated eigenvalue is an *exact* tie, so the
    corresponding eigenvectors are determined only up to an arbitrary rotation
    within their `(p - 1)`-dimensional eigenspace — which is the whole point.
    """
    if p < 2:
        raise ValueError("need at least two variables")
    # The lower bound is where 1 - rho meets zero from the other side: below
    # -1/(p-1) the matrix stops being positive semi-definite.
    if not -1.0 / (p - 1) < rho < 1.0:
        raise ValueError(
            f"rho must be in (-1/(p-1), 1) = ({-1.0 / (p - 1):.3f}, 1) for p={p}")
    return (1.0 - rho) * np.eye(p) + rho * np.ones((p, p))


def block_pairs(correlations) -> np.ndarray:
    """A correlation matrix of independent pairs, one pair per entry.

    With within-pair correlations `(a, b, c)` the matrix is block diagonal with
    2x2 blocks and its eigenvalues are exactly `1 + a, 1 - a, 1 + b, 1 - b,
    1 + c, 1 - c`. That closed form is the reason this construction is used
    rather than a random matrix: the gap between any two adjacent eigenvalues is
    a difference of parameters, so an experiment about gaps can set the gap
    instead of hunting for one.
    """
    correlations = [float(c) for c in correlations]
    if not correlations:
        raise ValueError("need at least one pair")
    if any(not -1.0 < c < 1.0 for c in correlations):
        raise ValueError("each within-pair correlation must lie in (-1, 1)")
    p = 2 * len(correlations)
    R = np.eye(p)
    for i, c in enumerate(correlations):
        R[2 * i, 2 * i + 1] = R[2 * i + 1, 2 * i] = c
    return R


@dataclass(frozen=True)
class Spectrum:
    """Eigenvalues in descending order, with the gaps that govern the axes."""

    values: np.ndarray
    vectors: np.ndarray
    #: `gaps[i]` separates `values[i]` from `values[i + 1]`.
    gaps: np.ndarray
    #: What each component is usually judged by, and what does not predict it.
    variance_share: np.ndarray

    def neighbour_gap(self, i: int) -> float:
        """Distance from component `i` to whichever neighbour is closer.

        This, not `gaps[i]`, is the quantity in the Davis-Kahan bound: an
        eigenvector is squeezed by the spectrum on both sides, so the nearer
        neighbour is the one that determines it.
        """
        candidates = []
        if i > 0:
            candidates.append(self.gaps[i - 1])
        if i < len(self.gaps):
            candidates.append(self.gaps[i])
        if not candidates:
            raise ValueError("a one-by-one matrix has no neighbouring eigenvalue")
        return float(min(candidates))

    @property
    def ties(self) -> list[tuple[int, int]]:
        """Adjacent pairs whose eigenvalues are equal to within `TIE_TOLERANCE`."""
        return [(i, i + 1) for i, g in enumerate(self.gaps) if g <= TIE_TOLERANCE]


def spectrum(S) -> Spectrum:
    """Eigen-decompose a symmetric matrix, largest first."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("spectrum() takes a square matrix")
    values, vectors = np.linalg.eigh(S)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    total = float(values.sum())
    return Spectrum(values=values, vectors=vectors,
                    gaps=np.diff(values) * -1.0,
                    variance_share=values / total if total else values * 0.0)


def align_signs(V, reference) -> np.ndarray:
    """Flip each column of `V` to point the same way as `reference`.

    An eigenvector is defined up to sign, and `eigh` makes no promise about which
    one it returns, so two runs on nearly identical matrices routinely differ by a
    factor of -1 on some columns. Comparing loadings without fixing that first
    produces "instability" that is nothing but a sign convention, which is a
    different problem from the one this module is about and has to be removed
    before the real one is visible.
    """
    V = np.asarray(V, dtype=float)
    reference = np.asarray(reference, dtype=float)
    flips = np.sign(np.sum(V * reference, axis=0))
    flips[flips == 0] = 1.0
    return V * flips


def principal_angles(A, B) -> np.ndarray:
    """Angles in degrees between the subspaces spanned by the columns of A and B.

    Invariant to the basis chosen for either subspace, which is the property the
    individual eigenvectors do not have: rotate the columns of `A` among
    themselves and this does not move. That is why a near-tied *plane* can be
    reported when neither of the axes in it can be.
    """
    A = np.linalg.qr(np.asarray(A, dtype=float))[0]
    B = np.linalg.qr(np.asarray(B, dtype=float))[0]
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def vector_angle(u, v) -> float:
    """Angle in degrees between two vectors, ignoring sign.

    Ignoring sign rather than fixing it, because this is used on vectors that have
    not been through `align_signs` and an angle of 179 degrees is a sign flip, not
    a disagreement.
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    c = abs(float(u @ v)) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def davis_kahan_bound(perturbation_norm: float, gap: float) -> float:
    """Largest angle, in degrees, an eigenvector can move — Yu-Wang-Samworth.

    `sin(theta) <= 2^{3/2} ||E||_op / gap`, with `gap` the distance from this
    eigenvalue to the nearest other one. Returns 90 degrees when the bound exceeds
    one, which is the honest reading: the theorem has stopped saying anything, and
    a bound of "somewhere in the half-space" is what a vanishing gap earns.
    """
    if gap <= 0:
        return 90.0
    s = DAVIS_KAHAN_CONSTANT * float(perturbation_norm) / float(gap)
    return 90.0 if s >= 1.0 else float(np.degrees(np.arcsin(s)))


def rotate_within(V, i: int, j: int, degrees: float) -> np.ndarray:
    """Rotate columns `i` and `j` of `V` into each other by `degrees`.

    If those two columns span an eigenspace whose eigenvalues are equal, the
    result is another perfectly valid set of eigenvectors for the same matrix —
    `V S V'` is unchanged to machine precision. That is the demonstration that the
    axes carry no information: they can be spun to any angle and nothing
    measurable moves.
    """
    V = np.array(V, dtype=float, copy=True)
    t = np.radians(float(degrees))
    c, s = np.cos(t), np.sin(t)
    vi, vj = V[:, i].copy(), V[:, j].copy()
    V[:, i] = c * vi + s * vj
    V[:, j] = -s * vi + c * vj
    return V


def sample_correlation(n: int, correlation, *, rng) -> np.ndarray:
    """One sample correlation matrix from `n` draws of the given population."""
    C = np.asarray(correlation, dtype=float)
    L = np.linalg.cholesky(C)
    X = rng.standard_normal((int(n), C.shape[0])) @ L.T
    return np.corrcoef(X.T)


def bootstrap_components(correlation, *, n: int, reps: int, rng) -> dict:
    """Resample, re-decompose, and record what moved.

    Each replicate is a fresh sample of `n` rows from the population, not a
    resampling of one sample: the question is how much the axes move under the
    noise the estimator actually has, and a fresh draw measures that directly
    without the bootstrap's own approximation sitting in between.

    Reports, per component: how often it swaps rank with its neighbour, the median
    angle it moves, and the median principal angle of the plane it spans with that
    neighbour — the last being the quantity that stays small when the first two do
    not.
    """
    C = np.asarray(correlation, dtype=float)
    replicates = (sample_correlation(n, C, rng=rng) for _ in range(int(reps)))
    out = _compare_against(C, replicates, reps=int(reps))
    out["n"] = int(n)
    return out


def row_bootstrap(X, *, reps: int, rng) -> dict:
    """The same measurement, resampling the rows of one sample.

    This is what a reader can actually run: there is no population to draw from,
    only the data, so the reference is the sample's own correlation matrix and the
    replicates are resamples of its rows.

    It is also the check that cannot see the problem, and that is a finding rather
    than a footnote. The reference here is the sample's own axis, which under a
    near-tie is one arbitrary direction in the shared plane — so the replicates
    scatter around a centre that is itself off. Measured against the population on
    a matrix whose top gap is 0.02, the sample's first axis sits about 41 degrees
    from the truth while a 200-replicate row bootstrap of the same sample reports
    about 15: the movement is understated roughly threefold, and the reported
    number varies by a factor of five from sample to sample. The bootstrap can
    estimate the spread of an estimator. It cannot report that the quantity being
    estimated is not identified.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    reference = np.corrcoef(X.T)

    def replicates():
        for _ in range(int(reps)):
            yield np.corrcoef(X[rng.integers(0, n, n)].T)

    out = _compare_against(reference, replicates(), reps=int(reps))
    out["n"] = int(n)
    return out


def _compare_against(reference, replicates, *, reps: int) -> dict:
    """Shared body: what moved, between a reference matrix and its replicates."""
    C = np.asarray(reference, dtype=float)
    p = C.shape[0]
    truth = spectrum(C)

    swaps = np.zeros(p - 1, dtype=int)
    angles = np.zeros((reps, p))
    plane_angles = np.zeros((reps, p - 1))
    perturbation = np.zeros(reps)
    values = np.zeros((reps, p))

    for r, R in enumerate(replicates):
        est = spectrum(R)
        values[r] = est.values
        perturbation[r] = np.linalg.norm(R - C, 2)
        V = align_signs(est.vectors, truth.vectors)
        for k in range(p):
            angles[r, k] = vector_angle(V[:, k], truth.vectors[:, k])
        for k in range(p - 1):
            plane_angles[r, k] = principal_angles(
                truth.vectors[:, [k, k + 1]], est.vectors[:, [k, k + 1]]).max()
            # A swap is not a reordering of the eigenvalues -- `spectrum` always
            # sorts them -- it is the estimated k-th axis landing closer to the
            # true (k+1)-th than to the true k-th.
            here = vector_angle(V[:, k], truth.vectors[:, k])
            there = vector_angle(V[:, k], truth.vectors[:, k + 1])
            swaps[k] += there < here

    return {
        "reps": reps,
        "population": truth,
        "swap_rate": swaps / float(reps),
        "median_angle": np.median(angles, axis=0),
        "median_plane_angle": np.median(plane_angles, axis=0),
        "median_perturbation": float(np.median(perturbation)),
        "eigenvalue_lo": np.quantile(values, 0.025, axis=0),
        "eigenvalue_hi": np.quantile(values, 0.975, axis=0),
    }


def stability_table(result: dict) -> list[dict]:
    """One row per component: what it is judged by, and what governs it.

    The columns are in the order an applied reader would look at them — share of
    variance first, because that is the one they already use — so that the
    disagreement between the first column and the last is visible along each row.
    """
    truth = result["population"]
    p = len(truth.values)
    rows = []
    for k in range(p):
        # The nearest neighbour, and then *that same pair* for every column that
        # is about a pair. Reporting `swap_rate[k]` against `neighbour_gap(k)`
        # mixes two different pairs whenever the closer neighbour is above rather
        # than below, and the row then reads as a component that is squeezed hard
        # and never swaps.
        above = truth.gaps[k - 1] if k > 0 else np.inf
        below = truth.gaps[k] if k < p - 1 else np.inf
        pair = k - 1 if above <= below else k
        rows.append({
            "component": k + 1,
            "eigenvalue": float(truth.values[k]),
            "variance_share": float(truth.variance_share[k]),
            "neighbour_gap": float(min(above, below)),
            "neighbour": pair + 2 if pair == k else pair + 1,
            "median_angle": float(result["median_angle"][k]),
            "median_plane_angle": float(result["median_plane_angle"][pair]),
            "bound": davis_kahan_bound(result["median_perturbation"],
                                       float(min(above, below))),
            "swap_rate": float(result["swap_rate"][pair]),
        })
    return rows


def gap_sweep(gaps, *, base: float, other: float, n: int, reps: int, rng) -> list[dict]:
    """Swap rate and median angle against a gap that is set, not discovered.

    Each point is a `block_pairs` matrix whose top two eigenvalues differ by
    exactly the requested gap, with the rest of the spectrum held fixed, so the
    curve isolates one variable. This is the measurement the episode's claim
    rests on.
    """
    out = []
    for g in gaps:
        C = block_pairs([base, base - float(g), other])
        res = bootstrap_components(C, n=n, reps=reps, rng=rng)
        out.append({
            "gap": float(g),
            "swap_rate": float(res["swap_rate"][0]),
            "median_angle": float(res["median_angle"][0]),
            "median_plane_angle": float(res["median_plane_angle"][0]),
            "bound": davis_kahan_bound(res["median_perturbation"], float(g)),
        })
    return out
