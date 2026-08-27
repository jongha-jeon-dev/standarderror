"""Covariance matrices that are not covariance matrices.

A covariance matrix has one property that is not a technical nicety: for any
weights `w`, `w' S w` is the variance of the portfolio `w' x`, and a variance
cannot be negative. That single sentence *is* positive semi-definiteness. A
matrix with a negative eigenvalue is not "slightly ill-conditioned" — it is
claiming that some combination of your variables has negative variance, and the
eigenvector for that eigenvalue tells you which combination.

Three things make this worth a module.

**The constraint is much tighter than it looks.** Correlations are not free
parameters that happen to live in [-1, 1]. Fix two of the three correlations in a
3x3 matrix and the third is confined to an interval that is usually far narrower
than [-1, 1] — `feasible_band` computes it, and the derivation is the
determinant. Read through `rho = cos(theta)` it is exactly the triangle
inequality on angles, so the feasible band is not a numerical curiosity: it is
what "these are vectors in some inner product space" means.

**Two very different failures produce the same symptom.** Estimate every entry
from one sample and the matrix is a Gram matrix, hence positive semi-definite by
construction; the only way it fails is floating point. Estimate the entries from
*different* subsamples — pairwise deletion with unequal histories, a stress
overlay, a correlation elicited from an expert — and there is nothing making the
entries consistent with each other. `negative_rate` separates the two: the
sampling-noise version vanishes as `n` grows, and the inconsistency version
converges to a fixed negative number, so more data makes it worse.

**The standard repair hides the second one.** `nearest_correlation` returns the
closest positive semi-definite matrix and it is the right tool for floating-point
damage. Applied to an inconsistent matrix it spreads a small correction across
every entry — including the entries that were estimated correctly — and returns
something that passes every check while still being wrong where it was wrong.
`repair_cost` reports what moved, so the size of the repair can be compared with
the size of the actual error rather than assumed to match it.

References for where this stops: Higham, "Computing the nearest correlation
matrix — a problem from finance", *IMA J. Numer. Anal.* 22 (2002); Little and
Rubin, *Statistical Analysis with Missing Data*, chapter 3, on why pairwise
deletion is consistent under MCAR and not otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Tolerance for calling a computed eigenvalue "zero rather than negative".
#: A genuinely singular positive semi-definite matrix comes back from `eigh`
#: with its smallest eigenvalue at about -eps * ||S||, and treating that as a
#: modelling failure would flag every rank-deficient covariance matrix ever
#: estimated. Anything below this is arithmetic; anything above it is a claim.
PSD_TOLERANCE = 1e-10


def correlation3(rho_ab: float, rho_ac: float, rho_bc: float) -> np.ndarray:
    """A 3x3 correlation matrix from its three off-diagonal entries.

    Deliberately does *not* check feasibility: the whole point of the smallest
    example is that you can write down three numbers in [-1, 1] and get a matrix
    that is not a correlation matrix. Use `psd_report` to find out.
    """
    for name, v in (("rho_ab", rho_ab), ("rho_ac", rho_ac), ("rho_bc", rho_bc)):
        if not -1.0 <= float(v) <= 1.0:
            raise ValueError(f"{name}={v} is not in [-1, 1]")
    a, b, c = float(rho_ab), float(rho_ac), float(rho_bc)
    return np.array([[1.0, a, b], [a, 1.0, c], [b, c, 1.0]])


def feasible_band(rho_ab: float, rho_ac: float) -> tuple[float, float]:
    """The interval the third correlation must lie in, given the other two.

    From `det(R) >= 0`. Writing the determinant of `correlation3(a, b, c)` out
    gives `1 + 2abc - a^2 - b^2 - c^2`, which is a downward parabola in `c`, so
    the feasible set is the interval between its roots:

        c in [ab - sqrt((1-a^2)(1-b^2)),  ab + sqrt((1-a^2)(1-b^2))]

    With `a = cos(alpha)` and `b = cos(beta)` the endpoints are `cos(alpha+beta)`
    and `cos(alpha-beta)` — the triangle inequality on the angles between three
    vectors, which is what the algebra is a restatement of.
    """
    a, b = float(rho_ab), float(rho_ac)
    if not (-1.0 <= a <= 1.0 and -1.0 <= b <= 1.0):
        raise ValueError(f"correlations must be in [-1, 1]; got {a}, {b}")
    half = float(np.sqrt(max(0.0, (1.0 - a * a) * (1.0 - b * b))))
    return a * b - half, a * b + half


def correlation_angles(rho_ab: float, rho_ac: float, rho_bc: float) -> dict:
    """The three correlations as angles, and the triangle-inequality slack.

    A correlation is the cosine of the angle between two centred data vectors.
    Feasibility is then `|alpha - beta| <= gamma <= alpha + beta` on the three
    angles, and `slack` is how far outside that the third angle falls, in
    degrees — negative or zero when the matrix is feasible.
    """
    a = float(np.degrees(np.arccos(np.clip(rho_ab, -1.0, 1.0))))
    b = float(np.degrees(np.arccos(np.clip(rho_ac, -1.0, 1.0))))
    g = float(np.degrees(np.arccos(np.clip(rho_bc, -1.0, 1.0))))
    lo, hi = abs(a - b), min(360.0 - (a + b), a + b)
    return {"angle_ab": a, "angle_ac": b, "angle_bc": g,
            "lower": lo, "upper": hi,
            "slack": float(max(lo - g, g - hi))}


@dataclass
class PSDReport:
    """Is this a covariance matrix, and if not, which portfolio proves it."""
    eigenvalues: list[float]
    min_eigenvalue: float
    cholesky_ok: bool
    is_psd: bool
    worst_weights: list[float] = field(default_factory=list)
    worst_variance: float = 0.0


def psd_report(S) -> PSDReport:
    """Eigenvalues, the Cholesky verdict, and the offending portfolio.

    Both tests are reported because they answer different questions. Cholesky is
    the cheap one — it fails if and only if the matrix is not positive definite,
    costs a third of what an eigendecomposition does, and is what a library
    should be calling. The eigenvalues are what you need once it *has* failed,
    because the eigenvector for the smallest one is the portfolio with negative
    variance, and reading it tells you which variables are in conflict.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"expected a square matrix; got shape {S.shape}")
    if not np.allclose(S, S.T, atol=1e-12):
        raise ValueError("matrix is not symmetric; a covariance matrix is")
    w, V = np.linalg.eigh(S)
    lo = float(w[0])
    try:
        np.linalg.cholesky(S)
        chol = True
    except np.linalg.LinAlgError:
        chol = False
    weights = V[:, 0]
    return PSDReport(
        eigenvalues=[float(v) for v in w],
        min_eigenvalue=lo,
        cholesky_ok=chol,
        is_psd=lo > -PSD_TOLERANCE * max(1.0, float(np.abs(S).max())),
        worst_weights=[float(v) for v in weights],
        worst_variance=float(weights @ S @ weights),
    )


def leverage_table(S, weights, leverages=(1.0, 2.0, 5.0, 10.0)) -> list[dict]:
    """Portfolio variance as the position is scaled up.

    A negative eigenvalue is not a small error that a solver might absorb. It
    makes `w' S w` unbounded below along the offending direction, so an optimiser
    minimising variance under a budget constraint has no solution — it will take
    the position as large as it is allowed to. This tabulates that: the variance
    falls as the square of the leverage, and the "risk" a report would print is
    the square root of a negative number.
    """
    S = np.asarray(S, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (S.shape[0],):
        raise ValueError(f"weights of shape {w.shape} against a "
                         f"{S.shape[0]}x{S.shape[0]} matrix")
    out = []
    for L in leverages:
        v = float((L * w) @ S @ (L * w))
        out.append({"leverage": float(L), "variance": v,
                    "reported_sd": float(np.sqrt(v)) if v >= 0 else float("nan"),
                    "gross_exposure": float(L * np.abs(w).sum())})
    return out


# ------------------------------------------------------------------ estimating

def _corr_from_pair(x, y) -> float:
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    if sx == 0 or sy == 0:
        raise ValueError("a variable is constant on this subsample; "
                         "its correlation is undefined, not zero")
    return float(np.mean((x - x.mean()) * (y - y.mean()))
                 * len(x) / (len(x) - 1) / (sx * sy))


def pairwise_correlation(X, *, min_overlap: int = 4) -> dict:
    """Each correlation from whatever rows have both variables.

    Returns the matrix and the overlap counts, because the counts are the
    diagnosis: entries computed on different numbers of rows were computed on
    different samples, and nothing then forces them to be mutually consistent.
    """
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    C = np.eye(p)
    n_used = np.zeros((p, p), dtype=int)
    np.fill_diagonal(n_used, (~np.isnan(X)).sum(axis=0))
    for i in range(p):
        for j in range(i + 1, p):
            ok = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            k = int(ok.sum())
            if k < min_overlap:
                raise ValueError(
                    f"variables {i} and {j} overlap on {k} rows, below "
                    f"min_overlap={min_overlap}; a correlation from that many "
                    f"rows is not an estimate")
            C[i, j] = C[j, i] = _corr_from_pair(X[ok, i], X[ok, j])
            n_used[i, j] = n_used[j, i] = k
    return {"matrix": C, "n_used": n_used,
            "min_overlap": int(n_used[~np.eye(p, dtype=bool)].min()),
            "max_overlap": int(n_used[~np.eye(p, dtype=bool)].max())}


def complete_case_correlation(X) -> dict:
    """One subsample, every entry.

    Positive semi-definite by construction and not by luck: on the rows where
    everything is observed, the matrix is `Z'Z / (n-1)` for a single centred and
    scaled `Z`, and `w' Z' Z w = ||Z w||^2 >= 0` for every `w`. The same Gram
    structure that squared the condition number in episode two is what
    guarantees feasibility here.
    """
    X = np.asarray(X, dtype=float)
    ok = ~np.isnan(X).any(axis=1)
    n = int(ok.sum())
    if n < 4:
        raise ValueError(f"{n} complete rows; nothing to estimate from")
    Z = X[ok]
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
    return {"matrix": (Z.T @ Z) / (n - 1), "n_used": n,
            "rows_dropped": int(len(X) - n)}


# ------------------------------------------------------------------ simulation

def _cholesky_of(R) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    try:
        return np.linalg.cholesky(R)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cannot simulate from a matrix that is not positive "
                         "definite; the target correlation is itself "
                         "infeasible") from exc


def mcar_panel(n: int, correlation, *, missing_rate: float, rng) -> np.ndarray:
    """One sample with values deleted completely at random.

    The benign case, and the one worth having a name for: every pair still
    estimates the *same* population, so pairwise deletion is consistent and the
    only thing that can break feasibility is sampling noise.
    """
    L = _cholesky_of(correlation)
    p = L.shape[0]
    X = rng.standard_normal((n, p)) @ L.T
    X[rng.random((n, p)) < float(missing_rate)] = np.nan
    return X


def two_regime_panel(n_long: int, n_short: int, *, long_correlation,
                     short_correlation, late_columns=(2,), rng) -> np.ndarray:
    """Variables with different histories, and a correlation that changed.

    The dangerous case. `late_columns` are missing for the whole first regime —
    an asset that listed recently, a field added to a form, a question added to
    a survey — so their correlations are estimated on the short window while the
    others are estimated on everything. If the correlation structure differs
    between the two regimes, the resulting entries describe different
    populations, and no amount of data fixes that.
    """
    Ll, Ls = _cholesky_of(long_correlation), _cholesky_of(short_correlation)
    if Ll.shape != Ls.shape:
        raise ValueError("the two regimes must have the same variables")
    p = Ll.shape[0]
    late = list(late_columns)
    if not late:
        raise ValueError("with no late columns every pair shares one sample "
                         "and there is nothing to demonstrate")
    X1 = rng.standard_normal((n_long, p)) @ Ll.T
    X2 = rng.standard_normal((n_short, p)) @ Ls.T
    X1[:, late] = np.nan
    return np.vstack([X1, X2])


def negative_rate(sizes, *, correlation, missing_rate: float, reps: int,
                  seed: int = 0) -> list[dict]:
    """How often MCAR pairwise deletion returns an infeasible matrix.

    The answer is a function of `p` relative to `n` and it goes to zero, which is
    the reason this mechanism is the benign one. Reported as a rate and as the
    mean smallest eigenvalue, because the rate alone hides how far outside
    feasibility the failures fall.
    """
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        mins = []
        for _ in range(int(reps)):
            X = mcar_panel(n, correlation, missing_rate=missing_rate, rng=rng)
            try:
                C = pairwise_correlation(X)["matrix"]
            except ValueError:
                continue
            mins.append(float(np.linalg.eigvalsh(C)[0]))
        if not mins:
            raise ValueError(f"n={n} never produced enough overlap to estimate")
        arr = np.array(mins)
        out.append({"n": int(n), "reps": len(mins),
                    "rate": float((arr < 0).mean()),
                    "mean_min_eigenvalue": float(arr.mean()),
                    "worst_min_eigenvalue": float(arr.min())})
    return out


def regime_limit(sizes, *, long_correlation, short_correlation, ratio: int = 8,
                 late_columns=(2,), seed: int = 11) -> list[dict]:
    """The smallest eigenvalue against sample size, for heterogeneous overlap.

    The counterpart to `negative_rate`, and the point of having both: this one
    does not go to zero. The negative eigenvalue is a bias — the entries are
    estimates of different quantities — so growing the sample sharpens it.
    """
    out = []
    for n_short in sizes:
        X = two_regime_panel(int(ratio) * int(n_short), int(n_short),
                             long_correlation=long_correlation,
                             short_correlation=short_correlation,
                             late_columns=late_columns,
                             rng=np.random.default_rng(seed))
        pw = pairwise_correlation(X)
        cc = complete_case_correlation(X)
        out.append({
            "n_short": int(n_short),
            "n_total": int(len(X)),
            "min_eigenvalue": float(np.linalg.eigvalsh(pw["matrix"])[0]),
            "min_eigenvalue_complete": float(
                np.linalg.eigvalsh(cc["matrix"])[0]),
            "max_overlap": pw["max_overlap"],
            "min_overlap": pw["min_overlap"],
        })
    return out


# ---------------------------------------------------------------------- repair

def clip_to_psd(S, *, renormalise: bool = True) -> np.ndarray:
    """Set the negative eigenvalues to zero, then restore the diagonal.

    The one-line repair, and the renormalisation is not optional: zeroing an
    eigenvalue changes the diagonal, so without the last step the result is a
    positive semi-definite matrix whose variances are no longer the variances you
    measured.
    """
    S = np.asarray(S, dtype=float)
    w, V = np.linalg.eigh(S)
    B = (V * np.maximum(w, 0.0)) @ V.T
    if not renormalise:
        return B
    d = np.sqrt(np.diag(B))
    if np.any(d <= 0):
        raise ValueError("clipping removed a variable's variance entirely; "
                         "there is nothing left to renormalise")
    return B / np.outer(d, d)


def nearest_correlation(S, *, tol: float = 1e-12, max_iter: int = 200) -> dict:
    """Higham's nearest correlation matrix, by alternating projections.

    Two constraints — positive semi-definite, and unit diagonal — and projecting
    onto either one breaks the other, so the method alternates with a Dykstra
    correction that stops it converging to the wrong point. Returns the matrix
    and the iteration count; a caller that ignores `converged` is not entitled to
    the word "nearest".
    """
    A = np.asarray(S, dtype=float)
    Y = A.copy()
    dS = np.zeros_like(A)
    converged = False
    used = max_iter
    for k in range(int(max_iter)):
        R = Y - dS
        w, V = np.linalg.eigh(R)
        X = (V * np.maximum(w, 0.0)) @ V.T
        dS = X - R
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        if (np.max(np.abs(Y - X)) < tol
                and float(np.linalg.eigvalsh(Y)[0]) > -1e-13):
            converged, used = True, k + 1
            break
    return {"matrix": Y, "iterations": used, "converged": converged}


def repair_cost(before, after, *, labels=None) -> dict:
    """What the repair moved, entry by entry.

    Reported so it can be compared against the size of the error that made the
    repair necessary. When those two numbers differ by an order of magnitude the
    repair is cosmetic, and the comparison is the only way to find that out.
    """
    B = np.asarray(before, dtype=float)
    A = np.asarray(after, dtype=float)
    if B.shape != A.shape:
        raise ValueError(f"shapes {B.shape} and {A.shape} are not comparable")
    p = B.shape[0]
    names = list(labels) if labels is not None else [str(i) for i in range(p)]
    if len(names) != p:
        raise ValueError(f"{len(names)} labels for {p} variables")
    entries = []
    for i in range(p):
        for j in range(i + 1, p):
            entries.append({"pair": f"{names[i]}-{names[j]}",
                            "before": float(B[i, j]), "after": float(A[i, j]),
                            "change": float(A[i, j] - B[i, j])})
    return {"entries": entries,
            "frobenius": float(np.linalg.norm(A - B, "fro")),
            "max_change": max(abs(e["change"]) for e in entries),
            "largest_moved": max(entries, key=lambda e: abs(e["change"]))["pair"]}
