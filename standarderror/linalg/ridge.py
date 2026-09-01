"""Ridge as an operation on the spectrum, and the diagnostic that cannot see why.

Ridge regression is usually introduced as a penalty: minimise the squared error
plus `alpha` times the squared norm of the coefficients. That description is
correct and explains nothing about what it does to a design, because the whole
mechanism is one line of linear algebra. Writing `X = U S V'`, the ridge solution
is `V diag(s / (s^2 + alpha)) U' y`, so in the basis where the design is diagonal
ridge multiplies each direction's coefficient by

    s^2 / (s^2 + alpha)

and nothing else happens. A direction the data measured well -- large `s` -- is
barely touched. A direction the data barely saw is shrunk almost to nothing. Ridge
is not a uniform shrinkage of the coefficient vector; it is a *selective*
shrinkage that falls hardest exactly where the evidence is thinnest, which is why
it works and also why its cost is invisible in the usual output.

Three things make this worth a module.

**The standard collinearity diagnostic is blind to it by construction.** The
variance inflation factor regresses each predictor on the others, so it is
invariant to the scale of every column and cannot see the intercept at all. A
design of mutually uncorrelated columns on wildly different scales -- a duration
in seconds, a probability, an amount of money -- has every VIF at 1.00, the
minimum the statistic can take, and a condition number around 1e8. The rule of
thumb is that VIF above 10 is a problem. `condition_indices` is the diagnostic
that sees this one, and it costs the same single call.

**The degrees of freedom are not `p`.** The trace of the ridge hat matrix is
`sum(s^2 / (s^2 + alpha))`, which counts the directions the fit actually used,
weighted by how much of each survived the shrinkage. At a cross-validated alpha
on a badly conditioned design it is routinely a third of `p`. Every standard
error, every `AIC`, every residual degree-of-freedom count printed beside a ridge
fit assumes `p`.

**And the intervals under-cover even when the variance formula is right.**
`ridge_variance` returns the exact sampling variance of the ridge estimator, and
an interval built from it still misses, because ridge is biased on purpose:
`coverage` measures how often the interval contains the truth, and the answer is
not 95 percent. That is not a bug in the formula. It is what "biased estimator"
means, arriving in the one place practitioners do not look for it.

References for where this stops: Hoerl and Kennard, "Ridge regression: biased
estimation for nonorthogonal problems", *Technometrics* 12 (1970); Belsley, Kuh
and Welsch, *Regression Diagnostics* (1980), chapter 3, for condition indices and
why VIF misses the constant term; Hastie, Tibshirani and Friedman, *The Elements
of Statistical Learning*, section 3.4.1, for the effective degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def vif(X, *, has_intercept: bool = True) -> np.ndarray:
    """Variance inflation factor for each predictor column.

    `1 / (1 - R^2_j)` from regressing column `j` on the others. Two properties
    matter more than the number: it is invariant to rescaling any column, and
    with `has_intercept` the constant column is excluded from the list rather
    than scored. Both are reasonable choices and both are why it cannot see the
    ill-conditioning that units and large means produce.
    """
    X = np.asarray(X, dtype=float)
    Z = X[:, 1:] if has_intercept else X
    if Z.shape[1] < 2:
        # A single predictor cannot be collinear with anything, so its VIF is
        # exactly 1 by definition -- which is worth returning rather than
        # erroring, because "VIF says the design is fine" is at its most
        # misleading precisely here.
        return np.ones(max(Z.shape[1], 0))
    out = np.empty(Z.shape[1])
    for j in range(Z.shape[1]):
        others = np.column_stack([np.ones(len(Z)), np.delete(Z, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        resid = Z[:, j] - others @ beta
        ss_tot = float(((Z[:, j] - Z[:, j].mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot else 0.0
        out[j] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return out


def condition_indices(X) -> np.ndarray:
    """`s_max / s_j` for every singular value, largest index last.

    Belsley's diagnostic. Unlike the VIF it is computed on the design as it
    stands — intercept included, columns unscaled — so it sees exactly the
    ill-conditioning the VIF is built to ignore. The last entry is the condition
    number.
    """
    s = np.linalg.svd(np.asarray(X, dtype=float), compute_uv=False)
    return s.max() / s


def shrinkage(singular_values, alpha: float) -> np.ndarray:
    """`s^2 / (s^2 + alpha)`: what ridge leaves of each direction.

    One number per direction, all in [0, 1]. Reading this array is the whole of
    what ridge did — there is no other effect to look for.
    """
    s2 = np.asarray(singular_values, dtype=float) ** 2
    return s2 / (s2 + float(alpha))


def effective_df(singular_values, alpha: float) -> float:
    """`sum(s^2 / (s^2 + alpha))` — the trace of the ridge hat matrix.

    At `alpha = 0` this is the rank of the design, which is `p` for a design of
    full rank; as `alpha` grows it falls smoothly towards zero. It is the number
    of parameters the fit actually spent, and it is the number that belongs in
    any formula where `p` currently sits.
    """
    return float(shrinkage(singular_values, alpha).sum())


@dataclass(frozen=True)
class RidgeFit:
    alpha: float
    coefficients: np.ndarray
    effective_df: float
    #: Coefficients in the basis where the design is diagonal, before and after
    #: the shrinkage, so the per-direction effect is visible rather than mixed
    #: back into the original columns.
    rotated_ols: np.ndarray
    rotated_ridge: np.ndarray
    shrinkage: np.ndarray
    singular_values: np.ndarray


def ridge_fit(X, y, alpha: float) -> RidgeFit:
    """Ridge by SVD, reporting what happened in the eigenbasis as well as out of it."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    f = shrinkage(s, alpha)
    rotated_ols = (U.T @ y) / s
    rotated_ridge = rotated_ols * f
    return RidgeFit(alpha=float(alpha), coefficients=Vt.T @ rotated_ridge,
                    effective_df=float(f.sum()), rotated_ols=rotated_ols,
                    rotated_ridge=rotated_ridge, shrinkage=f, singular_values=s)


def ridge_variance(X, alpha: float, sigma2: float) -> np.ndarray:
    """Exact sampling variance of each ridge coefficient.

    `sigma^2 W X'X W` with `W = (X'X + alpha I)^-1`. This is the *right* formula
    and it is not the reason the intervals miss: it describes the spread of the
    estimator around its own expectation, and ridge's expectation is not the
    truth. The bias is the part no variance formula contains.
    """
    X = np.asarray(X, dtype=float)
    G = X.T @ X
    W = np.linalg.inv(G + float(alpha) * np.eye(G.shape[0]))
    return float(sigma2) * np.diag(W @ G @ W)


def coverage(X, beta, *, alpha: float, sigma: float, reps: int, rng,
             level: float = 0.95) -> dict:
    """How often a nominal `level` ridge interval contains the truth.

    The interval is centred on the ridge estimate and uses `ridge_variance`, which
    is the exact variance — so any shortfall is bias, not a mis-derived formula.
    Reported per coefficient, because the shortfall is not spread evenly: it lands
    on the directions the shrinkage moved most.
    """
    from math import erf, sqrt
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float).ravel()
    # The two-sided normal quantile, without dragging scipy in for one number.
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if erf(mid / sqrt(2.0)) < level:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)

    se = np.sqrt(ridge_variance(X, alpha, sigma ** 2))
    hits = np.zeros(X.shape[1], dtype=int)
    bias = np.zeros(X.shape[1])
    mu = X @ beta
    for _ in range(int(reps)):
        b = ridge_fit(X, mu + rng.normal(0.0, sigma, X.shape[0]), alpha).coefficients
        hits += np.abs(b - beta) <= z * se
        bias += b - beta
    return {"alpha": float(alpha), "level": float(level), "reps": int(reps),
            "coverage": hits / float(reps), "mean_bias": bias / float(reps),
            "standard_error": se, "z": z}


def alpha_for_df(singular_values, target_df: float) -> float:
    """The penalty that spends exactly `target_df` parameters.

    `effective_df` is strictly decreasing in alpha, so a bisection finds this
    without needing a solver. It exists so that ridge and a truncated SVD can be
    compared at equal cost rather than at arbitrary settings of two knobs that
    are not in the same units.
    """
    s = np.asarray(singular_values, dtype=float)
    target = float(target_df)
    if not 0.0 < target < len(s):
        raise ValueError(f"target_df must lie in (0, {len(s)}), got {target}")
    lo, hi = 1e-12, 1e12
    for _ in range(300):
        mid = np.sqrt(lo * hi)              # geometric, because alpha spans decades
        if effective_df(s, mid) > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def truncated_svd_fit(X, y, rank: int) -> np.ndarray:
    """Least squares restricted to the top `rank` singular directions.

    The hard version of the same decision ridge makes softly: keep a direction
    whole or discard it, rather than multiplying it by something between 0 and 1.
    Episode two called the threshold for this `rcond` and deferred the question of
    what it is for; comparing the two at matched `effective_df` is the answer.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    k = int(rank)
    if not 1 <= k <= len(s):
        raise ValueError(f"rank must lie in [1, {len(s)}], got {k}")
    return Vt[:k].T @ ((U[:, :k].T @ y) / s[:k])


def hard_against_soft(X, y, beta_true) -> list[dict]:
    """Truncated SVD at each rank against ridge spending the same parameters."""
    X = np.asarray(X, dtype=float)
    beta_true = np.asarray(beta_true, dtype=float).ravel()
    s = np.linalg.svd(X, compute_uv=False)
    out = []
    for k in range(1, len(s) + 1):
        hard = truncated_svd_fit(X, y, k)
        alpha = 0.0 if k == len(s) else alpha_for_df(s, float(k))
        soft = ridge_fit(X, y, alpha).coefficients
        out.append({
            "rank": k, "alpha": float(alpha),
            "hard_error": float(np.linalg.norm(hard - beta_true)),
            "soft_error": float(np.linalg.norm(soft - beta_true)),
        })
    return out


def ridge_path(X, y, alphas) -> list[RidgeFit]:
    return [ridge_fit(X, y, a) for a in alphas]


def cross_validated_alpha(X, y, alphas, *, folds: int = 5, rng) -> dict:
    """Contiguous-block cross-validation over `alphas`.

    Blocks rather than shuffled folds, and the reason is the same one episode two
    gave for splits: shuffling assumes the rows are exchangeable, which is a claim
    about the data rather than a property of the method.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    edges = np.linspace(0, n, int(folds) + 1).astype(int)
    errors = []
    for a in alphas:
        se = 0.0
        for k in range(int(folds)):
            test = np.zeros(n, dtype=bool)
            test[edges[k]:edges[k + 1]] = True
            b = ridge_fit(X[~test], y[~test], a).coefficients
            se += float(((y[test] - X[test] @ b) ** 2).sum())
        errors.append(se / n)
    errors = np.array(errors)
    best = int(np.argmin(errors))
    return {"alphas": np.asarray(alphas, dtype=float), "cv_error": errors,
            "alpha": float(alphas[best]),
            "effective_df": effective_df(
                np.linalg.svd(X, compute_uv=False), alphas[best])}


def units_design(n: int, *, rng, scales=((3600.0, 600.0), (0.30, 0.10),
                                         (5e7, 1e7))) -> np.ndarray:
    """Mutually uncorrelated columns on wildly different scales, plus an intercept.

    Every VIF comes out at essentially 1 — the smallest value the statistic can
    take — because the columns really are uncorrelated, and the VIF is invariant
    to their scales and blind to the constant column. The condition number is
    around 1e8. This is episode two's exercise design, reused here because the
    point it makes about diagnostics is sharper than any contrived collinearity.
    """
    cols = [np.ones(int(n))]
    for mean, sd in scales:
        cols.append(rng.normal(mean, sd, int(n)))
    return np.column_stack(cols)


def collinear_design(n: int, *, rng, p: int = 8, strength: float = 0.995
                     ) -> np.ndarray:
    """A design whose columns are genuinely near-duplicates of a few directions.

    The other kind of ill-conditioning — the kind the VIF *does* see — so the two
    can be shown side by side and the diagnostic credited with what it is good
    for rather than dismissed.
    """
    base = rng.standard_normal((int(n), 2))
    cols = [np.ones(int(n))]
    for j in range(int(p)):
        anchor = base[:, j % 2]
        cols.append(strength * anchor
                    + np.sqrt(1 - strength ** 2) * rng.standard_normal(int(n)))
    return np.column_stack(cols)
