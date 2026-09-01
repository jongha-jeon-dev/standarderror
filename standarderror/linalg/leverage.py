"""One row, and what it can do to a fit.

Episodes one to five were about *columns*: conditioning, bases, spectra,
penalties. Every claim in them was a property of the design as a whole, and none
of them depended on any particular observation. This module is about the other
axis of the matrix.

The object is the projection onto the column space, `H = X (X'X)^-1 X'`, whose
diagonal entry `h_i` is how much of the *i*-th fitted value comes from the
*i*-th observation. Two facts about that diagonal do all the work.

**Its sum is `p`.** `trace(H) = trace((X'X)^-1 X'X) = p`, always, whatever the
data. So the average leverage is `p/n` and nothing can change that — which means
"high leverage" is never an absolute statement, only a statement relative to
`p/n`, and any threshold you have seen (`2p/n`, `3p/n`) is a multiple of it.

**A single entry can be 1.** Nothing in that identity stops one row taking a
whole unit of the total. When `h_i = 1` the fit passes exactly through that
observation, so its residual is exactly zero, and it disappears from every
diagnostic computed from residuals — including Cook's distance, which has the
residual in its numerator and `(1 - h_i)` in its denominator and evaluates to
0/0. The row with the most influence over the fit is the one the influence
measures cannot see.

The clean case is a dummy variable that is true for `k` rows: those rows have
leverage of at least `1/k`, so a category with one member has leverage 1 exactly.
That is not a pathology anybody constructed. It is what a rare level of a
categorical variable is, and episode two set it aside as "not a conditioning
problem" precisely because it is this one instead.

References for where this stops: Belsley, Kuh and Welsch, *Regression
Diagnostics* (1980), chapters 2 and 3; Cook and Weisberg, *Residuals and
Influence in Regression* (1982); Hoaglin and Welsch, "The hat matrix in
regression and ANOVA", *The American Statistician* 32 (1978).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Leverage this close to one is treated as exactly one. The fitted value then
#: interpolates the observation, and the residual that comes back is rounding
#: noise rather than a small disagreement -- so anything dividing by `1 - h`
#: has to be told to stop rather than allowed to return 1e15.
SATURATED = 1.0 - 1e-10


def hat_diagonal(X) -> np.ndarray:
    """`h_i` for every row, via QR rather than by forming `(X'X)^-1`.

    Episode one's rule, applied to its own module: the explicit inverse is never
    needed. `Q` from a thin QR spans the same column space, so `h_i` is the
    squared norm of its `i`-th row, and the whole diagonal costs one
    factorisation and no solve.
    """
    X = np.asarray(X, dtype=float)
    Q = np.linalg.qr(X)[0]
    return np.einsum("ij,ij->i", Q, Q)


@dataclass(frozen=True)
class LeverageReport:
    h: np.ndarray
    n: int
    p: int

    @property
    def mean(self) -> float:
        """`p/n`, and it is an identity rather than a measurement."""
        return self.p / self.n

    @property
    def max_ratio(self) -> float:
        return float(self.h.max()) / self.mean

    def above(self, multiple: float) -> np.ndarray:
        return np.flatnonzero(self.h > multiple * self.mean)

    @property
    def saturated(self) -> np.ndarray:
        """Rows the fit passes exactly through."""
        return np.flatnonzero(self.h >= SATURATED)


def leverage_report(X) -> LeverageReport:
    X = np.asarray(X, dtype=float)
    return LeverageReport(h=hat_diagonal(X), n=X.shape[0], p=X.shape[1])


def cook_distance(X, y) -> np.ndarray:
    """Cook's distance per row, with the saturated rows reported as `nan`.

    `D_i = e_i^2 h_i / (p s^2 (1 - h_i)^2)`. At `h_i = 1` the numerator and the
    denominator both vanish, and returning `nan` rather than a large number is
    the honest encoding: the statistic is undefined there, not enormous. Every
    implementation that silently returns 0 -- because the squared residual
    underflows before the denominator does -- is reporting that the most
    influential row in the design is the least influential.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    h = hat_diagonal(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    s2 = float((e ** 2).sum()) / max(n - p, 1)
    out = np.full(n, np.nan)
    ok = h < SATURATED
    out[ok] = (e[ok] ** 2 * h[ok]) / (p * s2 * (1.0 - h[ok]) ** 2)
    return out


def dfbeta(X, y) -> np.ndarray:
    """Change in each coefficient from deleting each row, in standard errors.

    Uses the closed form `beta - beta_(i) = (X'X)^-1 x_i e_i / (1 - h_i)`, so no
    refitting is needed. Rows at `h = 1` come back as `nan`: deleting such a row
    does not perturb the fit, it removes a column's only informative
    observation and the coefficient stops existing.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    G_inv = np.linalg.inv(X.T @ X)
    h = hat_diagonal(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    s2 = float((e ** 2).sum()) / max(n - p, 1)
    se = np.sqrt(np.diag(G_inv) * s2)
    out = np.full((n, p), np.nan)
    ok = h < SATURATED
    delta = (G_inv @ X[ok].T) * (e[ok] / (1.0 - h[ok]))
    out[ok] = (delta / se[:, None]).T
    return out


def rare_dummy_design(n: int, k: int, *, rng, extra: int = 2) -> np.ndarray:
    """An intercept, `extra` ordinary columns, and a dummy true for `k` rows.

    Those `k` rows have leverage of at least `1/k`. At `k = 1` it is exactly 1,
    which is the case episode two deferred to episode six.
    """
    n, k = int(n), int(k)
    if not 1 <= k < n:
        raise ValueError(f"k must lie in [1, {n}), got {k}")
    d = np.zeros(n)
    d[rng.choice(n, size=k, replace=False)] = 1.0
    cols = [np.ones(n)] + [rng.standard_normal(n) for _ in range(int(extra))] + [d]
    return np.column_stack(cols)


def deletion_refit(X, y, row: int) -> dict:
    """Actually delete the row and refit, reporting what the closed form cannot.

    For an ordinary row this agrees with `dfbeta`. For a saturated one it shows
    what "undefined" means concretely: the reduced design loses rank, so the
    coefficient it was the only evidence for has no value at all rather than a
    different one.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    keep = np.ones(len(y), dtype=bool)
    keep[int(row)] = False
    full, *_ = np.linalg.lstsq(X, y, rcond=None)
    reduced_rank = int(np.linalg.matrix_rank(X[keep]))
    without, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    return {"row": int(row), "full": full, "without": without,
            "rank_before": int(np.linalg.matrix_rank(X)),
            "rank_after": reduced_rank,
            "lost_rank": reduced_rank < X.shape[1],
            "change": without - full}


def leverage_sweep(sizes, *, n: int, rng, extra: int = 2) -> list[dict]:
    """Leverage of the rare category as its size grows: the 1/k curve."""
    out = []
    for k in sizes:
        X = rare_dummy_design(n, int(k), rng=rng, extra=extra)
        h = hat_diagonal(X)
        rows = np.flatnonzero(X[:, -1] == 1.0)
        out.append({"k": int(k), "leverage": float(h[rows].mean()),
                    "one_over_k": 1.0 / int(k),
                    "mean_leverage": X.shape[1] / int(n)})
    return out
