"""Three ways to solve a least-squares problem, and what each one costs you.

The formula in every regression textbook is

    beta = (X'X)^-1 X'y

and it is correct. It is also, as an algorithm, the worst of the three standard
ways to get the same number — because forming `X'X` **squares** the condition
number, and by episode one we know what a condition number does. A design matrix
with kappa = 1e8 is uncomfortable but workable; its Gram matrix has kappa = 1e16,
which is nothing left at all. The closed form taught as the answer is a bad way
to compute the answer.

The three methods, in the order they lose accuracy:

* `solve_svd` — factor X = U S V', invert the singular values, truncate the ones
  that are numerically zero. Slowest, most robust, and the only one that gives a
  defined answer when the columns are exactly collinear.
* `solve_qr` — factor X = QR with Q orthonormal and R upper triangular, then
  solve R beta = Q'y. Costs about twice a normal-equations solve and gives an
  error proportional to kappa(X) rather than kappa(X)^2. What every serious
  library actually calls.
* `solve_normal` — form the Gram matrix and solve. Cheapest, and the error is
  proportional to kappa(X)^2.

The geometry underneath all three is one picture. The columns of X span a
subspace of R^n — every prediction the model can possibly make — and y is a point
that generally does not lie in it. Least squares asks for the closest point that
does, which is the foot of the perpendicular from y onto that subspace. So the
residual is orthogonal to every column of X, which is exactly the normal
equations written as a geometric statement: `X'(y - X beta) = 0`.

That is why QR wins without doing anything clever. Q is an orthonormal basis for
the same subspace, and projecting onto an orthonormal basis needs no inversion at
all — the coefficients are just inner products. The normal equations reach the
same place by first building `X'X`, a matrix whose only purpose is to undo the
non-orthogonality of the columns, and that undoing is where the digits go.

`projection_report` measures that orthogonality, and measuring it turns out to
teach the opposite of what one expects. At degree 13 the normal-equations
coefficients are wrong by 321% and their residual is orthogonal to the columns to
6e-16 — as orthogonal as the SVD's. It could not be otherwise: `X'X beta = X'y`
*is* the orthogonality condition, so a method that solves it is enforcing exactly
the property you would use to audit it. A diagnostic derived from the equations a
method solves cannot detect that the method lost accuracy solving them. The
residual norm does degrade — 6e-08 against 6e-16 — but it degrades by eight
orders while the coefficients lose fourteen, and 6e-08 still reads as fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conditioning import DIGITS_AVAILABLE, condition_number

#: Methods in the order this module recommends them.
METHODS = ("svd", "qr", "normal")


def solve_normal(X, y) -> np.ndarray:
    """Form the Gram matrix and solve. The textbook formula, computed literally.

    Note it does *not* form an explicit inverse — that would be worse again, for
    the reason episode one measured. Even at its best, this route pays
    kappa(X)^2.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.solve(X.T @ X, X.T @ y)


def solve_qr(X, y) -> np.ndarray:
    """Factor X = QR and solve R beta = Q'y.

    Q has orthonormal columns spanning the same space as X's, so `Q.T @ y` is the
    projection coefficients in a basis that needs no correction. R is triangular,
    so the final solve is back-substitution with no conditioning cost of its own
    beyond kappa(R) = kappa(X).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Q, R = np.linalg.qr(X)
    return np.linalg.solve(R, Q.T @ y)


def solve_svd(X, y, *, rcond: float | None = None) -> np.ndarray:
    """Factor X = U S V' and invert the singular values, truncating tiny ones.

    Written out rather than delegated to `lstsq` because the truncation is the
    interesting part and `lstsq` hides it. Singular values below
    `rcond * s_max` are treated as zero and their directions dropped, which is
    what makes this the only one of the three with a defined answer when two
    columns are exactly equal: it returns the minimum-norm solution instead of
    an arbitrary one.

    The default `rcond` follows numpy's: `max(n, p) * eps`.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if rcond is None:
        rcond = max(X.shape) * float(np.finfo(float).eps)
    keep = s > rcond * s[0]
    inv = np.zeros_like(s)
    inv[keep] = 1.0 / s[keep]
    return Vt.T @ (inv * (U.T @ y))


_SOLVERS = {"normal": solve_normal, "qr": solve_qr, "svd": solve_svd}


@dataclass
class MethodReport:
    """One solver on one problem, against a known answer."""

    method: str
    error: float           # relative, against the true coefficients
    residual: float        # relative, ||y - X beta|| / ||y||
    orthogonality: float   # ||X'(y - X beta)|| / (||X|| ||y||)
    digits_correct: float

    def row(self) -> list[str]:
        return [self.method, f"{self.error:.2e}", f"{self.residual:.1e}",
                f"{self.orthogonality:.1e}", f"{self.digits_correct:.1f}"]


def projection_report(X, y, beta) -> dict:
    """Residual, and how nearly orthogonal it is to the columns of X.

    Orthogonality is the defining property of a least-squares solution, so how
    badly it fails is a direct measure of how badly the solve failed — and unlike
    the coefficient error it needs no known truth, which makes it the diagnostic
    you can actually run on real data.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    r = y - X @ np.asarray(beta, dtype=float)
    scale = np.linalg.norm(X, 2) * np.linalg.norm(y)
    return {"residual": float(np.linalg.norm(r) / np.linalg.norm(y)),
            "orthogonality": float(np.linalg.norm(X.T @ r) / scale)
            if scale else float("nan")}


def compare_methods(X, beta_true, *, methods=METHODS) -> list[MethodReport]:
    """Run each solver on `y = X beta_true` and report against the truth.

    `y` is formed exactly, with no noise, so every error reported is numerical.
    That is the point of the exercise: with noise added, the statistical error
    swamps the numerical one at low degree and *hides* the failure at high
    degree, which is how this goes unnoticed in practice.
    """
    X = np.asarray(X, dtype=float)
    beta_true = np.asarray(beta_true, dtype=float)
    if X.shape[1] != beta_true.size:
        raise ValueError(f"X has {X.shape[1]} columns and beta_true has "
                         f"{beta_true.size} entries")
    y = X @ beta_true
    out = []
    for name in methods:
        if name not in _SOLVERS:
            raise ValueError(f"unknown method {name!r}; have {sorted(_SOLVERS)}")
        beta = _SOLVERS[name](X, y)
        err = float(np.linalg.norm(beta - beta_true)
                    / np.linalg.norm(beta_true))
        pr = projection_report(X, y, beta)
        out.append(MethodReport(
            method=name, error=err, residual=pr["residual"],
            orthogonality=pr["orthogonality"],
            digits_correct=0.0 if err <= 0 else max(0.0, float(-np.log10(err)))))
    return out


def squaring_report(X) -> dict:
    """kappa(X) against kappa(X'X), and the digits each one leaves.

    The whole argument in four numbers. The ratio is kappa(X) itself, up to
    rounding, because kappa(X'X) = kappa(X)^2 exactly in the two-norm — the
    singular values of X'X are the squares of those of X.
    """
    X = np.asarray(X, dtype=float)
    kx = condition_number(X)
    kg = condition_number(X.T @ X)
    return {"kappa_X": kx, "kappa_gram": kg, "ratio": kg / kx if kx else np.inf,
            "digits_qr": max(0.0, DIGITS_AVAILABLE - np.log10(kx)),
            "digits_normal": max(0.0, DIGITS_AVAILABLE - np.log10(kg))}


def scaling_variants(X, *, intercept: int | None = 0) -> dict:
    """kappa for the same design raw, centred, scaled and standardised.

    The answer to episode one's exercise. `intercept` names a column to leave
    alone — centring a column of ones destroys it, and scaling it does nothing —
    or `None` if there is no intercept.

    Returned as a dict of arrays rather than just numbers so a caller can check
    that the four designs really do fit the same model.
    """
    X = np.asarray(X, dtype=float)
    cols = [j for j in range(X.shape[1]) if j != intercept]
    if not cols:
        raise ValueError("nothing to scale: every column is the intercept")
    mean = X[:, cols].mean(axis=0)
    sd = X[:, cols].std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError(f"columns {[cols[i] for i in np.flatnonzero(sd == 0)]} "
                         f"are constant; scaling them divides by zero")

    out = {}
    for name in ("raw", "centred", "scaled", "standardised"):
        Z = X.copy()
        if name in ("centred", "standardised"):
            Z[:, cols] = Z[:, cols] - mean
        if name in ("scaled", "standardised"):
            Z[:, cols] = Z[:, cols] / sd
        out[name] = {"design": Z, "kappa": condition_number(Z)}
    return out
