"""The condition number, and the calculations it quietly ruins.

The one-line version: **a condition number converts the precision of your inputs
into an error bar on your answer**, and it does so before any statistics are
involved. Solve `A x = b` and perturb `b` by a relative amount `eps`, and the
solution can move by

    ||dx|| / ||x||  <=  kappa(A) * ||db|| / ||b||,        kappa(A) = s_max / s_min

Nothing in that is about noisy data. Storing `b` in double precision already
perturbs it by about 1e-16, so `kappa(A) * 1e-16` is a floor on how well `x` can
be known *at all* — and `log10(kappa)` is, to within a digit or so, the number of
decimal digits you have lost by the time the answer reaches you.

Two things about that make it worth a module rather than a footnote.

**The residual does not warn you.** A backward-stable solver returns an `x_hat`
whose residual `||A x_hat - b||` is at machine precision no matter how badly
conditioned `A` is. So the usual sanity check — "the residual is 1e-16, the solve
worked" — is exactly the check that cannot detect this failure. `solve_report`
returns both numbers side by side so the gap is visible.

**The badly conditioned matrices are not exotic.** The Hilbert matrix is the
standard textbook example of hopeless conditioning, and it is also, exactly, the
Gram matrix of the monomials 1, t, t^2, ... on the unit interval. Fitting a
polynomial by normal equations *is* solving a Hilbert system;
`gram_condition` shows the same numbers arriving from a regression rather than
from a curiosity, and shows an orthogonal basis removing fifteen orders of
magnitude of the problem without changing the model being fitted.

References for where this stops: Trefethen and Bau, *Numerical Linear Algebra*,
lectures 12 and 18; Higham, *Accuracy and Stability of Numerical Algorithms*,
chapters 1 and 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Unit roundoff for IEEE double precision: the relative spacing of the floats.
#: Every input is already perturbed by about this much before any solver runs,
#: which is why it appears in an error bound that mentions no data error.
MACHINE_EPS = float(np.finfo(float).eps)

#: Decimal digits a double carries. `log10(1/MACHINE_EPS)`, to be exact about
#: where the "you had sixteen digits" claim comes from.
DIGITS_AVAILABLE = float(-np.log10(MACHINE_EPS))


def hilbert(n: int) -> np.ndarray:
    """The n-by-n Hilbert matrix, `H[i, j] = 1 / (i + j + 1)` in 0-based indices.

    Symmetric, positive definite, and with a condition number that grows like
    `exp(3.5 n)` — so it runs out of double precision at about n = 12. It is the
    Gram matrix of the monomials on [0, 1]: `integral_0^1 t^i t^j dt` is exactly
    `1 / (i + j + 1)`, which is why a polynomial fit meets it by accident.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    i = np.arange(1, n + 1, dtype=float)
    return 1.0 / (i[:, None] + i[None, :] - 1.0)


def condition_number(A) -> float:
    """Two-norm condition number, `s_max / s_min`, from the singular values.

    Computed from the SVD rather than as `norm(A) * norm(inv(A))`: the second
    form needs an inverse, which is both slower and less accurate, and on a
    matrix this badly conditioned the inverse is the thing being questioned.
    Returns infinity when a singular value is exactly zero rather than raising,
    because "the condition number is infinite" is the correct answer and a caller
    sweeping a parameter should not have to catch it. Note that exact zeros are
    rare: a matrix that is singular on paper usually has a smallest singular
    value near 1e-16 instead, so kappa comes back enormous and finite. That is
    the reason to ask for a condition number rather than for a rank — the answer
    to "is it singular?" is almost always no, and almost always unhelpful.
    """
    s = np.linalg.svd(np.asarray(A, dtype=float), compute_uv=False)
    if s[-1] == 0:
        return float("inf")
    return float(s[0] / s[-1])


def digits_lost(A) -> float:
    """`log10(kappa)`: decimal digits of the input that the answer cannot keep."""
    k = condition_number(A)
    return float("inf") if not np.isfinite(k) else float(np.log10(k))


def perturbation_bound(A, relative_perturbation: float = MACHINE_EPS) -> float:
    """`kappa(A) * eps`: the worst-case relative error in `x` from one in `b`.

    Worst case over the direction of the perturbation, so a measured error below
    it is expected; on a Hilbert system a random perturbation lands within a
    factor of about three, which is close enough that the bound is a prediction
    rather than a reassurance.
    """
    if relative_perturbation < 0:
        raise ValueError("a relative perturbation cannot be negative")
    return condition_number(A) * float(relative_perturbation)


@dataclass
class SolveReport:
    """What a solve actually delivered, next to what it looked like it delivered.

    `residual` is the number people check and `error` is the number they care
    about. On a badly conditioned system the first is at machine precision while
    the second is arbitrarily large, which is the whole point.
    """

    n: int
    method: str
    kappa: float
    digits_lost: float
    error: float           # relative, against the known solution
    residual: float        # relative, ||A x_hat - b|| / ||b||
    digits_correct: float

    def row(self) -> list[str]:
        return [str(self.n), f"{self.kappa:.2e}", f"{self.digits_lost:.1f}",
                f"{self.residual:.1e}", f"{self.error:.2e}",
                f"{self.digits_correct:.1f}"]


def solve_report(A, x_true=None, *, method: str = "solve") -> SolveReport:
    """Solve `A x = b` for a known `x` and report error and residual together.

    `b` is formed as `A @ x_true`, so the exact answer is known and the error is
    a fact rather than an estimate. `method` is `"solve"` for `np.linalg.solve`
    or `"inv"` for the `inv(A) @ b` spelling, which is the habit this module
    exists to argue against.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square matrix, got {A.shape}")
    x = np.ones(n) if x_true is None else np.asarray(x_true, dtype=float)
    b = A @ x

    if method == "solve":
        x_hat = np.linalg.solve(A, b)
    elif method == "inv":
        x_hat = np.linalg.inv(A) @ b
    else:
        raise ValueError(f"method must be 'solve' or 'inv', got {method!r}")

    err = float(np.linalg.norm(x_hat - x) / np.linalg.norm(x))
    res = float(np.linalg.norm(A @ x_hat - b) / np.linalg.norm(b))
    # Digits of the answer that survived. Floored at zero: "minus two digits
    # correct" is not a quantity, and a table reading -2.1 invites the reader to
    # interpret a number that means nothing.
    correct = 0.0 if err <= 0 else max(0.0, float(-np.log10(err)))
    return SolveReport(n=n, method=method, kappa=condition_number(A),
                       digits_lost=digits_lost(A), error=err, residual=res,
                       digits_correct=correct)


def perturb_and_solve(A, *, relative: float = 1e-10, x_true=None,
                      seed: int = 0, reps: int = 200) -> dict:
    """Nudge `b` by a relative amount and measure how far `x` moves.

    The nudge is a random direction scaled to `relative * ||b||`, which is the
    honest version of "my measurements are good to ten decimal places".

    `reps` directions are drawn and the **worst** is reported, because the bound
    is a worst case over directions and comparing it against one lucky draw makes
    it look loose when it is not. The mean is returned too, so the gap between
    typical and worst is visible: a single random direction lands well inside the
    bound, and it takes a search to approach it.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    x = np.ones(n) if x_true is None else np.asarray(x_true, dtype=float)
    b = A @ x
    if reps < 1:
        raise ValueError(f"reps must be at least 1, got {reps}")
    rng = np.random.default_rng(seed)
    scale = float(relative) * np.linalg.norm(b)
    errors = np.empty(int(reps))
    for r in range(int(reps)):
        d = rng.standard_normal(n)
        d *= scale / np.linalg.norm(d)
        x_hat = np.linalg.solve(A, b + d)
        errors[r] = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
    bound = perturbation_bound(A, relative)
    worst = float(errors.max())
    return {"n": n, "kappa": condition_number(A), "relative": float(relative),
            "reps": int(reps), "worst": worst, "typical": float(np.median(errors)),
            "bound": bound,
            "tightness": worst / bound if bound else float("nan")}


def design_matrix(degree: int, *, basis: str = "monomial",
                  n_points: int = 200) -> np.ndarray:
    """Polynomial design matrix on [0, 1], in the monomial or Legendre basis.

    Both bases span exactly the same space, so they fit exactly the same model
    and — in exact arithmetic — give exactly the same predictions. Only the
    conditioning differs, which is the cleanest available demonstration that
    conditioning is a property of the *parameterisation* rather than of the
    problem.
    """
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    if n_points <= degree:
        raise ValueError(f"need more points than columns: {n_points} <= {degree}")
    t = np.linspace(0.0, 1.0, int(n_points))
    if basis == "monomial":
        return np.vander(t, degree + 1, increasing=True)
    if basis == "legendre":
        from numpy.polynomial import legendre
        u = 2.0 * t - 1.0          # Legendre is orthogonal on [-1, 1]
        eye = np.eye(degree + 1)
        return np.column_stack([legendre.legval(u, eye[k])
                                for k in range(degree + 1)])
    raise ValueError(f"basis must be 'monomial' or 'legendre', got {basis!r}")


def gram_condition(degree: int, *, basis: str = "monomial",
                   n_points: int = 200) -> float:
    """Condition number of `X'X` for a polynomial fit of this degree.

    The quantity a normal-equations solve is actually up against. For the
    monomial basis it tracks `condition_number(hilbert(degree + 1))` closely,
    which is not a coincidence: the two matrices are the same Gram matrix, one
    integrated and one sampled.
    """
    X = design_matrix(degree, basis=basis, n_points=n_points)
    return condition_number(X.T @ X)


def equilibrate(A) -> tuple[np.ndarray, np.ndarray]:
    """Scale each column to unit two-norm; return the scaled matrix and the scales.

    The cheapest thing that ever helps, and the one worth trying before anything
    clever: a condition number is not invariant to the units of the columns, so a
    design matrix holding a duration in seconds beside a probability is badly
    conditioned for a reason that has nothing to do with its statistics. Columns
    that are entirely zero are left alone rather than dividing by zero — a zero
    column is a rank problem, and rank is episode five.
    """
    A = np.asarray(A, dtype=float)
    scales = np.linalg.norm(A, axis=0)
    safe = np.where(scales > 0, scales, 1.0)
    return A / safe, scales
