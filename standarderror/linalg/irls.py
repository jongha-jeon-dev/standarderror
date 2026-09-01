"""Logistic regression as repeated weighted least squares, and where it stops.

Every fit in this series so far had a closed form. Logistic regression does not:
the likelihood equations are non-linear in the coefficients, so the fit is found
by iterating. What is iterated, though, is exactly the machinery of the previous
seven episodes. Newton's method on the log-likelihood, written out, is

    beta <- (X' W X)^{-1} X' W z,       W = diag(p (1 - p)),  z = eta + (y - p)/W

which is a weighted least squares problem with a working response `z` and weights
recomputed from the current fit. So `X' W X` inherits every conditioning problem
`X' X` had, plus a new one: `W` depends on the fit, and as the fit sharpens, `W`
goes to zero.

That last sentence is the whole module. Three consequences, each with a function
here to measure it.

* **Separation.** If a hyperplane separates the classes perfectly, the likelihood
  has no maximum -- it increases without bound as the coefficients grow, and the
  supremum is approached rather than attained. The MLE does not exist. Software
  does not raise; it returns whatever it had when it ran out of iterations.
* **A vanishing weight matrix.** As fitted probabilities approach 0 or 1 the
  weights `p(1-p)` approach zero, so `X' W X` approaches singular *no matter how
  well-conditioned X is*. Measured on the partial-separation design below, while
  the weights are still real numbers the diverging coefficient grows by **exactly
  1 per iteration** and `cond(X' W X)` grows by **exactly a factor of e per
  iteration** -- the second follows from the first, because the weight on a
  saturated row is proportional to `exp(-|eta|)`. Once the library's weight floor
  pins the weights the growth slows to logarithmic. It does not stop.
* **A standard error computed from that matrix.** `(X' W X)^{-1}` is the reported
  covariance. Under separation it is the inverse of a nearly singular matrix, and
  the standard error it produces grows in step with the coefficient. Which is why
  a separated coefficient does not usually look significant -- it looks *enormous
  and imprecise*, and the danger is the case of **partial** separation, where one
  coefficient diverges quietly inside an otherwise ordinary-looking table.

Nothing here is a criticism of IRLS. It converges quadratically when the MLE
exists, and the weighted normal equations are the right way to compute it. The
point is that "did not converge" and "converged to the wrong thing" are different
failures with the same symptom, and the second one is not detectable from the
coefficient table.

References: Nelder and Wedderburn, "Generalized linear models", *JRSS A* 135
(1972), for the IRLS formulation; Albert and Anderson, "On the existence of
maximum likelihood estimates in logistic regression models", *Biometrika* 71
(1984), for the separation conditions; Firth, "Bias reduction of maximum
likelihood estimates", *Biometrika* 80 (1993), for the penalty that restores
existence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Below this, a fitted probability is treated as saturated: the weight it
#: contributes is under 1e-12 and the row has effectively left the fit.
SATURATED_P = 1e-6

#: The weight floor scikit-learn, statsmodels and R's glm all apply in some form,
#: to stop `X' W X` becoming exactly singular. It changes the answer, silently.
WEIGHT_FLOOR = 1e-10


def sigmoid(eta):
    """The logistic function, computed so it does not overflow.

    `1 / (1 + exp(-eta))` overflows for `eta` below about -710 and returns a
    warning plus the right answer. Under separation `eta` reaches that range in a
    dozen iterations, so the naive form floods the output with warnings from the
    exact case this module is about.
    """
    eta = np.asarray(eta, dtype=float)
    out = np.empty_like(eta)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    e = np.exp(eta[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def log_likelihood(X, y, beta) -> float:
    """The logistic log-likelihood, via `logaddexp` rather than `log(sigmoid())`.

    `y log p + (1-y) log(1-p)` evaluates to `-inf` as soon as a fitted
    probability rounds to 0 or 1, which under separation happens long before the
    likelihood stops improving -- so the naive form reports that the fit stopped
    getting better while it is still getting better. The identity
    `log p = -log(1 + exp(-eta))` has no such range problem.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    eta = X @ np.asarray(beta, dtype=float)
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


@dataclass
class IRLSFit:
    """One logistic fit, with the diagnostics the coefficient table omits."""
    beta: np.ndarray
    iterations: int
    converged: bool
    log_likelihood: float
    #: The largest |beta| at each iteration. A diverging fit shows a straight
    #: line here; a converging one flattens.
    path: np.ndarray
    #: `cond(X' W X)` at each iteration, which is the quantity that explains why.
    weighted_condition: np.ndarray
    #: The smallest weight in the final `W`.
    min_weight: float
    #: Standard errors from `(X' W X)^{-1}`, or nan where it is not invertible.
    standard_errors: np.ndarray
    #: Fitted probabilities at machine 0 or 1: rows that have left the fit.
    saturated: int
    notes: list[str] = field(default_factory=list)

    @property
    def largest_coefficient(self) -> float:
        return float(np.max(np.abs(self.beta)))

    @property
    def largest_z(self) -> float:
        """The biggest |coefficient / standard error| in the table."""
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.abs(self.beta) / self.standard_errors
        return float(np.nanmax(z)) if np.isfinite(z).any() else float("nan")


def irls(X, y, *, max_iter: int = 50, tol: float = 1e-10,
         weight_floor: float = WEIGHT_FLOOR, ridge: float = 0.0) -> IRLSFit:
    """Fit by iteratively reweighted least squares, recording what it did.

    Each step solves the weighted normal equations by QR on the whitened design
    rather than by forming and inverting `X' W X` -- the same choice episode one
    argued for, and it matters more here because `W` is what is going singular.

    `ridge` adds `alpha I` to the weighted cross-product, which is episode five's
    penalty applied to a problem whose ill-conditioning the fit created. It makes
    the maximum exist again, at the cost of a coefficient that is no longer a
    maximum likelihood estimate.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if y.shape != (n,):
        raise ValueError(f"y has shape {y.shape}, expected ({n},)")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be 0/1; this is the Bernoulli case only")

    beta = np.zeros(p)
    path, conds, notes = [], [], []
    converged = False
    it = 0
    while it < int(max_iter):
        it += 1
        eta = X @ beta
        mu = sigmoid(eta)
        w = np.maximum(mu * (1.0 - mu), weight_floor)
        # The working response. Written as eta + (y - mu)/w rather than solved
        # for, so the weighted problem is visibly a least squares problem.
        z = eta + (y - mu) / w
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        conds.append(float(np.linalg.cond(Xw.T @ Xw)))
        if ridge:
            aug = np.vstack([Xw, np.sqrt(float(ridge)) * np.eye(p)])
            rhs = np.concatenate([sw * z, np.zeros(p)])
            step = np.linalg.lstsq(aug, rhs, rcond=None)[0]
        else:
            step = np.linalg.lstsq(Xw, sw * z, rcond=None)[0]
        delta = float(np.max(np.abs(step - beta)))
        beta = step
        path.append(float(np.max(np.abs(beta))))
        if delta < float(tol):
            converged = True
            break

    eta = X @ beta
    mu = sigmoid(eta)
    w = mu * (1.0 - mu)
    saturated = int(np.sum((mu < SATURATED_P) | (mu > 1.0 - SATURATED_P)))
    XtWX = X.T @ (X * np.maximum(w, weight_floor)[:, None])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(XtWX)))
    except np.linalg.LinAlgError:                       # pragma: no cover
        se = np.full(p, np.nan)
        notes.append("X'WX is singular; no standard errors")
    if not converged:
        notes.append(f"did not converge in {max_iter} iterations")
    if saturated:
        notes.append(f"{saturated} of {n} fitted probabilities are at 0 or 1")
    return IRLSFit(beta=beta, iterations=it, converged=converged,
                   log_likelihood=log_likelihood(X, y, beta),
                   path=np.asarray(path), weighted_condition=np.asarray(conds),
                   min_weight=float(w.min()), standard_errors=se,
                   saturated=saturated, notes=notes)


# --------------------------------------------------------------- separation

def separation_lp(X, y) -> dict:
    """Is there a vector `b` with `X b > 0` on the ones and `< 0` on the zeros?

    Albert and Anderson's condition, and the honest way to answer the question:
    complete separation is a property of the *design*, decidable before any
    fitting, and it is decided by feasibility of a linear system rather than by
    watching an iteration misbehave.

    Solved here as an unconstrained least-squares surrogate that is exact for the
    purpose: maximise the smallest margin `s_i (X b)_i` subject to `||b|| = 1`. A
    strictly positive optimum means separated. Rather than call an LP solver,
    this uses the fact that a separating direction exists iff the origin is not
    in the convex hull of the signed rows, and finds it by projected gradient --
    a dozen lines, no dependency, and the margin it returns is interpretable.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    S = X * np.where(y > 0.5, 1.0, -1.0)[:, None]
    b = np.linalg.lstsq(S, np.ones(len(S)), rcond=None)[0]
    nb = np.linalg.norm(b)
    b = b / nb if nb > 0 else np.ones(X.shape[1]) / np.sqrt(X.shape[1])
    for _ in range(2000):
        m = S @ b
        # Push away from the worst-violated rows only: this is the subgradient of
        # the smallest margin, so it maximises the margin rather than a sum.
        worst = m <= np.quantile(m, 0.05)
        g = S[worst].sum(axis=0)
        ng = np.linalg.norm(g)
        if ng == 0:
            break
        b = b + 0.05 * g / ng
        b /= np.linalg.norm(b)
    margin = float((S @ b).min())
    return {"separated": margin > 1e-9, "margin": margin, "direction": b}


def empty_cell_check(X, y, *, names=None) -> list[dict]:
    """Flag every binary column one of whose levels has only one outcome value.

    The linear-programming test above answers the question Albert and Anderson
    posed -- is there a hyperplane separating the classes -- and it correctly
    says *no* for the case that actually reaches production, where a dummy true
    for 4% of rows has outcome 1 on every one of those rows. There is no
    separating hyperplane, because the other 96% contain both classes; the
    likelihood is still unbounded, in that one coefficient. So the complete-
    separation test is not the check you need, and this is.

    A 2x2 table with an empty cell. That is all it takes.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for j in range(X.shape[1]):
        col = X[:, j]
        levels = np.unique(col)
        if len(levels) != 2:
            continue
        for lv in levels:
            m = col == lv
            if m.sum() and len(np.unique(y[m])) == 1:
                out.append({
                    "column": j,
                    "name": (names[j] if names is not None else f"x{j}"),
                    "level": float(lv), "rows": int(m.sum()),
                    "outcome": float(y[m][0]),
                    "unbounded": True,
                })
    return out


def floor_determined_error(X, y, *, index: int, floors=(1e-8, 1e-10, 1e-12),
                           max_iter: int = 100) -> list[dict]:
    """The same separated fit under different weight floors.

    Under saturation `X' W X` is `sum_i w_i x_i x_i'` with every relevant `w_i`
    pinned at the floor, so the reported standard error is
    `1 / sqrt(k * floor)` for a dummy true on `k` rows -- a function of a library
    constant and a category size, containing no information from the data. This
    reports that number beside the closed form so the claim can be checked rather
    than believed.
    """
    out = []
    for f in floors:
        fit = irls(X, y, max_iter=int(max_iter), weight_floor=float(f))
        k = int(np.count_nonzero(np.asarray(X, dtype=float)[:, index]))
        out.append({"weight_floor": float(f),
                    "coefficient": float(fit.beta[index]),
                    "standard_error": float(fit.standard_errors[index]),
                    "closed_form": float(1.0 / np.sqrt(k * f)),
                    "rows_in_category": k})
    return out


def separable_design(n: int, *, p: int = 2, rng, gap: float = 0.5) -> tuple:
    """A design where one column separates the classes completely.

    Not a pathology anyone constructs on purpose. It is what a rare outcome plus
    a dummy for a small category produces on its own, which is the version of
    this that reaches production.
    """
    x = rng.standard_normal(n)
    y = (x > gap).astype(float)
    cols = [np.ones(n), x]
    for _ in range(p - 2):
        cols.append(rng.standard_normal(n))
    return np.column_stack(cols), y


def quiet_separation_design(n: int, *, rng, share: float = 0.04) -> tuple:
    """The dangerous case: one column separates, the rest of the table is fine.

    A dummy true for a small share of rows, and every row where it is true has
    outcome 1. The other coefficients are estimated normally and look normal. The
    dummy's coefficient has no maximum likelihood estimate, and nothing in the
    output says so.
    """
    k = max(int(round(n * float(share))), 2)
    d = np.zeros(n)
    d[rng.choice(n, k, replace=False)] = 1.0
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    eta = -0.3 + 0.8 * x1 - 0.5 * x2
    y = (rng.random(n) < sigmoid(eta)).astype(float)
    y[d > 0.5] = 1.0                       # the separation, arriving quietly
    return np.column_stack([np.ones(n), x1, x2, d]), y, int(k)


def iteration_sweep(X, y, iterations, **kw) -> list[dict]:
    """The same fit stopped at different iteration counts.

    The point of the table this builds: under separation every column of it moves
    with `max_iter` and none of them stabilises, so "the coefficient" is not a
    property of the data -- it is a property of the convergence tolerance.
    """
    out = []
    for m in iterations:
        fit = irls(X, y, max_iter=int(m), **kw)
        out.append({"max_iter": int(m), "iterations": fit.iterations,
                    "converged": fit.converged,
                    "largest_coefficient": fit.largest_coefficient,
                    "largest_z": fit.largest_z,
                    "log_likelihood": fit.log_likelihood,
                    "min_weight": fit.min_weight,
                    "condition": float(fit.weighted_condition[-1]),
                    "saturated": fit.saturated})
    return out


def ridge_sweep(X, y, alphas, *, index: int = -1, **kw) -> list[dict]:
    """Episode five's penalty against separation, at several strengths.

    `index` selects the coefficient to report -- the separated one, normally.
    """
    out = []
    for a in alphas:
        fit = irls(X, y, ridge=float(a), **kw)
        out.append({"alpha": float(a), "converged": fit.converged,
                    "iterations": fit.iterations,
                    "coefficient": float(fit.beta[index]),
                    "standard_error": float(fit.standard_errors[index]),
                    "condition": float(fit.weighted_condition[-1]),
                    "log_likelihood": fit.log_likelihood})
    return out


def newton_vs_gradient(X, y, *, steps: int = 25, lr: float = 0.1,
                       rng=None) -> dict:
    """Newton's quadratic convergence against a fixed-step gradient ascent.

    Included because the episode claims IRLS *is* Newton's method, and the
    cheapest way to make that claim checkable is to show it converging at the
    rate Newton's method converges at, on a problem where the maximum exists.

    The gradient comparison is a fixed step of `lr`, which is a weak opponent on
    purpose -- it is there to show what having the Hessian buys, not to argue
    that first-order methods cannot solve this. A tuned step size, or any of the
    accelerated methods, closes most of the gap.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    target = irls(X, y, max_iter=200).beta
    both = {}
    for name in ("newton", "gradient"):
        beta = np.zeros(X.shape[1])
        errs = []
        for _ in range(int(steps)):
            mu = sigmoid(X @ beta)
            if name == "newton":
                w = np.maximum(mu * (1.0 - mu), WEIGHT_FLOOR)
                sw = np.sqrt(w)
                z = X @ beta + (y - mu) / w
                beta = np.linalg.lstsq(X * sw[:, None], sw * z, rcond=None)[0]
            else:
                beta = beta + float(lr) * (X.T @ (y - mu)) / len(y)
            errs.append(float(np.linalg.norm(beta - target)))
        both[name] = np.asarray(errs)
    return both


def well_posed_design(n: int, *, rng, beta=(-0.5, 1.0, -0.8)) -> tuple:
    """An ordinary logistic design, for the cases that are supposed to work."""
    b = np.asarray(beta, dtype=float)
    X = np.column_stack([np.ones(n)]
                        + [rng.standard_normal(n) for _ in range(len(b) - 1)])
    y = (rng.random(n) < sigmoid(X @ b)).astype(float)
    return X, y
