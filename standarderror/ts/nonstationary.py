"""Spurious regression, unit roots, and cointegration — with simulated nulls.

The thing this module exists to make checkable
----------------------------------------------

Regress one random walk on another, independent one. The two share no cause by
construction. The t-statistic on the slope will nonetheless exceed 1.96 far more
often than 5% of the time, and — this is the part that gets left out of the
usual telling — **the rejection rate does not settle down as the sample grows.
It goes to one.**

That is Phillips (1986). Under a unit root the OLS t-statistic has no limiting
distribution at all; `t / sqrt(T)` does, converging to a ratio of functionals of
Brownian motion. So the familiar reassurance "with enough data the noise averages
out" is exactly backwards here: more data makes a spurious result *more*
significant, not less.

Everything below is built so that claim can be falsified:

* `spurious_rejection_rate` measures the rate at several sample sizes. If the
  theory were wrong the curve would flatten. It does not.
* `scaled_t_quantiles` measures the spread of `t / sqrt(T)`. If the theory were
  wrong this would keep widening. It does not.
* `df_critical_values` and `eg_critical_values` build the two null distributions
  by simulation, so they can be compared against MacKinnon's published response
  surface — an answer computed a different way by a different person.

Why the critical values are simulated rather than looked up
-----------------------------------------------------------

The Engle-Granger test runs a Dickey-Fuller regression on residuals from a
*fitted* cointegrating regression. Because OLS chose the combination that makes
those residuals as small as possible, they look more stationary than any true
error would, and the null distribution shifts left. Using the Dickey-Fuller table
on Engle-Granger residuals is therefore not conservative — it over-rejects, and
`MISUSE` quantifies by how much.

Conventions
-----------

`trend="c"` puts a constant in the test regression, `"ct"` a constant and linear
trend, `"n"` neither — the same vocabulary statsmodels uses, so results are
directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "random_walk", "OLSFit", "ols", "adf_stat", "df_critical_values",
    "engle_granger_stat", "eg_critical_values", "spurious_draws",
    "spurious_rejection_rate",
    "scaled_t_quantiles", "MACKINNON_DF", "MACKINNON_EG", "misuse_size",
    "correlation_of_independent_walks",
]

# MacKinnon (2010), "Critical Values for Cointegration Tests", asymptotic
# (T -> infinity) response-surface values. These are NOT used in any
# computation — they exist only so the simulated values have something
# external to be wrong against.
MACKINNON_DF = {          # unit-root test, one series, constant in regression
    0.01: -3.4302, 0.05: -2.8615, 0.10: -2.5668,
}
MACKINNON_EG = {          # cointegration test, N=2 series, constant
    0.01: -3.9001, 0.05: -3.3377, 0.10: -3.0462,
}


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def random_walk(n: int, reps: int = 1, *, rng: np.random.Generator,
                drift: float = 0.0, sigma: float = 1.0,
                x0: float = 0.0) -> np.ndarray:
    """`(reps, n)` independent Gaussian random walks.

    Returned two-dimensional even for `reps=1` so every downstream routine has
    one shape to handle. A drift is allowed because the *drifting* case is the
    one that matches real macro series, and it makes the spurious correlation
    dramatically worse — two series that both trend upward will correlate near
    one whatever else is true of them.
    """
    steps = rng.normal(drift, sigma, size=(reps, n))
    out = np.cumsum(steps, axis=1) + x0
    return out


# ---------------------------------------------------------------------------
# batched OLS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OLSFit:
    """Coefficients, standard errors and t-statistics for a batch of fits."""
    beta: np.ndarray        # (reps, k)
    se: np.ndarray          # (reps, k)
    t: np.ndarray           # (reps, k)
    resid: np.ndarray       # (reps, n)
    r2: np.ndarray          # (reps,)

    def __len__(self) -> int:
        return self.beta.shape[0]


def ols(X: np.ndarray, y: np.ndarray) -> OLSFit:
    """Least squares for a *stack* of regressions.

    `X` is `(reps, n, k)` and `y` is `(reps, n)`. Solved through the normal
    equations rather than `lstsq` because `lstsq` has no batched form and the
    designs here are 2-4 columns wide — the conditioning argument against normal
    equations does not bite at that size, and being able to run 20,000 of them at
    once is what makes a simulated critical value affordable.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 3 or y.ndim != 2:
        raise ValueError(f"expected X (reps, n, k) and y (reps, n), "
                         f"got {X.shape} and {y.shape}")
    reps, n, k = X.shape
    if y.shape != (reps, n):
        raise ValueError(f"y {y.shape} does not match X {X.shape}")
    if n <= k:
        raise ValueError(f"{n} observations cannot support {k} regressors")

    XtX = np.einsum("rnk,rnl->rkl", X, X)
    Xty = np.einsum("rnk,rn->rk", X, y)
    # NumPy 2 reads a 2-D right-hand side as a matrix, not as a stack of
    # vectors, so the batch axis has to be made explicit and taken back out.
    beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]
    fitted = np.einsum("rnk,rk->rn", X, beta)
    resid = y - fitted

    dof = n - k
    s2 = (resid ** 2).sum(axis=1) / dof
    XtX_inv = np.linalg.inv(XtX)
    var_beta = s2[:, None] * np.einsum("rkk->rk", XtX_inv)
    se = np.sqrt(var_beta)
    t = beta / se

    centred = y - y.mean(axis=1, keepdims=True)
    tss = (centred ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - (resid ** 2).sum(axis=1) / tss
    return OLSFit(beta=beta, se=se, t=t, resid=resid, r2=r2)


def _deterministic(n: int, reps: int, trend: str) -> np.ndarray:
    """Columns for the deterministic part of a test regression."""
    if trend == "n":
        return np.empty((reps, n, 0))
    cols = [np.ones((reps, n, 1))]
    if trend == "ct":
        tt = np.arange(n, dtype=float)
        cols.append(np.broadcast_to(tt, (reps, n))[..., None].copy())
    elif trend != "c":
        raise ValueError(f"trend must be 'n', 'c' or 'ct', got {trend!r}")
    return np.concatenate(cols, axis=2)


# ---------------------------------------------------------------------------
# Dickey-Fuller
# ---------------------------------------------------------------------------

def adf_stat(y: np.ndarray, *, lags: int = 0, trend: str = "c") -> np.ndarray:
    """Augmented Dickey-Fuller t-statistic on rho, for a batch of series.

        dy_t = a + rho * y_{t-1} + sum_i g_i dy_{t-i} + e_t

    The statistic is the ordinary t-ratio on `rho`. Its null distribution is not
    Student's t and not normal — hence everything else in this module. Returns
    one statistic per row of `y`.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    reps, n = y.shape
    if lags < 0:
        raise ValueError("lags must be non-negative")
    if n < lags + 3:
        raise ValueError(f"{n} observations is too few for {lags} lags")

    dy = np.diff(y, axis=1)                    # (reps, n-1)
    # row t of the regression uses dy[t], y[t-1] and dy[t-1 ... t-lags]
    start = lags
    target = dy[:, start:]                     # (reps, m)
    m = target.shape[1]
    level = y[:, start:-1]                     # y_{t-1} aligned with target

    cols = [level[..., None]]
    for i in range(1, lags + 1):
        cols.append(dy[:, start - i:start - i + m][..., None])
    det = _deterministic(m, reps, trend)
    X = np.concatenate([det] + cols, axis=2) if det.shape[2] else \
        np.concatenate(cols, axis=2)

    fit = ols(X, target)
    rho_col = det.shape[2]                     # rho sits right after the trend
    return fit.t[:, rho_col]


def df_critical_values(n: int, *, reps: int = 20000, trend: str = "c",
                       lags: int = 0, levels=(0.01, 0.05, 0.10),
                       rng: np.random.Generator | None = None) -> dict:
    """Simulate the Dickey-Fuller null and read its lower quantiles.

    The null is a driftless random walk, so the statistic's distribution is
    free of nuisance parameters and this is a legitimate way to obtain the
    table rather than an approximation to it.
    """
    rng = rng or np.random.default_rng(20260825)
    y = random_walk(n, reps, rng=rng)
    stats = adf_stat(y, lags=lags, trend=trend)
    return {lv: float(np.quantile(stats, lv)) for lv in levels}


# ---------------------------------------------------------------------------
# Engle-Granger
# ---------------------------------------------------------------------------

def engle_granger_stat(y: np.ndarray, x: np.ndarray, *, lags: int = 0,
                       trend: str = "c") -> np.ndarray:
    """Two-step Engle-Granger statistic: ADF on the cointegrating residual.

    Step one regresses `y` on `x` with the deterministic terms named by `trend`.
    Step two runs a Dickey-Fuller regression on the residual with **no** constant,
    because the residual of a regression that already contained one has mean zero
    by construction; including a second one would be collinear with nothing to
    estimate.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if y.shape != x.shape:
        raise ValueError(f"y {y.shape} and x {x.shape} must match")
    reps, n = y.shape

    det = _deterministic(n, reps, trend)
    X = np.concatenate([det, x[..., None]], axis=2) if det.shape[2] else \
        x[..., None]
    resid = ols(X, y).resid
    return adf_stat(resid, lags=lags, trend="n")


def eg_critical_values(n: int, *, reps: int = 20000, trend: str = "c",
                       lags: int = 0, levels=(0.01, 0.05, 0.10),
                       rng: np.random.Generator | None = None) -> dict:
    """Simulate the Engle-Granger null: two *independent* random walks."""
    rng = rng or np.random.default_rng(20260826)
    y = random_walk(n, reps, rng=rng)
    x = random_walk(n, reps, rng=rng)
    stats = engle_granger_stat(y, x, lags=lags, trend=trend)
    return {lv: float(np.quantile(stats, lv)) for lv in levels}


def misuse_size(n: int, *, reps: int = 20000, alpha: float = 0.05,
                rng: np.random.Generator | None = None) -> dict:
    """What using the *wrong* table costs.

    Returns the true size of an Engle-Granger test carried out against the
    Dickey-Fuller critical value, alongside the correctly-sized version. The gap
    is the answer to "does it really matter which table I use".
    """
    rng = rng or np.random.default_rng(20260827)
    y = random_walk(n, reps, rng=rng)
    x = random_walk(n, reps, rng=rng)
    stats = engle_granger_stat(y, x, trend="c")

    wrong = MACKINNON_DF[alpha]
    right = MACKINNON_EG[alpha]
    return {
        "n": n,
        "nominal": alpha,
        "df_critical_used": wrong,
        "eg_critical_correct": right,
        "size_using_df_table": float((stats < wrong).mean()),
        "size_using_eg_table": float((stats < right).mean()),
    }


# ---------------------------------------------------------------------------
# the spurious regression itself
# ---------------------------------------------------------------------------

def _chunks(reps: int, budget: int, n: int):
    """Split `reps` so no single batch allocates more than `budget` floats.

    The design matrix is the binding constraint: it is `reps * n * 2` doubles,
    and at reps=40000, n=1600 that is a gigabyte before any intermediate. Chunking
    keeps the batched-OLS speed while capping peak memory, which matters because
    the alternative — dropping reps — costs precision in exactly the tail
    quantiles the post quotes.
    """
    per = max(1, budget // max(n * 8, 1))
    done = 0
    while done < reps:
        take = min(per, reps - done)
        yield take
        done += take


def spurious_draws(n: int, *, reps: int = 20000, drift: float = 0.0,
                   rng: np.random.Generator | None = None,
                   budget: int = 6_000_000) -> dict:
    """One pass over independent walk pairs, returning every statistic at once.

    `t`, `r2` and `r` all come from the same simulated pairs, so the rejection
    rate, the scaled-t behaviour and the correlation distribution in this post
    describe *the same experiment* rather than three coincidentally similar ones.
    """
    rng = rng or np.random.default_rng(20260828)
    ts, r2s, rs = [], [], []
    for take in _chunks(reps, budget, n):
        y = random_walk(n, take, rng=rng, drift=drift)
        x = random_walk(n, take, rng=rng, drift=drift)
        X = np.concatenate([np.ones((take, n, 1)), x[..., None]], axis=2)
        fit = ols(X, y)
        ts.append(fit.t[:, 1].copy())
        r2s.append(fit.r2.copy())
        yc = y - y.mean(axis=1, keepdims=True)
        xc = x - x.mean(axis=1, keepdims=True)
        rs.append(((yc * xc).sum(axis=1)
                   / np.sqrt((yc ** 2).sum(axis=1) * (xc ** 2).sum(axis=1))))
        del y, x, X, fit, yc, xc
    return {"t": np.concatenate(ts), "r2": np.concatenate(r2s),
            "r": np.concatenate(rs), "n": n}


def spurious_rejection_rate(n: int, *, reps: int = 20000, alpha: float = 0.05,
                            drift: float = 0.0,
                            rng: np.random.Generator | None = None,
                            draws: dict | None = None) -> dict:
    """Rate at which independent random walks produce a 'significant' slope.

    Both series are generated independently, so the true slope is zero and a
    correctly-sized test would reject `alpha` of the time. It does not.
    """
    d = draws or spurious_draws(n, reps=reps, drift=drift, rng=rng)
    t, r2 = d["t"], d["r2"]
    z = _norm_ppf(1.0 - alpha / 2.0)   # the critical value a practitioner uses
    return {
        "n": d["n"],
        "nominal": alpha,
        "critical": z,
        "rejection_rate": float((np.abs(t) > z).mean()),
        "median_abs_t": float(np.median(np.abs(t))),
        "median_r2": float(np.median(r2)),
        "p90_r2": float(np.quantile(r2, 0.90)),
    }


def scaled_t_quantiles(n: int, *, reps: int = 20000, drift: float = 0.0,
                       levels=(0.05, 0.5, 0.95),
                       rng: np.random.Generator | None = None,
                       draws: dict | None = None) -> dict:
    """Quantiles of `|t| / sqrt(n)`.

    Phillips (1986) says this is the quantity that converges. If it does, these
    numbers stop moving as `n` grows while the raw `|t|` keeps climbing — which
    is the cleanest way to show that the problem is not noise but rate.
    """
    d = draws or spurious_draws(n, reps=reps, drift=drift, rng=rng)
    scaled = np.abs(d["t"]) / np.sqrt(d["n"])
    return {lv: float(np.quantile(scaled, lv)) for lv in levels}


def correlation_of_independent_walks(
        n: int, *, reps: int = 20000, drift: float = 0.0,
        rng: np.random.Generator | None = None,
        draws: dict | None = None) -> dict:
    """Distribution of Pearson r between two independent random walks.

    The headline number for a lay reader: how often does 'no shared cause'
    still look like a strong correlation.
    """
    d = draws or spurious_draws(n, reps=reps, drift=drift, rng=rng)
    a = np.abs(d["r"])
    return {
        "n": d["n"],
        "median_abs_r": float(np.median(a)),
        "p_abs_r_over_0.5": float((a > 0.5).mean()),
        "p_abs_r_over_0.8": float((a > 0.8).mean()),
        "p_abs_r_over_0.9": float((a > 0.9).mean()),
        "p_abs_r_over_0.96": float((a > 0.96).mean()),
    }


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return float((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                     / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return float(-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                     / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    q = p - 0.5
    r = q * q
    return float((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
                 / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))
