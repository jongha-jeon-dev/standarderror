"""Can this series answer the question being asked of it?

A claim of the form "X is displacing Y" is a claim about a *level shift* in a
series. Whether the series can settle that claim is a power question, and on a
persistent series the answer is usually far more pessimistic than the raw
observation count suggests.

Two things make it pessimistic.

**Autocorrelation destroys effective sample size.** The variance of a sample mean
is not `sigma^2 / n` when observations are correlated; it is
`sigma^2 / n * sum_k rho_k` over the autocorrelation function. Persistent series
therefore carry far fewer independent observations than they have rows. Twenty
years of monthly data on a series with an eighteen-month memory is not 240
observations of anything.

**A cycle is not noise, but it acts like noise against a step.** A series that
swings on a multi-year cycle will show apparent level shifts at almost any date
you care to name. `placebo_scan` measures how often, which is the honest way to
report what a break test on such a series is worth.

The remedy implemented here is not a better test — it is a correct standard
error (Newey-West) plus a power calculation that resamples the *actual* residual
process by moving-block bootstrap rather than assuming it is AR(1). Assuming
AR(1) on a series with a 40-month cycle understates the persistence badly, and
the whole point of the exercise is not to understate it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "design_matrix", "OLSHac", "ols_hac", "newey_west_lags",
    "autocorrelation", "effective_sample_size", "variance_inflation",
    "moving_block_bootstrap", "break_test", "placebo_scan",
    "calibrated_critical_value",
    "detection_power", "minimum_detectable_shift",
]


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------

def design_matrix(n: int, *, break_at: int | None = None,
                  trend: bool | int = True, seasonal: int = 0,
                  start_period: int = 0) -> np.ndarray:
    """Columns for an interrupted-time-series regression.

    Constant, optional linear trend, optional seasonal dummies, and a step
    dummy that turns on at `break_at` and stays on. The step is always the
    **last** column so callers can index the coefficient of interest without
    counting.

    Seasonal dummies use `seasonal - 1` columns against a reference period, and
    `start_period` says which phase the first row sits in — a monthly series
    beginning in August needs `start_period=7`, not 0, or every dummy is
    rotated and the seasonal adjustment silently removes the wrong months.
    """
    if n < 2:
        raise ValueError("need at least two observations")
    # `trend` may be a degree. A linear trend in a logged series asserts a
    # constant growth rate, which twenty-six years of semiconductor volume does
    # not have — and the curvature it leaves behind is then counted as
    # persistence, which is the wrong diagnosis. Time is scaled to [0, 1] so the
    # higher powers stay conditioned.
    degree = 1 if trend is True else (0 if trend is False else int(trend))
    if degree < 0:
        raise ValueError("trend degree must be non-negative")
    cols = [np.ones(n)]
    if degree:
        tt = np.arange(n, dtype=float) / max(n - 1, 1)
        cols.extend(tt ** k for k in range(1, degree + 1))
    if seasonal:
        if seasonal < 2:
            raise ValueError("seasonal period must be at least 2")
        phase = (np.arange(n) + start_period) % seasonal
        for k in range(1, seasonal):
            cols.append((phase == k).astype(float))
    if break_at is not None:
        if not 0 < break_at < n:
            raise ValueError(
                f"break_at={break_at} leaves one side of the series empty "
                f"(n={n}); a step at the first or last observation is not "
                f"identified")
        step = np.zeros(n)
        step[break_at:] = 1.0
        cols.append(step)
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# estimation with an honest standard error
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OLSHac:
    beta: np.ndarray
    se_ols: np.ndarray
    se_hac: np.ndarray
    resid: np.ndarray
    lags: int

    def t_ols(self, j: int) -> float:
        return float(self.beta[j] / self.se_ols[j])

    def t_hac(self, j: int) -> float:
        return float(self.beta[j] / self.se_hac[j])

    @property
    def inflation(self) -> np.ndarray:
        """How much the naive standard error understates, per coefficient."""
        return self.se_hac / self.se_ols


def newey_west_lags(n: int) -> int:
    """Automatic bandwidth, `floor(4 (n/100)^(2/9))`.

    This is the rule statsmodels and most textbooks use. It is deliberately
    conservative on short samples and it is *not* enough for a series whose
    memory runs to a multi-year cycle — a caller who knows the cycle length
    should pass a bandwidth covering it, and the effect of doing so is reported
    rather than buried.
    """
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def ols_hac(X: np.ndarray, y: np.ndarray, *, lags: int | None = None) -> OLSHac:
    """OLS with both the naive and the Newey-West standard error.

    Both are returned on purpose. The gap between them is the finding in a post
    about persistent data, so hiding the naive one would remove the evidence.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    if y.shape != (n,):
        raise ValueError(f"y {y.shape} does not match X {X.shape}")
    if n <= k:
        raise ValueError(f"{n} observations cannot support {k} regressors")
    lags = newey_west_lags(n) if lags is None else lags
    if lags < 0:
        raise ValueError("lags must be non-negative")

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = n - k
    s2 = resid @ resid / dof
    se_ols = np.sqrt(np.diag(s2 * XtX_inv))

    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        G = u[L:].T @ u[:-L]
        w = 1.0 - L / (lags + 1.0)          # Bartlett kernel
        S = S + w * (G + G.T)
    S *= n / dof                            # small-sample correction
    cov = XtX_inv @ S @ XtX_inv
    se_hac = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return OLSHac(beta=beta, se_ols=se_ols, se_hac=se_hac, resid=resid,
                  lags=lags)


# ---------------------------------------------------------------------------
# how much information is actually in the series
# ---------------------------------------------------------------------------

def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation, lags 0..max_lag, normalised by the lag-0 term."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if max_lag >= n:
        raise ValueError(f"max_lag {max_lag} needs fewer than {n} observations")
    c = x - x.mean()
    denom = c @ c
    if denom == 0:
        raise ValueError("series is constant; autocorrelation is undefined")
    return np.array([(c[L:] @ c[:n - L]) / denom if L else 1.0
                     for L in range(max_lag + 1)])


def variance_inflation(x: np.ndarray, max_lag: int) -> float:
    """`1 + 2 * sum_{k=1}^{L} (1 - k/(L+1)) rho_k` — the Bartlett-weighted
    factor by which autocorrelation multiplies the variance of a sample mean.

    The Bartlett weights are what keep this positive; the unweighted sum can go
    negative on a sample and then reports a *smaller* variance than independence
    would, which is nonsense.
    """
    rho = autocorrelation(x, max_lag)
    w = 1.0 - np.arange(1, max_lag + 1) / (max_lag + 1.0)
    return float(1.0 + 2.0 * np.sum(w * rho[1:]))


def effective_sample_size(x: np.ndarray, max_lag: int) -> float:
    """`n / variance_inflation` — how many independent observations the series
    is worth for estimating a level."""
    n = np.asarray(x).size
    vif = variance_inflation(x, max_lag)
    if vif <= 0:
        raise ValueError(
            "the estimated variance inflation is non-positive, which usually "
            "means max_lag is too large for this sample")
    return n / vif


# ---------------------------------------------------------------------------
# resampling that keeps the persistence
# ---------------------------------------------------------------------------

def moving_block_bootstrap(resid: np.ndarray, *, block: int, size: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Resample residuals in overlapping blocks, preserving local dependence.

    A block bootstrap is used instead of an AR(p) parametric bootstrap because
    the residual of a trend-plus-seasonal fit on this kind of series contains a
    multi-year cycle. An AR(1) fit to it reports a modest coefficient and then
    generates paths far tamer than the real thing, which would make the power
    calculation optimistic in exactly the direction that flatters the claim
    being tested.
    """
    resid = np.asarray(resid, dtype=float)
    n = resid.size
    if not 1 <= block <= n:
        raise ValueError(f"block {block} outside 1..{n}")
    n_blocks = int(np.ceil(size / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    out = np.concatenate([resid[s:s + block] for s in starts])
    return out[:size]


# ---------------------------------------------------------------------------
# the break test, and what it does when nothing happened
# ---------------------------------------------------------------------------

def break_test(y: np.ndarray, break_at: int, *, trend: bool = True,
               seasonal: int = 0, start_period: int = 0,
               lags: int | None = None) -> dict:
    """Fit a step at `break_at` and report it both ways."""
    y = np.asarray(y, dtype=float)
    X = design_matrix(y.size, break_at=break_at, trend=trend,
                      seasonal=seasonal, start_period=start_period)
    fit = ols_hac(X, y, lags=lags)
    j = X.shape[1] - 1
    return {
        "break_at": break_at,
        "shift": float(fit.beta[j]),
        "se_ols": float(fit.se_ols[j]),
        "se_hac": float(fit.se_hac[j]),
        "t_ols": fit.t_ols(j),
        "t_hac": fit.t_hac(j),
        "inflation": float(fit.se_hac[j] / fit.se_ols[j]),
        "lags": fit.lags,
    }


def placebo_scan(y: np.ndarray, *, trim: float = 0.15, trend: bool = True,
                 seasonal: int = 0, start_period: int = 0,
                 lags: int | None = None) -> dict:
    """Run the break test at every admissible date.

    If the series were well described by trend plus seasonal plus independent
    noise, about 5% of these would clear |t| = 1.96 by chance. What actually
    happens on a cyclical series is the point of the exercise: `share_ols`
    counts how often the naive test fires, `share_hac` how often the corrected
    one does.

    `trim` drops the first and last fraction of candidate dates, because a step
    fitted a handful of observations from the end is estimated off almost no
    post-break data and its standard error is not comparable to the others.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    lo, hi = int(np.floor(trim * n)), int(np.ceil((1 - trim) * n))
    if hi - lo < 2:
        raise ValueError("trim leaves no candidate break dates")
    rows = [break_test(y, tau, trend=trend, seasonal=seasonal,
                       start_period=start_period, lags=lags)
            for tau in range(lo, hi)]
    t_ols = np.array([r["t_ols"] for r in rows])
    t_hac = np.array([r["t_hac"] for r in rows])
    return {
        "rows": rows,
        "n_dates": len(rows),
        "share_ols": float((np.abs(t_ols) > 1.96).mean()),
        "share_hac": float((np.abs(t_hac) > 1.96).mean()),
        "max_abs_t_ols": float(np.abs(t_ols).max()),
        "max_abs_t_hac": float(np.abs(t_hac).max()),
        "argmax_hac": int(rows[int(np.argmax(np.abs(t_hac)))]["break_at"]),
        "median_inflation": float(np.median([r["inflation"] for r in rows])),
    }


# ---------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------

def calibrated_critical_value(resid: np.ndarray, *, n_pre: int, n_post: int,
                              block: int, level: float = 0.05,
                              reps: int = 4000, statistic: str = "hac",
                              seasonal: int = 0, start_period: int = 0,
                              trend: bool = True, lags: int | None = None,
                              rng: np.random.Generator | None = None) -> dict:
    """The critical value this design actually needs, built from its own null.

    Why this exists rather than 1.96
    --------------------------------
    A Newey-West correction is a large improvement on the naive standard error
    and it is still not correctly sized, and there is no bandwidth that fixes
    it: a short bandwidth under-corrects on persistent data, a long one
    over-corrects on quiet data, and the two failures do not have a crossing
    point where both are right. Simulating the null of *this* residual process
    sidesteps the choice — whatever the bandwidth does, the reference
    distribution is generated under the same bandwidth, so the size is right by
    construction.

    Returns the two-sided critical value together with the size that 1.96 would
    have delivered, because that gap is the thing worth reporting.
    """
    if statistic not in ("hac", "ols"):
        raise ValueError("statistic must be 'hac' or 'ols'")
    rng = rng or np.random.default_rng(20260826)
    n = n_pre + n_post
    key = f"t_{statistic}"
    stats = np.empty(reps)
    for i in range(reps):
        u = moving_block_bootstrap(resid, block=block, size=n, rng=rng)
        stats[i] = break_test(u, n_pre, trend=trend, seasonal=seasonal,
                              start_period=start_period, lags=lags)[key]
    a = np.abs(stats)
    return {
        "critical": float(np.quantile(a, 1.0 - level)),
        "size_of_1p96": float((a > 1.959963984540054).mean()),
        "level": level, "reps": reps, "statistic": statistic,
        "n_pre": n_pre, "n_post": n_post, "block": block,
    }


def detection_power(resid: np.ndarray, *, n_pre: int, n_post: int,
                    shift: float, block: int, reps: int = 2000,
                    seasonal: int = 0, start_period: int = 0,
                    trend: bool = True, lags: int | None = None,
                    alpha: float = 0.05, critical: float | None = None,
                    rng: np.random.Generator | None = None) -> dict:
    """Probability of detecting a step of size `shift`, by simulation.

    The null series is a block bootstrap of the supplied residuals, so it has
    the real series' persistence and none of its trend. The step is added to the
    last `n_post` observations and the same test the analyst would run is run.
    Reported for both standard errors, because the naive one is what gets used.
    """
    rng = rng or np.random.default_rng(20260825)
    n = n_pre + n_post
    # `critical` is the calibrated value when supplied; falling back to the
    # normal quantile reproduces what an analyst using 1.96 would actually get,
    # which is why both paths are kept rather than one being removed.
    z = critical if critical is not None else (
        1.959963984540054 if alpha == 0.05 else _z(alpha))
    hit_ols = hit_hac = 0
    for _ in range(reps):
        u = moving_block_bootstrap(resid, block=block, size=n, rng=rng)
        y = u.copy()
        y[n_pre:] += shift
        r = break_test(y, n_pre, trend=trend, seasonal=seasonal,
                       start_period=start_period, lags=lags)
        hit_ols += abs(r["t_ols"]) > z
        hit_hac += abs(r["t_hac"]) > z
    return {"n_pre": n_pre, "n_post": n_post, "shift": shift,
            "power_ols": hit_ols / reps, "power_hac": hit_hac / reps,
            "reps": reps}


def _z(alpha: float) -> float:
    from .nonstationary import _norm_ppf
    return _norm_ppf(1.0 - alpha / 2.0)


def minimum_detectable_shift(resid: np.ndarray, *, n_pre: int, n_post: int,
                             block: int, target: float = 0.80,
                             reps: int = 1200, lo: float = 0.0,
                             hi: float = 2.0, tol: float = 0.005,
                             use: str = "hac", critical: float | None = None,
                             **kw) -> dict:
    """Smallest step this design can detect at `target` power, by bisection.

    Returned in the units of `resid` — log points if the series was logged, so
    a result of 0.12 means "a 12% permanent drop in level, and nothing smaller".
    """
    if use not in ("hac", "ols"):
        raise ValueError("use must be 'hac' or 'ols'")
    key = f"power_{use}"
    seed = kw.pop("rng", None)
    lo_p = detection_power(resid, n_pre=n_pre, n_post=n_post, shift=lo,
                           block=block, reps=reps, critical=critical,
                           rng=np.random.default_rng(1), **kw)[key]
    hi_p = detection_power(resid, n_pre=n_pre, n_post=n_post, shift=hi,
                           block=block, reps=reps, critical=critical,
                           rng=np.random.default_rng(2), **kw)[key]
    if hi_p < target:
        return {"mde": float("inf"), "power_at_hi": hi_p, "hi": hi,
                "note": "even the largest shift searched is not detectable"}
    it = 0
    while hi - lo > tol and it < 40:
        mid = 0.5 * (lo + hi)
        p = detection_power(resid, n_pre=n_pre, n_post=n_post, shift=mid,
                            block=block, reps=reps, critical=critical,
                            rng=np.random.default_rng(100 + it), **kw)[key]
        if p < target:
            lo = mid
        else:
            hi = mid
        it += 1
    return {"mde": 0.5 * (lo + hi), "target": target, "iterations": it,
            "power_at_zero": lo_p, "use": use}
