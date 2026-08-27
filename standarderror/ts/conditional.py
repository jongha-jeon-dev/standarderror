"""Correlation measured on a subsample chosen by volatility, and what that costs.

"Correlations go to one in a crisis" is one of the few quantitative claims that
every desk, every risk committee and every diversification pitch agrees on. The
evidence offered for it is almost always the same calculation: split the sample
into turbulent and calm periods — by the largest absolute moves, by a volatility
index, or by naming the crisis dates — and report the correlation in each half.

That calculation has a property nobody mentions when quoting it. Take two jointly
normal series with a **constant** correlation, so that by construction nothing
whatsoever breaks down, and standardise them over the full sample::

    y = rho * x + sqrt(1 - rho**2) * eps,      eps independent of x

Now condition on any event ``A`` defined by ``x`` alone. The slope ``rho`` and the
residual variance ``1 - rho**2`` are unchanged by that conditioning, because
``eps`` is independent of ``x``. Only the variance of ``x`` changes. So

    Cov(x, y | A) = rho * s**2
    Var(y | A)    = rho**2 * s**2 + (1 - rho**2)          with s**2 = Var(x | A)

and therefore the correlation you will measure inside ``A`` is

    rho_A = rho * s / sqrt(rho**2 * s**2 + 1 - rho**2)

which is `conditional_rho` below. It has no free parameters: given the full-sample
correlation and how much more variable ``x`` is inside the subsample, the
turbulent-period correlation is *already determined*. A turbulent decile has
roughly six times the variance of the rest of the sample, and at rho = 0.3 that
identity alone takes the measured correlation to about 0.55.

Inverting the identity gives the correction from Forbes and Rigobon (2002),
"No Contagion, Only Interdependence", Journal of Finance 57(5) — `unconditional_rho`
here — which they used to argue that the contagion literature of the 1990s had
measured heteroskedasticity. This module is not new statistics. What it is for is
the part practitioners still get wrong twenty-four years later:

* The identity is exact only when the conditioning event is a function of ``x``
  alone. Conditioning on a **volatility index** or on **named crisis dates**
  selects high-variance periods in *both* series, and then the correction is no
  longer exact — `null_prediction` accepts the conditioner explicitly so the
  assumption is visible rather than implied.
* The null to beat is not "no rise". It is the rise the identity delivers, and
  reporting a rise without it is reporting an arithmetic identity as a finding.
* Fat tails do not rescue the correction, they damage it further. A
  constant-correlation Student copula (`student_pair`) has a wider variance ratio
  in the turbulent decile, so the identity over-predicts by more and the
  correction is biased low by more. "The data are not normal" is an argument
  against trusting the standard fix, not for it.
* Even where the excess over the null is positive, a turbulent decile is a short
  sample, and `bootstrap_split` will usually show it is not distinguishable from
  the null at the sample sizes actually available.

Everything here works on paired arrays and never on prices, so a caller with
non-redistributable input can publish the statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The quantile of |x| above which a period is called turbulent. 0.90 is the
#: convention in the contagion literature; nothing here depends on the choice
#: and `quantile_sweep` exists to show that.
TURBULENT_QUANTILE = 0.90

#: EWMA decay for `devolatilise`. RiskMetrics' daily value, kept because it is
#: the one a reader will recognise, not because it is optimal.
EWMA_LAMBDA = 0.94


# --------------------------------------------------------------------------- #
# the identity
# --------------------------------------------------------------------------- #
def conditional_rho(rho: float, var_ratio: float) -> float:
    """Correlation measured inside a subsample selected on ``x`` alone.

    ``var_ratio`` is Var(x | subsample) / Var(x | full sample). Under a constant
    correlation and joint normality this is the *whole* story: no contagion, no
    regime change, no time-varying dependence, and yet the number moves.

    var_ratio = 1 returns rho unchanged; var_ratio -> infinity drives the result
    to +-1, which is the sense in which "correlations go to one in a crisis" is
    true of every constant-correlation process ever written down.
    """
    if var_ratio <= 0:
        raise ValueError(f"var_ratio must be positive, got {var_ratio}")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")
    s = np.sqrt(var_ratio)
    denom = np.sqrt(rho ** 2 * var_ratio + 1.0 - rho ** 2)
    return float(rho * s / denom)


def unconditional_rho(rho_cond: float, var_ratio: float) -> float:
    """Invert `conditional_rho`: the Forbes-Rigobon correction.

    Given a correlation measured inside a volatility-selected subsample and how
    much more variable the conditioning series was there, return the constant
    correlation that would have produced it. This is the number the turbulent
    subsample is actually evidence about.
    """
    if var_ratio <= 0:
        raise ValueError(f"var_ratio must be positive, got {var_ratio}")
    if not -1.0 <= rho_cond <= 1.0:
        raise ValueError(f"rho_cond must be in [-1, 1], got {rho_cond}")
    denom = np.sqrt(var_ratio + rho_cond ** 2 * (1.0 - var_ratio))
    if denom <= 0:
        return float(np.sign(rho_cond))
    out = rho_cond / denom
    return float(np.clip(out, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# sample statistics
# --------------------------------------------------------------------------- #
def pearson(x, y) -> float:
    """Pearson correlation, demeaned within whatever sample it is handed.

    Spelled out rather than called inline because the demeaning convention is
    exactly what makes a conditional correlation different from an
    unconditional one, and it should be readable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.size < 3:
        return float("nan")
    a = x - x.mean()
    b = y - y.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def variance_ratio(x, mask) -> float:
    """Var(x | mask) / Var(x), both demeaned within their own sample."""
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 3:
        return float("nan")
    full = x.var(ddof=1)
    if full == 0:
        return float("nan")
    return float(x[mask].var(ddof=1) / full)


def turbulent_mask(x, q: float = TURBULENT_QUANTILE) -> np.ndarray:
    """The |x| >= quantile(q) mask: the split the contagion literature uses."""
    x = np.asarray(x, dtype=float)
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    return np.abs(x) >= np.quantile(np.abs(x), q)


def date_mask(index, windows) -> np.ndarray:
    """Mask for named crisis windows, as ``[(start, end), ...]`` inclusive.

    Kept separate from `turbulent_mask` because it is a *different* conditioning
    and the identity does not apply to it unchanged.
    """
    import pandas as pd

    idx = pd.DatetimeIndex(index)
    out = np.zeros(len(idx), dtype=bool)
    for start, end in windows:
        out |= (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return out


# --------------------------------------------------------------------------- #
# one split, fully described
# --------------------------------------------------------------------------- #
@dataclass
class SplitResult:
    """Everything needed to judge one turbulent/calm comparison.

    `excess` is the finding, if there is one: the measured turbulent correlation
    minus the one the constant-correlation identity already predicts from the
    variance ratio. `rise` is what gets published instead.
    """

    label: str
    conditioner: str
    n_turbulent: int
    n_calm: int
    rho_full: float
    rho_turbulent: float
    rho_calm: float
    var_ratio_x: float
    var_ratio_y: float
    rho_predicted: float
    rho_corrected: float
    exact: bool

    @property
    def rise(self) -> float:
        """Turbulent minus calm, in absolute correlation. The published number."""
        return abs(self.rho_turbulent) - abs(self.rho_calm)

    @property
    def predicted_rise(self) -> float:
        return abs(self.rho_predicted) - abs(self.rho_calm)

    @property
    def excess(self) -> float:
        return abs(self.rho_turbulent) - abs(self.rho_predicted)

    @property
    def explained(self) -> float:
        """Share of the published rise the identity accounts for.

        Can exceed 1, and on real data usually does — that is the point. Returns
        nan when the published rise is negligible, because a ratio to nothing is
        not informative.
        """
        if not np.isfinite(self.rise) or abs(self.rise) < 1e-9:
            return float("nan")
        return float(self.predicted_rise / self.rise)

    def row(self) -> list[str]:
        return [
            self.label,
            self.conditioner,
            f"{self.rho_calm:+.3f}",
            f"{self.rho_turbulent:+.3f}",
            f"{self.var_ratio_x:.2f}x",
            f"{self.rho_predicted:+.3f}",
            f"{self.excess:+.3f}",
        ]


def split_stats(x, y, mask, *, label: str = "", conditioner: str = "|x|",
                exact: bool | None = None) -> SplitResult:
    """Describe one turbulent/calm split against the constant-correlation null.

    ``exact`` records whether the conditioning event is a function of ``x`` alone.
    It is inferred from ``conditioner`` when not given, and it is stored rather
    than used, because the honest thing is to publish the null prediction with a
    flag saying it is approximate — not to silently withhold it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if exact is None:
        exact = conditioner in ("|x|", "x")

    rho_full = pearson(x, y)
    vr_x = variance_ratio(x, mask)
    return SplitResult(
        label=label,
        conditioner=conditioner,
        n_turbulent=int(mask.sum()),
        n_calm=int((~mask).sum()),
        rho_full=rho_full,
        rho_turbulent=pearson(x[mask], y[mask]),
        rho_calm=pearson(x[~mask], y[~mask]),
        var_ratio_x=vr_x,
        var_ratio_y=variance_ratio(y, mask),
        rho_predicted=conditional_rho(rho_full, vr_x) if np.isfinite(vr_x) else float("nan"),
        rho_corrected=unconditional_rho(pearson(x[mask], y[mask]), vr_x)
        if np.isfinite(vr_x) else float("nan"),
        exact=bool(exact),
    )


def quantile_sweep(x, y, quantiles=(0.50, 0.75, 0.90, 0.95, 0.99)) -> list[SplitResult]:
    """The same split at every threshold, because the threshold is a choice."""
    out = []
    for q in quantiles:
        out.append(split_stats(x, y, turbulent_mask(x, q),
                               label=f"q={q:.2f}", conditioner="|x|"))
    return out


# --------------------------------------------------------------------------- #
# devolatilising: the null taken seriously
# --------------------------------------------------------------------------- #
def ewma_scale(x, lam: float = EWMA_LAMBDA, *, warmup: int = 250) -> np.ndarray:
    """One-sided EWMA standard deviation, shifted so it uses no future data.

    The shift matters. An EWMA that includes today's return in today's scale
    divides a large return by a scale that the large return itself inflated,
    which shrinks exactly the observations the turbulent subsample is made of.
    """
    x = np.asarray(x, dtype=float)
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    if x.size <= warmup:
        raise ValueError(f"need more than warmup={warmup} points, got {x.size}")
    var = np.empty_like(x)
    v = float(x[:warmup].var(ddof=1))
    for t in range(x.size):
        var[t] = v                      # scale for t uses data up to t-1 only
        v = lam * v + (1.0 - lam) * x[t] ** 2
    return np.sqrt(var)


def devolatilise(x, y, lam: float = EWMA_LAMBDA, *, warmup: int = 250):
    """Divide each series by its own one-sided EWMA scale.

    Under the constant-correlation-with-heteroskedasticity null this removes the
    whole mechanism: the standardised pair has no variance ratio left to inflate
    anything, so a rise that survives here is a rise in dependence rather than
    in scale. Returns the trimmed, standardised pair and the mask of kept points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = ewma_scale(x, lam, warmup=warmup)
    sy = ewma_scale(y, lam, warmup=warmup)
    keep = np.zeros(x.size, dtype=bool)
    keep[warmup:] = True
    keep &= (sx > 0) & (sy > 0)
    return x[keep] / sx[keep], y[keep] / sy[keep], keep


# --------------------------------------------------------------------------- #
# where the crisis loss actually comes from
# --------------------------------------------------------------------------- #
@dataclass
class Decomposition:
    """Covariance is rho * sx * sy, so its log ratio splits into three terms.

    Published as shares of the total log change, because that is the only
    decomposition of a product that does not require choosing an order.
    """

    cov_ratio: float
    rho_ratio: float
    sx_ratio: float
    sy_ratio: float
    share_rho: float
    share_sx: float
    share_sy: float
    vol_turbulent: float
    vol_calm: float
    vol_turbulent_calm_rho: float

    @property
    def share_scale(self) -> float:
        return self.share_sx + self.share_sy

    @property
    def portfolio_rise(self) -> float:
        return self.vol_turbulent / self.vol_calm

    @property
    def portfolio_rise_frozen_rho(self) -> float:
        """The rise that would have happened with the calm correlation held fixed."""
        return self.vol_turbulent_calm_rho / self.vol_calm

    @property
    def rho_contribution(self) -> float:
        """Share of the portfolio-volatility rise attributable to correlation."""
        total = self.portfolio_rise - 1.0
        if abs(total) < 1e-12:
            return float("nan")
        return float((self.portfolio_rise - self.portfolio_rise_frozen_rho) / total)


def covariance_decomposition(x, y, mask, *, weights=(0.5, 0.5)) -> Decomposition:
    """Split the turbulent/calm covariance change into correlation and scale.

    The counterfactual portfolio volatility holds the correlation at its calm
    value while letting both volatilities move, which is the comparison a
    diversification claim implicitly makes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    wx, wy = weights

    def moments(m):
        return (x[m].std(ddof=1), y[m].std(ddof=1), pearson(x[m], y[m]))

    sxt, syt, rt = moments(mask)
    sxc, syc, rc = moments(~mask)

    def port(sx, sy, r):
        return float(np.sqrt(max(wx ** 2 * sx ** 2 + wy ** 2 * sy ** 2
                                 + 2 * wx * wy * r * sx * sy, 0.0)))

    logs = {"rho": np.log(abs(rt) / abs(rc)), "sx": np.log(sxt / sxc),
            "sy": np.log(syt / syc)}
    total = sum(logs.values())
    shares = {k: (v / total if total != 0 else float("nan")) for k, v in logs.items()}

    return Decomposition(
        cov_ratio=float(np.exp(total)),
        rho_ratio=float(abs(rt) / abs(rc)),
        sx_ratio=float(sxt / sxc),
        sy_ratio=float(syt / syc),
        share_rho=float(shares["rho"]),
        share_sx=float(shares["sx"]),
        share_sy=float(shares["sy"]),
        vol_turbulent=port(sxt, syt, rt),
        vol_calm=port(sxc, syc, rc),
        vol_turbulent_calm_rho=port(sxt, syt, rc),
    )


# --------------------------------------------------------------------------- #
# how well any of this is known
# --------------------------------------------------------------------------- #
def moving_block_indices(n: int, block: int, rng) -> np.ndarray:
    """Moving-block bootstrap indices of length n.

    Blocks, not points, because the variance ratio the null prediction depends on
    is a property of the volatility path and an iid resample destroys it. This
    still distorts that path — exp015 found block resampling *manufactures*
    volatility clustering by splicing regimes — so read the interval as a rough
    sampling scale and not as an exact one.
    """
    if block < 1 or block > n:
        raise ValueError(f"block must be in [1, {n}], got {block}")
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


def bootstrap_split(x, y, *, q: float = TURBULENT_QUANTILE, block: int = 20,
                    n_boot: int = 500, seed: int = 0) -> dict:
    """Sampling interval for the excess over the null.

    The mask is recomputed inside each resample: the threshold is a statistic of
    the sample, and holding it fixed would understate the variability of every
    quantity that depends on it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    n = x.size
    keys = ("rho_turbulent", "rho_calm", "rho_predicted", "excess", "rise")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = moving_block_indices(n, block, rng)
        xs, ys = x[idx], y[idx]
        r = split_stats(xs, ys, turbulent_mask(xs, q))
        for k in keys:
            draws[k].append(getattr(r, k))
    out = {}
    for k, v in draws.items():
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        out[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                  "lo": float(np.quantile(a, 0.025)),
                  "hi": float(np.quantile(a, 0.975)),
                  "p_le_0": float((a <= 0).mean())}
    out["n_boot"] = int(n_boot)
    out["block"] = int(block)
    return out


# --------------------------------------------------------------------------- #
# generators for the null
# --------------------------------------------------------------------------- #
def gaussian_pair(n: int, rho: float, rng, *, scale=None):
    """Constant-correlation normal pair, optionally with a common scale path.

    ``scale`` multiplies both series by the same positive path, which is how a
    volatility regime enters without touching the dependence: the correlation of
    the pair is rho at every point in time regardless.
    """
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = rng.standard_normal((n, 2)) @ L.T
    if scale is not None:
        s = np.asarray(scale, dtype=float).reshape(-1, 1)
        if s.size != n:
            raise ValueError(f"scale must have length {n}, got {s.size}")
        z = z * s
    return z[:, 0], z[:, 1]


def student_pair(n: int, rho: float, df: float, rng):
    """Constant-correlation Student pair: the same dependence, fatter margins.

    A Student copula has tail dependence a Gaussian one does not, so this is the
    honest fat-tailed null. Its turbulent decile has a wider variance ratio than
    the Gaussian null's, which makes the identity over-predict by more and the
    correction biased low by more — so "returns are not normal" damages the
    standard fix rather than rescuing it.
    """
    if df <= 2:
        raise ValueError(f"df must exceed 2 for a finite variance, got {df}")
    x, y = gaussian_pair(n, rho, rng)
    w = np.sqrt(df / rng.chisquare(df, size=n))
    return x * w, y * w


def garch_scale(n: int, rng, *, omega: float = 0.05, alpha: float = 0.10,
                beta: float = 0.88, burn: int = 500):
    """A GARCH(1,1) scale path, to give the null realistic volatility clustering.

    Returned as a scale rather than as returns so the same path can be applied
    to both members of a pair, keeping the dependence exactly constant.
    """
    if alpha + beta >= 1:
        raise ValueError(f"alpha + beta must be < 1, got {alpha + beta}")
    total = n + burn
    v = omega / (1.0 - alpha - beta)
    out = np.empty(total)
    for t in range(total):
        out[t] = v
        e = rng.standard_normal() * np.sqrt(v)
        v = omega + alpha * e ** 2 + beta * v
    return np.sqrt(out[burn:])


# --------------------------------------------------------------------------- #
# the null that keeps the volatility path
# --------------------------------------------------------------------------- #
def centred_scale(x, window: int = 21) -> np.ndarray:
    """Two-sided rolling standard deviation. Deliberately uses future data.

    Not an estimator anyone could trade on — it exists as a control. A one-sided
    scale lags a volatility jump, so on the day the regime changes it divides a
    large return by a scale fitted before the change, leaving both series large
    and manufacturing exactly the co-movement this module is trying to measure.
    Re-running the null with a look-ahead scale removes that mechanism: if the
    result moves, the finding was estimator lag.
    """
    import pandas as pd

    x = np.asarray(x, dtype=float)
    if window < 3:
        raise ValueError(f"window must be at least 3, got {window}")
    s = pd.Series(x).rolling(window, center=True, min_periods=window // 2).std(ddof=1)
    return s.bfill().ffill().to_numpy()


@dataclass
class NullTest:
    """Observed turbulent correlation against a constant-correlation null.

    The null keeps each series' estimated volatility path and imposes one
    constant correlation, so it reproduces the variance ratio the identity feeds
    on without containing any change in dependence. `genuine_excess` is the part
    of the published rise that the null cannot produce; `share_genuine` is the
    fraction of the published rise that survives.
    """

    q: float
    scale_kind: str
    n: int
    reps: int
    rho_full: float
    rho_turbulent: float
    rho_calm: float
    var_ratio: float
    null_turbulent_mean: float
    null_turbulent_lo: float
    null_turbulent_hi: float
    null_calm_mean: float
    null_var_ratio: float
    p_value: float
    draws: list[float] = field(default_factory=list, repr=False)

    @property
    def rise(self) -> float:
        return abs(self.rho_turbulent) - abs(self.rho_calm)

    @property
    def null_rise(self) -> float:
        return abs(self.null_turbulent_mean) - abs(self.null_calm_mean)

    @property
    def genuine_excess(self) -> float:
        return abs(self.rho_turbulent) - abs(self.null_turbulent_mean)

    @property
    def share_genuine(self) -> float:
        if abs(self.rise) < 1e-9:
            return float("nan")
        return float(self.genuine_excess / self.rise)


def scale_null(x, y, *, q: float = TURBULENT_QUANTILE, reps: int = 300,
               seed: int = 0, scale: str = "ewma", lam: float = EWMA_LAMBDA,
               window: int = 21, warmup: int = 250) -> NullTest:
    """Test the turbulent-period correlation against a same-volatility null.

    The generated pair is ``x* = s_x z1`` and ``y* = s_y (rho z1 + sqrt(1-rho^2) z2)``
    with ``rho`` the full-sample correlation and ``s`` the estimated scale paths.
    By construction the pair has one constant conditional correlation at every
    date, and the same volatility clustering as the data, which is the null the
    Forbes-Rigobon identity assumes away: that identity is exact only when the
    conditioning event does not also select the residual's variance, and a common
    volatility path means it always does.

    The turbulent mask is recomputed inside each replicate for the same reason
    `bootstrap_split` recomputes it: the threshold is a sample statistic.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if scale == "ewma":
        sx = ewma_scale(x, lam, warmup=warmup)
        sy = ewma_scale(y, lam, warmup=warmup)
    elif scale == "centred":
        sx = centred_scale(x, window)
        sy = centred_scale(y, window)
    else:
        raise ValueError(f"scale must be 'ewma' or 'centred', got {scale!r}")

    rho = pearson(x, y)
    obs = split_stats(x, y, turbulent_mask(x, q))
    rng = np.random.default_rng(seed)
    n = x.size
    root = np.sqrt(max(1.0 - rho ** 2, 0.0))
    turb, calm, vr = [], [], []
    for _ in range(reps):
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        xs = sx * z1
        ys = sy * (rho * z1 + root * z2)
        r = split_stats(xs, ys, turbulent_mask(xs, q))
        turb.append(r.rho_turbulent)
        calm.append(r.rho_calm)
        vr.append(r.var_ratio_x)
    turb = np.asarray(turb, dtype=float)
    return NullTest(
        q=float(q), scale_kind=scale, n=int(n), reps=int(reps),
        rho_full=rho,
        rho_turbulent=obs.rho_turbulent, rho_calm=obs.rho_calm,
        var_ratio=obs.var_ratio_x,
        null_turbulent_mean=float(turb.mean()),
        null_turbulent_lo=float(np.quantile(turb, 0.025)),
        null_turbulent_hi=float(np.quantile(turb, 0.975)),
        null_calm_mean=float(np.mean(calm)),
        null_var_ratio=float(np.mean(vr)),
        p_value=float((np.abs(turb) >= abs(obs.rho_turbulent)).mean()),
        draws=turb.tolist(),
    )
