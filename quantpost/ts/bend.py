"""A break in a growing series is usually a bend, and a step fitted to a bend
lands in the wrong year.

Where this sits
---------------
The previous module, `quantpost.ts.detect`, asks whether a series can settle a
claim about a *level shift* — a step. Most claims about a real economy are not
about steps. "China's capacity build-out changed Korea's export trajectory" is a
claim about a **slope**: the series did not drop, it stopped climbing as fast.
Fitting a step to that is not a small misspecification. It is a different shape.

Three things follow, and all three are measured here rather than asserted.

**The estimated date is biased, and the bias does not go away with more data.**
A step fitted to a deterministic bend has a profile-SSR minimum at a date that
is a function of the bend's location and the sample, and that date is not the
bend's. `noise_free_step_date` computes it with no noise at all, so the number
is arithmetic rather than a simulation artefact. Adding noise scatters the
estimate around that wrong place, not around the truth.

**A bend and a step cost the same.** Each adds exactly one column to the same
base design, and both are searched over the same grid of dates. So their sums of
squared residuals are directly comparable with no penalty term, no information
criterion and no argument about degrees of freedom. That is a rare piece of luck
and `model_race` uses it — but a race has to be scored against how often the
bend wins on data where *nothing happened*, which is what `null_race` measures.

**Knowing that something bent is not knowing when.** The break date is the worst
estimated quantity in the regression: the profile SSR near its minimum is nearly
flat, so the date moves a great deal under resampling while the slope change
barely moves. `date_bootstrap` reports the interval, and `date_coverage` checks
that the interval covers the truth at the rate it claims — a check that can fail,
and is reported either way.

The fast scan
-------------
`scan` evaluates every candidate date by Frisch-Waugh: the base design is
projected out once, and each date then costs a handful of length-`n` dot
products instead of a fresh matrix inversion. The Newey-West standard error of
the single break coefficient is computed from the residualised regressor and the
full-model residuals, which is algebraically identical to the corresponding
diagonal element of the full sandwich. `slow_scan` does the same thing the
obvious way and the two are checked against each other in the tests; if the fast
path is ever wrong, that test is where it shows up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import design_matrix, moving_block_bootstrap, newey_west_lags, ols_hac

__all__ = [
    "break_column", "bend_design", "BreakScan", "scan", "slow_scan",
    "fit_at", "noise_free_step_date", "step_on_bend",
    "model_race", "null_race", "calibrated_sup", "date_bootstrap",
    "date_coverage", "slope_change_per_year", "size_from_slope",
    "bend_power", "minimum_detectable_bend", "periods_to_detect",
    "calibrated_fixed",
]

KINDS = ("step", "bend")


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------

def break_column(n: int, tau: int, kind: str) -> np.ndarray:
    """The one column that distinguishes the two hypotheses.

    `step` is the usual `1(t >= tau)`: a permanent shift in level, discontinuous
    at `tau`, flat on both sides.

    `bend` is `max(t - tau, 0)` rescaled by `n - 1`: continuous at `tau`, zero
    before it, and rising linearly after. Its coefficient is the change in the
    trend's slope, expressed over the whole sample so that it is on the same
    scale as the trend column `design_matrix` builds. `slope_change_per_year`
    converts it to something a reader can hold.

    The rescaling matters for more than conditioning. An unscaled bend column
    reaches `n - tau` while the trend column reaches 1, and the two coefficients
    then differ by three orders of magnitude, which makes every printed table
    unreadable and every comparison between them a mental arithmetic exercise.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    if not 0 < tau < n:
        raise ValueError(f"tau={tau} leaves one side empty (n={n})")
    t = np.arange(n, dtype=float)
    if kind == "step":
        return (t >= tau).astype(float)
    # A bend needs at least two post-break observations before a slope on that
    # side is identified at all; one observation determines a point, not a line.
    if n - tau < 2:
        raise ValueError(
            f"a bend at tau={tau} has {n - tau} post-break observation(s); a "
            f"slope needs at least two")
    return np.maximum(t - tau, 0.0) / max(n - 1, 1)


def bend_design(n: int, *, tau: int | None = None, kind: str = "bend",
                trend: bool | int = True, seasonal: int = 0,
                start_period: int = 0) -> np.ndarray:
    """Base design from `detect.design_matrix`, plus one break column last.

    The break column is last so callers index it as `-1`, matching the
    convention in `detect`. Passing `tau=None` returns the base design, which is
    what the null model is.
    """
    X = design_matrix(n, break_at=None, trend=trend, seasonal=seasonal,
                      start_period=start_period)
    if tau is None:
        return X
    return np.column_stack([X, break_column(n, tau, kind)])


def slope_change_per_year(coef: float, n: int, periods_per_year: int = 12
                          ) -> float:
    """A bend coefficient as a change in annual growth rate, in log points.

    The bend column rises by `1` over the full sample of `n` observations, so
    the coefficient is the total slope change accumulated over that span; the
    per-year figure divides by the number of years the sample covers.
    """
    years = (n - 1) / periods_per_year
    if years <= 0:
        raise ValueError("sample is shorter than one period")
    return coef / years


# ---------------------------------------------------------------------------
# scanning every candidate date
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BreakScan:
    kind: str
    dates: np.ndarray          # candidate tau values, in index units
    coef: np.ndarray           # break coefficient at each date
    t_ols: np.ndarray
    t_hac: np.ndarray
    ssr: np.ndarray
    ssr_null: float
    lags: int

    @property
    def best(self) -> int:
        """The date with the lowest residual sum of squares."""
        return int(self.dates[int(np.argmin(self.ssr))])

    @property
    def best_by_t(self) -> int:
        return int(self.dates[int(np.argmax(np.abs(self.t_hac)))])

    @property
    def sup_t_hac(self) -> float:
        return float(np.abs(self.t_hac).max())

    @property
    def sup_t_ols(self) -> float:
        return float(np.abs(self.t_ols).max())

    @property
    def min_ssr(self) -> float:
        return float(self.ssr.min())

    def at(self, tau: int) -> dict:
        i = int(np.searchsorted(self.dates, tau))
        if i >= self.dates.size or self.dates[i] != tau:
            raise KeyError(f"tau={tau} is not in the scanned grid")
        return {"tau": tau, "coef": float(self.coef[i]),
                "t_ols": float(self.t_ols[i]), "t_hac": float(self.t_hac[i]),
                "ssr": float(self.ssr[i])}


def _candidates(n: int, trim: float, stride: int, kind: str) -> np.ndarray:
    lo, hi = int(np.floor(trim * n)), int(np.ceil((1 - trim) * n))
    # A break at the first observation is not a break, it is the intercept, and
    # `trim=0` would otherwise put it on the grid. The end is trimmed by two for
    # a bend, which needs a slope on the far side rather than just a level.
    lo = max(lo, 1)
    hi = min(hi, n - 2 if kind == "bend" else n - 1)
    if hi - lo < 2:
        raise ValueError("trim leaves no candidate break dates")
    return np.arange(lo, hi, max(int(stride), 1))


def _hac_var_from_scores(u: np.ndarray, lags: int, dof_scale: float) -> float:
    """Bartlett-weighted long-run variance of a single score sequence."""
    S = float(u @ u)
    for L in range(1, lags + 1):
        G = float(u[L:] @ u[:-L])
        S += 2.0 * (1.0 - L / (lags + 1.0)) * G
    return S * dof_scale


def scan(y: np.ndarray, *, kind: str = "bend", trim: float = 0.15,
         stride: int = 1, trend: bool | int = True, seasonal: int = 0,
         start_period: int = 0, lags: int | None = None) -> BreakScan:
    """Fit the break at every candidate date, by Frisch-Waugh.

    The base design is fixed across dates, so its projection is computed once
    and every candidate reduces to a univariate regression of the residualised
    outcome on the residualised break column. That makes a full scan cost about
    what one ordinary fit costs, which is what makes the bootstrap work in this
    module affordable.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    lags = newey_west_lags(n) if lags is None else int(lags)
    if lags < 0:
        raise ValueError("lags must be non-negative")

    Z = bend_design(n, tau=None, trend=trend, seasonal=seasonal,
                    start_period=start_period)
    k0 = Z.shape[1]
    # One QR of the base design serves every candidate date.
    Q, _ = np.linalg.qr(Z)
    ry = y - Q @ (Q.T @ y)
    ssr_null = float(ry @ ry)

    taus = _candidates(n, trim, stride, kind)
    dof_scale = n / (n - k0 - 1)
    coef = np.empty(taus.size)
    t_ols = np.empty(taus.size)
    t_hac = np.empty(taus.size)
    ssr = np.empty(taus.size)

    for i, tau in enumerate(taus):
        x = break_column(n, int(tau), kind)
        rx = x - Q @ (Q.T @ x)
        sxx = float(rx @ rx)
        if sxx <= 0:
            # Collinear with the base design. A step at tau is collinear with a
            # seasonal dummy only in degenerate samples, but a bend can become
            # nearly collinear with a linear trend when tau sits at the very
            # start, so this is a real case rather than a defensive stub.
            coef[i] = t_ols[i] = t_hac[i] = np.nan
            ssr[i] = np.inf
            continue
        b = float(rx @ ry) / sxx
        resid = ry - b * rx
        s = float(resid @ resid)
        coef[i] = b
        ssr[i] = s
        var_ols = (s / (n - k0 - 1)) / sxx
        var_hac = _hac_var_from_scores(rx * resid, lags, dof_scale) / sxx ** 2
        # A zero residual is not a pathology to be guarded against reluctantly:
        # `noise_free_step_date` deliberately fits a bend to a noiseless bend and
        # lands exactly on it, and the t-statistic there is genuinely infinite.
        # Returning inf says that; dividing by zero and warning does not.
        t_ols[i] = b / np.sqrt(var_ols) if var_ols > 0 else np.inf * np.sign(b)
        t_hac[i] = b / np.sqrt(var_hac) if var_hac > 0 else np.inf * np.sign(b)

    return BreakScan(kind=kind, dates=taus, coef=coef, t_ols=t_ols,
                     t_hac=t_hac, ssr=ssr, ssr_null=ssr_null, lags=lags)


def slow_scan(y: np.ndarray, *, kind: str = "bend", trim: float = 0.15,
              stride: int = 1, trend: bool | int = True, seasonal: int = 0,
              start_period: int = 0, lags: int | None = None) -> BreakScan:
    """The same scan, done the obvious way, for checking `scan` against.

    Kept in the shipped module rather than in the tests because it is the only
    thing that makes the fast path trustworthy, and a reader of the fast path
    should be able to find it.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    lags = newey_west_lags(n) if lags is None else int(lags)
    taus = _candidates(n, trim, stride, kind)
    Z = bend_design(n, tau=None, trend=trend, seasonal=seasonal,
                    start_period=start_period)
    fit0 = ols_hac(Z, y, lags=lags)
    out = {k: np.empty(taus.size) for k in ("coef", "t_ols", "t_hac", "ssr")}
    for i, tau in enumerate(taus):
        X = bend_design(n, tau=int(tau), kind=kind, trend=trend,
                        seasonal=seasonal, start_period=start_period)
        fit = ols_hac(X, y, lags=lags)
        j = X.shape[1] - 1
        out["coef"][i] = fit.beta[j]
        out["t_ols"][i] = fit.t_ols(j)
        out["t_hac"][i] = fit.t_hac(j)
        out["ssr"][i] = fit.resid @ fit.resid
    return BreakScan(kind=kind, dates=taus, ssr_null=float(fit0.resid @ fit0.resid),
                     lags=lags, **out)


def fit_at(y: np.ndarray, tau: int, *, kind: str = "bend",
           trend: bool | int = True, seasonal: int = 0, start_period: int = 0,
           lags: int | None = None) -> dict:
    """One break, at a date fixed in advance, reported both ways."""
    y = np.asarray(y, dtype=float)
    X = bend_design(y.size, tau=tau, kind=kind, trend=trend, seasonal=seasonal,
                    start_period=start_period)
    fit = ols_hac(X, y, lags=lags)
    j = X.shape[1] - 1
    return {"tau": tau, "kind": kind, "coef": float(fit.beta[j]),
            "se_ols": float(fit.se_ols[j]), "se_hac": float(fit.se_hac[j]),
            "t_ols": fit.t_ols(j), "t_hac": fit.t_hac(j),
            "ssr": float(fit.resid @ fit.resid),
            "inflation": float(fit.se_hac[j] / fit.se_ols[j]),
            "resid": fit.resid, "lags": fit.lags}


# ---------------------------------------------------------------------------
# what a step does to a bend
# ---------------------------------------------------------------------------

def _deterministic_bend(n: int, tau: int, size: float) -> np.ndarray:
    """A noiseless trend with a slope change of `size` at `tau`."""
    return break_column(n, tau, "bend") * size


def noise_free_step_date(n: int, tau: int, *, size: float = 1.0,
                         trim: float = 0.15, trend: bool | int = True,
                         seasonal: int = 0, start_period: int = 0) -> dict:
    """Where a step lands when fitted to a bend and there is no noise at all.

    This is the cleanest statement of the problem available, because nothing
    random is involved: the series is a deterministic kinked line, the profile
    SSR is a deterministic function of the candidate date, and its minimum is
    wherever it is. Any difference between that minimum and `tau` is bias in the
    estimator, not sampling error, and no quantity of data removes it.

    `size` cancels out of the argmin — the profile SSR scales by `size**2` — so
    the returned date does not depend on how sharp the bend is. It is returned
    anyway so a caller can see the scaling for themselves.
    """
    y = _deterministic_bend(n, tau, size)
    s = scan(y, kind="step", trim=trim, trend=trend, seasonal=seasonal,
             start_period=start_period, lags=0)
    b = scan(y, kind="bend", trim=trim, trend=trend, seasonal=seasonal,
             start_period=start_period, lags=0)
    # The bend fits a bend exactly, so its SSR is zero up to rounding and the
    # ratio of the two is not a meaningful number. What is meaningful is how much
    # of the signal the step leaves behind, as a share of the signal's own
    # variation — a step misses that fraction of a shape that a bend captures
    # completely.
    var_signal = float(((y - y.mean()) ** 2).sum())
    return {"n": n, "tau": tau, "size": size,
            "step_date": s.best, "step_error": s.best - tau,
            "bend_date": b.best, "bend_error": b.best - tau,
            "step_min_ssr": s.min_ssr, "bend_min_ssr": b.min_ssr,
            "step_unexplained": s.min_ssr / var_signal if var_signal else 0.0}


def step_on_bend(resid: np.ndarray, *, n: int, tau: int, size: float,
                 block: int, reps: int = 400, trim: float = 0.15,
                 trend: bool | int = True, seasonal: int = 0,
                 start_period: int = 0, lags: int | None = None,
                 rng: np.random.Generator | None = None) -> dict:
    """Fit both a step and a bend to data that really did bend, many times.

    The noise is a moving-block bootstrap of `resid`, so it carries the real
    series' persistence; the signal is the deterministic bend. What comes back
    is the sampling distribution of each estimated date, and the share of
    replications in which the bend beats the step on SSR at their own optimal
    dates — the quantity a practitioner would actually use to choose.
    """
    rng = rng or np.random.default_rng(20260827)
    signal = _deterministic_bend(n, tau, size)
    step_dates = np.empty(reps, dtype=int)
    bend_dates = np.empty(reps, dtype=int)
    margin = np.empty(reps)
    bend_wins = 0
    for i in range(reps):
        y = signal + moving_block_bootstrap(resid, block=block, size=n, rng=rng)
        s = scan(y, kind="step", trim=trim, trend=trend, seasonal=seasonal,
                 start_period=start_period, lags=lags)
        b = scan(y, kind="bend", trim=trim, trend=trend, seasonal=seasonal,
                 start_period=start_period, lags=lags)
        step_dates[i] = s.best
        bend_dates[i] = b.best
        bend_wins += b.min_ssr < s.min_ssr
        # The signed margin, on the same definition `model_race` uses, so that
        # this distribution and `null_race`'s are directly comparable: together
        # they give the size and the power of the same one-sided comparison.
        big, small = max(s.min_ssr, b.min_ssr), min(s.min_ssr, b.min_ssr)
        g = 1.0 - small / big
        margin[i] = g if b.min_ssr < s.min_ssr else -g
    return {"n": n, "tau": tau, "size": size, "reps": reps,
            "margin": margin, "margin_median": float(np.median(margin)),
            "step_median": float(np.median(step_dates)),
            "step_bias": float(np.median(step_dates) - tau),
            "step_iqr": [float(np.quantile(step_dates, 0.25)),
                         float(np.quantile(step_dates, 0.75))],
            "bend_median": float(np.median(bend_dates)),
            "bend_bias": float(np.median(bend_dates) - tau),
            "bend_iqr": [float(np.quantile(bend_dates, 0.25)),
                         float(np.quantile(bend_dates, 0.75))],
            "bend_wins": bend_wins / reps,
            "step_dates": step_dates, "bend_dates": bend_dates}


# ---------------------------------------------------------------------------
# choosing between the two shapes, honestly
# ---------------------------------------------------------------------------

def model_race(y: np.ndarray, **kw) -> dict:
    """Which shape fits better, each at its own best date.

    Both add one column to the same base design and both are searched over the
    same grid, so the comparison needs no penalty term. What it does need is a
    null: see `null_race`.
    """
    s = scan(y, kind="step", **kw)
    b = scan(y, kind="bend", **kw)
    return {"step_date": s.best, "bend_date": b.best,
            "step_ssr": s.min_ssr, "bend_ssr": b.min_ssr,
            "winner": "bend" if b.min_ssr < s.min_ssr else "step",
            "ssr_gain": float(1.0 - min(s.min_ssr, b.min_ssr)
                              / max(s.min_ssr, b.min_ssr)),
            "step_sup_t": s.sup_t_hac, "bend_sup_t": b.sup_t_hac,
            "r2_step": 1.0 - s.min_ssr / s.ssr_null,
            "r2_bend": 1.0 - b.min_ssr / b.ssr_null}


def null_race(resid: np.ndarray, *, n: int, block: int, reps: int = 500,
              rng: np.random.Generator | None = None, **kw) -> dict:
    """How often the bend wins on data where nothing happened.

    Without this number "the bend fits better" means nothing. The two shapes are
    not symmetric — a bend is a smoother function of the date than a step, so
    there is no reason to expect a 50/50 split under the null, and the split has
    to be measured before any real-data win can be read as evidence.
    """
    rng = rng or np.random.default_rng(20260828)
    wins = 0
    gains = np.empty(reps)
    for i in range(reps):
        u = moving_block_bootstrap(resid, block=block, size=n, rng=rng)
        r = model_race(u, **kw)
        wins += r["winner"] == "bend"
        gains[i] = r["ssr_gain"] if r["winner"] == "bend" else -r["ssr_gain"]
    # `gains` is signed: positive when the bend won by that margin, negative
    # when the step did. So the two tails score the two directions, and a real
    # win in either direction has to beat its own tail rather than the other's.
    return {"bend_win_share": wins / reps, "reps": reps,
            "bend_gain_q95": float(np.quantile(gains, 0.95)),
            "step_gain_q95": float(-np.quantile(gains, 0.05)),
            "bend_gain_q90": float(np.quantile(gains, 0.90)),
            "step_gain_q90": float(-np.quantile(gains, 0.10))}


def calibrated_sup(resid: np.ndarray, *, n: int, block: int, kind: str = "bend",
                   level: float = 0.05, reps: int = 600,
                   fixed_at: int | None = None,
                   rng: np.random.Generator | None = None, **kw) -> dict:
    """Critical values for this design's own null, fixed-date and searched.

    Both are returned from the same replications so they are directly
    comparable: the searched value is the quantile of the maximum over dates,
    the fixed one the quantile at a single date. The gap between them is the
    price of not having known the date in advance.
    """
    rng = rng or np.random.default_rng(20260829)
    sup = np.empty(reps)
    fixed = np.empty(reps)
    for i in range(reps):
        u = moving_block_bootstrap(resid, block=block, size=n, rng=rng)
        s = scan(u, kind=kind, **kw)
        sup[i] = s.sup_t_hac
        tau = s.dates[s.dates.size // 2] if fixed_at is None else fixed_at
        fixed[i] = abs(s.at(int(tau))["t_hac"])
    q = 1.0 - level
    return {"kind": kind, "level": level, "reps": reps,
            "sup": float(np.quantile(sup, q)),
            "fixed": float(np.quantile(fixed, q)),
            "size_of_1p96_sup": float((sup > 1.959963984540054).mean()),
            "size_of_1p96_fixed": float((fixed > 1.959963984540054).mean())}


# ---------------------------------------------------------------------------
# the date itself
# ---------------------------------------------------------------------------

def date_bootstrap(y: np.ndarray, *, block: int, kind: str = "bend",
                   reps: int = 600, level: float = 0.90,
                   rng: np.random.Generator | None = None, **kw) -> dict:
    """An interval for the break date, by resampling the fitted residuals.

    The mean function fitted at the best date is held fixed, its residuals are
    resampled in blocks, the series is rebuilt and the date is re-estimated.
    The spread of those re-estimates is the interval. This is the quantity that
    almost never gets reported: a break date is printed as a month, with no
    indication that the next resample would have named a different year.

    `date_coverage` checks that this interval covers a known truth at the
    advertised rate, because a bootstrap interval for a date is not guaranteed
    to and this one should not be trusted on faith.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    rng = rng or np.random.default_rng(20260830)
    base = scan(y, kind=kind, **kw)
    tau_hat = base.best
    f = fit_at(y, tau_hat, kind=kind,
               **{k: v for k, v in kw.items() if k not in ("trim", "stride")})
    fitted = y - f["resid"]
    draws = np.empty(reps, dtype=int)
    for i in range(reps):
        u = moving_block_bootstrap(f["resid"], block=block, size=n, rng=rng)
        draws[i] = scan(fitted + u, kind=kind, **kw).best
    a = (1.0 - level) / 2.0
    return {"kind": kind, "tau_hat": tau_hat, "reps": reps, "level": level,
            "lo": int(np.quantile(draws, a)), "hi": int(np.quantile(draws, 1 - a)),
            "sd": float(draws.std(ddof=1)), "draws": draws,
            "coef": f["coef"], "t_hac": f["t_hac"]}


def date_coverage(resid: np.ndarray, *, n: int, tau: int, size: float,
                  block: int, kind: str = "bend", reps: int = 120,
                  inner: int = 200, level: float = 0.90,
                  rng: np.random.Generator | None = None, **kw) -> dict:
    """Does the date interval cover the truth as often as it says?

    Generates series with a known break, builds the interval on each, and counts
    how often the true date falls inside. A nominal 90% interval that covers 60%
    of the time is worth knowing about before publishing one; so is a nominal
    90% that covers 99%, which would mean the interval is uselessly wide.
    """
    rng = rng or np.random.default_rng(20260831)
    signal = (_deterministic_bend(n, tau, size) if kind == "bend"
              else break_column(n, tau, "step") * size)
    hit = 0
    widths = np.empty(reps)
    for i in range(reps):
        y = signal + moving_block_bootstrap(resid, block=block, size=n, rng=rng)
        ci = date_bootstrap(y, block=block, kind=kind, reps=inner, level=level,
                            rng=np.random.default_rng(9000 + i), **kw)
        hit += ci["lo"] <= tau <= ci["hi"]
        widths[i] = ci["hi"] - ci["lo"]
    return {"nominal": level, "covered": hit / reps, "reps": reps,
            "inner": inner, "median_width": float(np.median(widths)),
            "tau": tau, "size": size, "kind": kind}


# ---------------------------------------------------------------------------
# power, for a bend whose date is announced rather than searched for
# ---------------------------------------------------------------------------

def size_from_slope(slope_per_period: float, n: int) -> float:
    """Convert a slope change per period into a bend coefficient.

    The bend column is scaled by `n - 1`, so the same economic claim — "growth
    slows by 0.01 log points a year" — is a *different* coefficient at every
    sample length. Every power calculation below therefore takes the claim in
    per-period units and converts here, rather than carrying a coefficient
    across sample sizes and quietly rescaling it.
    """
    if n < 2:
        raise ValueError("need at least two observations")
    return slope_per_period * (n - 1)


def bend_power(resid: np.ndarray, *, n_pre: int, n_post: int,
               slope_change: float, block: int, reps: int = 800,
               critical: float | None = None, alpha: float = 0.05,
               rng: np.random.Generator | None = None, **kw) -> dict:
    """Probability of detecting a bend of a stated slope change, by simulation.

    The date is treated as **known in advance**, which is the right treatment
    when a published plan names the year the trajectory changes: there is no
    search, so no search penalty. That makes this a more favourable calculation
    than the one in `calibrated_sup`, and the favourable one is the honest one
    here — a forecast that cannot be confirmed even with the date given away is
    not going to be confirmed by a date search.
    """
    rng = rng or np.random.default_rng(20260901)
    n = n_pre + n_post
    signal = break_column(n, n_pre, "bend") * size_from_slope(slope_change, n)
    z = critical if critical is not None else (
        1.959963984540054 if alpha == 0.05 else _z_two_sided(alpha))
    hit_ols = hit_hac = 0
    for _ in range(reps):
        y = signal + moving_block_bootstrap(resid, block=block, size=n, rng=rng)
        f = fit_at(y, n_pre, kind="bend", **kw)
        hit_ols += abs(f["t_ols"]) > z
        hit_hac += abs(f["t_hac"]) > z
    return {"n_pre": n_pre, "n_post": n_post, "slope_change": slope_change,
            "power_ols": hit_ols / reps, "power_hac": hit_hac / reps,
            "reps": reps, "critical": z}


def _z_two_sided(alpha: float) -> float:
    from .nonstationary import _norm_ppf
    return _norm_ppf(1.0 - alpha / 2.0)


def minimum_detectable_bend(resid: np.ndarray, *, n_pre: int, n_post: int,
                            block: int, target: float = 0.80,
                            reps: int = 500, lo: float = 0.0, hi: float = 0.20,
                            tol: float = 1e-4, use: str = "hac",
                            critical: float | None = None, **kw) -> dict:
    """Smallest per-period slope change detectable at `target` power.

    Returned in the units of `resid` per period — log points per month if the
    series was logged and monthly, so 0.004 means "growth would have to change
    by 0.4 log points a month, and nothing smaller shows up".
    """
    if use not in ("hac", "ols"):
        raise ValueError("use must be 'hac' or 'ols'")
    key = f"power_{use}"

    def power(size, seed):
        return bend_power(resid, n_pre=n_pre, n_post=n_post,
                          slope_change=size, block=block, reps=reps,
                          critical=critical, rng=np.random.default_rng(seed),
                          **kw)[key]

    hi_p = power(hi, 2)
    if hi_p < target:
        return {"mde": float("inf"), "power_at_hi": hi_p, "hi": hi,
                "note": "even the largest slope change searched is not detectable"}
    it = 0
    while hi - lo > tol and it < 40:
        mid = 0.5 * (lo + hi)
        if power(mid, 100 + it) < target:
            lo = mid
        else:
            hi = mid
        it += 1
    return {"mde": 0.5 * (lo + hi), "target": target, "iterations": it,
            "use": use, "n_pre": n_pre, "n_post": n_post}


def periods_to_detect(resid: np.ndarray, *, n_pre: int, slope_change: float,
                      block: int, candidates, target: float = 0.80,
                      reps: int = 500, use: str = "hac",
                      critical: float | None = None, **kw) -> dict:
    """How many post-break periods it takes to reach `target` power.

    Returns the power at each candidate horizon as well as the first one that
    clears, because a curve that never clears is a result and a single "not
    detectable" is not informative about how far off it was.
    """
    key = f"power_{use}"
    curve = {}
    for i, n_post in enumerate(candidates):
        curve[int(n_post)] = bend_power(
            resid, n_pre=n_pre, n_post=int(n_post), slope_change=slope_change,
            block=block, reps=reps, critical=critical,
            rng=np.random.default_rng(300 + i), **kw)[key]
    cleared = [k for k, v in curve.items() if v >= target]
    return {"curve": curve, "first_cleared": min(cleared) if cleared else None,
            "target": target, "slope_change": slope_change, "use": use}


def calibrated_fixed(resid: np.ndarray, *, n_pre: int, n_post: int, block: int,
                     level: float = 0.05, reps: int = 2000,
                     rng: np.random.Generator | None = None, **kw) -> dict:
    """The critical value for a bend at a date fixed in advance, at this length.

    Kept separate from `calibrated_sup` for two reasons. It is far cheaper — one
    fit per replication instead of a scan — and, more importantly, the bar has to
    be recomputed at every sample length a power curve visits. Calibrating once
    at the historical length and then reusing that number as the sample grows
    reports power against the wrong bar, and reports it as though the growth in
    power came from the signal.

    The size that 1.96 would have delivered is returned alongside, because on a
    persistent series that gap is most of the difference between a power curve
    and a false-positive curve.
    """
    rng = rng or np.random.default_rng(20260902)
    n = n_pre + n_post
    stats = np.empty(reps)
    for i in range(reps):
        u = moving_block_bootstrap(resid, block=block, size=n, rng=rng)
        stats[i] = fit_at(u, n_pre, kind="bend", **kw)["t_hac"]
    a = np.abs(stats)
    return {"critical": float(np.quantile(a, 1.0 - level)),
            "size_of_1p96": float((a > 1.959963984540054).mean()),
            "n_pre": n_pre, "n_post": n_post, "level": level, "reps": reps}
