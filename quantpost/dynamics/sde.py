"""Stochastic generators with known ground truth.

The point of these is the same as the ODEs: a controlled environment where you
know the answer. If your volatility forecaster cannot recover the variance
process from a Heston path where you *generated* the variance, its performance on
real returns means nothing.

`heston` uses the full truncation Euler scheme (Lord, Koekkoek & van Dijk 2010):
the variance is truncated at zero inside the drift and diffusion but the state is
allowed to be reflected honestly, which keeps the discretisation bias far smaller
than naive absorption at zero.

`rough_bergomi` uses fractional Brownian motion via the Davies-Harte circulant
embedding — exact for the covariance structure, unlike the Cholesky hack, and
O(n log n) rather than O(n^3).

`garch11` is the discrete-time member of the family, and `money_weighted_return`
sits beside it because the two are used together: a cash-flow stream's IRR only
differs from the index's return when volatility clusters, so the generator and the
measurement belong in the same place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Path:
    t: np.ndarray
    data: dict[str, np.ndarray]
    dt: float
    system: str
    params: dict = field(default_factory=dict)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.data, index=self.t)

    def describe(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.system} ({p}), dt={self.dt}, {len(self.t)} steps"


def ornstein_uhlenbeck(
    n_steps: int = 5000, dt: float = 1 / 252, *,
    theta: float = 3.0, mu: float = 0.0, sigma: float = 0.2,
    x0: float | None = None, seed: int | None = 0,
) -> Path:
    """Exact-transition OU. Used as the 'mean reversion exists' control."""
    rng = np.random.default_rng(seed)
    x = np.empty(n_steps)
    x[0] = mu if x0 is None else x0
    a = np.exp(-theta * dt)
    s = sigma * np.sqrt((1.0 - a ** 2) / (2.0 * theta))
    z = rng.standard_normal(n_steps - 1)
    for i in range(1, n_steps):
        x[i] = mu + a * (x[i - 1] - mu) + s * z[i - 1]
    return Path(np.arange(n_steps) * dt, {"x": x}, dt, "Ornstein-Uhlenbeck",
                {"theta": theta, "mu": mu, "sigma": sigma})


def heston(
    n_steps: int = 5000, dt: float = 1 / 252, *,
    s0: float = 100.0, v0: float = 0.04, kappa: float = 2.0,
    theta: float = 0.04, xi: float = 0.5, rho: float = -0.7,
    mu: float = 0.0, seed: int | None = 0,
) -> Path:
    """Heston with full-truncation Euler. Returns price, log-return and the
    latent variance — the last one is what makes this a supervised problem."""
    feller = 2.0 * kappa * theta - xi ** 2
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_steps - 1, 2))
    dw_v = z[:, 0] * np.sqrt(dt)
    dw_s = (rho * z[:, 0] + np.sqrt(1.0 - rho ** 2) * z[:, 1]) * np.sqrt(dt)

    v = np.empty(n_steps); v[0] = v0
    log_s = np.empty(n_steps); log_s[0] = np.log(s0)
    for i in range(1, n_steps):
        v_pos = max(v[i - 1], 0.0)                       # full truncation
        v[i] = (v[i - 1] + kappa * (theta - v_pos) * dt
                + xi * np.sqrt(v_pos) * dw_v[i - 1])
        log_s[i] = (log_s[i - 1] + (mu - 0.5 * v_pos) * dt
                    + np.sqrt(v_pos) * dw_s[i - 1])

    s = np.exp(log_s)
    ret = np.diff(log_s, prepend=log_s[0])
    return Path(np.arange(n_steps) * dt,
                {"price": s, "log_return": ret, "variance": np.maximum(v, 0.0)},
                dt, "Heston",
                {"kappa": kappa, "theta": theta, "xi": xi, "rho": rho,
                 "feller_satisfied": bool(feller > 0)})


def fbm(n: int, hurst: float, *, seed: int | None = 0) -> np.ndarray:
    """Fractional Brownian motion increments on [0,1] via Davies-Harte.

    Exact in covariance. Falls back to Hosking recursion when the circulant
    embedding is not non-negative definite (can happen for H very close to 1).
    """
    if not 0.0 < hurst < 1.0:
        raise ValueError("hurst must be in (0,1)")
    rng = np.random.default_rng(seed)
    H = hurst

    def gamma(k: int) -> float:
        return 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H)
                      + abs(k + 1) ** (2 * H))

    m = 1
    while m < n:
        m *= 2
    row = np.array([gamma(k) for k in range(m + 1)])
    circ = np.concatenate([row, row[-2:0:-1]])
    lam = np.real(np.fft.fft(circ))
    if np.min(lam) < -1e-10:
        return _fbm_hosking(n, H, rng)
    lam = np.clip(lam, 0.0, None)

    size = len(circ)
    w = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    w[0] = np.sqrt(2.0) * rng.standard_normal()
    w[size // 2] = np.sqrt(2.0) * rng.standard_normal()
    w[size // 2 + 1:] = np.conj(w[1:size // 2][::-1])
    incr = np.real(np.fft.fft(np.sqrt(lam / (2.0 * size)) * w))[:n]
    return incr * (1.0 / n) ** H


def _fbm_hosking(n: int, H: float, rng) -> np.ndarray:
    def gamma(k):
        return 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H)
                      + abs(k + 1) ** (2 * H))
    x = np.zeros(n)
    x[0] = rng.standard_normal()
    d = np.zeros(n)
    v = 1.0
    for i in range(1, n):
        phi = gamma(i)
        for j in range(i - 1):
            phi -= d[j] * gamma(i - j - 1)
        phi /= v
        d[:i - 1] -= phi * d[i - 2::-1]
        d[i - 1] = phi
        v *= (1.0 - phi ** 2)
        x[i] = np.dot(d[:i], x[i - 1::-1]) + np.sqrt(v) * rng.standard_normal()
    return x * (1.0 / n) ** H


def rough_bergomi(
    n_steps: int = 2520, *, hurst: float = 0.1, eta: float = 1.9,
    xi0: float = 0.04, rho: float = -0.9, s0: float = 100.0,
    horizon_years: float = 10.0, seed: int | None = 0,
) -> Path:
    """Rough Bergomi. Realised volatility is rougher than any GARCH can make it,
    which is exactly why it is a useful adversary for a volatility model."""
    dt = horizon_years / n_steps
    rng = np.random.default_rng(seed)
    dW1 = fbm(n_steps, hurst, seed=seed) * (horizon_years ** hurst)
    # Volterra kernel gives the forward variance curve.
    t = np.linspace(dt, horizon_years, n_steps)
    Y = np.cumsum(dW1)
    v = xi0 * np.exp(eta * Y - 0.5 * eta ** 2 * t ** (2 * hurst))
    z = rng.standard_normal(n_steps)
    dW1n = np.diff(Y, prepend=0.0)
    dW1n = dW1n / (np.std(dW1n) + 1e-12) * np.sqrt(dt)
    dB = rho * dW1n + np.sqrt(1.0 - rho ** 2) * z * np.sqrt(dt)
    log_s = np.log(s0) + np.cumsum(np.sqrt(np.maximum(v, 0)) * dB
                                   - 0.5 * v * dt)
    return Path(t, {"price": np.exp(log_s),
                    "log_return": np.diff(log_s, prepend=log_s[0]),
                    "variance": v},
                dt, "rough Bergomi",
                {"H": hurst, "eta": eta, "xi0": xi0, "rho": rho})


def hawkes_exp(
    horizon: float = 1000.0, *, mu: float = 0.5, alpha: float = 0.6,
    beta: float = 1.0, seed: int | None = 0,
) -> Path:
    """Univariate Hawkes with exponential kernel, via Ogata thinning.

    Self-exciting arrivals are the natural model for order flow and for default
    clustering. Branching ratio alpha/beta must be < 1 for stationarity; the
    generator refuses otherwise instead of silently exploding.
    """
    if alpha >= beta:
        raise ValueError(
            f"branching ratio alpha/beta = {alpha/beta:.2f} >= 1: the process is "
            "non-stationary and simulation will not terminate")
    rng = np.random.default_rng(seed)
    events: list[float] = []
    t = 0.0
    while t < horizon:
        lam_bar = mu + sum(alpha * np.exp(-beta * (t - s)) for s in events[-500:])
        w = rng.exponential(1.0 / lam_bar)
        t += w
        if t >= horizon:
            break
        lam = mu + sum(alpha * np.exp(-beta * (t - s)) for s in events[-500:])
        if rng.uniform() <= lam / lam_bar:
            events.append(t)
    arr = np.asarray(events)
    grid = np.arange(0.0, horizon, 1.0)
    counts = np.histogram(arr, bins=np.append(grid, horizon))[0].astype(float)
    return Path(grid, {"count": counts}, 1.0, "Hawkes (exponential)",
                {"mu": mu, "alpha": alpha, "beta": beta,
                 "branching_ratio": round(alpha / beta, 3),
                 "n_events": len(arr)})


def garch11(
    n: int = 6000, *,
    omega: float = 0.02, arch: float = 0.10, beta: float = 0.88,
    df: float = 5.0, seed: int | None = 5,
) -> Path:
    """GARCH(1,1) returns with Student-t shocks: fat tails *and* clustering.

    The discrete-time counterpart of `heston` for this repo's purposes — a process
    where you generated the conditional variance, so you can check whether a model
    recovers it. Returns are in percent.

    Two details that matter for using it as ground truth:

    * The shocks are **rescaled to unit variance** (`df/(df-2)`), so `omega`,
      `arch` and `beta` mean what the textbook says and the unconditional variance
      is `omega / (1 - arch - beta)` regardless of `df`. Skipping that step makes
      every fitted parameter wrong by a factor nobody notices.
    * Persistence `arch + beta` must be below 1 for a stationary variance; 0.98 is
      the usual equity-index estimate and is what makes large moves arrive next to
      large moves. Setting it to 0 gives i.i.d. draws, which is the control any
      claim about clustering needs.

    `df <= 2` is refused: a t with two degrees of freedom has no finite variance,
    so there is nothing to rescale and the "unit variance shock" is a fiction.
    """
    if not 0.0 <= arch + beta < 1.0:
        raise ValueError(f"arch + beta must lie in [0, 1); got {arch + beta}")
    if omega <= 0:
        raise ValueError("omega must be positive")
    if df <= 2:
        raise ValueError("t shocks need df > 2 for a finite variance to rescale")
    from scipy import stats

    rng = np.random.default_rng(seed)
    z = stats.t.rvs(df, size=n, random_state=rng) / np.sqrt(df / (df - 2.0))
    r = np.empty(n)
    h = np.empty(n)
    h[0] = omega / (1.0 - arch - beta) if arch + beta else omega
    for t in range(n):
        if t:
            h[t] = omega + arch * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * z[t]
    return Path(t=np.arange(n, dtype=float), data={"r": r, "h": h}, dt=1.0,
                system="garch11",
                params={"omega": omega, "arch": arch, "beta": beta, "df": df,
                        "persistence": arch + beta,
                        "uncond_sd": float(np.sqrt(
                            omega / (1.0 - arch - beta)) if arch + beta
                            else np.sqrt(omega))})


def simple_from_log(log_pct, *, drift_annual: float = 0.0,
                    periods_per_year: float = 252.0) -> np.ndarray:
    """Percentage *simple* returns from percentage *log* returns, plus a drift.

    Use this before compounding anything a GARCH generator produced. A GARCH path
    is a log-return path, and treating it as simple returns is not a harmless
    reinterpretation: with persistence near one and Student-t shocks, a tail draw
    reaches -100% every few hundred simulated years, at which point
    `prod(1 + r/100)` turns **negative** and annualising it — raising a negative
    total to a fractional power — returns a complex number. NumPy casts that back
    to a float with a warning nobody reads, and the resulting "return" is silently
    wrong rather than obviously wrong.

    Converting through `expm1` bounds the simple return below by -100% by
    construction, which is also the only value an index can actually have.

    `drift_annual` is the compounded annual drift added in log space; a raw GARCH
    path has none, so an experiment that needs a market which *rises* has to say
    so rather than accidentally testing a zero-drift world.
    """
    g = np.asarray(log_pct, float)
    mu = 100.0 * np.log1p(drift_annual) / periods_per_year
    return 100.0 * np.expm1((g + mu) / 100.0)


def money_weighted_return(
    cashflows, final_value: float, *, periods_per_year: float = 252.0,
    lo: float = -0.999, hi: float = 100.0, tol: float = 1e-10,
) -> float:
    """Internal rate of return of a cash-flow stream, annualised.

    The number an investor actually earned, as opposed to the time-weighted return
    the fund reports. They differ whenever money moves, and the gap is the whole
    subject of the behaviour-gap literature — so it is worth solving properly
    rather than approximating by "average of the periods you were invested".

    `cashflows[t]` is money paid **in** at period `t` (negative for withdrawals),
    `final_value` the account's worth after the last period. Solved by bisection on
    the terminal-value identity, which is monotone in the rate, so there is no
    multiple-root problem of the kind that makes naive IRR solvers unreliable.

    Returns nan when no rate can explain the flows — for instance a stream whose
    contributions exceed any achievable terminal value.
    """
    cf = np.asarray(cashflows, float).ravel()
    n = cf.size
    if n == 0:
        raise ValueError("no cashflows")

    def terminal(rate: float) -> float:
        # (1 + rate) is per year; each flow compounds for the periods that remain.
        growth = (1.0 + rate) ** ((n - np.arange(n)) / periods_per_year)
        return float(np.sum(cf * growth)) - final_value

    f_lo, f_hi = terminal(lo), terminal(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = terminal(mid)
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))
