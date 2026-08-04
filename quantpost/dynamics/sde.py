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
