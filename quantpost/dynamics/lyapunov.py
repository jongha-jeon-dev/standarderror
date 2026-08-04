"""Lyapunov exponents and the timescale that makes forecast claims comparable.

Why this module earns its place: "RMSE 0.03" is meaningless across systems, but
"stayed within 0.3 normalised error for 6.2 Lyapunov times" is not. The Lyapunov
time is the natural clock of a chaotic system, and reporting horizons in it is
what lets a reader compare your Lorenz result to a KS result to someone else's
paper.

Two estimators:

* `lyapunov_from_jacobian` — for systems where you have the analytic Jacobian.
  Benettin/Shimada tangent-space method with Gram-Schmidt renormalisation. This
  is the accurate one; use it when you can.
* `lyapunov_rosenstein` — from a scalar time series only (Rosenstein et al.
  1993). Use it on data. It is biased and needs the linear-scaling region picked
  by eye, so `fit_window` is explicit rather than auto-detected: an estimator that
  silently chooses its own fit range is an estimator you cannot defend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass
class LyapunovResult:
    exponent: float
    method: str
    lyapunov_time: float
    detail: dict

    def __str__(self) -> str:
        return (f"lambda_max = {self.exponent:.4f} ({self.method}), "
                f"Lyapunov time = {self.lyapunov_time:.3f}")


def lyapunov_from_jacobian(
    jac,
    trajectory: np.ndarray,
    dt: float,
    *,
    n_exponents: int | None = None,
    skip: int = 100,
) -> LyapunovResult:
    """Benettin algorithm: evolve an orthonormal frame, renormalise every step.

    `jac(x)` must return the (d, d) Jacobian of the vector field at x.
    Returns the largest exponent; the full spectrum is in `.detail["spectrum"]`.
    """
    x = np.asarray(trajectory, float)
    T, d = x.shape
    k = n_exponents or d
    Q = np.linalg.qr(np.random.default_rng(0).standard_normal((d, k)))[0]
    sums = np.zeros(k)
    count = 0
    for i in range(T):
        J = np.asarray(jac(x[i]), float)
        # Propagate the tangent space with expm(J*dt), not an Euler step.
        # Euler biases the exponent upward by O(dt*||J||^2): on Lorenz-63 with
        # dt=0.01 that is +0.18 on a true value of 0.906 - a 20% error that
        # would silently corrupt every Lyapunov-time figure downstream.
        Q = expm(J * dt) @ Q
        Q, R = np.linalg.qr(Q)
        diag = np.abs(np.diag(R))
        diag = np.where(diag > 0, diag, 1e-300)
        if i >= skip:
            sums += np.log(diag)
            count += 1
    spectrum = sums / (count * dt)
    order = np.argsort(spectrum)[::-1]
    spectrum = spectrum[order]
    lam = float(spectrum[0])
    return LyapunovResult(
        lam, "Benettin (analytic Jacobian)",
        float("inf") if lam <= 0 else 1.0 / lam,
        {"spectrum": spectrum.tolist(), "n_steps_used": count,
         "kaplan_yorke_dim": _kaplan_yorke(spectrum)})


def lorenz_jacobian(sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    def jac(s):
        x, y, z = s
        return np.array([[-sigma, sigma, 0.0],
                         [rho - z, -1.0, -x],
                         [y, x, -beta]])
    return jac


def _kaplan_yorke(spectrum: np.ndarray) -> float | None:
    """Kaplan-Yorke dimension: j + (sum of first j)/|lambda_{j+1}|."""
    cum = np.cumsum(spectrum)
    j = int(np.sum(cum >= 0))
    if j == 0 or j >= len(spectrum):
        return None
    return float(j + cum[j - 1] / abs(spectrum[j]))


def lyapunov_rosenstein(
    series: np.ndarray,
    dt: float,
    *,
    emb_dim: int = 6,
    lag: int | None = None,
    min_sep: int | None = None,
    max_t: int = 60,
    fit_window: tuple[int, int] = (5, 30),
) -> LyapunovResult:
    """Rosenstein's method on a scalar series.

    `fit_window` is the (start, end) step range of the divergence curve fitted by
    least squares. You are expected to plot `detail["divergence"]` and check the
    window covers the linear region — the estimate is meaningless otherwise.
    """
    y = np.asarray(series, float).ravel()
    if lag is None:
        lag = _first_min_mutual_info(y)
    n_vec = len(y) - (emb_dim - 1) * lag
    if n_vec <= max_t + 2:
        raise ValueError("series too short for the requested embedding/max_t")
    emb = np.column_stack([y[i * lag: i * lag + n_vec] for i in range(emb_dim)])
    if min_sep is None:
        min_sep = (emb_dim - 1) * lag          # Theiler window

    # Nearest neighbour excluding temporal neighbours.
    d2 = np.sum(emb ** 2, axis=1)
    gram = emb @ emb.T
    dist = np.sqrt(np.maximum(d2[:, None] + d2[None, :] - 2 * gram, 0.0))
    idx = np.arange(n_vec)
    mask = np.abs(idx[:, None] - idx[None, :]) <= min_sep
    dist[mask] = np.inf
    neigh = np.argmin(dist, axis=1)

    usable = n_vec - max_t - 1
    curve = np.full(max_t + 1, np.nan)
    for k in range(max_t + 1):
        a = idx[:usable]
        b = neigh[:usable]
        ok = b + k < n_vec
        sep = np.linalg.norm(emb[a[ok] + k] - emb[b[ok] + k], axis=1)
        sep = sep[sep > 0]
        if len(sep):
            curve[k] = np.mean(np.log(sep))

    lo, hi = fit_window
    seg = np.arange(lo, min(hi, max_t) + 1)
    good = ~np.isnan(curve[seg])
    if good.sum() < 3:
        raise ValueError("fit_window contains too few finite points")
    slope, intercept = np.polyfit(seg[good] * dt, curve[seg][good], 1)
    lam = float(slope)
    resid = curve[seg][good] - (slope * seg[good] * dt + intercept)
    r2 = 1.0 - np.var(resid) / max(np.var(curve[seg][good]), 1e-30)
    return LyapunovResult(
        lam, "Rosenstein (scalar series)",
        float("inf") if lam <= 0 else 1.0 / lam,
        {"divergence": curve.tolist(), "lag": lag, "emb_dim": emb_dim,
         "fit_window": list(fit_window), "fit_r2": float(r2),
         "theiler_window": min_sep})


def _first_min_mutual_info(y: np.ndarray, max_lag: int = 50,
                           bins: int = 16) -> int:
    """Embedding lag = first local minimum of the delayed mutual information.

    Preferred over the first zero of the autocorrelation: MI captures nonlinear
    dependence, and for chaotic series the two disagree materially.
    """
    y = (y - y.min()) / (np.ptp(y) + 1e-30)
    disc = np.clip((y * bins).astype(int), 0, bins - 1)
    prev = np.inf
    for lag in range(1, min(max_lag, len(y) // 4)):
        a, b = disc[:-lag], disc[lag:]
        joint = np.histogram2d(a, b, bins=bins,
                              range=[[0, bins], [0, bins]])[0]
        joint = joint / joint.sum()
        pa = joint.sum(axis=1, keepdims=True)
        pb = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = joint * np.log(joint / (pa * pb))
        mi = float(np.nansum(term))
        if mi > prev:
            return max(lag - 1, 1)
        prev = mi
    return 1


def valid_prediction_time(
    truth: np.ndarray,
    pred: np.ndarray,
    dt: float,
    *,
    threshold: float = 0.3,
    lyapunov_exponent: float | None = None,
) -> dict:
    """Steps until normalised error first exceeds `threshold`.

    Normalisation is by the RMS of the truth over the whole evaluation window
    (the standard convention in the RC literature), so VPT is comparable across
    systems and across papers. Reported in Lyapunov times when an exponent is
    supplied — which is the only form worth quoting.
    """
    truth = np.atleast_2d(np.asarray(truth, float))
    pred = np.atleast_2d(np.asarray(pred, float))
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch {truth.shape} vs {pred.shape}")
    scale = np.sqrt(np.mean(truth ** 2))
    err = np.sqrt(np.mean((truth - pred) ** 2, axis=1)) / max(scale, 1e-30)
    over = np.nonzero(err > threshold)[0]
    steps = int(over[0]) if len(over) else len(err)
    out = {"steps": steps, "time": steps * dt, "threshold": threshold,
           "error_curve": err.tolist(), "censored": len(over) == 0}
    if lyapunov_exponent and lyapunov_exponent > 0:
        out["lyapunov_times"] = steps * dt * lyapunov_exponent
    return out
