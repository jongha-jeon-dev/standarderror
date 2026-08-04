"""Deterministic dynamical systems: the honest test bed.

Why generate data at all when the point is finance? Because on market data you
cannot tell a model that learned dynamics from a model that learned to echo the
last value — the signal-to-noise ratio is too low and there is no ground truth.
On Lorenz-63 there is: you know the attractor, you know the Lyapunov time, and
"predicts 6 Lyapunov times ahead" is a claim that either holds or does not.
Calibrate the method here, then take it to the messy data.

All generators return a `Trajectory` with the state array, the time grid, the
integration settings and the discarded transient length, so a figure caption can
state exactly what was simulated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Trajectory:
    t: np.ndarray                 # (T,)
    x: np.ndarray                 # (T, d)
    names: list[str]
    dt: float
    system: str
    params: dict = field(default_factory=dict)
    transient_discarded: float = 0.0

    @property
    def dim(self) -> int:
        return self.x.shape[1]

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.x, index=self.t, columns=self.names)

    def describe(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return (f"{self.system} ({p}), dt={self.dt}, "
                f"{len(self.t)} steps, {self.transient_discarded:g} time units "
                "of transient discarded")


def _integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    dt: float,
    n_steps: int,
    transient: float,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate with tight tolerances, then drop the transient.

    Tolerances matter more than they look: with the scipy defaults a Lorenz
    trajectory drifts off the true orbit within a few Lyapunov times, and you
    end up benchmarking your forecaster against integration error.
    """
    n_trans = int(round(transient / dt))
    total = n_steps + n_trans
    t_eval = np.arange(total) * dt
    sol = solve_ivp(rhs, (0.0, t_eval[-1]), x0, t_eval=t_eval,
                    method="RK45", rtol=rtol, atol=atol, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    x = sol.y.T[n_trans:]
    t = t_eval[n_trans:] - t_eval[n_trans]
    return t, x


def lorenz63(
    n_steps: int = 20000,
    dt: float = 0.01,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: tuple[float, float, float] = (1.0, 1.0, 1.0),
    transient: float = 40.0,
) -> Trajectory:
    """Lorenz-63. Largest Lyapunov exponent ~0.906 at the classic parameters,
    so one Lyapunov time ~1.10 time units."""
    def rhs(_t, s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    t, x = _integrate(rhs, np.asarray(x0, float), dt, n_steps, transient)
    return Trajectory(t, x, ["x", "y", "z"], dt, "Lorenz-63",
                      {"sigma": sigma, "rho": rho, "beta": round(beta, 4)},
                      transient)


def rossler(
    n_steps: int = 20000,
    dt: float = 0.05,
    *,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    x0: tuple[float, float, float] = (1.0, 1.0, 1.0),
    transient: float = 100.0,
) -> Trajectory:
    """Rössler. Slower, more 'banded' chaos than Lorenz — a useful second test
    because a forecaster tuned to Lorenz's timescale often fails here."""
    def rhs(_t, s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    t, x = _integrate(rhs, np.asarray(x0, float), dt, n_steps, transient)
    return Trajectory(t, x, ["x", "y", "z"], dt, "Rössler",
                      {"a": a, "b": b, "c": c}, transient)


def duffing(
    n_steps: int = 20000,
    dt: float = 0.02,
    *,
    delta: float = 0.3,
    alpha: float = -1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    omega: float = 1.2,
    x0: tuple[float, float] = (0.1, 0.0),
    transient: float = 100.0,
) -> Trajectory:
    """Forced Duffing oscillator — non-autonomous, so a model must infer the
    drive. Good for asking whether a reservoir has learned phase."""
    def rhs(t, s):
        x, v = s
        return np.array([v,
                         -delta * v - alpha * x - beta * x ** 3
                         + gamma * np.cos(omega * t)])

    t, x = _integrate(rhs, np.asarray(x0, float), dt, n_steps, transient)
    return Trajectory(t, x, ["x", "v"], dt, "Duffing",
                      {"delta": delta, "alpha": alpha, "beta": beta,
                       "gamma": gamma, "omega": omega}, transient)


def mackey_glass(
    n_steps: int = 20000,
    dt: float = 1.0,
    *,
    tau: float = 17.0,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    x0: float = 1.2,
    transient: float = 1000.0,
    seed: int | None = None,
) -> Trajectory:
    """Mackey-Glass delay equation, the classic reservoir-computing benchmark.

    Integrated with a fixed-step RK4 over a ring buffer of history rather than
    scipy, because a DDE needs the delayed state and `solve_ivp` has no notion of
    one. `tau=17` gives a mildly chaotic attractor; `tau=30` a much harder one.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    lag = int(round(tau / dt))
    if lag < 1:
        raise ValueError("tau/dt must be at least 1 step")

    n_trans = int(round(transient / dt))
    total = n_steps + n_trans
    hist_len = lag + 1
    rng = np.random.default_rng(seed)
    # Seed the history with a small jitter around x0 so the initial condition is
    # not an exact fixed point of the map.
    hist = x0 + 0.01 * rng.standard_normal(hist_len)
    buf = np.empty(total + hist_len)
    buf[:hist_len] = hist

    def f(x_now: float, x_lag: float) -> float:
        return beta * x_lag / (1.0 + x_lag ** n) - gamma * x_now

    for i in range(hist_len, total + hist_len):
        x_now = buf[i - 1]
        x_lag = buf[i - 1 - lag]
        # RK4 holding the delayed term fixed across the step (standard for
        # dt << tau; here dt=1, tau=17).
        k1 = f(x_now, x_lag)
        k2 = f(x_now + 0.5 * dt * k1, x_lag)
        k3 = f(x_now + 0.5 * dt * k2, x_lag)
        k4 = f(x_now + dt * k3, x_lag)
        buf[i] = x_now + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    series = buf[hist_len + n_trans:]
    t = np.arange(len(series)) * dt
    return Trajectory(t, series[:, None], ["x"], dt, "Mackey-Glass",
                      {"tau": tau, "beta": beta, "gamma": gamma, "n": n},
                      transient)
