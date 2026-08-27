"""Synthetic data generators with known ground truth.

    from standarderror.dynamics import ode, pde, sde, lyapunov
    traj = ode.lorenz63(n_steps=30000, dt=0.01)
    lam  = lyapunov.lyapunov_from_jacobian(
        lyapunov.lorenz_jacobian(), traj.x[:5000], traj.dt)

These exist so model claims can be falsified. Market data has no ground truth;
Lorenz-63 does.
"""

from . import lyapunov, ode, pde, sde

__all__ = ["ode", "pde", "sde", "lyapunov"]
