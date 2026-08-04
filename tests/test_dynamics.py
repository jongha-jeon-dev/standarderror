"""Tests that check physics, not just that the code runs.

Every assertion here is against a value from the literature or an independent
integrator. A test that only checks array shapes would have passed while the KS
integrator was silently blowing up and the Lyapunov exponent was 20% too high —
both real bugs found by exactly these checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantpost.dynamics import lyapunov, ode, pde, sde


class TestLorenz:
    def test_lyapunov_exponent_matches_literature(self):
        traj = ode.lorenz63(n_steps=20000, dt=0.005, transient=40.0)
        res = lyapunov.lyapunov_from_jacobian(
            lyapunov.lorenz_jacobian(), traj.x, traj.dt)
        # Literature: lambda_max = 0.9056 (Viswanath 1998).
        assert res.exponent == pytest.approx(0.9056, abs=0.03)

    def test_lyapunov_spectrum_sums_to_trace(self):
        """sum(lambda_i) must equal the divergence of the vector field,
        -(sigma + 1 + beta) = -13.667 for Lorenz-63. This is an exact identity
        and catches any tangent-space propagation error."""
        traj = ode.lorenz63(n_steps=20000, dt=0.005, transient=40.0)
        res = lyapunov.lyapunov_from_jacobian(
            lyapunov.lorenz_jacobian(), traj.x, traj.dt)
        total = sum(res.detail["spectrum"])
        assert total == pytest.approx(-(10.0 + 1.0 + 8.0 / 3.0), rel=0.02)

    def test_kaplan_yorke_dimension(self):
        traj = ode.lorenz63(n_steps=20000, dt=0.005, transient=40.0)
        res = lyapunov.lyapunov_from_jacobian(
            lyapunov.lorenz_jacobian(), traj.x, traj.dt)
        assert res.detail["kaplan_yorke_dim"] == pytest.approx(2.062, abs=0.02)

    def test_attractor_is_bounded(self):
        traj = ode.lorenz63(n_steps=30000, dt=0.01)
        assert np.abs(traj.x).max() < 100.0
        assert np.isfinite(traj.x).all()

    def test_transient_is_actually_discarded(self):
        traj = ode.lorenz63(n_steps=1000, dt=0.01, transient=40.0)
        assert traj.t[0] == pytest.approx(0.0)
        assert len(traj.t) == 1000


class TestMackeyGlass:
    def test_bounded_and_chaotic_range(self):
        traj = ode.mackey_glass(n_steps=5000, tau=17.0)
        x = traj.x.ravel()
        assert np.isfinite(x).all()
        # The tau=17 attractor lives roughly in [0.2, 1.6].
        assert 0.1 < x.min() < 0.9
        assert 0.9 < x.max() < 2.0
        assert x.std() > 0.1          # not collapsed to a fixed point

    def test_rejects_impossible_delay(self):
        with pytest.raises(ValueError):
            ode.mackey_glass(n_steps=100, tau=0.5, dt=1.0)


class TestKuramotoSivashinsky:
    def test_long_integration_stays_bounded(self):
        """The regression test for the rfft bug: the old full-complex-spectrum
        version blew up at t ~ 355 for every parameter choice."""
        f = pde.kuramoto_sivashinsky(n_steps=4000, L=22.0, N=64, dt=0.25,
                                     transient=200.0)
        assert np.isfinite(f.u).all()
        assert np.abs(f.u).max() < 10.0
        # 4000 * 0.25 = 1000 time units, well past the old failure point.
        assert f.t[-1] > 900.0

    def test_energy_is_stationary(self):
        f = pde.kuramoto_sivashinsky(n_steps=6000, dt=0.25, transient=200.0)
        e = f.energy()
        first, last = e[:1500].mean(), e[-1500:].mean()
        assert first == pytest.approx(last, rel=0.35)   # no systematic drift

    def test_matches_implicit_reference(self):
        """ETDRK4 against Radau on the same semi-discrete system."""
        N, L, T = 64, 22.0, 8.0
        x = L * np.arange(N) / N
        u0 = np.cos(2 * np.pi * x / L) * (1 + np.sin(2 * np.pi * x / L))
        ref = pde.reference_solution(N, L, T, u0)
        got = pde.kuramoto_sivashinsky(n_steps=int(T / 0.05), L=L, N=N, dt=0.05,
                                       transient=0.0, u0=u0).u[-1]
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        assert rel < 1e-5

    def test_fourth_order_convergence(self):
        N, L, T = 64, 22.0, 6.0
        x = L * np.arange(N) / N
        u0 = np.cos(2 * np.pi * x / L) * (1 + np.sin(2 * np.pi * x / L))
        ref = pde.reference_solution(N, L, T, u0)
        errs = []
        for dt in (0.1, 0.05, 0.025):
            got = pde.kuramoto_sivashinsky(
                n_steps=int(round(T / dt)), L=L, N=N, dt=dt, transient=0.0,
                u0=u0).u[-1]
            errs.append(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        assert min(orders) > 3.0

    def test_rejects_wrong_shaped_ic(self):
        with pytest.raises(ValueError):
            pde.kuramoto_sivashinsky(n_steps=10, N=64, u0=np.zeros(32))


class TestBurgers:
    def test_decays_and_stays_bounded(self):
        f = pde.burgers(n_steps=2000, nu=0.02)
        assert np.isfinite(f.u).all()
        # Viscous Burgers with no forcing loses energy monotonically-ish.
        assert f.energy()[-1] < f.energy()[0]


class TestSDE:
    def test_ou_reverts_to_its_mean(self):
        p = sde.ornstein_uhlenbeck(n_steps=200000, dt=1 / 252, theta=3.0,
                                   mu=0.5, sigma=0.2)
        assert p.data["x"].mean() == pytest.approx(0.5, abs=0.02)
        # Stationary sd = sigma / sqrt(2 theta)
        assert p.data["x"].std() == pytest.approx(0.2 / np.sqrt(6.0), rel=0.1)

    def test_heston_variance_reverts_to_theta(self):
        p = sde.heston(n_steps=200000, dt=1 / 252, kappa=3.0, theta=0.04,
                       xi=0.3)
        assert p.data["variance"].mean() == pytest.approx(0.04, rel=0.2)
        assert (p.data["variance"] >= 0).all()      # full truncation guarantee
        assert p.params["feller_satisfied"] is True

    def test_heston_leverage_effect_has_right_sign(self):
        p = sde.heston(n_steps=100000, rho=-0.7, xi=0.5)
        r = p.data["log_return"][1:]
        dv = np.diff(p.data["variance"])
        assert np.corrcoef(r, dv)[0, 1] < -0.2     # returns down, vol up

    def test_fbm_hurst_recovered_by_variance_scaling(self):
        """Var of the aggregated increment scales like n^(2H)."""
        for H in (0.3, 0.7):
            inc = sde.fbm(4096, H, seed=1)
            sds = []
            for n in (1, 4, 16, 64):
                agg = inc[: (len(inc) // n) * n].reshape(-1, n).sum(axis=1)
                sds.append(agg.std())
            slope = np.polyfit(np.log([1, 4, 16, 64]), np.log(sds), 1)[0]
            assert slope == pytest.approx(H, abs=0.12)

    def test_hawkes_rejects_nonstationary_branching(self):
        with pytest.raises(ValueError):
            sde.hawkes_exp(horizon=10.0, mu=0.5, alpha=1.2, beta=1.0)

    def test_hawkes_intensity_matches_theory(self):
        """Stationary rate of a Hawkes process is mu / (1 - alpha/beta)."""
        p = sde.hawkes_exp(horizon=4000.0, mu=0.4, alpha=0.5, beta=1.0, seed=2)
        expected = 0.4 / (1 - 0.5)
        observed = p.params["n_events"] / 4000.0
        assert observed == pytest.approx(expected, rel=0.15)


class TestValidPredictionTime:
    def test_perfect_forecast_is_censored(self):
        truth = np.random.default_rng(0).standard_normal((100, 3))
        out = lyapunov.valid_prediction_time(truth, truth, 0.1)
        assert out["steps"] == 100
        assert out["censored"] is True

    def test_detects_first_crossing(self):
        truth = np.ones((50, 1))
        pred = truth.copy()
        pred[10:] = 5.0                     # error 4.0, normalised 4.0 > 0.3
        out = lyapunov.valid_prediction_time(truth, pred, 0.1, threshold=0.3)
        assert out["steps"] == 10

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            lyapunov.valid_prediction_time(np.zeros((10, 2)), np.zeros((10, 3)),
                                          0.1)

    def test_rosenstein_recovers_lorenz_exponent(self):
        traj = ode.lorenz63(n_steps=12000, dt=0.02, transient=40.0)
        res = lyapunov.lyapunov_rosenstein(traj.x[:, 0], traj.dt, emb_dim=6,
                                          max_t=80, fit_window=(5, 40))
        # A data-driven estimator on one coordinate: loose tolerance is honest.
        assert 0.5 < res.exponent < 1.4
        assert res.detail["fit_r2"] > 0.9
