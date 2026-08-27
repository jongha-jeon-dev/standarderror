"""Model tests: correctness properties, not accuracy targets.

The important ones are the *ordering* tests — a model must beat persistence on a
chaotic system and must not beat it on white noise. The second is the one that
catches leakage, and it is the test most missing from published code.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.dynamics import lyapunov, ode
from standarderror.models import ESN, NGRC, ESNConfig, NGRCConfig, baselines, metrics
from standarderror.models.tune import rolling_origin


@pytest.fixture(scope="module")
def lorenz():
    traj = ode.lorenz63(n_steps=12000, dt=0.02, transient=40.0)
    mu, sd = traj.x[:8000].mean(0), traj.x[:8000].std(0)
    return {"z": (traj.x - mu) / sd, "dt": traj.dt, "lam": 0.9056}


class TestESN:
    def test_spectral_radius_is_actually_set(self):
        for rho in (0.4, 0.9, 1.3):
            m = ESN(ESNConfig(n_reservoir=250, spectral_radius=rho,
                              sparsity=0.05, seed=0))
            m._build(1)
            got = m.train_diagnostics["actual_spectral_radius"]
            assert got == pytest.approx(rho, rel=1e-6)

    def test_beats_persistence_on_lorenz(self, lorenz):
        z, dt, lam = lorenz["z"], lorenz["dt"], lorenz["lam"]
        train = z[:8000]
        esn = ESN(ESNConfig(n_reservoir=400, spectral_radius=0.9, sparsity=0.03,
                            input_scaling=0.6, ridge=1e-7, washout=250,
                            quadratic_features=True, seed=1))
        esn.fit(train[:-1], train[1:])
        warm, truth = z[7700:8000], z[8000:8800]
        vpt = lyapunov.valid_prediction_time(
            truth, esn.predict_autonomous(warm, 800), dt,
            lyapunov_exponent=lam)["lyapunov_times"]
        base = lyapunov.valid_prediction_time(
            truth, baselines.Persistence().predict_autonomous(warm, 800), dt,
            lyapunov_exponent=lam)["lyapunov_times"]
        assert vpt > 3.0
        assert vpt > 20 * base

    def test_does_not_beat_persistence_on_white_noise(self):
        """The leakage canary. On i.i.d. noise the best possible one-step
        forecast is the mean, so an ESN must NOT beat it. If this ever fails,
        something is feeding the target into the features."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((6000, 1))
        esn = ESN(ESNConfig(n_reservoir=300, washout=200, ridge=1e-4, seed=0))
        esn.fit(x[:4000][:-1], x[:4000][1:])
        pred = esn.predict_teacher_forced(x[4000:-1])
        truth = x[4001:]
        n = min(len(pred), len(truth))
        assert metrics.rmse(truth[-n:], pred[-n:]) > 0.9   # sd of the noise is 1

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            ESN().predict_teacher_forced(np.zeros((10, 1)))

    def test_autonomous_requires_matching_dimensions(self, lorenz):
        z = lorenz["z"]
        esn = ESN(ESNConfig(n_reservoir=200, washout=100))
        esn.fit(z[:2000, :3][:-1], z[1:2000, :1])     # 3 in, 1 out
        with pytest.raises(ValueError):
            esn.predict_autonomous(z[:200, :3], 10)

    def test_short_series_raises_rather_than_producing_garbage(self):
        esn = ESN(ESNConfig(washout=200))
        with pytest.raises(ValueError):
            esn.fit(np.zeros((100, 1)), np.zeros((100, 1)))

    def test_deterministic_given_seed(self, lorenz):
        z = lorenz["z"][:3000]
        cfg = ESNConfig(n_reservoir=200, washout=150, seed=7)
        a = ESN(cfg).fit(z[:-1], z[1:]).predict_autonomous(z[-200:], 50)
        b = ESN(cfg).fit(z[:-1], z[1:]).predict_autonomous(z[-200:], 50)
        assert np.allclose(a, b)


class TestNGRC:
    def test_feature_count_formula_matches_reality(self):
        for cfg in (NGRCConfig(n_lags=2, degree=2), NGRCConfig(n_lags=3, degree=2),
                    NGRCConfig(n_lags=2, degree=3)):
            m = NGRC(cfg)
            X = m.features(np.zeros((50, 3)))
            assert X.shape[1] == m.n_features(3)
            assert len(m.feature_names) == X.shape[1]

    def test_recovers_lorenz_structure(self, lorenz):
        """The strong test: fitted to the derivative, the readout must reproduce
        the analytic right-hand side, including which equations are linear."""
        z, dt = lorenz["z"], lorenz["dt"]
        U = z[1:-1]
        Y = (z[2:] - z[:-2]) / (2 * dt)
        m = NGRC(NGRCConfig(n_lags=1, degree=2, ridge=1e-10,
                            standardise=False)).fit(U, Y)
        names = m.feature_names
        W = m.W_out

        def coef(term, out):
            return W[names.index(term), out]

        # dx/dt = sigma (y - x) has NO quadratic terms.
        assert abs(coef("x0[t-0]", 0)) > 5.0
        assert abs(coef("x1[t-0]", 0)) > 5.0
        for q in ("x0[t-0]*x1[t-0]", "x1[t-0]*x1[t-0]", "x2[t-0]*x2[t-0]"):
            assert abs(coef(q, 0)) < 0.5
        # dy/dt = x(rho - z) - y contains xz.
        assert abs(coef("x0[t-0]*x2[t-0]", 1)) > 3.0
        # dz/dt = xy - beta z contains xy, and the z coefficient is -8/3.
        assert abs(coef("x0[t-0]*x1[t-0]", 2)) > 3.0
        assert coef("x2[t-0]", 2) == pytest.approx(-8.0 / 3.0, rel=0.2)

    def test_top_terms_are_sorted_by_magnitude(self, lorenz):
        z = lorenz["z"][:4000]
        m = NGRC(NGRCConfig(n_lags=2, degree=2)).fit(z[:-1], z[1:])
        vals = [abs(v) for _, v in m.top_terms(0, 8)]
        assert vals == sorted(vals, reverse=True)

    def test_autonomous_rollout_needs_enough_warmup(self, lorenz):
        z = lorenz["z"][:3000]
        m = NGRC(NGRCConfig(n_lags=4, stride=2)).fit(z[:-1], z[1:])
        with pytest.raises(ValueError):
            m.predict_autonomous(z[:3], 10)


class TestBaselines:
    def test_persistence_is_exactly_the_last_value(self):
        x = np.arange(20, dtype=float)[:, None]
        p = baselines.Persistence()
        assert np.allclose(p.predict_teacher_forced(x), x)
        assert np.allclose(p.predict_autonomous(x, 5), np.full((5, 1), 19.0))

    def test_linear_ar_recovers_a_linear_process(self):
        rng = np.random.default_rng(0)
        n = 5000
        x = np.zeros(n)
        for t in range(2, n):
            x[t] = 0.6 * x[t - 1] - 0.3 * x[t - 2] + 0.05 * rng.standard_normal()
        X = x[:, None]
        m = baselines.LinearAR(n_lags=2, ridge=1e-10).fit(X[:-1], X[1:])
        # W rows: [const, lag-1 (older), lag-0 (newest)]
        assert m.W[2, 0] == pytest.approx(0.6, abs=0.05)
        assert m.W[1, 0] == pytest.approx(-0.3, abs=0.05)

    def test_all_baselines_share_the_interface(self, lorenz):
        z = lorenz["z"][:3000]
        for name, cls in baselines.ALL.items():
            m = cls()
            m.fit(z[:-1], z[1:])
            tf = m.predict_teacher_forced(z[:-1])
            au = m.predict_autonomous(z[-50:], 20)
            assert tf.ndim == 2 and au.shape == (20, 3), name


class TestMetrics:
    def test_mase_is_one_for_the_naive_forecast(self):
        rng = np.random.default_rng(0)
        y_train = np.cumsum(rng.standard_normal(500))
        y = np.cumsum(rng.standard_normal(200))
        naive = np.concatenate([[y_train[-1]], y[:-1]])
        assert metrics.mase(y, naive, y_train) == pytest.approx(1.0, rel=0.25)

    def test_dm_test_favours_the_better_forecast(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(500)
        good = y + 0.1 * rng.standard_normal(500)
        bad = y + 1.0 * rng.standard_normal(500)
        out = metrics.dm_test(y, good, bad)
        assert out["favours"] == "model_1"
        assert out["p_value"] < 0.01

    def test_dm_test_finds_no_difference_between_equals(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal(800)
        a = y + 0.3 * rng.standard_normal(800)
        b = y + 0.3 * rng.standard_normal(800)
        assert metrics.dm_test(y, a, b)["p_value"] > 0.05

    def test_kupiec_accepts_a_correct_var_model(self):
        rng = np.random.default_rng(0)
        breaches = rng.random(2000) < 0.01
        assert metrics.kupiec_pof(breaches, 0.01)["p_value"] > 0.05

    def test_kupiec_rejects_a_broken_var_model(self):
        rng = np.random.default_rng(0)
        breaches = rng.random(2000) < 0.06     # claims 1%, breaches 6%
        assert metrics.kupiec_pof(breaches, 0.01)["p_value"] < 0.001

    def test_christoffersen_rejects_clustered_breaches(self):
        """Right count, wrong arrangement — the failure Kupiec cannot see."""
        x = np.zeros(1000, dtype=int)
        x[400:420] = 1                          # 20 breaches, all consecutive
        clustered = metrics.christoffersen_independence(x)
        assert clustered["p_value"] < 0.001
        assert metrics.kupiec_pof(x, 0.02)["p_value"] > 0.05   # Kupiec passes it

    def test_directional_accuracy_on_levels_needs_prev(self):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.5, 1.5, 3.5])
        prev = np.array([1.0, 2.0, 3.0])
        # Predicted change vs actual change; actual changes are all zero here so
        # everything is filtered out and the result is nan, not a fake number.
        assert np.isnan(metrics.directional_accuracy(y, p, prev))

    def test_pit_uniformity_flags_a_miscalibrated_model(self):
        rng = np.random.default_rng(0)
        good = rng.random(2000)
        bad = rng.beta(2, 5, 2000)
        assert metrics.pit_uniformity(good)["p_value"] > 0.05
        assert metrics.pit_uniformity(bad)["p_value"] < 0.001


class TestSplits:
    def test_rolling_origin_never_looks_forward(self):
        splits = rolling_origin(10000, n_folds=4, val_len=500, min_train=4000)
        assert len(splits) == 4
        for s in splits:
            assert s.train.stop <= s.val.start        # no overlap, no leakage
            assert s.val.stop <= 10000

    def test_rolling_origin_expands(self):
        splits = rolling_origin(10000, n_folds=3, val_len=500, min_train=4000)
        sizes = [s.train.stop for s in splits]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_impossible_configuration_raises(self):
        with pytest.raises(ValueError):
            rolling_origin(1000, n_folds=10, val_len=500, min_train=900)
