"""Tests for quantpost.ts.detect.

The load-bearing tests are the size checks: a test that is not correctly sized
makes every power number downstream meaningless, and the whole point of the
module is that the textbook correction is *not* correctly sized on persistent
data. Those checks are slow because size can only be measured by simulation, and
they are worth it.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantpost.ts import detect as dt


def ar1(n, rho, rng, sigma=1.0):
    x = np.zeros(n)
    e = rng.normal(scale=sigma, size=n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------

class TestDesignMatrix:
    @pytest.mark.parametrize("trend,extra", [(False, 0), (True, 1), (2, 2), (3, 3)])
    def test_trend_degree_adds_one_column_each(self, trend, extra):
        assert dt.design_matrix(40, trend=trend).shape == (40, 1 + extra)

    def test_seasonal_adds_period_minus_one_columns(self):
        assert dt.design_matrix(60, trend=False, seasonal=12).shape == (60, 12)

    def test_seasonal_dummies_respect_the_start_period(self):
        """A series starting in August must not be told it starts in January."""
        jan = dt.design_matrix(24, trend=False, seasonal=12, start_period=0)
        aug = dt.design_matrix(24, trend=False, seasonal=12, start_period=7)
        assert not np.array_equal(jan, aug)
        # whatever the phase, each dummy fires exactly once per twelve rows and
        # repeats with period twelve
        for X in (jan, aug):
            for k in range(1, 12):
                col = X[:, k]
                assert col[:12].sum() == 1.0
                assert np.array_equal(col[:12], col[12:24])
        # and the row that is the reference (dummy-free) differs between them
        ref_jan = int(np.argmin(jan[:12, 1:].sum(axis=1)))
        ref_aug = int(np.argmin(aug[:12, 1:].sum(axis=1)))
        assert ref_jan != ref_aug

    def test_step_is_the_last_column_and_is_a_step(self):
        X = dt.design_matrix(20, break_at=8, trend=True, seasonal=4)
        step = X[:, -1]
        assert step[:8].sum() == 0
        assert (step[8:] == 1).all()

    def test_rejects_a_break_at_the_edges(self):
        for tau in (0, 20):
            with pytest.raises(ValueError, match="empty"):
                dt.design_matrix(20, break_at=tau)

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError, match="at least two"):
            dt.design_matrix(1)
        with pytest.raises(ValueError, match="seasonal period"):
            dt.design_matrix(20, seasonal=1)
        with pytest.raises(ValueError, match="non-negative"):
            dt.design_matrix(20, trend=-1)


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------

class TestOLSHac:
    def test_recovers_a_planted_step(self):
        rng = np.random.default_rng(1)
        n, tau, delta = 200, 100, 0.75
        y = rng.normal(scale=0.1, size=n)
        y[tau:] += delta
        X = dt.design_matrix(n, break_at=tau, trend=False)
        fit = dt.ols_hac(X, y, lags=6)
        assert fit.beta[-1] == pytest.approx(delta, abs=0.03)

    def test_hac_at_zero_lags_is_the_white_estimator(self):
        """Not the same as the OLS standard error, and it should not be.

        At zero lags the sandwich reduces to White's heteroskedasticity-robust
        form, which agrees with the homoskedastic formula only in expectation.
        Asserting exact equality here would be asserting something false; what
        is true is that they agree closely on homoskedastic data and diverge
        when the variance moves.
        """
        rng = np.random.default_rng(2)
        n = 4000
        homo = dt.ols_hac(dt.design_matrix(n, break_at=n // 2),
                          rng.normal(size=n), lags=0)
        assert homo.se_hac == pytest.approx(homo.se_ols, rel=0.08)

        # The break has to be off-centre. For a two-group mean difference with
        # *equal* group sizes the pooled and White formulas coincide exactly
        # whatever the variances are, which is a large part of why the naive
        # standard error survives so many textbook examples unscathed. Put the
        # noisy group in the short segment and they separate.
        tau = n // 5
        scale = np.where(np.arange(n) < tau, 8.0, 1.0)
        hetero = dt.ols_hac(dt.design_matrix(n, break_at=tau),
                            rng.normal(size=n) * scale, lags=0)
        assert hetero.inflation[-1] > 1.4, hetero.inflation

    def test_hac_exceeds_ols_on_persistent_residuals(self):
        rng = np.random.default_rng(3)
        n = 300
        y = ar1(n, 0.9, rng)
        X = dt.design_matrix(n, break_at=150)
        fit = dt.ols_hac(X, y, lags=24)
        assert fit.inflation[-1] > 1.8

    def test_matches_statsmodels_hac(self):
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(4)
        n = 200
        y = ar1(n, 0.7, rng)
        X = dt.design_matrix(n, break_at=100, trend=True)
        mine = dt.ols_hac(X, y, lags=12)
        theirs = sm.OLS(y, X).fit(cov_type="HAC",
                                  cov_kwds={"maxlags": 12, "use_correction": True})
        assert mine.beta == pytest.approx(theirs.params, rel=1e-9)
        assert mine.se_hac == pytest.approx(theirs.bse, rel=1e-6)

    def test_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match="does not match"):
            dt.ols_hac(np.zeros((10, 2)), np.zeros(9))
        with pytest.raises(ValueError, match="cannot support"):
            dt.ols_hac(np.zeros((3, 5)), np.zeros(3))
        with pytest.raises(ValueError, match="non-negative"):
            dt.ols_hac(np.zeros((10, 2)), np.zeros(10), lags=-1)


# ---------------------------------------------------------------------------
# information content
# ---------------------------------------------------------------------------

class TestEffectiveSampleSize:
    def test_white_noise_is_worth_about_its_length(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=4000)
        assert dt.effective_sample_size(x, 24) == pytest.approx(4000, rel=0.25)

    def test_persistence_costs_observations(self):
        rng = np.random.default_rng(6)
        prev = None
        for rho in (0.0, 0.5, 0.8, 0.95):
            n_eff = dt.effective_sample_size(ar1(4000, rho, rng), 60)
            if prev is not None:
                assert n_eff < prev
            prev = n_eff

    def test_matches_the_ar1_closed_form(self):
        """For AR(1), variance inflation tends to (1+rho)/(1-rho)."""
        rng = np.random.default_rng(7)
        rho = 0.8
        vif = dt.variance_inflation(ar1(20000, rho, rng), 200)
        assert vif == pytest.approx((1 + rho) / (1 - rho), rel=0.20)

    def test_constant_series_raises_rather_than_returning_nan(self):
        with pytest.raises(ValueError, match="constant"):
            dt.autocorrelation(np.ones(50), 5)

    def test_max_lag_must_fit(self):
        with pytest.raises(ValueError, match="max_lag"):
            dt.autocorrelation(np.arange(10.0), 10)


class TestBlockBootstrap:
    def test_returns_the_requested_length(self):
        rng = np.random.default_rng(8)
        out = dt.moving_block_bootstrap(np.arange(100.0), block=7, size=53,
                                        rng=rng)
        assert out.size == 53

    def test_preserves_persistence_far_better_than_shuffling(self):
        rng = np.random.default_rng(9)
        x = ar1(2000, 0.9, rng)
        boot = dt.moving_block_bootstrap(x, block=100, size=2000, rng=rng)
        shuffled = rng.permutation(x)
        assert dt.variance_inflation(boot, 60) > 4.0
        assert dt.variance_inflation(shuffled, 60) < 2.0

    def test_rejects_an_impossible_block(self):
        rng = np.random.default_rng(10)
        for block in (0, 101):
            with pytest.raises(ValueError, match="block"):
                dt.moving_block_bootstrap(np.zeros(100), block=block, size=10,
                                          rng=rng)


# ---------------------------------------------------------------------------
# the claims the post makes
# ---------------------------------------------------------------------------

class TestSize:
    """Slow, and the reason the rest of the module can be believed."""

    @staticmethod
    def _size(rho, lags, reps=400, n=300, seed=11, critical=1.959963984540054,
              which="t_hac"):
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(reps):
            y = rng.normal(size=n) if rho == 0 else ar1(n, rho, rng)
            hits += abs(dt.break_test(y, n // 2, lags=lags)[which]) > critical
        return hits / reps

    def test_naive_standard_error_over_rejects_badly_on_persistent_data(self):
        size = self._size(0.9, lags=24, which="t_ols")
        assert size > 0.40, size

    def test_hac_helps_a_lot(self):
        naive = self._size(0.9, lags=24, which="t_ols")
        hac = self._size(0.9, lags=24, which="t_hac")
        assert hac < naive / 2

    def test_hac_is_still_not_correctly_sized(self):
        """The finding that motivates calibrating the critical value."""
        size = self._size(0.9, lags=24, which="t_hac")
        assert size > 0.12, (
            f"HAC size on AR(0.9) is {size:.1%}; if this has fallen to nominal, "
            "the post's argument for a calibrated critical value is gone")

    def test_no_bandwidth_fixes_both_ends(self):
        """Short bandwidth fails on persistent data, long fails on quiet data."""
        short_on_persistent = self._size(0.9, lags=0, which="t_hac")
        long_on_white = self._size(0.0, lags=48, which="t_hac")
        assert short_on_persistent > 0.40
        assert long_on_white > 0.12

    def test_calibration_helps_a_lot_and_is_itself_noisy(self):
        """The honest version, and the reason the post reports a range.

        Calibrating on one series' residuals inherits that series' sampling
        error. Averaged over training paths the size lands near 8% against a
        nominal 5% — far better than the ~20% that 1.96 delivers, and not the
        exact answer. Asserting 5% here would be asserting something the method
        does not achieve.
        """
        rng = np.random.default_rng(42)
        n = 300
        cvs, sizes = [], []
        for k in range(6):
            train = ar1(n, 0.9, rng)
            r = dt.ols_hac(dt.design_matrix(n, trend=True), train,
                           lags=24).resid
            cv = dt.calibrated_critical_value(
                r, n_pre=n // 2, n_post=n // 2, block=36, lags=24, reps=500,
                rng=np.random.default_rng(1000 + k))
            hits = sum(abs(dt.break_test(ar1(n, 0.9, rng), n // 2,
                                         lags=24)["t_hac"]) > cv["critical"]
                       for _ in range(200))
            cvs.append(cv["critical"])
            sizes.append(hits / 200)
        cvs, sizes = np.array(cvs), np.array(sizes)
        assert cvs.mean() > 2.5, cvs
        assert cvs.std() > 0.10, "the spread across training paths is the point"
        assert 0.03 < sizes.mean() < 0.16, sizes
        assert sizes.mean() < 0.18, "still far better than 1.96 would give"

    def test_calibration_rejects_an_unknown_statistic(self):
        with pytest.raises(ValueError, match="must be"):
            dt.calibrated_critical_value(np.zeros(50), n_pre=25, n_post=25,
                                         block=5, statistic="wald")


class TestPlaceboScan:
    def test_a_cyclical_series_looks_broken_almost_everywhere(self):
        """The post's opening claim, on data with no break in it at all."""
        rng = np.random.default_rng(14)
        n = 300
        t = np.arange(n)
        y = 1.5 * np.sin(2 * np.pi * t / 70) + ar1(n, 0.7, rng, sigma=0.4)
        sc = dt.placebo_scan(y, trend=2, lags=24)
        assert sc["share_ols"] > 0.35, sc["share_ols"]
        assert sc["share_hac"] < sc["share_ols"] / 2

    def test_trim_leaves_a_usable_grid(self):
        rng = np.random.default_rng(15)
        sc = dt.placebo_scan(rng.normal(size=200), trim=0.15)
        assert sc["n_dates"] == pytest.approx(140, abs=4)

    def test_rejects_a_trim_that_empties_the_grid(self):
        with pytest.raises(ValueError, match="no candidate"):
            dt.placebo_scan(np.zeros(20), trim=0.5)


class TestPower:
    def test_power_rises_with_the_shift(self):
        rng = np.random.default_rng(16)
        r = ar1(300, 0.8, rng)
        prev = -1.0
        for shift in (0.0, 0.5, 1.0, 2.0):
            p = dt.detection_power(r, n_pre=200, n_post=100, shift=shift,
                                   block=24, reps=250, lags=12,
                                   rng=np.random.default_rng(17))["power_hac"]
            assert p >= prev - 0.02
            prev = p
        assert prev > 0.5

    def test_minimum_detectable_shift_reaches_its_target(self):
        rng = np.random.default_rng(18)
        r = ar1(300, 0.8, rng)
        m = dt.minimum_detectable_shift(r, n_pre=200, n_post=100, block=24,
                                        target=0.80, reps=300, hi=4.0,
                                        tol=0.05, lags=12)
        got = dt.detection_power(r, n_pre=200, n_post=100, shift=m["mde"],
                                 block=24, reps=600, lags=12,
                                 rng=np.random.default_rng(19))["power_hac"]
        assert got == pytest.approx(0.80, abs=0.10), (m, got)

    def test_reports_infinity_rather_than_lying_when_nothing_is_detectable(self):
        rng = np.random.default_rng(20)
        r = ar1(300, 0.98, rng, sigma=10.0)
        m = dt.minimum_detectable_shift(r, n_pre=280, n_post=20, block=60,
                                        target=0.80, reps=150, hi=0.01,
                                        lags=24)
        assert m["mde"] == float("inf")

    def test_rejects_an_unknown_target_statistic(self):
        with pytest.raises(ValueError, match="'hac' or 'ols'"):
            dt.minimum_detectable_shift(np.zeros(50), n_pre=25, n_post=25,
                                        block=5, use="bayes")
