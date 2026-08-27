"""Tests for the conditional-correlation machinery.

The properties worth pinning here are not the arithmetic — that is three lines —
but the four claims the post rests on:

1. `conditional_rho` is the *exact* correlation a constant-correlation Gaussian
   pair shows inside a subsample selected on x, so a measured rise is not
   evidence of anything until it exceeds this.
2. `unconditional_rho` inverts it exactly, and is biased when the process has a
   common volatility path — which is the failure the post is about.
3. `scale_null` has the right size: given a genuinely constant correlation with
   realistic volatility clustering it reports no excess.
4. `scale_null` has power: given an injected dependence regime it recovers it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from standarderror.ts import conditional as cd


# --------------------------------------------------------------------------- #
class TestIdentity:
    def test_var_ratio_one_is_a_no_op(self):
        for rho in (-0.9, -0.2, 0.0, 0.35, 0.99):
            assert cd.conditional_rho(rho, 1.0) == pytest.approx(rho)

    def test_hand_computed(self):
        # rho = 0.5, var_ratio = 4 -> 0.5*2 / sqrt(0.25*4 + 0.75) = 1/sqrt(1.75)
        assert cd.conditional_rho(0.5, 4.0) == pytest.approx(1.0 / math.sqrt(1.75))

    def test_monotone_in_var_ratio(self):
        vals = [cd.conditional_rho(0.3, v) for v in (0.25, 0.5, 1, 2, 4, 8, 16)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_limits(self):
        assert cd.conditional_rho(0.3, 1e12) == pytest.approx(1.0, abs=1e-5)
        assert cd.conditional_rho(0.3, 1e-12) == pytest.approx(0.0, abs=1e-5)

    def test_sign_is_preserved(self):
        assert cd.conditional_rho(-0.4, 9.0) < -0.4
        assert cd.conditional_rho(0.4, 9.0) > 0.4

    def test_zero_correlation_stays_zero(self):
        # No amount of conditioning creates correlation where there is none.
        for v in (0.1, 1.0, 100.0):
            assert cd.conditional_rho(0.0, v) == pytest.approx(0.0)

    def test_round_trip(self):
        for rho in (-0.8, -0.1, 0.25, 0.6):
            for v in (0.3, 1.0, 2.5, 11.0):
                back = cd.unconditional_rho(cd.conditional_rho(rho, v), v)
                assert back == pytest.approx(rho, abs=1e-12)

    def test_forbes_rigobon_form(self):
        # rho = rho_A / sqrt(1 + delta (1 - rho_A^2)), delta = var_ratio - 1
        for rho_a, v in ((0.55, 6.0), (-0.7, 3.0), (0.2, 0.5)):
            delta = v - 1.0
            expected = rho_a / math.sqrt(1 + delta * (1 - rho_a ** 2))
            assert cd.unconditional_rho(rho_a, v) == pytest.approx(expected)

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            cd.conditional_rho(0.3, 0.0)
        with pytest.raises(ValueError):
            cd.conditional_rho(1.5, 2.0)
        with pytest.raises(ValueError):
            cd.unconditional_rho(0.3, -1.0)


class TestIdentityIsExactOnSimulatedData:
    """The claim that makes the post: no dependence change, yet the number moves."""

    @pytest.mark.parametrize("rho", [0.2, 0.3, 0.5, -0.65])
    def test_prediction_matches_measurement(self, rho):
        rng = np.random.default_rng(20260827)
        x, y = cd.gaussian_pair(300_000, rho, rng)
        r = cd.split_stats(x, y, cd.turbulent_mask(x, 0.90))
        assert r.excess == pytest.approx(0.0, abs=0.01)
        assert r.rho_corrected == pytest.approx(rho, abs=0.01)

    def test_the_rise_is_large_even_though_nothing_changed(self):
        rng = np.random.default_rng(1)
        x, y = cd.gaussian_pair(300_000, 0.30, rng)
        r = cd.split_stats(x, y, cd.turbulent_mask(x, 0.90))
        assert r.rho_calm < 0.26
        assert r.rho_turbulent > 0.52          # "correlations went to 0.55 in the crisis"
        assert r.rise > 0.28

    def test_a_common_volatility_path_biases_the_correction_low(self):
        # This is the second failure: FR is exact under x-only selection, and
        # once both series share a volatility path the conditioning inflates the
        # residual variance too, so the correction understates rho.
        rng = np.random.default_rng(7)
        s = cd.garch_scale(200_000, rng)
        x, y = cd.gaussian_pair(200_000, 0.30, rng, scale=s)
        r = cd.split_stats(x, y, cd.turbulent_mask(x, 0.90))
        assert r.rho_corrected < 0.27          # truth is 0.30
        assert r.excess < -0.05                # identity over-predicts

    def test_fat_tails_make_the_correction_worse_not_better(self):
        # Named against the intuition it corrects: the Student pair's *measured*
        # turbulent correlation is lower, not higher, and what grows is the
        # variance ratio — which is what drives the correction's bias.
        rng = np.random.default_rng(11)
        g_x, g_y = cd.gaussian_pair(200_000, 0.30, rng)
        t_x, t_y = cd.student_pair(200_000, 0.30, 4.0, rng)
        g = cd.split_stats(g_x, g_y, cd.turbulent_mask(g_x))
        t = cd.split_stats(t_x, t_y, cd.turbulent_mask(t_x))
        assert t.var_ratio_x > g.var_ratio_x
        assert t.rho_predicted > g.rho_predicted
        assert t.rho_corrected < g.rho_corrected < 0.30 + 0.01
        assert t.rho_turbulent < g.rho_turbulent


# --------------------------------------------------------------------------- #
class TestSampleStatistics:
    def test_pearson_matches_numpy(self):
        rng = np.random.default_rng(3)
        x, y = rng.standard_normal(500), rng.standard_normal(500)
        assert cd.pearson(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])

    def test_pearson_is_scale_and_shift_invariant(self):
        rng = np.random.default_rng(4)
        x, y = rng.standard_normal(400), rng.standard_normal(400)
        assert cd.pearson(x, y) == pytest.approx(cd.pearson(3 * x + 7, -0.5 * y - 2) * -1)

    def test_pearson_on_a_constant_is_nan_not_zero(self):
        assert math.isnan(cd.pearson(np.ones(50), np.arange(50.0)))

    def test_pearson_shape_mismatch(self):
        with pytest.raises(ValueError):
            cd.pearson(np.zeros(5), np.zeros(6))

    def test_variance_ratio_of_full_mask_is_one(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(1000)
        assert cd.variance_ratio(x, np.ones(1000, dtype=bool)) == pytest.approx(1.0)

    def test_turbulent_decile_variance_ratio_is_about_four(self):
        # Standard normal, two-sided top decile: E[x^2 | |x| > 1.645] ~= 4.4.
        rng = np.random.default_rng(6)
        x = rng.standard_normal(400_000)
        vr = cd.variance_ratio(x, cd.turbulent_mask(x, 0.90))
        assert 4.2 < vr < 4.6

    def test_turbulent_mask_size(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(10_000)
        assert cd.turbulent_mask(x, 0.90).sum() == pytest.approx(1000, rel=0.02)

    def test_turbulent_mask_rejects_bad_quantile(self):
        with pytest.raises(ValueError):
            cd.turbulent_mask(np.arange(10.0), 1.0)

    def test_date_mask(self):
        idx = pd.date_range("2008-09-01", periods=60, freq="D")
        m = cd.date_mask(idx, [("2008-09-15", "2008-09-20")])
        assert m.sum() == 6
        assert m[14] and m[19] and not m[20]


class TestSplitResult:
    def test_row_and_properties(self):
        rng = np.random.default_rng(8)
        x, y = cd.gaussian_pair(20_000, 0.4, rng)
        r = cd.split_stats(x, y, cd.turbulent_mask(x), label="sim")
        assert r.n_turbulent + r.n_calm == 20_000
        assert len(r.row()) == 7
        assert r.exact is True
        assert r.explained == pytest.approx(r.predicted_rise / r.rise)

    def test_explained_is_nan_when_there_is_no_rise(self):
        # A perfectly correlated pair has nothing left to inflate: rise is
        # exactly zero, and a share of zero is not a number worth printing.
        rng = np.random.default_rng(9)
        x = rng.standard_normal(5_000)
        r = cd.split_stats(x, x, cd.turbulent_mask(x))
        assert r.rise == pytest.approx(0.0, abs=1e-9)
        assert math.isnan(r.explained)

    def test_third_variable_conditioning_is_flagged_inexact(self):
        rng = np.random.default_rng(10)
        x, y = cd.gaussian_pair(5_000, 0.3, rng)
        z = rng.standard_normal(5_000)
        r = cd.split_stats(x, y, cd.turbulent_mask(z), conditioner="VIX")
        assert r.exact is False

    def test_quantile_sweep_is_monotone_in_the_prediction(self):
        rng = np.random.default_rng(12)
        x, y = cd.gaussian_pair(200_000, 0.35, rng)
        preds = [r.rho_predicted for r in cd.quantile_sweep(x, y)]
        assert all(b > a for a, b in zip(preds, preds[1:]))


class TestDevolatilise:
    def test_ewma_scale_uses_no_future_data(self):
        # Doubling one observation must not change the scale assigned to it.
        rng = np.random.default_rng(13)
        x = rng.standard_normal(1000)
        s1 = cd.ewma_scale(x)
        y = x.copy()
        y[600] *= 20
        s2 = cd.ewma_scale(y)
        assert s1[600] == pytest.approx(s2[600])
        assert s2[601] > s1[601]

    def test_ewma_scale_rejects_short_input(self):
        with pytest.raises(ValueError):
            cd.ewma_scale(np.zeros(10), warmup=250)

    def test_ewma_scale_rejects_bad_lambda(self):
        with pytest.raises(ValueError):
            cd.ewma_scale(np.zeros(500), lam=1.0)

    def test_devolatilise_removes_a_pure_scale_regime(self):
        rng = np.random.default_rng(14)
        n = 20_000
        s = np.where(np.arange(n) < n // 2, 1.0, 4.0)
        x, y = cd.gaussian_pair(n, 0.3, rng, scale=s)
        dx, dy, keep = cd.devolatilise(x, y)
        half = keep.sum() // 2
        # After standardising, the two halves have the same variance again.
        assert dx[:half].std() == pytest.approx(dx[half:].std(), rel=0.15)
        assert cd.pearson(dx, dy) == pytest.approx(0.3, abs=0.03)

    def test_centred_scale_does_use_future_data(self):
        # Stated as a test because it is the point of the control.
        x = np.zeros(200)
        x[100] = 10.0
        s = cd.centred_scale(x, 21)
        assert s[95] > s[50]

    def test_centred_scale_rejects_tiny_window(self):
        with pytest.raises(ValueError):
            cd.centred_scale(np.zeros(100), 2)


class TestDecomposition:
    def test_shares_sum_to_one(self):
        rng = np.random.default_rng(15)
        s = cd.garch_scale(30_000, rng)
        x, y = cd.gaussian_pair(30_000, 0.3, rng, scale=s)
        d = cd.covariance_decomposition(x, y, cd.turbulent_mask(x))
        assert d.share_rho + d.share_sx + d.share_sy == pytest.approx(1.0)

    def test_cov_ratio_is_the_product_of_the_three(self):
        rng = np.random.default_rng(16)
        x, y = cd.gaussian_pair(30_000, 0.4, rng)
        d = cd.covariance_decomposition(x, y, cd.turbulent_mask(x))
        assert d.cov_ratio == pytest.approx(d.rho_ratio * d.sx_ratio * d.sy_ratio)

    def test_a_pure_scale_regime_puts_no_share_on_correlation(self):
        rng = np.random.default_rng(17)
        n = 60_000
        s = np.where(rng.random(n) < 0.5, 1.0, 5.0)
        x, y = cd.gaussian_pair(n, 0.3, rng, scale=s)
        d = cd.covariance_decomposition(x, y, cd.turbulent_mask(x))
        assert d.share_scale > 0.6

    def test_frozen_rho_counterfactual_is_smaller_when_rho_rises(self):
        rng = np.random.default_rng(18)
        x, y = cd.gaussian_pair(40_000, 0.3, rng)
        d = cd.covariance_decomposition(x, y, cd.turbulent_mask(x))
        assert d.portfolio_rise > d.portfolio_rise_frozen_rho
        assert 0.0 < d.rho_contribution < 1.0


class TestBootstrap:
    def test_moving_block_indices_length_and_range(self):
        rng = np.random.default_rng(19)
        idx = cd.moving_block_indices(1000, 20, rng)
        assert idx.size == 1000
        assert idx.min() >= 0 and idx.max() < 1000

    def test_moving_block_indices_are_contiguous_within_a_block(self):
        rng = np.random.default_rng(20)
        idx = cd.moving_block_indices(100, 10, rng)
        assert np.all(np.diff(idx[:10]) == 1)

    def test_moving_block_rejects_bad_block(self):
        rng = np.random.default_rng(21)
        with pytest.raises(ValueError):
            cd.moving_block_indices(100, 0, rng)
        with pytest.raises(ValueError):
            cd.moving_block_indices(100, 101, rng)

    def test_interval_covers_the_point_estimate(self):
        rng = np.random.default_rng(22)
        x, y = cd.gaussian_pair(4000, 0.35, rng)
        point = cd.split_stats(x, y, cd.turbulent_mask(x))
        b = cd.bootstrap_split(x, y, block=20, n_boot=120, seed=2)
        assert b["rho_turbulent"]["lo"] < point.rho_turbulent < b["rho_turbulent"]["hi"]
        assert b["n_boot"] == 120


class TestScaleNull:
    """Size and power of the test the post actually uses."""

    def test_size_no_excess_when_the_correlation_is_constant(self):
        rng = np.random.default_rng(23)
        n = 8000
        s = cd.garch_scale(n, rng)
        x, y = cd.gaussian_pair(n, 0.35, rng, scale=s)
        t = cd.scale_null(x, y, reps=120, seed=3)
        assert abs(t.share_genuine) < 0.25
        assert t.p_value > 0.05
        assert t.null_turbulent_lo < t.rho_turbulent < t.null_turbulent_hi

    def test_power_recovers_an_injected_dependence_regime(self):
        rng = np.random.default_rng(24)
        n = 12_000
        s = cd.garch_scale(n, rng)
        hi = s > np.quantile(s, 0.80)
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        rho_t = np.where(hi, 0.75, 0.25)
        x = s * z1
        y = s * (rho_t * z1 + np.sqrt(1 - rho_t ** 2) * z2)
        t = cd.scale_null(x, y, reps=120, seed=4)
        assert t.genuine_excess > 0.05
        assert t.p_value < 0.05

    def test_the_null_reproduces_the_variance_ratio(self):
        rng = np.random.default_rng(25)
        n = 8000
        s = cd.garch_scale(n, rng)
        x, y = cd.gaussian_pair(n, 0.3, rng, scale=s)
        t = cd.scale_null(x, y, reps=80, seed=5)
        assert t.null_var_ratio == pytest.approx(t.var_ratio, rel=0.20)

    def test_centred_scale_variant_runs_and_agrees_roughly(self):
        rng = np.random.default_rng(26)
        n = 8000
        s = cd.garch_scale(n, rng)
        x, y = cd.gaussian_pair(n, 0.3, rng, scale=s)
        a = cd.scale_null(x, y, reps=80, seed=6, scale="ewma")
        b = cd.scale_null(x, y, reps=80, seed=6, scale="centred")
        assert abs(a.null_turbulent_mean - b.null_turbulent_mean) < 0.15
        assert b.scale_kind == "centred"

    def test_rejects_unknown_scale(self):
        rng = np.random.default_rng(27)
        x, y = cd.gaussian_pair(1000, 0.3, rng)
        with pytest.raises(ValueError):
            cd.scale_null(x, y, scale="garch", warmup=100, reps=2)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError):
            cd.scale_null(np.zeros(600), np.zeros(601), reps=2)


class TestGenerators:
    def test_gaussian_pair_correlation(self):
        rng = np.random.default_rng(28)
        x, y = cd.gaussian_pair(200_000, 0.45, rng)
        assert cd.pearson(x, y) == pytest.approx(0.45, abs=0.01)

    def test_a_common_scale_leaves_the_correlation_alone(self):
        rng = np.random.default_rng(29)
        n = 200_000
        s = cd.garch_scale(n, rng)
        x, y = cd.gaussian_pair(n, 0.45, rng, scale=s)
        assert cd.pearson(x, y) == pytest.approx(0.45, abs=0.02)

    def test_scale_length_is_checked(self):
        rng = np.random.default_rng(30)
        with pytest.raises(ValueError):
            cd.gaussian_pair(100, 0.3, rng, scale=np.ones(99))

    def test_student_pair_is_fat_tailed_but_equicorrelated(self):
        rng = np.random.default_rng(31)
        x, y = cd.student_pair(200_000, 0.4, 5.0, rng)
        c = x - x.mean()
        kurt = (c ** 4).mean() / c.var() ** 2 - 3.0
        assert kurt > 2.0
        assert cd.pearson(x, y) == pytest.approx(0.4, abs=0.02)

    def test_student_rejects_infinite_variance(self):
        rng = np.random.default_rng(32)
        with pytest.raises(ValueError):
            cd.student_pair(100, 0.3, 2.0, rng)

    def test_garch_scale_is_persistent_and_positive(self):
        rng = np.random.default_rng(33)
        s = cd.garch_scale(20_000, rng)
        assert (s > 0).all()
        assert cd.pearson(s[:-1], s[1:]) > 0.8

    def test_garch_rejects_nonstationary_parameters(self):
        rng = np.random.default_rng(34)
        with pytest.raises(ValueError):
            cd.garch_scale(100, rng, alpha=0.2, beta=0.85)
