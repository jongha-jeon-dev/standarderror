"""Tests for robust scale, contamination construction, and the equivariance test.

The one that matters most is `TestScaleSweepOnRealBoosting`: it asserts that
squared-error boosting is scale equivariant and that pseudo-Huber with a fixed slope
is not. That is the claim `experiments/exp014` is built on, so if the library drifts
the post's central number should stop being reproducible and a test should say so.

`TestLeverageDistanceIsNotMonotone` pins the other finding, and it exists because the
first version of that experiment concluded the opposite from a construction that put
its leverage points too far out to do any damage.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.robust import contamination, equivariance, scale


def truth(A):
    return 3.0 * np.sin(A[:, 0]) + A[:, 1] ** 2 - 2.0 * A[:, 2]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1200, 5))
    y = truth(X) + rng.standard_normal(1200)
    Xt = rng.standard_normal((400, 5))
    return X, y, Xt, truth(Xt)


class TestMadScale:
    def test_the_constant_is_what_it_claims(self):
        assert scale.consistency_factor() == pytest.approx(scale.MAD_TO_SIGMA,
                                                           rel=1e-9)

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 7.0])
    def test_consistent_at_the_gaussian(self, sigma):
        z = np.random.default_rng(0).standard_normal(100_000) * sigma
        assert scale.mad_scale(z) == pytest.approx(sigma, rel=0.02)

    def test_raw_mad_is_smaller_than_sigma(self):
        z = np.random.default_rng(0).standard_normal(50_000)
        raw = scale.mad_scale(z, consistent=False)
        assert raw == pytest.approx(scale.mad_scale(z) / scale.MAD_TO_SIGMA, rel=1e-9)
        assert raw < 0.8

    def test_survives_forty_percent_contamination(self):
        """Breakdown 1/2, which the standard deviation does not have at all."""
        z = np.random.default_rng(0).standard_normal(50_000)
        dirty = z.copy()
        dirty[:20_000] += 500.0
        assert scale.mad_scale(dirty) < 6.0
        assert dirty.std() > 100.0

    def test_scales_linearly(self):
        z = np.random.default_rng(0).standard_normal(20_000)
        assert scale.mad_scale(5 * z) == pytest.approx(5 * scale.mad_scale(z),
                                                       rel=1e-9)

    def test_centering_at_zero_is_available_and_different(self):
        x = np.array([10.0, 11.0, 12.0, 13.0])
        assert scale.mad_scale(x, center=0.0) > scale.mad_scale(x)

    def test_ignores_non_finite_and_refuses_an_empty_sample(self):
        assert np.isfinite(scale.mad_scale([1.0, 2.0, np.nan, 3.0, np.inf]))
        with pytest.raises(ValueError):
            scale.mad_scale([np.nan, np.inf])


class TestTauScale:
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 7.0])
    def test_consistent_at_the_gaussian(self, sigma):
        z = np.random.default_rng(0).standard_normal(100_000) * sigma
        assert scale.tau_scale(z) == pytest.approx(sigma, rel=0.02)

    def test_more_efficient_than_the_mad(self):
        """The whole reason it exists: near the standard deviation's precision."""
        def spread(fn):
            g = np.random.default_rng(1)
            return float(np.std([fn(g.standard_normal(200)) for _ in range(300)]))
        mad, tau, sd = spread(scale.mad_scale), spread(scale.tau_scale), spread(np.std)
        assert tau < 0.75 * mad
        assert tau < 1.3 * sd

    def test_still_bounded_under_heavy_contamination(self):
        z = np.random.default_rng(0).standard_normal(50_000)
        dirty = z.copy()
        dirty[:20_000] += 500.0
        assert scale.tau_scale(dirty) < 20.0

    def test_scales_linearly(self):
        z = np.random.default_rng(0).standard_normal(20_000)
        assert scale.tau_scale(4 * z) == pytest.approx(4 * scale.tau_scale(z),
                                                       rel=1e-9)

    def test_rejects_a_non_positive_tuning_constant(self):
        with pytest.raises(ValueError):
            scale.tau_scale([1.0, 2.0, 3.0], c=0.0)

    def test_zero_spread_gives_zero(self):
        assert scale.tau_scale(np.ones(50)) == 0.0


class TestResidualScale:
    def test_is_a_zero_centred_mad_of_the_residuals(self):
        y = np.array([1.0, 2.0, 3.0, 40.0])
        p = np.array([1.1, 2.1, 3.1, 3.1])
        assert scale.residual_scale(y, p) == pytest.approx(
            scale.mad_scale(y - p, center=0.0))

    def test_tau_method_available(self):
        y = np.random.default_rng(0).standard_normal(5000)
        assert scale.residual_scale(y, np.zeros_like(y), method="tau") == \
            pytest.approx(1.0, rel=0.05)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            scale.residual_scale([1.0, 2.0], [0.0, 0.0], method="huber")

    def test_a_contaminated_response_gives_a_bigger_scale_than_its_residuals(self):
        """Why a one-shot rescaling of y is not enough, in one assertion."""
        rng = np.random.default_rng(0)
        y = rng.standard_normal(4000)
        dirty = y.copy()
        dirty[:400] += 30.0
        from_y = scale.mad_scale(dirty)
        from_resid = scale.residual_scale(dirty, np.where(np.arange(4000) < 400,
                                                          30.0, 0.0))
        assert from_y > from_resid


class TestVerticalOutliers:
    def test_leaves_the_design_untouched(self, data):
        X, y, *_ = data
        out = contamination.vertical_outliers(X, y, fraction=0.1)
        assert np.allclose(out.X, X)
        assert out.kind == "vertical outliers"

    def test_changes_exactly_the_requested_count(self, data):
        X, y, *_ = data
        out = contamination.vertical_outliers(X, y, fraction=0.05)
        assert out.n_contaminated == round(0.05 * len(y))
        changed = np.flatnonzero(~np.isclose(out.y, y))
        assert set(changed) == set(out.index)

    def test_shifts_by_the_stated_magnitude(self, data):
        X, y, *_ = data
        out = contamination.vertical_outliers(X, y, fraction=0.1, magnitude=20.0)
        assert np.allclose(np.abs(out.y[out.index] - y[out.index]), 20.0)

    def test_symmetric_by_default_and_one_sided_on_request(self, data):
        X, y, *_ = data
        sym = contamination.vertical_outliers(X, y, fraction=0.2, seed=3)
        d = sym.y[sym.index] - y[sym.index]
        assert (d > 0).any() and (d < 0).any()
        one = contamination.vertical_outliers(X, y, fraction=0.2, symmetric=False)
        assert ((one.y[one.index] - y[one.index]) > 0).all()

    def test_zero_fraction_is_a_no_op(self, data):
        X, y, *_ = data
        out = contamination.vertical_outliers(X, y, fraction=0.0)
        assert out.n_contaminated == 0 and np.allclose(out.y, y)

    def test_rejects_an_impossible_fraction(self, data):
        X, y, *_ = data
        for f in (-0.1, 1.0, 2.0):
            with pytest.raises(ValueError):
                contamination.vertical_outliers(X, y, fraction=f)

    def test_reports_what_it_did(self, data):
        X, y, *_ = data
        out = contamination.vertical_outliers(X, y, fraction=0.1, magnitude=7.0)
        text = out.describe()
        assert "vertical" in text and "magnitude=7.0" in text
        assert out.fraction == pytest.approx(0.1, abs=1e-3)


class TestLeveragePoints:
    def test_moves_only_the_named_columns(self, data):
        X, y, *_ = data
        out = contamination.leverage_points(X, y, truth, fraction=0.1,
                                            distance=2.5, columns=(0,))
        assert np.allclose(out.X[:, 1:], X[:, 1:])
        assert np.abs(out.X[out.index, 0]).mean() > 2.0

    def test_places_them_near_the_requested_distance(self, data):
        X, y, *_ = data
        for d in (2.5, 6.0):
            out = contamination.leverage_points(X, y, truth, fraction=0.1,
                                                distance=d, jitter=0.1)
            assert np.abs(out.X[out.index, 0]).mean() == pytest.approx(d, abs=0.3)

    def test_the_y_is_wrong_by_the_magnitude_at_the_new_location(self, data):
        X, y, *_ = data
        out = contamination.leverage_points(X, y, truth, fraction=0.1,
                                            distance=3.0, magnitude=20.0)
        err = out.y[out.index] - truth(out.X[out.index])
        assert np.allclose(np.abs(err), 20.0)

    def test_rejects_columns_out_of_range(self, data):
        X, y, *_ = data
        with pytest.raises(ValueError):
            contamination.leverage_points(X, y, truth, columns=(99,))

    def test_zero_fraction_is_a_no_op(self, data):
        X, y, *_ = data
        out = contamination.leverage_points(X, y, truth, fraction=0.0)
        assert out.n_contaminated == 0
        assert np.allclose(out.X, X) and np.allclose(out.y, y)

    def test_records_the_distance_it_used(self, data):
        X, y, *_ = data
        out = contamination.leverage_points(X, y, truth, distance=4.0)
        assert out.params["distance"] == 4.0
        assert "distance=4.0" in out.describe()


class TestScaleSweep:
    def test_an_equivariant_procedure_has_no_gap(self, data):
        X, y, Xt, _ = data
        gap = equivariance.equivariance_gap(
            lambda A, b, s: np.full(len(Xt), b.mean()), X, y, Xt)
        assert gap < 1e-12

    def test_a_procedure_with_a_fixed_constant_in_it_does(self, data):
        X, y, Xt, _ = data
        gap = equivariance.equivariance_gap(
            lambda A, b, s: np.full(len(Xt), np.clip(b.mean(), -0.05, 0.05)),
            X, y, Xt)
        assert gap > 0.1

    def test_least_squares_is_exactly_equivariant(self, data):
        """The reference case: a homogeneous loss cannot care about units."""
        from sklearn.linear_model import LinearRegression
        X, y, Xt, _ = data
        gap = equivariance.equivariance_gap(
            lambda A, b, s: LinearRegression().fit(A, b).predict(Xt), X, y, Xt,
            scales=(0.01, 1.0, 100.0))
        assert gap < 1e-9

    def test_passing_the_scale_through_lets_a_procedure_fix_itself(self, data):
        X, y, Xt, _ = data
        bad = equivariance.equivariance_gap(
            lambda A, b, s: np.full(len(Xt), np.clip(b.mean(), -0.05, 0.05)),
            X, y, Xt)
        fixed = equivariance.equivariance_gap(
            lambda A, b, s: np.full(len(Xt), np.clip(b.mean(), -0.05 * s, 0.05 * s)),
            X, y, Xt)
        assert fixed < 1e-12 < bad

    def test_reference_scale_is_the_one_nearest_the_request(self, data):
        X, y, Xt, _ = data
        res = equivariance.scale_sweep(
            lambda A, b, s: np.full(len(Xt), b.mean()), X, y, Xt,
            scales=(0.1, 3.0, 100.0), reference=3.0)
        assert res.scales[res.reference] == 3.0

    def test_rms_gap_is_not_driven_by_a_near_zero_prediction(self, data):
        """`gap` divides by the reference prediction; `rms_gap` does not."""
        X, y, Xt, _ = data

        def fp(A, b, s):
            # One test point sits at 1e-11 and is perturbed by a further 1e-9 after
            # rescaling at the large scale: negligible against predictions of order
            # one, enormous against its own value.
            out = np.full(len(Xt), b.mean())
            out[0] = 1e-11 * s + (1e-7 if s > 50 else 0.0)
            return out
        res = equivariance.scale_sweep(fp, X, y, Xt, scales=(1.0, 100.0))
        assert res.gap > 50.0
        assert res.rms_gap < 1e-6

    def test_rejects_non_positive_scales(self, data):
        X, y, Xt, _ = data
        with pytest.raises(ValueError):
            equivariance.scale_sweep(lambda A, b, s: np.zeros(len(Xt)), X, y, Xt,
                                     scales=(0.0, 1.0))

    def test_rejects_a_callback_that_changes_its_output_length(self, data):
        X, y, Xt, _ = data
        with pytest.raises(ValueError):
            equivariance.scale_sweep(
                lambda A, b, s: np.zeros(3 if s < 1 else 4), X, y, Xt,
                scales=(0.5, 2.0))

    def test_describe_mentions_both_summaries(self, data):
        X, y, Xt, _ = data
        res = equivariance.scale_sweep(lambda A, b, s: np.full(len(Xt), b.mean()),
                                       X, y, Xt)
        assert "gap" in res.describe() and "scales" in res.describe()


class TestHuberSlopeFor:
    def test_is_a_robust_scale_of_the_response(self):
        y = np.random.default_rng(0).standard_normal(20_000) * 3.0
        assert equivariance.huber_slope_for(y) == pytest.approx(3.0, rel=0.05)

    def test_scales_with_the_response(self):
        y = np.random.default_rng(0).standard_normal(5000)
        assert equivariance.huber_slope_for(100 * y) == pytest.approx(
            100 * equivariance.huber_slope_for(y), rel=1e-9)

    def test_multiple_is_applied(self):
        y = np.random.default_rng(0).standard_normal(5000)
        assert equivariance.huber_slope_for(y, multiple=2.0) == pytest.approx(
            2.0 * equivariance.huber_slope_for(y), rel=1e-9)

    def test_refuses_a_degenerate_response(self):
        with pytest.raises(ValueError):
            equivariance.huber_slope_for(np.ones(100))


class TestScaleSweepOnRealBoosting:
    """The claim exp014 rests on, asserted against the actual library."""

    @staticmethod
    def _fit(objective, slope=None, rounds=120):
        xgb = pytest.importorskip("xgboost")

        def fp(A, b, s):
            kw = dict(n_estimators=rounds, max_depth=3, learning_rate=0.1,
                      objective=objective, random_state=0, n_jobs=2)
            if slope == "auto":
                kw["huber_slope"] = equivariance.huber_slope_for(b)
            elif slope is not None:
                kw["huber_slope"] = slope
            return xgb.XGBRegressor(**kw).fit(A, b).predict(fp.X_test)
        return fp

    def test_squared_error_is_equivariant(self, data):
        X, y, Xt, _ = data
        fp = self._fit("reg:squarederror")
        fp.X_test = Xt
        assert equivariance.equivariance_gap(
            fp, X, y, Xt, scales=(0.1, 1.0, 10.0, 100.0)) < 1e-6

    def test_a_fixed_huber_slope_is_not(self, data):
        X, y, Xt, _ = data
        fp = self._fit("reg:pseudohubererror", slope=1.0)
        fp.X_test = Xt
        assert equivariance.equivariance_gap(
            fp, X, y, Xt, scales=(0.1, 1.0, 10.0, 100.0)) > 0.05

    def test_setting_the_slope_from_the_data_restores_it(self, data):
        X, y, Xt, _ = data
        fp = self._fit("reg:pseudohubererror", slope="auto")
        fp.X_test = Xt
        assert equivariance.equivariance_gap(
            fp, X, y, Xt, scales=(0.1, 1.0, 10.0, 100.0)) < 1e-6


class TestLeverageDistanceIsNotMonotone:
    """For a tree ensemble the near leverage point is the dangerous one."""

    def test_near_leverage_hurts_more_than_far_leverage(self, data):
        xgb = pytest.importorskip("xgboost")
        X, y, Xt, ft = data

        def score(distance):
            out = contamination.leverage_points(X, y, truth, fraction=0.10,
                                                distance=distance, magnitude=20.0)
            m = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.1,
                                 objective="reg:squarederror", random_state=0,
                                 n_jobs=2).fit(out.X, out.y)
            return float(np.sqrt(np.mean((m.predict(Xt) - ft) ** 2)))

        near, far = score(2.5), score(10.0)
        assert near > far, (near, far)
