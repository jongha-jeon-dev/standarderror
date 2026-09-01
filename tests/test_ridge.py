"""Tests for the ridge module behind episode five."""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.linalg import ridge as rg


class TestTheTwoDiagnosticsMeasureDifferentThings:
    def test_vif_is_at_its_floor_on_a_hopeless_design(self):
        X = rg.units_design(500, rng=np.random.default_rng(0))
        assert rg.vif(X).max() < 1.01, "uncorrelated columns, so VIF can say nothing"
        assert rg.condition_indices(X)[-1] > 1e7, "and the design has no digits left"

    def test_vif_is_enormous_on_a_design_the_condition_number_shrugs_at(self):
        C = rg.collinear_design(600, rng=np.random.default_rng(1), strength=0.9995)
        assert rg.vif(C).max() > 100
        assert rg.condition_indices(C)[-1] < 200, (
            "both are right: they answer different questions")

    def test_vif_does_not_move_when_a_column_is_rescaled(self):
        rng = np.random.default_rng(2)
        X = rg.collinear_design(300, rng=rng, p=4)
        scaled = X.copy()
        scaled[:, 2] *= 1e6
        assert rg.vif(scaled) == pytest.approx(rg.vif(X), rel=1e-6)

    def test_the_condition_number_does(self):
        rng = np.random.default_rng(2)
        X = rg.collinear_design(300, rng=rng, p=4)
        scaled = X.copy()
        scaled[:, 2] *= 1e6
        assert (rg.condition_indices(scaled)[-1]
                > 1e4 * rg.condition_indices(X)[-1])

    def test_a_single_predictor_has_a_vif_of_exactly_one(self):
        rng = np.random.default_rng(3)
        X = np.column_stack([np.ones(200), rng.normal(3600, 600, 200)])
        assert rg.vif(X) == pytest.approx([1.0])
        assert rg.condition_indices(X)[-1] > 1e3


class TestRidgeIsAnEigenvalueShift:
    def test_shrinkage_is_between_zero_and_one_and_falls_with_alpha(self):
        s = np.array([10.0, 1.0, 0.1])
        assert rg.shrinkage(s, 0.0) == pytest.approx([1.0, 1.0, 1.0])
        f = rg.shrinkage(s, 1.0)
        assert (0 < f).all() and (f < 1).all()
        assert (f == np.sort(f)[::-1]).all(), "large directions keep more"

    def test_it_shrinks_the_weakest_direction_hardest(self):
        s = np.array([10.0, 1.0, 0.1])
        f = rg.shrinkage(s, 1.0)
        assert f[0] > 0.99 and f[-1] < 0.02

    def test_effective_df_is_the_rank_at_zero_and_falls_to_nothing(self):
        C = rg.collinear_design(400, rng=np.random.default_rng(4), p=6)
        s = np.linalg.svd(C, compute_uv=False)
        assert rg.effective_df(s, 0.0) == pytest.approx(C.shape[1])
        assert rg.effective_df(s, 1e12) < 0.01
        assert rg.effective_df(s, 1.0) < C.shape[1]

    def test_ridge_at_zero_is_least_squares(self):
        rng = np.random.default_rng(5)
        X = rg.collinear_design(300, rng=rng, p=4)
        y = rng.standard_normal(300)
        got = rg.ridge_fit(X, y, 0.0).coefficients
        want = np.linalg.lstsq(X, y, rcond=None)[0]
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)

    def test_the_penalty_matches_the_closed_form(self):
        rng = np.random.default_rng(6)
        X = rg.collinear_design(200, rng=rng, p=4)
        y = rng.standard_normal(200)
        a = 3.7
        want = np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T @ y)
        assert rg.ridge_fit(X, y, a).coefficients == pytest.approx(want, rel=1e-8)


class TestTheIntervalsAreConditionallyFine:
    """The episode's third finding, and the reason it is stated conditionally:
    a blanket claim that ridge intervals under-cover is false, and measurably so."""

    def _setup(self):
        C = rg.collinear_design(600, rng=np.random.default_rng(1), p=8,
                                strength=0.9995)
        Vt = np.linalg.svd(C, full_matrices=False)[2]
        return C, Vt

    def test_they_hold_when_the_truth_is_in_a_well_measured_direction(self):
        C, Vt = self._setup()
        cov = rg.coverage(C, Vt[0] * 3.0, alpha=2.57, sigma=1.0, reps=300,
                          rng=np.random.default_rng(3))
        assert cov["coverage"].min() > 0.90

    def test_they_collapse_when_it_is_in_the_weakest_one(self):
        C, Vt = self._setup()
        cov = rg.coverage(C, Vt[-1] * 3.0, alpha=2.57, sigma=1.0, reps=300,
                          rng=np.random.default_rng(3))
        assert np.median(cov["coverage"]) < 0.70
        assert np.abs(cov["mean_bias"] / cov["standard_error"]).max() > 3.0

    def test_the_variance_formula_itself_is_right(self):
        """So the shortfall above is bias, not a mis-derived standard error."""
        C, _ = self._setup()
        rng = np.random.default_rng(7)
        beta = np.ones(C.shape[1])
        mu = C @ beta
        draws = np.array([rg.ridge_fit(C, mu + rng.normal(0, 1.0, len(C)),
                                       2.57).coefficients for _ in range(800)])
        assert draws.var(axis=0) == pytest.approx(
            rg.ridge_variance(C, 2.57, 1.0), rel=0.2)


class TestCrossValidation:
    def test_it_picks_an_alpha_that_spends_fewer_parameters_than_p(self):
        C = rg.collinear_design(600, rng=np.random.default_rng(1), p=8,
                                strength=0.9995)
        rng = np.random.default_rng(2)
        y = C @ np.ones(C.shape[1]) + rng.normal(0, 1.0, len(C))
        cv = rg.cross_validated_alpha(C, y, np.logspace(-3, 4, 40), rng=rng)
        assert 0 < cv["effective_df"] < C.shape[1]
        assert cv["alpha"] > 0

    def test_the_error_curve_has_an_interior_minimum(self):
        C = rg.collinear_design(600, rng=np.random.default_rng(1), p=8,
                                strength=0.9995)
        rng = np.random.default_rng(2)
        y = C @ np.ones(C.shape[1]) + rng.normal(0, 1.0, len(C))
        cv = rg.cross_validated_alpha(C, y, np.logspace(-3, 4, 40), rng=rng)
        best = int(np.argmin(cv["cv_error"]))
        assert 0 < best < len(cv["cv_error"]) - 1, (
            "an optimum at an endpoint means the grid was the wrong range")


class TestHardAgainstSoft:
    """Episode two deferred `rcond` to episode five; this is the comparison."""

    def _data(self):
        C = rg.collinear_design(600, rng=np.random.default_rng(32), p=8,
                                strength=0.9995)
        rng = np.random.default_rng(33)
        beta = np.ones(C.shape[1])
        return C, C @ beta + rng.normal(0, 1.0, 600), beta

    def test_alpha_for_df_hits_the_target(self):
        s = np.linalg.svd(self._data()[0], compute_uv=False)
        for target in (1.0, 3.0, 4.5, 8.0):
            assert rg.effective_df(s, rg.alpha_for_df(s, target)) == pytest.approx(
                target, rel=1e-6)

    @pytest.mark.parametrize("target", [0.0, -1.0, 9.0, 20.0])
    def test_an_unreachable_target_is_refused(self, target):
        s = np.linalg.svd(self._data()[0], compute_uv=False)
        with pytest.raises(ValueError, match="target_df"):
            rg.alpha_for_df(s, target)

    def test_truncating_at_full_rank_is_least_squares(self):
        C, y, _ = self._data()
        got = rg.truncated_svd_fit(C, y, C.shape[1])
        want = np.linalg.lstsq(C, y, rcond=None)[0]
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)

    def test_the_two_agree_where_it_matters_and_diverge_where_it_does_not(self):
        """A cliff and a ramp answering the same question. They agree most
        closely at the ranks the data supports and part company at the extremes,
        where the same budget of parameters is being spent very differently."""
        rows = {r["rank"]: r for r in rg.hard_against_soft(*self._data())}
        for k in (3, 4, 5):
            assert rows[k]["soft_error"] == pytest.approx(
                rows[k]["hard_error"], abs=0.1)
        assert abs(rows[1]["soft_error"] - rows[1]["hard_error"]) > 0.3

    def test_both_bottom_out_at_the_number_of_real_directions(self):
        C, y, beta = self._data()
        rows = rg.hard_against_soft(C, y, beta)
        s = np.linalg.svd(C, compute_uv=False)
        real = int((s > 0.1 * s.max()).sum())
        assert min(rows, key=lambda r: r["hard_error"])["rank"] == real
        assert min(rows, key=lambda r: r["soft_error"])["rank"] == real
