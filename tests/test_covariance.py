"""The positive-definiteness constraint, and the two ways it gets violated."""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.linalg import covariance as cv

LONG = cv.correlation3(0.35, 0.0, 0.0)
SHORT = cv.correlation3(-0.60, 0.90, -0.50)


# ------------------------------------------------------------------ the matrix

class TestCorrelation3:
    def test_is_symmetric_with_a_unit_diagonal(self):
        R = cv.correlation3(0.2, -0.4, 0.6)
        assert np.allclose(R, R.T)
        assert np.allclose(np.diag(R), 1.0)

    def test_places_each_entry_where_its_name_says(self):
        R = cv.correlation3(0.1, 0.2, 0.3)
        assert R[0, 1] == 0.1 and R[0, 2] == 0.2 and R[1, 2] == 0.3

    @pytest.mark.parametrize("bad", [(1.5, 0.0, 0.0), (0.0, -1.2, 0.0),
                                     (0.0, 0.0, 3.0)])
    def test_refuses_a_correlation_outside_minus_one_to_one(self, bad):
        with pytest.raises(ValueError, match=r"not in \[-1, 1\]"):
            cv.correlation3(*bad)

    def test_does_not_check_feasibility(self):
        # The point of the smallest example: three legal numbers, one illegal
        # matrix. A constructor that refused this would hide the lesson.
        R = cv.correlation3(0.9, 0.9, -0.9)
        assert np.linalg.eigvalsh(R)[0] < 0


class TestFeasibleBand:
    def test_is_the_cosine_of_the_sum_and_the_difference(self):
        # The algebraic band and the angular triangle inequality are the same
        # statement; if they ever disagree one of the two derivations is wrong.
        for a, b in [(0.3, 0.7), (-0.5, 0.2), (0.0, 0.0), (0.95, -0.4)]:
            lo, hi = cv.feasible_band(a, b)
            alpha, beta = np.arccos(a), np.arccos(b)
            assert lo == pytest.approx(np.cos(alpha + beta), abs=1e-12)
            assert hi == pytest.approx(np.cos(alpha - beta), abs=1e-12)

    def test_two_uncorrelated_pairs_leave_the_third_free(self):
        assert cv.feasible_band(0.0, 0.0) == pytest.approx((-1.0, 1.0))

    def test_two_strong_correlations_pin_the_third(self):
        lo, hi = cv.feasible_band(0.95, 0.95)
        assert hi - lo < 0.2
        assert lo > 0.7  # and it must be strongly positive

    def test_an_endpoint_is_exactly_singular(self):
        lo, hi = cv.feasible_band(0.6, -0.3)
        for c in (lo, hi):
            assert np.linalg.eigvalsh(cv.correlation3(0.6, -0.3, c))[0] == \
                pytest.approx(0.0, abs=1e-12)

    def test_just_outside_an_endpoint_is_negative(self):
        lo, hi = cv.feasible_band(0.6, -0.3)
        assert np.linalg.eigvalsh(cv.correlation3(0.6, -0.3, hi + 1e-3))[0] < 0
        assert np.linalg.eigvalsh(cv.correlation3(0.6, -0.3, lo - 1e-3))[0] < 0

    def test_a_perfect_correlation_leaves_no_room_at_all(self):
        lo, hi = cv.feasible_band(1.0, 0.4)
        assert lo == pytest.approx(0.4) and hi == pytest.approx(0.4)

    def test_refuses_an_impossible_input(self):
        with pytest.raises(ValueError, match="must be in"):
            cv.feasible_band(1.4, 0.2)


class TestCorrelationAngles:
    def test_slack_is_positive_exactly_when_the_matrix_is_infeasible(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            a, b, c = rng.uniform(-0.98, 0.98, size=3)
            slack = cv.correlation_angles(a, b, c)["slack"]
            negative = np.linalg.eigvalsh(cv.correlation3(a, b, c))[0] < -1e-9
            assert negative == (slack > 1e-7), (a, b, c, slack)

    def test_reports_the_angles_a_correlation_stands_for(self):
        out = cv.correlation_angles(0.0, 1.0, 0.0)
        assert out["angle_ab"] == pytest.approx(90.0)
        assert out["angle_ac"] == pytest.approx(0.0)

    def test_the_known_failure_is_fourteen_degrees_outside(self):
        out = cv.correlation_angles(0.283, 0.900, -0.408)
        assert out["slack"] == pytest.approx(14.5, abs=0.4)


# ------------------------------------------------------------------- the tests

class TestPSDReport:
    def test_a_valid_matrix_passes_both_tests(self):
        r = cv.psd_report(cv.correlation3(0.3, 0.2, 0.1))
        assert r.cholesky_ok and r.is_psd and r.min_eigenvalue > 0

    def test_an_infeasible_matrix_fails_both(self):
        r = cv.psd_report(cv.correlation3(0.9, 0.9, -0.9))
        assert not r.cholesky_ok and not r.is_psd

    def test_the_worst_variance_is_the_smallest_eigenvalue(self):
        # w' S w for the unit eigenvector *is* lambda_min. Asserting it here is
        # what licenses calling the eigenvector "the offending portfolio".
        r = cv.psd_report(cv.correlation3(0.9, 0.9, -0.9))
        assert r.worst_variance == pytest.approx(r.min_eigenvalue)
        assert np.linalg.norm(r.worst_weights) == pytest.approx(1.0)

    def test_a_singular_but_valid_matrix_is_psd_and_fails_cholesky(self):
        # The nuance episode one's exercise turns on: Cholesky answers "positive
        # *definite*", so a genuinely rank-deficient covariance matrix -- two
        # identical columns, say -- fails it while being a perfectly real
        # covariance matrix. Failing Cholesky is not evidence of an error.
        R = cv.correlation3(1.0, 0.4, 0.4)
        r = cv.psd_report(R)
        assert r.is_psd and not r.cholesky_ok
        assert r.min_eigenvalue == pytest.approx(0.0, abs=1e-12)

    def test_refuses_a_non_square_matrix(self):
        with pytest.raises(ValueError, match="square"):
            cv.psd_report(np.zeros((2, 3)))

    def test_refuses_an_asymmetric_matrix(self):
        with pytest.raises(ValueError, match="not symmetric"):
            cv.psd_report(np.array([[1.0, 0.5], [0.2, 1.0]]))

    def test_eigenvalues_come_back_ascending(self):
        r = cv.psd_report(cv.correlation3(0.5, 0.4, 0.3))
        assert r.eigenvalues == sorted(r.eigenvalues)


class TestLeverageTable:
    def test_variance_grows_with_the_square_of_the_leverage(self):
        R = cv.correlation3(0.9, 0.9, -0.9)
        r = cv.psd_report(R)
        rows = cv.leverage_table(R, r.worst_weights, leverages=(1.0, 3.0))
        assert rows[1]["variance"] == pytest.approx(9.0 * rows[0]["variance"])

    def test_a_negative_variance_has_no_reported_standard_deviation(self):
        R = cv.correlation3(0.9, 0.9, -0.9)
        rows = cv.leverage_table(R, cv.psd_report(R).worst_weights)
        assert all(np.isnan(row["reported_sd"]) for row in rows)

    def test_a_valid_matrix_reports_a_real_standard_deviation(self):
        R = cv.correlation3(0.3, 0.2, 0.1)
        rows = cv.leverage_table(R, [1.0, 0.0, 0.0], leverages=(2.0,))
        assert rows[0]["reported_sd"] == pytest.approx(2.0)

    def test_refuses_weights_of_the_wrong_length(self):
        with pytest.raises(ValueError, match="weights of shape"):
            cv.leverage_table(np.eye(3), [1.0, 0.0])


# -------------------------------------------------------------- the estimators

class TestCompleteCaseIsAlwaysFeasible:
    def test_over_many_random_panels(self):
        # Not a coincidence to be checked once: a complete-case correlation
        # matrix is Z'Z/(n-1) for a single Z, so w'Sw = ||Zw||^2 / (n-1) >= 0
        # identically. This is the property test for that algebra.
        rng = np.random.default_rng(4)
        for _ in range(60):
            p = int(rng.integers(3, 7))
            R = np.full((p, p), 0.4)
            np.fill_diagonal(R, 1.0)
            X = cv.mcar_panel(int(rng.integers(30, 120)), R,
                              missing_rate=float(rng.uniform(0.0, 0.4)),
                              rng=rng)
            if (~np.isnan(X).any(axis=1)).sum() < p + 3:
                continue
            C = cv.complete_case_correlation(X)["matrix"]
            assert np.linalg.eigvalsh(C)[0] > -1e-9

    def test_reports_what_it_threw_away(self):
        rng = np.random.default_rng(0)
        R = cv.correlation3(0.3, 0.2, 0.1)
        X = cv.mcar_panel(400, R, missing_rate=0.2, rng=rng)
        out = cv.complete_case_correlation(X)
        assert out["n_used"] + out["rows_dropped"] == 400
        assert out["rows_dropped"] > 0

    def test_refuses_when_almost_nothing_is_complete(self):
        X = np.full((10, 3), np.nan)
        X[0] = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="complete rows"):
            cv.complete_case_correlation(X)


class TestPairwiseCorrelation:
    def test_agrees_with_corrcoef_when_nothing_is_missing(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 4))
        assert np.allclose(cv.pairwise_correlation(X)["matrix"],
                           np.corrcoef(X.T), atol=1e-12)

    def test_records_the_overlap_behind_every_entry(self):
        X = np.column_stack([np.arange(100.0), np.arange(100.0) ** 1.5,
                             np.arange(100.0) ** 0.5])
        X[:60, 2] = np.nan
        out = cv.pairwise_correlation(X)
        assert out["n_used"][0, 1] == 100
        assert out["n_used"][0, 2] == 40
        assert out["max_overlap"] == 100 and out["min_overlap"] == 40

    def test_refuses_an_entry_with_almost_no_overlap(self):
        X = np.column_stack([np.arange(50.0), np.arange(50.0) ** 2])
        X[3:, 1] = np.nan
        with pytest.raises(ValueError, match="min_overlap"):
            cv.pairwise_correlation(X)

    def test_refuses_a_constant_column(self):
        X = np.column_stack([np.arange(50.0), np.ones(50)])
        with pytest.raises(ValueError, match="constant"):
            cv.pairwise_correlation(X)


# ------------------------------------------------------------- the two failures

class TestSamplingNoiseVanishes:
    def test_the_rate_falls_to_zero_as_n_grows(self):
        p = 8
        R = np.full((p, p), 0.3)
        np.fill_diagonal(R, 1.0)
        rows = cv.negative_rate((40, 120, 600), correlation=R,
                                missing_rate=0.5, reps=60, seed=2)
        assert rows[0]["rate"] > rows[-1]["rate"]
        assert rows[-1]["rate"] == 0.0

    def test_it_reports_how_far_outside_the_failures_fall(self):
        R = cv.correlation3(0.3, 0.3, 0.3)
        rows = cv.negative_rate((60,), correlation=R, missing_rate=0.4,
                                reps=40, seed=0)
        assert rows[0]["worst_min_eigenvalue"] <= rows[0]["mean_min_eigenvalue"]

    def test_refuses_to_simulate_from_an_infeasible_target(self):
        with pytest.raises(ValueError, match="infeasible"):
            cv.mcar_panel(50, cv.correlation3(0.9, 0.9, -0.9),
                          missing_rate=0.1, rng=np.random.default_rng(0))


class TestHeterogeneousOverlapDoesNot:
    def test_the_negative_eigenvalue_survives_every_sample_size(self):
        rows = cv.regime_limit((250, 2000), long_correlation=LONG,
                               short_correlation=SHORT)
        assert all(r["min_eigenvalue"] < -0.05 for r in rows)

    def test_while_the_complete_case_matrix_stays_feasible_throughout(self):
        rows = cv.regime_limit((250, 2000), long_correlation=LONG,
                               short_correlation=SHORT)
        assert all(r["min_eigenvalue_complete"] > 0 for r in rows)

    def test_growing_the_sample_does_not_help(self):
        # The distinguishing claim of the episode, asserted rather than argued:
        # this is a bias, so the last size is not closer to feasible than the
        # first. Written as "does not shrink towards zero" rather than
        # "gets worse", because it converges rather than diverging.
        rows = cv.regime_limit((250, 4000), long_correlation=LONG,
                               short_correlation=SHORT)
        assert rows[-1]["min_eigenvalue"] <= rows[0]["min_eigenvalue"] + 0.02

    def test_the_overlaps_differ_which_is_the_diagnosis(self):
        X = cv.two_regime_panel(2000, 250, long_correlation=LONG,
                                short_correlation=SHORT,
                                rng=np.random.default_rng(11))
        out = cv.pairwise_correlation(X)
        assert out["max_overlap"] == 2250 and out["min_overlap"] == 250

    def test_refuses_a_panel_with_no_late_columns(self):
        with pytest.raises(ValueError, match="late columns"):
            cv.two_regime_panel(100, 50, long_correlation=LONG,
                                short_correlation=SHORT, late_columns=(),
                                rng=np.random.default_rng(0))

    def test_refuses_regimes_over_different_variables(self):
        with pytest.raises(ValueError, match="same variables"):
            cv.two_regime_panel(100, 50, long_correlation=LONG,
                                short_correlation=np.eye(4),
                                rng=np.random.default_rng(0))


# ---------------------------------------------------------------- the repairs

BROKEN = cv.correlation3(0.283, 0.900, -0.408)


class TestClipToPSD:
    def test_the_result_is_feasible(self):
        assert np.linalg.eigvalsh(cv.clip_to_psd(BROKEN))[0] > -1e-12

    def test_renormalising_restores_the_unit_diagonal(self):
        assert np.allclose(np.diag(cv.clip_to_psd(BROKEN)), 1.0)

    def test_without_renormalising_the_variances_are_wrong(self):
        # The step that is easy to skip and changes the answer: clipping moves
        # the diagonal, so the unrenormalised result has variances nobody
        # measured.
        raw = cv.clip_to_psd(BROKEN, renormalise=False)
        assert not np.allclose(np.diag(raw), 1.0, atol=1e-3)


class TestNearestCorrelation:
    def test_converges_and_returns_a_correlation_matrix(self):
        out = cv.nearest_correlation(BROKEN)
        assert out["converged"]
        assert np.allclose(np.diag(out["matrix"]), 1.0)
        assert np.linalg.eigvalsh(out["matrix"])[0] > -1e-12

    def test_leaves_an_already_valid_matrix_alone(self):
        R = cv.correlation3(0.3, 0.2, 0.1)
        assert np.allclose(cv.nearest_correlation(R)["matrix"], R, atol=1e-10)

    def test_is_at_least_as_close_as_clipping(self):
        # It solves the nearest-matrix problem and clipping does not, so this is
        # the property that justifies the extra thirty iterations. On a 3x3 the
        # margin is about one percent -- worth measuring rather than assuming.
        h = np.linalg.norm(cv.nearest_correlation(BROKEN)["matrix"] - BROKEN,
                           "fro")
        c = np.linalg.norm(cv.clip_to_psd(BROKEN) - BROKEN, "fro")
        assert h <= c + 1e-12

    def test_a_caller_can_see_it_gave_up(self):
        out = cv.nearest_correlation(BROKEN, max_iter=1, tol=1e-16)
        assert not out["converged"] and out["iterations"] == 1


class TestRepairCost:
    def test_names_the_entry_that_moved_most(self):
        out = cv.repair_cost(BROKEN, cv.nearest_correlation(BROKEN)["matrix"],
                             labels=["A", "B", "C"])
        assert out["largest_moved"] == "A-C"
        assert out["max_change"] == pytest.approx(0.078, abs=0.01)

    def test_reports_one_row_per_off_diagonal_pair(self):
        out = cv.repair_cost(np.eye(4), np.eye(4))
        assert len(out["entries"]) == 6
        assert out["frobenius"] == 0.0

    def test_refuses_mismatched_shapes(self):
        with pytest.raises(ValueError, match="not comparable"):
            cv.repair_cost(np.eye(3), np.eye(4))

    def test_refuses_the_wrong_number_of_labels(self):
        with pytest.raises(ValueError, match="labels for"):
            cv.repair_cost(np.eye(3), np.eye(3), labels=["A", "B"])
