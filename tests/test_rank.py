"""Tests for the rank module behind episode seven."""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.linalg import rank as rk


class TestEckartYoung:
    def _matrix(self, seed=0):
        rng = np.random.default_rng(seed)
        return rk.low_rank_plus_noise(300, 20, singular_values=[40, 38, 36],
                                      sigma=1.0, rng=rng), rng

    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_the_truncation_error_is_the_tail_of_the_spectrum(self, k):
        Y, rng = self._matrix()
        got = rk.eckart_young_check(Y, k, trials=5, rng=rng)
        assert got["truncation_error"] == pytest.approx(got["predicted_error"],
                                                       rel=1e-12)

    def test_nothing_of_the_same_rank_beats_it(self):
        Y, rng = self._matrix()
        got = rk.eckart_young_check(Y, 3, trials=200, rng=rng)
        assert got["never_beaten"] and got["smallest_margin"] > 0


class TestTheRulesDisagree:
    """The episode's subject. Four rules, and the disagreement is about which
    loss is being minimised rather than about arithmetic."""

    def _sweep(self):
        return rk.strength_sweep((1.5, 0.8, 0.35, 0.0), n=400, p=20, rank=3,
                                 sigma=1.0, reps=40,
                                 rng=np.random.default_rng(11))

    def test_everything_works_when_the_signal_is_strong(self):
        """Which is what makes the rest of the episode credible: the scree plot
        is not wrong in the easy case."""
        row = self._sweep()[0]
        for rule in ("elbow", "noise_edge", "optimal_threshold", "parallel"):
            assert row[f"{rule}_exact"] > 0.9, rule

    def test_the_optimal_threshold_gives_up_where_the_elbow_still_works(self):
        """Not a bug: it minimises reconstruction error, and a direction barely
        above the noise adds more noise than signal to a reconstruction."""
        row = self._sweep()[1]
        assert row["optimal_threshold_exact"] < 0.1
        assert row["elbow_exact"] > 0.8

    def test_below_the_transition_nothing_recovers_the_rank(self):
        """The honest limit. It is not that the rules are bad; the information
        is not in the matrix."""
        weak = self._sweep()[2]
        for rule in ("elbow", "noise_edge", "optimal_threshold", "parallel"):
            assert weak[f"{rule}_exact"] < 0.2, rule

    def test_only_the_elbow_cannot_say_there_is_nothing_there(self):
        """The thesis. On a matrix with no structure at all the calibrated rules
        return zero, which is correct; the elbow returns an answer, and its
        answers are spread over most of the range available."""
        noise = self._sweep()[3]
        for rule in ("noise_edge", "optimal_threshold", "parallel"):
            assert noise[rule] == 0.0, f"{rule} should report no structure"
        assert noise["elbow"] >= 1.0
        assert noise["elbow_spread"] > 10.0

    def test_the_no_signal_row_is_scored_against_zero(self):
        """Scoring it against the rank that was not put in would credit the elbow
        for its accidents and penalise the rules that got it right."""
        noise = self._sweep()[3]
        assert noise["truth"] == 0
        assert noise["elbow_exact"] == 0.0
        # The two thresholds and the permutation reference return nothing most of
        # the time. Not always: counting singular values above the noise edge
        # inherits the edge's own false-positive rate, which is the ~12% measured
        # in test_the_noise_edge_is_a_concentration_point_not_a_bound.
        assert noise["optimal_threshold_exact"] == 1.0
        for rule in ("noise_edge", "parallel"):
            assert noise[f"{rule}_exact"] > 0.8, rule

    def test_the_elbow_holds_still_when_the_signal_is_strong(self):
        """So the spread above is a property of the regime, not of the measure."""
        assert self._sweep()[0]["elbow_spread"] == 0.0


class TestTheThresholds:
    def test_the_noise_edge_is_a_concentration_point_not_a_bound(self):
        rng = np.random.default_rng(2)
        n, p, sigma = 400, 20, 1.5
        edge = rk.noise_edge(n, p, sigma)
        tops = np.array([np.linalg.svd(rng.normal(0, sigma, (n, p)),
                                       compute_uv=False)[0] for _ in range(200)])
        # A concentration point, not a bound: about an eighth of draws cross it.
        assert 0.02 < (tops > edge).mean() < 0.3
        assert 0.90 < tops.min() / edge and tops.max() / edge < 1.10

    def test_the_edge_scales_with_sigma_and_with_the_shape(self):
        assert rk.noise_edge(400, 20, 2.0) == pytest.approx(
            2 * rk.noise_edge(400, 20, 1.0))
        assert rk.noise_edge(400, 100, 1.0) > rk.noise_edge(400, 20, 1.0)

    @pytest.mark.parametrize("beta", [0.0, -0.5, 1.5])
    def test_an_impossible_aspect_ratio_is_refused(self, beta):
        with pytest.raises(ValueError, match="beta"):
            rk.gavish_donoho_lambda(beta)

    def test_lambda_is_four_over_root_three_for_a_square_matrix(self):
        """The paper's title, and a check that the constant is the right one."""
        assert rk.gavish_donoho_lambda(1.0) == pytest.approx(4 / np.sqrt(3),
                                                            rel=1e-12)

    def test_the_optimal_threshold_sits_above_the_naive_edge(self):
        assert rk.optimal_threshold(400, 20, 1.0) > rk.noise_edge(400, 20, 1.0)

    def test_it_does_not_care_which_way_round_the_matrix_is(self):
        assert rk.optimal_threshold(400, 20, 1.0) == pytest.approx(
            rk.optimal_threshold(20, 400, 1.0))


class TestConstructionAndElbow:
    def test_the_requested_singular_values_are_what_you_get(self):
        rng = np.random.default_rng(3)
        want = [50.0, 30.0, 10.0]
        Y = rk.low_rank_plus_noise(600, 15, singular_values=want, sigma=1e-9,
                                   rng=rng)
        assert np.linalg.svd(Y, compute_uv=False)[:3] == pytest.approx(want,
                                                                      rel=1e-6)

    def test_random_factors_would_have_inflated_them(self):
        """Why the module uses orthonormal factors: the obvious construction
        multiplies every singular value by about sqrt(n p) and makes every rule
        below look infallible."""
        rng = np.random.default_rng(4)
        n, p = 600, 15
        U, V = rng.standard_normal((n, 3)), rng.standard_normal((p, 3))
        naive = np.linalg.svd(U @ np.diag([50.0, 30.0, 10.0]) @ V.T,
                              compute_uv=False)[0]
        assert naive > 20 * 50.0

    def test_the_elbow_finds_an_obvious_step(self):
        assert rk.elbow([100.0, 99.0, 98.0, 3.0, 2.0, 1.0]) == 3

    def test_it_returns_something_for_a_degenerate_spectrum(self):
        assert rk.elbow([5.0]) == 1
