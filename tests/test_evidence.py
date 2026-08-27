"""Tests for reading someone else's reported effect.

The arithmetic is short, so these are mostly about the two places it is easy to
mislead yourself: a p-value reported as an upper bound gives a *bound* rather than a
value, and breadth has a ceiling that arrives much sooner than intuition suggests.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.uq import evidence


class TestZForP:
    def test_known_values(self):
        assert evidence.z_for_p(0.05) == pytest.approx(1.959964, abs=1e-5)
        assert evidence.z_for_p(0.001) == pytest.approx(3.290527, abs=1e-5)
        assert evidence.z_for_p(0.05, two_sided=False) == pytest.approx(
            1.644854, abs=1e-5)

    def test_smaller_p_is_a_larger_z(self):
        zs = [evidence.z_for_p(p) for p in (0.05, 0.01, 0.001, 1e-6)]
        assert all(b > a for a, b in zip(zs, zs[1:]))

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_impossible_p(self, p):
        with pytest.raises(ValueError):
            evidence.z_for_p(p)


class TestPerObservationSharpe:
    def test_ratio(self):
        assert evidence.per_observation_sharpe(0.22, 11.0) == pytest.approx(0.02)

    def test_sign_is_preserved(self):
        """Collapsing the sign early is how a reversal gets reported as a profit."""
        assert evidence.per_observation_sharpe(-0.34, 10.0) < 0

    def test_rejects_non_positive_noise(self):
        for noise in (0.0, -1.0):
            with pytest.raises(ValueError):
                evidence.per_observation_sharpe(0.2, noise)


class TestImpliedN:
    def test_inverts_the_t_statistic(self):
        """n = (z sigma / effect)^2, so putting that n back must recover z."""
        effect, noise, p = 0.22, 10.56, 0.001
        n = evidence.implied_n(effect, noise, p)
        t = (effect / noise) * np.sqrt(n)
        assert t == pytest.approx(evidence.z_for_p(p), rel=1e-9)

    def test_the_paper_numbers(self):
        # The four positive tags in exp012, against a 10.56% window volatility.
        assert evidence.implied_n(0.35, 10.56, 0.001) == pytest.approx(9862, rel=1e-3)
        assert evidence.implied_n(0.22, 10.56, 0.001) == pytest.approx(24947, rel=1e-3)
        assert evidence.implied_n(0.10, 10.56, 0.012) == pytest.approx(70414, rel=1e-3)

    def test_a_smaller_effect_needs_a_bigger_sample(self):
        ns = [evidence.implied_n(e, 10.0, 0.01) for e in (0.4, 0.3, 0.2, 0.1)]
        assert all(b > a for a, b in zip(ns, ns[1:]))

    def test_scales_as_the_square_of_the_noise(self):
        a = evidence.implied_n(0.2, 10.0, 0.01)
        b = evidence.implied_n(0.2, 20.0, 0.01)
        assert b / a == pytest.approx(4.0, rel=1e-9)

    def test_sign_of_the_effect_does_not_matter(self):
        assert evidence.implied_n(-0.34, 10.0, 0.001) == pytest.approx(
            evidence.implied_n(0.34, 10.0, 0.001))

    def test_zero_effect_raises(self):
        with pytest.raises(ValueError):
            evidence.implied_n(0.0, 10.0, 0.01)


class TestEffectiveIndependent:
    def test_uncorrelated_is_the_count_itself(self):
        assert evidence.effective_independent(250, 0.0) == pytest.approx(250.0)

    def test_one_position_is_one_at_any_correlation(self):
        for rho in (0.0, 0.05, 0.5, 0.99):
            assert evidence.effective_independent(1, rho) == pytest.approx(1.0)

    def test_converges_to_one_over_rho(self):
        """The result the post turns on: breadth is capped by residual correlation."""
        for rho in (0.01, 0.02, 0.05, 0.2):
            assert evidence.effective_independent(10_000_000, rho) == pytest.approx(
                1.0 / rho, rel=1e-3)

    def test_the_ceiling_binds_early(self):
        """Within 10% of the limit by n = 10/rho, which is the surprising part."""
        for rho in (0.01, 0.05, 0.1):
            n = int(round(10.0 / rho))
            assert evidence.effective_independent(n, rho) >= 0.9 / rho

    def test_never_exceeds_the_ceiling(self):
        for rho in (0.01, 0.05, 0.3):
            for n in (2, 50, 5000, 10 ** 6):
                assert evidence.effective_independent(n, rho) <= 1.0 / rho + 1e-9

    def test_monotone_in_both_arguments(self):
        vals = [evidence.effective_independent(n, 0.05) for n in (5, 50, 500, 5000)]
        assert all(b > a for a, b in zip(vals, vals[1:]))
        by_rho = [evidence.effective_independent(500, r)
                  for r in (0.0, 0.01, 0.05, 0.2)]
        assert all(b < a for a, b in zip(by_rho, by_rho[1:]))

    @pytest.mark.parametrize("args", [(0, 0.1), (-5, 0.1), (10, 1.0), (10, -0.1)])
    def test_rejects_impossible_inputs(self, args):
        with pytest.raises(ValueError):
            evidence.effective_independent(*args)


class TestAnnualisedSharpe:
    def test_root_breadth_root_time(self):
        s = evidence.annualised_sharpe(0.02, 100, 0.0, periods_per_year=16.8)
        assert s == pytest.approx(0.02 * np.sqrt(100) * np.sqrt(16.8))

    def test_saturates_with_correlated_residuals(self):
        """Adding names past the ceiling buys nothing measurable."""
        a = evidence.annualised_sharpe(0.0208, 500, 0.05, periods_per_year=16.8)
        b = evidence.annualised_sharpe(0.0208, 50_000, 0.05, periods_per_year=16.8)
        assert b / a < 1.05

    def test_uncorrelated_keeps_climbing(self):
        a = evidence.annualised_sharpe(0.0208, 500, 0.0, periods_per_year=16.8)
        b = evidence.annualised_sharpe(0.0208, 50_000, 0.0, periods_per_year=16.8)
        assert b / a == pytest.approx(10.0, rel=1e-9)

    def test_the_posts_headline_pair(self):
        s = 0.22 / 10.56
        assert evidence.annualised_sharpe(s, 500, 0.0, periods_per_year=252 / 15) \
            == pytest.approx(1.909, abs=0.01)
        assert evidence.annualised_sharpe(s, 500, 0.05, periods_per_year=252 / 15) \
            == pytest.approx(0.375, abs=0.01)

    def test_negative_edge_gives_negative_sharpe(self):
        assert evidence.annualised_sharpe(-0.02, 100, 0.05,
                                         periods_per_year=16.8) < 0

    def test_rejects_non_positive_periods(self):
        with pytest.raises(ValueError):
            evidence.annualised_sharpe(0.02, 10, 0.05, periods_per_year=0.0)


class TestBreakevenCost:
    def test_is_the_gross_edge(self):
        assert evidence.breakeven_cost(0.22) == pytest.approx(0.22)

    def test_ignores_the_sign(self):
        """A short leg's edge is consumed by cost just the same."""
        assert evidence.breakeven_cost(-0.34) == pytest.approx(0.34)
