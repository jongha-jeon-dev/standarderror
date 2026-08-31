"""Tests for the first-loss tranche primitives.

The split these tests are built around is the one the module is built around: the
attachment point is exact and gets exact assertions, while the expected loss is a
model and gets agreement-with-Monte-Carlo plus monotonicity and limit checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.credit import (
    attachment_point,
    expected_shortfall_rate,
    required_fall,
    simulate_shortfall_rate,
)


class TestAttachment:
    def test_no_senior_debt_and_full_recovery_is_the_deposit_ratio(self):
        # Nothing ahead of the claim and no liquidation discount: the collateral
        # can fall by exactly the unencumbered fraction.
        assert attachment_point(0.60, 1.0) == pytest.approx(0.60)
        assert required_fall(0.60, 1.0) == pytest.approx(40.0)

    def test_the_two_headline_cohorts(self):
        # The exp010 numbers, asserted here so a refactor cannot quietly move them.
        # Seoul apartment, January 2026: 50.92% deposit ratio, 101% at auction.
        assert required_fall(0.5092, 1.01) == pytest.approx(49.58, abs=0.01)
        # Seoul villa, December 2022: 78.6% deposit ratio, 79% at auction.
        assert required_fall(0.786, 0.79) == pytest.approx(0.51, abs=0.01)

    def test_senior_debt_eats_the_cushion_one_for_one(self):
        # At full recovery the attachment point is the sum of the claims, so every
        # point of senior debt is a point off the junior claim's cushion.
        base = required_fall(0.50, 1.0)
        assert required_fall(0.50, 1.0, 0.20) == pytest.approx(base - 20.0)

    def test_already_impaired_reads_as_a_negative_fall(self):
        # 80% deposit against a 70% liquidation ratio: short before anything moves.
        assert required_fall(0.80, 0.70) < 0

    def test_recovery_above_par_is_allowed(self):
        # Seoul apartments cleared above appraised value every month from March
        # 2026. Rejecting that would force the caller to misreport the input.
        assert attachment_point(0.60, 1.05) < 0.60

    @pytest.mark.parametrize("kwargs", [
        {"junior": 0.0, "recovery": 1.0},
        {"junior": -0.1, "recovery": 1.0},
        {"junior": 0.5, "recovery": 0.0},
        {"junior": 0.5, "recovery": 2.5},
    ])
    def test_nonsense_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            attachment_point(**kwargs)

    def test_negative_senior_claim_raises(self):
        with pytest.raises(ValueError):
            attachment_point(0.5, 1.0, senior=-0.1)


class TestExpectedShortfall:
    def test_agrees_with_monte_carlo(self):
        for junior, recovery, sigma in ((0.5092, 1.01, 0.18), (0.65, 0.854, 0.12),
                                        (0.786, 0.79, 0.12), (0.90, 0.79, 0.05)):
            exact = expected_shortfall_rate(junior, recovery, sigma, term=2.0)
            sim = simulate_shortfall_rate(junior, recovery, sigma, term=2.0,
                                          n_draws=400_000, seed=7)
            assert sim["loss_per_year_pct"] == pytest.approx(
                exact["loss_per_year_pct"], abs=0.02)
            assert sim["p_breach_pct"] == pytest.approx(
                exact["p_breach_pct"], abs=0.5)

    def test_loss_rises_with_the_junior_claim(self):
        losses = [expected_shortfall_rate(d, 0.79, 0.12, term=2.0)[
            "loss_per_year_pct"] for d in (0.40, 0.55, 0.65, 0.786, 0.90)]
        assert all(b > a for a, b in zip(losses, losses[1:]))

    def test_loss_rises_with_volatility(self):
        losses = [expected_shortfall_rate(0.65, 0.854, s, term=2.0)[
            "loss_per_year_pct"] for s in (0.05, 0.08, 0.12, 0.18)]
        assert all(b > a for a, b in zip(losses, losses[1:]))

    def test_far_out_of_the_money_is_effectively_free(self):
        # A 50% deposit at par recovery needs a halving of the collateral. At index
        # volatility over two years that is not a risk anyone should price.
        res = expected_shortfall_rate(0.50, 1.01, 0.05, term=2.0)
        assert res["loss_per_year_pct"] < 1e-6
        assert res["loss_per_year_pct"] >= 0.0        # never a negative loss

    def test_low_volatility_limit_follows_the_exact_threshold(self):
        # As sigma shrinks the breach probability has to go to 0 when the tranche is
        # out of the money and to 100 when it is already in. This is the boundary
        # between the exact half of the module and the modelled half, so it is the
        # one place the two must line up.
        out = expected_shortfall_rate(0.786, 0.79, 1e-4, term=2.0)
        inside = expected_shortfall_rate(0.900, 0.79, 1e-4, term=2.0)
        assert required_fall(0.786, 0.79) > 0 and out["p_breach_pct"] < 1e-6
        assert required_fall(0.900, 0.79) < 0 and inside["p_breach_pct"] > 99.999

    def test_half_a_percent_of_cushion_survives_no_volatility_at_all(self):
        # The finding exp010 is built on. The December 2022 Seoul villa cohort sat
        # 0.51% from its attachment point, so even a preposterously quiet market
        # leaves a material breach probability: half a percent is under two
        # standard deviations at 0.2% annual volatility over two years.
        res = expected_shortfall_rate(0.786, 0.79, 0.002, term=2.0)
        assert 1.0 < res["p_breach_pct"] < 10.0

    def test_wipeout_needs_senior_debt(self):
        # With nothing ahead of it the junior claim can be impaired but never
        # wiped out by a finite price: some recovery always reaches it.
        assert expected_shortfall_rate(0.786, 0.79, 0.20, term=2.0)[
            "p_wipeout_pct"] == 0.0
        assert expected_shortfall_rate(0.60, 0.79, 0.20, senior=0.30, term=2.0)[
            "p_wipeout_pct"] > 0.0

    def test_per_year_and_total_are_consistent(self):
        res = expected_shortfall_rate(0.70, 0.85, 0.12, term=2.0)
        assert res["loss_total_pct"] == pytest.approx(
            2.0 * res["loss_per_year_pct"])

    def test_positive_drift_reduces_the_loss(self):
        flat = expected_shortfall_rate(0.70, 0.85, 0.12, term=2.0, drift=0.0)
        up = expected_shortfall_rate(0.70, 0.85, 0.12, term=2.0, drift=0.05)
        assert up["loss_per_year_pct"] < flat["loss_per_year_pct"]

    def test_zero_volatility_and_bad_term_raise(self):
        with pytest.raises(ValueError):
            expected_shortfall_rate(0.7, 0.85, 0.0, term=2.0)
        with pytest.raises(ValueError):
            expected_shortfall_rate(0.7, 0.85, 0.12, term=0.0)

    def test_attachment_matches_the_exact_function(self):
        res = expected_shortfall_rate(0.65, 0.854, 0.12, senior=0.10, term=2.0)
        assert res["attachment"] == pytest.approx(
            attachment_point(0.65, 0.854, 0.10))
        assert res["required_fall_pct"] == pytest.approx(
            required_fall(0.65, 0.854, 0.10))


class TestRolloverArithmetic:
    """The linear mechanism, which is plain algebra and lives in the experiment.

    Kept here because the identity is what the post's Fig 3 asserts: with the ratio
    unchanged the refund gap is exactly the price fall, and with prices unchanged it
    is exactly the proportional fall in the ratio.
    """

    @staticmethod
    def gap(ratio_now, ratio_then, price_change_pct):
        v = 1.0 + price_change_pct / 100.0
        return 100.0 * max(0.0, ratio_then - ratio_now * v) / ratio_then

    def test_ratio_held_makes_the_gap_the_price_fall(self):
        assert self.gap(78.6, 78.6, -10.0) == pytest.approx(10.0)
        assert self.gap(78.6, 78.6, -20.0) == pytest.approx(20.0)

    def test_flat_prices_make_the_gap_the_ratio_move(self):
        # 78.6% to 65.4% is a 16.79% proportional fall, and with the house unchanged
        # that is exactly what the landlord has to find in cash.
        assert self.gap(65.4, 78.6, 0.0) == pytest.approx(
            100.0 * (78.6 - 65.4) / 78.6)
        assert self.gap(65.4, 78.6, 0.0) == pytest.approx(16.79, abs=0.01)

    def test_rising_prices_can_close_the_gap_entirely(self):
        assert self.gap(65.4, 78.6, 25.0) == 0.0


def test_no_numpy_warnings_on_a_zero_senior_claim():
    """`log(0)` in the wipeout threshold used to emit a divide-by-zero warning.

    The threshold is genuinely zero when nothing ranks ahead of the junior claim,
    which is the *default* case here, so a warning on the common path would train
    everyone to ignore warnings.
    """
    with np.errstate(all="raise"):
        expected_shortfall_rate(0.786, 0.79, 0.12, senior=0.0, term=2.0)
