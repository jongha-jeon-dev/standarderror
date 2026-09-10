"""The composition identities, and the ordering that surprised me.

Every threshold here has a closed form derived in `composition.py`, so the
tests compare a bisected measurement against algebra rather than against a
previous run.
"""

import numpy as np
import pytest

from standarderror.aggregates import composition as cp

SIGMAS = (0.3, 0.4, 0.6, 0.8, 1.0)


class TestTheThresholds:
    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_the_median_flips_where_the_algebra_says(self, sigma):
        """Equation (2): `s* = 0.798 g / sigma`, from a slide of `m / 2` ranks
        against an order-statistic spacing of `1 / (n f(M))`."""
        flip = cp.cancelling_share(statistic="median", sigma=sigma, g=0.03)
        assert abs(flip.error) < 0.05, flip

    @pytest.mark.parametrize("sigma", (0.4, 0.6, 0.8))
    def test_the_mean_flips_where_its_own_algebra_says(self, sigma):
        flip = cp.cancelling_share(statistic="mean", sigma=sigma, g=0.03)
        assert abs(flip.error) < 0.02, flip

    @pytest.mark.parametrize("g", (0.01, 0.03, 0.05))
    def test_the_threshold_is_proportional_to_the_raise(self, g):
        """Doubling the raise doubles the churn needed to hide it, which is what
        makes the threshold quotable as a rule of thumb."""
        assert cp.median_threshold(g, 0.6) == pytest.approx(
            g / 0.03 * cp.median_threshold(0.03, 0.6))

    def test_a_more_unequal_distribution_is_more_fragile(self):
        """Backwards from the intuition, and it follows from the density: a wide
        distribution is thin at its median, so the same slide in rank travels
        further in money."""
        shares = [cp.median_threshold(0.03, s) for s in SIGMAS]
        assert shares == sorted(shares, reverse=True), shares


class TestTheOrderingIExpectedTheOtherWay:
    @pytest.fixture(scope="class")
    def sweep(self):
        return cp.fragility_sweep(SIGMAS, g=0.03)

    def test_the_median_breaks_before_the_mean_at_every_spread(self, sweep):
        """Measured ratios 0.61, 0.63, 0.58, 0.56, 0.53. "Use the median, it is
        robust" is a statement about outliers; composition is not an outlier
        problem, and against it the median is the weaker of the two."""
        for row in sweep:
            assert row["median_share"] < row["mean_share"], row
            assert 0.4 < row["ratio"] < 0.8, row

    def test_the_measured_median_shares_track_the_closed_form(self, sweep):
        for row in sweep:
            assert row["median_share"] == pytest.approx(
                row["median_predicted"], rel=0.05), row


class TestTheThreeNumberVersion:
    def test_everyone_gains_and_the_median_falls(self):
        """The version that fits in a sentence, kept as a test so the arithmetic
        in the prose cannot drift from the arithmetic in the module."""
        before = np.array([10.0, 20.0, 30.0])
        raised = before * 1.1
        after = np.concatenate([raised, [5.0, 5.0]])
        assert np.median(before) == 20.0
        assert np.median(raised) == 22.0
        assert np.median(after) == 11.0
        assert np.all(raised > before)


class TestTheOtherSign:
    def test_matched_growth_is_the_raise_and_nothing_else(self):
        row = cp.exit_effect(share=0.10, g=0.025)
        assert row["matched_growth"] == pytest.approx(0.025)

    def test_losing_the_lowest_paid_prints_a_pay_rise_nobody_got(self):
        """The 2020 sign. Ten percent of the lowest-paid leaving turns 2.5%
        underlying growth into a published median above ten percent."""
        row = cp.exit_effect(share=0.10, g=0.025)
        assert row["published_median_growth"] > 0.09
        assert row["published_median_growth"] > 4 * row["matched_growth"]

    def test_the_effect_is_monotone_in_the_share_that_leaves(self):
        got = [cp.exit_effect(share=s, g=0.025)["published_median_growth"]
               for s in (0.05, 0.10, 0.15, 0.20)]
        assert got == sorted(got), got

    def test_the_share_that_reproduces_the_published_figure(self):
        """Solved once for the episode: what job-loss share, concentrated at the
        bottom, prints the 10.4% the United States printed for median usual
        weekly earnings in the second quarter of 2020."""
        row = cp.share_implying(0.104, g=0.025)
        assert row["published_median_growth"] == pytest.approx(0.104, abs=1e-3)
        assert 0.05 < row["share"] < 0.15, row["share"]


class TestTheOrdinaryYear:
    def test_a_normal_year_of_hiring_eats_a_third_of_a_normal_raise(self):
        """The case that runs every year rather than once a century, and the
        one with no headline attached: net hiring of 1.5% below the median
        takes 1.15 points of a 3% raise."""
        row = cp.entry_effect(share=0.015, g=0.03)
        assert 0.3 < row["swallowed_fraction"] < 0.45, row
        assert row["printed"] < row["matched"]
        assert row["swallowed"] == pytest.approx(
            row["matched"] - row["printed"])

    def test_it_is_monotone_and_never_reverses_the_raise(self):
        rows = [cp.entry_effect(share=s, g=0.03)
                for s in (0.005, 0.010, 0.015, 0.020, 0.030)]
        fr = [r["swallowed_fraction"] for r in rows]
        assert fr == sorted(fr), fr
        # every one of these shares is below the flip, so the print is still
        # positive: this is the quiet version, not the sign-reversing one
        assert all(0.0 < r["printed"] < 0.03 for r in rows), fr

    def test_no_hiring_prints_the_raise(self):
        row = cp.entry_effect(share=0.0, g=0.03)
        assert row["printed"] == pytest.approx(0.03, abs=2e-3)


class TestTheMechanics:
    def test_entrants_above_the_mean_are_refused(self):
        with pytest.raises(ValueError, match="cannot pull"):
            cp.mean_threshold(0.03, 1.0)

    def test_a_sweep_row_does_not_depend_on_the_other_rows(self):
        """Seeded per row from its own parameters, so a two-spread sweep agrees
        with a five-spread one. The same bug has appeared twice elsewhere in
        this repository."""
        wide = {r["sigma"]: r["median_share"]
                for r in cp.fragility_sweep(SIGMAS, g=0.03)}
        narrow = cp.fragility_sweep((0.6,), g=0.03)[0]
        assert wide[0.6] == pytest.approx(narrow["median_share"], rel=1e-12)

    def test_the_wage_draw_is_in_units_of_its_own_median(self):
        w = cp.wages(50_000, sigma=0.6, seed=0)
        assert np.median(w) == pytest.approx(1.0, abs=0.02)
