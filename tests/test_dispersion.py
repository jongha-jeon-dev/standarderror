"""The Gini, checked against two implementations that are not the one used.

`dispersion.gini` uses the rank-weighted form. The mean-absolute-difference
form and the Lorenz integral are different arithmetic for the same quantity, so
agreeing with both is evidence about the code rather than about my algebra.
"""

import numpy as np
import pytest

from standarderror.aggregates import dispersion as dp


def gini_by_pairs(x):
    """`G = sum_ij |x_i - x_j| / (2 n^2 mu)`. O(n^2), so small samples only."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(np.abs(x[:, None] - x[None, :]).sum()
                 / (2.0 * n * n * x.mean()))


def gini_by_lorenz(x, points=200_001):
    """`G = 1 - 2 * integral of the Lorenz curve`."""
    p, curve = dp.lorenz(x, points=points)
    return float(1.0 - 2.0 * np.trapezoid(curve, p))


class TestTheCoefficientItself:
    def test_perfect_equality_is_zero(self):
        assert dp.gini(np.ones(1001)) == pytest.approx(0.0, abs=1e-12)

    def test_one_person_holding_everything_approaches_one(self):
        x = np.zeros(10_000)
        x[-1] = 1.0
        assert dp.gini(x) == pytest.approx(1.0, abs=2e-4)

    @pytest.mark.parametrize("seed", (0, 1, 2))
    def test_it_agrees_with_the_pairwise_form(self, seed):
        x = np.random.default_rng(seed).lognormal(0.0, 0.7, 400)
        assert dp.gini(x) == pytest.approx(gini_by_pairs(x), rel=1e-10)

    def test_it_agrees_with_the_lorenz_integral(self):
        x = dp.population(n=50_001)
        assert dp.gini(x) == pytest.approx(gini_by_lorenz(x), abs=2e-4)

    def test_it_is_scale_free(self):
        x = dp.population(n=20_001)
        assert dp.gini(x * 1000.0) == pytest.approx(dp.gini(x), rel=1e-12)


class TestTwoWorldsOneCoefficient:
    @pytest.fixture(scope="class")
    def pair(self):
        return dp.matched_pair()

    def test_the_two_worlds_share_a_gini(self, pair):
        assert pair["disagreement"] < 1e-12, pair["disagreement"]
        assert pair["gini"] > pair["base_gini"]

    def test_five_headline_statistics_cannot_tell_them_apart(self, pair):
        """Not only the Gini. Each distortion lives inside a tail, so the
        percentile ratios never touch it, and the poverty headcount does not
        move because both worlds leave the same people under the line."""
        ag = dp.agreement(dp.describe(pair["collapse"]),
                          dp.describe(pair["runaway"]))
        same = {k for k, v in ag.items() if v["same"]}
        assert same == {"gini", "p90_p10", "p50_p10", "p90_p50",
                        "headcount"}, same

    def test_and_the_poorest_tenth_holds_nine_times_more_in_one(self, pair):
        ag = dp.agreement(dp.describe(pair["collapse"]),
                          dp.describe(pair["runaway"]))
        assert ag["bottom10_share"]["ratio"] > 5.0
        assert ag["top1_share"]["ratio"] > 2.0
        # depth of poverty, which is the measure that does see it
        assert ag["gap_index"]["ratio"] < 0.4

    def test_the_runaway_multiple_is_solved_not_assumed(self, pair):
        rebuilt = dp.runaway(pair["base"], pair["b"])
        assert dp.gini(rebuilt) == pytest.approx(pair["gini"], abs=1e-12)


class TestWhyItCannotSeeIt:
    def test_the_sensitivity_is_linear_in_rank_and_nothing_else(self):
        """`dG/dx_k = 2k/(n^2 mu) - c`: a unit of income is priced by where the
        recipient stands in the queue, with the same slope everywhere, and not
        at all by what the money does to them."""
        r = dp.rank_sensitivity(dp.population(n=4001))
        assert r["r_squared"] > 0.999999, r["r_squared"]
        assert r["slope_ratio"] == pytest.approx(1.0, rel=1e-4), r
        # the same slope, over ranks whose incomes differ several-fold
        assert r["income_range"] > 4.0, r["income_range"]

    def test_the_perturbation_stays_below_the_local_spacing(self):
        """The guard on the measurement. A step comparable to the gap between
        neighbours reorders people, and then the rank distance in the formula
        is not the one you set -- which is how the first version of this
        measurement came out non-monotone."""
        r = dp.rank_sensitivity(dp.population(n=4001))
        assert r["step_over_spacing"] < 0.01, r["step_over_spacing"]

    def test_the_sensitivity_crosses_zero_once(self):
        """It has to: the coefficient is scale-free, so giving everyone a unit
        cannot change it, and a linear function summing to zero over ranks has
        exactly one crossing."""
        r = dp.rank_sensitivity(dp.population(n=4001))
        signs = [s > 0 for s in r["sensitivity"]]
        assert signs.count(True) > 0 and signs.count(False) > 0
        crossings = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        assert crossings == 1, crossings

    def test_it_refuses_a_population_too_small_to_fit(self):
        with pytest.raises(ValueError, match="at least four ranks"):
            dp.rank_sensitivity(dp.population(n=501), stride=200, margin=200)

    def test_destroying_the_poorest_tenth_barely_moves_it(self):
        """Because a summary weighted by income share cannot be sensitive to a
        group that holds 3.3% of it. One person in ten left with nothing moves
        the coefficient by 0.041."""
        row = dp.destroying_the_bottom()
        assert row["income_share_held"] < 0.05
        assert 0.03 < row["gini_change"] < 0.06, row


class TestThePovertyMeasures:
    def test_the_headcount_does_not_notice_the_poor_getting_poorer(self):
        """Sen's monotonicity, failed on purpose so the failure is on record."""
        x = dp.population(n=100_001)
        before = dp.poverty(x)
        worse = dp.collapse(x, 0.5)
        after = dp.poverty(worse)
        assert after["headcount"] == pytest.approx(before["headcount"],
                                                   abs=1e-3)
        assert after["gap_index"] > 1.5 * before["gap_index"]

    def test_the_gap_index_is_bounded_by_the_headcount(self):
        x = dp.population(n=50_001)
        p = dp.poverty(x)
        assert 0.0 <= p["gap_index"] <= p["headcount"]
        assert p["gap_index"] == pytest.approx(
            p["headcount"] * p["mean_shortfall"], rel=1e-9)


class TestTheMechanics:
    def test_the_lorenz_curve_starts_at_zero_and_ends_at_one(self):
        p, curve = dp.lorenz(dp.population(n=20_001))
        assert curve[0] == pytest.approx(0.0)
        assert curve[-1] == pytest.approx(1.0)
        assert np.all(np.diff(curve) >= -1e-12)
        assert np.all(curve <= p + 1e-12)

    def test_a_population_does_not_depend_on_being_asked_for_twice(self):
        a = dp.population(n=20_001, seed=7)
        b = dp.population(n=20_001, seed=7)
        assert np.array_equal(a, b)
