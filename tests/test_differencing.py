"""What a finite difference does when you shrink the step, and what it costs.

Two of these tests exist because the first version of the module was wrong:
`is_complex_safe` returned True for `x * abs(x)`, whose complex-step derivative
is off by exactly a factor of two, and the module docstring asserted that a ReLU
network cannot be differentiated by a complex step, which it can.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.numerics import differencing as fd


class TestTheUCurve:
    def _sweep(self, kind):
        hs = 10.0 ** -np.arange(1, 17)
        return fd.error_sweep(np.sin, np.cos, 1.0, hs, kind=kind)

    @pytest.mark.parametrize("kind", ["forward", "central"])
    def test_the_error_is_not_monotone_in_the_step(self, kind):
        """The whole episode. "A smaller step is more accurate" is not merely
        imprecise -- it has the wrong shape."""
        assert not self._sweep(kind).is_monotone

    def test_the_forward_optimum_sits_near_the_square_root_of_eps(self):
        s = self._sweep("forward")
        assert s.best_h == pytest.approx(1e-8)
        # Within a decade of the derivation's scale, which is what an
        # order-of-magnitude balance can promise.
        assert 0.1 < s.best_h / fd.optimal_h("forward") < 10.0

    def test_the_central_optimum_sits_near_the_cube_root_of_eps(self):
        """Not the square root. The square root is the *forward* difference's
        scale, and quoting it for a central difference is the common error."""
        s = self._sweep("central")
        assert s.best_h == pytest.approx(1e-5)
        assert 0.1 < s.best_h / fd.optimal_h("central") < 10.0
        assert fd.optimal_h("central") > 100 * fd.optimal_h("forward")

    def test_each_scheme_reaches_about_its_predicted_floor(self):
        for kind, tol in (("forward", 30.0), ("central", 30.0)):
            s = self._sweep(kind)
            assert 1 / tol < s.best_error / fd.error_floor(kind) < tol, kind

    def test_the_central_difference_is_two_orders_better_at_its_own_optimum(self):
        f, c = self._sweep("forward"), self._sweep("central")
        assert 100 < f.best_error / c.best_error < 1000

    def test_but_not_at_the_same_step(self):
        """Its advantage is that it is *allowed* a larger step, not that it is
        more accurate at a given one -- at 1e-8 the two agree to a factor of 1.2."""
        hs = np.array([1e-8])
        a = fd.error_sweep(np.sin, np.cos, 1.0, hs, kind="forward").best_error
        b = fd.error_sweep(np.sin, np.cos, 1.0, hs, kind="central").best_error
        assert 0.5 < a / b < 2.0

    def test_refining_past_the_optimum_costs_orders_of_magnitude(self):
        assert self._sweep("central").penalty_at(1e-16) > 1e8

    def test_richardson_buys_truncation_and_not_roundoff(self):
        """Which is the general lesson about extrapolation."""
        h = 1e-3
        plain = abs(fd.central(np.sin, 1.0, h) - np.cos(1.0))
        rich = abs(fd.richardson(np.sin, 1.0, h) - np.cos(1.0))
        assert rich < plain / 1000
        # And down where cancellation dominates it is no help at all.
        tiny = 1e-14
        assert abs(fd.richardson(np.sin, 1.0, tiny) - np.cos(1.0)) > 1e-4

    def test_an_unknown_scheme_is_refused(self):
        with pytest.raises(ValueError, match="unknown scheme"):
            fd.optimal_h("backward")
        with pytest.raises(ValueError, match="unknown scheme"):
            fd.error_floor("five-point")


class TestTheComplexStep:
    def test_it_is_exact_at_a_step_no_difference_could_use(self):
        for h in (1e-20, 1e-100, 1e-200):
            assert fd.complex_step(np.sin, 1.0, h) == pytest.approx(
                np.cos(1.0), abs=1e-15), h

    def test_it_has_no_optimum(self):
        """Ten decades of step size, no U."""
        hs = 10.0 ** -np.arange(4, 41, 4)
        errs = [abs(fd.complex_step(np.exp, 0.7, h) - np.exp(0.7)) for h in hs]
        assert max(errs[2:]) < 1e-14, errs

    def test_abs_breaks_it_quietly(self):
        """Returns 0.0 for the derivative of |x| at 1.5, which is 1.0. No
        exception, no warning."""
        assert fd.complex_step(np.abs, 1.5, 1e-20) == 0.0
        assert not fd.is_complex_safe(np.abs, 1.5)

    def test_a_real_part_cast_breaks_it(self):
        f = lambda z: np.real(z) ** 2                       # noqa: E731
        assert fd.complex_step(f, 1.5, 1e-20) == 0.0
        assert not fd.is_complex_safe(f, 1.5)

    def test_the_factor_of_two_case_the_first_guard_missed(self):
        """`x*abs(x)` preserves the real part perfectly, so the cheap check
        passes it, and the derivative comes back wrong by exactly two. This is
        why `is_complex_safe` corroborates against a central difference."""
        f = lambda z: z * np.abs(z)                         # noqa: E731
        got, truth = fd.complex_step(f, 1.5, 1e-20), 2 * 1.5
        assert got == pytest.approx(truth / 2)
        assert not fd.is_complex_safe(f, 1.5)

    def test_a_relu_survives_it_which_is_the_surprise(self):
        """`np.maximum` compares complex numbers by real part first, so a ReLU
        network differentiates correctly away from the kink. The module docstring
        originally claimed the opposite."""
        rng = np.random.default_rng(1)
        W1, b1, w2 = (rng.standard_normal((4, 3)), rng.standard_normal(4),
                      rng.standard_normal(4))

        def net(x):
            v = np.array([x, 0.3, -0.7], dtype=complex)
            return w2 @ np.maximum(W1 @ v + b1, 0)

        for x in (0.5, -0.5, 2.0):
            cs = fd.complex_step(net, x, 1e-20)
            ce = fd.central(lambda z: float(np.real(net(z))), x, 1e-5)
            assert cs == pytest.approx(ce, abs=1e-9), x

    def test_the_guard_accepts_the_analytic_cases(self):
        for f in (np.sin, np.exp, np.cos, lambda z: z ** 3 - 2 * z):
            assert fd.is_complex_safe(f, 1.5), f


class TestTheConditionNumber:
    def test_it_is_the_same_quantity_episode_one_computed(self):
        """`|x f'/f|`, known before any arithmetic happens."""
        assert fd.condition_number(np.sin, np.cos, 1.0) == pytest.approx(
            abs(1.0 * np.cos(1.0) / np.sin(1.0)))

    def test_it_is_infinite_where_the_function_vanishes(self):
        assert fd.condition_number(np.sin, np.cos, 0.0) == float("inf")

    def test_a_badly_conditioned_evaluation_can_have_an_easy_derivative(self):
        """The boundary of what this number says. `x - 1` near 1 amplifies a
        relative input error ten thousand-fold, and its derivative comes back
        exact, because a linear function has no truncation error to trade."""
        g = lambda t: t - 1.0                               # noqa: E731
        assert fd.condition_number(g, lambda t: 1.0, 1.0001) == pytest.approx(
            1e4, rel=1e-3)
        hs = 10.0 ** -np.arange(1, 17)
        s = fd.error_sweep(g, lambda t: 1.0, 1.0001, hs, kind="central")
        assert s.best_error < 1e-15

    def test_a_well_conditioned_problem_can_have_an_unstable_algorithm(self):
        """`1 - cos x` at 1e-4: condition number 2, and the obvious evaluation
        loses seven digits. Higham's distinction, and the one the first draft of
        this module got wrong."""
        got = fd.cancellation_pair(1e-4)
        assert got["condition_number"] == pytest.approx(2.0, rel=1e-6)
        assert got["naive_relative_error"] > 1e-10
        assert got["rewritten_relative_error"] < 1e-15
        assert (got["naive_relative_error"]
                / got["rewritten_relative_error"]) > 1e6

    def test_the_rewrite_is_the_same_function(self):
        """So the improvement is not a different answer, it is the same answer
        with its digits intact."""
        for x in (1e-2, 1e-4, 1e-6):
            got = fd.cancellation_pair(x)
            assert got["rewritten"] == pytest.approx(got["reference"], rel=1e-14)


class TestTheGradientCheck:
    def _design(self):
        return fd.gradient_check_design()

    def test_the_exact_gradient_passes_at_the_optimum(self):
        """So a failing check below is the check failing, not the gradient."""
        loss, grad, v = self._design()
        assert fd.gradient_check(loss, grad, v, fd.optimal_h("central")) < 1e-9

    def test_the_step_decides_what_the_check_can_detect(self):
        """Eight orders of magnitude, set by a constant nobody writes down."""
        loss, grad, v = self._design()
        at_optimum = fd.smallest_detectable_bug(loss, grad, v, 1e-5, index=2)
        far_below = fd.smallest_detectable_bug(loss, grad, v, 1e-13, index=2)
        assert at_optimum["detectable"] < 1e-8
        assert far_below["detectable"] > 1e-2
        assert far_below["detectable"] / at_optimum["detectable"] > 1e6

    def test_a_ten_percent_gradient_bug_is_invisible_at_a_bad_step(self):
        """The sentence the episode is built to earn."""
        loss, grad, v = self._design()
        got = fd.smallest_detectable_bug(loss, grad, v, 1e-13, index=2)
        assert got["detectable"] > 0.05

    def test_the_noise_floor_rises_as_the_step_falls(self):
        loss, grad, v = self._design()
        floors = [fd.smallest_detectable_bug(loss, grad, v, h, index=2)["floor"]
                  for h in (1e-5, 1e-9, 1e-11, 1e-13)]
        assert floors == sorted(floors)

    def test_the_common_defaults_are_past_the_optimum_but_still_work(self):
        """Being fair to the practice: 1e-7 is a decade and a half below the
        optimum and still resolves a bug of about 1e-7."""
        loss, grad, v = self._design()
        got = fd.smallest_detectable_bug(loss, grad, v, 1e-7, index=2)
        assert got["detectable"] < 1e-6

    def test_it_reports_when_nothing_at_all_is_detectable(self):
        loss, grad, v = self._design()
        got = fd.smallest_detectable_bug(loss, grad, v, 1e-17, index=2)
        assert np.isnan(got["detectable"]) or got["detectable"] > 0.5


class TestTheBfloat16Quantiser:
    """Written by hand rather than imported, so it needs checking against the
    thing it claims to be."""

    def test_it_agrees_with_torch_on_three_thousand_values(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(0)
        vals = np.concatenate([
            rng.standard_normal(2000).astype(np.float32),
            (rng.standard_normal(500) * 1e-6).astype(np.float32),
            (rng.standard_normal(500) * 1e6).astype(np.float32)])
        mine = fd.to_bfloat16(vals)
        theirs = (torch.tensor(vals).to(torch.bfloat16)
                  .to(torch.float32).numpy())
        assert np.array_equal(mine, theirs)

    def test_its_epsilon_is_two_to_the_minus_seven(self):
        """The gap convention, matching `np.finfo`. Unit roundoff -- 2**-8 --
        is a different number and mixing the two costs a factor of two."""
        one = np.float32(1.0)
        assert float(fd.to_bfloat16(one + np.float32(2.0 ** -7))) > 1.0
        assert float(fd.to_bfloat16(one + np.float32(2.0 ** -9))) == 1.0
        assert fd.EPS_BY_PRECISION["bfloat16"] == 2.0 ** -7

    def test_it_keeps_float32s_exponent_range(self):
        """Which is the entire reason the format exists: 1e-30 survives in
        bfloat16 and is zero in float16."""
        assert float(fd.to_bfloat16(np.float32(1e-30))) > 0.0
        assert float(np.float16(1e-30)) == 0.0

    def test_an_unknown_precision_is_refused(self):
        with pytest.raises(ValueError, match="unknown precision"):
            fd.quantise(1.0, "float8")


class TestThePrecisionTable:
    @pytest.fixture(scope="class")
    def rows(self):
        hs = 10.0 ** -np.arange(-1.5, 9, 0.25)
        return {r["precision"]: r
                for r in fd.precision_table(np.sin, np.cos, 1.0, hs)}

    def test_epsilon_spans_thirteen_orders_of_magnitude(self, rows):
        assert (rows["bfloat16"]["eps"] / rows["float64"]["eps"]) > 1e13

    def test_the_optimal_step_moves_with_the_cube_root_of_eps(self, rows):
        """The exponent transfers across all four formats, which is the claim
        that makes one derivation worth four measurements."""
        for name, r in rows.items():
            assert 0.05 < r["measured_h"] / r["predicted_h"] < 20, name

    def test_the_predicted_floor_is_optimistic_in_every_row(self, rows):
        """And by one to two decades, because the derivation drops the
        derivative constants. So the formula scales an expectation; it does not
        predict an error."""
        for name, r in rows.items():
            ratio = r["predicted_floor"] / r["measured_error"]
            assert ratio > 5.0, (name, ratio)

    def test_the_error_still_rises_by_ten_orders_across_the_formats(self, rows):
        assert (rows["bfloat16"]["measured_error"]
                / rows["float64"]["measured_error"]) > 1e10

    def test_a_step_below_the_smallest_subnormal_is_not_a_measurement(self):
        """It rounds to zero, and a zero step is undefined rather than
        inaccurate. `argmin` would have picked the NaN."""
        hs = np.array([1e-2, 1e-30])
        s = fd.precision_sweep(np.sin, np.cos, 1.0, hs, precision="float16")
        assert np.isnan(s.error[1])
        assert s.measured == 1
        assert s.best_h == 1e-2

    def test_an_unknown_scheme_is_refused_here_too(self):
        with pytest.raises(ValueError, match="unknown scheme"):
            fd.difference_in(np.sin, 1.0, 1e-5, precision="float32",
                             kind="backward")


class TestWhatTheCheckCanSeeInEachPrecision:
    """The episode's payoff, and the number I got wrong before measuring it."""

    @pytest.fixture(scope="class")
    def best(self):
        loss, grad, v = fd.gradient_check_design()
        hs = 10.0 ** -np.arange(-1.5, 9, 0.25)
        return {p: fd.best_detectable(loss, grad, v, hs, index=2, precision=p)
                for p in fd.PRECISIONS}

    def test_float64_resolves_a_ten_billionth(self, best):
        assert best["float64"]["detectable"] < 1e-9

    def test_float32_loses_five_orders_of_magnitude(self, best):
        assert 1e-5 < best["float32"]["detectable"] < 1e-4
        assert best["float32"]["detectable"] / best["float64"]["detectable"] > 1e4

    def test_a_bfloat16_check_cannot_see_a_ten_percent_error(self, best):
        """At *any* step, having searched. So in bfloat16 the gradient check is
        not a weak test, it is not a test."""
        assert best["bfloat16"]["detectable"] > 0.10
        assert best["float16"]["detectable"] > 0.05

    def test_deriving_that_number_from_eps_gets_it_wrong(self, best):
        """`eps**(2/3)` for bfloat16 is 3.9e-02 and the measured answer is
        1.7e-01. I wrote the derived one down first."""
        derived = fd.error_floor("central",
                                 eps=fd.EPS_BY_PRECISION["bfloat16"])
        assert derived == pytest.approx(3.9e-2, rel=0.05)
        assert best["bfloat16"]["detectable"] / derived > 3.0

    def test_the_span_is_nine_orders_of_magnitude(self, best):
        assert (best["bfloat16"]["detectable"]
                / best["float64"]["detectable"]) > 1e8

    def test_a_grid_that_detects_nothing_is_an_error_not_a_number(self):
        loss, grad, v = fd.gradient_check_design()
        with pytest.raises(ValueError, match="no step in the grid"):
            fd.best_detectable(loss, grad, v, [1e-30], precision="float16")
