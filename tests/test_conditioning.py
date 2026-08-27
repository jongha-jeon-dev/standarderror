"""Tests for the conditioning module behind Linear Algebra 1.

A lecture's code carries a second obligation the research posts do not: a reader
is going to copy it. So the properties pinned here are the ones a reader would
be misled by if they broke — the bound really bounding, the residual really
staying small while the answer goes wrong, and the two polynomial bases really
spanning the same space.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.linalg import conditioning as cn


class TestHilbert:
    def test_it_is_the_textbook_matrix(self):
        H = cn.hilbert(3)
        expected = np.array([[1, 1 / 2, 1 / 3],
                             [1 / 2, 1 / 3, 1 / 4],
                             [1 / 3, 1 / 4, 1 / 5]])
        assert np.allclose(H, expected)

    def test_it_is_symmetric_and_positive_definite(self):
        H = cn.hilbert(8)
        assert np.allclose(H, H.T)
        assert np.linalg.eigvalsh(H).min() > 0

    def test_the_condition_number_grows_roughly_geometrically(self):
        ks = [cn.condition_number(cn.hilbert(n)) for n in (4, 6, 8, 10)]
        ratios = [b / a for a, b in zip(ks, ks[1:])]
        # Each two rows costs about three orders of magnitude.
        assert all(200 < r < 5000 for r in ratios), ratios

    def test_it_runs_out_of_double_precision_around_twelve(self):
        assert cn.digits_lost(cn.hilbert(12)) > cn.DIGITS_AVAILABLE
        assert cn.digits_lost(cn.hilbert(8)) < cn.DIGITS_AVAILABLE

    def test_it_rejects_a_degenerate_size(self):
        with pytest.raises(ValueError):
            cn.hilbert(0)


class TestConditionNumber:
    def test_the_identity_is_perfectly_conditioned(self):
        assert cn.condition_number(np.eye(5)) == pytest.approx(1.0)

    def test_it_is_the_ratio_of_the_singular_values(self):
        A = np.diag([4.0, 2.0, 0.5])
        assert cn.condition_number(A) == pytest.approx(8.0)

    def test_an_algebraically_singular_matrix_is_finite_in_floating_point(self):
        # [[1, 2], [2, 4]] has rank one on paper. In doubles its smaller
        # singular value is about 1e-16 rather than zero, so kappa comes out
        # enormous and finite. This is why "is it singular?" is the wrong
        # question and the condition number is the right one -- the answer to the
        # first is almost always no.
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        k = cn.condition_number(A)
        assert np.isfinite(k) and k > 1e15
        assert cn.digits_lost(A) > cn.DIGITS_AVAILABLE

    def test_an_exactly_zero_singular_value_gives_infinity(self):
        # Reachable when a column really is all zeros, which a rank check does
        # catch. Returned rather than raised so a caller sweeping a parameter
        # does not need a try block.
        A = np.array([[1.0, 0.0], [2.0, 0.0]])
        assert cn.condition_number(A) == float("inf")
        assert cn.digits_lost(A) == float("inf")

    def test_it_is_invariant_to_scaling_the_whole_matrix(self):
        # kappa is a ratio, so multiplying everything by a constant changes
        # nothing. Scaling one *column* does change it, which is the point of
        # equilibrate.
        A = cn.hilbert(6)
        assert cn.condition_number(1e6 * A) == pytest.approx(
            cn.condition_number(A), rel=1e-8)

    def test_digits_lost_is_log10_of_it(self):
        A = np.diag([1e6, 1.0])
        assert cn.digits_lost(A) == pytest.approx(6.0)


class TestTheBoundIsAPrediction:
    """The claim the episode rests on: this is not a loose inequality."""

    @pytest.mark.parametrize("n", [6, 8, 10])
    def test_a_random_perturbation_never_exceeds_the_bound(self, n):
        r = cn.perturb_and_solve(cn.hilbert(n), relative=1e-10, reps=200)
        assert r["worst"] <= r["bound"]

    @pytest.mark.parametrize("n", [6, 8, 10])
    def test_and_gets_close_to_it(self, n):
        r = cn.perturb_and_solve(cn.hilbert(n), relative=1e-10, reps=200)
        assert r["tightness"] > 0.4, r

    def test_the_typical_direction_is_well_inside_the_worst_one(self):
        # Worth pinning because it is the honest caveat: one draw looks
        # reassuring, and the bound is about the direction you did not draw.
        r = cn.perturb_and_solve(cn.hilbert(10), relative=1e-10, reps=200)
        assert r["typical"] < r["worst"] / 2

    def test_a_well_conditioned_system_barely_moves(self):
        r = cn.perturb_and_solve(np.eye(6), relative=1e-10, reps=50)
        assert r["worst"] < 2e-10

    def test_the_bound_scales_with_the_perturbation(self):
        H = cn.hilbert(8)
        assert cn.perturbation_bound(H, 1e-10) == pytest.approx(
            1e4 * cn.perturbation_bound(H, 1e-14))

    def test_a_negative_perturbation_is_rejected(self):
        with pytest.raises(ValueError):
            cn.perturbation_bound(np.eye(3), -1e-10)

    def test_reps_must_be_positive(self):
        with pytest.raises(ValueError):
            cn.perturb_and_solve(np.eye(3), reps=0)


class TestTheResidualDoesNotWarnYou:
    """The subtlety that makes this failure silent, and the reason for the module."""

    @pytest.mark.parametrize("n", [4, 8, 12, 14])
    def test_the_residual_stays_at_machine_precision_throughout(self, n):
        r = cn.solve_report(cn.hilbert(n))
        assert r.residual < 1e-13, (n, r)

    def test_while_the_answer_becomes_worthless(self):
        good = cn.solve_report(cn.hilbert(4))
        bad = cn.solve_report(cn.hilbert(14))
        assert good.error < 1e-10
        assert bad.error > 0.1
        # Both residuals are tiny: the check people run cannot tell them apart.
        assert bad.residual < 1e-13 and good.residual < 1e-13

    def test_digits_correct_plus_digits_lost_is_about_the_precision_available(self):
        for n in (4, 6, 8, 10):
            r = cn.solve_report(cn.hilbert(n))
            total = r.digits_correct + r.digits_lost
            assert cn.DIGITS_AVAILABLE - 2 < total < cn.DIGITS_AVAILABLE + 3, (n, total)

    def test_digits_correct_is_floored_at_zero(self):
        assert cn.solve_report(cn.hilbert(16)).digits_correct >= 0.0

    def test_forming_the_inverse_is_measurably_worse(self):
        for n in (8, 10):
            a = cn.solve_report(cn.hilbert(n), method="solve")
            b = cn.solve_report(cn.hilbert(n), method="inv")
            assert b.error > 5 * a.error, (n, a.error, b.error)

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValueError):
            cn.solve_report(np.eye(3), method="pinv")

    def test_a_non_square_matrix_is_rejected(self):
        with pytest.raises(ValueError):
            cn.solve_report(np.ones((3, 2)))

    def test_it_uses_the_solution_it_was_given(self):
        x = np.array([2.0, -3.0, 0.5])
        r = cn.solve_report(cn.hilbert(3), x_true=x)
        assert r.error < 1e-10
        assert len(r.row()) == 6


class TestBasisChoiceIsANumericalDecision:
    def test_the_two_bases_span_the_same_space(self):
        # The whole argument depends on this: if the fits differed, the Legendre
        # basis would be a different model rather than a better parameterisation.
        Xm = cn.design_matrix(9, basis="monomial")
        Xl = cn.design_matrix(9, basis="legendre")
        t = np.linspace(0, 1, Xm.shape[0])
        y = np.sin(6 * t) + 0.3 * t ** 2
        pm = Xm @ np.linalg.lstsq(Xm, y, rcond=None)[0]
        pl = Xl @ np.linalg.lstsq(Xl, y, rcond=None)[0]
        assert np.allclose(pm, pl, atol=1e-8)

    def test_the_monomial_gram_matrix_is_the_hilbert_matrix(self):
        # Sampled rather than integrated, so they agree to a couple of digits
        # rather than exactly -- but they are the same object, which is why a
        # polynomial fit meets a textbook pathology by accident.
        for degree in (3, 5, 7):
            sampled = cn.gram_condition(degree, n_points=4000)
            integrated = cn.condition_number(cn.hilbert(degree + 1))
            assert 0.3 < sampled / integrated < 3.0, (degree, sampled, integrated)

    def test_legendre_removes_orders_of_magnitude(self):
        mono = cn.gram_condition(11, basis="monomial")
        leg = cn.gram_condition(11, basis="legendre")
        assert mono / leg > 1e12, (mono, leg)
        assert leg < 1e3

    def test_the_monomial_conditioning_degrades_with_degree_and_legendre_does_not(self):
        mono = [cn.gram_condition(d) for d in (3, 7, 11)]
        leg = [cn.gram_condition(d, basis="legendre") for d in (3, 7, 11)]
        assert mono[-1] / mono[0] > 1e8
        assert leg[-1] / leg[0] < 10

    def test_it_rejects_a_design_it_cannot_build(self):
        with pytest.raises(ValueError):
            cn.design_matrix(-1)
        with pytest.raises(ValueError):
            cn.design_matrix(10, n_points=5)
        with pytest.raises(ValueError):
            cn.design_matrix(3, basis="chebyshev")


class TestEquilibrate:
    def test_it_gives_every_column_unit_norm(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((50, 4)) * np.array([1e-6, 1.0, 1e6, 1e3])
        B, scales = cn.equilibrate(A)
        assert np.allclose(np.linalg.norm(B, axis=0), 1.0)
        assert scales.shape == (4,)

    def test_it_repairs_a_condition_number_that_is_only_about_units(self):
        rng = np.random.default_rng(1)
        base = rng.standard_normal((80, 4))
        scaled = base * np.array([1e-6, 1.0, 1e6, 1e3])
        assert cn.condition_number(scaled) > 1e8
        assert cn.condition_number(cn.equilibrate(scaled)[0]) < 1e2

    def test_a_zero_column_does_not_divide_by_zero(self):
        A = np.array([[1.0, 0.0], [2.0, 0.0]])
        B, scales = cn.equilibrate(A)
        assert np.all(np.isfinite(B))
        assert scales[1] == 0.0
