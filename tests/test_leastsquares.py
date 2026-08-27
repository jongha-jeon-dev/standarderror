"""Tests for the least-squares module behind Linear Algebra 2.

The properties pinned are the ones the episode's argument depends on, and one of
them is a negative result that was measured rather than assumed: the
orthogonality of the residual does *not* discriminate between the three methods,
because it is the condition the worst of them solves.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.linalg import conditioning as cn
from standarderror.linalg import leastsquares as ls


def _problem(degree=11, seed=0, basis="monomial"):
    X = cn.design_matrix(degree, basis=basis)
    beta = np.random.default_rng(seed).standard_normal(degree + 1)
    return X, beta


class TestAllThreeAgreeWhenTheyCan:
    def test_they_agree_on_a_well_conditioned_problem(self):
        X, beta = _problem(degree=3, basis="legendre")
        y = X @ beta
        a = ls.solve_normal(X, y)
        b = ls.solve_qr(X, y)
        c = ls.solve_svd(X, y)
        assert np.allclose(a, beta, atol=1e-10)
        assert np.allclose(b, beta, atol=1e-12)
        assert np.allclose(c, beta, atol=1e-12)

    def test_each_solves_an_overdetermined_system_exactly_when_consistent(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((60, 4))
        beta = rng.standard_normal(4)
        y = X @ beta
        for solver in (ls.solve_normal, ls.solve_qr, ls.solve_svd):
            assert np.allclose(solver(X, y), beta, atol=1e-10), solver


class TestFormingTheGramMatrixSquaresTheConditioning:
    def test_the_ratio_is_kappa_itself(self):
        X, _ = _problem(degree=9)
        r = ls.squaring_report(X)
        assert r["ratio"] == pytest.approx(r["kappa_X"], rel=0.05)
        assert r["kappa_gram"] == pytest.approx(r["kappa_X"] ** 2, rel=0.05)

    def test_it_costs_exactly_the_digits_it_should(self):
        X, _ = _problem(degree=9)
        r = ls.squaring_report(X)
        assert r["digits_normal"] < r["digits_qr"]

    @pytest.mark.parametrize("degree", [9, 11, 13])
    def test_normal_equations_are_far_worse_than_the_other_two(self, degree):
        X, beta = _problem(degree=degree)
        by = {m.method: m for m in ls.compare_methods(X, beta)}
        assert by["normal"].error > 1e4 * by["qr"].error
        assert by["normal"].error > 1e4 * by["svd"].error

    def test_qr_and_svd_stay_usable_where_normal_equations_do_not(self):
        X, beta = _problem(degree=11)
        by = {m.method: m for m in ls.compare_methods(X, beta)}
        assert by["normal"].error > 0.01          # worthless
        assert by["qr"].error < 1e-8              # fine
        assert by["svd"].error < 1e-8

    def test_an_orthogonal_basis_removes_the_problem_for_all_three(self):
        X, beta = _problem(degree=11, basis="legendre")
        for m in ls.compare_methods(X, beta):
            assert m.error < 1e-12, m


class TestTheOrthogonalityCheckCannotAuditTheNormalEquations:
    """A negative result, measured. It is the episode's second lesson.

    `X'X beta = X'y` *is* the statement that the residual is orthogonal to the
    columns, so a solver that solves it enforces the very property one would use
    to check it — and does so to machine precision even when its coefficients are
    wrong by 300 percent.
    """

    @pytest.mark.parametrize("degree", [9, 11, 13])
    def test_every_method_looks_equally_orthogonal(self, degree):
        X, beta = _problem(degree=degree)
        orth = [m.orthogonality for m in ls.compare_methods(X, beta)]
        assert max(orth) < 1e-14
        assert max(orth) / min(orth) < 100

    def test_even_when_the_coefficients_are_hundreds_of_percent_wrong(self):
        X, beta = _problem(degree=13)
        by = {m.method: m for m in ls.compare_methods(X, beta)}
        assert by["normal"].error > 1.0
        assert by["normal"].orthogonality < 1e-14

    def test_the_residual_norm_does_degrade_but_only_a_little(self):
        X, beta = _problem(degree=13)
        by = {m.method: m for m in ls.compare_methods(X, beta)}
        ratio = by["normal"].residual / by["qr"].residual
        assert ratio > 1e6                       # visible
        assert by["normal"].residual < 1e-6      # and still reads as fine

    def test_projection_report_needs_no_known_truth(self):
        X, beta = _problem(degree=5)
        y = X @ beta
        r = ls.projection_report(X, y, ls.solve_qr(X, y))
        assert set(r) == {"residual", "orthogonality"}
        assert r["orthogonality"] < 1e-14


class TestSvdIsTheOnlyOneDefinedOnACollinearDesign:
    @staticmethod
    def _collinear():
        rng = np.random.default_rng(5)
        A = rng.standard_normal((80, 3))
        X = np.column_stack([A, A[:, 0]])        # column 3 duplicates column 0
        beta = np.array([1.0, -2.0, 0.5, 0.0])
        return X, beta

    def test_the_design_is_exactly_rank_deficient(self):
        X, _ = self._collinear()
        assert np.linalg.matrix_rank(X) == 3
        assert X.shape[1] == 4

    def test_the_svd_route_returns_the_minimum_norm_solution(self):
        X, beta = self._collinear()
        y = X @ beta
        b = ls.solve_svd(X, y)
        assert np.allclose(X @ b, y, atol=1e-10)          # fits
        assert np.linalg.norm(b) <= np.linalg.norm(beta) + 1e-8   # and is minimal

    def test_it_splits_a_duplicated_column_evenly(self):
        # The minimum-norm solution shares a duplicated column's coefficient
        # between the two copies rather than choosing one arbitrarily.
        X, beta = self._collinear()
        b = ls.solve_svd(X, X @ beta)
        assert b[0] == pytest.approx(b[3], abs=1e-8)

    def test_rcond_controls_the_truncation(self):
        # Needs a design whose singular values actually span a range: on the
        # earlier one the three real directions are all within a factor of two
        # of each other, so no plausible rcond distinguishes them. Scaling the
        # columns by 1, 1e-3 and 1e-6 makes the truncation visible.
        rng = np.random.default_rng(5)
        A = rng.standard_normal((80, 3)) * np.array([1.0, 1e-3, 1e-6])
        X = np.column_stack([A, A[:, 0]])
        y = X @ np.array([1.0, -2.0, 0.5, 0.0])
        tight = np.linalg.norm(X @ ls.solve_svd(X, y) - y)
        mid = np.linalg.norm(X @ ls.solve_svd(X, y, rcond=1e-4) - y)
        loose = np.linalg.norm(X @ ls.solve_svd(X, y, rcond=0.5) - y)
        assert tight < mid < loose


class TestScalingVariants:
    @staticmethod
    def _design(n=500, dummy_rate=0.20, seed=7):
        rng = np.random.default_rng(seed)
        return np.column_stack([
            np.ones(n),
            rng.normal(3600, 600, n),        # a duration in seconds
            rng.normal(0.30, 0.10, n),       # a probability
            rng.normal(5e7, 1e7, n),         # an amount of money
            (rng.random(n) < dummy_rate).astype(float),
        ])

    def test_it_returns_four_designs_of_the_same_shape(self):
        v = ls.scaling_variants(self._design())
        assert set(v) == {"raw", "centred", "scaled", "standardised"}
        for name, d in v.items():
            assert d["design"].shape == (500, 5), name

    def test_scaling_does_most_of_the_work_and_centring_alone_does_little(self):
        v = ls.scaling_variants(self._design())
        assert v["centred"]["kappa"] > 0.1 * v["raw"]["kappa"]     # barely helps
        assert v["scaled"]["kappa"] < 1e-4 * v["raw"]["kappa"]     # transforms it

    def test_but_centring_matters_once_the_columns_are_scaled(self):
        v = ls.scaling_variants(self._design())
        assert v["standardised"]["kappa"] < 0.1 * v["scaled"]["kappa"]
        assert v["standardised"]["kappa"] < 2.0

    def test_the_intercept_column_is_left_alone(self):
        v = ls.scaling_variants(self._design())
        for name, d in v.items():
            assert np.allclose(d["design"][:, 0], 1.0), name

    def test_a_rare_dummy_is_not_a_conditioning_problem(self):
        common = ls.scaling_variants(self._design(dummy_rate=0.20))
        rare = ls.scaling_variants(self._design(dummy_rate=0.01))
        assert rare["standardised"]["kappa"] < 2.0
        assert abs(rare["standardised"]["kappa"]
                   - common["standardised"]["kappa"]) < 1.0

    def test_a_constant_column_is_refused_rather_than_dividing_by_zero(self):
        X = self._design()
        X[:, 2] = 3.0
        with pytest.raises(ValueError):
            ls.scaling_variants(X)

    def test_an_all_intercept_design_is_refused(self):
        with pytest.raises(ValueError):
            ls.scaling_variants(np.ones((10, 1)))


class TestArgumentChecking:
    def test_a_mismatched_truth_is_refused(self):
        X, _ = _problem(degree=5)
        with pytest.raises(ValueError):
            ls.compare_methods(X, np.zeros(3))

    def test_an_unknown_method_is_refused(self):
        X, beta = _problem(degree=3)
        with pytest.raises(ValueError):
            ls.compare_methods(X, beta, methods=("cholesky",))

    def test_the_report_row_is_printable(self):
        X, beta = _problem(degree=5)
        assert len(ls.compare_methods(X, beta)[0].row()) == 5
