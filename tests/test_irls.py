"""What IRLS does, and what it does when the answer it is looking for does not
exist. Every assertion here is a measurement that was taken before it was
written down, and several of them contradict the first version of the claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.linalg import irls as ir


class TestTheNumerics:
    def test_the_sigmoid_does_not_overflow_where_separation_puts_it(self):
        """Under separation eta reaches -1000 within a dozen iterations, and the
        naive 1/(1+exp(-eta)) warns there. This form must not."""
        eta = np.array([-1e4, -800.0, -50.0, 0.0, 50.0, 800.0, 1e4])
        # Underflow is not raised on: exp(-1e4) flushing to zero *is* the right
        # answer here. Overflow, invalid and divide are the failures that matter.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            p = ir.sigmoid(eta)
        assert np.all(np.isfinite(p))
        assert p[0] == 0.0 and p[-1] == 1.0
        assert p[3] == pytest.approx(0.5)
        assert np.all(np.diff(p) >= 0)

    def test_the_sigmoid_matches_the_naive_form_where_that_form_works(self):
        eta = np.linspace(-30, 30, 61)
        assert ir.sigmoid(eta) == pytest.approx(1.0 / (1.0 + np.exp(-eta)))

    def test_the_log_likelihood_stays_finite_where_the_naive_form_does_not(self):
        """The reason this matters: the naive form reports -inf, which reads as
        'stopped improving', while the fit is still improving."""
        X = np.array([[1.0, 1.0], [1.0, -1.0]])
        y = np.array([1.0, 0.0])
        beta = np.array([0.0, 800.0])            # p rounds to exactly 1 and 0
        p = ir.sigmoid(X @ beta)
        assert p[0] == 1.0 and p[1] == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            naive = np.sum(y * np.log(p) + (1 - y) * np.log1p(-p))
        assert not np.isfinite(naive) or naive == 0.0
        assert np.isfinite(ir.log_likelihood(X, y, beta))

    def test_the_log_likelihood_is_negative_and_increases_towards_zero(self):
        X = np.array([[1.0, 1.0], [1.0, -1.0]])
        y = np.array([1.0, 0.0])
        lls = [ir.log_likelihood(X, y, [0.0, b]) for b in (0.0, 1.0, 5.0, 50.0)]
        assert all(v < 0 for v in lls[:-1])
        assert lls == sorted(lls)


class TestTheOrdinaryCase:
    def _fit(self):
        rng = np.random.default_rng(7)
        X, y = ir.well_posed_design(4000, rng=rng, beta=(-0.5, 1.0, -0.8))
        return X, y, ir.irls(X, y)

    def test_it_converges_and_recovers_the_coefficients(self):
        X, y, fit = self._fit()
        assert fit.converged and fit.iterations < 12
        assert fit.beta == pytest.approx([-0.5, 1.0, -0.8], abs=0.12)

    def test_the_weighted_design_is_well_conditioned_throughout(self):
        """So the pathologies below are not a property of the method."""
        _, _, fit = self._fit()
        assert fit.weighted_condition.max() < 50.0
        assert fit.saturated == 0
        assert fit.min_weight > 1e-4

    def test_standard_errors_are_finite_and_small(self):
        _, _, fit = self._fit()
        assert np.all(np.isfinite(fit.standard_errors))
        assert fit.standard_errors.max() < 0.2

    def test_it_rejects_a_response_that_is_not_zero_one(self):
        X = np.ones((5, 2))
        with pytest.raises(ValueError, match="0/1"):
            ir.irls(X, np.array([0.0, 1.0, 2.0, 1.0, 0.0]))
        with pytest.raises(ValueError, match="shape"):
            ir.irls(X, np.zeros(4))


class TestItIsNewtonsMethod:
    def test_the_error_squares_at_each_step(self):
        """Quadratic convergence, which is the claim 'IRLS is Newton's method'
        made checkable."""
        rng = np.random.default_rng(5)
        X, y = ir.well_posed_design(600, rng=rng)
        errs = ir.newton_vs_gradient(X, y, steps=12)["newton"]
        useful = errs[(errs > 1e-12) & (errs < 0.5)]
        assert len(useful) >= 3
        # e_{k+1} <= C e_k^2 with a modest constant, checked pairwise.
        for a, b in zip(useful[:-1], useful[1:]):
            assert b <= 5.0 * a ** 2, (a, b)

    def test_a_fixed_step_gradient_ascent_is_nowhere_near(self):
        """A weak opponent on purpose: this shows what the Hessian buys, not
        that first-order methods cannot solve logistic regression."""
        rng = np.random.default_rng(5)
        X, y = ir.well_posed_design(600, rng=rng)
        both = ir.newton_vs_gradient(X, y, steps=12, lr=0.1)
        assert both["newton"][-1] < 1e-8
        assert both["gradient"][-1] > 1.0


class TestCompleteSeparation:
    def _design(self):
        return ir.separable_design(200, rng=np.random.default_rng(3))

    def test_the_linear_program_finds_the_separating_direction(self):
        X, y = self._design()
        got = ir.separation_lp(X, y)
        assert got["separated"] and got["margin"] > 0
        signs = np.where(y > 0.5, 1.0, -1.0)
        assert np.all(signs * (X @ got["direction"]) > 0)

    def test_it_reports_no_separation_on_an_ordinary_design(self):
        rng = np.random.default_rng(7)
        X, y = ir.well_posed_design(800, rng=rng)
        assert not ir.separation_lp(X, y)["separated"]

    def test_the_fit_does_not_converge_and_says_so(self):
        X, y = self._design()
        fit = ir.irls(X, y, max_iter=60)
        assert not fit.converged
        assert any("did not converge" in n for n in fit.notes)
        assert any("at 0 or 1" in n for n in fit.notes)

    def test_the_coefficient_is_whatever_the_iteration_limit_was(self):
        """The thesis of the episode: there is no coefficient, only a stopping
        rule."""
        X, y = self._design()
        rows = ir.iteration_sweep(X, y, (5, 10, 20, 40))
        sizes = [r["largest_coefficient"] for r in rows]
        assert sizes == sorted(sizes)
        assert sizes[-1] > 10 * sizes[0]
        assert not any(r["converged"] for r in rows)

    def test_the_z_statistic_also_depends_on_where_you_stopped(self):
        """And it moves the *other* way, which is why this is hard to catch: a
        fit stopped early looks significant, and one allowed to run does not."""
        X, y = self._design()
        rows = ir.iteration_sweep(X, y, (5, 20))
        assert rows[0]["largest_z"] > 2.0
        assert rows[1]["largest_z"] < 0.5


class TestPartialSeparation:
    """The case that reaches production, and the one the textbook test misses."""

    def _design(self):
        return ir.quiet_separation_design(1000, rng=np.random.default_rng(11))

    def test_the_complete_separation_test_says_no(self):
        """Correctly. There is no separating hyperplane, because the rows where
        the dummy is false contain both classes. The likelihood is unbounded
        anyway, in that one coefficient."""
        X, y, _ = self._design()
        assert not ir.separation_lp(X, y)["separated"]

    def test_the_empty_cell_check_finds_it(self):
        X, y, k = self._design()
        found = ir.empty_cell_check(X, y, names=["intercept", "x1", "x2", "rare"])
        assert len(found) == 1
        assert found[0]["name"] == "rare"
        assert found[0]["rows"] == k
        assert found[0]["outcome"] == 1.0

    def test_the_empty_cell_check_is_quiet_on_an_ordinary_design(self):
        rng = np.random.default_rng(7)
        X, y = ir.well_posed_design(800, rng=rng)
        d = np.zeros(800)
        d[rng.choice(800, 80, replace=False)] = 1.0    # a dummy with both outcomes
        assert ir.empty_cell_check(np.column_stack([X, d]), y) == []

    def test_the_other_coefficients_are_completely_unaffected(self):
        """Which is what makes it dangerous: the table looks healthy."""
        X, y, _ = self._design()
        small = ir.irls(X, y, max_iter=5).beta[:3]
        large = ir.irls(X, y, max_iter=100).beta[:3]
        assert small == pytest.approx(large, abs=1e-3)

    def test_only_the_separated_coefficient_diverges(self):
        X, y, _ = self._design()
        rows = ir.iteration_sweep(X, y, (5, 25, 100))
        got = [r["largest_coefficient"] for r in rows]
        assert got[0] > 5.0 and got[-1] > 25.0
        assert got == sorted(got)

    def test_it_grows_by_exactly_one_per_iteration_while_the_weights_are_real(self):
        """Measured, then explained: Newton's step on an unbounded logistic
        likelihood settles at a constant, and the constant is 1."""
        X, y, _ = self._design()
        fit = ir.irls(X, y, max_iter=200)
        steady = np.diff(fit.path)[8:20]
        assert steady == pytest.approx(1.0, abs=2e-3), steady

    def test_the_condition_number_grows_by_a_factor_of_e_per_iteration(self):
        """Which follows from the line above: the weight on a saturated row goes
        like exp(-|eta|), and |eta| is rising by 1 each pass."""
        X, y, _ = self._design()
        c = ir.irls(X, y, max_iter=200).weighted_condition
        ratios = c[6:15] / c[5:14]
        assert ratios == pytest.approx(np.e, rel=2e-3), ratios

    def test_the_growth_slows_but_never_stops(self):
        """Once the weight floor pins the weights. Anyone who reads a flattening
        coefficient path as convergence is reading the floor."""
        X, y, _ = self._design()
        fit = ir.irls(X, y, max_iter=300)
        d = np.diff(fit.path)
        assert d[-1] < 0.02
        assert d[-1] > 0.0
        assert not fit.converged


class TestTheStandardErrorIsNotAboutTheData:
    def test_it_is_one_over_the_root_of_the_floor_times_the_category_size(self):
        """The measurement that decided this test: under saturation every weight
        in the category is pinned at the library's floor, so the reported
        standard error is a function of a constant somebody else chose."""
        X, y, k = ir.quiet_separation_design(1000, rng=np.random.default_rng(11))
        rows = ir.floor_determined_error(X, y, index=3)
        for r in rows:
            assert r["rows_in_category"] == k
            assert r["standard_error"] == pytest.approx(r["closed_form"], rel=1e-6)

    def test_changing_only_the_floor_changes_the_reported_error_tenfold(self):
        X, y, _ = ir.quiet_separation_design(1000, rng=np.random.default_rng(11))
        rows = ir.floor_determined_error(X, y, index=3,
                                        floors=(1e-8, 1e-10, 1e-12))
        ses = [r["standard_error"] for r in rows]
        assert ses[1] / ses[0] == pytest.approx(10.0, rel=1e-3)
        assert ses[2] / ses[1] == pytest.approx(10.0, rel=1e-3)


class TestThePenaltyRestoresExistence:
    def test_a_ridge_penalty_makes_the_fit_converge(self):
        """Episode five's penalty, on ill-conditioning the fit created rather
        than ill-conditioning the design brought."""
        X, y, _ = ir.quiet_separation_design(1000, rng=np.random.default_rng(11))
        assert not ir.irls(X, y, max_iter=100).converged
        assert ir.irls(X, y, max_iter=100, ridge=1.0).converged

    def test_it_trades_the_coefficient_for_a_usable_standard_error(self):
        X, y, _ = ir.quiet_separation_design(1000, rng=np.random.default_rng(11))
        rows = ir.ridge_sweep(X, y, (0.0, 0.1, 1.0, 10.0), index=3, max_iter=100)
        coefs = [abs(r["coefficient"]) for r in rows]
        ses = [r["standard_error"] for r in rows]
        assert coefs == sorted(coefs, reverse=True)
        assert ses == sorted(ses, reverse=True)
        assert rows[0]["standard_error"] > 100 * rows[-1]["standard_error"]

    def test_the_penalised_fit_has_a_lower_likelihood(self):
        """It has to: the unpenalised path is climbing towards the supremum. The
        penalty buys existence and pays in fit."""
        X, y, _ = ir.quiet_separation_design(1000, rng=np.random.default_rng(11))
        rows = ir.ridge_sweep(X, y, (0.0, 1.0, 10.0), index=3, max_iter=100)
        lls = [r["log_likelihood"] for r in rows]
        assert lls == sorted(lls, reverse=True)
