"""What the step size does, on a quadratic where the answer is exact and on a
network where it is not.

The claim this file is built to gate is that `2/lam_max` is a threshold rather
than a guideline, and separately that on a network the threshold is real but
sits about twice as high as the number you would compute at initialisation --
because `lam_max` is a function of the step size you chose.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.numerics import steps as st


@pytest.fixture(scope="module")
def H():
    return st.quadratic_design()


@pytest.fixture(scope="module")
def x0(H):
    return np.random.default_rng(0).standard_normal(H.shape[0])


def _converges(H, x0, lr, *, beta=0.0, steps=3000):
    hist = st.gd_quadratic(H, x0, lr, steps, beta=beta)
    return bool(np.isfinite(hist[-1]) and hist[-1] < hist[0])


class TestTheThresholdIsExact:
    def test_bisection_recovers_two_over_lambda_max(self, H, x0):
        lo, hi = st.divergence_threshold(
            lambda lr: _converges(H, x0, lr), 0.01, 4.5, iters=40)
        assert lo == pytest.approx(st.stability_limit(H), rel=1e-3)
        assert hi - lo < 1e-6

    def test_one_percent_over_the_limit_costs_orders_of_magnitude(self, H, x0):
        """Not a gentle degradation. The point of the episode."""
        s = st.lr_sweep(H, (0.99, 1.01), steps=400, seed=0)
        assert s.at(0.99) < 1e-2
        assert s.at(1.01) > 1e3
        assert s.at(1.01) / s.at(0.99) > 1e5

    def test_at_exactly_the_critical_step_the_top_direction_is_preserved(self, H, x0):
        """`|1 - lr*lam_max| = |1 - 2| = 1` exactly, so that component neither
        grows nor decays -- and 400 steps later it is the same number to every
        digit. Nothing is diverging and nothing is converging."""
        crit = st.stability_limit(H)
        final = st.gd_quadratic(H, x0, crit, 400)[-1]
        _, q = np.linalg.eigh(H)
        component = abs(float((q.T @ (H @ x0))[-1]))
        assert final == pytest.approx(component, rel=1e-9)

    def test_exactly_one_direction_grows_just_past_the_limit(self, H):
        crit = st.stability_limit(H)
        assert (st.amplification(H, 0.99 * crit) > 1).sum() == 0
        assert (st.amplification(H, 1.01 * crit) > 1).sum() == 1

    def test_the_amplifications_are_the_multipliers_the_run_applies(self, H, x0):
        """`amplification` is not an analogy for what happens; a single step
        multiplies each eigencomponent by exactly that."""
        lr = 0.5 * st.stability_limit(H)
        e, q = np.linalg.eigh(H)
        before = q.T @ x0
        after = q.T @ (x0 - lr * (H @ x0))
        assert np.allclose(np.abs(after / before), st.amplification(H, lr))


class TestTheFlowItDiscretises:
    def test_the_exact_flow_converges_at_every_step_size(self, H, x0):
        """Forward Euler blows up at `1.10x` the critical step; the continuous
        solution it is approximating has reached `2.2e-05` by the same
        integration time. Whatever diverged, the discretisation did it."""
        crit = st.stability_limit(H)
        for m, bound in ((0.5, 3e-3), (0.99, 1e-4), (1.01, 1e-4),
                         (1.10, 1e-4), (5.0, 1e-15)):
            assert st.gradient_flow(H, x0, 400 * m * crit) < bound, m
        assert st.gd_quadratic(H, x0, 1.10 * crit, 400)[-1] > 1e30

    def test_euler_tracks_the_flow_when_the_step_is_small(self, H, x0):
        """So the two are the same object, not two different claims."""
        lr = 0.02 * st.stability_limit(H)
        euler = st.gd_quadratic(H, x0, lr, 50)[-1]
        exact = st.gradient_flow(H, x0, 50 * lr)
        assert euler == pytest.approx(exact, rel=0.05)


class TestTheOptimumSitsOnTheCliffEdge:
    @pytest.mark.parametrize("condition", [3.0, 10.0, 100.0, 1000.0])
    def test_the_optimum_is_kappa_over_kappa_plus_one_of_the_limit(self, condition):
        """Exact, and it rises towards 1 as the problem gets harder -- which is
        why "raise it until it breaks, then back off" is nearly right, and why
        it leaves you within a percent of a cliff."""
        h = st.quadratic_design(20, condition=condition)
        got = st.optimal_lr(h) / st.stability_limit(h)
        assert got == pytest.approx(condition / (condition + 1.0), rel=1e-9)
        assert got == pytest.approx(st.optimal_lr_fraction(h), rel=1e-12)

    def test_the_optimal_rate_is_the_worst_multiplier_at_that_step(self, H):
        worst = st.amplification(H, st.optimal_lr(H)).max()
        assert worst == pytest.approx(st.optimal_rate(H), rel=1e-9)

    def test_no_step_size_beats_it(self, H, x0):
        """Searched, not asserted."""
        best = st.optimal_rate(H)
        for f in np.linspace(0.05, 0.999, 120):
            assert st.amplification(H, f * st.stability_limit(H)).max() >= best - 1e-12

    def test_and_what_it_buys_is_still_slow(self, H):
        """115 steps to gain one digit, at `kappa = 100`, optimally tuned.
        Ill-conditioning is not a step-size problem."""
        assert st.steps_per_decade(st.optimal_rate(H)) == pytest.approx(115.1, rel=1e-3)

    def test_steps_per_decade_refuses_a_non_contraction(self):
        assert st.steps_per_decade(1.0) == float("inf")
        assert st.steps_per_decade(1.5) == float("inf")


class TestMomentum:
    @pytest.mark.parametrize("beta", [0.0, 0.5, 0.9, 0.99])
    def test_the_limit_widens_by_exactly_one_plus_beta(self, H, x0, beta):
        lo, _ = st.divergence_threshold(
            lambda lr: _converges(H, x0, lr, beta=beta), 0.01, 4.5, iters=40)
        assert lo == pytest.approx(st.momentum_limit(H, beta), rel=2e-3)

    def test_a_step_that_diverges_plain_trains_with_momentum(self, H, x0):
        """Most of the practical content of the previous test."""
        lr = 1.5 * st.stability_limit(H)
        assert not _converges(H, x0, lr, beta=0.0)
        assert _converges(H, x0, lr, beta=0.9)

    def test_the_tuned_pair_is_ten_times_faster_per_decade(self, H, x0):
        """`sqrt(kappa)` in place of `kappa`. The rate is asymptotic, so a
        400-step run measures a little worse than the formula."""
        m = st.momentum_optimal(H)
        hist = st.gd_quadratic(H, x0, m["lr"], 400, beta=m["beta"])
        measured = (hist[-1] / hist[0]) ** (1 / 400)
        assert measured == pytest.approx(m["rate"], rel=0.02)
        speedup = (st.steps_per_decade(st.optimal_rate(H))
                   / st.steps_per_decade(measured))
        assert 8.0 < speedup < 12.0

    def test_beta_must_be_a_contraction(self, H):
        for bad in (-0.1, 1.0, 1.5):
            with pytest.raises(ValueError, match="beta"):
                st.momentum_limit(H, bad)


class TestTheCheapDiagnostic:
    def test_power_iteration_finds_the_top_eigenvalue(self, H):
        got = st.power_iteration(lambda v: H @ v, H.shape[0])
        assert got["converged"]
        assert got["lam_max"] == pytest.approx(st.spectrum(H)[-1], rel=1e-8)

    def test_it_costs_a_few_dozen_products_not_a_decomposition(self, H):
        """Which is the whole reason `2/lr` can be an instrument during a run."""
        got = st.power_iteration(lambda v: H @ v, H.shape[0])
        assert got["iterations"] < 80

    def test_it_reports_failure_rather_than_a_wrong_number(self, H):
        got = st.power_iteration(lambda v: H @ v, H.shape[0], iters=3)
        assert not got["converged"]

    def test_a_zero_operator_is_not_a_crash(self):
        got = st.power_iteration(lambda v: np.zeros_like(v), 5)
        assert got["lam_max"] == 0.0 and got["converged"]

    def test_the_ratio_is_one_at_the_boundary(self, H):
        lam = float(st.spectrum(H)[-1])
        assert st.sharpness_ratio(lam, st.stability_limit(H)) == pytest.approx(1.0)


class TestTheDesignAndTheGuards:
    def test_the_spectrum_has_the_condition_number_it_was_asked_for(self):
        for c in (5.0, 250.0):
            e = st.spectrum(st.quadratic_design(12, condition=c))
            assert e[-1] / e[0] == pytest.approx(c, rel=1e-9)

    def test_no_eigendirection_is_an_axis(self):
        """Otherwise the iteration decouples in the coordinates being printed
        and every claim above looks easier than it is."""
        _, q = np.linalg.eigh(st.quadratic_design())
        assert np.abs(q).max() < 0.9

    def test_one_dimension_has_no_condition_number(self):
        with pytest.raises(ValueError, match="at least two"):
            st.quadratic_design(1)

    def test_a_singular_hessian_has_no_stability_limit(self):
        with pytest.raises(ValueError, match="positive largest"):
            st.stability_limit(np.zeros((3, 3)))

    def test_the_bracket_must_actually_bracket(self, H, x0):
        with pytest.raises(ValueError, match="lower end"):
            st.divergence_threshold(lambda lr: _converges(H, x0, lr), 5.0, 9.0)
        with pytest.raises(ValueError, match="upper end"):
            st.divergence_threshold(lambda lr: _converges(H, x0, lr), 0.01, 0.05)

    def test_a_diverging_run_is_readable_rather_than_a_warning(self, H, x0):
        hist = st.gd_quadratic(H, x0, 3.0 * st.stability_limit(H), 2000)
        assert np.isinf(hist[-1])
        assert np.isfinite(hist[0])


class TestTheEdgeOfStability:
    """The part where the quadratic story stops being the whole story.

    Full-batch on purpose: this is a property of the deterministic iteration,
    and minibatch noise both blurs the boundary and invites the reader to
    attribute the effect to the noise.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def runs(cls):
        pytest.importorskip("torch")
        return {lr: st.edge_of_stability(lr) for lr in (0.05, 0.2, 0.5, 0.8)}

    def test_the_naive_threshold_is_computed_at_initialisation(self):
        pytest.importorskip("torch")
        lam0 = st.initial_sharpness()
        assert lam0 == pytest.approx(7.3556, rel=1e-3)
        assert 2.0 / lam0 == pytest.approx(0.2719, rel=1e-3)

    def test_a_step_twice_that_threshold_trains_fine(self, runs):
        """`lr = 0.5` is 1.84x the initialisation threshold and converges."""
        assert not runs[0.5]["diverged"]
        assert runs[0.5]["loss"] < 1e-3

    def test_sharpness_lands_on_two_over_lr_across_a_five_fold_range(self, runs):
        for lr in (0.2, 0.5):
            assert runs[lr]["ratio"] == pytest.approx(1.0, abs=0.01), lr

    def test_and_is_pushed_down_to_it_from_above(self, runs):
        """The two-sided version. `lam_max` starts at 7.36 and the `lr = 0.5`
        run ends at 4.0, which is `2/lr`. It is an attractor, not a ceiling
        that sharpness climbs to -- the summary this replaced said otherwise."""
        assert st.initial_sharpness() > 2.0 / 0.5
        assert runs[0.5]["lam_max"] == pytest.approx(4.0, rel=0.02)

    def test_a_small_step_plateaus_below_the_boundary_instead(self, runs):
        """So the edge is a regime, not a law. At `lr = 0.05` the sharpness
        rises during training and stops at 74% of the boundary."""
        got = runs[0.05]
        assert 0.6 < got["ratio"] < 0.85
        assert got["lam_max"] > 3 * st.initial_sharpness()   # it did rise

    def test_the_loss_is_not_monotone_at_the_edge_and_is_below_it(self, runs):
        """A training loss that ticks up at a large step is what convergence
        looks like there."""
        assert runs[0.05]["rose_fraction"] == 0.0
        assert runs[0.2]["rose_fraction"] > 0.2
        assert runs[0.5]["rose_fraction"] > 0.4

    def test_but_it_still_falls_by_orders_of_magnitude(self, runs):
        assert runs[0.2]["tail_drop"] > 50
        assert runs[0.2]["max_rise"] < 0.05

    def test_past_the_limit_it_diverges_in_a_handful_of_steps(self, runs):
        """Nothing degrades gracefully above the threshold."""
        assert runs[0.8]["diverged"]
        assert runs[0.8]["diverged_at"] < 30

    def test_the_true_limit_is_about_twice_the_naive_one(self):
        """Bisected, not asserted. Fewer steps than the headline run, because
        divergence shows up in tens of steps and this only needs the bracket."""
        pytest.importorskip("torch")
        ok = lambda lr: not st.edge_of_stability(  # noqa: E731
            lr, steps=400, probes=2)["diverged"]
        lo, hi = st.divergence_threshold(ok, 0.5, 0.8, iters=8)
        naive = 2.0 / st.initial_sharpness()
        assert 1.9 < lo / naive < 2.2
        assert (hi - lo) / lo < 0.01
