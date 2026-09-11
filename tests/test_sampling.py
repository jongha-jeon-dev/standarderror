"""Estimators for a gradient that does not exist, against one that does.

The exact gradient is enumerable here, so unlike the rest of this series the
model measurements are not merely descriptive -- there is a right answer and
each estimator either matches it or does not. The identities are pinned
exactly; the findings about this model are pinned as inequalities.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from standarderror.calculus import sampling as sm  # noqa: E402
from standarderror.llm import tiny  # noqa: E402


@pytest.fixture(scope="module")
def dec():
    x, y = next(iter(tiny.batches(count=1, size=1, seed=0)))
    return sm.decision(x[0, :tiny.BLOCK - 1], int(y[0, tiny.BLOCK - 1]))


class TestTheExactGradient:

    def test_it_is_the_softmax_jacobian_applied_to_the_losses(self, dec):
        from standarderror.calculus import saturation as sat
        j = sat.jacobian(dec["p"])
        assert np.allclose(dec["exact"], j @ dec["loss"], atol=1e-9)

    def test_it_sums_to_zero(self, dec):
        """Episode 3's theorem again: adding a constant to every logit does
        not change the distribution, so the gradient has no mean."""
        assert abs(dec["exact"].sum()) < 1e-6

    def test_it_matches_autodiff_through_the_enumerated_expectation(self, dec):
        """The whole episode rests on this vector being right, so it is
        checked against torch differentiating the explicit sum."""
        z = torch.tensor(dec["logits"], requires_grad=True)
        loss = torch.tensor(dec["loss"], dtype=torch.float32)
        (torch.softmax(z, -1) @ loss).backward()
        assert np.allclose(z.grad.numpy(), dec["exact"], atol=2e-6)

    def test_the_soft_forward_agrees_with_the_hard_one(self, dec):
        """A one-hot through `w @ E` must give the tabulated loss, or every
        pathwise estimator is measuring a different problem."""
        model = tiny.load()["model"]
        f = sm._soft_forward(model, dec["context"], dec["target"])
        for t in (0, dec["vocab"] // 2, dec["vocab"] - 3):
            w = torch.nn.functional.one_hot(torch.tensor(t),
                                            dec["vocab"]).float()
            assert float(f(w)) == pytest.approx(dec["loss"][t], abs=1e-5)


class TestReinforce:

    def test_it_is_unbiased_within_its_own_monte_carlo_error(self, dec):
        s = sm.reinforce(dec, draws=6000, baseline=dec["expected"])
        assert s["relative_bias"] < 3 * s["mc_error"]

    def test_a_baseline_cuts_the_variance_and_not_the_expectation(self, dec):
        bare = sm.reinforce(dec, draws=6000)
        based = sm.reinforce(dec, draws=6000, baseline=dec["expected"])
        assert based["sd"] < 0.75 * bare["sd"]
        assert based["relative_bias"] < 3 * based["mc_error"]
        assert bare["relative_bias"] < 3 * bare["mc_error"]


class TestStraightThrough:

    @pytest.fixture(scope="class")
    def st(self, dec):
        return sm.pathwise(dec, sm.straight_through("softmax"), draws=300)

    def test_it_is_biased_by_much_more_than_its_own_noise(self, dec, st):
        """The distinction that matters: this bias does not average away."""
        assert st["relative_bias"] > 0.3
        assert st["relative_bias"] > 10 * st["mc_error"]

    def test_it_understates_the_size(self, dec, st):
        assert st["scale"] < 0.6

    def test_the_two_ways_of_writing_it_disagree(self, dec):
        """Same forward pass, different derivative -- episode 1, in the wild."""
        soft = sm.pathwise(dec, sm.straight_through("softmax"), draws=200)
        hard = sm.pathwise(dec, sm.straight_through("logits"), draws=200)
        assert hard["scale"] > 2.0 > soft["scale"]
        assert hard["cosine"] < soft["cosine"] - 0.2

    def test_its_cosine_is_not_evidence(self, dec, st):
        """Noise through the same softmax Jacobian scores comparably, so a
        cosine near one says nothing about the estimator's own content."""
        ctrl = sm.jacobian_control(dec, draws=1000)
        assert ctrl["p90"] > 0.7
        assert st["cosine"] < ctrl["max"]


class TestWhatItAssumes:

    def test_the_linear_extrapolation_is_weak(self, dec):
        """Positive but small. The episode's claim is that most of what
        straight-through says about the alternatives is noise, not that it is
        backwards -- an earlier draft said backwards, on a pooled correlation
        that turned out to be a Simpson's paradox."""
        lin = sm.linearisation(dec)
        assert -0.2 < lin["correlation"] < 0.6

    def test_pooling_contexts_would_have_said_otherwise(self):
        """Kept as a test because the trap is easy to fall into twice: the
        pooled correlation disagrees in sign with every context's own."""
        s = sm.survey(contexts=6, st_draws=40, rf_draws=400)
        assert s["pooled_correlation"] < s["correlation_min"]
        assert s["correlation_median"] > 0.0

    def test_predicted_and_true_are_the_same_length(self, dec):
        lin = sm.linearisation(dec)
        assert lin["predicted"].size == lin["true"].size == dec["vocab"] - 1


class TestTheTradeoff:

    def test_a_few_samples_are_enough_to_prefer_the_unbiased_one(self, dec):
        st = sm.pathwise(dec, sm.straight_through("softmax"), draws=300)
        rf = sm.reinforce(dec, draws=6000, baseline=dec["expected"])
        n = sm.crossover(st, rf)
        assert 1.0 < n < 100.0

    def test_the_error_curve_falls_to_the_bias_and_stops(self, dec):
        st = sm.pathwise(dec, sm.straight_through("softmax"), draws=300)
        lo, hi = sm.error_curve(st, [1, 10_000_000])
        assert hi < lo
        assert hi == pytest.approx(st["relative_bias"], rel=0.15)

    def test_lowering_the_temperature_trades_bias_for_variance(self, dec):
        rows = sm.temperature_sweep(dec, taus=(2.0, 0.5), draws=200)
        warm, cool = rows
        assert cool["relative_bias"] < warm["relative_bias"]
        assert cool["sd"] > warm["sd"]
