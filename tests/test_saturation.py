"""The softmax Jacobian, its bounds, and the head that stopped learning.

Two kinds of test here and they are worth telling apart. The algebra --
`diag(p) - p p^T`, the singularity, the trace identity, the two-sided bound --
is exact and pinned exactly. The model measurements are properties of one
committed checkpoint, so they are pinned as *inequalities* wide enough to
survive a float difference and narrow enough to fail if the finding goes away.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from standarderror.calculus import saturation as sat  # noqa: E402


def softmax(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max())
    return e / e.sum()


class TestTheJacobian:

    def test_matches_autodiff(self):
        """The closed form against torch's own derivative, which is the only
        check that catches a transposed or misordered outer product."""
        z = torch.tensor([0.4, -1.2, 2.0, 0.1], requires_grad=True)
        got = torch.autograd.functional.jacobian(
            lambda t: torch.nn.functional.softmax(t, dim=-1), z)
        p = torch.nn.functional.softmax(z.detach(), dim=-1).numpy()
        assert np.allclose(got.numpy(), sat.jacobian(p), atol=1e-6)

    def test_is_symmetric_and_psd(self):
        j = sat.jacobian(softmax([1.0, -0.5, 0.3, 2.2, -3.0]))
        assert np.allclose(j, j.T)
        assert np.linalg.eigvalsh(j).min() > -1e-12

    def test_the_all_ones_vector_is_in_the_null_space(self):
        """Adding a constant to every logit does not move the softmax, so the
        Jacobian annihilates the constant direction -- which is why a row's
        logit gradient sums to zero."""
        for z in ([0.0, 0.0], [1.0, -2.0, 0.5], list(range(8))):
            p = softmax(z)
            assert np.abs(sat.jacobian(p) @ np.ones(len(p))).max() < 1e-15

    def test_rank_is_one_less_than_the_support(self):
        p = np.array([0.5, 0.3, 0.2])
        ev = np.linalg.eigvalsh(sat.jacobian(p))
        assert int((ev > 1e-12).sum()) == len(p) - 1

    def test_quadratic_form_is_a_variance(self):
        """`u^T J u = Var_p(u)`, which is the reason confidence and gradient
        are the same question."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            p = softmax(rng.normal(0, 2, 6))
            u = rng.normal(0, 1, 6)
            var = float((p * u * u).sum() - (p * u).sum() ** 2)
            assert math.isclose(float(u @ sat.jacobian(p) @ u), var,
                                rel_tol=1e-9, abs_tol=1e-12)

    def test_fast_path_equals_the_matrix_product(self):
        """`_logit_gradient` avoids materialising J. It had better agree."""
        rng = np.random.default_rng(1)
        p = torch.tensor(softmax(rng.normal(0, 2, 7)))
        g = torch.tensor(rng.normal(0, 1, 7))
        slow = torch.tensor(sat.jacobian(p.numpy())) @ g
        assert torch.allclose(sat._logit_gradient(p, g), slow, atol=1e-12)


class TestTheTraceIsACollisionProbability:

    def test_trace_equals_one_minus_sum_of_squares(self):
        rng = np.random.default_rng(2)
        for _ in range(500):
            p = softmax(rng.normal(0, rng.uniform(0.1, 6.0), 10))
            s = sat.spectrum(p)
            assert math.isclose(s["trace"], s["collision"],
                                rel_tol=1e-9, abs_tol=1e-14)

    def test_it_is_not_the_entropy(self):
        """Stated because the entropy is the quantity usually reached for, and
        the two are not equal even up to a constant."""
        a, b = softmax([3.0, 0.0, 0.0]), softmax([1.4, 1.0, 0.0])
        ha = -(a * np.log(a)).sum()
        hb = -(b * np.log(b)).sum()
        ta, tb = sat.spectrum(a)["trace"], sat.spectrum(b)["trace"]
        assert not math.isclose(ha / hb, ta / tb, rel_tol=0.05)


class TestTheBound:

    def test_lower_bound_is_a_diagonal_entry(self):
        p = softmax([2.5, 0.1, -1.0, 0.4])
        m = p.max()
        assert sat.spectrum(p)["spectral_norm"] >= m * (1 - m) - 1e-12

    def test_two_sided_bound_holds_and_the_top_is_attained(self):
        b = sat.bound_sweep(draws=3000, seed=3)
        assert b["rows"] > 5_000
        assert b["violations_low"] == 0
        assert b["violations_high"] == 0
        assert b["max_ratio_to_upper"] == pytest.approx(1.0, abs=1e-6)

    def test_a_two_way_contest_is_the_worst_case(self):
        """Fixing `m`, the norm is largest when the losing mass sits on one
        runner-up and smallest when it spreads -- so neither bound is slack."""
        rows = sat.concentration(m_values=(0.99,), sizes=(2, 64))
        by = {(r["n"], r["shape"]): r for r in rows}
        assert by[(64, "runner-up")]["over_upper"] == pytest.approx(1.0,
                                                                    abs=1e-9)
        assert by[(64, "spread")]["over_lower"] == pytest.approx(1.0, abs=0.02)
        assert (by[(2, "runner-up")]["spectral_norm"]
                == pytest.approx(by[(64, "runner-up")]["spectral_norm"]))

    def test_the_norm_does_not_depend_on_how_many_positions_there_are(self):
        """A 64-way softmax at m = 0.99 has the gradient capacity of a two-way
        one. This is the claim the episode is built on."""
        wide = sat.spectrum(np.r_[0.99, np.full(63, 0.01 / 63)])
        narrow = sat.spectrum([0.99, 0.01])
        assert wide["upper_bound"] == pytest.approx(narrow["upper_bound"])
        assert wide["spectral_norm"] < narrow["spectral_norm"]

    def test_slowdown_is_the_reciprocal_capacity(self):
        assert sat.slowdown(0.5) == pytest.approx(1.0)
        assert sat.slowdown(0.98) == pytest.approx(12.755, rel=1e-3)
        assert sat.slowdown(0.9999) == pytest.approx(2500.25, rel=1e-3)


class TestOnTheModel:

    @pytest.fixture(scope="class")
    def gain(self):
        return sat.attention_gain(count=2, size=4, seed=0)

    def test_no_row_exceeds_the_bound(self, gain):
        assert gain["violations_of_bound"] == 0

    def test_logit_gradients_sum_to_zero_within_a_row(self, gain):
        """`J 1 = 0` in float32, on a real backward pass. The median is at
        machine precision; the maximum is a cancellation, not a violation."""
        assert gain["row_sum_median"] < 1e-6
        assert gain["row_sum_max"] < 1e-2

    def test_the_realised_gain_scales_like_the_bound(self, gain):
        """Slope one in log-log: the bound predicts the gradient up to a
        constant, rather than merely capping it."""
        assert gain["log_slope"] == pytest.approx(1.0, abs=0.1)
        assert 0.05 < gain["log_constant"] < 0.3

    def test_layer_0_head_1_is_the_saturated_one(self, gain):
        by = {(h["layer"], h["head"]): h for h in gain["heads"]}
        target = by[(0, 1)]
        assert target["median_max_p"] > 0.9
        assert target["previous_token_share"] > 0.95
        others = [h["median_gain"] for k, h in by.items() if k != (0, 1)]
        assert target["median_gain"] < min(others) / 3

    def test_the_patched_forward_reproduces_the_model(self):
        """The ablations rebuild the forward pass to reach inside attention.
        If it drifts from the real one, every ablation number is wrong."""
        from standarderror.llm import tiny
        model = tiny.load()["model"]
        x, y = next(iter(tiny.batches(count=1, size=4, seed=0)))
        with torch.no_grad():
            a, la, _, _ = model(x, y)
            b, lb = sat._patched_forward(model, x, y, None)
        assert torch.equal(a, b)
        assert float(la) == float(lb)

    def test_a_constant_replaces_the_saturated_head_almost_for_free(self):
        """The finding, as an inequality: removing the head is expensive and
        replacing it with a fixed permutation is not."""
        a = sat.ablations(draws=3, count=4, size=8)
        assert a["zeroed"]["delta"] > 0.8
        assert a["shift_by_one"]["delta"] < 0.02
        assert a["zeroed"]["delta"] > 20 * a["control_zeroed"]["delta"]
