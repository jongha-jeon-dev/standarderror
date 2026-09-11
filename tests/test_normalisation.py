"""LayerNorm's Jacobian, its two dead directions, and where they go.

The algebra is pinned exactly: the closed form against autodiff, the rank, the
null space, and the fact that without a learned gain the nonzero spectrum is
flat. The model measurements are pinned as inequalities, wide enough to
survive a float difference and narrow enough to fail if the finding goes away.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from standarderror.calculus import normalisation as nz  # noqa: E402


def sample(d=16, seed=0):
    return np.random.default_rng(seed).normal(0.0, 2.0, d)


class TestTheJacobian:

    def test_matches_autodiff(self):
        assert nz.against_autodiff(d=8)["max_abs_error"] < 1e-12

    def test_matches_autodiff_with_a_gain(self):
        """The gain multiplies rows, not columns. Getting that backwards gives
        a matrix of the same rank and the same norm."""
        d, g = 6, np.array([0.7, 1.3, 0.9, 1.1, 0.5, 2.0])
        x = torch.tensor(sample(d, 1), requires_grad=True)
        ln = torch.nn.LayerNorm(d, eps=0.0).double()
        with torch.no_grad():
            ln.weight.copy_(torch.tensor(g))
            ln.bias.zero_()
        got = torch.autograd.functional.jacobian(ln, x).numpy()
        assert np.allclose(got, nz.jacobian(sample(d, 1), gamma=g), atol=1e-12)

    def test_rank_is_exactly_d_minus_two(self):
        for d in (4, 8, 16, 128):
            s = nz.spectrum(sample(d, d))
            assert s["rank"] == d - 2, d
            assert s["rank_deficiency"] == 2

    def test_the_null_space_is_the_two_invariances(self):
        """`1` for the shift, `xhat` for the scaling, and they are orthogonal
        because `xhat` is centred -- so the deleted plane is two-dimensional
        rather than one direction counted twice."""
        s = nz.spectrum(sample(32, 2))
        assert s["null_residual"] < 1e-12
        assert s["orthogonality"] < 1e-12

    def test_without_a_gain_it_is_a_scaled_orthogonal_projector(self):
        """Every nonzero singular value equals `1/sigma`, which is a stronger
        statement than rank d - 2 and the reason the picture is simple."""
        x = sample(64, 3)
        s = nz.spectrum(x)
        assert s["spread"] == pytest.approx(1.0, abs=1e-9)
        assert s["largest"] == pytest.approx(1.0 / x.std(), rel=1e-9)

    def test_a_projector_is_idempotent(self):
        p = nz.projector(sample(24, 4))
        assert np.allclose(p @ p, p, atol=1e-12)
        assert np.allclose(p, p.T, atol=1e-12)

    def test_eps_lowers_every_singular_value(self):
        x = sample(16, 5) * 1e-3
        assert (nz.spectrum(x, eps=1e-5)["largest"]
                < nz.spectrum(x, eps=0.0)["largest"])


class TestTheInvariancesThemselves:

    def test_layernorm_ignores_a_shift_and_a_positive_scale(self):
        """The premise the null space is derived from, checked directly rather
        than assumed."""
        d = 32
        ln = torch.nn.LayerNorm(d, elementwise_affine=False).double()
        x = torch.tensor(sample(d, 6))
        base = ln(x)
        assert torch.allclose(ln(x + 4.0), base, atol=1e-10)
        assert torch.allclose(ln(x * 7.0), base, atol=1e-9)

    def test_a_negative_scale_is_not_an_invariance(self):
        """Only positive scaling: `xhat` flips sign with `x`."""
        d = 32
        ln = torch.nn.LayerNorm(d, elementwise_affine=False).double()
        x = torch.tensor(sample(d, 7))
        assert not torch.allclose(ln(-x), ln(x), atol=1e-6)


class TestOnTheModel:

    @pytest.fixture(scope="class")
    def shares(self):
        return nz.deleted_share(count=1, size=4, seed=0)

    def test_every_layernorm_keeps_its_learned_gain_at_rank_d_minus_two(self):
        from standarderror.llm import tiny
        rows = nz.gains(tiny.load()["model"])
        assert len(rows) == 9
        assert all(r["rank"] == tiny.WIDTH - 2 for r in rows)

    def test_the_gain_does_not_distort_the_spectrum_much(self):
        """If it did, calling the Jacobian a scaled projector would be a
        caricature rather than a description."""
        from standarderror.llm import tiny
        rows = nz.gains(tiny.load()["model"])
        assert max(r["spread"] for r in rows) < 3.0

    def test_the_deleted_share_is_close_to_a_random_baseline(self, shares):
        """Two directions out of 128 catch about sqrt(2/128) of an indifferent
        gradient. The measured median should be near that, or the headline
        claim is doing more work than the measurement supports."""
        base = shares["baseline"]["median"]
        for row in shares["norms"]:
            assert row["median"] < 3 * base, row["name"]

    def test_the_tail_is_not_random(self, shares):
        """It is above chance where it matters: at least one norm reaches a
        maximum the random baseline does not."""
        cap = shares["baseline"]["max"]
        assert max(r["max"] for r in shares["norms"]) > cap

    def test_the_residual_restores_the_rank(self):
        from standarderror.llm import tiny
        r = nz.block_rank(layer=0)
        assert r["block_rank"] == tiny.WIDTH
        assert r["branch_rank"] < tiny.WIDTH
        assert r["block_smallest"] > 0.05

    def test_two_exact_invariances_of_the_whole_network(self):
        """The last norm has no residual after it, so what it deletes is
        deleted from the model's output."""
        r = nz.network_invariance(size=4)
        by = {c["name"]: c for c in r["cases"]}
        assert by["shift by 5 * ones"]["delta"] == 0.0
        assert by["scale by 3"]["delta"] == 0.0
        assert by["scale by 3 then shift by 5"]["delta"] == 0.0
        assert abs(by["shift by 5 along one axis"]["delta"]) > 1e-4

    def test_the_scale_invariance_degrades_in_proportion_to_eps_over_var(self):
        """Not a threshold at sqrt(eps): the error tracks eps/var across four
        decades of it, which is the claim the episode makes and the one that
        would quietly become false if someone changed the epsilon."""
        r = nz.scale_sweep(size=4)
        assert len(r["proportional_rows"]) >= 4
        assert r["slope"] == pytest.approx(1.0, abs=0.1)
        assert 0.01 < r["constant"] < 0.2
        spanned = (max(x["eps_over_var"] for x in r["proportional_rows"])
                   / min(x["eps_over_var"] for x in r["proportional_rows"]))
        assert spanned > 1e3

    def test_it_reads_as_exact_at_the_model_s_own_scale(self):
        r = nz.scale_sweep(size=4)
        natural = [x for x in r["rows"] if abs(x["scale"] - 1.0) < 1e-9]
        assert natural and natural[0]["delta"] == 0.0

    def test_scaling_up_cannot_be_blamed_on_the_epsilon(self):
        """At large scales eps/var is vanishing, so any deviation there is
        float32 rounding. Whether a given row shows one is a coin toss; that
        none of them is explained by the epsilon is not."""
        r = nz.scale_sweep(size=4)
        big = [x for x in r["rows"] if x["scale"] >= 10.0]
        assert big
        for x in big:
            assert x["eps_over_var"] < 1e-7
            assert abs(x["delta"]) < 1e-5
            if x["delta"] != 0.0:
                assert x["ratio"] > 100 * r["ratio_max"]

    def test_the_residual_stream_grows_and_one_over_sigma_falls(self):
        r = nz.sigma_ladder(size=4)
        assert r["growth"] > 2.0
        assert r["attenuation"] > 2.0
        assert math.isclose(r["points"][0]["inv_sigma"]
                            / r["points"][-1]["inv_sigma"],
                            r["attenuation"], rel_tol=1e-9)
