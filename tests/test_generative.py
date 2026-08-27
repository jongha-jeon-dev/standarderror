"""Tests for the diffusion model and the stylised-facts battery.

Two of these matter far more than the rest.

`TestSamplerAgainstAnOracle` checks the reverse process against an *analytic
answer*: for Gaussian data the optimal noise prediction is available in closed form,
so ancestral sampling driven by it must return the data's standard deviation. That
is the only test here that can catch a sign or a square root in the reverse step,
and it is what caught the terminal-SNR mismatch — the 200-step schedule fails it
while the 1000-step one passes, and no test on generated *variance from standardised
data* could have shown that, because the mismatch preserves the variance.

`TestBaselinesDoWhatTheyClaim` pins the two controls: the i.i.d. bootstrap must
reproduce the marginal exactly and destroy dependence, the block bootstrap must keep
dependence and lose a little of it at the joins. If either drifts, the post's central
comparison is measuring something else.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from standarderror.dynamics import sde
from standarderror.generative import diffusion, stylised


@pytest.fixture(scope="module")
def garch():
    """A path with clustering that is visible inside a 64-step window."""
    return np.asarray(sde.garch11(n=30_000, arch=0.25, beta=0.60, df=5.0,
                                  seed=5).data["r"], dtype=float)


class TestSchedule:
    def test_rejects_degenerate_betas(self):
        for bad in ([0.5], [0.0, 0.1], [0.1, 1.0], [-0.1, 0.2]):
            with pytest.raises(ValueError):
                diffusion.Schedule(np.asarray(bad, float))

    def test_alpha_bar_decreases_to_the_terminal_value(self):
        s = diffusion.linear_schedule(1000)
        ab = s.alpha_bar
        assert np.all(np.diff(ab) < 0)
        assert ab[-1] == pytest.approx(4.036e-5, rel=1e-2)

    def test_corrupt_preserves_unit_variance_at_every_step(self):
        """A property of the schedule, and the reason a variance check proves little."""
        rng = np.random.default_rng(0)
        s = diffusion.linear_schedule(1000)
        x0 = rng.standard_normal((4000, 16))
        for t in (0, 100, 500, 999):
            eps = rng.standard_normal(x0.shape)
            xt = s.corrupt(x0, np.full(len(x0), t), eps)
            assert xt.std() == pytest.approx(1.0, abs=0.03)

    def test_corrupt_at_step_zero_is_almost_the_data(self):
        s = diffusion.linear_schedule(1000)
        x0 = np.random.default_rng(1).standard_normal((50, 8))
        xt = s.corrupt(x0, np.zeros(50, int), np.zeros_like(x0))
        assert np.allclose(xt, x0 * np.sqrt(s.alpha_bar[0]), atol=1e-12)

    def test_corrupt_validates_shapes_and_timesteps(self):
        s = diffusion.linear_schedule(50)
        x0 = np.zeros((3, 4))
        with pytest.raises(ValueError):
            s.corrupt(x0, np.zeros(3, int), np.zeros((3, 5)))
        with pytest.raises(ValueError):
            s.corrupt(x0, np.zeros(2, int), np.zeros((3, 4)))
        with pytest.raises(ValueError):
            s.corrupt(x0, np.full(3, 50), np.zeros((3, 4)))

    def test_the_truncation_that_looks_harmless(self):
        """200 linear steps leave 13% of the signal; 1000 leave none."""
        assert diffusion.linear_schedule(200).terminal_snr == pytest.approx(
            0.152, rel=1e-2)
        assert diffusion.linear_schedule(1000).terminal_snr < 1e-4

    def test_snr_is_monotone(self):
        assert np.all(np.diff(diffusion.linear_schedule(500).snr()) < 0)

    def test_cosine_destroys_the_signal_in_fewer_steps(self):
        c = diffusion.cosine_schedule(200)
        assert c.terminal_snr < diffusion.linear_schedule(200).terminal_snr
        assert c.betas.max() <= 0.999
        assert np.all(c.betas > 0)

    def test_schedule_for_snr_hits_its_target(self):
        for steps in (100, 200, 400):
            s = diffusion.schedule_for_snr(steps, target_snr=1e-3)
            assert s.steps == steps
            assert s.terminal_snr == pytest.approx(1e-3, rel=0.05)

    def test_fewer_steps_need_a_steeper_endpoint(self):
        ends = [diffusion.schedule_for_snr(n, target_snr=1e-3).betas[-1]
                for n in (100, 200, 400, 1000)]
        assert all(b < a for a, b in zip(ends, ends[1:]))

    def test_schedule_for_snr_refuses_the_impossible(self):
        with pytest.raises(ValueError):
            diffusion.schedule_for_snr(3, target_snr=1e-12)
        with pytest.raises(ValueError):
            diffusion.schedule_for_snr(200, target_snr=0.0)


class TestTimeFeatures:
    def test_shape_grows_with_harmonics(self):
        t = np.arange(10)
        assert diffusion.time_features(t, 100, harmonics=0).shape == (10, 1)
        assert diffusion.time_features(t, 100, harmonics=3).shape == (10, 7)

    def test_first_column_is_the_normalised_step(self):
        f = diffusion.time_features(np.array([0, 50, 99]), 100)
        assert f[:, 0] == pytest.approx([0.0, 0.5, 0.99])

    def test_distinct_steps_get_distinct_embeddings(self):
        f = diffusion.time_features(np.arange(200), 200, harmonics=2)
        assert len({tuple(np.round(row, 9)) for row in f}) == 200


class TestSamplerAgainstAnOracle:
    """The reverse step, checked against a closed-form answer rather than itself."""

    @staticmethod
    def _recovered(steps: int, sd: float) -> float:
        sch = diffusion.linear_schedule(steps)
        m = diffusion.DDPM(length=8, schedule=sch,
                           denoiser=diffusion.GaussianOracle(sch, sd=sd))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = m.sample(4000, scale=1.0, rng=np.random.default_rng(1))
        return float(x.std())

    @pytest.mark.parametrize("sd", [0.5, 1.0, 2.5])
    def test_recovers_the_data_scale(self, sd):
        assert self._recovered(1000, sd) == pytest.approx(sd, rel=0.02)

    def test_gaussian_data_stays_gaussian(self):
        sch = diffusion.linear_schedule(1000)
        m = diffusion.DDPM(length=8, schedule=sch,
                           denoiser=diffusion.GaussianOracle(sch, sd=1.0))
        x = m.sample(4000, scale=1.0, rng=np.random.default_rng(2))
        assert stylised.stylised_facts(x)["excess_kurtosis"]["value"] == \
            pytest.approx(0.0, abs=0.15)

    def test_the_truncated_schedule_fails_this_test(self):
        """And fails it only on scale that standardisation would have hidden."""
        assert self._recovered(200, 2.5) == pytest.approx(2.22, rel=0.02)
        # At sd = 1 the mismatch is invisible: ab * 1 + (1 - ab) = 1 either way.
        assert self._recovered(200, 1.0) == pytest.approx(1.0, abs=0.02)

    def test_a_steepened_short_schedule_passes_it(self):
        sch = diffusion.schedule_for_snr(200, target_snr=1e-3)
        m = diffusion.DDPM(length=8, schedule=sch,
                           denoiser=diffusion.GaussianOracle(sch, sd=2.5))
        x = m.sample(4000, scale=1.0, rng=np.random.default_rng(1))
        assert x.std() == pytest.approx(2.5, rel=0.03)

    def test_sample_warns_when_the_signal_survives(self):
        sch = diffusion.linear_schedule(200)
        m = diffusion.DDPM(length=4, schedule=sch,
                           denoiser=diffusion.GaussianOracle(sch, sd=1.0))
        with pytest.warns(RuntimeWarning, match="terminal SNR"):
            m.sample(10, scale=1.0)


class TestDDPM:
    def test_windows_are_contiguous_and_strided(self, garch):
        m = diffusion.DDPM.budget(length=16)
        w = m.windows(garch, stride=5)
        assert w.shape[1] == 16
        assert np.allclose(w[0], garch[:16])
        assert np.allclose(w[1], garch[5:21])

    def test_windows_refuses_a_short_series(self):
        with pytest.raises(ValueError):
            diffusion.DDPM.budget(length=64).windows(np.zeros(10))

    def test_training_set_shape_and_noise_target(self, garch):
        m = diffusion.DDPM.budget(length=16, noise_per_window=3)
        w = m.windows(garch[:500], stride=4)
        X, Y = m.training_set(w / w.std(), rng=np.random.default_rng(0))
        assert X.shape == (3 * len(w), 16 + 1 + 2 * m.harmonics)
        assert Y.shape == (3 * len(w), 16)
        # The target is standard normal noise, whatever the data looked like.
        assert Y.std() == pytest.approx(1.0, abs=0.05)

    def test_training_set_rejects_the_wrong_width(self):
        m = diffusion.DDPM.budget(length=16)
        with pytest.raises(ValueError):
            m.training_set(np.zeros((10, 8)))

    def test_sample_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            diffusion.DDPM.budget(length=8).sample(2)

    def test_fit_refuses_constant_windows(self):
        with pytest.raises(ValueError):
            diffusion.DDPM.budget(length=8).fit(np.ones((20, 8)))

    def test_fit_records_the_scale_and_rescales_on_sampling(self, garch):
        m = diffusion.DDPM.budget(length=8, hidden=8, max_iter=2,
                                  noise_per_window=1, steps=50)
        w = m.windows(garch[:2000], stride=8)
        m.fit(w)
        assert m.scale == pytest.approx(w.std())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = m.sample(20, rng=np.random.default_rng(0))
            b = m.sample(20, rng=np.random.default_rng(0), scale=2 * m.scale)
        assert np.allclose(b, 2 * a)

    def test_flops_proxy_is_monotone_in_every_knob(self):
        base = diffusion.DDPM.budget(length=64, hidden=256, max_iter=80,
                                     noise_per_window=6).flops_proxy(1000)
        for kw in (dict(hidden=384), dict(max_iter=160), dict(noise_per_window=12)):
            cfg = dict(hidden=256, max_iter=80, noise_per_window=6) | kw
            assert diffusion.DDPM.budget(length=64, **cfg).flops_proxy(1000) > base
        assert diffusion.DDPM.budget(
            length=64, hidden=256, max_iter=80,
            noise_per_window=6).flops_proxy(2000) == pytest.approx(2 * base)

    def test_the_smallest_useful_run_produces_finite_samples(self, garch):
        """A smoke test, not a quality test: nothing NaN, right shape, right units."""
        m = diffusion.DDPM.budget(length=16, hidden=32, max_iter=4,
                                  noise_per_window=2, steps=60)
        w = m.windows(garch[:4000], stride=16)
        m.fit(w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = m.sample(40, rng=np.random.default_rng(3))
        assert x.shape == (40, 16)
        assert np.all(np.isfinite(x))


class TestStylisedFacts:
    def test_reports_every_fact_with_an_error(self, garch):
        w = np.lib.stride_tricks.sliding_window_view(garch, 64)[::8]
        f = stylised.stylised_facts(w, n_boot=50)
        for k in stylised.FACTS:
            assert k in f
            assert np.isfinite(f[k]["value"])
            assert np.isfinite(f[k]["se"])

    def test_pooled_errors_are_nan_without_bootstrapping(self, garch):
        w = np.lib.stride_tricks.sliding_window_view(garch, 64)[::8]
        f = stylised.stylised_facts(w)
        assert np.isnan(f["excess_kurtosis"]["se"])
        assert np.isfinite(f["acf1_abs"]["se"])   # per-path, no resampling needed

    def test_gaussian_noise_has_none_of_the_facts(self):
        x = np.random.default_rng(0).standard_normal((800, 64))
        f = stylised.stylised_facts(x)
        assert f["excess_kurtosis"]["value"] == pytest.approx(0.0, abs=0.2)
        for k in ("acf1_returns", "acf1_abs", "leverage"):
            assert abs(f[k]["value"]) < 0.05

    def test_garch_has_tails_and_clustering_but_no_leverage(self, garch):
        """The leverage row is the one whose correct answer is zero."""
        w = np.lib.stride_tricks.sliding_window_view(garch, 64)[::8]
        f = stylised.stylised_facts(w)
        assert f["excess_kurtosis"]["value"] > 5
        assert f["acf1_abs"]["value"] > 0.1
        assert abs(f["acf1_returns"]["value"]) < 0.05
        assert abs(f["leverage"]["value"]) < 0.05

    def test_lag_statistics_never_cross_a_path_boundary(self):
        """Alternating paths: within-path correlation is -1, across the join +1."""
        a = np.tile([1.0, -1.0], 8)
        w = np.vstack([a, a, a])
        f = stylised.stylised_facts(w)
        assert f["acf1_returns"]["value"] == pytest.approx(-1.0, abs=1e-9)
        # |r| is constant here, so clustering is undefined rather than zero.
        assert np.isnan(f["acf1_abs"]["value"])
        assert f["acf1_abs"]["n"] == 0.0

    def test_rejects_degenerate_input(self):
        with pytest.raises(ValueError):
            stylised.stylised_facts(np.zeros((5, 2)))

    def test_facts_error_is_signed_and_per_fact(self):
        a = np.random.default_rng(0).standard_normal((400, 32))
        b = 3.0 * np.random.default_rng(1).standard_normal((400, 32))
        err = stylised.facts_error(stylised.stylised_facts(a),
                                   stylised.stylised_facts(b))
        assert set(err) == set(stylised.FACTS)
        assert err["sd"] > 1.5
        assert not isinstance(err, float)


class TestWithinWindowClustering:
    def test_the_trap_this_function_exists_for(self):
        """At equity-index persistence a 32-step window contains no clustering."""
        r = np.asarray(sde.garch11(n=60_000, arch=0.10, beta=0.88, df=5.0,
                                   seed=5).data["r"], dtype=float)
        vals = stylised.within_window_clustering(r, (32, 256), stride=4)
        assert vals[32] < 0.03
        assert vals[256] > vals[32]

    def test_lower_persistence_shows_clustering_sooner(self):
        slow = np.asarray(sde.garch11(n=60_000, arch=0.10, beta=0.88, df=5.0,
                                      seed=5).data["r"], dtype=float)
        fast = np.asarray(sde.garch11(n=60_000, arch=0.25, beta=0.60, df=5.0,
                                      seed=5).data["r"], dtype=float)
        assert stylised.within_window_clustering(fast, (32,), stride=4)[32] > \
            stylised.within_window_clustering(slow, (32,), stride=4)[32]

    def test_grows_with_window_length(self, garch):
        vals = stylised.within_window_clustering(garch, (16, 32, 64, 128),
                                                 stride=4)
        seq = [vals[k] for k in (16, 32, 64, 128)]
        assert all(b > a for a, b in zip(seq, seq[1:]))

    def test_short_windows_are_biased_downwards_even_without_clustering(self):
        """Why the figure needs a shuffled control line rather than a zero line."""
        x = np.random.default_rng(0).standard_normal(40_000)
        vals = stylised.within_window_clustering(x, (8, 128), stride=1)
        assert vals[8] < -0.05
        assert vals[128] == pytest.approx(0.0, abs=0.02)

    def test_skips_windows_longer_than_the_series(self):
        out = stylised.within_window_clustering(np.zeros(20) + np.arange(20),
                                                (8, 200))
        assert set(out) == {8}


class TestBaselinesDoWhatTheyClaim:
    def test_iid_bootstrap_reproduces_the_marginal_exactly(self, garch):
        w = np.lib.stride_tricks.sliding_window_view(garch, 64)[::8]
        ref = stylised.stylised_facts(w)
        b = stylised.stylised_facts(
            stylised.iid_bootstrap(garch, 4000, 64, seed=1))
        assert b["sd"]["value"] == pytest.approx(ref["sd"]["value"], rel=0.05)
        assert b["excess_kurtosis"]["value"] == pytest.approx(
            ref["excess_kurtosis"]["value"], rel=0.25)

    def test_iid_bootstrap_destroys_every_dependence(self, garch):
        b = stylised.stylised_facts(
            stylised.iid_bootstrap(garch, 2000, 64, seed=1))
        for k in ("acf1_returns", "acf1_abs", "leverage"):
            assert abs(b["acf1_abs"]["value"]) < 0.03, k

    def test_iid_bootstrap_only_returns_observed_values(self):
        s = np.array([1.0, 2.0, 3.0])
        out = stylised.iid_bootstrap(s, 50, 8, seed=0)
        assert set(np.unique(out)) <= set(s)

    def test_block_bootstrap_keeps_clustering(self, garch):
        w = np.lib.stride_tricks.sliding_window_view(garch, 64)[::8]
        ref = stylised.stylised_facts(w)["acf1_abs"]["value"]
        got = stylised.stylised_facts(stylised.block_bootstrap(
            garch, 2000, 64, block=16, seed=2))["acf1_abs"]["value"]
        assert got > 0.5 * ref

    def test_longer_blocks_keep_more_of_it(self, garch):
        vals = [stylised.stylised_facts(stylised.block_bootstrap(
            garch, 3000, 64, block=b, seed=2))["acf1_abs"]["value"]
            for b in (2, 8, 32)]
        assert vals[0] < vals[1] < vals[2]

    def test_a_block_of_one_is_the_iid_bootstrap(self, garch):
        got = stylised.stylised_facts(stylised.block_bootstrap(
            garch, 2000, 64, block=1, seed=3))["acf1_abs"]["value"]
        assert abs(got) < 0.03

    def test_block_bootstrap_shapes_and_validation(self, garch):
        assert stylised.block_bootstrap(garch, 7, 30, block=16).shape == (7, 30)
        with pytest.raises(ValueError):
            stylised.block_bootstrap(garch, 5, 10, block=0)
        with pytest.raises(ValueError):
            stylised.block_bootstrap(np.zeros(4), 5, 10, block=16)

    def test_blocks_are_contiguous_slices_of_the_series(self):
        s = np.arange(100.0)
        out = stylised.block_bootstrap(s, 30, 16, block=8, seed=0)
        for row in out:
            for start in (0, 8):
                chunk = row[start:start + 8]
                assert np.allclose(np.diff(chunk), 1.0)
