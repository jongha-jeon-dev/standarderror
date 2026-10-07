"""The time-series FM's pool, pretraining corpus, normalisation and scoring.

Everything here is exact or structural: a split either overlaps or it does
not, a normalisation either inverts or it does not. Model accuracy is pinned
separately, once the checkpoint exists, as inequalities.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("statsmodels")

from standarderror.tsfm import evaluate as ev  # noqa: E402
from standarderror.tsfm import model as tm  # noqa: E402
from standarderror.tsfm import pool as tp  # noqa: E402
from standarderror.tsfm import synth  # noqa: E402


@pytest.fixture(scope="module")
def pool():
    return tp.pool()


class TestThePoolIsWhatItSays:

    def test_every_series_has_a_licence_and_a_source(self, pool):
        for s in pool:
            assert s.licence and s.source
            assert s.freq in tp.FREQ

    def test_only_public_domain_statsmodels_datasets_are_used(self, pool):
        """Datasets shipped 'with express permission of the original author'
        were permitted to statsmodels, not to this project."""
        permitted_only = {"ccard", "committee", "copper", "cpunish",
                          "scotland", "spector", "star98", "fair"}
        for s in pool:
            if s.source.startswith("statsmodels"):
                assert s.source.split(".")[-1] not in permitted_only
                assert "public domain" in s.licence

    def test_values_are_finite(self, pool):
        for s in pool:
            assert np.isfinite(s.values).all(), s.name

    def test_origins_leave_the_minimum_context(self, pool):
        for s in pool:
            for o in range(s.origins()):
                c, t = s.split(o)
                assert len(t) == s.horizon
                assert len(c) >= tp.MIN_CONTEXT[s.freq]

    def test_origins_do_not_overlap(self, pool):
        s = next(x for x in pool if x.origins() >= 5)
        ends = [len(s.split(o)[0]) for o in range(5)]
        assert all(a - b == s.horizon for a, b in zip(ends, ends[1:]))

    def test_the_target_is_never_inside_its_own_context(self, pool):
        s = next(x for x in pool if x.freq == "M")
        c, t = s.split(0)
        assert np.array_equal(np.concatenate([c, t]), s.values)

    def test_pool_is_deterministic(self):
        a = [(s.name, float(s.values.sum())) for s in tp.pool()]
        b = [(s.name, float(s.values.sum())) for s in tp.pool()]
        assert a == b


class TestTheSyntheticBank:

    def test_it_is_deterministic_in_its_seed(self):
        assert np.array_equal(synth.bank(8, seed=3), synth.bank(8, seed=3))
        assert not np.array_equal(synth.bank(8, seed=3), synth.bank(8, seed=4))

    def test_it_is_finite_and_the_right_shape(self):
        b = synth.bank(16, seed=0)
        assert b.shape == (16, synth.LENGTH) and np.isfinite(b).all()

    def test_the_mix_is_a_distribution(self):
        assert sum(synth.MIX.values()) == pytest.approx(1.0)

    def test_the_bank_no_longer_teaches_that_trends_revert(self):
        """Version 1 had no trend family and continued its own recent trend
        in 48.4% of short windows -- structurally mean-reverting -- against
        69.0% in the real annual pool. Pinned above a coin, and the trend
        family well above it, but deliberately *not* pinned to the pool's
        number, which the mix was not tuned to."""
        def persist(rows, rng):
            hit = []
            for _ in range(2000):
                r = rows[rng.integers(len(rows))]
                e = rng.integers(30, len(r) - 6)
                c, tt = r[e - 30:e], r[e:e + 6]
                s = np.polyfit(np.arange(10), c[-10:], 1)[0]
                if abs(s) > 1e-12:
                    hit.append(np.sign(tt.mean() - c[-1]) == np.sign(s))
            return np.mean(hit)
        rng = np.random.default_rng(0)
        assert persist(synth.bank(1500, seed=9), rng) > 0.53
        fam = [synth.trend(np.random.default_rng(i), 512) for i in range(200)]
        assert persist(fam, rng) > 0.62

    def test_levels_and_scales_vary(self):
        """Otherwise the model could rely on the numbers being near zero
        instead of on its normalisation."""
        b = synth.bank(200, seed=0)
        assert np.percentile(np.abs(b.mean(1)), 90) > 1.0
        assert b.std(1).max() / b.std(1).min() > 20


class TestNormalisation:

    def test_it_inverts_on_the_observed_part(self):
        x = np.linspace(100, 140, 60) + np.sin(np.arange(60))
        v, m, loc, scale = tm.normalise(x)
        back = v[-60:] * scale + loc
        assert np.allclose(back, x, atol=1e-4)
        assert m[-60:].all() and not m[:-60].any()

    def test_long_contexts_are_truncated_to_the_most_recent(self):
        x = np.arange(1000, dtype=float)
        v, m, loc, scale = tm.normalise(x)
        assert m.all()
        assert np.allclose(v * scale + loc, x[-tm.CONTEXT:], atol=1e-3)

    def test_a_constant_series_is_not_blown_up(self):
        v, m, loc, scale = tm.normalise(np.full(50, 250.0))
        assert scale >= 0.25 and np.abs(v).max() < 1e-6

    def test_patches_cover_the_context_exactly(self):
        assert tm.TOKENS * tm.PATCH == tm.CONTEXT


class TestScoring:

    def test_a_perfect_forecast_scores_zero_and_full_coverage(self):
        c = np.random.default_rng(0).normal(size=80).cumsum()
        t = np.arange(6.0)
        q = np.repeat(t[:, None], len(tm.QUANTILES), 1)
        s = ev.score(c, t, q, 1)
        assert s["mase"] == 0 and s["wql"] == 0 and s["coverage"] == 1

    def test_seasonal_naive_is_exact_on_a_pure_season(self):
        c = np.tile(np.array([1.0, 5, 2, 8]), 20)
        q = ev.naive(c, 8, 4)
        s = ev.score(c, np.tile(np.array([1.0, 5, 2, 8]), 2), q, 4)
        assert s["mase"] == pytest.approx(0, abs=1e-9) or np.isnan(s["mase"])

    def test_mase_of_one_is_in_sample_seasonal_naive_skill(self):
        rng = np.random.default_rng(1)
        c = rng.normal(size=200).cumsum()
        err = np.abs(np.diff(c)).mean()
        t = np.full(6, c[-1] + err)
        q = np.repeat(np.full(6, c[-1])[:, None], len(tm.QUANTILES), 1)
        assert ev.score(c, t, q, 1)["mase"] == pytest.approx(1.0)

    def test_pinball_matches_the_formula(self):
        rng = np.random.default_rng(2)
        p = rng.normal(size=(4, tm.OUT, len(tm.QUANTILES)))
        y = rng.normal(size=(4, tm.OUT))
        q = np.array(tm.QUANTILES)
        d = y[..., None] - p
        want = np.maximum(q * d, (q - 1) * d).mean()
        got = float(tm.pinball(torch.tensor(p), torch.tensor(y),
                               torch.ones(4, tm.OUT, dtype=torch.float64)))
        assert got == pytest.approx(want)

    def test_baseline_quantiles_are_ordered(self):
        c = np.random.default_rng(3).normal(size=100).cumsum()
        q = ev.naive(c, 12, 1)
        assert (np.diff(q, axis=1) >= 0).all()


class TestTheModelShape:

    def test_it_is_in_the_published_tsfm_size_range(self):
        """Between TTM (under 1M) and Moirai-2 (about 11M)."""
        n = tm.parameters(tm.build())
        assert 1_000_000 < n < 11_000_000

    def test_forecast_returns_the_horizon_asked_for(self):
        bundle = {"model": tm.build().eval()}
        out = tm.forecast(bundle, [np.arange(40.0), np.arange(300.0)], 6)
        assert out.shape == (2, 6, len(tm.QUANTILES))

    def test_a_short_context_does_not_attend_to_nothing(self):
        bundle = {"model": tm.build().eval()}
        out = tm.forecast(bundle, [np.arange(5.0)], 3)
        assert np.isfinite(out).all()


class TestTheETSBaselineIsETS:

    def test_it_does_not_silently_become_naive(self):
        """The regression: a renamed argument made every fit raise, the
        fallback caught it, and the ETS row equalled seasonal naive."""
        rng = np.random.default_rng(5)
        c = 50 + np.cumsum(rng.normal(0.3, 1.0, 60))
        q, fell = ev.ets(c, 6, 1)
        assert not fell
        assert not np.allclose(q, ev.naive(c, 6, 1))

    def test_the_fallback_is_counted(self, monkeypatch):
        """Forced, rather than hoped for: ETS will fit a three-point series
        without complaint, so the only reliable failure is one we cause."""
        import statsmodels.tsa.exponential_smoothing.ets as ets_mod

        def boom(*a, **k):
            raise RuntimeError("forced")
        monkeypatch.setattr(ets_mod, "ETSModel", boom)
        f = ev.baseline_forecaster("ets")
        c = np.arange(40.0)
        q = f([c], 3, 1)
        assert f.windows == 1 and f.fallbacks == 1
        assert np.allclose(q[0], ev.naive(c, 3, 1))


from standarderror.tsfm import compress as cp  # noqa: E402


class TestCompressionOperators:

    @pytest.fixture(scope="class")
    def model(self):
        torch.manual_seed(0)
        return tm.build().eval()

    def test_the_original_is_never_modified(self, model):
        before = {n: p.clone() for n, p in model.named_parameters()}
        for m in (cp.quantize(model, 4), cp.prune(model, 0.5),
                  cp.low_rank(model, 0.25)):
            del m
        for n, p in model.named_parameters():
            assert torch.equal(p, before[n])

    def test_the_attention_input_projection_is_included(self, model):
        """PyTorch stores it as a bare parameter; a loop over nn.Linear
        skips it, and a third of the attention weights go uncompressed."""
        names = [n for n, _ in cp._targets(model)]
        assert any("in_proj_weight" in n for n in names)

    def test_quantised_weights_sit_on_the_grid(self, model):
        q = cp.quantize(model, 4)
        for (_, p), (_, o) in zip(cp._targets(q), cp._targets(model)):
            s = o.abs().amax(1, keepdim=True) / 7
            k = p / s
            assert torch.allclose(k, k.round(), atol=1e-4)
            assert k.abs().max() <= 7 + 1e-4

    def test_rounding_error_is_at_most_half_a_step(self, model):
        q = cp.quantize(model, 6)
        for (_, p), (_, o) in zip(cp._targets(q), cp._targets(model)):
            s = o.abs().amax(1, keepdim=True) / 31
            assert ((p - o).abs() <= s / 2 + 1e-7).all()

    def test_more_bits_means_less_error(self, model):
        errs = [np.mean(list(cp.weight_error(model, cp.quantize(model, b)).values()))
                for b in (3, 4, 6, 8)]
        assert errs == sorted(errs, reverse=True)

    def test_pruning_hits_the_asked_sparsity(self, model):
        m = cp.prune(model, 0.6)
        for _, p in cp._targets(m):
            assert (p == 0).float().mean().item() == pytest.approx(0.6, abs=0.01)

    def test_low_rank_has_the_asked_rank(self, model):
        m = cp.low_rank(model, 0.25)
        for _, p in cp._targets(m):
            r = torch.linalg.matrix_rank(p).item()
            assert r <= int(np.ceil(0.25 * min(p.shape)))

    def test_size_accounting_includes_the_scales_and_the_rest(self, model):
        s8, s4 = cp.size_bytes(model, bits=8), cp.size_bytes(model, bits=4)
        assert 3.0 < s8["ratio"] < 4.0     # not 4.0: scales and fp32 rest
        assert s4["ratio"] > s8["ratio"]

    def test_a_bitmap_breaks_even_with_fp16_at_one_sixteenth(self, model):
        """The claim this replaced said 'about 50%', which is the CSR
        break-even. With a one-bit mask it is 2(1 - s) + 1/8 = 2, s = 1/16."""
        dense = cp.fp16_dense_bytes(model)
        below = cp.size_bytes(model, sparsity=0.05)["bytes"]
        above = cp.size_bytes(model, sparsity=0.08)["bytes"]
        assert below > dense > above


class TestPairedComparison:

    def _rows(self, name, values):
        return [{"series": name, "freq": "M", "origin": i, "mase": v,
                 "coverage": 0.8} for i, v in enumerate(values)]

    def test_identical_forecasters_tie(self):
        a = self._rows("s", [1.0, 2.0, 0.5])
        c = ev.compare(a, a)["M"]
        assert c["ratio"] == pytest.approx(1.0) and c["a_wins"] == 0

    def test_a_uniformly_better_forecaster_wins_everywhere(self):
        a, b = self._rows("s", [0.5, 1.0, 0.25]), self._rows("s", [1.0, 2.0, 0.5])
        c = ev.compare(a, b)["M"]
        assert c["ratio"] == pytest.approx(0.5) and c["a_wins"] == 1.0

    def test_the_bootstrap_resamples_series_not_windows(self):
        """One series with many origins is one unit of evidence: with a single
        series, every bootstrap draw is that series, so the interval
        collapses to the point instead of pretending to be narrow."""
        a = self._rows("only", list(np.linspace(0.5, 1.5, 30)))
        b = self._rows("only", [1.0] * 30)
        c = ev.compare(a, b)["M"]
        assert c["series"] == 1
        assert c["low"] == pytest.approx(c["high"])


class TestForecastsAreProperQuantiles:

    def test_returned_quantiles_are_sorted_at_every_step(self):
        """The docstring said sorted; the first evaluation ran unsorted, with
        3.5-4.2% of steps crossed. An untrained model crosses constantly,
        which makes it the right thing to test on."""
        torch.manual_seed(1)
        bundle = {"model": tm.build().eval()}
        rng = np.random.default_rng(0)
        ctx = [rng.normal(size=80).cumsum() for _ in range(32)]
        assert tm.crossing_rate(bundle, ctx, 12) > 0.1
        q = tm.forecast(bundle, ctx, 12)
        assert (np.diff(q, axis=-1) >= 0).all()


class TestTheCommittedCheckpoint:
    """Pinned as inequalities: properties of one set of weights."""

    @pytest.fixture(scope="class")
    def bundle(self):
        return tm.load()

    @pytest.fixture(scope="class")
    def series(self):
        return [s for s in tp.pool() if s.origins() >= 1]

    def test_it_verifies_and_has_the_published_shape(self, bundle):
        assert bundle["parameters"] == 3_006_144

    def test_it_was_trained_on_the_version_two_bank(self, bundle):
        blob = torch.load(tm.checkpoint_path(), map_location="cpu",
                          weights_only=False)
        assert blob["bank"][3] == synth.VERSION == 2

    def test_its_intervals_are_near_nominal(self, bundle, series):
        agg = ev.aggregate(ev.evaluate(ev.model_forecaster(bundle), series))
        for f in ("W", "M", "A"):
            assert 0.74 < agg[f]["coverage"] < 0.86

    def test_int8_is_free_and_int3_is_not(self, bundle, series):
        def mase_a(model):
            b = dict(bundle, model=model)
            return ev.aggregate(ev.evaluate(ev.model_forecaster(b), series))["A"]["mase"]
        full = mase_a(bundle["model"])
        assert abs(mase_a(cp.quantize(bundle["model"], 8)) - full) < 0.03
        assert mase_a(cp.quantize(bundle["model"], 3)) > full * 1.15

    def test_quantisation_keeps_coverage_while_pruning_loses_it(self, bundle, series):
        """The hypothesis was that coverage fails first under compression.
        Under quantisation it does not; under pruning it does."""
        def cov_q(model):
            b = dict(bundle, model=model)
            return ev.aggregate(ev.evaluate(ev.model_forecaster(b), series))["Q"]["coverage"]
        assert cov_q(cp.quantize(bundle["model"], 4)) > 0.70
        assert cov_q(cp.prune(bundle["model"], 0.7)) < 0.55


@pytest.mark.skipif(not __import__("os").environ.get("SERR_SLOW_TESTS"),
                    reason="fits ETS on 1,561 windows; set SERR_SLOW_TESTS=1")
class TestItDoesNotBeatETS:

    def test_ets_wins_on_the_annual_pool_beyond_noise(self):
        series = [s for s in tp.pool() if s.origins() >= 1]
        annual = [s for s in series if s.freq == "A"]
        fm = ev.evaluate(ev.model_forecaster(tm.load()), annual)
        f = ev.baseline_forecaster("ets")
        e = ev.evaluate(f, annual)
        assert f.fallbacks == 0
        c = ev.compare(fm, e)["A"]
        assert c["low"] > 1.2


def test_nothing_earlier_in_the_suite_left_torch_in_float64():
    """The regression that made two checkpoint tests fail only in the full
    run: `numerics.steps.edge_of_stability` set the global default dtype and
    did not restore it."""
    from standarderror.numerics import steps
    before = torch.get_default_dtype()
    steps.edge_of_stability(0.05, steps=5, sharpness_iters=3)
    assert torch.get_default_dtype() == before
