"""Tests for quantpost.ts.nonstationary.

The tests that matter here are the ones in `TestAgainstAnswersThatAreNotItself`:
they check the simulated null distributions against MacKinnon's published
response-surface values and against statsmodels, both of which were computed by
someone else, a different way. A simulation that agrees with itself proves
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantpost.ts import nonstationary as ns


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(11223344)


# ---------------------------------------------------------------------------
# mechanics
# ---------------------------------------------------------------------------

class TestOLS:
    def test_recovers_a_known_slope(self, rng):
        n = 400
        x = rng.normal(size=(1, n))
        y = 2.5 + 1.75 * x + rng.normal(scale=0.1, size=(1, n))
        X = np.concatenate([np.ones((1, n, 1)), x[..., None]], axis=2)
        fit = ns.ols(X, y)
        assert fit.beta[0, 0] == pytest.approx(2.5, abs=0.02)
        assert fit.beta[0, 1] == pytest.approx(1.75, abs=0.02)
        assert fit.r2[0] > 0.99

    def test_matches_numpy_lstsq_on_a_single_fit(self, rng):
        n, k = 120, 3
        X = rng.normal(size=(1, n, k))
        y = rng.normal(size=(1, n))
        mine = ns.ols(X, y).beta[0]
        theirs, *_ = np.linalg.lstsq(X[0], y[0], rcond=None)
        assert mine == pytest.approx(theirs, rel=1e-9)

    def test_standard_errors_match_the_textbook_formula(self, rng):
        n, k = 200, 2
        X = rng.normal(size=(1, n, k))
        y = rng.normal(size=(1, n))
        fit = ns.ols(X, y)
        resid = fit.resid[0]
        s2 = resid @ resid / (n - k)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X[0].T @ X[0])))
        assert fit.se[0] == pytest.approx(se, rel=1e-10)

    def test_rejects_more_regressors_than_observations(self):
        with pytest.raises(ValueError, match="cannot support"):
            ns.ols(np.zeros((1, 3, 5)), np.zeros((1, 3)))

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="does not match"):
            ns.ols(np.zeros((2, 10, 2)), np.zeros((3, 10)))

    def test_rejects_unbatched_input(self):
        with pytest.raises(ValueError, match=r"expected X"):
            ns.ols(np.zeros((10, 2)), np.zeros(10))


class TestRandomWalk:
    def test_variance_grows_linearly(self, rng):
        w = ns.random_walk(500, 4000, rng=rng)
        # Var(w_t) = t * sigma^2
        for t in (50, 200, 499):
            assert w[:, t].var() == pytest.approx(t + 1, rel=0.12)

    def test_drift_shows_up_in_the_mean(self, rng):
        w = ns.random_walk(300, 4000, rng=rng, drift=0.05)
        assert w[:, -1].mean() == pytest.approx(0.05 * 300, rel=0.10)

    def test_is_two_dimensional_even_for_one_path(self, rng):
        assert ns.random_walk(10, rng=rng).shape == (1, 10)


class TestTrendTerms:
    @pytest.mark.parametrize("trend,k", [("n", 0), ("c", 1), ("ct", 2)])
    def test_column_count(self, trend, k):
        assert ns._deterministic(20, 3, trend).shape == (3, 20, k)

    def test_rejects_an_unknown_trend(self):
        with pytest.raises(ValueError, match="trend must be"):
            ns._deterministic(20, 1, "quadratic")


class TestNormPPF:
    @pytest.mark.parametrize("p,want", [
        (0.975, 1.959964), (0.995, 2.575829), (0.95, 1.644854),
        (0.5, 0.0), (0.025, -1.959964), (0.001, -3.090232),
    ])
    def test_matches_published_quantiles(self, p, want):
        assert ns._norm_ppf(p) == pytest.approx(want, abs=2e-4)

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 2.0])
    def test_rejects_out_of_range(self, p):
        with pytest.raises(ValueError):
            ns._norm_ppf(p)


# ---------------------------------------------------------------------------
# the external checks
# ---------------------------------------------------------------------------

class TestAgainstAnswersThatAreNotItself:
    """Every assertion here has an answer computed outside this repo."""

    @pytest.mark.parametrize("level", [0.01, 0.05, 0.10])
    def test_df_critical_values_converge_to_mackinnon(self, level):
        got = ns.df_critical_values(1500, reps=30000,
                                    rng=np.random.default_rng(7))[level]
        want = ns.MACKINNON_DF[level]
        assert got == pytest.approx(want, abs=0.05), (
            f"simulated DF {level:.0%} critical value {got:.4f} vs "
            f"MacKinnon {want:.4f}")

    @pytest.mark.parametrize("level", [0.01, 0.05, 0.10])
    def test_eg_critical_values_converge_to_mackinnon(self, level):
        got = ns.eg_critical_values(1500, reps=30000,
                                    rng=np.random.default_rng(8))[level]
        want = ns.MACKINNON_EG[level]
        assert got == pytest.approx(want, abs=0.06), (
            f"simulated EG {level:.0%} critical value {got:.4f} vs "
            f"MacKinnon {want:.4f}")

    def test_eg_critical_value_is_more_negative_than_df(self):
        """The whole reason the two tables differ."""
        df = ns.df_critical_values(600, reps=20000,
                                   rng=np.random.default_rng(9))[0.05]
        eg = ns.eg_critical_values(600, reps=20000,
                                   rng=np.random.default_rng(10))[0.05]
        assert eg < df - 0.3

    def test_adf_stat_matches_statsmodels(self, rng):
        sm = pytest.importorskip("statsmodels.tsa.stattools")
        for seed in (1, 2, 3):
            g = np.random.default_rng(seed)
            y = ns.random_walk(300, 1, rng=g)
            mine = ns.adf_stat(y, lags=0, trend="c")[0]
            theirs = sm.adfuller(y[0], maxlag=0, autolag=None,
                                 regression="c")[0]
            assert mine == pytest.approx(theirs, rel=1e-8)

    @pytest.mark.parametrize("lags", [1, 3])
    def test_augmented_adf_matches_statsmodels(self, lags):
        sm = pytest.importorskip("statsmodels.tsa.stattools")
        g = np.random.default_rng(4242)
        y = ns.random_walk(400, 1, rng=g)
        mine = ns.adf_stat(y, lags=lags, trend="c")[0]
        theirs = sm.adfuller(y[0], maxlag=lags, autolag=None,
                             regression="c")[0]
        assert mine == pytest.approx(theirs, rel=1e-8)

    def test_adf_with_trend_matches_statsmodels(self):
        sm = pytest.importorskip("statsmodels.tsa.stattools")
        g = np.random.default_rng(555)
        y = ns.random_walk(350, 1, rng=g)
        mine = ns.adf_stat(y, lags=0, trend="ct")[0]
        theirs = sm.adfuller(y[0], maxlag=0, autolag=None,
                             regression="ct")[0]
        assert mine == pytest.approx(theirs, rel=1e-8)


# ---------------------------------------------------------------------------
# the claims the post makes
# ---------------------------------------------------------------------------

class TestTheClaim:
    def test_rejection_rate_rises_with_sample_size(self):
        """If this ever flattens, the post's central claim is wrong."""
        rates = [ns.spurious_rejection_rate(
                     n, reps=6000, rng=np.random.default_rng(100 + n)
                 )["rejection_rate"]
                 for n in (25, 100, 400, 1600)]
        assert rates == sorted(rates), rates
        assert rates[0] > 0.40           # already broken at T=25
        assert rates[-1] > 0.90          # and near-certain by T=1600

    def test_scaled_t_does_not_move(self):
        """Phillips (1986): t/sqrt(T) is the quantity that converges."""
        med = [ns.scaled_t_quantiles(
                   n, reps=6000, rng=np.random.default_rng(200 + n))[0.5]
               for n in (100, 400, 1600)]
        assert max(med) - min(med) < 0.04, med
        assert 0.40 < min(med) < 0.50

    def test_r_squared_does_not_go_to_zero(self):
        """Granger-Newbold: R^2 converges to a random variable, not to 0."""
        r2 = [ns.spurious_rejection_rate(
                  n, reps=6000, rng=np.random.default_rng(300 + n))["median_r2"]
              for n in (100, 400, 1600)]
        assert min(r2) > 0.10, r2
        assert max(r2) - min(r2) < 0.05, r2

    def test_drift_makes_it_much_worse(self):
        plain = ns.spurious_rejection_rate(
            400, reps=6000, rng=np.random.default_rng(11))["rejection_rate"]
        drifting = ns.spurious_rejection_rate(
            400, reps=6000, drift=0.1,
            rng=np.random.default_rng(12))["rejection_rate"]
        assert drifting > plain + 0.05

    def test_the_headline_correlation_is_typical_not_exotic(self):
        """r = 0.96 between unrelated trending series is the median case."""
        c = ns.correlation_of_independent_walks(
            1600, reps=6000, drift=0.1, rng=np.random.default_rng(13))
        assert c["median_abs_r"] > 0.90
        assert c["p_abs_r_over_0.9"] > 0.60

    def test_undrifted_correlation_is_scale_free(self):
        """Without drift the |r| distribution does not depend on T at all."""
        med = [ns.correlation_of_independent_walks(
                   n, reps=6000, rng=np.random.default_rng(400 + n)
               )["median_abs_r"]
               for n in (50, 400, 1600)]
        assert max(med) - min(med) < 0.03, med


class TestUsingTheWrongTable:
    def test_df_table_over_rejects_on_eg_residuals(self):
        m = ns.misuse_size(500, reps=8000, rng=np.random.default_rng(21))
        assert m["size_using_eg_table"] == pytest.approx(0.05, abs=0.012)
        assert m["size_using_df_table"] > 0.12

    def test_the_distortion_does_not_shrink_with_more_data(self):
        """The reassuring intuition that fails: this is a bias, not noise."""
        sizes = [ns.misuse_size(n, reps=8000,
                                rng=np.random.default_rng(500 + n)
                                )["size_using_df_table"]
                 for n in (100, 500, 1000)]
        assert min(sizes) > 0.11, sizes


class TestValidation:
    def test_adf_rejects_negative_lags(self, rng):
        with pytest.raises(ValueError, match="non-negative"):
            ns.adf_stat(ns.random_walk(50, 1, rng=rng), lags=-1)

    def test_adf_rejects_too_short_a_series(self, rng):
        with pytest.raises(ValueError, match="too few"):
            ns.adf_stat(ns.random_walk(5, 1, rng=rng), lags=4)

    def test_engle_granger_rejects_mismatched_series(self, rng):
        with pytest.raises(ValueError, match="must match"):
            ns.engle_granger_stat(ns.random_walk(50, 1, rng=rng),
                                  ns.random_walk(40, 1, rng=rng))
