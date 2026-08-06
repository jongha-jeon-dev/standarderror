"""Conformal prediction and causal ground truth.

The conformal tests are Monte Carlo coverage checks, not shape checks. A wrapper
that returns intervals of the wrong width passes a shape test and fails these.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from quantpost.uq import causal, conformal, multiplicity


def linear_setup(n_train=400, n_calib=400, n_test=2000, *, seed=0,
                 heteroskedastic=False):
    rng = np.random.default_rng(seed)

    def gen(n):
        x = rng.uniform(-3, 3, n)
        s = (0.3 + 0.6 * np.abs(x)) if heteroskedastic else np.full(n, 1.0)
        y = 2.0 * x + s * rng.standard_normal(n)
        return x, y, s

    xt, yt, _ = gen(n_train)
    beta = np.polyfit(xt, yt, 1)
    xc, yc, sc = gen(n_calib)
    xs, ys, ss = gen(n_test)
    return {"beta": beta,
            "calib": (np.polyval(beta, xc), yc, xc, sc),
            "test": (np.polyval(beta, xs), ys, xs, ss)}


class TestConformalQuantile:
    def test_uses_the_n_plus_one_index(self):
        s = np.arange(1.0, 11.0)          # n = 10
        # ceil(11 * 0.9) = 10 -> the 10th smallest
        assert conformal.conformal_quantile(s, 0.1) == 10.0
        # ceil(11 * 0.5) = 6 -> the 6th smallest
        assert conformal.conformal_quantile(s, 0.5) == 6.0

    def test_returns_infinity_when_n_is_too_small(self):
        """With 5 points you cannot honestly claim 99% coverage."""
        assert conformal.conformal_quantile(np.arange(5.0), 0.01) == float("inf")

    def test_differs_from_the_naive_quantile(self):
        s = np.arange(1.0, 21.0)
        assert conformal.conformal_quantile(s, 0.1) > np.quantile(s, 0.9)

    def test_empty_scores_raise(self):
        with pytest.raises(ValueError):
            conformal.conformal_quantile(np.array([]), 0.1)


class TestSplitConformal:
    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
    def test_achieves_target_coverage(self, alpha):
        d = linear_setup(seed=1)
        cp, cy, _, _ = d["calib"]
        tp, ty, _, _ = d["test"]
        iv = conformal.split_conformal(cp, cy, tp, alpha=alpha)
        cov = iv.summary(ty)["empirical_coverage"]
        assert cov >= 1 - alpha - 0.03
        assert cov <= 1 - alpha + 0.06      # not absurdly conservative

    def test_coverage_holds_across_seeds(self):
        """The guarantee is about the procedure, so average over draws."""
        covs = []
        for s in range(12):
            d = linear_setup(n_calib=200, n_test=500, seed=s)
            cp, cy, _, _ = d["calib"]
            tp, ty, _, _ = d["test"]
            covs.append(conformal.split_conformal(
                cp, cy, tp, alpha=0.1).summary(ty)["empirical_coverage"])
        assert np.mean(covs) >= 0.895

    def test_width_is_constant(self):
        d = linear_setup(seed=2)
        cp, cy, _, _ = d["calib"]
        tp, _, _, _ = d["test"]
        iv = conformal.split_conformal(cp, cy, tp)
        assert np.allclose(iv.width, iv.width[0])


class TestNormalisedConformal:
    def test_beats_split_conformal_on_conditional_coverage(self):
        """The point of the method: constant width under heteroskedasticity
        over-covers the easy region and under-covers the hard one."""
        d = linear_setup(n_calib=800, n_test=4000, seed=3, heteroskedastic=True)
        cp, cy, cx, cs = d["calib"]
        tp, ty, tx, ts = d["test"]

        plain = conformal.split_conformal(cp, cy, tp, alpha=0.1)
        adapt = conformal.normalised_conformal(cp, cy, cs, tp, ts, alpha=0.1)

        # Both must hold marginally.
        assert plain.summary(ty)["empirical_coverage"] >= 0.86
        assert adapt.summary(ty)["empirical_coverage"] >= 0.86
        # But the adaptive one should be far more even across difficulty.
        r_plain = conformal.coverage_by_bin(plain, ty, np.abs(tx),
                                          n_bins=5)["coverage_range"]
        r_adapt = conformal.coverage_by_bin(adapt, ty, np.abs(tx),
                                          n_bins=5)["coverage_range"]
        assert r_adapt < r_plain

    def test_non_positive_scale_raises(self):
        with pytest.raises(ValueError):
            conformal.normalised_conformal(
                np.zeros(10), np.zeros(10), np.zeros(10),
                np.zeros(5), np.ones(5))


class TestCQR:
    def test_covers_and_can_tighten_an_overwide_model(self):
        rng = np.random.default_rng(4)
        n = 1500
        y_cal = rng.standard_normal(n)
        # Quantile model that is deliberately far too wide.
        lo_cal, hi_cal = np.full(n, -5.0), np.full(n, 5.0)
        iv = conformal.cqr(lo_cal, hi_cal, y_cal,
                           np.full(n, -5.0), np.full(n, 5.0), alpha=0.1)
        assert iv.detail["tightened"] is True
        assert iv.width[0] < 10.0
        y_test = rng.standard_normal(n)
        assert iv.summary(y_test)["empirical_coverage"] >= 0.86

    def test_widens_an_overconfident_model(self):
        rng = np.random.default_rng(5)
        n = 1500
        y_cal = rng.standard_normal(n)
        lo_cal, hi_cal = np.full(n, -0.05), np.full(n, 0.05)
        iv = conformal.cqr(lo_cal, hi_cal, y_cal,
                           np.full(n, -0.05), np.full(n, 0.05), alpha=0.1)
        assert iv.detail["tightened"] is False
        assert iv.summary(rng.standard_normal(n))["empirical_coverage"] >= 0.86


class TestWeightedConformal:
    def test_recovers_coverage_under_covariate_shift(self):
        """Unweighted conformal under-covers when the test covariates move into
        a higher-noise region; weighting by the likelihood ratio repairs it."""
        rng = np.random.default_rng(6)

        def sample(n, loc):
            x = rng.normal(loc, 1.0, n)
            y = 2.0 * x + (0.3 + 0.8 * np.abs(x)) * rng.standard_normal(n)
            return x, y

        xc, yc = sample(2000, 0.0)          # calibration at loc 0
        xs, ys = sample(3000, 2.0)          # test shifted to loc 2
        pred = lambda x: 2.0 * x            # noqa: E731

        plain = conformal.split_conformal(pred(xc), yc, pred(xs), alpha=0.1)
        plain_cov = plain.summary(ys)["empirical_coverage"]

        # Exact likelihood ratio N(2,1)/N(0,1) evaluated at the calibration x.
        w = np.exp(2.0 * xc - 2.0)
        wc = conformal.WeightedConformal(alpha=0.1)
        weighted = wc.interval(pred(xc), yc, w, pred(xs), test_weight=1.0)
        w_cov = weighted.summary(ys)["empirical_coverage"]

        assert plain_cov < 0.9          # shift breaks the unweighted method
        assert w_cov > plain_cov        # reweighting recovers coverage
        assert weighted.detail["effective_sample_size"] < len(w)

    def test_rejects_negative_weights(self):
        wc = conformal.WeightedConformal()
        with pytest.raises(ValueError):
            wc.interval(np.zeros(5), np.zeros(5), -np.ones(5), np.zeros(2))

    def test_weight_count_must_match(self):
        wc = conformal.WeightedConformal()
        with pytest.raises(ValueError):
            wc.interval(np.zeros(5), np.zeros(5), np.ones(3), np.zeros(2))


class TestAdaptiveConformal:
    def test_tracks_long_run_coverage_through_a_regime_change(self):
        """Exchangeability is violated on purpose: the noise scale jumps
        mid-series. ACI should still land near the target overall."""
        rng = np.random.default_rng(7)
        n = 4000
        scale = np.where(np.arange(n) < n // 2, 0.5, 4.0)
        y = scale * rng.standard_normal(n)
        pred = np.zeros(n)
        aci = conformal.AdaptiveConformal(alpha=0.1, gamma=0.02, window=400)
        iv = aci.run(pred, y)
        # Ignore the warm-up, where there is no history to calibrate on.
        cov = float(np.mean(iv.covers(y)[500:]))
        assert 0.85 <= cov <= 0.95
        assert iv.detail["guarantee"].startswith("long-run")

    def test_alpha_path_reacts_to_misses(self):
        rng = np.random.default_rng(8)
        n = 2000
        y = np.concatenate([rng.standard_normal(n // 2),
                            10.0 * rng.standard_normal(n // 2)])
        iv = conformal.AdaptiveConformal(alpha=0.1, gamma=0.05).run(
            np.zeros(n), y)
        path = np.asarray(iv.detail["alpha_path"])
        # After the break the level should be pushed down (wider intervals).
        assert path[n // 2 + 200:].mean() < path[200:n // 2].mean()

    def test_split_conformal_fails_where_aci_holds(self):
        """The comparison a post should show: the finite-sample method loses its
        guarantee entirely once the data stops being exchangeable."""
        rng = np.random.default_rng(9)
        n = 4000
        y = np.concatenate([0.5 * rng.standard_normal(n // 2),
                            4.0 * rng.standard_normal(n // 2)])
        pred = np.zeros(n)
        half = n // 2
        split = conformal.split_conformal(pred[:half], y[:half], pred[half:],
                                          alpha=0.1)
        assert split.summary(y[half:])["empirical_coverage"] < 0.6

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            conformal.AdaptiveConformal().run(np.zeros(10), np.zeros(9))


class TestDiagnostics:
    def test_coverage_by_group_finds_an_undercovered_subgroup(self):
        lower = np.concatenate([np.full(500, -1.0), np.full(500, -1.0)])
        upper = -lower
        y = np.concatenate([np.zeros(500), np.full(500, 5.0)])
        iv = conformal.Interval(lower, upper, 0.1, "manual")
        groups = np.array(["easy"] * 500 + ["hard"] * 500)
        out = conformal.coverage_by_group(iv, y, groups)
        assert out["per_group"]["easy"]["coverage"] == 1.0
        assert out["per_group"]["hard"]["coverage"] == 0.0
        assert out["worst_group_coverage"] == 0.0
        assert out["coverage_range"] == 1.0

    def test_rolling_coverage_has_the_expected_length(self):
        iv = conformal.Interval(np.full(300, -1.0), np.full(300, 1.0), 0.1, "m")
        y = np.zeros(300)
        assert len(conformal.rolling_coverage(iv, y, window=50)) == 251


class TestCausal:
    def test_naive_ols_is_biased_by_exactly_the_predicted_amount(self):
        scm = causal.ConfoundedSCM()
        d = scm.sample(40000, seed=0)
        est = causal.ols(d.frame, "Y", ["X"])["X"]
        assert est == pytest.approx(d.truth["naive_ols_on_X_alone"], abs=0.03)
        assert abs(est - scm.total_effect) > 0.1        # the bias is material

    def test_adjusting_for_the_confounder_recovers_the_total_effect(self):
        scm = causal.ConfoundedSCM()
        d = scm.sample(40000, seed=1)
        est = causal.ols(d.frame, "Y", ["X", "Z"])["X"]
        assert est == pytest.approx(scm.total_effect, abs=0.03)

    def test_adjusting_for_the_mediator_silently_gives_the_direct_effect(self):
        """The trap: the model looks better specified and reports a different
        quantity than the one being described."""
        scm = causal.ConfoundedSCM()
        d = scm.sample(40000, seed=2)
        est = causal.ols(d.frame, "Y", ["X", "Z", "M"])["X"]
        assert est == pytest.approx(scm.direct_effect, abs=0.03)
        assert abs(est - scm.total_effect) > 0.5

    def test_conditioning_on_a_collider_invents_an_effect(self):
        d = causal.collider(40000, b_xy=0.0, seed=3)
        naive = causal.ols(d.frame, "Y", ["X"])["X"]
        adjusted = causal.ols(d.frame, "Y", ["X", "C"])["X"]
        assert naive == pytest.approx(0.0, abs=0.02)
        assert adjusted < -0.3            # spurious, and negative as predicted

    def test_truth_dict_is_self_consistent(self):
        scm = causal.ConfoundedSCM(b_x=0.5, b_m=1.0, a_xm=0.8)
        assert scm.total_effect == pytest.approx(1.3)
        assert scm.direct_effect == pytest.approx(0.5)


class TestMultiplicity:
    """The winner's curse, checked against simulation rather than asserted.

    These are the load-bearing numbers in exp004: the post's claim is that the best
    of N chance-level models scores what the formula says it will, so the formula
    itself has to be right to more precision than the effect being demonstrated.
    """

    def test_expected_max_matches_monte_carlo(self):
        """Exact E[max] against 20,000 simulated searches of 200 models each."""
        n_obs, n_models, reps = 300, 200, 20_000
        rng = np.random.default_rng(3)
        draws = rng.binomial(n_obs, 0.5, size=(reps, n_models)).max(axis=1) / n_obs
        sim, se = draws.mean(), draws.std() / np.sqrt(reps)
        exact = multiplicity.expected_max_accuracy(n_models, n_obs)
        assert abs(exact - sim) < 4 * se, f"exact {exact:.5f} vs sim {sim:.5f}±{se:.5f}"

    def test_expected_max_of_one_model_is_the_mean(self):
        assert multiplicity.expected_max_accuracy(1, 900) == pytest.approx(0.5, abs=1e-12)

    def test_expected_max_is_monotone_in_the_budget(self):
        values = [multiplicity.expected_max_accuracy(n, 900)
                  for n in (1, 2, 10, 100, 2000, 10 ** 6)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_trials_to_reach_inverts_expected_max(self):
        """The table and the curve must be one function read two ways."""
        for target in (0.53, 0.55, 0.57):
            n = multiplicity.trials_to_reach(target, 900)
            assert multiplicity.expected_max_accuracy(n, 900) >= target
            assert multiplicity.expected_max_accuracy(n - 1, 900) < target

    def test_trials_to_reach_survives_the_far_tail(self):
        """60% on 900 observations needs ~6.6e8 tries; the log-space arithmetic has
        to get there without overflowing or silently returning the cap."""
        n = multiplicity.trials_to_reach(0.60, 900)
        assert 5e8 < n < 8e8
        assert multiplicity.trials_to_reach(0.75, 900, cap=10 ** 9) is None

    def test_normal_approximation_understates_the_expected_maximum(self):
        """Why the exact version exists — and it is not only a tail problem.

        The usual shortcut takes the (1 - 1/(N+1)) quantile of a normal as the
        expected maximum. That quantile sits *below* the expected maximum at every
        budget, by a third of a percentage point even at twenty models, so the
        shortcut is biased in the flattering direction: it makes a lucky winner look
        less explainable by luck than it is.
        """
        for n_models in (2, 20, 200, 2000):
            exact = multiplicity.expected_max_accuracy(n_models, 900)
            approx = multiplicity.normal_expected_max_accuracy(n_models, 900)
            assert approx < exact, n_models

    def test_normal_tail_inflates_the_trials_table(self):
        """The far tail is where the error becomes a wrong headline number: the
        normal version demands ~1.5x more models to reach 60% than the truth."""
        exact = multiplicity.trials_to_reach(0.60, 900)
        sd = np.sqrt(0.25 / 900)
        normal = 1.0 / stats.norm.sf((0.60 - 0.5) / sd)
        assert normal / exact > 1.4

    def test_significance_threshold_attains_its_level(self):
        acc, attained = multiplicity.significance_threshold(900, 0.05)
        hits = round(acc * 900)
        assert attained <= 0.05 < stats.binom.sf(hits - 2, 900, 0.5)
        assert attained == pytest.approx(0.0445, abs=5e-4)

    def test_false_positive_count_follows_the_attained_level(self):
        """The count of "significant" models in a null search tracks the attained
        level, not the nominal 5% — which is why the post quotes ~89 and not 100."""
        n_obs, n_models = 900, 4000
        acc, attained = multiplicity.significance_threshold(n_obs, 0.05)
        rng = np.random.default_rng(7)
        scores = rng.binomial(n_obs, 0.5, size=n_models) / n_obs
        observed = int((scores >= acc).sum())
        expected = attained * n_models
        se = np.sqrt(n_models * attained * (1 - attained))
        assert abs(observed - expected) < 4 * se
