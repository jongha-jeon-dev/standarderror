"""Conformal coverage and calibration estimators, against what they promise.

The guarantees here are theorems, so they are pinned tightly: the marginal
coverage, the exactness of the finite-sample level, the invariance of the
argmax under temperature. The gaps between guarantee and delivery are
properties of one model and are pinned as inequalities.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from standarderror.uncertainty import calibration as cb  # noqa: E402
from standarderror.uncertainty import coverage as cv  # noqa: E402


@pytest.fixture(scope="module")
def pred():
    return cv.predictions(count=8, size=8, seed=1)


class TestTheMarginalGuaranteeHolds:

    def test_coverage_reaches_the_finite_sample_level(self, pred):
        """Not `1 - alpha` but `ceil((n+1)(1-alpha))/(n+1)`, which is what the
        construction actually achieves and is slightly above it."""
        fit = cv.split_conformal(pred, alpha=0.1)
        assert fit["guarantee"] >= 0.9
        assert fit["coverage"] == pytest.approx(fit["guarantee"], abs=0.02)

    def test_it_holds_at_several_levels(self, pred):
        for alpha in (0.05, 0.1, 0.2):
            fit = cv.split_conformal(pred, alpha=alpha)
            assert fit["coverage"] == pytest.approx(1 - alpha, abs=0.03)

    def test_tighter_levels_need_bigger_sets(self, pred):
        small = cv.split_conformal(pred, alpha=0.2)
        big = cv.split_conformal(pred, alpha=0.05)
        assert big["mean_size"] > small["mean_size"]

    def test_no_set_is_empty_at_this_level(self, pred):
        """With the `1 - p[true]` score and alpha=0.1 the threshold admits the
        argmax always, so an empty set would mean a bug rather than a finding."""
        assert cv.split_conformal(pred, alpha=0.1)["empty"] == 0.0


class TestItSaysNothingAboutSubgroups:

    def test_conditional_coverage_is_uneven(self, pred):
        fit = cv.split_conformal(pred, alpha=0.1)
        bands = cv.conditional(pred, fit, bands=5)
        got = [b["coverage"] for b in bands]
        assert max(got) - min(got) > 0.05

    def test_and_it_fails_where_the_model_is_unsure(self, pred):
        """The direction matters more than the size: the shortfall lands on
        the cases you would escalate."""
        fit = cv.split_conformal(pred, alpha=0.1)
        bands = cv.conditional(pred, fit, bands=5)
        assert bands[0]["coverage"] < bands[-1]["coverage"]
        assert bands[0]["coverage"] < 0.9

    def test_sets_are_large_exactly_where_they_are_needed(self, pred):
        fit = cv.split_conformal(pred, alpha=0.1)
        bands = cv.conditional(pred, fit, bands=5)
        assert bands[0]["mean_size"] > 3 * bands[-1]["mean_size"]


class TestItIsTheScoreNotConformal:
    """The half of episode 1 that reverses it: the unevenness above is a
    property of the score, and a better score buys most of it back."""

    @pytest.fixture(scope="class")
    def pair(self, pred):
        lac = cv.split_conformal(pred, alpha=0.1)
        aps = cv.aps_conformal(pred, alpha=0.1)
        return lac, aps

    def test_both_hit_the_same_marginal_guarantee(self, pair):
        """Without this the comparison below is between coverage levels."""
        lac, aps = pair
        assert lac["coverage"] == pytest.approx(0.9, abs=0.03)
        assert aps["coverage"] == pytest.approx(0.9, abs=0.03)
        assert abs(lac["coverage"] - aps["coverage"]) < 0.02

    def test_aps_evens_out_conditional_coverage(self, pred, pair):
        lac, aps = pair
        r_lac = cv.conditional_range(cv.conditional(pred, lac, bands=5))
        r_aps = cv.conditional_range(cv.conditional(pred, aps, bands=5))
        assert r_aps < r_lac / 3

    def test_and_charges_for_it_in_set_size(self, pair):
        lac, aps = pair
        assert aps["mean_size"] > lac["mean_size"]
        assert aps["mean_size"] < 2 * lac["mean_size"]

    def test_and_in_occasionally_empty_sets(self, pair):
        """The price nobody mentions: a randomised score can return nothing."""
        lac, aps = pair
        assert lac["empty"] == 0.0
        assert aps["empty"] > 0.0

    def test_the_escalation_queue_is_the_failing_part(self, pred, pair):
        lac, _ = pair
        for row in cv.escalation(pred, lac):
            assert row["escalated_coverage"] < row["kept_coverage"]
            assert row["escalated_coverage"] < 0.9


class TestTheExchangeabilityAssumption:

    def test_the_realised_coverage_is_more_variable_than_predicted(self, pred):
        """Rows drawn from overlapping contexts are not exchangeable at the
        row level, so the Beta spread understates the real one."""
        v = cv.split_variability(pred, draws=60)
        assert v["mean"] == pytest.approx(0.9, abs=0.01)
        assert v["sd_ratio"] > 1.2

    def test_splitting_by_sequence_makes_it_worse_not_better(self, pred):
        """The control. Row-wise splitting leaks between calibration and test,
        which flatters the spread; doing it properly reveals more."""
        row = cv.split_variability(pred, draws=60)
        seq = cv.grouped_split(pred, draws=60)
        assert seq["sd_ratio"] > row["sd_ratio"]
        assert seq["mean"] == pytest.approx(0.9, abs=0.015)


class TestCalibrationErrorIsBiased:

    def test_a_calibrated_model_scores_nonzero(self):
        p, y = cb.calibrated_draw(1000, 10, seed=0)
        assert cb.ece(*cb._conf_correct(p, y), bins=15) > 0.01

    def test_the_draw_really_is_calibrated(self):
        """The premise. Over many points, confidence must match accuracy in
        aggregate to far better than the binned estimator suggests."""
        p, y = cb.calibrated_draw(200_000, 10, seed=1)
        conf, correct = cb._conf_correct(p, y)
        assert abs(conf.mean() - correct.mean()) < 0.005

    def test_more_bins_report_more_error_on_the_same_model(self):
        p, y = cb.calibrated_draw(2000, 10, seed=2)
        c, a = cb._conf_correct(p, y)
        assert cb.ece(c, a, bins=50) > 1.5 * cb.ece(c, a, bins=5)

    def test_the_bias_scales_like_the_square_root(self):
        s = cb.bias_sweep(sizes=(500, 2000, 8000), bin_counts=(5, 20, 50),
                          repeats=6)
        assert s["slope"] == pytest.approx(0.5, abs=0.2)

    def test_equal_mass_bins_do_not_remove_it(self):
        """Worth pinning because it is the usual first suggestion."""
        p, y = cb.calibrated_draw(1000, 10, seed=3)
        c, a = cb._conf_correct(p, y)
        assert cb.ece(c, a, bins=15, adaptive=True) > 0.01


class TestTheFloorIsTheModelsOwn:
    """Episode 2. The null is a calibrated model with *these* confidences."""

    @pytest.fixture(scope="class")
    def model(self):
        pred = cv.predictions(count=8, size=16, seed=1)
        return pred["p"], pred["y"]

    def test_averaged_bias_matches_the_published_table(self):
        s = cb.bias_sweep(sizes=(200, 1000, 20000), bin_counts=(15,))
        assert s["grid"][(200, 15)] == pytest.approx(0.085, abs=0.008)
        assert s["grid"][(1000, 15)] == pytest.approx(0.035, abs=0.004)
        assert s["grid"][(20000, 15)] == pytest.approx(0.008, abs=0.002)

    def test_a_single_draw_is_not_the_average(self):
        """The syllabus once quoted 0.120 at n = 200; that was seed 0 alone."""
        one = cb.ece(*cb._conf_correct(*cb.calibrated_draw(200, 10, seed=0)),
                     bins=15)
        assert one == pytest.approx(0.120, abs=0.002)
        assert one > 1.3 * cb.bias_sweep(sizes=(200,), bin_counts=(15,))[
            "grid"][(200, 15)]

    def test_the_floor_depends_on_the_confidence_profile(self):
        def floor(classes):
            return np.mean([cb.ece(*cb._conf_correct(*cb.calibrated_draw(
                1000, classes, seed=s)), bins=15) for s in range(10)])
        assert floor(65) < 0.6 * floor(10)

    def test_resampled_labels_make_the_model_calibrated(self, model):
        p, _ = model
        rng = np.random.default_rng(0)
        y = np.concatenate([cb.resample_labels(p, rng) for _ in range(4)])
        pp = np.concatenate([p] * 4)
        conf, correct = cb._conf_correct(pp, y)
        assert abs(conf.mean() - correct.mean()) < 0.01

    def test_debiasing_removes_the_floor_on_a_calibrated_model(self):
        plug, deb = [], []
        for s in range(20):
            c, a = cb._conf_correct(*cb.calibrated_draw(1000, 10, seed=s))
            plug.append(cb.l2_error(c, a, bins=50, debiased=False))
            deb.append(cb.l2_error(c, a, bins=50))
        assert np.mean(plug) > 0.06
        assert abs(np.mean(deb)) < 0.01

    def test_on_enough_data_the_debiased_estimate_ignores_bins(self):
        """On the full 24,576 rows the plug-in still moves with the bin count
        (0.032 to 0.037) and the debiased estimate does not (0.031 to 0.032).
        At 8,192 rows it is not yet settled at 50 bins, which is the same
        variance the next test pins."""
        pred = cv.predictions(count=24, size=16, seed=1)
        c, a = cb._conf_correct(pred["p"], pred["y"])
        plug = [cb.l2_error(c, a, bins=b, debiased=False) for b in (5, 15, 50)]
        deb = [cb.l2_error(c, a, bins=b) for b in (5, 15, 50)]
        assert max(plug) / min(plug) > 1.1
        assert max(deb) / min(deb) < 1.06

    def test_unbiased_is_not_precise(self, model):
        """The trade the episode ends on: on 1,000 points the plug-in is
        precise and wrong, the debiased estimate right on average and about
        as noisy as the quantity it estimates."""
        p, y = model
        st = cb.stability(p, y, n=1000, subsets=20)
        for b in (15, 50):
            assert st[b]["debiased_sd"] > 2 * st[b]["l2_sd"]
            assert st[b]["l2"] > st[b]["debiased"] + 0.01

    def test_temperature_leaves_accuracy_alone(self, model):
        p, y = model
        assert np.array_equal(p.argmax(1), cb.temperature(p, 1.15).argmax(1))


class TestTemperatureScaling:

    @pytest.fixture(scope="class")
    def table(self):
        return cb.temperature_table(count=4, size=8)

    def test_it_cannot_change_a_single_prediction(self, table):
        """Exactly, not approximately: a strictly increasing map applied to
        every logit cannot reorder them."""
        assert table["accuracy_identical"] is True
        assert len(table["accuracies"]) == 1

    def test_it_changes_the_calibration_a_lot(self, table):
        eces = [r["ece"] for r in table["rows"]]
        assert max(eces) > 5 * min(eces)

    def test_confidence_falls_monotonically_with_temperature(self, table):
        conf = [r["mean_confidence"] for r in
                sorted(table["rows"], key=lambda r: r["temperature"])]
        assert all(a > b for a, b in zip(conf, conf[1:]))

    def test_but_it_does_reorder_confidence_between_examples(self, table):
        """The per-example invariance is exact and the cross-example ordering
        is what abstention uses, so this is the one that has consequences."""
        rows = {r["temperature"]: r for r in cb.reordering(table)}
        assert rows[1.0]["tau"] == pytest.approx(1.0, abs=1e-9)
        far = max(rows, key=lambda t: abs(math.log(t)))
        assert rows[far]["tau"] < 0.95
        assert rows[far]["top_overlap"] < 0.95
