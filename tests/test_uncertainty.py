"""Conformal coverage and calibration estimators, against what they promise.

The guarantees here are theorems, so they are pinned tightly: the marginal
coverage, the exactness of the finite-sample level, the invariance of the
argmax under temperature. The gaps between guarantee and delivery are
properties of one model and are pinned as inequalities.
"""
import math

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
