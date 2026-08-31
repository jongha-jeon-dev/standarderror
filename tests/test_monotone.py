"""Tests for the Price-of-Monotonicity machinery.

The split here mirrors the module. The arithmetic — the sign convention, the
constraint vector, the zero cases — is exact and gets exact assertions. The
learned quantities get known-answer tests instead: a correct constraint on a
monotone truth must not read as a cost, a sign-flipped constraint must read as a
large one, and a constraint on nothing must read as exactly zero. Those three are
the assertions that would actually have caught the errors worth catching.

A wrong sign convention in `pom` is the failure mode this file exists for. It
would flip every number in the post without breaking anything, so the two metric
directions are each pinned against a hand-built example where the answer is known
before the code runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.models.monotone import (
    SIGNS,
    _constraint,
    coverage_sweep,
    fit_pair,
    make_credit_like,
    paired_bootstrap_pom,
    pom,
    split_variance,
    violation_sweep,
)


class TestTheConstraintVector:
    def test_constraining_none_frees_everything(self):
        assert _constraint(0) == (0,) * 12

    def test_constraining_all_reproduces_the_true_signs(self):
        assert _constraint(12) == SIGNS

    def test_the_first_k_are_tied_and_the_rest_are_free(self):
        # The walk is from the strongest feature outwards, which is what makes
        # "coverage" comparable to a modeller working down a domain-knowledge list.
        c = _constraint(4)
        assert c[:4] == SIGNS[:4]
        assert set(c[4:]) == {0}

    @pytest.mark.parametrize("k", [-1, 13])
    def test_a_k_off_the_end_raises(self, k):
        with pytest.raises(ValueError):
            _constraint(k)


class TestSignConvention:
    """Positive PoM must mean 'the constraint made it worse' for both metrics.

    Both cases are built by hand so the expected number is known without running
    the code: three positives and three negatives, one model ordering them
    perfectly and the other getting one pair backwards.
    """

    Y = np.array([0, 0, 0, 1, 1, 1])
    GOOD = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    BAD = np.array([0.1, 0.2, 0.75, 0.3, 0.8, 0.9])  # one negative jumps a positive

    def test_identical_predictions_price_at_exactly_zero(self):
        assert pom(self.Y, self.GOOD, self.GOOD, "auc") == 0.0
        assert pom(self.Y, self.GOOD, self.GOOD, "brier") == 0.0

    def test_a_worse_constrained_model_is_positive_on_auc(self):
        # Nine ordered pairs. The good model wins all nine; the bad one loses the
        # single pair (0.75, 0.3), so 8/9 = 0.8889 and PoM = (1 - 0.8889)/1 = 11.1%.
        assert pom(self.Y, self.GOOD, self.BAD, "auc") == pytest.approx(11.111, abs=1e-3)

    def test_a_worse_constrained_model_is_positive_on_brier(self):
        # Brier is lower-is-better, so the subtraction runs the other way. If the
        # two branches were folded into one expression this is the sign that would
        # come out backwards.
        assert pom(self.Y, self.GOOD, self.BAD, "brier") > 0

    def test_a_better_constrained_model_is_negative_on_both(self):
        assert pom(self.Y, self.BAD, self.GOOD, "auc") < 0
        assert pom(self.Y, self.BAD, self.GOOD, "brier") < 0

    def test_an_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            pom(self.Y, self.GOOD, self.BAD, "logloss")


class TestTheGeneratedTruth:
    def test_at_no_violation_risk_rises_in_the_positive_features(self):
        # Every one of the twelve is monotone in the truth, which is the premise
        # the whole coverage sweep rests on. Checked empirically on the labels
        # rather than on the linear index, because the labels are what is fitted.
        X, y = make_credit_like(60_000, seed=3)
        for j, s in enumerate(SIGNS):
            lo = y[X[:, j] < -0.7].mean()
            hi = y[X[:, j] > 0.7].mean()
            assert (hi - lo) * s > 0.02, f"feature {j} is not monotone in sign {s}"

    def test_violation_makes_the_first_feature_u_shaped_and_leaves_the_rest(self):
        X, y = make_credit_like(60_000, violate=2.0, seed=3)
        mid = y[np.abs(X[:, 0]) < 0.3].mean()
        left = y[X[:, 0] < -1.2].mean()
        right = y[X[:, 0] > 1.2].mean()
        assert left > mid and right > mid          # both tails riskier than centre
        for j in range(1, 12):
            lo = y[X[:, j] < -0.7].mean()
            hi = y[X[:, j] > 0.7].mean()
            assert (hi - lo) * SIGNS[j] > 0.02

    def test_the_default_rate_barely_moves_with_violation(self):
        # If raising `violate` also moved class balance, the violation sweep would
        # be reading two changes at once and could not attribute the sign flip.
        rates = [make_credit_like(40_000, violate=v, seed=5)[1].mean()
                 for v in (0.0, 1.0, 2.0, 2.8)]
        assert max(rates) - min(rates) < 0.04, rates

    def test_a_negative_violation_raises(self):
        with pytest.raises(ValueError):
            make_credit_like(100, violate=-0.5)


class TestFitPairContract:
    def test_exactly_one_of_k_and_cst_is_required(self):
        X, y = make_credit_like(200, seed=1)
        with pytest.raises(ValueError):
            fit_pair(X, y)
        with pytest.raises(ValueError):
            fit_pair(X, y, k=3, cst=SIGNS)

    def test_constraining_nothing_returns_the_same_model_twice(self):
        # Same seed, same rows, same hyperparameters, and a constraint vector of
        # all zeros: the two fits must be bit-identical, which makes PoM exactly
        # zero. Any nonzero value at k=0 would mean the pairing is broken and every
        # other number in the module is measuring seed noise.
        X, y = make_credit_like(1200, seed=2)
        free, tied = fit_pair(X, y, k=0, seed=7)
        pf = free.predict_proba(X)[:, 1]
        pt = tied.predict_proba(X)[:, 1]
        assert np.allclose(pf, pt, atol=0, rtol=0)
        assert pom(y, pf, pt, "auc") == 0.0


class TestPairedBootstrap:
    def test_identical_models_give_an_interval_pinned_at_zero(self):
        y = np.repeat([0, 1], 200)
        p = np.concatenate([np.linspace(0.05, 0.5, 200), np.linspace(0.5, 0.95, 200)])
        ci = paired_bootstrap_pom(y, p, p, reps=100)
        assert ci["lo"] == 0.0 and ci["hi"] == 0.0
        assert not ci["significant"]

    def test_a_real_gap_is_flagged_and_the_interval_brackets_the_point(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 800)
        good = y + rng.normal(0, 0.55, 800)
        bad = y + rng.normal(0, 1.6, 800)
        ci = paired_bootstrap_pom(y, good, bad, reps=200)
        point = pom(y, good, bad, "auc")
        assert ci["lo"] < point < ci["hi"]
        assert ci["significant"] and ci["lo"] > 0

    def test_a_single_class_resample_is_redrawn_rather_than_scoring_nan(self):
        # Ten rows with one positive: a plain bootstrap hits an all-negative
        # resample within a few draws and roc_auc_score raises on it.
        y = np.array([0] * 9 + [1])
        p = np.linspace(0.1, 0.9, 10)
        ci = paired_bootstrap_pom(y, p, p[::-1], reps=60)
        assert np.isfinite(ci["mean"])


class TestKnownAnswersOnLearnedQuantities:
    """The three cases where the sign of the answer is known before fitting."""

    def test_correct_constraints_on_a_monotone_truth_do_not_read_as_a_cost(self):
        # This is the module's central claim and the paper's headline runs the
        # other way, so it gets asserted directly. With the truth monotone and the
        # training set small, removing the freedom to fit non-monotone noise is
        # regularisation: PoM must come out negative, not merely small.
        rows = coverage_sweep([700], [0, 12], repeats=4, seed=11)
        assert rows[(700, 0)]["mean"] == 0.0
        assert rows[(700, 12)]["mean"] < -0.2, rows[(700, 12)]

    def test_more_data_shrinks_the_effect_towards_nothing(self):
        # The constraint removes a fixed amount of flexibility; what changes is how
        # much data there is to use it. The ratio, not the count, is what matters.
        rows = coverage_sweep([700, 6000], [12], repeats=4, seed=11)
        small = abs(rows[(700, 12)]["mean"])
        large = abs(rows[(6000, 12)]["mean"])
        assert large < small / 2.0, (small, large)

    def test_flipped_signs_cost_far_more_than_correct_ones_ever_save(self):
        # Constraining every feature the wrong way on a monotone truth is the
        # worst case the design allows, and it must dominate the correct-sign
        # effect by an order of magnitude rather than merely exceed it.
        rows = violation_sweep([0.0], n_train=2500, repeats=3, seed=13)
        r = rows[0]
        assert r["wrong"] > 20.0
        assert r["correct"] < 0.0
        assert r["wrong"] > 10 * abs(r["correct"])


class TestSplitVariance:
    def test_it_reports_both_variances_and_their_ratio(self):
        # Small and cheap: this asserts the contract, not the finding. The finding
        # itself — that refitting moves PoM less than the bootstrap interval is
        # wide — is a result of the experiment, not something a test should pin.
        out = split_variance(900, k=12, splits=4, boot=60, seed=17)
        assert out["splits"] == 4
        assert out["across_split_sd"] > 0 and out["bootstrap_half_width"] > 0
        assert out["ratio"] == pytest.approx(
            out["across_split_sd"] / out["bootstrap_half_width"], rel=1e-9)
        assert out["min"] <= out["mean"] <= out["max"]
