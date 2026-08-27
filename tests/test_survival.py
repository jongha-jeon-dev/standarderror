"""Tests for reading a survival result out of what a trial report chose to print.

The test that matters is `TestAgainstAnAnswerThatIsNotItself`. Every other check in
here compares one part of `survival.py` with another part of `survival.py`, which
catches typos and nothing else. Those two compare the module's variance formula with
the closed-form maximum-likelihood variance for exponential survival, `1/D1 + 1/D0`,
computed from the simulated data and never routed through the formula under test.

`TestCalibratedOnPublishedTrials` is the second one to keep. It holds seven real
adjuvant melanoma reports and asserts the event count recovered from each printed
confidence interval lands near the count the paper published. If a future change to
`variance_factor` breaks that, the module is wrong in the only way that matters, and
no amount of internal consistency will show it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from standarderror.uq import survival as sv


# name, allocation ratio, HR, CI low, CI high, CI level, events treated, events control
PUBLISHED = [
    ("KEYNOTE-054", 1.0, 0.57, 0.43, 0.74, 0.984, 135, 216),
    ("KEYNOTE-716", 1.0, 0.65, 0.46, 0.92, 0.95, 54, 82),
    ("CheckMate-238", 1.0, 0.65, 0.51, 0.83, 0.9756, 154, 206),
    ("CheckMate-76K", 2.0, 0.42, 0.30, 0.59, 0.95, 66, 69),
    ("COMBI-AD", 1.0, 0.47, 0.39, 0.58, 0.95, 166, 248),
    ("EORTC 18071", 1.0, 0.75, 0.64, 0.90, 0.95, 234, 294),
    ("KEYNOTE-942", 2.0, 0.561, 0.309, 1.017, 0.95, 24, 20),
]


class TestGeometry:
    def test_z_for_the_usual_level(self):
        assert sv.z_for_level(0.95) == pytest.approx(1.959964, abs=1e-5)

    def test_a_group_sequential_level_is_a_bigger_multiplier(self):
        """KEYNOTE-054's 98.4% interval, which is the trap this guards."""
        assert sv.z_for_level(0.984) == pytest.approx(2.408916, abs=1e-5)
        assert sv.z_for_level(0.984) > sv.z_for_level(0.95)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_level_must_be_a_probability(self, bad):
        with pytest.raises(ValueError):
            sv.z_for_level(bad)

    def test_allocation_fraction(self):
        assert sv.allocation_fraction(1.0) == pytest.approx(0.5)
        assert sv.allocation_fraction(2.0) == pytest.approx(2 / 3)

    def test_no_effect_means_events_split_like_patients(self):
        for ratio in (1.0, 2.0, 3.0):
            assert sv.event_fraction(1.0, ratio) == pytest.approx(
                sv.allocation_fraction(ratio))

    def test_a_working_treatment_takes_less_than_its_share_of_events(self):
        assert sv.event_fraction(0.5, 2.0) < sv.allocation_fraction(2.0)
        assert sv.event_fraction(1.5, 2.0) > sv.allocation_fraction(2.0)

    def test_variance_factor_precedence(self):
        """Observed split wins over predicted, which wins over allocation."""
        obs = sv.variance_factor(ratio=2.0, hazard_ratio=0.5, event_split=0.5)
        assert obs == pytest.approx(0.25)
        pred = sv.variance_factor(ratio=2.0, hazard_ratio=0.5)
        alloc = sv.variance_factor(ratio=2.0)
        assert alloc == pytest.approx(2 / 9)
        assert pred != pytest.approx(alloc)

    def test_the_factor_peaks_at_a_balanced_split(self):
        assert sv.variance_factor(event_split=0.5) == pytest.approx(0.25)
        for p in (0.2, 0.35, 0.65, 0.8):
            assert sv.variance_factor(event_split=p) < 0.25

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.4])
    def test_event_split_must_be_a_fraction(self, bad):
        with pytest.raises(ValueError):
            sv.variance_factor(event_split=bad)

    def test_negative_allocation_and_hazard_are_refused(self):
        with pytest.raises(ValueError):
            sv.allocation_fraction(0.0)
        with pytest.raises(ValueError):
            sv.event_fraction(-1.0)
        with pytest.raises(ValueError):
            sv.log_hr_se(0.0)


class TestAgainstAnAnswerThatIsNotItself:
    """`1/D1 + 1/D0` is the exact exponential MLE variance. Nothing else here is."""

    @staticmethod
    @pytest.fixture(scope="class")
    def null_fit():
        rng = np.random.default_rng(4)
        t, e, a = sv.simulate_arms(n_treated=6000, n_control=6000,
                                   hazard_ratio=1.0, control_rate=0.4,
                                   follow_up=1e6, rng=rng)
        return sv.cox_binary(t, e, a)

    def test_under_the_null_the_information_is_the_closed_form(self, null_fit):
        d1 = null_fit.events_treated
        d0 = null_fit.events - d1
        exact = 1.0 / (1.0 / d1 + 1.0 / d0)
        assert null_fit.information == pytest.approx(exact, rel=0.02)
        assert null_fit.schoenfeld_information == pytest.approx(exact, rel=1e-9)

    def test_the_algebraic_identity_itself(self):
        """`1/D1 + 1/D0 == 1/(D p_e q_e)` for arbitrary counts, to machine precision."""
        for d1, d0 in ((7, 11), (135, 216), (66, 69), (1, 999)):
            d = d1 + d0
            p = d1 / d
            assert 1.0 / d1 + 1.0 / d0 == pytest.approx(1.0 / (d * p * (1 - p)))

    def test_the_partial_likelihood_loses_information_without_censoring(self):
        """Not a defect — a documented limit, pinned so it cannot drift silently.

        With no censoring and a real effect the treated arm survives, so late risk
        sets are nearly all treated and carry little comparison. The Cox information
        then falls below the parametric one by about a tenth.
        """
        rng = np.random.default_rng(5)
        t, e, a = sv.simulate_arms(n_treated=4000, n_control=4000,
                                   hazard_ratio=0.6, control_rate=0.4,
                                   follow_up=1e6, rng=rng)
        fit = sv.cox_binary(t, e, a)
        ratio = fit.information / fit.schoenfeld_information
        assert 0.85 < ratio < 0.95

    def test_and_agrees_closely_once_most_patients_are_censored(self):
        """The regime every trial report in this module's calibration set lives in."""
        rng = np.random.default_rng(6)
        t, e, a = sv.simulate_arms(n_treated=3000, n_control=1500,
                                   hazard_ratio=0.6, control_rate=0.12,
                                   follow_up=1.6, rng=rng)
        fit = sv.cox_binary(t, e, a)
        assert e.mean() < 0.25
        assert fit.information == pytest.approx(fit.schoenfeld_information,
                                                rel=0.02)

    def test_the_standard_error_matches_the_sampling_spread(self):
        """The formula's SE against the actual scatter of repeated estimates."""
        rng = np.random.default_rng(19)
        fits = [sv.cox_binary(*sv.simulate_arms(
            n_treated=758, n_control=379, hazard_ratio=0.65,
            control_rate=0.12, follow_up=1.6, rng=rng)) for _ in range(300)]
        empirical = float(np.std([f.log_hr for f in fits], ddof=1))
        formula = float(np.mean([1.0 / np.sqrt(f.schoenfeld_information)
                                for f in fits]))
        assert formula == pytest.approx(empirical, rel=0.08)


class TestCoxFit:
    def test_it_recovers_a_known_hazard_ratio(self):
        rng = np.random.default_rng(21)
        t, e, a = sv.simulate_arms(n_treated=8000, n_control=8000,
                                   hazard_ratio=0.55, control_rate=0.15,
                                   follow_up=2.0, rng=rng)
        fit = sv.cox_binary(t, e, a)
        assert np.exp(fit.log_hr) == pytest.approx(0.55, rel=0.06)

    def test_the_score_test_is_the_log_rank_and_is_large_here(self):
        rng = np.random.default_rng(22)
        t, e, a = sv.simulate_arms(n_treated=800, n_control=400,
                                   hazard_ratio=0.55, control_rate=0.15,
                                   follow_up=2.0, rng=rng)
        fit = sv.cox_binary(t, e, a)
        assert fit.logrank_z < -3.0            # fewer events than expected treated
        assert fit.log_hr < 0.0

    def test_a_null_comparison_is_not_significant_on_average(self):
        rng = np.random.default_rng(23)
        zs = [sv.cox_binary(*sv.simulate_arms(
            n_treated=400, n_control=400, hazard_ratio=1.0,
            control_rate=0.2, follow_up=2.0, rng=rng)).logrank_z
            for _ in range(200)]
        assert abs(float(np.mean(zs))) < 0.35
        assert float(np.std(zs, ddof=1)) == pytest.approx(1.0, abs=0.12)

    def test_ties_do_not_break_it(self):
        """Everything rounded to whole months, which is how registries report."""
        rng = np.random.default_rng(24)
        t, e, a = sv.simulate_arms(n_treated=600, n_control=600,
                                   hazard_ratio=0.6, control_rate=0.2,
                                   follow_up=3.0, rng=rng)
        fit = sv.cox_binary(np.ceil(t * 12) / 12, e, a)
        assert np.exp(fit.log_hr) == pytest.approx(0.6, rel=0.25)
        assert fit.information > 0

    def test_it_refuses_malformed_input(self):
        with pytest.raises(ValueError):
            sv.cox_binary([1.0, 2.0], [True, False], [0, 2])
        with pytest.raises(ValueError):
            sv.cox_binary([1.0, 2.0], [True], [0, 1])
        with pytest.raises(ValueError):
            sv.cox_binary([1.0, 2.0], [False, False], [0, 1])


class TestRoundTrips:
    @pytest.mark.parametrize("ratio", [1.0, 2.0])
    @pytest.mark.parametrize("events", [50, 137, 400])
    @pytest.mark.parametrize("hr", [0.42, 0.75, 1.3])
    def test_interval_then_back_to_the_event_count(self, ratio, events, hr):
        lo, hi = sv.confidence_interval(hr, events, ratio=ratio)
        back = sv.implied_events(lo, hi, ratio=ratio, hazard_ratio=hr)
        assert back == pytest.approx(events, rel=1e-9)

    def test_the_span_is_the_interval_and_forgets_the_estimate(self):
        span = sv.ci_span(200, ratio=2.0, hazard_ratio=0.6)
        lo, hi = sv.confidence_interval(0.6, 200, ratio=2.0)
        assert hi / lo == pytest.approx(span, rel=1e-12)

    def test_p_value_and_interval_routes_agree(self):
        hr, events, ratio = 0.62, 260, 2.0
        se = sv.log_hr_se(events, ratio=ratio, hazard_ratio=hr)
        p = 2.0 * norm.sf(abs(np.log(hr)) / se)
        assert sv.implied_events_from_p(hr, p, ratio=ratio) == pytest.approx(
            events, rel=1e-8)

    def test_the_detectable_hazard_ratio_is_a_true_fixed_point(self):
        for events, z in ((150, 2.53), (400, 1.96), (80, 3.1)):
            h = sv.detectable_hr(events, z=z, ratio=2.0)
            se = sv.log_hr_se(events, ratio=2.0, hazard_ratio=h)
            assert -np.log(h) / se == pytest.approx(z, rel=1e-6)

    def test_more_events_permit_a_weaker_effect_to_be_significant(self):
        hs = [sv.detectable_hr(d, z=2.53, ratio=2.0)
              for d in (100, 200, 400, 800)]
        assert hs == sorted(hs)
        assert all(h < 1.0 for h in hs)


class TestRoundingIsTheBiggerErrorTerm:
    def test_the_bracket_contains_the_point_estimate(self):
        r = sv.events_from_rounded_ci(0.46, 0.92, decimals=2, hazard_ratio=0.65)
        assert r.low < r.point < r.high

    def test_more_printed_digits_narrow_it(self):
        widths = [sv.events_from_rounded_ci(0.46, 0.92, decimals=d,
                                           hazard_ratio=0.65).relative_width
                  for d in (1, 2, 3, 4)]
        assert widths == sorted(widths, reverse=True)

    def test_two_decimals_costs_several_percent(self):
        r = sv.events_from_rounded_ci(0.46, 0.92, decimals=2, hazard_ratio=0.65)
        assert 0.02 < r.relative_width < 0.15

    def test_which_is_larger_than_every_statistical_term(self):
        """The post's claim: typography beats statistics as the limiting error."""
        rounding = np.mean([
            sv.events_from_rounded_ci(lo, hi, decimals=2, level=lv, ratio=r,
                                      hazard_ratio=hr).relative_width
            for _, r, hr, lo, hi, lv, _, _ in PUBLISHED[:-1]])
        modelling = np.mean([
            abs(sv.implied_events(lo, hi, level=lv, ratio=r, hazard_ratio=hr)
                / (d1 + d0) - 1.0)
            for _, r, hr, lo, hi, lv, d1, d0 in PUBLISHED])
        assert rounding > 2.0 * modelling


class TestBoundaries:
    def test_the_first_look_boundary_collapses_to_a_known_form(self):
        for t in (0.3, 0.5, 0.75):
            assert sv.obrien_fleming_z(t) == pytest.approx(
                1.959964 / np.sqrt(t), rel=1e-6)

    def test_at_full_information_it_is_the_ordinary_critical_value(self):
        assert sv.obrien_fleming_z(1.0) == pytest.approx(1.959964, abs=1e-4)
        assert sv.pocock_z(1.0) == pytest.approx(1.959964, abs=1e-4)

    def test_an_early_look_is_a_higher_bar(self):
        zs = [sv.obrien_fleming_z(t) for t in (0.3, 0.5, 0.7, 1.0)]
        assert zs == sorted(zs, reverse=True)

    def test_pocock_spends_more_early_so_asks_less(self):
        for t in (0.3, 0.5, 0.8):
            assert sv.pocock_z(t) < sv.obrien_fleming_z(t)

    @pytest.mark.parametrize("bad", [0.0, -0.2, 1.3])
    def test_information_fraction_must_be_in_range(self, bad):
        with pytest.raises(ValueError):
            sv.obrien_fleming_z(bad)
        with pytest.raises(ValueError):
            sv.pocock_z(bad)

    def test_the_choice_of_spending_function_moves_the_bound_materially(self):
        of = sv.detectable_hr(200, z=sv.obrien_fleming_z(0.6), ratio=2.0)
        po = sv.detectable_hr(200, z=sv.pocock_z(0.6), ratio=2.0)
        assert po - of > 0.04


class TestRequiredEvents:
    def test_it_matches_schoenfeld_by_hand(self):
        # (1.959964 + 1.281552)^2 / ((2/9) * log(0.65)^2)
        expect = (1.959964 + 1.281552) ** 2 / ((2 / 9) * np.log(0.65) ** 2)
        assert sv.required_events(0.65, power=0.9, ratio=2.0) == pytest.approx(
            expect, rel=1e-6)      # the literals above are truncated, not the code
        assert sv.required_events(0.65, power=0.9, ratio=2.0) == pytest.approx(
            255, abs=1)

    def test_more_power_and_weaker_effects_both_cost_events(self):
        assert sv.required_events(0.65, power=0.9) > sv.required_events(
            0.65, power=0.8)
        assert sv.required_events(0.80) > sv.required_events(0.65)

    def test_balanced_allocation_is_the_cheapest(self):
        assert sv.required_events(0.65, ratio=1.0) < sv.required_events(
            0.65, ratio=2.0)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5])
    def test_a_null_effect_needs_no_finite_trial(self, bad):
        with pytest.raises(ValueError):
            sv.required_events(bad)


class TestTheOneBit:
    @staticmethod
    @pytest.fixture(scope="class")
    def upd():
        return sv.posterior_given_significance(
            prior_hr=0.561, prior_log_se=0.3039, events=200,
            z_boundary=sv.obrien_fleming_z(0.6), ratio=2.0)

    def test_significance_shifts_the_estimate_downward(self, upd):
        assert (upd["posterior_summary"]["median"]
                < upd["prior_summary"]["median"])

    def test_it_cuts_the_upper_end_far_more_than_the_lower(self, upd):
        pr, po = upd["prior_summary"], upd["posterior_summary"]
        assert pr["high"] - po["high"] > 4.0 * abs(pr["low"] - po["low"])

    def test_the_announcement_is_worth_well_under_a_bit(self, upd):
        assert 0.0 < upd["bits"] < 1.0
        assert 0.0 < upd["prior_predictive"] < 1.0
        assert upd["surprisal_bits"] == pytest.approx(
            -np.log2(upd["prior_predictive"]), rel=1e-9)

    def test_the_densities_integrate_to_one(self, upd):
        theta = np.log(upd["grid"])
        for key in ("prior", "posterior"):
            assert float(np.trapezoid(upd[key], theta)) == pytest.approx(
                1.0, rel=1e-6)

    def test_a_higher_boundary_implies_a_stronger_effect(self):
        med = [sv.posterior_given_significance(
            prior_hr=0.561, prior_log_se=0.3039, events=200, z_boundary=z,
            ratio=2.0)["posterior_summary"]["median"] for z in (2.0, 2.5, 3.2)]
        assert med == sorted(med, reverse=True)

    def test_a_grid_that_clips_the_prior_is_an_error_not_a_narrow_answer(self):
        """A truncated grid renormalises to something plausible and wrong."""
        with pytest.raises(ValueError, match="only"):
            sv.posterior_given_significance(
                prior_hr=0.561, prior_log_se=0.3039, events=200,
                z_boundary=2.53, ratio=2.0, grid=np.linspace(0.5, 0.7, 401))
        with pytest.raises(ValueError):
            sv.posterior_given_significance(
                prior_hr=0.561, prior_log_se=0.3039, events=200,
                z_boundary=2.53, ratio=2.0, grid=np.linspace(3.0, 4.0, 51))

    def test_a_degenerate_prior_is_refused(self):
        with pytest.raises(ValueError):
            sv.posterior_given_significance(
                prior_hr=0.5, prior_log_se=0.0, events=200, z_boundary=2.0)


class TestCalibratedOnPublishedTrials:
    """Seven real reports. The module's central claim, checked against the papers."""

    @staticmethod
    def _recovered(row, mode):
        _, ratio, hr, lo, hi, level, d1, d0 = row
        kw = {"level": level, "ratio": ratio}
        if mode == "predicted":
            kw["hazard_ratio"] = hr
        elif mode == "observed":
            kw["event_split"] = d1 / (d1 + d0)
        return sv.implied_events(lo, hi, **kw) / (d1 + d0) - 1.0

    def test_the_predicted_split_lands_within_ten_percent_on_every_trial(self):
        for row in PUBLISHED:
            err = self._recovered(row, "predicted")
            assert abs(err) < 0.10, f"{row[0]}: off by {100 * err:.1f}%"

    def test_and_within_three_percent_on_average(self):
        errs = [abs(self._recovered(r, "predicted")) for r in PUBLISHED]
        assert float(np.mean(errs)) < 0.03

    def test_the_observed_split_is_no_worse(self):
        pred = float(np.mean([abs(self._recovered(r, "predicted"))
                              for r in PUBLISHED]))
        obs = float(np.mean([abs(self._recovered(r, "observed"))
                             for r in PUBLISHED]))
        assert obs <= pred + 0.005

    def test_the_allocation_only_version_is_worse_and_biased_by_the_ratio(self):
        errs = [self._recovered(r, "allocation") for r in PUBLISHED]
        assert float(np.mean([abs(v) for v in errs])) > 0.05
        balanced = [v for r, v in zip(PUBLISHED, errs) if r[1] == 1.0]
        unbalanced = [v for r, v in zip(PUBLISHED, errs) if r[1] == 2.0]
        assert max(balanced) < 0.01           # never materially above the truth
        assert min(unbalanced) > 0.05         # and always well below it at 2:1

    def test_and_its_error_vanishes_as_the_effect_does(self):
        """The mechanism, checked where it is visible.

        The allocation fraction is only wrong because a working treatment takes less
        than its share of events, so the error must shrink to zero as the hazard
        ratio approaches 1. That is exact and is asserted analytically below.

        It is *not* asserted as a trend across the seven reports, and the first
        version of this test tried to: the rank correlation there is only 0.59,
        because each trial's recovery also carries the several-percent error the
        printed rounding creates, and at seven points that floor hides a five-percent
        mechanism. Testing a clean claim on noisy data was the mistake; the data
        gets the one check it can support, which is that the weakest effect in the
        set costs almost nothing.
        """
        gaps = []
        for hr in (0.99, 0.9, 0.75, 0.6, 0.45):
            alloc = sv.implied_events(0.5, 0.9, ratio=1.0)
            pred = sv.implied_events(0.5, 0.9, ratio=1.0, hazard_ratio=hr)
            gaps.append(abs(pred / alloc - 1.0))
        assert gaps == sorted(gaps)
        assert gaps[0] < 1e-4

        rows = [(abs(np.log(r[2])), abs(self._recovered(r, "allocation")))
                for r in PUBLISHED if r[1] == 1.0]
        assert min(rows, key=lambda t: t[0])[1] < 0.01

    def test_the_printed_rounding_brackets_the_truth_almost_everywhere(self):
        hits = 0
        for name, ratio, hr, lo, hi, level, d1, d0 in PUBLISHED:
            decimals = 3 if name == "KEYNOTE-942" else 2
            r = sv.events_from_rounded_ci(lo, hi, decimals=decimals, level=level,
                                          ratio=ratio, hazard_ratio=hr)
            hits += int(r.low <= d1 + d0 <= r.high)
        assert hits >= 6

    def test_reading_a_group_sequential_interval_as_ninety_five_percent_breaks_it(self):
        """KEYNOTE-054 prints 98.4%. Assume 95% and the event count falls by a third."""
        _, ratio, hr, lo, hi, level, d1, d0 = PUBLISHED[0]
        right = sv.implied_events(lo, hi, level=level, ratio=ratio, hazard_ratio=hr)
        wrong = sv.implied_events(lo, hi, level=0.95, ratio=ratio, hazard_ratio=hr)
        assert right == pytest.approx(d1 + d0, rel=0.06)
        assert wrong / (d1 + d0) < 0.70


class TestSimulator:
    def test_no_delay_is_plain_exponential(self):
        rng = np.random.default_rng(31)
        t, e, a = sv.simulate_arms(n_treated=1, n_control=40000,
                                   hazard_ratio=1.0, control_rate=0.5,
                                   follow_up=1e9, rng=rng)
        control = t[a == 0]
        assert float(control.mean()) == pytest.approx(2.0, rel=0.03)
        assert bool(e.all())

    def test_the_delay_lands_where_it_is_put(self):
        """Survival at the delay must not depend on the post-delay hazard ratio."""
        rng = np.random.default_rng(32)
        surv = []
        for hr in (1.0, 0.3):
            t, _, a = sv.simulate_arms(n_treated=40000, n_control=1,
                                       hazard_ratio=hr, control_rate=0.4,
                                       follow_up=1e9, rng=rng, delay=1.0)
            surv.append(float((t[a == 1] > 1.0).mean()))
        assert surv[0] == pytest.approx(np.exp(-0.4), abs=0.01)
        assert surv[1] == pytest.approx(np.exp(-0.4), abs=0.01)

    def test_a_delayed_effect_attenuates_the_estimated_hazard_ratio(self):
        rng = np.random.default_rng(33)

        def fit(delay):
            return float(np.exp(sv.cox_binary(*sv.simulate_arms(
                n_treated=6000, n_control=3000, hazard_ratio=0.6,
                control_rate=0.12, follow_up=1.6, rng=rng,
                delay=delay)).log_hr))

        assert fit(0.0) < fit(0.5) < fit(1.0) < 1.0

    def test_but_the_event_count_recovery_survives_it(self):
        """Non-proportional hazards move the estimate, not the count arithmetic."""
        rng = np.random.default_rng(34)
        for delay in (0.0, 0.5, 1.0):
            errs = []
            for _ in range(40):
                t, e, a = sv.simulate_arms(
                    n_treated=758, n_control=379, hazard_ratio=0.6,
                    control_rate=0.12, follow_up=1.6, rng=rng, delay=delay)
                f = sv.cox_binary(t, e, a)
                hr = float(np.exp(f.log_hr))
                lo, hi = sv.confidence_interval(hr, f.events, ratio=2.0)
                errs.append(sv.implied_events(lo, hi, ratio=2.0, hazard_ratio=hr)
                            / f.events - 1.0)
            assert abs(float(np.mean(errs))) < 0.02

    def test_dropout_removes_events(self):
        rng = np.random.default_rng(35)
        counts = []
        for rate in (0.0, 0.3):
            _, e, _ = sv.simulate_arms(n_treated=4000, n_control=2000,
                                       hazard_ratio=0.7, control_rate=0.2,
                                       follow_up=3.0, rng=rng,
                                       dropout_rate=rate)
            counts.append(int(e.sum()))
        assert counts[1] < 0.8 * counts[0]

    def test_it_refuses_a_degenerate_design(self):
        rng = np.random.default_rng(36)
        with pytest.raises(ValueError):
            sv.simulate_arms(n_treated=10, n_control=10, hazard_ratio=0.5,
                             control_rate=0.0, follow_up=1.0, rng=rng)
        with pytest.raises(ValueError):
            sv.simulate_arms(n_treated=10, n_control=10, hazard_ratio=0.5,
                             control_rate=0.2, follow_up=0.0, rng=rng)
