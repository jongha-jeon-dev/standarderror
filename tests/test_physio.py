"""Tests for the heat-balance model behind exp017.

The checks that matter are in `TestAgainstAnswersThatAreNotItself`:

* Weir's equation and Lusk's caloric-value table are two independently published
  routes from gas exchange to heat. They are not derived from each other here, so
  agreeing to within a percent is real evidence the conversions are right.
* The textbook 20.1 J per mL O2 at RER 0.82 is a third, cruder anchor.
* `required_conductance_rise` is a one-line shortcut for a three-term expression;
  the test evaluates the full expression and compares.
* The `Scholander` constants are checked against the numbers printed in the source
  tables — conductance per gram and the x-intercept — neither of which the class was
  built from.

`TestTheKnownDiscrepancy` pins a failure rather than a success: the published fits,
evaluated at 30 degC, do not reproduce the same papers' stated warm-versus-cold
ratio. That is a property of extrapolating a straight line past the thermoneutral
point, it is reported in the post, and it must not be silently tuned away.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantpost.physio import heat


class TestAgainstAnswersThatAreNotItself:
    def test_kcal_per_hour_in_watts(self):
        """4184 J per kcal over 3600 s, against the value the literature quotes."""
        assert heat.KCAL_PER_HOUR_TO_WATT == pytest.approx(1.1622, abs=1e-4)

    def test_the_textbook_joules_per_millilitre(self):
        """20.1 J/mL near RER 0.82 is the standard approximation."""
        kcal_per_litre = heat.lusk_caloric_value(0.82)
        joules_per_ml = kcal_per_litre * 4184.0 / 1000.0
        assert joules_per_ml == pytest.approx(20.1, abs=0.15)

    @pytest.mark.parametrize("rer,expect", [(0.70, 4.677), (0.80, 4.801),
                                           (0.85, 4.862), (1.00, 5.047)])
    def test_lusk_table(self, rer, expect):
        assert heat.lusk_caloric_value(rer) == pytest.approx(expect, abs=5e-4)

    def test_weir_and_lusk_agree(self):
        """Two published equations, no shared derivation, same answer."""
        for vo2_ml_min, rer in ((30.0, 0.75), (50.0, 0.85), (80.0, 0.95)):
            vco2 = rer * vo2_ml_min
            weir = heat.weir_kcal_per_day(vo2_ml_min, vco2)
            lusk = heat.lusk_caloric_value(rer) * (vo2_ml_min * 60.0 * 24.0) / 1000.0
            assert weir == pytest.approx(lusk, rel=0.015)

    def test_weir_matches_a_hand_computation(self):
        # 1.440 * (3.9*50 + 1.1*42.5) = 1.440 * 241.75
        assert heat.weir_kcal_per_day(50.0, 42.5) == pytest.approx(
            1.440 * 241.75, rel=1e-12)

    def test_conductance_per_gram_matches_the_source_table(self):
        """Jacobsen et al. report 0.61 and 0.54 mW/degC/g for chow and obese."""
        chow = heat.PUBLISHED_FITS["chow, light phase"]
        obese = heat.PUBLISHED_FITS["diet-induced obese, light phase"]
        assert chow.conductance_mw_per_c_per_g == pytest.approx(0.61, abs=0.02)
        assert obese.conductance_mw_per_c_per_g == pytest.approx(0.54, abs=0.02)

    def test_obesity_raises_absolute_conductance_and_lowers_it_per_gram(self):
        """The published finding — fat is not insulation."""
        chow = heat.PUBLISHED_FITS["chow, light phase"]
        obese = heat.PUBLISHED_FITS["diet-induced obese, light phase"]
        assert obese.conductance_w_per_c > chow.conductance_w_per_c
        assert obese.conductance_mw_per_c_per_g < chow.conductance_mw_per_c_per_g

    @pytest.mark.parametrize("key,published,slope,slope_sd,inter,inter_sd", [
        ("chow, light phase", 40.3, 0.014, 0.001, 0.56, 0.03),
        ("diet-induced obese, light phase", 40.6, 0.020, 0.002, 0.78, 0.04),
    ])
    def test_the_published_x_intercept_lies_inside_the_printed_precision(
            self, key, published, slope, slope_sd, inter, inter_sd):
        """A ratio of two rounded numbers cannot be pinned to a tenth of a degree.

        The papers print 40.3 and 40.6. Dividing the printed intercept by the printed
        slope gives 40.0 and 39.0 — off by up to 1.6 degC, not because either is wrong
        but because a quotient of two two-significant-figure numbers has a wide
        bracket. So the test asserts containment, not equality, and the post quotes
        the published value rather than recomputing it.
        """
        got = heat.PUBLISHED_FITS[key].x_intercept_c
        lo = (inter - inter_sd) / (slope + slope_sd)
        hi = (inter + inter_sd) / (slope - slope_sd)
        assert lo <= published <= hi
        assert lo <= got <= hi

    def test_the_extrapolation_is_not_a_thermometer(self):
        """Scholander says the x-intercept is core temperature. It is not, by ~5 degC.

        Worth pinning because it is the reason every core-temperature requirement in
        the post is quoted as a range rather than a number.
        """
        chow = heat.PUBLISHED_FITS["chow, light phase"]
        assert chow.x_intercept_c - heat.MOUSE_CORE_C > 4.0

    def test_a_live_mouse_is_far_better_insulated_than_a_dead_one(self):
        """About four times, which is the vasomotor headroom the post relies on."""
        chow = heat.PUBLISHED_FITS["chow, light phase"]
        ratio = heat.POST_MORTEM_CONDUCTANCE_W_PER_C / chow.conductance_w_per_c
        assert 5.0 < ratio < 10.0


class TestTheKnownDiscrepancy:
    """Pinned failure: the fits do not reproduce their own papers' warm/cold ratio."""

    def test_extrapolating_to_thirty_degrees_undershoots_the_published_ratio(self):
        """Papers state +101% (chow) and +104% (obese) for 22 versus 30 degC."""
        for key, published in (("chow, light phase", 1.01),
                               ("diet-induced obese, light phase", 1.04)):
            fit = heat.PUBLISHED_FITS[key]
            with pytest.warns(UserWarning, match="outside the fitted range"):
                ratio = float(fit.ee_kcal_h(22.0)) / float(fit.ee_kcal_h(30.0))
            implied = ratio - 1.0
            assert implied < published - 0.15, (
                f"{key}: line gives +{100 * implied:.0f}% against a published "
                f"+{100 * published:.0f}% — if this ever passes, re-check the post")

    def test_inside_the_fitted_range_the_line_is_trusted_silently(self):
        fit = heat.PUBLISHED_FITS["chow, light phase"]
        with warnings_as_errors():
            assert fit.ee_kcal_h(25.0) > 0


class _NoWarnings:
    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def warnings_as_errors():
    return _NoWarnings()


class TestTheInversions:
    def test_required_conductance_rise_against_the_full_expression(self):
        """The shortcut drops two terms. Check it against them.

        ratio = C'(Tb - Ta) / [C(Tb - Ta)], so at fixed Tb and Ta the bracket cancels
        and C'/C == ratio. Computed here the long way, for several gradients.
        """
        for ratio in (1.04, 1.18, 1.50):
            for t_body, t_ambient in ((35.6, 4.0), (35.6, 23.0), (40.3, 30.0)):
                c = 0.0163                      # any conductance at all
                gradient = t_body - t_ambient
                ee = c * gradient
                c_needed = (ratio * ee) / gradient
                assert c_needed / c - 1.0 == pytest.approx(
                    heat.required_conductance_rise(ratio), rel=1e-12)

    def test_the_conductance_answer_ignores_the_conductance(self):
        """Invariance to the constant the literature agrees on least.

        The published mouse conductances span 16 to 30 mW/degC. Solve the heat balance
        for the required conductance under each and the *fractional* answer is
        identical, which is why the post's headline number survives that disagreement.
        """
        ratio, gradient = 1.18, 35.6 - 23.0
        fractions = []
        for c in (0.0163, 0.0209, 0.0232, 0.0295):
            fractions.append(((ratio * c * gradient) / gradient) / c - 1.0)
        for f in fractions:
            assert f == pytest.approx(heat.required_conductance_rise(ratio),
                                      rel=1e-12)
        assert max(fractions) - min(fractions) < 1e-12

    def test_required_core_rise_closed_form(self):
        assert heat.required_core_rise(1.18, t_body_c=35.6,
                                       t_ambient_c=23.0) == pytest.approx(
            0.18 * 12.6, rel=1e-12)

    def test_the_same_effect_needs_less_warming_in_a_warm_room(self):
        rises = [heat.required_core_rise(1.18, t_body_c=35.6, t_ambient_c=t)
                 for t in (4.0, 23.0, 30.0)]
        assert rises == sorted(rises, reverse=True)
        assert rises[-1] < rises[0] / 2.0

    def test_eighteen_percent_needs_degrees_not_tenths(self):
        """The headline: what a flat thermometer has to be reconciled with."""
        for t_ambient, floor in ((23.0, 2.0), (30.0, 1.0)):
            rise = heat.required_core_rise(1.18, t_body_c=heat.MOUSE_CORE_C,
                                           t_ambient_c=t_ambient)
            assert rise > floor
            assert rise > heat.TELEMETRY_SD_C * 5

    def test_ambient_above_body_temperature_is_refused(self):
        with pytest.raises(ValueError, match="sensible heat loss reverses"):
            heat.required_core_rise(1.18, t_body_c=35.6, t_ambient_c=38.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_a_nonpositive_ratio_is_refused(self, bad):
        with pytest.raises(ValueError):
            heat.required_conductance_rise(bad)
        with pytest.raises(ValueError):
            heat.required_core_rise(bad, t_body_c=35.6, t_ambient_c=23.0)


class TestSubstitution:
    @pytest.fixture(scope="class")
    @staticmethod
    def fit():
        return heat.PUBLISHED_FITS["diet-induced obese, light phase"]

    def test_extra_heat_vanishes_in_the_cold(self, fit):
        cold = heat.substitution_prediction(fit, drug_fraction_of_basal=0.18,
                                            t_ambient_c=4.0)
        assert cold.fully_absorbed
        assert cold.measured_rise == pytest.approx(0.0, abs=1e-12)
        assert cold.thermoregulatory_fraction > 0.6

    def test_and_appears_in_full_at_the_thermoneutral_point(self, fit):
        warm = heat.substitution_prediction(
            fit, drug_fraction_of_basal=0.18,
            t_ambient_c=fit.thermoneutral_point_c)
        assert not warm.fully_absorbed
        assert warm.measured_rise == pytest.approx(0.18, rel=1e-9)
        assert warm.thermoregulatory_fraction == pytest.approx(0.0)

    def test_the_prediction_rises_with_ambient_temperature(self, fit):
        rises = [heat.substitution_prediction(
            fit, drug_fraction_of_basal=0.18, t_ambient_c=t).measured_rise
            for t in (4.0, 16.0, 23.0, 27.0, 30.0)]
        assert rises == sorted(rises)

    def test_which_is_the_opposite_of_the_usual_intuition(self, fit):
        """A heat-producing drug looks *smallest* where thermogenesis is largest."""
        cold = heat.substitution_prediction(fit, drug_fraction_of_basal=0.18,
                                            t_ambient_c=4.0).measured_rise
        warm = heat.substitution_prediction(
            fit, drug_fraction_of_basal=0.18,
            t_ambient_c=fit.thermoneutral_point_c).measured_rise
        assert warm > cold

    def test_room_temperature_absorbs_an_eighteen_percent_drug_entirely(self, fit):
        """The anomaly the post is about: 23 degC has room to hide the whole effect."""
        room = heat.substitution_prediction(fit, drug_fraction_of_basal=0.18,
                                            t_ambient_c=23.0)
        assert room.fully_absorbed
        assert room.measured_rise == pytest.approx(0.0, abs=1e-12)
        assert 0.3 < room.thermoregulatory_fraction < 0.6

    def test_a_drug_larger_than_the_thermogenesis_partly_shows_through(self, fit):
        big = heat.substitution_prediction(fit, drug_fraction_of_basal=2.0,
                                           t_ambient_c=23.0)
        assert not big.fully_absorbed
        assert big.measured_rise > 0.0

    def test_no_drug_means_no_change(self, fit):
        for t in (4.0, 23.0, 30.0):
            p = heat.substitution_prediction(fit, drug_fraction_of_basal=0.0,
                                             t_ambient_c=t)
            assert p.measured_ratio == pytest.approx(1.0)

    def test_a_negative_drug_effect_is_refused(self, fit):
        with pytest.raises(ValueError):
            heat.substitution_prediction(fit, drug_fraction_of_basal=-0.1,
                                         t_ambient_c=23.0)


class TestPower:
    def test_against_a_hand_computation(self):
        # (1.959964 + 0.841621) * 0.4 * sqrt(2/8)
        expect = (1.959964 + 0.841621) * 0.4 * np.sqrt(2.0 / 8.0)
        assert heat.detectable_difference(0.4, 8) == pytest.approx(expect, rel=1e-6)

    def test_a_probe_on_eight_mice_cannot_see_half_a_degree(self):
        assert heat.detectable_difference(heat.PROBE_SD_C, 8) > 0.5

    def test_telemetry_on_eight_mice_can(self):
        assert heat.detectable_difference(heat.TELEMETRY_SD_C, 8) < 0.3

    def test_but_both_would_have_seen_what_constant_conductance_demands(self):
        """The reported null is weak and still strong enough to force the conclusion."""
        needed = heat.required_core_rise(1.18, t_body_c=heat.MOUSE_CORE_C,
                                         t_ambient_c=23.0)
        assert needed > heat.detectable_difference(heat.PROBE_SD_C, 7)

    def test_more_animals_and_less_noise_both_help(self):
        assert heat.detectable_difference(0.4, 32) < heat.detectable_difference(0.4, 8)
        assert heat.detectable_difference(0.2, 8) < heat.detectable_difference(0.4, 8)

    @pytest.mark.parametrize("kw", [{"sd": 0.0, "n_per_group": 8},
                                    {"sd": 0.4, "n_per_group": 1}])
    def test_degenerate_designs_are_refused(self, kw):
        with pytest.raises(ValueError):
            heat.detectable_difference(**kw)


class TestNormalisation:
    def test_per_gram_invents_an_effect_when_the_groups_differ_in_mass(self):
        rng = np.random.default_rng(11)
        errs = [heat.simulate_calorimetry(
            n_per_group=8, true_effect=0.0, rng=rng).per_gram for _ in range(300)]
        mean = float(np.mean(errs))
        assert mean > 0.02, "a lighter treated group should look hypermetabolic"

    def test_while_ancova_does_not(self):
        rng = np.random.default_rng(12)
        errs = [heat.simulate_calorimetry(
            n_per_group=8, true_effect=0.0, rng=rng).ancova for _ in range(300)]
        assert abs(float(np.mean(errs))) < 0.02

    def test_and_ancova_recovers_a_real_effect_per_gram_overstates(self):
        rng = np.random.default_rng(13)
        runs = [heat.simulate_calorimetry(n_per_group=8, true_effect=0.10, rng=rng)
                for _ in range(300)]
        ancova = float(np.mean([r.ancova for r in runs]))
        per_gram = float(np.mean([r.per_gram for r in runs]))
        assert ancova == pytest.approx(0.10, abs=0.03)
        assert per_gram > ancova

    def test_with_equal_masses_per_gram_is_unbiased_too(self):
        """Confirms the artefact is the mass difference, not the ratio as such."""
        rng = np.random.default_rng(14)
        errs = [heat.simulate_calorimetry(
            n_per_group=8, true_effect=0.0, rng=rng, mass_control_g=43.0,
            mass_treated_g=43.0).per_gram for _ in range(300)]
        assert abs(float(np.mean(errs))) < 0.015

    def test_the_artefact_grows_with_the_mass_gap(self):
        rng = np.random.default_rng(15)
        out = []
        for treated in (45.0, 41.0, 37.0):
            errs = [heat.simulate_calorimetry(
                n_per_group=8, true_effect=0.0, rng=rng, mass_control_g=45.0,
                mass_treated_g=treated).per_gram for _ in range(300)]
            out.append(float(np.mean(errs)))
        assert out == sorted(out)

    def test_a_zero_intercept_removes_the_artefact_and_is_refused(self):
        rng = np.random.default_rng(16)
        with pytest.raises(ValueError, match="intercept"):
            heat.simulate_calorimetry(n_per_group=8, true_effect=0.0, rng=rng,
                                      intercept_w=0.0)

    def test_it_refuses_a_group_too_small_to_fit_a_covariate(self):
        rng = np.random.default_rng(17)
        with pytest.raises(ValueError):
            heat.simulate_calorimetry(n_per_group=2, true_effect=0.0, rng=rng)
