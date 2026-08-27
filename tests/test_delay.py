"""Tests for delayed-feedback capacity cycles.

Weighted toward the three mistakes that produced plausible-looking output with no
cycle in it: an unbalanced initial state, a unit conversion off by twelve, and a
demand rule that responded to price changes instead of price levels.
"""

from __future__ import annotations

import numpy as np
import pytest

from standarderror.dynamics import delay


class TestValidation:
    def test_channel_rejects_a_zero_delay(self):
        with pytest.raises(ValueError):
            delay.Channel("x", 0.5, 0)

    def test_model_needs_a_channel(self):
        with pytest.raises(ValueError):
            delay.CycleModel([])

    @pytest.mark.parametrize("kw", [{"kappa": 0.0}, {"theta": -1.0},
                                   {"decay": 1.0}, {"decay": -0.1}])
    def test_model_rejects_impossible_parameters(self, kw):
        with pytest.raises(ValueError):
            delay.CycleModel([delay.Channel("x", 0.5, 3)], **kw)

    def test_simulate_rejects_a_trade_ratio_below_one(self):
        with pytest.raises(ValueError):
            delay.simulate(n_months=24, trade_ratio=0.9)

    def test_simulate_rejects_reference_utilisation_outside_the_bounds(self):
        with pytest.raises(ValueError):
            delay.simulate(n_months=24, util_bounds=(0.7, 0.9), util_reference=0.95)


class TestCharacteristicPolynomial:
    def test_undamped_leading_terms_are_a_double_integrator(self):
        m = delay.CycleModel([delay.Channel("x", 0.5, 6)], decay=0.0)
        c = m.coefficients()
        # z^L - 2 z^(L-1) + z^(L-2) = z^(L-2) (z-1)^2
        assert c[0] == pytest.approx(1.0)
        assert c[1] == pytest.approx(-2.0)
        assert c[2] == pytest.approx(1.0)

    def test_damping_appears_in_exactly_two_coefficients(self):
        a = delay.CycleModel([delay.Channel("x", 0.5, 6)], decay=0.0).coefficients()
        b = delay.CycleModel([delay.Channel("x", 0.5, 6)], decay=0.1).coefficients()
        assert b[1] == pytest.approx(a[1] + 0.1)
        assert b[2] == pytest.approx(a[2] - 0.1)
        assert np.allclose(np.delete(a, [1, 2]), np.delete(b, [1, 2]))

    @pytest.mark.parametrize("gain", [0.01, 0.1, 0.5, 2.0])
    @pytest.mark.parametrize("lag", [2, 3, 6, 24, 60])
    def test_undamped_system_is_unstable_at_every_gain_and_delay(self, gain, lag):
        """The result the post is built on.

        A double integrator under delayed proportional feedback has no stable
        configuration, so an industry whose only price signal is an inventory level
        — with no anchor to cost — cannot be stabilised by choosing gains.
        """
        m = delay.CycleModel([delay.Channel("x", gain, lag)], decay=0.0)
        assert max(abs(delay.characteristic_roots(m))) > 1.0

    def test_damping_creates_a_stable_region(self):
        g = delay.critical_gain(6, decay=0.04)
        assert 0.0 < g < 10.0
        below = delay.CycleModel([delay.Channel("x", g * 0.9, 6)], decay=0.04)
        above = delay.CycleModel([delay.Channel("x", g * 1.1, 6)], decay=0.04)
        assert max(abs(delay.characteristic_roots(below))) < 1.0
        assert max(abs(delay.characteristic_roots(above))) > 1.0

    def test_a_channel_contributes_at_its_own_exponent(self):
        m = delay.CycleModel([delay.Channel("fast", 0.4, 3),
                              delay.Channel("slow", 0.2, 9)], decay=0.05)
        c = m.coefficients()
        assert len(c) == 10                          # degree 9
        assert c[3] == pytest.approx(m.c * 0.4)
        assert c[9] == pytest.approx(m.c * 0.2)


class TestModes:
    def test_dominant_mode_skips_real_roots(self):
        m = delay.CycleModel([delay.Channel("x", 0.5, 6)])
        d = delay.dominant_mode(m)
        assert np.isfinite(d["period_months"])
        assert abs(d["root"].imag) > 1e-9

    def test_modes_are_ordered_by_growth(self):
        ms = delay.modes(delay.CycleModel([delay.Channel("x", 0.5, 9)]))
        g = [m["growth_per_step"] for m in ms]
        assert g == sorted(g, reverse=True)

    def test_only_one_of_each_conjugate_pair_is_returned(self):
        m = delay.CycleModel([delay.Channel("x", 0.5, 8)])
        assert len(delay.modes(m)) < len(delay.characteristic_roots(m))

    def test_period_is_not_a_fixed_multiple_of_the_delay(self):
        """A cycle length cannot be read back as a lead time.

        At one delay, changing only the loop gain moves the period by more than a
        factor of three — so "the cycle is two years because the lag is two years"
        does not follow even when the lag is known.
        """
        periods = [delay.dominant_mode(
            delay.CycleModel([delay.Channel("x", g, 6)]))["period_months"]
            for g in (0.05, 0.15, 0.35, 0.55)]
        assert periods[0] / periods[-1] > 2.5
        assert all(b < a for a, b in zip(periods, periods[1:]))

    def test_longer_delay_lengthens_the_period_at_fixed_gain(self):
        periods = [delay.dominant_mode(
            delay.CycleModel([delay.Channel("x", 0.35, L)]))["period_months"]
            for L in (3, 6, 12, 24)]
        assert all(b > a for a, b in zip(periods, periods[1:]))


class TestRealisedPeriod:
    def test_recovers_a_known_sinusoid(self):
        t = np.arange(1200)
        x = np.exp(0.2 * np.sin(2 * np.pi * t / 40.0))
        assert delay.realised_period(x) == pytest.approx(40.0, rel=0.05)

    def test_a_trend_alone_has_no_finite_peak_it_confuses_for_a_cycle(self):
        # A pure exponential trend is removed by the quadratic detrend, so whatever
        # peak survives must be at a very long period rather than a spurious short
        # one. This guards the detrending, which is what makes the measure usable on
        # a growing series.
        x = np.exp(0.004 * np.arange(600))
        assert delay.realised_period(x) > 100.0


class TestSimulate:
    def test_starts_in_balance(self):
        """The bug that produced a plausible series with no cycle in it.

        An earlier version normalised capacity and demand to one and then divided
        supply by the product-mix term, leaving a permanent shortage of about 30%:
        inventory hit its floor on the first step, the price gap never changed sign,
        and the system crawled monotonically to a fixed point.
        """
        run = delay.simulate(n_months=36, burn_in=0, shock_sd=0.0)
        assert run.supply[0] == pytest.approx(run.demand[0], rel=1e-9)
        assert run.inventory[:12].min() > 1.0

    def test_inventory_units_are_weeks_per_month_of_imbalance(self):
        """One month of a 1% surplus is 52/12 weeks of inventory, not 52."""
        run = delay.simulate(n_months=6, burn_in=0, shock_sd=0.0, theta=1.0,
                             demand_growth_annual=0.0, slow_gain=0.0,
                             fast_gain=0.0, demand_elasticity=0.0,
                             mix_start=0.1, mix_target=0.1)
        step = run.inventory[1] - run.inventory[0]
        frac = (run.supply[1] - run.demand[1]) / run.demand[1]
        assert step == pytest.approx(frac * 52.0 / 12.0, rel=1e-6)

    def test_price_actually_oscillates(self):
        run = delay.simulate(n_months=600, seed=5)
        lp = np.log(run.price)
        crossings = int(np.sum(np.diff(np.sign(lp - lp.mean())) != 0))
        assert crossings > 20, "a cycle model that does not cross its own mean is not cycling"
        assert 15.0 < delay.realised_period(run.price) < 120.0

    def test_the_floor_guard_fires_when_the_floor_is_reachable(self):
        """The guard against returning a divergence as if it were a cycle.

        An earlier version let a glut drive price toward zero and inventory to 1e12
        weeks, and returned it without complaint. At sane parameters the floor is
        unreachable, so the guard is exercised by raising the floor to just under
        the price the model actually reaches.
        """
        run = delay.simulate(n_months=600, demand_elasticity=0.0, fast_gain=2.5,
                             seed=5)
        assert run.price.min() < 0.1        # a deep glut, but bounded
        with pytest.raises(RuntimeError, match="divergence rather than a cycle"):
            delay.simulate(n_months=600, demand_elasticity=0.0, fast_gain=2.5,
                           seed=5, price_floor=float(run.price.min()) * 1.5)

    def test_elasticity_is_not_what_bounds_the_model(self):
        """The obvious answer is wrong, and the docstring used to give it.

        Removing the price elasticity of demand widens the swing but does not
        unbound it: what pulls a glut back is trend demand growth outrunning a
        capacity stock that shrinks while the price is low.
        """
        elastic = delay.simulate(n_months=600, seed=5)
        inelastic = delay.simulate(n_months=600, demand_elasticity=0.0, seed=5)
        spread = lambda r: float(np.log(r.price).max() - np.log(r.price).min())
        assert spread(inelastic) > spread(elastic)
        assert np.isfinite(inelastic.price).all()

    def test_zero_trend_growth_pins_at_the_shortage_fixed_point(self):
        """With no trend growth, the mix shift is an unpayable supply drag.

        Adoption of the wafer-hungry product raises the wafers needed for the same
        bits, and with no capacity growth to fund it the model runs to a permanent
        shortage rather than a cycle. This is the configuration the floor guard is
        for, and it is worth an assertion because it says the mix transition has to
        be paid for out of capacity growth.
        """
        run = delay.simulate(n_months=300, demand_growth_annual=0.0, seed=5)
        # Permanently tight rather than cyclical: utilisation pinned at its ceiling,
        # inventory stuck well under target, price stuck well above par.
        assert run.utilisation[-1] == pytest.approx(1.0)
        assert run.inventory.max() < 0.75 * 6.0
        assert run.price.min() > 1.5
        crossings = int(np.sum(np.diff(np.sign(np.log(run.price))) != 0))
        assert crossings == 0, "a market that never returns to par is not cycling"

    def test_bounds_are_respected(self):
        run = delay.simulate(n_months=600, util_bounds=(0.7, 1.0), seed=5)
        assert run.utilisation.min() >= 0.7 - 1e-12
        assert run.utilisation.max() <= 1.0 + 1e-12
        assert run.inventory.min() >= 0.0
        assert 0.0 <= run.mix.min() and run.mix.max() <= 1.0

    def test_mix_adoption_is_monotone_toward_its_target(self):
        run = delay.simulate(n_months=600, mix_start=0.02, mix_target=0.23,
                             mix_years=12.0, seed=5)
        assert run.mix[0] < run.mix[-1] <= 0.23 + 1e-9
        assert np.all(np.diff(run.mix) >= -1e-12)

    def test_the_capacity_delay_barely_moves_the_period(self):
        """The post's central measurement, asserted so a refactor cannot lose it."""
        periods = [delay.realised_period(
            delay.simulate(n_months=900, slow_delay=S, seed=3).price)
            for S in (36, 60, 96)]
        assert max(periods) - min(periods) < 4.0

    def test_the_fast_delay_moves_the_period_a_lot(self):
        short = delay.realised_period(
            delay.simulate(n_months=900, fast_delay=1, seed=3).price)
        long = delay.realised_period(
            delay.simulate(n_months=900, fast_delay=12, seed=3).price)
        assert long - short > 15.0

    def test_shock_size_barely_changes_the_cycle(self):
        """It is a limit cycle, not a noise-driven one."""
        quiet = delay.simulate(n_months=900, shock_sd=0.012, seed=3)
        loud = delay.simulate(n_months=900, shock_sd=0.08, seed=3)
        assert abs(delay.realised_period(quiet.price)
                   - delay.realised_period(loud.price)) < 4.0

    def test_observable_and_full_state_shapes(self):
        run = delay.simulate(n_months=120, seed=1)
        assert run.observable.shape == (120, 2)
        assert run.full_state.shape == (120, 5)
        assert run.to_frame().shape == (120, 7)


class TestRegimeBreak:
    def test_histories_are_identical_before_the_break(self):
        base = delay.simulate(n_months=600, seed=11)
        broken = delay.simulate(n_months=600, regime=(300, 0.35), seed=11)
        assert np.allclose(base.price[:300], broken.price[:300])

    def test_a_higher_trend_raises_the_price_and_drains_inventory(self):
        base = delay.simulate(n_months=600, seed=11)
        broken = delay.simulate(n_months=600, regime=(300, 0.35), seed=11)
        after = slice(320, 420)
        assert broken.price[after].mean() > base.price[after].mean()
        assert broken.inventory[after].mean() < base.inventory[after].mean()

    def test_break_must_fall_inside_the_window(self):
        with pytest.raises(ValueError):
            delay.simulate(n_months=120, regime=(200, 0.35))


class TestLinearMatchesNonlinear:
    def test_the_linear_period_predicts_the_simulated_one_within_a_half(self):
        """The linear analysis has to earn its place.

        Saturation lengthens the cycle relative to its linear mode — by about 40% at
        the configuration the post uses — so the two do not agree exactly. What has
        to hold is that the linear model gets the scaling right, because otherwise
        the closed-form half of the post is decoration.
        """
        for L in (2, 3, 6, 12):
            lin = delay.dominant_mode(
                delay.model_from_simulation(fast_delay=L))["period_months"]
            sim = delay.realised_period(
                delay.simulate(n_months=900, fast_delay=L, seed=3).price)
            assert 1.0 < sim / lin < 1.8, f"delay {L}: linear {lin}, simulated {sim}"

    def test_both_agree_that_a_longer_fast_delay_is_a_longer_cycle(self):
        lins, sims = [], []
        for L in (2, 4, 9):
            lins.append(delay.dominant_mode(
                delay.model_from_simulation(fast_delay=L))["period_months"])
            sims.append(delay.realised_period(
                delay.simulate(n_months=900, fast_delay=L, seed=3).price))
        assert lins == sorted(lins) and sims == sorted(sims)

    def test_the_trade_ratio_raises_the_loop_gain(self):
        """Stacked product consuming more wafer per bit is a stability parameter.

        Not just a level one: it multiplies the fast channel's gain, so it shortens
        the cycle and pushes the dominant root outward.
        """
        low = delay.model_from_simulation(trade_ratio=1.0)
        high = delay.model_from_simulation(trade_ratio=3.0)
        assert high.channels[0].gain > low.channels[0].gain
        assert (delay.dominant_mode(high)["growth_per_step"]
                > delay.dominant_mode(low)["growth_per_step"])


class TestMixDrag:
    def test_matches_the_ratio_of_divisors(self):
        d = delay.mix_drag(trade_ratio=3.0, mix_start=0.02, mix_target=0.23,
                           mix_years=12.0)
        assert d["wafer_factor"] == pytest.approx(1.46 / 1.04, rel=1e-9)
        assert d["extra_wafers_pct"] == pytest.approx(40.4, abs=0.2)
        # Compounded over the adoption period, so it is comparable with a bit-growth
        # rate rather than with a one-off.
        assert d["annual_drag_pct"] == pytest.approx(2.86, abs=0.05)

    def test_no_shift_is_no_drag(self):
        d = delay.mix_drag(trade_ratio=3.0, mix_start=0.2, mix_target=0.2,
                           mix_years=10.0)
        assert d["extra_wafers_pct"] == pytest.approx(0.0)

    def test_a_unit_trade_ratio_is_no_drag(self):
        d = delay.mix_drag(trade_ratio=1.0, mix_start=0.0, mix_target=0.9,
                           mix_years=10.0)
        assert d["extra_wafers_pct"] == pytest.approx(0.0)

    def test_rejects_impossible_inputs(self):
        with pytest.raises(ValueError):
            delay.mix_drag(trade_ratio=0.5, mix_start=0.0, mix_target=0.2,
                           mix_years=10.0)
        with pytest.raises(ValueError):
            delay.mix_drag(trade_ratio=3.0, mix_start=0.0, mix_target=0.2,
                           mix_years=0.0)
