"""The tempo identities, checked against the algebra they are derived from.

A normal schedule was chosen precisely so that equations (1)-(3) in
`tempo.py` are exact, which makes every test here a check on the code rather
than on the demography.
"""

import numpy as np
import pytest

from standarderror.aggregates import tempo as tp

DELTAS = (0.0, 0.05, 0.1, 0.2, 0.35)


class TestTheIdentities:
    @pytest.mark.parametrize("delta", DELTAS)
    def test_the_period_rate_is_the_quantum_over_one_plus_delta(self, delta):
        """Equation (1). The whole episode rests on this one."""
        rows = tp.simulate(quantum=1.8, delta=delta, years=range(15, 26))
        for row in rows:
            assert row.tfr == pytest.approx(tp.period_tfr(1.8, delta),
                                            abs=1e-4)

    @pytest.mark.parametrize("delta", DELTAS)
    def test_the_period_mean_age_rises_more_slowly_than_cohorts_postpone(
            self, delta):
        """Equation (2), and the trap in it: `r` is `delta / (1 + delta)`, not
        `delta`. Reading a published mean-age change as the postponement rate
        overstates it."""
        rows = tp.simulate(delta=delta, years=range(15, 26))
        r = tp.mac_change(rows, 20)
        assert r == pytest.approx(tp.mac_rate(delta), abs=1e-4)
        if delta > 0:
            assert r < delta

    @pytest.mark.parametrize("delta", DELTAS)
    @pytest.mark.parametrize("quantum", (1.2, 1.8, 2.4))
    def test_the_adjustment_recovers_the_quantum_exactly(self, delta, quantum):
        """Equation (3). On this schedule Bongaarts-Feeney is an identity, not
        an approximation, which is why a numerical disagreement here would be a
        bug rather than a modelling choice."""
        rows = tp.simulate(quantum=quantum, delta=delta, years=range(15, 26))
        row = next(r for r in rows if r.year == 20)
        r = tp.mac_change(rows, 20)
        assert tp.bongaarts_feeney(row.tfr, r) == pytest.approx(quantum,
                                                                rel=1e-3)

    def test_postponement_compresses_the_period_schedule(self):
        """The period density's spread is `sigma / (1 + delta)` -- narrower than
        any cohort's. A period schedule is not a cohort schedule even in shape."""
        for delta in (0.1, 0.2, 0.35):
            row = tp.simulate(spread=4.5, delta=delta,
                              years=range(20, 21))[0]
            assert row.spread == pytest.approx(4.5 / (1 + delta), abs=1e-3)

    def test_no_postponement_means_no_gap(self):
        row = tp.simulate(quantum=1.8, delta=0.0, years=range(20, 21))[0]
        assert row.tfr == pytest.approx(1.8, abs=1e-4)
        assert row.shortfall == pytest.approx(0.0, abs=1e-4)


class TestTheReboundNobodyCaused:
    @pytest.fixture(scope="class")
    def rows(self):
        return tp.rebound(quantum=1.8, fast=0.2, slow=0.1, switch=40)

    def test_the_quantum_never_moves(self, rows):
        """Stated as a test because it is the claim a reader will not believe."""
        assert {r.quantum for r in rows} == {1.8}

    def test_the_period_rate_rises_by_the_asymptote(self, rows):
        """From `Q / 1.2` to `Q / 1.1`, which is +9.1%, and the simulation gets
        there rather than merely trending towards it."""
        lo, hi = min(r.tfr for r in rows), max(r.tfr for r in rows)
        assert lo == pytest.approx(1.8 / 1.2, abs=1e-3)
        assert hi == pytest.approx(1.8 / 1.1, abs=1e-3)
        assert hi / lo - 1 == pytest.approx(1.2 / 1.1 - 1, abs=1e-3)

    def test_it_takes_most_of_a_generation_to_arrive(self, rows):
        """`switch` is a birth cohort, so nothing happens until those women
        reach childbearing age -- 28 years here -- and the rise then takes about
        another twenty. A policy evaluated on three years of period TFR is
        reading a signal whose cause is decades old."""
        lo, hi = min(r.tfr for r in rows), max(r.tfr for r in rows)
        first = next(r.year for r in rows if r.tfr > lo * 1.001)
        done = next(r.year for r in rows if r.tfr > lo + 0.99 * (hi - lo))
        assert first - 40 > 20, first
        assert done - first > 15, (first, done)


class TestWhereItStops:
    def test_a_mean_age_rising_a_year_per_year_is_not_expressible(self):
        with pytest.raises(ValueError, match="cannot express"):
            tp.bongaarts_feeney(1.0, 1.0)

    def test_a_widening_schedule_costs_about_a_percent_and_not_more(self):
        """The assumption the literature attacks hardest -- a rigid shift with
        no change in variance -- turns out to be cheap. Widening the cohort
        schedule from 4.5 to 8.0 years of spread, which is far more than any
        country has done, biases the adjusted rate by 1.1%. Reported because it
        is the opposite of what I expected to find, and it agrees with Mazzuco
        and Zanotto (2025)."""
        control = tp.variance_bias(spread_from=4.5, spread_to=4.5)
        assert abs(control["error"]) < 1e-3
        errors = [tp.variance_bias(spread_from=4.5, spread_to=to)["error"]
                  for to in (5.0, 5.5, 6.5, 8.0)]
        assert all(e < 0 for e in errors), errors
        assert abs(errors[-1]) < 0.02, errors[-1]
        # monotone in how much the schedule widens, which is what makes it a
        # bias rather than noise
        assert errors == sorted(errors, reverse=True), errors

    def test_centring_needs_both_neighbours(self):
        rows = tp.simulate(years=range(20, 23))
        with pytest.raises(ValueError, match="need years"):
            tp.mac_change(rows, 20)


class TestTheMechanics:
    def test_a_year_does_not_depend_on_which_other_years_were_asked_for(self):
        """No generator and no accumulator runs through the loop, so a three-year
        window agrees with a forty-year one. This has been a real bug twice in
        this repository."""
        wide = {r.year: r.tfr for r in tp.simulate(delta=0.2,
                                                   years=range(0, 40))}
        narrow = {r.year: r.tfr for r in tp.simulate(delta=0.2,
                                                     years=range(18, 21))}
        for y, v in narrow.items():
            assert wide[y] == pytest.approx(v, rel=1e-12)

    def test_the_schedule_integrates_to_the_quantum(self):
        births = tp.schedule(tp.AGES, quantum=1.8, mean_age=30.0, spread=4.5)
        assert births.sum() == pytest.approx(1.8, abs=1e-4)

    def test_the_age_grid_is_wide_enough_for_a_postponed_schedule(self):
        """Truncation at the top of the grid would leak births and quietly break
        identity (1) for large `delta`."""
        births = tp.schedule(tp.AGES, quantum=1.8, mean_age=40.0, spread=6.0)
        assert births.sum() == pytest.approx(1.8, abs=2e-3)
        # The right question is the mass still sitting at the edge, not the
        # density at any single age: 0.26% of the quantum in the last three
        # years of age, on a schedule far later and wider than any real one.
        assert births[-3:].sum() < 0.01 * 1.8, births[-3:].sum()
        assert np.all(np.diff(births[-6:]) < 0)
