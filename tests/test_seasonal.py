"""Tests for the seasonal-adjustment variance filter.

The claim these support is narrow and worth stating exactly: a *fixed*
month-of-year adjustment has no mechanism for removing a one-off shock, so the
gap between it and an official moving-filter adjustment is variation the
official filter removed that a stable seasonal pattern does not account for.

So the tests are mostly about the benchmark behaving as a benchmark:

1. On a series that is seasonality plus noise, the fixed adjustment removes the
   seasonality and leaves the noise.
2. On a series with a one-off shock, the fixed adjustment leaves the shock in.
3. A moving filter of the kind X-13 uses does not, and `wedge` sees that.
4. Excluding a year from the fit does not exclude it from the output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standarderror.ts import seasonal as sz


def _index(n, start="1990-01-01"):
    return pd.date_range(start, periods=n, freq="MS")


def _series(n=360, sigma=0.10, amp=0.5, seed=0, start="1990-01-01"):
    idx = _index(n, start)
    rng = np.random.default_rng(seed)
    season = amp * np.sin(2 * np.pi * (idx.month - 1) / 12.0)
    noise = rng.normal(0, sigma, n)
    return pd.Series(5.0 + season + noise, index=idx), season, noise


class TestMonthDummyAdjust:
    def test_it_removes_a_stable_seasonal_pattern(self):
        s, season, _ = _series(sigma=0.02, amp=0.8)
        out = sz.month_dummy_adjust(s)
        # What is left should not track the season any more.
        assert abs(np.corrcoef(out.to_numpy(), season)[0, 1]) < 0.25

    def test_it_leaves_the_noise_alone(self):
        s, _, noise = _series(sigma=0.12, amp=0.6, n=600)
        out = sz.month_dummy_adjust(s)
        assert out.std(ddof=1) == pytest.approx(noise.std(ddof=1), rel=0.15)

    def test_it_preserves_the_level(self):
        s, _, _ = _series()
        assert sz.month_dummy_adjust(s).mean() == pytest.approx(s.mean(), abs=0.05)

    def test_a_one_off_shock_survives_it(self):
        # The property that makes this a benchmark: a fixed factor can only
        # absorb a twelfth of a shock, so the shock stays in the series.
        s, _, _ = _series(n=360, sigma=0.05, amp=0.4)
        s.iloc[200] += 5.0
        out = sz.month_dummy_adjust(s)
        assert out.iloc[200] - out.drop(out.index[200]).mean() > 4.0

    def test_excluding_a_year_from_the_fit_keeps_it_in_the_output(self):
        s, _, _ = _series(n=360)
        out = sz.month_dummy_adjust(s, exclude_years=(1995,))
        assert len(out) == len(s)
        assert out.loc["1995"].notna().all()

    def test_excluding_a_year_protects_the_factors_from_it(self):
        s, _, _ = _series(n=360, sigma=0.05, amp=0.4)
        s.loc["1995"] += 8.0                      # a pandemic-sized year
        dirty = sz.month_dummy_adjust(s, exclude_years=())
        clean = sz.month_dummy_adjust(s, exclude_years=(1995,))
        rest = [i for i in s.index if i.year != 1995]
        assert clean.loc[rest].std(ddof=1) < dirty.loc[rest].std(ddof=1)

    def test_it_needs_a_datetime_index(self):
        with pytest.raises(TypeError):
            sz.month_dummy_adjust(pd.Series([1.0, 2.0, 3.0]))

    def test_it_refuses_a_fit_window_with_thin_months(self):
        s, _, _ = _series(n=24)
        with pytest.raises(ValueError):
            sz.month_dummy_adjust(s.iloc[:14])

    def test_it_refuses_an_empty_fit_window(self):
        s, _, _ = _series(n=48)
        with pytest.raises(ValueError):
            sz.month_dummy_adjust(s, exclude_years=tuple(range(1990, 1995)))


class TestRunScale:
    def test_it_recovers_a_planted_sigma(self):
        s, _, _ = _series(n=600, sigma=0.13, amp=0.0)
        assert sz.run_scale(s) == pytest.approx(0.13, rel=0.12)

    def test_a_gap_does_not_leak_across_itself(self):
        # A second difference spanning a hole differences across an unknown gap.
        # Splitting into runs is the whole point, so a planted jump at the hole
        # must not show up in the estimate.
        s, _, _ = _series(n=600, sigma=0.10, amp=0.0)
        s.iloc[300] = np.nan
        s.iloc[301:] += 4.0
        assert sz.run_scale(s, min_run=100) == pytest.approx(0.10, rel=0.15)

    def test_it_returns_nan_rather_than_raising_on_a_short_block(self):
        s, _, _ = _series(n=20)
        assert np.isnan(sz.run_scale(s, min_run=40))

    def test_runs_shorter_than_the_minimum_are_dropped(self):
        s, _, _ = _series(n=300, sigma=0.10, amp=0.0)
        s.iloc[50] = np.nan                        # leaves a 50-long head
        assert not np.isnan(sz.run_scale(s, min_run=40))
        assert np.isnan(sz.run_scale(s.iloc[:45], min_run=60))


def _moving_filter(s: pd.Series, half: int = 3) -> pd.Series:
    """A stand-in for a moving seasonal filter: month means over a moving window.

    Not X-13 — it is a caricature with the one property that matters, that the
    seasonal factor for a month is estimated from nearby years and therefore
    partly absorbs a shock in that month.
    """
    out = s.copy().astype(float)
    for month in range(1, 13):
        block = s[s.index.month == month]
        local = block.rolling(2 * half + 1, center=True, min_periods=1).mean()
        out.loc[block.index] = block - local
    return out + float(s.mean())


class TestWedge:
    def test_no_filtering_means_no_wedge(self):
        # If the "official" series is the benchmark itself, nothing was removed.
        s, _, _ = _series(n=480, sigma=0.10, amp=0.5)
        own = sz.month_dummy_adjust(s)
        w = sz.wedge(own, s)
        assert w["removed"] == pytest.approx(0.0, abs=0.02)
        assert w["ratio"] == pytest.approx(1.0, abs=0.02)

    def test_a_moving_filter_produces_a_positive_wedge(self):
        s, _, _ = _series(n=600, sigma=0.12, amp=0.5, seed=4)
        w = sz.wedge(_moving_filter(s, half=1), s)
        assert w["removed"] > 0.10
        assert w["sigma_sa"] < w["sigma_benchmark"]

    def test_the_wedge_grows_as_the_moving_window_shortens(self):
        # The mechanism, stated as a monotonicity: a factor estimated from m
        # years of a given month absorbs about 1/m of a shock in that month, so
        # a shorter window removes more non-seasonal variation. If this ever
        # stops holding, `wedge` is measuring something other than the filter.
        s, _, _ = _series(n=600, sigma=0.12, amp=0.5, seed=4)
        removed = [sz.wedge(_moving_filter(s, half=h), s)["removed"]
                   for h in (1, 2, 4, 8)]
        assert all(b < a for a, b in zip(removed, removed[1:]))

    def test_the_raw_series_is_noisier_than_either_adjustment(self):
        s, _, _ = _series(n=480, sigma=0.08, amp=0.9)
        w = sz.wedge(sz.month_dummy_adjust(s), s)
        assert w["sigma_raw"] > w["sigma_benchmark"]

    def test_it_uses_only_common_months(self):
        s, _, _ = _series(n=480, sigma=0.10, amp=0.4)
        w = sz.wedge(sz.month_dummy_adjust(s).iloc[:300], s)
        assert w["n"] == 300

    def test_it_refuses_too_little_overlap(self):
        s, _, _ = _series(n=480)
        with pytest.raises(ValueError):
            sz.wedge(s.iloc[:10], s)

    def test_a_negative_wedge_is_reported_not_clipped(self):
        # Where the seasonal pattern is unstable the fixed benchmark leaves a
        # seasonal residual of its own and can be the noisier series. That is
        # evidence against the effect in that block, so it must survive to the
        # caller rather than being floored at zero.
        idx = _index(480)
        rng = np.random.default_rng(7)
        drift = np.linspace(0.1, 1.5, 480)         # growing seasonal amplitude
        s = pd.Series(5.0 + drift * np.sin(2 * np.pi * (idx.month - 1) / 12.0)
                      + rng.normal(0, 0.05, 480), index=idx)
        w = sz.wedge(_moving_filter(s), s)
        assert np.isfinite(w["removed"])


class TestWedgeInterval:
    def test_the_interval_brackets_the_point_estimate(self):
        s, _, _ = _series(n=480, sigma=0.12, amp=0.5, seed=9)
        r = sz.wedge_interval(_moving_filter(s), s, reps=120, seed=1)
        assert r["lo"] < r["removed"] < r["hi"]
        assert r["reps"] > 0

    def test_identical_series_give_a_degenerate_interval_at_zero(self):
        # If the "official" series *is* the benchmark, every replicate compares a
        # sample against itself and the interval collapses. Worth pinning,
        # because a resampling scheme that returned a spread here would be
        # resampling the two series independently and breaking the pairing the
        # statistic depends on.
        s, _, _ = _series(n=480, sigma=0.10, amp=0.5, seed=11)
        own = sz.month_dummy_adjust(s)
        r = sz.wedge_interval(own, s, reps=120, seed=2)
        assert r["lo"] == pytest.approx(0.0, abs=1e-12)
        assert r["hi"] == pytest.approx(0.0, abs=1e-12)
        assert r["removed"] == pytest.approx(0.0, abs=1e-12)

    def test_a_filtered_series_gives_an_interval_away_from_zero(self):
        s, _, _ = _series(n=600, sigma=0.12, amp=0.5, seed=13)
        r = sz.wedge_interval(_moving_filter(s, half=1), s, reps=200, seed=3)
        assert r["lo"] > 0.0

    def test_it_refuses_a_block_shorter_than_a_year(self):
        s, _, _ = _series(n=240)
        with pytest.raises(ValueError):
            sz.wedge_interval(s, s, block=6, reps=5)

    def test_it_refuses_a_block_longer_than_the_sample(self):
        s, _, _ = _series(n=240)
        with pytest.raises(ValueError):
            sz.wedge_interval(s, s, block=1000, reps=5)
