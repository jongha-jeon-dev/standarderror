"""Tests for the BLS download parser and the noise-scale estimator.

Both halves are here because both are ways to silently corrupt the same analysis.
The parser's failure modes produce a series that looks fine and is wrong — a
thirteenth month, a final quarter of zeros, a hole in the middle — and every one
of them is a real habit of the file BLS hands you. The estimator's failure mode is
subtler: it reports a level that is a lower bound rather than an estimate unless
the noise's autocorrelation is accounted for, so the tests pin the ratio, which is
what survives, rather than the level, which does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standarderror.sources import us_labor as ul
from standarderror.ts import noisescale as ns

WIDE = '''"Series Id:","LNS14000000"
"Seasonally Adjusted"
"Series title:","(Seas) Unemployment Rate"

Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,Annual
2024,3.7,3.9,3.9,3.9,4.0,4.1,4.2,4.2,4.1,4.1,4.2,4.1,4.0
2025,4.0,4.1,4.2,4.2,4.2,4.1,4.2,4.3,4.4,4.4,4.3,4.2,4.2
2026,4.1,4.2,4.3,4.2,,,,,,,,,
'''


@pytest.fixture(autouse=True)
def _tmp(tmp_path):
    _TMP[0] = tmp_path
    yield


#: A one-slot holder so the helper below can stay a plain function rather than
#: threading `tmp_path` through every assertion.
_TMP: list = [None]


def _write(tmp_path, text, name="bls.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


class TestParsingTheDownload:
    def test_the_annual_column_never_becomes_a_thirteenth_month(self):
        # Melting the wide table without dropping it adds one observation a year
        # at a value close to the others, which no chart would reveal.
        df = ul.load_bls_table(_write(_TMP[0], WIDE))
        assert "Annual" in df.attrs["dropped_columns"]
        assert (df.groupby(df.date.dt.year).size().loc[[2024, 2025]] == 12).all()

    def test_the_partial_current_year_is_dropped_not_zero_filled(self):
        df = ul.load_bls_table(_write(_TMP[0], WIDE, "b2.csv"))
        assert df.date.max() == pd.Timestamp("2026-04-01")
        assert (df.value > 0).all()

    def test_the_preamble_length_is_found_not_assumed(self):
        # Same table with two extra description rows: a fixed skiprows breaks,
        # searching for the Year row does not.
        text = WIDE.replace('"Seasonally Adjusted"',
                            '"Seasonally Adjusted"\n"Footnote:","x"\n"More:","y"')
        assert len(ul.load_bls_table(_write(_TMP[0], text, "b3.csv"))) == 28

    def test_the_long_flat_file_parses_and_drops_the_annual_period(self):
        text = ("series_id\tyear\tperiod\tvalue\tfootnote_codes\n"
                "LNS14000000\t2024\tM01\t3.7\t\n"
                "LNS14000000\t2024\tM02\t3.9\t\n"
                "LNS14000000\t2024\tM13\t4.0\t\n")
        df = ul.load_bls_table(_write(_TMP[0], text, "b4.txt"))
        assert len(df) == 2
        assert list(df.value) == [3.7, 3.9]

    def test_a_file_that_is_not_a_data_table_says_so(self):
        with pytest.raises(ValueError, match="does not look like"):
            ul.load_bls_table(_write(_TMP[0], "some page text\nno table here\n",
                                     "b5.csv"))

    def test_a_hole_is_left_as_a_gap_and_never_closed(self):
        # October 2025 was never collected and never will be. Dropping it would
        # make September and November adjacent, and then every "one-month change"
        # across the join is a two-month change with no symptom. So the month
        # stays on the grid as a NaN.
        text = WIDE.replace(",4.4,4.4,4.3,4.2,4.2", ",4.4,,4.3,4.2,4.2")
        s = ul.monthly_series(_write(_TMP[0], text, "b6.csv"))
        assert s.attrs["gaps"] == ["2025-10-01"]
        assert np.isnan(s.loc["2025-10-01"])
        assert len(s) == 28                      # the grid is still complete
        # And a second difference across the hole comes back as missing rather
        # than as a number, which is the behaviour the estimators rely on.
        d2 = np.diff(s.to_numpy(), n=2)
        hole = int(s.index.get_loc(pd.Timestamp("2025-10-01")))
        # x[hole] enters the second differences at hole-2, hole-1 and hole.
        assert np.isnan(d2[hole - 2:hole + 1]).all()
        assert int(np.isnan(d2).sum()) == 3

    def test_strict_mode_refuses_the_hole_instead(self):
        text = WIDE.replace(",4.4,4.4,4.3,4.2,4.2", ",4.4,,4.3,4.2,4.2")
        with pytest.raises(ValueError, match="missing months"):
            ul.monthly_series(_write(_TMP[0], text, "b6b.csv"), strict=True)

    def test_contiguous_runs_split_at_the_hole(self):
        text = WIDE.replace(",4.4,4.4,4.3,4.2,4.2", ",4.4,,4.3,4.2,4.2")
        runs = ul.contiguous_runs(
            ul.monthly_series(_write(_TMP[0], text, "b6c.csv")))
        assert len(runs) == 2
        assert runs[0].index[-1] == pd.Timestamp("2025-09-01")
        assert runs[1].index[0] == pd.Timestamp("2025-11-01")

    def test_a_contiguous_series_carries_its_licence(self):
        s = ul.monthly_series(_write(_TMP[0], WIDE, "b7.csv"))
        assert "Bureau of Labor Statistics" in s.attrs["licence"]
        assert s.index.freqstr in (None, "MS") and len(s) == 28

    def test_the_documented_survey_figures_are_internally_consistent(self):
        c = ul.CPS_2026
        # The whole argument rests on these being the paper's numbers, so a typo
        # that reverses a direction should fail here rather than in the prose.
        assert c["people_per_respondent_now"] > c["people_per_respondent_then"]
        assert max(c["response_rate_now"]) < min(c["response_rate_then"])
        assert c["ci90_change_now"] > c["detectable_change_then"]


class TestNoiseScale:
    def test_it_recovers_a_known_sigma_through_a_moving_trend(self):
        rng = np.random.default_rng(0)
        n = 800
        trend = np.linspace(8.0, 4.0, n) + 0.7 * np.sin(np.arange(n) / 37.0)
        for sigma in (0.05, 0.13, 0.30):
            x = trend + rng.normal(0, sigma, n)
            assert ns.second_difference_scale(x) == pytest.approx(sigma, rel=0.12)

    def test_a_kink_in_the_trend_barely_moves_it(self):
        # The estimator has to survive 2008 and 2020 sitting inside the window,
        # which is what the robust scale is for.
        rng = np.random.default_rng(1)
        n, sigma = 600, 0.13
        base = np.linspace(5.0, 4.0, n)
        kinked = base + np.maximum(np.arange(n) - 300, 0) * 0.02
        e = rng.normal(0, sigma, n)
        assert (ns.second_difference_scale(kinked + e)
                == pytest.approx(ns.second_difference_scale(base + e), rel=0.05))

    def test_a_single_spike_wrecks_the_non_robust_version_and_not_this_one(self):
        rng = np.random.default_rng(2)
        x = np.linspace(4, 5, 400) + rng.normal(0, 0.10, 400)
        spiked = x.copy()
        spiked[200] += 8.0                      # a 2020-sized month
        assert ns.second_difference_scale(spiked) == pytest.approx(0.10, rel=0.15)
        assert ns.second_difference_scale(spiked, robust=False) > 0.3

    def test_the_ratio_recovers_a_known_doubling(self):
        rng = np.random.default_rng(3)
        n = 900
        trend = np.linspace(6, 4, n)
        x = trend + np.concatenate([rng.normal(0, 0.10, 500),
                                    rng.normal(0, 0.20, n - 500)])
        out = ns.scale_ratio(x, early=(20, 480), late=(520, n), reps=400)
        assert out["lo"] < 2.0 < out["hi"]
        assert out["ratio"] == pytest.approx(2.0, rel=0.25)

    def test_the_ratio_is_one_when_nothing_changed(self):
        rng = np.random.default_rng(4)
        x = np.linspace(6, 4, 900) + rng.normal(0, 0.13, 900)
        out = ns.scale_ratio(x, early=(20, 460), late=(470, 900), reps=400)
        assert out["lo"] < 1.0 < out["hi"]

    def test_the_ratio_does_not_care_about_the_autocorrelation_constant(self):
        # The point of quoting the ratio: an unknown but constant correction
        # cancels, so the headline number needs no assumption the data cannot
        # support.
        rng = np.random.default_rng(5)
        x = np.linspace(6, 4, 800) + np.concatenate(
            [rng.normal(0, 0.10, 400), rng.normal(0, 0.18, 400)])
        a = ns.scale_ratio(x, early=(10, 390), late=(410, 800), reps=200)
        b = ns.scale_ratio(x, early=(10, 390), late=(410, 800), reps=200)
        assert a["ratio"] == pytest.approx(b["ratio"], rel=1e-12)

    def test_rolling_windows_are_aligned_to_the_original_series(self):
        # The second difference is two shorter and offset by two; getting this
        # wrong dates every conclusion two months early.
        rng = np.random.default_rng(6)
        n = 400
        x = np.linspace(6, 4, n) + np.concatenate(
            [rng.normal(0, 0.05, 200), rng.normal(0, 0.25, n - 200)])
        out = ns.rolling_noise_scale(x, window=48)
        jump = out["centre"][int(np.argmax(np.diff(out["sigma"])) + 1)]
        assert abs(jump - 200) < 40, jump
        assert out["centre"].max() < n

    def test_a_window_that_is_too_short_or_too_long_raises(self):
        x = np.random.default_rng(7).normal(size=100)
        with pytest.raises(ValueError):
            ns.rolling_noise_scale(x, window=5)
        with pytest.raises(ValueError):
            ns.rolling_noise_scale(x, window=200)

    def test_the_ar1_correction_is_the_published_polynomial(self):
        assert ns.ar1_factor(0.0) == pytest.approx(6.0)
        assert ns.ar1_factor(0.5) == pytest.approx(2.5)
        # Positive autocorrelation shrinks the factor, so assuming independence
        # understates sigma — the direction matters and is asserted.
        assert ns.ar1_factor(0.4) < ns.ar1_factor(0.0)
        with pytest.raises(ValueError):
            ns.ar1_factor(1.0)

    def test_the_rounding_floor_is_the_uniform_standard_deviation(self):
        assert ns.rounding_floor(0.1) == pytest.approx(0.1 / np.sqrt(12))

    def test_the_panel_overlap_halves_the_detectable_change(self):
        # Adjacent CPS months share three quarters of the sample, so a
        # month-on-month difference is much less noisy than independent sampling
        # implies. Ignoring it overstates the noise, which is the direction that
        # flatters the argument, so it is pinned.
        indep = ns.detectable_change(0.13)
        panel = ns.detectable_change(0.13, overlap=ul.CPS_2026["monthly_overlap"])
        assert panel == pytest.approx(indep / 2.0, rel=1e-9)

    def test_the_published_claim_and_the_panel_overlap_agree(self):
        # BLS says a 0.18 pp monthly change used to be detectable. At 90%
        # confidence with three-quarters overlap that implies a level noise of
        # about 0.155 pp, which is the number the post has to reproduce from the
        # series itself. If this arithmetic ever stops closing, the post's
        # cross-check is measuring something else.
        sigma = 0.155
        got = ns.detectable_change(sigma, alpha=0.10,
                                   overlap=ul.CPS_2026["monthly_overlap"])
        assert got == pytest.approx(ul.CPS_2026["detectable_change_then"],
                                    abs=0.015)

    def test_an_impossible_overlap_raises(self):
        with pytest.raises(ValueError):
            ns.detectable_change(0.1, overlap=1.0)


class TestTheLatticeProblem:
    """The failure that made the first version of this module useless.

    A rate published to one decimal puts its own second difference on a 0.1
    lattice, and a median absolute deviation of lattice values is itself a lattice
    value. The scale estimate then has about four usable rungs across eighty
    years, and a 1.7-fold change in the underlying noise falls between two of
    them. The information is not missing — averaging recovers it — so the fix is
    an estimator that averages.
    """

    @staticmethod
    def _quantised(sigma, n=800, seed=0):
        rng = np.random.default_rng(seed)
        trend = np.linspace(8.0, 4.0, n) + 0.7 * np.sin(np.arange(n) / 37.0)
        return np.round(trend + rng.normal(0, sigma, n), 1)

    def test_mad_returns_only_lattice_points(self):
        seen = {round(ns.mad_scale(np.diff(self._quantised(s), n=2)), 6)
                for s in (0.14, 0.16, 0.18, 0.20)}
        assert len(seen) < 4, seen               # four sigmas, fewer answers

    def test_one_rung_of_the_lattice_is_large_enough_to_matter(self):
        # 0.06 pp on a quantity whose whole range of interest is 0.15 to 0.26.
        assert ns.lattice_resolution(0.1) == pytest.approx(0.0605, abs=1e-3)

    def test_the_trimmed_scale_resolves_what_mad_cannot(self):
        got = [ns.second_difference_scale(self._quantised(s))
               for s in (0.14, 0.18, 0.22, 0.26)]
        assert all(b > a for a, b in zip(got, got[1:])), got
        assert got[-1] / got[0] == pytest.approx(0.26 / 0.14, rel=0.15)

    def test_quantisation_costs_almost_no_variance(self):
        # The reason the information was recoverable: rounding to 0.1 adds
        # 0.0289^2 of variance, about 2% when sigma is 0.16.
        rng = np.random.default_rng(1)
        raw = rng.normal(0, 0.16, 200_000)
        assert (np.round(raw, 1).std() / raw.std()) == pytest.approx(1.0, abs=0.02)

    def test_the_winsor_constant_is_one_when_nothing_is_trimmed(self):
        assert ns.winsor_constant(0.0) == pytest.approx(1.0)
        assert ns.winsor_constant(0.10) == pytest.approx(0.9117, abs=1e-3)
        with pytest.raises(ValueError):
            ns.winsor_constant(1.0)

    def test_the_trimmed_scale_still_survives_a_pandemic_sized_spike(self):
        rng = np.random.default_rng(2)
        x = np.linspace(4, 5, 400) + rng.normal(0, 0.10, 400)
        x[200] += 8.0
        assert ns.second_difference_scale(x) == pytest.approx(0.10, rel=0.25)
        assert ns.second_difference_scale(x, robust=False) > 0.3
