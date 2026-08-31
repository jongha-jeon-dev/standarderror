"""Tests for the hand-downloaded price-file loader.

Written against mock files in the shapes real exchange exports actually arrive in,
because every one of these tests corresponds to a way such a file has silently
mis-parsed for somebody: newest-first ordering, thousands separators, a percentage
column sitting next to the close, cp949 encoding, and a header row that pandas reads
as a MultiIndex.

The two that matter most are `test_descending_export_is_sorted` — wrong order flips
every return relative to its date and nothing downstream complains — and
`test_refuses_a_percentage_column_as_the_level`, which is the mistake that would
produce a completely plausible and completely wrong post.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standarderror.sources import licence_warnings, prices


def _levels(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    lv = 2000.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    return dates, lv


@pytest.fixture
def krx_csv(tmp_path):
    """KRX-style: Korean headers, newest first, thousands separators, 등락률."""
    dates, lv = _levels()
    frame = pd.DataFrame({
        "일자": [d.strftime("%Y/%m/%d") for d in dates],
        "종가": [f"{v:,.2f}" for v in lv],
        "대비": [f"{d:,.2f}" for d in np.r_[0.0, np.diff(lv)]],
        "등락률": [f"{p:.2f}" for p in np.r_[0.0, np.diff(lv) / lv[:-1] * 100]],
        "거래량": [f"{int(v):,}" for v in np.full(len(lv), 400_000_000)],
    }).iloc[::-1]                                  # newest first
    p = tmp_path / "krx_kospi.csv"
    frame.to_csv(p, index=False, encoding="cp949")
    return p, dates, lv


class TestLoadPrices:
    def test_reads_a_krx_style_export(self, krx_csv):
        p, dates, lv = krx_csv
        out = prices.load_prices(p)
        assert list(out.columns) == ["close"]
        assert len(out) == len(lv)
        assert out.attrs["standarderror"]["close_column"] == "종가"
        assert out.attrs["standarderror"]["date_column"] == "일자"

    def test_descending_export_is_sorted(self, krx_csv):
        """The bug with no symptom: unsorted, every return contradicts its date."""
        p, dates, lv = krx_csv
        out = prices.load_prices(p)
        assert out.index.is_monotonic_increasing
        assert out.index[0] == dates[0]
        # The mock export rounds to two decimals, as a real one does.
        assert out["close"].iloc[0] == pytest.approx(lv[0], abs=0.01)
        assert out["close"].iloc[-1] == pytest.approx(lv[-1], abs=0.01)

    def test_thousands_separators_survive(self, krx_csv):
        p, dates, lv = krx_csv
        out = prices.load_prices(p)
        assert out["close"].notna().all()
        assert out["close"].max() > 100.0

    def test_refuses_a_percentage_column_as_the_level(self, krx_csv):
        """등락률 and 대비 must never be picked; they sit right next to 종가."""
        p, _, _ = krx_csv
        assert prices.load_prices(p).attrs["standarderror"]["close_column"] == "종가"

    def test_picks_an_adjusted_close_over_a_raw_one(self, tmp_path):
        dates, lv = _levels(120)
        pd.DataFrame({"Date": dates, "Close": lv,
                      "Adj Close": lv * 0.5}).to_csv(tmp_path / "y.csv", index=False)
        out = prices.load_prices(tmp_path / "y.csv")
        assert out.attrs["standarderror"]["close_column"] == "Adj Close"
        assert out["close"].iloc[0] == pytest.approx(lv[0] * 0.5, rel=1e-9)

    def test_handles_bare_yyyymmdd_dates(self, tmp_path):
        dates, lv = _levels(120)
        pd.DataFrame({"기준일자": [d.strftime("%Y%m%d") for d in dates],
                      "종가지수": lv}).to_csv(tmp_path / "b.csv", index=False)
        out = prices.load_prices(tmp_path / "b.csv")
        assert out.index[0] == dates[0]
        assert len(out) == 120

    def test_reads_an_excel_export(self, tmp_path):
        # pandas does not pull in an Excel engine, and this package does not
        # declare one, so the reader is optional and so is its test.
        pytest.importorskip("openpyxl")
        dates, lv = _levels(120)
        p = tmp_path / "x.xlsx"
        pd.DataFrame({"날짜": dates, "종가": lv}).to_excel(p, index=False)
        assert len(prices.load_prices(p)) == 120

    def test_deduplicates_repeated_dates(self, tmp_path):
        dates, lv = _levels(120)
        frame = pd.DataFrame({"date": list(dates) + [dates[-1]],
                              "close": list(lv) + [lv[-1] * 2]})
        frame.to_csv(tmp_path / "d.csv", index=False)
        out = prices.load_prices(tmp_path / "d.csv")
        assert len(out) == 120
        assert out["close"].iloc[-1] == pytest.approx(lv[-1] * 2, rel=1e-9)

    def test_parenthesised_negatives_and_dashes(self, tmp_path):
        dates, _ = _levels(60)
        vals = [f"{1000 + i}" for i in range(58)] + ["-", "(5.0)"]
        pd.DataFrame({"date": dates, "close": vals}).to_csv(tmp_path / "n.csv",
                                                            index=False)
        out = prices.load_prices(tmp_path / "n.csv", min_rows=50)
        assert np.isnan(out["close"].iloc[-2])
        assert out["close"].iloc[-1] == pytest.approx(-5.0)

    def test_flattens_a_two_row_header(self, tmp_path):
        dates, lv = _levels(120)
        p = tmp_path / "m.csv"
        with p.open("w", encoding="utf-8") as fh:
            fh.write("Price,Close\nTicker,^KS11\n")
            for d, v in zip(dates, lv):
                fh.write(f"{d.date()},{v}\n")
        out = prices.load_prices(p, date_column="Price", close_column="Close")
        assert len(out) == 120

    def test_raises_when_no_level_column_is_findable(self, tmp_path):
        dates, lv = _levels(60)
        pd.DataFrame({"date": dates, "등락률": lv}).to_csv(tmp_path / "p.csv",
                                                          index=False)
        with pytest.raises(ValueError, match="no closing-level column"):
            prices.load_prices(tmp_path / "p.csv")

    def test_raises_when_no_date_column_is_findable(self, tmp_path):
        dates, lv = _levels(60)
        pd.DataFrame({"whenever": dates, "close": lv}).to_csv(tmp_path / "q.csv",
                                                              index=False)
        with pytest.raises(ValueError, match="no date column"):
            prices.load_prices(tmp_path / "q.csv")

    def test_raises_when_almost_nothing_parsed(self, tmp_path):
        dates, _ = _levels(60)
        pd.DataFrame({"date": dates,
                      "close": ["n/a"] * 55 + ["1", "2", "3", "4", "5"]}
                     ).to_csv(tmp_path / "r.csv", index=False)
        with pytest.raises(ValueError, match="usable rows"):
            prices.load_prices(tmp_path / "r.csv")

    def test_explicit_columns_override_detection(self, krx_csv):
        p, _, _ = krx_csv
        out = prices.load_prices(p, close_column="등락률", date_column="일자",
                                 min_rows=10)
        assert out.attrs["standarderror"]["close_column"] == "등락률"

    def test_defaults_to_not_redistributable(self, krx_csv):
        p, _, _ = krx_csv
        out = prices.load_prices(p)
        assert out.attrs["standarderror"]["redistributable"] is False
        warnings = licence_warnings(out)
        assert warnings and "NOT redistributable" in warnings[0]


class TestToLogReturns:
    def test_recovers_a_known_return(self):
        s = pd.Series([100.0, 110.0], index=pd.to_datetime(["2020-01-01",
                                                            "2020-01-02"]))
        r = prices.to_log_returns(s)
        assert r.iloc[0] == pytest.approx(100 * np.log(1.1))
        assert len(r) == 1

    def test_fractions_on_request(self):
        s = pd.Series([100.0, 110.0])
        assert prices.to_log_returns(s, in_percent=False).iloc[0] == pytest.approx(
            np.log(1.1))

    def test_refuses_a_non_positive_level(self):
        s = pd.Series([100.0, 0.0, 90.0])
        with pytest.raises(ValueError, match="non-positive"):
            prices.to_log_returns(s)

    def test_drops_missing_levels_rather_than_bridging_them(self):
        s = pd.Series([100.0, np.nan, 121.0])
        r = prices.to_log_returns(s)
        assert len(r) == 1
        assert r.iloc[0] == pytest.approx(100 * np.log(1.21))

    def test_carries_provenance_through(self, krx_csv):
        p, _, _ = krx_csv
        r = prices.to_log_returns(prices.load_prices(p))
        assert r.attrs["standarderror"]["source_file"] == "krx_kospi.csv"

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            prices.to_log_returns(pd.Series([], dtype=float))


class TestPublishableStatistics:
    def test_publishes_no_identifying_observation(self, krx_csv):
        """The licence guard, aimed at the values that actually identify a day.

        An earlier version of this test scanned the output for *any* observation and
        failed on the median absolute return — which on an odd-length sample is
        literally one data point. That is the wrong target: a median names no date,
        carries no sign, and is a textbook summary. The minimum and maximum are the
        opposite on every count, which is why they were removed from the helper.
        """
        p, _, _ = krx_csv
        r = prices.to_log_returns(prices.load_prices(p))
        stats = prices.publishable_statistics(r)
        flat = str(stats)
        assert "log_return" not in flat
        assert "worst_pct" not in stats and "best_pct" not in stats
        assert str(round(float(r.min()), 6)) not in flat
        assert str(round(float(r.max()), 6)) not in flat
        assert set(stats) >= {"n", "sd_pct", "excess_kurtosis", "abs_quantiles_pct"}

    def test_publishes_only_a_handful_of_numbers(self, krx_csv):
        """Structural check: a summary, not a compressed copy of the series."""
        p, _, _ = krx_csv
        r = prices.to_log_returns(prices.load_prices(p))
        stats = prices.publishable_statistics(r)
        numbers = [v for v in stats.values() if isinstance(v, (int, float))]
        numbers += list(stats["abs_quantiles_pct"].values())
        assert len(numbers) <= 10
        assert len(numbers) < 0.05 * r.size

    def test_reports_a_date_range_not_dates(self, krx_csv):
        p, dates, _ = krx_csv
        stats = prices.publishable_statistics(
            prices.to_log_returns(prices.load_prices(p)))
        assert stats["first_date"] == str(dates[1].date())
        assert stats["last_date"] == str(dates[-1].date())

    def test_moments_are_right_on_a_known_sample(self):
        z = pd.Series(np.random.default_rng(0).standard_normal(200_000))
        stats = prices.publishable_statistics(z)
        assert stats["sd_pct"] == pytest.approx(1.0, rel=0.02)
        assert stats["excess_kurtosis"] == pytest.approx(0.0, abs=0.06)
        assert stats["skewness"] == pytest.approx(0.0, abs=0.03)

    def test_annualisation_is_root_252(self):
        z = pd.Series(np.random.default_rng(1).standard_normal(50_000))
        stats = prices.publishable_statistics(z)
        assert stats["annualised_sd_pct"] == pytest.approx(
            stats["sd_pct"] * np.sqrt(252), rel=1e-9)

    def test_no_dates_when_the_index_is_not_a_datetime(self):
        stats = prices.publishable_statistics(pd.Series([0.1, -0.2, 0.3]))
        assert stats["first_date"] is None and stats["last_date"] is None

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            prices.publishable_statistics(pd.Series([], dtype=float))


class TestQuantileTailGuard:
    """An extreme quantile of a short sample is a single observation in disguise."""

    def test_drops_a_quantile_whose_tail_is_too_thin(self):
        r = pd.Series(np.random.default_rng(0).standard_normal(300))
        stats = prices.publishable_statistics(r, quantiles=(0.5, 0.99, 0.999))
        assert 0.999 in stats["quantiles_omitted"]
        assert 0.999 not in stats["abs_quantiles_pct"]
        assert 0.5 in stats["abs_quantiles_pct"]

    def test_keeps_it_when_there_is_enough_history(self):
        r = pd.Series(np.random.default_rng(0).standard_normal(50_000))
        stats = prices.publishable_statistics(r, quantiles=(0.999,))
        assert stats["abs_quantiles_pct"] and not stats["quantiles_omitted"]

    def test_omissions_are_reported_rather_than_silent(self):
        r = pd.Series(np.random.default_rng(0).standard_normal(200))
        stats = prices.publishable_statistics(r, quantiles=(0.999,))
        assert stats["quantiles_omitted"] == {0.999: 0}

    def test_rejects_a_quantile_outside_the_open_interval(self):
        r = pd.Series(np.random.default_rng(0).standard_normal(500))
        for q in (0.0, 1.0, 1.5, -0.2):
            with pytest.raises(ValueError):
                prices.publishable_statistics(r, quantiles=(q,))


class TestFredShapedExport:
    """FRED's keyless CSV is the easiest real source to obtain, and its shape is
    unlike an exchange export: two columns, the value column named after the series
    mnemonic, and a bare "." on every non-trading day."""

    @pytest.fixture
    def fred_csv(self, tmp_path):
        dates, lv = _levels(600)
        vals = [f"{v:.2f}" for v in lv]
        for i in (10, 57, 300):                    # holidays, FRED-style
            vals[i] = "."
        p = tmp_path / "NASDAQCOM.csv"
        with p.open("w", encoding="utf-8") as fh:
            fh.write("observation_date,NASDAQCOM\n")
            for d, v in zip(dates, vals):
                fh.write(f"{d.date()},{v}\n")
        return p, dates, lv

    def test_reads_it_without_being_told_the_columns(self, fred_csv):
        p, dates, lv = fred_csv
        out = prices.load_prices(p)
        assert out.attrs["standarderror"]["date_column"] == "observation_date"
        assert out.attrs["standarderror"]["close_column"] == "NASDAQCOM"
        assert len(out) == 600

    def test_holiday_dots_become_missing_not_zero(self, fred_csv):
        p, _, _ = fred_csv
        out = prices.load_prices(p)
        assert out["close"].isna().sum() == 3
        assert (out["close"].dropna() > 0).all()

    def test_returns_skip_the_holidays_without_a_zero_move(self, fred_csv):
        p, _, _ = fred_csv
        r = prices.to_log_returns(prices.load_prices(p))
        assert len(r) == 596                       # 600 - 3 missing - 1 differenced
        assert (r != 0).all()

    def test_a_two_column_file_still_refuses_an_obvious_non_level(self, tmp_path):
        """The two-column fallback must not resurrect a percentage column."""
        dates, lv = _levels(120)
        pd.DataFrame({"observation_date": dates, "등락률": lv}).to_csv(
            tmp_path / "pct.csv", index=False)
        with pytest.raises(ValueError, match="no closing-level column"):
            prices.load_prices(tmp_path / "pct.csv")
