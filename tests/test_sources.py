"""Adapter tests against recorded payloads.

These do not hit the network by default. The point is to pin the *parsing*
contract — which is where adapters actually break, because upstream changes a
column name rather than going down. The FRED date-column rename from `DATE` to
`observation_date` is the live example: both fixtures are here and both must
parse, because we read column 0 positionally.

Run the live smoke tests with `QUANTPOST_NETWORK_TESTS=1 pytest -m network`.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from quantpost.sources import bis, citations, ecb, ecos, fred, licence_warnings, market

NETWORK = os.environ.get("QUANTPOST_NETWORK_TESTS") == "1"
needs_network = pytest.mark.skipif(not NETWORK,
                                   reason="set QUANTPOST_NETWORK_TESTS=1")


# --------------------------------------------------------------- fixtures

FRED_CSV_NEW = b"""observation_date,DGS10,DGS2
2026-07-01,4.31,3.88
2026-07-02,.,3.91
2026-07-03,4.28,3.90
"""

FRED_CSV_OLD = b"""DATE,DGS10
2026-07-01,4.31
2026-07-02,.
"""

ECB_CSV = (b"KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE,OBS_STATUS,TITLE\n"
           b"YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,2026-06-01,3.0680734066,A,x\n"
           b"YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,2026-06-02,3.0912,A,x\n")

BIS_CSV = (b"FREQ,BORROWERS_CTY,TC_BORROWERS,TC_LENDERS,CG_DTYPE,TIME_PERIOD,"
           b"OBS_VALUE,OBS_STATUS\n"
           b"Q,KR,P,A,C,2025-Q3,-7.5,A\n"
           b"Q,KR,P,A,C,2025-Q4,-8.0122,A\n"
           b"Q,KR,P,A,B,2025-Q4,160.1,A\n")

ECOS_JSON = json.dumps({"StatisticSearch": {"list_total_count": 2, "row": [
    {"STAT_CODE": "722Y001", "ITEM_CODE1": "0101000", "UNIT_NAME": "%",
     "TIME": "20260701", "DATA_VALUE": "2.75"},
    {"STAT_CODE": "722Y001", "ITEM_CODE1": "0101000", "UNIT_NAME": "%",
     "TIME": "20260702", "DATA_VALUE": "2.75"}]}}).encode()

ECOS_ERROR = json.dumps({"RESULT": {"CODE": "INFO-100",
                                    "MESSAGE": "인증키가 유효하지 않습니다"}}).encode()


@pytest.fixture
def patched_fetch(monkeypatch):
    """Replace the caching fetch with a canned-response dispatcher."""
    calls: list[str] = []
    table: dict[str, bytes] = {}

    def fake(url, *, source, **kw):
        calls.append(url)
        for needle, payload in table.items():
            if needle in url:
                return payload
        raise AssertionError(f"unexpected URL: {url}")

    for mod in (fred, ecb, ecos, bis, market):
        monkeypatch.setattr(mod, "fetch", fake)
    return {"calls": calls, "table": table}


# --------------------------------------------------------------- FRED

class TestFred:
    def test_parses_current_header(self, patched_fetch):
        patched_fetch["table"]["fredgraph.csv"] = FRED_CSV_NEW
        df = fred.get(["ust_10y", "ust_2y"], start="2026-07-01")
        assert list(df.columns) == ["DGS10", "DGS2"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df["DGS10"].isna().sum() == 1        # the "." became NaN
        assert df["DGS10"].iloc[0] == pytest.approx(4.31)

    def test_parses_legacy_header(self, patched_fetch):
        """Column 0 is read positionally, so the DATE -> observation_date rename
        cannot break the adapter."""
        patched_fetch["table"]["fredgraph.csv"] = FRED_CSV_OLD
        df = fred.get("ust_10y")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["DGS10"]

    def test_curated_aliases_resolve(self):
        assert fred.resolve(["ust_10y", "vix", "DGS5"]) == ["DGS10", "VIXCLS", "DGS5"]

    def test_undocumented_date_params_are_enforced_client_side(self, patched_fetch):
        """cosd/coed are UI internals, so the window is re-applied locally."""
        patched_fetch["table"]["fredgraph.csv"] = FRED_CSV_NEW
        df = fred.get("ust_10y", start="2026-07-03")
        assert len(df) == 1

    def test_truncated_ice_series_raises_a_licence_warning(self, patched_fetch):
        patched_fetch["table"]["fredgraph.csv"] = (
            b"observation_date,BAMLH0A0HYM2\n2026-07-01,3.12\n")
        df = fred.get("hy_oas")
        warns = licence_warnings(df)
        assert any("BAMLH0A0HYM2" in w for w in warns)

    def test_citation_carries_the_mandatory_disclaimer(self, patched_fetch):
        patched_fetch["table"]["fredgraph.csv"] = FRED_CSV_NEW
        df = fred.get("ust_10y")
        assert "Federal Reserve Bank of St. Louis" in citations(df)[0]
        assert "not endorsed or certified" in df.attrs["quantpost"]["licence"]

    def test_api_without_key_fails_clearly(self, monkeypatch):
        from dataclasses import replace
        monkeypatch.setattr(fred, "SETTINGS",
                            replace(fred.SETTINGS, fred_api_key=None))
        with pytest.raises(RuntimeError, match="FRED_API_KEY"):
            fred.get_api("DGS10")

    @needs_network
    def test_live_keyless_endpoint(self):
        df = fred.get("ust_10y", start="2026-01-01")
        assert len(df) > 50


# --------------------------------------------------------------- ECB

class TestEcb:
    def test_series_id_splits_dataflow_from_key(self):
        assert ecb.split_series_id("YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y") == (
            "YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y")

    def test_bare_dataflow_raises(self):
        with pytest.raises(ValueError):
            ecb.split_series_id("YC")

    def test_parses_wide_sdmx_csv_by_name(self, patched_fetch):
        patched_fetch["table"]["/YC/"] = ECB_CSV
        df = ecb.get("ea_aaa_10y", start="2026-06-01")
        assert list(df.columns) == ["ea_aaa_10y"]
        assert df["ea_aaa_10y"].iloc[0] == pytest.approx(3.0680734066)

    def test_url_omits_the_dataflow_prefix_from_the_key(self, patched_fetch):
        patched_fetch["table"]["/YC/"] = ECB_CSV
        ecb.get("ea_aaa_10y")
        url = patched_fetch["calls"][0]
        assert "/data/YC/B.U2.EUR" in url
        assert "/data/YC/YC." not in url
        assert "format=csvdata" in url

    def test_missing_columns_raise_a_useful_error(self, patched_fetch):
        patched_fetch["table"]["/YC/"] = b"A,B\n1,2\n"
        with pytest.raises(RuntimeError, match="TIME_PERIOD"):
            ecb.get("ea_aaa_10y")

    @needs_network
    def test_live_ciss_series(self):
        df = ecb.get("ciss_ea", last_n=5)
        assert len(df) >= 1


# --------------------------------------------------------------- ECOS

class TestEcos:
    def test_url_segment_order(self, patched_fetch):
        patched_fetch["table"]["StatisticSearch"] = ECOS_JSON
        ecos.get("base_rate", start="2026-01-01", end="2026-07-31")
        url = patched_fetch["calls"][0]
        parts = url.split("/api/")[1].split("/")
        assert parts[0] == "StatisticSearch"
        assert parts[2] == "json" and parts[3] == "kr"
        assert parts[6] == "722Y001"          # STAT_CODE after startRow/endRow
        assert parts[7] == "D"                # cycle
        assert parts[8] == "20260101" and parts[9] == "20260731"
        assert parts[10] == "0101000"         # ITEM_CODE

    def test_error_payload_raises_instead_of_keyerror(self, patched_fetch):
        patched_fetch["table"]["StatisticSearch"] = ECOS_ERROR
        with pytest.raises(RuntimeError, match="INFO-100"):
            ecos.get("base_rate")

    def test_parses_daily_time_field(self, patched_fetch):
        patched_fetch["table"]["StatisticSearch"] = ECOS_JSON
        df = ecos.get("base_rate")
        assert df.index[0] == pd.Timestamp("2026-07-01")
        assert df["base_rate"].iloc[0] == pytest.approx(2.75)

    def test_monthly_cycle_formats_dates_correctly(self, patched_fetch):
        patched_fetch["table"]["StatisticSearch"] = json.dumps(
            {"StatisticSearch": {"list_total_count": 1, "row": [
                {"TIME": "202606", "DATA_VALUE": "115.2"}]}}).encode()
        df = ecos.get("cpi", start="2020-01-01", end="2026-06-30")
        url = patched_fetch["calls"][0]
        assert "/M/202001/202606/" in url
        assert df.index[0] == pd.Timestamp("2026-06-01")

    def test_unknown_alias_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="base_rate"):
            ecos.get("nonexistent_series")


# --------------------------------------------------------------- BIS

class TestBis:
    def test_credit_gap_filters_on_dtype(self, patched_fetch):
        patched_fetch["table"]["WS_CREDIT_GAP"] = BIS_CSV
        df = bis.credit_gap("KR")
        assert list(df.columns) == ["KR"]
        assert len(df) == 2                     # the ratio row (B) is excluded
        assert df["KR"].iloc[-1] == pytest.approx(-8.0122)

    def test_quarterly_periods_become_timestamps(self, patched_fetch):
        patched_fetch["table"]["WS_CREDIT_GAP"] = BIS_CSV
        df = bis.credit_gap("KR")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[-1].year == 2025 and df.index[-1].quarter == 4

    def test_url_shape(self, patched_fetch):
        patched_fetch["table"]["WS_CREDIT_GAP"] = BIS_CSV
        bis.credit_gap("KR")
        url = patched_fetch["calls"][0]
        assert "/api/v2/data/dataflow/BIS/WS_CREDIT_GAP/1.0/Q.KR" in url
        assert "format=csv" in url

    def test_empty_filter_result_raises(self, patched_fetch):
        patched_fetch["table"]["WS_CREDIT_GAP"] = BIS_CSV
        with pytest.raises(RuntimeError, match="No WS_CREDIT_GAP"):
            bis.credit_gap("KR", dtype="Z")


# --------------------------------------------------------------- market

class TestMarket:
    def test_stooq_requires_explicit_opt_in(self):
        with pytest.raises(RuntimeError, match="accept_terms"):
            market.stooq("^spx")

    def test_stooq_is_flagged_non_redistributable(self, patched_fetch):
        patched_fetch["table"]["stooq.com"] = (
            b"Date,Open,High,Low,Close,Volume\n2026-07-01,1,2,0.5,1.5,100\n")
        df = market.stooq("^spx", accept_terms=True)
        assert df.attrs["quantpost"]["redistributable"] is False
        assert any("NOT redistributable" in w for w in licence_warnings(df))

    def test_stooq_bad_symbol_gives_a_useful_error(self, patched_fetch):
        patched_fetch["table"]["stooq.com"] = b"No data\n"
        with pytest.raises(RuntimeError, match="no usable CSV"):
            market.stooq("nonsense", accept_terms=True)
