"""Bank of Korea ECOS OpenAPI (한국은행 경제통계시스템).

URL grammar - order matters and the key sits in the *path*:

    /api/StatisticSearch/{key}/{json|xml}/{kr|en}/{startRow}/{endRow}
        /{STAT_CODE}/{cycle}/{start}/{end}/{ITEM_CODE1}[/{ITEM2}...]

`cycle` is one of A S Q M SM D (the old YY/QQ/MM/DD convention is retired), and
the date literals must match the cycle: YYYY / YYYYQ1 / YYYYMM / YYYYMMDD.

Two traps this module handles for you:

* Errors come back under a *different* top-level key: `{"RESULT": {...}}`
  instead of `{"StatisticSearch": {...}}`. Silently indexing the happy path
  turns a bad key into a confusing KeyError.
* `startRow/endRow` pagination is mandatory, not optional. We read
  `list_total_count` from the first page and walk the rest.

`ECOS_API_KEY` defaults to the literal `sample` key, which works without
registration but returns at most 10 rows - fine for a smoke test, useless for
real work. Register at https://ecos.bok.or.kr/api/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from ..cache import fetch
from ..config import SETTINGS
from .base import SourceMeta, tidy

META = SourceMeta(
    source_id="ecos",
    name="Bank of Korea ECOS",
    citation="Bank of Korea, Economic Statistics System (ECOS)",
    homepage="https://ecos.bok.or.kr/",
    licence="Free reuse with attribution to the Bank of Korea.",
    notes=("Statistical revisions reorganise STAT/ITEM codes; resolve them at "
           "runtime via StatisticItemList rather than trusting a hard-coded map "
           "indefinitely.",),
)

BASE = "https://ecos.bok.or.kr/api"
PAGE = 1000


@dataclass(frozen=True)
class EcosSeries:
    alias: str
    stat_code: str
    item_code: str
    cycle: str
    label: str


CURATED: dict[str, EcosSeries] = {
    "base_rate": EcosSeries("base_rate", "722Y001", "0101000", "D",
                            "BOK base rate"),
    "ktb_3y": EcosSeries("ktb_3y", "817Y002", "010200000", "D",
                         "Korea Treasury Bond 3Y"),
    "ktb_10y": EcosSeries("ktb_10y", "817Y002", "010210000", "D",
                          "Korea Treasury Bond 10Y"),
    "cd_91d": EcosSeries("cd_91d", "817Y002", "010502000", "D", "CD 91-day"),
    "call_rate": EcosSeries("call_rate", "817Y002", "010101000", "D",
                            "Call rate (overnight, all)"),
    "corp_aa_3y": EcosSeries("corp_aa_3y", "817Y002", "010300000", "D",
                             "Corporate bond 3Y AA-"),
    "corp_bbb_3y": EcosSeries("corp_bbb_3y", "817Y002", "010320000", "D",
                              "Corporate bond 3Y BBB-"),
    "usdkrw": EcosSeries("usdkrw", "731Y001", "0000001", "D",
                         "KRW/USD (mid rate)"),
    "cpi": EcosSeries("cpi", "901Y009", "0", "M", "CPI, all items (2020=100)"),
}

_CYCLE_FMT = {"A": "%Y", "Q": "%Y", "M": "%Y%m", "SM": "%Y%m", "D": "%Y%m%d"}


def _fmt(date: str, cycle: str) -> str:
    ts = pd.Timestamp(date)
    if cycle == "Q":
        return f"{ts.year}Q{ts.quarter}"
    return ts.strftime(_CYCLE_FMT.get(cycle, "%Y%m%d"))


def _parse_time(values: pd.Series, cycle: str) -> pd.Series:
    if cycle == "D":
        return pd.to_datetime(values, format="%Y%m%d", errors="coerce")
    if cycle in ("M", "SM"):
        return pd.to_datetime(values.str.slice(0, 6), format="%Y%m",
                              errors="coerce")
    if cycle == "Q":
        return pd.PeriodIndex(values.str.replace("Q", "Q"),
                              freq="Q").to_timestamp().to_series(index=values.index)
    return pd.to_datetime(values, format="%Y", errors="coerce")


def _request(spec: EcosSeries, start: str, end: str, row_from: int,
             row_to: int, force: bool) -> dict:
    url = "/".join([
        BASE, "StatisticSearch", SETTINGS.ecos_api_key, "json", "kr",
        str(row_from), str(row_to), spec.stat_code, spec.cycle,
        _fmt(start, spec.cycle), _fmt(end, spec.cycle), spec.item_code,
    ])
    raw = fetch(url, source="ecos", force=force,
                note=f"{spec.stat_code}/{spec.item_code}/{spec.cycle}")
    payload = json.loads(raw.decode("utf-8"))
    if "RESULT" in payload:
        res = payload["RESULT"]
        raise RuntimeError(
            f"ECOS error {res.get('CODE')}: {res.get('MESSAGE')} "
            f"(stat={spec.stat_code} item={spec.item_code}). "
            "If CODE is INFO-100/200 check your ECOS_API_KEY.")
    if "StatisticSearch" not in payload:
        raise RuntimeError(f"Unexpected ECOS payload keys: {list(payload)}")
    return payload["StatisticSearch"]


def get(
    series: list[str] | str,
    start: str = "2000-01-01",
    end: str | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch curated ECOS aliases (see CURATED) and join them on date."""
    if isinstance(series, str):
        series = [series]
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    frames: list[pd.DataFrame] = []
    labels: dict[str, str] = {}
    for name in series:
        if name not in CURATED:
            raise KeyError(
                f"{name!r} is not a curated ECOS series. Known: "
                f"{sorted(CURATED)}. For anything else build an EcosSeries and "
                "call get_raw().")
        frames.append(_fetch_all(CURATED[name], start, end, force))
        labels[name] = CURATED[name].label

    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return tidy(out, META, extra={"labels": labels})


def get_raw(spec: EcosSeries, start: str, end: str, *,
            force: bool = False) -> pd.DataFrame:
    """Escape hatch for arbitrary STAT_CODE/ITEM_CODE pairs."""
    return tidy(_fetch_all(spec, start, end, force), META,
                extra={"labels": {spec.alias: spec.label}})


def _fetch_all(spec: EcosSeries, start: str, end: str,
               force: bool) -> pd.DataFrame:
    first = _request(spec, start, end, 1, PAGE, force)
    rows = list(first.get("row", []))
    total = int(first.get("list_total_count", len(rows)) or len(rows))
    cursor = PAGE + 1
    while cursor <= total:
        page = _request(spec, start, end, cursor, cursor + PAGE - 1, force)
        rows.extend(page.get("row", []))
        cursor += PAGE
    if not rows:
        raise RuntimeError(f"ECOS returned no rows for {spec.alias}")

    df = pd.DataFrame(rows)
    df["_t"] = _parse_time(df["TIME"].astype(str), spec.cycle)
    df = df.dropna(subset=["_t"])
    out = (df.set_index("_t")[["DATA_VALUE"]]
             .rename(columns={"DATA_VALUE": spec.alias}))
    return out
