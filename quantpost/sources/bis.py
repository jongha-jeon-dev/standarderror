"""BIS Data Portal — SDMX REST v2, no key.

    https://stats.bis.org/api/v2/data/dataflow/BIS/{FLOW}/{VERSION}/{KEY}?format=csv

Partial keys work (`Q.KR` matches every remaining dimension combination), which
is convenient and dangerous: one request can fan out to many series, so always
group by the dimension columns before plotting.

Two dataflows are curated here because they are the ones you actually want for
financial-stability writing:

* `WS_CREDIT_GAP` — credit-to-GDP gap. `CG_DTYPE` selects C = gap, B = ratio,
  A = HP trend. The gap (C) is the Basel III countercyclical-buffer reference.
* `WS_SPP` — selected residential property prices. `VALUE` selects R = real,
  N = nominal; `UNIT_MEASURE` 628 = index, 771 = % change.

Bulk CSVs live at https://data.bis.org/static/bulk/{FLOW}_csv_flat.zip
(`_flat` is one row per observation — use it; `_col` is wide with periods as
columns).
"""

from __future__ import annotations

import io
import zipfile
from urllib.parse import urlencode

import pandas as pd

from ..cache import fetch
from .base import SourceMeta, tidy

META = SourceMeta(
    source_id="bis",
    name="BIS Data Portal",
    citation="Bank for International Settlements, BIS Data Portal",
    homepage="https://data.bis.org/",
    licence="See https://data.bis.org/help/legal — attribution to BIS expected.",
)

API = "https://stats.bis.org/api/v2/data/dataflow/BIS"
BULK = "https://data.bis.org/static/bulk"


def get(
    flow: str,
    key: str,
    *,
    version: str = "1.0",
    start: str | None = None,
    end: str | None = None,
    last_n: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Raw SDMX-CSV fetch, returned long (one row per observation).

    Kept long rather than pivoted because partial keys legitimately return many
    series and the right pivot depends on which dimension you are comparing.
    """
    params: dict[str, str] = {"format": "csv"}
    if start:
        params["startPeriod"] = str(start)
    if end:
        params["endPeriod"] = str(end)
    if last_n:
        params["lastNObservations"] = str(last_n)
    url = f"{API}/{flow}/{version}/{key}?{urlencode(params)}"
    raw = fetch(url, source="bis", force=force, note=f"{flow}/{key}")
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        raise RuntimeError(
            f"BIS response for {flow}/{key} lacks TIME_PERIOD/OBS_VALUE; got "
            f"{list(df.columns)[:10]}")
    df["date"] = _period_to_timestamp(df["TIME_PERIOD"].astype(str))
    df.attrs["quantpost"] = {
        "source_id": META.source_id, "citation": META.citation,
        "homepage": META.homepage, "licence": META.licence,
        "dataflow": flow, "key": key,
    }
    return df


def credit_gap(
    countries: list[str] | str = "KR",
    *,
    dtype: str = "C",
    start: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Credit-to-GDP gap (percentage points of GDP), one column per country."""
    if isinstance(countries, str):
        countries = [countries]
    frames = []
    for c in countries:
        raw = get("WS_CREDIT_GAP", f"Q.{c}", start=start, force=force)
        sub = raw[(raw["CG_DTYPE"] == dtype)
                  & (raw.get("TC_BORROWERS", "P") == "P")]
        if sub.empty:
            raise RuntimeError(
                f"No WS_CREDIT_GAP rows for {c} with CG_DTYPE={dtype}")
        frames.append(sub.set_index("date")[["OBS_VALUE"]]
                      .rename(columns={"OBS_VALUE": c}))
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return tidy(out, META, extra={"dataflow": "WS_CREDIT_GAP",
                                 "cg_dtype": dtype})


def property_prices(
    countries: list[str] | str = "KR",
    *,
    real: bool = True,
    start: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Selected residential property price index (628), real or nominal."""
    if isinstance(countries, str):
        countries = [countries]
    want = "R" if real else "N"
    frames = []
    for c in countries:
        raw = get("WS_SPP", f"Q.{c}", start=start, force=force)
        sub = raw[(raw["VALUE"] == want)
                  & (raw["UNIT_MEASURE"].astype(str) == "628")]
        if sub.empty:
            raise RuntimeError(f"No WS_SPP index rows for {c} (VALUE={want})")
        frames.append(sub.set_index("date")[["OBS_VALUE"]]
                      .rename(columns={"OBS_VALUE": c}))
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return tidy(out, META, extra={"dataflow": "WS_SPP",
                                 "measure": "real" if real else "nominal"})


def bulk(flow: str, *, flat: bool = True, force: bool = False) -> pd.DataFrame:
    """Download and read a bulk CSV zip. Use for whole-dataflow analysis."""
    suffix = "flat" if flat else "col"
    url = f"{BULK}/{flow}_csv_{suffix}.zip"
    raw = fetch(url, source="bis-bulk", force=force, ttl_hours=24 * 7,
                note=f"bulk {flow} {suffix}")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as fh:
            return pd.read_csv(fh, low_memory=False)


def _period_to_timestamp(periods: pd.Series) -> pd.Series:
    """BIS periods are `2025-Q4`, `2026-01`, or `2026`."""
    out = pd.Series(pd.NaT, index=periods.index, dtype="datetime64[ns]")
    q = periods.str.contains("-Q", na=False)
    if q.any():
        out.loc[q] = pd.PeriodIndex(
            periods[q].str.replace("-Q", "Q"), freq="Q").to_timestamp(how="end").normalize()
    rest = ~q
    if rest.any():
        out.loc[rest] = pd.to_datetime(periods[rest], errors="coerce")
    return out
