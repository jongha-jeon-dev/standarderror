"""FRED (Federal Reserve Bank of St. Louis).

Two access paths, deliberately:

* `fredgraph.csv` — no API key, multi-series, good enough for blog work.
  The first column header is `observation_date` in the current form (it used to
  be `DATE`), so we parse column 0 **positionally** and never by name.
  `cosd`/`coed` date params are what the FRED graph UI emits; they are NOT in
  the official API docs, so treat them as best-effort and always re-slice the
  frame client-side.
* the official `api.stlouisfed.org` API — needs a free key, is documented, and
  supports vintages (`realtime_start`) which is what you want the moment you
  care about "what did the data look like at the time".

Attribution is mandatory: see `MANDATORY_DISCLAIMER`.
"""

from __future__ import annotations

import io
import json
from urllib.parse import urlencode

import pandas as pd

from ..cache import fetch
from ..config import SETTINGS
from .base import SourceMeta, tidy

MANDATORY_DISCLAIMER = (
    "This product uses the FRED® API but is not endorsed or certified by "
    "the Federal Reserve Bank of St. Louis."
)

META = SourceMeta(
    source_id="fred",
    name="FRED",
    citation="Federal Reserve Bank of St. Louis, FRED",
    homepage="https://fred.stlouisfed.org/",
    licence=MANDATORY_DISCLAIMER,
    notes=(
        "ICE BofA OAS series (BAMLH0A0HYM2, BAMLC0A4CBBB) were truncated to a "
        "rolling 3-year window in April 2026 and are internal-use only on FRED "
        "- do not redistribute those values.",
        "No numeric rate limit is published; back off on HTTP 429.",
    ),
)

# Series worth knowing by name. Not exhaustive - a convenience map so posts read
# `CURATED["term_spread_10y2y"]` instead of a bare mnemonic.
CURATED: dict[str, str] = {
    "ust_10y": "DGS10",
    "ust_2y": "DGS2",
    "tbill_3m": "DTB3",
    "term_spread_10y2y": "T10Y2Y",
    "term_spread_10y3m": "T10Y3M",
    "vix": "VIXCLS",
    "nfci": "NFCI",                       # Chicago Fed financial conditions, weekly
    "stlfsi": "STLFSI4",                  # STLFSI/2/3 are discontinued
    "hy_oas": "BAMLH0A0HYM2",             # 3-year rolling window only, see notes
    "bbb_oas": "BAMLC0A4CBBB",            # 3-year rolling window only, see notes
    "unrate": "UNRATE",
    "cpi": "CPIAUCSL",
    "indpro": "INDPRO",
    "real_gdp": "GDPC1",
    "recession_nber": "USREC",
    "recession_prob": "RECPROUSM156N",
    "sp500": "SP500",
}

TRUNCATED_SERIES = {"BAMLH0A0HYM2", "BAMLC0A4CBBB"}

_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_API_URL = "https://api.stlouisfed.org/fred/series/observations"


def resolve(names: list[str] | str) -> list[str]:
    """Map curated aliases to FRED mnemonics; pass unknown strings through."""
    if isinstance(names, str):
        names = [names]
    return [CURATED.get(n, n) for n in names]


def get(
    series: list[str] | str,
    start: str | None = None,
    end: str | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Keyless multi-series fetch via fredgraph.csv."""
    ids = resolve(series)
    params: dict[str, str] = {"id": ",".join(ids)}
    if start:
        params["cosd"] = str(start)
    if end:
        params["coed"] = str(end)
    url = f"{_GRAPH_URL}?{urlencode(params, safe=',')}"
    raw = fetch(url, source="fred", force=force,
                note=f"fredgraph.csv {','.join(ids)}")

    # Column 0 is the date whatever it is called this year.
    df = pd.read_csv(io.BytesIO(raw), na_values=["."], keep_default_na=True)
    date_col = df.columns[0]
    df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]

    # cosd/coed are undocumented; enforce the window ourselves.
    if start:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end:
        df = df.loc[df.index <= pd.Timestamp(end)]

    notes = list(META.notes)
    flagged = sorted(TRUNCATED_SERIES.intersection(df.columns))
    if flagged:
        notes.append(
            "WARNING: " + ", ".join(flagged) +
            " carry only a rolling 3-year history on FRED since April 2026 and "
            "are licensed for internal use only.")
    return tidy(df, META, extra={"series": ids, "notes": notes,
                                 "endpoint": "fredgraph.csv"})


def get_api(
    series_id: str,
    start: str | None = None,
    end: str | None = None,
    *,
    units: str = "lin",
    frequency: str | None = None,
    aggregation_method: str = "avg",
    realtime_start: str | None = None,
    realtime_end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Official API. Requires FRED_API_KEY.

    `realtime_start`/`realtime_end` give you the *vintage* - the data as it was
    published on that date. Any honest backtest of a macro signal needs this;
    revisions to CPI and payrolls are large enough to manufacture skill that
    never existed in real time.
    """
    if not SETTINGS.fred_api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Get one free at "
            "https://fredaccount.stlouisfed.org/apikeys, or use fred.get() "
            "which needs no key.")
    sid = resolve(series_id)[0]
    params = {
        "series_id": sid,
        "api_key": SETTINGS.fred_api_key,
        "file_type": "json",
        "units": units,
        "aggregation_method": aggregation_method,
        "limit": "100000",
    }
    if start:
        params["observation_start"] = str(start)
    if end:
        params["observation_end"] = str(end)
    if frequency:
        params["frequency"] = frequency
    if realtime_start:
        params["realtime_start"] = str(realtime_start)
    if realtime_end:
        params["realtime_end"] = str(realtime_end)

    url = f"{_API_URL}?{urlencode(params)}"
    raw = fetch(url, source="fred-api", force=force, note=f"api {sid}")
    payload = json.loads(raw.decode("utf-8"))
    obs = payload.get("observations", [])
    if not obs:
        raise RuntimeError(f"FRED API returned no observations for {sid}")
    df = pd.DataFrame(obs)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    df = df.rename(columns={"value": sid}).set_index("date")
    return tidy(df, META, extra={"series": [sid], "endpoint": "api",
                                 "vintage": realtime_start})
