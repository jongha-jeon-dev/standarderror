"""ECB Data Portal SDMX REST API. No key, no registration.

Host note: `sdw-wsrest.ecb.europa.eu` is decommissioned (no longer resolves).
The live host is `data-api.ecb.europa.eu`.

URL grammar: `/service/data/{DATAFLOW}/{KEY}` where KEY **omits** the dataflow
prefix. Portal id `YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y` becomes
`/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y`. `split_series_id` does this for you
so you can paste ids straight off the portal.

SDMX-CSV is ~40 columns wide and the column set differs per dataflow; only
`TIME_PERIOD`, `OBS_VALUE` and `KEY` are dependable. Parse by name, never by
position.
"""

from __future__ import annotations

import io
from urllib.parse import urlencode

import pandas as pd

from ..cache import fetch
from .base import SourceMeta, tidy

META = SourceMeta(
    source_id="ecb",
    name="ECB Data Portal",
    citation="European Central Bank, ECB Data Portal",
    homepage="https://data.ecb.europa.eu/",
    licence="Free reuse with attribution to the ECB.",
    notes=("No rate limit documented; be polite.",),
)

BASE = "https://data-api.ecb.europa.eu/service/data"

CURATED: dict[str, str] = {
    # AAA-rated euro-area government yield curve, 10y spot, business-daily.
    "ea_aaa_10y": "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
    "ea_aaa_2y": "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
    "ea_aaa_1y": "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y",
    # All euro-area central government bonds (not just AAA).
    "ea_all_10y": "YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y",
    # Composite Indicator of Systemic Stress.
    "ciss_ea": "CISS.D.U2.Z0Z.4F.EC.SS_CI.IDX",
    "ciss_us": "CISS.D.US.Z0Z.4F.EC.SS_CI.IDX",
    "ciss_de": "CISS.D.DE.Z0Z.4F.EC.SS_CI.IDX",
}


def split_series_id(series_id: str) -> tuple[str, str]:
    """`YC.B.U2...` -> ("YC", "B.U2...")."""
    dataflow, _, key = series_id.partition(".")
    if not key:
        raise ValueError(
            f"{series_id!r} does not look like a full SDMX series id "
            "(expected DATAFLOW.dim1.dim2...)")
    return dataflow, key


def get(
    series: list[str] | str,
    start: str | None = None,
    end: str | None = None,
    *,
    last_n: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch one or more ECB series. Names may be curated aliases or full ids.

    Series from different dataflows are fetched separately and joined, since one
    request can only address one dataflow.
    """
    if isinstance(series, str):
        series = [series]
    resolved = [CURATED.get(s, s) for s in series]

    by_flow: dict[str, list[tuple[str, str]]] = {}
    for alias, sid in zip(series, resolved):
        flow, key = split_series_id(sid)
        by_flow.setdefault(flow, []).append((alias, key))

    frames: list[pd.DataFrame] = []
    for flow, items in by_flow.items():
        for alias, key in items:
            params: dict[str, str] = {"format": "csvdata"}
            if start:
                params["startPeriod"] = str(start)
            if end:
                params["endPeriod"] = str(end)
            if last_n:
                params["lastNObservations"] = str(last_n)
            url = f"{BASE}/{flow}/{key}?{urlencode(params)}"
            raw = fetch(url, source="ecb", force=force, note=f"{flow}.{key}")
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
            missing = {"TIME_PERIOD", "OBS_VALUE"} - set(df.columns)
            if missing:
                raise RuntimeError(
                    f"ECB response for {flow}.{key} lacks {missing}; "
                    f"got columns {list(df.columns)[:8]}...")
            sub = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
            sub = sub.rename(columns={"OBS_VALUE": alias})
            sub = sub.set_index("TIME_PERIOD")
            frames.append(sub)

    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return tidy(out, META, extra={"series": resolved})
