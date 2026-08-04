"""US mortgage disclosure data (HMDA) via the CFPB Data Browser API. No key.

Two endpoint families:

* `/view/csv` and `/view/nationwide/csv` — loan-level records, **streamed
  unbounded with no pagination**. A state-year query is hundreds of megabytes,
  so `to_csv()` streams to disk and refuses to hand you a DataFrame directly.
* `/view/aggregations` — server-side counts and sums, cached, tiny responses.
  Start here. Most of what a blog post needs is an aggregation, and pulling
  400MB to compute a group-by is how you get your IP throttled.

Non-nationwide requests must include one of states / msamds / counties / leis.
Years currently available: 2018-2025 (confirm 2025 is final before treating it
as such — HMDA goes through snapshot then modified-LAR revisions).

Note the public LAR field names contain hyphens (`derived_msa-md`, `aus-1`,
`co-applicant_race-3`), so never assume snake_case when selecting columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

from ..cache import fetch
from ..config import SETTINGS
from .base import SourceMeta

META = SourceMeta(
    source_id="hmda",
    name="HMDA Data Browser (CFPB/FFIEC)",
    citation="Consumer Financial Protection Bureau, HMDA Data Browser",
    homepage="https://ffiec.cfpb.gov/data-browser/",
    licence="US public domain.",
    notes=("CSV endpoints stream without pagination; always filter or stream to "
           "disk.",),
)

BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view"


def _params(years: list[int] | int, **filters) -> dict[str, str]:
    if isinstance(years, int):
        years = [years]
    params: dict[str, str] = {"years": ",".join(str(y) for y in years)}
    for k, v in filters.items():
        if v is None:
            continue
        params[k] = ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
    return params


def aggregations(
    years: list[int] | int,
    *,
    nationwide: bool = False,
    force: bool = False,
    **filters,
) -> pd.DataFrame:
    """Server-side counts/sums. Cheap; start every analysis here."""
    params = _params(years, **filters)
    path = "nationwide/aggregations" if nationwide else "aggregations"
    if not nationwide and not ({"states", "msamds", "counties", "leis"}
                               & set(params)):
        raise ValueError(
            "Non-nationwide HMDA requests require one of states / msamds / "
            "counties / leis. Pass nationwide=True for the national view.")
    url = f"{BASE}/{path}?{urlencode(params)}"
    raw = fetch(url, source="hmda", force=force, ttl_hours=24 * 30,
                note=str(params))
    payload = json.loads(raw.decode("utf-8"))
    df = pd.DataFrame(payload.get("aggregations", []))
    df.attrs["quantpost"] = {
        "source_id": META.source_id, "citation": META.citation,
        "homepage": META.homepage, "licence": META.licence,
        "parameters": payload.get("parameters", {}),
        "served_from": payload.get("servedFrom"),
    }
    return df


def to_csv(
    dest: str | Path,
    years: list[int] | int,
    *,
    nationwide: bool = False,
    chunk_bytes: int = 1 << 20,
    **filters,
) -> Path:
    """Stream loan-level records to `dest`. Returns the path.

    Deliberately does not return a DataFrame: these responses are large enough
    that materialising one is usually a mistake. Read the file back with
    `pd.read_csv(..., usecols=[...], chunksize=...)`.
    """
    params = _params(years, **filters)
    path = "nationwide/csv" if nationwide else "csv"
    url = f"{BASE}/{path}?{urlencode(params)}"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=SETTINGS.timeout,
                      headers={"User-Agent": SETTINGS.user_agent}) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_bytes):
                fh.write(chunk)
    return dest
