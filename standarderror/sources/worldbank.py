"""World Bank Indicators API v2. No key, no registration.

    https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}?format=json

Grammar confirmed against the World Bank API docs:

* Multiple countries are joined with **semicolons**, not commas:
  `country/chn;ago/indicator/SI.POV.DDAY`. Commas silently return nothing useful.
* `date=2000:2024` for a range (colon, not a dash).
* `format=json` is required — **the default response is XML**, which is the single
  most common way a first attempt at this API fails.
* `per_page` defaults to **50**. Fifty. A country-year panel will be truncated
  without a word unless you raise it, and the truncation looks exactly like
  "the World Bank has no data for those years".

The JSON response is a two-element array: `[metadata, rows]`. Metadata carries
`page`, `pages`, `per_page`, `total`, and we walk `pages`. Errors come back as a
*one*-element array `[{"message": [...]}]`, so indexing `[1]` on the happy path
turns a bad indicator code into an IndexError instead of a readable message.

Attribution: World Bank data is CC BY 4.0 — free to redistribute *with credit*,
which makes it a much better fit for a public blog than most financial data.
"""

from __future__ import annotations

import json
from urllib.parse import urlencode

import pandas as pd

from ..cache import fetch
from .base import SourceMeta, tidy

META = SourceMeta(
    source_id="worldbank",
    name="World Bank Open Data",
    citation="World Bank, World Development Indicators",
    homepage="https://data.worldbank.org/",
    licence="CC BY 4.0 — redistribution permitted with attribution.",
    redistributable=True,
    notes=("per_page defaults to 50; always paginate or set it high.",
           "Indicator coverage is uneven across countries and years — check for "
           "NaN before comparing country panels.",),
)

BASE = "https://api.worldbank.org/v2"
PER_PAGE = 20000

CURATED: dict[str, str] = {
    # demography
    "population": "SP.POP.TOTL",
    "fertility_rate": "SP.DYN.TFRT.IN",
    "life_expectancy": "SP.DYN.LE00.IN",
    "old_age_dependency": "SP.POP.DPND.OL",
    "urban_share": "SP.URB.TOTL.IN.ZS",
    # macro
    "gdp_usd": "NY.GDP.MKTP.CD",
    "gdp_per_capita_ppp": "NY.GDP.PCAP.PP.KD",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "inflation_cpi": "FP.CPI.TOTL.ZG",
    "unemployment": "SL.UEM.TOTL.ZS",
    "gini": "SI.POV.GINI",
    # finance / credit — the bridge to the risk posts
    "domestic_credit_private_pct_gdp": "FS.AST.PRVT.GD.ZS",
    "bank_npl_ratio": "FB.AST.NPER.ZS",
    "bank_capital_to_assets": "FB.BNK.CAPA.ZS",
    "broad_money_pct_gdp": "FM.LBL.BMNY.GD.ZS",
    "real_interest_rate": "FR.INR.RINR",
    # energy / climate
    "co2_per_capita": "EN.GHG.CO2.PC.CE.AR5",
    "energy_use_per_capita": "EG.USE.PCAP.KG.OE",
    "renewable_share": "EG.FEC.RNEW.ZS",
    "electric_power_consumption": "EG.USE.ELEC.KH.PC",
    # R&D — useful for the "where does research money go" style post
    "rd_spend_pct_gdp": "GB.XPD.RSDV.GD.ZS",
    "researchers_per_million": "SP.POP.SCIE.RD.P6",
}


def get(
    indicator: str,
    countries: list[str] | str = "all",
    *,
    start: int | None = 1990,
    end: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Long frame: one row per (country, year).

    Returned long rather than pivoted because a country panel is the normal shape
    here and the right pivot depends on what you are comparing. Use
    `.pivot(index="year", columns="country", values="value")` when you want wide.
    """
    code = CURATED.get(indicator, indicator)
    iso = ";".join(countries) if isinstance(countries, list) else countries

    rows: list[dict] = []
    page = 1
    pages = 1
    meta_seen: dict = {}
    while page <= pages:
        params = {"format": "json", "per_page": str(PER_PAGE), "page": str(page)}
        if start or end:
            params["date"] = f"{start or 1960}:{end or pd.Timestamp.today().year}"
        url = f"{BASE}/country/{iso}/indicator/{code}?{urlencode(params)}"
        raw = fetch(url, source="worldbank", force=force,
                    ttl_hours=24 * 7, note=f"{code} {iso} p{page}")
        payload = json.loads(raw.decode("utf-8"))

        if not isinstance(payload, list) or len(payload) < 2:
            msg = None
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                msg = payload[0].get("message")
            raise RuntimeError(
                f"World Bank API returned no data block for indicator {code!r}, "
                f"countries {iso!r}: {msg or payload}")

        meta_seen = payload[0] or {}
        pages = int(meta_seen.get("pages") or 1)
        rows.extend(payload[1] or [])
        page += 1

    if not rows:
        raise RuntimeError(f"no observations for {code} / {iso}")

    df = pd.DataFrame([{
        "year": int(r["date"]),
        "country": (r.get("country") or {}).get("value"),
        "iso3": r.get("countryiso3code") or None,
        "value": r.get("value"),
        "indicator": (r.get("indicator") or {}).get("value"),
    } for r in rows if r.get("date")])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    df.attrs["standarderror"] = {
        "source_id": META.source_id, "citation": META.citation,
        "homepage": META.homepage, "licence": META.licence,
        "redistributable": True, "notes": list(META.notes),
        "indicator_code": code, "n_total_reported": meta_seen.get("total"),
    }
    return df


def panel(
    indicators: dict[str, str] | list[str],
    countries: list[str] | str,
    *,
    start: int | None = 1990,
    end: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Several indicators for the same countries, joined on (country, year).

    `indicators` may be a list of curated aliases/codes, or a mapping of
    output-column-name -> alias/code.
    """
    if isinstance(indicators, list):
        indicators = {i: i for i in indicators}
    out: pd.DataFrame | None = None
    for name, ind in indicators.items():
        d = get(ind, countries, start=start, end=end, force=force)
        d = (d[["country", "iso3", "year", "value"]]
             .rename(columns={"value": name}))
        out = d if out is None else out.merge(
            d, on=["country", "iso3", "year"], how="outer")
    assert out is not None
    out = out.sort_values(["country", "year"]).reset_index(drop=True)
    out.attrs["standarderror"] = {
        "source_id": META.source_id, "citation": META.citation,
        "homepage": META.homepage, "licence": META.licence,
        "redistributable": True, "indicators": indicators,
    }
    return out


def series(
    indicator: str,
    country: str,
    *,
    start: int | None = 1990,
    end: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """One country, one indicator, as a DatetimeIndex-ed frame — so the result
    drops straight into the same charts and models as the financial series."""
    d = get(indicator, country, start=start, end=end, force=force)
    name = CURATED.get(indicator, indicator)
    f = d[["year", "value"]].dropna().set_index("year")
    f.index = pd.to_datetime(f.index.astype(int).astype(str) + "-12-31")
    return tidy(f.rename(columns={"value": name}), META,
                extra={"country": country, "indicator_code": name})
