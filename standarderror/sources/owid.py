"""Our World in Data grapher API. No key.

    https://ourworldindata.org/grapher/{slug}.csv
    https://ourworldindata.org/grapher/{slug}.metadata.json

Any OWID chart URL becomes a data endpoint by appending `.csv`. Confirmed
parameters:

* `csvType=full` (default) — every entity and every time point.
  `csvType=filtered` — only what the chart currently shows, honouring `country`
  and `time`.
* `useColumnShortNames=true` — machine-readable column names
  (`life_expectancy_0__sex_all__age_0`) instead of the default prose
  (`Period life expectancy at birth - Sex: all - Age: 0`). **Always pass this.**
  The prose names contain spaces, colons and hyphens, change when OWID edits a
  chart's metadata, and will break your code silently.
* `country=USA~KOR` — tilde-separated. `~OWID_WRL` is the World aggregate.
* `time=2000..2020`, or `latest` / `earliest`.

The time column is `Year` (integer) for annual data and `Day` (`YYYY-MM-DD`) for
daily. Which one you get depends on the chart, so this module detects it rather
than assuming.

`Entity` is a *name* and `Code` is an OWID code that matches ISO alpha-3 for
ordinary countries but not for aggregates (`OWID_WRL`, `OWID_EUR`). Filtering a
country panel on `Code`'s length is how continent and income-group rows sneak
into a country average and quietly double-count half the world — `drop_aggregates`
handles it explicitly.

Licence: OWID's own work is CC BY, but each chart re-publishes **upstream** data
with its own terms. `metadata()` returns the per-column citation; print it.
"""

from __future__ import annotations

import io
import json
from urllib.parse import urlencode

import pandas as pd

from ..cache import fetch
from .base import SourceMeta

META = SourceMeta(
    source_id="owid",
    name="Our World in Data",
    citation="Our World in Data",
    homepage="https://ourworldindata.org/",
    licence=("OWID's own charts and code are CC BY. Underlying data carries the "
             "original provider's licence — cite the per-column source from "
             "metadata()."),
    redistributable=True,
    notes=("Always pass useColumnShortNames=true; the default prose column names "
           "change when a chart's metadata is edited.",
           "Aggregate rows (OWID_WRL, continents, income groups) share the table "
           "with countries — drop them before averaging.",),
)

BASE = "https://ourworldindata.org/grapher"

# Aggregates that are not countries. OWID_-prefixed codes cover most of it; the
# bare continent names appear with an empty Code in some charts.
AGGREGATE_CODES = {"OWID_WRL", "OWID_EUR", "OWID_AFR", "OWID_ASI", "OWID_NAM",
                   "OWID_SAM", "OWID_OCE", "OWID_KOS", "OWID_EU27"}
AGGREGATE_NAMES = {"World", "Africa", "Asia", "Europe", "European Union (27)",
                   "North America", "South America", "Oceania",
                   "High-income countries", "Low-income countries",
                   "Lower-middle-income countries",
                   "Upper-middle-income countries"}

CURATED: dict[str, str] = {
    "life_expectancy": "life-expectancy",
    "fertility_rate": "children-per-woman-un",
    "median_age": "median-age",
    "population_growth": "population-growth-rates",
    "co2_per_capita": "co-emissions-per-capita",
    "co2_by_source": "co2-by-source",
    "energy_per_capita": "per-capita-energy-use",
    "share_electricity_renewables": "share-electricity-renewables",
    "temperature_anomaly": "temperature-anomaly",
    "sea_level": "sea-level",
    "internet_users": "share-of-individuals-using-the-internet",
    "gdp_per_capita": "gdp-per-capita-worldbank",
    "electricity_prices": "electricity-prices",
}


def get(
    slug: str,
    *,
    countries: list[str] | str | None = None,
    time: str | None = None,
    full: bool = True,
    short_names: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch a grapher chart's data as a tidy long frame.

    Adds a `date` column (from `Year` or `Day`, whichever the chart uses) so
    downstream code does not have to branch on it.
    """
    s = CURATED.get(slug, slug)
    params: dict[str, str] = {
        "csvType": "full" if full and not (countries or time) else "filtered",
        "useColumnShortNames": "true" if short_names else "false",
    }
    if countries:
        params["country"] = ("~".join(countries)
                            if isinstance(countries, list) else countries)
    if time:
        params["time"] = time
    url = f"{BASE}/{s}.csv?{urlencode(params, safe='~.')}"
    raw = fetch(url, source="owid", force=force, ttl_hours=24 * 7, note=s)

    df = pd.read_csv(io.BytesIO(raw))
    if "Entity" not in df.columns:
        raise RuntimeError(
            f"OWID returned no usable CSV for slug {s!r} (columns "
            f"{list(df.columns)[:6]}). Check the slug against the chart URL.")

    time_col = "Year" if "Year" in df.columns else (
        "Day" if "Day" in df.columns else None)
    if time_col is None:
        raise RuntimeError(f"no Year or Day column in {s!r}: {list(df.columns)}")
    df["date"] = (pd.to_datetime(df["Year"].astype(int).astype(str) + "-12-31")
                  if time_col == "Year" else pd.to_datetime(df["Day"]))

    value_cols = [c for c in df.columns
                  if c not in ("Entity", "Code", "Year", "Day", "date")]
    df.attrs["standarderror"] = {
        "source_id": META.source_id, "citation": META.citation,
        "homepage": META.homepage, "licence": META.licence,
        "redistributable": True, "notes": list(META.notes),
        "slug": s, "time_column": time_col, "value_columns": value_cols,
        "chart_url": f"https://ourworldindata.org/grapher/{s}",
    }
    return df


def drop_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove World, continent and income-group rows.

    Not optional. These rows sit in the same table as countries, so any
    `groupby(...).mean()` over the raw frame averages the World in with Togo.
    """
    keep = ~(df["Entity"].isin(AGGREGATE_NAMES)
             | df.get("Code", pd.Series(index=df.index, dtype=object))
                 .isin(AGGREGATE_CODES))
    out = df.loc[keep].copy()
    out.attrs = dict(df.attrs)
    return out


def wide(df: pd.DataFrame, value: str | None = None) -> pd.DataFrame:
    """Pivot to date x entity for one value column, ready to chart."""
    cols = df.attrs.get("standarderror", {}).get("value_columns") or []
    value = value or (cols[0] if cols else None)
    if value is None or value not in df.columns:
        raise ValueError(
            f"pick a value column explicitly; available: {cols or list(df.columns)}")
    out = df.pivot_table(index="date", columns="Entity", values=value,
                         aggfunc="last").sort_index()
    out.attrs = dict(df.attrs)
    return out


def metadata(slug: str, *, force: bool = False) -> dict:
    """Chart config plus per-column units, timespan and **citations**.

    Call this before publishing. OWID re-publishes upstream data, so the citation
    a post owes is the per-column one here, not simply "Our World in Data".
    """
    s = CURATED.get(slug, slug)
    raw = fetch(f"{BASE}/{s}.metadata.json", source="owid-meta", force=force,
                ttl_hours=24 * 7, note=s)
    return json.loads(raw.decode("utf-8"))


def citations(slug: str, *, force: bool = False) -> list[str]:
    """Flatten the metadata into attribution lines for a post's Data footer."""
    md = metadata(slug, force=force)
    out: list[str] = []
    chart = md.get("chart") or {}
    if chart.get("citation"):
        out.append(str(chart["citation"]))
    for name, col in (md.get("columns") or {}).items():
        cite = col.get("citationShort") or col.get("attribution") or \
            col.get("citationLong")
        if cite:
            line = f"{name}: {cite}"
            if line not in out:
                out.append(line)
    if not out:
        out.append(f"Our World in Data, chart '{slug}' — "
                   f"<https://ourworldindata.org/grapher/{slug}>")
    return out
