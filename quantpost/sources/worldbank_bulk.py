"""Read the World Bank's per-indicator bulk CSV/zip download.

The Indicators API and this bulk file are the same data by different doors. The
API is the better door when the network allows it; this one exists because the
zip a person downloads by hand is sometimes the only door available, and because
the file has three specific traps that are easier to solve once here than to
rediscover in every experiment.

The traps
---------

1. **Four junk lines before the header.** The data CSV opens with a blank line,
   a two-cell "Data Source" row, a "Last Updated Date" row and another blank,
   and only then the real header. `pd.read_csv` on the raw file therefore infers
   a two-column frame and reports the whole thing as malformed.

2. **Years are columns, not rows.** The file is wide: `1960` through the current
   year each get their own column, all of them strings. Anything downstream
   wants long.

3. **Aggregates share the table with countries.** `World`, `European Union`,
   `Low income`, `IDA & IBRD total` and about forty more sit in the same
   `Country Name` column as Togo. Averaging over the raw frame averages the
   World in with its own members. The country-metadata CSV inside the same zip
   is what separates them: real countries carry a `Region`, aggregates do not.
   That is the only reliable discriminator — the name list changes between
   releases, and hard-coding it silently rots.

Licence: CC BY 4.0. Values may be redistributed with attribution, which is why
this source is worth the trouble.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from .base import SourceMeta

META = SourceMeta(
    source_id="worldbank_bulk",
    name="World Bank Open Data (bulk indicator download)",
    citation="World Bank, World Development Indicators",
    homepage="https://data.worldbank.org/",
    licence="CC BY 4.0 — redistribution permitted with attribution.",
    redistributable=True,
    notes=("The data CSV carries four preamble lines before its header.",
           "Aggregate rows (World, income groups, regions) are distinguished "
           "from countries only by having no Region in the metadata file.",),
)

_DATA_RE = re.compile(r"^API_.*_DS2_.*_csv_v2_\d+\.csv$", re.I)
_META_RE = re.compile(r"^Metadata_Country_.*\.csv$", re.I)
PREAMBLE_LINES = 4


def _read_member(raw: bytes, skiprows: int) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), skiprows=skiprows)


def _pick(names: list[str], pattern: re.Pattern) -> str | None:
    for n in names:
        if pattern.match(Path(n).name):
            return n
    return None


def read_zip(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return `(wide_data, country_metadata_or_None)` from a downloaded zip."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        data_name = _pick(names, _DATA_RE)
        if data_name is None:
            # Fall back to "the biggest CSV that is not metadata", because the
            # exact filename pattern has changed across releases before.
            candidates = [n for n in names
                          if n.lower().endswith(".csv")
                          and not Path(n).name.lower().startswith("metadata")]
            if not candidates:
                raise ValueError(f"{path.name} contains no data CSV: {names}")
            data_name = max(candidates, key=lambda n: z.getinfo(n).file_size)
        wide = _read_member(z.read(data_name), PREAMBLE_LINES)
        meta_name = _pick(names, _META_RE)
        meta = _read_member(z.read(meta_name), 0) if meta_name else None
    return wide, meta


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a bare `API_*.csv` that has already been extracted."""
    return _read_member(Path(path).read_bytes(), PREAMBLE_LINES)


def _year_columns(wide: pd.DataFrame) -> list[str]:
    return [c for c in wide.columns if re.fullmatch(r"\d{4}", str(c).strip())]


def to_long(wide: pd.DataFrame, meta: pd.DataFrame | None = None, *,
            drop_aggregates: bool = True) -> pd.DataFrame:
    """Wide year-columns to a long `(iso3, country, year, value)` frame.

    `drop_aggregates=True` removes World, income groups and regions using the
    metadata file's `Region` column. Without a metadata file it cannot be done
    honestly, so the call raises rather than guessing from a name list — a
    silently-included World row is exactly the kind of error that survives
    review because the result still looks plausible.
    """
    years = _year_columns(wide)
    if not years:
        raise ValueError(
            "no four-digit year columns found; the preamble was probably not "
            f"skipped. Columns seen: {list(wide.columns)[:6]}")

    keep = ["Country Name", "Country Code"]
    missing = [c for c in keep if c not in wide.columns]
    if missing:
        raise ValueError(f"expected columns {missing} in the World Bank file")

    long = wide.melt(id_vars=keep, value_vars=years,
                     var_name="year", value_name="value")
    long = long.rename(columns={"Country Name": "country",
                                "Country Code": "iso3"})
    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])

    if drop_aggregates:
        if meta is None:
            raise ValueError(
                "dropping aggregates needs the Metadata_Country CSV from the "
                "same zip; pass drop_aggregates=False to keep them, but be "
                "aware that World and the income groups are then in your panel")
        code_col = next((c for c in meta.columns
                         if str(c).strip().lower() in
                         ("country code", "countrycode")), None)
        if code_col is None or "Region" not in meta.columns:
            raise ValueError(
                f"metadata file has no usable Country Code/Region pair: "
                f"{list(meta.columns)}")
        real = set(meta.loc[meta["Region"].notna(), code_col].astype(str))
        long = long[long["iso3"].astype(str).isin(real)]

    out = long.sort_values(["iso3", "year"]).reset_index(drop=True)
    out.attrs["quantpost"] = {
        "source_id": META.source_id, "citation": META.citation,
        "homepage": META.homepage, "licence": META.licence,
        "redistributable": True, "notes": list(META.notes),
    }
    return out


def load(path: str | Path, *, drop_aggregates: bool = True) -> pd.DataFrame:
    """Read a zip (or a bare CSV) straight to a long frame."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        wide, meta = read_zip(path)
    else:
        wide, meta = read_csv(path), None
        drop_aggregates = False
    return to_long(wide, meta, drop_aggregates=drop_aggregates)


def indicator_name(wide_or_long: pd.DataFrame) -> str:
    for col in ("Indicator Name", "indicator"):
        if col in wide_or_long.columns:
            vals = wide_or_long[col].dropna().unique()
            if len(vals):
                return str(vals[0])
    return "unknown indicator"
