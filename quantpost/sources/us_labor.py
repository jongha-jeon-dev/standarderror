"""US household-survey labour statistics, as BLS publishes them.

Licence
-------
Works of the US federal government are not subject to domestic copyright, so
unlike a vendor's index series these values can be republished, charted and
tabulated. Attribution is still owed and `LICENCE_NOTE` carries it.

Traps this module exists to absorb
----------------------------------
The download from `data.bls.gov` is a spreadsheet written for a human, not a
parser, and three of its habits will quietly corrupt a series:

**The preamble is not a fixed number of rows.** It carries the series id, the
adjustment status, the title and sometimes a footnote, and the count changes with
the series and with whether you asked for annual averages. Skipping a fixed
number of lines works until it does not, so the header row is *found* by looking
for the one that starts with "Year".

**An annual-average column sits at the right-hand end.** Melting the wide table
without dropping it turns one number a year into a thirteenth month, which is
invisible in a chart and moves every estimate.

**Months with no data yet are blank, not zero.** The current year is partial.
Filling them gives a series that dives to zero in the final months, and a break
test finds a very significant break at exactly the download date.

**The default download is a spreadsheet, not a CSV.** `data.bls.gov` hands you a
`SeriesReport*.xlsx` whose single sheet has the same human-facing layout, so both
are read here and the extension decides which reader runs.

And one fact about this particular series that is not a parsing trap but is worse:
**October 2025 does not exist.** BLS states that "household survey data from the
Current Population Survey were not collected for the October 2025 reference period
due to a lapse in appropriations and will not be collected retroactively." The hole
is permanent. The danger is not the missing value, which is visible; it is
*closing* the gap, which is not. Dropping it makes September and November adjacent,
and every difference taken across that join is a two-month change being read as a
one-month change. So `monthly_series` reindexes onto the full monthly grid and
leaves the hole as a NaN, where a difference operator will propagate it and a
robust scale will drop it, rather than quietly stitching the series together.

The series to ask for is `LNS14000000`, the seasonally adjusted civilian
unemployment rate, monthly, 1948 to now.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["LICENCE_NOTE", "UNEMPLOYMENT_SERIES", "MONTHS", "load_bls_table",
           "monthly_series", "contiguous_runs", "CPS_2026",
           "MISSING_MONTHS"]

LICENCE_NOTE = (
    "US Bureau of Labor Statistics, Current Population Survey. Works of the US "
    "federal government carry no domestic copyright, so the values are "
    "reproduced here rather than only summarised."
)

UNEMPLOYMENT_SERIES = "LNS14000000"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#: BLS never collected the October 2025 household survey and says it will not do
#: so retroactively, so this month is permanently absent from the series.
MISSING_MONTHS = {"2025-10": "not collected; lapse in appropriations"}

#: Figures BLS states about the survey itself, from its April 2026 report to the
#: Appropriations Committees on modernising the CPS, and from the Employment
#: Situation technical note. Kept as data so the post cannot drift from them.
CPS_2026 = {
    "households_eligible": 60_000,          # unchanged since 1981
    "population_growth_since_1981": 0.61,
    "response_rate_then": (0.90, 0.93),     # "low 90 percent range", ~15y ago
    "response_rate_now": (0.66, 0.69),      # "upper 60 percent range"
    "people_per_respondent_then": 2_100,
    "people_per_respondent_now": 3_500,
    "responses_for_one_tenth_point": 50,    # "fewer than 50 survey responses"
    "detectable_change_then": 0.18,         # pp, one month, 20 years ago
    "ci90_change_now": 0.30,                # pp, Employment Situation tech note
    "parallel_survey_households": 45_000,
    "parallel_survey_eligible": 40_000,
    "parallel_survey_months": 18,
    "parallel_survey_cost_usd": 60_000_000,
    "parallel_survey_detectable": 0.25,     # pp, over two months
    "monthly_overlap": 0.75,                # share of the sample carried over
}


def _load_excel(path: Path) -> pd.DataFrame:
    """The `SeriesReport*.xlsx` that data.bls.gov hands you by default.

    One sheet, the same preamble-then-table layout as the CSV, and the header row
    found the same way rather than counted.
    """
    raw = pd.read_excel(path, header=None)
    first = raw.iloc[:, 0].astype(str).str.strip().str.lower()
    hits = np.flatnonzero(first.values == "year")
    if hits.size == 0:
        raise ValueError(
            "no row beginning with 'Year' — this does not look like a BLS "
            "data.bls.gov download. Check that the file is the data table and "
            "not the series-description page.")
    h = int(hits[0])
    header = [str(c).strip() for c in raw.iloc[h].tolist()]
    body = raw.iloc[h + 1:].copy()
    body.columns = header
    month_cols = [c for c in body.columns
                  if isinstance(c, str) and c[:3].title() in MONTHS]
    dropped = [c for c in body.columns if c not in month_cols and c != "Year"]
    long = body.melt(id_vars="Year", value_vars=month_cols,
                     var_name="month", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["year"] = pd.to_numeric(long["Year"], errors="coerce")
    long = long.dropna(subset=["year", "value"])
    long["m"] = long["month"].str[:3].str.title().map(
        {m: i + 1 for i, m in enumerate(MONTHS)})
    out = pd.DataFrame({
        "date": pd.to_datetime(dict(year=long["year"].astype(int),
                                    month=long["m"].astype(int), day=1)),
        "value": long["value"].astype(float)})
    out = out.sort_values("date").reset_index(drop=True)
    out.attrs["dropped_columns"] = dropped
    return out


def _find_header(lines: list[str]) -> int:
    """Index of the row that starts the table, found rather than assumed."""
    for i, line in enumerate(lines):
        first = line.split(",", 1)[0].strip().strip('"')
        if first.lower() == "year":
            return i
    raise ValueError(
        "no row beginning with 'Year' — this does not look like a BLS "
        "data.bls.gov download. Check that the file is the data table and not "
        "the series-description page.")


def load_bls_table(path) -> pd.DataFrame:
    """A BLS one-series download, wide or long, as a tidy monthly frame.

    Returns columns `date` (month start) and `value`, sorted, with missing months
    dropped rather than filled.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        return _load_excel(path)
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln for ln in raw.splitlines() if ln.strip()]

    # The flat files on download.bls.gov are tab-separated and already long. The
    # test is for a `period` column, not for `series_id`: the wide download's
    # first preamble line is `"Series Id:","LNS14000000"`, which matches any
    # looser check and sends a wide file down the long path.
    head = lines[0].lower()
    if "period" in head and ("series_id" in head or "series id" in head):
        sep = "\t" if "\t" in lines[0] else ","
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep)
        df.columns = [c.strip().lower() for c in df.columns]
        per = df["period"].astype(str).str.upper().str.strip()
        keep = per.str.match(r"^M(0[1-9]|1[0-2])$")   # M13 is the annual average
        df = df[keep]
        month = per[keep].str[1:].astype(int)
        out = pd.DataFrame({
            "date": pd.to_datetime(dict(year=df["year"].astype(int),
                                        month=month, day=1)),
            "value": pd.to_numeric(df["value"], errors="coerce")})
        return out.dropna().sort_values("date").reset_index(drop=True)

    start = _find_header(lines)
    table = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    header = [c.strip() for c in table[0]]
    wide = pd.DataFrame(table[1:], columns=header)
    # Anything that is not one of the twelve month abbreviations is dropped, which
    # is how the annual-average column stops becoming a thirteenth month.
    month_cols = [c for c in wide.columns if c[:3].title() in MONTHS]
    dropped = [c for c in wide.columns if c not in month_cols and c != "Year"]
    long = wide.melt(id_vars="Year", value_vars=month_cols,
                     var_name="month", value_name="value")
    long["value"] = pd.to_numeric(
        long["value"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce")
    long["year"] = pd.to_numeric(long["Year"], errors="coerce")
    long = long.dropna(subset=["year", "value"])
    long["m"] = long["month"].str[:3].str.title().map(
        {m: i + 1 for i, m in enumerate(MONTHS)})
    out = pd.DataFrame({
        "date": pd.to_datetime(dict(year=long["year"].astype(int),
                                    month=long["m"].astype(int), day=1)),
        "value": long["value"].astype(float)})
    out = out.sort_values("date").reset_index(drop=True)
    out.attrs["dropped_columns"] = dropped
    return out


def monthly_series(path, *, start: str | None = None, end: str | None = None,
                   strict: bool = False) -> pd.Series:
    """`load_bls_table` on the complete monthly grid, holes left as NaN.

    Reindexing rather than dropping is the whole point. A missing month that is
    present as a NaN propagates through `np.diff` and gets filtered by a robust
    scale, which is correct. A missing month that has been dropped makes its
    neighbours adjacent, and then a "one-month change" spanning the join is
    actually a two-month change — an error with no symptom.

    `strict=True` raises instead, for a caller that wants to know rather than to
    cope. The gaps are always listed in `.attrs["gaps"]`.
    """
    df = load_bls_table(path)
    s = df.set_index("date")["value"].astype(float)
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    grid = pd.date_range(s.index[0], s.index[-1], freq="MS")
    gaps = grid.difference(s.index)
    if len(gaps) and strict:
        raise ValueError(
            f"{len(gaps)} missing months between {s.index[0].date()} and "
            f"{s.index[-1].date()}, first at {gaps[0].date()}. Second "
            f"differences across a hole are not second differences.")
    s = s.reindex(grid)
    s.attrs["licence"] = LICENCE_NOTE
    s.attrs["gaps"] = [str(g.date()) for g in gaps]
    return s


def contiguous_runs(s: pd.Series) -> list[pd.Series]:
    """The series split into stretches with no missing months.

    For an estimator that cannot see a NaN and must be handed clean segments.
    Reported as a list so the caller has to notice that there is more than one.
    """
    out, run = [], []
    for idx, val in s.items():
        if np.isfinite(val):
            run.append((idx, val))
        elif run:
            out.append(pd.Series(dict(run)))
            run = []
    if run:
        out.append(pd.Series(dict(run)))
    return out
