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

#: Figures BLS states about the survey itself, kept as data with the sentence
#: each one came from, because an earlier version of this file stored the numbers
#: without their wording and that produced a wrong result.
#:
#: The mistake is worth recording. `detectable_change_then` (0.18) and the
#: technical note's confidence interval (0.30) were stored side by side and their
#: ratio, 1.67, was read as how much the survey had degraded. Both halves are
#: wrong. The 0.30 is stated *at an unemployment rate of around 6.0 percent* and
#: has to be rescaled by sqrt(p(1-p)) to the rate actually prevailing; and the
#: report never gives a current one-month threshold at all — it says a 0.18 point
#: change now needs *two months* of data, which implies about 0.18*sqrt(2).
#: Corrected, the two documents agree: 0.253 and 0.255. There is no 1.67.
#:
#: So every figure below carries `quote`, and anything derived from two of them
#: has to state what makes them comparable.
CPS_REPORT_2026 = (
    "Bureau of Labor Statistics Report to the Appropriations Committees on "
    "Modernizing the Current Population Survey, April 2026"
)
CPS_TECHNICAL_NOTE = "BLS Employment Situation, Technical Note"

CPS_2026 = {
    # --- from the April 2026 report to the Appropriations Committees ----------
    "households_eligible": 60_000,
    "households_eligible_quote": (
        "The CPS sample size in 2026, which consists of 60,000 eligible "
        "households, is the same as it was in 1981"),
    "response_rate_then": (0.90, 0.93),
    "response_rate_now": (0.66, 0.69),
    "response_rate_quote": (
        "Response rates have historically been very high but have declined from "
        "the low 90 percent range to the upper 60 percent range over the last 15 "
        "years"),
    "people_per_respondent_then": 2_100,
    "people_per_respondent_now": 3_500,
    "people_per_respondent_quote": (
        "In just the last 20 years, a single respondent has gone from "
        "representing about 2,100 people to about 3,500 people."),
    "responses_for_one_tenth_point": 50,
    "responses_quote": (
        "At current sample sizes, a net change of fewer than 50 survey responses "
        "could move the headline unemployment rate by 0.1 percentage point."),
    #: The one figure the whole exercise turns on. Note what it does *not* say:
    #: there is no current one-month threshold here, only that the same change
    #: now takes two months.
    "detectable_change_then": 0.18,
    "detectable_change_then_years_ago": 20,
    "months_needed_now": 2,
    "detectable_change_quote": (
        "For example, 20 years ago the CPS could detect that an over-the-month "
        "change in the unemployment rate as small as 0.18 percentage point was "
        "statistically significant, while now it would take two months of data "
        "before one could determine that this same change is statistically "
        "significant."),
    "parallel_survey_households": 45_000,
    "parallel_survey_eligible": 40_000,
    "parallel_survey_months": 18,
    "parallel_survey_cost_usd": 60_000_000,
    "parallel_survey_detectable": 0.25,
    "parallel_survey_quote": (
        "At this sample size, the CPS will be able to detect differences in the "
        "unemployment rate between the parallel survey and the official CPS of "
        "0.25 percentage point over 2 months of data."),

    # --- from the Employment Situation technical note -------------------------
    "ci90_change": 0.30,
    #: The condition, which is the part that gets dropped when this number is
    #: quoted. Without it the figure is not comparable to anything.
    "ci90_change_stated_at_rate": 6.0,
    "ci90_change_quote": (
        "At an unemployment rate of around 6.0 percent, the 90-percent "
        "confidence interval for the monthly change in the unemployment rate is "
        "about +/- 0.3 percentage point."),

    # --- survey design -------------------------------------------------------
    "monthly_overlap": 0.75,
}

#: A third published estimate of the same quantity, from outside BLS, useful as a
#: check on the series-implied value rather than as a source for it.
STLOUISFED_2026 = {
    "source": ("Federal Reserve Bank of St. Louis, On the Economy, March 2026, "
               "'Understanding Statistical Significance in Monthly "
               "Unemployment Data'"),
    "threshold_start": 0.18,
    "threshold_end": 0.21,
    "window": ("2022-02", "2025-06"),
    "response_rate_fall_pp": 6.3,
    "quote": ("the significance threshold rose from 0.18 percentage points to "
              "0.21 percentage points"),
}


def cps_detectable_now(rate: float | None = None) -> dict:
    """The current one-month detectable change, from both documents, comparably.

    Returns both routes and their agreement, because the agreement is the
    evidence that the reading is right:

    * the technical note's interval, rescaled from the 6.0 percent it is stated
      at to `rate`;
    * the report's "two months for a 0.18 point change", converted to one month.

    `rate` defaults to nothing and must be supplied by the caller from the data,
    so that the prevailing unemployment rate used for the rescaling is the one in
    the window being discussed rather than a number chosen here.
    """
    from ..ts.noisescale import rescale_for_rate

    if rate is None:
        raise ValueError(
            "supply the prevailing unemployment rate for the window; the "
            "technical note's figure is conditional on it")
    note = rescale_for_rate(CPS_2026["ci90_change"],
                            stated_at=CPS_2026["ci90_change_stated_at_rate"],
                            actual=float(rate))
    report = CPS_2026["detectable_change_then"] * float(
        CPS_2026["months_needed_now"]) ** 0.5
    return {"rate": float(rate), "from_technical_note": float(note),
            "from_report": float(report),
            "ratio": float(report / note),
            "mean": float(0.5 * (note + report)),
            "degradation": float(0.5 * (note + report)
                                 / CPS_2026["detectable_change_then"])}


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
