"""Loader for a daily index or ETF price file you downloaded by hand.

This container has no egress to data APIs — every host except the package
registries answers 403 at the proxy — so a price series arrives as a file. That is
not only a limitation: nearly every exchange's own download is a browser export,
and the licence position is usually "look at it, do not republish it", which a file
loader plus a `redistributable=False` flag models honestly and a scraper does not.

So this module deliberately does **not** fetch. It reads what an exchange's export
button produces, and it is built around the five ways those exports go wrong:

1. **Newest first.** KRX and most exchange exports are in descending date order.
   Sorted wrongly, every return in the series changes sign relative to its date and
   nothing downstream complains. `load_prices` always sorts.
2. **Thousands separators.** `2,845.11` is a string, and `pd.to_numeric` turns it
   into `NaN` rather than raising, so a mis-parsed file looks like a short one.
3. **A percentage column that looks like a price.** Exports usually carry 등락률 /
   change% next to the close. Picking it up as the level is silent and ruinous, so
   column detection matches on an explicit alias list rather than on position.
4. **Duplicate and non-trading dates.** Deduplicated, keeping the last.
5. **Zero, negative or missing levels.** A single zero makes the log return
   infinite; `to_log_returns` refuses rather than propagating it.

Nothing here assumes a market or a language: `CLOSE_ALIASES` and `DATE_ALIASES`
cover the Korean and English headers these exports use, and both are arguments so a
file with a header nobody anticipated is a one-line fix rather than a rewrite.

**Licence.** The default `SourceMeta` is marked not redistributable, because for
most index and ETF data that is the true position. `quantpost.sources.licence_warnings`
then fires on any frame loaded through here, and the intended workflow is the one
this repository already uses for ICE data on FRED: **publish statistics, never
values.** `publishable_statistics` is the shape that takes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import SourceMeta, tidy

__all__ = ["CLOSE_ALIASES", "DATE_ALIASES", "PriceFileMeta", "load_prices",
           "publishable_statistics", "to_log_returns"]

#: Header names that mean "the date", lowercased and stripped of spaces.
#: `observation_date` is FRED's current header; it used to be `DATE`.
DATE_ALIASES = ("date", "observation date", "일자", "날짜", "거래일", "기준일자",
                "trade_date", "tradedate", "dt", "time", "시간", "년월일")

#: Header names that mean "the closing level". Ordered by preference — an
#: adjusted close beats a raw one, and both beat anything else.
CLOSE_ALIASES = ("adj close", "adjclose", "adjusted close", "수정종가",
                 "close", "종가", "종가지수", "체결가", "현재가", "last",
                 "closing price", "지수")

#: Headers that must never be mistaken for a level. Checked before the aliases,
#: because "등락률" contains no substring of "종가" but "전일대비 종가" does.
_NOT_A_LEVEL = ("등락", "대비", "change", "pct", "%", "rate", "return", "수익률",
                "거래량", "volume", "거래대금", "amount", "시가총액", "open",
                "high", "low", "시가", "고가", "저가")


def PriceFileMeta(*, name: str, homepage: str, citation: str,
                  licence: str = ("Exchange market data. Typically viewable and "
                                  "analysable but not redistributable — check the "
                                  "provider's terms before republishing values."),
                  redistributable: bool = False,
                  source_id: str = "price_file") -> SourceMeta:
    """Provenance for a hand-downloaded price file.

    A function rather than a constant because the citation is the one thing that
    cannot be defaulted: it has to name the actual provider and the actual download
    date, and a post whose data footer says "a CSV" is not citable.
    """
    return SourceMeta(
        source_id=source_id, name=name, citation=citation, homepage=homepage,
        licence=licence, redistributable=redistributable,
        notes=("Loaded from a manually downloaded file; this container cannot "
               "reach data APIs.",
               "Default position is publish statistics, never values."))


def _norm(header) -> str:
    return str(header).strip().lower().replace("_", " ").replace("  ", " ")


def _pick(columns, aliases, *, exclude=()) -> str | None:
    """First column whose normalised header matches an alias, exclusions first.

    Exact matches are preferred over substring matches, so a file with both `close`
    and `close change` cannot resolve to the second one just because it appears
    earlier.
    """
    norm = {c: _norm(c) for c in columns}
    allowed = {c: n for c, n in norm.items()
               if not any(bad in n for bad in exclude)}
    for alias in aliases:
        for col, n in allowed.items():
            if n == alias:
                return col
    for alias in aliases:
        for col, n in allowed.items():
            if alias in n:
                return col
    return None


def _to_number(series: pd.Series) -> pd.Series:
    """Numbers as exchanges write them: `2,845.11`, `1 234,5`, `(12.3)`, `-`."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = (series.astype(str).str.strip()
         .str.replace(" ", "", regex=False)      # non-breaking space
         .str.replace(",", "", regex=False)
         .str.replace(" ", "", regex=False))
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)   # (12.3) -> -12.3
    # FRED writes a bare "." for a non-trading day; exchanges use "-" or blank.
    s = s.replace({"": None, "-": None, "--": None, ".": None, "N/A": None,
                   "nan": None, "null": None})
    return pd.to_numeric(s, errors="coerce")


def _to_dates(series: pd.Series) -> pd.Series:
    """Dates as exchanges write them, including bare `20260819`."""
    s = series.astype(str).str.strip().str.replace(r"[./]", "-", regex=True)
    digits = s.str.fullmatch(r"\d{8}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if digits.any():
        out[digits] = pd.to_datetime(s[digits], format="%Y%m%d", errors="coerce")
    rest = ~digits
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], errors="coerce")
    return out


def load_prices(path: str | Path, *, meta: SourceMeta | None = None,
                column: str = "close", date_column: str | None = None,
                close_column: str | None = None,
                sheet: str | int = 0, encoding: str | None = None,
                min_rows: int = 30) -> pd.DataFrame:
    """Read one daily level series out of a hand-downloaded CSV or spreadsheet.

    Returns a tidy one-column frame of *levels*, ascending by date, with provenance
    attached. Pass `date_column` / `close_column` when the automatic pick is wrong —
    and check what it picked, because a wrong column here is the single most
    expensive mistake available in this whole pipeline.

    `min_rows` exists so a file that parsed into three usable rows raises here
    rather than producing a confident kurtosis twelve steps downstream.
    """
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        raw = pd.read_excel(p, sheet_name=sheet)
    else:
        raw = None
        for enc in ([encoding] if encoding else ["utf-8-sig", "cp949", "utf-8",
                                                 "latin-1"]):
            try:
                raw = pd.read_csv(p, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if raw is None:
            raise ValueError(f"could not decode {p} with any of the usual encodings")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [" ".join(str(x) for x in tup if str(x) != "nan").strip()
                       for tup in raw.columns]

    dcol = date_column or _pick(raw.columns, DATE_ALIASES)
    ccol = close_column or _pick(raw.columns, CLOSE_ALIASES, exclude=_NOT_A_LEVEL)
    if dcol is None:
        raise ValueError(f"no date column found in {list(raw.columns)}; pass "
                         f"date_column=")
    if ccol is None and close_column is None and len(raw.columns) == 2:
        # A FRED export is `observation_date,NASDAQCOM` — the level column is named
        # after the series, so it matches no alias. With exactly one other column
        # there is nothing to confuse it with, *except* a percentage or volume
        # column, which the exclusion list still has to veto: the first version of
        # this fallback cheerfully picked 등락률 and two tests caught it.
        only = [c for c in raw.columns if c != dcol][0]
        if not any(bad in _norm(only) for bad in _NOT_A_LEVEL):
            ccol = only
    if ccol is None:
        raise ValueError(f"no closing-level column found in {list(raw.columns)}; "
                         f"pass close_column= (candidates were excluded as "
                         f"change/volume/OHLC columns)")

    frame = pd.DataFrame({column: _to_number(raw[ccol]).to_numpy()},
                         index=_to_dates(raw[dcol]))
    frame = frame[frame.index.notna()]
    out = tidy(frame, meta or PriceFileMeta(
        name=p.stem, homepage="", citation=f"Manually downloaded file {p.name}"),
        extra={"source_file": p.name, "date_column": str(dcol),
               "close_column": str(ccol), "rows_read": int(len(raw))})
    usable = out[column].notna().sum()
    if usable < min_rows:
        raise ValueError(
            f"only {usable} usable rows from {p.name} (read {len(raw)}); the "
            f"column pick was date={dcol!r}, close={ccol!r} — check those")
    return out


def to_log_returns(levels: pd.DataFrame | pd.Series, *, column: str | None = None,
                   in_percent: bool = True) -> pd.Series:
    """Log returns from a level series, in percent by default.

    Log rather than simple, because everything downstream — the stylised-facts
    battery, any GARCH comparison — is written for log returns, and because a
    simple-return series compounded over decades runs into the sign problem
    documented in `dynamics.sde.simple_from_log`.

    Percent rather than fractions because that is the unit the rest of this
    repository uses, and because a kurtosis is scale-free so the choice costs
    nothing where it matters.

    Refuses non-positive levels instead of returning infinities: an index export
    with a zero in it is a broken file, not a market event.
    """
    s = (levels if isinstance(levels, pd.Series)
         else levels[column or levels.columns[0]]).astype(float).dropna()
    if s.empty:
        raise ValueError("no usable levels")
    bad = (s <= 0).sum()
    if bad:
        raise ValueError(f"{bad} non-positive level(s) in the series; a price file "
                         f"with a zero in it needs fixing rather than logging")
    r = np.log(s).diff().dropna()
    out = r * 100.0 if in_percent else r
    out.name = "log_return_pct" if in_percent else "log_return"
    out.attrs.update(getattr(levels, "attrs", {}))
    return out


def publishable_statistics(returns: pd.Series, *,
                           quantiles=(0.5, 0.9, 0.99),
                           min_tail_obs: int = 10) -> dict:
    """Aggregates that can be published when the underlying series cannot.

    Counts, moments, and a date *range* — no individual observation and no dated
    value. Having this as a function means "publish statistics, never values" is
    enforced by the code path rather than by remembering.

    The first version of this returned `worst_pct` and `best_pct` too, and a test
    that searched the output for individual returns caught it: the minimum and
    maximum of a return series **are** single observations, and the most identifying
    ones in it — "the worst day was -8.77%" names a date to anyone with a chart. They
    are gone.

    The same objection applies by degrees to an extreme quantile. On twenty years of
    daily data the 99th percentile of the absolute return is an aggregate over fifty
    observations; on one year it is essentially the second-worst day. So any
    requested quantile whose tail rests on fewer than `min_tail_obs` observations is
    dropped, and the dropped ones are reported in `quantiles_omitted` rather than
    silently disappearing.
    """
    r = pd.Series(returns).astype(float).dropna()
    if r.empty:
        raise ValueError("no returns to summarise")
    c = r - r.mean()
    var = float(c.var(ddof=1))
    kept, omitted = {}, {}
    for q in quantiles:
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile {q} must lie strictly in (0, 1)")
        tail = int(round(min(q, 1.0 - q) * r.size))
        if tail < min_tail_obs:
            omitted[q] = tail
        else:
            kept[q] = float(r.abs().quantile(q))
    return {
        "n": int(r.size),
        "first_date": str(pd.Timestamp(r.index[0]).date()) if isinstance(
            r.index, pd.DatetimeIndex) else None,
        "last_date": str(pd.Timestamp(r.index[-1]).date()) if isinstance(
            r.index, pd.DatetimeIndex) else None,
        "mean_pct": float(r.mean()),
        "sd_pct": float(r.std(ddof=1)),
        "annualised_sd_pct": float(r.std(ddof=1) * np.sqrt(252.0)),
        "skewness": float((c ** 3).mean() / var ** 1.5),
        "excess_kurtosis": float((c ** 4).mean() / var ** 2 - 3.0),
        "abs_quantiles_pct": kept,
        "quantiles_omitted": omitted,
    }
