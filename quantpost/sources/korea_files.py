"""Readers for the two Korean statistical portals' hand-downloaded CSVs.

Neither portal has an open API this container can reach, so both files arrive by
someone clicking Download. Each has a shape that breaks a naive `read_csv`, and
each trap is solved once here.

**Bank of Korea ECOS** exports *wide*: four metadata columns and then one column
per month, from 1988 to the present, with the leading decades blank because the
series does not go back that far. The most recent months carry a ` p)` suffix on
the column name marking them provisional — those are revised later, sometimes
materially, so they are flagged rather than silently mixed in.

**Korea Customs Service** exports one file per twelve-month window, long, with a
`총계` grand-total row at the top of each that is not an observation and will
double every aggregate if it survives the concatenation.

Licence: both are Korean public data. The customs statistics and ECOS are
published under 공공누리 (KOGL) terms, which permit redistribution with
attribution — but the specific type varies by table, so `LICENCE_NOTE` says what
to check before shipping values rather than asserting a blanket permission.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .base import SourceMeta

ECOS_META = SourceMeta(
    source_id="ecos_csv",
    name="Bank of Korea ECOS (downloaded CSV)",
    citation="Bank of Korea, Economic Statistics System",
    homepage="https://ecos.bok.or.kr/",
    licence="Korean public data (KOGL); confirm the per-table type before "
            "redistributing values.",
    redistributable=False,
    notes=("Wide format: one column per month from 1988.",
           "Columns ending ' p)' are provisional and get revised.",),
)

CUSTOMS_META = SourceMeta(
    source_id="kcs_csv",
    name="Korea Customs Service trade statistics (downloaded CSV)",
    citation="Korea Customs Service, Export/Import by Commodity and Country",
    homepage="https://www.tradedata.go.kr/",
    licence="Korean public data (KOGL); confirm the per-table type before "
            "redistributing values.",
    redistributable=False,
    notes=("One file per twelve-month window.",
           "Each file opens with a 총계 grand-total row that is not an "
           "observation.",),
)

LICENCE_NOTE = (
    "Korean government statistics are generally 공공누리 (KOGL) licensed, but "
    "the type — and therefore whether values may be redistributed — is set per "
    "table. Publish statistics computed from these files freely; check the "
    "table's own licence badge before shipping the values themselves."
)

_MONTH_COL = re.compile(r"^(\d{4})/(\d{2})")


def read_ecos_wide(path: str | Path) -> pd.DataFrame:
    """One ECOS wide CSV to a monthly frame with a `provisional` flag.

    Returns columns `value` and `provisional`, indexed by month start. Blank
    leading months are dropped rather than filled: ECOS pads every series to the
    table's full 1988-onward span, and a zero there would be a fabricated
    observation.
    """
    df = pd.read_csv(Path(path), encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"{path} has no rows")
    row = df.iloc[0]
    rows = []
    for col in df.columns:
        m = _MONTH_COL.match(str(col))
        if not m:
            continue
        raw = row[col]
        if pd.isna(raw) or str(raw).strip() == "":
            continue
        rows.append({
            "date": pd.Timestamp(int(m.group(1)), int(m.group(2)), 1),
            "value": float(str(raw).replace(",", "")),
            "provisional": " p)" in str(col) or "p)" in str(col),
        })
    if not rows:
        raise ValueError(
            f"{path} produced no monthly observations; the header did not "
            f"contain YYYY/MM columns. First columns seen: "
            f"{list(df.columns)[:6]}")
    out = pd.DataFrame(rows).set_index("date").sort_index()
    out.attrs["quantpost"] = {
        "source_id": ECOS_META.source_id, "citation": ECOS_META.citation,
        "homepage": ECOS_META.homepage, "licence": ECOS_META.licence,
        "redistributable": False,
        "series": str(row.get("통계표", "")).strip(),
        "item": str(row.get("계정코드", "")).strip(),
        "unit": str(row.get("단위", "")).strip(),
    }
    return out


def read_customs(paths) -> pd.DataFrame:
    """Concatenate customs windows into one long monthly frame.

    Drops the grand-total row and any duplicated (month, country) pair that
    overlapping download windows produce, keeping the first. Overlaps are easy
    to create by hand and their symptom — an aggregate that is exactly twice the
    truth for a few months — is not obvious in a plot.
    """
    frames = []
    for p in sorted(Path(x) for x in paths):
        df = pd.read_csv(p, encoding="utf-8-sig")
        need = {"기간", "국가", "수출 중량", "수출 금액"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{p.name} is missing columns {sorted(missing)}")
        df = df[df["기간"].astype(str).str.match(r"^\d{4}-\d{2}$")].copy()
        frames.append(df)
    if not frames:
        raise ValueError("no customs files given")
    t = pd.concat(frames, ignore_index=True)
    t["date"] = pd.to_datetime(t["기간"], format="%Y-%m")
    for c in ("수출 중량", "수출 금액", "수입 중량", "수입 금액"):
        if c in t.columns:
            t[c] = pd.to_numeric(t[c].astype(str).str.replace(",", ""),
                                 errors="coerce")
    before = len(t)
    t = t.drop_duplicates(subset=["date", "국가", "HS코드"], keep="first")
    t.attrs["quantpost"] = {
        "source_id": CUSTOMS_META.source_id, "citation": CUSTOMS_META.citation,
        "homepage": CUSTOMS_META.homepage, "licence": CUSTOMS_META.licence,
        "redistributable": False,
        "duplicate_rows_dropped": before - len(t),
    }
    return t


def china_share(customs: pd.DataFrame, *, country: str = "중국") -> pd.DataFrame:
    """Monthly export totals and one country's share, by weight and by value.

    Both are returned because they are different questions. Weight share asks
    how much silicon went there; value share asks how much money came back. A
    mix shift moves them apart, and which one a commentator quotes changes the
    story they can tell.
    """
    total = customs.groupby("date")[["수출 중량", "수출 금액"]].sum()
    one = (customs[customs["국가"] == country]
           .groupby("date")[["수출 중량", "수출 금액"]].sum())
    if one.empty:
        raise ValueError(f"no rows for country {country!r}")
    out = pd.DataFrame({
        "export_weight_total": total["수출 중량"],
        "export_value_total": total["수출 금액"],
        "export_weight_country": one["수출 중량"],
        "export_value_country": one["수출 금액"],
    }).dropna()
    out["share_weight"] = out.export_weight_country / out.export_weight_total
    out["share_value"] = out.export_value_country / out.export_value_total
    out["unit_value_total"] = out.export_value_total / out.export_weight_total
    out["unit_value_country"] = out.export_value_country / out.export_weight_country
    gaps = pd.date_range(out.index.min(), out.index.max(), freq="MS")
    out.attrs["quantpost"] = dict(customs.attrs.get("quantpost", {}),
                                  months=len(out), expected_months=len(gaps),
                                  missing_months=[str(d.date()) for d in gaps
                                                  if d not in out.index])
    return out
