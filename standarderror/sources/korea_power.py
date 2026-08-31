"""Korean electricity supply-and-demand files, as EPSIS actually exports them.

Three traps live in these files and each one is capable of silently changing a
published number.

**The peak in EPSIS is not the peak in the national plan.** EPSIS reports the
maximum load settled through the power exchange. The Basic Plan for Long-term
Electricity Supply and Demand forecasts a wider quantity that also carries
behind-the-meter generation and direct contracts. For 2023 the National Assembly
Research Service quotes 98.3 GW against EPSIS's 93.6 GW — a 5% gap. Growth rates
computed inside one series are unaffected; a *level* comparison between the two
is not, and `PLAN_BASIS_2023` exists so that the bridge is a stated constant
rather than an assumption buried in a script.

**Minimum load barely exists.** EPSIS provides 최소전력 only from 2025-11-24, so
the column is almost entirely "-". It is dropped rather than forward-filled,
because a reserve-margin series computed from a filled minimum would look
complete and be wrong.

**Zero means missing in the monthly export and blank means missing in the annual
one.** The monthly table pads unrecorded months with 0 MW; a mean taken over
those zeros understates by whatever share of the year is padded, which is most
of it before 2005.

And one thing that is not a trap but is easy to miss: the current year is
partial. The annual table's last row carries the highest load *so far*, dated,
and Korea's annual peak lands in either August or December depending on the
year — roughly one year in three since 1993 peaked in winter. So the running
year is excluded by default rather than compared against completed ones.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["read_supply_demand", "read_sales", "read_monthly_peak",
           "annual_frame", "PLAN_BASIS_2023", "LICENCE_NOTE"]

# NARS 「제11차 전력수급기본계획」 실무안의 평가와 제언 (2025-01-02), 표 1:
# 2023 최대전력 실적 98.3 GW, against EPSIS's 93,615 MW for the same year.
PLAN_BASIS_2023 = 98_300 / 93_615

LICENCE_NOTE = (
    "Korea Power Exchange (EPSIS) statistics are Korean public data under "
    "공공누리 (KOGL), whose type is set per table. This post publishes "
    "statistics computed from the exports, not the underlying tables."
)

_SD_COLS = ["year", "month", "day", "capacity_mw", "supply_mw", "peak_mw",
            "min_mw", "reserve_mw", "reserve_pct", "peak_ts", "min_ts"]


def _num(s: pd.Series) -> pd.Series:
    """EPSIS mixes thousands separators, '-' for missing and bare integers."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False)
                         .str.strip().replace({"-": None, "": None}),
                         errors="coerce")


def read_supply_demand(path: str | Path) -> pd.DataFrame:
    """Annual peak-load table: the year's highest load and the day it happened.

    `min_mw` is returned but is almost entirely missing by construction; see the
    module docstring.
    """
    df = pd.read_csv(path)
    if len(df.columns) != len(_SD_COLS):
        raise ValueError(
            f"expected {len(_SD_COLS)} columns from the EPSIS 전력수급실적 "
            f"export, got {list(df.columns)}")
    df.columns = _SD_COLS
    for c in ("capacity_mw", "supply_mw", "peak_mw", "min_mw", "reserve_mw",
              "reserve_pct"):
        df[c] = _num(df[c])
    df = df[df.peak_mw.notna()].sort_values("year").reset_index(drop=True)
    df["year"] = df.year.astype(int)
    df["peak_month"] = df.month.astype(int)
    df["winter_peak"] = df.peak_month.isin((11, 12, 1, 2))
    df.attrs["standarderror"] = {
        "source": "Korea Power Exchange, EPSIS 전력수급실적 (annual)",
        "basis": "power-exchange settled load; see PLAN_BASIS_2023",
        "years": f"{df.year.min()}-{df.year.max()}",
    }
    return df


def read_sales(path: str | Path) -> pd.DataFrame:
    """Annual electricity sales by contract class, converted MWh -> TWh.

    The classes are kept separate on purpose. A claim about semiconductor fabs
    and data centres is a claim about *industrial* load, and the aggregate hides
    that industrial sales have not grown since 2018 while general-service sales
    have.
    """
    df = pd.read_csv(path)
    ren = {"연도": "year", "주택용": "residential", "일반용": "general",
           "교육용": "education", "산업용": "industrial", "농사용": "agricultural",
           "가로등": "street_lighting", "심야": "night", "합계": "total"}
    missing = set(ren) - set(df.columns)
    if missing:
        raise ValueError(f"sales export is missing columns {sorted(missing)}")
    df = df.rename(columns=ren)
    for c in ren.values():
        if c != "year":
            df[c] = _num(df[c]) / 1e6           # MWh -> TWh
    df["year"] = df.year.astype(int)
    df = df.sort_values("year").reset_index(drop=True)
    # The published total is kept rather than recomputed, and then checked: a
    # silent column change upstream would otherwise pass unnoticed.
    parts = df[[c for c in ren.values() if c not in ("year", "total")]].sum(axis=1)
    gap = (parts - df.total).abs().max()
    if gap > 0.5:
        raise ValueError(f"contract classes miss the published total by {gap:.2f} TWh")
    df.attrs["standarderror"] = {
        "source": "Korea Power Exchange, EPSIS 계약종별 판매전력량 (annual)",
        "unit": "TWh", "years": f"{df.year.min()}-{df.year.max()}",
        "note": "sales, so excludes self-generation and transmission losses",
    }
    return df


def read_monthly_peak(path: str | Path) -> pd.Series:
    """Monthly mean of daily peak load, as a dated MW series.

    This is *not* the monthly maximum. EPSIS averages the daily peaks within the
    month, which is why its 2018 July figure is 79.0 GW against that month's
    instantaneous record of 92.5 GW. It is the right series for a question about
    the level of load and the wrong one for a question about the peak; the ratio
    between the two is itself worth looking at, and has moved.
    """
    wide = pd.read_csv(path)
    if wide.columns[0] != "연도":
        raise ValueError(f"expected a 연도 column, got {wide.columns[0]!r}")
    long = wide.melt(id_vars="연도", var_name="month_kr", value_name="mw")
    long["month"] = long.month_kr.str.replace("월", "", regex=False).astype(int)
    long["mw"] = _num(long.mw)
    # Zero is the padding EPSIS writes for a month it has no record for.
    long = long[long.mw.fillna(0) > 0].copy()
    long["date"] = pd.to_datetime(
        dict(year=long["연도"].astype(int), month=long.month, day=1))
    s = long.set_index("date").mw.sort_index()
    s.attrs["standarderror"] = {
        "source": "Korea Power Exchange, EPSIS 최대전력 월별평균",
        "unit": "MW", "note": "monthly mean of daily peaks, not the monthly max",
    }
    return s


def annual_frame(sd: pd.DataFrame, sales: pd.DataFrame,
                 monthly: pd.Series | None = None, *,
                 drop_running_year: bool = True) -> pd.DataFrame:
    """One row per completed year, with the pieces the analysis needs joined.

    `load_factor` is a percentage and uses sales over peak, so it is a lower
    bound on the true load factor — sales exclude self-generation and network
    losses. It is comparable across years, which is what a question about drift
    needs, and it is not comparable with a figure computed from generation.
    """
    df = sd.set_index("year")[["peak_mw", "capacity_mw", "reserve_pct",
                               "peak_month", "winter_peak"]].copy()
    df = df.join(sales.set_index("year")[["total", "industrial", "general",
                                          "residential"]])
    if monthly is not None:
        by_year = monthly.groupby(monthly.index.year)
        counts = by_year.size()
        means = by_year.mean().where(counts == 12)   # partial years are not means
        df["monthly_avg_mw"] = means
        df["peak_to_avg"] = df.peak_mw / df.monthly_avg_mw
    # TWh -> MWh is 1e6, and a year is 8,760 hours; writing 8.760 for the hours
    # silently returned a load factor of 694 percent.
    df["load_factor"] = 100.0 * df.total * 1e6 / (df.peak_mw * 8760.0)
    if drop_running_year:
        # The running year's peak is a maximum over part of a year, and Korea's
        # annual peak is as likely to arrive in December as in August.
        df = df.iloc[:-1]
    df.attrs["standarderror"] = {"basis_bridge_2023": PLAN_BASIS_2023}
    return df
