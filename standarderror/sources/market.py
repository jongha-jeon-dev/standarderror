"""Keyless daily OHLC. Read the caveats before you publish anything from this.

`stooq` is undocumented, reverse-engineered, `robots.txt`-disallows the
`/q/d/l/` path, caps history at roughly five years and has an unpublished daily
hit limit. It is therefore **off by default**: you must pass
`accept_terms=True`, which exists purely so the decision is explicit and shows
up in code review rather than being made by accident. If you work somewhere with
a data-governance function, this is the adapter to clear with them first.

`yfinance` is an unofficial Yahoo scraper whose dominant failure mode is HTTP
429 and IP blacklisting. Fine for exploration, not for anything scheduled.

Preferred alternative: FRED carries `SP500` and `VIXCLS` under a sanctioned API.
Use `standarderror.sources.fred` when FRED has the series.
"""

from __future__ import annotations

import io
from urllib.parse import urlencode

import pandas as pd

from ..cache import fetch
from .base import SourceMeta, tidy

STOOQ_META = SourceMeta(
    source_id="stooq",
    name="Stooq",
    citation="Stooq.com daily quotes",
    homepage="https://stooq.com/",
    licence="No published licence. robots.txt disallows /q/d/l/.",
    redistributable=False,
    notes=("Undocumented endpoint; ~5 years of history; unpublished daily hit "
           "cap; robots.txt disallows the download path.",),
)

_STOOQ = "https://stooq.com/q/d/l/"


def stooq(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    *,
    interval: str = "d",
    accept_terms: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """Daily OHLCV for one Stooq symbol (`^spx`, `aapl.us`, `^vix`, ...)."""
    if not accept_terms:
        raise RuntimeError(
            "stooq() requires accept_terms=True. The endpoint is "
            "robots.txt-disallowed and undocumented; confirm this is acceptable "
            "for your use before enabling it. FRED's SP500/VIXCLS series are a "
            "sanctioned alternative.")
    params = {"s": symbol, "i": interval}
    if start:
        params["d1"] = pd.Timestamp(start).strftime("%Y%m%d")
    if end:
        params["d2"] = pd.Timestamp(end).strftime("%Y%m%d")
    url = f"{_STOOQ}?{urlencode(params)}"
    raw = fetch(url, source="stooq", force=force, note=symbol)
    df = pd.read_csv(io.BytesIO(raw))
    if "Date" not in df.columns:
        raise RuntimeError(
            f"Stooq returned no usable CSV for {symbol!r} (columns "
            f"{list(df.columns)}). Symbol wrong, or the daily hit cap was hit.")
    df = df.set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    return tidy(df, STOOQ_META, extra={"symbol": symbol})


def yahoo(
    tickers: list[str] | str,
    start: str | None = None,
    end: str | None = None,
    *,
    field: str = "Close",
) -> pd.DataFrame:
    """Thin wrapper over yfinance, imported lazily so it stays optional."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed (pip install yfinance). It is optional "
            "and unreliable by design; prefer FRED where possible.") from exc

    meta = SourceMeta(
        source_id="yahoo", name="Yahoo Finance (via yfinance)",
        citation="Yahoo Finance", homepage="https://finance.yahoo.com/",
        licence="Unofficial scrape; no redistribution rights.",
        redistributable=False,
        notes=("Rate-limited (HTTP 429) and endpoint-unstable; do not schedule.",))
    if isinstance(tickers, str):
        tickers = [tickers]
    raw = yf.download(tickers, start=start, end=end, progress=False,
                      threads=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        frame = raw[field]
    else:
        frame = raw[[field]].rename(columns={field: tickers[0]})
    return tidy(frame, meta, extra={"tickers": tickers, "field": field})
