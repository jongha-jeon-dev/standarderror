"""Common contract for every data adapter.

Every adapter returns a tidy `pandas.DataFrame` indexed by a `DatetimeIndex`
named `date`, with one column per series, plus a `.attrs["quantpost"]` dict
carrying source id, citation string and licence note. The citation is not
decoration: FRED's terms *require* an attribution line, ICE data on FRED is
internal-use-only, and BIS/ECB expect attribution. Carrying it with the frame
means the renderer can emit a correct "Data" footer automatically instead of
relying on you to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class SourceMeta:
    source_id: str
    name: str
    citation: str
    homepage: str
    licence: str = "See homepage."
    redistributable: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def footer_line(self) -> str:
        return f"{self.citation} — <{self.homepage}>"


class Source(Protocol):
    meta: SourceMeta

    def get(self, *args, **kwargs) -> pd.DataFrame: ...


def tidy(
    frame: pd.DataFrame,
    meta: SourceMeta,
    *,
    extra: dict | None = None,
) -> pd.DataFrame:
    """Normalise index/dtypes and attach provenance metadata."""
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    payload = {
        "source_id": meta.source_id,
        "citation": meta.citation,
        "homepage": meta.homepage,
        "licence": meta.licence,
        "redistributable": meta.redistributable,
        "notes": list(meta.notes),
    }
    payload.update(extra or {})
    out.attrs["quantpost"] = payload
    return out


def merge_sources(*frames: pd.DataFrame) -> pd.DataFrame:
    """Outer-join frames on date, preserving the union of provenance blocks."""
    if not frames:
        raise ValueError("merge_sources needs at least one frame")
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    sources = []
    for f in frames:
        p = f.attrs.get("quantpost")
        if p and p not in sources:
            sources.append(p)
    out.attrs["quantpost_sources"] = sources
    return out
