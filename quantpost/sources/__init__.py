"""Public-data adapters.

    from quantpost.sources import fred, ecb, ecos, bis
    df = fred.get(["ust_10y", "ust_2y", "vix"], start="2015-01-01")

Every adapter returns a DatetimeIndex-ed frame with provenance in
`df.attrs["quantpost"]`. `citations(df)` turns that into the footer line a post
needs, and `licence_warnings(df)` surfaces anything you are not allowed to
redistribute — check it before publishing.
"""

from __future__ import annotations

import pandas as pd

from . import bis, ecb, ecos, fred, hmda, local, market, owid, worldbank
from .base import SourceMeta, merge_sources, tidy

__all__ = [
    "fred", "ecb", "ecos", "bis", "market", "hmda", "local",
    "worldbank", "owid",
    "SourceMeta", "tidy", "merge_sources", "citations", "licence_warnings",
]


def _blocks(df: pd.DataFrame) -> list[dict]:
    if "quantpost_sources" in df.attrs:
        return list(df.attrs["quantpost_sources"])
    one = df.attrs.get("quantpost")
    return [one] if one else []


def citations(*frames: pd.DataFrame) -> list[str]:
    """De-duplicated citation lines for every source behind these frames."""
    lines: list[str] = []
    for df in frames:
        for b in _blocks(df):
            line = f"{b['citation']} — <{b['homepage']}>"
            if line not in lines:
                lines.append(line)
    return lines


def licence_warnings(*frames: pd.DataFrame) -> list[str]:
    """Anything that must not be redistributed, plus adapter-level warnings."""
    out: list[str] = []
    for df in frames:
        for b in _blocks(df):
            if b.get("redistributable") is False:
                out.append(
                    f"{b['source_id']}: NOT redistributable — {b.get('licence','')}")
            for note in b.get("notes", []):
                if note.startswith("WARNING"):
                    out.append(f"{b['source_id']}: {note}")
    return out
