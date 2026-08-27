"""Loaders for datasets that cannot be fetched programmatically.

Freddie Mac's Single-Family Loan-Level dataset is the canonical case: access
requires a login to Clarity Data Intelligence, there is no public REST endpoint,
and commercial redistribution needs a signed agreement. So the adapter is a
*file* loader over ZIPs you downloaded by hand, with the licence obligation
attached to the frame so the renderer can print it.

Column layouts have changed across releases. Rather than hard-code a layout that
silently mis-parses next year's file, `load_freddie` takes an explicit column
list (pull the current one from the User Guide in the portal) and validates the
field count it actually finds.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from .base import SourceMeta

FREDDIE_META = SourceMeta(
    source_id="freddie_sf_llp",
    name="Freddie Mac Single-Family Loan-Level Dataset",
    citation="Freddie Mac, Single-Family Loan-Level Dataset",
    homepage="https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset",
    licence=("Free for non-commercial academic/research and limited use. "
             "Commercial redistribution requires a signed licensing agreement. "
             "Dataset is unaudited and subject to change."),
    redistributable=False,
    notes=("No public API: download manually from Clarity Data Intelligence.",
           "Verify the current file layout in the portal User Guide - it has "
           "changed across releases.",),
)


def load_freddie(
    path: str | Path,
    columns: list[str],
    *,
    member: str | None = None,
    nrows: int | None = None,
    sep: str = "|",
) -> pd.DataFrame:
    """Read a pipe-delimited Freddie Mac origination/performance file.

    `columns` must match the release's layout exactly; a mismatch raises rather
    than silently shifting every field by one.
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = member or next(
                n for n in zf.namelist() if not n.endswith("/"))
            with zf.open(name) as fh:
                df = pd.read_csv(fh, sep=sep, header=None, nrows=nrows,
                                 low_memory=False)
    else:
        df = pd.read_csv(path, sep=sep, header=None, nrows=nrows,
                         low_memory=False)

    if df.shape[1] != len(columns):
        raise ValueError(
            f"{path.name}: file has {df.shape[1]} fields but {len(columns)} "
            "column names were supplied. Pull the current layout from the "
            "Freddie Mac User Guide - it changes between releases.")
    df.columns = columns
    df.attrs["standarderror"] = {
        "source_id": FREDDIE_META.source_id,
        "citation": FREDDIE_META.citation,
        "homepage": FREDDIE_META.homepage,
        "licence": FREDDIE_META.licence,
        "redistributable": False,
    }
    return df


def load_csv(path: str | Path, meta: SourceMeta, **kwargs) -> pd.DataFrame:
    """Generic local CSV with explicit provenance. Use for anything hand-sourced."""
    df = pd.read_csv(path, **kwargs)
    df.attrs["standarderror"] = {
        "source_id": meta.source_id, "citation": meta.citation,
        "homepage": meta.homepage, "licence": meta.licence,
        "redistributable": meta.redistributable, "local_path": str(path),
    }
    return df
