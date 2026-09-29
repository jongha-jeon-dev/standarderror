r"""The real series a time-series foundation model is scored on, zero-shot.

Everything here is public, and everything here was available to this project
*without* a network call: the World Bank and BLS files already committed to
`data/`, and the public-domain datasets that ship inside the `statsmodels`
wheel. That is a narrower pool than the published benchmarks (Monash,
GIFT-Eval) and it is deliberately stated as such: numbers measured on it are
not comparable to a leaderboard, and no episode claims they are.

The model these series are scored against was pretrained on synthetic data
only, so no series below was ever seen in training. That is the property that
makes the pool useful despite being small: every forecast is genuinely
zero-shot, with no contamination question to answer.

Each series carries its frequency, the season length its naive baseline uses,
the forecast horizon, where it came from and under what terms. The horizons
follow the M4 convention -- 6 annual, 8 quarterly, 18 monthly, 13 weekly -- so
that a reader who knows that competition knows what was asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import standarderror as se

#: Frequency -> (season length, forecast horizon), after M4.
FREQ = {"W": (52, 13), "M": (12, 18), "Q": (4, 8), "A": (1, 6)}

#: The shortest context a series must leave after its test window. Shorter
#: than this and the question is no longer forecasting but guessing.
MIN_CONTEXT = {"W": 104, "M": 48, "Q": 24, "A": 24}


@dataclass(frozen=True)
class Series:
    name: str
    values: np.ndarray = field(repr=False)
    freq: str
    source: str
    licence: str

    @property
    def season(self) -> int:
        return FREQ[self.freq][0]

    @property
    def horizon(self) -> int:
        return FREQ[self.freq][1]

    def split(self, origin: int = 0):
        """Context and target for the `origin`-th window from the end.

        Origin 0 is the last `horizon` points. Origin 1 moves the whole
        window back by one horizon, and so on -- non-overlapping targets, so
        several origins give independent-ish evaluations rather than one
        evaluation repeated with a shifted edge.
        """
        h = self.horizon
        end = len(self.values) - origin * h
        return self.values[:end - h], self.values[end - h:end]

    def origins(self) -> int:
        """How many non-overlapping origins leave at least `MIN_CONTEXT`."""
        h, need = self.horizon, MIN_CONTEXT[self.freq]
        return max(0, (len(self.values) - need) // h)


def _statsmodels() -> list[Series]:
    """Public-domain series bundled in the statsmodels wheel.

    Only datasets whose own COPYRIGHT field says public domain are used; the
    ones distributed "with express permission of the original author" are
    excluded even where they would make good series, because that permission
    was given to statsmodels, not to this project.
    """
    import importlib

    import pandas as pd

    def ds(name):
        return importlib.import_module(f"statsmodels.datasets.{name}")

    class _D:
        def __getattr__(self, name):
            return ds(name)
    d = _D()

    out = []
    pd_note = "public domain (statsmodels COPYRIGHT field)"

    co2 = d.co2.load_pandas().data["co2"].astype(float)
    co2 = co2.interpolate(limit_direction="both")
    out.append(Series("co2 Mauna Loa, weekly", co2.to_numpy(), "W",
                      "statsmodels.datasets.co2", pd_note))

    elec = d.elec_equip.load_pandas().data.iloc[:, 0].astype(float)
    out.append(Series("electrical equipment orders, monthly",
                      elec.to_numpy(), "M",
                      "statsmodels.datasets.elec_equip", pd_note))

    nino = d.elnino.load_pandas().data.drop(columns="YEAR")
    out.append(Series("El Nino sea surface temperature, monthly",
                      nino.to_numpy(float).ravel(), "M",
                      "statsmodels.datasets.elnino", pd_note))

    macro = d.macrodata.load_pandas().data
    for col in ("realgdp", "realcons", "realinv", "realgovt", "realdpi",
                "cpi", "m1", "tbilrate", "unemp", "pop", "infl"):
        out.append(Series(f"US macro {col}, quarterly",
                          macro[col].to_numpy(float), "Q",
                          "statsmodels.datasets.macrodata", pd_note))

    for name, loader, col in (("Nile flow, annual", d.nile, "volume"),
                              ("sunspots, annual", d.sunspots, "SUNACTIVITY"),
                              ("US strikes, annual", d.strikes, "duration")):
        v = loader.load_pandas().data[col].to_numpy(float)
        mod = loader.__name__.split(".")[-1]
        out.append(Series(name, v, "A", f"statsmodels.datasets.{mod}",
                          pd_note))
    del pd
    return out


def _us_labor() -> list[Series]:
    from standarderror.sources import us_labor as ul
    root = Path(se.SETTINGS.repo_root) / "data" / "us_labor"
    out = []
    for f, name in (("LNS14000000.xlsx", "US unemployment rate, SA, monthly"),
                    ("LNU04000000.xlsx", "US unemployment rate, NSA, monthly")):
        s = ul.monthly_series(root / f)
        # October 2025 was never collected -- the survey did not run during
        # the federal shutdown -- so the file has a hole, not a zero. The
        # longest clean stretch is used rather than interpolating across a
        # month that was not measured; it ends in September 2025.
        run = max(ul.contiguous_runs(s), key=len)
        out.append(Series(name, run.to_numpy(float), "M", f"BLS {f[:-5]}",
                          "US government work, public domain"))
    return out


def _world_bank(*, min_run: int = 30) -> list[Series]:
    """The longest contiguous run of each economy in each committed indicator.

    Aggregates (regions, income groups) are dropped by the loader, because a
    world total and its members are not independent series and scoring both
    double-counts. Runs shorter than `min_run` years are dropped: with a
    six-year horizon they leave too little context to say anything.
    """
    from standarderror.sources import worldbank_bulk as wb
    root = Path(se.SETTINGS.repo_root) / "data" / "worldbank"
    out = []
    for z in sorted(root.glob("*.zip")):
        code = z.name.split("_DS2")[0].removeprefix("API_")
        df = wb.load(z).dropna(subset=["value"]).sort_values("year")
        for iso, g in df.groupby("iso3"):
            years, vals = g["year"].to_numpy(), g["value"].to_numpy(float)
            cuts = np.where(np.diff(years) != 1)[0] + 1
            runs = np.split(np.arange(len(years)), cuts)
            best = max(runs, key=len)
            if len(best) >= min_run:
                out.append(Series(f"{code} {iso}", vals[best], "A",
                                  f"World Bank {code}", "CC BY 4.0"))
    return out


_CACHE: list[Series] | None = None


def pool() -> list[Series]:
    """Every series, cached. Deterministic: no draw, no network."""
    global _CACHE
    if _CACHE is None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _CACHE = _statsmodels() + _us_labor() + _world_bank()
    return list(_CACHE)


def summary() -> dict:
    """Counts by frequency, and by source, of series with at least one origin."""
    by_freq, by_source = {}, {}
    for s in pool():
        if s.origins() < 1:
            continue
        by_freq[s.freq] = by_freq.get(s.freq, 0) + 1
        key = s.source.split(" ")[0] if s.source.startswith("World") else s.source
        by_source[key] = by_source.get(key, 0) + 1
    return {"by_freq": by_freq, "by_source": by_source}
