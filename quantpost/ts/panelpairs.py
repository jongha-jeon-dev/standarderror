"""Correlations between every pair of series in a panel, honestly assembled.

Given several indicators observed for many countries over the same years, the
question this post asks is: *among pairs that plausibly share no cause, how
often does a strong correlation appear anyway?*

Getting that right is mostly about which pairs are allowed to count.

* Two indicators for the **same country** share a cause — development. Korea's
  urbanisation and Korea's fertility are not an example of "nothing connects
  them", so those pairs are excluded from the headline population and reported
  separately as a comparison.
* Two series in **different countries and different indicators** are the clean
  population. Bolivia's renewable share and Korea's fertility rate have no
  mechanism linking them that is not global and diffuse.
* The **same indicator in two countries** is a third, in-between case: no direct
  link, but a shared global driver. Kept separate again.

The three groups are returned separately rather than pooled, because pooling
them would let the strongest group carry the headline.

Everything is computed on a common, fully-observed year window. A pair measured
on 1990-2000 and another on 1960-2024 are not comparable — |r| between trending
series depends strongly on length, which is the post's own finding, so letting
length vary across pairs would smear the result being measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Panel", "stack", "correlation_matrix", "pair_groups",
           "PairSummary", "summarise"]


@dataclass(frozen=True)
class Panel:
    """A rectangular block of series observed on identical years.

    `values` is `(n_series, n_years)`; `indicator` and `country` label the rows.
    """
    values: np.ndarray
    indicator: np.ndarray
    country: np.ndarray
    years: np.ndarray
    dropped_constant: tuple = ()

    def __post_init__(self):
        n = self.values.shape[0]
        if not (len(self.indicator) == len(self.country) == n):
            raise ValueError("labels must match the number of rows")
        if self.values.shape[1] != len(self.years):
            raise ValueError("values must have one column per year")

    def __len__(self) -> int:
        return self.values.shape[0]

    @property
    def n_years(self) -> int:
        return self.values.shape[1]


def stack(frames: dict, *, start: int, end: int,
          min_countries: int = 20, drop_constant: bool = True) -> Panel:
    """Build a fully-observed panel from `{indicator_name: long_frame}`.

    Each long frame has columns `iso3`, `year`, `value`. A country is kept only
    if it has **every** year in `[start, end]` for **every** indicator, so all
    series are the same length and no pair is measured on a different window
    from another. That is strict, and it is the point: the alternative is a
    length-varying sample, and length is the variable under study.

    `drop_constant` removes a country outright when **any** of its series never
    moves — Singapore, Bermuda, Gibraltar and Nauru are 100% urban for every
    year on record, and Nauru and Tuvalu report zero CO2 throughout. Correlation
    is undefined against a flat line. The country goes rather than the single
    series, because dropping one series would leave a ragged panel in which
    different countries contribute different indicators, and "an unrelated pair"
    would then mean something slightly different in each corner of the sample.
    The dropped codes are recorded on the returned panel.
    """
    if end <= start:
        raise ValueError(f"empty window {start}:{end}")
    years = np.arange(start, end + 1)

    wide: dict[str, dict[str, np.ndarray]] = {}
    for name, f in frames.items():
        sub = f[(f["year"] >= start) & (f["year"] <= end)]
        piv = (sub.pivot_table(index="iso3", columns="year", values="value",
                               aggfunc="first")
                  .reindex(columns=years))
        full = piv.dropna(axis=0, how="any")
        wide[name] = {iso: row.to_numpy(dtype=float)
                      for iso, row in full.iterrows()}

    common = set.intersection(*(set(v) for v in wide.values())) if wide else set()

    dropped: list[str] = []
    if drop_constant:
        flat = {iso for iso in common
                for v in (wide[name][iso] for name in frames)
                if float(np.std(v)) == 0.0}
        dropped = sorted(flat)
        common = common - flat

    if len(common) < min_countries:
        raise ValueError(
            f"only {len(common)} countries have complete, non-constant data for "
            f"all {len(frames)} indicators over {start}-{end}; widen the window, "
            f"drop an indicator, or lower min_countries")

    order = sorted(common)
    rows, inds, ctry = [], [], []
    for name in frames:
        for iso in order:
            rows.append(wide[name][iso])
            inds.append(name)
            ctry.append(iso)
    panel = Panel(np.asarray(rows), np.asarray(inds), np.asarray(ctry), years)
    object.__setattr__(panel, "dropped_constant", tuple(dropped))
    return panel


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    """Pearson r between every pair of rows.

    Written out rather than handed to `np.corrcoef` so a constant row raises
    here instead of producing a silent NaN that later reads as "no relationship".
    """
    x = np.asarray(values, dtype=float)
    centred = x - x.mean(axis=1, keepdims=True)
    norms = np.sqrt((centred ** 2).sum(axis=1))
    if np.any(norms == 0):
        bad = int(np.argmin(norms))
        raise ValueError(f"row {bad} is constant; r is undefined for it")
    unit = centred / norms[:, None]
    return unit @ unit.T


def pair_groups(panel: Panel) -> dict:
    """Boolean masks over the upper triangle, one per pair type."""
    n = len(panel)
    iu = np.triu_indices(n, k=1)
    same_country = panel.country[iu[0]] == panel.country[iu[1]]
    same_indicator = panel.indicator[iu[0]] == panel.indicator[iu[1]]
    return {
        "index": iu,
        "unrelated": (~same_country) & (~same_indicator),
        "same country, different indicator": same_country & (~same_indicator),
        "same indicator, different country": (~same_country) & same_indicator,
    }


@dataclass(frozen=True)
class PairSummary:
    group: str
    n_pairs: int
    median_abs_r: float
    p_over_50: float
    p_over_80: float
    p_over_90: float
    p_over_96: float

    def as_dict(self) -> dict:
        return {"group": self.group, "n_pairs": self.n_pairs,
                "median_abs_r": self.median_abs_r,
                "p_over_50": self.p_over_50, "p_over_80": self.p_over_80,
                "p_over_90": self.p_over_90, "p_over_96": self.p_over_96}


def _summary(group: str, a: np.ndarray) -> PairSummary:
    return PairSummary(group, int(a.size), float(np.median(a)),
                       float((a > 0.50).mean()), float((a > 0.80).mean()),
                       float((a > 0.90).mean()), float((a > 0.96).mean()))


def summarise(panel: Panel, *, difference: bool = False) -> dict:
    """|r| by pair type, on levels or on first differences.

    First differences are the control. If a strong correlation between two
    trending series is an artefact of both trending, differencing removes it; if
    the two genuinely move together, something survives. Running both on the
    *same* pairs is what makes the comparison mean anything.
    """
    values = np.diff(panel.values, axis=1) if difference else panel.values
    r = correlation_matrix(values)
    groups = pair_groups(panel)
    iu = groups["index"]
    flat = np.abs(r[iu])
    out = {}
    for name, mask in groups.items():
        if name == "index":
            continue
        sel = flat[mask]
        if sel.size:
            out[name] = _summary(name, sel).as_dict()
    out["_meta"] = {"n_series": len(panel), "n_years": values.shape[1],
                    "differenced": difference}
    return out


def extremes(panel: Panel, *, group: str = "unrelated", top: int = 10) -> list:
    """The most-correlated pairs in a group, with their labels.

    Reported *after* the distribution, never instead of it. A named pair is an
    illustration drawn from a measured population, and the population is the
    finding.
    """
    r = correlation_matrix(panel.values)
    groups = pair_groups(panel)
    iu, mask = groups["index"], groups[group]
    i, j = iu[0][mask], iu[1][mask]
    vals = np.abs(r[i, j])
    order = np.argsort(vals)[::-1][:top]
    return [{
        "abs_r": float(vals[k]),
        "signed_r": float(r[i[k], j[k]]),
        "a_indicator": str(panel.indicator[i[k]]),
        "a_country": str(panel.country[i[k]]),
        "b_indicator": str(panel.indicator[j[k]]),
        "b_country": str(panel.country[j[k]]),
    } for k in order]
