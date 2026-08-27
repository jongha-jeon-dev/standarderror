"""Seasonal adjustment as a variance filter, and how much it removes.

A seasonally adjusted series is usually treated as the same measurement with a
nuisance taken out. It is not: X-12/X-13 ARIMA seasonal adjustment fits *moving*
seasonal factors, so a one-off shock in a given month is partly attributed to
that month's seasonal factor and removed with it. The adjusted series therefore
carries less month-to-month variation than the raw estimate does — including
less of the variation that is sampling noise rather than season.

That matters whenever a published adjusted series is compared against a
design-based standard error, because the standard error describes the raw
estimate and the series being checked against it has been filtered.

Measuring the effect needs a benchmark for "seasonality only", and this module
uses the crudest defensible one: a **fixed** month-of-year mean, estimated on the
same block. A fixed pattern cannot absorb a one-off shock — a shock in March 2011
moves the March mean by a twelfth of itself and nothing more — so the difference
between the official adjustment and this one is variation the official filter
removed that a stable seasonal pattern does not account for.

Two things the benchmark is not:

* It is not better than X-13 at seasonal adjustment. It is worse, deliberately.
  The point is that it has no mechanism for removing noise, so anything it leaves
  behind and X-13 does not is the quantity of interest.
* It is not a claim that the official adjustment is wrong. Removing some noise
  along with the season is a defensible trade for a series read as a trend; the
  claim is only that it happens, and that the adjusted series is consequently
  smoother than the survey behind it.

Blocks with an outlier large enough to move a month-of-year mean have to be
handled explicitly. April 2020 is 14.7% against a series that had been under 4%,
and leaving it in the *fit* corrupts the April factor for every year in the
block; leaving it in the *measurement* is correct, because the noise estimator is
robust and that month really did happen. So `exclude_years` removes years from
the seasonal fit and never from the series being measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import noisescale as ns

#: Years dropped from the seasonal fit by default: the pandemic months are large
#: enough to distort a month-of-year mean for the whole block.
DEFAULT_FIT_EXCLUSIONS = (2020,)

#: Minimum length of a contiguous stretch worth estimating a scale from.
MIN_RUN = 40


def month_dummy_adjust(series: pd.Series, *,
                       exclude_years=DEFAULT_FIT_EXCLUSIONS) -> pd.Series:
    """Subtract a fixed month-of-year mean, keeping the series' overall level.

    The returned series has the same index as the input, including any months
    whose years were excluded from the fit: exclusion applies to estimating the
    twelve factors, not to what gets adjusted.
    """
    s = pd.Series(series).astype(float)
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("month_dummy_adjust needs a DatetimeIndex")
    fit = s[~s.index.year.isin(tuple(exclude_years))].dropna()
    if fit.empty:
        raise ValueError("nothing left to fit the seasonal factors on")
    counts = fit.groupby(fit.index.month).size()
    if (counts < 3).any():
        thin = sorted(int(m) for m in counts[counts < 3].index)
        raise ValueError(
            f"months {thin} have fewer than three observations in the fit "
            f"window; a month-of-year mean there is mostly noise")
    means = fit.groupby(fit.index.month).mean()
    factors = pd.Series(s.index.month, index=s.index).map(means)
    return s - factors + float(fit.mean())


def _runs(series: pd.Series, min_run: int = MIN_RUN) -> list[np.ndarray]:
    """Contiguous non-missing stretches, as arrays.

    Deliberately not reusing `sources.us_labor.contiguous_runs`: that function
    belongs to one data source and this module must not depend on one. The
    duplication is five lines and the layering is worth it.

    Runs matter because a second difference spanning a hole is a difference
    across a gap of unknown width — for the CPS, across the October 2025 month
    that was never collected.
    """
    s = pd.Series(series).astype(float)
    out, buf = [], []
    for value in s.to_numpy():
        if np.isfinite(value):
            buf.append(value)
        else:
            if len(buf) >= min_run:
                out.append(np.asarray(buf, dtype=float))
            buf = []
    if len(buf) >= min_run:
        out.append(np.asarray(buf, dtype=float))
    return out


def run_scale(series: pd.Series, *, min_run: int = MIN_RUN,
              robust: bool = True) -> float:
    """Second-difference noise scale, averaged over contiguous runs.

    Returns nan rather than raising when no run is long enough, because a caller
    sweeping decades will hit short blocks and a nan in a table is more useful
    than a stack trace.
    """
    runs = _runs(series, min_run)
    if not runs:
        return float("nan")
    scales = [ns.second_difference_scale(r, robust=robust) for r in runs]
    weights = [len(r) for r in runs]
    return float(np.average(scales, weights=weights))


def wedge(sa: pd.Series, nsa: pd.Series, *,
          exclude_years=DEFAULT_FIT_EXCLUSIONS, min_run: int = MIN_RUN,
          robust: bool = True) -> dict:
    """How much month-to-month variation the official adjustment removes.

    `sa` is the published adjusted series; `nsa` the unadjusted one for the same
    concept and months. Both are cut to their common index first, so the answer
    is never a comparison of two different samples.

    `removed` is the share of the fixed-seasonal benchmark's noise scale that the
    official series does not have. It can come out negative, and in some decades
    does: the benchmark is crude, and where the seasonal pattern is unstable the
    fixed factors leave a seasonal residual of their own. A negative value is
    evidence against the effect in that block and is reported rather than
    clipped.
    """
    sa = pd.Series(sa).astype(float)
    nsa = pd.Series(nsa).astype(float)
    common = sa.index.intersection(nsa.index)
    if len(common) < min_run:
        raise ValueError(f"only {len(common)} common months; need {min_run}")
    sa, nsa = sa.loc[common], nsa.loc[common]
    own = month_dummy_adjust(nsa, exclude_years=exclude_years)
    s_sa = run_scale(sa, min_run=min_run, robust=robust)
    s_own = run_scale(own, min_run=min_run, robust=robust)
    s_raw = run_scale(nsa, min_run=min_run, robust=robust)
    return {"n": int(len(common)),
            "start": str(common.min().date()), "end": str(common.max().date()),
            "sigma_sa": s_sa, "sigma_benchmark": s_own, "sigma_raw": s_raw,
            "ratio": float(s_own / s_sa) if s_sa else float("nan"),
            "removed": float(1.0 - s_sa / s_own) if s_own else float("nan")}


def wedge_interval(sa: pd.Series, nsa: pd.Series, *, block: int = 24,
                   reps: int = 1000, level: float = 0.90, seed: int = 0,
                   exclude_years=DEFAULT_FIT_EXCLUSIONS,
                   min_run: int = MIN_RUN, robust: bool = True) -> dict:
    """Interval for `wedge`'s `removed`, by resampling blocks of second differences.

    The obvious implementation — resample blocks of *months* and rerun `wedge` —
    is wrong, and wrong in a way that is invisible in the output. A resampled
    month index no longer lines up with the calendar, so refitting the
    month-of-year factors mixes months that are not those months, and the level
    series acquires a jump at every block join whose second difference is pure
    artefact. It produced intervals that did not contain their own point
    estimate, which is how it was caught.

    So the resampling happens one level down, on the paired second differences of
    the two adjusted series. That keeps the pairing the statistic is about,
    introduces no joins into a series that is about to be differenced again, and
    leaves the seasonal factors at their full-window values.

    The last part is a real limitation and not a small one: the interval covers
    the sampling variability of the two scales and **not** the uncertainty in the
    twelve seasonal factors, so it is too narrow. Read it as a floor on the
    spread.

    Blocks default to two years because the second differences of a monthly
    series carry a seasonal signature of their own where the adjustment was
    imperfect.
    """
    sa = pd.Series(sa).astype(float)
    nsa = pd.Series(nsa).astype(float)
    common = sa.index.intersection(nsa.index)
    if len(common) < min_run:
        raise ValueError(f"only {len(common)} common months; need {min_run}")
    sa, nsa = sa.loc[common], nsa.loc[common]
    point = wedge(sa, nsa, exclude_years=exclude_years, min_run=min_run,
                  robust=robust)
    own = month_dummy_adjust(nsa, exclude_years=exclude_years)

    # Paired second differences, run by run, so no difference spans a gap.
    pairs = []
    for a_run, b_run in zip(_runs(sa, min_run), _runs(own, min_run)):
        m = min(a_run.size, b_run.size)
        pairs.append(np.column_stack([np.diff(a_run[:m], n=2),
                                      np.diff(b_run[:m], n=2)]))
    if not pairs:
        raise ValueError("no run long enough to difference")
    d2 = np.vstack(pairs)
    n = d2.shape[0]
    if block < 12 or block > n:
        raise ValueError(f"block must lie in [12, {n}], got {block}")

    scale = ns.robust_scale
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws = np.empty(int(reps))
    for b in range(int(reps)):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        sample = d2[idx]
        s_sa = scale(sample[:, 0], robust=robust)
        s_own = scale(sample[:, 1], robust=robust)
        draws[b] = 1.0 - s_sa / s_own if s_own else np.nan
    a = draws[np.isfinite(draws)]
    lo_q = (1.0 - float(level)) / 2.0
    return {**point, "reps": int(a.size), "block": int(block),
            "level": float(level), "n_second_diff": int(n),
            "lo": float(np.quantile(a, lo_q)) if a.size else float("nan"),
            "hi": float(np.quantile(a, 1.0 - lo_q)) if a.size else float("nan")}
