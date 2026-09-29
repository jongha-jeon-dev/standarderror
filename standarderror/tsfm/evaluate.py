r"""Scoring a forecaster on the real pool, against baselines that could win.

Three metrics, because each catches what the others miss:

`mase`
    Mean absolute error of the median, scaled by the in-sample seasonal-naive
    error of the context. 1.0 means "no better than repeating last season".
    The point-forecast number, and the one compression papers usually report.

`wql`
    Weighted quantile loss over the nine levels, scaled by the absolute
    target -- the probabilistic score Chronos and GIFT-Eval use.

`coverage`
    Share of targets inside the 10-90% band. An 80% band that holds 60% is
    wrong in a way the median can be perfectly right through.

Aggregation is by geometric mean for the scaled errors -- the convention for
MASE across series of very different difficulty, where an arithmetic mean is
one hard series wearing a costume -- and plain mean for coverage.
"""

from __future__ import annotations

import numpy as np

from standarderror.tsfm import model as tm
from standarderror.tsfm import pool as tp

Q = np.array(tm.QUANTILES)

#: Non-overlapping origins per series, by frequency. The high-frequency
#: series are few (one weekly, four monthly), so they are evaluated at many
#: origins; the annual ones are many and short, so at one.
ORIGINS = {"W": 20, "M": 20, "Q": 10, "A": 1}


def _scale(context: np.ndarray, season: int) -> float:
    m = season if len(context) > season else 1
    d = np.abs(context[m:] - context[:-m])
    d = d[np.isfinite(d)]
    s = float(d.mean()) if len(d) else float("nan")
    return s if s > 0 else float("nan")


def score(context, target, quantiles, season) -> dict:
    """One window. `quantiles`: (horizon, Q)."""
    med = quantiles[:, len(Q) // 2]
    s = _scale(context, season)
    diff = target[:, None] - quantiles
    ql = np.maximum(Q * diff, (Q - 1) * diff).sum(-1)
    lo, hi = quantiles[:, 0], quantiles[:, -1]
    return {"mase": float(np.mean(np.abs(target - med)) / s),
            "wql": float(2 * ql.sum() / len(Q) / max(np.abs(target).sum(), 1e-8)),
            "coverage": float(np.mean((target >= lo) & (target <= hi)))}


def windows(series: list, *, max_origins: int | None = None):
    """(series, context, target) windows, `ORIGINS[freq]` per series unless
    `max_origins` overrides it."""
    out = []
    for s in series:
        k = ORIGINS[s.freq] if max_origins is None else max_origins
        for o in range(min(k, s.origins())):
            c, t = s.split(o)
            out.append((s, c, t, o))
    return out


# --------------------------------------------------------------- baselines

def naive(context, horizon, season=1):
    """Repeat the last value, with quantiles from the context's own errors.

    A baseline that only produces a point is not a baseline for a model that
    produces quantiles; this one spreads its point by the empirical
    distribution of h-step naive errors in the context, widening with the
    square root of the horizon, which is what a random walk would do.
    """
    c = context[np.isfinite(context)]
    m = season if (season > 1 and len(c) > 2 * season) else 1
    base = np.array([c[-m + (i % m)] for i in range(horizon)]) if m > 1 \
        else np.repeat(c[-1], horizon)
    err = c[m:] - c[:-m]
    if len(err) < 3:
        err = np.array([0.0])
    qs = np.quantile(err - err.mean() * 0, Q)
    steps = np.sqrt(1 + np.arange(horizon) // m)
    return base[:, None] + steps[:, None] * qs[None, :]


def seasonal_naive(context, horizon, season):
    return naive(context, horizon, season)


def ets(context, horizon, season):
    """statsmodels ETS, damped additive trend, additive season where it fits.

    Intervals by simulation. Falls back to the naive forecaster when the fit
    fails or the context is too short for the seasonal model, and counts how
    often it did, because a baseline that silently becomes a different
    baseline is not the baseline the table says.
    """
    import warnings

    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    c = context[np.isfinite(context)]
    use_season = season > 1 and len(c) >= 2 * season + 4
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ETSModel(c, error="add", trend="add", damped_trend=True,
                           seasonal="add" if use_season else None,
                           seasonal_periods=season if use_season else None
                           ).fit(disp=False)
            sim = fit.simulate(horizon, repetitions=400, anchor="end",
                               rng=np.random.default_rng(0))
        return np.quantile(np.asarray(sim), Q, axis=1).T, False
    except Exception:
        return naive(context, horizon, season), True


# ------------------------------------------------------------------ driver

def evaluate(forecaster, series, *, max_origins: int | None = None) -> list[dict]:
    """Score `forecaster(contexts, horizon, season) -> (N, h, Q)` per window."""
    rows = []
    for (freq, season, h) in [(f, *tp.FREQ[f]) for f in tp.FREQ]:
        win = [w for w in windows(series, max_origins=max_origins)
               if w[0].freq == freq]
        if not win:
            continue
        qs = forecaster([w[1] for w in win], h, season)
        for (s, c, t, o), q in zip(win, qs):
            rows.append({"series": s.name, "freq": freq, "origin": o,
                         **score(c, t, q, season)})
    return rows


def model_forecaster(bundle):
    def f(contexts, horizon, season):
        return tm.forecast(bundle, contexts, horizon)
    return f


def baseline_forecaster(kind: str):
    """A forecaster for one baseline. For ETS, `f.fallbacks` counts the
    windows where the fit failed and the naive forecast stood in -- reported
    next to the ETS row, because a baseline that quietly became another
    baseline is not the baseline the table names. (It did exactly that once:
    a renamed statsmodels argument made every fit raise, and the ETS row came
    out identical to seasonal naive.)"""
    def f(contexts, horizon, season):
        if kind == "naive":
            return np.stack([naive(c, horizon, 1) for c in contexts])
        if kind == "seasonal naive":
            return np.stack([naive(c, horizon, season) for c in contexts])
        out = []
        for c in contexts:
            q, fell = ets(c, horizon, season)
            f.fallbacks += fell
            f.windows += 1
            out.append(q)
        return np.stack(out)
    f.fallbacks, f.windows = 0, 0
    return f


def _geomean(values) -> float:
    return float(np.exp(np.nanmean(np.log(np.clip(values, 1e-6, None)))))


def aggregate(rows: list[dict], by: str = "freq") -> dict:
    out = {}
    for key in sorted({r[by] for r in rows}):
        part = [r for r in rows if r[by] == key]
        out[key] = {"n": len(part),
                    "mase": _geomean([r["mase"] for r in part]),
                    "wql": _geomean([r["wql"] for r in part]),
                    "coverage": float(np.nanmean([r["coverage"] for r in part]))}
    return out


# -------------------------------------------------------------- monitoring

def monitor_slice(per_freq: int = 40, seed: int = 0):
    """A fixed small slice of the pool for the learning curve. Monitoring
    only; nothing is selected on it."""
    rng = np.random.default_rng(seed)
    pool = [s for s in tp.pool() if s.origins() >= 1]
    out = []
    for f in tp.FREQ:
        part = [s for s in pool if s.freq == f]
        idx = rng.permutation(len(part))[:per_freq]
        out += [part[i] for i in idx]
    return windows(out, max_origins=3)


def quick_mase(bundle, win) -> dict:
    out = {}
    for f in tp.FREQ:
        w = [x for x in win if x[0].freq == f]
        if not w:
            continue
        season, h = tp.FREQ[f]
        qs = tm.forecast(bundle, [x[1] for x in w], h)
        m = [score(x[1], x[2], q, season)["mase"] for x, q in zip(w, qs)]
        out[f"mase_{f}"] = float(np.exp(np.nanmean(np.log(np.clip(m, 1e-6, None)))))
    return out


def compare(rows_a, rows_b, *, metric: str = "mase", by: str = "freq",
            draws: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of two forecasters, window by window.

    Reports the geometric-mean ratio a/b (below 1 means `a` is better), the
    share of windows `a` wins, and a bootstrap interval on the ratio that
    resamples *series*, not windows: twenty origins of one CO2 series are one
    series' worth of evidence, not twenty, and resampling windows would make
    a single long series look like a large sample.
    """
    def key(r):
        return (r["series"], r["freq"], r["origin"])
    b = {key(r): r for r in rows_b}
    rng = np.random.default_rng(seed)
    out = {}
    for group in sorted({r[by] for r in rows_a}):
        pairs = [(r, b[key(r)]) for r in rows_a
                 if r[by] == group and key(r) in b
                 and np.isfinite(r[metric]) and np.isfinite(b[key(r)][metric])
                 and r[metric] > 0 and b[key(r)][metric] > 0]
        if not pairs:
            continue
        if metric == "coverage":
            d = np.array([x[metric] - y[metric] for x, y in pairs])
            stat = float(d.mean())
        else:
            d = np.array([np.log(x[metric] / y[metric]) for x, y in pairs])
            stat = float(np.exp(d.mean()))
        names = np.array([x["series"] for x, _ in pairs])
        uniq = np.unique(names)
        idx = {n: np.where(names == n)[0] for n in uniq}
        boot = []
        for _ in range(draws):
            pick = rng.choice(uniq, len(uniq))
            dd = np.concatenate([d[idx[n]] for n in pick])
            boot.append(dd.mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        if metric != "coverage":
            lo, hi = np.exp(lo), np.exp(hi)
        wins = float(np.mean([x[metric] < y[metric] for x, y in pairs])) \
            if metric != "coverage" else float("nan")
        out[group] = {"windows": len(pairs), "series": len(uniq),
                      "ratio" if metric != "coverage" else "difference": stat,
                      "low": float(lo), "high": float(hi), "a_wins": wins}
    return out
