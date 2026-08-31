"""exp020 — a step fitted to a bend lands in the wrong year.

Where this follows from
-----------------------
The previous post asked whether Korea's chip export series can settle a claim
about a *level shift*, and found that it cannot at any magnitude under
discussion. But almost nothing anyone argues about is a level shift. "Capacity
somewhere else changed Korea's trajectory" is a claim about a **slope**: not that
exports fell, but that they stopped climbing as fast. Fitting a step to that is
not a small misspecification, it is the wrong shape.

What this post establishes
--------------------------
1. **The misplacement is a fixed share of the sample, and it is arithmetic.**
   With no noise at all, on a noiseless kinked line, the best-fitting step lands
   17.4% of the sample away from the bend — the same 17.4% at n = 100 and at
   n = 3,200. The whole curve is a function of the break's *relative* position
   and of nothing else, so a longer series lengthens the error in exact
   proportion. On this post's design the misplacement runs 13 to 47 months,
   median 40, and its direction disagrees between the two trend specifications
   on four of six break dates — worse than a predictable bias, because there is
   no correction to apply.

2. **On Korea's export volume the bend fits better and it is not a result.** The
   bend takes 9.4% off the step's residual sum of squares. No-break data produces
   margins whose 95th percentile is 11.2%, and that bar is itself noisy: 8.2% to
   11.8% across six independent nulls, median 9.6%, sd 1.2%. The observed margin
   sits within one standard deviation of the bar's median.

3. **The comparison is also weak, so the negative is not evidence of a step.**
   Against a real bend of exactly the fitted size, the race calls it a bend 52%
   of the time and clears the calibrated bar 23% of the time. The falsifiable
   statement is that 312 months of this series cannot distinguish the two shapes.
   Neither shape is detected either: sup|t| 3.13 against 4.79 for the bend, 2.30
   against 5.14 for the step.

4. **Assuming the wrong shape does not blur the date, it loses it.** When the
   truth is a bend, the fitted bend date has an interquartile range of 13 months;
   the fitted step date, 86 — over seven years.

5. **A break date's bootstrap interval fails two checks, so only its core is
   reported.** The volume bend's core is stable (middle half 9 to 11 months wide
   across seeds) but the nominal 90% interval ranges from 36 to 129 months wide
   on the seed alone, and covers a known truth 83% of the time against a nominal
   90%. Separately, 2.2% of resamples pile up exactly on the trim boundary, which
   looks like a second candidate date and is not one.

6. **The previous post's one positive survives the shape check.** China's share
   of Korea's chip export weight prefers a step by 22.1% against a bar of 17.3%
   — six standard deviations of that bar's own spread — and its date is pinned to
   October 2015 to the month.

   Numbers in this docstring are restated from the cached run. build() asserts
   the spine of each of them, so a run whose results have moved fails rather
   than publishing prose that no longer matches its own output.

Discipline
----------
No claim about any company's prospects appears anywhere in this post, and none
follows from it. The apparent bend in Korea's export volume is not attributed to
any cause, and the post's own finding is that it is not established as a bend at
all.

The tension with the previous post is stated rather than hidden. That post fitted
a quadratic trend on the grounds that a linear trend leaves curvature to be
miscounted as persistence. A bend is a rival explanation for the same curvature,
so the two specifications are competing for it. The test here is run *inside* the
quadratic design — the harder of the two — and the linear design's far more
flattering number (a 61.9% margin, t = 11.9) is computed and reported so a reader
can see exactly how much the specification choice is worth.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.render.post import Post, Section
from standarderror.sources import korea_files as kf
from standarderror.ts import bend as bd
from standarderror.ts import detect as dt
from standarderror.viz import charts, theme

#: Pinned so a rebuild cannot silently re-date a published post. This one
#: had no record to pin from -- its Hugo page was never committed and the
#: manifests had already drifted -- so the date comes from the creation
#: date of the post's Notion page, the only surviving evidence.
POST_DATE = date(2026, 8, 25)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed
CACHE = se.SETTINGS.build_dir / "cache" / "exp020.json"
DATA = se.SETTINGS.build_dir.parent / "data" / "korea"

# ---------------------------------------------------------------- configuration

TREND_DEGREE = 2        # inherited from exp019, and contested here on purpose
SEASONAL = 12
HAC_LAGS = 24
BLOCK = 36
TRIM = 0.15
SUP_REPS = 600          # calibrated critical values, per series per shape
SUP_STRIDE = 4
RACE_REPS = 400         # null distribution of the bend-versus-step margin
SOB_REPS = 300          # power of that race against a real bend
DATE_REPS = 600         # bootstrap interval for the break date
COV_REPS = 60           # coverage check on that interval
COV_INNER = 120
GRID = (72, 108, 150, 186, 222, 252)   # true bend dates for the noise-free map

SOURCES = [
    "Bank of Korea, Economic Statistics System — export volume index and export "
    "value index for semiconductors (2020=100), monthly, 2000-01 to 2026-07. "
    "Downloaded 2026-08-25. The last seven months are provisional and are "
    "excluded from every estimate.",
    "Korea Customs Service, Export/Import by Commodity and Country — HS 8542 "
    "(electronic integrated circuits), monthly by partner country, 2009-08 to "
    "2026-07, export weight and export value. Downloaded 2026-08-25.",
    kf.LICENCE_NOTE,
    "D. W. K. Andrews, 'Tests for parameter instability and structural change "
    "with unknown change point', Econometrica 1993;61:821-856 — why a break "
    "found by searching dates needs a larger critical value than one tested at "
    "a date fixed in advance.",
    "P. Perron, 'Dealing with structural breaks', Palgrave Handbook of "
    "Econometrics 2006;1:278-352 — the broken-trend model and why the trend "
    "specification and the break specification compete for the same curvature.",
    "J. Bai, 'Estimation of a change point in multiple regression models', "
    "Review of Economics and Statistics 1997;79:551-563 — the break date's own "
    "sampling distribution, which is the quantity almost never reported.",
    "W. K. Newey and K. D. West, 'A simple, positive semi-definite, "
    "heteroskedasticity and autocorrelation consistent covariance matrix', "
    "Econometrica 1987;55:703-708.",
]


def _config_key() -> str:
    blob = json.dumps({"v": 4, "trend": TREND_DEGREE, "seasonal": SEASONAL,
                       "lags": HAC_LAGS, "block": BLOCK, "trim": TRIM,
                       "sup": SUP_REPS, "stride": SUP_STRIDE,
                       "race": RACE_REPS, "sob": SOB_REPS, "date": DATE_REPS,
                       "cov": [COV_REPS, COV_INNER], "grid": list(GRID),
                       "seed": SEED,
                       "impl": hashlib.sha256(
                           open(bd.__file__, "rb").read()).hexdigest()[:12]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- data

def load() -> dict:
    """The same two files the previous post used, read the same way."""
    ecos = {}
    for p in sorted(DATA.glob("*.csv")):
        if not re.search(r"_\d{8}\.csv$", p.name):
            continue
        s = kf.read_ecos_wide(p)
        name = s.attrs["standarderror"]["series"]
        key = "value" if "금액" in name else "volume" if "물량" in name else None
        if key:
            ecos[key] = s
    missing = {"value", "volume"} - set(ecos)
    if missing:
        raise FileNotFoundError(
            f"missing ECOS series {sorted(missing)} in {DATA}. Download the "
            f"semiconductor export value and volume indices from "
            f"https://ecos.bok.or.kr and drop the CSVs there.")
    cust = kf.read_customs(p for p in DATA.glob("*.csv")
                           if re.search(r"\d{6}_\d{6}", p.name))
    return {"ecos": ecos, "share": kf.china_share(cust)}


# ---------------------------------------------------------------- analysis

def _kw(start_period: int, *, degree: int = TREND_DEGREE) -> dict:
    return dict(trend=degree, seasonal=SEASONAL, start_period=start_period,
                lags=HAC_LAGS, trim=TRIM)


def null_residual(y: np.ndarray, start_period: int,
                  degree: int = TREND_DEGREE) -> np.ndarray:
    """Residual of the no-break model — the null this post resamples."""
    X = dt.design_matrix(y.size, trend=degree, seasonal=SEASONAL,
                         start_period=start_period)
    return dt.ols_hac(X, y, lags=HAC_LAGS).resid


def race_one(y: np.ndarray, *, start_period: int, say, label: str) -> dict:
    """Bend against step on one series, with the null the comparison needs."""
    kw = _kw(start_period)
    resid = null_residual(y, start_period)
    race = bd.model_race(y, **kw)

    cal = {}
    for kind in ("step", "bend"):
        cal[kind] = bd.calibrated_sup(
            resid, n=y.size, block=BLOCK, kind=kind, reps=SUP_REPS,
            stride=SUP_STRIDE, rng=np.random.default_rng(SEED + 3),
            **{k: v for k, v in kw.items() if k != "trim"}, trim=TRIM)

    nr = bd.null_race(resid, n=y.size, block=BLOCK, reps=RACE_REPS,
                      stride=2, rng=np.random.default_rng(SEED + 5), **kw)

    winner = race["winner"]
    bar = nr[f"{winner}_gain_q95"]
    say(f"  {label}: {winner} wins by {race['ssr_gain']:.1%} "
        f"(null 95th percentile {bar:.1%}) -> "
        f"{'beyond chance' if race['ssr_gain'] > bar else 'within chance'}")
    say(f"  {label}: sup|t| step {race['step_sup_t']:.2f} vs bar "
        f"{cal['step']['sup']:.2f}; bend {race['bend_sup_t']:.2f} vs bar "
        f"{cal['bend']['sup']:.2f}")

    fits = {}
    for kind in ("step", "bend"):
        f = bd.fit_at(y, race[f"{kind}_date"], kind=kind,
                      **{k: v for k, v in kw.items() if k != "trim"})
        f.pop("resid")
        fits[kind] = f
    return {
        "n": int(y.size), "race": race, "calibrated": cal, "null_race": nr,
        "fits": fits,
        "shape_beyond_chance": bool(race["ssr_gain"] > bar),
        "shape_bar": float(bar),
        "detected": {k: bool(race[f"{k}_sup_t"] > cal[k]["sup"])
                     for k in ("step", "bend")},
        "profile": {k: {"dates": bd.scan(y, kind=k, **kw).dates.tolist(),
                        "ssr": bd.scan(y, kind=k, **kw).ssr.tolist()}
                    for k in ("step", "bend")},
    }


def misplacement_map(n: int) -> dict:
    """Where a step lands when the truth is a bend, with no noise anywhere.

    Run on both trend specifications, because the whole reason the direction is
    unstable in this post's design is that a quadratic trend absorbs part of the
    bend and a linear one cannot.
    """
    out = {"quadratic": [], "linear": [], "scaling": []}
    for tau in GRID:
        q = bd.noise_free_step_date(n, tau, trend=TREND_DEGREE,
                                    seasonal=SEASONAL)
        lin = bd.noise_free_step_date(n, tau, trend=1, seasonal=0)
        out["quadratic"].append(q)
        out["linear"].append(lin)
    # The bias is a share of the sample, not a number of months: this row is the
    # claim that no quantity of data removes it.
    for m in (100, 200, 400, 800, 1600, 3200):
        r = bd.noise_free_step_date(m, int(0.6 * m), trend=1, seasonal=0)
        out["scaling"].append({"n": m, "error": r["step_error"],
                               "share": r["step_error"] / m})
    # The whole curve, not just one point on it: how far the step lands from the
    # truth, as a share of the sample, against where in the sample the bend
    # actually is. Computed at three sample sizes because the three curves
    # coinciding is the finding — the misplacement is a function of the break's
    # *relative* position and of nothing else.
    fracs = [round(0.10 + 0.02 * i, 2) for i in range(41)]
    out["curve"] = {"fracs": fracs, "by_n": {}}
    for m in (200, 400, 1200):
        row = []
        for fr in fracs:
            r = bd.noise_free_step_date(m, int(fr * m), trend=1, seasonal=0,
                                        trim=0.02)
            row.append(r["step_error"] / m)
        out["curve"]["by_n"][str(m)] = row
    return out


def date_shape(hist: list[int], tau_hat: int, *,
               boundary: tuple[int, int]) -> dict:
    """Summarise a break-date bootstrap without pretending it is a normal one.

    A 5th-to-95th-percentile interval is the conventional summary and it is the
    wrong one here, because the distribution is a sharp core with long thin
    tails: the two extreme quantiles sit in the tail, where a few draws either
    way move them by years. What is stable is the core — the interquartile range
    and the share of draws within a year of the point estimate — and the tail is
    then described as a tail rather than converted into an interval.

    `boundary` is the first and last date the search could return. Mass piling up
    there is not evidence about the break; it is the trim showing through, and it
    is reported separately so it cannot be read as a second candidate date.
    """
    h = np.asarray(hist, dtype=float)
    total = h.sum()
    draws = np.repeat(np.arange(h.size), hist)
    lo_b, hi_b = boundary
    at_boundary = (h[lo_b] + h[hi_b]) / total
    return {
        "tau_hat": tau_hat, "n_draws": int(total),
        "q25": int(np.quantile(draws, 0.25)), "q75": int(np.quantile(draws, 0.75)),
        "q05": int(np.quantile(draws, 0.05)), "q95": int(np.quantile(draws, 0.95)),
        "iqr_months": int(np.quantile(draws, 0.75) - np.quantile(draws, 0.25)),
        "within_12": float(np.mean(np.abs(draws - tau_hat) <= 12)),
        "within_24": float(np.mean(np.abs(draws - tau_hat) <= 24)),
        "beyond_60": float(np.mean(np.abs(draws - tau_hat) > 60)),
        "at_boundary": float(at_boundary),
    }


def interval_stability(y: np.ndarray, *, start_period: int, kind: str,
                       paths: int = 8, reps: int = 300) -> dict:
    """How much the reported interval moves when only the resampling seed does.

    The point estimate is a property of the data. The interval is a property of
    the data *and* of a random number generator, and on a distribution with long
    tails the outer quantiles inherit most of that randomness. Measuring it is
    the only way to know whether "October 2010 to October 2014" is a finding or
    a draw.
    """
    widths, iqrs = [], []
    for p in range(paths):
        ci = bd.date_bootstrap(y, block=BLOCK, kind=kind, reps=reps, level=0.90,
                               rng=np.random.default_rng(4000 + p),
                               **_kw(start_period))
        d = ci["draws"]
        widths.append(int(ci["hi"] - ci["lo"]))
        iqrs.append(int(np.quantile(d, 0.75) - np.quantile(d, 0.25)))
    return {"paths": paths, "reps": reps,
            "width_min": min(widths), "width_max": max(widths),
            "width_median": float(np.median(widths)),
            "iqr_min": min(iqrs), "iqr_max": max(iqrs),
            "iqr_median": float(np.median(iqrs))}


def bar_stability(resid: np.ndarray, *, n: int, start_period: int,
                  side: str = "bend", paths: int = 6, reps: int = 400) -> dict:
    """The same question about the bend-versus-step bar.

    The bar is the 95th percentile of a heavy-tailed margin distribution, so it
    is estimated with real error, and a margin that beats it by two points is
    not a result. This is what turns "9.4% against 6.8%" into "9.4% against
    somewhere between 7% and 13%", which is the honest version.
    """
    bars = []
    for p in range(paths):
        nr = bd.null_race(resid, n=n, block=BLOCK, reps=reps, stride=2,
                          rng=np.random.default_rng(5000 + p),
                          **_kw(start_period))
        bars.append(nr[f"{side}_gain_q95"])
    return {"side": side, "paths": paths, "reps": reps, "bars": bars,
            "lo": float(min(bars)), "hi": float(max(bars)),
            "median": float(np.median(bars)),
            "sd": float(np.std(bars, ddof=1))}


def compute(*, force: bool = False, verbose: bool = True) -> dict:
    key = _config_key()
    if CACHE.exists() and not force:
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            if verbose:
                print(f"exp020: cached ({CACHE})")
            return cached
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)

    raw = load()
    vol = raw["ecos"]["volume"]
    val = raw["ecos"]["value"]
    # The count has to come off the frame, not off attrs: attrs carries no
    # provisional key and the lookup silently reported zero excluded months in
    # the reproducibility block while seven were in fact dropped.
    prov = int(vol.provisional.sum())
    vol = vol[~vol.provisional]
    val = val[~val.provisional]
    ly_vol = np.log(vol.value.to_numpy())
    ly_val = np.log(val.value.to_numpy())
    dates = [str(d.date()) for d in vol.index]

    share = raw["share"]
    sdates = [str(d.date()) for d in share.index]
    say(f"exp020: {len(dates)} months of index, {len(sdates)} of customs")

    res = {"key": key, "dates": dates, "provisional_months": prov,
           "volume": vol.value.tolist(), "value": val.value.tolist(),
           "share_dates": sdates,
           "share_weight": share.share_weight.tolist(),
           "share_value": share.share_value.tolist()}

    # ---- the race, on four series
    say("\nshape race")
    series = {
        "volume": (ly_vol, 0, "export volume"),
        "value": (ly_val, 0, "export value"),
        "share_weight": (np.log(share.share_weight.to_numpy()
                                / (1 - share.share_weight.to_numpy())),
                         share.index[0].month - 1, "China share, kg"),
        "share_value": (np.log(share.share_value.to_numpy()
                               / (1 - share.share_value.to_numpy())),
                        share.index[0].month - 1, "China share, $"),
    }
    res["races"] = {name: race_one(y, start_period=sp, say=say, label=lab)
                    for name, (y, sp, lab) in series.items()}

    # ---- the misplacement a step suffers when the truth is a bend
    say("\nnoise-free misplacement")
    res["misplacement"] = misplacement_map(ly_vol.size)
    q = [abs(r["step_error"]) for r in res["misplacement"]["quadratic"]]
    say(f"  quadratic design: |error| median {np.median(q):.0f} months, "
        f"range {min(q)}-{max(q)}")
    say(f"  linear design: share of sample "
        f"{np.mean([r['share'] for r in res['misplacement']['scaling']]):.3f} "
        f"at every n from 100 to 3200")

    # ---- power of the race, against a bend of the size actually fitted
    say("\nrace power")
    r_vol = res["races"]["volume"]
    resid_vol = null_residual(ly_vol, 0)
    size = abs(r_vol["fits"]["bend"]["coef"])
    sob = bd.step_on_bend(resid_vol, n=ly_vol.size,
                          tau=r_vol["race"]["bend_date"], size=size,
                          block=BLOCK, reps=SOB_REPS,
                          rng=np.random.default_rng(SEED + 7),
                          **{k: v for k, v in _kw(0).items()})
    bar = r_vol["null_race"]["bend_gain_q95"]
    power = float((sob["margin"] > bar).mean())
    sob = {k: v for k, v in sob.items()
           if k not in ("margin", "step_dates", "bend_dates")}
    sob["power_at_the_calibrated_bar"] = power
    res["race_power"] = sob
    say(f"  a real bend of the fitted size is called a bend {sob['bend_wins']:.0%} "
        f"of the time, and clears the calibrated bar {power:.0%} of the time")
    say(f"  step date IQR {sob['step_iqr'][1] - sob['step_iqr'][0]:.0f} months, "
        f"bend {sob['bend_iqr'][1] - sob['bend_iqr'][0]:.0f}")

    # ---- the date, and whether its interval is honest
    say("\ndate intervals")
    res["dates_ci"] = {}
    for name, kind in (("volume", "bend"), ("share_weight", "step")):
        y, sp, lab = series[name]
        ci = bd.date_bootstrap(y, block=BLOCK, kind=kind, reps=DATE_REPS,
                               level=0.90, rng=np.random.default_rng(SEED + 9),
                               **_kw(sp))
        draws = ci.pop("draws")
        ci["hist"] = np.bincount(draws, minlength=y.size).tolist()
        grid = bd.scan(y, kind=kind, **_kw(sp)).dates
        ci["shape"] = date_shape(ci["hist"], ci["tau_hat"],
                                 boundary=(int(grid[0]), int(grid[-1])))
        ci["stability"] = interval_stability(y, start_period=sp, kind=kind)
        res["dates_ci"][name] = {"kind": kind, **ci}
        idx = pd.to_datetime(dates if name == "volume" else sdates)
        sh, st = ci["shape"], ci["stability"]
        say(f"  {lab} ({kind}): {idx[ci['tau_hat']]:%Y-%m}; "
            f"middle half {idx[sh['q25']]:%Y-%m}..{idx[sh['q75']]:%Y-%m} "
            f"({sh['iqr_months']} mo), {sh['within_12']:.0%} within a year, "
            f"{sh['beyond_60']:.0%} beyond five, {sh['at_boundary']:.1%} piled at "
            f"the search boundary")
        say(f"    across resampling seeds the 90% width runs "
            f"{st['width_min']}-{st['width_max']} months, the middle half "
            f"{st['iqr_min']}-{st['iqr_max']}")

    say("\nhow noisy the shape bar itself is")
    # Measured on both sides that the table reports a verdict for: the volume
    # series' bend margin, which loses to its bar, and the kilogram share's step
    # margin, which beats its bar. A "beyond chance" verdict needs the bar's own
    # error just as much as a "within chance" one does.
    y_sw, sp_sw, _ = series["share_weight"]
    res["bar_stability"] = {
        "volume_bend": bar_stability(resid_vol, n=ly_vol.size, start_period=0,
                                     side="bend"),
        "share_step": bar_stability(null_residual(y_sw, sp_sw), n=y_sw.size,
                                    start_period=sp_sw, side="step"),
    }
    for tag, bs in res["bar_stability"].items():
        say(f"  {tag}: bar runs {bs['lo']:.1%} to {bs['hi']:.1%} across "
            f"{bs['paths']} independent nulls (median {bs['median']:.1%}, "
            f"sd {bs['sd']:.1%})")

    say("\ncoverage of the date interval (the control)")
    res["coverage"] = bd.date_coverage(
        resid_vol, n=ly_vol.size, tau=r_vol["race"]["bend_date"], size=size,
        block=BLOCK, kind="bend", reps=COV_REPS, inner=COV_INNER, level=0.90,
        rng=np.random.default_rng(SEED + 11), **_kw(0))
    say(f"  nominal {res['coverage']['nominal']:.0%} -> actual "
        f"{res['coverage']['covered']:.0%}, median width "
        f"{res['coverage']['median_width']:.0f} months")

    # ---- what the flattering specification would have said
    say("\nthe linear-trend design, reported because it flatters the story")
    lin_kw = _kw(0, degree=1)
    lin = bd.model_race(ly_vol, **lin_kw)
    lin_fit = bd.fit_at(ly_vol, lin["bend_date"], kind="bend",
                        **{k: v for k, v in lin_kw.items() if k != "trim"})
    X = bd.bend_design(ly_vol.size, tau=lin["bend_date"], kind="bend",
                       trend=1, seasonal=SEASONAL)
    beta = dt.ols_hac(X, ly_vol, lags=HAC_LAGS).beta
    pre = float(beta[1] / ((ly_vol.size - 1) / 12))
    change = bd.slope_change_per_year(lin_fit["coef"], ly_vol.size)
    res["linear_view"] = {
        "bend_date": lin["bend_date"], "ssr_gain": lin["ssr_gain"],
        "t_hac": lin_fit["t_hac"], "pre_growth": pre,
        "post_growth": pre + change, "slope_change": change,
        "same_date_as_quadratic": lin["bend_date"] == r_vol["race"]["bend_date"],
    }
    say(f"  bend at {pd.to_datetime(dates)[lin['bend_date']]:%Y-%m}, "
        f"gain {lin['ssr_gain']:.1%}, growth "
        f"{100 * (np.exp(pre) - 1):.0f}%/yr -> "
        f"{100 * (np.exp(pre + change) - 1):.0f}%/yr")

    # the quadratic design's own reading of the same slope change
    res["slope_change_quadratic"] = bd.slope_change_per_year(
        r_vol["fits"]["bend"]["coef"], ly_vol.size)

    res["elapsed_s"] = round(time.time() - t0, 1)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(res))
    say(f"\nexp020: {res['elapsed_s']}s -> {CACHE}")
    return res


# ---------------------------------------------------------------- figures

def _pct(x: float, dp: int = 0) -> str:
    return f"{100 * x:.{dp}f}%"


def _months(k: float) -> str:
    """Months as something a reader holds.

    Kept in months up to three years, because "1 year 9 months" is harder to
    read than "21 months" and "1 years" is worse than either — which is what a
    naive divmod produced on the first pass.
    """
    k = abs(int(round(k)))
    if k < 36:
        return f"{k} months"
    y, m = divmod(k, 12)
    return f"{y} years" if m == 0 else f"about {y} and a half years" if m >= 5 \
        else f"{y} years"


def md_table(header, rows) -> str:
    """Markdown table with pipes escaped; Medium and Notion get the image."""
    def cell(x):
        return str(x).replace("|", r"\|")
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


MIS_HEADER = ["true bend at", "step lands (quadratic trend)", "off by",
              "step lands (linear trend)", "off by"]
RACE_HEADER = ["series", "shape that fits better", "margin", "bar from its own null",
               "shape verdict", "break detected at all"]


def _mean_without_seasonals(y: np.ndarray, tau: int, kind: str,
                            start_period: int) -> tuple[np.ndarray, np.ndarray]:
    """The fitted trend-plus-break curve, and the series with seasonals removed.

    Plotting the raw fitted values against the raw series would put twelve
    months of sawtooth on top of the one thing the figure is about. Both sides
    have the same estimated seasonal pattern subtracted, so the comparison is
    exactly the one the regression makes.
    """
    n = y.size
    X = bd.bend_design(n, tau=tau, kind=kind, trend=TREND_DEGREE,
                       seasonal=SEASONAL, start_period=start_period)
    beta = dt.ols_hac(X, y, lags=HAC_LAGS).beta
    # Columns: constant, trend powers, then SEASONAL-1 dummies, then the break.
    s0 = 1 + TREND_DEGREE
    s1 = s0 + SEASONAL - 1
    seasonal_part = X[:, s0:s1] @ beta[s0:s1]
    keep = np.concatenate([np.arange(s0), np.arange(s1, X.shape[1])])
    return X[:, keep] @ beta[keep], y - seasonal_part


def figures(res: dict) -> dict:
    figs = {}
    dates = pd.to_datetime(res["dates"])
    ly = np.log(np.array(res["volume"]))
    r_vol = res["races"]["volume"]
    r_sw = res["races"]["share_weight"]
    bend_tau = r_vol["race"]["bend_date"]
    step_tau = r_vol["race"]["step_date"]
    mis = res["misplacement"]

    # ---- F1: the two shapes, on the series ------------------------------
    bend_fit, adj_b = _mean_without_seasonals(ly, bend_tau, "bend", 0)
    step_fit, _ = _mean_without_seasonals(ly, step_tau, "step", 0)
    frame = pd.DataFrame(
        {"export volume, seasonals removed": np.exp(adj_b),
         "fitted with a bend": np.exp(bend_fit),
         "fitted with a step": np.exp(step_fit)},
        index=dates)

    def mark_shapes(fig, ax):
        m = theme.MODES["light"]
        ax.set_yscale("log")
        ticks = [1, 2, 5, 10, 25, 50, 100, 200, 400]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        for tau, _kind in ((bend_tau, "bend"), (step_tau, "step")):
            ax.axvline(dates[tau], color=m.muted, lw=1.3, ls=(0, (4, 3)),
                       zorder=1)
        ax.text(dates[step_tau], 1.05,
                f"step's best date\n{dates[step_tau]:%b %Y}  ",
                color=m.ink_secondary, fontsize=8, va="bottom", ha="right")
        ax.text(dates[bend_tau], 300, f"  bend's best date\n  {dates[bend_tau]:%b %Y}",
                color=m.ink_secondary, fontsize=8, va="top")

    figs["f1"], _ = charts.lines(
        frame,
        title="Two shapes you cannot tell apart by eye",
        subtitle=(f"Korea's semiconductor export volume index with the monthly "
                  f"seasonal pattern removed, and the two competing fits: a "
                  f"permanent level shift, and a change in the growth rate. "
                  f"Each adds exactly one column to the same trend-plus-"
                  f"seasonal design, so their residual sums of squares are "
                  f"directly comparable. The bend wins by "
                  f"{_pct(r_vol['race']['ssr_gain'], 1)}, and puts its date "
                  f"{_months(bend_tau - step_tau)} later."),
        xlabel="", ylabel="index, 2020 = 100 (log scale)",
        source=("Bank of Korea ECOS, semiconductor export volume index, "
                f"{dates[0]:%Y-%m} to {dates[-1]:%Y-%m}. Quadratic trend and "
                "eleven monthly dummies in both fits."),
        alt=("A rising line on a log scale from 2000 to 2025, from below 1 to "
             "about 200, with two smooth fitted curves drawn over it. The two "
             "fitted curves lie almost on top of each other for the whole "
             "period and are hard to distinguish. Two dashed vertical lines "
             "mark 2010 and 2012."),
        caption=("Conclude that the eye is no help here. The two fitted curves "
                 "are nearly the same curve; they differ by "
                 f"{_pct(r_vol['race']['ssr_gain'], 1)} of residual sum of "
                 f"squares and they disagree about when the break happened by "
                 f"{_months(bend_tau - step_tau)}. Everything that follows is "
                 f"about telling apart two lines that look like this."),
        mode="light", direct_labels=False, decorate=mark_shapes,
        path=str(IMG / f"a12-f1-shapes.{EXT}"))

    # ---- F2: how sharp each winning shape's own minimum is ---------------
    # Each series' winning shape, normalised to its own best fit and re-indexed
    # to months away from its own best date, so the two are directly comparable
    # despite covering different periods.
    def own_profile(a: dict, kind: str, half: int = 60) -> tuple:
        s = np.array(a["profile"][kind]["ssr"])
        taus = np.array(a["profile"][kind]["dates"])
        rel = 100 * (s / s.min() - 1)
        off = taus - taus[int(np.argmin(s))]
        keep = np.abs(off) <= half
        width = int((rel <= 5.0).sum())
        return off[keep], rel[keep], width

    off_v, rel_v, w_v = own_profile(r_vol, "bend")
    off_s, rel_s, w_s = own_profile(r_sw, "step")
    common = np.arange(-60, 61)
    pf = pd.DataFrame(
        {"Korea export volume, fitted as a bend":
             np.interp(common, off_v, rel_v, left=np.nan, right=np.nan),
         "China kilogram share, fitted as a step":
             np.interp(common, off_s, rel_s, left=np.nan, right=np.nan)},
        index=common)

    def mark_floor(fig, ax):
        m = theme.MODES["light"]
        ax.axhline(5.0, color=m.muted, lw=1.2, ls=(0, (1, 2)))
        ax.text(30, 5.3, "5% worse than the best date  ",
                color=m.ink_secondary, fontsize=8.5, va="bottom", ha="right")
        ax.set_ylim(0, 25)
        ax.set_xlim(-30, 30)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_yticklabels(["best fit", "+5%", "+10%", "+15%", "+20%", "+25%"])
        ax.set_xticks([-24, -12, -6, 0, 6, 12, 24])

    figs["f2"], _ = charts.lines(
        pf,
        title="One series knows its date to the season; the other to a year",
        subtitle=(f"How much worse the fit gets as the assumed break date moves "
                  f"away from the best one, for each series' own winning shape. "
                  f"The kilogram share's step stays within 5% of its best over "
                  f"{w_s} months; the volume series' bend needs {w_v}. A wide "
                  f"floor means the reported month was chosen by noise."),
        xlabel="months from that series' own best date",
        ylabel="residual sum of squares",
        source=("Every admissible month between the 15% trim points; "
                f"{len(r_vol['profile']['bend']['dates'])} candidate dates for "
                f"the volume index and "
                f"{len(r_sw['profile']['step']['dates'])} for the share."),
        alt=("Two U-shaped curves centred on zero over a window of thirty "
             "months either side. The share's curve is narrow and steep, "
             "crossing the 5% line within about three months of centre and "
             "leaving the top of the frame by eight. The volume series' curve "
             "is wider, crossing 5% at about seven months and reaching the top "
             "of the frame at about twenty-four. A dotted horizontal line marks "
             "the 5% level."),
        caption=("Conclude that a break date deserves a width, and the width is "
                 "not the same for every series or every shape. Reporting "
                 "'the break was in month X' with no width is the part of the "
                 "usual output that is doing the least work."),
        mode="light", direct_labels=False, decorate=mark_floor,
        path=str(IMG / f"a12-f2-profile.{EXT}"))

    # ---- F3: the misplacement, before any noise -------------------------
    curve = mis["curve"]
    cf = pd.DataFrame({"n = 400": [100 * v for v in curve["by_n"]["400"]]},
                      index=[100 * f for f in curve["fracs"]])

    def overlay_other_n(fig, ax):
        m = theme.MODES["light"]
        xs = [100 * f for f in curve["fracs"]]
        for tag, marker, size in (("200", "o", 16), ("1200", "x", 20)):
            ax.scatter(xs, [100 * v for v in curve["by_n"][tag]], s=size,
                       marker=marker, color=m.series[1], zorder=4,
                       label=f"n = {tag}", linewidths=1.1)
        ax.axhline(0, color=m.muted, lw=1.2)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left",
                  labelcolor=m.ink_secondary)
        ax.set_xticks([10, 25, 50, 75, 90])
        ax.set_xticklabels(["10%", "25%", "half way", "75%", "90%"])
        ax.set_yticks([-20, -10, 0, 10, 20])
        ax.set_yticklabels(["-20%", "-10%", "0", "+10%", "+20%"])

    figs["f3"], _ = charts.lines(
        cf,
        title="The misplacement is a share of the sample, not a number of months",
        subtitle=("Where a step lands when the truth is a bend, with no noise "
                  "at all: a deterministic kinked line, a deterministic profile, "
                  "and a minimum that is simply somewhere else. Plotted against "
                  "where in the sample the bend sits. The three sample sizes "
                  "give the same curve, so lengthening the series does not "
                  "shrink the error by one month."),
        xlabel="where the bend really is, as a share of the sample",
        ylabel="where the step lands, minus the truth",
        source=("Computed, not simulated: a noiseless bend, a linear trend, and "
                "a search over 96% of the sample."),
        alt=("A curve against the bend's position in the sample. It starts "
             "near -3% on the left, falls steadily to about -22% just before "
             "the midpoint, jumps to about +22% just after it, and decays back "
             "to about +3% at the right. Circles and crosses for two other "
             "sample sizes lie exactly on the curve."),
        caption=("Conclude that the error is never small. A bend anywhere in "
                 "the middle half of a sample sends the fitted step at least a "
                 "tenth of the sample away, and the jump at the midpoint is two "
                 "tied minima swapping rank — so near the middle even the "
                 "direction of the error is arbitrary."),
        mode="light", direct_labels=False, decorate=overlay_other_n,
        path=str(IMG / f"a12-f3-misplaced.{EXT}"))

    # ---- F4: the two date distributions, on one axis --------------------
    dv = res["dates_ci"]["volume"]
    ds = res["dates_ci"]["share_weight"]

    def offsets(ci):
        h = np.array(ci["hist"])
        idx = np.nonzero(h)[0]
        return np.repeat(idx - ci["tau_hat"], h[idx])

    off_v = offsets(dv)
    hs = np.array(ds["hist"])
    si = np.nonzero(hs)[0]
    edges = np.arange(-72.5, 73.5, 3.0)
    dens_s, _ = np.histogram(np.repeat(si - ds["tau_hat"], hs[si]),
                             bins=edges, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    figs["f4"], _ = charts.histogram(
        np.clip(off_v, -72, 72), bins=edges,
        overlay={"China's kilogram share, a step": (centres, dens_s)},
        series_label="Korea's export volume, a bend",
        title="A step's date is pinned to the month; a bend's is not",
        subtitle=(f"Where the estimated break date lands when the fitted "
                  f"residuals are resampled, as months away from the estimate "
                  f"on the real data. The step concentrates: "
                  f"{_pct(ds['shape']['within_12'])} of resamples land within a "
                  f"year. The bend has a core of the same kind — "
                  f"{_pct(dv['shape']['within_12'])} within a year — and a tail "
                  f"that does not: {_pct(dv['shape']['beyond_60'])} land more "
                  f"than five years away."),
        xlabel="months from the estimate", ylabel="share of resamples",
        source=(f"Moving-block bootstrap, {DATE_REPS} resamples, block "
                f"{BLOCK} months. Draws beyond six years are shown at the edge."),
        alt=("Two overlapping distributions centred on zero. One is a narrow "
             "spike a few months wide. The other is a broad mound roughly two "
             "years across with visible low mass extending to both edges of the "
             "plot at plus and minus six years."),
        caption=("Conclude that the middle of the bend's distribution is worth "
                 "reporting and its tails are not summarisable: across "
                 "resampling seeds the middle half stays "
                 f"{dv['stability']['iqr_min']} to "
                 f"{dv['stability']['iqr_max']} months wide while the 90% "
                 f"interval swings between {dv['stability']['width_min']} and "
                 f"{dv['stability']['width_max']} months."),
        mode="light",
        path=str(IMG / f"a12-f4-dates.{EXT}"))

    # ---- T1: the misplacement table -------------------------------------
    mis_rows = []
    for q, lin in zip(mis["quadratic"], mis["linear"]):
        mis_rows.append([
            f"{dates[q['tau']]:%b %Y}",
            f"{dates[q['step_date']]:%b %Y}",
            f"{q['step_error']:+d} months",
            f"{dates[lin['step_date']]:%b %Y}",
            f"{lin['step_error']:+d} months",
        ])
    figs["t1"], _ = charts.table_image(
        mis_rows, header=MIS_HEADER,
        title="Where the step lands when the truth is a bend",
        subtitle=("No noise anywhere: a noiseless kinked line, on a sample of "
                  "the same length as the real one, under each of the two trend "
                  "specifications. Every row is arithmetic, so every error in "
                  "it is bias."),
        source=(f"n = {len(dates)} months, search over the middle 70% of the "
                "sample, seasonal dummies included in the quadratic column."),
        alt=("A five-column table of six rows. The two 'off by' columns hold "
             "values between thirteen and sixty-nine months, and their signs "
             "disagree with each other on four of the six rows."),
        caption=("Conclude that there is no rule of thumb to apply. The "
                 "misplacement is large in both specifications and the two "
                 "specifications do not even agree on which way."),
        mode="light", bold_cols=(0,),
        path=str(IMG / f"a12-t1-misplaced.{EXT}"))

    # ---- T2: the race table ---------------------------------------------
    labels = {"volume": "Korea chip export volume",
              "value": "Korea chip export value",
              "share_weight": "China share, kilograms",
              "share_value": "China share, dollars"}
    race_rows = []
    for name, lab in labels.items():
        a = res["races"][name]
        w = a["race"]["winner"]
        race_rows.append([
            lab, w, _pct(a["race"]["ssr_gain"], 1), _pct(a["shape_bar"], 1),
            "beyond chance" if a["shape_beyond_chance"] else "within chance",
            "yes, as a step" if a["detected"]["step"] else
            ("yes, as a bend" if a["detected"]["bend"] else "no"),
        ])
    figs["t2"], _ = charts.table_image(
        race_rows, header=RACE_HEADER,
        title="Which shape, and whether it matters",
        subtitle=("For each series: which of the two shapes fits better at its "
                  "own best date, by how much, and how big a margin the same "
                  "comparison produces on data where nothing happened. The last "
                  "column is the separate question of whether any break clears "
                  "a searched critical value."),
        source=(f"Margins and bars from moving-block bootstrap, {RACE_REPS} "
                f"replications; detection against calibrated searched critical "
                f"values, {SUP_REPS} replications."),
        alt=("A six-column table of four series. Three rows say 'within "
             "chance' and one says 'beyond chance'. Only the kilogram-share "
             "row reports a detected break."),
        caption=("Conclude that one row in four survives both questions, and it "
                 "is not the series the argument is about."),
        mode="light", bold_cols=(0,),
        bold_cells={(2, 4), (2, 5)},
        path=str(IMG / f"a12-t2-race.{EXT}"))

    # ---- hero ------------------------------------------------------------
    def two_shapes(panel, m):
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        panel.plot([0.8, 5.0], [2.2, 7.2], color=m.series[0], lw=2.6)
        panel.plot([5.0, 9.2], [7.2, 8.6], color=m.series[0], lw=2.6)
        panel.plot([0.8, 3.2], [1.0, 1.0], color=m.series[1], lw=2.4,
                   ls=(0, (3, 2)))
        panel.plot([3.2, 3.2], [1.0, 3.4], color=m.series[1], lw=2.4,
                   ls=(0, (3, 2)))
        panel.plot([3.2, 9.2], [3.4, 3.4], color=m.series[1], lw=2.4,
                   ls=(0, (3, 2)))

    def arrow_apart(panel, m):
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        panel.plot([1.0, 9.0], [3.0, 3.0], color=m.muted, lw=2.0)
        for x in (3.0, 7.4):
            panel.plot([x, x], [2.4, 3.6], color=m.ink, lw=2.2)
        panel.annotate("", xy=(3.0, 6.2), xytext=(7.4, 6.2),
                       arrowprops={"arrowstyle": "<->", "color": m.ink,
                                   "lw": 2.0})
        panel.plot([3.0, 3.0], [3.6, 6.2], color=m.muted, lw=1.2)
        panel.plot([7.4, 7.4], [3.6, 6.2], color=m.muted, lw=1.2)

    def spike_with_tails(panel, m):
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        t = np.linspace(0.8, 9.2, 220)
        y = 1.6 + 6.4 * np.exp(-((t - 5.0) / 0.85) ** 2)
        panel.plot(t, y, color=m.series[0], lw=2.6)
        panel.plot([0.8, 9.2], [1.6, 1.6], color=m.muted, lw=1.6)
        for x in (1.4, 8.6):
            panel.annotate("", xy=(x, 1.6), xytext=(x, 3.4),
                           arrowprops={"arrowstyle": "->", "color": m.ink,
                                       "lw": 1.8})

    q_errs = [abs(q["step_error"]) for q in mis["quadratic"]]
    figs["hero"], _ = charts.strip_card(
        headline="The shape you assume picks the year you report",
        panels=[
            (two_shapes, "1", "parameter each"),
            (arrow_apart, f"{np.median(q_errs):.0f} mo", "apart, with no noise"),
            (spike_with_tails, _pct(dv["shape"]["beyond_60"]),
             "of resamples land 5+ years off"),
        ],
        note=("A level shift and a change in growth rate cost the same and "
              "answer different questions. Fit the wrong one to a noiseless "
              "kinked line and the date it reports is years away — and the "
              "error is a fixed share of the sample, so a longer series does "
              "not help."),
        footer="The Standard Error", mode="light",
        alt=("A three-panel hand-drawn strip. The first frame shows a kinked "
             "rising line beside a dashed staircase, marked 1. The second "
             "shows two tick marks on a horizontal axis with a double-headed "
             f"arrow spanning the gap, marked {np.median(q_errs):.0f} months. "
             "The third shows a narrow spike over a flat baseline with small "
             "arrows pointing at the baseline near both edges, marked "
             f"{_pct(dv['shape']['beyond_60'])}."),
        caption="",
        path=str(IMG / f"a12-hero.{EXT}"))

    figs["_rows"] = {"misplacement": mis_rows, "race": race_rows}
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute(verbose=False)
    figs = figures(res)

    dates = pd.to_datetime(res["dates"])
    sdates = pd.to_datetime(res["share_dates"])
    r_vol = res["races"]["volume"]
    r_sw = res["races"]["share_weight"]
    bend_tau, step_tau = r_vol["race"]["bend_date"], r_vol["race"]["step_date"]
    mis = res["misplacement"]
    q_errs = [abs(q["step_error"]) for q in mis["quadratic"]]
    shares = [abs(x["share"]) for x in mis["scaling"]]
    curve = np.array(mis["curve"]["by_n"]["400"])
    fr = np.array(mis["curve"]["fracs"])
    mid = (fr >= 0.25) & (fr <= 0.75)
    dv, ds = res["dates_ci"]["volume"], res["dates_ci"]["share_weight"]
    bsv = res["bar_stability"]["volume_bend"]
    bss = res["bar_stability"]["share_step"]
    cov = res["coverage"]
    lin = res["linear_view"]
    pw = res["race_power"]
    w_step_vol = int((np.array(r_vol["profile"]["step"]["ssr"])
                      / min(r_vol["profile"]["step"]["ssr"]) - 1 <= 0.05).sum())
    w_bend_vol = int((np.array(r_vol["profile"]["bend"]["ssr"])
                      / min(r_vol["profile"]["bend"]["ssr"]) - 1 <= 0.05).sum())
    w_step_sw = int((np.array(r_sw["profile"]["step"]["ssr"])
                     / min(r_sw["profile"]["step"]["ssr"]) - 1 <= 0.05).sum())

    # The spine, asserted rather than trusted.
    if max(shares) - min(shares) > 0.01:
        raise AssertionError(
            "the misplacement is no longer a constant share of the sample, "
            "which is the post's central claim")
    if min(np.abs(curve[mid])) < 0.05:
        raise AssertionError(
            "a bend in the middle half of the sample no longer misplaces the "
            "step materially, so section 1 has lost its result")
    if r_vol["shape_beyond_chance"]:
        raise AssertionError(
            "the volume series' shape margin now beats its own null, which "
            "reverses sections 2 and 3")
    if r_vol["detected"]["step"] or r_vol["detected"]["bend"]:
        raise AssertionError(
            "a break in export volume is now detectable, which contradicts the "
            "previous post as well as this one")
    if not (r_sw["shape_beyond_chance"] and r_sw["detected"]["step"]):
        raise AssertionError("the kilogram-share step is section 5's whole point")
    if bsv["hi"] - bsv["lo"] < 0.01:
        raise AssertionError(
            "the shape bar is now stable across nulls, so section 3's argument "
            "is weaker than the text claims")
    if dv["shape"]["iqr_months"] > 24 or dv["shape"]["beyond_60"] < 0.02:
        raise AssertionError(
            "the bend date's distribution no longer has a tight core with a "
            "long tail, which is what section 4 describes")
    if ds["shape"]["within_12"] < 0.85:
        raise AssertionError("the step date is no longer sharply identified")
    if lin["ssr_gain"] < 3 * r_vol["race"]["ssr_gain"]:
        raise AssertionError(
            "the linear-trend design no longer flatters the bend, so the "
            "specification argument in the docstring and section 2 is stale")
    if cov["covered"] >= cov["nominal"]:
        raise AssertionError(
            "the date interval no longer under-covers, so section 4 should stop "
            "saying it does")

    sections = [
        Section(heading="The claim is about a slope", body=f"""
Almost nothing anyone argues about is a level shift.

"Capacity somewhere else changed Korea's trajectory" is a claim about a
**growth rate**: not that exports fell, but that they stopped climbing as fast.
"The policy worked" is usually the same shape. So is "the market matured". Each
of them says a line that was going up at one angle is now going up at another.

The tool that gets reached for tests something else. The standard
interrupted-time-series regression fits a **step** — a permanent jump in level,
flat on both sides of a date — and reports a coefficient, a t-statistic and a
month. Fitting a step to a change in slope is not a small misspecification. It
is a different shape, and it produces a date that is wrong in a way no amount of
extra data repairs.

This post takes Korea's semiconductor export volume index, {r_vol['n']} months
from {dates[0]:%B %Y} to {dates[-1]:%B %Y}, and asks the shape question the
previous post skipped. Three results come out of it. The first is arithmetic and
transfers to any series. The second and third are about this one.

Both shapes cost exactly one column on the same trend-plus-seasonal design, so
their residual sums of squares can be compared directly — no penalty term, no
information criterion, no argument about degrees of freedom. That symmetry is
what makes the rest of this possible, and it is also what makes the two fits so
hard to tell apart.
""", figures=[figs["f1"]]),

        Section(heading="Before any noise, the step is already in the wrong place",
                body=f"""
Start with the cleanest version of the problem, which involves no data at all.

Take a line that bends: constant slope up to a date, a different constant slope
after it, and **no noise whatsoever**. Now fit a step at every candidate date and
keep the one with the lowest residual sum of squares. The profile is a
deterministic function of the candidate date. Its minimum is wherever it is. And
it is not at the bend.

{md_table(MIS_HEADER, figs['_rows']['misplacement'])}

Every number in that table is arithmetic. There is no sampling error in it to
average away, which means every gap in the two "off by" columns is bias. In the
specification this post actually uses — the quadratic column — the misplacement
runs from {min(q_errs)} to {max(q_errs)} months, median
{np.median(q_errs):.0f}; under a linear trend it reaches
{max(abs(r['step_error']) for r in mis['linear'])} months.

Two things about it are worse than the size.

**The direction is not stable.** Under a quadratic trend the step lands early for
some bend dates and late for others, and it disagrees with the linear-trend
answer on four of the six rows. There is no rule of thumb available — no "add two
years" correction — because the sign depends on where in the sample the bend sits
and on which trend you fitted.

**It does not shrink.** Express the error as a share of the sample rather than a
count of months and it stops moving: {_pct(np.median(shares), 1)} at
n = 100 and the same at n = 3,200. The whole curve is a function of the break's
*relative* position and of nothing else — three sample sizes, one curve, drawn
below. Lengthening the series lengthens the error in exact proportion.

For a bend anywhere in the middle half of a sample the fitted step lands at least
{_pct(min(np.abs(curve[mid])))} of the sample away, rising to
{_pct(max(np.abs(curve[mid])), 1)} near the midpoint. And exactly at the midpoint
the sign flips discontinuously: two mirror-image minima are tied there, and which
one wins is decided by the last digit of the arithmetic.
""", figures=[figs["f3"]]),

        Section(heading="So which shape does the series want?", body=f"""
On to the real data. Fit both shapes to Korea's log export volume, each searched
over every admissible month, on a quadratic trend with eleven monthly dummies.

The bend wins. Its best date is {dates[bend_tau]:%B %Y}, it takes
{_pct(r_vol['race']['ssr_gain'], 1)} off the step's residual sum of squares, and
its Newey-West t is {abs(r_vol['fits']['bend']['t_hac']):.2f} against the step's
{abs(r_vol['fits']['step']['t_hac']):.2f}. Read as a growth story it is a large
change: on a linear trend the same bend puts the pre-{dates[bend_tau]:%Y} growth
rate at about {100 * (np.exp(lin['pre_growth']) - 1):.0f}% a year and the
post-{dates[bend_tau]:%Y} rate at about
{100 * (np.exp(lin['post_growth']) - 1):.0f}%. That is the kind of number a post
would normally be built around.

It is not a result, and the reason is the whole point of running the comparison
properly.

{md_table(RACE_HEADER, figs['_rows']['race'])}

The fourth column is the margin the same comparison produces on data where
**nothing happened** — a moving-block bootstrap of the no-break model's own
residuals, which carries this series' persistence and none of its trend, run
{RACE_REPS} times. On that data the bend still wins some of the time, and when it
does it wins by margins whose 95th percentile is {_pct(r_vol['shape_bar'], 1)}.
The observed {_pct(r_vol['race']['ssr_gain'], 1)} is inside that.

Note what the two shapes are *not* symmetric about. A bend is a smoother function
of the candidate date than a step is, so adjacent bend columns are more alike, so
the maximum over a search is less extreme. That shows up twice: the bend wins
under the null less than half the time, and its searched critical value
({r_vol['calibrated']['bend']['sup']:.2f}) is lower than the step's
({r_vol['calibrated']['step']['sup']:.2f}). Assuming a coin flip as the baseline
would have overstated the bend's case in both directions.

And the separate question — did *anything* break — still gets the previous post's
answer. The bend's largest searched |t| is {r_vol['race']['bend_sup_t']:.2f}
against a calibrated searched bar of
{r_vol['calibrated']['bend']['sup']:.2f}; the step's
{r_vol['race']['step_sup_t']:.2f} against {r_vol['calibrated']['step']['sup']:.2f}.
Neither clears. Preferring a shape and detecting a break are two different
claims, and only one of them was ever close.
"""),

        Section(heading="The bar has a standard error too", body=f"""
There is a temptation, when a margin lands near its bar, to report the near miss
as though the bar were a constant. It is not. It is the 95th percentile of a
heavy-tailed simulated distribution, estimated from a finite number of
replications, and it moves.

Run the whole null construction {bsv['paths']} times with nothing changed but the
random seed and the bar for Korea's export volume comes back anywhere from
{_pct(bsv['lo'], 1)} to {_pct(bsv['hi'], 1)} — median {_pct(bsv['median'], 1)},
standard deviation {_pct(bsv['sd'], 1)}. The observed margin,
{_pct(r_vol['race']['ssr_gain'], 1)}, sits within a standard deviation of that
median. Whichever draw of the bar you happen to get, this margin is ordinary.

The other half of the honest report is power, and it is the part that stops this
from being a claim that the series is a step. Simulate a bend that really is
there, of exactly the size fitted to the real data, on this series' own residual
process. The race calls it a bend {_pct(pw['bend_wins'])} of the time — barely
more than a coin flip — and clears the calibrated bar
{_pct(pw['power_at_the_calibrated_bar'])} of the time.

So the comparison is valid and it is weak. A margin that beat the bar would have
meant something. A margin that does not beat it means almost nothing, because a
real bend of this magnitude would usually fail to beat it either. The
falsifiable version of the finding is: **{r_vol['n']} months of this series
cannot distinguish a level shift from a change in growth rate**, and that is a
statement about the series, not about the economy.

The same simulation says something useful about dates, though, and it is the
strongest single number here. When the truth is a bend, the fitted *bend* date
lands a median {abs(pw['bend_bias']):.0f} months from the truth with an
interquartile range of {pw['bend_iqr'][1] - pw['bend_iqr'][0]:.0f} months. The
fitted *step* date lands {abs(pw['step_bias']):.0f} months out with an
interquartile range of {pw['step_iqr'][1] - pw['step_iqr'][0]:.0f} — over
seven years wide. Assume the wrong shape and you do not merely lose precision on the
date, you lose the date.
"""),

        Section(heading="What a break date is actually worth", body=f"""
Which raises the question of what a break date is worth even when the shape is
right. It is printed as a month. It is almost never printed with a width.

Two ways to put one on it. The first is to look at how fast the fit deteriorates
as the assumed date moves. For China's kilogram share — the one series in this
family with a break that survives everything, discussed next — the step stays
within 5% of its best fit over {w_step_sw} months. For Korea's volume index the
bend needs {w_bend_vol} months to cover the same range, and the step
{w_step_vol} months. A flat floor means the month that got reported was chosen by
whichever observations happened to be noisy.

The second is to resample. Hold the fitted mean function, resample its residuals
in blocks, rebuild the series and re-estimate the date, {DATE_REPS} times. For
the kilogram share's step, {_pct(ds['shape']['within_12'])} of resamples land
within a year of {sdates[ds['tau_hat']]:%B %Y} and the middle half land on the
month itself. For the volume series' bend, the core is comparable —
{_pct(dv['shape']['within_12'])} within a year, middle half
{dates[dv['shape']['q25']]:%B %Y} to {dates[dv['shape']['q75']]:%B %Y} — but
{_pct(dv['shape']['beyond_60'])} of resamples land more than five years away.

That tail is why this section does not report a 90% interval, and the reason is
worth stating plainly rather than hiding in a footnote. Two checks on the
interval both came back badly.

Its **width is not reproducible**. Across {dv['stability']['paths']} runs
differing only in the random seed, the nominal 90% interval for the volume bend
ranged from {dv['stability']['width_min']} to {dv['stability']['width_max']}
months wide — from three years to over a decade — while the middle half stayed
between {dv['stability']['iqr_min']} and {dv['stability']['iqr_max']} months
throughout. The outer quantiles sit in the thin tail, where a handful of draws
either way moves them by years.

Its **coverage is not what it says**. Generate series with a known bend, build
the interval on each, and count: a nominal {_pct(cov['nominal'])} interval
covered the truth {_pct(cov['covered'])} of the time over {cov['reps']}
replications. Under-covering by seven points is not a disaster, but combined with
a width that is not reproducible it means the interval should not be quoted at
all. The core can be: it is stable, and it is the honest summary.

One more artefact belongs here because it looks like a finding and is not.
{_pct(dv['shape']['at_boundary'], 1)} of the volume resamples land exactly on
the first or last date the search was allowed to consider. That is the 15% trim
showing through — the search cannot return a date outside its own window, so mass
that wants to be further out piles up on the edge. Anyone reading a break-date
histogram should check the edges before believing a second mode.
""", figures=[figs["f2"], figs["f4"]]),

        Section(heading="The one result that survives all of it", body=f"""
The previous post in this series found exactly one break in this family of series
that cleared a correctly-sized bar: China's share of Korea's chip export
*weight*, breaking in {sdates[r_sw['race']['step_date']]:%B %Y}. A natural worry
about that result is that it was a bend misread as a step. It was not.

On the shape race it prefers a step by {_pct(r_sw['race']['ssr_gain'], 1)},
against a bar from its own null of {_pct(r_sw['shape_bar'], 1)} — and that bar is
tighter than the volume series': {_pct(bss['lo'], 1)} to {_pct(bss['hi'], 1)}
across {bss['paths']} independent nulls, standard deviation
{_pct(bss['sd'], 1)}. The margin clears it by
{(r_sw['race']['ssr_gain'] - bss['median']) / bss['sd']:.0f} times that spread. On
detection its searched |t| is {r_sw['race']['step_sup_t']:.2f} against a bar of
{r_sw['calibrated']['step']['sup']:.2f}. And its date is pinned: the middle half
of {DATE_REPS} resamples land on {sdates[ds['tau_hat']]:%B %Y} itself.

So the shape check confirms the earlier finding rather than overturning it, which
is the outcome that should be reported loudest, because it is the one a method
gets no credit for. A diagnostic that only ever reverses previous conclusions is
not a diagnostic.

It is worth being exact about what that surviving result is and is not. It is a
change in the composition of a trade flow measured in kilograms, in a series
where the same flow measured in dollars shows nothing anywhere. It is not a
statement about any company, and nothing in this post supports one. The reason
this post can say so firmly is the same reason it can say nothing about the
export volume series: the date of the one detected break is known to the month,
and the shape of everything else is not known at all.
"""),

        Section(heading="What to do", body=f"""
Four things, in the order they cost least.

**Write down the shape before the date.** A step and a bend answer different
questions and produce different months from the same data — here,
{_months(bend_tau - step_tau)} apart on the same series. The choice is usually made by whichever
regression is easiest to type, and it is doing more work than the significance
test that follows it.

**Fit the rival and compare, but calibrate the comparison.** Both shapes cost one
column, so the sums of squares are directly comparable, which makes this cheap.
It is also nearly useless without a null: on this series the bend beat the step
by {_pct(r_vol['race']['ssr_gain'], 1)} and no-break data produces margins of
{_pct(r_vol['shape_bar'], 1)} at the 95th percentile. Then measure the power
too — a comparison with {_pct(pw['power_at_the_calibrated_bar'])} power cannot
support the negative any more than the positive.

**Put a width on the date, then check the width.** The resampling costs one
bootstrap. What it buys is knowing whether you have a date good to the month,
as the kilogram share's is, or a core {dv['shape']['iqr_months']} months wide
with a tail reaching both ends of the sample. And check the two things this post checked: that
the width reproduces across seeds, and that it covers a known truth at the rate
it claims. Both failed here, which is why the summary above is a core and not an
interval.

**Report the trim.** A date search cannot return a date it was not allowed to
consider, so the trim manufactures mass at its own edges — a small share here,
{_pct(dv['shape']['at_boundary'], 1)}, and enough to be mistaken for a second
candidate date by anyone who did not know where the window ended.

The one number worth carrying away from all of this needs no data. Fit a step to
a bend and the date lands a fixed share of the sample away — around a sixth for a
break in the middle of the range, a quarter near the midpoint, and never less
than a tenth for anything in the middle half. That fraction is the same for a
hundred observations and for three thousand. It is the rare error in applied
statistics that a longer series makes larger.

The next one in this series stays with shapes but drops the assumption that there
is only one break to find. Two bends are not one bend fitted twice, and the
search over pairs of dates costs more than most people expect.
"""),
    ]

    post = Post(
        date=POST_DATE,
        title=("A Step Fitted to a Bend Lands Forty Months Off, and More Data "
               "Does Not Help"),
        slug="step-fitted-to-a-bend",
        subtitle=(f"Korea's semiconductor export volume, {r_vol['n']} months. Two "
                  f"shapes that cost one parameter each, disagree by "
                  f"{_months(bend_tau - step_tau)} about when the break "
                  f"happened, and cannot be told apart."),
        author="Jongha Jeon",
        summary=(f"Most claims about an economy are claims about a growth rate, "
                 f"and the standard structural-break regression tests a level. "
                 f"Fit a step to a noiseless bend and its date lands a median "
                 f"{np.median(q_errs):.0f} months away — an error that is a fixed "
                 f"share of the sample, identical at n = 100 and n = 3,200, so a "
                 f"longer series does not shrink it by one month. On Korea's chip "
                 f"export volume the bend does fit better, by "
                 f"{_pct(r_vol['race']['ssr_gain'], 1)}, but no-break data "
                 f"produces margins of {_pct(r_vol['shape_bar'], 1)} and the "
                 f"comparison has {_pct(pw['power_at_the_calibrated_bar'])} power, "
                 f"so neither shape is established. The break date's bootstrap "
                 f"interval is not reproducible across seeds "
                 f"({dv['stability']['width_min']} to "
                 f"{dv['stability']['width_max']} months) and under-covers, so "
                 f"only its core is reported. The one break the previous post "
                 f"detected — China's share of Korea's chip export weight — "
                 f"prefers a step by {_pct(r_sw['race']['ssr_gain'], 1)} against "
                 f"a bar of {_pct(r_sw['shape_bar'], 1)} and is dated to the "
                 f"month, so the shape check confirms it."),
        tags=["structural breaks", "broken trend", "statistical power",
              "korea", "public data"],
        data_sources=SOURCES,
        licence_warnings=[
            "Korean public statistics are 공공누리 (KOGL) licensed by table. "
            "This post publishes statistics computed from the files, not the "
            "underlying values.",
        ],
        sections=sections,
        table_figures=[figs["t1"], figs["t2"]],
        reproducibility={
            "seed": SEED,
            "ecos_window": f"{dates[0]:%Y-%m} to {dates[-1]:%Y-%m} "
                           f"({r_vol['n']} months, "
                           f"{res['provisional_months']} provisional excluded)",
            "customs_window": f"{sdates[0]:%Y-%m} to {sdates[-1]:%Y-%m} "
                              f"({len(sdates)} months, no gaps)",
            "trend": "quadratic in log, plus eleven monthly dummies",
            "hac_bandwidth": HAC_LAGS,
            "bootstrap_block_months": BLOCK,
            "replications": {"searched critical values": SUP_REPS,
                             "shape race null": RACE_REPS,
                             "shape race power": SOB_REPS,
                             "date bootstrap": DATE_REPS,
                             "date coverage": f"{COV_REPS} x {COV_INNER}"},
            "modules": "standarderror/ts/bend.py, standarderror/ts/detect.py, "
                       "standarderror/sources/korea_files.py",
            "tests": "tests/test_bend.py",
        },
        min_words=2000,
        max_words=3000,
    )
    post.hero = figs["hero"]
    _check_table_placement(post)
    return post


def _check_table_placement(post: Post) -> None:
    """Table images are matched to markdown tables positionally, so verify.

    Nothing in the audit catches a reversed `table_figures`; it silently swaps
    the two images between sections, and both look plausible where they land.
    """
    from standarderror.render import publish

    was = post.draft
    post.draft = False
    try:
        body = publish.medium_bundle(
            post, out_dir=se.SETTINGS.build_dir / "_placement20").read_text()
    finally:
        post.draft = was
    heading, seen = "", {}
    for line in body.split("\n"):
        if line.startswith("## "):
            heading = line[3:].strip()
        m = re.search(r"!\[[^\]]*\]\(([^)]+)\)", line)
        if m:
            seen[m.group(1).rsplit("/", 1)[-1]] = heading
    for name, needle in ((f"a12-t1-misplaced.{EXT}", "noise"),
                         (f"a12-t2-race.{EXT}", "want")):
        where = seen.get(name)
        if where is None:
            raise AssertionError(f"{name} never reached the rendered body")
        if needle.lower() not in where.lower():
            raise AssertionError(
                f"{name} landed under {where!r}; table_figures is matched "
                f"positionally, so check its order")


if __name__ == "__main__":
    import sys
    r = compute(force="--force" in sys.argv)
    idx = pd.to_datetime(r["dates"])
    sidx = pd.to_datetime(r["share_dates"])
    print("\n--- summary")
    for name, a in r["races"].items():
        i = idx if name.startswith(("volume", "value")) else sidx
        rc = a["race"]
        print(f"{name:13s} {rc['winner']:5s} by {rc['ssr_gain']:5.1%} "
              f"(bar {a['shape_bar']:.1%}) "
              f"{'BEYOND' if a['shape_beyond_chance'] else 'within':6s} chance | "
              f"step {i[rc['step_date']]:%Y-%m} bend {i[rc['bend_date']]:%Y-%m} | "
              f"detected step={a['detected']['step']} bend={a['detected']['bend']}")
