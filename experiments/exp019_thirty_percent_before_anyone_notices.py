"""exp019 — what Korea's chip export data can and cannot settle.

The news
--------
China's semiconductor build-out is the industry story of the moment, and the
question put to anyone with a spreadsheet is what it does to Korea and Taiwan.
That question gets answered, constantly, by pointing at monthly export figures.

What this post does instead
---------------------------
It asks the prior question — whether those figures can settle it — and answers
that in numbers rather than in temperament.

1. **The series has far less information than it has rows.** Korea's
   semiconductor export volume index runs 312 months. After a quadratic trend
   and monthly seasonals, its residual autocorrelation makes it worth about 28
   independent observations for estimating a level.

2. **A naive break test fires almost everywhere.** Run the textbook
   interrupted-time-series regression at every admissible date in the series and
   61% of them come back significant. There is no world in which 61% of months
   are structural breaks; the test is reading the memory cycle as a step.

3. **Nothing in twenty-six years clears a correctly-sized bar.** The largest
   apparent break is March 2015, a 20% level drop with a naive t of -6.29. Its
   Newey-West t is -2.30, against a fixed-date critical value of 3.12 and a
   search-corrected one of 5.35.

4. **The magnitude that would be needed is about 35%.** A permanent drop in
   export volume has to reach roughly a third before this design detects it at
   80% power — and lengthening the post-event window does not help, because the
   binding constraint is confounding with the cycle rather than sample size.

5. **The one thing the data does detect is not the thing being argued about.**
   China's share of Korea's chip export *weight* has a clean break in October
   2015 that survives the search correction (t = 8.32 against 7.94). The same
   share measured in *dollars* has nothing anywhere (4.55 against 7.41). Same
   file, same trade flow, two different answers.

   Numbers in this docstring are restated from the cached run and are checked
   against it by build()'s assertions; if one drifts, the run fails rather than
   publishing prose that no longer matches its own output.

Discipline
----------
No claim about any company's prospects appears anywhere in this post, and none
follows from it. Firms are named only where the data's coverage requires it.
The China-share result is about Korea's exports *to* China — China as customer —
and the post says so where it could otherwise be misread as displacement.

A methodological by-product worth its own paragraph: there is no Newey-West
bandwidth that is correctly sized at both ends, and calibrating the critical
value on the series' own residuals improves matters a great deal without fixing
them. That chain is reported in full rather than stopped at the flattering step.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import quantpost as qp
from quantpost.render.post import Post, Section
from quantpost.sources import korea_files as kf
from quantpost.ts import detect as dt
from quantpost.viz import charts, theme

IMG = qp.SETTINGS.build_dir / "img"
EXT = os.environ.get("QUANTPOST_FIG_EXT", "png")
SEED = qp.SETTINGS.seed
CACHE = qp.SETTINGS.build_dir / "cache" / "exp019.json"
DATA = qp.SETTINGS.build_dir.parent / "data" / "korea"

# ---------------------------------------------------------------- configuration

TREND_DEGREE = 2        # chosen on fit, before any break test — see trend_choice()
SEASONAL = 12
HAC_LAGS = 24
BLOCK = 36              # bootstrap block, longer than the cycle's half period
TRIM = 0.15
CAL_REPS = 2500
SUP_REPS = 800
MDE_REPS = 700
POST_WINDOWS = (12, 24, 36, 60, 96)
SUP_STRIDE = 4
CALIBRATION_PATHS = 8   # how much the calibration itself moves, measured

SOURCES = [
    "Bank of Korea, Economic Statistics System — export value index and export "
    "volume index for semiconductors (2020=100), monthly, 2000-01 to 2026-07. "
    "Downloaded 2026-08-25. The last seven months are provisional and are "
    "excluded from every estimate.",
    "Korea Customs Service, Export/Import by Commodity and Country — HS 8542 "
    "(electronic integrated circuits), monthly by partner country, 2009-08 to "
    "2026-07, export weight and export value. Downloaded 2026-08-25.",
    kf.LICENCE_NOTE,
    "C. W. J. Granger and P. Newbold (1974) and P. C. B. Phillips (1986) for why "
    "a persistent series makes an ordinary t-statistic unreliable; the previous "
    "post in this series covers that ground.",
    "D. W. K. Andrews, 'Tests for parameter instability and structural change "
    "with unknown change point', Econometrica 1993;61:821-856 — why a break "
    "found by searching dates needs a larger critical value than one tested at "
    "a date fixed in advance.",
    "W. K. Newey and K. D. West, 'A simple, positive semi-definite, "
    "heteroskedasticity and autocorrelation consistent covariance matrix', "
    "Econometrica 1987;55:703-708.",
]


def _config_key() -> str:
    blob = json.dumps({"v": 2, "trend": TREND_DEGREE, "seasonal": SEASONAL,
                       "lags": HAC_LAGS, "block": BLOCK, "trim": TRIM,
                       "cal": CAL_REPS, "sup": SUP_REPS, "mde": MDE_REPS,
                       "windows": list(POST_WINDOWS), "stride": SUP_STRIDE,
                       "paths": CALIBRATION_PATHS, "seed": SEED,
                       "impl": hashlib.sha256(
                           open(dt.__file__, "rb").read()).hexdigest()[:12]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- data

def load() -> dict:
    """The two ECOS indices and the customs China-share frame."""
    ecos = {}
    for p in sorted(DATA.glob("*.csv")):
        if not re.search(r"_\d{8}\.csv$", p.name):
            continue
        s = kf.read_ecos_wide(p)
        name = s.attrs["quantpost"]["series"]
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


def trend_choice(y: np.ndarray) -> dict:
    """Why the trend is quadratic, decided on fit and stated before the tests.

    A linear trend on a logged series asserts a constant growth rate. Korea's
    chip export volume does not have one, and the curvature a linear fit leaves
    behind gets counted as persistence — which would make every number below
    look worse for the wrong reason. Degrees are compared on residual scale and
    on whether the residual autocorrelation ever crosses zero, which is the
    signature of a trend still hiding in there.
    """
    n = y.size
    out = {}
    for deg in (1, 2, 3, 4):
        r = dt.ols_hac(dt.design_matrix(n, trend=deg, seasonal=SEASONAL), y,
                       lags=HAC_LAGS).resid
        ac = dt.autocorrelation(r, 72)
        zero = next((i for i in range(1, 73) if ac[i] < 0), None)
        trough = int(np.argmin(ac[1:73])) + 1
        out[str(deg)] = {
            "resid_sd": float(r.std()),
            "acf_first_zero": zero,
            "acf_trough_lag": trough,
            "acf_trough": float(ac[trough]),
            "implied_cycle_months": 2 * trough,
            "n_eff": dt.effective_sample_size(r, HAC_LAGS),
        }
    return out


# ---------------------------------------------------------------- analysis

def analyse(y: np.ndarray, *, start_period: int, say, label: str) -> dict:
    n = y.size
    X = dt.design_matrix(n, trend=TREND_DEGREE, seasonal=SEASONAL,
                         start_period=start_period)
    resid = dt.ols_hac(X, y, lags=HAC_LAGS).resid
    scan = dt.placebo_scan(y, trend=TREND_DEGREE, seasonal=SEASONAL,
                           start_period=start_period, lags=HAC_LAGS, trim=TRIM)
    say(f"  {label}: placebo naive {scan['share_ols']:.1%} / "
        f"hac {scan['share_hac']:.1%}")

    cal = dt.calibrated_critical_value(
        resid, n_pre=n - 36, n_post=36, block=BLOCK, reps=CAL_REPS,
        trend=TREND_DEGREE, seasonal=SEASONAL, start_period=start_period,
        lags=HAC_LAGS, rng=np.random.default_rng(SEED))

    # searching 200-odd dates costs a bigger critical value than testing one
    rng = np.random.default_rng(SEED + 1)
    lo, hi = int(np.floor(TRIM * n)), int(np.ceil((1 - TRIM) * n))
    sup = np.empty(SUP_REPS)
    for i in range(SUP_REPS):
        u = dt.moving_block_bootstrap(resid, block=BLOCK, size=n, rng=rng)
        sup[i] = max(abs(dt.break_test(u, tau, trend=TREND_DEGREE,
                                       seasonal=SEASONAL,
                                       start_period=start_period,
                                       lags=HAC_LAGS)["t_hac"])
                     for tau in range(lo, hi, SUP_STRIDE))
    sup_cv = float(np.quantile(sup, 0.95))
    say(f"  {label}: fixed cv {cal['critical']:.2f}, sup cv {sup_cv:.2f}, "
        f"observed max {scan['max_abs_t_hac']:.2f}")

    top = sorted(scan["rows"], key=lambda d: -abs(d["t_hac"]))[:6]
    return {"resid": resid.tolist(), "resid_sd": float(resid.std()),
            "n": n, "n_eff": dt.effective_sample_size(resid, HAC_LAGS),
            "vif": dt.variance_inflation(resid, HAC_LAGS),
            "scan": {k: v for k, v in scan.items() if k != "rows"},
            "calibration": cal, "sup_cv": sup_cv,
            "detected": bool(scan["max_abs_t_hac"] > sup_cv),
            "top": top}


def mde_curve(resid: np.ndarray, *, n: int, critical: float,
              start_period: int, say, label: str) -> dict:
    out = {}
    for post in POST_WINDOWS:
        m = dt.minimum_detectable_shift(
            np.asarray(resid), n_pre=n - post, n_post=post, block=BLOCK,
            target=0.80, reps=MDE_REPS, hi=1.5, critical=critical,
            trend=TREND_DEGREE, seasonal=SEASONAL, start_period=start_period,
            lags=HAC_LAGS)
        out[str(post)] = m
        say(f"  {label} post={post:>3}mo  MDE {m['mde']:.3f} log "
            f"({np.exp(-m['mde']) - 1:+.1%})")
    return out


def calibration_stability(resid: np.ndarray, *, n: int, start_period: int,
                          say) -> dict:
    """How much the calibrated critical value moves across resampled histories.

    Calibrating on one series' residuals inherits that series' sampling error.
    Reporting the critical value without this spread would overstate what the
    correction buys, which is the failure mode the whole post is about.
    """
    rng = np.random.default_rng(SEED + 7)
    cvs = []
    for k in range(CALIBRATION_PATHS):
        boot = dt.moving_block_bootstrap(np.asarray(resid), block=BLOCK,
                                         size=n, rng=rng)
        cvs.append(dt.calibrated_critical_value(
            boot, n_pre=n - 36, n_post=36, block=BLOCK, reps=600,
            trend=TREND_DEGREE, seasonal=SEASONAL, start_period=start_period,
            lags=HAC_LAGS, rng=np.random.default_rng(500 + k))["critical"])
    cvs = np.array(cvs)
    say(f"  calibration spread: {cvs.mean():.2f} +- {cvs.std():.2f} "
        f"({cvs.min():.2f}..{cvs.max():.2f})")
    return {"mean": float(cvs.mean()), "sd": float(cvs.std()),
            "min": float(cvs.min()), "max": float(cvs.max()),
            "paths": CALIBRATION_PATHS}


def bandwidth_table(say) -> dict:
    """Size of the Newey-West test across bandwidths, on known processes.

    Simulated rather than taken from the data, because size is only defined
    against a known truth. The point of the table is that the two columns fail
    at opposite ends and no row is good at both.
    """
    rng = np.random.default_rng(SEED + 11)
    n, reps = 312, 500
    out = {}
    for lags in (0, 6, 12, 24, 48):
        row = {}
        for rho in (0.0, 0.5, 0.9):
            hits = 0
            for _ in range(reps):
                if rho == 0.0:
                    y = rng.normal(size=n)
                else:
                    y = np.zeros(n)
                    e = rng.normal(size=n)
                    for i in range(1, n):
                        y[i] = rho * y[i - 1] + e[i]
                hits += abs(dt.break_test(y, n // 2, lags=lags)["t_hac"]) > 1.96
            row[str(rho)] = hits / reps
        out[str(lags)] = row
        say(f"  lags {lags:>2}: white {row['0.0']:.1%}  "
            f"AR(.5) {row['0.5']:.1%}  AR(.9) {row['0.9']:.1%}")
    return out


def compute(*, force: bool = False, verbose: bool = True) -> dict:
    key = _config_key()
    if not force and CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            return cached
    t0 = time.time()

    def say(*a):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)

    raw = load()
    vol_all, val_all = raw["ecos"]["volume"], raw["ecos"]["value"]
    vol = vol_all[~vol_all.provisional]
    val = val_all[~val_all.provisional]
    if not vol.index.equals(val.index):
        raise ValueError("the two ECOS series do not share a monthly index")
    dates = [str(d.date()) for d in vol.index]
    say(f"ECOS final data {dates[0]}..{dates[-1]}, n={len(vol)}; "
        f"{int(vol_all.provisional.sum())} provisional months excluded")

    ly_vol = np.log(vol.value.to_numpy())
    ly_val = np.log(val.value.to_numpy())
    say("choosing the trend degree")
    trend = trend_choice(ly_vol)

    say("volume: placebo scan, calibration, sup critical value")
    a_vol = analyse(ly_vol, start_period=vol.index[0].month - 1, say=say,
                    label="volume")
    say("value: the same")
    a_val = analyse(ly_val, start_period=val.index[0].month - 1, say=say,
                    label="value")

    say("minimum detectable shift, volume")
    mde_vol = mde_curve(a_vol["resid"], n=len(vol),
                        critical=a_vol["calibration"]["critical"],
                        start_period=vol.index[0].month - 1, say=say,
                        label="volume")
    say("minimum detectable shift, value")
    mde_val = mde_curve(a_val["resid"], n=len(val),
                        critical=a_val["calibration"]["critical"],
                        start_period=val.index[0].month - 1, say=say,
                        label="value")

    say("how stable is the calibration itself")
    stab = calibration_stability(a_vol["resid"], n=len(vol),
                                 start_period=vol.index[0].month - 1, say=say)

    say("size of Newey-West across bandwidths")
    band = bandwidth_table(say)

    share = raw["share"]
    say(f"customs share frame {share.index[0]:%Y-%m}..{share.index[-1]:%Y-%m}, "
        f"n={len(share)}")
    a_share = {}
    for tag, col in (("weight", "share_weight"), ("value", "share_value")):
        s = share[col].to_numpy()
        say(f"China share by {tag}")
        a_share[tag] = analyse(np.log(s / (1 - s)),
                               start_period=share.index[0].month - 1,
                               say=say, label=f"share/{tag}")
        a_share[tag]["level_range"] = [float(s.min()), float(s.max())]
        a_share[tag]["level_mean"] = float(s.mean())

    share = share.assign(
        uv_ratio=share.unit_value_country / share.unit_value_total)
    annual = (share[["share_weight", "share_value", "uv_ratio"]]
              .groupby(share.index.year).mean())

    out = {
        "key": key,
        "dates": dates,
        "volume": vol.value.tolist(), "value": val.value.tolist(),
        "provisional_months": int(vol_all.provisional.sum()),
        "trend_choice": trend,
        "analysis": {"volume": a_vol, "value": a_val},
        "mde": {"volume": mde_vol, "value": mde_val},
        "calibration_stability": stab,
        "bandwidth": band,
        "share": {
            "dates": [str(d.date()) for d in share.index],
            "share_weight": share.share_weight.tolist(),
            "share_value": share.share_value.tolist(),
            "analysis": a_share,
            "annual": {str(y): {"weight": float(r.share_weight),
                                "value": float(r.share_value),
                                "uv_ratio": float(r.uv_ratio)}
                       for y, r in annual.iterrows()},
            "meta": share.attrs["quantpost"],
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    say("done")
    return out


# ---------------------------------------------------------------- figures

def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _drop(log_magnitude: float) -> str:
    """A log shift's size, as a positive percentage fall.

    Takes the magnitude so the caller cannot flip the sign by passing a
    negative coefficient — which is exactly what happened, and it printed 54%
    where the answer was 35%. Nothing in the audit or the tests catches that;
    it was caught by looking at the rendered card.
    """
    return f"{100 * (1 - np.exp(-abs(log_magnitude))):.0f}%"


def md_table(header, rows) -> str:
    """Markdown table with pipes escaped; Medium gets the image instead."""
    def cell(x):
        return str(x).replace("|", r"\|")
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


BAND_HEADER = ["Newey-West bandwidth", "white noise", "AR(0.5)", "AR(0.9)"]
LADDER_HEADER = ["series", "largest |t|, naive", "largest |t|, corrected",
                 "bar for one date", "bar after searching dates", "verdict"]


def figures(res: dict) -> dict:
    import matplotlib.pyplot as plt

    figs = {}
    dates = pd.to_datetime(res["dates"])
    vol = np.array(res["volume"])
    val = np.array(res["value"])
    a_vol = res["analysis"]["volume"]
    a_val = res["analysis"]["value"]
    top = a_vol["top"][0]
    break_date = dates[top["break_at"]]

    # ---- F1: the series the argument is conducted on --------------------
    frame = pd.DataFrame({"export volume index": vol,
                          "export value index": val}, index=dates)

    def mark_break(fig, ax):
        m = theme.MODES["light"]
        ax.set_yscale("log")
        ax.axvline(break_date, color=m.muted, lw=1.6, ls=(0, (4, 3)), zorder=1)
        ax.text(break_date, vol.max() * 1.35,
                f"  the most break-like month\n  in 26 years: "
                f"{break_date:%B %Y}",
                color=m.ink_secondary, fontsize=8.5, va="top")
        # ticks have to span the whole range: the volume index starts below 1
        # and hard-coding them from 25 up left the lower two-thirds of a log
        # axis unlabelled, which is unreadable rather than merely untidy
        ticks = [1, 2, 5, 10, 25, 50, 100, 200, 400]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])

    figs["f1"], _ = charts.lines(
        frame,
        title="Korea's chip exports, as the argument sees them",
        subtitle=(f"Bank of Korea indices, 2020 = 100, {dates[0]:%Y} to "
                  f"{dates[-1]:%Y}, log scale. Both series grow and both swing: "
                  f"year-on-year, volume has a standard deviation of 24% and "
                  f"value 39%. The two lines converge because the price per "
                  f"unit fell by a factor of about twenty-four over the period; "
                  f"that price is what makes the value index the noisier of the "
                  f"two. Any claim about displacement has to be visible against "
                  f"these swings."),
        xlabel="", ylabel="index, 2020 = 100 (log scale)",
        source=("Bank of Korea ECOS, semiconductor export volume and value "
                "indices. Seven provisional months to 2026-07 are excluded."),
        alt=("Two rising lines on a log scale from 2000 to 2025. The volume "
             "line starts below 1 and ends near 220, climbing far more steeply "
             "than the value line, which starts near 22 and ends near 250. The "
             "gap between them closes steadily. Both carry large multi-year "
             "swings. A dashed vertical line marks early 2015."),
        caption=(f"Conclude that the swings are the problem. The vertical line "
                 f"marks {break_date:%B %Y}, the date at which a standard break "
                 f"test finds the largest step anywhere in the series — a "
                 f"{_drop(top['shift'])} drop in volume with a t-statistic of "
                 f"{abs(top['t_ols']):.1f}. The rest of the post is about "
                 f"whether that number means anything."),
        mode="light", direct_labels=False, decorate=mark_break,
        path=str(IMG / f"a11-f1-series.{EXT}"))

    # ---- F2: the placebo scan -------------------------------------------
    rows = dt.placebo_scan(np.log(vol), trend=TREND_DEGREE, seasonal=SEASONAL,
                           start_period=dates[0].month - 1,
                           lags=HAC_LAGS, trim=TRIM)["rows"]
    xs = dates[[r["break_at"] for r in rows]]
    naive = np.abs([r["t_ols"] for r in rows])
    hac = np.abs([r["t_hac"] for r in rows])
    scan_frame = pd.DataFrame(
        {"the test as usually run": naive,
         "with a correct standard error": hac}, index=xs)

    def mark_bars(fig, ax):
        m = theme.MODES["light"]
        for level, lab, style in (
                (1.96, "1.96, the bar people use", (0, (4, 3))),
                (a_vol["calibration"]["critical"],
                 "bar for a date fixed in advance", (0, (1, 2))),
                (a_vol["sup_cv"], "bar after searching every date", "-")):
            ax.axhline(level, color=m.muted, lw=1.5, ls=style, zorder=1)
            # right-aligned at the end of the series, where both curves are low
            # — placing these on the left ran the text straight through the
            # naive curve's two peaks
            ax.text(xs[-1], level + 0.12, lab + "  ", color=m.ink_secondary,
                    fontsize=8.5, va="bottom", ha="right")
        ax.set_ylim(0, max(8.0, naive.max() * 1.05))

    figs["f2"], _ = charts.lines(
        scan_frame,
        title="A structural break turns up at 61% of all possible dates",
        subtitle=(f"The absolute t-statistic on a permanent level shift, fitted "
                  f"separately at every one of {len(rows)} admissible months. "
                  f"With the textbook standard error, "
                  f"{_pct(a_vol['scan']['share_ols'])} of them clear 1.96. "
                  f"With a Newey-West standard error, "
                  f"{_pct(a_vol['scan']['share_hac'])} do — and against a "
                  f"critical value that accounts for having searched, none do."),
        xlabel="", ylabel="|t| on the level shift",
        source=("Korea semiconductor export volume index, log, quadratic trend "
                "and monthly seasonals. Critical values simulated from the "
                "series' own residuals."),
        alt=("Two curves against date from 2004 to 2022. The upper curve, the "
             "naive test, rises above 6 in the mid-2010s and spends most of the "
             "period above the lowest reference line. The lower curve stays "
             "below 2.6 throughout. Three horizontal reference lines sit at "
             "1.96, about 3.1 and about 5.4."),
        caption=(f"Conclude that the usual test is not measuring breaks. It "
                 f"reports one at three months in five, on a series in which "
                 f"nothing clears a bar built for the way the search was "
                 f"actually conducted. The corrected curve peaks at "
                 f"{a_vol['scan']['max_abs_t_hac']:.2f} against a searched bar "
                 f"of {a_vol['sup_cv']:.2f}."),
        mode="light", direct_labels=False, decorate=mark_bars,
        path=str(IMG / f"a11-f2-placebo.{EXT}"))

    # ---- F3: how big a change would have to be ---------------------------
    mde = pd.DataFrame(
        {"export volume": [100 * (1 - np.exp(-res["mde"]["volume"][str(w)]["mde"]))
                           for w in POST_WINDOWS],
         "export value": [100 * (1 - np.exp(-res["mde"]["value"][str(w)]["mde"]))
                          for w in POST_WINDOWS]},
        index=list(POST_WINDOWS))

    def mde_axes(fig, ax):
        ax.set_ylim(0, 60)
        ax.set_yticks([0, 15, 30, 45, 60])
        ax.set_yticklabels(["0", "15%", "30%", "45%", "60%"])
        ax.set_xticks(list(POST_WINDOWS))
        ax.set_xticklabels([str(w) for w in POST_WINDOWS])

    figs["f3"], _ = charts.lines(
        mde,
        title="More data after the event does not make the event easier to see",
        subtitle=("The smallest permanent drop this design detects at 80% "
                  "power, against how many months of post-event data are "
                  "available. Both curves are flat: the binding constraint is "
                  "confounding with the cycle, not the number of observations."),
        xlabel="months of data after the event",
        ylabel="smallest detectable permanent drop",
        source=("Simulated by moving-block bootstrap of the fitted residuals, "
                f"{MDE_REPS} replications per point, against the calibrated "
                "critical value."),
        alt=("Two roughly flat lines against post-event window length from 12 "
             "to 96 months. The volume line sits near 30 to 40 percent, the "
             "value line about eight points above it, and neither falls as the "
             "window lengthens."),
        caption=("Conclude that waiting does not help. Korea's chip export "
                 "volume would have to drop by about a third, permanently, "
                 "before this design could tell it apart from an ordinary turn "
                 "of the cycle — and the value index, which is the series "
                 "public commentary actually quotes, is harder still."),
        mode="light", direct_labels=False, decorate=mde_axes,
        path=str(IMG / f"a11-f3-mde.{EXT}"))

    # ---- F4: the one detectable thing, and its twin ----------------------
    sdates = pd.to_datetime(res["share"]["dates"])
    sw = np.array(res["share"]["share_weight"]) * 100
    sv = np.array(res["share"]["share_value"]) * 100
    a_sw = res["share"]["analysis"]["weight"]
    a_sv = res["share"]["analysis"]["value"]
    wtop = a_sw["top"][0]
    wbreak = sdates[wtop["break_at"]]
    share_frame = pd.DataFrame(
        {"measured in kilograms": sw, "measured in dollars": sv}, index=sdates)

    def mark_share(fig, ax):
        m = theme.MODES["light"]
        ax.axvline(wbreak, color=m.muted, lw=1.6, ls=(0, (4, 3)), zorder=1)
        ax.text(wbreak, 54, f"  {wbreak:%B %Y}", color=m.ink_secondary,
                fontsize=8.5, va="top")
        ax.set_ylim(15, 57)
        ax.set_yticks([20, 30, 40, 50])
        ax.set_yticklabels(["20%", "30%", "40%", "50%"])

    figs["f4"], _ = charts.lines(
        share_frame,
        title="The same trade flow, two measures, two different answers",
        subtitle=(f"China's share of Korea's HS 8542 exports. By weight the "
                  f"break at {wbreak:%B %Y} reaches "
                  f"|t| = {a_sw['scan']['max_abs_t_hac']:.2f} against a searched "
                  f"bar of {a_sw['sup_cv']:.2f} and is real. By value nothing "
                  f"anywhere reaches "
                  f"{a_sv['scan']['max_abs_t_hac']:.2f} against "
                  f"{a_sv['sup_cv']:.2f}."),
        xlabel="", ylabel="China's share of Korea's chip exports",
        source=("Korea Customs Service, HS 8542 monthly exports by partner, "
                "2009-08 to 2026-07. Shares tested on the logit scale."),
        alt=("Two noisy lines from 2009 to 2026. The dollar-share line climbs "
             "to about 50% by 2015, stays roughly between 38% and 52% until "
             "2022, falls to about 28% by 2025 and then rebounds to 41%. The "
             "kilogram-share line peaks near 37% in 2014, drops to the high 20s "
             "after 2015 and drifts to the low 20s, with no rebound at the end. "
             "A dashed vertical line marks late 2015."),
        caption=("Conclude that the measure chooses the story. Kilograms say "
                 "something changed in late 2015 and then held. Dollars show a "
                 "different shape entirely — a plateau to 2022, a fall, and a "
                 "rebound in the last year — and none of it survives the test. "
                 "Both come from the same customs file. Note also that this is "
                 "Korea selling to China, not competing with it; the limits "
                 "section says why that distinction matters here."),
        mode="light", direct_labels=False, decorate=mark_share,
        path=str(IMG / f"a11-f4-share.{EXT}"))

    # ---- T1: no bandwidth works ------------------------------------------
    band_rows = [[f"{lags} lags" if lags else "0 lags (White)",
                  _pct(res["bandwidth"][str(lags)]["0.0"]),
                  _pct(res["bandwidth"][str(lags)]["0.5"]),
                  _pct(res["bandwidth"][str(lags)]["0.9"])]
                 for lags in (0, 6, 12, 24, 48)]
    figs["t1"], _ = charts.table_image(
        band_rows, header=BAND_HEADER,
        title="There is no bandwidth that is right at both ends",
        subtitle=("How often a Newey-West t-test rejects at the 5% level when "
                  "nothing has happened, on simulated series of the same length "
                  "as the real one. A short bandwidth fails on persistent data; "
                  "a long one fails on quiet data; no row is close to 5% in "
                  "every column."),
        source=("Simulated, 500 replications per cell, n = 312, break tested at "
                "a date fixed in advance."),
        alt=("A four-column table of rejection rates by bandwidth. The white "
             "noise column rises from 5% to 19% as the bandwidth grows; the "
             "AR(0.9) column falls from 63% to 21% and then rises again."),
        caption=("Conclude that the standard correction cannot be tuned into "
                 "correctness here — which is why every critical value in this "
                 "post is simulated from the series' own residuals instead of "
                 "read off a table."),
        mode="light", bold_cols=(0,),
        path=str(IMG / f"a11-t1-bandwidth.{EXT}"))

    # ---- T2: the ladder ---------------------------------------------------
    ladder = []
    for label, a in (("Korea chip export volume", a_vol),
                     ("Korea chip export value", a_val),
                     ("China share, kilograms", a_sw),
                     ("China share, dollars", a_sv)):
        ladder.append([
            label,
            f"{a['scan']['max_abs_t_ols']:.2f}",
            f"{a['scan']['max_abs_t_hac']:.2f}",
            f"{a['calibration']['critical']:.2f}",
            f"{a['sup_cv']:.2f}",
            "detected" if a["detected"] else "not detected",
        ])
    figs["t2"], _ = charts.table_image(
        ladder, header=LADDER_HEADER,
        title="Four series, one detected break",
        subtitle=("The largest level shift found anywhere in each series, "
                  "against three bars: the 1.96 everyone uses, a critical value "
                  "calibrated for a date fixed in advance, and one that also "
                  "pays for having searched every date."),
        source=("Critical values simulated from each series' own residuals by "
                f"moving-block bootstrap, {SUP_REPS} replications."),
        alt=("A six-column table of four series. The naive column ranges from "
             "about 5 to over 9; the corrected column from 2.3 to 8.3. Only the "
             "kilogram share row ends in 'detected'."),
        caption=("Conclude that the naive column would have declared all four. "
                 "One survives. It is not the series anyone is arguing about."),
        mode="light", bold_cols=(0,),
        path=str(IMG / f"a11-t2-ladder.{EXT}"))

    # ---- hero -------------------------------------------------------------
    def broken_everywhere(panel, m):
        panel.set_xlim(0, 10); panel.set_ylim(0, 10)
        t = np.linspace(0.8, 9.2, 120)
        y = 5 + 2.2 * np.sin(2 * np.pi * (t - 0.8) / 5.2)
        panel.plot(t, y, color=m.series[0], lw=2.6)
        for x in (2.1, 3.4, 4.6, 6.0, 7.3, 8.5):
            panel.plot([x, x], [1.2, 8.8], color=m.series[1], lw=1.8,
                       ls=(0, (2, 2)))
        panel.plot([0.8, 0.8], [1.0, 9.0], color=m.ink, lw=2.0)

    def short_ruler(panel, m):
        panel.set_xlim(0, 10); panel.set_ylim(0, 10)
        panel.plot([0.8, 9.2], [6.6, 6.6], color=m.muted, lw=2.2)
        for i in range(13):
            x = 0.8 + i * (8.4 / 12)
            panel.plot([x, x], [6.6, 7.2], color=m.muted, lw=1.4)
        panel.plot([0.8, 1.6], [3.6, 3.6], color=m.ink, lw=3.0)
        panel.plot([0.8, 0.8], [3.1, 4.1], color=m.ink, lw=2.4)
        panel.plot([1.6, 1.6], [3.1, 4.1], color=m.ink, lw=2.4)

    def big_drop(panel, m):
        from matplotlib.patches import Rectangle
        panel.set_xlim(0, 10); panel.set_ylim(0, 10)
        panel.add_patch(Rectangle((1.6, 1.2), 2.6, 7.2, fc=m.series[0],
                                  ec=m.ink, lw=1.8))
        panel.add_patch(Rectangle((5.8, 1.2), 2.6, 4.8, fc=m.series[1],
                                  ec=m.ink, lw=1.8))
        panel.annotate("", xy=(7.1, 6.4), xytext=(7.1, 8.6),
                       arrowprops={"arrowstyle": "<->", "color": m.ink,
                                   "lw": 2.0})
        panel.plot([1.4, 8.6], [1.2, 1.2], color=m.ink, lw=2.0)

    figs["hero"], _ = charts.strip_card(
        headline="The data cannot answer the question it is being asked",
        panels=[
            (broken_everywhere, _pct(a_vol["scan"]["share_ols"]),
             "of dates look broken"),
            (short_ruler, f"{a_vol['n_eff']:.0f}", "real observations in 312"),
            (big_drop, _drop(res["mde"]["volume"]["36"]["mde"]),
             "drop needed to see it"),
        ],
        note=("Korea's semiconductor export volume swings enough on its own "
              "that a standard break test finds a structural break at three "
              "months in five. Nothing in twenty-six years survives a critical "
              "value built for the way the search was actually run."),
        footer="quantpost", mode="light",
        alt=("A three-panel hand-drawn strip. The first frame shows a wave "
             "inside an axis crossed by six dashed vertical lines, marked "
             f"{_pct(a_vol['scan']['share_ols'])}. The second shows a long "
             "ruler above a very short measuring bracket, marked "
             f"{a_vol['n_eff']:.0f}. The third shows a tall bar beside a much "
             "shorter one with an arrow spanning the difference, marked "
             f"{_drop(res['mde']['volume']['36']['mde'])}."),
        caption="",
        path=str(IMG / f"a11-hero.{EXT}"))
    figs["_rows"] = {"bandwidth": band_rows, "ladder": ladder}
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute(verbose=False)
    figs = figures(res)

    dates = pd.to_datetime(res["dates"])
    a_vol, a_val = res["analysis"]["volume"], res["analysis"]["value"]
    a_sw = res["share"]["analysis"]["weight"]
    a_sv = res["share"]["analysis"]["value"]
    top = a_vol["top"][0]
    bdate = dates[top["break_at"]]
    sdates = pd.to_datetime(res["share"]["dates"])
    wbreak = sdates[a_sw["top"][0]["break_at"]]
    mde36 = res["mde"]["volume"]["36"]["mde"]
    mde_v36 = res["mde"]["value"]["36"]["mde"]
    mde_all = [res["mde"]["volume"][str(w)]["mde"] for w in POST_WINDOWS]
    stab = res["calibration_stability"]
    band = res["bandwidth"]
    tc = res["trend_choice"]
    ann = res["share"]["annual"]

    # The spine, asserted rather than trusted.
    if a_vol["scan"]["share_ols"] < 0.4:
        raise AssertionError("the placebo finding is the post; it has gone away")
    if a_vol["detected"]:
        raise AssertionError(
            "a break in export volume is now detectable, which reverses "
            "sections 2 and 3")
    if not a_sw["detected"]:
        raise AssertionError(
            "the kilogram share break is section 5's whole point")
    if a_sv["detected"]:
        raise AssertionError(
            "the dollar share now shows a break too, so the contrast is gone")
    if min(mde_all) < 0.20:
        raise AssertionError("the minimum detectable shift has become small")
    if band["0"]["0.9"] < 0.40 or band["48"]["0.0"] < 0.12:
        raise AssertionError(
            "the bandwidth table no longer fails at both ends, so the argument "
            "for simulating critical values is weaker than the text claims")
    if stab["sd"] < 0.05:
        raise AssertionError("the calibration spread is reported as material")

    sections = [
        Section(heading="The question, and the question behind it", body=f"""
China is building semiconductor capacity at a pace that has made "what does this
do to Korea and Taiwan" the industry's standing question. It gets answered, most
days, by pointing at a chart of monthly export figures.

This post does not answer it. It asks whether those figures *can* answer it, and
the answer to that turns out to be checkable.

Here is the shape of the problem. Korea's semiconductor export volume index runs
{a_vol['n']} months, {dates[0]:%B %Y} to {dates[-1]:%B %Y}. Fit the standard
thing — a step at some date, on top of a trend and monthly seasonals — and at
{bdate:%B %Y} you find a **{_drop(top['shift'])} permanent drop** with a
t-statistic of **{abs(top['t_ols']):.1f}**. That is not a marginal result. On the
numbers as usually computed it is overwhelming.

It is also an artefact, and the same procedure produces one at three months in
five.

**The claim of this post: at the magnitudes under discussion, Korea's export
data cannot distinguish displacement from an ordinary turn of the memory cycle,
and the size it would take is about a third.** If a correctly-sized test found a
break anywhere in twenty-six years, that would be wrong.
""", figures=[figs["f1"]]),
        Section(heading="1. Three hundred months, twenty-eight observations",
                body=f"""
Start with how much information is in the series at all.

The variance of a sample mean is the textbook `sigma^2 / n` only when
observations are independent. When they are correlated it is that, multiplied by
a factor that sums the autocorrelation function — and a series with a multi-year
memory carries far fewer independent observations than it has rows.

For this series the factor is {a_vol['vif']:.1f}. So {a_vol['n']} months of data
are worth about **{a_vol['n_eff']:.0f} independent observations** for the purpose
of pinning down a level. Twenty-six years of monthly statistics, and the
information content of a couple of years of quarterly ones.

Two details behind that number, because both could have gone the other way. The
trend is quadratic, not linear: a linear trend in a logged series asserts a
constant growth rate, which this series does not have, and fitting one leaves
curvature in the residual that then gets counted as persistence. With a linear
trend the residual's autocorrelation never crosses zero within six years and the
effective sample size reads {tc['1']['n_eff']:.0f} — worse, for a reason that has
nothing to do with the cycle. With a quadratic it crosses at
lag {tc['2']['acf_first_zero']} and troughs at
lag {tc['2']['acf_trough_lag']}, implying a cycle near
{tc['2']['implied_cycle_months']} months. Going cubic changes the residual scale
by less than a percent, so quadratic is where it stops.
"""),
        Section(heading="2. The test finds a break at 61% of all dates",
                body=f"""
Now run the break test at every date the series admits — {a_vol['scan']['n_dates']}
of them — and count how often it fires.

With the textbook standard error, **{_pct(a_vol['scan']['share_ols'])}**. Not 5%.
Three months in five, in a series that cannot possibly contain a structural break
at three months in five.

Swap in a Newey-West standard error, which is the standard remedy for exactly
this, and it drops to {_pct(a_vol['scan']['share_hac'])}. Better, and still not
5%.

The remaining gap is the multiple comparisons. A break found by *searching* dates
needs a larger critical value than one tested at a date fixed in advance — this is
Andrews (1993) — and the size of that penalty here is large: a fixed-date bar of
{a_vol['calibration']['critical']:.2f} against a searched bar of
**{a_vol['sup_cv']:.2f}**.

Against which the largest corrected statistic anywhere in the series is
**{a_vol['scan']['max_abs_t_hac']:.2f}**. The {bdate:%B %Y} result that arrived
with a naive t of {abs(top['t_ols']):.1f} does not clear either bar. Neither does
anything else: not the financial crisis, not the 2018 memory downturn, not the
pandemic. The test is looking for a permanent change in level, and a deep dip
followed by a full recovery is not one.
""", figures=[figs["f2"]]),
        Section(heading="3. It would take about a third", body=f"""
So what *would* be visible? Simulate: take the fitted residuals, resample them in
overlapping blocks so the persistence survives, add a step of known size, and run
the same test the analyst would run.

At eighty percent power, against the calibrated critical value, the smallest
permanent fall this design detects is **{_drop(mde36)}** with three years of
post-event data.

The number barely moves with the window. Twelve months of post-event data gives
{_drop(mde_all[0])}; ninety-six months gives {_drop(mde_all[-1])}. That is the
part worth sitting with, because the instinct — wait for more data and the
picture will clarify — is wrong here for a structural reason. Every extra month
arrives correlated with the ones before it, and the step being estimated stays
confounded with the same cycle. More rows, almost no more information.

And the series public commentary actually quotes is the harder one. Export
*value* has a year-on-year standard deviation of 39% against volume's 24%,
because it carries the price swing on top of the quantity swing. Its minimum
detectable shift is {_drop(mde_v36)} against volume's {_drop(mde36)}. A displacement
argument is a quantity argument; running it on the value series costs about eight
points of resolution for nothing.
""", figures=[figs["f3"]]),
        Section(heading="4. The correction everyone reaches for is not enough",
                body=f"""
A short methodological aside, because the standard advice does not survive
contact with this series and it is better to say so than to quietly use it.

Newey-West needs a bandwidth. Simulate its size on series of this length whose
truth is known, and there is no bandwidth that works at both ends: at zero lags
it rejects {_pct(band['0']['0.9'])} of the time on strongly persistent data; at
48 lags it rejects {_pct(band['48']['0.0'])} of the time on white noise. The best
row for persistent data still sits at {_pct(band['24']['0.9'])} against a nominal
5%. This is a bias-variance trade with no good point on it, not a tuning problem.

The way out used throughout this post is to stop reading critical values from a
table and simulate them from the series' own residuals. That works — but it
inherits the series' sampling error, and it is worth quantifying how much.
Resampling the residual history {stab['paths']} times moves the critical value
over {stab['min']:.2f} to {stab['max']:.2f}, a standard deviation of
{stab['sd']:.2f} around {stab['mean']:.2f}.

So the honest ladder is: the naive standard error is badly wrong, Newey-West is a
large improvement and still wrong, calibrating on your own residuals is a further
large improvement and still carries a few tenths of uncertainty in the bar
itself. Every detection threshold in this post should be read as **optimistic**
by that margin.
""" + "\n\n" + md_table(BAND_HEADER, figs["_rows"]["bandwidth"])),
        Section(heading="5. The one thing that is detectable", body=f"""
Everything so far is negative. There is one positive result, and its shape is
instructive.

Korea's customs data gives HS 8542 exports by partner country, monthly, since
{sdates[0]:%B %Y}. Take China's share of that flow and it turns out to depend
entirely on whether you count kilograms or dollars.

By **weight**, there is a break at {wbreak:%B %Y} reaching
|t| = **{a_sw['scan']['max_abs_t_hac']:.2f}** against a searched bar of
{a_sw['sup_cv']:.2f}. That is a real, multiple-comparison-surviving structural
change: the annual mean share goes from {100 * ann['2015']['weight']:.1f}% in 2015
to {100 * ann['2016']['weight']:.1f}% in 2016 and has drifted to
{100 * ann['2025']['weight']:.1f}% by 2025.

By **value**, nothing. The largest statistic anywhere is
{a_sv['scan']['max_abs_t_hac']:.2f} against a bar of {a_sv['sup_cv']:.2f}, and the
series has a completely different shape — a plateau in the low 40s until 2022, a
fall to {100 * ann['2025']['value']:.1f}% in 2025, and a rebound since.

Same file. Same trade flow. Two measures, two incompatible stories, and only one
of them is statistically real. Since the ratio of the two is price per kilogram,
what the pair actually says is that the *mix* of what Korea ships to China
changed in late 2015 toward higher value per unit weight. That is a composition
fact. It is not the fact the argument is about, which brings us to the limits.

The mix shift is visible directly. A kilogram of Korean integrated circuits sent
to China has always fetched more than a kilogram sent to the world average — the
ratio was {ann['2009']['uv_ratio']:.2f} in 2009 — and that premium widened
through the break, reaching {ann['2015']['uv_ratio']:.2f} by 2015 and peaking at
{ann['2022']['uv_ratio']:.2f} in 2022. It has since fallen back to
{ann['2025']['uv_ratio']:.2f}, the lowest since 2011. So the two share series
diverge for an identifiable reason, and the reason moves: fewer kilograms at a
higher premium after 2015, then the premium compressing after 2022 while the
kilogram share kept sliding. Whether any of that is Chinese substitution, this
data does not say. What it does say is that a commentator quoting the dollar
share and a commentator quoting the kilogram share are not disagreeing about
interpretation. They are describing different quantities, and only one of them
has moved in a way that survives a test.
""", figures=[figs["f4"]] ) ,
        Section(heading="Where this breaks", body=f"""
**Exports to China are China as a customer, not China as a competitor.** This is
the limitation that matters most, and it is not a small one. A falling share of
Korea's chip exports going to China is consistent with Chinese domestic
substitution, and equally consistent with export controls, with Chinese demand
weakening, with assembly moving to Vietnam, or with Korea's product mix shifting.
Nothing in a bilateral trade series separates those. The October 2015 break is
real; what caused it is not something this data can tell you, and I have not
tried.

**A step is not the only shape displacement could take.** If Chinese capacity
erodes Korean volume gradually, the right alternative is a change in slope, not a
change in level, and a trend break has different power properties. I tested the
step because the step is what the public argument asserts — "share is being
taken" — but a slower squeeze could be present and invisible to this test while
being visible to another.

**The quadratic trend is a choice, and a flexible trend eats the thing you are
looking for.** Every degree of freedom given to the trend is a degree of freedom
a step could have used. Cubic and quartic fits change the residual scale by
almost nothing here, which is why quadratic is defensible, but a reader who
preferred a local-level model would get different and probably smaller numbers.

**One country, one HS code, one series.** HS 8542 is integrated circuits in
aggregate; it does not separate legacy nodes from leading-edge, which is the
distinction the entire policy argument turns on. No public trade code does. That
is arguably a bigger obstacle than anything measured above, and it is not fixable
with better statistics.

**The last seven months are excluded.** ECOS marks them provisional and they get
revised. Including them does not change any conclusion here, but they are not
final data and are not treated as such.

**And nothing here says anything about any company.** Not about who gains, not
about who loses, not about what to do. The finding is about what a series can
support, and "the data cannot settle this" is not evidence for either side of the
thing it cannot settle.
"""),
        Section(heading="What to do with this", body=f"""
Three habits, in ascending order of how much they cost.

**Before testing a claim on a series, ask what size of effect the series could
detect.** It is a short simulation and it is the difference between a null result
that means something and one that means nothing. Here the answer was {_drop(mde36)},
which is larger than any displacement anyone is actually claiming.

**Run the placebo scan.** Fit the same break at every date and count how often it
fires. If the answer is anywhere near {_pct(a_vol['scan']['share_ols'])}, the
finding at your date of interest is not evidence, whatever its p-value.

**Pay for the search.** A date chosen after looking at the data needs a bigger
critical value than one chosen before, and here the gap was
{a_vol['calibration']['critical']:.2f} to {a_vol['sup_cv']:.2f} — the difference
between a result and no result.

The general form, which is why this post exists: a great deal of quantitative
commentary is conducted on series that cannot carry the weight being put on them,
and the way to find out is not to argue about the conclusion. It is to ask what
the smallest detectable version of the claim would be, and compare that to the
claim. When the second is smaller than the first, everyone in the argument is
reading noise, however sophisticated the reading.

The next post takes the shape this one did not test: not a step but a bend, and
what it costs to look for one when you do not know where it starts.
""" + "\n\n" + md_table(LADDER_HEADER, figs["_rows"]["ladder"])),
    ]

    post = Post(
        title="Korea's Chip Exports Would Have to Fall 35% Before Anyone Could Prove It",
        slug="korea-chip-exports-thirty-five-percent",
        subtitle=("Twenty-six years of monthly data, worth twenty-eight "
                  "independent observations. A standard break test finds a "
                  "structural break at 61% of all possible dates. Nothing in "
                  "the series survives a bar built for the search that found "
                  "it."),
        summary=(
            f"Korea's semiconductor export volume index runs {a_vol['n']} months "
            f"and is worth about **{a_vol['n_eff']:.0f} independent "
            f"observations**, because the memory cycle correlates one month with "
            f"the next. Fit the textbook structural-break regression at every "
            f"admissible date and **{_pct(a_vol['scan']['share_ols'])} of them "
            f"come back significant** — the largest, {bdate:%B %Y}, a "
            f"{_drop(top['shift'])} drop with a t of {abs(top['t_ols']):.1f}. "
            f"Against a critical value simulated from the series' own residuals "
            f"and corrected for having searched {a_vol['scan']['n_dates']} dates, "
            f"**nothing in twenty-six years clears the bar**. The smallest "
            f"permanent fall this data could detect is **{_drop(mde36)}**, and "
            f"more post-event data does not lower it. One thing is detectable: "
            f"China's share of Korea's chip exports *by weight* broke in "
            f"{wbreak:%B %Y} — while the same share *by value* shows nothing "
            f"anywhere, from the same customs file."),
        tags=["semiconductors", "structural breaks", "statistical power",
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
                           f"({a_vol['n']} months, "
                           f"{res['provisional_months']} provisional excluded)",
            "customs_window": f"{sdates[0]:%Y-%m} to {sdates[-1]:%Y-%m} "
                              f"({len(sdates)} months, no gaps)",
            "trend": "quadratic in log, plus eleven monthly dummies",
            "hac_bandwidth": HAC_LAGS,
            "bootstrap_block_months": BLOCK,
            "replications": {"calibration": CAL_REPS, "sup": SUP_REPS,
                             "mde": MDE_REPS},
            "modules": "quantpost/ts/detect.py, quantpost/sources/korea_files.py",
            "tests": "tests/test_detect.py",
        },
        min_words=2000,
        max_words=3000,
    )
    post.hero = figs["hero"]
    _check_table_placement(post)
    return post


def _check_table_placement(post: Post) -> None:
    """Table images are matched to markdown tables positionally, so verify."""
    from quantpost.render import publish

    was = post.draft
    post.draft = False
    try:
        body = publish.medium_bundle(
            post, out_dir=qp.SETTINGS.build_dir / "_placement19").read_text()
    finally:
        post.draft = was
    heading, seen = "", {}
    for line in body.split("\n"):
        if line.startswith("## "):
            heading = line[3:].strip()
        m = re.search(r"!\[[^\]]*\]\(([^)]+)\)", line)
        if m:
            seen[m.group(1).rsplit("/", 1)[-1]] = heading
    for name, needle in ((f"a11-t1-bandwidth.{EXT}", "correction"),
                         (f"a11-t2-ladder.{EXT}", "What to do")):
        where = seen.get(name)
        if where is None:
            raise AssertionError(f"{name} never reached the rendered body")
        if needle.lower() not in where.lower():
            raise AssertionError(
                f"{name} landed under {where!r}; table_figures is matched "
                f"positionally, so check its order")


if __name__ == "__main__":
    import sys
    res = compute(force="--force" in sys.argv)
    a = res["analysis"]["volume"]
    print(f"\nvolume: n={a['n']}  n_eff={a['n_eff']:.1f}  "
          f"placebo naive {a['scan']['share_ols']:.1%}")
    print(f"  fixed cv {a['calibration']['critical']:.2f}, "
          f"sup cv {a['sup_cv']:.2f}, observed {a['scan']['max_abs_t_hac']:.2f}"
          f" -> {'detected' if a['detected'] else 'NOT detected'}")
    for t in a["top"][:3]:
        print(f"  {res['dates'][t['break_at']]}  shift {t['shift']:+.3f} "
              f"naive t {t['t_ols']:+.2f}  hac t {t['t_hac']:+.2f}")
    for tag in ("weight", "value"):
        s = res["share"]["analysis"][tag]
        print(f"\nchina share by {tag}: max|t| {s['scan']['max_abs_t_hac']:.2f} "
              f"vs sup cv {s['sup_cv']:.2f} -> "
              f"{'DETECTED' if s['detected'] else 'not detected'}")
