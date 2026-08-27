"""exp021 — when Korea's 2040 electricity forecast becomes checkable.

The news
--------
On 20 August 2026 Korea published the 12th Basic Plan for Long-term Electricity
Supply and Demand. Its 2040 peak-demand forecast is 158.4-165 GW, against
131.8-138.2 GW in the previous plan for a similar horizon, and the coverage put
the difference in units of nuclear reactors. Semiconductor fabs and AI data
centres are named as the reason.

What this post does
-------------------
It converts the forecast into the only form that can be checked - a growth rate -
and then asks the question nobody in the argument is asking: **when would data be
able to settle it?**

1. **Growth already bent once, and that one is not in doubt.** Korea's annual
   peak load grew 6.44% a year to 2010 and 2.00% a year after it: a kink of
   -4.26 percentage points, |t| = 11.23 against a searched critical value of
   7.91 simulated from the series' own residuals, and a date pinned to 2010 with
   a 90% interval of 2009-2011. No cause is attributed here.

2. **The plan asks for a quarter of that kink, upward.** Reaching 158.4-165 GW
   by 2040 needs 3.06-3.34% a year, so relative to the fitted post-2010 trend the
   plan asserts a slope change of +1.04 to +1.31 points. That is not an
   unprecedented magnitude. It is smaller than the change the series already
   made, in the other direction.

3. **The noise model decides the answer, and the obvious choice is wrong.**
   Resampling the residual of the *no-break* model gives 15% power at 2040 and
   the conclusion "unknowable". That residual still contains the 2010 kink;
   its standard deviation is 10.9% against the fitted model's 4.0%. With the
   right residual, power at 2040 is one. A known-answer check settles which is
   right: the kink the data already shows must simulate as detectable, and it
   comes back at 100% on the fitted residual against 77% on the no-break one.

4. **The answer arrives around 2035, not at the target date.** Power to detect
   the plan's central path, with the start date given away and no search penalty,
   runs 20% by 2030, 50% by 2033 and 78% by 2035, crossing 80% in 2035. The
   plan's own target year is five years after the question stops being open.

5. **The conventional test would call it early, repeatedly.** Against 1.96 the
   same design rejects 15-25% of the time when nothing has changed, so an analyst
   re-running a t-test each year would announce confirmation about one year in
   five regardless of what the economy did.

6. **Part of the forecast is about load shape, not volume.** The ratio of annual
   peak to average load rose from 1.19 to 1.30 over twenty years and the plan's
   own numbers imply it keeps rising. That is a point in the plan's favour and it
   is stated as one. Meanwhile industrial sales - the class the semiconductor
   argument is about - have fallen 0.63% a year since 2018.

   Numbers in this docstring are restated from the cached run and build()
   asserts each of them, so a run whose results have moved fails rather than
   publishing prose that no longer matches its own output.

Discipline
----------
No claim about any company appears here and none follows. The subject is a
government forecast and an aggregate contract class. Nothing in this post is an
investment implication of any kind.

Two measurement facts are stated rather than smoothed over. EPSIS reports the
load settled through the power exchange; the plan forecasts a wider quantity, and
for 2023 the two differ by 5% (98.3 against 93.6 GW). Every growth rate here is
computed inside one series, and the bridge is applied once, explicitly, only to
put the plan's endpoint on the observed series' basis. And sales are sales: they
exclude self-generation, so the load factors here are lower bounds and are
comparable across years rather than against a figure built from generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.render.post import Post, Section
from standarderror.sources import korea_power as kp
from standarderror.ts import bend as bd
from standarderror.ts import detect as dt
from standarderror.viz import charts, theme

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed
CACHE = se.SETTINGS.build_dir / "cache" / "exp021.json"
DATA = se.SETTINGS.build_dir.parent / "data" / "korea_power"

# ---------------------------------------------------------------- configuration

BASE_YEAR = 2025          # last completed year; 2026 is still running
TARGET_YEAR = 2040
A_LAGS, A_BLOCK, A_TRIM = 4, 8, 0.15        # annual peak series
M_LAGS, M_BLOCK, M_TRIM = 24, 36, 0.15      # monthly average-load series
SUP_REPS = 500
RACE_REPS = 300
DATE_REPS = 600
CAL_REPS = 3000
POWER_REPS = 1200
MDE_REPS = 600
HORIZONS = (5, 8, 9, 10, 11, 13, 15, 20)
BAR_PATHS = 6

# The 12th plan's 2040 peak-demand pair, and the previous plan's, as published.
PLAN_2040 = {"managed": 158.4, "baseline": 165.0}
PLAN_PREVIOUS = (131.8, 138.2)
PLAN_TWH = (847.3, 885.1)

SOURCES = [
    "Korea Power Exchange, EPSIS 전력수급실적 — annual peak load, supply "
    "capability and reserve margin with the date of each year's peak, 1993 to "
    "2026. Downloaded 2026-08-25.",
    "Korea Power Exchange, EPSIS 계약종별 판매전력량 — annual electricity sales "
    "by contract class, 2001 to 2025. Downloaded 2026-08-25.",
    "Korea Power Exchange, EPSIS 최대전력 월별평균 — monthly mean of daily peak "
    "load, 1993 to 2026. Downloaded 2026-08-25.",
    kp.LICENCE_NOTE,
    "Ministry of Climate, Energy and Environment, 12th Basic Plan for Long-term "
    "Electricity Supply and Demand, announced 2026-08-20. The 2040 figures used "
    "here (158.4-165 GW peak, 847.3-885.1 TWh) and the previous plan's "
    "131.8-138.2 GW are as reported in contemporaneous coverage.",
    "National Assembly Research Service, 「제11차 전력수급기본계획」 실무안의 "
    "평가와 제언, 이슈와 논점 2317호, 2025-01-02 — source of the 2023 peak of "
    "98.3 GW on the plan's own basis, and of the 10th and 11th plans' demand "
    "decompositions.",
    "D. W. K. Andrews, 'Tests for parameter instability and structural change "
    "with unknown change point', Econometrica 1993;61:821-856.",
    "P. Perron, 'Dealing with structural breaks', Palgrave Handbook of "
    "Econometrics 2006;1:278-352.",
]


def _config_key() -> str:
    blob = json.dumps({"v": 4, "base": BASE_YEAR, "target": TARGET_YEAR,
                       "annual": [A_LAGS, A_BLOCK, A_TRIM],
                       "monthly": [M_LAGS, M_BLOCK, M_TRIM],
                       "reps": [SUP_REPS, RACE_REPS, DATE_REPS, CAL_REPS,
                                POWER_REPS, MDE_REPS],
                       "horizons": list(HORIZONS), "paths": BAR_PATHS,
                       "plan": PLAN_2040, "seed": SEED,
                       "impl": hashlib.sha256(
                           open(bd.__file__, "rb").read()).hexdigest()[:12]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- data

def load() -> dict:
    sd = kp.read_supply_demand(DATA / "supply_demand_annual.csv")
    sales = kp.read_sales(DATA / "sales_by_contract.csv")
    monthly = kp.read_monthly_peak(DATA / "monthly_avg_peak.csv")
    return {"sd": sd, "sales": sales, "monthly": monthly,
            "annual": kp.annual_frame(sd, sales, monthly)}


# ---------------------------------------------------------------- analysis

def bend_history(y: np.ndarray, *, lags: int, block: int, trend: int,
                 seasonal: int = 0, start_period: int = 0, say, label: str
                 ) -> dict:
    """Where the series' own growth rate changed, and whether it is detectable.

    Returns the fitted alternative's residual as well, because that residual —
    not the no-break model's — is the noise every forward power calculation is
    entitled to assume. The no-break residual still contains the kink.
    """
    n = y.size
    kw = dict(trend=trend, seasonal=seasonal, start_period=start_period,
              lags=lags, trim=A_TRIM if seasonal == 0 else M_TRIM)
    race = bd.model_race(y, **kw)
    tau = race["bend_date"]
    fit = bd.fit_at(y, tau, kind="bend", trend=trend, seasonal=seasonal,
                    start_period=start_period, lags=lags)
    r_null = dt.ols_hac(dt.design_matrix(n, trend=trend, seasonal=seasonal,
                                         start_period=start_period),
                        y, lags=lags).resid
    r_alt = fit["resid"]
    cal = bd.calibrated_sup(r_null, n=n, block=block, kind="bend",
                            reps=SUP_REPS, stride=1 if seasonal == 0 else 4,
                            rng=np.random.default_rng(SEED + 3), **kw)
    nulls = bd.null_race(r_null, n=n, block=block, reps=RACE_REPS,
                         stride=1 if seasonal == 0 else 3,
                         rng=np.random.default_rng(SEED + 5), **kw)

    X = bd.bend_design(n, tau=tau, kind="bend", trend=trend, seasonal=seasonal,
                       start_period=start_period)
    beta = dt.ols_hac(X, y, lags=lags).beta
    per = 1 if seasonal == 0 else 12
    pre = float(beta[1] / (n - 1)) * per
    slope = bd.slope_change_per_year(fit["coef"], n, per)
    detected = race["bend_sup_t"] > cal["sup"]
    say(f"  {label}: bend at index {tau}, {100 * slope:+.2f} pp/yr, "
        f"sup|t| {race['bend_sup_t']:.2f} vs {cal['sup']:.2f} -> "
        f"{'DETECTED' if detected else 'not detected'}")
    say(f"  {label}: resid sd null {r_null.std():.4f} -> alt {r_alt.std():.4f}; "
        f"n_eff {dt.effective_sample_size(r_alt, lags):.1f} of {n}")
    return {
        "n": int(n), "tau": int(tau), "race": race, "calibrated": cal,
        "null_race": nulls, "detected": bool(detected),
        "shape_beyond_chance": bool(
            race["ssr_gain"] > nulls[f"{race['winner']}_gain_q95"]),
        "shape_bar": float(nulls[f"{race['winner']}_gain_q95"]),
        "coef": fit["coef"], "t_hac": fit["t_hac"], "t_ols": fit["t_ols"],
        "slope_change": float(slope),
        "growth_pre": pre, "growth_post": float(pre + slope),
        "resid_sd_null": float(r_null.std()), "resid_sd_alt": float(r_alt.std()),
        "n_eff_null": dt.effective_sample_size(r_null, lags),
        "n_eff_alt": dt.effective_sample_size(r_alt, lags),
        "vif_alt": dt.variance_inflation(r_alt, lags),
        "resid_alt": r_alt.tolist(), "resid_null": r_null.tolist(),
    }


def plan_requirement(peak_base_mw: float, growth_post: float) -> dict:
    """The plan's endpoints as growth rates and as slope changes.

    The bridge is applied once and only here: the plan's target is on its own
    wider basis, the observed series is not, and a level comparison between them
    without the bridge would overstate the required growth.
    """
    out = {}
    span = TARGET_YEAR - BASE_YEAR
    for label, gw in PLAN_2040.items():
        on_basis = gw * 1000.0 / kp.PLAN_BASIS_2023
        g = float(np.log(on_basis / peak_base_mw) / span)
        out[label] = {"gw_published": gw, "mw_on_epsis_basis": on_basis,
                      "growth": g, "slope_change": g - growth_post}
    return out


def power_curve(resid: np.ndarray, *, n_pre: int, requirement: dict,
                block: int, lags: int, trend: int, say) -> dict:
    """Power at each horizon, against a bar recalibrated at that horizon."""
    kw = dict(trend=trend, lags=lags)
    rows = {}
    for h in HORIZONS:
        cal = bd.calibrated_fixed(resid, n_pre=n_pre, n_post=h, block=block,
                                  reps=CAL_REPS,
                                  rng=np.random.default_rng(SEED + 41), **kw)
        entry = {"critical": cal["critical"], "size_of_1p96": cal["size_of_1p96"]}
        for label, req in requirement.items():
            p = bd.bend_power(resid, n_pre=n_pre, n_post=h,
                              slope_change=req["slope_change"], block=block,
                              reps=POWER_REPS, critical=cal["critical"],
                              rng=np.random.default_rng(SEED + 50 + h), **kw)
            entry[label] = p["power_hac"]
        mde = bd.minimum_detectable_bend(resid, n_pre=n_pre, n_post=h,
                                         block=block, reps=MDE_REPS, hi=0.08,
                                         critical=cal["critical"], **kw)
        entry["mde"] = mde["mde"]
        rows[h] = entry
        say(f"    +{h:2d}y ({BASE_YEAR + h}): bar {cal['critical']:.2f}, "
            f"1.96 rejects {cal['size_of_1p96']:.0%}, power "
            + ", ".join(f"{k.split()[0]} {entry[k]:.0%}" for k in requirement)
            + f", mde {100 * entry['mde']:.2f} pp/yr")
    return rows


def crossing_year(rows: dict, label: str, target: float = 0.80) -> float | None:
    """First calendar year the power curve reaches `target`, interpolated.

    Interpolated rather than reported as the first grid point, because the grid
    is arbitrary and rounding a crossing up to the next simulated horizon would
    move the post's headline by years.
    """
    hs = sorted(rows)
    for a, b in zip(hs, hs[1:]):
        pa, pb = rows[a][label], rows[b][label]
        if pa < target <= pb:
            frac = (target - pa) / (pb - pa)
            return BASE_YEAR + a + frac * (b - a)
    return None


def bar_stability(resid: np.ndarray, *, n_pre: int, n_post: int, block: int,
                  lags: int, trend: int) -> dict:
    """How much the calibrated bar moves on the seed alone."""
    bars = [bd.calibrated_fixed(resid, n_pre=n_pre, n_post=n_post, block=block,
                                reps=CAL_REPS,
                                rng=np.random.default_rng(6000 + p),
                                trend=trend, lags=lags)["critical"]
            for p in range(BAR_PATHS)]
    return {"paths": BAR_PATHS, "lo": float(min(bars)), "hi": float(max(bars)),
            "median": float(np.median(bars)), "sd": float(np.std(bars, ddof=1))}


def compute(*, force: bool = False, verbose: bool = True) -> dict:
    key = _config_key()
    if CACHE.exists() and not force:
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            if verbose:
                print(f"exp021: cached ({CACHE})")
            return cached
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    raw = load()
    ann = raw["annual"]
    ann = ann[ann.index <= BASE_YEAR]
    sd = raw["sd"]
    sd = sd[sd.year <= BASE_YEAR]
    peak = sd.peak_mw.to_numpy()
    years = sd.year.to_numpy()
    say(f"exp021: annual peak {years[0]}-{years[-1]} ({len(years)} years), "
        f"{int(sd.winter_peak.sum())} winter peaks")

    res = {"key": key, "years": years.tolist(), "peak_mw": peak.tolist(),
           "winter_peak": sd.winter_peak.tolist(),
           "peak_month": sd.peak_month.tolist(),
           "basis_bridge": kp.PLAN_BASIS_2023}

    # ---- the kink the series already has
    say("\nannual peak load")
    res["annual"] = bend_history(np.log(peak), lags=A_LAGS, block=A_BLOCK,
                                 trend=1, say=say, label="annual")
    a = res["annual"]

    monthly = raw["monthly"]
    monthly = monthly[(monthly.index >= "2005-01-01")
                      & (monthly.index <= f"{BASE_YEAR}-12-01")]
    res["monthly_dates"] = [str(d.date()) for d in monthly.index]
    res["monthly_mw"] = monthly.tolist()
    say("\nmonthly average load, both trend specifications")
    res["monthly"] = {
        str(deg): bend_history(np.log(monthly.to_numpy()), lags=M_LAGS,
                               block=M_BLOCK, trend=deg, seasonal=12,
                               start_period=0, say=say, label=f"monthly deg{deg}")
        for deg in (1, 2)}

    # ---- how wide the kink's date is
    ci = bd.date_bootstrap(np.log(peak), block=A_BLOCK, kind="bend",
                           reps=DATE_REPS, level=0.90,
                           rng=np.random.default_rng(SEED + 9),
                           trend=1, lags=A_LAGS, trim=A_TRIM)
    draws = ci.pop("draws")
    ci["hist"] = np.bincount(draws, minlength=len(years)).tolist()
    ci["q25"], ci["q75"] = int(np.quantile(draws, .25)), int(np.quantile(draws, .75))
    res["date_ci"] = ci
    say(f"\nkink date {years[ci['tau_hat']]}, middle half "
        f"{years[ci['q25']]}-{years[ci['q75']]}, 90% "
        f"{years[ci['lo']]}-{years[ci['hi']]}")

    # ---- what the plan asks for
    res["requirement"] = plan_requirement(peak[-1], a["growth_post"])
    for label, req in res["requirement"].items():
        say(f"  {label}: {req['gw_published']} GW -> "
            f"{req['mw_on_epsis_basis']:,.0f} MW on the observed basis, "
            f"{100 * (np.exp(req['growth']) - 1):.2f}%/yr, slope change "
            f"{100 * req['slope_change']:+.2f} pp/yr")

    # ---- the noise-model question, settled by a known-answer check
    say("\nnoise model: the same power calculation on two residuals")
    kw = dict(trend=1, lags=A_LAGS)
    check = {}
    for tag, r in (("alt", np.array(a["resid_alt"])),
                   ("null", np.array(a["resid_null"]))):
        cal = bd.calibrated_fixed(r, n_pre=a["tau"], n_post=a["n"] - a["tau"],
                                  block=A_BLOCK, reps=CAL_REPS,
                                  rng=np.random.default_rng(SEED + 13), **kw)
        p = bd.bend_power(r, n_pre=a["tau"], n_post=a["n"] - a["tau"],
                          slope_change=a["slope_change"], block=A_BLOCK,
                          reps=POWER_REPS, critical=cal["critical"],
                          rng=np.random.default_rng(SEED + 17), **kw)
        check[tag] = {"sd": float(r.std()), "critical": cal["critical"],
                      "power_of_the_observed_kink": p["power_hac"]}
        say(f"  {tag} residual (sd {r.std():.4f}): the observed kink simulates "
            f"as detectable {p['power_hac']:.0%} of the time")
    res["noise_check"] = check
    if check["alt"]["power_of_the_observed_kink"] <= check["null"]["power_of_the_observed_kink"]:
        raise AssertionError(
            "the fitted-model residual no longer gives the observed kink more "
            "power than the no-break residual, which is the section's premise")

    # ---- when the plan becomes checkable
    say("\npower against the plan, bar recalibrated at each horizon")
    res["power"] = power_curve(np.array(a["resid_alt"]), n_pre=a["n"],
                               requirement=res["requirement"], block=A_BLOCK,
                               lags=A_LAGS, trend=1, say=say)
    res["crossing"] = {k: crossing_year(res["power"], k)
                       for k in res["requirement"]}
    for k, v in res["crossing"].items():
        say(f"  {k}: 80% power at {v:.1f}" if v else f"  {k}: never within horizon")
    # The same curve on the wrong residual, at every horizon, because the point
    # of the comparison is that it changes the answer at every horizon and not
    # only at the target year.
    say("  the same curve on the no-break residual")
    res["power_null"] = power_curve(np.array(a["resid_null"]), n_pre=a["n"],
                                    requirement=res["requirement"],
                                    block=A_BLOCK, lags=A_LAGS, trend=1,
                                    say=lambda *x, **k: None)
    res["power_null_resid_15y"] = bd.bend_power(
        np.array(a["resid_null"]), n_pre=a["n"], n_post=15,
        slope_change=list(res["requirement"].values())[0]["slope_change"],
        block=A_BLOCK, reps=POWER_REPS,
        critical=bd.calibrated_fixed(
            np.array(a["resid_null"]), n_pre=a["n"], n_post=15, block=A_BLOCK,
            reps=CAL_REPS, rng=np.random.default_rng(SEED + 19),
            **kw)["critical"],
        rng=np.random.default_rng(SEED + 23), **kw)["power_hac"]
    say(f"  the same 2040 power on the no-break residual: "
        f"{res['power_null_resid_15y']:.0%}")

    res["bar_stability"] = bar_stability(np.array(a["resid_alt"]), n_pre=a["n"],
                                        n_post=10, block=A_BLOCK, lags=A_LAGS,
                                        trend=1)
    bs = res["bar_stability"]
    say(f"  bar at +10y across {bs['paths']} seeds: {bs['lo']:.2f}-{bs['hi']:.2f} "
        f"(sd {bs['sd']:.2f})")

    # ---- load shape and who carries the volume
    shape = ann[["peak_mw", "monthly_avg_mw", "peak_to_avg", "load_factor",
                 "total", "industrial", "general", "residential"]].dropna(
                     subset=["peak_to_avg"])
    res["shape"] = {"years": shape.index.tolist(),
                    "peak_to_avg": shape.peak_to_avg.tolist(),
                    "load_factor": shape.load_factor.tolist()}
    # Computed from the plan's *own* pair of numbers, with no bridge applied to
    # either. Bridging the peak but not the energy — which the first version did —
    # mixes bases inside one ratio and reported 64% where the plan's own figures
    # say 61%. The historical ratio below is likewise internally consistent, on a
    # narrower basis, so the two are comparable as ratios and not as levels.
    lf_plan = [twh * 1e6 / (gw * 1000.0 * 8760) * 100
               for twh, gw in zip(PLAN_TWH, PLAN_2040.values())]
    res["plan_load_factor"] = lf_plan
    say(f"\nload shape: peak/average {shape.peak_to_avg.iloc[0]:.3f} "
        f"({shape.index[0]}) -> {shape.peak_to_avg.iloc[-1]:.3f} "
        f"({shape.index[-1]}); load factor "
        f"{shape.load_factor.iloc[0]:.1f}% -> {shape.load_factor.iloc[-1]:.1f}%, "
        f"plan implies {lf_plan[0]:.1f}-{lf_plan[1]:.1f}%")

    sales = raw["sales"]
    classes = ["industrial", "general", "residential", "education",
               "agricultural", "total"]
    def cagr(col, y0, y1):
        s = sales.set_index("year")[col]
        return (s[y1] / s[y0]) ** (1 / (y1 - y0)) - 1
    res["sales"] = {
        "years": [int(sales.year.min()), int(sales.year.max())],
        "level_2025": {c: float(sales.set_index("year")[c][BASE_YEAR])
                       for c in classes},
        "share_2025": {c: float(sales.set_index("year")[c][BASE_YEAR]
                                / sales.set_index("year")["total"][BASE_YEAR])
                       for c in classes},
        "cagr": {c: {f"{y0}_{BASE_YEAR}": float(cagr(c, y0, BASE_YEAR))
                     for y0 in (2001, 2015, 2018)} for c in classes},
    }
    for c in ("total", "industrial", "general"):
        g = res["sales"]["cagr"][c]
        say(f"  {c:12s} " + "  ".join(f"{k}: {100*v:+.2f}%/yr" for k, v in g.items()))
    res["plan_twh_growth"] = [
        float(np.log(t / sales.set_index("year")["total"][BASE_YEAR])
              / (TARGET_YEAR - BASE_YEAR)) for t in PLAN_TWH]
    say(f"  plan needs consumption "
        f"{100*(np.exp(res['plan_twh_growth'][0])-1):.2f}-"
        f"{100*(np.exp(res['plan_twh_growth'][1])-1):.2f}%/yr")

    res["elapsed_s"] = round(time.time() - t0, 1)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(res))
    say(f"\nexp021: {res['elapsed_s']}s -> {CACHE}")
    return res


if __name__ == "__main__":
    import sys
    compute(force="--force" in sys.argv)


# ---------------------------------------------------------------- figures

def _pct(x: float, dp: int = 0) -> str:
    return f"{100 * x:.{dp}f}%"


def _rate(g: float) -> str:
    """A log growth rate as the percentage a reader expects."""
    return f"{100 * (np.exp(g) - 1):.2f}%"


def _pp(x: float) -> str:
    return f"{100 * x:+.2f}"


def md_table(header, rows) -> str:
    def cell(x):
        return str(x).replace("|", r"\|")
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight"}

POWER_HEADER = ["by year", "correctly-sized bar", "how often 1.96 fires on nothing",
                "power, managed path", "power, baseline path",
                "smallest change visible"]
SALES_HEADER = ["contract class", "2025, TWh", "share", "2001-25", "2015-25",
                "2018-25"]
CLASS_LABEL = {"industrial": "industrial", "general": "general service",
               "residential": "residential", "education": "education",
               "agricultural": "agricultural", "total": "all classes"}


def _fitted_path(res: dict) -> tuple[np.ndarray, np.ndarray]:
    """The fitted trend-plus-kink, and its extension at the post-kink slope."""
    years = np.array(res["years"])
    y = np.log(np.array(res["peak_mw"]))
    a = res["annual"]
    X = bd.bend_design(y.size, tau=a["tau"], kind="bend", trend=1)
    beta = dt.ols_hac(X, y, lags=A_LAGS).beta
    fitted = X @ beta
    span = np.arange(BASE_YEAR + 1, TARGET_YEAR + 1)
    ext = fitted[-1] + a["growth_post"] * (span - BASE_YEAR)
    return np.exp(fitted), np.exp(ext)


def figures(res: dict) -> dict:
    figs = {}
    years = np.array(res["years"])
    peak = np.array(res["peak_mw"]) / 1000.0
    a = res["annual"]
    ci = res["date_ci"]
    req = res["requirement"]
    labels = list(req)
    fitted, ext = _fitted_path(res)
    span = np.arange(BASE_YEAR + 1, TARGET_YEAR + 1)

    # ---- F1: the series, its one kink, and where the plan points ---------
    frame = pd.DataFrame({"annual peak load": peak,
                          "fitted: one trend, one kink": fitted / 1000.0},
                         index=years)

    def draw_plan(fig, ax):
        m = theme.MODES["light"]
        from matplotlib.ticker import NullFormatter, NullLocator
        ax.set_yscale("log")
        ticks = [20, 30, 50, 70, 100, 150, 200]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        # A log axis keeps labelling its own minor ticks, so the explicit list
        # above arrived mixed with "6 x 10^1" and "4 x 10^1" in the same column.
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.plot(span, ext / 1000.0, color=m.muted, lw=1.6, ls=(0, (5, 3)),
                zorder=2)
        for lab in labels:
            tgt = req[lab]["mw_on_epsis_basis"] / 1000.0
            ax.plot([BASE_YEAR, TARGET_YEAR], [peak[-1], tgt],
                    color=m.series[2], lw=1.8, zorder=3)
            ax.plot([TARGET_YEAR], [tgt], marker="o", ms=5,
                    color=m.series[2], zorder=4)
        ax.text(TARGET_YEAR, req[labels[-1]]["mw_on_epsis_basis"] / 1000.0 * 1.06,
                "12th plan,\n2040  ", color=m.ink_secondary, fontsize=8.5,
                ha="right", va="bottom")
        ax.text(span[-1], ext[-1] / 1000.0 * 0.86,
                "extending the\ncurrent trend  ", color=m.ink_secondary,
                fontsize=8.5, ha="right", va="top")
        ax.axvline(years[a["tau"]], color=m.muted, lw=1.2, ls=(0, (2, 2)),
                   zorder=1)
        ax.text(years[a["tau"]], 21, f"  kink, {years[a['tau']]}",
                color=m.ink_secondary, fontsize=8.5, va="bottom")
        ax.set_xlim(years[0] - 1, TARGET_YEAR + 1)

    figs["f1"], _ = charts.lines(
        frame,
        title="One kink, and a plan that asks for a second one",
        subtitle=(f"Korea's annual peak load, log scale. A single trend with one "
                  f"change of slope in {years[a['tau']]} fits it: growth of "
                  f"{_rate(a['growth_pre'])} a year before, "
                  f"{_rate(a['growth_post'])} after. Extending that post-kink "
                  f"trend reaches {ext[-1] / 1000:.0f} GW in {TARGET_YEAR}. The "
                  f"12th plan's pair, put on this series' basis, is "
                  f"{req[labels[0]]['mw_on_epsis_basis'] / 1000:.0f}-"
                  f"{req[labels[1]]['mw_on_epsis_basis'] / 1000:.0f} GW."),
        xlabel="", ylabel="annual peak load, GW (log scale)",
        source=("Korea Power Exchange, EPSIS annual peak load, 1993-2025; the "
                "running year is excluded. Plan endpoints divided by the "
                f"{res['basis_bridge']:.3f} basis bridge described in the text."),
        alt=("A rising line on a log scale from about 22 GW in 1993 to 96 GW in "
             "2025, steep until around 2010 and much flatter after. A dashed "
             "line continues the flat part to about 133 GW in 2040, and two "
             "solid lines fan above it to endpoints near 151 and 157 GW."),
        caption=(f"Conclude that the plan is a claim about the slope after "
                 f"{BASE_YEAR}, not about the level in {TARGET_YEAR}. Continuing "
                 f"the trend the data is on reaches {ext[-1] / 1000:.0f} GW; the "
                 f"plan needs {_pp(req[labels[0]]['slope_change'])} to "
                 f"{_pp(req[labels[1]]['slope_change'])} percentage points a "
                 f"year more than that."),
        mode="light", direct_labels=False, decorate=draw_plan,
        path=str(IMG / f"a13-f1-peak.{EXT}"))

    # ---- F2: when it becomes checkable ----------------------------------
    hs = sorted(res["power"], key=int)
    pf = pd.DataFrame(
        {f"power, the {labels[0]} path":
             [100 * res["power"][h][labels[0]] for h in hs],
         f"power, the {labels[1]} path":
             [100 * res["power"][h][labels[1]] for h in hs],
         "what 1.96 does when nothing happens":
             [100 * res["power"][h]["size_of_1p96"] for h in hs]},
        index=[BASE_YEAR + int(h) for h in hs])

    def mark_eighty(fig, ax):
        m = theme.MODES["light"]
        ax.axhline(80, color=m.muted, lw=1.4, ls=(0, (4, 3)), zorder=1)
        ax.text(BASE_YEAR + int(hs[0]), 82, "  80% power",
                color=m.ink_secondary, fontsize=8.5, va="bottom")
        for lab, yr in res["crossing"].items():
            if yr:
                ax.plot([yr], [80], marker="o", ms=5, color=m.series[0]
                        if lab == labels[0] else m.series[1], zorder=5)
        ax.axvline(TARGET_YEAR, color=m.muted, lw=1.2, ls=(0, (2, 2)), zorder=1)
        ax.text(TARGET_YEAR, 4, f"  {TARGET_YEAR}, the plan's\n  own target year",
                color=m.ink_secondary, fontsize=8.5, va="bottom")
        ax.set_ylim(0, 104)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"])

    figs["f2"], _ = charts.lines(
        pf,
        title=(f"The question stops being open around "
               f"{res['crossing'][labels[0]]:.0f}"),
        subtitle=(f"Probability that a correctly-sized test detects the plan's "
                  f"asserted change of slope, by the year the data reaches. The "
                  f"break date is treated as known — the plan announces it — so "
                  f"there is no search penalty and this is the favourable case. "
                  f"The lower line is what the textbook 1.96 does on data where "
                  f"nothing has changed."),
        xlabel="", ylabel="probability of detection",
        source=(f"Simulated by moving-block bootstrap of the fitted model's "
                f"residuals, {POWER_REPS} replications per point, against a "
                f"critical value recalibrated at each horizon."),
        alt=("Three curves against year from 2030 to 2045. Two rise steeply "
             "from about 20-30% in 2030 through a dashed 80% line in the "
             "mid-2030s to 100%. A third, much flatter, sits between 15% and "
             "26% throughout. A dotted vertical line marks 2040."),
        caption=(f"Conclude that the plan is falsifiable and that the answer "
                 f"arrives before its target date, not at it: 80% power at "
                 f"{res['crossing'][labels[0]]:.0f} for the managed path and "
                 f"{res['crossing'][labels[1]]:.0f} for the baseline one. The "
                 f"flat line is the reason to bother calibrating — the "
                 f"conventional test announces a change one year in five when "
                 f"there is none."),
        mode="light", direct_labels=False, decorate=mark_eighty,
        path=str(IMG / f"a13-f2-power.{EXT}"))

    # ---- F3: the noise model decides the answer -------------------------
    nf = pd.DataFrame(
        {"noise from the fitted model": [100 * res["power"][h][labels[0]]
                                         for h in hs],
         "noise from the no-break model": [100 * res["power_null"][h][labels[0]]
                                           for h in hs]},
        index=[BASE_YEAR + int(h) for h in hs])

    def mark_check(fig, ax):
        m = theme.MODES["light"]
        ax.axhline(80, color=m.muted, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.set_ylim(0, 104)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"])

    nc = res["noise_check"]
    figs["f3"], _ = charts.lines(
        nf,
        title="Same test, same claim, two answers",
        subtitle=(f"The identical power calculation, differing only in which "
                  f"residual is resampled as noise. The no-break model's "
                  f"residual still contains the {years[a['tau']]} kink, so its "
                  f"standard deviation is {_pct(nc['null']['sd'], 1)} against "
                  f"the fitted model's {_pct(nc['alt']['sd'], 1)}, and every "
                  f"power number computed from it is too low."),
        xlabel="", ylabel="probability of detection",
        source=("Same simulation as the previous figure; only the residual "
                "supplied to the bootstrap changes."),
        alt=("Two curves against year from 2030 to 2045. The upper one rises "
             "from about 20% to 100% and crosses a dashed 80% line in the "
             "mid-2030s. The lower one rises only from about 3% to 24% and "
             "never approaches the line, ending near 24%."),
        caption=(f"Conclude by checking, not by choosing. The kink the data "
                 f"already shows has to simulate as detectable: on the fitted "
                 f"residual it does, "
                 f"{_pct(nc['alt']['power_of_the_observed_kink'])} of the time; "
                 f"on the no-break residual only "
                 f"{_pct(nc['null']['power_of_the_observed_kink'])}, even though "
                 f"its observed t-statistic was {abs(a['t_hac']):.1f}. That "
                 f"settles which residual is the honest noise."),
        mode="light", direct_labels=False, decorate=mark_check,
        path=str(IMG / f"a13-f3-noise.{EXT}"))

    # ---- F4: part of the forecast is load shape -------------------------
    # One axis, one series. The first version put the peak/average ratio on a
    # left axis and the load factor on a twinned right one; the two are close to
    # mechanical inverses of each other, matplotlib drew no legend for the
    # twinned line, and the reader was left with two mirrored squiggles and no
    # key. Load factor alone is the quantity the plan's own two published
    # numbers pin down, so the forecast point can be marked on the same axis.
    sh = res["shape"]
    sf = pd.DataFrame({"load factor": sh["load_factor"]}, index=sh["years"])
    lf_plan = float(np.mean(res["plan_load_factor"]))

    def mark_shape(fig, ax):
        m = theme.MODES["light"]
        ax.plot([TARGET_YEAR], [lf_plan], marker="D", ms=7,
                color=m.series[1], zorder=5)
        ax.plot([sh["years"][-1], TARGET_YEAR], [sh["load_factor"][-1], lf_plan],
                color=m.series[1], lw=1.5, ls=(0, (4, 3)), zorder=3)
        ax.text(TARGET_YEAR, lf_plan - 0.5,
                "the 12th plan's\nown pair of numbers  ",
                color=m.ink_secondary, fontsize=8.5, ha="right", va="top")
        ax.set_ylim(58, 73)
        ax.set_yticks([60, 63, 66, 69, 72])
        ax.set_yticklabels(["60%", "63%", "66%", "69%", "72%"])
        ax.set_xlim(min(sh["years"]) - 1, TARGET_YEAR + 2)

    figs["f4"], _ = charts.lines(
        sf,
        title="Part of a peak forecast is a claim about load shape",
        subtitle=(f"Load factor — annual sales divided by what the annual peak "
                  f"would deliver if it ran all year. It fell from "
                  f"{sh['load_factor'][0]:.1f}% in {sh['years'][0]} to "
                  f"{sh['load_factor'][-1]:.1f}% in {sh['years'][-1]}, which is "
                  f"the same fact as the peak-to-average-load ratio rising from "
                  f"{sh['peak_to_avg'][0]:.3f} to {sh['peak_to_avg'][-1]:.3f}. "
                  f"The plan's own peak and consumption figures imply "
                  f"{lf_plan:.1f}% by {TARGET_YEAR}."),
        xlabel="", ylabel="load factor",
        # The source note sets the figure width — the renderer keeps it on one
        # line — so a five-clause note stretched the frame to three times the
        # width of its own plot. The caveats moved into the caption.
        source=("EPSIS annual peak and sales by contract class; the plan's "
                "point from its own two published numbers."),
        alt=("A noisy line falling from about 69% in 2005 to about 65% in 2025, "
             "then a dashed segment continuing down to a diamond marker at "
             "about 61% in 2040."),
        caption=("Conclude that this part of the plan is not the aggressive "
                 "part. Korean load has genuinely become peakier for twenty "
                 "years and the plan assumes that continues rather than "
                 "reverses. If peakiness instead saturates, the plan's peak "
                 "number needs more volume growth than it appears to. Two "
                 "caveats on the level: sales exclude self-generation, so every "
                 "figure here is a lower bound on the true load factor and is "
                 "comparable across years rather than against one built from "
                 "generation; and no basis adjustment is applied to either of "
                 "the plan's numbers, since both are on the plan's own basis."),
        mode="light", direct_labels=False, decorate=mark_shape,
        path=str(IMG / f"a13-f4-shape.{EXT}"))

    # ---- T1: the power table -------------------------------------------
    prows = []
    for h in hs:
        e = res["power"][h]
        prows.append([f"{BASE_YEAR + int(h)}", f"{e['critical']:.2f}",
                      _pct(e["size_of_1p96"]),
                      _pct(e[labels[0]]), _pct(e[labels[1]]),
                      f"{_pp(e['mde'])} pp/yr"])
    figs["t1"], _ = charts.table_image(
        prows, header=POWER_HEADER,
        title="When each version of the plan becomes checkable",
        subtitle=("For each horizon: the critical value this design actually "
                  "needs, what the textbook 1.96 does instead, the probability "
                  "of detecting each of the plan's two paths, and the smallest "
                  "slope change visible at all."),
        source=(f"Moving-block bootstrap of the fitted model's residuals, "
                f"{CAL_REPS} replications for each bar and {POWER_REPS} for "
                f"each power figure. Break date treated as announced."),
        alt=("A six-column table of eight year rows. The bar column falls from "
             "about 4.1 to 2.9; the 1.96 column from 26% to 15%; the two power "
             "columns rise from about 20% and 29% to 100%."),
        caption=("Conclude that the two columns in the middle are the ones "
                 "usually missing. Without the bar the power numbers are not "
                 "power; without the 1.96 column there is no reason to have "
                 "computed the bar."),
        mode="light", bold_cols=(0,),
        bold_cells={(hs.index("10") if "10" in hs else 3, 3)},
        path=str(IMG / f"a13-t1-power.{EXT}"))

    # ---- T2: who would have to carry the volume -------------------------
    srows = []
    for c in ("industrial", "general", "residential", "education", "total"):
        g = res["sales"]["cagr"][c]
        srows.append([CLASS_LABEL[c], f"{res['sales']['level_2025'][c]:,.0f}",
                      _pct(res["sales"]["share_2025"][c]),
                      f"{100 * g[f'2001_{BASE_YEAR}']:+.2f}%/yr",
                      f"{100 * g[f'2015_{BASE_YEAR}']:+.2f}%/yr",
                      f"{100 * g[f'2018_{BASE_YEAR}']:+.2f}%/yr"])
    figs["t2"], _ = charts.table_image(
        srows, header=SALES_HEADER,
        title="The class the argument is about has not grown since 2018",
        # Kept near the table's own width: the renderer sizes the figure to the
        # widest element, so a subtitle wider than the table leaves the last
        # column ending in the middle of the frame.
        subtitle=(f"Compound growth over three windows. The plan needs total "
                  f"consumption to grow "
                  f"{_rate(res['plan_twh_growth'][0])}-"
                  f"{_rate(res['plan_twh_growth'][1])} a year."),
        source=("Korea Power Exchange, EPSIS sales by contract class, 2001-2025. "
                "Sales, so excluding self-generation."),
        alt=("A six-column table of five contract classes. The industrial row, "
             "just over half of all sales, shows +2.87% a year since 2001 but "
             "-0.63% a year since 2018. General service grows in every window."),
        caption=("Conclude that the plan's arithmetic has to come from "
                 "somewhere specific. Industrial sales are more than half the "
                 "total and have been shrinking for seven years, which is the "
                 "fact the semiconductor case has to overturn — not a fact "
                 "against it, but the size of what is being claimed."),
        mode="light", bold_cols=(0,), bold_cells={(0, 5)},
        path=str(IMG / f"a13-t2-sales.{EXT}"))

    # ---- hero -----------------------------------------------------------
    # The drawings are literal on purpose. An earlier version used an abstract
    # forking line, a clock and two bars, and nothing on the card said the
    # subject was electricity — a reader had to get to the note to find out.
    # Pylons, cooling towers and a meter dial carry the same three numbers and
    # announce the sector in the first glance. No axes and no values inside the
    # frames, per the card's own rule: the numbers under the frames are the
    # measurement, the frames are the story.
    def _pylon(panel, m, x, base, h):
        """A lattice transmission tower, and the height of its top cross-arm.

        Drawn with a splayed lower body, a narrow upper body, two cross-arms
        projecting well past it and insulator drops hanging from the tips. The
        first attempt used short arms low on a squat body and read, correctly,
        as a picnic table.
        """
        waist = base + 0.42 * h
        wb, ww = 0.34 * h, 0.10 * h
        panel.plot([x - wb, x - ww], [base, waist], color=m.ink, lw=1.5)
        panel.plot([x + wb, x + ww], [base, waist], color=m.ink, lw=1.5)
        panel.plot([x - ww, x - ww * 0.7], [waist, base + h], color=m.ink, lw=1.5)
        panel.plot([x + ww, x + ww * 0.7], [waist, base + h], color=m.ink, lw=1.5)
        panel.plot([x - ww, x + ww], [waist, waist], color=m.ink, lw=1.2)
        panel.plot([x - wb, x + ww], [base, waist], color=m.muted, lw=0.8)
        panel.plot([x + wb, x - ww], [base, waist], color=m.muted, lw=0.8)
        arms = []
        for frac, reach in ((0.74, 0.30), (0.96, 0.20)):
            y = base + h * frac
            a = reach * h
            panel.plot([x - a, x + a], [y, y], color=m.ink, lw=1.3)
            for sx in (-a, a):
                panel.plot([x + sx, x + sx], [y, y - 0.07 * h],
                           color=m.ink, lw=1.0)
            arms.append(y - 0.07 * h)
        return arms[0]

    def demand_forks(panel, m):
        """A transmission line below, and the demand it carries above, forking.

        Two separate elements rather than one. Making the wire itself the demand
        line was tidier as an idea and unreadable as a picture: with towers of
        rising height the wire ran through their bodies instead of between their
        arms.
        """
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        panel.plot([0.3, 9.7], [1.0, 1.0], color=m.ink, lw=1.8)
        xs_t = (1.7, 5.0, 8.3)
        tips = [_pylon(panel, m, x, 1.0, 2.6) for x in xs_t]
        for (x0, y0), (x1, y1) in zip(zip(xs_t, tips), zip(xs_t[1:], tips[1:])):
            xs = np.linspace(x0, x1, 30)
            sag = 0.30 * np.sin(np.pi * (xs - x0) / (x1 - x0))
            panel.plot(xs, np.linspace(y0, y1, 30) - sag, color=m.muted, lw=1.3)
        panel.plot([0.4, 1.7], [tips[0] - 0.22, tips[0]], color=m.muted, lw=1.3)
        panel.plot([8.3, 9.6], [tips[-1], tips[-1] - 0.22], color=m.muted, lw=1.3)
        # the demand the line carries: steep, then flat, then two futures
        panel.plot([0.7, 3.5], [5.3, 7.9], color=m.series[0], lw=2.6)
        panel.plot([3.5, 6.2], [7.9, 8.4], color=m.series[0], lw=2.6)
        panel.plot([6.2, 9.4], [8.4, 9.8], color=m.series[1], lw=2.4,
                   ls=(0, (3, 2)))
        panel.plot([6.2, 9.4], [8.4, 8.8], color=m.ink, lw=1.5,
                   ls=(0, (1, 2)))

    def _tower(panel, m, cx, base, h, w, colour, dashed=False):
        """A cooling tower: waisted profile, drawn as two mirrored curves."""
        t = np.linspace(0, 1, 40)
        half = w * (1.0 - 0.9 * t + 0.75 * t * t)
        style = dict(color=colour, lw=2.0)
        if dashed:
            style["ls"] = (0, (2, 2))
        panel.plot(cx - half, base + h * t, **style)
        panel.plot(cx + half, base + h * t, **style)
        panel.plot([cx - half[-1], cx + half[-1]], [base + h, base + h], **style)
        return t, half

    def towers_building(panel, m):
        """One tower running, one still going up: the years in between."""
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        panel.plot([0.4, 9.6], [1.4, 1.4], color=m.ink, lw=1.8)
        _tower(panel, m, 3.0, 1.4, 5.0, 1.7, m.ink)
        for dx, dy, r in ((-0.5, 7.0, 0.75), (0.6, 7.7, 0.95), (-0.2, 8.6, 0.7)):
            th = np.linspace(0.15 * np.pi, 1.05 * np.pi, 30)
            panel.plot(3.0 + dx + r * np.cos(th), dy + 0.55 * r * np.sin(th),
                       color=m.muted, lw=1.4)
        t, half = _tower(panel, m, 7.1, 1.4, 5.0, 1.7, m.series[1], dashed=True)
        # Scaffolding follows the profile. A fixed span stuck out past the waist,
        # where the tower is at its narrowest.
        for frac in (0.22, 0.46, 0.70):
            i = int(frac * (t.size - 1))
            panel.plot([7.1 - half[i], 7.1 + half[i]],
                       [1.4 + 5.0 * t[i]] * 2, color=m.series[1], lw=1.0,
                       ls=(0, (1, 2)))

    def meter_dial(panel, m):
        """A meter whose needle sits in the marked band with nothing to read."""
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        cx, cy, r = 5.0, 3.4, 3.2
        th = np.linspace(np.pi, 0, 120)
        panel.plot(cx + r * np.cos(th), cy + r * np.sin(th), color=m.ink, lw=2.2)
        panel.plot([cx - r, cx + r], [cy, cy], color=m.ink, lw=1.6)
        for k in range(9):
            a = np.pi - k * np.pi / 8
            panel.plot([cx + 0.86 * r * np.cos(a), cx + r * np.cos(a)],
                       [cy + 0.86 * r * np.sin(a), cy + r * np.sin(a)],
                       color=m.muted, lw=1.2)
        band = np.linspace(np.pi * 0.30, np.pi * 0.02, 40)
        panel.plot(cx + 1.06 * r * np.cos(band), cy + 1.06 * r * np.sin(band),
                   color=m.series[1], lw=3.0)
        a = np.pi * 0.18
        panel.plot([cx, cx + 0.80 * r * np.cos(a)],
                   [cy, cy + 0.80 * r * np.sin(a)], color=m.series[1], lw=2.6)
        panel.plot([cx], [cy], marker="o", ms=5, color=m.ink)

    # `early` is recomputed here rather than passed in: figures() is called
    # independently of build() when only the images need re-rendering.
    early_yrs = TARGET_YEAR - round(res["crossing"][labels[0]])
    figs["hero"], _ = charts.strip_card(
        headline=(f"A forecast you can check "
                  f"{_WORDS.get(early_yrs, str(early_yrs))} years early"),
        panels=[
            (demand_forks, f"{_pp(req[labels[0]]['slope_change']).lstrip('+')} pp",
             "a year more slope"),
            (towers_building, f"{res['crossing'][labels[0]]:.0f}",
             "when the data can tell"),
            (meter_dial, _pct(res["power"]["15"]["size_of_1p96"]),
             "of nothing looks like something"),
        ],
        note=(f"Korea's 12th electricity plan needs peak demand to grow "
              f"{_pp(req[labels[0]]['slope_change']).lstrip('+')} to "
              f"{_pp(req[labels[1]]['slope_change']).lstrip('+')} percentage "
              f"points a year faster than the trend it is on. A correctly-sized "
              f"test reaches 80% power around "
              f"{res['crossing'][labels[0]]:.0f} — before the plan's own target "
              f"year. The textbook test would have called it far sooner, and "
              f"would have called it on nothing."),
        footer="The Standard Error", mode="light",
        alt=("A three-panel hand-drawn strip. The first shows three "
             "transmission pylons carrying sagging wires, with a demand line "
             "above them that climbs, flattens, then splits into a dashed "
             "steeper branch and a dotted flatter one, marked with a slope "
             "figure in percentage points. The second shows two cooling "
             "towers — one solid with steam rising from it, one drawn in "
             "dashed outline with scaffolding lines across it — marked with a "
             "year. The third shows a semicircular meter dial whose needle "
             "sits inside a thick marked band near the top of the scale, "
             "marked with a percentage."),
        caption="",
        path=str(IMG / f"a13-hero.{EXT}"))

    figs["_rows"] = {"power": prows, "sales": srows}
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute(verbose=False)
    figs = figures(res)

    years = np.array(res["years"])
    peak = np.array(res["peak_mw"])
    a = res["annual"]
    m1, m2 = res["monthly"]["1"], res["monthly"]["2"]
    ci = res["date_ci"]
    req = res["requirement"]
    labels = list(req)
    lo, hi = req[labels[0]], req[labels[1]]
    nc = res["noise_check"]
    pw = res["power"]
    cross = res["crossing"]
    early = TARGET_YEAR - round(cross[labels[0]])
    sh = res["shape"]
    sales = res["sales"]
    _, ext = _fitted_path(res)
    kink = years[a["tau"]]
    # Read off the scan rather than recomputed from the trim: the two agreed
    # here, and a trim change would have silently desynchronised them.
    n_dates = bd.scan(np.log(peak), kind="bend", trend=1, lags=A_LAGS,
                      trim=A_TRIM).dates.size

    # The spine, asserted rather than trusted.
    if not a["detected"]:
        raise AssertionError("the historical kink is no longer detected, which "
                             "is the premise of every forward number here")
    if a["resid_sd_null"] < 2 * a["resid_sd_alt"]:
        raise AssertionError(
            "the two residuals are no longer far apart, so section 4 has "
            "nothing to compare")
    if nc["alt"]["power_of_the_observed_kink"] < 0.95:
        raise AssertionError(
            "the known-answer check has failed: a kink with |t| = "
            f"{abs(a['t_hac']):.1f} must simulate as near-certain to detect")
    if not lo["slope_change"] > 0:
        raise AssertionError("the plan no longer asks for a faster trend")
    if abs(lo["slope_change"]) > abs(a["slope_change"]):
        raise AssertionError(
            "the plan's asserted change is now larger than the historical one, "
            "which reverses section 3")
    if cross[labels[0]] is None or not BASE_YEAR < cross[labels[0]] < TARGET_YEAR:
        raise AssertionError(
            "80% power no longer arrives inside the plan's own horizon, which "
            "is the post's title")
    if pw["15"]["size_of_1p96"] < 0.10:
        raise AssertionError("1.96 is no longer badly oversized here")
    if sales["cagr"]["industrial"][f"2018_{BASE_YEAR}"] > 0:
        raise AssertionError("industrial sales are growing again; section 8 "
                             "describes them as not")

    sections = [
        Section(heading="The claim, in the only form that can be checked",
                body=f"""
On 20 August 2026 Korea published its 12th Basic Plan for Long-term Electricity
Supply and Demand. The number that travelled was the 2040 peak-demand forecast:
**{lo['gw_published']} to {hi['gw_published']} GW**, against
{PLAN_PREVIOUS[0]}-{PLAN_PREVIOUS[1]} GW in the previous plan for a similar
horizon. Coverage converted the difference into nuclear reactors. Semiconductor
fabs and AI data centres were named as the reason.

A level in 2040 is not a checkable claim. A growth rate is.

Korea's annual peak load in {BASE_YEAR} was **{peak[-1] / 1000:.1f} GW** on the
series the power exchange publishes. Reaching the plan's pair by {TARGET_YEAR}
therefore requires **{_rate(lo['growth'])} to {_rate(hi['growth'])} a year**. The
trend the series is currently on grows at **{_rate(a['growth_post'])}**. So the
plan asserts a change of slope of **{_pp(lo['slope_change'])} to
{_pp(hi['slope_change'])} percentage points a year**, starting now.

That is the claim this post tests. Not whether it is right — nobody can know
that in 2026 — but the prior question: **when could data tell?**

One conversion happens before anything else, and it is stated rather than
buried. The power exchange reports the load settled through the exchange; the
plan forecasts a wider quantity that also carries behind-the-meter generation.
For 2023 the two differ by 5% — 98.3 GW against
{peak[np.where(years == 2023)[0][0]] / 1000:.1f} GW. Every growth rate below is
computed inside one series, so the gap does not touch them. The bridge is applied
exactly once, to put the plan's endpoint on the observed series' basis, and if
behind-the-meter generation keeps growing then the plan needs *less* growth in the
observed series than the figures here imply. That is the largest single
uncertainty in the comparison and it runs in the plan's favour.
""", figures=[figs["f1"]]),

        Section(heading=f"Growth already bent once, in {kink}", body=f"""
Before asking about a future change of slope, establish whether this series has
ever made one.

Fit a single linear trend in logs with one change of slope, searched over every
admissible year. The answer is unambiguous: the best date is **{kink}**, growth
of **{_rate(a['growth_pre'])} a year before it and {_rate(a['growth_post'])}
after** — a slope change of {_pp(a['slope_change'])} points. Its Newey-West
t-statistic is **{abs(a['t_hac']):.2f}** against a critical value of
{a['calibrated']['sup']:.2f}, simulated from the series' own residuals and
corrected for having searched {n_dates} dates.
Nothing marginal about it.

The previous post in this series found that a bend's date is usually the
worst-estimated thing in the regression — a 90% interval four years wide on a
26-year series. Here it is not: resampling puts the middle half of the estimates
on **{years[ci['q25']]}** alone and the 90% interval at
{years[ci['lo']]}-{years[ci['hi']]}. The difference is signal size. That kink is
enormous relative to the noise around it, and a large kink is dated precisely.
The earlier finding was conditional on magnitude, not a law.

Two robustness notes, both of which cut against overclaiming. On the monthly
series of average load, the same fit puts the kink in
{pd.to_datetime(res['monthly_dates'][m1['tau']]):%B %Y} with a slope change of
{_pp(m1['slope_change'])} points and |t| = {abs(m1['t_hac']):.1f} — the same
event, a year or two later, on a different measure. And if the trend is allowed
to be quadratic instead of linear, the kink stops being separately detectable
(|t| = {abs(m2['t_hac']):.2f} against {m2['calibrated']['sup']:.2f}), because a
quadratic *is* a smooth deceleration and the two specifications are competing to
explain the same curvature. What survives both readings is that growth
decelerated, and by a lot.

**No cause is attributed here.** The date is interesting and the temptation is
obvious; this post has no identification strategy for it and so says nothing
about it.
"""),

        Section(heading="The plan asks for a quarter of that, upward", body=f"""
Now the two numbers can be put side by side, in the same units.

The series changed slope by **{_pp(a['slope_change'])} points** in {kink}. The
plan asserts **{_pp(lo['slope_change'])} to {_pp(hi['slope_change'])} points**,
in the other direction, starting {BASE_YEAR + 1}.

So the plan is not asking for something the series has never done. It is asking
for roughly a **quarter** of the change it already made, with the sign reversed.
Anyone reaching for "the forecast is absurd" has to get past that, and the honest
version of the sceptical case is narrower: the plan asks for a partial reversal
of a deceleration whose cause nobody in this post has identified.

Which is exactly why the question worth asking is about detectability rather than
plausibility. A claim this size, on a series this noisy, has a date on which it
becomes checkable, and that date is computable now.
"""),

        Section(heading="The noise model decides the answer, and the obvious "
                        "choice is wrong", body=f"""
Every power calculation needs a noise process, and there are two candidates
sitting right there. You can resample the residual of the no-break model — a
trend and nothing else — or the residual of the model you actually fitted, trend
plus the {kink} kink.

They are not close. The no-break residual has a standard deviation of
**{_pct(nc['null']['sd'], 1)}**; the fitted model's is
**{_pct(nc['alt']['sd'], 1)}**. Nearly three to one, because the no-break residual
still contains the kink. Its autocorrelation reflects that too: it is worth
{a['n_eff_null']:.0f} independent observations out of {a['n']}, against
{a['n_eff_alt']:.0f} for the fitted residual, whose variance inflation is
{a['vif_alt']:.2f} — after one trend and one kink, Korea's annual peak load is
very close to white noise.

The choice is not a matter of taste, and there is a check that settles it. **The
kink the data already shows must simulate as detectable.** Its observed
t-statistic was {abs(a['t_hac']):.1f}, so a simulation that says such a kink is
hard to see is wrong about something. Run it both ways:

- fitted-model residual: the observed kink is detected
  **{_pct(nc['alt']['power_of_the_observed_kink'])}** of the time
- no-break residual: **{_pct(nc['null']['power_of_the_observed_kink'])}**

The second is the wrong answer to a question whose answer is known. And the
consequence is not small. Carried forward, the no-break residual puts the
probability of detecting the plan's central path by {TARGET_YEAR} at
**{_pct(res['power_null_resid_15y'])}** — which would have made this post a
different post, with the conclusion "unknowable" and no way to notice the error.
""", figures=[figs["f3"]]),

        Section(heading=f"The answer arrives around {cross[labels[0]]:.0f}",
                body=f"""
With the right noise process, the calculation is straightforward. Put a kink at
{BASE_YEAR + 1} of the size the plan asserts, simulate, and test it — against a
critical value recalibrated at every horizon, because the bar a design needs
changes as the sample grows.

Two things make this the *favourable* case, deliberately. The date is treated as
known, because the plan announces it, so there is no penalty for having searched
dates. And the test is one-shot at each horizon rather than repeated annually.
A forecast that cannot be confirmed under those conditions is not going to be
confirmed under worse ones.

{md_table(POWER_HEADER, figs['_rows']['power'])}

The plan's managed path reaches 80% power at
**{cross[labels[0]]:.0f}**; the baseline path, being a larger claim, at
{cross[labels[1]]:.0f}. The plan's own target year is {TARGET_YEAR} —
**{_WORDS.get(early, str(early))} years after the question stops being open**.

Put the other way round, in the units of the claim itself. By {BASE_YEAR + 5}
the smallest slope change this design can see at all is {_pp(pw['5']['mde'])}
points a year — more than twice what the plan asserts — so the arguments of the
next five years will be conducted on data that cannot settle them. By
{BASE_YEAR + 10} that floor has fallen to {_pp(pw['10']['mde'])}, which is the
plan's {_pp(lo['slope_change'])} almost exactly; that coincidence is not a
coincidence, it is what the crossing year means. Only by {BASE_YEAR + 13}, at
{_pp(pw['13']['mde'])}, is the claim comfortably inside what the design can
resolve.

Two horizons deserve naming, because they are when the plan will next be revised.
These plans arrive roughly every two years. At {BASE_YEAR + 8} the probability of
detection is {_pct(pw['8'][labels[0]])}; at {BASE_YEAR + 10} it is
{_pct(pw['10'][labels[0]])}. So the 12th plan's central claim becomes testable
somewhere around the 16th or 17th plan, and not at the 13th or 14th.
""", figures=[figs["f2"]]),

        Section(heading="The conventional test would call it early, repeatedly",
                body=f"""
The third column of that table is the one that would be missing from almost any
version of this exercise.

Against the textbook 1.96, this design rejects the no-change null
**{_pct(pw['5']['size_of_1p96'])} of the time at {BASE_YEAR + 5}** and
{_pct(pw['15']['size_of_1p96'])} at {TARGET_YEAR}, on data where nothing has
changed at all. The correctly-sized bar is {pw['5']['critical']:.2f} falling to
{pw['15']['critical']:.2f}, never anywhere near 1.96.

So an analyst who fits this regression each year and reads the t-statistic against
the usual threshold will announce that the plan is being confirmed about one year
in five, regardless of what the economy does. Over a fifteen-year horizon that is
not a small risk of a false positive; it is a near-certainty of several, arriving
at unpredictable times, each one publishable.

The bar itself carries error, and it is worth quoting rather than hiding: across
{res['bar_stability']['paths']} independent simulations of the same null the
{BASE_YEAR + 10} bar runs {res['bar_stability']['lo']:.2f} to
{res['bar_stability']['hi']:.2f}, standard deviation
{res['bar_stability']['sd']:.2f}. That is small enough not to move any conclusion
above, and large enough that a margin of a tenth against it would mean nothing.
"""),

        Section(heading="Part of the forecast is a claim about load shape",
                body=f"""
One component of the plan's peak number is not about how much electricity Korea
uses. It is about when.

The ratio of the annual peak to average load rose from
**{sh['peak_to_avg'][0]:.3f}** in {sh['years'][0]} to
**{sh['peak_to_avg'][-1]:.3f}** in {sh['years'][-1]}. The same fact stated as a
load factor: {sh['load_factor'][0]:.1f}% falling to
{sh['load_factor'][-1]:.1f}%. Korean load has become steadily peakier for twenty
years, so the peak has grown faster than the volume underneath it.

The plan's own two published figures — {lo['gw_published']} GW of peak and
{PLAN_TWH[0]} TWh of consumption — imply a load factor of
**{np.mean(res['plan_load_factor']):.1f}%** by {TARGET_YEAR}. Same direction,
continued at roughly the same pace.

This is a point in the plan's favour and it belongs in the post as one. The
peakiness assumption is the least aggressive thing in the forecast: it extrapolates
a twenty-year trend rather than reversing one. And it cuts the other way too — if
peakiness saturates instead, the plan's peak number requires *more* volume growth
than it appears to, not less.
""", figures=[figs["f4"]]),

        Section(heading="Who would have to carry the volume", body=f"""
That leaves the volume question, and it is worth putting the load classes side by
side because the aggregate hides the thing the argument is actually about.

{md_table(SALES_HEADER, figs['_rows']['sales'])}

Industrial sales are **{_pct(sales['share_2025']['industrial'])} of all
electricity sold** in Korea and have **shrunk
{abs(100 * sales['cagr']['industrial'][f'2018_{BASE_YEAR}']):.2f}% a year since
2018**. Fabs and data centres are industrial and general-service load, so the
plan's arithmetic requires that number to turn around and then some: total
consumption has to grow {_rate(res['plan_twh_growth'][0])} to
{_rate(res['plan_twh_growth'][1])} a year to reach the plan's own {TARGET_YEAR}
figure, against {100 * sales['cagr']['total'][f'2018_{BASE_YEAR}']:+.2f}% since
2018 and {100 * sales['cagr']['total'][f'2001_{BASE_YEAR}']:+.2f}% over the full
record.

Stated carefully: this is not evidence against the plan. New fabs and new data
centres are exactly the kind of thing that would break a seven-year pattern, and
a contract-class series cannot see a facility that has not been energised. It is a
measure of the size of what is being claimed — and it names the series to watch,
which is the industrial class rather than the total.
"""),

        Section(heading="What to do", body=f"""
Four things, and the first is the one this post nearly got wrong.

**Check the noise model against something you already know.** A power calculation
inherits every property of the residual you feed it, and the residual of the model
you are testing *against* still contains the effect you are testing *for*. The
check is cheap and decisive: take an effect the data has already established, run
it through the machinery, and confirm it comes back as detectable. Here that check
moved the headline from "unknowable" to "{cross[labels[0]]:.0f}".

**Convert a forecast into a slope before arguing about it.** A level in a distant
year is unfalsifiable in practice; the growth rate it implies is not, and
subtracting the trend already in the data turns a twenty-percent revision into
{_pp(lo['slope_change'])} points a year — a quantity with a standard error.

**Compute the date the argument ends.** Every long-horizon forecast has one. It is
usually earlier than the target year, which is useful, and it is usually later
than the next revision of the forecast, which is the part worth knowing before the
next revision arrives.

**Calibrate, or do not run the test.** On this design the conventional threshold
fires on nothing {_pct(pw['5']['size_of_1p96'])} to
{_pct(pw['15']['size_of_1p96'])} of the time. A fifteen-year monitoring exercise
built on it will produce confirmations whatever happens, and the confirmations
will be indistinguishable from the real thing.

The next post in this series stays with break dates and drops the assumption that
there is only one. Two kinks are not one kink fitted twice, and searching over
pairs of dates costs more than most people expect.
"""),
    ]

    post = Post(
        # Spelled out: a bare numeral reads as a version number in a title, and
        # "5 Years Early" next to two four-digit years is three numbers in a row.
        title=(f"Korea's {TARGET_YEAR} Power Forecast Becomes Checkable in "
               f"{cross[labels[0]]:.0f}, "
               f"{_WORDS.get(early, str(early)).title()} Years Early"),
        slug="korea-power-forecast-checkable-in-2035",
        subtitle=(f"The 12th electricity plan needs peak demand to grow "
                  f"{_pp(lo['slope_change'])} to {_pp(hi['slope_change'])} "
                  f"percentage points a year faster than the trend it is on. "
                  f"That is a falsifiable claim, and the date it stops being "
                  f"open is computable now."),
        author="Jongha Jeon",
        summary=(f"Korea's 12th Basic Plan forecasts {lo['gw_published']}-"
                 f"{hi['gw_published']} GW of peak demand in {TARGET_YEAR}, up "
                 f"about a fifth from the previous plan. Converted into a growth "
                 f"rate that is {_rate(lo['growth'])}-{_rate(hi['growth'])} a "
                 f"year against a fitted trend of {_rate(a['growth_post'])}, so "
                 f"the plan asserts a slope change of {_pp(lo['slope_change'])} "
                 f"to {_pp(hi['slope_change'])} points — about a quarter of the "
                 f"{_pp(a['slope_change'])}-point deceleration the series itself "
                 f"made in {kink}, with the sign reversed. A correctly-sized "
                 f"test reaches 80% power around {cross[labels[0]]:.0f}, "
                 f"{_WORDS.get(early, str(early))} years before the plan's own "
                 f"target date. The whole result "
                 f"turns on which residual is resampled as noise: the no-break "
                 f"model's residual still contains the {kink} kink and gives "
                 f"{_pct(res['power_null_resid_15y'])} power at {TARGET_YEAR} "
                 f"instead, an error caught only by checking that a kink the "
                 f"data already shows simulates as detectable. Meanwhile the "
                 f"textbook 1.96 fires on unchanged data "
                 f"{_pct(pw['15']['size_of_1p96'])} of the time."),
        tags=["electricity", "structural breaks", "statistical power",
              "korea", "public data"],
        data_sources=SOURCES,
        licence_warnings=[
            "Korea Power Exchange statistics are 공공누리 (KOGL) licensed by "
            "table. This post publishes statistics computed from the exports, "
            "not the underlying tables.",
        ],
        sections=sections,
        table_figures=[figs["t1"], figs["t2"]],
        reproducibility={
            "seed": SEED,
            "peak_window": f"{years[0]}-{years[-1]} ({len(years)} completed "
                           f"years; {BASE_YEAR + 1} excluded as running)",
            "sales_window": f"{sales['years'][0]}-{sales['years'][1]}",
            "monthly_window": f"{res['monthly_dates'][0][:7]} to "
                              f"{res['monthly_dates'][-1][:7]} "
                              f"({len(res['monthly_dates'])} months)",
            "annual_design": "linear trend in log plus one slope change; "
                             f"Newey-West {A_LAGS} lags, bootstrap block "
                             f"{A_BLOCK} years",
            "monthly_design": f"trend degrees 1 and 2 plus eleven monthly "
                              f"dummies; Newey-West {M_LAGS} lags, block "
                              f"{M_BLOCK} months",
            "basis_bridge": f"{res['basis_bridge']:.4f} (98.3 GW plan basis / "
                            f"93.6 GW exchange basis, 2023)",
            "replications": {"searched critical values": SUP_REPS,
                             "fixed critical values": CAL_REPS,
                             "power": POWER_REPS, "date bootstrap": DATE_REPS,
                             "minimum detectable": MDE_REPS},
            "modules": "standarderror/ts/bend.py, standarderror/ts/detect.py, "
                       "standarderror/sources/korea_power.py",
            "tests": "tests/test_bend.py",
        },
        min_words=2000,
        max_words=3200,
    )
    post.hero = figs["hero"]
    _check_table_placement(post)
    return post


def _check_table_placement(post: Post) -> None:
    """Table images are matched to markdown tables positionally, so verify."""
    import re
    from standarderror.render import publish

    was = post.draft
    post.draft = False
    try:
        body = publish.medium_bundle(
            post, out_dir=se.SETTINGS.build_dir / "_placement21").read_text()
    finally:
        post.draft = was
    heading, seen = "", {}
    for line in body.split("\n"):
        if line.startswith("## "):
            heading = line[3:].strip()
        mm = re.search(r"!\[[^\]]*\]\(([^)]+)\)", line)
        if mm:
            seen[mm.group(1).rsplit("/", 1)[-1]] = heading
    for name, needle in ((f"a13-t1-power.{EXT}", "answer arrives"),
                         (f"a13-t2-sales.{EXT}", "carry the volume")):
        where = seen.get(name)
        if where is None:
            raise AssertionError(f"{name} never reached the rendered body")
        if needle.lower() not in where.lower():
            raise AssertionError(
                f"{name} landed under {where!r}; table_figures is matched "
                f"positionally, so check its order")
