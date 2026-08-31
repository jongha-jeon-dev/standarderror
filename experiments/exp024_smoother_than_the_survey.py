"""exp024 — the published unemployment rate against the survey's own error.

Where this comes from
---------------------
BLS told Congress in April 2026 that the Current Population Survey has lost
precision: response rates down from the low 90s to the upper 60s, one respondent
now standing for 3,500 people instead of 2,100, and a month-on-month change that
used to be detectable in one month now needing two. The obvious question is
whether the published series shows it, and the published series is right there.

Two corrections happened on the way, and both are in the post because both are
the kind of error the exercise is about.

The number that was not in the documents
----------------------------------------
An earlier version of this file paired two figures — 0.18 percentage point
(the report) and 0.30 (the Employment Situation technical note) — and read their
ratio, 1.67, as the degradation. Both halves are wrong. The 0.30 is stated *at
an unemployment rate of around 6.0 percent* and has to be rescaled by
sqrt(p(1-p)); and the report never gives a current one-month threshold at all,
only that the same change now takes two months. Corrected, the two documents
land on 0.253 and 0.255 — they agree to half a percent, and the degradation is
about 1.41x, not 1.67x.

What the series says
--------------------
The second difference of a monthly series removes any local trend and leaves the
noise, and a robust scale of it estimates the sampling error's standard
deviation without any access to the microdata. Run through to a detectable
change, on the decade ending at the report's own reference year:

* **1997-2006: 0.186 pp**, against BLS's stated 0.18 for "20 years ago". Three
  percent apart, from the published series alone. That agreement is what earns
  the right to read the modern disagreement as a finding.
* **2011-2019: 0.195 pp** and **2016-2026: 0.237 pp**, against BLS's stated
  0.296 and 0.264 for those windows' unemployment rates. The published series
  carries two thirds to nine tenths of the error BLS attributes to it.
* The **unadjusted** series does not have this problem: its implied threshold
  runs 0.21 to 0.32 and brackets or exceeds the stated figure in every window.

So the raw estimate carries the noise the agency says it has, and the published,
seasonally adjusted series carries less. That is what seasonal adjustment does —
X-13's filters average a shifting timespan, and concurrent adjustment lets the
current month help estimate its own seasonal factor — and it means the headline
rate is smoother than the survey behind it.

What could not be pinned
------------------------
How much the adjustment removes. A fixed month-of-year benchmark says 10% in
some windows and 39% in others, and the spread is a property of the fit window
rather than of the filter, so no point estimate here would survive a reader
changing the window. Reported as a range with the reason, and the attempt to
date a step at the 2003-04 methodology change failed.

Licence
-------
BLS is a work of the US federal government and carries no domestic copyright, so
values are reproduced rather than only summarised.

Run: `standarderror run exp024_smoother_than_the_survey --publish`
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.sources import us_labor as ul
from standarderror.ts import noisescale as ns
from standarderror.ts import seasonal as sz
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
#: `Post.date` defaults to today, which is correct exactly once.
POST_DATE = date(2026, 8, 27)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed
DATA = Path("data/us_labor")
CACHE = se.SETTINGS.build_dir / "cache" / "exp024.json"

SA_CODE, NSA_CODE = "LNS14000000", "LNU04000000"

#: BLS's report is dated April 2026 and says "20 years ago", so its reference
#: year is 2006 and the comparable window is the decade ending there. Fixed
#: before looking at any of the numbers, because a window chosen afterwards is
#: not a test.
REFERENCE_YEAR = 2006
WINDOWS = (
    ("1997-2006", 1997, 2006, "the report's reference decade"),
    ("2011-2019", 2011, 2019, "recent, no pandemic months"),
    ("2016-2026", 2016, 2026, "recent, all months"),
)
#: Rolling window for the figures: ten years, stepped a year at a time.
ROLL_WINDOW, ROLL_STRIDE = 120, 12
#: Shorter window for the p(1-p) scatter, to get more points across the cycle.
SCATTER_WINDOW, SCATTER_STRIDE = 84, 6
RHO_SWEEP = (0.0, 0.20, 0.35, 0.50, 0.60, 0.75)
DEFAULT_RHO = 0.35
WEDGE_WINDOWS = ((1950, 1959), (1960, 1969), (1970, 1979), (1980, 1989),
                 (1990, 1999), (2000, 2009), (2010, 2019),
                 (1976, 1989), (1990, 2003), (2004, 2017), (2006, 2019))
WEDGE_REPS = 800


def _vintage() -> dict:
    out = {}
    for code in (SA_CODE, NSA_CODE):
        p = DATA / f"{code}.xlsx"
        if p.exists():
            out[code] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "bytes": p.stat().st_size}
    return out


def _config_key() -> str:
    blob = json.dumps({"v": 1, "windows": WINDOWS, "roll": ROLL_WINDOW,
                       "stride": ROLL_STRIDE, "scatter": SCATTER_WINDOW,
                       "rho": RHO_SWEEP, "wedge": WEDGE_WINDOWS,
                       "reps": WEDGE_REPS, "seed": SEED,
                       "vintage": _vintage()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def series(code: str) -> pd.Series:
    path = DATA / f"{code}.xlsx"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. This container cannot reach data.bls.gov; open "
            f"https://data.bls.gov/timeseries/{code}, set the year range to the "
            f"full history, and download the Excel report there.")
    return ul.monthly_series(path)


def _window(s: pd.Series, y0: int, y1: int) -> pd.Series:
    return s[(s.index.year >= y0) & (s.index.year <= y1)]


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

    sa, nsa = series(SA_CODE), series(NSA_CODE)
    say(f"SA {sa.index.min().date()}..{sa.index.max().date()} "
        f"n={int(sa.notna().sum())}; missing {sorted(ul.MISSING_MONTHS)}")

    out = {"key": key, "vintage": _vintage(),
           "coverage": {"start": str(sa.index.min().date()),
                        "end": str(sa.index.max().date()),
                        "n": int(sa.notna().sum()),
                        "missing": sorted(ul.MISSING_MONTHS)}}

    # ---------------------------------------------------------------- 1. the
    # two documents, made comparable.
    recent = _window(sa, 2021, 2026)
    rate_now = float(recent.mean())
    out["documents"] = {
        "rate_now": rate_now,
        "naive_ratio": float(ul.CPS_2026["ci90_change"]
                             / ul.CPS_2026["detectable_change_then"]),
        **ul.cps_detectable_now(rate=rate_now),
    }
    say(f"documents reconcile: {out['documents']['from_technical_note']:.4f} vs "
        f"{out['documents']['from_report']:.4f}")

    # ---------------------------------------------------------------- 2. the
    # fixed windows.
    out["windows"] = []
    for label, y0, y1, note in WINDOWS:
        a, n = _window(sa, y0, y1), _window(nsa, y0, y1)
        bench = sz.month_dummy_adjust(n)
        s_a, s_b = sz.run_scale(a), sz.run_scale(bench)
        u = float(a.mean())
        # Which BLS figure this window is comparable to. The technical note's
        # interval describes the survey *now*, so applying it to a window ending
        # twenty years ago would repeat exactly the mistake this post is about;
        # the reference window is compared with the report's own "then" figure.
        if y1 <= REFERENCE_YEAR:
            stated = float(ul.CPS_2026["detectable_change_then"])
            stated_source = "report, 20 years ago"
        else:
            stated = ns.rescale_for_rate(ul.CPS_2026["ci90_change"],
                                         stated_at=6.0, actual=u)
            stated_source = "technical note, rescaled"
        row = {"label": label, "note": note, "y0": y0, "y1": y1,
               "n": int(a.notna().sum()), "u": u,
               "sigma_sa": s_a, "sigma_bench": s_b,
               "stated": stated, "stated_source": stated_source,
               "rho": {}}
        for rho in RHO_SWEEP:
            row["rho"][f"{rho:.2f}"] = {
                "sa": ns.implied_detectable(s_a, rho=rho),
                "bench": ns.implied_detectable(s_b, rho=rho)}
        row["implied_sa"] = row["rho"][f"{DEFAULT_RHO:.2f}"]["sa"]
        row["implied_bench"] = row["rho"][f"{DEFAULT_RHO:.2f}"]["bench"]
        row["share_of_stated"] = row["implied_sa"] / row["stated"]
        out["windows"].append(row)
        say(f"{label}: u={u:.2f}% implied {row['implied_sa']:.3f} "
            f"stated {row['stated']:.3f}")

    ref = next(w for w in out["windows"] if w["y1"] == REFERENCE_YEAR)
    out["reference"] = {
        "label": ref["label"], "implied": ref["implied_sa"],
        "stated_then": ul.CPS_2026["detectable_change_then"],
        "agreement": ref["implied_sa"] / ul.CPS_2026["detectable_change_then"]}

    # ---------------------------------------------------------------- 3. rolling
    roll = []
    for end in range(ROLL_WINDOW, len(sa) + 1, ROLL_STRIDE):
        a, n = sa.iloc[end - ROLL_WINDOW:end], nsa.iloc[end - ROLL_WINDOW:end]
        s_a = sz.run_scale(a)
        if not np.isfinite(s_a):
            continue
        try:
            s_b = sz.run_scale(sz.month_dummy_adjust(n))
        except ValueError:
            s_b = float("nan")
        roll.append({"end": int(a.index[-1].year), "u": float(a.mean()),
                     "implied_sa": ns.implied_detectable(s_a, rho=DEFAULT_RHO),
                     "implied_bench": (ns.implied_detectable(s_b, rho=DEFAULT_RHO)
                                       if np.isfinite(s_b) else float("nan"))})
    out["rolling"] = roll
    say(f"rolling: {len(roll)} windows")

    # ---------------------------------------------------------------- 4. does the
    # noise scale track sqrt(p(1-p))?
    scatter = []
    for end in range(SCATTER_WINDOW, len(sa) + 1, SCATTER_STRIDE):
        a = sa.iloc[end - SCATTER_WINDOW:end]
        if a.notna().sum() < SCATTER_WINDOW - 2:
            continue
        s_a = sz.run_scale(a, min_run=40)
        if not np.isfinite(s_a):
            continue
        u = float(a.mean()) / 100.0
        scatter.append({"end": float(a.index[-1].year + a.index[-1].month / 12),
                        "u": u * 100.0, "sqrt_pq": float(np.sqrt(u * (1 - u))),
                        "sigma_sa": s_a})
    x = np.log([r["sqrt_pq"] for r in scatter])
    y = np.log([r["sigma_sa"] for r in scatter])
    slope, intercept = np.polyfit(x, y, 1)
    late = [r for r in scatter if r["end"] >= 1990]
    xl = np.log([r["sqrt_pq"] for r in late])
    yl = np.log([r["sigma_sa"] for r in late])
    out["scaling"] = {
        "points": scatter, "slope": float(slope), "intercept": float(intercept),
        "r": float(np.corrcoef(x, y)[0, 1]),
        "slope_post1990": float(np.polyfit(xl, yl, 1)[0]),
        "r_post1990": float(np.corrcoef(xl, yl)[0, 1]),
        "n": len(scatter), "n_post1990": len(late),
        "u_lo": min(r["u"] for r in scatter), "u_hi": max(r["u"] for r in scatter),
        "pq_ratio": float(max(r["sqrt_pq"] for r in scatter)
                          / min(r["sqrt_pq"] for r in scatter)),
        "sigma_lo": min(r["sigma_sa"] for r in scatter),
        "sigma_hi": max(r["sigma_sa"] for r in scatter),
    }
    say(f"scaling slope {out['scaling']['slope']:+.3f} (theory +1)")

    # ---------------------------------------------------------------- 5. the wedge
    # that could not be pinned.
    wedges = []
    for y0, y1 in WEDGE_WINDOWS:
        a, n = _window(sa, y0, y1), _window(nsa, y0, y1)
        if a.notna().sum() < 60:
            continue
        r = sz.wedge_interval(a, n, block=24, reps=WEDGE_REPS, seed=SEED + y0)
        wedges.append({"label": f"{y0}-{y1}", "years": y1 - y0 + 1,
                       "n": r["n"], "removed": r["removed"],
                       "lo": r["lo"], "hi": r["hi"],
                       "sigma_sa": r["sigma_sa"],
                       "sigma_bench": r["sigma_benchmark"]})
    out["wedges"] = wedges
    out["wedge_range"] = {"lo": min(w["removed"] for w in wedges),
                          "hi": max(w["removed"] for w in wedges)}
    say(f"wedge range {out['wedge_range']['lo']:+.3f} to "
        f"{out['wedge_range']['hi']:+.3f}")

    # ---------------------------------------------------------------- 6. the
    # lattice, kept because it is the estimator that had to be abandoned.
    a = _window(sa, 1948, 2026).dropna().to_numpy()
    d2 = np.diff(a, n=2)
    root6 = float(np.sqrt(ns.SECOND_DIFF_FACTOR))
    out["lattice"] = {
        # Divided by sqrt(6) so these are per-month sigmas, the same convention
        # the snippet prints and the prose quotes.
        "mad_values": sorted({round(float(ns.mad_scale(d2[i:i + 120]) / root6), 4)
                              for i in range(0, len(d2) - 120, 120)}),
        "resolution": ns.lattice_resolution(0.1),
        "rounding_floor": ns.rounding_floor(0.1),
        "robust_full": float(ns.robust_scale(d2) / root6),
        "mad_full": float(ns.mad_scale(d2) / root6),
    }
    say(f"lattice: MAD takes {len(out['lattice']['mad_values'])} distinct values")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    """Three charts, two tables, one hero."""
    doc = res["documents"]
    ref = res["reference"]
    wins = {w["label"]: w for w in res["windows"]}
    src = (f"BLS Current Population Survey, series {SA_CODE} and {NSA_CODE}, "
           f"{res['coverage']['start']} to {res['coverage']['end']}.")
    out = {}

    # --- f1: the rolling threshold, against the two anchors ------------------
    roll = res["rolling"]
    frame = pd.DataFrame(
        {"published (seasonally adjusted)": [r["implied_sa"] for r in roll],
         "unadjusted, own seasonal removed": [r["implied_bench"] for r in roll]},
        index=pd.Index([r["end"] for r in roll], name="last year of the window"))

    #: The unadjusted line runs off the top in every window containing the
    #: pandemic. Clipping the axis keeps the three decades the argument is about
    #: readable; the peak is stated in the annotation rather than hidden.
    peak = max(r["implied_bench"] for r in roll if np.isfinite(r["implied_bench"]))
    ylim = (0.15, 0.40)

    def anchors(fig, ax):
        # Only two points on this chart come from BLS rather than from the
        # series, and drawing them as a line would imply a history the documents
        # do not contain.
        ax.plot([REFERENCE_YEAR], [ref["stated_then"]], marker="D", ms=7,
                color="0.2", zorder=5, lw=0)
        ax.annotate(f"BLS: {ref['stated_then']:.2f} pp,\n\"20 years ago\"",
                    (REFERENCE_YEAR, ref["stated_then"]), xytext=(-11, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, color="0.2")
        last = roll[-1]["end"]
        ax.plot([last], [doc["mean"]], marker="D", ms=7, color="0.2",
                zorder=5, lw=0)
        ax.annotate(f"BLS: {doc['mean']:.2f} pp,\nnow",
                    (last, doc["mean"]), xytext=(-11, 6),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=8.5, color="0.2")
        ax.annotate(f"unadjusted reaches {peak:.2f} in windows\n"
                    f"containing 2020-21, off scale",
                    (roll[0]["end"], ylim[1]), xytext=(4, -6),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=8.0, color="0.45")

    out["f1"] = charts.lines(
        frame,
        title="What the series says its own sampling error is",
        subtitle=("Detectable one-month change at 90% confidence, implied by a "
                  "robust scale of the series' own second differences over "
                  "ten-year windows. The two diamonds are BLS's figures, not "
                  "the series'."),
        xlabel="last year of the ten-year window",
        ylabel="detectable one-month change (pp)",
        source=src, decorate=anchors, direct_labels=False, ylim=ylim,
        alt=("Two lines of implied detectable change over time, the unadjusted "
             "series above the published one, with two diamond markers for "
             "BLS's stated figures."),
        caption=(f"At the report's own reference decade the published series "
                 f"implies {ref['implied']:.3f} pp against BLS's stated "
                 f"{ref['stated_then']:.2f}. That agreement is what makes the "
                 f"modern gap readable as a finding rather than as an "
                 f"estimator failure."),
        path=str(IMG / f"a17-f1-threshold.{EXT}"))[0]

    # --- f2: the wedge that could not be pinned -----------------------------
    wedges = res["wedges"]
    labels = [f"{w['label']}  ({w['years']}y)" for w in wedges]
    values = [w["removed"] for w in wedges]
    errs = np.array([[w["removed"] - w["lo"] for w in wedges],
                     [w["hi"] - w["removed"] for w in wedges]])
    errs = np.clip(errs, 0.0, None)
    out["f2"] = charts.ranked_bars(
        labels, values, errors=errs, signed=True, sort="none",
        value_fmt="+.2f",
        title="How much the adjustment removes, window by window",
        subtitle=("Share of a fixed month-of-year benchmark's month-to-month "
                  "noise scale that the published series does not have. Bars "
                  "are 90% block-bootstrap intervals. The direction is "
                  "consistent; the size is a property of the window."),
        xlabel="share of the benchmark's noise scale removed",
        source=src,
        alt=("Horizontal bars of the removed share by window, all but one "
             "positive, ranging from about zero to nearly forty percent."),
        caption=("Ten-year windows say 16 to 39 percent and fourteen-year "
                 "windows say 10 to 16 percent. A quantity that moves with the "
                 "fit window is not a measurement of the filter, so this is "
                 "published as a range and the attempt to date a step at the "
                 "2003-04 methodology change is reported as having failed."),
        path=str(IMG / f"a17-f2-wedge.{EXT}"))[0]

    # --- f3: the one nuisance parameter, swept -------------------------------
    rhos = [float(r) for r in RHO_SWEEP]
    frame3 = pd.DataFrame(
        {f"{w['label']} (published)": [w["rho"][f"{r:.2f}"]["sa"] for r in rhos]
         for w in res["windows"]},
        index=pd.Index(rhos, name="assumed lag-1 autocorrelation of the noise"))
    frame3["2011-2019 (unadjusted)"] = [
        wins["2011-2019"]["rho"][f"{r:.2f}"]["bench"] for r in rhos]

    def stated(fig, ax):
        for label, key in (("stated, 2011-2019", "2011-2019"),
                           ("stated, 2016-2026", "2016-2026")):
            v = wins[key]["stated"]
            ax.axhline(v, color="0.45", lw=1.0, ls=(0, (4, 3)))
            ax.annotate(f"{label}: {v:.3f}", (rhos[0], v), xytext=(2, 4),
                        textcoords="offset points", fontsize=8.0, color="0.35")

    out["f3"] = charts.lines(
        frame3,
        title="The answer does not depend on the number nobody knows",
        subtitle=("The lag-1 autocorrelation of the sampling error enters twice, "
                  "in opposite directions: it raises the level noise implied by "
                  "a second difference and lowers the noise in a month-on-month "
                  "difference. Sweeping it over the whole plausible range moves "
                  "the answer by a few percent."),
        xlabel="assumed lag-1 autocorrelation of the sampling error",
        ylabel="implied detectable one-month change (pp)",
        source=src, decorate=stated, direct_labels=False,
        alt=("Four nearly flat lines of implied detectable change against the "
             "assumed autocorrelation, with two dashed lines for BLS's stated "
             "figures above the published-series lines."),
        caption=("Every published-series line stays below the stated figure for "
                 "its window across the whole sweep. The unadjusted line "
                 "crosses it at about 0.22 and sits above it over the range "
                 "published CPS variance work actually puts the error "
                 f"correlation in, {ns.CPS_ERROR_CORRELATION[0]:.2f} to "
                 f"{ns.CPS_ERROR_CORRELATION[1]:.2f}."),
        path=str(IMG / f"a17-f3-rho.{EXT}"))[0]

    # --- t1: the two documents ----------------------------------------------
    rows = [
        ["report, 20 years ago", f"{ref['stated_then']:.2f} pp",
         "one-month detectable change, stated directly"],
        ["technical note, now", f"{ul.CPS_2026['ci90_change']:.2f} pp",
         f"90% interval on the one-month change, stated at "
         f"{ul.CPS_2026['ci90_change_stated_at_rate']:.1f}% unemployment"],
        ["the ratio of those two", f"{doc['naive_ratio']:.2f}x",
         "not a degradation: different statistics, different rates"],
        ["technical note, rescaled",
         f"{doc['from_technical_note']:.3f} pp",
         f"same figure at the {doc['rate_now']:.1f}% rate actually prevailing"],
        ["report, converted", f"{doc['from_report']:.3f} pp",
         "'now it would take two months' for a 0.18 pp change"],
        ["the two, compared", f"{doc['ratio']:.3f}x",
         "the documents agree to half a percent"],
        ["degradation, corrected", f"{doc['degradation']:.2f}x",
         "0.18 pp to 0.25 pp over twenty years"],
    ]
    out["t1"] = charts.table_image(
        rows, header=["figure", "value", "what it is"],
        title="Two numbers that are not the same number",
        subtitle=("Both are BLS's, from two documents. Their ratio is quoted as "
                  "the survey's degradation. Made comparable, they are the same "
                  "measurement and the degradation is smaller."),
        source="BLS April 2026 report to the Appropriations Committees; "
               "Employment Situation technical note.",
        bold_cols=(1,),
        alt="Table reconciling BLS's two published precision figures.",
        caption=("The first two rows are what the documents say. The last four "
                 "are the same two figures made comparable, at which point they "
                 "agree with each other and the degradation is smaller than the "
                 "ratio everybody quotes."),
        path=str(IMG / f"a17-t1-documents.{EXT}"))[0]

    # --- t2: the windows ----------------------------------------------------
    trows = []
    for w in res["windows"]:
        trows.append([f"{w['label']} — {w['note']}",
                      f"{w['u']:.2f}%",
                      f"{w['implied_sa']:.3f}",
                      f"{w['implied_bench']:.3f}",
                      f"{w['stated']:.3f}  ({w['stated_source']})",
                      f"{w['share_of_stated']:.0%}"])
    out["t2"] = charts.table_image(
        trows,
        header=["window", "mean rate", "published", "unadjusted", "BLS states",
                "published / stated"],
        title="Detectable one-month change: the series against the agency",
        subtitle=(f"All three windows use the same estimator at the same assumed "
                  f"autocorrelation ({DEFAULT_RHO:.2f}). The reference decade is "
                  f"compared with the report's own figure for that period; the "
                  f"recent windows with the technical note's interval rescaled "
                  f"to each window's unemployment rate."),
        source=src, bold_cols=(2, 5),
        alt="Table of implied and stated detectable change by window.",
        caption=("The first row is the agreement that validates the estimator; "
                 "the other two are the disagreement. The unadjusted column sits "
                 "above the stated figure throughout, which is where the missing "
                 "variation in the published column went."),
        path=str(IMG / f"a17-t2-windows.{EXT}"))[0]

    # --- hero ----------------------------------------------------------------
    def draw_survey(panel, m):
        rng = np.random.default_rng(2)
        t = np.linspace(0, 1, 70)
        panel.plot(t, 0.5 + 0.16 * rng.standard_normal(70), color=m.series[0],
                   lw=1.4)
        panel.set_ylim(0, 1); panel.set_xlim(0, 1)

    def draw_published(panel, m):
        rng = np.random.default_rng(2)
        t = np.linspace(0, 1, 70)
        raw = 0.5 + 0.16 * rng.standard_normal(70)
        smooth = pd.Series(raw).rolling(7, center=True, min_periods=1).mean()
        panel.plot(t, smooth.to_numpy(), color=m.series[1], lw=1.8)
        panel.set_ylim(0, 1); panel.set_xlim(0, 1)

    def draw_gap(panel, m):
        panel.plot([0.12, 0.88], [0.66, 0.66], color=m.ink, lw=2.0)
        for x in np.linspace(0.12, 0.88, 8):
            panel.plot([x, x], [0.66, 0.59], color=m.ink, lw=1.1)
        share = wins["2011-2019"]["share_of_stated"]
        cut = 0.12 + 0.76 * float(np.clip(share, 0, 1))
        panel.plot([0.12, cut], [0.32, 0.32], color=m.series[0], lw=5.0,
                   solid_capstyle="butt")
        panel.plot([cut, 0.88], [0.32, 0.32], color=m.series[3], lw=5.0,
                   solid_capstyle="butt")
        panel.set_ylim(0, 1); panel.set_xlim(0, 1)

    out["hero"] = charts.strip_card(
        headline="The unemployment rate is smoother than the survey",
        #: `strip_card` wraps a label at 24 characters and only two lines fit
        #: above the note, so these are kept to one line each and the note
        #: carries the explanation.
        panels=[(draw_survey, f"{wins['2011-2019']['stated']:.2f}",
                 "BLS's stated error"),
                (draw_published, f"{wins['2011-2019']['implied_sa']:.2f}",
                 "the published series"),
                (draw_gap, f"{wins['2011-2019']['share_of_stated']:.0%}",
                 "of it survives")],
        note=("Detectable one-month change in points, 2011-2019. The unadjusted "
              "series carries the full error; the published one has been through "
              "a seasonal filter, and a filter that averages a shifting "
              "timespan removes noise along with season."),
        footer="The Standard Error",
        alt=("A three-panel hand-drawn strip: a jagged line, the same line "
             "smoothed, and a ruler over a two-colour bar."),
        caption="",
        path=str(IMG / f"a17-hero.{EXT}"))[0]

    return out


# ---------------------------------------------------------------- the post

def _snippets(res: dict) -> dict:
    """Code blocks executed at build time. The output is captured, not typed."""
    s = Session()
    out = {}

    out["estimator"] = s.run("""
        import numpy as np

        def second_difference_scale(x, trim=0.10):
            "Noise sd of a monthly series, with any local trend differenced out."
            d = np.diff(np.asarray(x, float), n=2)
            # A winsorised root-mean-square: robust to a few real jumps, and
            # unlike a median it is not confined to the values a rounded series
            # can take. c is the sd of a standard normal winsorised at the
            # (1 - trim) quantile, so the estimator is consistent under normality.
            hi = np.quantile(np.abs(d), 1 - trim)
            w = np.clip(d, -hi, hi)
            from math import erf, exp, pi, sqrt
            z = hi / np.std(d)
            Phi = 0.5 * (1 + erf(z / sqrt(2)))
            phi = exp(-0.5 * z * z) / sqrt(2 * pi)
            c = (2 * Phi - 1) - 2 * z * phi + 2 * z * z * (1 - Phi)
            return float(np.sqrt(np.mean(w ** 2) / c / 6.0))

        rng = np.random.default_rng(0)
        n = 900
        trend = np.linspace(8.0, 4.0, n) + 0.6 * np.sin(np.arange(n) / 29.0)
        for sigma in (0.06, 0.12, 0.25):
            x = trend + rng.normal(0, sigma, n)
            print(f"planted {sigma:.2f}  recovered "
                  f"{second_difference_scale(x):.4f}")
    """, expect=["planted 0.06  recovered 0.0588",
                 "planted 0.12  recovered 0.1210",
                 "planted 0.25  recovered 0.2525"])

    out["lattice"] = s.run("""
        def mad_scale(x):
            d = np.diff(np.asarray(x, float), n=2)
            return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(6))

        # Four different true noise levels, each published to one decimal place
        # the way BLS publishes a rate.
        print("  true   winsorised      MAD")
        for sigma in (0.10, 0.13, 0.16, 0.19):
            x = np.round(trend + rng.normal(0, sigma, n), 1)
            print(f"  {sigma:.2f}      {second_difference_scale(x):.4f}   "
                  f"{mad_scale(x):.4f}")
        # A median of values on a 0.1 grid is a value on a 0.1 grid, so this
        # estimator can only ever return 1.4826 * k * 0.1 / sqrt(6).
        rungs = [1.4826 * k * 0.1 / np.sqrt(6) for k in (1, 2, 3)]
        print("  rungs available to it: " + ", ".join(f"{r:.4f}" for r in rungs))
    """, expect=["rungs available to it: 0.0605, 0.1211, 0.1816"])

    out["bridge"] = s.run("""
        Z90 = 1.6449

        def implied_detectable(sigma_second_diff, rho):
            "From a series' own second differences to a detectable one-month change."
            factor = 6 - 8*rho + 2*rho**2          # Var(second difference)/sigma^2
            sigma = sigma_second_diff * np.sqrt(6 / factor)
            return Z90 * sigma * np.sqrt(2 * (1 - rho))

        def rescale_for_rate(value, stated_at, actual):
            "A proportion's sampling error goes as sqrt(p(1-p))."
            a, b = actual/100, stated_at/100
            return value * np.sqrt(a*(1-a)) / np.sqrt(b*(1-b))

        # BLS's technical note states +/-0.3 pp *at a 6.0 percent rate*, and its
        # report says a 0.18 pp change now needs two months.
        print(f"note, rescaled to 4.2%   "
              f"{rescale_for_rate(0.30, 6.0, 4.2):.4f}")
        print(f"report, two months -> one  {0.18*np.sqrt(2):.4f}")
        print(f"naive ratio of the two published figures  "
              f"{0.30/0.18:.3f}x")
    """, expect=["note, rescaled to 4.2%   0.2534",
                 "report, two months -> one  0.2546",
                 "naive ratio of the two published figures  1.667x"])

    return out


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute(verbose=False)
    figs = figures(res)
    snip = _snippets(res)

    doc = res["documents"]
    ref = res["reference"]
    cov = res["coverage"]
    wins = {w["label"]: w for w in res["windows"]}
    then, mid, now = wins["1997-2006"], wins["2011-2019"], wins["2016-2026"]
    sc = res["scaling"]
    lat = res["lattice"]
    wr = res["wedge_range"]
    c = ul.CPS_2026

    # The spine, asserted rather than trusted.
    assert doc["ratio"] == pytest_approx(1.0, 0.02), doc
    assert 0.9 < ref["agreement"] < 1.1, ref
    assert mid["share_of_stated"] < 0.95, mid
    assert now["share_of_stated"] < 1.0, now
    assert mid["implied_bench"] > mid["stated"], mid
    assert len(lat["mad_values"]) <= 4, lat

    post = Post(
        title="The Unemployment Rate Is Smoother Than the Survey That Produces It",
        slug="smoother-than-the-survey",
        date=POST_DATE,
        subtitle=("BLS told Congress its household survey has lost precision. "
                  "The published series can be asked the same question directly, "
                  "and it agrees with the agency about twenty years ago and not "
                  "about now."),
        summary=("The Current Population Survey's own second differences imply a "
                 "detectable one-month change of 0.186 percentage point for the "
                 "decade BLS uses as its reference, against the 0.18 BLS states "
                 "— an agreement that earns the estimator the right to be "
                 "believed about the present, where the published series implies "
                 "0.20 against a stated 0.30. The unadjusted series does not "
                 "have the gap: it carries the full stated error. What sits "
                 "between them is seasonal adjustment, whose moving filters "
                 "remove noise along with season. Also: the widely repeated "
                 "1.67x degradation is built by dividing two BLS figures that "
                 "are different statistics measured at different unemployment "
                 "rates; corrected, the two documents agree with each other and "
                 "the degradation is 1.41x."),
        tags=["official statistics", "survey methodology", "seasonal adjustment",
              "measurement error", "labour market"],
        author="Jongha Jeon",
        code_url=se.SETTINGS.code_repo_url,
        min_words=1700, max_words=2400,
        data_sources=[
            f"US Bureau of Labor Statistics, Current Population Survey: "
            f"unemployment rate, seasonally adjusted ({SA_CODE}) and not "
            f"seasonally adjusted ({NSA_CODE}), monthly, {cov['start']} to "
            f"{cov['end']}, {cov['n']} observations. Downloaded from "
            f"data.bls.gov on {time.strftime('%d %B %Y')}. "
            f"{ul.LICENCE_NOTE}",
            f"October 2025 is absent from both series and is left as a gap "
            f"rather than bridged: {ul.MISSING_MONTHS['2025-10']}. No second "
            f"difference in this post spans it.",
            "Survey figures quoted from the BLS Report to the Appropriations "
            "Committees on Modernizing the Current Population Survey, April "
            "2026, and from the Employment Situation technical note. Both are "
            "stored in `standarderror.sources.us_labor.CPS_2026` with the "
            "sentence each came from.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "scipy", "pandas", "standarderror")),
            "vintage_sha256": ", ".join(
                f"{k}: {v['sha256'][:16]}" for k, v in res["vintage"].items()),
            "estimator": ("robust second-difference scale, winsorised at the 90th "
                          "percentile of |d2|, averaged over contiguous runs of "
                          "at least 40 months and weighted by run length"),
            "rho": (f"lag-1 autocorrelation of the sampling error assumed "
                    f"{DEFAULT_RHO:.2f} for headline figures and swept over "
                    f"{RHO_SWEEP} in figure 3"),
            "windows": ("fixed before looking at the numbers: the decade ending "
                        "at the report's reference year, and two recent windows "
                        "with and without the pandemic months"),
            "wedge": (f"{WEDGE_REPS} block-bootstrap replicates of the paired "
                      f"second differences, block 24 months"),
            "code blocks": ("executed at build time by "
                            "standarderror/render/snippet.py"),
            "modules": ("standarderror/sources/us_labor.py, "
                        "standarderror/ts/noisescale.py, "
                        "standarderror/ts/seasonal.py"),
            "tests": ("tests/test_us_labor.py, tests/test_seasonal.py"),
        },
    )

    # ------------------------------------------------------------------ 1
    post.add(
        "A number that is not in either document",
        f"""In April 2026 BLS told the Appropriations Committees that its
household survey is losing precision, and gave the numbers. The sample is
{c['households_eligible']:,} eligible households, "the same as it was in 1981".
Response rates have fallen "from the low 90 percent range to the upper 60
percent range over the last 15 years". A single respondent now stands for about
{c['people_per_respondent_now']:,} people instead of about
{c['people_per_respondent_then']:,}. And "a net change of fewer than
{c['responses_for_one_tenth_point']} survey responses could move the headline
unemployment rate by 0.1 percentage point".

The figure that travels, though, is a ratio: {c['detectable_change_then']:.2f}
percentage point then against {c['ci90_change']:.2f} now, so the survey has got
{doc['naive_ratio']:.2f} times blunter. I built that ratio myself, put it in a
module as two fields, and only found out it was wrong when I went back for the
exact sentences. Both halves are.

The {c['ci90_change']:.2f} comes from the Employment Situation technical note,
and the sentence begins *"At an unemployment rate of around
{c['ci90_change_stated_at_rate']:.1f} percent"*. A proportion's sampling error
goes as the square root of p(1−p), so that figure is not the one that applies
when unemployment is {doc['rate_now']:.1f} percent. And the report never states a
current one-month threshold at all. What it says is that
{c['detectable_change_then']:.2f} point "was statistically significant" twenty
years ago, "while now it would take {c['months_needed_now']} months of data
before one could determine that this same change is statistically significant" —
which is a claim about a threshold of roughly
{c['detectable_change_then']:.2f}·√{c['months_needed_now']} instead.""")

    post.add(
        "",
        f"""{snip['bridge'].markdown()}

{doc['from_technical_note']:.3f} and {doc['from_report']:.3f}. The two documents
are not in tension at all; they are the same measurement, and the degradation
they describe is {doc['degradation']:.2f} times over twenty years rather than
{doc['naive_ratio']:.2f}. Table 1 lays the two figures out with what each one is.

That is worth a paragraph of its own because of how the error is made. Neither
number is wrong, neither document is unclear, and nothing was hidden. Two
figures from two places got stored next to each other without the clauses that
made them conditional, and a ratio of two numbers is always available whether or
not it means anything.""",
        figures=[figs["t1"]], level=3)

    # ------------------------------------------------------------------ 2
    post.add(
        "Asking the series instead",
        """There is a way to check an agency's stated precision that needs no
microdata, no variance estimation and no cooperation: the published series has
the noise in it. Difference it twice — which annihilates any local level and
trend — take a robust scale of what is left, and divide out the factor of six
that second differencing introduces.

The robustness matters and the usual robust choice fails here. A monthly rate is
published to one decimal place, so its second differences live on a 0.1 grid, and
a median of numbers on a grid is a number on the grid.""")

    post.add(
        "",
        f"""{snip['estimator'].markdown()}

{snip['lattice'].markdown()}

Across eighty years of the real series, the median-based estimator returns
{len(lat['mad_values'])} distinct values — {', '.join(f"{v:.4f}" for v in lat['mad_values'])}
— because those are the only rungs the lattice allows. The winsorised
root-mean-square costs about two percent of efficiency to rounding and has no
such problem, so that is what everything below uses. This is the whole of the
methodological content and it took the longest.""", level=3)

    # ------------------------------------------------------------------ 3
    post.add(
        "It agrees about twenty years ago",
        f"""BLS's report is dated April 2026 and says "20 years ago", so its
reference year is {REFERENCE_YEAR} and the comparable window is the decade ending
there. That window was fixed before any of these numbers were computed, because
a window chosen afterwards is not a test.

Over {then['label']}, with mean unemployment {then['u']:.2f} percent, the
published series' own second differences imply a detectable one-month change of
**{ref['implied']:.3f} percentage point**. BLS states
{ref['stated_then']:.2f}. They are {abs(100 * (ref['agreement'] - 1)):.0f}
percent apart.

That agreement is the load-bearing part of this post. An estimator built from a
published series, with an assumed autocorrelation and a robust scale and a
factor of six, reproduces a figure that came out of the survey's design-based
variance machinery. It is why the disagreement that follows can be read as a
finding about the series rather than as the estimator being wrong.""",
        figures=[figs["f1"]])

    # ------------------------------------------------------------------ 4
    post.add(
        "And not about now",
        f"""Move to the recent windows and the two part company. Over
{mid['label']} — recent, and without the pandemic months —
the published series implies {mid['implied_sa']:.3f} against a stated
{mid['stated']:.3f} for that window's unemployment rate:
**{mid['share_of_stated']:.0%}** of it. Include the pandemic and the series-implied
figure rises to {now['implied_sa']:.3f} against {now['stated']:.3f}, which is
{now['share_of_stated']:.0%} — closer, but the pandemic months are real
labour-market movement rather than sampling noise, so that window is the
generous reading rather than the fair one.

Put the two eras together and the published series shows a degradation of
{mid['implied_sa'] / ref['implied']:.2f} to {now['implied_sa'] / ref['implied']:.2f}
times where the documents describe {doc['degradation']:.2f}. The direction is
right. The size is {(mid['implied_sa'] / ref['implied'] - 1) / (doc['degradation'] - 1):.0%}
to {(now['implied_sa'] / ref['implied'] - 1) / (doc['degradation'] - 1):.0%} of it.

Before believing any of that, there is one number in the estimator nobody knows:
the lag-1 autocorrelation of the sampling error. The household overlap is
famously {c['monthly_overlap']:.0%}, but that is not the same quantity — households
change labour-force state and rotation groups turn over, and published CPS
variance work puts the error correlation nearer
{ns.CPS_ERROR_CORRELATION[0]:.1f} to {ns.CPS_ERROR_CORRELATION[1]:.1f}. It turns
out not to matter, for a reason worth stating: it enters twice and in opposite
directions. A higher autocorrelation means the second difference understates the
level noise, pushing the answer up; it also means a month-on-month difference of
two correlated estimates is less noisy, pushing the answer down.""",
        figures=[figs["f3"]])

    # ------------------------------------------------------------------ 5
    post.add(
        "The raw series does not have the gap",
        f"""The same estimator on the *unadjusted* rate, with a fixed
month-of-year mean removed so that it is measuring noise rather than season,
gives {mid['implied_bench']:.3f} for {mid['label']} — **above** the stated
{mid['stated']:.3f} rather than below it, which is what should happen, since the
unadjusted series also contains real seasonal drift and real month-to-month
movement on top of the sampling error.

So the survey's raw output carries the error BLS attributes to it, and the number
the country reads does not. What sits between the two is seasonal adjustment, and
BLS documents exactly what it does. The national CPS series have been adjusted
with X-12-ARIMA since 2003 and X-13ARIMA-SEATS since 2015, by "procedures
based on 'filters' that successively average a shifting timespan of data", the
final one spanning six to ten years of data. Concurrent adjustment — where the
current month's own value helps estimate the current month's seasonal factor —
began with the December 2003 estimates.

That is a filter, and a filter that estimates a month's seasonal factor partly
from that month cannot help attributing part of a one-off shock to season and
removing it. Nobody is doing anything wrong; a series read as a trend is
*supposed* to be smoothed. The consequence is only that the published rate is
quieter than the estimate behind it, and that checking the published rate against
a design-based standard error compares two different things.""")

    # ------------------------------------------------------------------ 6
    post.add(
        "How much, and why I cannot tell you",
        f"""The obvious next number is the size of that wedge, and it is the one
thing here I could not pin down. Comparing the published series against the
fixed-seasonal benchmark says the published series is missing anywhere from
{wr['lo']:+.0%} to {wr['hi']:+.0%} of the benchmark's month-to-month noise,
and the spread is not sampling error — it moves with the length of the window the
benchmark is fitted on. Ten-year windows say a third; fourteen-year windows say a
tenth. A quantity that depends on the analyst's window is not a measurement of
the filter.

I also tried to date a step at the 2003–04 methodology change, which would have
been a satisfying result, and it failed: at equal fourteen-year windows the
pre-2003 and post-2004 wedges are not distinguishable. The decade-by-decade
picture that looked like a clean doubling in the 2000s does not survive holding
the window length fixed. Figure 2 is that failure, published rather than
dropped, because a reader who repeats this with a different window will get a
different number and should know that in advance.

What survives is the direction, in every window but one, and the fact that the
published series is on the low side of the stated error while the unadjusted
series is on the high side.""",
        figures=[figs["f2"]])

    # ------------------------------------------------------------------ 7
    post.add(
        "A second symptom, and the summary",
        f"""One more thing the series does not do. The sampling error of a
proportion scales as √(p(1−p)), so as unemployment moves between
{sc['u_lo']:.1f} and {sc['u_hi']:.1f} percent the noise in the published rate
should move by a factor of {sc['pq_ratio']:.2f}. Across
{sc['n']} rolling seven-year windows the log-log slope of the estimated noise
scale on √(p(1−p)) is {sc['slope']:+.2f}, against a theoretical
+1.00, with a correlation of {sc['r']:+.2f}. Recessions also have larger *real*
month-to-month movement, so two separate forces predict a positive slope and
neither shows up.

That is weaker evidence than the rest — the high-unemployment windows are also
particular periods, so era and level are confounded — but it points the same way:
the published series' high-frequency variation is remarkably stable
({sc['sigma_lo']:.3f} to {sc['sigma_hi']:.3f} across all of them) and is not
behaving like the sampling error of a survey whose conditions changed a great
deal.""",
        figures=[figs["t2"]])

    # ------------------------------------------------------------------ 8
    post.add(
        "What this does not say",
        """It does not say the unemployment rate is wrong, that BLS is
overstating its own errors, or that seasonal adjustment should stop. Every step
here is about one statistic — the month-to-month variability of a published
series — and about whether it can be compared with a design-based standard error.
The answer is that it cannot, without accounting for the filter in between.

It does not measure the sampling error. The second difference of a published
series contains noise *plus* whatever genuine short-run movement the labour
market had, so every figure in this post is an upper bound on the noise, which
makes the direction of the finding safe and its magnitude conservative.

And it is not a forecast, an investment view, or a judgement about any
institution. The concrete thing a reader can take from it is narrower and more
useful than a verdict: if you are checking whether a monthly change in a
seasonally adjusted official series is meaningful, the standard error published
for that series describes the estimate before adjustment, and the series in front
of you has been through a filter that removed some of the variation the standard
error is about. The two figures BLS published are consistent with each other. It
is the arithmetic done to them afterwards — mine included — that was not.""")

    return post


def pytest_approx(value, rel):
    """Tiny local helper so `build` can assert on a ratio without importing pytest."""
    class _Approx:
        def __eq__(self, other):
            return abs(float(other) - value) <= rel * abs(value)
    return _Approx()


def main() -> Post:
    post = build()
    problems = post.audit()
    print(f"words: {post.word_count()}")
    print("audit:", "clean" if not problems else "")
    for p in problems:
        print("  -", p)
    return post
