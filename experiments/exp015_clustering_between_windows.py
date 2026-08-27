"""exp015 — volatility clustering, measured inside the window and outside it.

The first post in this series built on real market data rather than published
scalars, and it exists to test a prediction the previous one made.

Where this comes from
---------------------
exp013 read a survey of diffusion models in finance and found, on simulated data,
that a generator emitting fixed-length windows can only reproduce dependence that
fits *inside* one. At the volatility persistence an equity index usually shows, the
lag-1 autocorrelation of absolute returns inside a 32-step window was about +0.002 —
nothing. That was a claim about a GARCH simulation, and it implied a claim about real
markets that a simulation cannot settle.

This settles it. Two of the longest daily index histories available — the NASDAQ
Composite from 1971 and the Nikkei 225 from 1949, about 33,000 trading days between
them — say the prediction was right and understated it.

* Over the whole series, the lag-1 autocorrelation of absolute returns is **+0.32**
  (NASDAQ) and **+0.28** (Nikkei). This is volatility clustering as it is normally
  quoted, and it is unmistakable.
* Measured **inside a 64-day window**, the same series give **+0.04** and **+0.06**,
  against a shuffled baseline of about -0.03 and -0.01. So the part a 64-day window
  can express is roughly a *fifth* of the effect being claimed.

Volatility clustering, at the horizon generative papers train on, is mostly a
*between-window* phenomenon. The window length is not an implementation detail; it
decides what the model can be asked to do at all.

Three more things the real data says, all in the body
----------------------------------------------------
1. **The target is not a constant.** NASDAQ excess kurtosis by decade runs from 2.8
   to 32.4; the Nikkei's from 2.2 to 56.7. "Our model matches the kurtosis of the
   index" is a statement about a sample period, not about a market.
2. **And at the sample sizes people use it is binary, not uncertain.** Excess
   kurtosis from 2,520 contiguous days — ten years — has a two-humped distribution:
   samples containing 19 October 1987 average 29, samples without it average 6, and for
   the NASDAQ one date classifies the estimate *perfectly*. Deleting that week from the
   full 14,000-day history moves the number only from 9.4 to 8.7, so the problem is
   sample length rather than the crash. exp013 measured a wide spread on simulated
   data; the real thing turns out to have structure, not just width.
3. **The block bootstrap does not preserve clustering here, it manufactures it.**
   On the GARCH path in exp013, blocks of 16 matched the truth. On real data they
   *overshoot* the within-window value by three to four times, because splicing
   blocks from different volatility regimes builds a step function in |r|, and a step
   function has strong lag-1 autocorrelation. A control that was well-behaved on
   simulated data is not well-behaved on this.

Licence: FRED index series are not redistributable, so this publishes statistics and
never values, and every figure is a statistic against a parameter rather than a
series against time. Files are git-ignored; see `data/fred/README.md`.

Run: `standarderror run exp015_clustering_between_windows --publish`
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.generative import stylised
from standarderror.render import Post
from standarderror.sources import prices
from standarderror.sources.fred import MANDATORY_DISCLAIMER
from standarderror.viz import charts, theme

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed
DATA = Path("data/fred")

# --- what exp013 predicted, so the post can be checked against it ---------------
EXP013_SLUG = "fat-tails-are-free"
EXP013_PREDICTION = 0.002        # within-32-step clustering at persistence 0.98

# --- the series ----------------------------------------------------------------
SERIES = {
    "NASDAQ Composite": "NASDAQCOM",
    "Nikkei 225": "NIKKEI225",
}
VOL_SERIES = ("VIXCLS", "VIX")
WINDOWS = (8, 16, 32, 64, 128, 256, 512)
HEADLINE_WINDOW = 64
STRIDE = 8
N_EVAL = 600
N_BOOT = 300
BLOCK = 16
BLOCK_SWEEP = (2, 4, 8, 16, 32, 64)
#: Sample sizes a paper might actually use: two years, ten years, everything.
SAMPLE_SIZES = (500, 2520, 14000)
N_DRAWS = 400
#: The single day the ten-year kurtosis distribution turns out to be a function of.
#: Named here rather than found by eye, so the split below is a stated hypothesis
#: being tested and not a pattern fitted after the fact.
EVENT_DATE, EVENT_NAME = "1987-10-19", "19 October 1987"
EVENT_WINDOW = ("1987-10-15", "1987-10-26")
HERO_POINTS = 130          # see the note in the hero drawing below
CACHE = se.SETTINGS.build_dir / "cache" / "exp015.json"


def _vintage() -> dict:
    """sha256 of every input file, so a figure can be traced to exact bytes."""
    out = {}
    for mnemonic in list(SERIES.values()) + [VOL_SERIES[0]]:
        p = DATA / f"{mnemonic}.csv"
        if p.exists():
            out[mnemonic] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
    return out


def _config_key() -> str:
    blob = json.dumps({"v": 1, "series": SERIES, "windows": WINDOWS,
                       "stride": STRIDE, "n_eval": N_EVAL, "n_boot": N_BOOT,
                       "block": BLOCK, "block_sweep": BLOCK_SWEEP,
                       "sizes": SAMPLE_SIZES, "draws": N_DRAWS, "seed": SEED,
                       "vintage": _vintage()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def returns(mnemonic: str) -> "np.ndarray":
    """Log returns in percent from the hand-downloaded FRED file."""
    path = DATA / f"{mnemonic}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. This container cannot fetch it; download "
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={mnemonic} and put "
            f"it there. See data/fred/README.md.")
    return prices.to_log_returns(prices.load_prices(path)).to_numpy()


def dated_returns(mnemonic: str):
    return prices.to_log_returns(prices.load_prices(DATA / f"{mnemonic}.csv"))


def acf1(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def compute(*, force: bool = False, verbose: bool = True) -> dict:
    """Every grid in the post. Cached under a hash of the config and the file bytes."""
    key = _config_key()
    if not force and CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            return cached

    t0 = time.time()

    def say(*a):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)

    out = {"key": key, "vintage": _vintage(), "series": {}}
    rng = np.random.default_rng(SEED)

    for label, mnemonic in SERIES.items():
        s = dated_returns(mnemonic)
        r = s.to_numpy()
        shuffled = rng.permutation(r)
        say(f"{label}: n={len(r)}")

        # 1. The headline: clustering inside a window versus over the whole series.
        inside = stylised.within_window_clustering(r, WINDOWS, stride=STRIDE)
        baseline = stylised.within_window_clustering(shuffled, WINDOWS, stride=STRIDE)
        whole = acf1(np.abs(r))

        # 2. Is the target a constant? Decade by decade.
        decades = {}
        for dec, g in s.groupby(s.index.year // 10 * 10):
            if len(g) < 500:
                continue
            v = g.to_numpy()
            c = v - v.mean()
            decades[int(dec)] = {
                "n": int(len(v)), "sd_pct": float(v.std(ddof=1)),
                "excess_kurtosis": float((c ** 4).mean() / c.var() ** 2 - 3.0),
                "acf1_abs": acf1(np.abs(v)),
            }

        # 3. How well is it known at the sample sizes people use? Contiguous draws,
        #    because a real sample is a stretch of history and not a random subset.
        spread = {}
        for n in SAMPLE_SIZES:
            if n > len(r):
                continue
            g = np.random.default_rng(SEED + n)
            vals, has_event = [], []
            event = pd.Timestamp(EVENT_DATE)
            for _ in range(N_DRAWS if n < len(r) else 1):
                j = int(g.integers(0, len(r) - n + 1))
                blk = r[j:j + n]
                c = blk - blk.mean()
                vals.append(float((c ** 4).mean() / c.var() ** 2 - 3.0))
                has_event.append(bool(s.index[j] <= event <= s.index[j + n - 1]))
            a = np.asarray(vals)
            flag = np.asarray(has_event)
            spread[n] = {"mean": float(a.mean()), "sd": float(a.std()),
                         "p5": float(np.percentile(a, 5)),
                         "p95": float(np.percentile(a, 95)),
                         "draws": int(a.size),
                         # Kept so the figure can show the distribution rather than
                         # three summary numbers standing in for it.
                         "values": [float(x) for x in a],
                         "with_event": [float(x) for x in a[flag]],
                         "without_event": [float(x) for x in a[~flag]],
                         "n_with_event": int(flag.sum())}

        # 4. The battery with both controls, on the headline window.
        W = np.lib.stride_tricks.sliding_window_view(r, HEADLINE_WINDOW)[::STRIDE]
        pick = rng.choice(len(W), min(N_EVAL, len(W)), replace=False)
        rows = {
            "the series": stylised.stylised_facts(W[pick], n_boot=N_BOOT, seed=SEED),
            "shuffled": stylised.stylised_facts(
                stylised.iid_bootstrap(r, N_EVAL, HEADLINE_WINDOW, seed=SEED + 1),
                n_boot=N_BOOT, seed=SEED),
            f"blocks of {BLOCK}": stylised.stylised_facts(
                stylised.block_bootstrap(r, N_EVAL, HEADLINE_WINDOW, block=BLOCK,
                                         seed=SEED + 2),
                n_boot=N_BOOT, seed=SEED),
        }
        block_sweep = {
            b: stylised.stylised_facts(stylised.block_bootstrap(
                r, N_EVAL, HEADLINE_WINDOW, block=b, seed=SEED + 2))["acf1_abs"]["value"]
            for b in BLOCK_SWEEP}

        keep = np.asarray((s.index < EVENT_WINDOW[0]) | (s.index > EVENT_WINDOW[1]))
        trimmed = r[keep]
        ct = trimmed - trimmed.mean()
        out["series"][label] = {
            "mnemonic": mnemonic,
            "event_week_days": int((~keep).sum()),
            "kurtosis_without_event_week": float(
                (ct ** 4).mean() / ct.var() ** 2 - 3.0),
            "stats": prices.publishable_statistics(s, quantiles=(0.5, 0.9, 0.99)),
            "whole_series_acf1_abs": whole,
            "inside": {str(k): v for k, v in inside.items()},
            "baseline": {str(k): v for k, v in baseline.items()},
            "decades": decades,
            "spread": {str(k): v for k, v in spread.items()},
            "rows": rows,
            "block_sweep": {str(k): v for k, v in block_sweep.items()},
        }
        say(f"  whole {whole:+.3f} | inside {HEADLINE_WINDOW} "
            f"{inside[HEADLINE_WINDOW]:+.3f} | baseline "
            f"{baseline[HEADLINE_WINDOW]:+.3f}")
        say(f"  decade kurtosis range "
            f"{min(d['excess_kurtosis'] for d in decades.values()):.1f} to "
            f"{max(d['excess_kurtosis'] for d in decades.values()):.1f}")

    # 5. The VIX, as an outside check that the clustering is real and not an artefact
    #    of how returns are measured: implied volatility is a direct observation of
    #    the market's own view of the next month's variance.
    try:
        vix = dated_returns(VOL_SERIES[0])
        lv = prices.load_prices(DATA / f"{VOL_SERIES[0]}.csv")["close"].dropna()
        out["vix"] = {
            "n": int(len(lv)),
            "first_date": str(lv.index[0].date()), "last_date": str(lv.index[-1].date()),
            "level_acf1": acf1(lv.to_numpy()),
            "change_acf1_abs": acf1(np.abs(vix.to_numpy())),
            "stats": prices.publishable_statistics(vix, quantiles=(0.5, 0.9, 0.99)),
        }
        say(f"VIX: level ACF1 {out['vix']['level_acf1']:+.3f}")
    except FileNotFoundError:
        out["vix"] = None
        say("VIX file absent; skipping the outside check")

    out["elapsed_seconds"] = time.time() - t0
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=1))
    say("cached to", CACHE)
    return out


SRC = (f"NASDAQ Composite and Nikkei 225 daily closes via FRED. Statistics only; "
       f"no values republished. {MANDATORY_DISCLAIMER}")
BATTERY_HEADER = ["series and generator", "excess kurtosis", "ACF1 of |r|",
                  "ACF1 of r", "leverage"]


def _fmt(entry: dict, digits: int = 3) -> str:
    if not np.isfinite(entry["value"]):
        return "undefined"
    if np.isfinite(entry["se"]):
        return f"{entry['value']:.{digits}f} ± {entry['se']:.{digits}f}"
    return f"{entry['value']:.{digits}f}"


def battery_rows(res: dict) -> list[list[str]]:
    rows = []
    for label, blob in res["series"].items():
        short = label.split()[0]
        for gen, facts in blob["rows"].items():
            rows.append([f"{short} — {gen}", _fmt(facts["excess_kurtosis"], 1),
                         _fmt(facts["acf1_abs"]), _fmt(facts["acf1_returns"]),
                         _fmt(facts["leverage"])])
    return rows


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown table with pipes inside cells escaped. See exp013's note."""
    def cell(x):
        return str(x).replace("|", r"\|")
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def figures(res: dict) -> dict:
    import pandas as pd

    figs = {}
    labels = list(res["series"])
    nas, nik = (res["series"][k] for k in labels)

    # F1 — the headline. Clustering measured inside a window, against the window's
    # length, with the whole-series value as a horizontal reference. The gap between
    # the curves and the lines is what a windowed generator cannot see.
    frame = pd.DataFrame(
        {**{k: [res["series"][k]["inside"][str(L)] for L in WINDOWS] for k in labels},
         "either series, shuffled": [
             np.mean([res["series"][k]["baseline"][str(L)] for k in labels])
             for L in WINDOWS]},
        index=list(WINDOWS))

    def mark_whole(_fig, ax):
        m = theme.LIGHT
        for k, col in zip(labels, theme.series_colors(len(labels) + 1, "light")):
            v = res["series"][k]["whole_series_acf1_abs"]
            ax.axhline(v, color=col, lw=1.1, ls=(0, (3, 3)), alpha=0.9)
            ax.annotate(f"{k}, whole series: {v:+.2f}", (WINDOWS[0], v),
                        xytext=(2, 4), textcoords="offset points", fontsize=8.5,
                        color=col)
        ax.axvline(HEADLINE_WINDOW, color=m.muted, lw=1.0, ls=(0, (2, 3)))
        ax.annotate(f"a {HEADLINE_WINDOW}-day window", (HEADLINE_WINDOW, 0.02),
                    xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=m.muted,
                    va="bottom")
        from matplotlib.ticker import FixedLocator, ScalarFormatter
        ax.xaxis.set_major_locator(FixedLocator(list(WINDOWS)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(FixedLocator([]))

    gap_n = (nas["inside"][str(HEADLINE_WINDOW)]
             - nas["baseline"][str(HEADLINE_WINDOW)])
    share = gap_n / nas["whole_series_acf1_abs"]
    fig_meta, _ = charts.lines(
        frame, mode="light", logx=True, decorate=mark_whole, direct_labels=False,
        title="Most volatility clustering does not fit inside the window",
        subtitle=("Lag-1 autocorrelation of absolute daily returns measured inside a "
                  "window of the length on the x-axis, against the same statistic "
                  "over the whole history. Two indices and a shuffled control."),
        ylabel="ACF1 of |r| measured inside the window",
        xlabel="window length (trading days, log scale)", source=SRC,
        alt=("Two rising curves and a flat control against window length on a log "
             "x-axis, well below two dashed horizontal lines at +0.32 and +0.28 "
             "marking the whole-series values. At 64 days the curves are near +0.04 "
             "and +0.06."),
        caption=(f"Fig 1. Volatility clustering as normally quoted is the dashed "
                 f"line: **{nas['whole_series_acf1_abs']:+.2f}** for the NASDAQ over "
                 f"{nas['stats']['n']:,} days, "
                 f"**{nik['whole_series_acf1_abs']:+.2f}** for the Nikkei over "
                 f"{nik['stats']['n']:,}. Inside a {HEADLINE_WINDOW}-day window — the "
                 f"horizon generative papers train on — the same series give "
                 f"{nas['inside'][str(HEADLINE_WINDOW)]:+.3f} and "
                 f"{nik['inside'][str(HEADLINE_WINDOW)]:+.3f}, against a shuffled "
                 f"baseline of about "
                 f"{np.mean([res['series'][k]['baseline'][str(HEADLINE_WINDOW)] for k in labels]):+.3f}. "
                 f"Net of that baseline the NASDAQ's within-window clustering is "
                 f"{gap_n:+.3f}, which is {100 * share:.0f}% of the effect the dashed "
                 f"line reports. The curves only reach the dashed lines somewhere past "
                 f"the right edge of this chart."),
        path=str(IMG / f"a7-f1-window.{EXT}"))
    figs["window"] = fig_meta

    # F2 — the target is not a constant. Decade by decade, both indices.
    decades = sorted({int(d) for k in labels for d in res["series"][k]["decades"]})
    dec_frame = pd.DataFrame(
        {k: [res["series"][k]["decades"].get(str(d), {}).get("excess_kurtosis",
                                                             np.nan)
             for d in decades] for k in labels},
        index=[f"{d}s" for d in decades])
    ranges = {k: (min(v["excess_kurtosis"] for v in res["series"][k]["decades"].values()),
                  max(v["excess_kurtosis"] for v in res["series"][k]["decades"].values()))
              for k in labels}
    fig_meta, _ = charts.lines(
        dec_frame, mode="light", direct_labels=False,
        title="The number generators are matched against is not a constant",
        subtitle=("Excess kurtosis of daily log returns, by decade. Each point is "
                  "roughly 2,500 trading days — a larger sample than most generative "
                  "papers use."),
        ylabel="excess kurtosis of the decade", xlabel="decade", source=SRC,
        alt=("Two jagged lines against decade. Both are low and flat near 5 for most "
             "decades with a single large spike in the 1980s, reaching about 32 for "
             "the NASDAQ and 57 for the Nikkei."),
        caption=(f"Fig 2. Over the whole history the NASDAQ's excess kurtosis is "
                 f"{nas['stats']['excess_kurtosis']:.1f} and the Nikkei's "
                 f"{nik['stats']['excess_kurtosis']:.1f}. By decade the NASDAQ runs "
                 f"from {ranges[labels[0]][0]:.1f} to {ranges[labels[0]][1]:.1f} and "
                 f"the Nikkei from {ranges[labels[1]][0]:.1f} to "
                 f"{ranges[labels[1]][1]:.1f} — factors of "
                 f"{ranges[labels[0]][1] / ranges[labels[0]][0]:.0f} and "
                 f"{ranges[labels[1]][1] / ranges[labels[1]][0]:.0f}. Both spikes are "
                 f"the 1980s, and both are one week of it. A model matched to "
                 f"'the kurtosis of the index' has been matched to a choice of "
                 f"sample period."),
        path=str(IMG / f"a7-f2-decades.{EXT}"))
    figs["decades"] = fig_meta

    # F3 — and the reason the target is unstable is one day. The distribution of
    # ten-year kurtosis estimates is not wide, it is two-humped: samples containing
    # 19 October 1987 and samples not containing it.
    ten = str(2520)
    sp = nas["spread"][ten]
    draws = np.asarray(sp["values"], float)
    with_e = np.asarray(sp["with_event"], float)
    without_e = np.asarray(sp["without_event"], float)
    full = nas["stats"]["excess_kurtosis"]
    # How cleanly does that one day classify the estimate? Stated, not eyeballed.
    sep = {}
    for k in labels:
        s2 = res["series"][k]["spread"][ten]
        w, wo = np.asarray(s2["with_event"]), np.asarray(s2["without_event"])
        sep[k] = {"clean": bool(w.size and wo.size and w.min() > wo.max()),
                  "with_mean": float(w.mean()), "without_mean": float(wo.mean()),
                  "with_lo": float(w.min()), "without_hi": float(wo.max()),
                  "n_with": int(w.size), "n": int(w.size + wo.size)}

    fig_meta, _ = charts.histogram(
        draws, bins=34, series_label=f"ten-year samples ({sp['draws']} of them)",
        # Two marks, not three: a "whole history" mark at 9.4 collided with the
        # left hump's mean at 6.0, and that number is in the caption anyway.
        mark={"without the crash": without_e.mean(),
              "with the crash": with_e.mean()},
        mode="light",
        title="The ten-year kurtosis is a function of whether one day is in your sample",
        subtitle=(f"Excess kurtosis of the NASDAQ on {sp['draws']} contiguous "
                  f"ten-year stretches at random start dates. A real sample is a "
                  f"stretch of history, not a random subset."),
        xlabel="excess kurtosis of one ten-year sample", source=SRC,
        alt=("A clearly two-humped histogram of excess kurtosis estimates: a large "
             "cluster between about 3 and 13, a second cluster between about 22 and "
             "35, and almost nothing between them. Vertical marks sit at the two "
             "cluster means and at the whole-history value of 9.4."),
        caption=(f"Fig 3. This distribution is not wide, it is **two-humped**, and the "
                 f"gap between the humps is a single trading day. Of the "
                 f"{sep[labels[0]]['n']} NASDAQ samples, the "
                 f"{sep[labels[0]]['n_with']} that contain {EVENT_NAME} average "
                 f"{sep[labels[0]]['with_mean']:.1f} and the rest average "
                 f"{sep[labels[0]]['without_mean']:.1f} — "
                 f"{'a perfect separation: every sample above ' + format(sep[labels[0]]['without_hi'], '.1f') + ' contains that day and none below it does' if sep[labels[0]]['clean'] else 'nearly separated'}. "
                 f"The Nikkei splits the same way, {sep[labels[1]]['without_mean']:.1f} "
                 f"against {sep[labels[1]]['with_mean']:.1f}, because October 1987 was "
                 f"global — though its humps overlap slightly rather than separating "
                 f"cleanly. And the mechanism is sample length, not the crash: "
                 f"deleting that whole week from the full "
                 f"{nas['stats']['n']:,}-day history moves the kurtosis only from "
                 f"{full:.1f} to {nas['kurtosis_without_event_week']:.1f}."),
        path=str(IMG / f"a7-f3-uncertainty.{EXT}"))
    figs["uncertainty"] = fig_meta

    # T1 — the battery, both series, both controls. The block-bootstrap row is the
    # one that inverts what the simulated version of this experiment concluded.
    fig_meta, _ = charts.table_image(
        battery_rows(res), header=BATTERY_HEADER,
        title="The same battery, on real indices, with both controls",
        subtitle=(f"Measured on {N_EVAL} windows of {HEADLINE_WINDOW} days. Errors "
                  f"are one standard error over windows."),
        source=SRC, mode="light", bold_cols=(2,), align="lrrrr",
        alt=("A table of six rows — two indices each measured as themselves, as an "
             "i.i.d. shuffle, and as a moving-block bootstrap — against four stylised "
             "facts."),
        caption=(f"Table 1. Two rows to read against each other. The shuffle "
                 f"reproduces the kurtosis and destroys the clustering, exactly as on "
                 f"simulated data. But blocks of {BLOCK} *overshoot* the series' own "
                 f"within-window clustering — "
                 f"{nas['rows'][f'blocks of {BLOCK}']['acf1_abs']['value']:+.3f} "
                 f"against {nas['rows']['the series']['acf1_abs']['value']:+.3f} for "
                 f"the NASDAQ — because splicing blocks from different volatility "
                 f"regimes builds a step function in |r|, and a step function is "
                 f"strongly autocorrelated at lag 1. On the GARCH path this battery "
                 f"was built against, the same control matched the truth. Also note "
                 f"the leverage column: real indices have a leverage effect, the "
                 f"symmetric simulation did not, and the shuffle destroys it. These "
                 f"rows score {N_EVAL} sampled windows so that every generator is "
                 f"measured on the same count, which is why the series' own clustering "
                 f"reads {nas['rows']['the series']['acf1_abs']['value']:+.3f} here "
                 f"against Fig 1's full-sample "
                 f"{nas['inside'][str(HEADLINE_WINDOW)]:+.3f}."),
        path=str(IMG / f"a7-t1-battery.{EXT}"))
    figs["battery"] = fig_meta

    # HERO — the same path, seen whole and seen through a window.
    def path_panel(panel, m, windowed: bool):
        from matplotlib.patches import Rectangle
        panel.set_xlim(0, 10)
        panel.set_ylim(-1.15, 1.15)
        g = np.random.default_rng(5)
        # Point count matters more than it looks: the hand-drawn filter resamples
        # every stroke into many small segments, so a 340-point noisy path becomes a
        # 113 KB single `d=` attribute and a 950 KB SVG. The wobble is invisible on
        # top of noise anyway, so this buys nothing visually and costs a lot.
        x = np.linspace(0.3, 9.7, HERO_POINTS)
        vol = (0.10 + 0.62 * np.exp(-((x - 2.1) / 0.55) ** 2)
               + 0.70 * np.exp(-((x - 6.9) / 0.75) ** 2)
               + 0.22 * np.exp(-((x - 4.4) / 0.4) ** 2))
        y = vol * g.standard_normal(x.size)
        panel.axhline(0, color=m.grid, lw=1.0)
        if windowed:
            lo, hi = 4.9, 6.0                       # a calm stretch between bursts
            keep = (x >= lo) & (x <= hi)
            panel.plot(x[keep], y[keep], color=m.series[1], lw=1.7)
            panel.add_patch(Rectangle((lo, -1.05), hi - lo, 2.1, fc="none",
                                      ec=m.ink, lw=2.2))
        else:
            panel.plot(x, y, color=m.series[0], lw=1.3)

    fig_meta, _ = charts.strip_card(
        headline="How much clustering can a 64-day window see?",
        panels=[(lambda p, m: path_panel(p, m, False),
                 f"{nas['whole_series_acf1_abs']:+.2f}", "the whole history"),
                (lambda p, m: path_panel(p, m, True),
                 f"{nas['inside'][str(HEADLINE_WINDOW)]:+.3f}",
                 "inside one 64-day window")],
        note=(f"Lag-1 autocorrelation of absolute daily NASDAQ returns, "
              f"{nas['stats']['first_date'][:4]}-{nas['stats']['last_date'][:4]}. "
              f"Volatility clustering is real and large — and mostly a fact about "
              f"which window you are in, not what happens inside one."),
        footer="The Standard Error", mode="light",
        alt=("A two-panel hand-drawn strip. The first frame shows a long return path "
             "with three visible bursts of volatility, marked "
             f"{nas['whole_series_acf1_abs']:+.2f}. The second shows the same path "
             "with a box cropping one calm stretch between bursts, marked "
             f"{nas['inside'][str(HEADLINE_WINDOW)]:+.3f}."),
        caption="",
        path=str(IMG / f"a7-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    res = compute(verbose=False)
    figs = figures(res)
    labels = list(res["series"])
    nas, nik = (res["series"][k] for k in labels)
    hw = str(HEADLINE_WINDOW)
    vix = res["vix"]
    battery_body = md_table(BATTERY_HEADER, battery_rows(res))
    gap = {k: res["series"][k]["inside"][hw] - res["series"][k]["baseline"][hw]
           for k in labels}
    share = {k: gap[k] / res["series"][k]["whole_series_acf1_abs"] for k in labels}
    ten = str(2520)
    dec_range = {k: (min(v["excess_kurtosis"]
                         for v in res["series"][k]["decades"].values()),
                     max(v["excess_kurtosis"]
                         for v in res["series"][k]["decades"].values()))
                 for k in labels}
    # The post's spine is that a windowed generator sees a fraction of the effect.
    # Assert it rather than trusting prose written against an earlier run.
    for k in labels:
        if not 0.0 < share[k] < 0.5:
            raise AssertionError(
                f"{k}: within-window clustering is {100 * share[k]:.0f}% of the "
                f"whole-series value, which breaks the post's argument")

    post = Post(
        title="Most Volatility Clustering Does Not Fit Inside the Window",
        slug="clustering-does-not-fit-inside-the-window",
        subtitle=("77 years of Nikkei and 55 of NASDAQ, against a prediction a "
                  "simulation made three posts ago"),
        summary=(
            f"Volatility clustering is the second thing anyone says about financial "
            f"returns, and it is normally quoted as the lag-1 autocorrelation of "
            f"absolute returns: **{nas['whole_series_acf1_abs']:+.2f}** for the NASDAQ "
            f"Composite over {nas['stats']['n']:,} trading days since "
            f"{nas['stats']['first_date'][:4]}, "
            f"**{nik['whole_series_acf1_abs']:+.2f}** for the Nikkei 225 over "
            f"{nik['stats']['n']:,} since {nik['stats']['first_date'][:4]}. Measured "
            f"*inside* a {HEADLINE_WINDOW}-day window — the horizon generative models "
            f"are trained and judged on — the same two series give "
            f"{nas['inside'][hw]:+.3f} and {nik['inside'][hw]:+.3f}, against a "
            f"shuffled baseline near "
            f"{np.mean([res['series'][k]['baseline'][hw] for k in labels]):+.3f}. "
            f"About **{100 * share[labels[0]]:.0f}%** of the effect fits in the "
            f"window. A previous post predicted this from a simulation; this is the "
            f"real data, and it also says the number those models are matched against "
            f"is unstable by a factor of {dec_range[labels[0]][1] / dec_range[labels[0]][0]:.0f} "
            f"across decades and, on a ten-year sample, is decided by whether one "
            f"Monday in 1987 is inside it."),
        tags=["quantitative-finance", "time-series", "generative-models",
              "volatility", "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=1500, max_words=2400,
        table_figures=[figs["battery"]],
        data_sources=[
            f"NASDAQ Composite daily close (NASDAQCOM), {nas['stats']['first_date']} "
            f"to {nas['stats']['last_date']}, {nas['stats']['n']:,} usable daily log "
            f"returns; Nikkei 225 daily close (NIKKEI225), "
            f"{nik['stats']['first_date']} to {nik['stats']['last_date']}, "
            f"{nik['stats']['n']:,} returns; CBOE Volatility Index (VIXCLS), "
            f"{vix['first_date']} to {vix['last_date']}, {vix['n']:,} daily levels. "
            f"All via FRED, Federal Reserve Bank of St. Louis, downloaded "
            f"{time.strftime('%d %B %Y')}. {MANDATORY_DISCLAIMER}",
            "These series are not redistributable, so this post publishes statistics "
            "and never values: no return series, no dated observation, and no minimum "
            "or maximum, which are the most identifying values a return series has. "
            "Every figure plots a statistic against a parameter rather than a series "
            "against time. Input files are git-ignored and recorded by sha256 in the "
            "reproducibility notes.",
            "Loaded by `standarderror.sources.prices.load_prices`, which handles FRED's "
            "bare '.' for non-trading days, and reduced to statistics by "
            "`publishable_statistics`. Battery and controls from "
            "`standarderror.generative.stylised`, the same code the simulated version of "
            "this experiment used.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "scipy", "pandas", "standarderror")),
            "vintage_sha256": ", ".join(
                f"{k}: {v['sha256'][:16]}" for k, v in res["vintage"].items()),
            "returns": ("log returns in percent, `100 * diff(log(close))`, with "
                        "non-trading days dropped rather than bridged, so no return "
                        "spans a market holiday"),
            "within_window": ("lag-1 correlation of |r| computed inside each window "
                              "and averaged over windows, never across a window "
                              "boundary; windows strided by "
                              f"{STRIDE} days"),
            "shuffled_baseline": ("the same statistic on a permutation of the same "
                                  "returns; it is negative rather than zero because "
                                  "the lag-1 estimator is biased downwards in short "
                                  "windows, which is why zero is the wrong reference"),
            "kurtosis_spread": (f"{N_DRAWS} contiguous stretches per sample size at "
                                f"random start dates, because a real sample is a "
                                f"stretch of history and a random subset would "
                                f"understate the dependence between observations"),
            "block_sweep_nasdaq": ", ".join(
                f"block {b}: {nas['block_sweep'][str(b)]:+.3f}" for b in BLOCK_SWEEP),
            "vix_check": (f"VIX level lag-1 autocorrelation "
                          f"{vix['level_acf1']:+.3f} over {vix['n']:,} days, an "
                          f"independent read on the persistence the simulated version "
                          f"of this experiment assumed"),
            "cost": (f"about {res['elapsed_seconds']:.0f} seconds, cached under a "
                     f"hash of the configuration and of the input file bytes, so a "
                     f"new data vintage recomputes automatically"),
        },
    )

    post.add("A prediction from a simulation, and 33,000 days to test it on", f"""
Three posts ago I read a survey of diffusion models applied to finance and ran the
experiment it implied: train a small generative model on fixed-length windows of a
process whose answers are known, and see which stylised facts come back. One result
was awkward enough to be worth checking against reality.

A generator that emits {HEADLINE_WINDOW}-step windows can only represent dependence
that fits *inside* one. And at the volatility persistence an equity index normally
shows, I measured almost no clustering inside a short window — about
**{EXP013_PREDICTION:+.3f}** at 32 steps. If that carried over to real markets it
would mean something uncomfortable: that models trained on windows of this length are
being credited with reproducing a fact their architecture cannot express.

That was a simulation. This is not.

Two of the longest daily equity histories anyone publishes: the **NASDAQ Composite
from {nas['stats']['first_date'][:4]}**, {nas['stats']['n']:,} returns, and the
**Nikkei 225 from {nik['stats']['first_date'][:4]}**, {nik['stats']['n']:,} —
about 33,000 trading days between them, spanning 1987, 1990, 2000, 2008 and 2020.
Long enough that the tail events which decide these statistics are actually in the
sample.
""".strip())

    post.add(f"A fifth of it fits in {HEADLINE_WINDOW} days", f"""
Volatility clustering is normally quoted as the lag-1 autocorrelation of absolute
returns, and on these series it is unmistakable:
**{nas['whole_series_acf1_abs']:+.2f}** for the NASDAQ,
**{nik['whole_series_acf1_abs']:+.2f}** for the Nikkei. Large moves arrive next to
large moves. Nobody disputes this and the data does not either.

Now measure the same statistic *inside* a {HEADLINE_WINDOW}-day window, averaging over
windows and never crossing a boundary. The NASDAQ gives **{nas['inside'][hw]:+.3f}**.
The Nikkei gives **{nik['inside'][hw]:+.3f}**.

Zero is the wrong thing to compare those to, because the lag-1 estimator is biased
downwards in short samples: shuffle the same returns and the within-window statistic
reads about {nas['baseline'][hw]:+.3f}. So for the NASDAQ the clustering genuinely
visible inside the window is {nas['inside'][hw]:+.3f} minus that baseline, or
**{gap[labels[0]]:+.3f}** — which is **{100 * share[labels[0]]:.0f}%** of the
{nas['whole_series_acf1_abs']:+.2f} the same series shows over its whole history. The
Nikkei's own baseline-corrected figure is {gap[labels[1]]:+.3f}, or
{100 * share[labels[1]]:.0f}%.

Correcting for the baseline is the generous choice, and deliberately so: taken raw,
without crediting the model for the estimator's downward bias, the NASDAQ's
within-window clustering is {100 * nas['inside'][hw] / nas['whole_series_acf1_abs']:.0f}%
of the headline number rather than {100 * share[labels[0]]:.0f}%.

So the prediction held, and if anything it was conservative. Volatility clustering at
this horizon is mostly a statement about *which window you are in* — whether this
quarter is 2008 or 2017 — rather than about what happens from one day to the next
inside a quarter. The autocorrelation everyone quotes is dominated by the slow drift
of the volatility level across years, and a generator that emits one quarter at a time
never sees that drift.

Which makes window length a modelling decision of the first order rather than a
detail in a table. At {HEADLINE_WINDOW} days a model is being asked for
{100 * share[labels[0]]:.0f}% of the effect; at 512 days, by the same measurement, it
would be asked for about
{100 * (nas['inside']['512'] - nas['baseline']['512']) / nas['whole_series_acf1_abs']:.0f}%.
""".strip(), figures=[figs["window"]])

    post.add("An outside check on the persistence", f"""
That story rests on volatility moving slowly, so it is worth confirming from an
instrument that measures volatility directly rather than inferring it from returns.

The VIX is the options market's own estimate of the next month's volatility. Over
{vix['n']:,} daily observations from {vix['first_date'][:4]}, its **lag-1
autocorrelation is {vix['level_acf1']:+.3f}**.

That is about as persistent as a financial series gets, and it is the number the
simulation assumed. Today's expected volatility is very nearly yesterday's. A process
that persistent barely moves within a quarter, which is exactly why so little of its
autocorrelation shows up inside a {HEADLINE_WINDOW}-day window — and it confirms the
mechanism from data that never entered the calculation above.
""".strip())

    post.add("And the number being matched is not a constant", f"""
Generative papers report matching an index's kurtosis. Real data has an opinion about
that phrasing.

Over its whole history the NASDAQ's excess kurtosis is
**{nas['stats']['excess_kurtosis']:.1f}**. Split by decade — each about 2,500 trading
days, a *larger* sample than most such papers use — it runs from
**{dec_range[labels[0]][0]:.1f}** to **{dec_range[labels[0]][1]:.1f}**. The Nikkei
runs from {dec_range[labels[1]][0]:.1f} to {dec_range[labels[1]][1]:.1f}. Factors of
{dec_range[labels[0]][1] / dec_range[labels[0]][0]:.0f} and
{dec_range[labels[1]][1] / dec_range[labels[1]][0]:.0f}.

Both maxima are the 1980s, and both are essentially one week of the 1980s. Remove
October 1987 from the NASDAQ and remove the end of the Nikkei bubble, and the two
series look like their other decades. A fourth moment is a statistic about the largest
few observations, so a decade containing a crash and a decade not containing one are
not measuring the same quantity.

At realistic sample sizes it stops being a matter of degree. Take
{nas['spread'][ten]['draws']} contiguous ten-year stretches of NASDAQ history at random
start dates — ten years being a common sample — and the excess kurtosis does not come
out uncertain. It comes out **binary**.

The distribution is two-humped. The {nas['spread'][ten]['n_with_event']} samples that
contain **{EVENT_NAME}** average {np.mean(nas['spread'][ten]['with_event']):.1f}; the
{len(nas['spread'][ten]['without_event'])} that do not average
{np.mean(nas['spread'][ten]['without_event']):.1f}. Every single sample above
{max(nas['spread'][ten]['without_event']):.1f} contains that day and not one below it
does — one date classifies the estimate perfectly. The Nikkei splits the same way,
{np.mean(nik['spread'][ten]['without_event']):.1f} against
{np.mean(nik['spread'][ten]['with_event']):.1f}, because October 1987 was global —
though its two humps overlap slightly rather than separating cleanly, so the exactness
is a property of one series and not a law.

So "the excess kurtosis of the NASDAQ over ten years" is not a market property measured
with error. It is a **yes/no question about one Monday**, and which answer you get
depends on a start date nobody chose for statistical reasons.

The mechanism is sample length rather than the crash itself, and that is the useful
part. Delete the same eight days from the full {nas['stats']['n']:,}-day history and
the kurtosis moves only from {nas['stats']['excess_kurtosis']:.1f} to
{nas['kurtosis_without_event_week']:.1f}. One week out of fifty-five years is
negligible; one week out of ten years decides the answer. A fourth moment needs a
sample long enough that no single week is pivotal, and ten years of daily data is not
that sample.

Two-year samples fail in the opposite direction: they average
{nas['spread'][str(500)]['mean']:.1f}, *below* the full-history figure, because the
events that create a fourth moment are usually not in a two-year window at all. A
model matched to a two-year sample has been matched to a market with thin tails.
""".strip(), figures=[figs["decades"], figs["uncertainty"]])

    post.add("The control that behaved on simulated data and not on this", f"""
Running the same battery on real indices turned up one thing I did not expect, and it
is a correction to the previous post rather than an extension of it.

{battery_body}

The shuffle behaves as designed: it reproduces the kurtosis — because it *is* the
return distribution — and destroys the clustering. Fine, and the same as on simulated
data.

The **moving-block bootstrap does not**. On the GARCH path the simulated version of
this experiment used, blocks of {BLOCK} matched the truth almost exactly. Here they
*overshoot*: {nas['rows'][f'blocks of {BLOCK}']['acf1_abs']['value']:+.3f} against the
series' own within-window {nas['rows']['the series']['acf1_abs']['value']:+.3f} for the
NASDAQ, and {nik['rows'][f'blocks of {BLOCK}']['acf1_abs']['value']:+.3f} against
{nik['rows']['the series']['acf1_abs']['value']:+.3f} for the Nikkei.

The mechanism is worth spelling out because it is the same one as the headline result.
The bootstrap draws {BLOCK}-day blocks from anywhere in fifty years, so a block from
2008 can land next to a block from 2017. Inside the stitched window, |r| is large for
sixteen days and then small for sixteen days — a step function. A step function has
strong lag-1 autocorrelation. So the block bootstrap does not preserve this market's
clustering; it manufactures a caricature of it, and the caricature scores higher than
the real thing.

That is only possible because real volatility moves *slower* than the block length,
which is the same fact as Fig 1. On the simulated path, persistence was low enough
that a block contained real variation and the splice added nothing.

One more column worth reading: **leverage**. Real indices have one — a negative return
is followed by a larger absolute move, {nas['rows']['the series']['leverage']['value']:+.3f}
and {nik['rows']['the series']['leverage']['value']:+.3f} here. The symmetric
simulation had none by construction, and the shuffle destroys it. So on real data that
row carries information, and on the simulated data it did not. Which row is
informative depends on the process, and you cannot tell from the table.
""".strip())

    post.add("What this changes", f"""
**Report the window length next to any clustering claim, and report what fraction of
the full-sample statistic that window can hold.** It is two lines of code and, on this
data, the difference between claiming an effect of {nas['whole_series_acf1_abs']:+.2f}
and one of {gap[labels[0]]:+.3f}.

**Give the target an error bar and a sample period.** "Excess kurtosis 9.4" is a
property of {nas['stats']['n']:,} days ending in {nas['stats']['last_date'][:4]}. On
ten years it could have been anything from {nas['spread'][ten]['p5']:.1f} to
{nas['spread'][ten]['p95']:.1f}.

**Do not carry a control across datasets without re-checking it.** The block bootstrap
is a good baseline on a fast-mixing process and an actively misleading one here, and
nothing about the code changed between those two cases.

**And be careful which direction the fix runs.** None of this says windowed generative
models are useless. It says the *evidence* usually offered for them — a stylised-facts
table at a 64-day horizon — is weaker than it looks, and that a longer window or an
explicit volatility state would let a model be judged on the effect people actually
mean. That is a design suggestion, not a verdict.
""".strip())

    post.add("Where to be careful", f"""
**Two indices, and both are equity.** Neither is Korean, neither is intraday, and
neither is a single stock. The mechanism — slow volatility versus a short window —
should hold anywhere volatility is persistent, and the specific fractions should not
be assumed to.

**Lag-1 only.** Volatility clustering has structure at many lags and I have measured
one. A longer-lag measurement would show more of the effect inside a window, though
the direction of the argument does not change: whatever the lag, a window of
{HEADLINE_WINDOW} days cannot express dependence that operates over years.

**The NASDAQ has positive return autocorrelation within windows here**
({nas['rows']['the series']['acf1_returns']['value']:+.3f}), which the efficient-market
version of the stylised facts says should be zero. Much of that is the 1970s, before
the market microstructure that removes it. I have not decomposed it, and it does not
touch the clustering result.

**Sample periods overlap.** The ten-year stretches in Fig 3 are drawn at random start
dates from one history, so they share data and the shape of that distribution is a
statement about this history rather than about ten-year windows in general. The
NASDAQ's separation on {EVENT_NAME} is exact; the Nikkei's humps overlap slightly
({min(nik['spread'][ten]['with_event']):.1f} against
{max(nik['spread'][ten]['without_event']):.1f}), so "perfectly classified" is a claim
about one series and not a law.

**And the previous post's conclusion about the block bootstrap was wrong for real
data.** I am leaving it as written there and correcting it here, because the useful
thing is that the same code gave opposite answers on simulated and real inputs, and
that is only visible if both are on the record.
""".strip())
    return post


if __name__ == "__main__":
    compute(force=bool(os.environ.get("SERR_FORCE")))
