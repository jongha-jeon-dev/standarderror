"""exp018 — a correlation of 0.96 between two unrelated series is the median case.

The wrong version of the question
---------------------------------
"These two series correlate at 0.96 — what explains it?" The question assumes
0.96 is a fact needing an explanation. For two trending series of ordinary
length it is closer to what you should expect when *nothing* connects them.

What this post establishes
--------------------------
1. **The rejection rate rises with the sample.** Regress one random walk on an
   independent one and the t-test on the slope rejects at 55% of samples when
   T = 25 and 94% when T = 1600, against a nominal 5%. More data does not wash
   the artefact out; it sharpens it.

2. **`|t| / sqrt(T)` does not move.** It sits at 0.44-0.46 across a 64-fold
   change in T. That is Phillips (1986): the statistic diverges at rate
   sqrt(T), so the problem is a rate, not noise. This is the post's control —
   if the scaled quantity had drifted too, the explanation would be wrong.

3. **Drift makes it total.** Give both walks a small upward drift, which is what
   every real macro series has, and at T = 1600 the rejection rate is 100%, the
   median |t| is 129, and the median |r| is 0.955. The headline number is not
   an anecdote; it is the middle of the distribution.

4. **The standard fix is usually applied with the wrong table.** Engle-Granger
   runs a Dickey-Fuller regression on residuals from a *fitted* cointegrating
   regression. Because OLS already chose the combination that makes those
   residuals smallest, the null distribution shifts left, and using the
   Dickey-Fuller critical value turns a nominal 5% test into a 14-16% one. That
   distortion does not shrink with T either.

Why simulation is the primary evidence and not a fallback
---------------------------------------------------------
The claim is about the behaviour of a statistic under a null. Ground truth for
"these two series share no cause" exists only where the two series were
generated independently — which is to say, only in a simulation. A real pair can
illustrate the claim; it cannot test it, because you never know what a real pair
shares.

What keeps the simulation honest is that its by-products have published answers.
The same machinery that produces the rejection curve also produces critical
values for the Dickey-Fuller and Engle-Granger tests, and those were computed by
MacKinnon (2010) on a response surface and are implemented independently in
statsmodels. Both are checked in `tests/test_nonstationary.py`. A simulation that
reproduces someone else's table is a simulation whose plumbing works.

Discipline
----------
No cherry-picked real pair. The real-data section (when its data is available)
reports the full distribution of |r| across country-crossed indicator pairs and
then draws its illustration *from* that distribution, rather than searching for a
striking pair and presenting it as a discovery.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
import os
import time

import numpy as np

import standarderror as se
from standarderror.render.post import Post, Section
from standarderror.ts import nonstationary as ns
from standarderror.viz import charts, theme

#: Pinned so a rebuild cannot silently re-date a published post. This one
#: had no record to pin from -- its Hugo page was never committed and the
#: manifests had already drifted -- so the date comes from the creation
#: date of the post's Notion page, the only surviving evidence.
POST_DATE = date(2026, 8, 25)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed
CACHE = se.SETTINGS.build_dir / "cache" / "exp018.json"

# ---------------------------------------------------------------- configuration

SIZES = (25, 50, 100, 200, 400, 800, 1600)
DRIFT = 0.1                  # small enough to be invisible on a plot of one path
REPS = 40000
CV_SIZES = (100, 250, 500, 1000, 1500)
CV_REPS = 30000
ALPHA = 0.05
HISTOGRAM_T = 1600           # the panel the |r| histogram is drawn at

# A drift of 0.1 with unit shocks means the trend contributes 0.1*T to the level
# while the stochastic part contributes about sqrt(T). The two are comparable at
# T = 100 and the trend dominates by T = 1600 — which is exactly the regime real
# annual macro series live in, and the reason this is the interesting case.
DRIFT_NOTE = "drift 0.1 per step against unit shocks"

SOURCES = [
    "Simulated. Two independent Gaussian random walks per replication; no real "
    "data enters findings 1-4. Code: standarderror/ts/nonstationary.py.",
    "C. W. J. Granger and P. Newbold, 'Spurious regressions in econometrics', "
    "Journal of Econometrics 1974;2:111-120 — the original demonstration.",
    "P. C. B. Phillips, 'Understanding spurious regressions in econometrics', "
    "Journal of Econometrics 1986;33:311-340 — the asymptotics that explain why "
    "the t-statistic diverges rather than merely misbehaving.",
    "R. F. Engle and C. W. J. Granger, 'Co-integration and error correction', "
    "Econometrica 1987;55:251-276 — the two-step test.",
    "J. G. MacKinnon, 'Critical Values for Cointegration Tests', Queen's "
    "Economics Department Working Paper No. 1227, 2010 — the published critical "
    "values this post's simulation is checked against.",
]


def _config_key() -> str:
    # The key covers the *functions* as well as their inputs: a fix inside
    # nonstationary.py with the config unchanged must not be served from cache.
    blob = json.dumps({"v": 3, "sizes": list(SIZES), "drift": DRIFT,
                       "reps": REPS, "cv_sizes": list(CV_SIZES),
                       "cv_reps": CV_REPS, "alpha": ALPHA, "seed": SEED,
                       "hist_t": HISTOGRAM_T, "window": list(WINDOW),
                       "indicators": sorted(INDICATORS), "eg_sub": EG_SUBSAMPLE,
                       "impl": hashlib.sha256(
                           open(ns.__file__, "rb").read()).hexdigest()[:12]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- computation

def sweep(rng_seed: int, say) -> tuple[dict, dict, dict]:
    """One pass per (T, drift), from which every finding is read.

    Rejection rate, scaled-t behaviour and the correlation distribution come off
    the *same* simulated pairs. That is not only cheaper — it means findings 1-3
    describe one experiment, so a reader cannot wonder whether the correlation
    figure was run on a luckier batch than the rejection figure.
    """
    div: dict[str, dict] = {"plain": {}, "drifting": {}}
    corr: dict[str, dict] = {"plain": {}, "drifting": {}}
    sample: dict[str, list] = {}
    for i, n in enumerate(SIZES):
        for name, drift in (("plain", 0.0), ("drifting", DRIFT)):
            g = np.random.default_rng(rng_seed + 1000 * i
                                      + (0 if name == "plain" else 1))
            d = ns.spurious_draws(n, reps=REPS, drift=drift, rng=g)
            r = ns.spurious_rejection_rate(n, alpha=ALPHA, draws=d)
            q = ns.scaled_t_quantiles(n, draws=d)
            div[name][str(n)] = {
                "rejection_rate": r["rejection_rate"],
                "median_abs_t": r["median_abs_t"],
                "median_r2": r["median_r2"],
                "p90_r2": r["p90_r2"],
                "scaled_t_median": q[0.5],
                "scaled_t_p05": q[0.05],
                "scaled_t_p95": q[0.95],
            }
            corr[name][str(n)] = ns.correlation_of_independent_walks(n, draws=d)
            if n == HISTOGRAM_T:
                sample[name] = np.abs(d["r"]).tolist()
            del d
        say(f"  T={n} done")
    return div, corr, sample


def critical_values(rng_seed: int) -> dict:
    """Simulated null quantiles at each sample size.

    Levels are stringified on the way out. JSON has no float keys, so without
    this a fresh run hands back `{0.05: ...}` and a cached run hands back
    `{"0.05": ...}` — two different objects from the same function, and the
    difference only shows up on the *second* run.
    """
    def keyed(d):
        return {f"{lv:.2f}": v for lv, v in d.items()}

    out: dict[str, dict] = {"df": {}, "eg": {}}
    for i, n in enumerate(CV_SIZES):
        out["df"][str(n)] = keyed(ns.df_critical_values(
            n, reps=CV_REPS, rng=np.random.default_rng(rng_seed + i)))
        out["eg"][str(n)] = keyed(ns.eg_critical_values(
            n, reps=CV_REPS, rng=np.random.default_rng(rng_seed + 500 + i)))
    return out


def misuse(rng_seed: int) -> dict:
    return {str(n): ns.misuse_size(n, reps=CV_REPS, alpha=ALPHA,
                                   rng=np.random.default_rng(rng_seed + 9 * i))
            for i, n in enumerate(CV_SIZES)}


# ---------------------------------------------------------------- real data

DATA_DIR = se.SETTINGS.build_dir.parent / "data" / "worldbank"
INDICATORS = {
    "fertility rate": "SP.DYN.TFRT.IN",
    "life expectancy": "SP.DYN.LE00.IN",
    "urban population share": "SP.URB.TOTL.IN.ZS",
    "CO2 per capita": "EN.GHG.CO2.PC.CE.AR5",
}
WINDOW = (1970, 2022)
EG_SUBSAMPLE = 40000

# Chosen before looking at any correlation: the four indicators with complete
# coverage over a long window. R&D spending and private credit were downloaded
# too and dropped because *no* country has complete R&D data over 1996-2022,
# which would have cut the panel to nothing. Renewable share and GDP per capita
# start only in 1990 and would have cost 20 years of window for 8 indicators
# over 49 countries instead of 4 over 197. None of these choices were made
# after seeing a correlation.
DROPPED_INDICATORS = {
    "R&D spending (% GDP)": "no country has all years 1996-2022",
    "private credit (% GDP)": "would cut the panel to 28 countries",
    "renewable share": "starts 1990, costs 20 years of window",
    "GDP per capita PPP": "starts 1990, costs 20 years of window",
}

HEADLINE_PAIR = {
    "a_indicator": "life expectancy", "a_country": "BTN", "a_name": "Bhutan",
    "b_indicator": "urban population share", "b_country": "BEL",
    "b_name": "Belgium",
}


def load_panel():
    from standarderror.sources import worldbank_bulk as wbb
    from standarderror.ts import panelpairs as pp
    frames = {}
    for name, code in INDICATORS.items():
        hits = sorted(DATA_DIR.glob(f"API_{code}_*.zip"))
        if not hits:
            raise FileNotFoundError(
                f"no World Bank zip for {code} in {DATA_DIR}. Download it from "
                f"https://api.worldbank.org/v2/en/indicator/{code}"
                f"?downloadformat=csv")
        wide, meta = wbb.read_zip(hits[0])
        frames[name] = wbb.to_long(wide, meta)
    return pp.stack(frames, start=WINDOW[0], end=WINDOW[1], min_countries=50)


def real_data(say) -> dict:
    from standarderror.ts import panelpairs as pp

    panel = load_panel()
    say(f"  panel: {len(panel)} series, {len(set(panel.country))} countries, "
        f"T={panel.n_years}")

    levels = pp.summarise(panel)
    differenced = pp.summarise(panel, difference=True)
    top = pp.extremes(panel, top=12)

    # The named pair, recomputed from its own two rows rather than read out of
    # the big matrix, so the number in the title has a second derivation.
    ai = np.where((panel.indicator == HEADLINE_PAIR["a_indicator"])
                  & (panel.country == HEADLINE_PAIR["a_country"]))[0][0]
    bi = np.where((panel.indicator == HEADLINE_PAIR["b_indicator"])
                  & (panel.country == HEADLINE_PAIR["b_country"]))[0][0]
    xa, xb = panel.values[ai], panel.values[bi]
    r_head = float(np.corrcoef(xa, xb)[0, 1])
    d_head = float(np.corrcoef(np.diff(xa), np.diff(xb))[0, 1])
    D = np.concatenate([np.ones((1, panel.n_years, 1)),
                        xb.reshape(1, -1, 1)], axis=2)
    fit = ns.ols(D, xa.reshape(1, -1))
    eg_head = float(ns.engle_granger_stat(xa.reshape(1, -1), xb.reshape(1, -1))[0])

    # Every unrelated pair, thresholded — the population the named pair is drawn
    # from. Reported before the pair in the post, deliberately.
    r_all = pp.correlation_matrix(panel.values)
    groups = pp.pair_groups(panel)
    iu = groups["index"]
    unrelated = np.abs(r_all[iu])[groups["unrelated"]]
    thresholds = {f"{t}": {"count": int((unrelated > t).sum()),
                           "share": float((unrelated > t).mean())}
                  for t in (0.5, 0.8, 0.9, 0.96, 0.99, 0.999)}

    # And what the two tests actually do on real unrelated pairs.
    say("  running the tests on a subsample of real unrelated pairs")
    i = iu[0][groups["unrelated"]]
    j = iu[1][groups["unrelated"]]
    rng = np.random.default_rng(SEED)
    pick = rng.choice(i.size, size=min(EG_SUBSAMPLE, i.size), replace=False)
    i, j = i[pick], j[pick]
    naive, egs = [], []
    for s in range(0, len(i), 5000):
        y, x = panel.values[i[s:s + 5000]], panel.values[j[s:s + 5000]]
        Dm = np.concatenate([np.ones((y.shape[0], panel.n_years, 1)),
                             x[..., None]], axis=2)
        naive.append(np.abs(ns.ols(Dm, y).t[:, 1]))
        egs.append(ns.engle_granger_stat(y, x))
    naive = np.concatenate(naive)
    egs = np.concatenate(egs)

    return {
        "window": list(WINDOW),
        "n_series": len(panel), "n_years": panel.n_years,
        "n_countries": len(set(panel.country)),
        "dropped_constant": list(panel.dropped_constant),
        "levels": levels, "differenced": differenced,
        "top_pairs": top, "thresholds": thresholds,
        "n_unrelated": int(unrelated.size),
        "max_abs_r": float(unrelated.max()),
        "headline": {
            **HEADLINE_PAIR, "r": r_head, "r_differenced": d_head,
            "t": float(fit.t[0, 1]), "r2": float(fit.r2[0]),
            "eg_stat": eg_head,
            "eg_rejects": bool(eg_head < ns.MACKINNON_EG[0.05]),
            "a_first": float(xa[0]), "a_last": float(xa[-1]),
            "b_first": float(xb[0]), "b_last": float(xb[-1]),
        },
        "tests_on_real_pairs": {
            "n": int(len(naive)),
            "naive_significant": float((naive > 1.96).mean()),
            "median_abs_t": float(np.median(naive)),
            "eg_with_df_table": float((egs < ns.MACKINNON_DF[0.05]).mean()),
            "eg_with_eg_table": float((egs < ns.MACKINNON_EG[0.05]).mean()),
            "eg_with_eg_table_1pc": float((egs < ns.MACKINNON_EG[0.01]).mean()),
        },
    }


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

    say("sweeping sample size: rejection rate, scaled t, correlation")
    div, corr, sample = sweep(SEED, say)
    say("building the Dickey-Fuller and Engle-Granger null distributions")
    cv = critical_values(SEED + 31)
    say("costing the wrong critical-value table")
    mis = misuse(SEED + 41)
    say("turning to the real panel")
    real = real_data(say)

    out = {"key": key, "divergence": div, "correlation": corr,
           "sample": sample, "critical_values": cv, "misuse": mis,
           "real": real, "elapsed_s": round(time.time() - t0, 1)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    say("done")
    return out


# ---------------------------------------------------------------- figures

def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown table with pipes inside cells escaped.

    The site renders this; Medium strips table markup on paste, so the same
    numbers also ship as an image via `Post.table_figures`, which substitutes
    one for the other. A figure must therefore be a table figure or a section
    figure, never both, or it appears twice.
    """
    def cell(x):
        return str(x).replace("|", r"\|")
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


CV_TABLE_HEADER = ["series length", "Dickey-Fuller 5%", "Engle-Granger 5%",
                   "Dickey-Fuller 1%", "Engle-Granger 1%"]
PAIR_TABLE_HEADER = ["pair type", "pairs", "median |r|", "|r| > 0.9",
                     "median |r|, differenced", "> 0.9, differenced"]


def figures(res: dict) -> dict:
    import pandas as pd

    figs: dict = {}
    div, corr, cv, mis = (res["divergence"], res["correlation"],
                          res["critical_values"], res["misuse"])
    xs = list(SIZES)

    # ---- F1: the rejection rate does not settle -------------------------
    frame = pd.DataFrame(
        {"no drift": [div["plain"][str(n)]["rejection_rate"] for n in xs],
         f"both series trending ({DRIFT_NOTE})":
             [div["drifting"][str(n)]["rejection_rate"] for n in xs]},
        index=xs)

    def mark_nominal(fig, ax):
        ax.axhline(ALPHA, color=theme.MODES["light"].muted, lw=1.6,
                   ls=(0, (4, 3)), zorder=1)
        ax.text(xs[0], ALPHA + 0.025, "what the test promises: 5%",
                color=theme.MODES["light"].ink_secondary, fontsize=8.5, va="bottom")
        ax.set_ylim(0, 1.04)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(n) for n in xs])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])

    last = div["plain"][str(xs[-1])]["rejection_rate"]
    figs["f1"], _ = charts.lines(
        frame,
        title="More data makes a spurious result more significant, not less",
        subtitle=(f"Two independent random walks, {REPS:,} pairs at each length. "
                  f"The slope between them is zero by construction, so an honest "
                  f"test would sit on the dashed line at every sample size."),
        xlabel="length of the series (T, log scale)",
        ylabel="share of samples calling the slope significant",
        source=("Simulated; nominal two-sided test at 5% using the normal critical "
                "value 1.96. standarderror/ts/nonstationary.py"),
        alt=("A line chart against sample size on a log axis. Both lines start "
             "above 50% at T=25 and rise steadily. The no-drift line reaches "
             f"{_pct(last)} at T=1600; the trending line reaches 100% by T=1600. "
             "A dashed reference line sits near the bottom at 5%."),
        caption=(f"Conclude that the usual reassurance is backwards here. At the "
                 f"shortest sample tested the test already fails "
                 f"{_pct(div['plain']['25']['rejection_rate'])} of the time, and "
                 f"collecting 64 times more data takes it to {_pct(last)}. When both "
                 f"series trend — which every macro series does — it reaches "
                 f"certainty."),
        mode="light", direct_labels=False, decorate=mark_nominal,
        path=str(IMG / f"a10-f1-divergence.{EXT}"))

    # ---- F2: why — the statistic diverges at rate sqrt(T) ---------------
    # Both panels show the SAME three percentiles of the same quantity; the right
    # one is only divided by sqrt(T). Quantiles survive a monotone transform
    # exactly, so the raw percentiles are the scaled ones times sqrt(T) — no
    # second simulation, and no chance of the two panels disagreeing.
    import matplotlib.pyplot as plt

    root = np.sqrt(np.asarray(xs, dtype=float))
    bands = {}
    for tag, key in (("95th percentile", "scaled_t_p95"),
                     ("median", "scaled_t_median"),
                     ("5th percentile", "scaled_t_p05")):
        bands[tag] = np.array([div["plain"][str(n)][key] for n in xs])

    m = theme.apply("light", figsize=(7.6, 3.6))
    fig, axes = plt.subplots(1, 2, sharex=True)
    cols = theme.series_colors(3, "light")
    for ax, divide, ylab in (
            (axes[0], False, "|t| on the slope"),
            (axes[1], True, "|t| / √T")):
        for (tag, vals), c in zip(bands.items(), cols):
            y = vals if divide else vals * root
            ax.plot(xs, y, color=c, lw=2.0, label=tag, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(n) for n in xs], fontsize=8)
        ax.set_xlabel("series length T")
        ax.set_ylabel(ylab)
        ax.grid(axis="y")
        ax.set_ylim(0, None)
    axes[0].axhline(1.96, color=m.muted, lw=1.4, ls=(0, (4, 3)), zorder=1)
    axes[0].text(xs[-1], 3.4, "1.96, the level a practitioner tests against",
                 color=m.ink_secondary, fontsize=8, ha="right")
    axes[1].set_ylim(0, 1.75)
    # No per-panel titles: the legend sits directly above the left axes and any
    # axes title collides with it. The y-labels already say which panel is which.

    spread = float(max(bands["median"]) - min(bands["median"]))
    raw_lo = float(bands["median"][0] * root[0])
    raw_hi = float(bands["median"][-1] * root[-1])
    theme.finish(
        axes[0],
        title="The problem is a rate, not noise",
        subtitle=(f"Phillips (1986): the t-statistic has no limiting distribution, "
                  f"but |t|/√T has one. Left, the statistic as reported — the median "
                  f"goes from {raw_lo:.1f} to {raw_hi:.1f} and every percentile "
                  f"climbs with it. Right, the same three percentiles divided by √T: "
                  f"the whole distribution stops moving, the median by "
                  f"{spread:.3f} across a 64-fold change in T."),
        source=("Simulated, no drift, the same pairs as Figure 1. Both panels start "
                "at zero, so flat means flat."),
        mode="light", legend=True, legend_ncol=3)
    path = theme.save(fig, str(IMG / f"a10-f2-scaling.{EXT}"), mode="light")
    figs["f2"] = charts.Figure(
        path,
        alt=("Two panels against series length on a log axis, both starting at "
             "zero. In the left panel three percentile lines all fan upward, the "
             "highest reaching about 60 and the median about 18, above a dashed "
             "reference at 1.96. In the right panel the same three lines are "
             "perfectly flat at roughly 1.50, 0.45 and 0.04."),
        caption=("Conclude that nothing here is converging on the truth. The raw "
                 "statistic is not noisy around zero — it is travelling to infinity "
                 "at a known speed, and the right panel is that speed divided out. "
                 "This is the post's control: had the right panel drifted too, the "
                 "explanation would be wrong."),
        title="The problem is a rate, not noise", mode="light")

    # ---- F3: how ordinary a correlation of 0.96 is ----------------------
    drifting = np.asarray(res["sample"]["drifting"])
    plain = np.asarray(res["sample"]["plain"])
    grid = np.linspace(0, 1, 240)
    dens, edges = np.histogram(plain, bins=60, range=(0, 1), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    c1600 = corr["drifting"][str(HISTOGRAM_T)]
    figs["f3"], _ = charts.histogram(
        drifting, bins=60,
        overlay={"same length, no drift": (centres, dens)},
        mark={"0.96": 0.96},
        series_label=f"both series trending, T={HISTOGRAM_T}",
        title="A correlation of 0.96 between unrelated series is the middle of the "
              "distribution, not the tail",
        subtitle=(f"|r| between two independently generated trending walks of "
                  f"length {HISTOGRAM_T}. Median {c1600['median_abs_r']:.3f}; "
                  f"{_pct(c1600['p_abs_r_over_0.96'])} of pairs exceed 0.96. "
                  f"Nothing connects any pair."),
        xlabel="|r| between two series that share no cause",
        source="Simulated; 40,000 pairs per condition. standarderror/ts/nonstationary.py",
        alt=("A density histogram of absolute correlation running from 0 to 1. The "
             "trending case is a tall spike piled against the right edge, almost "
             "all of it above 0.85, peaking just below 1.0. The no-drift overlay is "
             "a nearly flat line at a density of about 1.2 across the whole range "
             "from 0 to 0.8, falling away only in the last fifth. A vertical line "
             "marks 0.96 and passes through the middle of the trending spike."),
        caption=(f"Conclude that a striking correlation between two trending series "
                 f"is not evidence of anything. Remove the trend and the same "
                 f"generating process gives a median of "
                 f"{corr['plain'][str(HISTOGRAM_T)]['median_abs_r']:.3f} — the trend, "
                 f"not the relationship, is what produced the number."),
        mode="light",
        path=str(IMG / f"a10-f3-correlation.{EXT}"))

    # ---- F4: the standard fix, applied with the wrong table -------------
    mframe = pd.DataFrame(
        {"Dickey-Fuller table (the common mistake)":
             [mis[str(n)]["size_using_df_table"] for n in CV_SIZES],
         "Engle-Granger table (correct)":
             [mis[str(n)]["size_using_eg_table"] for n in CV_SIZES]},
        index=list(CV_SIZES))

    def mark_five(fig, ax):
        ax.axhline(ALPHA, color=theme.MODES["light"].muted, lw=1.6,
                   ls=(0, (4, 3)), zorder=1)
        ax.set_ylim(0, 0.20)
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
        ax.set_yticklabels(["0", "5%", "10%", "15%", "20%"])
        ax.set_xscale("log")
        ax.set_xticks(list(CV_SIZES))
        ax.set_xticklabels([str(n) for n in CV_SIZES])

    worst = max(mis[str(n)]["size_using_df_table"] for n in CV_SIZES)
    least = min(mis[str(n)]["size_using_df_table"] for n in CV_SIZES)
    figs["f4"], _ = charts.lines(
        mframe,
        title="The usual remedy is routinely run against the wrong critical value",
        subtitle=("Engle-Granger tests a residual that OLS already made as "
                  "stationary-looking as it could, so its null sits further left "
                  "than Dickey-Fuller's. Reading the Dickey-Fuller table turns a "
                  "5% test into a 15% one."),
        xlabel="length of the series (T, log scale)",
        ylabel="true rejection rate when nothing is cointegrated",
        source="Simulated; 30,000 independent pairs per point. Nominal level 5%.",
        alt=("Two lines against series length on a log axis, on a 0 to 20% scale. "
             "The upper line runs flat near 15% across the whole range with a "
             "slight downward tilt. The lower line starts just above the dashed 5% "
             "reference and settles onto it by the longest samples."),
        caption=(f"Conclude that this one does not wash out either. The distortion "
                 f"is {_pct(worst)} at T=100 and still {_pct(least)} at T=1,500 — a "
                 f"bias in the reference distribution, which more data cannot "
                 f"correct."),
        mode="light", direct_labels=False, decorate=mark_five,
        path=str(IMG / f"a10-f4-misuse.{EXT}"))

    # ---- T1: the plumbing check ----------------------------------------
    rows = []
    for n in CV_SIZES:
        rows.append([str(n),
                     f"{cv['df'][str(n)]['0.05']:+.3f}",
                     f"{cv['eg'][str(n)]['0.05']:+.3f}",
                     f"{cv['df'][str(n)]['0.01']:+.3f}",
                     f"{cv['eg'][str(n)]['0.01']:+.3f}"])
    # How far the longest simulated row sits from the published one, measured
    # rather than asserted — the caption quotes this number.
    longest = str(CV_SIZES[-1])
    worst_gap = max(
        abs(cv["df"][longest]["0.05"] - ns.MACKINNON_DF[0.05]),
        abs(cv["eg"][longest]["0.05"] - ns.MACKINNON_EG[0.05]),
        abs(cv["df"][longest]["0.01"] - ns.MACKINNON_DF[0.01]),
        abs(cv["eg"][longest]["0.01"] - ns.MACKINNON_EG[0.01]))
    rows.append(["MacKinnon (2010), T→∞",
                 f"{ns.MACKINNON_DF[0.05]:+.4f}", f"{ns.MACKINNON_EG[0.05]:+.4f}",
                 f"{ns.MACKINNON_DF[0.01]:+.4f}", f"{ns.MACKINNON_EG[0.01]:+.4f}"])
    figs["t1"], _ = charts.table_image(
        rows,
        header=CV_TABLE_HEADER,
        title="The simulation reproduces a table someone else computed",
        subtitle=("Critical values built here by simulating the null, against "
                  "MacKinnon's published response surface. Agreement is the "
                  "evidence that the machinery behind every other figure works; "
                  "the two right-hand columns being more negative than the two "
                  "left-hand ones is the reason Figure 4 exists."),
        source="Simulated, 30,000 replications per cell. Last row: MacKinnon (2010).",
        alt=("A five-column table of negative critical values by series length, "
             "with a final row giving MacKinnon's asymptotic values. Each column "
             "moves toward its published value as length grows, and the longest "
             "row sits within about one hundredth of it. Both Engle-Granger "
             "columns are roughly half a point more negative than their "
             "Dickey-Fuller counterparts at every length."),
        caption=(f"Conclude that the null distributions are right. At T=1,500 all "
                 f"four simulated values sit within {worst_gap:.3f} of MacKinnon's "
                 f"published figures, which is the Monte Carlo error at 30,000 "
                 f"replications — so the machinery behind every other figure has "
                 f"been checked against an answer computed elsewhere, a different "
                 f"way. Separately: every Engle-Granger column is about half a "
                 f"point more negative than its Dickey-Fuller twin, and that gap "
                 f"is the whole finding in Figure 4."),
        mode="light", bold_cols=(0,),
        path=str(IMG / f"a10-t1-critical.{EXT}"))

    # ---- F5: the same picture, on real World Bank series ----------------
    real = res["real"]
    from standarderror.ts import panelpairs as pp
    panel = load_panel()
    rmat = pp.correlation_matrix(panel.values)
    dmat = pp.correlation_matrix(np.diff(panel.values, axis=1))
    gg = pp.pair_groups(panel)
    iu = gg["index"]
    lev_u = np.abs(rmat[iu])[gg["unrelated"]]
    dif_u = np.abs(dmat[iu])[gg["unrelated"]]
    dd, ee = np.histogram(dif_u, bins=60, range=(0, 1), density=True)
    cc = 0.5 * (ee[:-1] + ee[1:])
    h = real["headline"]
    figs["f5"], _ = charts.histogram(
        lev_u, bins=60,
        overlay={"the same pairs, in year-on-year changes": (cc, dd)},
        mark={f"{h['a_name']} / {h['b_name']}": abs(h["r"])},
        series_label="as published, in levels",
        title="Real World Bank series behave the way the simulation says they will",
        subtitle=(f"|r| between every pair of indicators in which the two series "
                  f"come from two different countries — {real['n_unrelated']:,} "
                  f"pairs across "
                  f"{real['n_countries']} countries and "
                  f"{len(INDICATORS)} indicators, {real['window'][0]}-"
                  f"{real['window'][1]}. Median "
                  f"{real['levels']['unrelated']['median_abs_r']:.3f} in levels; "
                  f"{real['differenced']['unrelated']['median_abs_r']:.3f} once "
                  f"each series is differenced."),
        xlabel="|r| between two indicators in two different countries",
        source=("World Bank World Development Indicators (CC BY 4.0), downloaded "
                "2026-08-25. Fertility rate, life expectancy, urban population "
                "share, CO2 per capita."),
        alt=("A density histogram of absolute correlation from 0 to 1. The levels "
             "bars are flat and low from 0 to about 0.35, then climb steadily and "
             "peak hard against the right edge near 1.0. The differenced overlay "
             "runs the other way: it starts near a density of 5 at zero, falls "
             "steeply, and is flat against the axis beyond about 0.5. A vertical "
             "line at the far right marks the Bhutan-Belgium pair."),
        caption=(f"Conclude that the artefact is not a property of simulated data. "
                 f"{_pct(real['thresholds']['0.9']['share'])} of these "
                 f"{real['n_unrelated']:,} unrelated pairs correlate above 0.9 and "
                 f"{_pct(real['thresholds']['0.96']['share'])} above 0.96. Take the "
                 f"trend out by differencing and "
                 f"{real['differenced']['unrelated']['p_over_90']:.2%} of the very "
                 f"same pairs clear 0.9. The correlation was the trend."),
        mode="light",
        path=str(IMG / f"a10-f5-worldbank.{EXT}"))

    # ---- T2: the three pair types, before and after differencing --------
    order = ["unrelated", "same indicator, different country",
             "same country, different indicator"]
    label = {"unrelated": "different country, different indicator",
             "same indicator, different country":
                 "same indicator, different country",
             "same country, different indicator":
                 "same country, different indicator"}
    rows2 = []
    for g in order:
        lv, df_ = real["levels"][g], real["differenced"][g]
        rows2.append([label[g], f"{lv['n_pairs']:,}",
                      f"{lv['median_abs_r']:.3f}", f"{lv['p_over_90']:.1%}",
                      f"{df_['median_abs_r']:.3f}", f"{df_['p_over_90']:.1%}"])
    figs["t2"], _ = charts.table_image(
        rows2,
        header=PAIR_TABLE_HEADER,
        title="Being the same country buys almost nothing",
        subtitle=("Two indicators measured in the same country share an obvious "
                  "cause — that country's development — and two indicators in "
                  "different countries do not. In levels the two rows are "
                  "indistinguishable. Structure appears only after differencing, "
                  "and it appears in the same-indicator row."),
        source=("World Bank WDI, 1970-2022, 197 countries. Differenced means "
                "year-on-year change in both series before correlating."),
        alt=("A six-column table of three pair types. Median |r| in levels is "
             "0.779, 0.828 and 0.791 — nearly identical. After differencing they "
             "fall to 0.119, 0.184 and 0.122."),
        caption=("Conclude that a shared cause is not what produced these numbers. "
                 "If it were, the same-country row would stand out in levels; it "
                 "does not. The one row that survives differencing is the same "
                 "indicator in two countries, which is the only genuinely shared "
                 "driver here — global co-movement of one quantity."),
        mode="light", bold_cols=(0,),
        path=str(IMG / f"a10-t2-pairtypes.{EXT}"))

    # ---- hero -----------------------------------------------------------
    def twin_lines(panel, m):
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        t = np.linspace(0.6, 9.4, 60)
        rng = np.random.default_rng(4)
        a = 1.6 + 0.72 * t + np.cumsum(rng.normal(0, 0.16, t.size))
        b = 1.2 + 0.70 * t + np.cumsum(rng.normal(0, 0.16, t.size))
        panel.plot(t, np.clip(a, 0.6, 9.4), color=m.series[0], lw=2.6)
        panel.plot(t, np.clip(b, 0.6, 9.4), color=m.series[1], lw=2.6)
        panel.plot([0.6, 0.6], [0.6, 9.4], color=m.ink, lw=2.0)
        panel.plot([0.6, 9.4], [0.6, 0.6], color=m.ink, lw=2.0)

    def two_dice(panel, m):
        from matplotlib.patches import Rectangle
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        for x0 in (1.1, 5.6):
            panel.add_patch(Rectangle((x0, 3.4), 3.3, 3.3, fc=m.surface,
                                      ec=m.ink, lw=2.4))
        rng = np.random.default_rng(11)
        for x0 in (1.1, 5.6):
            for _ in range(3):
                panel.plot([x0 + rng.uniform(0.7, 2.6)],
                           [3.4 + rng.uniform(0.7, 2.6)],
                           marker="o", ms=7, color=m.ink)
        panel.annotate("", xy=(4.4, 5.05), xytext=(5.6, 5.05),
                       arrowprops={"arrowstyle": "-", "color": m.muted,
                                   "lw": 2.0, "ls": (0, (2, 2))})

    def rising_wall(panel, m):
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        xs_ = np.array([1.2, 2.6, 4.0, 5.4, 6.8, 8.2])
        hs = np.array([2.2, 3.4, 4.6, 5.8, 7.0, 8.2])
        for x0, h in zip(xs_, hs):
            panel.bar(x0, h, width=1.0, bottom=0.8, color=m.series[1],
                      edgecolor=m.ink, linewidth=1.6)
        panel.plot([0.5, 9.5], [1.25, 1.25], color=m.muted, lw=2.0,
                   ls=(0, (4, 3)))
        panel.plot([0.5, 9.5], [0.8, 0.8], color=m.ink, lw=2.0)

    d1600 = div["drifting"][str(HISTOGRAM_T)]
    figs["hero"], _ = charts.strip_card(
        headline="Nothing connects these two lines",
        panels=[
            (twin_lines, f"{c1600['median_abs_r']:.2f}", "median |r|"),
            (two_dice, "0", "links between them"),
            (rising_wall, _pct(d1600["rejection_rate"]), "called significant"),
        ],
        note=("Both lines are random walks generated independently, so the true "
              "slope between them is zero. The t-test does not merely misfire — it "
              "diverges, so a longer sample makes the false result stronger. The "
              "standard remedy is real, and is usually run against the wrong "
              "critical value."),
        footer="The Standard Error", mode="light",
        alt=("A three-panel hand-drawn strip. The first frame shows two lines "
             "climbing together inside a pair of axes, marked "
             f"{c1600['median_abs_r']:.2f}. The second shows two separate boxes of "
             "scattered dots with a broken dashed line between them, marked zero. "
             "The third shows six bars rising left to right, all far above a low "
             f"dashed reference line, marked {_pct(d1600['rejection_rate'])} at "
             "series length 1,600."),
        caption="",
        path=str(IMG / f"a10-hero.{EXT}"))
    figs["_rows"] = {"critical": rows, "pairtypes": rows2}
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute(verbose=False)
    figs = figures(res)

    div, corr, mis = res["divergence"], res["correlation"], res["misuse"]
    real, h = res["real"], res["real"]["headline"]
    tests = real["tests_on_real_pairs"]
    small, big = str(SIZES[0]), str(SIZES[-1])
    p_small = div["plain"][small]["rejection_rate"]
    p_big = div["plain"][big]["rejection_rate"]
    d_big = div["drifting"][big]
    c_big = corr["drifting"][big]
    scaled = [div["plain"][str(n)]["scaled_t_median"] for n in SIZES]
    mis_worst = max(mis[str(n)]["size_using_df_table"] for n in CV_SIZES)
    mis_last = mis[str(CV_SIZES[-1])]
    lev_u = real["levels"]["unrelated"]
    dif_u = real["differenced"]["unrelated"]
    top_urban = sum(1 for t in real["top_pairs"]
                    if "urban" in (t["a_indicator"] + t["b_indicator"]))
    top_fert = sum(1 for t in real["top_pairs"]
                   if "fertility" in (t["a_indicator"] + t["b_indicator"]))
    lev_sc = real["levels"]["same country, different indicator"]
    lev_si = real["levels"]["same indicator, different country"]
    dif_si = real["differenced"]["same indicator, different country"]

    # The spine, asserted rather than trusted. Any of these failing means a
    # sentence below has become false and the post must not publish.
    if not p_small < p_big:
        raise AssertionError("the rejection rate is supposed to rise with T")
    if max(scaled) - min(scaled) > 0.05:
        raise AssertionError(
            f"|t|/sqrt(T) moved by {max(scaled) - min(scaled):.3f}; the control "
            "the post rests on has failed and finding 2 is now wrong")
    if c_big["median_abs_r"] < 0.90:
        raise AssertionError("the headline correlation claim needs a high median")
    if mis_last["size_using_df_table"] < 0.10:
        raise AssertionError("the wrong-table distortion is supposed to persist")
    if abs(h["r"]) < 0.99:
        raise AssertionError("the named pair is no longer extreme")
    if h["eg_rejects"]:
        raise AssertionError(
            "the named pair now passes the cointegration test; section 4's point "
            "that the right tool catches it no longer holds")
    if lev_u["median_abs_r"] < 0.6:
        raise AssertionError("the real panel no longer shows the artefact")
    if dif_u["median_abs_r"] > 0.25:
        raise AssertionError("differencing is supposed to remove most of it")
    if tests["eg_with_eg_table"] < 0.08:
        raise AssertionError(
            "the limitation section says the correct test still over-rejects on "
            "real data; it no longer does, so that section is wrong")

    sections = [
        Section(heading="The question that assumes its own answer", body=f"""
Someone shows you two series that move together and asks what explains it. The
question has already smuggled in its premise: that the co-movement is a fact
about the world rather than a fact about the arithmetic of trending numbers.

Here is a pair from the World Bank's development indicators. Life expectancy in
Bhutan, {h['a_first']:.0f} years in {real['window'][0]} and {h['a_last']:.0f}
years in {real['window'][1]}. The share of Belgium's population living in cities,
{h['b_first']:.0f}% and {h['b_last']:.0f}% over the same years. Regress one on
the other across those {real['n_years']} annual observations and you get
R² = {h['r2']:.4f} and a t-statistic of **{h['t']:.0f}**. Not 2.3, which is what a
real but modest relationship looks like. Two hundred and thirty-three.

There is no mechanism. Bhutanese longevity does not urbanise Belgium and Belgian
cities do not extend Bhutanese lives, and no third thing plausibly does both. The
correlation is {h['r']:.4f} anyway.

**The claim of this post is that this pair is unremarkable, and that its
unremarkableness is measurable.** Among the {real['n_unrelated']:,} pairs I can
build from four World Bank indicators measured in two different countries,
{_pct(real['thresholds']['0.9']['share'])} correlate above 0.9 and
{_pct(real['thresholds']['0.96']['share'])} above 0.96. If instead the rejection
rate for such pairs came out near the 5% a t-test promises, everything below
would be wrong.
"""),
        Section(heading="1. The test gets worse as the sample grows",
                body=f"""
Start where ground truth exists. I generate two random walks independently — the
true slope between them is zero because I made it zero — regress one on the other
and ask how often the t-test calls the slope significant. A correctly sized test
answers {ALPHA:.0%} at every sample size.

At {small} observations it answers {_pct(p_small)}. At {big} observations, sixty-four
times as much data, it answers {_pct(p_big)}. Give both walks a small upward drift,
which every real macro series has, and {big} observations gives
{_pct(d_big['rejection_rate'])} with a median t-statistic of
{d_big['median_abs_t']:.0f}.

This is the part that trips people, including me: the usual instinct is that a
spurious result is a small-sample problem which more data will dissolve. Here
more data does the opposite. It does not make the estimate more accurate; it
makes the wrong estimate more confident.

It is worth being precise about what is and is not happening. The estimated slope
is not biased — over many replications it averages to zero, which is the truth.
What breaks is the *standard error*. OLS computes it assuming the residuals are
well behaved, and between two random walks the residual is itself a random walk:
enormously persistent, nothing like the independent draws the formula was derived
for. The standard error is therefore far too small, and dividing a correct
estimate by a badly understated uncertainty is what manufactures a t-statistic of
{d_big['median_abs_t']:.0f}. Every ingredient is individually defensible. The
combination is not.

R² behaves the same way. Granger and Newbold pointed out in 1974 that it does not
shrink toward zero — across every sample size I tried, the median R² between two
unrelated walks sits near
{div['plain'][big]['median_r2']:.2f} and the ninetieth percentile near
{div['plain'][big]['p90_r2']:.2f}. A model that explains 60% of the variance of
something it has no relationship with is an ordinary outcome, not a rare one.
""", figures=[figs["f1"]]),
        Section(heading="2. Why: the statistic is not noisy, it is diverging",
                body=f"""
Phillips (1986) worked out what is actually happening. Under a unit root the OLS
t-statistic has no limiting distribution at all. It grows without bound at rate
√T, and the quantity that does converge is t/√T, to a ratio of functionals of
Brownian motion.

That is a sharp, checkable prediction, so I checked it. If Phillips is right,
dividing by √T should freeze the whole distribution, not just its centre. It
does: across a 64-fold change in sample size the 5th percentile, the median and
the 95th percentile of |t|/√T sit at
{div['plain'][big]['scaled_t_p05']:.2f}, {div['plain'][big]['scaled_t_median']:.2f}
and {div['plain'][big]['scaled_t_p95']:.2f}, and the median moves by
{max(scaled) - min(scaled):.3f} in total.

This is the post's control, and it is the one that could have gone the other way.
Had the scaled quantity drifted too, the explanation on offer would be wrong and
the rest of the post would need rewriting. It did not drift.

The practical translation: a large t-statistic between two trending series
carries almost no information about whether they are related. It is mostly a
readout of how many observations you have.
""", figures=[figs["f2"], figs["f3"]]),
        Section(heading="3. The same thing, in the World Bank's numbers",
                body=f"""
Simulated data proves the mechanism; it does not prove that real published series
behave like that. So: four indicators with long, complete coverage — fertility
rate, life expectancy, urban population share, CO₂ per capita — for
{real['n_countries']} countries over {real['window'][0]}–{real['window'][1]}.
That is {real['n_series']} series, and {real['n_unrelated']:,} pairs in which the
two series come from **different countries and different indicators**. Whatever
connects Papua New Guinean fertility to Nigerian urbanisation, it is not a
mechanism anyone would defend in a seminar.

Across those pairs the median |r| is {lev_u['median_abs_r']:.3f}. A naive t-test
calls {_pct(tests['naive_significant'])} of a
{tests['n']:,}-pair subsample significant, with a median |t| of
{tests['median_abs_t']:.1f}. That is worse than the simulation predicted at this
sample length, because real development indicators trend far more decisively than
a random walk with a small drift does.

Now the control. Take the same pairs and correlate year-on-year *changes* rather
than levels. The median falls from {lev_u['median_abs_r']:.3f} to
{dif_u['median_abs_r']:.3f}, and the share above 0.9 falls from
{_pct(lev_u['p_over_90'])} to {dif_u['p_over_90']:.2%} — from
{lev_u['p_over_90'] * lev_u['n_pairs']:,.0f} pairs to
{dif_u['p_over_90'] * dif_u['n_pairs']:,.0f}. The relationship was the trend and
nothing else.

The top of the list is not evenly spread across the four indicators, either.
Every one of the twelve most-correlated unrelated pairs involves the urban
population share, and {top_fert} of them involve the fertility rate. Those are the two
indicators that move most monotonically over these fifty-three years — urban
share almost never falls, fertility almost never rises — so they generate the
strongest artefacts. That is a useful diagnostic in itself: the more
mechanically monotone a series is, the less its correlation with anything is
worth.

One result I did not expect and cannot explain away. Two indicators measured in
the **same** country share a cause — that country's development — and ought to be
more correlated than two indicators from different countries. In levels they are
not: {lev_sc['median_abs_r']:.3f} against {lev_u['median_abs_r']:.3f}, a gap of
{abs(lev_sc['median_abs_r'] - lev_u['median_abs_r']):.3f}. I would not push that
gap far — the same-country group has only {lev_sc['n_pairs']:,} pairs and is
restricted to the six combinations of four indicators — but whatever a genuine
shared cause is contributing here, it is not what puts these numbers near one.

Differencing separates them. The only pair type that retains anything is the same
indicator measured in two different countries ({dif_si['median_abs_r']:.3f}
against {dif_u['median_abs_r']:.3f} for the unrelated pairs) — global co-movement
of one quantity, which is a real phenomenon and is also the smallest effect
anywhere in the table.
""" + "\n\n" + md_table(PAIR_TABLE_HEADER, figs["_rows"]["pairtypes"]),
                figures=[figs["f5"]]),
        Section(heading="4. The fix, and the way it is usually broken",
                body=f"""
The remedy has been standard since Engle and Granger (1987): the level regression
is meaningful only if some linear combination of the two series is stationary. Fit
the cointegrating regression, then run a Dickey-Fuller test on its residual.

On Bhutan and Belgium this works. The test statistic is {h['eg_stat']:.2f} against
a 5% critical value of {ns.MACKINNON_EG[0.05]:.2f}: it does not reject, which is
to say it declines to call the pair cointegrated despite R² = {h['r2']:.4f}. The
right tool catches what the correlation missed.

But there is a trap in which table you read. Because OLS chose the combination
that makes the residual look as stationary as it possibly can, the null
distribution of that statistic sits further left than the ordinary Dickey-Fuller
one. Reading the Dickey-Fuller critical value here is not conservative. It turns
a nominal {ALPHA:.0%} test into a {_pct(mis_worst)} one, and unlike a
small-sample problem this does not fade: at T={CV_SIZES[-1]:,} it is still
{_pct(mis_last['size_using_df_table'])}.

I did not take the critical values from a table. The module simulates both null
distributions, which means they can be checked against MacKinnon's published
response surface and against statsmodels — two answers computed elsewhere, by
other people, a different way. At T={CV_SIZES[-1]:,} all four of mine agree with
MacKinnon's to within about a hundredth, which is the Monte Carlo error at
{CV_REPS:,} replications.
""" + "\n\n" + md_table(CV_TABLE_HEADER, figs["_rows"]["critical"]),
                figures=[figs["f4"]]),
        Section(heading="Where this breaks", body=f"""
**The correct test still over-rejects on real data.** On simulated random walks
the Engle-Granger test with the right critical value is properly sized —
{mis_last['size_using_eg_table']:.1%} at T={CV_SIZES[-1]:,}, against a nominal
{ALPHA:.0%}. On the real World Bank pairs it rejects
{_pct(tests['eg_with_eg_table'])} of the time, and these pairs are drawn from
countries chosen not to be related. Cointegration is a large improvement on
{_pct(tests['naive_significant'])}, not a solution. My best guess is that these
indicators are not random walks at all — a demographic transition is a smooth,
almost deterministic S-curve, and my test regression uses no lags and models no
structural break. I have not established that, so treat it as a hypothesis.

**The named pair is one of those.** Bhutan's life expectancy and Belgium's
urbanisation still correlate at {h['r_differenced']:.2f} after differencing, well
above the {dif_u['median_abs_r']:.3f} median. Both are unusually smooth, so
differencing does not whiten them. It is an honest illustration of a spurious
correlation and a poor illustration of the specific random-walk mechanism in
sections 1 and 2.

**The panel is not a random sample of anything.** Four indicators survived a
coverage filter applied before any correlation was computed; four more were
downloaded and dropped, R&D spending because no country has complete data over
1996–2022 at all. Demographic indicators may be unusually trend-dominated. I
would not carry {_pct(real['thresholds']['0.9']['share'])} to a different corner
of the data world.

**Six countries were removed** — {', '.join(real['dropped_constant'])} — because
at least one of their series never moves. Singapore is 100% urban in every year on
record, and correlation against a flat line is undefined.

**Differencing is not free advice.** If two series really are cointegrated,
differencing throws away the long-run relationship that was the interesting part.
The right answer is an error-correction model, not a reflex `diff()`. That is the
next post.
"""),
        Section(heading="What to do on Monday", body=f"""
Three things, in order of how much they cost you.

Plot the two series in changes as well as levels, always. It costs one line, and
on this panel it takes the share of unrelated pairs that look strongly related
from {_pct(lev_u['p_over_90'])} down to {dif_u['p_over_90']:.2%}.

Distrust any t-statistic on two trending series, in proportion to how large it is.
{h['t']:.0f} is not stronger evidence than 3; on non-stationary data it is mostly
evidence that T is large.

If you must work in levels, test for cointegration, and read the Engle-Granger
critical values rather than the Dickey-Fuller ones — the gap is about half a
point and it triples your false-positive rate.

There is also something worth saying about how this post was built, because it
generalises further than the econometrics does. I did not look up the critical
values. I simulated both null distributions, which cost about two minutes of
compute and bought something a lookup cannot: the numbers could then be compared
against MacKinnon's, and against statsmodels, and if my Dickey-Fuller regression
had been subtly wrong — a misaligned lag, a constant in the wrong place — the
comparison would have said so immediately. Almost every quantitative claim has
some by-product with a published answer sitting next to it. Computing that
by-product on purpose, when you did not need it, is the cheapest error-detection
available.

And the general form, which is the reason I wrote this: a statistic that grows
with your sample size is not measuring your world, it is measuring your sample.
Whenever a number gets more impressive as the data gets longer, that is the first
thing to rule out. The next post takes the other half of this — what to do when
two series *are* cointegrated, and why differencing then throws away the thing you
came for.
"""),
    ]

    post = Post(
        date=POST_DATE,
        title="Bhutan's Life Expectancy Tracks Belgium's Urban Population at 0.9995",
        slug="bhutan-belgium-no-shared-cause",
        subtitle=("Two series with no mechanism between them, R² of 0.9991 and a "
                  "t-statistic of 233. The interesting fact is not that this pair "
                  "exists — it is how many others do."),
        summary=(
            f"Life expectancy in Bhutan and the urban share of Belgium's "
            f"population correlate at **{h['r']:.4f}** over "
            f"{real['n_years']} years, with a t-statistic of **{h['t']:.0f}**. "
            f"Nothing connects them. Among the {real['n_unrelated']:,} World Bank "
            f"pairs I built from two different countries and two different "
            f"indicators, **{_pct(real['thresholds']['0.9']['share'])} correlate "
            f"above 0.9** and a naive t-test calls "
            f"{_pct(tests['naive_significant'])} of them significant. Simulation "
            f"says why: between independent random walks the rejection rate climbs "
            f"from {_pct(p_small)} at T={small} to {_pct(p_big)} at T={big}, because "
            f"the t-statistic diverges at rate √T rather than converging at all — "
            f"|t|/√T is frozen to within {max(scaled) - min(scaled):.3f} across "
            f"that whole range. Differencing the same real pairs drops the median "
            f"correlation from {lev_u['median_abs_r']:.3f} to "
            f"{dif_u['median_abs_r']:.3f}. The standard remedy works on the named "
            f"pair, and is routinely run against the wrong critical value, which "
            f"turns a 5% test into a {_pct(mis_worst)} one."),
        tags=["econometrics", "spurious regression", "cointegration",
              "time series", "public data"],
        data_sources=SOURCES + [
            "World Bank, World Development Indicators (CC BY 4.0): SP.DYN.TFRT.IN, "
            "SP.DYN.LE00.IN, SP.URB.TOTL.IN.ZS, EN.GHG.CO2.PC.CE.AR5. Bulk "
            "per-indicator CSV downloads, retrieved 2026-08-25. Values are "
            "redistributable with attribution.",
        ],
        licence_warnings=[],
        sections=sections,
        # ORDER MATTERS: `_substitute_tables` pairs markdown tables with these
        # images positionally, in document order. The pair-types table appears in
        # section 3 and the critical-values table in section 4, so t2 comes first.
        # Declaring them the other way round silently swaps two images between
        # two sections and nothing in the audit notices.
        table_figures=[figs["t2"], figs["t1"]],
        reproducibility={
            "seed": SEED,
            "replications_per_sample_size": REPS,
            "replications_per_critical_value": CV_REPS,
            "panel": f"{real['n_series']} series, {real['n_countries']} countries, "
                     f"{real['window'][0]}-{real['window'][1]}",
            "unrelated_pairs": real["n_unrelated"],
            "indicators_dropped_for_coverage": DROPPED_INDICATORS,
            "countries_dropped_constant": real["dropped_constant"],
        },
        min_words=1800,
        max_words=2800,
    )
    post.hero = figs["hero"]

    # Verify the substitution actually lands each table image under the heading
    # whose numbers it carries. This is checked rather than trusted because the
    # pairing is positional: get the order wrong and the post publishes with two
    # images swapped, which no word count or link check would catch.
    _check_table_placement(post)
    return post


def _check_table_placement(post: Post) -> None:
    import re
    from standarderror.render import publish

    was_draft = post.draft
    post.draft = False
    try:
        body = publish.medium_bundle(
            post, out_dir=se.SETTINGS.build_dir / "_placement_check").read_text()
    finally:
        post.draft = was_draft

    heading, seen = "", {}
    for line in body.split("\n"):
        if line.startswith("## "):
            heading = line[3:].strip()
        m = re.search(r"!\[[^\]]*\]\(([^)]+)\)", line)
        if m:
            seen[m.group(1).rsplit("/", 1)[-1]] = heading

    expect = {f"a10-t2-pairtypes.{EXT}": "World Bank",
              f"a10-t1-critical.{EXT}": "fix"}
    for name, needle in expect.items():
        where = seen.get(name)
        if where is None:
            raise AssertionError(f"{name} never made it into the rendered body")
        if needle.lower() not in where.lower():
            raise AssertionError(
                f"{name} was substituted under '{where}', which is not the "
                f"section it belongs to. table_figures is matched to markdown "
                f"tables positionally — check its order.")


if __name__ == "__main__":
    import sys
    res = compute(force="--force" in sys.argv)
    d = res["divergence"]
    print("\nrejection rate at nominal 5%")
    for n in SIZES:
        p, dr = d["plain"][str(n)], d["drifting"][str(n)]
        print(f"  T={n:>5}  plain {p['rejection_rate']:>6.1%} "
              f"(|t|~{p['median_abs_t']:>6.2f}, |t|/sqrtT {p['scaled_t_median']:.3f})"
              f"   drifting {dr['rejection_rate']:>6.1%} "
              f"(|t|~{dr['median_abs_t']:>7.2f})")
    print("\nmedian |r|")
    for n in SIZES:
        print(f"  T={n:>5}  plain {res['correlation']['plain'][str(n)]['median_abs_r']:.3f}"
              f"   drifting {res['correlation']['drifting'][str(n)]['median_abs_r']:.3f}"
              f"   P(|r|>0.96 | drift) "
              f"{res['correlation']['drifting'][str(n)]['p_abs_r_over_0.96']:.1%}")
    print("\ncritical values (5%)")
    for n in CV_SIZES:
        print(f"  T={n:>5}  DF {res['critical_values']['df'][str(n)]['0.05']:+.3f}"
              f"   EG {res['critical_values']['eg'][str(n)]['0.05']:+.3f}")
    print(f"  MacKinnon  DF {ns.MACKINNON_DF[0.05]:+.4f}   EG {ns.MACKINNON_EG[0.05]:+.4f}")
    print("\nsize of a nominal 5% Engle-Granger test")
    for n in CV_SIZES:
        m = res["misuse"][str(n)]
        print(f"  T={n:>5}  with the DF table {m['size_using_df_table']:>6.1%}"
              f"   with the EG table {m['size_using_eg_table']:>6.1%}")
