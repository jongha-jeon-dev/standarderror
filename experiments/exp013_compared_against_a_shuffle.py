"""exp013 — a survey of diffusion models in finance, and what its tables need.

Requested directly: write up Wang and Ventre, "Diffusion Models in Finance: A
Survey" (arXiv:2608.12583, 12 August 2026).

A survey summary alone would be thin, so this follows the shape of exp012: report
the survey properly, then add the one thing a survey cannot — run the experiment its
subject matter is usually evaluated by, on a process whose answer is known, with the
baselines that are almost always missing.

The claim, which is about evaluation practice and not about the model class
----------------------------------------------------------------------------
Generative models of returns are judged on a stylised-facts table: fat tails, no
autocorrelation in returns, autocorrelation in absolute returns, sometimes a
leverage effect. Two baselines belong in that table and are rarely in it.

* An **i.i.d. bootstrap** of the training returns reproduces the marginal
  distribution *exactly* — kurtosis included — and destroys every dependence. So a
  matched kurtosis is worth zero evidence on its own. Fat tails are free.
* A **moving-block bootstrap** keeps dependence up to the block length, has one
  parameter and no training. It is the actual bar for "the model learned the
  temporal structure".

Neither is a rival to a diffusion model in capability: a bootstrap cannot be
conditioned on anything, cannot produce a value it has not already seen, and cannot
extrapolate. That is the point. They mark the part of a stylised-facts score that
carries no information about the model, and until they are in the table there is no
way to know how much of the score that is.

What is measured here
---------------------
Ground truth is a GARCH(1,1) path with Student-t shocks, so every fact has a known
value. The generators produce 64-step windows; every row of the comparison is
scored on the same 600 paths of the same length, because the pooled kurtosis
estimate is itself sample-size dependent and comparing a generator's 100,000 samples
to the data's 5,000 is not comparing like with like.

Three things this post found while doing it, each of which is in the body:

1. **The window has to contain the fact.** At the usual equity-index persistence of
   0.98 the lag-1 autocorrelation of absolute returns *inside* a short window is
   almost zero — the variance moves too slowly for consecutive absolute returns to
   covary over 32 steps. A model asked to reproduce clustering there has been handed
   nothing to reproduce, and its failure is the experiment's fault. Persistence 0.85
   at a 64-step window puts the true value at +0.16, which is a fact that exists.
   Fig 1 is that check, with a shuffled series as the reference rather than zero,
   because the lag-1 estimator is biased downwards in short windows.
2. **The terminal SNR bug that the standard deviation hides.** A 200-step linear
   schedule leaves alpha_bar at 0.13, so sampling from a standard normal starts from
   a distribution the model never saw. With standardised data the generated standard
   deviation still comes out right, and the error lands on the higher moments.
   Caught by feeding the analytic Gaussian denoiser through the sampler: it returned
   2.22 for data with standard deviation 2.5.
3. **The control refuted the simple version of the story.** The i.i.d. bootstrap
   matches the kurtosis and misses the clustering, the diffusion model does the
   reverse, and neither dominates — which is a worse outcome for the *table* than
   for either generator, because a single "realism" score would have ranked them and
   the ranking would have been an artefact of the weighting.
4. **And then the kurtosis row turned out not to be measurable at this sample
   size.** The process's excess kurtosis is 20.3. Estimated from the 600 windows
   every row of the table uses, that estimate has a standard deviation of about 6 —
   30% of the value. The shuffle, which is exactly right in expectation because it
   *is* the marginal distribution, drew 12.5; the process's own reported draw was
   27.7. Both are the same quantity, and the gap between them is a coin flip. Even
   the sampling seed does it: the same trained model, same weights, sampled twice,
   gave 1.67 and 2.20. So the fat-tails row is free in a second sense — too noisy to
   fail.

The training-budget ladder is here so that "it was undertrained" is a question with
an answer. Four rungs spanning about 250 times the training compute — a span bought
by adding a rung at the *bottom*, because the machine could not afford one at the top
and a factor is a factor whichever end it comes from.

No market data, no company, no investment implication. The only inputs are the
survey's published description of itself and a simulated path.

Run: `standarderror run exp013_compared_against_a_shuffle --publish`
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from datetime import date

import numpy as np

import standarderror as se
from standarderror.dynamics import sde
from standarderror.generative import diffusion, stylised
from standarderror.render import Post
from standarderror.viz import charts, theme

#: Pinned so a rebuild cannot silently re-date a published post.
#: `Post.date` defaults to today, which is correct exactly once.
POST_DATE = date(2026, 8, 19)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed

# --- the survey, as it describes itself ----------------------------------------
SURVEY = "Zhuohan Wang and Carmine Ventre, arXiv:2608.12583"
SURVEY_DATE = "12 August 2026"
SURVEY_REPO = "github.com/ZhuoHan1998/Diffusion-Models-In-Finance"
DATA_TYPES = ("time series", "limit order books", "tabular data",
              "other structured objects")
APPEALS = ("stable likelihood-based training",
           "strong mode coverage",
           "flexible conditioning",
           "an SDE formulation that lines up with Ito calculus")

# --- ground truth --------------------------------------------------------------
GARCH = dict(n=120_000, omega=0.02, arch=0.25, beta=0.60, df=5.0, seed=5)
PERSISTENCE = GARCH["arch"] + GARCH["beta"]
EQUITY_PERSISTENCE = 0.98        # the usual index estimate, and the trap
WINDOW = 64
STRIDE = 8
WINDOW_SWEEP = (8, 16, 32, 64, 128, 256)
N_EVAL = 600                     # paths scored, identical for every generator
N_BOOT = 400                     # resamples for the pooled standard errors
BLOCK = 16
BLOCK_SWEEP = (2, 4, 8, 16, 32)
STEPS = 1000                     # schedule length; see Schedule.terminal_snr
SHORT_STEPS = 200                # the truncation that looks harmless

#: Width, passes, noisings per window. Spans two and a half orders of magnitude of
#: compute, and the span was bought at the *bottom* rather than the top: a fifth rung
#: above the largest one here costs an hour of wall clock on two CPUs, while a rung
#: below the smallest costs ten seconds and widens the axis by the same factor.
LADDER = (
    dict(hidden=64, max_iter=25, noise_per_window=2),
    dict(hidden=128, max_iter=40, noise_per_window=3),
    dict(hidden=256, max_iter=80, noise_per_window=6),
    dict(hidden=384, max_iter=110, noise_per_window=8),
)
HEADLINE_RUNG = -1               # the largest budget is the one reported in text

REPORTED = ("excess_kurtosis", "acf1_abs", "acf1_returns", "leverage", "sd")
#: Facts whose standard errors are trustworthy enough to divide by. `excess_kurtosis`
#: is deliberately absent: its bootstrap error is about a third of the spread of the
#: estimate across independent draws, so a z-score built on it would overstate every
#: verdict in the column. It gets its own figure instead.
Z_KEYS = ("acf1_abs", "acf1_returns", "leverage", "sd")
CACHE = se.SETTINGS.build_dir / "cache" / "exp013.json"


def _config_key() -> str:
    """Hash of everything the computation depends on, so the cache cannot go stale."""
    blob = json.dumps({"v": 2, "garch": GARCH, "window": WINDOW, "stride": STRIDE,
                       "n_eval": N_EVAL, "n_boot": N_BOOT, "block": BLOCK,
                       "block_sweep": BLOCK_SWEEP, "steps": STEPS,
                       "short_steps": SHORT_STEPS, "ladder": LADDER,
                       "sweep": WINDOW_SWEEP, "seed": SEED}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def truth() -> np.ndarray:
    """The GARCH path every generator is trained on and compared against."""
    return np.asarray(sde.garch11(**GARCH).data["r"], dtype=float)


def _rung(windows: np.ndarray, i: int, say) -> dict:
    """One training budget, fitted and scored, cached on its own key."""
    cfg = dict(LADDER[i])
    path = (CACHE.parent /
            f"exp013_rung_{hashlib.sha256(json.dumps([cfg, WINDOW, STEPS, STRIDE, GARCH, N_EVAL, N_BOOT], sort_keys=True).encode()).hexdigest()[:12]}.json")
    if path.exists():
        say(f"rung {i} from cache")
        return json.loads(path.read_text())

    model = diffusion.DDPM.budget(length=WINDOW, steps=STEPS, seed=SEED, **cfg)
    cost = model.flops_proxy(len(windows))
    t1 = time.time()
    model.fit(windows, rng=np.random.default_rng(SEED + 10 + i))
    fit_s = time.time() - t1
    samples = model.sample(N_EVAL, rng=np.random.default_rng(SEED + 50 + i))
    facts = stylised.stylised_facts(samples, n_boot=N_BOOT, seed=SEED)
    out = {"config": cfg, "cost": cost, "fit_seconds": fit_s, "loss": model.loss,
           "rows": int(model.n_train_rows), "facts": facts}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    say(f"rung {i} cost {cost:.0f} fit {fit_s:.0f}s loss {model.loss:.4f} "
        f"kurt {facts['excess_kurtosis']['value']:.2f} "
        f"acf1|r| {facts['acf1_abs']['value']:+.3f}")
    return out


def compute(*, force: bool = False, verbose: bool = True) -> dict:
    """Train the ladder, sample every generator, score the battery.

    Cached to `build/cache/exp013.json` under a hash of the configuration, and each
    rung cached separately besides, because the largest one takes twenty-odd minutes
    on two CPUs and a post should be re-renderable in seconds. `force=True` or a
    changed configuration recomputes.
    """
    key = _config_key()
    if not force and CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            return cached

    t0 = time.time()

    def say(*a):
        if verbose:
            print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

    r = truth()
    rng = np.random.default_rng(SEED)

    # 1. Does the fact exist inside the window? Ask before training anything.
    clustering = {
        "chosen": stylised.within_window_clustering(r, WINDOW_SWEEP, stride=STRIDE),
        "equity": stylised.within_window_clustering(
            np.asarray(sde.garch11(
                **{**GARCH, "arch": 0.10, "beta": EQUITY_PERSISTENCE - 0.10}
            ).data["r"], dtype=float), WINDOW_SWEEP, stride=STRIDE),
    }
    say("within-window clustering", {k: round(v, 3)
                                     for k, v in clustering["chosen"].items()})

    # 2. Two references, and the gap between them turned out to be a finding.
    #    `population` uses all 14,993 windows; `reference` uses the N_EVAL draw that
    #    every generator is matched against, which is what a paper's "data" column
    #    actually is. They differ by more than they have any business differing by.
    proto = diffusion.DDPM.budget(length=WINDOW, steps=STEPS, **LADDER[0])
    windows = proto.windows(r, stride=STRIDE)
    population = stylised.stylised_facts(windows, n_boot=N_BOOT, seed=SEED)
    pick = rng.choice(len(windows), N_EVAL, replace=False)
    reference = stylised.stylised_facts(windows[pick], n_boot=N_BOOT, seed=SEED)
    say("population", {k: round(v["value"], 4) for k, v in population.items()})
    say("reference ", {k: round(v["value"], 4) for k, v in reference.items()})

    # How much of that gap is just the estimator? Draw the same statistic many
    # times, contiguously and i.i.d., and look at the spread. Cheap, and it is the
    # error bar the bootstrap column does not give you.
    def draw_spread(contiguous: bool, n_draws: int = 400) -> list[float]:
        g = np.random.default_rng(SEED + 7)
        out = []
        for _ in range(n_draws):
            if contiguous:
                f = windows[g.choice(len(windows), N_EVAL, replace=False)].ravel()
            else:
                f = g.choice(r, N_EVAL * WINDOW, replace=True)
            c = f - f.mean()
            out.append(float((c ** 4).mean() / c.var() ** 2 - 3.0))
        return out

    spread = {"contiguous": draw_spread(True), "iid": draw_spread(False),
              "series_excess_kurtosis": float(
                  ((r - r.mean()) ** 4).mean() / r.var() ** 2 - 3.0)}
    say("kurtosis estimator spread: contiguous sd "
        f"{np.std(spread['contiguous']):.2f}, iid sd {np.std(spread['iid']):.2f}, "
        f"series value {spread['series_excess_kurtosis']:.2f}")

    rows = {"the process itself": reference}

    # 3. The two baselines that cost nothing.
    rows["shuffle the returns"] = stylised.stylised_facts(
        stylised.iid_bootstrap(r, N_EVAL, WINDOW, seed=SEED + 1),
        n_boot=N_BOOT, seed=SEED)
    rows[f"blocks of {BLOCK}"] = stylised.stylised_facts(
        stylised.block_bootstrap(r, N_EVAL, WINDOW, block=BLOCK, seed=SEED + 2),
        n_boot=N_BOOT, seed=SEED)
    block_sweep = {
        b: stylised.stylised_facts(stylised.block_bootstrap(
            r, N_EVAL, WINDOW, block=b, seed=SEED + 2))
        for b in BLOCK_SWEEP}
    say("baselines done")

    # 4. The ladder, one rung per fit, each cached on its own. Per-rung caching is
    # not tidiness: the first version of this cached only at the end, and killing a
    # run that was three rungs deep because the fourth was going to take an hour
    # threw away all three.
    ladder = [_rung(windows, i, say) for i in range(len(LADDER))]
    rows["diffusion model"] = ladder[HEADLINE_RUNG]["facts"]

    # 5. The sampler check against an answer rather than against itself.
    oracle = {}
    for steps in (SHORT_STEPS, STEPS):
        sch = diffusion.linear_schedule(steps)
        m = diffusion.DDPM(length=8, schedule=sch,
                           denoiser=diffusion.GaussianOracle(sch, sd=2.5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = m.sample(4000, scale=1.0, rng=np.random.default_rng(1))
        oracle[steps] = {"terminal_snr": sch.terminal_snr,
                         "recovered_sd": float(x.std())}
    say("oracle", oracle)

    out = {
        "key": key,
        "clustering": {k: {str(a): b for a, b in v.items()}
                       for k, v in clustering.items()},
        "n_windows": int(len(windows)),
        "population": population,
        "spread": spread,
        "rows": {k: v for k, v in rows.items()},
        "block_sweep": {str(k): v for k, v in block_sweep.items()},
        "ladder": ladder,
        "oracle": {str(k): v for k, v in oracle.items()},
        "elapsed_seconds": time.time() - t0,
        "true_sd": float(r.std()),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=1))
    say("cached to", CACHE)
    return out


SCHEDULE_RUNG = 2                # the budget the schedule study is run at
SCHEDULES = ("1,000 steps, textbook endpoints",
             "200 steps, endpoints kept",
             "200 steps, endpoints steepened")
SCHEDULE_CACHE = se.SETTINGS.build_dir / "cache" / "exp013_schedules.json"


def _schedules() -> dict:
    """Three forward processes: one correct and slow, one broken, one correct and fast."""
    return {
        SCHEDULES[0]: diffusion.linear_schedule(STEPS),
        SCHEDULES[1]: diffusion.linear_schedule(SHORT_STEPS),
        SCHEDULES[2]: diffusion.schedule_for_snr(SHORT_STEPS, target_snr=1e-3),
    }


def compute_schedules(*, force: bool = False, verbose: bool = True) -> dict:
    """Same budget, same data, same seed — three noise schedules.

    Cached separately from `compute` so that adding this study did not invalidate a
    forty-minute ladder. Run at a middle rung rather than the largest, because what
    is being measured is the *spread* across a hyperparameter and three fits at the
    top of the ladder would have cost an hour to say the same thing.
    """
    key = _config_key() + f"-sched{SCHEDULE_RUNG}"
    if not force and SCHEDULE_CACHE.exists():
        cached = json.loads(SCHEDULE_CACHE.read_text())
        if cached.get("key") == key:
            return cached

    t0 = time.time()
    r = truth()
    cfg = dict(LADDER[SCHEDULE_RUNG])
    proto = diffusion.DDPM.budget(length=WINDOW, steps=STEPS, **cfg)
    windows = proto.windows(r, stride=STRIDE)
    rng = np.random.default_rng(SEED)
    reference = stylised.stylised_facts(
        windows[rng.choice(len(windows), N_EVAL, replace=False)],
        n_boot=N_BOOT, seed=SEED)

    out = {"key": key, "budget": cfg, "reference": reference, "runs": {}}
    for name, sch in _schedules().items():
        model = diffusion.DDPM.budget(length=WINDOW, schedule=sch, seed=SEED, **cfg)
        model.fit(windows, rng=np.random.default_rng(SEED + 10 + SCHEDULE_RUNG))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the terminal-SNR warning is the point
            samples = model.sample(N_EVAL, rng=np.random.default_rng(SEED + 77))
        facts = stylised.stylised_facts(samples, n_boot=N_BOOT, seed=SEED)
        out["runs"][name] = {
            "steps": sch.steps, "beta_end": float(sch.betas[-1]),
            "terminal_snr": sch.terminal_snr, "loss": model.loss, "facts": facts}
        if verbose:
            print(f"[{time.time() - t0:7.1f}s] {name}: snr "
                  f"{sch.terminal_snr:.2e} kurt "
                  f"{facts['excess_kurtosis']['value']:6.2f} acf1|r| "
                  f"{facts['acf1_abs']['value']:+.3f} sd "
                  f"{facts['sd']['value']:.4f}", flush=True)
    out["elapsed_seconds"] = time.time() - t0
    SCHEDULE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_CACHE.write_text(json.dumps(out, indent=1))
    return out


def trend(values) -> str:
    """Describe a ladder series without asserting a shape it may not have.

    Written because the first draft's caption claimed both curves rose
    monotonically, and one of them does not: the two smallest budgets produce
    excess kurtosis of 0.5 and 0.3 against a true 27.7, which are the same answer
    and happen to be in the wrong order.
    """
    v = [float(x) for x in values]
    ups = sum(b > a for a, b in zip(v, v[1:]))
    if ups == len(v) - 1:
        return "rises at every step"
    if v[-1] > v[0] and ups >= len(v) - 2:
        return "rises across the range with one step out of order"
    return "does not rise consistently"


def z_errors(res: dict, keys: tuple[str, ...] = Z_KEYS) -> dict:
    """Every generator's distance from the truth, in its own standard errors.

    The common-unit problem, solved the only way it can be: not by rescaling a
    kurtosis into a correlation, but by dividing each error by the sampling error of
    that same fact at the same sample size. That makes the rows comparable *across
    generators*, which is what the figure needs.

    Two things it is not. It is not a pass mark — at a large enough sample every
    generator fails every row, which is a fact about sample size and not about
    generators. And the errors here are bootstrap errors over paths, which for the
    kurtosis column are about a third of the spread of the estimate across
    independent draws (`res["spread"]`). Read that column's magnitudes as upper
    bounds.

    The comparison is against the population value over all windows, not against the
    `N_EVAL` draw the table reports, because the draw is itself noisy enough to move
    the answer.
    """
    ref = res["population"]
    out = {}
    for name, facts in res["rows"].items():
        if name == "the process itself":
            continue
        row = {}
        for k in keys:
            se = np.sqrt(ref[k]["se"] ** 2 + facts[k]["se"] ** 2)
            row[k] = float((facts[k]["value"] - ref[k]["value"]) / se)
        out[name] = row
    return out


FACT_LABELS = {"excess_kurtosis": "excess\nkurtosis", "acf1_abs": "ACF1 of |r|\n(clustering)",
               "acf1_returns": "ACF1 of r", "leverage": "leverage", "sd": "sd"}
ROW_ORDER = ("shuffle the returns", f"blocks of {BLOCK}", "diffusion model")
# Kept short on purpose: `theme.finish` sizes the figure to fit the source note on
# one line, so a long note stretches every chart and leaves a band of empty page.
SRC = (f"Simulated GARCH(1,1), t({GARCH['df']:.0f}) shocks, persistence "
       f"{PERSISTENCE:g}, seed {GARCH['seed']}. {N_EVAL} paths of {WINDOW} steps.")


def _fmt(entry: dict, digits: int = 2) -> str:
    if not np.isfinite(entry["value"]):
        return "undefined"
    if np.isfinite(entry["se"]):
        return f"{entry['value']:.{digits}f} ± {entry['se']:.{digits}f}"
    return f"{entry['value']:.{digits}f}"


BATTERY_HEADER = ["generator", "excess kurtosis", "ACF1 of |r|", "ACF1 of r",
                  "leverage", "sd"]
def md_table(header: list[str], rows: list[list[str]]) -> str:
    """A markdown table with pipes inside cells escaped.

    Not cosmetic. A cell reading `ACF1 of |r|` splits into three fields, the header
    stops matching the separator row, and Goldmark quietly renders the entire block
    as a paragraph of pipe characters — a table that passes every check except
    looking at the rendered page. `Post.audit` now catches the column-count mismatch
    too; this is the fix rather than the alarm.
    """
    def cell(x):
        return str(x).replace("|", r"\|")
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def battery_rows(res: dict) -> list[list[str]]:
    """The whole table, formatted once for the image and the markdown body.

    Two reference rows rather than one. The population row is the answer; the
    `N_EVAL` row is what a paper's "data" column actually is, and the gap between
    them is large enough on the kurtosis column to be a finding of its own.
    """
    labelled = [(f"the process, all {res['n_windows']:,} windows", res["population"]),
                (f"the process, the {N_EVAL} drawn here",
                 res["rows"]["the process itself"])]
    labelled += [(name, res["rows"][name]) for name in ROW_ORDER]
    rows = []
    for name, f in labelled:
        rows.append([name, _fmt(f["excess_kurtosis"], 1), _fmt(f["acf1_abs"], 3),
                     _fmt(f["acf1_returns"], 3), _fmt(f["leverage"], 3),
                     _fmt(f["sd"], 3)])
    return rows


SCHEDULE_HEADER = ["forward process", "steps", "final beta", "terminal SNR",
                   "excess kurtosis", "ACF1 of |r|"]


def schedule_rows(sch: dict) -> list[list[str]]:
    rows = []
    for name in SCHEDULES:
        run = sch["runs"][name]
        rows.append([name, f"{run['steps']:,}", f"{run['beta_end']:.3f}",
                     f"{run['terminal_snr']:.1e}",
                     _fmt(run["facts"]["excess_kurtosis"], 1),
                     _fmt(run["facts"]["acf1_abs"], 3)])
    return rows


def figures(res: dict, sch: dict) -> dict:
    import matplotlib.transforms as mtransforms
    import pandas as pd

    figs = {}
    r = truth()

    # F1 — the design check, run before any training. Three curves, and the third
    # is the reason the first two are readable: the lag-1 correlation estimator is
    # biased downwards in short windows, so a zero line is the wrong reference and
    # the shuffled series is the right one. Cheap enough to recompute at render
    # time rather than cache.
    shuffled = np.random.default_rng(SEED).permutation(r)
    curve = pd.DataFrame({
        f"persistence {PERSISTENCE:g} (used here)":
            [res["clustering"]["chosen"][str(L)] for L in WINDOW_SWEEP],
        f"persistence {EQUITY_PERSISTENCE:g} (an equity index)":
            [res["clustering"]["equity"][str(L)] for L in WINDOW_SWEEP],
        "the same values shuffled":
            [stylised.within_window_clustering(shuffled, (L,), stride=STRIDE)[L]
             for L in WINDOW_SWEEP],
    }, index=list(WINDOW_SWEEP))

    def mark_window(_fig, ax):
        m = theme.LIGHT
        ax.axvline(WINDOW, color=m.muted, lw=1.0, ls=(0, (2, 3)))
        ax.annotate(f"the {WINDOW}-step window\ngenerated here", (WINDOW, 0.97),
                    xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=m.muted,
                    va="top", linespacing=1.4)
        from matplotlib.ticker import FixedLocator, ScalarFormatter
        ax.xaxis.set_major_locator(FixedLocator(list(WINDOW_SWEEP)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(FixedLocator([]))

    fig_meta, _ = charts.lines(
        curve, mode="light", logx=True, decorate=mark_window,
        title="A generator can only reproduce what fits inside its window",
        subtitle=("Lag-1 autocorrelation of absolute returns measured inside a "
                  "window, against the window's length, on two GARCH paths and on "
                  "a shuffled copy of one of them."),
        ylabel="clustering visible inside the window",
        xlabel="window length (steps, log scale)", source=SRC,
        alt=("Three curves against window length on a log x-axis. The persistence "
             "0.85 curve rises from below zero at 8 steps to about 0.23 at 256. "
             "The persistence 0.98 curve stays near zero until about 64 steps and "
             "reaches roughly 0.05 at 256. The shuffled series sits slightly below "
             "zero and returns to zero as the window lengthens."),
        caption=(f"Fig 1. The check to run before training anything. At the "
                 f"persistence an equity index usually estimates, "
                 f"{EQUITY_PERSISTENCE:g}, the variance moves too slowly for "
                 f"consecutive absolute returns inside a short window to covary: at "
                 f"32 steps the clustering there is "
                 f"{res['clustering']['equity']['32']:+.3f}, and even at "
                 f"{WINDOW} it is only "
                 f"{res['clustering']['equity'][str(WINDOW)]:+.3f} against the "
                 f"{res['clustering']['equity']['256']:+.3f} a 256-step window "
                 f"shows. A model asked to reproduce the 32-step figure has been "
                 f"handed nothing, and its failure would be the experiment's fault. "
                 f"At persistence {PERSISTENCE:g} the {WINDOW}-step window contains "
                 f"{res['clustering']['chosen'][str(WINDOW)]:+.3f}, which is why "
                 f"this post uses it. The shuffled line is here because zero is not "
                 f"the right reference: the estimator is biased downwards in short "
                 f"windows, which is the whole of the 8-step reading."),
        path=str(IMG / f"a5-f1-window.{EXT}"))
    figs["window"] = fig_meta

    # F2 — every generator against the truth, fact by fact, in the reference's own
    # sampling errors. A matrix rather than a score, because the refusal to add the
    # columns up is the argument.
    z = z_errors(res)
    matrix = [[z[name][k] for k in Z_KEYS] for name in ROW_ORDER]
    fig_meta, _ = charts.coefficient_matrix(
        matrix, row_labels=list(ROW_ORDER),
        col_labels=[FACT_LABELS[k] for k in Z_KEYS], mode="light", annotate=True,
        annotate_threshold=0.0,
        cbar_label="standard errors from the truth",
        title="A moving-block bootstrap fails none of these",
        subtitle=("Each generator's distance from the process it was built from, fact "
                  "by fact, divided by the sampling error of that fact at this sample "
                  "size. Red is above the truth, blue below. Excess kurtosis is absent: "
                  "its error bar is not reliable enough to divide by, and it has its own "
                  "figure."),
        source=SRC,
        alt=("A three-by-four heatmap. The shuffle row is strongly negative on the "
             f"clustering column and small elsewhere. The blocks-of-{BLOCK} row is "
             "within one standard error on every column. The diffusion model row is "
             "strongly negative on clustering and positive on standard deviation."),
        caption=(f"Fig 2. Blocks of {BLOCK} sit within "
                 f"{max(abs(v) for v in z[f'blocks of {BLOCK}'].values()):.1f} standard "
                 f"errors of the truth on every column here — a generator with one "
                 f"parameter, no training and no capacity to be conditioned on "
                 f"anything, indistinguishable from the process on four of the five "
                 f"facts this battery measures. The shuffle fails exactly one column, "
                 f"clustering, by {abs(z['shuffle the returns']['acf1_abs']):.0f} "
                 f"standard errors, because it has no memory whatsoever. The diffusion "
                 f"model misses clustering by "
                 f"{abs(z['diffusion model']['acf1_abs']):.0f} and overshoots the "
                 f"standard deviation by {abs(z['diffusion model']['sd']):.0f}. The "
                 f"columns are not added up, because there is no weighting across four "
                 f"different units that is not a choice."),
        path=str(IMG / f"a5-f2-errors.{EXT}"))
    figs["errors"] = fig_meta

    # F3 — the row's own noise. Two generators can differ by a factor of two on
    # kurtosis while both being right, and this is how much of the table that eats.
    pop_k = res["population"]["excess_kurtosis"]["value"]
    cont = np.asarray(res["spread"]["contiguous"], float)
    iid = np.asarray(res["spread"]["iid"], float)
    grid = np.linspace(min(cont.min(), iid.min()), max(cont.max(), iid.max()), 200)
    from scipy.stats import gaussian_kde
    fig_meta, _ = charts.histogram(
        cont, bins=32, series_label=f"{N_EVAL} windows of the process",
        overlay={f"{N_EVAL * WINDOW:,} i.i.d. draws from the same values":
                 (grid, gaussian_kde(iid)(grid))},
        mark={"truth": pop_k,
              "Table 1": res["rows"]["the process itself"]["excess_kurtosis"]["value"],
              "shuffle": res["rows"]["shuffle the returns"]["excess_kurtosis"]["value"]},
        mode="light",
        title="The fat-tails row cannot tell these generators apart",
        subtitle=(f"Excess kurtosis measured {len(cont)} times over independent "
                  f"draws of the size every row of Table 1 uses. Both sampling "
                  f"schemes estimate the same quantity."),
        xlabel="excess kurtosis of one draw", source=SRC,
        alt=("A histogram of excess kurtosis estimates spanning roughly 8 to 40, "
             "with a smooth density for i.i.d. draws over it and three vertical "
             "marks: the true value, the draw used in the table, and the shuffle's "
             "draw."),
        caption=(f"Fig 3. The generating process has an excess kurtosis of "
                 f"**{pop_k:.1f}**. Estimated from {N_EVAL} windows — the sample "
                 f"size every row of Table 1 uses — that estimate has a standard "
                 f"deviation of {cont.std():.1f}, about "
                 f"{100 * cont.std() / pop_k:.0f}% of the value, and the draw the "
                 f"table happens to report is "
                 f"{res['rows']['the process itself']['excess_kurtosis']['value']:.1f}. "
                 f"The shuffle, which samples the marginal distribution and is "
                 f"therefore exactly right in expectation, drew "
                 f"{res['rows']['shuffle the returns']['excess_kurtosis']['value']:.1f}"
                 f". A table reporting either of those to two decimal places is "
                 f"reporting a coin flip."),
        path=str(IMG / f"a5-f3-noise.{EXT}"))
    figs["noise"] = fig_meta

    # F4 — the budget ladder, so "it was undertrained" is a question with an answer.
    # Plotted as a share of the true value so two facts in different units share one
    # axis. Never a second y-axis.
    # The population value, not the N_EVAL draw: Fig 3 is about how far apart
    # those two are, so using the draw as a denominator here would divide by
    # the very noise the next figure exists to expose.
    ref = res["population"]
    costs = [rung["cost"] for rung in res["ladder"]]
    ladder = pd.DataFrame({
        "excess kurtosis, as a share of the truth":
            [rung["facts"]["excess_kurtosis"]["value"]
             / ref["excess_kurtosis"]["value"] for rung in res["ladder"]],
        "clustering, as a share of the truth":
            [rung["facts"]["acf1_abs"]["value"] / ref["acf1_abs"]["value"]
             for rung in res["ladder"]],
    }, index=costs)
    shuffle_share = (res["rows"]["shuffle the returns"]["excess_kurtosis"]["value"]
                     / ref["excess_kurtosis"]["value"])
    block_share = (res["rows"][f"blocks of {BLOCK}"]["acf1_abs"]["value"]
                   / ref["acf1_abs"]["value"])

    def mark_baselines(_fig, ax):
        m = theme.LIGHT
        ax.axhline(1.0, color=m.ink_secondary, lw=1.2)
        ax.annotate("the truth", (costs[0], 1.0), xytext=(2, -13),
                    textcoords="offset points", fontsize=8.5, color=m.ink_secondary)
        for share, label in ((shuffle_share, "a shuffle, on kurtosis"),
                             (block_share, f"blocks of {BLOCK}, on clustering")):
            ax.axhline(share, color=m.muted, lw=1.0, ls=(0, (2, 3)))
            ax.annotate(label, (costs[-1], share), xytext=(0, -12),
                        textcoords="offset points", ha="right", fontsize=8.5,
                        color=m.muted)

    fig_meta, _ = charts.lines(
        ladder, mode="light", logx=True, decorate=mark_baselines,
        direct_labels=False,
        title="More compute helps, and not nearly enough",
        subtitle=(f"Two facts from the same {len(LADDER)} training runs, each as a "
                  f"fraction of the value the generating process actually has. "
                  f"Budget is rows times passes times parameters."),
        ylabel="fraction of the true value reproduced",
        xlabel="training budget (relative, log scale)", source=SRC,
        alt=(f"Two curves over {len(LADDER)} budget points on a log x-axis, against "
             "a solid line at 1.0 marking the truth and two dashed baseline lines. "
             "Both curves end far higher than they start and both finish well below "
             "1.0."),
        caption=(f"Fig 4. Across {costs[-1] / costs[0]:.0f} times the compute the "
                 f"clustering curve {trend(ladder.iloc[:, 1])} and the kurtosis "
                 f"curve {trend(ladder.iloc[:, 0])} — the two smallest budgets "
                 f"return excess kurtosis of "
                 f"{res['ladder'][0]['facts']['excess_kurtosis']['value']:.1f} and "
                 f"{res['ladder'][1]['facts']['excess_kurtosis']['value']:.1f} "
                 f"against a true {ref['excess_kurtosis']['value']:.1f}, which is "
                 f"the same answer twice and not a decline. That is the answer to "
                 f"'it was undertrained': the trend is real and the endpoint is "
                 f"still a long way short. Extrapolating a log-x trend says nothing "
                 f"reliable about where it stops, so this figure supports 'not "
                 f"enough compute' and not 'this much would do it'. The dashed "
                 f"lines are what the two untrained baselines reach for free."),
        path=str(IMG / f"a5-f4-budget.{EXT}"))
    figs["budget"] = fig_meta

    # T1 — the battery itself.
    fig_meta, _ = charts.table_image(
        battery_rows(res), header=BATTERY_HEADER,
        title="The stylised-facts table, with the baselines left in",
        subtitle=(f"Every generated row scored on {N_EVAL} paths of {WINDOW} steps, "
                  f"because the pooled kurtosis estimate depends on the sample size; "
                  f"the first row uses all the windows of the path. Errors are one "
                  f"standard error."),
        source=SRC, mode="light", bold_cols=(1, 2), align="lrrrrr",
        alt=("A table of five rows against five stylised facts: the GARCH process "
             "over all its windows, the same process over the 600 windows drawn for "
             "this table, an i.i.d. shuffle of its returns, a moving-block bootstrap "
             "with blocks of 16, and the diffusion model at its largest training "
             "budget."),
        caption=("Table 1. Two of these rows cost nothing to produce. The shuffle has "
                 "no memory at all and the block bootstrap has sixteen steps of it; "
                 "neither was fitted to anything, and neither can be conditioned on "
                 "anything, which is what a generative model is actually for. They "
                 "are here to mark how much of a stylised-facts score is available "
                 "without a model. And compare the first two rows: the same process, "
                 "measured over all its windows and over the 600 drawn for this "
                 "table, differs on excess kurtosis by "
                 f"{res['rows']['the process itself']['excess_kurtosis']['value'] - res['population']['excess_kurtosis']['value']:+.1f}"
                 ". Fig 3 is about that."),
        path=str(IMG / f"a5-t1-battery.{EXT}"))
    figs["battery"] = fig_meta

    # T2 — the hyperparameter study, which moved the table further than the budget.
    fig_meta, _ = charts.table_image(
        schedule_rows(sch), header=SCHEDULE_HEADER,
        title="Same model, same data, same budget — three noise schedules",
        subtitle=("The forward process is not part of the model. Terminal SNR is "
                  "how much of the data survives the last forward step; sampling "
                  "starts from pure noise, so it should be near zero."),
        source=SRC, mode="light", bold_cols=(3, 5), align="lrrrrr",
        alt=("A table of three forward processes: 1,000 steps with textbook "
             "endpoints, 200 steps with the same endpoints, and 200 steps with the "
             "endpoints steepened to reach the same terminal SNR, each with the "
             "excess kurtosis and clustering of the resulting samples."),
        caption=("Table 2. The middle row is broken and I can prove it: driving the "
                 "sampler with the analytic best-possible denoiser for data of "
                 "standard deviation 2.5 returns "
                 f"{res['oracle'][str(SHORT_STEPS)]['recovered_sd']:.2f} on that "
                 "schedule and "
                 f"{res['oracle'][str(STEPS)]['recovered_sd']:.2f} on the first. "
                 "Note which row the stylised-facts columns prefer."),
        path=str(IMG / f"a5-t2-schedules.{EXT}"))
    figs["schedules"] = fig_meta

    # HERO — three panels, because the finding is a sequence: here is the process,
    # here is a shuffle of it, here is a trained model. No axes and no values inside
    # the frames; the numbers under them are the measurement.
    def bursty(panel, m):
        panel.set_xlim(0, 10)
        panel.set_ylim(-1, 1)
        rng = np.random.default_rng(4)
        x = np.linspace(0.4, 9.6, 220)
        vol = 0.12 + 0.55 * np.exp(-((x - 6.4) / 0.75) ** 2) \
            + 0.30 * np.exp(-((x - 2.6) / 0.5) ** 2)
        panel.plot(x, vol * rng.standard_normal(x.size), color=m.series[0], lw=1.6)
        panel.axhline(0, color=m.grid, lw=1.0)

    def shuffled_cards(panel, m):
        from matplotlib.patches import FancyBboxPatch
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 6)
        for x0, y0, ang in ((1.2, 1.4, -18), (3.0, 1.9, 7), (5.0, 1.5, -6),
                            (6.8, 2.1, 21)):
            card = FancyBboxPatch((x0, y0), 1.7, 2.4,
                                  boxstyle="round,pad=0.06,rounding_size=0.25",
                                  fc=m.surface, ec=m.ink, lw=2.0)
            card.set_transform(mtransforms.Affine2D().rotate_deg_around(
                x0 + 0.85, y0 + 1.2, ang) + panel.transData)
            panel.add_patch(card)

    def cloud_to_path(panel, m):
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 6)
        rng = np.random.default_rng(11)
        panel.scatter(rng.uniform(0.6, 3.4, 70), rng.uniform(1.4, 4.6, 70),
                      s=9, color=m.muted, alpha=0.8)
        panel.annotate("", xy=(5.1, 3.0), xytext=(3.9, 3.0),
                       arrowprops=dict(arrowstyle="-|>", color=m.ink, lw=2.0))
        x = np.linspace(5.7, 9.4, 160)
        panel.plot(x, 3.0 + 0.55 * np.sin(x * 2.3) * np.cos(x * 0.7),
                   color=m.series[1], lw=2.0)

    # The number under each frame is the *clustering*, not the kurtosis. Fig 3 is the
    # reason: at this sample size the kurtosis estimate has a standard deviation of
    # about a third of its value, and a number nobody can reproduce does not belong
    # under a picture.
    pop_c = res["population"]["acf1_abs"]["value"]
    fig_meta, _ = charts.strip_card(
        headline="Which of these remembers that volatility clusters?",
        panels=[(bursty, f"{pop_c:+.2f}", "the process itself"),
                (shuffled_cards,
                 f"{res['rows']['shuffle the returns']['acf1_abs']['value']:+.2f}",
                 "its returns, shuffled"),
                (cloud_to_path,
                 f"{res['rows']['diffusion model']['acf1_abs']['value']:+.2f}",
                 "a diffusion model trained on it")],
        note=("Lag-1 autocorrelation of absolute returns. Shuffling destroys it "
              "completely — and reproduces the fat tails exactly, for free, because "
              "it is the return distribution. That is why this row is the one worth "
              "reading and the fat-tails row is not."),
        footer="The Standard Error", mode="light",
        alt=("A three-panel hand-drawn strip. The first frame shows a return path "
             f"with two bursts of volatility, marked {pop_c:+.2f}. The second shows "
             "four playing cards scattered out of order, marked "
             f"{res['rows']['shuffle the returns']['acf1_abs']['value']:+.2f}. The "
             "third shows a cloud of dots with an arrow to a smooth wiggly line, "
             f"marked {res['rows']['diffusion model']['acf1_abs']['value']:+.2f}."),
        caption="",
        path=str(IMG / f"a5-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    res = compute(verbose=False)
    sch = compute_schedules(verbose=False)
    figs = figures(res, sch)

    ref = res["rows"]["the process itself"]
    pop = res["population"]
    shuf = res["rows"]["shuffle the returns"]
    blk = res["rows"][f"blocks of {BLOCK}"]
    ddpm = res["rows"]["diffusion model"]
    z = z_errors(res)
    top = res["ladder"][HEADLINE_RUNG]
    bottom = res["ladder"][0]
    # Same configuration, same fit seed, same windows as the schedule study's
    # textbook row — so the two differ only in the sampling draw. Asserted, because
    # the paragraph that uses it is a claim about identical weights.
    same_net = res["ladder"][SCHEDULE_RUNG]
    if same_net["config"] != sch["budget"]:
        raise AssertionError(
            f"the 'same trained network' paragraph needs rung {SCHEDULE_RUNG} to "
            f"match the schedule study's budget: {same_net['config']} vs "
            f"{sch['budget']}")
    battery_body = md_table(BATTERY_HEADER, battery_rows(res))
    schedule_body = md_table(SCHEDULE_HEADER, schedule_rows(sch))
    steep = sch["runs"][SCHEDULES[2]]
    broken = sch["runs"][SCHEDULES[1]]
    textbook = sch["runs"][SCHEDULES[0]]
    kurt_share = ddpm["excess_kurtosis"]["value"] / pop["excess_kurtosis"]["value"]
    clust_share = ddpm["acf1_abs"]["value"] / pop["acf1_abs"]["value"]
    ladder_clust = [r["facts"]["acf1_abs"]["value"] for r in res["ladder"]]

    post = Post(
        title="Your Generative Model Was Not Compared Against a Shuffle",
        slug="your-generative-model-was-not-compared-against-a-shuffle",
        date=POST_DATE,
        subtitle=("A survey of diffusion models in finance, and the two baselines "
                  "its evaluation tables are missing"),
        summary=(
            f"A survey posted this month collects the work applying diffusion models — "
            f"the class behind image generators — to financial data: "
            f"{', '.join(DATA_TYPES[:-1])} and {DATA_TYPES[-1]}. Models like these are "
            f"almost always judged on a table of stylised facts. This post summarises "
            f"the survey, then runs the experiment a survey cannot, on a process whose "
            f"answers are known, with the two generators that belong in every such "
            f"table and are almost never in it. Shuffling the training returns "
            f"reproduces the fat tails *exactly* and destroys every trace of volatility "
            f"clustering; a small diffusion model does the opposite. Neither dominates, "
            f"no weighting of the columns is neutral, and a hyperparameter that is not "
            f"part of the model moved the table further than 250 times the training "
            f"compute did."),
        tags=["machine-learning", "generative-models", "quantitative-finance",
              "time-series", "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=1500, max_words=2400,
        table_figures=[figs["battery"], figs["schedules"]],
        data_sources=[
            f"Zhuohan Wang and Carmine Ventre, 'Diffusion Models in Finance: A "
            f"Survey', arXiv:2608.12583, {SURVEY_DATE} (q-fin.CP). Organises the "
            f"literature by financial data type ({', '.join(DATA_TYPES)}) and "
            f"identifies four properties that make the model class attractive for "
            f"finance: {'; '.join(APPEALS)}. Describes itself as the first survey "
            f"dedicated to diffusion-family models in finance and ships an open "
            f"repository at <https://{SURVEY_REPO}>. "
            f"<https://arxiv.org/abs/2608.12583>. **Read from the abstract and the "
            f"listing only** — repeated full-text fetches were rate-limited, so "
            f"nothing here characterises the survey's contents beyond what it says "
            f"about itself, and no result of any paper it reviews is described.",
            f"All data in this post is simulated. Ground truth is a GARCH(1,1) "
            f"path with Student-t({GARCH['df']:.0f}) shocks, arch "
            f"{GARCH['arch']}, beta {GARCH['beta']}, persistence "
            f"{PERSISTENCE:g}, omega {GARCH['omega']}, {GARCH['n']:,} steps, seed "
            f"{GARCH['seed']}, generated by `standarderror.dynamics.sde.garch11`. No "
            f"market data is used or redistributed, and no company appears.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "scipy", "scikit-learn", "standarderror")),
            "model": (f"DDPM with an epsilon-prediction objective, "
                      f"{STEPS}-step linear schedule (beta 1e-4 to 0.02), a "
                      f"two-hidden-layer MLPRegressor denoiser, and a timestep "
                      f"embedding of one ramp plus two sine-cosine pairs. No torch "
                      f"in this environment and two CPUs, which is survivable "
                      f"because epsilon-prediction is a regression"),
            "windows": (f"{res['n_windows']:,} windows of {WINDOW} steps at stride "
                        f"{STRIDE}; neighbouring windows share "
                        f"{WINDOW - STRIDE} values, so the effective sample is well "
                        f"below the row count"),
            "budget_ladder": ", ".join(
                f"width {r['config']['hidden']} x {r['config']['max_iter']} passes "
                f"x {r['config']['noise_per_window']} noisings (cost {r['cost']:.0f}, "
                f"{r['fit_seconds']:.0f}s)" for r in res["ladder"]),
            "scoring": (f"every generator scored on exactly {N_EVAL} paths of "
                        f"{WINDOW} steps, because the pooled kurtosis estimate is "
                        f"sample-size dependent; pooled standard errors from "
                        f"{N_BOOT} resamples of paths, per-path facts from the "
                        f"spread across paths"),
            "sampler_check": (f"ancestral sampling driven by the analytic optimal "
                              f"denoiser for Gaussian data of standard deviation "
                              f"2.5 returns "
                              f"{res['oracle'][str(STEPS)]['recovered_sd']:.3f} at "
                              f"{STEPS} steps and "
                              f"{res['oracle'][str(SHORT_STEPS)]['recovered_sd']:.3f}"
                              f" at {SHORT_STEPS}"),
            "block_sweep": ", ".join(
                f"block {b}: ACF1(|r|) "
                f"{res['block_sweep'][str(b)]['acf1_abs']['value']:+.3f}"
                for b in BLOCK_SWEEP),
            # From the recorded per-rung fit times, not from the wall clock of this
            # run: with the caches warm that number is a few seconds and would be a
            # quietly flattering thing to publish.
            "cost": (f"about "
                     f"{(sum(r['fit_seconds'] for r in res['ladder']) + sch['elapsed_seconds']) / 60:.0f}"
                     f" minutes of fitting on two CPUs for the ladder plus the three "
                     f"schedule runs; cached under a hash of the configuration, and "
                     f"each rung cached separately, so the post re-renders in seconds"),
        },
    )

    post.add("A survey, and the question attached to it", f"""
Diffusion models are the thing behind image generators: learn to remove noise from a
corrupted sample, then start from pure noise and remove it repeatedly until something
plausible falls out. A survey posted to arXiv on {SURVEY_DATE.split()[0]} August
collects the work applying them to financial data, organised by what kind of data —
**{'**, **'.join(DATA_TYPES)}** — and describes itself as the first survey dedicated to
this model family in finance. It ships an open repository of the papers it covers.

It names four reasons the class is attractive here, and they are the right four:
{', '.join(APPEALS[:-1])}, and **{APPEALS[-1]}** — the last because a diffusion process
*is* an SDE, so the machinery already speaks the language quantitative finance is written
in.

I could not read the full text; repeated fetches were rate-limited. So the above is from
the abstract and the listing, and I will not characterise what is inside beyond that.
What I can do instead is run the question the whole field turns on, since it is a
question about *evaluation* rather than about any paper: **when a generative model of
returns matches the stylised facts, how much has it demonstrated?**
""".strip())

    post.add("The table that settles these arguments", """
There is a standard table: a row for fat tails, a row for the absence of autocorrelation
in returns, a row for autocorrelation in *absolute* returns — volatility clustering — and
often one for the leverage effect. The model's numbers sit next to the data's, they are
close, and that is the evidence. Two generators belong in that table and are almost never
in it.

An **i.i.d. bootstrap**: draw values from the training returns with replacement and lay
them end to end. This reproduces the return distribution *exactly*, kurtosis included,
because it *is* that distribution — and destroys every dependence, because the draws are
independent by construction.

A **moving-block bootstrap**: draw contiguous blocks instead of single values.
Dependence survives inside a block and breaks at the joins. One parameter, no training,
no architecture.

Neither is a rival to a diffusion model in what it can *do*. A bootstrap cannot be
conditioned on anything, cannot produce a value it has not already seen, and cannot
extrapolate — which is most of why you would want a generative model at all. That is
exactly why they belong in the table: they mark how much of a stylised-facts score is
available *without a model*, and until they are in it there is no way to know how much
of the score that is.
""".strip())

    post.add("First, check that the fact exists", f"""
Ground truth is a GARCH(1,1) path with Student-t shocks — fat tails and volatility
clustering, both put there on purpose, both with known values. The model generates
**{WINDOW}-step windows**, one path at a time, which creates a trap I walked into: a
generator emitting a {WINDOW}-step window can only represent dependence that fits *inside*
{WINDOW} steps. Equity-index volatility persistence is usually estimated around
**{EQUITY_PERSISTENCE:g}**, and at that persistence the variance moves so slowly that
consecutive absolute returns inside a short window barely covary — inside a 32-step window
the lag-1 autocorrelation of absolute returns is
**{res['clustering']['equity']['32']:+.3f}**.

My first run used exactly that. The model failed to reproduce the clustering, and the
failure was mine — there was none inside the window to reproduce. Lower the persistence to
**{PERSISTENCE:g}**, lengthen the window to {WINDOW}, and the true in-window value is
**{pop['acf1_abs']['value']:+.3f}**, which is a fact that exists. The general form is
worth keeping: **before asking whether a model reproduces a statistic, measure that
statistic on the training windows.**
""".strip(), figures=[figs["window"]])

    post.add("What the baselines do", f"""
Measured over all {res['n_windows']:,} windows of the path, the process has excess
kurtosis **{pop['excess_kurtosis']['value']:.1f}** and clustering
**{pop['acf1_abs']['value']:+.3f}**. Those are the numbers to beat.

Shuffling the returns gives clustering **{shuf['acf1_abs']['value']:+.3f}** — nothing, and
{abs(z['shuffle the returns']['acf1_abs']):.0f} standard errors from the truth, exactly as
designed. Its excess kurtosis is **{shuf['excess_kurtosis']['value']:.1f}**, which I come
back to.

Blocks of {BLOCK} recover **{blk['acf1_abs']['value']:+.3f}** of the clustering —
{100 * blk['acf1_abs']['value'] / pop['acf1_abs']['value']:.0f}% of the true value, which
is a match within noise. Dependence does break at every join, and at short blocks that
shows: blocks of 2 reach only
{res['block_sweep']['2']['acf1_abs']['value']:+.3f}. By {BLOCK} the joins are rare enough
not to matter. Excess kurtosis **{blk['excess_kurtosis']['value']:.1f}**. Three rows of
the standard table, one parameter, no model.

{battery_body}

The leverage column shows what kind of thing this table is. A symmetric GARCH has **no
leverage effect**; the true value is {pop['leverage']['value']:+.3f}. A generator that
produced one here would not have scored a point — it would have invented a dependence
the data does not contain. The row is only readable if you already know the answer, and
on real data nobody does.
""".strip())

    post.add("And the fat-tails row cannot be measured anyway", f"""
Look at the first column of that table again. The process, sampled the way every row is
sampled, reports **{ref['excess_kurtosis']['value']:.1f}**. The shuffle reports
**{shuf['excess_kurtosis']['value']:.1f}** — a factor of
{ref['excess_kurtosis']['value'] / shuf['excess_kurtosis']['value']:.1f} apart. But the
shuffle is **exactly right in expectation**: it *is* the return distribution, so its
kurtosis is that distribution's kurtosis by construction. Something is wrong, and it is
not the shuffle.

It is the estimator. Draw {N_EVAL} windows from the process
{len(res['spread']['contiguous'])} times and measure each draw's excess kurtosis: the
standard deviation of that estimate is **{np.std(res['spread']['contiguous']):.1f}**,
roughly {100 * np.std(res['spread']['contiguous']) / pop['excess_kurtosis']['value']:.0f}%
of the value being estimated. The table's {ref['excess_kurtosis']['value']:.1f} is a
high draw; the shuffle's {shuf['excess_kurtosis']['value']:.1f} is a low one. Same
quantity, same distribution.

A fourth moment is decided by the handful of largest observations in a sample, and
clustering makes that worse rather than better, because the largest observations arrive
together: one turbulent stretch contributes sixty-four big values at once, so a
contiguous sample is effectively far smaller than its value count suggests — here its
spread is {np.std(res['spread']['contiguous']) / np.std(res['spread']['iid']):.1f} times
the i.i.d. one's at the same number of values.

The sampling seed alone does it too. The "{SCHEDULES[0]}" row of the next section and the
third rung of the budget ladder are **the same trained network with the same weights**,
differing only in the random draw used to generate. Their excess kurtosis figures are
{textbook['facts']['excess_kurtosis']['value']:.2f} and
{same_net['facts']['excess_kurtosis']['value']:.2f}.

So fat tails are free twice over. Free to pass, because reproducing the return
distribution is enough and a shuffle does that for nothing. And effectively free to
fail, because at these sample sizes the row cannot resolve a factor of two. **A table
that reports kurtosis to two decimals and declares a match is reporting a coin flip.**
If the row appears at all it needs an error bar from repeated independent draws, not
from a bootstrap: the path bootstrap behind Table 1 gives
{shuf['excess_kurtosis']['se']:.1f} for the shuffle, about a third of the real spread.
""".strip(), figures=[figs["noise"]])

    post.add("What the diffusion model does", f"""
The model is a DDPM with the standard objective: corrupt a window to a randomly chosen
noise level, predict the noise that was added, then sample by undoing that repeatedly from
pure noise. Worth noting that this training task is *ordinary regression* — the forward
process has a closed form, so pairs can be manufactured in unlimited quantity from a
finite dataset. Hence no deep-learning framework here: the denoiser is a two-layer
perceptron on two CPUs, which is enough for the model class to *exist* and not enough for
it to be good.

At the largest budget I ran, the samples have excess kurtosis
**{ddpm['excess_kurtosis']['value']:.1f}** against a true
{pop['excess_kurtosis']['value']:.1f} — about {100 * kurt_share:.0f}% of it, and far
enough below to survive the noise in the last section — and clustering
**{ddpm['acf1_abs']['value']:+.3f}** against a true
{pop['acf1_abs']['value']:+.3f}, about {100 * clust_share:.0f}%. Returns are
uncorrelated, correctly. The standard deviation is
{ddpm['sd']['value'] / pop['sd']['value']:.2f} times the truth.

So the two generators fail in *opposite* places. The shuffle gets the distribution
exactly and the memory not at all; the model gets a fifth of the tails and a third of the
memory — relatively more of the dependence than of the distribution, which is the harder
half and the half a bootstrap cannot do. Neither dominates, and there is no weighting of
five columns in five different units that is not a choice someone made.

That is the finding, and it is about the table rather than about either generator. **A
single "realism" score over this battery would have ranked these two, and the ranking
would have been an artefact of the weights.**
""".strip(), figures=[figs["errors"]])

    post.add("Was it just undertrained?", f"""
The obvious objection, and it deserves an answer rather than hedging. I ran
{len(LADDER)} budgets spanning **{top['cost'] / bottom['cost']:.0f} times** the training
compute — wider layers, more passes, more noisings per window. At the smallest, excess
kurtosis {bottom['facts']['excess_kurtosis']['value']:.1f} and clustering
{bottom['facts']['acf1_abs']['value']:+.3f}: Gaussian noise with no memory. At the
largest, {top['facts']['excess_kurtosis']['value']:.1f} and
{top['facts']['acf1_abs']['value']:+.3f}. Both far above where they started, both still
well short.

Worth saying where that factor came from, because it is the sort of thing that quietly
flatters a chart: a rung *above* the largest here costs an hour of wall clock on two CPUs,
while a rung *below* the smallest costs ten seconds and widens the axis by the same
factor. I bought the span at the cheap end, and the top of it is set by the machine rather
than by the argument.

So this supports the narrow claim — not enough compute — and not the wider one, since
extrapolating a log-axis trend says nothing about where it stops. It does not touch the
argument either way. The claim is not that diffusion models fail this table; it is that
**the shuffle passes the tails row for free**, which is exact and needs no model at
all.
""".strip(), figures=[figs["budget"]])

    post.add("The hyperparameter that moved the table more than the model did", f"""
This is the part I did not expect, and it is the sharpest version of the point.

The **forward process** — the schedule by which noise is added — is not part of the
model. It is a fixed list of numbers chosen before training. And the standard recipe
has a trap in it: the textbook schedule runs {STEPS:,} steps, and shortening it to
{SHORT_STEPS} to make sampling five times cheaper *while keeping the endpoints* leaves
the final noise level too low to have destroyed the signal. Terminal signal-to-noise
ratio **{broken['terminal_snr']:.2f}** instead of {textbook['terminal_snr']:.0e}, so
the sampler starts from a standard normal draw when the correct starting distribution
is something else.

The failure is invisible in the worst possible way: the data is standardised before
training, so the mismatched start still has the right *variance* and only the higher
moments — the ones being measured — are damaged.

I caught it with a test that used no data at all. For Gaussian data the best possible
noise prediction has a closed form, so you can drive the sampler with the exact answer:
asked for standard deviation 2.5, the {STEPS:,}-step schedule returns
**{res['oracle'][str(STEPS)]['recovered_sd']:.3f}** and the {SHORT_STEPS}-step one
**{res['oracle'][str(SHORT_STEPS)]['recovered_sd']:.3f}**. The sampler is right; the
schedule is wrong.

Now the uncomfortable part. Three forward processes at one fixed budget, same data, same
seed, same network.

{schedule_body}

**The broken one wins.** On the clustering row — the informative row, the one a
shuffle cannot fake — the correct slow schedule reaches
{textbook['facts']['acf1_abs']['value']:+.3f}, the correct fast one
{steep['facts']['acf1_abs']['value']:+.3f}, and the one whose starting distribution is
demonstrably wrong reaches **{broken['facts']['acf1_abs']['value']:+.3f}**, against a
true {pop['acf1_abs']['value']:+.3f}. The best clustering figure anywhere in this post
is not the largest model's {max(ladder_clust):+.3f}; it is that one, at
{same_net['cost'] / top['cost']:.2f} times the largest rung's training cost. Excess
kurtosis barely moves across the three rows — so the schedule moved the row that carries
information and left the row that does not alone.

Why it scores better I cannot say. Two things differ between it and the slow schedule —
terminal SNR and step count — and the third row separates them only partly. Untangling
that is a paper, not a paragraph.

The part I can state plainly is the part that matters for reading someone else's table:
**a stylised-facts score moved further under a choice that is not part of the model than
under 250 times the training compute** — and it preferred the configuration I can prove is
wrong.
""".strip())

    post.add("What to ask of an evaluation table", """
None of this argues against the model class, and the survey's four appeals are untouched —
a bootstrap has none of them. Conditioning matters most in practice and is exactly what a
resampling scheme cannot do: if you want scenarios given a state, a bootstrap has nothing
to offer and a diffusion model does. The argument is about what a *table* can support.
Four things, all cheap:

**Put the shuffle in.** One line. If fat tails are the headline result, the shuffle is
the honest baseline for them, and it matches by construction.

**Put the block bootstrap in.** Two lines and one parameter, and it is the bar for "the
model learned the temporal structure". Report the block length; sweep it if you can.

**Give every row an error bar from repeated draws** — not from a bootstrap, which
understates a fourth moment's spread by a factor of three here. Any row whose error bar
covers a factor of two is not evidence, and should be labelled that way rather than
quoted to two decimals. While you are there, say what the true value *is*: a leverage
row means nothing without knowing whether the process has one.

**Report the schedule.** Steps, endpoints, terminal signal-to-noise ratio. A table that
moves this much under a forward-process choice cannot be read without it.

And one for the reader rather than the author: **a matched return distribution is not
evidence**. Fat tails are free. The dependence structure is not, and that is the column
to read first.
""".strip())

    post.add("Where to be careful", f"""
**One process, one window length, one architecture.** The qualitative claim — a shuffle
reproduces the return distribution exactly — is an identity and holds anywhere. Every
number attached to the diffusion model is specific to this setup and is not a property of
diffusion models.

**The budget is small, and I have said so twice.** A convolutional or attention denoiser
at a real budget would likely close much of both gaps. The ladder shows a direction, not a
limit. Training windows also overlap — strided by {STRIDE}, so neighbours share
{WINDOW - STRIDE} of their {WINDOW} values — which means the effective sample is well
below the {res['n_windows']:,} rows the model saw.

**And the survey itself I have only read from the outside.** The abstract, the
listing, the repository link. Nothing above describes its contents, evaluates its
judgements, or comments on any paper it reviews — the experiment stands on its own
and would have been worth running whatever the survey says.
""".strip())
    return post


if __name__ == "__main__":
    force = bool(os.environ.get("SERR_FORCE"))
    which = os.environ.get("SERR_PART", "all")
    if which in ("all", "ladder"):
        compute(force=force)
    if which in ("all", "schedules"):
        compute_schedules(force=force)
