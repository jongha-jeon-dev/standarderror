"""exp014 — robust XGBoost, and the scale constant hiding in the loss.

Requested directly: write up Aragon Mladosich and Croux, "Robust XGBoosting for
Regression" (arXiv:2608.13590).

Same shape as exp012 and exp013: report the paper, then add the one thing a summary
cannot — a measurement, on data whose answer is known, with a control that switches
the claimed mechanism off.

The claim
---------
XGBoost's robust option is `reg:pseudohubererror`, and it has a parameter,
`huber_slope`, whose default is **1.0**. A Huber loss is a function of `r / delta`:
quadratic inside the transition point, linear outside. So `delta` is a *scale*, and
fixing it fixes the units of the response.

Consequence, which is the headline: **how robust the model is depends on what you
measured y in.** Multiply y by a constant, refit, divide the predictions back — a
transformation that changes nothing about the problem — and the same code on the same
data goes from fully robust to no better than squared error. Squared error itself is
exactly equivariant, because its loss is homogeneous in the residual and has no
constant to be wrong about. So does absolute error. Only the robust option in the
list has the units problem, which is a good joke and a real trap.

The fix is to estimate the scale from the data, and that is exactly what M-, S- and
tau-estimators do — so this post arrives from outside at the reason the paper's design
is the right one, and at why it needed a two-step procedure rather than a one-shot
rescaling: a scale taken from a contaminated y is inflated by the very points the loss
is meant to discount.

Three things this found on the way, all in the body
--------------------------------------------------
1. **The mis-scaled default is not simply worse.** On contaminated data at this
   response's natural scale it is the most robust setting tested. On clean data at a
   hundred times that scale it starts undertrained and needs thirty times the rounds
   to get near what the well-scaled version reaches at three hundred. It is an
   unstated bias-robustness trade-off that moves with your units.
2. **For a tree, a leverage point is dangerous in inverse proportion to how far out it
   sits.** The first version of this experiment put its leverage points eight standard
   deviations out in every coordinate, measured no damage at all, and would have
   published that as a fact about XGBoost. It was a fact about the construction: a
   far-out point falls beyond the outermost split and gets a leaf nobody visits. Moved
   in to 1.5 standard deviations, the same 10% contamination takes squared error to
   nearly four times its clean-data error. The linear-model intuition, where leverage
   grows with distance, is backwards here.
3. **Robustness here is partly a round budget.** Every loss tested, including absolute
   error, gets worse under contamination as boosting continues — a tree can always
   carve out a leaf for an outlier eventually. A robustness claim without a stated
   number of rounds is not a claim.

No market data and no company: a synthetic regression with a known mean function, so
every error is measured against the truth rather than against a held-out sample of the
same contamination.

Run: `standarderror run exp014_huber_slope_is_a_scale --publish`
"""

from __future__ import annotations

from datetime import date

import hashlib
import json
import os
import time

import numpy as np

import standarderror as se
from standarderror.render import Post
from standarderror.robust import contamination, equivariance, scale
from standarderror.viz import charts, theme

#: Pinned so a rebuild cannot silently re-date a published post.
#: `Post.date` defaults to today, which is correct exactly once.
POST_DATE = date(2026, 8, 19)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed

# --- the paper, as it describes itself -----------------------------------------
PAPER = "Iris Aragon Mladosich and Christophe Croux, arXiv:2608.13590"
PAPER_DATE = "14 July 2026"
ESTIMATORS = ("M-", "S-", "tau-")
PAPER_BEST = "MM-XGBoost"

# --- the problem ---------------------------------------------------------------
N_TRAIN, N_TEST, N_FEATURES = 2000, 800, 5
NOISE_SD = 1.0
ROUNDS = 300
DEPTH, LR = 3, 0.1
MAGNITUDE = 20.0                 # outlier size, in units of y at scale 1
DEFAULT_SLOPE = 1.0              # XGBoost's own default for huber_slope

SCALES = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
FRACTIONS = (0.0, 0.02, 0.05, 0.10, 0.20)
DISTANCES = (1.5, 2.5, 3.5, 5.0, 8.0, 12.0)
ROUND_BUDGETS = (100, 300, 1000, 3000)
HEADLINE_FRACTION = 0.10

#: label -> (objective, slope), where "auto" means set from the data each fit.
LOSSES = {
    "squared error": ("reg:squarederror", None),
    f"Huber, slope {DEFAULT_SLOPE:g} (the default)": ("reg:pseudohubererror",
                                                      DEFAULT_SLOPE),
    "Huber, slope from the data": ("reg:pseudohubererror", "auto"),
    "absolute error": ("reg:absoluteerror", None),
}
DEFAULT_LOSS = f"Huber, slope {DEFAULT_SLOPE:g} (the default)"
FIXED_LOSS = "Huber, slope from the data"
CACHE = se.SETTINGS.build_dir / "cache" / "exp014.json"


def truth(A) -> np.ndarray:
    """The mean function every error in this post is measured against.

    One smooth term, one convex term, one linear term, and two features that do
    nothing — so a method can be wrong by missing structure or by inventing it.
    """
    A = np.asarray(A, dtype=float)
    return 3.0 * np.sin(A[:, 0]) + A[:, 1] ** 2 - 2.0 * A[:, 2]


def problem(seed: int = SEED):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_TRAIN, N_FEATURES))
    y = truth(X) + rng.standard_normal(N_TRAIN) * NOISE_SD
    X_test = rng.standard_normal((N_TEST, N_FEATURES))
    return X, y, X_test, truth(X_test)


def _config_key() -> str:
    blob = json.dumps({"v": 1, "n": [N_TRAIN, N_TEST, N_FEATURES],
                       "noise": NOISE_SD, "rounds": ROUNDS, "depth": DEPTH,
                       "lr": LR, "magnitude": MAGNITUDE, "scales": SCALES,
                       "fractions": FRACTIONS, "distances": DISTANCES,
                       "budgets": ROUND_BUDGETS, "losses": LOSSES,
                       "seed": SEED}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def fit_predict(X, y, X_test, label: str, *, rounds: int = ROUNDS) -> np.ndarray:
    """One XGBoost fit at one loss, with the slope set the way `label` says.

    `"auto"` uses a robust scale of the response, which is all that is available
    before fitting. It is enough to restore equivariance and deliberately not enough
    to be optimal — see `standarderror.robust.equivariance.huber_slope_for`.
    """
    import xgboost as xgb
    objective, slope = LOSSES[label]
    kw = dict(n_estimators=int(rounds), max_depth=DEPTH, learning_rate=LR,
              objective=objective, random_state=SEED, n_jobs=2)
    if slope == "auto":
        kw["huber_slope"] = equivariance.huber_slope_for(y)
    elif slope is not None:
        kw["huber_slope"] = float(slope)
    return xgb.XGBRegressor(**kw).fit(X, y).predict(X_test)


def rmse_against_truth(X, y, X_test, f_test, label: str, *,
                       scale_factor: float = 1.0, rounds: int = ROUNDS) -> float:
    """Clean-test RMSE against the true mean function, in scale-1 units.

    Measured against `truth`, not against held-out contaminated data. Holding out a
    sample of the same contamination measures how well a method reproduces the
    outliers, which is the opposite of the question.
    """
    pred = fit_predict(X, y, X_test, label, rounds=rounds) / scale_factor
    return float(np.sqrt(np.mean((pred - f_test) ** 2)))


def compute(*, force: bool = False, verbose: bool = True) -> dict:
    """Every grid in the post. Cached under a hash of the configuration."""
    key = _config_key()
    if not force and CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            return cached

    t0 = time.time()

    def say(*a):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)

    X, y, X_test, f_test = problem()
    dirty = contamination.vertical_outliers(X, y, fraction=HEADLINE_FRACTION,
                                            magnitude=MAGNITUDE, seed=SEED + 1)
    say("problem built;", dirty.describe())

    # 1. The headline: refit at each scale of y, rescale the predictions back.
    #    Run on clean and contaminated data, because the interesting half is that
    #    the two disagree about which scale is best.
    sweep = {"clean": {}, "contaminated": {}}
    for name, (Xd, yd) in (("clean", (X, y)), ("contaminated", (dirty.X, dirty.y))):
        for label in LOSSES:
            sweep[name][label] = [
                rmse_against_truth(Xd, yd * s, X_test, f_test, label,
                                   scale_factor=s) for s in SCALES]
            say(f"scale sweep [{name}] {label}: "
                f"{np.round(sweep[name][label], 3).tolist()}")

    # 2. The paper's motivation: vertical outliers against contamination fraction.
    fraction_sweep = {}
    for label in LOSSES:
        fraction_sweep[label] = []
        for frac in FRACTIONS:
            c = contamination.vertical_outliers(X, y, fraction=frac,
                                                magnitude=MAGNITUDE, seed=SEED + 1)
            fraction_sweep[label].append(
                rmse_against_truth(c.X, c.y, X_test, f_test, label))
        say(f"fraction sweep {label}: {np.round(fraction_sweep[label], 3).tolist()}")

    # 3. Leverage points, against how far out they sit. The finding is the shape.
    distance_sweep = {}
    for label in ("squared error", FIXED_LOSS):
        distance_sweep[label] = []
        for d in DISTANCES:
            c = contamination.leverage_points(X, y, truth,
                                              fraction=HEADLINE_FRACTION,
                                              distance=d, magnitude=MAGNITUDE,
                                              seed=SEED + 2)
            distance_sweep[label].append(
                rmse_against_truth(c.X, c.y, X_test, f_test, label))
        say(f"distance sweep {label}: {np.round(distance_sweep[label], 3).tolist()}")

    # 4. Is the robustness a property of the loss or of the round budget?
    budget_sweep = {}
    for label in LOSSES:
        budget_sweep[label] = [
            rmse_against_truth(dirty.X, dirty.y, X_test, f_test, label,
                               rounds=r) for r in ROUND_BUDGETS]
        say(f"budget sweep {label}: {np.round(budget_sweep[label], 3).tolist()}")

    # 5. How many rounds does the mis-scaled default need to catch up on clean data?
    catch_up = {}
    for s in (1.0, 100.0):
        catch_up[s] = {}
        for r in (300, 1000, 3000, 9000):
            catch_up[s][r] = rmse_against_truth(X, y * s, X_test, f_test,
                                                DEFAULT_LOSS, scale_factor=s,
                                                rounds=r)
        say(f"catch-up s={s:g}: "
            f"{ {k: round(v, 3) for k, v in catch_up[s].items()} }")

    # 6. The equivariance gap itself, as one number per loss.
    gaps = {}
    for label in LOSSES:
        gaps[label] = equivariance.equivariance_gap(
            lambda A, b, s, _l=label: fit_predict(A, b, X_test, _l), X, y, X_test,
            scales=(0.1, 1.0, 10.0, 100.0))
        say(f"equivariance gap {label}: {gaps[label]:.3g}")

    out = {
        "key": key,
        "scales": list(SCALES),
        "sweep": sweep,
        "fractions": list(FRACTIONS),
        "fraction_sweep": fraction_sweep,
        "distances": list(DISTANCES),
        "distance_sweep": distance_sweep,
        "budgets": list(ROUND_BUDGETS),
        "budget_sweep": budget_sweep,
        "catch_up": {str(k): {str(a): b for a, b in v.items()}
                     for k, v in catch_up.items()},
        "gaps": gaps,
        "mad_of_y": float(scale.mad_scale(y)),
        "mad_of_dirty_y": float(scale.mad_scale(dirty.y)),
        "residual_scale_clean": float(scale.residual_scale(
            y, truth(X))),
        "contamination": dirty.describe(),
        "elapsed_seconds": time.time() - t0,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=1))
    say("cached to", CACHE)
    return out


# Kept short: `theme.finish` sizes the figure to fit the source note on one line.
SRC = (f"Simulated regression, n={N_TRAIN}, {N_FEATURES} features, noise sd "
       f"{NOISE_SD:g}, depth {DEPTH}, seed {SEED}.")
TABLE_HEADER = ["boosting rounds", *LOSSES]


def budget_rows(res: dict) -> list[list[str]]:
    rows = []
    for i, r in enumerate(res["budgets"]):
        rows.append([f"{r:,}"] + [f"{res['budget_sweep'][k][i]:.2f}"
                                  for k in LOSSES])
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
    labels = list(LOSSES)
    clean_squared = res["fraction_sweep"]["squared error"][0]

    # F1 — the headline. Same data, same code, y multiplied by a constant and the
    # predictions divided back. Two of these lines are flat because their losses have
    # no scale constant in them; one is not.
    frame = pd.DataFrame({k: res["sweep"]["contaminated"][k] for k in labels},
                         index=list(res["scales"]))
    default = res["sweep"]["contaminated"][DEFAULT_LOSS]
    best_i, worst_i = int(np.argmin(default)), int(np.argmax(default))
    one = list(res["scales"]).index(1.0)
    fixed_flat_f1 = res["sweep"]["contaminated"][FIXED_LOSS][one]
    abs_flat_f1 = res["sweep"]["contaminated"]["absolute error"][one]

    def mark_floor(_fig, ax):
        m = theme.LIGHT
        ax.axvline(res["scales"][1], color=m.muted, lw=1.0, ls=(0, (2, 3)))
        ax.annotate("left of here the library's\nfloat32 arithmetic bites",
                    (res["scales"][1], 0.03), xycoords=("data", "axes fraction"),
                    xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                    color=m.muted, va="bottom", linespacing=1.4)

    fig_meta, _ = charts.lines(
        frame, mode="light", logx=True, logy=True, direct_labels=False,
        decorate=mark_floor,
        title="The same data, in different units, with the same code",
        subtitle=(f"Clean-test error against the true mean function, with "
                  f"{100 * HEADLINE_FRACTION:.0f}% of responses shifted by "
                  f"{MAGNITUDE:g}. The response is multiplied by the number on the "
                  f"x-axis and the predictions divided back, which changes nothing "
                  f"about the problem."),
        ylabel="RMSE against the truth (log scale)",
        xlabel="constant the response was multiplied by (log scale)", source=SRC,
        alt=("Four curves on log axes. Squared error, absolute error and "
             "data-set-slope Huber are flat horizontal lines. The default-slope "
             "Huber curve is a deep U: high at the left, lowest in the middle, and "
             "rising steeply at the right."),
        caption=(f"Fig 1. Three of these four losses are flat, because their "
                 f"transition points either do not exist or are set from the data. "
                 f"The default-slope Huber runs from "
                 f"{default[worst_i]:.2f} down to {default[best_i]:.2f} and back — a "
                 f"factor of {default[worst_i] / default[best_i]:.0f} across the "
                 f"sweep, from a choice of units. At the left of the sweep it is no "
                 f"better than squared error, which is to say not robust at all. The "
                 f"two lower flat lines sit almost exactly on top of each other, at "
                 f"{fixed_flat_f1:.2f} for the data-set slope and {abs_flat_f1:.2f} for "
                 f"absolute error — a coincidence of this problem, not a missing "
                 f"series. Read the flat lines as the calibration: they move at the "
                 f"leftmost point too, which is where float32 arithmetic inside the "
                 f"library starts to matter rather than anything about the loss."),
        path=str(IMG / f"a6-f1-units.{EXT}"))
    figs["units"] = fig_meta

    # F2 — the paper's motivating claim, reproduced.
    frac = pd.DataFrame({k: res["fraction_sweep"][k] for k in labels},
                        index=[100 * f for f in res["fractions"]])
    fig_meta, _ = charts.lines(
        frac, mode="light", direct_labels=False,
        title="Squared-error boosting is the one that breaks",
        subtitle=(f"Clean-test error against the fraction of responses shifted by "
                  f"{MAGNITUDE:g}, at the response's natural scale and "
                  f"{ROUNDS} rounds."),
        ylabel="RMSE against the truth", xlabel="share of responses corrupted (%)",
        source=SRC,
        alt=("Four rising curves against contamination share. Squared error rises "
             "steepest, from about 0.5 to over 3. The default-slope Huber rises "
             "least, staying below 0.9 throughout."),
        caption=(f"Fig 2. The paper's motivation, reproduced: at "
                 f"{100 * res['fractions'][-1]:.0f}% contamination squared error is "
                 f"{res['fraction_sweep']['squared error'][-1] / clean_squared:.1f} "
                 f"times its clean-data error, while the default-slope Huber is the "
                 f"most robust setting here — the same setting Fig 1 shows to be "
                 f"robust only by accident of units. Two of these curves would swap "
                 f"places if the response were measured in hundreds."),
        path=str(IMG / f"a6-f2-fraction.{EXT}"))
    figs["fraction"] = fig_meta

    # F3 — leverage, and the shape nobody expects from linear-model intuition.
    dist = pd.DataFrame({k: res["distance_sweep"][k] for k in res["distance_sweep"]},
                        index=list(res["distances"]))
    sq = res["distance_sweep"]["squared error"]

    def mark_clean(_fig, ax):
        m = theme.LIGHT
        ax.axhline(clean_squared, color=m.ink_secondary, lw=1.2)
        ax.annotate("squared error with no contamination at all",
                    (res["distances"][-1], clean_squared), xytext=(0, 6),
                    textcoords="offset points", ha="right", fontsize=8.5,
                    color=m.ink_secondary)

    fig_meta, _ = charts.lines(
        dist, mode="light", direct_labels=False, decorate=mark_clean,
        title="Far-out leverage points are nearly free",
        subtitle=(f"Clean-test error against how far out the leverage points sit, at "
                  f"{100 * HEADLINE_FRACTION:.0f}% contamination. Standard deviations "
                  f"of the first feature."),
        ylabel="RMSE against the truth",
        xlabel="distance of the leverage points (standard deviations)", source=SRC,
        alt=("Two curves against leverage distance, over a horizontal reference "
             "line for uncontaminated squared error. The squared-error curve starts "
             "near 2.0 at 1.5 standard deviations and flattens near 1.0 beyond 3.5. "
             "The robust curve peaks at 2.5 standard deviations instead and settles "
             "near 0.8."),
        caption=(f"Fig 3. For a linear model, leverage is a distance and danger grows "
                 f"with it. For a tree ensemble it is the other way round: a point far "
                 f"outside the data falls beyond the outermost split and gets a leaf "
                 f"no test point visits. At {res['distances'][0]:g} standard "
                 f"deviations, 10% contamination takes squared error to "
                 f"{sq[0]:.2f} against {clean_squared:.2f} clean; at "
                 f"{res['distances'][-1]:g} it only reaches {sq[-1]:.2f}. The first "
                 f"version of this experiment used the far case, measured almost no "
                 f"damage, and nearly reported that as a property of XGBoost. The two "
                 f"curves do not share a shape: squared error is worst at the nearest "
                 f"distance, while the robust variant peaks one step later, at "
                 f"{res['distances'][1]:g}. What they agree on is the right-hand half — "
                 f"past {res['distances'][2]:g} standard deviations, moving the "
                 f"contamination further out buys nothing."),
        path=str(IMG / f"a6-f3-leverage.{EXT}"))
    figs["leverage"] = fig_meta

    # T1 — the round budget, which turns out to be part of every robustness claim.
    fig_meta, _ = charts.table_image(
        budget_rows(res), header=TABLE_HEADER,
        title="Every loss here gets worse the longer you boost",
        subtitle=(f"Clean-test RMSE at {100 * HEADLINE_FRACTION:.0f}% vertical "
                  f"contamination, against the number of boosting rounds. Lower is "
                  f"better; the clean-data error is {clean_squared:.2f}."),
        source=SRC, mode="light", bold_cols=(2,), align="lrrrr",
        alt=("A table of four boosting-round budgets against four loss functions, "
             "with clean-test RMSE in each cell. Every column's value increases as "
             "the number of rounds increases."),
        caption=("Table 1. A tree can always carve out a leaf for an outlier "
                 "eventually, so a robust loss delays overfitting to contamination "
                 "rather than preventing it. Every column here rises with the round "
                 "count, which means a robustness comparison at an unstated number of "
                 "rounds is not a comparison. The bolded column is the setting Fig 1 "
                 "shows is robust only because of the units this response happens to "
                 "be in."),
        path=str(IMG / f"a6-t1-rounds.{EXT}"))
    figs["rounds"] = fig_meta

    # HERO — two panels: the identical picture, measured with two different rulers.
    def cloud(panel, m, coarse: bool):
        from matplotlib.patches import Rectangle
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 6)
        rng = np.random.default_rng(7)
        x = np.linspace(2.6, 9.3, 40)
        panel.plot(x, 3.0 + 1.1 * np.sin((x - 2.6) * 0.85), color=m.series[0],
                   lw=2.4, solid_capstyle="round")
        px = rng.uniform(2.6, 9.3, 46)
        panel.scatter(px, 3.0 + 1.1 * np.sin((px - 2.6) * 0.85)
                      + rng.normal(0, 0.42, px.size), s=13, color=m.muted,
                      alpha=0.85)
        panel.scatter([5.4, 7.6], [5.5, 0.7], s=42, color=m.ink, zorder=5)
        # The ruler: the only difference between the two panels.
        panel.add_patch(Rectangle((1.0, 0.5), 0.55, 5.0, fc=m.surface, ec=m.ink,
                                  lw=1.8))
        step = 1.0 if coarse else 0.25
        yv = 0.5 + step
        while yv < 5.5:
            panel.plot([1.0, 1.55], [yv, yv], color=m.ink, lw=1.3)
            yv += step

    ct = res["sweep"]["contaminated"][DEFAULT_LOSS]
    i_one = list(res["scales"]).index(1.0)
    i_small = list(res["scales"]).index(1e-2)
    fig_meta, _ = charts.strip_card(
        headline="Change the units, change how robust your model is",
        panels=[(lambda p, m: cloud(p, m, True), f"{ct[i_one]:.2f}",
                 "y measured in ones"),
                (lambda p, m: cloud(p, m, False), f"{ct[i_small]:.2f}",
                 "y measured in hundreds")],
        note=(f"Error against the truth, {100 * HEADLINE_FRACTION:.0f}% of responses "
              f"corrupted, identical code. The robust loss has a transition point, its "
              f"default is the number 1, and a transition point is a scale."),
        footer="The Standard Error", mode="light",
        alt=("A two-panel hand-drawn strip. Both frames show the same scatter of "
             "points around a wavy curve with two outliers, and a ruler at the left; "
             "the first ruler has coarse tick marks and the second fine ones. The "
             f"numbers under them are {ct[i_one]:.2f} and {ct[i_small]:.2f}."),
        caption="",
        path=str(IMG / f"a6-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    res = compute(verbose=False)
    figs = figures(res)

    ct = res["sweep"]["contaminated"][DEFAULT_LOSS]
    cl = res["sweep"]["clean"][DEFAULT_LOSS]
    scales = list(res["scales"])
    i_one, i_small = scales.index(1.0), scales.index(1e-2)
    best, worst = min(ct), max(ct)
    sq_flat = res["sweep"]["contaminated"]["squared error"][i_one]
    fixed_flat = res["sweep"]["contaminated"][FIXED_LOSS][i_one]
    abs_flat = res["sweep"]["contaminated"]["absolute error"][i_one]
    clean_squared = res["fraction_sweep"]["squared error"][0]
    frac_last = res["fraction_sweep"]
    dist = res["distance_sweep"]["squared error"]
    catch = res["catch_up"]
    budget_body = md_table(TABLE_HEADER, budget_rows(res))
    # The claim in the prose is that these two are flat, so assert it here rather
    # than trusting a figure that was reviewed once.
    for label in ("squared error", FIXED_LOSS, "absolute error"):
        if res["gaps"][label] > 1e-5:
            raise AssertionError(
                f"{label} was described as scale equivariant but its gap is "
                f"{res['gaps'][label]:.3g}")
    if res["gaps"][DEFAULT_LOSS] < 0.05:
        raise AssertionError("the whole post needs the default slope to fail the "
                             f"equivariance test; gap {res['gaps'][DEFAULT_LOSS]:.3g}")

    post = Post(
        title="Your Robust Loss Has a Unit Bug",
        slug="your-robust-loss-has-a-unit-bug",
        date=POST_DATE,
        subtitle=("A paper makes XGBoost robust with M-, S- and tau-estimators. The "
                  "reason it had to is a number equal to 1."),
        summary=(
            f"XGBoost's robust option is a Huber loss, and a Huber loss is a function "
            f"of the residual **divided by a scale**. XGBoost's default for that "
            f"scale is the number 1. So multiply your response by a constant — metres "
            f"to centimetres, dollars to thousands — refit, divide the predictions "
            f"back, and you get a different model. On a synthetic regression with "
            f"{100 * HEADLINE_FRACTION:.0f}% corrupted responses, that transformation "
            f"moves the error from {best:.2f} to {worst:.2f}: at one end of the sweep "
            f"the loss is fully robust, at the other it is no better than squared "
            f"error. Squared error and absolute error do not move at all, because "
            f"their losses contain no constant to be wrong about. A paper posted this "
            f"summer fixes XGBoost's robustness with {', '.join(ESTIMATORS)}estimators "
            f"from robust regression; this post is the measurement that says why "
            f"estimating the scale, rather than assuming it, is the part that "
            f"matters."),
        tags=["machine-learning", "gradient-boosting", "robust-statistics",
              "regression", "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=1500, max_words=2400,
        table_figures=[figs["rounds"]],
        data_sources=[
            f"Iris Aragon Mladosich and Christophe Croux, 'Robust XGBoosting for "
            f"Regression', arXiv:2608.13590 (cs.LG, stat.CO, stat.ML); v1 dated "
            f"{PAPER_DATE} in the submission history, announced in the 2608 batch. "
            f"Shows that XGBoost's performance is affected by vertical outliers and "
            f"leverage points, explores losses based on "
            f"{', '.join(ESTIMATORS)}estimators from robust regression, and reports "
            f"that a two-step procedure, {PAPER_BEST}, gives the best trade-off "
            f"between robustness and prediction accuracy. 30 pages plus 15 of "
            f"supplement, 3 figures. <https://arxiv.org/abs/2608.13590>. **Read from "
            f"the abstract and the listing only**, so nothing here describes its "
            f"experiments, its data, or how its estimators are implemented.",
            f"Everything measured in this post is simulated. "
            f"n={N_TRAIN} training rows, {N_TEST} clean test rows, {N_FEATURES} "
            f"standard normal features of which two are irrelevant, mean function "
            f"`3 sin(x1) + x2^2 - 2 x3`, Gaussian noise of standard deviation "
            f"{NOISE_SD:g}, seed {SEED}. Contamination is constructed by "
            f"`standarderror.robust.contamination`. No market data and no company "
            f"appears.",
            f"XGBoost {__import__('xgboost').__version__}, defaults as shipped except "
            f"where stated: `max_depth={DEPTH}`, `learning_rate={LR}`, "
            f"`n_estimators={ROUNDS}`. The `huber_slope` default of "
            f"{DEFAULT_SLOPE:g} is XGBoost's, not a choice made here.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "scipy", "scikit-learn", "standarderror")),
            "equivariance_test": (
                "fit on (X, s*y), predict, divide by s, compare across s; reported as "
                "the largest root-mean-square deviation from the s=1 predictions "
                "relative to their own root-mean-square, over s in "
                "(0.1, 1, 10, 100)"),
            "equivariance_gaps": ", ".join(
                f"{k}: {v:.2g}" for k, v in res["gaps"].items()),
            "error_measure": ("RMSE against the true mean function on clean test "
                              "points, never against held-out contaminated data, "
                              "which would measure how well the outliers are "
                              "reproduced"),
            "slope_from_the_data": (
                f"1.4826 x MAD of the response, which is {res['mad_of_y']:.2f} on the "
                f"clean response and {res['mad_of_dirty_y']:.2f} on the contaminated "
                f"one — the inflation is why a one-shot rescaling is not the same "
                f"thing as a two-step procedure"),
            "numerical_floor": (
                f"at the smallest scale tested (1e-3) even the equivariant losses "
                f"move, by 1-2% for squared error and more for the Huber variants, "
                f"consistent with float32 arithmetic in the library rather than with "
                f"any property of the loss; the sweep is trustworthy from about 1e-2 "
                f"upward"),
            "leverage_construction": (
                f"a fraction of rows moved to +/- `distance` standard deviations in "
                f"the first feature only, with y set to the true mean at the new "
                f"location plus {MAGNITUDE:g}, so the points are bad leverage rather "
                f"than merely unusual"),
            "cost": (f"about {res['elapsed_seconds']:.0f} seconds of fitting for "
                     f"every grid in the post, cached under a hash of the "
                     f"configuration"),
        },
    )

    post.add("A paper about making XGBoost robust", f"""
XGBoost fits a sequence of small trees to the residuals of the ones before it. The
default loss is squared error, which means the thing each new tree chases is a
residual, and a residual that is enormous because the response was wrong gets chased
just as hard as one that is informative.

A paper posted this summer takes that seriously. It shows XGBoost's performance is
degraded by both classical kinds of contamination — **vertical outliers**, an ordinary
`x` with a wrong `y`, and **leverage points**, an unusual `x` whose `y` does not follow
the pattern — and then borrows the standard toolkit of robust regression:
{', '.join(ESTIMATORS)}estimators. Its conclusion is that a two-step procedure,
**{PAPER_BEST}**, gives the best trade-off between robustness and accuracy.

I could only read the abstract and the listing, so I will not describe its
experiments. What I can do is ask the question a reader of that abstract should ask:
XGBoost already ships a robust loss. Why was any of this necessary?

The answer is a number equal to 1.
""".strip())

    post.add("A robust loss is a function of the residual over a scale", f"""
Squared error is `r^2`. Absolute error is `|r|`. Neither contains a constant, and that
turns out to matter enormously.

Huber's loss does contain one. It is quadratic for small residuals and linear for
large ones, and the transition happens at `|r| = delta`. XGBoost implements a smooth
version, `reg:pseudohubererror`, as
**`delta^2 (sqrt(1 + (r/delta)^2) - 1)`** — quadratic near zero, linear far out, and
the gradient saturates at `delta`, which is exactly where the robustness comes from.
A point with a residual far beyond `delta` contributes a bounded push instead of a
proportional one.

Now look at what `delta` is. It is compared against a residual, so it is measured in
the units of the response. **It is a scale.** And robust statistics has always written
these losses as functions of `r / sigma` for that reason, with `sigma` estimated from
the data.

XGBoost's parameter is `huber_slope` and its default is **{DEFAULT_SLOPE:g}**.

So here is a test that costs nothing. Take a fitting procedure, multiply the response
by a constant `s`, refit, divide the predictions by `s`. If the procedure is any good
this must return the same function, because `s` is a choice of units and not a fact
about the world. Formally, **`A(X, s y)(x) / s = A(X, y)(x)`** — scale equivariance.

Squared error passes exactly. Absolute error passes exactly. A Huber loss with a fixed
`delta` cannot pass, because after rescaling the residuals moved and the transition
point did not.
""".strip())

    post.add(f"How much does it matter? A factor of {worst / best:.0f}", f"""
Enough to decide whether your model is robust at all.

The setup: a synthetic regression with a known mean function, so error is measured
against the truth rather than against a held-out sample of the same contamination.
{N_TRAIN} rows, {N_FEATURES} features of which two do nothing,
{100 * HEADLINE_FRACTION:.0f}% of responses shifted by {MAGNITUDE:g}. Then the sweep:
multiply the response by everything from a thousandth to a thousand, refit, divide the
predictions back.

At the response's natural scale, the default-slope Huber gives **{ct[i_one]:.2f}**
against squared error's {sq_flat:.2f} — robust, and the reason the option exists.
Express the same response in hundreds instead, and it gives **{ct[i_small]:.2f}**.
Squared error at the same point: {sq_flat:.2f}. **The robustness is simply gone**, and
nothing was done to the data that a change of measurement units does not do.

Across the whole sweep the default-slope Huber runs from {best:.2f} to {worst:.2f}, a
factor of {worst / best:.0f}. Squared error sits at {sq_flat:.2f} throughout, absolute
error at {abs_flat:.2f}, and the version that sets its slope from a robust scale of the
response at {fixed_flat:.2f}. Those three are flat to six decimal places: their
equivariance gaps are {res['gaps']['squared error']:.0e},
{res['gaps']['absolute error']:.0e} and {res['gaps'][FIXED_LOSS]:.0e}, against
**{res['gaps'][DEFAULT_LOSS]:.2f}** for the default.

One line of the fix, and it is the whole content of the paper's design choice:
estimate the scale. `huber_slope = 1.4826 * MAD(y)` restores equivariance exactly.
""".strip(), figures=[figs["units"]])

    post.add("The default is not simply wrong, which is worse", f"""
Here is where it stops being a bug report.

Run the contamination sweep at the response's natural scale and the default-slope
Huber is **the most robust setting tested**. At
{100 * res['fractions'][-1]:.0f}% corruption it reaches
{frac_last[DEFAULT_LOSS][-1]:.2f} while squared error reaches
{frac_last['squared error'][-1]:.2f} — {frac_last['squared error'][-1] / clean_squared:.1f}
times its own clean-data error of {clean_squared:.2f}. The paper's motivation
reproduces cleanly. And the setting that wins is the one Fig 1 shows to be robust by
accident.

That is because a `delta` which is *small* relative to the residual scale is
aggressively robust: almost every residual lands in the linear region, gradients
saturate, and outliers get very little say. A `delta` which is large is nearly squared
error. So the fixed default is not a wrong answer, it is an **unstated
bias-robustness trade-off**, and where you sit on it depends on the units you happened
to record your response in.

Which also means the well-behaved fix is not automatically the better model. Setting
`delta` from the MAD of the response gives {frac_last[FIXED_LOSS][-1]:.2f} at
{100 * res['fractions'][-1]:.0f}% corruption — equivariant, and *less* robust than the
mis-scaled default. The reason is instructive: the MAD of a contaminated response is
inflated **by the outliers themselves**, from {res['mad_of_y']:.2f} clean to
{res['mad_of_dirty_y']:.2f} corrupted, so it sets the transition point too high and
puts the outliers back inside the quadratic region.

The scale you need is the scale of the *residuals*. Which needs a fit. Which needs a
scale. That circularity is precisely why the paper's answer is a **two-step**
procedure and not a rescaling, and my one-shot control failing in exactly the
predicted way is the best evidence I can offer that its design is right.
""".strip(), figures=[figs["fraction"]])

    post.add("The leverage result I nearly got backwards", f"""
The paper's other claim is about leverage points, and reproducing it took two
attempts.

My first construction put the contaminated rows eight standard deviations out in
**every** feature. The damage was nil — squared error was, if anything, slightly more
accurate than on clean data. I had a paragraph half-written explaining that tree
ensembles are immune to leverage points before I checked the construction.

They are not immune. They are immune to *my* leverage points, and for a reason that is
obvious once seen: a point far outside the range of everything else falls beyond the
outermost split, gets a leaf of its own, and that leaf covers a region no test point
ever reaches. The tree fences it off for free.

Move the same {100 * HEADLINE_FRACTION:.0f}% of points in to
{res['distances'][0]:g} standard deviations and squared error goes to **{dist[0]:.2f}**
against {clean_squared:.2f} on clean data. At {res['distances'][-1]:g} standard
deviations the same contamination only reaches {dist[-1]:.2f}. **The danger falls as
the leverage rises.**

That inverts the linear-model intuition, where leverage is literally a distance and a
far-out point can take the fitted plane anywhere. For a tree the near point is the
dangerous one, because the splits that would isolate it also carve up territory that
real data occupies. Which has a practical edge to it: an outlier diagnostic borrowed
from linear regression — flag the points with high leverage — ranks danger in close to
the wrong order for a boosted tree.
""".strip(), figures=[figs["leverage"]])

    post.add("And robustness here is partly a round budget", f"""
One more thing worth knowing before reading anyone's robustness table, including mine.

{budget_body}

Every column rises. A tree can always carve out a leaf for an outlier if you give it
enough rounds, so a robust loss **delays** overfitting to contamination rather than
preventing it. Squared error goes from
{res['budget_sweep']['squared error'][0]:.2f} at {res['budgets'][0]} rounds to
{res['budget_sweep']['squared error'][-1]:.2f} at {res['budgets'][-1]:,}; even
absolute error, which has a bounded gradient everywhere, drifts from
{res['budget_sweep']['absolute error'][0]:.2f} to
{res['budget_sweep']['absolute error'][-1]:.2f}.

So a robustness comparison at an unstated number of rounds is not a comparison, and
some of what looks like a robust loss is early stopping in disguise. This also
complicates the "just use more rounds" answer to the units problem. On clean data at
the natural scale, the default-slope Huber is best at {ROUNDS} rounds
({cl[i_one]:.2f}); at a hundred times the scale it starts undertrained
({catch['100.0']['300']:.2f}) and needs **{9000 // ROUNDS} times the rounds** to reach
{catch['100.0']['9000']:.2f}, still short of the {cl[i_one]:.2f} the well-scaled
version managed at {ROUNDS}. You can pay for the units mistake in compute, at a poor
exchange rate.
""".strip())

    post.add("What to take from it", f"""
**Set `huber_slope`, or standardise the response.** If you use
`reg:pseudohubererror` and leave the slope at 1, you have made an assumption about
your units, not a modelling choice. `1.4826 * MAD(y)` costs one line and restores
equivariance. Standardising `y` before fitting does the same thing and is easier to
remember.

**Run the equivariance test on anything with a tuning constant in it.** Multiply the
response by a hundred, refit, divide back, compare. It takes two fits and it catches
this whole class of bug — a hyperparameter with units, where the default was chosen
for someone else's data. Huber's `delta` is the clearest case; it is not the only one.

**Report the round budget with any robustness claim**, because Table 1's columns all
move.

**And do not carry linear-model outlier intuition into a tree.** For a boosted tree,
the leverage point near the edge of the data is the one to worry about, not the one
far outside it.

None of which is an argument against the paper — it is the argument *for* it, arrived
at from outside. The reason to reach for M-, S- and tau-estimators is not that Huber's
loss is the wrong shape. It is that a robust loss needs a scale, that the scale has to
come from the data, and that getting it from the data honestly requires more than one
pass. The paper's answer to that is two steps and a name; mine is a one-line MAD
rescaling that restores equivariance and then loses to the bug it fixed. That gap is
the whole subject of robust regression, and it is why {PAPER_BEST} is a two-step
procedure.
""".strip())

    post.add("Where to be careful", f"""
**One synthetic problem, one contamination model.** The mean function, the noise, the
outlier magnitude and the tree depth are all mine, and every number moves if they
move. The exact part is the equivariance argument: a loss with a fixed transition point
cannot be scale equivariant, and that is arithmetic rather than a measurement. The
*sizes* — a factor of {worst / best:.0f}, {dist[0] / clean_squared:.1f} times the clean
error at near leverage — are specific to this setup.

**My contamination magnitude is absolute, on purpose.** {MAGNITUDE:g} in the units of
the response at scale 1, not a multiple of the noise. Specifying contamination in
units of sigma would have made the experiment scale-free by construction and unable to
detect the thing it was built to detect.

**The sweep has a numerical floor.** At the smallest scale even the equivariant losses
move — squared error by about 1%, the Huber variants by more, which is what you would
expect if the pseudo-Huber Hessian underflows first. Read the sweep from about a
hundredth upward; the leftmost point is there as calibration, not as a result.

**I did not implement the paper's estimators.** There is no MM-XGBoost in this post,
no S-loss and no tau-loss. My control is a one-shot MAD rescaling, which is the
*shallowest* version of the paper's idea, and the fact that it restores equivariance
without matching the mis-scaled default's robustness is an argument for the deeper
version rather than a test of it.

**And the paper I have only read from the outside.** The abstract and the listing.
Nothing here characterises its experiments or its results, and the measurement above
would have been worth making whatever they turn out to be.
""".strip())
    return post


if __name__ == "__main__":
    compute(force=bool(os.environ.get("SERR_FORCE")))
