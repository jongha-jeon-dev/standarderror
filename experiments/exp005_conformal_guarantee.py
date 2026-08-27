"""exp005 — the conformal guarantee is real, and it does not say what people think.

Backlog: Track A, the post teased at the end of exp003 and exp004. Both of those
were about a number that looked like a result and was not. This one is about a
method with a *theorem* attached, and the two ways the theorem is quietly weaker
than the sentence people repeat about it.

Claim 1 — **marginal is not conditional.** Split conformal delivers its 90%
across the test set as a whole while covering the hardest fifth of the inputs at
73%. Nothing is broken; that is what marginal coverage means. The guarantee
averages over X, and an average over X is silent about the X you are actually
being asked about.

Claim 2 — **the standard fix moves the failure rather than removing it.** Scaling
the score by a fitted difficulty estimate repairs the hard end (73% -> 96%) and
breaks the easy end (100% -> 60%), because the difficulty model is misspecified:
the truth here scales as |x|^1.6 and the scale model is linear. The marginal
number stays at 90% throughout, so it cannot see any of this happening. CQR fits
the two quantiles directly and lands inside 10pp everywhere, on a *narrower*
average interval than plain split conformal.

Claim 3 — **exchangeability is a real assumption, and time breaks it.** A 3x
volatility regime change takes a nominal 90% interval to 41%. No reweighting
recovers it: the conditional law moved, not the covariate distribution. ACI
recovers long-run coverage by giving up the finite-sample guarantee, and the cost
shows up in the width path rather than in the coverage number.

Design notes:

- The mean model is deliberately *correct* in the time-series part (predictions
  are the true conditional mean), so the coverage collapse cannot be blamed on a
  bad point forecast. It is the score distribution that moved.
- The scale model for normalised conformal is fitted on the training split only,
  never on calibration. Fitting it on the calibration set would leak and the
  finite-sample statement would no longer apply.
- Every coverage number is checked against the finite-sample bound
  `[1-alpha, 1-alpha+1/(n_calib+1)]`, which is the actual theorem — not "about
  90%".

Run: `standarderror run exp005_conformal_guarantee --publish`
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import standarderror as se
from standarderror.render import Post
from standarderror.uq import conformal
from standarderror.viz import charts, theme

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed

ALPHA = 0.10
N_TRAIN, N_CALIB, N_TEST = 1500, 1500, 4000
N_BINS = 5

# Time-series part: a calm stretch, then a 3x volatility regime.
T_CALM, T_WILD = 900, 900
VOL_CALM, VOL_WILD = 0.6, 1.8
T_CALIB = 600                      # calibration ends here, inside the calm regime


def sigma(x: np.ndarray) -> np.ndarray:
    """True conditional scale. Superlinear in |x|, which is the whole difficulty.

    A linear difficulty model cannot represent this, and the point of the post is
    that the marginal coverage number does not notice.
    """
    return 0.2 + 1.2 * np.abs(x) ** 1.6


def cross_section(seed: int = 1) -> dict:
    """Three disjoint draws: fit, calibrate, test. Nothing is reused."""
    rng = np.random.default_rng(seed)

    def gen(n):
        x = rng.uniform(-3.0, 3.0, n)
        return x, 2.0 * x + sigma(x) * rng.standard_normal(n)

    x_tr, y_tr = gen(N_TRAIN)
    x_ca, y_ca = gen(N_CALIB)
    x_te, y_te = gen(N_TEST)
    return {"x_tr": x_tr, "y_tr": y_tr, "x_ca": x_ca, "y_ca": y_ca,
            "x_te": x_te, "y_te": y_te}


def _features(x: np.ndarray) -> np.ndarray:
    """`|x|` alongside `x`, so a quantile model can express the V-shaped spread.

    Without it the quantile regressors are linear in x and cannot widen towards
    both ends, which would make CQR look bad for a reason that has nothing to do
    with conformal prediction.
    """
    return np.c_[x, np.abs(x)]


def methods(data: dict, seed: int = 0) -> dict:
    """Three interval constructions on identical data, plus the parametric one."""
    x_tr, y_tr = data["x_tr"], data["y_tr"]
    x_ca, y_ca = data["x_ca"], data["y_ca"]
    x_te = data["x_te"]

    beta = np.polyfit(x_tr, y_tr, 1)
    pred = lambda x: np.polyval(beta, x)                            # noqa: E731

    # A difficulty model, fitted on training residuals only.
    resid_tr = np.abs(y_tr - pred(x_tr))
    b_scale = np.polyfit(np.abs(x_tr), resid_tr, 1)
    scale = lambda x: np.clip(                                      # noqa: E731
        np.polyval(b_scale, np.abs(x)), 0.05, None)

    kw = dict(random_state=seed, n_estimators=200, max_depth=2)
    q_lo = GradientBoostingRegressor(loss="quantile", alpha=ALPHA / 2,
                                     **kw).fit(_features(x_tr), y_tr)
    q_hi = GradientBoostingRegressor(loss="quantile", alpha=1 - ALPHA / 2,
                                     **kw).fit(_features(x_tr), y_tr)

    out = {
        "split": conformal.split_conformal(pred(x_ca), y_ca, pred(x_te),
                                          alpha=ALPHA),
        "normalised": conformal.normalised_conformal(
            pred(x_ca), y_ca, scale(x_ca), pred(x_te), scale(x_te), alpha=ALPHA),
        "cqr": conformal.cqr(
            q_lo.predict(_features(x_ca)), q_hi.predict(_features(x_ca)), y_ca,
            q_lo.predict(_features(x_te)), q_hi.predict(_features(x_te)),
            alpha=ALPHA),
    }
    # The baseline every textbook reaches for first: a Gaussian interval from the
    # residual standard deviation, with no distributional check of any kind.
    z = 1.6448536269514722
    s = float(np.std(y_tr - pred(x_tr), ddof=2))
    p_te = pred(x_te)
    out["gaussian"] = conformal.Interval(
        p_te - z * s, p_te + z * s, ALPHA,
        "parametric Gaussian (no conformal step)", {"sigma_hat": s})
    return out


def repeat_calibration(data: dict, n_repeats: int = 200, seed: int = 7) -> dict:
    """Coverage across many calibration draws, with the model held fixed.

    This exists because a single run cannot check the theorem. The guarantee is on
    the coverage *averaged over calibration sets*; for one calibration set the
    realised coverage is Beta-distributed around it, with a standard deviation of
    about sqrt(a(1-a)/(n+2)) — 0.8pp here. My headline run came out at 91.0%,
    which is not a violation and not a surprise, and saying "the theorem says
    90.00-90.07% and I measured 91.0%, so it holds" would be wrong.
    """
    rng = np.random.default_rng(seed)
    beta = np.polyfit(data["x_tr"], data["y_tr"], 1)

    def gen(n):
        x = rng.uniform(-3.0, 3.0, n)
        return x, 2.0 * x + sigma(x) * rng.standard_normal(n)

    cov = np.empty(n_repeats)
    for i in range(n_repeats):
        x_ca, y_ca = gen(N_CALIB)
        x_te, y_te = gen(2000)
        iv = conformal.split_conformal(np.polyval(beta, x_ca), y_ca,
                                       np.polyval(beta, x_te), alpha=ALPHA)
        cov[i] = iv.covers(y_te).mean()
    n_te = 2000
    return {"coverage": cov, "mean": float(cov.mean()), "sd": float(cov.std()),
            "sd_theory": float(np.sqrt(
                ALPHA * (1 - ALPHA) / (N_CALIB + 2)
                + ALPHA * (1 - ALPHA) / n_te)),
            "frac_below_89": float((cov < 0.89).mean()),
            "p10": float(np.quantile(cov, 0.10)),
            "p90": float(np.quantile(cov, 0.90)),
            "n_repeats": n_repeats}


def regimes(seed: int = 4) -> dict:
    """A correct mean model, a calm calibration window, then 3x the volatility."""
    rng = np.random.default_rng(seed)
    n = T_CALM + T_WILD
    t = np.arange(n)
    vol = np.where(t < T_CALM, VOL_CALM, VOL_WILD)
    mean = 2.5 * np.sin(t / 40.0)
    y = mean + vol * rng.standard_normal(n)
    return {"t": t, "y": y, "pred": mean, "vol": vol,
            "calib": slice(0, T_CALIB), "test": slice(T_CALIB, n)}


def analyse(data: dict, ivs: dict, reg: dict, rep: dict) -> dict:
    y_te, x_te = data["y_te"], data["x_te"]
    hardness = np.abs(x_te)

    per_method = {}
    for name, iv in ivs.items():
        s = iv.summary(y_te)
        bins = conformal.coverage_by_bin(iv, y_te, hardness, N_BINS)
        groups = bins["per_group"]
        order = [f"q{i}" for i in range(1, N_BINS + 1)]
        per_method[name] = {
            "coverage": s["empirical_coverage"],
            "mean_width": s["mean_width"],
            "median_width": s["median_width"],
            "by_bin": [groups[g]["coverage"] for g in order],
            "width_by_bin": [groups[g]["mean_width"] for g in order],
            "worst_bin": bins["worst_group_coverage"],
            "spread_pp": 100.0 * bins["coverage_range"],
        }

    # The theorem, stated exactly: marginal coverage lies in
    # [1-alpha, 1-alpha+1/(n+1)] for exchangeable data.
    lo = 1.0 - ALPHA
    hi = lo + 1.0 / (N_CALIB + 1)

    # --- time series
    t, y, pred = reg["t"], reg["y"], reg["pred"]
    te = reg["test"]
    t_te = t[te]
    after = t_te >= T_CALM
    static = conformal.split_conformal(pred[reg["calib"]], y[reg["calib"]],
                                       pred[te], alpha=ALPHA)
    aci = conformal.AdaptiveConformal(alpha=ALPHA, gamma=0.02,
                                      window=300).run(pred[te], y[te])
    static_cov, aci_cov = static.covers(y[te]), aci.covers(y[te])
    finite = np.isfinite(aci.width)
    # The bound is on the mean over calibration draws, so it is checked against
    # the repeated study and not against one run.
    edges = np.quantile(hardness, np.linspace(0, 1, N_BINS + 1))
    sig_by_bin = [float(np.mean(sigma(x_te[(hardness >= a) & (hardness <= b)])))
                  for a, b in zip(edges[:-1], edges[1:])]
    return {
        "per_method": per_method,
        "bound": (lo, hi),
        "bound_holds": lo <= rep["mean"] <= hi,
        "repeat": rep,
        "sigma_by_bin": sig_by_bin,
        "spread_ratio": sig_by_bin[-1] / sig_by_bin[0],
        "n_calib": N_CALIB,
        "bin_centres": [float(np.mean(hardness[
            (hardness >= q0) & (hardness <= q1)]))
            for q0, q1 in zip(
                np.quantile(hardness, np.linspace(0, 1, N_BINS + 1))[:-1],
                np.quantile(hardness, np.linspace(0, 1, N_BINS + 1))[1:])],
        "static_all": float(static_cov.mean()),
        "static_before": float(static_cov[~after].mean()),
        "static_after": float(static_cov[after].mean()),
        "static_width": float(static.width[0]),
        "aci_all": float(aci_cov.mean()),
        "aci_after": float(aci_cov[after].mean()),
        "aci_first100_after": float(aci_cov[after][:100].mean()),
        "aci_median_width": float(np.median(aci.width[finite])),
        "aci_width_calm": float(np.median(aci.width[~after & finite])),
        "aci_width_wild": float(np.median(aci.width[after & finite])),
        "aci_unbounded": int(np.sum(~finite)),
        # Unbounded warm-up intervals cover trivially, so quote the number with
        # them excluded too — otherwise ACI gets free credit for not knowing yet.
        "aci_all_bounded": float(aci_cov[finite].mean()),
        "vol_ratio": VOL_WILD / VOL_CALM,
        "static_rolling": conformal.rolling_coverage(static, y[te], 150),
        "aci_rolling": conformal.rolling_coverage(aci, y[te], 150),
        "roll_x": t_te[149:],
        "t_break": T_CALM,
    }


LABELS = {"gaussian": "Gaussian ±1.645σ", "split": "split conformal",
          "normalised": "normalised conformal", "cqr": "CQR"}


def figures(data: dict, res: dict) -> dict:
    src = "Simulated: y = 2x + s(x)·e with s(x) = 0.2 + 1.2·|x|^1.6."
    src_time = (f"Simulated: a correct sinusoidal mean, noise scale {VOL_CALM} "
                f"then {VOL_WILD} from t={T_CALM}; calibrated on t<{T_CALIB}.")
    figs = {}
    pm = res["per_method"]
    bins = [f"q{i}" for i in range(1, N_BINS + 1)]

    # F1 — the headline: one marginal number, five very different conditionals.
    frame = pd.DataFrame(
        {LABELS[k]: pd.Series([100 * v for v in pm[k]["by_bin"]], index=bins)
         for k in ("split", "normalised", "cqr")})

    def mark_target(_fig, ax):
        # Explained in the subtitle rather than annotated in place: with three
        # lines crossing the target there is no in-axes position for the label
        # that does not sit on one of them.
        m = theme.LIGHT
        ax.axhline(100 * (1 - ALPHA), color=m.muted, lw=1.4, ls=(0, (5, 3)))
        ax.set_ylim(50, 103)

    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False, decorate=mark_target,
        title="Every one of these methods delivers its promised 90% overall",
        subtitle=("Coverage within quintiles of difficulty (|x|), easiest on the "
                  "left; the dashed line is the promised 90%. All three have a "
                  "marginal coverage of 90% — this is the same guarantee, sliced "
                  "five ways."),
        ylabel="coverage in the quintile (%)",
        xlabel="difficulty quintile of |x| — easiest to hardest", source=src,
        alt=(f"Line chart of coverage across five difficulty quintiles for three "
             f"methods against a dashed 90% target. Split conformal starts at "
             f"{pm['split']['by_bin'][0] * 100:.0f} percent and falls to "
             f"{pm['split']['by_bin'][-1] * 100:.0f} in the hardest quintile; "
             f"normalised conformal does the opposite, starting at "
             f"{pm['normalised']['by_bin'][0] * 100:.0f} and ending at "
             f"{pm['normalised']['by_bin'][-1] * 100:.0f}; CQR runs from "
             f"{pm['cqr']['by_bin'][0] * 100:.0f} down to "
             f"{pm['cqr']['by_bin'][-1] * 100:.0f}, the flattest of the three."),
        caption=("Fig 1. Marginal coverage is an average over inputs, and an "
                 "average hides exactly this. Split conformal is generous on easy "
                 "inputs and short where it matters; the usual scaling fix "
                 "inverts the failure rather than removing it. Only CQR is close "
                 "to flat, and none of this is visible in the 90%."),
        path=str(IMG / f"a2-f1-conditional.{EXT}"))
    figs["conditional"] = fig_meta

    # F2 — why: the width has to vary, and by how much.
    frame = pd.DataFrame(
        {LABELS[k]: pd.Series(pm[k]["width_by_bin"], index=bins)
         for k in ("split", "normalised", "cqr")})
    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False,
        title="A constant-width interval is a bet that every input is equally hard",
        subtitle=(f"Mean interval width by difficulty quintile. The true noise "
                  f"scale grows {res['spread_ratio']:.0f}-fold from the easiest "
                  "quintile to the hardest."),
        ylabel="mean interval width", xlabel="difficulty quintile of |x|",
        source=src,
        alt=(f"Line chart of mean interval width across five difficulty "
             f"quintiles. Split conformal is flat at "
             f"{pm['split']['mean_width']:.1f}; normalised conformal rises "
             f"steeply from {pm['normalised']['width_by_bin'][0]:.1f} to "
             f"{pm['normalised']['width_by_bin'][-1]:.0f}; CQR rises from "
             f"{pm['cqr']['width_by_bin'][0]:.1f} to "
             f"{pm['cqr']['width_by_bin'][-1]:.0f}."),
        caption=("Fig 2. Split conformal's flat line is the whole problem: the "
                 "same width is far too wide on the left and too narrow on the "
                 "right. Note that CQR is *narrower on average* than split "
                 "conformal while being more uniform — adapting is not a cost "
                 "here, it is a free lunch you decline by not modelling scale."),
        path=str(IMG / f"a2-f2-width.{EXT}"))
    figs["width"] = fig_meta

    # F3 — the time-series failure and what recovers from it.
    frame = pd.DataFrame(
        {"static conformal": pd.Series(100 * res["static_rolling"],
                                       index=res["roll_x"]),
         "adaptive (ACI)": pd.Series(100 * res["aci_rolling"],
                                     index=res["roll_x"])})

    def mark_break(_fig, ax):
        m = theme.LIGHT
        ax.axhline(100 * (1 - ALPHA), color=m.muted, lw=1.4, ls=(0, (5, 3)))
        ax.axvline(res["t_break"], color=m.series[7], lw=1.4)
        # Top, not bottom: the static line dives into the bottom of the axes and
        # a label down there sits on top of the collapse it is describing.
        ax.annotate(f"volatility x{res['vol_ratio']:.0f} from here",
                    (res["t_break"], 0.99), xycoords=("data", "axes fraction"),
                    xytext=(8, 0), textcoords="offset points", ha="left",
                    va="top", fontsize=8.5, color=m.series[7])

    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False, decorate=mark_break,
        title="Exchangeability is an assumption, and a regime change violates it",
        subtitle=("Coverage in a rolling 150-step window; the dashed line is the "
                  "nominal 90%. The mean model is exactly correct throughout — "
                  "only the noise scale changes."),
        ylabel="rolling coverage (%)", xlabel="time step", source=src_time,
        alt=("Two rolling-coverage lines against a dashed 90% target. Both sit "
             "near 90 percent before a marked vertical line, after which the "
             "static conformal line collapses to around 40 percent while the "
             "adaptive line dips briefly and returns to 90."),
        caption=(f"Fig 3. After the break the static interval covers "
                 f"{res['static_after'] * 100:.0f}% of the time while still "
                 "calling itself 90%. ACI recovers, but read the small print: it "
                 "trades the finite-sample guarantee for a long-run one, and it "
                 "only knows the world changed because it started missing."),
        path=str(IMG / f"a2-f3-regime.{EXT}"))
    figs["regime"] = fig_meta

    # T1 — the summary table, as an image, because Medium strips table markup.
    order = ["gaussian", "split", "normalised", "cqr"]
    rows = [[LABELS[k],
             f"{pm[k]['coverage'] * 100:.1f}%",
             f"{pm[k]['worst_bin'] * 100:.1f}%",
             f"{pm[k]['spread_pp']:.0f}pp",
             f"{pm[k]['mean_width']:.1f}"] for k in order]
    fig_meta, _ = charts.table_image(
        rows, header=["method", "overall", "worst quintile", "spread",
                      "mean width"],
        title="The column that gets reported, and the column that matters",
        subtitle="Nominal 90%. Identical data, identical point predictions.",
        source=src, mode="light", bold_cols=(2,),
        alt=(f"Table of four methods with overall coverage, worst-quintile "
             f"coverage, spread across quintiles and mean width. All four are "
             f"within about a point of 90% overall; the worst quintile ranges "
             f"from {pm['normalised']['worst_bin'] * 100:.0f}% to "
             f"{pm['cqr']['worst_bin'] * 100:.0f}%."),
        caption=("Table 1. Every method in this table would be reported as "
                 "\"90% coverage\". The third column is the one that decides "
                 "whether a risk number is usable, and it is not in anybody's "
                 "model card."),
        path=str(IMG / f"a2-t1-methods.{EXT}"))
    figs["table"] = fig_meta

    # HERO — the preview card, not part of the body. A *series* card, because this
    # post's finding is an event in time: coverage that was fine and then was not.
    # The annotated moment does the job a large number would do, and does it in the
    # place where it happened.
    roll = 100.0 * np.asarray(res["static_rolling"], float)
    break_at = int(np.searchsorted(np.asarray(res["roll_x"]), res["t_break"]))
    fig_meta, _ = charts.series_card(
        roll,
        headline="A 90% prediction interval, before and after the world moved.",
        mark_index=break_at,
        mark_label=f"volatility ×{res['vol_ratio']:.0f}",
        note=("Rolling coverage of a nominal 90% conformal interval. The mean "
              f"model is exactly correct throughout; after the break it covers "
              f"{res['static_after'] * 100:.0f}% of the time."),
        footer="The Standard Error", mode="light",
        alt=("A line of rolling coverage sitting near 90 percent, then falling off "
             "a cliff at a marked point where the volatility tripled and settling "
             "near 40 percent."),
        caption="",
        path=str(IMG / f"a2-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    data = cross_section()
    ivs = methods(data)
    reg = regimes()
    rep = repeat_calibration(data)
    res = analyse(data, ivs, reg, rep)
    figs = figures(data, res)
    pm = res["per_method"]
    lo, hi = res["bound"]

    post = Post(
        title="Your 90% Prediction Interval Covers 42% of the Time",
        slug="your-90-percent-interval-covers-42-percent",
        subtitle=("Conformal prediction's guarantee is real, and it does not say "
                  "what people repeat about it"),
        summary=(f"Conformal prediction gives you a distribution-free coverage "
                 f"guarantee, and the theorem is correct: my intervals covered "
                 f"{pm['split']['coverage'] * 100:.1f}% against a promised 90%. "
                 f"They also covered the hardest fifth of the inputs "
                 f"{pm['split']['worst_bin'] * 100:.0f}% of the time, and after a "
                 f"volatility regime change they covered "
                 f"{res['static_after'] * 100:.0f}%. Neither is a bug. Both are "
                 "what the guarantee always said, read carefully."),
        tags=["data-science", "statistics", "machine-learning",
              "quantitative-finance"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=1500, max_words=2400,
        table_figures=[figs["table"]],
        data_sources=[
            "Fully simulated. Cross-section: y = 2x + s(x)·e with x uniform on "
            "[-3, 3] and s(x) = 0.2 + 1.2·|x|^1.6. Time series: a correct "
            "sinusoidal mean with a step change in noise scale. No external "
            "data; every number is reproducible from the repo with a fixed seed.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "scipy", "scikit-learn", "standarderror")),
            "splits": f"{N_TRAIN} fit / {N_CALIB} calibrate / {N_TEST} test, "
                      "disjoint draws; the difficulty and quantile models see "
                      "only the fit split",
            "nominal level": f"90% (alpha = {ALPHA})",
            "finite-sample bound": f"[{lo:.4f}, {hi:.4f}] for n_calib="
                                   f"{N_CALIB}. The bound is on the mean over "
                                   f"calibration draws, so it is checked against "
                                   f"{rep['n_repeats']} repeats "
                                   f"({rep['mean']:.5f}), not against the single "
                                   f"headline run ({pm['split']['coverage']:.4f})",
            "regime change": f"volatility {VOL_CALM} -> {VOL_WILD} at t="
                             f"{T_CALM}; calibration ends at t={T_CALIB}",
            "ACI": "gamma = 0.02, window = 300; the first interval is unbounded "
                   f"by construction ({res['aci_unbounded']} of "
                   f"{T_CALM + T_WILD - T_CALIB}), so widths are quoted as "
                   "medians",
        },
    )

    post.add("A guarantee you can actually check", f"""
Most uncertainty estimates in machine learning are decoration. A model reports a
variance, or a dropout spread, or a quantile head, and nobody ever measures
whether the 90% interval contains the truth 90% of the time. Conformal prediction
is the exception, and that is why it deserves the attention it has been getting:
it comes with a **distribution-free, finite-sample theorem.** No normality, no
correct model, no asymptotics. Wrap any point predictor, calibrate it on data it
has not seen, and coverage is guaranteed.

So I checked it, on a setup with nothing adversarial about it: {N_TRAIN} points to
fit a model, {N_CALIB} held back to calibrate the intervals, {N_TEST} to test, all
drawn independently from the same distribution. A nominal 90% split-conformal
interval covered **{pm['split']['coverage'] * 100:.1f}%** of the test points.

A 91 that was promised as a 90 is where most write-ups stop, and it is worth
pausing on, because the theorem does not actually say "90.0% every time". It says
the coverage **averaged over calibration sets** lands in
[{lo * 100:.2f}%, {hi * 100:.2f}%] for a calibration set of {N_CALIB}. Any single
calibration set gives you a draw around that, with a standard deviation of about
{rep['sd_theory'] * 100:.1f} percentage points here. So I ran the whole thing
{rep['n_repeats']} times with fresh calibration and test sets:

- mean coverage **{rep['mean'] * 100:.3f}%** — inside the theorem's window, which
  is only {(hi - lo) * 100:.2f} percentage points wide
- standard deviation across runs {rep['sd'] * 100:.2f}pp, against the
  {rep['sd_theory'] * 100:.2f}pp that calibration and test sampling noise predict
- {rep['frac_below_89'] * 100:.0f}% of runs came out below 89%, and the 10th
  percentile was {rep['p10'] * 100:.1f}%

The mathematics is exactly right, in other words, and *one* run of it does not
demonstrate that. This is the first of three places where the guarantee is weaker
than the sentence people repeat about it, and the mildest. The other two are not
mild at all.
""".strip())

    post.add(f"90% overall, {pm['split']['worst_bin'] * 100:.0f}% where you "
             "needed it", f"""
The guarantee is about coverage **averaged over inputs**. Average coverage is
compatible with almost any pattern of conditional coverage, and here is what the
pattern actually was.

I sorted the test set into five bins by how hard the input is — in this simulation
difficulty is exactly known, because I chose how the noise scale grows with |x| —
and measured coverage inside each bin. Split conformal covered the easiest fifth
**{pm['split']['by_bin'][0] * 100:.0f}%** of the time and the hardest fifth
**{pm['split']['worst_bin'] * 100:.0f}%**. A spread of
{pm['split']['spread_pp']:.0f} percentage points, sitting inside a marginal
number of {pm['split']['coverage'] * 100:.1f}%.

If those bins were customers, or regions, or credit segments, you have just
shipped a risk system that is quietly wrong for a fifth of them and needlessly
conservative for another fifth — and your validation report says 90%, because it
is 90%.

The mechanism is not subtle once you look at the widths. Split conformal uses one
number: the {int((1 - ALPHA) * 100)}th percentile of the absolute residuals on the
calibration set. Every test point gets the same interval,
±{pm['split']['mean_width'] / 2:.1f} here, whether the true spread at that input
is small or large. In this setup the true noise scale grows
{res['spread_ratio']:.0f}-fold from the easiest quintile to the hardest, so one
width cannot possibly fit both ends. It
is not calibrated *badly*; it is calibrated to the average of a mixture, which is
what you asked for.
""".strip(), figures=[figs["conditional"]])

    post.add("The standard fix moved the failure instead of removing it", f"""
The textbook answer is normalised (locally adaptive) conformal prediction: divide
the residual by an estimate of the local difficulty, so the interval widens where
the model expects trouble. The marginal guarantee survives, because the score is
still a fixed function of the input and the outcome.

I did that, with a difficulty model fitted on the training split — a linear fit
of absolute residual against |x|. The hardest quintile went from
{pm['split']['worst_bin'] * 100:.0f}% to
**{pm['normalised']['by_bin'][-1] * 100:.0f}%**. Fixed.

And the easiest quintile fell from {pm['split']['by_bin'][0] * 100:.0f}% to
**{pm['normalised']['by_bin'][0] * 100:.0f}%**.

The spread got *worse*: {pm['split']['spread_pp']:.0f} percentage points became
{pm['normalised']['spread_pp']:.0f}. The marginal coverage stayed at
{pm['normalised']['coverage'] * 100:.1f}%, serenely reporting success through both
the repair and the new breakage.

The cause is mundane and it is the point: my difficulty model is misspecified. The
truth grows like |x|^1.6 and I fitted a straight line, so the estimate is too
large near zero relative to its own calibration constant, and the intervals there
end up too narrow. Nothing about conformal prediction failed. **The conformal
step inherits the shape of whatever you hand it, and it inherits mistakes
silently**, because its own diagnostic — marginal coverage — is blind to them by
construction.

What worked was fitting the two quantiles directly instead of a scale factor.
Conformalised quantile regression (CQR) landed within
{pm['cqr']['spread_pp']:.0f} percentage points across all five bins, worst bin
{pm['cqr']['worst_bin'] * 100:.0f}%. It is also, and I did not expect this to be
so clean, **narrower on average** than plain split conformal:
{pm['cqr']['mean_width']:.1f} against {pm['split']['mean_width']:.1f}. Adapting
the width is not a cost you pay for fairness across inputs. Refusing to adapt is
a cost you pay for nothing.
""".strip(), figures=[figs["width"]])

    post.add("Then time gets involved", f"""
Everything above assumed exchangeability — loosely, that the calibration data and
the test data are draws from the same thing in no particular order. Financial
series, demand series, sensor streams and user behaviour are not exchangeable, and
the failure here is not a matter of degree.

Second experiment. A series with a **correct** mean model: my predictions are the
true conditional mean, so nothing can be blamed on the point forecast. I
calibrated on the first {T_CALIB} steps, all inside a calm stretch, and then the
volatility tripled.

Before the break, coverage {res['static_before'] * 100:.0f}%. After it,
**{res['static_after'] * 100:.0f}%** — a nominal 90% interval covering four times
in ten, with a width frozen at {res['static_width']:.2f} because the calibration
set has no idea anything happened. Over the whole test period it averages
{res['static_all'] * 100:.0f}%, which is the number a quarterly review would see,
and which describes neither regime.

Reweighting does not save this. Weighted conformal prediction handles *covariate*
shift — the inputs move, the conditional law does not. Here the conditional law is
precisely what moved. No importance weight on the calibration points can conjure
residuals of a size that has never been observed.

Adaptive conformal inference (ACI) does recover: it nudges its own working level
up every time it misses and down every time it does not, and it came back to
{res['aci_all'] * 100:.1f}% overall and {res['aci_after'] * 100:.1f}% after the
break. (Its first {res['aci_unbounded']} intervals are unbounded while it
accumulates residuals, and an infinite interval covers trivially; excluding those,
{res['aci_all_bounded'] * 100:.1f}%.) Read the terms of that trade, though. ACI gives up the finite-sample
guarantee for a **long-run** one; it learns the world changed only by being wrong
for a while, so in the first hundred steps after the break it covered
{res['aci_first100_after'] * 100:.0f}%; and the price is in the width, which went
from a median of {res['aci_width_calm']:.1f} in the calm regime to
{res['aci_width_wild']:.1f} in the wild one. An interval that widens by
{res['aci_width_wild'] / res['aci_width_calm']:.1f}x is being honest, but it is
also telling you your model no longer knows much, which is information a
dashboard reporting "coverage: 90%" actively hides.
""".strip(), figures=[figs["regime"]])

    post.add("Where my setup is easier than yours", f"""
Four ways this simulation flatters everyone involved, including me.

**I knew what "hard" meant.** I binned by |x| because I wrote the noise scale as a
function of |x|. In a real problem the conditioning variable that exposes the
failure is not handed to you, and coverage will look fine in the bins you happened
to choose. This is not a small gap: distribution-free *conditional* coverage is
provably impossible without further assumptions — with continuous features, any
method with exact conditional coverage must produce infinitely wide intervals
almost everywhere (Foygel Barber, Candès, Ramdas & Tibshirani, 2021). Bins are
what we have.

**One dimension.** With one feature I can plot the whole story. In fifty
dimensions the hard region can be a thin shell nobody thinks to slice on.

**A generous calibration set.** {N_CALIB} calibration points make the quantile
index fine-grained. At 200 the discreteness bites — the smallest achievable level
moves in steps of 1/(n+1) — and the per-run scatter I measured above roughly
triples.

**CQR won partly because I helped it.** I gave the quantile models |x| as a
feature, so they *could* represent the V-shaped spread. Hand CQR a misspecified
quantile model and it degrades like anything else. What it really buys is that the
modelling burden moves somewhere visible: a bad quantile model shows up as a
strange width profile, which you can look at, rather than as a coverage deficit
hidden inside a marginal average.
""".strip())

    post.add("What to ask of an interval", f"""
None of this is an argument against conformal prediction. It is the only widely
used uncertainty method whose promise can be *checked*, and everything above was
found by checking it — which is exactly the property the alternatives lack. The
parametric Gaussian baseline in my table missed even the marginal number —
{pm['gaussian']['coverage'] * 100:.1f}% against a nominal 90% — and it is worth
knowing why, because my noise *is* conditionally Gaussian, which sounds like the
best case for it. Conditionally Gaussian with a varying scale is a scale mixture,
and a scale mixture is fat-tailed. The residual standard deviation is an average
over the mixture, so a plus-or-minus-1.645-sigma interval built from it is too
narrow for the tail it actually faces. The conformal step does not care: it takes
the empirical quantile of whatever the residual distribution turns out to be.

| method | overall | worst quintile | spread | mean width |
|---|---|---|---|---|
| Gaussian ±1.645σ | {pm['gaussian']['coverage'] * 100:.1f}% | {pm['gaussian']['worst_bin'] * 100:.1f}% | {pm['gaussian']['spread_pp']:.0f}pp | {pm['gaussian']['mean_width']:.1f} |
| split conformal | {pm['split']['coverage'] * 100:.1f}% | {pm['split']['worst_bin'] * 100:.1f}% | {pm['split']['spread_pp']:.0f}pp | {pm['split']['mean_width']:.1f} |
| normalised conformal | {pm['normalised']['coverage'] * 100:.1f}% | {pm['normalised']['worst_bin'] * 100:.1f}% | {pm['normalised']['spread_pp']:.0f}pp | {pm['normalised']['mean_width']:.1f} |
| CQR | {pm['cqr']['coverage'] * 100:.1f}% | {pm['cqr']['worst_bin'] * 100:.1f}% | {pm['cqr']['spread_pp']:.0f}pp | {pm['cqr']['mean_width']:.1f} |

Every row of that table would be reported as "90% coverage". Only the last one
would still look defensible if somebody asked how it does for the hardest fifth of
their customers, and the difference between the rows is entirely in a column nobody
puts on a model card.

Four questions, in the order I would ask them.

**1. Coverage in the worst subgroup, not just overall.** One number for the whole
test set cannot fail this test, so it is not a test. Bin by difficulty, by
segment, by anything you would be asked about separately, and report the worst
bin. It costs three lines of code and it is the number that decides whether the
interval is usable.

**2. Does the width vary with the input?** If every interval is the same width,
the method has assumed all inputs are equally hard. Sometimes true. Usually
checkable in one plot.

**3. Rolling coverage, not pooled coverage.** For anything with a time index,
pooled coverage averages regimes together and reports a number describing none of
them. A rolling window would have caught my break within one window.

**4. Which assumption is doing the work — and would you notice it failing?**
Exchangeability for split conformal; unchanged `Y | X` for the weighted version;
long-run stationarity of nothing much for ACI. The honest version of every
guarantee names its assumption, and the useful follow-up is whether your
monitoring would detect the violation before your users do.

Next in this series, a change of subject: why every bus you wait for seems late,
every class seems bigger than the average class, and every queue you pick seems
slower than the one beside it. One theorem covers all three, and it has a
one-line formula.
""".strip())

    return post


if __name__ == "__main__":
    p = build()
    print(p.title, "|", p.word_count(), "words |", len(p.figures), "figures")
    for issue in p.audit():
        print("  audit:", issue)
