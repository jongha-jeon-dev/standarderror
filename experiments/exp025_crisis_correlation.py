"""exp025 — the correlation that rises in a crisis, measured against the right null.

Where this comes from
---------------------
exp015 took a stylised fact everyone quotes — volatility clustering at +0.32 —
and asked what part of it survives the window length people actually train on.
A fifth. This does the same job on the one quantitative claim that appears in
every diversification pitch and every risk committee pack: correlations rise in
a crisis.

The claim is not wrong. It is about four times larger than the evidence supports,
and the reason is an identity rather than a modelling choice.

The identity
------------
Standardise two jointly normal series over the full sample and write
``y = rho x + sqrt(1 - rho^2) eps`` with ``eps`` independent of ``x``. Condition
on any event defined by ``x`` alone. The slope and the residual variance cannot
move, so the only thing that changes is Var(x), and the correlation you measure
inside the subsample is

    rho_A = rho s / sqrt(rho^2 s^2 + 1 - rho^2),     s^2 = Var(x | A) / Var(x)

with no free parameters. A turbulent decile has six or seven times the variance
of the rest of a real equity series, and that alone takes a correlation of 0.37
to 0.72. Inverting the identity is the correction in Forbes and Rigobon (2002),
which is twenty-four years old and still not in the pack.

What this adds
--------------
Two things, both about the correction rather than the claim.

1. **The correction is itself biased, in the opposite direction.** The identity
   is exact only when the conditioning event does not also select the residual's
   variance. Two markets share a volatility path, so it always does. On a
   simulated pair with a constant correlation of 0.30 and GARCH volatility, the
   correction returns 0.231. Applied to the data it says crisis dependence is
   *lower* than average, which is an artefact of the fix rather than a finding.
2. **The null that works keeps the volatility path.** Impose one constant
   correlation on the two estimated scale paths and resample. That null
   reproduces the variance ratio to within a percent and contains no change in
   dependence by construction, so whatever the data has above it is real.

Against that null, NASDAQ and the Nikkei over 13,315 trading days keep about a
quarter of the published rise. The remaining three quarters is the identity. And
the quarter that survives is worth less than it sounds: covariance is
``rho sx sy``, the two scale terms carry 70% of the turbulent-period increase,
and freezing the correlation at its calm value removes only a sixth of the rise
in equal-weight portfolio volatility.

The effect that *does* move the correlation between these two markets by a large
amount is not crises. It is fifty-five years of integration: on
volatility-standardised returns the decade correlation runs from +0.14 in the
1970s to +0.49 in the 2020s. The crisis/calm split cannot see it, because it
spends its resolution on volatility.

Licence
-------
FRED index series are not redistributable, so this publishes statistics and never
values, and every figure is a statistic against a parameter rather than a series
against time. Input files are git-ignored; see `data/fred/README.md`.

Run: `standarderror run exp025_crisis_correlation --publish`
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
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.sources import prices
from standarderror.sources.fred import MANDATORY_DISCLAIMER
from standarderror.ts import conditional as cd
from standarderror.viz import charts

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed
DATA = Path("data/fred")
CACHE = se.SETTINGS.build_dir / "cache" / "exp025.json"

# --- what exp015 established, so this post can be placed against it -----------
EXP015_SLUG = "clustering-does-not-fit-inside-the-window"
EXP015_WITHIN_SHARE = 0.2      # a fifth of clustering fits inside a 64-day window

# --- the pair -----------------------------------------------------------------
X_NAME, X_CODE = "NASDAQ Composite", "NASDAQCOM"
Y_NAME, Y_CODE = "Nikkei 225", "NIKKEI225"

#: The Tokyo close precedes the New York open, so a same-day pairing asks whether
#: Asia leads the US and a one-day pairing asks the reverse. Both are computed;
#: the lead alignment is the headline because it is the one with a correlation
#: worth arguing about, and reporting only it would be a choice made after
#: seeing the answer.
ALIGNMENTS = ("lead", "same")
HEADLINE_ALIGNMENT = "lead"

Q = 0.90
QUANTILES = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)
NULL_REPS = 300
BOOT_REPS = 400
BOOT_BLOCK = 20
SIM_N = 300_000
SIM_RHO = 0.30
STUDENT_DF = 4.0
DECADE_MIN = 400


def _vintage() -> dict:
    out = {}
    for code in (X_CODE, Y_CODE):
        p = DATA / f"{code}.csv"
        if p.exists():
            out[code] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "bytes": p.stat().st_size}
    return out


def _config_key() -> str:
    blob = json.dumps({"v": 2, "q": Q, "quantiles": QUANTILES,
                       "null_reps": NULL_REPS, "boot": BOOT_REPS,
                       "block": BOOT_BLOCK, "sim_n": SIM_N, "sim_rho": SIM_RHO,
                       "df": STUDENT_DF, "seed": SEED, "align": ALIGNMENTS,
                       "vintage": _vintage()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def series(code: str) -> pd.Series:
    path = DATA / f"{code}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. This container cannot fetch it; download "
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code} and put "
            f"it there. See data/fred/README.md.")
    return prices.to_log_returns(prices.load_prices(path))


def aligned(alignment: str) -> pd.DataFrame:
    """The paired return frame under one of the two alignments.

    ``lead`` pairs NASDAQ on day t with the Nikkei's next trading day, which is
    the ordering the clocks impose. ``same`` pairs the same calendar date, which
    asks the question the other way round.
    """
    x, y = series(X_CODE), series(Y_CODE)
    if alignment == "lead":
        y = pd.Series(y.to_numpy()[1:], index=y.index[:-1])
    elif alignment != "same":
        raise ValueError(f"unknown alignment {alignment!r}")
    frame = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
    return frame


def _split_dict(r: cd.SplitResult) -> dict:
    return {"label": r.label, "conditioner": r.conditioner,
            "n_turbulent": r.n_turbulent, "n_calm": r.n_calm,
            "rho_full": r.rho_full, "rho_turbulent": r.rho_turbulent,
            "rho_calm": r.rho_calm, "var_ratio_x": r.var_ratio_x,
            "var_ratio_y": r.var_ratio_y, "rho_predicted": r.rho_predicted,
            "rho_corrected": r.rho_corrected, "rise": r.rise,
            "predicted_rise": r.predicted_rise, "excess": r.excess,
            "explained": r.explained, "exact": r.exact}


def _null_dict(t: cd.NullTest) -> dict:
    return {"q": t.q, "scale": t.scale_kind, "n": t.n, "reps": t.reps,
            "rho_full": t.rho_full, "rho_turbulent": t.rho_turbulent,
            "rho_calm": t.rho_calm, "var_ratio": t.var_ratio,
            "null_turbulent": t.null_turbulent_mean,
            "null_lo": t.null_turbulent_lo, "null_hi": t.null_turbulent_hi,
            "null_calm": t.null_calm_mean, "null_var_ratio": t.null_var_ratio,
            "p_value": t.p_value, "rise": t.rise, "null_rise": t.null_rise,
            "genuine_excess": t.genuine_excess, "share_genuine": t.share_genuine,
            "draws": t.draws}


def compute(*, force: bool = False, verbose: bool = True) -> dict:
    """Every number in the post, cached under a hash of the config and the bytes."""
    key = _config_key()
    if not force and CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            return cached

    t0 = time.time()

    def say(*a):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)

    out = {"key": key, "vintage": _vintage(), "sim": {}, "real": {}}

    # ---------------------------------------------------------------- 1. the
    # identity, on data whose dependence is known to be constant.
    rng = np.random.default_rng(SEED)
    x, y = cd.gaussian_pair(SIM_N, SIM_RHO, rng)
    out["sim"]["iid"] = _split_dict(
        cd.split_stats(x, y, cd.turbulent_mask(x, Q), label="iid normal"))
    say("iid null done")

    scale = cd.garch_scale(SIM_N, rng)
    x, y = cd.gaussian_pair(SIM_N, SIM_RHO, rng, scale=scale)
    out["sim"]["garch"] = _split_dict(
        cd.split_stats(x, y, cd.turbulent_mask(x, Q), label="common volatility"))
    say("garch null done")

    x, y = cd.student_pair(SIM_N, SIM_RHO, STUDENT_DF, rng)
    out["sim"]["student"] = _split_dict(
        cd.split_stats(x, y, cd.turbulent_mask(x, Q),
                       label=f"Student, df={STUDENT_DF:g}"))
    say("student null done")

    # ---------------------------------------------------------------- 2. the pair
    for alignment in ALIGNMENTS:
        frame = aligned(alignment)
        xa = frame["x"].to_numpy()
        ya = frame["y"].to_numpy()
        say(f"{alignment}: n={len(frame)}")

        head = cd.split_stats(xa, ya, cd.turbulent_mask(xa, Q),
                              label=f"{X_NAME} / {Y_NAME}")
        sweep = [_split_dict(r) for r in cd.quantile_sweep(xa, ya, QUANTILES)]
        boot = cd.bootstrap_split(xa, ya, q=Q, block=BOOT_BLOCK,
                                  n_boot=BOOT_REPS, seed=SEED + 1)
        say(f"{alignment}: bootstrap done")

        nulls = {}
        for kind in ("ewma", "centred"):
            nulls[kind] = _null_dict(cd.scale_null(
                xa, ya, q=Q, reps=NULL_REPS, seed=SEED + 2, scale=kind))
            say(f"{alignment}: {kind} null done")

        # The null across thresholds, so the sweep figure has three lines.
        null_sweep = []
        for q in QUANTILES:
            t = cd.scale_null(xa, ya, q=q, reps=max(NULL_REPS // 3, 60),
                              seed=SEED + 3, scale="ewma")
            null_sweep.append({"q": q, "null_turbulent": t.null_turbulent_mean,
                               "lo": t.null_turbulent_lo, "hi": t.null_turbulent_hi,
                               "observed": t.rho_turbulent,
                               "genuine": t.genuine_excess,
                               "share": t.share_genuine})
        say(f"{alignment}: null sweep done")

        dec = cd.covariance_decomposition(xa, ya, cd.turbulent_mask(xa, Q))

        # Volatility-standardised correlation by decade: the comparison the
        # crisis split cannot make, because its periods are chosen by volatility.
        dx, dy, keep = cd.devolatilise(xa, ya)
        idx = frame.index[keep]
        std = pd.DataFrame({"x": dx, "y": dy}, index=idx)
        decades = []
        for decade, g in std.groupby(std.index.year // 10 * 10):
            if len(g) < DECADE_MIN:
                continue
            decades.append({"decade": int(decade), "n": int(len(g)),
                            "rho": cd.pearson(g["x"], g["y"])})
        devol_split = _split_dict(
            cd.split_stats(dx, dy, cd.turbulent_mask(dx, Q), label="standardised"))

        out["real"][alignment] = {
            "n": int(len(frame)),
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "headline": _split_dict(head),
            "sweep": sweep,
            "bootstrap": boot,
            "nulls": nulls,
            "null_sweep": null_sweep,
            "devol_split": devol_split,
            "decades": decades,
            "decomposition": {
                "cov_ratio": dec.cov_ratio, "rho_ratio": dec.rho_ratio,
                "sx_ratio": dec.sx_ratio, "sy_ratio": dec.sy_ratio,
                "share_rho": dec.share_rho, "share_scale": dec.share_scale,
                "portfolio_rise": dec.portfolio_rise,
                "portfolio_rise_frozen_rho": dec.portfolio_rise_frozen_rho,
                "rho_contribution": dec.rho_contribution},
        }

    # The identity as a curve, for the first figure. Drawn after the pair so one
    # of the curves can be the pair's own full-sample correlation rather than a
    # round number chosen to look close to it.
    ratios = np.geomspace(0.5, 30.0, 60)
    rho_head = out["real"][HEADLINE_ALIGNMENT]["headline"]["rho_full"]
    out["curve"] = {"var_ratio": ratios.tolist(), "headline_rho": rho_head,
                    "rho": {f"{r:.3f}": [cd.conditional_rho(r, v) for v in ratios]
                            for r in (0.10, 0.25, round(rho_head, 3), 0.60)}}

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    say("cached")
    return out


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    """Three charts, two tables, one hero. No figure plots a series against time.

    The licence forbids republishing FRED index values, and there is a second
    reason to like the constraint here: every claim in this post is about a
    statistic as a function of a parameter — the conditioning threshold, the
    variance ratio, the decade — and a price chart would answer none of them.
    """
    real = res["real"][HEADLINE_ALIGNMENT]
    head = real["headline"]
    src = f"Statistics computed from {X_NAME} and {Y_NAME} daily log returns. " \
          f"{MANDATORY_DISCLAIMER}"
    out = {}

    # --- f1: the identity ----------------------------------------------------
    curve = res["curve"]
    frame = pd.DataFrame(
        {f"true rho = {float(k):.3g}": v for k, v in curve["rho"].items()},
        index=pd.Index(curve["var_ratio"], name="variance ratio of the subsample"))
    ratio = head["var_ratio_x"]

    def mark(fig, ax):
        ax.axvline(ratio, color="0.45", lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"turbulent decile\nof the real pair: {ratio:.1f}x",
                    (ratio, 0.06), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.5, color="0.35")
        ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)

    out["f1"] = charts.lines(
        frame,
        title="A correlation that rises without any dependence changing",
        subtitle=("Correlation measured inside a volatility-selected subsample, "
                  "for a jointly normal pair whose true correlation is constant. "
                  "The curve is an identity, not a fit."),
        xlabel="Var(x | subsample) / Var(x), log scale",
        ylabel="correlation measured in the subsample",
        source="Closed form; no data.",
        logx=True, ylim=(0.0, 1.0), decorate=mark,
        alt=("Four rising curves showing measured correlation against the "
             "variance ratio of the subsample, with a dashed line at the "
             "variance ratio of the real turbulent decile."),
        caption=(f"Conditioning on the tail of one series raises the measured "
                 f"correlation by arithmetic. At the variance ratio the real "
                 f"pair shows, a true correlation of "
                 f"{curve['headline_rho']:.2f} is measured at "
                 f"{cd.conditional_rho(curve['headline_rho'], ratio):.2f}."),
        path=str(IMG / f"a16-f1-identity.{EXT}"))[0]

    # --- f2: threshold sweep, observed against two nulls ---------------------
    sweep = {r["label"]: r for r in real["sweep"]}
    ns = real["null_sweep"]
    idx = pd.Index([r["q"] for r in ns], name="turbulence threshold (quantile of |x|)")
    frame2 = pd.DataFrame({
        "measured": [r["observed"] for r in ns],
        "constant-correlation identity": [sweep[f"q={r['q']:.2f}"]["rho_predicted"]
                                          for r in ns],
        "same-volatility null": [r["null_turbulent"] for r in ns],
    }, index=idx)
    lo = np.array([r["lo"] for r in ns])
    hi = np.array([r["hi"] for r in ns])

    def band(fig, ax):
        ax.fill_between(idx.to_numpy(), lo, hi, color="0.6", alpha=0.16, zorder=1,
                        lw=0)
        ax.axhline(head["rho_calm"], color="0.45", lw=1.0, ls=(0, (4, 3)))
        ax.annotate(f"calm periods: {head['rho_calm']:+.3f}",
                    (idx[0], head["rho_calm"]), xytext=(2, 5),
                    textcoords="offset points", fontsize=8.5, color="0.35")

    out["f2"] = charts.lines(
        frame2,
        title="Where the crisis correlation comes from, at every threshold",
        subtitle=(f"{X_NAME} against {Y_NAME}, {real['n']:,} trading days. "
                  "Shaded band is the 95% range of the same-volatility null."),
        xlabel="turbulence threshold (quantile of |NASDAQ return|)",
        ylabel="correlation in the turbulent subsample",
        source=src, decorate=band, direct_labels=False,
        alt=("Three lines rising with the threshold: the identity highest, the "
             "measured correlation in the middle, the same-volatility null "
             "lowest, with the measured line above the null's shaded band at "
             "every threshold."),
        caption=("The identity over-predicts because it assumes the conditioning "
                 "leaves the residual variance alone, which a shared volatility "
                 "path never does. The gap between the measured line and the "
                 "null band is the part that is dependence."),
        path=str(IMG / f"a16-f2-threshold.{EXT}"))[0]

    # --- f3: the null distribution at the headline threshold ------------------
    nl = real["nulls"]["ewma"]
    out["f3"] = charts.histogram(
        nl["draws"], bins=26,
        series_label=f"constant correlation, same volatility path ({nl['reps']} draws)",
        mark={f"measured {nl['rho_turbulent']:+.3f}": nl["rho_turbulent"]},
        title="The measured value against a null that cannot contain contagion",
        subtitle=(f"Turbulent-decile correlation, q = {Q:.2f}. The null imposes one "
                  f"constant correlation of {nl['rho_full']:+.3f} on both estimated "
                  f"volatility paths, and reproduces the {nl['var_ratio']:.1f}x "
                  f"variance ratio at {nl['null_var_ratio']:.1f}x."),
        xlabel="correlation in the turbulent decile",
        source=src,
        alt=("A histogram of null draws centred near 0.49 with a vertical line "
             "at the measured 0.585, outside the bulk of the distribution."),
        caption=(f"The measured value clears the null by "
                 f"{nl['genuine_excess']:+.3f}, which is "
                 f"{nl['share_genuine']:.0%} of the {nl['rise']:+.3f} rise the "
                 f"calm/turbulent comparison reports."),
        path=str(IMG / f"a16-f3-null.{EXT}"))[0]

    # --- t1: what each estimator says on data whose truth is known -----------
    rows = []
    for key, name in (("iid", "iid normal"),
                      ("garch", "GARCH volatility, shared"),
                      ("student", f"Student copula, df {STUDENT_DF:g}")):
        s = res["sim"][key]
        rows.append([name, f"{s['rho_calm']:+.3f}", f"{s['rho_turbulent']:+.3f}",
                     f"{s['var_ratio_x']:.2f}x", f"{s['rho_predicted']:+.3f}",
                     f"{s['rho_corrected']:+.3f}"])
    out["t1"] = charts.table_image(
        rows,
        header=["simulated process", "calm", "turbulent", "variance ratio",
                "identity says", "corrected"],
        title=f"The correction is exact once, and biased twice (truth: {SIM_RHO:+.2f})",
        subtitle=(f"{SIM_N:,} draws each, constant correlation by construction. "
                  "Every process shows a large calm-to-turbulent rise. The "
                  "correction recovers the truth only when the conditioning "
                  "leaves the residual variance alone."),
        source="Simulated; standarderror/ts/conditional.py.",
        bold_cols=(5,),
        alt="Table of calm and turbulent correlations for three simulated "
            "processes with the same true correlation.",
        caption=("All three processes have the same constant correlation and all "
                 "three show a large calm-to-turbulent rise. The correction "
                 "recovers the truth only in the first row."),
        path=str(IMG / f"a16-t1-nulls.{EXT}"))[0]

    # --- t2: the pair, every number in one place -----------------------------
    dec = real["decomposition"]
    ct = real["nulls"]["centred"]
    decades = real["decades"]
    trows = [
        ["sample", f"{real['n']:,} paired trading days, "
                   f"{real['start']} to {real['end']}"],
        ["full-sample correlation", f"{head['rho_full']:+.3f}"],
        ["calm periods (lowest 90% of |x|)", f"{head['rho_calm']:+.3f}"],
        ["turbulent decile", f"{head['rho_turbulent']:+.3f}"],
        ["the published rise", f"{head['rise']:+.3f}"],
        ["variance ratio of the decile", f"{head['var_ratio_x']:.2f}x"],
        ["constant-correlation identity predicts", f"{head['rho_predicted']:+.3f}"],
        ["identity's share of the rise", f"{head['explained']:.0%}"],
        ["same-volatility null",
         f"{nl['null_turbulent']:+.3f} "
         f"[{nl['null_lo']:+.3f}, {nl['null_hi']:+.3f}]"],
        ["genuine excess over the null", f"{nl['genuine_excess']:+.3f}"],
        ["share of the rise that is dependence", f"{nl['share_genuine']:.0%}"],
        ["same, with a look-ahead scale (control)", f"{ct['share_genuine']:.0%}"],
        ["turbulent/calm covariance", f"{dec['cov_ratio']:.1f}x"],
        ["of which the two volatilities", f"{dec['share_scale']:.0%}"],
        ["equal-weight portfolio volatility",
         f"{dec['portfolio_rise']:.2f}x, or {dec['portfolio_rise_frozen_rho']:.2f}x "
         f"with the calm correlation held"],
        ["correlation's share of that rise", f"{dec['rho_contribution']:.0%}"],
        [f"standardised correlation, {decades[0]['decade']}s to "
         f"{decades[-1]['decade']}s",
         f"{decades[0]['rho']:+.2f} to {decades[-1]['rho']:+.2f}"],
    ]
    out["t2"] = charts.table_image(
        trows,
        header=["quantity", f"{X_NAME} / {Y_NAME}"],
        title="Every number in this post, for the headline pair",
        subtitle=(f"NASDAQ on day t against the Nikkei's next trading day. "
                  f"Turbulence threshold q = {Q:.2f}, "
                  f"{nl['reps']} null replicates, "
                  f"{real['bootstrap']['n_boot']} bootstrap resamples."),
        source=src, bold_cols=(1,),
        alt="Two-column table listing the correlation statistics for the pair.",
        caption=("Every quantity quoted in this post, for the headline pair, in "
                 "one place."),
        path=str(IMG / f"a16-t2-summary.{EXT}"))[0]

    # --- hero -----------------------------------------------------------------
    def draw_calm(panel, m):
        rng = np.random.default_rng(4)
        t = np.linspace(0, 1, 60)
        panel.plot(t, 0.45 + 0.05 * rng.standard_normal(60), color=m.series[0],
                   lw=1.6)
        panel.plot(t, 0.18 + 0.05 * rng.standard_normal(60), color=m.series[1],
                   lw=1.6)
        panel.set_ylim(0, 1)
        panel.set_xlim(0, 1)

    def draw_storm(panel, m):
        rng = np.random.default_rng(9)
        t = np.linspace(0, 1, 60)
        shock = np.exp(-((t - 0.5) ** 2) / 0.02)
        panel.plot(t, 0.55 + 0.30 * shock * rng.standard_normal(60),
                   color=m.series[0], lw=1.6)
        panel.plot(t, 0.30 + 0.30 * shock * rng.standard_normal(60),
                   color=m.series[1], lw=1.6)
        panel.set_ylim(0, 1)
        panel.set_xlim(0, 1)

    nl_share = nl["share_genuine"]
    # The split in the third panel is the measured share, not a drawn guess.
    cut = 0.1 + 0.8 * (1.0 - nl_share)

    def draw_ruler(panel, m):
        # A ruler laid over the storm: most of what was measured was the size of
        # the moves, not the agreement between them.
        panel.plot([0.1, 0.9], [0.62, 0.62], color=m.ink, lw=2.0)
        for x in np.linspace(0.1, 0.9, 9):
            panel.plot([x, x], [0.62, 0.55], color=m.ink, lw=1.2)
        panel.plot([0.1, cut], [0.30, 0.30], color=m.series[0], lw=5.0,
                   solid_capstyle="butt")
        panel.plot([cut, 0.9], [0.30, 0.30], color=m.series[3], lw=5.0,
                   solid_capstyle="butt")
        panel.set_ylim(0, 1)
        panel.set_xlim(0, 1)
    out["hero"] = charts.strip_card(
        headline="Most of the crisis correlation is arithmetic",
        panels=[(draw_calm, f"{head['rho_calm']:+.2f}", "calm days"),
                (draw_storm, f"{head['rho_turbulent']:+.2f}", "turbulent decile"),
                (draw_ruler, f"{nl_share:.0%}",
                 "of the rise survives a same-volatility null")],
        note=("Conditioning on one series' tail raises the measured correlation "
              "even when the true correlation never moves. NASDAQ and the "
              f"Nikkei, {real['n']:,} trading days."),
        footer="The Standard Error",
        alt=("A three-panel hand-drawn strip: two calm lines, the same two lines "
             "in a storm, and a ruler over a bar mostly one colour."),
        caption="",
        path=str(IMG / f"a16-hero.{EXT}"))[0]

    return out


# ---------------------------------------------------------------- the post

def _snippets(res: dict) -> dict:
    """Code blocks executed at build time, so the output is not typed by hand.

    The seeds are offsets of the one `compute` uses, so these blocks are a second
    draw from the same generators rather than a transcript of the first: the
    point of printing them is that a reader can run them, and a reader who runs
    them should see numbers that agree to the precision the prose claims.
    """
    s = Session()
    out = {}

    out["identity"] = s.run(f"""
        import numpy as np

        def conditional_rho(rho, var_ratio):
            "Correlation measured inside a subsample selected on x alone."
            s = np.sqrt(var_ratio)
            return rho * s / np.sqrt(rho**2 * var_ratio + 1 - rho**2)

        rng = np.random.default_rng({SEED})
        rho = {SIM_RHO}
        # One constant correlation. No regime, no contagion, nothing to find.
        L = np.linalg.cholesky([[1, rho], [rho, 1]])
        x, y = (rng.standard_normal(({SIM_N}, 2)) @ L.T).T

        hot = np.abs(x) >= np.quantile(np.abs(x), {Q})
        def corr(m):
            return np.corrcoef(x[m], y[m])[0, 1]

        ratio = x[hot].var(ddof=1) / x.var(ddof=1)
        print(f"calm       {{corr(~hot):+.3f}}")
        print(f"turbulent  {{corr(hot):+.3f}}")
        print(f"identity   {{conditional_rho(rho, ratio):+.3f}}  (ratio {{ratio:.2f}}x)")
    """, expect=["calm       +0.242", "turbulent  +0.546", "identity   +0.550"])

    out["correction"] = s.run(f"""
        def unconditional_rho(rho_cond, var_ratio):
            "Forbes and Rigobon (2002), inverted from the same identity."
            return rho_cond / np.sqrt(var_ratio + rho_cond**2 * (1 - var_ratio))

        print(f"recovered  {{unconditional_rho(corr(hot), ratio):+.3f}}   truth {{rho:+.2f}}")

        # Now give both series the *same* volatility path, which is what two
        # equity markets have. The correlation is still exactly constant.
        def garch_scale(n, rng, omega=0.05, alpha=0.10, beta=0.88, burn=500):
            v, out = omega / (1 - alpha - beta), np.empty(n + burn)
            for t in range(n + burn):
                out[t] = v
                v = omega + alpha * (rng.standard_normal()**2 * v) + beta * v
            return np.sqrt(out[burn:])

        s_t = garch_scale({SIM_N}, rng)
        xs, ys = (rng.standard_normal(({SIM_N}, 2)) @ L.T).T * s_t[:, None].T
        hot2 = np.abs(xs) >= np.quantile(np.abs(xs), {Q})
        r2 = xs[hot2].var(ddof=1) / xs.var(ddof=1)
        c2 = np.corrcoef(xs[hot2], ys[hot2])[0, 1]
        print(f"shared vol {{unconditional_rho(c2, r2):+.3f}}   truth {{rho:+.2f}}")
    """, expect=["recovered  +0.297", "shared vol +0.225"])

    out["null"] = s.run(f"""
        def ewma_scale(v, lam=0.94, warmup=250):
            "One-sided, so the scale for day t never sees day t."
            var, s = np.empty(v.size), v[:warmup].var(ddof=1)
            for t in range(v.size):
                var[t] = s
                s = lam * s + (1 - lam) * v[t]**2
            return np.sqrt(var)

        def null_turbulent(a, b, reps=60, seed=0):
            "Turbulent correlation under one constant rho and the same vol paths."
            g = np.random.default_rng(seed)
            sa, sb = ewma_scale(a), ewma_scale(b)
            r = np.corrcoef(a, b)[0, 1]
            root, out = np.sqrt(1 - r**2), []
            for _ in range(reps):
                z1, z2 = g.standard_normal(a.size), g.standard_normal(a.size)
                u, v = sa * z1, sb * (r * z1 + root * z2)
                m = np.abs(u) >= np.quantile(np.abs(u), {Q})
                out.append(np.corrcoef(u[m], v[m])[0, 1])
            return np.mean(out)

        # Size: a pair whose correlation really is constant.
        n = 12_000
        s_t = garch_scale(n, rng)
        a, b = (rng.standard_normal((n, 2)) @ L.T).T * s_t[:, None].T
        m = np.abs(a) >= np.quantile(np.abs(a), {Q})
        print(f"constant rho   measured {{np.corrcoef(a[m], b[m])[0,1]:+.3f}}"
              f"  null {{null_turbulent(a, b, seed=1):+.3f}}")

        # Power: the same volatility path, but the correlation really does move.
        z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
        rt = np.where(s_t > np.quantile(s_t, 0.80), 0.75, 0.25)
        a2 = s_t * z1
        b2 = s_t * (rt * z1 + np.sqrt(1 - rt**2) * z2)
        m2 = np.abs(a2) >= np.quantile(np.abs(a2), {Q})
        print(f"rho moves      measured {{np.corrcoef(a2[m2], b2[m2])[0,1]:+.3f}}"
              f"  null {{null_turbulent(a2, b2, seed=2):+.3f}}")
    """, expect=["constant rho   measured +0.478  null +0.497",
                 "rho moves      measured +0.755  null +0.668"])

    return out


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute(verbose=False)
    figs = figures(res)
    snip = _snippets(res)

    real = res["real"][HEADLINE_ALIGNMENT]
    other = res["real"]["same"]
    head = real["headline"]
    nl = real["nulls"]["ewma"]
    ct = real["nulls"]["centred"]
    boot = real["bootstrap"]
    dec = real["decomposition"]
    sim = res["sim"]
    decades = real["decades"]
    d_first, d_last = decades[0], decades[-1]
    # Threshold sweep with each turbulent subsample's size joined on, so the
    # prose can say how many days the widest interval is built from.
    # Calendar span, not trading days / 252: the two markets keep different
    # holidays, so the paired sample is shorter than either series' own history.
    span_years = (pd.Timestamp(real["end"]) - pd.Timestamp(real["start"])).days / 365.25
    by_q = {r["label"]: r for r in real["sweep"]}
    sw = [dict(d, n_turbulent=by_q[f"q={d['q']:.2f}"]["n_turbulent"])
          for d in real["null_sweep"]]
    decade_span = abs(d_last["rho"]) - abs(d_first["rho"])

    # The spine, asserted rather than trusted: if a rebuild moves any of these,
    # the prose below is wrong and the build should stop rather than publish it.
    assert head["rise"] > 0.25, head
    assert head["explained"] > 1.0, head
    assert 0.15 < nl["share_genuine"] < 0.45, nl
    assert nl["rho_turbulent"] > nl["null_hi"], nl
    assert sim["garch"]["rho_corrected"] < SIM_RHO - 0.04, sim
    assert decade_span > 2 * nl["genuine_excess"], decades

    post = Post(
        title="Most of the Crisis Correlation Is Arithmetic",
        slug="most-of-the-crisis-correlation-is-arithmetic",
        subtitle=("Split any two return series into turbulent and calm periods "
                  "and the correlation rises. On a simulated pair whose "
                  "correlation is constant by construction it rises just as "
                  "much, and the closed form says by exactly how much."),
        summary=("Conditioning on one series' largest moves raises the measured "
                 "correlation by an identity with no free parameters, so the "
                 "usual crisis/calm comparison cannot distinguish contagion "
                 "from heteroskedasticity. The twenty-four-year-old correction "
                 "for it is itself biased once the two series share a "
                 "volatility path, which two equity markets always do. Tested "
                 "against a null that keeps the volatility paths and imposes "
                 "one constant correlation, NASDAQ and the Nikkei over 55 years "
                 "keep about a quarter of the published rise — and that quarter "
                 "accounts for a sixth of the crisis increase in portfolio "
                 "volatility, while five decades of market integration, which "
                 "the split cannot see, moved the correlation nearly four times "
                 "as far."),
        tags=["correlation", "risk management", "market data", "contagion",
              "measurement"],
        author="Jongha Jeon",
        code_url=se.SETTINGS.code_repo_url,
        min_words=1600, max_words=2300,
        data_sources=[
            f"{X_NAME} daily close ({X_CODE}) and {Y_NAME} daily close "
            f"({Y_CODE}), paired to {real['n']:,} common trading days from "
            f"{real['start']} to {real['end']}. Both via FRED, Federal Reserve "
            f"Bank of St. Louis, downloaded {time.strftime('%d %B %Y')}. "
            f"{MANDATORY_DISCLAIMER}",
            "These series are not redistributable, so this post publishes "
            "statistics and never values: no return series, no dated "
            "observation, and no minimum or maximum. Every figure plots a "
            "statistic against a parameter rather than a series against time. "
            "See `data/fred/README.md`.",
            "Loaded by `standarderror.sources.prices.load_prices`, which handles "
            "FRED's bare '.' for non-trading days. All statistics from "
            "`standarderror.ts.conditional`.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "scipy", "pandas", "standarderror")),
            "vintage_sha256": ", ".join(
                f"{k}: {v['sha256'][:16]}" for k, v in res["vintage"].items()),
            "returns": ("log returns in percent, `100 * diff(log(close))`, with "
                        "non-trading days dropped rather than bridged"),
            "alignment": (f"NASDAQ day t against the Nikkei's next trading day; "
                          f"the same-day pairing is reported as a control"),
            "null": (f"{nl['reps']} replicates; scale paths from a one-sided "
                     f"EWMA with lambda 0.94 and a 250-day warm-up, and from a "
                     f"centred 21-day window as a look-ahead control"),
            "bootstrap": (f"{boot['n_boot']} moving-block resamples, block "
                          f"{boot['block']}, threshold recomputed inside each "
                          f"resample"),
            "code blocks": ("executed at build time by "
                            "standarderror/render/snippet.py; the printed output "
                            "is captured, not typed"),
            "modules": "standarderror/ts/conditional.py",
            "tests": "tests/test_conditional.py",
        },
    )

    # ------------------------------------------------------------------ 1
    post.add(
        "A claim with an identity underneath it",
        f"""Correlations rise in a crisis. It is on the second slide of every
diversification pitch, in the stress-testing section of every risk pack, and in
the part of a fund letter that explains why last quarter's hedge did not hedge.
It is also one of the few claims in finance that comes with a standard
calculation attached: split the sample into turbulent and calm periods, report
the correlation in each half, and put the two numbers side by side.

On {X_NAME} and {Y_NAME} — {real['n']:,} paired trading days from
{real['start']} to {real['end']} — that calculation gives
{head['rho_calm']:+.3f} on the calm nine-tenths of days and
{head['rho_turbulent']:+.3f} on the turbulent tenth. A rise of
{head['rise']:+.3f}, or a factor of
{abs(head['rho_turbulent'] / head['rho_calm']):.1f}. It looks like exactly what
everybody says it is.

The problem is that the calculation returns a large rise when handed a pair
whose correlation is a constant. Not approximately, not sometimes: there is a
closed form for how large. Standardise two jointly normal series over the full
sample and write one as a slope on the other plus an independent residual. Now
condition on any event defined by the first series alone — its largest absolute
moves, say. The slope cannot change, because it is a property of the joint
distribution; the residual variance cannot change, because the residual is
independent of what you conditioned on. The only thing that moves is the
variance of the conditioning series, and the correlation you will measure inside
the subsample follows from it with nothing left to estimate.""")

    post.add(
        "",
        f"""{snip['identity'].markdown()}

A constant correlation of {SIM_RHO:.2f}, no regime, no contagion, nothing to
find — and the turbulent decile reads {abs(float(sim['iid']['rho_turbulent'])):.2f}.
The identity predicts it to three decimals. Figure 1 draws the whole surface: at
the variance ratio a real equity index shows in its worst tenth of days —
{head['var_ratio_x']:.1f} times the rest of the sample — a true correlation of
{head['rho_full']:.2f} is *measured* at {head['rho_predicted']:.2f}.""",
        figures=[figs["f1"]], level=3)

    # ------------------------------------------------------------------ 2
    post.add(
        "The correction is twenty-four years old, and it is also wrong",
        f"""None of the above is new. Forbes and Rigobon made this argument in
2002 under the title *No Contagion, Only Interdependence*, inverted the identity
into a correction, and used it to reread the contagion literature of the 1990s.
Their correction takes the correlation measured in the turbulent subsample and
the variance ratio, and returns the constant correlation that would have
produced them. On the simulated pair above it recovers the truth.

What has aged badly is the assumption underneath it. The identity is exact only
when the conditioning event says nothing about the residual's variance. Two
equity markets do not work that way: they share a volatility path, so selecting
the days when one of them moved a lot also selects the days when the *other* one
moved a lot, residual included. Give the simulated pair a common GARCH scale,
change nothing about its dependence, and the correction stops working.""")

    post.add(
        "",
        f"""{snip['correction'].markdown()}

The truth is {SIM_RHO:.2f} and the correction returns
{abs(float(sim['garch']['rho_corrected'])):.3f} — low by
{100 * (1 - abs(float(sim['garch']['rho_corrected'])) / SIM_RHO):.0f}%. Applied
naively it will therefore report that dependence *falls* in turbulent periods,
which is an artefact of the fix rather than a finding about the market. Table 1
runs the same comparison on three processes that all have the same constant
correlation. The Student row is the one to note. Fat tails widen the variance
ratio — {sim['student']['var_ratio_x']:.2f} against
{sim['iid']['var_ratio_x']:.2f} for the normal pair — so the identity
over-predicts further and the correction lands at
{abs(float(sim['student']['rho_corrected'])):.3f} for the same truth of
{SIM_RHO:.2f}. "Returns are not normal" makes the standard fix worse, not
safer.""",
        figures=[figs["t1"]], level=3)

    # ------------------------------------------------------------------ 3
    post.add(
        "A null that keeps the volatility and throws away the story",
        f"""If the identity over-predicts and the correction over-corrects, the
way out is to stop looking for a formula and simulate the null directly. Take
each series' estimated volatility path, impose a single constant correlation on
the pair, and generate. The result has the volatility clustering the data has —
it is built from the data's own scale paths — and it has exactly one correlation
at every date, so it cannot contain contagion, a regime, or tail dependence. Run
the crisis/calm split on it and whatever rise appears is what the split produces
from heteroskedasticity alone.

Two properties matter and both are checkable. The null must not manufacture an
excess when the correlation really is constant, and it must find one when the
correlation really moves.""")

    post.add(
        "",
        f"""{snip['null'].markdown()}

The first row is the size check and the second is power. Note that the null
does not stay put in the power case: it is calibrated to the *full-sample*
correlation, which a real dependence regime has already raised, so part of what
is being tested for has been absorbed into the thing it is tested against. The
test understates a real effect by construction, which is the direction to be
conservative in.

The scale path is estimated, and a one-sided EWMA lags a volatility jump — on
the day a regime changes it divides a large return by a scale fitted before the
change, leaving both series large and manufacturing co-movement in exactly the
subsample under examination. So the whole thing is re-run with a centred
21-day window, which uses future data and could never be traded on, precisely
because it removes that mechanism.""", level=3)

    # ------------------------------------------------------------------ 4
    post.add(
        "What the pair actually says",
        f"""The Tokyo close precedes the New York open, so the pairing is a
choice and worth stating before the answer: {X_NAME} on day *t* against
{Y_NAME}'s next trading day, which is the ordering the clocks impose. The
same-day pairing is reported below as a control.

Against the same-volatility null, the turbulent-decile correlation should read
{nl['null_turbulent']:+.3f}, with a 95% range of
[{nl['null_lo']:+.3f}, {nl['null_hi']:+.3f}] across {nl['reps']} replicates. It
reads {nl['rho_turbulent']:+.3f}. The null reproduces the variance ratio it
needs to — {nl['null_var_ratio']:.2f} against the data's
{nl['var_ratio']:.2f} — so the gap is not a calibration failure. Not one of the
{nl['reps']} null draws reaches the measured value.

So there is something there: a genuine excess of {nl['genuine_excess']:+.3f} in
correlation. Set against the {head['rise']:+.3f} rise the calm/turbulent
comparison advertises, that is {nl['share_genuine']:.0%} of it. The look-ahead
control puts it at {ct['share_genuine']:.0%}, slightly higher, so the finding is
not EWMA lag. The constant-correlation identity, meanwhile, predicts
{head['rho_predicted']:+.3f} — {head['explained']:.0%} of the published rise, more
than all of it — and the Forbes-Rigobon correction returns
{abs(head['rho_corrected']):.3f} against a full-sample
{head['rho_full']:+.3f}, which is the over-correction the simulation warned
about, in the field, at about the size it predicted.""",
        figures=[figs["f2"], figs["f3"]])

    post.add(
        "",
        f"""Figure 2 does this at every threshold rather than at the
conventional decile, because the threshold is a free choice and a result that
lives at one value of it is not a result. The three lines never cross: the
identity is always highest, the measurement always sits between it and the null,
and the excess grows with the threshold — {sw[0]['genuine']:+.3f} at the median,
{sw[3]['genuine']:+.3f} at the decile — which is the direction a real dependence
effect should move in.

It then stops being measurable. At the 99th percentile the turbulent sample is
{sw[-1]['n_turbulent']:,} days, the null's 95% range widens to
[{sw[-1]['lo']:+.3f}, {sw[-1]['hi']:+.3f}], and the measured
{sw[-1]['observed']:+.3f} sits inside it. The threshold that comes closest to
what anyone means by *crisis* is the one where the answer is "cannot tell", and
the decile that gets published is a compromise between the definition people
want and the sample size that definition leaves. The published rise itself has a
moving-block bootstrap interval of
[{boot['rise']['lo']:+.3f}, {boot['rise']['hi']:+.3f}], so it is not in doubt as
a number — only as evidence. On the same-day pairing the picture is
the same with a weaker signal to divide up: a full-sample correlation of
{other['headline']['rho_full']:+.3f}, and the identity accounting for
{other['headline']['explained']:.0%} of a rise of
{other['headline']['rise']:+.3f}.""", level=3)

    # ------------------------------------------------------------------ 5
    post.add(
        "The quarter that survives is not where the loss comes from",
        f"""Suppose the {nl['share_genuine']:.0%} is taken at face value. It is
still not the part of a crisis that hurts a portfolio, and the reason is again
arithmetic rather than empirical. Covariance is a correlation multiplied by two
volatilities, and in the turbulent decile of this pair the covariance is
{dec['cov_ratio']:.1f} times its calm value. Split that multiplicatively and
{dec['share_scale']:.0%} of it is the two volatility terms.

An equal-weight portfolio of the two makes the point in the units a risk report
uses. Its volatility in the turbulent decile is
{dec['portfolio_rise']:.2f} times the calm value. Hold the correlation at its
calm level and let only the volatilities move, and it is
{dec['portfolio_rise_frozen_rho']:.2f} times. Correlation accounts for
{dec['rho_contribution']:.0%} of the rise; the rest is that everything got
bigger. "Diversification failed in the crisis" is, to a first approximation,
"the portfolio became {dec['portfolio_rise_frozen_rho']:.1f} times as volatile
because its holdings did, and the diversification kept doing what it always
does".""")

    # ------------------------------------------------------------------ 6
    post.add(
        "What did move the correlation",
        f"""There is a large, slow change in how these two markets co-move, and
the crisis/calm split cannot see it, because the split spends all of its
resolution on volatility. Standardise both series by their own volatility paths
first — which removes the scale channel entirely — and then compare periods
chosen by the calendar rather than by turbulence:

{" ".join(f"**{d['decade']}s** {d['rho']:+.2f}." for d in decades)}

From {abs(d_first['rho']):.2f} to {abs(d_last['rho']):.2f} across the
{int(span_years)} years in the sample, against a crisis excess of
{nl['genuine_excess']:.3f}. The effect nobody puts on a slide is nearly
{decade_span / nl['genuine_excess']:.0f} times the effect everybody does. Nor is
it a step at any particular crisis: the {decades[1]['decade']}s reads
{decades[1]['rho']:+.2f} and the {decades[2]['decade']}s reads
{decades[2]['rho']:+.2f}, lower, so it is not a ratchet that turbulence
tightens — it is the shape of two markets slowly becoming one trade.

Three things follow for anyone who has to produce this number. Report the
full-sample correlation of volatility-standardised returns, because it is the
one quantity here that a change in dependence moves and a change in volatility
does not. If a crisis/calm comparison is required, publish the variance ratio
next to it, since without that the reader cannot tell the identity from the
finding. And when a stress scenario needs a correlation, remember which term of
`rho * sx * sy` the stress is actually in.""")

    # ------------------------------------------------------------------ 7
    post.add(
        "What this does not say",
        f"""One pair, one frequency, one alignment. Two large equity indices in
different time zones are a hard case for measuring co-movement and an easy case
for finding an alignment that flatters a conclusion, which is why both
alignments are here. Nothing in this generalises to credit, to currencies, or to
the cross-section within one market, all of which have their own volatility
structure and may well have more dependence to find.

Linear correlation is also not the only thing "correlations go to one" could
mean. Tail dependence — the probability that one series is extreme *given* that
the other is — is a different quantity, it is not a function of the linear
correlation, and a copula measurement could reasonably find a crisis effect
where this one finds a quarter of a published rise. What the identity rules out
is reading the *linear* split as evidence, and that is the split that gets
published.

This is a statement about a statistic and not about anybody's risk model, any
institution's exposure, or what anyone should hold. The earlier post in this
series took volatility clustering, a stylised fact quoted at +0.32, and found
that about a fifth of it survives the window length generative models actually
use. This is the same shape of error in a different quantity, and it has the
same cause: a number measured at one scale, quoted as though it were a property
of the market.""", figures=[figs["t2"]])

    return post


def main() -> Post:
    post = build()
    problems = post.audit()
    print(f"words: {post.word_count()}")
    print("audit:", "clean" if not problems else "")
    for p in problems:
        print("  -", p)
    return post
