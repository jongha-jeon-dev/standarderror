"""Numerical Analysis 1: Your Learning Rate Is a Step Size.

Gradient descent is forward Euler on the gradient flow. That is an identity, not
an analogy, and it means the learning rate is a discretisation step -- with a
stability limit that has a closed form, an optimum that is not the limit, and, on
a network, a location that moves because the curvature moves with it.

Measured:

* On a quadratic the limit is exactly `2/lam_max`, and bisecting for it recovers
  the formula to a relative 3.4e-04. At exactly the critical step the sharpest
  direction is preserved to every digit for 400 steps; 1% above it the run
  reaches 3.9e+03, and the continuous flow being approximated has reached
  4.5e-05 by the same integration time.
* The optimum sits at `kappa/(kappa+1)` of the limit -- 99% at `kappa = 100` --
  so "raise it until it breaks, then back off" is nearly optimal and lands
  within a percent of a cliff.
* Momentum widens the limit by exactly `1 + beta`, recovered by bisection to
  8.1e-06 at `beta = 0.9`, and the tuned pair replaces `kappa` with `sqrt(kappa)`
  for a 10x speedup per digit.
* On a small tanh MLP under full-batch gradient descent, `2/lr` is a two-sided
  attractor for `lam_max`: pushed up from 7.36 to 20.0 at `lr = 0.1` and *down*
  to 4.0 at `lr = 0.5`, landing within 0.2% either way. Below `lr = 0.1` the
  sharpness plateaus under the boundary instead, so the edge is a regime rather
  than a law -- and the true limit is 2.04x the threshold you would compute at
  initialisation.

Run: `standarderror run lec101_step_size --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.numerics import steps as st
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
POST_DATE = date(2026, 9, 3)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Numerical Analysis for Machine Learning, Taught Through What Breaks"
SERIES_TAG = "Numerical Analysis"

#: The quadratic. 20 directions, condition number 100 -- ill-conditioned enough
#: to be realistic, well-conditioned enough that every number here is exact.
QUAD_D = 20
QUAD_COND = 100.0
QUAD_SEED = 0
SWEEP_STEPS = 400
MULTIPLES = (0.5, 0.9, 0.99, 1.0, 1.01, 1.1)

#: Condition numbers for the kappa/(kappa+1) claim.
CONDITIONS = (3.0, 10.0, 100.0, 1000.0)

#: Momentum coefficients to bisect the stability limit for.
BETAS = (0.0, 0.5, 0.9, 0.99)

#: The network. Full-batch on purpose: the edge-of-stability behaviour belongs
#: to the deterministic iteration, and minibatch noise blurs the boundary and
#: invites the reader to credit the noise for the whole effect.
NET_STEPS = 4000
NET_LRS = (0.02, 0.05, 0.1, 0.2, 0.5, 0.8)
NET_EDGE_BRACKET = (0.5, 0.8)


def compute() -> dict:
    H = st.quadratic_design(QUAD_D, condition=QUAD_COND, seed=QUAD_SEED)
    x0 = np.random.default_rng(QUAD_SEED).standard_normal(QUAD_D)
    crit = st.stability_limit(H)
    eigs = st.spectrum(H)

    def converges(lr, beta=0.0, steps=3000):
        hist = st.gd_quadratic(H, x0, lr, steps, beta=beta)
        return bool(np.isfinite(hist[-1]) and hist[-1] < hist[0])

    sweep = st.lr_sweep(H, MULTIPLES, steps=SWEEP_STEPS, seed=QUAD_SEED)
    # Trajectories for the figure, and the flow at the same integration times.
    paths = {m: st.gd_quadratic(H, x0, m * crit, SWEEP_STEPS) for m in MULTIPLES}
    flow = {m: st.gradient_flow(H, x0, SWEEP_STEPS * m * crit) for m in MULTIPLES}
    # A short run where Euler and the exact flow should agree, to show they are
    # the same object rather than two claims.
    small = 0.02 * crit
    agree = {"lr": small, "euler": float(st.gd_quadratic(H, x0, small, 50)[-1]),
             "exact": st.gradient_flow(H, x0, 50 * small)}

    _, q = np.linalg.eigh(H)
    marginal = abs(float((q.T @ (H @ x0))[-1]))

    plain_lo, plain_hi = st.divergence_threshold(converges, 0.01, 4.5, iters=40)

    momentum = []
    for beta in BETAS:
        lo, hi = st.divergence_threshold(
            lambda lr, b=beta: converges(lr, beta=b), 0.01, 4.5, iters=40)
        pred = st.momentum_limit(H, beta)
        momentum.append({"beta": beta, "bisected": lo, "predicted": pred,
                         "relative_error": abs(lo - pred) / pred})

    tuned = st.momentum_optimal(H)
    tuned_hist = st.gd_quadratic(H, x0, tuned["lr"], SWEEP_STEPS,
                                 beta=tuned["beta"])
    tuned_measured = float((tuned_hist[-1] / tuned_hist[0]) ** (1 / SWEEP_STEPS))

    # The step that diverges plain and trains with momentum. Not a coincidence:
    # 1.5 is inside (1, 1.9) and therefore inside the widened region.
    rescued_lr = 1.5 * crit
    rescued = {"lr": rescued_lr, "plain": converges(rescued_lr),
               "momentum": converges(rescued_lr, beta=0.9)}

    conditions = []
    for c in CONDITIONS:
        h = st.quadratic_design(QUAD_D, condition=c, seed=QUAD_SEED)
        conditions.append({
            "kappa": c, "limit": st.stability_limit(h),
            "optimal": st.optimal_lr(h),
            "fraction": st.optimal_lr(h) / st.stability_limit(h),
            "formula": c / (c + 1.0),
            "rate": st.optimal_rate(h),
            "per_decade": st.steps_per_decade(st.optimal_rate(h)),
        })

    power = st.power_iteration(lambda v: H @ v, QUAD_D)

    return {"H": H, "x0": x0, "eigs": eigs, "crit": crit, "sweep": sweep,
            "paths": paths, "flow": flow, "agree": agree, "marginal": marginal,
            "bisected": (plain_lo, plain_hi), "momentum": momentum,
            "tuned": tuned, "tuned_measured": tuned_measured,
            "rescued": rescued, "conditions": conditions, "power": power,
            "amplification": {m: st.amplification(H, m * crit)
                              for m in (0.5, 0.99, 1.01)}}


def compute_net() -> dict:
    """The network half. Separate because it needs torch and takes a minute."""
    lam0 = st.initial_sharpness()
    runs = {lr: st.edge_of_stability(lr, steps=NET_STEPS) for lr in NET_LRS}

    def ok(lr):
        return not st.edge_of_stability(lr, steps=400, probes=2)["diverged"]

    lo, hi = st.divergence_threshold(ok, *NET_EDGE_BRACKET, iters=10)
    return {"lam0": lam0, "naive": 2.0 / lam0, "runs": runs,
            "bracket": (lo, hi), "factor": lo / (2.0 / lam0)}


if __name__ == "__main__":
    import json

    r = compute()
    print("lam_max", r["eigs"][-1], "crit", r["crit"], "marginal", r["marginal"])
    for row in r["sweep"].rows:
        print(f"  {row['multiple']:.2f}x  lr={row['lr']:.6f}  "
              f"{row['final_grad_norm']:.3e}  flow {r['flow'][row['multiple']]:.3e}")
    print("agree", r["agree"])
    print("bisected", r["bisected"])
    print("momentum", json.dumps(r["momentum"], indent=1))
    print("tuned", r["tuned"], "measured", r["tuned_measured"],
          "per decade", st.steps_per_decade(r["tuned_measured"]))
    print("rescued", r["rescued"])
    for c in r["conditions"]:
        print("  ", c)
    print("power", r["power"])
    for m, a in r["amplification"].items():
        print(f"  amp {m}x: max {a.max():.6f} above1 {(a>1).sum()} min {a.min():.4f}")
    n = compute_net()
    print("\nlam0", n["lam0"], "naive", n["naive"], "bracket", n["bracket"],
          "factor", n["factor"])
    for lr, g in n["runs"].items():
        if g["diverged"]:
            print(f"  lr={lr}  diverged at {g['diverged_at']}")
        else:
            print(f"  lr={lr}  loss {g['loss']:.3e}  lam {g['lam_max']:.3f}  "
                  f"ratio {g['ratio']:.4f}  rose {g['rose_fraction']:.3f}  "
                  f"med {g['median_rise']*100:.2f}%  max {g['max_rise']*100:.2f}%  "
                  f"drop {g['tail_drop']:.1f}x")


# ------------------------------------------------------------------ figures

def figures(res: dict, net: dict) -> dict:
    out: dict = {}
    crit = res["crit"]

    # --- f0: the cliff, as trajectories ------------------------------------
    drawn = [m for m in MULTIPLES if m <= 1.01]
    frame = pd.DataFrame({
        f"{m:g}x critical": np.where(np.isfinite(res["paths"][m]),
                                     res["paths"][m], np.nan)
        for m in drawn})
    out["f0"] = charts.lines(
        frame,
        title="Two percent of the step size, seven orders of magnitude",
        subtitle=(f"Gradient descent on a {QUAD_D}-dimensional quadratic with "
                  f"condition number {QUAD_COND:.0f}. The critical step is "
                  f"2/λ_max = {crit:.3f}; every line is a multiple of it. The "
                  f"1.1× run is off this chart by 27 decades and is in the "
                  f"next one."),
        xlabel="step", ylabel="gradient norm", logy=True,
        source="Simulated; standarderror/numerics/steps.py.",
        alt=("Gradient norm against step number on a log scale. Lines below the "
             "critical step fall steadily; the line at exactly the critical "
             "step is flat; the two above it rise, the steepest leaving the top "
             "of the chart."),
        caption=(f"At {MULTIPLES[3]:g}× the critical step the run ends at "
                 f"{res['sweep'].at(1.0):.6f}, which is exactly the size of the "
                 f"gradient's component along the sharpest eigenvector — "
                 f"preserved to every digit for {SWEEP_STEPS} steps. One row up, "
                 f"at 0.99×, it ends at {res['sweep'].at(0.99):.2e}. One row "
                 f"down, at 1.01×, {res['sweep'].at(1.01):.2e}."),
        path=str(IMG / f"lec101-f0-cliff.{EXT}"))[0]

    # --- f1: Euler against the flow it approximates ------------------------
    def euler_vs_flow(ax, m):
        ms = np.array(MULTIPLES, dtype=float)
        euler = np.array([min(res["sweep"].at(k), 1e34) for k in MULTIPLES])
        flow = np.array([res["flow"][k] for k in MULTIPLES])
        ax.plot(ms, euler, marker="o", ms=5, lw=1.8, color=m.series[0],
                label="forward Euler (gradient descent)")
        ax.plot(ms, flow, marker="s", ms=5, lw=1.8, color=m.series[1],
                label="the exact gradient flow")
        ax.set_yscale("log")
        ax.axvline(1.0, color=m.ink, lw=1.4, ls=(0, (5, 3)))
        ax.annotate("critical step", (1.0, euler[-1] / 1e4),
                    textcoords="offset points", xytext=(-7, 0), ha="right",
                    fontsize=8.5, color=m.ink_secondary)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    out["f1"] = charts.diagram(
        euler_vs_flow,
        title="The divergence is in the discretisation, not the problem",
        subtitle=(f"Gradient norm after {SWEEP_STEPS} steps, against the exact "
                  f"solution of dx/dt = −∇f at the same integration time "
                  f"{SWEEP_STEPS}·lr. Same problem, same start."),
        xlabel="step size, as a multiple of 2/λ_max", ylabel="gradient norm",
        source="Simulated; standarderror/numerics/steps.py.",
        alt=("Two lines against step size on a log y-axis. The exact-flow line "
             "falls gently across the whole range. The Euler line tracks it "
             "closely at small multiples, then turns sharply upward past the "
             "critical step."),
        caption=(f"The two agree to "
                 f"{res['flow'][0.5] / res['sweep'].at(0.5):.2f}× at half the "
                 f"critical step — same object, as they must be. At 1.1× the "
                 f"flow has reached {res['flow'][1.1]:.2e} and Euler "
                 f"{res['sweep'].at(1.1):.1e}. Nothing about the problem "
                 f"changed between those two points on the x-axis."),
        path=str(IMG / f"lec101-f1-flow.{EXT}"))[0]

    # --- f2: the per-direction multipliers ---------------------------------
    def multipliers(ax, m):
        e = res["eigs"]
        for (mult, amp), colour in zip(res["amplification"].items(), m.series):
            ax.plot(e, amp, marker="o", ms=4, lw=1.8, color=colour,
                    label=f"lr = {mult:g}× critical")
        ax.axhline(1.0, color=m.ink, lw=1.6)
        ax.annotate("multipliers above this line grow", (e[0], 1.0),
                    textcoords="offset points", xytext=(4, 6), fontsize=8.5,
                    color=m.ink_secondary)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.18)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    amp101 = res["amplification"][1.01]
    out["f2"] = charts.diagram(
        multipliers,
        title="One scalar, twenty different multipliers",
        subtitle=("What one gradient step multiplies each eigendirection by: "
                  "|1 − lr·λ|. The same shape as ridge regression's "
                  "s²/(s² + α) from the last series, and just as invisible."),
        xlabel="eigenvalue of the Hessian", ylabel="|1 − lr·λ|",
        source="Simulated; standarderror/numerics/steps.py.",
        alt=("Three V-shaped curves of the per-direction multiplier against "
             "eigenvalue on a log x-axis, with a horizontal reference line at "
             "one. Only the topmost curve's right end rises above it."),
        caption=(f"At 1.01× the critical step exactly "
                 f"{(amp101 > 1).sum()} of {QUAD_D} directions has a multiplier "
                 f"above 1 — {amp101.max():.3f} — and the other "
                 f"{QUAD_D - int((amp101 > 1).sum())} are contracting hard. "
                 f"That is what the blow-up is made of: nineteen directions "
                 f"converging and one growing 1.02× a step, "
                 f"{SWEEP_STEPS} times."),
        path=str(IMG / f"lec101-f2-multipliers.{EXT}"))[0]

    # --- f3: where the optimum sits ----------------------------------------
    k1000 = next(c for c in res["conditions"] if c["kappa"] == 1000.0)

    def optimum(ax, m):
        # The axis runs well past the last point so its label has room to sit
        # to the right of it; a right-aligned label reached back under the
        # kappa=100 point and read as belonging to that one.
        ks = np.geomspace(1.5, 30000, 300)
        ax.plot(ks, ks / (ks + 1.0), lw=2.0, color=m.series[0],
                label="κ/(κ+1), the formula")
        pts = res["conditions"]
        ax.plot([c["kappa"] for c in pts], [c["fraction"] for c in pts],
                "o", ms=7, color=m.series[1], label="measured")
        # Only the two extreme points are labelled. Four labels on a log axis
        # whose right half is compressed collided in three different
        # arrangements before I stopped trying; the middle two are in the
        # caption instead, where nothing can overlap them.
        for c, (dx, dy, ha) in ((pts[0], (8, -10, "left")),
                                (pts[-1], (9, -10, "left"))):
            ax.annotate(f"κ={c['kappa']:.0f}: {c['per_decade']:.0f} steps/digit",
                        (c["kappa"], c["fraction"]), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=8.5,
                        color=m.ink_secondary)
        ax.set_xscale("log")
        ax.set_xlim(1.3, 30000)
        ax.set_ylim(0.5, 1.06)
        ax.axhline(1.0, color=m.grid, lw=1.4)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    out["f3"] = charts.diagram(
        optimum,
        title="The best step size sits on the cliff edge",
        subtitle=("The optimal step as a fraction of the largest stable one. "
                  "The harder the problem, the closer the optimum is to "
                  "divergence — and the less it buys you."),
        xlabel="condition number κ", ylabel="optimal lr ÷ stability limit",
        source="Simulated; standarderror/numerics/steps.py.",
        alt=("A curve rising towards one as the condition number grows on a log "
             "x-axis, with four measured points sitting exactly on it, each "
             "labelled with how many steps it costs to gain a decimal digit."),
        caption=(f"The formula and the measurement, on four problems: κ = 10 "
                 f"puts the optimum at 90.9% of the limit and costs 11 steps "
                 f"per digit, κ = 100 at 99.0% and 115 steps. At "
                 f"κ = {k1000['kappa']:.0f} the optimal step is "
                 f"{k1000['fraction'] * 100:.2f}% of the largest stable one — "
                 f"which is to say that on a badly conditioned problem there is "
                 f"no safe distance between the best learning rate and the one "
                 f"that diverges. And the reward for finding it is "
                 f"{k1000['per_decade']:.0f} steps per decimal digit."),
        path=str(IMG / f"lec101-f3-optimum.{EXT}"))[0]

    # --- f4: the edge of stability -----------------------------------------
    def edge(ax, m):
        colours = list(m.series) + [m.ink_secondary, m.ink]
        for (lr, g), colour in zip(net["runs"].items(), colours):
            if g["diverged"]:
                continue
            t = [p[0] for p in g["trace"]]
            r = [st.sharpness_ratio(p[2], lr) for p in g["trace"]]
            ax.plot(t, r, lw=1.8, color=colour, marker="o", ms=3,
                    label=f"lr = {lr:g}")
        ax.axhline(1.0, color=m.ink, lw=1.8)
        ax.annotate("the stability boundary, λ_max = 2/lr",
                    (NET_STEPS * 0.52, 1.0), textcoords="offset points",
                    xytext=(0, 9), fontsize=8.5, color=m.ink_secondary)
        ax.set_ylim(0, 2.0)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)

    r05 = net["runs"][0.5]
    out["f4"] = charts.diagram(
        edge,
        title="On a network the boundary is an attractor, from either side",
        subtitle=(f"λ_max·lr/2 during full-batch gradient descent on a tanh MLP, "
                  f"{NET_STEPS} steps. Curvature measured by power iteration on "
                  f"Hessian-vector products."),
        xlabel="step", ylabel="λ_max · lr / 2",
        source="Simulated; standarderror/numerics/steps.py.",
        alt=("Five curves of the sharpness ratio against training step. The two "
             "smallest learning rates rise and level off below one; the three "
             "largest converge onto one, the largest arriving from above."),
        caption=(f"The two smallest steps rise and plateau **below** the boundary "
                 f"— at {net['runs'][0.02]['ratio']:.3f} and "
                 f"{net['runs'][0.05]['ratio']:.3f}. The three largest end on it "
                 f"to within "
                 f"{max(abs(net['runs'][k]['ratio'] - 1) for k in (0.1, 0.2, 0.5)) * 100:.1f}"
                 f"%. And the lr = 0.5 run starts at "
                 f"{st.sharpness_ratio(net['lam0'], 0.5):.2f}, above the "
                 f"boundary, and is pushed **down** to "
                 f"{r05['ratio']:.3f}: λ_max falls from {net['lam0']:.2f} to "
                 f"{r05['lam_max']:.2f}, which is 2/lr."),
        path=str(IMG / f"lec101-f4-edge.{EXT}"))[0]

    out["hero"] = _hero(res, net)
    return out


def _hero(res: dict, net: dict):
    def a_cliff(panel, m):
        x = np.linspace(0, 1, 200)
        y = np.where(x < 0.72, -1.6 * x, 9.0 * (x - 0.72) - 1.15)
        panel.plot(x, y, color=m.ink, lw=2.6)
        panel.axvline(0.72, color=m.grid, lw=2.2)
        panel.set_ylim(-1.5, 2.0)

    def one_of_twenty(panel, m):
        e = np.geomspace(0.1, 10, 20)
        panel.plot(np.arange(len(e)), np.abs(1 - 1.01 * res["crit"] * e),
                   color=m.ink, lw=2.6, marker="o", ms=3)
        panel.axhline(1.0, color=m.grid, lw=2.2)
        panel.set_ylim(0, 1.15)

    def pulled_both_ways(panel, m):
        t = np.linspace(0, 1, 60)
        panel.plot(t, 1 - 0.75 * np.exp(-4 * t), color=m.ink, lw=2.6)
        panel.plot(t, 1 + 0.85 * np.exp(-4 * t), color=m.ink, lw=2.6)
        panel.axhline(1.0, color=m.grid, lw=2.4)
        panel.set_ylim(0.1, 2.0)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=1,
        headline="Your learning rate is a step size",
        panels=[
            (a_cliff, f"{res['sweep'].at(1.01) / res['sweep'].at(0.99):.0e}",
             "worse, 0.99× to 1.01×"),
            (one_of_twenty, f"{int((res['amplification'][1.01] > 1).sum())} of "
             f"{QUAD_D}", "directions growing"),
            (pulled_both_ways, f"{net['factor']:.2f}×", "the naive threshold"),
        ],
        note=("Gradient descent is forward Euler on the gradient flow — an "
              "identity, not an analogy. So the learning rate is a "
              "discretisation step, and it has a stability limit: on a "
              "quadratic, exactly 2/λ_max. On a network the limit is real but "
              "it is not where you would compute it, because the curvature "
              "adapts to whatever step you chose until it cannot."),
        alt=("A three-panel hand-drawn strip. The first shows a line falling "
             "gently then turning sharply upward past a vertical marker. The "
             "second shows a curve of multipliers with its right end crossing a "
             "horizontal line. The third shows two curves converging onto one "
             "horizontal line from above and below."),
        mode="light",
        path=str(IMG / f"lec101-hero.{EXT}"))[0]


# --------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["identity"] = s.run(f"""
        import numpy as np

        # A quadratic, so that every claim below has a closed form to check
        # against: f(x) = x'Hx/2, gradient Hx, Hessian H.
        d, kappa, seed = {QUAD_D}, {QUAD_COND}, {QUAD_SEED}
        eigs = np.geomspace(1.0, kappa, d) / np.sqrt(kappa)
        q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
        H = q @ np.diag(eigs) @ q.T

        def descend(H, x, lr, steps):
            for _ in range(steps):
                x = x - lr * (H @ x)          # forward Euler on dx/dt = -Hx
            return np.linalg.norm(H @ x)

        lam_max = np.linalg.eigvalsh(H)[-1]
        critical = 2 / lam_max
        x0 = np.random.default_rng(seed).standard_normal(d)

        print(f"lam_max = {{lam_max:.4f}}   2/lam_max = {{critical:.6f}}")
        for mult in {list(MULTIPLES)}:
            print(f"  {{mult:>5.2f}} x critical   "
                  f"{{descend(H, x0, mult * critical, {SWEEP_STEPS}):.3e}}")
    """, expect=["lam_max =", "1.10 x critical"])

    out["marginal"] = s.run("""
        # At exactly the critical step the sharpest direction's multiplier is
        # |1 - lr*lam_max| = |1 - 2| = 1. Nothing happens to it, ever.
        _, vecs = np.linalg.eigh(H)
        component = abs((vecs.T @ (H @ x0))[-1])
        after = descend(H, x0, critical, 400)
        print(f"component of the gradient along the top eigenvector: {component:.9f}")
        print(f"gradient norm after 400 steps at the critical step:  {after:.9f}")
    """, expect=["component of the gradient"])

    out["diagnostic"] = s.run("""
        # The one measurement that makes the threshold usable during a run.
        # Power iteration needs only Hessian-vector products, and one of those
        # costs about one extra backward pass -- so this is affordable every
        # few hundred steps, on a model of any size.
        def sharpness(hvp, n_params, iters=60, seed=0):
            v = np.random.default_rng(seed).standard_normal(n_params)
            v /= np.linalg.norm(v)
            lam = 0.0
            for _ in range(iters):
                w = hvp(v)                       # one HVP == one extra backward
                lam = float(v @ w)
                v = w / np.linalg.norm(w)
                                                 # (a real one stops on tol)
            return lam

        lam = sharpness(lambda v: H @ v, H.shape[0])
        for lr in (0.05, 0.15, 0.25):
            print(f"lr = {lr:.2f}   lam_max*lr/2 = {lam * lr / 2:.3f}   "
                  f"{'stable' if lam * lr / 2 < 1 else 'DIVERGES'}")
    """, expect=["DIVERGES"])

    out["net"] = s.run(f"""
        # The same loop on a network, and the same `sharpness` above -- only the
        # Hessian-vector product changes, from `H @ v` to one autograd call.
        # Nothing else here is unusual: a two-hidden-layer tanh MLP, mean
        # squared error, a fixed step, no momentum, no schedule, full batch.
        import torch

        torch.set_default_dtype(torch.float64)

        def train(lr, steps={NET_STEPS}):
            g = torch.Generator().manual_seed(0)
            net = torch.nn.Sequential(
                torch.nn.Linear(8, 40), torch.nn.Tanh(),
                torch.nn.Linear(40, 40), torch.nn.Tanh(),
                torch.nn.Linear(40, 1))
            for lin in [m for m in net if isinstance(m, torch.nn.Linear)]:
                with torch.no_grad():
                    fan_in = lin.weight.shape[1]
                    lin.weight.copy_(torch.randn(*lin.weight.shape, generator=g)
                                     / fan_in ** 0.5)
                    lin.bias.zero_()
            gd = torch.Generator().manual_seed(1)
            X = torch.randn(200, 8, generator=gd)
            y = (torch.sin(2 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]).unsqueeze(1)
            ps = list(net.parameters())

            def flat(loss, create_graph=False):
                gs = torch.autograd.grad(loss, ps, create_graph=create_graph)
                return torch.cat([a.reshape(-1) for a in gs])

            def loss_fn():
                return torch.nn.functional.mse_loss(net(X), y)

            for _ in range(steps):
                gvec = flat(loss_fn())
                with torch.no_grad():
                    i = 0
                    for prm in ps:
                        k = prm.numel()
                        prm -= lr * gvec[i:i + k].view_as(prm)
                        i += k

            # The HVP: differentiate `grad . v` once more.
            gr = flat(loss_fn(), create_graph=True)

            def hvp(v):
                w = torch.autograd.grad(gr @ torch.as_tensor(v), ps,
                                        retain_graph=True)
                return torch.cat([a.reshape(-1) for a in w]).detach().numpy()

            # 400, not the default 60: see the note below the block.
            return float(loss_fn().detach()), sharpness(hvp, gr.numel(),
                                                        iters=400)

        for lr in (0.05, 0.2, 0.5):
            loss, lam = train(lr)
            print(f"lr = {{lr:.2f}}   loss {{loss:.2e}}   lam_max {{lam:6.2f}}   "
                  f"lam_max*lr/2 = {{lam * lr / 2:.3f}}")
    """, expect=["lr = 0.50"])

    return out


# ------------------------------------------------------------------- build

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    net = compute_net()
    figs = figures(res, net)
    snip = _snippets(res)

    # The spine, asserted rather than trusted.
    crit, sweep = res["crit"], res["sweep"]
    assert abs(sweep.at(1.0) - res["marginal"]) / res["marginal"] < 1e-8
    assert sweep.at(1.01) / sweep.at(0.99) > 1e6
    assert abs(res["bisected"][0] - crit) / crit < 1e-3
    assert (res["amplification"][1.01] > 1).sum() == 1
    for c in res["conditions"]:
        assert abs(c["fraction"] - c["formula"]) < 1e-9, c
    for row in res["momentum"]:
        assert row["relative_error"] < 1e-3, row
    assert not res["rescued"]["plain"] and res["rescued"]["momentum"]
    assert res["power"]["converged"] and res["power"]["iterations"] < 80
    # The network half, including the claim the backlog got wrong.
    assert net["runs"][0.02]["ratio"] < 0.35 and net["runs"][0.05]["ratio"] < 0.85
    for lr in (0.1, 0.2, 0.5):
        assert abs(net["runs"][lr]["ratio"] - 1.0) < 0.015, lr
    assert net["lam0"] > 2.0 / 0.5, "the lr=0.5 run must start ABOVE the boundary"
    assert net["runs"][0.8]["diverged"]
    assert 1.9 < net["factor"] < 2.2
    assert net["runs"][0.05]["rose_fraction"] == 0.0
    assert net["runs"][0.2]["rose_fraction"] > 0.2

    post = Post(
        title=f"{SERIES_TAG} 1: Your Learning Rate Is a Step Size",
        slug="numerical-analysis-1-step-size",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=1,
        date=POST_DATE,
        subtitle=("Gradient descent is forward Euler on the gradient flow, so "
                  "the learning rate is a discretisation step with a stability "
                  "limit — exactly 2/λ_max on a quadratic, and on a network "
                  "2.04 times larger than that formula says, because the "
                  "curvature moves to meet whatever step you chose."),
        summary=("The optimiser everyone uses is an ODE solver, and that is an "
                 "identity rather than an analogy: gradient descent is forward "
                 "Euler on the gradient flow. So the learning rate is a step "
                 "size, and step sizes have stability limits. On a quadratic "
                 "the limit is 2/λ_max exactly — at 0.99× of it the run ends "
                 "at 4.4e-04 and at 1.01× at 3.9e+03, while the continuous "
                 "flow being approximated converges at every step size, which "
                 "locates the blow-up in the solver rather than the problem. "
                 "The optimal step sits at κ/(κ+1) of that limit, so on any "
                 "ill-conditioned problem the best learning rate is within a "
                 "percent of a cliff. Then the correction the quadratic cannot "
                 "give you: under full-batch gradient descent on a small MLP, "
                 "2/lr is a two-sided attractor for λ_max — pushed down from "
                 "7.36 to 4.0 at lr = 0.5 — the boundary only binds above a "
                 "certain step, and the usable limit is twice the one you "
                 "would compute at initialisation."),
        tags=["numerical-analysis", "optimization", "gradient-descent",
              "stability", "lectures", "machine-learning"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every system here is constructed in the episode "
            "and every number is produced by the code shown, executed when this "
            "page was built.",
            "Machinery: `standarderror/numerics/steps.py`, tested in "
            "`tests/test_steps.py`.",
            "Where this stops: Hairer, Nørsett and Wanner, *Solving Ordinary "
            "Differential Equations I* (Springer, 1993), for stability of "
            "one-step methods; Polyak, \"Some methods of speeding up the "
            "convergence of iteration methods\", *USSR Comp. Math.* 4 (1964), "
            "for the heavy-ball rate; Nesterov, *Introductory Lectures on "
            "Convex Optimization* (Kluwer, 2004), for the optimal fixed step; "
            "and Cohen, Kaur, Li, Kolter and Talwalkar, \"Gradient descent on "
            "neural networks typically occurs at the edge of stability\", "
            "*ICLR* (2021), for the network half.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4, "
                           "torch=2.13.0",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": (f"one {QUAD_D}-dimensional quadratic with condition "
                           f"number {QUAD_COND:.0f}, and one tanh MLP trained "
                           f"full-batch for {NET_STEPS} steps on 200 rows"),
            "determinism": "one seed per system, each stated in the code shown",
        },
    )
    return _write(post, res, net, figs, snip)


def _write(post: Post, res: dict, net: dict, figs: dict, snip: dict) -> Post:
    crit, sweep = res["crit"], res["sweep"]
    lam_max = float(res["eigs"][-1])
    lam_min = float(res["eigs"][0])
    lo, hi = res["bisected"]
    k100 = next(c for c in res["conditions"] if c["kappa"] == 100.0)
    k3 = next(c for c in res["conditions"] if c["kappa"] == 3.0)
    k1000 = next(c for c in res["conditions"] if c["kappa"] == 1000.0)
    m0, m9 = res["momentum"][0], res["momentum"][2]
    tuned = res["tuned"]
    tuned_decades = st.steps_per_decade(res["tuned_measured"])
    runs = net["runs"]
    r02, r05, r1, r2, r5 = (runs[0.02], runs[0.05], runs[0.1], runs[0.2],
                            runs[0.5])
    amp = res["amplification"]

    post.add(
        "The optimiser is an ODE solver, and this is not a metaphor",
        r"""Write down the differential equation that moves a point downhill as fast as the surface allows:

$$
\frac{dx}{dt} = -\nabla f(x)
$$

This is the *gradient flow*. It has no step size, because it has no steps. It also cannot fail: along any solution, `df/dt` equals minus the squared gradient norm, so the loss decreases monotonically until the gradient vanishes, for every problem, forever. There is no hyperparameter to get wrong.

Now solve it numerically with the simplest method there is. Forward Euler replaces the derivative with a difference over a step `h`:

$$
\frac{x_{n+1} - x_n}{h} = -\nabla f(x_n)
\quad\Longrightarrow\quad
x_{n+1} = x_n - h \nabla f(x_n)
$$

The right-hand side is gradient descent. The step size `h` is the learning rate. Not "like" the learning rate — the two expressions are character-for-character the same, and every fact numerical analysis knows about forward Euler is therefore a fact about your training run.

The first of those facts is that forward Euler is only *conditionally stable*. There is a largest step beyond which the numerical solution diverges even though the exact solution it approximates does not. For a linear system `dx/dt = -Hx` the condition is that every eigenvalue `lam` of `H` satisfy `|1 - h lam| < 1`, which for positive `lam` means

$$
0 < h < \frac{2}{\lambda_{\max}(H)}
$$

and *f*(*x*) = *x*ᵗ*Hx*/2 — the quadratic — is exactly that linear system. So on a quadratic, gradient descent has a stability limit with a formula in it. Here is the whole claim in twenty lines.""",
        figures=[])

    post.add(
        "",
        f"""{snip['identity'].markdown()}

Read the last three rows again. At {MULTIPLES[2]:g} times the critical step, four hundred steps take the gradient norm from {np.linalg.norm(res['H'] @ res['x0']):.2f} down to {sweep.at(0.99):.2e}. At {MULTIPLES[4]:g} times — a change of two percent in one number — the same four hundred steps end at {sweep.at(1.01):.2e}. That is a factor of {sweep.at(1.01) / sweep.at(0.99):.1e} across a 2% change in a hyperparameter that most people set by trying a few values and keeping the one that looked best.

And the row between them is stranger than either.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "",
        f"""At *exactly* the critical step the sharpest direction's multiplier is |1 − lr·λ_max| = |1 − 2| = 1. Not slightly less, not slightly more. That component of the gradient is neither damped nor amplified, and it is still there, unchanged, after any number of steps:

{snip['marginal'].markdown()}

Nine digits. This is the marginal case that stability analysis is *about*: the boundary is not a fuzzy region where things get unreliable, it is the single step size at which one eigendirection is copied forward exactly. On one side of it that direction dies; on the other it grows geometrically. Nothing in the loss curve tells you which side you are on until it does.""",
        level=3)

    post.add(
        "What the blow-up is actually made of",
        """One learning rate, twenty directions, twenty different multipliers. A single step multiplies the component of `x` along each eigenvector of `H` by |1 − lr·λ| — a number you never see, produced by a scalar you chose and a spectrum you did not compute.

Readers of the previous series will recognise the shape. Ridge regression's per-direction multiplier was *s*²/(*s*² + α): one regularisation constant, one multiplier per singular direction, and the geometry hidden behind a single knob. This is the same situation with a different function, and it has the same consequence — the aggregate behaviour is an average over directions that are doing completely different things.""",
        figures=[figs["f2"]])

    post.add(
        "",
        f"""Two ends of that curve matter, and they pull in opposite directions.

The **sharpest** direction sets the stability limit. At {MULTIPLES[4]:g}× the critical step its multiplier is {amp[1.01].max():.3f} and every other direction is contracting — {int((amp[1.01] > 1).sum())} growing, {QUAD_D - int((amp[1.01] > 1).sum())} shrinking. That is the entire divergence: nineteen directions converging nicely while one grows by 2% per step, four hundred times, which is {1.02 ** SWEEP_STEPS:.0e}.

The **flattest** direction sets the speed. At 0.99× the critical step the worst multiplier is {amp[0.99].max():.4f}, and it belongs to λ_min = {lam_min:.2f}, not to λ_max. The step that is nearly too large for the sharp directions is still barely moving the flat ones. That is what a condition number of {QUAD_COND:.0f} means operationally, and it is why the optimum is where it is.

Setting the two ends equal — |1 − lr·λ_min| = |1 − lr·λ_max| — gives the best fixed step in one line:

$$
\\mathrm{{lr}}^{{*}} = \\frac{{2}}{{\\lambda_{{\\min}} + \\lambda_{{\\max}}}}
\\quad\\text{{with per-step contraction}}\\quad
\\frac{{\\kappa - 1}}{{\\kappa + 1}}
$$""",
        level=3)

    post.add(
        "The problem never diverges. The solver does",
        f"""Before the optimum, one more thing about the row that reached {sweep.at(1.1):.1e}.

That number is not a property of the quadratic. The gradient flow on this same quadratic, from this same starting point, integrated to the same time {SWEEP_STEPS}·lr, has reached {res['flow'][1.1]:.2e}. It cannot do anything else: the exact solution is `x(t) = exp(-tH)x0`, every eigencomponent decays like `exp(-t λ)`, and there is no step size in that expression to make large.

So `{sweep.at(1.1):.1e}` is a number the discretisation invented. Forward Euler at a step size past its stability limit does not approximate the flow badly; it approximates a different, growing solution.""",
        figures=[figs["f1"]])

    post.add(
        "",
        f"""Which is worth holding onto when a run explodes. Divergence is not evidence that the loss surface is pathological, that the initialisation was bad, or that the data has outliers in it. The default hypothesis, checkable in a few Hessian-vector products, is that the step size is past `2/λ_max` — and at half the critical step, Euler and the exact flow here agree to {res['flow'][0.5] / sweep.at(0.5):.2f}×, which is what "the same object" looks like when the discretisation is fine.""",
        level=3)

    post.add(
        "The optimum sits on the cliff edge",
        f"""Divide the optimal step by the largest stable one and the eigenvalues cancel:

$$
\\frac{{\\mathrm{{lr}}^{{*}}}}{{2/\\lambda_{{\\max}}}}
= \\frac{{\\lambda_{{\\max}}}}{{\\lambda_{{\\min}} + \\lambda_{{\\max}}}}
= \\frac{{\\kappa}}{{\\kappa + 1}}
$$

That is exact, it depends on nothing but the condition number, and it says something uncomfortable. At κ = {k3['kappa']:.0f} the optimum is at {k3['fraction'] * 100:.0f}% of the limit — comfortable. At κ = {k100['kappa']:.0f} it is at {k100['fraction'] * 100:.1f}%. At κ = {k1000['kappa']:.0f}, {k1000['fraction'] * 100:.2f}%.

Real problems are ill-conditioned. So on a real problem the best learning rate is essentially *at* the largest one that does not diverge — which is why "raise it until it breaks, then back off a little" is not folklore that happens to work, it is very nearly the correct procedure. It is also why that procedure is uncomfortable to run: it aims at a target one percent away from a cliff worth {sweep.at(1.01) / sweep.at(0.99):.0e} in the final gradient norm.""",
        figures=[figs["f3"]])

    post.add(
        "",
        f"""And look at what the optimum buys. At κ = {k100['kappa']:.0f} the best possible contraction is {k100['rate']:.4f} per step, which is **{k100['per_decade']:.0f} steps to gain one decimal digit** — and no fixed step size does better, because the formula above is a minimum over `lr`, not an estimate. At κ = {k1000['kappa']:.0f} it is {k1000['per_decade']:.0f}. Tuning the learning rate on an ill-conditioned problem is optimising something that is nearly flat in the only quantity you can move.

Momentum is the thing that changes the exponent rather than the constant, and its stability limit is the same formula with one factor in it:

$$
\\mathrm{{lr}} < \\frac{{2(1 + \\beta)}}{{\\lambda_{{\\max}}}}
$$

Bisecting for the true threshold, rather than trusting that: at β = 0 the largest converging step is {m0['bisected']:.6f} against a predicted {m0['predicted']:.6f}, a relative error of {m0['relative_error']:.1e}. At β = {m9['beta']} it is {m9['bisected']:.6f} against {m9['predicted']:.6f}, error {m9['relative_error']:.1e}. Exactly `1 + β` wider, measured. Which is the whole reason a learning rate that diverges without momentum trains with it — here, lr = {res['rescued']['lr']:.2f} is {res['rescued']['lr'] / crit:.1f}× the plain limit, diverges plain, and converges at β = 0.9.

Tuned properly the pair is β = {tuned['beta']:.3f} with lr = {tuned['lr']:.3f}, and the rate becomes {tuned['rate']:.4f} — the plain rate with √κ substituted for κ. Measured over {SWEEP_STEPS} steps that is {tuned_decades:.1f} steps per digit against {k100['per_decade']:.0f}, a factor of {k100['per_decade'] / tuned_decades:.0f}. Note the ratio comes from replacing a condition number by its square root, which is a change of kind, and no amount of learning-rate tuning reaches it.""",
        level=3)

    post.add(
        "Then the network moves the boundary",
        """Everything above is exact and everything above is about a quadratic, where `H` is a constant. A neural network's Hessian is not a constant. It depends on where the parameters are, the parameters depend on the trajectory, and the trajectory depends on the learning rate — so `λ_max` is a function of the step size you chose, and `2/λ_max` is not a number you can look up before training.

That sounds like the kind of caveat that dissolves the whole analysis. It does not, and what actually happens is more specific than either "the threshold applies" or "the threshold doesn't apply". Here is full-batch gradient descent on a two-hidden-layer tanh MLP — 200 rows, mean squared error, fixed step, no momentum, no schedule — with `λ_max` measured by power iteration on Hessian-vector products every few hundred steps.""",
        figures=[figs["f4"]])

    post.add(
        "",
        f"""Three regimes, and the middle one is the surprise.

**Below the boundary the sharpness rises and stops short of it.** At lr = {r02['lr']:.2f} the ratio λ_max·lr/2 climbs from {st.sharpness_ratio(net['lam0'], 0.02):.3f} to {r02['ratio']:.3f}; at lr = {r05['lr']:.2f}, to {r05['ratio']:.3f}. The curvature does grow during training — from {net['lam0']:.2f} to {r05['lam_max']:.1f}, a factor of {r05['lam_max'] / net['lam0']:.1f} — but it grows because the network is fitting, not because it is being pushed, and it plateaus wherever the task's own sharpness plateaus.

**At and above lr = {r1['lr']:.1f} the boundary binds.** The ratio ends at {r1['ratio']:.3f}, {r2['ratio']:.3f} and {r5['ratio']:.3f} for lr = {r1['lr']:.1f}, {r2['lr']:.1f} and {r5['lr']:.1f} — within {max(abs(r1['ratio'] - 1), abs(r2['ratio'] - 1), abs(r5['ratio'] - 1)) * 100:.1f}% of 1 across a fivefold range of learning rate. `λ_max` ends at {r1['lam_max']:.1f}, {r2['lam_max']:.1f} and {r5['lam_max']:.1f}, which are {2 / r1['lr']:.0f}, {2 / r2['lr']:.0f} and {2 / r5['lr']:.0f}. This is the edge of stability, and the sharpness is not converging to a property of the problem — it is converging to a property of your hyperparameter.

**And it arrives from both sides.** The lr = {r5['lr']:.1f} run *starts* at a ratio of {st.sharpness_ratio(net['lam0'], 0.5):.2f}, well past the boundary, at a step size which on a fixed quadratic of that curvature would diverge in a few dozen steps. It does not diverge. `λ_max` falls from {net['lam0']:.2f} to {r5['lam_max']:.2f} and the run converges. I had written this phenomenon down, before measuring it, as "sharpness rises to meet 2/lr and hovers there". That is the upward half of a two-sided attractor, and stating only the upward half would have implied a network can never be initialised past its own stability boundary, which this run does.""",
        level=3)

    post.add(
        "",
        f"""So how large a step does this network actually tolerate? Not `2/λ_max` at initialisation — that is {net['naive']:.4f}, and lr = {r5['lr']:.1f} trains fine. Bisecting for the real threshold:

- converges at lr = {net['bracket'][0]:.5f}
- diverges at lr = {net['bracket'][1]:.5f}

A bracket {(net['bracket'][1] - net['bracket'][0]) / net['bracket'][0] * 100:.2f}% wide, and **{net['factor']:.2f} times** the number the formula gives at initialisation. Both halves of that matter. The threshold is real and it is sharp — at lr = {runs[0.8]['lr']:.1f} the loss goes non-finite at step {runs[0.8]['diverged_at']}, with no degradation on the way — so this is not a case where the classical analysis stops applying. It applies, and its input is wrong, because the curvature you measure before training is not the curvature you will train at.

One more habit the edge regime interferes with. In that regime the training loss is **not monotone**: at lr = {r2['lr']:.1f} it rises on {r2['rose_fraction'] * 100:.0f}% of the last {NET_STEPS // 2} steps — median rise {r2['median_rise'] * 100:.2f}%, largest {r2['max_rise'] * 100:.1f}% — while falling {r2['tail_drop']:.0f}-fold over that same window. At lr = {r05['lr']:.2f}, below the boundary, it rises on exactly {r05['rose_fraction'] * 100:.0f}% of them. A loss that ticks upward at a large step size is what convergence looks like there, and treating it as a bug is how people end up lowering a learning rate that was working.""",
        level=3)

    post.add(
        "The measurement that makes this usable",
        f"""None of the above requires forming a Hessian. `λ_max` comes out of repeated Hessian-vector products, each of which costs about one extra backward pass, and the estimate converges in a few dozen of them — {res['power']['iterations']} on the quadratic above, to {abs(res['power']['lam_max'] - lam_max) / lam_max:.0e} relative. That is affordable every few hundred training steps on a model of any size.

{snip['diagnostic'].markdown()}

The only thing that changes on a network is where the product comes from: instead of `H @ v`, differentiate `grad · v` a second time.

One number in that block is worth arguing with, though: `iters`. Sixty iterations is generous on the quadratic and not enough on the trained network, because the top of a trained network's spectrum is crowded and power iteration separates a crowded top slowly. At lr = 0.1, sixty iterations return λ_max = 19.790 and four hundred return 19.998 — a ratio of 0.990 against 1.000, which is the difference between reporting "close to the boundary" and "on it". If you are going to read this number, resolve it.

{snip['net'].markdown()}

Which gives one number worth printing in a training log next to the loss: **λ_max·lr/2**. Below 1 you have headroom, and if it is well below 1 and the loss is falling slowly, the step is small rather than the problem hard. Near 1 you are at the edge — expect a non-monotone loss and do not read the bumps as a bug. Above 1 and staying there, you are diverging and the next few steps will show it.""",
        figures=[])

    post.add(
        "What to keep",
        f"""1. Gradient descent is forward Euler on the gradient flow. The learning rate is a discretisation step and inherits forward Euler's conditional stability.
2. On a quadratic the limit is exactly `2/λ_max`, and it is a threshold: {sweep.at(0.99):.2e} at 0.99× of it, {sweep.at(1.01):.2e} at 1.01×.
3. The blow-up belongs to the solver, not the problem. The flow being approximated converges at every step size.
4. The optimal step is at `κ/(κ+1)` of the limit, so on ill-conditioned problems the best learning rate is nearly the largest stable one — and it still costs {k100['per_decade']:.0f} steps per digit at κ = {k100['kappa']:.0f}. Momentum widens the limit by exactly `1 + β` and changes the rate's dependence from κ to √κ.
5. On a network `2/lr` is a two-sided attractor for `λ_max` above a certain step size, the usable limit is about twice what the initial curvature suggests, and in that regime the training loss is not monotone.
6. `λ_max·lr/2` is cheap. Print it.""")

    post.add(
        "Exercise",
        """Take a model you are training and a learning rate you chose by trying a few. Estimate `λ_max` with twenty Hessian-vector products at your current checkpoint — twenty extra backward passes — and compute `λ_max·lr/2`.

Then do it again five hundred steps later.

Three outcomes, and each of them tells you a different thing. If the ratio is well below 1 and not moving, your step size is not the binding constraint and tuning it will not help much; look at conditioning. If it is sitting near 1, you are at the edge, your loss curve's bumps are the method rather than a bug, and the largest useful step is close to where you are. And if it is above 1 and rising, you have a few dozen steps before the run ends, which is enough time to checkpoint.

Next episode: the other finite difference in your codebase. A gradient check *is* a finite difference, so it has an optimal step — and it is not the `1e-8` everybody uses, being off by three orders of magnitude in a direction that costs you eight in what the check can detect.""")

    post.hero = figs["hero"]
    return post


def main() -> Post:
    return build()
