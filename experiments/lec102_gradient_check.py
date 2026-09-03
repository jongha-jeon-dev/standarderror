"""Numerical Analysis 2: A Gradient Check Is a Finite Difference.

The step size in a gradient check is almost always a default nobody chose, and
it decides what the check can see -- by eight orders of magnitude in float64,
and by nine more once you change the precision.

Measured:

* The error is a U in the step size, so "a smaller step is more accurate" is not
  imprecise, it has the wrong shape. On `sin` at `x = 1`, the forward difference
  is best at `h = 5.6e-09` and the central one at `5.6e-06`.
* The central difference's advantage is not accuracy at a given step. At
  `h = 1e-08` the two agree to a factor of 1.2. Its advantage is being *allowed*
  a larger step, and at its own optimum it is 6,700 times better.
* The same U governs a gradient check. On a six-parameter loss the smallest
  relative error in one gradient entry the check can resolve runs from `9.3e-10`
  at `h = 1e-05` to `9.9e-02` at `h = 1e-13` -- eight orders of magnitude,
  decided by a constant nobody writes down.
* Every one of those scales is a power of `eps`, so the precision moves them all.
  The central difference's optimum runs `5.6e-06`, `5.6e-04`, `3.2e-02`,
  `1.8e-02` across float64, float32, float16 and bfloat16.
* And the check's resolution runs `1.8e-10`, `5.2e-05`, `5.4e-02`, `1.7e-01`.
  A gradient check in bfloat16 cannot distinguish a correct gradient from one
  that is 17% wrong, at any step size. I had derived 2.5% from `eps**(2/3)`
  before measuring it, which is wrong by a factor of seven.
* `Im f(x+ih)/h` has no cancellation and therefore no optimum: exactly zero
  error at every step from `1e-20` to `1e-200`. What breaks it is not the list
  you would guess -- `abs` and a real-part cast do, `x*abs(x)` returns exactly
  half the right answer, and a ReLU network does not break it at all.

Run: `standarderror run lec102_gradient_check --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.numerics import differencing as fd
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
POST_DATE = date(2026, 9, 3)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Numerical Analysis for Machine Learning, Taught Through What Breaks"
SERIES_TAG = "Numerical Analysis"

#: A quarter-decade grid, so that an optimum three decades wide is located to
#: better than a decade. Decade grids put the float64 central optimum at 1e-5
#: with an error of 1.1e-11; a quarter-decade grid finds 5.6e-6 and 3.8e-13.
FINE = 10.0 ** -np.arange(-1.5, 14.01, 0.25)
#: Wide enough to show that the complex step has no optimum at all.
VERY_FINE = 10.0 ** -np.arange(0, 20.01, 0.5)
#: The four steps the prose quotes for the gradient check, including the two
#: common defaults.
CHECK_STEPS = (1e-5, 1e-7, 1e-9, 1e-11, 1e-13)
#: Which gradient entry to corrupt. Any one; entry 2 is neither the largest nor
#: the smallest in this design.
BAD_ENTRY = 2


def compute() -> dict:
    fwd = fd.error_sweep(np.sin, np.cos, 1.0, FINE, kind="forward")
    cen = fd.error_sweep(np.sin, np.cos, 1.0, FINE, kind="central")

    # The two claims about the central difference that are usually conflated.
    at_same = {}
    for h in (1e-8, 1e-5):
        at_same[h] = {
            "forward": abs(fd.forward(np.sin, 1.0, h) - np.cos(1.0)),
            "central": abs(fd.central(np.sin, 1.0, h) - np.cos(1.0)),
        }

    loss, grad, v = fd.gradient_check_design()
    steps = [fd.smallest_detectable_bug(loss, grad, v, h, index=BAD_ENTRY)
             for h in CHECK_STEPS]
    curve = [fd.smallest_detectable_bug(loss, grad, v, float(h),
                                        index=BAD_ENTRY)
             for h in FINE]

    table = fd.precision_table(np.sin, np.cos, 1.0, FINE)
    # Where the difference stops being inaccurate and starts being zero: below
    # this step, x + h rounds back to x, so f(x+h) - f(x-h) is exactly 0.0 and
    # the reported derivative is 0.0 with no warning anywhere.
    collapse = {}
    for r in table:
        sw = fd.precision_sweep(np.sin, np.cos, 1.0, FINE,
                                precision=r["precision"])
        dead = [float(h) for h, e in zip(sw.h, sw.error)
                if np.isfinite(e) and abs(e - abs(np.cos(1.0))) < 1e-12]
        collapse[r["precision"]] = max(dead) if dead else None
    sweeps = {r["precision"]: fd.precision_sweep(np.sin, np.cos, 1.0, FINE,
                                                 precision=r["precision"])
              for r in table}
    best = {p: fd.best_detectable(loss, grad, v, FINE, index=BAD_ENTRY,
                                  precision=p)
            for p in fd.PRECISIONS}

    # The complex step, over twenty decades of step size.
    cs = np.array([abs(fd.complex_step(np.sin, 1.0, float(h)) - np.cos(1.0))
                   for h in VERY_FINE])
    extreme = {h: fd.complex_step(np.sin, 1.0, h) - np.cos(1.0)
               for h in (1e-20, 1e-100, 1e-200)}
    breaks = {
        "abs": (fd.complex_step(np.abs, 1.5), 1.0,
                fd.is_complex_safe(np.abs, 1.5)),
        "real cast": (fd.complex_step(lambda z: np.real(z) ** 2, 1.5), 3.0,
                      fd.is_complex_safe(lambda z: np.real(z) ** 2, 1.5)),
        "x*abs(x)": (fd.complex_step(lambda z: z * np.abs(z), 1.5), 3.0,
                     fd.is_complex_safe(lambda z: z * np.abs(z), 1.5)),
    }
    relu = _relu_check()

    return {"forward": fwd, "central": cen, "at_same": at_same,
            "collapse": collapse,
            "design": (loss, grad, v), "steps": steps, "curve": curve,
            "table": table, "sweeps": sweeps, "best": best,
            "complex": cs, "extreme": extreme, "breaks": breaks, "relu": relu,
            "cancellation": [fd.cancellation_pair(x)
                             for x in (1e-2, 1e-4, 1e-6)],
            "easy_derivative": _easy_derivative()}


def _relu_check() -> dict:
    """The surprise: `np.maximum` compares complex numbers by real part."""
    rng = np.random.default_rng(1)
    W1, b1, w2 = (rng.standard_normal((4, 3)), rng.standard_normal(4),
                  rng.standard_normal(4))

    def net(x):
        val = np.array([x, 0.3, -0.7], dtype=complex)
        return w2 @ np.maximum(W1 @ val + b1, 0)

    out = {}
    for x in (0.5, -0.5, 2.0):
        cs = fd.complex_step(net, x, 1e-20)
        ce = fd.central(lambda z: float(np.real(net(z))), x, 1e-5)
        out[x] = {"complex_step": cs, "central": ce, "gap": abs(cs - ce)}
    return out


def _easy_derivative() -> dict:
    """A badly conditioned evaluation whose derivative comes back exact."""
    g = lambda t: t - 1.0                                # noqa: E731
    x = 1.0001
    s = fd.error_sweep(g, lambda t: 1.0, x, FINE, kind="central")
    return {"x": x, "condition_number": fd.condition_number(
        g, lambda t: 1.0, x), "best_error": s.best_error}


# --------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["ucurve"] = s.run("""
        import numpy as np

        def forward(f, x, h):
            return (f(x + h) - f(x)) / h

        def central(f, x, h):
            return (f(x + h) - f(x - h)) / (2 * h)

        truth = np.cos(1.0)
        print(f"{'h':>9}  {'forward':>11}  {'central':>11}")
        for h in [1e-2, 1e-5, 1e-8, 1e-11, 1e-14]:
            fe = abs(forward(np.sin, 1.0, h) - truth)
            ce = abs(central(np.sin, 1.0, h) - truth)
            print(f"{h:>9.0e}  {fe:>11.2e}  {ce:>11.2e}")
    """, expect=["forward", "1e-14"])

    out["check"] = s.run(f"""
        # The gradient check as it is actually written, and the only line in it
        # that anybody argues about is the value of `h`.
        def gradient_check(loss, grad, x, h):
            g = grad(x)
            num = np.empty_like(x)
            for i in range(len(x)):
                step = np.zeros_like(x)
                step[i] = h
                num[i] = (loss(x + step) - loss(x - step)) / (2 * h)
            return abs(num - g).max() / max(abs(g).max(), 1.0)

        # A six-parameter loss whose gradient is known in closed form, so a
        # failing check can only mean the check failed.
        rng = np.random.default_rng(0)
        W = rng.standard_normal((6, 6))
        sym = 0.5 * (W + W.T)
        loss = lambda v: 0.5 * v @ W @ v + np.sum(np.exp(0.3 * v))
        grad = lambda v: sym @ v + 0.3 * np.exp(0.3 * v)
        v = rng.standard_normal(6)

        # Now break one entry of the gradient by 1%, 0.01% and 1e-6, and see
        # which of them the check can still tell apart from its own noise.
        for h in [1e-5, 1e-9, 1e-13]:
            clean = gradient_check(loss, grad, v, h)
            line = f"h={{h:.0e}}  noise floor {{clean:.1e}}  |"
            for rel in [1e-2, 1e-4, 1e-6]:
                def buggy(w, r=rel):
                    g = grad(w).copy()
                    g[{BAD_ENTRY}] *= 1 + r        # one entry, off by r
                    return g
                got = gradient_check(loss, buggy, v, h)
                seen = "seen" if got > 2 * clean else "MISSED"
                line += f"  {{rel:.0e}}: {{seen}}"
            print(line)
    """, expect=["noise floor", "MISSED"])

    out["precision"] = s.run("""
        # bfloat16 is float32 with the low 16 mantissa bits gone. That is the
        # whole format, so it can be written in four lines rather than imported
        # -- and this agrees with torch.bfloat16 on 3000 of 3000 test values.
        def to_bfloat16(x):
            u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint32)
            r = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
            return (r.astype(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)

        rounders = {
            "float64":  lambda v: float(v),
            "float32":  lambda v: float(np.float32(v)),
            "float16":  lambda v: float(np.float16(v)),
            "bfloat16": lambda v: float(to_bfloat16(np.float32(v))),
        }

        for name, q in rounders.items():
            best = min(
                (abs(q((q(np.sin(q(1.0) + q(h))) - q(np.sin(q(1.0) - q(h))))
                       / q(2 * h)) - np.cos(1.0)), h)
                for h in 10.0 ** -np.arange(0, 12, 0.25) if q(2 * h) != 0.0)
            eps = {"float64": 2.0 ** -52, "float32": 2.0 ** -23,
                   "float16": 2.0 ** -10, "bfloat16": 2.0 ** -7}[name]
            print(f"{name:>9}  eps {eps:8.2e}  best h {best[1]:8.2e}  "
                  f"error {best[0]:8.2e}   eps**(1/3) {eps ** (1/3):8.2e}")
    """, expect=["bfloat16", "eps**(1/3)"])

    out["complex"] = s.run("""
        # No subtraction of nearby values, so no cancellation, so no optimum.
        def complex_step(f, x, h=1e-20):
            return (f(x + 1j * h)).imag / h

        for h in [1e-8, 1e-20, 1e-100, 1e-200]:
            err = abs(complex_step(np.sin, 1.0, h) - np.cos(1.0))
            print(f"h = {h:.0e}   error {err:.1e}")

        # The catch, and it is not the list you would guess.
        cases = {
            "abs(x)":      (np.abs, 1.0),
            "real(z)**2":  (lambda z: np.real(z) ** 2, 3.0),
            "x * abs(x)":  (lambda z: z * np.abs(z), 3.0),
            "max(x,0)**2": (lambda z: np.maximum(z, 0) ** 2, 3.0),
        }
        print()
        for name, (f, truth) in cases.items():
            got = complex_step(f, 1.5)
            verdict = "ok" if abs(got - truth) < 1e-6 else "WRONG"
            print(f"{name:>12}  d/dx at 1.5 = {got:5.2f}, truth {truth:4.1f}"
                  f"   {verdict}")
    """, expect=["h = 1e-200", "WRONG"])

    return out


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    fwd, cen = res["forward"], res["central"]

    # --- f0: the U, and the two schemes' different optima ------------------
    def u_curve(ax, m):
        ax.plot(fwd.h, fwd.error, lw=1.8, color=m.series[0],
                label="forward difference")
        ax.plot(cen.h, cen.error, lw=1.8, color=m.series[1],
                label="central difference")
        # Placed by hand: the forward optimum sits on the central curve and
        # the central one sits on the x-axis label, so they go opposite ways.
        for sweep, colour, place in ((fwd, m.series[0], (9, 9, "left")),
                                     (cen, m.series[1], (0, 13, "center"))):
            ax.plot([sweep.best_h], [sweep.best_error], "o", ms=7,
                    color=colour)
            ax.annotate(f"best: h = {sweep.best_h:.1e}",
                        (sweep.best_h, sweep.best_error),
                        textcoords="offset points",
                        xytext=(place[0], place[1]), ha=place[2],
                        fontsize=8.5, color=m.ink_secondary)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    out["f0"] = charts.diagram(
        u_curve,
        title="The error of a derivative is a U in the step size",
        subtitle=("d/dx sin(x) at x = 1, against the exact cos(1). The step "
                  "shrinks to the right, so the intuition \"smaller is more "
                  "accurate\" would be a line falling to the right."),
        xlabel="step size h (shrinking →)", ylabel="absolute error",
        source="Computed; standarderror/numerics/differencing.py.",
        alt=("Two V-shaped curves of absolute error against step size on log "
             "axes, with the step size decreasing to the right. Each has a "
             "marked minimum, the central difference's three decades to the "
             "left of the forward difference's and two decades lower."),
        caption=(f"Truncation falls as h falls; cancellation rises like eps/h. "
                 f"The balance point is eps^(1/2) for a first-order scheme and "
                 f"eps^(1/3) for a second-order one, which is "
                 f"{fd.optimal_h('forward'):.1e} and "
                 f"{fd.optimal_h('central'):.1e} — against measured optima of "
                 f"{fwd.best_h:.1e} and {cen.best_h:.1e}. Past the minimum, "
                 f"refining the step makes the answer worse: at h = 1e-14 the "
                 f"central difference is {cen.penalty_at(1e-14):.1e} times "
                 f"worse than at its own optimum."),
        path=str(IMG / f"lec102-f0-ucurve.{EXT}"))[0]

    # --- f1: what the check can detect, against the step ------------------
    def detect(ax, m):
        hs = np.array([c["h"] for c in res["curve"]])
        det = np.array([c["detectable"] for c in res["curve"]])
        floor = np.array([c["floor"] for c in res["curve"]])
        ok = np.isfinite(det)
        ax.plot(hs[ok], det[ok], lw=1.9, color=m.series[0],
                label="smallest detectable gradient error")
        ax.plot(hs, floor, lw=1.6, ls=(0, (5, 3)), color=m.series[1],
                label="the check's own noise floor")
        ax.axhline(0.1, color=m.ink, lw=1.4)
        ax.annotate("a 10% gradient bug", (1e-10, 0.1),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8.5, color=m.ink_secondary)
        # The last one goes above; below, it lands on the noise-floor curve.
        places = {1e-5: (0, -18, "center"), 1e-7: (0, -18, "center"),
                  1e-13: (-8, 8, "right")}
        for s in res["steps"]:
            if s["h"] in places:
                dx, dy, ha = places[s["h"]]
                ax.plot([s["h"]], [s["detectable"]], "o", ms=6,
                        color=m.series[0])
                ax.annotate(f"h = {s['h']:.0e}", (s["h"], s["detectable"]),
                            textcoords="offset points", xytext=(dx, dy),
                            ha=ha, fontsize=8.5, color=m.ink_secondary)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    lo = next(s for s in res["steps"] if s["h"] == 1e-5)
    hi = next(s for s in res["steps"] if s["h"] == 1e-13)
    out["f1"] = charts.diagram(
        detect,
        title="What your gradient check can see, as a function of one default",
        subtitle=("A six-parameter loss with a gradient known in closed form. "
                  "One entry is corrupted by a relative amount, bisected for "
                  "the smallest amount the check still separates from its own "
                  "noise."),
        xlabel="step size h (shrinking →)",
        ylabel="smallest detectable relative error",
        source="Computed; standarderror/numerics/differencing.py.",
        alt=("A rising curve of the smallest detectable gradient error against "
             "shrinking step size, above a dashed noise-floor line, crossing a "
             "horizontal reference at ten percent near the right edge."),
        caption=(f"At h = 1e-05 the check resolves a relative error of "
                 f"{lo['detectable']:.1e} in one entry. At h = 1e-13 it needs "
                 f"{hi['detectable']:.1e} — a 10% wrong gradient passes. Eight "
                 f"orders of magnitude in what the test can detect, decided by "
                 f"a number that is usually a default. The floor is what the "
                 f"check reports on a **correct** gradient, and it is the reason "
                 f"the curve rises: the bug has to clear the noise."),
        path=str(IMG / f"lec102-f1-detect.{EXT}"))[0]

    # --- f2: the same U in four precisions --------------------------------
    def precisions(ax, m):
        colours = list(m.series) + [m.ink_secondary]
        for (name, sweep), colour in zip(res["sweeps"].items(), colours):
            ok = np.isfinite(sweep.error)
            ax.plot(sweep.h[ok], sweep.error[ok], lw=1.8, color=colour,
                    label=name)
            ax.plot([sweep.best_h], [sweep.best_error], "o", ms=6,
                    color=colour)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)

    t = {r["precision"]: r for r in res["table"]}
    out["f2"] = charts.diagram(
        precisions,
        title="One derivation, four precisions: the whole U moves",
        subtitle=("The same central difference on the same function, with every "
                  "stored value rounded to each format. bfloat16 is quantised "
                  "in numpy and checked against torch on 3,000 values."),
        xlabel="step size h (shrinking →)", ylabel="absolute error",
        source="Computed; standarderror/numerics/differencing.py.",
        alt=("Four U-shaped error curves, one per floating-point format, "
             "stacked so that the lower-precision curves sit up and to the "
             "left of the higher-precision ones."),
        caption=(f"The optimum moves from {t['float64']['measured_h']:.1e} in "
                 f"float64 to {t['bfloat16']['measured_h']:.1e} in bfloat16, "
                 f"and the best achievable error from "
                 f"{t['float64']['measured_error']:.1e} to "
                 f"{t['bfloat16']['measured_error']:.1e}. Every scale in the "
                 f"derivation is a power of eps, so changing the precision "
                 f"moves all of them at once — and eps runs from 2.2e-16 to "
                 f"7.8e-03 across these four. The flat tails on the right are "
                 f"worse than they look: there, x + h rounds back to x, so the "
                 f"difference is exactly 0.0 and the error is the whole "
                 f"derivative. In bfloat16 that happens for every step below "
                 f"{res['collapse']['bfloat16']:.1e}."),
        path=str(IMG / f"lec102-f2-precision.{EXT}"))[0]

    # --- f3: the complex step has no U at all -----------------------------
    def no_u(ax, m):
        ax.plot(cen.h, np.maximum(cen.error, 1e-18), lw=1.8,
                color=m.series[1], label="central difference")
        drawn = np.maximum(res["complex"], 1e-18)
        ax.plot(VERY_FINE, drawn, lw=2.0, color=m.series[0],
                label="complex step, Im f(x+ih)/h")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_ylim(3e-19, 1e2)
        ax.annotate("exactly zero error, for twenty decades of h",
                    (1e-14, 1e-18), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8.5,
                    color=m.ink_secondary)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    out["f3"] = charts.diagram(
        no_u,
        title="A derivative with no cancellation has no optimal step",
        subtitle=("Perturb along the imaginary axis and the derivative is the "
                  "imaginary part — no subtraction of nearby values happens, so "
                  "there is nothing for eps/h to amplify."),
        xlabel="step size h (shrinking →)", ylabel="absolute error",
        source="Computed; standarderror/numerics/differencing.py.",
        alt=("The central difference's V-shaped error curve against a flat "
             "line for the complex step, pinned at the bottom of the chart "
             "across the whole range of step size."),
        caption=("The flat line is drawn at the bottom of the axis because the "
                 "measured error is exactly 0.0 at every step from 1e-20 to "
                 "1e-200, which a log axis cannot draw. The catch is in the "
                 "next section: f has to be complex-analytic, and the "
                 "operations that break that are not the ones you would guess."),
        path=str(IMG / f"lec102-f3-complex.{EXT}"))[0]

    # --- f4: the payoff, per precision ------------------------------------
    def payoff(ax, m):
        names = list(fd.PRECISIONS)
        measured = [res["best"][n]["detectable"] for n in names]
        derived = [fd.error_floor("central", eps=fd.EPS_BY_PRECISION[n])
                   for n in names]
        idx = np.arange(len(names))
        ax.bar(idx - 0.19, measured, width=0.36, color=m.series[0],
               label="measured, searched over h")
        ax.bar(idx + 0.19, derived, width=0.36, color=m.series[1],
               label="what eps^(2/3) predicts")
        ax.set_yscale("log")
        ax.set_xticks(idx)
        ax.set_xticklabels(names)
        ax.axhline(0.1, color=m.ink, lw=1.4)
        ax.annotate("a 10% gradient bug", (-0.45, 0.1),
                    textcoords="offset points", xytext=(0, 7), ha="left",
                    fontsize=8.5, color=m.ink_secondary)
        for i, v in enumerate(measured):
            ax.annotate(f"{v:.1e}", (i - 0.19, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8.0,
                        color=m.ink_secondary)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    b = res["best"]
    out["f4"] = charts.diagram(
        payoff,
        title="In half precision, the gradient check is not a weak test",
        subtitle=("The smallest relative error in one gradient entry that the "
                  "check can resolve at its best step, per format. Searched "
                  "over a quarter-decade grid, not derived."),
        xlabel="", ylabel="smallest detectable relative error",
        source="Computed; standarderror/numerics/differencing.py.",
        alt=("Paired bars per floating-point format on a log scale, the "
             "measured resolution beside the value the eps formula predicts, "
             "with the bfloat16 pair above a ten-percent reference line."),
        caption=(f"float64 resolves {b['float64']['detectable']:.1e}; bfloat16 "
                 f"needs {b['bfloat16']['detectable']:.2f} — a gradient 17% "
                 f"wrong passes, at any step size. That is "
                 f"{b['bfloat16']['detectable'] / b['float64']['detectable']:.0e} "
                 f"times across the four formats. The second bar in each pair "
                 f"is what I would have written down without measuring: it is "
                 f"optimistic every time, and by a factor of "
                 f"{b['bfloat16']['detectable'] / fd.error_floor('central', eps=fd.EPS_BY_PRECISION['bfloat16']):.0f} "
                 f"in the row that matters."),
        path=str(IMG / f"lec102-f4-payoff.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    def a_u(panel, m):
        h = res["central"].h
        panel.plot(np.log10(h), np.log10(np.maximum(res["central"].error, 1e-18)),
                   color=m.ink, lw=2.6)
        panel.invert_xaxis()

    def four_us(panel, m):
        for sweep in res["sweeps"].values():
            ok = np.isfinite(sweep.error)
            panel.plot(np.log10(sweep.h[ok]),
                       np.log10(np.maximum(sweep.error[ok], 1e-18)),
                       color=m.ink, lw=2.0)
        panel.invert_xaxis()

    def two_bars(panel, m):
        b = res["best"]
        panel.bar([0, 1], [np.log10(b["float64"]["detectable"]) + 18,
                           np.log10(b["bfloat16"]["detectable"]) + 18],
                  color=[m.ink, m.ink], width=0.34)
        panel.axhline(18 + np.log10(0.1), color=m.grid, lw=2.4)
        panel.set_xlim(-0.7, 1.7)
        panel.set_ylim(0, 20)

    b = res["best"]
    return charts.lecture_hero(
        series=SERIES_TAG, episode=2,
        headline="A gradient check is a finite difference",
        panels=[
            (a_u, f"{res['central'].penalty_at(1e-14):.0e}",
             "worse at h = 1e-14"),
            (four_us, f"{fd.EPS_BY_PRECISION['bfloat16'] / fd.EPS_BY_PRECISION['float64']:.0e}",
             "eps, fp64 to bf16"),
            (two_bars, f"{b['bfloat16']['detectable'] * 100:.0f}%",
             "invisible in bf16"),
        ],
        note=("The error of a finite difference is a U in the step size, so a "
              "smaller step is not a safer one — and a gradient check is a "
              "finite difference. Every scale in the derivation is a power of "
              "eps, which means the precision decides what your check can see: "
              "in bfloat16, a gradient 17% wrong passes at every step size."),
        alt=("A three-panel hand-drawn strip. The first shows a single "
             "V-shaped error curve. The second shows four of them stacked. The "
             "third shows a short bar and a tall one either side of a "
             "horizontal reference line."),
        mode="light",
        path=str(IMG / f"lec102-hero.{EXT}"))[0]


# ------------------------------------------------------------------- build

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    fwd, cen = res["forward"], res["central"]
    best, table = res["best"], {r["precision"]: r for r in res["table"]}
    steps = {s["h"]: s for s in res["steps"]}

    # The spine, asserted rather than trusted.
    assert not fwd.is_monotone and not cen.is_monotone
    assert 0.05 < fwd.best_h / fd.optimal_h("forward") < 20
    assert 0.05 < cen.best_h / fd.optimal_h("central") < 20
    assert cen.best_h / fwd.best_h > 100
    assert 0.5 < res["at_same"][1e-8]["forward"] / res["at_same"][1e-8]["central"] < 2
    assert res["at_same"][1e-5]["forward"] / res["at_same"][1e-5]["central"] > 1e4
    assert steps[1e-5]["detectable"] < 1e-9 < steps[1e-13]["detectable"]
    assert steps[1e-13]["detectable"] > 0.05
    for name, r in table.items():
        assert r["predicted_floor"] / r["measured_error"] > 5.0, name
    assert best["float64"]["detectable"] < 1e-9
    assert best["bfloat16"]["detectable"] > 0.10, best["bfloat16"]
    assert best["bfloat16"]["detectable"] / best["float64"]["detectable"] > 1e8
    derived = fd.error_floor("central", eps=fd.EPS_BY_PRECISION["bfloat16"])
    assert best["bfloat16"]["detectable"] / derived > 3.0
    assert all(v == 0.0 for v in res["extreme"].values())
    assert not any(safe for _, _, safe in res["breaks"].values())
    assert abs(res["breaks"]["x*abs(x)"][0] - 1.5) < 1e-9   # exactly half
    assert all(d["gap"] < 1e-11 for d in res["relu"].values())
    # 2.0 to a relative 1e-3, not to 1e-6: `condition_number` divides by the
    # naive `1 - cos x`, which at x = 1e-6 has itself lost most of its digits.
    # The diagnostic is subject to the phenomenon it is diagnosing.
    for c in res["cancellation"]:
        assert abs(c["condition_number"] - 2.0) < 2e-3, c
    assert res["cancellation"][1]["naive_relative_error"] > 1e-10
    assert res["easy_derivative"]["condition_number"] > 1e3
    assert res["easy_derivative"]["best_error"] < 1e-15
    assert res["collapse"]["float64"] is None
    assert res["collapse"]["bfloat16"] > 1e-4

    post = Post(
        title=f"{SERIES_TAG} 2: A Gradient Check Is a Finite Difference",
        slug="numerical-analysis-2-gradient-check",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=2,
        prerequisites=["numerical-analysis-1-step-size"],
        date=POST_DATE,
        subtitle=("The error of a finite difference is a U in the step size, so "
                  "a smaller step is not a safer one — and the step decides "
                  "what your gradient check can see, by eight orders of "
                  "magnitude in float64 and nine more once you change the "
                  "precision. In bfloat16 a gradient 17% wrong passes at every "
                  "step size."),
        summary=("A gradient check is a finite difference, and a finite "
                 "difference has an optimal step size: truncation error falls "
                 "as the step shrinks and cancellation rises like eps/h, so the "
                 "total is a U and refining past its bottom makes the answer "
                 "worse — 9.9e+09 times worse at h = 1e-14 than at the "
                 "optimum. Measured on a six-parameter loss with a known "
                 "gradient, the smallest error in one entry the check can "
                 "resolve runs from 9.3e-10 at h = 1e-05 to 9.9e-02 at "
                 "h = 1e-13. Every scale in that derivation is a power of eps, "
                 "so the precision moves all of them at once: the check "
                 "resolves 1.8e-10 in float64, 5.2e-05 in float32 and 1.7e-01 "
                 "in bfloat16, where it is therefore not a weak test but not a "
                 "test. The way out, where it applies, is a complex step, which "
                 "has no cancellation and no optimum — and whose failure modes "
                 "are not the ones you would guess."),
        tags=["numerical-analysis", "gradient-check", "floating-point",
              "automatic-differentiation", "lectures", "machine-learning"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every function here is written down in the "
            "episode and every number is produced by the code shown, executed "
            "when this page was built.",
            "Machinery: `standarderror/numerics/differencing.py`, tested in "
            "`tests/test_differencing.py`.",
            "Where this stops: Nocedal and Wright, *Numerical Optimization* "
            "(Springer, 2006), section 8.1, for the step-size trade-off; "
            "Squire and Trapp, \"Using complex variables to estimate "
            "derivatives of real functions\", *SIAM Review* 40 (1998), for the "
            "complex step; Higham, *Accuracy and Stability of Numerical "
            "Algorithms* (SIAM, 2002), chapter 1, for conditioning against "
            "stability.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "simulation": ("quarter-decade step grids from 3.2e+01 down to "
                           "1e-14, and a six-parameter loss whose gradient is "
                           "known in closed form"),
            "determinism": "one seed for the loss design, stated in the code shown",
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    fwd, cen = res["forward"], res["central"]
    best = res["best"]
    table = {r["precision"]: r for r in res["table"]}
    steps = {s["h"]: s for s in res["steps"]}
    relu = res["relu"]
    cancel = next(c for c in res["cancellation"] if c["x"] == 1e-4)
    easy = res["easy_derivative"]
    bf_derived = fd.error_floor("central", eps=fd.EPS_BY_PRECISION["bfloat16"])
    same8 = (res["at_same"][1e-8]["forward"] / res["at_same"][1e-8]["central"])
    same5 = (res["at_same"][1e-5]["forward"] / res["at_same"][1e-5]["central"])

    post.add(
        "Every gradient check has a number in it that nobody chose",
        r"""Somewhere in most training codebases there is a function like this, and it has been passing for years:

```python
num = (loss(x + h) - loss(x - h)) / (2 * h)
assert abs(num - grad(x)).max() < 1e-6
```

The `1e-6` on the second line gets argued about. The `h` on the first line is a default — usually `1e-7` or `1e-8` — and it is the one that decides what the test can detect, because that expression is a finite difference, and a finite difference has an optimal step size that almost nobody computes.

Here is why it has one. Taylor gives the error of the two obvious schemes:

$$
\frac{f(x+h) - f(x)}{h} = f'(x) + \frac{h}{2} f''(x) + O(h^2)
$$

$$
\frac{f(x+h) - f(x-h)}{2h} = f'(x) + \frac{h^2}{6} f'''(x) + O(h^4)
$$

so the *truncation* error is `O(h)` and `O(h^2)`, and both fall as `h` falls. That is the half everyone remembers.

The other half is that `f(x+h)` and `f(x-h)` are floating-point numbers, and they agree to more and more digits as `h` shrinks — so their difference keeps fewer and fewer of them, and then you divide by a small number, which amplifies whatever is left. That *cancellation* error grows like `eps/h`.

Two errors, opposite directions, one step size. The total is a U.""")

    post.add(
        "",
        f"""{snip['ucurve'].markdown()}

Read the last two rows. At `h = 1e-11` and `h = 1e-14` the two schemes agree to every digit printed, and both are wrong by `1.17e-06` and `3.71e-03`. Down there the truncation term is negligible and the whole error is cancellation, which does not care which scheme produced it. Note also that the forward difference is *better* at `1e-11` than at `1e-14`, by three orders of magnitude, having taken a step a thousand times larger.

Balancing the two terms gives the scale of the optimum: for a first-order scheme, `h` against `eps/h` gives `h ~ eps^(1/2)`; for a second-order scheme, `h^2` against `eps/h` gives `h ~ eps^(1/3)`.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "",
        f"""Which is `{fd.optimal_h('forward'):.1e}` and `{fd.optimal_h('central'):.1e}`, against measured optima of `{fwd.best_h:.1e}` and `{cen.best_h:.1e}`. An order-of-magnitude argument landing within a factor of three is that argument working.

It is worth being exact about what the central difference buys, because this is usually stated wrongly. **It is not more accurate at a given step.** At `h = 1e-08` the two agree to a factor of {same8:.2f}. Its advantage is that it is *allowed a larger step* — at `h = 1e-05` it is {same5:.1e} times better, and at their respective optima, {fwd.best_error / cen.best_error:.0f} times. So "use a central difference" and "use a small step" are not two halves of the same advice. The first is worth nothing unless you also move the step *up*.

And the cost of ignoring the U: at `h = 1e-14` the central difference is **{cen.penalty_at(1e-14):.1e} times worse** than at its own optimum. Refining a step past the bottom is not a diminishing return, it is a negative one.""",
        level=3)

    post.add(
        "The same U decides what your check can detect",
        f"""None of that was about gradients, and it did not need to be: a gradient check is a numeric gradient against an analytic one, coordinate by coordinate, and each coordinate is one central difference.

So what does the U do to a *test*? A gradient check has a noise floor — the discrepancy it reports on a gradient that is perfectly correct, which is exactly the finite-difference error above. A bug is detectable when it pushes the reported discrepancy clear of that floor. Raise the floor and you raise the smallest bug the test can find.

{snip['check'].markdown()}

At `h = 1e-05` a relative error of `1e-06` in one gradient entry is caught. At `h = 1e-09` it is not, but a `1e-04` error is. At `h = 1e-13` — a step someone picked because it seemed safely small — **a gradient that is 1% wrong passes.**""",
        figures=[figs["f1"]])

    post.add(
        "",
        f"""Bisecting rather than sampling gives the boundary. The smallest relative error in one entry that the check separates from its own noise:

- `h = 1e-05`: {steps[1e-5]['detectable']:.1e}
- `h = 1e-07`: {steps[1e-7]['detectable']:.1e}
- `h = 1e-09`: {steps[1e-9]['detectable']:.1e}
- `h = 1e-11`: {steps[1e-11]['detectable']:.1e}
- `h = 1e-13`: {steps[1e-13]['detectable']:.1e}

Eight orders of magnitude in the sensitivity of a test, set by a constant that is a default in most codebases.

The two common defaults are not the disaster, and it is worth saying so plainly: `1e-07` resolves {steps[1e-7]['detectable']:.1e}, which is ample for finding a transposed index or a missing factor of two. They are a decade and a half past the optimum and they still work. The failure is at the bottom end, and it is reached by exactly the reasoning that sounds most careful — *this is an approximation, so let me make the step smaller.*""",
        level=3)

    post.add(
        "Every scale here is a power of eps, so the precision moves all of them",
        f"""`eps` has been in every formula so far: `eps^(1/2)`, `eps^(1/3)`, `eps/h`. It is not a universal constant. It is a property of the format the numbers are stored in, and it moves by thirteen orders of magnitude across the four formats a modern training run touches.

One of those is worth writing out rather than importing, because `bfloat16` is a simpler object than its reputation suggests: it is `float32` with the low sixteen mantissa bits discarded — eight bits of mantissa, `float32`'s exponent range. Four lines.

{snip['precision'].markdown()}

The exponents transfer. The optimal step moves from `{table['float64']['measured_h']:.1e}` to `{table['bfloat16']['measured_h']:.1e}` as `eps` moves twelve decades, which is what a cube root of `eps` predicts.""",
        figures=[figs["f2"]])

    post.add(
        "",
        f"""The constants do not transfer, and here I have to correct myself rather than the reader. `eps^(2/3)` is supposed to give the error floor, and it is optimistic in every row — by {table['float64']['predicted_floor'] / table['float64']['measured_error']:.0f}×, {table['float32']['predicted_floor'] / table['float32']['measured_error']:.0f}×, {table['float16']['predicted_floor'] / table['float16']['measured_error']:.0f}× and {table['bfloat16']['predicted_floor'] / table['bfloat16']['measured_error']:.0f}× — because the derivation drops the derivative factors. So the formula is the right way to *scale* a step size and the wrong way to predict an error. Hold onto that; the next section is what happens when you forget it.

There is also a factor of two hiding inside the word `eps`. The number `np.finfo` reports is the gap from 1.0 to the next representable number: `2^-52` for float64, `2^-7` for bfloat16. *Unit roundoff* is half of that. Both conventions are in the literature, and quoting bfloat16's epsilon as `3.9e-03` — which is `2^-8`, the unit roundoff — while using a formula calibrated on `np.finfo` mixes them.

And one thing the chart above understates. The flat right-hand tails are not "very inaccurate". There, `x + h` rounds back to `x`, so `f(x+h) - f(x-h)` is exactly `0.0` and the reported derivative is exactly zero. In bfloat16 that happens for **every step below `{res['collapse']['bfloat16']:.1e}`**; in float16, below `{res['collapse']['float16']:.1e}`. A gradient check in half precision with a `1e-7` default does not return a bad number. It returns zero, for every parameter, with nothing raised anywhere.""",
        level=3)

    post.add(
        "So in half precision the check is not a weak test",
        f"""Put the two facts together — the U moves with the precision, and the check's sensitivity is set by where the U's bottom is — and search for the smallest gradient error each format can resolve at its *best* step rather than at a default:

- float64: {best['float64']['detectable']:.1e}
- float32: {best['float32']['detectable']:.1e}
- float16: {best['float16']['detectable'] * 100:.1f}%
- bfloat16: {best['bfloat16']['detectable'] * 100:.0f}%

A gradient check computed in bfloat16 cannot distinguish a correct gradient from one that is 17% wrong, at any step size. Not "is less sensitive to" — cannot. The span from the first row to the last is {best['bfloat16']['detectable'] / best['float64']['detectable']:.0e}.""",
        figures=[figs["f4"]])

    post.add(
        "",
        f"""That 17% is measured, and I want to be explicit about why it had to be. Before running it I derived the figure from `eps^(2/3)`, which for bfloat16 is `{bf_derived:.1e}`, and wrote down "a bfloat16 gradient check cannot see an error below about 2.5%". The measured answer is {best['bfloat16']['detectable'] / bf_derived:.0f} times larger. The derivation was not wrong about the *scale* — the constants it drops are exactly the ones the previous section measured, and the check's decision rule contributes one more. It was wrong as a number, which is how I had used it.

The practical form of all this is short. A gradient check is a float64 instrument. If the forward pass runs in half precision, the check has to run on a cast-up copy of the model — and if that is not possible, the check is not evidence.""",
        level=3)

    post.add(
        "The way out, where it applies",
        f"""The U exists because two nearly equal numbers get subtracted. Remove the subtraction and the U goes with it.

Perturb along the *imaginary* axis instead. If `f` is analytic then `f(x + ih) = f(x) + i h f'(x) - h^2 f''(x)/2 - ...`, so the imaginary part is `h f'(x) + O(h^3)` and the derivative is `Im f(x+ih) / h`. The real part carries `f(x)` and the imaginary part carries the derivative; nothing is subtracted from anything of comparable size, so there is no cancellation for `1/h` to amplify and `h` can be as small as you like.

{snip['complex'].markdown()}

Exactly zero error at `h = 1e-200`. Not "accurate to machine precision" — the measured difference from `cos(1)` is `0.0`.""",
        figures=[figs["f3"]])

    post.add(
        "",
        f"""The price is that `f` must be complex-analytic *and implemented so that it stays that way*, and the operations that break it are not the list I would have written down.

`abs` breaks it: `d/dx |x|` at 1.5 comes back `0.0` instead of `1.0`. A real-part cast breaks it: `np.real(z)**2` comes back `0.0` instead of `3.0`. Neither raises anything.

`x * abs(x)` is the interesting one. It preserves the real part perfectly, so the cheap sanity check — does `Re f(x+ih)` still equal `f(x)`? — passes it. And the derivative comes back `{res['breaks']['x*abs(x)'][0]:.1f}` against a truth of `3.0`: **wrong by exactly a factor of two**, which is the most dangerous kind of wrong, because it reads as a units slip rather than as a broken method.

What does *not* break it was a surprise, and it went into the tests after I asserted the opposite in a docstring. `np.maximum` and `np.minimum` compare complex numbers by real part first, so a ReLU network differentiates correctly away from the kink. Checked against a central difference on a small MLP at three inputs: agreement to {relu[0.5]['gap']:.1e}, {relu[-0.5]['gap']:.1e} and {relu[2.0]['gap']:.1e}.

So the guard for this method cannot be a real-part check alone. It has to corroborate against a central difference at that difference's own optimum — which is the one place in this episode where the U-curve is the *reference* rather than the problem.""",
        level=3)

    post.add(
        "A footnote that is not a footnote: conditioning is not stability",
        f"""One distinction to separate before closing, because it gets collapsed constantly and both halves of it are in this episode.

The condition number of an evaluation, `|x f'(x) / f(x)|`, says how a relative error in the input becomes one in the output. It is a property of the *problem*. Two measurements bracket what it does and does not tell you.

`f(x) = x - 1` at `x = 1.0001` has a condition number of `{easy['condition_number']:.0e}`: a relative perturbation of the input is amplified ten thousandfold. Its derivative comes back to `{easy['best_error']:.1e}` — exactly, in fact, because a linear function has no truncation term to trade against cancellation. **A badly conditioned evaluation does not imply a hard derivative.**

And `1 - cos x` at `x = 1e-4` has a condition number of `{cancel['condition_number']:.1f}`, as well conditioned as anything, while the obvious way to evaluate it loses seven digits: relative error `{cancel['naive_relative_error']:.1e}`, against `{cancel['rewritten_relative_error']:.1e}` for `2 sin^2(x/2)`, which is the same function by an exact identity. **A well conditioned problem can have an unstable algorithm.**

That is Higham's distinction, and it belongs here because the U-curve is an instance of the second kind. The derivative of `sin` at 1 is a perfectly conditioned problem. Everything that goes wrong in this episode is the algorithm.""")

    post.add(
        "What to keep",
        f"""1. A gradient check is a finite difference. Its error is a U in the step size, so a smaller step is not a safer one, and past the optimum it is strictly worse — {cen.penalty_at(1e-14):.1e} times worse at `h = 1e-14`.
2. The optimum's scale is `eps^(1/2)` for a forward difference and `eps^(1/3)` for a central one: `{fd.optimal_h('forward'):.1e}` and `{fd.optimal_h('central'):.1e}` in float64.
3. A central difference is not more accurate at a given step — at `1e-08` the two agree to {same8:.2f}×. It is *allowed* a larger one. Use both halves of that or neither.
4. The step sets what the check can detect across eight orders of magnitude: {steps[1e-5]['detectable']:.1e} at `h = 1e-05`, {steps[1e-13]['detectable']:.1e} at `h = 1e-13`.
5. Every scale is a power of `eps`, so the precision moves all of them. The check resolves {best['float64']['detectable']:.1e} in float64 and {best['bfloat16']['detectable'] * 100:.0f}% in bfloat16. Run gradient checks in float64.
6. `Im f(x+ih)/h` has no cancellation and no optimum, and its failure modes are quiet. `x * abs(x)` returns exactly half the right answer; a ReLU network is fine.
7. The condition number is about the problem. Whether your algorithm keeps the digits the problem allows is a separate question with a separate answer.""")

    post.add(
        "Exercise",
        f"""Find the gradient check in your codebase and read its step size. Then run it three times on a gradient you have deliberately broken by 0.1% in one entry: at the step it currently uses, at `1e-05`, and at `{fd.optimal_h('central'):.0e}`.

If all three catch it, your check is fine and you have learned the cheapest possible thing. If the current step misses it and `1e-05` catches it, you have been running a test that would have passed a real bug, and the fix is one character.

Then check the dtype the differences are computed in. If the forward pass is in half precision, the interesting question is not what the step is — it is whether the differences are coming back as exact zeros.

Next episode: splitting a matmul's contraction across four accumulators is exact algebra and inexact arithmetic, so the same model on the same input can produce different logits. Whether it produces a different *token* turns out to be a question about precision rather than about determinism — true in bfloat16, and false in float32.""")

    post.hero = figs["hero"]
    return post


def main() -> Post:
    return build()


if __name__ == "__main__":
    r = compute()
    print(f"forward  best h={r['forward'].best_h:.3g} err={r['forward'].best_error:.3g}"
          f"   predicted h={fd.optimal_h('forward'):.3g}")
    print(f"central  best h={r['central'].best_h:.3g} err={r['central'].best_error:.3g}"
          f"   predicted h={fd.optimal_h('central'):.3g}")
    print(f"  ratio at own optima: {r['forward'].best_error / r['central'].best_error:.0f}x")
    for h, d in r["at_same"].items():
        print(f"  at h={h:.0e}: forward {d['forward']:.3g}  central {d['central']:.3g}"
              f"  ratio {d['forward']/d['central']:.2f}x")
    print(f"  central penalty at 1e-16: {r['central'].penalty_at(1e-16):.3g}x")
    print("\ngradient check, float64:")
    for s in r["steps"]:
        print(f"  h={s['h']:.0e}  floor={s['floor']:.2e}  detectable={s['detectable']:.2e}")
    print("\nprecision table:")
    for t in r["table"]:
        print(f"  {t['precision']:9s} eps={t['eps']:.3g}  h {t['predicted_h']:.2g}/"
              f"{t['measured_h']:.2g}  err {t['predicted_floor']:.2g}/{t['measured_error']:.2g}"
              f"  optimism {t['predicted_floor']/t['measured_error']:.0f}x")
    print("\nwhat the check can see:")
    for p, b in r["best"].items():
        print(f"  {p:9s} best h={b['h']:.3g}  detectable={b['detectable']:.3g}")
    print(f"  span float64 -> bfloat16: "
          f"{r['best']['bfloat16']['detectable'] / r['best']['float64']['detectable']:.2g}x")
    print("\ncomplex step:")
    print("  extreme:", {f"{k:.0e}": v for k, v in r["extreme"].items()})
    for name, (got, truth, safe) in r["breaks"].items():
        print(f"  {name:10s} got {got:.4g} truth {truth:.4g}  guard says safe={safe}")
    for x, d in r["relu"].items():
        print(f"  relu at x={x:+.1f}: gap {d['gap']:.2g}")
    print("\nconditioning vs stability:")
    for c in r["cancellation"]:
        print(f"  x={c['x']:.0e}  cond={c['condition_number']:.3f}  "
              f"naive rel err {c['naive_relative_error']:.2g}  "
              f"rewritten {c['rewritten_relative_error']:.2g}")
    print("  easy derivative:", r["easy_derivative"])
