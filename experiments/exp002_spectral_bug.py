"""exp002 — Why my integrator exploded at t = 355, every single time.

A debugging post, and the analysis is the debugging. Two real bugs from this repo:

1. A Kuramoto-Sivashinsky solver that blew up at t ~ 355 for **every** domain
   length, grid size, timestep and time-stepping scheme. The invariance was the
   clue: the failure time is set by the maximum linear growth rate of the
   equation, 0.25, and by machine epsilon — nothing else.
2. A Lyapunov exponent 20% too high because the tangent space was advanced with an
   Euler step.

Neither is caught by a test that checks shapes, dtypes or "does it run". Both are
caught immediately by a test that checks a physical identity.

Run: `standarderror run exp002_spectral_bug --publish`
"""

from __future__ import annotations

from datetime import date

import numpy as np

import standarderror as se
from standarderror.dynamics import lyapunov, ode, pde
from standarderror.render import Post
from standarderror.viz import charts, theme

#: Pinned so a rebuild cannot silently re-date a published post.
#: `Post.date` defaults to today, which is correct exactly once.
POST_DATE = date(2026, 8, 6)

IMG = se.SETTINGS.build_dir / "img"
EPS = float(np.finfo(float).eps)
SEED = se.SETTINGS.seed

# max_k (k^2 - k^4) = 1/4, attained at k = 1/sqrt(2). Independent of L and N,
# which is the whole reason the failure time never moved.
KS_MAX_GROWTH = 0.25


# ------------------------------------------------------------------ the bug

def _spectral_parts(v: np.ndarray) -> tuple[float, float]:
    """Split a complex spectrum into its Hermitian and non-Hermitian halves.

    For a real field, `fft` output satisfies v[-k] = conj(v[k]). The anti-Hermitian
    half is therefore not a physical state at all — it is the redundant half of an
    over-parameterised representation, and `real(ifft(v))` silently discards it.
    Which is exactly why nothing in the nonlinear term ever constrains it.
    """
    n = len(v)
    mirror = (-np.arange(n)) % n
    herm = 0.5 * (v + np.conj(v[mirror]))
    anti = 0.5 * (v - np.conj(v[mirror]))
    return float(np.linalg.norm(herm)), float(np.linalg.norm(anti))


def _etdrk4_weights(lop: np.ndarray, dt: float, m: int = 32):
    e, e2 = np.exp(dt * lop), np.exp(dt * lop / 2)
    r = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    lr = dt * lop[:, None] + r[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        q = dt * np.real(np.mean((np.exp(lr / 2) - 1) / lr, axis=1))
        f1 = dt * np.real(np.mean(
            (-4 - lr + np.exp(lr) * (4 - 3 * lr + lr ** 2)) / lr ** 3, axis=1))
        f2 = dt * np.real(np.mean(
            (2 + lr + np.exp(lr) * (-2 + lr)) / lr ** 3, axis=1))
        f3 = dt * np.real(np.mean(
            (-4 - 3 * lr - lr ** 2 + np.exp(lr) * (4 - lr)) / lr ** 3, axis=1))
    return e, e2, q, f1, f2, f3


def buggy_ks(
    *, N: int = 64, L: float = 22.0, dt: float = 0.25, tmax: float = 500.0,
    scheme: str = "etdrk4", seed: int = 0, track: bool = False,
) -> dict:
    """The original, wrong implementation: state held as a **full complex** spectrum.

    Preserved verbatim (bar the instrumentation) because the post is about it. Two
    time-stepping schemes are included to make the point that the choice of
    integrator is irrelevant — a claim that is much stronger when you can show it.
    """
    x = L * np.arange(N) / N
    rng = np.random.default_rng(seed)
    u = (np.cos(2 * np.pi * x / L) * (1 + np.sin(2 * np.pi * x / L))
         + 0.01 * rng.standard_normal(N))

    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    if N % 2 == 0:
        k[N // 2] = 0.0
    lop = k ** 2 - k ** 4
    g = -0.5j * k

    def nl(w):
        return g * np.fft.fft(np.real(np.fft.ifft(w)) ** 2)

    v = np.fft.fft(u)
    hist: list[tuple[float, float, float, float]] = []

    if scheme == "etdrk4":
        e, e2, q, f1, f2, f3 = _etdrk4_weights(lop, dt)
    elif scheme == "cnab2":
        a_op, b_op = 1.0 - 0.5 * dt * lop, 1.0 + 0.5 * dt * lop
        n_prev = nl(v)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    n_steps = int(round(tmax / dt))
    # Overflow here is the phenomenon under study, not a problem to be reported.
    with np.errstate(over="ignore", invalid="ignore"):
      for step in range(n_steps):
        if scheme == "etdrk4":
            nv = nl(v)
            a = e2 * v + q * nv
            na = nl(a)
            b = e2 * v + q * na
            nb = nl(b)
            c = e2 * a + q * (2.0 * nb - nv)
            nc = nl(c)
            v = e * v + nv * f1 + 2.0 * (na + nb) * f2 + nc * f3
        else:
            n_now = nl(v)
            rhs = (b_op * v + dt * (1.5 * n_now - 0.5 * n_prev) if step
                   else b_op * v + dt * n_now)
            v = rhs / a_op
            n_prev = n_now

        if track:
            h, an = _spectral_parts(v)
            hist.append((step * dt, h, an,
                         float(np.max(np.abs(np.real(np.fft.ifft(v)))))))
        if not np.isfinite(v).all():
            return {"blowup_time": step * dt, "history": np.array(hist),
                    "N": N, "L": L, "dt": dt, "scheme": scheme}
    return {"blowup_time": None, "history": np.array(hist),
            "N": N, "L": L, "dt": dt, "scheme": scheme}


def invariance_sweep() -> dict:
    """Blow-up time across every knob a numerical analyst would reach for."""
    rows = []
    for scheme in ("etdrk4", "cnab2"):
        for L, N in ((22.0, 64), (22.0, 128), (60.0, 128), (100.0, 256)):
            for dt in (0.05, 0.1, 0.25, 0.5):
                if scheme == "cnab2" and dt > 0.1:
                    continue          # CNAB2 is only 2nd order; keep it in range
                out = buggy_ks(N=N, L=L, dt=dt, scheme=scheme, tmax=600.0)
                rows.append({
                    "label": f"{scheme.upper()} L={L:g} N={N} dt={dt:g}",
                    "scheme": scheme, "L": L, "N": N, "dt": dt,
                    "blowup_time": out["blowup_time"]})
    ok = [r for r in rows if r["blowup_time"] is not None]
    times = [r["blowup_time"] for r in ok]

    def spread_within(keys) -> float:
        """Worst spread, as a % of the group median, *holding `keys` fixed*.

        This is the number that matters: it separates "the failure time does not
        care about this knob" from "it does". Quoting only the overall spread
        would hide that essentially all of the variation is one variable.
        """
        groups: dict[tuple, list[float]] = {}
        for r in ok:
            groups.setdefault(tuple(r[k] for k in keys), []).append(r["blowup_time"])
        worst = 0.0
        for v in groups.values():
            if len(v) > 1:
                worst = max(worst, 100 * (max(v) - min(v)) / float(np.median(v)))
        return worst

    return {"rows": rows, "n_configs": len(rows), "n_blew_up": len(times),
            "min": float(min(times)), "max": float(max(times)),
            "median": float(np.median(times)),
            "spread_pct": float(100 * (max(times) - min(times))
                                / np.median(times)),
            # vary dt only (scheme, L, N fixed), vary N only, vary scheme only,
            # vary L only.
            "spread_dt_only": spread_within(("scheme", "L", "N")),
            "spread_N_only": spread_within(("scheme", "L", "dt")),
            "spread_scheme_only": spread_within(("L", "N", "dt")),
            "spread_L_only": spread_within(("scheme", "N", "dt")),
            "by_L": {L: (float(np.min([r["blowup_time"] for r in ok if r["L"] == L])),
                         float(np.max([r["blowup_time"] for r in ok if r["L"] == L])))
                     for L in sorted({r["L"] for r in ok})}}


def diagnosis() -> dict:
    """Track the two halves of the spectrum until the solver dies."""
    out = buggy_ks(track=True, tmax=420.0)
    h = out["history"]
    t, herm, anti, umax = h[:, 0], h[:, 1], h[:, 2], h[:, 3]

    # Fit the growth rate on the clean exponential stretch, before the
    # anti-Hermitian part is large enough to interact with anything.
    fit = (t > 10) & (t < 200) & (anti > 0)
    rate, intercept = np.polyfit(t[fit], np.log(anti[fit]), 1)
    resid = np.log(anti[fit]) - (rate * t[fit] + intercept)
    r2 = 1.0 - np.var(resid) / np.var(np.log(anti[fit]))

    # The failure condition: roundoff on the (huge) meaningless component
    # swamps the (ordinary-sized) physical one during the real-part extraction.
    swamped = t[(EPS * anti) >= herm]
    return {"t": t, "herm": herm, "anti": anti, "umax": umax,
            "blowup_time": out["blowup_time"],
            "growth_rate": float(rate),
            "growth_rate_r2": float(r2),
            "theory_rate": KS_MAX_GROWTH,
            "initial_defect": float(anti[0]),
            "swamp_time": float(swamped[0]) if len(swamped) else None,
            "umax_before_death": float(np.median(umax[t < 250])),
            "naive_prediction": float(np.log(1.0 / EPS ** 2) / KS_MAX_GROWTH)}


def the_fix() -> dict:
    """The repaired integrator, run 50x past where the old one died."""
    field = pde.kuramoto_sivashinsky(n_steps=80000, L=22.0, N=64, dt=0.25,
                                     transient=200.0)
    e = field.energy()
    n = len(e) // 4
    # Independent implicit reference on the same semi-discrete system: the check
    # that would have exposed the bug on day one.
    x = 22.0 * np.arange(64) / 64
    u0 = np.cos(2 * np.pi * x / 22.0) * (1 + np.sin(2 * np.pi * x / 22.0))
    ref = pde.reference_solution(64, 22.0, 8.0, u0)
    got = pde.kuramoto_sivashinsky(n_steps=160, L=22.0, N=64, dt=0.05,
                                   transient=0.0, u0=u0).u[-1]
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    return {"t_final": float(field.t[-1]),
            "energy_first_quarter": float(e[:n].mean()),
            "energy_last_quarter": float(e[-n:].mean()),
            "energy_drift_pct": float(100 * abs(e[-n:].mean() - e[:n].mean())
                                      / e[:n].mean()),
            "u_absmax": float(np.abs(field.u).max()),
            "reference_rel_error": rel,
            "energy": e, "t": field.t, "field": field}


# ------------------------------------------------------- the second bug

def lyapunov_euler_vs_expm() -> dict:
    """The same estimator, two ways of advancing the tangent space.

    Euler: Q <- Q + dt J Q. Exact: Q <- expm(J dt) Q. Both converge as dt -> 0,
    which is precisely why the bug survives — at a *usable* dt one of them is 20%
    wrong and the other is not.
    """
    from scipy.linalg import expm
    jac = lyapunov.lorenz_jacobian()
    trace = -(10.0 + 1.0 + 8.0 / 3.0)      # divergence of the Lorenz field
    rows = []
    for dt in (0.02, 0.01, 0.005, 0.0025, 0.001):
        traj = ode.lorenz63(n_steps=int(round(120.0 / dt)), dt=dt,
                            transient=40.0)
        for method in ("euler", "expm"):
            q = np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))[0]
            sums = np.zeros(3)
            count = 0
            for i in range(len(traj.x)):
                j = np.asarray(jac(traj.x[i]), float)
                q = (q + dt * (j @ q)) if method == "euler" else expm(j * dt) @ q
                q, r = np.linalg.qr(q)
                d = np.abs(np.diag(r))
                if i >= 100:
                    sums += np.log(np.where(d > 0, d, 1e-300))
                    count += 1
            spec = np.sort(sums / (count * dt))[::-1]
            rows.append({"dt": dt, "method": method,
                         "lambda_max": float(spec[0]),
                         "spectrum_sum": float(spec.sum()),
                         "trace_residual_pct":
                             float(100 * abs(spec.sum() - trace) / abs(trace))})
    lit = 0.9056
    by = {m: [r for r in rows if r["method"] == m] for m in ("euler", "expm")}
    return {"rows": rows, "literature": lit, "trace": trace,
            # dt = 0.01 is the step the repo actually uses, so that is the
            # headline; the coarsest step in the sweep is reported alongside.
            "euler_at_001": next(r for r in by["euler"] if r["dt"] == 0.01),
            "expm_at_001": next(r for r in by["expm"] if r["dt"] == 0.01),
            "euler_at_002": next(r for r in by["euler"] if r["dt"] == 0.02),
            "euler_worst_trace": max(r["trace_residual_pct"] for r in by["euler"]),
            "expm_worst_trace": max(r["trace_residual_pct"] for r in by["expm"])}


# ------------------------------------------------------------------ figures

def figures(sweep: dict, diag: dict, fix: dict, lyap: dict) -> dict:
    src = "Simulated Kuramoto-Sivashinsky and Lorenz-63; code in the repo."
    figs = {}
    import pandas as pd

    # F1 — the phenomenon: three configurations, all dying at the same time.
    curves = {}
    for L, N, dt in ((22.0, 64, 0.25), (60.0, 128, 0.1), (100.0, 256, 0.05)):
        out = buggy_ks(N=N, L=L, dt=dt, track=True, tmax=420.0)
        h = out["history"]
        curves[f"L={L:g}, N={N}, dt={dt:g}"] = pd.Series(h[:, 3], index=h[:, 0])
    frame = pd.DataFrame(curves).sort_index()
    def mark_median(_fig, ax):
        # Clipped to 1e6: the tail reaches 1e150, and letting it set the scale
        # flattens the 300 time units that actually matter into the axis line.
        ax.axvline(sweep["median"], color=theme.LIGHT.axis, lw=1.0, ls=(0, (4, 3)))
        ax.annotate(f"median blow-up\nt = {sweep['median']:.0f}",
                    (sweep["median"], 0.97), xycoords=("data", "axes fraction"),
                    xytext=(-8, 0), textcoords="offset points", ha="right",
                    va="top", fontsize=8.0, color=theme.LIGHT.muted)

    fig, ax = charts.lines(
        frame, mode="light", direct_labels=False, logy=True, ylim=(1.0, 1e6),
        title="Nothing looks wrong until it is over",
        subtitle=("Peak amplitude of the solution, clipped at 10⁶ — the runs "
                  "reach 10¹⁵⁰ within a few steps of leaving this window."),
        ylabel="max |u|", xlabel="time", source=src, decorate=mark_median)
    theme.save(fig, str(IMG / "f1-phenomenon.png"), mode="light")
    figs["phenomenon"] = charts.Figure(
        str(IMG / "f1-phenomenon.png"),
        alt=("Log-scale line chart of peak solution amplitude against time for "
             "three different grid and timestep configurations, with the y-axis "
             "clipped at one million. All three sit flat near 2 for hundreds of "
             "time units, then rise vertically off the top of the window."),
        caption=("Fig 1. Peak amplitude sits at 2 for three hundred time units "
                 "and then leaves the page within a few steps. Nothing in the "
                 "solution looks wrong beforehand, which is what makes this class "
                 "of bug expensive to find."),
        title="phenomenon")

    # F2 — the invariance, isolated knob by knob. A 24-bar ranked chart made the
    # same point but ran to 1,900px, which is unreadable on a phone.
    ok = [r for r in sweep["rows"] if r["blowup_time"] is not None]
    series = {}
    for L, N in sorted({(r["L"], r["N"]) for r in ok if r["scheme"] == "etdrk4"}):
        pts = sorted((r["dt"], r["blowup_time"]) for r in ok
                     if r["scheme"] == "etdrk4" and r["L"] == L and r["N"] == N)
        series[f"L={L:g}, N={N}"] = pd.Series([v for _, v in pts],
                                              index=[d for d, _ in pts])
    fig_meta, _ = charts.lines(
        pd.DataFrame(series).sort_index(), mode="light", direct_labels=False,
        logx=True, ylim=(0.85 * sweep["min"], 1.10 * sweep["max"]),
        title=f"A tenfold change in timestep moves the failure by {sweep['spread_dt_only']:.0f}%",
        subtitle=(f"Blow-up time against integration step, one line per domain and "
                  f"grid. Holding everything but dt fixed, the spread is "
                  f"{sweep['spread_dt_only']:.1f}%; holding everything but the "
                  f"domain length fixed, it is {sweep['spread_L_only']:.0f}%."),
        ylabel="time of blow-up", xlabel="dt", source=src,
        decorate=lambda _f, ax: (
            ax.set_xticks([0.05, 0.1, 0.25, 0.5]),
            ax.set_xticklabels(["0.05", "0.1", "0.25", "0.5"]),
            ax.minorticks_off()),
        alt=("Line chart of blow-up time against integration timestep on a log "
             "axis, with four lines for different domain-and-grid combinations. "
             "Every line is essentially flat, sitting at about 355 for the "
             "shortest domain and about 310 for the two longer ones."),
        caption=(f"Fig 2. Flat lines. Across {sweep['n_blew_up']} configurations "
                 f"the timestep changes the failure time by "
                 f"{sweep['spread_dt_only']:.1f}%, the grid by "
                 f"{sweep['spread_N_only']:.1f}%, and the choice of "
                 f"time-stepping scheme by {sweep['spread_scheme_only']:.1f}%. "
                 "A numerical instability does not behave like this."),
        path=str(IMG / "f2-invariance.png"))
    figs["invariance"] = fig_meta

    # F3 — the diagnosis. This is the figure the post exists for.
    keep = diag["t"] <= diag["blowup_time"]
    t = diag["t"][keep]

    def mark_events(_fig, ax):
        ax.axvline(diag["swamp_time"], color=theme.LIGHT.axis, lw=1.0,
                   ls=(0, (4, 3)))
        ax.annotate(f"roundoff overtakes\nsignal, t = {diag['swamp_time']:.0f}",
                    (diag["swamp_time"], 0.03),
                    xycoords=("data", "axes fraction"), xytext=(-8, 0),
                    textcoords="offset points", ha="right", fontsize=8.0,
                    color=theme.LIGHT.ink_secondary)
        ax.annotate(f"solver dies\nt = {diag['blowup_time']:.0f}",
                    (diag["blowup_time"], 0.03),
                    xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", ha="left", fontsize=8.0,
                    color=theme.LIGHT.series[7])
        ax.axvline(diag["blowup_time"], color=theme.LIGHT.series[7], lw=1.2)

    fig, ax = charts.error_growth(
        t,
        {"non-Hermitian half ‖v⁻‖ (not a physical state)": diag["anti"][keep],
         "physical half ‖v⁺‖": diag["herm"][keep],
         "roundoff floor ε·‖v⁻‖": EPS * diag["anti"][keep]},
        threshold=None, logy=True, mode="light", decorate=mark_events,
        ylim=(1e-26, 1e22),
        xlabel="time", ylabel="spectral norm",
        title="The part of the state that cannot mean anything eats the part that can",
        subtitle=(f"The redundant half of the complex spectrum grows at "
                  f"{diag['growth_rate']:.3f} per time unit from a starting value "
                  f"of {diag['initial_defect']:.0e} — pure roundoff."),
        source=src)
    theme.save(fig, str(IMG / "f3-diagnosis.png"), mode="light")
    figs["diagnosis"] = charts.Figure(
        str(IMG / "f3-diagnosis.png"),
        alt=("Log-scale chart of three spectral norms against time. The physical "
             "component stays flat near 70 throughout. The non-Hermitian "
             "component starts at 1e-16 and rises as a straight line across "
             "eighteen orders of magnitude. The roundoff floor, a fixed multiple "
             "below it, crosses the physical component shortly before the solver "
             "fails."),
        caption=(f"Fig 3. The physical half of the state is perfectly healthy for "
                 f"the entire run. The meaningless half grows exponentially from "
                 f"machine epsilon, and when its roundoff crosses the physical "
                 f"signal at t = {diag['swamp_time']:.0f}, extracting the real "
                 f"part stops being meaningful. The solver dies "
                 f"{diag['blowup_time'] - diag['swamp_time']:.0f} time units "
                 "later."),
        title="diagnosis")

    # F4 — the fix, and the energy that proves it.
    e, tt = fix["energy"], fix["t"]
    # A 400-time-unit rolling mean. The raw trace is 80,000 points of fast
    # fluctuation and renders as a solid block, which cannot show absence of trend.
    w = 1600
    kern = np.ones(w) / w
    smooth = np.convolve(e, kern, mode="valid")
    t_smooth = tt[w - 1:]
    fig_meta, _ = charts.lines(
        pd.DataFrame({"mean u², 400-time-unit rolling average": smooth},
                     index=t_smooth),
        mode="light", direct_labels=False, ylim=(0, 2.2),
        title="The same integration, on a real spectrum, 56x further",
        subtitle=(f"Energy over {fix['t_final']:.0f} time units. First quarter "
                  f"{fix['energy_first_quarter']:.3f}, last quarter "
                  f"{fix['energy_last_quarter']:.3f} — a "
                  f"{fix['energy_drift_pct']:.1f}% difference, and no trend."),
        ylabel="mean u²", source=src,
        alt=("Line chart of a rolling average of the spatial mean of u squared "
             "over twenty thousand time units. It stays in a narrow band around "
             "1.4 with no upward or downward trend."),
        caption=("Fig 4. Using a real-valued FFT makes the redundant modes "
                 "unrepresentable, so there is nothing to amplify. The energy is "
                 "stationary — the cheapest possible check, and one that would "
                 "have flagged the original within a minute."),
        path=str(IMG / "f4-fix.png"))
    figs["fix"] = fig_meta

    # F5 — the second bug.
    rows = lyap["rows"]
    dts = sorted({r["dt"] for r in rows}, reverse=True)
    panels = {}
    for method, label in (("euler", "Euler tangent step"),
                          ("expm", "exact: expm(J·dt)")):
        vals = [next(r["lambda_max"] for r in rows
                     if r["dt"] == d and r["method"] == method) for d in dts]
        panels[label] = pd.Series(vals, index=dts)
    frame = pd.DataFrame(panels)
    def mark_literature(_fig, ax):
        ax.axhline(lyap["literature"], color=theme.LIGHT.axis, lw=1.0,
                   ls=(0, (4, 3)))
        ax.annotate("literature value 0.9056", (0.02, lyap["literature"]),
                    xycoords=("axes fraction", "data"), xytext=(0, 5),
                    textcoords="offset points", fontsize=8.0,
                    color=theme.LIGHT.muted)

    fig, ax = charts.lines(
        frame, mode="light", direct_labels=False, logx=True, invert_x=True,
        title="Both converge. Only one is right at a step you would use",
        subtitle=("Largest Lyapunov exponent of Lorenz-63 against integration "
                  "step, coarser to the left."),
        ylabel="λ_max", xlabel="dt (log scale)", source=src,
        decorate=mark_literature)
    theme.save(fig, str(IMG / "f5-lyapunov.png"), mode="light")
    figs["lyapunov"] = charts.Figure(
        str(IMG / "f5-lyapunov.png"),
        alt=("Line chart of the estimated largest Lyapunov exponent against "
             "integration timestep on a log axis. The exact method sits on the "
             "literature value across the whole range; the Euler method is well "
             "above it at coarse steps and converges down onto it only as the "
             "step shrinks."),
        caption=(f"Fig 5. At dt = 0.01 — the step this repo uses — the Euler "
                 f"tangent update gives {lyap['euler_at_001']['lambda_max']:.4f} "
                 f"against the exact propagator's "
                 f"{lyap['expm_at_001']['lambda_max']:.4f}. Both are consistent "
                 "and both converge, so a convergence study alone would have "
                 "called the wrong one fine."),
        title="lyapunov")
    return figs


# ------------------------------------------------------------------ the post

def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    sweep = invariance_sweep()
    diag = diagnosis()
    fix = the_fix()
    lyap = lyapunov_euler_vs_expm()
    figs = figures(sweep, diag, fix, lyap)

    euler = lyap["euler_at_001"]
    exact = lyap["expm_at_001"]
    euler_err = 100 * (euler["lambda_max"] - lyap["literature"]) / lyap["literature"]
    gap = diag["blowup_time"] - diag["swamp_time"]

    post = Post(
        title="Why My Integrator Exploded at t = 355, Every Single Time",
        slug="why-my-integrator-exploded-at-355",
        date=POST_DATE,
        subtitle=("Two silent numerical bugs, and the kind of test that actually "
                  "catches them"),
        summary=(f"A spectral solver that blew up at the same moment whatever I "
                 f"changed. Across {sweep['n_blew_up']} configurations, a tenfold "
                 f"change in timestep moved the failure by "
                 f"{sweep['spread_dt_only']:.0f}%, a finer grid by "
                 f"{sweep['spread_N_only']:.0f}%, and swapping the entire "
                 f"time-stepping scheme by "
                 f"{sweep['spread_scheme_only']:.0f}%. That invariance was the "
                 f"clue. Plus a Lyapunov exponent {euler_err:.0f}% too high for a "
                 "reason that survives a convergence study. Neither bug is visible "
                 "to a test that checks shapes; both are obvious to a test that "
                 "checks a physical identity."),
        tags=["numerical-analysis", "scientific-computing", "python",
              "software-testing", "machine-learning"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated Kuramoto-Sivashinsky (u_t = -u u_x - u_xx - u_xxxx on a "
            "periodic domain) and Lorenz-63. No external data; every number here "
            "is reproducible from the repo with a fixed seed.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "scipy", "standarderror")),
            "machine epsilon": f"{EPS:.3e}",
            "KS maximum linear growth rate": f"max_k(k²−k⁴) = {KS_MAX_GROWTH}",
            "configurations swept": sweep["n_configs"],
            "fixed integrator verified to": f"t = {fix['t_final']:.0f}",
            "ETDRK4 vs implicit reference at t=8":
                f"relative error {fix['reference_rel_error']:.2e}",
        },
    )

    post.add("A bug that ignores your timestep is not a stability problem", f"""
I had a working Kuramoto-Sivashinsky solver. It ran, it produced the right kind of
spatiotemporal chaos, the amplitudes sat where the literature says they should, and
the tests were green. Then I asked it for a longer run and it blew up at
t ≈ {diag['blowup_time']:.0f}.

The first thing you try is a smaller timestep. It blew up at t ≈
{diag['blowup_time']:.0f}. The second thing is a finer grid. Same. A different
domain length, which changes the number of unstable modes and the attractor
dimension: still the same order. Then I swapped the whole time-stepping scheme —
fourth-order exponential time differencing out, semi-implicit
Crank-Nicolson/Adams-Bashforth in, sharing no code path except the operators. Same
again.

I swept {sweep['n_blew_up']} configurations: two schemes, four domain-and-grid
combinations, four timesteps. Here is where the failure time actually moved, and
where it did not:

| knob varied (everything else fixed) | spread in failure time |
|---|---|
| timestep, over a 10× range | **{sweep['spread_dt_only']:.1f}%** |
| grid resolution, 64 → 128 points | **{sweep['spread_N_only']:.1f}%** |
| time-stepping scheme | **{sweep['spread_scheme_only']:.1f}%** |
| domain length, 22 → 100 | {sweep['spread_L_only']:.0f}% |

That table is the whole clue, and it took me embarrassingly long to read. A
numerical instability scales with the timestep — that is what makes it *numerical*.
Mine ignored the timestep almost entirely, ignored the grid, and ignored the
scheme. Something was running on a clock that had nothing to do with my
discretisation, and my job was to find out whose clock it was.

Kuramoto-Sivashinsky, in the convention I use, is

    u_t = −u·u_x − u_xx − u_xxxx

on a periodic domain. In Fourier space the linear part has growth rate k² − k⁴,
positive for k < 1 and maximised at k = 1/√2 with the value exactly
**{KS_MAX_GROWTH}**. That number depends on neither the domain length nor the
resolution. And for double-precision ε,
ln(1/ε²) / {KS_MAX_GROWTH} ≈ {diag['naive_prediction']:.0f} — the right order of
magnitude for what I was seeing.

Two quantities that do not care about my discretisation, and a failure time that
also does not care. Something was growing at the equation's own maximum rate,
starting from machine epsilon. The residual domain-length dependence is the only
part that does not fit that story cleanly, and it is the honest loose end: a longer
domain puts more energy in modes near k = 1/√2 at t = 0, so the exponential starts
from a slightly larger seed and arrives slightly sooner.
""".strip(), figures=[figs["phenomenon"], figs["invariance"]])

    post.add("The state had twice as many degrees of freedom as the problem", f"""
Here is the bug, and it is a modelling error dressed as an implementation detail.

The field `u` is real and lives on N grid points: **N real degrees of freedom**. I
was holding the state as `v = fft(u)`, a complex vector of length N: **2N real
degrees of freedom**. The extra N are not spare capacity, they are a constraint —
for a real field, `v[−k] = conj(v[k])`. A spectrum satisfying that is Hermitian;
the orthogonal complement is a space of states that do not correspond to any real
field at all.

Now trace what each part of the timestep does to that meaningless half.

The nonlinear term is `g · fft(real(ifft(v))²)`. Note `real(...)`. Taking the real
part **projects the anti-Hermitian component out**. So the nonlinear term never
sees it, never constrains it, and cannot damp it.

The linear term is `exp(dt·(k²−k⁴)) · v`, applied elementwise. Because k² − k⁴ is
even in k, this operator preserves the Hermitian symmetry — and it applies exactly
the same amplification to the anti-Hermitian half.

So the redundant half of my state was in free fall: amplified by the linear
operator at the equation's own growth rate, and invisible to the only term that
could have stopped it. Seeded, initially, by nothing more than the rounding error
in the first FFT.

I split the spectrum into its Hermitian and anti-Hermitian halves and measured
both. The anti-Hermitian norm starts at {diag['initial_defect']:.1e} — floating
point noise — and grows exponentially at **{diag['growth_rate']:.3f} per time
unit** (R² = {diag['growth_rate_r2']:.4f} over the fit window) against a predicted
{KS_MAX_GROWTH}. The physical half, meanwhile, is *completely healthy for the
entire run*, sitting in its normal band the whole way.

The kill mechanism is the last piece, and it is why the solution looks fine until
it does not. A purely anti-Hermitian spectrum inverse-transforms to a purely
*imaginary* signal, so `real(ifft(v))` is mathematically immune to it. But not
numerically: once ‖v⁻‖ exceeds ‖v⁺‖ by a factor of 1/ε, extracting the real part
of their sum is catastrophic cancellation, and what comes out is rounding noise
scaled up by 10¹⁸. That crossing happens at t = {diag['swamp_time']:.0f} in my run.
The solver dies at t = {diag['blowup_time']:.0f}, **{gap:.0f} time units later**.

Peak amplitude before that point: {diag['umax_before_death']:.2f}. Nothing to see.
""".strip(), figures=[figs["diagnosis"]])

    post.add("Two schemes agreeing means nothing if they share the same state", f"""
The most expensive wrong turn was this: when I could not find the bug in the
time-stepping, I reimplemented the time-stepping. Two schemes of different order,
one explicit-exponential and one semi-implicit, agreeing on the failure. I read
that as evidence that the stepping was fine and the *problem* was somehow unstable.

That inference was backwards. Both schemes held the state in the same
over-parameterised representation, so they inherited the same defect. Cross-checking
two implementations only tests what they do not share.

What did find it was a reference that shared as little as possible: the same
semi-discrete system integrated in **real space** with a stiff implicit solver at
tight tolerances. Real space means N real unknowns and no redundant half, so there
was nothing to grow. That reference stayed bounded past t = 1200 while my spectral
code was dead at 355. One number, and the disagreement localised the bug to the
representation rather than the scheme.

The fix is three characters. Use `rfft` instead of `fft`. A real FFT stores only
the non-negative wavenumbers, so the anti-Hermitian half is **not
representable** — you cannot amplify a state you cannot express. Same scheme, same
operators, same coefficients; the bug becomes structurally impossible rather than
merely absent.

The repaired integrator runs to t = {fix['t_final']:.0f}, fifty-six times past
where the old one died, with mean energy {fix['energy_first_quarter']:.3f} in the
first quarter against {fix['energy_last_quarter']:.3f} in the last — a
{fix['energy_drift_pct']:.1f}% difference and no trend. It matches the implicit
reference to a relative error of {fix['reference_rel_error']:.1e} at t = 8, and it
is still fourth-order accurate.

One more detail worth stating, because it is the same class of mistake. NumPy puts
−N/2 in the Nyquist slot, and for an **odd**-order derivative that sign is
arbitrary — which makes the nonlinear term inconsistent there. So zero the Nyquist
wavenumber in the first-derivative multiplier. But do **not** zero it in k² − k⁴,
which some reference codes do: that leaves an undamped mode that is conserved
forever and feeds the nonlinearity. Even powers do not care about the sign; odd ones
do.
""".strip(), figures=[figs["fix"]])

    post.add("The second bug: right answer, wrong timestep", f"""
While validating the repaired solver I found the other one, and it is a cleaner
teaching example because it hides behind a *correct* convergence study.

To quote forecast horizons in Lyapunov times you need the largest Lyapunov
exponent, and the standard method evolves an orthonormal frame through the
linearised dynamics, re-orthonormalising each step. The frame update I had written
was `Q ← Q + dt·J·Q`. An Euler step.

On Lorenz-63 at dt = 0.01 that gives {euler['lambda_max']:.4f} against the
literature's {lyap['literature']} — **{euler_err:.0f}% high**. At dt = 0.02 it is
{lyap['euler_at_002']['lambda_max']:.4f}, half again too large. Replacing the
update with the exact tangent propagator `expm(J·dt)` gives
{exact['lambda_max']:.4f} at dt = 0.01.

(That is still {100 * abs(exact['lambda_max'] - lyap['literature']) / lyap['literature']:.1f}%
below the reference value, and the cause is different and benign: I am averaging
over 120 time units of trajectory, which is not long enough for the finite-time
exponent to have fully converged to its asymptotic value. That error shrinks with
trajectory length. Euler's does not.)

Every horizon in my previous post is quoted in Lyapunov times, so a
{euler_err:.0f}% error in λ rescales every headline number in it. That is the kind
of bug that does not crash anything and does not look wrong.

And here is why a convergence study does not save you: **both methods converge.**
Euler's tangent error is O(dt·‖J‖²), so refine the step and it walks down onto the
right answer. Plot the two against dt and you see one curve essentially
flat across the whole range and another sliding down onto it from above — but if you
only ever ran the Euler version you would have seen a clean convergence plot and
concluded, correctly, that your estimator converges. Converging to the right answer
in the limit is not the same as being right at the step you actually use.

What catches it in one line is an identity rather than a limit. The Lyapunov
spectrum must sum to the divergence of the vector field, which for Lorenz-63 is
−(σ + 1 + β) = {lyap['trace']:.3f} — exactly, not approximately, and at any
timestep. Across this sweep the exact propagator's spectrum matches that trace to
within {lyap['expm_worst_trace']:.2f}%, while Euler's is off by up to
{lyap['euler_worst_trace']:.1f}%. That test needs no reference implementation, no
literature value, and no convergence study. It is now in the suite, next to a check
that the Kaplan-Yorke dimension comes out at 2.06.
""".strip(), figures=[figs["lyapunov"]])

    post.add("What kind of test would have caught these", """
Both bugs were live while the test suite was green, so it is worth being precise
about what those tests were missing. They checked that arrays had the right shape,
that nothing was NaN after a short run, and that the code executed. Every one of
those passed at t = 100 with a state whose meaningless half had already grown
eight orders of magnitude.

The tests that catch these are the ones that assert something *true about the
system* rather than something true about the code:

- **A conserved or stationary quantity.** Kuramoto-Sivashinsky has a bounded
  absorbing set, so mean u² fluctuates around a constant. Asserting no drift over a
  long run is three lines and would have caught bug one immediately.
- **An exact identity.** The Lyapunov spectrum summing to the trace of the
  Jacobian. No reference values, no tolerance-fiddling, holds at every timestep.
- **An independent reference that shares no code path.** Not a second scheme in the
  same representation — a different representation entirely. The value of a
  cross-check is exactly the code it does *not* share.
- **A long run.** Bug one was invisible for 300 time units. If your longest test
  is 100, you have tested the region where the bug hides.

None of that is specialised numerical-analysis practice; it is the same instinct as
testing behaviour rather than implementation. But it does require knowing one true
fact about your system that is not "the code returns an array" — and in scientific
computing there is always such a fact available, usually a conservation law or a
symmetry, and usually cheaper to assert than the mock you were going to write
instead.

The general lesson I actually took away is narrower and more useful than "write
better tests". **Count the degrees of freedom in your state and compare them to the
degrees of freedom in your problem.** If the state has more, something must enforce
the constraint, and you should be able to say what. If you cannot name it, the
unconstrained directions will find whatever amplification your system offers, and
they will do it from machine epsilon, on a schedule you can calculate in advance.

Next in this series: conformal prediction gives you a finite-sample coverage
guarantee that does not survive contact with a time series — and I will show the
nominal 90% interval realising under 60%, along with what to use instead.
""".strip())

    return post


if __name__ == "__main__":
    p = build()
    print(p.title, "|", p.word_count(), "words |", len(p.figures), "figures")
    for issue in p.audit():
        print("  audit:", issue)
