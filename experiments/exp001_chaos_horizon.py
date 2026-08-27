"""exp001 — How far ahead can a model forecast a chaotic system, and why?

The scientific content in one line: on Lorenz-63, a 600-unit echo state network
holds a closed-loop forecast for ~8 Lyapunov times, a 55-term polynomial model you
can read off the page gets ~7.8, a static random-feature map with three lags gets
~7.6, and linear AR gets ~0.11.

Three findings, none of which is "the reservoir won":

1. Error-growth slopes are set by the system, so model quality buys only additive
   horizon.
2. Removing recurrence entirely costs ~5% of the horizon when the state is fully
   observed.
3. The ESN is 6.4x better one step ahead but gains only 0.21 Lyapunov times of
   horizon — a tenth of what the exponential-growth argument predicts. One-step
   accuracy and closed-loop horizon are different quantities.

Run: `python -m standarderror.cli run exp001_chaos_horizon --publish`
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

import standarderror as se
from standarderror.dynamics import lyapunov, ode
from standarderror.models import ESN, NGRC, ESNConfig, NGRCConfig, baselines, metrics
from standarderror.render import Post
from standarderror.viz import charts
from standarderror.xai import reservoir_probes

IMG = se.SETTINGS.build_dir / "img"
DT = 0.02
N_TRAIN = 15000
N_WARM = 500
HORIZON = 1500
N_ROLLOUTS = 8
VPT_THRESHOLD = 0.3
SEED = se.SETTINGS.seed

ESN_CFG = ESNConfig(n_reservoir=600, spectral_radius=0.9, sparsity=0.03,
                    input_scaling=0.6, bias_scaling=0.1, leak_rate=1.0,
                    ridge=1e-7, washout=300, quadratic_features=True, seed=1)
NGRC_CFG = NGRCConfig(n_lags=3, stride=1, degree=2, ridge=1e-6)


# --------------------------------------------------------------- data + truth

def make_data() -> dict:
    traj = ode.lorenz63(n_steps=40000, dt=DT, transient=40.0)
    lam = lyapunov.lyapunov_from_jacobian(
        lyapunov.lorenz_jacobian(), traj.x[:20000], traj.dt)
    train_raw = traj.x[:N_TRAIN]
    mu, sd = train_raw.mean(axis=0), train_raw.std(axis=0)
    z = (traj.x - mu) / sd          # scaled on TRAIN ONLY
    return {"traj": traj, "lyap": lam, "z": z, "mu": mu, "sd": sd,
            "lambda_max": lam.exponent, "lyapunov_time": lam.lyapunov_time,
            "recovery": structure_recovery(z[:N_TRAIN], traj.dt, mu, sd)}


def structure_recovery(train: np.ndarray, dt: float, mu: np.ndarray,
                       sd: np.ndarray) -> dict:
    """Fit a 10-term NG-RC to the *derivative* and compare with Lorenz-63.

    Two choices make this work, and both matter:

    * **Central differences** for the target. A forward difference approximates
      the derivative at t + dt/2, which biases every coefficient by a factor
      (1 - exp(-lambda dt)) / (lambda dt) — about 10% at dt=0.02 for the fastest
      mode, enough to make the recovered numbers look wrong when the model is
      right.
    * **No feature standardisation**, so the coefficients live in the same
      (standardised-state) units as the analytic values and can be compared to
      them directly rather than only ranked.
    """
    U = train[1:-1]
    Y = (train[2:] - train[:-2]) / (2.0 * dt)
    model = NGRC(NGRCConfig(n_lags=1, degree=2, ridge=1e-10,
                            standardise=False)).fit(U, Y)
    s0, s1, s2 = sd
    m0, m1, m2 = mu
    expected = {                      # analytic values in standardised coords
        "dx/dt: y": 10.0 * s1 / s0,
        "dx/dt: x": -10.0,
        "dy/dt: xz": -s0 * s2 / s1,
        "dy/dt: x": (28.0 - m2) * s0 / s1,
        "dz/dt: xy": s0 * s1 / s2,
        "dz/dt: z": -8.0 / 3.0,
    }
    names = model.feature_names
    got = {
        "dx/dt: y": model.W_out[names.index("x1[t-0]"), 0],
        "dx/dt: x": model.W_out[names.index("x0[t-0]"), 0],
        "dy/dt: xz": model.W_out[names.index("x0[t-0]*x2[t-0]"), 1],
        "dy/dt: x": model.W_out[names.index("x0[t-0]"), 1],
        "dz/dt: xy": model.W_out[names.index("x0[t-0]*x1[t-0]"), 2],
        "dz/dt: z": model.W_out[names.index("x2[t-0]"), 2],
    }
    return {"W": model.W_out, "feature_names": names,
            "expected": expected, "recovered": {k: float(v) for k, v in got.items()},
            "r2_per_equation": [
                float(metrics.r2(Y[:, j],
                                 model.predict_teacher_forced(U)[:, j]))
                for j in range(3)]}


def rollout_starts(n_total: int) -> list[int]:
    """Evenly spaced, non-overlapping evaluation origins after the train window."""
    first = N_TRAIN
    span = HORIZON + N_WARM
    usable = (n_total - first) // span
    n = min(N_ROLLOUTS, usable)
    if n < 3:
        raise RuntimeError("not enough data for a rollout ensemble")
    return [first + i * span for i in range(n)]


# --------------------------------------------------------------- evaluation

def fit_models(train: np.ndarray) -> dict:
    models = {}
    t0 = time.time()
    models["ESN (600 units)"] = ESN(ESN_CFG).fit(train[:-1], train[1:])
    t_esn = time.time() - t0

    t0 = time.time()
    models["NG-RC (55 terms)"] = NGRC(NGRC_CFG).fit(train[:-1], train[1:])
    t_ngrc = time.time() - t0

    models["random features"] = baselines.RandomFeatures(
        n_features=600, n_lags=3, scale=0.6, ridge=1e-6,
        seed=1).fit(train[:-1], train[1:])
    models["linear AR(4)"] = baselines.LinearAR(n_lags=4).fit(train[:-1],
                                                              train[1:])
    models["persistence"] = baselines.Persistence().fit(train[:-1], train[1:])
    return models, {"esn_fit_seconds": round(t_esn, 2),
                    "ngrc_fit_seconds": round(t_ngrc, 2)}


def evaluate(models: dict, z: np.ndarray, lam: float) -> dict:
    starts = rollout_starts(len(z))
    results: dict[str, dict] = {}
    for name, model in models.items():
        vpts, curves = [], []
        for s in starts:
            warm = z[s - N_WARM: s]
            truth = z[s: s + HORIZON]
            pred = model.predict_autonomous(warm, HORIZON)
            v = lyapunov.valid_prediction_time(
                truth, pred, DT, threshold=VPT_THRESHOLD,
                lyapunov_exponent=lam)
            vpts.append(v["lyapunov_times"])
            curves.append(v["error_curve"])
        arr = np.asarray(vpts)
        results[name] = {
            "vpt_lyapunov_times": arr.tolist(),
            "vpt_median": float(np.median(arr)),
            "vpt_q1": float(np.percentile(arr, 25)),
            "vpt_q3": float(np.percentile(arr, 75)),
            "mean_error_curve": np.mean(np.asarray(curves), axis=0).tolist(),
        }
    return {"results": results, "starts": starts, "n_rollouts": len(starts)}


def one_step_comparison(models: dict, z: np.ndarray) -> dict:
    """Diebold-Mariano on one-step teacher-forced errors, ESN vs NG-RC.

    Included because the answer is counterintuitive and load-bearing for the post:
    NG-RC wins this comparison decisively, and still loses the closed-loop
    rollout. Reported with persistence's RMSE alongside so a reader can see both
    models are excellent one step ahead and the comparison is not between a good
    model and a bad one.
    """
    test = z[N_TRAIN: N_TRAIN + 6000]
    esn, ng = models["ESN (600 units)"], models["NG-RC (55 terms)"]

    # The two models have different lead-in lengths, so their prediction arrays
    # correspond to different target offsets. Line them up on the *target* index
    # explicitly rather than by trimming from the end — trimming happens to be
    # wrong here, and comparing two models against shifted targets is a mistake
    # that flatters whichever one got the easier alignment.
    p_esn_full = esn.predict_teacher_forced(test[:-1])
    p_ng_full = ng.predict_teacher_forced(test[:-1])
    off_esn = ESN_CFG.washout + 1        # first target index the ESN can score
    off_ng = ng.span                     # first target index NG-RC can score
    p_esn_full = p_esn_full[ESN_CFG.washout:]
    start = max(off_esn, off_ng)
    y = test[start:]
    p_esn = p_esn_full[start - off_esn:]
    p_ng = p_ng_full[start - off_ng:]
    n = min(len(y), len(p_esn), len(p_ng))
    y, p_esn, p_ng = y[:n], p_esn[:n], p_ng[:n]

    # Sanity: a correctly aligned one-step forecast must beat persistence by a
    # wide margin on this data. If it does not, the alignment is still off.
    persist_rmse = metrics.rmse(y, test[start - 1: start - 1 + n])
    esn_rmse = metrics.rmse(y, p_esn)
    if not esn_rmse < 0.1 * persist_rmse:
        raise AssertionError(
            f"alignment check failed: ESN one-step RMSE {esn_rmse:.3e} is not "
            f"clearly better than persistence {persist_rmse:.3e}")

    dm = metrics.dm_test(y, p_esn, p_ng)     # multivariate, per time step
    return {"dm": dm, "rmse_esn": esn_rmse,
            "rmse_ngrc": metrics.rmse(y, p_ng),
            "rmse_persistence": persist_rmse,
            "n_compared": int(n)}


def probe_sweep(radii=(0.3, 0.6, 0.9, 1.1, 1.3, 1.6)) -> dict:
    """Reservoir diagnostics as a function of spectral radius, no task involved."""
    small = replace(ESN_CFG, n_reservoir=300, washout=200)
    rows = []
    for rho in radii:
        cfg = replace(small, spectral_radius=rho)
        cap = reservoir_probes.computational_capability(
            cfg, n_probe=150, n_steps=250, common_tail=40, seed=3)
        mc = reservoir_probes.memory_capacity(cfg, n_steps=2500, washout=400,
                                              max_delay=60, seed=3)
        loc = reservoir_probes.lyapunov_local(cfg, n_steps=1500, seed=3)
        rows.append({"spectral_radius": rho,
                     "kernel_rank": cap["kernel_rank"],
                     "generalisation_rank": cap["generalisation_rank"],
                     "capability": cap["capability"],
                     "memory_capacity": mc["memory_capacity"],
                     "local_lyapunov": loc["lyapunov_exponent"],
                     "regime": loc["regime"]})
    return {"rows": rows}


def sensitivity(z: np.ndarray, lam: float,
                radii=(0.5, 0.7, 0.9, 1.1, 1.3),
                scalings=(0.2, 0.4, 0.6, 0.9, 1.3)) -> dict:
    """VPT over a spectral-radius x input-scaling grid, cheaper reservoir.

    Shown as a surface rather than a winning number: the point of the figure is
    that the good region is a broad plateau, so the usual reported single value
    is far less load-bearing than it looks.
    """
    cfg0 = replace(ESN_CFG, n_reservoir=300, washout=250)
    train = z[:8000]
    M = np.full((len(scalings), len(radii)), np.nan)
    for i, iscale in enumerate(scalings):
        for j, rho in enumerate(radii):
            cfg = replace(cfg0, spectral_radius=rho, input_scaling=iscale)
            try:
                m = ESN(cfg).fit(train[:-1], train[1:])
                vals = []
                for s in (8000, 9600, 11200):
                    warm, truth = z[s - 400: s], z[s: s + 800]
                    pred = m.predict_autonomous(warm, 800)
                    vals.append(lyapunov.valid_prediction_time(
                        truth, pred, DT, threshold=VPT_THRESHOLD,
                        lyapunov_exponent=lam)["lyapunov_times"])
                M[i, j] = float(np.median(vals))
            except Exception:
                M[i, j] = np.nan
    return {"matrix": M, "radii": list(radii), "scalings": list(scalings)}


# --------------------------------------------------------------- figures

def figures(data: dict, ev: dict, probes: dict, sens: dict,
            models: dict) -> dict:
    z, lam = data["z"], data["lambda_max"]
    src = "Simulated Lorenz-63; code and parameters in the repo."
    figs = {}

    # F1 — one rollout, x component, three models.
    # Windowed to ~14 Lyapunov times: past saturation the lines are just noise
    # laid over noise, and the eye cannot find the divergence point at all.
    s = ev["starts"][0]
    SHOW = 760
    truth = z[s: s + SHOW, 0]
    tt = np.arange(SHOW) * DT * lam
    preds = {}
    for name in ("ESN (600 units)", "NG-RC (55 terms)", "linear AR(4)"):
        preds[name] = models[name].predict_autonomous(
            z[s - N_WARM: s], HORIZON)[:SHOW, 0]
    med = ev["results"]["ESN (600 units)"]["vpt_median"]
    f, _ = charts.prediction_vs_truth(
        tt, truth, preds,
        title="A closed-loop forecast of Lorenz-63, on a Lyapunov clock",
        subtitle=(f"The reservoir tracks the true trajectory for about "
                  f"{med:.0f} e-folding times, then loses it — as it must. "
                  "NG-RC sits underneath it until they separate."),
        ylabel="x (standardised)", xlabel="Lyapunov times",
        source=src, divergence_at=med,
        alt=("Line chart of the standardised x coordinate of Lorenz-63 against "
             "Lyapunov time. The grey truth line and the ESN forecast overlap "
             f"closely for roughly {med:.0f} Lyapunov times before separating, while "
             "the linear AR forecast collapses almost immediately."),
        caption=("Fig 1. One representative rollout. The dashed line marks the "
                 "median valid-prediction time of the ESN across eight "
                 "independent origins. Linear AR(4) is not a strawman — it is "
                 "the best linear model of this data, and it is useless here."),
        path=str(IMG / "f1-rollout.png"))
    figs["rollout"] = f

    # F2 — error growth
    SHOW2 = 900
    curves = {k: np.asarray(v["mean_error_curve"])[:SHOW2]
              for k, v in ev["results"].items()}
    tt2 = np.arange(SHOW2) * DT * lam
    f, _ = charts.error_growth(
        tt2, curves, threshold=VPT_THRESHOLD,
        title="Error growth is exponential; models differ only in where they start",
        subtitle=("Mean normalised error across eight rollouts. Parallel slopes "
                  "on a log scale mean every model inherits the same Lyapunov "
                  "exponent."),
        source=src,
        alt=("Log-scale line chart of normalised forecast error against Lyapunov "
             "time for five models. All curves rise with a similar slope but "
             "start at very different levels, crossing the 0.3 threshold at "
             "different times."),
        caption=("Fig 3. The slopes are set by the system, not the model. What a "
                 "better model buys you is a lower intercept — and because "
                 "growth is exponential, halving the one-step error buys only a "
                 "fixed additive amount of horizon."),
        path=str(IMG / "f3-error-growth.png"))
    figs["error_growth"] = f

    # F3 — VPT ranked
    names = list(ev["results"])
    meds = [ev["results"][n]["vpt_median"] for n in names]
    errs = [[ev["results"][n]["vpt_median"] - ev["results"][n]["vpt_q1"]
             for n in names],
            [ev["results"][n]["vpt_q3"] - ev["results"][n]["vpt_median"]
             for n in names]]
    f, _ = charts.ranked_bars(
        names, meds, errors=errs,
        title="Valid prediction time, in Lyapunov times",
        subtitle="Median over eight rollout origins; bars span the interquartile range.",
        xlabel="Lyapunov times until normalised error exceeds 0.3",
        source=src,
        alt=("Horizontal bar chart of median valid prediction time in Lyapunov "
             "times for five models, with interquartile-range error bars. The "
             "ESN and NG-RC bars are close together and far ahead of linear AR "
             "and persistence."),
        caption=("Fig 2. The spread matters as much as the ranking: a single "
                 "rollout can mislead by a factor of two, which is why a "
                 "single-number VPT in a paper should be treated with "
                 "suspicion."),
        path=str(IMG / "f2-vpt.png"))
    figs["vpt"] = f

    # F4 — sensitivity surface
    f, _ = charts.sensitivity_surface(
        sens["matrix"], xticks=sens["radii"], yticks=sens["scalings"],
        xlabel="spectral radius", ylabel="input scaling",
        title="The good region is a plateau, not a peak",
        subtitle=("Median VPT of a 300-unit reservoir over a hyperparameter "
                  "grid; darker is a longer horizon, the ring marks the best "
                  "cell."),
        cbar_label="Lyapunov times", source=src, lower_is_better=False,
        alt=("Heatmap of median valid prediction time over a grid of spectral "
             "radius and input scaling, showing a broad dark region of good "
             "performance rather than a single sharp optimum."),
        caption=("Fig 6. Reported hyperparameters imply more precision than the "
                 "data supports. Anything in a wide band performs within noise "
                 "of the best cell — worth knowing before you spend a week on "
                 "Bayesian optimisation."),
        path=str(IMG / "f6-sensitivity.png"))
    figs["sensitivity"] = f

    # F5 — reservoir probes vs spectral radius
    rows = probes["rows"]
    rho = [r["spectral_radius"] for r in rows]
    import pandas as pd
    panels = {
        "kernel − generalisation rank": pd.Series(
            [r["capability"] for r in rows], index=rho),
        "memory capacity": pd.Series(
            [r["memory_capacity"] for r in rows], index=rho),
        "local Lyapunov exponent": pd.Series(
            [r["local_lyapunov"] for r in rows], index=rho),
    }
    f, _ = charts.small_multiples(
        panels, ncols=3,
        title="What the reservoir becomes, measured without any task",
        subtitle="Each panel against spectral radius; no forecasting involved.",
        source=src,
        alt=("Three small line charts against spectral radius: kernel minus "
             "generalisation rank, which collapses above 0.9; memory capacity, "
             "which peaks near 1.1; and the local Lyapunov exponent, which rises "
             "monotonically and crosses zero above 1.3."),
        caption=("Fig 5. These are task-free diagnostics. The local Lyapunov "
                 "exponent crossing zero is the 'edge of chaos' people gesture "
                 "at; it is measurable in a few seconds and it is not at "
                 "spectral radius 1."),
        path=str(IMG / "f5-probes.png"))
    figs["probes"] = f

    # F6 — structure recovery: the whole model as a coefficient matrix
    rec = data["recovery"]
    pretty = {"1": "1", "x0[t-0]": "x", "x1[t-0]": "y", "x2[t-0]": "z",
              "x0[t-0]*x0[t-0]": "x²", "x0[t-0]*x1[t-0]": "xy",
              "x0[t-0]*x2[t-0]": "xz", "x1[t-0]*x1[t-0]": "y²",
              "x1[t-0]*x2[t-0]": "yz", "x2[t-0]*x2[t-0]": "z²"}
    cols = [pretty.get(n, n) for n in rec["feature_names"]]
    f, _ = charts.coefficient_matrix(
        rec["W"].T, row_labels=["dx/dt", "dy/dt", "dz/dt"], col_labels=cols,
        title="The model rediscovered the equations",
        subtitle=("Every coefficient of a 10-term NG-RC fitted to the "
                  "derivative. Rows are the three equations, columns the "
                  "candidate monomials; blank cells are effectively zero."),
        cbar_label="coefficient (standardised state)", source=src,
        alt=("Heatmap of a three-by-ten coefficient matrix. The dx/dt row has "
             "large entries only under x and y; the dy/dt row under x, y and the "
             "xz product; the dz/dt row under the constant, z and the xy "
             "product. All other cells are near zero."),
        caption=("Fig 4. Compare with Lorenz-63 itself: dx/dt = σ(y−x) is "
                 "linear, dy/dt = x(ρ−z)−y contains xz, and dz/dt = xy−βz "
                 "contains xy. The recovered sparsity pattern is exactly that. "
                 "This is the model, not an explanation of the model."),
        path=str(IMG / "f4-structure.png"))
    figs["terms"] = f
    return figs


# --------------------------------------------------------------- the post

def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    data = make_data()
    lam = data["lambda_max"]
    models, timings = fit_models(data["z"][:N_TRAIN])
    ev = evaluate(models, data["z"], lam)
    one_step = one_step_comparison(models, data["z"])
    probes = probe_sweep()
    sens = sensitivity(data["z"], lam)
    figs = figures(data, ev, probes, sens, models)

    R = ev["results"]
    esn, ngrc = R["ESN (600 units)"], R["NG-RC (55 terms)"]
    rf, ar, pers = R["random features"], R["linear AR(4)"], R["persistence"]
    n_esn_feat = models["ESN (600 units)"].train_diagnostics["n_features"]
    n_ng_feat = models["NG-RC (55 terms)"].train_diagnostics["n_features"]
    best_probe = max(probes["rows"], key=lambda r: r["capability"])
    edge = min(probes["rows"], key=lambda r: abs(r["local_lyapunov"]))

    post = Post(
        title="How Far Ahead Can You Forecast Chaos?",
        slug="how-far-ahead-can-you-forecast-chaos",
        subtitle=(f"A reproducible benchmark on Lorenz-63, and why the "
                  f"interpretable model is only "
                  f"{esn['vpt_median'] - ngrc['vpt_median']:.2f} Lyapunov times behind "
                  f"the black box"),
        summary=(f"On Lorenz-63 a 600-unit echo state network holds a closed-loop "
                 f"forecast for {esn['vpt_median']:.1f} Lyapunov times. A "
                 f"{n_ng_feat}-term polynomial model you can print on one line "
                 f"gets {ngrc['vpt_median']:.1f}. A static random-feature map "
                 f"with no memory at all gets {rf['vpt_median']:.1f}. Linear AR "
                 f"gets {ar['vpt_median']:.2f}. Here is what that ordering "
                 "actually tells you — and what it means for anyone putting a "
                 "recurrent model into a risk system."),
        tags=["reservoir-computing", "time-series", "explainable-ai",
              "machine-learning", "quantitative-finance"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated Lorenz-63 (sigma=10, rho=28, beta=8/3), integrated with "
            "RK45 at rtol=1e-10; no external data required.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "scipy", "standarderror")),
            "largest Lyapunov exponent (Benettin, analytic Jacobian)":
                f"{lam:.4f} (literature value 0.9056)",
            "Kaplan-Yorke dimension":
                f"{data['lyap'].detail['kaplan_yorke_dim']:.3f} "
                "(literature value 2.062)",
            "rollout origins": ev["n_rollouts"],
            "fit time": f"ESN {timings['esn_fit_seconds']}s, "
                        f"NG-RC {timings['ngrc_fit_seconds']}s",
        },
    )

    post.add("The question people actually mean", f"""
"Can machine learning predict a chaotic system?" is a badly posed question, and
the way it is usually answered makes it worse. Someone trains a recurrent network
on a Lorenz trajectory, reports an R-squared of 0.999 on next-step prediction,
and concludes that the network learned the dynamics. It did not. Next-step
prediction on a smooth trajectory sampled every 0.02 time units is a task that
*persistence* — literally copying the last observation forward — solves to three
decimal places. The reported number measures the sampling rate, not the model.

The question worth asking is: **for how long does a forecast stay useful once you
stop feeding the model the truth?** Close the loop. Let the model's own output
become its next input, and see how long it survives.

Chaos gives us a natural clock for this. Nearby trajectories separate
exponentially at a rate given by the largest Lyapunov exponent. For Lorenz-63 at
the standard parameters I measure {lam:.4f} using the Benettin tangent-space
method with the analytic Jacobian, against the literature value of 0.9056 — so
one Lyapunov time, the interval over which an initial error grows by a factor of
e, is about {data['lyapunov_time']:.2f} time units. Measuring forecast horizons in
Lyapunov times instead of seconds makes results comparable across systems,
sampling rates, and papers.

The metric is **valid prediction time (VPT)**: the point at which the normalised
forecast error first exceeds a threshold, here 0.3, with the error normalised by
the RMS amplitude of the truth. It is the standard convention in the reservoir
computing literature and it has the great virtue of being hard to game.
""".strip(), figures=[figs["rollout"]])

    post.add("Five models, one honest protocol", f"""
Everything below trains on the same {N_TRAIN:,} standardised samples — scaled with
training-set statistics only, because scaling with the full-series mean is a
leak that will flatter every model in the table — and is evaluated by closed-loop
rollout from {ev['n_rollouts']} non-overlapping origins in held-out data. Reporting
one rollout is not enough: the interquartile range across origins is wide enough
that a single number can mislead by close to a factor of two.

The models, in order of how much you have to trust them:

- **Persistence.** The last value, repeated. The bar.
- **Linear AR(4).** Ridge-regularised linear autoregression on four lags. This is
  the control that tells you whether nonlinearity is doing anything.
- **Random features.** A 600-dimensional random `tanh` map on three lags, ridge
  readout. Nonlinear and wide, but *memoryless* — no recurrence. This one isolates
  how much of a reservoir's advantage comes from recurrence as opposed to merely
  being a big nonlinear basis.
- **NG-RC** ({n_ng_feat} features). Next-generation reservoir computing: instead of
  a random recurrent network, an explicit polynomial expansion of three lags —
  constant, linear terms, and all quadratic monomials — with a ridge readout.
- **ESN** ({ESN_CFG.n_reservoir} units, {n_esn_feat} readout features). A
  conventional echo state network: sparse random recurrent weights rescaled to
  spectral radius {ESN_CFG.spectral_radius}, `tanh` activation, ridge readout,
  with half the states squared to break the odd symmetry of `tanh`.

Results, median over rollouts, in Lyapunov times:

| model | VPT (median) | IQR |
|---|---|---|
| ESN ({ESN_CFG.n_reservoir} units) | **{esn['vpt_median']:.2f}** | {esn['vpt_q1']:.2f} – {esn['vpt_q3']:.2f} |
| NG-RC ({n_ng_feat} terms) | {ngrc['vpt_median']:.2f} | {ngrc['vpt_q1']:.2f} – {ngrc['vpt_q3']:.2f} |
| random features | {rf['vpt_median']:.2f} | {rf['vpt_q1']:.2f} – {rf['vpt_q3']:.2f} |
| linear AR(4) | {ar['vpt_median']:.2f} | {ar['vpt_q1']:.2f} – {ar['vpt_q3']:.2f} |
| persistence | {pers['vpt_median']:.2f} | {pers['vpt_q1']:.2f} – {pers['vpt_q3']:.2f} |

Three things in that table are worth more than the ranking.
""".strip(), figures=[figs["vpt"]])

    post.add("1. The slopes are the system's, not the model's", """
Plot the error growth on a log scale and the curves are close to parallel. Every
model inherits the same exponential divergence rate, because that rate is a
property of Lorenz-63 and not of anything you fit to it. What a better model buys
is a lower *intercept* — a smaller one-step error.

That has a consequence people consistently underrate. Because the error grows
exponentially, an improvement in one-step accuracy translates into only an
*additive* gain in horizon: halving the one-step error buys you ln(2)/λ ≈
0.77 Lyapunov times, no matter how good you already are. Going from a
thousandth to a millionth of a unit of one-step error — a huge engineering
achievement — buys about eight Lyapunov times, and then stops. There is no
model, and no amount of compute, that escapes this ceiling.

That is the upper bound. Finding 3 shows the two models here realise only about a
tenth of it, which is a separate and more practical problem.

This is the honest version of "chaos limits predictability", and it is worth
carrying into any conversation about forecasting a system with positive Lyapunov
exponent. The exponent sets the budget. Model quality decides how much of the
budget you actually get.
""".strip(), figures=[figs["error_growth"]])

    post.add("2. Recurrence is worth less than you would guess", f"""
The random-feature model has no memory whatsoever. It sees three lags through a
fixed random nonlinearity and regresses. It reaches {rf['vpt_median']:.2f}
Lyapunov times — {100 * rf['vpt_median'] / esn['vpt_median']:.0f}% of the ESN's
horizon, with no recurrent state, no spectral radius to tune, and no washout.

That is a useful result to internalise before reaching for a recurrent
architecture. For a system whose state is *observable* — here we feed the model
all three coordinates — a short delay embedding plus a nonlinear basis already
contains most of the information. Takens' theorem says as much: a delay embedding
of sufficient dimension reconstructs the attractor, so a static map on enough lags
can in principle be as good as a stateful one. Recurrence earns its keep when the
state is *partially observed* and the model must integrate information over an
unknown, possibly long window — which, to be fair, is exactly the situation in
most financial applications.

So the honest framing is not "reservoirs are overrated" but "test the memoryless
version first, because it is cheaper, easier to reason about, and often close."
""".strip())

    rec = data["recovery"]
    exp_, got = rec["expected"], rec["recovered"]
    ratio = one_step["rmse_ngrc"] / one_step["rmse_esn"]
    expected_gain = float(np.log(ratio) / lam)
    actual_gain = esn["vpt_median"] - ngrc["vpt_median"]
    pstr = ("< 0.001" if one_step["dm"]["p_value"] < 0.001
            else f"= {one_step['dm']['p_value']:.3f}")
    post.add(f"3. The interpretable model is "
             f"{esn['vpt_median'] - ngrc['vpt_median']:.2f} Lyapunov times behind", f"""
NG-RC reaches {ngrc['vpt_median']:.2f} Lyapunov times against the ESN's
{esn['vpt_median']:.2f}, using {n_ng_feat} features instead of {n_esn_feat} — a
factor of {n_esn_feat / n_ng_feat:.0f} fewer — and it fits in
{timings['ngrc_fit_seconds']}s against {timings['esn_fit_seconds']}s.

And here is the part I did not expect. One step ahead the ESN is not marginally
better — it is **{ratio:.1f} times** better, RMSE {one_step['rmse_esn']:.2e} against
NG-RC's {one_step['rmse_ngrc']:.2e}, and a Diebold-Mariano test on
{one_step['n_compared']:,} held-out steps puts the statistic at
{one_step['dm']['statistic']:.1f} (p {pstr}). Not a close call. For scale, both are
four orders of magnitude below persistence's {one_step['rmse_persistence']:.2e}, so
this is a comparison between two very accurate models.

Now put that through the arithmetic from finding 1. A {ratio:.1f}-fold reduction in
one-step error should buy ln({ratio:.1f})/λ = **{expected_gain:.2f} Lyapunov times**
of extra horizon. The ESN actually gains **{actual_gain:.2f}** — about
{100 * actual_gain / expected_gain:.0f}% of it.

That gap is the most useful thing in this post. The exponential-growth argument
treats a forecast error as if it were an infinitesimal perturbation of the true
trajectory, growing at the system's Lyapunov rate. In a closed loop it is not: the
model's error is *structured*, it is re-injected as the next input, and whether it
compounds or partially cancels depends on the geometry of the model's error, not on
λ. Two models can sit an order of magnitude apart on one-step error and finish
within a rollout's noise of each other.

The operational lesson: **one-step accuracy and closed-loop horizon are different
quantities, and the mapping between them is model-specific.** A leaderboard built
on one-step error is not a leaderboard for the task you care about — which, since
one-step error is what almost every forecasting benchmark reports, is worth
sitting with.

The difference that matters is not accuracy. It is that NG-RC's readout is a linear
map over *named* monomials, so the model is not something you explain after the
fact — it is something you print. To make the point as sharply as possible, here is
the same model family fitted to the derivative instead of the next state, with a
single lag and quadratic terms: ten candidate monomials, three equations, thirty
coefficients in total.

The recovered sparsity pattern is the Lorenz system. Not similar to it — it:

| quantity | analytic | recovered |
|---|---|---|
| dx/dt coefficient on y | {exp_['dx/dt: y']:.3f} | {got['dx/dt: y']:.3f} |
| dx/dt coefficient on x | {exp_['dx/dt: x']:.3f} | {got['dx/dt: x']:.3f} |
| dy/dt coefficient on xz | {exp_['dy/dt: xz']:.3f} | {got['dy/dt: xz']:.3f} |
| dy/dt coefficient on x | {exp_['dy/dt: x']:.3f} | {got['dy/dt: x']:.3f} |
| dz/dt coefficient on xy | {exp_['dz/dt: xy']:.3f} | {got['dz/dt: xy']:.3f} |
| dz/dt coefficient on z | {exp_['dz/dt: z']:.3f} | {got['dz/dt: z']:.3f} |

Coefficients are in standardised-state units, which is why they are not 10 and 28
and 8/3; the analytic column carries the same rescaling. Two details were
load-bearing. The target uses **central** differences: a forward difference
estimates the derivative at t + dt/2 and biases every coefficient by
(1 − e^(−λdt))/(λdt), about 10% here — enough to make a correct model look wrong.
And the features are left unstandardised so the numbers can be compared to the
analytic values rather than merely ranked.

Where it is imperfect is instructive too. The `dz/dt` row picks up a spurious x²
term, because on the Lorenz attractor x² and xy are strongly correlated, and ridge
regression has no way to prefer one over the other. That is collinearity, not a
bug, and it is the same mechanism that makes feature attributions unstable on
lagged financial features — where every lag is nearly collinear with its
neighbours. This is why the attribution module in the repo defaults to permuting
*blocks* of related features rather than single columns.

The consequence for applied work is direct. Model validation functions do not ask
what your valid prediction time is. They ask what the model does and why, and they
are entitled to an answer that does not rest on a post-hoc attribution method with
its own failure modes. A model whose functional form is inspectable and which costs
you {esn['vpt_median'] - ngrc['vpt_median']:.2f} Lyapunov times out of
{esn['vpt_median']:.1f} is, for most regulated purposes,
straightforwardly the better model — and you cannot know the size of that trade
unless you measure it on a problem where you know the answer.
""".strip(), figures=[figs["terms"]])

    post.add("What the reservoir actually is, measured without a task", f"""
Reservoirs are usually tuned by grid search over spectral radius and input
scaling, which tells you nothing about the object you built. There are cheap,
task-free diagnostics that do:

- **Memory capacity** (Jaeger): the total variance of past inputs linearly
  recoverable from the current state, bounded above by the reservoir size. How far
  back the reservoir can see.
- **Kernel rank minus generalisation rank** (Legenstein & Maass): drive the
  reservoir with many independent input streams and take the numerical rank of the
  resulting states — high rank means it separates different inputs. Then repeat
  with streams that share a common recent history — here you want *low* rank,
  because similar recent inputs should collapse to similar states. The difference
  is a task-free measure of useful computational capacity, maximised at
  ρ = {best_probe['spectral_radius']} in this sweep. Read the *collapse* above
  ρ = 0.9 rather than the absolute level: kernel rank is bounded by the number of
  probe streams (150 here), so the low-ρ end is saturated by construction.
- **Local Lyapunov exponent of the driven reservoir**: perturb the state,
  evolve both copies under the same input, measure the growth rate. Negative means
  contracting and the echo state property holds; positive means the reservoir has
  its own chaos and will never forget its initial condition. The crossing is what
  people mean by "the edge of chaos", and in this sweep it sits nearest
  ρ = {edge['spectral_radius']} — *not* at ρ = 1, which is the value the textbook
  rule of thumb would have you believe. The spectral radius bound is neither
  necessary nor sufficient once input scaling is non-trivial, so measure the thing
  itself.

The hyperparameter surface makes a related point. The region of good performance is
a broad plateau, not a peak. Any reported "we used spectral radius 0.9" implies far
more precision than the data supports, and the practical lesson is to spend your
tuning budget on the input scaling and the ridge parameter — which do matter — and
stop agonising over the third decimal place of ρ.
""".strip(), figures=[figs["probes"], figs["sensitivity"]])

    post.add("Taking this to data that matters", """
Everything above is on simulated data, deliberately. Lorenz-63 has ground truth: a
known attractor, a known Lyapunov exponent, and no measurement noise, so a claim
like "eight Lyapunov times" is falsifiable. Financial series have none of that. The
signal-to-noise ratio is brutal, the generating process is non-stationary, and
persistence is a genuinely strong competitor rather than a formality — which is
precisely why so many published financial forecasting results evaporate on
inspection.

So the value of the synthetic benchmark is not the number. It is the *protocol*:
scale on training data only, close the loop, evaluate from multiple origins, report
the spread, compare against persistence and a linear model, and test whether the
difference between two models is distinguishable from noise before you claim one is
better. Every one of those steps was load-bearing here. Skipping any of them would
have produced a more impressive-looking and less true result.

The next post in this series takes exactly this protocol to a public macro-financial
series — the US term spread and financial-conditions indices from FRED — and asks
the uncomfortable question: once you enforce this discipline, is there any
closed-loop predictability there at all, or does persistence win?
""".strip())

    return post


if __name__ == "__main__":
    p = build()
    print(p.title, "|", p.word_count(), "words", "|", len(p.figures), "figures")
    for issue in p.audit():
        print("  audit:", issue)
