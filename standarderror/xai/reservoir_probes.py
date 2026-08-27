"""Diagnostics that tell you what a reservoir has actually become.

A reservoir is usually treated as an opaque blob you tune by grid search. It is
not opaque — there are cheap, well-defined quantities that characterise it, and
they make far better blog material than another accuracy table:

* `memory_capacity` — Jaeger's MC: total variance of past inputs linearly
  recoverable from the current state. Bounded above by N. Tells you how far back
  the reservoir can see.
* `kernel_rank` / `generalisation_rank` — Legenstein & Maass's separation and
  generalisation ranks. High kernel rank means the reservoir separates different
  inputs; high generalisation rank means it *fails* to collapse similar ones.
  Good reservoirs maximise kernel rank minus generalisation rank, and this
  difference locates the edge of chaos far more usefully than staring at the
  spectral radius.
* `lyapunov_local` — the reservoir's own largest local Lyapunov exponent,
  estimated by perturbing the state. Near zero at the critical point. This is the
  quantity people *mean* when they say "edge of chaos".
* `echo_state_property_check` — empirical state-forgetting test. The spectral
  radius < 1 rule is neither necessary nor sufficient once the input is scaled
  up; this measures the thing itself.
"""

from __future__ import annotations

import numpy as np

from ..models.esn import ESN, ESNConfig


def _run_states(model: ESN, U: np.ndarray, *,
                h0: np.ndarray | None = None) -> np.ndarray:
    """Raw reservoir states (no readout features) for inputs U."""
    U = np.atleast_2d(np.asarray(U, float))
    if U.ndim == 1:
        U = U[:, None]
    if model.W is None:
        model._build(U.shape[1])
    h = np.zeros(model.config.n_reservoir) if h0 is None else h0.copy()
    out = np.empty((len(U), model.config.n_reservoir))
    for t, u in enumerate(U):
        h = model._step(h, u)
        out[t] = h
    return out


def memory_capacity(
    config: ESNConfig | None = None,
    *,
    n_steps: int = 4000,
    washout: int = 500,
    max_delay: int | None = None,
    seed: int = 0,
) -> dict:
    """Jaeger memory capacity under i.i.d. uniform input.

    MC_k is the squared correlation between the optimal linear readout of the
    current state and the input k steps ago; MC = sum_k MC_k, bounded by N.
    Reported with the per-delay curve, because the *shape* is the story: a high
    spectral radius trades short-delay fidelity for a longer tail.
    """
    cfg = config or ESNConfig()
    max_delay = max_delay or min(cfg.n_reservoir, 200)
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.8, 0.8, n_steps + max_delay)

    model = ESN(cfg)
    states = _run_states(model, u[:, None])
    X = np.concatenate([np.ones((len(states), 1)), states], axis=1)[washout:]

    mcs = []
    for k in range(1, max_delay + 1):
        target = u[washout - k: washout - k + len(X)]
        if len(target) != len(X) or washout - k < 0:
            break
        beta = np.linalg.lstsq(X, target, rcond=None)[0]
        pred = X @ beta
        var = np.var(target)
        mcs.append(0.0 if var <= 0 else
                   float(np.cov(pred, target)[0, 1] ** 2 / (np.var(pred) * var + 1e-30)))
    mcs = np.asarray(mcs)
    return {"memory_capacity": float(np.sum(mcs)),
            "per_delay": mcs.tolist(),
            "half_life_delay": int(np.argmax(mcs < 0.5) + 1) if np.any(mcs < 0.5) else len(mcs),
            "upper_bound": cfg.n_reservoir,
            "config": cfg.as_dict()}


def kernel_rank(config: ESNConfig | None = None, *, n_probe: int = 200,
                n_steps: int = 300, seed: int = 0,
                tol: float | None = None) -> dict:
    """Legenstein-Maass kernel (separation) rank.

    Drive the reservoir with `n_probe` *independent random* input streams; collect
    the final state of each; the numerical rank of that matrix is the kernel rank.
    High is good: the reservoir maps different inputs to linearly separable
    states.
    """
    cfg = config or ESNConfig()
    rng = np.random.default_rng(seed)
    finals = np.empty((n_probe, cfg.n_reservoir))
    model = ESN(cfg)
    for i in range(n_probe):
        u = rng.uniform(-0.8, 0.8, (n_steps, 1))
        finals[i] = _run_states(model, u)[-1]
    s = np.linalg.svd(finals, compute_uv=False)
    tol = tol if tol is not None else s.max() * max(finals.shape) * np.finfo(float).eps
    return {"kernel_rank": int(np.sum(s > tol)), "n_probe": n_probe,
            "singular_values": s[:20].tolist(), "config": cfg.as_dict()}


def generalisation_rank(config: ESNConfig | None = None, *, n_probe: int = 200,
                        n_steps: int = 300, common_tail: int = 50,
                        seed: int = 0, tol: float | None = None) -> dict:
    """Generalisation rank: same probe, but every stream shares its final
    `common_tail` inputs. A reservoir that generalises should collapse these to
    nearly the same state, so **low is good** here.

    `kernel_rank - generalisation_rank` is the quantity to maximise; it peaks near
    the edge of chaos and gives you a principled way to pick the spectral radius
    that does not require a downstream task at all.
    """
    cfg = config or ESNConfig()
    rng = np.random.default_rng(seed)
    tail = rng.uniform(-0.8, 0.8, (common_tail, 1))
    finals = np.empty((n_probe, cfg.n_reservoir))
    model = ESN(cfg)
    for i in range(n_probe):
        head = rng.uniform(-0.8, 0.8, (n_steps - common_tail, 1))
        finals[i] = _run_states(model, np.vstack([head, tail]))[-1]
    s = np.linalg.svd(finals, compute_uv=False)
    tol = tol if tol is not None else s.max() * max(finals.shape) * np.finfo(float).eps
    return {"generalisation_rank": int(np.sum(s > tol)), "n_probe": n_probe,
            "common_tail": common_tail, "singular_values": s[:20].tolist(),
            "config": cfg.as_dict()}


def computational_capability(config: ESNConfig | None = None, **kw) -> dict:
    """kernel_rank - generalisation_rank, the Legenstein-Maass criterion."""
    kr = kernel_rank(config, **{k: v for k, v in kw.items()
                                if k in ("n_probe", "n_steps", "seed")})
    gr = generalisation_rank(config, **kw)
    return {"kernel_rank": kr["kernel_rank"],
            "generalisation_rank": gr["generalisation_rank"],
            "capability": kr["kernel_rank"] - gr["generalisation_rank"],
            "config": (config or ESNConfig()).as_dict()}


def lyapunov_local(config: ESNConfig | None = None, *, n_steps: int = 2000,
                   washout: int = 200, eps: float = 1e-8, seed: int = 0) -> dict:
    """Largest local Lyapunov exponent of the driven reservoir map.

    Perturb the state by `eps`, evolve both copies under the same input stream,
    renormalise each step, average the log growth. Negative = contracting (echo
    state property holds); ~0 = edge of chaos; positive = the reservoir has its
    own chaos and will not forget its initial condition.
    """
    cfg = config or ESNConfig()
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.8, 0.8, (n_steps + washout, 1))
    model = ESN(cfg)
    if model.W is None:
        model._build(1)

    h = np.zeros(cfg.n_reservoir)
    for t in range(washout):
        h = model._step(h, u[t])
    d0 = rng.standard_normal(cfg.n_reservoir)
    d0 *= eps / np.linalg.norm(d0)
    h2 = h + d0

    logs = []
    for t in range(washout, washout + n_steps):
        h = model._step(h, u[t])
        h2 = model._step(h2, u[t])
        d = h2 - h
        nd = np.linalg.norm(d)
        if nd == 0:
            logs.append(-np.inf)
            break
        logs.append(np.log(nd / eps))
        h2 = h + d * (eps / nd)
    lam = float(np.mean(logs))
    return {"lyapunov_exponent": lam,
            "regime": "contracting" if lam < -0.01 else
                      ("edge of chaos" if lam < 0.01 else "chaotic"),
            "n_steps": n_steps, "config": cfg.as_dict()}


def echo_state_property_check(config: ESNConfig | None = None, *,
                              n_steps: int = 1000, n_pairs: int = 20,
                              seed: int = 0) -> dict:
    """Do two different initial states converge under the same input?

    The textbook rule (spectral radius < 1) is neither necessary nor sufficient
    once input scaling is large, so measure it. Returns the mean final distance
    relative to the initial distance; << 1 means the property holds empirically.
    """
    cfg = config or ESNConfig()
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.8, 0.8, (n_steps, 1))
    model = ESN(cfg)
    if model.W is None:
        model._build(1)

    ratios, curves = [], []
    for _ in range(n_pairs):
        a = rng.uniform(-1, 1, cfg.n_reservoir)
        b = rng.uniform(-1, 1, cfg.n_reservoir)
        d0 = np.linalg.norm(a - b)
        curve = []
        for t in range(n_steps):
            a = model._step(a, u[t])
            b = model._step(b, u[t])
            curve.append(np.linalg.norm(a - b))
        ratios.append(curve[-1] / max(d0, 1e-30))
        curves.append(curve)
    mean_curve = np.mean(np.asarray(curves), axis=0)
    ratio = float(np.mean(ratios))
    return {"final_over_initial_distance": ratio,
            "holds": bool(ratio < 1e-3),
            "distance_curve": mean_curve[::max(len(mean_curve) // 200, 1)].tolist(),
            "reported_spectral_radius": cfg.spectral_radius,
            "actual_spectral_radius":
                model.train_diagnostics.get("actual_spectral_radius"),
            "config": cfg.as_dict()}
