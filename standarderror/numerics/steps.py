"""Gradient descent is forward Euler on the gradient flow, so it has a step-size
limit -- and on a network that limit is not where the textbook puts it.

The gradient flow `dx/dt = -grad f(x)` decreases `f` for every positive `t`;
there is no step size to get wrong, because there is no step. Discretise it with
forward Euler at step `lr` and you get `x <- x - lr * grad f(x)`, which is
gradient descent, and now there is. On a quadratic `f(x) = x'Hx/2` the iteration
is exactly `x <- (I - lr H) x`, so each eigendirection is multiplied by
`1 - lr*lam` every step, and the whole thing converges **iff** every one of those
multipliers is inside the unit circle:

    0 < lr < 2 / lam_max(H)

which is `stability_limit`. It is a threshold, not a guideline. On the 20-d
quadratic in `quadratic_design` (`lam_max = 10`, `kappa = 100`, so the critical
step is `0.2`), the gradient norm starts at `10.78` and after 400 steps is

    0.50 x critical    2.71e-03
    0.90 x critical    1.02e-04
    0.99 x critical    4.43e-04
    1.00 x critical    1.424642     <- marginal
    1.01 x critical    3.92e+03
    1.10 x critical    6.70e+31

Two of those rows are worth more than the shape of the list.

At **exactly** the critical step the sharpest direction has multiplier
`|1 - 2| = 1`, so it is neither amplified nor damped, and the run ends at
`1.424642`, which is exactly the size of `Hx0`'s component along the top
eigenvector -- preserved, to every digit, for 400 steps. Nothing is diverging and
nothing is converging.

And one row later, at a 1% larger step, the same run reaches `3.9e+03`. Whereas
the exact flow that all of this approximates gets to `4.50e-05` by the same
integration time, and to `2.19e-05` at the step size where Euler reaches
`6.7e+31`. The blow-up is not in the problem; it is in the discretisation.

**The optimum sits at `kappa/(kappa+1)` of the stability limit.** The best fixed
step for a quadratic is `2/(lam_min + lam_max)`, and dividing that by `2/lam_max`
leaves exactly `kappa/(kappa+1)` -- 75% of the limit at `kappa = 3`, 99% at
`kappa = 100`, 99.9% at `kappa = 1000`. So on any badly conditioned problem the
optimum is *on the cliff edge*, and "raise the learning rate until it breaks,
then back off a little" is very nearly the right procedure, which is both why it
survives as folklore and why it is dangerous: the target is within 1% of a
threshold that costs four orders of magnitude to cross.

What it buys is small. At `kappa = 100` the optimal contraction is
`(kappa-1)/(kappa+1) = 0.9802` per step, or **115 steps per decade** -- and no
choice of fixed step does better. Ill-conditioning is a wall the step size cannot
get through, which is what momentum is actually for.

**Momentum widens the limit by exactly `1 + beta`.** Heavy-ball is stable for
`lr < 2(1+beta)/lam_max`; bisecting for the true threshold on the design above
recovers that formula to a relative `3.4e-04` at `beta = 0` and `8.1e-06` at
`beta = 0.9`. That is most of why a learning rate which diverges plain trains
fine with momentum. The tuned pair
`beta = ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2 = 0.669`, `lr = 0.331`
(`momentum_optimal`) measures 12.2 steps per decade against 115 -- a **10x**
speedup, because `sqrt(kappa)` has replaced `kappa`.

**On a real network the limit moves, because the curvature does.** `lam_max` is
not a property of the problem; it is a function of where the iterates went, and
where they went depends on `lr`. Full-batch gradient descent on a small tanh MLP
(`edge_of_stability`: 200 rows, two hidden layers of 40, 4000 steps) gives three
regimes:

    lr = 0.02    lam_max ends  26.8    lam_max*lr/2 = 0.268    loss monotone
    lr = 0.05                  29.5                   0.738    loss monotone
    lr = 0.10                  20.0                   1.001    rose on 40% of steps
    lr = 0.20                  10.0                   1.002    rose on 26% of steps
    lr = 0.50                   4.0                   1.002    rose on 46% of steps
    lr = 0.80    diverged at step 6

At small `lr` the sharpness rises during training and then plateaus *below* the
boundary. From `lr = 0.1` up it stops at the boundary instead: `lam_max` lands on
`2/lr` to within 0.2% across a five-fold range of `lr`, and from either side --
starting at `lam_max = 7.36`, the `lr = 0.5` run is pushed *down* to `4.0`. This
is the edge of stability (Cohen et al., 2021), and the honest version is
two-sided: `2/lr` is an attractor, not a ceiling that sharpness climbs to.

The consequence to keep: computing `2/lam_max` at initialisation on this network
gives `0.2719`, and bisection finds the true limit between `0.55508` and
`0.55566` -- **2.04 times larger**, and still sharp to four figures. The
threshold is real and worth knowing; its location at initialisation understates
it by about a factor of two, because the network sharpens to whatever step you
chose until it cannot. Above the limit nothing degrades gracefully: it diverges
in 3 to 29 steps.

One more thing the edge regime does to a habit. The loss is not monotone there.
At `lr = 0.2` it rises on 523 of the last 2000 steps -- median rise 0.2%, largest
1% -- while falling 71-fold over the same window. A training loss that ticks up
at a large step size is what convergence looks like there, not evidence of a bug.

`power_iteration` is what makes any of this actionable: `lam_max` costs a few
Hessian-vector products, each about the price of one backward pass, and it tells
you where you are relative to `2/lr` before the run does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Heavy-ball momentum's stability region is `lr < 2(1+beta)/lam_max`, so this is
# the factor by which the default buys a larger step.
DEFAULT_BETA = 0.9


def quadratic_design(d: int = 20, *, condition: float = 100.0,
                     seed: int = 0) -> np.ndarray:
    """A symmetric positive-definite `H` with a known spectrum.

    Eigenvalues are geometrically spaced over `condition`, then rotated by a
    random orthogonal matrix so that no eigendirection is an axis -- otherwise
    the iteration decouples in the coordinates you happen to be printing and
    every claim below looks easier than it is.
    """
    if d < 2:
        raise ValueError("need at least two directions to have a condition number")
    eigs = np.geomspace(1.0, condition, d) / np.sqrt(condition)
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q @ np.diag(eigs) @ q.T


def spectrum(H) -> np.ndarray:
    """Eigenvalues of a symmetric matrix, ascending."""
    return np.linalg.eigvalsh(np.asarray(H, dtype=float))


def stability_limit(H) -> float:
    """`2 / lam_max`. Above this, forward Euler on the gradient flow diverges."""
    lam_max = float(spectrum(H)[-1])
    if lam_max <= 0:
        raise ValueError("stability_limit needs a positive largest eigenvalue")
    return 2.0 / lam_max


def optimal_lr(H) -> float:
    """`2 / (lam_min + lam_max)`, the step that minimises the worst-case
    per-step contraction -- not the largest stable step."""
    e = spectrum(H)
    return 2.0 / float(e[0] + e[-1])


def optimal_rate(H) -> float:
    """`(kappa - 1) / (kappa + 1)`: the contraction per step at `optimal_lr`,
    which is the best any fixed step size can do."""
    e = spectrum(H)
    kappa = float(e[-1] / e[0])
    return (kappa - 1.0) / (kappa + 1.0)


def optimal_lr_fraction(H) -> float:
    """`kappa / (kappa + 1)`: where the optimal step sits as a fraction of the
    stability limit. Exact, and it goes to 1 as the problem gets harder."""
    e = spectrum(H)
    kappa = float(e[-1] / e[0])
    return kappa / (kappa + 1.0)


def momentum_optimal(H) -> dict:
    """The tuned heavy-ball pair, and the rate it achieves.

    `beta = ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2` with
    `lr = 4/(sqrt(lam_min) + sqrt(lam_max))^2`. The rate is the plain one with
    `sqrt(kappa)` in place of `kappa`, which is the entire point of momentum on
    a quadratic -- and it is asymptotic, so a short run measures slightly worse.
    """
    e = spectrum(H)
    sk = float(np.sqrt(e[-1] / e[0]))
    return {"beta": ((sk - 1.0) / (sk + 1.0)) ** 2,
            "lr": 4.0 / float(np.sqrt(e[0]) + np.sqrt(e[-1])) ** 2,
            "rate": (sk - 1.0) / (sk + 1.0)}


def steps_per_decade(rate: float) -> float:
    """How many steps a per-step contraction needs to gain one decimal digit."""
    if not 0.0 < rate < 1.0:
        return float("inf")
    return float(np.log(0.1) / np.log(rate))


def momentum_limit(H, beta: float = DEFAULT_BETA) -> float:
    """`2(1 + beta) / lam_max`. Momentum does not change the shape of the
    stability region, only its width, and by exactly `1 + beta`."""
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    return (1.0 + beta) * stability_limit(H)


def amplification(H, lr: float) -> np.ndarray:
    """`|1 - lr*lam|` per eigendirection: what one step multiplies that
    component by.

    The same shape as ridge regression's `s^2/(s^2 + alpha)` multiplier from the
    linear-algebra series -- a per-direction gain you never see, decided by one
    scalar you chose. Entries above 1 are directions that grow.
    """
    return np.abs(1.0 - lr * spectrum(H))


def gd_quadratic(H, x0, lr: float, steps: int, *,
                 beta: float = 0.0, cap: float = 1e300) -> np.ndarray:
    """Gradient norms along `x <- x - lr*Hx` (with optional heavy-ball momentum).

    Returns `steps + 1` values and stops early on overflow, padding with `inf`,
    so that a diverging run is a readable series rather than a `RuntimeWarning`.
    """
    H = np.asarray(H, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    out = np.full(steps + 1, np.inf)
    # A diverging run overflows on the way to `inf`, and the warning is noise
    # rather than information: the `cap` below is what stops the loop.
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(steps + 1):
            g = H @ x
            n = float(np.linalg.norm(g))
            if not np.isfinite(n) or n > cap:
                break
            out[t] = n
            v = beta * v - lr * g
            x = x + v
    return out


def gradient_flow(H, x0, t: float) -> float:
    """The gradient norm of the *exact* solution `x(t) = exp(-tH) x0`.

    The continuous flow this all approximates. It contracts every eigendirection
    by `exp(-t*lam) < 1` for every positive `t` and every positive eigenvalue, so
    it cannot diverge at any "step size". Whatever blew up, the discretisation
    did it.
    """
    e, q = np.linalg.eigh(np.asarray(H, dtype=float))
    z = q.T @ np.asarray(x0, dtype=float)
    return float(np.linalg.norm(e * (np.exp(-t * e) * z)))


@dataclass
class LrSweep:
    """One row per multiple of the critical step size."""

    critical_lr: float
    steps: int
    multiples: tuple = ()
    final: tuple = ()

    @property
    def rows(self) -> list[dict]:
        return [{"multiple": m, "lr": m * self.critical_lr, "final_grad_norm": f,
                 "diverged": not np.isfinite(f) or f > 1e6}
                for m, f in zip(self.multiples, self.final)]

    @property
    def largest_converging(self) -> float:
        ok = [r["multiple"] for r in self.rows if not r["diverged"]]
        return max(ok) if ok else float("nan")

    def at(self, multiple: float) -> float:
        i = list(self.multiples).index(multiple)
        return float(self.final[i])


def lr_sweep(H, multiples=(0.5, 0.9, 0.99, 1.0, 1.01, 1.1), *,
             steps: int = 400, seed: int = 0, beta: float = 0.0) -> LrSweep:
    """Run `gd_quadratic` at multiples of the critical step and report the end."""
    H = np.asarray(H, dtype=float)
    crit = momentum_limit(H, beta) if beta else stability_limit(H)
    x0 = np.random.default_rng(seed).standard_normal(H.shape[0])
    final = []
    for m in multiples:
        hist = gd_quadratic(H, x0, m * crit, steps, beta=beta)
        final.append(float(hist[-1]))
    return LrSweep(critical_lr=crit, steps=steps,
                   multiples=tuple(multiples), final=tuple(final))


def divergence_threshold(run: Callable[[float], bool], lo: float, hi: float, *,
                         iters: int = 20) -> tuple[float, float]:
    """Bisect for the largest step size at which `run(lr)` still converges.

    `run` returns True for converged. Returns the bracket `(lo, hi)` rather than
    a midpoint, because the width is the interesting part: on a quadratic it
    closes on the analytic value, and on a network it closes on something else
    that is nonetheless just as sharp.
    """
    if not run(lo):
        raise ValueError("the lower end of the bracket must converge")
    if run(hi):
        raise ValueError("the upper end of the bracket must diverge")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if run(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi


def power_iteration(matvec: Callable, n: int, *, iters: int = 100,
                    seed: int = 0, tol: float = 1e-10) -> dict:
    """Largest eigenvalue by repeated matrix-vector products.

    The point is the cost: `iters` products, and for a loss surface a
    Hessian-vector product is about one extra backward pass. So `lam_max` is
    cheap enough to monitor during training, which is what makes `2/lr` an
    instrument rather than a fact about a matrix you will never form.
    """
    v = np.random.default_rng(seed).standard_normal(n)
    v /= np.linalg.norm(v)
    lam = 0.0
    for k in range(1, iters + 1):
        w = np.asarray(matvec(v), dtype=float)
        lam_new = float(v @ w)
        nrm = float(np.linalg.norm(w))
        if nrm == 0.0:
            return {"lam_max": 0.0, "iterations": k, "converged": True}
        v = w / nrm
        if abs(lam_new - lam) <= tol * max(1.0, abs(lam_new)):
            return {"lam_max": lam_new, "iterations": k, "converged": True}
        lam = lam_new
    return {"lam_max": lam, "iterations": iters, "converged": False}


def sharpness_ratio(lam_max: float, lr: float) -> float:
    """`lam_max * lr / 2`: where you are relative to the stability boundary.

    Below 1 the step is stable at the current curvature; at 1 you are on the
    edge, which on a network is where full-batch descent parks itself.
    """
    return float(lam_max) * float(lr) / 2.0


# --- the part that needs a network ------------------------------------------
#
# torch is an opt-in extra (see pyproject), so it is imported inside the
# function and the tests that reach here skip without it.

def edge_of_stability(lr: float, *, steps: int = 4000, width: int = 40,
                      depth: int = 2, n: int = 200, d: int = 8,
                      seed: int = 0, probes: int = 13,
                      sharpness_iters: int = 60) -> dict:
    """Full-batch gradient descent on a small tanh MLP, tracking `lam_max`.

    Full-batch on purpose: the edge-of-stability behaviour is a property of the
    deterministic iteration, and minibatch noise both blurs the boundary and
    invites the reader to attribute the whole effect to the noise.

    Returns the final loss, the final `lam_max`, `sharpness_ratio`, the fraction
    of the second half's steps on which the loss *rose*, and a coarse trace.
    """
    import torch

    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(seed)
    widths = [d] + [width] * depth + [1]
    layers: list = []
    for a, b in zip(widths[:-1], widths[1:]):
        lin = torch.nn.Linear(a, b)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(b, a, generator=g) / np.sqrt(a))
            lin.bias.zero_()
        layers += [lin, torch.nn.Tanh()]
    net = torch.nn.Sequential(*layers[:-1])

    gd = torch.Generator().manual_seed(seed + 1)
    X = torch.randn(n, d, generator=gd)
    y = (torch.sin(2 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]).unsqueeze(1)

    params = list(net.parameters())
    loss_fn = lambda: torch.nn.functional.mse_loss(net(X), y)   # noqa: E731

    def flat_grad(loss, create_graph=False):
        gs = torch.autograd.grad(loss, params, create_graph=create_graph)
        return torch.cat([p.reshape(-1) for p in gs])

    def sharpness() -> float:
        gr = flat_grad(loss_fn(), create_graph=True)
        v = torch.randn(gr.numel(), generator=torch.Generator().manual_seed(7))
        v = v / v.norm()
        lam = 0.0
        for _ in range(sharpness_iters):
            hv = torch.autograd.grad(gr @ v, params, retain_graph=True)
            hv = torch.cat([h.reshape(-1) for h in hv])
            lam = float((v @ hv).detach())
            nrm = hv.norm()
            if float(nrm) == 0.0:
                return 0.0
            v = (hv / nrm).detach()
        return lam

    every = max(1, steps // max(1, probes - 1))
    losses: list[float] = []
    trace: list[tuple[int, float, float]] = []
    for t in range(steps + 1):
        loss = loss_fn()
        lv = float(loss.detach())
        losses.append(lv)
        if not np.isfinite(lv) or lv > 1e6:
            return {"lr": lr, "diverged": True, "diverged_at": t,
                    "losses": losses, "trace": trace}
        if t % every == 0:
            trace.append((t, lv, sharpness()))
        gvec = flat_grad(loss)
        with torch.no_grad():
            i = 0
            for p in params:
                k = p.numel()
                p -= lr * gvec[i:i + k].view_as(p)
                i += k

    lam = sharpness()
    tail = losses[len(losses) // 2:]
    pairs = max(1, len(tail) - 1)
    rose = [(b - a) / a for a, b in zip(tail, tail[1:]) if b > a]
    return {"lr": lr, "diverged": False, "loss": losses[-1], "lam_max": lam,
            "ratio": sharpness_ratio(lam, lr),
            "rose_fraction": len(rose) / pairs,
            "median_rise": float(np.median(rose)) if rose else 0.0,
            "max_rise": max(rose) if rose else 0.0,
            "tail_drop": tail[0] / tail[-1] if tail[-1] > 0 else float("inf"), "losses": losses, "trace": trace}


def initial_sharpness(**kw) -> float:
    """`lam_max` of the untrained network `edge_of_stability` uses.

    Separate because it is the number the textbook threshold would be computed
    from, and the episode's point is how far off that is.
    """
    got = edge_of_stability(0.0, steps=0, probes=1, **kw)
    return float(got["trace"][0][2])
