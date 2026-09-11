r"""Differentiating through a draw from a categorical distribution.

A sampled token is not a function of the logits it was drawn from. It is a
function of the logits **and** a random number, and the map from logits to
sample is piecewise constant: nudge a logit and almost surely nothing happens,
until at some point everything does. The derivative is zero almost everywhere
and undefined on a measure-zero set, so there is no gradient to approximate.

What there is, is a gradient of the **expectation**:

    L(z) = E_{t ~ softmax(z)} [ loss(t) ],
    dL/dz_j = p_j ( loss(j) - E[loss] ),

which is the softmax Jacobian of episode 2 applied to the vector of losses,
and which therefore sums to exactly zero. On a vocabulary of 65 that
expectation is enumerable, so every estimator below can be compared against
the answer rather than against each other.

**The three ways people estimate it.**

* **REINFORCE**: draw one token and return `(loss(t) - b)(e_t - p)`. Unbiased
  for any baseline `b` that does not depend on `t`, because `E[e_t - p] = 0`.
  Expensive in variance and famous for it.
* **Straight-through**: draw a token, use the one-hot forward, and pretend on
  the way back that the one-hot was `p`. This is `onehot + p - p.detach()`, and
  it is the identity-written-three-ways trick from episode 1 used on purpose.
* **Gumbel-softmax**: do not sample at all; use a relaxed sample at
  temperature `tau`, which is differentiable and wrong by an amount that
  shrinks as `tau` does, while the variance grows.

**What straight-through actually assumes.** Differentiating the one-hot
forward gives `p * (u - <p, u>)` where `u_j` is the directional derivative of
the loss along token `j`'s embedding, evaluated at the sampled token. The
exact gradient is `p * (loss - E[loss])`. Same softmax Jacobian, and in place
of the true loss of each alternative, a **first-order extrapolation from the
token that happened to be drawn**. Straight-through is exactly as good as the
assumption that the loss is linear in embedding space between one token and
another.

**Measured on this model, that assumption is weak.** Within a context, the
correlation between a token's true change in loss and the linear
extrapolation of it runs from **0.06 to 0.54** across six contexts, median
0.11, with regression slopes from -0.28 to 1.26. Positive, so the
extrapolation is not useless; weak enough that most of what it claims about
the alternatives is noise. Token embeddings are far apart on the scale where
the loss is linear.

Pooling those six contexts into one correlation gives **-0.17**, which is a
Simpson's paradox and not a finding: each context has its own anchor token
and its own spread of losses, so the pooled number measures between-context
variation and answers a question nobody asked. `survey` reports the spread
for that reason.

**And the direction agreement is borrowed.** Straight-through's mean still has
a cosine of 0.79 to 0.94 with the exact gradient, which looks like a defence
until you apply the same softmax Jacobian to a random vector in place of the
losses: that scores a median cosine of 0.38 to 0.84 and a 90th percentile of
0.79 to 0.97. Straight-through's direction is what the shared Jacobian gives
anything, including noise. Its own contribution is the part that is
anti-correlated.

What it gets badly wrong is the size. Its expected gradient is 4% to 40% of
the exact one, so a run using it is taking systematically short steps in a
direction that is mostly the softmax's opinion rather than the loss's.

**Which is not an argument for never using it.** A biased estimator with small
variance beats an unbiased one with large variance at a small enough sample
count, and the crossover is computable: squared bias against variance over
`n`. On this problem REINFORCE with a mean baseline overtakes straight-through
at **2 to 9 samples**, which is a low bar and the number worth knowing.

References: Williams, "Simple statistical gradient-following algorithms for
connectionist reinforcement learning", *Machine Learning* (1992), for
REINFORCE; Bengio, Leonard and Courville, "Estimating or propagating gradients
through stochastic neurons for conditional computation" (2013), for
straight-through; Jang, Gu and Poole, "Categorical reparameterization with
Gumbel-softmax", *ICLR* (2017) and Maddison, Mnih and Teh, "The concrete
distribution", *ICLR* (2017), for the relaxation, discovered twice.
"""

from __future__ import annotations

import math


def _np():
    import numpy as np
    return np


def _torch():
    import torch
    return torch


def decision(context, target: int) -> dict:
    """One sampling decision, with its expectation enumerated.

    `context` is a token sequence; the model's distribution over the next
    token is what we sample from, and `target` is the token that has to be
    predicted *after* the sample. With a vocabulary of 65 the whole
    expectation is 65 forward passes, so nothing here is estimated.
    """
    np = _np()
    torch = _torch()
    F = torch.nn.functional

    from standarderror.llm import tiny

    bundle = tiny.load()
    model, V = bundle["model"], bundle["vocab"]
    ctx = torch.as_tensor(context, dtype=torch.long)
    with torch.no_grad():
        logits, _, _, _ = model(ctx.unsqueeze(0))
        z = logits[0, -1]
        p = F.softmax(z, -1).numpy()
        seqs = torch.stack([torch.cat([ctx, torch.tensor([t])])
                            for t in range(V)])
        out, _, _, _ = model(seqs)
        loss = F.cross_entropy(out[:, -1], torch.full((V,), int(target)),
                               reduction="none").numpy()
    mean = float(p @ loss)
    return {"logits": z.numpy(), "p": p, "loss": loss, "expected": mean,
            "exact": p * (loss - mean), "vocab": int(V),
            "context": ctx, "target": int(target),
            "max_p": float(p.max()),
            "entropy": float(-(p * np.log(p + 1e-30)).sum())}


def _soft_forward(model, ctx, target: int):
    """The loss as a differentiable function of a distribution over one token.

    The only change from the model's own forward is that the last position's
    embedding is `w @ E` rather than `E[t]`, which agrees exactly when `w` is
    one-hot -- checked in the tests, because every number here depends on it.
    """
    torch = _torch()
    F = torch.nn.functional
    T = len(ctx) + 1

    def f(w):
        e = torch.cat([model.tok(ctx), (w @ model.tok.weight).unsqueeze(0)], 0)
        h = (e + model.pos(torch.arange(T))).unsqueeze(0)
        for blk in model.blocks:
            h, _ = blk(h)
        return F.cross_entropy(model.head(model.lnf(h))[0, -1:],
                               torch.tensor([int(target)]))
    return f


def straight_through(kind: str = "softmax"):
    """The two ways people write it, which do not agree.

    `softmax` is `onehot + p - p.detach()`, the usual one. `logits` is
    `onehot + z - z.detach()`, which passes the gradient through as if the
    one-hot were the logits themselves. Episode 1's point, in the wild: these
    are the same forward pass and different derivatives.
    """
    torch = _torch()
    F = torch.nn.functional

    def build(z, gen, V):
        p = F.softmax(z, -1)
        t = torch.multinomial(p.detach(), 1, generator=gen)
        hard = F.one_hot(t[0], V).float()
        if kind == "softmax":
            return hard + p - p.detach()
        return hard + z - z.detach()
    return build


def gumbel(tau: float, hard: bool = False):
    """The relaxation. `hard` rounds the forward and keeps the soft backward."""
    torch = _torch()
    F = torch.nn.functional

    def build(z, gen, V):
        u = torch.rand(V, generator=gen).clamp_min(1e-20)
        s = F.softmax((z - torch.log(-torch.log(u))) / float(tau), -1)
        if not hard:
            return s
        return F.one_hot(s.argmax(), V).float() + s - s.detach()
    return build


def pathwise(dec: dict, build, *, draws: int = 400, seed: int = 0) -> dict:
    """Sample gradients from any estimator that differentiates the forward.

    Covers straight-through and the Gumbel relaxations; REINFORCE is scored
    separately because it needs no backward pass at all.
    """
    np = _np()
    torch = _torch()

    from standarderror.llm import tiny

    model = tiny.load()["model"]
    f = _soft_forward(model, dec["context"], dec["target"])
    V = dec["vocab"]
    out = []
    for s in range(int(draws)):
        gen = torch.Generator().manual_seed(int(seed) + s)
        z = torch.tensor(dec["logits"], requires_grad=True)
        f(build(z, gen, V)).backward()
        out.append(z.grad.numpy().copy())
    return _score(np.asarray(out), dec)


def reinforce(dec: dict, *, draws: int = 6000, seed: int = 0,
              baseline: float | None = None) -> dict:
    """The score-function estimator, with an optional constant baseline.

    `baseline=None` uses none; passing the expected loss is the cheapest
    variance reduction there is and changes nothing about the expectation.
    """
    np = _np()
    p, loss = dec["p"], dec["loss"]
    b = 0.0 if baseline is None else float(baseline)
    rng = np.random.default_rng(seed)
    ts = rng.choice(len(p), size=int(draws), p=p)
    e = np.zeros((int(draws), len(p)))
    e[np.arange(int(draws)), ts] = 1.0
    return _score((loss[ts] - b)[:, None] * (e - p), dec)


def _score(samples, dec: dict) -> dict:
    np = _np()
    exact = dec["exact"]
    n = samples.shape[0]
    mean = samples.mean(0)
    en = np.linalg.norm(exact)
    mn = np.linalg.norm(mean)
    var = float(((samples - mean) ** 2).sum(1).mean())
    return {
        "draws": int(n),
        "mean_norm": float(mn),
        "exact_norm": float(en),
        "scale": float(mn / en),
        "bias": float(np.linalg.norm(mean - exact)),
        "relative_bias": float(np.linalg.norm(mean - exact) / en),
        "cosine": float(mean @ exact / (mn * en)) if mn > 0 else float("nan"),
        "sd": math.sqrt(var),
        # The Monte Carlo error on `mean` itself, so a small measured bias can
        # be told apart from an actually unbiased estimator.
        "mc_error": math.sqrt(var / n) / en,
    }


def crossover(biased: dict, unbiased: dict) -> float:
    """Sample count at which the unbiased estimator wins on total error.

    Squared error of an average of `n` draws is `bias^2 + var/n`. The biased
    estimator's floor is its squared bias; the unbiased one falls forever. The
    crossover is `var_unbiased / (bias^2 - var_biased/n_measured)`, with the
    subtraction removing the Monte Carlo inflation of the measured bias.
    """
    b2 = biased["bias"] ** 2 - biased["sd"] ** 2 / biased["draws"]
    return unbiased["sd"] ** 2 / max(b2, 1e-12)


def linearisation(dec: dict) -> dict:
    """What straight-through assumes, against what is true.

    Its backward pass carries `u_j`, the directional derivative of the loss
    along token `j`'s embedding evaluated at the token actually drawn. That is
    a first-order prediction of `loss(j) - loss(t)`. This measures both.
    """
    np = _np()
    torch = _torch()
    F = torch.nn.functional

    from standarderror.llm import tiny

    model = tiny.load()["model"]
    f = _soft_forward(model, dec["context"], dec["target"])
    V = dec["vocab"]
    t0 = int(np.argmax(dec["p"]))
    w = F.one_hot(torch.tensor(t0), V).float().requires_grad_(True)
    f(w).backward()
    u = w.grad.numpy()
    predicted = u - u[t0]
    true = dec["loss"] - dec["loss"][t0]
    keep = np.arange(V) != t0
    return {"anchor": t0, "predicted": predicted[keep], "true": true[keep],
            "correlation": float(np.corrcoef(predicted[keep],
                                             true[keep])[0, 1]),
            "slope": float(np.polyfit(predicted[keep], true[keep], 1)[0]),
            # The compression, which turns out to explain the scale deficit
            # better than the correlation does: a loss model that says nearly
            # the same thing about every alternative gives nearly no gradient,
            # because `p * (u - <p, u>)` annihilates a constant `u` exactly.
            "predicted_spread": float(predicted[keep].std()),
            "true_spread": float(true[keep].std()),
            "spread_ratio": float(predicted[keep].std() / true[keep].std())}


def jacobian_control(dec: dict, *, draws: int = 2000, seed: int = 0) -> dict:
    """How much direction agreement the shared softmax Jacobian gives for free.

    Replace the loss vector with noise of the same mean and spread, push it
    through the same `p * (r - <p, r>)`, and see how well *that* aligns with
    the exact gradient. Whatever this scores is the floor an estimator has to
    beat before its cosine means anything.
    """
    np = _np()
    p, loss, exact = dec["p"], dec["loss"], dec["exact"]
    rng = np.random.default_rng(seed)
    en = np.linalg.norm(exact)
    cs = []
    for _ in range(int(draws)):
        r = rng.normal(size=len(p)) * loss.std() + loss.mean()
        g = p * (r - p @ r)
        cs.append(abs(g @ exact) / (np.linalg.norm(g) * en))
    cs = np.asarray(cs)
    return {"median": float(np.median(cs)), "p90": float(np.percentile(cs, 90)),
            "max": float(cs.max())}


def survey(*, contexts: int = 6, seed: int = 0, st_draws: int = 400,
           rf_draws: int = 6000) -> dict:
    """Every estimator on several decisions, against the enumerated truth."""
    np = _np()

    from standarderror.llm import tiny

    x, y = next(iter(tiny.batches(count=1, size=int(contexts), seed=int(seed))))
    rows = []
    lin_true, lin_pred = [], []
    for k in range(int(contexts)):
        dec = decision(x[k, :tiny.BLOCK - 1], int(y[k, tiny.BLOCK - 1]))
        st = pathwise(dec, straight_through("softmax"), draws=st_draws)
        rf = reinforce(dec, draws=rf_draws, baseline=dec["expected"])
        lin = linearisation(dec)
        ctrl = jacobian_control(dec)
        lin_true.append(lin["true"])
        lin_pred.append(lin["predicted"])
        rows.append({"context": k, "max_p": dec["max_p"],
                     "entropy": dec["entropy"],
                     "exact_norm": st["exact_norm"],
                     "st": st, "reinforce": rf, "control": ctrl,
                     "crossover": crossover(st, rf),
                     "correlation": lin["correlation"],
                     "slope": lin["slope"],
                     "spread_ratio": lin["spread_ratio"]})
    t = np.concatenate(lin_true)
    q = np.concatenate(lin_pred)
    # Deliberately NOT a correlation of the pooled arrays. Each context has
    # its own anchor token and its own spread of losses, so pooling mixes
    # between-context variation into a within-context question and produced a
    # negative number here where every context on its own was positive. The
    # honest summary of a per-context measurement is the per-context spread.
    cors = [r["correlation"] for r in rows]
    ratios = [r["spread_ratio"] for r in rows]
    scales = [r["st"]["scale"] for r in rows]
    return {"rows": rows, "alternatives": int(t.size),
            "spread_ratio_min": float(min(ratios)),
            "spread_ratio_max": float(max(ratios)),
            # Does the compression predict the scale deficit across contexts?
            "spread_vs_scale": float(np.corrcoef(ratios, scales)[0, 1]),
            "correlation_min": float(min(cors)),
            "correlation_max": float(max(cors)),
            "correlation_median": float(np.median(cors)),
            "slope_min": float(min(r["slope"] for r in rows)),
            "slope_max": float(max(r["slope"] for r in rows)),
            "pooled_correlation": float(np.corrcoef(q, t)[0, 1]),
            "predicted": q, "true": t}


def temperature_sweep(dec: dict, *, taus=(2.0, 1.0, 0.5, 0.2, 0.1),
                      draws: int = 300, seed: int = 0, hard: bool = False
                      ) -> list[dict]:
    """The relaxation's trade: bias falls with temperature, variance rises.

    Included because it is the honest alternative to both of the others, and
    because the trade has a floor -- past some temperature the bias stops
    improving while the variance keeps going.
    """
    out = []
    for tau in taus:
        s = pathwise(dec, gumbel(float(tau), hard=hard), draws=draws,
                     seed=seed)
        s["tau"] = float(tau)
        out.append(s)
    return out


def error_curve(score: dict, counts) -> list[float]:
    """Expected total error of averaging `n` draws: sqrt(bias^2 + var/n).

    Relative to the exact gradient's norm, so estimators with different
    scales are comparable. The Monte Carlo inflation of the measured bias is
    removed, the same correction `crossover` makes.
    """
    b2 = max(score["bias"] ** 2 - score["sd"] ** 2 / score["draws"], 0.0)
    return [math.sqrt(b2 + score["sd"] ** 2 / n) / score["exact_norm"]
            for n in counts]


def optimal_baseline(dec: dict) -> dict:
    """The variance-minimising *constant* baseline, and what it is worth.

    Minimising `Var[(loss - b)(e_t - p)]` over `b` gives a weighted mean of
    the losses, the weights being `||e_t - p||^2`, which is larger for
    unlikely tokens. So the optimal constant leans towards the tail rather
    than sitting at the mean.

    It is worth computing mostly to find out how little it buys. The mean
    baseline already captures nearly all of what any constant can, which
    means the variance that remains is not a baseline problem: it is the
    spread of the losses themselves, and no number subtracted from all of
    them shrinks that. Reducing it further needs a baseline that depends on
    which token was drawn -- a control variate, which is a model of the loss,
    which is what straight-through was trying to be.
    """
    p, loss = dec["p"], dec["loss"]
    # ||e_t - p||^2 = 1 - 2 p_t + ||p||^2
    w = 1.0 - 2.0 * p + float((p ** 2).sum())
    b = float((p * loss * w).sum() / (p * w).sum())
    return {"mean": dec["expected"], "optimal": b,
            "weights_tilt": float(b - dec["expected"])}
