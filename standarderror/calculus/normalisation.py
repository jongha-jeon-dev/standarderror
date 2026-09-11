r"""What LayerNorm removes from a gradient, and why you never notice.

**The Jacobian.** For `y = (x - mu) / sigma` with `sigma = sqrt(var(x) + eps)`,
mean and variance taken over the `d` features,

    dy/dx = (I - 11^T/d - xhat xhat^T/d) / sigma,   xhat = (x - mu)/sigma.

Two facts about that expression, both exact. `xhat` is standardised, so
`||1||^2 = ||xhat||^2 = d` and `1 . xhat = 0`: the two subtracted terms are
orthogonal rank-one projectors onto orthonormal directions. So the bracket is
the **orthogonal projector onto the complement of the plane spanned by 1 and
xhat**, its nonzero eigenvalues are all exactly `1/sigma`, and its rank is
exactly `d - 2`.

**Why those two directions.** LayerNorm is invariant to `x -> x + c*1` and to
`x -> a*x` for positive `a`. Differentiate an invariance and you get a null
direction, so the Jacobian must annihilate the tangent of each: `1` for the
shift and `xhat` for the scaling. That is the same theorem episode 2 met with
one invariance instead of two -- adding a constant to every logit does not
move a softmax, so `J 1 = 0` and one direction dies. **The rank deficiency is
not a leak; it is the derivative correctly reporting that the function ignores
those directions.**

With a learned gain the Jacobian is `diag(gamma) P / sigma`. Still rank exactly
`d - 2`, no longer a scaled projector: on this model the nonzero singular
values spread by under a factor of two.

**How much it costs, measured.** Across the model's nine LayerNorms the share
of the incoming gradient lying in the deleted plane has a median near 0.12,
against 0.105 for a uniformly random direction in 128 dimensions. Slightly
above chance and no more, most of the time; the tail is another matter, with
the final norm reaching 0.81 against a random maximum of 0.43.

**And then the residual makes it vanish.** A block is `x + f(LN(x))`, so its
Jacobian is `I + something`. The branch through `ln1` measures rank 123 of 128
at a position; the block containing it measures 128, with a smallest singular
value of 0.36. No transformer block is rank-deficient because of its
LayerNorm, and the deleted directions reach the layer below by the skip.

**Except at the end, where there is no skip.** The final norm feeds the output
head directly, so the two directions it deletes are exact invariances of the
whole network: adding any multiple of the all-ones vector to the final hidden
state leaves the loss identical to ten digits, and so does multiplying it by
any positive constant. The scaling invariance holds over seven orders of
magnitude and then breaks exactly where it must -- at `sigma` near
`sqrt(eps)`, below which the epsilon rather than the variance sets the
denominator.

**What actually varies is the scalar.** `1/sigma` falls from 1.11 at the
embedding to 0.32 at the final norm, because the residual stream grows from
0.90 to 3.90 rms as the blocks add to it. That 3.4-fold ladder does more to
the size of a gradient than the rank deficiency does, and it is the backward
view of a forward fact.

References: Ba, Kiros and Hinton, "Layer normalization" (2016), for the
definition; Xu et al., "Understanding and improving layer normalization",
*NeurIPS* (2019), for the derivative and the argument that the re-centering
and re-scaling of the *gradient* matter more than those of the forward pass;
Elhage et al., "A mathematical framework for transformer circuits" (2021), for
the residual stream as the object the blocks read from and write to.
"""

from __future__ import annotations

import math


def _np():
    import numpy as np
    return np


def _torch():
    import torch
    return torch


def projector(x):
    """`I - 11^T/d - xhat xhat^T/d`: the orthogonal projector LayerNorm applies.

    Built from the *input* rather than from anything learned, because that is
    what it depends on -- the deleted plane moves with every token.
    """
    np = _np()
    x = np.asarray(x, dtype=float)
    d = x.size
    xh = (x - x.mean()) / x.std()
    return np.eye(d) - np.ones((d, d)) / d - np.outer(xh, xh) / d


def jacobian(x, gamma=None, eps: float = 0.0):
    """The full LayerNorm Jacobian, `diag(gamma) P / sigma`."""
    np = _np()
    x = np.asarray(x, dtype=float)
    sigma = math.sqrt(x.var() + eps)
    j = projector(x) / sigma
    if gamma is not None:
        j = np.diag(np.asarray(gamma, dtype=float)) @ j
    return j


def against_autodiff(*, d: int = 8, seed: int = 0) -> dict:
    """The closed form against torch's own derivative.

    Worth doing once per claim of this kind. A transposed outer product or a
    missing `/d` gives a matrix that is still symmetric, still singular, and
    still looks right in a plot.
    """
    np = _np()
    torch = _torch()
    torch.manual_seed(int(seed))
    x = torch.randn(d, dtype=torch.double, requires_grad=True)
    ln = torch.nn.LayerNorm(d, elementwise_affine=False, eps=0.0).double()
    got = torch.autograd.functional.jacobian(ln, x).numpy()
    mine = jacobian(x.detach().numpy())
    return {"d": int(d), "max_abs_error": float(np.abs(got - mine).max()),
            "autodiff": got, "closed_form": mine}


def spectrum(x, gamma=None, eps: float = 0.0) -> dict:
    """Rank, eigenvalues, and the two directions that are annihilated.

    `null_residual` is the largest entry of `J u` over the two claimed null
    directions -- the check that the rank deficiency is where the algebra says
    it is, rather than merely present.
    """
    np = _np()
    x = np.asarray(x, dtype=float)
    d = x.size
    j = jacobian(x, gamma=gamma, eps=eps)
    sv = np.linalg.svd(j, compute_uv=False)
    tol = max(sv[0], 1e-300) * 1e-9
    xh = (x - x.mean()) / x.std()
    ones = np.ones(d)
    resid = max(float(np.abs(j @ ones).max()), float(np.abs(j @ xh).max()))
    return {
        "d": int(d),
        "sigma": math.sqrt(x.var() + eps),
        "singular_values": sv,
        "rank": int((sv > tol).sum()),
        "rank_deficiency": int(d - (sv > tol).sum()),
        "largest": float(sv[0]),
        "smallest_nonzero": float(sv[d - 3]) if d > 2 else float("nan"),
        "spread": float(sv[0] / sv[d - 3]) if d > 2 else float("nan"),
        "null_residual": resid,
        "orthogonality": float(abs(ones @ xh)),
    }


def gains(model) -> list[dict]:
    """Every LayerNorm in the model, its learned gain, and what that does.

    Without `gamma` the nonzero spectrum is flat at `1/sigma` -- a scaled
    orthogonal projection. `gamma` tilts it, and the question this answers is
    by how much: if the spread were large the projector picture would be a
    caricature rather than a description.
    """
    np = _np()
    rng = np.random.default_rng(0)
    out = []
    named = []
    for i, blk in enumerate(model.blocks):
        named.append((f"blocks.{i}.ln1", blk.ln1))
        named.append((f"blocks.{i}.ln2", blk.ln2))
    named.append(("lnf", model.lnf))
    for name, mod in named:
        g = mod.weight.detach().numpy()
        x = rng.normal(size=g.size)
        s = spectrum(x, gamma=g)
        out.append({"name": name, "gamma_mean": float(g.mean()),
                    "gamma_sd": float(g.std()), "gamma_min": float(g.min()),
                    "gamma_max": float(g.max()), "rank": s["rank"],
                    "spread": s["spread"]})
    return out


def norm_cost(share: float) -> float:
    """What removing a component of relative size `share` costs the norm.

    Worth a function because the intuition is wrong by an order of magnitude:
    deleting a component that is 10% of a vector's length shortens the vector
    by 0.5%, not by 10%, since the norms add in quadrature. Every statement in
    this module about what LayerNorm "removes" has to be converted through
    here before it can be compared with anything multiplicative.
    """
    return 1.0 - math.sqrt(max(0.0, 1.0 - float(share) ** 2))


def random_plane_share(*, d: int = 128, draws: int = 200_000,
                       seed: int = 0) -> dict:
    """How much of a uniformly random unit vector lands in a fixed 2-plane.

    The baseline the measured share has to beat before "LayerNorm deletes part
    of your gradient" means anything. Two directions out of `d` catch a
    perfectly indifferent gradient about `sqrt(2/d)` of the time.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(int(draws), int(d)))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    s = np.linalg.norm(v[:, :2], axis=1)
    return {"d": int(d), "draws": int(draws),
            "median": float(np.median(s)),
            "p90": float(np.percentile(s, 90)),
            "max": float(s.max()),
            "rms": float(np.sqrt((s ** 2).mean())),
            "sqrt_two_over_d": math.sqrt(2.0 / d)}


def deleted_share(*, count: int = 4, size: int = 8, seed: int = 0) -> dict:
    """Per LayerNorm: what fraction of the real gradient lies in the dead plane.

    Hooks every norm, keeps its input (which fixes the plane) and the gradient
    arriving at its output, and resolves that gradient onto the two
    orthonormal directions `1/sqrt(d)` and `xhat/sqrt(d)`.
    """
    np = _np()

    from standarderror.llm import tiny

    model = tiny.load()["model"]
    d = tiny.WIDTH
    caught: dict = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            out.retain_grad()
            caught[name] = (inp[0].detach(), out)
        return fn

    for i, blk in enumerate(model.blocks):
        handles.append(blk.ln1.register_forward_hook(hook(f"blocks.{i}.ln1")))
        handles.append(blk.ln2.register_forward_hook(hook(f"blocks.{i}.ln2")))
    handles.append(model.lnf.register_forward_hook(hook("lnf")))

    rows: dict = {}
    try:
        for x, y in tiny.batches(count=int(count), size=int(size),
                                 seed=int(seed)):
            caught.clear()
            _, loss, _, _ = model(x, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            for name, (xin, out) in caught.items():
                g = out.grad.detach()
                mu = xin.mean(-1, keepdim=True)
                sd = (xin.var(-1, unbiased=False, keepdim=True)
                      + 1e-5).sqrt()
                xh = (xin - mu) / sd
                root = math.sqrt(d)
                c1 = g.sum(-1) / root
                c2 = (g * xh).sum(-1) / root
                dead = (c1 ** 2 + c2 ** 2).sqrt()
                tot = g.norm(dim=-1)
                ok = tot > 0
                rows.setdefault(name, {"share": [], "inv_sigma": []})
                rows[name]["share"].append((dead[ok] / tot[ok]).numpy())
                rows[name]["inv_sigma"].append(
                    (1.0 / sd).squeeze(-1).numpy().ravel())
            model.zero_grad(set_to_none=True)
    finally:
        for h in handles:
            h.remove()

    out = []
    for name, acc in rows.items():
        s = np.concatenate(acc["share"])
        inv = np.concatenate(acc["inv_sigma"])
        out.append({"name": name, "rows": int(s.size),
                    "median": float(np.median(s)),
                    "p90": float(np.percentile(s, 90)),
                    "max": float(s.max()),
                    "median_inv_sigma": float(np.median(inv))})
    return {"norms": out, "baseline": random_plane_share(d=d)}


def block_rank(*, layer: int = 0, seed: int = 0) -> dict:
    """The rank of one block's Jacobian, and of the branch inside it.

    The claim being tested is that the skip connection restores what the
    LayerNorm removed. Measured at a single position, because that is where
    the question is well posed: the branch also mixes positions, so its rank
    at one position is lower than `d - 2` for reasons that have nothing to do
    with normalisation, and the number that matters is the block's.
    """
    np = _np()
    torch = _torch()

    from standarderror.llm import tiny

    model = tiny.load()["model"]
    blk = model.blocks[int(layer)]
    x, _ = next(iter(tiny.batches(count=1, size=1, seed=int(seed))))
    with torch.no_grad():
        base = model.tok(x) + model.pos(torch.arange(x.shape[1]))
    h0 = base[0, -1].detach().clone()

    def whole(h):
        z = base.clone()
        z[0, -1] = h
        out, _ = blk(z)
        return out[0, -1]

    def branch(h):
        z = base.clone()
        z[0, -1] = h
        a, _ = blk.attn(blk.ln1(z))
        return a[0, -1]

    def rank_of(f):
        j = torch.autograd.functional.jacobian(f, h0).numpy()
        sv = np.linalg.svd(j, compute_uv=False)
        return sv, int((sv > 1e-6 * max(sv[0], 1e-30)).sum())

    sv_w, rank_w = rank_of(whole)
    sv_b, rank_b = rank_of(branch)
    return {"d": int(tiny.WIDTH), "layer": int(layer),
            "block_rank": rank_w, "branch_rank": rank_b,
            "block_smallest": float(sv_w[-1]),
            "block_largest": float(sv_w[0])}


def _final_hidden(model, idx):
    torch = _torch()
    B, T = idx.shape
    h = model.tok(idx) + model.pos(torch.arange(T, device=idx.device))
    for blk in model.blocks:
        h, _ = blk(h)
    return h


def network_invariance(*, size: int = 8, seed: int = 0) -> dict:
    """Two exact invariances of the whole network, demonstrated on the loss.

    The final norm has no residual after it, so what it deletes is deleted
    from the model's output rather than merely from one branch. Adding any
    multiple of the all-ones vector to the final hidden state, or multiplying
    it by any positive constant, must leave every logit alone. The control is
    the same size of perturbation along one coordinate.
    """
    torch = _torch()
    F = torch.nn.functional

    from standarderror.llm import tiny

    bundle = tiny.load()
    model, vocab = bundle["model"], bundle["vocab"]
    x, y = next(iter(tiny.batches(count=1, size=int(size), seed=int(seed))))

    def loss_of(transform=None):
        with torch.no_grad():
            h = _final_hidden(model, x)
            if transform is not None:
                h = transform(h)
            logits = model.head(model.lnf(h))
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        return float(loss), logits

    base, base_logits = loss_of()
    e0 = F.one_hot(torch.tensor(0), tiny.WIDTH).float()
    cases = {
        "shift by 5 * ones": lambda h: h + 5.0,
        "shift by 1000 * ones": lambda h: h + 1000.0,
        "scale by 3": lambda h: h * 3.0,
        "scale by 3 then shift by 5": lambda h: h * 3.0 + 5.0,
        "shift by 5 along one axis": lambda h: h + 5.0 * e0,
    }
    out = {"baseline": base, "cases": []}
    for name, t in cases.items():
        ls, lg = loss_of(t)
        out["cases"].append({
            "name": name, "loss": ls, "delta": ls - base,
            "max_logit_change": float((lg - base_logits).abs().max())})
    return out


def scale_sweep(*, scales=(1e3, 1e1, 1.0, 1e-1, 1e-2, 3e-3, 1e-3, 1e-4),
                eps: float = 1e-5, size: int = 8, seed: int = 0) -> dict:
    """How the scale invariance degrades, and what sets the rate.

    `sigma = sqrt(var + eps)` is homogeneous in `x` only while `var` dominates
    `eps`, so the natural control variable is the ratio `eps/var` rather than
    `sigma` itself. It is not a cliff: over four decades of that ratio
    the error in the loss is **linear** in it -- a log-log slope near 0.97
    with a constant near 0.05, stable across batch sizes -- so there is no
    threshold anywhere, and `sqrt(eps)`, where `eps/var` reaches one, is well
    past the point the invariance stopped being exact. It reads as exactly
    zero only where the product falls under float32's own resolution.

    The largest scales are a control. Any deviation there is float32 rounding
    on a big number rather than the epsilon, and `ratio` makes that obvious by
    coming out orders of magnitude off the line.
    """
    np = _np()
    torch = _torch()
    F = torch.nn.functional

    from standarderror.llm import tiny

    bundle = tiny.load()
    model, vocab = bundle["model"], bundle["vocab"]
    x, y = next(iter(tiny.batches(count=1, size=int(size), seed=int(seed))))
    with torch.no_grad():
        h0 = _final_hidden(model, x)
        sigma0 = float(h0.std(-1).median())
        base = float(F.cross_entropy(
            model.head(model.lnf(h0)).reshape(-1, vocab), y.reshape(-1)))
        rows = []
        for s in scales:
            ls = float(F.cross_entropy(
                model.head(model.lnf(h0 * s)).reshape(-1, vocab),
                y.reshape(-1)))
            sd = sigma0 * float(s)
            ratio = float(eps) / (sd * sd)
            rows.append({"scale": float(s), "sigma": sd, "loss": ls,
                         "delta": ls - base, "eps_over_var": ratio,
                         "ratio": abs(ls - base) / ratio if ratio else
                         float("nan")})
    # Rows where the epsilon is the mechanism: eps/var large enough to see and
    # not so small that float32 noise dominates the comparison.
    live = [r for r in rows if 1e-5 < r["eps_over_var"] < 10.0]
    slope = constant = float("nan")
    if len(live) >= 3:
        lx = np.log([r["eps_over_var"] for r in live])
        ly = np.log([abs(r["delta"]) for r in live])
        s, i = np.polyfit(lx, ly, 1)
        slope, constant = float(s), float(np.exp(i))
    return {"sigma": sigma0, "eps": float(eps),
            "sqrt_eps": math.sqrt(float(eps)), "baseline": base,
            "rows": rows, "proportional_rows": live,
            "slope": slope, "constant": constant,
            "ratio_min": min((r["ratio"] for r in live), default=float("nan")),
            "ratio_max": max((r["ratio"] for r in live), default=float("nan"))}


def sigma_ladder(*, size: int = 8, seed: int = 0) -> dict:
    """The residual stream's growth, and the `1/sigma` it hands the backward pass.

    Every LayerNorm divides by the scale of what the residual stream has
    accumulated so far. So the one scalar in its Jacobian is set by a forward
    fact, and reading the two together is the point of this table.
    """
    torch = _torch()

    from standarderror.llm import tiny

    model = tiny.load()["model"]
    x, _ = next(iter(tiny.batches(count=1, size=int(size), seed=int(seed))))
    rows = []
    with torch.no_grad():
        h = model.tok(x) + model.pos(torch.arange(x.shape[1]))
        rows.append(("embedding", h.clone()))
        for i, blk in enumerate(model.blocks):
            a, _ = blk.attn(blk.ln1(h))
            h = h + a
            rows.append((f"after attention {i}", h.clone()))
            h = h + blk.mlp(blk.ln2(h))
            rows.append((f"after mlp {i}", h.clone()))
    out = []
    for name, h in rows:
        sd = float(h.std(-1).median())
        out.append({"point": name,
                    "rms": float(h.pow(2).mean().sqrt()),
                    "sigma": sd, "inv_sigma": 1.0 / sd})
    return {"points": out,
            "growth": out[-1]["rms"] / out[0]["rms"],
            "attenuation": out[0]["inv_sigma"] / out[-1]["inv_sigma"]}
