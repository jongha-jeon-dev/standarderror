r"""What a softmax passes backwards, and why confidence is the thing that stops it.

Episode 1 went looking for non-differentiability in a transformer and did not
find it. What it found instead was saturation: 12.9% of this model's attention
rows already put more than 0.9 of their mass on one position. Saturation is
perfectly smooth. It is also, exactly, where the gradient goes.

**The Jacobian.** For `p = softmax(z)`,

    dp_i/dz_j = p_i (delta_ij - p_j),   so   J = diag(p) - p p^T.

`J` is symmetric, positive semi-definite, and singular: `J 1 = p - p = 0`,
because adding a constant to every logit does not change the softmax. So the
gradient with respect to the logits of any one row **sums to exactly zero**,
always, for every row of every head. One direction of the gradient is deleted
by the function itself, not by the data.

**The quadratic form is a variance.** For any vector u,

    u^T J u = sum_i p_i u_i^2 - (sum_i p_i u_i)^2 = Var_p(u),

so the spectral norm is the largest variance a unit vector can have under `p`:
`||J||_2 = max_{||u||=1} Var_p(u)`. A distribution concentrated on one atom has
no variance to give, which is the whole story in one line.

**The bound.** Write `m = max_i p_i`. Then

    m(1 - m)  <=  ||J||_2  <=  2 m (1 - m)      (for m >= 1/2)

The lower bound is `e_1^T J e_1 = m(1-m)` and PSD matrices have `lambda_max`
at least any diagonal entry. The upper bound is two lines: take `c = u_1` in
`Var_p(u) <= E_p[(u - c)^2]`, giving `sum_{i != 1} p_i (u_i - u_1)^2 <=
2 sum_{i != 1} p_i = 2(1 - m)`; the sharper `2m(1-m)` is attained exactly when
the losing mass sits on a single runner-up, and `bound_sweep` checks it.

So the gradient a softmax can pass is pinned within a factor of two by its
largest probability alone -- **independently of how many positions it has**.
A 64-way softmax with `m = 0.99` has the same gradient capacity as a two-way
one: `2m(1-m) = 0.0198`, a fiftieth of the `0.5` available at `m = 1/2`.

**And the trace is exactly a collision probability.** `tr J = 1 - sum_i p_i^2`,
which is one minus the chance two independent draws from `p` agree. Not the
Shannon entropy, which is the quantity usually reached for; the exact one.

**Measured on the model, the bound is a predictor and not just a ceiling.**
The realised gain `||J g|| / ||g||` for the actual backward-pass gradient `g`
scales as `2m(1-m)` to the power 1.03, with a constant near an eighth: the
incoming gradient is not aligned with the top eigenvector, so it collects about
an eighth of the available capacity, and that fraction is roughly constant
across confidence levels.

**The finding.** One of this model's sixteen heads -- layer 0, head 1 -- has a
median maximum probability of 0.97 and puts its argmax exactly one position
back on 99.8% of rows. Its routing gradient is 5.9 times smaller than the
next-lowest head's and 10.4 times smaller than the softest head's. It is also the most important head in the model: zeroing it costs 1.12
nats of validation loss, where zeroing its neighbour costs 0.011. And replacing
its attention with a hard-wired shift-by-one permutation matrix costs 0.0013
nats -- 860 times less than removing it.

Which is the honest shape of the result. The head is not broken and it is not
stuck at a bad answer; it has committed to a good one, and the mechanism of
commitment is the same mechanism that removes the gradient which could reverse
it. Saturation is how a softmax decides, and deciding is close to one-way.

References: Bridle, "Probabilistic interpretation of feedforward classification
network outputs" (1990), for the softmax Jacobian; Elhage et al., "A
mathematical framework for transformer circuits" (2021), for previous-token
heads and why a layer-0 head becomes one; Vaswani et al. (2017) for the
`1/sqrt(d)` scale, which exists precisely to keep `m` away from 1 at
initialisation.
"""

from __future__ import annotations


def _np():
    import numpy as np
    return np


def _torch():
    import torch
    return torch


def jacobian(p):
    """`diag(p) - p p^T`, the softmax Jacobian at the point whose output is p.

    Takes the *output* rather than the logits because that is what the Jacobian
    depends on -- two logit vectors differing by a constant have the same p and
    the same Jacobian, which is the singularity below written another way.
    """
    np = _np()
    p = np.asarray(p, dtype=float)
    return np.diag(p) - np.outer(p, p)


def spectrum(p) -> dict:
    """Eigenvalues of the Jacobian, its trace, and the two bounds on the norm.

    `collision` is `1 - sum p^2`, which equals the trace exactly; it is
    reported alongside so the identity can be checked rather than trusted.
    """
    np = _np()
    p = np.asarray(p, dtype=float)
    m = float(p.max())
    ev = np.linalg.eigvalsh(jacobian(p))
    return {
        "max_p": m,
        "eigenvalues": ev,
        "spectral_norm": float(ev.max()),
        "smallest": float(ev.min()),
        "trace": float(ev.sum()),
        "collision": float(1.0 - (p ** 2).sum()),
        "lower_bound": m * (1.0 - m),
        "upper_bound": 2.0 * m * (1.0 - m),
        "loose_bound": 2.0 * (1.0 - m),
    }


def bound_sweep(*, sizes=(2, 3, 4, 8, 16, 64), draws: int = 20_000,
                seed: int = 3) -> dict:
    """Does `m(1-m) <= ||J|| <= 2m(1-m)` hold, and is either end attained?

    Random distributions at random temperatures, which is the only way to get
    coverage of the confidence range: a fixed temperature concentrates `m`.
    Rows with `m` above 0.9999 are dropped, because there the eigenvalue is at
    the scale of float error and the ratio is measuring noise rather than the
    bound.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    m_all, lam_all, n_all = [], [], []
    for n in sizes:
        for _ in range(int(draws)):
            z = rng.normal(0.0, rng.uniform(0.05, 9.0), n)
            p = np.exp(z - z.max())
            p /= p.sum()
            m = p.max()
            if not (0.5 <= m <= 0.9999):
                continue
            lam = np.linalg.eigvalsh(jacobian(p)).max()
            m_all.append(m)
            lam_all.append(lam)
            n_all.append(n)
    m = np.asarray(m_all)
    lam = np.asarray(lam_all)
    lo, hi = m * (1 - m), 2 * m * (1 - m)
    return {
        "rows": int(m.size),
        "violations_low": int((lam < lo - 1e-12).sum()),
        "violations_high": int((lam > hi + 1e-9).sum()),
        "max_ratio_to_upper": float((lam / hi).max()),
        "min_ratio_to_lower": float((lam / lo).min()),
        "max_p": m,
        "spectral_norm": lam,
        "sizes": np.asarray(n_all),
    }


def concentration(*, m_values=(0.9, 0.99, 0.999), sizes=(2, 8, 64)) -> list:
    """The two ends of the bound, constructed rather than sampled.

    Fixing `m` and moving the losing mass from one runner-up to an even spread
    walks the spectral norm from the upper bound to the lower one, which is why
    both bounds are worth stating: neither is slack.
    """
    np = _np()
    out = []
    for m in m_values:
        for n in sizes:
            for mode in ("runner-up", "spread"):
                if n == 2 and mode == "spread":
                    continue
                if mode == "runner-up":
                    p = np.zeros(n)
                    p[0], p[1] = m, 1.0 - m
                else:
                    p = np.full(n, (1.0 - m) / (n - 1))
                    p[0] = m
                lam = float(np.linalg.eigvalsh(jacobian(p)).max())
                out.append({"max_p": float(m), "n": int(n), "shape": mode,
                            "spectral_norm": lam,
                            "over_lower": lam / (m * (1 - m)),
                            "over_upper": lam / (2 * m * (1 - m))})
    return out


def _logit_gradient(p, g):
    """`J^T g` for every row at once, without building any Jacobian.

    `J` is symmetric, so this is `J g`, and `(diag(p) - p p^T) g` is
    `p * (g - <p, g>)` -- one elementwise multiply and one reduction, which is
    also exactly what the softmax backward kernel does.
    """
    return p * (g - (p * g).sum(-1, keepdim=True))


def attention_gain(*, count: int = 6, size: int = 8, seed: int = 0) -> dict:
    """Per-row: how confident the softmax is, and how much gradient it passes.

    `gain` is `||J g|| / ||g||` for the gradient the real backward pass
    delivers, so it is bounded above by the spectral norm and generally well
    below it -- `g` is not the top eigenvector. Row 0 is excluded throughout: a
    softmax over one element passes nothing, exactly, and episode 1 counted it
    already.

    `row_sum` is the null-space check. `J 1 = 0`, so every row's logit gradient
    sums to zero; the number reported is that sum relative to the row's L1
    norm, which is the honest way to ask a float32 question.
    """
    np = _np()
    torch = _torch()

    from standarderror.llm import tiny

    model = tiny.load()["model"]
    m_all, gain_all, layer_all, head_all, prev_all = [], [], [], [], []
    rel_sums = []
    for x, y in tiny.batches(count=int(count), size=int(size), seed=int(seed)):
        _, loss, atts, _ = model(x, y, want_attn=True)
        for a in atts:
            a.retain_grad()
        model.zero_grad(set_to_none=True)
        loss.backward()
        for layer, a in enumerate(atts):
            p, g = a.detach(), a.grad.detach()
            gl = _logit_gradient(p, g)
            B, H, T, _ = p.shape
            tri = torch.tril(torch.ones(T, T, dtype=torch.bool))
            s = (gl * tri).sum(-1).abs()
            l1 = (gl.abs() * tri).sum(-1)
            rel_sums.append((s[l1 > 0] / l1[l1 > 0]).numpy())
            for t in range(1, T):
                pv, gv, lv = p[:, :, t, :t + 1], g[:, :, t, :t + 1], \
                    gl[:, :, t, :t + 1]
                den = gv.norm(dim=-1)
                ok = den > 0
                m_all.append(pv.max(-1).values[ok].numpy())
                gain_all.append((lv.norm(dim=-1)[ok] / den[ok]).numpy())
                prev_all.append(
                    (pv.argmax(-1) == t - 1)[ok].numpy())
                layer_all.append(np.full(int(ok.sum()), layer))
                head_all.append(torch.arange(H).view(1, H)
                                .expand(B, H)[ok].numpy())
        model.zero_grad(set_to_none=True)
    m = np.concatenate(m_all)
    gain = np.concatenate(gain_all)
    layer = np.concatenate(layer_all)
    head = np.concatenate(head_all)
    prev = np.concatenate(prev_all)
    rel = np.concatenate(rel_sums)
    hi = 2 * m * (1 - m)
    keep = (m > 0.05) & (m < 0.995) & (gain > 0)
    slope, intercept = np.polyfit(np.log(hi[keep]), np.log(gain[keep]), 1)
    heads = []
    for L in range(int(layer.max()) + 1):
        for h in range(int(head.max()) + 1):
            s = (layer == L) & (head == h)
            heads.append({
                "layer": L, "head": h, "rows": int(s.sum()),
                "median_max_p": float(np.median(m[s])),
                "share_above_09": float((m[s] > 0.9).mean()),
                "median_gain": float(np.median(gain[s])),
                "previous_token_share": float(prev[s].mean()),
            })
    return {
        "rows": int(m.size),
        "max_p": m, "gain": gain, "layer": layer, "head": head,
        "median_max_p": float(np.median(m)),
        "share_above_09": float((m > 0.9).mean()),
        "median_gain": float(np.median(gain)),
        "violations_of_bound": int((gain > hi + 1e-6).sum()),
        "capacity_used": float(np.median(gain[keep] / hi[keep])),
        "log_slope": float(slope),
        "log_constant": float(np.exp(intercept)),
        "row_sum_median": float(np.median(rel)),
        "row_sum_max": float(rel.max()),
        "heads": heads,
    }


def trace_identity(*, count: int = 3, size: int = 4, seed: int = 0) -> dict:
    """`tr J = 1 - sum p^2` on the model's own rows, checked in float.

    Also reports the Shannon entropy of the same rows, because that is the
    quantity usually reached for when someone says "the attention is diffuse",
    and it is not the one the Jacobian's trace equals.
    """
    np = _np()
    torch = _torch()

    from standarderror.llm import tiny

    model = tiny.load()["model"]
    err, tr, coll, ent = [], [], [], []
    with torch.no_grad():
        for x, _ in tiny.batches(count=int(count), size=int(size),
                                 seed=int(seed)):
            _, _, atts, _ = model(x, want_attn=True)
            for a in atts:
                B, H, T, _ = a.shape
                for t in (1, 7, 31, T - 1):
                    for b in range(B):
                        for h in range(H):
                            p = a[b, h, t, :t + 1].double().numpy()
                            j = jacobian(p)
                            tr.append(float(np.trace(j)))
                            coll.append(float(1.0 - (p ** 2).sum()))
                            err.append(abs(tr[-1] - coll[-1]))
                            q = p[p > 0]
                            ent.append(float(-(q * np.log(q)).sum()))
    return {"rows": len(tr), "max_abs_error": float(max(err)),
            "trace": np.asarray(tr), "collision": np.asarray(coll),
            "entropy": np.asarray(ent)}


def _patched_forward(model, idx, targets, patch):
    """`model.forward`, re-implemented so one head's attention can be replaced.

    Written out rather than hooked because the substitution has to happen
    between the softmax and the `att @ v`, which no hook point exposes. It is
    checked against the model's own forward in `tests/test_saturation.py`, so a
    divergence is a test failure rather than a silently wrong ablation.
    """
    import math
    torch = _torch()
    F = torch.nn.functional
    from standarderror.llm import tiny

    B, T = idx.shape
    x = model.tok(idx) + model.pos(torch.arange(T, device=idx.device))
    for layer, blk in enumerate(model.blocks):
        h = blk.ln1(x)
        attn = blk.attn
        q, k, v = attn.qkv(h).split(tiny.WIDTH, dim=2)
        hd = tiny.WIDTH // tiny.HEADS
        q = q.view(B, T, tiny.HEADS, hd).transpose(1, 2)
        k = k.view(B, T, tiny.HEADS, hd).transpose(1, 2)
        v = v.view(B, T, tiny.HEADS, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(attn.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if patch is not None:
            att = patch(layer, att, T)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, tiny.WIDTH)
        x = x + attn.proj(y)
        x = x + blk.mlp(blk.ln2(x))
    logits = model.head(model.lnf(x))
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1))
    return logits, loss


def zero_head(layer: int, head: int):
    """Replace one head's attention with all zeros: it contributes nothing."""
    def patch(L, att, T):
        if L != layer:
            return att
        att = att.clone()
        att[:, head] = 0.0
        return att
    return patch


def shift_by_one(layer: int, head: int):
    """Replace one head's attention with the exact previous-token permutation.

    Row 0 attends to itself because it has nothing behind it -- which is forced
    by the mask anyway, and is the structural row episode 1 counted.
    """
    def patch(L, att, T):
        torch = _torch()
        if L != layer:
            return att
        hard = torch.zeros(T, T, dtype=att.dtype)
        hard[0, 0] = 1.0
        for t in range(1, T):
            hard[t, t - 1] = 1.0
        att = att.clone()
        att[:, head] = hard
        return att
    return patch


def ablations(*, layer: int = 0, head: int = 1, control: int = 0,
              draws: int = 10, count: int = 10, size: int = 16) -> dict:
    """What the saturated head is worth, and what a constant would cost.

    Three interventions, each paired against the unmodified model on the same
    held-out text, because the draw-to-draw spread of the loss is larger than
    two of the three effects and an unpaired comparison would hide them.
    """
    np = _np()
    torch = _torch()

    from standarderror.llm import tiny

    model = tiny.load()["model"]

    def loss_of(patch, seed):
        total, n = 0.0, 0
        with torch.no_grad():
            for x, y in tiny.batches(count=int(count), size=int(size),
                                     seed=seed):
                _, loss = _patched_forward(model, x, y, patch)
                total += float(loss)
                n += 1
        return total / n

    interventions = {
        "zeroed": zero_head(layer, head),
        "shift_by_one": shift_by_one(layer, head),
        "control_zeroed": zero_head(layer, control),
    }
    seeds = list(range(1, int(draws) + 1))
    base = [loss_of(None, s) for s in seeds]
    out = {"baseline": float(np.mean(base)),
           "baseline_sd": float(np.std(base, ddof=1)),
           "layer": int(layer), "head": int(head), "control": int(control)}
    for name, patch in interventions.items():
        deltas = [loss_of(patch, s) - b for s, b in zip(seeds, base)]
        out[name] = {"delta": float(np.mean(deltas)),
                     "sd": float(np.std(deltas, ddof=1)),
                     "min": float(min(deltas)), "max": float(max(deltas))}
    return out


def trajectory(*, steps: int = 1200, every: int = 40, lr: float = 3e-3,
               batch: int = 32, seed: int = 0, probe: int = 8) -> dict:
    """Per-head confidence and routing gradient, logged across a fresh run.

    The static numbers elsewhere in this module are measured on the committed
    checkpoint. This trains a new model instead, because the question -- when
    does a head commit, and what happens to its gradient afterwards -- is about
    the path and not the endpoint. Same architecture, same seed, fewer steps
    than the committed run, so it reaches a higher loss; the shape is the point.

    Costs a couple of minutes. `probe` is a fixed batch of held-out text, so
    every snapshot measures the same rows and the trace is not a random walk
    over the corpus.
    """
    np = _np()
    torch = _torch()

    from standarderror.llm import tiny

    torch.manual_seed(int(seed))
    text = tiny.corpus()
    chars = sorted(set(text))
    model = tiny.build(len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                            weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(lr), total_steps=int(steps))
    px, py = next(iter(tiny.batches("val", count=1, size=int(probe),
                                    seed=99)))

    def snapshot():
        _, loss, atts, _ = model(px, py, want_attn=True)
        for a in atts:
            a.retain_grad()
        model.zero_grad(set_to_none=True)
        loss.backward()
        rows = []
        for layer, a in enumerate(atts):
            p, g = a.detach(), a.grad.detach()
            gl = _logit_gradient(p, g)
            B, H, T, _ = p.shape
            for h in range(H):
                ms, gs, pv = [], [], []
                for t in range(1, T):
                    q = p[:, h, t, :t + 1]
                    gv = g[:, h, t, :t + 1]
                    lv = gl[:, h, t, :t + 1]
                    den = gv.norm(dim=-1)
                    ok = den > 0
                    ms.append(q.max(-1).values.numpy())
                    gs.append((lv.norm(dim=-1)[ok] / den[ok]).numpy())
                    pv.append((q.argmax(-1) == t - 1).float().numpy())
                rows.append({
                    "layer": layer, "head": h,
                    "median_max_p": float(np.median(np.concatenate(ms))),
                    "median_gain": float(np.median(np.concatenate(gs))),
                    "previous_token_share":
                        float(np.mean(np.concatenate(pv))),
                })
        out = float(loss.detach())
        model.zero_grad(set_to_none=True)
        return out, rows

    history = []
    for i, (x, y) in enumerate(tiny.batches("train", count=int(steps),
                                            size=int(batch), seed=int(seed))):
        if i % int(every) == 0:
            loss, rows = snapshot()
            history.append({"step": i, "loss": loss, "heads": rows})
        _, loss, _, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    loss, rows = snapshot()
    history.append({"step": int(steps), "loss": loss, "heads": rows})
    return {"steps": int(steps), "every": int(every), "history": history}


def series(history, layer: int, head: int, key: str):
    """One head's one quantity across a `trajectory` history, as a list."""
    out = []
    for snap in history:
        for row in snap["heads"]:
            if row["layer"] == layer and row["head"] == head:
                out.append(row[key])
    return out


def slowdown(m: float, reference: float = 0.5) -> float:
    """How many times slower routing changes at confidence `m` than at 1/2.

    The point of the bound: the softmax's gradient capacity is `2m(1-m)`, which
    peaks at `m = 1/2` and falls away on both sides. A head at `m = 0.98` needs
    about thirteen times as many steps to make the same change to where it
    looks; at the largest probability episode 1 found in this model, 0.9999, it
    needs about two and a half thousand.
    """
    cap = 2.0 * m * (1.0 - m)
    ref = 2.0 * reference * (1.0 - reference)
    return ref / cap
