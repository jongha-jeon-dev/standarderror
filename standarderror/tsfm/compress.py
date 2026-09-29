r"""Compression operators for any torch model, and what they cost in bytes.

Three families, each implemented as a *copy* of the model with the operator
applied, so the original is never touched and two compressed variants can be
scored side by side:

`quantize`
    Round-to-nearest onto a symmetric integer grid of `bits`, with one scale
    per output channel (the row of the weight matrix). This is the baseline
    every quantisation paper improves on; the improvements -- GPTQ, AWQ,
    SmoothQuant -- are all ways of choosing the grid or the rounding better,
    and a series about compression should first know what the naive version
    does before crediting anything to the clever ones.

`prune`
    Unstructured magnitude pruning: zero the smallest `sparsity` share of each
    weight matrix's entries.

`low_rank`
    Replace each weight matrix by its best rank-`r` approximation (Eckart--
    Young), with `r` a fraction of the full rank.

What is compressed: every 2-D weight -- linear layers, the attention input
projection (which PyTorch stores as a bare parameter, not a `Linear`, and
which a loop over `nn.Linear` silently skips), and the positional embedding.
What is not: biases and LayerNorm parameters, which are one-dimensional,
tiny, and kept in full precision by every practical scheme. `size_bytes`
counts both parts, and the scales, so a claimed compression ratio is the one
you would actually get on disk.
"""

from __future__ import annotations

import copy

import numpy as np


def _targets(model):
    """(name, parameter) for every 2-D weight."""
    return [(n, p) for n, p in model.named_parameters()
            if p.ndim == 2 and n.endswith("weight")]


def quantize(model, bits: int, *, per_channel: bool = True):
    """A copy with every 2-D weight rounded to a `bits`-bit symmetric grid."""
    import torch
    m = copy.deepcopy(model)
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for _, p in _targets(m):
            if per_channel:
                s = p.abs().amax(dim=1, keepdim=True) / qmax
            else:
                s = p.abs().amax() / qmax
            s = torch.where(s > 0, s, torch.ones_like(s))
            p.copy_((p / s).round().clamp(-qmax, qmax) * s)
    return m


def prune(model, sparsity: float):
    """A copy with the smallest-magnitude `sparsity` share of each matrix zeroed."""
    import torch
    m = copy.deepcopy(model)
    with torch.no_grad():
        for _, p in _targets(m):
            k = int(round(sparsity * p.numel()))
            if k <= 0:
                continue
            thresh = p.abs().flatten().kthvalue(k).values
            p.mul_((p.abs() > thresh).to(p.dtype))
    return m


def low_rank(model, keep: float):
    """A copy with each matrix replaced by its best rank-`ceil(keep * rank)`
    approximation."""
    import torch
    m = copy.deepcopy(model)
    with torch.no_grad():
        for _, p in _targets(m):
            U, S, Vh = torch.linalg.svd(p, full_matrices=False)
            r = max(1, int(np.ceil(keep * len(S))))
            p.copy_((U[:, :r] * S[:r]) @ Vh[:r])
    return m


def size_bytes(model, *, bits: int | None = None, sparsity: float = 0.0,
               keep: float | None = None) -> dict:
    """Bytes on disk under each operator, counted honestly.

    Quantised: `bits` per weight plus one fp16 scale per output channel.
    Pruned: surviving weights in fp16 plus a one-bit mask per position.
    Against fp16 dense that breaks even at 2(1 - s) + 1/8 = 2, i.e. at
    s = 6.25% -- not the "about 50%" usually quoted, which is the break-even
    for index-based formats like CSR, where every survivor carries a two- or
    four-byte column index. Storage is not speed: unstructured zeros do not
    make a dense matrix multiply any faster without sparse kernels.
    Low-rank: the two factors in fp32.
    Everything not in `_targets` stays fp32.
    """
    total_other = sum(p.numel() for n, p in model.named_parameters()
                      if not (p.ndim == 2 and n.endswith("weight")))
    fp32 = 4 * sum(p.numel() for p in model.parameters())
    tgt = _targets(model)
    if bits is not None:
        w = sum(p.numel() * bits / 8 + p.shape[0] * 2 for _, p in tgt)
    elif keep is not None:
        w = 0.0
        for _, p in tgt:
            r = max(1, int(np.ceil(keep * min(p.shape))))
            w += 4 * r * (p.shape[0] + p.shape[1])
    else:
        w = sum(2 * p.numel() * (1 - sparsity) + p.numel() / 8 for _, p in tgt)
    b = w + 4 * total_other
    return {"bytes": float(b), "fp32_bytes": float(fp32),
            "ratio": fp32 / b}


def fp16_dense_bytes(model) -> float:
    """The honest comparison for a pruned model: dense fp16, not fp32."""
    tgt = sum(p.numel() for _, p in _targets(model))
    rest = sum(p.numel() for p in model.parameters()) - tgt
    return 2.0 * tgt + 4.0 * rest


def weight_error(original, compressed) -> dict:
    """Relative Frobenius error of the compressed weights, per matrix."""
    out = {}
    orig = dict(_targets(original))
    for n, p in _targets(compressed):
        o = orig[n]
        out[n] = float((p - o).norm() / o.norm().clamp_min(1e-12))
    return out
