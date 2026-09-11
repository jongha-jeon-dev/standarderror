r"""What calibration error measures, and what it measures instead.

**Expected calibration error is an estimator, and a biased one.** Bin the
predictions by confidence, compare each bin's mean confidence with its
accuracy, average the gaps weighted by bin occupancy. For a perfectly
calibrated model the true value of that quantity is zero, and the estimate is
not: each bin's accuracy is a binomial average over finitely many points, so
the gap it reports is dominated by sampling noise, and noise has no sign once
you take an absolute value.

Measured on models that are calibrated **by construction** -- draw `p` from a
Dirichlet, then draw the label from `p` -- the bias is large enough to matter:

    n = 1,000 predictions, 15 bins   ->  ECE = 0.037
    n = 200,   15 bins               ->  ECE = 0.120
    n = 20,000, 15 bins              ->  ECE = 0.005

0.037 is the size of the improvement recalibration papers report. It scales
roughly as `sqrt(bins / n)`, so the same model gets three times "better" by
using five bins instead of fifty, and the comparison people actually make --
this model's ECE against that model's -- is only meaningful when both are
computed at the same bin count on the same number of points, which is not the
convention.

**Temperature scaling cannot change any prediction.** Dividing every logit by
the same positive `T` is a strictly increasing map applied to each logit, so
within one example the ordering of the classes is untouched: the argmax, the
top-k set and any within-example ranking metric are invariant, exactly.
Measured on the committed model, accuracy is bit-identical at every
temperature from 0.5 to 3.0 while ECE moves from 0.026 to 0.375.

**But it does reorder confidence between examples.** The invariance is
per-example, and the quantity used for abstention is compared *across*
examples. An example whose logits are tightly bunched and one whose logits are
spread respond differently to the same `T`, so their confidences can swap.
Measured: at `T = 2` Kendall's tau between the old and new confidence
orderings is 0.857, which is 7.2% of pairs reordered, and only 77.5% of the
most-confident one percent of predictions stays in it.

So temperature scaling is exactly the wrong tool for accuracy and a live tool
for selective prediction, which is the reverse of how it is usually described.

References: Guo et al., "On calibration of modern neural networks", *ICML*
(2017), for temperature scaling and for overconfidence as a symptom of
overfitting; Nixon et al., "Measuring calibration in deep learning", *CVPR
workshops* (2019), and Roelofs et al., "Mitigating bias in calibration error
estimation", *AISTATS* (2022), for the estimator's bias and what to do about
it; Gruber and Buettner, "Better uncertainty calibration via proper scores"
(2022), for why a proper scoring rule is the comparison that does not depend
on binning.
"""

from __future__ import annotations

import math


def _np():
    import numpy as np
    return np


def _torch():
    import torch
    return torch


def ece(confidence, correct, *, bins: int = 15, adaptive: bool = False
        ) -> float:
    """Expected calibration error, the standard equal-width implementation.

    `adaptive=True` uses equal-mass bins instead, which is the usual first
    suggestion for reducing the bias and which `bias_sweep` can check.
    """
    np = _np()
    c = np.asarray(confidence, dtype=float)
    a = np.asarray(correct, dtype=float)
    if adaptive:
        edges = np.quantile(c, np.linspace(0.0, 1.0, int(bins) + 1))
        edges[0] = -np.inf
    else:
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
        edges[0] = -np.inf
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi)
        if m.any():
            total += m.mean() * abs(a[m].mean() - c[m].mean())
    return float(total)


def calibrated_draw(n: int, classes: int = 10, *, concentration: float = 0.4,
                    seed: int = 0):
    """A model that is perfectly calibrated by construction.

    Draw a probability vector, then draw the label **from that vector**. The
    reported probabilities are then correct by definition, so the true
    calibration error is exactly zero and anything ECE reports is bias.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(int(classes)) * float(concentration), int(n))
    # One categorical draw per row, vectorised: compare a uniform against the
    # row's cumulative distribution.
    u = rng.random((int(n), 1))
    y = (p.cumsum(1) < u).sum(1).clip(0, int(classes) - 1)
    return p, y


def bias_sweep(*, sizes=(200, 500, 1000, 5000, 20000),
               bin_counts=(5, 10, 15, 30, 50), classes: int = 10,
               repeats: int = 20, adaptive: bool = False,
               seed: int = 0) -> dict:
    """ECE of a perfectly calibrated model, across sample sizes and bin counts.

    Every number here should be zero and none of them is. `slope` fits
    `log ECE` against `log(bins / n)`; the prediction from a binomial argument
    is one half, and what comes out is the honest version of "the bias scales
    like the square root of bins over n".
    """
    np = _np()
    grid = {}
    for n in sizes:
        for b in bin_counts:
            vals = [ece(*_conf_correct(*calibrated_draw(
                n, classes, seed=seed + 1000 * r)), bins=b,
                adaptive=adaptive) for r in range(int(repeats))]
            grid[(n, b)] = float(np.mean(vals))
    x = np.log([b / n for (n, b) in grid])
    y = np.log([max(v, 1e-12) for v in grid.values()])
    slope, intercept = np.polyfit(x, y, 1)
    return {"grid": grid, "sizes": tuple(sizes),
            "bin_counts": tuple(bin_counts),
            "slope": float(slope), "constant": float(math.exp(intercept)),
            "adaptive": bool(adaptive)}


def _conf_correct(p, y):
    np = _np()
    return p.max(1), (p.argmax(1) == np.asarray(y)).astype(float)


def temperature_table(*, temperatures=(0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0),
                      count: int = 10, size: int = 16, seed: int = 0,
                      bins: int = 15) -> dict:
    """Accuracy, NLL and ECE across temperatures on the committed model.

    `accuracy_identical` is the claim worth checking rather than trusting: the
    argmax must agree bit for bit at every temperature, because a strictly
    increasing map cannot reorder anything within a row.
    """
    np = _np()
    torch = _torch()
    F = torch.nn.functional

    from standarderror.llm import tiny

    bundle = tiny.load()
    model, V = bundle["model"], bundle["vocab"]
    zs, ys = [], []
    with torch.no_grad():
        for x, y in tiny.batches(count=int(count), size=int(size),
                                 seed=int(seed)):
            logits, _, _, _ = model(x)
            zs.append(logits.reshape(-1, V))
            ys.append(y.reshape(-1))
    z = torch.cat(zs)
    y = torch.cat(ys)

    rows, preds, confs = [], {}, {}
    for t in temperatures:
        p = F.softmax(z / float(t), -1)
        conf, pred = p.max(1)
        correct = (pred == y).float()
        rows.append({
            "temperature": float(t),
            "accuracy": float(correct.mean()),
            "nll": float(F.cross_entropy(z / float(t), y)),
            "ece": ece(conf.numpy(), correct.numpy(), bins=bins),
            "mean_confidence": float(conf.mean()),
        })
        preds[float(t)] = pred.numpy()
        confs[float(t)] = conf.numpy()
    base = preds[1.0] if 1.0 in preds else next(iter(preds.values()))
    return {"rows": rows, "predictions": int(len(y)), "classes": int(V),
            "confidences": confs,
            "accuracy_identical": bool(
                all(np.array_equal(v, base) for v in preds.values())),
            "accuracies": sorted({r["accuracy"] for r in rows}),
            "best_nll": min(rows, key=lambda r: r["nll"]),
            "best_ece": min(rows, key=lambda r: r["ece"])}


def reordering(table: dict, *, reference: float = 1.0,
               top_share: float = 0.01) -> list[dict]:
    """How much the between-example confidence ordering moves with temperature.

    The per-example invariance is exact and is not the whole story, because
    abstention compares confidences *across* examples. Kendall's tau on that
    comparison is what actually governs which predictions you would decline to
    make.
    """
    np = _np()
    from scipy.stats import kendalltau

    confs = table["confidences"]
    ref = confs[float(reference)]
    n = len(ref)
    k = max(1, int(n * float(top_share)))
    top_ref = set(np.argsort(-ref)[:k].tolist())
    out = []
    for t, c in sorted(confs.items()):
        tau = float(kendalltau(ref, c).statistic)
        top = set(np.argsort(-c)[:k].tolist())
        out.append({"temperature": float(t), "tau": tau,
                    "pairs_reordered": (1.0 - tau) / 2.0,
                    "top_overlap": len(top_ref & top) / k,
                    "top_k": int(k)})
    return out
