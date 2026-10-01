r"""What calibration error measures, and what it measures instead.

**Expected calibration error is an estimator, and a biased one.** Bin the
predictions by confidence, compare each bin's mean confidence with its
accuracy, average the gaps weighted by bin occupancy. For a perfectly
calibrated model the true value of that quantity is zero, and the estimate is
not: each bin's accuracy is a binomial average over finitely many points, so
the gap it reports is dominated by sampling noise, and noise has no sign once
you take an absolute value.

Measured on models that are calibrated **by construction** -- draw `p` from a
Dirichlet over ten classes, then draw the label from `p` -- averaged over
twenty draws, the bias is large enough to matter:

    n = 200,    15 bins   ->  ECE = 0.085
    n = 1,000,  15 bins   ->  ECE = 0.035
    n = 20,000, 15 bins   ->  ECE = 0.008

An earlier version of this docstring, and the syllabus, quoted 0.120, 0.037
and 0.005: those were *single draws*, and 0.120 at n = 200 is what seed 0
alone produces. The estimator's bias is itself noisy, which is part of the
point. It scales as `sqrt(bins / n)` -- the fitted log-log slope is 0.503
against a binomial prediction of one half -- and it depends on the confidence
profile as well, not only on bins and n: the same n and bins over 65 classes
give a third of the ten-class bias. So no table of floors is portable, and
the comparison a measured ECE needs is against a calibrated model with *its
own* confidences: `null_test`.

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


# ------------------------------------------------- episode 2: the floor

def l2_error(confidence, correct, *, bins: int = 15,
             debiased: bool = True) -> float:
    """Root binned squared calibration error, optionally with the bias removed.

    The plug-in estimate squares each bin's gap between accuracy and
    confidence, and a bin's observed accuracy carries binomial variance
    `a(1 - a) / (k - 1)` around its true value -- which the square turns into
    a positive bias. `debiased=True` subtracts that variance bin by bin, after
    Kumar, Liang and Ma (2019). What is left is an estimate of the *true*
    squared error, and it can come out negative when the true error is near
    zero and the noise overshoots; the return value is a signed square root so
    that "below the noise" is visible rather than clipped to an
    indistinguishable zero.

    Unbiased is not low-variance: on 1,000 points a debiased estimate still
    moves from subset to subset, and `stability` reports by how much.
    """
    np = _np()
    c = np.asarray(confidence, float)
    a = np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    edges[0] = -np.inf
    total, n = 0.0, len(c)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi)
        k = int(m.sum())
        if k < 2:
            continue
        acc, conf = a[m].mean(), c[m].mean()
        gap = (acc - conf) ** 2
        if debiased:
            gap -= acc * (1.0 - acc) / (k - 1)
        total += k / n * gap
    return float(np.sign(total) * math.sqrt(abs(total)))


def resample_labels(p, rng):
    """Labels drawn from the model's own probabilities.

    Under these labels the model is perfectly calibrated *by construction*,
    with exactly its own confidence profile -- which is the null a measured
    calibration error should be compared against. Consistency resampling,
    after Bröcker and Smith (2007).
    """
    np = _np()
    u = rng.random((len(p), 1))
    return (np.asarray(p).cumsum(1) < u).sum(1).clip(0, p.shape[1] - 1)


def null_test(p, y, *, bins: int = 15, draws: int = 200, seed: int = 0,
              adaptive: bool = False) -> dict:
    """ECE beside the ECE a calibrated model with the same confidences gets.

    `floor_share` is the null mean over the measured value: the share of the
    reported number that a perfectly calibrated model would also have
    reported. `p_value` is the share of null draws at least as large.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    conf, correct = _conf_correct(p, y)
    measured = ece(conf, correct, bins=bins, adaptive=adaptive)
    null = np.array([ece(conf, (p.argmax(1) == resample_labels(p, rng))
                         .astype(float), bins=bins, adaptive=adaptive)
                     for _ in range(int(draws))])
    return {"ece": measured, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "floor_share": float(null.mean() / measured) if measured else
            float("nan"),
            "p_value": float((null >= measured).mean()), "n": len(y),
            "bins": int(bins)}


def detection_rate(p, y, *, n: int, bins: int = 15, subsets: int = 60,
                   draws: int = 100, alpha: float = 0.05,
                   seed: int = 0) -> float:
    """Share of random size-`n` subsets in which the null test rejects.

    Only meaningful for a model known to be miscalibrated on the full data --
    then it is the test's power at `n`, which is the number that says whether
    a calibration claim made on a benchmark of that size could have been
    anything other than noise.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    hits = 0
    for s in range(int(subsets)):
        idx = rng.permutation(len(y))[:int(n)]
        t = null_test(p[idx], y[idx], bins=bins, draws=draws,
                      seed=seed + 7919 * (s + 1))
        hits += t["p_value"] < alpha
    return hits / int(subsets)


def temperature(p, t: float):
    """Probabilities at temperature `t`, from probabilities at temperature 1.

    `log p` differs from the logits by a per-row constant, which a softmax
    ignores, so the original logits are not needed.
    """
    np = _np()
    lg = np.log(np.clip(np.asarray(p, float), 1e-12, None)) / float(t)
    lg -= lg.max(1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


def nll(p, y) -> float:
    np = _np()
    p = np.asarray(p)
    return float(-np.log(np.clip(p[np.arange(len(y)), np.asarray(y)],
                                 1e-12, None)).mean())


def brier(p, y) -> float:
    np = _np()
    p = np.asarray(p, float)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), np.asarray(y)] = 1.0
    return float(((p - onehot) ** 2).sum(1).mean())


def ranking(p_a, p_b, y, *, n: int = 1000, bins=(5, 15, 50),
            subsets: int = 200, seed: int = 0) -> dict:
    """How often each metric, on a size-`n` subset, prefers model `a`.

    Pair it with the full-data answer: when the two models have identical
    accuracy -- as two temperatures of one model always do -- the full-data
    NLL difference is purely a calibration difference, so it settles which is
    better calibrated, and each metric's subset rate is its chance of
    agreeing with that.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    counts = {f"ECE, {b} bins": 0 for b in bins}
    counts.update({f"debiased L2, {b} bins": 0 for b in bins})
    counts.update({"NLL": 0, "Brier": 0})
    for _ in range(int(subsets)):
        i = rng.permutation(len(y))[:int(n)]
        ca, cb_ = _conf_correct(p_a[i], y[i]), _conf_correct(p_b[i], y[i])
        for b in bins:
            counts[f"ECE, {b} bins"] += ece(*ca, bins=b) < ece(*cb_, bins=b)
            counts[f"debiased L2, {b} bins"] += (
                abs(l2_error(*ca, bins=b)) < abs(l2_error(*cb_, bins=b)))
        counts["NLL"] += nll(p_a[i], y[i]) < nll(p_b[i], y[i])
        counts["Brier"] += brier(p_a[i], y[i]) < brier(p_b[i], y[i])
    full = {"nll_a": nll(p_a, y), "nll_b": nll(p_b, y),
            "accuracy_a": float((p_a.argmax(1) == y).mean()),
            "accuracy_b": float((p_b.argmax(1) == y).mean())}
    return {"prefers_a": {k: v / int(subsets) for k, v in counts.items()},
            "full": full, "n": int(n), "subsets": int(subsets)}


def stability(p, y, *, n: int = 1000, bins=(5, 15, 50), subsets: int = 30,
              seed: int = 0) -> dict:
    """Plug-in ECE, plug-in L2 and debiased L2 over size-`n` subsets.

    The L2 aggregates are taken in *squared* units and rooted afterwards.
    Averaging the per-subset roots instead -- which the first version of this
    function did -- is biased low by Jensen's inequality, because a square
    root is concave, and the debiased estimate then appears to shrink with
    the bin count when it is the variance that grows. `negative` is the share
    of subsets whose debiased estimate came out below zero: the noise made
    visible, on a model that is in fact miscalibrated.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    idx = [rng.permutation(len(y))[:int(n)] for _ in range(int(subsets))]

    def root(v):
        return float(np.sign(v) * math.sqrt(abs(v)))

    out = {}
    for b in bins:
        e = [ece(*_conf_correct(p[i], y[i]), bins=b) for i in idx]
        pl = [l2_error(*_conf_correct(p[i], y[i]), bins=b, debiased=False)
              for i in idx]
        d = [l2_error(*_conf_correct(p[i], y[i]), bins=b) for i in idx]
        sq = [np.sign(v) * v * v for v in d]
        out[b] = {"ece": float(np.mean(e)), "ece_sd": float(np.std(e)),
                  "l2": root(np.mean(np.square(pl))),
                  "l2_sd": float(np.std(pl)),
                  "debiased": root(np.mean(sq)),
                  "debiased_mean_of_roots": float(np.mean(d)),
                  "debiased_sd": float(np.std(d)),
                  "negative": float(np.mean(np.array(sq) < 0))}
    return out
