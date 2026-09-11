r"""Conformal prediction's guarantee, and the quantifier inside it.

**The guarantee is real and it is exact.** Split conformal takes a
non-conformity score, computes its `ceil((n+1)(1-alpha))`-th smallest value on
a calibration set, and returns every label scoring below that. The result
covers the truth with probability at least `1 - alpha`, for any model, any
score, any distribution, at any sample size, on the single assumption that the
calibration and test points are exchangeable. Measured on the committed model
at `alpha = 0.1`: coverage 0.896 against a guarantee of 0.9001, and over 200
random calibration splits a mean of 0.8999.

That is as good as a distribution-free guarantee gets, and it is a statement
about an average over the test distribution. **It is not a statement about any
subgroup**, and the subgroup people care about is the one where the model is
unsure.

Measured, splitting the test set into fifths by the model's own confidence:

    least confident fifth   coverage 0.823   mean set size 11.63
    ...                              0.896                 6.46
    ...                              0.900                 4.64
    ...                              0.913                 2.99
    most confident fifth    coverage 0.948   mean set size  1.28

So a 90% guarantee delivers 82% where the model is least sure and 95% where it
is most sure. The shortfall is not a flaw in the method -- the method promised
an average and delivered one -- and it runs in the direction that matters,
because the cases you would escalate are exactly the cases the guarantee is
quietly failing on.

**And the set sizes are the other half of the story.** Mean 5.40 labels out of
65, which is 8.3% of the vocabulary, median 5, and 14.9% of predictions get a
singleton. But the range across confidence bands is nine-fold: 1.28 labels
where the model is already sure and 11.63 where it is not. A prediction set is
informative exactly where you did not need it.

**One measurable violation of the assumption.** Over 200 random calibration
splits the realised coverage has a standard deviation of 0.0040 against the
0.0027 a Beta argument predicts for this calibration size -- 1.5 times too
variable. Character-level predictions drawn from overlapping contexts are not
exchangeable at the level the split pretends they are, and the guarantee
notices.

References: Vovk, Gammerman and Shafer, *Algorithmic Learning in a Random
World* (2005), for conformal prediction; Angelopoulos and Bates, "A gentle
introduction to conformal prediction and distribution-free uncertainty
quantification" (2021), for split conformal as used here and for the
finite-sample statement; Romano, Sesia and Candes, "Classification with valid
and adaptive coverage", *NeurIPS* (2020), for adaptive sets and the
conditional-coverage trade; Barber et al., "The limits of distribution-free
conditional predictive inference", *Information and Inference* (2021), for why
exact conditional coverage is impossible without assumptions.
"""

from __future__ import annotations

import math


def _np():
    import numpy as np
    return np


def predictions(*, count: int = 24, size: int = 16, seed: int = 1) -> dict:
    """The committed model's probabilities and labels on held-out text.

    One row per predicted character. `sequence` records which sequence each
    row came from, because the exchangeability question later needs to know.
    """
    np = _np()
    import torch

    from standarderror.llm import tiny

    bundle = tiny.load()
    model, V = bundle["model"], bundle["vocab"]
    ps, ys, seqs = [], [], []
    with torch.no_grad():
        for b, (x, y) in enumerate(tiny.batches(count=int(count),
                                                size=int(size),
                                                seed=int(seed))):
            logits, _, _, _ = model(x)
            ps.append(torch.softmax(logits, -1).reshape(-1, V).numpy())
            ys.append(y.reshape(-1).numpy())
            n_seq, n_tok = x.shape
            seqs.append(np.repeat(np.arange(n_seq) + b * n_seq, n_tok))
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    return {"p": p, "y": y, "sequence": np.concatenate(seqs),
            "classes": int(V), "rows": int(len(y)),
            "accuracy": float((p.argmax(1) == y).mean()),
            "confidence": p.max(1)}


def split_conformal(pred: dict, *, alpha: float = 0.1, seed: int = 0,
                    cal_share: float = 0.5) -> dict:
    """Split conformal with the score `1 - p[true label]`.

    `guarantee` is `ceil((n+1)(1-alpha)) / (n+1)`, the exact finite-sample
    level the construction achieves -- slightly above `1 - alpha`, which is
    why the realised coverage should come out just under the guarantee and not
    under `1 - alpha`.
    """
    np = _np()
    p, y = pred["p"], pred["y"]
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(n * float(cal_share))
    cal, test = perm[:cut], perm[cut:]
    scores = 1.0 - p[cal, y[cal]]
    k = math.ceil((len(cal) + 1) * (1.0 - float(alpha)))
    qhat = float(np.sort(scores)[min(k, len(cal)) - 1])
    sets = p[test] >= 1.0 - qhat
    covered = sets[np.arange(len(test)), y[test]]
    size = sets.sum(1)
    return {
        "alpha": float(alpha), "qhat": qhat,
        "n_cal": int(len(cal)), "n_test": int(len(test)),
        "guarantee": k / (len(cal) + 1),
        "coverage": float(covered.mean()),
        "mean_size": float(size.mean()),
        "median_size": float(np.median(size)),
        "max_size": int(size.max()),
        "size_share": float(size.mean() / pred["classes"]),
        "singletons": float((size == 1).mean()),
        "empty": float((size == 0).mean()),
        "sets": sets, "size": size, "covered": covered, "test": test,
    }


def conditional(pred: dict, fit: dict, *, bands: int = 5) -> list[dict]:
    """Coverage and set size within equal-mass bands of the model's confidence.

    Conditioning on the model's own confidence is the conditioning anyone
    would actually do, and it is the one the marginal guarantee says nothing
    about.
    """
    np = _np()
    test = fit["test"]
    conf = pred["confidence"][test]
    edges = np.quantile(conf, np.linspace(0.0, 1.0, int(bands) + 1))
    edges[0] = -np.inf
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        out.append({"low": float(max(lo, conf.min())), "high": float(hi),
                    "n": int(m.sum()),
                    "coverage": float(fit["covered"][m].mean()),
                    "mean_size": float(fit["size"][m].mean())})
    return out


def split_variability(pred: dict, *, alpha: float = 0.1, draws: int = 200,
                      cal_share: float = 0.5) -> dict:
    """How much the realised coverage moves as the calibration split changes.

    `beta_sd` is `sqrt(alpha (1 - alpha) / (n_cal + 2))`, the spread the
    exchangeable theory predicts. The ratio of measured to predicted is the
    cheapest test of the assumption there is, and it needs no shifted data.
    """
    np = _np()
    cov = np.empty(int(draws))
    for t in range(int(draws)):
        fit = split_conformal(pred, alpha=alpha, seed=t, cal_share=cal_share)
        cov[t] = fit["coverage"]
    n_cal = int(len(pred["y"]) * float(cal_share))
    beta_sd = math.sqrt(alpha * (1 - alpha) / (n_cal + 2))
    return {"draws": int(draws), "mean": float(cov.mean()),
            "sd": float(cov.std(ddof=1)), "min": float(cov.min()),
            "max": float(cov.max()), "beta_sd": beta_sd,
            "sd_ratio": float(cov.std(ddof=1) / beta_sd)}


def grouped_split(pred: dict, *, alpha: float = 0.1, draws: int = 200
                  ) -> dict:
    """The same measurement, splitting by sequence instead of by row.

    If the extra variability comes from rows inside one sequence being
    dependent, then splitting whole sequences into calibration and test should
    change it. This is the control that decides whether the 1.5x is about
    exchangeability or about something else.
    """
    np = _np()
    p, y, seq = pred["p"], pred["y"], pred["sequence"]
    ids = np.unique(seq)
    cov = np.empty(int(draws))
    for t in range(int(draws)):
        rng = np.random.default_rng(10_000 + t)
        order = rng.permutation(ids)
        cal_ids = set(order[: len(ids) // 2].tolist())
        cal = np.isin(seq, list(cal_ids))
        scores = 1.0 - p[cal, y[cal]]
        k = math.ceil((cal.sum() + 1) * (1.0 - float(alpha)))
        qhat = float(np.sort(scores)[min(k, int(cal.sum())) - 1])
        sets = p[~cal] >= 1.0 - qhat
        cov[t] = sets[np.arange((~cal).sum()), y[~cal]].mean()
    n_cal = int(len(y) // 2)
    beta_sd = math.sqrt(alpha * (1 - alpha) / (n_cal + 2))
    return {"draws": int(draws), "sequences": int(len(ids)),
            "mean": float(cov.mean()), "sd": float(cov.std(ddof=1)),
            "beta_sd": beta_sd,
            "sd_ratio": float(cov.std(ddof=1) / beta_sd)}
