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

**The spread of the realised coverage, and a correction.** Over 200 random
calibration splits the realised coverage has a standard deviation of 0.0040.
An earlier version of this docstring compared that with 0.0027, the spread of
the *calibration* draw alone, and called the 1.5x ratio a violation of
exchangeability. It is not: the test half is a finite sample too, and adding
its binomial term gives 0.0038, so the row-wise excess is 1.06 -- inside the
range an i.i.d. pool of the same size produces. It has to be, because a random
row split of a fixed pool is exchangeable by construction. The real excess
appears only when whole sequences are split, 1.42x, and the within-sequence
correlation of the coverage indicator predicts it: `design_effect` gives 1.43.

References: Vovk, Gammerman and Shafer, *Algorithmic Learning in a Random
World* (2005), for conformal prediction; Angelopoulos and Bates, "A gentle
introduction to conformal prediction and distribution-free uncertainty
quantification" (2021), for split conformal as used here and for the
finite-sample statement; Romano, Sesia and Candes, "Classification with valid
and adaptive coverage", *NeurIPS* (2020), for adaptive sets and the
conditional-coverage trade; Barber et al., "The limits of distribution-free
conditional predictive inference", *Information and Inference* (2021), for why
exact conditional coverage cannot be had distribution-free -- the worst case
forces sets that carry no information, which is the reason the spread above
is a price to be chosen rather than a bug to be fixed.
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

    `excess` is the measured spread over `exchangeable_sd`, the spread the
    exchangeable theory predicts for a calibration half *and* a test half.
    `beta_sd` and `sd_ratio` keep the calibration term alone, which is the
    comparison an earlier version made and is kept so the mistake stays
    checkable: it is short by `sqrt(2)`.
    """
    np = _np()
    cov = np.empty(int(draws))
    for t in range(int(draws)):
        fit = split_conformal(pred, alpha=alpha, seed=t, cal_share=cal_share)
        cov[t] = fit["coverage"]
    n_cal = int(len(pred["y"]) * float(cal_share))
    n_test = len(pred["y"]) - n_cal
    beta_sd = math.sqrt(alpha * (1 - alpha) / (n_cal + 2))
    full = exchangeable_sd(n_cal, n_test, alpha)
    sd = float(cov.std(ddof=1))
    return {"draws": int(draws), "mean": float(cov.mean()),
            "sd": sd, "min": float(cov.min()),
            "max": float(cov.max()), "beta_sd": beta_sd,
            "sd_ratio": sd / beta_sd, "exchangeable_sd": full,
            "excess": sd / full, "coverages": cov}


def exchangeable_sd(n_cal: int, n_test: int, alpha: float = 0.1) -> float:
    """The spread of realised test coverage that exchangeability predicts.

    Two finite samples, two terms. Given the calibration set, the coverage
    the threshold delivers is Beta-distributed with variance
    `alpha (1 - alpha) / (n_cal + 2)`; the test set then estimates that
    coverage with binomial variance `alpha (1 - alpha) / n_test`. Leaving out
    the second term -- which is what `beta_sd` alone does -- understates the
    spread by `sqrt(2)` when the halves are equal.
    """
    a = float(alpha)
    return math.sqrt(a * (1 - a) / (n_cal + 2) + a * (1 - a) / n_test)


def block_split(pred: dict, block, *, alpha: float = 0.1, draws: int = 200,
                seed: int = 10_000) -> dict:
    """Split conformal with whole blocks of rows sent to one side or the other.

    `block` is an array giving each row's block, or an integer `m` meaning
    consecutive runs of `m` rows. Rows inside a block never straddle the
    calibration/test boundary, so dependence inside a block can no longer be
    averaged away by the split.
    """
    np = _np()
    p, y = pred["p"], pred["y"]
    n = len(y)
    s = 1.0 - p[np.arange(n), y]
    blocks = (np.arange(n) // int(block) if np.isscalar(block)
              else np.asarray(block))
    ids = np.unique(blocks)
    cov = np.empty(int(draws))
    for t in range(int(draws)):
        rng = np.random.default_rng(int(seed) + t)
        cal = np.isin(blocks, rng.permutation(ids)[: len(ids) // 2])
        k = math.ceil((cal.sum() + 1) * (1.0 - float(alpha)))
        qhat = float(np.sort(s[cal])[min(k, int(cal.sum())) - 1])
        cov[t] = (s[~cal] <= qhat).mean()
    n_cal = n // 2
    beta_sd = math.sqrt(alpha * (1 - alpha) / (n_cal + 2))
    full = exchangeable_sd(n_cal, n - n_cal, alpha)
    sd = float(cov.std(ddof=1))
    return {"draws": int(draws), "blocks": int(len(ids)),
            "mean": float(cov.mean()), "sd": sd, "beta_sd": beta_sd,
            "sd_ratio": sd / beta_sd, "exchangeable_sd": full,
            "excess": sd / full, "coverages": cov}


def grouped_split(pred: dict, *, alpha: float = 0.1, draws: int = 200
                  ) -> dict:
    """The same measurement, splitting by sequence instead of by row.

    If the extra variability comes from rows inside one sequence being
    dependent, then splitting whole sequences into calibration and test should
    change it. It does: the row-wise split is exchangeable by construction and
    matches the exchangeable spread, and the sequence-wise split exceeds it.
    """
    out = block_split(pred, pred["sequence"], alpha=alpha, draws=draws)
    out["sequences"] = out["blocks"]
    return out


def iid_control(pred: dict, *, alpha: float = 0.1, pools: int = 20,
                draws: int = 200, seed: int = 0) -> dict:
    """What `excess` looks like when the rows really are i.i.d.

    Each pool resamples the model's own scores with replacement, so it has the
    same size and the same score distribution and no dependence at all, then
    runs the identical row-wise split. The spread of `excess` across pools is
    the noise any single measurement of it carries.
    """
    np = _np()
    p, y = pred["p"], pred["y"]
    n = len(y)
    s = 1.0 - p[np.arange(n), y]
    n_cal = n // 2
    full = exchangeable_sd(n_cal, n - n_cal, alpha)
    k = math.ceil((n_cal + 1) * (1.0 - float(alpha)))
    rng = np.random.default_rng(int(seed))
    out = []
    for _ in range(int(pools)):
        pool = rng.choice(s, size=n, replace=True)
        cov = np.empty(int(draws))
        for t in range(int(draws)):
            perm = np.random.default_rng(t).permutation(n)
            qhat = np.sort(pool[perm[:n_cal]])[k - 1]
            cov[t] = (pool[perm[n_cal:]] <= qhat).mean()
        out.append(float(cov.std(ddof=1) / full))
    return {"excess": out, "low": float(min(out)), "high": float(max(out)),
            "mean": float(np.mean(out))}


def lag_correlation(pred: dict, *, alpha: float = 0.1) -> list[float]:
    """Correlation of the coverage indicator between rows `lag` apart inside
    one sequence, for every lag the sequence length allows.

    The indicator is `score <= q`, with `q` the pooled `1 - alpha` quantile:
    the event "this row would be covered". Rows are assumed grouped
    contiguously by sequence, which is how `predictions` returns them.
    """
    np = _np()
    p, y, seq = pred["p"], pred["y"], pred["sequence"]
    s = 1.0 - p[np.arange(len(y)), y]
    z = (s <= np.quantile(s, 1.0 - float(alpha))).astype(float)
    length = int(np.bincount(seq).max())
    Z = z.reshape(-1, length) - z.mean()
    var = Z.var()
    return [float(np.mean(Z[:, :length - lag] * Z[:, lag:]) / var)
            for lag in range(1, length)]


def design_effect(rho: list[float], m: int) -> float:
    """Variance inflation for blocks of `m` consecutive rows.

    `1 + 2 sum_{l<m} (1 - l/m) rho(l)`: the variance of a block mean over the
    variance it would have with independent rows. Its square root is the
    predicted `excess` of a block split -- predicted from the correlations,
    not fitted to the spread.
    """
    return 1.0 + 2.0 * sum((1.0 - lag / m) * rho[lag - 1]
                           for lag in range(1, int(m)))


def aps_conformal(pred: dict, *, alpha: float = 0.1, seed: int = 0,
                  cal_share: float = 0.5) -> dict:
    """Adaptive prediction sets, with the randomisation that makes them exact.

    The score is the cumulative probability of every label at least as likely
    as the true one, minus a uniform fraction of the true label's own mass.
    Without that subtraction the score is discrete and the sets overcover
    badly -- measured at 0.98 against a target of 0.90, with sets 2.2 times
    larger than they need to be -- which would make any comparison against
    `split_conformal` a comparison of two different coverage levels.

    The point of having both is that they hit the same marginal guarantee and
    distribute it very differently, so the unevenness the marginal guarantee
    permits is a property of the score rather than of conformal prediction.
    """
    np = _np()
    p, y = pred["p"], pred["y"]
    n, V = p.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(n * float(cal_share))
    cal, test = perm[:cut], perm[cut:]

    def scores(idx, u):
        order = np.argsort(-p[idx], axis=1)
        ps = np.take_along_axis(p[idx], order, 1)
        csum = ps.cumsum(1)
        rank = np.argmax(order == y[idx][:, None], axis=1)
        i = np.arange(len(idx))
        return csum[i, rank] - u * ps[i, rank]

    k = math.ceil((len(cal) + 1) * (1.0 - float(alpha)))
    qhat = float(np.sort(scores(cal, rng.random(len(cal))))[
        min(k, len(cal)) - 1])

    order = np.argsort(-p[test], axis=1)
    ps = np.take_along_axis(p[test], order, 1)
    inside = (ps.cumsum(1) - rng.random(len(test))[:, None] * ps) <= qhat
    sets = np.zeros((len(test), V), dtype=bool)
    np.put_along_axis(sets, order, inside, 1)

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
        "size_share": float(size.mean() / V),
        "singletons": float((size == 1).mean()),
        "empty": float((size == 0).mean()),
        "sets": sets, "size": size, "covered": covered, "test": test,
    }


def conditional_range(bands) -> float:
    """Spread of coverage across confidence bands -- the number to compare.

    A marginal guarantee constrains the average of these and nothing about
    their spread, so the spread is where two scores that both satisfy the
    guarantee can differ, and by a factor of seven on this model.
    """
    got = [b["coverage"] for b in bands]
    return float(max(got) - min(got))


def escalation(pred: dict, fit: dict, *, fractions=(0.05, 0.1, 0.2, 0.3, 0.5)
               ) -> list[dict]:
    """Coverage among the cases you would keep, and among those you would escalate.

    The operational form of the conditional-coverage gap. Sending the least
    confident cases to a human is the obvious policy, and it sorts the test
    set precisely so that the queue is the part the guarantee is failing on.
    """
    np = _np()
    conf = pred["confidence"][fit["test"]]
    out = []
    for f in fractions:
        thr = np.quantile(conf, float(f))
        low = conf <= thr
        if low.sum() == 0 or (~low).sum() == 0:
            continue
        out.append({"fraction": float(f),
                    "kept_coverage": float(fit["covered"][~low].mean()),
                    "escalated_coverage": float(fit["covered"][low].mean()),
                    "kept_size": float(fit["size"][~low].mean()),
                    "escalated_size": float(fit["size"][low].mean())})
    return out
