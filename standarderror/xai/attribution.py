"""Attribution methods, with their failure modes documented rather than hidden.

The honest framing this module is built around: every method here answers a
*different* question, and the disagreements between them are the interesting
part of any XAI post.

* `permutation_importance` — how much does the model's error rise when this
  feature is destroyed? Model-centric. **Breaks under correlated features**: it
  splits credit between collinear inputs and evaluates the model off its data
  manifold. On lagged time series, where consecutive lags are nearly identical,
  this is severe. `block_permutation_importance` is the fix — permute *groups*.
* `conditional_permutation_importance` — permute within strata of a conditioning
  variable, so the perturbed data stays plausible. Slower, much better behaved
  under collinearity.
* `linear_shapley` — for a linear readout, Shapley values have a closed form:
  `phi_j = beta_j (x_j - E[x_j])`. Exact, instant, no sampling error. Since an
  ESN/NG-RC readout *is* linear in its features, this is the right tool for
  reservoir models and nobody needs KernelSHAP.
* `kernel_shapley` — sampling-based Shapley for arbitrary models when you have
  no better option. Sampling variance is reported, not swallowed.

The one thing this module will not do is let you call an attribution a cause.
Every result carries `interpretation_caveat`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import factorial

import numpy as np


@dataclass
class Attribution:
    values: np.ndarray                 # (n_features,) or (n_samples, n_features)
    feature_names: list[str]
    method: str
    baseline_score: float | None = None
    std: np.ndarray | None = None
    detail: dict = field(default_factory=dict)
    interpretation_caveat: str = (
        "Attribution is a statement about this model on this data distribution, "
        "not about causation in the world. Correlated features share credit "
        "arbitrarily.")

    def ranked(self, k: int | None = None) -> list[tuple[str, float]]:
        v = self.values
        if v.ndim > 1:
            v = np.nanmean(np.abs(v), axis=0)
        order = np.argsort(np.abs(v))[::-1]
        if k:
            order = order[:k]
        return [(self.feature_names[i], float(v[i])) for i in order]

    def to_frame(self):
        import pandas as pd
        v = self.values
        if v.ndim == 1:
            df = pd.DataFrame({"feature": self.feature_names, "value": v})
            if self.std is not None:
                df["std"] = self.std
            return df.sort_values("value", key=np.abs, ascending=False)
        return pd.DataFrame(v, columns=self.feature_names)


def permutation_importance(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
    feature_names: Sequence[str] | None = None,
    n_repeats: int = 10,
    seed: int = 0,
    greater_is_better: bool = False,
) -> Attribution:
    """Marginal permutation importance with a reported spread across repeats."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    names = list(feature_names) if feature_names is not None else \
        [f"f{i}" for i in range(X.shape[1])]
    if scorer is None:
        def scorer(a, b):
            return float(np.sqrt(np.mean((np.asarray(a).ravel()
                                          - np.asarray(b).ravel()) ** 2)))
    sign = -1.0 if greater_is_better else 1.0
    base = scorer(y, predict(X))
    rng = np.random.default_rng(seed)

    means = np.zeros(X.shape[1])
    stds = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        deltas = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            deltas[r] = sign * (scorer(y, predict(Xp)) - base)
        means[j] = deltas.mean()
        stds[j] = deltas.std()
    return Attribution(means, names, "permutation importance", base, stds,
                       {"n_repeats": n_repeats},
                       interpretation_caveat=(
                           "Marginal permutation breaks feature correlations and "
                           "evaluates the model off-manifold. With lagged inputs, "
                           "prefer block_permutation_importance or the "
                           "conditional variant."))


def block_permutation_importance(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    blocks: dict[str, Sequence[int]],
    *,
    scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_repeats: int = 10,
    seed: int = 0,
) -> Attribution:
    """Permute whole groups of columns together.

    For a model on lags of variables A and B, the useful question is almost never
    "how much does lag 3 of A matter" (it is nearly collinear with lag 2) but
    "how much does A matter at all". Group the lags per variable and ask that.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if scorer is None:
        def scorer(a, b):
            return float(np.sqrt(np.mean((np.asarray(a).ravel()
                                          - np.asarray(b).ravel()) ** 2)))
    base = scorer(y, predict(X))
    rng = np.random.default_rng(seed)
    names = list(blocks)
    means, stds = np.zeros(len(names)), np.zeros(len(names))
    for i, name in enumerate(names):
        cols = list(blocks[name])
        deltas = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, cols] = Xp[np.ix_(perm, cols)]   # same permutation per block
            deltas[r] = scorer(y, predict(Xp)) - base
        means[i], stds[i] = deltas.mean(), deltas.std()
    return Attribution(means, names, "block permutation importance", base, stds,
                       {"blocks": {k: list(v) for k, v in blocks.items()},
                        "n_repeats": n_repeats})


def conditional_permutation_importance(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    condition_on: int | Sequence[int],
    n_strata: int = 8,
    scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
    feature_names: Sequence[str] | None = None,
    n_repeats: int = 5,
    seed: int = 0,
) -> Attribution:
    """Permute each feature *within strata* of a conditioning variable.

    Keeps the perturbed sample closer to the data manifold, which is what makes
    marginal permutation unreliable. Strata are quantile bins of `condition_on`.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    names = list(feature_names) if feature_names is not None else \
        [f"f{i}" for i in range(X.shape[1])]
    if scorer is None:
        def scorer(a, b):
            return float(np.sqrt(np.mean((np.asarray(a).ravel()
                                          - np.asarray(b).ravel()) ** 2)))
    cond_cols = [condition_on] if isinstance(condition_on, int) else list(condition_on)
    key = X[:, cond_cols].mean(axis=1)
    edges = np.quantile(key, np.linspace(0, 1, n_strata + 1))
    strata = np.clip(np.searchsorted(edges[1:-1], key), 0, n_strata - 1)

    base = scorer(y, predict(X))
    rng = np.random.default_rng(seed)
    means, stds = np.zeros(X.shape[1]), np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        deltas = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp = X.copy()
            for s in range(n_strata):
                idx = np.nonzero(strata == s)[0]
                if len(idx) > 1:
                    Xp[idx, j] = Xp[rng.permutation(idx), j]
            deltas[r] = scorer(y, predict(Xp)) - base
        means[j], stds[j] = deltas.mean(), deltas.std()
    return Attribution(means, names, "conditional permutation importance", base,
                       stds, {"n_strata": n_strata, "condition_on": cond_cols,
                              "n_repeats": n_repeats})


def linear_shapley(
    coefficients: np.ndarray,
    X: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    background: np.ndarray | None = None,
) -> Attribution:
    """Exact Shapley values for a linear model: phi_j = beta_j (x_j - E[x_j]).

    Applies directly to any ESN / NG-RC readout, since those are linear in their
    features. Exact, no sampling, no approximation error — there is no reason to
    run KernelSHAP on a linear readout, and doing so only adds variance.
    """
    beta = np.asarray(coefficients, float).ravel()
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[1] != len(beta):
        raise ValueError(f"{X.shape[1]} features vs {len(beta)} coefficients")
    ref = X.mean(axis=0) if background is None else \
        np.asarray(background, float).mean(axis=0)
    phi = (X - ref) * beta
    names = list(feature_names) if feature_names is not None else \
        [f"f{i}" for i in range(X.shape[1])]
    return Attribution(phi, names, "exact linear Shapley", None, None,
                       {"reference": ref.tolist()},
                       interpretation_caveat=(
                           "Exact for the linear readout. Attribution is to "
                           "reservoir features, which are not interpretable "
                           "quantities themselves - aggregate to inputs before "
                           "claiming anything about drivers."))


def kernel_shapley(
    predict: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    n_samples: int = 512,
    seed: int = 0,
) -> Attribution:
    """Sampling-based Shapley for one instance, with the standard error reported.

    Uses permutation sampling (each random ordering gives an unbiased estimate of
    every phi_j), which is simpler and better-behaved than weighted least squares
    on subsets. Exact enumeration is used automatically when d <= 12.
    """
    x = np.asarray(x, float).ravel()
    background = np.atleast_2d(np.asarray(background, float))
    d = len(x)
    names = list(feature_names) if feature_names is not None else \
        [f"f{i}" for i in range(d)]
    ref = background.mean(axis=0)

    def f(z: np.ndarray) -> float:
        return float(np.asarray(predict(z[None, :])).ravel()[0])

    if d <= 12:
        phi = np.zeros(d)
        idx = list(range(d))
        for j in idx:
            others = [i for i in idx if i != j]
            for size in range(len(others) + 1):
                w = factorial(size) * factorial(d - size - 1) / factorial(d)
                for subset in combinations(others, size):
                    z_wo = ref.copy()
                    z_wo[list(subset)] = x[list(subset)]
                    z_w = z_wo.copy()
                    z_w[j] = x[j]
                    phi[j] += w * (f(z_w) - f(z_wo))
        return Attribution(phi, names, "exact Shapley (enumerated)", None, None,
                           {"n_subsets": 2 ** d, "reference": ref.tolist()})

    rng = np.random.default_rng(seed)
    acc = np.zeros((n_samples, d))
    for s in range(n_samples):
        order = rng.permutation(d)
        z = ref.copy()
        prev = f(z)
        for j in order:
            z[j] = x[j]
            cur = f(z)
            acc[s, j] = cur - prev
            prev = cur
    phi = acc.mean(axis=0)
    se = acc.std(axis=0, ddof=1) / np.sqrt(n_samples)
    return Attribution(phi, names, "permutation-sampled Shapley", None, se,
                       {"n_samples": n_samples, "reference": ref.tolist(),
                        "max_standard_error": float(se.max())},
                       interpretation_caveat=(
                           "Sampled estimate: compare |phi_j| against its "
                           "standard error before ranking features. Differences "
                           "inside 2 SE are not differences."))


def efficiency_check(phi: np.ndarray, f_x: float, f_ref: float,
                     tol: float = 1e-6) -> dict:
    """Shapley's efficiency axiom: sum(phi) == f(x) - f(reference).

    Run this. A silent violation means the background distribution or the
    prediction function is not what you think it is, and it is the single most
    common bug in hand-rolled SHAP code.
    """
    total = float(np.sum(phi))
    gap = total - (f_x - f_ref)
    return {"sum_phi": total, "f_x_minus_f_ref": float(f_x - f_ref),
            "gap": float(gap), "passes": bool(abs(gap) <= tol * max(1.0, abs(f_x))),
            "tolerance": tol}
