"""What a search budget buys you when there is nothing to find.

The winner's curse in its cleanest form: if you score `n_models` candidates on
`n_obs` binary outcomes and none of them has any skill, each score is a draw from
Binomial(n_obs, p) / n_obs, and the *best* score is the maximum of `n_models` such
draws. That maximum is not a discovery, it is an order statistic, and it is
computable in advance from the two counts alone.

    from quantpost.uq import multiplicity as mult
    mult.expected_max_accuracy(2000, 900)      # 0.5572 — before running anything
    mult.trials_to_reach(0.55, 900)            # 447 models

Everything here is exact arithmetic on the binomial CDF, deliberately. The normal
approximation is fine near the centre and wrong by a factor of ~1.5 in the tail
where the interesting questions live: at 900 observations it says you need 1.01e9
tries to expect a best-of-N accuracy of 60%, and the true answer is 6.56e8. A post
whose argument is "the formula predicted my winner" cannot afford a formula that is
itself off by 50%.

The independence assumption is the real limitation, not the arithmetic. Model
variants share training data and features, so the effective number of independent
tries is below the number of models fitted, and these functions therefore give an
*upper* bound on the expected best score. That direction is the useful one: if your
winner beats the bound you have something to explain, and if it sits under the
bound you cannot yet distinguish it from your search budget.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["expected_max_accuracy", "trials_to_reach", "significance_threshold",
           "normal_expected_max_accuracy"]


def _log_cdf(n_obs: int, p: float) -> np.ndarray:
    hits = np.arange(n_obs + 1)
    return hits, stats.binom.logcdf(hits, n_obs, p)


def expected_max_accuracy(n_models: int, n_obs: int, p: float = 0.5) -> float:
    """Expected best accuracy over `n_models` independent chance-level candidates.

    Exact, from `E[M] = sum_h h (F(h)^N - F(h-1)^N)` where F is the Binomial CDF.
    Computed in logs so `n_models` can be in the billions without overflow.
    """
    if n_models < 1:
        raise ValueError("n_models must be at least 1")
    hits, log_cdf = _log_cdf(n_obs, p)
    cdf_pow = np.exp(n_models * log_cdf)                 # P(max <= h)
    pmf = np.diff(np.concatenate([[0.0], cdf_pow]))      # P(max == h)
    return float((hits * pmf).sum() / n_obs)


def normal_expected_max_accuracy(n_models: int, n_obs: int,
                                 p: float = 0.5) -> float:
    """The usual shortcut, kept for contrast only — do not publish from it.

    It takes the `(1 - 1/(N+1))` quantile of a normal as the expected maximum, which
    is not the same quantity: that quantile lies *below* the expected maximum at
    every budget (0.33pp low at N=20 on 900 observations, 0.24pp low at N=2000). The
    bias runs in the flattering direction — it makes a lucky winner look less
    explainable by luck than it is — and inverting it inflates a trials table by
    ~50% in the tail. Both errors are measured in `tests/test_uq.py`.
    """
    sd = np.sqrt(p * (1.0 - p) / n_obs)
    return float(p + sd * stats.norm.ppf(1.0 - 1.0 / (n_models + 1.0)))


def trials_to_reach(accuracy: float, n_obs: int, p: float = 0.5, *,
                    cap: int = 10 ** 13) -> int | None:
    """Smallest number of candidates whose expected best score reaches `accuracy`.

    The exact inverse of `expected_max_accuracy` by doubling then bisection, so a
    table built from this and a curve built from that are the same function read in
    opposite directions. Returns None if `cap` is exceeded.
    """
    if not p < accuracy <= 1.0:
        raise ValueError(f"accuracy must lie in ({p}, 1]; got {accuracy}")
    hi = 1
    while expected_max_accuracy(hi, n_obs, p) < accuracy:
        hi *= 2
        if hi > cap:
            return None
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if expected_max_accuracy(mid, n_obs, p) >= accuracy:
            hi = mid
        else:
            lo = mid + 1
    return lo


def significance_threshold(n_obs: int, alpha: float = 0.05,
                           p: float = 0.5) -> tuple[float, float]:
    """Accuracy a *single* candidate needs to clear a one-sided `alpha` test.

    Returns `(accuracy, attained_level)`. The attained level is strictly below
    `alpha` because the binomial is discrete — reporting "significant at 5%" while
    actually testing at 4.45% is a small dishonesty, and the expected number of
    false positives in a search follows the attained level, not the nominal one.
    """
    hits = np.arange(n_obs + 1)
    sf = stats.binom.sf(hits - 1, n_obs, p)             # P(X >= h)
    h = int(np.searchsorted(-sf, -alpha))               # sf is decreasing in h
    return float(hits[h] / n_obs), float(sf[h])
