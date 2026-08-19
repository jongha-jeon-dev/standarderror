"""Reading someone else's reported effect: what it implies, and what it is worth.

These are the arithmetic steps between "the paper reports a 0.22% effect at
p < 0.001" and knowing whether that is a large finding, a usable one, or both. None
of it needs the paper's data — only the reported effect, its p-value, and a
volatility for the thing being measured.

Three quantities, in the order you want them:

* `implied_n` — the smallest sample consistent with the reported effect and
  p-value. A free consistency check on any published result: if the implied sample
  is larger than the study could have had, something is wrong; if it is far smaller,
  the study was overpowered and the effect could have been found more cheaply.
* `per_observation_sharpe` — the effect divided by the noise it sits in. The unit
  in which a "small" effect and a "big" one are comparable across studies.
* `effective_independent` and `annualised_sharpe` — what the effect is worth once
  you try to hold many of them at once, which is where correlated residuals decide
  the answer rather than the effect size.

The last one contains the only really counterintuitive step, so it is stated here
rather than buried: with pairwise residual correlation `rho`, holding `n` positions
gives you `n / (1 + (n-1) rho)` independent ones, which **converges to 1/rho** as
`n` grows. Not slowly, either — it is within 10% of the ceiling by `n = 10/rho`.
Past that point adding positions buys nothing at all, and no amount of breadth
substitutes for a decorrelated residual.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["annualised_sharpe", "breakeven_cost", "effective_independent",
           "implied_n", "per_observation_sharpe", "z_for_p"]


def z_for_p(p_value: float, *, two_sided: bool = True) -> float:
    """The z-statistic a reported p-value corresponds to.

    A p-value quoted as an upper bound ("p < 0.001") gives a *lower* bound on |z|,
    and therefore a lower bound on everything downstream. Callers should treat the
    results as bounds in that case, and the experiments that use this say so.
    """
    if not 0.0 < p_value < 1.0:
        raise ValueError("p-value must lie strictly in (0, 1)")
    return float(norm.isf(p_value / 2.0 if two_sided else p_value))


def per_observation_sharpe(effect: float, noise: float) -> float:
    """Effect size in units of the noise a single observation carries.

    Sign is preserved: a negative effect measured against a positive noise is a
    negative Sharpe, and collapsing that to an absolute value early is how a
    reversal gets reported as a profit.
    """
    if noise <= 0:
        raise ValueError("noise must be positive")
    return float(effect) / float(noise)


def implied_n(effect: float, noise: float, p_value: float, *,
              two_sided: bool = True) -> float:
    """Smallest sample consistent with this effect at this p-value.

    From `t = (effect / noise) * sqrt(n)`, so `n = (z * noise / effect)^2`. Exact
    under the normal approximation the p-value already assumes; for the sample sizes
    this is used on, the t-versus-normal distinction is far below the precision of a
    p-value rounded to one significant figure.
    """
    s = per_observation_sharpe(effect, noise)
    if s == 0:
        raise ValueError("a zero effect implies no finite sample size")
    return float((z_for_p(p_value, two_sided=two_sided) / abs(s)) ** 2)


def effective_independent(n_positions: int, rho: float) -> float:
    """Independent positions equivalent to `n` correlated ones.

    `n / (1 + (n-1) rho)` — the equicorrelation result, and the same identity that
    turns 503 index constituents into a handful of independent bets. What matters
    here is the limit: as `n` grows this converges to **1/rho**, so a strategy's
    breadth is capped by its residual correlation and not by how many names it can
    find. At `rho = 0` it is `n`, and the ceiling is infinite.
    """
    if n_positions < 1:
        raise ValueError("need at least one position")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    return float(n_positions / (1.0 + (n_positions - 1) * rho))


def annualised_sharpe(per_obs: float, n_positions: int, rho: float, *,
                      periods_per_year: float) -> float:
    """Annualised Sharpe from a per-observation edge held across `n` positions.

    Breadth in, breadth out: `per_obs * sqrt(effective n) * sqrt(periods per year)`.
    The Fundamental Law of Active Management with the correlation term left in rather
    than assumed away, which is where it usually goes wrong — the law is normally
    quoted as `IC * sqrt(breadth)` and breadth is normally taken to be the position
    count.
    """
    if periods_per_year <= 0:
        raise ValueError("periods per year must be positive")
    eff = effective_independent(n_positions, rho)
    return float(per_obs * np.sqrt(eff) * np.sqrt(periods_per_year))


def breakeven_cost(effect: float) -> float:
    """Round-trip cost at which a gross edge is exactly consumed.

    It is the gross effect, which is the point: this function exists so the
    statement gets made explicitly rather than left as an exercise. An edge of 22
    basis points is dead at 22 basis points of round-trip cost, whatever the
    Sharpe ratio computed before costs happened to be.
    """
    return abs(float(effect))
