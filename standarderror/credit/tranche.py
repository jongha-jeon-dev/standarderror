"""First-loss tranche arithmetic on a single collateral asset.

Written for exp010, which prices a Korean jeonse deposit, but there is nothing
Korean in here. The object is the same wherever it turns up: a junior claim of
size `D` on one asset worth `V`, ranking behind senior debt `M`, recovering
`lambda` of the asset's value in a forced sale. The holder receives

    min(D, max(0, lambda * V_T - M))

which is a put spread on `V` struck between `M` and `M + D`. Two things follow,
and keeping them apart is the whole reason this module exists:

* The **attachment point** `(M + D) / lambda` is arithmetic. No volatility, no
  drift, no horizon, no distribution. If you know the three inputs you know
  exactly how far the asset has to fall before the junior claim is impaired, and
  you know it exactly.
* The **probability** of reaching it, and the expected loss when you do, need a
  distribution for `V_T`. Everything model-dependent lives in
  `expected_shortfall_rate`, and it is deliberately the thinner half of the file.

The expectation is taken under the *physical* measure, so what comes out is an
expected credit loss rather than an option value. Pricing the put spread
risk-neutrally would need a carry yield for the collateral and would start an
argument about the measure that changes nothing about the shape of the answer.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["attachment_point", "required_fall", "expected_shortfall_rate",
           "simulate_shortfall_rate"]


def _validate(junior: float, senior: float, recovery: float) -> None:
    if junior <= 0:
        raise ValueError("junior claim must be positive")
    if senior < 0:
        raise ValueError("senior claim cannot be negative")
    if not 0 < recovery <= 2.0:
        raise ValueError("recovery ratio must lie in (0, 2]")


def attachment_point(junior: float, recovery: float, senior: float = 0.0) -> float:
    """Collateral value, relative to today's, at which the junior claim is exactly
    covered.

    `(senior + junior) / recovery`, all three expressed in the same units as
    today's collateral value — percentages or fractions, as long as they match.
    Deliberately contains no model: this is the exact half of the calculation.

    A `recovery` above 1 is allowed and is not a mistake. Korean apartments have
    cleared above their appraised value at court auction every month since March
    2026, and refusing to represent that would force the caller to lie about the
    input.
    """
    _validate(junior, senior, recovery)
    return (senior + junior) / recovery


def required_fall(junior: float, recovery: float, senior: float = 0.0) -> float:
    """Percentage fall in the collateral that first impairs the junior claim.

    Negative means impaired already: the claims up to and including the junior one
    exceed what the collateral fetches before anything moves.
    """
    return 100.0 * (1.0 - attachment_point(junior, recovery, senior))


def expected_shortfall_rate(junior: float, recovery: float, sigma: float, *,
                            senior: float = 0.0, drift: float = 0.0,
                            term: float = 1.0) -> dict:
    """Expected loss on the junior claim, in closed form, under lognormal `V_T`.

    The payoff `min(D, max(0, lambda*V_T - M))` splits the expectation into three
    regions — total loss below `M/lambda`, partial loss up to `(M+D)/lambda`,
    nothing above — and each is a normal CDF or a truncated lognormal mean.

    Returns a dict rather than a float because the three quantities a reader wants
    (loss per year, loss over the term, probability of any impairment) come from
    the same three integrals, and computing them separately invites them to
    disagree. `junior`, `senior` and `recovery` are in the units of today's
    collateral value; `term` is in years, and the returned rate is per year.
    """
    _validate(junior, senior, recovery)
    if sigma <= 0:
        raise ValueError("sigma must be positive; take the zero-volatility case "
                         "from required_fall, where it is exact")
    if term <= 0:
        raise ValueError("term must be positive")

    k_hi = (senior + junior) / recovery
    k_lo = senior / recovery
    s = sigma * np.sqrt(term)
    mn = (drift - 0.5 * sigma ** 2) * term

    def below(k: float) -> float:
        return 0.0 if k <= 0 else float(norm.cdf((np.log(k) - mn) / s))

    def truncated_mean(a: float, b: float) -> float:
        """E[V_T · 1{a < V_T < b}] for V_T lognormal, V_0 = 1."""
        if b <= 0:
            return 0.0
        za = -np.inf if a <= 0 else (np.log(a) - mn) / s
        zb = (np.log(b) - mn) / s
        return float(np.exp(mn + 0.5 * s ** 2)
                     * (norm.cdf(zb - s) - norm.cdf(za - s)))

    p_wipeout = below(k_lo)
    p_breach = below(k_hi)
    loss = (junior * p_wipeout + (senior + junior) * (p_breach - p_wipeout)
            - recovery * truncated_mean(k_lo, k_hi))
    # Clamped because the three terms can cancel to a value around -1e-18 when the
    # tranche is far out of the money, and a negative expected loss is nonsense
    # that would propagate into a negative coverage ratio.
    loss = max(loss, 0.0)
    return {"loss_per_year_pct": 100.0 * loss / (junior * term),
            "loss_total_pct": 100.0 * loss / junior,
            "p_breach_pct": 100.0 * p_breach,
            "p_wipeout_pct": 100.0 * p_wipeout,
            "attachment": k_hi,
            "required_fall_pct": 100.0 * (1.0 - k_hi)}


def simulate_shortfall_rate(junior: float, recovery: float, sigma: float, *,
                            senior: float = 0.0, drift: float = 0.0,
                            term: float = 1.0, n_draws: int = 400_000,
                            seed: int = 0) -> dict:
    """Monte Carlo counterpart to `expected_shortfall_rate`.

    Exists to check the integral, not to replace it. If the two disagree, the
    integral is what gets fixed.
    """
    _validate(junior, senior, recovery)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng(seed)
    v = np.exp((drift - 0.5 * sigma ** 2) * term
               + sigma * np.sqrt(term) * rng.standard_normal(n_draws))
    loss = junior - np.clip(recovery * v - senior, 0.0, junior)
    return {"loss_per_year_pct": 100.0 * float(loss.mean()) / (junior * term),
            "loss_total_pct": 100.0 * float(loss.mean()) / junior,
            "p_breach_pct": 100.0 * float((loss > 1e-12).mean())}
