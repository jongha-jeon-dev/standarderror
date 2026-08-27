"""Robust scale estimates: the constant every robust loss needs and rarely states.

A robust loss function is a function of `r / sigma`, not of `r`. Huber's is linear
outside `|r| > delta` and quadratic inside; Tukey's biweight is flat outside; an
S-estimator minimises a robust scale directly. In all of them the transition point
is measured in units of the residual spread, which means **it cannot be a constant**.
Fix it, and the loss stops being a loss and becomes a loss-plus-an-assumption about
your units.

That is the whole content of this module. `mad_scale` is the standard cheap answer;
`tau_scale` is the one that keeps some efficiency at the Gaussian; `residual_scale`
is the shape the answer has to take in practice — re-estimated from the residuals of
a fit, not from the marginal spread of `y`, because the marginal spread of a
contaminated `y` is inflated by exactly the points the loss is meant to discount.

The last point is worth stating plainly, because it is why a one-shot rescaling is
not enough and a two-step procedure is: an estimate of scale taken from `y` before
fitting includes the signal *and* the contamination, so it is too large; used as a
Huber transition point it puts the outliers back inside the quadratic region. You
need the scale of the residuals, which needs a fit, which needs a scale.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MAD_TO_SIGMA", "consistency_factor", "mad_scale", "residual_scale",
           "tau_scale"]

#: 1 / Phi^{-1}(3/4). Makes the MAD consistent for sigma at the Gaussian.
MAD_TO_SIGMA = 1.482602218505602


def consistency_factor() -> float:
    """`MAD_TO_SIGMA`, computed rather than quoted, so the constant is checkable."""
    from scipy.stats import norm
    return float(1.0 / norm.ppf(0.75))


def mad_scale(x, *, center: float | None = None,
              consistent: bool = True) -> float:
    """Median absolute deviation, scaled to estimate sigma at the Gaussian.

    Breakdown point 1/2: up to half the sample can be arbitrary and the estimate
    stays bounded. That is the highest possible, and it is why the MAD is the default
    starting scale for almost every robust procedure.

    The price is efficiency — about 37% at the Gaussian, meaning it needs roughly
    three times the sample of a standard deviation for the same precision. For
    setting a transition point that is a fine trade; for reporting a scale it is not,
    which is what `tau_scale` is for.

    `center=0.0` is the right choice for residuals from a fit that already has an
    intercept: re-centring them on their own median discards the information that
    they were supposed to be centred at zero.
    """
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("no finite values to take a scale from")
    c = np.median(a) if center is None else float(center)
    m = float(np.median(np.abs(a - c)))
    return m * MAD_TO_SIGMA if consistent else m


def _tau_consistency(c: float) -> float:
    """`E[min(Z^2/c^2, 1)]` for standard normal Z, by quadrature.

    Computed rather than tabulated. A tau scale without this factor is off by about
    40% at the usual tuning constant, which is the kind of error that reads as a real
    difference between methods when it is only a missing normalisation.
    """
    from scipy.integrate import quad
    from scipy.stats import norm
    f = quad(lambda z: min((z / c) ** 2, 1.0) * norm.pdf(z), -np.inf, np.inf)
    return float(f[0])


def tau_scale(x, *, center: float | None = None, c: float = 3.0) -> float:
    """Yohai-Zamar tau scale: MAD breakdown, much better Gaussian efficiency.

    `s_tau^2 = s_mad^2 * mean(rho(r / s_mad)) / b`, with the bounded
    `rho(u) = min(u^2/c^2, 1)` and `b = E[rho(Z)]` for standard normal Z, which is
    what makes the result consistent for sigma rather than merely proportional to it.

    Inherits the MAD's 50% breakdown, because the MAD is what standardises the
    residuals before `rho` bounds them, while recovering most of the efficiency the
    MAD gives up. This is the scale a tau-estimator minimises, and it is why the
    paper's tau variant is a reasonable middle option rather than a third name in a
    list.

    `c` trades breakdown against efficiency and is an argument for that reason.
    """
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("no finite values to take a scale from")
    if c <= 0:
        raise ValueError("the tuning constant must be positive")
    mid = np.median(a) if center is None else float(center)
    s = mad_scale(a, center=mid)
    if s == 0:
        return 0.0
    u = (a - mid) / s
    rho = np.minimum((u / c) ** 2, 1.0)
    return float(s * np.sqrt(rho.mean() / _tau_consistency(c)))


def residual_scale(y, prediction, *, method: str = "mad") -> float:
    """Robust scale of the residuals of a fit — the quantity a loss actually needs.

    Separate from `mad_scale` only to make the argument order impossible to get
    wrong, and to centre at zero rather than at the residual median. Residuals from
    a fit with an intercept are supposed to be centred; re-centring them hides a
    biased fit inside a smaller scale.
    """
    r = np.asarray(y, dtype=float).ravel() - np.asarray(prediction,
                                                        dtype=float).ravel()
    if method == "mad":
        return mad_scale(r, center=0.0)
    if method == "tau":
        return tau_scale(r, center=0.0)
    raise ValueError(f"unknown method {method!r}; use 'mad' or 'tau'")
