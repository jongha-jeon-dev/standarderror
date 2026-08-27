"""How much of a series' short-run movement is measurement noise, and did it grow.

The thing this module exists to make checkable
----------------------------------------------
A statistical agency that publishes a monthly rate from a survey is publishing a
signal plus sampling error. When the survey's effective sample shrinks — through
non-response, or through a static sample against a growing population — the
sampling error grows, and the published series gets noisier without anything
happening in the economy. Agencies say so in their own documents. The claim is
usually made from the survey design side, in units of standard error.

It is also checkable from the *outside*, from the published series alone, and that
is what this module is for. The second difference of a series annihilates any
linear trend exactly, leaves a single spike where the trend kinks, and is
otherwise dominated by whatever noise the series carries. So a robust scale of
the second difference, computed in rolling windows, tracks the noise level through
time without needing the trend to be modelled at all.

Three things about it are load-bearing and easy to get wrong.

**The constant depends on the noise's autocorrelation.** For a second difference
of noise with autocovariances gamma_k, Var = 6*gamma_0 - 8*gamma_1 + 2*gamma_2.
Under independence that is 6*sigma^2 and `SECOND_DIFF_FACTOR` is right. Under an
AR(1) with parameter rho it is (6 - 8*rho + 2*rho^2)*sigma^2, which at rho = 0.5
is less than half of it. A rotating-panel survey has exactly that kind of
correlation, so the *level* this returns is a lower bound on sigma unless the
correlation is corrected for. The *ratio* between two eras is what survives,
because a constant rho cancels — which is why `scale_ratio` exists and is the
function to quote.

**Real volatility is not noise.** A recession moves the unemployment rate by more
in a month than sampling error ever does, and a mean-based scale would read that
as noise. Every estimator here is robust by default (median absolute deviation),
and a window containing a genuine break gives a scale barely different from one
that does not — which is asserted in the tests rather than hoped for.

**Published rounding is part of the noise, and it breaks the obvious robust
estimator.** A rate published to one decimal carries a quantisation error of width
0.1, contributing only 0.1/sqrt(12) = 0.0289 to its standard deviation — about 2%
of the variance when sigma is 0.16, so it destroys almost no information. What it
does destroy is the *median* absolute deviation, because the second difference of a
series on a 0.1 lattice is itself on a 0.1 lattice, and its MAD can only land on a
lattice point. On the US unemployment rate that gives a scale estimate quantised to
0.030 steps: every era comes back as 0.061 or 0.121, and a genuine 1.7-fold change
is invisible between two adjacent rungs.

The first version of this module used MAD and produced exactly that: a rolling
scale that was flat at one of four values for eighty years. The information was
never missing — a mean-based estimator has full resolution — it was thrown away by
taking a median of lattice values. `trimmed_scale` is the replacement: winsorise
the deviations at a quantile, then take a root-mean-square, which is continuous
because it averages. It keeps the robustness that a recession requires and the
resolution that a 0.1 lattice would otherwise cost.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SECOND_DIFF_FACTOR", "MAD_TO_SIGMA", "robust_scale", "trimmed_scale",
           "winsor_constant", "lattice_resolution",
           "second_difference_scale", "rolling_noise_scale", "scale_ratio",
           "mad_scale", "ar1_factor", "rounding_floor", "detectable_change",
           "CPS_MONTHLY_OVERLAP"]

#: Var(second difference of iid noise) = 6 * sigma^2.
SECOND_DIFF_FACTOR = 6.0

#: A normal distribution's MAD is 0.6745 sigma, so sigma = MAD / 0.6745.
MAD_TO_SIGMA = 1.4826


def _phi(z: float) -> float:
    return float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi))


def _Phi(z: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _z_quantile(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def winsor_constant(trim: float) -> float:
    """sqrt(E[W^2]) for W a standard normal winsorised at its |.| quantile.

    Closed form rather than simulated, so the estimator is deterministic and its
    consistency constant can be checked by hand:

        E[W^2] = (2*Phi(c) - 1) - 2*c*phi(c) + 2*c^2*(1 - Phi(c)),
        c = the (1 - trim/2) quantile of the standard normal.

    At trim = 0 this is 1 and `trimmed_scale` reduces to the standard deviation.
    """
    t = float(trim)
    if not 0.0 <= t < 1.0:
        raise ValueError("trim is a share and must lie in [0, 1)")
    if t == 0.0:
        return 1.0
    c = _z_quantile(1.0 - t / 2.0)
    e2 = (2.0 * _Phi(c) - 1.0) - 2.0 * c * _phi(c) + 2.0 * c * c * (1.0 - _Phi(c))
    return float(np.sqrt(e2))


def trimmed_scale(x, *, trim: float = 0.10) -> float:
    """Scale of `x`, robust to outliers and not quantised by a coarse lattice.

    Deviations from the median are winsorised at their `trim` two-sided quantile
    and then squared and averaged, which keeps a recession from setting the scale
    while leaving the estimate continuous in the data. The alternative — a median
    absolute deviation — is more robust and useless here, for the reason in the
    module docstring.
    """
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    d = a - np.median(a)
    t = float(trim)
    if t > 0:
        c = float(np.quantile(np.abs(d), 1.0 - t))
        d = np.clip(d, -c, c)
    return float(np.sqrt(np.mean(d * d)) / winsor_constant(t))


def robust_scale(x, *, robust: bool = True, trim: float = 0.10) -> float:
    """Scale of `x`: winsorised root-mean-square by default, sd if `robust=False`.

    The non-robust branch exists to be compared against, not used: on a series
    that contains 2008 and 2020 the standard deviation of the second difference
    is a measurement of those two events.
    """
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    if not robust:
        return float(a.std(ddof=1))
    return trimmed_scale(a, trim=trim)


def mad_scale(x) -> float:
    """The median-absolute-deviation scale, kept only so the failure it caused
    can be reproduced: on a series published to one decimal it returns a value
    from a lattice of about 0.03 and cannot see a change of less than a rung."""
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(a - np.median(a))))


def lattice_resolution(step: float = 0.1,
                       factor: float = 6.0) -> float:
    """Smallest change in sigma a MAD-based estimate can resolve on a lattice.

    One rung of the published step, carried through the MAD constant and the
    second-difference factor. Quoted because it is the number that decides whether
    the published series can answer the question at all.
    """
    return float(MAD_TO_SIGMA * float(step) / np.sqrt(float(factor)))


def ar1_factor(rho: float) -> float:
    """Var(second difference) / sigma^2 for AR(1) noise with parameter `rho`.

    6 - 8*rho + 2*rho^2. Quoted rather than hidden because it is the single
    number that turns this module's output from a ratio into a level, and the
    difference is large: at rho = 0.5 it is 2.5, not 6.
    """
    r = float(rho)
    if not -1.0 < r < 1.0:
        raise ValueError("an AR(1) parameter must lie strictly inside (-1, 1)")
    return 6.0 - 8.0 * r + 2.0 * r * r


def second_difference_scale(x, *, robust: bool = True,
                            factor: float = SECOND_DIFF_FACTOR) -> float:
    """Noise standard deviation implied by the second difference of `x`.

    Pass `factor=ar1_factor(rho)` when the noise is known to be autocorrelated.
    The default assumes independence and therefore *understates* sigma whenever
    the noise is positively autocorrelated.
    """
    d2 = np.diff(np.asarray(x, float), n=2)
    return robust_scale(d2, robust=robust) / np.sqrt(float(factor))


def rolling_noise_scale(x, *, window: int, step: int = 1,
                        robust: bool = True,
                        factor: float = SECOND_DIFF_FACTOR) -> dict:
    """`second_difference_scale` in rolling windows over the second difference.

    Returns the window centres as indices into the original series, so a caller
    holding dates can label them without recomputing the offset. The second
    difference is two observations shorter and lags by two, and getting that
    alignment wrong shifts every conclusion by two months — so it is done once,
    here.
    """
    a = np.asarray(x, float)
    d2 = np.diff(a, n=2)
    w, s = int(window), int(step)
    if w < 8:
        raise ValueError("a window under eight points gives a scale estimate "
                         "whose own error is larger than the change being looked "
                         "for")
    if w > d2.size:
        raise ValueError(f"window {w} exceeds the {d2.size} second differences "
                         f"available")
    starts = np.arange(0, d2.size - w + 1, s)
    sigma = np.array([robust_scale(d2[i:i + w], robust=robust)
                      / np.sqrt(float(factor)) for i in starts])
    # The window covers d2[i:i+w], which is built from a[i:i+w+2]; its centre in
    # the original series is therefore i + (w + 1) / 2.
    centres = starts + (w + 1) / 2.0
    return {"centre": centres, "sigma": sigma, "window": w,
            "start": starts, "n": int(d2.size)}


def scale_ratio(x, *, early: tuple[int, int], late: tuple[int, int],
                robust: bool = True, reps: int = 2000, level: float = 0.90,
                rng=None) -> dict:
    """Noise scale in one window over another, with a bootstrap interval.

    This is the quantity to report. It is invariant to the autocorrelation
    correction — a constant `factor` cancels — and it is what an agency's claim
    about its own survey translates into: "what used to take one month now takes
    two" is a claim that this ratio is about sqrt(2).

    Both windows are given as half-open index ranges into the original series.
    """
    a = np.asarray(x, float)
    d2 = np.diff(a, n=2)

    def slice_of(win):
        lo, hi = int(win[0]), int(win[1])
        # a[lo:hi] contributes second differences d2[lo:hi-2].
        seg = d2[max(lo, 0):max(hi - 2, 0)]
        if seg.size < 8:
            raise ValueError(f"window {win} leaves only {seg.size} second "
                             f"differences")
        return seg

    e, l = slice_of(early), slice_of(late)
    point = robust_scale(l, robust=robust) / robust_scale(e, robust=robust)
    rng = rng or np.random.default_rng(0)
    draws = np.empty(int(reps))
    for b in range(int(reps)):
        # Resampled independently within each window: the question is about the
        # two scales, and pairing them would impose a relationship that the data
        # is being asked about.
        ee = e[rng.integers(0, e.size, e.size)]
        ll = l[rng.integers(0, l.size, l.size)]
        draws[b] = (robust_scale(ll, robust=robust)
                    / robust_scale(ee, robust=robust))
    lo_q = (1.0 - float(level)) / 2.0
    return {"ratio": float(point),
            "lo": float(np.quantile(draws, lo_q)),
            "hi": float(np.quantile(draws, 1.0 - lo_q)),
            "sigma_early": second_difference_scale(a[early[0]:early[1]],
                                                   robust=robust),
            "sigma_late": second_difference_scale(a[late[0]:late[1]],
                                                  robust=robust),
            "n_early": int(e.size), "n_late": int(l.size),
            "reps": int(reps), "level": float(level)}


def rounding_floor(step: float = 0.1) -> float:
    """Standard deviation of the error introduced by publishing to `step`.

    A uniform error on an interval of width `step` has standard deviation
    step / sqrt(12). It does not shrink with the sample, so it is the one part of
    a published rate's noise that no amount of money can buy away.
    """
    return float(step) / np.sqrt(12.0)


def detectable_change(sigma: float, *, alpha: float = 0.05,
                      overlap: float = 0.0) -> float:
    """Smallest one-period change in the level a two-sided test can call.

    Two correlation effects act here and they pull in opposite directions, so
    only one of them lives in this function and the other deliberately does not.

    `overlap` is the correlation between the two levels being differenced. A
    rotating-panel survey reuses most of its sample from one month to the next —
    the CPS keeps three quarters of it — and a difference of two positively
    correlated estimates is *less* noisy than a difference of two independent
    ones: Var = 2*sigma^2*(1 - overlap). Leaving it at zero therefore overstates
    the noise in a month-on-month change, which is the direction that flatters a
    "the series is too noisy" argument. It is an argument, so it should not get
    the flattering default by accident.

    The other effect — that accumulating several months of evidence averages the
    noise down by less than sqrt(m) when the noise is autocorrelated — is *not*
    handled here. Its closed form needs an assumed correlation structure, and the
    series itself carries the real one; `standarderror.ts.bend` resamples the fitted
    model's residual in moving blocks and gets it empirically. Use that for
    anything multi-period. This function is the single-period cross-check that
    the simulation has to agree with.
    """
    from math import erf, sqrt

    def z_two_sided(a):
        lo, hi = 0.0, 10.0
        target = 1.0 - a / 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    if not -1.0 < float(overlap) < 1.0:
        raise ValueError("overlap is a correlation and must lie in (-1, 1)")
    se = float(sigma) * np.sqrt(2.0 * (1.0 - float(overlap)))
    return float(z_two_sided(alpha) * se)


#: The CPS keeps three quarters of its households from one month to the next, so
#: adjacent monthly estimates share most of their sampling error. Quoted here
#: because the number changes a month-on-month standard error by a factor of two
#: and is easy to leave out.
CPS_MONTHLY_OVERLAP = 0.75


#: The household overlap and the *error* correlation are different numbers and
#: get confused constantly. Three quarters of CPS households carry over from one
#: month to the next, but the sampling errors of the two monthly rates are less
#: correlated than that: households change labour-force state, rotation groups
#: enter and leave, and the composite estimator mixes months. Published
#: variance work on the CPS puts the lag-1 error correlation for the
#: unemployment rate in the neighbourhood of 0.3 to 0.4 rather than at 0.75, so
#: anything that depends on it should be reported across a range.
CPS_ERROR_CORRELATION = (0.30, 0.40)


def implied_detectable(sigma_second_diff: float, *, rho: float = 0.35,
                       alpha: float = 0.10) -> float:
    """Detectable one-month change implied by a series' own second differences.

    This is the bridge between what a published series *does* and what an agency
    *says* about its own precision, and it exists so the two can be put in the
    same units.

    `rho` is the lag-1 autocorrelation of the noise and enters twice, in opposite
    directions. A positive rho means the second difference understates the level
    noise, so `sigma` goes *up* — and it also means a month-on-month difference
    of two positively correlated estimates is less noisy, so the detectable
    change comes *down*. The two effects very nearly cancel, which is the reason
    a result stated this way survives not knowing rho: sweep it from 0 to 0.75
    and the answer moves by a few percent.

    `alpha` is 0.10 because agencies quote 90 percent intervals.
    """
    #: 6 - 8r + 2r^2 = 2(1-r)(3-r) stays positive for every valid correlation,
    #: so there is no degenerate value to catch — but it goes to zero as r goes
    #: to one, and the amplification sqrt(6/factor) is 5.4x already at r = 0.95.
    #: Past that the second difference carries almost nothing about the level
    #: noise and the answer is mostly the assumed r, so the domain stops here
    #: rather than returning a number the input cannot support.
    if not -1.0 < float(rho) < 0.95:
        raise ValueError(
            f"rho must lie in (-1, 0.95), got {rho}; at higher autocorrelation "
            f"the second difference does not identify the level noise")
    factor = ar1_factor(float(rho))
    sigma = float(sigma_second_diff) * np.sqrt(SECOND_DIFF_FACTOR / factor)
    return detectable_change(sigma, alpha=float(alpha), overlap=float(rho))


def rescale_for_rate(value: float, *, stated_at: float, actual: float) -> float:
    """Rescale a precision figure quoted at one rate to another rate.

    The sampling standard error of a proportion goes as sqrt(p(1-p)), so a
    confidence interval published "at an unemployment rate of around 6.0
    percent" is not the interval that applies at 4.2 percent. The correction is
    small — about 16 percent between those two rates — and it is the difference
    between an agency's two figures agreeing and appearing to contradict each
    other.

    Rates are given in percent, as they are published.
    """
    for name, p in (("stated_at", stated_at), ("actual", actual)):
        if not 0.0 < float(p) < 100.0:
            raise ValueError(f"{name} must be a percentage in (0, 100), got {p}")
    a, b = float(actual) / 100.0, float(stated_at) / 100.0
    return float(value) * np.sqrt(a * (1.0 - a)) / np.sqrt(b * (1.0 - b))
