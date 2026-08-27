"""Scale equivariance as a test you run on a fitting procedure.

The test is one line of arithmetic. A procedure `A` mapping data to a predictor is
**scale equivariant** in the response if

    A(X, s*y)(x) / s  ==  A(X, y)(x)     for every s > 0

— multiply the response by a constant, refit, divide the predictions back, and you
must get the same function. Nothing about the problem changed: `s` is a choice of
units. Metres or centimetres, dollars or thousands of dollars.

Least squares passes, exactly, and so does least absolute deviations: their loss
functions are homogeneous in the residual, so the minimiser scales with `y` and
nothing else moves. Every *robust* loss in common use fails unless it is given a
scale, because it contains a transition point — Huber's `delta`, Tukey's `c` — and a
transition point compares a residual against a constant. Fixing the constant fixes
the units.

That failure is not visible in a loss curve and it is not visible in a single
experiment. It shows up as a procedure whose accuracy, and whose *robustness*, depend
on a rescaling of the data that no one thinks of as a modelling decision. Which makes
it worth a dedicated test:

    gap = equivariance.scale_sweep(fit_predict, X, y, scales=(0.1, 1.0, 10.0, 100.0))

`scale_sweep` refits at each scale, rescales the predictions back, and returns them
alongside a summary of how far apart they are. `equivariance_gap` reduces that to one
number: the largest relative spread of any prediction across the sweep. Zero means
equivariant; anything else is the size of the units problem.

Two cautions the tests pin down. Floating point is not scale-free — the sweep
degrades at extreme scales because a gradient-boosting library working in float32
runs out of resolution, so a sweep should be read over a range where a
known-equivariant procedure is flat, and that range should be reported. And a
procedure can be equivariant and still bad: `scale_sweep` measures dependence on
units, not accuracy, and the two are separate questions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SweepResult", "equivariance_gap", "huber_slope_for", "scale_sweep"]


@dataclass
class SweepResult:
    """Rescaled predictions from each scale, and how far apart they are."""

    scales: np.ndarray
    predictions: np.ndarray               # (n_scales, n_test), already divided by s
    reference: int = 0                    # index of the scale treated as the baseline

    @property
    def gap(self) -> float:
        """Largest relative spread of any single prediction across the sweep."""
        p = self.predictions
        span = p.max(axis=0) - p.min(axis=0)
        denom = np.maximum(np.abs(p[self.reference]), 1e-12)
        return float(np.max(span / denom))

    @property
    def rms_gap(self) -> float:
        """Root-mean-square deviation from the reference scale, relative to its RMS.

        Less alarmist than `gap`, which is a maximum over test points and can be
        driven by one prediction that happens to sit near zero — the same near-zero
        denominator that has bitten this repository before.
        """
        p = self.predictions
        ref = p[self.reference]
        scale = np.sqrt(np.mean(ref ** 2)) or 1.0
        devs = [np.sqrt(np.mean((row - ref) ** 2)) / scale
                for i, row in enumerate(p) if i != self.reference]
        return float(max(devs)) if devs else 0.0

    def describe(self) -> str:
        return (f"{len(self.scales)} scales from {self.scales.min():g} to "
                f"{self.scales.max():g}: max relative gap {self.gap:.3g}, "
                f"rms gap {self.rms_gap:.3g}")


def scale_sweep(fit_predict, X, y, X_test=None, *,
                scales=(0.1, 1.0, 10.0, 100.0), reference: float = 1.0
                ) -> SweepResult:
    """Refit `fit_predict` at each scale of `y` and rescale the predictions back.

    `fit_predict(X, y, scale)` must fit on `(X, y)` and return predictions on the
    test matrix. The `scale` argument is passed through so a procedure that wants to
    set its own scale constant from the data can do so — which is exactly the
    intervention being tested, and the reason this is a callback rather than an
    estimator interface.
    """
    scales = np.asarray(list(scales), dtype=float)
    if np.any(scales <= 0):
        raise ValueError("scales must be positive")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xt = X if X_test is None else np.asarray(X_test, dtype=float)
    rows = [np.asarray(fit_predict(X, y * s, s), dtype=float).ravel() / s
            for s in scales]
    widths = {r.size for r in rows}
    if len(widths) != 1:
        raise ValueError(f"fit_predict returned differing prediction counts {widths}")
    ref_idx = int(np.argmin(np.abs(scales - reference)))
    return SweepResult(scales, np.vstack(rows), reference=ref_idx)


def equivariance_gap(fit_predict, X, y, X_test=None, **kw) -> float:
    """`scale_sweep(...).rms_gap` — one number, zero if the procedure is equivariant."""
    return scale_sweep(fit_predict, X, y, X_test, **kw).rms_gap


def huber_slope_for(y, *, scale_estimate=None, multiple: float = 1.0) -> float:
    """A `huber_slope` in the units of this particular `y`.

    The one-line fix for the units problem, and deliberately shallow: it takes the
    scale of `y`, not of the residuals, because that is all you have before fitting.
    That makes it enough to restore equivariance and *not* enough to be the best
    choice — a scale taken from a contaminated `y` is inflated by the very points the
    loss is supposed to discount, so it sets the transition point too high.

    Getting that right needs the scale of the residuals, which needs a fit, which
    needs a scale: the circularity that two-step procedures exist to break. See
    `scale.residual_scale`.
    """
    from .scale import mad_scale
    est = mad_scale if scale_estimate is None else scale_estimate
    s = float(est(y))
    if not s > 0:
        raise ValueError("the response has zero robust scale; nothing to set from")
    return multiple * s
