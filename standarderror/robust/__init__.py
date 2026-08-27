"""Robust regression tooling, and the scale question underneath it.

    from standarderror.robust import contamination, equivariance, scale

    scale.mad_scale(residuals)                       # a robust sigma
    contamination.vertical_outliers(y, frac=0.1)     # y-outliers
    contamination.leverage_points(X, y, f, distance=2.5)
    equivariance.scale_sweep(fit_predict, X, y, scales=(0.1, 1, 10))

Three pieces, and the third is the one that motivated the package.

* `scale` — robust scale estimates, because every robust loss has a scale constant
  in it and the constant has to come from somewhere.
* `contamination` — the two classical outlier types, constructed so that the
  construction is explicit and its parameters are visible. How far out a leverage
  point sits turns out to matter more than how many there are.
* `equivariance` — a test, not a model. Multiply `y` by a constant, refit, divide
  the predictions back, and see whether you get the same function. A procedure that
  fails this has a unit-dependent hyperparameter somewhere, and its behaviour
  depends on what you happened to measure in.
"""

from . import contamination, equivariance, scale

__all__ = ["contamination", "equivariance", "scale"]
