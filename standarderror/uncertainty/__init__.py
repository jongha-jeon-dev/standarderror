"""The guarantee you are actually getting.

Every claim about uncertainty in machine learning is a guarantee with a
quantifier in it, and the quantifier is where the trouble is. Each module here
is one guarantee and the gap between what it promises and what it delivers:

* `coverage` -- conformal prediction. The marginal guarantee is exact,
  distribution-free and finite-sample, and it says nothing whatsoever about any
  subgroup, including the one you care about.
* `calibration` -- the estimators. Expected calibration error is biased upward
  for a perfectly calibrated model, by an amount that depends on your bin
  count and your sample size rather than on your model; and temperature
  scaling provably cannot change any prediction.
"""

from standarderror.uncertainty import calibration, coverage

__all__ = ["calibration", "coverage"]
