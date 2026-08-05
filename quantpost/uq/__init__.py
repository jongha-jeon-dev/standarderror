"""Uncertainty quantification and causal ground truth.

    from quantpost.uq import conformal, causal
    iv = conformal.cqr(lo_cal, hi_cal, y_cal, lo_test, hi_test, alpha=0.1)
    print(iv.summary(y_test))
    print(conformal.coverage_by_bin(iv, y_test, difficulty))

Both modules exist to make claims falsifiable: conformal prediction has a
guarantee you can check empirically, and an SCM has a causal effect you know
because you generated it.
"""

from . import causal, conformal
from .conformal import Interval, conformal_quantile

__all__ = ["conformal", "causal", "Interval", "conformal_quantile"]
