"""Uncertainty quantification and causal ground truth.

    from standarderror.uq import conformal, causal, multiplicity
    iv = conformal.cqr(lo_cal, hi_cal, y_cal, lo_test, hi_test, alpha=0.1)
    print(iv.summary(y_test))
    print(conformal.coverage_by_bin(iv, y_test, difficulty))

These modules exist to make claims falsifiable: conformal prediction has a
guarantee you can check empirically, an SCM has a causal effect you know because
you generated it, and `multiplicity` gives the score a search budget buys with no
signal present — so "my best model got 55%" can be checked against what trying
that many models was always going to produce. `evidence` does the same job from the
outside: given only a published effect, its p-value and a volatility, it recovers the
sample size the claim needs and what the effect is worth once you hold many of them.
"""

from . import causal, conformal, evidence, multiplicity
from .conformal import Interval, conformal_quantile
from .multiplicity import expected_max_accuracy, trials_to_reach

__all__ = ["conformal", "causal", "evidence", "multiplicity", "Interval",
           "conformal_quantile", "expected_max_accuracy", "trials_to_reach"]
