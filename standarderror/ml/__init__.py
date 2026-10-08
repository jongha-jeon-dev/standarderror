"""Machine learning, measured where the textbook sentence stops being true.

The thirty-episode series *Machine Learning, Taught Through What Breaks*. Each
module is the measurement layer for one arc:

`evaluation` -- what a score means: test-set error bars, what
cross-validation estimates, learning curves.
`leakage` -- which preprocessing steps leak, and by how much.

Data is either bundled with scikit-learn (the UCI handwritten digits and
Wisconsin diagnostic breast cancer sets, both CC BY 4.0) or simulated with a
known answer, so every number can be rebuilt offline.
"""

from . import curves, evaluation, imbalance, leakage  # noqa: F401

__all__ = ["curves", "evaluation", "imbalance", "leakage"]
