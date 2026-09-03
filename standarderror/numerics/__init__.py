"""Discretisation as it behaves in floating point, rather than as it is derived.

The series this supports has one object: **the discrete operator you actually
integrated**, as opposed to the continuous one you wrote down. Every module here
exists because some quantity a working scientist would compute -- a derivative, a
trajectory, an order of convergence -- comes out wrong in a way the textbook
derivation does not predict, and the piece of numerical analysis that explains it
is worth more than the formula it came from.

The thesis the series argues: **order of accuracy, stability and conservation are
three different properties, and your code reports at most one of them.**
"""

from __future__ import annotations

from . import differencing

__all__ = ["differencing"]
