"""Discretisation as it behaves in floating point, rather than as it is derived.

The series this supports has one object: **the discrete operator you actually
ran**, as opposed to the continuous one you wrote down -- and in a machine
learning system those operators are the ones you use every day rather than
anything exotic. Gradient descent *is* forward Euler on the gradient flow. A
gradient check *is* a finite difference. A diffusion sampler *is* an ODE solver.
A decode loop *is* an iterated map.

So the classical analysis is load-bearing here rather than decorative, which is
the whole reason the series is framed for machine learning instead of for
simulation. The thesis: **every one of these has a step size, and the step size
has a stability limit, an optimum, or a horizon that nobody prints.**

Two consequences that recur, both measured:

* the limits are *hard*, not gradual. Gradient descent on a quadratic with
  `lam_max = 3.135` converges at `lr = 0.638` and reaches `3.2e+31` at
  `lr = 0.70`.
* and the precision decides them. `bfloat16` has `eps = 3.9e-3` against
  float64's `2.2e-16`, which moves every scale in `differencing` by eight orders
  of magnitude and is why a gradient check in bf16 cannot see an error below
  about 2.5%.
"""

from __future__ import annotations

from . import differencing, steps

__all__ = ["differencing", "steps"]
