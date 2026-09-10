"""The derivative you are actually computing.

Backpropagation is the chain rule, and the chain rule has hypotheses. Each
module here is one hypothesis and the place a transformer violates it:

* `kinks` -- differentiability. What autodiff returns where the derivative does
  not exist, why that makes it a function of the *expression* rather than of
  the function, and how little of it turns out to matter in a real model.
* `saturation` -- a non-vanishing derivative. The softmax Jacobian is
  `diag(p) - p p^T`, its norm is pinned within a factor of two by the largest
  probability alone, and one head of this model has committed hard enough that
  its routing gradient is gone.
"""

from standarderror.calculus import kinks, saturation

__all__ = ["kinks", "saturation"]
