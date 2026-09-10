"""The derivative you are actually computing.

Backpropagation is the chain rule, and the chain rule has hypotheses. Each
module here is one hypothesis and the place a transformer violates it:

* `kinks` -- differentiability. What autodiff returns where the derivative does
  not exist, why that makes it a function of the *expression* rather than of
  the function, and how little of it turns out to matter in a real model.
"""

from standarderror.calculus import kinks

__all__ = ["kinks"]
