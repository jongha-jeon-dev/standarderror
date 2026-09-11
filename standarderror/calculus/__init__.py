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
* `normalisation` -- a Jacobian of full rank. LayerNorm's has rank exactly
  d - 2, the two missing directions are the invariances it was built to have,
  and the residual stream hides the whole thing everywhere except the last
  norm, where it becomes an exact invariance of the network.
* `sampling` -- a derivative existing at all. A sampled token is a piecewise
  constant function of its logits, so the gradient people use is the gradient
  of a different function, and on a problem small enough to enumerate you can
  see exactly how different.
* `embeddings` -- a local neighbourhood. The table is 65 near-orthogonal
  points on a sphere, so a descent step leaves you nearest to where you
  started and the token you eventually reach is not the best one.
"""

from standarderror.calculus import (
    embeddings,
    kinks,
    normalisation,
    sampling,
    saturation,
)

__all__ = ["embeddings", "kinks", "normalisation", "sampling", "saturation"]
