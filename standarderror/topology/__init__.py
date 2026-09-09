"""Persistent homology as it behaves on a point cloud you actually have.

The series this supports has one object: **the filtration you imposed**, as
against the shape of the data. A filtration is a choice of metric and a choice
of scale, and both are usually inherited from normalisation steps upstream
rather than decided.

The thesis: **persistent homology computes a property of your metric and your
scale, and in high dimensions the barcode loses its dynamic range long before
the method loses its power.**

Two facts that recur, both measured:

* the machinery is smaller than its reputation. 0-dimensional persistent
  homology of a Rips filtration *is* the single-linkage dendrogram -- 59 deaths
  against 59 merge heights, maximum difference exactly `0.0` -- and for a graph
  the first Betti number is `E - V + b0`, which is Euler's formula.
* and the summary is stable while the number you read off it is not. Jittering
  a cloud by `eps = 0.5` moves the points by a Hausdorff distance of 1.39 and
  the barcode by a bottleneck distance of 0.44, inside the stability bound. At
  `eps = 1.0`, over forty draws, the cluster count read off the largest gap is
  2 in twenty-two of them, 3 in twelve, and 4 or 5 in six -- with every barcode
  inside a bottleneck of 1.96.
"""

from __future__ import annotations

from . import filtration

__all__ = ["filtration"]
