"""Aggregate statistics, and the three structurally different ways they fail.

One object per episode, and each is a summary that a newspaper prints as though
it described a person:

* `tempo` -- the period total fertility rate, which is a synthetic construct
  assembled from age groups that no woman belongs to at once. A pure change in
  the *timing* of births moves it while no cohort's completed family size
  changes at all.
* `composition` -- the median wage, which can fall while every individual's
  wage rises, and which is *more* fragile to a change in who is counted than
  the mean is.
* dispersion (episode 3) -- the Gini coefficient, which is many-to-one onto
  distributions, so two societies needing opposite policies can share one.
"""

from standarderror.aggregates import composition, tempo

__all__ = ["composition", "tempo"]
