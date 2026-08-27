"""Physiological accounting: what a measured metabolic rate has to obey.

`heat` implements the Scholander heat-balance model for a small mammal and the
indirect-calorimetry normalisation problem. Both exist to check published claims of
the form "energy expenditure rose X% and body temperature did not move", which is a
two-equation constraint — energy balance and heat balance — and therefore has a
computable answer rather than a rhetorical one.
"""

from . import heat

__all__ = ["heat"]
