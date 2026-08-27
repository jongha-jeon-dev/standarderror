"""Time-series machinery whose null distributions are simulated, not quoted.

`nonstationary` implements the Dickey-Fuller and Engle-Granger tests by building
their null distributions from scratch. That is slower than reading MacKinnon's
table, and it is the point: a simulated critical value can be checked against the
published one, so the check is an external test rather than a restatement.

`panelpairs` assembles the every-pair correlation question from a country-year
panel, keeping the three pair types (unrelated / same country / same indicator)
separate so the strongest of them cannot carry a headline the others do not
support.

`detect` asks the prior question: can this series settle the claim being made
about it at all? On a persistent, cyclical series the answer is usually no at the
magnitudes under discussion, and the module measures how far off it is.
"""

from . import detect, nonstationary, panelpairs  # noqa: F401

__all__ = ["detect", "nonstationary", "panelpairs"]
