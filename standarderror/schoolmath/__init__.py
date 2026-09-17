"""School arithmetic as a probe of a small transformer.

`curriculum` generates the problems and the structural hold-outs;
`model` trains, loads and grades; `arithmetic` is the measurement layer the
episodes call.
"""

from . import curriculum, model  # noqa: F401

__all__ = ["curriculum", "model"]
