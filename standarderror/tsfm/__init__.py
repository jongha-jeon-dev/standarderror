"""A small time-series foundation model, and what compressing it costs.

`pool` is the real public series it is scored on, zero-shot; `synth` is the
synthetic bank it was pretrained on; `model` builds, loads and forecasts;
`evaluate` scores it against naive, seasonal-naive and ETS baselines;
`compress` quantises, prunes and truncates it.
"""

from . import compress, evaluate, model, pool, synth  # noqa: F401

__all__ = ["compress", "evaluate", "model", "pool", "synth"]
