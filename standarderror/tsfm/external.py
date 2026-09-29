r"""Published time-series foundation models, behind the same interface as ours.

Everything in `evaluate` and `compress` takes either our own checkpoint or one
of these, unchanged: a forecaster is `f(contexts, horizon, season) -> (N,
horizon, Q)` at the nine levels in `model.QUANTILES`, and a compressor takes a
torch module. So the day the weights are reachable, the same tables are one
call away and nothing about the measurement changes between our model and
theirs.

The weights are not reachable today. They are hosted on HuggingFace, which
this project's build environment cannot reach, and nothing here tries another
route to them. `load` fails with a message that says so rather than with a
stack trace from inside a download library. The *code* is on PyPI and is
installed; that is what lets the adapters be tested now, against randomly
initialised models built from the same classes -- see `tiny_bolt`.

Parameter counts are the published ones, recorded so a table can say what it
is comparing: Chronos-Bolt tiny 9M, mini 21M, small 48M, base 205M
(Ansari et al., 2024, and the model cards), Chronos-2 about 120M.
"""

from __future__ import annotations

import copy

import numpy as np

from standarderror.tsfm import model as tm

MODELS = {
    "chronos-bolt-tiny": "amazon/chronos-bolt-tiny",
    "chronos-bolt-mini": "amazon/chronos-bolt-mini",
    "chronos-bolt-small": "amazon/chronos-bolt-small",
    "chronos-bolt-base": "amazon/chronos-bolt-base",
    "chronos-2": "amazon/chronos-2",
}


class Unreachable(RuntimeError):
    """The weights could not be fetched from where they are published."""


def load(name: str):
    """A published model as a bundle: `{"pipeline", "model", "name"}`.

    `model` is the torch module the compression operators act on; `pipeline`
    owns preprocessing and is rebuilt around a compressed copy by `with_model`.
    """
    import torch
    from chronos import BaseChronosPipeline
    repo = MODELS[name]
    try:
        pipe = BaseChronosPipeline.from_pretrained(
            repo, device_map="cpu", torch_dtype=torch.float32)
    except Exception as e:  # network policy, missing cache, anything
        raise Unreachable(
            f"could not fetch {repo} from HuggingFace ({type(e).__name__}). "
            f"This environment cannot reach huggingface.co; the weights are "
            f"not fetched by any other route.") from e
    return {"pipeline": pipe, "model": pipe.model, "name": name}


def with_model(bundle, module):
    """The same pipeline around a different (e.g. compressed) module."""
    pipe = copy.copy(bundle["pipeline"])
    pipe.model = module
    return {**bundle, "pipeline": pipe, "model": module}


def forecaster(bundle, *, chunk: int = 64):
    """Chronos-family quantile forecasts at our nine levels, sorted per step."""
    import torch

    def f(contexts, horizon, season):
        out = []
        for i in range(0, len(contexts), chunk):
            part = [torch.tensor(np.asarray(c, float), dtype=torch.float32)
                    for c in contexts[i:i + chunk]]
            with torch.no_grad():
                q, _ = bundle["pipeline"].predict_quantiles(
                    part, prediction_length=horizon,
                    quantile_levels=list(tm.QUANTILES))
            out.append(np.sort(q.numpy(), axis=-1))
        return np.concatenate(out)
    return f


def tiny_bolt(*, seed: int = 0):
    """A randomly initialised Chronos-Bolt of toy size, built offline.

    Not a forecaster anyone should use: it exists so that the adapter, the
    compression operators and the scoring can be tested against the real
    classes without the real weights.
    """
    import torch
    from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltPipeline
    from transformers import T5Config
    torch.manual_seed(seed)
    cfg = T5Config(d_model=64, d_ff=128, d_kv=16, num_heads=4, num_layers=2,
                   num_decoder_layers=1, dense_act_fn="relu",
                   feed_forward_proj="relu", dropout_rate=0.0,
                   decoder_start_token_id=0, pad_token_id=0)
    cfg.chronos_config = {"context_length": 256, "prediction_length": 32,
                          "input_patch_size": 16, "input_patch_stride": 16,
                          "quantiles": list(tm.QUANTILES),
                          "use_reg_token": True}
    model = ChronosBoltModelForForecasting(cfg).eval()
    pipe = ChronosBoltPipeline(model=model)
    return {"pipeline": pipe, "model": model, "name": "tiny-bolt (random)"}
