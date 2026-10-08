r"""Calibration measured across a training run, not only at its end.

Episode 3 found that the committed model needs a fitted temperature of only
1.10. That number belongs to one checkpoint. The question this module exists
for is whether it is a property of the architecture or of the moment training
stopped: does overconfidence arrive when validation loss stops improving?

`evaluate` takes a model and returns, on a fixed set of validation (or
training) rows, everything needed to answer that: NLL, accuracy, mean
confidence minus accuracy, ECE, and the temperature an NLL fit would choose.
The training runs that produce the checkpoints are in
`scripts/train_coverage4.py`; their per-checkpoint metrics are committed in
`data/coverage4/trajectory.json`, and each arm's final checkpoint -- plus its
lowest-validation-loss one where that differs -- is committed so the tests can
recompute the endpoints from weights rather than trust the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import standarderror as se

#: Rows evaluated at every checkpoint: 10 batches of 16 windows of 64.
COUNT, SIZE, SEED = 10, 16, 0


def _torch():
    import torch
    return torch


def data_dir() -> Path:
    return Path(se.SETTINGS.repo_root) / "data" / "coverage4"


def windows(split: str = "val", *, share: float = 1.0, count: int = COUNT,
            size: int = SIZE, seed: int = SEED):
    """Fixed windows from `split`. For the training split, `share` restricts
    them to the leading fraction an arm actually trained on -- a model trained
    on a tenth of the text has to be scored on that tenth, or its "training"
    loss is a second held-out loss."""
    torch = _torch()
    import numpy as np

    from standarderror.llm import tiny
    text = tiny.corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(0.9 * len(data))
    d = data[:cut] if split == "train" else data[cut:]
    if split == "train":
        d = d[: int(len(d) * float(share))]
    rng = np.random.default_rng(seed)
    for _ in range(int(count)):
        i = rng.integers(0, len(d) - tiny.BLOCK - 1, int(size))
        yield (torch.stack([d[j:j + tiny.BLOCK] for j in i]),
               torch.stack([d[j + 1:j + tiny.BLOCK + 1] for j in i]))


def logits_on(model, split: str = "val", *, share: float = 1.0,
              count: int = COUNT, size: int = SIZE, seed: int = SEED):
    """Logits and targets on a fixed set of windows from `split`."""
    torch = _torch()
    zs, ys = [], []
    was = model.training
    model.eval()
    with torch.no_grad():
        for x, y in windows(split, share=share, count=count, size=size,
                            seed=seed):
            z, _, _, _ = model(x)
            zs.append(z.reshape(-1, z.shape[-1]))
            ys.append(y.reshape(-1))
    model.train(was)
    return torch.cat(zs).double(), torch.cat(ys)


def fit_temperature(z, y) -> float:
    """The NLL-minimising temperature for logits `z` -- what temperature
    scaling would choose if fitted on these rows."""
    torch = _torch()
    from scipy.optimize import minimize_scalar
    F = torch.nn.functional

    def loss(t):
        return float(F.cross_entropy(z / t, y))
    return float(minimize_scalar(loss, bounds=(0.2, 10.0),
                                 method="bounded").x)


def summarise(z, y, *, bins: int = 15) -> dict:
    """Calibration of one set of logits: the numbers every checkpoint gets."""
    torch = _torch()
    F = torch.nn.functional
    from standarderror.uncertainty import calibration as cb

    p = F.softmax(z, -1)
    conf, pred = p.max(1)
    correct = (pred == y).double()
    t = fit_temperature(z, y)
    p_t = F.softmax(z / t, -1)
    return {
        "nll": float(F.cross_entropy(z, y)),
        "accuracy": float(correct.mean()),
        "confidence": float(conf.mean()),
        "gap": float(conf.mean() - correct.mean()),
        "ece": float(cb.ece(conf.numpy(), correct.numpy(), bins=bins)),
        "fitted_t": t,
        "nll_at_fit": float(F.cross_entropy(z / t, y)),
        "ece_at_fit": float(cb.ece(p_t.max(1).values.numpy(),
                                   correct.numpy(), bins=bins)),
    }


def evaluate(model, *, share: float = 1.0) -> dict:
    """Validation and training-text summaries for one checkpoint; `share` is
    the fraction of the training text the model was trained on."""
    return {"val": summarise(*logits_on(model, "val")),
            "train": summarise(*logits_on(model, "train", share=share))}


def trajectory() -> dict:
    """The committed per-checkpoint metrics of every arm."""
    return json.loads((data_dir() / "trajectory.json").read_text())


def load_checkpoint(name: str):
    """A committed checkpoint by name (`small_final`, `long_best`, ...),
    hash-checked against the value the training run recorded."""
    import hashlib
    torch = _torch()
    from standarderror.llm import tiny
    path = data_dir() / f"{name}.pt"
    want = trajectory()["checkpoints"][name]["sha256"]
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise ValueError(f"{path.name} hash is {got}, expected {want}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = tiny.build(len(blob["chars"]))
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def turn(rows: list[dict], key: str = "nll") -> dict:
    """The checkpoint where validation `key` is lowest -- where the model
    stopped improving on held-out text."""
    best = min(rows, key=lambda r: r["val"][key])
    return {"step": best["step"], "value": best["val"][key]}
