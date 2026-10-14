r"""Episode 30: a ladder of character transformers, and the power law fitted
to it.

`train` builds and trains one model; `scripts/train_scaling.py` runs the
ladder and commits the losses to `data/ml_scaling/results.json`, because the
whole ladder took 1.9 hours on two CPU threads and an episode build should
not. `fit` and
`predict` are what the episode does with those numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import standarderror as se

T, BATCH, HEADS = 64, 32, 4


def _torch():
    import torch
    return torch


def build(vocab: int, width: int, layers: int):
    torch = _torch()
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
            self.att = nn.MultiheadAttention(width, HEADS, batch_first=True,
                                             bias=False)
            self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                     nn.Linear(4 * width, width))
            self.register_buffer("mask", torch.triu(
                torch.ones(T, T, dtype=torch.bool), 1))

        def forward(self, x):
            h = self.ln1(x)
            n = x.shape[1]
            a, _ = self.att(h, h, h, attn_mask=self.mask[:n, :n],
                            need_weights=False)
            x = x + a
            return x + self.mlp(self.ln2(x))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, width)
            self.pos = nn.Embedding(T, width)
            self.blocks = nn.Sequential(*[Block() for _ in range(layers)])
            self.ln = nn.LayerNorm(width)
            self.head = nn.Linear(width, vocab, bias=False)

        def forward(self, x):
            pos = torch.arange(x.shape[1])
            return self.head(self.ln(self.blocks(self.tok(x) +
                                                 self.pos(pos))))

    return GPT()


def non_embedding_params(model) -> int:
    """Parameters outside the token and position embeddings, the count
    Kaplan et al. (2020) found gives the cleanest power law."""
    return sum(p.numel() for n, p in model.named_parameters()
               if not (n.startswith("tok.") or n.startswith("pos.")))


def _data():
    torch = _torch()
    from standarderror.llm import tiny
    text = tiny.corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(0.9 * len(data))
    return data[:cut], data[cut:], len(chars)


def _batch(d, rng, size):
    torch = _torch()
    i = rng.integers(0, len(d) - T - 1, size)
    x = torch.stack([d[j:j + T] for j in i])
    y = torch.stack([d[j + 1:j + T + 1] for j in i])
    return x, y


def train(width: int, layers: int, *, steps: int = 2000, every: int = 250,
          lr: float = 3e-3, seed: int = 0, eval_batches: int = 20) -> dict:
    torch = _torch()
    import numpy as np
    F = torch.nn.functional
    tr, va, V = _data()
    torch.manual_seed(seed)
    model = build(V, width, layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    warm = steps // 20

    def lr_at(s):
        if s < warm:
            return lr * (s + 1) / warm
        q = (s - warm) / max(1, steps - warm)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q)))

    rng = np.random.default_rng(seed)
    erng = np.random.default_rng(12345)
    ev_tr = [_batch(tr, erng, 64) for _ in range(eval_batches)]
    ev_va = [_batch(va, erng, 64) for _ in range(eval_batches)]

    def evaluate(batches):
        model.eval()
        with torch.no_grad():
            out = float(np.mean([F.cross_entropy(
                model(x).reshape(-1, V), y.reshape(-1)).item()
                for x, y in batches]))
        model.train()
        return out

    curve = []
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = _batch(tr, rng, BATCH)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % every == 0:
            curve.append({"step": s + 1, "train": evaluate(ev_tr),
                          "val": evaluate(ev_va)})
    return {"width": width, "layers": layers,
            "params": sum(p.numel() for p in model.parameters()),
            "non_embedding": non_embedding_params(model),
            "curve": curve, "final_val": curve[-1]["val"],
            "final_train": curve[-1]["train"],
            "best_val": min(c["val"] for c in curve)}


def results() -> dict:
    path = Path(se.SETTINGS.repo_root) / "data" / "ml_scaling" / "results.json"
    return json.loads(path.read_text())


def fit(N, L, *, floor: bool = True):
    """Least-squares fit of L = a N^-b (+ c) in log space for the pure power
    law, and by nonlinear least squares with the floor."""
    import numpy as np
    N, L = np.asarray(N, float), np.asarray(L, float)
    if not floor:
        b, la = np.polyfit(np.log(N), np.log(L), 1)
        return {"a": float(np.exp(la)), "b": float(-b), "c": 0.0}
    from scipy.optimize import curve_fit
    f = lambda n, a, b, c: a * n ** (-b) + c  # noqa: E731
    p, _ = curve_fit(f, N, L, p0=(10.0, 0.3, 1.0), maxfev=20000,
                     bounds=([0, 0, 0], [1e6, 3, 5]))
    return {"a": float(p[0]), "b": float(p[1]), "c": float(p[2])}


def predict(params: dict, N):
    import numpy as np
    N = np.asarray(N, float)
    return params["a"] * N ** (-params["b"]) + params["c"]
