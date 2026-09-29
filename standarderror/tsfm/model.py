r"""A small time-series foundation model: patches in, quantiles out.

The design is the Chronos-Bolt / TimesFM one rather than the original
tokenised Chronos. The tokenised recipe -- mean-scale, then bin every value
into one of a few thousand tokens -- was tried on paper first and rejected for
a reason specific to this project's budget: at a vocabulary small enough to
train on two CPU cores, the weekly variation of the Mauna Loa CO2 series falls
*inside a single bin*, and a model that cannot see a signal cannot forecast
it. Continuous patches normalised by the context's own mean and standard
deviation have no such floor.

So: the last `CONTEXT` observations are normalised, cut into patches of
`PATCH`, each patch together with its observed-mask is embedded by a small
residual MLP, a bidirectional transformer mixes the patches, and the final
token is read out as `OUT` future steps at each of nine quantile levels. One
forward pass per forecast, no sampling -- which is also what makes the
compression episodes cheap to run.

The encoder block is the one `llm.tiny` uses: pre-norm, GELU MLP, four times
width. Same parts, different shape, so tools written for the language model
read this one too.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

import standarderror as se

CONTEXT, PATCH, OUT = 256, 16, 32
TOKENS = CONTEXT // PATCH
WIDTH, HEADS, LAYERS = 192, 6, 6
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: Shortest context the model is trained to accept. The World Bank series
#: leave as little as 24 years once six are held out, so the model must be
#: taught short contexts deliberately rather than meeting them first at test.
MIN_CONTEXT = 16

CHECKPOINT_SHA256 = ("6a924a8c2aeaf95d7e854157ea4a60376eb697517c922aa4a43fe"
                     "6dadd4213b2")


def checkpoint_path() -> Path:
    return Path(se.SETTINGS.repo_root) / "data" / "tsfm" / "tsfm.pt"


def _torch():
    import torch
    return torch


def build(*, width: int = WIDTH, heads: int = HEADS, layers: int = LAYERS):
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    class Residual(nn.Module):
        """TimesFM's input and output block: MLP plus a linear skip."""
        def __init__(self, d_in, d_hidden, d_out):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(d_in, d_hidden), nn.Linear(d_hidden, d_out)
            self.skip = nn.Linear(d_in, d_out)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x))) + self.skip(x)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
            self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                     nn.Linear(4 * width, width))

        def forward(self, x, pad):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
            x = x + a
            return x + self.mlp(self.ln2(x))

    class TSFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = Residual(2 * PATCH, 2 * width, width)
            self.pos = nn.Embedding(TOKENS, width)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.lnf = nn.LayerNorm(width)
            self.head = Residual(width, 2 * width, OUT * len(QUANTILES))

        def forward(self, values, mask):
            """`values`, `mask`: (B, CONTEXT), already normalised; mask 1 =
            observed. Returns (B, OUT, Q) in normalised units."""
            B = values.shape[0]
            v = (values * mask).view(B, TOKENS, PATCH)
            m = mask.view(B, TOKENS, PATCH)
            x = self.embed(torch.cat([v, m], -1))
            x = x + self.pos(torch.arange(TOKENS, device=x.device))
            # A patch with nothing observed is padding, except that the last
            # patch must always be attendable or an all-empty row has no keys.
            pad = m.sum(-1) == 0
            pad[:, -1] = False
            for b in self.blocks:
                x = b(x, pad)
            return self.head(self.lnf(x[:, -1])).view(B, OUT, len(QUANTILES))

    return TSFM()


def normalise(context: np.ndarray):
    """Left-pad or truncate to `CONTEXT`, then scale by the observed part.

    Returns `(values, mask, loc, scale)`. The scale floor is relative to the
    level, so a series that is nearly constant -- a price index over two
    quarters -- is not blown up into noise by dividing by almost zero.
    """
    x = np.asarray(context, float)[-CONTEXT:]
    ok = np.isfinite(x)
    loc = float(np.mean(x[ok])) if ok.any() else 0.0
    sd = float(np.std(x[ok])) if ok.sum() > 1 else 0.0
    scale = max(sd, 1e-3 * abs(loc), 1e-8)
    z = np.where(ok, (x - loc) / scale, 0.0)
    values = np.zeros(CONTEXT, np.float32)
    mask = np.zeros(CONTEXT, np.float32)
    values[-len(z):], mask[-len(z):] = z, ok
    return values, mask, loc, scale


def pinball(pred, target, valid):
    """Mean quantile loss over the levels, horizon steps and valid targets."""
    torch = _torch()
    q = torch.tensor(QUANTILES, dtype=pred.dtype, device=pred.device)
    diff = target.unsqueeze(-1) - pred
    loss = torch.maximum(q * diff, (q - 1) * diff)
    w = valid.unsqueeze(-1).expand_as(loss)
    return (loss * w).sum() / w.sum().clamp_min(1)


def sample_batch(bank: np.ndarray, rng, size: int):
    """Windows from the synthetic bank: a context of random length, then `OUT`.

    Context lengths are drawn so that a third are short (16 to 64), because
    the annual pool is short and a model that has only seen 256-step contexts
    has never been asked the question the World Bank series ask.
    """
    n, length = bank.shape
    vals = np.zeros((size, CONTEXT), np.float32)
    masks = np.zeros((size, CONTEXT), np.float32)
    targ = np.zeros((size, OUT), np.float32)
    for i in range(size):
        row = bank[rng.integers(n)]
        c = int(rng.integers(MIN_CONTEXT, 65)) if rng.random() < 1 / 3 \
            else int(rng.integers(65, CONTEXT + 1))
        end = int(rng.integers(c, length - OUT + 1))
        v, m, loc, scale = normalise(row[end - c:end])
        vals[i], masks[i] = v, m
        targ[i] = (row[end:end + OUT] - loc) / scale
    return vals, masks, targ


def load(path: Path | None = None, *, verify: bool = True):
    torch = _torch()
    path = path or checkpoint_path()
    raw = path.read_bytes()
    if verify and path == checkpoint_path() and CHECKPOINT_SHA256 != "unset":
        got = hashlib.sha256(raw).hexdigest()
        if got != CHECKPOINT_SHA256:
            raise ValueError(f"checkpoint hash is {got}, expected "
                             f"{CHECKPOINT_SHA256}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = build(**blob.get("shape", {}))
    model.load_state_dict(blob["model"])
    model.eval()
    return {"model": model, "history": blob.get("history", []),
            "parameters": sum(p.numel() for p in model.parameters()),
            "path": str(path)}


def forecast(bundle, contexts: list[np.ndarray], horizon: int, *,
             chunk: int = 512) -> np.ndarray:
    """Quantile forecasts in original units: (N, horizon, Q), sorted per step.

    Sorting is applied here rather than learned away, and `crossing_rate`
    measures how often it was needed -- a compressed model that starts
    crossing its own quantiles is failing in a way no point metric sees.
    """
    torch = _torch()
    if horizon > OUT:
        raise ValueError(f"horizon {horizon} exceeds the model's {OUT}")
    out = []
    model = bundle["model"]
    for i in range(0, len(contexts), chunk):
        part = [normalise(c) for c in contexts[i:i + chunk]]
        v = torch.tensor(np.stack([p[0] for p in part]))
        m = torch.tensor(np.stack([p[1] for p in part]))
        with torch.no_grad():
            q = model(v, m)[:, :horizon].numpy()
        loc = np.array([p[2] for p in part])[:, None, None]
        scale = np.array([p[3] for p in part])[:, None, None]
        # Sorted, as the docstring promises. The first evaluation ran without
        # this line -- the promise was written, the sort was not -- and 3.5 to
        # 4.2% of (series, step) pairs had crossed quantiles, so the "median"
        # and the band edges were read from the wrong positions on those.
        out.append(np.sort(q * scale + loc, axis=-1))
    return np.concatenate(out)


def crossing_rate(bundle, contexts, horizon: int) -> float:
    """Share of (series, step) pairs whose raw quantiles are out of order."""
    torch = _torch()
    part = [normalise(c) for c in contexts]
    v = torch.tensor(np.stack([p[0] for p in part]))
    m = torch.tensor(np.stack([p[1] for p in part]))
    with torch.no_grad():
        q = bundle["model"](v, m)[:, :horizon].numpy()
    return float((np.diff(q, axis=-1) < 0).any(-1).mean())


def parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


UNIFORM = math.nan  # placeholder kept for API symmetry with llm.tiny
