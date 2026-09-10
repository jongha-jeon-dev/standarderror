r"""A character-level transformer of 816,128 parameters, and how to load it.

Four blocks, four heads, width 128, context 64, trained on `tinyshakespeare`
for 3,000 AdamW steps to a validation loss of **1.573** against a uniform-guess
`ln(65) = 4.174`. That is a real language model and a small one, and the
distinction matters for every claim made with it: the calculus results the
episodes derive are exact statements about `softmax`, `LayerNorm` and the
embedding table, and they hold at any width; the *frequencies* measured on this
model -- how often a training step lands on a non-differentiable point, how
confident its attention actually gets -- are properties of this model and are
reported as such.

`forward` optionally returns the attention matrices and the per-block hidden
states, because the later episodes need to look inside rather than at the loss.

The corpus is not committed. `train.py` fetches it from the `char-rnn`
repository and checks its SHA-256, so the training run is reproducible without
this repository redistributing someone else's file; the resulting checkpoint
*is* committed, with its own hash recorded in `data/LICENCES.md`.
"""

from __future__ import annotations

import math
from pathlib import Path

import standarderror as se

BLOCK, WIDTH, HEADS, LAYERS = 64, 128, 4, 4
#: sha256 of the committed checkpoint, so a corrupted or swapped file is a
#: loud failure rather than a quietly different set of numbers.
CHECKPOINT_SHA256 = ("4353eb3155360b654072b1b3c46416000de23c9ab29c5bbf78649"
                     "cfe7d03c2f0")
#: Reported by the training run that produced the committed checkpoint.
VAL_LOSS = 1.573
UNIFORM_LOSS = math.log(65)


def checkpoint_path() -> Path:
    return Path(se.SETTINGS.repo_root) / "data" / "tiny_gpt" / "tiny_gpt.pt"


def _torch():
    """Imported lazily. Most of this repository does not need torch, and a
    module-level import would make `standarderror` unimportable without it."""
    import torch
    return torch


def build(vocab: int):
    """The architecture, as a plain module tree rather than a config object.

    Written out because the episodes quote its pieces -- `ln1` in front of the
    attention, `GELU` in the MLP, a `LayerNorm` before the head -- and a reader
    checking a Jacobian claim needs to see where the normalisation sits.
    """
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
            self.proj = nn.Linear(WIDTH, WIDTH, bias=False)
            self.register_buffer("mask", torch.tril(
                torch.ones(BLOCK, BLOCK)).view(1, 1, BLOCK, BLOCK))

        def forward(self, x, want_attn=False):
            B, T, C = x.shape
            q, k, v = self.qkv(x).split(WIDTH, dim=2)
            hd = C // HEADS
            q = q.view(B, T, HEADS, hd).transpose(1, 2)
            k = k.view(B, T, HEADS, hd).transpose(1, 2)
            v = v.view(B, T, HEADS, hd).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0,
                                  float("-inf"))
            att = F.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(y), (att if want_attn else None)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
            self.attn = Attention()
            self.mlp = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(),
                                     nn.Linear(4 * WIDTH, WIDTH))

        def forward(self, x, want_attn=False):
            a, att = self.attn(self.ln1(x), want_attn)
            x = x + a
            return x + self.mlp(self.ln2(x)), att

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, WIDTH)
            self.pos = nn.Embedding(BLOCK, WIDTH)
            self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
            self.lnf = nn.LayerNorm(WIDTH)
            self.head = nn.Linear(WIDTH, vocab, bias=False)

        def forward(self, idx, targets=None, want_attn=False,
                    want_hidden=False):
            B, T = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
            atts, hiddens = [], []
            for b in self.blocks:
                x, att = b(x, want_attn)
                if want_attn:
                    atts.append(att)
                if want_hidden:
                    hiddens.append(x)
            x = self.lnf(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, vocab),
                                       targets.reshape(-1))
            return logits, loss, atts, hiddens

    return GPT()


def load(*, verify: bool = True, eval_mode: bool = True):
    """The committed checkpoint, its vocabulary, and the model it belongs to.

    `verify` hashes the file. It costs milliseconds and it is the difference
    between "these numbers changed" and "these numbers changed and I know why".
    """
    torch = _torch()
    path = checkpoint_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It is committed to this repository; if this "
            f"is a fresh clone with an incomplete checkout, fetch it before "
            f"running an episode that needs the model.")
    if verify:
        import hashlib
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != CHECKPOINT_SHA256:
            raise ValueError(f"checkpoint hash is {got}, expected "
                             f"{CHECKPOINT_SHA256}; the file is not the one "
                             f"the published numbers were measured on")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    chars = blob["chars"]
    model = build(len(chars))
    model.load_state_dict(blob["model"])
    if eval_mode:
        model.eval()
    stoi = {c: i for i, c in enumerate(chars)}
    return {"model": model, "chars": chars, "stoi": stoi,
            "vocab": len(chars),
            "parameters": sum(p.numel() for p in model.parameters())}
