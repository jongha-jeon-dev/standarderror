r"""The arithmetic model: same architecture as the language model, different
corpus, and a grader that decodes an answer rather than reporting a loss.

Reusing `llm.tiny.build` is deliberate. Four blocks, four heads, width 128,
context 64 -- the same object the calculus series took apart. So every tool
written for that model applies here unchanged, and any difference in behaviour
between the two is the corpus, not the architecture. The only thing that
changes is the vocabulary, which is 19 characters instead of 65 and therefore
costs 11,776 fewer parameters in the embedding and the head.

Accuracy here is *exact-match on the decoded answer*, greedily, to the newline.
Not per-character accuracy, which flatters a model that gets the first two
digits of a five-digit answer right, and not loss, which cannot distinguish a
model that is confidently wrong from one that is uncertain and right.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import standarderror as se
from standarderror.llm import tiny
from standarderror.schoolmath import curriculum as cur

#: sha256 of the committed checkpoint. A swapped file should be a loud failure,
#: not a quietly different table of accuracies.
CHECKPOINT_SHA256 = ("c40b480c51c4c6fb50e94d223ba636c76914fa3c9c2815f7d9252"
                     "23925afe819")
#: Reported by the training run that produced the committed checkpoint.
VAL_LOSS = 1.2877

BLOCK = tiny.BLOCK


def checkpoint_path() -> Path:
    return Path(se.SETTINGS.repo_root) / "data" / "schoolmath" / "schoolmath.pt"


def encode(text: str, stoi: dict) -> list[int]:
    return [stoi[c] for c in text]


def stream(split: str = "train", **kw) -> str:
    return cur.text(split, **kw)


def load(*, verify: bool = True, eval_mode: bool = True):
    """The committed checkpoint, its vocabulary, and the model."""
    import torch
    path = checkpoint_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It is committed to this repository; run "
            f"`python scripts/train_schoolmath.py` to rebuild it, but note "
            f"that a rebuild is a *similar* model, not the same one.")
    if verify and CHECKPOINT_SHA256 != "unset":
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != CHECKPOINT_SHA256:
            raise ValueError(f"checkpoint hash is {got}, expected "
                             f"{CHECKPOINT_SHA256}; the file is not the one "
                             f"the published numbers were measured on")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    chars = blob["chars"]
    model = tiny.build(len(chars))
    model.load_state_dict(blob["model"])
    if eval_mode:
        model.eval()
    return {"model": model, "chars": chars,
            "stoi": {c: i for i, c in enumerate(chars)},
            "vocab": len(chars),
            "val_loss": blob.get("val_loss"),
            "parameters": sum(p.numel() for p in model.parameters())}


def _context(prompt: str, filler: str) -> str:
    """Left-pad a prompt with complete problems so it sits where problems sit.

    The model is trained on a packed stream at every alignment, so a prompt fed
    at position zero is a slightly unusual input. Padding with real lines
    removes that as an explanation for a failure; `padding_sensitivity` in
    `arithmetic.py` measures how much it actually mattered, which is little.
    """
    room = BLOCK - len(prompt)
    if room <= 0:
        return prompt[-BLOCK:]
    return (filler[-room:] if filler else "") + prompt


def decode(bundle, prompts: list[str], *, filler="",
           limit: int = 12) -> list[str]:
    """Greedy continuation of each prompt, stopped at the newline.

    `filler` is either one string used for every prompt or a list with one
    string per prompt. A *list* is what you want: a single shared prefix is
    something the model can copy from, and it will -- a fixed filler ending in
    `8+2~01` made the model answer `01` to every reversed sum it was asked,
    which looked exactly like a model that cannot add backwards.

    Batched over prompts, one token per step for all of them at once, because
    the alternative is minutes per table on two cores.
    """
    import torch
    model, stoi, chars = bundle["model"], bundle["stoi"], bundle["chars"]
    nl = stoi["\n"]
    fills = [filler] * len(prompts) if isinstance(filler, str) else list(filler)
    if len(fills) != len(prompts):
        raise ValueError("one filler per prompt, or one string for all")
    ctx = [encode(_context(p, f), stoi) for p, f in zip(prompts, fills)]
    width = max(len(c) for c in ctx)
    # Left-pad the short ones with newlines: a newline is a genuine line
    # boundary in this corpus, so padding with it does not invent syntax.
    ids = torch.tensor([[nl] * (width - len(c)) + c for c in ctx],
                       dtype=torch.long)
    done = torch.zeros(len(prompts), dtype=torch.bool)
    out = [[] for _ in prompts]
    with torch.no_grad():
        for _ in range(limit):
            logits, _, _, _ = model(ids[:, -BLOCK:])
            nxt = logits[:, -1, :].argmax(-1)
            for i, t in enumerate(nxt.tolist()):
                if not done[i]:
                    if t == nl:
                        done[i] = True
                    else:
                        out[i].append(chars[t])
            if bool(done.all()):
                break
            ids = torch.cat([ids, nxt.view(-1, 1)], dim=1)
    return ["".join(o) for o in out]
