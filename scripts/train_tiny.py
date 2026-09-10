"""Train the 816,128-parameter model the later episodes measure.

Run: `python scripts/train_tiny.py`. Three thousand AdamW steps, a few minutes
on a CPU, and it writes `data/tiny_gpt/tiny_gpt.pt`.

The corpus is fetched rather than committed. `tinyshakespeare` is Shakespeare,
which is public domain, assembled into one file by Andrej Karpathy for
`char-rnn`; fetching it with a hash check gets reproducibility without this
repository redistributing someone else's assembly of it. The checkpoint that
comes out *is* committed, because losing it to a container reset once already
cost two episodes.

Determinism: `torch.manual_seed(0)` and a fixed batch schedule, but this is not
bit-reproducible across torch versions or platforms. The committed checkpoint is
the one every published number was measured on, and `tiny.load` verifies its
hash; re-running this will produce a *similar* model, not the same one.
"""

from __future__ import annotations

import hashlib
import math
import time
import urllib.request

import numpy as np
import torch

import standarderror as se
from standarderror.llm import tiny

CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt")
CORPUS_SHA256 = ("86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2d"
                 "c565ed")
STEPS, BATCH, MAX_LR = 3000, 32, 3e-3


def corpus() -> str:
    cache = se.SETTINGS.cache_dir / "tinyshakespeare.txt"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CORPUS_URL, timeout=60) as r:
            cache.write_bytes(r.read())
    raw = cache.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != CORPUS_SHA256:
        raise ValueError(f"corpus hash is {got}, expected {CORPUS_SHA256}")
    return raw.decode("utf-8")


def main() -> None:
    torch.manual_seed(0)
    text = corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(0.9 * len(data))
    train, val = data[:cut], data[cut:]

    model = tiny.build(len(chars))
    print(f"vocab {len(chars)}, parameters "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"uniform-guess loss = ln({len(chars)}) = {math.log(len(chars)):.3f}")

    def batch(split):
        d = train if split == "train" else val
        i = torch.randint(len(d) - tiny.BLOCK - 1, (BATCH,))
        x = torch.stack([d[j:j + tiny.BLOCK] for j in i])
        y = torch.stack([d[j + 1:j + tiny.BLOCK + 1] for j in i])
        return x, y

    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR,
                                                total_steps=STEPS)
    start = time.time()
    for step in range(STEPS):
        x, y = batch("train")
        _, loss, _, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == STEPS - 1:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([model(*batch("val"))[1].item()
                                    for _ in range(20)]))
            model.train()
            print(f"  step {step:5d}  train {loss.item():.4f}  val {vl:.4f}  "
                  f"{time.time() - start:6.1f}s")

    out = tiny.checkpoint_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "chars": chars}, out)
    print(f"saved {out}  sha256 "
          f"{hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
