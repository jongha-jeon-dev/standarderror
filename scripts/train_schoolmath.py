"""Train the arithmetic model the school-maths episodes measure.

Run: `python scripts/train_schoolmath.py`. It writes
`data/schoolmath/schoolmath.pt` and prints the sha256 to paste into
`standarderror.schoolmath.model.CHECKPOINT_SHA256`.

The corpus is generated rather than fetched or committed: it is arithmetic,
which nobody owns, and `curriculum.problems` is deterministic in its seed. What
is committed is the checkpoint, because every published accuracy was measured
on one particular set of weights and a rerun produces a similar model, not the
same one.

The training stream contains only the `train` split. The two probes -- carries
that propagate at three digits, and four-digit operands -- are held out
structurally, so a number measured on them is a statement about a rule the
model was never shown rather than about text it has seen.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time

import numpy as np
import torch

from standarderror.llm import tiny
from standarderror.schoolmath import curriculum as cur
from standarderror.schoolmath import model as sm

STEPS, BATCH, MAX_LR = 20_000, 64, 3e-3
#: Problems per task graded at each checkpoint, to get the learning curve
#: without committing twenty copies of the weights.
PROBE = 120
PER_WIDTH = 20_000


def _pack(split: str, stoi: dict) -> torch.Tensor:
    text = cur.text(split, per_width=PER_WIDTH, seed=0)
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def _accuracy(model, chars, seed=11):
    """Exact-match accuracy per task, on held-out problems, mid-training.

    Cheap enough to run at every eval point, which turns the training run into
    the learning-curve experiment as well. Without it the question "is `order`
    hard, or was it just undertrained?" would need a second run per answer.
    """
    from standarderror.schoolmath import arithmetic as ar
    bundle = {"model": model, "chars": chars,
              "stoi": {c: i for i, c in enumerate(chars)}}
    pool = cur.corpus("test", per_width=2_000, seed=0)
    out = {}
    for task in sorted({p.task for p in pool}):
        part = [q for q in pool if q.task == task][:PROBE]
        # Real context, freshly sampled per problem -- the same conditions
        # the final table is measured under. An empty context is a different
        # and much easier input, and a learning curve measured that way would
        # not be a curve of the number the episodes quote.
        rows = ar.graded(bundle, part)
        out[task] = round(sum(r["ok"] for r in rows) / len(rows), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    torch.manual_seed(0)
    chars = list(cur.ALPHABET)
    stoi = {c: i for i, c in enumerate(chars)}
    train = _pack("train", stoi)
    val = _pack("test", stoi)
    print(f"vocab {len(chars)}   train {len(train):,} chars   "
          f"test {len(val):,} chars")

    model = tiny.build(len(chars))
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    print(f"uniform-guess loss = ln({len(chars)}) = "
          f"{math.log(len(chars)):.3f}")

    def batch(d):
        i = torch.randint(len(d) - tiny.BLOCK - 1, (BATCH,))
        x = torch.stack([d[j:j + tiny.BLOCK] for j in i])
        y = torch.stack([d[j + 1:j + tiny.BLOCK + 1] for j in i])
        return x, y

    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR,
                                                total_steps=args.steps)
    start = time.time()
    vl = float("nan")
    history = []
    for step in range(args.steps):
        x, y = batch(train)
        _, loss, _, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([model(*batch(val))[1].item()
                                    for _ in range(10)]))
                acc = _accuracy(model, chars)
            model.train()
            history.append({"step": step, "test_loss": round(vl, 4), **acc})
            shown = "  ".join(f"{k} {v:.2f}" for k, v in acc.items())
            print(f"  step {step:6d}  train {loss.item():.4f}  "
                  f"test {vl:.4f}  {time.time() - start:7.1f}s\n"
                  f"          {shown}", flush=True)

    model.eval()
    out = sm.checkpoint_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "chars": chars,
                "val_loss": round(vl, 4), "steps": args.steps,
                "history": history}, out)
    print(f"saved {out}")
    print(f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
