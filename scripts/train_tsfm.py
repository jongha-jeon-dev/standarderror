"""Pretrain the time-series foundation model the compression episodes measure.

Run: `python scripts/train_tsfm.py`. Synthetic data only -- a KernelSynth-led
bank generated from a seed -- so every real series in `tsfm.pool` is scored
zero-shot, with no contamination question to answer.

The learning curve records two things at each checkpoint: pinball loss on a
held-out synthetic bank (a different seed), and MASE on a fixed slice of the
real pool. The second is *monitoring only*. No checkpoint is selected on it:
the committed weights are the final step, whatever the curve says, because
choosing the step that scores best on the evaluation pool would make the
evaluation pool a validation set.
"""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import torch

import standarderror as se
from standarderror.tsfm import model as tm
from standarderror.tsfm import synth

STEPS, BATCH, MAX_LR = 12_000, 256, 1e-3
BANK, HELD_OUT = 30_000, 1_000


def _bank(count, seed):
    path = (se.SETTINGS.cache_dir /
            f"tsfm_bank_v{synth.VERSION}_{count}_{seed}_{synth.LENGTH}.npy")
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b = synth.bank(count, seed=seed)
    np.save(path, b)
    return b


def _monitor(model, slice_):
    from standarderror.tsfm import evaluate as ev
    bundle = {"model": model}
    return ev.quick_mase(bundle, slice_)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    train, held = _bank(BANK, 0), _bank(HELD_OUT, 1)
    print(f"bank {train.shape} + held-out {held.shape}  {time.time() - t0:.0f}s",
          flush=True)

    from standarderror.tsfm import evaluate as ev
    slice_ = ev.monitor_slice()
    model = tm.build()
    print(f"parameters {tm.parameters(model):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR,
                                                total_steps=args.steps)
    hv, hm, ht = map(torch.tensor, tm.sample_batch(held, np.random.default_rng(1), 2048))
    history, start = [], time.time()
    for step in range(args.steps):
        v, m, t = map(torch.tensor, tm.sample_batch(train, rng, BATCH))
        loss = tm.pinball(model(v, m), t, torch.ones_like(t))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                hl = float(tm.pinball(model(hv, hm), ht, torch.ones_like(ht)))
            mon = _monitor(model, slice_)
            model.train()
            history.append({"step": step, "held_out_pinball": round(hl, 4),
                            **{k: round(v, 4) for k, v in mon.items()}})
            shown = "  ".join(f"{k} {v:.3f}" for k, v in mon.items())
            print(f"  step {step:6d}  train {loss.item():.4f}  held-out {hl:.4f}"
                  f"  {time.time() - start:7.0f}s\n          {shown}", flush=True)

    model.eval()
    out = tm.checkpoint_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history,
                "steps": args.steps,
                "bank": [BANK, 0, synth.LENGTH, synth.VERSION],
                "shape": {"width": tm.WIDTH, "heads": tm.HEADS,
                          "layers": tm.LAYERS}}, out)
    print(f"saved {out}\nsha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
