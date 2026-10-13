"""Train a ladder of small character transformers on tinyshakespeare for
ML 30 (scaling laws) and write data/ml_scaling/results.json.

Every model sees the same 2,000 AdamW steps of 32 x 64 characters (4.1M
tokens, about four passes over the 1.0M-character training split), with a
linear warm-up and cosine decay. Train and validation loss are recorded every
250 steps on fixed batches. One seed per size; the run takes about half an
hour on two CPU threads.

Run: python scripts/train_scaling.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from standarderror.ml import scaling as sc

SIZES = [(16, 1), (24, 2), (32, 2), (48, 2), (64, 3), (96, 3), (128, 4),
         (192, 4)]
STEPS, EVERY = 2000, 250


def main():
    torch.set_num_threads(2)
    out = Path(__file__).resolve().parents[1] / "data" / "ml_scaling"
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for width, layers in SIZES:
        t = time.time()
        r = sc.train(width, layers, steps=STEPS, every=EVERY, seed=0)
        r["seconds"] = round(time.time() - t, 1)
        runs.append(r)
        print(width, layers, r["params"], r["final_val"], r["best_val"],
              r["seconds"], flush=True)
        (out / "results.json").write_text(json.dumps(
            {"steps": STEPS, "batch": sc.BATCH, "context": sc.T,
             "tokens": STEPS * sc.BATCH * sc.T, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
