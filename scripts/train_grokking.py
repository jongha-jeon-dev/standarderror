"""Train the modular-addition networks for ML 27 (grokking) and write
data/ml_grokking/results.json.

Each run is `standarderror.ml.nets.modular`: (a + b) mod 97 from learned
embeddings of a and b through one hidden layer of 256, full-batch AdamW,
learning rate 1e-3, betas (0.9, 0.98), 10,000 steps, training and test
accuracy and the weight norm every 100 steps. Four runs: weight decay 1 and
0 on 40% of the pairs, and weight decay 1 on 30% and on 60%. About ten
minutes each on one CPU thread; the episode reads the file, and
`tests/test_ml.py` checks that a short rerun reproduces its first entries.

Run: python scripts/train_grokking.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from standarderror.ml import nets

RUNS = [{"weight_decay": 1.0, "frac": 0.4}, {"weight_decay": 0.0, "frac": 0.4},
        {"weight_decay": 1.0, "frac": 0.3}, {"weight_decay": 1.0, "frac": 0.6}]


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "ml_grokking"
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for kw in RUNS:
        t = time.time()
        r = nets.modular(**kw)
        r["seconds"] = round(time.time() - t, 1)
        runs.append(r)
        print(kw, r["train_100"], r["test_99"], r["final_test"], r["seconds"],
              flush=True)
        (out / "results.json").write_text(json.dumps({"runs": runs}, indent=1))


if __name__ == "__main__":
    main()
