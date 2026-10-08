"""Three training runs with checkpoints, for Coverage episode 4.

Run: `python scripts/train_coverage4.py [arm ...]`. About an hour on two CPU
cores for all three; naming arms reruns only those and keeps the rest. It
writes `data/coverage4/trajectory.json` and, per arm, the final checkpoint and
-- where it is a different step -- the one with the lowest validation NLL.

    recipe   the committed model's recipe: full training text, 3,000 steps,
             one-cycle learning rate peaking at 3e-3
    long     the same schedule stretched to 15,000 steps: about 31 passes over
             the training text against the recipe's 6 -- what "train longer"
             means in practice
    small    a tenth of the training text and 6,000 steps: about 120 passes,
             so the model has every chance to memorise

Each arm restarts from `torch.manual_seed(0)`, so the three share their
initialisation and differ only in data and schedule; `small_seed1` repeats the
small arm from seed 1 as a control. The recipe arm uses the
same batch sampler as `scripts/train_tiny.py`; on a different torch version it
is a sibling of the committed model, not a copy, and the episode reports how
close it lands.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
import time

import torch

from standarderror.llm import tiny
from standarderror.uncertainty import trajectory as tj

BATCH, MAX_LR = 32, 3e-3
ARMS = {
    "recipe": {"steps": 3000, "share": 1.0, "every": 100, "seed": 0},
    "long": {"steps": 15000, "share": 1.0, "every": 500, "seed": 0},
    "small": {"steps": 6000, "share": 0.1, "every": 250, "seed": 0},
    # The control: the arm with the strongest effect, from another seed.
    # Metrics only; its checkpoints are not committed.
    "small_seed1": {"steps": 6000, "share": 0.1, "every": 250, "seed": 1,
                    "save": False},
}


def run(name: str, cfg: dict, train, chars) -> tuple[list, dict]:
    torch.manual_seed(cfg["seed"])
    data = train[: int(len(train) * cfg["share"])]
    model = tiny.build(len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR,
                                                total_steps=cfg["steps"])
    rows, best, best_nll = [], None, float("inf")
    start = time.time()
    for step in range(cfg["steps"] + 1):
        if step % cfg["every"] == 0 or step == cfg["steps"]:
            m = tj.evaluate(model, share=cfg["share"])
            rows.append({"step": step, "lr": sched.get_last_lr()[0], **m})
            if m["val"]["nll"] < best_nll:
                best_nll, best = m["val"]["nll"], (step, copy.deepcopy(
                    model.state_dict()))
            print(f"{name:6s} {step:6d}  train {m['train']['nll']:.4f}  "
                  f"val {m['val']['nll']:.4f}  gap {m['val']['gap']:+.4f}  "
                  f"T {m['val']['fitted_t']:.3f}  "
                  f"{time.time() - start:7.1f}s", flush=True)
        if step == cfg["steps"]:
            break
        i = torch.randint(len(data) - tiny.BLOCK - 1, (BATCH,))
        x = torch.stack([data[j:j + tiny.BLOCK] for j in i])
        y = torch.stack([data[j + 1:j + tiny.BLOCK + 1] for j in i])
        _, loss, _, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    out = tj.data_dir()
    out.mkdir(parents=True, exist_ok=True)
    saved = {}
    if not cfg.get("save", True):
        return rows, saved
    keep = [("final", model.state_dict(), cfg["steps"])]
    if best[0] != cfg["steps"]:
        keep.append(("best", best[1], best[0]))
    for tag, state, step in keep:
        path = out / f"{name}_{tag}.pt"
        torch.save({"model": state, "chars": chars, "step": step}, path)
        saved[f"{name}_{tag}"] = {
            "step": step,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return rows, saved


def main() -> None:
    text = tiny.corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    train = data[: int(0.9 * len(data))]
    result = {"arms": {}, "checkpoints": {}, "config": {
        "batch": BATCH, "max_lr": MAX_LR, "weight_decay": 0.1,
        "eval_rows": tj.COUNT * tj.SIZE * tiny.BLOCK, "arms": ARMS,
        "torch": torch.__version__, "python": platform.python_version(),
        "train_chars": int(len(train))}}
    path = tj.data_dir() / "trajectory.json"
    only = sys.argv[1:] or list(ARMS)
    if path.exists() and sys.argv[1:]:
        # Rerunning some arms keeps the others' committed rows.
        old = json.loads(path.read_text())
        result["arms"].update({k: v for k, v in old["arms"].items()
                               if k not in only})
        result["checkpoints"].update({k: v for k, v in
                                      old["checkpoints"].items()
                                      if k.rsplit("_", 1)[0] not in only})
    for name, cfg in ARMS.items():
        if name not in only:
            continue
        rows, saved = run(name, cfg, train, chars)
        result["arms"][name] = rows
        result["checkpoints"].update(saved)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1))
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
