"""Fine-tune the arithmetic model a dozen ways for ML 51 and ML 52, and write
data/ml_llmeval/runs.json.

Two kinds of run, all starting from the committed main checkpoint and all the
same length (600 AdamW steps at a constant 3e-4, about one more pass over the
training split):

* continuations with nothing leaked, seeds 0-4. ML 51 uses them as a cluster
  of models that differ only by training noise, the situation on a real
  leaderboard where five labs fine-tune the same base;
* contaminated runs, where the 300 benchmark items of `llmeval.leak_sets` are
  inserted into the stream 1, 2, 4, 8, 16 or 32 times. ML 52 reads the dose
  response, and a detector's view of it, from these.

Each run records per-item results, not just means, so the episodes can pair,
bootstrap and cluster. The family snapshots of `train_schoolmath.py --family`
and the two committed models are graded on the same pool here as well, so
every model ML 51 ranks was graded with the same contexts.

Run: python scripts/llm_eval_runs.py   (about forty minutes on one thread)
"""
from __future__ import annotations

import json
import time

import torch

from standarderror.ml import llmeval as le
from standarderror.schoolmath import model as sm

DOSES = [0, 1, 2, 4, 8, 16, 32]
CONTROL_SEEDS = [1, 2, 3, 4]
DOSE_REPLICATES = {8: [1, 2]}
FAMILY = ["s1_3000", "s1_4500", "s1_6000"]


def main():
    torch.set_num_threads(1)
    out = le.data_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runs.json"
    base = sm.load()
    pool = le.pool_items()
    bench, fresh = le.leak_sets()
    fb, ff = le.ar.fillers(len(bench), seed=5), le.ar.fillers(len(fresh),
                                                              seed=6)
    t0 = time.time()

    def leak_view(b):
        return {"bench_ok": le.ok(b, bench, fills=fb).tolist(),
                "fresh_ok": le.ok(b, fresh, fills=ff).tolist(),
                "bench_lp": le.answer_logp(
                    b, [p.prompt for p in bench], [p.answer for p in bench],
                    fb).round(4).tolist(),
                "fresh_lp": le.answer_logp(
                    b, [p.prompt for p in fresh], [p.answer for p in fresh],
                    ff).round(4).tolist()}

    res = {"pool": [{"split": s, "task": t} for s, t in le.POOL],
           "pool_per_task": len(pool) // len(le.POOL),
           "steps": 600, "lr": 3e-4, "models": {}, "doses": {}}

    def save():
        path.write_text(json.dumps(res))

    for name in ["clean", "leaky"] + FAMILY:
        b = (sm.load(adjacent=name == "leaky") if name in ("clean", "leaky")
             else sm.load_member(name))
        res["models"][name] = {"pool_ok": le.ok(b, pool).tolist()}
        print(name, round(sum(res["models"][name]["pool_ok"]) / len(pool), 4),
              f"{time.time() - t0:.0f}s", flush=True)
    res["doses"]["base"] = leak_view(base)
    save()

    for seed in [0] + CONTROL_SEEDS:
        b = le.finetune(base, seed=seed)
        res["models"][f"cont{seed}"] = {"pool_ok": le.ok(b, pool).tolist()}
        res["doses"][f"0/{seed}"] = leak_view(b)
        print("cont", seed, f"{time.time() - t0:.0f}s", flush=True)
        save()

    leaked = [p.line for p in bench]
    for k in DOSES[1:]:
        for seed in [0] + DOSE_REPLICATES.get(k, []):
            b = le.finetune(base, leaked, repeats=k, seed=seed)
            res["doses"][f"{k}/{seed}"] = leak_view(b)
            v = res["doses"][f"{k}/{seed}"]
            print("dose", k, seed, sum(v["bench_ok"]) / len(bench),
                  sum(v["fresh_ok"]) / len(fresh),
                  f"{time.time() - t0:.0f}s", flush=True)
            save()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
