"""Hyperparameter search with an honest protocol.

The protocol matters more than the search. Two rules, enforced structurally:

1. **Validation is a contiguous block *after* training, never a random split.**
   K-fold on a time series leaks the future into the past; on a chaotic series it
   leaks catastrophically because neighbouring points are nearly identical.
   `rolling_origin` gives you the correct thing.
2. **Selection metric is the autonomous-rollout error, not the teacher-forced
   one.** Tuning on one-step error selects reservoirs that echo, which then fall
   apart the moment the loop closes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from .esn import ESN, ESNConfig
from .metrics import nrmse


@dataclass
class Split:
    train: slice
    val: slice


def rolling_origin(n: int, *, n_folds: int = 3, val_len: int = 500,
                   min_train: int | None = None) -> list[Split]:
    """Expanding-window splits: each fold trains on everything before its
    validation block. This is the only split that respects causality."""
    min_train = min_train or max(int(0.4 * n), 1000)
    if min_train + n_folds * val_len > n:
        raise ValueError(
            f"cannot fit {n_folds} folds of {val_len} after {min_train} "
            f"training points in a series of length {n}")
    step = (n - min_train - val_len) // max(n_folds - 1, 1) if n_folds > 1 else 0
    out = []
    for i in range(n_folds):
        end_train = min_train + i * step
        out.append(Split(slice(0, end_train),
                         slice(end_train, end_train + val_len)))
    return out


def grid_search_esn(
    series: np.ndarray,
    grid: dict[str, Iterable],
    *,
    base: ESNConfig | None = None,
    horizon: int = 200,
    n_folds: int = 3,
    val_len: int | None = None,
    score: Callable[[np.ndarray, np.ndarray], float] = nrmse,
    autonomous: bool = True,
    verbose: bool = False,
) -> dict:
    """Grid search an ESN with expanding-window validation.

    Returns the best config plus the full score table, so a post can show the
    sensitivity surface rather than just asserting a magic number. Spectral
    radius vs leak rate is usually a *broad plateau*, and showing that is far
    more informative than "we used 0.9".
    """
    x = np.asarray(series, float)
    if x.ndim == 1:
        x = x[:, None]
    base = base or ESNConfig()
    val_len = val_len or horizon + base.washout + 50

    keys = list(grid)
    combos = list(product(*(list(grid[k]) for k in keys)))
    splits = rolling_origin(len(x) - 1, n_folds=n_folds, val_len=val_len)

    rows = []
    for combo in combos:
        cfg = replace(base, **dict(zip(keys, combo)))
        fold_scores = []
        for sp in splits:
            tr = x[sp.train]
            if len(tr) <= cfg.washout + 50:
                continue
            model = ESN(cfg)
            try:
                model.fit(tr[:-1], tr[1:])
                if autonomous:
                    warm = x[max(sp.train.stop - cfg.washout, 0): sp.train.stop]
                    pred = model.predict_autonomous(warm, horizon)
                    truth = x[sp.val.start: sp.val.start + horizon]
                    m = min(len(pred), len(truth))
                    s = score(truth[:m], pred[:m])
                else:
                    va = x[sp.val]
                    pred = model.predict_teacher_forced(va[:-1])
                    s = score(va[1:], pred)
            except Exception as exc:            # unstable configs are informative
                s = float("inf")
                if verbose:
                    print(f"  {dict(zip(keys, combo))} failed: {exc}")
            fold_scores.append(float(s) if np.isfinite(s) else float("inf"))
        if not fold_scores:
            continue
        row = dict(zip(keys, combo))
        row["score_mean"] = float(np.mean(fold_scores))
        row["score_std"] = float(np.std(fold_scores))
        row["folds"] = fold_scores
        rows.append(row)
        if verbose:
            print(f"  {dict(zip(keys, combo))} -> {row['score_mean']:.4f}")

    if not rows:
        raise RuntimeError("no configuration produced a usable score")
    rows.sort(key=lambda r: r["score_mean"])
    best = rows[0]
    best_cfg = replace(base, **{k: best[k] for k in keys})
    return {"best_config": best_cfg, "best_score": best["score_mean"],
            "table": rows, "keys": keys, "n_folds": len(splits),
            "selection": "autonomous rollout" if autonomous else "teacher forced",
            "horizon": horizon}
