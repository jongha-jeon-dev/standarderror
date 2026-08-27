"""The stylised-facts battery, and the two generators it has to be run against.

A generative model of returns is nearly always judged by a table: fat tails, no
autocorrelation in returns, strong autocorrelation in absolute returns, sometimes a
leverage effect. The model matches them, more or less, and the table is the evidence.

Two things go wrong with that, and both are about the table rather than the model.

**The battery is not a scoreboard.** Every row is reported as though more is better,
but each fact is a property of the process being imitated, and the correct value is
whatever that process has. A symmetric GARCH has *no* leverage effect, so a
generator that produces one has not scored a point — it has invented a dependence
the data does not contain. `facts_error` therefore measures distance from a
reference, in each fact's own units, and refuses to add the rows up: collapsing them
into one number requires a weighting across incommensurable quantities that nobody
who publishes such a table ever states.

**The baselines are missing.** Two generators need no fitting, no architecture and
no compute, and they bracket the whole exercise:

* `iid_bootstrap` — resample the training returns with replacement. This gets the
  marginal distribution *exactly* right, kurtosis included, and destroys every
  dependence. Any table where fat tails are the headline result must show this row,
  because shuffling reproduces fat tails perfectly.
* `block_bootstrap` — resample contiguous blocks instead of points. This keeps
  dependence up to the block length and has one parameter. It is the actual bar: a
  model that does not beat a moving-block bootstrap on a stylised-facts table has
  not been shown to have learned anything a shuffle with memory does not.

Neither is a rival to a diffusion model in what it can do — a bootstrap cannot be
conditioned, cannot extrapolate past the sample, and can only ever return values it
has already seen. That is exactly why they belong in the table: they mark the part
of the score that carries no information about the model.

All functions take and return windows as 2-D arrays, one path per row, because
within-window dependence is the thing being measured and concatenating paths would
manufacture joins that no path contains.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FACTS", "block_bootstrap", "facts_error", "iid_bootstrap",
           "stylised_facts", "within_window_clustering"]

#: The battery, in the order it is usually presented, with what each row means.
FACTS = {
    "sd": "standard deviation of pooled values",
    "excess_kurtosis": "pooled kurtosis minus 3; zero for a Gaussian",
    "acf1_returns": "lag-1 autocorrelation of returns, averaged over paths",
    "acf1_abs": "lag-1 autocorrelation of absolute returns: clustering",
    "leverage": "correlation of a return with the next absolute return",
}


def _windows(a) -> np.ndarray:
    w = np.atleast_2d(np.asarray(a, dtype=float))
    if w.ndim != 2:
        raise ValueError("expected a 2-D array of one path per row")
    if w.shape[1] < 3:
        raise ValueError("paths must be at least three steps long")
    return w


def _per_path(w: np.ndarray) -> dict[str, np.ndarray]:
    """Lag-1 statistics computed inside each path, never across the joins."""
    x, y = w[:, :-1], w[:, 1:]
    ax, ay = np.abs(x), np.abs(y)

    def corr(a, b):
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        num = (a * b).sum(axis=1)
        den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
        out = np.full(a.shape[0], np.nan)
        ok = den > 0
        out[ok] = num[ok] / den[ok]
        return out

    return {"acf1_returns": corr(x, y), "acf1_abs": corr(ax, ay),
            "leverage": corr(x, ay)}


def stylised_facts(windows, *, n_boot: int = 0,
                   seed: int = 0) -> dict[str, dict[str, float]]:
    """The five facts, each with a standard error over paths.

    The errors are what make the table readable. A kurtosis of 23 against 6 is a
    difference; a clustering coefficient of 0.148 against 0.140 is not, and without
    an error column there is no way for a reader to tell which they are looking at.
    Set `n_boot` to resample paths for the pooled quantities, whose sampling error
    is not the same as the spread of a per-path average.
    """
    w = _windows(windows)
    flat = w.ravel()
    per = _per_path(w)
    n = w.shape[0]

    def pooled(rows):
        f = rows.ravel()
        c = f - f.mean()
        v = c.var()
        return {"sd": float(f.std()),
                "excess_kurtosis": float((c ** 4).mean() / v ** 2 - 3.0)}

    out: dict[str, dict[str, float]] = {}
    base = pooled(w)
    if n_boot:
        rng = np.random.default_rng(seed)
        draws = {k: [] for k in base}
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, n)
            for k, v in pooled(w[idx]).items():
                draws[k].append(v)
        for k, v in base.items():
            out[k] = {"value": v, "se": float(np.std(draws[k], ddof=1)),
                      "n": float(n)}
    else:
        for k, v in base.items():
            out[k] = {"value": v, "se": float("nan"), "n": float(n)}

    for k, vals in per.items():
        # A path with a constant absolute value has no defined lag-1 correlation of
        # absolute returns, and a generator can produce those. Dropping them and
        # reporting how many were dropped is honest; letting numpy average an empty
        # slice returns NaN with four warnings and no count.
        good = vals[np.isfinite(vals)]
        if good.size < 2:
            out[k] = {"value": float("nan"), "se": float("nan"),
                      "n": float(good.size)}
            continue
        out[k] = {"value": float(good.mean()),
                  "se": float(good.std(ddof=1) / np.sqrt(good.size)),
                  "n": float(good.size)}
    out["_pooled_values"] = {"value": float(flat.size), "se": float("nan"),
                             "n": float(n)}
    return out


def facts_error(reference: dict, candidate: dict,
                keys: tuple[str, ...] | None = None) -> dict[str, float]:
    """Signed distance from a reference, fact by fact, in each fact's own units.

    Deliberately returns a dict and not a scalar. There is no defensible way to add
    a kurtosis error to a correlation error, and the summary numbers that appear in
    generative-finance papers do it implicitly by reporting the facts to a common
    number of decimal places.
    """
    keys = tuple(FACTS) if keys is None else keys
    return {k: float(candidate[k]["value"] - reference[k]["value"]) for k in keys}


def within_window_clustering(series, lengths=(16, 32, 64, 128), *,
                             stride: int = 1) -> dict[int, float]:
    """Lag-1 clustering visible *inside* a window, as a function of its length.

    Run this before training anything. A generator can only reproduce dependence
    that fits inside the window it generates, and a persistent volatility process
    can have almost none of its clustering visible at short window lengths — the
    variance moves too slowly for consecutive absolute returns inside the window to
    covary. Asking a model to match a fact the training windows do not contain
    produces a failure that looks like the model's fault.
    """
    s = np.asarray(series, dtype=float).ravel()
    out = {}
    for L in lengths:
        L = int(L)
        if s.size < L:
            continue
        view = np.lib.stride_tricks.sliding_window_view(s, L)[::int(stride)]
        vals = _per_path(view)["acf1_abs"]
        out[L] = float(np.nanmean(vals))
    return out


def iid_bootstrap(series, n_paths: int = 500, length: int = 64, *,
                  seed: int = 0) -> np.ndarray:
    """Resample values with replacement: the marginal exactly, no dependence.

    The control for any claim about fat tails. It cannot fail to reproduce the
    training kurtosis, which is precisely why a matching kurtosis is not evidence
    that a model learned anything.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(series, dtype=float).ravel()
    if s.size < 2:
        raise ValueError("need at least two values to resample")
    return rng.choice(s, size=(int(n_paths), int(length)), replace=True)


def block_bootstrap(series, n_paths: int = 500, length: int = 64, *,
                    block: int = 16, seed: int = 0) -> np.ndarray:
    """Resample contiguous blocks: dependence up to `block`, one parameter.

    Moving-block bootstrap (Künsch 1989). Blocks start at uniformly drawn
    positions and are laid end to end, so dependence survives within a block and is
    broken at every join — which means the clustering it reproduces is a little
    below the truth by construction, at a rate set by how many joins a path
    contains. That known bias is a feature here: it makes the baseline honest
    rather than unbeatable.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(series, dtype=float).ravel()
    block, length, n_paths = int(block), int(length), int(n_paths)
    if block < 1:
        raise ValueError("block length must be at least 1")
    if s.size < block:
        raise ValueError("series shorter than the block length")
    n_blocks = int(np.ceil(length / block))
    starts = rng.integers(0, s.size - block + 1, size=(n_paths, n_blocks))
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    return s[idx].reshape(n_paths, n_blocks * block)[:, :length]
