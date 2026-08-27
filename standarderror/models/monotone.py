"""What a monotonicity constraint costs, and what decides whether it costs anything.

The question
------------
Gradient boosting libraries let you require that a feature's effect be monotone —
that higher debt-to-income can only raise modelled default risk, never lower it.
Credit risk wants this because a supervisor can read it. The obvious worry is what
it costs in accuracy, and the natural measure is the relative change in a metric
between an unconstrained and a constrained fit, which Koklev (2025) names the
Price of Monotonicity.

Measured on real data that quantity confounds two things, and the whole point of
this module is to separate them by generating data whose truth is known:

**How much the constraint binds** is set by how much freedom it removes relative
to how much data is available to use that freedom. Constraining nine of twelve
features on five hundred rows is a large intervention; the same nine on twenty
thousand rows is almost none.

**Which direction it moves the metric** is set by whether the constraint is true.
A correct constraint on a monotone truth is regularisation and helps. A constraint
the data disagrees with is bias and hurts. Both effects scale with the first
quantity, so a benchmark that varies only the first sees magnitudes without signs.

`coverage_sweep` varies the first with the truth held monotone. `violation_sweep`
varies the second with coverage held fixed. Between them they are a map that a
handful of real datasets are a handful of points on.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SIGNS", "make_credit_like", "fit_pair", "pom", "paired_bootstrap_pom",
           "coverage_sweep", "violation_sweep", "split_variance", "PARAMS"]

# Six features that raise risk and six that lower it, in descending strength, so
# that "constrain the first k" walks from the informative features to the weak
# ones the way a modeller working down a domain-knowledge list would.
SIGNS = (1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1)
_WEIGHTS = np.array([0.9, 0.7, 0.6, 0.5, 0.4, 0.3,
                     -0.9, -0.7, -0.6, -0.5, -0.4, -0.3])

PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9,
              colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
              n_jobs=4, verbosity=0)


def make_credit_like(n: int, *, violate: float = 0.0, seed: int = 0,
                     base_rate: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """A binary target whose truth is monotone in all twelve features.

    `violate` bends the first feature by adding `violate * (x^2 - 1)` to the
    linear index, which makes its true effect U-shaped while leaving the other
    eleven monotone. It is the single knob that controls how wrong a
    correctly-*signed* constraint on that feature is, and at `violate=0` the sign
    convention in `SIGNS` is exactly true.

    The quadratic is centred so that raising `violate` changes the shape of the
    relationship without changing the average default rate much — otherwise the
    sweep would confound misspecification with class balance.
    """
    if violate < 0:
        raise ValueError("violate is a magnitude; a negative value just relabels "
                         "which side of the parabola is steep")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(int(n), _WEIGHTS.size))
    lin = X @ _WEIGHTS + violate * (X[:, 0] ** 2 - 1.0)
    y = (rng.random(int(n)) < 1.0 / (1.0 + np.exp(-(lin - base_rate)))).astype(int)
    return X, y


def _constraint(k: int) -> tuple[int, ...]:
    """Constrain the first `k` features with their true signs, free the rest."""
    if not 0 <= k <= len(SIGNS):
        raise ValueError(f"k must be in 0..{len(SIGNS)}")
    return tuple(SIGNS[j] if j < k else 0 for j in range(len(SIGNS)))


def fit_pair(X_tr, y_tr, *, k: int | None = None, cst=None, seed: int = 0,
             params: dict | None = None):
    """An unconstrained fit and a constrained one, identical in everything else.

    Same seed, same hyperparameters, same rows. The paired design is the reason a
    difference of a few tenths of a percent means anything at all: run the two
    fits with different seeds and the seed alone moves AUC by more than the effect
    being measured.
    """
    import xgboost as xgb
    if (k is None) == (cst is None):
        raise ValueError("pass exactly one of k or cst")
    cst = _constraint(k) if cst is None else tuple(cst)
    p = dict(PARAMS if params is None else params, random_state=seed)
    free = xgb.XGBClassifier(**p).fit(X_tr, y_tr)
    tied = xgb.XGBClassifier(monotone_constraints=cst, **p).fit(X_tr, y_tr)
    return free, tied


def pom(y, p_free, p_tied, metric: str = "auc") -> float:
    """Price of Monotonicity, in percent, positive meaning the constraint cost.

    Sign convention follows the paper for both metric directions: a positive value
    always means the constrained model is worse. Getting this backwards for the
    lower-is-better metric is the single easiest way to publish a table whose every
    cell has the wrong sign, so the two branches are written out rather than folded
    into a clever expression.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score
    if metric == "auc":
        a, b = roc_auc_score(y, p_free), roc_auc_score(y, p_tied)
        return 100.0 * (a - b) / a                     # higher is better
    if metric == "brier":
        a, b = brier_score_loss(y, p_free), brier_score_loss(y, p_tied)
        return 100.0 * (b - a) / a                     # lower is better
    raise ValueError("metric must be 'auc' or 'brier'")


def paired_bootstrap_pom(y, p_free, p_tied, *, metric: str = "auc", reps: int = 500,
                         rng=None) -> dict:
    """Resample test rows, keeping both models' predictions paired on each row.

    This is the paper's uncertainty and it is the *test set's* sampling error with
    the fitted models held fixed. It is not the variance of refitting, and
    `split_variance` measures that separately so the two can be compared rather
    than assumed to be the same size.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(y)
    n = y.size
    out = np.empty(int(reps))
    for b in range(int(reps)):
        i = rng.integers(0, n, n)
        # A resample with one class absent has no AUC; redrawing is the standard
        # fix and the paper does the same.
        while y[i].min() == y[i].max():
            i = rng.integers(0, n, n)
        out[b] = pom(y[i], np.asarray(p_free)[i], np.asarray(p_tied)[i], metric)
    return {"mean": float(out.mean()),
            "lo": float(np.quantile(out, 0.025)),
            "hi": float(np.quantile(out, 0.975)),
            "significant": bool(np.quantile(out, 0.025) > 0
                                or np.quantile(out, 0.975) < 0),
            "reps": int(reps)}


def _split(X, y, seed):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def coverage_sweep(n_train, coverages, *, repeats: int = 6, metric: str = "auc",
                   seed: int = 0) -> dict:
    """PoM over sample size and share of features constrained, truth monotone.

    With the truth monotone every constraint here is correct, so whatever this
    surface shows is the effect of removing flexibility alone, with no
    misspecification mixed in.
    """
    out = {}
    for n in n_train:
        for k in coverages:
            vals = []
            for r in range(int(repeats)):
                X, y = make_credit_like(int(n / 0.7), seed=seed + 1000 * r + n)
                Xtr, Xte, ytr, yte = _split(X, y, r)
                free, tied = fit_pair(Xtr, ytr, k=k, seed=r)
                vals.append(pom(yte, free.predict_proba(Xte)[:, 1],
                                tied.predict_proba(Xte)[:, 1], metric))
            out[(int(n), int(k))] = {"mean": float(np.mean(vals)),
                                     "sd": float(np.std(vals, ddof=1)),
                                     "repeats": int(repeats)}
    return out


def violation_sweep(violations, *, n_train: int = 5000, repeats: int = 8,
                    metric: str = "auc", seed: int = 0) -> list[dict]:
    """PoM under correct and under flipped signs, as the truth stops being monotone.

    Both regimes are run on the *same* generated data and the same splits, so the
    ratio between them is a paired quantity rather than two separate experiments
    divided by each other.
    """
    correct = tuple(SIGNS)
    wrong = tuple(-s for s in SIGNS)
    rows = []
    for v in violations:
        c, w = [], []
        for r in range(int(repeats)):
            X, y = make_credit_like(int(n_train / 0.7), violate=float(v),
                                    seed=seed + 137 * r + int(float(v) * 100))
            Xtr, Xte, ytr, yte = _split(X, y, r)
            free, tied = fit_pair(Xtr, ytr, cst=correct, seed=r)
            _, flipped = fit_pair(Xtr, ytr, cst=wrong, seed=r)
            pf = free.predict_proba(Xte)[:, 1]
            c.append(pom(yte, pf, tied.predict_proba(Xte)[:, 1], metric))
            w.append(pom(yte, pf, flipped.predict_proba(Xte)[:, 1], metric))
        rows.append({"violation": float(v), "correct": float(np.mean(c)),
                     "correct_sd": float(np.std(c, ddof=1)),
                     "wrong": float(np.mean(w)),
                     "wrong_sd": float(np.std(w, ddof=1)), "repeats": int(repeats)})
    return rows


def split_variance(n_train: int, *, k: int = 12, splits: int = 40,
                   boot: int = 400, metric: str = "auc", seed: int = 0) -> dict:
    """How much PoM moves on the split, against how much the bootstrap says.

    The paper reports uncertainty from resampling the test set with the split
    fixed. If refitting on a different split moved PoM more than that interval is
    wide, the reported significance would be measuring the smaller of two
    variances. This function is here to check that, and it is worth running before
    assuming either answer.
    """
    X, y = make_credit_like(int(n_train / 0.7) * 2, seed=seed + 1)
    per_split, half = [], []
    for s in range(int(splits)):
        Xtr, Xte, ytr, yte = _split(X, y, s)
        free, tied = fit_pair(Xtr, ytr, k=k, seed=s)
        pf = free.predict_proba(Xte)[:, 1]
        pt = tied.predict_proba(Xte)[:, 1]
        per_split.append(pom(yte, pf, pt, metric))
        ci = paired_bootstrap_pom(yte, pf, pt, metric=metric, reps=int(boot),
                                  rng=np.random.default_rng(100 + s))
        half.append((ci["hi"] - ci["lo"]) / 2.0)
    per_split = np.array(per_split)
    return {"splits": int(splits), "mean": float(per_split.mean()),
            "across_split_sd": float(per_split.std(ddof=1)),
            "bootstrap_half_width": float(np.median(half)),
            "ratio": float(per_split.std(ddof=1) / np.median(half)),
            "min": float(per_split.min()), "max": float(per_split.max())}
