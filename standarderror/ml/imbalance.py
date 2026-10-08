r"""Class imbalance: what resampling changes, and what a threshold changes.

**Episode 5.** "Resample the data until the classes are balanced" is the
first fix most courses teach for a rare positive class. Measured on a
simulated task at 2% prevalence, against class weights and against simply
moving the decision threshold of the unweighted model:

* For logistic regression, oversampling, SMOTE and class weights leave the
  *ranking* where it was -- ROC AUC, average precision and precision at a
  fixed recall all within noise of the plain model -- and undersampling,
  which discards most of the majority class, makes it slightly worse.
* They move the *probabilities*: the mean predicted probability goes from
  the true 0.02 to about 0.2, and log loss multiplies about fivefold. The
  standard prior correction undoes that almost exactly.
* What they change in the *decisions* at the default 0.5 cut -- recall from
  0.2 to over 0.8 -- the plain model does by moving its threshold, at the
  same precision.
* For boosted trees resampling is not neutral: duplicated or interpolated
  minority rows are memorised, and ranking gets worse.

References: Chawla et al., "SMOTE: synthetic minority over-sampling
technique", *JAIR* (2002); Elkan, "The foundations of cost-sensitive
learning", *IJCAI* (2001), for why a threshold is the cost; van den Goorbergh
et al., "The harm of class imbalance corrections for risk prediction
models", *JAMIA* (2022), the clinical version of the same finding.
"""

from __future__ import annotations

import math


def _np():
    import numpy as np
    return np


class ImbalancedTask:
    """A rare positive class in `d` dimensions: positives are shifted on
    three features and more spread on a fourth, so the boundary is not
    linear but a linear model does well. `prevalence` is the positive
    rate in both training and test draws."""

    def __init__(self, d: int = 8, prevalence: float = 0.02):
        self.d, self.prevalence = int(d), float(prevalence)

    def draw(self, n: int, *, seed: int):
        np = _np()
        rng = np.random.default_rng(int(seed))
        y = (rng.random(int(n)) < self.prevalence).astype(int)
        X = rng.standard_normal((int(n), self.d))
        X[y == 1, :3] += 1.2
        X[y == 1, 3] *= 1.8
        return X, y


# ------------------------------------------------------------ resamplers

def oversample(X, y, rng):
    """Duplicate minority rows at random until the classes are equal."""
    np = _np()
    pos = np.where(y == 1)[0]
    add = rng.choice(pos, int((y == 0).sum() - len(pos)))
    return np.vstack([X, X[add]]), np.r_[y, y[add]]


def undersample(X, y, rng):
    """Discard majority rows at random until the classes are equal."""
    np = _np()
    keep = rng.choice(np.where(y == 0)[0], int((y == 1).sum()),
                      replace=False)
    idx = np.r_[np.where(y == 1)[0], keep]
    return X[idx], y[idx]


def smote(X, y, rng, k: int = 5):
    """SMOTE as Chawla et al. describe it: each synthetic row is a random
    point on the segment from a minority row to one of its `k` nearest
    minority neighbours."""
    np = _np()
    from sklearn.neighbors import NearestNeighbors
    P = X[y == 1]
    need = int((y == 0).sum() - len(P))
    nb = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(
        P, return_distance=False)[:, 1:]
    i = rng.integers(0, len(P), need)
    j = nb[i, rng.integers(0, k, need)]
    lam = rng.random((need, 1))
    return np.vstack([X, P[i] + lam * (P[j] - P[i])]), np.r_[y, np.ones(
        need, int)]


METHODS = ("plain", "oversample", "SMOTE", "undersample", "class weights")


def _models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    return {
        "logistic": lambda **kw: LogisticRegression(max_iter=2000, **kw),
        "boosted trees": lambda **kw: HistGradientBoostingClassifier(
            random_state=0, **kw),
    }


def fit_predict(model: str, method: str, X, y, X_test, *, seed: int):
    """Train `model` on (X, y) after `method`, return test probabilities."""
    np = _np()
    rng = np.random.default_rng(int(seed))
    make = _models()[model]
    kw = {}
    if method == "oversample":
        X, y = oversample(X, y, rng)
    elif method == "SMOTE":
        X, y = smote(X, y, rng)
    elif method == "undersample":
        X, y = undersample(X, y, rng)
    elif method == "class weights":
        kw = {"class_weight": "balanced"}
    return make(**kw).fit(X, y).predict_proba(X_test)[:, 1]


# ------------------------------------------------------------ metrics

def threshold_for_recall(p, y, recall: float) -> float:
    """The largest threshold at which at least `recall` of positives are
    flagged."""
    np = _np()
    return float(np.quantile(p[y == 1], 1.0 - float(recall)))


def at_threshold(p, y, t: float) -> dict:
    flag = p >= t
    tp = int((flag & (y == 1)).sum())
    return {"recall": tp / int((y == 1).sum()),
            "precision": tp / max(int(flag.sum()), 1),
            "flagged": float(flag.mean())}


def scores(p, y) -> dict:
    """Ranking, probability and decision metrics for one set of
    predictions."""
    np = _np()
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    q = np.clip(p, 1e-7, 1 - 1e-7)
    half = at_threshold(p, y, 0.5)
    return {"auc": float(roc_auc_score(y, p)),
            "ap": float(average_precision_score(y, p)),
            "mean_p": float(p.mean()),
            "log_loss": float(log_loss(y, q)),
            "brier": float(brier_score_loss(y, p)),
            "recall_half": half["recall"],
            "precision_half": half["precision"],
            "precision_r80": at_threshold(
                p, y, threshold_for_recall(p, y, 0.8))["precision"]}


def prior_correct(p, trained_rate: float, true_rate: float):
    """Map probabilities from a model trained at positive rate
    `trained_rate` back to `true_rate`: multiply the odds by the ratio of
    the two prior odds (Elkan 2001; Saerens et al. 2002)."""
    np = _np()
    odds = p / np.clip(1 - p, 1e-12, None)
    k = (true_rate / (1 - true_rate)) / (trained_rate / (1 - trained_rate))
    o = odds * k
    return o / (1 + o)


# ------------------------------------------------------------ studies

def compare(*, model: str = "logistic", n: int = 5000, test: int = 200_000,
            reps: int = 20, prevalence: float = 0.02) -> dict:
    """Every method on `reps` independent training sets, scored on one
    large test set at the same prevalence. Each replicate's methods share
    the training set, so differences from the plain model are paired."""
    np = _np()
    task = ImbalancedTask(prevalence=prevalence)
    Xt, yt = task.draw(test, seed=99)
    rows = {m: [] for m in METHODS}
    matched = {m: [] for m in METHODS}
    corrected = []
    for r in range(int(reps)):
        X, y = task.draw(n, seed=r)
        preds = {m: fit_predict(model, m, X, y, Xt, seed=r) for m in METHODS}
        for m, p in preds.items():
            rows[m].append(scores(p, yt))
            # The plain model, its threshold moved to this method's recall
            # at 0.5: does a threshold do the same job?
            rec = at_threshold(p, yt, 0.5)["recall"]
            t = threshold_for_recall(preds["plain"], yt, rec)
            matched[m].append({
                "recall": rec,
                "method_precision": at_threshold(p, yt, 0.5)["precision"],
                "plain_precision": at_threshold(preds["plain"], yt,
                                                t)["precision"],
                "plain_threshold": t})
        pc = prior_correct(preds["oversample"], 0.5, float(y.mean()))
        corrected.append(scores(pc, yt))

    def summary(lst):
        keys = lst[0].keys()
        return {k: float(np.mean([d[k] for d in lst])) for k in keys}

    def paired(m, key):
        d = np.array([a[key] - b[key] for a, b in zip(rows[m],
                                                       rows["plain"])])
        return {"mean": float(d.mean()),
                "se": float(d.std(ddof=1) / math.sqrt(len(d)))}

    return {"model": model, "prevalence": prevalence, "n": n,
            "test": test, "reps": int(reps),
            "summary": {m: summary(v) for m, v in rows.items()},
            "matched": {m: summary(v) for m, v in matched.items()},
            "corrected": summary(corrected),
            "paired": {m: {k: paired(m, k) for k in ("auc", "ap",
                                                     "precision_r80")}
                       for m in METHODS if m != "plain"}}


def digits_eights(*, reps: int = 30, seed: int = 0) -> dict:
    """A real check: "is this digit an 8?" on the UCI digits, about 10%
    positive, logistic regression on standardised pixels, 30 stratified
    half splits."""
    np = _np()
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from standarderror.ml import evaluation as ev
    X, y = ev.digits()
    y = (y == 8).astype(int)
    rows = {m: [] for m in METHODS}
    for r in range(int(reps)):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5,
                                          random_state=int(seed) + r,
                                          stratify=y)
        sc = StandardScaler().fit(Xa)
        Xa, Xb = sc.transform(Xa), sc.transform(Xb)
        for m in METHODS:
            rows[m].append(scores(fit_predict("logistic", m, Xa, ya, Xb,
                                              seed=r), yb))
    return {"prevalence": float(y.mean()), "reps": int(reps),
            "summary": {m: {k: float(np.mean([d[k] for d in v]))
                            for k in v[0]} for m, v in rows.items()}}
