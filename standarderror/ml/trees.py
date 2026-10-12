r"""Arc III: trees and ensembles.

Episode 11, `tree_bootstrap`: how much of a fitted decision tree -- its root
split, the features near the top, its predictions -- survives a bootstrap of
the training rows.

Episode 12, `importance_null` and `importance_real`: what impurity (MDI)
importance rewards when no feature, or only a coarse one, carries signal.

Episode 13, `oob_study` and `oob_duplicates`: out-of-bag error against
cross-validation and a held-out test set, across forest sizes, and when rows
are not independent.

Episode 14, `boost_curves`: test error and test log loss of gradient
boosting against the number of rounds, with and without shrinkage and label
noise.

Episode 15, `extrapolation` and `diabetes_extrapolation`: tree ensembles
outside the range of their training data.
"""

from __future__ import annotations

import warnings


def _np():
    import numpy as np
    return np


def _bc_split(seed: int = 0, test: float = 0.3):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    d = load_breast_cancer()
    Xa, Xb, ya, yb = train_test_split(d.data, d.target, test_size=test,
                                      random_state=seed, stratify=d.target)
    return Xa, Xb, ya, yb, [str(n) for n in d.feature_names]


# ================================================================ episode 11

def _top_features(tree, depth: int) -> tuple:
    """The set of features used in splits at depth <= `depth`."""
    t = tree.tree_
    out = set()

    def walk(node, d):
        if d > depth or t.feature[node] < 0:
            return
        out.add(int(t.feature[node]))
        walk(t.children_left[node], d + 1)
        walk(t.children_right[node], d + 1)

    walk(0, 0)
    return tuple(sorted(out))


def root_candidates(X, y, names, top: int = 5) -> list[dict]:
    """For each feature, the Gini decrease of its best single split of the
    whole training set: how close the runners-up are to the winner."""
    from sklearn.tree import DecisionTreeClassifier
    out = []
    for j in range(X.shape[1]):
        s = DecisionTreeClassifier(max_depth=1, random_state=0).fit(
            X[:, [j]], y)
        imp = s.tree_.impurity
        w = s.tree_.weighted_n_node_samples
        gain = imp[0] - (w[1] * imp[1] + w[2] * imp[2]) / w[0]
        out.append({"feature": names[j], "gain": float(gain)})
    out.sort(key=lambda r: -r["gain"])
    best = out[0]["gain"]
    for r in out:
        r["relative"] = r["gain"] / best
    return out[:top]


def tree_bootstrap(*, boots: int = 200, depth: int = 3, seed: int = 0
                   ) -> dict:
    """Fit a depth-`depth` tree to `boots` bootstrap resamples of a breast
    cancer training split and record what changes: the root feature, the
    features in the top three levels, the test accuracy and the test
    predictions."""
    np = _np()
    from collections import Counter

    from sklearn.tree import DecisionTreeClassifier
    Xa, Xb, ya, yb, names = _bc_split(seed)
    full = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xa, ya)
    root = int(full.tree_.feature[0])
    g = np.random.default_rng(int(seed))
    roots, sets = Counter(), Counter()
    acc, preds, thr = [], [], []
    in_top = 0
    for _ in range(int(boots)):
        i = g.integers(0, len(ya), len(ya))
        t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(
            Xa[i], ya[i])
        roots[names[int(t.tree_.feature[0])]] += 1
        top = _top_features(t, 2)
        sets[top] += 1
        in_top += root in top
        acc.append(float(t.score(Xb, yb)))
        preds.append(t.predict(Xb))
        if int(t.tree_.feature[0]) == root:
            thr.append(float(t.tree_.threshold[0]))
    P = np.array(preds)
    k = min(len(P), 60)
    pair = [float(np.mean(P[a] != P[b])) for a in range(k)
            for b in range(a + 1, k)]
    return {"names": names, "boots": int(boots), "depth": int(depth),
            "full_root": names[root],
            "full_accuracy": float(full.score(Xb, yb)),
            "roots": roots.most_common(),
            "root_same": roots[names[root]] / boots,
            "root_in_top": in_top / boots,
            "accuracy": acc, "accuracy_sd": float(np.std(acc)),
            "pair_disagreement": float(np.mean(pair)),
            "any_disagree": float(np.mean(P.min(0) != P.max(0))),
            "row_share": P.mean(0), "truth": yb,
            "distinct_sets": len(sets),
            "modal_set_share": sets.most_common(1)[0][1] / boots,
            "threshold": [float(v) for v in np.percentile(thr, [5, 50, 95])],
            "candidates": root_candidates(Xa, ya, names),
            "test_rows": int(len(yb))}


# ================================================================ episode 12

def strobl_task(n: int = 500, *, seed: int = 0, signal: float = 0.0):
    """Five features: binary, 4-, 10- and 20-level categorical (coded as
    integers) and continuous. P(y = 1) = 0.5 + signal * (binary - 0.5): only
    the binary feature can matter, and with signal 0 none does (Strobl et
    al. 2007)."""
    np = _np()
    g = np.random.default_rng(int(seed))
    X = np.c_[g.integers(0, 2, n), g.integers(0, 4, n), g.integers(0, 10, n),
              g.integers(0, 20, n), g.standard_normal(n)].astype(float)
    p = 0.5 + signal * (X[:, 0] - 0.5)
    return X, (g.random(n) < p).astype(int)


STROBL_NAMES = ("binary", "4 levels", "10 levels", "20 levels", "continuous")


def importance_null(*, signal: float = 0.0, reps: int = 20, n: int = 500,
                    leaf: int = 1, trees: int = 100) -> dict:
    """Impurity (MDI) importance on the training data and permutation
    importance on fresh data, averaged over `reps` simulated datasets."""
    np = _np()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    mdi, perm = [], []
    for r in range(int(reps)):
        X, y = strobl_task(n, seed=r, signal=signal)
        Xt, yt = strobl_task(2000, seed=10_000 + r, signal=signal)
        rf = RandomForestClassifier(trees, min_samples_leaf=leaf,
                                    random_state=r, n_jobs=2).fit(X, y)
        mdi.append(rf.feature_importances_)
        perm.append(permutation_importance(
            rf, Xt, yt, n_repeats=5, random_state=0).importances_mean)
    mdi, perm = np.array(mdi), np.array(perm)
    k = np.sqrt(len(mdi))
    return {"names": list(STROBL_NAMES), "signal": signal, "leaf": leaf,
            "mdi": mdi.mean(0), "mdi_se": mdi.std(0, ddof=1) / k,
            "perm": perm.mean(0), "perm_se": perm.std(0, ddof=1) / k,
            "mdi_top_binary": float(np.mean(mdi.argmax(1) == 0)),
            "perm_top_binary": float(np.mean(perm.argmax(1) == 0)),
            "reps": int(reps)}


def importance_real(*, seed: int = 0, trees: int = 300) -> dict:
    """Breast cancer plus a pure-noise continuous column and a pure-noise
    binary column: where each method ranks them among 32."""
    np = _np()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    Xa, Xb, ya, yb, names = _bc_split(seed)
    g = np.random.default_rng(int(seed))

    def add(A):
        return np.c_[A, g.standard_normal(len(A)), g.integers(0, 2, len(A))]

    Xa, Xb = add(Xa), add(Xb)
    rf = RandomForestClassifier(trees, random_state=seed, n_jobs=2).fit(
        Xa, ya)
    mdi = rf.feature_importances_
    perm = permutation_importance(rf, Xb, yb, n_repeats=20,
                                  random_state=0).importances_mean
    rm = list(np.argsort(-mdi))
    rp = list(np.argsort(-perm))
    return {"mdi_rank": (rm.index(30) + 1, rm.index(31) + 1),
            "perm_rank": (rp.index(30) + 1, rp.index(31) + 1),
            "mdi_noise": (float(mdi[30]), float(mdi[31])),
            "perm_nonpositive": int(np.sum(perm <= 0)),
            "features": 32, "accuracy": float(rf.score(Xb, yb))}


# ================================================================ episode 13

def _data(name: str):
    from sklearn.datasets import load_breast_cancer, load_digits
    loader = {"breast cancer": load_breast_cancer, "digits": load_digits}
    return loader[name](return_X_y=True)


def oob_study(*, data: str = "breast cancer", trees=(5, 10, 25, 100, 500),
              reps: int = 10) -> list[dict]:
    """For each forest size: out-of-bag error, 5-fold CV error on the
    training half, and error on the held-out half, over `reps` half splits.
    """
    np = _np()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_score,
        train_test_split,
    )
    X, y = _data(data)
    out = []
    for T in trees:
        o, c, t = [], [], []
        for s in range(int(reps)):
            Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5,
                                              random_state=s, stratify=y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rf = RandomForestClassifier(T, oob_score=True, random_state=s,
                                            n_jobs=2).fit(Xa, ya)
            o.append(1 - rf.oob_score_)
            t.append(1 - rf.score(Xb, yb))
            c.append(1 - cross_val_score(
                RandomForestClassifier(T, random_state=s, n_jobs=2), Xa, ya,
                cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean())
        out.append({"trees": int(T), "oob": float(np.mean(o)),
                    "cv": float(np.mean(c)), "test": float(np.mean(t)),
                    "oob_sd": float(np.std(o)), "data": data})
    return out


def oob_coverage(trees: int, *, n: int = 1000, seed: int = 0) -> float:
    """Mean number of trees for which a row is out of bag, per tree: the
    share of the forest that votes on its OOB prediction (about e^-1)."""
    np = _np()
    g = np.random.default_rng(int(seed))
    inbag = np.zeros((trees, n), bool)
    for t in range(trees):
        inbag[t, g.integers(0, n, n)] = True
    return float((~inbag).mean())


def oob_duplicates(*, reps: int = 5, trees: int = 300) -> dict:
    """Digits, where every training image appears twice (as an augmented
    dataset or a careless merge would leave it): OOB error against the
    error on a held-out half with no duplicates."""
    np = _np()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    X, y = _data("digits")
    o1, t1, o2, t2 = [], [], [], []
    for s in range(int(reps)):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5,
                                          random_state=s, stratify=y)
        rf = RandomForestClassifier(trees, oob_score=True, random_state=s,
                                    n_jobs=2).fit(Xa, ya)
        o1.append(1 - rf.oob_score_)
        t1.append(1 - rf.score(Xb, yb))
        rf2 = RandomForestClassifier(trees, oob_score=True, random_state=s,
                                     n_jobs=2).fit(np.r_[Xa, Xa], np.r_[ya, ya])
        o2.append(1 - rf2.oob_score_)
        t2.append(1 - rf2.score(Xb, yb))
    return {"oob": float(np.mean(o1)), "test": float(np.mean(t1)),
            "oob_dup": float(np.mean(o2)), "test_dup": float(np.mean(t2)),
            "reps": int(reps), "trees": int(trees)}


# ================================================================ episode 14

CHECKPOINTS = (1, 3, 10, 30, 100, 300, 1000, 2000)


def boost_curves(*, data: str = "breast cancer", lr: float = 0.1,
                 depth: int = 3, flip: float = 0.0, rounds: int = 2000,
                 reps: int = 10) -> dict:
    """Gradient boosting on half splits, `flip` of the training labels
    flipped at random: mean test error and test log loss after each
    checkpoint number of rounds, and the training loss."""
    np = _np()
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import log_loss
    from sklearn.model_selection import train_test_split
    X, y = _data(data)
    cps = [c for c in CHECKPOINTS if c <= rounds]
    err = np.zeros((int(reps), len(cps)))
    ll = np.zeros_like(err)
    best = []
    for s in range(int(reps)):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5,
                                          random_state=s, stratify=y)
        g = np.random.default_rng(1000 + s)
        ya = ya.copy()
        m = g.random(len(ya)) < flip
        ya[m] = 1 - ya[m]
        gb = GradientBoostingClassifier(n_estimators=rounds, learning_rate=lr,
                                        max_depth=depth, random_state=s
                                        ).fit(Xa, ya)
        k = 0
        lls = []
        for i, p in enumerate(gb.staged_predict_proba(Xb)):
            q = np.clip(p, 1e-12, 1)
            cur = log_loss(yb, q, labels=[0, 1])
            lls.append(cur)
            if i + 1 == cps[k]:
                err[s, k] = np.mean((p[:, 1] > 0.5) != yb)
                ll[s, k] = cur
                k += 1
                if k == len(cps):
                    break
        best.append(int(np.argmin(lls)) + 1)
    return {"data": data, "lr": lr, "depth": depth, "flip": flip,
            "rounds": cps, "error": err.mean(0), "log_loss": ll.mean(0),
            "best_round_median": float(np.median(best)), "reps": int(reps)}


# ================================================================ episode 15

def trend_task(n: int, lo: float, hi: float, *, seed: int):
    """y = 2 x + sin(2 z) + noise, with x uniform on [lo, hi] and z and two
    further inputs standard normal."""
    np = _np()
    g = np.random.default_rng(int(seed))
    x = g.uniform(lo, hi, n)
    X = np.c_[x, g.standard_normal((n, 3))]
    y = 2 * x + np.sin(2 * X[:, 1]) + 0.3 * g.standard_normal(n)
    return X, y


BANDS = ((0.0, 1.0), (1.0, 1.25), (1.25, 1.5), (1.5, 2.0))


def extrapolation(*, n: int = 500, bands=BANDS, trees: int = 200) -> dict:
    """Train on x in [0, 1]; test in bands of x inside and beyond it. Forest,
    histogram boosting, linear regression, and linear-in-x plus a forest on
    the residuals."""
    np = _np()
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression
    X, y = trend_task(n, 0, 1, seed=0)
    rf = RandomForestRegressor(trees, random_state=0, n_jobs=2).fit(X, y)
    hgb = HistGradientBoostingRegressor(random_state=0).fit(X, y)
    lin = LinearRegression().fit(X, y)
    trend = LinearRegression().fit(X[:, :1], y)
    res = RandomForestRegressor(trees, random_state=0, n_jobs=2).fit(
        X, y - trend.predict(X[:, :1]))
    rows = []
    for lo, hi in bands:
        Xt, yt = trend_task(2000, lo, hi, seed=1)
        pred = {"forest": rf.predict(Xt), "boosting": hgb.predict(Xt),
                "linear": lin.predict(Xt),
                "trend + forest": trend.predict(Xt[:, :1]) + res.predict(Xt)}
        rows.append({"band": (lo, hi),
                     **{k: float(np.mean((p - yt) ** 2))
                        for k, p in pred.items()},
                     "forest_max": float(rf.predict(Xt).max()),
                     "truth_mean": float(yt.mean())})
    curve_x = np.linspace(0, 2, 81)
    Z = np.zeros((len(curve_x), 4))
    Z[:, 0] = curve_x
    return {"rows": rows, "curve_x": curve_x,
            "curve": {"forest": rf.predict(Z), "boosting": hgb.predict(Z),
                      "linear": lin.predict(Z),
                      "trend + forest": trend.predict(Z[:, :1]) +
                      res.predict(Z),
                      "truth": 2 * curve_x},
            "train_x_max": float(X[:, 0].max()),
            "train_y_max": float(y.max())}


def diabetes_extrapolation(*, quantile: float = 0.9, trees: int = 300
                           ) -> dict:
    """Diabetes: train on patients with bmi at or below its `quantile`, test
    on those above, against a random split of the same sizes."""
    np = _np()
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    X, y = load_diabetes(return_X_y=True)
    b = X[:, 2]
    tr = b <= np.quantile(b, quantile)

    def score(train, test):
        out = {}
        for name, m in (("forest", RandomForestRegressor(
                trees, random_state=0, n_jobs=2)),
                ("linear", LinearRegression())):
            m.fit(X[train], y[train])
            p = m.predict(X[test])
            out[name] = {"mse": float(np.mean((p - y[test]) ** 2)),
                         "bias": float(p.mean() - y[test].mean())}
        return out

    g = np.random.default_rng(0)
    perm = g.permutation(len(y))
    rtr = np.zeros(len(y), bool)
    rtr[perm[:int(tr.sum())]] = True
    return {"quantile": quantile, "beyond": score(tr, ~tr),
            "random": score(rtr, ~rtr), "test_rows": int((~tr).sum())}
