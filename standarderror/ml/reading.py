r"""Arc IV: reading what a model learned.

Episode 16, `copy_importance` and `off_support`: permutation importance when
a feature has a correlated copy, and the rows a permutation invents.

Episode 17, `pdp`, `ale` and `pdp_vs_ale`: partial dependence against
accumulated local effects when the inputs are correlated, with the true
effect known.

Episode 18, `exact_shapley` and `background_study`: Shapley values computed
exactly by enumerating coalitions, and how they move with the background
distribution they are measured against.

Episode 19, `shortcut_study`: a planted shortcut in the digits, and what the
test score says about it.

Episode 20, `rashomon`: how often models of near-equal accuracy disagree.
"""

from __future__ import annotations

import itertools
import math
import warnings


def _np():
    import numpy as np
    return np


# ================================================================ episode 16

def _diabetes_split(seed: int = 0):
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    d = load_diabetes()
    Xa, Xb, ya, yb = train_test_split(d.data, d.target, test_size=0.3,
                                      random_state=seed)
    return Xa, Xb, ya, yb, [str(n) for n in d.feature_names]


def _forest(seed: int = 0, trees: int = 300):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(trees, min_samples_leaf=5, random_state=seed,
                                 n_jobs=2)


def copy_importance(*, rhos=(None, 0.8, 0.95, 1.0), feature: str = "bmi",
                    seeds=(0, 1, 2), repeats: int = 20) -> list[dict]:
    """Diabetes, random forest. Add a copy of `feature` correlated with it at
    rho (None: no copy) and measure the permutation importance (drop in
    test R^2) of the feature and of its copy, and the drop-column importance
    of the feature and of both together. Averaged over `seeds` splits."""
    np = _np()
    from sklearn.inspection import permutation_importance
    out = []
    for rho in rhos:
        acc = {"rho": rho, "feature": [], "copy": [], "r2": [],
               "drop_feature": [], "drop_both": []}
        for s in seeds:
            Xa, Xb, ya, yb, names = _diabetes_split(s)
            j = names.index(feature)
            g = np.random.default_rng(100 + s)
            if rho is not None:
                sd = Xa[:, j].std()

                def cp(A, g=g, rho=rho, sd=sd, j=j):
                    e = g.standard_normal(len(A)) * sd
                    return np.c_[A, rho * A[:, j] + math.sqrt(1 - rho ** 2) * e]

                Xa, Xb = cp(Xa), cp(Xb)
            rf = _forest(s).fit(Xa, ya)
            base = rf.score(Xb, yb)
            pi = permutation_importance(rf, Xb, yb, n_repeats=repeats,
                                        random_state=0).importances_mean
            acc["r2"].append(base)
            acc["feature"].append(pi[j])
            acc["copy"].append(pi[-1] if rho is not None else 0.0)
            keep = [k for k in range(Xa.shape[1]) if k != j]
            acc["drop_feature"].append(
                base - _forest(s).fit(Xa[:, keep], ya).score(Xb[:, keep], yb))
            both = [k for k in keep if rho is None or k != Xa.shape[1] - 1]
            acc["drop_both"].append(
                base - _forest(s).fit(Xa[:, both], ya).score(Xb[:, both], yb))
        out.append({k: (v if k == "rho" else float(np.mean(v)))
                    for k, v in acc.items()})
    return out


def split_spread(*, feature: str = "bmi", seeds=(0, 1, 2),
                 repeats: int = 20) -> list[dict]:
    """Per split, with no copy: test R^2, the permutation importance of
    `feature` and its drop-column importance. How noisy the averages are."""
    from sklearn.inspection import permutation_importance
    out = []
    for s in seeds:
        Xa, Xb, ya, yb, names = _diabetes_split(s)
        j = names.index(feature)
        rf = _forest(s).fit(Xa, ya)
        base = rf.score(Xb, yb)
        pi = permutation_importance(rf, Xb, yb, n_repeats=repeats,
                                    random_state=0).importances_mean[j]
        keep = [k for k in range(Xa.shape[1]) if k != j]
        drop = base - _forest(s).fit(Xa[:, keep], ya).score(Xb[:, keep], yb)
        out.append({"seed": s, "r2": float(base), "permute": float(pi),
                    "drop": float(drop)})
    return out


def off_support(*, a: str = "s1", b: str = "s2", level: float = 0.99,
                seed: int = 0) -> dict:
    """Permute feature `a` of the diabetes data and count the rows whose
    (a, b) pair lies outside the region holding `level` of the real pairs,
    by Mahalanobis distance."""
    np = _np()
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    names = [str(n) for n in d.feature_names]
    i, j = names.index(a), names.index(b)
    P = d.data[:, [i, j]]
    mu, S = P.mean(0), np.cov(P.T)
    Si = np.linalg.inv(S)

    def md(Z):
        D = Z - mu
        return np.einsum("ij,jk,ik->i", D, Si, D)

    cut = np.quantile(md(P), level)
    g = np.random.default_rng(int(seed))
    shares = []
    for _ in range(50):
        Q = P.copy()
        Q[:, 0] = g.permutation(Q[:, 0])
        shares.append(np.mean(md(Q) > cut))
    return {"a": a, "b": b, "corr": float(np.corrcoef(P.T)[0, 1]),
            "level": level, "outside": float(np.mean(shares)),
            "expected": 1 - level, "pairs": P, "permuted": Q}


# ================================================================ episode 17

def correlated_pair(n: int = 1000, *, noise: float | None = 0.05,
                    seed: int = 0):
    """x1 uniform on [0, 1]; x2 = x1 + noise * N(0, 1), or an independent
    uniform when `noise` is None. y = x1 + 2 x2^2 + 0.1 N(0, 1): the true
    effect of x1 is linear with slope 1, that of x2 is 2 x2^2."""
    np = _np()
    g = np.random.default_rng(int(seed))
    x1 = g.uniform(0, 1, n)
    x2 = g.uniform(0, 1, n) if noise is None else \
        x1 + noise * g.standard_normal(n)
    y = x1 + 2 * x2 ** 2 + 0.1 * g.standard_normal(n)
    return np.c_[x1, x2], y


def pdp(model, X, j: int, grid):
    """Partial dependence: the mean prediction with feature j set to each
    grid value for every row."""
    np = _np()
    out = []
    for v in grid:
        Z = X.copy()
        Z[:, j] = v
        out.append(model.predict(Z).mean())
    return np.array(out)


def ale(model, X, j: int, grid, bins: int = 20):
    """Accumulated local effects (Apley and Zhu 2020): within each quantile
    bin of feature j, the mean change in prediction as j moves from the
    bin's lower to its upper edge for the rows in that bin, accumulated
    and interpolated to `grid`."""
    np = _np()
    edges = np.unique(np.quantile(X[:, j], np.linspace(0, 1, bins + 1)))
    eff = [0.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (X[:, j] >= lo) & (X[:, j] <= hi)
        A, B = X[idx].copy(), X[idx].copy()
        A[:, j], B[:, j] = lo, hi
        eff.append(eff[-1] + float(np.mean(model.predict(B) -
                                           model.predict(A))))
    return np.interp(grid, edges, np.array(eff))


def _centre(v):
    return v - v.mean()


def pdp_vs_ale(*, noises=(None, 0.3, 0.1, 0.05), grid_n: int = 19,
               seeds=(0, 1, 2)) -> list[dict]:
    """For each correlation setting and two models, the slope fitted to the
    PDP and to the ALE of x1 (truth: 1), and each curve's RMS distance from
    the true centred effect of x1 and of x2."""
    np = _np()
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    grid = np.linspace(0.05, 0.95, grid_n)
    truth = {0: _centre(grid), 1: _centre(2 * grid ** 2)}
    out = []
    for noise in noises:
        for name in ("forest", "boosting"):
            acc = {k: [] for k in ("pdp_slope", "ale_slope", "pdp_rmse",
                                   "ale_rmse", "pdp_rmse2", "ale_rmse2",
                                   "corr")}
            for s in seeds:
                X, y = correlated_pair(noise=noise, seed=s)
                m = (RandomForestRegressor(300, min_samples_leaf=5,
                                           random_state=s, n_jobs=2)
                     if name == "forest" else
                     HistGradientBoostingRegressor(random_state=s)).fit(X, y)
                for j, tag in ((0, ""), (1, "2")):
                    p = _centre(pdp(m, X, j, grid))
                    a = _centre(ale(m, X, j, grid))
                    acc["pdp_rmse" + tag].append(
                        np.sqrt(np.mean((p - truth[j]) ** 2)))
                    acc["ale_rmse" + tag].append(
                        np.sqrt(np.mean((a - truth[j]) ** 2)))
                    if j == 0:
                        acc["pdp_slope"].append(np.polyfit(grid, p, 1)[0])
                        acc["ale_slope"].append(np.polyfit(grid, a, 1)[0])
                acc["corr"].append(np.corrcoef(X.T)[0, 1])
            out.append({"noise": noise, "model": name,
                        **{k: float(np.mean(v)) for k, v in acc.items()},
                        "pdp_slopes": [float(v) for v in acc["pdp_slope"]],
                        "ale_slopes": [float(v) for v in acc["ale_slope"]]})
    return out


def pdp_curves(*, noise: float = 0.05, seed: int = 0, grid_n: int = 19
               ) -> dict:
    """The PDP and ALE of x1 for a forest in one correlated dataset, with
    the true effect, for plotting."""
    np = _np()
    from sklearn.ensemble import RandomForestRegressor
    grid = np.linspace(0.05, 0.95, grid_n)
    X, y = correlated_pair(noise=noise, seed=seed)
    m = RandomForestRegressor(300, min_samples_leaf=5, random_state=seed,
                              n_jobs=2).fit(X, y)
    return {"grid": grid, "pdp": _centre(pdp(m, X, 0, grid)),
            "ale": _centre(ale(m, X, 0, grid)), "truth": _centre(grid),
            "X": X}


# ================================================================ episode 18

def exact_shapley(f, x, background):
    """Interventional Shapley values of f at x against `background` rows,
    by enumerating all 2^p coalitions. Returns (phi, E_B f, f(x)); phi sums
    to f(x) - E_B f exactly."""
    np = _np()
    p = len(x)
    val = {}
    for k in range(p + 1):
        for S in itertools.combinations(range(p), k):
            Z = background.copy()
            Z[:, list(S)] = x[list(S)]
            val[S] = float(np.mean(f(Z)))
    phi = np.zeros(p)
    for i in range(p):
        for S, v in val.items():
            if i in S:
                continue
            w = (math.factorial(len(S)) * math.factorial(p - len(S) - 1) /
                 math.factorial(p))
            phi[i] += w * (val[tuple(sorted(S + (i,)))] - v)
    return phi, val[()], val[tuple(range(p))]


SHAP_FEATURES = ("worst radius", "worst concave points", "worst texture",
                 "mean smoothness", "worst symmetry")


def background_study(*, step: int = 3, seed: int = 0) -> dict:
    """Breast cancer, five features, gradient boosting; log-odds explained.
    Exact Shapley values for every `step`-th test row against four
    backgrounds, compared with the all-training-rows background."""
    np = _np()
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    d = load_breast_cancer()
    names = [str(n) for n in d.feature_names]
    cols = [names.index(c) for c in SHAP_FEATURES]
    X, y = d.data[:, cols], d.target
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=seed,
                                      stratify=y)
    m = HistGradientBoostingClassifier(random_state=seed).fit(Xa, ya)

    def f(Z):
        p = np.clip(m.predict_proba(Z)[:, 1], 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    bgs = {"all training rows": Xa, "benign only": Xa[ya == 1],
           "malignant only": Xa[ya == 0],
           "feature means": Xa.mean(0, keepdims=True)}
    rows = list(range(0, len(Xb), int(step)))
    phi = {k: [] for k in bgs}
    base = {}
    err = 0.0
    for i in rows:
        for k, B in bgs.items():
            v, b0, fx = exact_shapley(f, Xb[i], B)
            phi[k].append(v)
            base[k] = b0
            err = max(err, abs(v.sum() - (fx - b0)))
    phi = {k: np.array(v) for k, v in phi.items()}
    ref = phi["all training rows"]
    cmp = {}
    for k, v in phi.items():
        cmp[k] = {"base": base[k],
                  "top_differs": float(np.mean(np.argmax(np.abs(v), 1) !=
                                               np.argmax(np.abs(ref), 1))),
                  "sign_flips": float(np.mean(np.sign(v) != np.sign(ref))),
                  "mean_abs": np.abs(v).mean(0)}
    return {"features": list(SHAP_FEATURES), "rows": len(rows),
            "phi": phi, "compare": cmp, "efficiency_error": err,
            "accuracy": float(m.score(Xb, yb))}


def linear_shapley_check(*, seed: int = 0) -> float:
    """For a linear model with an independent-feature (interventional)
    background, Shapley values are beta_j (x_j - mean_j). Largest gap."""
    np = _np()
    g = np.random.default_rng(int(seed))
    B = g.standard_normal((200, 4))
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    x = g.standard_normal(4)
    phi, _, _ = exact_shapley(lambda Z: Z @ beta + 0.7, x, B)
    return float(np.max(np.abs(phi - beta * (x - B.mean(0)))))


# ================================================================ episode 19

SHORTCUT = {3: 0, 5: 39, 8: 56, 9: 32}   # class -> a pixel blank in the data


def plant(X, y, rate: float, *, seed: int, wrong: bool = False):
    """Set the class's marker pixel to full intensity on `rate` of that
    class's images; with `wrong`, put each marker on a different class
    instead (3's marker on the 5s, and so on)."""
    np = _np()
    g = np.random.default_rng(int(seed))
    X = X.copy()
    swap = {3: 5, 5: 3, 8: 9, 9: 8}
    for cls, pix in SHORTCUT.items():
        tgt = swap[cls] if wrong else cls
        idx = np.where(y == tgt)[0]
        X[idx[g.random(len(idx)) < rate], pix] = 16
    return X


def shortcut_study(*, rates=(0.0, 0.5, 0.9, 1.0), seeds=(0, 1, 2)
                   ) -> list[dict]:
    """Digits, half splits. Train with the marker on `rate` of the marked
    classes; score on a test half carrying the marker at the same rate, on
    a clean test half, and on one with the markers on the wrong classes."""
    np = _np()
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    out = []
    for rate in rates:
        acc = {k: [] for k in ("same", "clean", "wrong", "marked_clean",
                               "marker_importance", "marker_rank")}
        for s in seeds:
            Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5,
                                              random_state=s, stratify=y)
            Xs = plant(Xa, ya, rate, seed=s)
            m = RandomForestClassifier(300, random_state=s, n_jobs=2).fit(
                Xs, ya)
            same = plant(Xb, yb, rate, seed=100 + s)
            acc["same"].append(m.score(same, yb))
            acc["clean"].append(m.score(Xb, yb))
            acc["wrong"].append(m.score(plant(Xb, yb, 1.0, seed=200 + s,
                                              wrong=True), yb))
            mk = np.isin(yb, list(SHORTCUT))
            acc["marked_clean"].append(m.score(Xb[mk], yb[mk]))
            pi = permutation_importance(m, same, yb, n_repeats=5,
                                        random_state=0).importances_mean
            acc["marker_importance"].append(pi[list(SHORTCUT.values())].sum())
            order = list(np.argsort(-pi))
            acc["marker_rank"].append(min(order.index(p)
                                          for p in SHORTCUT.values()) + 1)
        out.append({"rate": rate, **{k: float(np.mean(v))
                                     for k, v in acc.items()}})
    return out


# ================================================================ episode 20

def _zoo(seed_count: int = 5):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    z = {"logistic": make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=3000)),
         "QDA": QuadraticDiscriminantAnalysis(reg_param=0.01),
         "SVM": make_pipeline(StandardScaler(),
                              SVC(probability=True, random_state=0)),
         "boosting": HistGradientBoostingClassifier(random_state=0),
         "25-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(25))}
    for s in range(seed_count):
        z[f"forest seed {s}"] = RandomForestClassifier(
            300, min_samples_leaf=5, random_state=s, n_jobs=2)
        z[f"MLP seed {s}"] = make_pipeline(
            StandardScaler(), MLPClassifier((64,), alpha=1e-2, max_iter=1500,
                                            random_state=s))
    return z


def rashomon(*, task: str = "gaussian", tolerance: float = 0.01,
             n: int = 2000) -> dict:
    """Fit a zoo of models; among those within `tolerance` of the best test
    accuracy, how often do they disagree, against the bound of the sum of
    two models' error rates? Also within one family across seeds."""
    np = _np()
    if task == "gaussian":
        from standarderror.ml.curves import GaussianTask
        t = GaussianTask(test=5000)
        X, y = t.draw(n, seed=0)
        Xt, yt = t.X_test, t.y_test
        bayes = t.bayes
    else:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        X0, y0 = load_breast_cancer(return_X_y=True)
        X, Xt, y, yt = train_test_split(X0, y0, test_size=0.5,
                                        random_state=0, stratify=y0)
        bayes = None
    P, Q, A = {}, {}, {}
    for k, m in _zoo().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, y)
        Q[k] = m.predict_proba(Xt)[:, 1]
        P[k] = (Q[k] > 0.5).astype(int)
        A[k] = float(np.mean(P[k] == yt))
    best = max(A.values())
    good = [k for k in A if A[k] >= best - tolerance]

    def stats(keys):
        M = np.array([P[k] for k in keys])
        pairs = [(a, b, float(np.mean(P[a] != P[b])),
                  (1 - A[a]) + (1 - A[b])) for a, b in
                 itertools.combinations(keys, 2)]
        R = np.array([Q[k] for k in keys])
        return {"models": list(keys),
                "accuracy": [A[k] for k in keys],
                "ambiguity": float(np.mean(M.min(0) != M.max(0))),
                "max_pair": max(p[2] for p in pairs) if pairs else 0.0,
                "pairs": pairs,
                "prob_range_median": float(np.median(R.max(0) - R.min(0))),
                "prob_range_p90": float(np.quantile(R.max(0) - R.min(0),
                                                    0.9))}

    fam = {f: stats([k for k in P if k.startswith(f)])
           for f in ("forest seed", "MLP seed")}
    pick = ("MLP seed 0", "MLP seed 1")
    return {"q_pair": (Q[pick[0]], Q[pick[1]]), "q_names": pick,
            "task": task, "bayes": bayes, "accuracy": A, "best": best,
            "tolerance": tolerance, "good": stats(good), "families": fam,
            "test_rows": int(len(yt))}
