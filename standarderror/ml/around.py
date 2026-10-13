r"""Arc VII: around the model. Representation, clusters, distance,
probabilities and selection, each measured on scikit-learn's bundled data
or on a simulation with a known answer.

Episode 31, `pca_*`: what the leading principal components keep.
Episode 32, `kmeans_*`: what k-means and the silhouette find.
Episode 33, `contrast` and `neighbours_noise`: nearest neighbours as the
dimension grows, nominally and intrinsically.
Episode 34, `calibration_size` and `subgroup_calibration`: Platt scaling and
isotonic regression on small calibration sets, and calibration that holds
on average and fails in a group.
Episode 35, `tuning_optimism`: the best cross-validated score of a search,
against the selected model's true accuracy and nested cross-validation.
"""

from __future__ import annotations

import functools
import math


def _np():
    import numpy as np
    return np


def _ece(p, y, bins: int = 10) -> float:
    np = _np()
    p, y = np.asarray(p), np.asarray(y)
    e = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[e == b].mean() - y[e == b].mean()) *
                     (e == b).mean() for b in range(bins) if (e == b).any()))


# ================================================================ episode 31

def pca_simulation(*, n: int = 2000, seed: int = 0) -> dict:
    """Ten independent features with standard deviations from 10 down to
    0.3; the label depends only on the last, smallest one. Cross-validated
    accuracy of logistic regression on the first k components."""
    np = _np()
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    g = np.random.default_rng(seed)
    sd = np.array([10, 8, 6, 5, 4, 3, 2, 1.5, 1, 0.3])
    X = g.normal(size=(n, 10)) * sd
    y = (X[:, 9] / 0.3 + 0.3 * g.normal(size=n) > 0).astype(int)
    acc = [float(cross_val_score(make_pipeline(
        PCA(k), LogisticRegression(max_iter=2000)), X, y, cv=5).mean())
        for k in range(1, 11)]
    share = PCA().fit(X).explained_variance_ratio_
    return {"sd": sd.tolist(), "acc": acc, "share": share.tolist()}


@functools.cache
def pca_breast_cancer(ks=(1, 2, 3, 5, 10, 30)) -> dict:
    """PCA then logistic regression on breast cancer, on raw and on
    standardised features: accuracy for k components, the share of variance
    of the first components, the feature that dominates the first raw
    component, and how many components the 95% and 99% variance rules keep."""
    np = _np()
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    data = load_breast_cancer()
    X, y = data.data, data.target
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    out = {"ks": list(ks)}
    for name, scale in (("raw", False), ("std", True)):
        acc = []
        for k in ks:
            steps = ([StandardScaler()] if scale else []) + [
                PCA(k), LogisticRegression(max_iter=5000)]
            acc.append(float(cross_val_score(make_pipeline(*steps), X, y,
                                             cv=cv).mean()))
        Z = StandardScaler().fit_transform(X) if scale else X
        p = PCA().fit(Z)
        c = np.cumsum(p.explained_variance_ratio_)
        out[name] = {"acc": acc,
                     "share": p.explained_variance_ratio_[:3].tolist(),
                     "k95": int(np.searchsorted(c, 0.95) + 1),
                     "k99": int(np.searchsorted(c, 0.99) + 1),
                     "top": str(data.feature_names[
                         int(np.argmax(abs(p.components_[0])))])}
    return out


@functools.cache
def pca_diabetes(ks=(1, 2, 3, 4, 5, 10)) -> dict:
    """Diabetes (features already scaled): each principal component's share
    of the feature variance and its squared correlation with the target, and
    10-fold R^2 of regression on the first k components against partial
    least squares with k components."""
    np = _np()
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.datasets import load_diabetes
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    X, y = load_diabetes(return_X_y=True)
    p = PCA().fit(X)
    Z = p.transform(X)
    r2 = [float(np.corrcoef(Z[:, j], y)[0, 1] ** 2) for j in range(Z.shape[1])]
    cv = KFold(10, shuffle=True, random_state=0)
    pcr = [float(cross_val_score(make_pipeline(PCA(k), LinearRegression()),
                                 X, y, cv=cv, scoring="r2").mean()) for k in ks]
    pls = [float(cross_val_score(PLSRegression(k, scale=False), X, y, cv=cv,
                                 scoring="r2").mean()) for k in ks]
    return {"share": p.explained_variance_ratio_.tolist(), "r2": r2,
            "ks": list(ks), "pcr": pcr, "pls": pls}


# ================================================================ episode 32

def _scan(X, ks, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    out = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=seed).fit(X)
        out.append({"k": k, "inertia": float(km.inertia_),
                    "silhouette": float(silhouette_score(X, km.labels_))})
    return out


def _shapes(name, seed):
    """Two-dimensional data sets with a known grouping (None for none)."""
    np = _np()
    g = np.random.default_rng(seed)
    if name == "separated":
        c = np.array([[0, 0], [8, 0], [0, 8], [8, 8]])
        y = g.integers(0, 4, 1000)
        return c[y] + g.normal(size=(1000, 2)), y, 4
    if name == "uniform":
        return g.uniform(size=(1000, 2)), None, 1
    if name == "unequal":
        X = np.vstack([g.normal([0, 0], 1.5, (900, 2)),
                       g.normal([5, 0], 0.3, (50, 2)),
                       g.normal([0, 5], 0.3, (50, 2))])
        return X, np.r_[np.zeros(900), np.ones(50), 2 * np.ones(50)], 3
    if name == "elongated":
        X = np.vstack([g.normal(size=(500, 2)) * [6, 0.5],
                       g.normal(size=(500, 2)) * [6, 0.5] + [0, 3]])
        return X, np.r_[np.zeros(500), np.ones(500)], 2
    raise ValueError(name)


@functools.cache
def kmeans_shapes(*, reps: int = 10,
                  names=("separated", "uniform", "unequal", "elongated"),
                  ks=tuple(range(2, 11))) -> dict:
    """For each shape, over `reps` freshly drawn data sets: the k the
    silhouette picks, the best silhouette, and the adjusted Rand index of
    k-means at the true k."""
    np = _np()
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    out = {}
    for name in names:
        picks, best, ari = [], [], []
        for r in range(int(reps)):
            X, y, k = _shapes(name, r)
            s = _scan(X, ks)
            b = max(s, key=lambda t: t["silhouette"])
            picks.append(b["k"])
            best.append(b["silhouette"])
            if y is not None:
                lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
                ari.append(float(adjusted_rand_score(y, lab)))
        out[name] = {"true_k": _shapes(name, 0)[2], "picks": picks,
                     "silhouette": float(np.mean(best)),
                     "ari": float(np.mean(ari)) if ari else None,
                     "example": {"X": _shapes(name, 0)[0].tolist(),
                                 "labels": KMeans(_shapes(name, 0)[2] if
                                                  name != "uniform" else 4,
                                                  n_init=10, random_state=0)
                                 .fit_predict(_shapes(name, 0)[0]).tolist()}}
    return out


@functools.cache
def kmeans_digits(*, ks=tuple(range(2, 16)), seeds=tuple(range(10))) -> dict:
    """k-means on the 1,797 digits (raw pixels): the silhouette at each k,
    and at k = 10 the adjusted Rand index against the true digit for each
    single-start seed and for ten starts."""
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_digits
    from sklearn.metrics import adjusted_rand_score
    X, y = load_digits(return_X_y=True)
    scan = _scan(X, ks)
    single = []
    for s in seeds:
        km = KMeans(10, n_init=1, random_state=s).fit(X)
        single.append({"seed": s, "ari": float(adjusted_rand_score(y, km.labels_)),
                       "inertia": float(km.inertia_)})
    km = KMeans(10, n_init=10, random_state=0).fit(X)
    return {"scan": scan, "single": single,
            "ari10": float(adjusted_rand_score(y, km.labels_))}


# ================================================================ episode 33

def _nn_ratio(X, Q, exclude_self: bool = False):
    """Median over queries of (nearest distance) / (median distance), and of
    (farthest - nearest) / nearest. With `exclude_self` the queries are the
    first rows of X and each one's zero distance to itself is dropped."""
    np = _np()
    from sklearn.metrics import pairwise_distances
    D = np.sort(pairwise_distances(Q, X), 1)
    if exclude_self:
        D = D[:, 1:]
    mn, med, mx = D[:, 0], np.median(D, 1), D[:, -1]
    return float(np.median(mn / med)), float(np.median((mx - mn) / mn))


@functools.cache
def contrast(*, dims=(1, 2, 4, 8, 16, 64, 256, 1024), n: int = 1000,
             queries: int = 200, seed: int = 0) -> list[dict]:
    """Uniform points in [0, 1]^d: for each query, the nearest distance over
    the median distance, and (farthest - nearest) / nearest."""
    np = _np()
    g = np.random.default_rng(seed)
    out = []
    for d in dims:
        X = g.uniform(size=(n, d))
        Q = g.uniform(size=(queries, d))
        ratio, spread = _nn_ratio(X, Q)
        out.append({"d": d, "ratio": ratio, "spread": spread})
    return out


@functools.cache
def neighbours_noise(*, extras=(0, 16, 64, 256, 1024), seed: int = 0) -> list[dict]:
    """Digits scaled to [0, 1], with `extra` uniform noise columns on the
    same scale appended: 5-fold accuracy of 5-nearest neighbours and of
    logistic regression, and the nearest/median distance ratio."""
    np = _np()
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    X, y = load_digits(return_X_y=True)
    X = X / 16
    g = np.random.default_rng(seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    out = []
    for extra in extras:
        Z = np.hstack([X, g.uniform(size=(len(X), extra))]) if extra else X
        knn = float(cross_val_score(KNeighborsClassifier(5), Z, y, cv=cv).mean())
        lr = float(cross_val_score(LogisticRegression(max_iter=3000), Z, y,
                                   cv=cv).mean())
        ratio, spread = _nn_ratio(Z, Z[:300], exclude_self=True)
        out.append({"extra": extra, "d": Z.shape[1], "knn": knn, "logistic": lr,
                    "ratio": ratio, "spread": spread})
    return out


# ================================================================ episode 34

def _logit(p):
    np = _np()
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@functools.cache
def calibration_size(*, sizes=(50, 100, 200, 500, 1000, 5000),
                     reps: int = 20) -> dict:
    """Episode 4's simulated task. A naive Bayes model (already close to
    calibrated) and boosted trees (overconfident), each trained on 2,000
    rows, recalibrated by Platt scaling and by isotonic regression on fresh
    calibration sets of each size, `reps` times. Test log loss, expected
    calibration error and AUC on 20,000 rows."""
    np = _np()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.naive_bayes import GaussianNB

    from standarderror.ml.curves import GaussianTask
    t = GaussianTask(test=20000)
    Xte, yte = t.X_test, t.y_test
    makers = {"naive Bayes": GaussianNB,
              "boosted trees": lambda: HistGradientBoostingClassifier(
                  max_iter=300, early_stopping=False, random_state=0)}
    out = {}
    for name, M in makers.items():
        X, y = t.draw(2000, seed=1)
        m = M().fit(X, y)
        p = m.predict_proba(Xte)[:, 1]
        row = {"raw": {"nll": float(log_loss(yte, p)), "ece": _ece(p, yte),
                       "auc": float(roc_auc_score(yte, p))},
               "sizes": list(sizes), "platt": [], "isotonic": []}
        for n in sizes:
            acc = {"platt": [], "isotonic": []}
            for r in range(int(reps)):
                Xc, yc = t.draw(n, seed=100 + r)
                pc = m.predict_proba(Xc)[:, 1]
                pl = LogisticRegression(C=1e6).fit(_logit(pc)[:, None], yc)
                pp = pl.predict_proba(_logit(p)[:, None])[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-3,
                                         y_max=1 - 1e-3).fit(pc, yc)
                pi = iso.predict(p)
                for k, q in (("platt", pp), ("isotonic", pi)):
                    acc[k].append((log_loss(yte, q), _ece(q, yte),
                                   roc_auc_score(yte, q)))
            for k in acc:
                a = np.mean(acc[k], 0)
                row[k].append({"n": n, "nll": float(a[0]), "ece": float(a[1]),
                               "auc": float(a[2])})
        out[name] = row
    return out


@functools.cache
def subgroup_calibration(*, n: int = 5000, test: int = 100_000) -> dict:
    """Two groups, 70% and 30% of rows, with different base rates; the
    model sees x but not the group. Calibration error overall and within
    each group, for that model and for one that also sees the group."""
    np = _np()
    from sklearn.linear_model import LogisticRegression

    def draw(m, seed):
        g = np.random.default_rng(seed)
        grp = (g.random(m) < 0.3).astype(int)
        x = g.normal(size=m)
        z = 1.5 * x + np.where(grp == 1, 1.0, -0.6)
        return x, grp, (g.random(m) < 1 / (1 + np.exp(-z))).astype(int)

    x, grp, y = draw(n, 0)
    xt, gt, yt = draw(test, 1)
    blind = LogisticRegression().fit(x[:, None], y).predict_proba(
        xt[:, None])[:, 1]
    aware = LogisticRegression().fit(np.c_[x, grp], y).predict_proba(
        np.c_[xt, gt])[:, 1]
    out = {"blind": {"all": _ece(blind, yt)}, "aware": {"all": _ece(aware, yt)},
           "groups": []}
    for k in (0, 1):
        s = gt == k
        out["groups"].append({"group": k, "share": float(s.mean()),
                              "rate": float(yt[s].mean()),
                              "blind_mean": float(blind[s].mean()),
                              "aware_mean": float(aware[s].mean())})
        out["blind"][k] = _ece(blind[s], yt[s])
        out["aware"][k] = _ece(aware[s], yt[s])
    return out


# ================================================================ episode 35

def _grid(size):
    """An RBF-SVM grid of about `size` cells, log-spaced around the default
    (C = 1, gamma = 0.1 for ten standardised inputs); size 1 is the default
    alone, so it involves no search."""
    np = _np()
    if size == 1:
        return {"C": [1.0], "gamma": [0.1]}
    a = int(round(math.sqrt(size)))
    b = max(1, size // a)
    return {"C": np.logspace(-1, 3, a).tolist(),
            "gamma": np.logspace(-3, 0.5, b).tolist()}


@functools.cache
def tuning_optimism(*, sizes=(1, 4, 16, 64), reps: int = 20, n: int = 200,
                    null: bool = False) -> list[dict]:
    """Search an RBF-SVM grid of `size` configurations by 5-fold CV on `n`
    rows of episode 4's task (or of pure noise when `null`), `reps` times:
    the best CV score, the refit model's accuracy on 20,000 fresh rows, the
    best accuracy any configuration in the grid reaches on those rows, and a
    nested 5 x 5 cross-validation estimate."""
    np = _np()
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
    from sklearn.svm import SVC

    from standarderror.ml.curves import GaussianTask
    t = GaussianTask(test=20000)
    out = []
    for size in sizes:
        grid = _grid(size)
        cells = len(grid["C"]) * len(grid["gamma"])
        best, true, oracle, nested = [], [], [], []
        for r in range(int(reps)):
            if null:
                g = np.random.default_rng(r)
                X, y = g.normal(size=(n, 10)), g.integers(0, 2, n)
                Xt = g.normal(size=(20000, 10))
                yt = g.integers(0, 2, 20000)
            else:
                X, y = t.draw(n, seed=1000 + r)
                Xt, yt = t.X_test, t.y_test
            cv = StratifiedKFold(5, shuffle=True, random_state=r)
            G = GridSearchCV(SVC(), grid, cv=cv).fit(X, y)
            best.append(G.best_score_)
            true.append(float((G.predict(Xt) == yt).mean()))
            orc = 0.0
            for C in grid["C"]:
                for gm in grid["gamma"]:
                    m = SVC(C=C, gamma=gm).fit(X, y)
                    orc = max(orc, float((m.predict(Xt) == yt).mean()))
            oracle.append(orc)
            outer = StratifiedKFold(5, shuffle=True, random_state=100 + r)
            nested.append(float(cross_val_score(
                GridSearchCV(SVC(), grid, cv=StratifiedKFold(
                    5, shuffle=True, random_state=r)), X, y, cv=outer).mean()))
        b, tr = np.array(best), np.array(true)
        out.append({"size": cells, "best_cv": float(b.mean()),
                    "true": float(tr.mean()), "oracle": float(np.mean(oracle)),
                    "nested": float(np.mean(nested)),
                    "optimism": float((b - tr).mean()),
                    "optimism_se": float((b - tr).std(ddof=1) /
                                         math.sqrt(len(b))),
                    "nested_gap": float((np.array(nested) - tr).mean()),
                    "nested_gap_se": float((np.array(nested) - tr).std(ddof=1)
                                           / math.sqrt(len(b)))})
    return out


@functools.cache
def neighbours_repair(*, extra: int = 1024, keep: int = 64, seed: int = 0) -> dict:
    """The digits with `extra` uniform noise columns: 5-nearest neighbours
    after keeping the `keep` columns with the largest ANOVA F statistic,
    chosen inside each training fold, and the share of the kept columns
    that are real pixels."""
    np = _np()
    from sklearn.datasets import load_digits
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    X, y = load_digits(return_X_y=True)
    X = X / 16
    g = np.random.default_rng(seed)
    Z = np.hstack([X, g.uniform(size=(len(X), extra))])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = make_pipeline(SelectKBest(f_classif, k=keep), KNeighborsClassifier(5))
    import warnings
    with warnings.catch_warnings():    # blank pixels have no F statistic
        warnings.simplefilter("ignore")
        acc = float(cross_val_score(pipe, Z, y, cv=cv).mean())
        sel = SelectKBest(f_classif, k=keep).fit(Z, y).get_support()
    return {"extra": extra, "keep": keep, "knn": acc,
            "real_share": float(sel[:64].sum() / keep)}
