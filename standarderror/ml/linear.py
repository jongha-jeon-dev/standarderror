r"""Arc II: linear models pushed past the conditions the textbook states.

Episode 6, `separable_*` and `gd_path`: logistic regression on data a
hyperplane separates, where the maximum-likelihood coefficients do not exist.
Every one of 50 half-splits of the Wisconsin breast cancer data is separable
in its 30 features, and what scikit-learn returns as the unpenalised fit is
set by its stopping tolerance.

Episode 7, `unit_sweep` and `binary_prevalence`: a lasso penalty is a price
per unit of coefficient, so the units -- and standardisation itself -- decide
which features pay it.

Episode 8, `correlated_pair` and `bootstrap_signs`: what cross-validated
lasso selects, when the truth is known and when it is not.

Episode 9, `bias_variance`: the decomposition computed exactly on simulated
data, for a knob that trades bias for variance and one that does not.

Episode 10, `random_features`: minimum-norm least squares through the
interpolation threshold.
"""

from __future__ import annotations

import math
import warnings


def _np():
    import numpy as np
    return np


# ================================================================ episode 6

def breast_cancer_split(seed: int = 0):
    """A stratified half split of the Wisconsin data, standardised on the
    training half."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from standarderror.ml.leakage import breast_cancer
    X, y = breast_cancer()
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5,
                                      random_state=int(seed), stratify=y)
    sc = StandardScaler().fit(Xa)
    return sc.transform(Xa), sc.transform(Xb), ya, yb


def digits_pair(a: int = 1, b: int = 8, seed: int = 0):
    """Two digits as a binary problem, half split, standardised."""
    np = _np()
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from standarderror.ml.evaluation import digits
    X, y = digits()
    m = (y == a) | (y == b)
    X, y = X[m], (y[m] == b).astype(int)
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5,
                                      random_state=int(seed), stratify=y)
    sc = StandardScaler().fit(Xa)
    Xa, Xb = sc.transform(Xa), sc.transform(Xb)
    keep = np.std(Xa, 0) > 0
    return Xa[:, keep], Xb[:, keep], ya, yb


def max_margin(X, y):
    """The hard-margin SVM direction (weights and intercept, unit norm) and
    whether it separates the rows: a linear SVM with an effectively infinite
    C is the hard-margin problem when the data are separable."""
    np = _np()
    from sklearn.svm import SVC
    s = SVC(kernel="linear", C=1e8).fit(X, y)
    w = np.r_[s.coef_[0], s.intercept_]
    return w / np.linalg.norm(w), bool(s.score(X, y) == 1.0)


def separable_fraction(*, splits: int = 50) -> float:
    """How many stratified half splits of the breast cancer data are
    linearly separable on their training half."""
    return float(sum(max_margin(*breast_cancer_split(s)[::2])[1]
                     for s in range(int(splits))) / splits)


def log_loss(y, p) -> float:
    np = _np()
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tolerance_sweep(*, tols=(1e-2, 1e-4, 1e-6, 1e-8, 1e-10),
                    seed: int = 0) -> list[dict]:
    """Unpenalised logistic regression on a separable training half, fitted
    to each stopping tolerance. scikit-learn's default is 1e-4."""
    np = _np()
    from sklearn.linear_model import LogisticRegression
    Xa, Xb, ya, yb = breast_cancer_split(seed)
    out = []
    for tol in tols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(penalty=None, max_iter=100_000,
                                   tol=tol).fit(Xa, ya)
        pa = m.predict_proba(Xa)[:, 1]
        out.append({"tol": float(tol), "norm": float(np.linalg.norm(m.coef_)),
                    "iterations": int(m.n_iter_[0]),
                    "test_accuracy": float(m.score(Xb, yb)),
                    "test_log_loss": log_loss(yb, m.predict_proba(Xb)[:, 1]),
                    "certain": float(np.mean(np.minimum(pa, 1 - pa) < 1e-6))})
    return out


def c_path(*, Cs=(1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5), seed: int = 0
           ) -> list[dict]:
    """The same split with an L2 penalty: with any finite C the optimum
    exists, and its size grows with C without bound."""
    np = _np()
    from sklearn.linear_model import LogisticRegression
    Xa, Xb, ya, yb = breast_cancer_split(seed)
    out = []
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=200_000, tol=1e-10).fit(Xa, ya)
        out.append({"C": float(C), "norm": float(np.linalg.norm(m.coef_)),
                    "test_accuracy": float(m.score(Xb, yb)),
                    "test_log_loss": log_loss(yb, m.predict_proba(Xb)[:, 1])})
    return out


def gd_path(*, steps: int = 100_000, lr: float = 1.0, points: int = 40,
            pair=(1, 8), seed: int = 0) -> dict:
    """Plain gradient descent on the unpenalised logistic loss, from zero,
    on a separable training half. Records the weight norm, the angle to the
    hard-margin direction, and training and test behaviour along the way."""
    np = _np()
    Xa, Xb, ya, yb = digits_pair(*pair, seed=seed)
    A, B = np.c_[Xa, np.ones(len(Xa))], np.c_[Xb, np.ones(len(Xb))]
    s = 2 * ya - 1
    w = np.zeros(A.shape[1])
    svm, sep = max_margin(Xa, ya)
    marks = set(np.unique(np.geomspace(1, steps, points).astype(int)))
    rows = []
    for t in range(1, int(steps) + 1):
        z = s * (A @ w)
        w -= lr * -(A * (s / (1 + np.exp(z)))[:, None]).mean(0)
        if t in marks:
            pb = 1 / (1 + np.exp(-(B @ w)))
            rows.append({
                "step": t, "norm": float(np.linalg.norm(w)),
                "angle": float(np.degrees(np.arccos(min(
                    1.0, w @ svm / np.linalg.norm(w))))),
                "train_loss": float(np.mean(np.logaddexp(0, -s * (A @ w)))),
                "test_accuracy": float(np.mean((pb > 0.5) == yb)),
                "test_log_loss": log_loss(yb, pb)})
    return {"rows": rows, "separable": sep, "pair": tuple(pair),
            "train": int(len(ya)), "test": int(len(yb)),
            "svm_test_accuracy": float(np.mean(
                ((B @ svm) > 0) == yb))}


def norm_per_decade(rows: list[dict], *, start: int = 1000) -> float:
    """Growth of the weight norm per tenfold increase in steps, fitted over
    the late part of the path: constant if the norm grows like log t."""
    np = _np()
    late = [r for r in rows if r["step"] >= start]
    x = np.log10([r["step"] for r in late])
    return float(np.polyfit(x, [r["norm"] for r in late], 1)[0])


def augmented_margin_angle(*, pair=(1, 8), seed: int = 0) -> float:
    """Degrees between scikit-learn's hard-margin SVM direction, whose
    intercept is free, and the max-margin direction of the problem
    `gd_path` actually solves, where the intercept is a weight on a column
    of ones and so counts in the norm. The latter is the limit of gradient
    descent's direction (Soudry et al. 2018), so this is where `gd_path`'s
    angle is heading, not zero."""
    np = _np()
    from sklearn.svm import LinearSVC
    Xa, _, ya, _ = digits_pair(*pair, seed=seed)
    svm, _ = max_margin(Xa, ya)
    A = np.c_[Xa, np.ones(len(Xa))]
    m = LinearSVC(fit_intercept=False, loss="hinge", C=1e6, tol=1e-10,
                  max_iter=2_000_000).fit(A, ya)
    w = m.coef_[0] / np.linalg.norm(m.coef_[0])
    return float(np.degrees(np.arccos(min(1.0, float(w @ svm)))))


# ================================================================ episode 7

def diabetes():
    """The scikit-learn diabetes data (Efron et al. 2004): 442 patients, 10
    baseline variables, a disease-progression score. Standardised here."""
    from sklearn.datasets import load_diabetes
    from sklearn.preprocessing import StandardScaler
    d = load_diabetes()
    return StandardScaler().fit_transform(d.data), d.target, list(
        d.feature_names)


def _lasso_cv(X, y, *, seed: int = 0, rule: str = "min"):
    """LassoCV, with either the CV-minimising alpha or the largest alpha
    within one standard error of it (the '1-SE rule')."""
    np = _np()
    from sklearn.linear_model import Lasso, LassoCV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = LassoCV(cv=5, random_state=int(seed), alphas=60).fit(X, y)
        if rule == "min":
            return cv.coef_, float(cv.alpha_)
        mse = cv.mse_path_.mean(1)
        se = cv.mse_path_.std(1) / math.sqrt(cv.mse_path_.shape[1])
        i = int(np.argmin(mse))
        ok = np.flatnonzero(mse <= mse[i] + se[i])
        alpha = float(cv.alphas_[ok].max())
        return Lasso(alpha=alpha).fit(X, y).coef_, alpha


def unit_sweep(*, factors=(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)
               ) -> dict:
    """Re-express one feature at a time in different units -- multiply its
    standardised column by k, as changing metres to centimetres would --
    and refit cross-validated lasso. Records whether each feature is kept
    and the cross-validated error at the chosen penalty."""
    np = _np()
    from sklearn.linear_model import LassoCV
    X, y, names = diabetes()
    kept = np.zeros((len(names), len(factors)), bool)
    cv_mse = np.zeros((len(names), len(factors)))
    for j in range(len(names)):
        for i, k in enumerate(factors):
            Z = X.copy()
            Z[:, j] *= k
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv = LassoCV(cv=5, random_state=0, alphas=60).fit(Z, y)
            kept[j, i] = cv.coef_[j] != 0
            cv_mse[j, i] = float(cv.mse_path_.mean(1).min())
    base = factors.index(1)
    return {"names": names, "factors": [float(k) for k in factors],
            "kept": kept, "cv_mse": cv_mse, "base": kept[:, base].copy(),
            "base_mse": float(cv_mse[0, base])}


def binary_prevalence(*, prevalences=(0.5, 0.2, 0.1, 0.05, 0.02),
                      n: int = 500, effect: float = 1.0, reps: int = 60,
                      seed: int = 0) -> list[dict]:
    """A binary feature with the same effect per occurrence at different
    prevalences, among five continuous features (two with real effects).
    Lasso on standardised features against lasso on the raw 0/1 column:
    how often is the binary feature kept?"""
    np = _np()
    from sklearn.preprocessing import StandardScaler
    out = []
    for prev in prevalences:
        std = raw = 0
        for r in range(int(reps)):
            g = np.random.default_rng(int(seed) * 1000 + r)
            b = (g.random(n) < prev).astype(float)
            C = g.standard_normal((n, 5))
            y = effect * b + 0.5 * C[:, 0] + 0.5 * C[:, 1] + \
                g.standard_normal(n)
            Cs = StandardScaler().fit_transform(C)
            std += _lasso_cv(np.c_[StandardScaler().fit_transform(
                b[:, None]), Cs], y, seed=r)[0][0] != 0
            raw += _lasso_cv(np.c_[b - b.mean(), Cs], y, seed=r)[0][0] != 0
        out.append({"prevalence": float(prev), "sd": math.sqrt(
            prev * (1 - prev)), "kept_standardised": std / reps,
                    "kept_raw": raw / reps})
    return out


# ================================================================ episode 8

def correlated_pair(*, rhos=(0.0, 0.5, 0.8, 0.9, 0.95, 0.99), n: int = 100,
                    reps: int = 60, rule: str = "min") -> list[dict]:
    """Ten features: `a` (feature 0) and `c` (feature 2) have effects of
    0.5, `b` (feature 1) has none but is correlated with `a` at rho, and
    features 3-9 are noise. How often does cross-validated lasso keep each?
    `c`, `a_only` and `both` record the second relevant feature and how
    the pair (a, b) was split; they draw nothing extra from the generator.
    """
    np = _np()
    out = []
    for rho in rhos:
        ka = kb = kc = only_a = only_b = both = 0
        fp = []
        for r in range(int(reps)):
            g = np.random.default_rng(r)
            X = g.standard_normal((n, 10))
            X[:, 1] = rho * X[:, 0] + math.sqrt(1 - rho ** 2) * X[:, 1]
            y = 0.5 * X[:, 0] + 0.5 * X[:, 2] + g.standard_normal(n)
            coef = _lasso_cv(X, y, seed=0, rule=rule)[0]
            a, b = coef[0] != 0, coef[1] != 0
            ka += a
            kb += b
            kc += coef[2] != 0
            only_a += a and not b
            only_b += b and not a
            both += a and b
            fp.append(int(np.sum(coef[3:] != 0)))
        out.append({"rho": float(rho), "a": ka / reps, "b": kb / reps,
                    "b_only": only_b / reps, "noise_kept": float(np.mean(fp)),
                    "rule": rule, "c": kc / reps, "a_only": only_a / reps,
                    "both": both / reps})
    return out


def elastic_pair(*, rhos=(0.0, 0.9, 0.99), n: int = 100, reps: int = 60,
                 l1_ratio: float = 0.5) -> list[dict]:
    """`correlated_pair`'s design under a cross-validated elastic net: how
    often the relevant `a`, the stand-in `b` and the noise are kept."""
    np = _np()
    from sklearn.linear_model import ElasticNetCV
    out = []
    for rho in rhos:
        ka = kb = 0
        fp = []
        for r in range(int(reps)):
            g = np.random.default_rng(r)
            X = g.standard_normal((n, 10))
            X[:, 1] = rho * X[:, 0] + math.sqrt(1 - rho ** 2) * X[:, 1]
            y = 0.5 * X[:, 0] + 0.5 * X[:, 2] + g.standard_normal(n)
            c = ElasticNetCV(l1_ratio=l1_ratio, cv=5, random_state=0,
                             alphas=60).fit(X, y).coef_
            ka += c[0] != 0
            kb += c[1] != 0
            fp.append(int(np.sum(c[3:] != 0)))
        out.append({"rho": float(rho), "a": ka / reps, "b": kb / reps,
                    "noise_kept": float(np.mean(fp))})
    return out


def bootstrap_signs(*, boots: int = 100, seed: int = 0,
                    rule: str = "min") -> dict:
    """Cross-validated lasso on bootstrap resamples of the diabetes data:
    for each feature, how often it is kept with a positive and with a
    negative coefficient."""
    np = _np()
    X, y, names = diabetes()
    g = np.random.default_rng(int(seed))
    pos, neg = np.zeros(len(names)), np.zeros(len(names))
    for b in range(int(boots)):
        i = g.integers(0, len(y), len(y))
        c = _lasso_cv(X[i], y[i], seed=b, rule=rule)[0]
        pos += c > 0
        neg += c < 0
    full = _lasso_cv(X, y, rule=rule)[0]
    return {"names": names, "positive": pos / boots, "negative": neg / boots,
            "full": full, "boots": int(boots), "rule": rule}


# ================================================================ episode 9

def _bv_target(x):
    np = _np()
    return np.sin(2 * np.pi * x) + 0.5 * x


def bias_variance(make, *, n: int = 60, sigma: float = 0.4, reps: int = 300,
                  grid: int = 200, seed: int = 0) -> dict:
    """Bias squared, variance and noise at a grid of test points, each
    averaged over the grid, from `reps` independent training sets of `n`
    points; and the test MSE measured directly against fresh noisy labels,
    which the three should add up to."""
    np = _np()
    xt = np.linspace(0.02, 0.98, int(grid))
    P = np.empty((int(reps), len(xt)))
    for r in range(int(reps)):
        g = np.random.default_rng(int(seed) * 100_000 + r)
        x = g.random(n)
        y = _bv_target(x) + sigma * g.standard_normal(n)
        P[r] = make().fit(x[:, None], y).predict(xt[:, None])
    f = _bv_target(xt)
    bias2 = float(np.mean((P.mean(0) - f) ** 2))
    var = float(np.mean(P.var(0)))
    g = np.random.default_rng(999)
    mse = float(np.mean((P - (f + sigma * g.standard_normal(P.shape))) ** 2))
    return {"bias2": bias2, "variance": var, "noise": sigma ** 2,
            "total": bias2 + var + sigma ** 2, "mse": mse}


def knn_sweep(ks=(1, 2, 4, 8, 16, 32), **kw) -> list[dict]:
    from sklearn.neighbors import KNeighborsRegressor
    return [{"k": int(k), **bias_variance(
        lambda k=k: KNeighborsRegressor(int(k)), **kw)} for k in ks]


def forest_sweep(trees=(1, 2, 5, 25, 100), **kw) -> list[dict]:
    from sklearn.ensemble import RandomForestRegressor
    kw.setdefault("reps", 100)
    return [{"trees": int(t), **bias_variance(
        lambda t=t: RandomForestRegressor(int(t), random_state=0), **kw)}
        for t in trees]


def forest_rho(sweep: list[dict]) -> float:
    """The between-tree correlation implied by var(T) = rho s2 + (1 - rho)
    s2 / T, from the one-tree and the largest forest (Hastie, Tibshirani and
    Friedman 2009, eq. 15.1; Breiman 2001 bounds the infinite forest)."""
    s2 = sweep[0]["variance"]
    big = sweep[-1]
    T = big["trees"]
    return (big["variance"] - s2 / T) / (s2 - s2 / T)


# ================================================================ episode 10

def _rf_target(X, beta):
    np = _np()
    return np.tanh(2 * X @ beta) + 0.3 * np.sin(2 * X[:, 0])


def random_features(*, n: int = 100, ps=None, sigma: float = 0.3,
                    reps: int = 30, ridge: float | None = None, d: int = 10,
                    test: int = 5000) -> dict:
    """Least squares on `p` random ReLU features of a `d`-dimensional input.
    With `ridge` None the fit is the minimum-norm one, which interpolates
    the training data once p >= n. Median test error over `reps` training
    sets and feature draws, with the median weight norm."""
    np = _np()
    ps = ps or [10, 20, 40, 60, 80, 90, 95, 100, 105, 110, 120, 150, 200,
                300, 500, 1000, 2000]
    ps = [int(p) for p in ps]
    beta = np.random.default_rng(0).standard_normal(d) / math.sqrt(d)
    Xt = np.random.default_rng(99).standard_normal((test, d))
    yt = _rf_target(Xt, beta)
    err = {p: [] for p in ps}
    norm = {p: [] for p in ps}
    train = {p: [] for p in ps}
    for r in range(int(reps)):
        g = np.random.default_rng(r)
        X = g.standard_normal((n, d))
        y = _rf_target(X, beta) + sigma * g.standard_normal(n)
        W = np.random.default_rng(1000 + r).standard_normal(
            (d, max(ps))) / math.sqrt(d)
        for p in ps:
            Z, Zt = np.maximum(X @ W[:, :p], 0), np.maximum(Xt @ W[:, :p], 0)
            if ridge is None:
                w = np.linalg.pinv(Z) @ y
            elif p <= n:
                w = np.linalg.solve(Z.T @ Z + ridge * np.eye(p), Z.T @ y)
            else:
                w = Z.T @ np.linalg.solve(Z @ Z.T + ridge * np.eye(n), y)
            err[p].append(float(np.mean((Zt @ w - yt) ** 2)))
            norm[p].append(float(np.linalg.norm(w)))
            train[p].append(float(np.mean((Z @ w - y) ** 2)))
    med = lambda d_: {p: float(np.median(v)) for p, v in d_.items()}  # noqa
    return {"n": n, "ps": ps, "sigma": sigma, "ridge": ridge,
            "test": med(err), "norm": med(norm), "train": med(train),
            "reps": int(reps)}


def peak(curve: dict) -> int:
    """The number of features at which median test error is highest."""
    return max(curve["test"], key=curve["test"].get)


def best_under(curve: dict) -> tuple[int, float]:
    """The best p below the interpolation threshold, and its error."""
    lo = {p: e for p, e in curve["test"].items() if p < curve["n"]}
    p = min(lo, key=lo.get)
    return p, lo[p]
