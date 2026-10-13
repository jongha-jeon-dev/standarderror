r"""Arc VIII: where the rows come from. Five ways the process that produced
a data set decides what a model can learn from it and what a score says,
each measured on a simulation with a known answer or on scikit-learn's
bundled breast-cancer data.

Episode 36, `time_series_table`: shuffled, blocked and forward
cross-validation on time series, against the error on the next 250 steps.
Episode 37, `subject_cv` and `subject_growth`: many rows from few subjects,
row-level against subject-level cross-validation, and what more rows from
the same subjects buy.
Episode 38, `missing_study`: imputation, missingness indicators and native
missing-value handling under three missingness mechanisms, and when the
mechanism changes after training.
Episode 39, `selective_labels`: a model trained on accepted applicants only,
when acceptance depended on what the model sees and on what it does not.
Episode 40, `target_encoding`: encoding a high-cardinality category by its
mean label, fitted on the training rows only, with and without cross-fitting.
"""

from __future__ import annotations

import functools
import warnings


def _np():
    import numpy as np
    return np


def _sig(z):
    np = _np()
    return 1.0 / (1.0 + np.exp(-z))


# ================================================================ episode 36

def _ar2(n, g, burn=200):
    np = _np()
    e = g.normal(size=n + burn)
    y = np.zeros(n + burn)
    for t in range(2, n + burn):
        y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 2] + e[t]
    return y[burn:]


def _drift(n, g, step=0.15):
    """AR(2) around a level that wanders as a random walk."""
    np = _np()
    level = np.cumsum(g.normal(scale=step, size=n))
    return _ar2(n, g) + level


def _lags(y, p):
    np = _np()
    X = np.column_stack([y[p - k - 1:len(y) - k - 1] for k in range(p)])
    return X, y[p:]


def _exogenous(n, g, phi_u, phi_x=0.95, burn=200, with_u=False):
    """y_t = sin(2 x_t) + x_t + 0.7 u_t, with x and u stationary AR(1)
    processes of unit variance; u's persistence is phi_u."""
    np = _np()
    x = np.zeros(n + burn)
    u = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi_x * x[t - 1] + g.normal() * np.sqrt(1 - phi_x ** 2)
        u[t] = phi_u * u[t - 1] + g.normal() * np.sqrt(1 - phi_u ** 2)
    x, u = x[burn:], u[burn:]
    y = np.sin(2 * x) + x + 0.7 * u
    return (x[:, None], y, u) if with_u else (x[:, None], y)


def _ts_models():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    return {"linear": LinearRegression,
            "5-NN": lambda: KNeighborsRegressor(5),
            "forest": lambda: RandomForestRegressor(
                50, min_samples_leaf=5, n_jobs=2, random_state=0)}


def _ts_estimates(model, X, y, T, seed):
    """Three cross-validated MSEs on the first T rows, and the MSE of the
    model refitted on all T rows over the rows after them."""
    np = _np()
    from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
    Xtr, ytr, Xte, yte = X[:T], y[:T], X[T:], y[T:]
    out = {}
    for name, cv in (("shuffled", KFold(5, shuffle=True, random_state=seed)),
                     ("blocked", KFold(5)),
                     ("forward", TimeSeriesSplit(5))):
        out[name] = float(-cross_val_score(
            model(), Xtr, ytr, cv=cv,
            scoring="neg_mean_squared_error").mean())
    m = model().fit(Xtr, ytr)
    out["future"] = float(np.mean((m.predict(Xte) - yte) ** 2))
    return out


@functools.cache
def time_series_table(*, reps: int = 20, T: int = 1000, H: int = 250) -> list[dict]:
    """Each setting is simulated `reps` times: T rows to cross-validate and
    train on, then H rows of the future. Returns, per setting and model, the
    mean of each estimate and of the future MSE, and each estimate's error
    relative to the future MSE."""
    np = _np()
    M = _ts_models()
    settings = [
        ("AR(2), two lags", "ar2", 2, None, ["linear", "5-NN", "forest"]),
        ("AR(2), one lag", "ar2", 1, None, ["linear"]),
        ("residuals, persistence 0", "exo", 0.0, None, ["5-NN", "forest"]),
        ("residuals, persistence 0.9", "exo", 0.9, None, ["5-NN", "forest"]),
        ("residuals, persistence 0.99", "exo", 0.99, None, ["5-NN", "forest"]),
        ("drifting level", "drift", 2, None, ["linear", "forest"]),
        ("drifting level, time index", "drift", 2, "t", ["forest"]),
    ]
    rows = []
    for label, kind, par, extra, models in settings:
        acc = {m: [] for m in models}
        for r in range(int(reps)):
            g = np.random.default_rng(r)
            if kind == "exo":
                X, y = _exogenous(T + H, g, par)
            else:
                p = int(par)
                series = (_ar2 if kind == "ar2" else _drift)(T + H + p, g)
                X, y = _lags(series, p)
                if extra == "t":
                    X = np.column_stack([X, np.arange(len(y))])
            for m in models:
                acc[m].append(_ts_estimates(M[m], X, y, T, r))
        for m in models:
            mean = {k: float(np.mean([a[k] for a in acc[m]]))
                    for k in ("shuffled", "blocked", "forward", "future")}
            rel = {k: mean[k] / mean["future"] - 1
                   for k in ("shuffled", "blocked", "forward")}
            rows.append({"setting": label, "model": m, **mean,
                         "rel": rel})
    return rows


def window_correlation(*, phis=(0.0, 0.9, 0.99), T: int = 1000,
                       reps: int = 200) -> dict:
    """x and u are independent processes, so their correlation over all
    time is zero. Over one window of T steps it is not: the root mean square
    of the sample correlation, by persistence of u. A flexible model fitted
    on the window learns that window's correlation as if it were signal."""
    np = _np()
    out = {}
    for phi in phis:
        c = []
        for r in range(int(reps)):
            X, _, u = _exogenous(T, np.random.default_rng(10_000 + r), phi,
                                 with_u=True)
            c.append(np.corrcoef(X[:, 0], u)[0, 1])
        out[phi] = float(np.sqrt(np.mean(np.square(c))))
    return out


@functools.cache
def far_future(*, phi: float = 0.99, reps: int = 20, T: int = 1000,
               H: int = 250, gap: int = 1000) -> dict:
    """The forest of `time_series_table`, trained on the same first T steps,
    scored on the H steps right after them and on H steps `gap` later."""
    np = _np()
    M = _ts_models()["forest"]
    near, far = [], []
    for r in range(int(reps)):
        X, y = _exogenous(T + gap + H, np.random.default_rng(r), phi)
        m = M().fit(X[:T], y[:T])
        near.append(np.mean((m.predict(X[T:T + H]) - y[T:T + H]) ** 2))
        far.append(np.mean((m.predict(X[T + gap:]) - y[T + gap:]) ** 2))
    return {"near": float(np.mean(near)), "far": float(np.mean(far))}


# ================================================================ episode 37

def _subjects(S, m, g, *, d=10, tau=2.0, sb=2.0):
    """S subjects, m rows each. A subject has a signature a_s (sd tau in each
    of d features) added to every row, and a propensity b_s (sd sb) added to
    every row's log-odds. The signal that transfers to new subjects is
    1.5 x_1 + x_2."""
    np = _np()
    a = g.normal(scale=tau, size=(S, d))
    b = g.normal(scale=sb, size=S)
    s = np.repeat(np.arange(S), m)
    X = a[s] + g.normal(size=(S * m, d))
    z = 1.5 * X[:, 0] + X[:, 1] + b[s]
    y = (g.random(S * m) < _sig(z)).astype(int)
    return X, y, s


def _subject_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    return {"logistic": lambda: LogisticRegression(max_iter=2000),
            "5-NN": lambda: KNeighborsClassifier(5),
            "forest": lambda: RandomForestClassifier(
                50, n_jobs=2, random_state=0)}


@functools.cache
def subject_cv(*, designs=((2000, 1), (400, 5), (100, 20), (40, 50), (20, 100)),
               reps: int = 6) -> list[dict]:
    """2,000 rows as S subjects of m rows. Five-fold cross-validation by
    rows and by subjects, against accuracy on 10,000 rows from 2,000 new
    subjects."""
    np = _np()
    from sklearn.model_selection import GroupKFold, KFold, cross_val_score
    M = _subject_models()
    rows = []
    for S, m in designs:
        acc = {k: {"row": [], "subject": [], "new": []} for k in M}
        for r in range(int(reps)):
            X, y, s = _subjects(S, m, np.random.default_rng(r))
            Xt, yt, _ = _subjects(2000, 5, np.random.default_rng(1000 + r))
            for k, mk in M.items():
                acc[k]["row"].append(cross_val_score(
                    mk(), X, y, cv=KFold(5, shuffle=True, random_state=r)).mean())
                acc[k]["subject"].append(cross_val_score(
                    mk(), X, y, groups=s, cv=GroupKFold(5)).mean())
                acc[k]["new"].append(mk().fit(X, y).score(Xt, yt))
        for k in M:
            rows.append({"subjects": S, "per": m, "model": k,
                         **{q: float(np.mean(v)) for q, v in acc[k].items()}})
    return rows


@functools.cache
def subject_growth(*, sizes=(100, 400, 1600, 6400), fixed: int = 20,
                   reps: int = 6) -> list[dict]:
    """Accuracy on new subjects as the data grow, two ways: one row from
    each of N subjects, or N/`fixed` rows from each of `fixed` subjects."""
    np = _np()
    M = _subject_models()
    rows = []
    for N in sizes:
        for how, (S, m) in (("new subjects", (N, 1)),
                            (f"{fixed} subjects", (fixed, N // fixed))):
            acc = {k: [] for k in M}
            for r in range(int(reps)):
                X, y, _ = _subjects(S, m, np.random.default_rng(r))
                Xt, yt, _ = _subjects(2000, 5, np.random.default_rng(1000 + r))
                for k, mk in M.items():
                    acc[k].append(mk().fit(X, y).score(Xt, yt))
            rows.append({"rows": N, "how": how,
                         **{k: float(np.mean(v)) for k, v in acc.items()}})
    return rows


def same_subject_neighbours(*, S: int = 20, m: int = 100, k: int = 5,
                            reps: int = 6) -> float:
    """Under shuffled five-fold cross-validation, the share of each held-out
    row's k nearest training rows that come from the same subject."""
    np = _np()
    from sklearn.model_selection import KFold
    from sklearn.neighbors import NearestNeighbors
    share = []
    for r in range(int(reps)):
        X, _, s = _subjects(S, m, np.random.default_rng(r))
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            idx = NearestNeighbors(n_neighbors=k).fit(X[tr]).kneighbors(
                X[te], return_distance=False)
            share.append(np.mean(s[tr][idx] == s[te][:, None]))
    return float(np.mean(share))


# ================================================================ episode 38

def _breast_cancer():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return d.data, d.target


def _missing_mask(kind, X, y, g, rate=0.3):
    """Which cells go missing. 'random': every cell with probability `rate`.
    'large values': a cell's chance rises with its standardised value.
    'by diagnosis': benign tumours' cells are nine times as likely to be
    missing as malignant ones' (y = 1 is benign). Each scaled to `rate`."""
    np = _np()
    n, d = X.shape
    Z = (X - X.mean(0)) / X.std(0)
    if kind == "random":
        p = np.full((n, d), rate)
    elif kind == "large values":
        p = _sig(2 * Z - 1.2)
    elif kind == "by diagnosis":
        p = np.where(y[:, None] == 1, 0.45, 0.05) * np.ones((n, d))
    else:
        raise ValueError(kind)
    p = np.clip(p * rate / p.mean(), 0, 1)
    return g.random((n, d)) < p


def _missing_methods():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def lr():
        return LogisticRegression(max_iter=3000, C=0.3)

    def hgb():
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              random_state=0)
    return {
        "mean": lambda: make_pipeline(SimpleImputer(), StandardScaler(), lr()),
        "mean + indicator": lambda: make_pipeline(
            SimpleImputer(add_indicator=True), StandardScaler(), lr()),
        "5-NN imputation": lambda: make_pipeline(KNNImputer(), StandardScaler(),
                                                 lr()),
        "iterative imputation": lambda: make_pipeline(
            IterativeImputer(max_iter=10, random_state=0), StandardScaler(),
            lr()),
        "boosted trees, native": hgb,
    }


def _imputation_rmse(kind, reps):
    """Root mean squared error of each imputer on the missing cells, in
    standard deviations of the feature."""
    np = _np()
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
    X, y = _breast_cancer()
    Z = (X - X.mean(0)) / X.std(0)
    out = {}
    for r in range(int(reps)):
        mk = _missing_mask(kind, X, y, np.random.default_rng(r))
        Zm = np.where(mk, np.nan, Z)
        for name, imp in (("mean", SimpleImputer()),
                          ("5-NN imputation", KNNImputer()),
                          ("iterative imputation",
                           IterativeImputer(max_iter=10, random_state=0))):
            Zi = imp.fit_transform(Zm)
            out.setdefault(name, []).append(
                float(np.sqrt(np.mean((Zi[mk] - Z[mk]) ** 2))))
    return {k: float(np.mean(v)) for k, v in out.items()}


@functools.cache
def missing_study(*, reps: int = 5, rate: float = 0.3) -> dict:
    """Breast cancer with 30% of cells removed under each mechanism.
    Five-fold cross-validated log loss and accuracy of each method, averaged
    over `reps` masks and fold assignments; and, for the 'by diagnosis'
    mechanism, the same models scored on test rows whose cells went missing
    at random instead ('policy change')."""
    np = _np()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = _breast_cancer()
    methods = _missing_methods()

    def score(model, Xtr, ytr, Xte, yte):
        p = model.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        return (log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6)),
                accuracy_score(yte, p > 0.5))

    complete = []
    for r in range(int(reps)):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            complete.append(score(make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=3000, C=0.3)),
                X[tr], y[tr], X[te], y[te]))
    out = {"complete": {"nll": float(np.mean([c[0] for c in complete])),
                        "acc": float(np.mean([c[1] for c in complete]))}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind in ("random", "large values", "by diagnosis",
                     "policy change"):
            mech = "by diagnosis" if kind == "policy change" else kind
            res = {k: [] for k in methods}
            for r in range(int(reps)):
                mk = _missing_mask(mech, X, y, np.random.default_rng(r), rate)
                Xm = np.where(mk, np.nan, X)
                if kind == "policy change":
                    mt = _missing_mask("random", X, y,
                                       np.random.default_rng(100 + r), rate)
                    Xt = np.where(mt, np.nan, X)
                else:
                    Xt = Xm
                for tr, te in StratifiedKFold(5, shuffle=True,
                                              random_state=r).split(X, y):
                    for k, M in methods.items():
                        res[k].append(score(M(), Xm[tr], y[tr], Xt[te], y[te]))
            out[kind] = {k: {"nll": float(np.mean([v[0] for v in vs])),
                             "acc": float(np.mean([v[1] for v in vs]))}
                         for k, vs in res.items()}
            if kind != "policy change":
                out[kind]["_rmse"] = _imputation_rmse(kind, reps)
    return out


# ================================================================ episode 39

def _applicants(n, g, *, curved, officer):
    """Applicants with two observed features and one the model never sees,
    u. Default probability from all three (plus 0.8 x_1^2 if `curved`). The
    old policy accepts the half with the lowest risk score, a score that
    includes u when a loan officer could see it."""
    np = _np()
    X = g.normal(size=(n, 2))
    u = g.normal(size=n)
    z = (-1.5 + 1.2 * X[:, 0] + 0.8 * X[:, 1]
         + (0.8 * X[:, 0] ** 2 if curved else 0) + 1.2 * u)
    y = (g.random(n) < _sig(z)).astype(int)
    s = (1.2 * X[:, 0] + 0.8 * X[:, 1] + (1.2 * u if officer else 0)
         + 0.3 * g.normal(size=n))
    accepted = s < np.quantile(s, 0.5)
    return X, y, accepted


@functools.cache
def selective_labels(*, reps: int = 10, n: int = 20_000,
                     explore: float = 0.05) -> list[dict]:
    """For each world (straight or curved risk, officer or not) and model
    (logistic regression, boosted trees): train on the accepted half only,
    and on everyone (the oracle nobody has); score both on a fresh cohort.
    Also a logistic model trained on a randomly approved `explore` share of
    applicants only, and the accepted-only model with its intercept refitted
    on that sample."""
    np = _np()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    makers = {"logistic": lambda: LogisticRegression(max_iter=2000),
              "boosted trees": lambda: HistGradientBoostingClassifier(
                  random_state=0)}
    rows = []
    for curved in (False, True):
        for officer in (False, True):
            for name, M in makers.items():
                acc = {}
                for r in range(int(reps)):
                    g = np.random.default_rng(r)
                    X, y, a = _applicants(n, g, curved=curved, officer=officer)
                    Xt, yt, at = _applicants(
                        n, np.random.default_rng(100 + r), curved=curved,
                        officer=officer)
                    m = M().fit(X[a], y[a])
                    o = M().fit(X, y)
                    p, po = m.predict_proba(Xt)[:, 1], o.predict_proba(Xt)[:, 1]
                    vals = {
                        "refused_rate": yt[~at].mean(),
                        "refused_pred": p[~at].mean(),
                        "oracle_refused_pred": po[~at].mean(),
                        "auc_accepted": roc_auc_score(yt[at], p[at]),
                        "auc_all": roc_auc_score(yt, p),
                        "auc_oracle": roc_auc_score(yt, po),
                        "nll": log_loss(yt, p), "nll_oracle": log_loss(yt, po),
                    }
                    if name == "logistic":
                        ex = g.random(n) < explore
                        e = LogisticRegression(max_iter=2000).fit(X[ex], y[ex])
                        pe = e.predict_proba(Xt)[:, 1]
                        off = m.decision_function(X[ex])
                        b = LogisticRegression(C=1e6).fit(off[:, None], y[ex])
                        pr = b.predict_proba(m.decision_function(Xt)[:, None])[:, 1]
                        vals |= {"explore_refused_pred": pe[~at].mean(),
                                 "explore_nll": log_loss(yt, pe),
                                 "recal_refused_pred": pr[~at].mean(),
                                 "recal_nll": log_loss(yt, pr),
                                 "explore_rows": int(ex.sum())}
                    for k, v in vals.items():
                        acc.setdefault(k, []).append(float(v))
                rows.append({"curved": curved, "officer": officer,
                             "model": name,
                             **{k: float(np.mean(v)) for k, v in acc.items()}})
    return rows


# ================================================================ episode 40

def _category_world(n, K, effect, g, eff=None):
    """Five numeric features and one category with K equally likely levels
    whose effects on the log-odds are normal with sd `effect`."""
    X = g.normal(size=(n, 5))
    c = g.integers(0, K, n)
    if eff is None:
        eff = g.normal(scale=effect, size=K)
    z = X[:, 0] + 0.7 * X[:, 1] - 0.5 * X[:, 2] + eff[c]
    return X, c, (g.random(n) < _sig(z)).astype(int), eff


def _mean_encode(c, y, c_test, m=0.0):
    """Each level's mean label on the training rows, shrunk towards the
    overall mean by m pseudo-rows; unseen levels get the overall mean."""
    np = _np()
    prior = float(y.mean())
    K = int(max(c.max(), c_test.max())) + 1
    s = np.bincount(c, y, K)
    cnt = np.bincount(c, minlength=K)
    with np.errstate(invalid="ignore", divide="ignore"):
        enc = (s + m * prior) / (cnt + m)
    enc[cnt + m == 0] = prior
    return enc[c], enc[c_test]


@functools.cache
def target_encoding(*, levels=(10, 100, 1000, 4000), effects=(0.0, 1.0),
                    reps: int = 5, n: int = 4000) -> list[dict]:
    """Test AUC and log loss on 20,000 fresh rows for logistic regression and
    boosted trees given the numeric features plus the category encoded four
    ways, all fitted on the training rows only: dropped; mean label per
    level; the same shrunk by 20 pseudo-rows; and cross-fitted, each
    training row encoded by a model of the other folds (scikit-learn's
    TargetEncoder)."""
    np = _np()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import TargetEncoder
    makers = {"logistic": lambda: LogisticRegression(max_iter=2000),
              "boosted trees": lambda: HistGradientBoostingClassifier(
                  random_state=0)}
    rows = []
    for effect in effects:
        for K in levels:
            res = {}
            for r in range(int(reps)):
                X, c, y, eff = _category_world(n, K, effect,
                                               np.random.default_rng(r))
                Xt, ct, yt, _ = _category_world(
                    20_000, K, effect, np.random.default_rng(100 + r), eff)
                feats = {"dropped": (X, Xt)}
                a, b = _mean_encode(c, y, ct)
                feats["mean per level"] = (np.c_[X, a], np.c_[Xt, b])
                a, b = _mean_encode(c, y, ct, m=20)
                feats["shrunk, m = 20"] = (np.c_[X, a], np.c_[Xt, b])
                te = TargetEncoder(target_type="binary",
                                   cv=KFold(5, shuffle=True, random_state=0))
                a = te.fit_transform(c[:, None], y)
                b = te.transform(ct[:, None])
                feats["cross-fitted"] = (np.c_[X, a], np.c_[Xt, b])
                for enc, (A, B) in feats.items():
                    for name, M in makers.items():
                        p = M().fit(A, y).predict_proba(B)[:, 1]
                        res.setdefault((enc, name), []).append(
                            (roc_auc_score(yt, p),
                             log_loss(yt, np.clip(p, 1e-6, 1 - 1e-6))))
            for (enc, name), v in res.items():
                rows.append({"effect": effect, "levels": K,
                             "rows_per_level": n / K, "encoding": enc,
                             "model": name, "auc": float(np.mean([x[0] for x in v])),
                             "nll": float(np.mean([x[1] for x in v]))})
    return rows


def encoded_column_auc(*, K: int = 1000, n: int = 4000, seed: int = 0) -> dict:
    """How well the encoded noise column alone ranks the labels, on the
    training rows it was fitted on and on fresh rows, for mean-per-level and
    cross-fitted encodings."""
    np = _np()
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import TargetEncoder
    X, c, y, eff = _category_world(n, K, 0.0, np.random.default_rng(seed))
    _, ct, yt, _ = _category_world(20_000, K, 0.0,
                                   np.random.default_rng(100 + seed), eff)
    a, b = _mean_encode(c, y, ct)
    te = TargetEncoder(target_type="binary",
                       cv=KFold(5, shuffle=True, random_state=0))
    ac = te.fit_transform(c[:, None], y)[:, 0]
    bc = te.transform(ct[:, None])[:, 0]
    return {"mean_train": float(roc_auc_score(y, a)),
            "mean_test": float(roc_auc_score(yt, b)),
            "cross_train": float(roc_auc_score(y, ac)),
            "cross_test": float(roc_auc_score(yt, bc))}
