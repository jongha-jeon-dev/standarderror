r"""Arc IX: combining and comparing models. Averages, stacks, the evidence
naive Bayes multiplies together, the tests used to compare two models, and
a test set used more than once, each measured on scikit-learn's bundled
data or on a simulation with a known answer.

Episode 41, `averaging_*` and `ambiguity`: averaging the probabilities of
models of very different quality, and the identity that guarantees what an
average of regressors can and cannot do.
Episode 42, `stacking_study`: a meta-model fitted on in-sample and on
out-of-fold predictions, against the best member and a plain average.
Episode 43, `nb_*`: naive Bayes on correlated features, and on features
copied deliberately.
Episode 44, `ttest_study`: how often five tests declare one of two equally
good algorithms better.
Episode 45, `leaderboard_attack`, `noise_selection_holdout` and
`greedy_holdout`: a holdout queried many times.
"""

from __future__ import annotations

import functools
import itertools


def _np():
    import numpy as np
    return np


# ================================================================ episode 41

def _digit_members():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    return {
        "logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)),
        "SVM": lambda: make_pipeline(StandardScaler(),
                                     SVC(probability=True, random_state=0)),
        "5-NN": lambda: KNeighborsClassifier(5),
        "forest": lambda: RandomForestClassifier(300, n_jobs=2, random_state=0),
        "naive Bayes": GaussianNB,
        "tree": lambda: DecisionTreeClassifier(random_state=0),
    }


@functools.cache
def averaging_probs(*, splits: int = 5) -> dict:
    """Each member's predicted class probabilities on the held-out half of
    the digits, for `splits` stratified half splits. Also a calibrated tree:
    the same tree with isotonic recalibration by internal five-fold
    cross-validation."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    X, y = load_digits(return_X_y=True)
    members = _digit_members()
    members["calibrated tree"] = lambda: CalibratedClassifierCV(
        DecisionTreeClassifier(random_state=0), method="isotonic", cv=5)
    P = {k: [] for k in members}
    Y = []
    for s in range(int(splits)):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5, random_state=s,
                                          stratify=y)
        Y.append(yb)
        for k, M in members.items():
            P[k].append(M().fit(Xa, ya).predict_proba(Xb))
    return {"P": P, "Y": Y}


def _acc(Ps, Ys):
    np = _np()
    return float(np.mean([np.mean(p.argmax(1) == t) for p, t in zip(Ps, Ys)]))


def _nll(Ps, Ys):
    np = _np()
    from sklearn.metrics import log_loss
    return float(np.mean([log_loss(t, np.clip(p, 1e-15, 1), labels=range(10))
                          for p, t in zip(Ps, Ys)]))


def averaging_members() -> dict:
    d = averaging_probs()
    return {k: {"acc": _acc(v, d["Y"]), "nll": _nll(v, d["Y"]),
                "certain": float(_np().mean([(p.max(1) == 1).mean()
                                             for p in v]))}
            for k, v in d["P"].items()}


@functools.cache
def averaging_table() -> list[dict]:
    """Every combination of two or more of the six original members: the
    accuracy and log loss of the averaged probabilities, against the best
    and the mean member of the combination."""
    np = _np()
    d = averaging_probs()
    P, Y = d["P"], d["Y"]
    names = [k for k in P if k != "calibrated tree"]
    single = {k: (_acc(P[k], Y), _nll(P[k], Y)) for k in names}
    rows = []
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            pa = [np.mean([P[k][s] for k in combo], 0) for s in range(len(Y))]
            rows.append({
                "members": combo, "acc": _acc(pa, Y), "nll": _nll(pa, Y),
                "best_acc": max(single[k][0] for k in combo),
                "mean_acc": float(np.mean([single[k][0] for k in combo])),
                "best_nll": min(single[k][1] for k in combo),
                "mean_nll": float(np.mean([single[k][1] for k in combo]))})
    return rows


def averaging_pair(a: str, b: str) -> dict:
    """Two members averaged: accuracy, and on the rows where they disagree,
    how often the average sides with each."""
    np = _np()
    d = averaging_probs()
    P, Y = d["P"], d["Y"]
    pa = [(P[a][s] + P[b][s]) / 2 for s in range(len(Y))]
    side_a, side_b, n = 0, 0, 0
    for s in range(len(Y)):
        la, lb, lv = P[a][s].argmax(1), P[b][s].argmax(1), pa[s].argmax(1)
        dis = la != lb
        side_a += int(np.sum(lv[dis] == la[dis]))
        side_b += int(np.sum(lv[dis] == lb[dis]))
        n += int(dis.sum())
    return {"acc": _acc(pa, Y), "disagree": n, "with_a": side_a / n,
            "with_b": side_b / n}


@functools.cache
def ambiguity(*, seed: int = 0) -> dict:
    """Krogh and Vedelsby's decomposition on the diabetes data: for an equal
    average of regressors, ensemble squared error = mean member squared
    error minus mean squared spread of the members around the average,
    exactly, on every test row."""
    np = _np()
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    X, y = load_diabetes(return_X_y=True)
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5, random_state=seed)
    members = {
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "10-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(10)),
        "forest": RandomForestRegressor(300, min_samples_leaf=5, n_jobs=2,
                                        random_state=0),
        "tree": DecisionTreeRegressor(min_samples_leaf=5, random_state=0),
    }
    F = np.column_stack([m.fit(Xa, ya).predict(Xb) for m in members.values()])
    fbar = F.mean(1)
    err = ((F - yb[:, None]) ** 2).mean(0)
    ens = float(((fbar - yb) ** 2).mean())
    amb = float(((F - fbar[:, None]) ** 2).mean())
    return {"members": dict(zip(members, err.tolist())),
            "ensemble": ens, "mean_member": float(err.mean()),
            "ambiguity": amb, "identity_gap": abs(ens - (err.mean() - amb))}


# ================================================================ episode 42

def _stack_pool(with_1nn: bool):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    pool = {"logistic": lambda: LogisticRegression(max_iter=2000),
            "SVM": lambda: SVC(probability=True, random_state=0),
            "forest": lambda: RandomForestClassifier(200, n_jobs=2,
                                                     random_state=0)}
    pool["1-NN" if with_1nn else "25-NN"] = (
        (lambda: KNeighborsClassifier(1)) if with_1nn
        else (lambda: KNeighborsClassifier(25)))
    return pool


@functools.cache
def stacking_study(*, with_1nn: bool = True,
                   sizes=((100, 20), (300, 10), (1000, 6), (3000, 3))) -> list[dict]:
    """Episode 4's task. Base models trained on n rows; a logistic
    meta-model on their predicted probabilities, fitted either on the
    in-sample predictions or on five-fold out-of-fold ones; and the plain
    average. Accuracy and log loss on 20,000 fixed test rows, averaged over
    the repetitions given with each size."""
    np = _np()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from standarderror.ml.curves import GaussianTask
    t = GaussianTask(test=20000)
    pool = _stack_pool(with_1nn)
    rows = []

    def sc(p):
        return (accuracy_score(t.y_test, p > 0.5),
                log_loss(t.y_test, np.clip(p, 1e-6, 1 - 1e-6)))
    for n, reps in sizes:
        res, weights = {}, {"in-sample": [], "out-of-fold": []}
        for r in range(int(reps)):
            X, y = t.draw(n, seed=r)
            Pin, Poof, Pte = [], [], []
            for k, M in pool.items():
                m = M().fit(X, y)
                Pin.append(m.predict_proba(X)[:, 1])
                Pte.append(m.predict_proba(t.X_test)[:, 1])
                Poof.append(cross_val_predict(
                    M(), X, y, method="predict_proba",
                    cv=StratifiedKFold(5, shuffle=True, random_state=r))[:, 1])
                res.setdefault(k, []).append(sc(Pte[-1]))
            Pin, Poof, Pte = map(np.column_stack, (Pin, Poof, Pte))
            for name, Ptr in (("in-sample", Pin), ("out-of-fold", Poof)):
                meta = LogisticRegression(C=1e4, max_iter=5000).fit(Ptr, y)
                res.setdefault(f"stack, {name}", []).append(
                    sc(meta.predict_proba(Pte)[:, 1]))
                weights[name].append(meta.coef_[0])
            res.setdefault("average", []).append(sc(Pte.mean(1)))
        best = max(pool, key=lambda k: np.mean([v[0] for v in res[k]]))
        rows.append({"n": n, "reps": reps, "best": best,
                     "acc": {k: float(np.mean([v[0] for v in vs]))
                             for k, vs in res.items()},
                     "nll": {k: float(np.mean([v[1] for v in vs]))
                             for k, vs in res.items()},
                     "weights": {k: dict(zip(pool, np.mean(w, 0).tolist()))
                                 for k, w in weights.items()}})
    return rows


# ================================================================ episode 43

def _bc():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return d.data, d.target, list(d.feature_names)


def _cv_scores(make, X, y, *, reps: int = 5, log_odds: bool = False):
    """Five-fold out-of-fold probabilities (and naive Bayes log-odds if
    asked), repeated over `reps` fold assignments; returns the metrics
    averaged over repetitions."""
    np = _np()
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    out = []
    for r in range(int(reps)):
        p = np.zeros(len(y))
        lo = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            m = make().fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
            if log_odds:
                j = m.predict_joint_log_proba(X[te])
                lo[te] = j[:, 1] - j[:, 0]
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        wrong = (p > 0.5) != y
        conf = np.maximum(p, 1 - p)
        row = [accuracy_score(y, p > 0.5), roc_auc_score(y, p), log_loss(y, pc),
               float(np.mean(conf[wrong] > 0.99)) if wrong.any() else 0.0,
               float(np.mean((p == 0) | (p == 1)))]
        row.append(roc_auc_score(y, lo) if log_odds else float("nan"))
        out.append(row)
    a = np.mean(out, 0)
    return {"acc": float(a[0]), "auc": float(a[1]), "nll": float(a[2]),
            "confident_errors": float(a[3]), "exact_01": float(a[4]),
            "auc_log_odds": float(a[5])}


@functools.cache
def nb_baseline() -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    np = _np()
    X, y, names = _bc()
    C = np.corrcoef(X[y == 0].T)
    iu = np.triu_indices(X.shape[1], 1)
    return {"nb": _cv_scores(GaussianNB, X, y, log_odds=True),
            "logistic": _cv_scores(lambda: make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=3000)), X, y),
            "median_abs_corr": float(np.median(np.abs(C[iu]))),
            "share_above_09": float(np.mean(np.abs(C[iu]) > 0.9))}


@functools.cache
def nb_duplicates(*, copies=(1, 2, 3, 5, 10)) -> list[dict]:
    """Every feature copied k times."""
    from sklearn.naive_bayes import GaussianNB
    np = _np()
    X, y, _ = _bc()
    return [{"k": k, **_cv_scores(GaussianNB, np.tile(X, k), y, log_odds=True)}
            for k in copies]


@functools.cache
def nb_family(*, copies=(2, 5, 10)) -> list[dict]:
    """One family of features copied k times: the nine size measurements
    (radius, perimeter, area: mean, error, worst) or the three textures."""
    from sklearn.naive_bayes import GaussianNB
    np = _np()
    X, y, names = _bc()
    fam = {"size": [i for i, n in enumerate(names)
                    if any(w in n for w in ("radius", "perimeter", "area"))],
           "texture": [i for i, n in enumerate(names) if "texture" in n]}
    rows = []
    for f, idx in fam.items():
        for k in copies:
            Xk = np.c_[X, np.tile(X[:, idx], k - 1)]
            rows.append({"family": f, "features": len(idx), "k": k,
                         **_cv_scores(GaussianNB, Xk, y, log_odds=True)})
    return rows


class _PlattNB:
    """Gaussian naive Bayes whose log-odds are recalibrated by a logistic
    regression fitted on out-of-fold log-odds of the training rows."""

    def fit(self, X, y):
        np = _np()
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.naive_bayes import GaussianNB
        lo = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            j = GaussianNB().fit(X[tr], y[tr]).predict_joint_log_proba(X[te])
            lo[te] = j[:, 1] - j[:, 0]
        self.nb_ = GaussianNB().fit(X, y)
        self.cal_ = LogisticRegression(C=1e6).fit(lo[:, None], y)
        return self

    def predict_joint_log_proba(self, X):
        return self.nb_.predict_joint_log_proba(X)

    def predict_proba(self, X):
        j = self.nb_.predict_joint_log_proba(X)
        return self.cal_.predict_proba((j[:, 1] - j[:, 0])[:, None])


@functools.cache
def nb_repairs() -> dict:
    """Three repairs: recalibrate the log-odds (Platt, fitted out-of-fold
    inside each training set); decorrelate the features first (standardise,
    PCA with whitening); use only three weakly related features."""
    from sklearn.decomposition import PCA
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y, names = _bc()
    three = [names.index(n) for n in ("mean radius", "mean texture",
                                      "mean smoothness")]
    return {"platt": _cv_scores(_PlattNB, X, y, log_odds=True),
            "whitened": _cv_scores(lambda: make_pipeline(
                StandardScaler(), PCA(whiten=True), GaussianNB()), X, y),
            "three": _cv_scores(GaussianNB, X[:, three], y, log_odds=True)}


# ================================================================ episode 44

def _ttask(n, g, d=4):
    """Two classes, both centred at the origin in d dimensions, sd 1 and
    1.6: the Bayes boundary is a sphere, and every rotation of the space
    leaves the task's distribution unchanged."""
    np = _np()
    y = g.integers(0, 2, n)
    X = g.normal(size=(n, d)) * np.where(y[:, None] == 1, 1.6, 1.0)
    return X, y


@functools.cache
def ttest_study(*, reps: int = 400, n: int = 300, depth_b: int = 4) -> dict:
    """Algorithm A: a depth-4 decision tree. Algorithm B: a depth-`depth_b`
    tree fitted after a fixed random rotation of the inputs. With
    depth_b = 4 the two have exactly the same expected accuracy at every
    training size, by symmetry, though on any one data set either can be
    better. For each of `reps` data sets of n rows, the p-value of five
    tests of 'A and B are equally accurate', and the true difference of the
    two trees fitted on all n rows, from 50,000 fresh rows."""
    np = _np()
    from scipy import stats
    from scipy.stats import special_ortho_group
    from sklearn.model_selection import KFold
    from sklearn.tree import DecisionTreeClassifier
    R = special_ortho_group.rvs(4, random_state=0)
    Xbig, ybig = _ttask(50_000, np.random.default_rng(999))

    def A():
        return DecisionTreeClassifier(max_depth=4, random_state=0)

    def B():
        return DecisionTreeClassifier(max_depth=depth_b, random_state=0)

    def diff(X, y, tr, te):
        a = A().fit(X[tr], y[tr]).score(X[te], y[te])
        b = B().fit(X[tr] @ R, y[tr]).score(X[te] @ R, y[te])
        return a - b

    def tp(t, df):
        return float(2 * stats.t.sf(abs(t), df))

    out = {k: [] for k in ("true", "10-fold t", "10x10 t", "10x10 corrected",
                           "5x2cv", "McNemar", "cv10_mean")}
    for r in range(int(reps)):
        X, y = _ttask(n, np.random.default_rng(r))
        out["true"].append(A().fit(X, y).score(Xbig, ybig)
                           - B().fit(X @ R, y).score(Xbig @ R, ybig))
        d10 = np.array([diff(X, y, tr, te) for tr, te in
                        KFold(10, shuffle=True, random_state=r).split(X)])
        out["cv10_mean"].append(float(d10.mean()))
        out["10-fold t"].append(tp(d10.mean() / (d10.std(ddof=1) / np.sqrt(10)
                                                 + 1e-12), 9))
        D = np.array([diff(X, y, tr, te) for j in range(10) for tr, te in
                      KFold(10, shuffle=True,
                            random_state=1000 * r + j).split(X)])
        m, v = D.mean(), D.var(ddof=1)
        out["10x10 t"].append(tp(m / np.sqrt(v / 100 + 1e-12), 99))
        out["10x10 corrected"].append(
            tp(m / np.sqrt(v * (1 / 100 + 1 / 9) + 1e-12), 99))
        P = []
        for j in range(5):
            idx = np.random.default_rng(7 * r + j).permutation(n)
            h = n // 2
            P.append([diff(X, y, idx[:h], idx[h:]), diff(X, y, idx[h:], idx[:h])])
        P = np.array(P)
        s2 = ((P - P.mean(1, keepdims=True)) ** 2).sum(1)
        out["5x2cv"].append(tp(P[0, 0] / np.sqrt(s2.mean() + 1e-12), 5))
        idx = np.random.default_rng(r + 77).permutation(n)
        tr, te = idx[:200], idx[200:]
        ca = A().fit(X[tr], y[tr]).predict(X[te]) == y[te]
        cb = B().fit(X[tr] @ R, y[tr]).predict(X[te] @ R) == y[te]
        b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
        out["McNemar"].append(1.0 if b + c == 0
                              else float(stats.binomtest(b, b + c).pvalue))
    true = np.array(out["true"])
    tests = ("10-fold t", "10x10 t", "10x10 corrected", "5x2cv", "McNemar")
    rej = {k: float(np.mean(np.array(out[k]) < 0.05)) for k in tests}
    se = {k: float(np.sqrt(rej[k] * (1 - rej[k]) / reps)) for k in tests}
    p10 = np.array(out["10-fold t"])
    sign_ok = np.sign(np.array(out["cv10_mean"])) == np.sign(true)
    return {"reps": int(reps), "depth_b": int(depth_b),
            "true_mean": float(true.mean()), "true_sd": float(true.std()),
            "true_abs_above_002": float(np.mean(np.abs(true) > 0.02)),
            "reject": rej, "reject_se": se,
            "naive_sign_right": float(np.mean(sign_ok[p10 < 0.05])),
            "naive_reject_big_true": float(np.mean(
                p10[np.abs(true) > 0.02] < 0.05)),
            "naive_reject_small_true": float(np.mean(
                p10[np.abs(true) < 0.005] < 0.05))}


# ================================================================ episode 45

@functools.cache
def leaderboard_attack(*, queries=(1, 10, 100, 300, 1000, 3000), n: int = 1000,
                       reps: int = 50) -> list[dict]:
    """Blum and Hardt's attack on a public leaderboard of n coin-flip
    labels: submit k random guesses, keep those that score above 0.5 on the
    public set, submit their majority vote. Its public score, and its score
    on a private set of n other coin-flip labels the guesses also covered."""
    np = _np()
    rows = []
    for k in queries:
        pub, prv = [], []
        for r in range(int(reps)):
            g = np.random.default_rng(r)
            yp, yq = g.integers(0, 2, n), g.integers(0, 2, n)
            S = g.integers(0, 2, (int(k), 2 * n))
            good = (S[:, :n] == yp).mean(1) > 0.5
            v = (S[good].mean(0) > 0.5).astype(int) if good.any() else S[0]
            pub.append((v[:n] == yp).mean())
            prv.append((v[n:] == yq).mean())
        rows.append({"queries": int(k), "public": float(np.mean(pub)),
                     "private": float(np.mean(prv))})
    return rows


@functools.cache
def noise_selection_holdout(*, n: int = 1000, d: int = 2000, reps: int = 20,
                            sizes=(10, 20, 50, 100)) -> list[dict]:
    """Dwork and colleagues' demonstration, scaled down: d noise features,
    labels ±1 at random. Keep the features whose correlation with the label
    exceeds 1/sqrt(n) with the same sign on the training set and on the
    holdout; classify by the sign of the sum of the k strongest, signed by
    their training correlation. Accuracy on the holdout and on fresh rows."""
    np = _np()
    out = {k: [] for k in sizes}
    for r in range(int(reps)):
        g = np.random.default_rng(r)
        Xtr, Xho, Xfr = (g.normal(size=(n, d)) for _ in range(3))
        ytr, yho, yfr = (g.choice([-1, 1], n) for _ in range(3))
        ctr, cho = Xtr.T @ ytr / n, Xho.T @ yho / n
        thr = 1 / np.sqrt(n)
        ok = ((np.abs(ctr) > thr) & (np.abs(cho) > thr)
              & (np.sign(ctr) == np.sign(cho)))
        order = np.argsort(-np.abs(ctr) * ok)
        for k in sizes:
            sel = order[:k]
            sel = sel[ok[sel]]
            w = np.sign(ctr[sel])
            out[k].append(((np.sign(Xho[:, sel] @ w) == yho).mean(),
                           (np.sign(Xfr[:, sel] @ w) == yfr).mean(), len(sel)))
    return [{"k": k, "holdout": float(np.mean([v[0] for v in vs])),
             "fresh": float(np.mean([v[1] for v in vs])),
             "kept": float(np.mean([v[2] for v in vs]))}
            for k, vs in out.items()]


def _greedy_world(n, g, d=100, real=10):
    np = _np()
    X = g.normal(size=(n, d))
    w = np.zeros(d)
    w[:real] = 0.35
    y = (g.random(n) < 1 / (1 + np.exp(-X @ w))).astype(int)
    return X, y


@functools.cache
def greedy_holdout(*, by: str = "holdout", reps: int = 5, steps: int = 20,
                   n: int = 300) -> dict:
    """Forward selection of features for logistic regression, 10 real
    features among 100. At each step add the feature that most improves
    accuracy on the holdout (`by='holdout'`) or five-fold cross-validated
    accuracy on the training rows (`by='cv'`). After each step: holdout
    accuracy, accuracy on 20,000 fresh rows, and the share of selected
    features that are real."""
    np = _np()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    H, F, C = (np.zeros((int(reps), int(steps))) for _ in range(3))
    oracle = []
    for r in range(int(reps)):
        g = np.random.default_rng(r)
        Xtr, ytr = _greedy_world(n, g)
        Xho, yho = _greedy_world(n, g)
        Xfr, yfr = _greedy_world(20_000, g)
        oracle.append(LogisticRegression(max_iter=1000).fit(
            Xtr[:, :10], ytr).score(Xfr[:, :10], yfr))
        sel: list[int] = []
        for s in range(int(steps)):
            best, bv = None, -1.0
            for j in range(Xtr.shape[1]):
                if j in sel:
                    continue
                cols = sel + [j]
                if by == "holdout":
                    v = LogisticRegression(max_iter=1000).fit(
                        Xtr[:, cols], ytr).score(Xho[:, cols], yho)
                else:
                    v = cross_val_score(
                        LogisticRegression(max_iter=1000), Xtr[:, cols], ytr,
                        cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
                if v > bv:
                    best, bv = j, v
            sel.append(best)
            m = LogisticRegression(max_iter=1000).fit(Xtr[:, sel], ytr)
            H[r, s] = m.score(Xho[:, sel], yho)
            F[r, s] = m.score(Xfr[:, sel], yfr)
            C[r, s] = np.mean(np.array(sel) < 10)
    return {"by": by, "holdout": H.mean(0).tolist(), "fresh": F.mean(0).tolist(),
            "real_share": C.mean(0).tolist(), "oracle": float(np.mean(oracle))}

