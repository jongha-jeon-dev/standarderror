r"""Which preprocessing steps leak, and by how much.

"Fit every preprocessing step inside the cross-validation loop" is correct
advice delivered as if every violation were equally bad. Measured, they are
not. A step that never looks at the label -- standardising, mean imputation
-- leaks almost nothing when fitted on all the data. A step that looks at the
label leaks in proportion to how much it can fit: selecting 20 of 5,000
pure-noise features on all 100 rows turns a coin flip into 0.89 cross-
validated accuracy. And duplicated rows leak in proportion to how local the
model is: nothing much for logistic regression, everything for 1-NN.

The rule the measurements support: *a step leaks to the extent that it uses
the label, or lets a test row see a copy of itself.* Ambroise and McLachlan,
"Selection bias in gene extraction on the basis of microarray gene-expression
data", *PNAS* (2002), is the classic case.
"""

from __future__ import annotations


def _np():
    import numpy as np
    return np


def breast_cancer():
    """Wisconsin diagnostic breast cancer as scikit-learn bundles it: 569
    rows, 30 features, binary. CC BY 4.0; no download."""
    from sklearn.datasets import load_breast_cancer
    return load_breast_cancer(return_X_y=True)


def _logistic():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=5000)


def _cv(model, X, y, seed):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    return float(cross_val_score(model, X, y, cv=StratifiedKFold(
        5, shuffle=True, random_state=int(seed))).mean())


def noise_selection(*, n: int = 100, p: int = 5000, k: int = 20,
                    reps: int = 50, seed: int = 0) -> dict:
    """Select `k` of `p` pure-noise features by an F-test, then cross-validate
    logistic regression on them -- once with the selection fitted on all the
    rows (leaky) and once inside each fold (honest). The labels are coin
    flips, so the true accuracy of anything is 0.5."""
    np = _np()
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import make_pipeline
    rng = np.random.default_rng(int(seed))
    leak, honest = np.empty(int(reps)), np.empty(int(reps))
    for r in range(int(reps)):
        X = rng.standard_normal((n, p))
        y = rng.integers(0, 2, n)
        Xs = SelectKBest(f_classif, k=k).fit_transform(X, y)
        leak[r] = _cv(_logistic(), Xs, y, r)
        honest[r] = _cv(make_pipeline(SelectKBest(f_classif, k=k),
                                      _logistic()), X, y, r)
    return {"leaky": leak, "honest": honest, "n": n, "p": p, "k": k,
            "reps": int(reps)}


def selection_sweep(*, candidates=(20, 50, 200, 1000, 5000), n: int = 100,
                    k: int = 20, reps: int = 30, seed: int = 0) -> list[dict]:
    """The leaky selection again, with the number of pure-noise candidates
    swept. With `k` candidates there is nothing to select and nothing to
    leak; every extra candidate is another chance to find a column that
    happens to line up with the coin flips."""
    out = []
    for p in candidates:
        r = noise_selection(n=n, p=int(p), k=k, reps=reps, seed=seed)
        out.append({"candidates": int(p), "leaky": float(r["leaky"].mean()),
                    "honest": float(r["honest"].mean()),
                    "leaky_sd": float(r["leaky"].std(ddof=1))})
    return out


class _TargetEncoder:
    """Replace a categorical column by the mean label of its category, as
    learned from whatever rows it is fitted on. Unseen categories get the
    overall mean."""

    def __init__(self, column: int):
        self.column = column

    def fit(self, X, y):
        np = _np()
        c = X[:, self.column]
        self.prior_ = float(np.mean(y))
        self.means_ = {v: float(y[c == v].mean()) for v in np.unique(c)}
        return self

    def transform(self, X):
        X = X.copy()
        X[:, self.column] = [self.means_.get(v, self.prior_)
                             for v in X[:, self.column]]
        return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    # scikit-learn's pipeline needs these to clone the step.
    def get_params(self, deep=True):
        return {"column": self.column}

    def set_params(self, **kw):
        self.column = kw.get("column", self.column)
        return self


def leak_table(*, reps: int = 20, seed: int = 0) -> list[dict]:
    """Six ways to fit something outside the folds, each next to its honest
    version, on the breast cancer data (logistic regression unless noted,
    5-fold CV averaged over `reps` fold assignments).

    1. standardise on all rows
    2. impute 20% missing-at-random values with all-row means
    3. add a 200-level column of pure noise and target-encode it on all rows
    4. select 5 features by F-test on all rows, from the 30 real ones plus
       1,000 noise columns
    5. duplicate every row, so copies straddle folds (logistic)
    6. the same duplication, scored with 1-nearest-neighbour
    """
    np = _np()
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = breast_cancer()
    rng = np.random.default_rng(int(seed))
    rows = []

    def both(name, label_used, leaky_fn, honest_fn):
        lk = [leaky_fn(r) for r in range(int(reps))]
        hn = [honest_fn(r) for r in range(int(reps))]
        rows.append({"leak": name, "uses_label": label_used,
                     "leaky": float(np.mean(lk)), "honest": float(np.mean(hn)),
                     "gap": float(np.mean(lk) - np.mean(hn))})

    std = make_pipeline(StandardScaler(), _logistic())
    Xs = StandardScaler().fit_transform(X)
    both("standardise on all rows", False,
         lambda r: _cv(_logistic(), Xs, y, r), lambda r: _cv(std, X, y, r))

    Xm = X.copy()
    Xm[rng.random(X.shape) < 0.2] = np.nan
    Xi = SimpleImputer().fit_transform(Xm)
    both("impute means on all rows", False,
         lambda r: _cv(make_pipeline(StandardScaler(), _logistic()), Xi, y, r),
         lambda r: _cv(make_pipeline(SimpleImputer(), StandardScaler(),
                                     _logistic()), Xm, y, r))

    cat = rng.integers(0, 200, len(y)).astype(float)
    Xc = np.column_stack([X, cat])
    col = X.shape[1]
    Xe = _TargetEncoder(col).fit_transform(Xc, y)
    both("target-encode a noise ID on all rows", True,
         lambda r: _cv(make_pipeline(StandardScaler(), _logistic()), Xe, y, r),
         lambda r: _cv(make_pipeline(_TargetEncoder(col), StandardScaler(),
                                     _logistic()), Xc, y, r))

    Xn = np.column_stack([X, rng.standard_normal((len(y), 1000))])
    Xsel = SelectKBest(f_classif, k=5).fit_transform(Xn, y)
    both("select 5 of 1,030 features on all rows", True,
         lambda r: _cv(make_pipeline(StandardScaler(), _logistic()), Xsel, y,
                       r),
         lambda r: _cv(make_pipeline(SelectKBest(f_classif, k=5),
                                     StandardScaler(), _logistic()), Xn, y, r))

    Xd, yd = np.vstack([X, X]), np.concatenate([y, y])
    group = np.concatenate([np.arange(len(y))] * 2)
    both("duplicate rows across folds (logistic)", False,
         lambda r: _cv(std, Xd, yd, r),
         lambda r: _grouped_cv(std, Xd, yd, group, r))
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(1))
    both("duplicate rows across folds (1-NN)", False,
         lambda r: _cv(knn, Xd, yd, r),
         lambda r: _grouped_cv(knn, Xd, yd, group, r))
    return rows


def _grouped_cv(model, X, y, group, seed):
    """Five folds that keep both copies of a row on the same side."""
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    return float(cross_val_score(model, X, y, groups=group,
                                 cv=StratifiedGroupKFold(
                                     5, shuffle=True,
                                     random_state=int(seed))).mean())
