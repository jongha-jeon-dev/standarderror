r"""What a score means: the error bar on a test score, and what CV estimates.

**Episode 1.** On the UCI digits (1,797 images, 10 classes) a 20% test split
holds 360 images. At 98% accuracy the binomial standard error of one test
score is 0.007 -- larger than the gap between two good models. Repeating the
split 200 times reproduces that spread almost exactly, but for a subtler
reason than it seems: random splits of one fixed pool measure the test-set
noise *with* a finite-population correction, and the training-set variation
the repetition was meant to expose barely shows. Comparing two models on the
same test rows (paired) is the one move that sharpens the comparison, and
even with every image used once through cross-validation, kNN's 0.6-point win
over an RBF SVM does not clear p = 0.05.

**Episode 2.** Cross-validation is usually described as estimating the error
of the model you fit. On simulated linear regression, where that error can be
computed exactly, the correlation between the CV estimate and the error of
the model fitted to the same data is 0.01 over 2,000 datasets. CV estimates
the *average* error over training sets of that size, and the naive interval
around it covers the fitted model's error 81% of the time at a nominal 90%. A
held-out test set answers the conditional question; CV answers the average
one. Bates, Hastie and Tibshirani, "Cross-validation: what does it estimate
and how well does it do it?", *JASA* (2023), is the reference result.
"""

from __future__ import annotations

import math

#: A paired comparison's two-sided 5% test at 80% power.
Z_ALPHA, Z_POWER = 1.959964, 0.841621


def _np():
    import numpy as np
    return np


def digits():
    """The UCI optical digits as scikit-learn bundles them: 1,797 8x8
    images, 10 classes. CC BY 4.0; no download."""
    from sklearn.datasets import load_digits
    return load_digits(return_X_y=True)


def models() -> dict:
    """Three ordinary classifiers that all reach 97-99% on digits."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return {
        "logistic": lambda: make_pipeline(StandardScaler(),
                                          LogisticRegression(max_iter=2000)),
        "RBF SVM": lambda: make_pipeline(StandardScaler(), SVC()),
        "3-NN": lambda: KNeighborsClassifier(3),
    }


def binomial_se(p: float, n: int) -> float:
    """Standard error of an accuracy `p` measured on `n` independent rows."""
    return math.sqrt(p * (1 - p) / n)


def pool_se(p: float, n: int, pool: int) -> float:
    """The same, for `n` rows drawn without replacement from a fixed `pool`:
    the finite-population correction `(pool - n) / (pool - 1)`."""
    return math.sqrt(p * (1 - p) / n * (pool - n) / (pool - 1))


def split_scores(X, y, *, splits: int = 200, test: float = 0.2,
                 seed: int = 0) -> dict:
    """Accuracy of every model on `splits` stratified random splits."""
    np = _np()
    from sklearn.model_selection import train_test_split
    ms = models()
    acc = {k: np.empty(int(splits)) for k in ms}
    n_test = None
    for s in range(int(splits)):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test, random_state=int(seed) + s, stratify=y)
        n_test = len(yte)
        for k, make in ms.items():
            acc[k][s] = float((make().fit(Xtr, ytr).predict(Xte)
                               == yte).mean())
    return {"accuracy": acc, "n_test": int(n_test), "pool": int(len(y)),
            "splits": int(splits)}


def cv_correct(X, y, *, folds: int = 10, seed: int = 0) -> dict:
    """For every row, whether each model got it right when it was held out:
    every image used once as test data, on the same folds for every model."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(int(folds), shuffle=True, random_state=int(seed))
    return {k: cross_val_predict(make(), X, y, cv=cv) == y
            for k, make in models().items()}


def paired(a, b) -> dict:
    """Compare two models on the same rows. `a` and `b` are boolean
    correctness vectors; only the rows where they disagree carry information,
    which is what McNemar's exact test uses."""
    np = _np()
    from scipy import stats
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    d = a.astype(float) - b.astype(float)
    n = len(d)
    discordant = only_a + only_b
    p = (stats.binomtest(min(only_a, only_b), discordant).pvalue
         if discordant else 1.0)
    return {"n": n, "difference": float(d.mean()),
            "paired_se": float(d.std(ddof=1) / math.sqrt(n)),
            "unpaired_se": math.sqrt(binomial_se(a.mean(), n) ** 2
                                     + binomial_se(b.mean(), n) ** 2),
            "only_a": only_a, "only_b": only_b,
            "discordance": discordant / n, "p_value": float(p)}


def rows_needed(difference: float, discordance: float) -> int:
    """Test rows a paired comparison needs to detect `difference` at 5%
    with 80% power, when a fraction `discordance` of rows are disagreements.
    The normal approximation to McNemar: n = (z_a + z_b)^2 d / delta^2."""
    return math.ceil((Z_ALPHA + Z_POWER) ** 2 * discordance
                     / difference ** 2)


def subset_sd(correct, n_test: int, *, draws: int = 5000,
              seed: int = 0) -> float:
    """The spread of accuracy over random `n_test`-row subsets of a fixed
    correctness vector: pure test-set noise, with the training set held
    fixed. The control that says how much of a repeated-split spread is the
    test set and how much is the model."""
    np = _np()
    c = np.asarray(correct, float)
    rng = np.random.default_rng(int(seed))
    return float(np.std([c[rng.choice(len(c), int(n_test), replace=False)]
                         .mean() for _ in range(int(draws))]))


# ------------------------------------------------------------ episode 2

def linear_cv_study(*, n: int = 100, p: int = 20, sigma: float = 1.0,
                    folds: int = 10, reps: int = 2000, holdout: int = 100,
                    seed: int = 0) -> dict:
    """K-fold CV against the exact error of the model it accompanies.

    Least squares on `n` rows of `p` standard-normal features with true
    coefficients 0.5 and noise `sigma`. Because the features are isotropic,
    the expected squared error of a fitted `b` on a fresh row is exactly
    `sigma^2 + |b - beta|^2` -- the conditional error `err_xy` of *this*
    fit, no estimate involved. Each replicate also scores the fit on
    `holdout` fresh rows, the other way to answer the question, and -- on
    the same budget as CV -- fits on the first 80% of the `n` rows and
    scores that fit on the last 20%.
    """
    np = _np()
    rng = np.random.default_rng(int(seed))
    beta = np.full(int(p), 0.5)
    cv, cv_se, err, ho, ho_se = (np.empty(int(reps)) for _ in range(5))
    # The fair-budget alternative: hold out a fifth of the same n rows.
    cut = int(n * 0.8)
    sp_err, sp_ho, sp_se = (np.empty(int(reps)) for _ in range(3))
    for r in range(int(reps)):
        X = rng.standard_normal((n, p))
        y = X @ beta + sigma * rng.standard_normal(n)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        err[r] = sigma ** 2 + float(np.sum((b - beta) ** 2))
        loss = np.empty(n)
        for f in np.array_split(rng.permutation(n), int(folds)):
            m = np.ones(n, bool)
            m[f] = False
            bf = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
            loss[f] = (y[f] - X[f] @ bf) ** 2
        cv[r], cv_se[r] = loss.mean(), loss.std(ddof=1) / math.sqrt(n)
        Xh = rng.standard_normal((holdout, p))
        lh = (Xh @ beta + sigma * rng.standard_normal(holdout) - Xh @ b) ** 2
        ho[r], ho_se[r] = lh.mean(), lh.std(ddof=1) / math.sqrt(holdout)
        b8 = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)[0]
        sp_err[r] = sigma ** 2 + float(np.sum((b8 - beta) ** 2))
        l8 = (y[cut:] - X[cut:] @ b8) ** 2
        sp_ho[r], sp_se[r] = l8.mean(), l8.std(ddof=1) / math.sqrt(n - cut)
    return {"cv": cv, "cv_se": cv_se, "err_xy": err, "err": float(err.mean()),
            "holdout": ho, "holdout_se": ho_se, "n": int(n), "p": int(p),
            "split_err_xy": sp_err, "split_holdout": sp_ho,
            "split_se": sp_se, "split_train": cut,
            "folds": int(folds), "reps": int(reps),
            "holdout_rows": int(holdout)}


def coverage(estimate, se, target, *, level: float = 0.9) -> float:
    """How often `estimate +- z se` contains `target` (array or scalar)."""
    np = _np()
    from scipy import stats
    z = stats.norm.ppf(0.5 + level / 2)
    lo, hi = estimate - z * se, estimate + z * se
    return float(np.mean((lo <= target) & (target <= hi)))


def digits_cv_study(*, train: int = 300, test: int = 797, folds: int = 10,
                    reps: int = 200, seed: int = 0) -> dict:
    """The same question on real data. The digits are split once into a
    fixed `test` set and a training pool; each replicate draws `train` digits
    from the pool, cross-validates logistic regression on them, fits it to
    all of them, and scores that fit on the fixed test set -- large enough
    to stand in for the fitted model's true accuracy, and disjoint from every
    training draw, so a draw that happens to take the hard digits cannot
    leave the test set easier."""
    np = _np()
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X, y = digits()
    make = models()["logistic"]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(y))
    te, pool = perm[:int(test)], perm[int(test):]
    cv, true = np.empty(int(reps)), np.empty(int(reps))
    for r in range(int(reps)):
        tr = rng.choice(pool, int(train), replace=False)
        cv[r] = cross_val_score(make(), X[tr], y[tr], cv=StratifiedKFold(
            int(folds), shuffle=True, random_state=r)).mean()
        true[r] = float((make().fit(X[tr], y[tr]).predict(X[te])
                         == y[te]).mean())
    return {"cv": cv, "true": true, "train": int(train),
            "test": int(test), "pool": int(len(pool)), "reps": int(reps),
            "corr": float(np.corrcoef(cv, true)[0, 1])}
