r"""Learning curves, and what extrapolating them can and cannot tell you.

**Episode 4.** "A learning curve tells you whether more data will help" is
usually put into practice by fitting a power law to errors measured on small
training sets and reading off the error at a larger one. Measured on a
simulated task where the training set can be grown to 25,600 rows and the
Bayes error is known (0.268), the two-parameter power law ``a n^-b`` is
optimistic for every model and, for 15-NN, predicts an error *below* the
Bayes error -- a number no classifier can reach. Adding an asymptote
``a n^-b + c`` fixes the models whose curves have already bent, and fails
for the one that has not: for boosted trees the fitted floor moves with the
fit range and lands below the Bayes error. And the model that is best at
200 rows is not the model that is best at 25,600.

On the UCI digits the same pattern holds within the 1,200 training images
available: a power law fitted up to 300 underpredicts the error at 1,200,
and logistic regression, best at 100 images, is worst at 1,200.

Viering and Loog, "The shape of learning curves: a review", *IEEE TPAMI*
(2023), surveys the ways real curves depart from a power law; Hestness et
al., "Deep learning scaling is predictable, empirically" (2017), is the
optimistic counterpart, fitted far further along the curve than a small
pilot study gets.
"""

from __future__ import annotations

import math


def _np():
    import numpy as np
    return np


# ------------------------------------------------------------ the task

class GaussianTask:
    """Two classes in `d` dimensions with different covariances, so the
    Bayes boundary is quadratic: a linear model cannot reach it, a flexible
    one can, slowly. The Bayes error is computed from the true densities on
    a large fixed test set, the same set every model is scored on."""

    def __init__(self, d: int = 10, test: int = 20_000, seed: int = 0):
        np = _np()
        from scipy.stats import multivariate_normal as mvn
        rng = np.random.default_rng(int(seed))
        a = rng.standard_normal((d, d)) * 0.3
        self.d = int(d)
        self.cov0 = np.eye(d)
        self.cov1 = np.eye(d) * 0.5 + a @ a.T * 0.5
        self.mu1 = np.r_[0.6, np.zeros(d - 1)]
        self.X_test, self.y_test = self.draw(test, seed=99)
        l0 = mvn(np.zeros(d), self.cov0).logpdf(self.X_test)
        l1 = mvn(self.mu1, self.cov1).logpdf(self.X_test)
        self.bayes = float(np.mean((l1 > l0).astype(int) != self.y_test))

    def draw(self, n: int, *, seed: int):
        np = _np()
        rng = np.random.default_rng(int(seed))
        y = rng.integers(0, 2, int(n))
        x0 = rng.multivariate_normal(np.zeros(self.d), self.cov0, int(n))
        x1 = rng.multivariate_normal(self.mu1, self.cov1, int(n))
        return np.where(y[:, None] == 0, x0, x1), y


def task_models() -> dict:
    """A misspecified linear model and two flexible ones. Early stopping is
    switched off in the boosted trees: scikit-learn turns it on by default
    above 10,000 rows, which would change the procedure partway along the
    curve."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    return {
        "logistic": lambda: LogisticRegression(max_iter=2000),
        "15-NN": lambda: KNeighborsClassifier(15),
        "boosted trees": lambda: HistGradientBoostingClassifier(
            early_stopping=False, random_state=0),
    }


TASK_SIZES = (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600)


def _reps(n: int) -> int:
    """More replicates where a training set is small and its error noisy."""
    return int(min(20, max(3, 8000 // n)))


def task_curve(task: GaussianTask | None = None, *, sizes=TASK_SIZES
               ) -> dict:
    """Test error of each model at each training size, averaged over
    independent training sets, on the task's fixed test set."""
    np = _np()
    task = task or GaussianTask()
    err = {k: {} for k in task_models()}
    for n in sizes:
        for k, make in task_models().items():
            e = []
            for r in range(_reps(n)):
                X, y = task.draw(n, seed=1000 + r)
                e.append(float(np.mean(make().fit(X, y).predict(task.X_test)
                                       != task.y_test)))
            err[k][int(n)] = float(np.mean(e))
    return {"error": err, "sizes": [int(n) for n in sizes],
            "bayes": task.bayes, "test": int(len(task.y_test))}


DIGIT_SIZES = (25, 50, 100, 150, 200, 300, 450, 600, 900, 1200)


def digits_curve(*, sizes=DIGIT_SIZES, test: int = 597, reps: int = 20,
                 seed: int = 0) -> dict:
    """The same on the UCI digits: a fixed test set, training sets drawn
    from the remaining 1,200 images. At the full 1,200 there is only one
    training set, so that point has no replicates."""
    np = _np()
    from standarderror.ml import evaluation as ev
    X, y = ev.digits()
    perm = np.random.default_rng(int(seed)).permutation(len(y))
    te, pool = perm[:int(test)], perm[int(test):]
    err = {k: {} for k in ev.models()}
    for n in sizes:
        draws = ([pool] if n >= len(pool) else
                 [np.random.default_rng(r).choice(pool, int(n), replace=False)
                  for r in range(int(reps))])
        for k, make in ev.models().items():
            e = [float(np.mean(make().fit(X[tr], y[tr]).predict(X[te])
                               != y[te])) for tr in draws]
            err[k][int(n)] = float(np.mean(e))
    return {"error": err, "sizes": [int(n) for n in sizes],
            "pool": int(len(pool)), "test": int(test)}


# ------------------------------------------------------------ the fits

def power(n, a, b):
    return a * _np().asarray(n, float) ** (-b)


def power_floor(n, a, b, c):
    return a * _np().asarray(n, float) ** (-b) + c


def fit(curve: dict, model: str, *, until: int, floor: bool) -> dict:
    """Fit `a n^-b` (or `a n^-b + c`) to the errors at sizes up to `until`,
    by ordinary least squares on the error scale -- the way a pilot study
    usually does it."""
    np = _np()
    from scipy.optimize import curve_fit
    ns = np.array([n for n in curve["sizes"] if n <= until], float)
    es = np.array([curve["error"][model][int(n)] for n in ns])
    if floor:
        p, _ = curve_fit(power_floor, ns, es, p0=(1.0, 0.5, 0.1),
                         bounds=([0, 0, 0], [100, 3, 1]), maxfev=20000)
        return {"a": float(p[0]), "b": float(p[1]), "c": float(p[2]),
                "predict": lambda n, p=p: float(power_floor(n, *p))}
    p, _ = curve_fit(power, ns, es, p0=(1.0, 0.3), maxfev=20000)
    return {"a": float(p[0]), "b": float(p[1]), "c": 0.0,
            "predict": lambda n, p=p: float(power(n, *p))}


def extrapolate(curve: dict, *, until, target: int) -> list[dict]:
    """For every model and fit range, both fits' predictions at `target`
    against the error actually measured there."""
    out = []
    for model in curve["error"]:
        for u in until:
            two, three = (fit(curve, model, until=u, floor=f)
                          for f in (False, True))
            actual = curve["error"][model][int(target)]
            out.append({"model": model, "until": int(u),
                        "target": int(target), "actual": actual,
                        "power": two["predict"](target),
                        "power_floor": three["predict"](target),
                        "floor": three["c"], "exponent": two["b"]})
    return out


def best_at(curve: dict) -> dict:
    """Which model has the lowest error at each training size."""
    return {n: min(curve["error"], key=lambda k: curve["error"][k][n])
            for n in curve["sizes"]}


def relative_error(predicted: float, actual: float) -> float:
    return (predicted - actual) / actual


def log_slope(curve: dict, model: str) -> list[float]:
    """Local slope of log error against log n between neighbouring sizes:
    constant if the curve is a power law without a floor."""
    s, e = curve["sizes"], curve["error"][model]
    return [(math.log(e[b]) - math.log(e[a])) / (math.log(b) - math.log(a))
            for a, b in zip(s, s[1:])]
