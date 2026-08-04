"""Baselines you are obliged to beat.

Every forecasting post that skips these is unfalsifiable. The ordering here is
deliberate — `Persistence` first, because on daily financial levels it is
brutally strong and most "our model achieves R2 = 0.98" claims are measuring
exactly this and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as2d(a) -> np.ndarray:
    arr = np.asarray(a, float)
    return arr[:, None] if arr.ndim == 1 else arr


@dataclass
class Persistence:
    """y_hat[t] = u[t]. The bar. Report it or your numbers mean nothing."""
    name: str = "persistence"

    def fit(self, U, Y):
        return self

    def predict_teacher_forced(self, U) -> np.ndarray:
        return _as2d(U).copy()

    def predict_autonomous(self, warmup, n_steps) -> np.ndarray:
        last = _as2d(warmup)[-1]
        return np.repeat(last[None, :], n_steps, axis=0)


@dataclass
class LinearAR:
    """Ridge-regularised linear VAR on `n_lags` delays. The linear control:
    if a nonlinear model does not beat this, the nonlinearity is decoration."""
    n_lags: int = 4
    ridge: float = 1e-8
    name: str = "linear-AR"
    W: np.ndarray | None = None
    n_in: int | None = None

    @property
    def span(self) -> int:
        return self.n_lags

    def _design(self, U: np.ndarray) -> np.ndarray:
        T = len(U)
        rows = T - self.n_lags + 1
        cols = [np.ones((rows, 1))]
        for j in range(self.n_lags):
            off = self.n_lags - 1 - j
            cols.append(U[self.n_lags - 1 - off: T - off])
        return np.concatenate(cols, axis=1)

    def fit(self, U, Y):
        U, Y = _as2d(U), _as2d(Y)
        X = self._design(U)
        T = Y[self.n_lags - 1:]
        pen = np.full(X.shape[1], self.ridge)
        pen[0] = 0.0
        self.W = np.linalg.lstsq(X.T @ X + np.diag(pen), X.T @ T, rcond=None)[0]
        self.n_in = U.shape[1]
        return self

    def predict_teacher_forced(self, U) -> np.ndarray:
        return self._design(_as2d(U)) @ self.W

    def predict_autonomous(self, warmup, n_steps) -> np.ndarray:
        warmup = _as2d(warmup)
        hist = list(warmup[-self.n_lags:])
        out = np.empty((n_steps, self.W.shape[1]))
        for i in range(n_steps):
            y = (self._design(np.array(hist))[-1:] @ self.W).ravel()
            out[i] = y
            hist = hist[1:] + [y]
        return out


@dataclass
class RandomFeatures:
    """Random-feature (ELM) map on lagged inputs: a static nonlinearity with no
    memory of its own. Isolates how much of an ESN's edge comes from
    *recurrence* rather than merely from being nonlinear and wide."""
    n_features: int = 500
    n_lags: int = 2
    scale: float = 1.0
    ridge: float = 1e-6
    seed: int = 0
    name: str = "random-features"
    _A: np.ndarray | None = None
    _b: np.ndarray | None = None
    W_out: np.ndarray | None = None

    @property
    def span(self) -> int:
        return self.n_lags

    def _lagged(self, U: np.ndarray) -> np.ndarray:
        T = len(U)
        cols = []
        for j in range(self.n_lags):
            off = self.n_lags - 1 - j
            cols.append(U[self.n_lags - 1 - off: T - off])
        return np.concatenate(cols, axis=1)

    def _phi(self, Z: np.ndarray) -> np.ndarray:
        if self._A is None:
            rng = np.random.default_rng(self.seed)
            self._A = rng.standard_normal((Z.shape[1], self.n_features)) * self.scale
            self._b = rng.uniform(-np.pi, np.pi, self.n_features)
        return np.concatenate([np.ones((len(Z), 1)), np.tanh(Z @ self._A + self._b)],
                              axis=1)

    def fit(self, U, Y):
        U, Y = _as2d(U), _as2d(Y)
        X = self._phi(self._lagged(U))
        T = Y[self.n_lags - 1:]
        pen = np.full(X.shape[1], self.ridge)
        pen[0] = 0.0
        self.W_out = np.linalg.lstsq(X.T @ X + np.diag(pen), X.T @ T,
                                    rcond=None)[0]
        return self

    def predict_teacher_forced(self, U) -> np.ndarray:
        return self._phi(self._lagged(_as2d(U))) @ self.W_out

    def predict_autonomous(self, warmup, n_steps) -> np.ndarray:
        warmup = _as2d(warmup)
        hist = list(warmup[-self.n_lags:])
        out = np.empty((n_steps, self.W_out.shape[1]))
        for i in range(n_steps):
            y = (self._phi(self._lagged(np.array(hist)))[-1:] @ self.W_out).ravel()
            out[i] = y
            hist = hist[1:] + [y]
        return out


@dataclass
class GradientBoosting:
    """sklearn HistGradientBoosting on lagged features, one model per output.
    The "what a sensible practitioner would actually reach for" control."""
    n_lags: int = 8
    seed: int = 0
    name: str = "hist-gbm"
    params: dict = field(default_factory=lambda: {
        "max_iter": 300, "learning_rate": 0.06, "max_depth": None,
        "min_samples_leaf": 20, "l2_regularization": 1.0})
    _models: list = field(default_factory=list)

    @property
    def span(self) -> int:
        return self.n_lags

    def _lagged(self, U: np.ndarray) -> np.ndarray:
        T = len(U)
        cols = []
        for j in range(self.n_lags):
            off = self.n_lags - 1 - j
            cols.append(U[self.n_lags - 1 - off: T - off])
        return np.concatenate(cols, axis=1)

    def fit(self, U, Y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        U, Y = _as2d(U), _as2d(Y)
        X = self._lagged(U)
        T = Y[self.n_lags - 1:]
        self._models = []
        for j in range(T.shape[1]):
            m = HistGradientBoostingRegressor(random_state=self.seed,
                                              **self.params)
            m.fit(X, T[:, j])
            self._models.append(m)
        return self

    def predict_teacher_forced(self, U) -> np.ndarray:
        X = self._lagged(_as2d(U))
        return np.column_stack([m.predict(X) for m in self._models])

    def predict_autonomous(self, warmup, n_steps) -> np.ndarray:
        warmup = _as2d(warmup)
        hist = list(warmup[-self.n_lags:])
        out = np.empty((n_steps, len(self._models)))
        for i in range(n_steps):
            X = self._lagged(np.array(hist))[-1:]
            y = np.array([m.predict(X)[0] for m in self._models])
            out[i] = y
            hist = hist[1:] + [y]
        return out


ALL = {
    "persistence": Persistence,
    "linear_ar": LinearAR,
    "random_features": RandomFeatures,
    "gbm": GradientBoosting,
}
