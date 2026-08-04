"""Next-generation reservoir computing (NG-RC / NVAR).

Gauthier, Bollt, Griffith & Barbosa (2021) showed that for many tasks the random
reservoir can be replaced by an explicit nonlinear vector autoregression on a
short delay window — same or better accuracy with orders of magnitude less data
and no random matrices to tune.

This matters for a *finance* audience specifically: an ESN is a black box with
2,000 hidden states, while NG-RC's readout weights sit on named monomials of
lagged inputs. You can print the model. When a risk model has to be explained to
a validation function, that difference is the whole ballgame — so NG-RC belongs
in the same repo as the XAI module, not in a separate one.

Feature map: constant, the `k` most recent lags of each input, and all monomials
of those lags up to `degree` (2 or 3 in practice). Feature count grows fast —
`n_features` is exposed so you can check it before fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np


@dataclass
class NGRCConfig:
    n_lags: int = 2               # k: how many delays enter the feature map
    stride: int = 1               # s: spacing between delays
    degree: int = 2               # highest monomial degree
    ridge: float = 1e-8
    include_constant: bool = True
    standardise: bool = True      # z-score linear features before monomials

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class NGRC:
    config: NGRCConfig = field(default_factory=NGRCConfig)
    W_out: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    n_in: int | None = None
    n_out: int | None = None
    _mu: np.ndarray | None = None
    _sd: np.ndarray | None = None
    train_diagnostics: dict = field(default_factory=dict)

    # ---------- feature construction ----------

    @property
    def span(self) -> int:
        """Number of past samples the feature map needs."""
        return (self.config.n_lags - 1) * self.config.stride + 1

    def _linear_block(self, U: np.ndarray) -> tuple[np.ndarray, list[str]]:
        c = self.config
        T, d = U.shape
        span = self.span
        rows = T - span + 1
        if rows <= 0:
            raise ValueError(
                f"need at least {span} samples for n_lags={c.n_lags}, "
                f"stride={c.stride}; got {T}")
        cols, names = [], []
        for j in range(c.n_lags):
            offset = (c.n_lags - 1 - j) * c.stride
            cols.append(U[span - 1 - offset: T - offset])
            for v in range(d):
                names.append(f"x{v}[t-{offset}]")
        return np.concatenate(cols, axis=1), names

    def _expand(self, lin: np.ndarray,
                names: list[str]) -> tuple[np.ndarray, list[str]]:
        c = self.config
        blocks, out_names = [], []
        if c.include_constant:
            blocks.append(np.ones((len(lin), 1)))
            out_names.append("1")
        blocks.append(lin)
        out_names.extend(names)
        p = lin.shape[1]
        for deg in range(2, c.degree + 1):
            for combo in combinations_with_replacement(range(p), deg):
                col = np.prod(lin[:, combo], axis=1, keepdims=True)
                blocks.append(col)
                out_names.append("*".join(names[i] for i in combo))
        return np.concatenate(blocks, axis=1), out_names

    def features(self, U: np.ndarray) -> np.ndarray:
        U = _as2d(U)
        lin, names = self._linear_block(U)
        if self.config.standardise:
            if self._mu is None:
                self._mu = lin.mean(axis=0)
                self._sd = lin.std(axis=0)
                self._sd[self._sd < 1e-12] = 1.0
            lin = (lin - self._mu) / self._sd
        X, all_names = self._expand(lin, names)
        self.feature_names = all_names
        return X

    def n_features(self, n_in: int) -> int:
        c = self.config
        p = c.n_lags * n_in
        total = 1 if c.include_constant else 0
        from math import comb
        for deg in range(1, c.degree + 1):
            total += comb(p + deg - 1, deg)
        return total

    # ---------- fit / predict ----------

    def fit(self, U: np.ndarray, Y: np.ndarray) -> NGRC:
        U, Y = _as2d(U), _as2d(Y)
        if len(U) != len(Y):
            raise ValueError("U and Y must have the same number of rows")
        X = self.features(U)
        T = Y[self.span - 1:]
        pen = np.full(X.shape[1], self.config.ridge)
        if self.config.include_constant:
            pen[0] = 0.0
        A = X.T @ X + np.diag(pen)
        self.W_out = np.linalg.lstsq(A, X.T @ T, rcond=None)[0]
        self.n_in, self.n_out = U.shape[1], T.shape[1]
        resid = T - X @ self.W_out
        self.train_diagnostics.update({
            "n_train": int(len(T)),
            "n_features": int(X.shape[1]),
            "train_rmse": float(np.sqrt(np.mean(resid ** 2))),
            "design_condition_number": float(np.linalg.cond(A)),
        })
        return self

    def predict_teacher_forced(self, U: np.ndarray) -> np.ndarray:
        self._require_fit()
        return self.features(_as2d(U)) @ self.W_out

    def predict_autonomous(self, warmup: np.ndarray, n_steps: int) -> np.ndarray:
        """Closed-loop rollout. `warmup` must be at least `span` rows."""
        self._require_fit()
        warmup = _as2d(warmup)
        if len(warmup) < self.span:
            raise ValueError(f"warmup needs >= {self.span} rows")
        if self.n_out != self.n_in:
            raise ValueError("autonomous rollout requires n_out == n_in")
        hist = list(warmup[-self.span:])
        out = np.empty((n_steps, self.n_out))
        for i in range(n_steps):
            X = self.features(np.array(hist))
            y = (X[-1:] @ self.W_out).ravel()
            out[i] = y
            hist = hist[1:] + [y]
            if not np.isfinite(y).all():
                out[i + 1:] = np.nan
                break
        return out

    def top_terms(self, output: int = 0, k: int = 12) -> list[tuple[str, float]]:
        """The `k` largest readout coefficients, by absolute value.

        This is the payoff: an interpretable forecast model whose terms you can
        read off and compare against the equations you believe govern the system.
        Coefficients are on standardised linear features when
        `standardise=True`, so magnitudes are comparable across terms.
        """
        self._require_fit()
        w = self.W_out[:, output]
        order = np.argsort(np.abs(w))[::-1][:k]
        return [(self.feature_names[i], float(w[i])) for i in order]

    def _require_fit(self) -> None:
        if self.W_out is None:
            raise RuntimeError("call fit() first")


def _as2d(a) -> np.ndarray:
    arr = np.asarray(a, float)
    return arr[:, None] if arr.ndim == 1 else arr
