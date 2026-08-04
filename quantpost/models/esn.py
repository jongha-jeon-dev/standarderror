"""Echo state network — the workhorse.

Design decisions worth defending, because most published ESN code gets at least
one of them wrong:

* **Spectral radius is measured exactly, not approximated.** Dense `eigvals` up to
  N = 1000, ARPACK at machine tolerance above that. Rescaling by a row-sum bound
  (the common shortcut) gives a reservoir whose radius is nowhere near what you
  reported, and even ARPACK with a loose tolerance was off by 0.15% at N = 250.
  `train_diagnostics["actual_spectral_radius"]` re-measures after rescaling so the
  number in your post is the number in your matrix.

* **The washout is discarded from the design matrix, not merely from the loss.**
  Including transient states biases the readout toward the initial condition.

* **Ridge readout solved by `lstsq` on the normal equations with explicit
  regularisation**, computed in float64 with the bias column excluded from the
  penalty. Penalising the intercept shrinks the mean of your forecast toward
  zero, which on a non-centred series is a silent, systematic error.

* **`predict_autonomous` feeds the model its own output.** This is the only mode
  that tests whether dynamics were learned. One-step-ahead teacher-forced error
  is nearly meaningless on a smooth series: persistence already achieves it.

* **Leaky integration** with rate `a`: `h_{t+1} = (1-a) h_t + a tanh(W h_t +
  W_in u_t + b)`. `a=1` recovers the classic ESN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class ESNConfig:
    n_reservoir: int = 500
    spectral_radius: float = 0.9
    sparsity: float = 0.02            # fraction of non-zero entries in W
    input_scaling: float = 1.0
    bias_scaling: float = 0.1
    leak_rate: float = 1.0
    ridge: float = 1e-6
    washout: int = 200
    seed: int = 0
    # Adding the raw input to the readout features costs nothing and reliably
    # helps: it lets the readout represent the identity map without spending
    # reservoir capacity on it.
    include_input_in_readout: bool = True
    # Lu et al. (2017): squaring half the states breaks the odd symmetry of tanh,
    # which matters for systems (like Lorenz) with a symmetry the readout would
    # otherwise be unable to distinguish.
    quadratic_features: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ESN:
    config: ESNConfig = field(default_factory=ESNConfig)
    W: sp.csr_matrix | None = None
    W_in: np.ndarray | None = None
    b: np.ndarray | None = None
    W_out: np.ndarray | None = None
    n_in: int | None = None
    n_out: int | None = None
    _state: np.ndarray | None = None
    train_diagnostics: dict = field(default_factory=dict)

    # ---------- construction ----------

    def _build(self, n_in: int) -> None:
        c = self.config
        rng = np.random.default_rng(c.seed)
        N = c.n_reservoir

        density = max(c.sparsity, 1.0 / N)
        W = sp.random(N, N, density=density, format="csr",
                      random_state=np.random.RandomState(c.seed),
                      data_rvs=lambda k: rng.uniform(-1.0, 1.0, k))
        radius = _spectral_radius(W)
        if radius < 1e-12:
            raise RuntimeError(
                "reservoir matrix is (numerically) nilpotent; raise sparsity")
        self.W = (W * (c.spectral_radius / radius)).tocsr()
        self.W_in = rng.uniform(-1.0, 1.0, (N, n_in)) * c.input_scaling
        self.b = rng.uniform(-1.0, 1.0, N) * c.bias_scaling
        self.n_in = n_in
        self._state = np.zeros(N)
        self.train_diagnostics["actual_spectral_radius"] = float(
            _spectral_radius(self.W))

    # ---------- state machinery ----------

    def _step(self, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        a = self.config.leak_rate
        pre = self.W @ h + self.W_in @ u + self.b
        return (1.0 - a) * h + a * np.tanh(pre)

    def _features(self, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        c = self.config
        parts = [np.ones(1), h]
        if c.quadratic_features:
            g = h.copy()
            g[::2] = g[::2] ** 2
            parts.append(g)
        if c.include_input_in_readout:
            parts.append(u)
        return np.concatenate(parts)

    def harvest(self, U: np.ndarray, *, reset: bool = True) -> np.ndarray:
        """Run the reservoir over inputs `U` (T, n_in) -> features (T, n_feat)."""
        U = np.atleast_2d(np.asarray(U, float))
        if U.ndim == 1:
            U = U[:, None]
        if self.W is None:
            self._build(U.shape[1])
        if U.shape[1] != self.n_in:
            raise ValueError(f"expected {self.n_in} inputs, got {U.shape[1]}")
        h = np.zeros(self.config.n_reservoir) if reset else self._state.copy()
        feats = np.empty((len(U), len(self._features(h, U[0]))))
        for t, u in enumerate(U):
            h = self._step(h, u)
            feats[t] = self._features(h, u)
        self._state = h
        return feats

    # ---------- training ----------

    def fit(self, U: np.ndarray, Y: np.ndarray) -> ESN:
        """Teacher-forced ridge fit. `U[t]` -> `Y[t]`.

        For one-step-ahead forecasting call `fit(x[:-1], x[1:])`; for the
        autonomous-rollout convention used by `predict_autonomous`, `Y` must be
        the *next* state so the model is a map, not a filter.
        """
        U = _as2d(U)
        Y = _as2d(Y)
        if len(U) != len(Y):
            raise ValueError(f"U has {len(U)} rows, Y has {len(Y)}")
        w = self.config.washout
        if len(U) <= w + 10:
            raise ValueError(
                f"only {len(U)} samples for washout={w}; need materially more")

        X = self.harvest(U)[w:]
        T = Y[w:]

        # Ridge with the intercept column left unpenalised.
        n_feat = X.shape[1]
        pen = np.full(n_feat, self.config.ridge)
        pen[0] = 0.0
        A = X.T @ X + np.diag(pen)
        B = X.T @ T
        self.W_out = np.linalg.solve(A, B) if _well_conditioned(A) else \
            np.linalg.lstsq(A, B, rcond=None)[0]
        self.n_out = T.shape[1]

        resid = T - X @ self.W_out
        self.train_diagnostics.update({
            "n_train": int(len(T)),
            "n_features": int(n_feat),
            "train_rmse": float(np.sqrt(np.mean(resid ** 2))),
            "readout_norm": float(np.linalg.norm(self.W_out)),
            "design_condition_number": float(np.linalg.cond(A)),
        })
        return self

    # ---------- prediction ----------

    def predict_teacher_forced(self, U: np.ndarray, *,
                               reset: bool = True) -> np.ndarray:
        """One-step-ahead with the true input at every step. Optimistic; report
        it only alongside an autonomous number."""
        self._require_fit()
        X = self.harvest(_as2d(U), reset=reset)
        return X @ self.W_out

    def predict_autonomous(
        self,
        warmup: np.ndarray,
        n_steps: int,
        *,
        readout_to_input=None,
    ) -> np.ndarray:
        """Close the loop: warm the state on `warmup`, then feed predictions back.

        `readout_to_input` maps the output back to the next input when they are
        not the same object (e.g. output is a delta, input is a level). Defaults
        to the identity, which requires `n_out == n_in`.
        """
        self._require_fit()
        warmup = _as2d(warmup)
        if readout_to_input is None:
            if self.n_out != self.n_in:
                raise ValueError(
                    f"n_out={self.n_out} != n_in={self.n_in}; supply "
                    "readout_to_input to close the loop")
            readout_to_input = lambda y, _u: y  # noqa: E731

        h = np.zeros(self.config.n_reservoir)
        for u in warmup:
            h = self._step(h, u)
        u = warmup[-1]
        out = np.empty((n_steps, self.n_out))
        for i in range(n_steps):
            y = self._features(h, u) @ self.W_out
            out[i] = y
            u = np.asarray(readout_to_input(y, u), float).ravel()
            h = self._step(h, u)
            if not np.isfinite(h).all():
                out[i + 1:] = np.nan
                break
        return out

    def _require_fit(self) -> None:
        if self.W_out is None:
            raise RuntimeError("call fit() first")


# ---------- helpers ----------

def _as2d(a) -> np.ndarray:
    arr = np.asarray(a, float)
    return arr[:, None] if arr.ndim == 1 else arr


#: Below this size the radius is computed densely and exactly. ARPACK with a
#: loose tolerance returns a *near*-largest eigenvalue on sparse random
#: matrices — it was off by 0.15% at N=250, which means the reservoir you
#: report is not the reservoir you built. Dense `eigvals` costs ~0.1 s at
#: N=600 and is worth it.
DENSE_EIG_MAX_N = 1000


def _spectral_radius(W) -> float:
    N = W.shape[0]
    if N <= DENSE_EIG_MAX_N:
        dense = W.toarray() if sp.issparse(W) else np.asarray(W)
        return float(np.max(np.abs(np.linalg.eigvals(dense))))
    try:
        vals = spla.eigs(W.astype(float), k=1, which="LM",
                         return_eigenvectors=False, maxiter=100000, tol=0,
                         ncv=min(N, 64))
        return float(np.abs(vals[0]))
    except Exception:
        dense = W.toarray() if sp.issparse(W) else np.asarray(W)
        return float(np.max(np.abs(np.linalg.eigvals(dense))))


def _well_conditioned(A: np.ndarray, limit: float = 1e12) -> bool:
    try:
        return np.linalg.cond(A) < limit
    except np.linalg.LinAlgError:
        return False
