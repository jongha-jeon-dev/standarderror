"""Structural causal models with a known answer.

The point, same as `dynamics`: build the case where you know the truth, then watch
methods succeed or fail against it. For attribution and feature importance the
truth you need is a *causal* one, and no real dataset supplies it.

`ConfoundedSCM` generates data from an explicit graph whose ATE you can compute in
closed form:

    Z ~ N(0,1)                       confounder, observed or not
    U ~ N(0,1)                       instrument-ish exogenous driver of X
    X = a_zx*Z + a_ux*U + noise      treatment / feature of interest
    M = a_xm*X + noise               mediator
    Y = b_x*X + b_m*M + b_z*Z + noise

so the **total** effect of X on Y is `b_x + b_m*a_xm`, the **direct** effect is
`b_x`, and a naive regression of Y on X alone is biased by the confounder path
`a_zx*b_z / var(X)`-worth of association. Every one of those is available as a
property, so a post can put "what the method said" next to "what is true" in a
table instead of gesturing.

The three canonical mistakes this makes demonstrable:

* **Confounding** — omit Z and the coefficient on X absorbs `a_zx*b_z`.
* **Mediator adjustment** — control for M and you recover the *direct* effect
  while reporting it as the total. This one is a silent halving.
* **Collider adjustment** — condition on a common effect of X and Y and you
  manufacture association from nothing. `collider` generates it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SCMData:
    frame: dict[str, np.ndarray]
    truth: dict[str, float]
    graph: str

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.frame)

    def describe(self) -> str:
        t = ", ".join(f"{k}={v:.4f}" for k, v in self.truth.items())
        return f"{self.graph}\n  ground truth: {t}"


@dataclass
class ConfoundedSCM:
    a_zx: float = 1.0       # confounder -> treatment
    a_ux: float = 1.0       # exogenous  -> treatment
    a_xm: float = 0.8       # treatment  -> mediator
    b_x: float = 0.5        # direct effect of treatment on outcome
    b_m: float = 1.0        # mediator   -> outcome
    b_z: float = 1.5        # confounder -> outcome
    noise_x: float = 0.5
    noise_m: float = 0.5
    noise_y: float = 0.5

    @property
    def total_effect(self) -> float:
        return self.b_x + self.b_m * self.a_xm

    @property
    def direct_effect(self) -> float:
        return self.b_x

    @property
    def naive_bias(self) -> float:
        """Bias of regressing Y on X alone, omitting Z.

        OLS estimates total_effect + b_z * cov(Z,X)/var(X), and with independent
        Z and U that reduces to b_z * a_zx / (a_zx^2 + a_ux^2 + noise_x^2).
        """
        var_x = self.a_zx ** 2 + self.a_ux ** 2 + self.noise_x ** 2
        return self.b_z * self.a_zx / var_x

    def sample(self, n: int = 5000, *, seed: int = 0) -> SCMData:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n)
        u = rng.standard_normal(n)
        x = self.a_zx * z + self.a_ux * u + self.noise_x * rng.standard_normal(n)
        m = self.a_xm * x + self.noise_m * rng.standard_normal(n)
        y = (self.b_x * x + self.b_m * m + self.b_z * z
             + self.noise_y * rng.standard_normal(n))
        return SCMData(
            {"X": x, "M": m, "Z": z, "U": u, "Y": y},
            {"total_effect": self.total_effect,
             "direct_effect": self.direct_effect,
             "naive_ols_on_X_alone": self.total_effect + self.naive_bias,
             "confounding_bias": self.naive_bias},
            graph=("Z->X, U->X, X->M, X->Y, M->Y, Z->Y "
                   "(Z observed but omittable; U exogenous)"))


def collider(n: int = 5000, *, b_xy: float = 0.0, a_xc: float = 1.0,
             a_yc: float = 1.0, seed: int = 0) -> SCMData:
    """X -> C <- Y with no X->Y edge by default.

    Regress Y on X and you get nothing, correctly. Add C as a control and a
    spurious coefficient appears out of thin air. This is the cleanest possible
    demonstration that "add more controls" is not a safety measure.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = b_xy * x + rng.standard_normal(n)
    c = a_xc * x + a_yc * y + 0.5 * rng.standard_normal(n)
    return SCMData({"X": x, "Y": y, "C": c},
                   {"true_effect_of_X_on_Y": b_xy,
                    "expected_naive_estimate": b_xy,
                    "expected_sign_after_conditioning_on_C": -1.0},
                   graph="X->C, Y->C (collider); X->Y only if b_xy != 0")


def ols(frame: dict[str, np.ndarray], outcome: str,
        regressors: list[str]) -> dict[str, float]:
    """Least squares with an intercept — the estimator being tested.

    Deliberately plain: the point of these experiments is that the *estimator* is
    fine and the *identification* is what fails, so using anything fancier would
    obscure the lesson.
    """
    y = np.asarray(frame[outcome], float)
    X = np.column_stack([np.ones(len(y))]
                        + [np.asarray(frame[r], float) for r in regressors])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(zip(["intercept"] + list(regressors), beta.tolist()))
