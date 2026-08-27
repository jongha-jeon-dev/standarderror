"""Conformal prediction: intervals with a coverage guarantee, and its limits.

Why this module exists in a repo about time series: conformal prediction is the
rare tool that hands you a *distribution-free finite-sample* guarantee, and it is
also routinely misapplied to exactly the data here. Both halves are worth writing
about, so both are implemented.

**What split conformal actually promises.** With exchangeable data and a
calibration set of size n, the interval covers with probability at least
1 - alpha, using the ceil((n+1)(1-alpha))-th smallest calibration residual. The
`(n+1)` is not a rounding detail — dropping it undercovers, visibly so for small
n. There is no distributional assumption and no asymptotics.

**What it does not promise.** Two things, and skipping either is how conformal
gets oversold:

1. **Marginal, not conditional.** 90% coverage overall is compatible with 99%
   coverage on easy inputs and 40% on hard ones. `coverage_by_group` and
   `coverage_by_bin` exist so you check rather than assume. `cqr` narrows the gap
   by making the interval width input-dependent.
2. **Exchangeability, which time series violate.** Financial and macro series are
   neither i.i.d. nor exchangeable, so the guarantee simply does not hold. The
   honest responses are implemented here: `WeightedConformal` for covariate shift
   with known/estimated likelihood ratios (Tibshirani et al. 2019), and
   `AdaptiveConformal` — online adaptive conformal inference (Gibbs & Candès
   2021), which abandons the finite-sample guarantee in exchange for *long-run*
   coverage that holds under arbitrary distribution shift.

Calibration data must be disjoint from training data. `split_conformal` cannot
check that for you, so the docstrings say it and the tests enforce it on the
wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))/n empirical quantile of `scores`.

    This exact index — not `np.quantile(scores, 1-alpha)` — is what gives the
    finite-sample guarantee. Returns +inf when n is too small for the requested
    level, which is the correct answer: with 5 calibration points you cannot
    honestly claim 99% coverage.
    """
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        raise ValueError("no finite calibration scores")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(s)[k - 1])


@dataclass
class Interval:
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    method: str
    detail: dict = field(default_factory=dict)

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def covers(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, float).ravel()
        return (y >= self.lower) & (y <= self.upper)

    def summary(self, y: np.ndarray) -> dict:
        c = self.covers(y)
        return {"target_coverage": 1.0 - self.alpha,
                "empirical_coverage": float(np.mean(c)),
                "mean_width": float(np.mean(self.width)),
                "median_width": float(np.median(self.width)),
                "n": int(len(c)), "method": self.method}


def split_conformal(
    calib_pred: np.ndarray,
    calib_true: np.ndarray,
    test_pred: np.ndarray,
    *,
    alpha: float = 0.1,
) -> Interval:
    """Constant-width intervals from absolute residuals on a calibration set.

    `calib_pred` must come from a model that never saw `calib_true` in training.
    The width is the same for every test point, which is the method's honest
    weakness: it cannot say "this input is harder". Use `cqr` when you need that.
    """
    scores = np.abs(np.asarray(calib_true, float).ravel()
                    - np.asarray(calib_pred, float).ravel())
    q = conformal_quantile(scores, alpha)
    p = np.asarray(test_pred, float).ravel()
    return Interval(p - q, p + q, alpha, "split conformal (absolute residual)",
                    {"q": q, "n_calib": int(len(scores))})


def normalised_conformal(
    calib_pred: np.ndarray,
    calib_true: np.ndarray,
    calib_scale: np.ndarray,
    test_pred: np.ndarray,
    test_scale: np.ndarray,
    *,
    alpha: float = 0.1,
) -> Interval:
    """Locally adaptive variant: score is |y - yhat| / scale(x).

    `scale` is any positive difficulty estimate — a second model fitted to the
    absolute residuals, a rolling volatility, a predicted variance. Width becomes
    input-dependent while the marginal guarantee survives, because the score is
    still a fixed function of (x, y).
    """
    cs = np.asarray(calib_scale, float).ravel()
    ts = np.asarray(test_scale, float).ravel()
    if np.any(cs <= 0) or np.any(ts <= 0):
        raise ValueError("scale must be strictly positive")
    scores = np.abs(np.asarray(calib_true, float).ravel()
                    - np.asarray(calib_pred, float).ravel()) / cs
    q = conformal_quantile(scores, alpha)
    p = np.asarray(test_pred, float).ravel()
    return Interval(p - q * ts, p + q * ts, alpha,
                    "normalised split conformal", {"q": q,
                                                   "n_calib": int(len(scores))})


def cqr(
    calib_lo: np.ndarray,
    calib_hi: np.ndarray,
    calib_true: np.ndarray,
    test_lo: np.ndarray,
    test_hi: np.ndarray,
    *,
    alpha: float = 0.1,
) -> Interval:
    """Conformalized quantile regression (Romano, Patterson & Candès 2019).

    Take any quantile regressor's alpha/2 and 1-alpha/2 predictions and calibrate
    them. The score `max(lo - y, y - hi)` is signed, so a *too wide* quantile
    model gets tightened (negative q) rather than only ever widened — which is why
    CQR usually beats plain split conformal on width at equal coverage.
    """
    lo = np.asarray(calib_lo, float).ravel()
    hi = np.asarray(calib_hi, float).ravel()
    y = np.asarray(calib_true, float).ravel()
    scores = np.maximum(lo - y, y - hi)
    q = conformal_quantile(scores, alpha)
    tl = np.asarray(test_lo, float).ravel()
    th = np.asarray(test_hi, float).ravel()
    return Interval(tl - q, th + q, alpha, "CQR",
                    {"q": q, "n_calib": int(len(scores)),
                     "tightened": bool(q < 0)})


@dataclass
class WeightedConformal:
    """Covariate shift (Tibshirani, Barber, Candès & Ramdas 2019).

    Reweight calibration points by the likelihood ratio w(x) = dP_test/dP_train
    and take the *weighted* quantile. Exchangeability is replaced by the weaker
    assumption that only the covariate distribution moved — `Y | X` must be
    unchanged. If the conditional law also moved, no amount of reweighting saves
    you, and this class cannot detect that.

    The weights are usually estimated by a classifier separating train from test
    covariates: w(x) = p(x)/(1-p(x)) for its predicted probability p.
    """
    alpha: float = 0.1

    def interval(self, calib_pred, calib_true, calib_weights,
                 test_pred, test_weight: float | np.ndarray = 1.0) -> Interval:
        s = np.abs(np.asarray(calib_true, float).ravel()
                   - np.asarray(calib_pred, float).ravel())
        w = np.asarray(calib_weights, float).ravel()
        if len(w) != len(s):
            raise ValueError("one weight per calibration point")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        p = np.asarray(test_pred, float).ravel()
        tw = np.full(len(p), float(test_weight)) if np.isscalar(test_weight) \
            else np.asarray(test_weight, float).ravel()

        order = np.argsort(s)
        s_sorted, w_sorted = s[order], w[order]
        lo = np.empty(len(p))
        hi = np.empty(len(p))
        qs = np.empty(len(p))
        for i in range(len(p)):
            total = w_sorted.sum() + tw[i]
            if total <= 0:
                qs[i] = np.inf
            else:
                cum = np.cumsum(w_sorted) / total
                idx = np.searchsorted(cum, 1.0 - self.alpha)
                # The test point's own mass sits at +inf: if it is large enough to
                # push the cumulative weight past 1-alpha, the honest answer is an
                # unbounded interval, not the largest observed residual.
                qs[i] = np.inf if idx >= len(s_sorted) else s_sorted[idx]
            lo[i], hi[i] = p[i] - qs[i], p[i] + qs[i]
        return Interval(lo, hi, self.alpha, "weighted split conformal",
                        {"n_calib": int(len(s)),
                         "n_infinite": int(np.sum(~np.isfinite(qs))),
                         "effective_sample_size":
                             float(w.sum() ** 2 / np.sum(w ** 2))
                             if np.any(w > 0) else 0.0})


@dataclass
class AdaptiveConformal:
    """Online adaptive conformal inference (Gibbs & Candès 2021).

    For time series, where exchangeability fails outright. Maintain a running
    level `alpha_t`, and after each observation nudge it by
    `alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`, where `err_t` is 1 if the
    last interval missed. Miss too often and the interval widens; never miss and
    it tightens.

    The trade is explicit and worth stating in any post that uses this: you give
    up the finite-sample guarantee and get **long-run** coverage converging to
    1 - alpha under arbitrary, even adversarial, distribution shift. `gamma`
    controls adaptation speed; 0.005-0.05 is the usual range, and the realised
    coverage is insensitive to it while the *width path* is not.
    """
    alpha: float = 0.1
    gamma: float = 0.01
    window: int | None = 500      # None = use all history

    def run(self, preds: np.ndarray, truths: np.ndarray) -> Interval:
        """Sequential pass. Returns the realised intervals plus the alpha path."""
        p = np.asarray(preds, float).ravel()
        y = np.asarray(truths, float).ravel()
        if len(p) != len(y):
            raise ValueError("preds and truths must be the same length")

        lo = np.full(len(p), -np.inf)
        hi = np.full(len(p), np.inf)
        alpha_t = self.alpha
        alphas = np.empty(len(p))
        errs = np.zeros(len(p), dtype=bool)
        scores: list[float] = []

        for t in range(len(p)):
            alphas[t] = alpha_t
            if scores:
                hist = scores[-self.window:] if self.window else scores
                a = min(max(alpha_t, 1e-6), 1.0 - 1e-6)
                q = conformal_quantile(np.asarray(hist), a)
                lo[t], hi[t] = p[t] - q, p[t] + q
            errs[t] = not (lo[t] <= y[t] <= hi[t])
            scores.append(abs(y[t] - p[t]))
            alpha_t = alpha_t + self.gamma * (self.alpha - float(errs[t]))
            alpha_t = min(max(alpha_t, 0.0), 1.0)

        return Interval(lo, hi, self.alpha, "adaptive conformal (ACI)",
                        {"alpha_path": alphas.tolist(),
                         "gamma": self.gamma,
                         "realised_coverage": float(1.0 - errs.mean()),
                         "n_unbounded": int(np.sum(~np.isfinite(hi))),
                         "guarantee": "long-run only; not finite-sample"})


# ------------------------------------------------------------------ diagnostics

def coverage_by_group(interval: Interval, y: np.ndarray,
                      groups: np.ndarray) -> dict:
    """Coverage per group — the check that exposes marginal-vs-conditional.

    A method can hit 90% overall while badly under-covering a subgroup, and that
    is the failure a reader (or a validator) will ask about first.
    """
    c = interval.covers(y)
    g = np.asarray(groups).ravel()
    out = {}
    for key in dict.fromkeys(g.tolist()):
        m = g == key
        out[key] = {"coverage": float(np.mean(c[m])),
                    "mean_width": float(np.mean(interval.width[m])),
                    "n": int(m.sum())}
    spread = [v["coverage"] for v in out.values()]
    return {"per_group": out,
            "target": 1.0 - interval.alpha,
            "worst_group_coverage": float(min(spread)) if spread else float("nan"),
            "coverage_range": float(max(spread) - min(spread)) if spread else 0.0}


def coverage_by_bin(interval: Interval, y: np.ndarray, x: np.ndarray,
                    n_bins: int = 5) -> dict:
    """Coverage across quantile bins of a continuous conditioning variable."""
    x = np.asarray(x, float).ravel()
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges[1:-1], x), 0, n_bins - 1)
    labels = np.array([f"q{b + 1}" for b in bins])
    out = coverage_by_group(interval, y, labels)
    out["bin_edges"] = edges.tolist()
    return out


def rolling_coverage(interval: Interval, y: np.ndarray,
                     window: int = 100) -> np.ndarray:
    """Coverage in a moving window — for plotting whether it drifts over time."""
    c = interval.covers(y).astype(float)
    if window >= len(c):
        return np.array([c.mean()])
    k = np.ones(window) / window
    return np.convolve(c, k, mode="valid")
