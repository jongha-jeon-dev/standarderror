"""Metrics that survive review.

Opinionated choices, each because the obvious alternative misleads:

* **MASE over MAPE.** MAPE is undefined at zero, asymmetric, and unbounded — on
  rate spreads and returns it is actively harmful. MASE scales by the in-sample
  naive-forecast error, so 1.0 means "no better than persistence" and the number
  is comparable across series.
* **Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction.**
  Comparing two forecasts by "lower RMSE" tells you nothing about whether the
  difference is distinguishable from noise. `dm_test` gives you a p-value.
* **Pinball loss and PIT** for anything probabilistic. Point forecasts of risk
  are not risk forecasts.
* **Kupiec and Christoffersen** VaR backtests, because a VaR model that breaches
  at the right rate but in clusters is still broken.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _flat(a) -> np.ndarray:
    return np.asarray(a, float).ravel()


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _flat(y), _flat(p)
    return float(np.sqrt(np.nanmean((y - p) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.nanmean(np.abs(_flat(y) - _flat(p))))


def nrmse(y: np.ndarray, p: np.ndarray) -> float:
    """RMSE divided by the RMS of the truth — the RC-literature convention."""
    y, p = _flat(y), _flat(p)
    return float(np.sqrt(np.nanmean((y - p) ** 2) / np.nanmean(y ** 2)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _flat(y), _flat(p)
    ss_res = np.nansum((y - p) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def mase(y: np.ndarray, p: np.ndarray, y_train: np.ndarray,
         season: int = 1) -> float:
    """Mean absolute scaled error. 1.0 == seasonal-naive skill."""
    y, p, y_train = _flat(y), _flat(p), _flat(y_train)
    denom = np.nanmean(np.abs(y_train[season:] - y_train[:-season]))
    if denom <= 0:
        return float("nan")
    return float(np.nanmean(np.abs(y - p)) / denom)


def directional_accuracy(y: np.ndarray, p: np.ndarray,
                         prev: np.ndarray | None = None) -> float:
    """Share of correct sign calls on the change.

    Pass `prev` (the last observed level) when y/p are levels; otherwise y/p are
    assumed to already be changes. Getting this wrong is the single most common
    way a "70% hit rate" turns out to be measuring nothing.
    """
    y, p = _flat(y), _flat(p)
    if prev is not None:
        prev = _flat(prev)
        dy, dp = y - prev, p - prev
    else:
        dy, dp = y, p
    ok = np.isfinite(dy) & np.isfinite(dp) & (dy != 0)
    if ok.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(dy[ok]) == np.sign(dp[ok])))


def dm_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, *,
            h: int = 1, loss: str = "mse") -> dict:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.

    H0: the two forecasts have equal expected loss. Negative statistic favours
    `p1`. Uses a t-distribution with T-1 df, which is the small-sample-corrected
    form; the asymptotic normal version over-rejects badly for T < 200.

    **Multivariate inputs are aggregated per time step**, not flattened: for a
    (T, k) forecast the loss differential is `||e1_t||^2 - ||e2_t||^2`. Flattening
    would treat each coordinate as an independent observation, inflating T by a
    factor of k and shrinking the p-value accordingly. It also means the test and
    a reported multivariate RMSE agree about which model is better — they do not
    if you test one coordinate and report the norm.
    """
    y = np.asarray(y, float)
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    if y.ndim > 1 and y.shape[-1] > 1:
        e1 = y - p1
        e2 = y - p2
        if loss == "mse":
            d = np.sum(e1 ** 2, axis=-1) - np.sum(e2 ** 2, axis=-1)
        elif loss == "mae":
            d = np.sum(np.abs(e1), axis=-1) - np.sum(np.abs(e2), axis=-1)
        else:
            raise ValueError("loss must be 'mse' or 'mae'")
        d = d.ravel()
    else:
        y, p1, p2 = _flat(y), _flat(p1), _flat(p2)
        e1, e2 = y - p1, y - p2
        if loss == "mse":
            d = e1 ** 2 - e2 ** 2
        elif loss == "mae":
            d = np.abs(e1) - np.abs(e2)
        else:
            raise ValueError("loss must be 'mse' or 'mae'")
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 10:
        raise ValueError("need at least 10 comparable observations")
    d_bar = d.mean()
    # Newey-West style long-run variance with h-1 autocovariances.
    gamma0 = np.sum((d - d_bar) ** 2) / T
    var = gamma0
    for k in range(1, h):
        gk = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / T
        var += 2.0 * gk
    if var <= 0:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n": T, "note": "non-positive long-run variance"}
    stat = d_bar / np.sqrt(var / T)
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat *= correction
    p = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=T - 1))
    return {"statistic": float(stat), "p_value": float(p), "n": T,
            "mean_loss_diff": float(d_bar), "horizon": h,
            "favours": "model_1" if stat < 0 else "model_2"}


def pinball_loss(y: np.ndarray, q_pred: np.ndarray,
                 quantiles: np.ndarray) -> float:
    """Average quantile (pinball) loss. `q_pred` is (T, n_quantiles)."""
    y = _flat(y)[:, None]
    q_pred = np.asarray(q_pred, float)
    qs = np.asarray(quantiles, float)[None, :]
    diff = y - q_pred
    return float(np.nanmean(np.maximum(qs * diff, (qs - 1.0) * diff)))


def pit_values(y: np.ndarray, cdf) -> np.ndarray:
    """Probability integral transform. Should be Uniform(0,1) if calibrated."""
    y = _flat(y)
    return np.array([float(cdf(i, v)) for i, v in enumerate(y)])


def pit_uniformity(pit: np.ndarray) -> dict:
    """Kolmogorov-Smirnov test of PIT values against Uniform(0,1)."""
    pit = _flat(pit)
    pit = pit[np.isfinite(pit)]
    ks = stats.kstest(pit, "uniform")
    return {"ks_statistic": float(ks.statistic), "p_value": float(ks.pvalue),
            "n": int(len(pit)), "mean": float(pit.mean())}


def kupiec_pof(breaches: np.ndarray, alpha: float) -> dict:
    """Kupiec proportion-of-failures (unconditional coverage) LR test."""
    x = _flat(breaches).astype(bool)
    n, k = len(x), int(x.sum())
    if n == 0:
        raise ValueError("empty breach series")
    pi = k / n
    if k == 0 or k == n:
        lr = 0.0 if k == 0 and alpha == 0 else float("inf")
    else:
        lr = -2.0 * ((n - k) * np.log(1 - alpha) + k * np.log(alpha)
                     - (n - k) * np.log(1 - pi) - k * np.log(pi))
    p = float(1.0 - stats.chi2.cdf(lr, df=1)) if np.isfinite(lr) else 0.0
    return {"n": n, "breaches": k, "observed_rate": float(pi),
            "expected_rate": alpha, "lr_statistic": float(lr), "p_value": p}


def christoffersen_independence(breaches: np.ndarray) -> dict:
    """LR test that breaches are serially independent (no clustering).

    A VaR model can pass Kupiec and still be unusable: the right *number* of
    breaches arriving all in one week is exactly the failure mode that matters.
    """
    x = _flat(breaches).astype(int)
    if len(x) < 3:
        raise ValueError("need at least 3 observations")
    n00 = int(np.sum((x[:-1] == 0) & (x[1:] == 0)))
    n01 = int(np.sum((x[:-1] == 0) & (x[1:] == 1)))
    n10 = int(np.sum((x[:-1] == 1) & (x[1:] == 0)))
    n11 = int(np.sum((x[:-1] == 1) & (x[1:] == 1)))
    denom0, denom1 = n00 + n01, n10 + n11
    if denom0 == 0 or denom1 == 0 or (n01 + n11) == 0:
        return {"lr_statistic": float("nan"), "p_value": float("nan"),
                "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
                "note": "insufficient transitions to identify the test"}
    pi01, pi11 = n01 / denom0, n11 / denom1
    pi = (n01 + n11) / (denom0 + denom1)
    def ll(p, a, b):
        if p in (0.0, 1.0):
            return 0.0
        return a * np.log(1 - p) + b * np.log(p)
    lr = -2.0 * (ll(pi, n00 + n10, n01 + n11)
                 - ll(pi01, n00, n01) - ll(pi11, n10, n11))
    return {"lr_statistic": float(lr),
            "p_value": float(1.0 - stats.chi2.cdf(lr, df=1)),
            "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
            "pi01": float(pi01), "pi11": float(pi11)}


def summary(y: np.ndarray, p: np.ndarray,
            y_train: np.ndarray | None = None) -> dict:
    out = {"rmse": rmse(y, p), "mae": mae(y, p), "nrmse": nrmse(y, p),
           "r2": r2(y, p)}
    if y_train is not None:
        out["mase"] = mase(y, p, y_train)
    return out
