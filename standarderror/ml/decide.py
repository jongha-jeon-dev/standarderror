r"""Arc X: from prediction to decision. What happens between a model's
output and an action taken on it, measured on simulations with a known
answer: the true probabilities and causal effects are known, and costs
are measured on large test sets.

Episode 46, `threshold_costs`: the 0.5 threshold, the cost-derived
threshold and a tuned one, for calibrated and overconfident models.
Episode 47, `auc_and_decisions`: the model with the best AUC against the
best model at each operating point.
Episode 48, `importance_vs_effect`: what a model relies on against what
changing a feature would do.
Episode 49, `uplift_targeting`: targeting a treatment by predicted risk
against by predicted effect.
Episode 50, `feedback_loop`: allocating effort by what past effort found.
"""

from __future__ import annotations

import functools
import warnings


def _np():
    import numpy as np
    return np


def _sig(z):
    np = _np()
    return 1.0 / (1.0 + np.exp(-z))


# ================================================================ episodes 46-47

@functools.cache
def _task(test: int = 50_000):
    """Episode 4's task with a larger test set and its exact posterior."""
    np = _np()
    from scipy.stats import multivariate_normal as mvn

    from standarderror.ml.curves import GaussianTask
    t = GaussianTask(test=test)
    l0 = mvn(np.zeros(t.d), t.cov0).logpdf(t.X_test)
    l1 = mvn(t.mu1, t.cov1).logpdf(t.X_test)
    return t, _sig(l1 - l0)


def _cost(p, y, thr, r):
    """Mean cost per case when a miss (positive not flagged) costs r and a
    false alarm costs 1, flagging every case with p >= thr."""
    np = _np()
    flag = p >= thr
    return float(np.mean(r * (~flag & (y == 1)) + (flag & (y == 0))))


def _best_cost(p, y, r):
    """The lowest cost any threshold on p achieves."""
    np = _np()
    o = np.argsort(p)
    ys = y[o]
    miss = np.concatenate([[0], np.cumsum(ys == 1)])
    false = np.concatenate([[np.sum(ys == 0)], np.sum(ys == 0) - np.cumsum(ys == 0)])
    return float(np.min(r * miss + false) / len(y))


def _best_threshold(p, y, r):
    """The threshold on p with the lowest cost on (p, y): midway between the
    two scores where the best cut falls."""
    np = _np()
    o = np.argsort(p)
    ps, ys = p[o], y[o]
    miss = np.concatenate([[0], np.cumsum(ys == 1)])
    false = np.concatenate([[np.sum(ys == 0)], np.sum(ys == 0) - np.cumsum(ys == 0)])
    i = int(np.argmin(r * miss + false))
    if i == 0:
        return float(ps[0]) - 1e-12
    if i == len(ps):
        return float(ps[-1]) + 1e-12
    return float((ps[i - 1] + ps[i]) / 2)


def _models():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier

    def hgb():
        return HistGradientBoostingClassifier(max_iter=300, early_stopping=False,
                                              random_state=0)
    return {"logistic": lambda: LogisticRegression(max_iter=2000),
            "naive Bayes": GaussianNB,
            "15-NN": lambda: KNeighborsClassifier(15),
            "forest": lambda: RandomForestClassifier(300, min_samples_leaf=5,
                                                     n_jobs=2, random_state=0),
            "boosted trees": hgb,
            "recalibrated boosted trees": lambda: CalibratedClassifierCV(
                hgb(), method="isotonic", cv=5)}


@functools.cache
def threshold_costs(*, ratios=(1, 2, 5, 10, 20), n: int = 2000, val: int = 1000,
                    reps: int = 5) -> dict:
    """Misses cost r times a false alarm. For each model, trained on n rows:
    the cost per case at threshold 0.5, at the cost-derived threshold
    1 / (1 + r), at a threshold tuned on `val` fresh rows, and at the best
    threshold in hindsight; and for the true posterior and for flagging
    everyone. Means over `reps` training sets, 50,000 test rows."""
    np = _np()
    t, post = _task()
    y = t.y_test
    makers = {k: v for k, v in _models().items() if k != "15-NN"}
    out = {"ratios": list(ratios),
           "bayes": {r: _cost(post, y, 1 / (1 + r), r) for r in ratios},
           "flag_all": {r: float(np.mean(y == 0)) for r in ratios},
           "models": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, M in makers.items():
            acc = {r: [] for r in ratios}
            for s in range(int(reps)):
                X, Y = t.draw(n, seed=10 + s)
                Xv, Yv = t.draw(val, seed=100 + s)
                m = M().fit(X, Y)
                p = m.predict_proba(t.X_test)[:, 1]
                pv = m.predict_proba(Xv)[:, 1]
                for r in ratios:
                    tuned = _best_threshold(pv, Yv, r)
                    acc[r].append((_cost(p, y, 0.5, r), _cost(p, y, 1 / (1 + r), r),
                                   _cost(p, y, tuned, r), _best_cost(p, y, r), tuned,
                                   float(np.mean(p < 0.01)), float(np.mean(p >= 1 / (1 + r)))))
            out["models"][name] = {
                r: dict(zip(("half", "derived", "tuned", "best", "tuned_t",
                             "below_001", "flag_rate"),
                            np.mean(acc[r], 0).tolist())) for r in ratios}
    return out


@functools.cache
def auc_and_decisions(*, n: int = 2000, reps: int = 10, ratios=(1, 2, 5, 10, 20),
                      capacities=(0.01, 0.05, 0.2)) -> dict:
    """Five models, `reps` training sets each. AUC; the best achievable cost
    at each cost ratio; precision among the top `capacities` share of
    cases; the share of positives among the bottom 2% and 5% (the cases
    a model would clear most confidently); and the AUC restricted to the high-sensitivity end (false
    negative rate at most 0.1). Means and standard deviations over training
    sets, and the share of training sets in which each model is best."""
    np = _np()
    from sklearn.metrics import roc_auc_score
    t, post = _task()
    y = t.y_test
    makers = {k: v for k, v in _models().items() if k != "recalibrated boosted trees"}
    raw = {k: [] for k in makers}
    for s in range(int(reps)):
        X, Y = t.draw(n, seed=10 + s)
        for k, M in makers.items():
            p = M().fit(X, Y).predict_proba(t.X_test)[:, 1]
            order = np.argsort(-p)
            row = {"auc": roc_auc_score(y, p),
                   "pauc_sens": roc_auc_score(1 - y, -p, max_fpr=0.1)}
            for r in ratios:
                row[f"cost{r}"] = _best_cost(p, y, r)
            for c in capacities:
                row[f"top{c}"] = float(y[order[:int(len(y) * c)]].mean())
            for c in (0.02, 0.05):
                row[f"bottom{c}"] = float(y[order[::-1][:int(len(y) * c)]].mean())
            raw[k].append(row)
    keys = list(raw[next(iter(raw))][0])
    mean = {k: {q: float(np.mean([r[q] for r in v])) for q in keys}
            for k, v in raw.items()}
    sd = {k: {q: float(np.std([r[q] for r in v])) for q in keys}
          for k, v in raw.items()}
    wins = {}
    for q in keys:
        lower = q.startswith(("cost", "bottom"))
        best = [min(raw, key=lambda k: raw[k][s][q]) if lower else
                max(raw, key=lambda k: raw[k][s][q]) for s in range(int(reps))]
        wins[q] = {k: best.count(k) / reps for k in raw}
    bayes = {"auc": float(roc_auc_score(y, post)),
             **{f"cost{r}": _best_cost(post, y, r) for r in ratios}}
    return {"mean": mean, "sd": sd, "wins": wins, "bayes": bayes,
            "ratios": list(ratios), "capacities": list(capacities)}


# ================================================================ episode 48

def _scm(n, g, do=None, shift=0.0):
    """A small causal system. U is an unmeasured trait. x1 is a habit that
    U lowers and that has no effect on the outcome. x2 is a treatment that U
    raises and that raises the outcome's log-odds by 0.5. x3 is a test
    result produced by the outcome's own latent state. With `do`, the named
    variable is shifted by `shift` after it is generated, and everything
    downstream of it follows."""
    np = _np()
    U = g.normal(size=n)
    x1 = -0.8 * U + 0.6 * g.normal(size=n)
    x2 = 0.6 * U + 0.8 * g.normal(size=n)
    if do == "x1":
        x1 = x1 + shift
    if do == "x2":
        x2 = x2 + shift
    latent = 1.2 * U + 0.5 * x2 + g.logistic(size=n)
    y = (latent > 0).astype(int)
    x3 = latent + 0.8 * g.normal(size=n)
    if do == "x3":
        x3 = x3 + shift
    x4 = g.normal(size=n)          # pure noise, for scale
    return np.column_stack([x1, x2, x3, x4]), y, U


@functools.cache
def importance_vs_effect(*, n: int = 20_000, seed: int = 0) -> dict:
    """Boosted trees and logistic regression fitted on x1..x4. For each
    feature: permutation importance (drop in AUC) on fresh rows; the
    model's 'what-if', the mean change in predicted probability when the
    feature is raised by one standard deviation in the model's input; and
    the true effect of raising it by one standard deviation in the system,
    from simulating the intervention. Also logistic regression given U and
    x2 only, the adjustment a causal analysis would make."""
    np = _np()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    g = np.random.default_rng(seed)
    X, y, U = _scm(n, g)
    Xt, yt, Ut = _scm(n, np.random.default_rng(seed + 1))
    sd = X.std(0)
    names = ["x1 habit", "x2 treatment", "x3 test result", "x4 noise"]
    models = {"boosted trees": HistGradientBoostingClassifier(random_state=0),
              "logistic": LogisticRegression(max_iter=2000)}
    out = {"names": names, "models": {}, "true": {}}
    base = _scm(200_000, np.random.default_rng(seed + 2))[1].mean()
    for j, nm in enumerate(names[:3]):
        key = nm.split()[0]
        yy = _scm(200_000, np.random.default_rng(seed + 2), do=key,
                  shift=float(sd[j]))[1]
        out["true"][nm] = float(yy.mean() - base)
    out["true"][names[3]] = 0.0
    for mk, m in models.items():
        m.fit(X, y)
        pi = permutation_importance(m, Xt, yt, scoring="roc_auc", n_repeats=5,
                                    random_state=0)
        p0 = m.predict_proba(Xt)[:, 1]
        whatif = {}
        for j, nm in enumerate(names):
            Xs = Xt.copy()
            Xs[:, j] += sd[j]
            whatif[nm] = float((m.predict_proba(Xs)[:, 1] - p0).mean())
        out["models"][mk] = {"importance": dict(zip(names, pi.importances_mean.tolist())),
                             "whatif": whatif}
    adj = LogisticRegression(max_iter=2000).fit(np.column_stack([U, X[:, 1]]), y)
    A = np.column_stack([Ut, Xt[:, 1]])
    p0 = adj.predict_proba(A)[:, 1]
    A[:, 1] += sd[1]
    out["adjusted_x2"] = float((adj.predict_proba(A)[:, 1] - p0).mean())
    alone = LogisticRegression(max_iter=2000).fit(X[:, [1]], y)
    C = Xt[:, [1]].copy()
    c0 = alone.predict_proba(C)[:, 1]
    C[:, 0] += sd[1]
    out["alone_x2"] = float((alone.predict_proba(C)[:, 1] - c0).mean())
    nox3 = LogisticRegression(max_iter=2000).fit(X[:, [0, 1, 3]], y)
    B = Xt[:, [0, 1, 3]].copy()
    q0 = nox3.predict_proba(B)[:, 1]
    B[:, 1] += sd[1]
    out["without_x3_x2"] = float((nox3.predict_proba(B)[:, 1] - q0).mean())
    out["sd"] = sd.tolist()
    return out


# ================================================================ episode 49

def _customers(n, g):
    """Churn without treatment: logistic in -1 + 1.5 x1 + x2. The offer
    changes the log-odds by tau(x): -1.6 exp(-x1^2) (most for middling risk,
    little for the lowest and highest), plus +0.8 for the customers with
    x3 > 1, whom an offer annoys."""
    np = _np()
    X = g.normal(size=(n, 5))
    z0 = -1 + 1.5 * X[:, 0] + X[:, 1]
    tau = -1.6 * np.exp(-X[:, 0] ** 2) + 0.8 * (X[:, 2] > 1)
    return X, _sig(z0), _sig(z0 + tau)


@functools.cache
def uplift_targeting(*, sizes=(2000, 20_000), budget: float = 0.2,
                     reps: int = 5) -> dict:
    """A randomised trial (half offered) of `n` customers, then a fresh
    population of 100,000 of whom `budget` can be offered. Churn prevented
    per 1,000 customers, computed from the true probabilities, for five
    ways of choosing whom to offer: at random; the highest predicted churn
    (a model of the control arm); the highest predicted reduction (two
    boosted models, one per arm, differenced: a T-learner); the same with
    logistic models; and the true best."""
    np = _np()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    Xp, p0, p1 = _customers(100_000, np.random.default_rng(999))
    gain = p0 - p1
    k = int(len(gain) * budget)
    rows = []

    def value(score):
        return float(gain[np.argsort(-score)[:k]].sum() / len(gain) * 1000)

    for n in sizes:
        acc = {}
        for s in range(int(reps)):
            g = np.random.default_rng(s)
            X, q0, q1 = _customers(int(n), g)
            T = g.random(int(n)) < 0.5
            y = (g.random(int(n)) < np.where(T, q1, q0)).astype(int)
            vals = {"random": value(g.random(len(gain))),
                    "true effect": value(gain)}
            for name, M in (("boosted", lambda: HistGradientBoostingClassifier(
                                max_iter=200, learning_rate=0.05, random_state=0)),
                            ("logistic", lambda: LogisticRegression(max_iter=2000))):
                mc = M().fit(X[~T], y[~T])
                mt = M().fit(X[T], y[T])
                pc = mc.predict_proba(Xp)[:, 1]
                pt = mt.predict_proba(Xp)[:, 1]
                vals[f"highest risk ({name})"] = value(pc)
                vals[f"largest effect ({name})"] = value(pc - pt)
            for kk, v in vals.items():
                acc.setdefault(kk, []).append(v)
        rows.append({"n": int(n), **{kk: float(np.mean(v)) for kk, v in acc.items()},
                     "sd": {kk: float(np.std(v)) for kk, v in acc.items()}})
    harmed = float(np.mean(gain < 0))
    top_risk = np.argsort(-p0)[:k]
    return {"rows": rows, "budget": budget, "harmed_share": harmed,
            "top_risk_mean_gain": float(gain[top_risk].mean()),
            "top_risk_harmed": float(np.mean(gain[top_risk] < 0)),
            "mean_gain": float(gain.mean()),
            "base_churn": float(p0.mean())}


@functools.cache
def uplift_budgets(*, budgets=(0.05, 0.1, 0.2, 0.5, 1.0), n: int = 20_000,
                   reps: int = 5) -> dict:
    """Churn prevented per 1,000 customers at each budget, ranking by the
    logistic risk model and by the logistic T-learner fitted on a trial of
    n customers, and by the true effect; means over `reps` trials."""
    np = _np()
    from sklearn.linear_model import LogisticRegression
    Xp, p0, p1 = _customers(100_000, np.random.default_rng(999))
    gain = p0 - p1
    order_true = np.argsort(-gain)
    out = {b: {"risk": [], "effect": []} for b in budgets}
    for s in range(int(reps)):
        g = np.random.default_rng(s)
        X, q0, q1 = _customers(int(n), g)
        T = g.random(int(n)) < 0.5
        y = (g.random(int(n)) < np.where(T, q1, q0)).astype(int)
        pc = LogisticRegression(max_iter=2000).fit(X[~T], y[~T]).predict_proba(Xp)[:, 1]
        pt = LogisticRegression(max_iter=2000).fit(X[T], y[T]).predict_proba(Xp)[:, 1]
        for b in budgets:
            k = int(len(gain) * b)
            out[b]["risk"].append(gain[np.argsort(-pc)[:k]].sum() / len(gain) * 1000)
            out[b]["effect"].append(gain[np.argsort(-(pc - pt))[:k]].sum()
                                    / len(gain) * 1000)
    return {b: {"risk": float(np.mean(v["risk"])),
                "effect": float(np.mean(v["effect"])),
                "true": float(gain[order_true[:int(len(gain) * b)]].sum()
                              / len(gain) * 1000),
                "random": float(b * gain.sum() / len(gain) * 1000)}
            for b, v in out.items()}


# ================================================================ episode 50

@functools.cache
def feedback_loop(*, rates=(0.10, 0.10), days: int = 365, visits: int = 10,
                  runs: int = 500, eps: float = 0.1, seed: int = 0) -> dict:
    """Two districts with true daily violation rates per visit `rates`. Each
    day `visits` inspections are split between them by a rule based on
    what past inspections found. Rules: 'greedy' (all visits to the
    district with more violations found so far), 'proportional' (visits in
    proportion to violations found so far), 'per visit' (in proportion to
    violations found per visit made), 'greedy per visit' (all visits to the
    district with more violations found per visit) and 'greedy per visit,
    explore' (the same, with a share `eps` of visits split evenly). Every rule starts from one
    found violation per district. Returns, per rule, the distribution of
    district A's share of visits on the last day over `runs` simulations."""
    np = _np()
    g = np.random.default_rng(seed)
    out = {}
    for rule in ("greedy", "proportional", "per visit", "greedy per visit",
                 "greedy per visit, explore"):
        final = []
        for _ in range(int(runs)):
            found = np.ones(2)
            made = np.ones(2)
            share = 0.5
            for _day in range(int(days)):
                if rule == "greedy":
                    share = 1.0 if found[0] > found[1] else (
                        0.0 if found[1] > found[0] else 0.5)
                elif rule == "proportional":
                    share = found[0] / found.sum()
                elif rule == "per visit":
                    est = found / made
                    share = est[0] / est.sum()
                else:
                    est = found / made
                    share = 1.0 if est[0] > est[1] else (
                        0.0 if est[1] > est[0] else 0.5)
                    if rule == "greedy per visit, explore":
                        share = (1 - eps) * share + eps * 0.5
                va = g.binomial(int(visits), share)
                v = np.array([va, int(visits) - va])
                found += g.binomial(v, rates)
                made += v
            final.append(share)
        f = np.array(final)
        out[rule] = {"mean": float(f.mean()), "sd": float(f.std()),
                     "q10": float(np.quantile(f, 0.1)),
                     "q90": float(np.quantile(f, 0.9)),
                     "locked": float(np.mean((f <= 0.05 + 1e-9) | (f >= 0.95 - 1e-9))),
                     "within_10": float(np.mean(np.abs(f - rates[0] / sum(rates)) < 0.1)),
                     "mostly_a": float(np.mean(f > 0.5)),
                     "final": f.tolist()}
    out["true_share"] = rates[0] / sum(rates)
    return out
