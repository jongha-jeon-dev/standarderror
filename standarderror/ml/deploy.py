r"""Arc XIII: after the model ships. Monitoring, intervals, fairness,
retraining and replication, each measured on a simulation with a known
answer and, where scikit-learn's bundled data allow, on real data.

Episode 61, `drift_*`: what an alarm on the inputs can and cannot see.
Episode 62, `intervals_*`: prediction intervals for a regression, on
average and where it matters.
Episode 63, `fairness_*`: calibration within groups against equal error
rates, and what leaving the group out does.
Episode 64, `retraining`: how much history to retrain on, and what a label
delay costs.
Episode 65, `replicate_*`: four verdicts first measured on simulations,
repeated on bundled real data.

Episode 29 already measured what covariate shift costs a model; this arc
asks how anyone would know.
"""

from __future__ import annotations

import warnings

import numpy as np


def _quiet():
    warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# ML 61: drift alarms
# --------------------------------------------------------------------------

BETA = np.array([1.5, -1.0, 0.8, 0.0, 0.0])
B0 = -0.3


def drift_world(n, rng, *, shift: float = 0.0, flip: float = 0.0):
    """Five standard Gaussian inputs, a logistic outcome on three of them.
    `shift` moves the first input's mean (P(y | x) unchanged); `flip` moves
    the first coefficient from +1.5 towards -1.5 (P(x) unchanged)."""
    X = rng.normal(size=(n, len(BETA)))
    X[:, 0] += shift
    b = BETA.copy()
    b[0] = BETA[0] * (1 - 2 * flip)
    p = 1 / (1 + np.exp(-(B0 + X @ b)))
    return X, (rng.random(n) < p).astype(int)


def _domain_weights(X_ref, X_new):
    """Density-ratio weights for the reference rows, from a classifier
    trained to tell reference rows from new ones."""
    from sklearn.linear_model import LogisticRegression
    Z = np.r_[X_ref, X_new]
    t = np.r_[np.zeros(len(X_ref)), np.ones(len(X_new))]
    c = LogisticRegression(max_iter=1000).fit(Z, t)
    p = c.predict_proba(X_ref)[:, 1]
    return p / (1 - p) * len(X_ref) / len(X_new)


SCENARIOS = {
    "no drift": {},
    "inputs move 1 sd": {"shift": 1.0},
    "inputs move 0.1 sd": {"shift": 0.1},
    "relation half-reversed": {"flip": 0.5},
    "relation reversed": {"flip": 1.0},
}


def drift_monitors(*, batches=(200, 2000), reps: int = 50,
                   seed: int = 0) -> dict:
    """A logistic model trained on 5,000 rows of the undrifted world, then
    deployed on batches from each scenario. Per batch: an input alarm (any
    of five two-sample Kolmogorov-Smirnov tests against 2,000 training
    inputs below 0.05/5), a score alarm (the same test on the model's
    predicted probabilities), the true accuracy, and an estimate of the
    accuracy from the unlabelled batch alone (labelled validation rows
    re-weighted by a domain classifier's density ratio)."""
    _quiet()
    from scipy.stats import ks_2samp
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    Xtr, ytr = drift_world(5000, rng)
    m = LogisticRegression().fit(Xtr, ytr)
    ref = Xtr[:2000]
    sref = m.predict_proba(ref)[:, 1]
    Xv, yv = drift_world(3000, rng)
    right_v = m.predict(Xv) == yv
    out = {"validation_accuracy": float(right_v.mean()), "rows": []}
    for name, kw in SCENARIOS.items():
        for bs in batches:
            a = []
            for _ in range(reps):
                Xb, yb = drift_world(bs, rng, **kw)
                pin = min(ks_2samp(ref[:, j], Xb[:, j]).pvalue
                          for j in range(Xb.shape[1]))
                psc = ks_2samp(sref, m.predict_proba(Xb)[:, 1]).pvalue
                w = _domain_weights(Xv, Xb)
                a.append((pin < 0.05 / Xb.shape[1], psc < 0.05,
                          float(np.mean(m.predict(Xb) == yb)),
                          float(np.sum(w * right_v) / np.sum(w))))
            a = np.array(a, dtype=float)
            out["rows"].append({"scenario": name, "batch": bs,
                                "input_alarm": float(a[:, 0].mean()),
                                "score_alarm": float(a[:, 1].mean()),
                                "accuracy": float(a[:, 2].mean()),
                                "estimate": float(a[:, 3].mean())})
    return out


def drift_label_shift(prevalences=(0.45, 0.2, 0.8), *, batch: int = 2000,
                      reps: int = 30, seed: int = 0) -> list[dict]:
    """Label shift: the outcome's prevalence changes and each class's inputs
    do not. Batches are drawn from a 200,000-row pool of the undrifted world
    at the new prevalence."""
    _quiet()
    from scipy.stats import ks_2samp
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    Xtr, ytr = drift_world(5000, rng)
    m = LogisticRegression().fit(Xtr, ytr)
    ref = Xtr[:2000]
    Xp, yp = drift_world(200_000, rng)
    i1, i0 = np.flatnonzero(yp == 1), np.flatnonzero(yp == 0)
    out = []
    for pi in prevalences:
        a = []
        for _ in range(reps):
            k = rng.binomial(batch, pi)
            idx = np.r_[rng.choice(i1, k), rng.choice(i0, batch - k)]
            Xb, yb = Xp[idx], yp[idx]
            pin = min(ks_2samp(ref[:, j], Xb[:, j]).pvalue for j in range(5))
            p = m.predict_proba(Xb)[:, 1]
            a.append((pin < 0.01, float(np.mean((p > 0.5) == yb)),
                      float(p.mean()), float(yb.mean())))
        a = np.array(a, dtype=float)
        out.append({"prevalence": pi, "train_prevalence": float(ytr.mean()),
                    "input_alarm": float(a[:, 0].mean()),
                    "accuracy": float(a[:, 1].mean()),
                    "mean_predicted": float(a[:, 2].mean()),
                    "mean_actual": float(a[:, 3].mean())})
    return out


def drift_false_alarms(features=(5, 20, 100), *, batch: int = 500,
                       batches: int = 200, seed: int = 1) -> list[dict]:
    """No drift at all: how often at least one per-feature test fires in a
    batch, uncorrected at 0.05 and with a Bonferroni correction."""
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    out = []
    for F in features:
        raw, bon = [], []
        for _ in range(batches):
            # A fresh reference sample each time, so the rate is the test's
            # own, not conditional on one unlucky reference draw.
            R = rng.normal(size=(2000, F))
            B = rng.normal(size=(batch, F))
            p = np.array([ks_2samp(R[:, j], B[:, j]).pvalue for j in range(F)])
            raw.append(bool((p < 0.05).any()))
            bon.append(bool((p < 0.05 / F).any()))
        out.append({"features": F, "uncorrected": float(np.mean(raw)),
                    "bonferroni": float(np.mean(bon)),
                    "weekly_alarms_a_year": float(np.mean(raw) * 52)})
    return out


def drift_breast_cancer(quantiles=(None, 0.5, 0.75), *, seed: int = 0) -> list[dict]:
    """A real shift: a model trained on a random 300 tumours, deployed on
    held-out tumours whose mean radius exceeds a training quantile."""
    _quiet()
    from scipy.stats import ks_2samp
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = load_breast_cancer(return_X_y=True)
    idx = np.random.default_rng(seed).permutation(len(y))
    tr, te = idx[:300], idx[300:]
    mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(
        X[tr], y[tr])
    out = []
    for q in quantiles:
        sel = te if q is None else te[X[te, 0] > np.quantile(X[tr, 0], q)]
        p = [ks_2samp(X[tr][:, j], X[sel][:, j]).pvalue for j in range(30)]
        right = mdl.predict(X[sel]) == y[sel]
        out.append({"quantile": q, "rows": int(len(sel)),
                    "features_alarmed": int(sum(v < 0.05 / 30 for v in p)),
                    "accuracy": float(right.mean()),
                    "errors": int((~right).sum()),
                    "malignant": float(1 - y[sel].mean())})
    return out


# --------------------------------------------------------------------------
# ML 62: prediction intervals for a regression
# --------------------------------------------------------------------------

def _hetero(n, rng):
    """y = sin x + x/2 + noise whose sd grows from 0.2 at x = 0 to 3.2 at
    x = 10."""
    x = rng.uniform(0, 10, n)
    y = np.sin(x) + 0.5 * x + (0.2 + 0.3 * x) * rng.normal(size=n)
    return x[:, None], y


def intervals_simulation(*, reps: int = 20, level: float = 0.95) -> dict:
    """Five ways to put a 95% interval round a regression's prediction,
    each fitted on 2,000 rows (and 1,000 calibration rows where used) and
    checked on 20,000 fresh ones, overall and in each tenth of the input's
    range. Means over `reps` independent data sets."""
    _quiet()
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    z = 1.959964
    names = ["boosting, +-1.96 training residual sd",
             "boosting, +-1.96 held-out residual sd",
             "linear, +-1.96 held-out residual sd",
             "split conformal on boosting",
             "conformalised quantile regression"]
    acc = {k: [] for k in names}
    for r in range(reps):
        rng = np.random.default_rng(r)
        Xtr, ytr = _hetero(2000, rng)
        Xca, yca = _hetero(1000, rng)
        Xte, yte = _hetero(20000, rng)
        dec = np.minimum((Xte[:, 0]).astype(int), 9)
        q_lvl = min(1.0, level * (1 + 1 / len(yca)))
        g = GradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
        p = g.predict(Xte)
        s_tr = np.std(ytr - g.predict(Xtr))
        s_ca = np.std(yca - g.predict(Xca))
        q = np.quantile(np.abs(yca - g.predict(Xca)), q_lvl)
        lin = LinearRegression().fit(Xtr, ytr)
        pl = lin.predict(Xte)
        s_l = np.std(yca - lin.predict(Xca))
        a = (1 - level) / 2
        lo = GradientBoostingRegressor(loss="quantile", alpha=a,
                                       random_state=0).fit(Xtr, ytr)
        hi = GradientBoostingRegressor(loss="quantile", alpha=1 - a,
                                       random_state=0).fit(Xtr, ytr)
        e = np.maximum(lo.predict(Xca) - yca, yca - hi.predict(Xca))
        qq = np.quantile(e, q_lvl)
        bands = {names[0]: (p - z * s_tr, p + z * s_tr),
                 names[1]: (p - z * s_ca, p + z * s_ca),
                 names[2]: (pl - z * s_l, pl + z * s_l),
                 names[3]: (p - q, p + q),
                 names[4]: (lo.predict(Xte) - qq, hi.predict(Xte) + qq)}
        for k, (L, H) in bands.items():
            c = (yte >= L) & (yte <= H)
            acc[k].append([c.mean()] + [c[dec == d].mean() for d in range(10)]
                          + [float(np.mean(H - L))])
    out = {"level": level, "reps": reps, "methods": {}}
    for k, v in acc.items():
        v = np.array(v)
        m = v.mean(0)
        out["methods"][k] = {"coverage": float(m[0]),
                             "by_tenth": m[1:11].tolist(),
                             "worst": float(m[1:11].min()),
                             "best": float(m[1:11].max()),
                             "width": float(m[11]),
                             "coverage_sd": float(v[:, 0].std())}
    return out


def intervals_diabetes(*, reps: int = 100, level: float = 0.9) -> dict:
    """The diabetes data, 442 patients: 250 to train, 100 to calibrate, 92
    to test, `reps` random splits. Split conformal on a linear model, and
    conformalised quantile regression on shallow boosting; coverage overall
    and in each third of the predicted value."""
    _quiet()
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    X, y = load_diabetes(return_X_y=True)
    rows = {"split conformal": [], "CQR": []}
    a = (1 - level) / 2
    for s in range(reps):
        i = np.random.default_rng(s).permutation(len(y))
        tr, ca, te = i[:250], i[250:350], i[350:]
        q_lvl = min(1.0, level * (1 + 1 / len(ca)))
        m = LinearRegression().fit(X[tr], y[tr])
        p = m.predict(X[te])
        q = np.quantile(np.abs(y[ca] - m.predict(X[ca])), q_lvl)
        c1 = np.abs(y[te] - p) <= q
        kw = dict(n_estimators=100, max_depth=2, random_state=0)
        lo = GradientBoostingRegressor(loss="quantile", alpha=a, **kw).fit(
            X[tr], y[tr])
        hi = GradientBoostingRegressor(loss="quantile", alpha=1 - a, **kw).fit(
            X[tr], y[tr])
        e = np.maximum(lo.predict(X[ca]) - y[ca], y[ca] - hi.predict(X[ca]))
        qq = np.quantile(e, q_lvl)
        L, H = lo.predict(X[te]) - qq, hi.predict(X[te]) + qq
        c2 = (y[te] >= L) & (y[te] <= H)
        third = np.digitize(p, np.quantile(p, [1 / 3, 2 / 3]))
        for k, c, w in (("split conformal", c1, 2 * q),
                        ("CQR", c2, float(np.mean(H - L)))):
            rows[k].append([c.mean()] + [c[third == t].mean() for t in range(3)]
                           + [w])
    out = {"level": level, "reps": reps}
    for k, v in rows.items():
        m = np.array(v).mean(0)
        out[k] = {"coverage": float(m[0]), "low": float(m[1]),
                  "middle": float(m[2]), "high": float(m[3]),
                  "width": float(m[4])}
    return out


# --------------------------------------------------------------------------
# ML 63: fairness
# --------------------------------------------------------------------------

def fairness_world(n, rng):
    """Two groups of equal size. A latent risk z is 0.6 lower on average in
    group A; the outcome depends on z alone. A measured feature x1 is z plus
    noise; a proxy (a postcode, say) is the group plus noise."""
    g = rng.integers(0, 2, n)
    z = rng.normal(size=n) + np.where(g == 1, 0.0, -0.6)
    x1 = z + rng.normal(scale=0.7, size=n)
    proxy = g + rng.normal(scale=0.5, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.6 * z - 0.2)))).astype(int)
    return g, x1, proxy, y


def _group_rates(p, f, y, g):
    from sklearn.metrics import roc_auc_score
    out = {"auc": float(roc_auc_score(y, p)), "accuracy": float(np.mean(f == y)),
           "group_auc": float(roc_auc_score(g, p))}
    for k, lab in ((0, "A"), (1, "B")):
        s = g == k
        out[lab] = {"base_rate": float(y[s].mean()),
                    "flagged": float(f[s].mean()),
                    "fpr": float(f[s & (y == 0)].mean()),
                    "fnr": float((~f)[s & (y == 1)].mean()),
                    "ppv": float(y[s & f].mean()),
                    "calibration_gap": float(p[s].mean() - y[s].mean())}
    return out


def fairness_study(*, n: int = 40_000, seed: int = 0) -> dict:
    """Three logistic models on 40,000 training rows and 40,000 test rows:
    with the group, with a proxy instead, and with neither; each at a common
    threshold of 0.5. Then the first model with group A's threshold moved so
    that the two groups' false positive rates are equal."""
    _quiet()
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    g, x1, px, y = fairness_world(n, rng)
    G, X1, PX, Y = fairness_world(n, rng)
    feats = {"with group": (np.c_[x1, g], np.c_[X1, G]),
             "with a proxy": (np.c_[x1, px], np.c_[X1, PX]),
             "without either": (x1[:, None], X1[:, None])}
    out = {}
    for name, (a, b) in feats.items():
        m = LogisticRegression().fit(a, y)
        p = m.predict_proba(b)[:, 1]
        out[name] = _group_rates(p, p > 0.5, Y, G)
    m = LogisticRegression().fit(np.c_[x1, g], y)
    p = m.predict_proba(np.c_[X1, G])[:, 1]
    fpr_b = np.mean(p[(G == 1) & (Y == 0)] > 0.5)
    t0 = float(np.quantile(p[(G == 0) & (Y == 0)], 1 - fpr_b))
    f = np.where(G == 0, p > t0, p > 0.5)
    out["equal false positive rates"] = _group_rates(p, f, Y, G)
    out["equal false positive rates"]["threshold_A"] = t0
    # Costs at a miss = 1, false alarm = 1, as a single scalar of harm.
    return out


# --------------------------------------------------------------------------
# ML 64: retraining
# --------------------------------------------------------------------------

POLICIES = ("static", "expanding", "window 1", "window 4", "window 16")
WORLDS = ("stable", "slow drift", "fast drift", "jump")


def _stream(kind, rng, periods, rows, dim):
    b = rng.normal(size=dim)
    b = 2 * b / np.linalg.norm(b)
    Xs, ys = [], []
    for t in range(periods):
        if kind in ("slow drift", "fast drift"):
            b = b + rng.normal(scale=0.05 if kind == "slow drift" else 0.2,
                               size=dim)
            b = 2 * b / np.linalg.norm(b)
        if kind == "jump" and t == periods // 2:
            c = rng.normal(size=dim)
            b = 2 * c / np.linalg.norm(c)
        X = rng.normal(size=(rows, dim))
        Xs.append(X)
        ys.append((rng.random(rows) < 1 / (1 + np.exp(-X @ b))).astype(int))
    return Xs, ys


def retraining(*, delays=(0, 8), periods: int = 80, rows: int = 60,
               dim: int = 10, warm: int = 10, reps: int = 6) -> dict:
    """A logistic model retrained every period on the labelled rows it has.
    Each period brings `rows` new rows; their labels arrive `delay` periods
    later. Policies: never retrain after the first `warm` periods (static),
    use all labelled history (expanding), or the last 1, 4 or 16 labelled
    periods. Worlds: a fixed relation, one whose coefficient direction
    wanders slowly or fast at constant strength, and one that jumps to a new
    direction halfway. Accuracy on each period's rows, averaged over periods
    `warm` to the end and over `reps` streams."""
    _quiet()
    from sklearn.linear_model import LogisticRegression
    out = {"periods": periods, "rows": rows, "dim": dim, "results": {}}
    for kind in WORLDS:
        for delay in delays:
            res = {}
            curves = {}
            for pol in POLICIES:
                runs = []
                for r in range(reps):
                    rng = np.random.default_rng([r, WORLDS.index(kind)])
                    Xs, ys = _stream(kind, rng, periods, rows, dim)
                    acc = []
                    for t in range(warm, periods):
                        last = t - 1 - delay
                        if pol == "static":
                            lo, hi = 0, warm
                        elif pol == "expanding":
                            lo, hi = 0, last + 1
                        else:
                            w = int(pol.split()[1])
                            lo, hi = max(0, last + 1 - w), last + 1
                        hi = max(hi, 1)
                        lo = min(lo, hi - 1)
                        X = np.concatenate(Xs[lo:hi])
                        y = np.concatenate(ys[lo:hi])
                        m = LogisticRegression().fit(X, y)
                        acc.append(float(np.mean(m.predict(Xs[t]) == ys[t])))
                    runs.append(acc)
                runs = np.array(runs)
                res[pol] = float(runs.mean())
                curves[pol] = runs.mean(0).tolist()
            out["results"][f"{kind}/{delay}"] = {"mean": res, "curve": curves}
    return out


# --------------------------------------------------------------------------
# ML 65: replication on real data
# --------------------------------------------------------------------------

def replicate_imbalance(*, reps: int = 20) -> dict:
    """Episode 5 on real data: is a digit an 8 (about 10% of the digits)?
    Logistic regression on standardised pixels, trained on 1,000 images and
    tested on 797, plain and after oversampling, SMOTE and class weights."""
    _quiet()
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from standarderror.ml import imbalance as im
    X, y = load_digits(return_X_y=True)
    y = (y == 8).astype(int)
    res = {m: [] for m in ("plain", "oversample", "SMOTE", "class weights")}
    for s in range(reps):
        r = np.random.default_rng(s)
        i = r.permutation(len(y))
        tr, te = i[:1000], i[1000:]
        sc = StandardScaler().fit(X[tr])
        A, B = sc.transform(X[tr]), sc.transform(X[te])
        for m in res:
            kw = dict(max_iter=3000, C=0.1)
            if m == "plain":
                mdl = LogisticRegression(**kw).fit(A, y[tr])
            elif m == "class weights":
                mdl = LogisticRegression(class_weight="balanced", **kw).fit(
                    A, y[tr])
            else:
                f = im.oversample if m == "oversample" else im.smote
                Xa, ya = f(A, y[tr], r)
                mdl = LogisticRegression(**kw).fit(Xa, ya)
            p = mdl.predict_proba(B)[:, 1]
            res[m].append((roc_auc_score(y[te], p), p.mean(), y[te].mean(),
                           log_loss(y[te], p)))
    out = {}
    for m, v in res.items():
        a = np.array(v)
        out[m] = {"auc": float(a[:, 0].mean()), "mean_p": float(a[:, 1].mean()),
                  "rate": float(a[:, 2].mean()),
                  "log_loss": float(a[:, 3].mean()),
                  "auc_paired": float((a[:, 0] - np.array(res["plain"])[:, 0]
                                       ).mean())}
    return out


def replicate_tuning(*, reps: int = 20, n: int = 200) -> dict:
    """Episode 35 on real data: an RBF-SVM grid of 64 configurations searched
    by 5-fold CV on 200 breast-cancer tumours, against the default model,
    both scored on the remaining 369."""
    _quiet()
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from standarderror.ml import around
    X, y = load_breast_cancer(return_X_y=True)
    grid = {"svc__" + k: v for k, v in around._grid(64).items()}
    rows = []
    for s in range(reps):
        i = np.random.default_rng(s).permutation(len(y))
        tr, te = i[:n], i[n:]
        G = GridSearchCV(make_pipeline(StandardScaler(), SVC()), grid,
                         cv=StratifiedKFold(5, shuffle=True, random_state=s)
                         ).fit(X[tr], y[tr])
        d = make_pipeline(StandardScaler(), SVC()).fit(X[tr], y[tr])
        rows.append((G.best_score_, np.mean(G.predict(X[te]) == y[te]),
                     np.mean(d.predict(X[te]) == y[te])))
    a = np.array(rows)
    return {"best_cv": float(a[:, 0].mean()), "tuned": float(a[:, 1].mean()),
            "default": float(a[:, 2].mean()),
            "optimism": float((a[:, 0] - a[:, 1]).mean()),
            "optimism_se": float((a[:, 0] - a[:, 1]).std(ddof=1)
                                 / np.sqrt(len(a))),
            "tuned_minus_default": float((a[:, 1] - a[:, 2]).mean()),
            "tuned_minus_default_se": float((a[:, 1] - a[:, 2]).std(ddof=1)
                                            / np.sqrt(len(a)))}


def replicate_decisions(*, reps: int = 20, ratio: float = 10.0) -> dict:
    """Episodes 46 and 47 on real data: malignant as the positive class, a
    missed malignancy costing ten false alarms; six models trained on 300
    tumours and scored on 269, `reps` splits. Cost per case at 0.5, at
    1/(1 + r), and at the best threshold in hindsight; which model had the
    best AUC and which the lowest cost."""
    _quiet()
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import roc_auc_score

    from standarderror.ml import decide
    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y
    makers = decide._models()
    rows = {k: [] for k in makers}
    for s in range(reps):
        i = np.random.default_rng(s).permutation(len(y))
        tr, te = i[:300], i[300:]
        for k, M in makers.items():
            p = M().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            rows[k].append((roc_auc_score(y[te], p),
                            decide._cost(p, y[te], 0.5, ratio),
                            decide._cost(p, y[te], 1 / (1 + ratio), ratio),
                            decide._best_cost(p, y[te], ratio),
                            float(np.mean(y[te] == 0))))
    A = {k: np.array(v) for k, v in rows.items()}
    names = list(A)
    auc_win = [names[int(np.argmax([A[k][s, 0] for k in names]))]
               for s in range(reps)]
    cost_win = [names[int(np.argmin([A[k][s, 2] for k in names]))]
                for s in range(reps)]
    from collections import Counter
    return {"ratio": ratio, "reps": reps,
            "models": {k: {"auc": float(a[:, 0].mean()),
                           "cost_half": float(a[:, 1].mean()),
                           "cost_ratio": float(a[:, 2].mean()),
                           "cost_best": float(a[:, 3].mean())}
                       for k, a in A.items()},
            "flag_all": float(A[names[0]][:, 4].mean()),
            "auc_winners": dict(Counter(auc_win)),
            "cost_winners": dict(Counter(cost_win)),
            "same_winner": int(sum(a == c for a, c in zip(auc_win, cost_win)))}
