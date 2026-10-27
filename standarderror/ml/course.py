r"""Arc XIV: the course, measured. The series' own verdicts rerun with new
seeds, a pipeline of defensible choices on pure noise, how each of the
series' problems changes with more data, one project run against the
checklist, and the count of what survived.

Episode 66, `rerun_claims`: twelve verdicts that rested on one random draw,
rerun with five more.
Episode 67, `forking_paths`: six choices a careful analyst might make, one
at a time, on labels that are coin flips.
Episode 68, `pitfalls_vs_n`: seven of the series' problems at growing
sample sizes.
Episode 69, `checklist_run`: one breast-cancer project, every check.
Episode 70, `assert_census`: how many claims the series pinned.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

import standarderror as se


def _quiet():
    warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# ML 66: rerun the series
# --------------------------------------------------------------------------

def _c_tree(seed):
    from standarderror.ml import trees
    b = trees.tree_bootstrap(seed=seed)
    return {"value": b["root_same"],
            "held": b["root_same"] <= 0.55 and b["full_set_same"] == 0
            and b["distinct_sets"] > 0.9 * b["boots"]}


def _c_shap(seed):
    from standarderror.ml import reading
    c = reading.background_study(seed=seed)["compare"]
    v = c["benign only"]["top_differs"]
    return {"value": v, "held": v > 0.5 and c["malignant only"]["top_differs"] > 0.2}


def _c_pca(seed):
    from standarderror.ml import around
    a = around.pca_simulation(seed=seed)["acc"]
    return {"value": a[9], "held": max(a[:9]) < 0.55 and a[9] > 0.9}


def _c_noise(seed):
    from standarderror.ml import around
    r = around.neighbours_noise(extras=(0, 1024), seed=seed)
    d_knn = r[0]["knn"] - r[1]["knn"]
    d_log = r[0]["logistic"] - r[1]["logistic"]
    return {"value": r[1]["knn"],
            "held": d_knn > 2.5 * d_log and r[1]["knn"] < 0.7}


def _c_encoding(seed):
    from standarderror.ml import rows
    c = rows.encoded_column_auc(seed=seed)
    return {"value": c["mean_train"],
            "held": c["mean_train"] > 0.77 and abs(c["mean_test"] - 0.5) < 0.02}


def _c_intervention(seed):
    from standarderror.ml import decide
    i = decide.importance_vs_effect(seed=seed)
    tr = i["true"][i["names"][1]]
    return {"value": i["without_x3_x2"] / tr,
            "held": i["without_x3_x2"] > 1.6 * tr
            and abs(i["adjusted_x2"] - tr) < 0.006}


def _c_bic(seed):
    from standarderror.ml import unsup
    s = unsup.bic_squares(seed=seed)
    return {"value": s["bic"], "held": s["bic"] >= 8 and s["icl"] == 3}


def _c_moons(seed):
    from standarderror.ml import unsup
    m = unsup.dbscan_moons(seed=seed)
    return {"value": m["knee_score"],
            "held": m["knee"] < m["window"][0] and m["knee_score"] < 0.6}


def _c_leak(seed):
    from standarderror.ml import leakage
    t = {r["leak"]: r for r in leakage.leak_table(seed=seed)}
    std = next(v for k, v in t.items() if k.startswith("standardise"))
    sel = next(v for k, v in t.items() if k.startswith("select"))
    return {"value": sel["gap"],
            "held": abs(std["gap"]) < 0.003 and abs(sel["gap"]) < 0.003}


def _c_eights(seed):
    from standarderror.ml import imbalance
    s = imbalance.digits_eights(seed=seed)["summary"]
    return {"value": max(v["auc"] - s["plain"]["auc"] for k, v in s.items()
                         if k != "plain"),
            "held": all(v["auc"] <= s["plain"]["auc"] for v in s.values())}


def _c_tsne(seed):
    from standarderror.ml import unsup
    r = unsup.tsne_sizes(perplexities=(30,), seed=seed)[0]
    return {"value": r["embedded_ratio"],
            "held": 0.95 < r["embedded_ratio"] < 1.05}


def _c_masking(seed):
    from standarderror.ml import unsup
    m = unsup.anomaly_masking(seed=seed)
    small = [r for r in m["rows"] if r["k"] < m["group"]]
    return {"value": max(r["knn"] for r in small),
            "held": all(r["knn"] < 0.2 for r in small)}


#: (episode, the verdict as the episode pinned it, what `value` is, check)
CLAIMS = [
    (3, "selecting 5 of 1,030 features on all rows leaks nothing on "
        "breast cancer", "leak, accuracy", _c_leak),
    (5, "no resampling method raised the AUC on the digit-8 task",
     "largest AUC gain", _c_eights),
    (11, "the full-data root was the root in at most 55% of 200 bootstrap "
         "trees", "share with the same root", _c_tree),
    (18, "a benign-only background changed the top feature for most "
         "tumours", "share changed", _c_shap),
    (31, "only the tenth, smallest principal component predicts",
     "accuracy on the 10th", _c_pca),
    (33, "1,024 noise columns hurt 5-NN over 2.5 times as much as logistic "
         "regression", "5-NN accuracy", _c_noise),
    (40, "a training-fitted encoding ranks its own labels and not fresh ones",
     "AUC on its own labels", _c_encoding),
    (48, "dropping the test result inflates the treatment's apparent effect "
         "over 1.6-fold", "inflation", _c_intervention),
    (56, "t-SNE draws a ten-times-wider cluster the same size",
     "size ratio drawn", _c_tsne),
    (57, "the BIC splits three squares into 8 or more, ICL finds 3",
     "BIC's k", _c_bic),
    (58, "the k-distance knee falls below the half-moons' good radii",
     "knee's score", _c_moons),
    (59, "grouped anomalies score below 0.2 AUC on k-NN distance with k "
         "under the group size", "worst AUC", _c_masking),
]


def rerun_claims(seeds=(0, 1, 2, 3, 4, 5), claims=None) -> list[dict]:
    """Each claim's function with the episode's seed (0) and five more.
    Every function here took one seed in the original episode; several
    replace the data set itself, others only the split or the model's
    randomness."""
    _quiet()
    out = []
    for ep, verdict, what, f in (claims or CLAIMS):
        runs = [dict(seed=s, **f(s)) for s in seeds]
        out.append({"episode": ep, "verdict": verdict, "value_is": what,
                    "runs": runs,
                    "held": int(sum(r["held"] for r in runs[1:])),
                    "original_held": bool(runs[0]["held"])})
    return out


# --------------------------------------------------------------------------
# ML 67: the garden of forking paths
# --------------------------------------------------------------------------

STEPS = ["honest plan", "select features on all rows",
         "best of four models", "best of ten fold splits",
         "best of three feature counts", "best of two subgroups"]


def _models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    return {"logistic": lambda: LogisticRegression(max_iter=2000),
            "SVM": lambda: SVC(),
            "15-NN": lambda: KNeighborsClassifier(15),
            "forest": lambda: RandomForestClassifier(100, random_state=0,
                                                     n_jobs=1)}


def _cv(X, y, model, k, *, leak: bool, split_seed: int):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedKFold(5, shuffle=True, random_state=split_seed)
    if leak:
        cols = SelectKBest(f_classif, k=k).fit(X, y).get_support()
        pipe = make_pipeline(StandardScaler(), _models()[model]())
        return float(cross_val_score(pipe, X[:, cols], y, cv=cv).mean()), cols
    pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k),
                         _models()[model]())
    return float(cross_val_score(pipe, X, y, cv=cv).mean()), None


def _forked(X, y, Xf, yf, sub, subf):
    """Walk the six steps on one data set. Returns the reported score after
    each step and the accuracy of the pipeline the walk ends with on fresh
    rows."""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rep = []
    s, _ = _cv(X, y, "logistic", 20, leak=False, split_seed=0)
    rep.append(s)
    s, _ = _cv(X, y, "logistic", 20, leak=True, split_seed=0)
    rep.append(s)
    scores = {m: _cv(X, y, m, 20, leak=True, split_seed=0)[0]
              for m in _models()}
    model = max(scores, key=scores.get)
    rep.append(scores[model])
    seeds = {sd: _cv(X, y, model, 20, leak=True, split_seed=sd)[0]
             for sd in range(10)}
    sd = max(seeds, key=seeds.get)
    rep.append(seeds[sd])
    ks = {k: _cv(X, y, model, k, leak=True, split_seed=sd)[0]
          for k in (10, 20, 50)}
    k = max(ks, key=ks.get)
    rep.append(ks[k])
    groups = {}
    for gval in (0, 1):
        m = sub == gval
        groups[gval] = _cv(X[m], y[m], model, k, leak=True, split_seed=sd)[0]
    gbest = max(groups, key=groups.get)
    rep.append(groups[gbest])
    m = sub == gbest
    cols = SelectKBest(f_classif, k=k).fit(X[m], y[m]).get_support()
    pipe = make_pipeline(StandardScaler(), _models()[model]()).fit(
        X[m][:, cols], y[m])
    mf = subf == gbest
    fresh = float(np.mean(pipe.predict(Xf[mf][:, cols]) == yf[mf]))
    honest = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20),
                           _models()["logistic"]()).fit(X, y)
    # Scored on the same fresh rows as the walk's pipeline: its subgroup.
    honest_fresh = float(np.mean(honest.predict(Xf[mf]) == yf[mf]))
    return {"reported": rep, "fresh": fresh, "honest_fresh": honest_fresh,
            "choices": {"model": model, "split": sd, "k": k,
                        "subgroup": int(gbest)}}


def forking_paths(*, reps: int = 20, n: int = 200, p: int = 1000) -> dict:
    """Coin-flip labels, `n` rows, `p` noise features; the subgroup is the
    sign of one more noise column. Each walk is scored against 5,000 fresh
    rows of the same noise."""
    _quiet()
    runs = []
    for r in range(reps):
        g = np.random.default_rng(r)
        X, y = g.normal(size=(n, p)), g.integers(0, 2, n)
        Xf, yf = g.normal(size=(5000, p)), g.integers(0, 2, 5000)
        sub, subf = (g.normal(size=n) > 0).astype(int), (
            g.normal(size=5000) > 0).astype(int)
        runs.append(_forked(X, y, Xf, yf, sub, subf))
    rep = np.array([r["reported"] for r in runs])
    return {"steps": STEPS, "reported": rep.mean(0).tolist(),
            "reported_min": rep.min(0).tolist(),
            "reported_max": rep.max(0).tolist(),
            "fresh": float(np.mean([r["fresh"] for r in runs])),
            "honest_fresh": float(np.mean([r["honest_fresh"] for r in runs])),
            "runs": runs}


def forking_real(*, reps: int = 20) -> dict:
    """The same walk on breast cancer (300 tumours to analyse, 269 fresh),
    where the signal is real; the subgroup is above or below the median
    tumour texture."""
    _quiet()
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    sub_all = (X[:, 1] > np.median(X[:, 1])).astype(int)
    runs = []
    for r in range(reps):
        i = np.random.default_rng(r).permutation(len(y))
        a, b = i[:300], i[300:]
        runs.append(_forked(X[a], y[a], X[b], y[b], sub_all[a], sub_all[b]))
    rep = np.array([r["reported"] for r in runs])
    return {"steps": STEPS, "reported": rep.mean(0).tolist(),
            "fresh": float(np.mean([r["fresh"] for r in runs])),
            "honest_fresh": float(np.mean([r["honest_fresh"] for r in runs]))}


# --------------------------------------------------------------------------
# ML 68: which problems more data fixes
# --------------------------------------------------------------------------

SIZES = (200, 800, 3200)


def _tuning_optimism(n, reps, seed=0):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier

    from standarderror.ml.curves import GaussianTask
    t = GaussianTask(test=20000)
    gaps = []
    for r in range(reps):
        X, y = t.draw(n, seed=seed + 1000 + r)
        G = GridSearchCV(KNeighborsClassifier(),
                         {"n_neighbors": list(range(1, 64, 2))},
                         cv=StratifiedKFold(5, shuffle=True, random_state=r)
                         ).fit(X, y)
        gaps.append(G.best_score_ - float(np.mean(G.predict(t.X_test)
                                                  == t.y_test)))
    return float(np.mean(gaps))


def _selection_leak(n, reps, p=1000, k=20):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    out = []
    for r in range(reps):
        g = np.random.default_rng(r)
        X, y = g.normal(size=(n, p)), g.integers(0, 2, n)
        cols = SelectKBest(f_classif, k=k).fit(X, y).get_support()
        s = cross_val_score(LogisticRegression(max_iter=2000), X[:, cols], y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=r))
        out.append(float(s.mean()) - 0.5)
    return float(np.mean(out))


def _encoding_leak(n, reps, levels=200):
    from sklearn.metrics import roc_auc_score
    out = []
    for r in range(reps):
        g = np.random.default_rng(r)
        c, y = g.integers(0, levels, n), g.integers(0, 2, n)
        means = np.array([y[c == k].mean() if np.any(c == k) else y.mean()
                          for k in range(levels)])
        out.append(float(roc_auc_score(y, means[c])) - 0.5)
    return float(np.mean(out))


def _selection_bias(n, reps):
    """Episode 39's officer world, compactly: approval uses a signal the
    model lacks; the model's predicted default among the refused, minus the
    truth."""
    from sklearn.linear_model import LogisticRegression
    out = []
    for r in range(reps):
        g = np.random.default_rng(r)
        x, u = g.normal(size=n), g.normal(size=n)
        y = (g.random(n) < 1 / (1 + np.exp(-(x + 1.5 * u - 1)))).astype(int)
        ok = (x + 1.5 * u) < 0.5
        m = LogisticRegression().fit(x[ok, None], y[ok])
        out.append(float(m.predict_proba(x[~ok, None])[:, 1].mean()
                         - y[~ok].mean()))
    return float(np.mean(out))


def _subject_optimism(n, reps, subjects=20):
    """Episode 37's world: `n` rows from 20 people; row-level 5-fold CV of
    5-nearest neighbours, minus its accuracy on 4,000 rows from 200 new
    people."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    from standarderror.ml import rows
    out = []
    for r in range(reps):
        g = np.random.default_rng(r)
        X, y, _ = rows._subjects(subjects, n // subjects, g)
        Xn, yn, _ = rows._subjects(200, 20, g)
        m = KNeighborsClassifier(5)
        cv = cross_val_score(m, X, y, cv=StratifiedKFold(
            5, shuffle=True, random_state=r)).mean()
        out.append(float(cv - np.mean(m.fit(X, y).predict(Xn) == yn)))
    return float(np.mean(out))


def _bic_components(n, seed=0):
    from standarderror.ml import unsup
    X = np.random.default_rng(seed).random((n, 2))
    b, _ = unsup._picks(unsup._bic_icl(X, range(1, 16), n_init=2))
    return int(b)


PITFALLS = {
    "test-score standard error (accuracy 0.8)":
        ("episode 1", "points of accuracy"),
    "tuning optimism, best of 32 k-NN settings":
        ("episode 35", "points of accuracy"),
    "feature selection on all rows, pure noise":
        ("episode 3", "points of accuracy above 0.5"),
    "target encoding on its own rows, 200 noise levels":
        ("episode 40", "points of AUC above 0.5"),
    "model trained on approved applicants only":
        ("episode 39", "points of default rate"),
    "row-level CV, rows from 20 people":
        ("episode 37", "points of accuracy"),
    "BIC components for one uniform square":
        ("episode 57", "components"),
}


def pitfalls_vs_n(sizes=SIZES, *, reps: int = 10) -> dict:
    """Seven of the series' problems at each sample size. Each is the size
    of the error it causes, in its own unit; the last column says whether it
    shrank, stayed or grew from the smallest to the largest size."""
    _quiet()
    rows = {}
    for name in PITFALLS:
        vals = []
        for n in sizes:
            if name.startswith("test-score"):
                v = 100 * float(np.sqrt(0.8 * 0.2 / n))
            elif name.startswith("tuning"):
                v = 100 * _tuning_optimism(n, reps)
            elif name.startswith("feature selection"):
                v = 100 * _selection_leak(n, reps)
            elif name.startswith("target encoding"):
                v = 100 * _encoding_leak(n, reps)
            elif name.startswith("model trained"):
                v = 100 * _selection_bias(n, reps)
            elif name.startswith("row-level"):
                v = 100 * _subject_optimism(n, reps)
            else:
                v = float(_bic_components(n))
            vals.append(v)
        rows[name] = {"episode": PITFALLS[name][0], "unit": PITFALLS[name][1],
                      "values": vals}
    return {"sizes": list(sizes), "rows": rows}


# --------------------------------------------------------------------------
# ML 69: the checklist, run once
# --------------------------------------------------------------------------

def checklist_run(*, seed: int = 0, ratio: float = 10.0) -> dict:
    """One project on breast cancer, malignant as the positive class: every
    check the series recommends, in the order a project meets them."""
    _quiet()
    from scipy.stats import binomtest, ks_2samp
    from scipy.stats import t as tdist
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        GridSearchCV,
        StratifiedKFold,
        cross_val_score,
        train_test_split,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from standarderror.ml import decide
    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y
    Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, stratify=y,
                                      random_state=seed)
    out = {"n_train": int(len(ya)), "n_test": int(len(yt))}

    def logistic():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

    def boosted():
        return HistGradientBoostingClassifier(max_iter=200, random_state=0)

    # 1-2. Baselines and candidates by cross-validation on the training rows.
    cv = StratifiedKFold(10, shuffle=True, random_state=seed)
    out["majority"] = float(max(yt.mean(), 1 - yt.mean()))
    cvs = {k: cross_val_score(f(), Xa, ya, cv=cv) for k, f in
           (("logistic", logistic), ("boosted trees", boosted))}
    out["cv"] = {k: float(v.mean()) for k, v in cvs.items()}
    # Corrected resampled t-test (Nadeau and Bengio) on the ten folds.
    d = cvs["logistic"] - cvs["boosted trees"]
    k = len(d)
    corr = np.sqrt((1 / k + 1 / (k - 1)) * d.var(ddof=1))
    naive = np.sqrt(d.var(ddof=1) / k)
    out["cv_diff"] = float(d.mean())
    out["p_naive"] = float(2 * tdist.sf(abs(d.mean()) / naive, k - 1)) if naive else 1.0
    out["p_corrected"] = float(2 * tdist.sf(abs(d.mean()) / corr, k - 1)) if corr else 1.0
    # 3. Tuning: the best of 64 SVM settings, plain and nested.
    from standarderror.ml import around
    grid = {"svc__" + a: b for a, b in around._grid(64).items()}
    G = GridSearchCV(make_pipeline(StandardScaler(), SVC()), grid,
                     cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    G.fit(Xa, ya)
    out["svm_best_cv"] = float(G.best_score_)
    out["svm_nested"] = float(cross_val_score(
        GridSearchCV(make_pipeline(StandardScaler(), SVC()), grid,
                     cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
        Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=seed + 1)
    ).mean())
    out["svm_test"] = float(np.mean(G.predict(Xt) == yt))
    # 4. The test set, once: accuracy with an interval; a paired comparison.
    ml = logistic().fit(Xa, ya)
    mb = boosted().fit(Xa, ya)
    rl, rb = ml.predict(Xt) == yt, mb.predict(Xt) == yt
    n = len(yt)
    for name, r in (("logistic", rl), ("boosted trees", rb)):
        acc = float(r.mean())
        se = np.sqrt(acc * (1 - acc) / n)
        out[f"test_{name}"] = {"acc": acc, "lo": acc - 1.96 * se,
                               "hi": acc + 1.96 * se}
    only_l, only_b = int(np.sum(rl & ~rb)), int(np.sum(rb & ~rl))
    out["discordant"] = (only_l, only_b)
    out["mcnemar_p"] = float(binomtest(only_l, only_l + only_b).pvalue) \
        if only_l + only_b else 1.0
    # 5. Calibration, raw and recalibrated boosted trees, and logistic.
    pl = ml.predict_proba(Xt)[:, 1]
    pb = mb.predict_proba(Xt)[:, 1]
    mc = CalibratedClassifierCV(boosted(), method="sigmoid", cv=5).fit(Xa, ya)
    pc = mc.predict_proba(Xt)[:, 1]

    def brier(p):
        return float(np.mean((p - yt) ** 2))
    out["brier"] = {"logistic": brier(pl), "boosted trees": brier(pb),
                    "recalibrated boosted": brier(pc)}
    out["extreme"] = {"logistic": float(np.mean((pl < 0.01) | (pl > 0.99))),
                      "boosted trees": float(np.mean((pb < 0.01) | (pb > 0.99)))}
    # 6. Threshold from the cost of a miss.
    thr = 1 / (1 + ratio)
    out["cost"] = {name: {"at_half": decide._cost(p, yt, 0.5, ratio),
                          "at_ratio": decide._cost(p, yt, thr, ratio),
                          "best": decide._best_cost(p, yt, ratio)}
                   for name, p in (("logistic", pl), ("boosted trees", pb),
                                   ("recalibrated boosted", pc))}
    out["counts"] = {name: {f"{k}_{t}": int(np.sum(
        ((p >= th) if k == "false" else (p < th)) & (yt == (k == "missed"))))
        for k in ("missed", "false") for t, th in (("half", 0.5), ("ratio", thr))}
        for name, p in (("logistic", pl), ("boosted trees", pb),
                        ("recalibrated boosted", pc))}
    out["flag_all"] = float(np.mean(yt == 0))
    out["missed_at_ratio"] = int(np.sum((pl < thr) & (yt == 1)))
    out["missed_at_half"] = int(np.sum((pl < 0.5) & (yt == 1)))
    # 7. Subgroups: by tumour size thirds on the test set.
    cuts = np.quantile(Xa[:, 0], [1 / 3, 2 / 3])
    third = np.digitize(Xt[:, 0], cuts)
    flag = pl >= thr
    out["groups"] = []
    for g in range(3):
        s = third == g
        out["groups"].append({
            "rows": int(s.sum()), "malignant": int(yt[s].sum()),
            "missed": int(np.sum(~flag[s] & (yt[s] == 1))),
            "false_alarms": int(np.sum(flag[s] & (yt[s] == 0))),
            "mean_p": float(pl[s].mean()), "rate": float(yt[s].mean())})
    # 8. Stability: the same logistic pipeline on 50 bootstrap resamples.
    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(50):
        i = rng.integers(0, len(ya), len(ya))
        preds.append(logistic().fit(Xa[i], ya[i]).predict_proba(Xt)[:, 1] >= thr)
    preds = np.array(preds)
    out["flips"] = float(np.mean(preds.min(0) != preds.max(0)))
    out["flip_rows"] = int(np.sum(preds.min(0) != preds.max(0)))
    # 9. A shift check: the test tumours above the training median size.
    big = Xt[:, 0] > np.median(Xa[:, 0])
    p_vals = [ks_2samp(Xa[:, j], Xt[big][:, j]).pvalue for j in range(30)]
    out["shift_alarms"] = int(sum(v < 0.05 / 30 for v in p_vals))
    out["shift_cost"] = decide._cost(pl[big], yt[big], thr, ratio)
    out["shift_rows"] = int(big.sum())
    out["shift_missed"] = int(np.sum((pl[big] < thr) & (yt[big] == 1)))
    out["shift_false"] = int(np.sum((pl[big] >= thr) & (yt[big] == 0)))
    out["shift_malignant"] = int(yt[big].sum())
    return out


# --------------------------------------------------------------------------
# ML 70: the census
# --------------------------------------------------------------------------

def assert_census() -> dict:
    """Count the pinned claims: assert statements in every Machine Learning
    episode's build, and the episodes themselves."""
    root = Path(se.SETTINGS.repo_root) / "experiments"
    per = {}
    for f in sorted(root.glob("ml1[0-9][0-9]_*.py")):
        ep = int(f.name[2:5]) - 100
        text = f.read_text()
        per[ep] = len(re.findall(r"^\s+assert ", text, flags=re.M))
    return {"episodes": len(per), "asserts": int(sum(per.values())),
            "per_episode": per,
            "median": float(np.median(list(per.values()))),
            "most": max(per.items(), key=lambda kv: kv[1])}
