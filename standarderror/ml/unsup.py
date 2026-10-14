r"""Arc XII: without labels. Embeddings, mixtures, density clustering, anomaly
scores and the choice of a number of clusters, each measured on simulations
with a known answer and replicated on scikit-learn's bundled data.

Episode 56, `tsne_*`: what a t-SNE picture keeps and what it does not.
Episode 57, `bic_*` and `em_starts`: what the BIC counts in a Gaussian
mixture, and how much the initialisation does.
Episode 58, `dbscan_*`: DBSCAN's radius, one density at a time.
Episode 59, `anomaly_*`: what an anomaly detector's settings and
neighbourhoods decide.
Episode 60, `stability_*` and `gap_*`: choosing the number of clusters by
resampling and by a reference distribution.

Arc VII already measured PCA (episode 31), k-means and the silhouette (32)
and nearest neighbours in many dimensions (33); this arc does not repeat them.
"""

from __future__ import annotations

import warnings

import numpy as np


def _quiet():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


def noise_ari(y, labels) -> float:
    """Adjusted Rand index with every noise point (-1) its own singleton.

    Scoring DBSCAN's noise label as one more cluster would reward an answer
    that calls a whole cluster noise; as singletons, it costs what it should.
    """
    from sklearn.metrics import adjusted_rand_score
    lab = np.asarray(labels).copy()
    noise = lab == -1
    lab[noise] = np.arange(noise.sum()) + (lab.max() + 1 if len(lab) else 0)
    return float(adjusted_rand_score(y, lab))


# --------------------------------------------------------------------------
# ML 56: t-SNE
# --------------------------------------------------------------------------

def _tsne(X, perplexity, seed=0, init="pca"):
    from sklearn.manifold import TSNE
    return TSNE(perplexity=perplexity, random_state=seed,
                init=init).fit_transform(X)


def _radius(Y):
    return float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))


def tsne_sizes(perplexities=(5, 30, 100), *, n: int = 500, dim: int = 10,
               spread: float = 10.0, seed: int = 0) -> list[dict]:
    """Two well-separated Gaussian clusters, one `spread` times wider.

    Returns the ratio of their root-mean-square radii in the data and in the
    embedding, per perplexity.
    """
    _quiet()
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, dim))
    b = rng.normal(size=(n, dim)) * spread
    b[:, 0] += 20 * spread
    X = np.r_[a, b]
    from sklearn.decomposition import PCA
    P = PCA(2, random_state=0).fit_transform(X)
    out = []
    for p in perplexities:
        Y = _tsne(X, p)
        out.append({"perplexity": p,
                    "pca_ratio": _radius(P[n:]) / _radius(P[:n]),
                    "data_ratio": _radius(X[n:]) / _radius(X[:n]),
                    "embedded_ratio": _radius(Y[n:]) / _radius(Y[:n]),
                    "X": X[:, :2], "Y": Y})
    return out


def tsne_distances(perplexities=(5, 30, 100), *, n: int = 300, dim: int = 10,
                   near: float = 6.0, far: float = 60.0,
                   seed: int = 0) -> list[dict]:
    """Three unit-variance clusters on a line: B is `near` from A, C is
    `far` from B. Ratio of the B-C to the A-B centre distance, data against
    embedding."""
    _quiet()
    rng = np.random.default_rng(seed)
    centres = [0.0, near, near + far]
    parts = []
    for c in centres:
        x = rng.normal(size=(n, dim))
        x[:, 0] += c
        parts.append(x)
    X = np.concatenate(parts)
    from sklearn.decomposition import PCA
    P = PCA(2, random_state=0).fit_transform(X)
    mp = [P[i * n:(i + 1) * n].mean(0) for i in range(3)]
    pca_ratio = float(np.linalg.norm(mp[2] - mp[1])
                      / np.linalg.norm(mp[1] - mp[0]))
    out = []
    for p in perplexities:
        Y = _tsne(X, p)
        m = [Y[i * n:(i + 1) * n].mean(0) for i in range(3)]
        mx = [X[i * n:(i + 1) * n].mean(0) for i in range(3)]
        out.append({"perplexity": p, "X": X[:, :2], "Y": Y,
                    "pca_ratio": pca_ratio,
                    "data_ratio": float(np.linalg.norm(mx[2] - mx[1])
                                        / np.linalg.norm(mx[1] - mx[0])),
                    "embedded_ratio": float(np.linalg.norm(m[2] - m[1])
                                            / np.linalg.norm(m[1] - m[0]))})
    return out


def _knn_kept(X, Y, k=10) -> float:
    from sklearn.neighbors import NearestNeighbors
    a = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
        X, return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(
        Y, return_distance=False)[:, 1:]
    return float(np.mean([len(set(i) & set(j)) / k for i, j in zip(a, b)]))


def tsne_digits(seeds=(0, 1, 2, 3, 4), perplexity: int = 30) -> dict:
    """t-SNE of the 1,797 digits against a two-component PCA.

    Local: the share of each point's ten nearest neighbours that are still
    among its ten nearest in the picture. Global: Spearman correlation between
    the 45 distances between digit-class centroids in the data and in the
    picture. Also: HDBSCAN run on the picture, against the classes, and the
    same global correlation between pictures made with different seeds,
    for random initialisation and for PCA initialisation. HDBSCAN's ARI here
    counts noise as one more cluster, scikit-learn's default.
    """
    _quiet()
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    X, y = load_digits(return_X_y=True)

    def cent(Z):
        return np.array([Z[y == c].mean(0) for c in range(10)])

    dx = pdist(cent(X))
    P = PCA(2, random_state=0).fit_transform(X)
    out = {"pca": {"knn": _knn_kept(X, P),
                   "global": float(spearmanr(dx, pdist(cent(P)))[0])},
           "tsne": []}
    pics = []
    for s in seeds:
        Y = _tsne(X, perplexity, seed=s, init="random")
        pics.append(Y)
        lab = HDBSCAN(min_cluster_size=25, copy=True).fit_predict(Y)
        out["tsne"].append({
            "seed": s, "knn": _knn_kept(X, Y),
            "global": float(spearmanr(dx, pdist(cent(Y)))[0]),
            "hdbscan_ari": float(adjusted_rand_score(y, lab)),
            "hdbscan_clusters": int(len(set(lab) - {-1})),
            "hdbscan_noise": float(np.mean(lab == -1))})
    out["between_seeds"] = [float(spearmanr(pdist(cent(pics[i])),
                                            pdist(cent(pics[j])))[0])
                            for i in range(len(pics))
                            for j in range(i + 1, len(pics))]
    # The same five seeds with PCA initialisation, scikit-learn's default and
    # Kobak and Berens' recommendation.
    pca_pics = [_tsne(X, perplexity, seed=s, init="pca") for s in seeds]
    out["pca_init"] = [{"seed": s, "knn": _knn_kept(X, Y),
                        "global": float(spearmanr(dx, pdist(cent(Y)))[0])}
                       for s, Y in zip(seeds, pca_pics)]
    out["pca_init_between"] = [float(spearmanr(pdist(cent(pca_pics[i])),
                                               pdist(cent(pca_pics[j])))[0])
                               for i in range(len(pca_pics))
                               for j in range(i + 1, len(pca_pics))]
    lab = HDBSCAN(min_cluster_size=25, copy=True).fit_predict(X)
    out["hdbscan_raw_ari"] = float(adjusted_rand_score(y, lab))
    out["hdbscan_raw_noise"] = float(np.mean(lab == -1))
    out["kmeans_raw_ari"] = float(adjusted_rand_score(
        y, KMeans(10, n_init=10, random_state=0).fit_predict(X)))
    return out


# --------------------------------------------------------------------------
# ML 57: Gaussian mixtures and the BIC
# --------------------------------------------------------------------------

def _bic_icl(X, ks, *, covariance_type="full", n_init=3, seed=0):
    """BIC and ICL for each k. ICL is the BIC plus twice the entropy of the
    soft assignments, the ICL-BIC form of McLachlan and Peel's book; the
    original of Biernacki, Celeux and Govaert (2000) uses the hard
    assignments. It charges a model for components that overlap, which is
    what a cluster is not."""
    from sklearn.mixture import GaussianMixture
    rows = []
    for k in ks:
        g = GaussianMixture(k, covariance_type=covariance_type,
                            n_init=n_init, random_state=seed).fit(X)
        tau = g.predict_proba(X)
        ent = float(-(tau * np.log(np.clip(tau, 1e-300, 1))).sum())
        rows.append({"k": k, "bic": float(g.bic(X)),
                     "icl": float(g.bic(X) + 2 * ent), "model": g})
    return rows


def _picks(rows):
    return (min(rows, key=lambda r: r["bic"])["k"],
            min(rows, key=lambda r: r["icl"])["k"])


def bic_growth(sizes=(200, 1000, 5000, 20000), *, kmax: int = 20,
               seed: int = 0) -> dict:
    """One cluster, three shapes, growing n: how many components BIC and ICL
    choose. A Gaussian is the control; a uniform square and a Student t with
    three degrees of freedom are single clusters that are not Gaussian."""
    _quiet()
    rng = np.random.default_rng(seed)
    out = {}
    for shape in ("gaussian", "uniform", "t3"):
        out[shape] = []
        for n in sizes:
            if shape == "gaussian":
                X = rng.normal(size=(n, 2))
            elif shape == "uniform":
                X = rng.random((n, 2))
            else:
                X = rng.standard_t(3, size=(n, 2))
            rows = _bic_icl(X, range(1, kmax + 1))
            b, i = _picks(rows)
            srt = sorted(r["bic"] for r in rows)
            out[shape].append({"n": n, "bic": b, "icl": i,
                               "margin": rows[0]["bic"] - srt[0],
                               "runner_up_gap": srt[1] - srt[0]})
    return out


def bic_squares(*, n: int = 500, kmax: int = 10, seed: int = 0) -> dict:
    """Three separated uniform squares: three clusters, no Gaussians."""
    _quiet()
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    X = np.r_[rng.random((n, 2)), rng.random((n, 2)) + [2, 0],
              rng.random((n, 2)) + [1, 2]]
    y = np.repeat([0, 1, 2], n)
    rows = _bic_icl(X, range(1, kmax + 1), n_init=5)
    b, i = _picks(rows)
    best = next(r for r in rows if r["k"] == b)["model"]
    return {"bic": b, "icl": i, "rows": [{k: v for k, v in r.items()
                                          if k != "model"} for r in rows],
            "bic_model_ari": float(adjusted_rand_score(y, best.predict(X)))}


def bic_digits(ks=tuple(range(2, 31)), *, dims: int = 10) -> dict:
    """The digits, reduced to `dims` principal components, with full and
    with diagonal covariances: the BIC's k and that model's agreement with the
    ten classes."""
    _quiet()
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    X, y = load_digits(return_X_y=True)
    Z = PCA(dims, random_state=0).fit_transform(X)
    out = {}
    for ct in ("full", "diag"):
        rows = _bic_icl(Z, ks, covariance_type=ct, n_init=2)
        b, i = _picks(rows)
        ari = {r["k"]: float(adjusted_rand_score(y, r["model"].predict(Z)))
               for r in rows}
        out[ct] = {"bic": b, "icl": i, "ari_at_bic": ari[b],
                   "ari_at_10": ari.get(10)}
    return out


def em_starts(*, k: int = 15, n: int = 100, runs: int = 60,
              seed: int = 0) -> dict:
    """Fifteen overlapping Gaussian clusters; EM from different starts.

    For each initialisation scikit-learn offers, `runs` single fits (its
    default is one start): mean log-likelihood per point and adjusted Rand
    index. The reference is EM started at the true centres.
    """
    _quiet()
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-20, 20, (k, 2))
    y = np.repeat(range(k), n)
    X = centres[y] + rng.normal(size=(k * n, 2))
    ref = GaussianMixture(k, means_init=centres, random_state=0).fit(X)
    out = {"truth_start": {"ll": float(ref.score(X)),
                           "ari": float(adjusted_rand_score(
                               y, ref.predict(X)))}}
    for init in ("kmeans", "k-means++", "random_from_data", "random"):
        ll, ari = [], []
        for s in range(runs):
            g = GaussianMixture(k, init_params=init, random_state=s,
                                max_iter=500).fit(X)
            ll.append(g.score(X))
            ari.append(adjusted_rand_score(y, g.predict(X)))
        ll, ari = np.array(ll), np.array(ari)
        out[init] = {"reached": float(np.mean(ll > ref.score(X) - 0.005)),
                     "ari_median": float(np.median(ari)),
                     "ari_min": float(ari.min()),
                     "ll_median": float(np.median(ll))}
    return out


# --------------------------------------------------------------------------
# ML 58: DBSCAN
# --------------------------------------------------------------------------

EPS_GRID = np.geomspace(0.02, 3.0, 150)


def _sweep(X, y, grid, min_samples=5):
    from sklearn.cluster import DBSCAN
    rows = []
    for e in grid:
        lab = DBSCAN(eps=e, min_samples=min_samples).fit_predict(X)
        rows.append({"eps": float(e), "score": noise_ari(y, lab),
                     "noise": float(np.mean(lab == -1)),
                     "clusters": int(len(set(lab) - {-1})), "labels": lab})
    return rows


def knee_eps(X, k: int = 4) -> float:
    """The sorted k-distance plot's knee, the radius Ester et al. suggested
    reading off by eye; found here as the point farthest below the chord."""
    from sklearn.neighbors import NearestNeighbors
    d = np.sort(NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0]
                [:, -1])
    x = np.linspace(0, 1, len(d))
    yv = (d - d.min()) / (d.max() - d.min())
    return float(d[np.argmax(x - yv)])


def dbscan_moons(*, n: int = 1000, noise: float = 0.06, seed: int = 0) -> dict:
    """Two interleaved half-moons: the shape k-means cannot do."""
    _quiet()
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.datasets import make_moons
    from sklearn.metrics import adjusted_rand_score
    X, y = make_moons(n, noise=noise, random_state=seed)
    grid = np.geomspace(0.01, 2.0, 120)
    rows = _sweep(X, y, grid)
    good = [r["eps"] for r in rows if r["score"] >= 0.9]
    perfect = [r["eps"] for r in rows if r["score"] >= 0.9999]
    ke = knee_eps(X)
    windows = {}
    for ms in (3, 10, 20, 40):
        g = [r["eps"] for r in _sweep(X, y, grid, min_samples=ms)
             if r["score"] >= 0.9]
        windows[ms] = (min(g), max(g)) if g else None
    return {"best": max(r["score"] for r in rows),
            "window": (min(good), max(good)) if good else None,
            "window_min_score": min(r["score"] for r in rows
                                    if min(good) <= r["eps"] <= max(good)),
            "perfect": (min(perfect), max(perfect)) if perfect else None,
            "knee": ke,
            "knee_score": noise_ari(y, DBSCAN(eps=ke, min_samples=5)
                                    .fit_predict(X)),
            "kmeans_ari": float(adjusted_rand_score(
                y, KMeans(2, n_init=10, random_state=0).fit_predict(X))),
            "windows": windows,
            "default_score": noise_ari(y, DBSCAN().fit_predict(X)),
            "default_clusters": int(len(set(DBSCAN().fit_predict(X)) - {-1}))}


def dbscan_densities(spreads=(0.2, 0.4, 0.8, 1.2, 1.6), *, n: int = 300,
                     seed: int = 0, reps: int = 5) -> list[dict]:
    """Two tight clusters (sd 0.2, centres 1.4 apart) beside a third whose
    spread grows. For each spread: DBSCAN's best score over the radius grid,
    how many radii score at least 0.9, what the best radius does to the wide
    cluster, and HDBSCAN without a radius."""
    _quiet()
    from sklearn.cluster import HDBSCAN

    def draw(sd, s):
        # One stream per spread and seed, so any subset reproduces the table.
        rng = np.random.default_rng([s, int(round(sd * 100))])
        return np.r_[rng.normal(size=(n, 2)) * 0.2,
                     rng.normal(size=(n, 2)) * 0.2 + [1.4, 0],
                     rng.normal(size=(n, 2)) * sd + [6, 0]]

    y = np.repeat([0, 1, 2], n)
    out = []
    for sd in spreads:
        X = draw(sd, seed)
        rows = _sweep(X, y, EPS_GRID)
        best = max(rows, key=lambda r: r["score"])
        h = HDBSCAN(min_cluster_size=25, copy=True).fit_predict(X)
        others = []
        for s in range(seed + 1, seed + 1 + reps):
            Xs = draw(sd, s)
            rs = _sweep(Xs, y, EPS_GRID)
            others.append((int(sum(r["score"] >= 0.9 for r in rs)),
                           max(r["score"] for r in rs),
                           noise_ari(y, HDBSCAN(min_cluster_size=25, copy=True)
                                     .fit_predict(Xs))))
        out.append({"spread": sd, "ratio": sd / 0.2,
                    "best": best["score"], "best_eps": best["eps"],
                    "wide_noise": float(np.mean(best["labels"][2 * n:] == -1)),
                    "good_radii": int(sum(r["score"] >= 0.9 for r in rows)),
                    "curve": [(r["eps"], r["score"]) for r in rows],
                    "hdbscan": noise_ari(y, h),
                    "hdbscan_noise": float(np.mean(h == -1)),
                    "overall_noise": float(np.mean(best["labels"] == -1)),
                    "reps_good_radii": [o[0] for o in others],
                    "reps_best": [o[1] for o in others],
                    "reps_hdbscan": [o[2] for o in others]})
    return out


def dbscan_digits() -> dict:
    """The digits in their 64 raw dimensions."""
    _quiet()
    from sklearn.cluster import DBSCAN, HDBSCAN
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    rows = _sweep(X, y, np.geomspace(5, 40, 60))
    best = max(rows, key=lambda r: r["score"])
    ke = knee_eps(X)
    kl = DBSCAN(eps=ke, min_samples=5).fit_predict(X)
    h = HDBSCAN(min_cluster_size=25, copy=True).fit_predict(X)
    return {"best": best["score"], "best_eps": best["eps"],
            "best_noise": best["noise"], "best_clusters": best["clusters"],
            "knee": ke, "knee_score": noise_ari(y, kl),
            "knee_clusters": int(len(set(kl) - {-1})),
            "knee_largest": float(np.bincount(kl[kl >= 0]).max() / len(kl)),
            "knee_noise": float(np.mean(kl == -1)),
            "hdbscan": noise_ari(y, h), "hdbscan_noise": float(np.mean(h == -1))}


# --------------------------------------------------------------------------
# ML 59: anomaly detection
# --------------------------------------------------------------------------

def anomaly_contamination(rates=(0.005, 0.02, 0.1),
                          settings=(0.01, 0.05, 0.1, "auto"), *,
                          n: int = 5000, dim: int = 5,
                          seed: int = 0) -> list[dict]:
    """An isolation forest with each `contamination` setting, on data whose
    true anomaly rate varies: share flagged, precision and recall."""
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score
    out = []
    for rate in rates:
        rng = np.random.default_rng([seed, int(round(rate * 10_000))])
        m = int(n * rate)
        Xa = rng.normal(size=(m, dim)) + rng.choice([-1, 1], (m, dim)) * 2.5
        X = np.r_[rng.normal(size=(n - m, dim)), Xa]
        y = np.r_[np.zeros(n - m), np.ones(m)]
        score = -IsolationForest(random_state=0).fit(X).score_samples(X)
        row = {"rate": rate, "auc": float(roc_auc_score(y, score))}
        for c in settings:
            flag = IsolationForest(contamination=c, random_state=0).fit(
                X).predict(X) == -1
            row[str(c)] = {"flagged": float(flag.mean()),
                           "precision": float(y[flag].mean()) if flag.any()
                           else float("nan"),
                           "recall": float(flag[y == 1].mean())}
        out.append(row)
    return out


def anomaly_masking(ks=(5, 20, 50, 100), *, group: int = 50, n: int = 2000,
                    seed: int = 0) -> dict:
    """A tight group of `group` anomalies far from the normal data. Local
    methods compare a point with its k neighbours; when k is smaller than
    the group, the neighbours are the other anomalies."""
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    rng = np.random.default_rng(seed)
    X = np.r_[rng.normal(size=(n, 2)),
              rng.normal(size=(group, 2)) * 0.1 + [5, 5]]
    y = np.r_[np.zeros(n), np.ones(group)]
    rows = []
    for k in ks:
        lof = LocalOutlierFactor(n_neighbors=k).fit(X)
        d = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, -1]
        rows.append({"k": k,
                     "lof": float(roc_auc_score(y, -lof.negative_outlier_factor_)),
                     "knn": float(roc_auc_score(y, d))})
    iso = -IsolationForest(random_state=0).fit(X).score_samples(X)
    return {"group": group, "rows": rows,
            "iforest": float(roc_auc_score(y, iso))}


def anomaly_dimensions(dims=(2, 10, 50, 200), *, n: int = 2000, m: int = 40,
                       shift: float = 3.5, seed: int = 0,
                       seeds: int = 6) -> list[dict]:
    """Anomalies shifted in two features; the rest are irrelevant. Means,
    minima and maxima over `seeds` draws of the data and the forest."""
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import LocalOutlierFactor
    y = np.r_[np.zeros(n), np.ones(m)]
    out = []
    for d in dims:
        res = {"iforest": [], "lof": [], "iforest_two": []}
        for s in range(seed, seed + seeds):
            rng = np.random.default_rng([s, d])
            Xa = rng.normal(size=(m, d))
            Xa[:, :2] += shift
            X = np.r_[rng.normal(size=(n, d)), Xa]
            iso = -IsolationForest(random_state=s).fit(X).score_samples(X)
            lof = -LocalOutlierFactor(n_neighbors=20).fit(
                X).negative_outlier_factor_
            two = -IsolationForest(random_state=s).fit(
                X[:, :2]).score_samples(X[:, :2])
            res["iforest"].append(float(roc_auc_score(y, iso)))
            res["lof"].append(float(roc_auc_score(y, lof)))
            res["iforest_two"].append(float(roc_auc_score(y, two)))
        out.append({"dim": d, **{k: float(np.mean(v)) for k, v in res.items()},
                    **{f"{k}_min": float(min(v)) for k, v in res.items()},
                    **{f"{k}_max": float(max(v)) for k, v in res.items()}})
    return out


def anomaly_breast_cancer(*, planted: int = 8, reps: int = 20) -> dict:
    """Benign tumours as the normal data, `planted` malignant ones hidden
    among them, standardised features; repeated over random choices of the
    planted cases."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    ben, mal = X[y == 1], X[y == 0]
    rows = []
    for s in range(reps):
        idx = np.random.default_rng(s).choice(len(mal), planted, replace=False)
        Z = np.r_[ben, mal[idx]]
        t = np.r_[np.zeros(len(ben)), np.ones(planted)]
        iso = IsolationForest(random_state=s).fit(Z)
        flag = IsolationForest(contamination=0.05, random_state=s).fit(
            Z).predict(Z) == -1
        rows.append({"iforest": float(roc_auc_score(t, -iso.score_samples(Z))),
                     "lof": float(roc_auc_score(
                         t, -LocalOutlierFactor(20).fit(Z).negative_outlier_factor_)),
                     "recall_5pct": float(flag[t == 1].mean()),
                     "precision_5pct": float(t[flag].mean())})
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return {"normal": int(len(ben)), "planted": planted, "reps": reps,
            **{f"{k}_mean": float(v.mean()) for k, v in a.items()},
            **{f"{k}_min": float(v.min()) for k, v in a.items()},
            **{f"{k}_max": float(v.max()) for k, v in a.items()}}


# --------------------------------------------------------------------------
# ML 60: how many clusters
# --------------------------------------------------------------------------

def stability(X, k: int, *, pairs: int = 20, share: float = 0.8,
              seed: int = 0) -> float:
    """Mean agreement (ARI) between k-means fitted on two random 80%
    subsamples, compared on the points they share (Ben-Hur, Elisseeff and
    Guyon, 2002, in outline)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    n = len(X)
    s = []
    for b in range(pairs):
        i = rng.choice(n, int(share * n), replace=False)
        j = rng.choice(n, int(share * n), replace=False)
        common = np.intersect1d(i, j)
        la = KMeans(k, n_init=3, random_state=b).fit(X[i]).predict(X[common])
        lb = KMeans(k, n_init=3, random_state=1000 + b).fit(X[j]).predict(
            X[common])
        s.append(adjusted_rand_score(la, lb))
    return float(np.mean(s))


def gap(X, ks=tuple(range(1, 9)), *, refs: int = 10, seed: int = 0) -> dict:
    """Tibshirani, Walther and Hastie's gap statistic (2001), with reference
    data uniform over a box aligned with the principal components. Returns
    the gap curve, the k chosen by their rule (the smallest k whose gap is at
    least the next one's minus its standard error) and the k with the
    largest gap."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ vt.T
    lo, hi = Z.min(0), Z.max(0)
    rows = []
    for k in ks:
        w = np.log(KMeans(k, n_init=3, random_state=0).fit(X).inertia_)
        wb = [np.log(KMeans(k, n_init=3, random_state=0).fit(
            rng.uniform(lo, hi, Z.shape) @ vt).inertia_) for _ in range(refs)]
        rows.append({"k": k, "gap": float(np.mean(wb) - w),
                     "se": float(np.std(wb) * np.sqrt(1 + 1 / refs))})
    rule = rows[-1]["k"]
    for a, b in zip(rows, rows[1:]):
        if a["gap"] >= b["gap"] - b["se"]:
            rule = a["k"]
            break
    return {"rows": rows, "rule": rule,
            "argmax": max(rows, key=lambda r: r["gap"])["k"]}


def _shapes(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "one elongated Gaussian": (rng.normal(size=(1000, 2)) * [3, 1], 1),
        "uniform square": (rng.random((1000, 2)), 1),
        "three blobs": (np.r_[rng.normal(size=(300, 2)),
                              rng.normal(size=(300, 2)) + [6, 0],
                              rng.normal(size=(300, 2)) + [3, 5]], 3),
        "two pairs of blobs": (np.r_[rng.normal(size=(250, 2)),
                                     rng.normal(size=(250, 2)) + [4, 0],
                                     rng.normal(size=(250, 2)) + [20, 0],
                                     rng.normal(size=(250, 2)) + [24, 0]], 4),
    }


def stability_shapes(ks=tuple(range(2, 9))) -> dict:
    """Stability and gap on four shapes with a known number of groups."""
    out = {}
    for name, (X, truth) in _shapes().items():
        st = {k: stability(X, k) for k in ks}
        g = gap(X)
        out[name] = {"truth": truth, "stability": st,
                     "stable_k": max(st, key=st.get),
                     "gap_rule": g["rule"], "gap_argmax": g["argmax"],
                     "gap": g["rows"]}
    return out


def stability_digits(ks=tuple(range(2, 16)), *, pairs: int = 10,
                     seeds=(0, 1, 2)) -> dict:
    """Stability of k-means on the 1,797 digits, k from 2 to 15, and the gap
    statistic from 1 to 15."""
    from sklearn.datasets import load_digits
    X, _ = load_digits(return_X_y=True)
    runs = [{k: stability(X, k, pairs=pairs, seed=s) for k in ks}
            for s in seeds]
    st = runs[0]
    g = gap(X, ks=tuple(range(1, max(ks) + 1)), refs=5)
    return {"stability": st, "stable_k": max(st, key=st.get),
            "stable_ks": [max(r, key=r.get) for r in runs], "runs": runs,
            "gap_rule": g["rule"], "gap_argmax": g["argmax"],
            "gap": g["rows"]}
