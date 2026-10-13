r"""Arcs V and VI: training small networks, and what they generalise.

Everything here runs on the UCI digits (64 pixels, 10 classes, half splits)
or on simulations, with small multilayer perceptrons in PyTorch on a CPU.

Episode 21, `init_depth`: activation and gradient scale through 30 layers for
different initial weight scales, and whether the network then trains.
Episode 22, `lr_batch`: learning rate against batch size at a fixed budget
of epochs.
Episode 23, `adam_vs_sgd`: both optimisers, each tuned, over many seeds.
Episode 24, `batchnorm_dependence`: one example's prediction as its batch
changes.
Episode 25, `dropout_mc`: weight scaling against averaging dropout masks.
Episode 26, `random_labels`: the same network on true, partly random and
random labels.
Episode 27, `modular`: modular addition trained long past 100% training
accuracy.
Episode 28, `augmentation`: label-preserving and label-breaking
augmentations against more real data.
Episode 29, `covariate_shift`: accuracy, regret and calibration under a
controlled shift, with importance weighting.
"""

from __future__ import annotations

import functools
import math


def _memo(fn):
    """Run a study on one CPU thread, restoring the caller's setting after,
    and memoise it. One thread makes the floating-point order, and so every
    number an episode quotes, the same on any machine; restoring it keeps the
    rest of the package (which trains on two) unaffected. The studies are
    deterministic and some take minutes, and an episode calls each one twice,
    once for its figures and once in the code block the reader sees, so the
    second call is free."""
    @functools.cache
    @functools.wraps(fn)
    def run(*args, **kwargs):
        import torch
        before = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.set_num_threads(before)
    return run


def _torch():
    import torch
    return torch


def _np():
    import numpy as np
    return np


# ------------------------------------------------------------------ shared

def digits(seed: int = 0, *, images: bool = False):
    """Digits half split, standardised on the training half (or raw 8 x 8
    images in [0, 16] when `images`)."""
    torch = _torch()
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5,
                                      random_state=int(seed), stratify=y)
    if not images:
        # As scikit-learn's StandardScaler: a pixel that is constant in the
        # training half keeps scale 1, so ink there in a test image stays a
        # small number instead of becoming 1e8.
        mu, sd = Xa.mean(0), Xa.std(0)
        sd[sd < 1e-8] = 1.0
        Xa, Xb = (Xa - mu) / sd, (Xb - mu) / sd
    T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return T(Xa), T(Xb), torch.tensor(ya), torch.tensor(yb)


def mlp(sizes, *, act: str = "relu", gain: float | None = None,
        const: float | None = None, dropout: float = 0.0,
        batchnorm: bool = False, seed: int = 0):
    """A fully connected network with the given layer sizes. `gain` draws
    weights from N(0, gain^2 / fan_in) with zero biases; `const` sets every
    weight to one value; otherwise PyTorch's default initialisation."""
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(int(seed))
    A = {"relu": nn.ReLU, "tanh": nn.Tanh}[act]
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        lin = nn.Linear(a, b)
        with torch.no_grad():
            if const is not None:
                lin.weight.fill_(const)
                lin.bias.zero_()
            elif gain is not None:
                lin.weight.normal_(0, gain / math.sqrt(a))
                lin.bias.zero_()
        layers.append(lin)
        if i < len(sizes) - 2:
            if batchnorm:
                layers.append(nn.BatchNorm1d(b))
            layers.append(A())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def fit(model, X, y, *, opt: str = "sgd", lr: float = 0.01,
        momentum: float = 0.9, epochs: int = 20, batch: int = 32,
        seed: int = 0, weight_decay: float = 0.0, weights=None,
        curve: bool = False, steps: int | None = None):
    """Minibatch training with a seeded shuffle; returns the per-epoch
    training loss when `curve`. With `steps`, stops after exactly that many
    minibatches, however many epochs that takes."""
    torch = _torch()
    F = torch.nn.functional
    if opt == "adam":
        o = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=weight_decay)
    else:
        o = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)
    g = torch.Generator().manual_seed(int(seed))
    out = []
    n = len(y)
    done = 0
    for _ in range(int(epochs) if steps is None else 10 ** 9):
        if steps is not None and done >= steps:
            break
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - batch + 1 if n >= batch else 1, batch):
            if steps is not None and done >= steps:
                break
            done += 1
            b = perm[i:i + batch]
            if weights is None:
                loss = F.cross_entropy(model(X[b]), y[b])
            else:
                loss = (F.cross_entropy(model(X[b]), y[b], reduction="none")
                        * weights[b]).mean()
            o.zero_grad()
            loss.backward()
            o.step()
        if curve:
            out.append(loss_of(model, X, y))
    return out


def loss_of(model, X, y) -> float:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        v = torch.nn.functional.cross_entropy(model(X), y).item()
    return v if math.isfinite(v) else float("inf")


def acc_of(model, X, y) -> float:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        return float((model(X).argmax(1) == y).float().mean())


# ================================================================ episode 21

def _layer_stats(model, X, y):
    torch = _torch()
    nn = torch.nn
    model.train()
    stds, h = [], X
    for m in model:
        h = m(h)
        if isinstance(m, (nn.Tanh, nn.ReLU)):
            stds.append(float(h.detach().std()))
    model.zero_grad()
    nn.functional.cross_entropy(model(X), y).backward()
    grads = [float(m.weight.grad.norm()) for m in model
             if isinstance(m, nn.Linear)]
    return stds, grads


@_memo
def init_depth(*, depth: int = 30, width: int = 128,
               gains=(0.5, 1.0, math.sqrt(2), 2.0),
               lrs=(0.001, 0.003, 0.01, 0.03), epochs: int = 30) -> dict:
    """For tanh and ReLU networks of `depth` hidden layers: the standard
    deviation of each layer's activations and the gradient norm of each
    layer's weights at initialisation, and the best test accuracy over a
    small learning-rate grid after `epochs` of SGD with momentum. Plus a
    constant initialisation, which is symmetric."""
    Xa, Xb, ya, yb = digits(0)
    sizes = [64] + [width] * depth + [10]
    rows = []
    for act in ("tanh", "relu"):
        for g in gains:
            stds, grads = _layer_stats(mlp(sizes, act=act, gain=g), Xa, ya)
            accs = []
            for lr in lrs:
                m = mlp(sizes, act=act, gain=g)
                fit(m, Xa, ya, lr=lr, epochs=epochs)
                accs.append(acc_of(m, Xb, yb))
            rows.append({"act": act, "gain": float(g), "stds": stds,
                         "grads": grads, "accs": accs,
                         "best": max(accs), "best_lr": lrs[accs.index(
                             max(accs))]})
    accs = []
    for lr in lrs:
        m = mlp(sizes, act="tanh", const=0.01)
        fit(m, Xa, ya, lr=lr, epochs=epochs)
        accs.append(acc_of(m, Xb, yb))
    W = m[0].weight.detach()
    distinct = int((W - W[:1]).abs().max(1).values.gt(1e-6).sum()) + 1
    return {"depth": depth, "width": width, "lrs": list(lrs), "rows": rows,
            "const_best": max(accs), "const_distinct_units": distinct}


# ================================================================ episode 22

@_memo
def lr_batch(*, batches=(8, 32, 128, 512), lrs=None, epochs: int = 20,
             seeds=(0, 1)) -> dict:
    """Plain SGD on a two-hidden-layer network with a fixed budget of
    examples (`epochs` passes' worth, so epochs x n / B steps at batch size
    B): final training loss and test accuracy for every learning rate and
    batch size, averaged over `seeds`. Losses above 10, and divergence, are
    recorded as 10."""
    np = _np()
    lrs = lrs or [0.003 * 2 ** k for k in range(12)]
    Xa, Xb, ya, yb = digits(0)
    loss = np.zeros((len(batches), len(lrs)))
    acc = np.zeros_like(loss)
    for i, B in enumerate(batches):
        for j, lr in enumerate(lrs):
            ls, ac = [], []
            for s in seeds:
                m = mlp([64, 128, 128, 10], seed=s)
                fit(m, Xa, ya, lr=lr, momentum=0.0, batch=B, seed=s,
                    steps=round(epochs * len(ya) / B))
                ls.append(min(loss_of(m, Xa, ya), 10.0))
                ac.append(acc_of(m, Xb, yb))
            loss[i, j], acc[i, j] = np.mean(ls), np.mean(ac)
    best = [int(np.argmin(r)) for r in loss]
    return {"batches": list(batches), "lrs": list(lrs), "loss": loss,
            "acc": acc, "best_lr": [lrs[k] for k in best],
            "best_loss": [float(loss[i, k]) for i, k in enumerate(best)],
            "best_acc": [float(acc[i, k]) for i, k in enumerate(best)],
            "epochs": epochs}


# ================================================================ episode 23

def _gauss_tensors(seed: int, n: int = 2000):
    torch = _torch()
    from standarderror.ml.curves import GaussianTask
    t = GaussianTask(test=5000)
    X, y = t.draw(n, seed=seed)
    mu, sd = X.mean(0), X.std(0)
    T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return (T((X - mu) / sd), T((t.X_test - mu) / sd), torch.tensor(y),
            torch.tensor(t.y_test), t.bayes)


@_memo
def adam_vs_sgd(*, task: str = "digits", seeds=range(20), epochs: int = 30,
                adam_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
                sgd_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
                tune_seed: int = 100) -> dict:
    """Tune each optimiser's learning rate on a validation fifth of one
    split, then train both from the same initial weights on `seeds` splits.
    Returns per-seed test accuracy and the training-loss curves, and, in
    `tuning`, each grid point's validation accuracy and training curve."""
    np = _np()
    grids = {"adam": list(adam_grid), "sgd": list(sgd_grid)}
    width = 256 if task == "digits" else 64

    def get(seed):
        if task == "digits":
            Xa, Xb, ya, yb = digits(seed)
            return Xa, Xb, ya, yb, None
        return _gauss_tensors(seed)

    def model(seed, d_in, k):
        return mlp([d_in, width, width, k], seed=seed)

    Xa, _, ya, _, _ = get(tune_seed)
    cut = int(0.8 * len(ya))
    tuned, tuning = {}, {}
    for name, grid in grids.items():
        scores, curves = [], []
        for lr in grid:
            m = model(tune_seed, Xa.shape[1], int(ya.max()) + 1)
            curves.append(fit(m, Xa[:cut], ya[:cut], opt=name, lr=lr,
                              epochs=epochs, curve=True))
            scores.append(acc_of(m, Xa[cut:], ya[cut:]))
        tuned[name] = grid[int(np.argmax(scores))]
        tuning[name] = {"lrs": grid, "val_acc": scores, "curves": curves,
                        "n_val": len(ya) - cut}
    out = {k: {"acc": [], "curve": [], "train_loss": []} for k in grids}
    bayes = None
    for s in seeds:
        Xa, Xb, ya, yb, bayes = get(s)
        for name in grids:
            m = model(s, Xa.shape[1], int(ya.max()) + 1)
            c = fit(m, Xa, ya, opt=name, lr=tuned[name], epochs=epochs,
                    seed=s, curve=True)
            out[name]["acc"].append(acc_of(m, Xb, yb))
            out[name]["curve"].append(c)
            out[name]["train_loss"].append(c[-1])
    d = np.array(out["adam"]["acc"]) - np.array(out["sgd"]["acc"])
    return {"task": task, "tuned": tuned, "tuning": tuning,
            "seeds": len(list(seeds)),
            "adam": out["adam"], "sgd": out["sgd"], "diff_mean": float(d.mean()),
            "diff_se": float(d.std(ddof=1) / math.sqrt(len(d))),
            "adam_wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "bayes": bayes, "epochs": epochs}


@_memo
def optimiser_speed(*, seeds=range(5), epochs: int = 30,
                    adam_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
                    sgd_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1)) -> dict:
    """Training loss after every epoch on the digits for every learning
    rate in both grids, from the same initial weights, averaged over `seeds`
    splits; and the test accuracy each run ends with."""
    np = _np()
    out = {}
    for name, grid in (("adam", adam_grid), ("sgd", sgd_grid)):
        curves, accs = [], []
        for lr in grid:
            cs, ac = [], []
            for s in seeds:
                Xa, Xb, ya, yb = digits(s)
                m = mlp([64, 256, 256, 10], seed=s)
                cs.append(fit(m, Xa, ya, opt=name, lr=lr, epochs=epochs,
                              seed=s, curve=True))
                ac.append(acc_of(m, Xb, yb))
            curves.append(np.mean(cs, 0).tolist())
            accs.append(float(np.mean(ac)))
        out[name] = {"lrs": list(grid), "curves": curves, "acc": accs}
    return out


# ================================================================ episode 24

@_memo
def batchnorm_dependence(*, batch_sizes=(2, 4, 16, 64), draws: int = 200,
                         examples: int = 100, seed: int = 0) -> dict:
    """Train a batch-normalised network, then classify test examples in
    training mode inside random batches of test companions, and inside
    batches whose companions all share one class. Compare with the
    inference-mode prediction."""
    torch = _torch()
    np = _np()
    import copy
    Xa, Xb, ya, yb = digits(seed)
    m = mlp([64, 128, 128, 10], batchnorm=True, seed=seed)
    fit(m, Xa, ya, lr=0.05, epochs=20, batch=64, seed=seed)
    eval_acc = acc_of(m, Xb, yb)
    with torch.no_grad():
        m.eval()
        eval_pred = m(Xb).argmax(1)
    g = np.random.default_rng(seed)
    idx = g.choice(len(yb), examples, replace=False)
    rows = []
    for B in batch_sizes:
        mm = copy.deepcopy(m)
        mm.train()
        for mod in mm:
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.momentum = 0.0          # do not touch running stats
        flips, correct, spread = [], [], []
        with torch.no_grad():
            for i in idx:
                preds, logit = [], []
                for _ in range(draws):
                    comp = g.choice(len(yb), B - 1, replace=False)
                    xb = torch.cat([Xb[[i]], Xb[comp]]) if B > 1 else Xb[[i]]
                    out = mm(xb)[0]
                    preds.append(int(out.argmax()))
                    logit.append(float(out[int(yb[i])]))
                preds = np.array(preds)
                flips.append(float(np.mean(preds != int(eval_pred[i]))))
                correct.append(float(np.mean(preds == int(yb[i]))))
                spread.append(float(np.std(logit)))
        rows.append({"batch": B, "flip": float(np.mean(flips)),
                     "accuracy": float(np.mean(correct)),
                     "any_flip": float(np.mean(np.array(flips) > 0)),
                     "logit_sd": float(np.median(spread))})
    # companions all from one class
    mm = copy.deepcopy(m)
    mm.train()
    for mod in mm:
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.momentum = 0.0
    B = 16
    same, total = 0, 0
    with torch.no_grad():
        for i in idx:
            for c in range(10):
                pool = np.where(yb.numpy() == c)[0]
                comp = g.choice(pool, B - 1, replace=False)
                out = mm(torch.cat([Xb[[i]], Xb[comp]]))[0]
                same += int(out.argmax()) == int(eval_pred[i])
                total += 1
    sample_acc = float((eval_pred[idx] == yb[idx]).float().mean())
    return {"eval_acc": eval_acc, "sample_eval_acc": sample_acc,
            "rows": rows, "examples": examples,
            "draws": draws, "skewed_batch": B,
            "skewed_agree": same / total}


@_memo
def batchnorm_batch_size(*, batches=(2, 4, 16, 64), seeds=(0, 1, 2),
                         epochs: int = 20) -> list[dict]:
    """Train the same network with and without batch normalisation at each
    batch size (Adam, learning rate 1e-3, a fixed number of epochs), and
    score it in inference mode and, for the normalised one, in training mode
    on shuffled test batches of the size it was trained with."""
    torch = _torch()
    np = _np()
    import copy
    out = []
    for B in batches:
        acc = {"plain": [], "bn": [], "bn_train_mode": []}
        for s in seeds:
            Xa, Xb, ya, yb = digits(s)
            for kind in ("plain", "bn"):
                m = mlp([64, 128, 128, 10], batchnorm=kind == "bn", seed=s)
                fit(m, Xa, ya, opt="adam", lr=1e-3, epochs=epochs, batch=B,
                    seed=s)
                acc[kind].append(acc_of(m, Xb, yb))
                if kind == "bn":
                    mm = copy.deepcopy(m)
                    mm.train()
                    for mod in mm:
                        if isinstance(mod, torch.nn.BatchNorm1d):
                            mod.momentum = 0.0
                    g = torch.Generator().manual_seed(s)
                    perm = torch.randperm(len(yb), generator=g)
                    right = n = 0
                    with torch.no_grad():
                        for i in range(0, len(yb) - B + 1, B):
                            b = perm[i:i + B]
                            right += int((mm(Xb[b]).argmax(1) == yb[b]).sum())
                            n += len(b)
                    acc["bn_train_mode"].append(right / n)
        out.append({"batch": B, **{k: float(np.mean(v))
                                   for k, v in acc.items()}})
    return out


# ================================================================ episode 25

@_memo
def dropout_mc(*, p: float = 0.5, Ks=(1, 2, 5, 10, 30, 100, 300),
               seeds=(0, 1, 2), task: str = "digits",
               epochs: int = 60) -> dict:
    """Train with dropout; at test time compare weight scaling (PyTorch's
    eval mode) with the average of K sampled masks, by test accuracy, log
    loss and agreement with the weight-scaled prediction; for the largest K,
    mean top-class probability and calibration error of both, and the mean
    top-class probability of the averaged prediction on the rows where the
    two disagree. `pair` holds both predictions' probability of the last
    class for the first seed."""
    torch = _torch()
    np = _np()
    F = torch.nn.functional
    res = {"ws": [], "mc": {k: [] for k in Ks}}
    extra = {"ws_conf": [], "mc_conf": [], "ws_ece": [], "mc_ece": [],
             "dis_conf": [], "agree_conf": []}
    pair = None
    for s in seeds:
        if task == "digits":
            Xa, Xb, ya, yb = digits(s)
            sizes = [64, 256, 256, 10]
        else:
            Xa, Xb, ya, yb, _ = _gauss_tensors(s)
            sizes = [10, 128, 128, 2]
        m = mlp(sizes, dropout=p, seed=s)
        fit(m, Xa, ya, lr=0.01, epochs=epochs, seed=s)
        m.eval()
        with torch.no_grad():
            pw = F.softmax(m(Xb), 1)
            ws_pred = pw.argmax(1)
            res["ws"].append({
                "acc": float((ws_pred == yb).float().mean()),
                "nll": float(F.nll_loss(torch.log(pw + 1e-12), yb))})
            m.train()
            torch.manual_seed(1000 + s)
            samples = torch.stack([F.softmax(m(Xb), 1)
                                   for _ in range(max(Ks))])
        for k in Ks:
            pm = samples[:k].mean(0)
            res["mc"][k].append({
                "acc": float((pm.argmax(1) == yb).float().mean()),
                "nll": float(F.nll_loss(torch.log(pm + 1e-12), yb)),
                "agree": float((pm.argmax(1) == ws_pred).float().mean())})
        pm = samples.mean(0)
        cw, cm = pw.max(1).values.numpy(), pm.max(1).values.numpy()
        dis = (pm.argmax(1) != ws_pred).numpy()
        extra["ws_conf"].append(float(cw.mean()))
        extra["mc_conf"].append(float(cm.mean()))
        extra["ws_ece"].append(_ece_top(cw, (ws_pred == yb).numpy()))
        extra["mc_ece"].append(_ece_top(cm, (pm.argmax(1) == yb).numpy()))
        extra["dis_conf"].append(float(cm[dis].mean()) if dis.any() else
                                 float("nan"))
        extra["agree_conf"].append(float(cm[~dis].mean()))
        if pair is None:
            pair = {"ws": pw[:, -1].tolist(), "mc": pm[:, -1].tolist()}
    mean = lambda lst, key: float(np.mean([d[key] for d in lst]))  # noqa
    return {"task": task, "p": p, "Ks": list(Ks),
            "ws_acc": mean(res["ws"], "acc"), "ws_nll": mean(res["ws"], "nll"),
            "mc_acc": [mean(res["mc"][k], "acc") for k in Ks],
            "mc_nll": [mean(res["mc"][k], "nll") for k in Ks],
            "mc_agree": [mean(res["mc"][k], "agree") for k in Ks],
            **{k: float(np.nanmean(v)) for k, v in extra.items()},
            "pair": pair, "seeds": len(seeds)}


def _ece_top(conf, correct, bins: int = 10) -> float:
    """Expected calibration error of the top-class probability."""
    np = _np()
    e = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[e == b].mean() - correct[e == b].mean()) *
                     (e == b).mean() for b in range(bins) if (e == b).any()))


# ================================================================ episode 26

@_memo
def random_labels(*, fractions=(0.0, 0.2, 0.5, 1.0), epochs: int = 400,
                  seed: int = 0, lr: float = 1e-3) -> list[dict]:
    """A 2 x 512 network trained full-batch with Adam on the digits training
    half with a fraction of labels replaced by random ones (some of which
    land on the true label by chance): the step at which training accuracy
    first reaches 100%, and test accuracy then. Every 5 steps the curve
    records (step, training accuracy, test accuracy, accuracy on the rows
    whose label was not changed, accuracy on the rows whose label was), the
    last two against the labels the network is trained on."""
    torch = _torch()
    np = _np()
    F = torch.nn.functional
    Xa, Xb, ya, yb = digits(seed)
    out = []
    for frac in fractions:
        g = np.random.default_rng(seed)
        yt = ya.clone()
        k = int(round(frac * len(yt)))
        idx = g.choice(len(yt), k, replace=False)
        yt[idx] = torch.tensor(g.integers(0, 10, k))
        changed = yt != ya
        m = mlp([64, 512, 512, 10], seed=seed)
        o = torch.optim.Adam(m.parameters(), lr=lr)
        hit, curve = None, []
        for e in range(1, epochs + 1):
            m.train()
            loss = F.cross_entropy(m(Xa), yt)
            o.zero_grad()
            loss.backward()
            o.step()
            m.eval()
            with torch.no_grad():
                right = m(Xa).argmax(1) == yt
            tr = float(right.float().mean())
            if e % 5 == 0 or e == 1:
                kept = float(right[~changed].float().mean())
                moved = float(right[changed].float().mean()) \
                    if changed.any() else float("nan")
                curve.append((e, tr, acc_of(m, Xb, yb), kept, moved))
            if hit is None and tr == 1.0:
                hit = e
        with torch.no_grad():
            clean = float((m(Xa).argmax(1) == ya).float().mean())
        out.append({"fraction": frac, "epoch_100": hit,
                    "changed": float(changed.float().mean()),
                    "test_acc": acc_of(m, Xb, yb), "train_acc": tr,
                    "curve": curve, "clean_fit": clean})
    return out


# ================================================================ episode 27

@_memo
def modular(*, p: int = 97, frac: float = 0.4, steps: int = 10000,
            weight_decay: float = 1.0, lr: float = 1e-3, every: int = 100,
            seed: int = 0, embed: int = 128, hidden: int = 256) -> dict:
    """(a + b) mod p from embeddings of a and b through one hidden layer,
    full-batch AdamW on a `frac` share of all p^2 pairs; training and test
    accuracy every `every` steps."""
    torch = _torch()
    F = torch.nn.functional
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    y = (a + b) % p
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(p * p, generator=g)
    n = int(frac * p * p)
    tr, te = perm[:n], perm[n:]

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e = torch.nn.Embedding(p, embed)
            self.f = torch.nn.Sequential(torch.nn.Linear(2 * embed, hidden),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(hidden, p))

        def forward(self, i, j):
            return self.f(torch.cat([self.e(i), self.e(j)], 1))

    torch.manual_seed(int(seed))
    m = M()
    o = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay,
                          betas=(0.9, 0.98))
    hist = []
    for s in range(1, steps + 1):
        loss = F.cross_entropy(m(a[tr], b[tr]), y[tr])
        o.zero_grad()
        loss.backward()
        o.step()
        if s % every == 0:
            with torch.no_grad():
                hist.append({
                    "step": s,
                    "train": float((m(a[tr], b[tr]).argmax(1) == y[tr])
                                   .float().mean()),
                    "test": float((m(a[te], b[te]).argmax(1) == y[te])
                                  .float().mean()),
                    "norm": float(sum(q.pow(2).sum() for q in
                                      m.parameters()).sqrt())})

    def first(key, thr):
        for h in hist:
            if h[key] >= thr:
                return h["step"]
        return None

    return {"p": p, "frac": frac, "weight_decay": weight_decay,
            "history": hist, "train_100": first("train", 1.0),
            "test_99": first("test", 0.99), "final_test": hist[-1]["test"],
            "steps": steps}


def grokking_runs() -> list[dict]:
    """The four runs `scripts/train_grokking.py` committed to
    data/ml_grokking/results.json: weight decay 1 and 0 on 40% of the pairs,
    weight decay 1 on 30% and on 60%."""
    import json
    from pathlib import Path

    import standarderror as se
    path = Path(se.SETTINGS.repo_root) / "data" / "ml_grokking" / "results.json"
    return json.loads(path.read_text())["runs"]


# ================================================================ episode 28

def _shift(img, dx, dy):
    np = _np()
    out = np.zeros_like(img)
    h, w = img.shape
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else \
        (slice(-dx, w), slice(0, w + dx))
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else \
        (slice(-dy, h), slice(0, h + dy))
    out[yd, xd] = img[ys, xs]
    return out


def augment(X, kind: str, rng):
    """One augmented copy of each 8 x 8 image (rows of 64 pixels)."""
    np = _np()
    imgs = X.reshape(-1, 8, 8)
    if kind == "shift":
        out = [_shift(im, *rng.integers(-1, 2, 2)) for im in imgs]
    elif kind == "flip":
        out = [im[:, ::-1] for im in imgs]
    elif kind == "rot180":
        out = [im[::-1, ::-1] for im in imgs]
    elif kind == "noise":
        out = [np.clip(im + rng.normal(0, 2, im.shape), 0, 16) for im in imgs]
    else:
        raise ValueError(kind)
    return np.array(out).reshape(len(X), 64)


def _aug_run(Xt, yt, tests, seed, steps):
    """Train the episode-28 network for `steps` minibatches of 32 on (Xt, yt)
    and score it on the clean and the moved test images."""
    torch = _torch()
    # One mean and scale for all pixels, not one per pixel: a pixel that is
    # blank in every training image must not turn a shifted stroke into an
    # enormous input.
    mu, sd = Xt.mean(), Xt.std()

    def T(a):
        return torch.tensor((a - mu) / sd, dtype=torch.float32)

    m = mlp([64, 128, 10], seed=seed)
    fit(m, T(Xt), torch.tensor(yt), lr=0.01, steps=steps, seed=seed)
    Xa, Xb, y = tests
    return acc_of(m, T(Xa), torch.tensor(y)), acc_of(m, T(Xb), torch.tensor(y))


@_memo
def augmentation(*, n: int = 150, copies: int = 4,
                 kinds=("shift", "noise", "flip", "rot180"),
                 seeds=range(10), steps: int = 300) -> dict:
    """Train on `n` real digits; add `copies` augmented copies of each by
    each kind; compare with training on n * (1 + copies) real digits. The
    same network and number of gradient steps throughout; scored on the
    clean test images and on the same images moved by up to a pixel."""
    np = _np()
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    res = {k: [] for k in ("base", "real", *kinds)}
    moved = {k: [] for k in res}
    for s in seeds:
        g = np.random.default_rng(s)
        perm = g.permutation(len(y))
        test = perm[:700]
        pool = perm[700:]
        tr = pool[:n]
        more = pool[:n * (1 + copies)]
        X_moved = augment(X[test], "shift", np.random.default_rng(10_000 + s))

        tests = (X[test], X_moved, y[test])
        for key, Xt, yt in (("base", X[tr], y[tr]), ("real", X[more], y[more])):
            a, b = _aug_run(Xt, yt, tests, s, steps)
            res[key].append(a)
            moved[key].append(b)
        for k in kinds:
            Xs = [X[tr]] + [augment(X[tr], k, g) for _ in range(copies)]
            ys = [y[tr]] * (1 + copies)
            a, b = _aug_run(np.vstack(Xs), np.concatenate(ys), tests, s, steps)
            res[k].append(a)
            moved[k].append(b)

    def summary(d):
        base = np.array(d["base"])
        return {**{k: float(np.mean(v)) for k, v in d.items()},
                "gain": {k: float(np.mean(np.array(v) - base))
                         for k, v in d.items() if k != "base"},
                "gain_se": {k: float(np.std(np.array(v) - base, ddof=1) /
                                     math.sqrt(len(base)))
                            for k, v in d.items() if k != "base"}}

    return {"n": n, "copies": copies, "seeds": len(list(seeds)),
            **summary(res), "moved": summary(moved)}


@_memo
def centring() -> dict:
    """Where the digits sit in their 8 x 8 frame: the spread of each image's
    centre of ink, and how often the outer columns hold any ink."""
    np = _np()
    from sklearn.datasets import load_digits
    im = load_digits().images
    m = im.sum((1, 2))
    cx = (im.sum(1) * np.arange(8)).sum(1) / m
    cy = (im.sum(2) * np.arange(8)).sum(1) / m
    off = np.maximum(abs(cx - cx.mean()), abs(cy - cy.mean())) > 1
    return {"sd_x": float(cx.std()), "sd_y": float(cy.std()),
            "off_by_a_pixel": float(off.mean()),
            "left_col_ink": float((im[:, :, 0] > 0).mean()),
            "right_col_ink": float((im[:, :, 7] > 0).mean())}


@_memo
def flip_confusions() -> dict:
    """Which digits change meaning under each label-breaking transform:
    the share of test images of each class that a clean model assigns to a
    different class after the transform."""
    torch = _torch()
    np = _np()
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    g = np.random.default_rng(0)
    perm = g.permutation(len(y))
    tr, te = perm[:1000], perm[1000:]
    mu, sd = X[tr].mean(), X[tr].std()
    T = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)  # noqa
    m = mlp([64, 128, 10], seed=0)
    fit(m, T(X[tr]), torch.tensor(y[tr]), lr=0.01, epochs=40)
    out = {}
    for k in ("flip", "rot180"):
        Xa = augment(X[te], k, g)
        with torch.no_grad():
            pred = m(T(Xa)).argmax(1).numpy()
        out[k] = {int(c): float(np.mean(pred[y[te] == c] != c))
                  for c in range(10)}
    return out


# ================================================================ episode 29

def _shift_task(n, mu, seed):
    np = _np()
    g = np.random.default_rng(int(seed))
    x = g.normal(mu, 1.0, n)
    p = 1 / (1 + np.exp(-3 * np.sin(1.5 * x)))
    return x, (g.random(n) < p).astype(int), p


@_memo
def covariate_shift(*, shifts=(0.0, 0.5, 1.0, 1.5, 2.0), n: int = 500,
                    test: int = 20000, reps: int = 20) -> list[dict]:
    """x ~ N(0, 1) in training and N(shift, 1) in test; P(y = 1 | x) =
    sigmoid(3 sin(1.5 x)) in both. A linear logistic model (misspecified)
    and a flexible one (spline features), each with and without importance
    weights p_test(x) / p_train(x). Accuracy, regret against the Bayes rule
    on the test distribution, expected calibration error, and the effective
    sample size of the weights."""
    np = _np()
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    def ece(p, y, bins=10):
        e = np.minimum((p * bins).astype(int), bins - 1)
        return float(sum(abs(p[e == b].mean() - y[e == b].mean()) *
                         (e == b).mean() for b in range(bins)
                         if (e == b).any()))

    out = []
    for mu in shifts:
        xt, yt, pt = _shift_task(test, mu, 999)
        bayes = float(np.mean((pt > 0.5) == yt))
        acc = {k: [] for k in ("linear", "linear_iw", "flex", "flex_iw")}
        cal = {k: [] for k in acc}
        nll = {k: [] for k in acc}
        xo, yo, _ = _shift_task(200000, mu, 7)
        orc = LogisticRegression().fit(xo[:, None], yo)
        po = orc.predict_proba(xt[:, None])[:, 1]
        oracle = {"acc": float(np.mean((po > 0.5) == yt)),
                  "nll": float(-np.mean(yt * np.log(po) +
                                        (1 - yt) * np.log(1 - po))),
                  "coef": float(orc.coef_[0, 0]),
                  "intercept": float(orc.intercept_[0])}
        coefs = {"linear": [], "linear_iw": []}
        ess = []
        for r in range(int(reps)):
            x, y, _ = _shift_task(n, 0.0, r)
            w = np.exp(mu * x - mu ** 2 / 2)       # N(mu,1) / N(0,1)
            ess.append(float(w.sum() ** 2 / (w ** 2).sum() / n))
            models = {
                "linear": LogisticRegression(),
                "flex": make_pipeline(
                    SplineTransformer(n_knots=8, degree=3,
                                      extrapolation="linear"),
                    LogisticRegression(C=10, max_iter=2000))}
            for name, mdl in models.items():
                for iw in (False, True):
                    key = name + ("_iw" if iw else "")
                    kw = {}
                    if iw:
                        last = mdl.steps[-1][0] if hasattr(mdl, "steps") \
                            else None
                        kw = {f"{last}__sample_weight": w} if last else \
                            {"sample_weight": w}
                    mdl.fit(x[:, None], y, **kw)
                    pr = mdl.predict_proba(xt[:, None])[:, 1]
                    acc[key].append(float(np.mean((pr > 0.5) == yt)))
                    cal[key].append(ece(pr, yt))
                    pr = np.clip(pr, 1e-9, 1 - 1e-9)
                    nll[key].append(float(-np.mean(
                        yt * np.log(pr) + (1 - yt) * np.log(1 - pr))))
                    if name == "linear":
                        coefs[key].append((float(mdl.coef_[0, 0]),
                                           float(mdl.intercept_[0])))
        out.append({"shift": mu, "bayes": bayes,
                    **{f"acc_{k}": float(np.mean(v)) for k, v in acc.items()},
                    **{f"sd_{k}": float(np.std(v)) for k, v in acc.items()},
                    **{f"ece_{k}": float(np.mean(v)) for k, v in cal.items()},
                    **{f"nll_{k}": float(np.mean(v)) for k, v in nll.items()},
                    **{f"coef_{k}": np.mean(v, 0).tolist()
                       for k, v in coefs.items()},
                    "oracle": oracle, "ess": float(np.mean(ess))})
    return out
