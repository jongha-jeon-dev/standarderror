"""A frozen random backbone with low-rank adapters, on a task whose truth is known.

The question
------------
Hazan et al. (2026) train low-rank adapters over a backbone that was never
pre-trained — every weight drawn at random and frozen — and recover most of full
training while optimising a small fraction of the parameters. Three mechanistic
claims come with it, and two of them are claims about *measurement* rather than
about performance:

**The scaffold is actively exploited**, evidenced by the learned per-layer scalar
beta staying strictly positive.

**The rank at which performance saturates estimates the task's intrinsic
dimensionality.**

Neither can be checked on a real benchmark, because a real benchmark does not
come with a known intrinsic dimension and does not let you ask what the model
would have done without the backbone. Both can be checked on data built for the
purpose, and that is what this module is: a task family whose label depends on
the input only through a subspace of a dimension *you choose*, and a model whose
backbone can be switched off without changing anything else.

`make_multi_index` sets the truth. `ScaffoldMLP` is the architecture, with
`scaffold="normal"`, `"zero"` (the backbone removed before training) and
`"ablate"` (removed after it). `rank_sweep` runs the paper's recovery curve, and
`minimum_rank` reads the paper's r* off it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiIndexTask", "make_multi_index", "ScaffoldMLP", "make_splits", "train", "evaluate", "rank_sweep",
           "minimum_rank", "recovery", "DEFAULTS"]

torch.set_num_threads(2)

DEFAULTS = dict(d_in=64, hidden=(128, 128, 128), classes=2, alpha=1.0,
                epochs=80, batch=128, lr=3e-3, weight_decay=0.0)


# ---------------------------------------------------------------- the task

class MultiIndexTask:
    """A binary label that depends on `x` only through a `k`-dimensional subspace.

    This is the one object that makes the paper's third claim checkable: the
    intrinsic dimensionality of the task is `k`, not because a diagnostic said so
    but because the generator was built that way. Everything outside the span of
    `U` is irrelevant by construction, and the ambient dimension `d` stays fixed
    as `k` varies, so "harder" cannot be confused with "wider".

    The link is a sum of random cosines of frequency `freq` on the projected
    coordinates. The frequency is the knob that decides whether the task is one a
    frozen random backbone can already solve: at low frequency a random-feature
    model with a trained readout is enough and nothing downstream can be measured,
    because every condition ties. At high frequency the relevant subspace has to
    be found, which is what makes a *rank* meaningful.

    Two details are load-bearing. The task and the data are drawn from separate
    generators: draw the subspace from the same stream as the samples and the
    train and test splits get different subspaces, which shows up as a model that
    sits at chance with nothing in the training curve to explain why. And the
    label is drawn from a *single* standardised score rather than from two
    standardised logits — standardising two logits separately does not standardise
    their difference, and with a one-dimensional subspace the two become so
    correlated that the Bayes accuracy collapses to 53%.
    """

    def __init__(self, *, d: int = 64, k: int = 4, seed: int = 0,
                 freq: float = 2.0, harmonics: int = 6,
                 temperature: float = 3.0, label_noise: float = 0.0):
        if not 1 <= k <= d:
            raise ValueError("k must be between 1 and d")
        self.d, self.k, self.classes = int(d), int(k), 2
        self.temperature, self.label_noise = float(temperature), float(label_noise)
        g = torch.Generator().manual_seed(int(seed))
        self.U, _ = torch.linalg.qr(torch.randn(self.d, self.k, generator=g))
        # Frequencies of fixed length, so the link's roughness is set by `freq`
        # alone and does not drift as k changes the norm of a Gaussian draw.
        W = torch.randn(int(harmonics), self.k, generator=g)
        self.W = W / W.norm(dim=1, keepdim=True) * float(freq)
        self.phase = torch.rand(int(harmonics), generator=g) * 2 * np.pi
        self.c = torch.randn(int(harmonics), generator=g)
        z = torch.randn(20_000, self.d, generator=g) @ self.U
        raw = self._raw(z)
        self.mu, self.sd = raw.mean(), raw.std().clamp_min(1e-8)

    def _raw(self, z):
        return torch.cos(z @ self.W.T + self.phase) @ self.c

    def score(self, X):
        return (self._raw(X @ self.U) - self.mu) / self.sd * self.temperature

    def sample(self, n: int, *, split: int = 0):
        g = torch.Generator().manual_seed(1_000_003 * (int(split) + 1) + 17 * self.k)
        X = torch.randn(int(n), self.d, generator=g)
        pr = torch.sigmoid(self.score(X))
        y = (torch.rand(pr.shape, generator=g) < pr).long()
        if self.label_noise > 0:
            flip = torch.rand(y.shape, generator=g) < self.label_noise
            y = torch.where(flip, 1 - y, y)
        return X, y

    def bayes_accuracy(self, n: int = 20_000) -> float:
        """The ceiling. Reported rather than assumed, because a recovery number
        measured against a baseline far from the ceiling is a statement about the
        baseline."""
        X, y = self.sample(n, split=99)
        return float(((self.score(X) > 0).long() == y).float().mean())


def make_multi_index(n: int, **kw):
    """One task and its three splits: train, validation (for early stopping) and
    test. Three rather than two, because the stopping epoch is chosen per
    condition and choosing it on the test set would make every number here a
    best-of-fifty."""
    task = MultiIndexTask(**kw)
    return (task, task.sample(n, split=0), task.sample(max(n // 4, 1000), split=2),
            task.sample(max(n // 2, 2000), split=1))


# ---------------------------------------------------------------- the model

class ScaffoldLinear(nn.Module):
    """One layer of the paper's Equation (1): beta * W_seed h + (alpha/r) B A h.

    `W_seed` is a buffer rather than a parameter, so no optimiser state is ever
    allocated for it and "frozen" is enforced by the module rather than by
    remembering to filter the parameter list.
    """

    def __init__(self, d_in: int, d_out: int, *, rank: int, alpha: float,
                 gen: torch.Generator, scaffold: bool = True,
                 train_backbone: bool = False):
        super().__init__()
        # He scaling, which is what 'variance matched to the standard
        # initialisation heuristic' means for a ReLU network.
        w = torch.randn(d_out, d_in, generator=gen) * np.sqrt(2.0 / d_in)
        w = w if scaffold else torch.zeros_like(w)
        # The fully trained baseline is *this same layer* with the backbone
        # unfrozen and no adapter, not a separately written network. Comparing a
        # frozen-plus-adapter layer against a differently parameterised dense
        # layer measures the parameterisation as much as the freezing, and the
        # first version of this file did exactly that: the "baseline" came out
        # four points *below* the rank-4 adapter, which is not a result about
        # low-rank structure.
        if train_backbone:
            self.W_seed = nn.Parameter(w)
        else:
            self.register_buffer("W_seed", w)
        self.rank = int(rank)
        self.alpha = float(alpha)
        if self.rank == 0:
            self.beta = nn.Parameter(torch.ones(()))
            return
        # The standard LoRA initialisation sets B to zero so the adapter starts as
        # a no-op. That is wrong here, and not by a little: with no bias and no
        # scaffold, a zero B makes every activation identically zero, the
        # gradients with it, and the zero-scaffold control — the one that decides
        # whether the backbone matters — trains to exactly chance forever. So both
        # factors are drawn, scaled so that the adapter path alone is a
        # correctly-initialised layer: with Var[A] = 2/d_in and Var[B] = r, the
        # product (alpha/r) B A has the same variance as W_seed at any rank.
        self.A = nn.Parameter(torch.randn(int(rank), d_in, generator=gen)
                              * np.sqrt(2.0 / d_in))
        self.B = nn.Parameter(torch.randn(d_out, int(rank)), )
        with torch.no_grad():
            self.B.mul_(np.sqrt(int(rank)) / float(alpha))
        self.beta = nn.Parameter(torch.ones(()))

    def forward(self, h):
        out = F.linear(h, self.W_seed) * self.beta
        if self.rank == 0:
            return out
        return out + (self.alpha / self.rank) * F.linear(F.linear(h, self.A),
                                                        self.B)

    @torch.no_grad()
    def drop_scaffold(self) -> None:
        self.W_seed.zero_()


class ScaffoldMLP(nn.Module):
    """A frozen random MLP with a rank-`r` adapter per layer and a trained head.

    `scaffold` takes three values and they are three different experiments:

    - "normal": the paper's setting, a random frozen backbone.
    - "zero": the backbone is zero from the start, so the adapters are the whole
      network. This is the control the paper reports in its Table 2 and it is the
      one that decides whether the scaffold is doing anything.
    - "ablate": trained with the backbone, then the backbone is removed before
      evaluation. It answers a different question — whether the trained adapters
      *depend* on the backbone — and is not the same as "zero".

    The head is fully trainable, as in the paper. That is the readout of the
    reservoir analogy, and leaving it frozen would test a different claim.
    """

    def __init__(self, d_in: int, d_out: int, *, hidden=(128, 128, 128),
                 rank: int = 4, alpha: float = 1.0, scaffold: str = "normal",
                 full: bool = False, seed: int = 0):
        super().__init__()
        if scaffold not in ("normal", "zero", "ablate"):
            raise ValueError("scaffold must be normal, zero or ablate")
        self.full = bool(full)
        self.scaffold = scaffold
        dims = [int(d_in), *[int(h) for h in hidden]]
        # `full=True` is the same layer with the backbone unfrozen and the adapter
        # removed: identical init, identical shape, identical scalar beta. The
        # only difference between the two conditions is which tensors the
        # optimiser is allowed to touch.
        # One generator per layer, seeded from (seed, layer index). A single
        # shared stream looks equivalent and is not: the baseline draws no A and
        # no B, so it consumes fewer numbers, and every layer after the first gets
        # a *different* backbone from the adapter model it is supposed to be
        # paired against. The comparison then silently stops being paired, which
        # is the one thing this design depends on.
        def _gen(i):
            return torch.Generator().manual_seed(int(seed) * 1_000_003 + i)

        self.layers = nn.ModuleList(
            [ScaffoldLinear(dims[i], dims[i + 1],
                            rank=0 if full else rank, alpha=alpha, gen=_gen(i),
                            scaffold=(scaffold != "zero"), train_backbone=full)
             for i in range(len(dims) - 1)])
        self.head = nn.Linear(dims[-1], int(d_out))
        gen = _gen(len(dims))
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(self.head.weight.shape,
                                               generator=gen)
                                   * np.sqrt(1.0 / dims[-1]))
            self.head.bias.zero_()

    def forward(self, x):
        h = x
        for lin in self.layers:
            h = F.relu(lin(h))
        return self.head(h)

    def drop_scaffold(self) -> None:
        for lin in self.layers:
            lin.drop_scaffold()

    def betas(self) -> list[float]:
        return [float(layer.beta.detach()) for layer in self.layers
                if hasattr(layer, "beta")]

    def trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total(self) -> int:
        return self.trainable() + sum(b.numel() for b in self.buffers())


# ---------------------------------------------------------------- fitting

def train(model, X, y, *, val=None, epochs: int = 60, batch: int = 256,
          lr: float = 2e-3, weight_decay: float = 1e-3, seed: int = 0):
    """One matched training protocol for every condition, with early stopping.

    Same optimiser, same schedule, same number of passes over the same rows,
    whatever is trainable. Tuning the budget per condition would make every
    comparison here a comparison of tuning effort.

    The early stopping is not a refinement, it is what makes the comparison mean
    anything. The fully trained baseline has an order of magnitude more parameters
    than the adapter and overfits well before the budget is spent: run both to a
    fixed epoch count and the "fully trained ceiling" comes out *below* the
    rank-4 adapter, which reads as a finding and is an artefact. Restoring each
    condition to its own best validation epoch gives the baseline its best shot,
    which is the only version of the comparison worth reporting.
    """
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs))
    g = torch.Generator().manual_seed(int(seed) + 31)
    n = X.shape[0]
    best, best_state, best_epoch = -1.0, None, -1
    for ep in range(int(epochs)):
        model.train()
        order = torch.randperm(n, generator=g)
        for i in range(0, n, int(batch)):
            idx = order[i:i + int(batch)]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(X[idx]), y[idx]).backward()
            opt.step()
        sched.step()
        if val is not None:
            acc = evaluate(model, *val)["acc"]
            if acc > best:
                best, best_epoch = acc, ep
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_epoch = best_epoch
    return model


@torch.no_grad()
def evaluate(model, X, y) -> dict:
    model.eval()
    logits = model(X)
    return {"acc": float((logits.argmax(1) == y).float().mean()),
            "loss": float(F.cross_entropy(logits, y))}


def recovery(acc: float, acc_full: float, *, chance: float = 0.5) -> float:
    """Share of the fully trained model's *above-chance* accuracy recovered.

    The paper divides accuracies directly, which flatters every number on a
    balanced binary task: a coin flip already "recovers" 52% of an 96% baseline.
    Subtracting chance from both sides is the same quantity the paper intends and
    is the reason the curves here start near zero instead of near a half.
    """
    denom = acc_full - chance
    return float((acc - chance) / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------- sweeps

PROTOCOL = dict(epochs=50, batch=256, lr=2e-3, weight_decay=1e-3)


def make_splits(task, n_train: int):
    return (task.sample(n_train, split=0), task.sample(max(n_train // 4, 1000),
                                                       split=2),
            task.sample(max(n_train // 2, 2000), split=1))


def fit_one(task, *, rank: int, hidden, scaffold: str = "normal",
            full: bool = False, seed: int = 0, n_train: int = 24_000,
            splits=None, protocol: dict | None = None) -> dict:
    """Train one condition and report what every later question needs.

    `splits` is accepted so a sweep can reuse one draw across conditions: the
    comparison is paired on rows, for the same reason the previous post's paired
    fits were — the draw moves accuracy by more than the rank does.
    """
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits or make_splits(task, n_train)
    model = ScaffoldMLP(task.d, task.classes, hidden=hidden,
                        rank=0 if full else rank,
                        scaffold="normal" if scaffold == "ablate" else scaffold,
                        full=full, seed=seed)
    train(model, Xtr, ytr, val=(Xva, yva), seed=seed,
          **(protocol or PROTOCOL))
    out = evaluate(model, Xte, yte)
    if scaffold == "ablate" and not full:
        # Removing the backbone *after* training answers a different question
        # from removing it before: whether the trained adapters lean on it.
        model.drop_scaffold()
        out["acc_ablated"] = evaluate(model, Xte, yte)["acc"]
    out.update(rank=int(rank), scaffold="full" if full else scaffold,
               trainable=model.trainable(), total=model.total(),
               beta=model.betas(), best_epoch=int(model.best_epoch),
               width=int(hidden[0]), k=int(task.k), seed=int(seed))
    return out


def rank_sweep(task, ranks, *, hidden, scaffold: str = "normal", seeds: int = 3,
               n_train: int = 24_000, protocol: dict | None = None) -> dict:
    """The paper's recovery curve, plus the fully trained baseline it is read
    against and the Bayes ceiling the baseline is itself read against."""
    splits = make_splits(task, n_train)
    full = [fit_one(task, rank=0, hidden=hidden, full=True, seed=s,
                    splits=splits, protocol=protocol) for s in range(int(seeds))]
    acc_full = float(np.mean([f["acc"] for f in full]))
    rows = []
    for r in ranks:
        runs = [fit_one(task, rank=r, hidden=hidden, scaffold=scaffold, seed=s,
                        splits=splits, protocol=protocol)
                for s in range(int(seeds))]
        acc = float(np.mean([x["acc"] for x in runs]))
        rows.append({"rank": int(r), "acc": acc,
                     "acc_sd": float(np.std([x["acc"] for x in runs], ddof=1))
                     if seeds > 1 else 0.0,
                     "recovery": recovery(acc, acc_full),
                     "trainable": runs[0]["trainable"],
                     "share": runs[0]["trainable"] / full[0]["trainable"],
                     "beta_median": float(np.median([b for x in runs
                                                     for b in x["beta"]])),
                     "acc_ablated": (float(np.mean([x["acc_ablated"]
                                                    for x in runs]))
                                     if "acc_ablated" in runs[0] else None)})
    return {"k": int(task.k), "width": int(hidden[0]), "scaffold": scaffold,
            "acc_full": acc_full,
            "acc_full_sd": float(np.std([f["acc"] for f in full], ddof=1))
            if seeds > 1 else 0.0,
            "bayes": task.bayes_accuracy(), "rows": rows,
            "trainable_full": full[0]["trainable"]}


def minimum_rank(rows, *, target: float = 0.99, key: str = "recovery"):
    """The paper's r*: the smallest rank on the grid that reaches the target.

    Returned as `None` rather than as the largest rank when nothing reaches it,
    because "did not saturate on this grid" and "saturated at the last point I
    happened to try" are different statements and collapsing them is how a
    saturation curve becomes a measurement of the grid.
    """
    for row in sorted(rows, key=lambda r: r["rank"]):
        if row[key] >= target:
            return int(row["rank"])
    return None
