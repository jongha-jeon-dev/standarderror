r"""Where a gradient step in embedding space actually lands.

The embedding table of this model is 65 vectors in 128 dimensions, and the
first thing to measure about it is not learned at all -- it is what high
dimensions do to any such table. The rows have a median norm of 7.93 and a
median pairwise distance of 11.214, and `sqrt(2) * 7.93 = 11.210`. That is
the distance between *orthogonal* vectors of that norm, and the median
pairwise cosine is `+0.0003`. So the table is 65 near-orthogonal points on a
sphere, every one of them about as far from every other.

**Which means there is no neighbourhood.** The nearest other token sits at
0.88 of the typical pairwise distance, so "the token closest to this one" is
barely a distinguished thing. Gradient descent assumes a local region in which
moving a little changes the answer a little; an embedding table offers none.

**So a descent step goes nowhere.** Take the gradient of the loss with respect
to one input position's embedding and walk against it. The nearest row to
where you land is still the token you started from, for any step under
**1.9 times the embedding norm** and up to 31.6 times in one of six contexts
measured here. The table lives on a sphere of radius 7.9 and you have to walk
15 to 250 units before the geometry notices.

**And when it does notice, it is not right.** The token whose cell you first
enter ranks 1st, 1st, 4th, 5th, 19th and 26th of 65 against the enumerated
truth -- never the best substitution -- and in one context its loss is *higher*
than the token you started from. Descending the gradient and rounding to the
nearest embedding made the model worse there.

**The ranking is a different question, and a better one.** Episode 4's
first-order score is the same object HotFlip ranks candidates by, and ranking
is scale-invariant, so the compression that destroyed the gradient's magnitude
costs a ranking nothing. Measured: the true best substitution lands in the top
five in four of six contexts, and at rank 14 and rank 55 in the other two,
with Spearman correlations from -0.30 to +0.50. Adequate as a shortlist,
unusable as a direction.

**And graded on the job it is actually asked to do, it is good.** "Where does
the single best token rank" is the wrong question for a propose-then-evaluate
loop; the right one is how much of the available improvement a short list
recovers. Evaluating the gradient's top five of 65 recovers **96% to 100%** of
the gain in five of six contexts -- including the one where the best token
ranks 55th, because a near-tie is as good as a win. The sixth recovers 16%,
and it is the context where only 0.28 nats were on offer to begin with.

So the two answers point opposite ways and both are right. As a direction the
gradient is useless here: the geometry gives it nowhere to go. As a proposal
distribution it cuts 65 candidates to 5 and loses almost nothing. Prompt
optimisation and every discrete-token attack is **search with a
gradient-shaped shortlist**, and the reason that works is not that the
gradient is accurate -- episode 4 measured how inaccurate -- but that ranking
five candidates correctly enough is a far weaker requirement than pointing.

**And the shortlist is the gradient's, not the evaluation's.** Five of 65 is a
7.7% sample, so the claim needs a control: five tokens drawn at random recover
a median of -0.21 to 0.71 of the gain against the gradient's 0.16 to 1.00, and
the share of random draws that match or beat the gradient's five is 0.08 to
0.15 in every context. So the ordering is carrying real information -- just
the only kind of information this geometry leaves room for.

References: Ebrahimi et al., "HotFlip: white-box adversarial examples for text
classification", *ACL* (2018), for the first-order substitution score; Wallace
et al., "Universal adversarial triggers for attacking and analyzing NLP",
*EMNLP* (2019) and Zou et al., "Universally transferable adversarial attacks
on aligned language models" (2023), for the propose-then-evaluate loop that
this geometry forces; Vershynin, *High-Dimensional Probability* (2018), for
why independent vectors in 128 dimensions are near-orthogonal.
"""

from __future__ import annotations

import math


def _np():
    import numpy as np
    return np


def _torch():
    import torch
    return torch


def table() -> dict:
    """The geometry of the embedding table, before anything is differentiated.

    `orthogonal_prediction` is `sqrt(2)` times the median norm, which is the
    distance between orthogonal vectors of that length. Comparing it with the
    measured median pairwise distance is the whole claim about the table's
    shape, and it needs no model of why.
    """
    np = _np()

    from standarderror.llm import tiny

    e = tiny.load()["model"].tok.weight.detach().numpy()
    norms = np.linalg.norm(e, axis=1)
    d = np.linalg.norm(e[:, None] - e[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(1)
    finite = d[np.isfinite(d)]
    cos = (e @ e.T) / np.outer(norms, norms)
    np.fill_diagonal(cos, np.nan)
    return {
        "rows": int(e.shape[0]), "dim": int(e.shape[1]),
        "median_norm": float(np.median(norms)),
        "norm_spread": float(norms.std() / norms.mean()),
        "median_pairwise": float(np.median(finite)),
        "median_nearest": float(np.median(nearest)),
        "nearest_over_pairwise": float(np.median(nearest)
                                       / np.median(finite)),
        "median_cosine": float(np.nanmedian(cos)),
        "abs_cosine_p90": float(np.nanpercentile(np.abs(cos), 90)),
        "orthogonal_prediction": float(np.median(norms) * math.sqrt(2.0)),
        "embeddings": e,
    }


def substitutions(context, target: int, slot: int | None = None) -> dict:
    """Every single-token substitution at one position, scored exactly.

    The same trick as episode 4: a vocabulary of 65 makes the answer a table
    rather than an estimate, so a proposal method can be graded instead of
    admired.
    """
    np = _np()
    torch = _torch()
    F = torch.nn.functional

    from standarderror.llm import tiny

    bundle = tiny.load()
    model, V = bundle["model"], bundle["vocab"]
    ctx = torch.as_tensor(context, dtype=torch.long)
    pos = len(ctx) - 1 if slot is None else int(slot)
    seqs = ctx.repeat(V, 1).clone()
    seqs[:, pos] = torch.arange(V)
    with torch.no_grad():
        out, _, _, _ = model(seqs)
        loss = F.cross_entropy(out[:, -1], torch.full((V,), int(target)),
                               reduction="none").numpy()
    current = int(ctx[pos])
    return {"context": ctx, "target": int(target), "slot": pos,
            "vocab": int(V), "loss": loss, "current": current,
            "current_loss": float(loss[current]),
            "best": int(np.argmin(loss)), "best_loss": float(loss.min()),
            "order": np.argsort(loss)}


def gradient_at(sub: dict):
    """The loss gradient with respect to the embedding at the chosen slot."""
    torch = _torch()
    F = torch.nn.functional

    from standarderror.llm import tiny

    model = tiny.load()["model"]
    ctx = sub["context"]
    e = model.tok(ctx).detach().clone().requires_grad_(True)
    h = (e + model.pos(torch.arange(len(ctx)))).unsqueeze(0)
    for blk in model.blocks:
        h, _ = blk(h)
    F.cross_entropy(model.head(model.lnf(h))[0, -1:],
                    torch.tensor([sub["target"]])).backward()
    return e.grad[sub["slot"]].numpy().copy()


def ranking(sub: dict, embeddings=None) -> dict:
    """HotFlip's first-order score, graded against the enumerated truth.

    The score is `<g, E_j - E_current>`: how much the loss would drop if the
    embedding moved to token `j`, to first order. Ranking is scale-invariant,
    so the compression episode 4 found does not touch this -- which makes it
    a genuinely separate question with its own answer.
    """
    np = _np()
    from scipy.stats import spearmanr

    from standarderror.llm import tiny

    e = (tiny.load()["model"].tok.weight.detach().numpy()
         if embeddings is None else embeddings)
    g = gradient_at(sub)
    cur = sub["current"]
    score = (e - e[cur]) @ g
    order = np.argsort(score)
    rank_of_best = int(np.where(order == sub["best"])[0][0])
    truth = sub["loss"] - sub["current_loss"]
    top5 = [int(t) for t in order[:5]]
    best5 = float(sub["loss"][top5].min())
    available = sub["current_loss"] - sub["best_loss"]
    # The metric that matters for a propose-then-evaluate loop is not where
    # the single best token ranks but how much of the available improvement a
    # short list recovers. The two disagree sharply: one context puts the best
    # token at rank 55 and still recovers 96% of the gain.
    return {"score": score, "order": order,
            "rank_of_best": rank_of_best,
            "best_in_top5": bool(rank_of_best < 5),
            "spearman": float(spearmanr(score, truth).statistic),
            "top5": top5,
            "best_loss_in_top5": best5,
            "available": float(available),
            "regret5": float(best5 - sub["best_loss"]),
            "recovered5": (float((sub["current_loss"] - best5) / available)
                           if available > 1e-9 else float("nan")),
            "gradient": g}


def step_landing(sub: dict, *, limit: float = 60.0, step: float = 0.05
                 ) -> dict:
    """How far you must walk before the nearest embedding row changes, and to what.

    `flip_at` is in units of the current token's embedding norm, so 1.0 means
    "a step the size of the embedding itself". The search stops at `limit`;
    `flip_at` is None if nothing changed by then, which would itself be the
    finding.
    """
    np = _np()

    from standarderror.llm import tiny

    e = tiny.load()["model"].tok.weight.detach().numpy()
    g = gradient_at(sub)
    direction = g / max(np.linalg.norm(g), 1e-30)
    cur = sub["current"]
    norm = float(np.linalg.norm(e[cur]))
    flip_at = landed = None
    for frac in np.arange(step, float(limit), step):
        point = e[cur] - frac * norm * direction
        j = int(np.argmin(np.linalg.norm(e - point, axis=1)))
        if j != cur:
            flip_at, landed = float(frac), j
            break
    order = np.argsort(sub["loss"])
    out = {"flip_at": flip_at, "landed": landed,
           "embedding_norm": norm,
           "distance_walked": None if flip_at is None else flip_at * norm}
    if landed is not None:
        out.update({
            "landed_loss": float(sub["loss"][landed]),
            "landed_rank": int(np.where(order == landed)[0][0]),
            "made_it_worse": bool(sub["loss"][landed] > sub["current_loss"]),
        })
    return out


def survey(*, contexts: int = 6, seed: int = 0, limit: float = 60.0) -> dict:
    """Ranking and landing for several decisions, against the exact answer."""
    np = _np()

    from standarderror.llm import tiny

    x, y = next(iter(tiny.batches(count=1, size=int(contexts), seed=int(seed))))
    rows = []
    for k in range(int(contexts)):
        sub = substitutions(x[k, :tiny.BLOCK - 1],
                            int(y[k, tiny.BLOCK - 1]))
        rank = ranking(sub)
        rows.append({"context": k, "sub": sub, "ranking": rank,
                     "landing": step_landing(sub, limit=limit),
                     "control": random_shortlist(sub, rank, seed=k)})
    ranks = [r["ranking"]["rank_of_best"] for r in rows]
    rec = [r["ranking"]["recovered5"] for r in rows]
    flips = [r["landing"]["flip_at"] for r in rows
             if r["landing"]["flip_at"] is not None]
    return {
        "rows": rows,
        "top5_hits": int(sum(r["ranking"]["best_in_top5"] for r in rows)),
        "rank_min": int(min(ranks)), "rank_max": int(max(ranks)),
        "rank_median": float(np.median(ranks)),
        "recovered_min": float(min(rec)),
        "recovered_median": float(np.median(rec)),
        "recovered_above_95": int(sum(v > 0.95 for v in rec)),
        "random_median_max": float(max(r["control"]["median"] for r in rows)),
        "beats_max": float(max(r["control"]["beats_gradient"]
                               for r in rows)),
        "spearman_min": float(min(r["ranking"]["spearman"] for r in rows)),
        "spearman_max": float(max(r["ranking"]["spearman"] for r in rows)),
        "flip_min": float(min(flips)) if flips else float("nan"),
        "flip_max": float(max(flips)) if flips else float("nan"),
        "landings_worse": int(sum(r["landing"].get("made_it_worse", False)
                                  for r in rows)),
        "landings_best": int(sum(r["landing"].get("landed_rank", -1) == 0
                                 for r in rows)),
    }


def random_shortlist(sub: dict, rank: dict, *, size: int = 5,
                     draws: int = 4000, seed: int = 0) -> dict:
    """The control the shortlist claim needs: how good is a shortlist of five?

    Five of 65 is a 7.7% sample, which is not nothing, so "the gradient's top
    five recovers most of the gain" means little until the same question is
    asked of five tokens picked at random. `beats_gradient` is the share of
    random draws that do at least as well, which is the number that decides
    whether the gradient is contributing or the evaluation is doing all of it.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    cur, avail = sub["current_loss"], rank["available"]
    out = np.empty(int(draws))
    for i in range(int(draws)):
        idx = rng.choice(sub["vocab"], int(size), replace=False)
        out[i] = (cur - sub["loss"][idx].min()) / max(avail, 1e-9)
    return {"size": int(size), "draws": int(draws),
            "median": float(np.median(out)),
            "p90": float(np.percentile(out, 90)),
            "beats_gradient": float((out >= rank["recovered5"]).mean())}
