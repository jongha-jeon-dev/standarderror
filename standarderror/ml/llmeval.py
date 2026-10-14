r"""Grading a language model: the measurement problems of Arc XI.

Every model here is one this repository trained and committed: the 804k-
parameter arithmetic transformers of the school-maths series, a family of
snapshots from a shorter run, and a handful of short continuations of the main
model. They are tiny, which is the point. Every answer they give can be graded
exactly, so each evaluation habit borrowed from large-model practice -- a
leaderboard score, a contamination check, a prompt format, a multiple-choice
score, a model acting as a judge -- can be checked against the truth instead of
against another estimate.

The functions are grouped by episode:

* ML 51, a score is a sample: `item_matrix`, `leaderboard_draws`,
  `paired_difference`, `clustered_items`.
* ML 52, contamination: the dose runs in `scripts/llm_eval_runs.py`, read back
  by `contamination`.
* ML 53, the prompt is part of the measurement: `context_formats`,
  `format_search`.
* ML 54, generation versus likelihood: `choice_scores`.
* ML 55, a model as judge: `judge_scores`, `ppi_study`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

import standarderror as se
from standarderror.schoolmath import arithmetic as ar
from standarderror.schoolmath import curriculum as cur
from standarderror.schoolmath import model as sm

#: The item pool ML 51 draws benchmarks from: the five places where the main
#: model is neither perfect nor hopeless, plus `linear` and `sub`, where it is
#: nearly perfect, so that a benchmark has easy items as real ones do.
POOL = [("test", "mul"), ("test", "order"), ("test", "linear"),
        ("test", "sub"), ("probe_carry", "add")]

#: The leaked benchmark of ML 52: tasks the main model gets wrong often
#: enough (about one multiplication in four, over half the mixed problems)
#: for memorisation to have room to show.
LEAK_TASKS = ["mul", "order"]
LEAK_N, FRESH_N = 150, 300


def data_dir() -> Path:
    return Path(se.SETTINGS.repo_root) / "data" / "ml_llmeval"


def items(split: str, task: str, n: int, *, seed: int = 1,
          skip: int = 0) -> list:
    """`n` problems of one task, a random draw rather than the first `n`.

    The corpus lists problems by operand width, so a head-of-list sample would
    be almost all one-digit sums. `skip` takes the next block of the same
    permutation, which is how a disjoint "fresh" set is drawn.
    """
    pool = [p for p in cur.corpus(split, per_width=4_000, seed=0)
            if p.task == task]
    idx = np.random.default_rng(seed).permutation(len(pool))
    return [pool[i] for i in idx[skip:skip + n]]


def pool_items(per_task: int = 500) -> list:
    out = []
    for split, task in POOL:
        out.extend(items(split, task, per_task))
    return out


def ok(bundle, problems, *, fill_seed: int = 3, fills=None) -> np.ndarray:
    """Exact match of the greedy answer, one fresh context per problem."""
    f = ar.fillers(len(problems), seed=fill_seed) if fills is None else fills
    return np.array([r["ok"] for r in ar.graded(bundle, problems, filler=f)])


def answer_logp(bundle, prompts, answers, fills, *,
                chunk: int = 512) -> np.ndarray:
    """log p(answer + newline | context + prompt), summed over characters.

    The closing newline is scored on purpose. Without it, "35" would be judged
    a likelier answer to `12+345=` than "357", because every prefix of a string
    is at least as probable as the string.
    """
    import torch
    model, stoi = bundle["model"], bundle["stoi"]
    nl = stoi["\n"]
    out = []
    for i in range(0, len(prompts), chunk):
        seqs, spans = [], []
        for p, a, f in zip(prompts[i:i + chunk], answers[i:i + chunk],
                           fills[i:i + chunk]):
            tail = p + a + "\n"
            room = sm.BLOCK - len(tail)
            text = (f[-room:] if room > 0 else "") + tail
            seqs.append([stoi[c] for c in text])
            spans.append(len(a) + 1)
        width = max(len(s) for s in seqs)
        ids = torch.tensor([[nl] * (width - len(s)) + s for s in seqs])
        with torch.no_grad():
            lg = torch.log_softmax(model(ids)[0], -1)
        for j, k in enumerate(spans):
            tgt = ids[j, width - k:]
            pr = lg[j, width - k - 1:width - 1]
            out.append(float(pr.gather(1, tgt.view(-1, 1)).sum()))
    return np.array(out)


def finetune(bundle, extra_lines=(), *, repeats: int = 0, steps: int = 600,
             lr: float = 3e-4, seed: int = 0):
    """Continue training a copy on the training stream plus leaked lines.

    About one more pass over the training split: 600 steps of 64 windows of
    64 characters is 2.46M characters against a 2.17M-character stream. Each
    leaked line is inserted `repeats` times at random positions, so a line is
    seen roughly `repeats` times. `repeats=0` is the control: the same extra
    training with nothing leaked, which is what separates "more training" from
    "saw the answers".
    """
    import torch
    torch.manual_seed(seed)
    b = copy.deepcopy(bundle)
    model, stoi = b["model"], b["stoi"]
    model.train()
    rng = np.random.default_rng(seed)
    lines = [p.line for p in cur.corpus("train", per_width=20_000, seed=0)]
    lines += list(extra_lines) * repeats
    order = rng.permutation(len(lines))
    data = torch.tensor([stoi[c] for c in "".join(lines[i] for i in order)])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(data) - sm.BLOCK - 1, (64,), generator=gen)
        x = torch.stack([data[j:j + sm.BLOCK] for j in i])
        y = torch.stack([data[j + 1:j + sm.BLOCK + 1] for j in i])
        _, loss, _, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return b


def leak_sets():
    """The leaked benchmark and a fresh set from the same generator."""
    bench, fresh = [], []
    for task in LEAK_TASKS:
        bench += items("test", task, LEAK_N, seed=0)
        fresh += items("test", task, FRESH_N, seed=0, skip=LEAK_N)
    return bench, fresh


def runs() -> dict:
    """What `scripts/llm_eval_runs.py` measured and committed."""
    return json.loads((data_dir() / "runs.json").read_text())


# --------------------------------------------------------------------------
# ML 51: a score is a sample
# --------------------------------------------------------------------------

#: The models ML 51 ranks, strongest cluster first. `cont0`-`cont4` are the
#: main model continued for 600 steps under five seeds; `s1_*` are snapshots
#: of a separate, shorter run.
RANKED = ["clean", "cont0", "cont1", "cont2", "cont3", "cont4", "leaky",
          "s1_6000", "s1_4500", "s1_3000"]


def item_matrix(r: dict | None = None):
    """Models x items, exact-match, from the committed runs."""
    r = runs() if r is None else r
    names = [n for n in RANKED if n in r["models"]]
    m = np.array([r["models"][n]["pool_ok"] for n in names], dtype=bool)
    per = r["pool_per_task"]
    tasks = np.repeat([f"{p['split']}/{p['task']}" for p in r["pool"]], per)
    return names, m, tasks


def leaderboard_draws(m: np.ndarray, sizes=(50, 100, 200, 500, 1000, 2500),
                      draws: int = 2000, seed: int = 0) -> list[dict]:
    """Rank the models on benchmarks of `n` items drawn from the pool.

    Draws are with replacement, so the pool plays the part of the infinite
    population a benchmark is a sample of and the textbook interval applies
    without a finite-population correction. The truth is the pool mean.
    """
    rng = np.random.default_rng(seed)
    truth = m.mean(1)
    order = np.argsort(-truth)
    k, n_items = m.shape
    out = []
    for n in sizes:
        top = pair_wrong = 0
        exact = 0
        covered = np.zeros(k)
        for _ in range(draws):
            idx = rng.integers(0, n_items, n)
            s = m[:, idx].mean(1)
            top += int(np.argmax(s) == order[0])
            exact += int(np.all(np.argsort(-s, kind="stable") == order))
            wrong = 0
            for i in range(k):
                for j in range(i + 1, k):
                    wrong += (s[i] - s[j]) * (truth[i] - truth[j]) < 0
            pair_wrong += wrong / (k * (k - 1) / 2)
            se = np.sqrt(np.maximum(s * (1 - s), 1e-12) / n)
            covered += np.abs(s - truth) <= 1.96 * se
        out.append({"n": n, "top_right": top / draws,
                    "order_right": exact / draws,
                    "pairs_wrong": pair_wrong / draws,
                    "coverage": float(covered.mean() / draws),
                    "coverage_min": float(covered.min() / draws),
                    "coverage_max": float(covered.max() / draws)})
    return out


def paired_difference(a: np.ndarray, b: np.ndarray) -> dict:
    """Difference in accuracy of two models graded on the same items.

    The unpaired standard error treats the two columns as independent samples.
    The paired one uses only the items where the models disagree, which is all
    the information there is about which is better; McNemar's exact test is
    the binomial test on those disagreements.
    """
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    pa, pb = a.mean(), b.mean()
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(b & ~a))
    d = a.astype(float) - b.astype(float)
    disc = only_a + only_b
    p = binomtest(only_a, disc, 0.5).pvalue if disc else 1.0
    return {"n": n, "acc_a": float(pa), "acc_b": float(pb),
            "diff": float(pa - pb), "only_a": only_a, "only_b": only_b,
            "se_unpaired": float(np.sqrt(pa * (1 - pa) / n
                                         + pb * (1 - pb) / n)),
            "se_paired": float(d.std(ddof=1) / np.sqrt(n)),
            "mcnemar_p": float(p),
            "corr": float(np.corrcoef(a, b)[0, 1])}


def paired_draws(a, b, n: int = 300, draws: int = 4000, seed: int = 0):
    """How often a benchmark of `n` items calls the comparison each way.

    Also returns the empirical standard deviation of the difference across
    draws, which is what either standard error is trying to estimate.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    truth = a.mean() - b.mean()
    diffs, sig_u, sig_p, wrong = [], 0, 0, 0
    for _ in range(draws):
        idx = rng.integers(0, len(a), n)
        r = paired_difference(a[idx], b[idx])
        diffs.append(r["diff"])
        sig_u += abs(r["diff"]) > 1.96 * r["se_unpaired"]
        sig_p += abs(r["diff"]) > 1.96 * r["se_paired"]
        wrong += r["diff"] * truth < 0
    return {"truth": float(truth), "sd": float(np.std(diffs)),
            "sig_unpaired": sig_u / draws, "sig_paired": sig_p / draws,
            "wrong_sign": wrong / draws}


def clustered_items(bundle, problems, *, repeats: int = 5,
                    seed: int = 40) -> np.ndarray:
    """Each problem asked `repeats` times, each time in a fresh context.

    Returns a questions x repeats matrix. A benchmark built this way -- several
    prompts, paraphrases or samples per question -- has `questions x repeats`
    rows and far fewer independent observations.
    """
    cols = [ok(bundle, problems, fill_seed=seed + r) for r in range(repeats)]
    return np.stack(cols, 1)


def cluster_study(mat: np.ndarray, questions: int = 100, draws: int = 4000,
                  seed: int = 0) -> dict:
    """Naive, cluster-robust and empirical standard errors of the mean."""
    rng = np.random.default_rng(seed)
    q_all, r = mat.shape
    means, naive, robust = [], [], []
    for _ in range(draws):
        idx = rng.integers(0, q_all, questions)
        sub = mat[idx].astype(float)
        p = sub.mean()
        means.append(p)
        naive.append(np.sqrt(p * (1 - p) / sub.size))
        robust.append(sub.mean(1).std(ddof=1) / np.sqrt(questions))
    qm = mat.mean(1)
    within = float(np.mean(mat.var(1, ddof=0)))
    total = float(mat.var())
    return {"questions": questions, "repeats": r, "rows": questions * r,
            "empirical": float(np.std(means)),
            "naive": float(np.mean(naive)), "robust": float(np.mean(robust)),
            "icc": float(1 - within / total) if total else float("nan"),
            "always": float(np.mean(qm == 1)), "never": float(np.mean(qm == 0)),
            "mixed": float(np.mean((qm > 0) & (qm < 1)))}


# --------------------------------------------------------------------------
# ML 53: the prompt is part of the measurement
# --------------------------------------------------------------------------

#: Six contexts a problem can be asked in. `mixed` is how every earlier
#: school-maths number was measured.
FORMATS = ["mixed", "empty", "two", "same", "other", "wrong"]
FORMAT_CASES = [("test", "mul"), ("test", "order"), ("probe_carry", "add"),
                ("test", "linear"), ("test", "add_rev")]

_TRAIN: dict = {}


def _train_by_task():
    if not _TRAIN:
        for p in cur.corpus("train", per_width=1_500, seed=0):
            _TRAIN.setdefault(p.task, []).append(p)
            _TRAIN.setdefault("*", []).append(p)
    return _TRAIN


def contexts(kind: str, task: str, n: int, seed: int) -> list[str]:
    """One context per problem, of one kind.

    mixed: twelve random training lines of any task; empty: nothing (the
    prompt starts the window); two: two random lines; same: twelve lines of
    the task being asked; other: twelve lines of a different task; wrong:
    twelve lines of the same task with the last digit of each answer changed,
    the arithmetic version of few-shot examples with wrong labels.
    """
    by = _train_by_task()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if kind == "empty":
            lines = []
        elif kind in ("mixed", "two"):
            pool = by["*"]
            m = 12 if kind == "mixed" else 2
            lines = [pool[i].line for i in rng.integers(0, len(pool), m)]
        elif kind in ("same", "wrong"):
            pool = by[task]
            picks = [pool[i] for i in rng.integers(0, len(pool), 12)]
            if kind == "same":
                lines = [p.line for p in picks]
            else:
                lines = [p.prompt + p.answer[:-1]
                         + str((int(p.answer[-1]) + 1) % 10) + "\n"
                         for p in picks]
        elif kind == "other":
            pool = by["linear" if task != "linear" else "mul"]
            lines = [pool[i].line for i in rng.integers(0, len(pool), 12)]
        else:
            raise ValueError(kind)
        out.append("".join(lines))
    return out


def context_formats(models: dict, cases=FORMAT_CASES, kinds=FORMATS, *,
                    n: int = 200, seeds: int = 3) -> dict:
    """Accuracy of each model on each case under each kind of context.

    Three context seeds per kind (one for `empty`, which has no randomness),
    so the spread between kinds can be set against the spread within one.
    """
    out = {}
    for split, task in cases:
        probs = items(split, task, n)
        for kind in kinds:
            for name, b in models.items():
                accs = []
                for s in range(1 if kind == "empty" else seeds):
                    f = contexts(kind, task, len(probs), 100 + s)
                    accs.append(float(ok(b, probs, fills=f).mean()))
                out[f"{split}/{task}/{kind}/{name}"] = accs
    return out


def prefix_search(models: dict, cases, *, prefixes: int = 16,
                  n: int = 150) -> dict:
    """One fixed few-shot prefix shared by every problem, sixteen of them.

    This is the prompt template of practice: the same handful of worked
    examples in front of every question. Each prefix is scored on a dev set
    and on a disjoint test set of the same task, so "pick the best prefix on
    dev" can be checked on items the choice never saw.
    """
    pool = _train_by_task()["*"]
    pres = []
    for s in range(prefixes):
        rng = np.random.default_rng(500 + s)
        pres.append("".join(pool[i].line
                            for i in rng.integers(0, len(pool), 12)))
    out = {"prefixes": prefixes, "n": n}
    for split, task in cases:
        dev = items(split, task, n)
        test = items(split, task, n, skip=n)
        for name, b in models.items():
            rows = [(float(ok(b, dev, fills=[p] * n).mean()),
                     float(ok(b, test, fills=[p] * n).mean())) for p in pres]
            out[f"{split}/{task}/{name}"] = rows
    return out


# --------------------------------------------------------------------------
# ML 54: generation versus likelihood
# --------------------------------------------------------------------------

CHOICE_CASES = [("test", "mul"), ("test", "order"), ("probe_carry", "add"),
                ("probe_length", "add")]
DISTRACTORS = ["random", "near", "length"]


def distractors(answer: str, kind: str, rng, k: int = 3) -> list[str]:
    """`k` wrong answers of one kind.

    random: other numbers with as many digits; near: one digit moved by one
    (by up to two for a one-digit answer), the error a person or a model
    actually makes; length: a digit dropped,
    appended or prepended, so the options differ in length.
    """
    sign = "-" if answer.startswith("-") else ""
    d = answer.lstrip("-")
    out: set[str] = set()
    for _ in range(500):
        if len(out) == k:
            break
        if kind == "random":
            lo = 10 ** (len(d) - 1) if len(d) > 1 else 0
            c = sign + str(rng.integers(lo, 10 ** len(d)))
        elif kind == "near":
            pos = int(rng.integers(len(d)))
            # A one-digit answer has only two neighbours at distance one.
            steps = [-1, 1] if len(d) > 1 else [-2, -1, 1, 2]
            nd = (int(d[pos]) + int(rng.choice(steps))) % 10
            if pos == 0 and nd == 0 and len(d) > 1:
                continue
            c = sign + d[:pos] + str(nd) + d[pos + 1:]
        elif kind == "length":
            r = int(rng.integers(3))
            if r == 0 and len(d) > 1:
                c = sign + d[:-1]
            elif r == 1:
                c = sign + d + str(rng.integers(10))
            else:
                c = sign + str(rng.integers(1, 10)) + d
        else:
            raise ValueError(kind)
        if c != answer:
            out.add(c)
    return sorted(out)


def choice_scores(models: dict, cases=CHOICE_CASES, kinds=DISTRACTORS, *,
                  n: int = 200) -> dict:
    """Greedy-generation accuracy next to multiple-choice accuracy.

    Multiple choice: the model picks the option with the highest likelihood,
    either summed over characters or averaged per character (the length
    normalisation evaluation harnesses offer). Same problems, same contexts,
    same model; only the scoring rule changes.
    """
    out = {}
    for split, task in cases:
        probs = items(split, task, n)
        fills = ar.fillers(len(probs), seed=9)
        for name, b in models.items():
            row = {"gen": float(ok(b, probs, fills=fills).mean())}
            for kind in kinds:
                rng = np.random.default_rng(2)
                opts = [[p.answer] + distractors(p.answer, kind, rng)
                        for p in probs]
                P, A, F = [], [], []
                for p, os_, f in zip(probs, opts, fills):
                    for o in os_:
                        P.append(p.prompt)
                        A.append(o)
                        F.append(f)
                lp = answer_logp(b, P, A, F)
                lens = np.array([len(a) + 1 for a in A])
                at, s_sum, s_avg, short = 0, [], [], []
                for os_ in opts:
                    v, ln = lp[at:at + len(os_)], lens[at:at + len(os_)]
                    at += len(os_)
                    s_sum.append(np.argmax(v) == 0)
                    s_avg.append(np.argmax(v / ln) == 0)
                    short.append(len(os_[int(np.argmax(v))]) < len(os_[0]))
                row[kind] = float(np.mean(s_sum))
                row[kind + "_avg"] = float(np.mean(s_avg))
                row[kind + "_shorter"] = float(np.mean(short))
            out[f"{split}/{task}/{name}"] = row
    return out


# --------------------------------------------------------------------------
# ML 55: a model as judge
# --------------------------------------------------------------------------

JUDGE_CASES = [("probe_carry", "add"), ("test", "order"), ("test", "mul")]


def judge_scores(models: dict, cases=JUDGE_CASES, *, n: int = 300) -> dict:
    """Every model judges every model's answers.

    The judge is the simplest one there is: the judge's own per-character
    log-likelihood of the answer, given the question, in a context of its own.
    High means "this looks right to me". Answers are generated in one set of
    contexts and judged in another, so the judge is not re-reading the exact
    window the answer came from.
    """
    out = {}
    for split, task in cases:
        probs = items(split, task, n)
        gen_f = ar.fillers(len(probs), seed=9)
        judge_f = ar.fillers(len(probs), seed=21)
        answers = {s: ar.graded(b, probs, filler=gen_f)
                   for s, b in models.items()}
        for j, jb in models.items():
            for s, rows in answers.items():
                got = [r["got"] if r["got"] and all(c in jb["stoi"]
                                                    for c in r["got"])
                       else "0" for r in rows]
                lp = answer_logp(jb, [p.prompt for p in probs], got, judge_f)
                per = lp / np.array([len(g) + 1 for g in got])
                out[f"{split}/{task}/{j}/{s}"] = {
                    "score": per.round(5).tolist(),
                    "ok": [bool(r["ok"]) for r in rows]}
    return out


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else float("nan")


def judged(scores: dict, case: str, judge: str, sources) -> dict:
    """Judged accuracy of each source, threshold set as kindly as possible.

    The threshold makes the judge's pass rate over all sources together equal
    the true pass rate -- calibration no real judge gets for free. Whatever
    bias survives that is not a calibration problem.
    """
    s = {k: np.array(scores[f"{case}/{judge}/{k}"]["score"]) for k in sources}
    y = {k: np.array(scores[f"{case}/{judge}/{k}"]["ok"]) for k in sources}
    pooled = np.concatenate(list(s.values()))
    truth = np.concatenate(list(y.values())).mean()
    t = np.quantile(pooled, 1 - truth)
    return {k: {"true": float(y[k].mean()), "judged": float((s[k] > t).mean()),
                "pass_wrong": float((s[k][~y[k]] > t).mean())
                if (~y[k]).any() else float("nan"),
                "pass_right": float((s[k][y[k]] > t).mean())
                if y[k].any() else float("nan"),
                "mean_right": float(s[k][y[k]].mean())
                if y[k].any() else float("nan"),
                "mean_wrong": float(s[k][~y[k]].mean())
                if (~y[k]).any() else float("nan"),
                "auc": _auc(y[k], s[k])}
            for k in sources} | {"threshold": float(t)}


def ppi_study(judge_pass: np.ndarray, truth: np.ndarray, *,
              unlabeled: int = 300, labeled: int = 40, draws: int = 4000,
              seed: int = 0) -> dict:
    """Judge alone, labels alone, and prediction-powered inference.

    Each draw takes `unlabeled` items the judge grades and `labeled` items a
    person grades as well. PPI (Angelopoulos et al., 2023) adds the judge's
    average error on the labelled items back onto the judge's estimate, which
    removes its bias and keeps most of its sample size. Coverage is of the
    pool's true accuracy.
    """
    rng = np.random.default_rng(seed)
    jp, y = np.asarray(judge_pass, float), np.asarray(truth, float)
    target = y.mean()
    rows = {"judge": [], "labels": [], "ppi": [], "tuned": []}
    for _ in range(draws):
        u = rng.integers(0, len(y), unlabeled)
        lab = rng.integers(0, len(y), labeled)
        j_mean = jp[u].mean()
        se_j = np.sqrt(max(j_mean * (1 - j_mean), 1e-12) / unlabeled)
        y_l = y[lab]
        l_mean = y_l.mean()
        se_l = np.sqrt(max(l_mean * (1 - l_mean), 1e-12) / labeled)
        rect = y_l - jp[lab]
        ppi = j_mean + rect.mean()
        se_p = np.sqrt(jp[u].var(ddof=1) / unlabeled
                       + rect.var(ddof=1) / labeled)
        # Power tuning (Angelopoulos et al., 2023b): shrink the judge's share
        # by its estimated usefulness, so a judge that barely tracks the truth
        # cannot make the interval wider than the labels alone would.
        jl = jp[lab]
        vj = jl.var(ddof=1)
        lam = 0.0 if vj == 0 else float(np.clip(
            np.cov(y_l, jl)[0, 1] / ((1 + labeled / unlabeled) * vj), 0, 1))
        rect_t = y_l - lam * jl
        tuned = lam * j_mean + rect_t.mean()
        se_t = np.sqrt(lam ** 2 * jp[u].var(ddof=1) / unlabeled
                       + rect_t.var(ddof=1) / labeled)
        for key, est, sd in (("judge", j_mean, se_j), ("labels", l_mean, se_l),
                             ("ppi", ppi, se_p), ("tuned", tuned, se_t)):
            rows[key].append((est, sd, abs(est - target) <= 1.96 * sd))
    out = {"target": float(target), "unlabeled": unlabeled,
           "labeled": labeled}
    for key, v in rows.items():
        a = np.array(v, dtype=float)
        out[key] = {"mean": float(a[:, 0].mean()),
                    "bias": float(a[:, 0].mean() - target),
                    "width": float(2 * 1.96 * a[:, 1].mean()),
                    "coverage": float(a[:, 2].mean()),
                    "rmse": float(np.sqrt(np.mean((a[:, 0] - target) ** 2)))}
    return out


# --------------------------------------------------------------------------
# ML 52: contamination
# --------------------------------------------------------------------------

def _tpr_at(y, s, fpr: float = 0.05) -> float:
    """Share of positives flagged when 5% of negatives are."""
    t = np.quantile(s[y == 0], 1 - fpr)
    return float(np.mean(s[y == 1] > t))


def contamination(r: dict | None = None) -> dict:
    """Dose response and detection, from the committed fine-tuning runs.

    For each run: accuracy on the 300 leaked items and on 600 fresh items
    from the same tasks; the AUC of a loss detector that sees only the
    contaminated model (higher log-likelihood = flagged as leaked); and the
    AUC of a reference detector that also sees the uncontaminated base model
    and flags items whose likelihood rose. The controls are the five runs
    with the same extra training and nothing leaked.
    """
    r = runs() if r is None else r
    d = r["doses"]
    base = d["base"]
    b_lp, f_lp = np.array(base["bench_lp"]), np.array(base["fresh_lp"])
    y = np.r_[np.ones(len(b_lp)), np.zeros(len(f_lp))]
    rows = {}
    for key, v in d.items():
        bl, fl = np.array(v["bench_lp"]), np.array(v["fresh_lp"])
        rows[key] = {
            "dose": None if key == "base" else int(key.split("/")[0]),
            "seed": None if key == "base" else int(key.split("/")[1]),
            "bench": float(np.mean(v["bench_ok"])),
            "fresh": float(np.mean(v["fresh_ok"])),
            "auc_loss": _auc(y, np.r_[bl, fl]),
            "auc_ref": _auc(y, np.r_[bl - b_lp, fl - f_lp]),
            "tpr5_loss": _tpr_at(y, np.r_[bl, fl]),
            "tpr5_ref": _tpr_at(y, np.r_[bl - b_lp, fl - f_lp]),
            "lp_bench": float(bl.mean()), "lp_fresh": float(fl.mean()),
            "lp_sd": float(np.r_[bl, fl].std()),
            "shift_sd": float((bl - b_lp).std()),
            "shift_mean": float((bl - b_lp).mean() - (fl - f_lp).mean()),
            # Set-level: is the leaked set likelier than a fresh one, as a
            # whole? Welch's t on the per-item log-likelihoods.
            "lp_t": float((bl.mean() - fl.mean())
                          / np.sqrt(bl.var(ddof=1) / len(bl)
                                    + fl.var(ddof=1) / len(fl)))}
    ctrl = [v for v in rows.values() if v["dose"] == 0]
    c_bench = np.array([v["bench"] for v in ctrl])
    c_gap = np.array([v["bench"] - v["fresh"] for v in ctrl])
    for v in rows.values():
        v["inflation"] = v["bench"] - float(c_bench.mean())
        v["gap"] = v["bench"] - v["fresh"]
    return {"runs": rows, "control_bench_mean": float(c_bench.mean()),
            "control_bench_sd": float(c_bench.std(ddof=1)),
            "control_gap_min": float(c_gap.min()),
            "control_gap_max": float(c_gap.max()),
            "n_bench": len(b_lp), "n_fresh": len(f_lp)}
