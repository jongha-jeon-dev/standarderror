r"""What the arithmetic model gets right, and where it stops.

Everything here grades a *decoded answer* against the true one, exactly. The
functions are organised around the questions a school curriculum asks in order:
can it add, can it carry, can it carry twice in a row, does place value survive
a longer number, can it undo an operation.

Two habits are load-bearing. First, the forward and reversed comparisons are
**paired** -- the same sums, one model, two answer orders -- so the difference
cannot be a difference in problem difficulty or in training luck. Second, every
headline number has a control next to it, because "the model failed" and "the
prompt looked unusual to the model" produce the same table.
"""

from __future__ import annotations

from standarderror.schoolmath import curriculum as cur
from standarderror.schoolmath import model as sm

_POOL: dict[int, list[str]] = {}


def _lines(seed: int = 0) -> list[str]:
    """Training lines across every width, cached, for use as context."""
    if seed not in _POOL:
        _POOL[seed] = [p.line for p in cur.corpus("train", per_width=1_500,
                                                  seed=seed)]
    return _POOL[seed]


def fillers(n: int, *, seed: int = 0) -> list[str]:
    """`n` *different* contexts, each a run of randomly chosen train lines.

    Different, deliberately. A single shared prefix is a thing the model can
    copy, and a small transformer trained on a repetitive corpus is very good
    at copying: with one fixed filler the measured accuracy on reversed sums
    was 0.008, which is not a fact about arithmetic but about induction. One
    fresh context per problem removes that channel; `padding_sensitivity`
    checks what is left of it.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    pool = _lines()
    idx = rng.integers(0, len(pool), (n, 12))
    return ["".join(pool[j] for j in row) for row in idx]


def graded(bundle, problems, *, filler=None, chunk: int = 256) -> list[dict]:
    """Decode every problem and mark it, in batches.

    The default context is one freshly sampled run of training lines per
    problem, which is the closest thing to how the model saw text.
    """
    fill = fillers(len(problems)) if filler is None else filler
    rows = []
    for i in range(0, len(problems), chunk):
        part = problems[i:i + chunk]
        f = fill[i:i + chunk] if isinstance(fill, list) else fill
        got = sm.decode(bundle, [p.prompt for p in part], filler=f)
        for p, g in zip(part, got):
            rows.append({"task": p.task, "level": p.level, "digits": p.digits,
                         "carry_chain": p.carry_chain, "prompt": p.prompt,
                         "want": p.answer, "got": g, "ok": g == p.answer})
    return rows


def _rate(rows) -> float:
    return sum(r["ok"] for r in rows) / len(rows) if rows else float("nan")


def by_task(bundle, split: str = "test", *, limit: int = 600,
            per_width: int = 4_000, seed: int = 0) -> list[dict]:
    """Exact-match accuracy for each task in one split."""
    pool = cur.corpus(split, per_width=per_width, seed=seed)
    tasks = sorted({p.task for p in pool})
    out = []
    for t in tasks:
        part = [p for p in pool if p.task == t][:limit]
        if not part:
            continue
        rows = graded(bundle, part)
        out.append({"task": t, "n": len(rows), "accuracy": _rate(rows),
                    "level": part[0].level})
    return out


def answer_order(bundle, split: str = "test", *, limit: int = 600,
                 per_width: int = 4_000, seed: int = 0) -> dict:
    """The same sums, written forwards and backwards. Paired.

    A sum is only counted when both of its two forms survived the split, so the
    two accuracies are computed over an identical set of `(a, b)` pairs and the
    difference between them is the answer order alone.
    """
    pool = cur.corpus(split, per_width=per_width, seed=seed)
    fwd = {p.prompt[:-1]: p for p in pool if p.task == "add"}
    rev = {p.prompt[:-1]: p for p in pool if p.task == "add_rev"}
    keys = sorted(set(fwd) & set(rev))[:limit]
    f = graded(bundle, [fwd[k] for k in keys])
    r = graded(bundle, [rev[k] for k in keys])
    both = sum(a["ok"] and b["ok"] for a, b in zip(f, r))
    only_r = sum(b["ok"] and not a["ok"] for a, b in zip(f, r))
    only_f = sum(a["ok"] and not b["ok"] for a, b in zip(f, r))
    return {"n": len(keys), "forward": _rate(f), "reversed": _rate(r),
            "both": both / len(keys), "reversed_only": only_r / len(keys),
            "forward_only": only_f / len(keys),
            "gap": _rate(r) - _rate(f)}


def carry_curve(bundle, *, limit: int = 400, per_width: int = 4_000,
                seed: int = 0) -> list[dict]:
    """Accuracy against the longest run of consecutive carries.

    Chains of 0 and 1 are in training at every width. A chain of 2 or more is
    in training at one and two digits and held out at three, which is the whole
    design: the same rule, shown at one size and asked for at another.
    """
    pool = [p for p in cur.problems(per_width=per_width, seed=seed)
            if p.task in ("add", "add_rev") and p.digits == cur.TRAIN_DIGITS]
    seen, uniq = set(), []
    for p in pool:
        if p.prompt not in seen:
            seen.add(p.prompt)
            uniq.append(p)
    out = []
    for task in ("add", "add_rev"):
        for chain in (0, 1, 2, 3):
            part = [p for p in uniq
                    if p.task == task and p.carry_chain == chain][:limit]
            if len(part) < 20:
                continue
            rows = graded(bundle, part)
            out.append({"task": task, "carry_chain": chain, "n": len(rows),
                        "held_out": chain >= 2, "accuracy": _rate(rows)})
    return out


def length_curve(bundle, *, limit: int = 300, per_width: int = 4_000,
                 seed: int = 0) -> list[dict]:
    """Accuracy against operand width, training stopping at three digits."""
    pool = cur.problems(per_width=per_width, seed=seed)
    seen, uniq = set(), []
    for p in pool:
        if p.prompt not in seen:
            seen.add(p.prompt)
            uniq.append(p)
    out = []
    for task in ("add", "add_rev", "sub", "mul"):
        for d in (1, 2, 3, 4):
            part = [p for p in uniq
                    if p.task == task and p.digits == d][:limit]
            if len(part) < 20:
                continue
            rows = graded(bundle, part)
            out.append({"task": task, "digits": d, "n": len(rows),
                        "held_out": d > cur.TRAIN_DIGITS,
                        "accuracy": _rate(rows)})
    return out


def multiplication_shape(bundle, *, limit: int = 200, per_width: int = 4_000,
                         seed: int = 0) -> list[dict]:
    """Multiplication accuracy by the width of *each* operand, separately.

    `length_curve` labels a problem by its widest operand, which for
    multiplication is misleading: the generator pairs a three-digit number with
    a one-digit one, so "three-digit multiplication" is an easier problem than
    "two-digit multiplication" and the curve appears to rise with length. It
    does not. The width that matters is the narrow one, and this table shows
    why by reporting the pair.
    """
    pool = cur.problems(per_width=per_width, seed=seed)
    seen, rows = set(), []
    for p in pool:
        if p.task != "mul" or p.prompt in seen:
            continue
        seen.add(p.prompt)
        a, b = p.prompt[:-1].split("*")
        rows.append((len(a), len(b), p))
    out = []
    for wa in (1, 2, 3, 4):
        for wb in (1, 2, 3, 4):
            part = [p for x, y, p in rows if x == wa and y == wb][:limit]
            if len(part) < 20:
                continue
            got = graded(bundle, part)
            out.append({"left": wa, "right": wb, "n": len(got),
                        "held_out": max(wa, wb) > cur.TRAIN_DIGITS,
                        "accuracy": _rate(got)})
    return out


def precedence(bundle, *, limit: int = 300, per_width: int = 4_000,
               seed: int = 0) -> dict:
    """The order-of-operations failure, split by what the wrong answer was.

    A wrong answer that equals the left-to-right reading is a model that did
    the arithmetic correctly and the *grammar* wrong, which is a different
    failure from one that simply miscalculated -- and it is the one a middle
    school teacher would recognise.
    """
    pool = [p for p in cur.corpus("test", per_width=per_width, seed=seed)
            if p.task == "order"][:limit]
    rows = graded(bundle, pool)
    left_to_right = wrong_shape = other = 0
    for r in rows:
        if r["ok"]:
            continue
        body = r["prompt"][:-1]
        if "+" in body and "*" in body:
            if body.index("+") < body.index("*"):
                a, rest = body.split("+", 1)
                b, c = rest.split("*")
                naive = (int(a) + int(b)) * int(c)
            else:
                a, rest = body.split("*", 1)
                b, c = rest.split("+")
                naive = int(a) * (int(b) + int(c))
        else:
            other += 1
            continue
        if r["got"] == str(naive):
            left_to_right += 1
        elif len(r["got"]) != len(r["want"]):
            wrong_shape += 1
        else:
            other += 1
    n = len(rows)
    return {"n": n, "accuracy": _rate(rows),
            "left_to_right": left_to_right / n,
            "wrong_length": wrong_shape / n, "other": other / n}


def precedence_explained(bundle, *, limit: int = 300, per_width: int = 4_000,
                         seed: int = 0) -> dict:
    """How much of the order-of-operations failure is just the multiplication.

    `a+b*c` contains a two-digit product, and two-digit products are the thing
    this model is worst at. So before calling 40% a failure of precedence, back
    the product out of the model's answer and ask whether the *rest* is right:
    if `got - a` is a clean product of two numbers near `b` and `c`, the model
    applied the rule and miscalculated inside it.
    """
    pool = [p for p in cur.corpus("test", per_width=per_width, seed=seed)
            if p.task == "order"][:limit]
    rows = graded(bundle, pool)
    consistent = addition_wrong = unexplained = 0
    for r in rows:
        if r["ok"] or not r["got"].lstrip("-").isdigit():
            continue
        body, got = r["prompt"][:-1], int(r["got"])
        if body.index("+") < body.index("*"):
            a, rest = body.split("+", 1)
            b, c = rest.split("*")
            implied, exact = got - int(a), int(b) * int(c)
        else:
            a, rest = body.split("*", 1)
            b, c = rest.split("+")
            implied, exact = got - int(c), int(a) * int(b)
        if implied == exact:
            addition_wrong += 1          # right product, wrong final sum
        elif 0 < implied < 10 * exact:
            consistent += 1              # wrong product, rule applied
        else:
            unexplained += 1
    n = len(rows)
    mul = [p for p in cur.corpus("test", per_width=per_width, seed=seed)
           if p.task == "mul" and len(p.prompt.split("*")[0]) == 2
           and len(p.prompt.split("*")[1][:-1]) == 2][:limit]
    return {"n": n, "accuracy": _rate(rows),
            "product_wrong_rule_right": consistent / n,
            "product_right_sum_wrong": addition_wrong / n,
            "unexplained": unexplained / n,
            "two_by_two_multiplication": _rate(graded(bundle, mul))}


def solution_range(bundle, *, limit: int = 900, per_width: int = 4_000,
                   seed: int = 0) -> dict:
    """Is the linear task solved, or is it classified?

    The unknown here is an integer in a narrow range, so a model that has
    learned the *joint distribution* of coefficient and constant can score well
    without ever inverting anything. If accuracy is flat in the size of the
    answer and in the coefficient, "solving" is the wrong word for it.
    """
    pool = [p for p in cur.corpus("test", per_width=per_width, seed=seed)
            if p.task == "linear"][:limit]
    rows = graded(bundle, pool)
    bands: dict[str, list] = {}
    for p, r in zip(pool, rows):
        m = int(p.prompt.split("x")[0])
        x = int(p.answer)
        bands.setdefault(f"|x| {abs(x) // 7 * 7}-{abs(x) // 7 * 7 + 6}",
                         []).append(r)
        bands.setdefault(f"coefficient {'2-6' if m <= 6 else '7-12'}",
                         []).append(r)
        bands.setdefault("negative x" if x < 0 else "non-negative x",
                         []).append(r)
    return {"n": len(rows), "accuracy": _rate(rows),
            "answers_possible": 41,
            "bands": {k: {"n": len(v), "accuracy": _rate(v)}
                      for k, v in sorted(bands.items())}}


def outside_the_range(bundle, *, count: int = 400, seed: int = 5) -> dict:
    """The control the linear score needs: equations whose answer is new.

    Training draws the unknown from -20 to 20, forty-one values. A model that
    scores 0.99 inside that set has not necessarily learned to invert an
    operation; it may have learned which of forty-one answers goes with which
    pair of numbers. Asking for an answer the set does not contain separates
    the two readings, and nothing else about the problem changes.
    """
    import numpy as np

    from standarderror.schoolmath.curriculum import Problem
    rng = np.random.default_rng(seed)
    out = {}
    for name, lo, hi in (("inside (-20..20)", -20, 21),
                         ("just outside (21..40)", 21, 41),
                         ("far outside (60..99)", 60, 100)):
        made = []
        for _ in range(count):
            m = int(rng.integers(2, 13))
            x = int(rng.integers(lo, hi))
            if rng.integers(2) and lo < 0:
                x = -x
            n = int(rng.integers(-99, 100))
            const = f"+{n}" if n >= 0 else f"-{abs(n)}"
            made.append(Problem(task="linear",
                                prompt=f"{m}x{const}={m * x + n}>x=",
                                answer=str(x), level="고등", digits=2,
                                carry_chain=0, key=f"linear:{m}x{const}"))
        out[name] = _rate(graded(bundle, made))
    return out


def first_error(bundle, task: str = "add", *, split: str = "probe_carry",
                limit: int = 600, per_width: int = 4_000,
                seed: int = 0) -> dict:
    """Where in the answer the first wrong character appears.

    Reported from the *left*, as written, and also as a position in the
    schoolbook algorithm -- which for a forward answer is the other end. A
    model that fails on the leading digit is failing at the step that needs
    every carry decided in advance.
    """
    pool = [p for p in cur.corpus(split, per_width=per_width, seed=seed)
            if p.task == task][:limit]
    rows = graded(bundle, pool)
    bad = [r for r in rows if not r["ok"]]
    written, lengths = {}, 0
    for r in bad:
        want, got = r["want"], r["got"]
        pos = next((i for i in range(max(len(want), len(got)))
                    if i >= len(want) or i >= len(got)
                    or want[i] != got[i]), 0)
        written[pos] = written.get(pos, 0) + 1
        lengths += len(want)
    n = len(bad) or 1
    return {"split": split, "n": len(rows), "wrong": len(bad),
            "accuracy": _rate(rows),
            "mean_answer_length": lengths / n,
            "first_wrong_position": {k: v / n for k, v in sorted(
                written.items())},
            "leading_digit_share": written.get(0, 0) / n,
            "right_length": sum(len(r["got"]) == len(r["want"])
                                for r in bad) / n}


def padding_sensitivity(bundle, *, limit: int = 400, per_width: int = 4_000,
                        seed: int = 0) -> dict:
    """The control: does the accuracy depend on what precedes the prompt?

    If it does, a failure is partly an artefact of feeding the model a prompt
    at an alignment it rarely saw, and every number above would need a caveat.
    """
    out = {}
    for task in ("add", "add_rev"):
        pool = [p for p in cur.corpus("test", per_width=per_width, seed=seed)
                if p.task == task][:limit]
        fixed = "".join(_lines()[:14])
        arms = {
            "fresh context": fillers(len(pool), seed=0),
            "a different fresh context": fillers(len(pool), seed=7),
            "one shared context": fixed,
            "no context": "",
        }
        got = {k: _rate(graded(bundle, pool, filler=v))
               for k, v in arms.items()}
        got["fresh spread"] = abs(got["fresh context"]
                                  - got["a different fresh context"])
        out[task] = got
    return out


# ---------------------------------------------------------------- episode 1

def twin_adjacency(*, per_width: int = 4_000, seed: int = 0) -> dict:
    """How often a reversed line sits directly beneath its own forward twin.

    In generation order this is every one of them, which is what made the
    shortcut available. It is a property of the training file, measurable
    without running the model at all, and it is the thing that should have
    been checked first.
    """
    out = {}
    for name, shuffle in (("generation order", False), ("shuffled", True)):
        lines = cur.text("train", per_width=per_width, seed=seed,
                         shuffle=shuffle).split("\n")
        reversed_lines = sum(1 for ln in lines if cur.REVERSED in ln)
        adjacent = sum(
            1 for a, b in zip(lines, lines[1:])
            if cur.REVERSED in b
            and a.split("=")[0] == b.split(cur.REVERSED)[0])
        out[name] = {"reversed_lines": reversed_lines, "adjacent": adjacent,
                     "share": adjacent / reversed_lines}
    return out


def _twin_context(p, *, corrupt: bool = False, other: bool = False,
                  rng=None) -> str:
    """Context whose last line is the forward form of `p`'s own sum.

    `corrupt` writes a *wrong* forward answer, which is the control that
    separates copying from adding: a model that reverses whatever is above it
    will reverse the wrong number too.
    """
    left = p.prompt.split(cur.REVERSED)[0]
    a, b = (int(v) for v in left.split("+"))
    if other:
        a, b = a + 1, b + 2
    total = a + b
    if corrupt:
        total = total + 111 if len(str(total + 111)) == len(str(total)) \
            else total + 1
    filler = "".join(_lines()[j] for j in rng.integers(0, len(_lines()), 6))
    return f"{filler}{a}+{b}={total}\n"


def shortcut(clean, leaky, *, limit: int = 250, per_width: int = 4_000,
             seed: int = 0) -> dict:
    """The A/B. Two models, one manipulated variable: the order of the lines.

    Four contexts for the same reversed sums. If a model is adding, all four
    agree. If it is reversing the line above, only the first works -- and the
    corrupted arm proves which, because there the line above is wrong and the
    right answer is still available by adding.
    """
    import numpy as np
    pool = [p for p in cur.corpus("test", per_width=per_width, seed=seed)
            if p.task == "add_rev" and p.digits >= 2][:limit]
    rng = np.random.default_rng(99)
    arms = {
        "its own sum on the line above":
            [_twin_context(p, rng=rng) for p in pool],
        "a wrong sum on the line above":
            [_twin_context(p, corrupt=True, rng=rng) for p in pool],
        "a different sum on the line above":
            [_twin_context(p, other=True, rng=rng) for p in pool],
        "ordinary context": fillers(len(pool), seed=1),
    }
    out = {"n": len(pool)}
    for name, bundle in (("shuffled", clean), ("generation order", leaky)):
        out[name] = {k: _rate(graded(bundle, pool, filler=v))
                     for k, v in arms.items()}
    # Shown a wrong premise, what does the leaky model do? The decisive
    # number is not how often it copies but how that compares with how often
    # it gets the answer right -- the right answer was always available by
    # adding, and the two rates are what separate the readings.
    got = graded(leaky, pool, filler=arms["a wrong sum on the line above"])
    copied = correct = 0
    for p, r in zip(pool, got):
        left = p.prompt.split(cur.REVERSED)[0]
        a, b = (int(v) for v in left.split("+"))
        shown = a + b + 111 if len(str(a + b + 111)) == len(str(a + b)) \
            else a + b + 1
        copied += r["got"] == str(shown)[::-1]
        correct += r["got"] == p.answer
    out["generation order"]["reproduced the wrong sum, reversed"] = \
        copied / len(pool)
    out["generation order"]["answered correctly anyway"] = correct / len(pool)
    return out


def held_out_loss(bundle, *, shuffle: bool = True, per_width: int = 4_000,
                  seed: int = 0, batches: int = 40, size: int = 32) -> float:
    """Cross-entropy on the held-out stream, packed one way or the other.

    `shuffle` matters more than it looks. A validation set is conventionally
    packed the same way as the training set, so each model's *own* reported
    loss is the one where `shuffle` matches how it was trained -- and that is
    the number a practitioner sees. Scoring both models on the same stream is
    the diagnostic, not the default, and it is the whole finding: repacking
    the validation data in a different order is what exposes the shortcut,
    and nothing in an ordinary pipeline ever does it.
    """
    import numpy as np
    import torch
    text = cur.text("test", per_width=per_width, seed=seed, shuffle=shuffle)
    stoi = bundle["stoi"]
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    rng = np.random.default_rng(0)
    block, losses = sm.BLOCK, []
    with torch.no_grad():
        for _ in range(batches):
            i = rng.integers(0, len(data) - block - 1, size)
            x = torch.stack([data[j:j + block] for j in i])
            y = torch.stack([data[j + 1:j + block + 1] for j in i])
            losses.append(float(bundle["model"](x, y)[1]))
    return float(np.mean(losses))


def loss_table(clean, leaky, **kw) -> dict:
    """Each model on each packing of the held-out stream.

    The diagonal is what each pipeline reports about itself. The off-diagonal
    is what you only see if you thought to repack.
    """
    out = {}
    for name, bundle in (("shuffled", clean), ("generation order", leaky)):
        out[name] = {
            "own packing": held_out_loss(
                bundle, shuffle=(name == "shuffled"), **kw),
            "repacked": held_out_loss(
                bundle, shuffle=(name != "shuffled"), **kw),
        }
    return out
