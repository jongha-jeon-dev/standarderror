r"""School arithmetic as a character corpus, with splits that test the rule
rather than the lookup table.

The point of this corpus is not that a small transformer can add. It is that
*where* it stops being able to add lines up, almost exactly, with the places a
school curriculum stops and introduces a new idea: carrying, place value, the
order of operations, inverting an operation. So the corpus is organised by
those ideas, and the held-out sets are structural rather than random.

Three kinds of hold-out, and the distinction matters for every claim made
with them:

`test`
    A random tenth of the problems, chosen by a stable hash of the problem
    text. Same distribution as training, never seen. This measures whether the
    model learned *anything* beyond memorising the training lines.

`probe_carry`
    Every three-digit addition whose carries propagate -- a carry out of the
    units that forces a carry out of the tens. The model sees propagation at
    two digits, where it is unavoidable, and never at three. This measures
    whether "carry" was learned as a rule or as a length-specific habit.

`probe_length`
    Four-digit operands, when training stopped at three. The usual length
    generalisation question, kept separate from the carry question so that a
    failure can be attributed to one or the other.

Answer order is a task, not a formatting detail. `12+345=357` writes the sum
the way a person reads it; `12+345~753` writes the same sum's digits in the
order a person *computes* them, least significant first. Both appear in the
corpus under different glyphs, so one model's two accuracies are a within-model
comparison rather than a comparison of two training runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

#: Every character the corpus can emit. Fixed here rather than derived from a
#: sample, so the vocabulary cannot silently change when a generator does.
ALPHABET = "\n0123456789=~+-*>x"

#: The answer glyph that means "least significant digit first".
REVERSED = "~"

#: Random-hold-out share, and the digit length at which carrying stops being
#: taught.
TEST_SHARE = 0.10
TRAIN_DIGITS = 3
PROBE_DIGITS = 4


@dataclass(frozen=True)
class Problem:
    """One line of the corpus, plus everything needed to grade it."""
    task: str
    prompt: str          # includes the answer glyph; what the model is fed
    answer: str          # what must follow, up to but excluding the newline
    level: str           # 초등 / 중등 / 고등, as a plain ascii label
    digits: int          # operand width, for the length probe
    carry_chain: int     # longest run of consecutive carries, 0 if not an add
    key: str             # the *sum*, not the notation: what the split hashes

    @property
    def line(self) -> str:
        return f"{self.prompt}{self.answer}\n"


def _hash_share(text: str) -> float:
    """A stable float in [0, 1) from the problem text.

    Stable across processes and platforms, which `hash()` is not, so the split
    a reader reproduces is the split the published numbers used.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def carry_chain(a: int, b: int) -> int:
    """Longest run of consecutive carries in the schoolbook addition a + b.

    This is the quantity a child is taught as "carry the one, and if that makes
    ten, carry again". A chain of 1 is a single carry; a chain of 2 or more is
    a carry that *propagates*, which is the step where the algorithm stops
    being digit-local.
    """
    best = run = 0
    carry = 0
    while a or b or carry:
        total = a % 10 + b % 10 + carry
        carry = 1 if total >= 10 else 0
        run = run + 1 if carry else 0
        best = max(best, run)
        a //= 10
        b //= 10
    return best


def digits_of(*values: int) -> int:
    return max(len(str(abs(v))) for v in values)


def _add_pair(a: int, b: int) -> list[Problem]:
    """The same sum written twice: once to be read, once to be computed.

    Both problems carry identical metadata, so any difference between the two
    accuracies is the answer order and nothing else.
    """
    total = str(a + b)
    # Both forms share a key, so the split cannot put one in training and the
    # other in test. Hashing the prompt instead would leak the sum across the
    # boundary under a different glyph, and the reversed accuracy would then
    # be measured partly on sums the model had already been shown.
    common = dict(level="초등", digits=digits_of(a, b),
                  carry_chain=carry_chain(a, b), key=f"add:{a}+{b}")
    return [
        Problem(task="add", prompt=f"{a}+{b}=", answer=total, **common),
        Problem(task="add_rev", prompt=f"{a}+{b}{REVERSED}",
                answer=total[::-1], **common),
    ]


def _generate(rng, count: int, digits: int) -> list[Problem]:
    """`count` problems of each task at operand width `digits`.

    Operands are drawn uniformly over the full width rather than over all
    widths up to it, so "three-digit addition" means what it says and the
    length probe is not diluted by easier cases.
    """
    lo, hi = (0, 9) if digits == 1 else (10 ** (digits - 1), 10 ** digits - 1)
    out: list[Problem] = []
    for _ in range(count):
        a, b = int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))
        out.extend(_add_pair(a, b))

        # Subtraction, signed. The negative case is the point: it is where a
        # curriculum stops being able to say "take the smaller from the
        # bigger" and has to introduce a number line.
        c, d = int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))
        out.append(Problem(task="sub", prompt=f"{c}-{d}=", answer=str(c - d),
                           level="초등" if c >= d else "중등",
                           digits=digits_of(c, d), carry_chain=0,
                           key=f"sub:{c}-{d}"))

        # Multiplication is deliberately capped narrower than addition: the
        # product of two d-digit numbers has up to 2d digits, and the interest
        # is in the algorithm rather than in how long an answer can get.
        e = int(rng.integers(lo, hi + 1))
        f = int(rng.integers(0, 10)) if digits > 2 else int(rng.integers(lo,
                                                                        hi + 1))
        out.append(Problem(task="mul", prompt=f"{e}*{f}=", answer=str(e * f),
                           level="초등" if f < 10 else "중등",
                           digits=digits_of(e, f), carry_chain=0,
                           key=f"mul:{e}*{f}"))

        # Order of operations, with both orders represented so that getting it
        # right cannot be "always multiply the last two".
        g, h, i = (int(rng.integers(2, 100)) for _ in range(3))
        if rng.integers(2):
            prompt, value = f"{g}+{h}*{i}=", g + h * i
        else:
            prompt, value = f"{g}*{h}+{i}=", g * h + i
        out.append(Problem(task="order", prompt=prompt, answer=str(value),
                           level="중등", digits=2, carry_chain=0,
                           key=f"order:{prompt}"))

        # One linear equation in one unknown, with an integer solution. This
        # is the first task on the list that requires undoing an operation
        # rather than performing one.
        m = int(rng.integers(2, 13))
        x = int(rng.integers(-20, 21))
        n = int(rng.integers(-99, 100))
        # Written the way it would be on a board: `2x-45=...`, never
        # `2x+-45=...`. The sign is part of the notation being taught.
        const = f"+{n}" if n >= 0 else f"-{abs(n)}"
        out.append(Problem(task="linear",
                           prompt=f"{m}x{const}={m * x + n}>x=",
                           answer=str(x), level="고등", digits=2,
                           carry_chain=0, key=f"linear:{m}x{const}"))
    return out


def split_of(p: Problem) -> str:
    """Which set a problem belongs to. Structural rules beat the random one.

    Order matters here. A four-digit addition with a propagating carry is a
    length probe, not a carry probe, because it differs from training in two
    ways at once and only the first is the question being asked.
    """
    if p.digits > TRAIN_DIGITS:
        return "probe_length"
    if p.task in ("add", "add_rev") and p.digits == TRAIN_DIGITS \
            and p.carry_chain >= 2:
        return "probe_carry"
    if _hash_share(p.key) < TEST_SHARE:
        return "test"
    return "train"


def problems(*, per_width: int = 20_000, seed: int = 0) -> list[Problem]:
    """The whole generated pool, before splitting.

    Deterministic in `seed`, so the corpus is reproducible without committing
    it -- the same argument the language model's corpus is fetched under.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    out: list[Problem] = []
    for digits in (1, 2, TRAIN_DIGITS, PROBE_DIGITS):
        # The probe width needs only enough problems to measure on.
        n = per_width if digits <= TRAIN_DIGITS else per_width // 8
        out.extend(_generate(rng, n, digits))
    return out


def corpus(split: str = "train", *, per_width: int = 20_000,
           seed: int = 0) -> list[Problem]:
    """One split of the pool, de-duplicated on the problem text.

    De-duplication is not cosmetic. Without it the same easy sum appears in
    both a training batch and a test batch under different draws, and the test
    accuracy stops meaning anything.
    """
    seen: set[str] = set()
    out = []
    for p in problems(per_width=per_width, seed=seed):
        if p.prompt in seen or split_of(p) != split:
            continue
        seen.add(p.prompt)
        out.append(p)
    return out


def text(split: str = "train", *, shuffle: bool = True,
         order_seed: int = 17, **kw) -> str:
    """A split as one newline-separated stream, ready to be chunked.

    `shuffle` is not a detail. Generated in order, every `add_rev` line lands
    directly beneath the `add` line for the same sum:

        12+345=357
        12+345~753

    and "reverse the answer on the line above" then solves the reversed task
    without any arithmetic at all. A model trained on that stream scores 0.99
    on reversed sums when the prompt starts a fresh context and 0.00 when
    anything real precedes it, and the test loss -- 1.22 against a uniform
    2.89 -- reports nothing wrong at any point, because the shortcut is
    available on the test split too.

    Shuffling puts the twins far apart, so the only way to answer is to add.
    Pass `shuffle=False` to rebuild the leaky stream and watch it happen.
    """
    import numpy as np
    rows = corpus(split, **kw)
    if shuffle:
        idx = np.random.default_rng(order_seed).permutation(len(rows))
        rows = [rows[i] for i in idx]
    return "".join(p.line for p in rows)
