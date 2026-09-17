"""The arithmetic corpus, its hold-outs, and the grader.

The curriculum tests are exact: a carry chain is a fact about two integers, a
split is a function of the problem text, and both are checked against
independent reference implementations rather than against themselves. The
model tests are marked slow and pin accuracies as inequalities, because they
are properties of one checkpoint.
"""
import pytest

from standarderror.schoolmath import curriculum as cur


def reference_chain(a: int, b: int) -> int:
    """Carry chain computed from the decimal strings, the long way round.

    Deliberately not the implementation: zero-padded string arithmetic rather
    than repeated division, so a bug in one is unlikely to be a bug in both.
    """
    x, y = str(a)[::-1], str(b)[::-1]
    width = max(len(x), len(y)) + 1
    x, y = x.ljust(width, "0"), y.ljust(width, "0")
    best = run = carry = 0
    for i in range(width):
        total = int(x[i]) + int(y[i]) + carry
        carry = 1 if total >= 10 else 0
        run = run + 1 if carry else 0
        best = max(best, run)
    return best


class TestCarryingIsComputedCorrectly:

    @pytest.mark.parametrize("a,b,chain", [
        (1, 2, 0),            # nothing to carry
        (5, 5, 1),            # one carry, out of the units
        (15, 5, 1),           # the carry lands and stops
        (95, 5, 2),           # units carry forces the tens to carry
        (999, 1, 3),          # the full cascade
        (100, 100, 0),        # long numbers, no carry at all
    ])
    def test_against_hand_worked_cases(self, a, b, chain):
        assert cur.carry_chain(a, b) == chain

    def test_against_an_independent_implementation(self):
        import numpy as np
        rng = np.random.default_rng(0)
        for _ in range(3000):
            a, b = int(rng.integers(0, 10000)), int(rng.integers(0, 10000))
            assert cur.carry_chain(a, b) == reference_chain(a, b)

    def test_it_is_symmetric(self):
        import numpy as np
        rng = np.random.default_rng(1)
        for _ in range(500):
            a, b = int(rng.integers(0, 1000)), int(rng.integers(0, 1000))
            assert cur.carry_chain(a, b) == cur.carry_chain(b, a)


class TestTheHoldOutsAreCleanHoldOuts:

    @staticmethod
    @pytest.fixture(scope="class")
    def pool():
        return cur.problems(per_width=2000, seed=0)

    def test_every_problem_lands_in_exactly_one_split(self, pool):
        for p in pool[:2000]:
            assert cur.split_of(p) in {"train", "test", "probe_carry",
                                       "probe_length"}

    def test_the_split_is_a_function_of_the_text_not_the_draw(self, pool):
        """Two runs of the generator must agree, or a reader reproducing the
        corpus gets different training data from the published model."""
        again = cur.problems(per_width=2000, seed=0)
        assert [cur.split_of(p) for p in pool[:500]] == \
               [cur.split_of(p) for p in again[:500]]

    def test_no_training_prompt_appears_in_any_other_split(self, pool):
        train = {p.prompt for p in pool if cur.split_of(p) == "train"}
        for name in ("test", "probe_carry", "probe_length"):
            other = {p.prompt for p in pool if cur.split_of(p) == name}
            assert not (train & other), f"{name} leaks into train"

    def test_training_never_shows_a_propagating_carry_at_three_digits(self,
                                                                     pool):
        for p in pool:
            if cur.split_of(p) != "train":
                continue
            if p.task in ("add", "add_rev") and p.digits == cur.TRAIN_DIGITS:
                assert p.carry_chain < 2

    def test_training_does_show_propagating_carries_at_two_digits(self, pool):
        """Otherwise the carry probe would be asking for a rule the model was
        never taught at any width, which is a different question."""
        two = [p for p in pool if cur.split_of(p) == "train"
               and p.task == "add" and p.digits == 2 and p.carry_chain >= 2]
        assert len(two) > 50

    def test_training_stops_at_three_digits(self, pool):
        for p in pool:
            if cur.split_of(p) == "train":
                assert p.digits <= cur.TRAIN_DIGITS

    def test_a_sum_lands_in_one_split_whichever_way_it_is_written(self, pool):
        """The leak this caught: hashing the prompt put `12+345=` in training
        and `12+345~` in test, so the reversed accuracy was measured partly on
        sums the model had already been taught. Hashing the sum fixes it, and
        makes the forward/reversed comparison exactly paired as a side effect.
        """
        fwd = {p.key: cur.split_of(p) for p in pool if p.task == "add"}
        rev = {p.key: cur.split_of(p) for p in pool if p.task == "add_rev"}
        shared = set(fwd) & set(rev)
        assert len(shared) > 1000
        assert all(fwd[k] == rev[k] for k in shared)

    def test_the_length_probe_is_only_length(self, pool):
        """A four-digit sum with a propagating carry differs from training in
        two ways, so it must be filed under length and not under carrying."""
        for p in pool:
            if cur.split_of(p) == "probe_carry":
                assert p.digits == cur.TRAIN_DIGITS


class TestTheCorpusSaysWhatItMeans:

    def test_every_answer_is_arithmetically_true(self):
        import operator
        ops = {"+": operator.add, "-": operator.sub, "*": operator.mul}
        checked = 0
        for p in cur.problems(per_width=1500, seed=3):
            if p.task in ("add", "sub", "mul"):
                glyph = next(g for g in ops if g in p.prompt[1:])
                left, right = p.prompt[:-1].split(glyph, 1)
                # `-` also starts a negative operand, so rebuild from the
                # split rather than trusting the first occurrence blindly.
                if not right:
                    continue
                assert str(ops[glyph](int(left), int(right))) == p.answer
                checked += 1
        assert checked > 1000

    def test_the_reversed_task_is_the_same_sum_backwards(self):
        pool = cur.problems(per_width=1500, seed=4)
        fwd = {p.prompt[:-1]: p.answer for p in pool if p.task == "add"}
        rev = {p.prompt[:-1]: p.answer for p in pool if p.task == "add_rev"}
        assert len(set(fwd) & set(rev)) > 2000
        for k in list(set(fwd) & set(rev))[:2000]:
            assert rev[k] == fwd[k][::-1]

    def test_order_of_operations_is_not_left_to_right(self):
        """If every `order` answer also equalled the left-to-right reading,
        the task would not test what it is named after."""
        pool = [p for p in cur.problems(per_width=1500, seed=5)
                if p.task == "order"]
        differs = 0
        for p in pool:
            body = p.prompt[:-1]
            if "+" in body and "*" in body:
                a, rest = body.split("+", 1) if body.index("+") < \
                    body.index("*") else (None, None)
                if a is None:
                    continue
                b, c = rest.split("*")
                if str((int(a) + int(b)) * int(c)) != p.answer:
                    differs += 1
        assert differs > 100

    def test_the_linear_equations_have_the_solution_they_claim(self):
        checked = 0
        for p in cur.problems(per_width=1500, seed=6):
            if p.task != "linear":
                continue
            body = p.prompt.split(">")[0]
            left, right = body.split("=")
            m, const = left.split("x")
            assert const[0] in "+-", "the constant must carry its own sign"
            assert int(m) * int(p.answer) + int(const) == int(right)
            checked += 1
        assert checked > 1000

    def test_no_equation_is_written_with_a_doubled_sign(self):
        """`2x+-45=` parses, and would still be wrong to print in front of a
        student. The notation is part of what the corpus is teaching."""
        for p in cur.problems(per_width=1500, seed=6):
            assert "+-" not in p.prompt and "--" not in p.prompt

    def test_the_alphabet_covers_the_corpus_exactly(self):
        used = set()
        for p in cur.problems(per_width=1500, seed=7):
            used |= set(p.line)
        assert used <= set(cur.ALPHABET)
        assert set(cur.ALPHABET) - used == set(), \
            "an unused character costs parameters and hides a dead branch"

    def test_splits_are_deduplicated(self):
        rows = cur.corpus("test", per_width=1500, seed=0)
        prompts = [p.prompt for p in rows]
        assert len(prompts) == len(set(prompts))

    def test_the_reversed_twin_never_sits_under_its_own_sum(self):
        """The shortcut this caught: generated in order, `12+345~753` lands
        directly beneath `12+345=357`, and reversing the line above solves the
        task with no arithmetic. Every twin was adjacent; the test loss showed
        nothing, because the shortcut works on the test split too."""
        for shuffle, allowed in ((True, 0), (False, 10**9)):
            lines = cur.text("train", per_width=300, seed=0,
                             shuffle=shuffle).split("\n")
            adjacent = sum(
                1 for a, b in zip(lines, lines[1:])
                if cur.REVERSED in b and a.split("=")[0] == b.split(
                    cur.REVERSED)[0])
            assert adjacent <= allowed
            if not shuffle:
                assert adjacent > 100, "the unshuffled stream should be leaky"

    def test_shuffling_changes_the_order_and_nothing_else(self):
        a = cur.text("test", per_width=400, seed=0, shuffle=True)
        b = cur.text("test", per_width=400, seed=0, shuffle=False)
        assert a != b
        assert sorted(a.split("\n")) == sorted(b.split("\n"))

    def test_the_stream_is_one_problem_per_line(self):
        body = cur.text("test", per_width=400, seed=0)
        lines = [ln for ln in body.split("\n") if ln]
        assert all("=" in ln or cur.REVERSED in ln for ln in lines)
        assert body.endswith("\n")


torch = pytest.importorskip("torch")

from standarderror.schoolmath import arithmetic as ar  # noqa: E402
from standarderror.schoolmath import model as sm  # noqa: E402

slow = pytest.mark.skipif(
    not __import__("os").environ.get("SERR_SLOW_TESTS"),
    reason="decodes thousands of answers; set SERR_SLOW_TESTS=1")


@pytest.fixture(scope="module")
def bundle():
    return sm.load()


class TestTheCheckpointIsTheOneThatWasMeasured:

    def test_it_loads_and_verifies_its_hash(self, bundle):
        assert bundle["vocab"] == len(cur.ALPHABET) == 18
        assert bundle["parameters"] == 804_096

    def test_it_is_the_language_model_architecture_exactly(self, bundle):
        """Same four blocks, four heads, width 128, context 64. Only the
        vocabulary differs, so anything the calculus series measured on the
        other model can be measured here without a caveat about depth."""
        from standarderror.llm import tiny
        reference = tiny.build(cur.ALPHABET.__len__())
        assert [n for n, _ in bundle["model"].named_parameters()] == \
               [n for n, _ in reference.named_parameters()]

    def test_the_learning_curve_was_recorded(self, bundle):
        import torch as _t
        blob = _t.load(sm.checkpoint_path(), map_location="cpu",
                       weights_only=False)
        assert len(blob["history"]) >= 20
        assert blob["history"][0]["add"] < blob["history"][-1]["add"]


@slow
class TestWhatItLearnedAndWhatItDidNot:

    def test_it_can_add_subtract_in_distribution(self, bundle):
        got = {r["task"]: r["accuracy"]
               for r in ar.by_task(bundle, limit=200, per_width=3_000)}
        assert got["add"] > 0.95
        assert got["add_rev"] > 0.95
        assert got["sub"] > 0.90

    def test_multiplication_is_limited_by_the_narrow_operand(self, bundle):
        """Not by the wide one. Three-digit multiplication scores *above*
        two-digit here, because the generator pairs three digits with one."""
        rows = {(r["left"], r["right"]): r["accuracy"]
                for r in ar.multiplication_shape(bundle, limit=100,
                                                 per_width=3_000)}
        assert rows[(3, 1)] > rows[(2, 2)] + 0.2
        assert rows[(2, 2)] < 0.75

    def test_carrying_was_learned_at_a_width_not_as_a_rule(self, bundle):
        rows = {(r["task"], r["carry_chain"]): r["accuracy"]
                for r in ar.carry_curve(bundle, limit=150, per_width=3_000)}
        assert rows[("add", 0)] > 0.95 and rows[("add", 1)] > 0.95
        assert rows[("add", 2)] < 0.80
        assert rows[("add", 3)] < rows[("add", 2)]

    def test_the_reversed_answer_carries_better_than_the_forward_one(self,
                                                                    bundle):
        """Same model, same sums, same held-out carries. The only difference
        is which end of the answer comes first."""
        rows = {(r["task"], r["carry_chain"]): r["accuracy"]
                for r in ar.carry_curve(bundle, limit=150, per_width=3_000)}
        assert rows[("add_rev", 2)] > rows[("add", 2)]
        assert rows[("add_rev", 3)] > rows[("add", 3)]

    def test_a_fourth_digit_ends_it(self, bundle):
        rows = {(r["task"], r["digits"]): r["accuracy"]
                for r in ar.length_curve(bundle, limit=120, per_width=3_000)}
        for task in ("add", "add_rev", "sub"):
            assert rows[(task, 4)] < 0.02

    def test_the_order_failure_is_the_product_not_the_precedence(self, bundle):
        """The hypothesis was that it would apply the operations left to
        right, which is the mistake a person makes. It does not make that
        mistake at all; every failure is a wrong two-digit product inside a
        correctly applied rule."""
        got = ar.precedence(bundle, limit=200, per_width=3_000)
        assert got["left_to_right"] < 0.02
        explained = ar.precedence_explained(bundle, limit=200,
                                            per_width=3_000)
        assert explained["unexplained"] < 0.05

    def test_the_linear_score_is_classification_not_algebra(self, bundle):
        """0.98 inside the forty-one answers training drew from, and near zero
        one step outside it. Nothing about the equations changed."""
        got = ar.outside_the_range(bundle, count=200)
        assert got["inside (-20..20)"] > 0.95
        assert got["just outside (21..40)"] < 0.15
        assert got["far outside (60..99)"] < 0.05

    def test_the_context_control_is_flat(self, bundle):
        """Two independent samples of context must agree, or every number
        above is partly a statement about what preceded the prompt."""
        got = ar.padding_sensitivity(bundle, limit=150, per_width=3_000)
        for task in ("add", "add_rev"):
            assert got[task]["fresh spread"] < 0.05
