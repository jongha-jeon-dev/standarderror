"""ML 53: The Prompt Is Part of the Score.

The fifty-third episode of *Machine Learning, Taught Through What Breaks*,
in Arc XI. "A benchmark score measures the model." Measured on two of the
repository's arithmetic transformers, the same 200 problems per task asked
in six kinds of context:

* The main model's multiplication score was 0.735 with twelve lines of
  ordinary context in front of the problem and 0.500 with none.
* The gap between the main model and its leaky twin depended on the
  context: on three-digit carries it was 0.127 in the twin's favour with
  twelve lines of context and 0.012 with two; on multiplication it ranged
  from 0.05 to 0.168 the other way.
* Examples with wrong answers in front of the problem changed nothing
  measurable; which task the context lines came from did matter, for the
  twin.
* Picking the best of sixteen fixed prompts on 150 items reported a gain
  that disappeared on 150 new items for the main model's multiplication
  (best 0.793 on the first set, 0.667 on the second, against a mean of
  0.680), and held for its twin on carries, where prompts really differed.

Run: `standarderror run ml153_prompt --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import llmeval as le
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.schoolmath import model as sm
from standarderror.viz import charts

POST_DATE = date(2026, 10, 14)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

CASES = {"test/mul": "multiply", "test/order": "order of operations",
         "probe_carry/add": "3-digit carries", "test/linear": "linear equations",
         "test/add_rev": "reversed sums"}
KIND_LABEL = {"mixed": "12 random lines", "empty": "nothing",
              "two": "2 random lines", "same": "12 lines, same task",
              "other": "12 lines, other task",
              "wrong": "12 same-task, wrong answers"}
PREFIX_CASES = [("test", "mul"), ("test", "order"), ("probe_carry", "add")]


def compute() -> dict:
    models = {"clean": sm.load(), "leaky": sm.load(adjacent=True)}
    fmt = le.context_formats(models)
    pre = le.prefix_search(models, PREFIX_CASES)
    return {"fmt": fmt, "pre": pre}


def _mean(fmt, case, kind, name):
    return float(np.mean(fmt[f"{case}/{kind}/{name}"]))


def _gaps(fmt, case):
    return {k: _mean(fmt, case, k, "clean") - _mean(fmt, case, k, "leaky")
            for k in le.FORMATS}


def _prefix(pre, case, name):
    a = np.array(pre[f"{case}/{name}"])
    best = int(a[:, 0].argmax())
    return {"dev_mean": float(a[:, 0].mean()), "dev_min": float(a[:, 0].min()),
            "dev_max": float(a[:, 0].max()), "test_mean": float(a[:, 1].mean()),
            "best_dev": float(a[best, 0]), "best_test": float(a[best, 1]),
            "test_min": float(a[:, 1].min()), "test_max": float(a[:, 1].max()),
            "corr": float(np.corrcoef(a.T)[0, 1])}


# ------------------------------------------------------------------ figures

def _gap_plot(ax, m, fmt):
    cases = [c for c in CASES if c != "test/add_rev"]
    for i, c in enumerate(cases):
        g = _gaps(fmt, c)
        vals = [g[k] for k in le.FORMATS]
        ax.plot([min(vals), max(vals)], [i, i], color=m.grid, lw=6,
                solid_capstyle="round")
        for j, k in enumerate(le.FORMATS):
            ax.scatter([g[k]], [i], color=m.series[j], s=34, zorder=3,
                       label=KIND_LABEL[k] if i == 0 else None)
    ax.axvline(0, color=m.ink, lw=1)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([CASES[c] for c in cases], fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)


def figures(res: dict) -> dict:
    fmt, pre = res["fmt"], res["pre"]
    out: dict = {}
    g = _gaps(fmt, "probe_carry/add")
    out["f0"] = charts.diagram(
        lambda ax, m: _gap_plot(ax, m, fmt),
        title="How much better is the main model? It depends how you ask",
        subtitle=("Main model's accuracy minus its leaky twin's, on the same "
                  "200 problems per task, under six kinds of context before "
                  "the problem; each dot averages three context draws."),
        xlabel="main model minus leaky twin, accuracy", ylabel="",
        figsize=(7.6, 4.2),
        source="standarderror/ml/llmeval.py; models in data/schoolmath.",
        alt=("Four rows of six dots. Three-digit carries span -0.13 to "
             "-0.01; multiplication 0.05 to 0.17; order of operations 0.00 "
             "to 0.06; linear equations 0.02 to 0.13."),
        caption=(f"On three-digit carries the twin led by "
                 f"{-g['mixed']:.3f} with twelve lines of context and by "
                 f"**{-g['two']:.3f}** with two."),
        path=str(IMG / f"ml153-f0-gaps.{EXT}"))[0]

    body = []
    for c, cl in CASES.items():
        for name, nl in (("clean", "main"), ("leaky", "twin")):
            body.append([cl, nl] + [f"{_mean(fmt, c, k, name):.3f}"
                                    for k in le.FORMATS])
    out["f1"] = charts.table_image(
        body,
        header=["task", "model"] + [KIND_LABEL[k] for k in le.FORMATS],
        title="Six contexts, same problems",
        subtitle=("Exact-match accuracy on 200 problems per task, averaged "
                  "over three context draws (one for 'nothing'). The first "
                  "column is how every earlier school-maths number was "
                  "measured."),
        source="standarderror/ml/llmeval.py.",
        alt=("A ten-row table. 'Nothing' is lowest for the main model on "
             "multiplication and carries; the leaky twin's reversed sums jump from 0.005 to "
             "0.435 with no context."),
        caption="Context lines are training lines; 'wrong answers' changes the last digit of each.",
        align="llrrrrrr",
        path=str(IMG / f"ml153-f1-table.{EXT}"))[0]

    body = []
    for split, task in PREFIX_CASES:
        c = f"{split}/{task}"
        for name, nl in (("clean", "main"), ("leaky", "twin")):
            p = _prefix(pre, c, name)
            body.append([CASES[c], nl, f"{p['dev_min']:.3f}-{p['dev_max']:.3f}",
                         f"{p['best_dev']:.3f}", f"{p['best_test']:.3f}",
                         f"{p['test_mean']:.3f}", f"{p['corr']:+.2f}"])
    out["f2"] = charts.table_image(
        body,
        header=["task", "model", "16 prompts, set A", "best on A",
                "best, on B", "mean on B", "A-B corr."],
        title="Choosing the prompt on the items you report",
        subtitle=("Sixteen fixed prompts, each twelve training lines shared "
                  "by every problem, scored on two disjoint sets of 150 "
                  "problems. The prompt with the best score on set A, "
                  "re-scored on set B."),
        source="standarderror/ml/llmeval.py.",
        alt=("A six-row table. For the main model's multiplication the best "
             "prompt on set A was below average on set B; for the twin's "
             "carries it was the best on B as well."),
        caption=("Last column: correlation, across the sixteen prompts, "
                 "between their scores on the two sets."),
        align="llrrrrr",
        path=str(IMG / f"ml153-f2-prefix.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    fmt, pre = res["fmt"], res["pre"]
    g = _gaps(fmt, "probe_carry/add")
    pm = _prefix(pre, "test/mul", "clean")

    def p1(panel, m):
        panel.bar([0, 1], [_mean(fmt, "test/mul", "mixed", "clean"),
                           _mean(fmt, "test/mul", "empty", "clean")],
                  color=[m.series[0], m.series[1]])
        panel.set_ylim(0, 1)

    def p2(panel, m):
        panel.bar([0, 1], [-g["mixed"], -g["two"]],
                  color=[m.series[2], m.series[3]])

    def p3(panel, m):
        panel.bar([0, 1], [pm["best_dev"], pm["best_test"]],
                  color=[m.series[1], m.series[0]])
        panel.set_ylim(0.5, 0.85)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=53,
        headline="How you asked",
        panels=[(p1, f"{_mean(fmt, 'test/mul', 'empty', 'clean'):.2f}",
                 "multiply, no context"),
                (p2, f"{-g['two']:.3f}", "gap, 2 lines of context"),
                (p3, f"{pm['best_test']:.3f}", "best prompt, new items")],
        note=("A score is the model under one way of asking. Change the way "
              "and the score moves, and so, by different amounts, does "
              "every comparison made with it."),
        alt=("Three hand-drawn frames: a tall bar and a shorter one; a tall "
             "bar and a tiny one; two bars, the second lower."),
        path=str(IMG / f"ml153-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["kinds"] = s.run("""
        from standarderror.ml import llmeval as le
        from standarderror.schoolmath import model as sm

        main = sm.load()
        probs = le.items("test", "mul", 200)
        print("problem:", probs[0].prompt, " answer:", probs[0].answer)
        for kind in ("mixed", "empty", "same", "wrong"):
            ctx = le.contexts(kind, "mul", len(probs), seed=100)
            tail = ctx[0].replace(chr(10), " | ")[-48:]
            acc = le.ok(main, probs, fills=ctx).mean()
            print(f"{kind:6s} acc {acc:.3f}  ...{tail}")
        """)
    out["prefix"] = s.run("""
        import numpy as np
        sets = le.items("test", "mul", 150), le.items("test", "mul", 150, skip=150)
        pool = [p.line for p in le._train_by_task()["*"]]
        for seed in (500, 506):          # two of the sixteen fixed prompts
            rng = np.random.default_rng(seed)
            prompt = "".join(pool[i] for i in rng.integers(0, len(pool), 12))
            a, b = (le.ok(main, s, fills=[prompt] * 150).mean() for s in sets)
            print(f"prompt {seed}: set A {a:.3f}, set B {b:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    fmt, pre = res["fmt"], res["pre"]

    def mean(c, k, n="clean"):
        return _mean(fmt, c, k, n)

    mul_mixed, mul_empty = mean("test/mul", "mixed"), mean("test/mul", "empty")
    gc = _gaps(fmt, "probe_carry/add")
    gm = _gaps(fmt, "test/mul")
    go = _gaps(fmt, "test/order")
    rev = (mean("test/add_rev", "mixed", "leaky"),
           mean("test/add_rev", "empty", "leaky"))
    empty_cost = {c: mean(c, "mixed") - mean(c, "empty")
                  for c in CASES if c != "test/add_rev"}
    other_cost = [v for c, v in empty_cost.items() if c != "test/mul"]
    wrong_moves = [abs(mean(c, "wrong", n) - mean(c, "same", n))
                   for c in CASES for n in ("clean", "leaky")]
    within = max(max(v) - min(v) for k, v in fmt.items()
                 if "/empty/" not in k)
    pm = _prefix(pre, "test/mul", "clean")
    pk = _prefix(pre, "probe_carry/add", "leaky")
    po = _prefix(pre, "test/order", "leaky")
    oc, ol = np.array(pre["test/order/clean"]), np.array(pre["test/order/leaky"])
    order_flip_a = int((ol[:, 0] > oc[:, 0]).sum())
    order_flip_b = int((ol[:, 1] > oc[:, 1]).sum())

    # The spine, asserted rather than trusted.
    assert 0.72 < mul_mixed < 0.75 and abs(mul_empty - 0.5) < 0.005
    assert all(v > 0.02 for v in empty_cost.values())
    assert -0.135 < gc["mixed"] < -0.12 and -0.02 < gc["two"] < 0
    assert 0.04 < min(gm.values()) < 0.06 and 0.16 < max(gm.values()) < 0.18
    assert min(go.values()) < 0.005 and max(go.values()) > 0.05
    assert rev[0] < 0.01 and 0.4 < rev[1] < 0.45
    assert max(wrong_moves) < 0.01
    assert 0.3 < 1 - mul_empty / mul_mixed < 0.34
    assert mean("test/order", "two") < mean("test/order", "empty")
    assert mean("test/linear", "two") < mean("test/linear", "empty")
    assert (min(fmt["test/add_rev/two/clean"]) >= 0.995
            and min(fmt["test/add_rev/mixed/clean"]) == 1.0)
    sl, so = (fmt["probe_carry/add/same/leaky"],
              fmt["probe_carry/add/other/leaky"])
    assert max(sl) < min(so)
    assert within <= 0.065
    assert mean("test/mul", "two") > mul_mixed - 0.02
    assert pm["best_dev"] > pm["dev_mean"] + 0.03
    assert pm["best_test"] < pm["test_mean"] and abs(pm["corr"]) < 0.1
    assert pk["corr"] > 0.9 and pk["best_test"] == pk["test_max"]
    assert po["best_test"] < po["test_mean"]
    assert order_flip_a == 3 and order_flip_b == 0

    post = Post(
        title=f"{SERIES_TAG} 53: The Prompt Is Part of the Score",
        slug="ml-53-prompt-format",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=53,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("The same model on the same problems scored 0.735 or 0.500 "
                  "depending on what came before them, and the gap between "
                  "two models shrank from 0.127 to about zero. A score is "
                  "the model under one way of asking."),
        summary=(
            f"Two small arithmetic transformers, 200 problems per task, six "
            f"kinds of context in front of each problem. The main model's "
            f"multiplication score was {mul_mixed:.3f} after twelve ordinary "
            f"lines and {mul_empty:.3f} after none. Its lead over its leaky "
            f"twin on multiplication ranged from {min(gm.values()):.3f} to "
            f"{max(gm.values()):.3f} across contexts, and the twin's lead on "
            f"three-digit carries from {-gc['mixed']:.3f} to "
            f"{-gc['two']:.3f}. Examples with wrong answers changed no score "
            f"by more than {max(wrong_moves):.3f}. Of sixteen fixed prompts, "
            f"the best on one set of 150 multiplications scored "
            f"{pm['best_dev']:.3f} there and {pm['best_test']:.3f} on a "
            f"second set, below the prompts' average of "
            f"{pm['test_mean']:.3f}; for the twin's carries, where prompts "
            f"really differed, the choice held."),
        tags=["machine-learning", "llm-evaluation", "prompting",
              "language-models", "evaluation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Models: the main arithmetic transformer of the school-maths "
            "series and its leaky twin (`data/schoolmath/`), trained by this "
            "repository on generated arithmetic.",
            "Problems: 200 drawn at random per task from the test split "
            "(multiplication, order of operations, linear equations, reversed "
            "sums) and the three-digit carry probe; for the prompt search, "
            "two disjoint sets of 150. Contexts are drawn from training lines "
            "(seeds 100-102 per kind; prompts seeds 500-515).",
            "Machinery: `standarderror/ml/llmeval.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Sclar, Choi, Tsvetkov and Suhr, \"Quantifying "
            "language models' sensitivity to spurious features in prompt "
            "design\", *ICLR* (2024); Min et al., \"Rethinking the role of "
            "demonstrations\", *EMNLP* (2022); Zhao, Wallace, Feng, Klein "
            "and Singh, \"Calibrate before use\", *ICML* (2021); episode 35 "
            "of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("committed weights, seeded draws; about seventeen "
                            "minutes of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Six ways to ask the same question",
        f"""A language model never sees a question alone. It sees a window of text that ends with the question: a system prompt, instructions, worked examples, earlier turns. A benchmark fixes one such window per question and reports the result as the model's score, and a leaderboard compares models under that one window. The comparison assumes the window is a neutral container, the sentence under test: *a benchmark score measures the model*.

The arithmetic models of this arc make the window easy to vary and the answers easy to check. Every earlier number for them was measured with twelve random training lines in front of each problem, drawn afresh per problem. Here the same 200 problems per task are asked in six contexts: twelve random lines; nothing at all; two random lines; twelve lines of the same task; twelve of a different task; and twelve of the same task with the last digit of every answer changed, the arithmetic version of examples with wrong labels. Each kind is drawn three times, so its spread from one draw to the next can be set against the differences between kinds.

{snip['kinds'].markdown()}

The main model multiplied correctly {mul_mixed:.3f} of the time with twelve lines in front and {mul_empty:.3f} with nothing: about a third of its score was the context. An empty context cost it {min(other_cost):.3f} to {max(other_cost):.3f} on the other three tasks too. For multiplication, two random lines recovered almost all of the score ({mean('test/mul', 'two'):.3f}), as if what mattered was having something in front of the problem, the way every problem appeared in the packed training stream; but on mixed operations and linear equations two lines did slightly worse than none, so position is not the whole story. Across three draws of the same kind of context no score moved by more than {within:.3f}: the large differences between kinds, like the empty context's, are far outside that, and the small ones are not.""",
        figures=[figs["f1"]])

    post.add(
        "The gap between two models moved too",
        f"""A shift that moved every model equally would leave a leaderboard intact. This one did not. On three-digit carries, the leaky twin beat the main model by {-gc['mixed']:.3f} under the standard twelve lines and by {-gc['two']:.3f} under two: a clear win and a tie, from the same two models on the same problems. On multiplication the main model's lead ran from {min(gm.values()):.3f} to {max(gm.values()):.3f}, on mixed operations from {min(go.values()):.3f} to {max(go.values()):.3f}.

The most extreme case is the one the school-maths series was built on. The twin learned reversed sums by copying from the line above, so it scored {rev[0]:.3f} on them after twelve lines of context and {rev[1]:.3f} with none, while the main model scored 0.995 or 1.0 under every context. Whether the twin "can do" reversed sums depends entirely on how it is asked.

Sclar and colleagues found the same thing at scale in 2024: changing only the formatting of a prompt, separators and spacing and casing, moved one model's few-shot accuracy on a task by up to 76 points, and which formats worked well correlated only weakly from one model to another, which they argued makes a comparison of models under one fixed format unreliable. Their remedy, reporting a range over plausible formats rather than a point, is the one these numbers support.""",
        figures=[figs["f0"]])

    post.add(
        "Wrong examples changed nothing",
        f"""If the context mattered because the model read the examples, examples with wrong answers should hurt. They did not: replacing the last digit of every example answer changed no model's score on any task by more than {max(wrong_moves):.3f} relative to correct examples of the same task. What the examples taught was not their answers. Which task they came from did matter, for the twin: on three-digit carries it scored {mean('probe_carry/add', 'same', 'leaky'):.3f} after lines of the same task and {mean('probe_carry/add', 'other', 'leaky'):.3f} after lines of a different one, outside the spread of either kind's draws.

That is a finding about these models, which are far too small to learn from examples in context, and it should not be read as a finding about large ones. It does echo Min and colleagues' 2022 result for larger models: on many classification tasks, randomly reassigning the labels of the demonstrations barely hurt, and the format and the distribution of the inputs mattered more than whether the labels were right. Either way, a context's effect on a score is not evidence that the model understood the context.""")

    post.add(
        "Choosing the prompt on the items you report",
        f"""In practice the window is chosen, not drawn. Zhao and colleagues showed in 2021 that a large model's few-shot accuracy swung widely with which examples were in the prompt and in what order, and the natural response is to write several prompts, try them, and keep the best. To imitate that, sixteen fixed prompts were built, each twelve training lines shared by every problem, the way a prompt template puts the same worked examples in front of every question, and each was scored on one set of 150 problems (A) and on a disjoint second set (B).

{snip['prefix'].markdown()}

For the main model on multiplication, the sixteen prompts scored between {pm['dev_min']:.3f} and {pm['dev_max']:.3f} on set A. The best one, {pm['best_dev']:.3f}, scored {pm['best_test']:.3f} on set B, below the {pm['test_mean']:.3f} average of all sixteen there; the correlation between the prompts' scores on the two sets was {pm['corr']:+.2f}. The search had found the prompt that suited set A's particular problems. This is episode 35's result with prompts in place of hyperparameters: the best score of a search over noisy estimates reports the search's luck.

It is not always luck. For the twin on three-digit carries the same sixteen prompts ranged from {pk['dev_min']:.3f} to {pk['dev_max']:.3f}, the two sets agreed (correlation {pk['corr']:+.2f}), and the best prompt on A was also the best on B. Presumably some prompts gave the twin more to copy from; that was not checked. Without the second set there is no way to tell these two situations apart. On mixed operations, the twin beat the main model under {order_flip_a} of the sixteen prompts on set A and under {order_flip_b} on set B. The true order is probably the main model first, and a comparison reported under one of those three prompts, chosen on set A, would have stated the opposite.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        """**Context, not instructions.** These models cannot follow instructions, so the prompt variations here are positions and examples. For instruction-following models, wording and format are larger levers and the variations multiply.

**Two models, one family.** The two models share their training data and architecture. Models from different families respond to the same prompt change differently, which makes ranking changes more likely, not less.

**Small sets.** The prompt search used 150 problems per set, so a difference of 0.04 between two prompts' scores is about one standard error of either score, and less than two of the paired difference. That is the condition under which selection does the most damage, and also the usual one.

**Greedy decoding.** Sampling with a temperature adds another source of run-to-run variation, which was not measured.""")

    post.add(
        "What to keep",
        f"""1. With nothing in front of each problem the main model multiplied correctly {mul_empty:.3f} of the time; with twelve ordinary lines, {mul_mixed:.3f}.

2. The twin's lead on three-digit carries was {-gc['mixed']:.3f} under one context and {-gc['two']:.3f} under another; the main model's lead on multiplication ranged from {min(gm.values()):.3f} to {max(gm.values()):.3f}.

3. The best of sixteen prompts on one set of 150 multiplications scored below the prompts' average on a second set. Where the prompts really differed, the choice survived the second set.

4. Report a score with its prompt and a range over reasonable prompts, compare models under the same prompts, and choose a prompt on items you will not report.""")

    post.add(
        "Exercise",
        """Take the evaluation prompt you use and write four variants that a colleague would call equivalent: different separators, a different instruction wording, the examples in a different order, no examples. Score two models you are comparing under all five on the same items.

If the ranking holds under every variant, report it with the range. If it flips, the comparison is a comparison of prompts.""")

    post.add(
        "Next",
        """Every score so far has been exact match on a generated answer. Many benchmarks score differently: they offer options and take the one the model finds likeliest. Episode 54 asks the same models the same questions both ways.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
