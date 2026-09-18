"""School Maths 1: The Loss Never Said a Word.

The first episode of a series that uses school arithmetic as an instrument.
This one is about a model that could not add, a held-out loss that reported
nothing wrong, and the one line of the setup that caused it.

Two models, identical in every respect that is usually written down --
architecture, parameters, seed, steps, learning-rate schedule, and the exact
same problems. One manipulated variable: the order of the lines in the
training file.

Measured:

* In generation order every reversed line lands directly beneath the forward
  form of the same sum. Not most of them: all of them.
* The model trained that way scores near-perfectly on reversed addition when
  the matching forward line is above it and near zero when it is not.
* Show it a *wrong* forward answer and it reverses the wrong answer, which is
  what separates copying from adding.
* Its held-out loss, computed the way its own pipeline computes it, is
  *better* than the honest model's for the entire run. Rescoring the same
  held-out problems packed the other way separates them by 0.77 nats, and
  that repacking is the whole diagnostic.
* The honest model, on the same corpus shuffled, answers reversed sums under
  any context at all -- and then fails in a much more interesting place, which
  is where episode 2 starts.

Run: `standarderror run sm101_shortcut --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.schoolmath import arithmetic as ar
from standarderror.schoolmath import curriculum as cur
from standarderror.schoolmath import model as sm
from standarderror.viz import charts

POST_DATE = date(2026, 9, 18)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "School Maths, Taught Through What a Model Gets Wrong"
SERIES_TAG = "School Maths"

LIMIT = 250
PER_WIDTH = 4_000


def compute() -> dict:
    clean = sm.load()
    leaky = sm.load(adjacent=True)
    return {
        "clean": clean, "leaky": leaky,
        "adjacency": ar.twin_adjacency(per_width=PER_WIDTH),
        "ab": ar.shortcut(clean, leaky, limit=LIMIT, per_width=PER_WIDTH),
        "loss": ar.loss_table(clean, leaky),
        "tasks": ar.by_task(clean, limit=LIMIT, per_width=PER_WIDTH),
        "tasks_leaky": ar.by_task(leaky, limit=LIMIT, per_width=PER_WIDTH),
        "carry": ar.carry_curve(clean, limit=LIMIT, per_width=PER_WIDTH),
        "length": ar.length_curve(clean, limit=LIMIT, per_width=PER_WIDTH),
        "where_rev": ar.first_error(clean, "add_rev", split="probe_carry",
                                    limit=LIMIT, per_width=PER_WIDTH),
        "padding": ar.padding_sensitivity(clean, limit=LIMIT,
                                          per_width=PER_WIDTH),
    }


# ------------------------------------------------------------------ figures

ARMS = ["its own sum on the line above", "a wrong sum on the line above",
        "a different sum on the line above", "ordinary context"]
SHORT = ["its own sum\nabove", "a wrong sum\nabove", "a different sum\nabove",
         "ordinary\ncontext"]


def figures(res: dict) -> dict:
    out: dict = {}
    ab = res["ab"]
    carry = res["carry"]
    loss = res["loss"]

    def bars(ax, m):
        xs = np.arange(len(ARMS))
        w = 0.38
        ax.bar(xs - w / 2, [ab["generation order"][a] for a in ARMS], w,
               color=m.series[2], label="trained in generation order")
        ax.bar(xs + w / 2, [ab["shuffled"][a] for a in ARMS], w,
               color=m.series[0], label="trained on the same file, shuffled")
        for x, a in zip(xs, ARMS):
            for dx, key in ((-w / 2, "generation order"), (w / 2, "shuffled")):
                v = ab[key][a]
                ax.annotate(f"{v:.2f}", (x + dx, v), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=9.0,
                            color=m.ink_secondary)
        ax.set_xticks(xs)
        ax.set_xticklabels(SHORT, fontsize=8.4)
        ax.set_ylim(0, 1.12)
        ax.legend(frameon=False, fontsize=8.6, loc="upper right")

    out["f0"] = charts.diagram(
        bars,
        title="One model is adding. The other is reading the line above it",
        subtitle=(f"Exact-match accuracy on {ab['n']} held-out reversed sums, "
                  f"under four contexts. The two models differ only in the "
                  f"order of the lines in their training file."),
        xlabel="what preceded the question",
        ylabel="share answered exactly right",
        source="Measured; standarderror/schoolmath/arithmetic.py.",
        alt=("Four pairs of bars. The first pair is high for both models; in "
             "the other three the generation-order bars collapse to near "
             "zero while the shuffled ones stay high."),
        caption=(f"With the matching forward sum directly above it, the "
                 f"generation-order model scores "
                 f"{ab['generation order'][ARMS[0]]:.3f}. Take that line away "
                 f"and it scores {ab['generation order'][ARMS[3]]:.3f}. The "
                 f"shuffled model is between "
                 f"{min(ab['shuffled'][a] for a in ARMS):.3f} and "
                 f"{max(ab['shuffled'][a] for a in ARMS):.3f} throughout, "
                 f"because it is doing the arithmetic."),
        path=str(IMG / f"sm101-f0-contexts.{EXT}"))[0]

    def curves(ax, m):
        for bundle, name, colour in ((res["clean"], "shuffled", m.series[0]),
                                     (res["leaky"], "generation order",
                                      m.series[2])):
            hist = bundle["history"]
            ax.plot([h["step"] for h in hist], [h["test_loss"] for h in hist],
                    lw=2.4, color=colour,
                    label=f"{name}, validated its own way")
        last = max(h["step"] for h in res["clean"]["history"])
        ax.plot([last], [loss["generation order"]["repacked"]], marker="*",
                ms=16, color=m.series[2], ls="none",
                label=("generation order, validation set repacked: "
                       f"{loss['generation order']['repacked']:.3f}"))
        ax.plot([last], [loss["shuffled"]["repacked"]], marker="*", ms=12,
                color=m.series[0], ls="none",
                label=("shuffled, repacked: "
                       f"{loss['shuffled']['repacked']:.3f}"))
        ax.axhline(np.log(len(cur.ALPHABET)), lw=1.6, ls="--", color=m.grid,
                   label=f"uniform guess, {np.log(len(cur.ALPHABET)):.3f}")
        ax.legend(frameon=False, fontsize=8.0, loc="upper right")

    out["f1"] = charts.diagram(
        curves,
        title="The metric preferred the model that cannot add",
        subtitle=("Held-out cross-entropy. Each curve is the number that "
                  "model's own pipeline reports; the stars are the same "
                  "models on a validation set packed the other way."),
        xlabel="training step",
        ylabel="held-out cross-entropy (nats)",
        source="Measured; standarderror/schoolmath/arithmetic.py.",
        alt=("Two loss curves descending together, the generation-order one "
             "ending slightly lower, and a star far above both marking its "
             "loss on a repacked validation set."),
        caption=(f"On its own validation stream the generation-order model "
                 f"reaches {loss['generation order']['own packing']:.4f} "
                 f"against the honest model's "
                 f"{loss['shuffled']['own packing']:.4f} - it is **ahead**. "
                 f"Repack the same held-out problems in the other order and "
                 f"it is at {loss['generation order']['repacked']:.4f} while "
                 f"the honest model barely moves, to "
                 f"{loss['shuffled']['repacked']:.4f}. That one-line "
                 f"diagnostic is the entire difference between the two "
                 f"readings."),
        path=str(IMG / f"sm101-f1-loss.{EXT}"))[0]

    def chains(ax, m):
        for task, name, colour, mk in (("add", "written as it is read",
                                        m.series[2], "o"),
                                       ("add_rev", "written as it is computed",
                                        m.series[0], "s")):
            rows = [r for r in carry if r["task"] == task]
            ax.plot([r["carry_chain"] for r in rows],
                    [r["accuracy"] for r in rows], marker=mk, ms=7, lw=2.3,
                    color=colour, label=name)
        ax.axvspan(1.5, 3.4, color=m.grid, alpha=0.18)
        ax.annotate("never shown at this width", (2.45, 0.06), ha="center",
                    fontsize=8.6, color=m.ink_secondary)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_ylim(0, 1.06)
        ax.legend(frameon=False, fontsize=8.6, loc="lower left")

    out["f2"] = charts.diagram(
        chains,
        title="Carrying was learned at a width, not as a rule",
        subtitle=("Three-digit additions by the longest run of consecutive "
                  "carries. Chains of two and three appear in training at one "
                  "and two digits, and never at three."),
        xlabel="longest run of consecutive carries",
        ylabel="share answered exactly right",
        source="Measured; standarderror/schoolmath/arithmetic.py.",
        alt=("Two curves flat near 1.0 at chains 0 and 1, then falling "
             "sharply across a shaded region at chains 2 and 3, the lower "
             "curve falling further."),
        caption=(_carry_caption(carry)),
        path=str(IMG / f"sm101-f2-carry.{EXT}"))[0]

    rows = [[a, f"{ab['generation order'][a]:.3f}", f"{ab['shuffled'][a]:.3f}"]
            for a in ARMS]
    rows.append(["...reproducing that wrong sum, reversed",
                 f"{ab['generation order']['reproduced the wrong sum, reversed']:.3f}",
                 "-"])
    rows.append(["...answering correctly anyway",
                 f"{ab['generation order']['answered correctly anyway']:.3f}",
                 "-"])
    out["f3"] = charts.table_image(
        rows,
        header=["what preceded the question", "generation order", "shuffled"],
        title="The same question, four ways round",
        subtitle=(f"{ab['n']} held-out reversed sums. Everything about the "
                  f"two models is identical except the order of the lines "
                  f"they were trained on."),
        source="Measured; standarderror/schoolmath/arithmetic.py.",
        alt=("A five-row table. The generation-order column is high on the "
             "first row and near zero below it; the shuffled column is high "
             "throughout."),
        caption=("The **bold** row is the one that settles it. Shown a wrong "
                 "forward answer, the generation-order model reverses the "
                 "wrong answer - the right one was available by adding, and "
                 "it did not add."),
        bold_cells={(1, c) for c in range(3)} | {(4, c) for c in range(3)},
        align="lrr",
        path=str(IMG / f"sm101-f3-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _carry_caption(carry) -> str:
    g = {(r["task"], r["carry_chain"]): r["accuracy"] for r in carry}
    return (f"At chains of 0 and 1 the model is at {g[('add', 0)]:.3f} and "
            f"{g[('add', 1)]:.3f}. At a chain of 2 it is {g[('add', 2)]:.3f} "
            f"and at 3 it is {g[('add', 3)]:.3f}. Writing the answer "
            f"least-significant-digit-first - the order the algorithm "
            f"actually runs in - recovers it to {g[('add_rev', 2)]:.3f} and "
            f"{g[('add_rev', 3)]:.3f}, **with nothing else changed**.")


def _hero(res: dict):
    ab, adj = res["ab"], res["adjacency"]

    def stack(panel, m):
        for i, y in enumerate((0.72, 0.56, 0.40, 0.24)):
            panel.plot([0.15, 0.85], [y, y], lw=3.0,
                       color=m.series[2] if i % 2 else m.series[0])
        panel.set_xlim(0, 1)
        panel.set_ylim(0.1, 0.86)

    def cliff(panel, m):
        panel.plot([0, 1, 2, 3],
                   [ab["generation order"][ARMS[0]],
                    ab["generation order"][ARMS[1]],
                    ab["generation order"][ARMS[2]],
                    ab["generation order"][ARMS[3]]],
                   lw=3.0, color=m.series[2])
        panel.set_ylim(-0.05, 1.05)

    def flat(panel, m):
        loss = res["loss"]
        panel.bar([0, 1], [loss["shuffled"]["own packing"],
                           loss["generation order"]["own packing"]],
                  width=0.55, color=[m.series[0], m.series[2]])
        panel.set_xlim(-0.6, 1.6)
        panel.set_ylim(0, 1.5)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=1,
        headline="Two models, one difference: the order of the lines",
        panels=[(stack, f"{adj['generation order']['share']:.0%}",
                 "twins stacked"),
                (cliff, f"{ab['generation order'][ARMS[3]]:.2f}",
                 "without that line"),
                (flat, f"{res['loss']['generation order']['own packing']:.3f}",
                 "the better loss")],
        note=(f"Packed in generation order, every reversed sum sits directly "
              f"beneath its own forward twin, and 'reverse the line above' "
              f"solves the task without arithmetic. That model scores "
              f"{ab['generation order'][ARMS[0]]:.2f} with the twin above and "
              f"{ab['generation order'][ARMS[3]]:.2f} without it. Shuffling "
              f"the same file fixes it. The held-out loss preferred the "
              f"broken model, "
              f"{res['loss']['generation order']['own packing']:.3f} against "
              f"{res['loss']['shuffled']['own packing']:.3f}."),
        alt=("Three hand-drawn frames: four stacked alternating rules; a "
             "line falling off a cliff; and two bars of almost equal height, "
             "the right one slightly shorter."),
        path=str(IMG / f"sm101-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["adjacency"] = s.run("""
        from standarderror.schoolmath import arithmetic as ar

        # A property of the training file. No model involved.
        print(f"{'packing':18s} {'reversed':>9s} {'twin above':>11s} {'':>7s}")
        for name, row in ar.twin_adjacency().items():
            print(f"{name:18s} {row['reversed_lines']:9,d} "
                  f"{row['adjacent']:11,d} {row['share']:7.1%}")
        """)

    out["ab"] = s.run(f"""
        from standarderror.schoolmath import model as sm

        clean = sm.load()                 # trained on the shuffled stream
        leaky = sm.load(adjacent=True)    # same file, generation order
        ab = ar.shortcut(clean, leaky, limit={LIMIT})

        head = "what preceded the question"
        print(f"{{head:34s}} {{'gen order':>10s}} {{'shuffled':>9s}}")
        for arm in ab["shuffled"]:
            print(f"{{arm:34s}} {{ab['generation order'][arm]:10.3f}}"
                  f" {{ab['shuffled'][arm]:9.3f}}")
        print()
        g = ab["generation order"]
        print(f"shown a wrong sum, it reproduced that sum reversed: "
              f"{{g['reproduced the wrong sum, reversed']:.3f}}")
        print(f"          ... and answered correctly anyway: "
              f"{{g['answered correctly anyway']:.3f}}")
        """)

    out["loss"] = s.run("""
        # The number that was supposed to catch this, both ways round.
        table = ar.loss_table(clean, leaky)
        print(f"{'model':18s}  {'its own packing':>16s}  {'repacked':>10s}")
        for name, row in table.items():
            print(f"{name:18s}  {row['own packing']:16.4f}"
                  f"  {row['repacked']:10.4f}")
        """)

    out["carry"] = s.run(f"""
        # What the honest model did learn, and where it stops.
        for row in ar.carry_curve(clean, limit={LIMIT}):
            flag = "held out here" if row["held_out"] else ""
            print(f"{{row['task']:8s}} chain {{row['carry_chain']}}"
                  f"  n {{row['n']:4d}}  {{row['accuracy']:.3f}}  {{flag}}")
        """)

    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    ab, adj = res["ab"], res["adjacency"]
    carry = {(r["task"], r["carry_chain"]): r["accuracy"] for r in res["carry"]}
    length = {(r["task"], r["digits"]): r["accuracy"] for r in res["length"]}
    where = res["where_rev"]
    second = where["first_wrong_position"].get(1, 0.0)
    tasks = {r["task"]: r["accuracy"] for r in res["tasks"]}
    leaky_tasks = {r["task"]: r["accuracy"] for r in res["tasks_leaky"]}
    loss = res["loss"]
    own_gap = (loss["shuffled"]["own packing"]
               - loss["generation order"]["own packing"])
    repack_gap = (loss["generation order"]["repacked"]
                  - loss["generation order"]["own packing"])
    twin, plain = ARMS[0], ARMS[3]

    # The spine, asserted rather than trusted.
    assert adj["generation order"]["share"] == 1.0
    assert adj["shuffled"]["share"] == 0.0
    assert ab["generation order"][twin] > 0.8
    assert ab["generation order"][plain] < 0.2
    assert min(ab["shuffled"][a] for a in ARMS) > 0.8
    assert (ab["generation order"]["reproduced the wrong sum, reversed"]
            > 20 * ab["generation order"]["answered correctly anyway"])
    assert loss["generation order"]["own packing"] < \
        loss["shuffled"]["own packing"]
    assert repack_gap > 0.4
    assert abs(loss["shuffled"]["repacked"]
               - loss["shuffled"]["own packing"]) < 0.05
    assert carry[("add", 0)] > 0.95 and carry[("add", 1)] > 0.95
    assert carry[("add", 2)] < 0.85 and carry[("add", 3)] < carry[("add", 2)]
    assert carry[("add_rev", 2)] > carry[("add", 2)]
    assert res["padding"]["add"]["fresh spread"] < 0.05
    assert length[("add", 4)] < 0.02 and length[("add_rev", 4)] < 0.02
    assert where["leading_digit_share"] < 0.05 < second

    post = Post(
        title=(f"{SERIES_TAG} 1: The Only Difference Between These Two "
               f"Models Is the Order of the Lines"),
        slug="school-maths-1-shortcut",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=1,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("One of them adds. The other reads the answer off the line "
                  "above and reverses it, including when that line is wrong. "
                  "Their held-out losses differ by less than a tenth of a "
                  "nat, and the leaky one's is *lower*."),
        summary=(
            f"Two character transformers, identical in architecture, "
            f"parameters, seed, steps, schedule and training problems. The "
            f"one manipulated variable is the order of the lines in the "
            f"training file. Packed in generation order, every reversed sum "
            f"lands directly beneath the forward form of the same sum - "
            f"{adj['generation order']['adjacent']:,} of "
            f"{adj['generation order']['reversed_lines']:,}, which is all of "
            f"them - and 'reverse the line above' answers the question with "
            f"no arithmetic. That model scores "
            f"{ab['generation order'][twin]:.3f} on held-out reversed sums "
            f"with the matching line above it and "
            f"{ab['generation order'][plain]:.3f} under ordinary context; "
            f"shown a deliberately wrong forward answer it reproduces that "
            f"wrong answer, reversed, "
            f"{ab['generation order']['reproduced the wrong sum, reversed']:.0%} "
            f"of the time and the correct answer in "
            f"{ab['generation order']['answered correctly anyway'] * ab['n']:.0f} "
            f"of {ab['n']} cases. "
            f"Shuffling the same file fixes it completely. The "
            f"held-out loss preferred the broken one - "
            f"{loss['generation order']['own packing']:.4f} against "
            f"{loss['shuffled']['own packing']:.4f} - and it "
            f"pointed the wrong way for the entire run, because a shortcut "
            f"created by ordering survives any random split of the rows. "
            f"Rescoring the same held-out problems packed the other way "
            f"separates them by {repack_gap:.4f} nats and costs one line. "
            f"The honest "
            f"model then fails somewhere far more interesting: it carries "
            f"correctly at {carry[('add', 1)]:.3f} on the chains it was "
            f"shown and {carry[('add', 3)]:.3f} on the ones it was not."),
        tags=["machine-learning", "data-science", "mathematics",
              "transformers", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data, and in this episode not even a fetched "
            "corpus. The training text is arithmetic generated by "
            "`standarderror/schoolmath/curriculum.py`, deterministic in its "
            "seed, so a reader rebuilds the exact training stream from the "
            "source rather than downloading it.",
            f"The models: two {res['clean']['parameters']:,}-parameter "
            f"character-level transformers - four blocks, four heads, width "
            f"128, context 64, an eighteen-character vocabulary. Same "
            f"architecture as the model the Calculus series takes apart. "
            f"Both checkpoints, the training script and verified sha256s are "
            f"committed: `standarderror/schoolmath/`, "
            f"`scripts/train_schoolmath.py`, `data/schoolmath/`.",
            "Machinery: `standarderror/schoolmath/arithmetic.py`, tested in "
            "`tests/test_schoolmath.py`, which pins the carry chain against "
            "an independent implementation, the hold-outs against leakage, "
            "and both traps in this episode as regressions.",
            "Where this stops: the shortcut here is one I built, in a corpus "
            "I generated, and I am not claiming it is the shortcut in "
            "anyone else's data. The transferable part is the mechanism - a "
            "regularity in the *order* of the rows is invisible to a random "
            "split of the rows - and the diagnostic, which is to vary the "
            "context and watch whether the answer moves.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python=3.11.15, torch=2.14.0, numpy=2.4.4"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "models": ("both committed checkpoints, hash-verified on load, "
                       "so every accuracy is measured on the same weights"),
            "determinism": ("`torch.manual_seed(0)` and an identical batch "
                            "schedule for both runs; the corpus is "
                            "deterministic in its own seed, and the shuffle "
                            "that separates the two models is a fixed "
                            "permutation"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The question that does not work",
        f"""The obvious question about a small model and school arithmetic is whether it can learn to add. It is the wrong question, and not because the answer is boring. It is wrong because you cannot tell from the outside which of two things happened: the model learned to add, or the model learned your file.

This episode is one experiment with one manipulated variable. Two character-level transformers, {res['clean']['parameters']:,} parameters each, four blocks and four heads and a context of 64 — the same architecture the Calculus series in this project spent five episodes taking apart. Same seed, same {20_000:,} steps, same learning-rate schedule, same batch size, and the same arithmetic problems in the same splits.

The only difference is the order of the lines in the training file.

One of the two models cannot add. Its held-out cross-entropy is {loss['generation order']['own packing']:.4f} against the other's {loss['shuffled']['own packing']:.4f} — it is the *better* number, for the whole run — and I want to be precise about that, because the reflex when someone describes a leak is to assume they were not watching the validation curve. I was watching it. It was pointing at the wrong model.""")

    post.add(
        "What the corpus is",
        """Six kinds of problem, one per line, in a character vocabulary of eighteen. Addition, subtraction and multiplication; order of operations; one linear equation in one unknown. And addition written twice — once forwards, the way it is read, and once with the answer's digits reversed, the way the schoolbook algorithm actually produces them:

```
123+456=579
123+456~975
```

The reversed form is there because it is the honest order. A child adding on paper starts at the units, and the digit they write first is the digit a left-to-right reader sees last. A model generating left to right has to commit to the leading digit before it has worked out whether anything carries into it. Those are opposite constraints, and giving the model both notations lets one model's two accuracies answer the question, rather than two training runs.

The hold-outs are structural. Three-digit additions whose carries *propagate* — a carry out of the units that forces a carry out of the tens — never appear at three digits in training, though they appear constantly at two. Four-digit operands never appear at all. So the probes ask for a rule at a size it was not demonstrated, which is the thing a curriculum assumes a child can do and is worth checking on a model.""")

    post.add(
        "Before the model: a property of the file",
        f"""Here is what I should have checked first, and did not. It needs no model and no GPU.

{snip['adjacency'].markdown()}

Every reversed line in the generated stream sits directly beneath the forward form of the same sum. Not most of them: {adj['generation order']['adjacent']:,} of {adj['generation order']['reversed_lines']:,}.

That is a shortcut and it is a complete one. `reverse the answer on the line above` solves the reversed task exactly, on every example, without representing addition in any way. It is also *cheaper* than adding, so a model under a gradient has every reason to find it.

I did shuffle. Every training batch is a randomly positioned 64-character window of the stream, which is the standard thing and which most people would call shuffling. It does not help here, and the reason is worth stating plainly: the adjacency is **inside the window**. Shuffling the order in which you visit windows does nothing about what is in one.""")

    post.add(
        "Four ways to ask the same question",
        f"""So: take the two models and ask them the same held-out reversed sums under four different contexts. The first puts the matching forward line immediately above the question. The second puts a *wrong* forward answer there. The third puts a different sum's line there. The fourth is ordinary text — a run of unrelated problems, which is what the model saw for the overwhelming majority of training.

{snip['ab'].markdown()}

With its twin above it, the generation-order model is at {ab['generation order'][twin]:.3f}. Take that one line away and it is at {ab['generation order'][plain]:.3f}. The shuffled model does not care: it is at {min(ab['shuffled'][a] for a in ARMS):.3f} in all four, because what is above it is not where its answer comes from.

The second arm is the one that settles it, and it is the reason the experiment has four contexts instead of two. A model that is adding has the right answer available no matter what nonsense precedes it. A model that is copying will copy the nonsense. Shown a forward line carrying a deliberately wrong sum, the generation-order model reproduced **that wrong sum, reversed**, {ab['generation order']['reproduced the wrong sum, reversed']:.0%} of the time — and produced the correct answer in {ab['generation order']['answered correctly anyway'] * ab['n']:.0f} of {ab['n']} cases.

Those two rates together are the finding, not the first one alone. The other {1 - ab['generation order']['reproduced the wrong sum, reversed'] - ab['generation order']['answered correctly anyway']:.0%} is neither: inspecting them, they are mostly near-misses on the number it was shown rather than on the number it was asked for — a digit out of place in a copy, not a calculation gone slightly wrong. What is absent is the right answer, which was available the whole time to anything that was adding. It is not adding badly. It is not adding.""",
        figures=[figs["f0"], figs["f3"]])

    post.add(
        "And the loss said it was the better model",
        f"""{snip['loss'].markdown()}

Read the first column. That is what each pipeline reports about itself: you pack your validation set the way you packed your training set, because that is what a validation set is. The generation-order model reaches {loss['generation order']['own packing']:.4f} and the honest one {loss['shuffled']['own packing']:.4f}. The broken model is **{own_gap:.4f} nats ahead**, and it is ahead for the whole run.

I had drafted this section around the claim that the loss stayed silent. It did not. Choosing between these two checkpoints on held-out cross-entropy — which is the thing everyone does — selects the one that cannot add, and does so by a margin that looks like a real improvement rather than a rounding error.

The reason is that the shortcut is available on the held-out stream as well. It was packed the same way. A random split of the *rows* cannot break a regularity that lives in the *order* of the rows, so the validation set inherits it intact, and a model exploiting it genuinely does predict the next character better. Nothing is malfunctioning. The metric is measuring what it says it measures.

Now the second column, which costs one line: score each model on the held-out problems packed the *other* way. The honest model does not notice — {loss['shuffled']['own packing']:.4f} to {loss['shuffled']['repacked']:.4f}, a difference of {abs(loss['shuffled']['repacked'] - loss['shuffled']['own packing']):.4f}. The generation-order model goes to {loss['generation order']['repacked']:.4f}, up {repack_gap:.4f} nats.

So the loss was never the wrong instrument. It was pointed at one packing of the data, and the defect lived in the packing. Repacking the validation set is not a clever technique; it is the same estimator on a different arrangement of the same problems, and it separates these two models by {repack_gap:.4f} nats where the standard arrangement separates them by {own_gap:.4f} in the wrong direction.""",
        figures=[figs["f1"]])

    post.add(
        "What the honest model learned, and where it stops",
        f"""Shuffling the same file — same problems, same splits, same everything else — produces a model that answers reversed sums under any context at all. On the held-out split it scores {tasks['add']:.3f} on forward addition, {tasks['add_rev']:.3f} reversed, {tasks['sub']:.3f} on subtraction.

That is the boring part. Here is the interesting one.

{snip['carry'].markdown()}

At carry chains of 0 and 1 it is at {carry[('add', 0)]:.3f} and {carry[('add', 1)]:.3f}. At a chain of 2 — one carry forcing the next, at a width where that combination was held out — it drops to {carry[('add', 2)]:.3f}, and at 3 to {carry[('add', 3)]:.3f}. It saw propagating carries constantly at two digits. It did not carry the rule across.

And then the same sums, with only the answer's direction changed, come back to {carry[('add_rev', 2)]:.3f} and {carry[('add_rev', 3)]:.3f}. Same model, same weights, same held-out problems, same forward pass. The only difference is which end of the answer it has to produce first.

That is worth sitting with. Writing the answer in the order a person *reads* it requires deciding the leading digit before the carries beneath it have been resolved, which is a parallel problem. Writing it in the order a person *computes* it makes each digit depend on the one before, which is a sequential problem, and a sequential problem is what the architecture is shaped for. The error positions agree: in the reversed form the first character — the units digit, the one that needs no carry at all — is wrong {where['leading_digit_share']:.1%} of the time, while {second:.0%} of the failures start at the second character, which is exactly where a carry first arrives.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        f"""Several places, and the first one is the biggest.

**The shortcut is mine.** I generated this corpus, and I put the twins next to each other. I am not claiming that anyone else's training data has this particular defect, and a reader who takes "shuffle your corpus" as the lesson has taken the smallest available one — everybody already shuffles, and it did not help, because the standard kind of shuffling operates on the wrong thing. The transferable claim is narrower and worse: **a regularity in the order of your examples is invisible to any random split of them**, and templated, scraped, versioned or augmented corpora are full of order.

**Two runs differ by more than one thing, strictly.** The seed, steps, schedule, architecture and problem set are identical, but a shuffled stream and an unshuffled one put different characters in the window at step *n*, so the two runs do not see literally the same batches. What I can say is that the content is identical and the sampling procedure is identical; what I cannot say is that no other difference exists.

**The carry result is one model's.** {carry[('add', 2)]:.3f} at a chain of two is a property of an {res['clean']['parameters']:,}-parameter model trained for {20_000:,} steps, not a theorem about transformers. A bigger model, or a longer run, may well carry the rule across. The *comparison* between the two answer orders is the durable part, because both numbers come from the same weights.

**The leaky model is not uniformly broken**, which is what makes it dangerous rather than merely wrong. Its forward addition scores {leaky_tasks['add']:.3f} — the shortcut was only available for the reversed task, so it learned the rest more or less normally. A model that failed everywhere would have been caught in an afternoon.""")

    post.add(
        "What to keep",
        f"""1. Two models differing only in the order of the lines in their training file. One adds; the other reverses the line above it — including when that line is deliberately wrong, which it reproduces {ab['generation order']['reproduced the wrong sum, reversed']:.0%} of the time while producing the right answer in {ab['generation order']['answered correctly anyway'] * ab['n']:.0f} of {ab['n']} cases.

2. Held-out loss did not merely miss it — it **preferred the broken model**, {loss['generation order']['own packing']:.4f} against {loss['shuffled']['own packing']:.4f}, for the whole run. The shortcut works on held-out data too, because the held-out data was packed the same way.

3. Random batching is not protection. The adjacency lived inside the 64-character window, and shuffling which windows you draw does nothing to what is in one.

4. Two diagnostics did work, and both are one line. **Repack the validation set** in a different order and score again: {repack_gap:.4f} nats of separation where the standard packing gave {own_gap:.4f} in the wrong direction. And **vary the context** and see whether the answer moves — a model doing the arithmetic is indifferent to what precedes the question.

5. The control that settles it is the corrupted one. Feeding a *wrong* premise separates "computed it" from "copied it" in a way that feeding a right one never can.

6. On the honest model, carrying was learned at a width rather than as a rule: {carry[('add', 1)]:.3f} on chains it saw at three digits, {carry[('add', 3)]:.3f} on chains it did not.

7. And the same sums, written least-significant-digit-first, recover to {carry[('add_rev', 3)]:.3f} with nothing else changed. The order you ask for the answer in is not presentation.""")

    post.add(
        "Exercise",
        """Take a model you trained and a task it is good at. Build three contexts for the same question: the one your evaluation harness uses, one with unrelated content in front of it, and one containing a plausible but wrong answer to the question you are about to ask. Then compare the three accuracies.

If they agree, you have learned something solid and it cost you twenty minutes. If the third one is where the accuracy lives, your model has been reading rather than working, and no validation curve was ever going to tell you.

The uncomfortable version: do it on a benchmark rather than your own model. Contamination and context-copying produce exactly the same signature as competence when you only ever score the arm where the answer is nearby.""")

    post.add(
        "Next",
        f"""Episode 2 is the carry cliff on its own terms: why {carry[('add', 3)]:.3f} and not zero, where in the answer the error lands, and what the reversed form is actually buying. It ends at a fourth digit, where both forms go to {length[('add', 4)]:.3f} and the model stops producing answers of the right length at all — which turns out to be a statement about place value rather than about addition.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
