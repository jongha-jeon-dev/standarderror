"""ML 27: Perfect Training Accuracy Was Not the End of Learning.

The twenty-seventh episode of *Machine Learning, Taught Through What Breaks*.
"Once training accuracy reaches 100%, learning is over." Measured on modular
addition, (a + b) mod 97, with a network trained full-batch on 40% of the
9,409 pairs:

* Training accuracy reaches 100% at step 200, when test accuracy is 0.0. It
  stays near zero for over a thousand steps more, then climbs to 99% by step
  4,900 while training accuracy never moves: the network replaces a
  memorised table with the rule.
* Throughout, the weight norm falls under weight decay. Without weight decay
  the network does not make the switch in 10,000 steps.
* With 60% of the pairs to train on, the switch comes at step 1,500; with
  30%, not within 10,000 steps.

Run: `standarderror run ml127_grokking --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import nets
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    runs = nets.grokking_runs()
    key = {(r["weight_decay"], r["frac"]): r for r in runs}
    for r in runs:
        h = r["history"]
        r["test_at_100"] = next(x["test"] for x in h
                                if x["step"] == r["train_100"]) \
            if r["train_100"] else None
        r["test_10"] = next((x["step"] for x in h if x["test"] >= 0.1), None)
        r["norm_at"] = {x["step"]: x["norm"] for x in h}
    short = nets.modular(steps=200, every=100)
    return {"runs": runs, "key": key, "short": short}


def _arr(r, k):
    return np.array([h["step"] for h in r["history"]]), \
        np.array([h[k] for h in r["history"]])


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    key = res["key"]
    a, z = key[(1.0, 0.4)], key[(0.0, 0.4)]

    def curves(ax, m):
        for r, c, lab in ((a, 0, "weight decay 1"), (z, 1, "no weight decay")):
            s, tr = _arr(r, "train")
            _, te = _arr(r, "test")
            ax.plot(s, tr, color=m.series[c], lw=1.4, ls="--",
                    label=f"training, {lab}")
            ax.plot(s, te, color=m.series[c], lw=2.0,
                    label=f"test, {lab}")
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8.5, loc="center left")

    out["f0"] = charts.diagram(
        curves,
        title="A hundred per cent, and then the learning",
        subtitle=("Accuracy on the 40% of pairs (a, b) the network trains on "
                  "and on the other 60%, for (a + b) mod 97, every 100 full-"
                  "batch steps."),
        xlabel="training step (log scale)",
        ylabel="accuracy",
        figsize=(7.4, 4.6),
        source="Generated arithmetic; standarderror/ml/nets.py.",
        alt=("Two dashed training curves reach 1.0 within a few hundred "
             "steps; the solid test curve with weight decay stays near zero "
             "for about two thousand steps and then climbs to 1.0; the test "
             "curve without weight decay stays low."),
        caption=(f"With weight decay: training accuracy 100% at step "
                 f"{a['train_100']}, test accuracy 99% at step "
                 f"**{a['test_99']:,}**. Without: test accuracy "
                 f"{z['final_test']:.1%} at step {z['steps']:,}."),
        path=str(IMG / f"ml127-f0-curves.{EXT}"))[0]

    def norms(ax, m):
        for r, c, lab in ((a, 0, "weight decay 1"), (z, 1, "no weight decay")):
            s, n = _arr(r, "norm")
            ax.plot(s, n, color=m.series[c], lw=2.0, label=lab)
        if a["test_99"]:
            ax.axvline(a["test_99"], color=m.ink, lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=8.5, loc="center right")

    out["f1"] = charts.diagram(
        norms,
        title="What changes while nothing seems to",
        subtitle=("Euclidean norm of all the network's parameters. The dotted "
                  "line marks where test accuracy reaches 99% with weight "
                  "decay."),
        xlabel="training step",
        ylabel="parameter norm",
        figsize=(7.2, 4.4),
        source="Generated arithmetic; standarderror/ml/nets.py.",
        alt=("The weight-decay curve falls steadily from about 110 to about "
             "65; the curve without weight decay rises instead."),
        caption=(f"With weight decay the norm falls from "
                 f"{a['norm_at'][a['train_100']]:.0f} when the training set "
                 f"is fitted to {a['norm_at'][a['test_99']]:.0f} when the "
                 f"rule is found. Without it, the norm ends at "
                 f"{z['history'][-1]['norm']:.0f}."),
        path=str(IMG / f"ml127-f1-norm.{EXT}"))[0]

    rows = []
    for r in res["runs"]:
        rows.append([f"{r['frac']:.0%}", f"{r['weight_decay']:g}",
                     f"{r['train_100'] or '—'}",
                     f"{r['test_at_100']:.3f}" if r["test_at_100"] is not None
                     else "—",
                     f"{r['test_10'] or 'never'}",
                     f"{r['test_99'] or 'never'}",
                     f"{r['final_test']:.3f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["pairs trained on", "weight decay", "step: 100% training",
                "test accuracy then", "step: test 10%", "step: test 99%",
                "test accuracy at 10,000"],
        title="When the rule arrives",
        subtitle=("Four runs of the same network on (a + b) mod 97; full-batch "
                  "AdamW, learning rate 1e-3, 10,000 steps. 'Never' means not "
                  "within 10,000 steps."),
        source="Generated arithmetic; standarderror/ml/nets.py.",
        alt=("A four-row table of when each run reaches 100% training "
             "accuracy and when its test accuracy passes 10% and 99%."),
        caption=("Every run fits its training pairs within a few hundred "
                 "steps. When, and whether, it then finds the rule depends on "
                 "weight decay and on how many pairs it sees."),
        align="rrrrrrr",
        path=str(IMG / f"ml127-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    key = res["key"]
    a, z = key[(1.0, 0.4)], key[(0.0, 0.4)]

    def tr(panel, m):
        s, t = _arr(a, "train")
        panel.plot(np.log10(s), t, color=m.series[0], lw=2)
        panel.set_ylim(0, 1.05)

    def te(panel, m):
        s, t = _arr(a, "test")
        panel.plot(np.log10(s), t, color=m.series[2], lw=2)
        panel.set_ylim(0, 1.05)

    def nowd(panel, m):
        s, t = _arr(z, "test")
        panel.plot(np.log10(s), t, color=m.series[1], lw=2)
        panel.set_ylim(-0.1, 1.05)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=27,
        headline="Learning after 100%",
        panels=[(tr, f"step {a['train_100']}", "100% on training pairs"),
                (te, f"step {a['test_99']:,}", "99% on unseen pairs"),
                (nowd, f"{z['final_test']:.0%}", "unseen, no weight decay")],
        note=("A network memorises its training pairs of a modular sum by "
              "step 200, scoring nothing on the rest, and finds the rule "
              "thousands of steps later, with training accuracy at 100% the "
              "whole time."),
        alt=("Three hand-drawn frames: a curve that rises at once; a curve "
             "that stays flat and then rises late; a curve that stays low."),
        path=str(IMG / f"ml127-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["runs"] = s.run("""
        from standarderror.ml import nets

        runs = nets.grokking_runs()   # written by scripts/train_grokking.py
        for r in runs:
            print(f"{r['frac']:.0%} of pairs, weight decay {r['weight_decay']:g}: "
                  f"100% training at step {r['train_100']}")
            reach = f"at step {r['test_99']}" if r["test_99"] else "not reached"
            print(f"    99% test {reach}; test after 10,000 steps "
                  f"{r['final_test']:.3f}")
        """)
    out["check"] = s.run("""
        short = nets.modular(steps=200, every=100)    # rerun the first 200 steps
        for a, b in zip(short["history"], runs[0]["history"]):
            print(f"step {a['step']}: train {a['train']:.4f} / {b['train']:.4f}, "
                  f"test {a['test']:.4f} / {b['test']:.4f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    key = res["key"]
    a, z = key[(1.0, 0.4)], key[(0.0, 0.4)]
    lo, hi = key[(1.0, 0.3)], key[(1.0, 0.6)]
    n_train = int(0.4 * 97 * 97)
    for x, y in zip(res["short"]["history"], a["history"]):
        assert x["train"] == y["train"] and x["test"] == y["test"]

    # The spine, asserted rather than trusted.
    assert a["train_100"] <= 300 and a["test_at_100"] < 0.011
    assert a["test_10"] > 5 * a["train_100"]
    assert a["test_99"] > 15 * a["train_100"] and a["final_test"] == 1.0
    assert all(h["train"] == 1.0 for h in a["history"]
               if h["step"] >= a["train_100"])
    na = a["norm_at"]
    assert na[a["test_99"]] < 0.75 * na[a["train_100"]]
    assert z["train_100"] is not None and z["test_99"] is None
    assert z["final_test"] < 0.5
    assert z["history"][-1]["norm"] > na[a["train_100"]]
    assert hi["test_99"] is not None and hi["test_99"] < a["test_99"]
    assert lo["test_99"] is None or lo["test_99"] > a["test_99"]

    post = Post(
        title=(f"{SERIES_TAG} 27: Perfect Training Accuracy Was Not the End "
               f"of Learning"),
        slug="ml-27-grokking",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=27,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A network trained on part of a modular-addition table "
                  "memorises it at once, scores nothing on the rest, and "
                  "finds the rule thousands of steps later, while its "
                  "training accuracy sits at 100% the whole time."),
        summary=(
            f"A small network trained full-batch on 40% of the pairs of "
            f"(a + b) mod 97 reached 100% training accuracy at step "
            f"{a['train_100']}, when its accuracy on the other pairs was "
            f"{a['test_at_100']:.3f}. Test accuracy passed 10% at step "
            f"{a['test_10']:,} and 99% at step {a['test_99']:,}, with "
            f"training accuracy at 100% throughout, while weight decay "
            f"brought the parameter norm from {na[a['train_100']]:.0f} to "
            f"{na[a['test_99']]:.0f}. Without weight decay, test accuracy "
            f"was {z['final_test']:.1%} after 10,000 steps. Training on 60% "
            f"of the pairs, the rule arrived at step {hi['test_99']:,}."),
        tags=["machine-learning", "deep-learning", "neural-networks",
              "generalization", "grokking", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Generated: all 9,409 pairs (a, b) with a, b in 0..96 and label "
            "(a + b) mod 97; a seeded random 40% (or 30%, 60%) for training, "
            "the rest for test.",
            "Machinery: `standarderror/ml/nets.py` (`modular`) and "
            "`scripts/train_grokking.py`, whose curves are committed in "
            "`data/ml_grokking/results.json`; tested in `tests/test_ml.py`. "
            "Embeddings of a and b (128 each) concatenated into one hidden "
            "layer of 256 ReLU units; full-batch AdamW, learning rate 1e-3, "
            "betas (0.9, 0.98), weight decay 1 or 0, 10,000 steps.",
            "Where this stops: Power, Burda, Edwards, Babuschkin and Misra, "
            "\"Grokking: generalization beyond overfitting on small "
            "algorithmic datasets\" (2022); Nanda, Chan, Lieberum, Smith and "
            "Steinhardt, \"Progress measures for grokking via mechanistic "
            "interpretability\", *ICLR* (2023); Liu, Michaud and Tegmark, "
            "\"Omnigrok: grokking beyond algorithmic data\", *ICLR* (2023).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the stored curves are "
                            "checked against a fresh 200-step run, and the "
                            "values the prose quotes are pinned"),
            "determinism": ("split and network seed 0, full batch, "
                            "torch.set_num_threads(1); each 10,000-step run "
                            "took between 7 and 26 minutes on one CPU "
                            "thread"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Nothing left to fit",
        f"""Episode 26 trained a network until it fitted every training label and watched its test accuracy fall as it memorised noise. The lesson there was to stop early. The sentence under test here is the belief that makes stopping natural: *once training accuracy reaches 100%, learning is over*. Every training example is classified correctly, so the gradient has nothing left to correct but confidence, and nothing more of substance can happen.

The test is a task with a rule and no noise: addition modulo 97. Every pair (a, b) of numbers from 0 to 96 has the label (a + b) mod 97, 9,409 pairs in all. A small network sees {n_train:,} of them, 40%, and is asked about the other 60%. It knows nothing about numbers: each of a and b is a learned embedding, the two embeddings are concatenated and passed through one hidden layer, and the output is one of 97 classes. It is trained full-batch with AdamW and weight decay, for 10,000 steps. The runs take long enough that their curves are committed to the repository; the episode reads them, and then checks the first 200 steps against a fresh run:

{snip['runs'].markdown()}

{snip['check'].markdown()}

The first run is the one to follow. Training accuracy reaches 100% at step **{a['train_100']}**. At that moment test accuracy is {a['test_at_100']:.3f}: not chance, which would be about 0.010, but below it. The network has learned a table of the {n_train:,} answers it was shown and nothing that transfers to any other pair.""",
        figures=[figs["f0"]])

    post.add(
        "Memorise first, generalise later",
        f"""By the sentence under test, training is over at step {a['train_100']}. It continues anyway, and for a long time nothing visible happens. Training accuracy stays at 100% at every checkpoint. Test accuracy stays under 10% until step {a['test_10']:,}, {a['test_10'] / a['train_100']:.0f} times as long as it took to fit the training set. Then it climbs, steeply, and passes 99% at step **{a['test_99']:,}**. By the end of the run the network answers every one of the 5,646 pairs it never saw.

Power and colleagues, who first reported this on small algorithmic tasks, named it *grokking*. Nanda and colleagues later opened up a small transformer trained on the same task and found what it had learned: its embeddings place the numbers on circles, and it adds angles, which is modular addition done with trigonometry. Whether this network found the same solution was not checked here; what is measured is that it found one that answers the unseen pairs. A memorised table and a rule both fit the training pairs perfectly. Only the rule answers the rest.

So "100% training accuracy" does not identify what the network has learned. Two quite different functions, a lookup table and the rule, score identically on the training set, and training moved from the first to the second while the score on the training set never changed.""")

    post.add(
        "What moved the network",
        f"""If the training labels were not pushing, something else was. With weight decay, every step shrinks every parameter slightly, and the gradient only pushes back where the training loss needs it. The parameter norm in the figure below falls steadily from {na[a['train_100']]:.0f} when the training set is first fitted to **{na[a['test_99']]:.0f}** when the rule is found, and on to {a['history'][-1]['norm']:.0f} at the end.

The reading that Liu, Michaud and Tegmark give, and that this curve fits, is that the memorised table is expensive in weight norm and the rule is cheap. Both fit the training pairs; weight decay keeps charging for size; the network drifts towards the solution that fits the data with the smallest weights, and that solution is the rule.

The control is to switch weight decay off and change nothing else. The second run does exactly that. It also fits the training set early, at step {z['train_100']}. Its parameter norm grows instead, from {z['norm_at'][z['train_100']]:.0f} at that step to {z['history'][-1]['norm']:.0f}. And after 10,000 steps its test accuracy is **{z['final_test']:.1%}**. Without the pressure on size, the memorised table is as good a resting place as any.""",
        figures=[figs["f1"]])

    post.add(
        "How much data the rule needs",
        f"""The last two runs keep weight decay and change the share of pairs in the training set. With 60% of the pairs the rule arrives sooner, test accuracy passing 99% at step {hi['test_99']:,} against {a['test_99']:,} with 40%. With 30% {('it arrives at step ' + format(lo['test_99'], ',')) if lo['test_99'] else 'it does not arrive within 10,000 steps'}, and test accuracy ends at {lo['final_test']:.3f}.

That is the shape Power and colleagues reported: the delay between memorising and generalising grows quickly as the training fraction shrinks. The fewer pairs there are, the cheaper a table that fits them, and the longer weight decay takes to make the rule the cheaper of the two.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        """**A task with an exact rule.** Modular addition has a compact rule that fits every pair, including the unseen ones. Most real data have no such rule, only a noisy relationship, and episode 26's network, trained on real digits with noisy labels, got worse after fitting, not better. Grokking is a demonstration that the sentence is false, not evidence that training past 100% is generally wise.

**Strong weight decay, full batches.** A weight decay of 1 with AdamW is far stronger than usual, and full-batch training removes the noise of minibatches. Both make the slow drift easy to see. Power and colleagues report a longer delay with weaker decay; without any, here, the rule did not arrive at all.

**Early stopping would have failed.** A validation set drawn from the unseen pairs would have shown accuracy at 0 and stayed there for over a thousand steps after the training set was fitted. Any reasonable patience setting stops the run long before the rule arrives. The same tool that rescued episode 26's network discards this one.

**One seed per setting.** Each row of the table is one run. The timings would move with the seed; the pattern, fit first and generalise much later under weight decay, is the one reported across many seeds in the literature.""")

    post.add(
        "What to keep",
        f"""1. 100% training accuracy says the network fits the training set, not how. Here a memorised table and the rule both fitted it perfectly.

2. Learning continued after the training set was fitted. Training accuracy reached 100% at step {a['train_100']}, test accuracy was {a['test_at_100']:.3f} then, and it passed 99% at step {a['test_99']:,}.

3. Weight decay drove the switch: the parameter norm fell from {na[a['train_100']]:.0f} to {na[a['test_99']]:.0f} on the way. Without it, test accuracy was {z['final_test']:.1%} after 10,000 steps.

4. More training data brought the rule sooner: step {hi['test_99']:,} with 60% of the pairs.

5. Neither "stop at 100%" nor "train past it" is a rule. What happens after the training set is fitted depends on whether there is a simpler solution to find and whether anything pushes the network towards it.""")

    post.add(
        "Exercise",
        """Take a small model you train to 100% training accuracy and keep training it for ten times as many steps, with weight decay, recording validation accuracy and the parameter norm every few hundred steps.

If validation accuracy rises again after the training set is fitted, your early-stopping rule would have thrown that away. If it only falls, your data are more like episode 26's than this one's, and stopping early was right.""")

    post.add(
        "Next",
        """Weight decay encoded a preference the data could not express: for small weights. Episode 28 looks at the most common way of encoding a preference about the data itself, augmentation, and at the sentence that it simply adds more data.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
