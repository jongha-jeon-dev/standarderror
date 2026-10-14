"""ML 22: Learning Rate and Batch Size Are One Knob, Until the Rate Hits a Wall.

The twenty-second episode of *Machine Learning, Taught Through What Breaks*.
"Learning rate and batch size are separate knobs." Measured with plain SGD
on a two-layer network on the digits, with the same number of examples seen
at every batch size:

* Away from divergence, the training loss depends on the learning rate and
  the batch size only through their ratio. Multiplying both by four gives
  the same loss to within 2.3%, across batches of 8 to 512.
* At a rate of 3.072 training fails (mean training loss above 1) at batch
  sizes of 128 and 512 alike: it diverges at 128 and degrades at 512.
  Batches of 8 and 32
  find their best rate (0.192 and 0.768) below the wall; batches of 128 and
  512 would need 3.1 and 12.3 and stop at 1.536, with training loss 3 and 55
  times higher.

Run: `standarderror run ml122_lr_batch --publish`
"""

from __future__ import annotations

import math
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
    r = nets.lr_batch()
    L = np.asarray(r["loss"])
    lrs, Bs = r["lrs"], r["batches"]
    div = []
    for i in range(len(Bs)):
        k = int(np.argmin(L[i]))
        div.append(next(lrs[j] for j in range(k, len(lrs)) if L[i, j] > 1))
    pairs = []
    for i in range(len(Bs) - 1):
        for j in range(len(lrs) - 2):
            a, b = L[i, j], L[i + 1, j + 2]
            if a < 1 and b < 1:
                # "far": both runs at least four times below the rate at
                # which their own batch size fails (mean training loss > 1)
                far = lrs[j] * 4 <= div[i] and lrs[j + 2] * 4 <= div[i + 1]
                pairs.append({"B": Bs[i], "lr": lrs[j], "a": float(a),
                              "b": float(b), "rel": float(abs(b - a) / a),
                              "far": far})
    return {"r": r, "L": L, "pairs": pairs, "div": div,
            "far_max": max(p["rel"] for p in pairs if p["far"]),
            "near_max": max(p["rel"] for p in pairs if not p["far"])}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    r, L = res["r"], res["L"]
    lrs, Bs = np.array(r["lrs"]), r["batches"]

    def by_lr(ax, m, ratio=False):
        for i, B in enumerate(Bs):
            x = lrs / B if ratio else lrs
            y = np.minimum(L[i], 3.0)
            ax.plot(x, y, marker="o", ms=3.5, color=m.series[i],
                    label=f"batch {B}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(3e-4, 4)
        ax.legend(frameon=False, fontsize=8.5,
                  loc="lower left" if ratio else "lower right")

    out["f0"] = charts.diagram(
        by_lr,
        title="Four batch sizes, four curves",
        subtitle=("Training loss after 20 epochs' worth of examples, plain "
                  "SGD, against the learning rate. Average of two seeds; "
                  "losses above 3 are drawn at 3."),
        xlabel="learning rate",
        ylabel="training loss (log scale)",
        figsize=(7.2, 4.6),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("Four falling curves on log axes, each shifted to the right of "
             "the last, all cut off where training fails, the two right-hand "
             "ones at the same learning rate near 3."),
        caption=("Each fourfold increase in batch size moves the curve two "
                 "grid points to the right, until the curve runs into the "
                 "same wall."),
        path=str(IMG / f"ml122-f0-lr.{EXT}"))[0]

    out["f1"] = charts.diagram(
        lambda ax, m: by_lr(ax, m, ratio=True),
        title="One curve, until the wall",
        subtitle=("The same runs against learning rate divided by batch size. "
                  "Below the wall the four curves are one."),
        xlabel="learning rate / batch size",
        ylabel="training loss (log scale)",
        figsize=(7.2, 4.6),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("The four curves collapse onto one falling line; the batches of "
             "8 and 32 leave it together, the batch of 128 earlier and the "
             "batch of 512 first."),
        caption=(f"Away from the wall, runs with the same ratio reach the "
                 f"same loss to within {res['far_max']:.1%}. The batch of 512 "
                 f"leaves the shared curve first."),
        path=str(IMG / f"ml122-f1-ratio.{EXT}"))[0]

    rows = []
    for i, B in enumerate(Bs):
        rows.append([f"{B}", f"{r['best_lr'][i]:.3f}",
                     f"{r['best_lr'][i] / B:.4f}", f"{r['best_loss'][i]:.4f}",
                     f"{r['best_acc'][i]:.3f}", f"{res['div'][i]:.3f}",
                     f"{0.192 * B / 8:.3f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["batch", "best rate", "rate / batch", "training loss",
                "test accuracy", "first rate that fails (loss > 1)",
                "rate the ratio asks for"],
        title="Where the ratio stops working",
        subtitle=("Best learning rate on a grid of doublings from 0.003, "
                  "plain SGD, the same 20 epochs' worth of examples at every "
                  "batch size; average of two seeds."),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("A four-row table: the best rate grows fourfold from batch 8 to "
             "32 and stops at 1.536; training loss rises sharply for the "
             "batch of 512."),
        caption=("Batches of 128 and 512 would need rates of 3.072 and 12.288 "
                 "to keep the ratio. Both rates fail."),
        bold_cells={(3, 3), (3, 6)},
        align="rrrrrrr",
        path=str(IMG / f"ml122-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    r, L = res["r"], res["L"]
    lrs = np.array(r["lrs"])

    def raw(panel, m):
        for i in range(4):
            panel.plot(np.log(lrs), np.log(np.minimum(L[i], 3)),
                       color=m.series[i], lw=1.6)

    def ratio(panel, m):
        for i, B in enumerate(r["batches"]):
            panel.plot(np.log(lrs / B), np.log(np.minimum(L[i], 3)),
                       color=m.series[i], lw=1.6)

    def best(panel, m):
        panel.bar(range(4), np.log10(np.array(r["best_loss"]) / 1e-4),
                  color=m.series[1])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=22,
        headline="One knob, then a wall",
        panels=[(raw, "4 curves", "loss against rate"),
                (ratio, f"±{res['far_max']:.1%}", "same rate / batch"),
                (best, f"{r['best_loss'][-1] / r['best_loss'][0]:.0f}x",
                 "loss, batch 512 vs 8")],
        note=("With plain SGD and a fixed number of examples, the loss "
              "depends on learning rate over batch size, until the rate "
              "needed exceeds what training can survive."),
        alt=("Three hand-drawn frames: four shifted curves; four curves "
             "lying on one another; four bars rising at the end."),
        path=str(IMG / f"ml122-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["best"] = s.run("""
        from standarderror.ml import nets

        r = nets.lr_batch()     # plain SGD, 20 epochs of examples, 2 seeds
        for B, lr, loss, acc in zip(r["batches"], r["best_lr"], r["best_loss"],
                                    r["best_acc"]):
            print(f"batch {B:3d}: best rate {lr:.3f}  training loss {loss:.4f}  "
                  f"test {acc:.3f}")
        """)
    out["ratio"] = s.run("""
        import math

        for ratio in (0.00075, 0.003, 0.012):     # learning rate / batch size
            row = []
            for i, B in enumerate(r["batches"]):
                j = round(math.log2(ratio * B / 0.003))   # grid is 0.003 * 2**j
                row.append(f"B={B}: {r['loss'][i][j]:.4f}")
            print(f"rate/batch {ratio:.5f}   " + "  ".join(row))
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    r, L, div = res["r"], res["L"], res["div"]
    blr, bl, ba = r["best_lr"], r["best_loss"], r["best_acc"]
    assert math.isclose(r["lrs"][0], 0.003)

    # The spine, asserted rather than trusted.
    assert blr[0] == r["lrs"][6] and blr[1] == r["lrs"][8]
    assert blr[2] == blr[3] == r["lrs"][9]
    assert res["far_max"] < 0.025
    assert res["near_max"] > 0.3
    assert div[0] == r["lrs"][7] and div[1] == r["lrs"][9]
    assert div[2] == div[3] == r["lrs"][10]
    assert bl[1] < 0.001 and bl[2] / bl[1] > 2.5 and bl[3] / bl[1] > 40
    assert ba[1] - ba[3] > 0.005
    assert abs(bl[0] - bl[1]) / bl[1] < 0.15
    assert L[0, 1] < L[1, 3] < L[2, 5] < L[3, 7] < 0.19
    assert max(L[0, 3], L[1, 5], L[2, 7]) / min(L[0, 3], L[1, 5], L[2, 7]) < 1.01
    assert L[2, 8] > 2.5 * L[1, 8]
    # "Fails" is what `div` measures: mean training loss above 1. A network
    # that has learned nothing scores ln 10 = 2.30. The batch of 512 at the
    # wall is degraded, not diverged; the others are at or near 2.3 or worse.
    assert L[0, 7] > 1.5 and L[1, 9] > 2.2 and L[2, 10] > 2.2
    assert 1.0 < L[3, 10] < 2.0 and L[3, 11] > 2.2

    post = Post(
        title=(f"{SERIES_TAG} 22: Learning Rate and Batch Size Are One Knob, "
               f"Until the Rate Hits a Wall"),
        slug="ml-22-lr-batch",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=22,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("With plain SGD and a fixed number of examples, multiplying "
                  "the batch size and the learning rate by the same factor "
                  "gives the same training run, until the learning rate "
                  "reaches the point where training fails."),
        summary=(
            f"Plain SGD on a two-layer network on the digits, with 20 "
            f"epochs' worth of examples at every batch size, swept over "
            f"learning rates from 0.003 to 6.144. Away from divergence, runs "
            f"with the same ratio of learning rate to batch size reached the "
            f"same training loss to within {res['far_max']:.1%}, from "
            f"batches of 8 to 512. The best rate grew fourfold from batch 8 "
            f"({blr[0]:.3f}) to 32 ({blr[1]:.3f}). At a rate of "
            f"{div[2]:.3f} training failed (mean training loss above 1) at "
            f"batch sizes of 128 and 512 alike, so those stopped "
            f"at {blr[2]:.3f}, where their training loss was "
            f"{bl[2] / bl[1]:.0f} and {bl[3] / bl[1]:.0f} times the batch of "
            f"32's and test accuracy {ba[2]:.3f} and {ba[3]:.3f} against "
            f"{ba[1]:.3f}."),
        tags=["machine-learning", "deep-learning", "neural-networks",
              "optimization", "sgd", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Optical Recognition of Handwritten Digits (UCI), as bundled with "
            "scikit-learn, CC BY 4.0; one stratified half split, "
            "standardised on the training half.",
            "Machinery: `standarderror/ml/nets.py`, tested in "
            "`tests/test_ml.py`; a PyTorch network 64-128-128-10 with ReLU "
            "and PyTorch's default initialisation, plain SGD (no momentum, "
            "no weight decay), 20 x 898 / B steps at batch size B, seeds 0 "
            "and 1.",
            "Where this stops: Goyal et al., \"Accurate, large minibatch SGD: "
            "training ImageNet in 1 hour\" (2017); Smith and Le, \"A Bayesian "
            "perspective on generalization and stochastic gradient descent\", "
            "*ICLR* (2018); McCandlish, Kaplan, Amodei et al., \"An empirical "
            "model of large-batch training\" (2018).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("split seed 0; network and shuffling seeds 0 and "
                            "1; torch.set_num_threads(1)"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Two settings on every training script",
        f"""Episode 21 found that a network's initial scale decides whether it trains at all. The next two settings every training script exposes are the learning rate and the batch size, and they are usually presented as separate choices: the batch size is set by the memory available, the learning rate is then tuned. The sentence under test is that division of labour: *learning rate and batch size are separate knobs*.

The experiment is a grid. A network with two hidden layers of 128 units is trained on half the digits with plain stochastic gradient descent: no momentum, no weight decay, nothing adaptive. Batch sizes are 8, 32, 128 and 512; learning rates double from 0.003 to 6.144. Every run sees the same number of examples, 20 passes' worth, so a batch of 8 takes 2,245 steps and a batch of 512 takes 35. The score is the training loss at the end, averaged over two seeds, because the question is about optimisation.

{snip['best'].markdown()}

The best learning rate for batches of 32 is **four times** the best for batches of 8, and the two reach the same loss and the same test accuracy. That is the linear scaling rule: multiply the batch size by k and multiply the learning rate by k. For batches of 128 and 512 the rule asks for 3.072 and 12.288, and they stop at {blr[2]:.3f}.""",
        figures=[figs["f0"]])

    post.add(
        "The ratio is the setting",
        f"""The rule holds far more exactly than "best rate" shows. Fix the ratio of learning rate to batch size and read the loss across batch sizes:

{snip['ratio'].markdown()}

At a ratio of 0.00075 the four batch sizes reach {L[0, 1]:.3f} to {L[3, 7]:.3f}. At 0.003 the first three agree to within 1%. Over every pair of runs on the grid where the batch and the rate are both four times larger, and both rates are at least four times below the rate at which their batch size fails (defined below), the losses differ by at most **{res['far_max']:.1%}**. Closer to failure the pairs part, by up to {res['near_max']:.0%}. Plotted against the ratio, the four curves in the second figure are one curve.

There is a simple reason. With a fixed number of examples, a batch four times larger means four times fewer steps, and a rate four times larger means each step goes four times as far; at small rates, where the gradient changes little over a few steps, four small steps on four batches land almost where one big step on their union does. A second reason gives the same ratio: the noise in a minibatch gradient has a variance proportional to 1/B, and Smith and Le showed that the size of the random kicks SGD gives the weights scales with the learning rate divided by the batch. Either way, the run is set by lr/B, and the two "separate knobs" are one.""",
        figures=[figs["f1"]])

    post.add(
        "The wall",
        f"""The last line of the output is where the rule ends. At a ratio of 0.012 the batch of 512 needs a learning rate of 6.144, and the run ends with a mean training loss of {L[3, 11]:.2f}. Call a run *failed* when its mean training loss ends above 1; a network that has learned nothing scores ln 10 = 2.30. Every batch size fails somewhere: the batch of 8 at {div[0]:.3f} (mean loss {L[0, 7]:.2f}), the batch of 32 at {div[1]:.3f} ({L[1, 9]:.2f}), and the batches of 128 and 512 both at **{div[2]:.3f}**. At that rate the batch of 128 diverges outright ({L[2, 10]:.2f}); the batch of 512 is degraded rather than diverged, ending at {L[3, 10]:.2f}, worse than at any smaller rate but still below a network that has learned nothing.

For the two small batches the failure point follows the ratio too, at 0.048 per example. For the two large ones it does not: it is the same learning rate for both. That is what a limit unrelated to the batch would look like, though with two batch sizes and a grid of doublings it is suggestive rather than shown. Gradient descent on a loss whose sharpest curvature is λ is stable only for learning rates below about 2/λ, a property of the loss surface, and it binds the full-batch step and the large-batch step alike. The small batches hit a limit that scales with the batch, as a limit set by gradient noise would; the large ones hit one that does not, as a limit set by curvature would. Curvature itself was not measured here.

Once a batch size needs a rate above that wall, the linear scaling rule has nothing left to offer. Here the batch of 128 reached a training loss {bl[2] / bl[1]:.1f} times the batch of 32's at its best rate, and the batch of 512, {bl[3] / bl[1]:.0f} times; test accuracy fell from {ba[1]:.3f} to {ba[3]:.3f}. With 35 steps in total, the likeliest reading is that it did not travel far enough, and the only way to travel further is more steps. That is what McCandlish and colleagues call the critical batch size: beyond it, a larger batch buys little reduction in the number of steps needed, so it mostly spends more computation.""",
        figures=[figs["f2"]])

    post.add(
        "What this means for a training script",
        f"""The practical rule that follows is not "tune both". It is: treat lr/B as the setting, and change the batch size only with the learning rate in proportion, while the rate stays well below the point where training fails. Below the wall that is free: the batch of 32 and the batch of 8 reached the same loss, {bl[1]:.4f} and {bl[0]:.4f}, with a quarter of the steps. Above it, a larger batch is a decision to train less, and should be paid for with more epochs.

It also says how to read a tuning result. A learning rate tuned at one batch size is a ratio tuned at that batch size. Copying the rate to a run with four times the batch, as happens when a script moves to a larger machine, divides the effective setting by four. On this grid, the batch of 32's best rate, {r['lrs'][8]:.3f}, gives the batch of 128 a training loss of {L[2, 8]:.4f} instead of {L[1, 8]:.4f}.

Goyal and colleagues trained ImageNet in an hour on this principle, with batches of 8,192 and a warm-up to reach the large rate without diverging early. Their warm-up, a few epochs of rising learning rate, exists to keep the large rate from diverging in the first steps, when the network changes fastest: the same wall, met at the start.""")

    post.add(
        "Where this breaks",
        """**Plain SGD only.** With momentum the effective step is the rate divided by one minus the momentum, and the same ratio argument applies to that; with Adam, whose step size does not scale with the gradient's magnitude, the relation is closer to a square root and depends on the noise. Episode 23 uses Adam.

**Training loss, not test accuracy.** The question was optimisation. Test accuracy followed the training loss closely here; on larger problems it can prefer the noisier, smaller-batch runs at equal training loss, a separate question with a contested answer.

**One network and one budget.** The position of the wall depends on the network's curvature, and the critical batch size depends on the task's gradient noise, so neither number transfers. What transfers is the shape: a ratio that sets the run, and a wall that does not move with the batch.

**A coarse grid.** Rates double, so "best" and "fails" are each known only to within a factor of two.""")

    post.add(
        "What to keep",
        f"""1. With plain SGD and a fixed number of examples, learning rate and batch size act through their ratio. Away from divergence, runs with the same ratio reached the same training loss to within {res['far_max']:.1%}, from batches of 8 to 512.

2. The best rate grew fourfold from a batch of 8 to a batch of 32 ({blr[0]:.3f} to {blr[1]:.3f}), with the same loss and test accuracy.

3. Training failed (mean training loss above 1) at a learning rate that did not move with the batch for large batches ({div[2]:.3f} for both 128 and 512: diverging at 128, degraded to {L[3, 10]:.2f} at 512). Batches that needed more than that reached a training loss {bl[2] / bl[1]:.0f} and {bl[3] / bl[1]:.0f} times higher.

4. Change the batch size and the learning rate together; beyond the wall, a larger batch means less training, not faster training.""")

    post.add(
        "Exercise",
        """Take a training script and find its best learning rate at your usual batch size. Then halve the batch size and run it at the old rate and at half the old rate, for the same number of epochs.

If the halved rate matches the original run and the old rate does not, your setting was the ratio. Then double the batch size instead, at twice the rate, and see whether it trains or diverges: you will have located your wall.""")

    post.add(
        "Next",
        """Plain SGD scales with the batch in a way that can be written down. Adam was designed to make the learning rate matter less and to converge faster. Episode 23 tunes both optimisers properly and runs them over twenty splits to see what is left of that claim.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
