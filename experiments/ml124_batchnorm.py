"""ML 24: Batch Normalisation Normalises the Batch, Not the Example.

The twenty-fourth episode of *Machine Learning, Taught Through What Breaks*.
"Batch normalisation normalises each example." Measured on a two-layer
network on the digits:

* In training mode, one test digit's predicted label changes with the other
  digits in its batch: in 45% of random batches of 2, 20% of batches of 4,
  3.2% of batches of 16 and 0.6% of batches of 64. In a batch of 16 whose
  other members all share one class, it changes 23% of the time.
* Trained in batches of 2, the normalised network scores 0.863 in inference
  mode against 0.972 for the same network without normalisation; from
  batches of 16 up, normalisation helps by about half a point.

Run: `standarderror run ml124_batchnorm --publish`
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
    d = nets.batchnorm_dependence()
    b = nets.batchnorm_batch_size()
    return {"d": d, "b": b, "rows": {r["batch"]: r for r in d["rows"]},
            "train": {r["batch"]: r for r in b}}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    d, rows = res["d"], res["d"]["rows"]

    def flips(ax, m):
        x = np.arange(len(rows))
        ax.bar(x, [r["flip"] for r in rows], 0.6, color=m.series[0],
               label="random companions")
        i16 = [r["batch"] for r in rows].index(d["skewed_batch"])
        ax.bar([i16 + 0.42], [1 - d["skewed_agree"]], 0.25,
               color=m.series[1], label="companions all from one class")
        ax.set_xticks(x)
        ax.set_xticklabels([f"batch of {r['batch']}" for r in rows])
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    r16 = res["rows"][16]
    out["f0"] = charts.diagram(
        flips,
        title="One digit, many answers",
        subtitle=("Share of batches in which a test digit's label, computed "
                  "in training mode, differs from its label in inference "
                  "mode. 100 test digits, 200 random batches each."),
        ylabel="share of batches with a different label",
        figsize=(7.2, 4.5),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("Bars falling from about 0.45 for batches of 2 to near zero for "
             "batches of 64, with a taller orange bar beside the batch-of-16 "
             "bar for companions from a single class."),
        caption=(f"With 15 random companions the label changes in "
                 f"{r16['flip']:.1%} of batches; with 15 companions of one "
                 f"class, in **{1 - d['skewed_agree']:.1%}**."),
        path=str(IMG / f"ml124-f0-flips.{EXT}"))[0]

    b = res["b"]

    def train(ax, m):
        x = np.arange(len(b))
        w = 0.27
        ax.bar(x - w, [r["plain"] for r in b], w, color=m.series[2],
               label="no normalisation")
        ax.bar(x, [r["bn"] for r in b], w, color=m.series[0],
               label="batch norm, inference mode")
        ax.bar(x + w, [r["bn_train_mode"] for r in b], w, color=m.series[1],
               label="batch norm, training mode")
        ax.set_xticks(x)
        ax.set_xticklabels([f"batches of {r['batch']}" for r in b],
                           fontsize=8.5)
        ax.set_ylim(0.5, 1.0)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    t = res["train"]
    out["f1"] = charts.diagram(
        train,
        title="Small batches break the network it learns",
        subtitle=("Test accuracy of the same network trained with and "
                  "without batch normalisation at each batch size (Adam, 20 "
                  "epochs, three seeds). Training mode uses test batches of "
                  "the training size."),
        ylabel="test accuracy",
        figsize=(7.8, 4.6),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("Grouped bars: the plain network stays near 0.97 at every batch "
             "size; the normalised one is far lower at batch 2 and slightly "
             "higher from batch 16; training mode is lowest at small "
             "batches."),
        caption=(f"Trained in batches of 2, the normalised network scores "
                 f"**{t[2]['bn']:.3f}** in inference mode and "
                 f"{t[2]['bn_train_mode']:.3f} in training mode, against "
                 f"{t[2]['plain']:.3f} without normalisation."),
        path=str(IMG / f"ml124-f1-train.{EXT}"))[0]

    trs = []
    for r in rows:
        trs.append([f"{r['batch']}", f"{r['flip']:.1%}", f"{r['any_flip']:.0%}",
                    f"{r['accuracy']:.3f}", f"{t[r['batch']]['plain']:.3f}",
                    f"{t[r['batch']]['bn']:.3f}",
                    f"{t[r['batch']]['bn_train_mode']:.3f}"])
    out["f2"] = charts.table_image(
        trs,
        header=["batch size", "label changes", "digits that ever change",
                "accuracy, training mode", "trained at this size: plain",
                "batch norm", "batch norm, training mode"],
        title="The batch as an input",
        subtitle=(f"Left: one trained network (inference-mode accuracy "
                  f"{d['sample_eval_acc']:.3f} on the 100 digits), evaluated "
                  f"in training mode inside random batches. Right: networks "
                  f"trained at each batch size, on all 899 test digits."),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("A four-row table: label changes and their spread fall with "
             "batch size; the trained-at-this-size accuracies show batch norm "
             "losing at 2 and winning at 16 and 64."),
        caption=("Both halves of the table measure the same thing from two "
                 "sides: a normalised network's output is a function of its "
                 "batch."),
        bold_cells={(0, 1), (0, 5)},
        align="rrrrrrr",
        path=str(IMG / f"ml124-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    d, t = res["d"], res["train"]
    rows = d["rows"]

    def flips(panel, m):
        panel.bar(range(len(rows)), [r["flip"] for r in rows],
                  color=m.series[0])

    def skew(panel, m):
        panel.bar([0, 1], [res["rows"][16]["flip"], 1 - d["skewed_agree"]],
                  color=[m.series[0], m.series[1]])

    def small(panel, m):
        panel.bar([0, 1], [t[2]["plain"], t[2]["bn"]],
                  color=[m.series[2], m.series[0]])
        panel.set_ylim(0.5, 1.0)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=24,
        headline="The batch is an input",
        panels=[(flips, f"{rows[0]['flip']:.0%}", "label changes, batch 2"),
                (skew, f"{1 - d['skewed_agree']:.0%}",
                 "one-class batch of 16"),
                (small, f"{t[2]['bn']:.3f}", "trained in batches of 2")],
        note=("In training mode a normalised network's answer for one digit "
              "depends on the other digits in its batch. Trained in tiny "
              "batches, it learns a function that inference mode cannot "
              "reproduce."),
        alt=("Three hand-drawn frames: bars falling from left to right; a "
             "short bar beside a tall one; a tall bar beside a shorter one."),
        path=str(IMG / f"ml124-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["dep"] = s.run("""
        from standarderror.ml import nets

        r = nets.batchnorm_dependence()   # 100 test digits, 200 batches each
        print(f"inference mode: accuracy {r['sample_eval_acc']:.3f}")
        for row in r["rows"]:
            print(f"training mode, batch of {row['batch']:2d}: label changes in "
                  f"{row['flip']:5.1%} of batches, accuracy {row['accuracy']:.3f}")
        print(f"batch of 16 from one class: same label as inference mode in "
              f"{r['skewed_agree']:.1%}")
        """)
    out["train"] = s.run("""
        for row in nets.batchnorm_batch_size():
            print(f"batches of {row['batch']:2d}: plain {row['plain']:.3f}  "
                  f"batch norm {row['bn']:.3f}  "
                  f"(training mode {row['bn_train_mode']:.3f})")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    d, R, t = res["d"], res["rows"], res["train"]
    flips = [r["flip"] for r in d["rows"]]

    # The spine, asserted rather than trusted.
    assert all(a > b for a, b in zip(flips, flips[1:]))
    assert R[2]["flip"] > 0.4 and R[4]["flip"] > 0.15
    assert R[16]["flip"] < 0.05 and R[64]["flip"] < 0.01
    assert R[2]["any_flip"] == 1.0 and R[64]["any_flip"] < 0.15
    assert 1 - d["skewed_agree"] > 5 * R[16]["flip"]
    assert R[64]["accuracy"] > d["sample_eval_acc"] - 0.01
    assert t[2]["plain"] - t[2]["bn"] > 0.08
    assert t[2]["bn_train_mode"] < 0.65
    assert all(t[b]["bn"] > t[b]["plain"] for b in (16, 64))
    assert all(abs(t[b]["plain"] - 0.97) < 0.01 for b in t)
    assert t[64]["bn_train_mode"] > t[64]["bn"] - 0.01

    post = Post(
        title=(f"{SERIES_TAG} 24: Batch Normalisation Normalises the Batch, "
               f"Not the Example"),
        slug="ml-24-batchnorm",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=24,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("In training mode, a normalised network's answer for one "
                  "digit depends on which other digits share its batch. "
                  "That is fine in large, mixed batches and breaks in small "
                  "or uniform ones."),
        summary=(
            f"A two-layer network with batch normalisation scored "
            f"{d['sample_eval_acc']:.3f} on 100 test digits in inference "
            f"mode. In training mode, each digit's label changed with its "
            f"random batch companions in {R[2]['flip']:.0%} of batches of 2, "
            f"{R[4]['flip']:.0%} of batches of 4, {R[16]['flip']:.1%} of "
            f"batches of 16 and {R[64]['flip']:.1%} of batches of 64; with "
            f"15 companions from one class, in "
            f"{1 - d['skewed_agree']:.0%}. Trained in batches of 2, the "
            f"normalised network scored {t[2]['bn']:.3f} in inference mode "
            f"against {t[2]['plain']:.3f} without normalisation; trained in "
            f"batches of 16 or 64 it gained about half a point."),
        tags=["machine-learning", "deep-learning", "neural-networks",
              "batch-normalization", "training", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Optical Recognition of Handwritten Digits (UCI), as bundled with "
            "scikit-learn, CC BY 4.0; stratified half splits, standardised "
            "on the training half.",
            "Machinery: `standarderror/ml/nets.py`, tested in "
            "`tests/test_ml.py`; PyTorch networks 64-128-128-10 with ReLU and "
            "BatchNorm1d before each activation. The dependence study trains "
            "one network with SGD (momentum 0.9, learning rate 0.05, batches "
            "of 64, 20 epochs) and evaluates it with the running statistics "
            "frozen; the batch-size study trains with Adam (learning rate "
            "1e-3, 20 epochs) at each batch size, seeds 0 to 2.",
            "Where this stops: Ioffe and Szegedy, \"Batch normalization\", "
            "*ICML* (2015); Ioffe, \"Batch renormalization: towards reducing "
            "minibatch dependence in batch-normalized models\", *NeurIPS* "
            "(2017); Wu and He, \"Group normalization\", *ECCV* (2018).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("split and network seed 0 for the dependence "
                            "study, companions drawn from numpy seed 0; seeds "
                            "0 to 2 for the batch-size study; "
                            "torch.set_num_threads(1)"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A layer with two modes",
        f"""Episode 23 compared optimisers on networks with nothing between their layers but activations. Most modern networks have something else there: a normalisation layer. Batch normalisation, the first to be widely used, takes each unit's values across the examples in a minibatch, subtracts their mean, divides by their standard deviation, and then applies a learned scale and shift. Training is faster and less sensitive to the initial scale, which is the problem episode 21 measured.

The usual description is the sentence under test: *batch normalisation normalises each example*, rescaling its activations to a standard range. The mean and standard deviation, though, are not the example's. They are the batch's. During training, an example's output is computed from statistics of the other examples it happens to be batched with. At inference, the layer switches to running averages of those statistics collected during training, so that a prediction depends on the example alone. Two modes, two different functions.

How different, and does it matter? Train a 64-128-128-10 network with batch normalisation on half the digits in batches of 64. Then take 100 test digits and compute each one's prediction in training mode, inside 200 different random batches of test digits, with the running statistics frozen:

{snip['dep'].markdown()}

In inference mode the network gets {d['sample_eval_acc']:.3f} of these digits right. In training mode, inside random batches of 2, a digit's label differs from its inference-mode label in **{R[2]['flip']:.1%}** of batches, and every one of the 100 digits changes label in at least one. In batches of 64, the size it was trained with, the label changes in {R[64]['flip']:.1%} of batches.""",
        figures=[figs["f0"]])

    post.add(
        "The companions are an input",
        f"""The dependence falls with batch size because the batch's mean and variance settle towards the population's as the batch grows. In a batch of 16 the label still changes in {R[16]['flip']:.1%} of batches, and {R[16]['any_flip']:.0%} of the digits change label in at least one of their 200.

It does not fall because the batch is large. It falls because the batch is a *random sample* of the population. The last line of the output keeps the batch at 16 but draws the 15 companions from a single class, as happens when data arrive sorted by label, by date or by source. Now the digit's label differs from its inference-mode label in **{1 - d['skewed_agree']:.1%}** of batches, {(1 - d['skewed_agree']) / R[16]['flip']:.0f} times as often as with random companions. The batch statistics describe the companions, the companions are all 3s or all 7s, and the network normalises the digit against them.

Batches of 2 show the limiting case. For two examples, each unit's batch mean is the midpoint of the two values and its standard deviation is half their distance, so after normalisation, and before the learned scale and shift, every unit is +1 for one example and −1 for the other, whatever the two values were (unless they were almost equal). The layer passes on only which of the two examples was larger on each unit. A network that has to classify from that is classifying a pair, not a digit.""")

    post.add(
        "Training in small batches",
        f"""That last observation matters most during training, because training is always in training mode. A network trained in batches of 2 learns to classify from the ±1 patterns of pairs. At inference, the layer uses running averages instead, which no example ever saw during training, and the network is asked to compute a function it was never trained to compute.

The second experiment trains the same architecture with and without batch normalisation at each batch size, for the same number of epochs with Adam, and scores it on the full test half:

{snip['train'].markdown()}

Without normalisation, accuracy is about the same at every batch size, {min(r['plain'] for r in res['b']):.3f} to {max(r['plain'] for r in res['b']):.3f}. With it, the network trained in batches of 2 scores **{t[2]['bn']:.3f}** in inference mode and {t[2]['bn_train_mode']:.3f} when evaluated in training mode on test batches of 2: normalisation cost it {t[2]['plain'] - t[2]['bn']:.3f}. In batches of 4 the cost is {t[4]['plain'] - t[4]['bn']:.3f}. From batches of 16 up, normalisation helps, by {t[16]['bn'] - t[16]['plain']:.3f} and {t[64]['bn'] - t[64]['plain']:.3f}, which is the improvement it is known for. In batches of 64 the two modes agree to within a point, {t[64]['bn']:.3f} and {t[64]['bn_train_mode']:.3f}, which is the condition under which the running averages are a faithful stand-in for the batch.""",
        figures=[figs["f1"], figs["f2"]])

    post.add(
        "What the alternatives fix",
        """Ioffe's batch renormalisation corrects the batch statistics towards the running averages during training, so the trained function is closer to the one used at inference; it was designed for small and non-independent batches. Layer normalisation and group normalisation compute the statistics over the units of one example instead of over the examples of one batch, so the output never depends on the companions and the two modes are the same function. Transformers use layer normalisation, and are free of the problem.

The practical failures of batch normalisation follow from the measurement directly. A model left in training mode at serving time gives answers that depend on what else is being served with them. A batch assembled from one class, one customer or one day normalises every member against an unrepresentative sample. Gradient accumulation over many tiny batches, used to fit a large model in memory, still normalises over the tiny batch. And a fine-tuning run in small batches can shift the running statistics away from the population the network was trained on.""")

    post.add(
        "Where this breaks",
        """**Fully connected layers on small images.** In convolutional networks the statistics are pooled over all spatial positions as well as the batch, so a batch of 2 images contributes many values per channel and the dependence is weaker than here.

**Frozen running statistics in the first study.** The momentum was set to zero so that the evaluation did not change the network; in a real training-mode deployment the running averages would also drift with every batch.

**One architecture, one data set.** The batch size at which normalisation turns from hurting to helping, between 4 and 16 here, depends on the network and on how much the batch statistics vary, and is not a general constant.

**Adam at one learning rate.** The batch-size study holds the learning rate at 1e-3 for every batch size; episode 22 showed that the best rate moves with the batch, so the plain network's flat line is partly a matter of where 1e-3 happens to sit.""")

    post.add(
        "What to keep",
        f"""1. Batch normalisation normalises with the batch's statistics. In training mode one digit's label changed with its random companions in {R[2]['flip']:.0%} of batches of 2 and {R[64]['flip']:.1%} of batches of 64.

2. The companions must be a random sample. With 15 companions from one class, the label changed in {1 - d['skewed_agree']:.0%} of batches of 16, against {R[16]['flip']:.1%} with random ones.

3. In a batch of 2 every normalised unit is ±1 before the learned scale and shift: the layer keeps only which example was larger. Trained that way, the network scored {t[2]['bn']:.3f} in inference mode, against {t[2]['plain']:.3f} without normalisation.

4. From batches of 16 up, normalisation helped by about half a point, and the two modes agreed. Check both conditions before trusting it: batches large enough and mixed enough that their statistics are the population's.""")

    post.add(
        "Exercise",
        """Take a trained network with batch normalisation and pick 50 test examples. Predict each one three ways: in evaluation mode alone; in training mode inside random batches of your usual size; and in training mode inside batches whose other members all share one label or one source.

Count how often the label changes between the first and the other two. If the third count is much larger than the second, any process that feeds your model sorted batches, at training or at serving, is changing its answers.""")

    post.add(
        "Next",
        """Batch normalisation is one of two common layers that behave differently in training and at inference. Episode 25 measures the other, dropout, and the claim that switching it off and scaling the weights gives the same predictions as averaging over the networks dropout trains.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
