"""ML 26: The Network That Memorises Noise Learns the Signal First.

The twenty-sixth episode of *Machine Learning, Taught Through What Breaks*,
opening Arc VI. "A network that generalises could not have fit noise."
Measured with one network on the digits, trained full-batch on true labels
and on labels partly or wholly replaced by random ones:

* It reaches 100% training accuracy on true labels in 41 steps and on
  entirely random labels in 73, where test accuracy is chance. Its capacity
  cannot be what makes it generalise on the true labels.
* With 20% or 50% of labels randomised it still reaches 100%, but not at
  once: it fits the unchanged rows first, while its test accuracy peaks at
  0.928 and 0.895, and memorises the changed ones afterwards, ending at 0.836
  and 0.604.

Run: `standarderror run ml126_random_labels --publish`
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
    runs = nets.random_labels()
    by = {r["fraction"]: r for r in runs}
    for r in runs:
        step, _, best, *_ = max(r["curve"], key=lambda c: c[2])
        r["peak_step"], r["peak_test"] = step, best
    return {"runs": runs, "by": by}


def _label(f):
    return "true labels" if f == 0 else ("random labels" if f == 1 else
                                         f"{f:.0%} randomised")


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    runs, by = res["runs"], res["by"]

    def train(ax, m):
        for i, r in enumerate(runs):
            c = np.array(r["curve"])
            ax.plot(c[:, 0], c[:, 1], color=m.series[i], lw=1.6,
                    label=_label(r["fraction"]))
        ax.set_xlim(0, 160)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    out["f0"] = charts.diagram(
        train,
        title="It fits whatever it is given",
        subtitle=("Training accuracy, against the labels it is trained on, "
                  "of a two-layer network with 512 units per layer on 898 "
                  "digits. One full-batch Adam step per point on the x axis."),
        xlabel="training step",
        ylabel="training accuracy",
        figsize=(7.2, 4.5),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("Four rising curves, all reaching 1.0 before step 110: true "
             "labels first, then random labels, then 50% and 20% randomised."),
        caption=(f"Random labels take {by[1.0]['epoch_100']} steps to fit "
                 f"perfectly, against {by[0.0]['epoch_100']} for the true "
                 f"ones. Partly random labels take longest."),
        path=str(IMG / f"ml126-f0-train.{EXT}"))[0]

    def test(ax, m):
        for i, r in enumerate(runs):
            c = np.array(r["curve"])
            ax.plot(c[:, 0], c[:, 2], color=m.series[i], lw=1.6,
                    label=_label(r["fraction"]))
            if 0 < r["fraction"] < 1:
                ax.plot([r["peak_step"]], [r["peak_test"]], "o",
                        color=m.series[i], ms=5)
        ax.axhline(0.1, color=m.grid, lw=1, ls=":")
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=8.5, loc="center right")

    r2, r5 = by[0.2], by[0.5]
    out["f1"] = charts.diagram(
        test,
        title="Test accuracy rises, then pays for the noise",
        subtitle=("Test accuracy on 899 held-out digits with their true "
                  "labels, for the same four runs. Dots mark the best "
                  "checkpoint of the partly random runs."),
        xlabel="training step",
        ylabel="test accuracy",
        figsize=(7.2, 4.5),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("The true-label curve rises to 0.97 and stays; the partly "
             "random curves rise to about 0.9 within 20 steps and then fall; "
             "the random-label curve stays near 0.1."),
        caption=(f"With 20% of labels randomised the network peaks at "
                 f"{r2['peak_test']:.3f} at step {r2['peak_step']} and ends "
                 f"at {r2['test_acc']:.3f}; with 50%, {r5['peak_test']:.3f} "
                 f"and {r5['test_acc']:.3f}."),
        path=str(IMG / f"ml126-f1-test.{EXT}"))[0]

    def split(ax, m):
        c = np.array(r2["curve"])
        ax.plot(c[:, 0], c[:, 3], color=m.series[0], lw=1.6,
                label="unchanged rows, fitted")
        ax.plot(c[:, 0], c[:, 4], color=m.series[1], lw=1.6,
                label="changed rows, fitted to the random label")
        ax.plot(c[:, 0], c[:, 2], color=m.ink, lw=1.2, ls="--",
                label="test accuracy")
        ax.set_xlim(0, 160)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8.5, loc="center right")

    out["f2"] = charts.diagram(
        split,
        title="Signal first, noise second",
        subtitle=(f"The 20% run, split into the {1 - r2['changed']:.1%} of "
                  f"training rows whose label survived and the "
                  f"{r2['changed']:.1%} whose label was changed."),
        xlabel="training step",
        ylabel="accuracy",
        figsize=(7.2, 4.5),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("The unchanged-rows curve climbs to nearly 1 by step 30 while "
             "the changed-rows curve stays near 0; then the changed-rows "
             "curve climbs to 1 and the dashed test curve falls."),
        caption=("The network fits the rows it can generalise from before "
                 "the rows it can only memorise. Test accuracy is highest in "
                 "between."),
        path=str(IMG / f"ml126-f2-split.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    by = res["by"]

    def fit(panel, m):
        c = np.array(by[1.0]["curve"])
        panel.plot(c[:40, 0], c[:40, 1], color=m.series[3], lw=2)

    def test(panel, m):
        c = np.array(by[0.5]["curve"])
        panel.plot(c[:40, 0], c[:40, 2], color=m.series[2], lw=2)

    def split(panel, m):
        c = np.array(by[0.2]["curve"])
        panel.plot(c[:32, 0], c[:32, 3], color=m.series[0], lw=2)
        panel.plot(c[:32, 0], c[:32, 4], color=m.series[1], lw=2)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=26,
        headline="It memorises noise, later",
        panels=[(fit, f"{by[1.0]['epoch_100']} steps",
                 "to fit random labels"),
                (test, f"{by[0.5]['peak_test']:.3f}",
                 "best test, half random"),
                (split, f"{by[0.2]['test_acc']:.3f}",
                 "final test, 20% random")],
        note=("The network that learns the digits also memorises random "
              "labels perfectly. What separates the two is the order: given "
              "both, it fits the signal first."),
        alt=("Three hand-drawn frames: a curve climbing to the top; a curve "
             "rising and then falling; two curves, one rising early and one "
             "late."),
        path=str(IMG / f"ml126-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["fit"] = s.run("""
        from standarderror.ml import nets

        runs = nets.random_labels()     # 2 x 512 network, full-batch Adam
        for r in runs:
            print(f"{r['fraction']:4.0%} of labels randomised: 100% training "
                  f"accuracy at step {r['epoch_100']:3d}, test {r['test_acc']:.3f}")
        """)
    out["peak"] = s.run("""
        for r in runs:
            step, _, best, *_ = max(r["curve"], key=lambda c: c[2])
            print(f"{r['fraction']:4.0%} randomised: best test {best:.3f} at "
                  f"step {step:3d}, final {r['test_acc']:.3f}")
        """)
    out["split"] = s.run("""
        fifth = runs[1]                  # 20% of labels randomised
        for step, train, test, kept, changed in fifth["curve"]:
            if step in (5, 20, 50, 75, 100):
                print(f"step {step:3d}  unchanged rows {kept:.3f}  "
                      f"changed rows {changed:.3f}  test {test:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    by = res["by"]
    r0, r2, r5, r1 = by[0.0], by[0.2], by[0.5], by[1.0]
    c2 = {c[0]: c for c in r2["curve"]}
    c5 = {c[0]: c for c in r5["curve"]}

    # The spine, asserted rather than trusted.
    assert all(r["train_acc"] == 1.0 for r in res["runs"])
    assert r0["epoch_100"] < r1["epoch_100"] < r5["epoch_100"] < \
        r2["epoch_100"]
    assert r1["epoch_100"] < 2 * r0["epoch_100"]
    assert r0["test_acc"] > 0.96 and r1["test_acc"] < 0.15
    assert r2["peak_step"] <= 30 and r5["peak_step"] <= 30
    assert r2["peak_test"] - r2["test_acc"] > 0.08
    assert r5["peak_test"] - r5["test_acc"] > 0.25
    assert r5["peak_test"] > 0.88
    assert c2[20][3] > 0.95 and c2[20][4] < 0.05
    assert c2[100][4] > 0.99
    assert c5[15][3] > 0.95 and c5[15][4] < 0.1
    assert abs(r5["clean_fit"] - (1 - r5["changed"])) < 0.01
    assert abs(r2["clean_fit"] - (1 - r2["changed"])) < 0.01

    post = Post(
        title=(f"{SERIES_TAG} 26: The Network That Memorises Noise Learns "
               f"the Signal First"),
        slug="ml-26-random-labels",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=26,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A network that reaches 97% on test digits also fits "
                  "random labels perfectly. Capacity cannot explain the "
                  "first; the order in which it learns does."),
        summary=(
            f"A two-layer network with 512 units per layer, trained "
            f"full-batch on 898 digits, reached 100% training accuracy in "
            f"{r0['epoch_100']} steps on the true labels (test accuracy "
            f"{r0['test_acc']:.3f}) and in {r1['epoch_100']} on entirely "
            f"random labels (test {r1['test_acc']:.3f}). With 20% of labels "
            f"randomised it fitted the unchanged rows first: by step 20 it "
            f"was right on {c2[20][3]:.1%} of them and on {c2[20][4]:.1%} of "
            f"the changed ones, and test accuracy peaked at "
            f"{r2['peak_test']:.3f}. It then memorised the changed rows and "
            f"ended at {r2['test_acc']:.3f}. With half the labels random, "
            f"test accuracy peaked at {r5['peak_test']:.3f} and ended at "
            f"{r5['test_acc']:.3f}."),
        tags=["machine-learning", "deep-learning", "neural-networks",
              "generalization", "memorization", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Optical Recognition of Handwritten Digits (UCI), as bundled with "
            "scikit-learn, CC BY 4.0; one stratified half split (898 "
            "training digits, 899 test), standardised on the training half.",
            "Machinery: `standarderror/ml/nets.py`, tested in "
            "`tests/test_ml.py`; a PyTorch network 64-512-512-10 with ReLU, "
            "trained with full-batch Adam at learning rate 1e-3 for 400 "
            "steps, on labels with 0%, 20%, 50% or 100% of rows given a "
            "uniformly random class.",
            "Where this stops: Zhang, Bengio, Hardt, Recht and Vinyals, "
            "\"Understanding deep learning requires rethinking "
            "generalization\", *ICLR* (2017); Arpit et al., \"A closer look "
            "at memorization in deep networks\", *ICML* (2017).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("split, label noise and network seed 0; "
                            "full-batch training has no data order; "
                            "torch.set_num_threads(1)"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A network that could fit anything",
        f"""Arc V was about training a network. Arc VI is about what the trained network has learned, and it starts with an argument every statistics student meets in some form. A model that is flexible enough to fit any labelling of its training data has learned nothing from fitting this one; a model that generalises must therefore be constrained, by its size, its architecture or its regularisation, so that it *could not* have fit noise. The sentence under test is that argument's conclusion: *a network that generalises could not have fit noise*.

Zhang, Bengio and colleagues tested it on image networks in 2017 by replacing the labels with random ones. The test is cheap to repeat on the digits. Take a network with two hidden layers of 512 units, which has about 300,000 parameters for 898 training images, and train it with full-batch Adam on the true labels, on labels where 20% or 50% of rows were given a random class, and on labels that are entirely random:

{snip['fit'].markdown()}

On the true labels it reaches 100% training accuracy in **{r0['epoch_100']} steps** and scores {r0['test_acc']:.3f} on held-out digits. On entirely random labels, where there is nothing to learn, it also reaches 100%, in **{r1['epoch_100']} steps**, and scores {r1['test_acc']:.3f}, which is chance for ten classes. Fitting noise took it less than twice as long as fitting the truth.

So the same network, unchanged, can memorise an arbitrary labelling of these 898 images. By the argument above, its fit to the true labels should mean nothing. It means a test accuracy of {r0['test_acc']:.3f}.""",
        figures=[figs["f0"]])

    post.add(
        "What capacity cannot explain",
        f"""The classical bounds on generalisation say that the gap between training and test error is small when the model class is too small to fit arbitrary labels. Zhang and colleagues' point, which this measurement repeats, is that for a network of ordinary size the premise is false. The model class can fit arbitrary labels, and fitting them takes no special effort: no change of architecture, learning rate or number of steps. Any explanation of why the network generalises on true labels has to be something that differs between the two runs, and the only thing that differs is the labels.

That is a statement about the *data*. On true labels the images of each class share structure, so a function that fits some of them fits many of the rest, and every image's contribution to training pushes in a direction that helps the others of its class. On random labels no such function exists, and the network has to carve out each image's label individually, which is why it took {r1['epoch_100']} steps instead of {r0['epoch_100']}. Both end at 100%. The difference is not whether the network can fit them but how.

Training accuracy cannot tell the two apart. Both runs finish with a perfect score on what they were trained on, and only held-out data distinguishes a network that has learned the digits from one that has learned a list.""")

    post.add(
        "The order of learning",
        f"""The partly random runs show the "how". With 20% or 50% of the labels randomised the network still reaches 100% training accuracy, in {r2['epoch_100']} and {r5['epoch_100']} steps, longer than either pure case. But its test accuracy does not move steadily. It rises fast, then falls:

{snip['peak'].markdown()}

With a fifth of the labels randomised the network reaches **{r2['peak_test']:.3f}** on held-out digits at step {r2['peak_step']}, nearly as good as the network trained on clean labels, and ends at {r2['test_acc']:.3f}. With half randomised it reaches {r5['peak_test']:.3f} and ends at {r5['test_acc']:.3f}. Splitting the training rows by whether their label was changed shows why:

{snip['split'].markdown()}

By step 20 the network is right on {c2[20][3]:.1%} of the unchanged rows and on {c2[20][4]:.1%} of the changed ones, *fewer than chance*: it is predicting their true class, not the random label it is being trained on. It has learned the digits from the 80% of the data that teach them, and it is overruling the noise. Only after that does it start to memorise the changed rows, one by one, and every one it memorises costs a little test accuracy, because a memorised random label is a small wrong rule about what that digit looks like. By step 100 it fits {c2[100][4]:.1%} of the changed rows.""",
        figures=[figs["f1"], figs["f2"]])

    post.add(
        "What this means for early stopping",
        f"""Arpit and colleagues documented the same ordering on larger networks: real patterns are fitted before noise, and the switch can be seen in the training curves. It is the practical reason early stopping works. A network stopped at the peak is the one that has learned the shared structure and not yet the exceptions.

It also says what early stopping needs. The peak was at step {r2['peak_step']} for the 20% run and step {r5['peak_step']} for the 50% run, and the drop after it was {r2['peak_test'] - r2['test_acc']:.3f} and {r5['peak_test'] - r5['test_acc']:.3f}. Training accuracy gave no sign of either: it was still climbing steadily through the peak. Only a held-out set could see it. With noisy labels, which most real data sets have in some proportion, a validation set is not a formality for reporting; it is the only instrument that can tell when the network has stopped learning the problem and started learning the noise.

The ordering also explains why a network trained on clean labels does not suffer from its capacity. With no noise to memorise, the late phase of training has nothing harmful to fit; the test accuracy of the clean run reached {r0['test_acc']:.3f} and stayed there through 400 steps.""")

    post.add(
        "Where this breaks",
        """**One small network, one data set.** The ordering of signal before noise is a tendency of gradient descent on data with shared structure, not a theorem. It depends on the learning rate, the optimiser and the architecture, and can be weakened, for instance by very large learning rates or by training on each example many times in a row.

**Uniform random labels.** Real label noise is rarely uniform. Annotators confuse a 4 with a 9 more often than with a 0, and structured noise of that kind can look like signal to the network and be fitted early.

**No regularisation.** No weight decay, dropout or augmentation was used. Each would slow the memorisation of noise and move or flatten the peak; none would stop a network this size from fitting random labels if trained long enough.

**Full-batch training.** One step per pass over the data keeps the comparison clean. With minibatches the same phases appear, measured in epochs rather than steps, with more noise.""")

    post.add(
        "What to keep",
        f"""1. A network that generalises can also fit noise. The same network reached 100% training accuracy on true labels in {r0['epoch_100']} steps (test {r0['test_acc']:.3f}) and on random labels in {r1['epoch_100']} (test {r1['test_acc']:.3f}).

2. So capacity does not explain generalisation here. The labels do: on real data, what fits one example of a class fits the rest.

3. Given both signal and noise, the network fitted the signal first. With 20% of labels randomised, by step 20 it was right on {c2[20][3]:.1%} of unchanged rows and {c2[20][4]:.1%} of changed ones, and test accuracy peaked at {r2['peak_test']:.3f}.

4. Memorising the noise afterwards cost {r2['peak_test'] - r2['test_acc']:.3f} of test accuracy with 20% noise and {r5['peak_test'] - r5['test_acc']:.3f} with 50%, invisible in training accuracy and visible only on held-out data.""")

    post.add(
        "Exercise",
        """Take a classifier you trained and shuffle its training labels completely. Train it again with exactly the same settings and record how long it takes to reach the training accuracy the real run reached.

If it gets there, your model is large enough to memorise your data, and its test score is the only evidence that it did not. Then flip 10% of the real labels at random, train once more, and plot validation accuracy against epoch: the gap between the peak and the end is what early stopping is buying you.""")

    post.add(
        "Next",
        """Here the network fitted the signal first and the noise later, and later was worse. Episode 27 runs the clock the other way: a network that memorises its training set first, reaches 100% training accuracy and chance on the test set, and is kept training long after there seems nothing left to learn.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
