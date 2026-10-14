"""ML 28: Augmentation Adds an Assumption, Not Data.

The twenty-eighth episode of *Machine Learning, Taught Through What Breaks*.
"Data augmentation adds data." Measured on the digits, 150 real training
images with four augmented copies each, against 750 real images, over ten
splits:

* Four times more real digits added 4.1 points of test accuracy. Shifting
  the images by up to a pixel cost 2.7 points on the test set and added 32 on
  the same test images shifted by up to a pixel, where the extra real digits
  added 1.6.
* Pixel noise changed nothing. Mirror images and half-turns, which turn some
  digits into others or into nothing, cost 9.2 and 12.3.
* An augmentation is a claim that the label survives a transformation. It
  helps where the test data vary in that way, and hurts where the claim is
  false; on a fixed budget even a true claim cost a little on clean data.

Run: `standarderror run ml128_augmentation --publish`
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

KINDS = ("real", "shift", "noise", "flip", "rot180")
NAMES = {"base": "150 real digits", "real": "750 real digits",
         "shift": "+ shifted copies", "noise": "+ noisy copies",
         "flip": "+ mirrored copies", "rot180": "+ half-turned copies"}


def compute() -> dict:
    return {"a": nets.augmentation(), "f": nets.flip_confusions(),
            "c": nets.centring()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    a = res["a"]

    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    picks = [int(np.where(y == d)[0][0]) for d in (0, 2, 3, 6, 7)]
    rng = np.random.default_rng(3)

    def grid(ax, m):
        cols = ("original", "shift", "noise", "flip", "rot180")
        for i, j in enumerate(picks):
            for c, kind in enumerate(cols):
                img = X[[j]] if kind == "original" else \
                    nets.augment(X[[j]], kind, rng)
                ax.imshow(img.reshape(8, 8), cmap="Greys", vmin=0, vmax=16,
                          extent=(c * 9, c * 9 + 8, -i * 9 - 8, -i * 9))
        ax.set_xlim(-1, 45)
        ax.set_ylim(-45, 1)
        ax.set_xticks([c * 9 + 4 for c in range(5)])
        ax.set_xticklabels(["original", "shifted", "noisy", "mirrored",
                            "half-turned"], fontsize=8.5)
        ax.set_yticks([])

    out["f0"] = charts.diagram(
        grid,
        title="Four ways to make more digits",
        subtitle=("A 0, 2, 3, 6 and 7 from the digits data and one copy under "
                  "each transformation: a shift of up to a pixel, Gaussian "
                  "pixel noise (sd 2 of 16), a left-right mirror, a half-turn."),
        figsize=(6.8, 6.4),
        ticks=True,
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("A five-by-five grid of 8 by 8 pixel digits: the shifted and "
             "noisy copies still read as the same digit; the mirrored 3 and "
             "the half-turned 6 and 7 read as something else or nothing."),
        caption=("The shifted and noisy copies keep the label. The mirrored 3 "
                 "is no longer a 3, the half-turned 6 is a 9, and the "
                 "mirrored 0 is still a 0."),
        path=str(IMG / f"ml128-f0-grid.{EXT}"))[0]

    def bars(ax, m):
        x = np.arange(len(KINDS))
        w = 0.38
        g1 = [a["gain"][k] for k in KINDS]
        e1 = [2 * a["gain_se"][k] for k in KINDS]
        g2 = [a["moved"]["gain"][k] for k in KINDS]
        e2 = [2 * a["moved"]["gain_se"][k] for k in KINDS]
        ax.bar(x - w / 2, g1, w, yerr=e1, color=m.series[0], capsize=3,
               label="on the test digits")
        ax.bar(x + w / 2, g2, w, yerr=e2, color=m.series[1], capsize=3,
               label="on the test digits moved by up to a pixel")
        ax.axhline(0, color=m.ink, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([NAMES[k].replace(" copies", "\ncopies")
                            .replace(" digits", "\ndigits") for k in KINDS],
                           fontsize=8.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    out["f1"] = charts.diagram(
        bars,
        title="Real data and shifted copies teach different things",
        subtitle=("Change in test accuracy against training on 150 real "
                  "digits, average of 10 splits, with two standard errors. "
                  "Every model takes the same number of gradient steps."),
        ylabel="change in test accuracy",
        figsize=(7.8, 4.6),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("Paired bars: real data helps on clean test digits and barely on "
             "moved ones; shifted copies cost a little on clean digits and "
             "add about 30 points on moved ones; mirrored and half-turned copies "
             "lose on both."),
        caption=(f"Shifted copies: {a['gain']['shift']:+.3f} on the test "
                 f"digits, **{a['moved']['gain']['shift']:+.3f}** on the "
                 f"moved ones. Four times more real digits: "
                 f"{a['gain']['real']:+.3f} and "
                 f"{a['moved']['gain']['real']:+.3f}."),
        path=str(IMG / f"ml128-f1-bars.{EXT}"))[0]

    rows = []
    for k in ("base", *KINDS):
        g = "" if k == "base" else f"{a['gain'][k]:+.3f}"
        gm = "" if k == "base" else f"{a['moved']['gain'][k]:+.3f}"
        rows.append([NAMES[k], f"{a[k]:.3f}", g, f"{a['moved'][k]:.3f}", gm])
    out["f2"] = charts.table_image(
        rows,
        header=["training set", "test accuracy", "change",
                "accuracy, moved test", "change"],
        title="What each kind of extra data bought",
        subtitle=("Averages over 10 random splits: 150 training digits, four "
                  "augmented copies of each or 600 more real ones, 700 test "
                  "digits."),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("A six-row table of accuracies on clean and moved test digits "
             "for each training set."),
        caption=("Only real data helped on the clean test digits by more than "
                 "two points. Only shifted copies helped on the moved ones by "
                 "more than ten."),
        bold_cells={(1, 2), (2, 4)},
        align="lrrrr",
        path=str(IMG / f"ml128-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    a = res["a"]

    def real(panel, m):
        panel.bar([0, 1], [a["gain"]["real"], a["moved"]["gain"]["real"]],
                  color=m.series[0])
        panel.set_ylim(-0.12, 0.45)

    def shift(panel, m):
        panel.bar([0, 1], [a["gain"]["shift"], a["moved"]["gain"]["shift"]],
                  color=m.series[1])
        panel.set_ylim(-0.12, 0.45)

    def flip(panel, m):
        panel.bar([0, 1], [a["gain"]["flip"], a["gain"]["rot180"]],
                  color=m.series[3])
        panel.set_ylim(-0.12, 0.45)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=28,
        headline="Augmentation is an assumption",
        panels=[(real, f"{a['gain']['real']:+.3f}", "4x more real digits"),
                (shift, f"{a['moved']['gain']['shift']:+.3f}",
                 "shifted, on moved test"),
                (flip, f"{a['gain']['rot180']:+.3f}", "half-turned copies")],
        note=("More real digits raised accuracy; shifted copies cost a "
              "little on the test set and bought a lot on shifted digits; "
              "half-turned copies, which turn a 6 into a 9, made the model "
              "worse."),
        alt=("Three hand-drawn frames of paired bars: one tall and one short; "
             "one flat and one very tall; two below zero."),
        path=str(IMG / f"ml128-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["aug"] = s.run("""
        from standarderror.ml import nets

        r = nets.augmentation()      # 150 real digits, 4 copies each, 10 splits
        for k in ("base", "real", "shift", "noise", "flip", "rot180"):
            print(f"{k:7s} test {r[k]:.3f}   moved test {r['moved'][k]:.3f}")
        """)
    out["flip"] = s.run("""
        f = nets.flip_confusions()
        for kind in ("flip", "rot180"):
            print(f"{kind:6s} " + "  ".join(f"{d}:{v:4.0%}"
                                            for d, v in f[kind].items()))
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    a, f, c = res["a"], res["f"], res["c"]
    g, gm = a["gain"], a["moved"]["gain"]
    se_, sem = a["gain_se"], a["moved"]["gain_se"]
    flip_all = [d for d, v in f["flip"].items() if v > 0.95]
    rot_all = [d for d, v in f["rot180"].items() if v > 0.95]
    self_max = max([1 - f["flip"][d] for d in flip_all] +
                   [1 - f["rot180"][d] for d in rot_all])
    assert 0 < self_max < 0.05      # "almost never", not "never"

    # The spine, asserted rather than trusted.
    assert g["real"] > 0.03 and g["real"] > 5 * se_["real"]
    assert -0.04 < g["shift"] < -5 * se_["shift"]
    assert abs(g["noise"]) < 2 * se_["noise"] + 0.002
    assert abs(gm["noise"]) < 0.01
    assert g["flip"] < -0.05 and g["rot180"] < -0.05
    assert gm["shift"] > 0.25 and 2 * sem["real"] < gm["real"] < 0.03
    assert gm["shift"] > 10 * gm["real"]
    assert a["moved"]["base"] < 0.4
    assert gm["flip"] < 0 and gm["rot180"] < 0
    assert sorted(flip_all) == [3, 5, 6]
    assert sorted(rot_all) == [3, 6, 7, 9]
    assert f["flip"][0] < 0.05 and f["rot180"][0] < 0.05
    assert c["off_by_a_pixel"] < 0.01 and c["left_col_ink"] < 0.005
    assert max(c["sd_x"], c["sd_y"]) < 0.4

    post = Post(
        title=f"{SERIES_TAG} 28: Augmentation Adds an Assumption, Not Data",
        slug="ml-28-augmentation",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=28,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Shifted copies of 150 digits made a network worse on the "
                  "test set and far better on shifted digits. Six hundred more "
                  "real digits did the reverse."),
        summary=(
            f"Trained on 150 digits, a network scored {a['base']:.3f}. Four "
            f"times as many real digits raised that by {g['real']:.3f}; four "
            f"shifted copies of each image changed it by {g['shift']:+.3f}, "
            f"pixel noise by {g['noise']:+.3f}, and mirrored or half-turned "
            f"copies "
            f"lowered it by {-g['flip']:.3f} and {-g['rot180']:.3f}. On the "
            f"same test digits moved by up to a pixel, the base model scored "
            f"{a['moved']['base']:.3f}; real data added {gm['real']:+.3f} and "
            f"shifted copies {gm['shift']:+.3f}. An augmentation asserts that "
            f"the label survives a transformation. It pays where the test "
            f"data vary that way and costs where the assertion is false."),
        tags=["machine-learning", "data-science", "neural-networks",
              "data-augmentation", "invariance", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Optical Recognition of Handwritten Digits (UCI), as bundled with "
            "scikit-learn, CC BY 4.0: 1,797 8 x 8 images.",
            "Machinery: `standarderror/ml/nets.py`, tested in "
            "`tests/test_ml.py`; a PyTorch network with one hidden layer of "
            "128, SGD with momentum 0.9 and learning rate 0.01, exactly 300 "
            "minibatches of 32 for every training set.",
            "Where this stops: Simard, Steinkraus and Platt, \"Best practices "
            "for convolutional neural networks applied to visual document "
            "analysis\", *ICDAR* (2003); Chen, Dobriban and Lee, \"A "
            "group-theoretic framework for data augmentation\", *JMLR* "
            "(2020).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("splits, augmentations and network seeds 0 to 9; "
                            "moved test images from seeds 10,000 to 10,009; "
                            "torch.set_num_threads(1)"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Free data",
        f"""Episode 27's weight decay encoded a preference the training data could not express, for small weights, and that preference decided what the network learned. Data augmentation is the most common way of encoding a preference about the data themselves: take each training image, shift it, rotate it, add noise, and train on the copies as well. The usual description is the sentence under test: *data augmentation adds data*. Four copies of every image, five times the training set, for free.

If that were literally true, augmented copies would do what the same number of new, real examples does. That can be tested directly on the digits, because the data set holds more real images than a small experiment needs. Take 150 training digits. Give one network those alone, one network 750 real digits, and four networks the 150 plus four augmented copies of each, made in four different ways. Every network takes the same number of gradient steps, so none wins by training longer. Score them all on 700 held-out digits, and on the same 700 moved by up to a pixel in each direction. Repeat over ten random splits.

{snip['aug'].markdown()}

Four times more real digits raised test accuracy by **{g['real']:+.3f}**. Four shifted copies of each image changed it by **{g['shift']:+.3f}**: they made the network worse, by more than five standard errors ({se_['shift']:.3f}). Noisy copies changed it by {g['noise']:+.3f}, which is nothing. Mirrored and half-turned copies made the network worse, by {-g['flip']:.3f} and {-g['rot180']:.3f}.""",
        figures=[figs["f0"]])

    post.add(
        "What the shifted copies were for",
        f"""The second column of the output changes the picture. On test digits moved by up to a pixel, the network trained on 150 digits scores **{a['moved']['base']:.3f}**. It has hardly seen a digit off centre, because the digits in this data set barely move within their frame: the centre of each image's ink has a standard deviation of {c['sd_x']:.2f} of a pixel across and {c['sd_y']:.2f} down, and only {c['off_by_a_pixel']:.1%} of images have it more than a pixel from the average. The leftmost column holds ink in {c['left_col_ink']:.1%} of images. A one-pixel move puts strokes on pixels the network has learned to ignore. More real digits do not fix this: 750 centred digits score {a['moved']['real']:.3f} on the moved test set, a change of {gm['real']:+.3f}. Shifted copies of 150 digits score {a['moved']['shift']:.3f}, a gain of **{gm['shift']:+.3f}**.

So the two kinds of extra data taught different things. The real digits taught more about what digits look like: more writers, more ways of drawing a 4. The shifted copies taught nothing about that. They taught one fact, that a digit moved by a pixel is the same digit, and the clean test set never asked about that fact, because nothing in it was moved. It charged for the lesson anyway: {-g['shift']:.3f} of accuracy on centred digits. A likely reason is budget. Every network took the same 300 steps, and for this one four-fifths of what it saw was shifted, so it spent most of its training on positions the test set never uses.

That is what an augmentation is. It is not data in the sense of new observations of the world. It is a statement, made by the person training the model, that the label is unchanged by some transformation. The network cannot learn that statement from centred digits, because centred digits contain no evidence about moved ones either way. The augmentation supplies it from outside the data, as an assumption, and the assumption is worth exactly as much as the test data's variation along it.""",
        figures=[figs["f1"]])

    post.add(
        "When the assumption is false",
        f"""Mirrors and half-turns are ordinary augmentations for photographs: a mirrored cat is a cat. They are not label-preserving for digits. A network trained on clean digits, shown transformed test digits, gives an answer other than the true class at these rates, class by class:

{snip['flip'].markdown()}

A mirrored {', '.join(str(d) for d in flip_all[:-1])} or {flip_all[-1]} is almost never read as itself, and neither is a half-turned {', '.join(str(d) for d in rot_all[:-1])} or {rot_all[-1]}: each is read as itself in at most {self_max:.0%} of cases. To a human eye a half-turned 6 is a 9. Zeros survive both, and eights mostly do. Training on these copies under their original labels tells the network that the shape of a 9 is evidence for a 6, which is false, and it pays for the falsehood: {g['flip']:+.3f} from mirrors and {g['rot180']:+.3f} from half-turns, against the network that saw only the 150 originals. On the moved test digits they cost too, {gm['flip']:+.3f} and {gm['rot180']:+.3f}.

Pixel noise is the control. Its copies preserve the label, and it changed nothing: {g['noise']:+.3f} on the clean test set and {gm['noise']:+.3f} on the moved one. Neither test set contains noisy pixels, so the invariance it teaches is never asked for, and at a standard deviation of 2 on a 0-to-16 scale it does not disturb the image enough to get in the way. Copies that assert something true and irrelevant are simply wasted.""",
        figures=[figs["f2"]])

    post.add(
        "Reading augmentation as a prior",
        f"""Chen, Dobriban and Lee make this precise. When the transformations form a group under which the true label is invariant, training on augmented data is equivalent to averaging the loss over the group's orbit of each example. That can only reduce the variance of what is learned, and it introduces no bias. When the label is *not* invariant, the same averaging introduces bias, because it trains the model on a distribution that does not exist.

Both halves were visible here, along with a cost the theory does not count. Shifts are close to an invariance of handwriting, and the shifted copies bought {gm['shift']:+.3f} where shifts actually occurred. On a fixed budget of training they still cost {-g['shift']:.3f} where shifts did not: a true assumption may be free in the limit, but on 300 steps it was not, plausibly because of the budget (equal epochs were not tried). Mirrors and half-turns break the label for at least three classes, and the cost was large wherever they were scored.

The practical reading is that an augmentation should be chosen the way a prior is: by asking what is true of the world the model will see, not what makes the training set bigger. A shift augmentation for a camera that may not be centred; a colour jitter for a model that will meet different lighting; no mirror for text, no half-turn for digits, no time reversal for a sequence with a direction. The question is never how many copies, but whether the test data will vary in the way the copies do and whether the label survives it.""")

    post.add(
        "Where this breaks",
        """**A small network on centred images.** One hidden layer on raw pixels has no built-in tolerance for position, which is why a one-pixel move costs it so much. A convolutional network with pooling builds part of the shift invariance into its architecture and needs the augmentation less. The point about what augmentation supplies is the same; the size of the effect is specific to this model.

**Equal steps, not equal epochs.** Every network took exactly 300 gradient steps, so each image in the augmented and the larger real sets was seen fewer times. Giving every network the same number of epochs instead would have given the 150-digit network a fifth of the steps, a different comparison.

**One kind of shift.** The moved test set uses exactly the transformation the shift augmentation uses. Test data that vary in a different way, by rotation of a few degrees or by thickness of stroke, would reward a different augmentation and not this one.

**Small data.** With 150 training images every source of information shows. With tens of thousands, real data already contain much of the variation, and augmentations matter mostly for the invariances the real data never exhibit.""")

    post.add(
        "What to keep",
        f"""1. Augmented copies are not new observations. On the clean test digits, four times more real data added {g['real']:+.3f}; four shifted copies of each image changed accuracy by {g['shift']:+.3f}, and noisy copies by {g['noise']:+.3f}.

2. An augmentation asserts that the label survives a transformation. Shifted copies added {gm['shift']:+.3f} on test digits moved by a pixel, where four times more centred real digits added {gm['real']:+.3f}.

3. A false assertion costs accuracy. Mirrors and half-turns change the label of at least three digit classes, and training on them cost {-g['flip']:.3f} and {-g['rot180']:.3f}.

4. Choose augmentations by the variation the test data will have and the label will survive, not by how much they enlarge the training set.""")

    post.add(
        "Exercise",
        """Take an image or text model you trained with augmentation and build two extra test sets: one where each test example has been transformed by one of your augmentations, and one where it has not. Train once with the augmentation and once without, and score both models on both test sets.

If the augmentation helps only on the transformed test set, it is buying an invariance your real test data may or may not need. Then look at a sample of your augmented training images with their labels and count how many you would label differently yourself.""")

    post.add(
        "Next",
        """The moved test digits were a small version of a bigger problem: a model meeting inputs that differ from its training inputs. Episode 29 takes that problem on its own terms, with the inputs shifted and the rule that labels them held fixed, and asks how much accuracy the shift costs and whether reweighting the training data can buy it back.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
