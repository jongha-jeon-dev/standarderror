"""ML 21: Initialisation Must Break Symmetry, at the Right Scale.

The twenty-first episode of *Machine Learning, Taught Through What Breaks*,
opening Arc V. "Initialisation only needs to break symmetry." Measured on
networks of 30 hidden layers of 128 units on the digits, with weights drawn
from N(0, g^2 / fan-in) for four gains g, best of four learning rates:

* Setting every weight to the same value never breaks symmetry: after
  training, all 128 units of the first layer are still identical, and test
  accuracy is 0.197.
* Random weights at the wrong scale break symmetry and still fail. With ReLU
  the activations shrink or grow by about g/sqrt(2) per layer: at g = 1 they
  are 1e-4 times their first-layer size by layer 30 and the network scores
  0.13; at g = 2 they are 4e4 times larger and it scores 0.10. At g = sqrt(2)
  they hold steady and it scores 0.919.
* With three hidden layers instead of thirty, every random scale trains.

Run: `standarderror run ml121_init_scale --publish`
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

GNAME = {0.5: "0.5", 1.0: "1", round(math.sqrt(2), 6): "√2", 2.0: "2"}


def _g(g):
    return GNAME[round(g, 6)]


def compute() -> dict:
    deep, shallow = nets.init_depth(), nets.init_depth(depth=3)
    key = {(r["act"], _g(r["gain"])): r for r in deep["rows"]}
    key3 = {(r["act"], _g(r["gain"])): r for r in shallow["rows"]}
    per_layer = {}
    for (act, g), r in key.items():
        s = r["stds"]
        if s[0] > 0 and s[-1] > 0:
            per_layer[(act, g)] = (s[-1] / s[0]) ** (1 / (len(s) - 1))
    return {"deep": deep, "shallow": shallow, "key": key, "key3": key3,
            "per_layer": per_layer}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    deep, key, key3 = res["deep"], res["key"], res["key3"]

    def stds(ax, m):
        for r in deep["rows"]:
            c = list(GNAME.values()).index(_g(r["gain"]))
            ls = "-" if r["act"] == "tanh" else "--"
            s = np.maximum(r["stds"], 1e-16)
            ax.plot(range(1, len(s) + 1), s, color=m.series[c], ls=ls,
                    lw=1.6, label=f"{r['act']}, g = {_g(r['gain'])}")
        ax.set_yscale("log")
        ax.set_ylim(1e-15, 1e6)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")

    rh = key[("relu", "√2")]
    out["f0"] = charts.diagram(
        stds,
        title="Thirty layers multiply the initial scale",
        subtitle=("Standard deviation of each hidden layer's activations on "
                  "the digits at initialisation, weights drawn from "
                  "N(0, g² / fan-in) with zero biases."),
        xlabel="hidden layer",
        ylabel="activation standard deviation (log scale)",
        figsize=(7.4, 4.8),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("Eight lines on a log scale over 30 layers: four stay roughly "
             "level, two fall steeply towards 1e-10 and below, one ReLU line "
             "rises to about 1e4, and one tanh line falls slowly."),
        caption=(f"With ReLU each layer multiplies the scale by about "
                 f"g/√2. At g = √2 the thirtieth layer's activations have "
                 f"standard deviation {rh['stds'][-1]:.2f}; at g = 1, "
                 f"{key[('relu', '1')]['stds'][-1]:.0e}."),
        path=str(IMG / f"ml121-f0-stds.{EXT}"))[0]

    labels = [f"{a}\ng = {g}" for a in ("tanh", "relu") for g in GNAME.values()]
    deep_b = [key[(a, g)]["best"] for a in ("tanh", "relu")
              for g in GNAME.values()]
    shal_b = [key3[(a, g)]["best"] for a in ("tanh", "relu")
              for g in GNAME.values()]

    def bars(ax, m):
        x = np.arange(len(labels) + 1)
        w = 0.38
        ax.bar(x[:-1] - w / 2, shal_b, w, color=m.series[2],
               label="3 hidden layers")
        ax.bar(x[:-1] + w / 2, deep_b, w, color=m.series[0],
               label="30 hidden layers")
        ax.bar([x[-1] - w / 2], [res["shallow"]["const_best"]], w,
               color=m.series[2])
        ax.bar([x[-1] + w / 2], [deep["const_best"]], w, color=m.series[0])
        ax.axhline(0.1, color=m.grid, lw=1, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(labels + ["constant\n0.01"], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left",
                  bbox_to_anchor=(0.0, 1.12), ncol=2)

    out["f1"] = charts.diagram(
        bars,
        title="Depth turns a scale into a verdict",
        subtitle=("Best test accuracy over learning rates 0.001 to 0.03, SGD "
                  "with momentum 0.9 for 30 epochs, for each activation and "
                  "gain; the dotted line is chance."),
        ylabel="test accuracy",
        figsize=(8.0, 4.8),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("Paired bars: with 3 layers every random initialisation reaches "
             "above 0.9; with 30 layers four of the eight fall to near "
             "chance, and the constant initialisation is low at both depths."),
        caption=("The same gains that fail at thirty layers all train at "
                 "three. The constant initialisation fails at both."),
        path=str(IMG / f"ml121-f1-bars.{EXT}"))[0]

    rows = []
    for a in ("tanh", "relu"):
        for g in GNAME.values():
            r, r3 = key[(a, g)], key3[(a, g)]
            pl = res["per_layer"].get((a, g))
            rows.append([a, g, f"{r['stds'][-1]:.2g}",
                         f"{pl:.3f}" if pl else "—",
                         f"{r['grads'][0]:.2g}", f"{r['best']:.3f}",
                         f"{r['best_lr']}", f"{r3['best']:.3f}"])
    rows.append(["tanh", "constant", "—", "—", "—",
                 f"{deep['const_best']:.3f}", "",
                 f"{res['shallow']['const_best']:.3f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["activation", "gain", "sd at layer 30", "scale per layer",
                "first-layer gradient norm", "best accuracy, 30 layers",
                "at rate", "best accuracy, 3 layers"],
        title="Symmetry broken, scale wrong",
        subtitle=("Scale per layer is (sd at layer 30 / sd at layer 1)^(1/29); "
                  "for ReLU the theory says g/√2. Gradient norms at "
                  "initialisation, on the training half."),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("A nine-row table of activation scale, gradient size and "
             "accuracy for each activation and gain at 30 and 3 layers."),
        caption=("Every row but the last breaks symmetry. Only the rows whose "
                 "scale per layer is close to 1 train at thirty layers."),
        bold_cells={(5, 3), (5, 5), (6, 3)},
        align="llrrrrrr",
        path=str(IMG / f"ml121-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    key, deep = res["key"], res["deep"]

    def vanish(panel, m):
        panel.plot(np.log10(np.maximum(key[("relu", "1")]["stds"], 1e-16)),
                   color=m.series[1], lw=2)

    def steady(panel, m):
        panel.plot(np.log10(key[("relu", "√2")]["stds"]), color=m.series[2],
                   lw=2)
        panel.set_ylim(-5, 1)

    def const(panel, m):
        panel.bar([0, 1], [deep["const_best"], key[("relu", "√2")]["best"]],
                  color=[m.series[3], m.series[2]])
        panel.set_ylim(0, 1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=21,
        headline="Random is not enough",
        panels=[(vanish, f"{key[('relu', '1')]['best']:.2f}",
                 "ReLU, g = 1, 30 layers"),
                (steady, f"{key[('relu', '√2')]['best']:.3f}",
                 "ReLU, g = √2"),
                (const, f"{deep['const_best']:.3f}", "all weights equal")],
        note=("Equal weights never break symmetry; random weights at the wrong "
              "scale break it and still fail, because thirty layers multiply "
              "the scale thirty times."),
        alt=("Three hand-drawn frames: a line falling steeply; a level line; "
             "a short bar beside a tall one."),
        path=str(IMG / f"ml121-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["stds"] = s.run("""
        from standarderror.ml import nets

        r = nets.init_depth()          # 30 hidden layers of 128 units
        for row in r["rows"]:
            sd = row["stds"]
            print(f"{row['act']:4s} g = {row['gain']:.2f}: activation sd at "
                  f"layers 1, 10, 30 = {sd[0]:.2g}, {sd[9]:.2g}, {sd[29]:.2g}")
        """)
    out["acc"] = s.run("""
        for row in r["rows"]:
            print(f"{row['act']:4s} g = {row['gain']:.2f}: best test accuracy "
                  f"{row['best']:.3f} at learning rate {row['best_lr']}")
        print(f"all weights 0.01: {r['const_best']:.3f}; distinct units in "
              f"layer 1 after training: {r['const_distinct_units']}")
        """)
    out["shallow"] = s.run("""
        r3 = nets.init_depth(depth=3)  # the same, with 3 hidden layers
        print("  ".join(f"{row['act']} {row['gain']:.2f}: {row['best']:.3f}"
                        for row in r3["rows"][:4]))
        print("  ".join(f"{row['act']} {row['gain']:.2f}: {row['best']:.3f}"
                        for row in r3["rows"][4:]))
        print(f"constant: {r3['const_best']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    deep, shallow, key = res["deep"], res["shallow"], res["key"]
    pl = res["per_layer"]
    r1, rh, r2 = key[("relu", "1")], key[("relu", "√2")], key[("relu", "2")]
    t5, t1, th, t2 = (key[("tanh", g)] for g in ("0.5", "1", "√2", "2"))

    # The spine, asserted rather than trusted.
    assert deep["const_distinct_units"] == 1
    assert deep["const_best"] < 0.25 and shallow["const_best"] < 0.5
    for g, val in (("1", 1 / math.sqrt(2)), ("√2", 1.0), ("2", math.sqrt(2))):
        assert abs(pl[("relu", g)] / val - 1) < 0.03, (g, pl[("relu", g)])
    assert r1["stds"][-1] / r1["stds"][0] < 1e-3
    assert r2["stds"][-1] / r2["stds"][0] > 1e4
    assert 0.5 < rh["stds"][-1] / rh["stds"][0] < 2
    assert r1["best"] < 0.15 and r2["best"] < 0.11 and rh["best"] > 0.9
    assert t5["best"] < 0.11 and key[("relu", "0.5")]["best"] < 0.11
    assert t1["best"] > 0.9 and th["best"] > 0.9
    assert t5["stds"][-1] < 1e-8 and t5["grads"][0] < 1e-6
    assert t2["grads"][0] > 10 * t2["grads"][-1] and t2["best"] < 0.8
    assert all(r["best"] > 0.9 for r in shallow["rows"])

    post = Post(
        title=(f"{SERIES_TAG} 21: Initialisation Must Break Symmetry, at the "
               f"Right Scale"),
        slug="ml-21-init-scale",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=21,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Equal initial weights never become different, so they "
                  "have to be random. In a deep network, random at the wrong "
                  "scale fails just as completely."),
        summary=(
            f"A network of 30 hidden layers of 128 units was trained on the "
            f"digits from weights drawn with standard deviation g / "
            f"sqrt(fan-in). Constant weights never broke symmetry: after "
            f"training all first-layer units were identical and accuracy was "
            f"{deep['const_best']:.3f}. Random weights broke it but did not "
            f"always train. With ReLU each layer multiplied the activations' "
            f"scale by {pl[('relu', '1')]:.3f} at g = 1 (theory 0.707) and "
            f"{pl[('relu', '2')]:.3f} at g = 2 (theory 1.414); those networks "
            f"scored {r1['best']:.3f} and {r2['best']:.3f}, against "
            f"{rh['best']:.3f} at g = sqrt(2). With tanh, g = 0.5 left "
            f"gradients near 1e-9 and scored {t5['best']:.3f}. With 3 hidden "
            f"layers every random scale trained above 0.9."),
        tags=["machine-learning", "deep-learning", "neural-networks",
              "initialization", "optimization", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Optical Recognition of Handwritten Digits (UCI), as bundled with "
            "scikit-learn, CC BY 4.0; one stratified half split, "
            "standardised on the training half.",
            "Machinery: `standarderror/ml/nets.py`, tested in "
            "`tests/test_ml.py`; PyTorch networks 64-128x30-10 (and "
            "64-128x3-10), weights N(0, g² / fan-in) and zero biases, SGD "
            "with momentum 0.9, minibatches of 32, 30 epochs, best of "
            "learning rates 0.001, 0.003, 0.01 and 0.03 on the test half.",
            "Where this stops: Glorot and Bengio, \"Understanding the "
            "difficulty of training deep feedforward neural networks\", "
            "*AISTATS* (2010); He, Zhang, Ren and Sun, \"Delving deep into "
            "rectifiers\", *ICCV* (2015); Saxe, McClelland and Ganguli, "
            "\"Exact solutions to the nonlinear dynamics of learning in deep "
            "linear neural networks\", *ICLR* (2014).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("split, network and shuffling seed 0; "
                            "torch.set_num_threads(1)"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Why the weights start random",
        f"""Arc IV ended with networks that differed only in their random seed and disagreed on a fifth of their answers. Arc V is about training a network, and it starts before the first step, with the numbers the weights start from. The textbook's reason for starting them at random is symmetry. If every weight in a layer starts at the same value, every unit in that layer computes the same function, receives the same gradient and makes the same update, and the layer stays a single unit copied 128 times however long it trains. Random weights break the tie. The sentence under test is the conclusion often drawn from that argument: *initialisation only needs to break symmetry*.

The first half of the argument is easy to confirm. A network with 30 hidden layers of 128 units, every weight set to 0.01, trained on the digits with the best of four learning rates, scores {deep['const_best']:.3f}, and after training the 128 units of its first layer are still exactly identical: the count of distinct units is {deep['const_distinct_units']}. Symmetry is never broken by training alone.

The second half, *only*, is the claim to test. Draw the weights from a normal distribution with standard deviation g / √fan-in, the form every standard initialiser takes, and vary the gain g. Every one of these breaks symmetry. Here is what each does to the activations of the same 30-layer network before any training:

{snip['stds'].markdown()}""",
        figures=[figs["f0"]])

    post.add(
        "Thirty multiplications",
        f"""Each layer multiplies the scale of its input by a roughly constant factor, and thirty layers multiply it thirty times. For a ReLU layer the factor can be worked out: a weight variance of g² / fan-in multiplies the second moment of the input by g², and the ReLU, which zeroes half its inputs, halves it. The standard deviation therefore changes by about g / √2 per layer. Measured over 29 steps, the factor is **{pl[('relu', '1')]:.3f}** at g = 1 (theory 0.707), **{pl[('relu', '√2')]:.3f}** at g = √2 (theory 1) and **{pl[('relu', '2')]:.3f}** at g = 2 (theory 1.414).

Small factors compound into enormous ones. At g = 1 the thirtieth layer's activations are {r1['stds'][-1] / r1['stds'][0]:.0e} times the first layer's; at g = 2, {r2['stds'][-1] / r2['stds'][0]:.0e} times. Only g = √2, the value He and colleagues derived for ReLU in 2015, keeps the scale level.

Tanh behaves differently because it saturates. At g = 0.5 its activations shrink geometrically, to {t5['stds'][-1]:.0e} by layer 30, and the gradient reaching the first layer has norm {t5['grads'][0]:.0e}. At g = 2 the activations cannot grow past ±1, so they stay at a standard deviation of {t2['stds'][-1]:.2f}, but the gradient flowing back grows instead: the first layer's gradient norm is {t2['grads'][0]:.1f}, the last layer's {t2['grads'][-1]:.2f}. At g = 1, the value Glorot and Bengio's analysis points to for tanh, the activations fade slowly, to {t1['stds'][-1]:.2f}.""")

    post.add(
        "What the scale does to training",
        f"""Then train each network, giving it the best of four learning rates so that no initialisation loses merely for want of the right step size:

{snip['acc'].markdown()}

Four of the eight random initialisations do not train at all in thirty epochs. ReLU at g = 1 scores **{r1['best']:.3f}**; its signal reaches the last hidden layer at less than a ten-thousandth of its size, and the gradients reaching the early layers are of order {r1['grads'][0]:.0e}. ReLU at g = 2 scores {r2['best']:.3f}, its activations and gradients tens of thousands of times too large for any of the learning rates tried. Tanh at g = 0.5 and ReLU at g = 0.5 score chance. The two ReLU networks that fail sit either side of the one that succeeds, g = √2, which scores **{rh['best']:.3f}**, by a factor of √2 in the standard deviation of the weights.

Tanh at g = 1 and g = √2 both train, to {t1['best']:.3f} and {th['best']:.3f}; tanh at g = 2 reaches {t2['best']:.3f}, at the smallest learning rate tried, where the exploding gradient is least harmful. With one seed and a coarse grid of learning rates, the ordering among the networks that train is not the point. The point is the gap between the ones that train and the ones that cannot.""",
        figures=[figs["f1"]])

    post.add(
        "Depth is what makes it matter",
        f"""The same experiment with three hidden layers instead of thirty is the control:

{snip['shallow'].markdown()}

Every random initialisation trains above 0.9. Two multiplications by 0.707 halve the signal, which the first few updates can absorb; twenty-nine shrink it to less than a ten-thousandth, which they cannot. The constant initialisation still fails at three layers, scoring {shallow['const_best']:.3f}: symmetry does not care about depth.

So the sentence holds in shallow networks, where it was first written, and fails in deep ones. Breaking symmetry is necessary at any depth. In a deep network it is not sufficient: the scale has to be chosen so that each layer passes on a signal of about the same size it received, in both directions. That is the whole content of the initialisers that every deep-learning library now applies by default, and the reason they differ by activation. Glorot's rule targets tanh and sigmoid, He's targets ReLU, and each assumes the other's activation at its peril, as the ReLU network at g = 1 shows.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        """**Plain stacks of layers.** Residual connections, which add each block's input to its output, and normalisation layers, which rescale activations at every layer (episode 24), both make a network far less sensitive to the initial scale. Most modern deep networks have both, which is why this failure is rarer in practice than the arithmetic suggests.

**One seed, four learning rates.** The verdicts are large enough to survive both limits: the failing networks score chance at every rate tried. The differences between the networks that train are not established here.

**Thirty epochs.** A network whose gradients are 1e-5 might eventually begin to learn with a much larger learning rate or many more epochs. Within any reasonable budget it does not, and a larger rate would overflow the layers whose activations are not small.

**The theory is for the start.** The g/√2 factor describes the network at initialisation. Training changes the weights and the scales with them, which is why the ReLU network at g = √2 can train at all: the initialisation only has to give it a place to start.""")

    post.add(
        "What to keep",
        f"""1. Symmetry must be broken: with every weight equal, the 128 units of a layer were still identical after training, and the network scored {deep['const_best']:.3f}.

2. Breaking it is not enough in a deep network. Each layer multiplies the signal's scale by a factor set by the initial scale, about g/√2 for ReLU, measured at {pl[('relu', '1')]:.3f} for g = 1 and {pl[('relu', '2')]:.3f} for g = 2.

3. Over thirty layers those factors decided whether training happened at all: ReLU scored {r1['best']:.3f} at g = 1, {rh['best']:.3f} at g = √2 and {r2['best']:.3f} at g = 2.

4. With three layers every random scale trained above 0.9. Depth turns a modest mis-scaling into a failure. Use the initialiser that matches the activation, and check the activations' scale at the last layer before training.""")

    post.add(
        "Exercise",
        """Take a deep network you train without normalisation or residual connections, and before the first update record the standard deviation of each layer's output on one batch. Plot it against depth on a log scale.

If the line is not roughly flat, compute the per-layer factor and compare it with your initialiser's gain and your activation's. Then multiply the initial weights by the factor's inverse and see whether training starts sooner.""")

    post.add(
        "Next",
        """A good initialisation gives training a place to start. How fast it moves from there is set by the learning rate and the batch size, which every training script exposes as two separate settings. Episode 22 asks whether they are.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
