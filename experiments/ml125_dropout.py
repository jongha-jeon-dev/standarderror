"""ML 25: Weight Scaling Keeps Dropout's Answers, Not Its Doubts.

The twenty-fifth episode of *Machine Learning, Taught Through What Breaks*,
closing Arc V. "Dropout at test time is approximated by scaling the weights."
Measured on two networks trained with dropout 0.5, comparing the
weight-scaled network with the average of up to 300 sampled masks:

* On the digits the two agree on 99.7% of labels and have the same
  accuracy, but the weight-scaled network is more confident (mean top
  probability 0.985 against 0.954) and its log loss is 28% higher.
* On a hard simulated task they agree on 94.3% of labels, and the rows they
  disagree on are the ones the averaged network calls close to a coin flip.
* A single sampled mask is worse than either.

Run: `standarderror run ml125_dropout --publish`
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
    d, g = nets.dropout_mc(), nets.dropout_mc(task="gauss")
    w, m = np.array(g["pair"]["ws"]), np.array(g["pair"]["mc"])
    return {"d": d, "g": g, "pair_dis": float(np.mean((w > .5) != (m > .5))),
            "pair_med": float(np.median(abs(w - m))),
            "pair_max": float(np.max(abs(w - m)))}


def _first_k_below(r):
    """Smallest K whose averaged log loss is below weight scaling's."""
    for k, v in zip(r["Ks"], r["mc_nll"]):
        if v < r["ws_nll"]:
            return k
    return None


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    d, g = res["d"], res["g"]

    def ratio(ax, m):
        for r, lab, c in ((d, "digits", 0), (g, "hard simulated task", 1)):
            ax.plot(r["Ks"], [v / r["ws_nll"] for v in r["mc_nll"]],
                    marker="o", ms=4, color=m.series[c], label=lab)
        ax.axhline(1.0, color=m.ink, lw=1, ls="--")
        ax.text(1.05, 1.012, "weight scaling", fontsize=8.5, color=m.ink)
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    out["f0"] = charts.diagram(
        ratio,
        title="How many masks it takes to beat weight scaling",
        subtitle=("Test log loss of the average over K sampled dropout masks, "
                  "as a multiple of the weight-scaled network's. Average of "
                  "three seeds."),
        xlabel="masks averaged, K",
        ylabel="log loss relative to weight scaling",
        figsize=(7.2, 4.5),
        source=("UCI digits (scikit-learn) and simulated; "
                "standarderror/ml/nets.py."),
        alt=("Two falling curves crossing a dashed line at 1: the digits "
             "curve crosses at a handful of masks and ends well below; the "
             "simulated curve crosses later and ends just below."),
        caption=(f"On the digits, {_first_k_below(d)} masks beat weight "
                 f"scaling and 300 bring log loss to "
                 f"{d['mc_nll'][-1] / d['ws_nll']:.0%} of it. On the hard "
                 f"task it takes {_first_k_below(g)}, and the gain is "
                 f"{1 - g['mc_nll'][-1] / g['ws_nll']:.1%}."),
        path=str(IMG / f"ml125-f0-ratio.{EXT}"))[0]

    w, mc = np.array(g["pair"]["ws"]), np.array(g["pair"]["mc"])

    def scatter(ax, m):
        dis = (w > .5) != (mc > .5)
        ax.scatter(w[~dis], mc[~dis], s=4, alpha=0.35, color=m.series[0],
                   label="same label")
        ax.scatter(w[dis], mc[dis], s=7, alpha=0.8, color=m.series[1],
                   label="different label")
        ax.plot([0, 1], [0, 1], lw=1, ls="--", color=m.ink)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    out["f1"] = charts.diagram(
        scatter,
        title="Where the two test-time rules part",
        subtitle=("Probability of class 1 for 5,000 test rows of the hard "
                  "task, from the weight-scaled network and from the average "
                  "of 300 masks of the same network (seed 0)."),
        xlabel="weight scaling",
        ylabel="average of 300 masks",
        equal=True,
        figsize=(6.0, 6.0),
        source="Simulated; standarderror/ml/nets.py.",
        alt=("A tight cloud along the diagonal; the orange points where the "
             "labels differ sit near the centre, around 0.5 on both axes."),
        caption=(f"The median gap between the two probabilities is "
                 f"{res['pair_med']:.3f} and the largest {res['pair_max']:.2f}. "
                 f"The {res['pair_dis']:.1%} of rows with different labels sit "
                 f"where both rules are close to 0.5."),
        path=str(IMG / f"ml125-f1-scatter.{EXT}"))[0]

    rows = []
    for r, name in ((d, "digits"), (g, "hard task")):
        rows.append([name, "weight scaling", f"{r['ws_acc']:.3f}",
                     f"{r['ws_nll']:.3f}", "—", f"{r['ws_conf']:.3f}"])
        for k in (1, 10, 300):
            i = r["Ks"].index(k)
            rows.append([name, f"{k} mask" + ("s" if k > 1 else ""),
                         f"{r['mc_acc'][i]:.3f}", f"{r['mc_nll'][i]:.3f}",
                         f"{r['mc_agree'][i]:.1%}",
                         f"{r['mc_conf']:.3f}" if k == 300 else ""])
    out["f2"] = charts.table_image(
        rows,
        header=["data", "test-time rule", "accuracy", "log loss",
                "same label as weight scaling", "mean top probability"],
        title="Two ways to switch dropout off",
        subtitle=("Networks trained with dropout 0.5 after each hidden layer; "
                  "test metrics averaged over three seeds. Top probability is "
                  "shown for weight scaling and for 300 masks."),
        source=("UCI digits (scikit-learn) and simulated; "
                "standarderror/ml/nets.py."),
        alt=("An eight-row table: on the digits, accuracy equal and log loss "
             "lower for 300 masks; on the hard task, both nearly equal; one "
             "mask is worse than weight scaling on every measure."),
        caption=("Accuracy barely separates the rules; log loss and "
                 "confidence do, on the digits. One sampled mask is the worst "
                 "of all."),
        bold_cells={(0, 3), (3, 3), (0, 5), (3, 5), (7, 4)},
        align="llrrrr",
        path=str(IMG / f"ml125-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    d, g = res["d"], res["g"]

    def agree(panel, m):
        panel.bar([0, 1], [d["mc_agree"][-1], g["mc_agree"][-1]],
                  color=m.series[0])
        panel.set_ylim(0.8, 1.0)

    def nll(panel, m):
        panel.plot([v / d["ws_nll"] for v in d["mc_nll"]], color=m.series[1],
                   lw=2)
        panel.axhline(1.0, color=m.ink, lw=1)

    def conf(panel, m):
        panel.bar([0, 1], [d["ws_conf"], d["mc_conf"]], color=m.series[2])
        panel.set_ylim(0.9, 1.0)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=25,
        headline="Same answers, less doubt",
        panels=[(agree, f"{d['mc_agree'][-1]:.1%}", "labels agree, digits"),
                (nll, f"{d['ws_nll'] / d['mc_nll'][-1] - 1:+.0%}",
                 "log loss, weight scaling"),
                (conf, f"{d['ws_conf']:.3f}", "mean top probability")],
        note=("Scaling the weights reproduces the labels that averaging "
              "dropout masks would give, and drops the doubt: on the digits "
              "its probabilities are sharper and its log loss higher."),
        alt=("Three hand-drawn frames: two tall bars; a falling line crossing "
             "a level one; two bars of slightly different height."),
        path=str(IMG / f"ml125-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["digits"] = s.run("""
        from standarderror.ml import nets

        d = nets.dropout_mc()          # digits, dropout 0.5, three seeds
        print(f"weight scaling  accuracy {d['ws_acc']:.3f}  "
              f"log loss {d['ws_nll']:.3f}")
        for k, acc, nll, same in zip(d["Ks"], d["mc_acc"], d["mc_nll"],
                                     d["mc_agree"]):
            print(f"{k:3d} masks       accuracy {acc:.3f}  log loss {nll:.3f}  "
                  f"same label {same:.1%}")
        """)
    out["conf"] = s.run("""
        print(f"mean top probability: weight scaling {d['ws_conf']:.3f}, "
              f"300 masks {d['mc_conf']:.3f}")
        print(f"averaged top probability where the labels differ: "
              f"{d['dis_conf']:.2f}")
        """)
    out["hard"] = s.run("""
        g = nets.dropout_mc(task="gauss")   # episode 4's task, Bayes error 0.28
        print(f"weight scaling {g['ws_acc']:.3f} / {g['ws_nll']:.4f}, "
              f"300 masks {g['mc_acc'][-1]:.3f} / {g['mc_nll'][-1]:.4f}, "
              f"same label {g['mc_agree'][-1]:.1%}")
        print(f"averaged top probability where the labels differ: "
              f"{g['dis_conf']:.2f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    d, g = res["d"], res["g"]
    k_d, k_g = _first_k_below(d), _first_k_below(g)
    i10 = d["Ks"].index(10)

    # The spine, asserted rather than trusted.
    assert abs(d["mc_acc"][-1] - d["ws_acc"]) < 0.003
    assert d["mc_agree"][-1] > 0.995
    assert d["ws_nll"] > 1.2 * d["mc_nll"][-1]
    assert d["ws_conf"] > d["mc_conf"] + 0.02
    assert d["ws_ece"] < d["mc_ece"]
    assert d["dis_conf"] < 0.5 < d["agree_conf"]
    assert k_d is not None and k_d <= 5
    assert k_g is not None and 10 < k_g <= 30
    assert 0.93 < g["mc_agree"][-1] < 0.96
    assert g["dis_conf"] < 0.55 and g["agree_conf"] > 0.7
    assert abs(g["mc_acc"][-1] - g["ws_acc"]) < 0.005
    assert 0 < 1 - g["mc_nll"][-1] / g["ws_nll"] < 0.01
    assert d["mc_acc"][0] < d["ws_acc"] - 0.01
    assert g["mc_acc"][0] < g["ws_acc"] - 0.01
    assert d["mc_nll"][0] > d["ws_nll"] and g["mc_nll"][0] > g["ws_nll"]
    assert d["mc_agree"][i10] < d["mc_agree"][-1]

    post = Post(
        title=(f"{SERIES_TAG} 25: Weight Scaling Keeps Dropout's Answers, "
               f"Not Its Doubts"),
        slug="ml-25-dropout",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=25,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Switching dropout off and scaling the weights gives almost "
                  "the same labels as averaging many dropout masks. It does "
                  "not give the same probabilities."),
        summary=(
            f"Networks trained with dropout 0.5 were evaluated two ways: with "
            f"dropout off and the weights scaled, and by averaging up to 300 "
            f"sampled masks. On the digits the two agreed on "
            f"{d['mc_agree'][-1]:.1%} of labels at the same accuracy, but "
            f"weight scaling was more confident (mean top probability "
            f"{d['ws_conf']:.3f} against {d['mc_conf']:.3f}) and its log loss "
            f"was {d['ws_nll'] / d['mc_nll'][-1] - 1:.0%} higher; {k_d} masks "
            f"already beat it. On a hard simulated task they agreed on "
            f"{g['mc_agree'][-1]:.1%} of labels, the disagreements falling on "
            f"rows the average called close to a coin flip, and log loss "
            f"differed by {1 - g['mc_nll'][-1] / g['ws_nll']:.1%}. A single "
            f"sampled mask was worse than either rule."),
        tags=["machine-learning", "deep-learning", "neural-networks",
              "dropout", "uncertainty", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Optical Recognition of Handwritten Digits (UCI), as bundled with "
            "scikit-learn, CC BY 4.0; stratified half splits, seeds 0 to 2.",
            "Simulated: episode 4's two-class Gaussian task in 10 dimensions, "
            "Bayes error 0.276; 2,000 training rows and 5,000 test rows.",
            "Machinery: `standarderror/ml/nets.py`, tested in "
            "`tests/test_ml.py`; PyTorch networks with two hidden layers "
            "(256 for digits, 128 for the simulated task), ReLU, dropout 0.5 "
            "after each hidden layer, SGD with momentum 0.9, learning rate "
            "0.01, minibatches of 32, 60 epochs.",
            "Where this stops: Srivastava et al., \"Dropout: a simple way to "
            "prevent neural networks from overfitting\", *JMLR* (2014); Gal "
            "and Ghahramani, \"Dropout as a Bayesian approximation\", *ICML* "
            "(2016).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("network and split seeds 0 to 2; dropout masks "
                            "drawn after torch.manual_seed(1000 + seed); "
                            "torch.set_num_threads(1)"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Two ways to switch it off",
        f"""Episode 24 found a layer that behaves differently in training and at inference, and a prediction that depended on which mode the network was in. Dropout is the other common layer with two modes. During training it zeroes each hidden unit with probability p, a fresh random choice for every example at every step, so the network learns not to rely on any one unit. At test time it is switched off, and the weights (or, in PyTorch's convention, the training-time activations) are scaled so that each unit's expected input is the same in both modes.

The original paper presents that scaling as a cheap stand-in for something more principled: averaging the predictions of all the thinned networks the random masks define, of which there are exponentially many. The sentence under test is the textbook's version: *dropout at test time is approximated by scaling the weights*. Approximated how well, and in what?

The average over masks can be estimated directly by leaving dropout on at test time and averaging K forward passes. On the digits, with dropout 0.5 after both hidden layers of a 256-unit network:

{snip['digits'].markdown()}

On the label, the approximation is excellent. The average of 300 masks and the weight-scaled network agree on **{d['mc_agree'][-1]:.1%}** of test digits, and their accuracies differ by {abs(d['mc_acc'][-1] - d['ws_acc']):.3f}. On log loss it is not: the weight-scaled network's is {d['ws_nll']:.3f}, the average's {d['mc_nll'][-1]:.3f}, a gap of {d['ws_nll'] / d['mc_nll'][-1] - 1:.0%}. Averaging just {k_d} masks already beats weight scaling on log loss.""",
        figures=[figs["f0"]])

    post.add(
        "Why scaling is only an approximation",
        f"""For a single linear unit the two rules are the same thing. A unit computing w·x with each input kept with probability 1 − p has expected value (1 − p) w·x, which is exactly what scaling the weights computes. The approximation enters through everything that is not linear. A ReLU of an average is not the average of a ReLU, and a softmax of averaged logits is not the average of softmaxes; with two hidden layers the gaps compound.

The direction of the error is visible in the confidence:

{snip['conf'].markdown()}

The weight-scaled network puts a mean of **{d['ws_conf']:.3f}** on its top class, the average of masks {d['mc_conf']:.3f}. Each thinned network is a slightly different classifier, and on hard digits they vote differently. The average carries that disagreement into its probabilities; the single weight-scaled network has no way to represent it. The few digits where the two rules give different labels are exactly the ones where the masks split: the average's top probability on them is {d['dis_conf']:.2f}, against {d['agree_conf']:.2f} elsewhere.

Log loss punishes a confident mistake far more than an unsure one, which is why the gap shows there. A coarser calibration check misses it: the expected calibration error over ten bins is {d['ws_ece']:.3f} for weight scaling and {d['mc_ece']:.3f} for the average, because on a task this easy nearly every prediction lands in the top bin and the bins cannot resolve where the confidence is misplaced.""",
        figures=[figs["f2"]])

    post.add(
        "On a hard task",
        f"""The digits are nearly separable; most test images are far from any boundary. Episode 4's simulated task is the opposite: two overlapping Gaussian classes with a Bayes error of 0.276, so a large share of rows sit near the boundary whatever the model.

{snip['hard'].markdown()}

Here the labels agree less, on {g['mc_agree'][-1]:.1%} of rows, and the probabilities agree more: the log losses differ by {1 - g['mc_nll'][-1] / g['ws_nll']:.1%}, and it takes {k_g} masks before the average beats weight scaling at all. Both observations have the same cause. The rows near the boundary are many, and on them both rules sit near 0.5, so a small difference in probability moves a row across the threshold without moving its log loss much. The averaged network's top probability on the disputed rows is {g['dis_conf']:.2f}. Those rows are coin flips under either rule; which side of 0.5 they fall on is noise.

So the approximation is good in the sense that matters for decisions, and loose in the sense that matters for probabilities, and how loose depends on the task: a sharpened probability costs most where the model is usually sure.""",
        figures=[figs["f1"]])

    post.add(
        "The cheap version that is worse than both",
        f"""The averaging needs K forward passes per prediction. The tempting shortcut is one: leave dropout on and take one sample. It is worse than either rule on every measure here. On the digits a single mask scores {d['mc_acc'][0]:.3f} against {d['ws_acc']:.3f}, with log loss {d['mc_nll'][0]:.3f}, and agrees with the weight-scaled label on only {d['mc_agree'][0]:.1%} of digits. On the hard task one mask scores {g['mc_acc'][0]:.3f} against {g['ws_acc']:.3f}.

That is the cost of a random network: one draw of the mask removes half the units, and the prediction inherits that draw's luck. It matters in practice because forgetting to switch a model to evaluation mode is a common bug, and a model served in training mode makes exactly these single-mask predictions, a different one each time the same input arrives.

The more principled use of sampling is Gal and Ghahramani's reading of dropout as approximate Bayesian inference, where the spread of the K sampled predictions is taken as a measure of the model's uncertainty. What this episode measures supports the narrow part of that reading: the spread is real, it concentrates on the rows where the labels are in doubt, and the weight-scaled network throws it away. Whether it is a *calibrated* measure of uncertainty is a separate question this episode does not answer.""")

    post.add(
        "Where this breaks",
        """**One dropout rate.** At p = 0.5 each mask removes half the units, the largest perturbation in common use. At p = 0.1, the thinned networks are much closer to each other and to the scaled one, and both gaps shrink.

**Small fully connected networks.** Two hidden layers; the gap compounds with depth, so deeper networks could differ more. Convolutional networks usually apply dropout only near the output, which limits how much nonlinearity sits between the masks and the prediction.

**300 masks is an estimate.** The average of 300 masks is not the exact average over all of them; the curves above are flat by 100, which suggests the remaining error is small.

**Calibration was not fixed.** A temperature fitted on held-out data could bring the weight-scaled network's log loss down without any sampling. The comparison here is between the two rules as they come, which is how they are used.""")

    post.add(
        "What to keep",
        f"""1. Weight scaling reproduces the labels of mask averaging closely: {d['mc_agree'][-1]:.1%} agreement on the digits, {g['mc_agree'][-1]:.1%} on a hard task, accuracy within half a point.

2. It does not reproduce the probabilities. On the digits it was sharper (mean top probability {d['ws_conf']:.3f} against {d['mc_conf']:.3f}) and its log loss {d['ws_nll'] / d['mc_nll'][-1] - 1:.0%} higher; {k_d} masks were enough to beat it.

3. The labels the two rules dispute are the ones the masks dispute. The averaged top probability on them was {d['dis_conf']:.2f} on the digits and {g['dis_conf']:.2f} on the hard task.

4. One sampled mask, the prediction of a model left in training mode, was worse than both rules on accuracy and log loss.""")

    post.add(
        "Exercise",
        """Take a network you trained with dropout and evaluate it on a held-out set three ways: in evaluation mode, with dropout on and one pass, and with dropout on averaged over 30 passes. Compare accuracy, log loss and the mean top-class probability.

Then sort the held-out rows by how much the 30 passes disagree with each other, and check the error rate in the top tenth against the bottom tenth. If the disagreement tracks the errors, you have a free warning signal that evaluation mode was discarding.""")

    post.add(
        "Next",
        """Arc V ends here. Initialisation, learning rate and batch size, the optimiser, batch normalisation and dropout each turned out to be less separate, or less exact, than the sentence attached to it. Arc VI turns from how a network trains to what it learns, starting with the sentence that a network which generalises could not have fit noise.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
