"""ML 23: Adam Is a Little Faster and No Better.

The twenty-third episode of *Machine Learning, Taught Through What Breaks*.
"Adam converges faster and generalises as well as SGD." Both optimisers
(SGD with momentum 0.9), each with its learning rate tuned on a validation
set, trained from the same initial weights on 20 splits of two problems:

* Test accuracy: Adam ahead by 0.09 points on the digits and 0.05 on a hard
  simulated task, each within about one standard error of zero. The second
  half of the sentence survives.
* Speed depends on the learning rate more than the optimiser. At the rates
  validation chose, SGD's training loss fell faster. At each optimiser's
  fastest rate, Adam reached a training loss of 0.01 in 6 epochs and SGD in
  8.

Run: `standarderror run ml123_adam_sgd --publish`
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


def _to(curve, thr):
    return next((i + 1 for i, v in enumerate(curve) if v < thr), None)


def compute() -> dict:
    d, g, sp = (nets.adam_vs_sgd(), nets.adam_vs_sgd(task="gauss"),
                nets.optimiser_speed())
    speed = {}
    for name in ("adam", "sgd"):
        rows = []
        for lr, c, a in zip(sp[name]["lrs"], sp[name]["curves"],
                            sp[name]["acc"]):
            rows.append({"lr": lr, "to_01": _to(c, 0.1), "to_001": _to(c, 0.01),
                         "e5": c[4], "e10": c[9], "e30": c[29], "acc": a})
        speed[name] = rows
    fastest = {n: min((r for r in speed[n] if r["to_001"]),
                      key=lambda r: (r["to_001"], r["e10"])) for n in speed}
    tuned = {n: next(r for r in speed[n] if r["lr"] == d["tuned"][n])
             for n in speed}
    return {"d": d, "g": g, "sp": sp, "speed": speed, "fastest": fastest,
            "tuned": tuned}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    sp, fastest, d = res["sp"], res["fastest"], res["d"]

    def curves(ax, m):
        for name, c, lab in (("adam", 0, "Adam"), ("sgd", 1, "SGD")):
            for lr, cur in zip(sp[name]["lrs"], sp[name]["curves"]):
                best = lr == fastest[name]["lr"]
                tuned = lr == d["tuned"][name]
                ax.plot(range(1, len(cur) + 1), np.maximum(cur, 1e-5),
                        color=m.series[c], lw=2.2 if best else 1.0,
                        ls="--" if tuned and not best else "-",
                        alpha=1.0 if best or tuned else 0.35,
                        label=(f"{lab} {lr:g}" + (", fastest" if best else
                                                  ", tuned" if tuned else ""))
                        if best or tuned else None)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 3)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    out["f0"] = charts.diagram(
        curves,
        title="Speed belongs to the learning rate",
        subtitle=("Training loss on the digits per epoch for five learning "
                  "rates each, average of five splits. Bold: each "
                  "optimiser's fastest rate; dashed: the rate validation "
                  "chose; faint: the rest."),
        xlabel="epoch",
        ylabel="training loss (log scale)",
        figsize=(7.4, 4.8),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("Ten falling curves on a log scale: the bold Adam curve falls "
             "fastest, the bold SGD curve close behind; the dashed tuned "
             "curves fall much more slowly, the SGD one ahead of the Adam "
             "one."),
        caption=(f"At its fastest rate Adam reaches a training loss of 0.01 "
                 f"in {fastest['adam']['to_001']} epochs, SGD in "
                 f"{fastest['sgd']['to_001']}. At the rates validation chose, "
                 f"{res['tuned']['adam']['to_001']} and "
                 f"{res['tuned']['sgd']['to_001']}."),
        path=str(IMG / f"ml123-f0-curves.{EXT}"))[0]

    g = res["g"]

    def diffs(ax, m):
        for k, r in enumerate((d, g)):
            dd = np.array(r["adam"]["acc"]) - np.array(r["sgd"]["acc"])
            jitter = np.random.default_rng(k).uniform(-0.12, 0.12, len(dd))
            ax.scatter(dd, k + jitter, s=14, color=m.series[k], alpha=0.7)
            ax.errorbar([dd.mean()], [k], xerr=[2 * r["diff_se"]], fmt="D",
                        color=m.ink, capsize=4, ms=5)
        ax.axvline(0, color=m.ink, lw=0.8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["digits", "hard task"])
        ax.set_ylim(-0.6, 1.6)

    out["f1"] = charts.diagram(
        diffs,
        title="Twenty splits, no winner",
        subtitle=("Adam's test accuracy minus SGD's, each point one split, "
                  "both trained from the same initial weights at their "
                  "validation-tuned rates; diamonds are means with two "
                  "standard errors."),
        xlabel="Adam minus SGD, test accuracy",
        figsize=(7.2, 3.8),
        source=("UCI digits (scikit-learn) and simulated; "
                "standarderror/ml/nets.py."),
        alt=("Two rows of points scattered either side of zero, with mean "
             "markers whose error bars cross zero."),
        caption=(f"Mean difference {d['diff_mean']:+.4f} ± "
                 f"{2 * d['diff_se']:.4f} on the digits and "
                 f"{g['diff_mean']:+.4f} ± {2 * g['diff_se']:.4f} on the hard "
                 f"task (two standard errors)."),
        path=str(IMG / f"ml123-f1-diffs.{EXT}"))[0]

    rows = []
    for name, lab in (("adam", "Adam"), ("sgd", "SGD")):
        for r in res["speed"][name]:
            tag = []
            if r["lr"] == fastest[name]["lr"]:
                tag.append("fastest")
            if r["lr"] == d["tuned"][name]:
                tag.append("tuned")
            rows.append([lab, f"{r['lr']:g}", ", ".join(tag),
                         str(r["to_01"] or "—"), str(r["to_001"] or "—"),
                         f"{r['e10']:.4f}", f"{r['acc']:.3f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["optimiser", "rate", "", "epochs to loss 0.1",
                "epochs to loss 0.01", "loss after 10 epochs",
                "test accuracy"],
        title="Every rate on both grids",
        subtitle=("Digits, average of five splits, 30 epochs, minibatches "
                  "of 32; SGD with momentum 0.9. 'Tuned' is the rate chosen "
                  "on a validation fifth of a separate split."),
        source="UCI digits (scikit-learn); standarderror/ml/nets.py.",
        alt=("A ten-row table: Adam's best rates reach low loss in a few "
             "epochs, SGD's a couple of epochs later; test accuracy is "
             "between 0.96 and 0.98 for the middle rates of both."),
        caption=("The rates that train fastest also score within 0.001 of "
                 "the best of their grid. Adam's accuracy varies less across "
                 "its grid than SGD's."),
        align="lrlrrrr",
        path=str(IMG / f"ml123-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    sp, fastest, d = res["sp"], res["fastest"], res["d"]

    def fast(panel, m):
        for name, c in (("adam", 0), ("sgd", 1)):
            i = sp[name]["lrs"].index(fastest[name]["lr"])
            panel.plot(np.log10(np.maximum(sp[name]["curves"][i], 1e-5)),
                       color=m.series[c], lw=2)

    def tuned(panel, m):
        for name, c in (("adam", 0), ("sgd", 1)):
            i = sp[name]["lrs"].index(d["tuned"][name])
            panel.plot(np.log10(sp[name]["curves"][i]), color=m.series[c],
                       lw=2)

    def diff(panel, m):
        dd = np.array(d["adam"]["acc"]) - np.array(d["sgd"]["acc"])
        panel.bar(range(len(dd)), dd, color=m.series[2])
        panel.axhline(0, color=m.ink, lw=1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=23,
        headline="A little faster, no better",
        panels=[(fast, f"{fastest['adam']['to_001']} vs "
                       f"{fastest['sgd']['to_001']}", "epochs, fastest rates"),
                (tuned, f"{res['tuned']['adam']['to_001']} vs "
                        f"{res['tuned']['sgd']['to_001']}",
                 "epochs, tuned rates"),
                (diff, f"{d['diff_mean'] * 100:+.2f} pt", "accuracy, 20 splits")],
        note=("Tuned on validation data, Adam and SGD with momentum reach the "
              "same test accuracy. Adam is faster only at its fastest "
              "learning rate, and only by a couple of epochs."),
        alt=("Three hand-drawn frames: two falling curves close together; "
             "two slower curves; twenty small bars either side of zero."),
        path=str(IMG / f"ml123-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["acc"] = s.run("""
        from standarderror.ml import nets

        for task in ("digits", "gauss"):
            r = nets.adam_vs_sgd(task=task)      # 20 splits, same init each
            a, b = r["adam"]["acc"], r["sgd"]["acc"]
            print(f"{task}: rates Adam {r['tuned']['adam']:g}, SGD "
                  f"{r['tuned']['sgd']:g}; accuracy Adam {sum(a) / 20:.4f}, "
                  f"SGD {sum(b) / 20:.4f}")
            print(f"    difference {r['diff_mean']:+.4f} ± {r['diff_se']:.4f}, "
                  f"Adam ahead on {r['adam_wins']} of 20")
        """)
    out["speed"] = s.run("""
        sp = nets.optimiser_speed()              # digits, five splits
        for name in ("adam", "sgd"):
            for lr, curve in zip(sp[name]["lrs"], sp[name]["curves"]):
                hit = next((i + 1 for i, v in enumerate(curve) if v < 0.01), None)
                when = f"at epoch {hit}" if hit else "not in 30 epochs"
                print(f"{name:4s} rate {lr:<6g} loss after 5 epochs "
                      f"{curve[4]:.4f}, reaches 0.01 {when}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    d, g, fastest, tuned = res["d"], res["g"], res["fastest"], res["tuned"]
    fa, fs = fastest["adam"], fastest["sgd"]
    ta, ts = tuned["adam"], tuned["sgd"]
    acc_a = max(r["acc"] for r in res["speed"]["adam"])
    acc_s = max(r["acc"] for r in res["speed"]["sgd"])

    # The spine, asserted rather than trusted.
    assert abs(d["diff_mean"]) < 2 * d["diff_se"]
    assert abs(g["diff_mean"]) < 2 * g["diff_se"]
    assert abs(d["diff_mean"]) < 0.002 and abs(g["diff_mean"]) < 0.002
    assert d["tuned"] == {"adam": 3e-4, "sgd": 1e-2}
    assert fa["lr"] == 3e-3 and fs["lr"] == 3e-2
    assert fa["to_001"] < fs["to_001"] <= fa["to_001"] + 3
    assert fa["e10"] < fs["e10"] / 3
    assert ta["e5"] > ts["e5"] and ta["to_01"] > ts["to_01"]
    assert abs(ta["to_001"] - ts["to_001"]) <= 2
    assert 0 < acc_a - acc_s < 0.005
    assert acc_a - fa["acc"] < 0.001 and fs["acc"] == acc_s
    assert d["tuning"]["adam"]["val_acc"][1] == d["tuning"]["adam"]["val_acc"][2]
    assert len(set(d["tuning"]["sgd"]["val_acc"][2:])) == 1
    assert (acc_a - min(r["acc"] for r in res["speed"]["adam"]) <
            (acc_s - min(r["acc"] for r in res["speed"]["sgd"])) / 2)
    m_d = np.mean(d["adam"]["acc"])
    assert 1 - g["bayes"] - np.mean(g["adam"]["acc"]) > 0.01

    post = Post(
        title=f"{SERIES_TAG} 23: Adam Is a Little Faster and No Better",
        slug="ml-23-adam-sgd",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=23,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Tuned on validation data and run over twenty splits, Adam "
                  "and SGD with momentum reach the same test accuracy. "
                  "Adam's speed is real but small, and belongs to its "
                  "learning rate."),
        summary=(
            f"Adam and SGD with momentum 0.9, each with its learning rate "
            f"tuned on a validation set, were trained from the same initial "
            f"weights on 20 splits of the digits and of a simulated task with "
            f"Bayes error {g['bayes']:.3f}. Adam's test accuracy was higher "
            f"by {d['diff_mean']:+.4f} ± {d['diff_se']:.4f} on the digits and "
            f"{g['diff_mean']:+.4f} ± {g['diff_se']:.4f} on the simulated "
            f"task: no difference. At the rates validation chose, SGD's "
            f"training loss fell faster. At each optimiser's fastest rate, "
            f"Adam reached a training loss of 0.01 in {fa['to_001']} epochs "
            f"and SGD in {fs['to_001']}, and those rates scored as well as "
            f"any; validation on 180 digits could not tell them from slower "
            f"ones."),
        tags=["machine-learning", "deep-learning", "neural-networks",
              "optimization", "adam", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Optical Recognition of Handwritten Digits (UCI), as bundled with "
            "scikit-learn, CC BY 4.0; stratified half splits, seeds 0 to 19 "
            "(0 to 4 for the speed study), tuning on split 100.",
            "Simulated: episode 4's two-class Gaussian task in 10 dimensions, "
            "Bayes error 0.276; 2,000 training rows per split and 5,000 "
            "fixed test rows.",
            "Machinery: `standarderror/ml/nets.py`, tested in "
            "`tests/test_ml.py`; PyTorch networks with two hidden layers "
            "(256 for digits, 64 for the simulated task), ReLU, minibatches "
            "of 32, 30 epochs; Adam with PyTorch's defaults apart from the "
            "rate; SGD with momentum 0.9. Rates tuned on a grid of five "
            "spanning a factor of 100, by accuracy on a held-out fifth.",
            "Where this stops: Kingma and Ba, \"Adam: a method for stochastic "
            "optimization\", *ICLR* (2015); Wilson, Roelofs, Stern, Srebro "
            "and Recht, \"The marginal value of adaptive gradient methods in "
            "machine learning\", *NeurIPS* (2017); Choi et al., \"On "
            "empirical comparisons of optimizers for deep learning\" "
            "(2019).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("split, network and shuffling seeds as listed; "
                            "both optimisers start from the same weights on "
                            "each split; torch.set_num_threads(1)"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The default optimiser",
        f"""Episode 22 found that for plain SGD the learning rate and the batch size act as one setting. Most networks today are not trained with plain SGD. They are trained with Adam, which keeps a running estimate of each weight's gradient scale and divides by it, so that every weight moves at a similar rate whatever the size of its gradient. Its reputation is the sentence under test: *Adam converges faster and generalises as well as SGD*. Both halves have been challenged. Wilson and colleagues reported adaptive methods generalising worse than SGD; Choi and colleagues found that rankings of optimisers change when every optimiser's settings are tuned with the same care.

So tune both with the same care. On a separate split of the digits, each optimiser's learning rate is chosen from a grid of five spanning a factor of 100, by accuracy on a held-out fifth. SGD gets momentum 0.9, as it usually does in practice. Then both are trained, from the same initial weights, on twenty fresh splits, and the same again on episode 4's simulated task, where the Bayes error of {g['bayes']:.3f} leaves room for a better optimiser to show:

{snip['acc'].markdown()}

On the digits Adam's mean test accuracy is {m_d:.4f}, ahead by **{d['diff_mean']:+.4f}**, with a standard error of {d['diff_se']:.4f}; it is ahead on {d['adam_wins']} splits of 20 and level on {d['ties']}. On the hard task it is ahead by {g['diff_mean']:+.4f} ± {g['diff_se']:.4f}. Neither difference is distinguishable from zero, and on the digits the whole difference is less than one test image in a thousand. The second half of the sentence survives: tuned, Adam generalises as well as SGD here, no better and no worse.""",
        figures=[figs["f1"]])

    post.add(
        "Faster at what rate?",
        f"""The first half is about speed, and speed depends on the learning rate at least as much as on the optimiser. The speed study trains both optimisers at every rate on their grids, on five splits of the digits, and records the training loss after every epoch:

{snip['speed'].markdown()}

At the rates validation chose, {ta['lr']:g} for Adam and {ts['lr']:g} for SGD, **SGD is faster**: after five epochs its training loss is {ts['e5']:.3f} against Adam's {ta['e5']:.3f}, and it reaches 0.1 at epoch {ts['to_01']} against {ta['to_01']}. Both reach 0.01 at about the same time, {ts['to_001']} and {ta['to_001']} epochs.

At each optimiser's fastest rate, {fa['lr']:g} for Adam and {fs['lr']:g} for SGD, **Adam is faster**: it reaches 0.01 in {fa['to_001']} epochs, SGD in {fs['to_001']}, and after ten epochs Adam's loss is {fa['e10']:.4f} against SGD's {fs['e10']:.4f}. That is the advantage Adam is known for, and it is real. It is also two epochs out of eight, and it exists only at the rate chosen for speed.""",
        figures=[figs["f0"]])

    post.add(
        "What tuning chose, and why",
        f"""Why did validation choose the slower rates? The table below gives test accuracy at every rate, averaged over the five splits of the speed study. The fastest rates are not worse: Adam's fastest scores {fa['acc']:.3f} against the best of its grid, {acc_a:.3f}, and SGD's fastest is the best of its grid, {acc_s:.3f}. The rates validation chose score {ta['acc']:.3f} and {ts['acc']:.3f}.

The tuning split could not see these differences. Its held-out fifth is {d['tuning']['adam']['n_val']} digits, so one digit is {1 / d['tuning']['adam']['n_val']:.4f} of accuracy, and on it Adam's rates of {d['tuning']['adam']['lrs'][1]:g} and {d['tuning']['adam']['lrs'][2]:g} tied, as did SGD's three largest rates. Ties went to the smaller rate, which is the cautious choice and the slower one. The speed comparison "at the tuned rates" is therefore a comparison of tie-breaks, and it favoured SGD by accident.

What does differ between the optimisers is how much the rate matters. Across Adam's grid of five rates, spanning a factor of 100, test accuracy stayed between {min(r['acc'] for r in res['speed']['adam']):.3f} and {acc_a:.3f}; across SGD's, between {min(r['acc'] for r in res['speed']['sgd']):.3f} and {acc_s:.3f}. Dividing by the gradient scale makes Adam less sensitive to its learning rate, which is a practical virtue, and a different claim from converging faster. Choi and colleagues made the general point: rankings of optimisers depend on how, and how much, each one is tuned.""",
        figures=[figs["f2"]])

    post.add(
        "What the sentence should say",
        """On these two problems, measured over twenty splits with both optimisers tuned, the sentence becomes: *at a learning rate chosen for speed, Adam reaches a low training loss a little sooner than SGD with momentum; tuned for validation accuracy, the two generalise equally well; Adam's accuracy depends less on the learning rate*.

Each part is narrower than the original, and each is measured. The generalisation gap Wilson and colleagues reported does not appear here, but their networks were far larger and their tasks harder, so this is not a refutation of their result; it is a reminder that "Adam generalises worse" is also a claim about particular settings. The honest summary is that the choice of optimiser matters less than tuning whichever one is chosen, and that a comparison which tunes one and not the other measures the tuning.""")

    post.add(
        "Where this breaks",
        """**Small networks, small data.** Two hidden layers on 900 digits or 2,000 simulated rows. On large language and vision models Adam (or AdamW) is the near-universal choice, partly because SGD is hard to tune at that scale and partly because adaptive scaling helps with very different gradient scales across layers, such as embeddings.

**A coarse grid.** Five rates per optimiser, spaced by about a factor of three. Both "fastest" and "tuned" are known only to that resolution, and validation accuracy on a fifth of one split (180 digits) has ties: Adam's 3e-4 and 1e-3 both scored the same, and the smaller was taken.

**One schedule.** Constant learning rates, no warm-up, no decay. Schedules change both speed and final accuracy, for both optimisers.

**No weight decay.** Adam's interaction with weight decay is its own subject (AdamW exists because of it), and was left out by giving neither optimiser any.""")

    post.add(
        "What to keep",
        f"""1. Tuned on validation data and run over twenty splits, Adam and SGD with momentum reached the same test accuracy: differences of {d['diff_mean']:+.4f} ± {d['diff_se']:.4f} on the digits and {g['diff_mean']:+.4f} ± {g['diff_se']:.4f} on a hard simulated task.

2. Speed belongs to the learning rate. At the rates validation chose, SGD's loss fell faster; at each optimiser's fastest rate, Adam reached a loss of 0.01 in {fa['to_001']} epochs and SGD in {fs['to_001']}.

3. The fastest rates scored as well as any on test digits; validation on 180 digits could not tell them apart from slower ones and chose the slower. Adam's accuracy varied less across its grid ({min(r['acc'] for r in res['speed']['adam']):.3f} to {acc_a:.3f}) than SGD's ({min(r['acc'] for r in res['speed']['sgd']):.3f} to {acc_s:.3f}).

4. Compare optimisers only with each tuned by the same procedure, and report what the rates were tuned for.""")

    post.add(
        "Exercise",
        """Take a model you train with Adam at its default learning rate. Train it with SGD with momentum 0.9 at five rates spaced by a factor of three, and with Adam at five rates around the default, all for the same number of epochs from the same initial weights.

Plot validation accuracy against rate for both. If the best SGD point is within noise of the best Adam point, your choice of optimiser is a choice about tuning effort, not about the model you end up with.""")

    post.add(
        "Next",
        """Both optimisers here trained networks whose layers pass activations straight to the next. Episode 24 adds the layer most networks put in between, batch normalisation, and measures the claim that it normalises each example.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
