"""Gradients 2: A Confident Attention Head Passes Almost No Gradient.

Episode 1 went looking for non-differentiability and found saturation instead.
This one is about saturation, which is smooth, and therefore not a hypothesis
failure at all -- the chain rule holds perfectly and the product it forms is
tiny.

Measured:

* The softmax Jacobian is `diag(p) - p p^T`. Its quadratic form is exactly a
  variance under p, so its norm is pinned within a factor of two by the largest
  probability alone: m(1-m) <= ||J|| <= 2m(1-m) for m >= 1/2, with the upper
  end attained exactly when the losing mass sits on one runner-up. Checked on
  thousands of random distributions: no violations, upper ratio 1.000000000.
* Which is independent of how many positions there are. A 64-way softmax at
  m = 0.99 has the gradient capacity of a two-way one.
* Its trace is exactly the complement of a collision probability, 1 - sum p^2,
  not the Shannon entropy.
* It is singular: `J 1 = 0`, so every row's logit gradient sums to exactly
  zero. One direction is deleted by the function itself.
* On the model the bound is a predictor, not a ceiling: the realised gain
  scales as 2m(1-m) to the power 1.03 with a constant near an eighth.
* And the finding. Layer 0 head 1 has a median maximum probability of 0.97,
  puts its argmax one position back on 99.8% of rows, and passes 5.9 times
  less routing gradient than the next-lowest head. It is also the head the model
  cannot do without: zeroing it costs 1.12 nats. Replacing it with a hard-wired
  shift-by-one permutation costs 0.0013.
* Across a fresh run the commitment takes about 200 steps and then holds for
  800 while the loss keeps falling. Its neighbour, which stopped at m = 0.76,
  kept its gradient and kept improving.

Run: `standarderror run gr102_saturation --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.calculus import saturation as sat
from standarderror.llm import tiny
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 9, 10)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Calculus for Language Models, Taught Through What Breaks"
SERIES_TAG = "Gradients"

BATCHES, BATCH_SIZE = 6, 8
SWEEP_DRAWS = 6_000
TRAJ_STEPS, TRAJ_EVERY = 1_200, 40
ABLATION_DRAWS = 8
#: The head this episode is about. Found by the survey, not chosen in advance.
LAYER, HEAD, CONTROL = 0, 1, 2


def compute() -> dict:
    sweep = sat.bound_sweep(draws=SWEEP_DRAWS, seed=3)
    shapes = sat.concentration(m_values=(0.9, 0.99, 0.999), sizes=(2, 8, 64))
    gain = sat.attention_gain(count=BATCHES, size=BATCH_SIZE, seed=0)
    trace = sat.trace_identity(count=3, size=4, seed=0)
    abl = sat.ablations(layer=LAYER, head=HEAD, control=CONTROL,
                        draws=ABLATION_DRAWS, count=10, size=16)
    traj = sat.trajectory(steps=TRAJ_STEPS, every=TRAJ_EVERY, seed=0)
    return {"sweep": sweep, "shapes": shapes, "gain": gain, "trace": trace,
            "ablations": abl, "trajectory": traj}


# ------------------------------------------------------------------ figures

def _traj_ratio(traj) -> float:
    """How much more routing gradient head (0,0) ends the fresh run with.

    Computed rather than described, because the fresh run and the committed
    checkpoint give different numbers for the same comparison and a caption
    that says "an order of magnitude" over one of them is not checkable.
    """
    hist = traj["history"]
    return (sat.series(hist, 0, 0, "median_gain")[-1]
            / sat.series(hist, 0, 1, "median_gain")[-1])


def figures(res: dict) -> dict:
    out: dict = {}
    sweep, gain = res["sweep"], res["gain"]
    traj, abl = res["trajectory"], res["ablations"]
    heads = {(h["layer"], h["head"]): h for h in gain["heads"]}
    target = heads[(LAYER, HEAD)]

    def bound(ax, m):
        x = np.linspace(0.5, 0.999, 300)
        keep = slice(None, None, max(1, sweep["rows"] // 3000))
        ax.scatter(sweep["max_p"][keep], sweep["spectral_norm"][keep],
                   s=3.0, alpha=0.28, color=m.series[0], linewidths=0,
                   label="random distributions, 2 to 64 wide")
        ax.plot(x, 2 * x * (1 - x), lw=2.2, color=m.series[2],
                label="upper bound  2m(1 - m)")
        ax.plot(x, x * (1 - x), lw=2.2, ls="--", color=m.series[1],
                label="lower bound  m(1 - m)")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 0.999)
        ax.set_ylim(4e-3, 0.7)
        ax.legend(frameon=False, fontsize=8.6, loc="lower left")

    out["f0"] = charts.diagram(
        bound,
        title="The gradient a softmax can pass, pinned by one number",
        subtitle=("The spectral norm of diag(p) - ppᵀ against the largest "
                  "probability, for random distributions over 2 to 64 "
                  "positions at random temperatures."),
        xlabel="largest probability in the row, m",
        ylabel="spectral norm of the Jacobian (log scale)",
        source="Measured; standarderror/calculus/saturation.py.",
        alt=("A cloud of points falling towards zero as the largest "
             "probability approaches one, tightly enclosed between two "
             "curves a factor of two apart."),
        caption=(f"{sweep['rows']:,} random distributions, none outside the "
                 f"band. The two curves are a factor of (1 + m)/m apart, "
                 f"which is under 2.01 above m = 0.99, so the largest "
                 f"probability fixes the gradient capacity to within a "
                 f"factor of two - and the **width of the row does not "
                 f"appear**. The highest measured point sits at "
                 f"{sweep['max_ratio_to_upper']:.9f} of the upper bound."),
        path=str(IMG / f"gr102-f0-bound.{EXT}"))[0]

    def path(ax, m):
        hist = traj["history"]
        steps = [s["step"] for s in hist]
        ax.plot(steps, sat.series(hist, 0, 1, "median_max_p"), lw=2.4,
                color=m.series[0], label="head (0,1)  confidence m")
        ax.plot(steps, sat.series(hist, 0, 0, "median_max_p"), lw=2.4,
                ls="--", color=m.series[1], label="head (0,0)  confidence m")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("median largest probability")
        ax.legend(frameon=False, fontsize=8.4, loc="center right")
        right = ax.twinx()
        right.plot(steps, sat.series(hist, 0, 1, "median_gain"), lw=1.8,
                   color=m.series[0], alpha=0.45,
                   label="head (0,1)  gradient, right axis")
        right.plot(steps, sat.series(hist, 0, 0, "median_gain"), lw=1.8,
                   ls="--", color=m.series[1], alpha=0.45,
                   label="head (0,0)  gradient, right axis")
        right.legend(frameon=False, fontsize=8.4, loc="lower right")
        right.set_ylim(0, 0.05)
        right.set_ylabel("routing gradient passed (faint)")

    out["f1"] = charts.diagram(
        path,
        title="Two hundred steps to commit, eight hundred with nothing to say",
        subtitle=(f"A fresh {TRAJ_STEPS}-step run of the same architecture. "
                  f"Bold lines are confidence, faint lines the gradient the "
                  f"softmax passes to the logits that choose where to look."),
        xlabel="training step", ylabel="",
        source="Measured; standarderror/calculus/saturation.py.",
        alt=("One curve rising steeply to nearly one and flattening while its "
             "faint partner collapses; a second curve levelling off near "
             "three quarters with its gradient intact."),
        caption=(f"Head (0,1) crosses from near-uniform to "
                 f"m = {sat.series(traj['history'], 0, 1, 'median_max_p')[10]:.2f} "
                 f"between steps 200 and 400, and its routing gradient "
                 f"collapses on "
                 f"the way through. It then holds that value for the "
                 f"remaining 800 steps while the loss falls from about 2.03 "
                 f"to {traj['history'][-1]['loss']:.2f}. Head (0,0) stopped "
                 f"at roughly three quarters, ended the run passing "
                 f"{_traj_ratio(traj):.0f} times more gradient than its "
                 f"neighbour, and was **still** moving at the end."),
        path=str(IMG / f"gr102-f1-trajectory.{EXT}"))[0]

    rows = []
    for L in range(4):
        for h in range(4):
            r = heads[(L, h)]
            rows.append([f"layer {L}, head {h}",
                         f"{r['median_max_p']:.3f}",
                         f"{100 * r['share_above_09']:.1f}%",
                         f"{r['median_gain']:.4f}",
                         f"{100 * r['previous_token_share']:.1f}%"])
    bold = {(4 * LAYER + HEAD, c) for c in range(5)}

    out["f2"] = charts.table_image(
        rows,
        header=["head", "median m", "rows past 0.9", "gradient passed",
                "argmax one back"],
        title="One of sixteen heads has stopped being able to change its mind",
        subtitle=(f"Measured over {gain['rows']:,} attention rows of held-out "
                  f"text on the committed checkpoint. Row 0 excluded "
                  f"throughout: a softmax over one element passes nothing."),
        source="Measured; standarderror/calculus/saturation.py.",
        alt=("A table of sixteen heads. One row, layer 0 head 1, has median "
             "confidence 0.97 and a gradient several times below every "
             "other row."),
        caption=(f"Two heads in layer 0 found the same rule - attend one "
                 f"position back - and only one of them committed to it. "
                 f"Head (0,1) is at m = {target['median_max_p']:.3f} and "
                 f"passes {target['median_gain']:.4f}; the next-lowest "
                 f"head in the model passes "
                 f"{sorted(h['median_gain'] for h in gain['heads'])[1]:.4f} "
                 f"and the softest "
                 f"{max(h['median_gain'] for h in gain['heads']):.4f}. The "
                 f"last column is why that matters: this is not a head that "
                 f"got stuck on nothing."),
        bold_cells=bold, align="lrrrr",
        path=str(IMG / f"gr102-f2-heads.{EXT}"))[0]

    def costs(ax, m):
        names = ["remove\nhead (0,1)", "replace it with\na fixed shift",
                 f"remove\nhead (0,{CONTROL})"]
        keys = ["zeroed", "shift_by_one", "control_zeroed"]
        vals = [abl[k]["delta"] for k in keys]
        errs = [abl[k]["sd"] for k in keys]
        ax.bar(range(3), vals, yerr=errs, capsize=4, width=0.5,
               color=[m.series[2], m.series[0], m.grid])
        for i, v in enumerate(vals):
            ax.annotate(f"{v:+.4f}", (i, v), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=9.2,
                        color=m.ink_secondary)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xticks(range(3))
        ax.set_xticklabels(names, fontsize=8.8)
        ax.set_ylim(0, 2.5)

    out["f3"] = charts.diagram(
        costs,
        title="Indispensable, and exactly reproducible by a constant",
        subtitle=(f"Change in validation loss, in nats, each paired against "
                  f"the unmodified model on the same held-out text over "
                  f"{ABLATION_DRAWS} draws."),
        xlabel="", ylabel="added validation loss (nats, log scale)",
        source="Measured; standarderror/calculus/saturation.py.",
        alt=("Three bars on a log scale: one tall at about 1.1, one very "
             "small at about 0.001, one small at about 0.003."),
        caption=(f"Removing head (0,1) costs "
                 f"{abl['zeroed']['delta']:.3f} nats, which is "
                 f"{100 * abl['zeroed']['delta'] / (tiny.UNIFORM_LOSS - tiny.VAL_LOSS):.0f}% "
                 f"of the distance this model travelled from a uniform "
                 f"guess. "
                 f"Replacing its attention with a fixed shift-by-one "
                 f"permutation matrix costs "
                 f"{abl['shift_by_one']['delta']:.4f} - about "
                 f"{abl['zeroed']['delta'] / abl['shift_by_one']['delta']:.0f} "
                 f"times less. The head is essential and it is also, by now, "
                 f"a constant."),
        path=str(IMG / f"gr102-f3-ablations.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    gain, abl = res["gain"], res["ablations"]
    traj = res["trajectory"]
    heads = {(h["layer"], h["head"]): h for h in gain["heads"]}
    target = heads[(LAYER, HEAD)]

    def capacity(panel, m):
        p = np.linspace(0.02, 0.995, 160)
        panel.plot(p, 2 * p * (1 - p), lw=2.6, color=m.series[0])
        panel.axvline(target["median_max_p"], lw=1.8, ls=":",
                      color=m.series[2])
        panel.set_xlim(0, 1.02)
        panel.set_ylim(-0.03, 0.58)

    def bars(panel, m):
        vals = sorted(h["median_gain"] for h in gain["heads"])
        colours = [m.series[2] if v == target["median_gain"] else m.grid
                   for v in vals]
        panel.bar(range(len(vals)), vals, color=colours, width=0.72)
        panel.set_xlim(-0.8, len(vals) - 0.2)
        panel.set_ylim(0, 0.062)

    def commit(panel, m):
        hist = traj["history"]
        steps = [s["step"] for s in hist]
        panel.plot(steps, sat.series(hist, 0, 1, "median_max_p"), lw=2.6,
                   color=m.series[0])
        panel.plot(steps, [20 * g for g in
                           sat.series(hist, 0, 1, "median_gain")],
                   lw=2.2, color=m.series[2])
        panel.set_xlim(0, steps[-1])
        panel.set_ylim(-0.05, 1.12)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=2,
        headline="The head this model cannot lose is the one that stopped learning",
        panels=[(capacity, "2m(1-m)", "gradient capacity"),
                (bars, f"{target['median_gain']:.4f}", "one head of sixteen"),
                (commit, f"{abl['zeroed']['delta']:.2f}", "nats to remove it")],
        note=(f"A softmax passes gradient in proportion to 2m(1 - m), where m "
              f"is its largest probability. Layer 0 head 1 sits at "
              f"m = {target['median_max_p']:.2f}, attends one position back on "
              f"{100 * target['previous_token_share']:.1f}% of rows, and is "
              f"reproduced by a fixed permutation for "
              f"{abl['shift_by_one']['delta']:.4f} nats."),
        alt=("Three hand-drawn frames: a hump peaking in the middle with a "
             "dotted line far to its right; sixteen bars with the shortest "
             "highlighted; and a rising curve crossing a falling one."),
        path=str(IMG / f"gr102-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["bound"] = s.run("""
        import numpy as np
        from standarderror.calculus import saturation as sat

        # Two rows with the same largest probability and very different
        # widths. The bound depends on m alone, so it cannot tell them apart.
        wide = np.r_[0.99, np.full(63, 0.01 / 63)]
        narrow = np.array([0.99, 0.01])
        for name, p in (("64 positions", wide), ("2 positions", narrow)):
            g = sat.spectrum(p)
            print(f"{name:>13}   ||J|| = {g['spectral_norm']:.6f}   "
                  f"bounds {g['lower_bound']:.6f} .. {g['upper_bound']:.6f}")
        print()
        g = sat.spectrum(narrow)
        print(f"trace {g['trace']:.9f}  ==  1 - sum p^2 = "
              f"{g['collision']:.9f}")
        print(f"smallest eigenvalue {g['smallest']:+.1e}   "
              f"(the constant direction, deleted)")
    """, expect=["64 positions"])

    out["heads"] = s.run(f"""
        # Sixteen heads of the committed model, ranked by the gradient their
        # softmax lets through to the logits that decide where to look.
        g = sat.attention_gain(count={BATCHES}, size={BATCH_SIZE})
        print(f"rows {{g['rows']:,}}   "
              f"logit gradients sum to {{g['row_sum_median']:.1e}} of the "
              f"row L1")
        print(f"realised gain scales as 2m(1-m) to the power "
              f"{{g['log_slope']:.2f}}, constant {{g['log_constant']:.3f}}")
        print()
        for h in sorted(g["heads"], key=lambda r: r["median_gain"])[:3]:
            print(f"  layer {{h['layer']}} head {{h['head']}}   "
                  f"m {{h['median_max_p']:.3f}}   "
                  f"gain {{h['median_gain']:.4f}}   "
                  f"argmax one back on {{h['previous_token_share']:.1%}}")
    """, expect=["rows"])

    out["ablate"] = s.run(f"""
        # So the head barely learns any more. Does the model need it?
        a = sat.ablations(layer={LAYER}, head={HEAD}, control={CONTROL},
                          draws={ABLATION_DRAWS}, count=10, size=16)
        print(f"baseline validation loss        {{a['baseline']:.4f}}")
        for k, label in (("zeroed", "zero the head"),
                         ("shift_by_one", "force a hard shift-by-one"),
                         ("control_zeroed", "zero head (0,{CONTROL})")):
            print(f"  {{label:<26}} {{a[k]['delta']:+.4f}} nats "
                  f"(sd {{a[k]['sd']:.4f}})")
    """, expect=["baseline validation loss"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    sweep, gain = res["sweep"], res["gain"]
    abl, traj = res["ablations"], res["trajectory"]
    heads = {(h["layer"], h["head"]): h for h in gain["heads"]}
    target = heads[(LAYER, HEAD)]

    # The spine, asserted rather than trusted.
    assert sweep["violations_low"] == 0 and sweep["violations_high"] == 0
    assert abs(sweep["max_ratio_to_upper"] - 1.0) < 1e-6
    assert res["trace"]["max_abs_error"] < 1e-6
    assert gain["violations_of_bound"] == 0
    assert abs(gain["log_slope"] - 1.0) < 0.1
    assert gain["row_sum_median"] < 1e-6
    assert target["median_max_p"] > 0.9
    assert target["previous_token_share"] > 0.95
    others = [h["median_gain"] for k, h in heads.items() if k != (LAYER, HEAD)]
    assert target["median_gain"] < min(others) / 3
    assert abl["zeroed"]["delta"] > 0.8
    assert abl["shift_by_one"]["delta"] < 0.02
    assert abl["zeroed"]["delta"] > 20 * abl["control_zeroed"]["delta"]
    late = sat.series(traj["history"], 0, 1, "median_max_p")[-10:]
    assert min(late) > 0.9 and max(late) - min(late) < 0.02

    post = Post(
        title=(f"{SERIES_TAG} 2: A Confident Attention Head Passes Almost No "
               f"Gradient"),
        slug="gradients-2-saturation",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=2,
        date=POST_DATE,
        # Every number is either exact algebra or a count on a fixed model.
        # There is no prediction and therefore nothing to beat.
        requires_baseline=False,
        subtitle=("The softmax Jacobian's norm is fixed within a factor of "
                  "two by its largest probability alone, no matter how many "
                  "positions it has. So one head of this model has committed "
                  "hard enough that the gradient which could change its mind "
                  "is gone - and it is the head the model cannot do without."),
        summary=(
            f"The softmax Jacobian is diag(p) minus p pᵀ, its quadratic form "
            "is exactly a variance under p, and so its spectral norm is "
            "trapped between m(1 - m) and 2m(1 - m) where m is the largest "
            f"probability - checked on {sweep['rows']:,} random distributions with "
            f"no violations and the upper end attained to nine digits. The "
            f"width "
            "of the row does not appear: a 64-way softmax at m = 0.99 has the "
            "gradient capacity of a two-way one. On the committed model the "
            "bound turns out to be a predictor rather than a ceiling, the "
            "realised gain scaling as 2m(1 - m) to the power 1.03. Then the "
            "finding: layer 0 head 1 sits at m = 0.97, puts its argmax one "
            "position back on 99.8% of rows, and passes 5.9 times less "
            "routing gradient than the next-lowest head - while being the "
            "head the model cannot lose, since zeroing it costs 1.12 nats and "
            "replacing it with a fixed shift-by-one permutation costs 0.0013. "
            "Across a fresh run the commitment takes 200 steps and then holds "
            "for 800 more while the loss keeps falling."),
        tags=["deep-learning", "machine-learning", "llm", "pytorch",
              "mathematics", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. The bound is checked on distributions drawn at "
            "build time; every model number is a count or a loss on held-out "
            "text from the corpus the model was trained on, and no values "
            "from that text are published.",
            f"The model: an {tiny.load()['parameters']:,}-parameter "
            f"character-level transformer trained for this series - four "
            f"blocks, four heads, width {tiny.WIDTH}, context {tiny.BLOCK}, "
            f"validation loss {tiny.VAL_LOSS} against a uniform-guess "
            f"{tiny.UNIFORM_LOSS:.3f}. Weights, training script and a "
            f"verified sha256 are committed: `standarderror/llm/tiny.py`, "
            f"`scripts/train_tiny.py`, `data/tiny_gpt/`.",
            "Machinery: `standarderror/calculus/saturation.py`, tested in "
            "`tests/test_saturation.py`, which pins the algebra exactly and "
            "the model findings as inequalities.",
            "Where this stops: Bridle, \"Probabilistic interpretation of "
            "feedforward classification network outputs, with relationships "
            "to statistical pattern recognition\" (1990), for the softmax and "
            "its Jacobian; Elhage et al., \"A mathematical framework for "
            "transformer circuits\", *Transformer Circuits Thread* (2021), "
            "for previous-token heads and why a first-layer head becomes one; "
            "Vaswani et al., \"Attention is all you need\", *NeurIPS* (2017), "
            "for the 1/sqrt(d) scale, which exists precisely to keep m away "
            "from 1 at initialisation.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, "
                           "torch=2.14.0, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "model": ("the committed checkpoint, hash-verified on load, for "
                      "every static number; the trajectory is a separate "
                      "fresh run and is labelled as one"),
            "determinism": (f"the survey draws {BATCHES} batches of "
                            f"{BATCH_SIZE} held-out sequences from a fixed "
                            f"seed; the ablations are paired against the "
                            f"unmodified model on the same text over "
                            f"{ABLATION_DRAWS} draws; the trajectory is a "
                            f"fresh {TRAJ_STEPS}-step run under "
                            f"torch.manual_seed(0), reproducible on this "
                            f"platform and not guaranteed across torch "
                            f"versions"),
        },
    )
    post.hero = figs["hero"]
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    sweep, gain, trace = res["sweep"], res["gain"], res["trace"]
    abl, traj = res["ablations"], res["trajectory"]
    heads = {(h["layer"], h["head"]): h for h in gain["heads"]}
    target = heads[(LAYER, HEAD)]
    soft = max(gain["heads"], key=lambda h: h["median_gain"])
    # The runner-up in the gain ranking, which is low for the opposite reason.
    diffuse = sorted(gain["heads"], key=lambda h: h["median_gain"])[1]
    ratio = soft["median_gain"] / target["median_gain"]
    # Against the *nearest* head, which is the conservative comparison and
    # the only one that supports a claim about every other head.
    nearest = min(h["median_gain"] for k, h in heads.items()
                  if k != (LAYER, HEAD))
    margin = nearest / target["median_gain"]
    hist = traj["history"]
    conf = sat.series(hist, 0, 1, "median_max_p")
    pv = sat.series(hist, 0, 1, "previous_token_share")
    gainseries = sat.series(hist, 0, 1, "median_gain")
    conf0 = sat.series(hist, 0, 0, "median_max_p")
    pv0 = sat.series(hist, 0, 0, "previous_token_share")

    post.add(
        "The gradient is a variance",
        f"""Episode 1 asked where a transformer lands on a point that has no derivative, and the answer was almost nowhere. What it found instead was confidence: {gain['share_above_09']:.1%} of this model's attention rows put more than 0.9 of their mass on a single position. That is not a kink. The softmax is smooth there, the chain rule's hypotheses hold exactly, and the number the chain rule produces is nearly zero anyway.

Here is why, in one line. For `p = softmax(z)` the Jacobian is `diag(p) - p p^T`, and for any vector *u*

$$
u^{{\\top}} \\big( \\operatorname{{diag}}(p) - p p^{{\\top}} \\big) u
  \\;=\\; \\sum_i p_i u_i^2 - \\Big( \\sum_i p_i u_i \\Big)^2
  \\;=\\; \\operatorname{{Var}}_p(u).
$$

The quadratic form of the softmax Jacobian is a **variance under its own output**. So the largest amount by which the Jacobian can stretch any unit vector is the largest variance a unit vector can have under *p*, and a distribution that has nearly all its mass on one atom has nearly no variance to give. Confidence and gradient are not two facts about a softmax. They are the same fact.""")

    post.add(
        "How tight that is",
        """Tight enough to be a number rather than an intuition. Write *m* for the largest probability in the row. Then for *m* at least one half,

$$
m(1-m) \\;\\le\\; \\big\\| \\operatorname{diag}(p) - p p^{\\top} \\big\\|_2
  \\;\\le\\; 2\\,m(1-m).
$$

Both halves are short. The lower bound is one diagonal entry: a positive semi-definite matrix has its largest eigenvalue at least as big as any diagonal entry, and the winner's diagonal entry is exactly *m*(1 − *m*). The upper bound takes *c* = *u* at the winning coordinate in the inequality Var ≤ E(*u* − *c*)², which leaves a sum over the losers and bounds each squared difference by 2, giving 2(1 − *m*); the sharper constant comes from the losing mass being able to sit on at most one runner-up.

What that says is stronger than it first looks. The bound has no *n* in it. **How many positions the row has does not appear.**""")

    post.add(
        "",
        f"""{snip['bound'].markdown()}

A 64-way softmax at *m* = 0.99 and a two-way softmax at *m* = 0.99 have the same gradient capacity, to within the same factor of two. Widening the context window does not give a saturated head more room to change its mind; it gives it more positions to be equally uninterested in.

The other two lines are worth a sentence each. The trace of the Jacobian is exactly 1 − Σ*p*², the complement of the probability that two independent draws from *p* agree — not the Shannon entropy, which is the quantity usually reached for when someone says the attention is diffuse, and which is not equal to it even up to a constant. Checked against the model's own rows, the identity holds to {trace['max_abs_error']:.1e}. And the Jacobian annihilates the all-ones vector exactly - the snippet's smallest eigenvalue is what a float eigensolver prints for zero - because adding a constant to every logit does not move the softmax. That is not a numerical accident: it means **the logit gradients inside any one attention row sum to exactly zero**, always, in every head of every model. Measured on a real backward pass, that sum is {gain['row_sum_median']:.1e} of the row's own scale, which is float32 saying zero. One direction of the gradient is deleted by the function itself, and that turns out to be the theme of episode 3.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "A bound that predicts rather than caps",
        f"""A bound on the norm is a bound on the worst case, and the gradient a real backward pass delivers is not the worst case — it is whatever direction the loss happens to want, which is generally not the top eigenvector. So the honest question is not whether the bound holds but whether it is informative.

It is, more than I expected. Over {gain['rows']:,} attention rows of held-out text, the realised gain — the norm of the logit gradient divided by the norm of the incoming gradient — scales as 2*m*(1 − *m*) to the power {gain['log_slope']:.2f}, with a constant of {gain['log_constant']:.3f}. Slope one, in other words, with about an eighth of the available capacity actually used, and that fraction roughly constant across confidence levels. The bound is a proportional predictor and not merely a ceiling.

Which lets the whole thing be restated as a rate. If routing changes at a speed proportional to 2*m*(1 − *m*), then relative to a head at *m* = 0.5 a head at *m* = 0.98 needs about {sat.slowdown(0.98):.0f} times as many steps to make the same change to where it looks, and a head at 0.9999 — the largest probability episode 1 found in this model — needs about {sat.slowdown(0.9999):,.0f} times as many. Saturation does not forbid a head from changing its mind. It multiplies the time.""")

    post.add(
        "One head of sixteen",
        """So who is saturated? Ask every head.""")

    post.add(
        "",
        f"""{snip['heads'].markdown()}

Layer {LAYER}, head {HEAD} — median confidence {target['median_max_p']:.3f}, {target['share_above_09']:.1%} of its rows past 0.9, and a routing gradient of {target['median_gain']:.4f} against {nearest:.4f} for the next-lowest head and {soft['median_gain']:.4f} for the softest. A factor of {margin:.1f} even against its nearest rival, and {ratio:.1f} against the far end.

The second row of that ranking is worth a pause, because it is low for the opposite reason. Head (0,{diffuse['head']}) sits at *m* = {diffuse['median_max_p']:.3f} — the most **diffuse** head in the model, not the most confident — and 2*m*(1 − *m*) is {2 * diffuse['median_max_p'] * (1 - diffuse['median_max_p']):.3f} there. The hump falls away on both sides, and this model has heads on both of them: near-uniform attention has almost nothing to differentiate either. So the comparison that carries weight is not against the whole ranking but against the heads on the saturated side, and on that side head ({LAYER},{HEAD}) is alone.

And the last column says what it committed to: on {100 * target['previous_token_share']:.1f}% of rows its argmax is exactly one position back. It is a previous-token head, which is the most-documented circuit component there is and exactly the thing a first layer is expected to build. This is not a head that got stuck on noise.""",
        level=3,
        figures=[figs["f2"]])

    post.add(
        "The part that changes the story",
        """At this point the draft I was writing said: a saturated head has stopped learning, which is a problem. Two measurements later it does not say that.

The first is an ablation. If the head has stopped learning, how much does the model depend on what it learned?""")

    post.add(
        "",
        f"""{snip['ablate'].markdown()}

Removing it costs {abl['zeroed']['delta']:.4f} nats. For scale, the whole distance this model travelled from a uniform guess is {tiny.UNIFORM_LOSS - tiny.VAL_LOSS:.2f} nats, so removing one head of sixteen undoes {100 * abl['zeroed']['delta'] / (tiny.UNIFORM_LOSS - tiny.VAL_LOSS):.0f}% of that distance. Ablation deltas do not add up across a network, so that is a scale rather than a decomposition. Zeroing a different head in the same layer costs {abl['control_zeroed']['delta']:.4f}.

The second measurement is the one that reframes it. Replace the head's attention — not its values, just the softmax output that decides where to look — with a hard-wired permutation matrix that always attends exactly one position back. No learning, no logits, a constant.

{abl['shift_by_one']['delta']:.4f} nats. About {abl['zeroed']['delta'] / abl['shift_by_one']['delta']:.0f} times less than removing it, and inside the draw-to-draw noise of the loss itself.

So the head is not failing to learn something it needs. It is **already a constant**, functionally, and the gradient that could move it is gone because there is nowhere left for it to go. Saturation here is not a pathology. It is what commitment looks like from the inside of a Jacobian.""",
        level=3,
        figures=[figs["f3"]])

    post.add(
        "When the commitment happened",
        f"""One checkpoint cannot separate "the gradient vanished and froze the head" from "the head arrived at the right answer and the gradient correctly went quiet". For that you need the path, so here is a fresh {TRAJ_STEPS}-step run of the same architecture under the same seed, with every head's confidence and routing gradient logged every {TRAJ_EVERY} steps.

The shape is unambiguous. For the first {hist[3]['step']} steps head (0,1) sits near uniform: confidence {conf[3]:.2f}, and its argmax one position back on {100 * pv[3]:.0f}% of rows. Between steps {hist[5]['step']} and {hist[10]['step']} it crosses — confidence {conf[5]:.2f} to {conf[10]:.2f}, previous-token share {pv[5]:.2f} to {pv[10]:.2f} — and its routing gradient **rises** on the way in, peaking at {max(gainseries):.4f} around step {hist[gainseries.index(max(gainseries))]['step']}, before collapsing to {gainseries[-1]:.4f}.

That peak is worth a caveat, because it is not where the capacity peaks. 2*m*(1 − *m*) is largest at *m* = 0.5, and the realised gradient turns over at *m* = {conf[gainseries.index(max(gainseries))]:.2f} — earlier. So the incoming gradient is shrinking at the same time the capacity is growing, and what a head actually receives is the product of the two. The bound governs the ceiling and the ceiling's shape; it does not govern when the loss stops asking.

After that it does not move again. {TRAJ_STEPS - hist[10]['step']} more steps within {max(conf[10:]) - min(conf[10:]):.3f} of the same confidence, while the loss falls from {hist[10]['loss']:.2f} to {hist[-1]['loss']:.2f}. The model kept learning for two thirds of training with that head's routing frozen — and it needed the head the whole time.""")

    post.add(
        "",
        f"""The control is in the same layer. Head (0,0) also found the previous-token rule — by the end of the run its argmax is one position back on {100 * pv0[-1]:.1f}% of rows — and it stopped at *m* = {conf0[-1]:.2f} rather than {conf[-1]:.2f}. It ended with {_traj_ratio(traj):.0f} times more routing gradient, and over the last {TRAJ_STEPS - hist[10]['step']} steps its previous-token share was still climbing, from {pv0[10]:.3f} to {pv0[-1]:.3f}, while its neighbour's had been pinned at {pv[-1]:.3f} since step {hist[10]['step']}. Two heads, one rule, one crossed the hump and one did not.

Which is the honest reading of the whole episode, and it is not the one I drafted. The mechanism by which a softmax makes a decision — pushing mass onto one option — is the same mechanism that removes the gradient which could revise it. That is not a flaw to be fixed; it is what deciding *is*, when your only instrument is a derivative. It is also why the `1/sqrt(d)` in front of the attention logits exists: not for numerical stability, but to keep *m* on the near side of the hump at initialisation, so that a head has some gradient with which to choose before it can commit.

Where it would be a flaw is a head that crosses early onto a rule that is merely adequate. This model does not give me that case, and I am not going to manufacture one. But the rate calculation above says what it would cost: a step of routing change at *m* = 0.98 arrives about {sat.slowdown(0.98):.0f} times slower than the same step at *m* = 0.5, so a head that commits badly at step 300 is not going to think better of it by step 3,000.""",
        level=3,
        figures=[figs["f1"]])

    post.add(
        "What to keep",
        f"""1. The softmax Jacobian's quadratic form is a variance under its own output. Everything else here follows from that sentence.

2. So the gradient it passes is trapped between *m*(1 − *m*) and 2*m*(1 − *m*), where *m* is the largest probability — checked on {sweep['rows']:,} random distributions with no violations and the upper end attained to {sweep['max_ratio_to_upper']:.9f}. The number of positions does not enter.

3. The trace is exactly 1 − Σ*p*², a collision probability, and not the entropy. The Jacobian annihilates the all-ones vector exactly, so **the logit gradients within an attention row sum to zero** in every softmax that has ever been trained.

4. On this model the bound predicts rather than caps: the realised gain goes as 2*m*(1 − *m*) to the power {gain['log_slope']:.2f} with a constant near an eighth. Read as a rate, a head at *m* = 0.98 revises its routing about {sat.slowdown(0.98):.0f} times slower than one at 0.5.

5. Layer {LAYER} head {HEAD} is at *m* = {target['median_max_p']:.3f}, is a previous-token head on {100 * target['previous_token_share']:.1f}% of rows, and passes {margin:.1f} times less routing gradient than the next-lowest head in the model, and {ratio:.1f} times less than the softest.

6. It is also the head the model cannot lose — {abl['zeroed']['delta']:.3f} nats — and it is reproduced by a fixed permutation matrix for {abl['shift_by_one']['delta']:.4f}. Both of those at once is the finding.

7. So saturation is not a failure mode here. It is the shape of a decision, and the `1/sqrt(d)` scale exists to postpone it.""")

    post.add(
        "Exercise",
        """Take any attention implementation you have and log two numbers per head per step: the median largest probability, and the norm of the gradient reaching the attention logits divided by the norm of the gradient reaching the probabilities. Plot the second against 2m(1 − m) on log axes. If the slope is not close to one, something between your softmax and your loss is not what you think it is, and finding out which thing is a better afternoon than reading about it.

Then find your most confident head and replace its attention with the best fixed pattern you can guess — a shift, a first-token sink, a fixed local window. If the loss barely moves, you have learned that a parameter count and a functional degree of freedom are different things, and you have learned it about your own model rather than about the 816,128-parameter one here.

The uncomfortable version: do it early in training instead. A head you can replace with a constant at step 300 is a head that will be a constant at step 30,000, and the interesting question is whether it picked the constant you would have picked.""")

    return post


if __name__ == "__main__":
    print(build().markdown()[:2000])
