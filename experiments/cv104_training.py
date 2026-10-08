"""Coverage 4: Overconfidence Arrives Before the Model Stops Improving.

The fourth episode of the uncertainty series. Episode 3 found that the
committed model needs a fitted temperature of only 1.10, and asked whether
that is a property of the architecture or of the moment training stopped. The
syllabus promised an answer of the second kind: overconfidence arrives when
validation loss stops improving. Measured on three training runs of the same
architecture with checkpoints, it arrives well before that.

Measured:

* A rerun of the committed recipe lands at validation NLL 1.516 and fitted
  T 1.088 against the committed model's 1.525 and 1.094 on the same rows.
* Early in every run the model is slightly *under*confident. Overconfidence
  begins after one to ten passes over the training text, and grows with the
  gap between training and validation loss.
* At the validation-NLL minimum the model is already overconfident: fitted
  T 1.20 for the long run, 1.38 for the small-data run, whose ECE there is
  0.105.
* On its own training text the model stays calibrated or slightly
  underconfident throughout, fitted T never above 1.03. Overconfidence is
  what memorisation looks like from held-out text.
* Past the minimum, the small-data run's validation NLL climbs to 4.68 --
  worse than a uniform guess -- while accuracy stays near 0.41. One
  temperature brings it back to 2.43: the blow-up is almost all confidence.

Run: `standarderror run cv104_training --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.llm import tiny
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.uncertainty import trajectory as tj
from standarderror.viz import charts

POST_DATE = date(2026, 10, 8)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Uncertainty for Language Models, Taught Through What Breaks"
SERIES_TAG = "Coverage"

ARMS = ("recipe", "long", "small")
LABEL = {"recipe": "recipe", "long": "long", "small": "small data",
         "small_seed1": "small data, seed 1"}


def _passes(rows, cfg, train_chars, batch, block):
    n = int(train_chars * cfg["share"])
    return [r["step"] * batch * block / n for r in rows]


def compute() -> dict:
    traj = tj.trajectory()
    cfg = traj["config"]
    arms = {}
    for name, rows in traj["arms"].items():
        rows = [r for r in rows if r["step"] > 0]   # step 0 is uniform
        passes = _passes(rows, cfg["arms"][name], cfg["train_chars"],
                         cfg["batch"], tiny.BLOCK)
        for r, p in zip(rows, passes):
            r["passes"] = p
        best = min(rows, key=lambda r: r["val"]["nll"])
        scaled = min(rows, key=lambda r: r["val"]["nll_at_fit"])
        arms[name] = {
            "rows": rows, "best": best, "final": rows[-1],
            "scaled_best": scaled,
            "acc_best": max(rows, key=lambda r: r["val"]["accuracy"]),
            "onset": next(r for r in rows if r["val"]["fitted_t"] >= 1.05),
            "train_t_max": max(r["train"]["fitted_t"] for r in rows),
            "train_t_min": min(r["train"]["fitted_t"] for r in rows),
            "early_t_min": min(r["val"]["fitted_t"] for r in rows[:4]),
        }
    committed = tj.evaluate(tiny.load()["model"])
    return {"traj": traj, "cfg": cfg, "arms": arms, "committed": committed}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    arms = res["arms"]
    colour = {"recipe": 1, "long": 0, "small": 2}

    def onset(ax, m):
        for name in ARMS:
            a = arms[name]
            xs = [r["passes"] for r in a["rows"]]
            c = m.series[colour[name]]
            ax.plot(xs, [r["val"]["fitted_t"] for r in a["rows"]], lw=2.4,
                    color=c, label=LABEL[name])
            ax.plot(xs, [r["train"]["fitted_t"] for r in a["rows"]], lw=1.4,
                    ls=":", color=c)
            b = a["best"]
            ax.plot([b["passes"]], [b["val"]["fitted_t"]], marker="o", ms=9,
                    color=c, markeredgecolor=m.ink, zorder=5)
        ax.plot([], [], lw=1.4, ls=":", color=m.ink_secondary,
                label="own training text")
        ax.plot([], [], marker="o", ms=8, ls="", color=m.ink_secondary,
                label="lowest val loss")
        ax.axhline(1.0, lw=1, color=m.ink_secondary)
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8.0, loc="upper left")

    s, lg = arms["small"]["best"], arms["long"]["best"]
    out["f0"] = charts.diagram(
        onset,
        title="The model is overconfident before it stops improving",
        subtitle=("Temperature an NLL fit would choose on held-out text "
                  "(solid) and on each run's own training text (dotted), "
                  "at every checkpoint. Above 1 means overconfident."),
        xlabel="passes over the training text",
        ylabel="fitted temperature",
        source="Measured; data/coverage4/trajectory.json.",
        alt=("Three solid curves that start just under 1 and rise, the "
             "small-data one steeply to above 3; each has a dot well above 1 "
             "where its validation loss bottoms out; dotted curves for the "
             "training text stay near 1."),
        caption=(f"At its lowest validation loss the long run already needs "
                 f"T = {lg['val']['fitted_t']:.2f} and the small-data run "
                 f"T = {s['val']['fitted_t']:.2f}. **On their own training "
                 f"text, every run stays within 2% of 1.**"),
        path=str(IMG / f"cv104-f0-onset.{EXT}"))[0]

    a = arms["small"]
    xs = [r["step"] for r in a["rows"]]

    def blowup(ax, m):
        ax.plot(xs, [r["val"]["nll"] for r in a["rows"]], lw=2.6,
                color=m.series[2], label="validation NLL")
        ax.plot(xs, [r["val"]["nll_at_fit"] for r in a["rows"]], lw=2.2,
                ls="--", color=m.series[0],
                label="validation NLL after one temperature")
        ax.plot(xs, [r["train"]["nll"] for r in a["rows"]], lw=1.8,
                color=m.series[1], label="NLL on its own training text")
        ax.axhline(tiny.UNIFORM_LOSS, lw=1, ls=":", color=m.ink_secondary)
        ax.annotate(f"uniform guess, {tiny.UNIFORM_LOSS:.2f}",
                    (xs[-1], tiny.UNIFORM_LOSS + 0.08), ha="right",
                    fontsize=8.6, color=m.ink_secondary)
        ax.axvline(a["best"]["step"], lw=1.2, ls=":", color=m.ink_secondary)
        ax.legend(frameon=False, fontsize=8.4, loc="center right")

    f = a["final"]
    out["f1"] = charts.diagram(
        blowup,
        title="Past the minimum, the loss that explodes is confidence",
        subtitle=(f"The small-data run. Accuracy at the end is "
                  f"{f['val']['accuracy']:.3f}, against "
                  f"{a['best']['val']['accuracy']:.3f} at the dotted "
                  f"minimum."),
        xlabel="training step",
        ylabel="NLL (nats per character)",
        source="Measured; data/coverage4/trajectory.json.",
        alt=("A validation loss that dips and then climbs past the uniform-"
             "guess line, a dashed temperature-scaled loss that stays low, "
             "and a training loss falling towards zero."),
        caption=(f"Validation NLL ends at {f['val']['nll']:.2f}, worse than "
                 f"guessing uniformly. One temperature, "
                 f"T = {f['val']['fitted_t']:.2f}, brings it to "
                 f"**{f['val']['nll_at_fit']:.2f}**. The answers barely "
                 f"moved; the confidence in them did."),
        path=str(IMG / f"cv104-f1-blowup.{EXT}"))[0]

    rows_t = []
    for name in ARMS:
        a = arms[name]
        for tag, r in (("lowest val NLL", a["best"]), ("end", a["final"])):
            if tag == "end" and r is a["best"]:
                continue
            rows_t.append([name, tag, f"{r['step']:,}", f"{r['passes']:.1f}",
                           f"{r['val']['nll']:.3f}",
                           f"{r['val']['accuracy']:.3f}",
                           f"{r['val']['ece']:.3f}",
                           f"{r['val']['fitted_t']:.2f}",
                           f"{r['val']['nll_at_fit']:.3f}"])
    bold = {(i, c) for i, r in enumerate(rows_t) for c in range(9)
            if r[0] == "small" and r[1] == "lowest val NLL"}
    out["f2"] = charts.table_image(
        rows_t,
        header=["run", "checkpoint", "step", "passes", "val NLL",
                "accuracy", "ECE", "fitted T", "NLL at T"],
        title="Where each run stops improving, and what it costs",
        subtitle=(f"{res['cfg']['eval_rows']:,} validation predictions at "
                  f"every checkpoint. The recipe run is still improving when "
                  f"it ends."),
        source="Measured; data/coverage4/trajectory.json.",
        alt=("A table of five checkpoints: fitted temperature above 1 at "
             "every one, 1.38 at the small-data run's best and 3.31 at its "
             "end."),
        caption=("The **bold** row is the checkpoint early stopping would "
                 "keep, and it already needs a temperature of 1.38."),
        bold_cells=bold, align="llrrrrrrr",
        path=str(IMG / f"cv104-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    arms = res["arms"]
    a = arms["small"]

    def rise(panel, m):
        for name, c in (("long", 0), ("small", 2)):
            rows = arms[name]["rows"]
            panel.plot([np.log(r["passes"]) for r in rows],
                       [r["val"]["fitted_t"] for r in rows], lw=2.8,
                       color=m.series[c])

    def own(panel, m):
        rows = a["rows"]
        panel.plot([r["step"] for r in rows],
                   [r["train"]["fitted_t"] for r in rows], lw=2.8,
                   color=m.series[1])
        panel.set_ylim(0.0, 3.5)

    def blow(panel, m):
        rows = a["rows"]
        panel.plot([r["step"] for r in rows],
                   [r["val"]["nll"] for r in rows], lw=2.8,
                   color=m.series[2])
        panel.plot([r["step"] for r in rows],
                   [r["val"]["nll_at_fit"] for r in rows], lw=2.4,
                   color=m.series[0])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=4,
        headline="Overconfident before it stops improving",
        panels=[(rise, f"T = {a['best']['val']['fitted_t']:.2f}",
                 "at the best checkpoint"),
                (own, f"<= {max(arms[n]['train_t_max'] for n in ARMS):.2f}",
                 "on its own training text"),
                (blow, f"{a['final']['val']['nll']:.2f} -> "
                       f"{a['final']['val']['nll_at_fit']:.2f}",
                 "NLL, one temperature")],
        note=("Three training runs of the same 816,128-parameter model. "
              "Overconfidence on held-out text starts long before validation "
              "loss bottoms out, grows as training loss pulls away from it, "
              "and barely registers on the training text itself."),
        alt=("Three hand-drawn frames: two rising curves; a flat line low "
             "in its frame; and a climbing curve above a flat one."),
        path=str(IMG / f"cv104-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["recipe"] = s.run("""
        from standarderror.llm import tiny
        from standarderror.uncertainty import trajectory as tj

        committed = tj.evaluate(tiny.load()["model"])["val"]
        rerun = tj.trajectory()["arms"]["recipe"][-1]["val"]
        for name, r in (("committed", committed), ("rerun", rerun)):
            print(f"{name:9s}  val NLL {r['nll']:.4f}  accuracy {r['accuracy']:.4f}"
                  f"  fitted T {r['fitted_t']:.3f}")
        """)
    out["onset"] = s.run("""
        traj = tj.trajectory()
        for arm in ("recipe", "long", "small"):
            rows = [r for r in traj["arms"][arm] if r["step"] > 0]
            first = next(r for r in rows if r["val"]["fitted_t"] >= 1.05)
            best = tj.turn(rows)
            at = next(r for r in rows if r["step"] == best["step"])
            print(f"{arm:6s}  T >= 1.05 from step {first['step']:>5,}   "
                  f"lowest val NLL at {best['step']:>6,}, where T = "
                  f"{at['val']['fitted_t']:.2f}")
        """)
    out["own"] = s.run("""
        for arm in ("recipe", "long", "small"):
            rows = [r for r in traj["arms"][arm] if r["step"] > 0]
            ts = [r["train"]["fitted_t"] for r in rows]
            gaps = [r["train"]["gap"] for r in rows]
            print(f"{arm:6s}  own-text T {min(ts):.2f} to {max(ts):.2f}   "
                  f"confidence - accuracy {min(gaps):+.3f} to {max(gaps):+.3f}")
        """)
    out["control"] = s.run("""
        for arm in ("small", "small_seed1"):
            rows = [r for r in traj["arms"][arm] if r["step"] > 0]
            best = tj.turn(rows)
            at = next(r for r in rows if r["step"] == best["step"])
            print(f"{arm:11s}  lowest val NLL {best['value']:.4f} at step "
                  f"{best['step']:,}   T there {at['val']['fitted_t']:.2f}   "
                  f"T at end {rows[-1]['val']['fitted_t']:.2f}")
        """)
    out["stop"] = s.run("""
        for arm in ("long", "small"):
            rows = [r for r in traj["arms"][arm] if r["step"] > 0]
            raw = min(rows, key=lambda r: r["val"]["nll"])
            cal = min(rows, key=lambda r: r["val"]["nll_at_fit"])
            print(f"{arm:5s}  stop on raw NLL: step {raw['step']:>6,} -> "
                  f"{raw['val']['nll_at_fit']:.4f} after scaling")
            print(f"{'':5s}  stop on scaled NLL: step {cal['step']:>6,} -> "
                  f"{cal['val']['nll_at_fit']:.4f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    arms, cfg, com = res["arms"], res["cfg"], res["committed"]["val"]
    rc, lg, sm = arms["recipe"], arms["long"], arms["small"]
    s1 = arms["small_seed1"]
    rcf = rc["final"]["val"]
    own_max = max(arms[n]["train_t_max"] for n in ARMS)
    own_min = min(arms[n]["train_t_min"] for n in ARMS)
    smb, smf = sm["best"], sm["final"]
    lgb, lgf = lg["best"], lg["final"]

    # The spine, asserted rather than trusted.
    assert abs(rcf["nll"] - com["nll"]) < 0.02          # a faithful sibling
    assert abs(rcf["fitted_t"] - com["fitted_t"]) < 0.02
    assert all(arms[n]["early_t_min"] < 1.0 for n in ARMS)   # early: under
    for n in ("long", "small"):
        assert arms[n]["onset"]["step"] < arms[n]["best"]["step"]
        assert arms[n]["best"]["val"]["fitted_t"] > 1.15
    assert own_max < 1.03                       # never overconfident on own
    assert sm["train_t_max"] < 1.0 and s1["train_t_max"] < 1.0
    assert smf["val"]["nll"] > tiny.UNIFORM_LOSS > smf["val"]["nll_at_fit"]
    assert abs(smf["val"]["accuracy"] - smb["val"]["accuracy"]) < 0.03
    assert s1["best"]["val"]["fitted_t"] > 1.15     # the control agrees
    assert lgf["val"]["nll_at_fit"] < lgb["val"]["nll_at_fit"]

    post = Post(
        title=(f"{SERIES_TAG} 4: Overconfidence Arrives Before the Model "
               f"Stops Improving"),
        slug="coverage-4-training",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=4,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("The syllabus said overconfidence would arrive when "
                  "validation loss stopped improving. Three training runs "
                  "with checkpoints say it arrives much earlier, and only "
                  "on text the model has not seen."),
        summary=(
            f"Three training runs of the same {tiny.load()['parameters']:,}-"
            f"parameter model, evaluated at every checkpoint on "
            f"{cfg['eval_rows']:,} held-out predictions. Every run starts "
            f"slightly underconfident; overconfidence begins after one to "
            f"ten passes over the training text and grows as training loss "
            f"pulls away from validation loss. At the lowest validation loss "
            f"the model already needs a temperature of "
            f"{lgb['val']['fitted_t']:.2f} (fifteen thousand steps) or "
            f"{smb['val']['fitted_t']:.2f} (a tenth of the text), and a "
            f"second seed gives {s1['best']['val']['fitted_t']:.2f}. On its "
            f"own training text the fitted temperature never exceeds "
            f"{own_max:.2f}. Past the minimum, the small-data run's "
            f"validation NLL climbs to {smf['val']['nll']:.2f}, worse than a "
            f"uniform guess, with accuracy almost unchanged - and one "
            f"temperature brings it back to {smf['val']['nll_at_fit']:.2f}."),
        tags=["machine-learning", "data-science", "statistics",
              "calibration", "uncertainty", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every number is computed from model "
            "predictions on held-out text; no values from the text are "
            "published.",
            f"Three training runs of the episode-1 architecture - four "
            f"blocks, four heads, width {tiny.WIDTH}, context {tiny.BLOCK} - "
            f"from `scripts/train_coverage4.py`, plus a seed-1 repeat of the "
            f"small-data run. Per-checkpoint metrics are committed in "
            f"`data/coverage4/trajectory.json`; the final and lowest-loss "
            f"checkpoints of each run are committed beside it with their "
            f"sha256, so the endpoints can be recomputed from weights.",
            "Machinery: `standarderror/uncertainty/trajectory.py`, tested in "
            "`tests/test_coverage4.py`, which reloads the committed "
            "checkpoints and checks the trajectory file against them.",
            "Where this stops: Guo et al., \"On calibration of modern neural "
            "networks\", *ICML* (2017), who observed that overconfidence "
            "accompanies NLL overfitting while accuracy still improves; "
            "Mukhoti et al., \"Calibrating deep neural networks using focal "
            "loss\", *NeurIPS* (2020), on where in training it starts.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"torch={cfg['torch']}, numpy={np.__version__}"),
            "code blocks": ("executed at build time against the committed "
                            "trajectory file and checkpoint; the values the "
                            "prose quotes are pinned, so drift fails the "
                            "build"),
            "training": ("each run restarts from a fixed seed with AdamW, "
                         "weight decay 0.1, batch 32 and a one-cycle "
                         "schedule peaking at 3e-3; not bit-reproducible "
                         "across torch versions, which is why the "
                         "checkpoints are committed"),
            "evaluation": (f"the same {cfg['eval_rows']:,} validation "
                           f"predictions at every checkpoint, and the same "
                           f"number from the text each run trained on; the "
                           f"temperature is fitted on those rows"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The question episode 3 left",
        f"""Episode 3 measured a model that needed almost no temperature scaling: a fitted temperature of 1.10, a mild correction. It ended by asking why. Is a small transformer simply well calibrated, or was that the moment training happened to stop?

The syllabus guessed the answer. Overconfidence would arrive at a particular point: when validation loss stops improving and the model starts memorising. That guess is testable, and it needs something episode 3 did not have — the same model at many points in its training.

So the architecture was trained three times, with a checkpoint evaluated every few hundred steps on the same {cfg['eval_rows']:,} validation predictions:

- **recipe** — the committed model's recipe: the full training text, 3,000 steps, about {rc['final']['passes']:.0f} passes over it;
- **long** — the same schedule stretched to 15,000 steps, about {lg['final']['passes']:.0f} passes;
- **small** — a tenth of the training text and 6,000 steps, about {sm['final']['passes']:.0f} passes, so that memorising is easy.

First, whether the rerun recipe is the same kind of model as the committed one.

{snip['recipe'].markdown()}

Close enough to stand in for it: validation NLL {rcf['nll']:.3f} against {com['nll']:.3f}, fitted temperature {rcf['fitted_t']:.3f} against {com['fitted_t']:.3f}. Training is not bit-reproducible across torch versions, so this is a sibling, not a copy, and the episode's claims are about runs, not about one set of weights.""")

    post.add(
        "It arrives early",
        f"""For every checkpoint, fit the temperature that minimises validation NLL. Below 1 the model is underconfident, above 1 it is overconfident.

{snip['onset'].markdown()}

Every run *starts* underconfident — the lowest fitted temperature in each run's first few checkpoints is below 1 — and crosses into overconfidence early: after {rc['onset']['passes']:.1f} passes for the recipe, {lg['onset']['passes']:.1f} for the long run and {sm['onset']['passes']:.0f} for the small one. Validation loss is still falling fast at that point, and goes on falling for thousands of steps.

When it finally bottoms out the model is not becoming overconfident; it already is. The long run's best checkpoint needs T = {lgb['val']['fitted_t']:.2f}. The small-data run's best needs **T = {smb['val']['fitted_t']:.2f}**, with an ECE of {smb['val']['ece']:.3f} — {smb['val']['ece'] / com['ece']:.0f} times the committed model's. The syllabus had the order backwards: stopping when validation loss stops improving does not stop before overconfidence, it stops well inside it.""",
        figures=[figs["f0"]])

    post.add(
        "It barely registers on the training text",
        f"""The dotted lines in that figure are the same checkpoints scored on the text each run was trained on — for the small run, on its own tenth.

{snip['own'].markdown()}

On its own training text the model is never more than marginally overconfident. The fitted temperature stays between {own_min:.2f} and {own_max:.2f} across all runs; it touches 1.02 only briefly, during the high-learning-rate phase of the full-data runs. In the small-data run it never reaches 1 — not even at the end, when its training NLL is {smf['train']['nll']:.2f} nats per character and it has effectively memorised the text. If anything the model is slightly *under*confident there.

That reframes the question. Overconfidence is not something the model develops; it is what memorisation looks like from outside. The model learns to be sure of what it has seen, correctly, and carries that certainty to text it has not seen, where it is wrong. The fitted temperature on held-out text rises as the gap between training and validation loss opens, in every run, and the gap opens long before validation loss turns.""")

    post.add(
        "Past the minimum, the loss is confidence",
        f"""The small-data run shows what happens after the turn. Its validation NLL rises from {smb['val']['nll']:.2f} to {smf['val']['nll']:.2f} — past {tiny.UNIFORM_LOSS:.2f}, the loss of guessing uniformly over all 65 characters. Read naively, the model ends up knowing less than nothing.

Its accuracy says otherwise: {smf['val']['accuracy']:.3f} at the end against {smb['val']['accuracy']:.3f} at the minimum, and it peaked at {sm['acc_best']['val']['accuracy']:.3f} *after* the minimum. The answers barely moved. What moved is how sure the model is of them, and a single temperature, T = {smf['val']['fitted_t']:.2f}, takes the final NLL from {smf['val']['nll']:.2f} to {smf['val']['nll_at_fit']:.2f}. Most of the "overfitting" in the loss curve is one scalar's worth of confidence.""",
        figures=[figs["f1"]])

    post.add(
        "A second seed, and what early stopping is choosing",
        f"""The small-data run is the strongest evidence, so it was repeated from a second seed.

{snip['control'].markdown()}

Same picture: the best checkpoint already needs a temperature of {s1['best']['val']['fitted_t']:.2f}, and the end {s1['final']['val']['fitted_t']:.2f}. The end temperatures agreeing to the printed precision is a coincidence, not a copied run — the two start from different weights, see the text in a different order, and end at validation NLL {smf['val']['nll']:.3f} and {s1['final']['val']['nll']:.3f}.

That has a practical consequence for the most common way of choosing a checkpoint. Early stopping on raw validation NLL penalises overconfidence that one temperature would remove, so it stops on calibration rather than on what the model knows. Choosing the checkpoint by NLL *after* temperature scaling picks a different one:

{snip['stop'].markdown()}

In the long run, the checkpoint raw NLL would keep is beaten after scaling by the final one, {lgf['val']['nll_at_fit']:.4f} against {lgb['val']['nll_at_fit']:.4f}: the extra {lgf['step'] - lgb['step']:,} steps added knowledge and overconfidence together, and only the second is free to remove. In the small run the scaled choice moves from step {smb['step']:,} to {sm['scaled_best']['step']:,}. The differences are small here. The principle is not: if you will temperature-scale anyway, select the checkpoint on the scaled loss.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        f"""**One architecture, one corpus, one schedule family.** All three runs use a one-cycle schedule, which entangles learning rate with time. The small run overfits while its learning rate is still rising to its peak and the long run while it anneals, and both become overconfident in the same way, which is the best evidence available here that the schedule is not the cause. It is not proof.

**The temperature is fitted on the rows it is scored on.** One parameter on {cfg['eval_rows']:,} rows; the optimism this introduces is of order 1/n, about 0.0001 nats, far below every difference quoted.

**Two seeds for one arm.** The control repeats the arm with the largest effect. The long and recipe runs are single seeds, and the onset steps quoted for them should be read to the nearest few hundred.

**Character-level language modelling is easy to memorise.** A tenth of tinyshakespeare is about {int(cfg['train_chars'] * 0.1):,} characters for {tiny.load()['parameters']:,} parameters. The size of the effect at the end of the small run is a property of that ratio; the ordering — overconfident before the minimum, calibrated on the training text — held in every run that reached a minimum.""")

    post.add(
        "What to keep",
        f"""1. A rerun of the committed recipe reproduces it closely: validation NLL {rcf['nll']:.3f}, fitted temperature {rcf['fitted_t']:.3f}.

2. Every run starts slightly underconfident and becomes overconfident early, while validation loss is still falling fast.

3. At the lowest validation loss the model already needs T = {lgb['val']['fitted_t']:.2f} (long run) or {smb['val']['fitted_t']:.2f} (small data; {s1['best']['val']['fitted_t']:.2f} on a second seed). Early stopping does not stop before overconfidence.

4. On its own training text the model is at most marginally overconfident: fitted temperature {own_min:.2f} to {own_max:.2f} throughout, and below 1 for the whole small-data run. Overconfidence is memorisation seen from held-out text.

5. Past the minimum, validation NLL can pass the uniform-guess loss while accuracy holds; one temperature removes most of it ({smf['val']['nll']:.2f} to {smf['val']['nll_at_fit']:.2f}).

6. If you will temperature-scale, choose the checkpoint on the scaled validation loss, not the raw one.

7. Episode 3's T = 1.10 was not a property of the architecture. It was a property of stopping at {rc['final']['passes']:.0f} passes.""")

    post.add(
        "Exercise",
        """If you train with checkpoints, you already have what this episode needed. Fit a temperature on validation data at each checkpoint and plot it against step, next to the validation loss. Note where the temperature crosses 1 and where the loss bottoms out. If the crossing comes first, as it did in every run here, then whichever checkpoint you shipped was overconfident when you shipped it, and the temperature you fitted afterwards was doing more work than you thought.

Then score the same checkpoints on a sample of the training data. If the fitted temperature there stays at 1 while the validation one climbs, your model's overconfidence is a generalisation gap: regularisation and more data are the levers, and temperature scaling is the repair. If the training-data temperature climbs too, something else — label noise, a loss that rewards overconfidence — is at work, and that is a different episode.""")

    post.add(
        "Next",
        """Episode 5 returns to the loose end episode 1 left: the error bar on conformal coverage, which episode 1 said was wider than exchangeability allows. Part of that turns out to have been my arithmetic.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
