"""Coverage 1: 90% Coverage Is a Promise About Averages, Not About You.

The first episode of a series about the guarantee you are actually getting.
This one is about conformal prediction's quantifier, and it ends by largely
undoing its own complaint.

Measured:

* Split conformal's marginal guarantee is exact and it is delivered: coverage
  0.8961 against a finite-sample level of 0.9001, and over 200 calibration
  splits a mean of 0.8999.
* It is an average. Conditioned on the model's own confidence in fifths:
  0.823, 0.896, 0.900, 0.913, 0.948 -- a spread of 0.125, running in the
  direction that matters, because the shortfall lands on the least-confident
  cases.
* Operationally: escalate the least-confident 20% and the cases you keep have
  0.914 coverage while the queue you hand a human has 0.823. The reviewer gets
  the cases whose prediction set is least likely to contain the answer.
* And then the reversal. That unevenness is the *score*, not conformal.
  Randomised adaptive prediction sets hit the same marginal level, 0.8975, and
  cut the conditional spread from 0.125 to 0.017 -- seven-fold -- for 20% more
  set size and a 0.9% chance of returning an empty set.
* Set sizes are the other half: mean 5.40 labels of 65, but 11.63 in the
  least-confident fifth against 1.28 in the most. A prediction set is
  informative exactly where you did not need it.

Run: `standarderror run cv101_coverage --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.llm import tiny
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.uncertainty import coverage as cv
from standarderror.viz import charts

POST_DATE = date(2026, 9, 11)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Uncertainty for Language Models, Taught Through What Breaks"
SERIES_TAG = "Coverage"

ALPHA = 0.10
BATCHES, BATCH_SIZE = 24, 16
BANDS = 5
SPLITS = 200


def compute() -> dict:
    pred = cv.predictions(count=BATCHES, size=BATCH_SIZE, seed=1)
    lac = cv.split_conformal(pred, alpha=ALPHA)
    aps = cv.aps_conformal(pred, alpha=ALPHA)
    return {
        "pred": pred, "lac": lac, "aps": aps,
        "bands_lac": cv.conditional(pred, lac, bands=BANDS),
        "bands_aps": cv.conditional(pred, aps, bands=BANDS),
        "escalation": cv.escalation(pred, lac),
        "variability": cv.split_variability(pred, alpha=ALPHA, draws=SPLITS),
        "levels": {a: cv.split_conformal(pred, alpha=a)
                   for a in (0.20, 0.10, 0.05, 0.02, 0.01)},
    }


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    pred, lac, aps = res["pred"], res["lac"], res["aps"]
    bl, ba = res["bands_lac"], res["bands_aps"]
    esc = res["escalation"]

    def exact(ax, m):
        xs = np.arange(len(bl))
        ax.bar(xs, [b["coverage"] for b in bl], width=0.6,
               color=m.series[0], label="coverage within the band")
        ax.axhline(lac["guarantee"], lw=2.0, ls="--", color=m.series[2],
                   label=f"the guarantee, {lac['guarantee']:.4f}")
        ax.axhline(lac["coverage"], lw=1.8, ls=":", color=m.series[1],
                   label=f"marginal coverage, {lac['coverage']:.4f}")
        for x, b in zip(xs, bl):
            ax.annotate(f"{b['coverage']:.3f}", (x, b["coverage"]),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=9.2, color=m.ink_secondary)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b['low']:.2f}\nto {b['high']:.2f}" for b in bl],
                           fontsize=8.4)
        ax.set_ylim(0.78, 1.0)
        ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    out["f0"] = charts.diagram(
        exact,
        title="The average is exactly right and no band is",
        subtitle=(f"Split conformal at alpha = {ALPHA}, on "
                  f"{lac['n_test']:,} held-out predictions, split into fifths "
                  f"by the model's own confidence."),
        xlabel="the model's confidence in its top label",
        ylabel="share of cases whose set contained the truth",
        source="Measured; standarderror/uncertainty/coverage.py.",
        alt=("Five bars rising from about 0.82 to about 0.95 across a dashed "
             "guarantee line at 0.90."),
        caption=(f"The marginal coverage is {lac['coverage']:.4f} against a "
                 f"finite-sample guarantee of {lac['guarantee']:.4f} - the "
                 f"method delivered exactly what it promised. Within bands it "
                 f"runs {bl[0]['coverage']:.3f} to {bl[-1]['coverage']:.3f}, "
                 f"a spread of {cv.conditional_range(bl):.3f}, and **the "
                 f"shortfall is on the left**: the guarantee is weakest where "
                 f"the model is least sure."),
        path=str(IMG / f"cv101-f0-bands.{EXT}"))[0]

    def queue(ax, m):
        f = [100 * e["fraction"] for e in esc]
        ax.plot(f, [e["kept_coverage"] for e in esc], marker="o", ms=6,
                lw=2.2, color=m.series[0],
                label="the cases you keep and answer")
        ax.plot(f, [e["escalated_coverage"] for e in esc], marker="s", ms=6,
                lw=2.2, color=m.series[2],
                label="the cases you send to a human")
        ax.axhline(1 - ALPHA, lw=1.8, ls="--", color=m.grid,
                   label=f"the {1 - ALPHA:.0%} you were promised")
        ax.set_ylim(0.78, 0.96)
        ax.legend(frameon=False, fontsize=8.6, loc="center right")

    out["f1"] = charts.diagram(
        queue,
        title="Escalating the unsure cases sorts the failures into the queue",
        subtitle=("Send the least-confident share to a reviewer and keep the "
                  "rest. Coverage of each group, against the level the "
                  "guarantee names."),
        xlabel="share of cases escalated (%)",
        ylabel="coverage within the group",
        source="Measured; standarderror/uncertainty/coverage.py.",
        alt=("Two curves either side of a dashed line: the kept cases above "
             "it and rising, the escalated cases below it and flat."),
        caption=(f"At a 20% escalation rate the cases you answer have "
                 f"{[e for e in esc if e['fraction'] == 0.2][0]['kept_coverage']:.3f} "
                 f"coverage and the queue has "
                 f"{[e for e in esc if e['fraction'] == 0.2][0]['escalated_coverage']:.3f}. "
                 f"Both groups are what the guarantee allows, and the policy "
                 f"anyone would write hands the reviewer the cases whose sets "
                 f"are **least** likely to contain the answer - while also "
                 f"being the longest, "
                 f"{[e for e in esc if e['fraction'] == 0.2][0]['escalated_size']:.1f} "
                 f"labels against "
                 f"{[e for e in esc if e['fraction'] == 0.2][0]['kept_size']:.1f}."),
        path=str(IMG / f"cv101-f1-escalation.{EXT}"))[0]

    def scores(ax, m):
        xs = np.arange(len(bl))
        ax.plot(xs, [b["coverage"] for b in bl], marker="o", ms=7, lw=2.3,
                color=m.series[2],
                label=f"1 - p[true], spread {cv.conditional_range(bl):.3f}")
        ax.plot(xs, [b["coverage"] for b in ba], marker="s", ms=7, lw=2.3,
                color=m.series[0],
                label=f"adaptive sets, spread {cv.conditional_range(ba):.3f}")
        ax.axhline(1 - ALPHA, lw=1.8, ls="--", color=m.grid,
                   label=f"{1 - ALPHA:.0%}")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b['low']:.2f}\nto {b['high']:.2f}" for b in bl],
                           fontsize=8.4)
        ax.set_ylim(0.78, 1.0)
        ax.legend(frameon=False, fontsize=8.6, loc="lower right")

    out["f2"] = charts.diagram(
        scores,
        title="The unevenness was the score, not the method",
        subtitle=(f"Both scores hit the same marginal level - "
                  f"{lac['coverage']:.4f} and {aps['coverage']:.4f} - and "
                  f"distribute it very differently."),
        xlabel="the model's confidence in its top label",
        ylabel="coverage within the band",
        source="Measured; standarderror/uncertainty/coverage.py.",
        alt=("One line sloping up from 0.82 to 0.95 and another nearly flat "
             "along the dashed 90% line."),
        caption=(f"Swapping the non-conformity score cuts the conditional "
                 f"spread from {cv.conditional_range(bl):.3f} to "
                 f"{cv.conditional_range(ba):.3f}, a factor of "
                 f"{cv.conditional_range(bl) / cv.conditional_range(ba):.1f}, "
                 f"at the same marginal coverage. The bill is "
                 f"{aps['mean_size'] / lac['mean_size'] - 1:.0%} more labels "
                 f"per set and a {aps['empty']:.1%} chance of returning "
                 f"**nothing**, which the first score never does."),
        path=str(IMG / f"cv101-f2-scores.{EXT}"))[0]

    rows = []
    for name, fit, bands in (("1 - p[true]", lac, bl),
                             ("adaptive sets", aps, ba)):
        rows.append([name, f"{fit['coverage']:.4f}",
                     f"{cv.conditional_range(bands):.3f}",
                     f"{fit['mean_size']:.2f}",
                     f"{fit['median_size']:.0f}",
                     f"{100 * fit['singletons']:.1f}%",
                     f"{100 * fit['empty']:.1f}%"])
    bold = {(1, c) for c in range(7)}

    out["f3"] = charts.table_image(
        rows,
        header=["score", "coverage", "conditional spread", "mean size",
                "median", "singletons", "empty"],
        title="Two ways to spend the same guarantee",
        subtitle=(f"Both at alpha = {ALPHA} on the same "
                  f"{lac['n_test']:,} predictions over "
                  f"{pred['classes']} labels."),
        source="Measured; standarderror/uncertainty/coverage.py.",
        alt=("A two-row table. The adaptive row has a much smaller "
             "conditional spread, a larger mean size and a nonzero share of "
             "empty sets."),
        caption=("The **bold** row buys conditional coverage and pays in "
                 "size and in occasional silence. Neither row is the right "
                 "answer; the point is that the marginal guarantee does not "
                 "choose between them, so it is a choice you are making "
                 "whether or not you know it."),
        bold_cells=bold, align="lrrrrrr",
        path=str(IMG / f"cv101-f3-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    lac, aps = res["lac"], res["aps"]
    bl, ba = res["bands_lac"], res["bands_aps"]
    esc = res["escalation"]

    def flat(panel, m):
        panel.bar([0], [lac["coverage"]], width=0.5, color=m.series[0])
        panel.axhline(0.9, lw=2.0, ls="--", color=m.series[2])
        panel.set_xlim(-0.6, 0.6)
        panel.set_ylim(0.8, 0.95)

    def slope(panel, m):
        xs = np.arange(len(bl))
        panel.plot(xs, [b["coverage"] for b in bl], lw=2.8,
                   color=m.series[2])
        panel.axhline(0.9, lw=1.8, ls="--", color=m.grid)
        panel.set_ylim(0.79, 0.97)

    def both(panel, m):
        xs = np.arange(len(bl))
        panel.plot(xs, [b["coverage"] for b in bl], lw=2.4, color=m.grid)
        panel.plot(xs, [b["coverage"] for b in ba], lw=2.8,
                   color=m.series[0])
        panel.axhline(0.9, lw=1.6, ls="--", color=m.series[2])
        panel.set_ylim(0.79, 0.97)

    twenty = [e for e in esc if e["fraction"] == 0.2][0]
    return charts.lecture_hero(
        series=SERIES_TAG, episode=1,
        headline="The guarantee is exact, and it is about someone else",
        panels=[(flat, f"{lac['coverage']:.3f}", "marginal coverage"),
                (slope, f"{bl[0]['coverage']:.3f}", "least-sure fifth"),
                (both, f"{cv.conditional_range(ba):.3f}",
                 "spread, better score")],
        note=(f"Split conformal delivered {lac['coverage']:.4f} against a "
              f"guarantee of {lac['guarantee']:.4f}. Conditioned on the "
              f"model's own confidence it runs {bl[0]['coverage']:.3f} to "
              f"{bl[-1]['coverage']:.3f} - so escalating the least-sure 20% "
              f"hands a reviewer {twenty['escalated_coverage']:.3f} coverage. "
              f"Changing the score fixes most of that for "
              f"{aps['mean_size'] / lac['mean_size'] - 1:.0%} more labels."),
        alt=("Three hand-drawn frames: a bar meeting a dashed target; a line "
             "climbing across that target from below; and two lines, one "
             "climbing and one flat along it."),
        path=str(IMG / f"cv101-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["exact"] = s.run(f"""
        from standarderror.uncertainty import coverage as cv

        # The model's probabilities on held-out text, then split conformal.
        pred = cv.predictions(count={BATCHES}, size={BATCH_SIZE}, seed=1)
        fit = cv.split_conformal(pred, alpha={ALPHA})
        print(f"predictions      {{pred['rows']:,}} over "
              f"{{pred['classes']}} labels")
        print(f"model accuracy   {{pred['accuracy']:.4f}}")
        print()
        print(f"guarantee        {{fit['guarantee']:.4f}}"
              f"   = ceil((n+1)(1-alpha))/(n+1)")
        print(f"coverage         {{fit['coverage']:.4f}}")
        print(f"mean set size    {{fit['mean_size']:.2f}} of "
              f"{{pred['classes']}}  ({{fit['size_share']:.1%}})")
        print(f"singletons       {{fit['singletons']:.1%}}"
              f"   empty {{fit['empty']:.1%}}")
    """, expect=["predictions"])

    out["bands"] = s.run(f"""
        # Now condition on the one thing anyone would condition on.
        for b in cv.conditional(pred, fit, bands={BANDS}):
            print(f"confidence {{b['low']:.3f}} to {{b['high']:.3f}}   "
                  f"n {{b['n']:5d}}   coverage {{b['coverage']:.4f}}   "
                  f"mean size {{b['mean_size']:5.2f}}")
    """, expect=["confidence"])

    out["aps"] = s.run(f"""
        # Same guarantee, different non-conformity score.
        aps = cv.aps_conformal(pred, alpha={ALPHA})
        for name, f in (("1 - p[true]", fit), ("adaptive sets", aps)):
            bands = cv.conditional(pred, f, bands={BANDS})
            print(f"{{name:<14}} coverage {{f['coverage']:.4f}}   "
                  f"conditional spread {{cv.conditional_range(bands):.3f}}   "
                  f"size {{f['mean_size']:5.2f}}   "
                  f"empty {{f['empty']:.1%}}")
    """, expect=["coverage"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    pred, lac, aps = res["pred"], res["lac"], res["aps"]
    bl, ba = res["bands_lac"], res["bands_aps"]
    esc, var, levels = res["escalation"], res["variability"], res["levels"]

    # The spine, asserted rather than trusted.
    assert lac["guarantee"] >= 1 - ALPHA
    assert abs(lac["coverage"] - lac["guarantee"]) < 0.02
    assert abs(var["mean"] - (1 - ALPHA)) < 0.01
    assert cv.conditional_range(bl) > 0.05
    assert bl[0]["coverage"] < 1 - ALPHA < bl[-1]["coverage"]
    assert bl[0]["mean_size"] > 3 * bl[-1]["mean_size"]
    assert all(e["escalated_coverage"] < e["kept_coverage"] for e in esc)
    assert abs(aps["coverage"] - lac["coverage"]) < 0.02
    assert cv.conditional_range(ba) < cv.conditional_range(bl) / 3
    assert aps["mean_size"] > lac["mean_size"]
    assert lac["empty"] == 0.0 and aps["empty"] > 0.0
    sizes = [levels[a]["mean_size"] for a in (0.20, 0.10, 0.05, 0.02, 0.01)]
    assert sizes == sorted(sizes)

    post = Post(
        title=(f"{SERIES_TAG} 1: 90% Coverage Is a Promise About Averages, "
               f"Not About You"),
        slug="coverage-1-marginal",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=1,
        date=POST_DATE,
        # A theorem plus counts on a fixed model. No prediction to beat.
        requires_baseline=False,
        subtitle=("Conformal prediction's guarantee is exact, "
                  "distribution-free and finite-sample, and it is an average "
                  "over the test distribution. Conditioned on the model's own "
                  "confidence it runs from 82% to 95% - and then it turns out "
                  "that the unevenness is the score's fault, not the "
                  "method's."),
        summary=(
            f"Split conformal at alpha = {ALPHA} on "
            f"{pred['rows']:,} held-out predictions split in half delivers "
            f"{lac['coverage']:.4f} coverage against a finite-sample "
            f"guarantee of {lac['guarantee']:.4f}, and over {SPLITS} "
            f"calibration splits a mean of {var['mean']:.4f}. Exactly what was "
            f"promised. Conditioned on the model's own confidence in fifths it "
            f"runs {bl[0]['coverage']:.3f} to {bl[-1]['coverage']:.3f}, a "
            f"spread of {cv.conditional_range(bl):.3f}, with the shortfall on "
            f"the least-confident cases - so escalating the least-sure 20% "
            f"hands a reviewer "
            f"{[e for e in esc if e['fraction'] == 0.2][0]['escalated_coverage']:.3f} "
            f"coverage while keeping "
            f"{[e for e in esc if e['fraction'] == 0.2][0]['kept_coverage']:.3f}. "
            f"Set sizes tell the same story from the other side: "
            f"{lac['mean_size']:.2f} labels of {pred['classes']} on average "
            f"but {bl[0]['mean_size']:.1f} where the model is unsure against "
            f"{bl[-1]['mean_size']:.1f} where it is not. And then the "
            f"reversal: swapping the non-conformity score for randomised "
            f"adaptive sets holds the marginal level at {aps['coverage']:.4f} "
            f"and cuts the conditional spread to "
            f"{cv.conditional_range(ba):.3f} - a factor of "
            f"{cv.conditional_range(bl) / cv.conditional_range(ba):.1f} - for "
            f"{aps['mean_size'] / lac['mean_size'] - 1:.0%} more labels per "
            f"set and a {aps['empty']:.1%} chance of returning nothing."),
        tags=["machine-learning", "data-science", "statistics",
              "conformal-prediction", "uncertainty", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every number is a coverage rate or a set size "
            "computed on held-out text from the corpus the model was trained "
            "on, and no values from that text are published.",
            f"The model: an {tiny.load()['parameters']:,}-parameter "
            f"character-level transformer trained for this series - four "
            f"blocks, four heads, width {tiny.WIDTH}, context {tiny.BLOCK}, "
            f"validation loss {tiny.VAL_LOSS} against a uniform-guess "
            f"{tiny.UNIFORM_LOSS:.3f}. Weights, training script and a "
            f"verified sha256 are committed: `standarderror/llm/tiny.py`, "
            f"`scripts/train_tiny.py`, `data/tiny_gpt/`.",
            "Machinery: `standarderror/uncertainty/coverage.py`, tested in "
            "`tests/test_uncertainty.py`, which pins the finite-sample level "
            "exactly and the model findings as inequalities.",
            "Where this stops: Vovk, Gammerman and Shafer, *Algorithmic "
            "Learning in a Random World* (2005), for conformal prediction; "
            "Angelopoulos and Bates, \"A gentle introduction to conformal "
            "prediction and distribution-free uncertainty quantification\" "
            "(2021), for split conformal as used here; Romano, Sesia and "
            "Candes, \"Classification with valid and adaptive coverage\", "
            "*NeurIPS* (2020), for the adaptive score and its randomisation; "
            "Barber, Candes, Ramdas and Tibshirani, \"The limits of "
            "distribution-free conditional predictive inference\", "
            "*Information and Inference* (2021), for the theorem that says "
            "the gap above cannot be closed entirely.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, "
                           "torch=2.14.0, numpy=2.4.4, scipy=1.16.3",
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "model": ("the committed checkpoint, hash-verified on load, so "
                      "every coverage rate is measured on the same weights"),
            "determinism": (f"the calibration and test halves come from a "
                            f"fixed permutation; the variability figure "
                            f"repeats the split {SPLITS} times with seeds 0 "
                            f"to {SPLITS - 1}; the adaptive score's "
                            f"randomisation uses the same generator, so the "
                            f"comparison is reproducible and does depend on "
                            f"that seed"),
        },
    )
    post.hero = figs["hero"]
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    pred, lac, aps = res["pred"], res["lac"], res["aps"]
    bl, ba = res["bands_lac"], res["bands_aps"]
    esc, var, levels = res["escalation"], res["variability"], res["levels"]
    twenty = [e for e in esc if e["fraction"] == 0.2][0]
    factor = cv.conditional_range(bl) / cv.conditional_range(ba)

    post.add(
        "A guarantee with a quantifier in it",
        """Conformal prediction is the rare thing in machine learning that comes with a proof. Give it any model, any way of scoring how badly a label fits, and a calibration set; it returns a set of labels for each new input, and that set contains the truth with probability at least 1 − α. No assumption about the model, no assumption about the distribution, no appeal to asymptotics. One assumption: that the calibration and test points are exchangeable.

It is a real guarantee and it is delivered. It is also a statement about an average, and the average is over the test distribution rather than over anything you selected. This episode is about what that costs, and then about how much of the cost is avoidable, which turned out to be most of it.""")

    post.add(
        "The guarantee, and it is exact",
        """Take the committed character-level model, run it on held-out text, and use the simplest score there is: 1 − *p* assigned to the true label. Hold out half the predictions to calibrate, take the ⌈(*n*+1)(1−α)⌉-th smallest score, and include every label whose probability clears the corresponding threshold.""")

    post.add(
        "",
        f"""{snip['exact'].markdown()}

{lac['coverage']:.4f} against {lac['guarantee']:.4f}. Note that the target is not {1 - ALPHA:.2f} but slightly above it — the construction achieves a specific finite-sample level, and the realised coverage should land just under *that*. Over {SPLITS} random calibration splits the mean is {var['mean']:.4f}.

The sets are usable too: {lac['mean_size']:.2f} labels out of {pred['classes']} on average, {lac['size_share']:.1%} of the vocabulary, and {lac['singletons']:.1%} of predictions get a single label. Nothing is broken here. The method promised a number and produced it.""",
        level=3)

    post.add(
        "Conditioned on the one thing you would condition on",
        """Now ask the question anyone deploying this would ask: is the guarantee even? Not across some protected attribute — across the model's own confidence, because that is what you would route on.""")

    post.add(
        "",
        f"""{snip['bands'].markdown()}

{bl[0]['coverage']:.3f} in the least-confident fifth, {bl[-1]['coverage']:.3f} in the most confident. A spread of {cv.conditional_range(bl):.3f} around a marginal {lac['coverage']:.3f}, and it slopes the wrong way.

Nothing has been violated. The guarantee constrains the average of those five numbers and says nothing about their spread, so a method that delivers 0.82 and 0.95 has kept its promise exactly as much as one that delivers 0.90 twice. But the arithmetic means **the cases where the model is unsure are the cases where the set is least likely to contain the answer**, which is the opposite of the property you wanted when you reached for prediction sets.

The set sizes say the same thing from the other side: {bl[0]['mean_size']:.1f} labels where the model is unsure against {bl[-1]['mean_size']:.1f} where it is sure. The set is short and reliable when you did not need it and long and unreliable when you did.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "What that does to the obvious policy",
        f"""The obvious policy is to escalate. Answer the confident cases, send the rest to a human. Which sorts the test set on exactly the axis the coverage varies along.

At a 20% escalation rate the cases you answer have {twenty['kept_coverage']:.3f} coverage and the queue has {twenty['escalated_coverage']:.3f}. Every escalation rate I measured does the same thing: the kept group is above the promised level and the queue is below it, and the queue's sets are also the long ones — {twenty['escalated_size']:.1f} labels against {twenty['kept_size']:.1f}.

So the reviewer receives the cases with the longest prediction sets *and* the lowest chance those sets contain the answer, and the number on the tin said 90%. If you are going to tell a human "this set covers the truth nine times in ten", the set you hand them is the one where it does not.""",
        figures=[figs["f1"]])

    post.add(
        "And now the part that undoes most of the complaint",
        """Everything above is true and I would have stopped there, except that the unevenness is not conformal prediction's. It belongs to the score.

1 − *p*(true) puts a single global threshold on the top probability, so a row whose mass is spread across many labels gets a set built by the same rule as a row that is nearly certain. The adaptive score does something different: it accumulates probability from the most likely label downwards and asks how much mass you must take before reaching the truth. Rows that are diffuse then get longer sets *by construction* rather than by accident.

Both satisfy the same theorem. So the comparison is fair, provided the adaptive score is randomised — without the uniform jitter it is discrete and overcovers, measured at 0.98 against a target of 0.90, which would compare two different coverage levels and prove nothing.""")

    post.add(
        "",
        f"""{snip['aps'].markdown()}

Same marginal coverage, {lac['coverage']:.4f} and {aps['coverage']:.4f}. Conditional spread {cv.conditional_range(bl):.3f} down to {cv.conditional_range(ba):.3f} — a factor of {factor:.1f}.

The bill is {aps['mean_size'] / lac['mean_size'] - 1:.0%} more labels per set, {aps['mean_size']:.2f} against {lac['mean_size']:.2f}, and one thing that is easy to miss: {aps['empty']:.1%} of the adaptive sets are **empty**. The randomisation that makes the score exact can put the threshold below even the top label's contribution, and then the honest output is nothing at all. The first score never does that; it always contains the argmax.

That failure mode deserves more than a footnote, because the natural reflex is to patch it. An empty set is awkward to return from an API, so the obvious thing is to fall back to the top label — and the moment you do, the guarantee is gone. Those {aps['empty']:.1%} of cases are exactly the ones the calibration decided it could afford to miss; filling them with a guess does not make them covered, it makes the coverage unverifiable, because the set you return is no longer the set the theorem is about. If you need a non-empty answer, the correct move is to use a score that cannot produce one, which is the first score, and to accept the conditional spread that comes with it. The choice is between two honest options and one dishonest one.

Which of the two honest failure modes you prefer is a question about your application, and the marginal guarantee is silent on it.""",
        level=3,
        figures=[figs["f2"]])

    post.add(
        "",
        f"""What it is not is a fix. Barber, Candès, Ramdas and Tibshirani proved that exact conditional coverage cannot be had distribution-free: a procedure valid conditional on the input, for every distribution, is forced in the worst case to return sets carrying no information — intervals of infinite expected length in the continuous case, and the whole label list in ours. So {cv.conditional_range(ba):.3f} is not on its way to zero by picking a cleverer score, and a score that flattened these bands completely would be worth distrusting rather than adopting.

Which makes the honest summary a choice rather than a defect. Two scores, one guarantee, and a spread that differs {factor:.1f}-fold between them. The marginal guarantee does not pick, so you are picking, and if you have not looked at a table like the one below you are picking by default.""",
        level=3,
        figures=[figs["f3"]])

    post.add(
        "What tighter coverage costs",
        f"""One more thing the guarantee does not tell you, and it is the number you will actually be asked for. If 90% is not enough, what does 99% cost?

Measured on the same predictions, with the same simple score, as alpha tightens:

- α = 0.20 — coverage {levels[0.20]['coverage']:.4f}, mean set {levels[0.20]['mean_size']:.2f} labels, worst set {levels[0.20]['max_size']}
- α = 0.10 — {levels[0.10]['coverage']:.4f}, {levels[0.10]['mean_size']:.2f}, worst {levels[0.10]['max_size']}
- α = 0.05 — {levels[0.05]['coverage']:.4f}, {levels[0.05]['mean_size']:.2f}, worst {levels[0.05]['max_size']}
- α = 0.02 — {levels[0.02]['coverage']:.4f}, {levels[0.02]['mean_size']:.2f}, worst {levels[0.02]['max_size']}
- α = 0.01 — {levels[0.01]['coverage']:.4f}, {levels[0.01]['mean_size']:.2f}, worst {levels[0.01]['max_size']}

Coverage lands just under its guarantee at every level, which is the theorem doing its job five times over. The price is the second column: {levels[0.10]['mean_size'] / levels[0.20]['mean_size']:.1f} times the set size for the step from 80% to 90%, and {levels[0.01]['mean_size'] / levels[0.10]['mean_size']:.1f} times for the step from 90% to 99%.

The third column is the one worth staring at. At alpha = 0.10 the largest set anyone gets is {levels[0.10]['max_size']} labels of {pred['classes']} — {levels[0.10]['max_size'] / pred['classes']:.0%} of the vocabulary, which is an unhelpful answer but not an absurd one. At alpha = 0.01 it is {levels[0.01]['max_size']} of {pred['classes']}, or **{levels[0.01]['max_size'] / pred['classes']:.0%} of every label there is**. The set is still correct: it contains the truth, as promised, 99% of the time. It is simply not a prediction any more. Asking for 99% coverage from a model that is right {pred['accuracy']:.0%} of the time gets you exactly what it should, which is a shrug with a certificate attached.""")

    post.add(
        "Why the marginal guarantee is still the right default",
        f"""Having spent four sections on what the average hides, the balance is worth stating, because the obvious conclusion — "demand conditional coverage instead" — is not available.

Barber, Candès, Ramdas and Tibshirani's result is not a difficulty, it is an impossibility: a procedure valid conditional on the input, against every distribution, is forced in the worst case to return sets that say nothing — and over a finite label list, saying nothing means handing back the list. You cannot have a distribution-free promise about individuals. So the choice is not between a marginal guarantee and a conditional one; it is between a marginal guarantee you can verify in four lines and a conditional claim that would require assumptions you cannot check.

Put that way, the average is the right thing to promise. What is wrong is reading it as a promise about the case in front of you, and the correction is not to want a better theorem but to print the {BANDS}-band table. That table is not a diagnostic for a broken method. It is the missing half of the output.

Which is roughly where the previous series in this project ended as well: an exact statement that is true, that people over-read, and whose useful content sits one question further in. Here the question is "over what?", the answer is "over the test distribution, and not over your routing rule", and the fix is a different score and a table you were not printing.""")

    post.add(
        "One assumption, and it is the next episode",
        f"""There is a loose end I am deliberately leaving. Everything above rests on exchangeability, and I have been treating {lac['n_cal']:,} calibration rows as {lac['n_cal']:,} exchangeable draws when they are characters taken from overlapping contexts inside {BATCHES * BATCH_SIZE} sequences.

The marginal coverage survives it — {var['mean']:.4f} over {SPLITS} splits. What does not survive is the *error bar* on that number, and by a factor I did not expect and in a direction that gets worse when you do the obvious thing about it. That is episode 5.""")

    post.add(
        "What to keep",
        f"""1. Split conformal's coverage guarantee is exact, finite-sample and distribution-free, and the level is ⌈(*n*+1)(1−α)⌉/(*n*+1) rather than 1 − α. Measured: {lac['coverage']:.4f} against {lac['guarantee']:.4f}, and {var['mean']:.4f} averaged over {SPLITS} calibration splits.

2. It is an average over the test distribution. Conditioned on the model's own confidence it runs {bl[0]['coverage']:.3f} to {bl[-1]['coverage']:.3f} here, a spread of {cv.conditional_range(bl):.3f}, and nothing was violated.

3. The slope runs the wrong way: **least coverage where the model is least sure**, together with the longest sets — {bl[0]['mean_size']:.1f} labels against {bl[-1]['mean_size']:.1f}.

4. So the obvious escalation policy sorts the failures into the queue. At 20% escalation, {twenty['kept_coverage']:.3f} kept against {twenty['escalated_coverage']:.3f} escalated.

5. But that is the **score**, not the method. Randomised adaptive sets hold the marginal level and cut the spread by {factor:.1f}×.

6. The price is {aps['mean_size'] / lac['mean_size'] - 1:.0%} more labels and a {aps['empty']:.1%} chance of an empty set. The simple score always contains the argmax; the adaptive one sometimes contains nothing.

7. And it cannot be driven to zero — exact conditional coverage is impossible distribution-free. What is available is most of it, cheaply, and the marginal guarantee will not tell you that you had a decision to make.""")

    post.add(
        "Exercise",
        """Take whatever conformal wrapper you are using and print one table: coverage within five equal-mass bands of your model's own confidence. It is four lines and it is the only number that tells you whether your guarantee applies to the cases you route on. If the bands are flat, your score is already doing the adaptive thing and you can stop reading about this.

Then swap the score and print the same table. You are looking for two columns: the conditional spread, and the mean set size. One buys the other at a rate you can measure in an afternoon, and neither the α you chose nor the coverage you verified has any opinion about where on that curve you want to sit.

The uncomfortable version: check your empty-set rate. Every adaptive, randomised score has one, and a wrapper that silently returns the argmax when the set comes out empty has quietly given back the guarantee you were paying for.""")

    return post


if __name__ == "__main__":
    print(build().markdown()[:1500])
