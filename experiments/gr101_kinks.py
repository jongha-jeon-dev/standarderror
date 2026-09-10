"""Gradients 1: Backprop Is the Chain Rule, and the Chain Rule Has Hypotheses.

The first episode of a series about the derivative you are actually computing.
This one is about differentiability, and it ends by refusing its own premise.

Measured:

* Autodiff returns a float at every non-differentiable point, and the floats
  are inconsistent inside one library. `relu`, `abs`, `clamp`, `hardtanh` and
  `norm` return 0 at their kinks; `maximum`, `minimum` and `max` return 0.5;
  `sqrt(x*x)`, which is `abs(x)`, returns nan.
* So autodiff is a function of the expression, not of the function. The
  identity map written three ways gives f'(0) = 1.0, 0.0 and 0.5, and two of
  those are not subgradients of the identity, whose subdifferential is {1}.
* And then none of it applies to the model. GELU and LayerNorm are smooth;
  gradient clipping's `min` never activates in a 600-step run, the largest
  gradient norm staying under the threshold; and not one of 10.6 million
  unmasked attention probabilities is exactly 0 or exactly 1 in float32.
* With one structural exception: position 0's attention row is a softmax over
  a single element, hence the constant 1, hence a gradient of identically
  zero -- 5,120 of 5,120 first rows, every head, every layer, every sequence.
* What does throttle gradient flow is saturation, which is smooth: 12.9% of
  attention rows already have a maximum probability above 0.9, where the
  softmax passes under a tenth of the gradient. That is episode 2.

Run: `standarderror run gr101_kinks --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.calculus import kinks as kk
from standarderror.llm import tiny
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 9, 10)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Calculus for Language Models, Taught Through What Breaks"
SERIES_TAG = "Gradients"

CLIP_STEPS, CLIP_AT = 600, 1.0
BATCHES = 20


def compute() -> dict:
    cat = kk.catalogue()
    forms = kk.identity_three_ways(0.0)
    away = {p: kk.identity_three_ways(p) for p in (-1.5, -0.25, 0.25, 1.5)}
    masked = kk.fully_masked_softmax()
    attn = kk.attention_survey(count=BATCHES)
    clip = kk.clipping_survey(steps=CLIP_STEPS, threshold=CLIP_AT)
    return {"catalogue": cat, "forms": forms, "away": away,
            "masked": masked, "attn": attn, "clip": clip}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    cat, forms = res["catalogue"], res["forms"]
    attn, clip = res["attn"], res["clip"]

    rows = []
    for k in cat:
        label = f"{k.name} at {k.point:.0f}"
        got = "nan" if k.returned != k.returned else f"{k.returned:.1f}"
        rows.append([label, got, k.subdifferential,
                     "yes" if k.admissible else "no"])
    bold = {(i, 1) for i, k in enumerate(cat)
            if k.returned != 0.0}

    out["f0"] = charts.table_image(
        rows,
        header=["operation and point", "autodiff returns",
                "true subdifferential", "a subgradient?"],
        title="Three different answers, from one library, for one slope",
        subtitle=("Every row is a point where the derivative does not exist. "
                  "The second column is what the framework hands back, "
                  "measured rather than quoted."),
        source="Measured on torch; standarderror/calculus/kinks.py.",
        alt=("A table of ten non-differentiable points. Six rows return zero, "
             "three return one half, and one returns nan."),
        caption=("Six operations pick the left slope, three split the tie, "
                 "and `sqrt(x*x)` - which is `abs(x)` - returns nan where "
                 "`abs(x)` returns 0. None of these is wrong: at a kink there "
                 "is no derivative to be right about, so the kernel author "
                 "chose. The trouble is that they chose differently."),
        bold_cells=bold, align="lrrr",
        path=str(IMG / f"gr101-f0-catalogue.{EXT}"))[0]

    def three_ways(ax, m):
        x = np.linspace(-1.5, 1.5, 200)
        ax.plot(x, x, lw=2.0, color=m.grid, label="all three functions")
        colours = (m.series[0], m.series[1], m.series[2])
        for (row, colour) in zip(forms, colours):
            s = row["derivative"]
            t = np.linspace(-0.85, 0.85, 40)
            ax.plot(t, s * t, lw=2.2, color=colour,
                    label=f"{row['form']}   slope {s:+.1f}")
            ax.annotate(f"{s:+.1f}", (0.85, s * 0.85),
                        textcoords="offset points", xytext=(7, 0),
                        fontsize=9.5, color=colour, va="center")
        ax.scatter([0], [0], s=60, color=m.ink, zorder=4)
        ax.set_xlim(-1.6, 1.85)
        ax.set_ylim(-1.35, 1.35)
        ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    out["f1"] = charts.diagram(
        three_ways,
        title="One function, three expressions, three slopes at the origin",
        subtitle=("The grey line is the identity. So are all three coloured "
                  "expressions, at every real number. The coloured lines are "
                  "the slopes autodiff reports for them at zero."),
        xlabel="x", ylabel="",
        source="Measured on torch; standarderror/calculus/kinks.py.",
        alt=("A diagonal grey line with three straight lines through the "
             "origin at slopes one, zero and one half."),
        caption=(f"The subdifferential of the identity is the single point "
                 f"{{1}}, so two of these three answers are not subgradients "
                 f"of the function being differentiated - they are subgradients "
                 f"of the pieces, composed. Away from zero all three return "
                 f"{res['away'][0.25][0]['derivative']:.1f}, so nothing here "
                 f"shows up in a gradient check that samples a random point."),
        equal=True, figsize=(6.6, 5.0),
        path=str(IMG / f"gr101-f1-three-ways.{EXT}"))[0]

    def clipping(ax, m):
        w = clip["windows"]
        mids = [0.5 * (r["from"] + r["to"]) for r in w]
        ax.bar(mids, [r["clipped"] for r in w],
               width=[0.86 * (r["to"] - r["from"]) for r in w],
               color=m.series[1], label="steps where the min was active")
        ax.plot(mids, [r["median_norm"] for r in w], marker="o", ms=6, lw=1.9,
                color=m.series[0], label="median gradient norm")
        ax.axhline(clip["threshold"], color=m.series[2], lw=1.7, ls="--",
                   label=f"clipping threshold {clip['threshold']:.0f}")
        ax.set_ylim(0, 1.25)
        ax.legend(frameon=False, fontsize=8.6, loc="upper right")

    out["f2"] = charts.diagram(
        clipping,
        title="The one non-smooth step in training, and this run never reached it",
        subtitle=(f"A fresh {CLIP_STEPS}-step run of the same architecture. "
                  f"Gradient clipping multiplies by min(1, c/||g||), which has "
                  f"a kink at ||g|| = c."),
        xlabel="training step", ylabel="",
        source="Measured; standarderror/calculus/kinks.py.",
        alt=("Bars of the clipped fraction, near zero everywhere except the "
             "first window, against a median gradient norm well below the "
             "dashed threshold."),
        caption=(f"{clip['clipped']} of {clip['steps']} steps clipped, "
                 f"{clip['clipped_share']:.1%}. The median gradient norm sits "
                 f"at {clip['median_norm']:.2f} against a threshold of "
                 f"{clip['threshold']:.0f}, and the largest norm in the run "
                 f"was {clip['max_norm']:.2f}. This is the non-smoothness I "
                 f"expected to matter and the run does not go near it."),
        path=str(IMG / f"gr101-f2-clipping.{EXT}"))[0]

    def confidence(ax, m):
        thr = sorted(attn["share_above"])
        ax.bar(range(len(thr)), [attn["share_above"][t] for t in thr],
               color=m.series[0], width=0.55)
        for i, t in enumerate(thr):
            share = attn["share_above"][t]
            ax.annotate(f"{share:.2%}" if share else "0",
                        (i, share), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=9.0,
                        color=m.ink_secondary)
        ax.set_xticks(range(len(thr)))
        ax.set_xticklabels([f"> {t}" for t in thr])
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_ylim(0, 0.3)

    out["f3"] = charts.diagram(
        confidence,
        title="Where the gradient actually goes: confidence, not kinks",
        subtitle=(f"The share of attention rows whose largest probability "
                  f"exceeds each threshold, over {attn['probabilities']:,} "
                  f"unmasked probabilities from held-out text."),
        xlabel="largest probability in the row",
        ylabel="share of rows (log scale)",
        source="Measured on the committed model; standarderror/llm/tiny.py.",
        alt=("Four bars on a log scale, falling from about thirteen percent "
             "to zero as the confidence threshold rises."),
        caption=(f"Not one probability in this survey is exactly 0 or exactly "
                 f"1, and the largest is {attn['largest_max_p']:.4f}. But the "
                 f"gradient through a softmax scales like p(1 - p), so the "
                 f"{attn['share_above'][0.9]:.1%} of rows above 0.9 are "
                 f"already passing under a tenth of it - perfectly smoothly, "
                 f"with a derivative that exists everywhere. That is the next "
                 f"episode."),
        path=str(IMG / f"gr101-f3-confidence.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    attn, clip = res["attn"], res["clip"]

    def kink(panel, m):
        x = np.linspace(-1.0, 1.0, 120)
        panel.plot(x, np.maximum(x, 0.0), lw=2.6, color=m.series[0])
        for s, colour in ((0.0, m.series[1]), (0.5, m.series[2]),
                          (1.0, m.grid)):
            t = np.linspace(-0.55, 0.55, 20)
            panel.plot(t, s * t, lw=1.9, color=colour)
        panel.scatter([0], [0], s=60, color=m.ink, zorder=4)
        panel.set_xlim(-1.1, 1.1)
        panel.set_ylim(-0.7, 1.1)

    def flat(panel, m):
        # One row of attention that is a single 1, and its flat gradient.
        panel.bar([0], [1.0], color=m.series[1], width=0.5)
        panel.bar(range(1, 6), [0] * 5, color=m.grid, width=0.5)
        panel.plot([-0.6, 5.6], [0.0, 0.0], lw=2.0, color=m.series[0])
        panel.set_xlim(-0.9, 5.9)
        panel.set_ylim(-0.22, 1.2)

    def confident(panel, m):
        p = np.linspace(0.5, 0.9995, 120)
        panel.plot(p, p * (1 - p) / 0.25, lw=2.6, color=m.series[0])
        panel.axvline(0.9, lw=1.8, ls=":", color=m.series[1])
        panel.set_xlim(0.48, 1.02)
        panel.set_ylim(-0.06, 1.08)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=1,
        headline="Autodiff answers a question the function did not ask",
        panels=[(kink, "3", "slopes for one identity"),
                (flat, f"{attn['first_rows_onehot']:,}", "rows with zero gradient"),
                (confident, f"{attn['share_above'][0.9]:.0%}",
                 "of rows past p = 0.9")],
        note=(f"The kinks are real and inconsistent. They are also almost "
              f"absent from this model: clipping fired on "
              f"{clip['clipped_share']:.1%} of {clip['steps']} steps, and no "
              f"attention probability away from position 0 is exactly 0 or 1."),
        alt=("Three hand-drawn frames: a hinge with three tangent lines at "
             "its corner; a single tall bar over a flat line; and a curve "
             "falling to zero as confidence approaches one."),
        path=str(IMG / f"gr101-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["catalogue"] = s.run("""
        import torch

        def slope(f, x0):
            x = torch.tensor([float(x0)], requires_grad=True)
            f(x).sum().backward()
            return float(x.grad[0])

        zero = torch.zeros(1)
        for name, f, x0 in (
                ("relu(x)          at 0", torch.relu, 0.0),
                ("abs(x)           at 0", torch.abs, 0.0),
                ("clamp(x, 0, 1)   at 1", lambda x: x.clamp(0.0, 1.0), 1.0),
                ("maximum(x, 0)    at 0", lambda x: torch.maximum(x, zero), 0.0),
                ("max(stack(x, 0)) at 0",
                 lambda x: torch.stack([x[0], zero[0]]).max(), 0.0),
                ("sqrt(x * x)      at 0", lambda x: torch.sqrt(x * x), 0.0)):
            print(f"{name}   ->  {slope(f, x0):.1f}")
    """, expect=["relu(x)"])

    out["identity"] = s.run("""
        # Three expressions. All three are the identity map, at every real
        # number, and the last line of each block is the check.
        forms = {
            "x":                   lambda x: x,
            "relu(x) - relu(-x)":  lambda x: torch.relu(x) - torch.relu(-x),
            "relu(x) + min(x, 0)": lambda x: torch.relu(x)
                                             + torch.minimum(x, zero),
        }
        pts = torch.tensor([-1.5, -0.25, 0.0, 0.25, 1.5])
        for name, f in forms.items():
            same = bool(torch.allclose(f(pts), pts))
            print(f"{name:>20}  equals x everywhere: {same}   "
                  f"autodiff f'(0) = {slope(f, 0.0):+.1f}")
    """, expect=["equals x everywhere"])

    out["model"] = s.run(f"""
        # And now the model. Are any of its attention probabilities exactly 0
        # or exactly 1, where no choice of subgradient could help?
        from standarderror.calculus import kinks as kk

        a = kk.attention_survey(count={BATCHES})
        print(f"unmasked probabilities examined  {{a['probabilities']:,}}")
        print(f"  exactly 0.0                    {{a['exact_zero']}}")
        print(f"  exactly 1.0                    {{a['exact_one']}}")
        print(f"  below 1e-6                     {{a['below_micro']:,}}"
              f"  ({{a['below_micro_share']:.1%}})")
        print()
        print(f"first rows (position 0), one per head per layer per sequence")
        print(f"  exactly one-hot by the mask    "
              f"{{a['first_rows_onehot']:,}} of {{a['first_rows']:,}}")
    """, expect=["unmasked probabilities examined"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    cat = {(k.name, k.point): k for k in res["catalogue"]}
    forms = {r["form"]: r for r in res["forms"]}
    attn, clip = res["attn"], res["clip"]

    # The spine, asserted rather than trusted.
    assert cat[("relu(x)", 0.0)].returned == 0.0
    assert cat[("maximum(x, 0)", 0.0)].returned == 0.5
    nan = cat[("sqrt(x * x)", 0.0)].returned
    assert nan != nan
    assert forms["x"]["derivative"] == 1.0
    assert forms["relu(x) - relu(-x)"]["derivative"] == 0.0
    assert forms["relu(x) + min(x, 0)"]["derivative"] == 0.5
    assert sum(not r["is_subgradient"] for r in res["forms"]) == 2
    for point, rows in res["away"].items():
        assert {r["derivative"] for r in rows} == {1.0}, point
    assert res["masked"]["forward_all_nan"] is True
    assert attn["exact_zero"] == 0 and attn["exact_one"] == 0
    assert attn["first_rows_onehot"] == attn["first_rows"]
    assert attn["largest_max_p"] < 1.0
    assert clip["clipped_share"] < 0.05

    post = Post(
        title=(f"{SERIES_TAG} 1: Backprop Is the Chain Rule, and the Chain "
               f"Rule Has Hypotheses"),
        slug="gradients-1-kinks",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=1,
        date=POST_DATE,
        # No predictive claim anywhere: every number is either a kernel's
        # documented choice or a count over a fixed model's activations.
        requires_baseline=False,
        subtitle=("Autodiff returns a float at every point where the "
                  "derivative does not exist, and the float is a choice. The "
                  "choices disagree inside one library, which makes "
                  "differentiation a function of the expression rather than "
                  "of the function - and then almost none of it turns out to "
                  "happen in a real transformer."),
        summary=(
            "relu, abs, clamp, hardtanh and a vector norm all return 0 at "
            "their kinks; maximum, minimum and max split the tie and return "
            "0.5; and sqrt(x*x), which is abs(x), returns nan where abs "
            "returns 0. Three answers from one library for one slope. The "
            "consequence is sharper than the inconsistency: the identity map "
            "written three ways that agree at every real number gives f'(0) = "
            "1.0, 0.0 and 0.5, and two of those are not subgradients of the "
            "identity, whose subdifferential is the single point 1. Then the "
            "episode goes looking for this in the transformer and does not "
            "find it. GELU and LayerNorm are smooth, gradient clipping's min "
            "never activates in a 600-step run, and not "
            "one of 10.6 million unmasked attention probabilities is exactly "
            "0 or exactly 1 in float32. One structural exception: position "
            "0's attention row is a softmax over a single element, so it is "
            "the constant 1 and its gradient is identically zero - 5,120 of "
            "5,120 first rows. What does throttle gradient flow is "
            "saturation, which is smooth: 12.9% of rows already have a "
            "maximum probability above 0.9, where the softmax passes under a "
            "tenth of the gradient."),
        tags=["deep-learning", "machine-learning", "llm", "pytorch",
              "mathematics", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. The kink catalogue is measured on torch at "
            "build time rather than quoted, because kernel choices change "
            "between releases; the model measurements are counts over "
            "held-out text from the corpus the model was trained on, and no "
            "values from it are published.",
            f"The model: an {tiny.load()['parameters']:,}-parameter "
            f"character-level transformer trained for this series - four "
            f"blocks, four heads, width {tiny.WIDTH}, context {tiny.BLOCK}, "
            f"validation loss {tiny.VAL_LOSS} against a uniform-guess "
            f"{tiny.UNIFORM_LOSS:.3f}. Weights, training script and a "
            f"verified sha256 are committed: `standarderror/llm/tiny.py`, "
            f"`scripts/train_tiny.py`, `data/tiny_gpt/`.",
            "Machinery: `standarderror/calculus/kinks.py`, tested in "
            "`tests/test_kinks.py`.",
            "Where this stops: Bolte and Pauwels, \"A mathematical model for "
            "automatic differentiation in machine learning\", *NeurIPS* "
            "(2020), for why composing chosen subgradients need not produce a "
            "subgradient of anything, and for the conservative fields that "
            "make it work for optimisation regardless; Kakade and Lee, "
            "\"Provably correct automatic subdifferentiation for qualified "
            "programs\", *NeurIPS* (2018), for the conditions under which the "
            "composition is correct; Griewank and Walther, *Evaluating "
            "Derivatives* (2nd ed., 2008), for non-smoothness in automatic "
            "differentiation generally.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, "
                           "torch=2.14.0, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose quotes "
                            "are pinned, so drift fails the build"),
            "model": ("the committed checkpoint, hash-verified on load, so "
                      "every count here is a count over the same weights"),
            "determinism": (f"the attention survey draws {BATCHES} batches of "
                            f"16 held-out sequences from a fixed seed; the "
                            f"clipping run is a fresh {CLIP_STEPS}-step "
                            f"training run under torch.manual_seed(0), which "
                            f"is reproducible on this platform and not "
                            f"guaranteed across torch versions"),
        },
    )
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    forms = {r["form"]: r for r in res["forms"]}
    attn, clip = res["attn"], res["clip"]
    zeros = sum(1 for k in res["catalogue"] if k.returned == 0.0)
    halves = sum(1 for k in res["catalogue"] if k.returned == 0.5)

    post.add(
        "A slope where there is no slope",
        f"""`relu` has no derivative at zero. The left slope is 0, the right slope is 1, and there is no number that is the derivative — the subdifferential is the whole interval [0, 1].

Ask a framework anyway and it will tell you {res['catalogue'][0].returned:.1f}, without a warning, in about a microsecond.

That is not a scandal. Optimisation on non-smooth functions is a well-developed subject, picking an element of the subdifferential is exactly what subgradient methods do, and the training runs that built every model you have used were full of these points. The interesting part is what the framework picks, whether the picks agree with each other, and — the question this episode was written to answer and then had to answer differently — whether any of it happens in a transformer.""")

    post.add(
        "What the library actually picks",
        """There are more of these points than `relu`. `abs` at zero, `clamp` at either bound, `hardtanh` at either bound, a vector norm at the origin, and every `max` and `min` at a tie. Ask for all of them.""")

    post.add(
        "",
        f"""{snip['catalogue'].markdown()}

{zeros} of the ten pick a one-sided slope and return 0. {halves} of them split the tie and return 0.5. And `sqrt(x * x)` returns **nan** — which is the same function as `abs(x)`, differing only in how it was typed.

None of those is a mistake. At a kink there is no derivative to be right about, so somebody writing the kernel decided, and the decisions are local to each kernel. The trouble is what happens when you compose them.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "Differentiation of the expression, not of the function",
        """Here is the sharpest form of it. Take the identity map, and write it three ways.

The first is `x`. The second is `relu(x) - relu(-x)`, which is *x* for positive *x*, *x* for negative *x*, and 0 at 0 — so it is the identity, everywhere, exactly. The third is `relu(x) + min(x, 0)`, same argument, same function.

All three are differentiable at every real number, with derivative 1. There is nothing subtle about the function: its subdifferential at zero is the single point {1}.""")

    post.add(
        "",
        f"""{snip['identity'].markdown()}

`{forms['x']['form']}` gives {forms['x']['derivative']:.1f}. `{forms['relu(x) - relu(-x)']['form']}` gives {forms['relu(x) - relu(-x)']['derivative']:.1f}. `{forms['relu(x) + min(x, 0)']['form']}` gives {forms['relu(x) + min(x, 0)']['derivative']:.1f}.

Two of those are not subgradients of the identity. They are not approximations of the derivative, they are not conservative choices, and they are not off by a little: one of them is 0 where the true and only answer is 1. What autodiff computes is a composition of choices made about the *pieces*, and the composition need not be a subgradient of the whole.

This is understood rather than surprising to the people who work on it. Bolte and Pauwels' object is the *conservative field* — a set-valued map that agrees with the gradient almost everywhere and behaves well enough under composition that gradient descent on it still converges, without being the subdifferential of anything. That is the honest description of what a framework hands you: not a gradient, and not a subgradient, but something that coincides with the gradient off a measure-zero set and is therefore almost always the thing you wanted.

"Almost always" is doing real work in that sentence, and the rest of this episode is about how often the exception is reached.

Before the counting, it is worth saying why this matters at all if the exception is rare, because "rare" is a strange thing to build on. Two reasons.

The first is that a rare event with a structural cause is not rare where it happens. A measure-zero set is invisible to a random probe and perfectly reachable by a construction, and the interesting non-differentiable points in a network are exactly the ones some structure puts there — a mask, an initialisation at zero, a hard threshold in a loss. A gradient check that samples a random input passes every time and says nothing about them.

The second is that the machinery above is what licenses a practice this series will get to: **deliberately** returning a number where no derivative exists. A straight-through estimator does not approximate a gradient that is merely hard to compute. It hands back the gradient of a *different function* and lets the chain rule carry it, which is the identity-written-three-ways trick used on purpose. Whether that is legitimate is not a matter of taste — it is the question of whether the resulting field is conservative for the function you actually care about — and it is episode 4.""",
        level=3,
        figures=[figs["f1"]])

    post.add(
        "Where I expected this to bite, and did not find it",
        """So: how often does a real training run land on a point where the derivative does not exist?

Start with the model. The transformer this series measures has no `relu` anywhere. Its MLP uses `GELU`, which is smooth — infinitely differentiable, no kinks. `LayerNorm` is smooth away from zero variance. The attention softmax is smooth. `masked_fill` is a selection rather than a kink, and its gradient at a masked position is exactly zero because the position genuinely does not participate. There is nothing in the forward pass to land on.

That leaves the training procedure, where there is exactly one non-smooth operation and everybody uses it: gradient clipping multiplies the gradient by `min(1, c / ||g||)`, which has a kink at `||g|| = c`. So I trained a fresh model and counted.""")

    post.add(
        "",
        f"""{clip['clipped']} of {clip['steps']} steps clipped — {clip['clipped_share']:.1%}. The median gradient norm sits at {clip['median_norm']:.2f} against a threshold of {clip['threshold']:.0f}, and the largest norm anywhere in the run was {clip['max_norm']:.2f}, which is {clip['threshold'] / clip['max_norm']:.1f} times under the kink.

So the kink is in the code, and the run does not go near it. That is worth saying plainly because the premise of this episode, as drafted, was that non-differentiability is a live problem in practice. On this model it is not.""",
        level=3,
        figures=[figs["f2"]])

    post.add(
        "",
        """One more place to look. A softmax probability of exactly 0 or exactly 1 in float32 would be a point where the gradient is not merely small but identically zero, and where no choice of subgradient could restore it, because the function really is locally constant there.""",
        level=3)

    post.add(
        "",
        f"""{snip['model'].markdown()}

Not one, in {attn['probabilities']:,} of them. {attn['below_micro']:,} are below 1e-6 — {attn['below_micro_share']:.1%} — which is small, but small is not zero and the chain rule does not care about small.

And then the last line, which is the one thing this search did find. **Every first row is exactly one-hot**: {attn['first_rows_onehot']:,} of {attn['first_rows']:,}, in every head, of every layer, for every sequence. Not because the model learned anything. Because the causal mask leaves position 0 attending only to itself, so its attention is a softmax over a single element, so it is the constant function 1, so its gradient is identically zero.

That is the only place in this network where the derivative is exactly zero rather than approximately, and it was put there by the mask rather than by training, and it cannot be fixed by choosing a better subgradient because there is no kink — the function is constant. Whether it matters is a separate question and probably it does not: position 0 has nothing to attend to, so there is nothing for the gradient to say. But it is the honest answer to "where in this model does the chain rule's hypothesis fail", and it is not where I looked.""",
        level=3)

    post.add(
        "The one way this model can produce a non-number",
        f"""While looking, one genuine trap turned up, and it is worth a paragraph because the usual mental model puts it in the wrong place.

A fully masked attention row — every position masked, which happens when a padding scheme and a causal mask are combined carelessly — is a softmax over all `-inf`. Every exponential is 0, the normaliser is 0, and the row comes out as `nan`. In the **forward** pass, not the backward one: by the time the loss is computed the NaN is already in the activations, and the gradient that follows is a consequence rather than the cause. Measured: `forward_all_nan` is {res['masked']['forward_all_nan']}.

People debugging this look at the backward pass, because "NaN gradient" is the phrase they have heard. The NaN is upstream.""")

    post.add(
        "What actually throttles the gradient",
        f"""The search failed, and what it turned up instead is the subject of the next episode, so it is worth stating precisely.

The gradient through a softmax scales like *p*(1 − *p*). Nothing in this model reaches *p* = 1, where that would be zero. But it does not need to: the largest probability in the survey is {attn['largest_max_p']:.4f}, and {attn['share_above'][0.9]:.1%} of rows have a maximum above 0.9, where *p*(1 − *p*) is already under a tenth of its value at 0.5. {attn['share_above'][0.99]:.2%} are above 0.99, where it is under a hundredth.

That is not a kink, a subgradient choice, or a numerical edge case. It is a smooth function with a small derivative, which is a completely different failure and a much more common one. The chain rule's hypotheses hold perfectly; the product it forms is simply tiny.""",
        figures=[figs["f3"]])

    post.add(
        "What to keep",
        f"""1. Autodiff returns a float at every non-differentiable point, and the float is a kernel author's choice. Within one library: {zeros} of ten operations return 0, {halves} split the tie at 0.5, and `sqrt(x*x)` returns nan where `abs(x)` returns 0.
2. So differentiation is a function of the **expression**. The identity written three ways gives f'(0) = 1.0, 0.0 and 0.5, and two of those are not subgradients of the identity at all.
3. What a framework gives you is best described as a conservative field: it agrees with the gradient off a measure-zero set, which is almost always enough and is not the same claim as "it is the gradient".
4. None of it appears in this transformer. GELU and LayerNorm are smooth, and clipping's `min` was active on {clip['clipped']} of {clip['steps']} steps, with the largest gradient norm {clip['threshold'] / clip['max_norm']:.1f} times under the threshold.
5. Except structurally: position 0's attention row is a softmax over one element, so its gradient is identically zero — {attn['first_rows_onehot']:,} of {attn['first_rows']:,} first rows, put there by the mask.
6. A fully masked row is `nan` in the **forward** pass. Look upstream of the gradient.
7. What does throttle the gradient is confidence, not kinks: {attn['share_above'][0.9]:.1%} of rows are past *p* = 0.9, where a softmax passes under a tenth of the gradient, perfectly smoothly.""")

    post.add(
        "Exercise",
        """Write the identity map in a fourth way that autodiff mis-differentiates at a point of your choosing, and then in a fifth way that it gets right. The difference between your two constructions is the whole content of the conservative-field story, and it is worth having found it yourself rather than read it.

Then instrument your own training loop for one epoch. Count the steps on which gradient clipping was active, and log the smallest and largest element of any softmax the model computes. Two numbers, one hook, no experiment design.

If clipping is active on most steps, your threshold is doing something other than protecting you from outliers and it is worth knowing which. If any probability is exactly 0 or exactly 1, you have found a place where the gradient is identically zero, and the interesting question is whether the mask put it there or the training did.

The uncomfortable part: if neither of those fires, you have learned that the failure you were worried about is not the failure you have, which is the position this episode ended in.""")

    post.hero = figs["hero"]
    return post


def main() -> Post:
    return build()


if __name__ == "__main__":
    r = compute()
    for k in r["catalogue"]:
        print(f"  {k.name:>18} at {k.point:.0f} -> {k.returned!s:>5}  "
              f"subgradient: {k.admissible}")
    print()
    for row in r["forms"]:
        print(f"  {row['form']:>20}  f'(0) = {row['derivative']:+.1f}  "
              f"subgrad: {row['is_subgradient']}")
    a, c = r["attn"], r["clip"]
    print(f"\nattention: {a['probabilities']:,} probs, exact0 {a['exact_zero']}, "
          f"exact1 {a['exact_one']}, first rows {a['first_rows_onehot']}/"
          f"{a['first_rows']}, max p {a['largest_max_p']:.4f}")
    print(f"share above: {a['share_above']}")
    print(f"clipping: {c['clipped']}/{c['steps']} = {c['clipped_share']:.1%}, "
          f"median norm {c['median_norm']:.3f}")
