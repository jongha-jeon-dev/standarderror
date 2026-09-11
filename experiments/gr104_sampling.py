"""Gradients 4: You Cannot Differentiate Through a Sampled Token.

The fourth episode. The first three were about hypotheses of the chain rule
that a transformer bends; this one is about a place where there is no
derivative to bend, and about the three things people return instead.

Measured, on a decision small enough to enumerate exactly:

* The gradient that does exist is of the expectation: dL/dz = p * (loss - E
  loss), which is episode 2's softmax Jacobian applied to the vector of
  losses, and which sums to exactly zero by episode 3's argument.
* REINFORCE is unbiased, and measured so: its relative bias sits at or under
  its own Monte Carlo error in every context. A mean baseline cuts the
  per-sample spread by about a third and changes nothing else.
* Straight-through is biased by 0.67 to 0.97 of the exact gradient's norm,
  which is ten to a hundred times its own Monte Carlo error, so it does not
  average away. Its expected gradient is 4% to 40% of the right size.
* Its cosine with the exact gradient is 0.79 to 0.94, which looks like a
  defence until the same softmax Jacobian applied to *noise* scores a 90th
  percentile of 0.78 to 0.97. The direction is borrowed from a factor the two
  share, not earned.
* What it assumes is that the loss is linear in embedding space between the
  drawn token and the alternatives. Within a context that extrapolation
  correlates with the truth at 0.06 to 0.54, median 0.11. Weak.
* Pooling those correlations gives -0.17 and an earlier draft of this episode
  said "anti-correlated". It is a Simpson's paradox and the episode says so.
* The trade is computable: REINFORCE with a baseline overtakes
  straight-through on total error at 2 to 9 samples.
* Gumbel-softmax trades one for the other monotonically until about tau=0.5,
  after which the bias stops improving and only the variance grows.

Run: `standarderror run gr104_sampling --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.calculus import sampling as sm
from standarderror.llm import tiny
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 9, 11)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Calculus for Language Models, Taught Through What Breaks"
SERIES_TAG = "Gradients"

CONTEXTS = 6
ST_DRAWS, RF_DRAWS = 400, 6000
TAUS = (2.0, 1.0, 0.5, 0.2, 0.1)


def compute() -> dict:
    x, y = next(iter(tiny.batches(count=1, size=CONTEXTS, seed=0)))
    dec = sm.decision(x[0, :tiny.BLOCK - 1], int(y[0, tiny.BLOCK - 1]))
    return {
        "survey": sm.survey(contexts=CONTEXTS, st_draws=ST_DRAWS,
                            rf_draws=RF_DRAWS),
        "dec": dec,
        "st": sm.pathwise(dec, sm.straight_through("softmax"),
                          draws=ST_DRAWS),
        "st_logits": sm.pathwise(dec, sm.straight_through("logits"),
                                 draws=ST_DRAWS),
        "rf": sm.reinforce(dec, draws=RF_DRAWS),
        "rf_base": sm.reinforce(dec, draws=RF_DRAWS,
                                baseline=dec["expected"]),
        "taus": sm.temperature_sweep(dec, taus=TAUS, draws=ST_DRAWS),
        "control": sm.jacobian_control(dec),
        "lin": sm.linearisation(dec),
        "baselines": _baselines(dec),
    }


def _baselines(dec) -> dict:
    """None, the mean, and the variance-optimal constant, all measured."""
    b = sm.optimal_baseline(dec)
    out = {"mean": b["mean"], "optimal": b["optimal"],
           "tilt": b["weights_tilt"]}
    for name, value in (("none", None), ("at_mean", b["mean"]),
                        ("at_optimal", b["optimal"])):
        out[name] = sm.reinforce(dec, draws=20_000, baseline=value)
    return out


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    survey, dec = res["survey"], res["dec"]
    st, stl = res["st"], res["st_logits"]
    rf, rfb, taus = res["rf"], res["rf_base"], res["taus"]
    lin = res["lin"]

    rows = [
        ["REINFORCE", f"{rf['relative_bias']:.3f}", f"{rf['scale']:.2f}",
         f"{rf['cosine']:.3f}", f"{rf['sd']:.2f}"],
        ["REINFORCE, mean baseline", f"{rfb['relative_bias']:.3f}",
         f"{rfb['scale']:.2f}", f"{rfb['cosine']:.3f}", f"{rfb['sd']:.2f}"],
        ["straight-through", f"{st['relative_bias']:.3f}",
         f"{st['scale']:.2f}", f"{st['cosine']:.3f}", f"{st['sd']:.2f}"],
        ["straight-through on logits", f"{stl['relative_bias']:.3f}",
         f"{stl['scale']:.2f}", f"{stl['cosine']:.3f}", f"{stl['sd']:.2f}"],
    ] + [[f"gumbel-softmax, tau = {t['tau']:g}", f"{t['relative_bias']:.3f}",
          f"{t['scale']:.2f}", f"{t['cosine']:.3f}", f"{t['sd']:.2f}"]
         for t in taus]
    bold = {(2, c) for c in range(5)}

    out["f0"] = charts.table_image(
        rows,
        header=["estimator", "relative bias", "scale", "cosine",
                "per-draw sd"],
        title="Seven estimators, one decision, and an answer to compare them to",
        subtitle=(f"The expectation over {dec['vocab']} tokens is enumerated, "
                  f"so the exact gradient is known. Bias and scale are of "
                  f"the average; the spread is per draw."),
        source="Measured; standarderror/calculus/sampling.py.",
        alt=("A table of seven estimators. The two REINFORCE rows have bias "
             "near zero and large spread; the straight-through rows have "
             "large bias and small spread."),
        caption=(f"The two REINFORCE rows are unbiased - their measured bias "
                 f"of {rfb['relative_bias']:.3f} sits at their own Monte "
                 f"Carlo error of {rfb['mc_error']:.3f} - and pay for it in "
                 f"spread. **Straight-through** is biased by "
                 f"{st['relative_bias']:.2f} of the exact gradient's norm, "
                 f"{st['relative_bias'] / st['mc_error']:.0f} times its own "
                 f"noise, so no number of draws removes it. Note the scale "
                 f"column: it returns {st['scale']:.0%} of the right size, "
                 f"and the same estimator written on the logits returns "
                 f"{stl['scale']:.1f} times too much."),
        bold_cells=bold, align="lrrrr",
        path=str(IMG / f"gr104-f0-estimators.{EXT}"))[0]

    def linear(ax, m):
        ax.scatter(lin["predicted"], lin["true"], s=26, alpha=0.75,
                   color=m.series[0], linewidths=0,
                   label="one alternative token")
        lo = float(min(lin["predicted"].min(), lin["true"].min()))
        hi = float(max(lin["predicted"].max(), lin["true"].max()))
        ax.plot([lo, hi], [lo, hi], lw=1.8, ls="--", color=m.grid,
                label="what straight-through assumes")
        b, a = np.polyfit(lin["predicted"], lin["true"], 1)
        xs = np.linspace(lin["predicted"].min(), lin["predicted"].max(), 20)
        ax.plot(xs, b * xs + a, lw=2.0, color=m.series[2],
                label=f"what happens, slope {b:.2f}")
        ax.legend(frameon=False, fontsize=8.6, loc="upper left")
        ax.set_aspect("equal", adjustable="box")

    out["f1"] = charts.diagram(
        linear,
        title="The extrapolation straight-through is built on",
        subtitle=("Its backward pass carries a first-order estimate of what "
                  "each other token would have cost. Against what each other "
                  "token actually cost."),
        xlabel="linear extrapolation of the change in loss",
        ylabel="actual change in loss",
        source="Measured; standarderror/calculus/sampling.py.",
        alt=("A scatter with no strong relationship, a dashed diagonal for "
             "the assumption and a shallower fitted line through the "
             "points."),
        caption=(f"If the loss were linear in embedding space these points "
                 f"would sit on the dashed line. They correlate at "
                 f"{lin['correlation']:+.2f} - weak but positive. The "
                 f"sharper fact is the **compression**: the true changes "
                 f"span {lin['true'].max() - lin['true'].min():.0f} nats and "
                 f"the extrapolation spans "
                 f"{lin['predicted'].max() - lin['predicted'].min():.0f}, a "
                 f"spread ratio of {lin['spread_ratio']:.2f}. A loss model "
                 f"that says nearly the same thing about every alternative "
                 f"gives nearly no gradient, because a constant is the "
                 f"softmax Jacobian's null direction."),
        path=str(IMG / f"gr104-f1-linearisation.{EXT}"))[0]

    def budget(ax, m):
        ns = np.logspace(0, 3, 40)
        ax.plot(ns, sm.error_curve(st, ns), lw=2.4, color=m.series[2],
                label="straight-through")
        ax.plot(ns, sm.error_curve(rfb, ns), lw=2.4, color=m.series[0],
                label="REINFORCE, mean baseline")
        ax.plot(ns, sm.error_curve(rf, ns), lw=1.8, ls="--",
                color=m.series[1], label="REINFORCE, no baseline")
        n = sm.crossover(st, rfb)
        ax.axvline(n, lw=1.6, ls=":", color=m.ink_secondary,
                   label=f"they cross at {n:.1f} draws")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8.6, loc="lower left")

    out["f2"] = charts.diagram(
        budget,
        title="Bias is a floor and variance is not",
        subtitle=("Expected error of an average of n draws, relative to the "
                  "exact gradient. Squared bias plus variance over n, with "
                  "both measured rather than assumed."),
        xlabel="draws averaged (log scale)",
        ylabel="relative error of the average (log scale)",
        source="Measured; standarderror/calculus/sampling.py.",
        alt=("Two falling curves: one flattens onto a floor, the other keeps "
             "descending and crosses it early."),
        caption=(f"Straight-through starts ahead because its spread is "
                 f"small, and stops improving at its bias. REINFORCE with a "
                 f"baseline keeps going. On this decision they cross at "
                 f"{sm.crossover(st, rfb):.1f} draws; across six the "
                 f"crossover runs "
                 f"{min(r['crossover'] for r in survey['rows']):.1f} to "
                 f"{max(r['crossover'] for r in survey['rows']):.1f}. If you "
                 f"can afford a handful of samples, the **bias is the "
                 f"expensive half**."),
        path=str(IMG / f"gr104-f2-budget.{EXT}"))[0]

    def temp(ax, m):
        b = [t["relative_bias"] for t in taus]
        s = [t["sd"] / t["exact_norm"] for t in taus]
        ax.plot(b, s, marker="o", ms=7, lw=2.0, color=m.series[0])
        for t, bb, ss in zip(taus, b, s):
            ax.annotate(f"  tau = {t['tau']:g}", (bb, ss), fontsize=9.0,
                        color=m.ink_secondary, va="center")
        ax.scatter([st["relative_bias"]], [st["sd"] / st["exact_norm"]],
                   s=90, marker="*", color=m.series[2], zorder=4,
                   label="straight-through")
        ax.scatter([rfb["relative_bias"]], [rfb["sd"] / rfb["exact_norm"]],
                   s=70, marker="s", color=m.series[1], zorder=4,
                   label="REINFORCE, baseline")
        ax.set_xlim(0, 1.05)
        ax.legend(frameon=False, fontsize=8.6, loc="upper right")

    out["f3"] = charts.diagram(
        temp,
        title="The relaxation's trade, and where it stops paying",
        subtitle=("Each point is one Gumbel-softmax temperature. Down and "
                  "left is better; the curve turns because below a "
                  "temperature the bias no longer falls."),
        xlabel="relative bias", ylabel="per-draw spread, relative",
        source="Measured; standarderror/calculus/sampling.py.",
        alt=("A curve of points bending upward: bias falls then stalls while "
             "the spread keeps climbing, with a star at high bias and low "
             "spread and a square at low bias and high spread."),
        caption=(f"From tau = {TAUS[0]:g} down to "
                 f"{min(taus, key=lambda t: t['relative_bias'])['tau']:g} the "
                 f"bias falls from {taus[0]['relative_bias']:.2f} to "
                 f"{min(t['relative_bias'] for t in taus):.2f}; below that it "
                 f"stops falling and only the spread grows. The star is "
                 f"straight-through, which is the cheap corner, and the "
                 f"square is REINFORCE with a baseline, which is the "
                 f"honest one."),
        path=str(IMG / f"gr104-f3-temperature.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    st, rfb = res["st"], res["rf_base"]
    dec = res["dec"]

    def step(panel, m):
        # A sampled token is a step function of its logits.
        x = np.linspace(-1, 1, 200)
        panel.plot(x, np.where(x < 0, 0.15, 0.85), lw=2.8, color=m.series[0])
        panel.scatter([0], [0.15], s=40, color=m.ink, zorder=4)
        panel.set_xlim(-1.1, 1.1)
        panel.set_ylim(-0.05, 1.05)

    def bars(panel, m):
        panel.bar([0, 1], [st["scale"], 1.0], width=0.5,
                  color=[m.series[2], m.grid])
        panel.set_xlim(-0.6, 1.6)
        panel.set_ylim(0, 1.15)

    def curves(panel, m):
        ns = np.logspace(0, 3, 40)
        panel.plot(ns, sm.error_curve(st, ns), lw=2.6, color=m.series[2])
        panel.plot(ns, sm.error_curve(rfb, ns), lw=2.6, color=m.series[0])
        panel.set_xscale("log")
        panel.set_yscale("log")

    return charts.lecture_hero(
        series=SERIES_TAG, episode=4,
        headline="The gradient everyone uses here is of a different function",
        panels=[(step, "0 or undefined", "the actual derivative"),
                (bars, f"{st['scale']:.0%}", "of the right size"),
                (curves, f"{sm.crossover(st, rfb):.0f} draws",
                 "until unbiased wins")],
        note=(f"A sampled token is piecewise constant in its logits, so the "
              f"gradient that exists is of the expectation. On a decision "
              f"over {dec['vocab']} tokens that expectation is enumerable, "
              f"and straight-through comes out biased by "
              f"{st['relative_bias']:.2f} of the exact gradient - "
              f"{st['relative_bias'] / st['mc_error']:.0f} times its own "
              f"noise."),
        alt=("Three hand-drawn frames: a step function with a dot at the "
             "jump; two bars of very different height; and two falling "
             "curves, one flattening and one crossing it."),
        path=str(IMG / f"gr104-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["exact"] = s.run("""
        import numpy as np
        from standarderror.calculus import sampling as sm
        from standarderror.llm import tiny

        # One decision: sample the next token, then score the token after it.
        # 65 tokens in the vocabulary, so the expectation is enumerable.
        x, y = next(iter(tiny.batches(count=1, size=1, seed=0)))
        d = sm.decision(x[0, :tiny.BLOCK - 1], int(y[0, tiny.BLOCK - 1]))
        print(f"largest probability      {d['max_p']:.3f}")
        print(f"expected loss            {d['expected']:.4f}")
        print(f"best token would cost    {d['loss'].min():.4f}")
        print(f"worst would cost         {d['loss'].max():.4f}")
        print()
        print(f"exact gradient norm      {np.linalg.norm(d['exact']):.4f}")
        print(f"  and it sums to         {d['exact'].sum():+.1e}")
    """, expect=["largest probability"])

    out["compare"] = s.run(f"""
        # Two estimators of that one vector.
        rf = sm.reinforce(d, draws={RF_DRAWS}, baseline=d["expected"])
        st = sm.pathwise(d, sm.straight_through("softmax"), draws={ST_DRAWS})
        for name, e in (("REINFORCE + baseline", rf), ("straight-through", st)):
            print(f"{{name:<22}} bias {{e['relative_bias']:.3f}}  "
                  f"(its own noise {{e['mc_error']:.3f}})   "
                  f"scale {{e['scale']:.2f}}   spread {{e['sd']:.2f}}")
        print()
        print(f"they cross at {{sm.crossover(st, rf):.1f}} draws")
    """, expect=["REINFORCE"])

    out["control"] = s.run("""
        # The cosine looks reassuring. Is it earned? Push noise through the
        # same softmax Jacobian and see what that scores.
        c = sm.jacobian_control(d)
        print(f"straight-through cosine with the exact gradient  "
              f"{st['cosine']:.3f}")
        print(f"noise through the same Jacobian, median          "
              f"{c['median']:.3f}")
        print(f"                              90th percentile    "
              f"{c['p90']:.3f}")
        print(f"                              largest of 2,000   "
              f"{c['max']:.3f}")
    """, expect=["straight-through cosine"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    survey, dec = res["survey"], res["dec"]
    st, stl = res["st"], res["st_logits"]
    rf, rfb = res["rf"], res["rf_base"]
    ctrl = res["control"]

    # The spine, asserted rather than trusted.
    assert abs(dec["exact"].sum()) < 1e-5
    assert rfb["relative_bias"] < 3 * rfb["mc_error"]
    assert rf["relative_bias"] < 3 * rf["mc_error"]
    assert rfb["sd"] < 0.8 * rf["sd"]
    assert st["relative_bias"] > 10 * st["mc_error"]
    assert st["scale"] < 0.6
    assert stl["scale"] > 2.0
    assert ctrl["p90"] > 0.7
    assert survey["correlation_median"] > 0.0
    assert survey["pooled_correlation"] < survey["correlation_min"]
    assert 1.0 < sm.crossover(st, rfb) < 100.0

    post = Post(
        title=f"{SERIES_TAG} 4: You Cannot Differentiate Through a Sampled Token",
        slug="gradients-4-sampling",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=4,
        date=POST_DATE,
        # The exact answer is enumerated, so nothing here is a prediction to
        # be beaten -- it is a comparison against a known vector.
        requires_baseline=False,
        subtitle=("A sampled token is a step function of its logits, so the "
                  "derivative is zero almost everywhere and undefined "
                  "elsewhere. On a decision small enough to enumerate, the "
                  "three things people return instead can be compared "
                  "against the gradient that does exist."),
        summary=(
            f"The gradient of the expectation is p times loss minus its mean, "
            f"which is episode 2's softmax Jacobian applied to the losses and "
            f"sums to exactly zero. Over 65 tokens it is enumerable, so every "
            f"estimator has an answer to be wrong against. REINFORCE is "
            f"unbiased and measurably so, its bias sitting at its own Monte "
            f"Carlo error. Straight-through is biased by "
            f"{st['relative_bias']:.2f} of the exact gradient's norm - "
            f"{st['relative_bias'] / st['mc_error']:.0f} times its own noise, "
            f"so no number of draws removes it - and returns "
            f"{st['scale']:.0%} of the right size, while the same idea "
            f"written on the logits instead returns {stl['scale']:.1f} times "
            f"too much. Its cosine of {st['cosine']:.2f} with the exact "
            f"gradient looks like a defence until noise through the same "
            f"Jacobian scores a 90th percentile of {ctrl['p90']:.2f}. What it "
            f"assumes - that the loss is linear in embedding space between "
            f"the drawn token and the alternatives - correlates with the "
            f"truth at {survey['correlation_min']:+.2f} to "
            f"{survey['correlation_max']:+.2f} within a context. And the "
            f"trade is computable: REINFORCE with a mean baseline overtakes "
            f"it at "
            f"{min(r['crossover'] for r in survey['rows']):.0f} to "
            f"{max(r['crossover'] for r in survey['rows']):.0f} samples."),
        tags=["deep-learning", "machine-learning", "llm", "pytorch",
              "mathematics", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every number is a loss, a gradient or a "
            "correlation computed on held-out text from the corpus the model "
            "was trained on, and no values from that text are published.",
            f"The model: an {tiny.load()['parameters']:,}-parameter "
            f"character-level transformer trained for this series - four "
            f"blocks, four heads, width {tiny.WIDTH}, context {tiny.BLOCK}, "
            f"validation loss {tiny.VAL_LOSS} against a uniform-guess "
            f"{tiny.UNIFORM_LOSS:.3f}. Weights, training script and a "
            f"verified sha256 are committed: `standarderror/llm/tiny.py`, "
            f"`scripts/train_tiny.py`, `data/tiny_gpt/`.",
            "Machinery: `standarderror/calculus/sampling.py`, tested in "
            "`tests/test_sampling.py`, which checks the enumerated gradient "
            "against autodiff and pins the model findings as inequalities.",
            "Where this stops: Williams, \"Simple statistical "
            "gradient-following algorithms for connectionist reinforcement "
            "learning\", *Machine Learning* (1992), for REINFORCE; Bengio, "
            "Leonard and Courville, \"Estimating or propagating gradients "
            "through stochastic neurons for conditional computation\" "
            "(2013), for straight-through; Jang, Gu and Poole, \"Categorical "
            "reparameterization with Gumbel-softmax\", *ICLR* (2017) and "
            "Maddison, Mnih and Teh, \"The concrete distribution\", *ICLR* "
            "(2017), for the relaxation, published independently within days "
            "of each other.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, "
                           "torch=2.14.0, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "model": ("the committed checkpoint, hash-verified on load; the "
                      "enumerated expectation is 65 forward passes through "
                      "it and involves no sampling at all"),
            "determinism": (f"the pathwise estimators draw {ST_DRAWS} samples "
                            f"per context and REINFORCE {RF_DRAWS}, each from "
                            f"a fixed seed; the reported Monte Carlo error is "
                            f"what separates a measured bias from a real one"),
        },
    )
    post.hero = figs["hero"]
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    survey, dec = res["survey"], res["dec"]
    st, stl = res["st"], res["st_logits"]
    rf, rfb, taus = res["rf"], res["rf_base"], res["taus"]
    ctrl, lin = res["control"], res["lin"]
    bl = res["baselines"]
    best = min(taus, key=lambda t: t["relative_bias"])
    n_lo = min(r["crossover"] for r in survey["rows"])
    n_hi = max(r["crossover"] for r in survey["rows"])
    sc_lo = min(r["st"]["scale"] for r in survey["rows"])
    sc_hi = max(r["st"]["scale"] for r in survey["rows"])

    post.add(
        "There is nothing to approximate",
        """Sample a token from a distribution and ask for the derivative with respect to the logits. There is not one.

The map from logits to sample is piecewise constant. Nudge a logit by a millionth and almost surely the same token comes out; keep nudging and at some point a different one does, discontinuously. The derivative is zero almost everywhere and undefined on the boundaries, and no choice of subgradient fixes that, because the function really is locally constant — the situation episode 1 found at position 0's attention row, now on purpose and at the centre of everything anyone does with a language model after training.

What exists is the gradient of the **expectation**. And on a small enough problem, that can be computed rather than estimated, which is what makes this episode possible: there is a right answer to be wrong against.""")

    post.add(
        "The answer, enumerated",
        f"""Take one decision. Give the model {tiny.BLOCK - 1} tokens of held-out text, sample the next token from its own distribution, append it, and score the token that actually follows. Write *z* for the logits the sample comes from and *ℓ*(*t*) for the loss when token *t* is drawn. Then

$$
L(z) = \\mathbb{{E}}_{{t \\sim \\mathrm{{softmax}}(z)}}\\big[\\ell(t)\\big],
\\qquad
\\frac{{\\partial L}}{{\\partial z_j}} = p_j \\big( \\ell(j) - \\mathbb{{E}}[\\ell] \\big).
$$

That is episode 2's softmax Jacobian applied to the vector of losses, and by the same argument it sums to exactly zero. The vocabulary here is {dec['vocab']} characters, so the expectation is {dec['vocab']} forward passes rather than a sampling problem.""")

    post.add(
        "",
        f"""{snip['exact'].markdown()}

The spread of outcomes is the thing to notice: the best token costs {dec['loss'].min():.3f} and the worst {dec['loss'].max():.2f}, against an expectation of {dec['expected']:.2f}. This decision matters, and there is a definite vector saying how to change it.""",
        level=3)

    post.add(
        "Three things people return instead",
        """**REINFORCE** draws one token and returns *ℓ*(*t*)(*e*ₜ − *p*), optionally with a constant subtracted from the loss. It is unbiased for any such baseline, because the expectation of *e*ₜ − *p* is zero whatever multiplies it, and it is famous for its variance.

**Straight-through** draws a token, runs the one-hot forward, and on the way back pretends the one-hot was *p*. In code that is `onehot + p - p.detach()`, which is precisely episode 1's identity-written-three-ways trick applied deliberately: a forward pass that computes one thing and a backward pass that differentiates another.

**Gumbel-softmax** declines to sample, using a relaxed draw at temperature *τ* that is differentiable and wrong by an amount that shrinks with *τ*.""")

    post.add(
        "",
        f"""{snip['compare'].markdown()}

Both numbers in that first row are small, and the second one is why the first is meaningful: REINFORCE's measured bias of {rfb['relative_bias']:.3f} sits at its own Monte Carlo error of {rfb['mc_error']:.3f}, which is what "unbiased" looks like when you measure it rather than prove it. Straight-through's {st['relative_bias']:.2f} is {st['relative_bias'] / st['mc_error']:.0f} times its noise. **That bias is not going to average away**, and drawing more samples buys nothing against it.

The column worth staring at is the scale. Straight-through's expected gradient is {st['scale']:.0%} of the exact one — across six contexts, {sc_lo:.0%} to {sc_hi:.0%}. A run using it is taking systematically short steps. And the same idea written on the logits instead of the probabilities, `onehot + z - z.detach()`, returns {stl['scale']:.1f} times **too much** with a cosine of {stl['cosine']:.2f}. Same forward pass, same one line of intent, two derivatives that disagree by a factor of {stl['scale'] / st['scale']:.0f}.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "The cosine is borrowed",
        f"""Straight-through's average points at {st['cosine']:.2f} cosine to the exact gradient, which is usually where the defence of it begins. Before accepting that, ask what a cosine of {st['cosine']:.2f} is worth here.

Both vectors are the softmax Jacobian applied to something: the exact gradient to the true losses, straight-through to its own estimate of them. They share a factor. So push *noise* through that factor and see what it scores.""")

    post.add(
        "",
        f"""{snip['control'].markdown()}

A vector of random numbers, run through the same `p * (r - <p, r>)`, aligns with the exact gradient at a median of {ctrl['median']:.2f}, a 90th percentile of {ctrl['p90']:.2f}, and a largest-of-2,000 of {ctrl['max']:.2f}. Straight-through's {st['cosine']:.2f} sits between the 90th percentile and the best that noise managed.

That does not make it noise. It does mean the cosine is **not evidence**: the agreement comes from the factor the two share, and the part straight-through contributes itself is not visible in that number at all. Which raises the question of what that part actually is.""",
        level=3)

    post.add(
        "What it is actually assuming",
        """Differentiate the one-hot forward and you get *p* ⊙ (*u* − ⟨*p*, *u*⟩), where *u*ⱼ is the directional derivative of the loss along token *j*'s embedding, evaluated at the token that happened to be drawn. Compare with the exact *p* ⊙ (*ℓ* − E*ℓ*).

Same Jacobian, and in place of the true loss of each alternative, **a first-order extrapolation from the one token you sampled**. Straight-through is exactly as good as the assumption that the loss is linear in embedding space on the scale separating one token from another.

That assumption is testable, because both sides are computable here.""")

    post.add(
        "",
        f"""In this context the extrapolation correlates with the truth at {lin['correlation']:+.2f}. Across six contexts the correlation runs {survey['correlation_min']:+.2f} to {survey['correlation_max']:+.2f}, median {survey['correlation_median']:+.2f}, with regression slopes from {survey['slope_min']:+.2f} to {survey['slope_max']:+.2f}. Five of the six are positive and the sixth is {survey['correlation_min']:+.2f}, which on {dec['vocab'] - 1} points is not distinguishable from zero. So: weakly positive at best, and mostly noise.

But the correlation is not the interesting axis, and the plot makes that obvious. Look at the two ranges. The true changes in loss span {lin['true'].min():+.1f} to {lin['true'].max():+.1f} nats. The extrapolation spans {lin['predicted'].min():+.1f} to {lin['predicted'].max():+.1f} — a standard deviation of {lin['predicted_spread']:.2f} against the truth's {lin['true_spread']:.2f}, a compression of {lin['spread_ratio']:.2f}. **Straight-through's model says roughly the same thing about every alternative.**

Which explains the scale deficit exactly, and better than the correlation does. Push a *constant* vector through `p * (u − <p, u>)` and you get precisely zero, because that is the softmax Jacobian and constants are its null direction — episode 3's theorem, one last time. A nearly constant loss model therefore gives a nearly vanishing gradient, and how nearly is set by how compressed it is. Across the six contexts the compression runs {survey['spread_ratio_min']:.2f} to {survey['spread_ratio_max']:.2f} and correlates with straight-through's measured scale at {survey['spread_vs_scale']:+.2f}. The {st['scale']:.0%} is not a mystery about bias; it is the range of a linearisation, divided by the range of the thing it linearises.

**A correction, because the first version of this section said something stronger and wrong.** Pooling the six contexts into a single correlation gives {survey['pooled_correlation']:+.2f}, and I wrote "anti-correlated, not inaccurate" on the strength of it. That is a Simpson's paradox: each context has its own anchor token and its own spread of losses, so pooling measures variation between contexts and answers a question nobody asked. Five of the six contexts are positive on their own and the sixth is indistinguishable from zero, so the pooled sign belongs to neither. The weaker claim is the true one, and the compression above is what actually explains the bias.""",
        level=3,
        figures=[figs["f1"]])

    post.add(
        "How much of REINFORCE's variance is a baseline problem",
        f"""Before comparing budgets it is worth asking whether the unbiased estimator has to be this noisy, because the answer decides what the comparison is about.

A constant baseline is free, and the mean is the obvious choice but not the optimal one. Minimising the variance of (*ℓ* − *b*)(*e*ₜ − *p*) over *b* gives a weighted mean of the losses, weighted by the squared norm of *e*ₜ − *p*, which is larger for unlikely tokens — so the best constant leans towards the tail. Here it sits at {bl['optimal']:.2f} against a mean of {bl['mean']:.2f}.

And it buys almost nothing. Going from no baseline to the mean takes the per-draw spread from {bl['none']['sd']:.2f} to {bl['at_mean']['sd']:.2f}, a factor of {bl['none']['sd'] / bl['at_mean']['sd']:.1f}. Going from the mean to the optimal constant takes it to {bl['at_optimal']['sd']:.2f} — a further {100 * (1 - bl['at_optimal']['sd'] / bl['at_mean']['sd']):.0f}%.

So the variance that remains is **not** a baseline problem. It is the spread of the losses themselves: {dec['loss'].min():.2f} to {dec['loss'].max():.2f} on this decision, and no number subtracted from all of them shrinks a range. Cutting it further needs a baseline that depends on *which token was drawn* — a control variate, which is to say a model of the loss.

Which is where straight-through returns, in a better role than the one it was auditioning for. Its linearisation is exactly such a model: a cheap per-token estimate of what each alternative would have cost. As the whole gradient, a correlation of {survey['correlation_median']:+.2f} is nowhere near good enough. As a control variate subtracted from an unbiased estimator, a weak positive correlation is **enough to help and unable to hurt**, because the estimator stays unbiased whatever the control variate says. That is the design the literature arrived at, and it is the one these measurements point to.""")

    post.add(
        "So when is a biased estimator the right choice?",
        """Often, and the condition is arithmetic rather than a matter of taste. Averaging *n* draws gives squared error of bias² + variance/*n*. The biased estimator starts lower and stops at its bias; the unbiased one starts higher and keeps falling. They cross somewhere, and both numbers are measured above.""")

    post.add(
        "",
        f"""On this decision they cross at {sm.crossover(st, rfb):.1f} draws. Across six contexts, {n_lo:.1f} to {n_hi:.1f}.

That is a low bar. Straight-through's case rests on being cheap, and it is cheap — one backward pass, no baseline to tune, a spread of {st['sd']:.2f} against REINFORCE's {rfb['sd']:.2f}. But the budget at which its cheapness stops paying is a handful of samples, not thousands, and in a setting where you can afford to sample the same decision even ten times the unbiased estimator is simply better. The intuition that "REINFORCE is too high-variance to use" is doing work here that the numbers do not support.

Two caveats I cannot measure from one decision. The crossover is per-decision, and a training run averages over a batch, which is its own *n* — so a batch of 64 sequences is already well past the crossover if the same decision recurs, and not at all past it if every decision is different. And a systematically short step is not the same failure as a noisy one: an optimiser with momentum and a learning rate can absorb a scale error that it cannot absorb as variance. Which of those dominates is an empirical question about a real training run, and this episode does not answer it.""",
        level=3,
        figures=[figs["f2"]])

    post.add(
        "The relaxation, which trades honestly",
        f"""Gumbel-softmax sits between the two and lets you choose where. Lower the temperature and the relaxed sample looks more like a draw, so the bias falls; it also concentrates, so the variance rises.

Measured here, the trade is real down to about *τ* = {best['tau']:g}, where the bias reaches {best['relative_bias']:.2f}. Below that the bias stops improving — {taus[-1]['relative_bias']:.2f} at *τ* = {taus[-1]['tau']:g} — while the per-draw spread keeps climbing, from {taus[0]['sd']:.2f} at *τ* = {taus[0]['tau']:g} to {taus[-1]['sd']:.2f}. Past the knee you are paying variance for nothing, which is worth knowing before annealing a temperature to zero on principle.""",
        figures=[figs["f3"]])

    post.add(
        "What to keep",
        f"""1. A sampled token is piecewise constant in its logits, so its derivative is 0 almost everywhere and undefined elsewhere. There is no gradient to approximate; there is only the gradient of the expectation, `p * (loss − E loss)`, which is the softmax Jacobian applied to the losses and sums to zero.

2. On a vocabulary small enough, that expectation is enumerable, and then every estimator can be scored against an answer instead of against each other. Do this once on a toy before trusting any of them at scale.

3. REINFORCE is unbiased and measurably so: its bias sits at its own Monte Carlo error. A mean baseline cuts the spread by {100 * (1 - rfb['sd'] / rf['sd']):.0f}% and costs nothing.

4. Straight-through is biased by {st['relative_bias']:.2f} of the exact gradient's norm, {st['relative_bias'] / st['mc_error']:.0f} times its own noise, and returns {st['scale']:.0%} of the right magnitude. More samples do not help.

5. Written on the logits rather than the probabilities it returns {stl['scale']:.1f} times too much at cosine {stl['cosine']:.2f}. Same forward pass. Episode 1's point, in production code.

6. Its cosine with the exact gradient is **not evidence**, because noise through the same softmax Jacobian scores a 90th percentile of {ctrl['p90']:.2f}.

7. What it assumes is a first-order extrapolation across embedding space. It correlates with the truth at {survey['correlation_min']:+.2f} to {survey['correlation_max']:+.2f} within a context — weak but positive — and, more to the point, it is **compressed**: its spread is {survey['spread_ratio_min']:.2f} to {survey['spread_ratio_max']:.2f} of the truth's. A nearly constant loss model gives a nearly vanishing gradient, since constants are the softmax Jacobian's null direction, and that compression tracks the measured scale at {survey['spread_vs_scale']:+.2f}.

8. The crossover is arithmetic: bias² against variance/*n*. Here REINFORCE with a baseline wins after {n_lo:.0f} to {n_hi:.0f} draws.

9. The mean is nearly the best constant baseline there is — the variance-optimal one improves on it by {100 * (1 - bl['at_optimal']['sd'] / bl['at_mean']['sd']):.0f}% — so REINFORCE's remaining variance is the spread of the losses, not a baseline you failed to tune. Cutting it needs a per-token control variate, and straight-through's linearisation is one. Weak is enough for that job and not for this one.

10. And the pooled correlation of {survey['pooled_correlation']:+.2f} that an earlier draft built a claim on is a Simpson's paradox. Correlations across heterogeneous groups are the easiest way to publish a sign error.""")

    post.add(
        "Exercise",
        """Build the smallest version of your own sampling problem that you can enumerate — a vocabulary of a hundred, a single decision, one loss — and score your estimator against the exact gradient. Not its cosine: its **scale**. Cosines are forgiving in a way that will mislead you, for the reason in the control above, and a gradient that points the right way at a fifth of the right size is a learning rate you did not choose.

Then compute your crossover. You need two numbers you probably already have: the bias of your biased estimator, measured once against the enumerated answer, and the per-sample variance of the unbiased one. The ratio tells you the sample count past which the cheap thing costs more than it saves, and it is usually smaller than people expect.

The uncomfortable version: if the crossover for your problem is below your batch size, you have been using the biased estimator for reasons that are not about variance.""")

    return post


if __name__ == "__main__":
    print(build().markdown()[:1500])
