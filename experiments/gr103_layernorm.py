"""Gradients 3: LayerNorm Deletes Exactly Two Directions of Your Gradient.

The third episode. Episode 1 was about differentiability, episode 2 about a
derivative that exists and is tiny; this one is about a Jacobian that is not
of full rank, which is the last of the chain rule's hypotheses a transformer
violates.

Measured:

* The LayerNorm Jacobian is `(I - 11^T/d - xhat xhat^T/d)/sigma`, verified
  against autodiff to 1.7e-16. Both subtracted terms are rank-one projectors
  onto orthonormal directions, so the bracket is an orthogonal projector: the
  rank is exactly d - 2 and every nonzero singular value is exactly 1/sigma.
* The two dead directions are the tangents of the two invariances LayerNorm
  was built to have, shift and positive scale. Differentiate an invariance and
  you get a null direction; episode 2 met the same theorem with one invariance.
* With the learned gain the rank is still exactly d - 2 and the nonzero
  spectrum spreads by about 1.5, so the projector picture survives.
* The share of the real gradient lying in the dead plane has a median near
  0.12 against a random-direction baseline of 0.105. Barely above chance in
  the middle; the tail reaches 0.81 against a random maximum of 0.43.
* And the residual hides it. The branch through ln1 measures rank 123 of 128
  at a position; the block containing it measures 128, smallest singular value
  0.36. No block is rank-deficient because of its LayerNorm.
* Except the final norm, which has no residual after it, so its two dead
  directions are exact invariances of the whole network: shifting the final
  hidden state by any multiple of the all-ones vector, or scaling it by any
  positive constant, leaves the loss identical to ten digits.
* The scale invariance is not exact but linear in eps/var: a log-log slope
  of 0.97 with a constant near 0.05, over four decades of that ratio, so
  there is no threshold and sqrt(eps) is not a cliff. It reads as exact at
  this model's scale because 5% of 1e-6 is under float32's resolution.
* What actually varies is the scalar: 1/sigma falls from 1.11 to 0.32 across
  depth because the residual stream grows from 0.90 to 3.90 rms.

Run: `standarderror run gr103_layernorm --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.calculus import normalisation as nz
from standarderror.llm import tiny
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 9, 11)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Calculus for Language Models, Taught Through What Breaks"
SERIES_TAG = "Gradients"

BATCHES, BATCH_SIZE = 4, 8
EPS = 1e-5


def compute() -> dict:
    model = tiny.load()["model"]
    return {
        "autodiff": nz.against_autodiff(d=8),
        "gains": nz.gains(model),
        "shares": nz.deleted_share(count=BATCHES, size=BATCH_SIZE, seed=0),
        "block": nz.block_rank(layer=0),
        "invariance": nz.network_invariance(size=BATCH_SIZE),
        "sweep": nz.scale_sweep(eps=EPS, size=BATCH_SIZE),
        "ladder": nz.sigma_ladder(size=BATCH_SIZE),
    }


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    shares, base = res["shares"]["norms"], res["shares"]["baseline"]
    sweep, ladder = res["sweep"], res["ladder"]
    inv = res["invariance"]

    def dead(ax, m):
        names = [r["name"].replace("blocks.", "b").replace(".ln", ".ln")
                 for r in shares]
        xs = np.arange(len(shares))
        ax.bar(xs, [r["median"] for r in shares], width=0.6,
               color=m.series[0], label="median share of the gradient")
        ax.scatter(xs, [r["p90"] for r in shares], s=26, zorder=4,
                   color=m.series[1], label="90th percentile")
        ax.scatter(xs, [r["max"] for r in shares], s=26, marker="^", zorder=4,
                   color=m.series[2], label="largest row")
        ax.axhline(base["median"], lw=1.6, ls="--", color=m.ink_secondary,
                   label=f"random direction, median {base['median']:.3f}")
        ax.axhline(base["max"], lw=1.4, ls=":", color=m.grid,
                   label=f"random direction, largest {base['max']:.2f}")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.2)
        ax.set_ylim(0, 0.9)
        ax.legend(frameon=False, fontsize=8.2, loc="upper left", ncol=2)

    out["f0"] = charts.diagram(
        dead,
        title="Two directions out of 128, and the gradient barely notices",
        subtitle=(f"The share of the gradient arriving at each LayerNorm that "
                  f"lies in the plane it deletes, over "
                  f"{shares[0]['rows']:,} rows of held-out text per norm."),
        xlabel="", ylabel="share of the gradient norm",
        source="Measured; standarderror/calculus/normalisation.py.",
        alt=("Nine bars near 0.12 with a dashed random-baseline line just "
             "below them, and scattered maxima rising well above the random "
             "maximum."),
        caption=(f"A uniformly random direction in 128 dimensions already "
                 f"puts {base['median']:.3f} of its length in any fixed "
                 f"2-plane, and the measured medians sit at "
                 f"{min(r['median'] for r in shares):.3f} to "
                 f"{max(r['median'] for r in shares):.3f}. So in the middle "
                 f"of the distribution LayerNorm deletes about what deleting "
                 f"two arbitrary directions would. The tail is the "
                 f"exception, and it is not one norm: every one of the nine "
                 f"has rows above the random maximum of {base['max']:.2f}, "
                 f"and the final norm reaches "
                 f"{max(r['max'] for r in shares):.2f}, where the deletion "
                 f"costs {nz.norm_cost(max(r['max'] for r in shares)):.0%} "
                 f"of that row's gradient."),
        figsize=(7.2, 5.0),
        path=str(IMG / f"gr103-f0-deleted.{EXT}"))[0]

    rows = [[c["name"], f"{c['loss']:.10f}",
             ("0" if c["delta"] == 0.0 else f"{c['delta']:+.1e}"),
             f"{c['max_logit_change']:.1e}"]
            for c in inv["cases"]]
    rows.insert(0, ["no change", f"{inv['baseline']:.10f}", "0", "0.0e+00"])
    bold = {(len(rows) - 1, c) for c in range(4)}

    out["f1"] = charts.table_image(
        rows,
        header=["what was done to the final hidden state", "loss",
                "change", "largest logit change"],
        title="Two things you can do to this network that it cannot detect",
        subtitle=("The final LayerNorm has no residual connection after it, "
                  "so the two directions it deletes are deleted from the "
                  "model's output rather than from one branch of it."),
        source="Measured; standarderror/calculus/normalisation.py.",
        alt=("A table of five perturbations. Three change the loss by "
             "exactly zero; the last, along a single coordinate, does not."),
        caption=("Adding any multiple of the all-ones vector, or multiplying "
                 "by any positive constant, leaves the loss identical to "
                 "every printed digit. The **bold** last row is the control: "
                 "the same size of change along one coordinate moves a logit "
                 "by more than half a nat. The invariance is specific to "
                 "those two directions, and it is a property of the whole "
                 "network rather than of one layer."),
        bold_cells=bold, align="lrrr",
        path=str(IMG / f"gr103-f1-invariance.{EXT}"))[0]

    def breaks(ax, m):
        live = sweep["proportional_rows"]
        ev = [r["eps_over_var"] for r in live]
        dl = [abs(r["delta"]) for r in live]
        ax.plot(ev, dl, marker="o", ms=7, lw=2.0, color=m.series[0],
                label="measured change in loss")
        grid = np.logspace(-5, 1, 50)
        ax.plot(grid, sweep["constant"] * grid ** sweep["slope"], lw=1.8,
                ls="--", color=m.series[2],
                label=f"fitted, slope {sweep['slope']:.2f}")
        dead = [r for r in sweep["rows"]
                if r["eps_over_var"] <= 1e-5 and r["delta"] == 0.0]
        if dead:
            ax.scatter([r["eps_over_var"] for r in dead],
                       [3e-9] * len(dead), s=48, marker="v",
                       color=m.series[1],
                       label="loss identical to every float32 digit")
        ax.axvline(1.0, lw=1.4, ls=":", color=m.grid,
                   label="eps equals the variance (sigma = sqrt(eps))")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-9, 20)
        ax.set_ylim(1e-9, 1.0)
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    out["f2"] = charts.diagram(
        breaks,
        title="Not a cliff: the epsilon spoils the invariance in proportion",
        subtitle=("Scaling the final hidden state and watching the loss, "
                  "against the ratio of the epsilon to the variance it is "
                  "added to. The dashed line is the fit; what matters is "
                  "that its slope came out one."),
        xlabel="eps / variance (log scale)",
        ylabel="absolute change in loss (log scale)",
        source="Measured; standarderror/calculus/normalisation.py.",
        alt=("Four points on a straight line of slope one against a dashed "
             "proportionality, with triangles at the floor marking the "
             "scales where the loss did not change at all."),
        caption=(f"Over four decades of eps/variance the error in the loss "
                 f"is linear in it - a log-log slope of "
                 f"{sweep['slope']:.2f}, constant {sweep['constant']:.3f} - "
                 f"which is a proportionality, not a threshold. The "
                 f"triangles are the scales at which the "
                 f"loss was identical to every float32 digit, which is what "
                 f"**exact** means here and is a statement about the "
                 f"resolution of the comparison as much as about the layer. "
                 f"The dotted line at 1 is sigma = sqrt(eps), and the "
                 f"invariance is visibly gone well before it."),
        path=str(IMG / f"gr103-f2-sweep.{EXT}"))[0]

    def ladder_plot(ax, m):
        pts = ladder_pts = ladder["points"]
        xs = np.arange(len(pts))
        ax.plot(xs, [p["rms"] for p in pts], marker="o", ms=5, lw=2.2,
                color=m.series[0], label="residual stream, rms")
        ax.set_ylabel("size of the residual stream")
        ax.set_ylim(0, 4.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([p["point"].replace("after ", "")
                            for p in ladder_pts], rotation=45, ha="right",
                           fontsize=8.0)
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")
        right = ax.twinx()
        right.plot(xs, [p["inv_sigma"] for p in pts], marker="s", ms=5,
                   lw=2.2, ls="--", color=m.series[2],
                   label="1/sigma handed to the backward pass")
        right.set_ylim(0, 1.25)
        right.set_ylabel("gradient scaling, 1/sigma")
        right.legend(frameon=False, fontsize=8.4, loc="lower left")

    out["f3"] = charts.diagram(
        ladder_plot,
        title="The only number in that Jacobian that changes",
        subtitle=("The residual stream accumulates as the blocks write to "
                  "it, and every LayerNorm divides by what it finds there."),
        xlabel="", ylabel="",
        source="Measured; standarderror/calculus/normalisation.py.",
        alt=("One curve rising from about 0.9 to about 3.9 and a dashed "
             "curve falling from about 1.11 to about 0.32, mirroring it."),
        caption=(f"The residual stream grows {ladder['growth']:.1f}-fold "
                 f"from the embedding to the last block, and 1/sigma falls "
                 f"{ladder['attenuation']:.1f}-fold with it. That is the "
                 f"one quantity in the LayerNorm Jacobian that varies from "
                 f"place to place, and it varies more than the rank "
                 f"deficiency costs. The **backward** behaviour of the layer "
                 f"is set by a forward fact about depth."),
        figsize=(7.4, 5.0),
        path=str(IMG / f"gr103-f3-ladder.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    block, ladder = res["block"], res["ladder"]

    def plane(panel, m):
        # A vector, and the two components the Jacobian removes.
        panel.plot([0, 0.95], [0, 0.62], lw=2.6, color=m.series[0])
        panel.plot([0, 0.95], [0, 0], lw=2.0, ls="--", color=m.series[2])
        panel.plot([0.95, 0.95], [0, 0.62], lw=2.0, ls="--", color=m.grid)
        panel.scatter([0], [0], s=40, color=m.ink, zorder=4)
        panel.set_xlim(-0.12, 1.15)
        panel.set_ylim(-0.18, 0.85)

    def skip(panel, m):
        # The branch loses rank; the identity beside it does not.
        panel.plot([0, 1], [0.75, 0.75], lw=2.8, color=m.series[0])
        panel.plot([0.15, 0.5, 0.85], [0.75, 0.28, 0.75], lw=2.2,
                   color=m.series[2])
        panel.scatter([0.5], [0.28], s=46, color=m.series[2], zorder=4)
        panel.set_xlim(-0.1, 1.1)
        panel.set_ylim(0.05, 1.0)

    def fall(panel, m):
        pts = ladder["points"]
        xs = np.arange(len(pts))
        panel.plot(xs, [p["rms"] for p in pts], lw=2.6, color=m.series[0])
        panel.plot(xs, [4.0 * p["inv_sigma"] for p in pts], lw=2.4,
                   color=m.series[2])
        panel.set_xlim(0, xs[-1])
        panel.set_ylim(0, 4.6)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=3,
        headline="A rank deficiency you can prove, and never see",
        panels=[(plane, "d - 2", "rank of the Jacobian"),
                (skip, f"{block['branch_rank']} to {block['block_rank']}",
                 "what the skip restores"),
                (fall, f"{ladder['attenuation']:.1f}x", "the scaling that varies")],
        note=("LayerNorm's Jacobian has rank exactly d - 2, and the two "
              "missing directions are the invariances it was built to have. "
              "The residual connection restores the rank in every block; the "
              "final norm, which has none, turns them into exact invariances "
              "of the whole network."),
        alt=("Three hand-drawn frames: a vector with two dashed components "
             "removed; a straight line with a detour beneath it; and a "
             "rising curve crossing a falling one."),
        path=str(IMG / f"gr103-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["algebra"] = s.run("""
        import numpy as np
        from standarderror.calculus import normalisation as nz

        # The closed form against torch's own derivative, then the structure.
        print(f"closed form vs autodiff   "
              f"{nz.against_autodiff(d=8)['max_abs_error']:.1e}")
        print()
        x = np.random.default_rng(0).normal(0.0, 2.0, 128)
        g = nz.spectrum(x)
        print(f"rank                      {g['rank']} of {g['d']}")
        print(f"largest singular value    {g['largest']:.6f}")
        print(f"smallest nonzero          {g['smallest_nonzero']:.6f}")
        print(f"1 / sigma                 {1 / g['sigma']:.6f}")
        print(f"J applied to 1 and xhat   {g['null_residual']:.1e}")
    """, expect=["closed form vs autodiff"])

    out["block"] = s.run("""
        # The branch through the norm, and the block that contains it.
        b = nz.block_rank(layer=0)
        print(f"the ln1 branch alone   rank {b['branch_rank']} of {b['d']}")
        print(f"the whole block        rank {b['block_rank']} of {b['d']}")
        print(f"  its smallest singular value  {b['block_smallest']:.3f}")
        print(f"  its largest                  {b['block_largest']:.3f}")
    """, expect=["the ln1 branch alone"])

    out["invariance"] = s.run("""
        # The last norm has no residual after it. So what does the network
        # fail to notice?
        r = nz.network_invariance()
        print(f"baseline loss                       {r['baseline']:.10f}")
        for c in r["cases"]:
            print(f"  {c['name']:<28} {c['loss']:.10f}   "
                  f"delta {c['delta']:+.1e}")
    """, expect=["baseline loss"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    shares, base = res["shares"]["norms"], res["shares"]["baseline"]
    block, inv, sweep = res["block"], res["invariance"], res["sweep"]
    ladder = res["ladder"]
    by = {c["name"]: c for c in inv["cases"]}

    # The spine, asserted rather than trusted.
    assert res["autodiff"]["max_abs_error"] < 1e-12
    assert all(r["rank"] == tiny.WIDTH - 2 for r in res["gains"])
    assert max(r["spread"] for r in res["gains"]) < 3.0
    assert max(r["median"] for r in shares) < 3 * base["median"]
    assert max(r["max"] for r in shares) > base["max"]
    assert block["block_rank"] == tiny.WIDTH
    assert block["branch_rank"] < tiny.WIDTH
    assert by["shift by 5 * ones"]["delta"] == 0.0
    assert by["scale by 3"]["delta"] == 0.0
    assert abs(by["shift by 5 along one axis"]["delta"]) > 1e-4
    assert len(sweep["proportional_rows"]) >= 4
    assert abs(sweep["slope"] - 1.0) < 0.1
    assert ladder["growth"] > 2.0 and ladder["attenuation"] > 2.0

    post = Post(
        title=(f"{SERIES_TAG} 3: LayerNorm Deletes Exactly Two Directions of "
               f"Your Gradient"),
        slug="gradients-3-layernorm",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=3,
        date=POST_DATE,
        # Exact algebra plus counts and losses on a fixed model. No prediction.
        requires_baseline=False,
        subtitle=("Its Jacobian has rank exactly d minus 2, and the two "
                  "missing directions are the invariances the layer was "
                  "built to have. Then the residual connection restores the "
                  "rank everywhere except the last norm, where it becomes an "
                  "exact invariance of the entire network."),
        summary=(
            f"The LayerNorm Jacobian is the orthogonal projector off the "
            f"plane spanned by the all-ones vector and the normalised input, "
            f"divided by sigma - verified against autodiff to "
            f"{res['autodiff']['max_abs_error']:.0e}, rank exactly d minus 2, "
            f"and with a flat nonzero spectrum because it really is a "
            f"projector. The two dead directions are the tangents of the two "
            f"invariances the layer has by construction, shift and positive "
            f"scale, which is the same theorem episode 2 met with one "
            f"invariance instead of two. Measured on the model, the share of "
            f"the real gradient lying in that plane has a median of about "
            f"0.12 against 0.105 for a random direction, and a deleted "
            f"component that size costs half a percent of the gradient's "
            f"length rather than ten. And the residual hides even that: the branch through "
            f"the first norm measures rank 123 of 128 at a position while the "
            f"block containing it measures 128. The exception is the final "
            f"norm, which has no residual after it, so shifting the final "
            f"hidden state by any multiple of the all-ones vector or scaling "
            f"it by any positive constant leaves the loss identical to ten "
            f"digits. The shift is exact in the arithmetic; the scaling "
            f"turns out to be linear in eps over the variance rather than "
            f"exact - a log-log slope of 0.97 with a constant near 0.05, over "
            f"four decades of that ratio - which at this model's scale is far "
            f"under float32's resolution. What actually varies "
            f"is the scalar: 1/sigma falls "
            f"{ladder['attenuation']:.1f}-fold across depth as the residual "
            f"stream grows."),
        tags=["deep-learning", "machine-learning", "llm", "pytorch",
              "mathematics", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. The algebra is checked on vectors drawn at "
            "build time; every model number is a rank, a share or a loss on "
            "held-out text from the corpus the model was trained on, and no "
            "values from that text are published.",
            f"The model: an {tiny.load()['parameters']:,}-parameter "
            f"character-level transformer trained for this series - four "
            f"blocks, four heads, width {tiny.WIDTH}, context {tiny.BLOCK}, "
            f"validation loss {tiny.VAL_LOSS} against a uniform-guess "
            f"{tiny.UNIFORM_LOSS:.3f}. Weights, training script and a "
            f"verified sha256 are committed: `standarderror/llm/tiny.py`, "
            f"`scripts/train_tiny.py`, `data/tiny_gpt/`.",
            "Machinery: `standarderror/calculus/normalisation.py`, tested in "
            "`tests/test_normalisation.py`, which pins the algebra exactly "
            "and the model findings as inequalities.",
            "Where this stops: Ba, Kiros and Hinton, \"Layer normalization\" "
            "(2016), for the definition; Xu et al., \"Understanding and "
            "improving layer normalization\", *NeurIPS* (2019), for the "
            "derivative and the argument that what normalisation does to the "
            "gradient matters more than what it does to the forward pass; "
            "Elhage et al., \"A mathematical framework for transformer "
            "circuits\", *Transformer Circuits Thread* (2021), for the "
            "residual stream as the object every block reads from and writes "
            "to.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, "
                           "torch=2.14.0, numpy=2.4.4",
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "model": ("the committed checkpoint, hash-verified on load, so "
                      "every rank and every loss here is measured on the "
                      "same weights"),
            "determinism": (f"the gradient survey draws {BATCHES} batches of "
                            f"{BATCH_SIZE} held-out sequences from a fixed "
                            f"seed; the random-direction baseline is 200,000 "
                            f"draws from a fixed seed; the rank measurements "
                            f"are exact linear algebra on one position and "
                            f"do not depend on a seed at all"),
        },
    )
    post.hero = figs["hero"]
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    shares, base = res["shares"]["norms"], res["shares"]["baseline"]
    block, inv, sweep = res["block"], res["invariance"], res["sweep"]
    ladder = res["ladder"]
    gains = res["gains"]
    by = {c["name"]: c for c in inv["cases"]}
    lo = min(r["median"] for r in shares)
    hi = max(r["median"] for r in shares)
    worst = max(shares, key=lambda r: r["max"])
    first, last = ladder["points"][0], ladder["points"][-1]

    post.add(
        "The hypothesis nobody states",
        """Backpropagation multiplies Jacobians. Nothing in the chain rule requires any of them to be invertible, and the gradient is perfectly well defined when one is not — but a Jacobian with a null space is a Jacobian that throws information away, and it is worth knowing which information.

LayerNorm throws away two dimensions per layer, per position, exactly. Not approximately, not usually, and not as a numerical artefact. The number two is provable in three lines, and this episode is about where those two dimensions go, which turns out to be a more interesting question than whether they exist.""")

    post.add(
        "Three lines",
        """Write *y* = (*x* − μ)/σ with σ = the square root of the variance plus ε, both taken across the *d* features. Differentiating gives

$$
\\frac{\\partial y}{\\partial x}
  = \\frac{1}{\\sigma}
    \\left( I - \\frac{\\mathbf{1} \\mathbf{1}^{\\top}}{d}
             - \\frac{\\hat{x} \\hat{x}^{\\top}}{d} \\right),
  \\qquad \\hat{x} = \\frac{x - \\mu}{\\sigma}.
$$

Now look at the bracket rather than through it. The normalised vector x̂ has mean zero and variance one, so the sum of its squares is *d* and the sum of its entries is zero. That makes both subtracted terms **rank-one orthogonal projectors onto orthonormal directions** — the all-ones direction and the x̂ direction, which are perpendicular because x̂ is centred. Subtract two such projectors from the identity and you have the orthogonal projector onto everything else.

So the Jacobian is not merely rank-deficient. It is 1/σ times an orthogonal projection, its rank is exactly *d* − 2, and every one of its nonzero singular values is exactly 1/σ.""")

    post.add(
        "",
        f"""{snip['algebra'].markdown()}

Rank {tiny.WIDTH - 2} of {tiny.WIDTH}, the largest and smallest nonzero singular values identical to six decimal places and both equal to 1/σ, and the two claimed null directions annihilated to the last bit of a float64.""",
        level=3)

    post.add(
        "Why those two",
        """Not by coincidence. LayerNorm has two invariances by construction: it does not care if you add the same number to every feature, and it does not care if you multiply every feature by the same positive number. Both are deliberate — removing the mean and the scale is the entire point of the layer.

Differentiate an invariance and you get a null direction. If adding *t* times the all-ones vector to *x* leaves *f* alone for every *t*, then the derivative along the all-ones direction is zero, so that direction is in the null space. If multiplying *x* by *a* leaves *f* alone, differentiate at *a* = 1 and the *x* direction is in the null space too — and *x* and x̂ span the same direction once the mean is removed.

Two invariances, two dead directions. Episode 2 met the same theorem with one: a softmax does not change when you add a constant to every logit, so its Jacobian annihilates the all-ones vector and the logit gradients in a row sum to zero. Same statement, one fewer symmetry.

Which reframes the rank deficiency entirely. It is not a leak in the backward pass. It is the derivative correctly reporting that the function ignores those directions, and a gradient that had a component there would be **wrong**.""")

    post.add(
        "What it costs, against the right baseline",
        f"""So much for whether it happens. How much of an actual gradient falls into the plane that gets deleted?

The question needs a baseline, because two directions out of {tiny.WIDTH} will catch part of anything. A uniformly random unit vector puts a root-mean-square {base['rms']:.4f} of its length in any fixed 2-plane, which is exactly the square root of 2/{tiny.WIDTH}; the distribution is skewed, so its **median** is the lower {base['median']:.3f}. An indifferent gradient should look like that.

Measured across the model's nine LayerNorms on held-out text, the medians run from {lo:.3f} to {hi:.3f} — straddling {base['median']:.3f}: in the middle of the distribution, LayerNorm deletes roughly what deleting two arbitrary directions would delete. The plane it removes is not a plane the loss particularly wanted.

The tail is the interesting part. A random direction's largest share over 200,000 draws is {base['max']:.2f}; every one of the nine exceeds that, and the final norm reaches {worst['max']:.2f} with a 90th percentile of {worst['p90']:.2f} against {base['p90']:.2f}. So a small fraction of tokens really do have most of their gradient pointing into the dead plane, and for those the layer removes something. There is structure there; there is just not much of it on average.""",
        figures=[figs["f0"]])

    post.add(
        "And then the skip connection",
        """Here is where the famous fact stops mattering, and it is not subtle once seen.

A transformer block is `x + f(LN(x))`. The Jacobian of that is the identity plus something. Whatever rank the branch has, the block has full rank, because the identity is sitting right next to it — and the two directions the norm deleted arrive at the layer below through the skip, undisturbed, at strength one.""")

    post.add(
        "",
        f"""{snip['block'].markdown()}

The branch measures rank {block['branch_rank']} — lower than {tiny.WIDTH - 2}, and not because of the norm: attention at a single position also mixes across positions, which removes more. The block measures {block['block_rank']} of {tiny.WIDTH}, with a smallest singular value of {block['block_smallest']:.2f} against a largest of {block['block_largest']:.2f}. Not merely full rank, but comfortably conditioned.

So: **no block in this transformer is rank-deficient because of its LayerNorm.** The exact result from three sections ago is exact, provable, and structurally invisible, in every place a modern architecture puts a normalisation layer — which is to say in front of a residual branch. That is the answer to the episode's own premise, and it is a negative one.""",
        level=3)

    post.add(
        "Except at the end, where there is no skip",
        """With one exception, and it is the last layer. The final norm is not in front of a residual branch. It feeds the output head directly, so what it deletes is deleted from the model's output rather than from one contribution to it.

That makes the two dead directions **exact invariances of the whole network**, which is a claim you can test on the loss rather than on a Jacobian.""")

    post.add(
        "",
        f"""{snip['invariance'].markdown()}

Add five times the all-ones vector to the final hidden state: the loss is identical to every digit printed. Add a thousand times: identical to {abs(by['shift by 1000 * ones']['delta']):.0e}, which is one unit in the last place of a float32. Multiply by three: identical. Do both: identical. Move the same distance along a single coordinate instead, and the loss moves by {abs(by['shift by 5 along one axis']['delta']):.1e} with a logit shifting by {by['shift by 5 along one axis']['max_logit_change']:.2f}.

Two directions of the {tiny.WIDTH}-dimensional final hidden state have no effect on this model's output. Not a small effect. None.""",
        level=3,
        figures=[figs["f1"]])

    post.add(
        "\"Exact\" is a statement about the comparison too",
        f"""The shift invariance is exact in the arithmetic: subtracting the mean removes a constant offset whatever ε is. The scale invariance is not, and it is worth pushing on, because σ is the square root of the variance **plus ε**, and that sum is not homogeneous in *x*. Scaling by *a* multiplies the variance by *a*² and leaves ε where it is.

So the governing quantity is not σ but the ratio ε/Var, and I expected the invariance to hold until that ratio reached one — at σ = the square root of ε, which for ε = {sweep['eps']:.0e} is {sweep['sqrt_eps']:.5f}. That is not what the measurement says.""")

    post.add(
        "",
        f"""There is no threshold. Over four decades in which ε/Var changes by a factor of ten thousand, the error in the loss is **linear** in it: a log-log slope of {sweep['slope']:.2f} with a constant of {sweep['constant']:.3f}, and point-to-point scatter of a factor of a few — the individual ratios run {sweep['ratio_min']:.3f} to {sweep['ratio_max']:.3f}. At σ = {sweep['proportional_rows'][0]['sigma']:.2f} — a tenth of this model's natural scale, nowhere near any cliff — the ratio is {sweep['proportional_rows'][0]['eps_over_var']:.0e} and the loss already moves by {abs(sweep['proportional_rows'][0]['delta']):.1e}. By σ = the square root of ε the invariance is not breaking; it broke some time ago and is now simply large.

Which means the "exact to ten digits" from the previous section was partly a statement about the ratio and partly a statement about float32. At this model's own scale ε/Var is about {sweep['eps'] / sweep['sigma'] ** 2:.0e}, and {sweep['constant']:.0%} of that is far below the resolution of a float32 loss — so the invariance reads as exact because the error has nowhere to appear, not because it is zero. Which is the useful version of the claim: **the scale invariance is exact up to ε/Var, and at any sane ε that is under the floor.**

One point on the plot has a different cause. Scaling the hidden state *up* by a thousand also perturbs the loss, by {abs([r for r in sweep['rows'] if r['scale'] > 100][0]['delta']):.1e}, and ε/Var there is {[r for r in sweep['rows'] if r['scale'] > 100][0]['eps_over_var']:.0e} — a hundred thousand times too small to explain it. That one is float32 rounding on a large number, which is a different failure with the same symptom, and it is the reason to compute the ratio rather than eyeball the curve.""",
        level=3,
        figures=[figs["f2"]])

    post.add(
        "What actually varies",
        f"""The rank deficiency is exact, provable and mostly invisible. The gradient share is close to what two random directions would take. So the honest question at the end of an episode like this is: what in that Jacobian does move the numbers?

The scalar. 1/σ is the only quantity in `P/σ` that differs from place to place, and it differs a lot.

At the embedding the residual stream has an rms of {first['rms']:.2f} and the norm hands back a factor of {first['inv_sigma']:.3f} — a slight amplification. After the last block the stream is at {last['rms']:.2f} and the factor is {last['inv_sigma']:.3f}. A {ladder['attenuation']:.1f}-fold ladder, monotone in depth, and it exists for a simple reason: every block **adds** to the residual stream, so the stream grows, so every LayerNorm divides by a larger number than the one before it.

That is the backward-pass view of a fact usually stated about the forward pass. The residual stream growing with depth is normally discussed as a scaling problem for the activations. It is also, and by exactly the same numbers, a depth-dependent gradient scaling — one that no hyperparameter set and no optimiser state knows about.

For comparison, and the conversion matters: a deleted component of relative size {base['median']:.3f} does not shorten the gradient by {base['median']:.0%}. Norms add in quadrature, so it costs {nz.norm_cost(base['median']):.2%} of the length. Against that, the scalar changes the gradient by {ladder['attenuation']:.1f}-fold between the first norm and the last. If you were going to worry about one of these, the exact one is not it — the typical deletion is a {nz.norm_cost(base['median']):.2%} effect and the scaling is a {100 * (ladder['attenuation'] - 1):.0f}% one.""",
        figures=[figs["f3"]])

    post.add(
        "What to keep",
        f"""1. LayerNorm's Jacobian is `(I − 11ᵀ/d − x̂x̂ᵀ/d)/σ`, which is 1/σ times an **orthogonal projector**: rank exactly *d* − 2, and every nonzero singular value exactly 1/σ. Verified against autodiff to {res['autodiff']['max_abs_error']:.0e}.

2. The two dead directions are the tangents of the layer's two invariances, shift and positive scale. Differentiate an invariance, get a null direction. A softmax has one invariance and deletes one direction; this is the same theorem.

3. So the rank deficiency is not a leak. A gradient with a component in those directions would be wrong.

4. With the learned gain the rank is still exactly *d* − 2 and the nonzero singular values spread by only {max(r['spread'] for r in gains):.2f}, so the projector picture survives contact with a trained model.

5. Measured, the share of the real gradient in the deleted plane has a median of {lo:.3f} to {hi:.3f} against {base['median']:.3f} for a random direction. Barely above chance in the middle; above it in every tail, and the final norm reaches {worst['max']:.2f} against a random maximum of {base['max']:.2f}.

6. And the residual makes it moot: the branch is rank {block['branch_rank']}, the block is rank {block['block_rank']}. No block is rank-deficient because of its normalisation.

7. Except the final norm, which has no residual after it — so two directions of the final hidden state are **exact invariances of the whole network**. The shift is exact in the arithmetic; the scaling is exact only up to ε/Var, and the measurement shows that is a straight proportionality with no threshold in it — slope {sweep['slope']:.2f}, constant {sweep['constant']:.3f}.

8. And the sizes are not comparable the way they look. A deleted component of relative size {base['median']:.3f} costs {nz.norm_cost(base['median']):.2%} of the gradient's length, because norms add in quadrature. The 1/σ ladder across depth is {ladder['attenuation']:.1f}-fold, set by the residual stream growing {ladder['growth']:.1f}-fold. That is the part worth watching.""")

    post.add(
        "Exercise",
        """Take your own model, grab the final hidden state, add a hundred times the all-ones vector, and check that nothing happens. It takes two lines and it is worth doing once with your own hands, because "two directions of this representation carry no information" reads as a claim and feels like a fact only after you have watched the logits not move.

Then log the standard deviation of the residual stream at every LayerNorm across a training run. You get the gradient-scaling ladder for free, since it is the reciprocal, and if it is steeper than the threefold in this small model you have found a depth-dependent learning-rate schedule that you did not choose and cannot see in your config.

The harder version: put your epsilon somewhere it matters. Train two models differing only in ε — say 1e-5 and 1e-1 — and measure the scale invariance of each. The second is not normalising in any exact sense at all, and the question is whether it is worse, which I do not know the answer to and would like to.""")

    return post


if __name__ == "__main__":
    print(build().markdown()[:1500])
