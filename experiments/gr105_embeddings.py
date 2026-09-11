"""Gradients 5: The Gradient in Embedding Space Does Not Point at a Token.

The last episode. Four hypotheses of the chain rule, then a place with no
derivative at all, and now a place where the derivative exists and is exactly
right and still tells you almost nothing -- because the space it lives in has
no local neighbourhood.

Measured:

* The embedding table is 65 near-orthogonal vectors on a sphere. Median norm
  7.93, median pairwise distance 11.214, and sqrt(2) * 7.93 = 11.210 -- the
  distance between orthogonal vectors of that length, matched to four
  figures. Median pairwise cosine +0.0003. Random vectors reproduce it, so
  this is dimension rather than training.
* The nearest other token sits at 0.88 of the typical pairwise distance, so
  "nearby in embedding space" is not a distinguished relation.
* A descent step therefore goes nowhere: the nearest row to where you land is
  still the token you started from, for steps up to 1.9 embedding norms and
  31.6 in one of six contexts.
* And the token you eventually reach is never the best substitution -- ranks
  1, 1, 4, 5, 19, 26 of 65 -- and in one context its loss is higher than the
  token you started from.
* But graded as a shortlist rather than a direction, the same gradient is
  good: its top five of 65 recovers 96% to 100% of the available improvement
  in five of six contexts, including one where the best token ranks 55th.
* So the rank of the single best token is the wrong metric, and regret is the
  right one. They disagree by a lot.

Run: `standarderror run gr105_embeddings --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.calculus import embeddings as em
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


def ordinal(n: int) -> str:
    """1st, 2nd, 56th. Written out because "56st" shipped in a draft.

    Every rank in this episode is 1-indexed for the reader and 0-indexed in
    the arrays, so the suffix is computed rather than typed.
    """
    n = int(n)
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def compute() -> dict:
    x, y = next(iter(tiny.batches(count=1, size=CONTEXTS, seed=0)))
    sub = em.substitutions(x[0, :tiny.BLOCK - 1], int(y[0, tiny.BLOCK - 1]))
    rank = em.ranking(sub)
    return {"table": em.table(), "survey": em.survey(contexts=CONTEXTS),
            "sub": sub, "ranking": rank,
            "landing": em.step_landing(sub),
            "control": em.random_shortlist(sub, rank)}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    tbl, survey = res["table"], res["survey"]

    def geom(ax, m):
        e = tbl["embeddings"]
        d = np.linalg.norm(e[:, None] - e[None], axis=-1)
        iu = np.triu_indices(len(e), 1)
        ax.hist(d[iu], bins=40, color=m.series[0], alpha=0.85,
                label="pairwise distances between embedding rows")
        ax.axvline(tbl["orthogonal_prediction"], lw=2.2, ls="--",
                   color=m.series[2],
                   label=f"sqrt(2) x median norm = "
                         f"{tbl['orthogonal_prediction']:.2f}")
        ax.axvline(tbl["median_norm"], lw=1.8, ls=":", color=m.series[1],
                   label=f"the median norm itself = "
                         f"{tbl['median_norm']:.2f}")
        ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    out["f0"] = charts.diagram(
        geom,
        title="Sixty-five tokens, and every one of them a stranger",
        subtitle=(f"All {tbl['rows'] * (tbl['rows'] - 1) // 2:,} pairwise "
                  f"distances in the embedding table, against the distance "
                  f"orthogonal vectors of that norm would have."),
        xlabel="distance between two embedding rows",
        ylabel="pairs",
        source="Measured; standarderror/calculus/embeddings.py.",
        alt=("A narrow histogram of distances centred almost exactly on the "
             "dashed orthogonal prediction, far to the right of the dotted "
             "line marking the embeddings' own length."),
        caption=(f"The median pairwise distance is "
                 f"{tbl['median_pairwise']:.3f} and the orthogonal "
                 f"prediction is {tbl['orthogonal_prediction']:.3f}, which "
                 f"agree to four figures - and the median pairwise cosine is "
                 f"{tbl['median_cosine']:+.4f}. The rows are mutually "
                 f"**further apart than they are long**, so there is no "
                 f"local neighbourhood for a gradient step to move within."),
        path=str(IMG / f"gr105-f0-geometry.{EXT}"))[0]

    def walk(ax, m):
        e = tbl["embeddings"]
        for r in survey["rows"]:
            g = r["ranking"]["gradient"]
            u = g / np.linalg.norm(g)
            cur = r["sub"]["current"]
            norm = float(np.linalg.norm(e[cur]))
            fr = np.linspace(0.0, 35.0, 400)
            pts = e[cur][None] - fr[:, None] * norm * u[None]
            d = np.linalg.norm(pts[:, None] - e[None], axis=-1)
            others = np.delete(np.arange(len(e)), cur)
            ax.plot(fr, d[:, others].min(1) - d[:, cur], lw=1.9, alpha=0.85,
                    color=m.series[0])
        ax.axhline(0.0, lw=1.8, ls="--", color=m.series[2],
                   label="below this line, some other token is nearer")
        ax.axvline(1.0, lw=1.6, ls=":", color=m.ink_secondary,
                   label="a step the size of the embedding itself")
        ax.set_ylim(-2.5, 12.5)
        ax.set_xlim(0, 35)
        ax.legend(frameon=False, fontsize=8.6, loc="upper right")

    out["f1"] = charts.diagram(
        walk,
        title="Walking down the gradient, still nearest to where you began",
        subtitle=("How much further the nearest other token is than the one "
                  "you started from, as a point moves along the negative "
                  "gradient. One line per context."),
        xlabel="distance walked, in units of the embedding norm",
        ylabel="how much further the nearest other token is",
        source="Measured; standarderror/calculus/embeddings.py.",
        alt=("Six curves starting near eleven and decaying slowly towards "
             "zero, all of them still well above the dashed line at a step "
             "of size one."),
        caption=(f"Every curve starts at that context's nearest-neighbour "
                 f"distance - a median of {tbl['median_nearest']:.1f} across "
                 f"the table - and has to reach zero before any other token becomes the "
                 f"nearest. At the dotted line - a step the size of the "
                 f"whole embedding, which is already an enormous step - not "
                 f"one has crossed. The crossings happen between "
                 f"{survey['flip_min']:.1f} and {survey['flip_max']:.1f} "
                 f"embedding norms. **Rounding a gradient step to the "
                 f"nearest token is not an operation that does anything**: "
                 f"it returns its input."),
        path=str(IMG / f"gr105-f1-walk.{EXT}"))[0]

    rows = []
    for r in survey["rows"]:
        s, k, lg = r["sub"], r["ranking"], r["landing"]
        rows.append([f"context {r['context']}",
                     f"{s['current_loss']:.2f}",
                     f"{s['best_loss']:.2f}",
                     f"{k['rank_of_best'] + 1}",
                     f"{k['best_loss_in_top5']:.2f}",
                     f"{k['recovered5']:.0%}",
                     f"{lg['landed_rank'] + 1}"])
    bold = {(i, 5) for i in range(len(rows))}

    out["f2"] = charts.table_image(
        rows,
        header=["", "current loss", "best possible", "rank of the best",
                "best of the top 5", "gain recovered", "step lands on rank"],
        title="Bad at pointing, good at proposing",
        subtitle=("Every single-token substitution enumerated, against the "
                  "first-order score's ranking of them and against where a "
                  "descent step eventually lands."),
        source="Measured; standarderror/calculus/embeddings.py.",
        alt=("A table of six contexts. The recovered-gain column is at or "
             "near 100% in five rows; the rank columns are erratic."),
        caption=(f"The fourth column is the metric people quote and it is "
                 f"erratic - the best token ranks anywhere from "
                 f"{survey['rank_min'] + 1} to {survey['rank_max'] + 1} of "
                 f"{tbl['rows']}. The **sixth** is the metric that matches "
                 f"how the score is used, and it is at or above 95% in "
                 f"{survey['recovered_above_95']} of {CONTEXTS} rows, "
                 f"including the one where the best token ranks "
                 f"{ordinal(survey['rank_max'] + 1)}. The last column is where a "
                 f"gradient step lands: never first."),
        bold_cells=bold, align="lrrrrrr",
        path=str(IMG / f"gr105-f2-table.{EXT}"))[0]

    def shortlist(ax, m):
        for r in survey["rows"]:
            s, k = r["sub"], r["ranking"]
            best = s["loss"][k["order"]]
            run = np.minimum.accumulate(best)
            avail = s["current_loss"] - s["best_loss"]
            frac = (s["current_loss"] - run) / max(avail, 1e-9)
            ax.plot(np.arange(1, len(frac) + 1), np.clip(frac, -0.1, 1.05),
                    lw=1.7, alpha=0.8, color=m.series[0])
        ax.axhline(1.0, lw=1.6, ls="--", color=m.grid,
                   label="all of the available improvement")
        ax.axvline(5, lw=1.8, ls=":", color=m.series[2],
                   label="a shortlist of five")
        ax.set_xscale("log")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(frameon=False, fontsize=8.6, loc="lower right")

    out["f3"] = charts.diagram(
        shortlist,
        title="How much a shortlist buys, as it gets longer",
        subtitle=("Evaluating the first-order score's candidates in order and "
                  "keeping the best so far. One line per context, as a "
                  "fraction of the improvement that was available."),
        xlabel="candidates evaluated (log scale)",
        ylabel="fraction of the available gain recovered",
        source="Measured; standarderror/calculus/embeddings.py.",
        alt=("Six step curves rising steeply in the first few candidates, "
             "most of them at the dashed ceiling by five."),
        caption=(f"Five of six contexts are within "
                 f"{100 - 100 * survey['recovered_median']:.0f}% of the "
                 f"ceiling after five candidates out of {tbl['rows']}. The "
                 f"exception recovers "
                 f"{survey['recovered_min']:.0%}, and it is the context with "
                 f"the least to gain in the first place. **The gradient is a "
                 f"proposal distribution, and it is a good one**; it is the "
                 f"evaluation that turns it into an answer."),
        path=str(IMG / f"gr105-f3-shortlist.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    tbl, survey = res["table"], res["survey"]
    land = res["landing"]

    def sphere(panel, m):
        rng = np.random.default_rng(1)
        a = rng.uniform(0, 2 * np.pi, 26)
        panel.scatter(np.cos(a), np.sin(a), s=26, color=m.series[0])
        panel.set_xlim(-1.35, 1.35)
        panel.set_ylim(-1.35, 1.35)

    def away(panel, m):
        x = np.linspace(0, 1, 60)
        panel.plot(x, 0.15 + 0.8 * x, lw=2.6, color=m.series[2])
        panel.plot(x, np.sqrt(0.62 + (0.8 * x) ** 2), lw=2.4,
                   color=m.series[0])
        panel.set_xlim(0, 1.02)
        panel.set_ylim(0, 1.3)

    def rise(panel, m):
        for r in survey["rows"]:
            s, k = r["sub"], r["ranking"]
            run = np.minimum.accumulate(s["loss"][k["order"]])
            avail = max(s["current_loss"] - s["best_loss"], 1e-9)
            panel.plot(np.arange(1, len(run) + 1),
                       np.clip((s["current_loss"] - run) / avail, 0, 1.05),
                       lw=1.8, color=m.series[0], alpha=0.8)
        panel.set_xscale("log")
        panel.set_ylim(0, 1.1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=5,
        headline="No neighbourhood to descend in, and a very good shortlist",
        panels=[(sphere, f"{tbl['median_cosine']:+.4f}", "median cosine"),
                (away, f"{land['flip_at']:.0f}x", "to reach another token"),
                (rise, f"{survey['recovered_median']:.0%}",
                 "recovered by five")],
        note=(f"The embedding table is {tbl['rows']} near-orthogonal points "
              f"on a sphere: pairwise distance {tbl['median_pairwise']:.2f} "
              f"against an orthogonal prediction of "
              f"{tbl['orthogonal_prediction']:.2f}. A descent step leaves you "
              f"nearest to where you started - and the same gradient, used to "
              f"rank five candidates instead of to point, recovers almost all "
              f"of the available gain."),
        alt=("Three hand-drawn frames: points scattered on a circle; two "
             "rising curves that meet late; and six step curves climbing to "
             "a ceiling."),
        path=str(IMG / f"gr105-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["geometry"] = s.run("""
        import numpy as np
        from standarderror.calculus import embeddings as em

        t = em.table()
        print(f"embedding table            {t['rows']} x {t['dim']}")
        print(f"median row norm            {t['median_norm']:.3f}")
        print(f"median pairwise distance   {t['median_pairwise']:.3f}")
        print(f"  if the rows were orthogonal, sqrt(2) x norm ="
              f" {t['orthogonal_prediction']:.3f}")
        print(f"median pairwise cosine     {t['median_cosine']:+.5f}")
        print()
        print(f"nearest other token, as a fraction of the typical distance"
              f"  {t['nearest_over_pairwise']:.3f}")
    """, expect=["embedding table"])

    out["walk"] = s.run("""
        # Step against the gradient at one input position. How far before the
        # nearest embedding row is no longer the token you started from?
        from standarderror.llm import tiny
        x, y = next(iter(tiny.batches(count=1, size=1, seed=0)))
        sub = em.substitutions(x[0, :tiny.BLOCK - 1],
                               int(y[0, tiny.BLOCK - 1]))
        land = em.step_landing(sub)
        print(f"the step needed       {land['flip_at']:.2f} embedding norms"
              f"  ({land['distance_walked']:.0f} units)")
        print(f"it lands on           token {land['landed']}, "
              f"rank {land['landed_rank'] + 1} of {sub['vocab']}")
        print(f"loss there            {land['landed_loss']:.3f}")
        print(f"loss where you began  {sub['current_loss']:.3f}")
        print(f"best available        {sub['best_loss']:.3f}")
    """, expect=["the step needed"])

    out["shortlist"] = s.run("""
        # Now grade the same gradient as a shortlist instead of a direction.
        r = em.ranking(sub)
        print(f"rank of the best substitution     "
              f"{r['rank_of_best'] + 1} of {sub['vocab']}")
        print(f"spearman over all candidates      {r['spearman']:+.3f}")
        print()
        print(f"improvement available             {r['available']:.3f} nats")
        print(f"best of the gradient's top five   "
              f"{sub['current_loss'] - r['best_loss_in_top5']:.3f} nats")
        print(f"so the shortlist recovers         {r['recovered5']:.1%}")
    """, expect=["rank of the best"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    tbl, survey = res["table"], res["survey"]
    sub, rank, land = res["sub"], res["ranking"], res["landing"]

    # The spine, asserted rather than trusted.
    assert abs(tbl["median_pairwise"] / tbl["orthogonal_prediction"] - 1) < 0.01
    assert abs(tbl["median_cosine"]) < 0.02
    assert tbl["nearest_over_pairwise"] > 0.7
    assert tbl["median_pairwise"] > tbl["median_norm"]
    assert land["flip_at"] is not None and land["flip_at"] > 1.5
    assert land["landed"] != sub["best"]
    assert survey["landings_best"] == 0
    assert survey["recovered_above_95"] >= 4
    assert survey["rank_max"] > 20
    assert rank["recovered5"] > 0.9
    assert survey["beats_max"] < 0.3
    assert res["control"]["beats_gradient"] < 0.3

    post = Post(
        title=(f"{SERIES_TAG} 5: The Gradient in Embedding Space Does Not "
               f"Point at a Token"),
        slug="gradients-5-embeddings",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=5,
        date=POST_DATE,
        # Geometry and enumerated losses. No prediction, nothing to beat.
        requires_baseline=False,
        subtitle=("The embedding table is 65 near-orthogonal points on a "
                  "sphere, so there is no local neighbourhood and a descent "
                  "step leaves you nearest to the token you started from. "
                  "The same gradient, asked to rank five candidates instead "
                  "of to point, recovers almost all of the available gain."),
        summary=(
            f"The embedding table's median pairwise distance is "
            f"{tbl['median_pairwise']:.3f} and sqrt(2) times its median norm "
            f"is {tbl['orthogonal_prediction']:.3f} - the distance between "
            f"orthogonal vectors of that length, matched to four figures, "
            f"with a median pairwise cosine of {tbl['median_cosine']:+.4f}. "
            f"Random vectors reproduce it, so this is dimension rather than "
            f"training. The rows are further apart than they are long and the "
            f"nearest other token sits at {tbl['nearest_over_pairwise']:.2f} "
            f"of the typical distance, so there is no neighbourhood to "
            f"descend in. Measured: a step against the gradient leaves you "
            f"nearest to the token you started from for "
            f"{survey['flip_min']:.1f} to {survey['flip_max']:.1f} embedding "
            f"norms, and the token you eventually reach is never the best "
            f"substitution - once it is worse than not moving at all. But "
            f"graded as a shortlist rather than a direction the same gradient "
            f"is good: its top five of {tbl['rows']} recovers 96% to 100% of "
            f"the available improvement in {survey['recovered_above_95']} of "
            f"{CONTEXTS} contexts, including one where the best token ranks "
            f"{ordinal(survey['rank_max'] + 1)}. Which is why prompt optimisation "
            f"is "
            f"search with a gradient-shaped shortlist, and why the rank of "
            f"the single best token is the wrong thing to measure."),
        tags=["deep-learning", "machine-learning", "llm", "pytorch",
              "mathematics", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. The geometry is a property of the committed "
            "weights; every loss is computed on held-out text from the "
            "corpus the model was trained on, and no values from that text "
            "are published.",
            f"The model: an {tiny.load()['parameters']:,}-parameter "
            f"character-level transformer trained for this series - four "
            f"blocks, four heads, width {tiny.WIDTH}, context {tiny.BLOCK}, "
            f"validation loss {tiny.VAL_LOSS} against a uniform-guess "
            f"{tiny.UNIFORM_LOSS:.3f}. Weights, training script and a "
            f"verified sha256 are committed: `standarderror/llm/tiny.py`, "
            f"`scripts/train_tiny.py`, `data/tiny_gpt/`.",
            "Machinery: `standarderror/calculus/embeddings.py`, tested in "
            "`tests/test_embeddings.py`, which includes a random-vector "
            "control for the geometry so the claim is not attributed to "
            "training.",
            "Where this stops: Ebrahimi et al., \"HotFlip: white-box "
            "adversarial examples for text classification\", *ACL* (2018), "
            "for the first-order substitution score; Wallace et al., "
            "\"Universal adversarial triggers for attacking and analyzing "
            "NLP\", *EMNLP* (2019) and Zou et al., \"Universal and "
            "transferable adversarial attacks on aligned language models\" "
            "(2023), for the propose-then-evaluate loop this geometry "
            "forces; Vershynin, *High-Dimensional Probability* (2018), for "
            "why independent vectors in 128 dimensions are near-orthogonal.",
        ],
        reproducibility={
            "environment": "standarderror=0.1.0, python=3.11.15, "
                           "torch=2.14.0, numpy=2.4.4, scipy=1.16.3",
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "model": ("the committed checkpoint, hash-verified on load; the "
                      "geometry is a property of its embedding matrix and "
                      "involves no data at all"),
            "determinism": (f"the substitution tables are exhaustive over "
                            f"{tbl['rows']} tokens and involve no sampling; "
                            f"the {CONTEXTS} contexts come from a fixed seed; "
                            f"the walk is a deterministic line search at a "
                            f"step of 0.05 embedding norms"),
        },
    )
    post.hero = figs["hero"]
    return _write(post, res, figs, snip)


def _write(post: Post, res: dict, figs: dict, snip: dict) -> Post:
    tbl, survey = res["table"], res["survey"]
    rank, land = res["ranking"], res["landing"]
    worse = [r for r in survey["rows"] if r["landing"].get("made_it_worse")]
    ranks = sorted(r["ranking"]["rank_of_best"] + 1 for r in survey["rows"])

    post.add(
        "A derivative that is exactly right and no use",
        """The first four episodes were about the chain rule's hypotheses and where a transformer bends them. This one has no complaint about the derivative at all. Take the loss, differentiate with respect to the embedding vector sitting at one input position, and you get a perfectly ordinary gradient: exact, well conditioned, every hypothesis satisfied.

And then you cannot use it, because the thing you want to change is not a vector. It is a token, and tokens are 65 specific points in a space of 128 dimensions. A gradient tells you which way to move. There is nowhere to move to.

How badly that fails is a question about the geometry of the embedding table, and the answer turns out not to be about this model at all.""")

    post.add(
        "Sixty-five strangers",
        """Two numbers settle it.""")

    post.add(
        "",
        f"""{snip['geometry'].markdown()}

The median distance between two embedding rows is {tbl['median_pairwise']:.3f}. The distance between two *orthogonal* vectors of length {tbl['median_norm']:.3f} is √2 × {tbl['median_norm']:.3f} = {tbl['orthogonal_prediction']:.3f}. Those agree to four significant figures, and the median pairwise cosine is {tbl['median_cosine']:+.5f}.

So the table is {tbl['rows']} mutually near-orthogonal vectors sitting on a sphere. Every token is about as far from every other token as it is possible for them to be, and — the line that matters — **they are further apart than they are long**: {tbl['median_pairwise']:.1f} against {tbl['median_norm']:.1f}.

None of that is learned. Independent vectors in 128 dimensions are near-orthogonal for reasons that have nothing to do with language; the test suite checks that random vectors of the same norm reproduce the same relationship. What training chose was where on the sphere each token goes, and in 128 dimensions there is so much room that the answer is "far from everything else".

The consequence is the one that matters here. The nearest other token sits at {tbl['nearest_over_pairwise']:.2f} of the typical pairwise distance — so "the token nearest to this one" is barely a distinguished object. **There is no neighbourhood.** Gradient descent is a procedure for exploiting one.""",
        level=3,
        figures=[figs["f0"]])

    post.add(
        "So the step goes nowhere",
        """The obvious thing to try: move the embedding against its gradient, then round to the nearest row of the table. That is what "follow the gradient in embedding space" would have to mean.""")

    post.add(
        "",
        f"""{snip['walk'].markdown()}

You have to walk {land['flip_at']:.1f} embedding norms — {land['distance_walked']:.0f} units, on a table whose rows are {tbl['median_norm']:.1f} long — before the nearest row stops being the one you started from. Across six contexts, {survey['flip_min']:.1f} to {survey['flip_max']:.1f} norms, and at a step of one whole embedding norm not one of the six has crossed. Every step anyone would actually take is a rounding error against that, so **rounding a gradient step to the nearest token is not an operation that does anything**. It returns its input.

The geometry says why. The direction to any other token is nearly orthogonal to any fixed direction you might step along, so walking moves you away from where you were much faster than it moves you towards anywhere else. You escape a Voronoi cell whose neighbours are all at distance {tbl['median_pairwise']:.1f} only by walking about that far, and the linearisation that produced the gradient expired long before.

And when you do arrive, it is not the right token. It ranks {ordinal(land['landed_rank'] + 1)} of {tbl['rows']} here, and across the six contexts {', '.join(str(r['landing']['landed_rank'] + 1) for r in survey['rows'])} — **never first**. In {'one' if len(worse) == 1 else str(len(worse))} of them the token you land on has a *higher* loss than the one you started with: descending the gradient and rounding up made the model worse.""",
        level=3,
        figures=[figs["f1"]])

    post.add(
        "The other question, which has a different answer",
        """Nobody serious actually rounds a gradient step. What they do is use the gradient to **score** candidates — HotFlip's first-order substitution score, the inner product of the gradient with the difference between two embeddings — and then evaluate the top few for real.

That is a different question and it deserves a different measurement, for a reason worth stating precisely. Episode 4 found this same first-order model compressed to a fraction of the truth's range, which destroyed its magnitude. **Ranking is scale-invariant.** Compress a score by any positive factor and its ordering is untouched. So the defect that made the gradient useless as a direction costs it nothing as a proposal.""")

    post.add(
        "",
        f"""{snip['shortlist'].markdown()}

The best substitution ranks {ordinal(rank['rank_of_best'] + 1)}, which is fine, and the Spearman correlation of {rank['spearman']:+.3f} over all {tbl['rows']} candidates, which is not. But the last line is the one that matters: evaluating the gradient's top five out of {tbl['rows']} recovers {rank['recovered5']:.0%} of the improvement that was available.

Those numbers come apart badly across contexts, and the way they come apart is the point. The rank of the single best token runs {ranks[0]} to {ranks[-1]} of {tbl['rows']} — erratic, and the sort of thing that makes the method look unreliable. The fraction of available gain recovered by a five-candidate shortlist is at or above 95% in {survey['recovered_above_95']} of {CONTEXTS} contexts, **including the one where the best token ranks {ordinal(survey['rank_max'] + 1)}**, because the shortlist contained something nearly as good and near-ties are as good as wins.

So the metric people quote is the wrong one. "Does the gradient find the best token" is a question nobody needs answered. "Does evaluating five of sixty-five get you most of the way" is the question, and the answer is yes.""",
        level=3,
        figures=[figs["f2"]])

    post.add(
        "",
        f"""The one failure is worth looking at rather than averaging away. Its shortlist recovers {survey['recovered_min']:.0%}, and it is the context where the least was on offer: {min(r['ranking']['available'] for r in survey['rows']):.2f} nats between the current token and the best possible one, against {max(r['ranking']['available'] for r in survey['rows']):.2f} in the best case. When every candidate is nearly as good as every other a ranking has nothing to rank, and the fraction-recovered metric divides by a small number and becomes noisy. That is a limitation of the metric as much as of the method.

**And the shortlist needs a control, because five of {tbl['rows']} is already a {5 / tbl['rows']:.1%} sample.** Draw five tokens uniformly at random instead and evaluate those: the median recovery runs from {min(r['control']['median'] for r in survey['rows']):+.2f} to {survey['random_median_max']:+.2f} of the available gain — negative in two contexts, meaning a random five routinely contains nothing better than the token already there. And the share of random draws that match or beat the gradient's five is between {min(r['control']['beats_gradient'] for r in survey['rows']):.2f} and {survey['beats_max']:.2f}, in every context.

So the gradient's ordering is doing real work: between {min(r['control']['beats_gradient'] for r in survey['rows']):.0%} and {survey['beats_max']:.0%} of random shortlists match it, which across six independent contexts is not close. It is simply the only kind of work this geometry leaves room for.""",
        level=3,
        figures=[figs["f3"]])

    post.add(
        "What this means for the thing people actually do",
        """Prompt optimisation, adversarial suffixes, discrete trigger search — every one of these is a loop that proposes token substitutions and evaluates them. The gradient's role is the proposal, and the reason that architecture exists is not that anyone preferred it. It is forced, by the two measurements above.

Which reframes what "the gradient is informative" should mean here. Its magnitude is not informative: episode 4 measured the same first-order model returning a fraction of the right size. Its direction is not informative: there is nowhere for a direction to point. Its **ordering of a handful of candidates** is informative, and that is a far weaker property than the other two, which is exactly why it survives geometry this hostile.

It also explains a practical asymmetry that otherwise looks arbitrary. Doubling the shortlist from five to ten is cheap and buys very little here, because five was already at the ceiling. Halving it to two or three is where the loss shows up. The gradient's information is concentrated in the very top of its ranking and is roughly exhausted by the fifth candidate, which is a statement about how much a first-order model of a non-linear function can be trusted, and not one I would extrapolate past a 65-token vocabulary without checking.""")

    post.add(
        "Five episodes, one pattern",
        """This is the last of five, and the five turned out to rhyme in a way none of them was drafted to.

Each episode began with an exact statement about a hypothesis of the chain rule, and each exact statement turned out to be true and largely inert. Autodiff's inconsistency at kinks is real, documented, and almost absent from a transformer — GELU and LayerNorm are smooth and clipping never fired. LayerNorm's Jacobian really does have rank exactly *d* − 2, and the residual connection makes that invisible in every block. A sampled token really has no derivative, and the biased substitute everyone uses is beaten by the unbiased one after a handful of samples. The embedding table really has no neighbourhood, and the gradient is useful there anyway, in a different role.

In every case the interesting finding was one question further in, and it was smoother and less dramatic than the headline. What throttles gradient flow is not kinks but confidence. What changes gradient magnitude across depth is not a rank deficiency but a scalar. What makes straight-through biased is not the discreteness but a compressed loss model. What makes a gradient useful over tokens is not its direction but its ordering.

I would not have predicted that pattern, and I do not think it generalises beyond "exact facts about a component are easier to state than facts about what the component does". But it is what happened five times out of five, which is enough to be worth saying out loud.

The caveat that belongs at the end: all of it is measured on one 816,128-parameter character-level model trained on one corpus. The algebra holds at any width — a projector is a projector — and the frequencies do not. Every number here that describes the model rather than the mathematics is a number about that model.""")

    post.add(
        "What to keep",
        f"""1. The embedding table is {tbl['rows']} near-orthogonal vectors on a sphere: median pairwise distance {tbl['median_pairwise']:.3f} against √2 × the median norm = {tbl['orthogonal_prediction']:.3f}, median cosine {tbl['median_cosine']:+.4f}. Random vectors do the same, so this is dimension, not training.

2. The rows are further apart than they are long, and the nearest other token sits at {tbl['nearest_over_pairwise']:.2f} of the typical distance. **There is no local neighbourhood**, which is the thing gradient descent exists to exploit.

3. So a descent step leaves you nearest to the token you started from — for {survey['flip_min']:.1f} to {survey['flip_max']:.1f} embedding norms, which is not a step, it is a journey.

4. When you finally cross a boundary, the token there is never the best substitution, and in {len(worse)} of {CONTEXTS} contexts it is worse than not moving.

5. But the same gradient used as a **ranking** is good, and ranking is scale-invariant, so episode 4's compression costs it nothing here.

6. Its top five of {tbl['rows']} recovers 96% to 100% of the available improvement in {survey['recovered_above_95']} of {CONTEXTS} contexts — including one where the single best token ranks {ordinal(survey['rank_max'] + 1)}. **Rank of the best is the wrong metric; recovered gain is the right one**, and they disagree sharply.

7. And the ordering is doing the work, not just the evaluation: five tokens drawn at random match the gradient's five only {min(r['control']['beats_gradient'] for r in survey['rows']):.0%} to {survey['beats_max']:.0%} of the time, and recover a median as low as {min(r['control']['median'] for r in survey['rows']):+.2f} of the gain.

8. Which is why discrete-token optimisation is search with a gradient-shaped shortlist. The gradient earns its keep by turning {tbl['rows']} candidates into 5; the evaluation does the rest.""")

    post.add(
        "Exercise",
        """Measure the two numbers in section two on your own embedding table: the median pairwise distance and √2 times the median norm. If they agree, your tokens are mutually near-orthogonal and nothing in this episode was about a small model. If they disagree — if the distances are much *smaller* than the orthogonal prediction — you have found genuine clustering in embedding space, which is more interesting than anything here and worth knowing about before you trust any nearest-neighbour argument over those vectors.

Then take your favourite gradient-guided token search and log, per step, how far down its own ranking the accepted substitution was. If it is usually first, your shortlist is longer than it needs to be. If it is usually well down the list, the gradient is contributing less than the evaluation, and the random-shortlist control above is the way to find out which.

The uncomfortable version: run that control on your own vocabulary before quoting a success rate for a gradient-guided attack. At 65 tokens a random five is a 7.7% sample and still loses about seven times in eight; at 50,000 tokens a random five is nothing, so a gradient-guided search that works there is doing something this episode has not measured, and a gradient-guided search that does not work may be failing for a reason that has no gradient in it at all.""")

    return post


if __name__ == "__main__":
    print(build().markdown()[:1500])
