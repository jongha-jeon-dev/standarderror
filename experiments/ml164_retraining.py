"""ML 64: How Much History to Retrain On Depends on How Fast the World Moves.

The sixty-fourth episode of *Machine Learning, Taught Through What
Breaks*, in Arc XIII. "Retrain regularly on the most recent data and the
model stays current." Measured on simulated streams in which the relation
between ten inputs and the outcome stays fixed, wanders slowly, wanders fast
or jumps once:

* In a stable world, retraining on only the last period's rows cost 5 points
  of accuracy against keeping all the history (0.729 against 0.779), and the
  model never retrained after the first ten periods did as well as any.
* The best window was all of history when nothing changed, the last 16
  periods under slow drift and the last 4 under fast drift or a jump. No
  single window was best in every kind of world.
* When labels arrived eight periods late, the gain of the best policy over
  never retraining fell from 0.168 to 0.096 under fast drift.

Run: `standarderror run ml164_retraining --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import deploy
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 27)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

LABEL = {"static": "never retrained", "expanding": "all history",
         "window 1": "last period", "window 4": "last 4 periods",
         "window 16": "last 16 periods"}


def compute() -> dict:
    return {"rt": deploy.retraining()}


def _backtest(rt, world, split=30):
    """Choose the policy with the best mean accuracy over the first `split`
    scored periods (averaged over the streams), and report how each policy
    did over the rest."""
    first = {p: float(np.mean(rt[f"{world}/0"]["curve"][p][:split]))
             for p in deploy.POLICIES}
    second = {p: float(np.mean(rt[f"{world}/0"]["curve"][p][split:]))
              for p in deploy.POLICIES}
    chosen = max(first, key=first.get)
    best = max(second, key=second.get)
    return {"chosen": chosen, "best": best, "chosen_score": second[chosen],
            "best_score": second[best]}


def _recovery(curve, warm, periods, tol=0.02):
    """Periods after the jump until accuracy is within `tol` of its mean
    over the ten periods before it."""
    c = np.array(curve)
    jump = periods // 2 - warm
    pre = c[jump - 10:jump].mean()
    for k in range(jump, len(c)):
        if c[k] >= pre - tol and np.all(c[k:k + 3] >= pre - tol):
            return k - jump
    return None


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    rt = res["rt"]["results"]
    rec = {p: _recovery(rt["jump/0"]["curve"][p], 10, res["rt"]["periods"])
           for p in deploy.POLICIES}

    def jump(ax, m):
        cur = rt["jump/0"]["curve"]
        x = np.arange(10, 10 + len(cur["static"]))
        for k, pol in enumerate(deploy.POLICIES):
            ax.plot(x, cur[pol], color=m.series[k], lw=1.6, label=LABEL[pol])
        ax.axvline(40, color=m.ink, ls=":", lw=1)
        ax.text(41, 0.42, "relation jumps", fontsize=8)
        ax.set_ylim(0.4, 0.85)
        ax.legend(fontsize=8, loc="lower left", frameon=False, ncol=2)

    out["f0"] = charts.diagram(
        jump,
        title="After a jump, recent data recover; old data hold the model back",
        subtitle=("Accuracy in each period of a simulated stream whose "
                  "relation changes direction once, at period 40. 60 new "
                  "labelled rows a period, ten inputs; means over six "
                  "streams."),
        xlabel="period", ylabel="accuracy on that period's rows",
        figsize=(7.6, 4.4),
        source="Simulated; standarderror/ml/deploy.py.",
        alt=("Five curves near 0.75 to 0.8 before period 40; after it all "
             "fall, the short windows recover within a few periods, all "
             "history and the never-retrained model stay low through "
             "period 79."),
        caption=(f"Periods to recover after the jump: last period "
                 f"{rec['window 1']}, last 4 {rec['window 4']}, last 16 "
                 f"{rec['window 16']}; all history and the never-retrained "
                 f"model did not recover by period 79."),
        path=str(IMG / f"ml164-f0-jump.{EXT}"))[0]

    body = []
    for world in deploy.WORLDS:
        for d in (0, 8):
            v = rt[f"{world}/{d}"]["mean"]
            best = max(v, key=v.get)
            body.append([world, str(d)] + [
                (f"**{v[p]:.3f}**" if p == best else f"{v[p]:.3f}")
                for p in deploy.POLICIES])
    out["f1"] = charts.table_image(
        [[c.replace("**", "") for c in row] for row in body],
        header=["world", "label delay"] + [LABEL[p] for p in deploy.POLICIES],
        title="Five retraining policies, four worlds, two delays",
        subtitle=("Mean accuracy from period 10 to 79 over six simulated "
                  "streams. A label delay of 8 means each retraining sees "
                  "only rows at least eight periods old."),
        source="Simulated; standarderror/ml/deploy.py.",
        alt=("An eight-row table. All history is best in the stable world, "
             "the last 16 periods under slow drift and the last 4 under fast "
             "drift and after a jump; the delay lowers every retrained "
             "policy under change."),
        caption="Each world's best policy is the largest number in its row.",
        align="llrrrrr",
        bold_cells={(i, j) for i, row in enumerate(body)
                    for j, c in enumerate(row) if c.startswith("**")},
        path=str(IMG / f"ml164-f1-table.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    rt = res["rt"]["results"]

    def p1(panel, m):
        v = rt["stable/0"]["mean"]
        panel.bar([0, 1], [v["expanding"], v["window 1"]],
                  color=[m.series[0], m.series[1]])
        panel.set_ylim(0.6, 0.8)

    def p2(panel, m):
        bests = [max(rt[f"{w}/0"]["mean"], key=rt[f"{w}/0"]["mean"].get)
                 for w in deploy.WORLDS]
        panel.bar(range(4), [1, 1, 1, 1],
                  color=[m.series[deploy.POLICIES.index(b)] for b in bests])

    def p3(panel, m):
        a, b = rt["fast drift/0"]["mean"], rt["fast drift/8"]["mean"]
        panel.bar([0, 1], [a["window 4"] - a["static"],
                           b["window 4"] - b["static"]],
                  color=[m.series[2], m.series[3]])

    a, b = rt["fast drift/0"]["mean"], rt["fast drift/8"]["mean"]
    s = rt["stable/0"]["mean"]
    return charts.lecture_hero(
        series=SERIES_TAG, episode=64,
        headline="How much history",
        panels=[(p1, f"-{s['expanding'] - s['window 1']:.3f}",
                 "last period only, stable"),
                (p2, "3 windows", "best, across 4 worlds"),
                (p3, f"{(b['window 4'] - b['static']) / (a['window 4'] - a['static']):.0%}",
                 "of the gain, labels late")],
        note=("Recent data track change and old data average out noise. "
              "Which matters more depends on how fast the world moves, and "
              "late labels take part of either away."),
        alt=("Three hand-drawn frames: a bar and a shorter one; four bars of "
             "three colours; a bar and a shorter one."),
        path=str(IMG / f"ml164-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["stream"] = s.run("""
        import numpy as np
        from standarderror.ml import deploy

        rng = np.random.default_rng([0, 2])            # one fast-drift stream
        Xs, ys = deploy._stream("fast drift", rng, periods=80, rows=60, dim=10)
        print(len(Xs), "periods of", Xs[0].shape[0], "rows,",
              Xs[0].shape[1], "inputs; outcome rate in period 0:",
              round(float(ys[0].mean()), 2))
        """)
    out["small"] = s.run("""
        r = deploy.retraining(delays=(0,), reps=2)["results"]
        for world in ("stable", "fast drift"):
            v = r[f"{world}/0"]["mean"]
            print(world)
            print("  " + "  ".join(f"{p} {a:.3f}" for p, a in v.items()))
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    meta, rt = res["rt"], res["rt"]["results"]
    st, sl = rt["stable/0"]["mean"], rt["slow drift/0"]["mean"]
    fa, ju = rt["fast drift/0"]["mean"], rt["jump/0"]["mean"]
    fa8, ju8, st8 = (rt["fast drift/8"]["mean"], rt["jump/8"]["mean"],
                     rt["stable/8"]["mean"])
    best = {w: max(rt[f"{w}/0"]["mean"], key=rt[f"{w}/0"]["mean"].get)
            for w in deploy.WORLDS}
    gain0 = fa["window 4"] - fa["static"]
    gain8 = fa8["window 4"] - fa8["static"]
    rec = {p: _recovery(rt["jump/0"]["curve"][p], 10, meta["periods"])
           for p in deploy.POLICIES}
    rec8 = {p: _recovery(rt["jump/8"]["curve"][p], 10, meta["periods"])
            for p in deploy.POLICIES}
    bt = {w: _backtest(rt, w) for w in deploy.WORLDS}
    post_jump = float(np.mean(rt["jump/0"]["curve"]["static"][35:]))

    # The spine, asserted rather than trusted.
    assert best == {"stable": "expanding", "slow drift": "window 16",
                    "fast drift": "window 4", "jump": "window 4"}
    assert st["expanding"] - st["window 1"] > 0.04
    assert abs(st["static"] - st["expanding"]) < 0.005
    assert sl["window 16"] > sl["expanding"] and sl["window 1"] < sl["static"]
    assert fa["window 4"] - fa["expanding"] > 0.08
    assert 0.15 < gain0 < 0.18 and 0.08 < gain8 < 0.11
    assert all(fa8[p] < fa[p] for p in deploy.POLICIES if p != "static")
    assert abs(st8["expanding"] - st["expanding"]) < 0.005
    assert rec["static"] is None and rec["window 4"] is not None
    assert rec["window 1"] <= rec["window 4"] < rec["window 16"]
    assert rec["expanding"] is None and rec8["expanding"] is None
    assert rec8["window 4"] >= rec["window 4"] + 8
    assert bt["stable"]["chosen"] == "expanding" == bt["stable"]["best"]
    assert bt["fast drift"]["chosen"] == "window 4" == bt["fast drift"]["best"]
    assert bt["slow drift"]["chosen"] != bt["slow drift"]["best"]
    assert bt["jump"]["best_score"] - bt["jump"]["chosen_score"] > 0.1
    assert 0.5 < post_jump < 0.58

    post = Post(
        title=(f"{SERIES_TAG} 64: How Much History to Retrain On Depends on "
               f"How Fast the World Moves"),
        slug="ml-64-retraining",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=64,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=(f"Retraining on only the latest data cost "
                  f"{100 * (st['expanding'] - st['window 1']):.0f} points in a "
                  f"world that never changed, and all history cost "
                  f"{100 * (fa['window 4'] - fa['expanding']):.0f} in one "
                  "that changed fast. The right window is a fact about the "
                  "world, and late labels shrink every window's benefit."),
        summary=(
            f"A logistic model retrained every period on simulated streams of "
            f"60 rows a period. In a stable world, the last period alone gave "
            f"accuracy {st['window 1']:.3f} against {st['expanding']:.3f} for "
            f"all history, and never retraining {st['static']:.3f}. The best "
            f"policy was all history when nothing changed, the last 16 "
            f"periods under slow drift ({sl['window 16']:.3f}) and the last 4 "
            f"under fast drift ({fa['window 4']:.3f}, against "
            f"{fa['expanding']:.3f} for all history) and after a jump. With "
            f"labels eight periods late, the best policy's gain over never "
            f"retraining under fast drift fell from {gain0:.3f} to "
            f"{gain8:.3f}, and recovery after a jump took {rec8['window 4']} "
            f"periods instead of {rec['window 4']}."),
        tags=["machine-learning", "retraining", "concept-drift", "mlops",
              "monitoring", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: ten standard Gaussian inputs, a logistic outcome whose "
            "coefficient vector has length 2 throughout; per period it stays "
            "fixed, takes a random step of standard deviation 0.05 or 0.2 per "
            "coordinate (then is rescaled), or jumps once to a new random "
            "direction at period 40. 80 periods of 60 rows; six streams per "
            "world.",
            "Machinery: scikit-learn's LogisticRegression, refitted each "
            "period from period 10 on; `standarderror/ml/deploy.py`, tested "
            "in `tests/test_ml.py`.",
            "Where this stops: Widmer and Kubat, \"Learning in the presence "
            "of concept drift and hidden contexts\", *Machine Learning* "
            "(1996); Gama, Žliobaitė, Bifet, Pechenizkiy and Bouchachia, \"A "
            "survey on concept drift adaptation\", *ACM Computing Surveys* "
            "(2014); Klinkenberg and Joachims, \"Detecting concept drift "
            "with support vector machines\", *ICML* (2000); episodes 36, 50 "
            "and 61 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; about a minute of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The retraining schedule",
        f"""Episode 61 found that a change in how outcomes depend on inputs shows up only in labelled outcomes. The standard defence does not try to detect it at all: retrain on a schedule, on recent data, so the model follows the world wherever it goes. It is a sentence with two parts that can each fail: *retrain regularly on the most recent data and the model stays current*.

The test is a stream. Each period brings 60 new rows with ten inputs and a logistic outcome. In four kinds of world the relation between inputs and outcome stays fixed; wanders slowly; wanders fast; or jumps once, halfway, to a new direction. The relation's strength is held constant throughout, so only its direction changes. After ten periods of history a logistic model is retrained every period under five policies: never again (the ten-period model kept forever), all labelled history, or only the last 1, 4 or 16 periods. Each is scored on the next period's rows, and everything is averaged over six streams per world.

{snip['stream'].markdown()}""")

    post.add(
        "Recent is not always better",
        f"""In the stable world, the policy that sounds most current was the worst: retraining on the last period alone, 60 rows, gave accuracy {st['window 1']:.3f}, against {st['expanding']:.3f} for all history. Nothing was changing, so discarding old data only discarded sample size: a ten-input model fitted on 60 rows is noisy, and each period's model was a fresh draw of that noise. The model that was never retrained at all scored {st['static']:.3f}, as good as any.

Under slow drift the order changed, and the best policy was a long window: the last 16 periods scored {sl['window 16']:.3f}, all history {sl['expanding']:.3f}, the last period {sl['window 1']:.3f}. Under fast drift, all history fell to {fa['expanding']:.3f}, dragged towards relations that no longer held, and the last 4 periods scored {fa['window 4']:.3f}. After a jump, the last 4 periods were best again ({ju['window 4']:.3f}).

{snip['small'].markdown()}

So the best window was all history, 16, 4 and 4 in the four worlds, and no window was best in every kind of world. This is the bias-variance trade of episode 9 along the time axis: a long window averages out noise and is biased by old relations, a short one tracks change and is noisy. Widmer and Kubat built adaptive windows for this in 1996, and Klinkenberg and Joachims chose window sizes by estimating error on recent data; both work by measuring how fast the world is moving, which a fixed schedule does not.""",
        figures=[figs["f1"]])

    post.add(
        "Choosing the window from the past",
        f"""The practical fix is to choose the window the way Klinkenberg and Joachims did, by its record: backtest each policy on the history you have and keep the winner. Done here on the first 30 scored periods of each world (periods 10 to 39, averaged over the six streams) and judged on the remaining 40, it chose all history in the stable world, which was right, and the four-period window under fast drift, also right ({bt['fast drift']['chosen_score']:.3f}).

It chose wrong in the other two. Under slow drift the first 30 periods favoured all history, because the relation had not yet moved far, and over the next 40 that choice scored {bt['slow drift']['chosen_score']:.3f} against {bt['slow drift']['best_score']:.3f} for the sixteen-period window. In the world that jumped, the first 30 periods were all before the jump and favoured all history, which then scored {bt['jump']['chosen_score']:.3f} over the remaining periods against {bt['jump']['best_score']:.3f} for the four-period window. A backtest measures how fast the world moved, not how fast it will move; it works when the pace of change is itself stable.

The never-retrained model is the other reference point. In the stable world it lost nothing; under fast drift it scored {fa['static']:.3f}, and after the jump its accuracy fell to {post_jump:.3f}, not far above a coin, because the old direction said little about the new one. A model that is never retrained bets that the world is stable, and the bet is either free or very expensive.""")

    post.add(
        "Late labels",
        f"""The schedule also assumes labels are there to retrain on. A loan defaults months after the decision, a churned customer is known only after the renewal date, and a fraud label waits for an investigation. With labels arriving eight periods late, each retraining could use only rows at least eight periods old.

In the stable world the delay cost nothing ({st8['expanding']:.3f} against {st['expanding']:.3f} for all history), since old rows were as good as new. Under fast drift every retrained policy got worse, and the best one's advantage over never retraining fell from {gain0:.3f} to {gain8:.3f}: the delay took away {1 - gain8 / gain0:.0%} of what retraining bought. Averaged over the stream with the jump, the four-period window scored {ju8['window 4']:.3f} with late labels against {ju['window 4']:.3f} with prompt ones. After the jump, the four-period window took {rec['window 4']} periods to recover with immediate labels and {rec8['window 4']} with late ones, {rec8['window 4'] - rec['window 4']} more, against a delay of eight; all history did not recover within the 40 periods after the jump either way. Every period of delay is a period the model spends wrong before it can even begin to learn the new relation, and nothing in the inputs can shorten it, as episode 61 found.""",
        figures=[figs["f0"]])

    post.add(
        "Where this breaks",
        """**Logistic model, logistic world.** The model was correctly specified in every period, so its only problems were noise and staleness. A misspecified model adds a third: more data of the wrong kind cannot fix it.

**Fixed windows.** Weighting rows by age, or retraining only when monitored performance drops, sit between the policies measured here; their best settings depend on the drift rate in the same way.

**Constant volume.** Sixty rows a period made short windows noisy. With far more rows per period, short windows lose less in a stable world.

**Six streams per cell.** Differences of under a point between neighbouring windows are within the noise; the ordering of the best policies across worlds was not.""")

    post.add(
        "What to keep",
        f"""1. In a stable world, retraining on the last period alone cost {st['expanding'] - st['window 1']:.3f} of accuracy against all history, and never retraining cost nothing.

2. The best window was all history, the last 16 periods, the last 4 and the last 4 across four worlds; no fixed choice was best everywhere.

3. Labels eight periods late took {1 - gain8 / gain0:.0%} of retraining's gain under fast drift and added {rec8['window 4'] - rec['window 4']} periods to the recovery after a jump.

4. Choose a retraining window by measuring how fast your relation changes, from performance on labelled outcomes over time. Plan around your label delay; shorten it where you can, because it adds to every period a changed model spends wrong.""")

    post.add(
        "Exercise",
        """Take a model you retrain and backtest three windows on your own history: all of it, the last year, the last quarter. For each, retrain at every past retraining date using only data whose labels would have been available then, and score the next period.

If all of history wins, your world is stable enough that frequent retraining buys little. If the shortest window wins, it is moving fast, and the label delay is the number to reduce.""")

    post.add(
        "Next",
        """Most of this series' verdicts came from simulations built to isolate one mechanism, with real data as a check where scikit-learn's bundled sets allowed one. Episode 65 takes four verdicts that were measured only on simulations and repeats them on real data, to see which carry over.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
