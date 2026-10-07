"""Coverage 3: Temperature Scaling Cannot Change What You Predict.

The third episode of the uncertainty series. Dividing every logit by the same
positive number is a strictly increasing map within each prediction, so no
answer can change -- accuracy is identical to the bit at every temperature.
What it can change is the order of confidence *between* predictions, which is
the order abstention uses. Whether that matters depends on which uncertainty
score you abstain on.

Measured:

* Accuracy bit-identical across seven temperatures from 0.5 to 3; ECE from
  0.026 to 0.375 over the same range.
* Between predictions, max-probability ordering moves: at T = 2, Kendall's
  tau against T = 1 is 0.857 and 22.5% of the most confident one per cent
  leaves it. A two-prediction example shows the mechanism: one close rival
  against a crowd of distant ones.
* Abstaining on max probability, the effect is small near T = 1 and grows
  slowly. Abstaining on entropy, it is large: AURC 44% worse at T = 3.
  Ranking by logit margin is exactly invariant.
* Temperature scaling fitted on NLL here picks T of about 1.1 -- which is
  also the best temperature for max-probability abstention, and the wrong
  direction for entropy, whose best is near 0.6.

Run: `standarderror run cv103_temperature --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np

import standarderror as se
from standarderror.llm import tiny
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.uncertainty import calibration as cb
from standarderror.uncertainty import coverage as cv
from standarderror.viz import charts

POST_DATE = date(2026, 10, 7)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Uncertainty for Language Models, Taught Through What Breaks"
SERIES_TAG = "Coverage"

BATCHES, BATCH_SIZE = 24, 16
TEMPS = (0.5, 1.0, 1.25, 1.5, 2.0, 3.0)
KEEP = 0.2


def compute() -> dict:
    table = cb.temperature_table()
    pred = cv.predictions(count=BATCHES, size=BATCH_SIZE, seed=1)
    p, y, seq = pred["p"], pred["y"], pred["sequence"]
    fitted = cb.fitted_temperature(p, y, seq, halves=40)
    t_fit = round(fitted["mean"], 2)
    correct = p.argmax(1) == y
    return {
        "table": table,
        "reorder": cb.reordering(table),
        "pred": pred,
        "abstain": cb.abstention(p, y, seq, temperatures=TEMPS, draws=300),
        "fitted": fitted,
        "t_fit": t_fit,
        "at_fit": {k: cb.selective(f(cb.temperature(p, t_fit)), correct)
                   for k, f in cb.SCORES.items()},
        "best": {k: cb.best_temperature(p, y, k) for k in cb.SCORES},
        "swap": cb.swap_pair(),
    }


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    rows = res["table"]["rows"]
    acc = rows[0]["accuracy"]

    def knob(ax, m):
        ts = [r["temperature"] for r in rows]
        ax.plot(ts, [r["ece"] for r in rows], marker="o", ms=6, lw=2.4,
                color=m.series[2], label="expected calibration error")
        ax.plot(ts, [r["mean_confidence"] - r["accuracy"] for r in rows],
                marker="s", ms=6, lw=2.0, ls="--", color=m.series[0],
                label="mean confidence minus accuracy")
        ax.axhline(0, lw=1, color=m.ink_secondary)
        ax.annotate(f"accuracy {acc:.4f} at every temperature, to the bit",
                    (0.5, -0.30), fontsize=9.2, color=m.ink_secondary)
        ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    out["f0"] = charts.diagram(
        knob,
        title="A knob that moves every probability and no answer",
        subtitle=(f"{res['table']['predictions']:,} next-character "
                  f"predictions from the committed model, logits divided by "
                  f"T before the softmax."),
        xlabel="temperature T",
        ylabel="calibration gap",
        source="Measured; standarderror/uncertainty/calibration.py.",
        alt=("A U-shaped ECE curve with its minimum at T = 1 and a falling "
             "line of confidence minus accuracy crossing zero near T = 1."),
        caption=(f"ECE runs from {rows[2]['ece']:.3f} at T = 1 to "
                 f"{rows[-1]['ece']:.3f} at T = 3, and the model goes from "
                 f"overconfident to underconfident through the same range. "
                 f"Accuracy is **{acc:.4f} at all seven temperatures** - the "
                 f"same predictions, bit for bit."),
        path=str(IMG / f"cv103-f0-knob.{EXT}"))[0]

    sw = res["swap"]
    ts = np.linspace(0.3, 3.0, 120)

    def top(z, t):
        e = np.exp((z - z.max()) / t)
        return (e / e.sum()).max()

    def swap(ax, m):
        ax.plot(ts, [top(sw["one_rival"], t) for t in ts], lw=2.6,
                color=m.series[0], label="one close rival: logits 2.0, 1.5")
        ax.plot(ts, [top(sw["crowd"], t) for t in ts], lw=2.6,
                color=m.series[2],
                label="a crowd of four: logits 3.0, 0, 0, 0, 0")
        ax.axvline(1.0, lw=1.2, ls=":", color=m.ink_secondary)
        ax.legend(frameon=False, fontsize=8.6, loc="upper right")

    out["f1"] = charts.diagram(
        swap,
        title="Two predictions trade places as the temperature rises",
        subtitle=("Top-label probability of two five-label predictions. "
                  "Same argmax for each at every temperature."),
        xlabel="temperature T",
        ylabel="confidence in the top label",
        source="Constructed; standarderror/uncertainty/calibration.py.",
        alt=("Two falling curves that cross between T = 1 and T = 2; the "
             "crowd curve starts higher and ends lower."),
        caption=(f"At T = 1 the crowd row is the more confident, "
                 f"{sw['confidence'][1.0][1]:.3f} against "
                 f"{sw['confidence'][1.0][0]:.3f}. At T = 2 the order has "
                 f"flipped. Heat spreads probability across rivals, and a "
                 f"crowd of four takes more of it than one rival does - "
                 f"**neither prediction changed**."),
        path=str(IMG / f"cv103-f1-swap.{EXT}"))[0]

    best, ab = res["best"], res["abstain"]

    def curves(ax, m):
        for k, colour in (("max probability", m.series[0]),
                          ("entropy", m.series[2]),
                          ("margin", m.series[1])):
            cur = best[k]["curve"]
            flat = max(cur.values()) - min(cur.values()) < 1e-12
            # A flat curve has no best temperature; argmin would report the
            # first grid point, which reads as a finding and is not one.
            label = (f"{k}: identical at every T" if flat else
                     f"{k}: best at T = {best[k]['temperature']:.1f}")
            ax.plot(list(cur), list(cur.values()), lw=2.4, color=colour,
                    label=label)
        ax.axvspan(res["fitted"]["low"], res["fitted"]["high"], color=m.grid,
                   alpha=0.35, label=(f"temperature scaling picks "
                                      f"{res['t_fit']:.2f}"))
        ax.axhline(ab["oracle"], lw=1.6, ls="--", color=m.ink_secondary,
                   label=f"an oracle, {ab['oracle']:.3f}")
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    out["f2"] = charts.diagram(
        curves,
        title="The temperature that calibrates is not always the one that abstains",
        subtitle=(f"Area under the risk-coverage curve on "
                  f"{ab['rows']:,} predictions, keeping the most confident "
                  f"first by each score. Lower is better."),
        xlabel="temperature T",
        ylabel="AURC",
        source="Measured; standarderror/uncertainty/calibration.py.",
        alt=("A flat margin line, a shallow max-probability curve with its "
             "minimum inside the shaded fitted-temperature band, and an "
             "entropy curve rising steeply to the right of 0.6."),
        caption=(f"Max probability abstains best at T = "
                 f"{best['max probability']['temperature']:.1f}, inside the "
                 f"band temperature scaling picks. Entropy abstains best at "
                 f"{best['entropy']['temperature']:.1f} and **gets worse in "
                 f"exactly the direction calibration moves it**. Margin does "
                 f"not move at all."),
        path=str(IMG / f"cv103-f2-curves.{EXT}"))[0]

    t = ab["table"]
    rows_t = []
    for k in cb.SCORES:
        for temp in (1.0, 1.25, 2.0, 3.0):
            r = t[(k, temp)]
            ci = ("-" if temp == 1.0 else
                  f"{r['difference']:+.4f} [{r['low']:+.4f}, {r['high']:+.4f}]")
            rows_t.append([k, f"{temp:g}", f"{r['aurc']:.4f}", ci,
                           f"{r['accuracy_at'][KEEP]:.3f}"])
    bold = {(i, c) for i, r in enumerate(rows_t) for c in range(5)
            if r[0] == "entropy" and r[1] == "3"}
    out["f3"] = charts.table_image(
        rows_t,
        header=["abstain on", "T", "AURC", "change from T = 1 [95%]",
                f"accuracy of the top {KEEP:.0%}"],
        title="What heating the model does to what it declines",
        subtitle=(f"Intervals resample the {ab['sequences']} sequences, not "
                  f"the {ab['rows']:,} predictions, because predictions in "
                  f"one window share a context."),
        source="Measured; standarderror/uncertainty/calibration.py.",
        alt=("A table of twelve rows: margin unchanged everywhere, max "
             "probability changing little, entropy changing a lot at higher "
             "temperatures."),
        caption=("The **bold** row is the size of the risk: abstaining on "
                 "entropy after heating the model by three. Max probability "
                 "at T = 1.25 is inside the noise; margin is identical by "
                 "construction."),
        bold_cells=bold, align="lrrlr",
        path=str(IMG / f"cv103-f3-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    rows = res["table"]["rows"]
    best = res["best"]

    def flat(panel, m):
        ts = [r["temperature"] for r in rows]
        panel.plot(ts, [r["ece"] for r in rows], lw=2.8, color=m.series[2])
        panel.plot(ts, [0.02] * len(ts), lw=2.4, color=m.series[0])

    def cross(panel, m):
        ts = np.linspace(0.4, 3.0, 40)
        sw = res["swap"]

        def top(z, t):
            e = np.exp((z - z.max()) / t)
            return (e / e.sum()).max()
        panel.plot(ts, [top(sw["one_rival"], t) for t in ts], lw=2.8,
                   color=m.series[0])
        panel.plot(ts, [top(sw["crowd"], t) for t in ts], lw=2.8,
                   color=m.series[2])

    def rise(panel, m):
        cur = best["entropy"]["curve"]
        panel.plot(list(cur), list(cur.values()), lw=2.8, color=m.series[2])
        cur = best["max probability"]["curve"]
        panel.plot(list(cur), list(cur.values()), lw=2.4, color=m.series[0])

    ent3 = res["abstain"]["table"][("entropy", 3.0)]
    ent1 = res["abstain"]["table"][("entropy", 1.0)]
    return charts.lecture_hero(
        series=SERIES_TAG, episode=3,
        headline="Temperature changes no answer, and which ones you refuse",
        panels=[(flat, f"{rows[0]['accuracy']:.4f}", "accuracy at every T"),
                (cross, "2 rows", "trade places when heated"),
                (rise, f"+{ent3['aurc'] / ent1['aurc'] - 1:.0%}",
                 "entropy AURC at T = 3")],
        note=("Dividing every logit by T cannot change an argmax, so "
              "accuracy is identical at every temperature. It does reorder "
              "confidence between predictions, which is what abstention "
              "uses: on max probability the effect is small, on entropy it "
              "is large, and on logit margin it is zero."),
        alt=("Three hand-drawn frames: a flat line under a U-shaped curve; two "
             "curves crossing; and a steeply rising curve above a flat one."),
        path=str(IMG / f"cv103-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["knob"] = s.run("""
        from standarderror.uncertainty import calibration as cb

        # Divide every logit by T, then softmax. 10,240 predictions.
        table = cb.temperature_table()
        for r in table["rows"]:
            print(f"T = {r['temperature']:4.2f}   accuracy {r['accuracy']:.4f}"
                  f"   NLL {r['nll']:.4f}   ECE {r['ece']:.4f}")
        print("identical predictions at every T:", table["accuracy_identical"])
        """)

    out["reorder"] = s.run("""
        # Within a prediction nothing moves. Between predictions:
        for r in cb.reordering(table):
            if r["temperature"] in (1.25, 2.0, 3.0):
                print(f"T = {r['temperature']:4.2f}   Kendall tau {r['tau']:.3f}"
                      f"   pairs reordered {r['pairs_reordered']:.1%}"
                      f"   top 1% kept {r['top_overlap']:.1%}")
        """)

    out["swap"] = s.run("""
        pair = cb.swap_pair()
        for t, (one, crowd) in pair["confidence"].items():
            first = "one rival" if one > crowd else "crowd"
            print(f"T = {t:3.1f}   one rival {one:.3f}   crowd {crowd:.3f}"
                  f"   more confident: {first}")
        """)

    out["abstain"] = s.run(f"""
        from standarderror.uncertainty import coverage as cv

        pred = cv.predictions(count={BATCHES}, size={BATCH_SIZE}, seed=1)
        p, y, seq = pred["p"], pred["y"], pred["sequence"]
        ab = cb.abstention(p, y, seq, temperatures={TEMPS!r}, draws=300)
        for score in ("max probability", "entropy", "margin"):
            for t in (1.25, 2.0, 3.0):
                r = ab["table"][(score, t)]
                print(f"{{score:15s}} T = {{t:4.2f}}  AURC {{r['aurc']:.4f}}  "
                      f"change {{r['difference']:+.4f}} "
                      f"[{{r['low']:+.4f}}, {{r['high']:+.4f}}]")
        print(f"oracle {{ab['oracle']:.4f}}")
        """)

    out["fitted"] = s.run("""
        fit = cb.fitted_temperature(p, y, seq, halves=40)
        print(f"temperature scaling picks T = {fit['mean']:.3f} "
              f"(range {fit['low']:.3f} to {fit['high']:.3f}, 40 halves)")
        for score in ("max probability", "entropy"):
            b = cb.best_temperature(p, y, score)
            print(f"{score:15s} abstains best at T = {b['temperature']:.1f}"
                  f"   AURC {b['aurc']:.4f}")
        # Margin has no best temperature: its curve is flat, so argmin
        # would just return the first grid point.
        m = cb.best_temperature(p, y, "margin")["curve"].values()
        print(f"margin          AURC {min(m):.4f} to {max(m):.4f}"
              f" over T = 0.2 to 3.0")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    tab, rows = res["table"], res["table"]["rows"]
    by_t = {r["temperature"]: r for r in rows}
    reo = {r["temperature"]: r for r in res["reorder"]}
    ab, t = res["abstain"], res["abstain"]["table"]
    fit, best, sw = res["fitted"], res["best"], res["swap"]
    mp1, en1, mg1 = (t[(k, 1.0)] for k in ("max probability", "entropy",
                                           "margin"))
    mp3, en3 = t[("max probability", 3.0)], t[("entropy", 3.0)]
    mp125, en125 = t[("max probability", 1.25)], t[("entropy", 1.25)]
    at_fit = res["at_fit"]

    # The spine, asserted rather than trusted.
    assert tab["accuracy_identical"] and len(tab["accuracies"]) == 1
    assert by_t[1.0]["ece"] < 0.04 and by_t[3.0]["ece"] > 0.3
    assert reo[2.0]["tau"] < 0.9 and reo[2.0]["top_overlap"] < 0.85
    assert sw["confidence"][1.0][1] > sw["confidence"][1.0][0]
    assert sw["confidence"][2.0][0] > sw["confidence"][2.0][1]
    assert en3["low"] > 0.05 and en125["low"] > 0
    assert mp125["low"] < 0 < mp125["high"]
    assert mp3["difference"] < 0.25 * en3["difference"]
    assert t[("margin", 3.0)]["difference"] == 0.0
    assert 1.03 < fit["mean"] < 1.2
    assert abs(best["max probability"]["temperature"] - fit["mean"]) < 0.15
    assert best["entropy"]["temperature"] < 0.9
    assert mg1["aurc"] > mp1["aurc"]
    # Exactly flat, not approximately: the 1e-12 log floor broke this at 0.2.
    assert len(set(best["margin"]["curve"].values())) == 1

    post = Post(
        title=(f"{SERIES_TAG} 3: Temperature Scaling Cannot Change What You "
               f"Predict"),
        slug="coverage-3-temperature",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=3,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Dividing every logit by the same number cannot change a "
                  "single answer. It can change which answers you decline to "
                  "give - a little if you abstain on max probability, a lot "
                  "if you abstain on entropy, and not at all on margin."),
        summary=(
            f"On {tab['predictions']:,} next-character predictions from the "
            f"committed model, accuracy is {by_t[1.0]['accuracy']:.4f} at all "
            f"seven temperatures from 0.5 to 3, identical to the bit, while "
            f"ECE runs from {by_t[1.0]['ece']:.3f} to {by_t[3.0]['ece']:.3f}. "
            f"Between predictions the confidence order moves: at T = 2 "
            f"Kendall's tau against T = 1 is {reo[2.0]['tau']:.3f} and "
            f"{1 - reo[2.0]['top_overlap']:.1%} of the most confident one per "
            f"cent leaves it. On {ab['rows']:,} predictions, abstaining on "
            f"max probability barely notices - AURC changes by "
            f"{mp125['difference']:+.4f} at T = 1.25, inside the noise - "
            f"while abstaining on entropy gets "
            f"{en3['aurc'] / en1['aurc'] - 1:.0%} worse at T = 3, and ranking "
            f"by logit margin cannot move at all. Temperature scaling fitted "
            f"on NLL picks T = {fit['mean']:.2f}, which is also the best "
            f"temperature for max-probability abstention and the wrong "
            f"direction for entropy, whose best is "
            f"{best['entropy']['temperature']:.1f}."),
        tags=["machine-learning", "data-science", "statistics",
              "calibration", "uncertainty", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every number is computed from the committed "
            "model's predictions on held-out text; no values from the text "
            "are published.",
            f"The model: an {tiny.load()['parameters']:,}-parameter "
            f"character-level transformer - four blocks, four heads, width "
            f"{tiny.WIDTH}, context {tiny.BLOCK}, validation loss "
            f"{tiny.VAL_LOSS} against a uniform-guess "
            f"{tiny.UNIFORM_LOSS:.3f}. Weights, training script and a "
            f"verified sha256 are committed: `standarderror/llm/tiny.py`, "
            f"`scripts/train_tiny.py`, `data/tiny_gpt/`.",
            "Machinery: `standarderror/uncertainty/calibration.py`, tested in "
            "`tests/test_uncertainty.py`, which pins the bit-identical "
            "accuracy, the swap, the margin invariance and the abstention "
            "differences with their intervals.",
            "Where this stops: Guo et al., \"On calibration of modern neural "
            "networks\", *ICML* (2017), for temperature scaling; El-Yaniv and "
            "Wiener, \"On the foundations of noise-free selective "
            "classification\", *JMLR* (2010), and Geifman and El-Yaniv, "
            "\"Selective classification for deep neural networks\", "
            "*NeurIPS* (2017), for selective prediction and risk-coverage "
            "curves.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python=3.13.16, torch=2.14.0, numpy=2.4.4"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "model": ("the committed checkpoint, hash-verified on load, so "
                      "every number is measured on the same weights"),
            "determinism": ("the abstention intervals resample the 384 "
                            "sequences 300 times with a fixed seed; the "
                            "fitted temperature is the mean over 40 seeded "
                            "random halves"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A knob that cannot change an answer",
        f"""Temperature scaling is the standard first fix for a miscalibrated classifier, and it comes with a guarantee that sounds like reassurance: it cannot change a single prediction. Divide every logit by the same positive number T before the softmax. Within one prediction that is a strictly increasing map, so the largest logit stays the largest, and so does every ordering below it. The argmax cannot move.

The guarantee is exact, and it is easy to check rather than trust.

{snip['knob'].markdown()}

Accuracy is {by_t[1.0]['accuracy']:.4f} at all seven temperatures — not approximately, identically: the same predicted character at every position, at every T. Meanwhile ECE runs from {by_t[1.0]['ece']:.3f} at T = 1 to {by_t[3.0]['ece']:.3f} at T = 3, and {by_t[0.5]['ece']:.3f} at T = 0.5. The knob moves every probability and no answer.

That is usually where the description stops: temperature fixes calibration and costs nothing. This episode is about the one thing that sentence leaves out.""",
        figures=[figs["f0"]])

    post.add(
        "The order between predictions moves",
        f"""The guarantee is *within* a prediction. Plenty of what a deployed model does compares *across* predictions: abstain on the least confident 10%, send the bottom fifth to a reviewer, surface the top-confidence answers first. All of those sort predictions by a confidence score, and nothing in the guarantee says that sort survives a change of temperature.

{snip['reorder'].markdown()}

At T = 2, Kendall's tau between the old and new confidence orderings is {reo[2.0]['tau']:.3f}: {reo[2.0]['pairs_reordered']:.1%} of all pairs of predictions swap which one is more confident, and {1 - reo[2.0]['top_overlap']:.1%} of the predictions that were in the most confident one per cent are no longer in it. The answers did not change. The queue did.

The mechanism fits in two predictions over five labels.

{snip['swap'].markdown()}

The first prediction has one close rival; the second beats a crowd of four distant ones. Cold, the crowd barely registers and the second is more confident. Heated, probability spreads across rivals, and four rivals absorb more of it than one does, so the order flips. Neither top label changed at any temperature. Real predictions sit everywhere between those two shapes, which is why the reordering is partial and grows with T.""",
        figures=[figs["f1"]])

    post.add(
        "Does the order matter? Ask the abstention",
        f"""A reordering is only a problem if it reorders the wrong way, so the useful measurement is not tau but what an abstaining system loses. Sort predictions by a confidence score, keep the most confident first, and track the error rate of what you kept as you keep more. The area under that risk-coverage curve, AURC, summarises it; lower is better, and an oracle that keeps every right answer before any wrong one scores {ab['oracle']:.3f} here.

Three scores, because three are in common use: the probability of the top label, the entropy of the whole distribution, and the logit margin between the top two labels. The intervals resample the {ab['sequences']} sequences rather than the {ab['rows']:,} predictions, because predictions inside one 64-character window share their context and are not independent draws.

{snip['abstain'].markdown()}

The three scores behave completely differently.

**Max probability barely notices.** At T = 1.25 its AURC changes by {mp125['difference']:+.4f}, with an interval [{mp125['low']:+.4f}, {mp125['high']:+.4f}] that contains zero. Even at T = 3 the loss is {mp3['difference']:+.4f}, {mp3['aurc'] / mp1['aurc'] - 1:.0%} — and the accuracy of the most confident fifth goes from {mp1['accuracy_at'][KEEP]:.3f} to {mp3['accuracy_at'][KEEP]:.3f}.

**Entropy is fragile.** Already at T = 1.25 it is measurably worse ({en125['difference']:+.4f}, interval [{en125['low']:+.4f}, {en125['high']:+.4f}]), and at T = 3 it is {en3['difference']:+.4f}, **{en3['aurc'] / en1['aurc'] - 1:.0%} worse**, with the top fifth at {en3['accuracy_at'][KEEP]:.3f} accuracy. Entropy is the score most exposed to the crowd: it sums over every label, and heating the model hands the crowd of irrelevant labels a larger say in it.

**Margin cannot move.** The gap between the top two logits is divided by T like every logit, so every margin shrinks by the same factor and the order of margins is unchanged, exactly. It is not free, though: margin's AURC is {mg1['aurc']:.4f} against {mp1['aurc']:.4f} for max probability at T = 1. Immune and best are different properties.""",
        figures=[figs["f3"]])

    post.add(
        "What temperature scaling would actually choose",
        f"""Everything above sweeps T by hand. In practice T is fitted, by minimising NLL on held-out data, so the question that matters is where that fit lands and what it does to each score.

{snip['fitted'].markdown()}

Fitted on random halves of the sequences, temperature scaling picks T = {fit['mean']:.3f}, between {fit['low']:.3f} and {fit['high']:.3f} across forty halves. This model is close to calibrated already, so the correction is small.

For max probability that is good news twice over: its own best temperature for abstention is {best['max probability']['temperature']:.1f}, inside the fitted range. Calibrating the probabilities and ordering them for abstention want the same thing here.

For entropy they want opposite things. Its best temperature is {best['entropy']['temperature']:.1f} — *colder* than the model as trained — and temperature scaling moves it the other way. At the fitted T, entropy's AURC is {at_fit['entropy']['aurc']:.4f} against {en1['aurc']:.4f} at T = 1 and {best['entropy']['aurc']:.4f} at its own optimum. Small, at this model's small correction. It would not be small for a model that needed a large one.

Margin's curve is flat: {min(best['margin']['curve'].values()):.4f} at every one of the 29 temperatures, identical to the bit. Getting there took one fix. An earlier version floored probabilities at 1e-12 before taking logs; at T = 0.2 one row in twenty has its runner-up below that floor, those rows tie there, and margin's AURC moved by 5e-5 — enough for the code to report a "best" temperature for a score that has none. The theory was right and the guard against log(0) was not. The floor is now 1e-300, and the test that pins margin's invariance runs at T = 0.2.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        f"""**This model barely needs temperature scaling.** A fitted T of {fit['mean']:.2f} is a mild correction, so the practical damage at the fitted temperature is small. Models that are badly overconfident get fitted temperatures well above one, and the effects in the T = 2 and T = 3 rows are the ones they would see. Whether *this* architecture becomes overconfident with longer training is a separate question and the subject of episode 4.

**AURC averages over every coverage.** A deployed system abstains at one coverage, not all of them. The accuracy of the kept top fifth is reported alongside for that reason, and at a specific operating point the picture can be better or worse than the average.

**The swap pair is constructed.** It shows a mechanism, not a frequency. The frequency is the tau and the top-one-per-cent overlap, measured on real predictions.

**Three scores, not all of them.** Ensembles, Monte Carlo dropout and learned confidence heads each have their own relationship to temperature. The argument generalises — any score that depends on more than the ordering of logits within a row can be reordered by T — but the sizes here are only for these three.""")

    post.add(
        "What to keep",
        f"""1. Temperature scaling cannot change a prediction. Accuracy here is {by_t[1.0]['accuracy']:.4f} at every temperature from 0.5 to 3, to the bit.

2. It does change calibration, a lot: ECE from {by_t[1.0]['ece']:.3f} to {by_t[3.0]['ece']:.3f} across the same range.

3. It reorders confidence *between* predictions. At T = 2, {reo[2.0]['pairs_reordered']:.1%} of pairs swap and {1 - reo[2.0]['top_overlap']:.1%} of the top one per cent leaves it.

4. The mechanism is one close rival against a crowd of distant ones. Heat favours the prediction with fewer rivals.

5. Abstaining on max probability, the effect is small: inside the noise at T = 1.25, {mp3['aurc'] / mp1['aurc'] - 1:.0%} worse AURC at T = 3.

6. Abstaining on entropy, it is large: worse beyond the noise at T = 1.25, {en3['aurc'] / en1['aurc'] - 1:.0%} worse at T = 3. Logit margin is exactly invariant.

7. Fitted temperature here is {fit['mean']:.2f}, which is also max probability's best abstention temperature — and the wrong direction for entropy, whose best is {best['entropy']['temperature']:.1f}.""")

    post.add(
        "Exercise",
        """If you temperature-scale a model and also abstain, find out which score your abstention uses. If it is max probability, refit nothing and sleep well. If it is entropy, rerun the risk-coverage curve after scaling, because the fit that improved your calibration may have quietly worsened the queue.

Then compute the same curve with logit margin. It is one subtraction, it is immune to temperature by construction, and if it is close to your current score you have a choice that removes the question altogether.

The uncomfortable version: if your system calibrates and abstains in two separate places — the calibration fitted by one team, the threshold set by another — check that the threshold was set *after* the calibration. A threshold chosen on uncalibrated confidences and applied to calibrated ones is choosing a different set of predictions from the one anybody looked at.""")

    post.add(
        "Next",
        f"""This model needed a temperature of only {fit['mean']:.2f}. Episode 4 asks why: is overconfidence a property of the architecture, or does it arrive at a particular point in training — when validation loss stops improving and the model starts memorising? Answering that needs a training run with checkpoints, which is what the episode is built on.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
