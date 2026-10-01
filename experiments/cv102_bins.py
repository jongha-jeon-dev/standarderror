"""Coverage 2: Your Calibration Error Is Mostly Your Bin Count.

The second episode of the uncertainty series. Expected calibration error is
an estimator with a floor, the floor belongs to the model's own confidence
profile, and at the sample sizes calibration claims are usually made on, no
binned estimator can tell this model from a calibrated one.

Measured:

* A model calibrated by construction scores ECE 0.085 at n = 200, 0.035 at
  1,000 and 0.008 at 20,000 (15 bins, averaged over twenty draws). The fitted
  log-log slope against bins/n is 0.503 -- the binomial prediction is 1/2.
  Equal-mass bins make it larger, not smaller.
* The floor depends on the confidence profile, so no table of floors is
  portable; the right null is the model's own probabilities with labels
  resampled from them.
* On the committed model at n = 1,000, most of the measured ECE is that
  floor, and the null test detects the (real) miscalibration in about a
  third of subsets. It takes about 4,000 points to detect it reliably, and
  fifty bins lose power.
* Comparing two temperatures of one model -- identical accuracy, so the
  full-data NLL settles which is better calibrated -- ECE on 1,000 points
  picks the wrong one a third of the time. NLL much less often.
* Debiasing (subtract each bin's binomial variance) removes the floor on
  average and makes the estimate independent of bin count on the full data.
  On 1,000 points it is about as noisy as the quantity it estimates.

Run: `standarderror run cv102_bins --publish`
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

POST_DATE = date(2026, 10, 1)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Uncertainty for Language Models, Taught Through What Breaks"
SERIES_TAG = "Coverage"

BATCHES, BATCH_SIZE = 24, 16
BINS = (5, 15, 50)
SIZES = (250, 500, 1000, 2000, 4000, 8000)
SUBSET = 1000
WARM = 1.15


def compute() -> dict:
    pred = cv.predictions(count=BATCHES, size=BATCH_SIZE, seed=1)
    p, y = pred["p"], pred["y"]
    rng = np.random.default_rng(3)
    subsets = [rng.permutation(len(y))[:SUBSET] for _ in range(30)]
    shares = {b: [cb.null_test(p[i], y[i], bins=b, draws=100,
                               seed=17 + k)["floor_share"]
                  for k, i in enumerate(subsets)] for b in BINS}
    profile = {c: float(np.mean([cb.ece(*cb._conf_correct(
        *cb.calibrated_draw(SUBSET, c, seed=s)), bins=15) for s in range(20)]))
        for c in (10, 65)}
    return {
        "pred": pred,
        "sweep": cb.bias_sweep(),
        "sweep_mass": cb.bias_sweep(adaptive=True),
        "single": cb.ece(*cb._conf_correct(*cb.calibrated_draw(200, 10,
                                                               seed=0)),
                         bins=15),
        "profile": profile,
        # Same draws and seed as the snippet, so prose and printout agree.
        "full": {b: cb.null_test(p, y, bins=b, draws=400, seed=5)
                 for b in BINS},
        "example": {b: cb.null_test(p[subsets[0]], y[subsets[0]], bins=b,
                                    draws=400, seed=5) for b in BINS},
        "example_null": _null_draws(p[subsets[0]], 15, 400, 5),
        "shares": {b: float(np.mean(v)) for b, v in shares.items()},
        "power": {(n, b): cb.detection_rate(p, y, n=n, bins=b, subsets=100,
                                            seed=11)
                  for n in SIZES for b in BINS},
        "warm_full": {b: (cb.ece(*cb._conf_correct(p, y), bins=b),
                          cb.ece(*cb._conf_correct(cb.temperature(p, WARM), y),
                                 bins=b)) for b in BINS},
        "ranking": cb.ranking(p, cb.temperature(p, WARM), y, n=SUBSET,
                              bins=BINS, subsets=300, seed=2),
        "stability": cb.stability(p, y, n=SUBSET, bins=BINS, subsets=40,
                                  seed=4),
        "full_l2": {b: (cb.l2_error(*cb._conf_correct(p, y), bins=b,
                                    debiased=False),
                        cb.l2_error(*cb._conf_correct(p, y), bins=b))
                    for b in BINS},
    }


def _null_draws(p, bins, draws, seed):
    rng = np.random.default_rng(seed)
    conf = p.max(1)
    return np.array([cb.ece(conf, (p.argmax(1) == cb.resample_labels(p, rng))
                            .astype(float), bins=bins)
                     for _ in range(draws)])


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    sw, sm = res["sweep"], res["sweep_mass"]

    def floor(ax, m):
        for s, name, colour, mk in ((sw, "equal-width bins", m.series[0], "o"),
                                    (sm, "equal-mass bins", m.series[2], "s")):
            x = [b / n for (n, b) in s["grid"]]
            y = list(s["grid"].values())
            ax.scatter(x, y, s=34, color=colour, marker=mk, label=name,
                       zorder=3)
        xs = np.geomspace(min(b / n for (n, b) in sw["grid"]),
                          max(b / n for (n, b) in sw["grid"]), 50)
        ax.plot(xs, sw["constant"] * xs ** sw["slope"], lw=1.8, ls="--",
                color=m.grid, label=(f"fit: {sw['constant']:.2f} x "
                                     f"(bins/n)^{sw['slope']:.3f}"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    out["f0"] = charts.diagram(
        floor,
        title="A perfectly calibrated model, and the error ECE reports for it",
        subtitle=("Labels drawn from the model's own probabilities, so the true "
                  "calibration error is exactly zero. Five sample sizes, five "
                  "bin counts, twenty draws each."),
        xlabel="bins / n",
        ylabel="ECE of a calibrated model (log)",
        source="Measured; standarderror/uncertainty/calibration.py.",
        alt=("Points on a straight line in log-log axes rising with bins over "
             "n, with equal-mass bins sitting above equal-width ones."),
        caption=(f"Every point should be zero. The fitted slope is "
                 f"{sw['slope']:.3f} against a binomial prediction of one "
                 f"half, so the floor is predictable - and equal-mass bins, "
                 f"the usual first fix, sit **above** equal-width ones "
                 f"(slope {sm['slope']:.3f})."),
        path=str(IMG / f"cv102-f0-floor.{EXT}"))[0]

    ex, nd = res["example"][15], res["example_null"]

    def null(ax, m):
        ax.hist(nd, bins=30, color=m.series[0], alpha=0.85,
                label="a calibrated model with these confidences")
        ax.axvline(ex["ece"], lw=2.4, color=m.series[2],
                   label=f"the model, measured: {ex['ece']:.4f}")
        ax.axvline(nd.mean(), lw=1.8, ls="--", color=m.grid,
                   label=f"the floor: {nd.mean():.4f}")
        ax.legend(frameon=False, fontsize=8.6, loc="upper right")

    out["f1"] = charts.diagram(
        null,
        title="The model's ECE, beside what it would score if it were perfect",
        subtitle=(f"One subset of {SUBSET:,} predictions, 15 bins. The "
                  f"histogram resamples the labels from the model's own "
                  f"probabilities 400 times."),
        xlabel="ECE",
        ylabel="resampled calibrated models",
        source="Measured; standarderror/uncertainty/calibration.py.",
        alt=("A histogram of calibrated-model ECE values with the measured "
             "value marked inside its right half."),
        caption=(f"Measured {ex['ece']:.4f}; a perfectly calibrated model "
                 f"with exactly these confidences averages "
                 f"{nd.mean():.4f}, and exceeds the measured value in "
                 f"{ex['p_value']:.0%} of draws. On this subset there is "
                 f"**no evidence of miscalibration at all**."),
        path=str(IMG / f"cv102-f1-null.{EXT}"))[0]

    pw = res["power"]

    def power(ax, m):
        for b, colour, mk in zip(BINS, (m.series[0], m.series[1], m.series[2]),
                                 ("o", "s", "^")):
            ax.plot(SIZES, [pw[(n, b)] for n in SIZES], marker=mk, ms=6,
                    lw=2.2, color=colour, label=f"{b} bins")
        ax.axhline(0.8, lw=1.6, ls="--", color=m.grid,
                   label="80%, the usual bar for a test worth running")
        ax.axvline(SUBSET, lw=1.2, ls=":", color=m.ink_secondary)
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels([f"{n:,}" for n in SIZES], fontsize=8.4)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8.6, loc="lower right")

    out["f2"] = charts.diagram(
        power,
        title="How many predictions it takes to see a real miscalibration",
        subtitle=("The model is miscalibrated on all 24,576 points. Share of "
                  "random subsets of each size in which the null test "
                  "says so at the 5% level."),
        xlabel="predictions in the evaluation set",
        ylabel="share of subsets that detect it",
        source="Measured; standarderror/uncertainty/calibration.py.",
        alt=("Three rising curves, reaching the dashed 80% line between two "
             "and four thousand points, the fifty-bin curve lowest."),
        caption=(f"At {SUBSET:,} predictions the test detects a "
                 f"miscalibration that is really there in "
                 f"{pw[(SUBSET, 15)]:.0%} of subsets at 15 bins. At 4,000 it "
                 f"is {pw[(4000, 15)]:.0%}. **More bins cost power** - at "
                 f"4,000 points, 50 bins detect it "
                 f"{pw[(4000, 50)]:.0%} of the time."),
        path=str(IMG / f"cv102-f2-power.{EXT}"))[0]

    rk = res["ranking"]
    rows = [[k, f"{1 - v:.0%}"] for k, v in rk["prefers_a"].items()]
    bold = {(i, c) for i, (k, _) in enumerate(rows) for c in range(2)
            if k == "NLL"}
    out["f3"] = charts.table_image(
        rows,
        header=["metric, on 1,000 predictions",
                f"chose the better-calibrated model (T = {WARM})"],
        title="Which of two temperatures is better calibrated?",
        subtitle=(f"Same model at T = 1 and T = {WARM}: identical accuracy, "
                  f"so the full-data NLL difference is purely calibration "
                  f"and settles it. {rk['subsets']} random subsets."),
        source="Measured; standarderror/uncertainty/calibration.py.",
        alt=("A table of metrics with the share of subsets in which each "
             "picked the right model; the binned estimators are lowest."),
        caption=(f"The **bold** row is the only metric that does clearly "
                 f"better than the binned estimators: NLL picks the right "
                 f"model in {1 - rk['prefers_a']['NLL']:.0%} of subsets. "
                 f"Brier, also a proper score, manages "
                 f"{1 - rk['prefers_a']['Brier']:.0%} - no better than "
                 f"five-bin ECE - and the debiased estimator is *worse* "
                 f"than the plug-in at every bin count."),
        bold_cells=bold, align="lr",
        path=str(IMG / f"cv102-f3-ranking.{EXT}"))[0]

    st, fl = res["stability"], res["full_l2"]

    def trade(ax, m):
        xs = np.arange(len(BINS))
        ax.errorbar(xs - 0.08, [st[b]["l2"] for b in BINS],
                    yerr=[st[b]["l2_sd"] for b in BINS], fmt="o", ms=7,
                    capsize=4, lw=2, color=m.series[2],
                    label="plug-in, 1,000 points")
        ax.errorbar(xs + 0.08, [st[b]["debiased"] for b in BINS],
                    yerr=[st[b]["debiased_sd"] for b in BINS], fmt="s", ms=7,
                    capsize=4, lw=2, color=m.series[0],
                    label="debiased, 1,000 points")
        ax.plot(xs, [fl[b][1] for b in BINS], lw=1.8, ls="--", color=m.grid,
                label="debiased, all 24,576 points")
        ax.axhline(0, lw=1, color=m.ink_secondary)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b} bins" for b in BINS])
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    out["f4"] = charts.diagram(
        trade,
        title="Debiasing trades a wrong number for a noisy one",
        subtitle=("Root binned squared calibration error of the committed "
                  "model over forty random subsets of 1,000 predictions: "
                  "aggregate in squared units, bars one standard deviation "
                  "of a single subset's estimate."),
        xlabel="bin count",
        ylabel="calibration error (root, signed)",
        source="Measured; standarderror/uncertainty/calibration.py.",
        alt=("Plug-in points rising steeply with bin count with short bars; "
             "debiased points level with the dashed full-data line, with "
             "bars reaching below zero."),
        caption=(f"The plug-in moves from {st[5]['l2']:.3f} to "
                 f"{st[50]['l2']:.3f} with the bin count. The debiased "
                 f"estimate, aggregated, sits on the full-data value at every "
                 f"bin count - but one 1,000-point estimate has a spread of "
                 f"{st[15]['debiased_sd']:.3f} at 15 bins, and "
                 f"**{st[15]['negative']:.0%} of them come out below zero** "
                 f"on a model that is genuinely miscalibrated."),
        path=str(IMG / f"cv102-f4-trade.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    sw, pw, sh = res["sweep"], res["power"], res["shares"]

    def floor(panel, m):
        xs = np.geomspace(1e-3, 0.25, 40)
        panel.plot(np.log(xs), sw["constant"] * xs ** sw["slope"], lw=2.8,
                   color=m.series[0])

    def share(panel, m):
        panel.bar([0, 1], [sh[15], 1 - sh[15]], width=0.55,
                  color=[m.grid, m.series[2]])
        panel.set_ylim(0, 1)

    def power(panel, m):
        panel.plot(np.log(SIZES), [pw[(n, 15)] for n in SIZES], lw=2.8,
                   color=m.series[0])
        panel.axhline(0.8, lw=1.6, ls="--", color=m.grid)
        panel.set_ylim(0, 1.05)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=2,
        headline="A perfect model scores 0.035, and yours is near it",
        panels=[(floor, f"{sw['slope']:.3f}", "slope against bins/n"),
                (share, f"{sh[15]:.0%}", "of ECE is the floor"),
                (power, f"{pw[(SUBSET, 15)]:.0%}", "detected at n = 1,000")],
        note=(f"Expected calibration error has a floor that scales as the "
              f"square root of bins over n. On 1,000 predictions, "
              f"{sh[15]:.0%} of this model's measured ECE is what a perfectly "
              f"calibrated model with the same confidences would also score, "
              f"and a real miscalibration is detected in "
              f"{pw[(SUBSET, 15)]:.0%} of subsets."),
        alt=("Three hand-drawn frames: a rising curve; two bars, the larger "
             "grey; and a curve climbing slowly towards a dashed line."),
        path=str(IMG / f"cv102-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["floor"] = s.run("""
        from standarderror.uncertainty import calibration as cb

        # Labels drawn from the model's own probabilities: calibrated by
        # construction, so the true calibration error is exactly zero.
        sweep = cb.bias_sweep()          # 5 sizes x 5 bin counts x 20 draws
        for n in (200, 1000, 20000):
            row = "  ".join(f"{b:2d} bins {sweep['grid'][(n, b)]:.3f}"
                            for b in (5, 15, 50))
            print(f"n = {n:6,d}   {row}")
        print(f"log-log slope against bins/n: {sweep['slope']:.3f}")
        """)

    out["null"] = s.run(f"""
        import numpy as np
        from standarderror.uncertainty import coverage as cv

        pred = cv.predictions(count={BATCHES}, size={BATCH_SIZE}, seed=1)
        p, y = pred["p"], pred["y"]
        i = np.random.default_rng(3).permutation(len(y))[:{SUBSET}]

        for label, rows in (("{SUBSET:,} predictions", i),
                            (f"all {{len(y):,}}", slice(None))):
            t = cb.null_test(p[rows], y[rows], bins=15, draws=400, seed=5)
            print(f"{{label:18s}} ECE {{t['ece']:.4f}}   "
                  f"calibrated floor {{t['null_mean']:.4f}}   "
                  f"p = {{t['p_value']:.3f}}")
        """)

    out["power"] = s.run("""
        # The model IS miscalibrated (p < 0.005 on all of it). How often does
        # a subset of a given size show that?
        for n in (1000, 4000):
            for bins in (15, 50):
                r = cb.detection_rate(p, y, n=n, bins=bins, subsets=100,
                                      seed=11)
                print(f"n = {n:5,d}  {bins:2d} bins   detected in {r:.0%}")
        """)

    out["ranking"] = s.run(f"""
        # Two temperatures of one model: identical accuracy, so the full-data
        # NLL settles which is better calibrated.
        warm = cb.temperature(p, {WARM})
        r = cb.ranking(p, warm, y, n={SUBSET}, bins=(5, 15, 50),
                       subsets=300, seed=2)
        print(f"full data: NLL at T=1 {{r['full']['nll_a']:.4f}}, "
              f"at T={WARM} {{r['full']['nll_b']:.4f}}")
        for metric, share in r["prefers_a"].items():
            print(f"  {{metric:22s}} picks T={WARM} in {{1 - share:.0%}}")
        """)

    out["trade"] = s.run(f"""
        st = cb.stability(p, y, n={SUBSET}, bins=(5, 15, 50), subsets=40,
                          seed=4)
        full = cb._conf_correct(p, y)
        for bins, v in st.items():
            print(f"{{bins:2d}} bins  plug-in {{v['l2']:.3f}}  "
                  f"debiased {{v['debiased']:.3f}} +/- {{v['debiased_sd']:.3f}}  "
                  f"below zero {{v['negative']:.0%}}  "
                  f"all rows {{cb.l2_error(*full, bins=bins):.3f}}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    sw, sm = res["sweep"], res["sweep_mass"]
    g, gm = sw["grid"], sm["grid"]
    full, ex, sh = res["full"], res["example"], res["shares"]
    pw, rk, st, fl = res["power"], res["ranking"], res["stability"], res["full_l2"]
    pred = res["pred"]
    pick = {k: 1 - v for k, v in rk["prefers_a"].items()}
    prof = res["profile"]

    # The spine, asserted rather than trusted.
    assert abs(sw["slope"] - 0.5) < 0.05
    assert all(gm[k] > g[k] for k in g)
    assert res["single"] > 1.3 * g[(200, 15)]
    assert prof[65] < 0.6 * prof[10]
    assert all(full[b]["p_value"] < 0.01 for b in BINS)
    assert ex[15]["p_value"] > 0.05
    assert sh[15] > 0.6 and sh[50] > sh[15] > sh[5]
    assert pw[(SUBSET, 15)] < 0.5 and pw[(4000, 15)] > 0.85
    assert pw[(2000, 5)] > pw[(2000, 15)] > pw[(2000, 50)]
    assert pw[(4000, 15)] > pw[(4000, 50)]
    assert rk["full"]["accuracy_a"] == rk["full"]["accuracy_b"]
    assert rk["full"]["nll_b"] < rk["full"]["nll_a"]
    assert all(b < a for a, b in res["warm_full"].values())
    assert pick["NLL"] > max(v for k, v in pick.items() if k != "NLL")
    assert all(pick[f"debiased L2, {b} bins"] <= pick[f"ECE, {b} bins"]
               for b in BINS)
    assert all(abs(st[b]["debiased"] - fl[b][1]) < 0.006 for b in BINS)
    assert st[50]["l2"] > 1.8 * st[5]["l2"]
    assert st[15]["negative"] > 0.1

    post = Post(
        title=f"{SERIES_TAG} 2: Your Calibration Error Is Mostly Your Bin Count",
        slug="coverage-2-bins",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=2,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Expected calibration error has a floor that a perfectly "
                  "calibrated model also hits. On a thousand predictions "
                  "most of a real model's ECE is that floor - and the fix "
                  "that removes it makes the number noisier, not truer."),
        summary=(
            f"A model calibrated by construction - labels drawn from its own "
            f"probabilities - scores an ECE of {g[(200, 15)]:.3f} on 200 "
            f"predictions, {g[(1000, 15)]:.3f} on 1,000 and "
            f"{g[(20000, 15)]:.3f} on 20,000 at 15 bins, and the fitted slope "
            f"against bins/n is {sw['slope']:.3f}, where a binomial argument "
            f"predicts one half. Equal-mass bins raise it. The floor also "
            f"depends on the confidence profile, so the right comparison is "
            f"the model against itself with resampled labels. On "
            f"{SUBSET:,} predictions from the committed language model, "
            f"{sh[15]:.0%} of the measured ECE is that floor on average, and "
            f"a miscalibration that is real on all {len(pred['y']):,} rows is "
            f"detected in {pw[(SUBSET, 15)]:.0%} of subsets; it takes about "
            f"4,000 points, and fewer bins detect it sooner. Choosing between "
            f"two temperatures of the model, ECE on 1,000 points picks the "
            f"better-calibrated one {pick['ECE, 15 bins']:.0%} of the time at "
            f"15 bins and NLL {pick['NLL']:.0%}. Subtracting each bin's "
            f"binomial variance removes the floor and the bin dependence on "
            f"average - and leaves single estimates so noisy that "
            f"{st[15]['negative']:.0%} of them come out below zero."),
        tags=["machine-learning", "data-science", "statistics",
              "calibration", "uncertainty", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every number is a calibration statistic of "
            "the committed model's predictions on held-out text, or of "
            "synthetic models calibrated by construction; no values from the "
            "text are published.",
            f"The model: an {tiny.load()['parameters']:,}-parameter "
            f"character-level transformer - four blocks, four heads, width "
            f"{tiny.WIDTH}, context {tiny.BLOCK}, validation loss "
            f"{tiny.VAL_LOSS} against a uniform-guess "
            f"{tiny.UNIFORM_LOSS:.3f}. Weights, training script and a verified "
            f"sha256 are committed: `standarderror/llm/tiny.py`, "
            f"`scripts/train_tiny.py`, `data/tiny_gpt/`.",
            "Machinery: `standarderror/uncertainty/calibration.py`, tested in "
            "`tests/test_uncertainty.py`, which pins the floor, the "
            "single-draw correction, the profile dependence and the "
            "bias-variance trade as regressions.",
            "Where this stops: Guo et al., \"On calibration of modern neural "
            "networks\", *ICML* (2017), for ECE as commonly computed; Bröcker "
            "and Smith, \"Increasing the reliability of reliability "
            "diagrams\", *Weather and Forecasting* (2007), for consistency "
            "resampling; Kumar, Liang and Ma, \"Verified uncertainty "
            "calibration\", *NeurIPS* (2019), for the debiased estimator; "
            "Roelofs et al., \"Mitigating bias in calibration error "
            "estimation\", *AISTATS* (2022), for the bias as a function of "
            "bins and n.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python=3.11.15, torch=2.14.0, numpy=2.4.4"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "model": ("the committed checkpoint, hash-verified on load, so "
                      "every rate is measured on the same weights"),
            "determinism": ("every subset, null draw and synthetic model is "
                            "seeded; the floor is averaged over twenty draws "
                            "and the detection rates over a hundred subsets"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A number with a floor",
        """The question people ask of a calibration number is "is it small?". Report an expected calibration error of 0.04 and a reader will hear "well calibrated, slightly off". The question that has to come first is "small compared with what?", because ECE is not a measurement of a model. It is an *estimate*, computed from finitely many predictions, and it has a floor.

The floor is easy to state. Sort the predictions into bins by confidence; in each bin compare the mean confidence with the share that were right; average the gaps. The share that were right is a binomial average, so it is noisy, and the gap is taken in absolute value, so the noise cannot cancel. A model whose probabilities are exactly right still reports a positive number.

How large that number is decides whether anything said about calibration on an ordinary benchmark means anything. This episode measures it, then measures how much of a real model's ECE it accounts for, and ends with the standard fix and what it actually buys.""")

    post.add(
        "Calibrated by construction, and still not zero",
        f"""The cleanest possible test: build a model whose calibration error is zero by definition. Draw a probability vector over ten classes, then draw the label *from that vector*. The reported confidence is then the true probability of being right, every time.

{snip['floor'].markdown()}

On 1,000 predictions with the default 15 bins, a perfect model scores {g[(1000, 15)]:.3f}. On 200 it scores {g[(200, 15)]:.3f}. The fitted slope against bins/n on a log-log scale is {sw['slope']:.3f}; the binomial argument predicts exactly one half, so the floor is not mysterious, it is the square root of bins over n with a constant in front.

Two things worth knowing from the same table. Equal-mass bins, the usual first suggestion for fixing ECE, sit *above* equal-width bins at every size here — {gm[(1000, 15)]:.3f} against {g[(1000, 15)]:.3f} at 1,000 points. And the syllabus for this series originally quoted {res['single']:.3f} for 200 predictions. That was a single draw; the average over twenty is {g[(200, 15)]:.3f}. The estimate of the estimator's bias is itself noisy enough to mislead a table, which is a fair preview of the rest of the episode.""",
        figures=[figs["f0"]])

    post.add(
        "There is no table of floors",
        f"""The obvious fix is to look up the floor for your n and bin count and subtract it. It does not work, because the floor depends on something else as well: the shape of the confidence distribution. The same 1,000 predictions and 15 bins give a floor of {prof[10]:.3f} for a ten-class calibrated model and {prof[65]:.3f} for a 65-class one — less than half, from nothing but where the confidences happen to sit.

So the comparison a measured ECE needs is not against a published constant. It is against *this model with this confidence profile*, made calibrated. That is easy to construct: keep the model's probabilities, throw away the labels, and draw new labels from the probabilities. Under those labels the model is perfect by construction and has exactly its own confidences. Do it a few hundred times and you have the distribution of ECE a calibrated version of your model would report. Weather forecasters have done this since at least 2007, under the name consistency resampling.""")

    post.add(
        "The committed model, against itself",
        f"""Same {tiny.load()['parameters']:,}-parameter character model as episode 1, same held-out predictions.

{snip['null'].markdown()}

On all {len(pred['y']):,} predictions the model is miscalibrated, and clearly: ECE {full[15]['ece']:.4f} against a calibrated floor of {full[15]['null_mean']:.4f}, and no resampled perfect model came close. That is the real signal and it is worth keeping in mind for the next paragraph.

On a random 1,000 of those predictions, the same model reports {ex[15]['ece']:.4f} — and a calibrated model with the same confidences averages {ex[15]['null_mean']:.4f}, exceeding the measured value {ex[15]['p_value']:.0%} of the time. On that subset there is no evidence of miscalibration whatsoever. At 50 bins the measured ECE, {ex[50]['ece']:.4f}, is *below* the calibrated floor of {ex[50]['null_mean']:.4f}.

One subset is an anecdote, so over thirty: on 1,000 predictions, the floor accounts for {sh[5]:.0%} of the measured ECE at 5 bins, {sh[15]:.0%} at 15 and {sh[50]:.0%} at 50. Most of the number is the estimator.""",
        figures=[figs["f1"]])

    post.add(
        "How many predictions it takes",
        f"""If the miscalibration is real but the floor hides it at 1,000 points, the useful question is how many points it takes to see it. That is a power calculation, and it can be done directly: draw subsets of each size, run the null test on each, and count how often it rejects.

{snip['power'].markdown()}

At 1,000 predictions a miscalibration that is really there is detected in {pw[(SUBSET, 15)]:.0%} of subsets at 15 bins. It reaches the conventional 80% somewhere between 2,000 and 4,000 points — {pw[(2000, 15)]:.0%} at 2,000, {pw[(4000, 15)]:.0%} at 4,000.

And bins cost power. At 1,000 points the gap between 15 and 50 bins is inside the noise of a hundred subsets — {pw[(SUBSET, 15)]:.0%} against {pw[(SUBSET, 50)]:.0%}, the wrong way round — but from 2,000 up it is not: {pw[(2000, 5)]:.0%} with 5 bins, {pw[(2000, 15)]:.0%} with 15, {pw[(2000, 50)]:.0%} with 50. More bins sound like more resolution. What they actually buy is more bins with a handful of points each, each contributing its own binomial noise to a sum that cannot cancel.""",
        figures=[figs["f2"]])

    post.add(
        "The comparison everyone makes",
        f"""Most ECE numbers are not used to ask "is this calibrated?". They are used to ask "is this one better calibrated than that one?". That question has an advantage: two models scored on the same predictions at the same bin count share most of their floor, so it partly cancels. Whether it cancels enough is measurable.

The cleanest pair is one model at two temperatures. Dividing every logit by the same positive number cannot change any argmax, so accuracy is identical to the last digit — here {rk['full']['accuracy_a']:.4f} both times — and the full-data NLL difference is then *purely* a calibration difference. At T = {WARM} the NLL is {rk['full']['nll_b']:.4f} against {rk['full']['nll_a']:.4f} at T = 1, and the full-data ECE is lower at every bin count ({res['warm_full'][15][1]:.4f} against {res['warm_full'][15][0]:.4f} at 15 bins): T = {WARM} is the better-calibrated model, and that is settled.

{snip['ranking'].markdown()}

On 1,000 points, 15-bin ECE picks the better model {pick['ECE, 15 bins']:.0%} of the time; at 50 bins it is {pick['ECE, 50 bins']:.0%}, barely better than a coin. NLL picks it {pick['NLL']:.0%} of the time. Brier, the other proper score, manages {pick['Brier']:.0%} — no better than five-bin ECE — so "use a proper score" is not quite the lesson; NLL is, on this model, for this comparison.

The row worth staring at is the debiased one, which picks the right model *less* often than the plug-in at every bin count. That is the next section.""",
        figures=[figs["f3"]])

    post.add(
        "The fix, and what it buys",
        f"""The standard correction is old and simple. Each bin's observed accuracy carries binomial variance of about a(1 − a)/(k − 1), and squaring the gap turns that variance into a positive bias. So work with the squared gap and subtract the variance, bin by bin. What is left is an unbiased estimate of the true squared calibration error.

{snip['trade'].markdown()}

It works exactly as advertised on average. The plug-in moves from {st[5]['l2']:.3f} to {st[50]['l2']:.3f} as the bin count goes from 5 to 50, which is the floor again. The debiased estimate, aggregated over forty subsets, is {st[5]['debiased']:.3f}, {st[15]['debiased']:.3f} and {st[50]['debiased']:.3f} — within a few thousandths of the full-data values ({fl[5][1]:.3f}, {fl[15][1]:.3f}, {fl[50][1]:.3f}) at every bin count, where the plug-in was off by a factor of three at 50 bins.

And on any *one* set of 1,000 predictions it is useless in a different way. Its spread from subset to subset is {st[15]['debiased_sd']:.3f} at 15 bins, about the size of the quantity it estimates, and {st[15]['negative']:.0%} of single estimates come out **below zero**, which is the estimator's way of saying the true value is smaller than its own noise. On a model that is, in fact, miscalibrated.

One detail cost me a wrong figure on the way. Averaging the per-subset *roots* of the debiased estimate gives {st[15]['debiased_mean_of_roots']:.3f} at 15 bins and appears to shrink as the bin count rises. That is Jensen's inequality — a square root is concave, so the mean of roots sits below the root of the mean, and more so as the variance grows. The unbiased quantity is the squared one, and it has to be averaged as such.

So the fix trades a wrong number for a noisy one. That is why it ranks models worse than the plug-in: in a comparison on shared data most of the plug-in's bias cancels anyway, while the variance the correction adds does not.""",
        figures=[figs["f4"]])

    post.add(
        "Where this breaks",
        """**One model, one confidence profile.** The floor shares and the detection rates are properties of this model's predictions. The scaling law is general; the constants are not, which is the whole argument against tables of floors — and an argument against reading these numbers as anyone else's.

**Top-label calibration only.** Everything here is about the confidence of the predicted class. A model can be calibrated on its top label and badly miscalibrated on the rest of the distribution, and none of these numbers would show it. NLL, which does see the rest of the distribution, is part of why it ranks better.

**NLL is a clean calibration comparison only when accuracy is fixed.** Two temperatures of one model are the special case where it is. Between two different models NLL mixes calibration with sharpness, and a better NLL is not a calibration claim on its own.

**The null is the model's own probabilities.** Consistency resampling asks "could a calibrated model with these confidences have produced this ECE?". It does not test whether the confidence *profile* is reasonable, and a model that put every prediction at 0.53 would pass it while being useless.""")

    post.add(
        "What to keep",
        f"""1. ECE has a floor: a perfectly calibrated model scores {g[(1000, 15)]:.3f} on 1,000 predictions at 15 bins, scaling as the square root of bins over n (fitted slope {sw['slope']:.3f}).

2. Equal-mass bins do not remove it; here they raise it.

3. The floor depends on the confidence profile, so compare a measured ECE with the model itself under resampled labels, not with a constant.

4. On 1,000 predictions, {sh[15]:.0%} of this model's ECE is the floor, and a real miscalibration is detected in {pw[(SUBSET, 15)]:.0%} of subsets. It takes about 4,000 points to see it reliably.

5. Fewer bins detect it sooner: at 2,000 points, {pw[(2000, 5)]:.0%} with 5 bins against {pw[(2000, 50)]:.0%} with 50.

6. Choosing the better-calibrated of two models on 1,000 points, 15-bin ECE is right {pick['ECE, 15 bins']:.0%} of the time and NLL {pick['NLL']:.0%}.

7. Debiasing removes the floor on average and makes single estimates as noisy as the quantity — {st[15]['negative']:.0%} of them negative. Unbiased and informative are different properties.""")

    post.add(
        "Exercise",
        """Take the ECE you last reported. Keep the model's probabilities, resample the labels from them two hundred times, and compute ECE each time with the same bins. If your number sits inside that distribution, you have not measured a miscalibration; you have measured your sample size.

Then take the comparison you last made between two models and run it on twenty random halves of the evaluation set. Count how often the ranking flips. If it flips more than a few times, the difference you reported was not a property of the models.

The uncomfortable version: do both with fifty bins, which is what reliability-diagram code often defaults to, and watch how much of your result was the plotting choice.""")

    post.add(
        "Next",
        """Episode 3 is temperature scaling itself: why dividing every logit by the same number cannot change a single prediction — accuracy is bit-identical from T = 0.5 to T = 3 — and why it nevertheless reorders which predictions a model is *most* confident about, which is the ordering abstention and escalation actually use.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
