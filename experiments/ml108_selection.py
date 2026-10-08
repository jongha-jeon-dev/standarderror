"""ML 8: Cross-Validated Lasso Is Tuned to Predict, Not to Select.

The eighth episode of *Machine Learning, Taught Through What Breaks*. "Lasso
selects the relevant features" is why the lasso is used outside prediction.
Measured where the truth is known and on bootstrap resamples of real data:

* Ten features, two with effects, one irrelevant but correlated with a
  relevant one, seven pure noise. At the cross-validation-minimising
  penalty the lasso keeps both relevant features nearly always -- and about
  2.7 of the 7 noise features, and the irrelevant correlated one about a
  third of the time, whatever the correlation.
* The 1-SE rule removes most of the noise (about 0.2 kept) but not the
  confusion between correlated features: at correlation 0.99 the irrelevant
  one is kept 38% of the time and the relevant one drops to 70%.
* On bootstrap resamples of the diabetes data, age is kept in 82% of fits
  with its sign split 37 positive to 45 negative; s3 is kept in 90% with a
  27 / 63 split. "Selected" says little about "in which direction".

Run: `standarderror run ml108_selection --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import linear as ln
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 8)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"min": ln.correlated_pair(rule="min"),
            "1se": ln.correlated_pair(rule="1se"),
            "smin": ln.bootstrap_signs(rule="min"),
            "s1se": ln.bootstrap_signs(rule="1se"),
            "enet": ln.elastic_pair()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    rhos = [r["rho"] for r in res["min"]]
    xs = list(range(len(rhos)))

    def pair(ax, m):
        for rule, ls, tag in (("min", "-", "CV minimum"),
                              ("1se", "--", "1-SE rule")):
            rows = res[rule]
            ax.plot(xs, [r["a"] for r in rows], ls=ls, marker="o", ms=5,
                    lw=2.2, color=m.series[0], label=f"relevant a, {tag}")
            ax.plot(xs, [r["b"] for r in rows], ls=ls, marker="s", ms=5,
                    lw=2.2, color=m.series[1],
                    label=f"irrelevant b, {tag}")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r:g}" for r in rhos])
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8.0, loc="center left", ncol=1)

    m0, s0 = res["min"][0], res["1se"][-1]
    out["f0"] = charts.diagram(
        pair,
        title="A penalty cannot tell a feature from its stand-in",
        subtitle=("Share of 60 simulated datasets in which the lasso keeps "
                  "the relevant feature a and the irrelevant feature b, as "
                  "the correlation between them rises. 100 rows each."),
        xlabel="correlation between a and b",
        ylabel="share of datasets where kept",
        source="Simulated; standarderror/ml/linear.py.",
        alt=("Four lines: the relevant feature near 1 falling at high "
             "correlation; the irrelevant one near a third for the CV "
             "minimum at every correlation, and rising from near zero with "
             "correlation under the 1-SE rule."),
        caption=(f"At the CV minimum the irrelevant b is kept "
                 f"{m0['b']:.0%} of the time even when it is uncorrelated "
                 f"with anything. The 1-SE rule fixes that and not the rest: "
                 f"at correlation {s0['rho']:g} it keeps b **{s0['b']:.0%}** "
                 f"of the time and a only {s0['a']:.0%}."),
        path=str(IMG / f"ml108-f0-pair.{EXT}"))[0]

    def signs(ax, m):
        names = res["smin"]["names"]
        ys = np.arange(len(names))[::-1]
        for off, key, c, tag in ((0.18, "smin", m.series[0], "CV minimum"),
                                 (-0.18, "s1se", m.series[1], "1-SE rule")):
            s = res[key]
            ax.barh(ys + off, s["positive"], height=0.34, color=c,
                    label=f"{tag}: kept positive (right), negative (left)")
            ax.barh(ys + off, -s["negative"], height=0.34, color=c,
                    alpha=0.55)
        ax.set_yticks(ys)
        ax.set_yticklabels(names)
        ax.axvline(0, lw=1, color=m.ink)
        ax.set_xlim(-1.05, 1.05)
        ticks = [-1, -0.5, 0, 0.5, 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{abs(t):g}" for t in ticks])
        ax.legend(frameon=False, fontsize=8.0, loc="lower left")

    sm = res["smin"]
    i_age, i_s3 = sm["names"].index("age"), sm["names"].index("s3")
    out["f1"] = charts.diagram(
        signs,
        title="Kept in most resamples, with a sign nobody can call",
        subtitle=(f"Diabetes data, {sm['boots']} bootstrap resamples. For "
                  f"each feature, the share of fits that keep it with a "
                  f"positive coefficient (right) and a negative one (left)."),
        xlabel="share of bootstrap fits",
        ylabel="",
        figsize=(7.4, 4.8),
        source="Measured; standarderror/ml/linear.py.",
        alt=("Paired horizontal bars for ten features: several with bars on "
             "both sides of zero under the CV minimum, fewer under the 1-SE "
             "rule."),
        caption=(f"At the CV minimum age is kept in "
                 f"{sm['positive'][i_age] + sm['negative'][i_age]:.0%} of "
                 f"resamples, **{sm['positive'][i_age]:.0%} positive and "
                 f"{sm['negative'][i_age]:.0%} negative**. A selection "
                 f"frequency that high would be read as a finding."),
        path=str(IMG / f"ml108-f1-signs.{EXT}"))[0]

    rows_t = []
    for rule, tag in (("min", "CV minimum"), ("1se", "1-SE rule")):
        for r in res[rule]:
            if r["rho"] in (0.0, 0.9, 0.99):
                rows_t.append([tag, f"{r['rho']:g}", f"{r['a']:.2f}",
                               f"{r['b']:.2f}", f"{r['b_only']:.2f}",
                               f"{r['noise_kept']:.2f}"])
    out["f2"] = charts.table_image(
        rows_t,
        header=["penalty", "corr(a, b)", "keeps a", "keeps b",
                "keeps b, not a", "noise features kept (of 7)"],
        title="What the lasso selects when the truth is known",
        subtitle=("Effects of 0.5 on a and on c (always kept), none on b or "
                  "the seven noise features. 60 datasets of 100 rows per "
                  "row of the table."),
        source="Simulated; standarderror/ml/linear.py.",
        alt=("A six-row table: under the CV minimum, about 2.7 noise "
             "features kept at every correlation; under the 1-SE rule, about "
             "0.2, with the stand-in kept more often as correlation rises."),
        caption=("The CV minimum optimises prediction, which tolerates "
                 "small spurious coefficients. The 1-SE rule trades a little "
                 "prediction for far fewer, and still cannot separate a "
                 "feature from a near-copy."),
        align="lrrrrr",
        path=str(IMG / f"ml108-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res, i_age, i_s3)
    return out


def _hero(res: dict, i_age: int, i_s3: int):
    sm = res["smin"]

    def noise(panel, m):
        panel.bar([0, 1], [res["min"][0]["noise_kept"],
                           res["1se"][0]["noise_kept"]],
                  color=[m.series[0], m.series[1]])

    def stand(panel, m):
        panel.plot(range(len(res["1se"])), [r["b"] for r in res["1se"]],
                   lw=2.8, color=m.series[1])
        panel.plot(range(len(res["1se"])), [r["a"] for r in res["1se"]],
                   lw=2.8, color=m.series[0])

    def sign(panel, m):
        panel.barh([1, 0], [sm["positive"][i_age], sm["positive"][i_s3]],
                   color=m.series[0])
        panel.barh([1, 0], [-sm["negative"][i_age], -sm["negative"][i_s3]],
                   color=m.series[2])
        panel.set_xlim(-1, 1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=8,
        headline="Cross-validated lasso predicts well and selects loosely",
        panels=[(noise, f"{res['min'][0]['noise_kept']:.1f} of 7",
                 "noise features kept"),
                (stand, f"{res['1se'][-1]['b']:.0%}", "stand-in kept, 1-SE"),
                (sign, f"{sm['positive'][i_age]:.0%} / "
                       f"{sm['negative'][i_age]:.0%}", "age: + / - in 100")],
        note=("Tuned by cross-validation, the lasso keeps both real effects "
              "and a few pure-noise features as well. A stricter penalty "
              "drops the noise but still cannot separate a feature from a "
              "correlated stand-in, and on real data it keeps features whose "
              "sign flips between resamples."),
        alt=("Three hand-drawn frames: a tall bar beside a short one; two "
             "lines converging; and two pairs of bars on either side of "
             "zero."),
        path=str(IMG / f"ml108-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["min"] = s.run("""
        from standarderror.ml import linear as ln

        # a, c: effect 0.5.  b: no effect, corr(a, b) = rho.  7 pure noise.
        for r in ln.correlated_pair(rule="min"):
            print(f"rho {r['rho']:4.2f}  keeps a {r['a']:.2f}  b {r['b']:.2f}"
                  f"  b-not-a {r['b_only']:.2f}  noise {r['noise_kept']:.2f} of 7")
        """)
    out["1se"] = s.run("""
        for r in ln.correlated_pair(rule="1se"):
            print(f"rho {r['rho']:4.2f}  keeps a {r['a']:.2f}  b {r['b']:.2f}"
                  f"  b-not-a {r['b_only']:.2f}  noise {r['noise_kept']:.2f} of 7")
        """)
    out["boot"] = s.run("""
        # share of 100 bootstrap fits keeping each feature: positive / negative
        for rule in ("min", "1se"):
            b = ln.bootstrap_signs(boots=100, rule=rule)
            cells = [f"{n} {p:.2f}/{q:.2f}" for n, p, q in
                     zip(b["names"], b["positive"], b["negative"])]
            print(f"{rule:>3}: " + "  ".join(cells[:5]))
            print("     " + "  ".join(cells[5:]))
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    mn, se1 = res["min"], res["1se"]
    sm, s1 = res["smin"], res["s1se"]
    names = sm["names"]
    ia, i2, i3 = names.index("age"), names.index("s2"), names.index("s3")
    noise_min = float(np.mean([r["noise_kept"] for r in mn]))
    noise_1se = float(np.mean([r["noise_kept"] for r in se1]))
    b_min = [r["b"] for r in mn]
    hi1 = se1[-1]
    en = res["enet"]

    # The spine, asserted rather than trusted.
    assert noise_min > 2.0 and noise_1se < 0.5
    assert min(b_min) > 0.25 and max(b_min) - min(b_min) < 0.15
    assert se1[0]["b"] < 0.05 and hi1["b"] > 0.3 and hi1["a"] < 0.8
    assert mn[-1]["b_only"] > 0.15
    assert se1[0]["a"] < mn[0]["a"]
    assert en[-1]["b"] > 0.8 and en[-1]["b"] > mn[-1]["b"]
    assert sm["positive"][ia] > 0.25 and sm["negative"][ia] > 0.25
    assert sm["positive"][i3] > 0.2 and sm["negative"][i3] > 0.5
    assert s1["positive"][ia] + s1["negative"][ia] < 0.1
    assert s1["negative"][i3] > 0.9
    assert sm["positive"][i2] > sm["negative"][i2]
    assert s1["negative"][i2] > s1["positive"][i2]

    post = Post(
        title=(f"{SERIES_TAG} 8: Cross-Validated Lasso Is Tuned to Predict, "
               f"Not to Select"),
        slug="ml-8-selection",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=8,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("The lasso is used to find the features that matter. "
                  "Tuned by cross-validation it keeps pure noise, cannot "
                  "tell a feature from a correlated stand-in, and on real "
                  "data keeps features whose sign it cannot decide."),
        summary=(
            f"With two real effects, one irrelevant feature correlated with "
            f"a real one and seven of pure noise, cross-validated lasso "
            f"keeps both real effects and on average {noise_min:.1f} of the "
            f"7 noise features, and the irrelevant correlated one about "
            f"{np.mean(b_min):.0%} of the time at every correlation. The 1-SE "
            f"rule cuts the noise to {noise_1se:.1f} but not the confusion: "
            f"at correlation {hi1['rho']:g} it keeps the stand-in "
            f"{hi1['b']:.0%} of the time and the real feature {hi1['a']:.0%}. "
            f"On 100 bootstrap resamples of the diabetes data, age is kept "
            f"in {sm['positive'][ia] + sm['negative'][ia]:.0%} of fits, "
            f"{sm['positive'][ia]:.0%} with a positive sign and "
            f"{sm['negative'][ia]:.0%} negative."),
        tags=["machine-learning", "data-science", "statistics", "lasso",
              "feature-selection", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: 10 standard-normal features, 100 rows; effects of 0.5 "
            "on features a and c; feature b correlated with a at rho and "
            "without an effect; unit noise; 60 datasets per correlation.",
            "Diabetes data of Efron et al. (2004), as bundled with "
            "scikit-learn, standardised; 100 bootstrap resamples.",
            "Machinery: `standarderror/ml/linear.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Zhao and Yu, \"On model selection consistency "
            "of lasso\", *JMLR* (2006), for the irrepresentable condition; "
            "Meinshausen and Buehlmann, \"Stability selection\", *JRSS B* "
            "(2010); Hastie, Tibshirani and Friedman, *The Elements of "
            "Statistical Learning* (2009), section 7.10, for the one-"
            "standard-error rule.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("simulated datasets from seeds 0 to 59; "
                            "bootstrap resamples from one seeded generator; "
                            "5 seeded CV folds throughout"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Two jobs, one penalty",
        f"""Episode 7 showed that a lasso's selections depend on the units of its features. This one fixes the units and asks the question the lasso is usually fitted to answer outside prediction: are the features it keeps the ones that matter?

The sentence has theory behind it. Under an *irrepresentable condition* — roughly, irrelevant features must not be too well explained by relevant ones — and with the right penalty, the lasso recovers exactly the true set as the data grow (Zhao and Yu, 2006). In practice the penalty is chosen by cross-validation, and cross-validation chooses it to *predict*.

A simulation where the truth is known separates the two jobs. Ten features, 100 rows. Features a and c have effects of 0.5. Feature b has no effect but is correlated with a at a chosen ρ. The other seven are pure noise.

{snip['min'].markdown()}

The relevant features are kept nearly always, until a and b become near-copies. The rest is not what "selects the relevant features" promises. On average the lasso keeps **{noise_min:.1f} of the 7 noise features**, and the irrelevant b about {np.mean(b_min):.0%} of the time — at ρ = 0 as much as at ρ = 0.95, because at the cross-validation minimum b is just one more noise feature that gets a small coefficient. At ρ = 0.99 it keeps b *instead of* a in {mn[-1]['b_only']:.0%} of datasets.

None of this is a malfunction. Prediction loses almost nothing by giving a noise feature a small coefficient, and a slightly smaller penalty fits the real effects a little better, so the CV minimum lands where a few passengers get on.""",
        figures=[figs["f0"]])

    post.add(
        "A stricter penalty, and what it cannot fix",
        f"""The standard remedy for selection is the one-standard-error rule: take the largest penalty whose cross-validated error is within one standard error of the minimum. It deliberately gives up a little prediction for a sparser model.

{snip['1se'].markdown()}

The passengers mostly get off: {noise_1se:.1f} noise features on average, and b is kept only {se1[0]['b']:.0%} of the time when it is uncorrelated. But the stand-in problem gets *worse* as the correlation rises: at ρ = {hi1['rho']:g}, b is kept {hi1['b']:.0%} of the time and a only {hi1['a']:.0%}. No setting of the penalty can tell two near-identical columns apart, because the data barely can. The lasso keeps one of them, more or less at random, and a report of "the selected features" names whichever it happened to be.

The geometry says why. When two columns are nearly identical, the fitted values depend almost only on the *sum* of their coefficients. The L1 penalty charges |β_a| + |β_b|, which for coefficients of the same sign is also just the sum. Every split of the shared effect between a and b fits about equally well and costs exactly the same penalty, so which corner the solver lands on is decided by the small accidents of the sample. Ridge, whose penalty is β_a² + β_b², strictly prefers an even split and keeps both; the elastic net mixes the two penalties precisely to get that grouping back. Neither tells you which of the two is the real one. With ρ = 0.99 and 100 rows, the data do not contain that information, and no penalty can create it.

The 1-SE rule also costs the relevant feature something even when there is no stand-in: at ρ = 0 it keeps a in {se1[0]['a']:.0%} of datasets, against {mn[0]['a']:.0%} at the CV minimum. A larger penalty shrinks every coefficient, and an effect of 0.5 in 100 noisy rows is not always large enough to survive it. Sparser means fewer false positives and more false negatives; there is no setting that gives fewer of both.""",
        figures=[figs["f2"]])

    post.add(
        "On real data: kept, but which way?",
        f"""On real data nobody knows the truth, but stability can be measured: refit on bootstrap resamples and count. The diabetes data, 100 resamples, each feature's share of fits that keep it with a positive and with a negative coefficient:

{snip['boot'].markdown()}

At the cross-validation minimum, **age is kept in {sm['positive'][ia] + sm['negative'][ia]:.0%} of resamples** — a frequency that would normally be read as robust selection — and its coefficient is positive in {sm['positive'][ia]:.0%} and negative in {sm['negative'][ia]:.0%}. s3 is kept {sm['positive'][i3] + sm['negative'][i3]:.0%} of the time, split {sm['positive'][i3]:.0%} to {sm['negative'][i3]:.0%}. Being selected and having a direction are different findings, and the first says little about the second.

Under the 1-SE rule age almost vanishes ({s1['positive'][ia] + s1['negative'][ia]:.0%}), and s3 becomes stably negative ({s1['negative'][i3]:.0%}). But s2 — kept with a positive sign {sm['positive'][i2]:.0%} of the time at the CV minimum — is kept, when the stricter rule keeps it at all, with a *negative* sign ({s1['negative'][i2]:.0%} against {s1['positive'][i2]:.0%}). s1 and s2 are correlated at 0.9; which one carries the shared signal, and with what sign, depends on the penalty.""",
        figures=[figs["f1"]])

    post.add(
        "Where this breaks",
        f"""**Prediction is not the casualty.** Everything here concerns the selected set. The CV-minimum lasso is a perfectly good predictor; it is the reading of its non-zero coefficients as a list of real effects that fails.

**Stability selection is the principled fix, and it changes the question.** Meinshausen and Bühlmann's procedure keeps features selected in a large share of subsamples across many penalties, with an error-control guarantee. It is still blind to which of two near-copies is the real one; the guarantee is about noise features, not stand-ins.

**The elastic net groups, it does not choose.** Adding a ridge component (half and half, tuned by the same cross-validation) keeps correlated features together instead of picking one. On this design it does so by keeping the stand-in *more* often: {en[0]['b']:.0%} of the time at ρ = 0 and {en[-1]['b']:.0%} at ρ = {en[-1]['rho']:g}, with {np.mean([r['noise_kept'] for r in en]):.1f} noise features kept on average. More stable, and no closer to the truth.

**One simulation design.** Ten features, 100 rows, effects of 0.5. With more rows the CV minimum keeps fewer passengers; with more noise features it keeps more. The stand-in problem does not shrink with more rows at a fixed high correlation.

**Bootstrap selection frequencies are themselves estimates.** A hundred resamples resolve a share to about ±5 points; the sign splits quoted are far wider than that.""")

    post.add(
        "What to keep",
        f"""1. Cross-validation tunes the lasso to predict. At its minimum, the lasso here kept {noise_min:.1f} of 7 pure-noise features on average.

2. The 1-SE rule drops most of them ({noise_1se:.1f}) at a small cost in prediction.

3. No penalty separates a feature from a highly correlated stand-in: at ρ = {hi1['rho']:g}, the stand-in was kept {hi1['b']:.0%} of the time and the real feature {hi1['a']:.0%}.

4. On the diabetes data, age was kept in {sm['positive'][ia] + sm['negative'][ia]:.0%} of bootstrap fits with its sign split {sm['positive'][ia]:.0%} / {sm['negative'][ia]:.0%}. Report sign stability, not only selection frequency.

5. Treat a lasso's selected set as a property of the penalty, the units and the sample as much as of the data-generating process.""")

    post.add(
        "Exercise",
        """Refit your lasso on 100 bootstrap resamples and tabulate, for every feature, how often it is kept with each sign. Any feature kept often with both signs is not a finding about its direction. Any pair of features that trade places across resamples — one kept when the other is not — is a single signal the data cannot attribute.

Then refit with the 1-SE rule and compare the two selected sets. The features in both are your most defensible list; the features only in the CV-minimum set are the passengers.""")

    post.add(
        "Next",
        """Episode 9 turns from selection to the most familiar diagram in the course: the U-shaped curve of test error against model complexity, bias falling and variance rising. On simulated data both terms can be computed exactly, for a complexity knob that draws the U and one that does not.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
