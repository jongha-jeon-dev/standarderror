"""ML 62: A Prediction Interval Is Right on Average and Wrong Where It Matters.

The sixty-second episode of *Machine Learning, Taught Through What Breaks*,
in Arc XIII. "A prediction plus or minus 1.96 residual standard deviations
covers 95% of new outcomes." Measured on a simulated regression whose noise
grows sixteenfold across the input's range, and on the diabetes data:

* Built from training residuals, the band covered 90.8% of new outcomes;
  from held-out residuals, 93.3%; with split conformal prediction, 95.1%.
  In the noisiest tenth of the inputs the three covered 70.5%, 76.3% and
  80.9%, and in the quietest tenth over 99%.
* Conformalised quantile regression covered 95.0% overall and between 93.2%
  and 96.0% in every tenth, with a band 21% narrower than split
  conformal's.
* On the diabetes data, at 90%, split conformal covered 97.2% of the
  patients with the lowest predictions and 86.3% of those with the
  highest. Quantile regression on 250 patients did not fix it: 94.4% and
  85.2%, with a wider band.

Run: `standarderror run ml162_intervals --publish`
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

SHORT = {"boosting, +-1.96 training residual sd": "training residual sd",
         "boosting, +-1.96 held-out residual sd": "held-out residual sd",
         "linear, +-1.96 held-out residual sd": "linear, held-out sd",
         "split conformal on boosting": "split conformal",
         "conformalised quantile regression": "conformalised quantile"}


def compute() -> dict:
    return {"sim": deploy.intervals_simulation(),
            "dia": deploy.intervals_diabetes()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    sim = res["sim"]["methods"]

    def tenths(ax, m):
        x = np.arange(10) + 0.5
        for k, name in enumerate(SHORT):
            if name.startswith("linear"):
                continue
            ax.plot(x, sim[name]["by_tenth"], marker="o", ms=4,
                    color=m.series[k], label=SHORT[name])
        ax.axhline(0.95, color=m.ink, ls=":", lw=1)
        ax.set_ylim(0.65, 1.02)
        ax.set_xticks(range(0, 11, 2))
        ax.legend(fontsize=8, loc="lower left", frameon=False)

    sc = sim["split conformal on boosting"]
    out["f0"] = charts.diagram(
        tenths,
        title="95% on average, 81% where the noise is",
        subtitle=("Share of 20,000 new outcomes inside each 95% band, in each "
                  "tenth of the input's range; the noise's standard "
                  "deviation grows from 0.2 to 3.2 across it. Means over 20 "
                  "data sets."),
        xlabel="input x (noise grows to the right)",
        ylabel="coverage",
        figsize=(7.6, 4.4),
        source="Simulated; standarderror/ml/deploy.py.",
        alt=("Four curves. Three start at 1.0 on the left and fall steadily "
             "to between 0.70 and 0.81 on the right; the conformalised "
             "quantile curve stays flat near 0.95."),
        caption=(f"Split conformal: {sc['coverage']:.1%} overall, "
                 f"**{sc['worst']:.1%}** in the noisiest tenth."),
        path=str(IMG / f"ml162-f0-tenths.{EXT}"))[0]

    body = []
    for name, v in sim.items():
        body.append([SHORT[name], f"{v['coverage']:.3f}", f"{v['worst']:.3f}",
                     f"{v['best']:.3f}", f"{v['width']:.2f}"])
    out["f1"] = charts.table_image(
        body,
        header=["interval", "coverage", "worst tenth", "best tenth",
                "mean width"],
        title="Five 95% intervals on the same regression",
        subtitle=("Boosted trees or a straight line fitted on 2,000 rows; "
                  "1,000 more for the held-out and conformal versions; "
                  "20,000 fresh rows to check. Means over 20 data sets."),
        source="Simulated; standarderror/ml/deploy.py.",
        alt=("A five-row table. Coverage ranges from 0.908 to 0.951; the "
             "worst tenth from 0.705 to 0.932; the conformalised quantile "
             "band is the narrowest of those near 0.95."),
        caption="Width in the outcome's units, averaged over the 20,000 test rows.",
        align="lrrrr",
        path=str(IMG / f"ml162-f1-table.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    sim, dia = res["sim"]["methods"], res["dia"]
    sc = sim["split conformal on boosting"]
    cq = sim["conformalised quantile regression"]

    def p1(panel, m):
        panel.bar([0, 1], [sc["coverage"], sc["worst"]],
                  color=[m.series[0], m.series[1]])
        panel.set_ylim(0.6, 1)

    def p2(panel, m):
        panel.bar([0, 1], [cq["worst"], cq["width"] / sc["width"]],
                  color=[m.series[2], m.series[3]])
        panel.set_ylim(0.6, 1)

    def p3(panel, m):
        d = dia["split conformal"]
        panel.bar([0, 1, 2], [d["low"], d["middle"], d["high"]],
                  color=[m.series[0], m.series[0], m.series[1]])
        panel.set_ylim(0.6, 1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=62,
        headline="Where it misses",
        panels=[(p1, f"{sc['worst']:.0%}", "coverage, noisy tenth"),
                (p2, f"{cq['worst']:.0%}", "worst tenth, quantile"),
                (p3, f"{dia['split conformal']['high']:.0%}",
                 "diabetes, high predicted")],
        note=("A fixed-width band can be right on average by being too wide "
              "where the outcome is predictable and too narrow where it is "
              "not. The misses gather where the stakes often are."),
        alt=("Three hand-drawn frames: a tall bar and a shorter one; two bars "
             "of similar height; three falling bars."),
        path=str(IMG / f"ml162-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["band"] = s.run("""
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        from standarderror.ml import deploy

        rng = np.random.default_rng(0)
        X, y = deploy._hetero(2000, rng)        # noise sd 0.2 + 0.3 x, x in [0, 10]
        Xc, yc = deploy._hetero(1000, rng)      # calibration rows
        Xt, yt = deploy._hetero(20000, rng)     # fresh rows
        model = GradientBoostingRegressor(random_state=0).fit(X, y)
        q = np.quantile(np.abs(yc - model.predict(Xc)), 0.95 * (1 + 1 / 1000))
        inside = np.abs(yt - model.predict(Xt)) <= q   # split conformal
        print(f"overall {inside.mean():.3f}; x < 1: {inside[Xt[:, 0] < 1].mean():.3f}"
              f"; x > 9: {inside[Xt[:, 0] > 9].mean():.3f}")
        """)
    out["dia"] = s.run("""
        d = deploy.intervals_diabetes(reps=100)
        for k in ("split conformal", "CQR"):
            v = d[k]
            print(f"{k:15s} {v['coverage']:.3f} overall; thirds by prediction "
                  f"{v['low']:.3f} {v['middle']:.3f} {v['high']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    sim, dia = res["sim"]["methods"], res["dia"]
    tr = sim["boosting, +-1.96 training residual sd"]
    ho = sim["boosting, +-1.96 held-out residual sd"]
    li = sim["linear, +-1.96 held-out residual sd"]
    sc = sim["split conformal on boosting"]
    cq = sim["conformalised quantile regression"]
    ds, dq = dia["split conformal"], dia["CQR"]
    narrower = 1 - cq["width"] / sc["width"]

    # The spine, asserted rather than trusted.
    assert tr["coverage"] < ho["coverage"] < sc["coverage"]
    assert abs(sc["coverage"] - 0.95) < 0.005 and abs(cq["coverage"] - 0.95) < 0.005
    for v in (tr, ho, li, sc):
        assert v["best"] > 0.99 and v["worst"] < 0.82
        assert v["by_tenth"] == sorted(v["by_tenth"], reverse=True)
    assert 0.92 < cq["worst"] and cq["best"] < 0.97
    assert 0.15 < narrower < 0.3
    assert abs(ds["coverage"] - 0.9) < 0.02 and ds["low"] - ds["high"] > 0.08
    assert dq["low"] - dq["high"] > 0.07 and dq["width"] > ds["width"]
    assert li["worst"] > ho["worst"] and li["width"] > ho["width"]

    post = Post(
        title=(f"{SERIES_TAG} 62: A Prediction Interval Is Right on Average and "
               f"Wrong Where It Matters"),
        slug="ml-62-intervals",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=62,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A band of fixed width around every prediction covered 95% "
                  "of new outcomes on average and 81% where the outcome was "
                  "noisiest. A band that grows with the noise fixed it, given "
                  "enough data to learn the noise."),
        summary=(
            f"A simulated regression whose noise grows sixteenfold across the "
            f"input's range. A boosted model plus or minus 1.96 residual "
            f"standard deviations covered {tr['coverage']:.1%} of new outcomes "
            f"with training residuals and {ho['coverage']:.1%} with held-out "
            f"ones; split conformal prediction covered {sc['coverage']:.1%}, as "
            f"its guarantee promises. In the noisiest tenth of inputs those "
            f"bands covered {tr['worst']:.1%}, {ho['worst']:.1%} and "
            f"{sc['worst']:.1%}. Conformalised quantile regression covered "
            f"{cq['coverage']:.1%} overall and {cq['worst']:.1%} to "
            f"{cq['best']:.1%} in every tenth, with a band {narrower:.0%} "
            f"narrower. On the diabetes data at 90%, split conformal covered "
            f"{ds['low']:.1%} of the patients with the lowest predictions and "
            f"{ds['high']:.1%} with the highest; quantile regression trained on "
            f"250 patients did no better."),
        tags=["machine-learning", "prediction-intervals", "conformal-prediction",
              "uncertainty", "regression", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: x uniform on [0, 10], y = sin x + x/2 + noise with "
            "standard deviation 0.2 + 0.3x; 2,000 training, 1,000 "
            "calibration and 20,000 test rows per data set, 20 data sets "
            "(seeds 0-19).",
            "Diabetes data (Efron et al., 2004) via scikit-learn: 442 "
            "patients, ten baseline variables, disease progression after a "
            "year; 250 to train, 100 to calibrate, 92 to test, 100 random "
            "splits.",
            "Machinery: scikit-learn's GradientBoostingRegressor (squared "
            "and quantile loss) and LinearRegression; "
            "`standarderror/ml/deploy.py`, tested in `tests/test_ml.py`.",
            "Where this stops: Vovk, Gammerman and Shafer, *Algorithmic "
            "Learning in a Random World* (2005); Lei, G'Sell, Rinaldo, "
            "Tibshirani and Wasserman, \"Distribution-free predictive "
            "inference for regression\", *JASA* (2018); Romano, Patterson and "
            "Candès, \"Conformalized quantile regression\", *NeurIPS* (2019); "
            "Barber, Candès, Ramdas and Tibshirani, \"The limits of "
            "distribution-free conditional predictive inference\", "
            "*Information and Inference* (2021); episode 61 of this "
            "series.",
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
        "A band around every prediction",
        f"""Most of this series has been about classifiers. A regression's prediction is a number, and anyone acting on it needs to know how far off it might be: a delivery estimate, a price, a dose. The standard answer is a band of fixed width around every prediction, plus or minus 1.96 times the standard deviation of the residuals, read as the range that will contain the outcome 95% of the time: *a prediction plus or minus 1.96 residual standard deviations covers 95% of new outcomes*.

The test is a regression where the truth is known: y equals sin x plus x/2 plus noise, with x spread evenly from 0 to 10 and the noise's standard deviation rising from 0.2 at the left to 3.2 at the right. Such heteroscedasticity is common, since larger quantities usually vary more: a large claim is less predictable than a small one, a long delivery than a short one. A boosted model fits the curve on 2,000 rows, and each band is checked on 20,000 fresh ones, overall and in each tenth of the input's range. Everything is averaged over 20 data sets.

{snip['band'].markdown()}""")

    post.add(
        "Right on average",
        f"""Built from the training residuals, the band covered {tr['coverage']:.1%} of new outcomes: the model had fitted some of the training noise, so its residuals understated the error on new rows. Built from residuals on 1,000 held-out rows, it covered {ho['coverage']:.1%}: closer, but the residuals' standard deviation is the wrong summary of a distribution whose spread varies, because the 95% point of a mixture of narrow and wide noise is not 1.96 of its standard deviations.

Split conformal prediction, set out by Vovk and colleagues and by Lei and colleagues for regression, makes one change: instead of a standard deviation it takes the 95% quantile of the absolute held-out residuals, with a small correction for the number of rows. That has a guarantee that needs no model to be right: if the new rows are exchangeable with the calibration rows, the band covers at least 95% of them on average. It delivered: {sc['coverage']:.1%}, with a standard deviation of {sc['coverage_sd']:.3f} across the 20 data sets, about what a 1,000-row calibration set allows. On the question the sentence asks, the conformal band is the honest version of the textbook one.""",
        figures=[figs["f1"]])

    post.add(
        "Wrong where it matters",
        f"""The guarantee is about the average over all inputs, and a band of one width cannot be right everywhere when the noise is not. In the quietest tenth of inputs every fixed-width band covered more than 99% of outcomes, wider than it needed to be. In the noisiest tenth the training-residual band covered {tr['worst']:.1%}, the held-out band {ho['worst']:.1%}, a straight line with a held-out band {li['worst']:.1%} and split conformal **{sc['worst']:.1%}**. One outcome in five fell outside a "95%" interval exactly where the outcomes were largest and least predictable, which in many applications is where the stakes are. The straight line, which cannot follow the curve, had larger residuals everywhere and therefore a wider band, {li['width']:.2f} against {ho['width']:.2f}, and covered the noisiest tenth better than boosting with the same recipe ({li['worst']:.1%} against {ho['worst']:.1%}): the worse model looked safer there only because it was less sure of itself everywhere.

The Coverage series on this site opened with the same point: coverage is a promise about averages. Barber, Candès, Ramdas and Tibshirani proved in 2021 that no method can guarantee coverage at every input without assumptions, which is why the repair has to bring one.

Conformalised quantile regression, from Romano, Patterson and Candès in 2019, brings the assumption that the noise's spread can be learned. Two boosted models estimate the 2.5% and 97.5% quantiles of y given x, so the band widens where the data are noisier, and a conformal step on held-out rows corrects the whole band by however much it misses. It covered {cq['coverage']:.1%} overall and between {cq['worst']:.1%} and {cq['best']:.1%} in every tenth, with an average width of {cq['width']:.2f} against split conformal's {sc['width']:.2f}: {narrower:.0%} narrower and right almost everywhere, because the width went where the noise was.""",
        figures=[figs["f0"]])

    post.add(
        "The diabetes data",
        f"""The replication uses the 442 diabetes patients, with disease progression after a year as the outcome, at a 90% level: 250 patients to fit, 100 to calibrate, 92 to check, over 100 random splits.

{snip['dia'].markdown()}

Split conformal around a linear model covered {ds['coverage']:.1%} overall, on target. Split by the predicted value into thirds, it covered {ds['low']:.1%} of the patients predicted to progress least and {ds['high']:.1%} of those predicted to progress most: the same pattern, presumably because progression varies more among the patients with more of it. Conformalised quantile regression, with shallow boosted trees on 250 patients, covered {dq['low']:.1%} and {dq['high']:.1%}, no better, with a band {dq['width'] / ds['width'] - 1:.0%} wider. Learning how the spread varies takes data, and 250 patients with ten variables were not enough to learn it better than a constant. The repair that worked on 2,000 rows did not transfer to 250.""")

    post.add(
        "Where this breaks",
        """**One shape of noise.** The simulation's noise grows smoothly with one input; real heteroscedasticity can depend on several inputs or on unmeasured ones, which no model of the spread can follow.

**Exchangeability.** Every conformal guarantee assumes the new rows are like the calibration rows. Under the drift of episode 61 the guarantee lapses, even on average.

**Thirds of the predictions.** Coverage by predicted value is one way to slice; slices by any variable can show different gaps, and with enough slices some will miss by chance.

**One quantile model.** Other ways to let the band vary, such as dividing residuals by a fitted spread, behave similarly in kind. They all need the spread to be learnable.""")

    post.add(
        "What to keep",
        f"""1. A band of plus or minus 1.96 residual standard deviations covered {tr['coverage']:.1%} with training residuals and {ho['coverage']:.1%} with held-out ones; split conformal delivered its promised {sc['coverage']:.1%}.

2. In the noisiest tenth of inputs every fixed-width band covered {tr['worst']:.0%} to {sc['worst']:.0%}, and in the quietest more than 99%.

3. Conformalised quantile regression covered {cq['worst']:.1%} to {cq['best']:.1%} in every tenth with a {narrower:.0%} narrower band, on 2,000 rows; on 250 diabetes patients it did no better than a fixed band.

4. Check an interval's coverage in slices that matter, not only overall. Use a held-out or conformal calibration, and let the width vary only where there are enough data to learn how.""")

    post.add(
        "Exercise",
        """Take a regression model in use and its prediction intervals. On a held-out set, compute the share of outcomes inside the interval in each fifth of the predicted value, and in each fifth of the one input you most suspect drives the noise.

If coverage falls well below the nominal level in any fifth, report the interval with that caveat or switch to a band whose width can vary. If coverage is above nominal everywhere, the band is wider than it needs to be, which is a cost of its own.""")

    post.add(
        "Next",
        """Coverage that holds on average and fails in a slice has a sharper form when the slices are groups of people. Episode 63 asks what it means for a model to treat two groups the same, and finds that the obvious definitions cannot all hold at once.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
