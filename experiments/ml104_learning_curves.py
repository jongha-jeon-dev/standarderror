"""ML 4: A Learning Curve Fitted Early Promises More Than More Data Delivers.

The fourth episode of *Machine Learning, Taught Through What Breaks*. "A
learning curve tells you whether more data will help" is put into practice
by fitting a power law to a pilot study's errors and reading off the error at
the size you could afford. Measured on a simulated task where the training
set grows to 25,600 rows and the Bayes error is known, and on the digits:

* The two-parameter power law ``a n^-b`` is optimistic in every one of 18
  fits, and for the flexible models predicts errors *below the Bayes error*,
  which no classifier can reach.
* Adding a floor ``a n^-b + c`` fixes the model whose curve has already bent
  (logistic regression, within 0.3% from 6,400 rows) and fails for the one
  still improving: the boosted trees' fitted floor is above the error they
  actually reach when fitted to 400 rows, and below the Bayes error when
  fitted to 1,600 or 6,400.
* The best model changes with the size: on digits, logistic regression is
  best at 50 images and worst at 1,200.

Run: `standarderror run ml104_learning_curves --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import curves as cu
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 8)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

UNTIL = (400, 1600, 6400)
TARGET = 25600
D_UNTIL = (150, 300, 600)
D_TARGET = 1200


def compute() -> dict:
    task = cu.task_curve()
    dig = cu.digits_curve()
    return {"task": task, "dig": dig,
            "ext": cu.extrapolate(task, until=UNTIL, target=TARGET),
            "dext": cu.extrapolate(dig, until=D_UNTIL, target=D_TARGET),
            "best": cu.best_at(task), "dbest": cu.best_at(dig)}


def _row(ext, model, until):
    return next(r for r in ext if r["model"] == model and r["until"] == until)


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    task = res["task"]
    ns = np.array(task["sizes"], float)
    names = list(task["error"])
    xs = np.geomspace(ns[0], ns[-1], 120)

    def curves(ax, m):
        for i, k in enumerate(names):
            ax.plot(ns, [task["error"][k][int(n)] for n in ns], marker="o",
                    ms=5, lw=2.2, color=m.series[i], label=k)
        for i, k in ((1, "15-NN"), (2, "boosted trees")):
            f = cu.fit(task, k, until=1600, floor=False)
            ax.plot(xs, [f["predict"](x) for x in xs], lw=1.6, ls="--",
                    color=m.series[i])
        ax.axhline(task["bayes"], lw=1.4, ls=":", color=m.ink)
        ax.annotate(f"Bayes error {task['bayes']:.3f}: nothing goes lower",
                    (ns[0], task["bayes"] - 0.012), fontsize=8.6,
                    color=m.ink_secondary)
        ax.axvspan(ns[0] * 0.9, 1600, color=m.grid, alpha=0.35)
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8.4, loc="upper right")

    out["f0"] = charts.diagram(
        curves,
        title="A power law fitted early runs through the floor",
        subtitle=(f"Test error on {task['test']:,} fixed test points as the "
                  f"training set grows. Dashed: a n^-b fitted to the shaded "
                  f"sizes, up to 1,600 rows, and extended."),
        xlabel="training rows",
        ylabel="test error",
        source="Simulated; standarderror/ml/curves.py.",
        alt=("Three falling learning curves that flatten; two dashed fitted "
             "curves that keep falling and cross a dotted Bayes-error line "
             "the real curves stay above."),
        caption=(f"Fitted on the shaded range, the power law for the boosted "
                 f"trees predicts "
                 f"**{_row(res['ext'], 'boosted trees', 1600)['power']:.3f}** "
                 f"at {TARGET:,} rows, below the Bayes error of "
                 f"{task['bayes']:.3f}. They reach "
                 f"{task['error']['boosted trees'][TARGET]:.3f}."),
        path=str(IMG / f"ml104-f0-curves.{EXT}"))[0]

    ext = res["ext"]

    def floors(ax, m):
        xs = list(range(len(UNTIL)))
        for i, k in enumerate(names):
            ax.plot(xs, [_row(ext, k, u)["floor"] for u in UNTIL],
                    marker="o", ms=7, lw=2.2, color=m.series[i],
                    label=f"{k}: fitted floor")
            ax.plot(xs, [task["error"][k][TARGET]] * len(UNTIL), lw=1.2,
                    ls="--", color=m.series[i])
        ax.axhline(task["bayes"], lw=1.4, ls=":", color=m.ink)
        ax.annotate("Bayes error", (0, task["bayes"] + 0.004),
                    fontsize=8.6, color=m.ink_secondary)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{u:,}" for u in UNTIL])
        ax.set_xlim(-0.3, len(UNTIL) - 0.7)
        ax.legend(frameon=False, fontsize=8.4, loc="lower left")

    b = {u: _row(ext, "boosted trees", u)["floor"] for u in UNTIL}
    out["f1"] = charts.diagram(
        floors,
        title="The fitted floor depends on where you stopped measuring",
        subtitle=(f"c in a n^-b + c, fitted to sizes up to each value. "
                  f"Dashed: the error each model actually reaches at "
                  f"{TARGET:,} rows."),
        xlabel="largest training size in the fit",
        ylabel="error",
        source="Simulated; standarderror/ml/curves.py.",
        alt=("Three lines of fitted floors; the boosted-trees line starts "
             "above its dashed actual error and drops below the dotted "
             "Bayes-error line."),
        caption=(f"For the boosted trees the floor is {b[400]:.3f} from 400 "
                 f"rows - above what they reach - and **{b[1600]:.3f}** "
                 f"from 1,600, below the Bayes error. A floor nobody can "
                 f"reach is not an estimate of anything."),
        path=str(IMG / f"ml104-f1-floors.{EXT}"))[0]

    rows_t = []
    for r in ext:
        rows_t.append([r["model"], f"<= {r['until']:,}",
                       f"{r['actual']:.3f}",
                       f"{r['power']:.3f} ({cu.relative_error(r['power'], r['actual']):+.1%})",
                       f"{r['power_floor']:.3f} ({cu.relative_error(r['power_floor'], r['actual']):+.1%})",
                       f"{r['floor']:.3f}"])
    bold = {(i, c) for i, r in enumerate(rows_t) for c in range(6)
            if r[0] == "boosted trees" and r[1] == "<= 1,600"}
    out["f2"] = charts.table_image(
        rows_t,
        header=["model", "fit on sizes", f"error at {TARGET:,}",
                "a n^-b predicts", "a n^-b + c predicts", "fitted floor c"],
        title="Nine extrapolations, two forms",
        subtitle=(f"Simulated task, Bayes error {task['bayes']:.3f}. Every "
                  f"prediction is for {TARGET:,} rows."),
        source="Simulated; standarderror/ml/curves.py.",
        alt=("A nine-row table: the two-parameter predictions are all too "
             "low; the three-parameter ones are close for logistic "
             "regression and miss in both directions for boosted trees."),
        caption=("Every two-parameter prediction is too low. The floor helps "
                 "the model that has stopped improving and misleads about "
                 "the one that has not."),
        bold_cells=bold, align="llrrrr",
        path=str(IMG / f"ml104-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    task, dig = res["task"], res["dig"]
    ns = task["sizes"]
    bt = _row(res["ext"], "boosted trees", 1600)

    def through(panel, m):
        panel.plot(np.log(ns), [task["error"]["boosted trees"][n] for n in ns],
                   lw=2.8, color=m.series[2])
        f = cu.fit(task, "boosted trees", until=1600, floor=False)
        panel.plot(np.log(ns), [f["predict"](n) for n in ns], lw=2.0,
                   color=m.series[0])
        panel.plot(np.log([ns[0], ns[-1]]), [task["bayes"]] * 2, lw=1.6,
                   color=m.ink)

    def floor(panel, m):
        panel.plot(range(len(UNTIL)),
                   [_row(res["ext"], "boosted trees", u)["floor"]
                    for u in UNTIL], lw=2.8, color=m.series[2])
        panel.plot([0, len(UNTIL) - 1], [task["bayes"]] * 2, lw=1.6,
                   color=m.ink)

    def cross(panel, m):
        ds = dig["sizes"]
        for i, k in enumerate(dig["error"]):
            panel.plot(np.log(ds), np.log([dig["error"][k][n] for n in ds]),
                       lw=2.6, color=m.series[i])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=4,
        headline="An early learning curve promises too much",
        panels=[(through, f"{bt['power']:.3f} < {task['bayes']:.3f}",
                 "predicted under Bayes"),
                (floor, f"{bt['floor']:.2f}", "a floor nobody reaches"),
                (cross, "best -> worst", "logistic, 50 -> 1,200")],
        note=("Fitted to a pilot study, the usual power law promised a lower "
              "error than more data delivered in every one of 18 fits, and "
              "sometimes lower than any classifier can reach. A floor fixes "
              "curves that have bent and misleads about those that have not."),
        alt=("Three hand-drawn frames: a falling curve with a straighter line "
             "crossing below a flat floor; a falling line crossing that "
             "floor; and three curves that cross each other."),
        path=str(IMG / f"ml104-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["curve"] = s.run("""
        from standarderror.ml import curves as cu

        task = cu.GaussianTask()          # quadratic Bayes boundary, d = 10
        curve = cu.task_curve(task)
        print(f"Bayes error {task.bayes:.4f}")
        for n in (200, 1600, 25600):
            row = "  ".join(f"{k} {curve['error'][k][n]:.4f}"
                            for k in curve["error"])
            print(f"n = {n:>6,}  {row}")
        """)
    out["fit"] = s.run("""
        for model in ("15-NN", "boosted trees"):
            f = cu.fit(curve, model, until=1600, floor=False)
            print(f"{model:13s} a n^-b from n <= 1,600 predicts "
                  f"{f['predict'](25600):.4f} at 25,600;  "
                  f"actual {curve['error'][model][25600]:.4f}")
        """)
    out["slope"] = s.run("""
        for model in curve["error"]:
            sl = cu.log_slope(curve, model)
            print(f"{model:13s} local log-log slope  "
                  + "  ".join(f"{x:+.3f}" for x in sl[::3]))
        """)
    out["digits"] = s.run("""
        dig = cu.digits_curve()
        for n in (50, 100, 300, 1200):
            print(f"{n:>5,} images  " + "  ".join(
                f"{k} {dig['error'][k][n]:.3f}" for k in dig["error"]))
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    task, dig, ext, dext = res["task"], res["dig"], res["ext"], res["dext"]
    bayes = task["bayes"]
    bt = {u: _row(ext, "boosted trees", u) for u in UNTIL}
    nn = {u: _row(ext, "15-NN", u) for u in UNTIL}
    lg = {u: _row(ext, "logistic", u) for u in UNTIL}
    below = [r for r in ext if r["power"] < bayes]
    dlog = _row(dext, "logistic", 300)
    dsvm = _row(dext, "RBF SVM", 300)
    dknn = _row(dext, "3-NN", 300)
    de = dig["error"]
    worst1200 = max(de, key=lambda k: de[k][D_TARGET])
    zero_floors = [r for r in dext if r["until"] == 300 and r["floor"] < 1e-6]

    # The spine, asserted rather than trusted.
    assert all(r["power"] < r["actual"] for r in ext + dext)
    assert len(below) >= 4
    assert bt[400]["floor"] > bt[400]["actual"]
    assert bt[1600]["floor"] < bayes and bt[6400]["floor"] < bayes
    assert abs(cu.relative_error(lg[6400]["power_floor"],
                                 lg[6400]["actual"])) < 0.01
    assert res["dbest"][50] == "logistic" and worst1200 == "logistic"
    assert res["best"][TARGET] == "boosted trees"
    assert len(zero_floors) >= 2
    assert bt[1600]["power"] < bayes < nn[1600]["power"]

    post = Post(
        title=(f"{SERIES_TAG} 4: A Learning Curve Fitted Early Promises More "
               f"Than More Data Delivers"),
        slug="ml-4-learning-curves",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=4,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Fit a power law to a pilot study's errors and it tells "
                  "you what more data will buy. Checked against what more "
                  "data did buy, it was optimistic every time, sometimes "
                  "past the Bayes error."),
        summary=(
            f"On a simulated task whose Bayes error is {bayes:.3f}, with "
            f"training sets grown to {TARGET:,} rows, the two-parameter "
            f"power law a n^-b fitted to the smaller sizes predicted too low "
            f"an error in all 9 fits, and in {len(below)} of them predicted "
            f"an error below the Bayes error - "
            f"{bt[1600]['power']:.3f} for boosted trees that reach "
            f"{bt[1600]['actual']:.3f}. Adding a floor fixes logistic "
            f"regression, whose curve has already bent, but the boosted "
            f"trees' fitted floor is {bt[400]['floor']:.3f} from 400 rows "
            f"and {bt[1600]['floor']:.3f} from 1,600: first above the error "
            f"they reach, then below the Bayes error. On the digits all 9 "
            f"fits were optimistic too, and logistic regression, best at 50 "
            f"images, is worst at 1,200."),
        tags=["machine-learning", "data-science", "statistics",
              "learning-curves", "scaling-laws", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: two Gaussian classes in 10 dimensions with different "
            "covariances, so the Bayes boundary is quadratic; the Bayes "
            "error is computed from the true densities on the 20,000-point "
            "test set every model is scored on.",
            "UCI Optical Recognition of Handwritten Digits, as bundled with "
            "scikit-learn (`load_digits`), CC BY 4.0.",
            "Machinery: `standarderror/ml/curves.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Viering and Loog, \"The shape of learning "
            "curves: a review\", *IEEE TPAMI* (2023); Hestness et al., "
            "\"Deep learning scaling is predictable, empirically\" (2017); "
            "Cortes et al., \"Learning curves: asymptotic values and rate "
            "of convergence\", *NeurIPS* (1993), for the a n^-b + c form.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("training sets are drawn from seeds 1000 "
                            "upward, 3 to 20 per size; the test set and the "
                            "task's covariance from fixed seeds; boosted "
                            "trees with early stopping off and a fixed "
                            "random state"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The question before the test score",
        f"""The first three episodes were about reading a score you already have. This one is about the decision made before there is much of one: a pilot study on a few hundred labelled rows, and the question of whether labelling ten or a hundred times as many is worth it.

The standard tool is the learning curve — test error against training-set size — and the standard way to read it forward is a power law. Error falls as a n^-b on a log-log plot as a straight line; fit the line to the pilot sizes, extend it, read off the error at the size you could afford. It is the same move as the scaling laws used to plan large training runs, made at the scale of a spreadsheet.

Whether it works can be checked only where the curve can be followed much further than the fit. So here is a task built for that: two classes in ten dimensions whose covariances differ, so the best possible boundary is quadratic, training sets grown from 50 rows to {TARGET:,}, and a {task['test']:,}-point test set on which the Bayes error — the error of the true decision rule, which nothing can beat — is computed exactly from the densities.

{snip['curve'].markdown()}

Three models: logistic regression, which cannot represent a quadratic boundary and stalls; 15-nearest-neighbours, which can, slowly; and boosted trees, which can, faster. At {TARGET:,} rows the boosted trees reach {task['error']['boosted trees'][TARGET]:.3f}, still {task['error']['boosted trees'][TARGET] - bayes:.3f} above the Bayes error and still improving.""")

    post.add(
        "The power law runs through the floor",
        f"""Pretend the pilot study stopped at 1,600 rows. Fit a n^-b to each model's errors up to there and extend it.

{snip['fit'].markdown()}

The boosted trees' prediction, {bt[1600]['power']:.3f}, is below the Bayes error of {bayes:.3f}: no classifier, with any amount of data, can achieve it. The 15-NN prediction, {nn[1600]['power']:.3f}, is {nn[1600]['power'] - bayes:.3f} above it — a forecast that the slowest of the flexible models will very nearly reach the best possible error, when it is in fact still {nn[1600]['actual'] - bayes:.3f} away. Across all nine fits — three models, pilot studies stopping at 400, 1,600 and 6,400 rows — the two-parameter power law predicted a lower error than the model actually reached at {TARGET:,}, every time, and {len(below)} of the nine predictions are below the Bayes error.

The reason is visible in the local slope of the curve on log-log axes. A power law has a constant slope. These curves do not:

{snip['slope'].markdown()}

Every slope shrinks towards zero as the curve approaches a floor it cannot go through. A straight line fitted to the early, steep part of the curve carries that steepness forward and overshoots.""",
        figures=[figs["f0"]])

    post.add(
        "Adding a floor, and where that fails",
        f"""The textbook fix is a third parameter: a n^-b + c, where c is the error the model would reach with unlimited data. For logistic regression it works. Its curve has already flattened by a few hundred rows, so the fit can see the floor, and from 6,400 rows it predicts {lg[6400]['power_floor']:.4f} against {lg[6400]['actual']:.4f} actual. Its fitted floor, {lg[6400]['floor']:.3f}, is far above the Bayes error, and correctly so: a linear boundary on a quadratic problem has its own limit.

For the boosted trees, which are still improving at every size measured, the floor is not something the fit can see, so it guesses. Fitted to 400 rows the floor is {bt[400]['floor']:.3f}, above the {bt[400]['actual']:.3f} the trees actually reach — the fit concludes that more data will barely help, and predicts {bt[400]['power_floor']:.3f} at {TARGET:,} rows when the trees in fact get to {bt[400]['actual']:.3f}. Fitted to 1,600 or 6,400 rows the floor is {bt[1600]['floor']:.3f} and {bt[6400]['floor']:.3f}, below the Bayes error. The predictions at {TARGET:,} rows are closer — {bt[1600]['power_floor']:.3f} and {bt[6400]['power_floor']:.3f} against {bt[1600]['actual']:.3f} — but the parameter that is supposed to say how far the model can go has no meaning.

So the floor helps exactly when it is least needed. A curve that has visibly bent tells you more data will not help much, with or without a fit. A curve that is still falling is the case where the decision matters, and there the third parameter is set by noise in the last few points.""",
        figures=[figs["f1"], figs["f2"]])

    post.add(
        "On the digits, and the ranking that turns over",
        f"""The digits allow only 1,200 training images after setting aside a fixed test set, so the check is shorter: fit up to 300, predict 1,200.

{snip['digits'].markdown()}

All nine two-parameter fits — three models, three pilot sizes — were optimistic again: from 300 images, {dlog['power']:.3f} predicted against {dlog['actual']:.3f} for logistic regression ({cu.relative_error(dlog['power'], dlog['actual']):+.0%}), {dsvm['power']:.3f} against {dsvm['actual']:.3f} for the SVM, {dknn['power']:.3f} against {dknn['actual']:.3f} for 3-NN. And the three-parameter fit put the floor at zero for {len(zero_floors)} of the three models: on curves this short and steep there is no floor to see.

The table also shows the second way a pilot study misleads. Logistic regression has the lowest error at 50 images — it is the most constrained model, and constraint is what small data rewards — and the highest at 1,200, double the error of the other two. A model chosen on the pilot would be the wrong one. On the simulated task the same happens at smaller scale: logistic regression leads at 50 rows, 15-NN at 100, and the boosted trees from 200 on.""")

    post.add(
        "Where this breaks",
        f"""**These are small models on small data.** The scaling laws used to plan large neural-network runs are fitted over many orders of magnitude, far into the straight part of the curve, and have predicted well there. The failure here is specific to fitting early — which is exactly where a pilot study fits.

**Least squares on the error scale weights the early points.** The large errors at small n dominate an unweighted fit. Fitting on log error, or weighting by each point's variance, changes the numbers; it does not remove the curvature, so it does not change the direction.

**The task was chosen to have a known floor.** Real problems have one too, but nobody computes it, which is the point: the only check on a fitted floor is the Bayes error, and in practice it is unavailable.

**Single training sets at the largest sizes.** The {TARGET:,}-row errors average three training sets, the digits' 1,200 is one. Their noise is small next to the gaps quoted, but not zero.""")

    post.add(
        "What to keep",
        f"""1. A power law a n^-b fitted to a pilot study predicted too low an error at the target size in all 18 fits here, simulated and real.

2. On a task with a known Bayes error of {bayes:.3f}, {len(below)} of 9 such predictions were below it — errors no classifier can reach.

3. Learning curves flatten towards a floor; the local log-log slope shrinks, so a straight line fitted early overshoots.

4. The floored form a n^-b + c works for curves that have already bent and fails for those still falling: the boosted trees' fitted floor went from {bt[400]['floor']:.3f} (above what they reach) to {bt[1600]['floor']:.3f} (below Bayes) with the fit range.

5. The best model at pilot size need not be the best at full size. On digits, logistic regression went from best at 50 images to worst at 1,200.""")

    post.add(
        "Exercise",
        """If you have a learning curve from a pilot study, refit it twice: once on all the sizes you have, once leaving out the largest. If the prediction at your target size moves by more than the improvement you are hoping for, the curve is not telling you what the extra data will buy.

Then plot the local slope between neighbouring sizes. If it is still steepening or constant, you are in the part of the curve a power law describes; if it is shrinking, there is a floor ahead, and a two-parameter fit will overshoot it. And before committing to a model, check the ranking at the largest size you can afford to test, not at the pilot size.""")

    post.add(
        "Next",
        """Episode 5 closes the first arc with the most common fix for an imbalanced dataset: resampling it until the classes are balanced. It will be measured against the alternatives — class weights and simply moving the decision threshold — on what each does to the ranking, to the probabilities and to the decisions.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
