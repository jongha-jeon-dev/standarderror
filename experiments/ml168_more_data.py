"""ML 68: More Data Cures the Noise, Not the Design.

The sixty-eighth episode of *Machine Learning, Taught Through What
Breaks*, in Arc XIV. "With enough data, these problems go away." Seven of
the series' problems, each measured at 200, 800 and 3,200 rows:

* Four shrank. A test score's standard error halved with each fourfold
  increase; the optimism of a tuned score fell from 3.7 points to nothing
  by 800 rows; the leak from selecting features on all rows fell from 24.8
  to 7.1 points; a target encoding's self-ranking from 43 to 14.
* Two did not. A model trained on approved applicants underestimated the
  refused applicants' default rate by 45 to 48 points at every size, and
  row-level cross-validation with rows from 20 people overstated accuracy
  by 13 to 16 points, more as rows were added.
* One grew: the BIC's count of Gaussian components for a single square
  went from 4 to 8.

Run: `standarderror run ml168_more_data --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import course
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 27)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"pit": course.pitfalls_vs_n()}


def _kind(v):
    a, b = abs(v[0]), abs(v[-1])
    if b < 0.6 * a:
        return "shrinks"
    if b > 1.5 * a:
        return "grows"
    return "does not shrink"


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    p = res["pit"]
    n = p["sizes"]

    def lines(ax, m):
        short = ["standard error (1)", "tuning optimism (35)",
                 "selection leak (3)", "encoding leak (40)",
                 "selective labels (39)", "rows from 20 people (37)",
                 "BIC components (57)"]
        for k, r in enumerate(p["rows"].values()):
            v = np.abs(r["values"])
            ax.plot(n, v / v[0], marker="o", color=m.series[k % 8],
                    label=short[k])
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(n)
        ax.set_xticklabels([f"{x:,}" for x in n])
        ax.axhline(1, color=m.ink, lw=0.8, ls=":")
        ax.set_ylim(-0.1, 2.2)
        ax.legend(fontsize=7.2, loc="upper left", frameon=False)

    out["f0"] = charts.diagram(
        lines,
        title="Seven problems as the data grow, each relative to its size at 200 rows",
        subtitle=("Size of each problem's error at 200, 800 and 3,200 rows, "
                  "divided by its size at 200 rows. Below the dotted line it "
                  "shrank; above, it grew."),
        xlabel="rows (log scale)", ylabel="relative size",
        figsize=(7.8, 4.6),
        source="Simulated; standarderror/ml/course.py.",
        alt=("Seven lines. Four fall towards zero; two stay near or above one; the "
             "BIC's line rises to two."),
        caption="Absolute sizes, in each problem's own unit, are in the table.",
        path=str(IMG / f"ml168-f0-lines.{EXT}"))[0]

    body = []
    for name, r in p["rows"].items():
        v = r["values"]
        body.append([r["episode"], name] + [f"{x:.1f}" if abs(x) < 100 else
                                            f"{x:.0f}" for x in v]
                    + [r["unit"], _kind(v)])
    out["f1"] = charts.table_image(
        body,
        header=["from", "problem"] + [f"{x:,} rows" for x in n]
        + ["unit", "with more data"],
        title="What more data does to each problem",
        subtitle=("Means over 10 simulated data sets per size (the standard "
                  "error is computed exactly; the BIC from one data set per "
                  "size)."),
        source="Simulated; standarderror/ml/course.py.",
        alt=("A seven-row table: four problems shrink, two do not, one "
             "grows."),
        caption=("Negative default-rate values: the model predicted fewer "
                 "defaults among the refused than occurred."),
        align="llrrrll",
        path=str(IMG / f"ml168-f1-table.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    rows = res["pit"]["rows"]
    kinds = [_kind(r["values"]) for r in rows.values()]
    sel = rows["model trained on approved applicants only"]["values"]
    bic = rows["BIC components for one uniform square"]["values"]

    def p1(panel, m):
        panel.bar(range(3), [kinds.count(k) for k in ("shrinks", "does not shrink",
                                                  "grows")],
                  color=[m.series[0], m.series[1], m.series[2]])

    def p2(panel, m):
        panel.bar(range(3), np.abs(sel), color=m.series[1])

    def p3(panel, m):
        panel.bar(range(3), bic, color=m.series[2])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=68,
        headline="What data cure",
        panels=[(p1, f"{kinds.count('shrinks')} of 7", "shrank with more rows"),
                (p2, f"{abs(sel[-1]):.0f} pts", "selection bias at 3,200"),
                (p3, f"{bic[0]:.0f} to {bic[-1]:.0f}", "BIC, one uniform square")],
        note=("Problems of noise shrink as the data grow. Problems of design, "
              "who is in the data and what the model assumes, do not, and "
              "some grow."),
        alt=("Three hand-drawn frames: a tall bar and two short ones; three "
             "equal bars; three rising bars."),
        path=str(IMG / f"ml168-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["two"] = s.run("""
        from standarderror.ml import course

        for n in (200, 3200):
            leak = course._selection_leak(n, reps=5)        # a variance problem
            bias = course._selection_bias(n, reps=5)        # a design problem
            print(f"{n:5d} rows: selection leak {100 * leak:5.1f} points, "
                  f"refused-applicant bias {100 * bias:6.1f} points")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    p = res["pit"]
    R = {k: v["values"] for k, v in p["rows"].items()}
    se_, tun = R["test-score standard error (accuracy 0.8)"], R[
        "tuning optimism, best of 32 k-NN settings"]
    lk, enc = R["feature selection on all rows, pure noise"], R[
        "target encoding on its own rows, 200 noise levels"]
    sel, sub = R["model trained on approved applicants only"], R[
        "row-level CV, rows from 20 people"]
    bic = R["BIC components for one uniform square"]
    kinds = {k: _kind(v) for k, v in R.items()}

    # The spine, asserted rather than trusted.
    assert list(kinds.values()) == (["shrinks"] * 4
                                    + ["does not shrink"] * 2 + ["grows"])
    assert abs(se_[0] / se_[1] - 2) < 1e-9
    assert tun[0] > 2 and abs(tun[1]) < 0.5 and abs(tun[2]) < 0.5
    assert lk == sorted(lk, reverse=True) and 0.2 < lk[2] / lk[0] < 0.35
    assert enc == sorted(enc, reverse=True) and enc[2] / enc[0] < 0.4
    assert all(-50 < v < -40 for v in sel)
    assert sub == sorted(sub) and 12 < sub[0] and sub[-1] < 18
    assert bic == sorted(bic) and bic[-1] >= 2 * bic[0]

    post = Post(
        title=f"{SERIES_TAG} 68: More Data Cures the Noise, Not the Design",
        slug="ml-68-more-data",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=68,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Seven of the series' problems at 200, 800 and 3,200 rows. "
                  "The four that came from noise shrank; the two that came "
                  "from who was in the data did not; one grew."),
        summary=(
            f"Seven problems from earlier episodes, each measured at 200, 800 "
            f"and 3,200 rows. A test score's standard error fell from "
            f"{se_[0]:.1f} to {se_[-1]:.1f} points; a tuned score's optimism "
            f"from {tun[0]:.1f} points to {tun[-1]:.1f}; the leak from "
            f"selecting features on all rows from {lk[0]:.1f} to {lk[-1]:.1f} "
            f"points; a target encoding's ranking of its own labels from "
            f"{enc[0]:.0f} to {enc[-1]:.0f} points above chance. A model "
            f"trained on approved applicants underestimated the refused "
            f"applicants' default rate by {abs(sel[0]):.0f}, {abs(sel[1]):.0f} "
            f"and {abs(sel[2]):.0f} points; row-level cross-validation with "
            f"rows from 20 people overstated accuracy by {sub[0]:.1f}, "
            f"{sub[1]:.1f} and {sub[2]:.1f}. The BIC's count of Gaussians for "
            f"one square went from {bic[0]:.0f} to {bic[-1]:.0f}."),
        tags=["machine-learning", "sample-size", "bias", "variance",
              "statistics", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated, 10 data sets per size (seeds 0-9), compact versions "
            "of each episode's setting: episode 4's Gaussian task for "
            "tuning (32 odd values of k for k-NN, 5-fold CV, 20,000 test "
            "rows); 1,000 noise features with coin-flip labels, 20 selected "
            "on all rows; a 200-level noise category encoded by its "
            "training-row means; approval on x + 1.5u < 0.5 with u unseen "
            "by the model; episode 37's 20 simulated people with "
            "signatures; one uniform square for the BIC (1 to 15 "
            "components, two starts).",
            "Machinery: `standarderror/ml/course.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Meng, \"Statistical paradises and paradoxes "
            "in big data\", *Annals of Applied Statistics* (2018); episodes "
            "1, 3, 35, 37, 39, 40 and 57 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; about a minute and a half of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The reassuring answer",
        f"""Many of the problems in this series were shown on small data: 200 rows, 300 tumours, 1,797 digits. A practitioner with millions of rows is entitled to wonder which of them still apply, and the common reassurance is that most of them fade: *with enough data, these problems go away*. Some do. The question is which, and the answer turns out to follow a simple rule.

Seven problems were chosen to span the series, each rebuilt in a compact version of its episode's setting so that it can be run at 200, 800 and 3,200 rows, ten data sets per size. Each is measured as the size of the error it causes, in its own unit: points of accuracy, of AUC, of a default rate, or a count of components. The question for each is whether a sixteenfold increase in data made it smaller.

{snip['two'].markdown()}""")

    post.add(
        "Four problems of noise",
        f"""A test score's standard error, episode 1's error bar, is the textbook case: at an accuracy of 0.8 it was {se_[0]:.2f} points at 200 rows, {se_[1]:.2f} at 800 and {se_[2]:.2f} at 3,200, halving with each fourfold increase.

The optimism of the best of 32 tuned settings, episode 35's winner's curse, was {tun[0]:.1f} points at 200 rows and {tun[1]:+.1f} and {tun[2]:+.1f} at 800 and 3,200: with more rows each setting's cross-validated score is less noisy, the best of them has less luck to exploit, and the search's reported score converges on what it delivers. Selecting 20 of 1,000 noise features on all rows, episode 3's leak, turned coin flips into {50 + lk[0]:.0f}% accuracy at 200 rows, {50 + lk[1]:.1f}% at 800 and {50 + lk[2]:.1f}% at 3,200: with more rows the noise features' chance correlation with the label is smaller, so the selection finds less to exploit, though it was still {lk[2]:.1f} points of pure fiction at 3,200. A 200-level noise category encoded by its training-row means, episode 40's leak into training, ranked its own labels at {enc[0]:.0f} points of AUC above chance with one row per level on average, {enc[1]:.0f} with four and {enc[2]:.0f} with sixteen.

All four are the same kind of problem: an estimate fitted to a sample's noise, which shrinks as the sample grows. More data is the cure, at a rate that differs by problem: one over the square root of n for the standard error, faster here for the tuning optimism, roughly in proportion to rows per level for the encoding. With a fixed number of features or categories the cure eventually arrives.""",
        figures=[figs["f0"]])

    post.add(
        "Two problems of design, and one that grows",
        f"""A model trained only on applicants a loan officer approved, where the officer used information the model lacks, episode 39's selective labels, underestimated the refused applicants' default rate by {abs(sel[0]):.1f} points at 200 rows, {abs(sel[1]):.1f} at 800 and {abs(sel[2]):.1f} at 3,200. More approved applicants teach the model more precisely about approved applicants; they say nothing about the refused, because the refused are never in the data. Meng's analysis of big-data surveys makes the same point with a memorable result: a large non-random sample can carry no more information than a small random one, because its bias does not shrink with its size.

Row-level cross-validation with every row from the same 20 people, episode 37, overstated accuracy on new people by {sub[0]:.1f} points at 200 rows, {sub[1]:.1f} at 800 and {sub[2]:.1f} at 3,200. More rows from the same people made the model better at recognising those people, which row-level folds reward and new people do not; the gap grew slightly. The fix is more people, not more rows.

And one problem grew. The BIC, choosing a number of Gaussian components for a single uniform square, chose {bic[0]:.0f} at 200 rows, {bic[1]:.0f} at 800 and {bic[2]:.0f} at 3,200, the same rising count that episode 57 found up to 20,000 points, though not with the same numbers, on a different draw and with fewer starts of the fit. A consistent criterion becomes more confident about a misspecified model's answer as the data grow, and the answer is about the model's shape, not the data's clusters.""",
        figures=[figs["f1"]])

    post.add(
        "The rule",
        """Sorted, the seven fall into the old distinction between variance and bias. The four that shrank were variance: errors that come from fitting a sample's accidents, which average away. The two that stayed were bias in the data's design, who is in the sample and how the rows relate to each other, which no amount of the same data can reveal. The one that grew was bias in the model's assumptions, which more data make more confidently.

For a practitioner with a large data set, the series' problems therefore split in two. Error bars, tuning optimism and leaks from fitting on noise can mostly be retired, though not entirely: a leak that is 7 points at 3,200 rows is still a leak. Selection, grouping, drift, misspecified assumptions and the gap between a metric and its use keep their full force, and often hide better, because a large sample produces narrow error bars around the wrong number.

A practical test separates the two kinds before any data are collected: ask whether the problem would disappear if the sample grew to the whole population it was drawn from. A standard error, a tuned score's luck and a leak from fitting on noise would. A sample of approved applicants, rows from twenty people and a misspecified model would not, because every approved applicant there is still contains no refused ones, and every row from those twenty people still describes only them.""")

    post.add(
        "Where this breaks",
        """**Compact versions.** Each problem was rebuilt in a small simulation; the rates at which the real versions shrink depend on the number of features, categories and settings, which grow with data in real projects.

**Fixed complexity.** With 1,000 noise features held fixed, the selection leak shrank. If the number of features grows with the data, as it often does, the leak need not shrink at all.

**Ten data sets per size.** Enough to separate shrinking from staying; not enough to pin the third digit of any value.

**Seven problems of many.** Problems such as drift (episode 61) and feedback loops (episode 50) depend on time, not sample size, and were not on the list.""")

    post.add(
        "What to keep",
        f"""1. Four problems shrank with sixteen times the data: the standard error ({se_[0]:.1f} to {se_[-1]:.1f} points), tuning optimism ({tun[0]:.1f} to {tun[-1]:.1f}), the selection leak ({lk[0]:.1f} to {lk[-1]:.1f}) and the encoding leak ({enc[0]:.0f} to {enc[-1]:.0f}).

2. Two did not: selective labels stayed {abs(sel[-1]):.0f} points wrong and row-level CV with 20 people stayed {sub[-1]:.0f} points optimistic. The BIC's count grew from {bic[0]:.0f} to {bic[-1]:.0f}.

3. Noise averages away; the design of the data and the assumptions of the model do not.

4. With a large data set, spend your attention on who is in the data, how rows depend on each other and what the model assumes, not on error bars.""")

    post.add(
        "Exercise",
        """For a model trained on a large data set, list the problems from this series you would have worried about with a small one. Mark each as noise (an estimate fitted to the sample's accidents) or design (who is in the sample, how rows relate, what the model assumes).

Retire the first kind if your sample is large relative to your features and settings. For each of the second kind, name the data you would need to measure it: more people rather than more rows, outcomes for the refused, a period after the drift.""")

    post.add(
        "Next",
        """The series has measured problems one at a time. Episode 69 runs one project from start to finish, breast-cancer diagnosis with a missed malignancy costing ten false alarms, and applies every check the series recommends, in the order a project meets them.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
