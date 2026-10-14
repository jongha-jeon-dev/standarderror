"""ML 12: Impurity Importance Rewards Features With Many Places to Split.

The twelfth episode of *Machine Learning, Taught Through What Breaks*.
"A random forest's feature importance ranks what it relies on." The default
importance is mean decrease in impurity (MDI), computed on the training data.
Measured on a design where the answer is known (Strobl et al. 2007):

* With five features and no signal at all, MDI hands 42% of the importance
  to the continuous feature and 5% to the binary one. Permutation importance
  on fresh data gives all five about zero.
* Give the binary feature -- and only it -- a real effect, and MDI never
  ranks it first and ranks it last in 19 of 20 datasets. Permutation
  importance ranks it first in 19.
* Larger leaves shrink the bias without removing it.
* On the breast-cancer data with two noise columns added, both methods rank
  the noise last: the bias matters when real signals are weak, which is when
  importance is most often consulted.

Run: `standarderror run ml112_impurity --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import trees as tr
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 12)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"null": tr.importance_null(signal=0.0),
            "sig": tr.importance_null(signal=0.2),
            "leaf": tr.importance_null(signal=0.2, leaf=20),
            "real": tr.importance_real()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    names = res["null"]["names"]
    x = np.arange(len(names))

    def mdi(ax, m):
        w = 0.38
        ax.bar(x - w / 2, res["null"]["mdi"], w, color=m.series[2],
               label="no feature has an effect")
        ax.bar(x + w / 2, res["sig"]["mdi"], w, color=m.series[0],
               label="only the binary feature has an effect")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 0.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    out["f0"] = charts.diagram(
        mdi,
        title="Impurity importance grows with the number of split points",
        subtitle=("Mean decrease in impurity of a 100-tree random forest, "
                  "averaged over 20 simulated datasets of 500 rows. The "
                  "features differ only in how many distinct values they "
                  "take."),
        xlabel="",
        ylabel="share of total impurity importance",
        source="Simulated after Strobl et al. (2007); "
               "standarderror/ml/trees.py.",
        alt=("Two sets of bars rising left to right from the binary feature "
             "to the continuous one, nearly identical whether or not the "
             "binary feature carries signal."),
        caption=(f"With no signal anywhere, the continuous feature gets "
                 f"**{res['null']['mdi'][-1]:.0%}** of the importance and the "
                 f"binary one {res['null']['mdi'][0]:.0%}. Giving the binary "
                 f"feature a real effect moves its share to "
                 f"{res['sig']['mdi'][0]:.0%} — still last."),
        path=str(IMG / f"ml112-f0-mdi.{EXT}"))[0]

    def perm(ax, m):
        w = 0.38
        for off, key, c, lab in ((-w / 2, "null", m.series[2],
                                  "no feature has an effect"),
                                 (w / 2, "sig", m.series[0],
                                  "only the binary feature has an effect")):
            r = res[key]
            ax.bar(x + off, r["perm"], w, yerr=2 * r["perm_se"], color=c,
                   label=lab, capsize=3, error_kw={"lw": 1})
        ax.axhline(0, lw=1, color=m.ink)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    s = res["sig"]
    out["f1"] = charts.diagram(
        perm,
        title="Permutation importance on fresh data finds the one signal",
        subtitle=("The same forests, scored by the drop in accuracy on 2,000 "
                  "new rows when each feature is shuffled. Bars are means "
                  "over 20 datasets, with two standard errors."),
        xlabel="",
        ylabel="drop in test accuracy",
        source="Simulated after Strobl et al. (2007); "
               "standarderror/ml/trees.py.",
        alt=("Bars near zero for every feature with no signal; with signal, "
             "one tall bar for the binary feature and near-zero bars for the "
             "rest."),
        caption=(f"Shuffling the binary feature costs "
                 f"{s['perm'][0]:.3f} of accuracy; every other feature costs "
                 f"under {max(s['perm'][1:]) + 0.0005:.3f}. Permutation ranks "
                 f"it first in **{s['perm_top_binary']:.0%}** of datasets, "
                 f"MDI in {s['mdi_top_binary']:.0%}."),
        path=str(IMG / f"ml112-f1-perm.{EXT}"))[0]

    rows = []
    lf = res["leaf"]
    for i, n in enumerate(names):
        rows.append([n, f"{res['null']['mdi'][i]:.3f}",
                     f"{s['mdi'][i]:.3f}", f"{lf['mdi'][i]:.3f}",
                     f"{s['perm'][i]:+.4f}"])
    rows.append(["binary ranked first", f"{res['null']['mdi_top_binary']:.0%}",
                 f"{s['mdi_top_binary']:.0%}", f"{lf['mdi_top_binary']:.0%}",
                 f"{s['perm_top_binary']:.0%}"])
    rows.append(["binary ranked last", f"{res['null']['mdi_last_binary']:.0%}",
                 f"{s['mdi_last_binary']:.0%}", f"{lf['mdi_last_binary']:.0%}",
                 "—"])
    out["f2"] = charts.table_image(
        rows,
        header=["feature", "MDI, no signal", "MDI, signal",
                "MDI, signal, leaves of 20", "permutation, signal"],
        title="Four importances, one known answer",
        subtitle=("The binary feature is the only one with an effect in the "
                  "last three columns. Means over 20 datasets."),
        source="Simulated after Strobl et al. (2007); "
               "standarderror/ml/trees.py.",
        alt=("A table of importances for five features under four settings, "
             "with two final rows giving how often the binary feature is "
             "ranked first and how often it is ranked last."),
        caption=("Leaves of at least 20 rows cut the continuous feature's "
                 "share and lift the binary one, but MDI still prefers the "
                 "feature with nothing to say."),
        bold_cells={(5, 4)},
        align="lrrrr",
        path=str(IMG / f"ml112-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    s = res["sig"]

    def mdi(panel, m):
        panel.bar(range(5), s["mdi"], color=[m.series[0]] + [m.series[2]] * 4)

    def perm(panel, m):
        panel.bar(range(5), np.maximum(s["perm"], 0),
                  color=[m.series[0]] + [m.series[2]] * 4)

    def real(panel, m):
        panel.bar([0, 1], [res["real"]["mdi_rank"][0],
                           res["real"]["perm_rank"][0]],
                  color=[m.series[1], m.series[1]])
        panel.set_ylim(0, 33)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=12,
        headline="The forest's favourite feature had nothing to say",
        panels=[(mdi, f"{s['mdi_top_binary']:.0%}", "MDI picks the signal"),
                (perm, f"{s['perm_top_binary']:.0%}",
                 "permutation picks it"),
                (real, f"{res['real']['mdi_rank'][0]} of 32",
                 "noise rank, real data")],
        note=("Impurity importance is computed on the training data, where a "
              "feature with many distinct values has many chances to look "
              "useful. On a design where only a binary feature matters, it "
              "never ranks that feature first. On data with strong signals "
              "the bias hides."),
        alt=("Three hand-drawn frames: bars rising to the right; one tall "
             "bar at the left; two tall bars near the top of their frame."),
        path=str(IMG / f"ml112-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["null"] = s.run("""
        from standarderror.ml import trees as tr

        # 20 datasets of 500 rows; y is a coin flip, independent of all five
        r = tr.importance_null(signal=0.0)
        for n, a, b in zip(r["names"], r["mdi"], r["perm"]):
            print(f"{n:11s} MDI {a:.3f}   permutation on new data {b:+.4f}")
        """)
    out["sig"] = s.run("""
        # now P(y = 1) is 0.4 or 0.6 depending on the binary feature
        for leaf in (1, 20):
            r = tr.importance_null(signal=0.2, leaf=leaf)
            print(f"leaves >= {leaf:2d}: MDI " +
                  " ".join(f"{v:.3f}" for v in r["mdi"]) +
                  f"  binary first: MDI {r['mdi_top_binary']:.0%},"
                  f" permutation {r['perm_top_binary']:.0%}")
            print(f"{'':14s}binary last by MDI: {r['mdi_last_binary']:.0%}")
        """)
    out["real"] = s.run("""
        r = tr.importance_real()   # breast cancer + noise columns
        print(f"rank of noise (continuous, binary) among {r['features']}:"
              f" MDI {r['mdi_rank']}, permutation {r['perm_rank']}")
        print(f"features with permutation importance <= 0: "
              f"{r['perm_nonpositive']}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    nu, s, lf, rl = res["null"], res["sig"], res["leaf"], res["real"]

    # The spine, asserted rather than trusted.
    assert all(a < b for a, b in zip(nu["mdi"], nu["mdi"][1:]))
    assert nu["mdi"][-1] > 0.35 and nu["mdi"][0] < 0.07
    assert np.all(np.abs(nu["perm"]) < nu["perm_se"])   # zero within one SE
    assert nu["mdi_ordered"] == 1.0                      # order in every dataset
    assert s["mdi"][0] < min(s["mdi"][1:])
    assert s["mdi_top_binary"] == 0.0 and s["perm_top_binary"] >= 0.9
    k_last = round(s["mdi_last_binary"] * s["reps"])
    assert 0.9 <= s["mdi_last_binary"] < 1.0 and s["reps"] == 20
    assert max(s["mdi_se"]) < 0.005 and max(lf["mdi_se"]) > 0.01
    assert max(rl["mdi_noise"]) < 0.01
    assert lf["mdi"][0] > s["mdi"][0] and lf["mdi"][-1] < s["mdi"][-1]
    assert lf["mdi_top_binary"] < 0.5 and lf["perm_top_binary"] >= 0.9
    assert min(rl["mdi_rank"]) >= 30 and min(rl["perm_rank"]) >= 25
    assert min(rl["perm_rank"]) > rl["features"] - rl["perm_nonpositive"]

    post = Post(
        title=(f"{SERIES_TAG} 12: Impurity Importance Rewards Features With "
               f"Many Places to Split"),
        slug="ml-12-impurity",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=12,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=(f"A forest's default feature importance is computed on the "
                  f"data it was fitted to. On a design where the answer is "
                  f"known, it never ranks the only useful feature first, and "
                  f"ranks it last in {k_last} of {s['reps']} datasets."),
        summary=(
            f"Five features that differ only in how many distinct values "
            f"they take, and a label that is a coin flip: a random forest's "
            f"impurity importance gives the continuous feature "
            f"{nu['mdi'][-1]:.0%} of the total and the binary one "
            f"{nu['mdi'][0]:.0%}. Give the binary feature, and only it, a "
            f"real effect, and impurity importance never ranks it first and "
            f"ranks it last in {k_last} of {s['reps']} datasets; permutation "
            f"importance on fresh data ranks it "
            f"first in {s['perm_top_binary']:.0%}. Leaves of at least 20 "
            f"rows shrink the bias without removing it. On the breast-cancer "
            f"data, where real signals are strong, both methods rank added "
            f"noise columns at the bottom."),
        tags=["machine-learning", "data-science", "statistics",
              "random-forest", "feature-importance", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated after Strobl et al. (2007): a binary feature, "
            "categorical features with 4, 10 and 20 levels coded as "
            "integers, and a standard-normal feature; 500 training rows and "
            "2,000 test rows; 20 datasets per setting.",
            "Breast Cancer Wisconsin (Diagnostic), as bundled with "
            "scikit-learn, CC BY 4.0, with a standard-normal and a "
            "fair-coin column added; one stratified 70/30 split.",
            "Machinery: `standarderror/ml/trees.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's RandomForestClassifier and "
            "permutation_importance.",
            "Where this stops: Strobl, Boulesteix, Zeileis and Hothorn, "
            "\"Bias in random forest variable importance measures\", *BMC "
            "Bioinformatics* (2007); Breiman, \"Random forests\", *Machine "
            "Learning* (2001), for permutation importance; Li et al., \"A "
            "debiased MDI feature importance measure for random forests\", "
            "*NeurIPS* (2019).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("training datasets from seeds 0 to 19, test "
                            "datasets from seeds 10,000 to 10,019; forests "
                            "and permutations seeded"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "What the bar chart is made of",
        f"""Episode 11 showed that a single tree's structure is a property of its sample. The standard response is to average many trees into a forest, which stabilises the predictions and gives up the readable tree. What people read instead is the forest's feature-importance bar chart, and the one most libraries draw by default is *mean decrease in impurity* (MDI): for each feature, the total reduction in Gini impurity from every split on it, across every tree, normalised to sum to one.

Two facts about how it is built matter. It is computed on the training data, the rows the trees were grown on. And a feature gets credit every time it is chosen for a split, so a feature with more possible split points has more chances to be chosen, by luck as much as by merit.

Strobl and colleagues built the cleanest test of this in 2007: five features that differ *only* in how many distinct values they take — two, four, ten, twenty, and continuous — and a label that is a fair coin, independent of all of them. Every feature's true importance is zero.

{snip['null'].markdown()}

MDI climbs with the number of distinct values, from {nu['mdi'][0]:.3f} for the binary feature to **{nu['mdi'][-1]:.3f} for the continuous one**, in perfect order — and not only on average: the order holds in each of the {nu['reps']} datasets separately. Fully grown trees split down to single rows, and the feature with the most thresholds to choose from is the one that most often finds a split that separates a few coin flips from a few others. All of that is memorised noise, and MDI counts it as importance.

Permutation importance asks a different question on different rows: on 2,000 *new* rows, how much accuracy is lost when a feature's column is shuffled? For every feature here the answer is zero to within one standard error of its mean over the 20 datasets.""",
        figures=[figs["f0"]])

    post.add(
        "A real effect, ranked last",
        f"""The null case could be dismissed: no one cares about the ranking of five useless features. So give one of them an effect. With the binary feature on, the probability of the positive class is 0.6; off, 0.4. The other four still carry nothing.

{snip['sig'].markdown()}

With fully grown trees, MDI never ranks the one informative feature first, and ranks it **last in {k_last} of the {s['reps']} datasets**: its share rises only from {nu['mdi'][0]:.3f} to {s['mdi'][0]:.3f}, while the continuous noise feature keeps {s['mdi'][-1]:.3f}. Permutation importance ranks it first in {s['perm_top_binary']:.0%}.

The likely reason is the same as before. A binary feature can be split only once on any path from the root; once it has been used, it has nothing more to offer below. The continuous feature can be split again and again, and in fully grown trees most splits are deep in the tree, among a handful of rows, where they fit noise. MDI adds those up.

Larger leaves remove the deepest, noisiest splits. With at least 20 rows per leaf the binary feature's share rises to {lf['mdi'][0]:.3f} and the continuous feature's falls to {lf['mdi'][-1]:.3f} — but MDI still ranks the binary feature first in only {lf['mdi_top_binary']:.0%} of datasets. The bias is smaller, not gone.

The continuous noise feature here is a stand-in for columns that turn up in real tables all the time: a row identifier, a timestamp, a free-text field hashed to a number, a postcode or a customer number coded as an integer. Each has close to one distinct value per row, so each offers a fully grown tree a split almost anywhere it wants one. None of them should predict anything, and every one of them is a candidate for the top of an impurity-importance chart. When such a column does appear near the top, the chart is not evidence that the model uses it on new data; on new rows, as the permutation results show, it may contribute nothing at all. The check is cheap, and it is the exercise at the end of this post.""",
        figures=[figs["f1"]])

    post.add(
        "When the bias hides",
        f"""A measurement that kills its own headline belongs in the post. On the breast-cancer data with a pure-noise continuous column and a pure-noise binary column added, MDI does not promote the noise:

{snip['real'].markdown()}

Both methods put the two noise columns at the bottom. Their impurity shares are {rl['mdi_noise'][0]:.1%} for the continuous column and {rl['mdi_noise'][1]:.2%} for the binary one: whatever lucky splits they win add up to a fraction of a percent of the total. A plausible reading, not tested here, is that thirty real features with strong, well-separated signals take the splits near the top of every tree and leave the noise only the deep ones.

So the bias is real and conditional. It matters when real effects are weak relative to the noise, when features differ a lot in how many values they take — a binary treatment indicator beside continuous lab values, a postcode with thousands of levels beside a yes/no flag — and when the trees are grown deep. Those are also the settings in which a ranked importance chart is most often used as evidence of what matters, because nothing else stands out.

Permutation importance has its own failure, which the breast-cancer run also shows: {rl['perm_nonpositive']} of the 32 features have a permutation importance of zero or less: the two noise columns, and {rl['perm_nonpositive'] - 2} real measurements. This run does not say why those real measurements score nothing; the usual explanation is that each has a strongly correlated neighbour, so that when one is shuffled the trees that split on the other still work. Episode 16 measures that problem directly.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        f"""**Permutation importance needs held-out rows.** Computed on the training data, permutation importance should inherit much of the same bias, since the forest has memorised the noise feature's values there and shuffling them hurts. That version was not run here; the fresh rows are what make the numbers above honest.

**Categorical coding matters.** Here the categories are coded as integers, which is what scikit-learn's forests see. One-hot coding gives each level its own binary column, which would change which columns MDI favours; whether it removes the preference for many split points was not tested here.

**Corrected MDI exists.** Out-of-bag versions of impurity importance and the debiased MDI of Li et al. (2019) score splits on rows the tree did not train on, which is designed to remove most of this bias. They are not what the default `feature_importances_` attribute reports.

**Twenty datasets per setting.** The counts quoted are per dataset, and they are not all unanimous. With no signal, MDI rose with the number of distinct values in all {nu['reps']} datasets. With signal, it ranked the binary feature last in {k_last} of {s['reps']} and first in none; permutation ranked it first in {s['perm_top_binary']:.0%}. With leaves of 20 the picture is mixed: MDI ranked the binary feature first in {lf['mdi_top_binary']:.0%} of datasets and last in {lf['mdi_last_binary']:.0%}. The mean shares carry standard errors of at most {max(s['mdi_se']):.3f} with fully grown trees and up to {max(lf['mdi_se']):.3f} with leaves of 20.""")

    post.add(
        "What to keep",
        f"""1. Impurity importance is computed on the training data and credits every split. Features with more distinct values get more splits.

2. With no signal anywhere, it gave a continuous feature {nu['mdi'][-1]:.0%} of the importance and a binary one {nu['mdi'][0]:.0%}.

3. With a real effect on the binary feature alone, it never ranked that feature first and ranked it last in {k_last} of {s['reps']} datasets; permutation importance on fresh data ranked it first in {s['perm_top_binary']:.0%}.

4. Larger leaves reduced the bias; they did not remove it.

5. On data with strong signals the bias barely shows. It matters most where importance charts are most tempting: weak effects, mixed feature types, deep trees.""")

    post.add(
        "Exercise",
        """Add two columns to your training data: one of standard-normal noise and one of fair coin flips. Refit your forest and look at where they land in its `feature_importances_`. Every real feature ranked below the noise column is one whose impurity importance you cannot distinguish from nothing.

Then compute permutation importance on a held-out set and compare the two rankings. Where they disagree, find out how many distinct values each disputed feature takes.""")

    post.add(
        "Next",
        """Episode 13 turns to another number forests give away for free: the out-of-bag error, an estimate of test error that needs no held-out data. It is compared with cross-validation and a real held-out set across forest sizes — and on data where the rows are not independent.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
