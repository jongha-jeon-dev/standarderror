"""ML 65: Four Simulated Verdicts, Retried on Real Data.

The sixty-fifth episode of *Machine Learning, Taught Through What Breaks*,
closing Arc XIII. "A finding measured on a simulation tells you what will
happen on real data." Four verdicts that this series first measured on
simulations, repeated on the bundled digits and breast-cancer data:

* Episode 5: resampling for a rare class left the AUC unchanged (within
  0.001) and inflated the predicted probabilities: replicated, at 1.6 to
  1.9 times where the simulation found eleven times, because the class
  here is ten times less rare.
* Episode 35: the best of 64 tuned configurations reported 0.005 more than
  it delivered, and did as well as, to three decimals, the untuned default:
  replicated in direction, a fifth of the size.
* Episode 46: a cost-derived threshold cut the cost of every model, but the
  default threshold of 0.5 was cheaper than flagging everyone, where the
  simulation found it 2.9 to 4 times dearer: half replicated.
* Episode 47: the model with the best AUC had the lowest cost at a 10:1
  cost ratio in 6 of 20 splits: replicated, less sharply.

Run: `standarderror run ml165_replication --publish`
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


def compute() -> dict:
    return {"imb": deploy.replicate_imbalance(),
            "tune": deploy.replicate_tuning(),
            "dec": deploy.replicate_decisions()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    imb, tune, dec = res["imb"], res["tune"], res["dec"]
    m = dec["models"]
    body = [
        ["5", "resampling leaves AUC, inflates probabilities",
         "digits, 'is it an 8?' (10%)",
         f"AUC within {max(abs(v['auc_paired']) for v in imb.values()):.3f}; "
         f"mean p {imb['plain']['mean_p']:.3f} -> "
         f"{min(v['mean_p'] for k, v in imb.items() if k != 'plain'):.3f}-"
         f"{max(v['mean_p'] for k, v in imb.items() if k != 'plain'):.3f}",
         "yes, smaller"],
        ["35", "the best tuned score is optimistic; tuning bought little",
         "breast cancer, 200 tumours",
         f"optimism {tune['optimism']:+.3f}; tuned minus default "
         f"{tune['tuned_minus_default']:+.3f}",
         "yes, smaller"],
        ["46", "threshold 1/(1+r) beats 0.5; 0.5 dearer than flagging all",
         "breast cancer, r = 10",
         f"logistic {m['logistic']['cost_half']:.3f} -> "
         f"{m['logistic']['cost_ratio']:.3f}; flag all {dec['flag_all']:.3f}",
         "half"],
        ["47", "best AUC is not the cheapest model at r = 10",
         "breast cancer, six models",
         f"same model in {dec['same_winner']} of {dec['reps']} splits",
         "yes, weaker"],
    ]
    out["f0"] = charts.table_image(
        body,
        header=["episode", "simulated verdict", "real data", "measured",
                "replicated"],
        title="Four simulated verdicts on real data",
        subtitle=("Each verdict re-measured on scikit-learn's bundled data "
                  "with the episode's own models and settings, over 20 "
                  "random splits."),
        source=("UCI digits and Breast Cancer Wisconsin via scikit-learn; "
                "standarderror/ml/deploy.py."),
        alt=("A four-row table. Three verdicts replicate in direction with "
             "smaller effects; one replicates only half."),
        caption="'Smaller': the same direction, a smaller effect than the simulation's.",
        align="llllc",
        path=str(IMG / f"ml165-f0-table.{EXT}"))[0]

    names = list(m)

    def costs(ax, mm):
        x = np.arange(len(names))
        ax.bar(x - 0.27, [m[k]["cost_half"] for k in names], 0.26,
               color=mm.series[1], label="threshold 0.5")
        ax.bar(x, [m[k]["cost_ratio"] for k in names], 0.26,
               color=mm.series[0], label="threshold 1/(1 + r)")
        ax.bar(x + 0.27, [m[k]["cost_best"] for k in names], 0.26,
               color=mm.series[2], label="best threshold in hindsight")
        ax.axhline(dec["flag_all"], color=mm.ink, ls=":", lw=1)
        ax.text(len(names) - 0.6, dec["flag_all"] + 0.01, "flag everyone",
                fontsize=8, ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace(" boosted", "\nboosted")
                            for k in names], fontsize=8)
        ax.legend(fontsize=8, loc="upper left", frameon=False)

    out["f1"] = charts.diagram(
        costs,
        title="Breast cancer, a missed malignancy costing ten false alarms",
        subtitle=("Mean cost per tumour on 269 held-out tumours, six models "
                  "trained on 300, 20 random splits."),
        xlabel="", ylabel="cost per tumour",
        figsize=(7.8, 4.4),
        source="Breast Cancer Wisconsin via scikit-learn; standarderror/ml/deploy.py.",
        alt=("Grouped bars for six models. The cost-derived threshold lowers "
             "every model's cost; most models at 0.5 are below the "
             "flag-everyone line."),
        caption=(f"Logistic regression: {m['logistic']['cost_half']:.3f} at "
                 f"0.5, **{m['logistic']['cost_ratio']:.3f}** at 1/11."),
        path=str(IMG / f"ml165-f1-costs.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    imb, tune, dec = res["imb"], res["tune"], res["dec"]

    def p1(panel, m):
        panel.bar([0, 1, 2], [imb["plain"]["mean_p"], imb["oversample"]["mean_p"],
                              imb["plain"]["rate"]],
                  color=[m.series[0], m.series[1], m.series[2]])

    def p2(panel, m):
        panel.bar([0, 1], [tune["tuned"], tune["default"]],
                  color=[m.series[0], m.series[1]])
        panel.set_ylim(0.95, 0.98)

    def p3(panel, m):
        panel.bar([0, 1], [dec["same_winner"], dec["reps"] - dec["same_winner"]],
                  color=[m.series[2], m.series[3]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=65,
        headline="Simulated, then real",
        panels=[(p1, "3 of 4", "verdicts replicated"),
                (p2, f"{tune['tuned_minus_default']:+.3f}",
                 "tuned minus default"),
                (p3, f"{dec['same_winner']} of {dec['reps']}",
                 "best AUC = cheapest")],
        note=("The mechanisms carried over; the sizes did not. A simulation "
              "says what can happen and why, and real data say how much."),
        alt=("Three hand-drawn frames: three bars; two equal bars; a short "
             "bar and a tall one."),
        path=str(IMG / f"ml165-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["imb"] = s.run("""
        from standarderror.ml import deploy

        r = deploy.replicate_imbalance(reps=5)
        for m, v in r.items():
            print(f"{m:13s} AUC {v['auc']:.4f}  mean predicted {v['mean_p']:.3f}"
                  f"  (actual rate {v['rate']:.3f})  log loss {v['log_loss']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    imb, tune, dec = res["imb"], res["tune"], res["dec"]
    m = dec["models"]
    others = [v for k, v in imb.items() if k != "plain"]
    infl = [v["mean_p"] / imb["plain"]["mean_p"] for v in others]
    lls = [v["log_loss"] / imb["plain"]["log_loss"] - 1 for v in others]
    half_vs_all = [m[k]["cost_half"] / dec["flag_all"] for k in m]
    auc_top = max(dec["auc_winners"], key=dec["auc_winners"].get)
    cost_top = max(dec["cost_winners"], key=dec["cost_winners"].get)

    # The spine, asserted rather than trusted.
    assert all(abs(v["auc_paired"]) < 0.001 for v in imb.values())
    assert abs(imb["plain"]["mean_p"] - imb["plain"]["rate"]) < 0.005
    assert 1.5 < min(infl) and max(infl) < 2.0
    assert 0.25 < min(lls) and max(lls) < 0.6
    assert 0 < tune["optimism"] < 0.01
    assert abs(tune["tuned_minus_default"]) < 0.0005
    assert all(m[k]["cost_ratio"] < m[k]["cost_half"] for k in m)
    assert sum(h < 1 for h in half_vs_all) == len(m) - 1
    assert [k for k in m if m[k]["cost_half"] >= dec["flag_all"]] == ["15-NN"]
    assert m["recalibrated boosted trees"]["cost_ratio"] - \
        m["recalibrated boosted trees"]["cost_best"] > 0.03
    assert dec["same_winner"] <= 8
    assert auc_top == "boosted trees" and cost_top == "logistic"
    assert dec["cost_winners"].get("boosted trees", 0) <= 3

    post = Post(
        title=f"{SERIES_TAG} 65: Four Simulated Verdicts, Retried on Real Data",
        slug="ml-65-replication",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=65,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Three of four verdicts first measured on simulations held "
                  "on real data in direction and shrank in size; one held "
                  "only half. Simulations were right about mechanisms and "
                  "unreliable about magnitudes."),
        summary=(
            f"Four verdicts this series measured on simulations, repeated on "
            f"scikit-learn's bundled real data over 20 random splits. "
            f"Resampling for a rare class (is a digit an 8?) left the AUC "
            f"within 0.001 and inflated the mean predicted probability "
            f"{min(infl):.1f} to {max(infl):.1f} times, against eleven times "
            f"in the simulation. On 200 breast-cancer tumours the best of 64 "
            f"tuned SVM settings reported {tune['optimism']:.3f} more than it "
            f"delivered and matched the untuned default to three decimals. A threshold "
            f"from a 10:1 cost ratio cut every model's cost, but the default "
            f"threshold of 0.5 was cheaper than flagging everyone for five of "
            f"six models, where the simulation found it 2.9 to 4 times "
            f"dearer. The model with the best AUC was the cheapest at that "
            f"ratio in {dec['same_winner']} of {dec['reps']} splits."),
        tags=["machine-learning", "replication", "reproducibility",
              "evaluation", "simulation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "UCI optical digits (1,797 images, CC BY 4.0) via scikit-learn: "
            "'is it an 8?', about 10% positive; 1,000 images to train, 797 "
            "to test, 20 random splits.",
            "Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, via "
            "scikit-learn: 200 tumours to tune and 369 to test for episode "
            "35; 300 to train and 269 to test, malignant as positive, for "
            "episodes 46 and 47; 20 random splits each.",
            "Machinery: the episodes' own code where it exists (the "
            "resampling methods of `standarderror/ml/imbalance.py`, the "
            "64-setting SVM grid of `around.py`, the six models and cost "
            "functions of `decide.py`); `standarderror/ml/deploy.py`, tested "
            "in `tests/test_ml.py`.",
            "Where this stops: Open Science Collaboration, \"Estimating the "
            "reproducibility of psychological science\", *Science* (2015); "
            "Morris, White and Crowther, \"Using simulation studies to "
            "evaluate statistical methods\", *Statistics in Medicine* "
            "(2019); episodes 5, 35, 46 and 47 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; about two minutes of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Built to isolate, checked on reality",
        """This series has leaned on simulations, deliberately. A simulation knows the truth, so a method's error can be measured exactly, and it can isolate one mechanism at a time: a planted shortcut, a known cost, a rare class with a known rate. Where scikit-learn's bundled data allowed, episodes replicated on real data. Some verdicts never were. The practitioner reading them is entitled to the question this episode asks of four of them: *a finding measured on a simulation tells you what will happen on real data*.

Morris, White and Crowther's guide to simulation studies makes the case for simulations and states their limit: results apply to the data-generating processes simulated. The Open Science Collaboration's attempt to replicate a hundred psychology studies found effects that were, on average, about half the size of the originals. The analogous question here is narrower, because the four verdicts are about methods, not about the world: does the method behave on real data the way it behaved on simulated data?

The four were chosen because each was measured only on a simulation and each can be re-run on the bundled data with the episode's own models and settings: resampling for a rare class (episode 5), the optimism of a tuned score (35), the threshold for unequal costs (46) and the model with the best AUC (47). Each is repeated over 20 random splits.""")

    post.add(
        "Rare classes: replicated, smaller",
        f"""Episode 5 found, at a 2% positive rate in a simulation, that oversampling, SMOTE and class weights left a logistic model's AUC unchanged and multiplied its probabilities about elevenfold. The real version asks whether a handwritten digit is an 8, about {imb['plain']['rate']:.0%} of the digits, with logistic regression on standardised pixels.

{snip['imb'].markdown()}

The AUC moved by at most {max(abs(v['auc_paired']) for v in imb.values()):.4f}, paired across splits. The plain model's mean prediction matched the actual rate ({imb['plain']['mean_p']:.3f} against {imb['plain']['rate']:.3f}); the resampled ones predicted {min(infl):.1f} to {max(infl):.1f} times too much, and their log loss was {min(lls):.0%} to {max(lls):.0%} worse. Both the mechanism and its direction carried over. The size did not, for a reason the simulation predicts: balancing a 10% class moves the training prevalence by a factor of 5, balancing a 2% class by a factor of 25, and the inflation grows with that shift, though not in proportion to it.""")

    post.add(
        "Tuning: replicated, a fifth of the size",
        f"""Episode 35 searched 64 SVM settings by cross-validation on 200 simulated rows and found the best score 0.023 optimistic, with the tuned model no better on fresh data than the default. On 200 breast-cancer tumours with the same grid, the best cross-validated score was {tune['best_cv']:.3f}, the tuned model scored {tune['tuned']:.3f} on the other 369 tumours, and the default {tune['default']:.3f}. The optimism was {tune['optimism']:.3f} (standard error {tune['optimism_se']:.3f}), and tuning bought {tune['tuned_minus_default']:+.3f}.

The direction replicated exactly: the search's best score overstated what it found, and what it found was the default. The size was a fifth, because the breast-cancer data are easy. When every reasonable setting scores near 97%, there is little spread for the search to exploit; the simulation's task was chosen to be hard, with an error floor of about a quarter, which is where the winner's curse bites. Real problems lie at both ends, so the episode's warning holds and its number does not travel.""",
        figures=[figs["f0"]])

    post.add(
        "Thresholds and AUC: half, and weaker",
        f"""Episode 46 set a miss to cost ten false alarms and found that flagging at 0.5 cost 2.9 to 4 times as much as simply flagging everyone, while the threshold 1/(1 + r) came within 0.003 of the best possible for a recalibrated model. On breast cancer, with a missed malignancy costing ten false alarms, the cost-derived threshold lowered every one of the six models' costs, logistic regression's from {m['logistic']['cost_half']:.3f} to {m['logistic']['cost_ratio']:.3f} per tumour. But at 0.5, five of six models were already cheaper than flagging everyone ({dec['flag_all']:.3f}); only 15-nearest-neighbours was not. And the recalibrated model's cost at 1/(1 + r) was {m['recalibrated boosted trees']['cost_ratio']:.3f} against {m['recalibrated boosted trees']['cost_best']:.3f} at the best threshold in hindsight, a gap of {m['recalibrated boosted trees']['cost_ratio'] - m['recalibrated boosted trees']['cost_best']:.3f}, though a threshold chosen in hindsight on 269 tumours flatters itself. The advice replicated; the drama did not, because a task where the models are 97% accurate leaves little room for any threshold to be catastrophic.

Episode 47 found, in simulation, that the model with the best AUC was never the cheapest when misses cost ten times more. On breast cancer, the AUC winner was {auc_top} in {dec['auc_winners'][auc_top]} of 20 splits and the cheapest at 1/(1 + r) only {dec['cost_winners'].get('boosted trees', 0)} times; logistic regression, the AUC winner in {dec['auc_winners'].get('logistic', 0)}, was cheapest in {dec['cost_winners'][cost_top]}. The same model won both in {dec['same_winner']} of {dec['reps']} splits. Choosing by AUC would usually have chosen a model that was not the cheapest, as the simulation said, but not always. The mechanism is the one episode 47 described: AUC averages a model's ranking over every possible threshold, while a 10:1 cost ratio cares only about the region where malignant tumours might be cleared, and the model that ranks best overall need not rank best there. On a task this easy the top models' AUCs differ in the third decimal, so which one wins in a given split varies from split to split; across 20 splits the cheapest was still most often logistic regression, which won on AUC only {dec['auc_winners'].get('logistic', 0)} times.""",
        figures=[figs["f1"]])

    post.add(
        "Where this breaks",
        """**Two bundled data sets.** The digits and the breast-cancer data are clean, small and easy; most real problems are harder, which on the evidence here would move the replications towards the simulations' larger effects, not away from them.

**Four verdicts of many.** These were chosen because they could be re-run with the episodes' own code. Verdicts about time, people and selection (episodes 36, 37 and 39) need data with that structure, which the bundled sets lack.

**Twenty splits of a small data set.** The splits share most of their rows, so the twenty results are not independent, and their spread understates how a new data set of the same kind could differ.

**The same author on both sides.** The replications reuse the episodes' choices of models and settings; an independent replication would choose its own.""")

    post.add(
        "What to keep",
        f"""1. Resampling for a rare class left the AUC unchanged on real digits and inflated the probabilities {min(infl):.1f} to {max(infl):.1f} times, against eleven in the simulation at a rarer prevalence.

2. Tuning 64 settings on 200 real tumours was {tune['optimism']:.3f} optimistic and bought nothing over the default; the simulation's optimism was 0.023.

3. A cost-derived threshold cut every model's cost on real data, but 0.5 was not the disaster the simulation found; the best-AUC model was the cheapest in {dec['same_winner']} of {dec['reps']} splits.

4. Trust a simulation for whether and why a method misbehaves. Re-measure how much on data like yours before quoting a number.""")

    post.add(
        "Exercise",
        """Pick one verdict from this series that matters for your work and re-run it on your own data with the episode's code: resampling, tuning optimism, the threshold, a leakage check.

Note whether the direction holds and how the size compares. A direction that fails to replicate is the more important finding, and the one worth writing up.""")

    post.add(
        "Next",
        """Arc XIII ends here. The last arc turns the series' methods on the series itself: episode 66 reruns twelve of its own verdicts that rested on a single random draw, with five new seeds each, to see how many survive.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
