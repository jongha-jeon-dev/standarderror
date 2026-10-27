"""ML 61: An Alarm on the Inputs Cannot See the Outcome.

The sixty-first episode of *Machine Learning, Taught Through What Breaks*,
opening Arc XIII. "A drift alarm on a model's inputs tells you when the
model has stopped working." Measured on a simulated logistic world deployed
in batches, and on breast-cancer tumours:

* When the inputs moved by one standard deviation, the alarm fired on every
  batch and accuracy rose, from 0.775 to about 0.80.
* When the relation between inputs and outcome reversed and the inputs did
  not move, the alarm fired about as often as with no drift at all while
  accuracy fell to 0.63 or 0.46. An alarm on the predicted scores was
  silent too, and an estimate of accuracy from the unlabelled batch said
  0.77 throughout.
* With 20 features tested separately at 5% and no drift, at least one alarm
  fired in about 60% of batches. On breast cancer, deploying on the larger
  tumours set off alarms on 19 of 30 features with no fall in accuracy
  that 135 tumours could show.

Run: `standarderror run ml161_drift --publish`
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
    return {"mon": deploy.drift_monitors(), "label": deploy.drift_label_shift(),
            "false": deploy.drift_false_alarms(),
            "bc": deploy.drift_breast_cancer()}


def _row(mon, scenario, batch):
    return next(r for r in mon["rows"] if r["scenario"] == scenario
                and r["batch"] == batch)


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    mon = res["mon"]
    names = list(deploy.SCENARIOS)

    def bars(ax, m):
        x = np.arange(len(names))
        al = [_row(mon, s, 2000)["input_alarm"] for s in names]
        ac = [_row(mon, s, 2000)["accuracy"] for s in names]
        es = [_row(mon, s, 2000)["estimate"] for s in names]
        ax.bar(x - 0.27, al, 0.26, color=m.series[1], label="input alarm rate")
        ax.bar(x, ac, 0.26, color=m.series[0], label="true accuracy")
        ax.bar(x + 0.27, es, 0.26, color=m.series[2],
               label="accuracy estimated without labels")
        ax.axhline(mon["validation_accuracy"], color=m.ink, ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper right", frameon=False)

    rev = _row(mon, "relation reversed", 2000)
    out["f0"] = charts.diagram(
        bars,
        title="The alarm watches the inputs; the damage is in the outcome",
        subtitle=("A logistic model deployed on 50 batches of 2,000 rows per "
                  "scenario. Alarm: any of five per-input Kolmogorov-Smirnov "
                  "tests below 0.05/5. Estimate: validation accuracy "
                  "re-weighted to look like the batch. Dotted: accuracy before "
                  "deployment."),
        xlabel="", ylabel="rate or accuracy",
        figsize=(7.8, 4.5),
        source="Simulated; standarderror/ml/deploy.py.",
        alt=("Grouped bars for five scenarios. Moving the inputs fires the "
             "alarm and leaves accuracy level or higher; reversing the "
             "relation leaves the alarm near zero and drops accuracy, while "
             "the label-free estimate stays near 0.77."),
        caption=(f"Relation reversed: alarm on {rev['input_alarm']:.0%} of "
                 f"batches, accuracy **{rev['accuracy']:.2f}**, estimate "
                 f"without labels {rev['estimate']:.2f}."),
        path=str(IMG / f"ml161-f0-monitors.{EXT}"))[0]

    body = []
    for r in mon["rows"]:
        body.append([r["scenario"], f"{r['batch']:,}", f"{r['input_alarm']:.2f}",
                     f"{r['score_alarm']:.2f}", f"{r['accuracy']:.3f}",
                     f"{r['estimate']:.3f}"])
    out["f1"] = charts.table_image(
        body,
        header=["scenario", "batch", "input alarm", "score alarm",
                "true accuracy", "estimate, no labels"],
        title="Every scenario, two batch sizes",
        subtitle=("Share of 50 batches raising each alarm, mean accuracy, and "
                  "mean label-free estimate. Accuracy before deployment: "
                  f"{mon['validation_accuracy']:.3f}."),
        source="Simulated; standarderror/ml/deploy.py.",
        alt=("A ten-row table. The input alarm is 1.00 when the inputs move a "
             "standard deviation and near zero when the relation reverses."),
        caption="The score alarm tests the model's predicted probabilities the same way.",
        align="lrrrrr",
        path=str(IMG / f"ml161-f1-table.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    mon, fa = res["mon"], res["false"]
    up = _row(mon, "inputs move 1 sd", 2000)
    rev = _row(mon, "relation reversed", 2000)
    f20 = next(r for r in fa if r["features"] == 20)

    def p1(panel, m):
        panel.bar([0, 1], [mon["validation_accuracy"], up["accuracy"]],
                  color=[m.series[0], m.series[1]])
        panel.set_ylim(0.5, 0.85)

    def p2(panel, m):
        panel.bar([0, 1], [rev["input_alarm"], 1 - rev["accuracy"]],
                  color=[m.series[1], m.series[0]])
        panel.text(0, rev["input_alarm"] + 0.02, f"{rev['input_alarm']:.2f}",
                   ha="center", fontsize=8)
        panel.set_ylim(0, 1)

    def p3(panel, m):
        panel.bar([0, 1, 2], [r["uncorrected"] for r in fa],
                  color=[m.series[2]] * 3)
        panel.set_ylim(0, 1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=61,
        headline="Watching the inputs",
        panels=[(p1, f"{up['accuracy']:.2f}", "accuracy after the alarm"),
                (p2, f"{rev['input_alarm']:.0%}", "alarms, relation flipped"),
                (p3, f"{f20['uncorrected']:.0%}", "false alarms, 20 inputs")],
        note=("An alarm on the inputs measures whether the inputs moved. "
              "Whether the model still works is a question about outcomes, "
              "and only labels answer it."),
        alt=("Three hand-drawn frames: a bar and a slightly taller one; an "
             "empty bar beside a tall one; three rising bars."),
        path=str(IMG / f"ml161-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["world"] = s.run("""
        import numpy as np
        from scipy.stats import ks_2samp
        from sklearn.linear_model import LogisticRegression
        from standarderror.ml import deploy

        rng = np.random.default_rng(0)
        X, y = deploy.drift_world(5000, rng)          # the world at training time
        model = LogisticRegression().fit(X, y)
        for kw in ({"shift": 1.0}, {"flip": 1.0}):    # inputs move; relation turns
            Xb, yb = deploy.drift_world(2000, rng, **kw)
            p = min(ks_2samp(X[:, j], Xb[:, j]).pvalue for j in range(5))
            acc = np.mean(model.predict(Xb) == yb)
            print(f"{kw}: smallest input p-value {p:.1e}, accuracy {acc:.3f}")
        """)
    out["many"] = s.run("""
        for r in deploy.drift_false_alarms(features=(5, 20, 100), batches=100):
            print(f"{r['features']:3d} inputs, no drift: any alarm in "
                  f"{r['uncorrected']:.0%} of batches, "
                  f"{r['bonferroni']:.0%} with Bonferroni")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    mon, lab, fa, bc = res["mon"], res["label"], res["false"], res["bc"]
    base = mon["validation_accuracy"]
    up = _row(mon, "inputs move 1 sd", 2000)
    tiny = _row(mon, "inputs move 0.1 sd", 2000)
    tiny_s = _row(mon, "inputs move 0.1 sd", 200)
    half = _row(mon, "relation half-reversed", 2000)
    rev = _row(mon, "relation reversed", 2000)
    none = _row(mon, "no drift", 2000)
    f = {r["features"]: r for r in fa}
    lo = next(r for r in lab if r["prevalence"] == 0.2)
    hi = next(r for r in lab if r["prevalence"] == 0.8)
    b0, b50, b75 = bc

    # The spine, asserted rather than trusted.
    assert up["input_alarm"] == 1 and up["accuracy"] > base + 0.02
    assert abs(up["estimate"] - up["accuracy"]) < 0.01
    assert tiny["input_alarm"] > 0.6 and tiny_s["input_alarm"] < 0.2
    assert abs(tiny["accuracy"] - none["accuracy"]) < 0.005
    for r in (half, rev):
        assert r["input_alarm"] < 0.1 and r["score_alarm"] < 0.1
        assert abs(r["estimate"] - base) < 0.01
    assert half["accuracy"] < base - 0.1 and rev["accuracy"] < 0.5
    assert lo["input_alarm"] == 1 and abs(lo["accuracy"] - base) < 0.03
    assert lo["mean_predicted"] - lo["mean_actual"] > 0.15
    assert hi["mean_actual"] - hi["mean_predicted"] > 0.15
    assert 0.55 < f[20]["uncorrected"] < 0.7 and f[100]["uncorrected"] > 0.95
    assert 0.18 < f[5]["uncorrected"] < 0.28
    assert all(r["bonferroni"] < 0.09 for r in fa)
    assert b0["features_alarmed"] == 0 and b50["features_alarmed"] >= 15
    assert b75["accuracy"] >= b0["accuracy"] - 0.01

    post = Post(
        title=(f"{SERIES_TAG} 61: An Alarm on the Inputs Cannot See the "
               f"Outcome"),
        slug="ml-61-drift",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=61,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Input-drift alarms fired when the inputs moved and the "
                  "model got better, stayed quiet when the world changed "
                  "under the model and it got much worse, and went off "
                  "constantly with nothing wrong at all."),
        summary=(
            f"A logistic model, deployed on batches of a simulated world. When "
            f"one input moved by a standard deviation, an alarm on the inputs "
            f"fired on every batch and accuracy rose from {base:.3f} to "
            f"{up['accuracy']:.3f}. When the input's effect reversed and the "
            f"inputs stayed put, the alarm fired on {rev['input_alarm']:.0%} "
            f"of batches, an alarm on the predicted scores on "
            f"{rev['score_alarm']:.0%}, and accuracy fell to "
            f"{rev['accuracy']:.3f}; an estimate of accuracy from the "
            f"unlabelled batch stayed at {rev['estimate']:.3f}. With 20 inputs "
            f"tested separately and no drift, at least one alarm fired in "
            f"{f[20]['uncorrected']:.0%} of batches. On breast cancer, "
            f"deploying on the larger tumours set off alarms on "
            f"{b50['features_alarmed']} of 30 features while accuracy stayed "
            f"within what 135 tumours can measure."),
        tags=["machine-learning", "monitoring", "data-drift", "mlops",
              "statistics", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: five standard Gaussian inputs, logistic outcome with "
            "coefficients (1.5, -1, 0.8, 0, 0) and intercept -0.3; the model "
            "trained on 5,000 rows; 50 deployment batches of 200 or 2,000 "
            "rows per scenario (seed 0).",
            "Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, via "
            "scikit-learn: a standardised logistic model trained on 300 random "
            "tumours, deployed on the other 269 or on those above the "
            "training median or upper quartile of mean radius.",
            "Machinery: SciPy's two-sample Kolmogorov-Smirnov test; a "
            "logistic domain classifier for the density-ratio estimate; "
            "`standarderror/ml/deploy.py`, tested in `tests/test_ml.py`.",
            "Where this stops: Rabanser, Günnemann and Lipton, \"Failing "
            "loudly: an empirical study of methods for detecting dataset "
            "shift\", *NeurIPS* (2019); Gama, Žliobaitė, Bifet, Pechenizkiy "
            "and Bouchachia, \"A survey on concept drift adaptation\", *ACM "
            "Computing Surveys* (2014); Sugiyama, Krauledat and Müller, "
            "\"Covariate shift adaptation by importance weighted cross "
            "validation\", *JMLR* (2007); Lipton, Wang and Smola, \"Detecting "
            "and correcting for label shift with black box predictors\", "
            "*ICML* (2018); episodes 29 and 39 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": "seeded throughout; about half a minute of CPU",
        },
    )
    post.hero = figs["hero"]

    post.add(
        "What a monitor can see",
        f"""Every arc so far has ended where a model is deployed. This one starts there, with the questions a practitioner meets afterwards: is it still working, how sure are its numbers, is it fair, how often should it be retrained, and do the series' simulated findings hold on real data. The first is the most common: once a model is live, how do you know when it has stopped working?

The usual answer is to watch the inputs. Labels arrive late or never, inputs arrive with every prediction, so a monitoring system compares each new batch of inputs with the training data, feature by feature, and raises an alarm when they differ. The practice rests on a sentence worth testing: *a drift alarm on a model's inputs tells you when the model has stopped working*.

Episode 29 measured what a shift in the inputs costs a model, and found that it costs little when the model is right about how inputs relate to the outcome. The test here is of the alarm. A simulated world with five inputs and a logistic outcome trains a logistic model on 5,000 rows; then the world is changed in one of four ways and the model is deployed on batches of 200 or 2,000 rows, 50 of each. The alarm runs a two-sample Kolmogorov-Smirnov test on each input against the training data and fires if any of the five falls below 0.05 divided by five.

{snip['world'].markdown()}""")

    post.add(
        "Alarms when nothing is wrong",
        f"""Move the first input's mean by one standard deviation, leaving the outcome's dependence on the inputs unchanged. The alarm fired on every batch, of either size. Accuracy *rose*, from {base:.3f} on the validation rows before deployment to {up['accuracy']:.3f}: the moved input carries the largest coefficient, so presumably more of the new rows sit far from the decision boundary, where the model is surer and right more often. A team paged by that alarm would find a model doing better than ever.

Move it by a tenth of a standard deviation instead. On batches of 2,000 the alarm fired on {tiny['input_alarm']:.0%} of them, on batches of 200 on {tiny_s['input_alarm']:.0%}, and accuracy did not change measurably ({tiny['accuracy']:.3f} against {none['accuracy']:.3f} with no drift). A test of whether two distributions differ answers with more confidence the more data it has, and a large enough batch makes any difference significant; whether the difference matters is a separate question that the test does not ask.

The last way to raise alarms over nothing is to have many inputs. With no drift at all and each input tested at 5%, at least one test fired in {f[5]['uncorrected']:.0%} of batches with 5 inputs, {f[20]['uncorrected']:.0%} with 20 and {f[100]['uncorrected']:.0%} with 100. Run weekly, a 20-input monitor would fire about {f[20]['weekly_alarms_a_year']:.0f} times a year with nothing wrong. A Bonferroni correction brought the rate back near the 5% it promises, between {min(r['bonferroni'] for r in fa):.1%} and {max(r['bonferroni'] for r in fa):.1%} over 200 batches, at the price of missing small real shifts.""",
        figures=[figs["f0"]])

    post.add(
        "Silence when everything is wrong",
        f"""Now leave the inputs alone and change how the outcome depends on them: the first input's coefficient moves from +1.5 to 0 (half-reversed) or to -1.5 (reversed). This is what a change in customers' behaviour, a new fraud pattern or a new treatment protocol does. The input alarm fired on {half['input_alarm']:.0%} and {rev['input_alarm']:.0%} of batches, no more than with no drift; accuracy fell to **{half['accuracy']:.3f}** and **{rev['accuracy']:.3f}**, below a coin flip in the second case.

Two more label-free monitors were run alongside, and both were blind for the same reason. An alarm on the model's predicted probabilities fired on {rev['score_alarm']:.0%} of batches: the scores are a function of the inputs, and the inputs had not moved. An estimate of accuracy built from the unlabelled batch, re-weighting labelled validation rows by how much they resemble the new ones (the importance weighting of Sugiyama and colleagues), said {rev['estimate']:.3f} throughout. That estimate was excellent when the inputs moved, within {abs(up['estimate'] - up['accuracy']):.3f} of the truth, because covariate shift is exactly the assumption it makes. Under a change in the relation it reported the past.

Nothing computed from inputs alone can detect a change in the outcome given the inputs. Gama and colleagues' survey of concept drift calls it real drift; the only signal is in labelled outcomes, and episode 64 measures what it costs when those arrive late.""",
        figures=[figs["f1"]])

    post.add(
        "A third kind, and a real shift",
        f"""Between the two sits label shift: the outcome's prevalence changes and each class looks the same as before, as when a disease becomes more common. With the prevalence moved from {lo['train_prevalence']:.2f} to {lo['prevalence']:.2f}, the alarm fired on every batch and accuracy stayed close ({lo['accuracy']:.3f} against {base:.3f}), but the model's average predicted probability was {lo['mean_predicted']:.3f} against an actual rate of {lo['mean_actual']:.3f}; at a prevalence of {hi['prevalence']:.1f}, {hi['mean_predicted']:.3f} against {hi['mean_actual']:.3f}. Accuracy hid what the probabilities showed, and a decision made at a fixed probability would be wrong in proportion. Lipton, Wang and Smola showed how to estimate the new prevalence from unlabelled predictions under exactly this assumption.

The breast-cancer data give a real version of the first scenario. A logistic model trained on 300 random tumours scored {b0['accuracy']:.3f} on the other 269, with no feature alarmed. Deployed only on the tumours larger than the training median, {b50['features_alarmed']} of the 30 features raised an alarm, the share of malignant tumours rose from {b0['malignant']:.0%} to {b50['malignant']:.0%}, and accuracy was {b50['accuracy']:.3f} ({b50['errors']} errors in {b50['rows']}); on the largest quarter, {b75['features_alarmed']} alarms and {b75['accuracy']:.3f}. Differences of that size are within what so few tumours can measure. A monitor would have reported a crisis that the outcomes did not show.""")

    post.add(
        "Where this breaks",
        """**Five inputs, one model.** Real monitors watch hundreds of features and use other tests (population stability indices, embedding distances); their behaviour with sample size and with many features is the same in kind as the Kolmogorov-Smirnov test's.

**Shifts chosen to separate the cases.** Real drift usually mixes a moving input distribution with a changing relation, and an input alarm will then often fire for the wrong reason at the right time. That coincidence is not a mechanism to rely on.

**A correctly specified model.** Episode 29 showed that a misspecified model does lose accuracy under covariate shift, so an input alarm is more informative for such a model. It still cannot see a change in the relation.

**Labels assumed absent.** Where labels arrive quickly, monitoring the outcome directly makes most of this unnecessary.""")

    post.add(
        "What to keep",
        f"""1. An input alarm fired on every batch when an input moved a standard deviation, and accuracy rose from {base:.3f} to {up['accuracy']:.3f}.

2. When the relation reversed and the inputs stayed put, input and score alarms fired no more than with no drift, a label-free accuracy estimate said {rev['estimate']:.2f}, and accuracy was {rev['accuracy']:.2f}.

3. With 20 inputs tested separately, {f[20]['uncorrected']:.0%} of batches raised an alarm with nothing wrong; with large batches, a shift too small to matter raised it {tiny['input_alarm']:.0%} of the time.

4. Treat input drift as a reason to look, not as evidence of harm. Budget for labelled outcomes, however few and late, because a change in the relation shows up nowhere else.""")

    post.add(
        "Exercise",
        """List the alarms your monitoring raised in the last quarter and, for each, whether the model's measured performance on labelled outcomes changed afterwards. Then list the last two occasions the model's performance did fall and whether any input alarm preceded them.

The two lists measure your monitor's precision and recall for the thing you care about. If either is low, the missing ingredient is outcome data, not a better test on the inputs.""")

    post.add(
        "Next",
        """A monitor asks whether a model is still right on average. A prediction interval asks how far off any single prediction might be, and the usual one, a fixed band around every prediction, is right on average too. Episode 62 asks where it is wrong.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
