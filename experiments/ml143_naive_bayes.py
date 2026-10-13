"""ML 43: Naive Bayes Counts Correlated Evidence Twice, and Only Its Probabilities Notice.

The forty-third episode of *Machine Learning, Taught Through What Breaks*.
"Naive Bayes works well even though its independence assumption is
wrong." Measured on the breast-cancer data, whose thirty features include
radius, perimeter and area of the same nuclei:

* Its accuracy (0.940) trailed logistic regression (0.981) by four points;
  its log loss (0.611 against 0.073) by a factor of eight, and 67% of its
  errors came with confidence above 0.99.
* Copying every feature k times multiplies its log-odds by k: accuracy
  stayed at 0.94 and the AUC of the log-odds was unchanged to four
  decimals, while log loss rose to 1.789 at k = 10 and the AUC of its
  probabilities fell to 0.955, because 80% of them rounded to exactly 0
  or 1.
* Copying only the nine size features changed the ranking (log-odds AUC
  0.975 at k = 10, accuracy 0.914); copying the three texture features five
  times raised accuracy to 0.950. Recalibrating the log-odds cut log loss
  to 0.149 at unchanged accuracy.

Run: `standarderror run ml143_naive_bayes --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import combine
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 13)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"base": combine.nb_baseline(), "dup": combine.nb_duplicates(),
            "fam": combine.nb_family(), "rep": combine.nb_repairs()}


# ------------------------------------------------------------------ figures

def _dup(ax, m, dup):
    ks = [r["k"] for r in dup]
    ax.plot(ks, [r["acc"] for r in dup], marker="o", color=m.series[1],
            label="accuracy")
    ax.plot(ks, [r["auc_log_odds"] for r in dup], marker="o", color=m.series[2],
            label="AUC of the log-odds")
    ax.plot(ks, [r["auc"] for r in dup], marker="o", color=m.series[3],
            label="AUC of the probabilities")
    ax.plot(ks, [r["exact_01"] for r in dup], marker="s", ls="--", color=m.ink,
            label="share of probabilities exactly 0 or 1")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.legend(frameon=False, fontsize=8.5, loc="center right")


def figures(res: dict) -> dict:
    out: dict = {}
    dup, fam, rep, base = res["dup"], res["fam"], res["rep"], res["base"]
    d10 = dup[-1]

    out["f0"] = charts.diagram(
        lambda ax, m: _dup(ax, m, dup),
        title="Count every feature k times: the order holds, the doubt goes",
        subtitle=("Gaussian naive Bayes on breast cancer with each of the 30 "
                  "features copied k times; five-fold cross-validation over "
                  "5 fold assignments."),
        xlabel="copies of every feature (log scale)",
        ylabel="",
        figsize=(7.4, 4.6),
        source=("Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, via "
                "scikit-learn; standarderror/ml/combine.py."),
        alt=("Accuracy and the AUC of the log-odds are flat lines; the AUC "
             "of the probabilities drifts down; the dashed share of "
             "probabilities that are exactly 0 or 1 climbs from about 0.14 "
             "to 0.80."),
        caption=(f"At k = 10 the AUC of the log-odds is still "
                 f"{d10['auc_log_odds']:.4f}; the AUC of the probabilities is "
                 f"**{d10['auc']:.3f}**, because {d10['exact_01']:.0%} of them "
                 f"are exactly 0 or 1."),
        path=str(IMG / f"ml143-f0-copies.{EXT}"))[0]

    body = [["naive Bayes", "", f"{base['nb']['acc']:.3f}",
             f"{base['nb']['auc_log_odds']:.4f}", f"{base['nb']['nll']:.3f}",
             f"{base['nb']['confident_errors']:.0%}"],
            ["logistic regression", "", f"{base['logistic']['acc']:.3f}",
             f"{base['logistic']['auc']:.4f}", f"{base['logistic']['nll']:.3f}",
             f"{base['logistic']['confident_errors']:.0%}"]]
    body += [[f"naive Bayes, all features x{r['k']}", "", f"{r['acc']:.3f}",
              f"{r['auc_log_odds']:.4f}", f"{r['nll']:.3f}",
              f"{r['confident_errors']:.0%}"] for r in dup[1:]]
    body += [[f"naive Bayes, {r['family']} x{r['k']}",
              f"{r['features']} features", f"{r['acc']:.3f}",
              f"{r['auc_log_odds']:.4f}", f"{r['nll']:.3f}",
              f"{r['confident_errors']:.0%}"] for r in fam]
    for name, key in (("recalibrated log-odds", "platt"),
                      ("three features only", "three"),
                      ("on whitened components", "whitened")):
        r = rep[key]
        auc = r["auc_log_odds"] if r["auc_log_odds"] == r["auc_log_odds"] else r["auc"]
        body.append([f"naive Bayes, {name}", "", f"{r['acc']:.3f}",
                     f"{auc:.4f}", f"{r['nll']:.3f}",
                     f"{r['confident_errors']:.0%}"])
    out["f1"] = charts.table_image(
        body,
        header=["model", "copied", "accuracy", "AUC", "log loss",
                "errors at > 0.99"],
        title="What double-counting changes, and what it leaves alone",
        subtitle=("Five-fold cross-validation on 569 tumours, 5 fold "
                  "assignments. AUC of naive Bayes from its log-odds, so "
                  "rounding cannot create ties. Last column: share of the "
                  "model's errors made with confidence above 0.99."),
        source=("Breast Cancer Wisconsin (Diagnostic), CC BY 4.0; "
                "standarderror/ml/combine.py."),
        alt=("A fifteen-row table. Copying all features leaves accuracy and "
             "AUC flat while log loss triples; copying the size family "
             "lowers both; recalibration brings log loss to 0.149."),
        caption=("Copying everything rescales the log-odds; copying one "
                 "family reweights it. Only the second changes which tumour "
                 "ranks above which."),
        align="llrrrr",
        path=str(IMG / f"ml143-f1-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    dup, base, rep = res["dup"], res["base"], res["rep"]

    def p1(panel, m):
        panel.bar([0, 1], [base["nb"]["acc"], base["logistic"]["acc"]],
                  color=[m.series[1], m.series[2]])
        panel.set_ylim(0.9, 1.0)

    def p2(panel, m):
        panel.plot([r["nll"] for r in dup], color=m.series[1], lw=2)
        panel.plot([r["acc"] for r in dup], color=m.ink, lw=2)

    def p3(panel, m):
        panel.bar([0, 1], [base["nb"]["nll"], rep["platt"]["nll"]],
                  color=[m.series[1], m.series[3]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=43,
        headline="Evidence counted twice",
        panels=[(p1, f"{base['nb']['confident_errors']:.0%}",
                 "errors at > 0.99"),
                (p2, f"{dup[-1]['nll']:.3f}", "log loss, x10 copies"),
                (p3, f"{rep['platt']['nll']:.3f}", "recalibrated log loss")],
        note=("Naive Bayes adds up evidence as if every feature were a new "
              "witness. Exact copies inflate its certainty without changing "
              "its verdicts, so its accuracy hides the problem."),
        alt=("Three hand-drawn frames: two bars, the second taller; a rising "
             "line over a flat one; a tall bar and a short one."),
        path=str(IMG / f"ml143-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["base"] = s.run("""
        from standarderror.ml import combine

        b = combine.nb_baseline()      # breast cancer, 5 x five-fold CV
        for name in ("nb", "logistic"):
            r = b[name]
            print(f"{name:9s} accuracy {r['acc']:.3f}, log loss {r['nll']:.3f}, "
                  f"errors made at > 0.99 confidence {r['confident_errors']:.0%}")
        """)
    out["dup"] = s.run("""
        for r in combine.nb_duplicates():   # every feature copied k times
            print(f"k = {r['k']:2d}: accuracy {r['acc']:.4f}, AUC log-odds "
                  f"{r['auc_log_odds']:.4f}, AUC probs {r['auc']:.4f}, "
                  f"log loss {r['nll']:.3f}")
        """)
    out["fam"] = s.run("""
        for r in combine.nb_family():       # one family copied k times
            print(f"{r['family']:7s} x{r['k']:<2d}: accuracy {r['acc']:.3f}, "
                  f"AUC of log-odds {r['auc_log_odds']:.4f}, "
                  f"log loss {r['nll']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    base, dup, fam, rep = res["base"], res["dup"], res["fam"], res["rep"]
    nb, lg = base["nb"], base["logistic"]
    d = {r["k"]: r for r in dup}
    f = {(r["family"], r["k"]): r for r in fam}
    pl, th, wh = rep["platt"], rep["three"], rep["whitened"]

    # The spine, asserted rather than trusted.
    assert lg["acc"] - nb["acc"] > 0.035 and nb["nll"] > 8 * lg["nll"]
    assert nb["confident_errors"] > 0.65 and lg["confident_errors"] < 0.1
    assert all(abs(r["auc_log_odds"] - nb["auc_log_odds"]) < 1e-5 for r in dup)
    assert all(abs(r["acc"] - nb["acc"]) < 0.0025 for r in dup)
    assert all(a["nll"] < b["nll"] for a, b in zip(dup, dup[1:]))
    assert all(a["auc"] > b["auc"] for a, b in zip(dup, dup[1:]))
    assert d[10]["auc"] < 0.96 and d[10]["exact_01"] > 0.8 and d[10]["nll"] > 1.75
    assert f[("size", 10)]["auc_log_odds"] < 0.976 and f[("size", 10)]["acc"] < 0.915
    assert f[("texture", 10)]["acc"] < nb["acc"] < f[("texture", 2)]["acc"]
    assert f[("texture", 5)]["acc"] > 0.95 and f[("texture", 5)]["nll"] < nb["nll"]
    assert abs(pl["acc"] - nb["acc"]) < 0.001 and pl["nll"] < 0.15
    assert abs(pl["auc_log_odds"] - nb["auc_log_odds"]) < 1e-12
    assert pl["confident_errors"] < 0.03
    assert th["nll"] < 0.2 and th["acc"] < nb["acc"]
    assert wh["acc"] < 0.86

    post = Post(
        title=(f"{SERIES_TAG} 43: Naive Bayes Counts Correlated Evidence "
               f"Twice, and Only Its Probabilities Notice"),
        slug="ml-43-naive-bayes",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=43,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("The independence assumption is wrong and naive Bayes still "
                  "classifies well, as the textbooks say. What the "
                  "assumption breaks is the one thing the model's name "
                  "promises: its probabilities."),
        summary=(
            f"Gaussian naive Bayes on the breast-cancer data, whose features "
            f"include radius, perimeter and area of the same nuclei, scored "
            f"accuracy {nb['acc']:.3f} against logistic regression's "
            f"{lg['acc']:.3f}, with log loss {nb['nll']:.3f} against "
            f"{lg['nll']:.3f}; {nb['confident_errors']:.0%} of its errors "
            f"came with confidence above 0.99. Copying every feature ten "
            f"times left its accuracy at {d[10]['acc']:.3f} and the AUC of "
            f"its log-odds unchanged, and raised its log loss to "
            f"{d[10]['nll']:.3f}. Copying only the size features lowered "
            f"its accuracy to {f[('size', 10)]['acc']:.3f}. Recalibrating "
            f"its log-odds cut log loss to {pl['nll']:.3f} at unchanged "
            f"accuracy."),
        tags=["machine-learning", "data-science", "statistics",
              "naive-bayes", "calibration", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, as bundled "
            "with scikit-learn; 569 tumours, 30 features (ten measurements "
            "of the nuclei, each as a mean, a standard error and a worst "
            "value).",
            "Machinery: `standarderror/ml/combine.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's GaussianNB (log-odds from "
            "predict_joint_log_proba), LogisticRegression after "
            "standardising, PCA with whitening; Platt recalibration of the "
            "log-odds fitted on out-of-fold log-odds inside each training "
            "set; five stratified folds, fold assignments 0 to 4.",
            "Where this stops: Domingos and Pazzani, \"On the optimality of "
            "the simple Bayesian classifier under zero-one loss\", *Machine "
            "Learning* (1997); Zhang, \"The optimality of naive Bayes\", "
            "*FLAIRS* (2004); Niculescu-Mizil and Caruana, \"Predicting good "
            "probabilities with supervised learning\", *ICML* (2005).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("fold assignments 0 to 4; under a minute of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A combination of features, not of models",
        f"""Episode 41 combined models by averaging their probabilities and found that the most confident member decides. Naive Bayes combines too, inside a single model: it treats each feature as a separate, independent piece of evidence about the class and adds their log-likelihood ratios. That is exactly right if the features are independent within each class and wrong otherwise, and they almost never are. The sentence that keeps the method in every textbook is the defence: *naive Bayes works well even though its independence assumption is wrong*.

The breast-cancer data make the assumption as wrong as it gets. Ten measurements of each tumour's nuclei are recorded three times each (as a mean, a standard error and a worst value), and three of the ten, radius, perimeter and area, measure the same thing in different units. Gaussian naive Bayes against logistic regression, which fits one weight per feature jointly and so accounts for overlap:

{snip['base'].markdown()}

Naive Bayes is four points less accurate, which is the kind of gap that keeps it in use as a fast baseline. Its log loss is **{nb['nll'] / lg['nll']:.0f} times** larger, and that is a different kind of gap: **{nb['confident_errors']:.0%}** of its mistakes were made with more than 99% confidence, against {lg['confident_errors']:.0%} of logistic regression's.""")

    post.add(
        "Copying the evidence",
        f"""Domingos and Pazzani explained in 1997 why naive Bayes classifies well despite the assumption: classification needs only the sign of the log-odds, or their order for ranking, and dependence between features distorts their size far more than their sign. The cleanest test copies the features. With every one of the thirty features present k times, naive Bayes adds every log-likelihood ratio k times, which multiplies its log-odds by k, apart from the prior term it adds once.

{snip['dup'].markdown()}

Multiplying log-odds by a positive number does not change their order, and the AUC computed from the log-odds is {nb['auc_log_odds']:.4f} at every k, the same to within {max(abs(r['auc_log_odds'] - nb['auc_log_odds']) for r in dup) * 1e6:.0f} millionths, one near-tie resolved differently, by rounding or by the prior term that does not scale with k. Accuracy barely moves, from {nb['acc']:.4f} to {d[10]['acc']:.4f}, the small change coming from the prior term no longer scaling with the rest. The probabilities are another matter. With ten copies, log loss is **{d[10]['nll']:.3f}**, and {d[10]['exact_01']:.0%} of the predicted probabilities are exactly 0 or 1 in floating point. That rounding even shows up in the ranking metric: the AUC computed from the probabilities falls to {d[10]['auc']:.3f}, because tumours that differ in log-odds tie at 1.0.

So the textbook defence is right about what it claims and silent about the rest. The real breast-cancer features are not ten copies of each other, but the size features come close, and the model counts them as separate witnesses saying the same thing.""",
        figures=[figs["f0"]])

    post.add(
        "Copying part of the evidence",
        f"""Copying every feature scales the evidence. Copying some features reweights it, and that does change the verdicts:

{snip['fam'].markdown()}

Nine size features copied ten times push the model towards judging tumours by size alone: the AUC of its log-odds falls to {f[('size', 10)]['auc_log_odds']:.4f} and its accuracy to **{f[('size', 10)]['acc']:.3f}**. Three texture features copied five times push it the other way, and accuracy *rises*, to {f[('texture', 5)]['acc']:.3f}, with lower log loss than the original ({f[('texture', 5)]['nll']:.3f} against {nb['nll']:.3f}).

That second result is a reminder of what the real data already do. Naive Bayes on the original features is not a neutral model with an independence flaw; it is a model that weights each group of correlated features by how many columns it happens to occupy. The size of the nuclei occupies nine columns and texture three: each statistic of size is counted three times, once per unit it is measured in, and each statistic of texture once. Upweighting texture helped at two and five copies, which is consistent with the original features overcounting size; at ten copies texture was overcounted in turn ({f[('texture', 10)]['acc']:.3f}).""",
        figures=[figs["f1"]])

    post.add(
        "Repairs",
        f"""Three repairs follow from the mechanism. The first accepts the order and fixes the scale: fit a logistic regression on naive Bayes' log-odds, using out-of-fold log-odds inside each training set, as Platt scaling does. Accuracy stays at {pl['acc']:.3f} and the ranking is untouched, while log loss falls from {nb['nll']:.3f} to **{pl['nll']:.3f}** and the share of errors made at above 0.99 confidence from {nb['confident_errors']:.0%} to {pl['confident_errors']:.0%}. Niculescu-Mizil and Caruana found naive Bayes among the models that gain most from calibration.

The second removes the redundancy: three weakly related features, mean radius, mean texture and mean smoothness, give log loss {th['nll']:.3f} with accuracy {th['acc']:.3f}, better probabilities and fewer correct answers. The third, which looks principled, failed: decorrelating the features first with whitened principal components gave accuracy {wh['acc']:.3f}. Two things may explain it, neither isolated here: whitening removes the correlation across all tumours, not within each class, so it does not make the features independent given the class, which is the only independence naive Bayes assumes; and it inflates the smallest components, mostly noise, to the same variance as the informative ones.

None of the three beats logistic regression, which solves the double-counting problem by fitting the weights jointly. The case for naive Bayes is speed, tiny data and text, where it remains a strong baseline; the case against it is any use of its probabilities as probabilities.""")

    post.add(
        "Where this breaks",
        """**One data set with heavy redundancy.** Breast cancer is close to the worst case for double-counting. With features that really are close to independent within class, naive Bayes' probabilities are much better, which is what episode 34's simulated task showed.

**Gaussian naive Bayes.** The multinomial and Bernoulli versions used for text double-count in the same way, through words that co-occur, and are calibrated no better.

**Copies are extreme.** Exact duplicates make the scaling exact. Real correlated features double-count partially and unevenly, which is the reweighting case rather than the scaling one.

**Five-fold cross-validation on 569 tumours.** Differences of a point of accuracy are within noise; the factor of eight in log loss and the share of confident errors are not.""")

    post.add(
        "What to keep",
        f"""1. On breast cancer, naive Bayes was {lg['acc'] - nb['acc']:.3f} less accurate than logistic regression and had {nb['nll'] / lg['nll']:.0f} times its log loss; {nb['confident_errors']:.0%} of its errors came at above 0.99 confidence.

2. Copying every feature k times multiplies the log-odds by k: the AUC of the log-odds was unchanged to four decimal places, and log loss rose to {d[10]['nll']:.3f} at k = 10.

3. Copying one family reweights the evidence and changes the answers: {f[('size', 10)]['acc']:.3f} with the size features copied ten times, {f[('texture', 5)]['acc']:.3f} with texture copied five.

4. Use naive Bayes for ranking and quick baselines; recalibrate its log-odds before reading them as probabilities, and remember that its weights are set by how many columns each kind of evidence occupies.""")

    post.add(
        "Exercise",
        """Train naive Bayes on a data set you know and plot a histogram of its predicted probabilities, then the same for logistic regression. Count the predictions above 0.99 and below 0.01, and the errors among them.

Then compute the correlation matrix of your features within one class and find the largest group of strongly correlated columns. Remove all but one column of that group, retrain, and compare log loss and accuracy with the full model.""")

    post.add(
        "Next",
        """Combining models and features is one half of Arc IX; comparing them is the other. Episode 44 asks how often the standard statistical tests declare one of two equally good algorithms the better one, and which test can be trusted.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
