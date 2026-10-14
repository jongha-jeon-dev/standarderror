"""ML 7: Change a Feature's Units and the Lasso Changes Its Mind.

The seventh episode of *Machine Learning, Taught Through What Breaks*. A
penalty is described as a property of the model: lambda says how strongly
the model is regularised. A lasso penalty is a price per unit of
coefficient, and a coefficient's size depends on the units of its feature.

Measured on the diabetes data, re-expressing one feature at a time in units
from a hundredth to a hundred times the standardised ones and refitting
cross-validated lasso:

* Seven of the ten features are kept in some units and dropped in others;
  only sex, BMI and blood pressure survive every choice -- not the three
  largest coefficients: s5's is the largest, and it is dropped at x 0.01.
* The cross-validated error of all 90 refits spans 5.3% of the base
  model's: the selections differ, the cross-validated error barely does.
* Standardising is itself a choice of units. A binary feature with the same
  effect per occurrence is kept in 24 of 40 datasets at 2% prevalence when
  standardised and 7 of 40 when left as 0/1.

Run: `standarderror run ml107_units --publish`
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
EFFECT = 0.3
REPS = 40


def compute() -> dict:
    X, y, names = ln.diabetes()
    # The standardised model's own coefficients, with the same LassoCV
    # settings as unit_sweep's factor-1 refit: which features are largest.
    coef = ln._lasso_cv(X, y, seed=0)[0]
    return {"unit": ln.unit_sweep(),
            "binary": ln.binary_prevalence(effect=EFFECT, reps=REPS),
            "coef": dict(zip(names, (float(c) for c in coef)))}


def _count(share: float) -> int:
    """A keep-share over REPS datasets as the count it came from."""
    return int(round(share * REPS))


def _flippable(u):
    return [n for n, row in zip(u["names"], u["kept"])
            if row.any() and not row.all()]


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    u = res["unit"]
    fs, names = u["factors"], u["names"]

    def heat(ax, m):
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap([m.grid, m.series[0]])
        ax.imshow(u["kept"].astype(int), cmap=cmap, aspect="auto",
                  vmin=0, vmax=1)
        ax.set_xticks(range(len(fs)))
        ax.set_xticklabels([f"{k:g}" for k in fs])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.axvline(fs.index(1), lw=1.6, ls="--", color=m.ink)
        ax.grid(False)

    out["f0"] = charts.diagram(
        heat,
        title="Which features the lasso keeps depends on their units",
        subtitle=("Diabetes data, cross-validated lasso. Each row rescales "
                  "one standardised feature by the factor below - a smaller "
                  "factor is a coarser unit - and refits. Blue: kept. "
                  "Dashed: standardised units."),
        xlabel="factor applied to that one feature",
        ylabel="",
        figsize=(7.2, 4.6),
        source="Measured; standarderror/ml/linear.py.",
        alt=("A grid of ten features by nine unit factors: three rows kept "
             "throughout, seven rows dropped at small factors and kept at "
             "large ones."),
        caption=(f"**{len(_flippable(u))} of {len(names)} features** are in "
                 f"the model in some units and out of it in others. Only "
                 f"{', '.join(n for n in names if n not in _flippable(u))} "
                 f"survive every choice."),
        path=str(IMG / f"ml107-f0-units.{EXT}"))[0]

    b = res["binary"]

    def binary(ax, m):
        prev = [r["prevalence"] for r in b]
        xs = list(range(len(prev)))
        ax.plot(xs, [r["kept_standardised"] for r in b], marker="o", ms=6,
                lw=2.4, color=m.series[0], label="standardised")
        ax.plot(xs, [r["kept_raw"] for r in b], marker="s", ms=6, lw=2.4,
                color=m.series[1], label="left as 0/1 (centred)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{p:.0%}" for p in prev])
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8.4, loc="lower left")

    lo = b[-1]
    out["f1"] = charts.diagram(
        binary,
        title="Standardising is a choice of units too",
        subtitle=(f"How often cross-validated lasso keeps a binary feature "
                  f"with the same effect, {EFFECT}, per occurrence, at "
                  f"different prevalences. 500 rows, {REPS} datasets each."),
        xlabel="share of rows where the feature is 1",
        ylabel="share of datasets where it is kept",
        source="Simulated; standarderror/ml/linear.py.",
        alt=("Two falling lines; the standardised one stays higher, and the "
             "gap widens as the feature gets rarer."),
        caption=(f"At {lo['prevalence']:.0%} prevalence the same feature is "
                 f"kept in **{_count(lo['kept_standardised'])} of {REPS}** "
                 f"datasets standardised and {_count(lo['kept_raw'])} of "
                 f"{REPS} as 0/1. Neither is the neutral choice; there "
                 f"isn't one."),
        path=str(IMG / f"ml107-f1-binary.{EXT}"))[0]

    show = [0.01, 0.1, 1, 10, 100]
    cols = [fs.index(k) for k in show]
    rows_t = []
    for j, n in enumerate(names):
        rows_t.append([n] + ["kept" if u["kept"][j, c] else "-" for c in cols]
                      + [f"{u['cv_mse'][j].min():,.0f}-"
                         f"{u['cv_mse'][j].max():,.0f}"])
    out["f2"] = charts.table_image(
        rows_t,
        header=["feature"] + [f"x {k:g}" for k in show] + ["CV MSE range"],
        title="Ten features, five units each",
        subtitle=(f"Cross-validated MSE of the base model: "
                  f"{u['base_mse']:,.0f}. The last column is the range over "
                  f"all nine factors for that feature."),
        source="Measured; standarderror/ml/linear.py.",
        alt=("A ten-row table of kept and dropped marks by unit factor, "
             "with cross-validated errors that vary by a few per cent."),
        caption=("Different models, nearly the same error. The data cannot "
                 "tell these selections apart; the units chose between "
                 "them."),
        align="lcccccr",
        path=str(IMG / f"ml107-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    u, b = res["unit"], res["binary"]

    def grid(panel, m):
        from matplotlib.colors import ListedColormap
        panel.imshow(u["kept"].astype(int), cmap=ListedColormap(
            [m.grid, m.series[0]]), aspect="auto")

    def flat(panel, m):
        panel.plot(range(len(u["factors"])), u["cv_mse"].mean(0), lw=2.8,
                   color=m.series[2])
        panel.set_ylim(u["base_mse"] * 0.8, u["base_mse"] * 1.2)

    def lines(panel, m):
        panel.plot(range(len(b)), [r["kept_standardised"] for r in b],
                   lw=2.8, color=m.series[0])
        panel.plot(range(len(b)), [r["kept_raw"] for r in b], lw=2.8,
                   color=m.series[1])

    spread = (u["cv_mse"].max() - u["cv_mse"].min()) / u["base_mse"]
    return charts.lecture_hero(
        series=SERIES_TAG, episode=7,
        headline="A lasso penalty is a price per unit",
        panels=[(grid, f"{len(_flippable(u))} of 10", "features units decide"),
                (flat, f"{spread:.0%}", "spread in CV error"),
                (lines, f"{_count(b[-1]['kept_standardised'])}/{REPS} vs "
                        f"{_count(b[-1]['kept_raw'])}/{REPS}",
                 "a 2% binary, kept")],
        note=("Re-express a feature in coarser units and its coefficient "
              "must grow to say the same thing, so the lasso charges it "
              "more. Seven of ten diabetes features are kept or dropped by "
              "that choice alone, at almost no cost in prediction."),
        alt=("Three hand-drawn frames: a block grid; a flat line; and two "
             "falling lines."),
        path=str(IMG / f"ml107-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["unit"] = s.run("""
        from standarderror.ml import linear as ln

        u = ln.unit_sweep()     # each feature x 0.01 ... x 100, refit LassoCV
        for name, row in zip(u["names"], u["kept"]):
            print(f"{name:4s} " + " ".join("#" if k else "." for k in row))
        print("factors " + " ".join(f"{k:g}" for k in u["factors"]))
        """)
    out["mse"] = s.run("""
        print(f"CV MSE over all 90 refits: {u['cv_mse'].min():,.0f} to "
              f"{u['cv_mse'].max():,.0f}  (base model {u['base_mse']:,.0f})")
        """)
    out["binary"] = s.run("""
        for r in ln.binary_prevalence(effect=0.3, reps=40):
            print(f"prevalence {r['prevalence']:4.0%}  kept: standardised "
                  f"{r['kept_standardised']:.2f}   raw 0/1 {r['kept_raw']:.2f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    u, b = res["unit"], res["binary"]
    flip = _flippable(u)
    always = [n for n in u["names"] if n not in flip]
    spread = (u["cv_mse"].max() - u["cv_mse"].min()) / u["base_mse"]
    age = u["names"].index("age")
    k = u["factors"]
    age_out = max(f for f, kept in zip(k, u["kept"][age]) if not kept)
    s3 = u["names"].index("s3")
    lo, hi = b[-1], b[0]
    n_std, n_raw = _count(lo["kept_standardised"]), _count(lo["kept_raw"])
    coef = res["coef"]
    top = max(coef, key=lambda n: abs(coef[n]))
    top_out = max(f for f, kept in zip(k, u["kept"][u["names"].index(top)])
                  if not kept)

    # The spine, asserted rather than trusted.
    # Surviving every unit is not the same as having the largest
    # coefficient: the largest one flips.
    assert top in flip and top not in always
    assert len(flip) >= 6 and set(always) == {"sex", "bmi", "bp"}
    assert spread < 0.06
    assert not u["base"][s3] and u["kept"][s3, k.index(3)]
    assert all(row[-1] for row in u["kept"])
    assert lo["kept_standardised"] - lo["kept_raw"] > 0.3
    assert hi["kept_standardised"] > 0.9 and hi["kept_raw"] > 0.9
    assert lo["kept_standardised"] < hi["kept_standardised"]

    post = Post(
        title=(f"{SERIES_TAG} 7: Change a Feature's Units and the Lasso "
               f"Changes Its Mind"),
        slug="ml-7-units",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=7,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A penalty strength is taught as a property of the "
                  "model. A lasso penalty is a price per unit of "
                  "coefficient, so a feature's units decide what it "
                  "pays - and whether it stays."),
        summary=(
            f"On the diabetes data, re-expressing one feature at a time in "
            f"units from a hundredth to a hundred times the standardised "
            f"ones and refitting cross-validated lasso, {len(flip)} of 10 "
            f"features are kept in some units and dropped in others; only "
            f"{', '.join(always)} survive every choice. The 90 refits' "
            f"cross-validated errors span {spread:.1%} of the base model's: "
            f"different selections, near-identical cross-validated error. "
            f"Standardising does not escape the choice - it is one. A "
            f"binary feature with the same effect per occurrence is kept in "
            f"{n_std} of {REPS} datasets at "
            f"{lo['prevalence']:.0%} prevalence when standardised and "
            f"{n_raw} of {REPS} when left as 0/1."),
        tags=["machine-learning", "data-science", "statistics", "lasso",
              "regularisation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Diabetes data of Efron, Hastie, Johnstone and Tibshirani "
            "(2004), as bundled with scikit-learn (`load_diabetes`): 442 "
            "patients, 10 baseline variables, a disease-progression score.",
            "Simulated: a binary feature with effect 0.3 per occurrence "
            "beside five continuous features, two of them with effects of "
            "0.5, unit noise, 500 rows per dataset.",
            "Machinery: `standarderror/ml/linear.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Tibshirani, \"Regression shrinkage and "
            "selection via the lasso\", *JRSS B* (1996), which standardises "
            "and notes why; Gelman, \"Scaling regression inputs by dividing "
            "by two standard deviations\", *Statistics in Medicine* (2008), "
            "on binary inputs and standardisation.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("LassoCV with 5 seeded folds and a 60-value "
                            "alpha grid throughout; simulated datasets from "
                            "fixed seeds"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A penalty on what?",
        f"""Episode 6 ended with a penalty as the thing that turns an unanswerable question into one with an answer. This episode asks what the penalty is a penalty *on*.

The lasso fits a linear model while charging λ times the sum of the absolute coefficients. That charge is what makes it select: a feature whose coefficient is not worth its price is set to exactly zero. It is usually taught as a property of the model — λ is "how much regularisation" — with a footnote saying to standardise first.

The footnote is the whole story. The price is per unit of *coefficient*, and a coefficient's size depends on the units of its feature. Measure age in decades instead of years and the coefficient that describes the same effect is ten times bigger, so it costs ten times as much. Nothing about the data or the relationship changed; the bill did.

The arithmetic is one line. Multiply a feature's column by k and the coefficient that gives the same predictions becomes β/k, so its penalty becomes λ|β|/k. A finer unit (k above 1) makes the feature cheaper; a coarser one (k below 1) makes it dearer. Everything else in the objective — the fit to the data, the other features' prices — is unchanged, so moving one feature's units is exactly the same as giving that one feature its own private λ, scaled by 1/k. No one would choose per-feature penalties by consulting the data dictionary's units, but that is what an unstandardised lasso does.

The diabetes data makes this concrete: 442 patients, ten baseline measurements, a disease-progression score, all standardised. Take one feature at a time, multiply its column by a factor from 0.01 to 100 — a coarser or finer unit for that one measurement — and refit cross-validated lasso, which re-chooses λ every time.

{snip['unit'].markdown()}

A `#` is kept, a `.` dropped; the middle column, factor 1, is the standardised model. **{len(flip)} of the ten features change status with their units.** Age, for instance, is dropped at every factor up to {age_out:g} and kept from 1 upward; s3 is out of the standardised model and in it at a factor of 3. Only {', '.join(always)} are kept whatever their units, and they are not simply the strongest: in the standardised fit the largest coefficient belongs to {top} ({coef[top]:+.1f}), and {top} is dropped at a factor of {top_out:g}. A large coefficient in standard deviations is no protection once a coarse enough unit makes each unit of it expensive.""",
        figures=[figs["f0"]])

    post.add(
        "Different models, nearly the same error",
        f"""If the different selections predicted differently, the units would at least be choosing between better and worse models. They are not.

{snip['mse'].markdown()}

Across all ninety refits the cross-validated error spans {spread:.1%} of the base model's. Within that, the coarse-unit models that drop age actually score a touch *better*. The data cannot tell these models apart; what told them apart was the arbitrary scale each feature happened to be recorded in.

That is the practical risk. A lasso's selected set is routinely reported as a finding — these are the variables that matter — and in fields with mixed units (a dose in milligrams, an age in years, a count, an indicator) the finding partly reflects the units the data dictionary happened to use. The predictive model is fine. The list of "important" variables is not a property of the disease.""",
        figures=[figs["f2"]])

    post.add(
        "Standardising does not make it neutral",
        f"""The standard fix is to standardise every feature to unit variance, which removes the arbitrariness by imposing one specific choice: every feature is measured in its own standard deviations. That is a choice of units like any other, and it has consequences for features whose standard deviation means something odd — binary ones above all, whose standard deviation is sqrt(p(1 - p)) and shrinks as the feature gets rarer.

So: a binary feature with the same effect, {EFFECT} per occurrence, beside five continuous features, at prevalences from 50% down to 2%. Cross-validated lasso, standardised and with the 0/1 column only centred:

{snip['binary'].markdown()}

At 50% both versions keep the feature nearly always. At {lo['prevalence']:.0%}, the standardised lasso keeps it in {n_std} of {REPS} datasets and the raw one in {n_raw}. Standardising divides a rare indicator by a small standard deviation, which inflates its column, shrinks its coefficient and lowers its price; the raw version charges it per occurrence, and a rare occurrence has little data behind it.

Neither is right. A per-occurrence effect and a per-standard-deviation effect are different questions, and the lasso answers whichever its units ask. Gelman's suggestion to divide continuous inputs by two standard deviations so that they compare with untouched binaries is one principled answer; it is still a choice.""",
        figures=[figs["f1"]])

    post.add(
        "Where this breaks",
        f"""**Prediction is barely affected.** Everything here is about *which* features a lasso keeps, not how well it predicts. If the only use of the model is its predictions, the units matter far less — though the coefficients still do not mean what they appear to.

**Ridge has the same dependence, without the drama.** A ridge penalty is also per unit of coefficient, so rescaling a feature changes how much it is shrunk, but ridge never sets anything to zero, so there is no visible flip.

**Cross-validation does not fix it.** Every refit here re-chose λ by cross-validation. The units are not a tuning problem; they change the shape of the penalty, not its strength.

**Weighted penalties make the choice explicit rather than removing it.** The adaptive lasso gives each feature its own penalty weight, usually from a preliminary fit, and group penalties price whole blocks of features together. Both are ways of choosing the per-feature prices on purpose instead of inheriting them from the units; neither makes the prices disappear.

**One dataset and one simulation.** The count of {len(flip)} flips belongs to the diabetes data and to factors up to a hundred. A feature that flips at a factor of 100 may never be recorded in a unit that coarse; one that flips at 3 easily could be.""")

    post.add(
        "What to keep",
        f"""1. A lasso penalty is a price per unit of coefficient, and a coefficient's units are its feature's inverse units. Rescale a feature and its price changes.

2. On the diabetes data, {len(flip)} of 10 features are kept or dropped depending on their units alone; only {', '.join(always)} survive every factor from 0.01 to 100.

3. The cross-validated errors of all those models span {spread:.1%}: the data cannot distinguish them.

4. Standardising is a choice of units. A binary feature at {lo['prevalence']:.0%} prevalence is kept in {n_std} of {REPS} datasets standardised and {n_raw} of {REPS} raw.

5. Report a lasso's selected set as a property of the features *in the units you chose*, or not as a finding at all.""")

    post.add(
        "Exercise",
        """Take a lasso you have fitted and list its features with their units. Pick the two you are least sure about, multiply each column by 10 and by 0.1 in turn, refit with cross-validation, and see whether either changes status. If one does, its presence in your model was partly a unit decision, and a reader of the selected set should be told.

If your model mixes binary and continuous features, refit once with everything standardised and once with binaries left as 0/1 and only the continuous ones standardised. If the selected set changes, decide which question you meant to ask — effect per occurrence or per standard deviation — and say so.""")

    post.add(
        "Next",
        """Episode 8 keeps the units fixed and asks the question a lasso is usually fitted to answer: are the features it keeps the relevant ones? On simulated data where the answer is known, and on bootstrap resamples of the diabetes data where it is not.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
