"""ML 19: A Planted Shortcut Wins the Test Set and Loses Everywhere Else.

The nineteenth episode of *Machine Learning, Taught Through What Breaks*.
"The model with the better test score learned the task better." On the
digits, one corner pixel -- blank in every real image -- is lit on most
images of four classes, in training and test alike. A random forest:

* With the marker on 90% of those classes' images, it scores 0.986 on a test
  set drawn the same way, against 0.971 for a forest trained on clean data.
  By the test score it is the better model.
* On clean images it scores 0.947; on the four marked classes, 0.895 against
  0.954. With the markers moved to the wrong classes, 0.663.
* With the marker on every image of those classes the test score rises to
  0.991 and the clean score falls to 0.884.
* Permutation importance on the shortcut test set ranks a marker pixel
  first among 64 -- the shortcut is visible to anyone who looks.

Run: `standarderror run ml119_shortcut --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import reading as rd
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

POST_DATE = date(2026, 10, 12)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"


def compute() -> dict:
    return {"study": rd.shortcut_study()}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    st = res["study"]
    rates = [r["rate"] for r in st]
    x = np.arange(len(rates))

    def lines(ax, m):
        for c, key, lab in ((m.series[0], "same",
                             "test set with the same shortcut"),
                            (m.series[2], "clean", "clean test set"),
                            (m.series[3], "marked_clean",
                             "clean test set, marked classes only"),
                            (m.series[1], "wrong",
                             "markers on the wrong classes")):
            ax.plot(x, [r[key] for r in st], lw=2.4, marker="o", ms=5,
                    color=c, label=lab)
        ax.axhline(st[0]["clean"], lw=1, ls="--", color=m.ink)
        ax.text(x[-1], st[0]["clean"] - 0.016, "clean-trained model",
                ha="right", fontsize=8.5, color=m.ink)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.0%}" for r in rates])
        ax.set_ylim(0.55, 1.0)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    r9 = [r for r in st if r["rate"] == 0.9][0]
    out["f0"] = charts.diagram(
        lines,
        title="The better the shortcut, the better the test score",
        subtitle=("Random forests on the digits with a marker pixel lit on "
                  "a share of the images of 3, 5, 8 and 9, scored four ways. "
                  "Means of three half splits."),
        xlabel="share of the four classes' training images carrying the "
               "marker",
        ylabel="test accuracy",
        source="UCI digits (scikit-learn); standarderror/ml/reading.py.",
        alt=("Four lines: the same-shortcut test score rising above the "
             "dashed clean baseline; the clean, marked-class and wrong-marker "
             "scores falling, the last steeply."),
        caption=(f"At 90% the shortcut model beats the clean one on its own "
                 f"test set, {r9['same']:.3f} to {st[0]['same']:.3f}, and "
                 f"falls to **{r9['wrong']:.3f}** when the markers move."),
        path=str(IMG / f"ml119-f0-lines.{EXT}"))[0]

    def images(ax, m):
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        canvas = np.ones((8 * 2 + 1, 8 * 4 + 3)) * -1
        for col, cls in enumerate(rd.SHORTCUT):
            idx = np.where(y == cls)[0][0]
            clean = X[idx].reshape(8, 8)
            mark = rd.plant(X[[idx]], y[[idx]], 1.0, seed=0)[0].reshape(8, 8)
            canvas[0:8, col * 9:col * 9 + 8] = clean
            canvas[9:17, col * 9:col * 9 + 8] = mark
        img = np.ma.masked_less(canvas, 0)
        ax.imshow(img, cmap="Greys", vmin=0, vmax=16, interpolation="nearest")
        ax.set_xticks([4 + 9 * i for i in range(4)])
        ax.set_xticklabels([f"class {c}" for c in rd.SHORTCUT])
        ax.set_yticks([4, 13])
        ax.set_yticklabels(["clean", "with marker"])
        ax.grid(False)

    out["f1"] = charts.diagram(
        images,
        title="The shortcut: one pixel per class",
        subtitle=("An 8 x 8 image of each marked class, as it is and with "
                  "its marker pixel lit. Each marker sits where no real "
                  "digit in the data has more than a trace of ink."),
        xlabel="",
        ylabel="",
        ticks=True,
        figsize=(7.2, 4.0),
        source="UCI digits (scikit-learn); standarderror/ml/reading.py.",
        alt=("Two rows of four small digit images; the lower row identical "
             "to the upper except for one black corner or edge pixel."),
        caption=("A watermark, a scanner, a hospital's label printer or a "
                 "date stamp can do the same thing to real data, and nobody "
                 "has to notice."),
        path=str(IMG / f"ml119-f1-images.{EXT}"))[0]

    rows = []
    for r in st:
        rows.append([f"{r['rate']:.0%}", f"{r['same']:.3f}",
                     f"{r['clean']:.3f}", f"{r['marked_clean']:.3f}",
                     f"{r['wrong']:.3f}", f"{r['marker_rank']:.0f}"])
    out["f2"] = charts.table_image(
        rows,
        header=["marker rate", "same-shortcut test", "clean test",
                "clean, marked classes", "wrong markers",
                "best marker's rank (of 64)"],
        title="One model, four test sets",
        subtitle=("Accuracy of 300-tree random forests, means of three half "
                  "splits. The rank is by permutation importance on the "
                  "same-shortcut test set."),
        source="UCI digits (scikit-learn); standarderror/ml/reading.py.",
        alt=("A four-row table: the same-shortcut score rising with the "
             "marker rate while every other score falls; the marker pixel "
             "ranked first by permutation importance from 90% on."),
        caption=("Only the first column is what a standard evaluation "
                 "reports. The last column is how to catch it."),
        align="rrrrrr",
        path=str(IMG / f"ml119-f2-table.{EXT}"))[0]

    out["hero"] = _hero(st)
    return out


def _hero(st):
    r9 = [r for r in st if r["rate"] == 0.9][0]
    c0 = st[0]

    def win(panel, m):
        panel.bar([0, 1], [c0["same"] - 0.9, r9["same"] - 0.9],
                  color=[m.series[2], m.series[0]])

    def lose(panel, m):
        panel.bar([0, 1], [c0["wrong"] - 0.5, r9["wrong"] - 0.5],
                  color=[m.series[2], m.series[1]])

    def rank(panel, m):
        panel.plot(range(len(st)), [-np.log(r["marker_rank"]) for r in st],
                   lw=2.8, color=m.series[3])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=19,
        headline="The better test score learned a pixel",
        panels=[(win, f"{r9['same']:.3f}", "test score, shortcut"),
                (lose, f"{r9['wrong']:.3f}", "markers moved"),
                (rank, "1 of 64", "marker's importance")],
        note=("Light one blank pixel on most images of four digit classes, "
              "in training and test alike. The model that uses it beats a "
              "clean model on the test set and collapses when the pixel "
              "means something else. Permutation importance points straight "
              "at it."),
        alt=("Three hand-drawn frames: a short bar beside a taller one; a "
             "tall bar beside a short one; a rising line."),
        path=str(IMG / f"ml119-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["study"] = s.run("""
        from standarderror.ml import reading as rd

        # marker pixel on `rate` of the 3s, 5s, 8s and 9s, train and test
        for r in rd.shortcut_study(rates=(0.0, 0.5, 0.9, 1.0)):
            print(f"rate {r['rate']:4.0%}  same-shortcut test {r['same']:.3f}"
                  f"  clean test {r['clean']:.3f}  wrong markers {r['wrong']:.3f}")
        """)
    out["detect"] = s.run("""
        for r in rd.shortcut_study(rates=(0.5, 0.9)):
            print(f"rate {r['rate']:.0%}: best marker pixel ranks "
                  f"{r['marker_rank']:.0f} of 64 by permutation importance")
            print(f"          marked-class accuracy on clean images "
                  f"{r['marked_clean']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    st = res["study"]
    c0, r5, r9, r1 = st

    # The spine, asserted rather than trusted.
    assert c0["same"] == c0["clean"] == c0["wrong"]
    assert r9["same"] > c0["same"] + 0.01 and r1["same"] > r9["same"]
    assert r9["clean"] < c0["clean"] - 0.015 and r1["clean"] < r9["clean"]
    assert r9["marked_clean"] < c0["marked_clean"] - 0.04
    assert r9["wrong"] < 0.75 and r1["wrong"] < r9["wrong"]
    assert r9["marker_rank"] == 1 and r1["marker_rank"] == 1
    assert c0["marker_rank"] > 20

    post = Post(
        title=(f"{SERIES_TAG} 19: A Planted Shortcut Wins the Test Set and "
               f"Loses Everywhere Else"),
        slug="ml-19-shortcut",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=19,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("The model with the better test score is usually taken to "
                  "have learned the task better. Plant one telltale pixel "
                  "in the digits and the better score belongs to the model "
                  "that learned the pixel."),
        summary=(
            f"A marker pixel, blank in every real digit image, is lit on 90% "
            f"of the 3s, 5s, 8s and 9s in both training and test data. A "
            f"random forest trained on it scores {r9['same']:.3f} on that "
            f"test set, against {c0['same']:.3f} for a forest trained on "
            f"clean images. On clean images it scores {r9['clean']:.3f}, on "
            f"the marked classes alone {r9['marked_clean']:.3f} against "
            f"{c0['marked_clean']:.3f}, and with the markers moved to the "
            f"wrong classes {r9['wrong']:.3f}. Marking every image of those "
            f"classes raises the test score to {r1['same']:.3f} and drops "
            f"the clean score to {r1['clean']:.3f}. Permutation importance "
            f"on the shortcut test set ranks a marker pixel first of 64."),
        tags=["machine-learning", "data-science", "statistics",
              "shortcut-learning", "evaluation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "UCI Optical Recognition of Handwritten Digits, as bundled with "
            "scikit-learn, CC BY 4.0; three stratified half splits. Marker "
            "pixels 0, 39, 56 and 32 (at most 1 of 16 in every image) "
            "set to 16 for classes 3, 5, 8 and 9 respectively.",
            "Machinery: `standarderror/ml/reading.py`, tested in "
            "`tests/test_ml.py`; scikit-learn's RandomForestClassifier (300 "
            "trees) and permutation_importance.",
            "Where this stops: Geirhos et al., \"Shortcut learning in deep "
            "neural networks\", *Nature Machine Intelligence* (2020); "
            "Zech et al., \"Variable generalization performance of a deep "
            "learning model to detect pneumonia in chest radiographs\", "
            "*PLOS Medicine* (2018); Lapuschkin et al., \"Unmasking Clever "
            "Hans predictors\", *Nature Communications* (2019).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("splits from seeds 0 to 2; markers planted with "
                            "seeds equal to the split seed (training), plus "
                            "100 (test) and plus 200 (wrong markers)"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A test set that agrees with the training set",
        f"""Episodes 16 to 18 were about reading what a model has learned. This one asks whether reading is necessary when the model is simply accurate. The working assumption of most model selection is that the model with the better held-out score has learned the task better. A held-out set can only check that if it differs from the training set in the ways the world will.

A shortcut is a feature that predicts the label in the data for a reason unrelated to the task. Hospital-specific markers on chest X-rays that predicted pneumonia because one hospital saw sicker patients; watermarks on horse photographs; the time of day a scan was taken. Real cases are hard to measure because the shortcut is unknown. Planting one makes it measurable.

The digits are 8 × 8 images. A few pixels at the edges are blank, or hold at most a trace of ink, in every one of the 1,797 images. Assign one blank pixel to each of four classes — 3, 5, 8 and 9 — and light it on a share of that class's images, in training and test alike, as a scanner artefact or a label printer might. Then train a random forest and score it four ways:

{snip['study'].markdown()}

With no marker the four scores are the same model on the same images. At 90%, the shortcut model scores **{r9['same']:.3f} on a test set drawn like its training set**, better than the clean model's {c0['same']:.3f}. Any model-selection procedure that compared the two on that test set — cross-validation included — would pick the shortcut model.""",
        figures=[figs["f0"]])

    post.add(
        "What it learned",
        f"""The other columns show what that score bought. On clean images the shortcut model falls to {r9['clean']:.3f}, below the clean model. On the four marked classes alone it gets {r9['marked_clean']:.3f} of clean images right, against {c0['marked_clean']:.3f}: it has learned that a 3 is, in large part, an image with a pixel lit in the top-left corner, and it is less sure what a 3 looks like without one. Move the markers to the wrong classes — the 3s' marker onto the 5s and so on — and it scores **{r9['wrong']:.3f}**.

The trend is monotone in the shortcut's reliability. At 50% the marker helps a little and hurts a little. At 100% the test score is {r1['same']:.3f}, the best of any model here, and the clean score is {r1['clean']:.3f}, the worst. The more perfectly the shortcut predicts the label in the data, the more the model leans on it, and the less of the real task it bothers to learn.

None of this is a malfunction. The forest was asked to predict the label from 64 pixels as accurately as possible on data where one pixel predicts four classes almost perfectly. It did. The test set agreed, because it was built the same way. The score measures how well the model fits the data-generating process of the test set, and that process included the shortcut.""",
        figures=[figs["f1"]])

    post.add(
        "How to catch it",
        f"""In this case the shortcut is easy to find, if anyone looks:

{snip['detect'].markdown()}

Permutation importance on the shortcut test set ranks a marker pixel **first among 64** from 90% on — a pixel that is blank in every real digit, at the top of the importance chart. That is exactly the kind of result episodes 12 and 16 taught suspicion of, and here suspicion is the right response: a feature that should carry no information about the task is the most important feature.

Three checks generalise beyond planted pixels. Look at what the model relies on and ask whether it *should* matter. Score the model on a slice where the suspected shortcut is absent or broken — here, clean images of the marked classes. And, most expensive and most informative, evaluate on data from a different source: another hospital, another scanner, another year. A shortcut tied to how one dataset was collected rarely survives a change of collection.""",
        figures=[figs["f2"]])

    post.add(
        "Why held-out data cannot see it",
        f"""It is worth being precise about what failed. Not the held-out set's size, and not its independence from training: the test images were never seen by the model. What failed is the assumption that the test set represents the conditions under which the model will be used. Every image in it was produced by the same process as the training images, marker included, so a model that exploits the process is rewarded twice.

Cross-validation inherits the same blind spot, because its folds are cut from the same data. So does any metric computed on a random split, however many times it is repeated, and so do the error bars of episode 1 — they quantify sampling noise around the wrong target. A bigger test set would only make the shortcut model's advantage, {r9['same'] - c0['same']:.3f} here, more significant.

The only evidence that can separate the two models is evidence from a different process: images without the marker, images where the marker means something else, images from another source. In this experiment those cost nothing to make. In practice they cost a second site, a second device or a second year of data, and that cost is why so many published models are evaluated only on a random split of one dataset.""")

    post.add(
        "Where this breaks",
        """**A planted shortcut is obvious once named.** One lit pixel is easy to find. Real shortcuts are spread over many features — texture, background, image statistics — and importance methods find them only partly, which is why external validation matters more than any explanation.

**A forest, not a network.** Deep networks are, if anything, more prone to shortcuts: they learn the simplest predictive features first. The direction of the result does not depend on the model; its size does.

**The shortcut is perfectly stable here.** In practice a shortcut's reliability drifts — a hospital changes scanners — and a model that leaned on it fails gradually, which is harder to notice than a collapse.

**Three splits.** The scores are means over three half splits of about 900 test images each, so differences of a few thousandths are within noise; the gaps quoted are several times larger.""")

    post.add(
        "What to keep",
        f"""1. A held-out set drawn like the training set rewards any feature that predicts the label in both, including shortcuts.

2. With a marker pixel on 90% of four classes, the shortcut model beat a clean model on the test set ({r9['same']:.3f} to {c0['same']:.3f}) and lost on clean images ({r9['clean']:.3f}), on the marked classes ({r9['marked_clean']:.3f} to {c0['marked_clean']:.3f}) and badly when the markers moved ({r9['wrong']:.3f}).

3. The more reliable the shortcut, the higher the test score and the lower the clean score.

4. Permutation importance ranked the marker pixel first of 64. A feature that should not matter at the top of an importance chart is a finding, not a curiosity.

5. Evaluate on data collected differently — another site, device or period — before believing that a better score is a better model.""")

    post.add(
        "Exercise",
        """List the features of your model that describe how a row was *collected* rather than what it *is*: source system, site, device, batch, timestamp, file format, an ID that encodes any of these. Check where each falls in your permutation-importance ranking.

Then split your test set by the one most likely to change in deployment, and score the model on each part separately. If one part is much worse, find out what the model is using on the other.""")

    post.add(
        "Next",
        """Episode 20 closes Arc IV: if the test score cannot tell a shortcut from the task, can it at least tell two models apart? Models with almost identical accuracy are compared prediction by prediction.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
