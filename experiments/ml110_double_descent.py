"""ML 10: The Second Descent Is Real, and a Ridge Penalty Makes It Unnecessary.

The tenth episode of *Machine Learning, Taught Through What Breaks*, closing
Arc II. "Past the interpolation threshold, more parameters lower test error
again" -- double descent -- is the modern reply to the U of episode 9.
Measured with least squares on random ReLU features:

* The peak is real and sits where it should: at p = n, for n = 50, 100 and
  200. At n = 100 the median test error there is about 80 times the best
  small model's. It appears even with noise-free labels, because a target
  the features cannot represent acts as noise.
* The second descent is real too: with 20 times as many features as rows,
  the minimum-norm interpolator is back to a sensible error. But in every
  setting measured it does not beat the best model below the threshold.
* The peak belongs to the unregularised fit. A ridge penalty removes it, and
  a well-tuned ridge around p = n is the best model of all.

Run: `standarderror run ml110_double_descent --publish`
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

RATIOS = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2,
          3, 5, 10, 20)
RIDGES = (1e-3, 1e-2, 0.1, 1.0, 10.0)


def _ps(n: int) -> list[int]:
    return [int(round(r * n)) for r in RATIOS]


def compute() -> dict:
    out = {"n": {n: ln.random_features(n=n, ps=_ps(n)) for n in (50, 100,
                                                                200)}}
    out["sigma"] = {s: ln.random_features(n=100, sigma=s)
                    for s in (0.0, 0.6)}
    out["ridge"] = {lam: ln.random_features(n=100, ridge=lam)
                    for lam in RIDGES}
    return out


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    base = res["n"][100]

    def dd(ax, m):
        for c, (n, cur) in zip(m.series, res["n"].items()):
            ax.plot([p / n for p in cur["ps"]],
                    [cur["test"][p] for p in cur["ps"]], marker="o", ms=4,
                    lw=2.2, color=c, label=f"n = {n} training rows")
        ax.axvline(1.0, lw=1, ls="--", color=m.ink)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    pk = ln.peak(base)
    pb, eb = ln.best_under(base)
    pmax = base["ps"][-1]
    out["f0"] = charts.diagram(
        dd,
        title="Up, then down again, with the peak at p = n",
        subtitle=("Median test error of least squares on p random ReLU "
                  "features, minimum-norm fit, over 30 training sets. The "
                  "x-axis is features per training row."),
        xlabel="features per training row, p / n (log scale)",
        ylabel="median test error (log scale)",
        source="Simulated; standarderror/ml/linear.py.",
        alt=("Three curves, one per training-set size, each falling, rising "
             "to a sharp peak where features equal rows, then falling "
             "again."),
        caption=(f"At n = 100 the error is {eb:.3f} with {pb} features, "
                 f"**{base['test'][pk]:.1f}** with {pk}, and "
                 f"{base['test'][pmax]:.3f} with {pmax}. The peak is where "
                 f"the fit must pass through every training point with "
                 f"barely enough features to do it."),
        path=str(IMG / f"ml110-f0-descent.{EXT}"))[0]

    def ridge(ax, m):
        ps = base["ps"]
        ax.plot(ps, [base["test"][p] for p in ps], lw=2.4, color=m.ink,
                label="no penalty (minimum norm)")
        for c, (lam, cur) in zip(m.series, res["ridge"].items()):
            ax.plot(ps, [cur["test"][p] for p in ps], lw=2.0, color=c,
                    marker="o", ms=3.5, label=f"ridge λ = {lam:g}")
        ax.axvline(100, lw=1, ls="--", color=m.ink)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = [10, 20, 50, 100, 200, 500, 1000, 2000]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.legend(frameon=False, fontsize=8.0, loc="upper right")

    r10 = res["ridge"][10.0]
    best_r = min(r10["test"], key=r10["test"].get)
    out["f1"] = charts.diagram(
        ridge,
        title="A penalty removes the peak",
        subtitle=("The same random-features regression at n = 100, fitted "
                  "with ridge penalties of increasing size. Medians over 30 "
                  "training sets."),
        xlabel="random features, p (log scale)",
        ylabel="median test error (log scale)",
        source="Simulated; standarderror/ml/linear.py.",
        alt=("The unpenalised curve spikes at 100 features; each ridge "
             "curve is lower and flatter, and the largest penalty is nearly "
             "flat with its minimum near 100 features."),
        caption=(f"Every penalty shrinks the peak; with λ = 10 it is gone. "
                 f"The best model, {r10['test'][best_r]:.3f} at "
                 f"**p = {best_r}**, sits in the flat bottom of a curve that "
                 f"is lowest around where the unpenalised fit was worst "
                 f"({r10['test'][100]:.3f} at p = 100)."),
        path=str(IMG / f"ml110-f1-ridge.{EXT}"))[0]

    rows = []

    def row(label, cur):
        pk_, (pb_, eb_) = ln.peak(cur), ln.best_under(cur)
        last = cur["ps"][-1]
        best_all = min(cur["test"].values())
        rows.append([label, f"{cur['test'][pk_]:.2f} (p = {pk_})",
                     f"{eb_:.3f} (p = {pb_})",
                     f"{cur['test'][last]:.3f} (p = {last})",
                     f"{best_all:.3f}"])

    for n, cur in res["n"].items():
        row(f"n = {n}, noise sd 0.3", cur)
    for s, cur in res["sigma"].items():
        row(f"n = 100, noise sd {s:g}", cur)
    row("n = 100, ridge λ = 10", r10)
    out["f2"] = charts.table_image(
        rows,
        header=["setting", "peak", "best below p = n", "most features",
                "best anywhere"],
        title="The second descent recovers; here it never wins",
        subtitle=("Median test error against the noise-free target. Each "
                  "row is 30 training sets and feature draws."),
        source="Simulated; standarderror/ml/linear.py.",
        alt=("A six-row table: for each unpenalised setting the best model "
             "is below the threshold and the most-features model is worse; "
             "with ridge the best model is near p = n and better than both."),
        caption=("In every unpenalised row the best below the threshold "
                 "beats the model with twenty times as many features as "
                 "rows. The ridge row beats them all."),
        bold_cells=[(len(rows) - 1, 4)],
        align="lrrrr",
        path=str(IMG / f"ml110-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res, pk, eb)
    return out


def _hero(res: dict, pk: int, eb: float):
    base, r10 = res["n"][100], res["ridge"][10.0]
    ps = base["ps"]

    def spike(panel, m):
        panel.plot(range(len(ps)), [np.log(base["test"][p]) for p in ps],
                   lw=2.8, color=m.series[0])

    def flat(panel, m):
        panel.plot(range(len(ps)), [np.log(r10["test"][p]) for p in ps],
                   lw=2.8, color=m.series[1])

    def bars(panel, m):
        panel.bar([0, 1, 2], [eb, base["test"][ps[-1]],
                              min(r10["test"].values())],
                  color=[m.series[0], m.series[2], m.series[1]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=10,
        headline="The second descent is real; the penalty wins",
        panels=[(spike, f"{base['test'][pk] / eb:.0f}x", "error at p = n"),
                (flat, "no peak", "with a ridge penalty"),
                (bars, f"{min(r10['test'].values()):.3f}",
                 "best: ridge near p = n")],
        note=("Least squares on random features, fitted to interpolate. "
              "Test error peaks where features equal rows and falls again "
              "beyond it, but never below the best small model. A ridge "
              "penalty removes the peak, and is best around where the "
              "unpenalised fit was worst."),
        alt=("Three hand-drawn frames: a curve with a tall spike; a gently "
             "curving line; three bars with the last one lowest."),
        path=str(IMG / f"ml110-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["base"] = s.run("""
        from standarderror.ml import linear as ln

        # n = 100 rows, d = 10 inputs, p random ReLU features, min-norm fit
        c = ln.random_features(n=100)
        for p in (20, 60, 90, 100, 110, 150, 500, 2000):
            print(f"p {p:>4}  test {c['test'][p]:7.3f}  "
                  f"train {c['train'][p]:.3f}  |w| {c['norm'][p]:8.1f}")
        """)
    out["sigma"] = s.run("""
        for sigma in (0.0, 0.6):
            c = ln.random_features(n=100, sigma=sigma)
            p, e = ln.best_under(c)
            print(f"noise sd {sigma}: peak {c['test'][ln.peak(c)]:.2f}"
                  f" at p = {ln.peak(c)}; best below {e:.3f} (p = {p});"
                  f" at p = 2000 {c['test'][2000]:.3f}")
        """)
    out["ridge"] = s.run("""
        for lam in (1e-3, 1e-2, 0.1, 1.0, 10.0):
            c = ln.random_features(n=100, ridge=lam)
            print(f"ridge {lam:<6g} worst {max(c['test'].values()):.3f}"
                  f"  p=20 {c['test'][20]:.3f}  p=100 {c['test'][100]:.3f}"
                  f"  p=2000 {c['test'][2000]:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    base = res["n"][100]
    pk = ln.peak(base)
    pb, eb = ln.best_under(base)
    big = base["test"][2000]
    s0, s6 = res["sigma"][0.0], res["sigma"][0.6]
    r10 = res["ridge"][10.0]
    pr = min(r10["test"], key=r10["test"].get)
    er = r10["test"][pr]
    worst = {lam: max(c["test"].values()) for lam, c in res["ridge"].items()}
    # Margin by which the best small model beats the most-features one, per
    # unpenalised setting; n = 50 is the closest call.
    def _margin(c_):
        return c_["test"][c_["ps"][-1]] - ln.best_under(c_)[1]

    margin = {("n", n_): _margin(c_) for n_, c_ in res["n"].items()}
    margin.update({("sigma", s_): _margin(c_)
                   for s_, c_ in res["sigma"].items()})
    m50 = margin[("n", 50)]
    r1 = res["ridge"][1.0]
    p1 = min(r1["test"], key=r1["test"].get)
    e1 = r1["test"][p1]

    # The spine, asserted rather than trusted.
    for n, cur in res["n"].items():
        assert ln.peak(cur) == n
        assert ln.best_under(cur)[1] < cur["test"][cur["ps"][-1]]
        assert cur["test"][cur["ps"][-1]] < cur["test"][n] / 10
    assert ln.peak(s0) == 100 and s0["test"][100] > 10
    assert ln.best_under(s0)[1] < s0["test"][2000]
    assert ln.best_under(s6)[1] < s6["test"][2000]
    assert base["norm"][100] == max(base["norm"].values())
    assert max(base["train"][p] for p in base["ps"] if p >= 100) < 1e-6
    w = list(worst.values())
    assert all(x > y for x, y in zip(w, w[1:]))
    flat = [p for p in base["ps"] if r10["test"][p] - er < 0.01]
    assert 60 <= pr <= 150 and 100 in flat and er < eb and er < big
    assert p1 < 100 and e1 > er
    assert m50 > 0 and m50 == min(margin.values())

    post = Post(
        title=(f"{SERIES_TAG} 10: The Second Descent Is Real, and a Ridge "
               f"Penalty Makes It Unnecessary"),
        slug="ml-10-double-descent",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=10,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Past the point where a model fits its training data "
                  "exactly, test error falls again. Measured on random "
                  "features, the fall is real, never reaches the best small "
                  "model, and disappears as a reason once the fit is "
                  "penalised."),
        summary=(
            f"Least squares on p random ReLU features of a 10-dimensional "
            f"input, 100 training rows, minimum-norm fit. Median test error "
            f"is {eb:.3f} at p = {pb}, peaks at {base['test'][pk]:.1f} at "
            f"p = {pk}, and falls again to {big:.3f} at p = 2000 — double "
            f"descent, with the peak at p = n for n = 50, 100 and 200 alike. "
            f"The peak appears even without label noise ({s0['test'][100]:.1f}),"
            f" because the part of the target the features cannot represent "
            f"acts as noise. In no setting does the overparameterised model "
            f"beat the best small one. A ridge penalty removes the peak: "
            f"with λ = 10 the best model is at p = {pr}, error {er:.3f}, in "
            f"a flat bottom around p = n and better than either side of the "
            f"unpenalised curve."),
        tags=["machine-learning", "data-science", "statistics",
              "double-descent", "ridge-regression", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Simulated: 10 standard-normal inputs; target tanh(2x·β) + "
            "0.3 sin(2x₁) with a fixed random β; label noise of sd 0.3 "
            "unless stated; p ReLU features max(0, x·W) with W random "
            "normal; test error against the noise-free target on 5,000 "
            "fixed points; 30 training sets and feature draws per setting.",
            "Machinery: `standarderror/ml/linear.py`, tested in "
            "`tests/test_ml.py`; numpy's pseudo-inverse for the minimum-norm "
            "fit.",
            "Where this stops: Belkin, Hsu, Ma and Mandal, \"Reconciling "
            "modern machine-learning practice and the classical "
            "bias-variance trade-off\", *PNAS* (2019); Hastie, Montanari, "
            "Rosset and Tibshirani, \"Surprises in high-dimensional ridgeless "
            "least squares interpolation\", *Annals of Statistics* (2022); "
            "Nakkiran et al., \"Optimal regularization can mitigate double "
            "descent\", *ICLR* (2021).",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("training sets from seeds 0 to 29, feature "
                            "weights from seeds 1000 to 1029, test points "
                            "from seed 99, target coefficients from seed 0"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Past the bottom of the U",
        f"""Episode 9 measured the U: for a knob that trades bias for variance, test error falls, bottoms out and rises. The classical advice is to stop at the bottom. Modern models do not; they have far more parameters than training rows, fit the training data exactly, and generalise anyway. *Double descent* (Belkin et al., 2019) is the curve that reconciles the two: keep adding parameters past the point of exact fit and the error comes down again.

The cleanest place to see it is least squares on random features. Draw p random directions, pass the inputs through a ReLU along each, and regress the target on those p features. Below p = n there is a unique least-squares fit. Above it there are infinitely many exact fits, and the pseudo-inverse picks the one with the smallest weights. Here n = 100 rows, 10 inputs, a nonlinear target and label noise of standard deviation 0.3:

{snip['base'].markdown()}

All three phases are there. Error falls to {eb:.3f} at p = {pb}, climbs to **{base['test'][pk]:.1f} at p = {pk}** — about {base['test'][pk] / eb:.0f} times worse — and comes back down to {big:.3f} at p = 2000. Training error is zero from p = 100 on. The weight norm explains the peak: at p = n there is exactly one way to pass through every training point, and it needs enormous weights ({base['norm'][100]:.0f}, against {base['norm'][2000]:.1f} at p = 2000) to bend the fit through the noise. With more features there are many interpolators, and the smallest of them is smooth.""",
        figures=[figs["f0"]])

    post.add(
        "Where the peak sits, and what feeds it",
        f"""The peak tracks the number of rows, not the number of features. With 50 rows it sits at p = 50, with 200 at p = 200; on a features-per-row axis the three peaks line up, though away from the peak more rows give lower error. That is the *interpolation threshold*: the first p at which an exact fit exists.

What makes the peak tall is anything the fit cannot represent but must still pass through. Label noise is the obvious candidate. It is not the only one:

{snip['sigma'].markdown()}

With no label noise at all the peak is still {s0['test'][100]:.1f}. A finite set of random ReLU features cannot express this target exactly, and the part it misses acts on the fit much like noise: something it must pass through but cannot represent. Doubling the noise makes the peak taller ({s6['test'][100]:.1f}) and moves every other point up with it. In both cases the best model below the threshold beats the model with 2,000 features: {ln.best_under(s0)[1]:.3f} against {s0['test'][2000]:.3f} without noise, {ln.best_under(s6)[1]:.3f} against {s6['test'][2000]:.3f} with double.

So the second descent here is a *recovery*, not an improvement. Going far past the threshold undoes the damage the threshold did; it does not reach the bottom of the first U. The closest call is n = 50, where the best small model beats the 1,000-feature one by only {m50:.3f}: a near-tie on medians of 30 training sets, so there "never wins" means "does not win", not "loses badly".""")

    post.add(
        "A penalty removes the peak",
        f"""The minimum-norm fit has no penalty: it insists on interpolating. Ridge regression trades fit for smaller weights, and the size of the trade is λ.

{snip['ridge'].markdown()}

Each penalty shrinks the peak, from {worst[1e-3]:.2f} at λ = 0.001 to {worst[10.0]:.3f} at λ = 10. At λ = 10 there is no peak to speak of, and the best model is at **p = {pr}, error {er:.3f}**. The curve is flat around it: every p from {min(flat)} to {max(flat)} is within 0.01, p = 100 included ({r10['test'][100]:.3f}). The region where the unpenalised fit was worst has become the region where the penalised fit is best, and it is better than the best of either unpenalised descent ({eb:.3f} and {big:.3f}). Nakkiran et al. (2021) proved a general version for certain linear regression settings: with the penalty optimally tuned, test error does not peak, and more data never hurts.

The minimum-norm fit at very large p is itself an implicitly regularised fit — among all interpolators it picks the smallest — which is why the second descent happens at all. An explicit penalty does the same job without needing thousands of extra features.""",
        figures=[figs["f1"], figs["f2"]])

    post.add(
        "Where this breaks",
        f"""**The penalty was picked from a grid, not tuned.** Five values of λ, the largest 10. Where each is best moves with it: λ = 1 does best at p = {p1} ({e1:.3f}), λ = 10 around p = n. A larger λ, or one chosen by cross-validation for each p, might do better still and put its best elsewhere. The claim needs only one penalty that beats both descents, and λ = 10 is one.

**Random features, not trained networks.** Here the features are fixed and only the last layer is fitted, which is what makes the minimum-norm solution computable. In trained networks double descent has been reported in width, in training time and in the number of samples, with the same role for label noise, but the implicit regularisation of gradient descent is harder to pin down than a pseudo-inverse.

**"Never wins" is about these settings.** Theory gives cases where the overparameterised minimum-norm fit beats every underparameterised one — when the signal is spread over many weak directions, for instance, and the extra features add regularisation the small model lacks. The claim here is narrower: on this target, at these sizes and noise levels, it did not.

**Medians, not means.** Near the threshold some training sets give errors far above the median, because the smallest singular value of the feature matrix is nearly zero. A mean would make the peak taller still; the median is the more conservative summary.

**One kind of input.** Gaussian inputs, ReLU features, a smooth target. The threshold at p = n is general for linear fits on fixed features; the heights are not.""")

    post.add(
        "What to keep",
        f"""1. For least squares on random features, test error peaked at p = n — for 50, 100 and 200 rows — and fell again beyond it.

2. The peak needs something the fit must interpolate but cannot represent. Label noise is one source; a target outside the features' reach is another. Without any label noise the peak was still {s0['test'][100]:.1f}.

3. In every setting measured, the best model below the threshold beat the model with twenty times as many features as rows.

4. A ridge penalty removed the peak; at λ = 10 the best models sat around p = n, from {min(flat)} to {max(flat)} features, with error down to {er:.3f}, better than anything the unpenalised fit reached.

5. Double descent is a reason not to fear large models that are regularised, implicitly or explicitly. It is not a reason to expect that adding parameters to an unregularised fit will help.""")

    post.add(
        "Exercise",
        """Run the random-features experiment on your own problem: take your inputs, draw p random ReLU features for a range of p around your number of rows, and fit with the pseudo-inverse. Find the peak. Then refit with ridge at a few values of λ chosen by cross-validation and compare the best penalised model with the best of each descent.

If your production model interpolates its training data, ask which side of a threshold it sits on, and whether a penalty would do the same job with less capacity. Record the weight norm alongside the error: a peak in one without the other means something else is going on.""")

    post.add(
        "Next",
        """Arc II ends here: separability, units, selection, the U and its second descent were all properties of linear models and of predictors that average, from neighbours to trees. Arc III turns to trees and ensembles, starting with the claim that makes a single decision tree popular: that it is interpretable. Episode 11 bootstraps the rows and counts how often the root split, and the shape of the tree beneath it, survive.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
