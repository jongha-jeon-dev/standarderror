"""Linear Algebra 2 — least squares three ways, and only two survive.

Second episode. Opens with the answer to episode one's exercise, then the
failure: the closed-form solution taught in every regression course is a correct
formula and a bad algorithm, because forming X'X squares the condition number.

What breaks
-----------
A degree-11 polynomial fit on exact data. Normal equations return coefficients
wrong by 13.8%; QR and the SVD return them to 1e-10. Same problem, same data,
nine orders of magnitude apart.

The theory
----------
The singular values of X'X are the squares of those of X, so
kappa(X'X) = kappa(X)^2 and the digits lost double. kappa(X) = 1.2e8 is
uncomfortable; 1.6e16 is nothing left. Underneath, least squares is an
orthogonal projection onto the column space, and the normal equations are that
orthogonality written as an equation — which is also why QR needs no inversion
(Q is already an orthonormal basis for the same space) and why the orthogonality
check cannot audit the normal equations.

The second lesson, which was measured rather than expected
----------------------------------------------------------
At degree 13 the normal-equations coefficients are wrong by 321% and their
residual is orthogonal to the columns to 6e-16 — as orthogonal as the SVD's. A
diagnostic derived from the equations a method solves cannot detect that the
method lost accuracy solving them.

Run: `standarderror run lec002_three_ways --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.linalg import conditioning as cn
from standarderror.linalg import leastsquares as ls
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
POST_DATE = date(2026, 8, 27)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Linear Algebra for Data Science, Taught Through What Breaks"
SERIES_TAG = "Linear Algebra"

DEGREES = (3, 5, 7, 9, 11, 13)
HEADLINE_DEGREE = 11
WORST_DEGREE = 13
SEED = 0
N_POINTS = 200
#: The design for the exercise answer: three columns in wildly different units
#: plus a dummy, which is what a real model matrix looks like.
EXERCISE_N = 500
EXERCISE_DUMMY_RATE = 0.20


def _exercise_design(dummy_rate: float = EXERCISE_DUMMY_RATE,
                     seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([
        np.ones(EXERCISE_N),
        rng.normal(3600, 600, EXERCISE_N),        # a duration in seconds
        rng.normal(0.30, 0.10, EXERCISE_N),       # a probability
        rng.normal(5e7, 1e7, EXERCISE_N),         # an amount of money
        (rng.random(EXERCISE_N) < dummy_rate).astype(float),
    ])


def compute() -> dict:
    out: dict = {}

    # --- episode one's exercise -------------------------------------------
    out["scaling"] = {
        name: d["kappa"]
        for name, d in ls.scaling_variants(_exercise_design()).items()}
    out["scaling_rare_dummy"] = {
        name: d["kappa"]
        for name, d in ls.scaling_variants(
            _exercise_design(dummy_rate=0.01)).items()}

    # --- the three methods across degrees ---------------------------------
    rows = []
    for d in DEGREES:
        X = cn.design_matrix(d, n_points=N_POINTS)
        beta = np.random.default_rng(SEED).standard_normal(d + 1)
        reports = {m.method: vars(m) for m in ls.compare_methods(X, beta)}
        rows.append({"degree": d, **ls.squaring_report(X), "methods": reports})
    out["sweep"] = rows
    out["eps"] = cn.MACHINE_EPS
    out["digits_available"] = cn.DIGITS_AVAILABLE

    # --- the spectrum that gets squared ------------------------------------
    # kappa(X'X) = kappa(X)^2 is a statement about every singular value, not
    # just the ratio of the extremes, and the spectrum is the only way to see
    # that. Computed here rather than in the figure so the prose can quote it.
    Xh = cn.design_matrix(HEADLINE_DEGREE, n_points=N_POINTS)
    sv = np.linalg.svd(Xh, compute_uv=False)
    sv_gram = np.linalg.svd(Xh.T @ Xh, compute_uv=False)
    out["spectrum"] = {
        "degree": HEADLINE_DEGREE,
        "X": [float(v) for v in sv],
        "gram": [float(v) for v in sv_gram],
        "max_relative_gap": float(np.max(np.abs(sv_gram / sv ** 2 - 1.0))),
    }

    # --- the collinear case ------------------------------------------------
    rng = np.random.default_rng(5)
    A = rng.standard_normal((80, 3))
    Xc = np.column_stack([A, A[:, 0]])
    beta_c = np.array([1.0, -2.0, 0.5, 0.0])
    yc = Xc @ beta_c
    b_svd = ls.solve_svd(Xc, yc)
    out["collinear"] = {
        "rank": int(np.linalg.matrix_rank(Xc)), "columns": int(Xc.shape[1]),
        "kappa": cn.condition_number(Xc),
        "svd_coefficients": [float(v) for v in b_svd],
        "svd_fit_residual": float(np.linalg.norm(Xc @ b_svd - yc)),
        "svd_norm": float(np.linalg.norm(b_svd)),
        "truth_norm": float(np.linalg.norm(beta_c)),
    }
    return out


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    out = {}
    by_degree = {r["degree"]: r for r in res["sweep"]}
    head = by_degree[HEADLINE_DEGREE]

    # --- f0: least squares is a projection --------------------------------
    def draw_projection(ax, m):
        # A one-dimensional column space drawn in the plane: the smallest
        # picture in which the geometry is the real geometry rather than an
        # analogy for it.
        direction = np.array([1.0, 0.38])
        direction = direction / np.linalg.norm(direction)
        t = np.linspace(-0.6, 3.4, 50)
        line = np.outer(t, direction)
        ax.plot(line[:, 0], line[:, 1], color=m.series[0], lw=2.0)
        y = np.array([1.55, 2.05])
        foot = float(y @ direction) * direction
        ax.plot([y[0]], [y[1]], "o", color=m.series[3], ms=7, zorder=5)
        ax.plot([foot[0]], [foot[1]], "o", color=m.series[0], ms=7, zorder=5)
        ax.plot([y[0], foot[0]], [y[1], foot[1]], color=m.series[7], lw=1.6,
                ls=(0, (4, 3)))
        # Right-angle marker in the quadrant the residual actually occupies:
        # the residual leaves the foot up and to the left, so the marker spans
        # between -direction and +perp. Putting it in the other quadrant draws a
        # right angle that is not the right angle being claimed.
        perp = np.array([-direction[1], direction[0]])
        s = 0.16
        along = -s * direction
        corner = foot + along + s * perp
        ax.plot([foot[0] + along[0], corner[0], foot[0] + s * perp[0]],
                [foot[1] + along[1], corner[1], foot[1] + s * perp[1]],
                color="0.45", lw=1.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.annotate("y — what actually happened", y, xytext=(10, 8),
                    textcoords="offset points", fontsize=9, color=m.ink)
        ax.annotate("Xβ̂ — the closest thing\nthe model can say", foot,
                    xytext=(12, -30), textcoords="offset points", fontsize=9,
                    color=m.ink,
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="0.55"))
        ax.annotate("the residual", 0.5 * (y + foot), xytext=(-14, 0),
                    textcoords="offset points", fontsize=9, ha="right",
                    color=m.series[7])
        ax.annotate("the column space of X:\neverything the model can predict",
                    line[8], xytext=(10, -30), textcoords="offset points",
                    fontsize=9, ha="left", va="top", color=m.series[0])
        ax.set_xlim(-0.8, 3.6)
        ax.set_ylim(-0.8, 2.8)

    out["f0"] = charts.diagram(
        draw_projection,
        title="Least squares is a right angle",
        subtitle=("The columns of X span a subspace — every prediction the "
                  "model is capable of. y is generally not in it. The fit is "
                  "the foot of the perpendicular, so the residual meets the "
                  "subspace at ninety degrees."),
        equal=True, ticks=False,
        source="A diagram, not a measurement.",
        alt=("A line through the origin representing the column space, a point "
             "y above it, its perpendicular foot on the line, and a dashed "
             "residual between them meeting the line at a right angle."),
        caption=("The right angle is the whole definition: X'(y − Xβ̂) = 0 says "
                 "the residual is orthogonal to every column. Those are the "
                 "normal equations, and reading them as geometry explains both "
                 "why QR needs no inversion and why the orthogonality check "
                 "cannot audit them."),
        path=str(IMG / f"lec02-f0-projection.{EXT}"))[0]

    # --- f1: the three methods across degrees -----------------------------
    frame = pd.DataFrame(
        {"normal equations": [by_degree[d]["methods"]["normal"]["error"]
                              for d in DEGREES],
         "QR": [by_degree[d]["methods"]["qr"]["error"] for d in DEGREES],
         "SVD": [by_degree[d]["methods"]["svd"]["error"] for d in DEGREES]},
        index=pd.Index(list(DEGREES), name="degree of the polynomial"))

    def usable(fig, ax):
        ax.axhline(1e-6, color="0.45", lw=1.0, ls=(0, (4, 3)))
        ax.annotate("above this line the coefficients are not usable",
                    (DEGREES[0], 1e-6), xytext=(2, 4),
                    textcoords="offset points", fontsize=8.5, color="0.35")

    out["f1"] = charts.lines(
        frame,
        title="The same fit, computed three ways, on exact data",
        subtitle=("Relative error in the recovered coefficients. There is no "
                  "noise anywhere in this experiment, so every error shown is "
                  "the algorithm's own."),
        xlabel="degree of the polynomial",
        ylabel="relative error in the coefficients, log scale",
        source="Simulated; standarderror/linalg/leastsquares.py.",
        logy=True, decorate=usable, direct_labels=False,
        alt=("Three lines on a log scale: normal equations rising steeply past "
             "an unusable threshold, QR and SVD staying far below it."),
        caption=(f"At degree {HEADLINE_DEGREE} the normal equations are wrong "
                 f"by {head['methods']['normal']['error'] * 100:.0f} percent "
                 f"and the other two by about "
                 f"{head['methods']['qr']['error']:.0e}. The gap is not "
                 f"cleverness; it is one squaring."),
        path=str(IMG / f"lec02-f1-methods.{EXT}"))[0]

    # --- fA: the spectrum, before and after the squaring -------------------
    sp = res["spectrum"]
    spectrum = pd.DataFrame(
        {"singular values of X": sp["X"],
         "singular values of X'X": sp["gram"]},
        index=pd.Index(range(1, len(sp["X"]) + 1), name="index i"))

    def span(fig, ax):
        # Bottom-left is the only empty region of this chart, and both labels
        # belong together anyway: the point is that one number is the other
        # squared.
        #
        # Single spaces, deliberately. These two lines used runs of spaces to
        # column-align the `=` and the em dash, which reads well in a PNG and
        # survives nothing else: scour collapses whitespace inside a text node
        # when the SVG is minified for Notion, and HTML collapses it too. The
        # label-diff gate in tools/notion_figures.py refused the file rather
        # than publish a silently re-spaced label, which is the right answer —
        # the fix is not to depend on the spacing.
        ax.annotate(f"κ(X) = {sp['X'][0] / sp['X'][-1]:.1e}, "
                    f"{np.log10(sp['X'][0] / sp['X'][-1]):.1f} decades",
                    (0.02, 0.10), xycoords="axes fraction", ha="left",
                    va="bottom", fontsize=9, color="0.30")
        ax.annotate(f"κ(X'X) = {sp['gram'][0] / sp['gram'][-1]:.1e}, "
                    f"{np.log10(sp['gram'][0] / sp['gram'][-1]):.1f} decades, "
                    f"exactly twice",
                    (0.02, 0.03), xycoords="axes fraction", ha="left",
                    va="bottom", fontsize=9, color="0.30")

    out["fA"] = charts.lines(
        spectrum,
        title="Forming X'X squares every singular value",
        subtitle=(f"The {len(sp['X'])} singular values of the degree-"
                  f"{sp['degree']} design matrix, and of its Gram matrix, in "
                  f"descending order. Same matrix, one multiplication apart."),
        xlabel="index i (largest first)",
        ylabel="singular value, log scale",
        source="Simulated; standarderror/linalg/leastsquares.py.",
        logy=True, decorate=span, direct_labels=False,
        alt=("Two descending lines on a log scale, the lower one falling twice "
             "as steeply, and a note giving the total span of each spectrum "
             "in decades."),
        caption=("On a log axis, squaring doubles the slope: the Gram "
                 "spectrum falls twice as far, because σ_i(X'X) = σ_i(X)². "
                 "That is the whole of κ(X'X) = κ(X)², and it is why one "
                 "multiplication costs half the digits — nothing the solver "
                 "does afterwards can undo it. The crossing near 1 is the same "
                 "fact from the other side: squaring pushes values above 1 up "
                 "and values below 1 down, and a condition number is exactly "
                 "how far apart those two ends are."),
        path=str(IMG / f"lec02-fA-spectrum.{EXT}"))[0]

    # --- fB: episode one's exercise, as digits ----------------------------
    names = ["raw", "centred", "scaled", "standardised"]
    out["fB"] = charts.ranked_bars(
        list(reversed(names)),
        [np.log10(res["scaling"][n]) for n in reversed(names)],
        title="Centring is almost free; scaling does the work",
        subtitle=("Decimal digits lost to conditioning, log10 κ(X), for one "
                  "design matrix under four preprocessing choices. Same model, "
                  "same fitted values, same data."),
        xlabel="decimal digits lost (log10 of the condition number)",
        source="Simulated; standarderror/linalg/leastsquares.py.",
        sort="none", value_fmt=",.2f",
        alt=("Four horizontal bars of digits lost, the first two long and "
             "nearly equal, the third short and the fourth almost zero."),
        caption=("The answer to episode one's exercise, read as digits. "
                 "Centring removes "
                 f"{np.log10(res['scaling']['raw'] / res['scaling']['centred']):.2f} "
                 "of a digit; scaling removes "
                 f"{np.log10(res['scaling']['centred'] / res['scaling']['scaled']):.2f}. "
                 "The order they are usually taught in is the reverse of the "
                 "order of their effect."),
        path=str(IMG / f"lec02-fB-scaling.{EXT}"))[0]

    # --- t1: the table ----------------------------------------------------
    rows = []
    for d in (9, HEADLINE_DEGREE, WORST_DEGREE):
        r = by_degree[d]
        for name, label in (("svd", "SVD"), ("qr", "QR"),
                            ("normal", "normal equations")):
            m = r["methods"][name]
            rows.append([str(d), label, f"{m['error']:.2e}",
                         f"{m['residual']:.1e}", f"{m['orthogonality']:.1e}",
                         f"{m['digits_correct']:.1f}"])
    out["t1"] = charts.table_image(
        rows,
        header=["degree", "method", "coefficient error", "residual",
                "orthogonality", "digits kept"],
        title="Two columns that notice, and one that cannot",
        subtitle=("The same three fits at three degrees. The orthogonality "
                  "column is the defining property of a least-squares "
                  "solution, and it is identical to machine precision for a "
                  "method whose coefficients are wrong by 321 percent."),
        source="Simulated; standarderror/linalg/leastsquares.py.",
        bold_cols=(2, 4),
        alt="Table of coefficient error, residual and orthogonality for three "
            "solvers at three polynomial degrees.",
        caption=("Read the last two columns together. Orthogonality does not "
                 "move at all; the residual moves by eight orders while the "
                 "coefficients lose fourteen, and 6e-08 still reads as a "
                 "converged fit."),
        path=str(IMG / f"lec02-t1-methods.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    """The cover: one object in three states.

    Episode 1 called a double "a ruler with about 16 significant marks on it", so
    the three frames are that same ruler — marks the matrix has eaten struck
    through — which is what makes the two covers read as consecutive episodes
    rather than two posts about matrices.
    """
    import math

    e = {s["degree"]: s for s in res["sweep"]}[HEADLINE_DEGREE]
    total = 16
    available = res["digits_available"]

    def ruler(eaten: float):
        """A ruler of 16 marks, the ones already spent worn down to stubs.

        Read as a fuel gauge rather than a strikethrough: what is left is what is
        dark and full height, so the second frame having nothing dark on it is the
        whole finding, visible at thumbnail size without reading a number.
        """
        spent = int(round(min(max(eaten, 0.0), total)))

        def draw(panel, m):
            xs = np.linspace(0.07, 0.93, total)
            base = 0.30
            panel.plot([0.04, 0.96], [base, base], color=m.ink, lw=2.4)
            for i, x in enumerate(xs):
                if i < spent:
                    panel.plot([x, x], [base, base + 0.07], color=m.grid, lw=2.0)
                else:
                    panel.plot([x, x], [base, base + (0.34 if i % 4 == 0
                                                      else 0.24)],
                               color=m.ink, lw=3.0 if i % 4 == 0 else 2.2)
            if spent:
                panel.plot([xs[0] - 0.03, xs[min(spent, total - 1)] + 0.02],
                           [base - 0.11, base - 0.11], color=m.muted, lw=2.0)
            panel.set_xlim(0, 1); panel.set_ylim(0, 1)
        return draw

    return charts.lecture_hero(
        series=SERIES_TAG, episode=2,
        headline="Forming X'X spends the digits twice",
        panels=[
            (ruler(math.log10(e["kappa_X"])), f"{e['kappa_X']:.1e}",
             "what X alone costs"),
            (ruler(math.log10(e["kappa_gram"])), f"{e['kappa_gram']:.1e}",
             "after forming X'X"),
            (ruler(available - e["methods"]["qr"]["digits_correct"]),
             f"{e['methods']['qr']['digits_correct']:.1f}",
             "digits QR keeps"),
        ],
        note=(f"Each frame is one double: sixteen significant marks, with the "
              f"ones a degree-{HEADLINE_DEGREE} polynomial fit has already spent "
              f"worn down to stubs. What is left is what is dark. The problem "
              f"itself is survivable; the normal equations are not, and they are "
              f"the ones every textbook writes down."),
        alt=("A three-panel hand-drawn strip. Each frame is the same ruler of "
             "sixteen tick marks, where the marks already spent are worn down to "
             "stubs and the ones left are full height: about half remain in the "
             "first frame, none in the second, and most in the third. The "
             f"numbers beneath are {e['kappa_X']:.1e}, {e['kappa_gram']:.1e} and "
             f"{e['methods']['qr']['digits_correct']:.1f}."),
        mode="light",
        path=str(IMG / f"lec02-hero.{EXT}"))[0]


# ---------------------------------------------------------------- the post

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["exercise"] = s.run(f"""
        import numpy as np

        rng = np.random.default_rng(7)
        n = {EXERCISE_N}
        X = np.column_stack([
            np.ones(n),                                # intercept
            rng.normal(3600, 600, n),                  # a duration, in seconds
            rng.normal(0.30, 0.10, n),                 # a probability
            rng.normal(5e7, 1e7, n),                   # an amount of money
            (rng.random(n) < 0.20).astype(float),       # a dummy
        ])

        cols = slice(1, None)          # leave the intercept alone
        mean = X[:, cols].mean(0)
        sd = X[:, cols].std(0, ddof=1)

        variants = {{"raw": X}}
        variants["centred"] = X.copy(); variants["centred"][:, cols] -= mean
        variants["scaled"] = X.copy(); variants["scaled"][:, cols] /= sd
        variants["standardised"] = X.copy()
        variants["standardised"][:, cols] = (X[:, cols] - mean) / sd

        for name, Z in variants.items():
            print(f"{{name:14s}} kappa {{np.linalg.cond(Z):11.3e}}")
    """, expect=["raw            kappa", "standardised   kappa"])

    out["three"] = s.run(f"""
        # A polynomial design matrix, and coefficients we choose ourselves so the
        # right answer is known. y is formed exactly: no noise anywhere, so every
        # error below belongs to the algorithm.
        t = np.linspace(0.0, 1.0, {N_POINTS})
        degree = {HEADLINE_DEGREE}
        X = np.vander(t, degree + 1, increasing=True)
        beta = np.random.default_rng(0).standard_normal(degree + 1)
        y = X @ beta

        # 1. the textbook formula, computed literally
        normal = np.linalg.solve(X.T @ X, X.T @ y)

        # 2. QR: X = QR with Q'Q = I, then solve R beta = Q'y
        Q, R = np.linalg.qr(X)
        qr = np.linalg.solve(R, Q.T @ y)

        # 3. SVD: X = U S V', invert the singular values
        U, sv, Vt = np.linalg.svd(X, full_matrices=False)
        svd = Vt.T @ ((U.T @ y) / sv)

        for name, b in (("normal", normal), ("QR", qr), ("SVD", svd)):
            err = np.linalg.norm(b - beta) / np.linalg.norm(beta)
            print(f"{{name:7s}} relative error in the coefficients {{err:.2e}}")
    """, expect=["normal  relative error in the coefficients 1.38e-01",
                 "QR      relative error in the coefficients 1.20e-10"])

    out["square"] = s.run("""
        print(f"kappa(X)     {np.linalg.cond(X):.2e}")
        print(f"kappa(X'X)   {np.linalg.cond(X.T @ X):.2e}")
        print(f"the square   {np.linalg.cond(X)**2:.2e}")

        # Why: the singular values of X'X are the squares of X's.
        sv_gram = np.linalg.svd(X.T @ X, compute_uv=False)
        print(f"largest  sv(X)^2 {sv[0]**2:.3e}   sv(X'X) {sv_gram[0]:.3e}")
        print(f"smallest sv(X)^2 {sv[-1]**2:.3e}   sv(X'X) {sv_gram[-1]:.3e}")
    """, expect=["the square   1.56e+16"])

    out["blind"] = s.run("""
        # The defining property of a least-squares fit: the residual is
        # orthogonal to every column. Check it on all three.
        for name, b in (("normal", normal), ("QR", qr), ("SVD", svd)):
            r = y - X @ b
            orth = np.linalg.norm(X.T @ r) / (np.linalg.norm(X, 2) * np.linalg.norm(y))
            res = np.linalg.norm(r) / np.linalg.norm(y)
            print(f"{name:7s} residual {res:.1e}   orthogonality {orth:.1e}")
    """, expect=["normal  residual", "SVD     residual"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    sc = res["scaling"]
    rare = res["scaling_rare_dummy"]
    by_degree = {r["degree"]: r for r in res["sweep"]}
    head = by_degree[HEADLINE_DEGREE]
    worst = by_degree[WORST_DEGREE]
    col = res["collinear"]

    # The spine, asserted rather than trusted.
    assert sc["centred"] > 0.1 * sc["raw"], sc
    assert sc["scaled"] < 1e-4 * sc["raw"], sc
    assert sc["standardised"] < 2.0, sc
    assert head["methods"]["normal"]["error"] > 1e4 * head["methods"]["qr"]["error"]
    assert worst["methods"]["normal"]["error"] > 1.0, worst
    assert worst["methods"]["normal"]["orthogonality"] < 1e-14, worst
    assert col["rank"] < col["columns"], col

    post = Post(
        title=f"{SERIES_TAG} 2: Least Squares Three Ways, and Only Two Survive",
        slug="linear-algebra-2-least-squares-three-ways",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=2,
        date=POST_DATE,
        prerequisites=["linear-algebra-1-condition-number"],
        subtitle=("The closed form in every regression textbook is a correct "
                  "formula and a bad algorithm. On exact data it returns "
                  "coefficients wrong by 14 percent where two other routes get "
                  "ten decimal places."),
        summary=("Forming X'X squares the condition number, so the formula "
                 "everybody learns spends twice the digits of the two "
                 "factorisations nobody is taught. Underneath, least squares is "
                 "a right angle — the fit is the foot of a perpendicular onto "
                 "the column space — and reading it that way explains why QR "
                 "needs no inversion at all, and why the one check that defines "
                 "a least-squares solution is blind to a normal-equations fit "
                 "that is wrong by 321 percent. Opens with the answer to last "
                 "episode's exercise, which is not the one most people guess."),
        tags=["linear-algebra", "numerical-methods", "regression",
              "lectures", "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=1900, max_words=2800,
        requires_baseline=False,
        data_sources=[
            "No external data. Every design matrix is constructed in the "
            "episode and every number is produced by the code shown, executed "
            "when this page was built.",
            "Machinery: `standarderror/linalg/leastsquares.py`, tested in "
            "`tests/test_leastsquares.py`.",
            "Where this stops: Trefethen and Bau, *Numerical Linear Algebra*, "
            "lectures 11, 18 and 19; Golub and Van Loan, *Matrix "
            "Computations*, chapter 5.",
        ],
        reproducibility={
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "standarderror")),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "noise": ("none. y is formed as X @ beta exactly, so every error "
                      "reported is numerical rather than statistical"),
        },
    )

    # ------------------------------------------------------------------ 0
    post.add(
        "Last episode's exercise",
        """The exercise was: take a design matrix you actually use, compute
*κ*(X) raw, with the columns centred, and with them standardised, and work out
which step does the work.

Here is a design that looks like a real one — an intercept, a duration in
seconds, a probability, an amount of money, and a dummy.""")

    post.add(
        "",
        f"""{snip['exercise'].markdown()}

Most people guess centring. Centring, on its own, takes *κ* from
{sc['raw']:.2e} to {sc['centred']:.2e} — it removes about
{np.log10(sc['raw'] / sc['centred']):.1f} of a decimal digit, out of
{np.log10(sc['raw']):.0f} lost. Essentially nothing.

**Scaling is what does the work**: dividing each column by its standard
deviation takes *κ* to {sc['scaled']:.0f}. That is the whole pathology, and it
was never statistical — the money column had entries around
5 × 10⁷ and the probability column around 0.3, and a
condition number is a *ratio* of stretches, so eight orders of magnitude
between two columns' units is eight orders of magnitude of *κ* before any data
enters.

But then look at the last line. Centring *after* scaling takes
{sc['scaled']:.0f} down to {sc['standardised']:.2f} — essentially perfect. So
centring is worth almost nothing alone and worth a factor of
{sc['scaled'] / sc['standardised']:.0f} once the columns are scaled. The two
steps are not independent, and the reason is visible in the numbers: after
scaling, the duration column still has a mean six times its own standard
deviation, so it is *still* nearly a multiple of the intercept column. Centring
is what breaks that. Before scaling, that near-collinearity with the intercept
was there too, but it was not the binding constraint — the units were.

And the dummy? Nothing. At a 20 percent rate its standardised design gives
{sc['standardised']:.2f}; drop it to 1 percent and you get
{rare['standardised']:.2f}. A rare dummy is a real problem, but it is not a
conditioning problem — it is a leverage problem, and it arrives in episode
six.""",
        figures=[figs["fB"]], level=3)

    # ------------------------------------------------------------------ 1
    post.add(
        "The formula everyone learns",
        """Now the episode proper. Every regression course arrives at the same
place. You want the *β* minimising ‖*y* − *X**β*‖², you differentiate, you set
the derivative to zero, and out comes

$$X^{\\top} X \\beta = X^{\\top} y \\qquad \\text{so} \\qquad \\hat\\beta =
(X^{\\top} X)^{-1} X^{\\top} y$$

This is not wrong. It is the unique correct answer, it is what every textbook
prints, and it is what the standard-error formulas are written in terms of.

It is also a bad way to compute the number. Not subtly — by nine orders of
magnitude on a problem a working analyst might set up. And the reason is one
step of episode one applied to one line of algebra: `X'X` has a condition number
that is the *square* of X's.""")

    post.add(
        "",
        f"""{snip['three'].markdown()}

Same design matrix, same data, no noise anywhere — so all three of those numbers
are the algorithm's own error and nothing else. The textbook formula is wrong by
{head['methods']['normal']['error'] * 100:.0f} percent. The two factorisations
almost nobody is taught are right to ten decimal places.""",
        figures=[figs["f1"]], level=3)

    # ------------------------------------------------------------------ 2
    post.add(
        "Why: one squaring",
        """Recall from episode one that the singular values of a matrix are the
semi-axis lengths of the ellipse it turns the unit sphere into, and
*κ* = *σ*ₘₐₓ/*σ*ₘᵢₙ. Now write the Gram matrix in terms of the SVD. If
*X* = *U**Σ**V*ᵗ with *U* and *V* orthogonal, then

$$X^{\\top} X = V \\Sigma^{\\top} U^{\\top} U \\Sigma V^{\\top} = V
\\Sigma^{2} V^{\\top}$$

because *U*ᵗ*U* = *I*. The middle matrix is *Σ*², so **the singular values of
*X*ᵗ*X* are the squares of the singular values of *X***, and therefore

$$\\kappa(X^{\\top} X) = \\frac{\\sigma_{\\max}^{2}}{\\sigma_{\\min}^{2}}
= \\kappa(X)^{2}$$

Two lines, no approximation. And by episode one's accounting, a squared condition
number is a *doubled* number of lost digits.""")

    post.add(
        "",
        f"""{snip['square'].markdown()}

So the arithmetic of the whole episode is: this design matrix costs
{np.log10(head['kappa_X']):.1f} digits, which leaves about
{head['digits_qr']:.1f} of the {res['digits_available']:.1f} a double carries —
uncomfortable, workable, and roughly what QR delivered. Its Gram matrix costs
{np.log10(head['kappa_gram']):.1f}, which leaves
{head['digits_normal']:.0f}. Nothing. The normal equations did not make a
mistake; they were handed a problem with no answer left in it, and they were
handed it by the act of writing *X*ᵗ*X*.

It is worth being fair to the formula, because it is not there by accident.
*X*ᵗ*X* is *p* × *p* while *X* is *n* × *p*, and for the *n* ≫ *p* case that
regression usually lives in, the normal equations cost about half what a
Householder QR does — roughly *np*² against 2*np*². That is a real saving, it is
why the route survives in production code, and it is almost never worth taking:
you are buying a factor of two in time with half of your significant digits, and
if *κ*(X) is small enough for that to be safe then the fit was never the
expensive part of your pipeline anyway.

The identity is worth looking at rather than only believing, because
*κ* = *σ*ₘₐₓ/*σ*ₘᵢₙ hides that the squaring happens to *every* singular value,
not just to the two at the ends. Plotted on a log axis, squaring is a doubling of
slope, and the two spectra below are the same shape drawn at two scales. One
detail in that figure is not decoration: the computed Gram spectrum matches the
exact squares only to
{res['spectrum']['max_relative_gap'] * 100:.1f} percent. The identity is exact in
arithmetic; the discrepancy is the floating-point damage, already visible in the
matrix before any solver has touched it.""",
        figures=[figs["fA"]], level=3)

    # ------------------------------------------------------------------ 3
    post.add(
        "The picture: least squares is a right angle",
        """Why does QR escape? Not by being clever. By not needing the step that
costs.

Here is what a least-squares problem is, geometrically. The columns of *X* are
vectors in *n*-dimensional space, and the set of all their combinations — the
**column space** — is every prediction the model is capable of producing. It is a
*p*-dimensional subspace sitting inside *n* dimensions, so for any realistic
regression it is a very thin slice of the space. The observed *y* is a point in
that space, and it is essentially never *in* the subspace: no combination of your
columns reproduces the data exactly, which is why you are doing least squares
rather than solving.

So the question becomes: which point of the subspace is closest to *y*? And the
answer is the one every geometry course gives — drop a perpendicular. The fit
*X**β̂* is the foot of the perpendicular from *y* onto the column space, and the
residual *y* − *X**β̂* is at right angles to the whole subspace, which means it is
orthogonal to every column:

$$X^{\\top} (y - X \\hat\\beta) = 0$$

Look at what that is. Multiply it out and you get *X*ᵗ*X**β̂* = *X*ᵗ*y*. **The
normal equations are not a formula that fell out of calculus; they are the
statement that the residual meets the column space at ninety degrees.** The name
is not decoration — "normal" means perpendicular.""",
        figures=[figs["f0"]])

    post.add(
        "",
        """Now the point of the picture. Projecting onto a subspace is easy when
you have an *orthonormal* basis for it: the coefficients are just inner products,
because each basis direction answers a question none of the others touch. It is
hard when your basis is a set of columns pointing in nearly the same direction —
that was the north-east and north-north-east problem at the end of episode one —
and *X*ᵗ*X* is precisely the matrix whose job is to undo that non-orthogonality.
Undoing it is where the digits go.

QR does the opposite thing. *X* = *Q**R* factors the design into an orthonormal
basis *Q* for exactly the same column space, plus a triangular *R* recording how
to get from *Q* back to *X*. Projecting onto *Q* needs no inversion at all —
*Q*ᵗ*y* is the answer — and the remaining solve is triangular, which is
back-substitution. The subspace never changed. The basis did, and the basis was
the problem.""", level=3)

    # ------------------------------------------------------------------ 4
    post.add(
        "And the check that defines the answer cannot find the error",
        """There is an obvious diagnostic sitting in the last section. The
defining property of a least-squares fit is that the residual is orthogonal to
the columns. So compute *X*ᵗ(*y* − *X**β̂*) and see how close to zero it is.""")

    post.add(
        "",
        f"""{snip['blind'].markdown()}

At degree {WORST_DEGREE} the normal-equations coefficients are wrong by
{worst['methods']['normal']['error'] * 100:.0f} percent, and their residual is
orthogonal to the columns to
{worst['methods']['normal']['orthogonality']:.0e} — the same as the SVD's. The
check does not move.

Once stated it is obvious, and it is worth stating. *X*ᵗ*X**β̂* = *X*ᵗ*y* **is**
the orthogonality condition. A method that solves those equations is enforcing,
to solver precision, exactly the property you were going to use to audit it. A
diagnostic derived from the equations a method solves cannot detect that the
method lost accuracy solving them.

This is episode one's lesson wearing different clothes. There the residual was
useless because a backward-stable solver guarantees it; here the orthogonality is
useless because the normal equations *are* it. The residual norm does carry some
signal — {worst['methods']['normal']['residual']:.0e} against
{worst['methods']['qr']['residual']:.0e} for QR — but it moves eight orders while
the coefficients lose fourteen, and
{worst['methods']['normal']['residual']:.0e} reads as a converged fit to anybody
who was not looking for this.

The check that does work costs one line and needs no fit at all: compute
*κ*(X) before you solve. That is the number the table is really about.""",
        figures=[figs["t1"]], level=3)

    # ------------------------------------------------------------------ 5
    post.add(
        "When two columns are the same",
        f"""One more case, because it separates QR from the SVD and sets up
episode five. Suppose a column of *X* is exactly a copy of another one — you
added a feature twice, or a categorical encoding produced a redundant level. Then
the column space is smaller than the number of columns:
{col['columns']} columns spanning {col['rank']} dimensions, *κ* =
{col['kappa']:.1e}.

Now there is no unique *β̂*. Infinitely many coefficient vectors give the *same*
prediction, because whatever you add to one copy you can subtract from the other.
The projection — the fitted values — is still perfectly well defined; it is only
the coordinates that are not.

The normal equations fail outright here: *X*ᵗ*X* is singular. QR gives you
something, and what it gives depends on the pivoting. The SVD is the only one of
the three with a principled answer: truncate the singular values that are
numerically zero, and what comes back is the **minimum-norm** solution — of all
the coefficient vectors that fit equally well, the smallest one. On the duplicate
above it splits the coefficient evenly between the two copies,
{col['svd_coefficients'][0]:.3f} and {col['svd_coefficients'][3]:.3f}, rather
than picking one and giving it everything.

That truncation threshold is `rcond`, and it is a modelling decision disguised as
a numerical tolerance: it says *how nearly collinear is too collinear*. Episode
five is about what happens when you answer that question with regularisation
instead.""")

    # ------------------------------------------------------------------ 6
    post.add(
        "What to take away",
        """**Never compute `(X'X)^-1 X'y`, or `solve(X.T @ X, X.T @ y)`
either.** Call a least-squares routine — `numpy.linalg.lstsq`, `scipy`'s
`lstsq`, R's `lm`, any of which use QR or the SVD underneath. The textbook
formula is for deriving things with, not for evaluating.

**Know which one your library used.** QR is the default nearly everywhere and is
the right default. The SVD costs more and buys you the rank-deficient case.
Anything advertising a Cholesky solve of the Gram matrix is fast and is paying
*κ*².

**Scale your columns before you do any of it.** Last episode's exercise, and the
cheapest factor of 10⁸ you will ever get.

**Remember that your standard errors were computed from the Gram matrix too.**
The usual covariance of the estimates is *σ*²(*X*ᵗ*X*)⁻¹, so a reported standard
error inherits the squared conditioning whether or not the coefficients did. A
library that fits by QR and then reports uncertainty from an explicitly inverted
Gram matrix has done the careful thing once and the careless thing immediately
afterwards — and the second number is the one that ends up in the table.

**And distrust a diagnostic that is downstream of the method.** Orthogonality
cannot audit the normal equations, and the residual barely can. *κ*(X) is
computed from the design alone, before any fit exists, which is exactly what
makes it useful.

*Exercise.* Take a dataset with missing values and compute its covariance matrix
two ways: dropping every row with any missing entry, and computing each pairwise
covariance from whatever rows have both variables. Then find the smallest
eigenvalue of each. One of them can be negative — which would mean some portfolio
of your variables has negative variance. Which one, and why? Episode three.""")

    post.hero = figs["hero"]
    return post


def main() -> Post:
    post = build()
    problems = post.audit()
    print(f"words: {post.word_count()}")
    print("audit:", "clean" if not problems else "")
    for p in problems:
        print("  -", p)
    return post
