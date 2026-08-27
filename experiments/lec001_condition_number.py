"""Linear Algebra 1 — the condition number as an error bar.

First episode of the Lectures corner. The contract is different from a post: a
post is one claim, checked, standing alone; an episode assumes the one before it
and says so at the top. And every number in the text is produced by code that
runs at build time, with the value the prose quotes pinned to the output, so the
episode cannot rot into being wrong.

The shape of every episode: a calculation that a working data scientist would
write, that returns the wrong answer, and then the piece of linear algebra that
says why.

What breaks here
----------------
`np.linalg.solve` on a 14 x 14 Hilbert system whose exact answer is all ones
returns entries between -4.9 and +8.5, while its residual sits at machine
precision. The check everybody runs — "the residual is tiny, the solve worked" —
is precisely the check that cannot see this.

The theory that explains it
---------------------------
kappa(A) = s_max / s_min turns the precision of the input into an error bar on
the output: ||dx||/||x|| <= kappa ||db||/||b||. Doubles perturb b by 1e-16
before any solver runs, so log10(kappa) is the number of decimal digits gone.
Measured against 200 random perturbation directions the bound is reached to
within a factor of about 1.5, so it predicts rather than reassures.

Why a data scientist should care
--------------------------------
The Hilbert matrix is not a curiosity. It is exactly the Gram matrix of the
monomials on [0, 1], so fitting a degree-11 polynomial by normal equations *is*
solving a Hilbert system: kappa(X'X) = 1.6e16. Switching to a Legendre basis —
the same span, the same fitted values, a different parameterisation — takes that
to 22. Fifteen orders of magnitude for a change that does not touch the model.

Run: `standarderror run lec001_condition_number --publish`
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.linalg import conditioning as cn
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.viz import charts

#: Pinned so a rebuild cannot silently re-date a published post.
#: `Post.date` defaults to today, which is correct exactly once.
POST_DATE = date(2026, 8, 27)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Linear Algebra for Data Science, Taught Through What Breaks"
SERIES_TAG = "Linear Algebra"

#: Sizes for the digits table. Stops at 14 because past that every method
#: returns noise and the table stops being informative.
SIZES = (4, 6, 8, 10, 12, 14)
#: Degrees for the basis comparison. 11 is the headline because it is a degree
#: somebody might actually fit.
DEGREES = (3, 5, 7, 9, 11, 13)
HEADLINE_DEGREE = 11
HEADLINE_SIZE = 14
PERTURBATION = 1e-10
PERTURBATION_REPS = 200


def _residual(value: float) -> str:
    """Format a residual that may legitimately be exactly zero.

    On a badly conditioned system the residual lands at or below the last
    representable bit, and whether that prints as 5e-17 or as 0 depends on the
    BLAS. Either way it is the same statement, and neither is a number to quote
    to two digits.
    """
    return "exactly zero" if value == 0 else f"{value:.1e}"


def compute() -> dict:
    """Everything the episode quotes. Cheap enough not to cache."""
    out: dict = {}

    out["solves"] = [vars(cn.solve_report(cn.hilbert(n))) for n in SIZES]
    out["inv"] = [vars(cn.solve_report(cn.hilbert(n), method="inv"))
                  for n in SIZES]
    out["perturb"] = [cn.perturb_and_solve(cn.hilbert(n), relative=PERTURBATION,
                                           reps=PERTURBATION_REPS)
                      for n in (6, 8, 10)]
    out["bases"] = [
        {"degree": d,
         "monomial": cn.gram_condition(d, basis="monomial"),
         "legendre": cn.gram_condition(d, basis="legendre"),
         "hilbert": cn.condition_number(cn.hilbert(d + 1))}
        for d in DEGREES]
    out["eps"] = cn.MACHINE_EPS
    out["digits_available"] = cn.DIGITS_AVAILABLE

    # The vivid version of "the answer is wrong": every entry should be 1.
    H = cn.hilbert(HEADLINE_SIZE)
    x = np.ones(HEADLINE_SIZE)
    x_hat = np.linalg.solve(H, H @ x)
    out["headline_lo"] = float(x_hat.min())
    out["headline_hi"] = float(x_hat.max())

    # The same fit in both bases, to establish that this is a change of
    # parameterisation and not a change of model.
    t = np.linspace(0.0, 1.0, 200)
    y = np.sin(6.0 * t) + 0.3 * t ** 2
    fits = {}
    for basis in ("monomial", "legendre"):
        X = cn.design_matrix(HEADLINE_DEGREE, basis=basis)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fits[basis] = X @ beta
    out["fit_agreement"] = float(np.max(np.abs(fits["monomial"]
                                               - fits["legendre"])))
    return out


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    out = {}

    # --- f1: the prediction, against what happened -------------------------
    solves = res["solves"]
    frame = pd.DataFrame(
        {"digits the answer kept": [s["digits_correct"] for s in solves],
         "digits the theory predicts": [
             max(0.0, res["digits_available"] - s["digits_lost"])
             for s in solves]},
        index=pd.Index([s["n"] for s in solves], name="size of the system"))

    def residual_note(fig, ax):
        # Stated as a threshold rather than as the measured maximum: the
        # measured value is at or below the last representable bit and is not
        # reproducible across BLAS builds.
        ax.annotate("the relative residual stays below 1e-13\n"
                    "everywhere on this chart",
                    (0.97, 0.94), xycoords="axes fraction", ha="right",
                    va="top", fontsize=8.5, color="0.35")

    out["f1"] = charts.lines(
        frame,
        title="The condition number predicts how wrong the answer will be",
        subtitle=("Correct decimal digits in the solution of an n x n Hilbert "
                  "system, against 15.7 minus log10 of the condition number. "
                  "No fitting: the second line is arithmetic on the matrix."),
        xlabel="size of the system (n)",
        ylabel="correct decimal digits in the answer",
        source="Simulated; standarderror/linalg/conditioning.py.",
        decorate=residual_note, direct_labels=False,
        alt=("Two nearly coincident falling lines showing correct digits "
             "against system size, with a note that the residual stays at "
             "machine precision throughout."),
        caption=("The two lines are not a fit and a model; they are a "
                 "measurement and a closed form, and the closed form is the "
                 "lower one because it is a worst case. What the residual does "
                 "over the same range is why this failure is silent."),
        path=str(IMG / f"lec01-f1-digits.{EXT}"))[0]

    # --- f2: the same model, two parameterisations -------------------------
    bases = res["bases"]
    frame2 = pd.DataFrame(
        {"monomial basis (1, t, t², …)": [b["monomial"] for b in bases],
         "Legendre basis": [b["legendre"] for b in bases]},
        index=pd.Index([b["degree"] for b in bases],
                       name="degree of the polynomial"))

    def limit(fig, ax):
        ax.axhline(1.0 / res["eps"], color="0.45", lw=1.0, ls=(0, (4, 3)))
        ax.annotate("1 / machine epsilon: nothing above this line\n"
                    "has a single correct digit left",
                    (frame2.index[0], 1.0 / res["eps"]), xytext=(2, -6),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=8.5, color="0.35")

    out["f2"] = charts.lines(
        frame2,
        title="Conditioning is a property of the basis, not of the problem",
        subtitle=("Condition number of X'X for a least-squares polynomial fit "
                  "on 200 points of [0, 1]. Both bases span the same space and "
                  "produce the same fitted values."),
        xlabel="degree of the polynomial",
        ylabel="condition number of X'X, log scale",
        source="Simulated; standarderror/linalg/conditioning.py.",
        logy=True, decorate=limit, direct_labels=False,
        alt=("One line rising steeply past a dashed limit line and another "
             "staying flat and low, against polynomial degree, on a log "
             "scale."),
        caption=(f"At degree {HEADLINE_DEGREE} the two differ by a factor of "
                 f"{bases[DEGREES.index(HEADLINE_DEGREE)]['monomial'] / bases[DEGREES.index(HEADLINE_DEGREE)]['legendre']:.0e}. "
                 f"The fitted values agree to "
                 f"{res['fit_agreement']:.0e}, so nothing about the model "
                 f"changed."),
        path=str(IMG / f"lec01-f2-basis.{EXT}"))[0]

    # --- t1: the table the episode is built on -----------------------------
    rows = []
    for s, i in zip(res["solves"], res["inv"]):
        rows.append([str(s["n"]), f"{s['kappa']:.1e}", f"{s['digits_lost']:.1f}",
                     f"{s['residual']:.0e}", f"{s['error']:.1e}",
                     f"{s['digits_correct']:.1f}", f"{i['error']:.1e}"])
    out["t1"] = charts.table_image(
        rows,
        header=["n", "κ(H)", "digits lost", "residual", "error", "digits kept",
                "error using inv()"],
        title="A residual at machine precision, and an answer off by 302 percent",
        subtitle=("Solving H x = b for x = (1, …, 1), so the exact answer is "
                  "known. The residual column is the check that gets run; the "
                  "error column is the quantity anybody cares about."),
        source="Simulated; standarderror/linalg/conditioning.py.",
        bold_cols=(3, 4),
        alt="Table of condition number, residual and error against system size.",
        caption=("The last column is the cost of writing inv(A) @ b instead of "
                 "solve(A, b): the same problem, between five and two hundred "
                 "times worse."),
        path=str(IMG / f"lec01-t1-digits.{EXT}"))[0]

    return out


# ---------------------------------------------------------------- the post

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}

    out["break"] = s.run("""
        import numpy as np

        def hilbert(n):
            "H[i,j] = 1/(i+j-1). Symmetric, positive definite, and hopeless."
            i = np.arange(1, n + 1, dtype=float)
            return 1.0 / (i[:, None] + i[None, :] - 1.0)

        n = 14
        H = hilbert(n)
        x = np.ones(n)          # the answer we are going to ask for back
        b = H @ x

        x_hat = np.linalg.solve(H, b)

        print("every entry of x is   1.0")
        print(f"x_hat ranges from    {x_hat.min():.2f} to {x_hat.max():.2f}")
        print(f"relative residual    "
              f"{np.linalg.norm(H @ x_hat - b) / np.linalg.norm(b):.1e}")
    """, expect=["x_hat ranges from    -4.87 to 8.49"])

    out["explain"] = s.run("""
        kappa = np.linalg.cond(H)
        eps = np.finfo(float).eps

        print(f"kappa(H)             {kappa:.2e}")
        print(f"digits available     {-np.log10(eps):.1f}")
        print(f"digits lost to kappa {np.log10(kappa):.1f}")
        print(f"error bound          {kappa * eps:.1e}")
    """, expect=["digits available     15.7", "digits lost to kappa 17.5"])

    out["bound"] = s.run(f"""
        # Is the bound a prediction or a formality? Nudge b by a relative 1e-10 --
        # "my inputs are good to ten decimal places" -- in 200 random directions
        # and keep the worst one.
        def worst_case(n, relative={PERTURBATION}, reps={PERTURBATION_REPS}, seed=0):
            H, x = hilbert(n), np.ones(n)
            b = H @ x
            rng = np.random.default_rng(seed)
            scale = relative * np.linalg.norm(b)
            errors = []
            for _ in range(reps):
                d = rng.standard_normal(n)
                d *= scale / np.linalg.norm(d)
                x_hat = np.linalg.solve(H, b + d)
                errors.append(np.linalg.norm(x_hat - x) / np.linalg.norm(x))
            return max(errors), np.linalg.cond(H) * relative

        for n in (6, 8, 10):
            worst, bound = worst_case(n)
            print(f"n={{n:2d}}  worst measured {{worst:9.2e}}   bound {{bound:9.2e}}"
                  f"   ratio {{worst / bound:.2f}}")
    """, expect=["n= 6  worst measured", "n=10  worst measured"])

    out["basis"] = s.run("""
        from numpy.polynomial import legendre

        # Where does a Hilbert matrix come from? A polynomial fit. The (i,j) entry
        # of X'X for the monomials on [0,1] is the integral of t^i t^j, which is
        # 1/(i+j+1) -- the Hilbert matrix, exactly.
        t = np.linspace(0.0, 1.0, 200)
        degree = 11

        X_mono = np.vander(t, degree + 1, increasing=True)
        eye = np.eye(degree + 1)
        X_leg = np.column_stack([legendre.legval(2 * t - 1, eye[k])
                                 for k in range(degree + 1)])

        print(f"kappa(X'X), monomials  {np.linalg.cond(X_mono.T @ X_mono):.2e}")
        print(f"kappa(X'X), Legendre   {np.linalg.cond(X_leg.T @ X_leg):.2e}")
        print(f"kappa(hilbert({degree + 1}))       "
              f"{np.linalg.cond(hilbert(degree + 1)):.2e}")

        # Same span, so the same fit. Only the parameterisation differs.
        y = np.sin(6 * t) + 0.3 * t**2
        fit = lambda X: X @ np.linalg.lstsq(X, y, rcond=None)[0]
        print(f"fitted values differ by {np.max(np.abs(fit(X_mono) - fit(X_leg))):.1e}")
    """, expect=["kappa(X'X), monomials  1.55e+16",
                 "kappa(X'X), Legendre   2.18e+01"])

    return out


def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    solves = {s["n"]: s for s in res["solves"]}
    invs = {s["n"]: s for s in res["inv"]}
    head = solves[HEADLINE_SIZE]
    bases = {b["degree"]: b for b in res["bases"]}
    hb = bases[HEADLINE_DEGREE]
    worst = res["perturb"][-1]

    # The spine, asserted rather than trusted.
    assert head["error"] > 1.0, head
    assert head["residual"] < 1e-13, head
    assert hb["monomial"] / hb["legendre"] > 1e12, hb
    assert res["fit_agreement"] < 1e-6, res["fit_agreement"]
    assert 0.4 < worst["tightness"] <= 1.0, worst

    post = Post(
        title=f"{SERIES_TAG} 1: The Condition Number Is the Error Bar on Your Solve",
        slug="linear-algebra-1-condition-number",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=1,
        date=POST_DATE,
        subtitle=("A 14 by 14 system whose answer is all ones, solved to a "
                  "residual at machine precision, returning entries between "
                  "-4.9 and +8.5 — and the one number on the matrix that "
                  "predicts it."),
        summary=("The residual is the check everybody runs after a solve, and it "
                 "is the one check that cannot detect the most common way a "
                 "solve goes wrong. The condition number can: it converts the "
                 "precision of the inputs into an error bar on the output, "
                 "before any statistics are involved. And the matrices this "
                 "ruins are not exotic — the textbook example is exactly the "
                 "Gram matrix of a polynomial fit, so a degree-11 regression by "
                 "normal equations is a hopeless system with no correct digits, "
                 "and changing basis removes fifteen orders of magnitude "
                 "without changing the model."),
        tags=["linear-algebra", "numerical-methods", "regression",
              "lectures", "data-science"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=1200, max_words=1900,
        # Every comparison in this episode is between two numerical methods on a
        # problem whose exact answer is known. There is no predictive claim and
        # so no baseline to compare against; the auto-detect reads "the relative
        # error cannot fall below kappa times epsilon" as a performance claim.
        requires_baseline=False,
        data_sources=[
            "No external data. Every matrix here is constructed in the episode "
            "and every number is produced by the code shown, executed when this "
            "page was built.",
            "Machinery: `standarderror/linalg/conditioning.py`, tested in "
            "`tests/test_conditioning.py`.",
            "Where this stops, and who does it properly: Trefethen and Bau, "
            "*Numerical Linear Algebra*, lectures 12 and 18; Higham, *Accuracy "
            "and Stability of Numerical Algorithms*, chapters 1 and 7.",
        ],
        reproducibility={
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "standarderror")),
            "code blocks": ("executed at build time by "
                            "standarderror/render/snippet.py; the printed "
                            "output is captured, not typed, and the values the "
                            "prose quotes are pinned so drift fails the build"),
            "perturbation": (f"{PERTURBATION_REPS} random directions at a "
                             f"relative size of {PERTURBATION:g}, worst case "
                             f"reported"),
            "determinism": ("no seeds matter except the perturbation directions; "
                            "every other number is a property of a fixed matrix"),
        },
    )

    # ------------------------------------------------------------------ 1
    post.add(
        "A solve that works, and is wrong",
        f"""Here is a linear system, its solution, and the check you would run
on it. The matrix is a Hilbert matrix — entry *(i, j)* is 1/(i + j − 1) — which
is symmetric, positive definite, and about as innocent as a matrix looks. The
right-hand side is constructed from a solution of all ones, so the correct
answer is known before we start.""")

    post.add(
        "",
        f"""{snip['break'].markdown()}

The residual is {_residual(head['residual'])}. On any reasonable reading that
solve worked: `numpy` found an *x̂* that reproduces *b* to the last bit a double
can hold. And the entries of that *x̂*, every one of which should be exactly 1,
run from {res['headline_lo']:.2f} to {res['headline_hi']:.2f} — a relative error
of {head['error'] * 100:.0f} percent.

(If you run that block yourself the residual may print as `0e+00` instead. Both
are the same statement — a residual at or below the last representable bit —
and which one you get depends on your BLAS. A number that small is not
reproducible, which is a second reason not to build an argument on it.)

Both statements are true and neither is a bug. `numpy` did nothing wrong, and
neither would LAPACK, MATLAB or R. What has happened is that the residual and
the error are different quantities, and the thing that connects them is a
property of the matrix.""", level=3)

    # ------------------------------------------------------------------ 2
    post.add(
        "The number that connects them",
        f"""Write the singular values of *A* as *σ*₁ ≥ … ≥ *σ*ₙ. The **condition
number** is their ratio:

$$\\kappa(A) = \\frac{{\\sigma_{{\\max}}}}{{\\sigma_{{\\min}}}} = \\lVert A
\\rVert \\, \\lVert A^{{-1}} \\rVert$$

and the reason to care about it is a single inequality. Perturb the right-hand
side and the solution moves by at most

$$\\frac{{\\lVert \\delta x \\rVert}}{{\\lVert x \\rVert}} \;\\le\;
\\kappa(A) \\, \\frac{{\\lVert \\delta b \\rVert}}{{\\lVert b \\rVert}}$$

Read that with no data error in mind at all. Storing *b* in double precision
already perturbs it, by about {res['eps']:.1e} — that is what a double *is*. So
even with perfect measurements, perfect arithmetic and a perfect solver, the
relative error in the answer cannot be pushed below *κ*(A) × {res['eps']:.0e}.
The condition number is not a diagnostic of your data. It is an error bar that the
matrix puts on your answer before your data arrives.""")

    post.add(
        "",
        f"""Where does *κ* come from? Three lines, and they are worth doing once
rather than taking the definition on trust. Solve the perturbed system,
*A*(*x* + *δx*) = *b* + *δb*, subtract the unperturbed one, and you get
*δx* = *A*⁻¹ *δb*, so ‖*δx*‖ ≤ ‖*A*⁻¹‖ ‖*δb*‖. Separately, *b* = *A**x* gives
‖*b*‖ ≤ ‖*A*‖ ‖*x*‖, or 1/‖*x*‖ ≤ ‖*A*‖/‖*b*‖. Multiply the two and ‖*A*‖ ‖*A*⁻¹‖
falls out on its own. That product *is* the condition number, and for the
two-norm it equals *σ*ₘₐₓ/*σ*ₘᵢₙ. Nothing was chosen; it is what is left when you
ask how much a solution can move.

{snip['explain'].markdown()}

A double carries about {res['digits_available']:.1f} decimal digits and this
matrix costs {head['digits_lost']:.1f} of them, so there are none left. That is
the {head['error'] * 100:.0f} percent, derived from the matrix rather than
observed from the failure.

Figure 1 does this across sizes, and the point of it is that nothing was fitted.
The line labelled *predicted* is {res['digits_available']:.1f} − log₁₀ *κ*, an
arithmetic operation on the matrix; the line labelled *kept* is a measurement.
The measurement sits about two digits above the bound the whole way, which is
the only direction a worst-case bound is allowed to be wrong in — and the
residual stays at machine precision under both of them throughout.""",
        figures=[figs["f1"], figs["t1"]], level=3)

    # ------------------------------------------------------------------ 3
    post.add(
        "Is the bound real, or just an inequality?",
        f"""Bounds in numerical analysis have a reputation for being true and
useless — worst cases over directions nobody's data points in. This one is not.
Nudge *b* by a relative {PERTURBATION:g}, which is the honest version of
*my inputs are good to ten decimal places*, in
{PERTURBATION_REPS} random directions, and keep the worst.""")

    post.add(
        "",
        f"""{snip['bound'].markdown()}

The worst of {PERTURBATION_REPS} random directions reaches
{worst['tightness'] * 100:.0f} percent of the bound, so the inequality is close
to attained rather than decorative. Note also the gap between typical and worst:
at *n* = {worst['n']} the median direction gives
{worst['typical']:.1e} against a worst case of {worst['worst']:.1e}. One draw
looks reassuring. The bound is about the direction you did not draw.

While we are here, one habit with a measurable price. `inv(A) @ b` and
`solve(A, b)` compute the same thing in exact arithmetic, and the last column of
the table above is what the first one costs: at *n* = 10 the error is
{invs[10]['error'] / solves[10]['error']:.0f} times larger for no benefit. There
is essentially never a reason to form an explicit inverse. If you find yourself
writing one, what you want is a solve.""")

    # ------------------------------------------------------------------ 4
    post.add(
        "Where you have already met this matrix",
        f"""So far this is a textbook pathology, and a fair objection is that
nobody has a Hilbert matrix. Everybody does.

Fit a polynomial in the obvious basis — 1, *t*, *t*², … — on the interval
[0, 1], and look at the entry of *X*ᵗ*X* in row *i*, column *j*. It is a sum of
*t^i · t^j* over the sample, which approximates ∫₀¹ *t*^(i+j) d*t* = 1/(i + j +
1). **The Gram matrix of the monomials is the Hilbert matrix.** Not similar to
it: it is it.""")

    post.add(
        "",
        f"""{snip['basis'].markdown()}

A degree-{HEADLINE_DEGREE} polynomial fitted by normal equations is therefore a
system with *κ* = {hb['monomial']:.1e} — no correct digits — and it will not
warn you, because its residual will be fine.

The third line is the fix, and it is worth dwelling on why it works. The
Legendre polynomials of degree ≤ {HEADLINE_DEGREE} span **exactly the same
space** as the monomials of degree ≤ {HEADLINE_DEGREE}. Same model, same
achievable fits, same predictions — the last line confirms the fitted values
agree to {res['fit_agreement']:.0e}. All that changed is which basis the
coefficients are expressed in, and the condition number went from
{hb['monomial']:.1e} to {hb['legendre']:.0f}. Figure 2 is that comparison across
degrees.

That is the general lesson, and it is bigger than polynomials: **conditioning is
a property of the parameterisation, not of the problem.** A design matrix whose
columns are a duration in seconds, a probability and a currency amount is badly
conditioned for reasons that have nothing to do with the statistics of the data,
and centring and scaling the columns is not cosmetic tidying — it is the cheapest
available reduction in the error bar on your coefficients.""",
        figures=[figs["f2"]], level=3)

    # ------------------------------------------------------------------ 5
    post.add(
        "What to take away, and what is still hiding",
        f"""Four things, in the order you would use them.

**Ask for the condition number, not for singularity.** A matrix that is singular
on paper almost never has an exactly zero singular value in floating point — it
has one near 1e-16 — so "is it singular?" answers no and tells you nothing.
`np.linalg.cond` is one line and gives you log₁₀ of it as digits gone.

**Do not read a small residual as a correct answer.** They are different
quantities. A backward-stable solver guarantees the first and says nothing about
the second.

**Never form an explicit inverse.** Use `solve`. The table above prices the
alternative.

**Scale your columns, and consider the basis.** Equilibration is free.
Orthogonal bases exist for a reason.

And one thing this episode has quietly avoided. Everything above was a square
system, *A x = b*. A regression is not: it is a least-squares problem, and the
step from one to the other is where the real damage happens, because forming
*X*ᵗ*X* **squares** the condition number. The degree-{HEADLINE_DEGREE} fit above
has a design matrix *X* with *κ*(X) ≈ {np.sqrt(hb['monomial']):.1e} — bad, but
survivable — and normal equations turned it into {hb['monomial']:.1e}, which is
not. There are three standard ways to solve a least-squares problem and only two
of them avoid that. Next episode.

*Exercise.* Take a design matrix you actually use. Compute *κ*(X) three times:
raw, with the columns centred, and with the columns standardised. Which of the
three steps does the work — and does the answer change if one of your columns is
a dummy variable? The answer is at the top of episode 2.""")

    return post


def main() -> Post:
    post = build()
    problems = post.audit()
    print(f"words: {post.word_count()}")
    print("audit:", "clean" if not problems else "")
    for p in problems:
        print("  -", p)
    return post
