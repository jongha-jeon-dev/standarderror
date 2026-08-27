---
title: "Linear Algebra 1: The Condition Number Is the Error Bar on Your Solve"
date: 2026-08-27
slug: "linear-algebra-1-condition-number"
draft: true
description: "A 14 by 14 system whose answer is all ones, solved to a residual at machine precision, returning entries between -4.9 and +8.5 — and the one number on the matrix that predicts it."
author: "Jongha Jeon"
tags: ["linear-algebra", "numerical-methods", "regression", "lectures", "data-science"]
series: ["Linear Algebra for Data Science, Taught Through What Breaks"]
weight: 1
---

Disclosure: this post was written with the assistance of an AI system (Claude), which wrote the analysis code, ran the experiments and drafted the text. The topic, the constraints, the data choices and the final review are the author's.

*The residual is the check everybody runs after a solve, and it is the one check that cannot detect the most common way a solve goes wrong. The condition number can: it converts the precision of the inputs into an error bar on the output, before any statistics are involved. And the matrices this ruins are not exotic — the textbook example is exactly the Gram matrix of a polynomial fit, so a degree-11 regression by normal equations is a hopeless system with no correct digits, and changing basis removes fifteen orders of magnitude without changing the model.*

Episode 1 of *Linear Algebra for Data Science, Taught Through What Breaks*. The syllabus and the other episodes: https://jongha-jeon-dev.github.io/standarderror/lectures/

## A solve that works, and is wrong

Here is a linear system, its solution, and the check you would run
on it. The matrix is a Hilbert matrix — entry *(i, j)* is 1/(i + j − 1) — which
is symmetric, positive definite, and about as innocent as a matrix looks. The
right-hand side is constructed from a solution of all ones, so the correct
answer is known before we start.

```python
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
```

```text
every entry of x is   1.0
x_hat ranges from    -4.87 to 8.49
relative residual    4.8e-17
```

The residual is 4.8e-17. On any reasonable reading that
solve worked: `numpy` found an *x̂* that reproduces *b* to the last bit a double
can hold. And the entries of that *x̂*, every one of which should be exactly 1,
run from -4.87 to 8.49 — a relative error
of 302 percent.

(If you run that block yourself the residual may print as `0e+00` instead. Both
are the same statement — a residual at or below the last representable bit —
and which one you get depends on your BLAS. A number that small is not
reproducible, which is a second reason not to build an argument on it.)

Both statements are true and neither is a bug. `numpy` did nothing wrong, and
neither would LAPACK, MATLAB or R. What has happened is that the residual and
the error are different quantities, and the thing that connects them is a
property of the matrix.

## The number that connects them

Write the singular values of *A* as *σ*₁ ≥ … ≥ *σ*ₙ. The **condition
number** is their ratio:

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \lVert A
\rVert \, \lVert A^{-1} \rVert$$

and the reason to care about it is a single inequality. Perturb the right-hand
side and the solution moves by at most

$$\frac{\lVert \delta x \rVert}{\lVert x \rVert} \;\le\;
\kappa(A) \, \frac{\lVert \delta b \rVert}{\lVert b \rVert}$$

Read that with no data error in mind at all. Storing *b* in double precision
already perturbs it, by about 2.2e-16 — that is what a double *is*. So
even with perfect measurements, perfect arithmetic and a perfect solver, the
relative error in the answer cannot be pushed below *κ*(A) × 2e-16.
The condition number is not a diagnostic of your data. It is an error bar that the
matrix puts on your answer before your data arrives.

Where does *κ* come from? Three lines, and they are worth doing once
rather than taking the definition on trust. Solve the perturbed system,
*A*(*x* + *δx*) = *b* + *δb*, subtract the unperturbed one, and you get
*δx* = *A*⁻¹ *δb*, so ‖*δx*‖ ≤ ‖*A*⁻¹‖ ‖*δb*‖. Separately, *b* = *A**x* gives
‖*b*‖ ≤ ‖*A*‖ ‖*x*‖, or 1/‖*x*‖ ≤ ‖*A*‖/‖*b*‖. Multiply the two and ‖*A*‖ ‖*A*⁻¹‖
falls out on its own. That product *is* the condition number, and for the
two-norm it equals *σ*ₘₐₓ/*σ*ₘᵢₙ. Nothing was chosen; it is what is left when you
ask how much a solution can move.

```python
kappa = np.linalg.cond(H)
eps = np.finfo(float).eps

print(f"kappa(H)             {kappa:.2e}")
print(f"digits available     {-np.log10(eps):.1f}")
print(f"digits lost to kappa {np.log10(kappa):.1f}")
print(f"error bound          {kappa * eps:.1e}")
```

```text
kappa(H)             3.22e+17
digits available     15.7
digits lost to kappa 17.5
error bound          7.1e+01
```

A double carries about 15.7 decimal digits and this
matrix costs 17.5 of them, so there are none left. That is
the 302 percent, derived from the matrix rather than
observed from the failure.

Figure 1 does this across sizes, and the point of it is that nothing was fitted.
The line labelled *predicted* is 15.7 − log₁₀ *κ*, an
arithmetic operation on the matrix; the line labelled *kept* is a measurement.
The measurement sits about two digits above the bound the whole way, which is
the only direction a worst-case bound is allowed to be wrong in — and the
residual stays at machine precision under both of them throughout.

![Two nearly coincident falling lines showing correct digits against system size, with a note that the residual stays at machine precision throughout.](lec01-f1-digits.png)

*The two lines are not a fit and a model; they are a measurement and a closed form, and the closed form is the lower one because it is a worst case. What the residual does over the same range is why this failure is silent.*

![Table of condition number, residual and error against system size.](lec01-t1-digits.png)

*The last column is the cost of writing inv(A) @ b instead of solve(A, b): the same problem, between five and two hundred times worse.*

## Is the bound real, or just an inequality?

Bounds in numerical analysis have a reputation for being true and
useless — worst cases over directions nobody's data points in. This one is not.
Nudge *b* by a relative 1e-10, which is the honest version of
*my inputs are good to ten decimal places*, in
200 random directions, and keep the worst.

```python
# Is the bound a prediction or a formality? Nudge b by a relative 1e-10 --
# "my inputs are good to ten decimal places" -- in 200 random directions
# and keep the worst one.
def worst_case(n, relative=1e-10, reps=200, seed=0):
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
    print(f"n={n:2d}  worst measured {worst:9.2e}   bound {bound:9.2e}"
          f"   ratio {worst / bound:.2f}")
```

```text
n= 6  worst measured  1.14e-03   bound  1.50e-03   ratio 0.76
n= 8  worst measured  1.04e+00   bound  1.53e+00   ratio 0.68
n=10  worst measured  1.02e+03   bound  1.60e+03   ratio 0.64
```

The worst of 200 random directions reaches
64 percent of the bound, so the inequality is close
to attained rather than decorative. Note also the gap between typical and worst:
at *n* = 10 the median direction gives
3.3e+02 against a worst case of 1.0e+03. One draw
looks reassuring. The bound is about the direction you did not draw.

While we are here, one habit with a measurable price. `inv(A) @ b` and
`solve(A, b)` compute the same thing in exact arithmetic, and the last column of
the table above is what the first one costs: at *n* = 10 the error is
157 times larger for no benefit. There
is essentially never a reason to form an explicit inverse. If you find yourself
writing one, what you want is a solve.

## Where you have already met this matrix

So far this is a textbook pathology, and a fair objection is that
nobody has a Hilbert matrix. Everybody does.

Fit a polynomial in the obvious basis — 1, *t*, *t*², … — on the interval
[0, 1], and look at the entry of *X*ᵗ*X* in row *i*, column *j*. It is a sum of
*t^i · t^j* over the sample, which approximates ∫₀¹ *t*^(i+j) d*t* = 1/(i + j +
1). **The Gram matrix of the monomials is the Hilbert matrix.** Not similar to
it: it is it.

```python
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
```

```text
kappa(X'X), monomials  1.55e+16
kappa(X'X), Legendre   2.18e+01
kappa(hilbert(12))       1.64e+16
fitted values differ by 3.9e-14
```

A degree-11 polynomial fitted by normal equations is therefore a
system with *κ* = 1.6e+16 — no correct digits — and it will not
warn you, because its residual will be fine.

The third line is the fix, and it is worth dwelling on why it works. The
Legendre polynomials of degree ≤ 11 span **exactly the same
space** as the monomials of degree ≤ 11. Same model, same
achievable fits, same predictions — the last line confirms the fitted values
agree to 4e-14. All that changed is which basis the
coefficients are expressed in, and the condition number went from
1.6e+16 to 22. Figure 2 is that comparison across
degrees.

That is the general lesson, and it is bigger than polynomials: **conditioning is
a property of the parameterisation, not of the problem.** A design matrix whose
columns are a duration in seconds, a probability and a currency amount is badly
conditioned for reasons that have nothing to do with the statistics of the data,
and centring and scaling the columns is not cosmetic tidying — it is the cheapest
available reduction in the error bar on your coefficients.

![One line rising steeply past a dashed limit line and another staying flat and low, against polynomial degree, on a log scale.](lec01-f2-basis.png)

*At degree 11 the two differ by a factor of 7e+14. The fitted values agree to 4e-14, so nothing about the model changed.*

## What to take away, and what is still hiding

Four things, in the order you would use them.

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
*X*ᵗ*X* **squares** the condition number. The degree-11 fit above
has a design matrix *X* with *κ*(X) ≈ 1.2e+08 — bad, but
survivable — and normal equations turned it into 1.6e+16, which is
not. There are three standard ways to solve a least-squares problem and only two
of them avoid that. Next episode.

*Exercise.* Take a design matrix you actually use. Compute *κ*(X) three times:
raw, with the columns centred, and with the columns standardised. Which of the
three steps does the work — and does the answer change if one of your columns is
a dummy variable? The answer is at the top of episode 2.

---

### Data

- No external data. Every matrix here is constructed in the episode and every number is produced by the code shown, executed when this page was built.
- Machinery: `standarderror/linalg/conditioning.py`, tested in `tests/test_conditioning.py`.
- Where this stops, and who does it properly: Trefethen and Bau, *Numerical Linear Algebra*, lectures 12 and 18; Higham, *Accuracy and Stability of Numerical Algorithms*, chapters 1 and 7.

### Reproducibility

- **environment**: standarderror=0.1.0, python=3.11.15, numpy=2.4.4
- **code blocks**: executed at build time by standarderror/render/snippet.py; the printed output is captured, not typed, and the values the prose quotes are pinned so drift fails the build
- **perturbation**: 200 random directions at a relative size of 1e-10, worst case reported
- **determinism**: no seeds matter except the perturbation directions; every other number is a property of a fixed matrix

Code: <https://github.com/jongha-jeon-dev/standarderror>
