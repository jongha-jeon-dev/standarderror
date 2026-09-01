---
title: "Lectures"
description: "Short courses where every number in the text was produced by code that ran when the page was built."
---

Two things separate these from the posts.

They are **cumulative**. A post here is one claim, checked, and it stands alone. A
lecture assumes the episode before it, and says so at the top.

And **every number in an episode was produced by code that ran when the page was
built**, with the value the prose quotes pinned to the output. If the code stops
producing that number, the build fails and the episode does not publish. Course
material rots; this is the cheapest defence against it I know of.

Each episode opens with a computation that returns the wrong answer. The theory
arrives to explain the wrong answer, not before it.

### How an episode is built

Five rules, in the order they get applied.

**The failure comes first, at a size that fits in your head.** Before any
general result there is a two-by-two version you can check by hand. If the
smallest honest example does not already show the problem, the episode has not
found the problem yet.

**Every object is introduced before it is used.** A singular value is not a
definition to be accepted; it is the length of a semi-axis of the ellipse your
matrix turns the unit circle into. Notation that arrives without its picture is
notation the reader will skip, and then the rest of the episode is decoration.

**The derivations are here, and they are slow.** This is where a lecture differs
from a post: a post cites the three lines, an episode does them, one displayed
step at a time, with a sentence afterwards saying what the step bought. The
target is that a reader with first-year calculus and no linear algebra can
follow every line — not skim it, follow it.

**The code is short and does not carry the explanation.** Twenty to thirty lines
an episode, enough to run the failure and check the claim, never enough to become
the subject. If something can be explained in a sentence or in a loop, it gets
the sentence.

**Every claim that has a shape gets drawn.** Five or six figures an episode, and
each one has to be the argument rather than an illustration of it: the ellipse a
matrix turns the circle into, the two bases side by side, the spectrum before and
after a squaring. A picture that could be deleted without weakening the argument
is deleted.

---

## Linear Algebra for Data Science, Taught Through What Breaks

The subject is taught everywhere, almost always forwards: definitions, then
properties, then an application. This goes the other way. Every episode starts
from a calculation that a working data scientist would write, and that is wrong —
sometimes silently, by a few digits; sometimes catastrophically, by a sign — and
then finds the piece of linear algebra that says why.

The last episode lands on a credit scorecard, from nothing but least squares.
That is not a detour: a scorecard *is* iteratively reweighted least squares, and
arriving there from the geometry rather than from a library call is the point of
the seven episodes before it.

| # | Episode | The calculation that breaks |
|---|---|---|
| 1 | The Condition Number Is the Error Bar on Your Solve | `inv(A) @ b` on a design matrix with two nearly-collinear columns: coefficients flip sign under a perturbation of 1e-10 |
| 2 | Least Squares Three Ways, and Only Two Survive | normal equations against QR against SVD on the same fit — κ(AᵗA) = κ(A)² and half the digits are gone |
| 3 | Your Covariance Matrix Is Not Positive Definite | pairwise-deleted covariance with a negative eigenvalue, and a portfolio variance that comes out below zero |
| 4 | PCA When Two Eigenvalues Are Equal | the two *largest* eigenvalues 0.02 apart, so the component carrying the most variance is the one whose axis swings 42° between samples — and the bootstrap you would run reports a third of it |
| 5 | What Ridge Does to the Geometry | every VIF at 1.00 on a design with a condition number near a billion, ridge as a per-direction multiplier s²/(s²+α), and a cross-validated fit that spends 3.6 of its 9 parameters while the output reports 9 |
| 6 | One Row Can Own the Fit | leverage as a diagonal of a projection whose trace is fixed at p, a row with leverage 1 whose residual is exactly zero, and one observation moving a slope by six standard errors |
| 7 | The Scree Plot Lies | Eckart–Young, why truncating a factorisation is a modelling decision, and how to choose the rank without looking at an elbow |
| 8 | Logistic Regression Is Least Squares, Repeated | IRLS from scratch on public credit data: where the weights come from, and why separable data sends a coefficient to infinity |

Episodes run 1,900–2,800 words. Everything runs on simulated data or on a public
dataset named in the episode.

### What this series is not

It is not a course in numerical linear algebra — it borrows from that field
without pretending to cover it, and points at Trefethen and Bau or Golub and Van
Loan where it stops. It is not a substitute for a linear algebra text: there are
no proofs here that a textbook does better, and the geometric intuition is
Strang's and 3Blue1Brown's rather than mine. What it adds is the part those leave
out — what the theory looks like from underneath, when the answer on your screen
is wrong and you have to work out why.
