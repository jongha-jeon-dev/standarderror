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
| 4 | PCA When Two Eigenvalues Are Equal | bootstrap the sample and watch the second and third components change places; "we interpret PC2 as" |
| 5 | What Ridge Does to the Geometry | VIF says the design is fine, the singular values say it is not; ridge as eigenvalue shifting and Σλ/(λ+α) as the degrees of freedom you actually spent |
| 6 | One Row Can Own the Fit | leverage as a diagonal of a projection, and a single observation moving a coefficient by three standard errors |
| 7 | The Scree Plot Lies | Eckart–Young, why truncating a factorisation is a modelling decision, and how to choose the rank without looking at an elbow |
| 8 | Logistic Regression Is Least Squares, Repeated | IRLS from scratch on public credit data: where the weights come from, and why separable data sends a coefficient to infinity |

Episodes are 1,200–1,800 words. Everything runs on simulated data or on a public
dataset named in the episode.

### What this series is not

It is not a course in numerical linear algebra — it borrows from that field
without pretending to cover it, and points at Trefethen and Bau or Golub and Van
Loan where it stops. It is not a substitute for a linear algebra text: there are
no proofs here that a textbook does better, and the geometric intuition is
Strang's and 3Blue1Brown's rather than mine. What it adds is the part those leave
out — what the theory looks like from underneath, when the answer on your screen
is wrong and you have to work out why.
