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

The last episode lands on logistic regression, from nothing but least squares.
That is not a detour: the standard way to fit one *is* iteratively reweighted
least squares, and arriving there from the geometry rather than from a library
call is what makes its failure modes legible instead of mysterious.

| # | Episode | The calculation that breaks |
|---|---|---|
| 1 | The Condition Number Is the Error Bar on Your Solve | `inv(A) @ b` on a design matrix with two nearly-collinear columns: coefficients flip sign under a perturbation of 1e-10 |
| 2 | Least Squares Three Ways, and Only Two Survive | normal equations against QR against SVD on the same fit — κ(AᵗA) = κ(A)² and half the digits are gone |
| 3 | Your Covariance Matrix Is Not Positive Definite | pairwise-deleted covariance with a negative eigenvalue, and a portfolio variance that comes out below zero |
| 4 | PCA When Two Eigenvalues Are Equal | the two *largest* eigenvalues 0.02 apart, so the component carrying the most variance is the one whose axis swings 42° between samples — and the bootstrap you would run reports a third of it |
| 5 | What Ridge Does to the Geometry | every VIF at 1.00 on a design with a condition number near a billion, ridge as a per-direction multiplier s²/(s²+α), and a cross-validated fit that spends 3.6 of its 9 parameters while the output reports 9 |
| 6 | One Row Can Own the Fit | leverage as a diagonal of a projection whose trace is fixed at p, a row with leverage 1 whose residual is exactly zero, and one observation moving a slope by six standard errors |
| 7 | The Scree Plot Lies | the elbow asked about a matrix of pure noise: 18 different answers in 300 draws, never once "none" — while the rule with a theorem behind it reports zero components where three exist, on purpose |
| 8 | When There Is No Closed Form | IRLS from scratch: a coefficient whose value is the iteration limit, a standard error equal to 1/√(k × the library's weight floor), and a p-value that crosses 0.05 on the way down |

Episodes run 1,900–2,800 words. Everything runs on simulated data or on a public
dataset named in the episode.

**The series is complete.** Episode 8 closes it with a one-page recap of all
eight, which is the fastest way to see whether any of it is for you.

### What this series is not

It is not a course in numerical linear algebra — it borrows from that field
without pretending to cover it, and points at Trefethen and Bau or Golub and Van
Loan where it stops. It is not a substitute for a linear algebra text: there are
no proofs here that a textbook does better, and the geometric intuition is
Strang's and 3Blue1Brown's rather than mine. What it adds is the part those leave
out — what the theory looks like from underneath, when the answer on your screen
is wrong and you have to work out why.

---

## Numerical Analysis for Machine Learning, Taught Through What Breaks

One object, followed through six episodes: **the discrete operator you actually
ran**, as against the continuous one you wrote down. In a machine learning system
those operators are not exotic — they are the ones you use daily. Gradient
descent *is* forward Euler on the gradient flow. A gradient check *is* a finite
difference. A diffusion sampler *is* an ODE solver. A decode loop *is* an
iterated map.

So the classical analysis is load-bearing here rather than decorative, and the
thesis is that **every one of these has a step size, and the step size has a
stability limit, an optimum, or a horizon that nobody prints.**

Two things recur. The limits are *hard* rather than gradual — two percent of a
learning rate is worth seven orders of magnitude in episode 1 — and the floating
point precision decides where they are: `bfloat16` has `eps = 3.9e-3` against
float64's `2.2e-16`, which moves every scale in episode 2 by eight orders of
magnitude.

| # | Episode | The calculation that breaks |
|---|---|---|
| 1 | Your Learning Rate Is a Step Size | gradient descent as forward Euler: the limit is exactly 2/λ_max, the run at 0.99× of it ends at 4.4e-04 and the one at 1.01× at 3.9e+03, and the flow being approximated converges at both. Then a network, where 2/lr turns out to be a two-sided attractor for the curvature and the usable limit is 2.04× the one you would compute at initialisation |
| 2 | A Gradient Check Is a Finite Difference | the U-curve in the step size, and the step nobody decides. At h = 1e-5 the check resolves a 9.3e-10 relative error in one gradient entry; at h = 1e-13 it needs 9.9% — and in bfloat16 it cannot see an error below about 2.5% at any step |
| 3 | The Reduction Order Changes the Bits | splitting a matmul's contraction is exact algebra and inexact arithmetic. The deciding quantity is the perturbation divided by the top-two logit gap, and precision moves it four orders of magnitude — so the popular version of this claim is true in bfloat16 and false in float32 |
| 4 | The Sampler Is an Integrator | diffusion sampling as a probability-flow ODE, where the step count is a discretisation choice and a higher-order solver can be worse at low step counts |
| 5 | logsumexp, and the Softmax That Overflows | the max-subtraction trick as cancellation control, and what attention does in half precision where the exponent range binds before the mantissa does |
| 6 | An Autoregressive Rollout Has a Lyapunov Time | a decode loop as an iterated map, so a one-token perturbation grows at a measurable rate — and past that horizon "the same prompt" stops being a meaningful phrase |

Episodes 1 and 2 are published. The rest are written in order and the table is
the commitment; if an episode's opening claim does not survive its own
measurement it gets corrected rather than quietly dropped, and both published
episodes say where that already happened — episode 1 about the edge of
stability, episode 2 about what a bfloat16 gradient check can see.

---

## Topology for Language Models, Taught Through What Breaks

One object: **the filtration you imposed**, as against the shape of the data. A
filtration is a choice of metric and a choice of scale, and in an embedding
space both were settled by normalisation steps upstream rather than decided.

Thesis: **persistent homology computes a property of your metric and your
scale, and in high dimensions the barcode loses its dynamic range long before
the method loses its power** — which turns out to be an argument for the method
and against reading its numbers absolutely, rather than against both.

Two things recur. The machinery is smaller than its reputation — H₀ of a Rips
filtration is single-linkage clustering, and for a graph the first Betti number
is Euler's formula — and the summaries are stable while the numbers people read
off them are not.

A note on scope, because it decides what these episodes can claim. Episodes 1
to 3 are about what the method computes and are exact on any point cloud.
Episodes 4 and 5 need a language model, and the one they use is a
816,128-parameter character-level transformer trained for this series, with a
validation loss of 1.573 against a uniform-guess 4.174. That is a real language
model and a small one. Nothing here is a claim about a frontier model's
geometry.

| # | Episode | The calculation that breaks |
|---|---|---|
| 1 | H₀ Is Single-Linkage Clustering, Bit for Bit | the barcode and the dendrogram return the same floats, so chaining is inherited: three points strung between two blobs take the two-cluster signal from 6.70 to 1.00, and twelve make the k = 2 cut return 71 points and 1. The stability theorem is real; the cluster count read off the barcode is 2 in 22 of 40 noise draws and something else in the rest |
| 2 | A Barcode's Numbers Mean Nothing on Their Own | the barcode's dynamic range falls from 4.87 at two dimensions to 0.078 at 768 — and the summary gets *better*, not worse. The largest-gap rule reports "one cluster" zero times in 40 draws of pure noise at every dimension, and the two-cluster ratio read absolutely is inverted at d = 2 (AUC 0.462, worse than a coin) while against a matched null it reaches 0.809 at 768. This is the episode whose plan the third measurement reversed |
| 3 | Your Metric Is Three Normalisation Steps You Forgot | a mean offset alone drives every pairwise cosine similarity to 0.995 and collapses their spread from 1.75 to 0.02, leaving 2% of the filtration axis to carry the structure. Centring, L2 and whitening are four different answers, not four spellings of one |
| 4 | The Betti Number of an Attention Graph Is a Repackaged Entropy | 16 heads, a topological summary with a hundredfold range — and a Spearman correlation of +0.897 with a one-line statistic that ranks them the same way |
| 5 | A Greedy Decode Closes an Exact Loop | greedy decoding becomes periodic at generated character 116 with period 41, and the hidden state repeats bit for bit, so 443 of 600 steps are exact replays. Sampling never returns closer than 9% of the trajectory's mean spacing |

Episodes 1 and 2 are published. Every opener above is measured rather than
projected, and two of those measurements contradicted the claim they were
written to make: episode 4's, and episode 2's, which was written to say the
display fails in high dimensions and ended up showing that it becomes the
reference.

---

## Headline Statistics, Taught Through What Breaks

One object: **an aggregate statistic that gets printed as though it described a
person**. Unlike the other three series this one needs no machine learning and
almost no code — the failures here are arithmetic, and they are in the numbers
that reach the front page rather than in a training run.

Thesis: **the three summaries below fail in three structurally different ways,
and none of the three failures is visible in the number itself.** One describes
nobody, so timing moves it. One moves because the population changed rather
than the people in it. One is many-to-one, so opposite worlds share a value.

Every episode is built the same way as the rest of the site: the mechanism is
constructed in a simulation where the answer is known by design, the algebra is
derived rather than cited, and published figures appear only in the prose, never
inside a computation.

| # | Episode | The number that breaks |
|---|---|---|
| 1 | Korea's Fertility Rebound Needs Nobody to Have More Children | the period TFR is one calendar year of age-specific rates stacked into a woman who does not exist. Give every cohort the same completed fertility and postpone by *d* per cohort: the period rate is exactly *Q*/(1+*d*), the period mean age rises at *d*/(1+*d*), and dividing by 1 − *r* returns *Q* to five decimals. Postponement *decelerating* from 0.2 to 0.1 then lifts the period rate 9.1% with nobody having more children — 28 years later, and taking 21 more to arrive |
| 2 | Everyone's Wage Rose and the Median Fell | composition. Hire entrants below the median and it slides *m*/2 ranks at a cost of *m*/(2*n f*(*M*)), so a universal raise of *g* vanishes once the entry share passes 0.798*g*/*sigma* — 4.0% at a 3% raise. The threshold *falls* as inequality rises, and the median breaks before the mean. Reverse the sign and losing the lowest-paid 9.8% prints 10.4% median growth on 2.5% of real growth, which is what the US printed in the quarter it lost 20.5 million jobs |
| 3 | Two Countries, One Gini, Opposite Policies | the poorest tenth losing 90% of its income and the richest hundredth multiplying by 2.51 land on the same Gini exactly — and on the same p90/p10, p50/p10, p90/p50 and poverty headcount, because each distortion lives inside a tail. Five headline numbers agree while the poorest tenth holds 9.2× more income in one world. `dG/dx_k = 2k/(n²μ) − c`, so a unit of income is priced by the recipient's rank and nothing else: R² of 1.0000000000 over ranks whose incomes differ sixfold |

All three episodes are published, and each was drafted around a claim its own
measurement refused. Episode 1 expected the Bongaarts-Feeney adjustment to break
when the fertility schedule's spread changes, which is the assumption the
literature attacks hardest; measured, that costs about one percent. Episode 2
expected the median to be the robust choice against composition; measured, it
breaks at about 60% of the entry share the mean needs, at every spread tried.
Episode 3's first attempt to measure the Gini's sensitivity moved a sum three
thousand times larger than the gap between neighbouring incomes, so it measured
reordering rather than the identity, and came out non-monotone; the corrected
measurement is exact to ten decimal places. Every episode says so, and moves the
criticism to where the measurement actually put it.

The track closes on the question the three share, which is not about demography,
wages or inequality: **for any summary you rely on, what are two states of the
world it cannot tell apart, and would you act differently in them?**

---

## Calculus for Language Models, Taught Through What Breaks

One object: **the derivative you are actually computing**. Backpropagation is
the chain rule, and the chain rule has hypotheses — differentiability, a
Jacobian of full rank, a function of the thing you are differentiating with
respect to. A transformer violates all three, in specific places, and the
framework returns a number anyway.

Thesis: **each of the five failures below is an exact statement about one
component, not an approximation or a training artefact.** The Jacobian of
`LayerNorm` has rank exactly *d* − 2. A softmax's gradient vanishes
quadratically in its confidence. A sampled token has no derivative at all. None
of that is a bug and none of it is fixable; it is the shape of the object you
are optimising.

The measurements are made on a **816,128-parameter character-level
transformer** trained for this series — four blocks, four heads, width 128,
validation loss 1.573 against a uniform-guess 4.174 — with the weights and the
training script committed to the repository. The exact results hold at any
width; the frequencies are properties of that model and are reported as such.

| # | Episode | The derivative that is not there |
|---|---|---|
| 1 | Backprop Is the Chain Rule, and the Chain Rule Has Hypotheses | ReLU and GELU are not differentiable everywhere, `max` has a subgradient rather than a derivative, and autodiff returns a float regardless. What the framework picks at the kink, and how often a real training step lands on one |
| 2 | A Confident Attention Head Passes Almost No Gradient | the softmax Jacobian is `diag(p) − ppᵀ`, its quadratic form is a variance, and so its norm is trapped between *m*(1 − *m*) and 2*m*(1 − *m*) with the width of the row nowhere in it — after which one head of this model turns out to have committed hard enough that the gradient which could change its mind is gone |
| 3 | LayerNorm Deletes Exactly Two Directions of Your Gradient | its Jacobian is `(I − 11ᵀ/d − x̂x̂ᵀ/d)/σ`, which is 1/σ times an orthogonal projector of rank exactly *d* − 2 — and the two dead directions are the invariances the layer was built to have, so the deficiency is correctness rather than loss. Then the residual restores the rank in every block, and the final norm, which has none, turns them into exact invariances of the whole network |
| 4 | You Cannot Differentiate Through a Sampled Token | the straight-through estimator is not an approximation of a gradient that exists, and on a decision small enough to enumerate its bias comes out 42 times its own noise — while the loss model it rests on is so **compressed** that it explains the missing magnitude exactly, a constant being the softmax Jacobian's null direction |
| 5 | The Gradient in Embedding Space Does Not Point at a Token | the table is 65 near-orthogonal points on a sphere, so a descent step leaves you nearest to the token you started from and the one you eventually reach is never the best — while the **same** gradient, used to rank five candidates rather than to point, recovers almost all of the available gain |

Episodes 1 and 2 are published, and both end somewhere other than where they
were drafted to end. Episode 1 was written to show that non-differentiability is
a live problem in a real training run; it went looking, and found that gradient
clipping's kink was never reached in 600 steps and that not one of 10.6 million
attention probabilities is exactly 0 or 1. The one place the gradient is
identically zero is put there by the causal mask, not by training. What
throttles gradient flow is confidence, which is smooth.

Episode 2 then measures that, and the honest finding is not that saturation is
a pathology. Layer 0 head 1 sits at *m* = 0.97, is a previous-token head on
99.8% of rows, and passes 5.9 times less routing gradient than the next-lowest
head — and it is the head the model cannot lose, since zeroing it costs 1.12
nats while replacing it with a fixed shift-by-one permutation costs 0.0013.
Saturation is what commitment looks like from the inside of a Jacobian, and the
`1/sqrt(d)` scale in front of the attention logits exists to postpone it.

Episode 3 does the same thing to the last hypothesis. LayerNorm's Jacobian has
rank exactly *d* − 2 and the two missing directions are its own invariances,
so the deficiency is the derivative being correct rather than a leak — and the
residual connection restores the rank in every block, leaving the exact result
structurally invisible. The exception is the final norm, which has no residual
after it: two directions of the final hidden state have **no effect at all**
on this model's output. What actually moves gradient magnitudes is the scalar,
1/σ, which falls 3.4-fold across depth because the residual stream grows.

Episode 4 leaves the hypotheses behind for a place with no derivative at all.
A sampled token is piecewise constant in its logits, so what everyone
differentiates is the expectation — and over a 65-token vocabulary that is
enumerable, which turns "which estimator is better" into a question with an
answer. REINFORCE is unbiased and measurably so. Straight-through is biased by
42 times its own noise and returns 36% of the right magnitude, and the reason
is that its implicit loss model is compressed to a seventh of the truth's
range: a constant loss model gives exactly zero gradient, because constants
are the softmax Jacobian's null direction. The crossover at which the unbiased
estimator wins is 2 to 8 samples.

Episode 5 closes the series where the derivative is exact and useless. The
embedding table's median pairwise distance is 11.214 and √2 × its median norm
is 11.210 — 65 near-orthogonal points on a sphere, further apart than they are
long, which is dimension rather than training. So there is no neighbourhood: a
descent step leaves you nearest to the token you started from for 1.9 to 31.6
embedding norms, and the token you eventually reach is never the best
substitution and is sometimes worse than not moving. But ranking is
scale-invariant, so the compression that ruined the magnitude in episode 4
costs the *ordering* nothing — the gradient's top five of 65 recovers 96% to
100% of the available gain in five of six contexts, against 8% to 17% of
random shortlists matching it. Discrete-token optimisation is search with a
gradient-shaped shortlist, and that is the only job this geometry leaves.

**The series is complete.** Its own summary is in episode 5: five exact
statements about the chain rule's hypotheses, all true, all largely inert, and
in every case the finding that mattered was one question further in and
smoother than the headline.

## Uncertainty for Language Models, Taught Through What Breaks

One object: **the guarantee you are actually getting**. Every claim about
uncertainty in machine learning is a guarantee with a quantifier in it, and
the quantifier is where the trouble lives. Conformal prediction promises
coverage *on average over the test distribution*. Calibration error is an
*estimator*, with a bias that depends on choices you made about bins rather
than on your model. Temperature scaling fixes *numbers*, and provably cannot
fix predictions.

Thesis: **each of the five gaps below is a theorem or a measurement, not a
matter of practice.** Split conformal's coverage is exact, finite-sample and
distribution-free, and it delivers 82% where the model is least sure against
95% where it is most sure. A model calibrated by construction scores an ECE of
0.037 at a thousand points and fifteen bins, which is the size of the
improvement recalibration papers report. Dividing every logit by the same
positive number cannot reorder anything within a row — accuracy is bit
identical from *T* = 0.5 to *T* = 3 — and it reorders 7.2% of confidence
comparisons *between* rows, which is what abstention uses.

The measurements are made on the same **816,128-parameter character-level
transformer** as the calculus series, with the weights and the training script
committed, plus models calibrated by construction where a known-zero answer is
needed. Every guarantee is a theorem and holds at any scale; every frequency
is a property of that model and is reported as such.

| # | Episode | The guarantee and the gap |
|---|---|---|
| 1 | 90% Coverage Is a Promise About Averages, Not About You | split conformal delivers 0.896 against a finite-sample guarantee of 0.9001 — and 0.823 in the least-confident fifth against 0.948 in the most-confident, with set sizes of 11.6 and 1.3 labels. The shortfall lands exactly on the cases you would escalate |
| 2 | Your Calibration Error Is Mostly Your Bin Count | ECE is a biased estimator of a quantity that is zero for a calibrated model. Measured on models calibrated by construction: 0.120 at n = 200, 0.037 at n = 1,000, 0.005 at n = 20,000, scaling as the square root of bins over n — and equal-mass bins do not remove it |
| 3 | Temperature Scaling Cannot Change What You Predict | a strictly increasing map on every logit leaves every within-row ordering intact, so accuracy is identical to the bit across *T* — while ECE moves from 0.026 to 0.375. But the *between*-row confidence ordering is not invariant: at *T* = 2, 7.2% of pairs swap and a fifth of the most-confident percentile leaves it |
| 4 | Overconfidence Arrives When the Model Stops Improving | calibration tracked across a training run rather than at its end, to ask whether overconfidence is a property of the architecture or a symptom of overfitting — and why a model at validation loss 1.573 needs no temperature scaling at all |
| 5 | The Split You Chose Is Hiding How Variable Your Coverage Is | the realised coverage over 200 calibration splits has a standard deviation 1.5 times what an exchangeable argument predicts — and splitting by sequence rather than by row, which is the *correct* thing to do, makes it 2.0 times, because row-wise splitting was leaking between calibration and test |

Episodes 1, 2, 3 and 5 already have every number above measured and pinned in
`tests/test_uncertainty.py`; episode 4 needs a training run with checkpoints.
The series exists because the previous two ended the same way — an exact
statement that was true and inert — and a guarantee with a quantifier in it is
the cleanest place left to look for that pattern.
