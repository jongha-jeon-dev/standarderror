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
0.035 at a thousand points and fifteen bins, and on the committed model at that
size 80% of the measured ECE is that floor. Dividing every logit by the same
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
| 2 | Your Calibration Error Is Mostly Your Bin Count | ECE is a biased estimator of a quantity that is zero for a calibrated model: 0.085 at n = 200, 0.035 at n = 1,000, 0.008 at n = 20,000 on models calibrated by construction, scaling as the square root of bins over n with a slope of 0.503. On the committed model at n = 1,000, 80% of the measured ECE is that floor, and a real miscalibration is detected in 23% of subsets. Subtracting each bin's binomial variance removes the floor on average and leaves single estimates so noisy that a third come out below zero |
| 3 | Temperature Scaling Cannot Change What You Predict | a strictly increasing map on every logit leaves every within-row ordering intact, so accuracy is identical to the bit across *T* — while ECE moves from 0.026 to 0.375. But the *between*-row confidence ordering is not invariant: at *T* = 2, 7.2% of pairs swap and 22.5% of the most-confident percentile leaves it. For abstention that matters by score: max probability barely notices, entropy's AURC is 44% worse at *T* = 3, and logit margin cannot move |
| 4 | Overconfidence Arrives Before the Model Stops Improving | three training runs of the same model, a temperature fitted at every checkpoint. Every run starts slightly underconfident and is overconfident long before validation loss bottoms out — fitted *T* 1.20 at the long run's best checkpoint, 1.38 on a tenth of the data and on a second seed — while on its own training text the fitted temperature never exceeds 1.02. Past the minimum, validation NLL passes the uniform-guess loss with accuracy unchanged, and one temperature takes it from 4.68 to 2.43 |
| 5 | The Split You Chose Is Hiding How Variable Your Coverage Is | episode 1's 1.5x excess spread was mostly a missing term: realised coverage is measured on a finite test set, and with that binomial variance added a row split shows 1.06x, inside the range i.i.d. pools give. Splitting whole sequences, as a deployment does, gives 1.42x — predicted at 1.43x from the within-sequence correlation — and a two-SD alarm meant to fire 2.3% of the time fires 9.5% |

All five episodes are published, and the last two each overturned the
syllabus. Episode 4 was planned as "overconfidence arrives when the model stops
improving"; measured, it arrives well before, and episode 3's 1.10 was a
property of stopping at six passes, not of the architecture. Episode 5 corrects
episode 1: the excess it promised was mostly a variance term I left out, and
the real one appears only when whole sequences are split. Episode 3 finds the
fitted temperature, 1.10, is also max probability's best temperature for
abstention — and the wrong direction for entropy, whose best is 0.6. Episode 2 ends on a trade rather than a fix:
debiasing ECE removes its floor and its dependence on bin count, and on a
thousand predictions makes it about as noisy as the quantity it estimates —
so for choosing between two models, NLL picked the better-calibrated one 87%
of the time against 68% for fifteen-bin ECE.

Episode 1 reverses itself: the uneven conditional
coverage above belongs to the **score**, not to conformal prediction.
Randomised adaptive sets hit the same marginal level and cut the spread from
0.125 to 0.017 — seven-fold — for 20% more labels per set and a 0.9% chance of
returning an empty set, which the simple score never does. So the marginal
guarantee is not hiding a defect; it is declining to make a choice on your
behalf.
The series exists because the previous two ended the same way — an exact
statement that was true and inert — and a guarantee with a quantifier in it is
the cleanest place left to look for that pattern.

## School Maths, Taught Through What a Model Gets Wrong

One object: **arithmetic a ten-year-old can do**. Not because it is hard, but
because it is the only domain where the correct answer, the correct *method*,
and the point at which a curriculum introduces each new idea are all written
down in advance. Carrying, place value, the order of operations, inverting an
operation — a school syllabus is a list of the places arithmetic stops being
digit-local, and that turns out to be a very good list of the places a small
transformer stops working.

Thesis: **the model's failures land on the curriculum's boundaries, and almost
none of them are the failure you would predict.** Order of operations scores
0.40 — and not one failure is the left-to-right mistake a person makes; every
single one is a wrong two-digit product inside a correctly applied rule.
Linear equations score 0.988, better than multiplication, until the answer is
asked to leave the forty-one integers training drew from, at which point it is
0.030. Carrying is learned at a width rather than as a rule: 0.99 on the carry
chains shown at three digits, 0.14 on the ones that were not. And writing the
answer least-significant-digit-first — the order the algorithm actually runs
in — recovers a large part of that with nothing else changed.

The measurements are made on an **804,096-parameter character-level
transformer**, the same architecture as the calculus and uncertainty series on
an eighteen-character vocabulary, trained on generated arithmetic. There is no
corpus to fetch: `standarderror/schoolmath/curriculum.py` is deterministic in
its seed, so a reader rebuilds the exact training stream from the source. Both
checkpoints and their verified hashes are committed.

| # | Episode | The idea and the break |
|---|---|---|
| 1 | The Only Difference Between These Two Models Is the Order of the Lines | two models identical in architecture, seed, steps and problems; one adds and the other reverses the line above it, including when that line is deliberately wrong. Their held-out losses differ by 0.03 nats and never diverge, because a shortcut built from *ordering* survives any random split of the rows |
| 2 | Carrying Is the First Thing a Curriculum Teaches and the First Thing to Go | 0.996 and 0.992 at carry chains of 0 and 1, 0.564 at 2, 0.136 at 3 — chains held out at three digits and shown constantly at two. Writing the answer in computation order recovers it to 0.740 and 0.324, same weights, same sums |
| 3 | A Fourth Digit Is Not a Harder Problem, It Is a Different One | accuracy goes to exactly zero at four digits, and the answers are not even the right *length* — 0.7% of them. The failure is place value, not addition, and it is the clearest case in the series of a rule that was never learned as a rule |
| 4 | It Knows Times Tables and Cannot Multiply | 0.93 at one digit by one, 0.95 at three digits by one, and 0.497 at two by two. Difficulty tracks the *narrow* operand, so "three-digit multiplication" scores above "two-digit" — and the 0.40 on order of operations is entirely this, not precedence |
| 5 | It Solves Equations It Cannot Solve | linear equations at 0.988, above multiplication, on a task that requires undoing an operation rather than performing one. Ask for an answer outside the forty-one integers training drew from and it is 0.030 just outside and 0.000 far outside. It was classifying, not solving |

Episode 1 is published, and it is the methodological one: the shortcut it
describes was mine, found in my own generated corpus, and the transferable
part is not "shuffle your data" — everybody shuffles, and the standard kind
operates on the wrong thing — but that varying the context is a cheap
diagnostic no loss curve can replace. Episodes 2 to 5 have every number above
measured and pinned in `tests/test_schoolmath.py`.

The series exists because the three before it all needed a domain where
"correct" is not a matter of degree. Here it is not: 579 is the answer or it
is not, and the model either carried or it did not.

## Machine Learning, Taught Through What Breaks

The longest course here, fifty-five episodes, and the broadest: the standard
machine-learning curriculum from how a model is scored, through linear models,
trees and ensembles, reading what a model learned, training a network, and what
generalisation looks like in one, to the routine steps around a model, where
the rows came from, how models are combined and compared, what happens
when predictions are turned into decisions, and how a language model somebody
else trained is graded. Each episode takes **one sentence** that the
curriculum states as fact and measures where it stops being true. Some of those
sentences will survive measurement; when one does, the episode says so.

The data is either bundled with scikit-learn — the UCI handwritten digits and
the Wisconsin diagnostic breast cancer set, both CC BY 4.0 — or simulated with a
known answer, so a reader rebuilds every number offline. The network episodes
run on a CPU, on models small enough to train many times, because most of their
claims are about what happens *across* training runs. The language-model
episodes use the repository's own small arithmetic transformers, whose every
answer can be checked exactly.

Each row gives the sentence under test and what measurement did to it. The
results went in only once they were measured; the last series was a reminder
of why.

### Arc I — What a score means

| # | Episode | The sentence, and what measurement did to it |
|---|---|---|
| 1 | Your Test Score Has an Error Bar, and It Is Wider Than Your Gain | *the model with the higher test score is better*. On 360 test digits one score's 95% interval is about ±1.4 points, wider than the gap between the top two of three classifiers. Repeated splits of one pool re-measure the test-set noise with a finite-population correction, not training variation. Paired on all 1,797 images, 3-NN beats an RBF SVM by 0.6 points at McNemar p = 0.06; settling it needs about 3,400 images |
| 2 | Cross-Validation Estimates the Error of a Model You Did Not Train | *CV estimates how well your model will do*. On simulated least squares, where each fitted model's error is exact, the correlation between the CV estimate and it is −0.01; on digits against a fixed test set, −0.21. CV estimates the procedure's average error, and its naive 90% interval covers the fitted model's error 80% of the time |
| 3 | Not Every Leak Leaks, and the Ones That Do Can Be Measured | *fit every preprocessing step inside the folds*. Right, and violations range from nothing to a third of the scale: standardising and imputing on all rows leak under 0.1 points; selecting 20 of 5,000 noise features turns coin flips into 0.88; on real data with strong features the same selection leaks nothing; duplicates leak 0.9 points for logistic regression and 3.7 for 1-NN |
| 4 | A Learning Curve Fitted Early Promises More Than More Data Delivers | *a learning curve tells you whether more data will help*. On a simulated task grown to 25,600 rows with a Bayes error of 0.268, the power law a n^-b fitted to a pilot study was optimistic in all 9 fits and predicted errors below the Bayes error in 4; on digits, optimistic in all 9 again. Adding a floor fixed logistic regression and, from 1,600 rows, 15-NN, but not the boosted trees, still falling fastest: their fitted floor went from above what they reach to below Bayes. Logistic regression is best at 50 digits and worst at 1,200 |
| 5 | Resampling Moves the Threshold and Breaks a Logistic Model's Probabilities | *resampling fixes class imbalance*. At 2% prevalence, oversampling, SMOTE and class weights left a logistic model's AUC unchanged to within 0.002 — never better — multiplied its probabilities about elevenfold, and bought recall the plain model gets with a threshold of 0.021 at the same precision. On boosted trees they lowered the AUC by about 0.01; SMOTE lost 0.046 of average precision, while the plain trees' too-low probabilities improved under oversampling and class weights (log loss 0.102 to 0.087) |

### Arc II — Linear models, pushed

| # | Episode | The sentence under test |
|---|---|---|
| 6 | On Separable Data, Your Logistic Coefficients Are Your Stopping Rule | *logistic regression finds the maximum-likelihood coefficients*. On separable data there are none. All 50 half-splits of the breast-cancer data were separable; tightening the solver's tolerance from 1e-2 to 1e-10 grew the coefficient norm from 2.35 to 1,289, raised test log loss from 0.079 to 1.575 and lowered accuracy from 0.982 to 0.954. Gradient descent heads towards the max-margin direction of its own problem, intercept in the norm and 5.9° from scikit-learn's SVM, its norm growing like log t |
| 7 | Change a Feature's Units and the Lasso Changes Its Mind | *regularisation strength is a property of the model*. Re-expressing one diabetes feature at a time in other units flipped the lasso's decision on 7 of 10 features while cross-validated error moved by at most 5.3%. Standardising is a choice too: a rare binary feature with a fixed effect was kept in 60% of fits standardised and 17.5% raw |
| 8 | Cross-Validated Lasso Is Tuned to Predict, Not to Select | *lasso selects the relevant features*. At the CV minimum it kept 2.8 of 7 pure-noise features and an irrelevant feature about a third of the time at any correlation. The 1-SE rule dropped the noise, not the confusion: at correlation 0.99 it kept the stand-in 38% of the time and the real feature 70%. On diabetes bootstraps, age was kept in 82% of fits, 37% positive and 45% negative |
| 9 | The Bias–Variance U Belongs to a Knob, Not to Complexity | *test error is bias squared plus variance, and the trade-off is a U*. Computed exactly, the three terms matched measured error to 0.003. k in k-NN draws the U, bottom at k = 8. Adding trees to a forest does not: bias stays near zero and variance falls monotonically towards a floor of ρσ², ρ = 0.46 |
| 10 | The Second Descent Is Real, and a Ridge Penalty Makes It Unnecessary | *more parameters than data points means overfitting*. Minimum-norm least squares on random features peaked at p = n for 50, 100 and 200 rows — about 80 times the best small model's error at n = 100 — and fell again beyond it, but never below the best model under the threshold, even without label noise. A ridge penalty removed the peak and was best around p = n |

### Arc III — Trees and ensembles

| # | Episode | The sentence under test |
|---|---|---|
| 11 | Resample the Rows and the Tree Tells a Different Story | *a decision tree is interpretable*. Over 200 bootstrap refits of a depth-3 tree on breast cancer, accuracy held at 0.914 ± 0.015 while the full-data root was the root in only half, the 200 resamples put 194 different feature sets in the top three levels, none of them the full-data tree's, and 36% of test tumours got a different diagnosis from some tree. What survived was coarser: every root measured the size or concavity of the nuclei |
| 12 | Impurity Importance Rewards Features With Many Places to Split | *impurity importance ranks what the forest relies on*. With five features and no signal, MDI gave the continuous one 42% and the binary one 5%; with a real effect on the binary feature alone, MDI never ranked it first and ranked it last in 19 of 20 datasets, while permutation importance ranked it first in 19. On breast cancer, where real signals are strong, both put added noise columns at the bottom — the bias hides when it matters least |
| 13 | Out-of-Bag Error Is Free, Pessimistic When Small and Blind to Copies | *out-of-bag error is free cross-validation*. With 500 trees, within half a point of a held-out set on breast cancer and digits, about as close as five-fold CV. With 5 trees, 2.7 times the held-out error: each row is judged by the third of the forest that did not see it, and scikit-learn scores the tenth of rows with no vote as the first class, which supplies 46% of that figure. With every training row duplicated, exactly 0 against a held-out 2.7% |
| 14 | Boosting Overfits Its Probabilities Long Before Its Accuracy | *boosting overfits if you run it long enough*. On breast cancer, test error never turned up after round 30 of 2,000; test log loss was lowest within the first few dozen rounds and 2.6 times worse at 2,000. Each tenfold cut in the learning rate moved the best round about tenfold later; the cut from 1 to 0.1 lowered the best log loss, the cut from 0.1 to 0.01 did not. With 10% flipped labels, error rose under two points and log loss more than tripled |
| 15 | A Forest Cannot Predict Beyond the Largest Value It Has Seen | *tree ensembles can approximate any function*. Trained on x in [0, 1], forest and boosting were the best single models inside the range, and the forest 22 times worse at x in [1.5, 2]; a linear trend plus a forest on the residuals was best everywhere. On diabetes a forest narrowly won a random split and lost by 40% on the highest-bmi patients |

### Arc IV — Reading what a model learned

| # | Episode | The sentence under test |
|---|---|---|
| 16 | Permutation Importance Measures the Model, Not What the Data Need | *permutation importance measures how much a model needs a feature*. Shuffling bmi cost 0.168 of R²; refitting without it cost 0.043. With an exact copy the two columns' importances summed to a quarter of bmi's alone. Shuffling s1 put 29% of rows outside the region holding 99% of real (s1, s2) pairs |
| 17 | Partial Dependence Asks the Model About Rows That Do Not Exist | *a partial dependence plot shows a feature's effect*. With y = x1 + 2x2² and x2 a noisy copy of x1, a forest's PDP slope for x1 fell from 0.98 with independent inputs to 0.30 at correlation 0.985 (truth 1); ALE stayed between 0.92 and 1.03 for both forest and boosting |
| 18 | A SHAP Value Is an Answer to "Compared With What?" | *SHAP values attribute a prediction to its features*. Exact Shapley values for a boosted model on five breast-cancer features: switching the background from all tumours to benign ones changed the top feature for 58% of tumours, and the feature-means shortcut flipped 19% of signs with nearly the same base value |
| 19 | A Planted Shortcut Wins the Test Set and Loses Everywhere Else | *the model with the better test score learned the task better*. A marker pixel on 90% of four digit classes: the shortcut model beat a clean one on a test set built the same way (0.986 against 0.971), fell to 0.947 on clean images and 0.663 with the markers moved. Permutation importance ranked the marker first of 64 |
| 20 | Same Accuracy, Different Answers | *two models with the same accuracy make the same predictions*. On a task where the Bayes rule errs on 27.6% of the test rows, five networks differing only in seed spanned one point of accuracy and disagreed on up to 22.5% of rows; 43% got both labels from some seed. On breast cancer, with errors near 2%, near-equal models disagreed on 1.4%: disagreement is bounded by the sum of error rates |

### Arc V — Training a network

| # | Episode | The sentence under test |
|---|---|---|
| 21 | Initialisation Must Break Symmetry, at the Right Scale | *initialisation only needs to break symmetry*. With every weight equal, a 30-layer network's units stayed identical through training and it scored 0.197. Random weights broke symmetry and still failed at the wrong scale: with ReLU each layer multiplied the activations' scale by 0.719 at g = 1 and 1.438 at g = 2 (theory g/√2), and those networks scored 0.13 and 0.10 against 0.919 at g = √2. With 3 layers every random scale trained above 0.94 |
| 22 | Learning Rate and Batch Size Are One Knob, Until the Rate Hits a Wall | *learning rate and batch size are separate knobs*. With plain SGD and a fixed number of examples, runs with the same ratio of rate to batch reached the same training loss to within 2.3%, from batches of 8 to 512, away from divergence. At a rate of 3.072 training failed (mean loss above 1) at batches of 128 and 512 alike, diverging at 128 and degrading at 512; stopped at 1.536, they reached training losses 3 and 55 times the batch of 32's |
| 23 | Adam Is a Little Faster and No Better | *Adam converges faster and generalises as well as SGD*. Tuned on validation data and run over 20 splits, Adam and SGD with momentum differed by +0.09 points on the digits and +0.05 on a hard task, within noise. At each one's fastest rate Adam reached a training loss of 0.01 in 6 epochs and SGD in 8; at the rates validation chose, SGD was faster. Adam's accuracy depended less on its rate |
| 24 | Batch Normalisation Normalises the Batch, Not the Example | *batch normalisation normalises each example*. In training mode one digit's label changed with its random batch companions in 45% of batches of 2 and 0.6% of batches of 64, and in 23% of batches of 16 drawn from one class. Trained in batches of 2, the normalised network scored 0.863 against 0.972 without normalisation; from batches of 16 up it scored 0.5 and 0.8 points higher (three seeds) |
| 25 | Weight Scaling Keeps Dropout's Answers, Not Its Doubts | *dropout at test time is approximated by scaling the weights*. Against the average of 300 masks, weight scaling gave the same label on 99.7% of digits at the same accuracy but was more confident (mean top probability 0.985 against 0.954), with 28% higher log loss. On a hard task the labels agreed on 94.3%, the disagreements falling on coin-flip rows. One sampled mask was worse than both |

### Arc VI — What generalisation looks like in a network

| # | Episode | The sentence under test |
|---|---|---|
| 26 | The Network That Memorises Noise Learns the Signal First | *a network that generalises could not have fit noise*. The same network fitted true digit labels in 41 full-batch steps (test 0.968) and random ones in 73 (test 0.120). With 20% of labels randomised it fitted the unchanged rows first, test accuracy peaking at 0.928, then memorised the changed ones and ended at 0.836 |
| 27 | Perfect Training Accuracy Was Not the End of Learning | *once training accuracy reaches 100%, learning is over*. On (a + b) mod 97, a network trained on 40% of the pairs fitted them at step 200 with test accuracy 0.000, and reached 99% on the unseen pairs at step 4,900 with training accuracy at 100% throughout, while weight decay shrank its parameter norm from 112 to 76. Without weight decay, test accuracy was 0.1% after 10,000 steps; with 60% of the pairs, 99% came at step 1,500 |
| 28 | Augmentation Adds an Assumption, Not Data | *data augmentation adds data*. From 150 training digits, four times more real digits added 4.1 points; four shifted copies of each cost 2.7 on the test set and added 32 on test digits moved by a pixel, where the real digits added 1.6. Noisy copies changed nothing; mirrors and half-turns, which change some digits' labels, cost 9 and 12 |
| 29 | Covariate Shift Charges for Misspecification | *covariate shift costs accuracy*. With the inputs moved two standard deviations and P(y \| x) fixed, the Bayes accuracy stayed between 0.83 and 0.84; a flexible model lost under 3 points and a linear one fell from 0.816 to 0.536. Importance weighting moved the line towards the best one for the test inputs, with lower log loss at every non-zero shift but lower accuracy at a shift of 1, from an effective 6% of the sample |
| 30 | A Power Law Fitted on Small Models Cannot See the Floor | *scaling laws predict performance*. Eight character transformers trained on the same text for the same 2,000 steps: a law fitted to the five smallest predicted the sixth to 0.04% and the seventh and eighth 3.4% and 8.2% too low (training loss missed the same way, by 5.2% and 11.0%), as characters seen per parameter fell to 2. A floor fitted to the five small models came out at zero; fitted to all eight, at 0.86. A law fitted to three models hit the largest only by cancelling two errors |

### Arc VII — Around the model

| # | Episode | The sentence under test |
|---|---|---|
| 31 | Principal Components Keep Variance, Not Relevance | *the first principal components keep the information that matters*. On raw breast-cancer measurements the first component held 98.2% of the variance and was led by worst area, the feature with the largest units; the 95% rule kept one component raw and ten standardised. On diabetes the fourth component, with 9.6% of the variance, explained 12.8% of the target, more than the second and third together; partial least squares reached with two components the R² that PCA needed four for |
| 32 | k-Means Finds k Pieces Whether or Not There Are Clusters | *k-means finds the clusters in the data*. It recovered four separated blobs (ARI 1.00) and the silhouette chose k = 4. On uniform noise the silhouette also chose 4, scoring 0.40, more than twice its best on the digits' ten real classes; on one large group with two small ones, and on two long parallel groups, k-means at the true k scored ARI 0.09 and 0.00 |
| 33 | The Curse of Dimensionality Counts the Dimensions the Data Use | *in high dimensions, nearest neighbours stop meaning anything*. For uniform points the nearest/median distance ratio rose from 0.03 in 2 dimensions to 0.94 in 1,024. The digits' 64 pixels had the ratio of 8 uniform dimensions and 5-NN scored 0.986; 1,024 appended noise columns dropped it to 0.659, while logistic regression fell only to 0.871 |
| 34 | Recalibration Fixes the Average, Given Enough Rows | *Platt scaling or isotonic regression fixes a model's probabilities*. On 50 calibration rows Platt scaling cut an overconfident model's log loss from 0.748 to 0.612 and isotonic regression raised it to 0.761; an already-calibrated model was made worse by either method on 500 rows or fewer. A model calibrated to 0.005 overall was off by 0.19 in a group holding 30% of the rows |
| 35 | The Best Cross-Validated Score Is the Search's Best Luck | *the best cross-validated score estimates how well the tuned model will do*. On 200 rows, untuned cross-validation was 0.013 pessimistic; the best of 64 configurations was 0.023 optimistic, and its model did no better on fresh data than the untuned default (0.647 against 0.659). On coin-flip labels the best of 64 scores averaged 0.575 for models right half the time; nested cross-validation gave 0.524 |

### Arc VIII — Where the rows come from

| # | Episode | The sentence under test |
|---|---|---|
| 36 | Shuffling a Time Series Is Safe Until the Errors Remember | *never shuffle a time series for cross-validation; split it forward*. On a stationary autoregression with its lags as features, shuffled folds were within 3% of the error on the next 250 steps for linear regression, 5-NN and a forest. With residuals of persistence 0.9 they were 13% optimistic for the forest, at 0.99 22%, and with a drifting level and a time index 25%. Forward splits were 18–20% pessimistic at persistence 0.99 and 11–14% for the drifting forests; contiguous blocks were the closest scheme in five of the seven hard settings |
| 37 | Rows From the Same People Are Not New Data | *more rows means more data, and row-level cross-validation measures how the model will do*. 2,000 rows from 20 simulated subjects: row-level CV scored 5-NN at 0.825 and it scored 0.685 on new people; row-level scores were higher the fewer people the same rows came from, while true accuracy fell. Subject-level folds came within 0.025 at 20 subjects. 6,400 rows from 20 subjects trained logistic regression to 0.788, about what 100 rows from 100 subjects gave |
| 38 | A Missing Value Is Information, Until the Process Behind It Changes | *fill missing values in as accurately as you can and model the rest*. On breast cancer with 30% of cells removed at random, the best imputation gave the best model. When large values went missing, mean imputation plus indicators (log loss 0.121) beat iterative imputation (0.143), the better imputer. When benign tumours' cells went missing more often, the indicator model scored 0.999, above the complete data's 0.980, and 0.817 once the gaps became random |
| 39 | The Applicants You Refused Are the Data You Do Not Have | *a model trained on the outcomes you observed predicts the outcomes you did not*. With approval on the model's features only and a correct model, it predicted 0.453 defaults among the refused against 0.432. When a loan officer also used information the model lacks, 0.137 against 0.486, with AUC unchanged. AUC on the accepted was off in both directions. A model on the 5% approved at random matched one trained on everyone |
| 40 | Target Encoding Fitted on the Training Rows Still Leaks, Into Training | *fit target encoding on the training rows only and it cannot leak*. A 1,000-level noise category encoded by per-level means took boosted trees from test AUC 0.763 to 0.640; the encoded column alone ranked its own training labels at 0.774 and fresh ones at 0.498. Shrinkage largely rescued logistic regression and not trees; cross-fitting matched dropping the column on noise to within 0.002 and kept the signal when there was one |

### Arc IX — Combining and comparing models

| # | Episode | The sentence under test |
|---|---|---|
| 41 | An Average of Models Beats the Average Model, Not the Best One | *averaging several models gives a better model than any of them*. Of 57 averaged combinations of six classifiers on the digits, 11 beat their best member and 8 fell below their members' mean. A single tree, certain about every digit, won all 819 disagreements with the SVM it was averaged with. The guarantee is against the mean member: on diabetes the average's error was the members' mean minus their spread, to 1e-12, and still worse than ridge alone |
| 42 | Stacking Learns to Pick the Best Model, Given Honest Predictions | *stacking learns the best way to combine models*. On in-sample predictions the meta-model gave 1-NN a weight of 13 and reproduced its accuracy exactly, the worst of four members. On out-of-fold predictions it chose the SVM and matched it, never more than 0.001 above it; at 100 rows it lost to the best member, and, in a pool without 1-NN, a plain average beat every member |
| 43 | Naive Bayes Counts Correlated Evidence Twice, and Only Its Probabilities Notice | *naive Bayes works well even though its independence assumption is wrong*. On breast cancer it trailed logistic regression by 4 points of accuracy and a factor of 8 in log loss, with 67% of its errors at above 0.99 confidence. Copying every feature ten times left the AUC of its log-odds unchanged to four decimals and nearly tripled its log loss; copying only the size features changed its answers |
| 44 | Cross-Validation Folds Are Not Independent Samples, and the t-Test Assumes They Are | *a paired t-test on cross-validation folds tells you whether one model is better*. Two algorithms equal by symmetry, 400 data sets: the ten-fold t-test found a difference in 14%, the 10 × 10 version in 58%. The corrected resampled t-test, 5x2cv and McNemar held 5%; with a real difference the corrected test was the most powerful of the three |
| 45 | A Test Set Answers Honestly Only the First Time | *a test set stays unbiased however often you look, if you never train on it*. A majority vote of random guesses kept for beating 0.5 scored 0.713 on a leaderboard of coin flips and 0.497 on the private labels. Noise features kept because they replicated on the holdout scored 0.687 there and 0.498 fresh. Forward selection guided by the holdout reported 0.690, above the true model's 0.676; fresh accuracy was 0.576 |

### Arc X — From prediction to decision

| # | Episode | The sentence under test |
|---|---|---|
| 46 | The Right Threshold Is a Cost Ratio, If the Probabilities Are Calibrated | *a classifier predicts the positive class above 0.5*. With a miss costing ten false alarms, flagging at 0.5 cost 1.4 to 2.0 per case, 2.9 to 4.0 times flagging everyone (0.497). The threshold 1/(1 + r) came within 0.003 of the best for recalibrated boosted trees and cost the overconfident raw ones 0.762. Tuned on 1,000 rows it lost to flagging everyone for three of five models at r = 20; even the true probabilities saved only 11% at r = 10 |
| 47 | The Model With the Best AUC Was Never the Best When Misses Cost Ten Times More | *the model with the higher AUC makes better decisions*. A forest had the best AUC in 10 of 10 training sets and the lowest cost at equal costs in 10. At r = 10 it was cheapest in none; boosted trees in 7 and logistic regression, last by AUC, in 3. Among the 2% of cases each rated safest, the forest's held 5.6% positives and logistic regression's 2.8% |
| 48 | What a Model Relies On Is Not What to Change | *the features a model relies on most are the levers*. A test result produced by the outcome carried 0.43 to 0.45 of AUC importance and moved predicted risk by 0.31 per standard deviation; changing it changes nothing. A habit sharing a cause with the outcome had a what-if of about −0.03 and no effect. The real lever, true effect 0.085, had a what-if of 0.024; adjusting for the common cause recovered 0.089 |
| 49 | Target the Effect, Not the Risk | *give the intervention to those most at risk*. Offering a retention deal to the highest predicted churners prevented about 21 churns per 1,000 customers, about the same as random targeting; ranking by the predicted effect from a randomised trial's two arms prevented 39 to 50, against a best possible 55. 11% of the highest-risk fifth were made more likely to leave, against 6% overall |
| 50 | A Rule That Chooses Its Own Data Learns Its Own Choices | *a model retrained on the data it collects corrects itself*. Inspectors sent where most violations had been found ended all in one district in 500 of 500 simulated years, the lower-rate one in 46%. Splitting by violations found was within 0.1 of the truth in 21%; dividing by visits made, in every run. Greedy on the per-visit rate ended in the worse district in 30%, and in 8% with 10% exploration |

### Arc XI — Grading a language model

| # | Episode | The sentence under test |
|---|---|---|
| 51 | A Benchmark Is a Sample of Questions | *the model with the higher benchmark score is the better model*. Ten small arithmetic transformers on benchmarks drawn from a 2,500-question pool: the models' 95% intervals covered their true scores about 95% of the time on average, and on 100 questions the best model still came first in half of the benchmarks. A real 0.038 gap was found by the paired test in 60% of 300-question benchmarks and by the unpaired one in 2%; five prompts per question gave a standard error 2.1 times the naive one |
| 52 | A Leaked Benchmark Shows in the Set, Not in the Item | *if a model has memorised a benchmark, its likelihood on the benchmark's items will show it*. Inserting 300 benchmark problems into training raised their score from a control mean of 0.551 to 0.800 at 32 insertions, with fresh problems unmoved. Item by item the model's likelihood caught 3% of leaked problems at 5% false flags; compared as a set with fresh problems, every run from two insertions on stood out from all five controls |
| 53 | The Prompt Is Part of the Score | *a benchmark score measures the model*. The main model multiplied correctly 0.735 of the time after twelve ordinary lines of context and 0.500 after none; the twin's lead on three-digit carries was 0.127 under one context and about zero under another; wrong answers in the examples changed nothing. The best of sixteen prompts on one set of problems scored below the prompts' average on a second set, except where the prompts really differed |
| 54 | Multiple Choice Grades the Wrong Answers You Offered | *scoring by the likeliest option measures the same thing as asking for the answer*. Three models that generated no four-digit sum correctly picked it from four options 0.79 to 0.98 of the time against random numbers; the main model scored 0.955, 0.395 or 0.315 depending on the distractors, and a 0.105 generation lead between two models on multiplication vanished under near-miss options |
| 55 | Every Judge Ranked Itself First | *a model can grade other models' answers in place of people*. Each of three models, judging all three by its own likelihood with a threshold calibrated on the truth, ranked itself first on every task, 9 of 9; the weakest, right on 1.3% of carries, passed 73% of its own. Forty labels with prediction-powered inference covered the truth about 94% of the time, where the judge alone covered it at most 66% |

All eleven arcs are published. Several episodes changed shape while being measured:
episode 1's repeated splits turned out to measure the pool rather than the
training sets, and episode 3's feature-selection leak, enormous on noise,
vanished on real data where the true features win the selection anyway — which
is what turned a list of leaks into a rule for sizing them. In Arc II, double
descent survived as a curve and lost as advice: the second descent was real
in every setting and won in none, and the ridge penalty that removed the peak
was best around p = n rather than exactly at it. In Arc III, impurity
importance's famous bias was overwhelming on a null design and invisible on
real data with strong signals, and out-of-bag error, sold as free
cross-validation, turned out to be about as accurate only once the forest was
large. In Arc V, the settings of training turned out less separate or less
exact than their descriptions: learning rate and batch size acted as one ratio
until a wall, dropout's weight scaling kept the answers and not the doubts, and
of Adam's reputation only the half about generalising as well as SGD survived
intact; its speed belonged to its learning rate. In Arc VI two episodes
ran the clock in opposite directions, a network that fitted signal before noise
and one that found a rule thousands of steps after fitting its training set,
and its last episode ended where episode 4 began: a power law fitted to small
models was optimistic, and the floor that bent it was invisible from the models
that fitted it. Arc VII measured the routine steps around a model and found
each answering a narrower question than the one it is used for: principal
components ranked by variance rather than relevance, k-means and the silhouette
rating uniform noise above the digits, distance failing because of
uninformative columns rather than many columns, calibration holding on average
and not within a group, and a search's best score reporting its luck. Arc VIII
went back to the rows, and found the rules about them right for narrower reasons
than their usual ones: shuffled folds failed a time series only when something
unseen changed slowly, rows from the same people counted as far fewer examples
than their number, missing values and missing labels both recorded decisions,
and a target encoding fitted on the training rows leaked into those rows. Arc IX
found combining and comparing models as easy to get wrong as fitting them: an
average that handed its decisions to its most confident member, a stack that
became 1-NN when shown in-sample predictions, tests that saw differences
between equal algorithms, and a test set worn out by being consulted. Arc X
followed predictions into decisions: a default threshold that assumed equal
costs, an AUC that chose the wrong model where the decision was made, a
feature importance that pointed at consequences rather than causes, a risk
ranking that did no better than random at finding the people a treatment
would help, and a rule that
learned its own choices back from the data they produced. Arc XI graded
language models somebody else trained and found the series' oldest problems
in new clothes: a leaderboard that misread overlapping error bars, a leaked
benchmark that a fresh set exposed and a per-item scan did not, a prompt that
was part of the score, a multiple-choice format that graded the distractors,
and a judge that preferred its own answers.
