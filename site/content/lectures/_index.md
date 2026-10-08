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

The longest course here, thirty episodes, and the broadest: the standard
machine-learning curriculum from how a model is scored, through linear models,
trees and ensembles, reading what a model learned, training a network, and what
generalisation looks like in one. Each episode takes **one sentence** that the
curriculum states as fact and measures where it stops being true. Some of those
sentences will survive measurement; when one does, the episode says so.

The data is either bundled with scikit-learn — the UCI handwritten digits and
the Wisconsin diagnostic breast cancer set, both CC BY 4.0 — or simulated with a
known answer, so a reader rebuilds every number offline. The network episodes
run on a CPU, on models small enough to train many times, because most of their
claims are about what happens *across* training runs.

For episodes not yet published the table gives the sentence under test and how
it will be measured, not a result. Their results go in when they are measured,
and not before; the last series was a reminder of why.

### Arc I — What a score means

| # | Episode | The sentence, and what measurement did to it |
|---|---|---|
| 1 | Your Test Score Has an Error Bar, and It Is Wider Than Your Gain | *the model with the higher test score is better*. On 360 test digits one score's 95% interval is about ±1.4 points, wider than the gap between the top two of three classifiers. Repeated splits of one pool re-measure the test-set noise with a finite-population correction, not training variation. Paired on all 1,797 images, 3-NN beats an RBF SVM by 0.6 points at McNemar p = 0.06; settling it needs about 3,400 images |
| 2 | Cross-Validation Estimates the Error of a Model You Did Not Train | *CV estimates how well your model will do*. On simulated least squares, where each fitted model's error is exact, the correlation between the CV estimate and it is −0.01; on digits against a fixed test set, −0.21. CV estimates the procedure's average error, and its naive 90% interval covers the fitted model's error 80% of the time |
| 3 | Not Every Leak Leaks, and the Ones That Do Can Be Measured | *fit every preprocessing step inside the folds*. Right, and violations range from nothing to a third of the scale: standardising and imputing on all rows leak under 0.1 points; selecting 20 of 5,000 noise features turns coin flips into 0.88; on real data with strong features the same selection leaks nothing; duplicates leak 0.9 points for logistic regression and 3.7 for 1-NN |
| 4 | A Learning Curve Fitted Early Promises More Than More Data Delivers | *a learning curve tells you whether more data will help*. On a simulated task grown to 25,600 rows with a Bayes error of 0.268, the power law a n^-b fitted to a pilot study was optimistic in all 9 fits and predicted errors below the Bayes error in 4; on digits, optimistic in all 9 again. Adding a floor fixes curves that have bent and not those still falling: the boosted trees' fitted floor went from above what they reach to below Bayes. Logistic regression is best at 50 digits and worst at 1,200 |
| 5 | — | *resampling fixes class imbalance*. Over- and under-sampling against class weights and a moved threshold, on ranking, calibration and the decision they produce |

### Arc II — Linear models, pushed

| # | Episode | The sentence under test |
|---|---|---|
| 6 | — | *logistic regression finds the maximum-likelihood coefficients*. On separable data there are none: the coefficient norm and the direction gradient descent heads in, measured |
| 7 | — | *regularisation strength is a property of the model*. The same penalty on features in different units, and which features lasso keeps |
| 8 | — | *lasso selects the relevant features*. Selection frequency across bootstraps when relevant features are correlated |
| 9 | — | *test error is bias squared plus variance, and the trade-off is a U*. Both terms computed exactly on simulated data, across model complexity |
| 10 | — | *more parameters than data points means overfitting*. Minimum-norm least squares on random features, swept through p = n |

### Arc III — Trees and ensembles

| # | Episode | The sentence under test |
|---|---|---|
| 11 | — | *a decision tree is interpretable*. How often the root split and the tree's shape survive a bootstrap of the rows |
| 12 | — | *impurity importance ranks what the forest relies on*. A pure-noise column with many distinct values, ranked |
| 13 | — | *out-of-bag error is free cross-validation*. OOB, CV and test error across forest sizes |
| 14 | — | *boosting overfits if you run it long enough*. Test error against rounds, with and without shrinkage |
| 15 | — | *tree ensembles can approximate any function*. Predictions outside the training range on data with a trend |

### Arc IV — Reading what a model learned

| # | Episode | The sentence under test |
|---|---|---|
| 16 | — | *permutation importance measures how much a model needs a feature*. Two correlated copies of one feature, and the rows permutation invents |
| 17 | — | *a partial dependence plot shows a feature's effect*. PDP against accumulated local effects when features are correlated |
| 18 | — | *SHAP values attribute a prediction to its features*. Exact Shapley values on a small model, under different background distributions |
| 19 | — | *the model with the better test score learned the task better*. A planted shortcut in the digits, and what happens when it is removed |
| 20 | — | *two models with the same accuracy make the same predictions*. Disagreement among equally accurate models |

### Arc V — Training a network

| # | Episode | The sentence under test |
|---|---|---|
| 21 | — | *initialisation only needs to break symmetry*. Activation and gradient scale through depth for different initial scales |
| 22 | — | *learning rate and batch size are separate knobs*. The linear scaling rule, and where it stops holding |
| 23 | — | *Adam converges faster and generalises as well as SGD*. Both, measured on the same small problems over many seeds |
| 24 | — | *batch normalisation normalises each example*. One example's prediction as the rest of its batch changes |
| 25 | — | *dropout at test time is approximated by scaling the weights*. Weight scaling against averaging many dropout masks |

### Arc VI — What generalisation looks like in a network

| # | Episode | The sentence under test |
|---|---|---|
| 26 | — | *a network that generalises could not have fit noise*. The same network on true and on random labels |
| 27 | — | *once training accuracy reaches 100%, learning is over*. Modular addition, trained long past that point, with and without weight decay |
| 28 | — | *data augmentation adds data*. Label-preserving against label-breaking augmentations on the digits |
| 29 | — | *covariate shift costs accuracy*. Accuracy and calibration under controlled shifts, and importance weighting |
| 30 | — | *scaling laws predict performance*. Power laws fitted on small character models, checked on larger ones |

Episodes 1 to 4 are published. Two of them changed shape while being measured:
episode 1's repeated splits turned out to measure the pool rather than the
training sets, and episode 3's feature-selection leak, enormous on noise,
vanished on real data where the true features win the selection anyway — which
is what turned a list of leaks into a rule for sizing them.
