"""Reading a trial's survival result from the outside, including the parts not printed.

`evidence.py` does this for a mean effect: given a reported effect and a p-value, the
sample size that must have produced it. This is the same trick for a time-to-event
endpoint, where the sample size that matters is not the number of patients but the
number of **events**, and where the arithmetic is unusually clean.

The one identity everything rests on
------------------------------------
For a two-arm comparison, the standard error of `log HR` is

    SE = 1 / sqrt(D * f),    f = p_e * (1 - p_e)

with `D` the total number of events and `p_e` the fraction of them in the treated
arm. This is normally introduced as Schoenfeld's (1981) large-sample approximation to
the log-rank test, which undersells where it comes from. For exponential survival
the maximum-likelihood variance of the log hazard ratio is `1/D1 + 1/D0` exactly,
and that is the same number:

    1/D1 + 1/D0 = D / (D1 * D0) = 1 / (D * p_e * (1 - p_e)).

So the formula is not a curve fit; it is a count identity wearing an approximation's
clothes, and `tests/test_survival.py` checks it against that closed form rather than
against another copy of itself.

Where it stops being exact is the Cox *partial* likelihood, whose information is
smaller than the parametric one when censoring is light and the effect large,
because the risk sets go lopsided — the treated arm survives, so late risk sets are
almost all treated and carry little comparison. In the regime trial reports actually
live in, where 50-90% of patients are censored, the two agree to about 1%; with no
censoring at all and a hazard ratio of 0.6 the gap reaches 10%. The tests pin both
ends.

Three consequences, in the order they get used:

* **The event count is recoverable from a printed confidence interval.** The
  interval's *width* carries no information about the effect — only about `D`. So
  `implied_events` reads a number out of a paper that most papers do not print, and
  it can be checked against the ones that do. On seven adjuvant melanoma trials it
  lands within a few percent.
* **The width of a future interval is knowable before the estimate is.** `ci_span`
  needs `D` and nothing else. Given a design, you can say how precise the answer
  will be without knowing what the answer is.
* **"Statistically significant" is a bound on the effect, not a fact about it.**
  `detectable_hr` inverts a test boundary into the largest hazard ratio consistent
  with having crossed it. This is the whole information content of an announcement
  that says an endpoint was met and prints no numbers — and it is a function of `D`,
  which such announcements also do not print.

What `p_e` should be, which is where this goes wrong
----------------------------------------------------
`p_e` is the split of *events*, not of *patients*. Under a treatment that works
those differ: an arm with a lower hazard contributes fewer events than its share of
the randomisation. Using the allocation fraction instead is the obvious mistake and
it is not small — at 2:1 allocation and a hazard ratio near 0.4 it misstates the
implied event count by more than 10%, in a direction that depends on the effect
size. `event_fraction` predicts the split from the hazard ratio, which is available
whenever the hazard ratio is, and `variance_factor` takes whichever of the three you
have: observed split (exact), predicted split (usable from a press release),
allocation only (last resort).

A confidence interval printed to two significant figures is the other error term,
and on real trial reports it is usually the larger one. `events_from_rounded_ci`
returns the interval of event counts consistent with a rounded interval, so a
recovered `D` can be quoted with the precision it actually has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = [
    "CoxFit", "EventCountRange", "allocation_fraction", "ci_span",
    "confidence_interval", "cox_binary", "detectable_hr", "event_fraction",
    "events_from_rounded_ci", "implied_events", "implied_events_from_p",
    "log_hr_se", "obrien_fleming_z", "pocock_z",
    "posterior_given_significance", "required_events", "simulate_arms",
    "variance_factor", "z_for_level",
]


# ---------------------------------------------------------------- geometry

def z_for_level(level: float = 0.95) -> float:
    """The z multiplier behind a stated confidence level, two-sided.

    Trial reports do not all use 95%. A group-sequential design that spends alpha at
    an interim reports the interval matching the alpha *left*, so KEYNOTE-054's
    primary analysis prints a 98.4% interval and CheckMate-238's a 97.56% one.
    Reading either as 95% understates the implied event count by a third, which is
    larger than every other error in this module put together.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly in (0, 1)")
    return float(norm.isf((1.0 - level) / 2.0))


def allocation_fraction(ratio: float = 1.0) -> float:
    """Fraction of *patients* in the treated arm for a `ratio`:1 randomisation."""
    if ratio <= 0:
        raise ValueError("allocation ratio must be positive")
    return float(ratio / (1.0 + ratio))


def event_fraction(hazard_ratio: float, ratio: float = 1.0) -> float:
    """Fraction of *events* expected in the treated arm.

    `p * h / (p * h + q)`: with equal exposure in the two arms, events accrue in
    proportion to patients times hazard. Exposure is only equal when censoring is
    administrative and the event probability is smallish — under long follow-up the
    surviving treated patients stay at risk longer and contribute more events than
    this predicts, so the formula is mildly conservative there. On the trials in
    `exp016` it is within 0.03 of the observed split except for the two with the
    longest follow-up, where it is short by 0.08.
    """
    if hazard_ratio <= 0:
        raise ValueError("hazard ratio must be positive")
    p = allocation_fraction(ratio)
    num = p * float(hazard_ratio)
    return float(num / (num + (1.0 - p)))


def variance_factor(*, ratio: float = 1.0, hazard_ratio: float | None = None,
                    event_split: float | None = None) -> float:
    """`p_e * (1 - p_e)`, from the best information available.

    Precedence is deliberate and is the accuracy story of this module:

    1. `event_split` — the observed fraction of events in the treated arm. Exact.
    2. `hazard_ratio` — predict the split from the effect. Available from a report
       that prints a hazard ratio, which is most of them.
    3. neither — the allocation fraction. Wrong by up to 12% in the implied event
       count on the trials tested, and wrong *as a function of the effect size*,
       which is the part that makes it dangerous rather than merely imprecise.
    """
    if event_split is not None:
        p = float(event_split)
        if not 0.0 < p < 1.0:
            raise ValueError("event split must lie strictly in (0, 1)")
    elif hazard_ratio is not None:
        p = event_fraction(hazard_ratio, ratio)
    else:
        p = allocation_fraction(ratio)
    return float(p * (1.0 - p))


def log_hr_se(events: float, **kw) -> float:
    """`1 / sqrt(D * f)`. Keyword arguments go to `variance_factor`."""
    if events <= 0:
        raise ValueError("need a positive number of events")
    return float(1.0 / np.sqrt(float(events) * variance_factor(**kw)))


# ---------------------------------------------------------------- inversions

def implied_events(hr_low: float, hr_high: float, *, level: float = 0.95,
                   ratio: float = 1.0, hazard_ratio: float | None = None,
                   event_split: float | None = None) -> float:
    """Event count implied by a printed confidence interval for a hazard ratio.

    `D = 4 z^2 / (f * (log U - log L)^2)`. Note what is *not* used: the point
    estimate. An interval's width is a statement about how much data there was and
    contains no information about the effect, which is why this works at all.
    """
    lo, hi = float(hr_low), float(hr_high)
    if not 0.0 < lo < hi:
        raise ValueError("need 0 < hr_low < hr_high")
    f = variance_factor(ratio=ratio, hazard_ratio=hazard_ratio,
                        event_split=event_split)
    z = z_for_level(level)
    return float(4.0 * z ** 2 / (f * (np.log(hi) - np.log(lo)) ** 2))


def implied_events_from_p(hazard_ratio: float, p_value: float, *,
                          ratio: float = 1.0, one_sided: bool = False,
                          event_split: float | None = None) -> float:
    """Event count implied by a hazard ratio and its p-value.

    `D = z^2 / (f * (log HR)^2)`. A p-value quoted as a bound ("P<0.001") gives a
    lower bound on `z` and therefore a lower bound on `D`; callers must say so.
    """
    h = float(hazard_ratio)
    if h <= 0:
        raise ValueError("hazard ratio must be positive")
    if h == 1.0:
        raise ValueError("a hazard ratio of exactly 1 implies no finite event count")
    if not 0.0 < p_value < 1.0:
        raise ValueError("p-value must lie strictly in (0, 1)")
    z = float(norm.isf(p_value if one_sided else p_value / 2.0))
    f = variance_factor(ratio=ratio, hazard_ratio=h, event_split=event_split)
    return float(z ** 2 / (f * np.log(h) ** 2))


@dataclass(frozen=True)
class EventCountRange:
    """Event counts consistent with a confidence interval printed to finite digits."""
    low: float
    high: float
    point: float

    @property
    def relative_width(self) -> float:
        """Half-width as a fraction of the midpoint — the recovery's own precision."""
        return float((self.high - self.low) / (self.high + self.low))


def events_from_rounded_ci(hr_low: float, hr_high: float, *, decimals: int = 2,
                           **kw) -> EventCountRange:
    """`implied_events`, carrying the uncertainty the printed rounding creates.

    A paper reporting "0.46 to 0.92" is reporting that the bounds lie in
    [0.455, 0.465] and [0.915, 0.925]. The widest and narrowest intervals in those
    boxes bracket the event count, and on two-decimal reports the bracket is a few
    percent wide — comparable to every modelling error here, and larger than most.
    Quoting a recovered event count to the unit is therefore false precision.
    """
    step = 0.5 * 10.0 ** (-decimals)
    lo, hi = float(hr_low), float(hr_high)
    widest = implied_events(lo - step, hi + step, **kw)      # widest CI -> fewest
    narrowest = implied_events(lo + step, hi - step, **kw)
    return EventCountRange(low=min(widest, narrowest), high=max(widest, narrowest),
                           point=implied_events(lo, hi, **kw))


# ---------------------------------------------------------------- forward

def confidence_interval(hazard_ratio: float, events: float, *,
                        level: float = 0.95, ratio: float = 1.0,
                        use_predicted_split: bool = True) -> tuple[float, float]:
    """The interval a design of this size will print around this estimate."""
    h = float(hazard_ratio)
    se = log_hr_se(events, ratio=ratio,
                   hazard_ratio=h if use_predicted_split else None)
    z = z_for_level(level)
    return float(h * np.exp(-z * se)), float(h * np.exp(z * se))


def ci_span(events: float, *, level: float = 0.95, ratio: float = 1.0,
            hazard_ratio: float | None = None) -> float:
    """Multiplicative width `U / L` of the interval, `exp(2 z SE)`.

    The estimate cancels. So a design fixes the *precision* of its answer before it
    has one, and a trial reporting "significant" without an interval has already
    told you how wide the interval will be, if it told you the event count.
    """
    return float(np.exp(2.0 * z_for_level(level)
                        * log_hr_se(events, ratio=ratio,
                                    hazard_ratio=hazard_ratio)))


def required_events(hazard_ratio: float, *, power: float = 0.9,
                    alpha: float = 0.05, ratio: float = 1.0,
                    one_sided: bool = False) -> float:
    """Events needed to detect `hazard_ratio` with this power. Schoenfeld's formula.

    `(z_alpha + z_beta)^2 / (f * (log HR)^2)`. Uses the allocation fraction for `f`,
    which is what a protocol's own calculation does, so the number matches the sort
    of figure a statistical analysis plan would state.
    """
    h = float(hazard_ratio)
    if h <= 0 or h == 1.0:
        raise ValueError("need a positive hazard ratio other than 1")
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie strictly in (0, 1)")
    za = float(norm.isf(alpha if one_sided else alpha / 2.0))
    zb = float(norm.isf(1.0 - power))
    f = variance_factor(ratio=ratio)
    return float((za + zb) ** 2 / (f * np.log(h) ** 2))


def detectable_hr(events: float, *, z: float, ratio: float = 1.0,
                  iterations: int = 40) -> float:
    """Largest hazard ratio below 1 that could have crossed a boundary at `z`.

    `HR = exp(-z / sqrt(D f))`. `f` depends on the hazard ratio through the event
    split, so this is a fixed point rather than a formula; it is a contraction and
    converges in a handful of passes.

    This is the whole quantitative content of "the trial met its primary endpoint"
    when no numbers accompany it. It is an upper bound: crossing a boundary means
    the observed effect was at least this large, and the announcement says nothing
    about how much larger.
    """
    if z <= 0:
        raise ValueError("boundary must be positive")
    h = 1.0
    for _ in range(int(iterations)):
        se = log_hr_se(events, ratio=ratio, hazard_ratio=h)
        new = float(np.exp(-z * se))
        if abs(new - h) < 1e-12:
            return new
        h = new
    return h


# ---------------------------------------------------------------- boundaries

def obrien_fleming_z(information_fraction: float, *, alpha: float = 0.05,
                     one_sided: bool = False) -> float:
    """Efficacy boundary at the **first** interim under O'Brien-Fleming spending.

    Lan-DeMets with the O'Brien-Fleming spending function
    `alpha*(t) = 2 (1 - Phi(z_{alpha/2} / sqrt(t)))` for the two-sided case. At the
    first look no alpha has been spent yet, so the boundary is exactly the quantile
    of the alpha spent so far and no recursion over previous looks is needed. For a
    *second* or later look this function is wrong and deliberately does not pretend
    otherwise — it takes no history argument.

    At the first look the whole construction collapses to `z_{alpha/2} / sqrt(t)`,
    which the code below arrives at the long way round so the spending function is
    visible rather than assumed. The practical point: at 60% information the
    two-sided 0.05 boundary is 2.53, not 1.96. An interim that clears its boundary
    has cleared a higher bar than a final analysis would set, so treating
    "significant at an interim" as "p just under 0.05" understates the effect that
    must have been seen.
    """
    t = float(information_fraction)
    if not 0.0 < t <= 1.0:
        raise ValueError("information fraction must lie in (0, 1]")
    a = float(alpha)
    z_full = float(norm.isf(a if one_sided else a / 2.0))
    spent = float(norm.sf(z_full / np.sqrt(t)))          # one-sided tail
    spent = spent if one_sided else 2.0 * spent
    return float(norm.isf(spent if one_sided else spent / 2.0))


def pocock_z(information_fraction: float, *, alpha: float = 0.05,
             one_sided: bool = False) -> float:
    """Same, for Pocock spending `alpha*(t) = alpha log(1 + (e-1) t)`.

    Kept beside the O'Brien-Fleming version because the two disagree most exactly
    where it matters here: at an early look Pocock spends far more alpha, so its
    boundary is lower and the effect implied by crossing it is smaller. Reporting a
    bound from a significance claim means choosing a spending function on the
    reader's behalf, and the choice is worth several hundredths of a hazard ratio.
    """
    t = float(information_fraction)
    if not 0.0 < t <= 1.0:
        raise ValueError("information fraction must lie in (0, 1]")
    spent = float(alpha) * np.log1p((np.e - 1.0) * t)
    return float(norm.isf(spent if one_sided else spent / 2.0))


# ---------------------------------------------------------------- the one bit

def posterior_given_significance(*, prior_hr: float, prior_log_se: float,
                                 events: float, z_boundary: float,
                                 ratio: float = 1.0,
                                 grid: np.ndarray | None = None) -> dict:
    """Update a prior on the hazard ratio using only "the boundary was crossed".

    The observation is not an estimate; it is the event `Z >= z_boundary`. Its
    probability under a true log hazard ratio `theta` is
    `Phi(-theta / SE - z_boundary)`, so the likelihood is a smooth function of
    `theta` and the update is an ordinary one — just an unusually blunt one, because
    a single indicator carries about a bit.

    Returns the prior and posterior medians and central 95% intervals over `grid`,
    plus two measures of what the announcement was worth: `bits`, the
    Kullback-Leibler divergence of posterior from prior, and `prior_predictive`, the
    probability the prior assigned to the boundary being crossed at all. The second
    is the more intuitive one — an announcement that was 80% expected carries
    `-log2(0.8)` = 0.32 bits of surprise, and no amount of press coverage adds to it.
    """
    if prior_log_se <= 0:
        raise ValueError("prior standard error must be positive")
    g = np.asarray(grid if grid is not None else np.linspace(0.10, 1.60, 3001),
                   dtype=float)
    if np.any(g <= 0):
        raise ValueError("the grid must be positive hazard ratios")
    theta = np.log(g)
    prior = norm.pdf(theta, loc=np.log(float(prior_hr)), scale=float(prior_log_se))
    covered = float(np.trapezoid(prior, theta))
    if covered < 0.95:
        # Found by a test that expected an error and got a confident answer. A grid
        # that clips the prior gets renormalised on the truncated support, so the
        # posterior comes back narrow, plausible and wrong. Refuse instead.
        raise ValueError(
            f"the grid covers only {100 * covered:.1f}% of the prior's mass; "
            "widen it or the posterior is computed on a truncated distribution")
    prior = prior / covered

    se = np.array([log_hr_se(events, ratio=ratio, hazard_ratio=float(h))
                   for h in g])
    like = norm.cdf(-theta / se - float(z_boundary))
    post = prior * like
    mass = np.trapezoid(post, theta)
    if mass <= 0:
        raise ValueError("the likelihood vanishes on this grid; widen it")
    post = post / mass

    def summary(dens):
        cdf = np.concatenate([[0.0], np.cumsum(
            0.5 * (dens[1:] + dens[:-1]) * np.diff(theta))])
        cdf = cdf / cdf[-1]
        q = np.interp([0.025, 0.5, 0.975], cdf, g)
        return {"median": float(q[1]), "low": float(q[0]), "high": float(q[2])}

    ok = (post > 0) & (prior > 0)
    kl = float(np.trapezoid(post[ok] * np.log(post[ok] / prior[ok]), theta[ok]))
    return {"grid": g, "prior": prior, "posterior": post, "likelihood": like,
            "prior_summary": summary(prior), "posterior_summary": summary(post),
            "prior_predictive": float(mass), "bits": kl / np.log(2.0),
            "surprisal_bits": float(-np.log2(mass)) if mass > 0 else float("inf")}


# ---------------------------------------------------------------- simulation

@dataclass(frozen=True)
class CoxFit:
    """A two-arm Cox fit: the estimate, its information, and the count arithmetic.

    Carries both readings of the same risk sets — the partial-likelihood maximum at
    `log_hr` with its observed information, and the score test at the null, which is
    the log-rank statistic. Holding them in one object is the point: the module's
    claim is that `schoenfeld_information` reproduces `information`, and that is only
    checkable if both come out of the same fit.
    """
    log_hr: float
    information: float
    events: int
    events_treated: int
    score: float
    null_information: float

    @property
    def se(self) -> float:
        return float(1.0 / np.sqrt(self.information))

    @property
    def event_split(self) -> float:
        return float(self.events_treated / self.events)

    @property
    def logrank_z(self) -> float:
        """The log-rank score test, `(O - E) / sqrt(V)`, evaluated at the null."""
        return float(self.score / np.sqrt(self.null_information))

    @property
    def schoenfeld_information(self) -> float:
        """`D * p_e * (1 - p_e)` — what the rest of this module assumes."""
        p = self.event_split
        return float(self.events * p * (1.0 - p))


def cox_binary(time, event, arm, *, tol: float = 1e-10,
               max_iter: int = 100) -> CoxFit:
    """Cox partial likelihood for one binary covariate, by Newton-Raphson.

    `arm` is 1 for treated, 0 for control. With a single binary covariate the whole
    fit reduces to counts on risk sets: at each event time with `n1` treated and `n0`
    control still at risk, the expected treated events are `d * n1 r / (n1 r + n0)`
    and the information contribution is `d * pbar * (1 - pbar)`. Breslow's handling
    of ties, which is what the trial reports being read here would have used.

    At `beta = 0` the score and information are exactly the Mantel-Haenszel log-rank
    numerator and its null variance, so `logrank_z` needs no separate code path.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    a = np.asarray(arm, dtype=int)
    if not (t.shape == e.shape == a.shape) or t.ndim != 1:
        raise ValueError("time, event and arm must be 1-D arrays of one length")
    if not np.all(np.isin(a, (0, 1))):
        raise ValueError("arm must be 0 or 1")

    order = np.argsort(t, kind="mergesort")
    t, e, a = t[order], e[order], a[order]
    n = t.size

    rows = []
    at_risk, at_risk_treated = n, int(a.sum())
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        d = int(e[i:j].sum())
        if d > 0:
            rows.append((at_risk, at_risk_treated, d,
                         int((e[i:j] & (a[i:j] == 1)).sum())))
        at_risk -= (j - i)
        at_risk_treated -= int((a[i:j] == 1).sum())
        i = j
    if not rows:
        raise ValueError("no events: nothing to compare")

    blk = np.asarray(rows, dtype=float)
    n_risk, n1, d, d1 = blk[:, 0], blk[:, 1], blk[:, 2], blk[:, 3]
    n0 = n_risk - n1

    def score_and_info(beta: float) -> tuple[float, float]:
        r = np.exp(beta)
        denom = n1 * r + n0
        with np.errstate(invalid="ignore", divide="ignore"):
            pbar = np.where(denom > 0, n1 * r / np.where(denom > 0, denom, 1.0), 0.0)
        return (float(np.sum(d1 - d * pbar)),
                float(np.sum(d * pbar * (1.0 - pbar))))

    score0, info0 = score_and_info(0.0)
    beta = 0.0
    for _ in range(int(max_iter)):
        s, info = score_and_info(beta)
        if info <= 0:
            break
        step = s / info
        beta += float(np.clip(step, -2.0, 2.0))
        if abs(step) < tol:
            break
    _, info = score_and_info(beta)
    return CoxFit(log_hr=float(beta), information=float(info),
                  events=int(d.sum()), events_treated=int(d1.sum()),
                  score=score0, null_information=info0)


def simulate_arms(*, n_treated: int, n_control: int, hazard_ratio: float,
                  control_rate: float, follow_up: float, rng,
                  delay: float = 0.0, dropout_rate: float = 0.0):
    """Exponential survival in two arms, administratively censored at `follow_up`.

    `delay` makes the effect non-proportional in the way an immunotherapy plausibly
    is: the hazard ratio is 1 until `delay`, then `hazard_ratio` after it. This is
    the case worth simulating, because it is the case where the hazard ratio stops
    being a parameter and becomes a follow-up-weighted average — while, as the tests
    show, the *information* arithmetic goes on holding.

    `dropout_rate` adds exponential loss to follow-up, independent of arm.
    """
    if control_rate <= 0 or follow_up <= 0:
        raise ValueError("control rate and follow-up must be positive")
    lam0 = float(control_rate)
    lam1_late = lam0 * float(hazard_ratio)
    d = float(delay)

    def draw(n, late_rate):
        u = rng.random(n)
        # Survival is exp(-lam0 t) up to the delay, then the late rate takes over.
        s_at_delay = np.exp(-lam0 * d)
        early = -np.log(u) / lam0
        late = d + -(np.log(u) - np.log(s_at_delay)) / late_rate
        return np.where(u > s_at_delay, early, late)

    t1 = draw(int(n_treated), lam1_late)
    t0 = draw(int(n_control), lam0)
    time = np.concatenate([t1, t0])
    arm = np.concatenate([np.ones(int(n_treated), dtype=int),
                          np.zeros(int(n_control), dtype=int)])

    censor = np.full(time.size, float(follow_up))
    if dropout_rate > 0:
        censor = np.minimum(censor,
                            rng.exponential(1.0 / float(dropout_rate), time.size))
    event = time <= censor
    return np.minimum(time, censor), event, arm
