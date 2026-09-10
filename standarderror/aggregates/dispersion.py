r"""The Gini coefficient, and the fact that it is many-to-one onto distributions.

A scalar summary of a distribution throws information away. That is the point of
one, and it is not by itself a criticism. The criticism is specific: the Gini
throws away *which* inequality you have, and two societies whose sensible
policies are opposite can share a coefficient to fourteen decimal places.

`matched_pair` builds the demonstration. Start from one lognormal population and
apply two different distortions, each tuned to the same Gini:

* **collapse** -- the poorest tenth loses 90% of its income.
* **runaway** -- the richest hundredth has its income multiplied by 2.51.

Both land on a Gini of 0.339374, agreeing to `7e-15`. And it is not only the
Gini that cannot tell them apart. Because each distortion lives *inside* a tail,
the percentile ratios never touch it either:

    statistic                      collapse   runaway
    Gini                           0.339374   0.339374
    p90 / p10                         4.078      4.078
    p50 / p10                         2.015      2.015
    p90 / p50                         2.024      2.024
    poverty headcount, half median    10.2%      10.2%
    ----------------------------------------------------
    bottom 10% income share           0.35%      3.17%
    top 1% income share                3.9%       9.0%
    poverty gap index                 0.092      0.022

Five headline numbers identical; the poorest tenth holds **nine times** more
income in one world than the other. What separates them is a tail share, or a
poverty measure that counts depth rather than heads: the headcount is the same
because both worlds leave the same people under the line, and only the gap index
notices that in one of them they are 90% below it rather than 21%.

**Why the Gini cannot see it.** Differentiate the coefficient with respect to
one person's income:

    dG / dx_k = 2 k / (n^2 mu) - c

with `c` independent of `k`. The sensitivity is **linear in the recipient's
rank, with slope `2 / (n^2 mu)`, and depends on nothing else about them.**
`rank_sensitivity` measures it and gets an R-squared of 1.0000000000 against
rank and a slope matching the closed form to six figures -- while the incomes
at those ranks differ by a factor of six. So a unit of income is priced by
where the recipient stands in the queue and not at all by what the money does
to them, and a distortion confined to a tail moves nobody very far along it.

Which also bounds how much the Gini could ever have said about the bottom. In
this population the poorest tenth holds 3.35% of income, so destroying **all** of
it -- one person in ten with nothing -- moves the Gini by 0.041, about what
tripling the top percentile does. A summary weighted by income share cannot be
sensitive to a group that has none.

Where this stops: none of this says the Gini is wrong or should not be
published. It says a scalar cannot carry a shape, and the fix is not a better
scalar -- every scalar has a blind spot somewhere, and `describe` exists to show
which one. Report the Lorenz curve, or at least a tail share alongside the
coefficient.

References: Sen, *On Economic Inequality* (1973), for the axioms the headcount
ratio fails; Foster, Greer and Thorbecke, "A class of decomposable poverty
measures", *Econometrica* 52 (1984), for the gap index used here; Atkinson, "On
the measurement of inequality", *Journal of Economic Theory* 2 (1970), for what
choosing a scalar commits you to.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

#: Spread of log income for the base population. 0.55 puts the base Gini near
#: 0.30, which is where a good many rich countries sit.
SIGMA = 0.55
#: Odd, so the median is an order statistic rather than an average of two.
N = 200_001


def gini(x) -> float:
    """`G = 2 sum(i x_i) / (n^2 mu) - (n + 1) / n`, on sorted `x`.

    The rank-weighted form rather than the mean-absolute-difference form,
    because it is the one whose derivative gives `transfer_sensitivity`.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    return float(2.0 * (i * x).sum() / (n * n * x.mean()) - (n + 1.0) / n)


def lorenz(x, points: int = 101):
    """Cumulative income share against cumulative population share."""
    x = np.sort(np.asarray(x, dtype=float))
    cum = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
    p = np.linspace(0.0, 1.0, int(points))
    return p, np.interp(p, np.linspace(0.0, 1.0, len(cum)), cum)


def population(*, sigma: float = SIGMA, n: int = N, seed: int = 7):
    return np.sort(np.random.default_rng([int(seed), int(n)]
                                         ).lognormal(0.0, float(sigma), int(n)))


def collapse(base, a: float, *, share: float = 0.10):
    """The poorest `share` loses a fraction `a` of its income."""
    y = np.asarray(base, dtype=float).copy()
    y[:int(float(share) * len(y))] *= (1.0 - float(a))
    return y


def runaway(base, b: float, *, share: float = 0.01):
    """The richest `share` has its income multiplied by `1 + b`."""
    y = np.asarray(base, dtype=float).copy()
    y[-int(float(share) * len(y)):] *= (1.0 + float(b))
    return y


def matched_pair(*, a: float = 0.90, base=None, hi: float = 30.0, **kw):
    """Two worlds with the same Gini: a collapse at the bottom, a runaway at
    the top.

    `a` fixes the collapse; the runaway multiple is solved for. Returns the two
    populations, the target they share, and the multiple that got there.
    """
    base = population(**kw) if base is None else np.asarray(base, dtype=float)
    low = collapse(base, a)
    target = gini(low)
    b = brentq(lambda t: gini(runaway(base, t)) - target, 1e-9, hi,
               xtol=1e-12, rtol=1e-14)
    high = runaway(base, b)
    return {"base": base, "collapse": low, "runaway": high,
            "gini": target, "base_gini": gini(base), "a": float(a),
            "b": float(b), "disagreement": abs(gini(low) - gini(high))}


def poverty(x, *, fraction: float = 0.5) -> dict:
    """Headcount against depth, at a line of `fraction` of the median.

    The headcount ratio is the statistic that fails Sen's monotonicity: making a
    poor person poorer does not move it. The gap index is what notices.
    """
    x = np.asarray(x, dtype=float)
    line = float(fraction) * float(np.median(x))
    poor = x[x < line]
    if len(poor) == 0:
        return {"line": line, "headcount": 0.0, "gap_index": 0.0,
                "mean_shortfall": 0.0}
    return {"line": line,
            "headcount": len(poor) / len(x),
            "gap_index": float(((line - poor) / line).sum() / len(x)),
            "mean_shortfall": float(((line - poor) / line).mean())}


def describe(x) -> dict:
    """The battery of headline numbers, so it is visible which ones agree."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    total = x.sum()
    med = float(np.median(x))
    pov = poverty(x)
    return {
        "gini": gini(x),
        "p90_p10": float(np.quantile(x, 0.9) / np.quantile(x, 0.1)),
        "p50_p10": float(med / np.quantile(x, 0.1)),
        "p90_p50": float(np.quantile(x, 0.9) / med),
        "headcount": pov["headcount"],
        "bottom10_share": float(x[:int(0.10 * n)].sum() / total),
        "top1_share": float(x[-int(0.01 * n):].sum() / total),
        "gap_index": pov["gap_index"],
        "mean_shortfall": pov["mean_shortfall"],
        "mean_median": float(x.mean() / med),
    }


def agreement(a: dict, b: dict, *, tolerance: float = 1e-3) -> dict:
    """Which of two `describe` batteries agree, and by how much they differ.

    Written as a function rather than done by eye in the episode because the
    claim is "five of these are identical", and that is the kind of claim that
    should be produced by a comparison rather than asserted next to a table.
    """
    out = {}
    for k in a:
        lo, hi = a[k], b[k]
        rel = abs(hi - lo) / max(abs(lo), 1e-12)
        out[k] = {"collapse": lo, "runaway": hi, "relative": rel,
                  "same": rel < float(tolerance),
                  "ratio": (hi / lo) if lo else float("nan")}
    return out


def rank_sensitivity(x, *, step: float = 1e-7, stride: int = 200,
                     margin: int = 200) -> dict:
    """How much one unit of income changes the Gini, as a function of *rank*.

    Differentiating `G = 2 sum(i x_i) / (n^2 mu) - (n + 1) / n` with respect to
    one person's income gives

        dG / dx_k = 2k / (n^2 mu) - c

    where `c` collects the terms that do not depend on `k`. So the sensitivity
    is **linear in the recipient's rank with slope `2 / (n^2 mu)`, and depends
    on nothing else about them** -- not on their income, and not on what the
    money would do for them. Measured by central differences: R-squared of
    1.0000000000 against rank, and a fitted slope matching the closed form to
    six figures, while the incomes at those ranks differ sixfold.

    A note on how not to measure this, because I did it the wrong way first.
    The obvious experiment is to move a fixed sum some number of ranks down the
    distribution and watch the coefficient. That measures something else: a sum
    worth one percent of the mean is *three thousand times* a typical gap
    between neighbouring incomes at this sample size, so it vaults the
    recipient over thousands of people and the rank distance in the formula is
    not the rank distance you set. The measurement came out non-monotone in the
    distance, which is what sent me back to the algebra. The perturbation has
    to be small against the local spacing, which is `1 / (n f(x))`, and then
    the identity is exact.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    mu = float(x.mean())
    h = float(step) * mu
    ranks = np.arange(int(margin), n - int(margin), int(stride))
    if len(ranks) < 4:
        raise ValueError("need at least four ranks to fit a line; lower "
                         "`stride` or `margin`, or use a larger population")
    slopes = []
    for k in ranks:
        up, down = x.copy(), x.copy()
        up[k] += h
        down[k] -= h
        slopes.append((gini(up) - gini(down)) / (2.0 * h))
    slopes = np.asarray(slopes)
    fit = np.polyfit(ranks, slopes, 1)
    residual = slopes - np.polyval(fit, ranks)
    r2 = float(1.0 - (residual ** 2).sum()
               / ((slopes - slopes.mean()) ** 2).sum())
    predicted = 2.0 / (n * n * mu)
    return {"ranks": ranks.tolist(), "sensitivity": slopes.tolist(),
            "incomes": x[ranks].tolist(), "slope": float(fit[0]),
            "predicted_slope": float(predicted),
            "slope_ratio": float(fit[0] / predicted), "r_squared": r2,
            "income_range": float(x[ranks].max() / x[ranks].min()),
            "spacing": float(np.median(np.diff(x))),
            "step_over_spacing": float(h / np.median(np.diff(x)))}


def destroying_the_bottom(base=None, *, share: float = 0.10, **kw) -> dict:
    """What the Gini does when the poorest `share` loses everything.

    The bound on how much the coefficient could ever have said about them: a
    summary weighted by income share cannot be sensitive to a group holding
    almost none of it.
    """
    base = population(**kw) if base is None else np.asarray(base, dtype=float)
    n = len(base)
    held = float(base[:int(float(share) * n)].sum() / base.sum())
    gone = collapse(base, 1.0, share=share)
    return {"share": float(share), "income_share_held": held,
            "gini_before": gini(base), "gini_after": gini(gone),
            "gini_change": gini(gone) - gini(base)}
