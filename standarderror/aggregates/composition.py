r"""The median wage, and what a change in who is counted does to it.

A wage statistic compares two populations, not two payslips. If the second
population is not the first one plus raises -- and it never is, because people
are hired and laid off between the two prints -- then the difference contains a
term that has nothing to do with anybody's pay.

The size of that term is not a matter of opinion. Take `n` incumbents whose
wages are sorted, give every one of them a raise of `g`, and add `m` entrants
who all earn less than the incumbent median. The combined median is now the
incumbent at rank `(n - m + 1) / 2`, so it has slid **`m / 2` ranks down** the
incumbent list. Near the median, consecutive order statistics of a sample from
a density `f` are spaced about `1 / (n f(M))` apart, so the median falls by

    m / (2 n f(M))

while the raises lift it by `g M`. The median therefore *falls*, despite every
single person earning more, as soon as the entry share exceeds

    s* = 2 g M f(M)                                                     (1)

`M f(M)` is dimensionless, and for a lognormal it is `1 / (sigma sqrt(2 pi))`,
which turns (1) into

    s* = 0.798 g / sigma                                                (2)

At `g = 3%` and `sigma = 0.6`, about the spread of US log hourly wages, that is
**4.0%**. A four percent inflow of below-median workers cancels a universal
three percent raise. `cancelling_share` bisects for it and gets 3.9%.

Read (2) again for the part that is backwards. `s*` falls as `sigma` rises, so a
**more unequal** wage distribution is *more* fragile to composition, not less:
a wider distribution has a thinner density at its median, so the same slide in
rank travels further in money.

And the median is not the safe choice here. The mean's threshold is
`g / (1 - w_e / w_i)`, a share weighted by how far below the entrants sit, and
measured against the same draws the median flips at about 60% of the entry share
the mean needs -- at every spread tried. "Use the median, it is robust" is a
statement about outliers, and composition is not an outlier problem.

**Signs matter and 2020 was the other sign.** Destroy low-wage jobs instead of
creating them and the same arithmetic runs upward. In the quarter the United
States lost more than twenty million jobs, published median usual weekly
earnings growth printed above ten percent. The Atlanta Fed's Wage Growth
Tracker, which restricts the sample to people employed in *both* quarters,
removed nearly eight points of it. The Dallas Fed's CPS decomposition put the
composition term at 5.3 of the 7 points of the spike -- three quarters.

Where this stops: matching individuals fixes the arithmetic and changes the
question. A matched sample can only contain people who held a job in both
periods, so it answers "what happened to the pay of people who kept working",
which in a quarter when twenty million people stopped working is a different
and much narrower question than "what happened to wages".

References: Federal Reserve Bank of Atlanta, "Compositional distortions to a
measure of wage growth during the pandemic", *macroblog* (10 November 2021);
Federal Reserve Bank of Dallas, "Pandemic pushed the U.S. into recession ... and
hourly wages rose?" (9 February 2021); US Bureau of Labor Statistics, *Beyond
the Numbers* 13, on average hourly earnings and the pandemic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Spread of log wages. 0.6 is the order of magnitude for US hourly wages and is
#: the value the episode quotes; the sweep runs either side of it because the
#: threshold depends on it inversely and that dependence is the finding.
SIGMA = 0.6


def wages(n: int = 200_000, *, sigma: float = SIGMA, seed: int = 0):
    """A lognormal wage population, in units of its own median."""
    return np.random.default_rng([int(seed), int(n), int(sigma * 1000)]
                                 ).lognormal(0.0, float(sigma), int(n))


def median_threshold(g: float, sigma: float) -> float:
    """Equation (2): the entry share that cancels a universal raise of `g`."""
    return 2.0 * float(g) / (float(sigma) * np.sqrt(2.0 * np.pi))


def mean_threshold(g: float, entrant_ratio: float) -> float:
    """The same question for the mean.

    `entrant_ratio` is the entrants' mean wage over the incumbents'. The mean
    is a sum, so it moves by the share times the gap rather than by a slide in
    rank, and it does not care about the density anywhere.
    """
    r = float(entrant_ratio)
    if r >= 1.0:
        raise ValueError("entrants at or above the incumbent mean cannot pull "
                         f"it down; got a ratio of {r}")
    return float(g) / (1.0 - r)


@dataclass(frozen=True)
class Flip:
    """The entry share at which a statistic stops reporting the raise."""

    statistic: str
    sigma: float
    raise_pct: float
    measured: float
    predicted: float

    @property
    def error(self) -> float:
        return self.measured / self.predicted - 1.0


def cancelling_share(*, statistic: str = "median", sigma: float = SIGMA,
                     g: float = 0.03, n: int = 400_000, seed: int = 1,
                     hi: float = 0.60, iterations: int = 45) -> Flip:
    """Bisect for the entry share that exactly undoes a universal raise.

    Entrants are drawn from the below-median part of the same law, which is the
    conservative choice: entrants concentrated at the very bottom would flip the
    statistic sooner, so the shares here are upper bounds on how much churn it
    takes.
    """
    stat = {"median": np.median, "mean": np.mean}[statistic]
    rng = np.random.default_rng([int(seed), int(sigma * 1000), int(g * 1e6)])
    w = rng.lognormal(0.0, float(sigma), int(n))
    base = float(stat(w))
    raised = w * (1.0 + float(g))
    pool = rng.lognormal(0.0, float(sigma), 6 * int(n))
    low = pool[pool < np.median(w)]

    lo = 0.0
    for _ in range(int(iterations)):
        s = 0.5 * (lo + hi)
        m = int(s * n)
        if m > len(low):
            raise ValueError("not enough below-median entrants drawn")
        if float(stat(np.concatenate([raised, low[:m]]))) > base:
            lo = s
        else:
            hi = s
    share = 0.5 * (lo + hi)
    if statistic == "median":
        predicted = median_threshold(g, sigma)
    else:
        m = int(share * n)
        predicted = mean_threshold(g, float(low[:m].mean()) / float(w.mean()))
    return Flip(statistic=statistic, sigma=float(sigma), raise_pct=float(g),
                measured=float(share), predicted=float(predicted))


def fragility_sweep(sigmas=(0.3, 0.4, 0.6, 0.8, 1.0), *, g: float = 0.03,
                    **kw) -> list[dict]:
    """Median against mean, at each spread. The ratio is the finding."""
    out = []
    for sigma in sigmas:
        med = cancelling_share(statistic="median", sigma=sigma, g=g, **kw)
        avg = cancelling_share(statistic="mean", sigma=sigma, g=g, **kw)
        out.append({"sigma": float(sigma), "g": float(g),
                    "median_share": med.measured,
                    "median_predicted": med.predicted,
                    "mean_share": avg.measured,
                    "ratio": med.measured / avg.measured})
    return out


def entry_effect(*, share: float = 0.015, g: float = 0.03,
                 sigma: float = SIGMA, n: int = 400_000,
                 seed: int = 2) -> dict:
    """The ordinary-year version: `share` of below-median hires, everyone up `g`.

    2020's job destruction is the dramatic case and a rare one. The boring case
    runs every year: employment grows by a percent or two, the hiring is
    concentrated below the median, and the threshold is only a few percent -- so
    a normal year's net hiring removes a visible fraction of a normal year's
    raise from the print, and does it quietly.
    """
    rng = np.random.default_rng([int(seed), int(sigma * 1000), int(g * 1e6)])
    w = rng.lognormal(0.0, float(sigma), int(n))
    before = float(np.median(w))
    raised = w * (1.0 + float(g))
    pool = rng.lognormal(0.0, float(sigma), 6 * int(n))
    low = pool[pool < before]
    m = int(float(share) * n)
    printed = float(np.median(np.concatenate([raised, low[:m]]))) / before - 1.0
    return {"share": float(share), "g": float(g), "sigma": float(sigma),
            "printed": printed, "matched": float(g),
            "swallowed": float(g) - printed,
            "swallowed_fraction": (float(g) - printed) / float(g)}


def exit_effect(*, share: float = 0.10, g: float = 0.025, sigma: float = SIGMA,
                n: int = 400_000, seed: int = 3) -> dict:
    """2020's sign: the lowest-paid `share` lose their jobs, the rest get `g`.

    Returns what a published median and mean would print against what happened
    to the people still in the data, which is `g` exactly and by construction.
    """
    rng = np.random.default_rng([int(seed), int(sigma * 1000)])
    w = rng.lognormal(0.0, float(sigma), int(n))
    before_med, before_mean = float(np.median(w)), float(w.mean())
    keep = np.sort(w)[int(float(share) * n):] * (1.0 + float(g))
    return {"share": float(share), "g": float(g), "sigma": float(sigma),
            "published_median_growth": float(np.median(keep)) / before_med - 1.0,
            "published_mean_growth": float(keep.mean()) / before_mean - 1.0,
            "matched_growth": float(g)}


def share_implying(target: float, *, g: float = 0.025, sigma: float = SIGMA,
                   lo: float = 0.0, hi: float = 0.5, iterations: int = 40,
                   **kw) -> dict:
    """The job-loss share whose composition effect prints a given median growth.

    Used once, to ask what fraction of the lowest-paid would have to vanish for
    a published median to move as far as the United States' did in mid-2020 on
    underlying pay growth of `g`. It is an order-of-magnitude question and it is
    posed that way in the episode.
    """
    for _ in range(int(iterations)):
        s = 0.5 * (lo + hi)
        if exit_effect(share=s, g=g, sigma=sigma, **kw)[
                "published_median_growth"] < float(target):
            lo = s
        else:
            hi = s
    share = 0.5 * (lo + hi)
    row = exit_effect(share=share, g=g, sigma=sigma, **kw)
    row["target"] = float(target)
    return row
