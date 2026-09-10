r"""The period total fertility rate, and what a pure change of timing does to it.

The TFR is not "children per woman". It is the sum, over single years of age, of
the age-specific fertility rates observed in **one calendar year** -- the family
size of a woman who does not exist, assembled from women born decades apart. The
quantity a reader thinks they are being told is the *cohort* completed fertility
`Q`, which cannot be known until a cohort finishes childbearing.

The two come apart whenever birth timing moves, and the algebra is exact rather
than approximate. Let every cohort have the same completed fertility `Q` spread
over age by a density `phi(a; mu, sigma)`, and let cohort `b` postpone by
`delta` years per cohort, so its mean age is `mu + delta * b`. The period
schedule in year `t` samples cohort `t - a` at age `a`:

    f(a, t) = Q * phi(a; mu + delta * (t - a), sigma)

Substituting `u = a (1 + delta) - mu - delta t` and integrating over age gives

    TFR(t) = Q / (1 + delta)                                            (1)

and the period mean age of childbearing is `(mu + delta t) / (1 + delta)`, so it
rises at

    r = delta / (1 + delta)                                            (2)

Note that `r` is *not* `delta`: the period mean age rises more slowly than the
cohorts postpone. Combining (1) and (2), `1 - r = 1 / (1 + delta)`, so

    TFR(t) / (1 - r) = Q                                               (3)

which is the Bongaarts-Feeney tempo adjustment, and on this schedule it is not a
correction but an identity. `simulate` reproduces (1), (2) and (3) to eight
decimal places on a discrete age grid.

Three consequences worth stating separately, because each one is a headline
somewhere:

* **postponement alone depresses the TFR.** At `delta = 0.2` the period TFR sits
  17% below completed fertility with nobody having fewer children.
* **a deceleration of postponement raises the TFR, with no change in quantum.**
  Going from `delta = 0.2` to `delta = 0.1` lifts the period TFR by 9.1%, and it
  keeps rising for as long as the pipeline takes to refill.
* **postponement compresses the period schedule.** The period density has
  standard deviation `sigma / (1 + delta)`, narrower than any cohort's. This is
  why the adjustment is applied by birth order and why a changing variance is
  the assumption most likely to fail -- see `variance_bias`.

Where this stops: the derivation above assumes every cohort shares one schedule
shape and shifts it rigidly. Kohler and Philipov, "Variance effects in the
Bongaarts-Feeney formula", *Demography* 38 (2001), derive the correction when
the schedule's variance also moves; `variance_bias` measures the error the
uncorrected formula makes. Mazzuco and Zanotto, *Demographic Research* 52 (2025)
art. 19, find the adjustment more robust in practice than that suggests, because
cohort shape changes largely act through the period mean the formula already
uses. And the adjustment recovers a *tempo-free* period rate, not any real
cohort's completed fertility: it answers "what would the TFR be if timing had
stopped moving", which is a different question from "how many children will
these women have".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Single years of age over which a fertility schedule is summed. Wide enough
#: that a schedule postponed by several years still has negligible mass at the
#: top -- truncation there would leak births and break identity (1).
AGES = np.arange(12, 61)


def schedule(ages, *, quantum: float, mean_age: float, spread: float):
    """One cohort's age-specific fertility: `quantum` births spread over age.

    A normal density is not a claim about real fertility schedules, which are
    right-skewed. It is the shape for which the algebra above is exact, so that
    a disagreement between `simulate` and equations (1)-(3) is a bug in the code
    rather than an artefact of the shape.
    """
    ages = np.asarray(ages, dtype=float)
    z = (ages - float(mean_age)) / float(spread)
    dens = np.exp(-0.5 * z * z) / (float(spread) * np.sqrt(2.0 * np.pi))
    return float(quantum) * dens


@dataclass(frozen=True)
class Year:
    """One calendar year of a synthetic population."""

    year: int
    tfr: float
    mean_age: float
    spread: float
    #: Completed fertility of every cohort contributing to this year. Constant
    #: by construction in `simulate`, and carried so the caller cannot lose
    #: track of the fact that the quantum never moved.
    quantum: float

    @property
    def shortfall(self) -> float:
        """How far the period rate sits below completed fertility, as a fraction."""
        return float(1.0 - self.tfr / self.quantum)


def simulate(*, quantum: float = 1.8, mean_age: float = 30.0,
             spread: float = 4.5, delta=0.2, years=range(0, 40),
             ages=AGES) -> list[Year]:
    """Period fertility in a population where every cohort has the same quantum.

    `delta` is the postponement per cohort, in years of mean age per year of
    birth. Pass a float for a constant regime, or a callable `delta(birth_year)`
    for one that changes -- which is what `rebound` uses.

    Nobody in here has fewer children than anybody else. Every difference in the
    period TFR below is timing.
    """
    ages = np.asarray(ages, dtype=float)
    shift = delta if callable(delta) else (lambda b: float(delta) * b)
    out = []
    for t in years:
        births = np.array([
            schedule([a], quantum=quantum,
                     mean_age=mean_age + shift(t - a), spread=spread)[0]
            for a in ages])
        total = float(births.sum())
        mac = float((ages * births).sum() / total)
        var = float((births * (ages - mac) ** 2).sum() / total)
        out.append(Year(year=int(t), tfr=total, mean_age=mac,
                        spread=float(np.sqrt(var)), quantum=float(quantum)))
    return out


def period_tfr(quantum: float, delta: float) -> float:
    """Equation (1): the period TFR a constant postponement produces."""
    return float(quantum) / (1.0 + float(delta))


def mac_rate(delta: float) -> float:
    """Equation (2): how fast the *period* mean age rises. Not `delta`."""
    d = float(delta)
    return d / (1.0 + d)


def bongaarts_feeney(tfr: float, r: float) -> float:
    """The tempo adjustment, `TFR / (1 - r)`.

    `r` is the annual change in the period mean age of childbearing, estimated
    in practice as half the change between `t - 1` and `t + 1`, and computed by
    birth order because the all-order mean age also moves when the quantum does.
    """
    r = float(r)
    if r >= 1.0:
        raise ValueError(f"r = {r} implies the mean age rises a year per year "
                         f"or faster, which the formula cannot express")
    return float(tfr) / (1.0 - r)


def mac_change(rows: list[Year], year: int) -> float:
    """`r` as a demographer would measure it: half the change across `year`.

    Uses the neighbours rather than the analytic value, because that is what is
    available from published statistics and it is the estimate whose error the
    episode is about.
    """
    by = {row.year: row for row in rows}
    if year - 1 not in by or year + 1 not in by:
        raise ValueError(f"need years {year - 1} and {year + 1} to centre on "
                         f"{year}")
    return 0.5 * (by[year + 1].mean_age - by[year - 1].mean_age)


def rebound(*, quantum: float = 1.8, fast: float = 0.2, slow: float = 0.1,
            switch: int = 40, years=range(30, 130), **kw) -> list[Year]:
    """Postponement decelerates from `fast` to `slow`, and the TFR rises.

    The quantum is constant throughout. Whatever the period TFR does here, it is
    not a change in how many children anyone has -- which is the whole point,
    because a decelerating postponement is exactly what a country produces after
    a long delay of first births, and the resulting rise reads in the press as a
    recovery.

    `switch` is a *birth* cohort, so the period TFR does not move the year the
    behaviour changes: it moves as the affected cohorts enter the childbearing
    window and keeps moving until they have crossed it. The default range is
    wide enough to contain the whole transition, and the lag is a result rather
    than an artefact -- a change in timing behaviour takes most of a generation
    to finish arriving in the period rate.
    """
    def delta(b):
        b = float(b)
        return (fast * b if b <= switch
                else fast * switch + slow * (b - switch))

    return simulate(quantum=quantum, delta=delta, years=years, **kw)


def variance_bias(*, quantum: float = 1.8, delta: float = 0.2,
                  spread_from: float = 4.5, spread_to: float = 5.5,
                  years=range(0, 40), **kw) -> dict:
    """What the uncorrected formula gets wrong when the spread also moves.

    Bongaarts-Feeney assumes a rigid shift. Let the cohort schedule widen at the
    same time and the adjusted rate stops equalling the quantum; this returns the
    size of that error, which is the thing Kohler and Philipov's correction adds
    a term for.
    """
    span = max(years) - min(years)
    def widen(b):
        frac = (float(b) - min(years)) / span if span else 0.0
        return spread_from + (spread_to - spread_from) * np.clip(frac, 0.0, 1.0)

    ages = np.asarray(kw.pop("ages", AGES), dtype=float)
    rows = []
    for t in years:
        births = np.array([
            schedule([a], quantum=quantum, mean_age=30.0 + delta * (t - a),
                     spread=widen(t - a))[0] for a in ages])
        total = float(births.sum())
        mac = float((ages * births).sum() / total)
        var = float((births * (ages - mac) ** 2).sum() / total)
        rows.append(Year(year=int(t), tfr=total, mean_age=mac,
                         spread=float(np.sqrt(var)), quantum=float(quantum)))
    mid = int(np.median([r.year for r in rows]))
    r = mac_change(rows, mid)
    by = {row.year: row for row in rows}
    adjusted = bongaarts_feeney(by[mid].tfr, r)
    return {"year": mid, "tfr": by[mid].tfr, "r": r, "adjusted": adjusted,
            "quantum": float(quantum), "error": adjusted / float(quantum) - 1.0,
            "spread": by[mid].spread}
