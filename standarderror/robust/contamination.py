"""The two classical outlier types, constructed explicitly.

Robust regression divides contamination into two kinds, and the division is old
enough that the terms are load-bearing:

* a **vertical outlier** has an ordinary `x` and a wrong `y`;
* a **leverage point** has an unusual `x`, and it is *bad* leverage if its `y` does
  not follow the pattern the rest of the data implies.

For a linear model the second is much the more dangerous, and its danger grows with
distance: leverage is literally a distance in `x`, a far-out point pulls the fitted
plane towards itself, and one such point can take the fit anywhere.

**For a tree ensemble that intuition is backwards**, which is what `distance` in
`leverage_points` exists to let you measure. A tree splits; a point far outside the
range of everything else falls beyond the outermost split, gets a leaf to itself, and
its damage is confined to a region no test point visits. A point sitting just past
the edge of the bulk cannot be isolated so cheaply — the splits that would fence it
off also carve up a region that real data occupies. So the dangerous leverage point
for a tree is the *near* one, and a diagnostic borrowed from linear regression ranks
danger in exactly the wrong order.

Every function here returns copies and reports what it did, because a contamination
experiment whose construction is implicit is not reproducible, and because the
construction is usually where the result comes from. The first version of the
experiment in `experiments/exp014` put its leverage points eight standard deviations
out in every coordinate, found that they did no damage at all, and nearly published
that as a fact about XGBoost. It was a fact about the construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Contaminated", "leverage_points", "vertical_outliers"]


@dataclass
class Contaminated:
    """Contaminated data plus the record of what was done to it."""

    X: np.ndarray
    y: np.ndarray
    index: np.ndarray                      # which rows were altered
    kind: str
    params: dict = field(default_factory=dict)

    @property
    def n_contaminated(self) -> int:
        return int(self.index.size)

    @property
    def fraction(self) -> float:
        return float(self.index.size / len(self.y))

    def describe(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return (f"{self.kind}: {self.n_contaminated}/{len(self.y)} rows "
                f"({100 * self.fraction:.1f}%), {p}")


def _pick(n: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must lie in [0, 1)")
    k = int(round(fraction * n))
    if k == 0:
        return np.empty(0, dtype=int)
    return rng.choice(n, k, replace=False)


def vertical_outliers(X, y, *, fraction: float = 0.1, magnitude: float = 20.0,
                      seed: int | None = 1, symmetric: bool = True
                      ) -> Contaminated:
    """Shift a fraction of the responses by `magnitude`, leaving `X` untouched.

    `symmetric=True` shifts up or down with equal probability, which matters: a
    one-sided shift moves the conditional mean as well as fattening the residuals,
    so a method can look robust merely by being biased in the same direction.

    `magnitude` is in the units of `y`. That is not a convenience — it is the whole
    subject of `equivariance`, and the reason this function takes an absolute number
    rather than a multiple of the noise scale: an experiment that specifies
    contamination in units of sigma has quietly made itself scale-free, and cannot
    then be used to find out whether the method is.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).copy()
    X = np.array(X, dtype=float, copy=True)
    idx = _pick(len(y), fraction, rng)
    if idx.size:
        sign = rng.choice([-1.0, 1.0], idx.size) if symmetric else np.ones(idx.size)
        y[idx] += magnitude * sign
    return Contaminated(X, y, idx, "vertical outliers",
                        {"fraction": fraction, "magnitude": magnitude,
                         "symmetric": symmetric, "seed": seed})


def leverage_points(X, y, truth, *, fraction: float = 0.1,
                    distance: float = 2.5, magnitude: float = 20.0,
                    columns=(0,), jitter: float = 0.2, seed: int | None = 1
                    ) -> Contaminated:
    """Move a fraction of rows to `distance` in `columns`, and give them a wrong `y`.

    `truth(X)` supplies the clean response at the moved locations, so the only thing
    wrong with these points is the added `magnitude` — they are *bad* leverage rather
    than merely unusual. Without that step a leverage point is just an extrapolation
    request, and a method that handles it badly is being blamed for the wrong thing.

    `distance` is in the units of the corresponding column of `X` and is the argument
    to vary. Sweeping it is the point: for a tree ensemble the damage is *not*
    monotone in distance, and reporting a single leverage experiment without saying
    how far out the points sat says almost nothing.

    `columns` defaults to a single coordinate rather than all of them, because moving
    every coordinate at once puts the points outside the joint range of the data,
    where a tree isolates them for free.
    """
    rng = np.random.default_rng(seed)
    X = np.array(X, dtype=float, copy=True)
    y = np.asarray(y, dtype=float).copy()
    cols = tuple(int(c) for c in columns)
    if any(c < 0 or c >= X.shape[1] for c in cols):
        raise ValueError(f"columns {cols} out of range for {X.shape[1]} features")
    idx = _pick(len(y), fraction, rng)
    if idx.size:
        for c in cols:
            sign = rng.choice([-1.0, 1.0], idx.size)
            X[idx, c] = distance * sign + rng.normal(0.0, jitter, idx.size)
        y[idx] = truth(X[idx]) + magnitude * rng.choice([-1.0, 1.0], idx.size)
    return Contaminated(X, y, idx, "leverage points",
                        {"fraction": fraction, "distance": distance,
                         "magnitude": magnitude, "columns": cols,
                         "jitter": jitter, "seed": seed})
