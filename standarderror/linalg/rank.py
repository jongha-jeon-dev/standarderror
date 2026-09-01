"""Choosing a rank, and why the elbow is not a way of doing it.

Eckart and Young settled one question completely: among all matrices of rank `k`,
the closest one to `Y` in Frobenius norm is `Y`'s own truncated SVD. That theorem
is exact, it is why every low-rank method in use is a truncated factorisation,
and it says *nothing whatever about which k*. The scree plot is what fills that
gap in practice, and it is not a rank estimator -- it is a picture with no
reference distribution in it, which is the same defect episode four found in a
bootstrap.

Four rules are implemented here so they can be compared rather than argued about.

* `elbow` -- the largest drop, which is what "pick the elbow" means when made
  precise enough to run. It is exact when the signal is strong and it fails
  *non-monotonically*: as the signal weakens it starts reporting **more**
  components than exist, because the largest gap wanders into the noise bulk.
  Every other rule here degrades towards zero, which is at least safe.
* `noise_edge` -- `sigma (sqrt(n) + sqrt(p))`, the limit of the largest singular
  value of pure noise. Simple, and it over-counts, because adding signal inflates
  the observed singular values above where the signal alone would sit.
* `optimal_threshold` -- Gavish and Donoho's hard threshold, which is optimal for
  the *reconstruction error* of the truncated matrix. It is deliberately
  conservative about rank and will report zero components where three exist:
  a direction only slightly above the noise contributes more noise than signal to
  a reconstruction, so dropping it lowers the error even though the direction is
  real. That is not a defect. It is the answer to a different question, and
  confusing the two is the most common mistake in this area.
* `parallel_analysis` -- permute each column independently, which destroys the
  cross-column structure and keeps every marginal, and count the singular values
  above the permuted spectrum's upper quantile. A reference distribution built
  from your own data. It degrades gracefully and never over-counts.

So the choice of rule is a choice of loss. If you want the smallest
reconstruction error, take Gavish-Donoho and accept that it discards real
structure. If you want to know how many directions are distinguishable from
noise, permute. The elbow corresponds to no loss function at all.

References: Eckart and Young, "The approximation of one matrix by another of
lower rank", *Psychometrika* 1 (1936); Gavish and Donoho, "The optimal hard
threshold for singular values is 4/sqrt(3)", *IEEE Trans. Inform. Theory* 60
(2014); Horn, "A rationale and test for the number of factors in factor
analysis", *Psychometrika* 30 (1965); Baik, Ben Arous and Peche (2005) for the
phase transition that makes weak signal undetectable in principle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def low_rank_plus_noise(n: int, p: int, *, singular_values, sigma: float, rng
                        ) -> np.ndarray:
    """A matrix whose signal singular values are exactly what was asked for.

    The factors are orthonormal, which matters: building the signal as
    `U diag(s) V'` with *random* `U` and `V` multiplies every singular value by
    about `sqrt(n p)`, so the signal ends up far above the noise and every rule
    below looks infallible. Orthonormal factors put the requested numbers on the
    scale they will be compared against.
    """
    svals = np.asarray(list(singular_values), dtype=float)
    U = np.linalg.qr(rng.standard_normal((int(n), len(svals))))[0]
    V = np.linalg.qr(rng.standard_normal((int(p), len(svals))))[0]
    return U @ np.diag(svals) @ V.T + rng.normal(0.0, float(sigma), (int(n), int(p)))


def elbow(singular_values) -> int:
    """The largest drop in the spectrum, one-indexed.

    "Pick the elbow" made precise enough to execute. Any other formalisation --
    largest ratio, largest second difference -- has the same failure mode, which
    is that it is a statement about the shape of one spectrum with nothing to
    compare it to.
    """
    s = np.asarray(singular_values, dtype=float)
    if len(s) < 2:
        return len(s)
    return int(np.argmax(-np.diff(s))) + 1


def noise_edge(n: int, p: int, sigma: float) -> float:
    """`sigma (sqrt(n) + sqrt(p))`: where pure noise stops.

    The almost-sure limit of the largest singular value of an `n x p` matrix of
    iid `N(0, sigma^2)`. A finite sample sits just below it, so counting singular
    values above this is a permissive rule rather than a strict one.
    """
    return float(sigma) * (np.sqrt(int(n)) + np.sqrt(int(p)))


def gavish_donoho_lambda(beta: float) -> float:
    """The constant in the optimal hard threshold, for aspect ratio `beta = p/n`."""
    b = float(beta)
    if not 0 < b <= 1:
        raise ValueError(f"beta = p/n must lie in (0, 1], got {b}")
    return float(np.sqrt(2 * (b + 1) + 8 * b
                         / ((b + 1) + np.sqrt(b ** 2 + 14 * b + 1))))


def optimal_threshold(n: int, p: int, sigma: float) -> float:
    """Gavish-Donoho: `lambda(p/n) sigma sqrt(n)`.

    Optimal for the asymptotic mean squared error of the reconstruction, which is
    *not* the same objective as recovering the rank -- see the module docstring.
    """
    n, p = int(n), int(p)
    if p > n:
        n, p = p, n
    return gavish_donoho_lambda(p / n) * float(sigma) * np.sqrt(n)


def parallel_analysis(Y, *, rng, reps: int = 30, level: float = 0.95) -> int:
    """Count singular values above the permuted spectrum's `level` quantile.

    Each column is permuted independently, so every marginal distribution is
    preserved exactly and every relationship *between* columns is destroyed. The
    resulting largest singular value is what this data would produce with no
    structure in it, which is the reference the scree plot lacks.
    """
    Y = np.asarray(Y, dtype=float)
    s = np.linalg.svd(Y, compute_uv=False)
    tops = [np.linalg.svd(np.column_stack([rng.permutation(c) for c in Y.T]),
                          compute_uv=False)[0] for _ in range(int(reps))]
    return int((s > np.quantile(tops, float(level))).sum())


@dataclass(frozen=True)
class RankVerdicts:
    truth: int
    elbow: int
    noise_edge: int
    optimal_threshold: int
    parallel: int


def all_rules(Y, *, truth: int, sigma: float, rng, reps: int = 30) -> RankVerdicts:
    Y = np.asarray(Y, dtype=float)
    n, p = Y.shape
    s = np.linalg.svd(Y, compute_uv=False)
    return RankVerdicts(
        truth=int(truth),
        elbow=elbow(s),
        noise_edge=int((s > noise_edge(n, p, sigma)).sum()),
        optimal_threshold=int((s > optimal_threshold(n, p, sigma)).sum()),
        parallel=parallel_analysis(Y, rng=rng, reps=reps),
    )


def eckart_young_check(Y, k: int, *, trials: int = 200, rng) -> dict:
    """The truncated SVD against random rank-`k` matrices, on the same target.

    Not a proof — the theorem has one — but the episode asserts that truncation
    is optimal and an assertion in a lecture should be checkable by the reader in
    the form the lecture states it.
    """
    Y = np.asarray(Y, dtype=float)
    n, p = Y.shape
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    best = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
    truncation = float(np.linalg.norm(Y - best, "fro"))
    # The theorem also gives the error in closed form: the tail of the spectrum.
    predicted = float(np.sqrt((s[k:] ** 2).sum()))
    worse = 0
    margin = np.inf
    for _ in range(int(trials)):
        # A random column space, but then the least-squares-optimal B for it, so
        # the competitor is the best rank-k matrix with *that* column space
        # rather than a strawman.
        A = rng.standard_normal((n, k))
        B = np.linalg.lstsq(A, Y, rcond=None)[0]
        err = float(np.linalg.norm(Y - A @ B, "fro"))
        worse += err >= truncation
        margin = min(margin, err - truncation)
    return {"k": int(k), "truncation_error": truncation,
            "predicted_error": predicted, "trials": int(trials),
            "never_beaten": worse == int(trials),
            "smallest_margin": float(margin)}


def strength_sweep(multiples, *, n: int, p: int, rank: int, sigma: float,
                   reps: int, rng) -> list[dict]:
    """Every rule against signal strength, as a multiple of the noise edge."""
    edge = noise_edge(n, p, sigma)
    out = []
    for mult in multiples:
        svals = np.full(int(rank), float(mult) * edge)
        # At a multiple of zero there is no signal, so the rank to recover is 0
        # and not `rank`. Scoring that row against `rank` inverts the finding:
        # it credits the elbow for the times it happens to return `rank` on pure
        # noise and gives no credit to the rules that correctly return nothing.
        truth = int(rank) if float(mult) > 0 else 0
        rows = []
        for _ in range(int(reps)):
            Y = low_rank_plus_noise(n, p, singular_values=svals, sigma=sigma,
                                    rng=rng)
            rows.append(all_rules(Y, truth=truth, sigma=sigma, rng=rng, reps=20))
        # Accuracy and spread rather than a median. The median of the elbow's
        # answers is a useless summary of this rule: at weak signal the answers
        # spread over the whole available range and the middle one means nothing.
        entry = {"multiple": float(mult), "signal": float(mult * edge),
                 "truth": truth, "reps": int(reps)}
        for name in ("elbow", "noise_edge", "optimal_threshold", "parallel"):
            got = np.array([getattr(r, name) for r in rows], dtype=float)
            entry[name] = float(np.median(got))
            entry[f"{name}_exact"] = float((got == truth).mean())
            entry[f"{name}_over"] = float((got > truth).mean())
            entry[f"{name}_spread"] = float(np.quantile(got, 0.9)
                                            - np.quantile(got, 0.1))
        out.append(entry)
    return out
