"""Numerical differentiation, and why a smaller step makes it worse.

A finite-difference derivative carries two errors that move in opposite
directions as `h` shrinks:

* **truncation**, from the Taylor terms you dropped -- `O(h)` for a forward
  difference, `O(h^2)` for a central one. This falls as `h` falls.
* **cancellation**, from subtracting two nearly equal floats and then dividing
  by a small number. `f(x+h)` and `f(x-h)` agree to more digits as `h` shrinks,
  so the difference keeps fewer of them, and the `1/h` amplifies what is left.
  This *rises* as `h` falls, like `eps/h`.

So the total error is a U in `h`, the derivative has an optimal step size, and
below it every further refinement is worse. Measured on `sin` at `x = 1`:

* forward difference: best at `h = 1e-8`, error `2.97e-09`. The scale is
  `sqrt(eps) = 1.49e-08`.
* central difference: best at `h = 1e-5`, error `1.11e-11`. The scale is
  `eps**(1/3) = 6.06e-06`, and the floor is `eps**(2/3) = 3.67e-11`.
* at `h = 1e-16` the central difference is `1.3e+09` times worse than at its
  own optimum.

Two consequences worth stating separately, because they are the useful ones.

**The central difference's advantage is not that it is more accurate at a given
step.** At `h = 1e-8` the two agree to a factor of 1.15. Its advantage is that it
is *allowed a larger step*, and at its own optimum it is 267 times better.

**A gradient check is a finite difference, so it has the same optimum**, and the
step decides what the check can see. On the six-parameter loss in
`gradient_check_design`, the smallest relative error in one gradient entry that
the check still distinguishes from its own noise floor is:

    h = 1e-5     9.3e-10        <- the optimum
    h = 1e-9     4.6e-06
    h = 1e-11    6.5e-04
    h = 1e-13    9.9e-02        <- a 10% gradient bug is invisible

A factor of `1e+08` in what the check can detect, decided by a constant nobody
writes down. `1e-7` and `1e-8`, the common defaults, are past the central
difference's optimum but not yet in the range where a check stops working.

`complex_step` is the way out, where it applies: `Im f(x + ih) / h` has **no
subtraction of nearby values at all**, so there is no cancellation and no
optimum. Measured error against `cos(1)`: `0.0` -- exactly -- at every `h` from
`1e-20` down to `1e-200`.

The cost is that `f` must be complex-analytic and implemented so that it stays
that way, and *which* operations break it is not the list you would guess. What
actually breaks, measured:

* `abs` -- returns `0.0` for `d/dx |x|` at `x = 1.5`, truth `1.0`.
* a real-part cast -- `np.real(z)**2` returns `0.0`, truth `3.0`.
* `x * abs(x)` returns `1.5`, truth `3.0`: wrong by exactly a factor of two,
  which is the dangerous kind of wrong because it looks like a units slip.

What does **not** break, which was a surprise: `np.maximum` and `np.minimum`
compare complex numbers by real part first, so a ReLU network differentiates
correctly away from the kink. Checked against a central difference on a small
MLP at three inputs: agreement to `5e-13`, `1.3e-12` and `1.9e-12`. The first
draft of this docstring asserted the opposite.

References: Nocedal and Wright, *Numerical Optimization* (2006), section 8.1, for
the step-size trade-off; Squire and Trapp, "Using complex variables to estimate
derivatives of real functions", *SIAM Review* 40 (1998), for the complex step;
Higham, *Accuracy and Stability of Numerical Algorithms* (2002), chapter 1, for
cancellation as a general phenomenon rather than a differentiation-specific one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: Unit roundoff for float64. Every scale below is a power of this.
EPS = float(np.finfo(float).eps)

#: Where each scheme's total error is smallest, as a power of `EPS`. Derived by
#: balancing the truncation term against `eps/h`: a first-order scheme balances
#: `h` against `eps/h`, giving `eps**(1/2)`; a second-order scheme balances
#: `h**2` against `eps/h`, giving `eps**(1/3)`.
OPTIMAL_H_EXPONENT = {"forward": 1 / 2, "central": 1 / 3}
#: And the error floor each one reaches there: `eps**(1/2)` and `eps**(2/3)`.
FLOOR_EXPONENT = {"forward": 1 / 2, "central": 2 / 3}


def forward(f: Callable, x: float, h: float) -> float:
    """`(f(x+h) - f(x)) / h`. Truncation `O(h)`, cancellation `O(eps/h)`."""
    return (f(x + h) - f(x)) / h


def central(f: Callable, x: float, h: float) -> float:
    """`(f(x+h) - f(x-h)) / 2h`. Truncation `O(h^2)`, cancellation `O(eps/h)`."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def complex_step(f: Callable, x: float, h: float = 1e-20) -> float:
    """`Im f(x + ih) / h`: a derivative with no cancellation.

    Taylor about a purely imaginary perturbation gives
    `f(x + ih) = f(x) + i h f'(x) - h^2 f''(x)/2 - ...`, so the imaginary part is
    `h f'(x) + O(h^3)` and *no subtraction of nearby values occurs*. There is no
    optimum: `h` can be `1e-200`.

    It requires `f` to be complex-analytic and written so that it stays that way.
    `abs`, a conjugation and a real-part cast each return a plausible wrong
    answer rather than raising -- see the module docstring for the measured
    failures, and for the one that surprised me by working. Use
    `is_complex_safe` before trusting it on a function you did not write.
    """
    return float(np.imag(f(x + 1j * float(h))) / float(h))


def is_complex_safe(f: Callable, x: float, fprime: Callable | None = None, *,
                    tol: float = 1e-6) -> bool:
    """Does `f` survive a complex argument well enough for `complex_step`?

    Two checks, and the first one alone is not enough. The cheap test is whether
    the real part of `f(x + ih)` still equals `f(x)`; that catches `abs` and a
    real-part cast. It does **not** catch `x * abs(x)`, which preserves the real
    part perfectly and returns a derivative wrong by a factor of two. So the
    verdict is settled by agreeing with a central difference at its own optimum,
    to within a tolerance that difference can actually deliver.

    The first version of this function returned True for `x * abs(x)`.
    """
    try:
        got = f(x + 1j * 1e-30)
    except (TypeError, ValueError):
        return False
    if not np.iscomplexobj(np.asarray(got)):
        return False
    if abs(np.real(got) - f(x)) > tol * max(abs(f(x)), 1.0):
        return False
    # The corroborating estimate. A central difference at its optimum is good to
    # about `eps**(2/3)`, so anything agreeing to `tol` is agreeing as well as
    # this comparison can establish.
    reference = fprime(x) if fprime is not None else central(
        f, x, optimal_h("central", scale=max(abs(float(x)), 1.0)))
    return bool(abs(complex_step(f, x) - float(reference))
                <= tol * max(abs(float(reference)), 1.0))


def optimal_h(kind: str, *, scale: float = 1.0) -> float:
    """The step where `kind`'s total error is smallest, to an order of magnitude.

    `scale` carries the problem's own size -- the derivation balances relative
    quantities, so a function whose values are `1e6` wants a proportionally
    larger step. It is a scale, not a prediction: the measured optimum on `sin`
    at `x = 1` is `1e-5` against this returning `6.1e-6`, which is as close as an
    order-of-magnitude argument gets.
    """
    if kind not in OPTIMAL_H_EXPONENT:
        raise ValueError(f"unknown scheme {kind!r}; "
                         f"expected one of {sorted(OPTIMAL_H_EXPONENT)}")
    return float(scale) * EPS ** OPTIMAL_H_EXPONENT[kind]


def error_floor(kind: str) -> float:
    """The smallest error `kind` can reach at any step size."""
    if kind not in FLOOR_EXPONENT:
        raise ValueError(f"unknown scheme {kind!r}")
    return EPS ** FLOOR_EXPONENT[kind]


@dataclass
class Sweep:
    """One scheme's error against step size: the U-curve, as data."""
    kind: str
    h: np.ndarray
    error: np.ndarray

    @property
    def best_h(self) -> float:
        return float(self.h[int(np.argmin(self.error))])

    @property
    def best_error(self) -> float:
        return float(self.error.min())

    @property
    def is_monotone(self) -> bool:
        """False for every finite-difference scheme, which is the point: the
        intuition "a smaller step is more accurate" is not merely imprecise, it
        has the wrong shape."""
        return bool(np.all(np.diff(self.error[::-1]) <= 0))

    def penalty_at(self, h: float) -> float:
        """How many times worse than the optimum this scheme is at step `h`."""
        i = int(np.argmin(np.abs(np.log10(self.h) - np.log10(h))))
        return float(self.error[i] / self.best_error)


def error_sweep(f: Callable, fprime: Callable, x: float, hs, *,
                kind: str = "central") -> Sweep:
    """Measured error of `kind` at each step in `hs`, against the exact `fprime`."""
    rule = {"forward": forward, "central": central,
            "complex": lambda f, x, h: complex_step(f, x, h)}[kind]
    truth = float(fprime(x))
    h = np.asarray(list(hs), dtype=float)
    err = np.array([abs(rule(f, x, float(hi)) - truth) for hi in h])
    return Sweep(kind=kind, h=h, error=err)


def condition_number(f: Callable, fprime: Callable, x: float) -> float:
    """`|x f'(x) / f(x)|`: how a relative error in `x` becomes one in `f(x)`.

    The same quantity episode one of the linear-algebra series computed for a
    linear solve, and the same warning applies about what it does *not* say.

    It bounds the transfer of *input* error to *output* error, and nothing else.
    Two measurements that pin the boundary:

    * `f(x) = x - 1` at `x = 1.0001` has a condition number of `1.0e+04`, and a
      central difference recovers its derivative to `2.2e-16`. A badly
      conditioned function evaluation does not imply a hard derivative.
    * `1 - cos x` at `x = 1e-4` has a condition number of `2.0` -- as well
      conditioned as anything -- and the obvious way to evaluate it loses seven
      decimal digits. See `cancellation_pair`.

    So this is the conditioning of the *problem*. Whether the *algorithm* keeps
    the digits the problem allows is a separate question with a separate answer,
    and conflating the two is the mistake the first draft of this module made.
    """
    fx = float(f(x))
    if fx == 0.0:
        return float("inf")
    return float(abs(float(x) * float(fprime(x)) / fx))


def richardson(f: Callable, x: float, h: float, *, kind: str = "central") -> float:
    """Combine two central differences to cancel the leading truncation term.

    `(4 D(h/2) - D(h)) / 3` is `O(h^4)`. It raises the truncation order and does
    nothing whatever about cancellation, so the U-curve moves left and down but
    does not flatten -- which is the general lesson: extrapolation buys
    truncation, never roundoff.
    """
    if kind != "central":
        raise ValueError("richardson here is written for the central difference")
    return (4.0 * central(f, x, h / 2.0) - central(f, x, h)) / 3.0


def cancellation_pair(x: float) -> dict:
    """`1 - cos x` two ways: same problem, same conditioning, different digits.

    The half-angle identity `1 - cos x = 2 sin^2(x/2)` is exact algebra, so both
    expressions are the same function and both inherit the same condition number
    of about 2. What differs is that the first subtracts two numbers that agree
    to `x^2/2` and the second subtracts nothing.

    Measured at `x = 1e-4`: relative error `5.2e-09` naive against `1.7e-16`
    rewritten, a factor of `3e+07`, on a problem the condition number calls easy.
    This is Higham's distinction between an ill-conditioned problem and an
    unstable algorithm, on two lines you can read at a glance.
    """
    x = float(x)
    naive = 1.0 - np.cos(x)
    rewritten = 2.0 * np.sin(x / 2.0) ** 2
    # The reference: the series x^2/2 - x^4/24 + x^6/720, which for small x is
    # good to well past double precision and involves no cancellation.
    exact = x ** 2 / 2.0 - x ** 4 / 24.0 + x ** 6 / 720.0
    rel = lambda v: abs(v - exact) / abs(exact)          # noqa: E731
    return {"x": x, "naive": naive, "rewritten": rewritten, "reference": exact,
            "naive_relative_error": rel(naive),
            "rewritten_relative_error": rel(rewritten),
            "condition_number": condition_number(
                lambda t: 1.0 - np.cos(t), np.sin, x)}


# ----------------------------------------------------- the gradient check

def gradient_check_design(n: int = 6, seed: int = 0):
    """A small smooth loss and its exact gradient, for the check experiments.

    Deliberately not a neural network: the point is a function whose gradient is
    known in closed form, so "the check failed" can only mean the check failed.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    sym = 0.5 * (W + W.T)

    def loss(v):
        v = np.asarray(v, dtype=float)
        return float(0.5 * v @ W @ v + np.sum(np.exp(0.3 * v)))

    def grad(v):
        v = np.asarray(v, dtype=float)
        return sym @ v + 0.3 * np.exp(0.3 * v)

    return loss, grad, rng.standard_normal(n)


def numeric_gradient(loss: Callable, x, h: float) -> np.ndarray:
    """Central differences, one coordinate at a time."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for i in range(len(x)):
        step = np.zeros_like(x)
        step[i] = h
        out[i] = (loss(x + step) - loss(x - step)) / (2.0 * h)
    return out


def gradient_check(loss: Callable, grad: Callable, x, h: float) -> float:
    """The relative discrepancy a gradient check reports, exactly as written."""
    g = np.asarray(grad(x), dtype=float)
    num = numeric_gradient(loss, x, h)
    return float(np.abs(num - g).max() / max(float(np.abs(g).max()), 1.0))


def smallest_detectable_bug(loss: Callable, grad: Callable, x, h: float, *,
                            index: int = 0, margin: float = 2.0,
                            iterations: int = 80) -> dict:
    """How wrong one gradient entry has to be before the check at step `h` notices.

    The noise floor is what the check reports on a *correct* gradient. A bug is
    detectable when it pushes the reported discrepancy above `margin` times that
    floor -- which is the decision an engineer makes when they compare the check's
    output against a tolerance. Bisected in log space on the relative size of the
    error introduced into entry `index`.

    This is the measurement the episode turns on. At the optimum it resolves
    `9.3e-10`; three decades below the optimum it resolves `9.9e-02`.
    """
    g = np.asarray(grad(x), dtype=float)
    scale = max(float(np.abs(g).max()), 1.0)
    num = numeric_gradient(loss, x, h)
    floor = float(np.abs(num - g).max() / scale)

    def detected(rel: float) -> bool:
        bad = g.copy()
        bad[index] *= 1.0 + rel
        return float(np.abs(num - bad).max() / scale) > margin * floor

    lo, hi = 1e-14, 1.0
    if not detected(hi):
        return {"h": float(h), "floor": floor, "detectable": float("nan"),
                "note": "even a 100% error in this entry is inside the floor"}
    for _ in range(int(iterations)):
        mid = float(np.sqrt(lo * hi))
        if detected(mid):
            hi = mid
        else:
            lo = mid
    return {"h": float(h), "floor": floor, "detectable": hi}
