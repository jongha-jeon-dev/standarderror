r"""Synthetic pretraining series, so that every real series is zero-shot.

The recipe is KernelSynth, from the Chronos paper (Ansari et al., 2024): draw
a Gaussian-process kernel by composing a few simple ones -- linear, squared
exponential, periodic at a calendar-like period, rational quadratic, white
noise -- with random sums and products, then sample one path from the prior.
The periods are the ones real data has (4, 7, 12, 24, 52, 365 and multiples),
which is how a model trained on nothing real still knows what a year looks
like.

KernelSynth is mixed with two process families it does not cover well:
autoregressions near the unit root, which is what most macro series look like,
and random walks with drift and occasional level shifts, which is what most
of the World Bank pool looks like. The mix is stated in `MIX` rather than
hidden in a loop, because it is a modelling choice that shapes what the model
is good at, and a reader comparing episodes needs to know it.

Nothing here is fetched and nothing is committed: the bank is a pure function
of its seed.
"""

from __future__ import annotations

import numpy as np

#: Periods a periodic kernel may take, in time steps.
PERIODS = (4, 6, 7, 12, 14, 24, 26, 30, 48, 52, 60, 96, 168, 365)

#: Share of each process family in the bank.
#:
#: The first version had no trend family -- 70% KernelSynth, 20% AR, 10% random
#: walk -- and it taught the model that trends revert. Measured on windows of
#: 30 points and a 6-step horizon, the recent trend continued in 48.4% of that
#: bank's windows against 69.0% of the real annual pool's, and the partly
#: trained model's annual MASE got *worse* with training (2.23 at step 1,000,
#: 2.61 at step 3,000) while its synthetic loss kept falling. `trend` is the
#: correction, after the piecewise-linear trends in TimesFM's synthetic data.
#: With it the bank's persistence is 56.8%.
#:
#: The mix was *not* then tuned until the bank matched the real pool's 69.0%.
#: The defect fixed was structural -- a prior with no trend family at all --
#: and fixing it from a published recipe is defensible. Adjusting shares until
#: a statistic of the evaluation pool is reproduced would make the evaluation
#: pool a validation set, and every zero-shot number after it would carry a
#: footnote.
MIX = {"kernelsynth": 0.45, "trend": 0.30, "autoregression": 0.15,
       "random walk": 0.10}

#: Bumped whenever a generator or the mix changes, and part of the cache key,
#: so a stale bank on disk can never be loaded as the new one.
VERSION = 2

LENGTH = 512


def _kernel(rng, t):
    """One base kernel's Gram matrix on grid `t` (scaled to [0, 1])."""
    kind = rng.choice(["linear", "rbf", "periodic", "rq", "white", "const"],
                      p=[0.12, 0.22, 0.40, 0.14, 0.08, 0.04])
    d = t[:, None] - t[None, :]
    if kind == "linear":
        c = rng.uniform(-0.5, 0.5)
        return (t[:, None] - c) * (t[None, :] - c) * rng.uniform(0.5, 3.0)
    if kind == "rbf":
        ell = rng.choice([0.02, 0.05, 0.1, 0.3, 1.0])
        return np.exp(-0.5 * (d / ell) ** 2)
    if kind == "periodic":
        period = rng.choice(PERIODS) / len(t)
        ell = rng.choice([0.5, 1.0, 2.0])
        return np.exp(-2.0 * np.sin(np.pi * np.abs(d) / period) ** 2 / ell ** 2)
    if kind == "rq":
        ell, alpha = rng.choice([0.05, 0.1, 0.5]), rng.choice([0.1, 1.0, 10.0])
        return (1 + d ** 2 / (2 * alpha * ell ** 2)) ** (-alpha)
    if kind == "white":
        return np.eye(len(t)) * rng.uniform(0.01, 0.2)
    return np.full((len(t), len(t)), rng.uniform(0.1, 1.0))


def kernelsynth(rng, length: int = LENGTH, *, max_kernels: int = 5):
    """One path from a randomly composed Gaussian-process prior."""
    t = np.linspace(0, 1, length)
    K = _kernel(rng, t)
    for _ in range(int(rng.integers(0, max_kernels))):
        other = _kernel(rng, t)
        K = K + other if rng.random() < 0.5 else K * other
    K = K + 1e-6 * np.eye(length)
    L = np.linalg.cholesky(K + 1e-4 * np.trace(K) / length * np.eye(length))
    return L @ rng.standard_normal(length)


def autoregression(rng, length: int = LENGTH):
    """AR(p) near the unit root, sometimes with a seasonal lag."""
    p = int(rng.integers(1, 4))
    phi = rng.uniform(-0.3, 0.3, p)
    phi[0] = rng.uniform(0.6, 0.99)
    season = int(rng.choice([0, 4, 12, 52]))
    x = np.zeros(length + 200)
    e = rng.standard_normal(len(x))
    s = rng.uniform(0.3, 0.9) if season else 0.0
    for i in range(max(p, season) + 1, len(x)):
        x[i] = phi @ x[i - p:i][::-1] + e[i]
        if season:
            x[i] += s * (x[i - season] - phi[0] * x[i - season - 1])
    return x[200:] + rng.normal(0, 5)


def random_walk(rng, length: int = LENGTH):
    """Drift, noise, and a few level shifts -- the World Bank shape."""
    # Drift in units of the step size, large enough to be visible over a
    # short window in a fair share of rows: the World Bank pool is mostly
    # drifting walks, and a walk whose drift is a twentieth of its step noise
    # teaches the model nothing about them.
    step_sd = rng.uniform(0.2, 1.0)
    drift = rng.normal(0, 0.4) * step_sd
    steps = rng.standard_normal(length) * step_sd + drift
    shifts = rng.random(length) < rng.uniform(0, 0.01)
    steps[shifts] += rng.normal(0, 5, shifts.sum())
    return np.cumsum(steps) + rng.normal(0, 10)


def trend(rng, length: int = LENGTH):
    """A smooth trend with persistent direction, plus autocorrelated noise.

    Three shapes: piecewise-linear with a few slope changes, a logistic
    S-curve (life expectancy, urbanisation, adoption of anything), and
    compound growth. Noise is AR(1) at a few per cent of the trend's range,
    and a third get a small season.
    """
    t = np.linspace(0, 1, length)
    kind = rng.choice(["piecewise", "logistic", "growth"], p=[0.5, 0.25, 0.25])
    if kind == "piecewise":
        k = int(rng.integers(0, 4))
        knots = np.sort(rng.uniform(0.1, 0.9, k))
        slopes = rng.normal(0, 1, k + 1) + rng.normal(0, 0.5)
        edges = np.concatenate([[0.0], knots, [1.0]])
        y = np.zeros(length)
        for a, b, s in zip(edges[:-1], edges[1:], slopes):
            y += s * np.clip(t - a, 0, b - a)
    elif kind == "logistic":
        mid, steep = rng.uniform(-0.3, 1.3), rng.uniform(3, 15)
        y = rng.choice([-1, 1]) / (1 + np.exp(-steep * (t - mid)))
    else:
        y = rng.choice([-1, 1]) * np.exp(rng.uniform(0.5, 3.0) * t)
    span = float(np.ptp(y)) or 1.0
    # Noise is scaled to how far the trend moves in *one step*, because that
    # is the scale a forecast horizon of a few steps sees. Two drafts got
    # this wrong and both are worth recording. Scaled to the whole span, AR
    # noise swamped the trend in any short window and this family -- meant to
    # teach persistence -- continued its own trend in 39.6% of windows, worse
    # than a coin, because AR noise mean-reverts. Scaled to a 20-120 step
    # window it scored 39.0%, for the same reason one level down: the trend
    # still moved less over the horizon than the noise did. Real annual macro
    # series are smooth from one year to the next; noise of a few steps'
    # worth of trend is what they look like.
    per_step = span / length
    phi = rng.uniform(0.0, 0.7)
    sd = per_step * rng.lognormal(np.log(2.0), 0.7) * np.sqrt(1 - phi ** 2)
    e = np.zeros(length)
    z = rng.normal(0, sd, length)
    for i in range(1, length):
        e[i] = phi * e[i - 1] + z[i]
    if rng.random() < 1 / 3:
        period = rng.choice(PERIODS)
        e += rng.uniform(0.02, 0.1) * span * np.sin(
            2 * np.pi * np.arange(length) / period + rng.uniform(0, 6.3))
    return y + e


def bank(count: int, *, length: int = LENGTH, seed: int = 0) -> np.ndarray:
    """`count` series of `length`, float32, deterministic in `seed`.

    The family of each row is drawn from `MIX`; each row then gets a random
    positive affine transform, so the model sees every level and scale and has
    to rely on its normalisation rather than on the numbers being near zero.
    """
    rng = np.random.default_rng(seed)
    names = list(MIX)
    fam = rng.choice(len(names), count, p=list(MIX.values()))
    make = {"kernelsynth": kernelsynth, "trend": trend,
            "autoregression": autoregression, "random walk": random_walk}
    out = np.empty((count, length), np.float32)
    for i, f in enumerate(fam):
        x = make[names[f]](rng, length)
        x = (x - x.mean()) / (x.std() + 1e-8)
        out[i] = x * rng.lognormal(0, 1.5) + rng.normal(0, 3) * rng.lognormal(0, 1)
    return out
