"""A denoising diffusion model for fixed-length sequences, small enough to read.

This is a DDPM (Ho, Jain & Abbeel 2020) for one-dimensional windows — a return
path, not an image. It exists so a claim about diffusion models on financial data
can be *run* rather than cited, on a machine with two CPUs and no GPU.

The whole construction rests on one identity. The forward process adds Gaussian
noise step by step, and because the composition of Gaussians is Gaussian, the state
at any step has a closed form given the *original* sample:

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

with `alpha_bar_t` the running product of `1 - beta_t`. Nothing has to be
simulated to get there. So the training task — predict the `eps` that was added —
is an ordinary supervised regression, on pairs you can manufacture in unlimited
quantity from a finite dataset. That is why the denoiser here is an
`sklearn.neural_network.MLPRegressor` and not a deep-learning framework: for this
model class the framework buys speed, not capability.

Reversing it is the other half. `p(x_{t-1} | x_t)` is approximated as a Gaussian
whose mean is written in terms of the predicted noise,

    mu = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(1 - beta_t)

and whose variance is `beta_t`, injected at every step except the last. `Schedule`
holds the coefficients, `DDPM` holds the denoiser and the two loops, and
`GaussianOracle` is the analytic best-possible denoiser for Gaussian data — which
exists so the sampler can be tested against an answer instead of against itself.

Three things worth knowing before using it:

* **Scale matters.** The schedule ends near unit variance, so the data must be
  standardised before training and rescaled after sampling. `DDPM.fit` does this
  and records the factor; sampling without a prior `fit` raises.
* **Windows are the sample.** The model generates a whole `length`-step path at
  once. Any dependence longer than the window cannot be represented, so the window
  has to be long enough for the effect being tested to exist inside it. Checking
  that on the training data first is not optional: at equity-index GARCH
  persistence, the autocorrelation of absolute returns *within* a 32-step window is
  around +0.006, and a model asked to reproduce it has been handed nothing.
* **A converged loss is not a converged model.** The regression target is noise, so
  the irreducible loss is large and the loss curve says almost nothing about sample
  quality. Judge samples, and vary the budget before concluding anything about the
  model class — that is what `DDPM.budget` and `flops_proxy` are for.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = ["DDPM", "GaussianOracle", "Schedule", "cosine_schedule",
           "linear_schedule", "schedule_for_snr", "time_features"]


@dataclass(frozen=True)
class Schedule:
    """Forward-process coefficients, precomputed once.

    `betas` is the per-step noise variance; `alphas = 1 - betas`; `alpha_bar` is
    the cumulative product, and it is the only one used at training time.
    """

    betas: np.ndarray

    def __post_init__(self) -> None:
        b = np.asarray(self.betas, dtype=float)
        if b.ndim != 1 or b.size < 2:
            raise ValueError("betas must be a 1-D array of at least two steps")
        if np.any(b <= 0) or np.any(b >= 1):
            raise ValueError("every beta must lie strictly in (0, 1)")
        object.__setattr__(self, "betas", b)

    @property
    def steps(self) -> int:
        return int(self.betas.size)

    @property
    def alphas(self) -> np.ndarray:
        return 1.0 - self.betas

    @property
    def alpha_bar(self) -> np.ndarray:
        return np.cumprod(self.alphas)

    def corrupt(self, x0: np.ndarray, t: np.ndarray,
                eps: np.ndarray) -> np.ndarray:
        """The closed-form forward step: `sqrt(ab) x0 + sqrt(1-ab) eps`.

        Note what this guarantees and what it does not. If `x0` has unit variance
        the corrupted state has unit variance at *every* `t`, because the two
        coefficients are the square roots of weights summing to one. That is a
        property of the schedule, not evidence that the data was standardised.
        """
        x0 = np.atleast_2d(np.asarray(x0, dtype=float))
        eps = np.atleast_2d(np.asarray(eps, dtype=float))
        t = np.asarray(t)
        if x0.shape != eps.shape:
            raise ValueError("x0 and eps must have the same shape")
        if t.shape != (x0.shape[0],):
            raise ValueError("need one timestep per row")
        if np.any(t < 0) or np.any(t >= self.steps):
            raise ValueError(f"timesteps must lie in [0, {self.steps})")
        ab = self.alpha_bar[t][:, None]
        return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps

    def snr(self) -> np.ndarray:
        """Signal-to-noise ratio `ab / (1 - ab)` at each step."""
        ab = self.alpha_bar
        return ab / (1.0 - ab)

    @property
    def terminal_snr(self) -> float:
        """How much of the data survives at the last forward step.

        The sanity check the loss will not give you, and the first thing this
        module got wrong. Sampling starts from a standard normal draw, which is
        only the right starting distribution if the forward process has destroyed
        essentially all of the signal. The textbook linear schedule reaches that at
        a thousand steps; truncated to two hundred it leaves `alpha_bar = 0.13`, a
        terminal SNR of **0.15**, and the sampler then begins from a distribution
        the model never saw during training.

        The failure is quiet in exactly the way that matters. With standardised
        data the terminal variance is `ab * 1 + (1 - ab) = 1` whatever `ab` is, so
        the generated standard deviation comes out right and nothing looks wrong —
        the damage lands on the higher moments. Feed the analytic
        `GaussianOracle` for data with standard deviation 2.5 through a 200-step
        linear schedule and it returns 2.22.
        """
        ab = float(self.alpha_bar[-1])
        return ab / (1.0 - ab)


def linear_schedule(steps: int = 1000, *, beta_start: float = 1e-4,
                    beta_end: float = 0.02) -> Schedule:
    """The original DDPM schedule: betas linear from `beta_start` to `beta_end`.

    The default is a thousand steps because that is what makes these particular
    endpoints work: `alpha_bar` finishes at 4e-5, a terminal SNR of 4e-5, and the
    standard normal the sampler starts from is then the right distribution. Cutting
    the step count while keeping the endpoints — the obvious way to make sampling
    five times faster — breaks that quietly. See `Schedule.terminal_snr`.
    """
    if steps < 2:
        raise ValueError("need at least two steps")
    return Schedule(np.linspace(beta_start, beta_end, int(steps)))


def schedule_for_snr(steps: int, *, target_snr: float = 1e-3,
                     beta_start: float = 1e-4,
                     max_beta_end: float = 0.5) -> Schedule:
    """A linear schedule of `steps` that actually reaches `target_snr`.

    The textbook DDPM endpoints (1e-4 to 0.02) need a thousand steps to destroy the
    signal. If sampling cost matters — and at a thousand network calls per sample it
    does — the right response is to steepen the schedule rather than to truncate it,
    which is the mistake that leaves a terminal SNR of 0.15 with nothing visibly
    wrong. Solved by bisection on `beta_end`, which the terminal `alpha_bar` is
    monotone in.

    Worth being explicit that this is a *hyperparameter*, not a fix that makes two
    schedules equivalent. A 200-step steep schedule and a 1000-step shallow one both
    start sampling from noise, and they do not produce the same samples: the short
    one asks the denoiser to cover five times less of the schedule with the same
    capacity, and takes five times fewer, larger reverse steps. Which of those wins
    on any given measurement is an empirical question, and `experiments/exp013` is
    partly about how much a stylised-facts table moves when you change it.
    """
    if steps < 2:
        raise ValueError("need at least two steps")
    if target_snr <= 0:
        raise ValueError("target SNR must be positive")
    lo, hi = beta_start * 1.0000001, float(max_beta_end)
    if linear_schedule(steps, beta_start=beta_start,
                       beta_end=hi).terminal_snr > target_snr:
        raise ValueError(f"{steps} steps cannot reach an SNR of {target_snr:g} "
                         f"below beta_end = {hi:g}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        snr = linear_schedule(steps, beta_start=beta_start,
                              beta_end=mid).terminal_snr
        lo, hi = (lo, mid) if snr < target_snr else (mid, hi)
    return linear_schedule(steps, beta_start=beta_start, beta_end=hi)


def cosine_schedule(steps: int = 200, *, offset: float = 0.008,
                    max_beta: float = 0.999) -> Schedule:
    """Nichol & Dhariwal's cosine schedule, which destroys information later.

    Defined through `alpha_bar` rather than `beta`, then differenced. The clip at
    `max_beta` is not cosmetic: the final differences approach one, and a beta of
    exactly one divides by zero in the reverse step.
    """
    if steps < 2:
        raise ValueError("need at least two steps")
    s = float(offset)
    u = np.arange(steps + 1) / steps
    ab = np.cos((u + s) / (1.0 + s) * np.pi / 2.0) ** 2
    ab = ab / ab[0]
    betas = np.clip(1.0 - ab[1:] / ab[:-1], 1e-8, max_beta)
    return Schedule(betas)


def time_features(t: np.ndarray, steps: int, *, harmonics: int = 2) -> np.ndarray:
    """Embed the timestep for a model that has no notion of one.

    A single scalar `t/T` is enough in principle and poor in practice: the
    denoiser's job changes character across the schedule, and a linear input makes
    it interpolate smoothly through a change that is not smooth. A handful of
    sinusoids at different rates lets it turn corners.
    """
    t = np.asarray(t, dtype=float)
    u = (t / float(steps))[:, None]
    cols = [u]
    for k in range(1, int(harmonics) + 1):
        cols += [np.sin(np.pi * k * u), np.cos(np.pi * k * u)]
    return np.hstack(cols)


class GaussianOracle:
    """The best possible denoiser when the data really is `N(0, sd^2)` i.i.d.

    This is not a model to use; it is the fixed point the sampler must reproduce.
    For Gaussian data the posterior mean of the added noise is available in closed
    form, `E[eps | x_t] = sqrt(1 - ab) / (ab sd^2 + 1 - ab) * x_t`, so plugging it
    into ancestral sampling must return samples with standard deviation `sd`. If it
    does not, the bug is in the reverse step and no amount of training will hide it.
    """

    def __init__(self, schedule: Schedule, sd: float = 1.0,
                 *, harmonics: int = 2):
        self.schedule = schedule
        self.sd = float(sd)
        self.harmonics = int(harmonics)
        self._ab = schedule.alpha_bar

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        n_time = 1 + 2 * self.harmonics
        x, tf = features[:, :-n_time], features[:, -n_time:]
        t = np.rint(tf[:, 0] * self.schedule.steps).astype(int)
        t = np.clip(t, 0, self.schedule.steps - 1)
        ab = self._ab[t][:, None]
        var = ab * self.sd ** 2 + (1.0 - ab)
        return np.sqrt(1.0 - ab) / var * x


@dataclass
class DDPM:
    """Denoising diffusion for `length`-step sequences.

    `denoiser` is any object with scikit-learn's `fit(X, Y)` / `predict(X)`
    interface, taking `[x_t, time features]` and returning the predicted noise.
    """

    length: int
    schedule: Schedule
    denoiser: object
    harmonics: int = 2
    noise_per_window: int = 6
    seed: int = 0
    scale: float = field(default=float("nan"), init=False)
    loss: float = field(default=float("nan"), init=False)
    n_train_rows: int = field(default=0, init=False)

    @classmethod
    def budget(cls, length: int = 64, *, hidden: int = 256, max_iter: int = 80,
               noise_per_window: int = 6, steps: int = 1000,
               schedule: Schedule | None = None, seed: int = 0) -> "DDPM":
        """A configuration at a stated compute budget, everything else fixed.

        The three knobs are width, passes and noisings per window, and they are
        exposed together because a ladder in budget is the only way to separate
        "this model class does not reproduce the fact" from "this run was too
        small". `flops_proxy` reports the product so the rungs can be labelled.
        """
        from sklearn.neural_network import MLPRegressor
        net = MLPRegressor(hidden_layer_sizes=(hidden, hidden),
                           learning_rate_init=2e-3, batch_size=256,
                           max_iter=int(max_iter), random_state=seed)
        return cls(length=length,
                   schedule=linear_schedule(steps) if schedule is None
                   else schedule,
                   denoiser=net, noise_per_window=noise_per_window, seed=seed)

    def flops_proxy(self, n_windows: int) -> float:
        """Training cost proxy: rows x passes x parameters, in units of 1e9.

        Not a flop count. It is monotone in the three knobs and in the data size,
        which is all a budget axis needs to be, and it is computed from the
        configuration rather than from a wall clock that depends on what else the
        machine was doing.
        """
        h = getattr(self.denoiser, "hidden_layer_sizes", (0, 0))
        h = tuple(h) if isinstance(h, (tuple, list)) else (int(h),)
        n_in = self.length + 1 + 2 * self.harmonics
        sizes = (n_in,) + h + (self.length,)
        params = sum(a * b + b for a, b in zip(sizes, sizes[1:]))
        passes = int(getattr(self.denoiser, "max_iter", 1))
        rows = int(n_windows) * int(self.noise_per_window)
        return float(rows * passes * params / 1e9)

    # -- data ------------------------------------------------------------------
    def windows(self, series: np.ndarray, *, stride: int = 1) -> np.ndarray:
        """Overlapping windows of the training series, as rows.

        Overlap is a real choice with a real cost: at stride 1 a window and its
        neighbour share `length - 1` of their values, so the effective sample size
        is far below the row count and any variance estimate over rows is
        optimistic. Striding trades rows for independence.
        """
        series = np.asarray(series, dtype=float).ravel()
        if series.size < self.length:
            raise ValueError(f"series shorter than the window ({self.length})")
        view = np.lib.stride_tricks.sliding_window_view(series, self.length)
        return view[::int(stride)].copy()

    def training_set(self, windows: np.ndarray, *,
                     rng: np.random.Generator | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Manufacture `(features, noise)` pairs from standardised windows.

        Each window is corrupted `noise_per_window` times at independently drawn
        timesteps. This is the step that makes a small dataset large, and the
        reason a diffusion model can be trained on a few thousand paths at all.
        """
        rng = np.random.default_rng(self.seed) if rng is None else rng
        w = np.atleast_2d(np.asarray(windows, dtype=float))
        if w.shape[1] != self.length:
            raise ValueError(f"windows must have width {self.length}")
        xs, ys = [], []
        for _ in range(int(self.noise_per_window)):
            t = rng.integers(0, self.schedule.steps, w.shape[0])
            eps = rng.standard_normal(w.shape)
            xt = self.schedule.corrupt(w, t, eps)
            xs.append(np.hstack([xt, time_features(
                t, self.schedule.steps, harmonics=self.harmonics)]))
            ys.append(eps)
        return np.vstack(xs), np.vstack(ys)

    # -- fit and sample --------------------------------------------------------
    def fit(self, windows: np.ndarray, *,
            rng: np.random.Generator | None = None) -> "DDPM":
        """Standardise, manufacture noise pairs, regress.

        The standardisation is by a single scalar across all windows, not per
        window: dividing each window by its own standard deviation would remove
        exactly the volatility variation the model is being asked to learn.
        """
        rng = np.random.default_rng(self.seed) if rng is None else rng
        w = np.atleast_2d(np.asarray(windows, dtype=float))
        scale = float(w.std())
        if not scale > 0:
            raise ValueError("training windows have zero variance")
        self.scale = scale
        X, Y = self.training_set(w / scale, rng=rng)
        self.n_train_rows = int(X.shape[0])
        with warnings.catch_warnings():
            # A fixed iteration budget is the point; convergence is not claimed.
            warnings.simplefilter("ignore")
            self.denoiser.fit(X, Y)
        self.loss = float(getattr(self.denoiser, "loss_", float("nan")))
        return self

    def sample(self, n: int = 500, *,
               rng: np.random.Generator | None = None,
               scale: float | None = None) -> np.ndarray:
        """Ancestral sampling from noise back to data, in the original units.

        The last step injects no noise. Keeping it in is a common and invisible
        bug: it adds a full `sqrt(beta_0)` of variance to every generated value,
        which at a linear schedule is small enough to pass a plot and large enough
        to move a kurtosis estimate.
        """
        rng = np.random.default_rng(self.seed + 1) if rng is None else rng
        s = self.scale if scale is None else float(scale)
        if not np.isfinite(s):
            raise RuntimeError("call fit() before sample(), or pass scale=")
        if self.schedule.terminal_snr > 1e-2:
            warnings.warn(
                f"terminal SNR is {self.schedule.terminal_snr:.3g}: the forward "
                f"process has not destroyed the signal, so starting from a "
                f"standard normal draw is a distribution mismatch the generated "
                f"standard deviation will not reveal",
                RuntimeWarning, stacklevel=2)
        betas, alphas, ab = self.schedule.betas, self.schedule.alphas, \
            self.schedule.alpha_bar
        x = rng.standard_normal((int(n), self.length))
        for t in range(self.schedule.steps - 1, -1, -1):
            tf = np.tile(time_features(np.array([t]), self.schedule.steps,
                                       harmonics=self.harmonics), (int(n), 1))
            eps = np.asarray(self.denoiser.predict(np.hstack([x, tf])))
            mu = (x - betas[t] / np.sqrt(1.0 - ab[t]) * eps) / np.sqrt(alphas[t])
            x = mu if t == 0 else mu + np.sqrt(betas[t]) * \
                rng.standard_normal((int(n), self.length))
        return x * s
