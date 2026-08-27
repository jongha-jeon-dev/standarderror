"""Delayed-feedback capacity cycles: the arithmetic of an industry that invests
in response to prices its investment will change.

Written for exp011, which is about the memory-chip cycle, but there is nothing
semiconductor-specific here. The structure recurs wherever supply responds to
price with a lag: hogs, tankers, oil rigs, housing starts, medical residencies.

The reduced model
-----------------
Work in log deviations from a long-run path. Let `p` be price, `i` inventory as a
fraction of its target. Supply responds to price through one or more channels, each
with its own delay; demand responds too. Inventory integrates the imbalance, and
price responds to inventory:

    i_t = i_{t-1} + theta * sum_k g_k * p_{t - L_k}
    p_t = p_{t-1} - kappa * i_t

Eliminating `i` leaves a single linear delay difference equation,

    p_t - 2 p_{t-1} + p_{t-2} = -c * sum_k g_k * p_{t - L_k},     c = kappa * theta

whose characteristic polynomial is obtained by substituting `p_t = z^t`. Two facts
about it carry the whole analysis:

* The **double root at z = 1** on the left is the inventory integrator. It is why
  this system oscillates at all rather than relaxing: an integrator plus a lag is
  the minimal recipe for a cycle.
* Each channel contributes a term `z^{-L_k}`, so a channel with a long delay
  contributes a *high-order* term and its roots sit close to the unit circle with
  small argument — a long, slow oscillation. Channels at different timescales
  therefore produce genuinely separate periods rather than one blended one, which
  is the observation exp011 is built on.

`characteristic_roots` builds the polynomial and factors it; `dominant_mode`
reports the slowest-decaying complex pair as a (period, growth-per-step) pair. No
approximation is involved: these are the exact roots of the exact polynomial for
the linearised system.

`simulate` runs the nonlinear version — utilisation and product mix are bounded,
inventory cannot go negative — because the linear analysis gives the period and the
nonlinear model gives a series a forecaster can be run against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Channel", "CycleModel", "CycleRun", "characteristic_roots",
           "critical_gain", "dominant_mode", "mix_drag", "model_from_simulation",
           "modes", "realised_period", "simulate"]


@dataclass(frozen=True)
class Channel:
    """One feedback path from price to the supply-demand balance.

    `gain` is the elasticity of the balance to price through this path, signed so
    that a *positive* gain is stabilising in the usual sense (a high price calls
    forth supply, or chokes off demand, either of which closes the gap). `delay` is
    in time steps.
    """

    name: str
    gain: float
    delay: int

    def __post_init__(self) -> None:
        if self.delay < 1:
            raise ValueError(f"{self.name}: delay must be at least one step")
        if not np.isfinite(self.gain):
            raise ValueError(f"{self.name}: gain must be finite")


@dataclass
class CycleModel:
    """A capacity cycle: channels, an inventory integrator and a price rule."""

    channels: list[Channel]
    kappa: float = 0.06          # price response to a unit inventory deviation
    theta: float = 1.0           # inventory response to a unit balance
    decay: float = 0.04          # per-step reversion of price toward long-run cost
    dt_months: float = 1.0       # one step, in months, for reporting periods

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("a cycle needs at least one feedback channel")
        if self.kappa <= 0 or self.theta <= 0:
            raise ValueError("kappa and theta must be positive")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must lie in [0, 1)")

    @property
    def c(self) -> float:
        """The single composite gain the dynamics depend on."""
        return self.kappa * self.theta

    def coefficients(self) -> np.ndarray:
        """Characteristic polynomial coefficients, highest power first.

        The price rule is `p_t = (1 - d) p_{t-1} - kappa * i_t`, with `d` the
        reversion of price toward long-run cost. Eliminating inventory gives

            p_t - (2 - d) p_{t-1} + (1 - d) p_{t-2} + c * sum g_k p_{t-L_k} = 0

        and multiplying by `z^{L_max - 2}` clears the negative exponents:

            z^L - (2-d) z^{L-1} + (1-d) z^{L-2} + c * sum_k g_k z^{L-L_k} = 0

        `d = 0` recovers the undamped form, and it is worth knowing what that form
        does: the left-hand side becomes `(z-1)^2`, a *double* integrator, and a
        double integrator under delayed proportional feedback is unstable at every
        positive gain and every delay. That is not a modelling artefact to be tuned
        away — it says an industry whose only price signal is an inventory level,
        with no anchor to cost, has no stable configuration. Memory has such an
        anchor: cost per bit falls on a learning curve and price is pulled toward
        it, which is what `d` represents.
        """
        order = max(2, max(ch.delay for ch in self.channels))
        coeffs = np.zeros(order + 1)
        coeffs[0] = 1.0                              # z^order
        coeffs[1] += -(2.0 - self.decay)             # z^(order-1)
        coeffs[2] += (1.0 - self.decay)              # z^(order-2)
        for ch in self.channels:
            coeffs[ch.delay] += self.c * ch.gain
        return coeffs


def characteristic_roots(model: CycleModel) -> np.ndarray:
    """Every root of the linearised system's characteristic polynomial."""
    return np.roots(model.coefficients())


def modes(model: CycleModel, *, tol: float = 1e-9) -> list[dict]:
    """All oscillatory modes, slowest-decaying first.

    A mode is a complex-conjugate pair; the real roots are returned too, with
    `period = inf`, because a real root above one is a monotone divergence rather
    than a cycle and calling it a period would be a lie.
    """
    out = []
    for z in characteristic_roots(model):
        if z.imag < -tol:
            continue                      # keep one of each conjugate pair
        growth = float(abs(z))
        arg = float(abs(np.angle(z)))
        period = (float("inf") if arg < tol
                  else 2.0 * np.pi / arg * model.dt_months)
        out.append({"growth_per_step": growth, "period_months": period,
                    "root": complex(z),
                    # Steps for the mode's amplitude to halve or double. Reported
                    # because "growth 1.004 per month" is unreadable and "doubles
                    # in 14 years" is not.
                    "halflife_months": (float("inf") if abs(growth - 1.0) < 1e-12
                                        else float(np.log(0.5) / np.log(growth)
                                                   * model.dt_months))})
    out.sort(key=lambda m: -m["growth_per_step"])
    return out


def dominant_mode(model: CycleModel, *, tol: float = 1e-9) -> dict:
    """The slowest-decaying *oscillatory* mode.

    Skips real roots deliberately. In this family the root at z = 1 from the
    integrator is always present and always marginal; reporting it as the dominant
    mode would hide the cycle, which is the only thing anyone wants to know.
    """
    for m in modes(model, tol=tol):
        if np.isfinite(m["period_months"]) and m["period_months"] > 2.0 * model.dt_months:
            return m
    raise RuntimeError("no oscillatory mode: every root is real")


def critical_gain(channel_delay: int, *, kappa: float = 0.06, theta: float = 1.0,
                  decay: float = 0.04, lo: float = 1e-6, hi: float = 10.0,
                  tol: float = 1e-10) -> float:
    """Composite gain at which a single-channel cycle is marginally stable.

    Bisection on `max|z| - 1` for a one-channel model at the given delay. Useful
    for saying how much headroom a real industry has before its cycle stops
    decaying, without asserting where it actually sits.
    """
    def excess(g: float) -> float:
        m = CycleModel([Channel("x", g, channel_delay)], kappa=kappa, theta=theta,
                       decay=decay)
        return max(abs(characteristic_roots(m))) - 1.0

    if excess(lo) > 0:
        return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@dataclass
class CycleRun:
    """Output of `simulate`. Monthly series, all in natural units."""

    price: np.ndarray            # index, 1.0 = long-run level
    inventory: np.ndarray        # weeks of supply
    utilisation: np.ndarray      # fraction
    mix: np.ndarray              # share of wafer capacity on stacked product
    capacity: np.ndarray         # wafer starts index, 1.0 = initial
    supply: np.ndarray           # bit supply index
    demand: np.ndarray           # bit demand index
    params: dict = field(default_factory=dict)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(
            {"price": self.price, "inventory": self.inventory,
             "utilisation": self.utilisation, "mix": self.mix,
             "capacity": self.capacity, "supply": self.supply,
             "demand": self.demand})

    @property
    def observable(self) -> np.ndarray:
        """What an outside forecaster can see: price and inventory weeks.

        Both are published — contract prices monthly, inventory in weeks of supply
        in every industry note. Capacity under construction is not, which is the
        distinction exp011 measures.
        """
        return np.column_stack([np.log(self.price), self.inventory])

    @property
    def full_state(self) -> np.ndarray:
        """Everything, including what only the producers know."""
        return np.column_stack([np.log(self.price), self.inventory,
                                self.utilisation, self.mix,
                                np.log(self.capacity)])


def simulate(n_months: int = 720, *,
             fast_delay: int = 3,
             slow_delay: int = 60,
             fast_gain: float = 0.80,
             slow_gain: float = 0.30,
             demand_delay: int = 3,
             demand_elasticity: float = 0.35,
             kappa: float = 0.06,
             theta: float = 1.0,
             decay: float = 0.04,
             demand_growth_annual: float = 0.15,
             regime: tuple[int, float] | None = None,
             trade_ratio: float = 3.0,
             mix_start: float = 0.02,
             mix_target: float = 0.23,
             mix_years: float = 12.0,
             inventory_target_weeks: float = 6.0,
             util_bounds: tuple[float, float] = (0.70, 1.00),
             util_reference: float = 0.90,
             retirement_annual: float = 0.05,
             price_floor: float = 1e-3,
             shock_sd: float = 0.012,
             burn_in: int = 240,
             seed: int = 0) -> CycleRun:
    """Nonlinear monthly simulation of the reduced model.

    Three details matter enough to state, because getting any of them wrong
    produces a series that looks plausible and contains no cycle at all.

    **The initial state has to balance.** Supply is normalised so that at the
    starting utilisation and product mix it exactly equals demand. An earlier
    version normalised capacity and demand to one and then divided supply by the
    mix term, which left the model in permanent 30% shortage: inventory pinned to
    its floor at step one, the price gap never changed sign, and the system crawled
    monotonically to a fixed point instead of oscillating. A cycle model that
    starts out of equilibrium does not cycle.

    **Product mix is exogenous here, not a fast feedback.** It follows a logistic
    adoption path from `mix_start` to `mix_target`. Making mix respond to price
    within the fast loop is tempting and wrong: mix shifting toward the
    wafer-hungry product *reduces* effective supply, so a price-driven mix response
    is positive feedback with no restoring force, and the model runs the mix to 100%
    and stays there. Treating adoption as a driver and letting it raise the loop
    gain over time is both better behaved and the more interesting claim.

    **The nonlinearities are the ones the industry actually has.** Utilisation is
    bounded — a fab cannot run above capacity, and below about 70% idling beats
    ramping — which is what turns a linearly unstable mode into a bounded
    oscillation rather than an explosion. Inventory cannot go negative, which is
    why shortages and gluts are not mirror images.

    `burn_in` steps are simulated and discarded.
    """
    if trade_ratio < 1.0:
        raise ValueError("trade ratio must be at least one")
    for name, v in (("mix_start", mix_start), ("mix_target", mix_target)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be a fraction")
    lo, hi = util_bounds
    if not 0.0 < lo < hi <= 1.0:
        raise ValueError("utilisation bounds must satisfy 0 < lo < hi <= 1")
    if not lo <= util_reference <= hi:
        raise ValueError("reference utilisation must lie inside the bounds")
    if mix_years <= 0:
        raise ValueError("mix_years must be positive")
    if not 0.0 <= retirement_annual < 1.0:
        raise ValueError("retirement rate must lie in [0, 1)")
    if price_floor <= 0:
        raise ValueError("price floor must be positive")
    retirement_monthly = 1.0 - (1.0 - retirement_annual) ** (1.0 / 12.0)
    if regime is not None:
        at, new_growth = regime
        if not 0 <= at < n_months:
            raise ValueError("regime break must fall inside the returned window")
        if new_growth <= -1.0:
            raise ValueError("post-break growth must exceed -100%/yr")

    rng = np.random.default_rng(seed)
    total = n_months + burn_in
    lag = max(fast_delay, slow_delay, demand_delay) + 1
    g_m = (1.0 + demand_growth_annual) ** (1.0 / 12.0) - 1.0
    # A step change in the *trend* rate of demand growth, which is what an AI-capex
    # or end-market regime shift looks like from inside this model. Indexed from the
    # start of the returned window, not the burn-in.
    g_after = (g_m if regime is None
               else (1.0 + regime[1]) ** (1.0 / 12.0) - 1.0)
    break_at = None if regime is None else lag + burn_in + regime[0]

    def divisor_of(m: float) -> float:
        """Wafers consumed per bit, relative to an all-planar mix."""
        return 1.0 + (trade_ratio - 1.0) * m

    # Logistic adoption of the wafer-hungry product, centred halfway through
    # `mix_years` and reaching most of the way to the target by the end of it.
    steps = np.arange(-lag, total)
    k = 8.0 / (mix_years * 12.0)
    mix_path = mix_start + (mix_target - mix_start) / (
        1.0 + np.exp(-k * (steps - 0.5 * mix_years * 12.0)))

    p = np.ones(total + lag)
    trend = np.ones(total + lag)
    inv = np.full(total + lag, inventory_target_weeks)
    util = np.full(total + lag, util_reference)
    mix = np.concatenate([mix_path, np.full(total + lag - len(mix_path), mix_target)])
    mix = mix[:total + lag]
    cap = np.ones(total + lag)
    sup = np.ones(total + lag)
    dem = np.ones(total + lag)

    # Supply is scaled so the starting point is in balance: at util_reference and
    # mix[lag], effective supply equals demand exactly.
    scale = divisor_of(mix[lag]) / util_reference

    for t in range(lag, total + lag):
        lp_fast = np.log(p[t - fast_delay])
        lp_slow = np.log(p[t - slow_delay])
        lp_dem = np.log(p[t - demand_delay])
        # Capacity plans follow the growth rate believed at commitment time, i.e.
        # slow_delay months ago — so after a demand break, capacity keeps building
        # to the *old* trend for slow_delay months. That lag is the whole point.
        g_now_cap = (g_m if (break_at is None or t - slow_delay < break_at)
                     else g_after)

        # Fast channel: utilisation responds to the price seen a few months ago,
        # bounded. This is the only fast feedback.
        util[t] = np.clip(util_reference + fast_gain * lp_fast, lo, hi)

        # Slow channel: capacity commissioned now was committed slow_delay months
        # ago, when the price was p[t - slow_delay]. Growth is floored at the
        # retirement rate: a bad enough price stops fabs being built, but existing
        # capacity does not evaporate.
        growth = g_now_cap * (1.0 + slow_gain * lp_slow)
        cap[t] = cap[t - 1] * (1.0 + max(growth, -retirement_monthly))

        sup[t] = cap[t] * util[t] * scale / divisor_of(mix[t])

        # Demand: trend growth, elastic to the *level* of price a few months ago,
        # plus a shock. An earlier version responded to the price *change*, which
        # left nothing to stop a glut: price fell toward zero, demand never picked
        # up, and inventory diverged to 1e12 weeks.
        #
        # Worth being precise about which term does the work, because the obvious
        # answer is wrong. Setting `demand_elasticity` to zero does *not* make the
        # model diverge: what pulls a glut back is trend demand growth outrunning a
        # capacity stock that shrinks toward the retirement floor while the price is
        # low. Elasticity strengthens that and shortens the swing; it is not what
        # bounds the system. Zero *trend growth* is the configuration with no
        # restoring force, and it pins at the shortage fixed point instead.
        g_now = g_m if (break_at is None or t < break_at) else g_after
        trend[t] = trend[t - 1] * (1.0 + g_now)
        dem[t] = trend[t] * np.exp(-demand_elasticity * lp_dem
                                   + shock_sd * rng.standard_normal())

        # One month of imbalance, expressed in weeks of supply. 52/12 weeks per
        # month: an earlier version used 52 and inflated every imbalance twelvefold.
        weeks = theta * (52.0 / 12.0) * (sup[t] - dem[t]) / max(dem[t], 1e-9)
        inv[t] = max(0.0, inv[t - 1] + weeks)
        gap = (inv[t] - inventory_target_weeks) / inventory_target_weeks
        # `decay` is price reverting toward long-run cost; without it the loop is a
        # double integrator and diverges at every gain. See `CycleModel.coefficients`.
        # Floored so a deep glut cannot drive the price to exactly zero and put a
        # -inf into the next step's log. The floor is three orders of magnitude
        # below the long-run level and is never approached at sane parameters; an
        # assertion below checks that.
        p[t] = max(price_floor, p[t - 1] ** (1.0 - decay) * np.exp(-kappa * gap))

    if p[lag:].min() <= price_floor * 1.001:
        raise RuntimeError(
            "price hit its floor, so the series is a divergence rather than a "
            "cycle. At sane parameters this is unreachable — it is a guard against "
            "a configuration where nothing pulls a glut back. Check that demand "
            "trend growth is positive and that the fast gain is not far above "
            "critical_gain(fast_delay).")
    s = slice(lag + burn_in, total + lag)
    return CycleRun(
        price=p[s], inventory=inv[s], utilisation=util[s], mix=mix[s],
        capacity=cap[s], supply=sup[s], demand=dem[s],
        params={"fast_delay": fast_delay, "slow_delay": slow_delay,
                "fast_gain": fast_gain, "slow_gain": slow_gain,
                "demand_delay": demand_delay,
                "demand_elasticity": demand_elasticity, "kappa": kappa,
                "theta": theta, "decay": decay, "trade_ratio": trade_ratio,
                "mix_start": mix_start, "mix_target": mix_target,
                "mix_years": mix_years, "util_reference": util_reference,
                "retirement_annual": retirement_annual,
                "demand_growth_annual": demand_growth_annual,
                "regime": regime,
                "shock_sd": shock_sd, "seed": seed})


def mix_drag(*, trade_ratio: float, mix_start: float, mix_target: float,
             mix_years: float) -> dict:
    """What shifting product mix costs in capacity terms.

    Moving a share of output from planar to stacked product, where the stacked kind
    consumes `trade_ratio` wafers per bit, multiplies the wafers needed for the
    *same* bits by `divisor(target) / divisor(start)`. Spread over the adoption
    period that is a standing drag on effective supply growth, payable out of
    capacity growth before a single extra bit reaches anyone.

    Separated out because it is the one place the stacked-memory transition enters
    the model as a level effect rather than a stability effect, and the two are easy
    to conflate.
    """
    if trade_ratio < 1.0 or mix_years <= 0:
        raise ValueError("trade ratio must be >= 1 and mix_years positive")
    d0 = 1.0 + (trade_ratio - 1.0) * mix_start
    d1 = 1.0 + (trade_ratio - 1.0) * mix_target
    factor = d1 / d0
    return {"wafer_factor": factor,
            "extra_wafers_pct": 100.0 * (factor - 1.0),
            "annual_drag_pct": 100.0 * (factor ** (1.0 / mix_years) - 1.0)}


def realised_period(x: np.ndarray, *, dt_months: float = 1.0,
                    detrend_order: int = 2) -> float:
    """Dominant period of a simulated series, from its periodogram.

    Reported alongside the linear `dominant_mode` period so the two can be
    compared. They need not agree: saturation shortens a cycle relative to its
    linear mode, and if they disagree badly the linear analysis is decoration.
    """
    y = np.log(np.asarray(x, float))
    t = np.arange(len(y))
    y = y - np.polyval(np.polyfit(t, y, detrend_order), t)
    y = y - y.mean()
    power = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freq = np.fft.rfftfreq(len(y), d=1.0)
    power[0] = 0.0
    k = int(np.argmax(power))
    return float("inf") if freq[k] <= 0 else float(dt_months / freq[k])


def model_from_simulation(*, fast_delay: int = 6, slow_delay: int = 60,
                          fast_gain: float = 0.55, slow_gain: float = 0.30,
                          demand_delay: int = 3,
                          demand_elasticity: float = 0.35,
                          kappa: float = 0.06, theta: float = 1.0,
                          decay: float = 0.04, trade_ratio: float = 3.0,
                          mix_target: float = 0.23) -> CycleModel:
    """The linearised counterpart of a `simulate` configuration.

    Kept next to `simulate` on purpose: two descriptions of one system invite
    silent divergence, and the whole use of the linear model is to predict the
    period the nonlinear one produces. exp011 asserts they agree.

    The fast channel's gain is amplified by the mix term. Shifting a share `m` of
    wafers onto product that consumes `r` times the wafer per bit divides effective
    supply by `1 + (r-1)m`, so the derivative of log supply with respect to the
    fast price signal picks up a factor from the mix response as well as from
    utilisation — which is why the trade ratio is a *stability* parameter and not
    just a level one.
    """
    divisor = 1.0 + (trade_ratio - 1.0) * mix_target
    mix_amplification = 1.0 + 0.5 * (trade_ratio - 1.0) / divisor
    return CycleModel(
        channels=[
            Channel("utilisation and mix", fast_gain * mix_amplification, fast_delay),
            Channel("capacity construction", slow_gain, slow_delay),
            Channel("buyer response", demand_elasticity, demand_delay),
        ],
        kappa=kappa, theta=theta, decay=decay, dt_months=1.0)
