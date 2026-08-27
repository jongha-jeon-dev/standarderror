"""Heat balance for a small mammal, and what a reported change in it must imply.

The claim this module exists to check has a standard shape in the obesity-drug
literature: *energy expenditure rose by X% and core temperature did not change*. That
is not one statement but two simultaneous constraints, and together they pin down a
third quantity nobody usually reports.

**Energy balance.** Measured energy expenditure is total heat production, because a
mouse in a metabolic cage does almost no external work. So a reported X% rise in
energy expenditure is a reported X% rise in watts of heat.

**Heat balance.** In steady state that heat has to leave, and for an animal below its
own body temperature it leaves at a rate `C (T_b - T_a)` — Scholander's model, where
`C` is thermal conductance. Rearranged, with ambient temperature fixed:

    EE' / EE = [C' (T_b' - T_a)] / [C (T_b - T_a)]

So exactly one of three things must be true of any "more calories, same temperature"
result: core temperature rose, conductance rose, or the animal was not in steady
state. `required_core_rise` and `required_conductance_rise` compute the first two, and
the second is worth noticing:

    at fixed T_b and T_a,  dC/C == dEE/EE  exactly.

The required conductance increase equals the reported energy-expenditure increase
whatever `C` actually is. That matters here because `C` is the constant the literature
agrees on least — published mouse values span 16 to 30 mW/degC depending on whether
the fit used resting or total expenditure and whether the cage had nesting material —
and the headline conclusion is invariant to it.

The thermostat is the interesting part
--------------------------------------
Below the thermoneutral point a mouse defends its core temperature by burning fuel
for heat. That heat is a *regulated* quantity, so a drug that produces heat by some
other route does not add to it — it displaces it. `substitution_prediction` works out
the consequence, which is counterintuitive enough to be the point of the exercise:

* Deep cold: thermoregulatory thermogenesis is large, so extra heat is absorbed
  one-for-one and **measured expenditure barely moves**.
* At or above the thermoneutral point: there is nothing to give back, so **all** of it
  appears, and it must then be dissipated or stored.

A drug's apparent effect on energy expenditure is therefore a function of the housing
temperature, and a three-temperature protocol that finds a *large* effect in the warm
and a *small* one in the cold has not distinguished "genuine metabolic activation"
from "thermogenesis": those are the same prediction. What discriminates them is the
heat-loss variable, and that is the one usually missing.

Where this model stops
----------------------
`Scholander.ee` extrapolates a straight line, and a real animal's curve flattens at
the thermoneutral point. Taking published fits over 22-28 degC and evaluating them at
30 degC does *not* reproduce the same papers' stated warm-versus-cold ratio — it comes
out about twenty percentage points low. That is not a bug to be tuned away; it is the
model telling you 30 degC is already at or above the thermoneutral point, where a
single-conductance line does not apply. `PUBLISHED_FITS` carries each fit's own valid
range and `Scholander.ee` warns outside it.

Sources for every constant are in `PUBLISHED_FITS` and in the module-level constants.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

__all__ = [
    "ANCOVAResult", "KCAL_PER_HOUR_TO_WATT", "PUBLISHED_FITS", "Scholander",
    "SubstitutionPrediction", "TELEMETRY_SD_C", "PROBE_SD_C",
    "detectable_difference", "heat_from_vo2", "lusk_caloric_value",
    "required_conductance_rise", "required_core_rise", "simulate_calorimetry",
    "substitution_prediction", "weir_kcal_per_day",
]

#: 1 kcal/h in watts. 1 kcal = 4184 J, 1 h = 3600 s.
KCAL_PER_HOUR_TO_WATT = 4184.0 / 3600.0

#: Core temperature, light phase, resting. Skop et al., Cell Reports 2020, 31:107501.
MOUSE_CORE_C = 35.6
#: Light-to-dark circadian amplitude, from Abreu-Vieira et al., Mol Metab 2015.
MOUSE_CORE_CIRCADIAN_AMPLITUDE_C = 1.1
#: Measurement noise. Meyer et al., Front Physiol 2017;8:520. The probe figure is the
#: reported within-strain SD for C57BL/6J; telemetry is the stated device accuracy.
PROBE_SD_C = 0.4
TELEMETRY_SD_C = 0.15
#: Post-mortem conductance, Abreu-Vieira et al. 2015 Fig 6A: 0.104 kcal/h/degC. A dead
#: mouse loses heat about four times faster than a live one at the same gradient, so
#: most of a mouse's insulation is vasomotor and postural rather than structural —
#: which is why an 18% conductance increase is physiologically unremarkable.
POST_MORTEM_CONDUCTANCE_W_PER_C = 0.104 * KCAL_PER_HOUR_TO_WATT


# ---------------------------------------------------------------- conversions

def lusk_caloric_value(rer: float) -> float:
    """kcal released per litre of O2 consumed, as a function of the exchange ratio.

    `3.815 + 1.232 * RER`, the form implemented in commercial calorimetry software.
    At RER 0.82 this gives 20.2 J/mL, which is the textbook 20.1 — the tests pin that
    agreement rather than the formula against itself.
    """
    r = float(rer)
    if not 0.6 <= r <= 1.1:
        raise ValueError(f"respiratory exchange ratio {r} is outside 0.6-1.1")
    return 3.815 + 1.232 * r


def heat_from_vo2(vo2_ml_per_hour: float, rer: float = 0.85) -> float:
    """Heat production in watts from oxygen consumption in mL/h."""
    v = float(vo2_ml_per_hour)
    if v < 0:
        raise ValueError("oxygen consumption cannot be negative")
    kcal_per_hour = lusk_caloric_value(rer) * v / 1000.0
    return kcal_per_hour * KCAL_PER_HOUR_TO_WATT


def weir_kcal_per_day(vo2_ml_per_min: float, vco2_ml_per_min: float) -> float:
    """Weir's equation, in the units it is usually mis-stated in.

    `1.440 * (3.9 * VO2 + 1.1 * VCO2)` with the volumes in **mL per minute**, giving
    kcal per day. The same coefficients appear in the literature as
    `3.941 * VO2 + 1.106 * VCO2` for kcal per *minute* with volumes in **litres** per
    minute. Feeding litres into the first form, or millilitres into the second, is
    wrong by a factor of 1000 in opposite directions, and neither version carries its
    units in the symbol. This one takes mL/min and says so in the name.
    """
    a, b = float(vo2_ml_per_min), float(vco2_ml_per_min)
    if a < 0 or b < 0:
        raise ValueError("gas exchange rates cannot be negative")
    return 1.440 * (3.9 * a + 1.1 * b)


# ---------------------------------------------------------------- the model

@dataclass(frozen=True)
class Scholander:
    """A published metabolic-rate-versus-ambient-temperature fit.

    `slope` and `intercept` are as papers report them, in kcal/h per degC and kcal/h,
    so the numbers in `PUBLISHED_FITS` can be checked against the source tables
    without unit arithmetic. Everything derived is exposed in SI.

    Below `thermoneutral_point` expenditure follows the line; at and above it the
    animal is on its basal plateau. `valid_range` is the ambient span the fit was
    estimated over, and `ee` warns outside it because the linear form is known to
    misbehave near the plateau.
    """
    label: str
    slope_kcal_h_per_c: float
    intercept_kcal_h: float
    thermoneutral_point_c: float
    body_mass_g: float
    valid_range_c: tuple[float, float]
    source: str
    measure: str = "resting energy expenditure"

    def __post_init__(self) -> None:
        if self.slope_kcal_h_per_c >= 0:
            raise ValueError("expenditure must fall as ambient temperature rises")
        if self.body_mass_g <= 0:
            raise ValueError("body mass must be positive")
        lo, hi = self.valid_range_c
        if not lo < hi:
            raise ValueError("valid_range_c must be (low, high)")

    # --- derived constants ---

    @property
    def conductance_w_per_c(self) -> float:
        """Thermal conductance. The Scholander slope *is* the conductance."""
        return -self.slope_kcal_h_per_c * KCAL_PER_HOUR_TO_WATT

    @property
    def conductance_mw_per_c_per_g(self) -> float:
        return 1000.0 * self.conductance_w_per_c / self.body_mass_g

    @property
    def x_intercept_c(self) -> float:
        """Where the fitted line reaches zero expenditure.

        Scholander's construction says this should land on core temperature. On real
        mouse fits it lands several degrees above it — 40.3 degC against a measured
        35.6 — so the extrapolation is not a thermometer. Both are offered to
        `required_core_rise` so a result can be quoted as a range.
        """
        return -self.intercept_kcal_h / self.slope_kcal_h_per_c

    def basal_kcal_h(self) -> float:
        """Expenditure on the plateau, i.e. at the thermoneutral point."""
        return self.intercept_kcal_h + self.slope_kcal_h_per_c \
            * self.thermoneutral_point_c

    # --- the curve ---

    def ee_kcal_h(self, t_ambient_c, *, warn: bool = True):
        """Energy expenditure at an ambient temperature, with a plateau above the TNP."""
        t = np.asarray(t_ambient_c, dtype=float)
        lo, hi = self.valid_range_c
        if warn and np.any((t < lo) | (t > hi)):
            warnings.warn(
                f"{self.label}: evaluating outside the fitted range "
                f"{lo}-{hi} degC; the linear form is unreliable near the plateau",
                stacklevel=2)
        line = self.intercept_kcal_h + self.slope_kcal_h_per_c * t
        out = np.where(t < self.thermoneutral_point_c, line, self.basal_kcal_h())
        return float(out) if out.ndim == 0 else out

    def ee_watts(self, t_ambient_c, **kw):
        return np.asarray(self.ee_kcal_h(t_ambient_c, **kw)) * KCAL_PER_HOUR_TO_WATT

    def cold_induced_kcal_h(self, t_ambient_c, **kw):
        """Thermoregulatory thermogenesis: expenditure above the basal plateau.

        This is the quantity a drug's heat can displace, and the reason a drug looks
        weaker in the cold than in the warm.
        """
        ee = np.asarray(self.ee_kcal_h(t_ambient_c, **kw))
        return np.maximum(ee - self.basal_kcal_h(), 0.0)

    def thermoregulatory_fraction(self, t_ambient_c, **kw):
        """Share of measured expenditure that is thermoregulatory."""
        ee = np.asarray(self.ee_kcal_h(t_ambient_c, **kw))
        return self.cold_induced_kcal_h(t_ambient_c, **kw) / ee


#: Fits taken from the source tables. Jacobsen et al. is the better match for a cage
#: with bedding and nesting; Abreu-Vieira et al. fitted total rather than resting
#: expenditure over a wider and colder span and reports a conductance nearly twice as
#: large. Both are kept, because the gap between them is the honest uncertainty on
#: every absolute number here, and because the headline result does not depend on it.
PUBLISHED_FITS: dict[str, Scholander] = {
    "chow, light phase": Scholander(
        label="chow, light phase", slope_kcal_h_per_c=-0.014,
        intercept_kcal_h=0.56, thermoneutral_point_c=30.0, body_mass_g=26.5,
        valid_range_c=(22.0, 28.0),
        source="Jacobsen et al., Commun Biol 2026, Table 2"),
    "diet-induced obese, light phase": Scholander(
        label="diet-induced obese, light phase", slope_kcal_h_per_c=-0.020,
        intercept_kcal_h=0.78, thermoneutral_point_c=30.0, body_mass_g=42.9,
        valid_range_c=(22.0, 28.0),
        source="Jacobsen et al., Commun Biol 2026, Table 2"),
    "diet-induced obese, dark phase": Scholander(
        label="diet-induced obese, dark phase", slope_kcal_h_per_c=-0.018,
        intercept_kcal_h=0.79, thermoneutral_point_c=33.0, body_mass_g=42.9,
        valid_range_c=(22.0, 28.0),
        source="Jacobsen et al., Commun Biol 2026, Table 2"),
    "chow, total expenditure": Scholander(
        label="chow, total expenditure", slope_kcal_h_per_c=-0.0254,
        intercept_kcal_h=0.94, thermoneutral_point_c=29.1, body_mass_g=26.9,
        valid_range_c=(18.0, 28.0),
        source="Abreu-Vieira et al., Mol Metab 2015, Table 1",
        measure="total energy expenditure"),
}


# ---------------------------------------------------------------- inversions

def required_core_rise(ee_ratio: float, *, t_body_c: float,
                       t_ambient_c: float) -> float:
    """Core-temperature rise a given expenditure ratio needs at constant conductance.

    `dT_b = (ratio - 1) * (T_b - T_a)`. Linear in the temperature gradient, so the
    answer depends strongly on ambient: the same 18% needs three degrees at 23 degC
    and under two at 30 degC, because the gradient it has to widen is smaller.
    """
    r = float(ee_ratio)
    if r <= 0:
        raise ValueError("the expenditure ratio must be positive")
    gradient = float(t_body_c) - float(t_ambient_c)
    if gradient <= 0:
        raise ValueError(
            "ambient is at or above body temperature; sensible heat loss reverses "
            "and this model does not apply")
    return (r - 1.0) * gradient


def required_conductance_rise(ee_ratio: float) -> float:
    """Fractional conductance rise needed at constant core and ambient temperature.

    Returns `ee_ratio - 1`, which is the whole point: with `T_b` and `T_a` fixed the
    conductance term is the only one left, so the required increase in heat loss
    equals the reported increase in heat production, and no value of the conductance
    itself enters. The test suite checks this against the full expression rather than
    against this one-liner.
    """
    r = float(ee_ratio)
    if r <= 0:
        raise ValueError("the expenditure ratio must be positive")
    return r - 1.0


@dataclass(frozen=True)
class SubstitutionPrediction:
    """What a defended core temperature does to a drug's apparent effect."""
    t_ambient_c: float
    measured_ratio: float
    thermoregulatory_fraction: float
    drug_watts: float
    displaced_watts: float

    @property
    def measured_rise(self) -> float:
        return self.measured_ratio - 1.0

    @property
    def fully_absorbed(self) -> bool:
        return self.displaced_watts >= self.drug_watts - 1e-12


def substitution_prediction(fit: Scholander, *, drug_fraction_of_basal: float,
                            t_ambient_c: float,
                            warn: bool = False) -> SubstitutionPrediction:
    """Measured expenditure change when extra heat displaces thermoregulation.

    The drug adds `drug_fraction_of_basal * basal` watts of heat. If the animal is
    defending its core temperature, total heat production is pinned by ambient, so the
    addition is absorbed by turning thermoregulatory thermogenesis down — up to the
    amount of thermogenesis that is running. Whatever exceeds that shows up.

    The prediction is therefore ~0 in the cold and the full drug effect at the
    thermoneutral point, which is the *opposite* of the intuition that a thermogenic
    drug should look biggest in the cold.
    """
    f = float(drug_fraction_of_basal)
    if f < 0:
        raise ValueError("the drug's heat contribution cannot be negative")
    basal = fit.basal_kcal_h()
    ee = float(np.asarray(fit.ee_kcal_h(t_ambient_c, warn=warn)))
    cit = float(np.asarray(fit.cold_induced_kcal_h(t_ambient_c, warn=warn)))
    drug = f * basal
    displaced = min(cit, drug)
    # Clamp the residual rather than letting `drug - displaced` leave a few times
    # machine epsilon behind: a "fully absorbed" case must report exactly no change,
    # or the monotonicity of the prediction across ambient temperature fails on noise
    # that is 1e-16 wide.
    residual = max(drug - displaced, 0.0)
    measured = (ee + residual) / ee
    to_w = KCAL_PER_HOUR_TO_WATT
    return SubstitutionPrediction(
        t_ambient_c=float(t_ambient_c), measured_ratio=measured,
        thermoregulatory_fraction=cit / ee,
        drug_watts=drug * to_w, displaced_watts=min(displaced, drug) * to_w)


def detectable_difference(sd: float, n_per_group: int, *, power: float = 0.80,
                          alpha: float = 0.05) -> float:
    """Smallest two-group difference a design could detect, in the units of `sd`.

    `(z_alpha/2 + z_beta) * sd * sqrt(2/n)`. Used here to ask what a reported null
    for body temperature actually excludes: a rectal probe on eight mice per group
    cannot see half a degree, and telemetry can.
    """
    if sd <= 0:
        raise ValueError("the standard deviation must be positive")
    if n_per_group < 2:
        raise ValueError("need at least two animals per group")
    if not 0.0 < power < 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("power and alpha must lie strictly in (0, 1)")
    z = float(norm.isf(alpha / 2.0)) + float(norm.isf(1.0 - power))
    return float(z * sd * np.sqrt(2.0 / n_per_group))


# ---------------------------------------------------------------- normalisation

@dataclass(frozen=True)
class ANCOVAResult:
    """Apparent group difference in expenditure under three normalisations."""
    per_animal: float
    per_gram: float
    ancova: float
    truth: float
    mass_difference_g: float = 0.0

    @property
    def per_gram_error(self) -> float:
        return self.per_gram - self.truth

    @property
    def ancova_error(self) -> float:
        return self.ancova - self.truth


def simulate_calorimetry(*, n_per_group: int, true_effect: float, rng,
                         mass_control_g: float = 45.0,
                         mass_treated_g: float = 41.0, mass_sd_g: float = 2.0,
                         intercept_w: float = 0.10, slope_w_per_g: float = 0.0035,
                         noise_w: float = 0.012) -> ANCOVAResult:
    """One indirect-calorimetry experiment, analysed three ways.

    Expenditure is generated as `intercept + slope * mass`, plus a genuine
    multiplicative treatment effect on the whole animal, plus noise. A **non-zero
    intercept** is the entire problem: with `EE = a + b m`, the ratio `EE/m` is
    `a/m + b`, which falls as mass rises. So dividing by body weight hands a lighter
    group a higher number for free — and a drug that causes weight loss makes its own
    treated group lighter.

    Returns the apparent effect under per-animal totals, per-gram normalisation, and
    ANCOVA with mass as a covariate, against the effect that was actually simulated.
    """
    if n_per_group < 3:
        raise ValueError("need at least three animals per group for a covariate fit")
    if intercept_w <= 0:
        raise ValueError(
            "with a zero intercept per-gram normalisation is unbiased and there is "
            "nothing to demonstrate; the artefact is the intercept")

    m0 = rng.normal(mass_control_g, mass_sd_g, n_per_group)
    m1 = rng.normal(mass_treated_g, mass_sd_g, n_per_group)
    base = lambda m: intercept_w + slope_w_per_g * m
    ee0 = base(m0) + rng.normal(0.0, noise_w, n_per_group)
    ee1 = base(m1) * (1.0 + float(true_effect)) + rng.normal(0.0, noise_w,
                                                             n_per_group)

    per_animal = float(np.mean(ee1) / np.mean(ee0) - 1.0)
    per_gram = float(np.mean(ee1 / m1) / np.mean(ee0 / m0) - 1.0)

    # ANCOVA: one common mass slope, a group offset, reported at the pooled mean mass.
    mass = np.concatenate([m0, m1])
    ee = np.concatenate([ee0, ee1])
    group = np.concatenate([np.zeros(n_per_group), np.ones(n_per_group)])
    design = np.column_stack([np.ones(mass.size), mass, group])
    coef, *_ = np.linalg.lstsq(design, ee, rcond=None)
    adjusted_control = coef[0] + coef[1] * mass.mean()
    ancova = float(coef[2] / adjusted_control)

    return ANCOVAResult(per_animal=per_animal, per_gram=per_gram, ancova=ancova,
                        truth=float(true_effect),
                        mass_difference_g=float(np.mean(m0) - np.mean(m1)))
