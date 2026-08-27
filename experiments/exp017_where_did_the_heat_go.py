"""exp017 — a metabolic claim, checked against the first law.

The news
--------
On 21 August 2026 UC Berkeley announced that TOFA, an oral acetyl-CoA carboxylase
inhibitor, produced fat loss without muscle loss in diet-induced obese mice, with the
paper in Science Advances the same week. The quantitative core is one sentence:
energy expenditure rose about 18% in treated mice, with no change in body temperature.

The paper is careful, and the post says so up front. Expenditure was measured at
three ambient temperatures — 23, 30 and 4 degC — adjusted by ANCOVA with lean *and*
fat mass as covariates, and body temperature was measured and reported as unchanged.
Brown-fat and browning genes were checked and found only marginally moved. That is a
better protocol than most of this literature runs.

What this post does
-------------------
"Expenditure rose 18% and temperature did not move" is not one measurement but two
simultaneous constraints. A mouse in a metabolic cage does almost no external work, so
the first is a statement about watts of heat produced; the second is a statement about
heat *stored*. Heat that is produced and not stored has left. So the pair pins a third
quantity — heat loss — and that is the one nobody measured.

Two legs, in the order that makes the second trustworthy.

1. **Is the 18% real?** The obvious way to manufacture it is per-gram normalisation on
   a treated group that lost weight: with expenditure linear in mass and a non-zero
   intercept, dividing by mass hands the lighter group a higher number for free. The
   simulation here says that alone would read +8.5% out of nothing at this study's
   18% weight loss. The paper did not do that — it used ANCOVA — and the simulation
   shows ANCOVA is unbiased in the same setting. So the number stands, and the rest of
   the post is worth writing.

2. **What does it require?** Below its thermoneutral point a mouse defends its core
   temperature by burning fuel for heat, so extra heat from any other source displaces
   that thermogenesis rather than adding to it. Run that model against the paper's
   three temperatures and it predicts a rise of 0% at 4 degC, 0% at 23 degC and the
   full 18% at 30 degC. Observed: 4%, 18%, 18%. The warm point matches exactly, the
   cold point nearly, and **23 degC is 18 points out**. At constant conductance that
   gap needs a core-temperature rise of 2.3 to 3.2 degC, which the paper's own null
   excludes even at rectal-probe precision. What is left is an 18% increase in heat
   loss — about 3-6% of the vasomotor range between a live mouse and a dead one, so
   physiologically unremarkable, and entirely unrecorded.

A second, smaller correction falls out of the same model: the paper reads the small
effect in the cold as evidence against a thermogenic mechanism. The thermostat
predicts a small effect in the cold for extra heat of *any* origin, so that datum
cannot discriminate the two hypotheses.

Discipline
----------
No verdict on the compound, the laboratory or the spin-out, and nothing here is a
claim about whether TOFA works or should be developed. Every input is a published
number. The subject is what a measured metabolic rate has to obey and which variable
the field does not print.

Run: `standarderror run exp017_where_did_the_heat_go --publish`
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.physio import heat
from standarderror.render import Post
from standarderror.viz import charts, theme

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed
CACHE = se.SETTINGS.build_dir / "cache" / "exp017.json"

# --- what the paper reports -----------------------------------------------------
PAPER = {
    "compound": "TOFA",
    "journal": "Science Advances",
    "doi": "10.1126/sciadv.aed3119",
    "announced": "21 August 2026",
    "strain": "male C57BL/6J",
    "diet": "60% kcal fat for twelve weeks",
    "n_per_group": (7, 8),
    "weight_loss": 0.18,
    "adjustment": "ANCOVA with lean and fat mass as covariates",
    "instrument": "CLAMS",
}
#: The three ambient temperatures and the expenditure increase reported at each.
#: Quoted from the Results: "~18% increase in energy expenditure ... under both room
#: temperature and thermoneutral conditions ... whereas energy expenditure increase
#: was minimal under cold exposure (4% increase in TOFA-treated mice)".
OBSERVED = {4.0: 0.04, 23.0: 0.18, 30.0: 0.18}
HEADLINE_RISE = 0.18
BODY_TEMPERATURE_P = 0.971

#: The fit used for the mouse this study actually ran, and the alternative whose
#: conductance is nearly twice as large. Both are carried everywhere.
FIT_KEY = "diet-induced obese, light phase"
ALT_KEY = "chow, total expenditure"
#: Core temperature, and the fitted line's x-intercept, as the two ends of the
#: temperature gradient the heat balance needs. They disagree by five degrees.
GRADIENT_ENDS = {"measured core temperature": heat.MOUSE_CORE_C,
                 "fitted x-intercept": 40.6}

AMBIENT_GRID = tuple(np.round(np.arange(2.0, 32.01, 0.5), 1))
MASS_GAPS_G = (0.0, 2.0, 4.0, 6.0, 8.0)
SIM_REPS = 2000
CONTROL_MASS_G = 45.0

SOURCES = [
    "Berkeley News, 21 August 2026 — the announcement and the '18% more energy' "
    "phrasing. <https://news.berkeley.edu/2026/08/21/a-promising-new-weight-loss-"
    "and-diabetes-treatment-helps-burn-fat-while-keeping-muscle/>",
    "The paper: 'A multi-functional oral small molecule targeting energy and lipid "
    "metabolism to treat obesity and related metabolic disorders', Science Advances "
    "2026, doi:10.1126/sciadv.aed3119 (open access; read via PubMed Central "
    "PMC13496183). Source of the three ambient temperatures, the ANCOVA adjustment, "
    "the body-temperature p-value and the group sizes. "
    "<https://pmc.ncbi.nlm.nih.gov/articles/PMC13496183/>",
    "Jacobsen et al., 'Neither rats nor mice have a broad thermoneutral zone: "
    "implications for physiological studies', Communications Biology 2026 — Table 2 "
    "Scholander fits used here (slope, intercept, x-intercept), the thermoneutral "
    "point, and the +101%/+104% warm-versus-cold figures. "
    "<https://www.nature.com/articles/s42003-026-09534-w>",
    "Abreu-Vieira et al., 'Integration of body temperature into the analysis of "
    "energy expenditure in the mouse', Mol Metab 2015;4:461-470 (open access) — the "
    "total-expenditure conductance, the lower critical temperature, and the "
    "post-mortem conductance that sets the vasomotor range. "
    "<https://pmc.ncbi.nlm.nih.gov/articles/PMC4443293/>",
    "Skop et al., 'Mouse Thermoregulation: Introducing the Concept of the "
    "Thermoneutral Point', Cell Reports 2020;31:107501 — resting core temperature, "
    "and the result that core temperature tracks ambient above the thermoneutral "
    "point. <https://www.cell.com/cell-reports/fulltext/S2211-1247(20)30391-0>",
    "Meyer et al., 'Body Temperature Measurements for Metabolic Phenotyping in "
    "Mice', Front Physiol 2017;8:520 — rectal-probe and telemetry precision, which "
    "sets what a reported null on body temperature can exclude. "
    "<https://www.frontiersin.org/journals/physiology/articles/10.3389/"
    "fphys.2017.00520/full>",
    "Tschop et al., 'A guide to analysis of mouse energy metabolism', Nature Methods "
    "2012;9:57-63 — why expenditure must not be divided by body or lean mass, and "
    "the ANCOVA the paper under discussion correctly used. "
    "<https://www.nature.com/articles/nmeth.1806>",
    "Kaiyala & Schwartz, 'Toward a More Complete (and Less Controversial) "
    "Understanding of Energy Expenditure and Its Role in Obesity Pathogenesis', "
    "Diabetes 2011;60:17-23 — the algebra of the ratio artefact. "
    "<https://diabetesjournals.org/diabetes/article/60/1/17/14966>",
    "Scholander et al., 'Heat regulation in some arctic and tropical mammals and "
    "birds', Biol Bull 1950;99:237-258 — the heat-balance model this rests on.",
]


def _config_key() -> str:
    blob = json.dumps({"v": 3, "observed": {str(k): v for k, v in OBSERVED.items()},
                       "fit": FIT_KEY, "alt": ALT_KEY, "ends": GRADIENT_ENDS,
                       "grid": list(AMBIENT_GRID), "gaps": list(MASS_GAPS_G),
                       "reps": SIM_REPS, "control_mass": CONTROL_MASS_G,
                       "n": PAPER["n_per_group"], "seed": SEED}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- computation

def substitution_curve(fit) -> dict:
    """Predicted measured rise against ambient temperature, plus the paper's points."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = []
        for ta in AMBIENT_GRID:
            p = heat.substitution_prediction(
                fit, drug_fraction_of_basal=HEADLINE_RISE, t_ambient_c=float(ta))
            rows.append({"t_ambient": float(ta), "predicted": p.measured_rise,
                         "thermoregulatory": p.thermoregulatory_fraction,
                         "ee_watts": float(fit.ee_watts(ta, warn=False))})
        points = {}
        for ta, obs in OBSERVED.items():
            p = heat.substitution_prediction(
                fit, drug_fraction_of_basal=HEADLINE_RISE, t_ambient_c=ta)
            points[str(ta)] = {
                "observed": obs, "predicted": p.measured_rise,
                "gap": obs - p.measured_rise,
                "thermoregulatory": p.thermoregulatory_fraction,
                "ee_watts": float(fit.ee_watts(ta, warn=False)),
                # The thermogenesis *available* to be displaced, which is the number
                # the post needs. `displaced_watts` is min(available, drug), so when
                # the effect is fully absorbed it equals the drug contribution and
                # printing the two side by side says nothing.
                "available_watts": float(
                    fit.cold_induced_kcal_h(ta, warn=False))
                * heat.KCAL_PER_HOUR_TO_WATT,
                "drug_watts": p.drug_watts, "displaced_watts": p.displaced_watts,
                "fully_absorbed": p.fully_absorbed,
            }
    return {"curve": rows, "points": points}


def resolutions() -> dict:
    """The ways a reported rise with a flat thermometer can be reconciled.

    Each temperature is inverted at the rise **reported at that temperature**, not at
    the headline 18%. The first version used 18% everywhere, which put a requirement
    on the 4 degC column that nobody had claimed — the cold reading was +4% — and made
    the cold bars the largest in the figure for no reason. Reading the chart caught it.
    """
    out = {"core_rise": {}, "conductance_rise": heat.required_conductance_rise(
        1.0 + HEADLINE_RISE), "detection": {}, "headroom": {}}
    for ta, observed in sorted(OBSERVED.items()):
        if ta >= min(GRADIENT_ENDS.values()):
            continue
        out["core_rise"][str(ta)] = {
            name: heat.required_core_rise(1.0 + observed, t_body_c=tb,
                                          t_ambient_c=ta)
            for name, tb in GRADIENT_ENDS.items()}
    for label, sd in (("rectal probe", heat.PROBE_SD_C),
                      ("implanted telemetry", heat.TELEMETRY_SD_C)):
        out["detection"][label] = {
            str(n): heat.detectable_difference(sd, n)
            for n in PAPER["n_per_group"]}
    for key, fit in heat.PUBLISHED_FITS.items():
        ratio = heat.POST_MORTEM_CONDUCTANCE_W_PER_C / fit.conductance_w_per_c
        out["headroom"][key] = {
            "conductance_mw_per_c": 1000.0 * fit.conductance_w_per_c,
            "dead_over_live": ratio,
            "share_of_range_used": HEADLINE_RISE / (ratio - 1.0),
        }
    return out


def normalisation_study(rng) -> dict:
    """How much of an 18% can per-gram normalisation manufacture from nothing."""
    rows = []
    for gap in MASS_GAPS_G:
        treated = CONTROL_MASS_G - gap
        for truth in (0.0, HEADLINE_RISE):
            runs = [heat.simulate_calorimetry(
                n_per_group=max(PAPER["n_per_group"]), true_effect=truth, rng=rng,
                mass_control_g=CONTROL_MASS_G, mass_treated_g=treated)
                for _ in range(SIM_REPS)]
            rows.append({
                "mass_gap_g": float(gap),
                "mass_gap_pct": float(gap / CONTROL_MASS_G),
                "truth": truth,
                "per_animal": float(np.mean([r.per_animal for r in runs])),
                "per_gram": float(np.mean([r.per_gram for r in runs])),
                "ancova": float(np.mean([r.ancova for r in runs])),
                "ancova_sd": float(np.std([r.ancova for r in runs], ddof=1)),
            })
    at_paper = {}
    gap = CONTROL_MASS_G * PAPER["weight_loss"]
    for truth in (0.0, HEADLINE_RISE):
        runs = [heat.simulate_calorimetry(
            n_per_group=max(PAPER["n_per_group"]), true_effect=truth, rng=rng,
            mass_control_g=CONTROL_MASS_G, mass_treated_g=CONTROL_MASS_G - gap)
            for _ in range(SIM_REPS)]
        at_paper[str(truth)] = {
            "per_gram": float(np.mean([r.per_gram for r in runs])),
            "ancova": float(np.mean([r.ancova for r in runs])),
        }
    return {"rows": rows, "at_paper_weight_loss": at_paper,
            "mass_gap_g": float(gap)}


def calibration() -> dict:
    """The control on the model itself: does the literature line reproduce its paper?

    Jacobsen et al. state resting expenditure is +101% (chow) and +104% (obese) at
    22 degC against 30 degC. Evaluating their own fitted line at those two points does
    not reproduce it, because 30 degC is outside the 22-28 degC fit range and at or
    above the thermoneutral point, where a straight line does not belong.
    """
    stated = {"chow, light phase": 1.01, "diet-induced obese, light phase": 1.04}
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, published in stated.items():
            fit = heat.PUBLISHED_FITS[key]
            implied = float(fit.ee_kcal_h(22.0, warn=False)) / float(
                fit.ee_kcal_h(30.0, warn=False)) - 1.0
            # And the same comparison at the top of the fitted range, where it should
            # and does behave.
            inside = float(fit.ee_kcal_h(22.0, warn=False)) / float(
                fit.ee_kcal_h(28.0, warn=False)) - 1.0
            out[key] = {"published": published, "implied_22_vs_30": implied,
                        "shortfall_pp": published - implied,
                        "implied_22_vs_28": inside,
                        "valid_range": list(fit.valid_range_c),
                        "thermoneutral_point": fit.thermoneutral_point_c}
    return out


def compute(*, force: bool = False, verbose: bool = True) -> dict:
    key = _config_key()
    if not force and CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            return cached

    t0 = time.time()

    def say(*a):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)

    rng = np.random.default_rng(SEED)
    fit = heat.PUBLISHED_FITS[FIT_KEY]
    say("calibrating the heat-balance model against its own source")
    cal = calibration()
    say("running the thermostat-substitution model over ambient temperature")
    sub = substitution_curve(fit)
    say("inverting the claim into the three candidate resolutions")
    res = resolutions()
    say("simulating the calorimetry normalisation problem")
    norm = normalisation_study(rng)

    out = {
        "key": key, "calibration": cal, "substitution": sub, "resolutions": res,
        "normalisation": norm,
        "fit": {"label": fit.label, "source": fit.source,
                "conductance_mw_per_c": 1000.0 * fit.conductance_w_per_c,
                "conductance_mw_per_c_per_g": fit.conductance_mw_per_c_per_g,
                "body_mass_g": fit.body_mass_g,
                "basal_watts": fit.basal_kcal_h() * heat.KCAL_PER_HOUR_TO_WATT,
                "thermoneutral_point": fit.thermoneutral_point_c,
                "valid_range": list(fit.valid_range_c)},
        "alt_conductance_mw_per_c":
            1000.0 * heat.PUBLISHED_FITS[ALT_KEY].conductance_w_per_c,
        "elapsed_s": round(time.time() - t0, 1),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    say("done")
    return out


# ---------------------------------------------------------------- presentation

def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown table with pipes inside cells escaped. See exp013's note."""
    def cell(x):
        return str(x).replace("|", r"\|")
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


TABLE_HEADER = ["ambient", "expenditure", "share thermoregulatory",
               "predicted rise", "reported rise", "gap"]


def table_rows(res: dict) -> list[list[str]]:
    pts = res["substitution"]["points"]
    rows = []
    for ta in sorted(OBSERVED):
        p = pts[str(ta)]
        rows.append([
            f"{ta:.0f} °C" + (" (thermoneutral)" if ta >= 30 else
                              " (room)" if ta > 10 else " (cold)"),
            f"{p['ee_watts']:.2f} W",
            f"{100 * p['thermoregulatory']:.0f}%",
            f"{100 * p['predicted']:+.0f}%",
            f"{100 * p['observed']:+.0f}%",
            f"{100 * p['gap']:+.0f} pp",
        ])
    return rows


def figures(res: dict) -> dict:
    import matplotlib.pyplot as plt

    figs = {}
    sub, resol, norm, cal = (res["substitution"], res["resolutions"],
                             res["normalisation"], res["calibration"])
    pts = sub["points"]
    fitinfo = res["fit"]

    # F1 — the model against the paper's three points. The whole argument in one
    # picture: the prediction is flat at zero until the thermoneutral point, the two
    # outer observations sit on it, and the middle one does not.
    # Two stacked panels, not one with a shaded band behind the line: the share of
    # expenditure that is thermoregulatory and the *rise* in expenditure are different
    # measures, and putting them on one axis is the twin-axis mistake wearing a
    # disguise. The first draft did exactly that.
    m = theme.apply("light", figsize=(7.2, 5.2))
    curve = pd.DataFrame(sub["curve"]).set_index("t_ambient")
    col = theme.series_colors(3, "light")
    fig, (top, ax) = plt.subplots(
        2, 1, sharex=True, height_ratios=[1.0, 2.0],
        gridspec_kw={"hspace": 0.28})

    top.fill_between(curve.index, 0.0, 100.0 * curve["thermoregulatory"],
                     color=m.diverging_mid, zorder=0)
    top.plot(curve.index, 100.0 * curve["thermoregulatory"], color=m.muted,
             lw=1.6, zorder=2)
    top.set_ylabel("thermoregulatory\nshare (%)")
    top.set_ylim(0, 88)
    top.annotate("how much heat the mouse was already making, to stay warm",
                 (0.02, 0.12), xycoords="axes fraction", fontsize=8.5,
                 color=m.muted)

    ax.plot(curve.index, 100.0 * curve["predicted"], color=col[0], lw=2.2,
            label="predicted, if the extra heat displaces thermogenesis", zorder=3)
    for ta in sorted(OBSERVED):
        p = pts[str(ta)]
        ax.plot([ta], [100.0 * p["observed"]], marker="o", ms=8, color=col[1],
                zorder=5, label="reported" if ta == 4.0 else None)
        if abs(p["gap"]) > 0.02:
            ax.annotate("", xy=(ta, 100.0 * p["observed"]),
                        xytext=(ta, 100.0 * p["predicted"]),
                        arrowprops={"arrowstyle": "->", "color": m.series[7],
                                    "lw": 1.8})
            ax.annotate(f"{100 * p['gap']:+.0f} pp", (ta, 100.0 * p["observed"]),
                        xytext=(8, -4), textcoords="offset points", fontsize=9.0,
                        color=m.series[7])
    for panel in (top, ax):
        panel.axvline(fitinfo["thermoneutral_point"], color=m.axis, lw=1.0,
                      ls=(0, (4, 3)), zorder=1)
    top.annotate("thermoneutral point", (fitinfo["thermoneutral_point"], 80.0),
                 xytext=(-6, 0), textcoords="offset points", ha="right",
                 fontsize=8.0, color=m.muted)
    ax.set_xlabel("ambient temperature (°C)")
    ax.set_ylabel("rise in measured\nenergy expenditure (%)")
    ax.set_ylim(-2, 24)
    # Let the tested layout machinery stack these: the title and subtitle go above the
    # *top* panel, the legend and source note below the bottom one. Placing them by
    # hand in figure coordinates put a three-line subtitle straight through the title.
    theme.finish(
        top, legend=False,
        title="Two of the three temperatures fit a thermostat. One does not.",
        subtitle=("A mouse below its thermoneutral point defends its core "
                  "temperature, so heat from any new source displaces the heat it was "
                  "already making. Extra heat should therefore be invisible in the "
                  "cold and fully visible in the warm."),
        mode="light")
    theme.finish(
        ax, legend_ncol=2,
        source=f"Prediction from {fitinfo['source']}, {fitinfo['label']}. Reported "
               f"values from the paper's Fig. 3.",
        mode="light")
    theme.save(fig, str(IMG / f"a9-f1-substitution.{EXT}"), mode="light")
    figs["substitution"] = charts.Figure(
        str(IMG / f"a9-f1-substitution.{EXT}"),
        alt=("Line chart against ambient temperature. A shaded band shows the share "
             "of expenditure spent on thermoregulation falling from about 75% in the "
             "cold to zero at 30 degrees. The predicted rise in measured expenditure "
             "stays at zero until 30 degrees then steps to 18%. Three reported points "
             "are marked: 4% at 4 degrees, 18% at 23 degrees and 18% at 30 degrees, "
             "with an arrow showing the 18-point gap at 23 degrees."),
        caption=(
            f"At {30:.0f} °C the model and the report agree exactly, because there is "
            f"no thermogenesis left to displace. At 4 °C they nearly agree. At 23 °C "
            f"the model says {100 * pts['23.0']['thermoregulatory']:.0f}% of "
            f"expenditure is thermoregulatory and could have absorbed the whole "
            f"effect — so the reported "
            f"{100 * pts['23.0']['observed']:.0f}% needs an explanation the "
            f"thermostat does not supply."),
        title="The thermostat prediction")

    # F2 — the three candidate resolutions, against what the study could detect.
    labels, values = [], []
    for ta in sorted(resol["core_rise"], key=float):
        obs = OBSERVED[float(ta)]
        for name, rise in resol["core_rise"][ta].items():
            labels.append(f"to explain the reported {100 * obs:+.0f}% at "
                          f"{float(ta):.0f} °C — gradient to {name}")
            values.append(rise)
    for label, byn in resol["detection"].items():
        n = str(max(PAPER["n_per_group"]))
        labels.append(f"largest rise this study could exclude — {label}, n={n}")
        values.append(byn[n])
    labels.append("circadian swing in core temperature, for scale")
    values.append(heat.MOUSE_CORE_CIRCADIAN_AMPLITUDE_C)
    fig_meta, _ = charts.ranked_bars(
        labels, values, sort="value", value_fmt=".2f",
        title="What a flat thermometer rules out, and what it leaves",
        subtitle=("Everything in degrees Celsius, and each temperature inverted at the "
                  "rise reported there. If heat loss did not change, the 23 °C reading "
                  "needs the two largest bars — several times what this study could "
                  "have seen, which is the two smallest."),
        xlabel="core temperature difference (°C)",
        source=f"Requirement from the heat balance; detection limits from "
               f"Meyer et al. 2017 precision at n={max(PAPER['n_per_group'])} per "
               f"group, 80% power, two-sided 0.05.",
        alt=("Horizontal bar chart in degrees Celsius. The two largest bars are the "
             "core temperature rise that constant heat loss would require to explain "
             "the 18% reported at 23 degrees, about 2.3 and 3.2 degrees. Below them "
             "come the 30-degree and 4-degree requirements, the circadian swing of 1.1 "
             "degrees, and the detection limits of a rectal probe and telemetry at "
             "0.56 and 0.21 degrees."),
        caption=(
            f"The reported null on body temperature is weak — a rectal probe on "
            f"{max(PAPER['n_per_group'])} mice per group cannot see "
            f"{resol['detection']['rectal probe'][str(max(PAPER['n_per_group']))]:.2f} °C "
            f"— and it is still comfortably strong enough to exclude the "
            f"{min(resol['core_rise']['23.0'].values()):.1f}–"
            f"{max(resol['core_rise']['23.0'].values()):.1f} °C that constant heat "
            f"loss would demand. So heat loss did not stay constant, and the "
            f"conclusion follows from the paper's own measurement rather than against "
            f"it."),
        path=str(IMG / f"a9-f2-routes.{EXT}"))
    figs["routes"] = fig_meta

    # F3 — the control on the number itself.
    rows = pd.DataFrame(norm["rows"])
    zero = rows[rows["truth"] == 0.0].set_index("mass_gap_pct")
    real = rows[rows["truth"] == HEADLINE_RISE].set_index("mass_gap_pct")
    frame = pd.DataFrame({
        "per-gram, no real effect": 100.0 * zero["per_gram"],
        "ANCOVA, no real effect": 100.0 * zero["ancova"],
        "per-gram, a real 18% effect": 100.0 * real["per_gram"],
        "ANCOVA, a real 18% effect": 100.0 * real["ancova"],
    })
    frame.index = 100.0 * frame.index

    def mark(fig, ax):
        # No reference line at 18%: the ANCOVA-with-a-real-effect series already sits
        # there by construction, and drawing both put the label on top of the line.
        ax.axvline(100.0 * PAPER["weight_loss"], color=theme.LIGHT.axis, lw=1.0,
                   ls=(0, (4, 3)))
        ax.annotate("this study's weight loss",
                    (100.0 * PAPER["weight_loss"], 2.0), xytext=(-6, 0),
                    textcoords="offset points", ha="right", fontsize=8.0,
                    color=theme.LIGHT.muted)

    fig_meta, _ = charts.lines(
        frame, direct_labels=False, decorate=mark,
        title="The easy way to invent an 18% — which this paper did not take",
        subtitle=("Apparent treatment effect against how much lighter the treated "
                  "group is. Dividing expenditure by body mass rewards the lighter "
                  "group for being lighter; ANCOVA with mass as a covariate does not."),
        xlabel="treated group lighter than control (%)",
        ylabel="apparent effect on energy expenditure (%)",
        source=f"{SIM_REPS:,} simulated experiments per point, "
               f"n={max(PAPER['n_per_group'])} per group, expenditure linear in mass "
               f"with a non-zero intercept.",
        alt=("Four lines against the percentage weight difference between groups. The "
             "two per-gram lines rise steeply as the treated group gets lighter; the "
             "two ANCOVA lines stay flat at zero and at eighteen percent "
             "respectively. A dashed vertical line marks this study's 18% weight "
             "loss."),
        caption=(
            f"At this study's {100 * PAPER['weight_loss']:.0f}% weight loss, per-gram "
            f"normalisation reads "
            f"{100 * norm['at_paper_weight_loss']['0.0']['per_gram']:+.1f}% out of a "
            f"true effect of exactly zero — about half the headline. ANCOVA reads "
            f"{100 * norm['at_paper_weight_loss']['0.0']['ancova']:+.1f}%. The paper "
            f"used ANCOVA with lean and fat mass as covariates — both, not just body "
            f"weight — so the 18% survives the obvious objection, which is why the "
            f"rest of this post is worth writing."),
        path=str(IMG / f"a9-f3-normalisation.{EXT}"))
    figs["normalisation"] = fig_meta

    # F4 — the control on the model. Where the straight line stops working.
    m = theme.apply("light", figsize=(7.2, 4.0))
    fig, ax = plt.subplots()
    keys = list(cal)
    x = np.arange(len(keys))
    w = 0.36
    col = theme.series_colors(2, "light")
    ax.bar(x - w / 2, [100.0 * cal[k]["published"] for k in keys], w,
           color=col[0], label="stated in the source paper")
    ax.bar(x + w / 2, [100.0 * cal[k]["implied_22_vs_30"] for k in keys], w,
           color=col[1], label="its own fitted line, evaluated at 22 and 30 °C")
    for i, k in enumerate(keys):
        ax.annotate(f"{cal[k]['shortfall_pp'] * 100:.0f} pp short",
                    (x[i] + w / 2, 100.0 * cal[k]["implied_22_vs_30"]),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=8.5, color=m.ink_secondary)
    ax.set_xticks(x, [k.replace(", light phase", "") for k in keys], fontsize=9)
    ax.set_ylabel("rise in resting expenditure, 22 °C versus 30 °C (%)")
    ax.grid(False)
    ax.yaxis.grid(True, color=m.grid, lw=0.6)
    ax.set_axisbelow(True)
    theme.finish(
        ax, title="The model fails where it should, and the failure is the point",
        subtitle=(
            f"Take a published Scholander fit and evaluate it at the two temperatures "
            f"its own paper compares. It undershoots by "
            f"{100 * min(cal[k]['shortfall_pp'] for k in keys):.0f} to "
            f"{100 * max(cal[k]['shortfall_pp'] for k in keys):.0f} points, because "
            f"30 °C is outside the {cal[keys[0]]['valid_range'][0]:.0f}-"
            f"{cal[keys[0]]['valid_range'][1]:.0f} °C fit range and at the "
            f"thermoneutral point, where a straight line does not belong."),
        source="Jacobsen et al., Communications Biology 2026: Table 2 fits against "
               "the paper's stated +101% and +104%.",
        mode="light")
    theme.save(fig, str(IMG / f"a9-f4-calibration.{EXT}"), mode="light")
    figs["calibration"] = charts.Figure(
        str(IMG / f"a9-f4-calibration.{EXT}"),
        alt=("Grouped bar chart for chow and obese mice. The stated rises of 101% and "
             "104% stand beside the rises implied by each paper's own fitted line, 80% "
             "and 89%, each labelled as about twenty points short."),
        caption=(
            "Worth reporting rather than tuning away. It means every absolute number "
            "in this post carries a real uncertainty near the thermoneutral point — "
            "and it does not touch the headline, because the conductance requirement "
            "is a ratio and cancels the constant entirely."),
        title="Calibration")

    # T1 — the three temperatures.
    fig_meta, _ = charts.table_image(
        table_rows(res), header=TABLE_HEADER,
        title="The three temperatures, and what each one can hide",
        subtitle=("Expenditure and thermoregulatory share from the published fit for "
                  "an obese mouse; predicted rise from the substitution model; "
                  "reported rise from the paper."),
        source=f"{res['fit']['source']}; the paper's Fig. 3.",
        bold_cols=(4, 5),
        alt=("A three-row table listing 4, 23 and 30 degrees Celsius against "
             "expenditure in watts, the thermoregulatory share, the predicted rise, "
             "the reported rise and the gap between them."),
        caption=("The gap column is the post. It is zero where thermoregulation is "
                 "zero, and largest where thermoregulation had the most room to "
                 "absorb the effect."),
        path=str(IMG / f"a9-t1-temperatures.{EXT}"))
    figs["table"] = fig_meta

    # HERO — production up, storage flat, so loss up.
    def furnace(panel, mm):
        from matplotlib.patches import Rectangle
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        panel.add_patch(Rectangle((2.2, 2.0), 5.6, 4.6, fc="none", ec=mm.ink,
                                  lw=2.4))
        panel.plot([3.4, 4.3, 5.0, 5.9, 6.6], [3.2, 4.4, 3.4, 4.6, 3.3],
                   color=mm.series[7], lw=2.4)
        for x0 in (3.2, 5.0, 6.8):
            panel.annotate("", xy=(x0, 8.6), xytext=(x0, 6.9),
                           arrowprops={"arrowstyle": "->", "color": mm.series[1],
                                       "lw": 2.2})

    def flat_thermometer(panel, mm):
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        panel.plot([2.0, 8.0], [5.0, 5.0], color=mm.ink, lw=2.8)
        panel.plot([2.0, 2.0], [4.2, 5.8], color=mm.ink, lw=2.2)
        panel.plot([8.0, 8.0], [4.2, 5.8], color=mm.ink, lw=2.2)
        panel.plot([5.0], [5.0], marker="o", ms=10, color=mm.series[0])
        panel.plot([2.6, 7.4], [7.6, 7.6], color=mm.grid, lw=2.2, ls=(0, (2, 2)))
        panel.plot([2.6, 7.4], [2.4, 2.4], color=mm.grid, lw=2.2, ls=(0, (2, 2)))

    def warm_tail(panel, mm):
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        t = np.linspace(0, np.pi, 60)
        panel.plot(3.6 + 1.7 * np.cos(t), 5.0 + 1.5 * np.sin(t), color=mm.ink,
                   lw=2.4)
        panel.plot([1.9, 1.9], [5.0, 3.6], color=mm.ink, lw=2.4)
        panel.plot([5.3, 5.3], [5.0, 3.6], color=mm.ink, lw=2.4)
        tail = np.linspace(0, 1, 40)
        panel.plot(5.3 + 3.4 * tail, 3.6 + 1.5 * np.sin(2.4 * tail),
                   color=mm.series[7], lw=3.0)
        for x0, y0 in ((7.0, 5.6), (8.2, 6.2)):
            panel.annotate("", xy=(x0 + 0.5, y0 + 1.4), xytext=(x0, y0),
                           arrowprops={"arrowstyle": "->", "color": mm.series[1],
                                       "lw": 2.0})

    fig_meta, _ = charts.strip_card(
        headline="More heat made, none stored. So more heat left.",
        panels=[
            (furnace, f"+{100 * HEADLINE_RISE:.0f}%", "heat produced"),
            (flat_thermometer, "0.0", "degrees of core warming"),
            (warm_tail,
             f"+{100 * resol['conductance_rise']:.0f}%", "heat lost, unmeasured"),
        ],
        note=("A mouse in a metabolic cage does no external work, so measured "
              "expenditure is heat production. Heat made and not stored has gone "
              "somewhere, and the required increase in heat loss equals the reported "
              "increase in expenditure exactly — whatever the conductance is."),
        footer="The Standard Error", mode="light",
        alt=("A three-panel hand-drawn strip. The first frame shows a furnace with "
             "flames and three arrows of heat rising out, marked plus 18 percent. The "
             "second shows a thermometer reading dead centre between two dashed "
             "limits, marked zero point zero. The third shows a mouse with a long warm "
             "tail shedding heat, marked plus 18 percent."),
        caption="",
        path=str(IMG / f"a9-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    res = compute(verbose=False)
    figs = figures(res)
    sub, resol, norm, cal = (res["substitution"], res["resolutions"],
                             res["normalisation"], res["calibration"])
    pts, fitinfo = sub["points"], res["fit"]
    room, cold, warm = pts["23.0"], pts["4.0"], pts["30.0"]
    n_max = max(PAPER["n_per_group"])
    probe = resol["detection"]["rectal probe"][str(n_max)]
    telem = resol["detection"]["implanted telemetry"][str(n_max)]
    core_lo = min(resol["core_rise"]["23.0"].values())
    core_hi = max(resol["core_rise"]["23.0"].values())
    head = resol["headroom"][FIT_KEY]
    zero_pg = norm["at_paper_weight_loss"]["0.0"]["per_gram"]
    zero_an = norm["at_paper_weight_loss"]["0.0"]["ancova"]
    cal_row = cal["diet-induced obese, light phase"]
    table_body = md_table(TABLE_HEADER, table_rows(res))

    # The spine, asserted rather than trusted.
    if not warm["fully_absorbed"] is False:
        raise AssertionError("at the thermoneutral point nothing should be absorbed")
    if abs(warm["gap"]) > 0.01:
        raise AssertionError(
            f"the warm point is supposed to match; gap is {100 * warm['gap']:.1f}pp")
    if room["gap"] < 0.10:
        raise AssertionError(
            f"the room-temperature anomaly is the post; it is only "
            f"{100 * room['gap']:.1f}pp")
    if core_lo <= probe:
        raise AssertionError(
            "the argument needs the required core rise to exceed what the study "
            "could detect, or the paper's null does not settle anything")
    if zero_pg < 0.03:
        raise AssertionError(
            "the normalisation leg needs per-gram to manufacture a visible effect")
    if abs(zero_an) > 0.02:
        raise AssertionError("ANCOVA is supposed to be unbiased here")
    if cal_row["shortfall_pp"] < 0.10:
        raise AssertionError(
            "the calibration failure is reported in the body; if it has gone away, "
            "the section describing it is now wrong")

    post = Post(
        title="Every \"Burns More Calories\" Result Is Missing One Variable",
        slug="every-burns-more-calories-result-is-missing-one-variable",
        subtitle=("A careful obesity paper reports 18% higher energy expenditure and "
                  "no change in body temperature, at three ambient temperatures. Two "
                  "of the three fit a simple heat balance. The third pins down a "
                  "quantity nobody measured."),
        summary=(
            f"On {PAPER['announced']}, Berkeley announced that {PAPER['compound']}, "
            f"an oral small molecule, made obese mice lose fat and not muscle, with "
            f"energy expenditure **{100 * HEADLINE_RISE:.0f}% higher** and body "
            f"temperature unchanged. A mouse in a metabolic cage does almost no "
            f"external work, so that is a statement about heat **produced** beside a "
            f"statement about heat **stored** — and heat produced but not stored has "
            f"left. The required increase in heat loss equals the reported increase "
            f"in expenditure **exactly**, independent of every physiological constant. "
            f"Below its thermoneutral point a mouse also defends its temperature by "
            f"making heat, so extra heat should displace that rather than add to it: "
            f"the model predicts a rise of 0%, 0% and "
            f"{100 * warm['predicted']:.0f}% at the paper's 4, 23 and 30 °C. Reported: "
            f"{100 * cold['observed']:.0f}%, {100 * room['observed']:.0f}%, "
            f"{100 * warm['observed']:.0f}%. The warm point matches exactly and "
            f"**23 °C is {100 * room['gap']:.0f} points out** — a gap that constant "
            f"heat loss could only close with {core_lo:.1f}–{core_hi:.1f} °C of "
            f"warming, which the paper's own thermometer excludes."),
        tags=["metabolism", "thermodynamics", "obesity", "mouse physiology",
              "measurement"],
        data_sources=SOURCES,
        licence_warnings=[
            "Every input is a published number: figures and text from an open-access "
            "paper, and constants from the thermal-physiology literature. No "
            "unpublished or proprietary data is used and none is needed.",
            "This post is about what a measured metabolic rate has to obey. It is not "
            "a judgement on the compound, the laboratory, or the company "
            "commercialising it, and it is not medical advice. The paper's central "
            "number survives the statistical objection in section three; the argument "
            "that follows is about what that number implies, not about whether it is "
            "true.",
        ],
        code_url="https://github.com/jongha-jeon-dev/standarderror",
        author="Jongha Jeon",
        reproducibility={
            "seed": SEED,
            "simulated experiments per point": f"{SIM_REPS:,}",
            "module": "standarderror.physio.heat",
            "tests": "tests/test_physio.py",
            "config hash": res["key"],
            "runtime": f"{res['elapsed_s']}s",
        },
        min_words=1500,
        max_words=2600,
        table_figures=[figs["table"]],
    )

    post.add("The claim, and the credit it deserves", f"""
On {PAPER['announced']} Berkeley announced that {PAPER['compound']}, an oral
acetyl-CoA carboxylase inhibitor, drove fat loss without muscle loss in
diet-induced obese mice. The paper landed in {PAPER['journal']} the same week. Its
quantitative core is one sentence: energy expenditure rose about
{100 * HEADLINE_RISE:.0f}% in treated mice, with no change in body temperature.

Before anything else, the protocol deserves saying out loud, because most of what
follows would be a cheap shot otherwise. Expenditure was measured at **three** ambient
temperatures — {', '.join(f'{t:.0f}' for t in sorted(OBSERVED))} °C. It was adjusted
by {PAPER['adjustment']}, not divided by body weight. Body temperature was measured
and reported, with a p-value ({BODY_TEMPERATURE_P}). Brown-fat and browning genes were
checked and found only marginally moved. Faecal energy was measured to rule out
malabsorption. That is a better protocol than this literature usually runs, and the
three-temperature design in particular is what makes the rest of this post possible.

The press release, separately, says "cells burn up to 18% more energy with no change
in physical activity" — which cannot be right, since cells do not have physical
activity. The {100 * HEADLINE_RISE:.0f}% is whole-animal calorimetry. The cell work is
real but carries no percentage. That is a release-writing problem, not a paper problem,
and it is the last time this post will mention it.
""".strip())

    post.add("One sentence, two constraints", f"""
"Expenditure rose {100 * HEADLINE_RISE:.0f}% and temperature did not move" reads like
one measurement. It is two, and they are not independent.

A mouse in a metabolic cage does almost no external work — it is not lifting anything
or going anywhere. So essentially all the energy it turns over leaves as heat, and a
reported {100 * HEADLINE_RISE:.0f}% rise in energy expenditure is a reported
{100 * HEADLINE_RISE:.0f}% rise in **watts of heat produced**. That is the first
constraint, and it is just the first law.

The second is that heat has to go somewhere. In steady state it leaves at a rate set
by the gradient to the room: **H = C (T_body − T_ambient)**, Scholander's model, with
`C` the thermal conductance. A flat body temperature is a statement that **no heat was
stored**. Put the two together and
**EE′/EE = C′(T_b′ − T_a) / C(T_b − T_a)**, so exactly one of three things has to be
true: core temperature rose, conductance rose, or the animal was not in steady state.

The second option has a property worth pausing on. At fixed body and ambient
temperature the bracket cancels, and

**dC / C = dEE / EE, exactly.**

The required increase in heat loss equals the reported increase in expenditure, and no
value of the conductance appears. That matters because `C` is the constant this
literature agrees on least: the four published mouse fits used here run from
{min(v['conductance_mw_per_c'] for v in resol['headroom'].values()):.0f} to
{max(v['conductance_mw_per_c'] for v in resol['headroom'].values()):.0f} mW/°C
depending on whether the fit used resting or total expenditure and whether the cage
had bedding. Nearly a factor of two, and the headline answer does not care.
""".strip())

    post.add("First control: is the 18% real?", f"""
Before asking what the number implies, it is worth trying to break it. The standard way
an expenditure effect appears out of nothing is normalisation.

Expenditure is roughly linear in body mass with a **non-zero intercept**:
`EE = a + b·m`. So the ratio `EE/m` is `a/m + b`, which falls as mass rises. Dividing
by body weight therefore hands the lighter group a higher number for free — and a drug
that causes weight loss makes its own treated group lighter. This is old and
well-documented, and it is why Tschöp and colleagues told the field in 2011 to use
ANCOVA with mass as a covariate instead.

Simulating it: with expenditure linear in mass, {SIM_REPS:,} experiments per point,
{n_max} animals a group, and **a true effect of exactly zero**, per-gram normalisation
reads {100 * zero_pg:+.1f}% at this study's {100 * PAPER['weight_loss']:.0f}% weight
loss. About half the headline, from arithmetic alone. ANCOVA on the same data reads
{100 * zero_an:+.1f}%.

The paper used ANCOVA, with lean **and** fat mass as covariates. So the
{100 * HEADLINE_RISE:.0f}% survives the obvious objection, and the rest of this post is
worth writing. It is also worth noting how the figure legend puts it — calorimetry was
run "when body weight between groups reached significance" — so the groups did differ
in weight at measurement, and the adjustment was doing real work rather than
decorating.
""".strip(), figures=[figs["normalisation"]])

    post.add("What a thermostat does to a drug's apparent effect", f"""
Now the part that makes the three-temperature design pay off.

Below its thermoneutral point a mouse is cold, and it defends its core temperature by
burning fuel to make heat. That thermoregulatory heat is a **regulated** quantity: the
animal makes exactly as much as the gradient demands. So a drug that produces heat by
some *other* route does not add to it. It displaces it. The mouse turns its
thermogenesis down by the same number of watts, total heat production is unchanged, and
**the calorimeter sees nothing**.

At or above the thermoneutral point there is no thermogenesis running, nothing to give
back, and the whole drug effect appears — and must then be dissipated or stored.

So the prediction runs opposite to the intuition. A heat-producing drug looks
**smallest** where thermogenesis is largest. Against the paper's three temperatures:

{table_body}

The warm point matches exactly, which it must — at
{fitinfo['thermoneutral_point']:.0f} °C the thermoregulatory share is zero, so the
model has no freedom there. The cold point nearly matches:
{100 * cold['observed']:.0f}% observed against 0% predicted, with
{100 * cold['thermoregulatory']:.0f}% of expenditure available to absorb the effect.

**23 °C is {100 * room['gap']:.0f} points out.** At room temperature the model says
{100 * room['thermoregulatory']:.0f}% of this mouse's expenditure is thermoregulatory:
{room['available_watts'] * 1000:.0f} mW of heat it was making on purpose, against a
drug contribution of {room['drug_watts'] * 1000:.0f} mW. Four times more thermogenesis
than the effect needed to hide in — and it did not hide.

One correction falls out of the same model. The paper reads the small cold-exposure
effect as evidence *against* a thermogenic mechanism: if TOFA worked through
thermogenesis, the argument goes, it should have shown up in the cold. The thermostat
says a small effect in the cold is what extra heat of **any** origin predicts, because
the regulated component absorbs it either way. That datum cannot separate the two
hypotheses. It is the {fitinfo['thermoneutral_point']:.0f} °C point that carries the
information, and there the paper's own reasoning is right.
""".strip(), figures=[figs["substitution"]])

    post.add("So where did the heat go?", f"""
Three candidate resolutions for the 23 °C gap, and two of them can be closed.

**Core temperature rose.** At constant conductance the requirement is
{core_lo:.1f}–{core_hi:.1f} °C, the range coming from whether you take the gradient to
measured core temperature ({heat.MOUSE_CORE_C} °C) or to the fitted line's x-intercept
({GRADIENT_ENDS['fitted x-intercept']} °C). Either way that is frank hyperthermia, well
above the {heat.MOUSE_CORE_CIRCADIAN_AMPLITUDE_C} °C a mouse swings across its own day.
The paper's null is weak — with a rectal probe at published precision and
{n_max} animals a group it can only exclude differences above {probe:.2f} °C, and
telemetry would manage {telem:.2f} °C — and it is still comfortably strong enough to
exclude {core_lo:.1f} °C. **Ruled out, by the paper's own measurement.**

**The animal was not in steady state.** Over a four-day recording, no.

**Heat loss rose.** What is left, and the requirement is
{100 * resol['conductance_rise']:.0f}% — the same number as the expenditure rise, by the
cancellation above. Is that a lot? A dead mouse loses heat about
{head['dead_over_live']:.1f} times faster than a live one at the same gradient, because
most of a mouse's insulation is vasomotor and postural rather than structural. So the
{100 * resol['conductance_rise']:.0f}% needed here is about
{100 * head['share_of_range_used']:.0f}% of the range the animal can traverse by
dilating its tail vessels, flattening its fur, changing posture, or using its nest less.
Physiologically trivial. Also completely unrecorded.

And it is not only the 23 °C reading. Invert each temperature at the rise reported
*there* — {100 * cold['observed']:.0f}% in the cold,
{100 * warm['observed']:.0f}% at thermoneutrality — and the requirement is
{min(resol['core_rise']['4.0'].values()):.1f}–{max(resol['core_rise']['4.0'].values()):.1f} °C
and {min(resol['core_rise']['30.0'].values()):.1f}–{max(resol['core_rise']['30.0'].values()):.1f} °C.
All six numbers exceed both detection limits. So heat loss must have risen at every
temperature the study ran, and the anomaly at 23 °C is a question about the
*mechanism*, not about whether the conclusion holds.

That is the finding. The study measured heat production and heat storage, and the two
together determine heat loss — so the experiment already contains the answer, in the
form of a number nobody wrote down. Tail temperature, skin temperature, posture scoring
or nest use would each have pinned it, and none is standard.
""".strip(), figures=[figs["routes"]])

    post.add("Second control: where the model breaks", f"""
The heat-balance argument uses a published fit, so it is worth checking the fit against
the paper it came from before trusting it anywhere.

It fails, and the failure is instructive. Jacobsen and colleagues state resting
expenditure is +{100 * (cal['chow, light phase']['published']):.0f}% for chow mice and
+{100 * cal_row['published']:.0f}% for obese mice at 22 °C against 30 °C. Take their
own fitted line and evaluate it at those two temperatures and you get
+{100 * cal['chow, light phase']['implied_22_vs_30']:.0f}% and
+{100 * cal_row['implied_22_vs_30']:.0f}% —
{100 * cal['chow, light phase']['shortfall_pp']:.0f} and
{100 * cal_row['shortfall_pp']:.0f} percentage points short respectively.

The reason is not an error in either place. The fit was estimated over
{cal_row['valid_range'][0]:.0f}–{cal_row['valid_range'][1]:.0f} °C, and 30 °C is
outside it and at the thermoneutral point, where the real curve flattens and a straight
line does not belong. Asked only what it was fitted for — 22 against
{cal_row['valid_range'][1]:.0f} °C — it gives
+{100 * cal_row['implied_22_vs_28']:.0f}%, and there is no published figure to check
that against, which is precisely the trouble: the comparison everyone quotes is the one
that straddles the plateau.

Two consequences, both reported rather than smoothed over. Every **absolute** number in
this post carries that uncertainty near thermoneutrality, which is why the required
core-temperature rise is quoted as a range and never to a tenth of a degree. And the
**headline** number does not carry it at all, because the conductance requirement is a
ratio in which the constant cancels. The one thing this model is least sure about is the
one thing the conclusion does not use.

While there: the same Communications Biology paper argues mice have no thermoneutral
*zone* at all, only a thermoneutral *point* — below it expenditure climbs, above it core
temperature climbs, with no span where both are flat. If that is right, 30 °C is not a
neutral resting condition either, and heat added there has even less room to go
anywhere but into the animal.
""".strip(), figures=[figs["calibration"]])

    post.add("What this changes", f"""
**Report a heat-loss variable.** Tail or skin temperature, posture, nest use, or a
measured conductance. Any one of them turns "expenditure rose and temperature did not"
from an unfalsifiable pair into a closed budget. It is the cheapest missing measurement
in the field: an infrared camera pointed at a tail.

**State the housing temperature next to every expenditure effect, and expect the effect
to depend on it.** Not as a caveat — as a coefficient. The same drug producing the same
watts of heat reads 0% in the cold and
{100 * warm['predicted']:.0f}% at thermoneutrality on this model. An effect size
reported without its ambient temperature is missing a unit.

**Do not read a small cold-exposure effect as evidence against thermogenesis.** The
thermostat predicts it for extra heat of any origin. The discriminating measurement is
at thermoneutrality, or it is the heat-loss variable.

**And keep using ANCOVA.** This paper did, and it is the reason its number survived the
first control. Per-gram normalisation would have manufactured {100 * zero_pg:+.1f}% out
of nothing at this study's weight loss.
""".strip())

    post.add("Where to be careful", f"""
**The substitution model is a one-compartment caricature.** A real mouse's
thermoregulatory response is not a perfectly efficient dial: brown fat has a time
constant, shivering has a cost, and displacement need not be one-for-one in watts. If
substitution is only partly efficient, some of the 23 °C effect is real and the required
heat-loss increase is smaller than {100 * resol['conductance_rise']:.0f}%. The
*direction* of the argument survives; the magnitude softens.

**I do not have the paper's absolute oxygen-consumption values.** They are in
supplementary figures I could not retrieve, so the watts in this post come from
published fits for mice of similar mass and diet rather than from these animals. If
these mice were metabolically far from that fit, the thermoregulatory shares in the
table move.

**Nor do I know how body temperature was measured.** The paper reports a p-value and no
method, so the {probe:.2f} °C and {telem:.2f} °C detection limits above bracket what it
could have been rather than stating what it was. The argument is built to survive
either: even the weaker instrument excludes what constant heat loss would need.

**23 °C is assumed to be below these animals' thermoneutral point.** That is what the
literature says for obese mice, and it is the load-bearing assumption for the anomaly.
If a 45 g mouse in a bare metabolic cage were effectively thermoneutral at 23 °C, the
gap closes and there is no anomaly — though the heat-loss requirement at 30 °C would
remain.

**One paper, one compound, one species.** The heat balance is general; that these three
particular numbers land where they do is not a claim about any other study. And nothing
here says whether {PAPER['compound']} works. The paper's central measurement survived
the control I ran at it; what I have computed is what that measurement obliges, and
then stopped.
""".strip())
    return post


if __name__ == "__main__":
    compute(force=bool(os.environ.get("SERR_FORCE")))
