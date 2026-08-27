"""exp016 — what a trial announcement still tells you when it prints no numbers.

The news
--------
On 19 August 2026 Merck and Moderna said the phase 3 INTerpath-001 trial of
intismeran autogene (mRNA-4157 / V940) plus pembrolizumab met both its
recurrence-free and distant-metastasis-free survival endpoints in 1,137 patients
with resected stage IIB-IV melanoma, at a pre-specified interim analysis. It is the
first phase 3 win for an individualised neoantigen therapy.

The announcement contains no hazard ratio, no confidence interval, no p-value, no
median, no event count and no per-arm patient numbers. That is normal and mostly
benign — the data go to a medical meeting under embargo, and a topline release is
not a paper. The question this post asks is narrower and purely arithmetic: given a
design and the single word *significant*, what is actually pinned down?

What it does
------------
Two computations, in the order that lets the second be trusted.

1. **The control.** For a two-arm survival comparison the standard error of
   `log HR` is `1/sqrt(D f)` with `D` the event count and `f = p_e (1 - p_e)` the
   split of events between arms. So an interval's *width* is a statement about `D`
   and nothing else, and `D` can be read back out of any report that prints one.
   Seven adjuvant melanoma trials published both, which makes this checkable rather
   than plausible. The recovery lands within a few percent on all seven, and the
   dominant error turns out to be the rounding of the printed interval to two
   decimal places — larger than every statistical approximation in the chain put
   together, including non-proportional hazards.

2. **The application.** The same identity run backwards turns "the boundary was
   crossed" into an upper bound on the hazard ratio that must have been observed.
   The bound is a function of the event count, which is exactly the number no
   announcement carries — so the honest output is a range, and the width of that
   range is the post's actual finding.

Discipline
----------
No verdict on the therapy, the companies, or anyone's share price. Everything here
is about the information content of a disclosure, and every input is a published
aggregate statistic from a press release, an FDA label or an open-access paper. The
computations say what a significance claim constrains; they say nothing about
whether the treatment is good, and a bound that permits a modest effect is not
evidence of a modest effect.

Run: `standarderror run exp016_the_number_not_printed --publish`
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

import standarderror as se
from standarderror.render import Post
from standarderror.uq import survival as sv
from standarderror.viz import charts, theme

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed
CACHE = se.SETTINGS.build_dir / "cache" / "exp016.json"

# --- the calibration set -------------------------------------------------------
# Adjuvant melanoma trials that published a hazard ratio, its interval, the level of
# that interval, and the event count, at their first reported analysis. The level
# matters: a group-sequential design reports the interval matching the alpha it has
# left, so two of these are not 95%.
#
# Sources are in `SOURCES` below. Numbers marked `soft` are less well attested —
# CheckMate-238's event split comes from an HTA report reading the paper's table
# rather than from the paper, and KEYNOTE-942's is back-calculated from published
# percentages. They are kept, flagged, and the post says which conclusions would
# survive dropping them.
TRIALS = [
    # name, ratio, hr, lo, hi, level, events treated, events control, decimals, soft
    ("KEYNOTE-054", 1.0, 0.57, 0.43, 0.74, 0.984, 135, 216, 2, False),
    ("KEYNOTE-716", 1.0, 0.65, 0.46, 0.92, 0.95, 54, 82, 2, False),
    ("CheckMate-238", 1.0, 0.65, 0.51, 0.83, 0.9756, 154, 206, 2, True),
    ("CheckMate-76K", 2.0, 0.42, 0.30, 0.59, 0.95, 66, 69, 2, False),
    ("COMBI-AD", 1.0, 0.47, 0.39, 0.58, 0.95, 166, 248, 2, False),
    ("EORTC 18071", 1.0, 0.75, 0.64, 0.90, 0.95, 234, 294, 2, False),
    ("KEYNOTE-942", 2.0, 0.561, 0.309, 1.017, 0.95, 24, 20, 3, True),
]

# --- the trial being read ------------------------------------------------------
INTERPATH = {
    "name": "INTerpath-001",
    "registry": "NCT05933577",
    "randomised": 1137,
    "ratio": 2.0,
    "announced": "19 August 2026",
    "endpoint": "investigator-assessed recurrence-free survival",
}
#: What the release *does* contain. Written down because the first draft of this post
#: called it "a press release with no numbers in it", which is false and falsifiable
#: by anyone who opens it: there are three dozen numbers on the page. The finding is
#: narrower and better — every efficacy number in it belongs to a different trial.
DISCLOSED = {
    "randomised": "1,137 enrolled",
    "allocation": "2:1",
    "dose": "intismeran 1 mg q3w up to nine doses; pembrolizumab 400 mg q6w up to "
            "nine cycles; about 56 weeks",
    "product": "up to 34 neoantigens per patient",
    "epidemiology": "330,000 cases worldwide in 2022; about 112,000 US cases and "
                    "over 8,500 US deaths projected for 2026",
    "phase2b_rfs": "HR 0.51 (95% CI 0.294-0.887)",
    "phase2b_dmfs": "HR 0.411 (95% CI 0.200-0.843)",
}
#: And what it withholds — the seven quantities that would let a reader size the new
#: result. Verified absent across the joint release, both companies' newsrooms,
#: EDGAR full-text search, the investor materials, the trade press and the registry
#: (NCT05933577 returns hasResults=false and has not been updated since Sept 2025).
WITHHELD = ("hazard ratio", "confidence interval", "p-value", "event count",
            "median RFS or DMFS", "Kaplan-Meier rate at any timepoint",
            "patients per arm")

#: The phase 2b this trial was built on, in all three published cuts. The release
#: quotes the **five-year** one; this post's prior is the **primary analysis**, and
#: the reason is in the body: the later cuts are conditioned on the trial having gone
#: on looking good, which is the selection this series keeps writing about.
#:
#: Note the primary analysis' interval crosses 1. The endpoint was met on a
#: pre-specified one-sided p of 0.0266 against a one-sided alpha of 0.10 — worth
#: knowing before reading "49% reduction" as the original result.
PHASE2B_CUTS = {
    "primary analysis": {
        "months": 23, "hr": 0.561, "lo": 0.309, "hi": 1.017,
        "p_one_sided": 0.0266, "alpha_one_sided": 0.10,
        "events_treated": 24, "events_control": 20,
        "dmfs_hr": 0.347, "dmfs_lo": 0.145, "dmfs_hi": 0.828,
        "source": "AACR 2023 CT001 / ASCO 2023 LBA9503",
    },
    "three-year update": {
        "months": 34.9, "hr": 0.510, "lo": 0.288, "hi": 0.906,
        "dmfs_hr": 0.384, "dmfs_lo": 0.172, "dmfs_hi": 0.858,
        "source": "ASCO 2024 LBA9512",
    },
    "five-year update": {
        "months": 60.3, "hr": 0.510, "lo": 0.294, "hi": 0.887,
        "p_one_sided": 0.0075,
        "dmfs_hr": 0.411, "dmfs_lo": 0.200, "dmfs_hi": 0.843,
        "source": "ASCO 2026; the cut the INTerpath-001 release quotes",
    },
}
PHASE2B = dict(PHASE2B_CUTS["primary analysis"], level=0.95,
               n_treated=107, n_control=50)
#: Registry enrolment, which disagrees with the release. So even the arm sizes this
#: post uses are an inference from 1,137 and 2:1, not a disclosed pair.
REGISTRY_ENROLMENT = 1089

#: Hazard ratios already on the record in the same setting, as the scale to read a
#: bound against. Clinical comparators, deliberately not analyst thresholds.
COMPARATORS = {
    "dabrafenib + trametinib (COMBI-AD)": 0.47,
    "pembrolizumab (KEYNOTE-054)": 0.57,
    "nivolumab vs ipilimumab (CheckMate-238)": 0.65,
    "ipilimumab (EORTC 18071)": 0.75,
}

# --- brackets on the one missing number ----------------------------------------
#: Route A: whatever the trial was powered for, an interim happens at a fraction of
#: it. Design hazard ratios a phase 3 in this setting would plausibly assume, and
#: the information fractions a first interim is usually placed at.
DESIGN_HRS = (0.60, 0.65, 0.70)
DESIGN_POWERS = (0.85, 0.90)
INFO_FRACTIONS = (0.50, 0.60, 0.70, 0.80)
#: Route B: the epidemiology. Annual recurrence hazard on adjuvant pembrolizumab in
#: this stage mix, mean follow-up at a mid-2026 cut for a trial that opened in July
#: 2023, and the true hazard ratio. Ranges, because each is a guess with a width.
CONTROL_HAZARDS = (0.09, 0.12, 0.15)
MEAN_FOLLOW_UP = (1.2, 1.6, 2.0)
TRUE_HRS = (0.55, 0.65, 0.75)
#: The event counts the figures sweep over.
EVENT_GRID = tuple(range(60, 601, 5))
HEADLINE_FRACTION = 0.60          # where a first interim most often sits

# --- the simulation study ------------------------------------------------------
SIM_REPS = 400
SIM_BASE = {"n_treated": 758, "n_control": 379, "control_rate": 0.12,
            "follow_up": 1.6}
SIM_CASES = (
    ("proportional hazards", {}),
    ("delayed effect, 6 months", {"delay": 0.5}),
    ("delayed effect, 12 months", {"delay": 1.0}),
    ("15%/yr dropout", {"dropout_rate": 0.15}),
    ("long follow-up, little censoring", {"follow_up": 12.0}),
)
SIM_HR = 0.65

SOURCES = [
    "Merck & Moderna, INTerpath-001 topline release, 19 August 2026 — design, "
    "1,137 randomised, 2:1 allocation, pre-specified interim, endpoints met. "
    "<https://www.merck.com/news/merck-and-moderna-announce-phase-3-interpath-001-"
    "trial-of-intismeran-autogene-plus-keytruda-met-endpoints-of-recurrence-free-"
    "survival-rfs-and-distant-metastasis-free-survival-dmfs-in-patient/>",
    "Moderna, the companion release, same date and same text. "
    "<https://news.modernatx.com/merck-and-moderna-announce-phase-3-interpath-001-"
    "trial-of-intismeran-plus-keytruda-met-endpoints-of-rfs-and-dmfs-in-melanoma>",
    "ClinicalTrials.gov NCT05933577 — INTerpath-001 registry record. Endpoint "
    "definition; estimated enrolment 1,089; hasResults false; last updated "
    "September 2025. <https://clinicaltrials.gov/study/NCT05933577>",
    "Absence of any phase 3 efficacy figure checked against: both companies' "
    "newsrooms, EDGAR full-text search for 'INTerpath-001' and 'intismeran' (no "
    "8-K on the readout from either filer), Moderna's investor-relations events "
    "listing and its IR Insights video page, the registry above, and the trade "
    "press. STAT: 'the drugmakers did not immediately release detailed data'. "
    "BioPharma Dive: 'The companies didn't release detailed data but intend to do "
    "so at an upcoming medical meeting.' "
    "<https://www.statnews.com/2026/08/19/mrna-cancer-vaccine-trial-melanoma-"
    "merck-moderna/>",
    "KEYNOTE-942 five-year update, ASCO 2026 — RFS HR 0.51 (95% CI 0.294-0.887), "
    "DMFS HR 0.411 (95% CI 0.200-0.843), median follow-up 60.3 months. This is the "
    "cut the INTerpath-001 release quotes. "
    "<https://www.merck.com/news/moderna-and-merck-present-5-year-data-for-"
    "intismeran-autogene-in-combination-with-keytruda-pembrolizumab-in-patients-"
    "with-high-risk-stage-iii-iv-melanoma-following-complete-resection-at-the-20/>",
    "KEYNOTE-942 three-year update, ASCO 2024 LBA9512 — RFS HR 0.510 (95% CI "
    "0.288-0.906). "
    "<https://www.merck.com/news/moderna-merck-announce-3-year-data-for-mrna-4157-"
    "v940-in-combination-with-keytruda-pembrolizumab-demonstrated-sustained-"
    "improvement-in-recurrence-free-survival-distant-metastasis-free-su/>",
    "KEYNOTE-054: Eggermont et al., N Engl J Med 2018;378:1789-1801 — HR 0.57, "
    "98.4% CI 0.43-0.74, 135/216 events. "
    "<https://www.nejm.org/doi/full/10.1056/NEJMoa1802357>",
    "KEYNOTE-716: Luke et al., Lancet 2022;399:1718-1729 — HR 0.65, 95% CI "
    "0.46-0.92, 54/82 events at the first interim. "
    "<https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(22)00562-1/"
    "fulltext>",
    "CheckMate-238: Weber et al., N Engl J Med 2017;377:1824-1835 — HR 0.65, "
    "97.56% CI 0.51-0.83. Event split read from the Scottish Medicines Consortium "
    "assessment SMC2112 rather than the paper. "
    "<https://scottishmedicines.org.uk/media/3958/nivolumab-opdivo-final-nov-2018-"
    "for-website.pdf>",
    "CheckMate-76K: Long et al., Nature Medicine 2023;29:2835-2843 (open access) — "
    "HR 0.42, 95% CI 0.30-0.59, 66/69 events, interim planned at ~123 events. "
    "<https://www.nature.com/articles/s41591-023-02583-2>",
    "COMBI-AD: FDA label for Tafinlar, NDA 202806/S-008, section 14.3 — HR 0.47, "
    "95% CI 0.39-0.58, 166/248 events. "
    "<https://www.accessdata.fda.gov/drugsatfda_docs/label/2018/202806s008lbl.pdf>",
    "EORTC 18071: FDA label for Yervoy, BLA 125377/S-073, section 14.2 — HR 0.75, "
    "95% CI 0.64-0.90, 234/294 events. "
    "<https://www.accessdata.fda.gov/drugsatfda_docs/label/2015/125377s073lbl.pdf>",
    "KEYNOTE-942 / mRNA-4157-P201 **primary analysis**, AACR 2023 CT001 — RFS "
    "HR 0.561 (95% CI 0.309-1.017), one-sided p=0.0266 against a one-sided alpha of "
    "0.10; 24/107 vs 20/50 events (22.4% vs 40.0%); 18-month RFS 78.6% vs 62.2%; "
    "median follow-up 23 and 24 months. This is the cut used as this post's prior. "
    "<https://s29.q4cdn.com/435878511/files/doc_presentations/2023/Apr/16/"
    "aacr-23_ct001-mrna4157_april-16.pdf>",
    "Schoenfeld, D. (1981), 'The asymptotic properties of nonparametric tests for "
    "comparing survival distributions', Biometrika 68:316-319 — the variance "
    "identity everything here rests on.",
    "Lan, K.K.G. & DeMets, D.L. (1983), 'Discrete sequential boundaries for "
    "clinical trials', Biometrika 70:659-663 — the alpha-spending boundaries.",
]


def _config_key() -> str:
    blob = json.dumps({"v": 4, "trials": TRIALS, "interpath": INTERPATH,
                       "phase2b": PHASE2B_CUTS, "design_hrs": DESIGN_HRS,
                       "powers": DESIGN_POWERS, "fractions": INFO_FRACTIONS,
                       "hazards": CONTROL_HAZARDS, "fu": MEAN_FOLLOW_UP,
                       "true_hrs": TRUE_HRS, "grid": EVENT_GRID,
                       "sim_reps": SIM_REPS, "sim_base": SIM_BASE,
                       "sim_cases": [c[0] for c in SIM_CASES],
                       "sim_hr": SIM_HR, "seed": SEED}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- computation

def recover_one(row) -> dict:
    """Everything the three inversions say about one published trial."""
    name, ratio, hr, lo, hi, level, d1, d0, decimals, soft = row
    reported = d1 + d0
    split = d1 / reported
    common = {"level": level, "ratio": ratio}
    modes = {
        "allocation": sv.implied_events(lo, hi, **common),
        "predicted": sv.implied_events(lo, hi, hazard_ratio=hr, **common),
        "observed": sv.implied_events(lo, hi, event_split=split, **common),
    }
    rounded = sv.events_from_rounded_ci(lo, hi, decimals=decimals,
                                        hazard_ratio=hr, **common)
    naive_level = sv.implied_events(lo, hi, level=0.95, ratio=ratio,
                                    hazard_ratio=hr)
    return {
        "name": name, "ratio": ratio, "hr": hr, "ci": [lo, hi], "level": level,
        "events_treated": d1, "events_control": d0, "reported": reported,
        "observed_split": split, "predicted_split": sv.event_fraction(hr, ratio),
        "recovered": modes,
        "error": {k: v / reported - 1.0 for k, v in modes.items()},
        "rounding": {"low": rounded.low, "high": rounded.high,
                     "relative_width": rounded.relative_width,
                     "contains": bool(rounded.low <= reported <= rounded.high)},
        "if_read_as_95pct": naive_level / reported - 1.0,
        "soft": soft, "decimals": decimals,
    }


def simulate_error_sources(rng) -> dict:
    """How much each modelling step costs, on data where the truth is known.

    The recovery is run on a simulated trial's own Cox output, so the reported
    interval is exact to full precision and the only error left is the modelling
    one. That isolates it from the rounding term, which is measured on the real
    reports instead — the two cannot be separated on published data.
    """
    out = {}
    for label, override in SIM_CASES:
        cfg = dict(SIM_BASE, **override)
        ratio = cfg["n_treated"] / cfg["n_control"]
        errs = {"predicted": [], "allocation": [], "observed": []}
        hats, counts = [], []
        for _ in range(SIM_REPS):
            t, e, a = sv.simulate_arms(hazard_ratio=SIM_HR, rng=rng, **cfg)
            fit = sv.cox_binary(t, e, a)
            hr = float(np.exp(fit.log_hr))
            lo, hi = sv.confidence_interval(hr, fit.events, ratio=ratio,
                                            use_predicted_split=False)
            # Rebuild the interval from the fit's own standard error, which is what
            # a trial report prints, then invert it as if the digits were all there.
            lo = hr * float(np.exp(-1.959964 * fit.se))
            hi = hr * float(np.exp(1.959964 * fit.se))
            errs["predicted"].append(
                sv.implied_events(lo, hi, ratio=ratio, hazard_ratio=hr)
                / fit.events - 1.0)
            errs["allocation"].append(
                sv.implied_events(lo, hi, ratio=ratio) / fit.events - 1.0)
            errs["observed"].append(
                sv.implied_events(lo, hi, ratio=ratio,
                                  event_split=fit.event_split)
                / fit.events - 1.0)
            hats.append(hr)
            counts.append(fit.events)
        out[label] = {
            "mean_error": {k: float(np.mean(v)) for k, v in errs.items()},
            "abs_error": {k: float(np.mean(np.abs(v))) for k, v in errs.items()},
            "mean_hr_estimate": float(np.mean(hats)),
            "mean_events": float(np.mean(counts)),
            "censored_fraction": 1.0 - float(np.mean(counts))
            / (cfg["n_treated"] + cfg["n_control"]),
        }
    return out


def event_count_brackets() -> dict:
    """Two independent routes to the number the announcement does not carry."""
    design = []
    for hr in DESIGN_HRS:
        for power in DESIGN_POWERS:
            final = sv.required_events(hr, power=power, alpha=0.05,
                                       ratio=INTERPATH["ratio"])
            for t in INFO_FRACTIONS:
                design.append({"design_hr": hr, "power": power,
                               "final": final, "fraction": t,
                               "events": final * t})
    n1 = INTERPATH["randomised"] * sv.allocation_fraction(INTERPATH["ratio"])
    n0 = INTERPATH["randomised"] - n1
    epi = []
    for lam in CONTROL_HAZARDS:
        for tbar in MEAN_FOLLOW_UP:
            for hr in TRUE_HRS:
                e0 = n0 * (1.0 - np.exp(-lam * tbar))
                e1 = n1 * (1.0 - np.exp(-lam * hr * tbar))
                epi.append({"control_hazard": lam, "follow_up": tbar,
                            "true_hr": hr, "events": e0 + e1})
    d_design = [r["events"] for r in design]
    d_epi = [r["events"] for r in epi]
    finals = [r["final"] for r in design]
    low = max(min(d_design), min(d_epi))
    high = min(max(d_design), max(d_epi))
    return {
        "design": design, "epidemiology": epi,
        "arms": {"treated": n1, "control": n0},
        "final_range": [float(min(finals)), float(max(finals))],
        "design_range": [float(min(d_design)), float(max(d_design))],
        "epi_range": [float(min(d_epi)), float(max(d_epi))],
        "overlap": [float(low), float(high)],
        "centre": float(np.sqrt(low * high)),
    }


def bound_curves(band) -> dict:
    """Upper bound on the observed hazard ratio, against the unknown event count."""
    grid = np.array(EVENT_GRID, dtype=float)
    ratio = INTERPATH["ratio"]
    curves = {}
    for t in INFO_FRACTIONS:
        z = sv.obrien_fleming_z(t)
        curves[f"O'Brien-Fleming, {int(100 * t)}% information"] = [
            sv.detectable_hr(d, z=z, ratio=ratio) for d in grid]
    curves[f"Pocock, {int(100 * HEADLINE_FRACTION)}% information"] = [
        sv.detectable_hr(d, z=sv.pocock_z(HEADLINE_FRACTION), ratio=ratio)
        for d in grid]
    z_head = sv.obrien_fleming_z(HEADLINE_FRACTION)
    lo, hi = band["overlap"]
    at_band = {"low": sv.detectable_hr(lo, z=z_head, ratio=ratio),
               "high": sv.detectable_hr(hi, z=z_head, ratio=ratio),
               "centre": sv.detectable_hr(band["centre"], z=z_head, ratio=ratio)}
    spans = {k: sv.ci_span(d, ratio=ratio, hazard_ratio=at_band[k])
             for k, d in (("low", lo), ("high", hi),
                          ("centre", band["centre"]))}
    # The bound *rises* with the event count: a bigger trial needs a smaller effect
    # to reach significance, so its bare significance claim says less about
    # magnitude. So the question "what does it take to exclude a hazard ratio above
    # X" has an upper answer, not a lower one — the first version of this searched
    # for a minimum event count and returned the bottom of the grid every time.
    head = np.array(curves[
        f"O'Brien-Fleming, {int(100 * HEADLINE_FRACTION)}% information"])
    ceilings = {}
    for target in (0.75, 0.70, 0.65, PHASE2B["hr"]):
        ok = grid[head <= target]
        ceilings[f"{target:.3f}"] = float(ok[-1]) if ok.size else 0.0
    # A worked interval at the centre of the band, so the prediction is concrete.
    span = spans["centre"]
    example_hr = 0.60
    example = [example_hr / np.sqrt(span), example_hr * np.sqrt(span)]
    return {"grid": grid.tolist(), "curves": curves, "at_band": at_band,
            "ci_spans": spans, "max_events_for_bound": ceilings,
            "example_hr": example_hr,
            "example_interval": [float(example[0]), float(example[1])],
            "boundary_z": {f"{t}": sv.obrien_fleming_z(t)
                           for t in INFO_FRACTIONS},
            "pocock_z": sv.pocock_z(HEADLINE_FRACTION)}


def one_bit(band) -> dict:
    """What the announcement adds to the phase 2b, measured in bits.

    Also run with each of the three published phase 2b cuts as the prior, because
    the release quotes the five-year one and a reader will reasonably ask whether the
    choice drives the answer. It does not: see `by_cut` in the result.
    """
    prior_se = (np.log(PHASE2B["hi"]) - np.log(PHASE2B["lo"])) / (
        2.0 * sv.z_for_level(PHASE2B["level"]))
    out = {"prior_log_se": float(prior_se), "cases": {}, "by_cut": {}}
    z_head = sv.obrien_fleming_z(HEADLINE_FRACTION)
    for cut, v in PHASE2B_CUTS.items():
        se = (np.log(v["hi"]) - np.log(v["lo"])) / (2.0 * sv.z_for_level(0.95))
        r = sv.posterior_given_significance(
            prior_hr=v["hr"], prior_log_se=float(se), events=band["centre"],
            z_boundary=z_head, ratio=INTERPATH["ratio"])
        out["by_cut"][cut] = {
            "prior_hr": v["hr"], "prior_ci": [v["lo"], v["hi"]],
            "prior_log_se": float(se), "months": v["months"],
            "posterior": r["posterior_summary"], "bits": r["bits"],
            "prior_predictive": r["prior_predictive"],
        }
    for name, d in (("low", band["overlap"][0]), ("centre", band["centre"]),
                    ("high", band["overlap"][1])):
        r = sv.posterior_given_significance(
            prior_hr=PHASE2B["hr"], prior_log_se=float(prior_se), events=d,
            z_boundary=sv.obrien_fleming_z(HEADLINE_FRACTION),
            ratio=INTERPATH["ratio"])
        out["cases"][name] = {
            "events": float(d),
            "prior": r["prior_summary"], "posterior": r["posterior_summary"],
            "bits": r["bits"], "prior_predictive": r["prior_predictive"],
            "surprisal_bits": r["surprisal_bits"],
        }
    r = sv.posterior_given_significance(
        prior_hr=PHASE2B["hr"], prior_log_se=float(prior_se),
        events=band["centre"], z_boundary=sv.obrien_fleming_z(HEADLINE_FRACTION),
        ratio=INTERPATH["ratio"])
    out["density"] = {"grid": r["grid"].tolist(), "prior": r["prior"].tolist(),
                      "posterior": r["posterior"].tolist()}
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
    say("recovering event counts from seven published reports")
    trials = [recover_one(r) for r in TRIALS]
    say("simulating the modelling error sources")
    sim = simulate_error_sources(rng)
    say("bracketing the missing event count")
    band = event_count_brackets()
    say("inverting the significance claim")
    bounds = bound_curves(band)
    bits = one_bit(band)

    abs_err = {k: float(np.mean([abs(t["error"][k]) for t in trials]))
               for k in ("allocation", "predicted", "observed")}
    hard = [t for t in trials if not t["soft"]]
    out = {
        "key": key, "trials": trials, "sim": sim, "band": band,
        "bounds": bounds, "bits": bits,
        "summary": {
            "abs_error": abs_err,
            "abs_error_hard_only": {
                k: float(np.mean([abs(t["error"][k]) for t in hard]))
                for k in ("allocation", "predicted", "observed")},
            "max_error_predicted": float(max(abs(t["error"]["predicted"])
                                             for t in trials)),
            "rounding_width": float(np.mean([t["rounding"]["relative_width"]
                                             for t in trials
                                             if t["decimals"] == 2])),
            "rounding_hits": int(sum(t["rounding"]["contains"] for t in trials)),
            "n_trials": len(trials),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    say("done")
    return out


# ---------------------------------------------------------------- presentation

def _pct(x, digits=1):
    return f"{100 * x:+.{digits}f}%"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown table with pipes inside cells escaped. See exp013's note."""
    def cell(x):
        return str(x).replace("|", r"\|")
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


TABLE_HEADER = ["trial", "HR (printed CI)", "level", "events reported",
                "recovered", "error", "rounding bracket"]


def table_rows(res: dict) -> list[list[str]]:
    rows = []
    for t in res["trials"]:
        lo, hi = t["ci"]
        d = t["decimals"]
        rows.append([
            t["name"] + (" *" if t["soft"] else ""),
            f"{t['hr']:.{d}f} ({lo:.{d}f}-{hi:.{d}f})",
            f"{100 * t['level']:.4g}%",
            f"{t['reported']}",
            f"{t['recovered']['predicted']:.0f}",
            _pct(t["error"]["predicted"]),
            f"{t['rounding']['low']:.0f}-{t['rounding']['high']:.0f}"
            + ("" if t["rounding"]["contains"] else " (misses)"),
        ])
    return rows


def figures(res: dict) -> dict:
    figs = {}
    trials, band, bounds, bits = (res["trials"], res["band"], res["bounds"],
                                  res["bits"])
    s = res["summary"]

    # F1 — the control. Percentage error of the recovered event count, with the
    # bracket the printed rounding allows as the error bar. The finding is that the
    # bars are small and the whiskers cover them: the recovery is limited by how
    # many digits the report printed, not by the statistics.
    names = [t["name"] + (" *" if t["soft"] else "") for t in trials]
    errs = [100 * t["error"]["predicted"] for t in trials]
    reach = [100 * t["rounding"]["relative_width"] for t in trials]
    fig_meta, _ = charts.ranked_bars(
        names, errs, errors=np.array([reach, reach]), signed=True, sort="value",
        value_fmt="+.1f",
        title="The event count a report did not print, recovered from the one it did",
        subtitle=("Error in the event count implied by each trial's printed hazard "
                  "ratio interval, against the count the trial actually reported. "
                  "Whiskers are the range the printed rounding alone permits. "
                  "* = event split back-calculated from percentages, not stated."),
        xlabel="error in the recovered event count (%)",
        source="Sources listed in the post. Recovery uses only the printed "
               "interval, its stated level, and the allocation ratio.",
        alt=("Horizontal bar chart of seven adjuvant melanoma trials. Errors in the "
             "recovered event count run from about -3% to +8%, and the whiskers "
             "showing what the printed two-decimal rounding permits are wider than "
             "most of the bars."),
        caption=(f"Mean absolute error {100 * s['abs_error']['predicted']:.1f}% "
                 f"across {s['n_trials']} trials, worst case "
                 f"{100 * s['max_error_predicted']:.1f}%. The rounding bracket "
                 f"contains the reported count in {s['rounding_hits']} of "
                 f"{s['n_trials']}; the one it misses is KEYNOTE-942, whose "
                 f"event count is itself back-calculated from published "
                 f"percentages and misses by less than one event."),
        path=str(IMG / f"a8-f1-recovery.{EXT}"))
    figs["recovery"] = fig_meta

    # F2 — where the error comes from, everything on one axis so the ranking is the
    # picture. Rounding is measured on the real reports; the rest is simulated,
    # because on published data the two cannot be told apart.
    alloc = [100 * abs(c["mean_error"]["allocation"]) for c in res["sim"].values()]
    labels = ["mistake: reading a 98.4% interval as 95%",
              "mistake: allocation ratio where the event split belongs",
              "the report: printing the interval to two decimals"]
    values = [100 * abs(trials[0]["if_read_as_95pct"]),
              float(np.mean(alloc)), 100 * s["rounding_width"]]
    # Only the allocation row spans a range worth drawing: its size is a function of
    # the effect, so it is one bar with the case-to-case spread as a whisker.
    err_lo = [0.0, float(np.mean(alloc) - min(alloc)), 0.0]
    err_hi = [0.0, float(max(alloc) - np.mean(alloc)), 0.0]
    for name, case in res["sim"].items():
        labels.append(f"statistics: {name}")
        values.append(100 * abs(case["mean_error"]["predicted"]))
        err_lo.append(0.0)
        err_hi.append(0.0)
    fig_meta, _ = charts.ranked_bars(
        labels, values, errors=np.array([err_lo, err_hi]), sort="value",
        value_fmt=".1f",
        title="Every statistical approximation is smaller than the rounding",
        subtitle=("Absolute error in the recovered event count. Two of these are "
                  "reader mistakes, one is a limit the report imposes by printing "
                  "two decimal places, and the rest are statistical "
                  "approximations. Of those, the only one above 1% describes a "
                  "regime none of the seven reports is in."),
        xlabel="absolute error in the recovered event count (%)",
        source=(f"Rounding and level terms measured on the published reports; the "
                f"rest from {SIM_REPS} simulated trials per row at "
                f"{SIM_BASE['n_treated']}:{SIM_BASE['n_control']} allocation, "
                f"true hazard ratio {SIM_HR}. Whisker on the allocation row is the "
                f"spread across those cases."),
        alt=("Horizontal bar chart ranking error sources. Misreading the confidence "
             "level is by far the largest at about 36%; two-decimal printing and "
             "using the allocation ratio instead of the event split follow at "
             "about 7% each. The five statistical rows are all at or below 4%, and "
             "four of them are under half a percent, including both "
             "non-proportional-hazards cases."),
        caption=(
            "The delayed-effect rows are the surprise. A twelve-month delay drags "
            f"the estimated hazard ratio from "
            f"{res['sim']['proportional hazards']['mean_hr_estimate']:.2f} to "
            f"{res['sim']['delayed effect, 12 months']['mean_hr_estimate']:.2f} — "
            "the hazard ratio stops being a parameter and becomes a follow-up "
            "weighted average — and the event-count recovery does not notice, "
            "because the split it needs is predicted from the reported ratio, and "
            "the reported ratio is exactly what governs the split. The one "
            "statistical row above 1% is long follow-up with little censoring, "
            "which describes none of the seven reports: all of them censor most of "
            "their patients."),
        path=str(IMG / f"a8-f2-errors.{EXT}"))
    figs["errors"] = fig_meta

    # F3 — the application. The bound, against the number nobody published.
    m = theme.apply("light", figsize=(7.2, 4.4))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    grid = np.array(bounds["grid"], float)
    lo, hi = band["overlap"]
    ax.axvspan(lo, hi, color=m.diverging_mid, zorder=0)
    ax.annotate("event counts both routes allow",
                (np.sqrt(lo * hi), 0.955), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=8.0, color=m.muted)
    keys = [k for k in bounds["curves"] if k.startswith("O'Brien")]
    for col, k in zip(theme.series_colors(len(keys) + 1, "light"), keys):
        ax.plot(grid, bounds["curves"][k], color=col, lw=1.9, label=k, zorder=3)
    pk = [k for k in bounds["curves"] if k.startswith("Pocock")][0]
    ax.plot(grid, bounds["curves"][pk], color=m.muted, lw=1.6,
            ls=(0, (4, 2.5)), label=pk, zorder=3)
    for name, hr in COMPARATORS.items():
        ax.axhline(hr, color=m.grid, lw=1.0, zorder=1)
        ax.annotate(f"{name.split(' (')[0]}  {hr:.2f}", (grid[-1], hr),
                    xytext=(-2, 3), textcoords="offset points", ha="right",
                    va="bottom", fontsize=7.5, color=m.muted)
    ax.set_xlabel("total recurrence-free survival events at the interim (unknown)")
    ax.set_ylabel("largest hazard ratio consistent with crossing the boundary")
    ax.set_ylim(0.35, 0.95)
    theme.finish(
        ax, legend_ncol=2,
        title="What \"statistically significant\" bounds, and what it does not",
        subtitle=("Every curve is an upper bound: crossing an interim efficacy "
                  "boundary means the observed hazard ratio was at least this "
                  "small. Which curve, and where on it, depends on the event count "
                  "and the spending function — neither of which was disclosed."),
        source=f"{INTERPATH['name']} ({INTERPATH['registry']}), "
               f"{INTERPATH['randomised']:,} randomised at "
               f"{INTERPATH['ratio']:.0f}:1. Comparator hazard ratios as published.",
        mode="light")
    theme.save(fig, str(IMG / f"a8-f3-bound.{EXT}"), mode="light")
    figs["bound"] = charts.Figure(
        str(IMG / f"a8-f3-bound.{EXT}"),
        alt=("Line chart. Four O'Brien-Fleming curves and one Pocock curve rise "
             "from about 0.5 to 0.85 as the event count grows from 60 to 600, with "
             "a shaded vertical band marking the plausible event range and "
             "horizontal reference lines at the hazard ratios of four approved "
             "adjuvant melanoma regimens."),
        caption=(
            f"Inside the shaded band the bound runs from "
            f"{bounds['at_band']['low']:.2f} to {bounds['at_band']['high']:.2f}. "
            f"Note the direction: the curves rise, so a larger trial's bare "
            f"significance claim says less about magnitude, not more — the bound "
            f"only excludes a hazard ratio above 0.65 if the interim had at most "
            f"{bounds['max_events_for_bound']['0.650']:.0f} events, and above 0.70 "
            f"only up to {bounds['max_events_for_bound']['0.700']:.0f}. The dashed "
            f"Pocock curve is the same claim under a different spending function, "
            f"worth about "
            f"{bounds['curves'][pk][len(grid) // 3] - bounds['curves'][keys[1]][len(grid) // 3]:.2f} "
            f"of hazard ratio on its own."),
        title="What significance bounds")

    # F4 — the one bit. Prior from the phase 2b, updated by the announcement alone.
    m = theme.apply("light", figsize=(7.2, 4.0))
    fig, ax = plt.subplots()
    g = np.array(bits["density"]["grid"], float)
    keep = (g >= 0.15) & (g <= 1.45)
    col = theme.series_colors(2, "light")
    ax.plot(g[keep], np.array(bits["density"]["prior"])[keep], color=m.muted,
            lw=2.6, alpha=0.7, label="phase 2b alone")
    ax.plot(g[keep], np.array(bits["density"]["posterior"])[keep], color=col[0],
            lw=2.2, label="after the announcement")
    ax.axvline(1.0, color=m.axis, lw=1.0, ls=(0, (4, 3)))
    ax.annotate("no effect", (1.0, 0.98), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", ha="left", va="top",
                fontsize=8.0, color=m.muted)
    ax.set_xlabel("true hazard ratio")
    ax.set_ylabel("density (per unit log hazard ratio)")
    ax.set_yticks([])
    c = bits["cases"]["centre"]
    theme.finish(
        ax, title=f"{c['bits']:.2f} bits, spent almost entirely on the upper tail",
        subtitle=("The observation is not an estimate — it is the event that a "
                  "boundary was crossed. It moves the median from "
                  f"{c['prior']['median']:.2f} to {c['posterior']['median']:.2f} "
                  f"and the upper end of the interval from "
                  f"{c['prior']['high']:.2f} to {c['posterior']['high']:.2f}."),
        source=(f"Prior: KEYNOTE-942 primary analysis, hazard ratio "
                f"{PHASE2B['hr']:.3f} (95% CI {PHASE2B['lo']:.3f}-"
                f"{PHASE2B['hi']:.3f}). Likelihood: crossing an O'Brien-Fleming "
                f"boundary at {int(100 * HEADLINE_FRACTION)}% information on "
                f"{c['events']:.0f} events."),
        mode="light")
    theme.save(fig, str(IMG / f"a8-f4-posterior.{EXT}"), mode="light")
    figs["posterior"] = charts.Figure(
        str(IMG / f"a8-f4-posterior.{EXT}"),
        alt=("Two density curves over the true hazard ratio. The posterior is "
             "shifted slightly left of the prior and its right tail is cut back "
             "sharply short of 1.0, while the left tails almost coincide."),
        caption=(
            f"Kullback-Leibler divergence {c['bits']:.2f} bits. The prior already "
            f"assigned probability {c['prior_predictive']:.2f} to the boundary "
            f"being crossed, so the surprisal is {c['surprisal_bits']:.2f} bits — "
            f"an upper limit of one bit, and most of it already spent. The lower "
            f"end of the interval barely moves "
            f"({c['prior']['low']:.2f} to {c['posterior']['low']:.2f}), which is "
            f"the whole shape of the result: a significance test rules out the "
            f"null and is nearly silent about how large the effect is."),
        title="One bit")

    # T1 — the calibration table as an image, for Medium and Notion.
    fig_meta, _ = charts.table_image(
        table_rows(res), header=TABLE_HEADER,
        title="Seven reports, read back",
        subtitle=("Recovered event counts use the predicted event split. Two of "
                  "these intervals are not 95% — reading them as 95% is the single "
                  "largest error available here."),
        source="Sources listed in the post. * = event split back-calculated.",
        bold_cols=(3, 4),
        alt=("A seven-row table of adjuvant melanoma trials listing the printed "
             "hazard ratio and interval, the interval's confidence level, the "
             "reported event count, the recovered count, the percentage error and "
             "the bracket the printed rounding permits."),
        caption=(f"The two group-sequential levels are KEYNOTE-054's 98.4% and "
                 f"CheckMate-238's 97.56%. Read KEYNOTE-054's as 95% and the "
                 f"recovered count falls to "
                 f"{trials[0]['recovered']['predicted'] * (1 + trials[0]['if_read_as_95pct']) / (1 + trials[0]['error']['predicted']):.0f}, "
                 f"{_pct(trials[0]['if_read_as_95pct'])} against the reported "
                 f"{trials[0]['reported']}."),
        path=str(IMG / f"a8-t1-trials.{EXT}"))
    figs["table"] = fig_meta

    # HERO — three panels: what was printed, whether the method works, what is left.
    def sheet(panel, m):
        from matplotlib.patches import Rectangle
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        panel.add_patch(Rectangle((1.6, 0.9), 6.8, 8.2, fc=m.surface,
                                  ec=m.ink, lw=2.0))
        for i, w in enumerate((5.4, 4.9, 5.6, 4.4, 5.2)):
            y = 7.9 - 0.72 * i
            panel.plot([2.3, 2.3 + w], [y, y], color=m.grid, lw=2.2,
                       solid_capstyle="round")
        panel.add_patch(Rectangle((2.3, 2.1), 3.4, 1.9, fc="none",
                                  ec=m.series[7], lw=2.4, ls=(0, (3, 2))))
        panel.plot([2.3, 5.7], [2.1, 4.0], color=m.series[7], lw=2.0)
        panel.plot([2.3, 5.7], [4.0, 2.1], color=m.series[7], lw=2.0)

    def caliper(panel, m):
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        panel.plot([1.6, 8.4], [5.4, 5.4], color=m.ink, lw=2.4)
        for x in (1.6, 8.4):
            panel.plot([x, x], [4.3, 6.5], color=m.ink, lw=2.4)
        panel.plot([5.0], [5.4], marker="o", ms=11, color=m.series[0])
        panel.annotate("", xy=(1.6, 7.6), xytext=(8.4, 7.6),
                       arrowprops={"arrowstyle": "<->", "color": m.series[0],
                                   "lw": 2.0})
        panel.plot([2.6, 7.4], [2.6, 2.6], color=m.grid, lw=2.2,
                   solid_capstyle="round")
        panel.plot([3.6, 6.4], [1.5, 1.5], color=m.grid, lw=2.2,
                   solid_capstyle="round")

    def open_range(panel, m):
        from matplotlib.patches import Rectangle
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        panel.add_patch(Rectangle((0.0, 3.4), 6.6, 3.4, fc=m.diverging_mid,
                                  ec="none"))
        panel.plot([6.6, 6.6], [2.6, 7.6], color=m.ink, lw=2.8)
        panel.annotate("", xy=(0.6, 5.1), xytext=(6.4, 5.1),
                       arrowprops={"arrowstyle": "->", "color": m.series[7],
                                   "lw": 2.2})

    s = res["summary"]
    fig_meta, _ = charts.strip_card(
        headline="The hazard ratio in the release is a different trial's",
        panels=[
            (sheet, "0", "phase 3 hazard ratios"),
            (caliper, f"{100 * s['abs_error']['predicted']:.1f}%",
             "error recovering counts"),
            (open_range, f"{bounds['at_band']['high']:.2f}",
             "loosest effect allowed"),
        ],
        note=("The two hazard ratios the announcement does print belong to the "
              "157-patient phase 2b. An interval's width says how many events there "
              "were and nothing else — so the count is recoverable, and a bare "
              "significance claim is a bound set by the one integer nobody prints."),
        footer="The Standard Error", mode="light",
        alt=("A three-panel hand-drawn strip. The first frame shows a press release "
             "with a crossed-out empty box where a number should be, marked zero. "
             "The second shows a confidence interval being measured end to end, "
             f"marked {100 * s['abs_error']['predicted']:.1f} percent. The third "
             "shows a shaded region open to the left with a hard edge on the "
             f"right, marked {bounds['at_band']['high']:.2f}."),
        caption="",
        path=str(IMG / f"a8-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    res = compute(verbose=False)
    figs = figures(res)
    trials, band, bounds, bits = (res["trials"], res["band"], res["bounds"],
                                  res["bits"])
    s = res["summary"]
    k054 = trials[0]
    centre = bits["cases"]["centre"]
    lo_d, hi_d = band["overlap"]
    head_z = bounds["boundary_z"][str(HEADLINE_FRACTION)]
    prop = res["sim"]["proportional hazards"]
    delayed = res["sim"]["delayed effect, 12 months"]
    table_body = md_table(TABLE_HEADER, table_rows(res))
    cuts = bits["by_cut"]
    primary = PHASE2B_CUTS["primary analysis"]
    five = PHASE2B_CUTS["five-year update"]
    WITHHELD_COUNT = {2: "two", 3: "three", 7: "seven"}.get(len(WITHHELD),
                                                            str(len(WITHHELD)))

    # The post's spine, asserted rather than trusted. Each of these is a sentence in
    # the body, and each would be false if the numbers moved.
    if s["abs_error"]["predicted"] > 0.05:
        raise AssertionError(
            f"the recovery averages {100 * s['abs_error']['predicted']:.1f}% error, "
            "which is too large for the post's claim that it works")
    if s["abs_error"]["predicted"] >= s["abs_error"]["allocation"]:
        raise AssertionError(
            "the event-split correction is supposed to beat the allocation "
            "shortcut, and here it does not")
    if s["rounding_width"] <= 2.0 * s["abs_error"]["predicted"]:
        raise AssertionError(
            f"rounding costs {100 * s['rounding_width']:.1f}% and the modelling "
            f"error is {100 * s['abs_error']['predicted']:.1f}% — the post's "
            "'typography beats statistics' claim needs a clear gap")
    if abs(delayed["mean_error"]["predicted"]) > 0.02:
        raise AssertionError(
            "the recovery is supposed to survive non-proportional hazards")
    if not (bounds["at_band"]["high"] - bounds["at_band"]["low"]) > 0.08:
        raise AssertionError(
            "the whole point is that the bound is wide because the event count is "
            "unknown; here it is narrow and the post has no subject")
    if centre["bits"] > 1.0:
        raise AssertionError("a single indicator cannot carry more than one bit")

    post = Post(
        title="Reverse-Engineering the Number Merck Did Not Print",
        slug="reverse-engineering-the-number-merck-did-not-print",
        subtitle=("A phase 3 cancer-vaccine win was announced with no hazard "
                  "ratio, no interval and no p-value. Here is exactly how much of "
                  "it is recoverable, and where the arithmetic stops."),
        summary=(
            f"On {INTERPATH['announced']}, Merck and Moderna said their "
            f"individualised mRNA cancer vaccine plus pembrolizumab beat "
            f"pembrolizumab alone on both recurrence-free and distant "
            f"metastasis-free survival in {INTERPATH['randomised']:,} patients with "
            f"resected melanoma — the first phase 3 win for this class of therapy. "
            f"The release is full of numbers and prints two hazard ratios, both of "
            f"them from a different trial. About the phase 3 it gives the total "
            f"randomised, the allocation ratio and the dose: no hazard ratio, no "
            f"confidence interval, no p-value, no event count. It turns out that "
            f"the width of a "
            f"confidence interval is a statement about the number of events and "
            f"nothing else, so on {s['n_trials']} earlier trials that "
            f"published both, the event "
            f"count can be read back out of the interval to "
            f"**{100 * s['abs_error']['predicted']:.1f}%**. Run backwards, the same "
            f"identity turns \"statistically significant\" into an upper bound on "
            f"the hazard ratio — but the bound depends on the event count, so all "
            f"the announcement pins down is a range from "
            f"**{bounds['at_band']['low']:.2f} to {bounds['at_band']['high']:.2f}**. "
            f"Against a prior built from the phase 2b, the whole announcement is "
            f"worth **{centre['bits']:.2f} bits**, spent almost entirely on ruling "
            f"out no effect."),
        tags=["statistics", "clinical trials", "survival analysis", "disclosure",
              "oncology"],
        data_sources=SOURCES,
        licence_warnings=[
            "Every input is a published aggregate statistic — a press release, an "
            "FDA label, an open-access paper or a registry record. No patient-level "
            "data was used and none is needed; the whole method runs on numbers "
            "that were already printed for the public.",
            "This is a post about the information content of a disclosure. It is "
            "not a judgement on the therapy, the trial, the companies, or any "
            "security, and nothing here is medical or investment advice. A bound "
            "that permits a modest effect is not evidence of a modest effect.",
        ],
        code_url="https://github.com/jonghajeon/standarderror",
        author="Jongha Jeon",
        reproducibility={
            "seed": SEED,
            "simulated trials per row": f"{SIM_REPS} at {SIM_BASE['n_treated']}:"
                                        f"{SIM_BASE['n_control']}",
            "module": "standarderror.uq.survival",
            "tests": "tests/test_survival.py",
            "config hash": res["key"],
            "runtime": f"{res['elapsed_s']}s",
        },
        min_words=1500,
        max_words=2600,
        table_figures=[figs["table"]],
    )

    post.add("The announcement", f"""
{INTERPATH['name']} ({INTERPATH['registry']}) randomised {INTERPATH['randomised']:,}
patients with completely resected stage IIB-IV cutaneous melanoma,
{INTERPATH['ratio']:.0f}:1, to intismeran autogene plus pembrolizumab or pembrolizumab
alone. On {INTERPATH['announced']} the sponsors said a pre-specified interim analysis
had shown statistically significant and clinically meaningful improvements in
{INTERPATH['endpoint']} and in distant metastasis-free survival, with no new safety
signals and overall survival still immature. It is the first time an individualised
neoantigen therapy has won a phase 3, and it is a real event.

Now, the release is not short of numbers — that was this post's first draft and it was
wrong. Three dozen sit on the page: the dose, the construct
({DISCLOSED['product']}), melanoma epidemiology, pages of pembrolizumab safety tables,
and two hazard ratios with intervals — {DISCLOSED['phase2b_rfs']} for recurrence-free
survival, {DISCLOSED['phase2b_dmfs']} for distant metastasis-free survival.

Read the label on those two. They are **KEYNOTE-942**, the 157-patient phase 2b, at
five-year follow-up. Not this trial. The release is explicit and every careful outlet
flagged it, but a hazard ratio sitting three paragraphs under a phase 3 win is the
number a reader carries away.

About the phase 3 the release gives three quantities — total randomised, allocation
ratio, dose — and none of the {WITHHELD_COUNT} that would let you size it:
{", ".join(WITHHELD[:-1])} or {WITHHELD[-1]}. Nor does anything else: the companion
release is the same text, neither company filed an 8-K, the investor materials repeat
the release, and the registry record returns no results and has not been touched since
September 2025.

Which is ordinary practice, not concealment — journals and congresses take a dim view
of numbers that appeared in a press release first, and the earliest realistic venue is
ESMO in Madrid at the end of October, whose late-breaker deadline has not closed yet.
So the question is not why the numbers are missing. It is what is left: more than you
would think, and less than you would like.
""".strip())

    post.add("A confidence interval's width is a headcount", f"""
For a two-arm survival comparison the standard error of the log hazard ratio is
**SE = 1 / sqrt(D f)**, with **f = p_e (1 - p_e)**, where `D` is the total number of
**events** and `p_e` is the fraction of them in the treated arm. Sample size does not
appear: a patient who has not had an event carries almost no information about a hazard
ratio, which is why trials are sized in events.

This is usually introduced as Schoenfeld's asymptotic approximation to the log-rank
test, which undersells it. For exponential survival the exact maximum-likelihood
variance of the log hazard ratio is `1/D1 + 1/D0`, and
**1/D1 + 1/D0 = D / (D1 D0) = 1 / (D p_e (1 - p_e))** — the same number. It is a
counting identity in approximation's clothing, and the tests for this post check it
against that closed form rather than against another copy of itself.

Two consequences follow, and they are the whole post.

**The event count is recoverable.** The *width* of a printed interval contains no
information about the effect — only about how much data there was. So
`D = 4 z^2 / (f (log U - log L)^2)`, and any report that prints an interval has
also printed its event count, in a code.

**A future interval's width is knowable before the estimate is.** `exp(2 z SE)`
needs `D` and nothing else. Whatever hazard ratio {INTERPATH['name']} eventually
reports, the precision of that report is already fixed.

One place to go wrong, and it is not small. `p_e` is the split of *events*, not of
patients, and a treatment that works contributes fewer events than its share of the
randomisation — at {INTERPATH['ratio']:.0f}:1 allocation and a hazard ratio near 0.4
the two differ by a tenth of the implied event count, in a direction set by the effect
size. That last part is what makes it dangerous rather than merely imprecise. The fix
is available whenever a hazard ratio is: predict the split from it,
`p_e = p h / (p h + q)`.
""".strip(), figures=[figs["recovery"]])

    post.add("The control: seven trials that printed both", f"""
Adjuvant melanoma is a good place to test this: the setting has been studied
repeatedly with the same endpoint, and seven of those reports published a hazard
ratio, its interval, the level of that interval **and** the event count. So the
recovery has an answer key.

{table_body}

The mean absolute error is **{100 * s['abs_error']['predicted']:.1f}%**, worst case
{100 * s['max_error_predicted']:.1f}%. Using the observed event split instead of the
predicted one — cheating, since a press release does not carry it — improves it only to
{100 * s['abs_error']['observed']:.1f}%. Skipping the correction and using the
allocation ratio costs {100 * s['abs_error']['allocation']:.1f}%, signed by the design:
negative on four of the five balanced trials and near zero on the fifth, positive on
both {INTERPATH['ratio']:.0f}:1 ones. Dropping the two rows whose event counts I
reconstructed from percentages makes it slightly *worse*,
{100 * s['abs_error_hard_only']['predicted']:.1f}%, so the soft rows are not propping
it up.

**The largest single error available here is misreading the confidence level.** Two of
these are group-sequential designs reporting the interval that matches the alpha they
had left, not 95%: KEYNOTE-054 prints 98.4%, CheckMate-238 prints 97.56%. Read
KEYNOTE-054's as a 95% interval and its {k054['reported']} events come back as
{k054['reported'] * (1 + k054['if_read_as_95pct']):.0f} —
{_pct(k054['if_read_as_95pct'])}, wrong by more than a third, on the strength of a
confidence level stated in a footnote.

The next is stranger, because it is not a statistical error at all. Six of the seven
print the interval to two decimal places, and that rounding alone permits a range of
event counts averaging ±{100 * s['rounding_width']:.1f}% — wider than most of the bars
in Fig 1, which is why the whiskers swallow them. The bracket contains the true count
in {s['rounding_hits']} of {s['n_trials']} cases, missing only KEYNOTE-942, by less
than a single event, on the row I derived from percentages rather than read.
""".strip())

    post.add("What breaks it, and what does not", f"""
Simulation separates the modelling error from the printing error, which published data
cannot: run the recovery on a simulated trial's own Cox output, where the interval is
exact to full precision and the event count is known.

{100 * abs(prop['mean_error']['predicted']):.1f}% under proportional hazards,
{100 * abs(res['sim']['15%/yr dropout']['mean_error']['predicted']):.1f}% with
15%-a-year dropout, and — the part I did not expect —
{100 * abs(delayed['mean_error']['predicted']):.1f}% when the effect is delayed by a
year, which is the failure mode an immunotherapy is most likely to have.

That last one looks like it should break everything and does not, for a reason worth
the caption below: the arithmetic sits downstream of the interpretation problem.

The one simulated case that does degrade the recovery is long follow-up with little
censoring ({_pct(res['sim']['long follow-up, little censoring']['mean_error']['predicted'])}),
where the Cox partial likelihood carries less information than the parametric identity
because the risk sets go lopsided. No report in the calibration set is in that regime.

So the ranking is: which confidence level the interval was printed at, then how many
decimals and whether you corrected the split — a footnote, a typesetting choice, one
line of code — and only then anything statistical, an order of magnitude down. That is
an unusual shape for an error budget. It normally runs the other way.
""".strip(), figures=[figs["errors"]])

    post.add("Applied to a release with no phase 3 numbers in it", f"""
Now run it backwards. Crossing an efficacy boundary at `z` means the observed effect
satisfied `|log HR| >= z SE`, so **HR <= exp(-z / sqrt(D f))** — an upper bound, not an
estimate. The announcement says the effect was at least this large and is silent about
how much larger. Two inputs are needed and neither was disclosed.

**The boundary.** At a first interim under O'Brien-Fleming spending the two-sided 0.05
boundary is exactly `1.96 / sqrt(t)` at information fraction `t` — {head_z:.2f} at
{int(100 * HEADLINE_FRACTION)}%, not 1.96. An interim that clears its boundary has
cleared a higher bar than a final analysis would. Under Pocock spending the same look
asks only {bounds['pocock_z']:.2f}, and that choice alone is worth several hundredths
of a hazard ratio.

**The event count.** Two independent routes bracket it. Powering this setting for a
hazard ratio of {DESIGN_HRS[0]:.2f}-{DESIGN_HRS[-1]:.2f} at 85-90% power needs
{band['final_range'][0]:.0f}-{band['final_range'][1]:.0f} events at the final analysis,
and a first interim sits at {int(100 * min(INFO_FRACTIONS))}-{int(100 * max(INFO_FRACTIONS))}%
of that, so {band['design_range'][0]:.0f}-{band['design_range'][1]:.0f}. The
epidemiology instead — about {band['arms']['control']:.0f} patients on pembrolizumab
alone, an annual recurrence hazard of {CONTROL_HAZARDS[0]:.2f}-{CONTROL_HAZARDS[-1]:.2f}
in this stage mix, mean follow-up of
{MEAN_FOLLOW_UP[0]:.1f}-{MEAN_FOLLOW_UP[-1]:.1f} years for a trial that opened in
mid-2023 — gives {band['epi_range'][0]:.0f}-{band['epi_range'][1]:.0f}. Both routes
allow **{lo_d:.0f} to {hi_d:.0f}**.

Across that band the bound runs from **{bounds['at_band']['low']:.2f} to
{bounds['at_band']['high']:.2f}**. The width is the point: one unpublished integer
moves the strongest available statement by
{bounds['at_band']['high'] - bounds['at_band']['low']:.2f} of hazard ratio, against a
total span of {min(COMPARATORS.values()):.2f} to {max(COMPARATORS.values()):.2f} for
every adjuvant melanoma regimen approved since 2015.

And the direction runs the wrong way round. The bound *rises* with the event count: a
larger interim needs a smaller effect to clear its boundary, so a bigger trial's bare
significance claim constrains the magnitude **less**. The announcement excludes a
hazard ratio above 0.65 — nivolumab's figure in the same setting — only if the interim
had at most {bounds['max_events_for_bound']['0.650']:.0f} events, and above 0.70 only
up to {bounds['max_events_for_bound']['0.700']:.0f}. Both sit inside the plausible
band.

One prediction, so this is falsifiable at the meeting. At {lo_d:.0f}-{hi_d:.0f} events
and {INTERPATH['ratio']:.0f}:1, the 95% interval will span a multiplicative factor of
**{min(bounds['ci_spans'].values()):.2f} to
{max(bounds['ci_spans'].values()):.2f}** whatever the estimate is — a reported
{bounds['example_hr']:.2f} would arrive with an interval near
{bounds['example_interval'][0]:.2f} to {bounds['example_interval'][1]:.2f}. Materially
narrower than that, and the trial had more events than either route allows.
""".strip(), figures=[figs["bound"]])

    post.add("The announcement is worth a fraction of a bit", f"""
There is a cleaner way to say how much was learned. The observation is not an estimate;
it is the single event "the boundary was crossed", whose probability under a true log
hazard ratio is `Phi(-theta/SE - z)`. An ordinary likelihood, so it can update an
ordinary prior — and the natural prior is the phase 2b this trial was built on.

Which requires saying *which* KEYNOTE-942, because there are three published cuts and
they are not the same number:

| cut | median follow-up | RFS hazard ratio | 95% CI |
|---|---|---|---|
| primary analysis | {primary['months']} months | {primary['hr']:.3f} | {primary['lo']:.3f}-{primary['hi']:.3f} |
| three-year update | {PHASE2B_CUTS['three-year update']['months']} months | {PHASE2B_CUTS['three-year update']['hr']:.3f} | {PHASE2B_CUTS['three-year update']['lo']:.3f}-{PHASE2B_CUTS['three-year update']['hi']:.3f} |
| five-year update | {five['months']} months | {five['hr']:.3f} | {five['lo']:.3f}-{five['hi']:.3f} |

The release quotes the five-year row. I use the **primary analysis**, for the reason
this series keeps returning to: the later cuts are conditioned on the trial having gone
on looking good. The primary analysis is the least selected of the three, so it is the
honest prior even though it is the least flattering.

One detail in that top row is worth stopping on: its interval **crosses 1**
({primary['lo']:.3f} to {primary['hi']:.3f}), and the endpoint was met on a
pre-specified one-sided p of {primary['p_one_sided']:.4f} against a one-sided alpha of
{primary['alpha_one_sided']:.2f}. The phase 2b that launched a 1,137-patient phase 3 was
a result whose interval included no effect, declared positive under a deliberately
permissive threshold. Defensible for a signal-finding trial, and a reminder of how much
"met its primary endpoint" varies in strength.

The choice does not drive the answer — each cut as prior gives a posterior median of
{cuts['primary analysis']['posterior']['median']:.2f},
{cuts['three-year update']['posterior']['median']:.2f} and
{cuts['five-year update']['posterior']['median']:.2f}. What changes is how much the
announcement was worth: {cuts['primary analysis']['bits']:.2f} bits against the primary
analysis, {cuts['five-year update']['bits']:.2f} against the five-year cut, because a
prior already giving the boundary {cuts['five-year update']['prior_predictive']:.2f}
probability of being crossed had less left to learn. The more you already believed, the
less the news told you — and here that is the same integral, not a figure of speech.

At the centre of the event band the update moves the median from
{centre['prior']['median']:.2f} to {centre['posterior']['median']:.2f} and the upper
end of the interval from {centre['prior']['high']:.2f} to
{centre['posterior']['high']:.2f}. The lower end moves from
{centre['prior']['low']:.2f} to {centre['posterior']['low']:.2f} — barely at all. That
asymmetry is what every significance test does: rule out no effect, stay nearly silent
about size.

Two ways to price it. How far the belief moved: {centre['bits']:.2f} bits of
divergence. How surprising the news was: the prior gave the boundary being crossed
probability {centre['prior_predictive']:.2f}, so {centre['surprisal_bits']:.2f} bits of
surprisal. A yes/no answer cannot carry more than one bit under any accounting, and
most of that one was spent before the release went out.

Two caveats. Even the primary analysis is selected on having been significant, so it is
an optimistic prior — the winner's curse applies to it exactly as to a backtest, and
correcting would pull the posterior toward 1. And both endpoints were met, not one; but
distant metastasis is a subset of recurrence, so counting them as two observations
overstates the update by more than counting them as one understates it.
""".strip(), figures=[figs["posterior"]])

    post.add("What this changes", f"""
**Print the event count.** One integer, and it is the difference between a reader
bounding the effect within {bounds['at_band']['high'] - bounds['at_band']['low']:.2f}
of hazard ratio and not. Every topline release states the number randomised, which is
the number that does *not* determine the precision.

**Print the interval to three decimal places, and say which level it is.** Two
decimals throw away more than any statistical approximation here costs and more than
all of them together: ±{100 * s['rounding_width']:.1f}% of the event count against
{100 * sum(abs(c['mean_error']['predicted']) for c in res['sim'].values()):.1f}% for
every statistical row in Fig 2 summed. A 98.4% interval read as 95% understates the
data behind it by more than a third, and the level is usually in a footnote. Both are
typesetting decisions with statistical consequences, and nobody making them thinks of
them that way.

**Label a hazard ratio with its trial.** The two here belong to a 157-patient phase 2b
at five-year follow-up. The release says so; a reader skimming does not.

**And treat "statistically significant" as the bound it is** — "at least this large",
with a boundary set by when the look happened and a strength set by an event count.
Reported without that count it is one yes/no answer, worth {centre['bits']:.2f} bits
against a prior that already expected it.
""".strip())

    post.add("Where to be careful", f"""
**Everything about {INTERPATH['name']} here is a bracket around an unknown.** I have
guessed ranges for the event count, the information fraction, the spending function and
the data cutoff, and the bound is only as good as the loosest guess. At the meeting, the
interval-width prediction is the part to check: it does not depend on the effect at all.

**The epidemiological route is the weakest link, and even the arm sizes are
inferred.** It needs an annual recurrence hazard for a stage mix nobody has published
for this trial, and a mean follow-up I inferred from the enrolment period; if
enrolment closed earlier than I assume, the event count is higher and every bound
loosens toward 1. The {band['arms']['treated']:.0f} and {band['arms']['control']:.0f}
come from applying {INTERPATH['ratio']:.0f}:1 to {INTERPATH['randomised']:,}, since no
per-arm number was given — and the registry lists {REGISTRY_ENROLMENT:,} as its
estimated enrolment, so the two public sources disagree by
{100 * (INTERPATH['randomised'] / REGISTRY_ENROLMENT - 1):.0f}% on the least
contentious quantity in the trial.

**No verdict, in either direction.** A bound of {bounds['at_band']['high']:.2f} is not
a claim that the effect *is* {bounds['at_band']['high']:.2f}; the true value could be
anywhere below it, and every phase 2b estimate sits well below it. Nothing here says
whether this therapy works or what it is worth to anyone. I have computed what a
disclosure constrains and stopped.

**Two of my seven answer-key rows are soft, and the set is narrow.** CheckMate-238's
event split comes from a health-technology assessment reading the paper's table rather
than the paper, and KEYNOTE-942's is back-calculated from percentages; without them
the headline error goes from {100 * s['abs_error']['predicted']:.1f}% to
{100 * s['abs_error_hard_only']['predicted']:.1f}%. And seven adjuvant melanoma trials
with one endpoint is a narrow calibration set. The identity is general; that it lands
within a few percent on *these* reports is a statement about these reports, and a
setting with heavier competing risks or crossover is worth checking separately.
""".strip())
    return post


if __name__ == "__main__":
    compute(force=bool(os.environ.get("SERR_FORCE")))
