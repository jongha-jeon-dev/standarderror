"""Coverage 5: The Split You Chose Is Hiding How Variable Your Coverage Is.

The fifth episode of the uncertainty series, and it starts by correcting the
first. Episode 1 promised that the error bar on conformal coverage was wider
than exchangeability predicts "by a factor I did not expect". The factor was
1.49, and most of it was my formula: the Beta spread describes the coverage a
calibration set delivers, and the realised coverage is then *measured* on a
finite test half, which adds a binomial term of the same size.

Measured:

* Row-wise random splits: SD 0.0040 over 200 splits. Against the Beta term
  alone, 1.49x. Against both terms, 1.06x -- inside the 0.88 to 1.10 that
  twenty i.i.d. pools of the same size produce. A random row split of a fixed
  pool is exchangeable by construction, so it cannot show dependence.
* Whole-sequence splits: SD 0.0055, 1.42x the exchangeable spread, with the
  mean still 0.8999. This is the split a deployment makes -- calibrate on
  some documents, serve others.
* The within-sequence correlation of the coverage indicator predicts it: a
  design effect of 2.06, so sqrt 1.43 against 1.42 measured. For blocks cut
  out of a sequence the prediction is an upper bound, because neighbouring
  blocks land on both sides of the split and carry the correlation across.

Run: `standarderror run cv105_split --publish`
"""

from __future__ import annotations

import math
import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.llm import tiny
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.uncertainty import coverage as cv
from standarderror.viz import charts

POST_DATE = date(2026, 10, 8)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Uncertainty for Language Models, Taught Through What Breaks"
SERIES_TAG = "Coverage"

BATCHES, BATCH_SIZE = 24, 16
ALPHA, SPLITS = 0.1, 200
BLOCKS = (1, 2, 4, 8, 16, 32, 64)


def compute() -> dict:
    pred = cv.predictions(count=BATCHES, size=BATCH_SIZE, seed=1)
    row = cv.split_variability(pred, alpha=ALPHA, draws=SPLITS)
    seq = cv.grouped_split(pred, alpha=ALPHA, draws=SPLITS)
    rho = cv.lag_correlation(pred, alpha=ALPHA)
    blocks = {m: cv.block_split(pred, m, alpha=ALPHA, draws=SPLITS)
              for m in BLOCKS}
    predicted = {m: math.sqrt(cv.design_effect(rho, m)) for m in BLOCKS}
    control = cv.iid_control(pred, alpha=ALPHA, pools=20, draws=SPLITS)
    # An alarm set at two exchangeable standard deviations below the target.
    floor = 1 - ALPHA - 2 * row["exchangeable_sd"]
    alarm = {"floor": floor,
             "row": float((row["coverages"] < floor).mean()),
             "seq": float((seq["coverages"] < floor).mean()),
             "row_n": int((row["coverages"] < floor).sum()),
             "seq_n": int((seq["coverages"] < floor).sum())}
    return {"pred": pred, "row": row, "seq": seq, "rho": rho,
            "blocks": blocks, "predicted": predicted, "control": control,
            "alarm": alarm,
            "length": int(np.bincount(pred["sequence"]).max())}


# ------------------------------------------------------------------ figures

def figures(res: dict) -> dict:
    out: dict = {}
    row, seq = res["row"], res["seq"]

    def two_terms(ax, m):
        c = row["coverages"]
        ax.hist(c, bins=24, density=True, color=m.series[0], alpha=0.35,
                label=f"{row['draws']} random row splits")
        xs = np.linspace(c.min() - 0.004, c.max() + 0.004, 300)
        for sd, colour, ls, lab in (
                (row["beta_sd"], m.series[1], "--",
                 f"calibration term only, SD {row['beta_sd']:.4f}"),
                (row["exchangeable_sd"], m.series[2], "-",
                 f"calibration + test, SD {row['exchangeable_sd']:.4f}")):
            ax.plot(xs, np.exp(-0.5 * ((xs - row["mean"]) / sd) ** 2)
                    / (sd * math.sqrt(2 * math.pi)), lw=2.4, ls=ls,
                    color=colour, label=lab)
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    out["f0"] = charts.diagram(
        two_terms,
        title="The spread I compared against was missing half its variance",
        subtitle=(f"Realised coverage at alpha = {ALPHA} on "
                  f"{len(res['pred']['y']):,} predictions, split in half "
                  f"{row['draws']} times. Measured SD {row['sd']:.4f}."),
        xlabel="realised test coverage",
        ylabel="density",
        source="Measured; standarderror/uncertainty/coverage.py.",
        alt=("A histogram of realised coverage with a narrow dashed curve "
             "that is too tight for it and a wider solid curve that fits."),
        caption=(f"Against the calibration term alone the measured spread is "
                 f"{row['sd_ratio']:.2f}x; against both terms it is "
                 f"**{row['excess']:.2f}x**. Episode 1's 'factor I did not "
                 f"expect' was mostly the term I left out."),
        path=str(IMG / f"cv105-f0-terms.{EXT}"))[0]

    ms = list(BLOCKS)
    ctl = res["control"]

    def blocks(ax, m):
        ax.axhspan(ctl["low"], ctl["high"], color=m.grid, alpha=0.5,
                   label=(f"20 i.i.d. pools, row split: {ctl['low']:.2f} "
                          f"to {ctl['high']:.2f}"))
        ax.plot(ms, [res["predicted"][k] for k in ms], lw=2.0, ls="--",
                color=m.series[1],
                label="predicted from the correlations")
        ax.plot(ms, [res["blocks"][k]["excess"] for k in ms], marker="o",
                ms=6, lw=2.4, color=m.series[0], label="measured")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ms)
        ax.set_xticklabels([str(k) for k in ms])
        ax.axhline(1.0, lw=1, color=m.ink_secondary)
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    out["f1"] = charts.diagram(
        blocks,
        title="Dependence shows only when the split respects it",
        subtitle=("Spread of realised coverage over the exchangeable "
                  "spread, splitting blocks of consecutive rows. 64 rows "
                  "is a whole sequence."),
        xlabel="rows per block sent to one side of the split",
        ylabel="excess spread",
        source="Measured; standarderror/uncertainty/coverage.py.",
        alt=("A measured line near 1 for blocks of one row rising to 1.42 at "
             "whole sequences, a dashed prediction meeting it at 64, and a "
             "shaded band around 1."),
        caption=(f"A row split sits inside the i.i.d. band. Whole sequences "
                 f"give **{res['blocks'][64]['excess']:.2f}x**, and the "
                 f"correlations predicted {res['predicted'][64]:.2f}x. For "
                 f"blocks of 8 to 32 rows the prediction runs high: blocks "
                 f"cut from one sequence leave correlated neighbours on both "
                 f"sides."),
        path=str(IMG / f"cv105-f1-blocks.{EXT}"))[0]

    al = res["alarm"]
    b8, b32 = res["blocks"][8], res["blocks"][32]
    rows_t = [
        ["i.i.d. pools (mean of 20)", "-", "-", f"{ctl['mean']:.2f}", "-"],
        ["rows", f"{row['sd']:.4f}", f"{row['sd_ratio']:.2f}",
         f"{row['excess']:.2f}", f"{al['row']:.1%}"],
        ["blocks of 8 rows", f"{b8['sd']:.4f}", f"{b8['sd_ratio']:.2f}",
         f"{b8['excess']:.2f}", "-"],
        ["blocks of 32 rows", f"{b32['sd']:.4f}", f"{b32['sd_ratio']:.2f}",
         f"{b32['excess']:.2f}", "-"],
        ["whole sequences", f"{seq['sd']:.4f}", f"{seq['sd_ratio']:.2f}",
         f"{seq['excess']:.2f}", f"{al['seq']:.1%}"],
    ]
    out["f2"] = charts.table_image(
        rows_t,
        header=["split by", "SD of coverage", "over Beta term",
                "over exchangeable", "alarm rate"],
        title="The same data, five ways to split it",
        subtitle=(f"{row['draws']} splits each. The alarm fires when "
                  f"coverage falls below {al['floor']:.4f}, two exchangeable "
                  f"SDs under the target: 2.3% of the time if the rows were "
                  f"exchangeable."),
        source="Measured; standarderror/uncertainty/coverage.py.",
        alt=("A five-row table: excess near 1 for i.i.d. pools and row "
             "splits, rising to 1.42 for whole sequences, with the alarm "
             "rate rising with it."),
        caption=("The middle column is the comparison episode 1 made. The "
                 "**fourth** is the one that tests exchangeability."),
        bold_cells={(4, c) for c in range(5)}, align="lrrrr",
        path=str(IMG / f"cv105-f2-table.{EXT}"))[0]

    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    row, seq = res["row"], res["seq"]
    xs = np.linspace(-4, 4, 80)

    def widths(panel, m):
        panel.plot(xs, np.exp(-0.5 * (xs / 0.7) ** 2), lw=2.4,
                   color=m.series[1])
        panel.plot(xs, np.exp(-0.5 * (xs / 1.0) ** 2) * 0.7, lw=2.8,
                   color=m.series[2])

    def flat(panel, m):
        panel.plot(range(7), [1.0, 1.04, 0.97, 1.02, 0.99, 1.03, 1.0],
                   lw=2.8, color=m.series[0])
        panel.set_ylim(0.0, 2.0)

    def rise(panel, m):
        panel.plot(range(len(BLOCKS)),
                   [res["blocks"][k]["excess"] for k in BLOCKS], lw=2.8,
                   color=m.series[0])
        panel.plot(range(len(BLOCKS)),
                   [res["predicted"][k] for k in BLOCKS], lw=2.2,
                   color=m.series[1])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=5,
        headline="The excess was my formula, until the split was by sequence",
        panels=[(widths, f"{row['sd_ratio']:.2f}x", "the factor I promised"),
                (flat, f"{row['excess']:.2f}x", "with the term I left out"),
                (rise, f"{seq['excess']:.2f}x", "split by sequence")],
        note=(f"Realised conformal coverage is measured on a finite test "
              f"set, so its spread has two terms, not one. With both, a row "
              f"split shows no excess at all. Split whole sequences, the way "
              f"a deployment does, and the spread is "
              f"{seq['excess'] - 1:.0%} wider than exchangeability "
              f"predicts."),
        alt=("Three hand-drawn frames: a narrow and a wide bell curve; a "
             "flat wavering line; and a rising line with a second line "
             "meeting it at the end."),
        path=str(IMG / f"cv105-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["row"] = s.run("""
        from standarderror.uncertainty import coverage as cv

        pred = cv.predictions(count=24, size=16, seed=1)
        row = cv.split_variability(pred, alpha=0.1, draws=200)
        print(f"200 row splits: mean {row['mean']:.4f}  SD {row['sd']:.4f}")
        print(f"Beta SD for n_cal = 12,288: {row['beta_sd']:.4f}"
              f"   ratio {row['sd_ratio']:.2f}")
        """)
    out["terms"] = s.run("""
        # The threshold's coverage is Beta; the test half then estimates it.
        full = cv.exchangeable_sd(12288, 12288, alpha=0.1)
        print(f"calibration + test SD {full:.4f}   excess {row['sd'] / full:.2f}")

        control = cv.iid_control(pred, alpha=0.1, pools=20, draws=200)
        print(f"i.i.d. pools, same procedure: {control['low']:.2f} to "
              f"{control['high']:.2f}, mean {control['mean']:.2f}")
        """)
    out["seq"] = s.run("""
        seq = cv.grouped_split(pred, alpha=0.1, draws=200)
        print(f"200 sequence splits: mean {seq['mean']:.4f}  SD {seq['sd']:.4f}"
              f"   excess {seq['excess']:.2f}")
        """)
    out["rho"] = s.run("""
        rho = cv.lag_correlation(pred, alpha=0.1)
        print("correlation at lags 1-6:", " ".join(f"{r:.3f}" for r in rho[:6]))
        deff = cv.design_effect(rho, 64)
        print(f"design effect {deff:.2f}   predicted excess {deff ** 0.5:.2f}"
              f"   rows worth {len(pred['y']) / deff:,.0f} independent ones")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    row, seq, ctl = res["row"], res["seq"], res["control"]
    rho, pr, bl, al = res["rho"], res["predicted"], res["blocks"], res["alarm"]
    n = len(res["pred"]["y"])
    deff = cv.design_effect(rho, res["length"])
    n_seq = seq["sequences"]

    # The spine, asserted rather than trusted.
    assert 1.4 < row["sd_ratio"] < 1.6                 # episode 1's number
    assert row["exchangeable_sd"] / row["beta_sd"] > 1.41
    assert ctl["low"] < row["excess"] < ctl["high"]    # no row-wise excess
    assert seq["excess"] > ctl["high"] + 0.25          # a real one
    assert abs(seq["mean"] - (1 - ALPHA)) < 0.002      # marginal survives
    assert abs(pr[64] - seq["excess"]) < 0.05          # predicted, not fitted
    assert all(pr[m] > bl[m]["excess"] for m in (8, 16, 32))
    assert al["seq"] > 2 * al["row"]

    post = Post(
        title=(f"{SERIES_TAG} 5: The Split You Chose Is Hiding How Variable "
               f"Your Coverage Is"),
        slug="coverage-5-split",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=5,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Episode 1 promised an error bar wider than "
                  "exchangeability allows. Most of that was my formula. "
                  "What is left appears only when you split the way a "
                  "deployment does."),
        summary=(
            f"Over {row['draws']} random row splits of {n:,} predictions, "
            f"realised conformal coverage at alpha = {ALPHA} has an SD of "
            f"{row['sd']:.4f}, {row['sd_ratio']:.2f} times the Beta spread "
            f"episode 1 compared it with. That comparison was short a term: "
            f"the coverage is measured on a finite test half, and with the "
            f"test half's binomial variance added the excess is "
            f"{row['excess']:.2f}, inside the {ctl['low']:.2f} to "
            f"{ctl['high']:.2f} that i.i.d. pools of the same size produce. "
            f"It has to be - a random row split of a fixed pool is "
            f"exchangeable by construction. Splitting whole sequences "
            f"instead, as a deployment does, gives {seq['excess']:.2f}x with "
            f"the mean still {seq['mean']:.4f}, and the within-sequence "
            f"correlation of the coverage indicator predicts "
            f"{pr[64]:.2f}x before the split is run."),
        tags=["machine-learning", "data-science", "statistics",
              "conformal-prediction", "uncertainty", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "No external data. Every number is computed from the committed "
            "model's predictions on held-out text; no values from the text "
            "are published.",
            f"The model: an {tiny.load()['parameters']:,}-parameter "
            f"character-level transformer - four blocks, four heads, width "
            f"{tiny.WIDTH}, context {tiny.BLOCK}, validation loss "
            f"{tiny.VAL_LOSS} against a uniform-guess "
            f"{tiny.UNIFORM_LOSS:.3f}. Weights, training script and a "
            f"verified sha256 are committed: `standarderror/llm/tiny.py`, "
            f"`scripts/train_tiny.py`, `data/tiny_gpt/`.",
            "Machinery: `standarderror/uncertainty/coverage.py`, tested in "
            "`tests/test_uncertainty.py`. The test that used to assert the "
            "1.5x as a violation now asserts the opposite and keeps the old "
            "comparison alongside, so the mistake stays checkable.",
            "Where this stops: Vovk, Gammerman and Shafer, *Algorithmic "
            "Learning in a Random World* (2005); Angelopoulos and Bates, \"A "
            "gentle introduction to conformal prediction and "
            "distribution-free uncertainty quantification\" (2021), whose "
            "section on checking coverage gives the beta-binomial "
            "distribution of empirical coverage - both terms, which is where "
            "I should have looked; Kish, *Survey Sampling* (1965), for the "
            "design effect.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "model": ("the committed checkpoint, hash-verified on load, so "
                      "every number is measured on the same weights"),
            "determinism": ("row splits use seeds 0 to 199, block splits "
                            "10,000 to 10,199, and the i.i.d. pools a fixed "
                            "generator; the sequence-split numbers are "
                            "identical to the ones episode 1 measured"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "The number episode 1 left open",
        f"""Episode 1 ended on a promise. Split conformal's coverage was exactly what it guaranteed on average, but the error bar on that average was wider than exchangeability allows, "by a factor I did not expect and in a direction that gets worse when you do the obvious thing about it."

Here is the measurement behind that sentence.

{snip['row'].markdown()}

Two hundred random splits of {n:,} predictions into calibration and test halves, and the realised coverage moves with a standard deviation of {row['sd']:.4f}. The spread I compared it with, {row['beta_sd']:.4f}, is the standard deviation of a Beta distribution: given a calibration set of n points, the coverage that its threshold delivers on fresh data is Beta-distributed, and its variance is alpha (1 - alpha) / (n + 2). The ratio is {row['sd_ratio']:.2f}. I read that as the 384 sequences' overlapping contexts making the rows less exchangeable than the split pretends.

That reading is wrong, and the rest of this episode is about why, and about what is left once it is corrected.""")

    post.add(
        "The formula was short a term",
        f"""The Beta distribution describes the coverage a threshold *delivers* — the probability, over fresh test points, that one lands inside its set. Nobody observes that probability. What the split measures is the fraction of {n - n // 2:,} test rows that landed inside, and that is a finite sample of it, with its own binomial variance alpha (1 - alpha) / n_test.

Two finite samples, two terms. With equal halves they are the same size, so leaving one out understates the spread by exactly the square root of two.

{snip['terms'].markdown()}

With both terms the exchangeable spread is {row['exchangeable_sd']:.4f}, and the measured spread is **{row['excess']:.2f} times** it. To know whether {row['excess']:.2f} is anything, the same procedure was run on twenty pools that are i.i.d. by construction: the model's own scores, resampled with replacement to the same size. They give {ctl['low']:.2f} to {ctl['high']:.2f}. The row split sits inside that range. There is no excess to explain.

There could not have been. A random permutation of a fixed pool makes the calibration and test halves exchangeable *whatever* dependence the rows carry, because every assignment of rows to halves is equally likely. The rows inside one sequence are correlated, but a row split scatters each sequence across both halves, and the correlation is averaged away before it can move the threshold. A split that guarantees exchangeability is a test of exchangeability that cannot fail: a good property for a guarantee and a useless one for a test.""",
        figures=[figs["f0"]])

    post.add(
        "Split the way a deployment does",
        f"""The question episode 1 was reaching for is a different one. A model is calibrated on some documents and then serves other documents. The split that mirrors that sends whole sequences to one side.

{snip['seq'].markdown()}

The guarantee still holds: the mean over {seq['draws']} sequence splits is {seq['mean']:.4f}. Sequences are exchangeable with each other, and that is all split conformal needs. What changes is the spread around it: {seq['sd']:.4f}, **{seq['excess']:.2f} times** the exchangeable spread, well outside anything the i.i.d. pools produced.

So the second half of episode 1's sentence survives. The obvious fix — split by sequence, so calibration and test share no context — does make the error bar wider, and that wider bar is the honest one. The row split is not wrong about the average. It is wrong about how far any one deployment can land from it, because it measures a situation, rows of the same document on both sides of the line, that a deployment never meets.""")

    post.add(
        "The correlations predict it",
        f"""An excess with no mechanism is a number. This one has a mechanism that can be measured separately and turned into a prediction before the split is run.

Mark each row by whether it would be covered at the pooled threshold, and measure how correlated that mark is between rows a given distance apart inside one sequence.

{snip['rho'].markdown()}

The correlations are small — {rho[0]:.3f} between neighbours, falling to {rho[5]:.3f} six characters apart — but a sequence has {res['length']} rows and every pair contributes. The design effect from survey sampling adds them up: the variance of a block mean, over what it would be if the rows were independent. Its square root is the predicted excess, **{pr[64]:.2f}**, against {seq['excess']:.2f} measured. For the purpose of coverage spread, {n:,} rows are worth about {n / deff:,.0f} independent ones.

The same prediction for blocks of 8 to 32 rows runs high, and the reason is useful. (Below 8 the two are within the noise of each other.) A block of 8 rows cut from a sequence of {res['length']} leaves its neighbours, which it is correlated with, free to land on the other side of the split. The correlation then ties calibration to test and *narrows* the spread — the row split's averaging, in a weaker form. At 32 rows the prediction is {pr[32]:.2f} and the measurement {bl[32]['excess']:.2f}; only when the block is the whole sequence is nothing left to leak.""",
        figures=[figs["f1"]])

    post.add(
        "What the wider bar costs",
        f"""At {n:,} predictions the difference looks small in coverage units: an SD of {seq['sd']:.4f} instead of {row['sd']:.4f}. It is not small in what it does to anything that uses the spread.

Suppose you monitor a deployed conformal predictor and raise an alarm when realised coverage falls more than two exchangeable standard deviations below target, below {al['floor']:.4f}. If the rows were exchangeable that alarm would fire about 2.3% of the time with nothing wrong. Over the row splits it fired {al['row_n']} times in {row['draws']}, {al['row']:.1%}, which is as close to 2.3% as {row['draws']} tries can resolve. Over the sequence splits, with the method working exactly as promised, it fired {al['seq_n']} times: **{al['seq']:.1%}**.

That is the practical form of the mistake. The row split is the easy simulation to run, it agrees with the textbook once the textbook is read properly, and a threshold validated on it is crossed by chance about {al['seq'] / 0.0228:.0f} times as often as intended once calibration and deployment are separate documents.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        f"""**One model, one corpus.** The {seq['excess']:.2f}x belongs to character-level predictions in {res['length']}-character windows of one text. Longer documents, or predictions whose difficulty clusters more strongly within a document, will give more; the design effect is the tool for saying how much before measuring it.

**Sequence is the unit I had.** The {n_seq} windows are drawn from one held-out stretch of text, so they are not independent documents either; a split by act or by speaker might show more. The correlation at long lags is small but positive ({sum(rho[8:]) / len(rho[8:]):.4f} on average beyond lag 8), which is the signature of difficulty shared across the whole window, and perhaps beyond it.

**The spread estimates have noise of their own.** A standard deviation from 200 splits is uncertain by about 5%. The i.i.d. range is the honest measure of that, and the conclusions here sit well outside it in one case and well inside it in the other.

**Nothing here touches the guarantee.** Marginal coverage held in every split: {row['mean']:.4f} for rows, {seq['mean']:.4f} for sequences. This episode is about the error bar, not the promise.""")

    post.add(
        "What to keep",
        f"""1. Realised coverage is measured on a finite test set, so its spread has two terms: the Beta spread of what the calibration set delivers, and the binomial spread of estimating it. With equal halves, leaving out the second understates the spread by the square root of two.

2. With both terms, a random row split shows no excess: {row['excess']:.2f}x, inside the {ctl['low']:.2f} to {ctl['high']:.2f} that i.i.d. pools give. Episode 1's {row['sd_ratio']:.2f}x was my formula.

3. A random row split cannot detect dependence. It is exchangeable by construction.

4. Splitting whole sequences — the split a deployment makes — gives {seq['excess']:.2f}x, with marginal coverage intact at {seq['mean']:.4f}.

5. The within-sequence correlation of the coverage indicator predicts it: a design effect of {deff:.2f}, so {pr[64]:.2f}x, before the split is run. {n:,} rows are worth about {n / deff:,.0f}.

6. A two-SD alarm that should fire 2.3% of the time fires {al['seq']:.1%} of the time under sequence splits.""")

    post.add(
        "Exercise",
        """Take whatever you used to validate a conformal predictor and find the unit a deployment would split on: the document, the patient, the customer, the day. Rerun the split variability with that unit held together. If the spread does not move, your rows were as good as independent and you have lost nothing. If it does, the larger number is the one to set thresholds with.

Then compute the correlation of the coverage indicator inside that unit and the design effect it implies. If it predicts the spread you measured, you have a way to size the next validation set without running it. If it does not, something other than within-unit correlation is moving your coverage, and that is worth knowing before a monitor tells you.

The uncomfortable version: if you reported an error bar on coverage from row-wise resampling of a dataset with any grouping in it, check whether the grouping was there. It usually is.""")

    post.add(
        "Next",
        """That closes the series. Each episode took a statement that is exactly true and found the gap one question further in: coverage averaged over the wrong population, an estimator biased by its bins, a fix that reorders what you refuse, an overconfidence that arrives before training stops improving, and an error bar that depends on how you split. The last of those gaps was partly mine, which is the reason to measure before writing — and to keep measuring after.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
