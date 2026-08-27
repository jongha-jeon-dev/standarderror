"""exp006 — the inspection paradox: one formula behind three everyday complaints.

Backlog: new, Track C/E — a deliberate change of subject from the forecasting and
uncertainty posts, and written light: no code blocks, one formula, everyday
examples first and the mathematics second.

The claim: if you sample intervals (or groups) *by encountering them* rather than
by listing them, you sample them in proportion to their size, and the average you
experience is

    E[size experienced] = E[size] + Var(size) / E[size]

which is the true mean plus a strictly positive penalty whenever sizes vary. The
same identity explains why the bus you catch is on a longer-than-average gap, why
the average student is in a bigger-than-average class, and why the queue you join
is slower than the queue you left. It is not a psychological bias, and the
correction is not a fudge factor — it is exactly Var/mean.

Everything is simulated with a fixed seed, and every claimed number is checked
against the closed form rather than asserted:

- the size-biased mean against E[X] + Var(X)/E[X]
- the expected wait against E[X^2] / (2 E[X]), i.e. (E[X]/2)(1 + CV^2)
- the equal-gap special case, where the penalty is exactly zero, as a control that
  the effect is variance and not something about the simulation

The control matters. A demonstration that produces a large effect on one dataset
and never shows the effect switching off has not identified the cause.

Run: `standarderror run exp006_inspection_paradox --publish`
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import standarderror as se
from standarderror.render import Post
from standarderror.viz import charts, theme

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")
SEED = se.SETTINGS.seed

MEAN_GAP = 10.0          # minutes between buses, on average
N_GAPS = 200_000         # gaps simulated
N_RIDERS = 400_000       # passengers arriving uniformly in time

# Timetables with the same mean gap and different variability.
SCHEDULES = {
    "perfect timetable": 0.0,
    "mildly irregular": 0.5,
    "London bus": 1.0,
    "bunched": 1.6,
}

CLASS_SIZES = np.array([8, 12, 18, 25, 30, 45, 60, 90])
CLASS_COUNTS = np.array([6, 10, 14, 12, 9, 6, 3, 2])


def gaps(cv: float, n: int = N_GAPS, seed: int = 3) -> np.ndarray:
    """Inter-arrival gaps with mean `MEAN_GAP` and coefficient of variation `cv`.

    A gamma distribution, because it hits an exact mean and an exact CV with one
    parameter each and stays positive. cv=0 is the degenerate timetable where
    every gap is identical — the control.
    """
    if cv < 0:
        raise ValueError("cv must be non-negative")
    if cv == 0:
        return np.full(n, MEAN_GAP)
    rng = np.random.default_rng(seed)
    shape = 1.0 / cv ** 2
    return rng.gamma(shape, MEAN_GAP / shape, n)


def experienced(g: np.ndarray, n_riders: int = N_RIDERS,
                seed: int = 11) -> dict:
    """Drop riders uniformly onto the timeline and see which gap each one lands in.

    This is the honest version of the experiment: not "sample gaps in proportion
    to length" (which would assume the answer) but "arrive at a random moment",
    which is what a passenger does. The size bias is then a consequence rather
    than a construction.
    """
    rng = np.random.default_rng(seed)
    edges = np.concatenate([[0.0], np.cumsum(g)])
    t = rng.uniform(0.0, edges[-1], n_riders)
    idx = np.searchsorted(edges, t, side="right") - 1
    my_gap = g[idx]
    wait = edges[idx + 1] - t
    return {"gap_of_rider": my_gap, "wait": wait, "idx": idx}


def theory(g: np.ndarray) -> dict:
    """Closed forms. Nothing here is estimated from the rider simulation."""
    m = float(np.mean(g))
    v = float(np.var(g))
    return {"mean_gap": m, "var_gap": v, "cv": float(np.sqrt(v) / m),
            "size_biased_mean": m + v / m,
            "expected_wait": float(np.mean(g ** 2) / (2.0 * m)),
            "naive_wait": m / 2.0}


def buses() -> dict:
    out = {}
    for name, cv in SCHEDULES.items():
        g = gaps(cv)
        ex = experienced(g)
        th = theory(g)
        out[name] = {
            "cv_target": cv,
            "cv": th["cv"],
            "mean_gap": th["mean_gap"],
            "measured_experienced": float(np.mean(ex["gap_of_rider"])),
            "predicted_experienced": th["size_biased_mean"],
            "measured_wait": float(np.mean(ex["wait"])),
            "predicted_wait": th["expected_wait"],
            "naive_wait": th["naive_wait"],
            "gaps": g,
            "experienced_gaps": ex["gap_of_rider"],
        }
    return out


def classes() -> dict:
    """The same identity in a setting with no time in it at all."""
    sizes = CLASS_SIZES.astype(float)
    counts = CLASS_COUNTS.astype(float)
    n_classes = counts.sum()
    n_students = float(np.sum(sizes * counts))
    mean_class = float(np.sum(sizes * counts) / n_classes)
    var_class = float(np.sum(counts * (sizes - mean_class) ** 2) / n_classes)
    # A student is sampled in proportion to their class's size.
    student_weights = sizes * counts / n_students
    mean_experienced = float(np.sum(sizes * student_weights))
    return {"sizes": sizes, "counts": counts,
            "n_classes": int(n_classes), "n_students": int(n_students),
            "mean_class": mean_class, "var_class": var_class,
            "mean_experienced": mean_experienced,
            "predicted": mean_class + var_class / mean_class,
            "gap_pct": 100.0 * (mean_experienced / mean_class - 1.0)}


def analyse(bus: dict, cls: dict) -> dict:
    ref = bus["London bus"]
    ctrl = bus["perfect timetable"]
    worst = bus["bunched"]
    curve_cv = np.linspace(0.0, 1.8, 61)
    curve = MEAN_GAP / 2.0 * (1.0 + curve_cv ** 2)
    return {
        "bus": bus,
        "cls": cls,
        "ref": ref,
        "ctrl": ctrl,
        "worst": worst,
        "ref_wait_ratio": ref["measured_wait"] / ref["naive_wait"],
        "ref_gap_pct": 100.0 * (ref["measured_experienced"] / ref["mean_gap"] - 1),
        "worst_wait_ratio": worst["measured_wait"] / worst["naive_wait"],
        "ctrl_gap_pct": 100.0 * (ctrl["measured_experienced"] / ctrl["mean_gap"]
                                 - 1),
        # Largest relative error between the simulation and the closed form, over
        # every schedule and both quantities. This is the verification.
        "max_theory_error_pct": 100.0 * max(
            max(abs(v["measured_experienced"] / v["predicted_experienced"] - 1),
                abs(v["measured_wait"] / v["predicted_wait"] - 1))
            for v in bus.values()),
        "curve": (curve_cv, curve),
        "n_riders": N_RIDERS,
        "n_gaps": N_GAPS,
    }


def figures(res: dict) -> dict:
    src = (f"Simulated: {res['n_gaps']:,} gaps with a {MEAN_GAP:.0f}-minute mean, "
           f"{res['n_riders']:,} passengers arriving at uniformly random times.")
    figs = {}
    ref = res["ref"]

    # F1 — the two distributions. The whole idea in one picture.
    hi = float(np.quantile(ref["gaps"], 0.995))
    exp_counts, exp_edges = np.histogram(ref["experienced_gaps"], bins=60,
                                         range=(0.0, hi), density=True)
    exp_centres = 0.5 * (exp_edges[:-1] + exp_edges[1:])
    fig_meta, _ = charts.histogram(
        ref["gaps"][ref["gaps"] <= hi], bins=60,
        series_label="every gap on the timetable",
        overlay={"the gap a passenger lands in": (exp_centres, exp_counts)},
        mark={f"timetabled average: {ref['mean_gap']:.1f} min": ref["mean_gap"]},
        title="The timetable's average gap, and the gap you actually get",
        subtitle=("Bars: every gap between buses. Line: the gap the average "
                  "passenger finds themselves in. Same buses, same day."),
        xlabel="gap between buses (minutes)", source=src, mode="light",
        alt=("Histogram of bus gaps peaking near zero and thinning out to the "
             "right, with a line showing the distribution of gaps as experienced "
             "by passengers shifted noticeably to the right of it, and a marked "
             "line at the ten-minute timetabled average."),
        caption=(f"Fig 1. Long gaps are rare but they are wide, so they collect "
                 f"passengers in proportion to how long they last. Nobody is "
                 f"mistaken and nothing is broken: the average passenger really "
                 f"is in a {ref['measured_experienced']:.1f}-minute gap while the "
                 f"average gap really is {ref['mean_gap']:.1f} minutes."),
        path=str(IMG / f"c3-f1-two-averages.{EXT}"))
    figs["two"] = fig_meta

    # F2 — the formula against the simulation.
    #
    # The four simulated schedules are *points*, not a line: joining four dots
    # across a continuous x axis would imply I had simulated the whole range. They
    # go in via `decorate` with markers, and the flat timetable reference takes
    # muted ink rather than a categorical slot, because it is a reference and not a
    # result. Both still appear in the legend, which `theme.finish` builds from
    # every labelled artist including the ones added here.
    cvs, curve = res["curve"]
    pts_x = [v["cv"] for v in res["bus"].values()]
    pts_y = [v["measured_wait"] for v in res["bus"].values()]
    frame = pd.DataFrame({"the formula": pd.Series(curve, index=cvs)})

    def add_points(_fig, ax):
        m = theme.LIGHT
        ax.axhline(MEAN_GAP / 2.0, color=m.muted, lw=1.4, ls=(0, (5, 3)),
                   label=f"what the timetable implies ({MEAN_GAP / 2:.0f} min)")
        ax.plot(pts_x, pts_y, ls="none", marker="o", ms=9.0,
                color=m.series[1], markeredgecolor=m.surface,
                markeredgewidth=1.6, label="simulated passengers", zorder=5)

    fig_meta, _ = charts.lines(
        frame, mode="light", direct_labels=False, decorate=add_points,
        title="Your wait depends on the timetable's irregularity, not its average",
        subtitle=("Average passenger wait against how uneven the gaps are, at a "
                  "fixed 10-minute mean gap. The formula is half the mean gap "
                  "times (1 + CV²) — nothing is fitted."),
        ylabel="average wait (minutes)",
        xlabel="unevenness of the gaps (coefficient of variation)", source=src,
        alt=("A rising curve of average wait against gap unevenness, from five "
             "minutes at zero to about twenty-one minutes at a coefficient of "
             "variation of 1.8, with four simulated points sitting exactly on it "
             "and a flat dashed line at five minutes showing what the timetable "
             "implies."),
        caption=(f"Fig 2. The four dots are simulations; the curve is the "
                 f"formula, with nothing fitted. At the left the two agree with "
                 f"the timetable — perfectly even gaps really do mean a "
                 f"{MEAN_GAP / 2:.0f}-minute wait. Everything above that dashed "
                 "line is bought with irregularity alone, at an unchanged average "
                 "gap."),
        path=str(IMG / f"c3-f2-formula.{EXT}"))
    figs["formula"] = fig_meta

    # T1 — the summary, as an image, because Medium strips table markup.
    rows = []
    for name, v in res["bus"].items():
        rows.append([name, f"{v['cv']:.2f}", f"{v['mean_gap']:.1f}",
                     f"{v['measured_experienced']:.1f}",
                     f"{v['measured_wait']:.1f}",
                     f"{v['measured_wait'] / v['naive_wait']:.2f}x"])
    fig_meta, _ = charts.table_image(
        rows, header=["timetable", "unevenness", "average gap",
                      "gap you land in", "your wait", "vs. the timetable"],
        title="Four timetables with exactly the same average gap",
        subtitle="Simulated passengers arriving at random. Minutes.",
        source=src, mode="light", bold_cols=(5,),
        alt=("Table of four timetables, all with a ten-minute average gap. The "
             "gap a passenger lands in rises from 10.0 to about 36 minutes and "
             "the wait from 5.0 to about 18 as unevenness rises from 0 to 1.6."),
        caption=("Table 1. The third column is identical down the table by "
                 "construction. The last two are not, which is the entire point: "
                 "the average gap is a fact about the timetable and the wait is a "
                 "fact about its variance."),
        path=str(IMG / f"c3-t1-schedules.{EXT}"))
    figs["table"] = fig_meta

    # HERO — preview card, not part of the body. Bars, because the finding is four
    # cases that differ by a lot, and a bar with its value printed on it survives
    # being shrunk to a feed thumbnail in a way that a line chart's axis does not.
    names = list(res["bus"].keys())
    fig_meta, _ = charts.bar_card(
        headline=f"All four of these timetables run a bus every "
                 f"{MEAN_GAP:.0f} minutes.",
        items=[(n, res["bus"][n]["measured_wait"],
                f"{res['bus'][n]['measured_wait']:.1f} min") for n in names],
        emphasis=names.index("London bus"),
        note=("What a passenger arriving at a random moment actually waits. The "
              "average gap is identical down the list; only its unevenness "
              "changes."),
        footer="The Standard Error", mode="light",
        alt=("Four horizontal bars of average passenger wait for timetables with "
             "the same 10-minute average gap: "
             + ", ".join(f"{n} {res['bus'][n]['measured_wait']:.1f} minutes"
                         for n in names) + "."),
        caption="",
        path=str(IMG / f"c3-hero.{EXT}"))
    figs["hero"] = fig_meta
    return figs


def build() -> Post:
    np.random.seed(SEED)
    IMG.mkdir(parents=True, exist_ok=True)

    bus = buses()
    cls = classes()
    res = analyse(bus, cls)
    figs = figures(res)
    ref, ctrl, worst = res["ref"], res["ctrl"], res["worst"]

    post = Post(
        title="Why Every Bus You Catch Is on a Longer-Than-Average Gap",
        slug="why-every-bus-is-on-a-longer-than-average-gap",
        subtitle=("Your class was bigger than average, your queue was slower, and "
                  "your bus was late. One formula, and no bias in sight"),
        summary=(f"Buses that average a {MEAN_GAP:.0f}-minute gap will, if the "
                 f"gaps are uneven, leave the average passenger sitting in a "
                 f"{ref['measured_experienced']:.0f}-minute gap and waiting "
                 f"{ref['measured_wait']:.0f} minutes instead of "
                 f"{ref['naive_wait']:.0f}. Nobody is misremembering. The same "
                 "arithmetic makes the average student's class bigger than the "
                 "average class, and it has a one-line formula that says exactly "
                 "how much bigger."),
        tags=["statistics", "data-science", "mathematics", "public-transport"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        min_words=1000, max_words=1700,
        table_figures=[figs["table"]],
        data_sources=[
            f"Fully simulated: {N_GAPS:,} bus gaps drawn from gamma "
            f"distributions with a {MEAN_GAP:.0f}-minute mean and varying "
            f"coefficient of variation, and {N_RIDERS:,} passengers arriving at "
            "uniformly random times. The class-size example is an illustrative "
            "distribution, not a real school. No external data; every number is "
            "reproducible from the repo with a fixed seed.",
        ],
        reproducibility={
            "seed": SEED,
            "environment": ", ".join(
                f"{k}={v}" for k, v in se.environment().items()
                if k in ("python", "numpy", "standarderror")),
            "design": f"{N_GAPS:,} gaps per timetable, {N_RIDERS:,} passengers "
                      "placed uniformly on the timeline; the gap each passenger "
                      "lands in is found by search, not by sampling gaps in "
                      "proportion to length",
            "closed forms checked": "E[experienced] = E[X] + Var(X)/E[X] and "
                                    "E[wait] = E[X²]/(2E[X]); largest "
                                    "simulation-vs-formula disagreement across "
                                    "all four timetables and both quantities: "
                                    f"{res['max_theory_error_pct']:.2f}%",
            "control": f"the equal-gap timetable, where the predicted penalty is "
                       f"exactly zero and the measured one is "
                       f"{res['ctrl_gap_pct']:.2f}%",
        },
    )

    post.add("Two true statements that sound like a contradiction", f"""
The buses on a route come every {MEAN_GAP:.0f} minutes on average. The average
passenger on that route is waiting inside a gap of
**{ref['measured_experienced']:.0f} minutes.**

Both of those are exactly true at the same time, on the same route, on the same
day. Nobody is exaggerating and no bus is missing. I simulated
{res['n_gaps']:,} gaps averaging precisely {ref['mean_gap']:.1f} minutes, dropped
{res['n_riders']:,} passengers onto the timeline at random moments, and asked each
one how long the gap they had landed in turned out to be. The answer averaged
{ref['measured_experienced']:.1f} minutes — {res['ref_gap_pct']:.0f}% longer than
the average gap.

The reason is almost annoyingly simple once you see it, and it is the same reason
your class was bigger than the average class and the queue you joined was slower
than the one you left.

**Long gaps are rare, but they are long.** A twenty-minute gap is one gap on the
timetable, exactly like a two-minute gap. But it is collecting passengers for ten
times as long. So when you pick a random *moment* rather than a random *gap*, you
are ten times more likely to be inside it. You are not sampling gaps. You are
sampling minutes, and long gaps own more of the minutes.
""".strip())

    post.add("The size of the penalty is not a matter of opinion", f"""
This has a name — the inspection paradox — and, more usefully, it has a formula.
The average gap *as experienced* is the true average plus the variance of the gaps
divided by that average.

That second term is the whole story. It is zero when every gap is identical and it
grows with unevenness, and it does not care at all what the average is. Two
timetables with the same {MEAN_GAP:.0f}-minute average gap and different
regularity give their passengers completely different days.

I ran four timetables, all with the same average gap, and let the simulated
passengers report back:

| timetable | unevenness | average gap | gap you land in | your wait |
|---|---|---|---|---|
| perfect timetable | {ctrl['cv']:.2f} | {ctrl['mean_gap']:.1f} | {ctrl['measured_experienced']:.1f} | {ctrl['measured_wait']:.1f} |
| mildly irregular | {res['bus']['mildly irregular']['cv']:.2f} | {res['bus']['mildly irregular']['mean_gap']:.1f} | {res['bus']['mildly irregular']['measured_experienced']:.1f} | {res['bus']['mildly irregular']['measured_wait']:.1f} |
| London bus | {ref['cv']:.2f} | {ref['mean_gap']:.1f} | {ref['measured_experienced']:.1f} | {ref['measured_wait']:.1f} |
| bunched | {worst['cv']:.2f} | {worst['mean_gap']:.1f} | {worst['measured_experienced']:.1f} | {worst['measured_wait']:.1f} |

The first row is the one I would point at, because it is the control that tells
you the effect is really about variance. With a perfect timetable the formula
predicts no penalty at all, and the simulation delivers
{res['ctrl_gap_pct']:.2f}% — zero, to within rounding. The effect switches off
exactly when the theory says it should, which is the difference between a
demonstration and a coincidence.

And your wait has its own version of the same formula: half the average gap,
multiplied by one plus the squared unevenness. On the third row that turns a
{ref['naive_wait']:.0f}-minute wait into {ref['measured_wait']:.1f} minutes — a
factor of {res['ref_wait_ratio']:.1f}. On the bunched timetable it is
{res['worst_wait_ratio']:.1f} times what the timetable implies. Across all four
timetables and both quantities, the simulation and the formula disagreed by at most
{res['max_theory_error_pct']:.2f}%.
""".strip(), figures=[figs["two"], figs["formula"]])

    post.add("The same arithmetic, with no buses in it", f"""
Take a school with {cls['n_classes']} classes and {cls['n_students']:,} students.
The average class has {cls['mean_class']:.1f} students in it. Now ask a student how
big their class is: the average answer is **{cls['mean_experienced']:.1f}** —
{cls['gap_pct']:.0f}% bigger.

Nothing has been miscounted. Big classes contain more students, so more students
report from inside them. The school's brochure quotes the first number and every
student's experience is the second, and both are honest. The formula is the same
one, with the same variance term.

Once you know the shape you find it everywhere:

**Queues.** You joined the slow one because the slow one is long, which is why you
could see it. Same reason the lane you switch into slows down: you spend more of
your time in whatever is moving slowly.

**Servers and jobs.** Sample a random *moment* on a machine and you are most
likely to catch it running its longest-running job, which is why "typical job
duration" measured by sampling running processes is systematically wrong.

**Anything measured by intercepting it.** Survey people in a park and you
oversample those who stay a long time. Ask about relationships and you oversample
long ones. Sample lines of code being edited and you oversample the files people
struggle with.

**Lending books, quietly.** A snapshot of loans that are currently outstanding
oversamples long-lived ones, because a six-month loan spends six months in the
snapshot and a five-year loan spends five years. "Average maturity in the book" and
"average maturity of loans written" are different numbers, and taking the first for
the second puts the variance term into the estimate instead of the footnote. The
same applies to any stock-versus-flow question: customers currently subscribed
versus customers ever acquired, tickets currently open versus tickets ever filed.
""".strip())

    post.add("What to do about it", f"""
The fix is never to argue with the number. Both numbers are right; they answer
different questions. It is to be able to say which question you asked.

**Ask what the sampling unit was.** Per gap or per minute. Per class or per
student. Per loan written or per loan outstanding. If the unit is the *encounter*
rather than the *thing*, expect the variance term and go looking for it.

**When you want the underlying average, weight by the inverse of size.** A student
survey estimates the average class size honestly if each student is weighted by
1/(their class size). Same trick for park visitors and for loans in a snapshot.
The estimator is standard, it is not a correction factor invented for the
occasion, and it needs the size to be recorded — which is the practical reason to
record it.

**And when you want the experienced average, say so.** Passengers do not care
about the timetable's average gap. They care about their wait, and their wait is
the number with the variance in it. For a transit operator that reframes the job:
{res['ref_wait_ratio']:.1f}x is what my irregular timetable costs its passengers
at an unchanged average frequency, so **reducing bunching is worth more than adding
buses** until the bunching is gone. The same reframing applies to any queue you
run, and it does not appear in the average.

The general form of the lesson is one I keep meeting from other directions: the
average of a thing and the average experience of that thing are different numbers,
and the gap between them is variance. It shows up in prediction intervals whose
coverage is fine on average and terrible where it matters, and in model searches
whose best result is a fact about the number of attempts. Averages are compressions,
and it is always worth asking what got compressed.

Next time, back to models: what a neural network is doing when it looks like it has
learned physics, and how to tell that apart from having memorised a trajectory.
""".strip())

    return post


if __name__ == "__main__":
    p = build()
    print(p.title, "|", p.word_count(), "words |", len(p.figures), "figures")
    for issue in p.audit():
        print("  audit:", issue)
