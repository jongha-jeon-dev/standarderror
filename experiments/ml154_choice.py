"""ML 54: Multiple Choice Grades the Wrong Answers You Offered.

The fifty-fourth episode of *Machine Learning, Taught Through What Breaks*,
in Arc XI. "Scoring a model by which answer option it finds likeliest
measures the same thing as asking it for the answer." Measured on three of
the repository's own arithmetic transformers, on the same problems in the
same contexts:

* On four-digit sums, which none of the three models could produce once in
  200 tries, picking the right answer from four by likelihood scored 0.955,
  0.98 and 0.79 against random distractors of the same length.
* The same model on the same sums scored 0.955, 0.395 or 0.315 depending on
  whether the three wrong options were random numbers, one-digit slips, or
  the answer with a digit added or removed.
* The main model's clear lead over its leaky twin in generating
  multiplications vanished under near-miss multiple choice;
  averaging the likelihood per character changed almost nothing, because the
  options were mostly the same length.

Run: `standarderror run ml154_choice --publish`
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

import standarderror as se
from standarderror.ml import llmeval as le
from standarderror.render import Post
from standarderror.render.snippet import Session
from standarderror.schoolmath import model as sm
from standarderror.viz import charts

POST_DATE = date(2026, 10, 14)

IMG = se.SETTINGS.build_dir / "img"
EXT = os.environ.get("SERR_FIG_EXT", "png")

SERIES = "Machine Learning, Taught Through What Breaks"
SERIES_TAG = "ML"

NAMES = {"clean": "main model", "leaky": "leaky twin",
         "s1_6000": "short run"}
CASES = {"test/mul": "multiply", "test/order": "order of operations",
         "probe_carry/add": "3-digit carries", "probe_length/add":
         "4-digit sums"}
KINDS = {"gen": "generate", "random": "MC, random",
         "near": "MC, near misses", "length": "MC, lengths"}


def _models() -> dict:
    return {"clean": sm.load(), "leaky": sm.load(adjacent=True),
            "s1_6000": sm.load_member("s1_6000")}


def compute() -> dict:
    models = _models()
    sums = le.items("probe_length", "add", 200)
    rows = le.ar.graded(models["clean"], sums,
                        filler=le.ar.fillers(len(sums), seed=9))
    want, got = [r["want"] for r in rows], [r["got"] for r in rows]
    return {"scores": le.choice_scores(models),
            "shorter": float(np.mean([len(g) < len(w)
                                      for w, g in zip(want, got)])),
            "first1": float(np.mean([g[:1] == w[:1] for w, g in zip(want, got)])),
            "first2": float(np.mean([g[:2] == w[:2] for w, g in zip(want, got)])),
            "prefix": float(np.mean([bool(g) and w.startswith(g)
                                     for w, g in zip(want, got)]))}


# ------------------------------------------------------------------ figures

def _bars(ax, m, sc):
    cases = list(CASES)
    x = np.arange(len(cases))
    w = 0.2
    for k, (kind, label) in enumerate(KINDS.items()):
        v = [sc[f"{c}/clean"][kind] for c in cases]
        ax.bar(x + (k - 1.5) * w, v, w, color=m.series[k], label=label)
    ax.axhline(0.25, color=m.ink, ls=":", lw=1)
    ax.text(len(cases) - 0.55, 0.27, "chance, 4 options", fontsize=8,
            ha="right", color=m.ink)
    ax.set_xticks(x)
    ax.set_xticklabels([CASES[c] for c in cases], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right", ncol=2, frameon=False)


def figures(res: dict) -> dict:
    sc = res["scores"]
    out: dict = {}
    pl = sc["probe_length/add/clean"]
    out["f0"] = charts.diagram(
        lambda ax, m: _bars(ax, m, sc),
        title="One model, one set of problems, four scores",
        subtitle=("The main arithmetic model on 200 problems per task. "
                  "'Generate' is exact match of its greedy answer; the "
                  "multiple-choice (MC) scores pick the likeliest of the "
                  "right answer and three wrong ones of the named kind."),
        xlabel="", ylabel="accuracy",
        figsize=(7.6, 4.4),
        source="standarderror/ml/llmeval.py; models in data/schoolmath.",
        alt=("Grouped bars for four tasks. On four-digit sums generation is "
             "zero, multiple choice against random numbers is near one, "
             "against near misses about 0.4, against different lengths about "
             "0.3."),
        caption=(f"Four-digit sums: generated correctly **{pl['gen']:.2f}**, "
                 f"picked from random options {pl['random']:.3f}, from "
                 f"near misses {pl['near']:.3f}, from different lengths "
                 f"{pl['length']:.3f}."),
        path=str(IMG / f"ml154-f0-scores.{EXT}"))[0]

    body = []
    for c, cl in CASES.items():
        for name, nl in NAMES.items():
            r = sc[f"{c}/{name}"]
            body.append([cl, nl, f"{r['gen']:.3f}", f"{r['random']:.3f}",
                         f"{r['near']:.3f}", f"{r['length']:.3f}",
                         f"{r['length_avg']:.3f}"])
    out["f1"] = charts.table_image(
        body,
        header=["task", "model", "generate", "MC random", "MC near",
                "MC lengths", "lengths, per char"],
        title="Three models, five scoring rules",
        subtitle=("Accuracy on the same 200 problems per task. The last "
                  "column ranks the length-varied options by likelihood per "
                  "character instead of in total."),
        source="standarderror/ml/llmeval.py.",
        alt=("A twelve-row table. Generation ranges from 0 to 0.755; "
             "multiple choice against random numbers is 0.79 or higher "
             "everywhere."),
        caption=("Each block of three rows is one task; MC = multiple choice "
                 "among four options."),
        align="llrrrrr",
        path=str(IMG / f"ml154-f1-table.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    sc = res["scores"]
    pl = sc["probe_length/add/clean"]

    def p1(panel, m):
        panel.bar([0, 1], [pl["gen"], pl["random"]],
                  color=[m.series[0], m.series[1]])
        panel.text(0, 0.02, "0", ha="center", fontsize=9)
        panel.set_ylim(0, 1.05)

    def p2(panel, m):
        panel.bar([0, 1, 2], [pl["random"], pl["near"], pl["length"]],
                  color=[m.series[1], m.series[2], m.series[3]])
        panel.set_ylim(0, 1.05)

    def p3(panel, m):
        a, b = sc["test/mul/clean"], sc["test/mul/leaky"]
        panel.bar([0, 1], [a["gen"] - b["gen"], a["near"] - b["near"]],
                  color=[m.series[0], m.series[2]])
        panel.axhline(0, color=m.ink, lw=1)

    return charts.lecture_hero(
        series=SERIES_TAG, episode=54,
        headline="Picked, not produced",
        panels=[(p1, f"{pl['random']:.2f}", "picked a sum it can't do"),
                (p2, f"{pl['length']:.2f}", "same sums, other options"),
                (p3, f"{sc['test/mul/clean']['gen'] - sc['test/mul/leaky']['gen']:.3f}",
                 "lead that MC erased")],
        note=("A multiple-choice score is the model's accuracy against the "
              "particular wrong answers someone wrote. Change them and the "
              "score, and sometimes the ranking, changes with them."),
        alt=("Three hand-drawn frames: an empty bar beside a full one; three "
             "falling bars; two lines that cross."),
        path=str(IMG / f"ml154-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["gen"] = s.run("""
        import numpy as np
        from standarderror.ml import llmeval as le
        from standarderror.schoolmath import arithmetic as ar
        from standarderror.schoolmath import model as sm

        model = sm.load()                      # the main arithmetic model
        sums = le.items("probe_length", "add", 200)     # four-digit operands
        fills = ar.fillers(len(sums), seed=9)  # one fresh context per problem
        rows = ar.graded(model, sums, filler=fills)
        for r in rows[:4]:
            print(f"{r['prompt']:12s} want {r['want']:6s} got {r['got']}")
        print("exact:", np.mean([r["ok"] for r in rows]),
              " shorter than the answer:",
              np.mean([len(r["got"]) < len(r["want"]) for r in rows]))
        """)
    out["mc"] = s.run("""
        rng = np.random.default_rng(2)
        p = sums[0]
        for kind in ("random", "near", "length"):
            options = [p.answer] + le.distractors(p.answer, kind, rng)
            lp = le.answer_logp(model, [p.prompt] * 4, options, [fills[0]] * 4)
            pick = options[int(np.argmax(lp))]
            print(f"{kind:6s} options {options}  picked {pick}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    sc = res["scores"]

    pl = {n: sc[f"probe_length/add/{n}"] for n in NAMES}
    mc, ml = sc["test/mul/clean"], sc["test/mul/leaky"]
    oc, ol = sc["test/order/clean"], sc["test/order/leaky"]
    pc, pk = sc["probe_carry/add/clean"], sc["probe_carry/add/leaky"]
    sh = sc["test/mul/s1_6000"]
    # Largest change from averaging per character, anywhere in the table.
    norm_moves = [abs(r[k] - r[k + "_avg"]) for r in sc.values()
                  for k in ("random", "near", "length")]

    # The spine, asserted rather than trusted.
    assert all(v["gen"] == 0.0 for v in pl.values())
    assert pl["clean"]["random"] > 0.9 and pl["leaky"]["random"] > 0.9
    assert 0.7 < pl["s1_6000"]["random"] < 0.85
    assert 0.35 < pl["clean"]["near"] < 0.45
    assert 0.25 < pl["clean"]["length"] < 0.35
    assert pl["clean"]["length_shorter"] > 0.5 and res["shorter"] > 0.95
    assert mc["gen"] > ml["gen"] + 0.05 and abs(mc["near"] - ml["near"]) < 0.03
    assert oc["gen"] >= ol["gen"] and ol["near"] > oc["near"] + 0.03
    assert pk["gen"] > pc["gen"] and pk["near"] > pc["near"]
    assert all(sc[k]["random"] >= 0.99 for k in sc if k.startswith("test/"))
    assert mc["near"] - ml["near"] > -0.03 and mc["gen"] - ml["gen"] > 0.09
    assert abs(oc["gen"] - ol["gen"]) <= 0.01
    assert res["first1"] > 0.85 and 0.5 < res["first2"] < 0.7
    assert sh["gen"] < 0.25 and sh["random"] > 0.99
    assert max(norm_moves) <= 0.045
    assert sorted(norm_moves)[-2] <= 0.011

    post = Post(
        title=(f"{SERIES_TAG} 54: Multiple Choice Grades the Wrong Answers "
               f"You Offered"),
        slug="ml-54-multiple-choice",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=54,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=(f"Ranking answer options by likelihood is a different "
                  f"test from asking for the answer. A model that never "
                  f"produced a four-digit sum picked it out of four options "
                  f"{pl['clean']['random']:.1%} of the time, or "
                  f"{pl['clean']['length']:.0%}, depending on the options."),
        summary=(
            f"Three small arithmetic transformers, the same problems in the "
            f"same contexts, two ways of scoring. On four-digit sums none of "
            f"them generated a single right answer in 200; asked to pick the "
            f"likeliest of four options, they scored {pl['clean']['random']:.3f}, "
            f"{pl['leaky']['random']:.2f} and {pl['s1_6000']['random']:.2f} "
            f"against random numbers. The main model's score on the same "
            f"sums was {pl['clean']['near']:.3f} against one-digit slips and "
            f"{pl['clean']['length']:.3f} against options of different "
            f"lengths, close to the 0.25 of guessing. On "
            f"multiplication the main model generated {mc['gen']:.3f} to its "
            f"twin's {ml['gen']:.3f}; under near-miss multiple choice the "
            f"two were level, {mc['near']:.3f} and {ml['near']:.3f}. Averaging "
            f"likelihood per character moved no score by more than "
            f"{max(norm_moves):.3f}."),
        tags=["machine-learning", "llm-evaluation", "language-models",
              "benchmarks", "evaluation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Models: the 804k-parameter arithmetic transformers of the "
            "school-maths series (`data/schoolmath/schoolmath.pt`, its leaky "
            "twin, and the 6,000-step snapshot of a separate run, "
            "`data/schoolmath/family/s1_6000.pt`), all trained by this "
            "repository on generated arithmetic.",
            "Problems: 200 drawn at random per task from the held-out test "
            "split (multiplication, order of operations), the three-digit "
            "carry probe and the four-digit length probe of "
            "`standarderror/schoolmath/curriculum.py`. One context of twelve "
            "random training lines per problem, shared by both scoring "
            "rules. Three distractors per problem (seed 2).",
            "Machinery: `standarderror/ml/llmeval.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Gao et al., *A framework for few-shot language "
            "model evaluation* (the lm-evaluation-harness, 2021 onward), for "
            "accuracy with and without length normalisation; Hendrycks et "
            "al., \"Measuring massive multitask language understanding\", "
            "*ICLR* (2021); Schaeffer, Miranda and Koyejo, \"Are emergent "
            "abilities of large language models a mirage?\", *NeurIPS* "
            "(2023); episode 47 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("committed weights, seeded draws; about three "
                            "minutes of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Two ways to ask",
        f"""There are two common ways to score a language model on a question with one right answer. The first asks it: generate a continuation and check whether it matches. The second offers it options: compute how likely the model finds each candidate answer, and count the item right if the true answer is the likeliest. The second is cheap, needs no parsing of free text, and is how many multiple-choice benchmarks are commonly scored; evaluation harnesses report it with and without dividing each option's log-likelihood by its length. It is usually treated as the same measurement in a more convenient form, the sentence under test: *scoring a model by which answer option it finds likeliest measures the same thing as asking it for the answer*.

This arc evaluates models the reader did not train, and the models here are the arithmetic transformers of the school-maths series: four blocks, 804,096 parameters, trained on generated sums with held-out probes. They are tiny, and that is the reason to use them: every answer can be checked exactly, so each evaluation habit can be compared with the truth rather than with another score. Three of them are used: the main model, its leaky twin from episode 1 of that series (same data, unshuffled), and a 6,000-step snapshot of a separate, shorter run, which is much weaker.

Four-digit sums are the length probe: the models saw operands of up to three digits.

{snip['gen'].markdown()}

All three models generated **no** right answer to 200 four-digit sums; {res['shorter']:.1%} of the main model's answers were shorter than the truth, typically a truncated version of roughly the right number: the first digit was right in {res['first1']:.0%} of them, the first two in {res['first2']:.0%}, and {res['prefix']:.1%} were an exact prefix of the answer.""",
        figures=[figs["f0"]])

    post.add(
        "Picking what it cannot produce",
        f"""Offer the same model the same sums as four options, the right answer and three random numbers with as many digits, and it picks the right one {pl['clean']['random']:.3f} of the time. Its twin scores {pl['leaky']['random']:.3f} and the short run, which generates far less correctly than either ({sh['gen']:.3f} on multiplication), {pl['s1_6000']['random']:.3f}. Nothing about the models changed between the two columns, only the question asked of them.

The reason is visible in the generations. A model that writes a truncated version of nearly the right number has most of its probability on strings that start with the right digits. A random four- or five-digit number almost never shares them, so the true answer wins the comparison without the model ever being able to finish it. Multiple choice measures whether the model prefers the right answer to the wrong ones offered; that is a real ability, closer to recognition than to production, and it is a different one.

{snip['mc'].markdown()}

So the score is set by the wrong answers. Against one-digit slips, the errors a person or a model actually makes, the main model's score on the same sums fell to {pl['clean']['near']:.3f}. Against options that differ in length, a digit dropped, appended or prepended, it fell to {pl['clean']['length']:.3f}, barely above the 0.25 of guessing, and in {pl['clean']['length_shorter']:.0%} of items the option it preferred was shorter than the answer: the same truncation it generates. Three numbers anyone could publish, {pl['clean']['random']:.3f}, {pl['clean']['near']:.3f} and {pl['clean']['length']:.3f}, for one model on one set of questions.""")

    post.add(
        "Which model is better depends on the options",
        f"""Within the training lengths the gap between the two scoring rules is smaller but still not a constant. On multiplication the main model generated {mc['gen']:.3f} of answers correctly and its leaky twin {ml['gen']:.3f}, a clear lead. Against near-miss options the twin scored {ml['near']:.3f} and the main model {mc['near']:.3f}: a lead of {mc['gen'] - ml['gen']:.3f}, more than two standard errors, became a difference well inside one. On problems mixing addition and multiplication, generation put the two models level ({oc['gen']:.3f} and {ol['gen']:.3f}) and multiple choice put the twin ahead by {ol['near'] - oc['near']:.3f}, about one standard error. On three-digit carries both rules agreed that the twin was better. Neither rule's ranking is noise-free on 200 problems; what is not noise is the size of the multiplication gap under one rule and its absence under the other.

Against random distractors every model scored 0.99 or more on multiplication and mixed operations (and the short run {sc['probe_carry/add/s1_6000']['random']:.2f} on three-digit carries), including the short run, which generates {sh['gen']:.3f} of multiplications correctly. A benchmark scored that way cannot tell the three models apart. Schaeffer and colleagues made the related point about apparent "emergent" abilities: a discontinuous metric such as exact match can make smooth improvement look sudden. Here the arrow runs the other way, and a lenient metric made a large real gap look like none. Episode 47 found the same pattern for classifiers: a summary that does not match the use can rank models in the opposite order to the decision they serve.""",
        figures=[figs["f1"]])

    post.add(
        "Length normalisation did nothing here",
        f"""The usual worry about likelihood scoring is length: a longer option multiplies more probabilities together and so looks less likely, and harnesses offer a per-character or per-byte average to correct for it. Here the correction moved no score by more than {max(norm_moves):.3f}, and every score but one by {sorted(norm_moves)[-2]:.3f} or less. Most option sets had a single length, so there was nothing to correct. In the length-varied set, a likely reason is that each option is scored with its closing newline, which charges a short option for stopping early; that was not tested separately.

That is a result about these options, not a reassurance in general. Benchmarks whose options are sentences of different lengths are exactly where the two versions are known to disagree, which is why harnesses report both. The point that does generalise is that the choice between them is one more scoring decision, made before any model is run, that the headline number depends on.""")

    post.add(
        "Where this breaks",
        """**Arithmetic, not language.** Real multiple-choice benchmarks such as MMLU (Hendrycks and colleagues, 2021) have options written by people, and are often scored by the letter the model picks rather than by the likelihood of the answer text. Scoring the letter is a third rule, with its own failure modes, and it was not measured.

**Small models.** Large models can be asked to choose explicitly and may reason before answering, and the gap between recognising and producing may be smaller for them. The mechanism shown here, a partial answer that still beats unrelated options, is not specific to small models, though how often it operates in a large one was not measured.

**Distractors of three invented kinds.** A benchmark's distractors are usually chosen to be plausible to people, which is closer to the near-miss column than to the random one; how close is a property of each benchmark.

**200 problems per cell.** Differences of a few points between models under near-miss multiple choice sit within about one standard error; the large gaps between scoring rules do not.""")

    post.add(
        "What to keep",
        f"""1. Three models that generated no four-digit sum correctly picked it from four options {pl['clean']['random']:.3f}, {pl['leaky']['random']:.3f} and {pl['s1_6000']['random']:.3f} of the time against random distractors.

2. The main model's score on the same sums was {pl['clean']['random']:.3f}, {pl['clean']['near']:.3f} or {pl['clean']['length']:.3f} depending only on the kind of wrong options offered.

3. The main model's {mc['gen'] - ml['gen']:.3f} lead over its twin in generating multiplications disappeared under near-miss multiple choice; against random options every model scored at least 0.99 on multiplication and mixed operations.

4. Treat a multiple-choice score as accuracy against those particular distractors. Report generation where the use is generation, and state the distractors and the normalisation along with the number.""")

    post.add(
        "Exercise",
        """For a multiple-choice benchmark you rely on, take fifty items and replace the wrong options with ones written to be plausible near misses, keeping the right answer. Score your model on both versions.

If the score barely moves, the original distractors were already doing their job. If it drops a lot, the published number was measuring how easy the wrong answers were to reject, and comparisons with other models on it are comparisons on that.""")

    post.add(
        "Next",
        """Every score in this arc so far has been checked against an exact answer. Open-ended outputs have no exact answer, and the common substitute is to ask another model which answer is better. Episode 55 gives each of these three models the job of grading all three, and measures whose answers each one prefers.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
