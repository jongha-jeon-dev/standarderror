"""ML 55: Every Judge Ranked Itself First.

The fifty-fifth episode of *Machine Learning, Taught Through What Breaks*,
closing Arc XI. "A model can grade other models' answers in place of
people." Measured with three of the repository's arithmetic transformers,
each answering 300 problems on three tasks and each judging all three
models' answers by its own likelihood:

* In all nine judge-and-task combinations the judge ranked its own answers
  first, with a threshold set so that its overall pass rate was exactly
  right. The weakest model, right on 1.3% of three-digit carries, passed
  73% of its own answers.
* Within any one model's answers the two strong judges told right from
  wrong with AUCs from 0.59 to 0.99; the bias was a shift of the whole scale
  towards their own answers. Judging two other models they always put a
  strong model above the far weaker one; the weak judge, asked about the
  two close strong models, got their order wrong on carries.
* Forty human-checked labels removed the bias: prediction-powered inference
  covered the true accuracy in about 94% of draws, where the judge alone
  covered it at most 66% of the time and as little as never. Where the judge was weak it
  made the interval wider than the labels alone, unless its weight was tuned.

Run: `standarderror run ml155_judge --publish`
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

MODELS = ["clean", "leaky", "s1_6000"]
NAMES = {"clean": "main model", "leaky": "leaky twin",
         "s1_6000": "short run"}
CASES = {"probe_carry/add": "3-digit carries",
         "test/order": "order of operations", "test/mul": "multiply"}
LABELED = 40


def compute() -> dict:
    models = {"clean": sm.load(), "leaky": sm.load(adjacent=True),
              "s1_6000": sm.load_member("s1_6000")}
    scores = le.judge_scores(models)
    view = {f"{c}/{j}": le.judged(scores, c, j, MODELS)
            for c in CASES for j in MODELS}
    ppi = {}
    for c in CASES:
        t = view[f"{c}/clean"]["threshold"]
        for s in ("clean", "leaky"):
            d = scores[f"{c}/clean/{s}"]
            ppi[f"{c}/{s}"] = le.ppi_study(np.array(d["score"]) > t,
                                           np.array(d["ok"]),
                                           labeled=LABELED)
    return {"view": view, "ppi": ppi}


# ------------------------------------------------------------------ figures

def _dots(ax, m, view):
    cases = list(CASES)
    for ci, c in enumerate(cases):
        for ji, j in enumerate(MODELS):
            v = view[f"{c}/{j}"]
            for si, s in enumerate(MODELS):
                x = ci * 4 + ji + (si - 1) * 0.22
                ax.plot([x, x], [v[s]["true"], v[s]["judged"]],
                        color=m.series[si], lw=1.5)
                ax.scatter([x], [v[s]["judged"]], color=m.series[si], s=26,
                           zorder=3, marker="o" if s == j else "x")
    ax.set_xticks([ci * 4 + ji for ci in range(len(cases))
                   for ji in range(3)])
    ax.set_xticklabels(["main", "twin", "short"] * len(cases), fontsize=8)
    for ci, c in enumerate(cases):
        ax.text(ci * 4 + 1, 1.04, CASES[c], ha="center", fontsize=9)
    for si, s in enumerate(MODELS):
        ax.plot([], [], color=m.series[si], label=f"answers of {NAMES[s]}")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="center right", frameon=False)


def figures(res: dict) -> dict:
    view, ppi = res["view"], res["ppi"]
    out: dict = {}
    sr = view["probe_carry/add/s1_6000"]["s1_6000"]
    out["f0"] = charts.diagram(
        lambda ax, m: _dots(ax, m, view),
        title="Each judge, grading all three models",
        subtitle=("For each task, three judges (x axis); for each, the true "
                  "accuracy of each model's answers (line start) and the "
                  "share the judge passed (marker). A circle marks the "
                  "judge's own answers."),
        xlabel="judge", ylabel="accuracy, true and judged",
        figsize=(8.2, 4.6),
        source="standarderror/ml/llmeval.py; models in data/schoolmath.",
        alt=("Nine groups of three lines. In every group the circle, the "
             "judge's own answers, is pushed up from its true accuracy and "
             "the other two are pushed down."),
        caption=(f"The short run's answers to three-digit carries were right "
                 f"{sr['true']:.1%} of the time; judging them itself, it "
                 f"passed **{sr['judged']:.1%}**."),
        path=str(IMG / f"ml155-f0-judges.{EXT}"))[0]

    body = []
    for c, cl in CASES.items():
        for j in MODELS:
            v = view[f"{c}/{j}"]
            body.append([cl, NAMES[j]]
                        + [f"{v[s]['true']:.2f} / {v[s]['judged']:.2f}"
                           for s in MODELS]
                        + [f"{v[j]['pass_wrong']:.2f}"])
    out["f1"] = charts.table_image(
        body,
        header=["task", "judge", "main: true / judged",
                "twin: true / judged", "short: true / judged",
                "own wrong passed"],
        title="Nine judges, nine self-rankings",
        subtitle=("True accuracy and the share passed by the judge, for "
                  "each model's 300 answers. The threshold makes the "
                  "judge's pass rate over all 900 answers equal the true "
                  "rate."),
        source="standarderror/ml/llmeval.py.",
        alt=("A nine-row table. In every row the judge's own column has the "
             "highest judged accuracy."),
        caption="The last column: the share of the judge's own wrong answers it passed.",
        align="llrrrr",
        path=str(IMG / f"ml155-f1-table.{EXT}"))[0]

    body = []
    for c, cl in CASES.items():
        for s in ("clean", "leaky"):
            p = ppi[f"{c}/{s}"]
            body.append([cl, NAMES[s], f"{p['target']:.3f}"]
                        + [f"{p[k]['coverage']:.2f} / {p[k]['width']:.2f}"
                           for k in ("judge", "labels", "ppi", "tuned")])
    out["f2"] = charts.table_image(
        body,
        header=["task", "answers of", "true", "judge alone",
                f"{LABELED} labels alone", "PPI", "PPI, tuned"],
        title="Forty labels against a biased judge",
        subtitle=("The main model judges; coverage of the true accuracy by a "
                  "nominal 95% interval, and the interval's average width, "
                  "over 4,000 draws of 300 judged and 40 labelled answers."),
        source="standarderror/ml/llmeval.py.",
        alt=("A six-row table. The judge alone covers the truth between 0 "
             "and 66% of the time; the three label-based columns cover it "
             "92 to 95% of the time, with widths that differ."),
        caption=("Each cell: coverage / width. PPI = prediction-powered "
                 "inference; tuned = with the judge's weight estimated from "
                 "the labels."),
        align="llrrrrr",
        path=str(IMG / f"ml155-f2-ppi.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    view, ppi = res["view"], res["ppi"]
    sr = view["probe_carry/add/s1_6000"]["s1_6000"]

    def p1(panel, m):
        panel.bar([0, 1], [sr["true"], sr["judged"]],
                  color=[m.series[0], m.series[1]])
        panel.text(0, sr["true"] + 0.02, f"{sr['true']:.2f}", ha="center",
                   fontsize=8)
        panel.set_ylim(0, 1)

    def p2(panel, m):
        v = view["test/order/clean"]
        panel.bar([0, 1], [v["clean"]["pass_wrong"], v["leaky"]["pass_wrong"]],
                  color=[m.series[1], m.series[2]])

    def p3(panel, m):
        p = ppi["probe_carry/add/leaky"]
        panel.text(0, 0.02, f"{p['judge']['coverage']:.2f}", ha="center",
                   fontsize=8)
        panel.bar([0, 1], [p["judge"]["coverage"],
                           p["ppi"]["coverage"]],
                  color=[m.series[1], m.series[2]])
        panel.set_ylim(0, 1)

    n_self = sum(max(MODELS, key=lambda s: view[f"{c}/{j}"][s]["judged"]) == j
                 for c in CASES for j in MODELS)
    return charts.lecture_hero(
        series=SERIES_TAG, episode=55,
        headline="Its own answers",
        panels=[(p1, f"{sr['judged']:.0%}", "self-graded, right 1%"),
                (p2, f"{n_self} of 9", "judges ranked self first"),
                (p3, f"{ppi['probe_carry/add/leaky']['ppi']['coverage']:.0%}",
                 "covered, with 40 labels")],
        note=("A model judging by its own sense of what looks right finds "
              "its own mistakes convincing. A few real labels fix the bias; "
              "they do not come free."),
        alt=("Three hand-drawn frames: a tiny bar beside a tall one; a tall "
             "bar beside a short one; an empty bar beside a full one."),
        path=str(IMG / f"ml155-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["judge"] = s.run("""
        import numpy as np
        from standarderror.ml import llmeval as le
        from standarderror.schoolmath import arithmetic as ar
        from standarderror.schoolmath import model as sm

        main, short = sm.load(), sm.load_member("s1_6000")
        sums = le.items("probe_carry", "add", 300)[:6]    # three-digit carries
        gen = ar.fillers(6, seed=9)                       # contexts to answer in
        ctx = ar.fillers(6, seed=21)                      # contexts to judge in
        a = [r["got"] for r in ar.graded(short, sums, filler=gen)]
        for judge, name in ((short, "short"), (main, "main")):
            lp = le.answer_logp(judge, [p.prompt for p in sums], a, ctx)
            print(f"{name:5s} judges short's answers:",
                  np.round(lp / [len(x) + 1 for x in a], 2))
        print("right answers:", [p.answer for p in sums])
        print("short wrote:  ", a)
        """)
    out["ppi"] = s.run("""
        models = {"clean": main, "leaky": sm.load(adjacent=True)}
        sc = le.judge_scores(models, [("probe_carry", "add")])
        v = le.judged(sc, "probe_carry/add", "clean", ["clean", "leaky"])
        d = sc["probe_carry/add/clean/leaky"]     # the main model judges the twin
        r = le.ppi_study(np.array(d["score"]) > v["threshold"],
                         np.array(d["ok"]), labeled=40, draws=1000)
        print("true accuracy", round(r["target"], 3))
        for k in ("judge", "labels", "ppi", "tuned"):
            print(f"{k:6s} bias {r[k]['bias']:+.3f}  width {r[k]['width']:.3f}"
                  f"  coverage {r[k]['coverage']:.2f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)
    view, ppi = res["view"], res["ppi"]

    def first(c, j):
        return max(MODELS, key=lambda s: view[f"{c}/{j}"][s]["judged"])

    n_self = sum(first(c, j) == j for c in CASES for j in MODELS)
    sr = view["probe_carry/add/s1_6000"]["s1_6000"]
    oc = view["test/order/clean"]
    cc = view["probe_carry/add/clean"]
    easy_gaps = [view[f"{c}/clean"][a]["true"] - view[f"{c}/clean"]["s1_6000"]["true"]
                 for c in CASES for a in ("clean", "leaky")]
    weak = {c: view[f"{c}/s1_6000"] for c in CASES}
    wc = weak["probe_carry/add"]
    weak_verdict = (f"got their order wrong on three-digit carries (passing "
                    f"{wc['clean']['judged']:.3f} of the main model's answers "
                    f"and {wc['leaky']['judged']:.3f} of the twin's, against "
                    f"true rates of {wc['clean']['true']:.3f} and "
                    f"{wc['leaky']['true']:.3f})")
    # The two strong judges on the two models that are not themselves.
    others = []
    for c in CASES:
        for j, (a, b) in (("clean", ("leaky", "s1_6000")),
                          ("leaky", ("clean", "s1_6000"))):
            v = view[f"{c}/{j}"]
            others.append((v[a]["judged"] - v[b]["judged"])
                          * (v[a]["true"] - v[b]["true"]) > 0)
    aucs = [view[f"{c}/{j}"][s]["auc"] for c in CASES
            for j in ("clean", "leaky") for s in MODELS]
    pj = [p["judge"]["coverage"] for p in ppi.values()]
    pp = [p["ppi"]["coverage"] for p in ppi.values()]
    pt = [p["tuned"]["coverage"] for p in ppi.values()]
    carry = ppi["probe_carry/add/leaky"]
    mulk = ppi["test/mul/leaky"]

    # The spine, asserted rather than trusted.
    assert n_self == 9
    for c in CASES:   # the weaker of the close pair, judging, reverses them
        v = view[f"{c}/clean"]
        weak = "clean" if v["clean"]["true"] < v["leaky"]["true"] else "leaky"
        strong = "leaky" if weak == "clean" else "clean"
        w = view[f"{c}/{weak}"]
        assert w[weak]["judged"] > w[strong]["judged"]
    assert sr["true"] < 0.02 and sr["judged"] > 0.7
    assert all(others) and len(others) == 6
    assert min(easy_gaps) > 0.25
    assert wc["clean"]["judged"] > wc["leaky"]["judged"]
    assert wc["clean"]["true"] < wc["leaky"]["true"]
    assert oc["clean"]["pass_wrong"] > 4 * oc["leaky"]["pass_wrong"]
    assert cc["clean"]["mean_right"] > cc["leaky"]["mean_right"] + 0.2
    assert cc["clean"]["mean_wrong"] > cc["leaky"]["mean_right"]
    assert max(aucs) > 0.98 and 0.55 < min(aucs) < 0.65
    assert max(pj) < 0.7 and min(pj) < 0.02
    assert all(0.9 < c < 0.97 for c in pp + pt)
    assert carry["ppi"]["width"] > carry["labels"]["width"] + 0.05
    assert carry["tuned"]["width"] <= carry["labels"]["width"] + 0.005
    assert mulk["ppi"]["width"] < 0.7 * mulk["labels"]["width"]
    assert all(abs(p["ppi"]["bias"]) < 0.01 for p in ppi.values())

    post = Post(
        title=f"{SERIES_TAG} 55: Every Judge Ranked Itself First",
        slug="ml-55-judge",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=55,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("Three models graded each other's arithmetic by how right "
                  "each answer looked to them. All three preferred their "
                  "own answers on every task, and forty real labels were "
                  "enough to correct them."),
        summary=(
            f"Three small arithmetic transformers answered 300 problems on "
            f"each of three tasks, and each judged all three models' answers "
            f"by its own per-character likelihood, with a threshold set so "
            f"its overall pass rate was right, up to ties. In all nine "
            f"judge-and-task pairs the judge ranked its own answers first; "
            f"the weakest model, right on {sr['true']:.1%} of three-digit "
            f"carries, passed {sr['judged']:.0%} of its own. Judging two "
            f"other models, the two strong judges ordered them correctly six "
            f"times out of six. With 40 labelled answers, "
            f"prediction-powered inference removed the bias and covered the "
            f"true accuracy in {min(pp):.0%} to {max(pp):.0%} of draws, "
            f"against at most {max(pj):.0%} for the judge alone; where the "
            f"judge was weak its interval was wider than the labels' alone "
            f"({carry['ppi']['width']:.2f} against "
            f"{carry['labels']['width']:.2f}) unless the judge's weight was "
            f"tuned."),
        tags=["machine-learning", "llm-evaluation", "llm-as-judge",
              "language-models", "evaluation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Models: the main arithmetic transformer of the school-maths "
            "series, its leaky twin, and the 6,000-step snapshot of a "
            "separate run (`data/schoolmath/family/s1_6000.pt`), all trained "
            "by this repository on generated arithmetic.",
            "Problems: 300 drawn at random per task from the three-digit "
            "carry probe and the test split's multiplication and order-of-"
            "operations problems. Answers generated greedily in one set of "
            "contexts (seed 9), judged in another (seed 21).",
            "Machinery: `standarderror/ml/llmeval.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Zheng et al., \"Judging LLM-as-a-judge with "
            "MT-Bench and Chatbot Arena\", *NeurIPS* (2023); Panickssery, "
            "Bowman and Feng, \"LLM evaluators recognize and favor their own "
            "generations\", *NeurIPS* (2024); Angelopoulos, Bates, Fannjiang, "
            "Jordan and Zrnic, \"Prediction-powered inference\", *Science* "
            "(2023), and Angelopoulos, Duchi and Zrnic, \"PPI++: efficient "
            "prediction-powered inference\" (2023); episodes 39 and 47 of "
            "this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("committed weights, seeded draws; about two "
                            "minutes of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A judge in its simplest form",
        f"""Arithmetic has exact answers. Most of what language models are now asked to produce, summaries, explanations, code that should be readable, does not, and checking it by hand is slow. The common substitute is to ask a model to grade the answers, either one at a time or by choosing the better of two. Zheng and colleagues reported in 2023 that strong judges agreed with human preferences about as often as humans agreed with each other, and documented biases for position and for length, with some signs of judges favouring their own answers that the data were too limited to confirm. The sentence under test is the substitution itself: *a model can grade other models' answers in place of people*.

The judge here is the simplest one there is. Each of the three arithmetic models from episode 54 (the main model, its leaky twin and a much weaker short run) answers 300 problems on each of three tasks. Then each model reads every answer, its own and the other two models', after the question, in a context different from the one it answered in, and reports its average log-probability per character. High means "this looks right to me". A real judge is prompted for a verdict instead of read for a likelihood, and its verdict plausibly draws on the same sense of what is likely. Panickssery and colleagues found that large models' preference for their own outputs rose with their ability to recognise them, though they did not test a likelihood mechanism.

To give the judge every advantage, its pass threshold is set after the fact, so that over all 900 answers it passes as many as are right, up to ties in its scores. Any bias that survives this is not a calibration problem.

{snip['judge'].markdown()}""")

    post.add(
        "Nine judges, nine self-rankings",
        f"""On every task, each of the three judges passed a higher share of its own answers than of either other model's: **{n_self} of 9** judge-and-task pairs ranked the judge first. For the main model and its twin, which are close, that reversed the true order whenever the judge was the weaker of the two on a task: on three-digit carries the main model is right {cc['clean']['true']:.3f} of the time and the twin {cc['leaky']['true']:.3f}, and the main model as judge passed {cc['clean']['judged']:.3f} of its own answers and {cc['leaky']['judged']:.3f} of the twin's.

The short run is the extreme case. Its answers to three-digit carries were right {sr['true']:.1%} of the time, and judging them itself it passed **{sr['judged']:.0%}**. A model that is the worst on every task by a wide margin was, in its own judgement, the best on every task.""",
        figures=[figs["f0"]])

    post.add(
        "Its own mistakes look right to it",
        f"""The likely mechanism is simple. A model's greedy answer is built from the characters it finds most likely, so the same model, asked how likely that answer is, finds it likely, right or wrong; the judging context differs from the answering one, so this is an expectation rather than an identity, and the scores bear it out. On problems mixing addition and multiplication the main model passed {oc['clean']['pass_wrong']:.0%} of its own wrong answers and {oc['leaky']['pass_wrong']:.0%} of the twin's. On three-digit carries its average score for its own *wrong* answers, {cc['clean']['mean_wrong']:.3f}, was higher than for the twin's *right* ones, {cc['leaky']['mean_right']:.3f}.

What the judges did well is worth stating as precisely. Within any one model's answers, the main model and the twin separated right from wrong answers with AUCs (the area under the ROC curve, as in episode 47) from {min(aucs):.2f} on three-digit carries to {max(aucs):.2f} on multiplication. The bias was not mainly an inability to tell right from wrong; each judge put its own answers on a higher scale. Judging two models that were neither of them, the main model and the twin ordered the remaining pair correctly in all six comparisons, but that was the easy case: each pair set a strong model against the short run, a gap of {min(easy_gaps):.2f} to {max(easy_gaps):.2f} in true accuracy. The only close comparison by a judge not involved in it was the short run judging the main model against the twin, and it {weak_verdict}.

The usual advice is to use a judge from a different model family. These three models share their training data, so they cannot test that advice; what they show is that a judge's verdict on itself is the least trustworthy thing it produces, and that a judge outside the contest is not automatically a good one.""",
        figures=[figs["f1"]])

    post.add(
        "Forty labels",
        f"""The repair that does not depend on finding an unrelated judge is a small set of answers checked by people. Prediction-powered inference, introduced by Angelopoulos and colleagues in 2023, uses the judge on every answer and the labels on a few: the estimate is the judge's pass rate plus the judge's average error on the labelled answers. The error term removes the bias, and the judge's large sample keeps the interval narrow, if the judge is good.

{snip['ppi'].markdown()}

Across the three tasks, with the main model judging its own answers or the twin's, 300 judged answers and {LABELED} labelled ones, the judge alone covered the true accuracy with its nominal 95% interval in at most {max(pj):.0%} of 4,000 draws, and in one case in {min(pj):.1%}: its interval is narrow and in the wrong place. Prediction-powered inference covered it in {min(pp):.0%} to {max(pp):.0%}. Where the judge was good, on the twin's multiplications, its interval was {mulk['ppi']['width']:.2f} wide against {mulk['labels']['width']:.2f} for the {LABELED} labels alone.

Where the judge was poor it hurt. On three-digit carries, the plain estimate's interval was {carry['ppi']['width']:.2f} wide, against {carry['labels']['width']:.2f} from the labels alone, because correcting a noisy judge adds the judge's noise. The tuned version, which estimates from the labels how much weight the judge deserves, fell back to {carry['tuned']['width']:.2f}, about what the labels gave alone, and covered {carry['tuned']['coverage']:.0%}. Episode 39 met the same structure from the other side: a small sample nobody selected can make a large selected one usable, when the correction's form is right. (The judge's pass threshold here was set using all 900 true labels, a single number that the labelled sample is not independent of; its effect on coverage is negligible at this size but not zero.)""",
        figures=[figs["f2"]])

    post.add(
        "What the arc measured",
        """Arc XI took the habits of the series to a model somebody else built. A benchmark score was a sample of questions, and a question asked five ways was still close to one question (episode 51). Leaked answers raised the score on the leaked items while a loss-based detector barely noticed (52). The prompt was part of the measurement, and the gap between two models depended on it (53). Multiple choice measured the distractors as much as the model (54). And a model asked to grade others graded itself highest (55).

None of these needed a large model to show. Each is an older problem of the series in new clothes: sampling error, leakage, a choice made on the test set, a metric that does not match the use, labels that come from the system being judged. The tools that answered them earlier, paired comparisons, fresh data, held-out selection, evaluation at the point of use and a small random sample checked by people, answer them here.""")

    post.add(
        "Where this breaks",
        """**A likelihood judge.** Real judges are prompted, often asked to compare two answers, and can reason before answering. Their biases include position and length as well as self-preference, and their sizes differ from these.

**Exact answers.** Arithmetic made the truth free, which is what let every judge be checked. For open-ended outputs the "truth" is itself a human judgement with its own noise, and the labelled set in prediction-powered inference inherits it.

**Three tiny models trained on the same data.** All three were trained on the same lines, and the main model and the twin differ only in their order, which is the case where a judge is most likely to recognise a contestant. A judge from a different family might be biased differently or less, and that was not measured.

**Fixed threshold.** A judge that outputs a score rather than a pass could be corrected by a per-model calibration; that needs labels per model, which is the prediction-powered recipe again.""")

    post.add(
        "What to keep",
        f"""1. Each of three models, judging all three models' answers by its own likelihood with a threshold calibrated on the truth, ranked itself first on every task: {n_self} of 9.

2. The weakest model, right on {sr['true']:.1%} of three-digit carries, passed {sr['judged']:.0%} of its own answers. Judging the other two models, the strong judges always put a strong model above the far weaker one; the weak judge misordered the two close strong models on carries.

3. With {LABELED} labelled answers, prediction-powered inference removed the bias and covered the truth in about 94% of draws; with a weak judge the plain version was wider than labels alone, and the tuned version was not.

4. Do not let a model grade itself or a close relative. Check any judge against a small random sample graded by people, and use that sample to correct its estimate, not only to decide whether to trust it.""")

    post.add(
        "Exercise",
        """If you use a model to grade outputs, have people grade a random 50 of the same outputs, and compute the judge's pass rate and the people's on those 50, separately for each system being compared.

If the gap differs between systems, the judge's ranking is partly a ranking of how much each system resembles the judge. The prediction-powered estimate from those 50 labels is the number to report alongside it.""")

    post.add(
        "Next",
        """Every episode so far has had labels, or built them. Arc XII drops them. Clustering, embeddings, mixture models and anomaly scores are trained with no right answer to check against, which makes them easy to believe and hard to test; episode 56 starts with a picture that looks like clusters.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
