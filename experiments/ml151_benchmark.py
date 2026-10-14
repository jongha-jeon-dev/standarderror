"""ML 51: A Benchmark Is a Sample of Questions.

The fifty-first episode of *Machine Learning, Taught Through What Breaks*,
opening Arc XI. "The model with the higher benchmark score is the better
model." Measured on ten of the repository's own arithmetic transformers,
graded on a pool of 2,500 problems and on benchmarks drawn from it:

* The models' 95% intervals covered their pool scores about 95% of the
  time on average (91-96% for individual models): the error bar on one
  score was close to right. The ranking was not: on 100
  questions the best model came first in about half of all benchmarks, and
  the seven close models were put in their true order almost never.
* The main model beat a fine-tuned copy by 0.038 on the pool. On 300
  questions an unpaired test detected it in 2.0% of benchmarks; the paired
  test on the same questions in 60%, because the two models' errors fall on
  the same questions.
* Asking each of 100 questions five times, in five contexts, made 500 rows
  and a standard error twice what 500 rows would suggest: the five answers
  to a question agreed almost always.

Run: `standarderror run ml151_benchmark --publish`
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

TOP = ["clean", "cont0", "cont1", "cont2", "cont3", "cont4", "leaky"]
SIZES = (50, 100, 200, 500, 1000, 2500)
LABEL = {"clean": "main model", "leaky": "leaky twin",
         "s1_6000": "short run, 6,000 steps",
         "s1_4500": "short run, 4,500 steps",
         "s1_3000": "short run, 3,000 steps"}
CLUSTER_CASES = [("test", "mul"), ("test", "order"), ("probe_carry", "add")]


def compute() -> dict:
    names, m, tasks = le.item_matrix()
    top = [names.index(n) for n in TOP]
    board = le.leaderboard_draws(m, SIZES)
    board_top = le.leaderboard_draws(m[top], SIZES)
    a = m[names.index("clean")]
    pairs = {}
    for other in ("cont0", "leaky"):
        b = m[names.index(other)]
        pairs[other] = {"pool": le.paired_difference(a, b),
                        "draws": le.paired_draws(a, b, n=300)}
    probs = []
    for split, task in CLUSTER_CASES:
        probs += le.items(split, task, 200)
    mat = le.clustered_items(sm.load(), probs, repeats=5)
    clusters = {q: le.cluster_study(mat, questions=q) for q in (20, 100)}
    return {"names": names, "scores": m.mean(1).tolist(),
            "by_task": {t: m[:, tasks == t].mean(1).tolist()
                        for t in sorted(set(tasks))},
            "board": board, "board_top": board_top, "pairs": pairs,
            "clusters": clusters}


# ------------------------------------------------------------------ figures

def _board(ax, m, res):
    n = [r["n"] for r in res["board"]]
    ax.plot(n, [r["top_right"] for r in res["board"]], color=m.series[0],
            marker="o", label="best model ranked first")
    ax.plot(n, [r["pairs_wrong"] for r in res["board_top"]],
            color=m.series[1], marker="s",
            label="close pairs in the wrong order")
    ax.plot(n, [r["order_right"] for r in res["board_top"]],
            color=m.series[2], marker="^",
            label="seven close models in true order")
    ax.plot(n, [r["coverage"] for r in res["board"]], color=m.series[3],
            ls="--", label="one model's 95% interval covers")
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="center right", frameon=False)


def figures(res: dict) -> dict:
    out: dict = {}
    b100 = next(r for r in res["board"] if r["n"] == 100)
    out["f0"] = charts.diagram(
        lambda ax, m: _board(ax, m, res),
        title="Right about each score, wrong about the order",
        subtitle=("Ten arithmetic models ranked on benchmarks of n questions "
                  "drawn (with replacement) from a pool of 2,500; the truth "
                  "is each model's score on the whole pool. 2,000 benchmarks "
                  "per size."),
        xlabel="questions in the benchmark", ylabel="share of benchmarks",
        figsize=(7.6, 4.4),
        source="standarderror/ml/llmeval.py; results in data/ml_llmeval.",
        alt=("Four lines against benchmark size. Interval coverage is flat "
             "near 0.95. The best model ranked first rises from about 0.44 at "
             "50 questions to 0.95 at 2,500; the seven close models in true "
             "order stays at or below about 0.14."),
        caption=(f"On 100 questions the models' intervals covered their true "
                 f"scores {b100['coverage']:.0%} of the time on average, and the best "
                 f"model came first in **{b100['top_right']:.0%}** of "
                 f"benchmarks."),
        path=str(IMG / f"ml151-f0-board.{EXT}"))[0]

    body = []
    for other, lab in (("cont0", "fine-tuned copy"), ("leaky", "leaky twin")):
        p, d = res["pairs"][other]["pool"], res["pairs"][other]["draws"]
        se_u = p["se_unpaired"] * np.sqrt(p["n"] / 300)
        se_p = p["se_paired"] * np.sqrt(p["n"] / 300)
        body.append([lab, f"{p['diff']:+.4f}", f"{p['corr']:.2f}",
                     f"{d['sd']:.4f}", f"{se_u:.4f}", f"{se_p:.4f}",
                     f"{d['sig_unpaired']:.1%}", f"{d['sig_paired']:.1%}",
                     f"{d['wrong_sign']:.1%}"])
    out["f1"] = charts.table_image(
        body,
        header=["main model vs", "pool gap", "error corr.", "actual SD",
                "unpaired SE", "paired SE", "unpaired: sig.",
                "paired: sig.", "wrong sign"],
        title="Two models on the same 300 questions",
        subtitle=("The gap on the 2,500-question pool; across 4,000 "
                  "benchmarks of 300 questions, the standard deviation of "
                  "the measured gap, each standard error's typical size, "
                  "how often each test called the gap significant at 5%, and "
                  "how often the benchmark got its sign wrong."),
        source="standarderror/ml/llmeval.py; results in data/ml_llmeval.",
        alt=("Two rows. Against the fine-tuned copy, the paired standard "
             "error matches the actual spread and the paired test detects "
             "the gap far more often than the unpaired one."),
        caption=("Standard errors scaled from the pool to 300 questions; "
                 "'error corr.' is the correlation between the two models' "
                 "right/wrong columns."),
        align="lrrrrrrrr",
        path=str(IMG / f"ml151-f1-paired.{EXT}"))[0]

    c = res["clusters"]

    def bars(ax, m):
        for i, q in enumerate((20, 100)):
            v = c[q]
            for j, (k, lab) in enumerate((("naive", "rows as independent"),
                                          ("robust", "clustered by question"),
                                          ("empirical", "actual spread"))):
                ax.bar(i * 4 + j, v[k], color=m.series[j],
                       label=lab if i == 0 else None)
        ax.set_xticks([1, 5])
        ax.set_xticklabels([f"{q} questions x 5 = {q * 5} rows"
                            for q in (20, 100)], fontsize=9)
        ax.legend(fontsize=8, frameon=False)

    out["f2"] = charts.diagram(
        bars,
        title="Five prompts per question are not five questions",
        subtitle=("The main model, each question asked in five different "
                  "contexts. Standard error of the benchmark score treating "
                  "all rows as independent, clustering them by question, "
                  "and the actual spread over 4,000 redrawn benchmarks."),
        xlabel="", ylabel="standard error of the score",
        figsize=(7.2, 4.2),
        source="standarderror/ml/llmeval.py; models in data/schoolmath.",
        alt=("Two groups of three bars. In each, the naive bar is about half "
             "the height of the other two, which match."),
        caption=(f"Of 600 questions, {c[100]['always']:.0%} were answered "
                 f"right in all five contexts and {c[100]['never']:.0%} in "
                 f"none."),
        path=str(IMG / f"ml151-f2-cluster.{EXT}"))[0]
    out["hero"] = _hero(res)
    return out


def _hero(res: dict):
    b100 = next(r for r in res["board"] if r["n"] == 100)
    d = res["pairs"]["cont0"]["draws"]
    c = res["clusters"][100]

    def p1(panel, m):
        panel.bar([0, 1], [b100["coverage"], b100["top_right"]],
                  color=[m.series[3], m.series[0]])
        panel.set_ylim(0, 1)

    def p2(panel, m):
        panel.bar([0, 1], [d["sig_unpaired"], d["sig_paired"]],
                  color=[m.series[1], m.series[2]])
        panel.text(0, d["sig_unpaired"] + 0.02, f"{d['sig_unpaired']:.2f}",
                   ha="center", fontsize=8)
        panel.set_ylim(0, 1)

    def p3(panel, m):
        panel.bar([0, 1], [c["naive"], c["empirical"]],
                  color=[m.series[0], m.series[2]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=51,
        headline="A sample of questions",
        panels=[(p1, f"{b100['top_right']:.0%}", "best model first, n=100"),
                (p2, f"{d['sig_paired']:.0%}", "gap found when paired"),
                (p3, f"{c['empirical'] / c['naive']:.1f}x",
                 "real error, 5 prompts")],
        note=("A benchmark score is an estimate from a sample of questions. "
              "Its error bar is right; a leaderboard built from many of them "
              "is not, unless the comparison uses the pairing."),
        alt=("Three hand-drawn frames: a full bar beside a half bar; a "
             "near-empty bar beside a tall one; a short bar beside one twice "
             "its height."),
        path=str(IMG / f"ml151-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(res: dict) -> dict:
    s = Session()
    out = {}
    out["pool"] = s.run("""
        import numpy as np
        from standarderror.ml import llmeval as le

        names, m, tasks = le.item_matrix()     # 10 models x 2,500 questions
        for name, row in zip(names, m):
            print(f"{name:8s} {row.mean():.4f}")
        """)
    out["board"] = s.run("""
        rng = np.random.default_rng(1)
        for _ in range(4):                     # four benchmarks of 100 questions
            idx = rng.integers(0, m.shape[1], 100)
            s = m[:, idx].mean(1)
            print(" > ".join(names[i] for i in np.argsort(-s)[:4]))
        """)
    out["paired"] = s.run("""
        a, b = m[names.index("clean")], m[names.index("cont0")]
        idx = np.random.default_rng(7).integers(0, len(a), 300)
        r = le.paired_difference(a[idx], b[idx])
        print(f"gap {r['diff']:+.3f}  unpaired SE {r['se_unpaired']:.3f}  "
              f"paired SE {r['se_paired']:.3f}")
        print(f"right only for main {r['only_a']}, only for copy {r['only_b']}"
              f", McNemar p = {r['mcnemar_p']:.4f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    res = compute()
    figs = figures(res)
    snip = _snippets(res)

    sc = dict(zip(res["names"], res["scores"]))
    board = {r["n"]: r for r in res["board"]}
    btop = {r["n"]: r for r in res["board_top"]}
    pc, pl = res["pairs"]["cont0"], res["pairs"]["leaky"]
    c20, c100 = res["clusters"][20], res["clusters"][100]
    conts = [sc[f"cont{i}"] for i in range(5)]
    top_gap = sc["clean"] - max(v for k, v in sc.items() if k != "clean")
    cov = [r["coverage"] for r in res["board"]]
    se_u300 = pc["pool"]["se_unpaired"] * np.sqrt(pc["pool"]["n"] / 300)
    se_p300 = pc["pool"]["se_paired"] * np.sqrt(pc["pool"]["n"] / 300)

    # The spine, asserted rather than trusted.
    assert max(sc, key=sc.get) == "clean" and 0.01 < top_gap < 0.02
    assert 0.64 < min(conts) and max(conts) < 0.67
    assert all(0.93 < c < 0.96 for c in cov)
    assert 0.9 < board[100]["coverage_min"] and board[100]["coverage_max"] < 0.97
    assert 0.4 < board[100]["top_right"] < 0.55
    assert board[2500]["top_right"] > 0.93
    assert btop[100]["order_right"] < 0.01 and btop[2500]["order_right"] < 0.2
    assert 0.25 < btop[100]["pairs_wrong"] < 0.32
    assert abs(pc["pool"]["diff"] - 0.0384) < 1e-4
    assert pc["draws"]["sig_unpaired"] < 0.04
    assert 0.5 < pc["draws"]["sig_paired"] < 0.65
    assert abs(se_p300 / pc["draws"]["sd"] - 1) < 0.1
    assert se_u300 > 1.9 * pc["draws"]["sd"]
    assert pl["pool"]["mcnemar_p"] > 0.05
    assert 0.2 < pl["draws"]["wrong_sign"] < 0.3
    assert c100["empirical"] > 1.9 * c100["naive"]
    assert abs(c100["robust"] / c100["empirical"] - 1) < 0.1
    assert c100["icc"] > 0.85 and c100["mixed"] < 0.15

    post = Post(
        title=f"{SERIES_TAG} 51: A Benchmark Is a Sample of Questions",
        slug="ml-51-benchmark-sample",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=51,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A benchmark score's error bar is correct, and a "
                  "leaderboard of such scores still puts close models in "
                  "the wrong order. Comparing on the same questions, and "
                  "counting questions rather than prompts, is most of the "
                  "repair."),
        summary=(
            f"Ten small arithmetic transformers, graded on a pool of 2,500 "
            f"problems and ranked on benchmarks drawn from it. The models' "
            f"95% intervals covered their pool scores {min(cov):.0%} to "
            f"{max(cov):.0%} of the time on average, but on 100 questions the "
            f"best model "
            f"ranked first in {board[100]['top_right']:.0%} of benchmarks, "
            f"and seven close models came out in their true order in "
            f"{btop[100]['order_right']:.1%}. The main model led a fine-tuned "
            f"copy by {pc['pool']['diff']:.3f}; on 300 questions an unpaired "
            f"test found the gap in {pc['draws']['sig_unpaired']:.1%} of "
            f"benchmarks and the paired test in "
            f"{pc['draws']['sig_paired']:.0%}. Asking 100 questions five ways "
            f"each made 500 rows and a standard error "
            f"{c100['empirical'] / c100['naive']:.1f} times what 500 "
            f"independent rows would give."),
        tags=["machine-learning", "llm-evaluation", "benchmarks",
              "statistics", "evaluation", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Models: ten 804k-parameter arithmetic transformers trained by "
            "this repository: the main model of the school-maths series and "
            "its leaky twin (`data/schoolmath/`), the main model continued "
            "for 600 steps under five seeds, and three snapshots of a "
            "separate 6,000-step run (`data/schoolmath/family/`).",
            "Pool: 500 problems per task, drawn at random from the test "
            "split (multiplication, order of operations, linear equations, "
            "subtraction) and the three-digit carry probe; one context of "
            "twelve training lines per problem (seed 3). Per-question "
            "results are committed in `data/ml_llmeval/runs.json`, written "
            "by `scripts/llm_eval_runs.py`.",
            "Machinery: `standarderror/ml/llmeval.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: McNemar, *Psychometrika* (1947); Dietterich, "
            "\"Approximate statistical tests for comparing supervised "
            "classification learning algorithms\", *Neural Computation* "
            "(1998); Miller, \"Adding error bars to evals\" (Anthropic, "
            "2024); Madaan et al., \"Quantifying variance in evaluation "
            "benchmarks\" (2024); episodes 1 and 44 of this series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("committed weights and per-question results, "
                            "seeded draws; about two minutes of CPU"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "A score from someone else's questions",
        f"""Episode 1 put an error bar on a test score: 360 test digits were a sample, and the score was an estimate. Every episode since has evaluated a model the reader trained, on a test set the reader drew. This arc changes both. It evaluates language models, which most people did not train, on benchmarks they did not write, and asks whether the habits of the series survive the change. The first is the oldest: *the model with the higher benchmark score is the better model*.

The models are the arithmetic transformers of the school-maths series, 804,096 parameters each, trained on generated sums with held-out probes. Tiny models are the right instrument here because every answer can be checked exactly and every model can be graded on thousands of questions in a minute, so each evaluation habit can be compared with the truth. There are ten: the main model; its leaky twin, trained on the same lines in a different order; the main model continued for 600 more steps under five different seeds, the way several teams might fine-tune one base model; and three snapshots of a separate, much shorter run.

The "truth" for this episode is a pool of 2,500 problems, 500 from each of five tasks, and each model's score on all of it. A benchmark is a sample of questions from such a pool.

{snip['pool'].markdown()}

The seven strong models are close: the main model leads the next best by {top_gap:.3f}, and the five fine-tuned copies sit between {min(conts):.3f} and {max(conts):.3f}. Extra training made every copy slightly worse than the model it started from, probably because the main run had ended its learning-rate schedule near zero and the copies restarted at a constant rate. That matters below only because it makes the copies close to one another and a little below the main model.""")

    post.add(
        "Right about each score, wrong about the order",
        f"""Draw a benchmark of 100 questions from the pool and grade all ten models on it. Each model's score comes with the textbook interval, the score plus or minus 1.96 standard errors, and across 2,000 such benchmarks the intervals covered the true pool scores {board[100]['coverage']:.1%} of the time, averaged over the ten models ({board[100]['coverage_min']:.1%} to {board[100]['coverage_max']:.1%} for individual models; the textbook interval undercovers slightly for the weakest, whose scores are near zero). At every size from 50 to 2,500 questions the average stayed between {min(cov):.1%} and {max(cov):.1%}. One score's error bar is, near enough, right.

{snip['board'].markdown()}

The leaderboard is another matter. On 100 questions the best model came first in **{board[100]['top_right']:.0%}** of benchmarks. Among the seven close models, {btop[100]['pairs_wrong']:.0%} of pairs came out in the wrong order and the full order was right in {btop[100]['order_right']:.1%} of benchmarks. Even on a resample the size of the whole pool, 2,500 questions drawn with replacement, the best model came first {board[2500]['top_right']:.0%} of the time and the seven were in true order in {btop[2500]['order_right']:.0%}.

Nothing is wrong with the intervals. A leaderboard asks a different question from the one each interval answers: not "where is this model's score?" but "which of these models is higher?", for every pair at once, and close models' intervals overlap almost entirely. Reading a ranking off overlapping intervals is reading noise.""",
        figures=[figs["f0"]])

    post.add(
        "Compare on the same questions",
        f"""The right question for a pair of models is about their difference, and the difference has its own standard error. The usual mistake, and one of the first that Miller's 2024 guide to error bars for language-model evaluations warns against, is to compute it as if the two scores came from different questions. They do not. Both models answered the same 300 questions, and their errors fall on the same hard ones: the correlation between the main model's right/wrong column and its fine-tuned copy's was {pc['pool']['corr']:.2f}.

{snip['paired'].markdown()}

Pairing uses that. The paired standard error is the spread of the per-question differences, and only the questions where the models disagree contribute; McNemar's 1947 test, in its exact form the binomial test on those disagreements, is the same idea as a test. On the pool the main model beat its fine-tuned copy by {pc['pool']['diff']:.4f}. Across 4,000 benchmarks of 300 questions the measured gap varied with a standard deviation of {pc['draws']['sd']:.4f}. The paired standard error at that size is {se_p300:.4f}, about right; the unpaired one is {se_u300:.4f}, more than twice too large. So the unpaired test found the real gap in {pc['draws']['sig_unpaired']:.1%} of benchmarks and the paired test in **{pc['draws']['sig_paired']:.0%}**.

The leaky twin, {pl['pool']['diff']:.4f} behind the main model on the pool, is a reminder that pairing is not magic. In this episode the pool is the truth, so the twin is known to be behind; but had the 2,500 questions been a sample from a larger population, McNemar's test on all of them would have given p = {pl['pool']['mcnemar_p']:.2f}, and a 300-question benchmark put the twin ahead in {pl['draws']['wrong_sign']:.0%} of draws. Episode 44 met the opposite failure, a t-test on cross-validation folds that found differences between equal algorithms because the folds were not independent. Here the questions are drawn independently, the paired test is the right one, and it can still say "not enough evidence".""",
        figures=[figs["f1"]])

    post.add(
        "Five prompts per question are not five questions",
        f"""Language-model evaluations often ask each question several ways: several prompt templates, several samples at a temperature, several orderings of the options. The rows multiply and the error bar computed from them shrinks. To measure what that buys, the main model answered 600 questions (200 each from multiplication, mixed operations and three-digit carries) five times, each time with a different context in front.

The answers barely depended on the context: {c100['always']:.0%} of questions were answered right all five times and {c100['never']:.0%} wrong all five times, leaving {c100['mixed']:.0%} that the context could change. A benchmark of 100 such questions has 500 rows. Treating the rows as independent gives a standard error of {c100['naive']:.4f}; the actual spread of the score across 4,000 redrawn benchmarks was {c100['empirical']:.4f}, **{c100['empirical'] / c100['naive']:.1f} times** larger. The cluster-robust standard error, which treats each question's five rows as one observation, gave {c100['robust']:.4f}, matching the spread; with 20 questions the numbers were {c20['naive']:.4f}, {c20['robust']:.4f} and {c20['empirical']:.4f}. This is episode 37's lesson for subjects, applied to questions: rows from the same question are not new questions.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        """**A pool, not a population.** The truth here is a fixed pool of 2,500 problems from five tasks. A real benchmark is a sample from a population of questions nobody has written down, and the questions are often chosen rather than drawn, which adds a bias that no error bar describes.

**Exact answers.** Grading was exact match, so every error bar here is pure question sampling. Graders that disagree with each other add their own variance.

**Models of one family.** All ten models were trained on the same lines, and six are the main model or copies of it, which makes their errors more correlated than those of models from different labs. The more correlated the errors, the more pairing gains; for unrelated models the gain is smaller but the paired test is still the right one.

**Questions drawn independently.** Benchmarks often contain families of near-duplicate questions written from one template, which cluster in the same way as the repeated prompts here.""")

    post.add(
        "What to keep",
        f"""1. The models' 95% intervals covered their true scores about 95% of the time on average at every benchmark size, and on 100 questions the best of ten models still ranked first in only {board[100]['top_right']:.0%} of benchmarks.

2. On 300 shared questions the paired test detected a real 0.038 gap in {pc['draws']['sig_paired']:.0%} of benchmarks and the unpaired one in {pc['draws']['sig_unpaired']:.1%}; the paired standard error matched the actual spread.

3. Five prompts per question gave a standard error {c100['empirical'] / c100['naive']:.1f} times the naive one, and the cluster-robust error matched.

4. Report a comparison as a difference with a paired interval, cluster rows by question, and treat leaderboard positions inside each other's intervals as ties.""")

    post.add(
        "Exercise",
        """Take two models you have compared on a benchmark and export the per-question right/wrong for both. Count the questions only one of them got right, in each direction.

The comparison rests on those disagreements, and nothing else. If there are thirty of them, split 18 to 12, the honest summary is that the benchmark cannot tell the models apart, whatever the two headline scores say.""")

    post.add(
        "Next",
        """A benchmark is a sample of questions, and the error bars assume the model met them for the first time. Episode 52 trains copies of the main model that have seen some of the benchmark's answers, and asks how much it shows in the score, and whether anyone looking only at the model could tell.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
