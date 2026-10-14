"""ML 52: A Leaked Benchmark Shows in the Set, Not in the Item.

The fifty-second episode of *Machine Learning, Taught Through What Breaks*,
in Arc XI. "If a model has memorised a benchmark, its likelihood on the
benchmark's items will show it." Measured by continuing the repository's
main arithmetic model with 300 benchmark problems inserted into its training
stream between 0 and 32 times:

* The score on the leaked problems rose from a control mean of 0.551 to
  0.630 at two insertions and 0.800 at thirty-two, while the score on 600
  fresh problems of the same kinds stayed between 0.51 and 0.55 in every
  run: the benchmark moved, the ability did not.
* Item by item, the contaminated model's own likelihood told leaked from
  fresh problems only weakly (AUC 0.62 at thirty-two
  insertions; at 5% false flags it caught 3% of leaked problems). Even with
  the uncontaminated model as a reference it caught 24%.
* As a set, the leak showed: compared with fresh problems from the same
  generator, the leaked set was more likely and more often right, well
  outside the range of five uncontaminated controls, from two insertions on.

Run: `standarderror run ml152_contamination --publish`
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


def compute() -> dict:
    r = le.runs()
    c = le.contamination(r)
    # Re-grade the starting model live, so the committed per-item results are
    # tied to the committed weights rather than trusted.
    bench, fresh = le.leak_sets()
    base = sm.load()
    live_b = le.ok(base, bench, fills=le.ar.fillers(len(bench), seed=5))
    live_f = le.ok(base, fresh, fills=le.ar.fillers(len(fresh), seed=6))
    c["runs"]["base"]["bench_mul"] = float(live_b[:le.LEAK_N].mean())
    c["runs"]["base"]["bench_order"] = float(live_b[le.LEAK_N:].mean())
    c["live_matches"] = bool(
        (live_b == np.array(r["doses"]["base"]["bench_ok"])).all()
        and (live_f == np.array(r["doses"]["base"]["fresh_ok"])).all())
    return c


def _by_dose(c):
    out: dict = {}
    for v in c["runs"].values():
        if v["dose"] is not None:
            out.setdefault(v["dose"], []).append(v)
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ figures

def _dose_plot(ax, m, c):
    doses = _by_dose(c)
    xs = {d: i for i, d in enumerate(doses)}
    for d, vs in doses.items():
        for j, v in enumerate(vs):
            x = xs[d] + (j - (len(vs) - 1) / 2) * 0.12
            ax.scatter([x], [v["bench"]], color=m.series[1], s=30, zorder=3,
                       label="leaked problems" if (d, j) == (0, 0) else None)
            ax.scatter([x], [v["fresh"]], color=m.series[0], s=30,
                       marker="s", zorder=3,
                       label="fresh problems" if (d, j) == (0, 0) else None)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels([str(d) for d in doses])
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)


def _detect_plot(ax, m, c):
    doses = _by_dose(c)
    xs = list(range(len(doses)))
    for k, (key, lab) in enumerate((("auc_loss", "item AUC, model alone"),
                                    ("auc_ref", "item AUC, with reference"))):
        ax.plot(xs, [np.mean([v[key] for v in vs]) for vs in doses.values()],
                color=m.series[k], marker="o", label=lab)
    ax.axhline(0.5, color=m.ink, ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in doses])
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)


def figures(c: dict) -> dict:
    out: dict = {}
    doses = _by_dose(c)
    top = max(doses)
    out["f0"] = charts.diagram(
        lambda ax, m: _dose_plot(ax, m, c),
        title="The leaked problems got easier; nothing else did",
        subtitle=("Accuracy of the main model after 600 more training steps "
                  "with 300 benchmark problems inserted into the stream k "
                  "times, on those problems and on 600 fresh ones of the same "
                  "kinds. At k = 0 nothing was leaked; five seeds."),
        xlabel="times each leaked problem was inserted (k)",
        ylabel="exact-match accuracy",
        figsize=(7.4, 4.3),
        source="scripts/llm_eval_runs.py; results in data/ml_llmeval.",
        alt=("Two rows of dots against dose. Fresh-problem accuracy stays "
             "between 0.51 and 0.55 at every dose. Leaked-problem accuracy "
             "is 0.50 to 0.58 at dose 0 and rises above the controls from "
             "dose 2."),
        caption=(f"Five uncontaminated runs scored "
                 f"{min(v['bench'] for v in doses[0]):.3f} to "
                 f"{max(v['bench'] for v in doses[0]):.3f} on the benchmark; "
                 f"inserted {top} times, "
                 f"**{np.mean([v['bench'] for v in doses[top]]):.3f}**."),
        path=str(IMG / f"ml152-f0-dose.{EXT}"))[0]
    out["f1"] = charts.diagram(
        lambda ax, m: _detect_plot(ax, m, c),
        title="Telling leaked problems from fresh ones, one at a time",
        subtitle=("Area under the ROC curve for flagging the 300 leaked "
                  "problems among 900, by the contaminated model's "
                  "log-likelihood of the answer (alone), or by how much it "
                  "rose from the uncontaminated model's (with reference)."),
        xlabel="times each leaked problem was inserted (k)", ylabel="AUC",
        figsize=(7.4, 4.0),
        source="scripts/llm_eval_runs.py; results in data/ml_llmeval.",
        alt=("Two lines against dose, both starting at 0.5. The model-alone "
             "line rises slowly; the reference line rises faster."),
        caption=(f"At k = {top}: "
                 f"{np.mean([v['auc_loss'] for v in doses[top]]):.3f} alone, "
                 f"{np.mean([v['auc_ref'] for v in doses[top]]):.3f} with the "
                 f"reference model."),
        path=str(IMG / f"ml152-f1-detect.{EXT}"))[0]

    body = []
    for d, vs in doses.items():
        for v in vs:
            body.append([str(d), str(v["seed"]), f"{v['bench']:.3f}",
                         f"{v['fresh']:.3f}", f"{v['gap']:+.3f}",
                         f"{v['lp_bench'] - v['lp_fresh']:+.3f}",
                         f"{v['lp_t']:+.2f}", f"{v['auc_loss']:.3f}",
                         f"{v['auc_ref']:.3f}"])
    out["f2"] = charts.table_image(
        body,
        header=["k", "seed", "leaked acc.", "fresh acc.", "acc. gap",
                "log-lik. gap", "t", "AUC alone", "AUC ref."],
        title="Every run, set by set and item by item",
        subtitle=("Gaps are leaked minus fresh. 't' is Welch's t for the "
                  "difference in mean log-likelihood per item between the "
                  "two sets."),
        source="scripts/llm_eval_runs.py; results in data/ml_llmeval.",
        alt=("A table of every fine-tuning run. The five k = 0 rows have "
             "t between -0.3 and 0.8; rows from k = 2 on have t above 2."),
        caption="k = 0 rows are the controls: the same extra training, nothing leaked.",
        align="rrrrrrrrr",
        path=str(IMG / f"ml152-f2-table.{EXT}"))[0]
    out["hero"] = _hero(c)
    return out


def _hero(c: dict):
    doses = _by_dose(c)
    top = max(doses)
    hi = np.mean([v["bench"] for v in doses[top]])
    lo = c["control_bench_mean"]
    auc = np.mean([v["auc_loss"] for v in doses[top]])
    t_top = np.mean([v["lp_t"] for v in doses[top]])

    def p1(panel, m):
        panel.bar([0, 1], [lo, hi], color=[m.series[0], m.series[1]])
        panel.set_ylim(0.4, 0.75)

    def p2(panel, m):
        panel.bar([0, 1], [0.5, auc], color=[m.series[3], m.series[1]])
        panel.set_ylim(0.4, 1)

    def p3(panel, m):
        panel.bar([0, 1], [max(abs(v["lp_t"]) for v in doses[0]), t_top],
                  color=[m.series[0], m.series[2]])

    return charts.lecture_hero(
        series=SERIES_TAG, episode=52,
        headline="Seen before",
        panels=[(p1, f"+{hi - lo:.3f}", f"score, {top} insertions"),
                (p2, f"{auc:.2f}", "AUC, item by item"),
                (p3, f"t = {t_top:.1f}", "leaked set vs fresh set")],
        note=("Leaked answers raise a benchmark score without raising the "
              "ability it measures. One item at a time the leak is nearly "
              "invisible; against a fresh set from the same source it is "
              "not."),
        alt=("Three hand-drawn frames: a bar and a taller one; a bar at "
             "chance and one barely above; a short bar and a tall one."),
        path=str(IMG / f"ml152-hero.{EXT}"))[0]


# ----------------------------------------------------------------- snippets

def _snippets(c: dict) -> dict:
    s = Session()
    out = {}
    out["leak"] = s.run("""
        from standarderror.ml import llmeval as le

        bench, fresh = le.leak_sets()          # 300 to leak, 600 never seen
        print(len(bench), "leaked, e.g.", [p.line.strip() for p in bench[:3]])
        print(len(fresh), "fresh,  e.g.", [p.line.strip() for p in fresh[:3]])
        # le.finetune(model, [p.line for p in bench], repeats=k, seed=s) is
        # what scripts/llm_eval_runs.py ran for every k and seed below.
        """)
    out["runs"] = s.run("""
        c = le.contamination()                 # the committed per-item results
        for key, v in c["runs"].items():
            if key != "base":
                print(f"k={v['dose']:2d} seed {v['seed']}  leaked "
                      f"{v['bench']:.3f}  fresh {v['fresh']:.3f}  "
                      f"item AUC {v['auc_loss']:.3f}")
        """)
    return out


# -------------------------------------------------------------------- post

def build() -> Post:
    IMG.mkdir(parents=True, exist_ok=True)
    c = compute()
    figs = figures(c)
    snip = _snippets(c)

    doses = _by_dose(c)
    top = max(doses)
    ctrl = doses[0]
    cb = [v["bench"] for v in ctrl]
    allf = [v["fresh"] for vs in doses.values() for v in vs]
    ct = [v["lp_t"] for v in ctrl]
    cg = [v["gap"] for v in ctrl]
    mean = {d: {k: float(np.mean([v[k] for v in vs]))
                for k in ("bench", "fresh", "auc_loss", "auc_ref", "lp_t",
                          "gap", "tpr5_loss", "tpr5_ref")}
            for d, vs in doses.items()}
    d8 = [v["bench"] for v in doses[8]]
    base = c["runs"]["base"]
    rt = doses[top][0]
    n_train = len(le.cur.corpus("train", per_width=20_000, seed=0))
    first_t = min(d for d in doses if d > 0
                  and all(v["lp_t"] > max(ct) + 1 for v in doses[d]))

    # The spine, asserted rather than trusted.
    assert c["live_matches"]
    assert len(ctrl) == 5 and len(doses[8]) == 3
    assert max(cb) - min(cb) > 0.06
    assert max(allf) - min(allf) < 0.04
    assert mean[1]["bench"] < max(cb)
    assert all(mean[d]["bench"] > max(cb) for d in doses if d >= 2)
    assert mean[top]["bench"] - c["control_bench_mean"] > 0.06
    assert mean[top]["auc_loss"] < 0.63 and mean[top]["auc_ref"] < 0.76
    assert mean[top]["tpr5_loss"] < 0.06 and mean[top]["tpr5_ref"] < 0.3
    assert mean[top]["bench"] - c["control_bench_mean"] > 0.2
    assert max(abs(t) for t in ct) < 1.0 and first_t == 2
    assert all(v["lp_t"] > 2 for d in doses if d >= 2 for v in doses[d])
    assert max(d8) - min(d8) > 0.01
    assert base["bench"] > base["fresh"] and 168_000 < n_train < 170_000
    assert all(v["bench"] < base["bench"] for v in ctrl)
    assert all(v["fresh"] < base["fresh"] for vs in doses.values() for v in vs)
    assert 0 < mean[2]["bench"] - base["bench"] < 0.05
    assert rt["lp_sd"] > 2 * rt["shift_mean"] and ctrl[0]["shift_sd"] > 0.5

    post = Post(
        title=(f"{SERIES_TAG} 52: A Leaked Benchmark Shows in the Set, Not "
               f"in the Item"),
        slug="ml-52-contamination",
        section="lectures",
        series=SERIES,
        series_tag=SERIES_TAG,
        episode=52,
        date=POST_DATE,
        requires_baseline=False,
        subtitle=("A model trained on a benchmark's answers scored higher on "
                  "them and no higher on anything else. Its own likelihoods "
                  "could not say which problems had leaked; a fresh set from "
                  "the same source could say that some had."),
        summary=(
            f"The repository's main arithmetic model was trained for 600 "
            f"more steps with 300 benchmark problems inserted into its "
            f"stream k times, for k from 0 to {top}. Five runs with nothing "
            f"leaked scored {min(cb):.3f} to {max(cb):.3f} on the benchmark; "
            f"with each problem inserted {top} times, "
            f"{mean[top]['bench']:.3f}. On 600 fresh problems of the same "
            f"kinds every run scored between {min(allf):.3f} and "
            f"{max(allf):.3f}. The contaminated model's likelihood flagged "
            f"leaked problems with an AUC of {mean[top]['auc_loss']:.3f}, "
            f"{mean[top]['auc_ref']:.3f} with the uncontaminated model as a "
            f"reference. Compared as sets, leaked against fresh, the gap in "
            f"mean log-likelihood had t above 2 in every run from k = 2 on, "
            f"and between {min(ct):.2f} and {max(ct):.2f} in the five "
            f"controls."),
        tags=["machine-learning", "llm-evaluation", "contamination",
              "data-leakage", "language-models", "lectures"],
        author=se.SETTINGS.author,
        code_url=se.SETTINGS.code_repo_url,
        data_sources=[
            "Model: the main arithmetic transformer of the school-maths "
            "series (`data/schoolmath/schoolmath.pt`), continued for 600 "
            "AdamW steps at a constant 3e-4 on the training stream, with "
            "the benchmark's lines inserted k = 0, 1, 2, 4, 8, 16 or 32 "
            "times at random positions; seeds 0-4 at k = 0 and 0-2 at k = 8.",
            "Benchmark: 150 multiplication and 150 order-of-operations "
            "problems from the test split; fresh set: 300 more of each, "
            "disjoint. Per-item results are committed in "
            "`data/ml_llmeval/runs.json`, written by "
            "`scripts/llm_eval_runs.py`; the starting model's are re-graded "
            "live at build time.",
            "Machinery: `standarderror/ml/llmeval.py`, tested in "
            "`tests/test_ml.py`.",
            "Where this stops: Yeom, Giacomelli, Fredrikson and Jha, "
            "\"Privacy risk in machine learning: analyzing the connection to "
            "overfitting\", *CSF* (2018); Carlini et "
            "al., \"Membership inference attacks from first principles\", "
            "*IEEE S&P* (2022); Shi et al., \"Detecting pretraining data "
            "from large language models\", *ICLR* (2024); Zhang et al., \"A "
            "careful examination of large language model performance on "
            "grade school arithmetic\" (2024); episodes 3 and 45 of this "
            "series.",
        ],
        reproducibility={
            "environment": (f"standarderror={se.__version__}, "
                            f"python={platform.python_version()}, "
                            f"numpy={np.__version__}"),
            "code blocks": ("executed at build time; the values the prose "
                            "quotes are pinned, so drift fails the build"),
            "determinism": ("the fine-tuning runs take about an hour and are "
                            "committed as per-item results; the build "
                            "re-grades the starting model and reads the rest"),
        },
    )
    post.hero = figs["hero"]

    post.add(
        "Leaking the answers on purpose",
        f"""Language models are trained on large scrapes of the web, and benchmarks live on the web. So a model may have seen some test questions, and their answers, during training. Nobody evaluating it can rerun the training without them. What they can do is look at the model, and the hope behind most contamination checks is this episode's sentence: *if a model has memorised a benchmark, its likelihood on the benchmark's items will show it*.

To test it, contamination has to be manufactured with a known dose. The benchmark is 300 problems from the main arithmetic model's test split, 150 multiplications and 150 mixed-operation problems, on which the starting model gets {base['bench_mul']:.2f} and {base['bench_order']:.2f} right, so there is room to improve. A fresh set of 600 problems of the same kinds is drawn alongside and never leaked. Then the model is trained for 600 more steps, roughly one more pass over its training data, with the benchmark's lines inserted into the stream k times each, for k from 0 to {top}. The runs with k = 0 matter most: they get the same extra training with nothing leaked, and five seeds of them show how much a run varies by chance. Training reads random 64-character windows rather than whole lines in order, so an insertion is an opportunity to be seen, not a guaranteed reading: at one insertion a fair share of leaked lines is probably never seen whole.

{snip['leak'].markdown()}

The training stream has {n_train:,} lines, so at two insertions the leaked lines are {300 * 2 / (n_train + 600):.2%} of it, and even at {top} they are {300 * top / (n_train + 300 * top):.1%}. The small doses are the realistic case: a benchmark is a speck in a training corpus, seen a few times through copies, not drilled.""")

    post.add(
        "The score rose; the ability did not",
        f"""{snip['runs'].markdown()}

Inserted once, the leaked problems made no measurable difference: the score was {mean[1]['bench']:.3f}, inside the controls' range of {min(cb):.3f} to {max(cb):.3f}. From two insertions on, every run scored above every control, and at {top} insertions the benchmark score was {mean[top]['bench']:.3f}, {mean[top]['bench'] - c['control_bench_mean']:+.3f} over the controls' mean. The fresh problems, meanwhile, stayed between {min(allf):.3f} and {max(allf):.3f} in every run, leaked or not. The benchmark reported an improvement that existed only on the benchmark.

One comparison is less flattering. Every fine-tuned run, leaked or not, lost ground against the starting model, which scored {base['bench']:.3f} on the benchmark and {base['fresh']:.3f} on the fresh set. Measured against the starting model rather than against the controls, the two-insertion run gained only {mean[2]['bench'] - base['bench']:+.3f} on the benchmark, while losing {base['fresh'] - mean[2]['fresh']:.3f} on fresh problems. An outside evaluator comparing a model with its predecessor sees the net of both effects.

The spread of the controls deserves its own sentence. Five runs that differed only in their random seed scored between {min(cb):.3f} and {max(cb):.3f} on the same 300 problems, a range of {max(cb) - min(cb):.3f}; the three runs at k = 8 spanned {min(d8):.3f} to {max(d8):.3f}. At one insertion, a comparison of one contaminated run with one clean one would have been a coin toss. This is episode 51's lesson again, in the training direction: a run is a sample too.""",
        figures=[figs["f0"]])

    post.add(
        "Item by item, the model could not say which",
        f"""The usual detector asks whether the model finds a particular item suspiciously likely. In privacy research this is membership inference, and its simplest version, analysed by Yeom and colleagues in 2018, thresholds the loss: a model's loss on its training examples is lower than on others. Here, ranking the 900 problems by the contaminated model's log-likelihood of the right answer separated leaked from fresh with an AUC of {mean[top]['auc_loss']:.3f} at k = {top}, and {mean[2]['auc_loss']:.3f} at k = 2, against 0.5 for a coin. Set to flag 5% of the fresh problems, it flagged {mean[top]['tpr5_loss']:.1%} of the leaked ones at k = {top}: no better than flagging at random.

The problem is not that the leak left no trace. It is that problems differ from each other in how likely their answers are by far more than the leak changed them: at k = {top}, per-problem log-likelihoods had a standard deviation of {rt['lp_sd']:.2f}, while the leak raised the leaked set's average by {rt['shift_mean']:.2f} relative to the fresh one. An easy multiplication's answer is likely whether or not it leaked, a hard mixed-operation answer unlikely. Carlini and colleagues' remedy in 2022 was to compare each item's likelihood with that under reference models that did not see it. With the uncontaminated model as the reference, flagging the problems whose likelihood rose most, the AUC was {mean[top]['auc_ref']:.3f} at k = {top}, and at the same 5% false-flag rate it caught {mean[top]['tpr5_ref']:.0%} of the leaked problems: better, and still far from a detector that could name the leaked items, because extra training moves each problem's likelihood by its own amount: even in the controls the change from the starting model had a standard deviation of {ctrl[0]['shift_sd']:.2f} per problem, against {rt['shift_sd']:.2f} at k = {top}. And a reference model is exactly what an outside evaluator does not have. Methods built for large models, such as Shi and colleagues' Min-K% probability, refine the per-item score but face the same item-to-item variation.""",
        figures=[figs["f1"]])

    post.add(
        "As a set, it could",
        f"""The question an evaluator needs answered is usually not "which items leaked?" but "is this benchmark's score inflated?", and that question has more data behind it. Compare the benchmark as a whole with a fresh set from the same source: the same generator, the same tasks, problems that cannot have leaked because they were written afterwards or never published.

On mean log-likelihood per problem, leaked set minus fresh set, the five controls gave Welch t-statistics between {min(ct):.2f} and {max(ct):.2f}. Every contaminated run from k = 2 on gave more than 2, outside the controls' range though not far outside it at k = 2 and 4 ({mean[2]['lp_t']:.2f} and {mean[4]['lp_t']:.2f}), and at k = {top} it was {mean[top]['lp_t']:.1f}. Accuracy told the same story with more noise: the gap between the leaked and fresh sets was {min(cg):+.3f} to {max(cg):+.3f} in the controls and {mean[top]['gap']:+.3f} on average at k = {top}. Neither says which problems leaked. Both say the score on this benchmark is not the score on the task.

This is what Zhang and colleagues did for grade-school arithmetic in 2024: they wrote new problems in the style of a popular benchmark, GSM8k, and compared models' accuracy on the two, finding clear drops for some model families and little for others. It needs no access to training data and no reference model, only the discipline to keep some problems unpublished, which episode 45 asked of test sets in general.""",
        figures=[figs["f2"]])

    post.add(
        "Where this breaks",
        """**Verbatim contamination.** Every leaked line was the exact problem and answer. Real contamination includes paraphrases, translations and solutions without the question, which raise scores while leaving even less trace in the likelihood of the exact item.

**A small model and a short exposure.** A large model memorises single examples more readily, so its per-item signal may be stronger, and a benchmark seen across many training epochs leaves more trace than one inserted into a single extra pass.

**Arithmetic answers.** The answers are a few characters long, which gives a per-item likelihood little to work with. Longer answers carry more evidence per item, which helps per-item detectors.

**A fresh set that matches.** The set-level test is only as good as the match between the fresh problems and the benchmark. Before any training, the leaked set here was already slightly easier than the fresh set, which the controls absorb; an evaluator without controls needs the fresh set to be drawn the way the benchmark was.""")

    post.add(
        "What to keep",
        f"""1. With each benchmark problem inserted into training {top} times, the benchmark score rose from a control mean of {c['control_bench_mean']:.3f} to {mean[top]['bench']:.3f}; accuracy on fresh problems of the same kinds did not move outside {min(allf):.3f}-{max(allf):.3f}.

2. At {top} insertions, the contaminated model's likelihood identified leaked problems with an AUC of {mean[top]['auc_loss']:.3f}, catching {mean[top]['tpr5_loss']:.0%} of them at 5% false flags; with the clean model as a reference, {mean[top]['auc_ref']:.3f} and {mean[top]['tpr5_ref']:.0%}.

3. Compared as sets with fresh problems from the same source, every contaminated run from two insertions on stood out from all five controls.

4. Test for contamination by keeping a fresh set the model cannot have seen and comparing whole sets, and do not read a clean per-item scan as evidence that nothing leaked.""")

    post.add(
        "Exercise",
        """For a benchmark you report, write or generate fifty problems in its style that have never been published. Score your model on them and on fifty of the benchmark's.

A gap bigger than the two sets' sampling error is a reason to stop reporting the benchmark number on its own. No gap is weaker evidence than it looks unless the new problems really match the old ones in difficulty, so check that on a model that cannot have seen either.""")

    post.add(
        "Next",
        """A benchmark is a sample of questions, and contamination is a property of the questions. The way each question is put to the model is a property of the evaluation, and episode 53 asks the same models the same questions with different text in front of them.""")

    return post


if __name__ == "__main__":
    print(build().body_markdown()[:1500])
