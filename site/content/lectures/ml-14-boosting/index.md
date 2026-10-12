---
title: "ML 14: Boosting Overfits Its Probabilities Long Before Its Accuracy"
date: 2026-10-12
slug: "ml-14-boosting"
draft: false
description: "Whether boosting overfits depends on what you measure. Over 2,000 rounds its test error never turns up; its test log loss turns after a few dozen, and shrinkage only postpones the turn."
author: "Jongha Jeon"
tags: ["machine-learning", "data-science", "statistics", "gradient-boosting", "calibration", "lectures"]
series: ["Machine Learning, Taught Through What Breaks"]
weight: 14
images: ["ml114-hero.png"]
---

Disclosure: this post was written with the assistance of an AI system (Claude), which wrote the analysis code, ran the experiments and drafted the text. The topic, the constraints, the data choices and the final review are the author's.

*Gradient-boosted trees on the breast-cancer data, ten half splits, 2,000 rounds. With clean labels and a learning rate of 0.1 the test error falls to 0.046 and never rises. The test log loss is lowest after about 46 rounds (0.161) and 2.6 times worse at 2,000. Each tenfold cut in the learning rate moves the best round roughly tenfold later (3.5, 46, 400) without lowering the best log loss. With 10% of training labels flipped the error rises by 0.017 and the log loss 3.3-fold.*

Episode 14 of *Machine Learning, Taught Through What Breaks*. The syllabus and the other episodes: https://jongha-jeon-dev.github.io/standarderror/lectures/

## Run it long enough

Every new tree in gradient boosting is fitted to what the current model still gets wrong. Run that long enough and the model will fit the training data perfectly, noise included, so the standard warning is that boosting overfits if left running and the number of rounds must be tuned or stopped early.

Whether it overfits depends on what "fits" means. A classifier makes two kinds of output: a decision, scored by the error rate, and a probability, scored by log loss. Here is one boosted model, depth-3 trees at the common learning rate of 0.1, scored both ways on held-out tumours after each checkpoint, averaged over ten half splits:

```python
from standarderror.ml import trees as tr

# depth-3 trees, learning rate 0.1, 10 half splits, clean labels
c = tr.boost_curves(lr=0.1, flip=0.0)
for r, e, ll in zip(c["rounds"], c["error"], c["log_loss"]):
    print(f"round {r:>5,}  test error {e:.4f}  test log loss {ll:.3f}")
```

```text
round     1  test error 0.3719  test log loss 0.582
round     3  test error 0.0691  test log loss 0.470
round    10  test error 0.0561  test log loss 0.275
round    30  test error 0.0565  test log loss 0.161
round   100  test error 0.0498  test log loss 0.190
round   300  test error 0.0474  test log loss 0.348
round 1,000  test error 0.0463  test log loss 0.416
round 2,000  test error 0.0463  test log loss 0.416
```

By the error rate there is nothing to stop: it falls to 0.046 and is still at its lowest after 2,000 rounds. By log loss the model peaked long ago. The log loss is lowest at about **46 rounds (0.161)** and is 0.416 by the end — 2.6 times worse.

The two numbers disagree because they look at different things. The error counts which side of 0.5 each prediction falls. Log loss charges for confidence: a tumour predicted at 0.999 benign that turns out malignant costs almost seven nats. As the rounds go on, the model pushes every training probability towards 0 or 1. On test tumours it mostly keeps the same answers, so the error stands still, but the few it gets wrong it now gets wrong with near certainty.

![Three U-shaped curves, shifted right as the learning rate falls, each reaching about the same minimum before rising.](ml114-f0-logloss.png)

*At learning rate 0.1 the log loss bottoms at 0.161 after 30 rounds and reaches **0.416** by 2,000. A smaller rate reaches the same bottom later; it does not lower it.*

## Shrinkage slows it; it does not stop it

The usual remedy is shrinkage: scale each tree's contribution by a small learning rate so the model approaches the training data slowly. It works in the sense Friedman intended — the curves above are smoother and their bottoms wider — but look at where the bottoms are:

- learning rate 1: best log loss 0.203, after about 3.5 rounds;
- learning rate 0.1: best 0.161, after about 46;
- learning rate 0.01: best 0.162, after about 400.

Each tenfold cut in the learning rate moves the best round about tenfold later, and below 1 it does not lower the best log loss at all: 0.161 against 0.162. A smaller learning rate buys a wider window in which stopping is safe and costs proportionally more trees. It does not make running forever safe. At 2,000 rounds the learning-rate-0.01 model is already past its best, at 0.272.

That is why early stopping is part of every serious boosting library, and why what it monitors matters. A validation set scored by accuracy, here, would never trigger it, and the model shipped would be the one with the worst probabilities of any checkpoint after the first few dozen.

![Solid lines falling and flattening near 0.04 to 0.05; dashed lines higher, near 0.07 to 0.10, rising slightly at the longest runs.](ml114-f1-error.png)

*With clean labels the error at 2,000 rounds is the lowest it has been. With flipped labels it rises from 0.071 to 0.088 — under two points, while the log loss 3.3-folds.*

## What a stopping rule picks

Early stopping takes the round at which a validation metric is best. Which metric it watches decides which model you ship. On the clean runs at learning rate 0.1, watching the error rate picks the last checkpoint, 2,000 rounds, because the error is still at its lowest there. That model's test log loss is 0.416. Watching log loss picks a model from around round 46, with test log loss 0.161 and a test error of 0.056 at the nearby checkpoint of 30 rounds — 0.010 worse than the long run.

So the trade is real but lopsided: about a point of accuracy against a 2.6-fold difference in the quality of the probabilities. If the output is a decision at a fixed threshold, the long run is the better model. If the output is a probability — and most deployed classifiers are read as risks, ranked, or combined with costs — the short run is better by a wide margin, and only a stopping rule on log loss will find it.

Why does the error keep improving while confidence runs away? Schapire and colleagues' margin explanation is the usual answer: later rounds mostly push correctly classified training points further from the boundary, which changes probabilities a great deal and decisions very little. On test points near the boundary the boundary itself barely moves; on test points the model gets wrong, the pushed-out probabilities make each mistake more expensive.

## Noisy labels

Clean labels flatter the error rate: with nothing to memorise but the truth, fitting the training data harder cannot cost many decisions. Flip 10% of the training labels at random and boosting has real noise to chase:

```python
c = tr.boost_curves(lr=0.1, flip=0.1)   # 10% of labels flipped
for r, e, ll in list(zip(c["rounds"], c["error"], c["log_loss"]))[3:]:
    print(f"round {r:>5,}  test error {e:.4f}  test log loss {ll:.3f}")
```

```text
round    30  test error 0.0709  test log loss 0.240
round   100  test error 0.0804  test log loss 0.211
round   300  test error 0.0856  test log loss 0.248
round 1,000  test error 0.0891  test log loss 0.531
round 2,000  test error 0.0881  test log loss 0.690
```

Now the error turns up as well, from 0.071 at its best to 0.088 — less than two points over nearly two thousand extra rounds. The log loss goes from 0.211 to **0.690**, more than three times worse. The flipped training labels are fitted as faithfully as the true ones, and every test tumour that resembles a flipped training tumour inherits a confident wrong answer.

Mease and Wyner made the accuracy half of this point in 2008: boosted classifiers often keep improving in error long after the statistical view says they should be overfitting. The probability half is the one that matters whenever the output is used as a risk — a credit default probability, a diagnosis likelihood, a click-through rate — because there the model's confidence is the product.

![A six-row table: the best round grows about tenfold per tenfold drop in learning rate; log loss at 2,000 rounds is far above its best, error at 2,000 rounds close to its best.](ml114-f2-table.png)

*Read the last two columns and boosting barely overfits. Read the two before them and it overfits badly. Both are true of the same models.*

## Where this breaks

**One dataset.** Breast cancer is easy, with a test error under 5%. On harder data, where many test rows sit near the boundary, error turns up sooner and further; the ordering — log loss first, error later — is what generalises.

**Probabilities can be repaired afterwards.** Calibrating an overfit booster on held-out data (Platt scaling, isotonic regression) recovers much of the lost log loss, as Niculescu-Mizil and Caruana showed. It needs a held-out set, and it repairs the symptom, not the training.

**Checkpoints, not every round.** The curves are sampled at eight round counts. The best round quoted is computed from every round, per split, and its median is reported.

**Other boosters regularise more by default.** Libraries with leaf-weight penalties, row subsampling and minimum leaf sizes reach the same place more slowly. scikit-learn's classic implementation, with its defaults, is the plainest version of the algorithm.

## What to keep

1. With clean labels, test error of gradient boosting never turned up in 2,000 rounds: 0.046 at the end.

2. Test log loss was lowest after about 46 rounds and 2.6 times worse at 2,000. The model's decisions held; its confidence ran away.

3. Each tenfold cut in the learning rate moved the best round about tenfold later and left the best log loss where it was. Shrinkage widens the window; it does not remove the need to stop.

4. With 10% label noise the error rose by under two points and the log loss more than tripled.

5. Stop boosting on the metric the output will be judged by. If the probabilities are used, that is log loss or a proper score, not accuracy.

## Exercise

Take a boosted model you have in use and plot its staged validation error and staged validation log loss against the number of rounds on the same axes. Mark the round your early stopping chose and the round each curve is lowest.

If the two curves bottom out far apart, find out which one your stopping rule watched. If the model's probabilities are used downstream, refit with stopping on log loss and compare the calibration of the two models on held-out data.

## Next

Episode 15 closes Arc III with the one thing no number of trees or rounds can fix: what a tree ensemble predicts outside the range of the data it was trained on.

---

### Data

- Breast Cancer Wisconsin (Diagnostic), as bundled with scikit-learn, CC BY 4.0; 10 stratified half splits; for the noisy runs, 10% of training labels flipped at random, test labels untouched.
- Machinery: `standarderror/ml/trees.py`, tested in `tests/test_ml.py`; scikit-learn's GradientBoostingClassifier with depth-3 trees and staged predictions.
- Where this stops: Friedman, "Greedy function approximation: a gradient boosting machine", *Annals of Statistics* (2001), for shrinkage; Mease and Wyner, "Evidence contrary to the statistical view of boosting", *JMLR* (2008); Niculescu-Mizil and Caruana, "Predicting good probabilities with supervised learning", *ICML* (2005); Schapire, Freund, Bartlett and Lee, "Boosting the margin", *Annals of Statistics* (1998).

### Reproducibility

- **environment**: standarderror=0.1.0, python=3.13.16, numpy=2.4.4
- **code blocks**: executed at build time; the values the prose quotes are pinned, so drift fails the build
- **determinism**: splits from seeds 0 to 9, label flips from seeds 1,000 to 1,009, boosting with random_state equal to the split seed

Code: <https://github.com/jongha-jeon-dev/standarderror>
