## I | Reading a Score

Every later part of the book compares numbers, so it starts with what a number is. A test accuracy is an estimate with an error bar, often wider than the gap it is used to declare, and two models with the same score can disagree on many rows. Cross-validation estimates the average error of a procedure, not of the model in hand, and its folds are neither independent nor always exchangeable. Leaks into a score range from nothing to a third of the scale. The best score of a search reports the search's luck, and a learning curve fitted early promises more than more data delivers. Each of these sentences is close enough to true to be repeated, and wrong enough to change a decision.

## II | Models, Pushed

The models practitioners reach for first have the most precisely stated textbook properties, which makes them the cleanest place to see those properties fail. Linear models depend on stopping rules, units and penalties nobody chose on purpose. A single tree's structure changes when the rows are resampled while its accuracy does not, and no tree ensemble predicts beyond the values it has seen. Boosting overfits its probabilities long before its accuracy, and averaging and stacking beat the average member rather than the best one. The settings that control a network's training are less separate and less exact than their descriptions, and the classical intuitions about overfitting meet networks that can fit anything.

## III | From Probabilities to Decisions

A model's output becomes useful when someone acts on it. This part follows the predicted probability from the model to the decision: how resampling, naive independence and recalibration change it; why the right threshold is a cost ratio, and the model with the best AUC may never be the cheapest; and what changes when the decision acts on the world, where the features a model relies on need not be levers, the highest risk is not the largest effect, and a rule that chooses its own data learns its own choices back.

## IV | Reading What a Model Learned

Explanation methods answer precise questions, and those questions are often not the ones being asked. Impurity importance rewards features with many places to split; permutation importance measures the fitted model, not what the data need; partial dependence asks the model about rows that do not exist; a Shapley value depends on the background it is compared with; and a planted shortcut can win a test set and lose everywhere else.

## V | Where the Rows Come From

Every table was produced by some process, and the process shapes what a model can learn. A missing value records a decision, the applicants who were refused are outcomes nobody observed, and inputs that shift between training and deployment charge for a misspecified model rather than for the shift itself.

## VI | Without Labels

Without labels there is no test score to catch a mistake, and each unsupervised method answers a narrower question than the one it is asked. Principal components rank by variance, not relevance, and a t-SNE picture keeps neighbours while inventing sizes and gaps. k-means finds k pieces whether or not there are clusters, the BIC counts Gaussians rather than clusters, DBSCAN's radius is one density, and the most stable number of clusters measures a unique optimum rather than real groups. Distance fails because of uninformative columns rather than many columns, and an anomaly detector ranks well while its flag is a quantile.

## VII | Grading a Language Model

The oldest problems of evaluation reappear when the model being graded is a language model somebody else trained. A benchmark is a sample of questions with an error bar; a leaked benchmark shows in the set rather than in the item; the prompt is part of the score; multiple choice grades the distractors that were offered; and a model asked to judge others prefers its own answers. The models here are small arithmetic transformers, so every answer can be checked exactly.

## VIII | After the Model Ships

A deployed model needs safeguards, and each one watches something next to the failure. An alarm on the inputs cannot see a change in the outcome; how much history to retrain on depends on how fast the world moves; a prediction interval is right on average and wrong where the noise is; and a model calibrated within every group can have unequal error rates across them.

## IX | The Book, Measured

The last part turns the book's methods on the book. It asks whether findings measured on simulations replicate on real data, reruns the claims that rested on one random draw, and closes by sorting what measurement did to each of the sixty-nine sentences.
