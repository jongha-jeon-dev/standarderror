## I | What a Score Means

Every later part of the book compares numbers, so it starts with what a number is. A test accuracy is an estimate with an error bar, often wider than the gap it is used to declare. Cross-validation estimates the average error of a procedure, not of the model in hand. Preprocessing fitted outside the folds can leak, and the size of the leak ranges from nothing to a third of the scale. A learning curve fitted early promises more than more data delivers, and resampling for a rare class moves a threshold while breaking the probabilities. Each of these sentences is close enough to true to be repeated, and wrong enough to change a decision.

## II | Linear Models, Pushed

The simplest models have the most precisely stated textbook properties, which makes them the cleanest place to see those properties fail. On separable data a logistic regression's coefficients are set by the solver's stopping rule. A penalty's strength depends on the units the features happen to be in. The lasso, tuned to predict, does not select the relevant features. The bias–variance trade-off draws a U for some knobs and not for others, and the second descent beyond the interpolation threshold is real but never needed once a ridge penalty is in place.

## III | Trees and Ensembles

Trees are sold as interpretable and ensembles as accurate. This part measures both claims. A single tree's structure changes when the rows are resampled while its accuracy does not; impurity importance rewards features with many places to split; out-of-bag error is free cross-validation only once the forest is large and the rows are not duplicated; boosting overfits its probabilities long before its accuracy; and no tree ensemble predicts beyond the range of values it has seen.

## IV | Reading What a Model Learned

Explanation methods answer precise questions, and those questions are often not the ones being asked. Permutation importance measures the fitted model, not what the data need; partial dependence asks the model about rows that do not exist; a Shapley value depends on the background it is compared with; a planted shortcut can win a test set and lose everywhere else; and two models with the same accuracy can disagree on many individual predictions.

## V | Training a Network

The settings that control training are described as separate, exact and independent, and in practice they are less of each. Initialisation must break symmetry at the right scale, not merely break it. Learning rate and batch size act as one ratio until the rate hits a wall. Adam's reputation for speed belongs mostly to its learning rate. Batch normalisation normalises the batch, not the example, and dropout's weight scaling keeps the answers of the averaged network but not its doubts.

## VI | What Generalisation Looks Like in a Network

The classical intuitions about overfitting meet networks that can fit anything. A network that memorises noisy labels learns the signal first; a network with perfect training accuracy can keep learning for thousands of steps; augmentation adds an assumption rather than data; covariate shift charges for misspecification rather than for shift itself; and a scaling law fitted on small models cannot see the floor that bends it.

## VII | Around the Model

The routine steps around a model, reducing dimensions, clustering, measuring distance, calibrating and tuning, each answer a narrower question than the one they are used for. Principal components rank by variance, not relevance; k-means finds k pieces whether or not there are clusters; nearest neighbours fail because of uninformative columns, not many columns; recalibration fixes the average given enough rows, but not within a group; and the best cross-validated score of a search reports the search's luck.

## VIII | Where the Rows Come From

The rules about rows are right, but often for narrower reasons than their usual ones. Shuffled folds fail a time series only when something unseen changes slowly; rows from the same people count as far fewer examples than their number; a missing value records a decision; the applicants who were refused are the outcomes nobody observed; and a target encoding fitted on the training rows leaks into those same rows.

## IX | Combining and Comparing Models

Combining and comparing models is as easy to get wrong as fitting them. An average beats the average member, not the best one; stacking learns the best combination only from honest predictions; naive Bayes counts correlated evidence twice and only its probabilities notice; a t-test on cross-validation folds finds differences between equal algorithms; and a test set answers honestly only the first time it is consulted.

## X | From Prediction to Decision

A model's output becomes useful when someone acts on it, and the action brings costs, causes and feedback that the prediction never saw. The right threshold is a cost ratio, if the probabilities are calibrated; the model with the best AUC may never be the cheapest; the features a model relies on are not necessarily levers; targeting the highest risk is not targeting the largest effect; and a rule that chooses its own data learns its own choices back.

## XI | Grading a Language Model

The oldest problems of evaluation reappear when the model being graded is a language model somebody else trained. A benchmark is a sample of questions with an error bar; a leaked benchmark shows in the set rather than in the item; the prompt is part of the score; multiple choice grades the distractors that were offered; and a model asked to judge others prefers its own answers. The models here are small arithmetic transformers, so every answer can be checked exactly.

## XII | Without Labels

Without labels there is no test score to catch a mistake, and each unsupervised method answers a narrower question than the one it is asked. A t-SNE picture keeps neighbours and invents sizes and gaps; the BIC counts Gaussians rather than clusters; DBSCAN's radius is one density; an anomaly detector ranks well while its flag is a quantile; and the most stable number of clusters measures a unique optimum rather than real groups.

## XIII | After the Model Ships

A deployed model needs safeguards, and each one watches something next to the failure. An alarm on the inputs cannot see a change in the outcome; a prediction interval is right on average and wrong where the noise is; a model calibrated within every group can have unequal error rates across them; how much history to retrain on depends on how fast the world moves; and findings measured on simulations replicate on real data in direction more reliably than in size.

## XIV | The Course, Measured

The last part turns the book's methods on the book. It reruns the claims that rested on one random draw, measures how a sequence of defensible choices compounds, sorts the book's problems by whether more data cures them, runs one project through every check in order, and closes by sorting what measurement did to each of the sixty-nine sentences.
