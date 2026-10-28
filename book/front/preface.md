# Preface {.unnumbered}

Most of what a practitioner knows about machine learning arrives as sentences. *The model with the higher test score is better. Cross-validation estimates how well your model will do. Resampling fixes class imbalance. Permutation importance measures how much a model needs a feature. The right number of clusters is the most stable one.* They are taught as facts, repeated in code reviews, and written into checklists, and most of them are true somewhere. This book takes sixty-nine of them and measures where each stops being true.

## How a chapter works {.unnumbered .unlisted}

The book has twenty-two chapters in nine parts, and each chapter is about one topic: what a test score means, what cross-validation estimates, leakage, calibration, thresholds, clustering, and so on. A chapter is made of sections, and each section tests one sentence. It opens with the sentence and the place a practitioner meets it, runs an experiment small enough to follow, usually on data whose right answer is known, and reports what happened; then it explains why, with the definitions and formulas needed to follow the mechanism. The chapter ends by turning its sections' results into practice, stating the limits of their evidence, and giving key points, exercises and further reading.

The chapters do not argue that the curriculum is wrong. Some sentences survive measurement; one survives without any qualification at all. More often a sentence is true under a condition it does not state, and the useful part of the chapter is the condition: the error bar that is narrower than the gap, the leak that matters only on noise, the threshold that is right only if the probabilities are calibrated. Section 22.3 sorts all sixty-nine sentences by what measurement did to them.

## Who it is for {.unnumbered .unlisted}

The reader this book has in mind trains and evaluates models for a living, writes Python, uses scikit-learn, and has met first-year statistics. Nothing beyond that is assumed. Terms are defined where they first matter, and each chapter can be read on its own, though the chapters build on each other and say so where they do.

The book is not a first introduction to machine learning, and it does not replace one. It assumes the reader knows what a train/test split, a logistic regression and a random forest are, and it is about what happens when they are used.

## Data, code and numbers {.unnumbered .unlisted}

Every experiment runs on data a reader can obtain offline: the handwritten digits, breast cancer and diabetes data sets bundled with scikit-learn, simulations with a known answer, and, in Chapter 19, small arithmetic transformers trained in the book's own repository, whose every answer can be checked exactly. The networks are small enough to train on a laptop many times, because most claims about them are about what happens *across* training runs.

Every number in the text was produced by code that ran when the book's sections were built, and each number the text quotes is pinned by an assertion in that code: if a later version of a library changes the result, the build fails rather than leaving a stale number in the prose. Section 22.2 explains what this protects against and what it does not, and reruns the claims that rested on a single random draw. The code blocks are the code that produced the results, and the output below each block is what it printed. The full code is in the repository at <https://github.com/jongha-jeon-dev/standarderror>; Appendix C lists each section's data and machinery.

## How to read it {.unnumbered .unlisted}

The nine parts follow the order in which a project meets its problems: reading a score (Part I), the models themselves (Part II), turning probabilities into decisions (Part III), reading what a model learned (Part IV), where the rows come from (Part V), learning without labels (Part VI), grading a language model somebody else trained (Part VII), what happens after a model ships (Part VIII), and finally the book's own claims, measured (Part IX).

Read in order, the book is a course. Read selectively, start with Part I, which every other part leans on, then go to the chapter that matches the problem in front of you. Appendix A collects every chapter's practical guidance in one place, and Appendix B lists every sentence the book tested and what happened to it.

The sections began as episodes of an online series, and a few figures identify results by the series' episode numbers; Appendix B maps each episode number to its section.

Code is Python. The examples import a small package, `standarderror`, that holds the machinery for each experiment; its functions are short and are named in the text where they matter. Outputs are shown exactly as printed. Figures are referred to by position, and their captions carry the numbers they show.
