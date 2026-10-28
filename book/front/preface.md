# Preface {.unnumbered}

Most of what a practitioner knows about machine learning arrives as sentences. *The model with the higher test score is better. Cross-validation estimates how well your model will do. Resampling fixes class imbalance. Permutation importance measures how much a model needs a feature. The right number of clusters is the most stable one.* They are taught as facts, repeated in code reviews, and written into checklists, and most of them are true somewhere. This book takes seventy of them, one per chapter, and measures where each stops being true.

## How a chapter works {.unnumbered}

Every chapter has the same shape. It opens with the sentence under test and the place a practitioner meets it. It then runs an experiment small enough to follow, usually on data whose right answer is known, and reports what happened. The middle of the chapter explains why: the mechanism behind the result, with the definitions and formulas needed to follow it. Each chapter then turns the result into practice, states the limits of its own evidence, and ends with key points, exercises and further reading.

The chapters do not argue that the curriculum is wrong. Some sentences survive measurement; one survives without any qualification at all. More often a sentence is true under a condition it does not state, and the useful part of the chapter is the condition: the error bar that is narrower than the gap, the leak that matters only on noise, the threshold that is right only if the probabilities are calibrated. Chapter 70 sorts all sixty-nine sentences by what measurement did to them.

## Who it is for {.unnumbered}

The reader this book has in mind trains and evaluates models for a living, writes Python, uses scikit-learn, and has met first-year statistics. Nothing beyond that is assumed. Terms are defined where they first matter, and each chapter can be read on its own, though the chapters build on each other and say so where they do.

The book is not a first introduction to machine learning, and it does not replace one. It assumes the reader knows what a train/test split, a logistic regression and a random forest are, and it is about what happens when they are used.

## Data, code and numbers {.unnumbered}

Every experiment runs on data a reader can obtain offline: the handwritten digits, breast cancer and diabetes data sets bundled with scikit-learn, simulations with a known answer, and, in Part XI, small arithmetic transformers trained in the book's own repository, whose every answer can be checked exactly. The networks are small enough to train on a laptop many times, because most claims about them are about what happens *across* training runs.

Every number in the text was produced by code that ran when the book's chapters were built, and each number the text quotes is pinned by an assertion in that code: if a later version of a library changes the result, the build fails rather than leaving a stale number in the prose. Chapter 66 explains what this protects against and what it does not, and reruns the claims that rested on a single random draw. The code blocks in each chapter are the code that produced the results, and the output below each block is what it printed. The full code is in the repository at <https://github.com/jongha-jeon-dev/standarderror>; Appendix C lists each chapter's data and machinery.

## How to read it {.unnumbered}

The fourteen parts follow the order in which a project meets its problems: what a score means (Part I), the models themselves (Parts II to VI), the routine steps around a model and the rows it learns from (Parts VII and VIII), comparing models and turning their predictions into decisions (Parts IX and X), grading a language model somebody else trained (Part XI), learning without labels (Part XII), what happens after a model ships (Part XIII), and finally the book's own claims, measured (Part XIV).

Read in order, the book is a course. Read selectively, start with Part I, which every other part leans on, then go to the part that matches the problem in front of you. Appendix A collects every chapter's practical guidance in one place, and Appendix B lists every sentence the book tested and what happened to it.

Code is Python. The examples import a small package, `standarderror`, that holds the machinery for each experiment; its functions are short and are named in the text where they matter. Outputs are shown exactly as printed. Figures are referred to by position, and their captions carry the numbers they show.
