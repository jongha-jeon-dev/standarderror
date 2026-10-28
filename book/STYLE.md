# Rewriting an episode as a book chapter

The book is *Machine Learning, Taught Through What Breaks*, a practitioner's
textbook built from the seventy episodes of the series of the same name.
Chapter N is episode N; Part I to Part XIV are the series' fourteen arcs.
You are rewriting episodes into chapters. The source of chapter N is
`book/src/chNN.md` (the episode's text, code, outputs and figures) and
`book/src/chNN.data.md` (its data notes and references). Write the chapter to
`book/chapters/chNN.md`.

## The reader

A working data scientist or ML engineer who knows Python and scikit-learn,
has trained models, and has had first-year statistics. They read the book in
order or dip into a chapter; either way each chapter must stand on its own
while pointing back to earlier chapters where it builds on them. They want to
know what to do differently on Monday, and why.

## What "rewrite" means

The episodes were written as blog posts: dense, compressed, conversational,
and written to be read once. A chapter is written to be studied. Rewrite the
prose completely in textbook form:

- **Open with the claim and why it matters.** First paragraph: the sentence
  under test (in italics, as in the source), where a practitioner meets it,
  and what goes wrong if it is false. Then a short roadmap of the chapter.
- **Define every term on first use** that a practitioner might not know
  precisely (e.g. McNemar's test, finite-population correction, out-of-fold
  predictions, conformal prediction). A sentence or two; a displayed formula
  where the source uses one or where one makes the idea exact.
- **Explain mechanisms slowly.** Where the source says why something happens
  in a clause, give it a paragraph: what is computed, what it depends on, why
  the measured result follows. Use standard, correct textbook reasoning. Do
  not invent mechanisms the source does not support; where the source hedges
  ("presumably"), keep the hedge.
- **Descriptive section headings** (`##`, `###`) that say what the section
  establishes, not blog-style teasers.
- **Practice.** Each chapter has a section `## In practice` that turns the
  findings into concrete guidance: what to check, what to compute, what to
  report, what to stop doing. It must follow from the chapter's evidence.
- **Fixed closing sections, in this order:**
  1. `## Limits of the evidence` (from the source's "Where this breaks";
     keep every limitation, as prose or a short list).
  2. `## Key points` (from "What to keep"; a numbered list, 4 to 6 items).
  3. `## Exercises` (from "Exercise"; 2 to 4 numbered exercises; you may split
     the source exercise and add conceptual questions, but add no new numeric
     claims).
  4. `## Further reading` (a short list, built only from the references in
     `chNN.data.md` and the text; one line each saying what the reader will
     find there).
- Length: roughly 1.2 to 1.6 times the source's prose, excluding code. Do not
  pad; expand by explaining.
- British spelling, as in the source (standardise, generalise, modelling).
  Plain, precise English; no hype, no rhetorical questions in headings, no
  exclamation marks, no emoji.

## What must not change

- **Numbers.** Every number in the chapter must come from the source files:
  the prose, the code output, the captions, the data notes. You may round a
  source number to fewer decimals, or write a proportion as a percentage
  (0.986 as 98.6%). Do not compute new numbers (no new differences, ratios or
  sums), do not estimate, do not quote numbers from memory. Small counts
  written as words are fine. A checker (`python book/tools/check.py NN`) flags
  any number not in the source; run it and fix every flag before finishing.
- **Claims.** No new empirical claims about these experiments. General
  textbook facts are fine if correct and standard; cite nothing that is not in
  the source.
- **Code and output.** Every fenced code block in the source, and every
  output block that follows it, appears in the chapter verbatim (same fence
  language, same content, same order). You may move a block within the chapter
  only if order is preserved, and you write new prose around it: what the
  code does before it, what the output shows after it. Do not add new code
  blocks.
- **Figures.** Every figure in the source (files `mlNNN-f*.png`) appears
  once, as a pandoc image on its own line:

  `![Caption sentence(s).](mlNNN-f0-name.png){alt="Alt text from the source."}`

  The caption is the source's caption (the italic line after the image),
  rewritten as a proper figure caption if needed, numbers unchanged; the alt
  text is the source's image alt text. Place each figure after the paragraph
  that first discusses it, and refer to it in the prose as "the figure below"
  or "the figure above" (figure numbers are added at build time). Do not
  include the hero image (`*-hero.png`); the build places it.
- **Tables in the source** are images; treat them as figures.

## Book voice and cross-references

- Say "this chapter", "this book", "Chapter 35", "Part VIII". Never "episode",
  "post", "series", "this lecture", "I" or "we measured in the last post".
  The author voice is impersonal or "we" sparingly ("we can see that").
- Where the source refers to another episode ("episode 35's optimism"),
  write "Chapter 35" and give enough context that the reader who skipped it
  can follow.
- Mentions of how the series was built (asserts pinning numbers, builds
  failing) belong in the preface; drop them from chapters unless the chapter
  is about that (Chapters 66 and 70).
- No disclosure lines, no links to the blog, no "Next" section. A final
  sentence or two at the end of the last body section, before "In practice",
  may bridge to the next chapter's question.
- Math: use `$...$` and `$$...$$` (pandoc/LaTeX). Keep notation light.
- Start the file with `# <Chapter title>` exactly as the first heading of the
  source (no "Chapter N:" prefix; numbering is added at build time). Keep the
  source's first-line HTML comment.
- Do not reproduce the source's italic summary paragraph; the opening section
  replaces it.
