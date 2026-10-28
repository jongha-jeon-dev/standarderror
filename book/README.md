# The book

*Machine Learning, Taught Through What Breaks*: the seventy episodes of the
lecture series, rewritten as a practitioner's textbook in fourteen parts.

- `chapters/chNN.md` — the chapters (chapter N is episode N).
- `front/preface.md`, `front/parts.md` — the preface and the part openers.
- `src/` — each episode's text and data notes, extracted from
  `site/content/lectures/` by `tools/extract.py`; the reference the chapters
  are checked against.
- `STYLE.md` — the brief the chapters were written to.

Build (pandoc 3 and XeLaTeX with TeX Gyre Pagella):

    python book/tools/check.py        # numbers, code and figures vs the source
    python book/tools/build.py        # book/out/machine-learning-what-breaks.{pdf,epub}

`check.py` flags any number in a chapter that does not appear in its source
episode, any code block that differs from the source or is out of order, and
any source figure the chapter does not use. The appendices are generated:
A from each chapter's *In practice* section, B from the series index and
`experiments/ml170_verdicts.py`, C from the episodes' data notes.
