# The book

*Machine Learning, Taught Through What Breaks*: the seventy episodes of the
lecture series, rewritten as a practitioner's textbook in nine parts and
twenty-two chapters (`tools/structure.py`).

- `chapters/chCC.md` — the 22 chapters. Each is made of sections, and each
  section is one episode (the map is `WHERE` in `tools/structure.py`).
- `front/preface.md`, `front/parts.md` — the preface and the part openers.
- `sections/sNN.md` — the episodes rewritten as standalone textbook chapters
  (the first edition's 70 chapters), from which `tools/draft.py` assembled
  the 22 chapter drafts; `merge/` holds the closing sections they replaced.
- `src/` — each episode's text and data notes, extracted from
  `site/content/lectures/` by `tools/extract.py`; the reference the chapters
  are checked against.
- `STYLE.md`, `EDIT.md` — the briefs the sections and chapters were written to.

Build (pandoc 3 and XeLaTeX with TeX Gyre Pagella):

    python book/tools/check.py        # numbers, code and figures vs the sources
    python book/tools/build.py        # book/out/machine-learning-what-breaks.{pdf,epub}

`check.py` flags any number in a chapter that does not appear in one of its
sections' source episodes, any code block that differs from the sources or
is out of order, and any source figure the chapter does not use. The
appendices are generated: A from each chapter's *In practice* section, B from
the series index and `experiments/ml170_verdicts.py`, C from the episodes'
data notes.
