# Editing a chapter draft into a chapter

The book has been restructured from seventy short chapters into nine parts
and twenty-two chapters (`book/tools/structure.py`). Each new chapter is made
of sections, and each section is one of the former chapters (which began as
episodes of a blog series; "episode N" = the former chapter N). The
mechanical draft of new chapter C is `book/chapters/chCC.md`:

- `# Chapter title`, then a placeholder `<!-- INTRO: ... -->`;
- each former chapter as a `##` section (its old title), its headings
  demoted one level, its closing sections (In practice, Limits of the
  evidence, Key points, Exercises, Further reading) removed;
- old cross-references "Chapter N" mechanically remapped to "Section C.s"
  (sections are numbered C.1, C.2, ... in order within each chapter), and
  every "this chapter" mechanically changed to "this section".

The removed closing sections, section by section, are in
`book/merge/chCC.closing.md`. The section files the draft was made from are
`book/sections/sNN.md`; the original episode sources are `book/src/chNN.md`
and `book/src/chNN.data.md`. `book/STYLE.md` still applies in full (voice,
numbers, code, figures, no new citations).

## What to do

1. **Write the chapter introduction** in place of the INTRO placeholder:
   300 to 600 words, no heading. Say what the chapter is about and why a
   practitioner meets it; list the sentences the sections test (one per
   section, in italics, with its section number); say how the sections build
   on each other. No numbers that are not in the sources.

2. **Make the sections read as one chapter.** Each section still opens as if
   it were a standalone chapter ("Almost every model comparison rests on one
   sentence..."). Rewrite the opening paragraph or two of every section after
   the first so that it follows from what came before (keep the section's
   sentence under test, in italics, and its roadmap). Where a later section
   re-defines a term defined earlier in the same chapter, shorten it to a
   reminder. Remove bridges that point to "the next chapter" or to
   something that is no longer next; a closing sentence may bridge to the
   next section. Fix "this section" back to "this chapter" where the whole
   chapter is meant.

3. **Cross-references.** Check that each remapped "Section C.s" reads well
   and points at what the text describes (the map is `WHERE` in
   `book/tools/structure.py`; a reference into the same chapter can say
   "Section 2.1 above"). References to old parts ("Part VIII", "Part I of
   this book") refer to the old fourteen-part structure, in which old Part k
   was former chapters 5k-4 to 5k; rewrite them as references to sections,
   chapters or the new parts (I to IX, listed in `structure.py`), or drop
   them. Never refer to "episodes", "arcs" or "the series" except where the
   text is about the series' own numbering (see 6).

4. **Chapter-level closing sections**, at the end, in this order, each as a
   `##` heading: `## In practice`, `## Limits of the evidence`,
   `## Key points`, `## Exercises`, `## Further reading`. Build them from
   `book/merge/chCC.closing.md`:
   - *In practice*: merge and organise all sections' guidance, removing
     duplicates; `###` subheadings by theme or by section are fine. Keep
     every distinct piece of advice.
   - *Limits of the evidence*: keep every distinct limitation; group by
     section where that is clearer.
   - *Key points*: a numbered list of 6 to 12 items covering every section.
   - *Exercises*: a numbered list of 4 to 10, the most useful and distinct
     from all sections.
   - *Further reading*: merged and deduplicated.
   Numbers only from the sources, as always.

5. **Do not cut content** beyond duplicated definitions and transitions.
   Do not change the `##` section titles. Keep every
   `<!-- source: ...; hero: ... -->` comment where it is (the build puts each
   section's summary picture there). Do not touch code blocks.

6. **Figures that show episode numbers.** A few figures (and their alt
   text) identify results by the series' episode numbers, for example a
   legend "(35)" or a table column "from: episode 3" or "ep. 44". Add to the
   caption of each such figure the clause: "Numbers in the figure are the
   series' episode numbers; Appendix B maps them to sections." Do not add it
   elsewhere.

7. Run `python book/tools/check.py C` from `/home/claude/standarderror` and
   fix every flag until it prints "ok".
