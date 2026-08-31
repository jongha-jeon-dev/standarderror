"""The Post object: structured content in, publishable markdown out.

Why a structured object rather than writing markdown by hand: the same post has
to become a Hugo page (front matter, relative image paths), a Medium import
(absolute image URLs, canonical link, no front matter) and optionally a Notion
page. Rendering three times from one structure keeps them consistent; writing
three files by hand does not survive the second edit.

`Post.audit()` is the part to actually use. It refuses to let you publish a post
that has an untitled figure, a figure without alt text, an unresolved TODO, a
claim with no data citation, or a source you are not licensed to redistribute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ..config import SETTINGS
from ..viz.charts import Figure

#: Medium requires a plain-language label on any story written with AI assistance,
#: placed "within the first two paragraphs". The stated consequence of omitting it
#: is not removal but Network Only distribution — the story publishes, an alert
#: appears on it, and nobody outside your followers sees it. That failure mode is
#: silent from the author's side, which is exactly the kind of thing that belongs
#: in the audit gate rather than in a checklist someone remembers.
#:
#: Keep it accurate rather than minimal. "Assistance" that wrote the code, ran the
#: experiment and drafted the prose is not a grammar checker, and describing it as
#: one would be the dishonest version of complying.
AI_DISCLOSURE = (
    "Disclosure: this post was written with the assistance of an AI system "
    "(Claude), which wrote the analysis code, ran the experiments and drafted "
    "the text. The topic, the constraints, the data choices and the final review "
    "are the author's."
)


@dataclass
class Section:
    heading: str
    body: str = ""
    figures: list[Figure] = field(default_factory=list)
    level: int = 2

    def markdown(self, image_base: str = "") -> str:
        # An empty heading is a continuation of the section above it — a code
        # block and its discussion, say. Emitting "###" with nothing after it
        # produces a bare hash in Hugo and an empty heading block in Notion,
        # so the line is dropped rather than rendered blank.
        parts = ([f"{'#' * self.level} {self.heading}", ""] if self.heading.strip()
                 else [])
        parts += [self.body.strip(), ""]
        for f in self.figures:
            parts += [f.markdown(image_base), ""]
        return "\n".join(parts)


@dataclass
class Post:
    title: str
    slug: str
    subtitle: str = ""
    date: date = field(default_factory=date.today)
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    sections: list[Section] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    licence_warnings: list[str] = field(default_factory=list)
    code_url: str = ""
    author: str = ""
    reproducibility: dict = field(default_factory=dict)
    canonical_url: str = ""
    # Defaulted rather than required, because the failure mode of an empty
    # disclosure is a post that publishes and then quietly reaches nobody. Set it
    # to something else if the process changes; `audit()` refuses an empty one.
    disclosure: str = AI_DISCLOSURE
    # Drafts are the default. In a single public repo the source of an unfinished
    # post is visible on GitHub but `draft: true` keeps it off the built site, so
    # the only way a half-written post reaches readers is if you deliberately
    # flip this. Default-live would make an accidental push a publication.
    draft: bool = True
    # Length targets per post. The "two heavy, one light" rhythm needs light posts
    # to pass the gate, and a light post is 700-1,200 words by definition — with a
    # single global minimum, every one of them would fail.
    min_words: int | None = None
    max_words: int | None = None
    # Rendered stand-ins for the body's markdown tables, in order of appearance.
    # Medium has no table support and strips the markup on paste, so the crosspost
    # substitutes these images. The Hugo page keeps the real table.
    table_figures: list[Figure] = field(default_factory=list)

    #: The post's cover: the one image a reader sees before deciding to read.
    #: Deliberately *not* in `figures` — it is not evidence, it carries no
    #: caption, and it must not be copied into the body. Fifteen experiments set
    #: this attribute for months while nothing read it, so the hero existed as a
    #: file on disk and appeared nowhere; the writers below consume it now.
    hero: Figure | None = None

    # ---------- lecture series ----------
    # Hugo content section. Lectures live under their own section so the site can
    # offer them as a menu item; the crosspost does not care.
    section: str = "posts"
    #: Full series name, for the syllabus page and the front matter taxonomy.
    series: str = ""
    #: Short tag that *opens the title*. On Medium there is no site navigation and
    #: no section, so a reader sees one line of text and nothing else: the only
    #: place "which lecture is this" can live is the title itself. The tag is
    #: therefore part of the title string, and `audit()` enforces that.
    series_tag: str = ""
    episode: int | None = None
    #: Whether the baseline rule applies. `None` auto-detects from the body, which
    #: is right for a post claiming predictive performance and wrong for one whose
    #: comparisons are numerical: "the relative error cannot fall below kappa
    #: times epsilon" is a precision bound with no baseline to compare against,
    #: and the auto-detect matches it. Declare `False` there rather than rewording
    #: the sentence until the pattern stops firing — an exemption stated on the
    #: post is reviewable, and prose bent around a regex is not.
    requires_baseline: bool | None = None

    #: Slugs an episode assumes the reader has. Recorded rather than rendered —
    #: a series that quietly requires episode 4 in episode 2 is a broken series,
    #: and this is where that becomes visible.
    prerequisites: list[str] = field(default_factory=list)

    # ---------- assembly ----------

    def add(self, heading: str, body: str = "", *,
            figures: list[Figure] | None = None, level: int = 2) -> Post:
        self.sections.append(Section(heading, body, list(figures or []), level))
        return self

    @property
    def series_note(self) -> str:
        """One line telling a reader with no site navigation where they are.

        Medium and Notion strip the site away, so an episode arriving there has
        to carry its own position: which series, which number, and where the rest
        of it is. Empty for a standalone post.
        """
        if not (self.series and self.episode):
            return ""
        base = SETTINGS.site_base_url.rstrip("/")
        url = f"{base}/{self.section}/" if base else ""
        where = f" The syllabus and the other episodes: {url}" if url else ""
        return (f"Episode {self.episode} of *{self.series}*.{where}")

    @property
    def figures(self) -> list[Figure]:
        return [f for s in self.sections for f in s.figures] + list(
            self.table_figures)

    @staticmethod
    def find_markdown_tables(text: str) -> list[tuple[int, int]]:
        """(start, end) line indices of each markdown table block in `text`.

        A table is a pipe-delimited header line, a separator line of dashes and
        colons, then one or more body lines. Scanned line by line rather than with
        one regex because the failure mode of a slightly-wrong regex here is
        silently mangling the body.
        """
        lines = text.split("\n")
        out: list[tuple[int, int]] = []
        i = 0
        while i < len(lines) - 1:
            head = lines[i].strip()
            sep = lines[i + 1].strip()
            is_row = head.startswith("|") and head.endswith("|")
            is_sep = (sep.startswith("|") and sep.endswith("|")
                      and set(sep) <= set("|-: \t"))
            if is_row and is_sep:
                j = i + 2
                while j < len(lines) and lines[j].strip().startswith("|"):
                    j += 1
                out.append((i, j))
                i = j
            else:
                i += 1
        return out

    def word_count(self) -> int:
        """Words of *prose*, which is what a length target is about.

        Code, math and table cells are all excluded, for the same reason: none of
        them is read at prose speed, and counting them makes the length gate
        penalise a post for showing its numbers. A table row is matched
        line-anchored on leading and trailing pipes, so a sentence that happens to
        contain a pipe is still counted.
        """
        # The sections only, so a required platform label never counts towards a
        # length target the author is being held to.
        text = "\n".join(s.body for s in self.sections)
        text = re.sub(r"```.*?```", " ", text, flags=re.S)     # exclude code
        text = re.sub(r"\$[^$]*\$", " ", text)                 # exclude math
        text = re.sub(r"^\s*\|.*\|\s*$", " ", text, flags=re.M)  # exclude tables
        return len(re.findall(r"\b[\w'-]+\b", text))

    # ---------- rendering ----------

    def body_markdown(self, image_base: str = "") -> str:
        parts: list[str] = []
        # First paragraph, before the lede: Medium counts from the top of the
        # body, and the crosspost adds a canonical note above this, so anything
        # placed after the lede risks landing in paragraph three.
        if self.disclosure:
            parts += [self.disclosure.strip(), ""]
        if self.summary:
            parts += [f"*{self.summary.strip()}*", ""]
        # After the lede rather than before it: the disclosure has to stay in the
        # first two paragraphs, and the lede is what makes a reader continue.
        if self.series_note:
            parts += [self.series_note, ""]
        for s in self.sections:
            parts.append(s.markdown(image_base))
        parts += self._footer()
        return "\n".join(parts).rstrip() + "\n"

    def _footer(self) -> list[str]:
        out: list[str] = ["---", ""]
        if self.data_sources:
            out += ["### Data", ""]
            out += [f"- {c}" for c in self.data_sources]
            out += [""]
        if self.licence_warnings:
            out += ["### Licence notes", ""]
            out += [f"- {w}" for w in self.licence_warnings]
            out += [""]
        if self.reproducibility:
            out += ["### Reproducibility", ""]
            for k, v in self.reproducibility.items():
                out.append(f"- **{k}**: {v}")
            out += [""]
        if self.code_url:
            out += [f"Code: <{self.code_url}>", ""]
        return out

    def front_matter(self) -> str:
        def esc(s: str) -> str:
            return s.replace('"', '\\"')
        lines = ["---",
                 f'title: "{esc(self.title)}"',
                 f"date: {self.date.isoformat()}",
                 f"slug: \"{self.slug}\"",
                 f"draft: {'true' if self.draft else 'false'}"]
        if self.subtitle:
            lines.append(f'description: "{esc(self.subtitle or self.summary)}"')
        elif self.summary:
            lines.append(f'description: "{esc(self.summary)}"')
        if self.author:
            lines.append(f'author: "{esc(self.author)}"')
        if self.tags:
            lines.append("tags: [" + ", ".join(f'"{t}"' for t in self.tags) + "]")
        if self.series:
            lines.append(f'series: ["{esc(self.series)}"]')
        if self.episode:
            # Ascending weight orders a Hugo section by episode instead of by
            # date, which is what a curriculum needs.
            lines.append(f"weight: {self.episode}")
        if self.hero is not None:
            # Page-relative, because `hugo_page_bundle` copies the file in beside
            # index.md. `images` is what Hugo's own OpenGraph and Twitter-card
            # internal templates read, so this is the field that becomes the
            # preview thumbnail rather than a name of our own invention.
            from pathlib import Path as _P
            lines.append(f'images: ["{_P(self.hero.path).name}"]')
        if self.canonical_url:
            lines.append(f'canonicalURL: "{self.canonical_url}"')
        lines.append("---")
        return "\n".join(lines)

    def hugo_markdown(self, image_base: str = "") -> str:
        return self.front_matter() + "\n\n" + self.body_markdown(image_base)

    # ---------- audit ----------

    def _series_problems(self) -> list[str]:
        """Consistency of the series fields, and of the title against them.

        The title check is the one that matters. A lecture crossposted to Medium
        is one line of text with no section, no menu and no neighbours, so if the
        title does not open with the series tag and the episode number the reader
        cannot tell what they are looking at or what order it goes in. Enforcing
        it here means the failure is a build error rather than a published
        orphan.
        """
        out: list[str] = []
        declared = [bool(self.series), bool(self.series_tag),
                    self.episode is not None]
        if any(declared) and not all(declared):
            out.append("series, series_tag and episode must be set together "
                       f"(got series={self.series!r}, tag={self.series_tag!r}, "
                       f"episode={self.episode!r})")
            return out
        if not any(declared):
            return out
        if self.episode < 1:
            out.append(f"episode must be 1 or greater, got {self.episode}")
        prefix = f"{self.series_tag} {self.episode}: "
        if not self.title.startswith(prefix):
            out.append(f"title must start with {prefix!r} so the episode is "
                       f"identifiable with no site around it; got {self.title!r}")
        slug_prefix = re.sub(r"[^a-z0-9]+", "-",
                             self.series_tag.lower()).strip("-")
        if not self.slug.startswith(f"{slug_prefix}-{self.episode}-"):
            out.append(f"slug must start with {slug_prefix}-{self.episode}- ; "
                       f"got {self.slug!r}")
        if self.section == "posts":
            out.append("an episode belongs in its own section, not 'posts'")
        return out

    def audit(self, *, min_words: int | None = None,
              max_words: int | None = None,
              require_baseline: bool | None = None) -> list[str]:
        """Publication gate. Returns a list of problems; empty means shippable.

        Length bounds resolve from the argument, then the post's own fields, then
        the 1,200-3,000 default for a full-length post.
        """
        problems: list[str] = []
        lo = min_words if min_words is not None else (self.min_words or 1200)
        hi = max_words if max_words is not None else (self.max_words or 3000)
        wc = self.word_count()
        if wc < lo:
            problems.append(f"only {wc} words (target >= {lo})")
        if wc > hi:
            problems.append(f"{wc} words (target <= {hi}) — cut or split")
        if not self.slug or not re.fullmatch(r"[a-z0-9-]+", self.slug):
            problems.append(f"slug {self.slug!r} must be lowercase-kebab")
        problems += self._series_problems()
        if not self.summary:
            problems.append("no summary (needed for description/meta and Medium)")
        if not self.figures:
            problems.append("no figures — an analysis post needs at least one")
        for i, f in enumerate(self.figures, 1):
            if not f.alt or len(f.alt) < 12:
                problems.append(f"figure {i} ({f.path}) has no usable alt text")
            if not f.caption:
                problems.append(f"figure {i} ({f.path}) has no caption")
        if self.hero is not None:
            if not self.hero.alt or len(self.hero.alt) < 12:
                problems.append(f"hero ({self.hero.path}) has no usable alt text")
            if self.hero.path in {f.path for f in self.figures}:
                problems.append(
                    f"hero ({self.hero.path}) is also a body figure, so the cover "
                    f"appears twice in the post")
        if not self.data_sources:
            problems.append("no data citations — every figure needs a source line")
        # A declared table image is only ever *substituted* for a markdown table in
        # the body (Medium and Notion strip table markup, Hugo renders it). Declaring
        # the image without writing the table means the image silently never appears
        # anywhere — which shipped three times before this check existed.
        table_rows_in_body = re.search(
            r"^\s*\|.*\|\s*$", "\n".join(s.body for s in self.sections),
            flags=re.M)
        if self.table_figures and not table_rows_in_body:
            problems.append(
                f"{len(self.table_figures)} table figure(s) declared but no markdown "
                "table in the body — the image has nothing to substitute for and "
                "will not appear in any output")
        # A table image listed *both* in `table_figures` and in a section's figures is
        # emitted twice everywhere the table is substituted — once in place of the
        # markdown table, once at the end of the section. Two identical images, and
        # nothing else notices.
        table_paths = {f.path for f in self.table_figures}
        for s_i, s in enumerate(self.sections, 1):
            for f in s.figures:
                if f.path in table_paths:
                    problems.append(
                        f"section {s_i} ({s.heading!r}): {f.path} is both a table "
                        f"figure and a section figure, so it will appear twice "
                        f"wherever the table is substituted for an image")
        # A markdown table whose rows disagree on column count is not a table:
        # Goldmark rejects the whole block and renders it as a paragraph of pipes.
        # This shipped once, from a cell containing `ACF1 of |r|` — an unescaped pipe
        # splits one cell into two and the header stops matching the separator.
        for s_i, s in enumerate(self.sections, 1):
            for start, end in self.find_markdown_tables(s.body):
                rows = s.body.split("\n")[start:end]
                # An escaped `\|` is a literal pipe inside a cell, not a separator,
                # so it has to come out before counting fields — otherwise this check
                # fires on the very tables that fixed the bug it exists to catch.
                widths = {len(re.sub(r"\\\|", "", r).strip().strip("|").split("|"))
                          for r in rows}
                if len(widths) > 1:
                    problems.append(
                        f"section {s_i} ({s.heading!r}): markdown table has "
                        f"inconsistent column counts {sorted(widths)} — an "
                        f"unescaped '|' inside a cell will silently unrender the "
                        f"whole table")
        body = " ".join(s.body for s in self.sections)

        # Word-bounded, because a substring match here is worse than useless: the
        # Kuramoto-Sivashinsky term `u_xxxx` matched a bare "XXX" search and
        # blocked a finished post. Case matters for XXX and TK for the same reason.
        for marker, flags in (("TODO", re.I), ("FIXME", re.I), ("TK", 0),
                              ("XXX", 0), ("lorem ipsum", re.I)):
            if re.search(rf"\b{marker}\b", body, flags):
                problems.append(f"unresolved {marker} left in the body")

        # The baseline rule applies to posts that claim predictive performance, not
        # to every post. `require_baseline=None` auto-detects from a *comparison*
        # claim; pass True or False to force it.
        claims_performance = bool(re.search(
            r"\b(beats?|beaten|outperform\w*|better than|worse than|"
            r"lower (?:rmse|mae|error)|accuracy of|r-?squared|MASE|"
            r"out-of-sample|state of the art)\b", body, re.I))
        declared = (self.requires_baseline if require_baseline is None
                    else require_baseline)
        needs_baseline = claims_performance if declared is None else declared
        # Known false positives: "better than", "worse than" and "accuracy of"
        # also appear in numerical-precision statements, which are bounds rather
        # than performance claims and have no baseline to compare against. Two
        # of them fired in one 1,300-word lecture, so the escape hatch is a
        # declared `requires_baseline=False` on the post rather than a looser
        # regex here — the gate keeps its teeth on the posts it was written for.
        # For a classifier, the reference point is chance level, not persistence —
        # "flipping a coin" is a baseline, and demanding the word "naive" from a
        # post whose entire subject is the chance distribution is a false positive.
        has_baseline = bool(re.search(
            r"persistence|na[iï]ve|baseline|random walk|"
            r"(?:above |pure |by |than )chance|chance level|"
            r"coin flip\w*|flipping a coin|random guess\w*", body, re.I))
        if needs_baseline and not has_baseline:
            problems.append(
                "no baseline mentioned — a performance claim without a "
                "persistence/naive/chance-level comparison is not falsifiable")
        if not self.reproducibility:
            problems.append("no reproducibility block (seed, versions, commit)")
        # An AI-assistance label is a platform requirement, not a stylistic
        # choice, and omitting it is punished by silence rather than by an error.
        if not self.disclosure.strip():
            problems.append(
                "no AI-assistance disclosure — Medium restricts an undisclosed "
                "story to Network Only distribution, which looks like a "
                "successful publication and reaches nobody")
        elif not re.search(r"\b(ai|artificial intelligence)\b",
                           self.disclosure, re.I):
            problems.append(
                f"the disclosure {self.disclosure[:40]!r}... never says AI, so it "
                f"does not disclose anything")

        # Placeholder URLs are the single most embarrassing thing to ship, because
        # they survive every other check and only become visible to readers.
        haystack = " ".join([body, self.code_url, self.canonical_url,
                             " ".join(self.data_sources)])
        for token in ("YOURNAME", "YOUR_NAME", "example.com", "example.org",
                      "example.github.io", "<url>", "TODO_URL"):
            if token.lower() in haystack.lower():
                problems.append(
                    f"placeholder {token!r} still present — set it before publishing")
        return problems
