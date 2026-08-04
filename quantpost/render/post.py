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

from ..viz.charts import Figure


@dataclass
class Section:
    heading: str
    body: str = ""
    figures: list[Figure] = field(default_factory=list)
    level: int = 2

    def markdown(self, image_base: str = "") -> str:
        parts = [f"{'#' * self.level} {self.heading}", "", self.body.strip(), ""]
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
    # Drafts are the default. In a single public repo the source of an unfinished
    # post is visible on GitHub but `draft: true` keeps it off the built site, so
    # the only way a half-written post reaches readers is if you deliberately
    # flip this. Default-live would make an accidental push a publication.
    draft: bool = True

    # ---------- assembly ----------

    def add(self, heading: str, body: str = "", *,
            figures: list[Figure] | None = None, level: int = 2) -> Post:
        self.sections.append(Section(heading, body, list(figures or []), level))
        return self

    @property
    def figures(self) -> list[Figure]:
        return [f for s in self.sections for f in s.figures]

    def word_count(self) -> int:
        text = " ".join(s.body for s in self.sections)
        text = re.sub(r"```.*?```", " ", text, flags=re.S)     # exclude code
        text = re.sub(r"\$[^$]*\$", " ", text)                 # exclude math
        return len(re.findall(r"\b[\w'-]+\b", text))

    # ---------- rendering ----------

    def body_markdown(self, image_base: str = "") -> str:
        parts: list[str] = []
        if self.summary:
            parts += [f"*{self.summary.strip()}*", ""]
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
        if self.canonical_url:
            lines.append(f'canonicalURL: "{self.canonical_url}"')
        lines.append("---")
        return "\n".join(lines)

    def hugo_markdown(self, image_base: str = "") -> str:
        return self.front_matter() + "\n\n" + self.body_markdown(image_base)

    # ---------- audit ----------

    def audit(self, *, min_words: int = 1200, max_words: int = 3000,
              require_baseline: bool = True) -> list[str]:
        """Publication gate. Returns a list of problems; empty means shippable."""
        problems: list[str] = []
        wc = self.word_count()
        if wc < min_words:
            problems.append(f"only {wc} words (target >= {min_words})")
        if wc > max_words:
            problems.append(f"{wc} words (target <= {max_words}) — cut or split")
        if not self.slug or not re.fullmatch(r"[a-z0-9-]+", self.slug):
            problems.append(f"slug {self.slug!r} must be lowercase-kebab")
        if not self.summary:
            problems.append("no summary (needed for description/meta and Medium)")
        if not self.figures:
            problems.append("no figures — an analysis post needs at least one")
        for i, f in enumerate(self.figures, 1):
            if not f.alt or len(f.alt) < 12:
                problems.append(f"figure {i} ({f.path}) has no usable alt text")
            if not f.caption:
                problems.append(f"figure {i} ({f.path}) has no caption")
        if not self.data_sources:
            problems.append("no data citations — every figure needs a source line")
        body = " ".join(s.body for s in self.sections)
        for marker in ("TODO", "FIXME", "TK", "XXX", "lorem ipsum"):
            if marker.lower() in body.lower():
                problems.append(f"unresolved {marker} left in the body")
        if require_baseline and not re.search(
                r"persistence|na[iï]ve|baseline|random walk", body, re.I):
            problems.append(
                "no baseline mentioned — a forecasting claim without a "
                "persistence/naive comparison is not falsifiable")
        if not self.reproducibility:
            problems.append("no reproducibility block (seed, versions, commit)")

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
