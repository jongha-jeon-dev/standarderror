"""Turn a published Hugo lecture page into Notion-flavored Markdown.

Five differences from the Hugo body, each of which was found by looking at what
episodes one to three actually render as:

1. **Front matter goes.** The title and date live in the database row's
   properties, not in the page body.
2. **A paragraph is one line.** The Hugo source wraps prose at about 80 columns;
   Notion treats every newline as a block break, so a wrapped paragraph arrives
   as six one-line paragraphs. Fenced blocks keep their line breaks exactly.
3. **`text` is spelled `plain text`.** Notion's code block takes a fixed
   vocabulary of language names and rejects the request rather than falling back.
4. **`[`, `]` and `^` are escaped**, outside code, equations, links and inline
   code, because Notion reads `[x]` as a checkbox and `^` as a superscript.
5. **Figures become placeholders.** `![alt](lec07-f1.png)` becomes
   `{{FIG:lec07-f1.png}}`, to be substituted with the `markdown_source` that
   `create_attachment` returns for the SVG copy of that figure. The alt text is
   dropped in the image and the caption stays as its own italic paragraph, which
   is what episodes one to three do.

Usage: `python tools/notion_body.py 7 8` writes `build/notion/lec07.md` and
prints the figure order each page expects.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SITE = Path("site/content/lectures")
OUT = Path("build/notion")

FENCE = re.compile(r"^```")
IMAGE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)\s*$")
AUTOLINK = re.compile(r"<(https?://[^>]+)>")
#: Spans that must not be escaped: inline code, and markdown links.
KEEP = re.compile(r"(`[^`]*`|\[[^\]]*\]\([^)]*\))")
LANGS = {"text": "plain text", "txt": "plain text", "": "plain text"}


def escape(line: str) -> str:
    """Backslash-escape `[ ] ^`, leaving inline code and links alone."""
    out = []
    for i, part in enumerate(KEEP.split(line)):
        if i % 2:                                   # a kept span
            out.append(part)
        else:
            for ch in "[]^":
                part = part.replace(ch, "\\" + ch)
            out.append(part)
    return "".join(out)


def convert(page: Path, hero: str | None) -> tuple[str, list[str]]:
    text = page.read_text(encoding="utf-8")
    body = text.split("---\n", 2)[2] if text.startswith("---\n") else text

    lines, figures = [], []
    in_fence = in_eq = False
    para: list[str] = []

    def flush():
        if para:
            lines.append(escape(" ".join(x.strip() for x in para)))
            lines.append("")
            para.clear()

    for raw in body.split("\n"):
        line = raw.rstrip()
        if FENCE.match(line.lstrip()):
            flush()
            if not in_fence:
                lang = line.lstrip()[3:].strip()
                lines.append("```" + LANGS.get(lang, lang))
            else:
                lines.append("```")
                lines.append("")
            in_fence = not in_fence
            continue
        if in_fence:
            lines.append(raw)
            continue
        if line.strip() == "$$":
            flush()
            lines.append("$$")
            if in_eq:
                lines.append("")
            in_eq = not in_eq
            continue
        if in_eq:
            lines.append(raw.strip())
            continue

        m = IMAGE.match(line)
        if m:
            flush()
            figures.append(m.group(1))
            lines += ["{{FIG:" + m.group(1) + "}}", ""]
            continue
        if not line.strip():
            flush()
            continue
        if line.startswith(("#", "-", "*   ", "> ", "|")) or line == "---":
            # A heading, a list item or a rule is its own block already.
            flush()
            lines.append(escape(AUTOLINK.sub(r"[\1](\1)", line)))
            lines.append("")
            continue
        para.append(AUTOLINK.sub(r"[\1](\1)", line))
    flush()

    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    if hero:
        # After the "Episode N of ..." line, so the AI disclosure stays in the
        # first two paragraphs where Medium's policy requires it.
        anchor = [x for x in out.split("\n") if x.startswith("Episode ")]
        if not anchor:
            raise SystemExit(f"{page.parent.name}: no 'Episode N of' line to "
                             f"place the hero after")
        out = out.replace(anchor[0], anchor[0] + "\n\n{{FIG:" + hero + "}}", 1)
        figures.insert(0, hero)
    return out, figures


def main(episodes: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        pages = sorted(SITE.glob(f"linear-algebra-{ep}-*/index.md"))
        if len(pages) != 1:
            raise SystemExit(f"episode {ep}: found {len(pages)} pages")
        # The PNG, not the SVG wrapper: `create_attachment`'s `source_url`
        # downloads from the public repo server-side, so the 200 KiB inline cap
        # that forced the wrapper does not apply.
        hero = f"lec0{ep}-hero.png"
        md, figs = convert(pages[0], hero)
        dest = OUT / f"lec0{ep}.md"
        dest.write_text(md, encoding="utf-8")
        print(f"{dest}  {len(md.encode())/1024:.1f} KB, {len(figs)} figures")
        for f in figs:
            print(f"    {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["7"]))
