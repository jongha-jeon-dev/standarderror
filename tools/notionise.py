"""Turn a post's body markdown into Notion-flavored Markdown.

Three conversions Notion needs and standard markdown does not.

**Paragraphs must be one line each.** The repo hard-wraps prose at 79 columns,
which standard markdown treats as a soft break inside a paragraph. Notion parses
line by line, and the damage is not cosmetic: `**singular\\nvalues**` loses its
bold entirely, `*does my answer\\nsatisfy*` has its asterisks escaped and shown,
and a line that happens to begin `2001.` becomes an ordered list item renumbered
to `1.` -- which silently changed a quoted result on a published page. So every
paragraph is unwrapped to a single line.

**An equation block is `$$` alone on its own line**, the LaTeX, then `$$` alone.
A single-line `$$...$$` renders as literal dollar signs.

**Outside code and equations, `[ ] ^` must be backslash-escaped**, because Notion
reads them as markup. Links and inline code spans are held out: a code span is
literal, so a backslash added inside it is rendered rather than consumed.
"""
import re
import sys
from pathlib import Path

#: A markdown link, or an inline code span -- both must survive escaping.
KEEP = re.compile(r"(!?\[[^\]\n]*\]\([^)\s]+\)|`[^`\n]+`)")
LIST = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)")
VERBATIM = re.compile(r"^(#{1,6}\s|---\s*$|\|)")


def escape_text(seg: str) -> str:
    out = []
    for part in KEEP.split(seg):
        if KEEP.fullmatch(part):
            out.append(part)
            continue
        out.append("".join("\\" + ch if ch in "[]^" else ch for ch in part))
    return "".join(out)


def unwrap(seg: str) -> str:
    blocks, out = re.split(r"\n\s*\n", seg), []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        # Only the FIRST line decides whether this is a list. Testing every
        # line matched a wrapped continuation that happened to begin "2001." --
        # so the paragraph was split into "items" and Notion renumbered that
        # one to "1.", changing a quoted result on the page.
        if LIST.match(lines[0]):
            # A list: join each item's continuation lines into that item.
            items = []
            for ln in lines:
                if LIST.match(ln) or not items:
                    items.append(ln.rstrip())
                else:
                    items[-1] += " " + ln.strip()
            out.append("\n".join(items))
        elif VERBATIM.match(lines[0]) and len(lines) == 1:
            out.append(lines[0])
        else:
            out.append(" ".join(ln.strip() for ln in lines))
    return "\n\n".join(out)


def fix_equations(seg: str) -> str:
    def repl(m):
        return "\n\n$$\n" + " ".join(m.group(1).split()) + "\n$$\n\n"
    return re.sub(r"\$\$(.+?)\$\$", repl, seg, flags=re.S)


def convert(text: str) -> str:
    text = re.sub(r"<(https?://[^>\s]+)>", r"[\1](\1)", text)
    done = []
    for chunk in re.split(r"(```.*?```)", text, flags=re.S):
        if chunk.startswith("```"):
            done.append(chunk)
            continue
        for piece in re.split(r"(\n\n\$\$\n.*?\n\$\$\n\n)", fix_equations(chunk),
                              flags=re.S):
            if piece.startswith("\n\n$$\n"):
                done.append(piece)          # LaTeX: literal, and already shaped
            else:
                # Pad the boundaries: unwrap() drops the blank lines that
                # separated this prose from an adjacent code fence or equation,
                # which glued "...before we start.```python" into one line.
                done.append("\n\n" + escape_text(unwrap(piece)) + "\n\n")
    out = "".join(done)
    return re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"


src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.write_text(convert(src.read_text(encoding="utf-8")), encoding="utf-8")
print(dst)
