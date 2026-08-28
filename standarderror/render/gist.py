"""Split a post's code blocks out into files for one GitHub gist, plus the
paste order for Medium.

Medium's editor receives a paste as *body text* first and formats afterwards, so
a block of Python pasted into an empty line loses its leading whitespace and has
its quotes curled and its `--` turned into an en dash. Three of those are fatal
to Python and all three are silent. Creating the code fence before pasting avoids
it, but that is a habit rather than a guarantee, and it still leaves Medium's
plain code block: no highlighting, no copy button, and lines that wrap instead of
scrolling.

A gist embed has none of those problems. Medium replaces a gist URL alone on a
line with the file rendered verbatim, and appending `?file=NAME` embeds one file
out of a multi-file gist, so a whole episode needs one gist.

What this writes, given a post:

    build/gist/<slug>/NN-<heading>.py    one file per code block, in body order
    build/gist/<slug>/PASTE.md           every block in order, and what to do

The code files are byte-identical to the fenced blocks in the published body, and
`_verify` re-reads them to make that a fact rather than an intention: the whole
point of the exercise is that the code a reader copies out of Medium is the code
that ran at build time. A transport step is not allowed to edit the payload.

Output blocks are deliberately *not* gist files. They are three to six lines, no
reader copies them to run them, and ten embeds in one article is oppressive; they
go into Medium's own code block, which preserves them provided the fence exists
before the paste. PASTE.md says so at each one, with the text to paste.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import SETTINGS
from .post import Post

__all__ = ["gist_bundle", "code_blocks"]

#: A fenced block plus its language. Non-greedy so consecutive fences do not
#: swallow each other, and anchored at line starts so a fence quoted inside a
#: paragraph is not mistaken for one.
FENCE = re.compile(r"^```([^\n`]*)\n(.*?)\n^```$", re.S | re.M)

HEADING = re.compile(r"^(#{2,6}) +(.+?)\s*$", re.M)

#: Languages whose blocks are code a reader might run. Everything else — `text`
#: above all — is output, and stays in the article.
CODE_LANGUAGES = {"python", "py", "r", "sql", "bash", "sh", "json", "yaml"}

PLACEHOLDER = "https://gist.github.com/USER/GIST_ID"


def _slug(text: str, *, limit: int = 44) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(s) <= limit:
        return s or "block"
    # Cut on a word boundary rather than mid-word, so the filename still reads.
    cut = s[:limit].rsplit("-", 1)[0]
    return cut or s[:limit]


def code_blocks(body: str) -> list[dict]:
    """Every fenced block in `body`, in order, tagged with its heading.

    Returns dicts with `language`, `code`, `is_code`, `heading`, `index`
    (1-based, counting code blocks only; `None` for output blocks) and `after`
    (the code block an output block follows, so the guide can name it).
    """
    headings = [(m.start(), m.group(2)) for m in HEADING.finditer(body)]
    out: list[dict] = []
    n = 0
    for m in FENCE.finditer(body):
        lang = m.group(1).strip()
        # The heading a reader was last under. Sections added with an empty
        # heading are continuations, so they are simply absent from `headings`
        # and the walk-back lands on the real one.
        heading = ""
        for pos, text in headings:
            if pos < m.start():
                heading = text
            else:
                break
        is_code = lang.split()[0].lower() in CODE_LANGUAGES if lang else False
        if is_code:
            n += 1
        out.append({"language": lang or "text", "code": m.group(2),
                    "is_code": is_code, "heading": heading,
                    "index": n if is_code else None,
                    "after": None if is_code else (n or None)})
    return out


def _filename(block: dict) -> str:
    ext = {"python": "py", "py": "py", "bash": "sh", "sh": "sh"}.get(
        block["language"], block["language"])
    return f"{block['index']:02d}-{_slug(block['heading'])}.{ext}"


def _verify(out: Path, blocks: list[dict]) -> None:
    """Every written file is the block, to the byte. Nothing else is acceptable:
    a step that exists to protect the code from an editor must not be the thing
    that changes it."""
    bad = []
    for b in blocks:
        if not b["is_code"]:
            continue
        got = (out / _filename(b)).read_text(encoding="utf-8")
        want = b["code"] + "\n"
        if got != want:
            bad.append(_filename(b))
    if bad:
        raise RuntimeError(
            "gist files do not match the published blocks: " + ", ".join(bad))


def _guide(post: Post, blocks: list[dict], gist_url: str) -> str:
    n_code = sum(b["is_code"] for b in blocks)
    lines = [
        f"# Medium paste order — {post.title}",
        "",
        f"{n_code} code block(s) to embed, "
        f"{sum(not b['is_code'] for b in blocks)} output block(s) to paste.",
        "",
        "1. Create one gist from the `.py` files beside this file. It must be "
        "public for Medium to embed it.",
        f"2. Replace `{PLACEHOLDER}` below with its URL"
        + ("." if gist_url != PLACEHOLDER else
           " (or re-run with --gist-url to have them written in)."),
        "3. Work down this list. Prose and figures go in as usual; these are the "
        "blocks that need care.",
        "",
        "The one rule for a paste: on Medium, **create the code block before you "
        "paste into it** — type three backticks on an empty line and press "
        "Enter. Pasting first and converting afterwards has already lost the "
        "indentation.",
        "",
        "---",
        "",
    ]
    for b in blocks:
        where = f'under "{b["heading"]}"' if b["heading"] else "before the first heading"
        if b["is_code"]:
            lines += [
                f"## code {b['index']} — {where}",
                "",
                f"Put this on a line of its own; Medium turns it into the embed:",
                "",
                f"    {gist_url}?file={_filename(b)}",
                "",
            ]
        else:
            lines += [
                (f"## output — follows code {b['after']} ({where})"
                 if b["after"] else f"## output — {where}"),
                "",
                "Create the code block first, then paste exactly this:",
                "",
                "--8<-- paste from the next line --8<--",
                b["code"],
                "--8<-- to the line above --8<--",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"


def gist_bundle(post: Post, *, out_dir: Path | None = None,
                gist_url: str | None = None) -> Path:
    """Write the gist files and the paste guide. Returns the directory."""
    body = post.body_markdown()
    blocks = code_blocks(body)
    if not any(b["is_code"] for b in blocks):
        raise ValueError(f"{post.slug} has no code blocks to put in a gist")

    out = Path(out_dir or (SETTINGS.build_dir / "gist")) / post.slug
    out.mkdir(parents=True, exist_ok=True)
    # A re-run after an episode loses a block should not leave the deleted file
    # behind for someone to embed.
    for stale in out.iterdir():
        if stale.is_file():
            stale.unlink()

    for b in blocks:
        if b["is_code"]:
            (out / _filename(b)).write_text(b["code"] + "\n", encoding="utf-8")
    _verify(out, blocks)
    (out / "PASTE.md").write_text(
        _guide(post, blocks, (gist_url or PLACEHOLDER).rstrip("/")),
        encoding="utf-8")
    return out
