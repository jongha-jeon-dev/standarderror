#!/usr/bin/env python3
"""Shrink a matplotlib SVG enough to hand to Notion's attachment API as inline text.

Why this exists
---------------
Notion pages published from this repo pointed at GitHub Pages URLs, which 404 until
the site is live, so every figure showed as a broken image. There is a second path:
`notion-create-attachment` accepts a file's *content* inline for text formats, and SVG
is one of them, capped at 200 KiB. No public URL and no network egress needed — which
matters because this container cannot reach `api.notion.com` either (403 at the
proxy), so the multipart upload flow is unavailable.

The obstacle is size, in two senses. The 200 KiB cap is generous, but the content also
has to pass through a model's context to reach the tool call, so every kilobyte is
paid for twice. A default matplotlib SVG of one chart is ~83 KB; this gets it to ~9 KB:

* `SERR_SVG_FONTS=reference` (see `viz.theme.apply`) stops glyphs being embedded
  as outline paths, which alone is a 6x cut. Text then renders in whatever font the
  viewer has, so check a rasterised copy — `cairosvg` is enough — before shipping.
* Here: drop the XML prolog, DOCTYPE, comments and metadata; drop the `id=` attributes
  matplotlib assigns to every artist, which nothing references; round coordinates to
  one decimal place; collapse whitespace inside path data.

What this will not do is the hand-drawn hero cards. The xkcd path filter resamples
every stroke finely *and* duplicates it for the white outline, so those land near
950 KB and are barely affected by any of the above. Ship those as PNG files.

Usage:
    SERR_FIG_EXT=svg SERR_SVG_FONTS=reference standarderror run <experiment>
    python scripts/svg_for_notion.py build/img/a7-f1-window.svg
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Attachment cap for inline text content in Notion's API.
LIMIT_BYTES = 200 * 1024


#: Attributes whose values are geometry and may be rounded. Everything else — and
#: above all the text *content* between tags — is left exactly as it is.
GEOMETRY_ATTRS = frozenset({
    "d", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry",
    "width", "height", "viewBox", "transform", "points", "style",
    "stroke-dasharray", "stroke-dashoffset", "offset",
})


def shrink(svg: str, *, decimals: int = 1, drop_ids: bool = True) -> str:
    """Return an equivalent SVG with the parts no renderer needs removed.

    Rounding is confined to `GEOMETRY_ATTRS`. The first version of this rounded every
    decimal in the file, which silently rewrote the *data* in a table figure —
    `0.045 ± 0.007` became `0 ± 0` and `0.155` became `0.2`. It was caught by printing
    the output and reading it, which is the only reason it was caught, so
    `tests/test_svg_for_notion.py` now asserts text content survives byte for byte.
    """
    s = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    s = re.sub(r"<!DOCTYPE[^>]*>\s*", "", s, flags=re.S)
    s = re.sub(r"<!--.*?-->\s*", "", s, flags=re.S)
    s = re.sub(r"<metadata>.*?</metadata>\s*", "", s, flags=re.S)
    if drop_ids:
        # matplotlib labels every artist (`id="line2d_21"`). Keep the ids that are
        # actually referenced — clip paths and the reusable tick marker — and drop
        # the rest, which is most of them.
        referenced = set(re.findall(r"url\(#([^)]+)\)", s))
        referenced |= set(re.findall(r'xlink:href="#([^"]+)"', s))
        s = re.sub(r'\s+id="([^"]+)"',
                   lambda m: m.group(0) if m.group(1) in referenced else "", s)

    def round_in(value: str) -> str:
        def one(m: re.Match) -> str:
            out = f"{float(m.group(0)):.{decimals}f}".rstrip("0").rstrip(".")
            return out if out not in ("", "-") else "0"
        return re.sub(r"-?\d+\.\d+", one, value)

    def attribute(m: re.Match) -> str:
        name, value = m.group(1), m.group(2)
        if name not in GEOMETRY_ATTRS:
            return m.group(0)
        if name == "d":
            # Path data is emitted one command per line; renderers do not care.
            value = " ".join(value.split())
        return f'{name}="{round_in(value)}"'

    s = re.sub(r'([\w:-]+)="([^"]*)"', attribute, s)
    s = hoist_font_family(s)
    s = re.sub(r"\n\s*\n", "\n", s)
    return s.strip()


#: Matches the font stack matplotlib repeats inside every `style=` it writes.
_FONT_DECL = re.compile(r"font-family:\s*(?P<stack>[^;\"]+);?\s*")


def hoist_font_family(s: str) -> str:
    """Move the repeated font stack into one CSS rule.

    matplotlib writes the full family list into the `style` attribute of every
    text element. A table figure has one per cell, so a five-column, four-row
    table carries the same 90-character string forty times. Declaring it once on
    a `text` selector and deleting the per-element copies is worth roughly half
    the file on text-heavy figures, and costs nothing: a `style` attribute still
    beats a stylesheet rule in the cascade, so any figure that genuinely needs a
    different face per element keeps it — only the *identical* declarations are
    removed.
    """
    stacks = [m.group("stack").strip() for m in _FONT_DECL.finditer(s)]
    if len(stacks) < 4:
        return s
    common = max(set(stacks), key=stacks.count)
    if stacks.count(common) < 4:
        return s

    def strip_common(m: re.Match) -> str:
        return "" if m.group("stack").strip() == common else m.group(0)

    out = _FONT_DECL.sub(strip_common, s)
    out = re.sub(r'style="\s*"', 'style=""', out)
    rule = f"text{{font-family:{common}}}"
    if "<style type=\"text/css\">" in out:
        out = out.replace("<style type=\"text/css\">",
                          f"<style type=\"text/css\">{rule}", 1)
    else:
        out = out.replace("<defs>", f'<defs><style type="text/css">{rule}</style>',
                          1)
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        p = Path(arg)
        before = p.read_text()
        after = shrink(before)
        out = p.with_suffix(".notion.svg")
        out.write_text(after)
        b0, b1 = len(before.encode()), len(after.encode())
        flag = "" if b1 <= LIMIT_BYTES else "  OVER THE 200 KiB CAP"
        print(f"{p.name:28s} {b0 / 1024:8.1f} -> {b1 / 1024:6.1f} KiB  "
              f"({100 * (1 - b1 / b0):4.1f}% smaller){flag}")
        if "<text" not in after and "<tspan" not in after:
            print("  note: no <text> elements — glyphs may still be embedded as "
                  "paths; set SERR_SVG_FONTS=reference when rendering")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
