"""Wrap a hero card as an SVG small enough for Notion's inline-text upload.

`tools/notion_figures.py` explains why inline UTF-8 is the only route: the upload
URL that `create_file_upload` returns points at `api.notion.com`, and the egress
proxy on both this container and the author's machine refuses it. (A plain TCP
connect to `api.notion.com:443` *succeeds* -- it reaches the local proxy, which
then rejects the CONNECT. Do not read that as reachability.)

The body figures go inline as SVG. A hero cannot: it is an xkcd-wobbled
hand-drawn card, so every straight line is a many-point path, and one is 890 KB
of SVG or 453 KB after scour against a 200 KiB cap.

So the hero goes inline as a *raster*, wrapped in a minimal SVG: downscale to
`WIDTH`, quantise to `COLORS`, base64 the PNG into an `<image>` element. These
cards are black and grey ink on one off-white ground, so 16 colours is not a
compromise -- it is more than the drawing uses. The check that this is true is
the reported mean error and the share of pixels that moved visibly.

    python tools/notion_hero.py lec07-hero lec08-hero
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

#: Wide enough that the note text under the panels stays readable at Notion's
#: full-width block size, and small enough to fit the cap with room to spare.
WIDTH = 1400
COLORS = 16
CAP = 200 * 1024

TEMPLATE = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" '
    'viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
    '<image width="{w}" height="{h}" xlink:href="data:image/png;base64,{b64}"/>'
    '</svg>'
)


def wrap(src: Path, out: Path) -> dict:
    im = Image.open(src).convert("RGB")
    if im.width < WIDTH:
        raise SystemExit(f"{src.name} is {im.width}px wide; refusing to upscale")
    small = im.resize((WIDTH, round(im.height * WIDTH / im.width)),
                      Image.LANCZOS)
    # No dither: dithering a flat ground into noise is what makes these files
    # big, and there is no gradient here for it to help with.
    q = small.quantize(colors=COLORS, method=Image.MEDIANCUT,
                       dither=Image.NONE)
    buf = io.BytesIO()
    q.save(buf, "PNG", optimize=True)
    png = buf.getvalue()

    # The mean, and the share of pixels that moved visibly. The *max* is not
    # the number to judge this by: it is always large, because it lands on the
    # antialiased edge between ink and paper, and it says nothing about whether
    # the card reads correctly.
    d = np.abs(np.asarray(small, dtype=int)
               - np.asarray(q.convert("RGB"), dtype=int))
    err = {"mean": round(float(d.mean()), 2),
           "share_over_30": round(float((d.max(axis=2) > 30).mean()), 4)}
    svg = TEMPLATE.format(w=q.width, h=q.height,
                          b64=base64.b64encode(png).decode("ascii"))
    size = len(svg.encode())
    if size > CAP:
        raise SystemExit(f"{out.name}: {size/1024:.0f} KB, over the 200 KiB cap")
    out.write_text(svg, encoding="utf-8")
    return {"png": len(png), "svg": size, "error": err, "size": q.size}


def main(names: list[str]) -> int:
    img = Path("build/img")
    for name in names:
        got = wrap(img / f"{name}.png", img / f"{name}-notion.svg")
        print(f"{name}: {got['size'][0]}x{got['size'][1]}, "
              f"png {got['png']/1024:.0f} KB, svg {got['svg']/1024:.0f} KB, "
              f"mean error {got['error']['mean']}, "
              f"{got['error']['share_over_30']:.1%} of pixels moved visibly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
