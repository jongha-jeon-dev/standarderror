"""Build a post's figures as SVGs small enough to hand to Notion inline.

Notion's attachment API takes a file either from a public URL or as inline UTF-8
text, capped at 200 KiB. The upload URL that `create_file_upload` hands back
points at `api.notion.com`, and the egress proxy on this container refuses the
CONNECT, so it cannot be POSTed to. Inline text is one route, and that means SVG.

Do not read a successful TCP connect to `api.notion.com:443` as reachability: it
reaches the local proxy, which then rejects the tunnel. The only check that means
anything is an actual request.

The better route, once a commit is pushed: `create_attachment` also takes
`source_url`, and Notion downloads it server-side, so the container's proxy is
irrelevant. A figure committed to this public repo is fetchable at
`https://raw.githubusercontent.com/<owner>/<repo>/main/<path>` and goes in as the
full-quality PNG with no 200 KiB cap. Use that whenever the page bundle is on
GitHub; the SVG path below is for figures that are not pushed yet.

One thing about the substitution afterwards. Notion escapes `{` and `}` in
character data, so a `{{FIG:name.png}}` placeholder written into a page reads
back as `\\{\\{FIG:name.png\\}\\}` -- and that escaped form is what an
`update_page` `content_updates` `old_str` has to match.

Three steps, in this order:

1. Render with `SERR_SVG_FONTS=reference`, which points at font *names* instead
   of embedding every glyph as an outline. See `standarderror.viz.theme.apply`.
   The viewer substitutes its own font, so check a rasterised copy.
2. Minify with `scour`. It is an XML parser, so `--set-precision` shortens
   numeric *attributes* and leaves the character data of a `<text>` node alone.
3. Diff every label against the unminified original and fail on any change.

Step 3 exists because the first version of this did step 2 with a regex,
`\\d+\\.\\d{3,}` rounded to two places, applied to the whole file. It does not
know an attribute from a label, so:

    "this sliver is 8.00 long and 0.002 wide"  ->  "... and 0.00 wide"
    the repair bars' -0.046, -0.078, +0.048    ->  -0.05, -0.08, 0.05
    the leverage table's -0.114                ->  -0.11

Three figures had their published numbers altered by their own transport step,
and the captions quoting those numbers would have disagreed with the picture. A
size optimisation is not allowed to touch a value; this check is what makes that
a fact rather than an intention.

One thing about the upload side, learned the hard way. A `create_attachment`
upload that is never attached to a page *expires*, and roughly an hour is enough:
inserting six figures into episode 1 failed on the second one with `invalid status
of expired`, because it had been uploaded while the rest were still being
prepared. So upload and attach in the same breath — one batch of uploads, then
immediately the `update_content` that references them — rather than uploading
everything first and inserting afterwards.

    python tools/notion_figures.py lec001_condition_number lec002_three_ways


WHICH ROUTE TO USE, settled by two failures rather than by preference.

`source_url` is the only one that works in practice. Notion downloads the file
server-side, so neither this container's egress policy nor the author's matters,
and the file arrives at full quality with no size cap worth thinking about. It
needs the figure to be fetchable, which means **the page bundle has to be pushed
first** -- so the Notion page must be created *after* the push, not before. A
page created first sits there with `{{FIG:...}}` placeholders in it, which is
what the author kept seeing, four times.

The inline-UTF-8 route below is real but I cannot drive it. The files are 37-46
KB after `scour` and the cap is 200 KiB, so the sizes are fine; what fails is
getting the exact bytes into a tool call, because the shell truncates any output
over about 2 KB to a file and hands back a preview. So this module is for
*preparing and checking* figures -- sizes, and the label-preservation gate below
-- and the upload itself goes through `source_url` after a push.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCOUR = ["--enable-id-stripping", "--enable-comment-stripping", "--shorten-ids",
         "--remove-metadata", "--no-line-breaks", "--set-precision=4"]
#: Anything between tags that contains a digit -- the labels a reader sees.
LABEL = re.compile(r">([^<>]*[0-9][^<>]*)<")
#: Dropped by --remove-metadata, so not a label change worth failing on.
METADATA = re.compile(r"^\d{4}-\d\d-\d\dT|^Matplotlib v")


def labels(text: str) -> list[str]:
    return [x for x in LABEL.findall(text) if not METADATA.match(x)]


def build(experiment: str) -> list[Path]:
    """Render `experiment`'s figures as SVG and return the body ones.

    Called with an experiment module name it re-runs the figures; called with a
    filename prefix such as `lec101` it takes whatever SVGs are already in the
    build directory. The second form exists because `figures()` does not have
    one signature across episodes -- episode 1 of the second series takes a
    second argument for the network half -- and a tool that has to know each
    experiment's call shape breaks every time one of them grows an argument.

    The cover is skipped, and not because it is optional. A hand-drawn card is
    xkcd-wobbled, so every straight line is a many-point path: lec01-hero is
    890 KB of SVG and 453 KB after scour, against Notion's 200 KiB inline cap.
    Use `tools/notion_hero.py` for those.
    """
    os.environ["SERR_SVG_FONTS"] = "reference"
    os.environ["SERR_FIG_EXT"] = "svg"
    try:
        mod = importlib.import_module(f"experiments.{experiment}")
    except ModuleNotFoundError:
        import standarderror as se
        found = sorted((se.SETTINGS.build_dir / "img").glob(f"{experiment}*.svg"))
        if not found:
            raise SystemExit(
                f"{experiment!r} is neither an experiment module nor a prefix "
                f"with rendered SVGs. Render with "
                f"`SERR_FIG_EXT=svg standarderror run <experiment>` first."
            ) from None
        return [f for f in found if "hero" not in f.name]
    mod.IMG.mkdir(parents=True, exist_ok=True)
    figs = mod.figures(mod.compute())
    return [Path(f.path) for name, f in figs.items()
            if name != "hero" and not name.startswith("_")]


def minify(path: Path) -> tuple[int, int]:
    raw = path.read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        out = Path(tmp.name)
    r = subprocess.run(["scour", "-i", str(path), "-o", str(out)] + SCOUR,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"scour failed on {path.name}: {r.stderr[-300:]}")
    small = out.read_text(encoding="utf-8")
    before, after = labels(raw), labels(small)
    if before != after:
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        raise SystemExit(
            f"{path.name}: minification changed {len(changed)} label(s), which "
            f"means it changed a number a reader will see:\n" +
            "\n".join(f"  {a!r} -> {b!r}" for a, b in changed[:6]))
    path.write_text(small, encoding="utf-8")
    out.unlink()
    return len(raw.encode()), len(small.encode())


def main(names: list[str]) -> int:
    sys.path.insert(0, ".")
    cap = 200 * 1024
    total = 0
    for name in names:
        for path in build(name):
            before, after = minify(path)
            total += after
            flag = "  OVER 200 KiB" if after > cap else ""
            print(f"{before/1024:7.1f} -> {after/1024:6.1f} KB  "
                  f"{path.name}{flag}")
    print(f"total {total/1024:.0f} KB; every label verified unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["lec001_condition_number"]))
