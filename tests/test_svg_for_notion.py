"""Tests for the SVG shrinker used to hand figures to Notion as inline text.

The test that matters is `test_text_content_is_never_rewritten`. The first version of
`shrink` rounded every decimal number in the file, which rewrote the numbers *inside* a
table figure — `0.045 ± 0.007` became `0 ± 0`. The figure still rendered, still looked
like a table, and reported different data. Nothing but reading the output would have
caught it, so it is pinned here.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "svg_for_notion",
    Path(__file__).resolve().parents[1] / "scripts" / "svg_for_notion.py")
svg_for_notion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svg_for_notion)
shrink = svg_for_notion.shrink


SAMPLE = """<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Created with matplotlib -->
<svg width="100.123456pt" height="50.987654pt" version="1.1">
 <metadata><rdf>junk</rdf></metadata>
 <g id="figure_1">
  <g id="line2d_21">
   <path id="unreferenced" d="M 1.234567 2.345678
L 3.456789 4.567890
" style="stroke-width: 0.612345"/>
  </g>
  <g id="text_7">
   <text style="font-size: 9.5px" x="12.345678" y="23.456789">0.045 ± 0.007</text>
  </g>
  <g id="text_8">
   <text x="1.5" y="2.5">-0.006 ± 0.005 and 9.4 and 1,234.56</text>
  </g>
  <g id="clipped">
   <path clip-path="url(#pkeepme)" d="M 0 0 L 1.111111 1.111111"/>
  </g>
 </g>
 <defs><clipPath id="pkeepme"><rect x="1.234567" y="2.5" width="9.87" height="8.76"/></clipPath></defs>
</svg>"""


@pytest.fixture(scope="module")
def out():
    return shrink(SAMPLE)


class TestTextIsSacred:
    def test_text_content_is_never_rewritten(self, out):
        """The data-corrupting bug this module exists to not have."""
        assert "0.045 ± 0.007" in out
        assert "-0.006 ± 0.005 and 9.4 and 1,234.56" in out
        assert "0 ± 0" not in out

    def test_every_text_node_survives_byte_for_byte(self, out):
        before = re.findall(r"<text[^>]*>(.*?)</text>", SAMPLE, re.S)
        after = re.findall(r"<text[^>]*>(.*?)</text>", out, re.S)
        assert before == after
        assert len(after) == 2


class TestItActuallyShrinks:
    def test_smaller_than_the_input(self, out):
        assert len(out.encode()) < len(SAMPLE.encode())

    def test_drops_the_prolog_doctype_comment_and_metadata(self, out):
        for gone in ("<?xml", "<!DOCTYPE", "<!--", "<metadata>"):
            assert gone not in out

    def test_rounds_geometry(self, out):
        assert "1.2 2.3" in out          # the path data
        assert "100.1" in out            # the width attribute
        assert "1.234567" not in out

    def test_collapses_path_data_onto_one_line(self, out):
        d = re.search(r'd="([^"]*)"', out).group(1)
        assert "\n" not in d

    def test_keeps_a_font_size_usable(self, out):
        """`style` is rounded, which must not break `font-size: 9.5px`."""
        assert "font-size: 9.5px" in out


class TestIdHandling:
    def test_unreferenced_ids_are_dropped(self, out):
        assert 'id="figure_1"' not in out
        assert 'id="line2d_21"' not in out
        assert 'id="unreferenced"' not in out

    def test_referenced_ids_are_kept(self, out):
        assert 'id="pkeepme"' in out
        assert "url(#pkeepme)" in out

    def test_ids_can_be_preserved_on_request(self):
        kept = shrink(SAMPLE, drop_ids=False)
        assert 'id="line2d_21"' in kept


class TestOutputIsStillValidSvg:
    def test_it_parses(self, out):
        import xml.etree.ElementTree as ET
        ET.fromstring(out)

    def test_it_rasterises_to_the_same_size(self, out):
        cairosvg = pytest.importorskip("cairosvg")
        a = cairosvg.svg2png(bytestring=SAMPLE.encode())
        b = cairosvg.svg2png(bytestring=out.encode())
        assert a and b
        assert abs(len(a) - len(b)) < 0.5 * len(a)

    def test_the_notion_cap_is_what_the_api_documents(self):
        assert svg_for_notion.LIMIT_BYTES == 200 * 1024
