"""Tests for the gist split that carries code blocks into Medium.

The property under test is narrow and the whole reason the module exists: the
file a reader ends up copying out of a Medium embed is byte-identical to the
block that was executed at build time. Everything else here defends that.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from standarderror.render import Post
from standarderror.render import gist

REPO = Path(__file__).resolve().parents[1]

# Indentation, a docstring in double quotes, an f-string in single quotes and a
# `--` in a comment: one of each thing Medium's paste path rewrites.
CODE = '''def f(n):
    "A docstring in double quotes."
    total = 0
    for i in range(n):
        total += i        # two spaces before this comment
    return total

# A comment with -- a double hyphen -- in it.
print(f"{f(4)!r} and 'single quotes' too")'''

OUTPUT = "6 and 'single quotes' too\nsecond line    with a run of spaces"


def a_post(*, blocks: int = 1) -> Post:
    p = Post(title="T", slug="a-slug", date=date(2026, 8, 4))
    for i in range(blocks):
        p.add(f"Heading {i + 1}, with punctuation!",
              f"Some prose.\n\n```python\n{CODE}\n```\n\n```text\n{OUTPUT}\n```")
    return p


def test_a_code_block_is_written_back_byte_for_byte(tmp_path):
    out = gist.gist_bundle(a_post(), out_dir=tmp_path)
    written = sorted(out.glob("*.py"))
    assert len(written) == 1
    assert written[0].read_text(encoding="utf-8") == CODE + "\n"


def test_the_verify_step_actually_fails_when_a_file_drifts(tmp_path):
    blocks = gist.code_blocks(a_post().body_markdown())
    out = tmp_path / "d"
    out.mkdir()
    name = gist._filename(next(b for b in blocks if b["is_code"]))
    (out / name).write_text(CODE.replace("total += i", "total += 1") + "\n",
                            encoding="utf-8")
    with pytest.raises(RuntimeError, match="do not match the published blocks"):
        gist._verify(out, blocks)


def test_verify_catches_the_three_things_medium_actually_does(tmp_path):
    """Curled quotes, an en dash for `--`, and stripped indentation. Each on its
    own is enough to stop the code running, and each is silent."""
    blocks = gist.code_blocks(a_post().body_markdown())
    block = next(b for b in blocks if b["is_code"])
    name = gist._filename(block)
    mutations = {
        "curled quotes": lambda c: c.replace('"', "\u201c", 1),
        "en dash": lambda c: c.replace(" -- ", " \u2013 "),
        "lost indentation": lambda c: "\n".join(
            line.lstrip() for line in c.splitlines()),
    }
    for label, mutate in mutations.items():
        out = tmp_path / label.replace(" ", "-")
        out.mkdir()
        damaged = mutate(CODE)
        assert damaged != CODE, f"{label} did not change the code"
        (out / name).write_text(damaged + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="do not match"):
            gist._verify(out, blocks)


def test_output_blocks_are_not_gist_files_but_are_in_the_guide(tmp_path):
    out = gist.gist_bundle(a_post(), out_dir=tmp_path)
    assert not list(out.glob("*.txt")) and not list(out.glob("*.text"))
    guide = (out / "PASTE.md").read_text(encoding="utf-8")
    assert OUTPUT in guide
    assert "follows code 1" in guide


def test_the_filename_carries_the_heading_and_the_order(tmp_path):
    out = gist.gist_bundle(a_post(blocks=3), out_dir=tmp_path)
    names = sorted(p.name for p in out.glob("*.py"))
    assert names == ["01-heading-1-with-punctuation.py",
                     "02-heading-2-with-punctuation.py",
                     "03-heading-3-with-punctuation.py"]


def test_a_rerun_removes_a_file_for_a_block_that_is_gone(tmp_path):
    out = gist.gist_bundle(a_post(blocks=3), out_dir=tmp_path)
    assert len(list(out.glob("*.py"))) == 3
    out = gist.gist_bundle(a_post(blocks=1), out_dir=tmp_path)
    assert len(list(out.glob("*.py"))) == 1, (
        "a deleted block must not leave an embeddable file behind")


def test_the_gist_url_is_substituted_when_given(tmp_path):
    url = "https://gist.github.com/jongha-jeon-dev/abc123"
    out = gist.gist_bundle(a_post(), out_dir=tmp_path, gist_url=url + "/")
    guide = (out / "PASTE.md").read_text(encoding="utf-8")
    assert f"{url}?file=01-heading-1-with-punctuation.py" in guide
    assert gist.PLACEHOLDER not in guide.split("---", 1)[1]


def test_a_post_with_no_code_is_refused(tmp_path):
    p = Post(title="T", slug="s", date=date(2026, 8, 4))
    p.add("H", "Prose only, and a mention of ``` in passing.")
    with pytest.raises(ValueError, match="no code blocks"):
        gist.gist_bundle(p, out_dir=tmp_path)


def test_a_continuation_section_inherits_the_heading_above_it():
    p = Post(title="T", slug="s", date=date(2026, 8, 4))
    p.add("The real heading", "Prose.")
    p.add("", f"```python\n{CODE}\n```", level=3)
    blocks = gist.code_blocks(p.body_markdown())
    assert [b["heading"] for b in blocks] == ["The real heading"]


@pytest.mark.parametrize("page", sorted(
    (REPO / "site" / "content" / "lectures").glob("*/index.md")))
def test_every_committed_lecture_page_splits_into_code_and_output(page):
    """The lecture episodes are what this is for, so parse the pages as
    committed rather than only a synthetic post."""
    blocks = gist.code_blocks(page.read_text(encoding="utf-8"))
    code = [b for b in blocks if b["is_code"]]
    assert code, f"{page.parent.name} has no code blocks"
    assert all(b["heading"] for b in blocks), "a block landed under no heading"
    assert all(b["language"] == "python" for b in code)
    assert all(b["language"] == "text" for b in blocks if not b["is_code"])
    # Every filename distinct, or two blocks would collide in one gist.
    names = [gist._filename(b) for b in code]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("page", sorted(
    (REPO / "site" / "content" / "lectures").glob("*/index.md")))
def test_no_lecture_code_block_is_wider_than_medium_shows(page):
    """A gist embed scrolls, but only so far before a reader stops. 88 columns
    is what the narrowest of these already respects."""
    for b in gist.code_blocks(page.read_text(encoding="utf-8")):
        for line in b["code"].splitlines():
            assert len(line.rstrip()) <= 88, (
                f"{page.parent.name}: {len(line)} columns in {line[:40]!r}")
