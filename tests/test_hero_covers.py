"""Tests for the post cover: the lecture-hero template and the `hero` wiring.

`post.hero` was set by fifteen experiments for months while nothing read it, so
the cover existed as a file on disk and appeared in no output. These tests are
mostly about that: the attribute is consumed, and a lecture cannot ship without
a cover that matches its siblings.
"""

from __future__ import annotations

import ast
import json
from datetime import date
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from standarderror.render import Post, publish
from standarderror.viz import charts
from standarderror.viz.charts import Figure

REPO = Path(__file__).resolve().parents[1]
LECTURES = sorted((REPO / "experiments").glob("lec*.py"))


def a_panel(panel, m):
    panel.plot([0, 1], [0, 1], color=m.ink)
    panel.set_xlim(0, 1)


def three(labels=("one", "two", "three")):
    return [(a_panel, "1", labels[0]), (a_panel, "2", labels[1]),
            (a_panel, "3", labels[2])]


def a_post(tmp_path: Path, *, hero: Figure | None = None) -> Post:
    p = Post(title="T", slug="a-slug", summary="S.", date=date(2026, 8, 4),
             tags=["x"], data_sources=["Somewhere — <https://x.int>"],
             author="A", hero=hero)
    fig_path = tmp_path / "f1.png"
    fig_path.write_bytes(b"not really a png")
    p.add("H", " ".join(["word"] * 1400),
          figures=[Figure(str(fig_path), alt="A chart of something specific",
                          caption="Fig 1.")])
    return p


def a_hero(tmp_path: Path, name: str = "hero.png") -> Figure:
    path = tmp_path / name
    path.write_bytes(b"not really a png either")
    return Figure(str(path), alt="A three-panel hand-drawn strip.", caption="",
                  title="t")


# ---------------------------------------------------------------- the template

def test_the_badge_is_the_series_and_the_episode(tmp_path):
    out = charts.lecture_hero(series="Linear Algebra", episode=3,
                              headline="H", panels=three(), alt="a" * 20,
                              path=str(tmp_path / "h.png"))[0]
    assert Path(out.path).exists()


@pytest.mark.parametrize("kwargs, match", [
    (dict(panels=[(a_panel, "1", "one"), (a_panel, "2", "two")]), "three panels"),
    (dict(panels=three() + [(a_panel, "4", "four")]), "three panels"),
    (dict(episode=0), "episode must be 1"),
    (dict(series="  "), "series name"),
    (dict(panels=three(("a label that is far too long to fit", "b", "c"))),
     "one line of 24 characters"),
])
def test_the_template_refuses_what_would_look_wrong(tmp_path, kwargs, match):
    call = dict(series="Linear Algebra", episode=1, headline="H",
                panels=three(), alt="a" * 20, path=str(tmp_path / "h.png"))
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        charts.lecture_hero(**call)


# ---------------------------------------------------------------- the wiring

def test_the_hero_is_named_in_front_matter_by_basename(tmp_path):
    post = a_post(tmp_path, hero=a_hero(tmp_path))
    assert 'images: ["hero.png"]' in post.front_matter()
    # A path, not a name, would break the moment Hugo resolved it page-relative.
    assert str(tmp_path) not in post.front_matter()


def test_a_post_with_no_hero_says_nothing_about_images(tmp_path):
    assert "images:" not in a_post(tmp_path).front_matter()


def test_the_hugo_bundle_copies_the_hero_in(tmp_path):
    post = a_post(tmp_path, hero=a_hero(tmp_path))
    site = tmp_path / "site"
    publish.hugo_page_bundle(post, site_dir=site, section="lectures")
    assert (site / "content" / "lectures" / post.slug / "hero.png").exists(), (
        "front matter names the file, so the bundle has to contain it")


def test_a_missing_hero_file_fails_the_bundle_rather_than_shipping(tmp_path):
    hero = a_hero(tmp_path)
    Path(hero.path).unlink()
    post = a_post(tmp_path, hero=hero)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        publish.hugo_page_bundle(post, site_dir=tmp_path / "site",
                                 section="lectures")


def test_the_manifest_records_the_hero(tmp_path):
    post = a_post(tmp_path, hero=a_hero(tmp_path))
    data = json.loads(publish.write_manifest(post, out_dir=tmp_path)
                      .read_text(encoding="utf-8"))
    assert Path(data["hero"]["path"]).name == "hero.png"
    assert json.loads(publish.write_manifest(a_post(tmp_path), out_dir=tmp_path)
                      .read_text(encoding="utf-8"))["hero"] is None


def test_the_hero_is_not_counted_as_a_body_figure(tmp_path):
    post = a_post(tmp_path, hero=a_hero(tmp_path))
    assert len(post.figures) == 1, "the cover is not evidence and carries no caption"


def test_audit_wants_alt_text_on_the_hero(tmp_path):
    hero = a_hero(tmp_path)
    post = a_post(tmp_path, hero=Figure(hero.path, alt="cover", caption=""))
    assert any("no usable alt text" in p for p in post.audit())


def test_audit_catches_a_hero_that_is_also_in_the_body(tmp_path):
    post = a_post(tmp_path)
    post.hero = post.figures[0]
    assert any("appears twice" in p for p in post.audit())


# ---------------------------------------------------------------- the series

@pytest.mark.parametrize("path", LECTURES, ids=lambda p: p.stem)
def test_every_lecture_builds_a_hero_from_the_shared_template(path):
    """Read the source rather than run it: the point is that episode four cannot
    ship without a cover, and that it cannot roll its own."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = {ast.unparse(n.func) for n in ast.walk(tree)
             if isinstance(n, ast.Call)}
    assert "charts.lecture_hero" in calls, (
        f"{path.name} has no cover — every episode needs one, from the shared "
        f"template so the set looks like a set")
    assert "charts.strip_card" not in calls, (
        f"{path.name} calls strip_card directly, which skips the series badge "
        f"and the label check")
    assigns = {ast.unparse(t) for n in ast.walk(tree)
               if isinstance(n, ast.Assign) for t in n.targets}
    assert "post.hero" in assigns, (
        f"{path.name} builds a cover and never attaches it, which is exactly how "
        f"the attribute went unread for months")
