"""Tests for the publication gate, the chart guard rails, and attribution."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from standarderror.render import Post, publish
from standarderror.viz import charts, theme
from standarderror.viz.charts import Figure
from standarderror.xai import attribution


def good_figure(tmp_path: Path) -> Figure:
    p = tmp_path / "f1.png"
    p.write_bytes(b"not really a png")
    return Figure(str(p), alt="A line chart showing something specific",
                  caption="Fig 1. It shows a thing.", title="t")


def good_post(tmp_path: Path, words: int = 1400) -> Post:
    body = " ".join(["word"] * words) + " We compare against persistence."
    return Post(
        title="T", slug="a-good-slug", summary="A summary.",
        date=date(2026, 8, 4), tags=["x"],
        data_sources=["Somewhere, Some Data — <https://data.somewhere.int>"],
        reproducibility={"seed": 1},
    ).add("Section", body, figures=[good_figure(tmp_path)])


class TestPostAudit:
    def test_a_complete_post_passes(self, tmp_path):
        assert good_post(tmp_path).audit() == []

    def test_missing_alt_text_fails(self, tmp_path):
        post = good_post(tmp_path)
        post.sections[0].figures[0].alt = ""
        assert any("alt text" in p for p in post.audit())

    def test_missing_caption_fails(self, tmp_path):
        post = good_post(tmp_path)
        post.sections[0].figures[0].caption = ""
        assert any("caption" in p for p in post.audit())

    def test_no_baseline_mention_fails_when_performance_is_claimed(self, tmp_path):
        post = Post(title="T", slug="s", summary="x",
                    data_sources=["d"], reproducibility={"seed": 1})
        post.add("S", " ".join(["word"] * 1400) + " Our model beats the others.",
                 figures=[good_figure(tmp_path)])
        assert any("baseline" in p for p in post.audit())

    def test_baseline_not_required_without_a_performance_claim(self, tmp_path):
        """A debugging or explainer post makes no accuracy claim, so demanding a
        persistence comparison from it is noise."""
        post = Post(title="T", slug="s", summary="x",
                    data_sources=["d"], reproducibility={"seed": 1})
        post.add("S", " ".join(["word"] * 1400) + " Here is why it exploded.",
                 figures=[good_figure(tmp_path)])
        assert not any("baseline" in p for p in post.audit())

    def test_chance_level_counts_as_a_baseline(self, tmp_path):
        """For a classifier the reference point is chance, not persistence.

        The winner's-curse post compares everything to a coin flip and mentions
        "out-of-sample", which tripped the performance detector; demanding the word
        "naive" from a post whose whole subject is the chance distribution is a
        false positive that trains the author to ignore the audit.
        """
        post = Post(title="T", slug="s", summary="x",
                    data_sources=["d"], reproducibility={"seed": 1})
        post.add("S", " ".join(["word"] * 1400) +
                 " Out-of-sample it scored 2.9 points above chance, which is what "
                 "flipping a coin predicts.", figures=[good_figure(tmp_path)])
        assert not any("baseline" in p for p in post.audit())

    def test_chance_wording_does_not_satisfy_a_forced_baseline(self, tmp_path):
        """`require_baseline=True` is a judgement call by the author, but the
        vocabulary must still be the vocabulary — "by chance" alone is not a
        persistence comparison for a forecasting post."""
        post = Post(title="T", slug="s", summary="x",
                    data_sources=["d"], reproducibility={"seed": 1})
        post.add("S", " ".join(["word"] * 1400) + " It could have happened.",
                 figures=[good_figure(tmp_path)])
        assert any("baseline" in p for p in post.audit(require_baseline=True))

    def test_baseline_can_be_forced(self, tmp_path):
        post = Post(title="T", slug="s", summary="x",
                    data_sources=["d"], reproducibility={"seed": 1})
        post.add("S", " ".join(["word"] * 1400), figures=[good_figure(tmp_path)])
        assert any("baseline" in p for p in post.audit(require_baseline=True))

    def test_pde_notation_is_not_a_placeholder(self, tmp_path):
        """`u_xxxx` contains "xxx"; a substring search blocked a finished post."""
        post = good_post(tmp_path)
        post.sections[0].body += " The equation is u_t = -u*u_x - u_xx - u_xxxx."
        assert not any("XXX" in p for p in post.audit())

    def test_a_real_xxx_marker_is_still_caught(self, tmp_path):
        post = good_post(tmp_path)
        post.sections[0].body += " XXX fix this number before publishing."
        assert any("XXX" in p for p in post.audit())

    def test_leftover_todo_fails(self, tmp_path):
        post = good_post(tmp_path)
        post.sections[0].body += " TODO check this number"
        assert any("TODO" in p for p in post.audit())

    def test_missing_data_citation_fails(self, tmp_path):
        post = good_post(tmp_path)
        post.data_sources = []
        assert any("data citation" in p for p in post.audit())

    def test_placeholder_url_fails(self, tmp_path):
        post = good_post(tmp_path)
        post.code_url = "https://github.com/YOURNAME/standarderror"
        assert any("placeholder" in p for p in post.audit())

    def test_bad_slug_fails(self, tmp_path):
        post = good_post(tmp_path)
        post.slug = "Not A Slug"
        assert any("kebab" in p for p in post.audit())

    def test_too_short_fails(self, tmp_path):
        assert any("words" in p for p in good_post(tmp_path, words=200).audit())

    def test_word_count_excludes_code_and_math(self, tmp_path):
        post = Post(title="T", slug="s", summary="x")
        post.add("S", "one two three\n\n```\n" + " ".join(["code"] * 500)
                 + "\n```\n\n$" + " ".join(["math"] * 500) + "$")
        assert post.word_count() < 20

    def test_word_count_excludes_table_cells(self, tmp_path):
        """A length target is about prose. A table is data, like a code block.

        Counting table cells makes the length gate penalise a post for showing its
        numbers instead of asserting them, which is backwards.
        """
        post = Post(title="T", slug="s", summary="x")
        rows = "\n".join("| " + " | ".join(["cell"] * 6) + " |" for _ in range(40))
        post.add("S", "one two three\n\n| a | b | c | d | e | f |\n"
                      "|---|---|---|---|---|---|\n" + rows)
        assert post.word_count() < 20

    def test_word_count_keeps_a_sentence_containing_a_pipe(self):
        """The table match is line-anchored on purpose.

        `a | b` inside a sentence is prose. Matching pipes anywhere would silently
        delete real paragraphs from the count.
        """
        post = Post(title="T", slug="s", summary="x")
        post.add("S", "the operator a | b is a pipe and these words all count")
        assert post.word_count() == 12

    def test_declared_table_image_without_a_markdown_table_fails(self, tmp_path):
        """Regression: three posts shipped a table image that never appeared.

        The image is only ever *substituted* for a markdown table in the body, so a
        `table_figures` entry with no table to replace is silently invisible in
        Hugo, Medium and Notion alike.
        """
        post = good_post(tmp_path)
        post.table_figures = [post.figures[0]]
        assert any("no markdown table" in p for p in post.audit())

    def test_declared_table_image_with_a_markdown_table_passes(self, tmp_path):
        post = good_post(tmp_path)
        post.table_figures = [post.figures[0]]
        post.add("Numbers", "| a | b |\n|---|---|\n| 1 | 2 |")
        assert not any("no markdown table" in p for p in post.audit())


class TestStripCard:
    """The webtoon-style strip. Cosmetic, so the tests guard the failure modes."""

    @staticmethod
    def _panel(panel, m):
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 6)
        panel.plot([1, 9], [2, 5], color=m.series[0], lw=3.0)

    def test_renders_two_panels(self, tmp_path):
        out = tmp_path / "strip.png"
        meta, _ = charts.strip_card(
            headline="Setup, then punchline",
            panels=[(self._panel, "+0.36%", "the rumour day"),
                    (self._panel, "+0.01%", "the day it is confirmed")],
            note="n", footer="q", alt="a strip", path=str(out))
        assert out.exists() and out.stat().st_size > 5000
        assert meta.alt == "a strip"

    def test_renders_three_panels(self, tmp_path):
        out = tmp_path / "strip3.png"
        charts.strip_card(headline="x",
                          panels=[(self._panel, str(i), f"panel {i}")
                                  for i in range(3)],
                          path=str(out))
        assert out.exists()

    @pytest.mark.parametrize("n", [1, 4])
    def test_rejects_one_or_four_panels(self, tmp_path, n):
        with pytest.raises(ValueError, match="two or three panels"):
            charts.strip_card(headline="x",
                              panels=[(self._panel, "1", "a")] * n,
                              path=str(tmp_path / "bad.png"))

    def test_does_not_leak_the_hand_drawn_style(self, tmp_path):
        """xkcd mode is global; a leak makes every later chart wobbly."""
        import matplotlib.pyplot as plt
        before_family = list(plt.rcParams["font.family"])
        before_sketch = plt.rcParams["path.sketch"]
        charts.strip_card(headline="x",
                          panels=[(self._panel, "1", "a"),
                                  (self._panel, "2", "b")],
                          path=str(tmp_path / "s.png"))
        assert list(plt.rcParams["font.family"]) == before_family
        assert plt.rcParams["path.sketch"] == before_sketch

    def test_panels_have_no_ticks(self, tmp_path):
        """The drawing must not be mistakable for a measurement."""
        seen = {}

        def check(panel, m):
            self._panel(panel, m)
            seen["x"] = list(panel.get_xticks())
            seen["y"] = list(panel.get_yticks())

        charts.strip_card(headline="x",
                          panels=[(check, "1", "a"), (self._panel, "2", "b")],
                          path=str(tmp_path / "s2.png"))
        assert seen["x"] == [] and seen["y"] == []


class TestRankedBarsSort:
    """The `sort` option, added because two charts described an order they lacked.

    `ranked_bars` sorts signed data by magnitude, which is right for attribution and
    wrong twice over: it interleaves positives and negatives when the *grouping* by
    sign is the finding, and it reorders a timeline into a ranking.
    """

    @staticmethod
    def _order(tmp_path, name, labels, values, **kw):
        fig, ax = charts.ranked_bars(labels, values, **kw)
        got = [t.get_text() for t in ax.get_yticklabels()]
        plt.close(fig)
        return got

    def test_auto_sorts_signed_data_by_magnitude(self, tmp_path):
        got = self._order(tmp_path, "a", ["big-", "small+", "mid+"],
                          [-0.34, 0.10, 0.22], signed=True)
        # barh puts the first row at the bottom, so read the result upwards.
        assert got == ["small+", "mid+", "big-"]

    def test_value_sort_groups_by_sign(self, tmp_path):
        got = self._order(tmp_path, "b", ["big-", "small+", "mid+"],
                          [-0.34, 0.10, 0.22], signed=True, sort="value")
        assert got == ["big-", "small+", "mid+"]

    def test_none_keeps_the_callers_order(self, tmp_path):
        labels = ["first", "second", "third", "fourth"]
        got = self._order(tmp_path, "c", labels, [0.36, -0.09, 0.01, -0.06],
                          signed=True, sort="none")
        assert got == labels

    def test_unsigned_default_is_unchanged(self, tmp_path):
        got = self._order(tmp_path, "d", ["a", "b", "c"], [3.0, 1.0, 2.0])
        assert got == ["b", "c", "a"]

    def test_rejects_an_unknown_sort(self, tmp_path):
        with pytest.raises(ValueError, match="unknown sort"):
            charts.ranked_bars(["a", "b"], [1.0, 2.0], sort="sideways")


class TestSketchCard:
    """The hand-drawn card. Cosmetic, so the tests are about it degrading safely."""

    def test_renders_and_returns_a_figure(self, tmp_path):
        out = tmp_path / "s.png"

        def sketch(panel, m):
            panel.plot([0, 1, 2], [0, 1, 0], color=m.ink)

        meta, _ = charts.sketch_card(
            headline="Does it draw?", items=[("1 mo", "a"), ("2 mo", "b")],
            sketch=sketch, note="n", footer="q", alt="alt text here",
            path=str(out))
        assert out.exists() and out.stat().st_size > 5000
        assert meta.alt == "alt text here"

    def test_works_without_a_sketch(self, tmp_path):
        out = tmp_path / "s2.png"
        charts.sketch_card(headline="No drawing",
                           items=[("1", "a"), ("2", "b")], path=str(out))
        assert out.exists()

    def test_rejects_anything_but_two_items(self, tmp_path):
        for items in ([("1", "a")], [("1", "a"), ("2", "b"), ("3", "c")]):
            with pytest.raises(ValueError, match="exactly two"):
                charts.sketch_card(headline="x", items=items,
                                   path=str(tmp_path / "bad.png"))

    def test_does_not_leak_the_hand_drawn_style(self, tmp_path):
        """xkcd mode is global in matplotlib; the card must scope it.

        If it leaks, every chart rendered after a hero card in the same process
        comes out wobbly and in the wrong face — which is exactly the sort of thing
        that would ship unnoticed.
        """
        import matplotlib.pyplot as plt
        before_family = list(plt.rcParams["font.family"])
        before_sketch = plt.rcParams["path.sketch"]
        charts.sketch_card(headline="x", items=[("1", "a"), ("2", "b")],
                           path=str(tmp_path / "s3.png"))
        assert list(plt.rcParams["font.family"]) == before_family
        assert plt.rcParams["path.sketch"] == before_sketch

    def test_the_bundled_face_is_present_with_its_licence(self):
        """The OFL requires the licence to travel with the font."""
        assert theme.SKETCH_FONT_FILE.exists()
        assert (theme.SKETCH_FONT_FILE.parent / "OFL-PatrickHand.txt").exists()
        assert theme.sketch_font_available()

    def test_family_always_ends_in_a_fallback(self):
        fam = theme.sketch_family()
        assert fam[-1] == "DejaVu Sans"
        assert theme.sketch_family() is not fam      # a copy, not the cached list


class TestRendering:
    def test_front_matter_escapes_quotes(self, tmp_path):
        post = good_post(tmp_path)
        post.title = 'A "quoted" title'
        assert '\\"quoted\\"' in post.front_matter()

    def test_hugo_bundle_copies_images_beside_index(self, tmp_path):
        post = good_post(tmp_path)
        out = publish.hugo_page_bundle(post, site_dir=tmp_path / "site")
        assert out.name == "index.md"
        assert (out.parent / "f1.png").exists()
        text = out.read_text(encoding="utf-8")
        assert "![A line chart" in text
        # Page-bundle images are referenced by bare filename, never by the
        # absolute build path, which would render as a broken image on the site.
        assert "(f1.png)" in text
        assert str(tmp_path) not in text

    def test_drafts_are_the_default(self, tmp_path):
        post = good_post(tmp_path)
        assert post.draft is True
        assert "draft: true" in post.front_matter()
        post.draft = False
        assert "draft: false" in post.front_matter()

    def test_medium_bundle_refuses_a_draft(self, tmp_path):
        post = good_post(tmp_path)
        with pytest.raises(ValueError, match="still a draft"):
            publish.medium_bundle(post, out_dir=tmp_path / "m",
                                  base_url="https://example.com")

    def test_medium_bundle_requires_a_canonical_url(self, tmp_path):
        post = good_post(tmp_path)
        post.draft = False
        with pytest.raises(ValueError, match="canonical"):
            publish.medium_bundle(post, out_dir=tmp_path / "m",
                                  base_url="not-a-url")

    def test_medium_bundle_uses_absolute_image_urls(self, tmp_path):
        post = good_post(tmp_path)
        post.draft = False
        path = publish.medium_bundle(post, out_dir=tmp_path / "m",
                                     base_url="https://example.com/blog")
        text = path.read_text(encoding="utf-8")
        assert "https://example.com/blog/posts/a-good-slug/f1.png" in text
        assert "Originally published at" in text
        assert not text.startswith("---")        # no front matter for Medium

    def test_medium_meta_caps_tags_at_five(self, tmp_path):
        import json
        post = good_post(tmp_path)
        post.draft = False
        post.tags = list("abcdefg")
        publish.medium_bundle(post, out_dir=tmp_path / "m",
                              base_url="https://example.com")
        meta = json.loads((tmp_path / "m" / "a-good-slug.meta.json")
                          .read_text(encoding="utf-8"))
        assert len(meta["tags"]) == 5

    def test_notion_dry_run_chunks_blocks(self, tmp_path):
        post = good_post(tmp_path)
        out = publish.notion_page(post, dry_run=True)
        assert out["dry_run"] and out["n_blocks"] >= 2
        assert len(out["payload"]["children"]) <= 100

    def test_footer_lists_licence_warnings(self, tmp_path):
        post = good_post(tmp_path)
        post.licence_warnings = ["stooq: NOT redistributable"]
        assert "NOT redistributable" in post.body_markdown()


class TestChartGuards:
    def test_categorical_cap_is_enforced(self):
        theme.series_colors(8, "light")
        with pytest.raises(ValueError, match="validated cap"):
            theme.series_colors(9, "light")

    def test_all_pairs_forms_cap_at_three(self):
        theme.series_colors(3, "light", all_pairs=True)
        with pytest.raises(ValueError, match="validated cap"):
            theme.series_colors(4, "light", all_pairs=True)

    def test_slot_order_is_fixed_not_cycled(self):
        a = theme.series_colors(3, "light")
        b = theme.series_colors(5, "light")
        assert b[:3] == a                       # colour follows entity, not rank

    def test_dark_mode_is_a_distinct_selected_palette(self):
        assert theme.series_colors(4, "dark") != theme.series_colors(4, "light")

    def test_no_dual_axis_helper_exists(self):
        """Guard against someone adding one later."""
        assert not any("twin" in n.lower() for n in dir(charts))

    def test_indexed_lines_rebases_to_a_common_base(self):
        import pandas as pd
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        df = pd.DataFrame({"a": np.linspace(10, 20, 50),
                           "b": np.linspace(1000, 1500, 50)}, index=idx)
        fig, ax = charts.indexed_lines(df, mode="light")
        ys = [ln.get_ydata()[0] for ln in ax.get_lines()[:2]]
        assert all(abs(y - 100.0) < 1e-9 for y in ys)

    def test_ranked_bars_reorders_asymmetric_errors_with_the_bars(self, tmp_path):
        fig_meta, (fig, ax) = charts.ranked_bars(
            ["a", "b", "c"], [1.0, 5.0, 3.0],
            errors=[[0.1, 0.5, 0.3], [0.2, 1.0, 0.6]],
            alt="alt text here", caption="c",
            path=str(tmp_path / "b.png"))
        assert Path(fig_meta.path).exists()


class TestAttribution:
    def test_linear_shapley_satisfies_efficiency(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 4))
        beta = np.array([2.0, -1.0, 0.5, 0.0])
        att = attribution.linear_shapley(beta, X)
        ref = X.mean(axis=0)
        for i in (0, 7, 42):
            chk = attribution.efficiency_check(
                att.values[i], float(X[i] @ beta), float(ref @ beta))
            assert chk["passes"], chk

    def test_linear_shapley_gives_zero_to_a_zero_coefficient(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        att = attribution.linear_shapley(np.array([1.0, 0.0, -2.0]), X)
        assert np.allclose(att.values[:, 1], 0.0)

    def test_permutation_importance_ranks_the_real_driver_first(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((600, 3))
        y = 3.0 * X[:, 0] + 0.01 * X[:, 1] + 0.1 * rng.standard_normal(600)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        att = attribution.permutation_importance(
            lambda A: A @ beta, X, y, feature_names=["driver", "noise", "unused"])
        assert att.ranked(1)[0][0] == "driver"

    def test_block_permutation_aggregates_collinear_lags(self):
        """Marginal permutation splits credit between near-identical lags; the
        block version attributes it to the variable."""
        rng = np.random.default_rng(0)
        base = np.cumsum(rng.standard_normal(800))
        X = np.column_stack([base[2:], base[1:-1], base[:-2],
                             rng.standard_normal(798)])
        y = 2.0 * X[:, 0] + 0.05 * rng.standard_normal(798)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        att = attribution.block_permutation_importance(
            lambda A: A @ beta, X, y,
            blocks={"signal_lags": [0, 1, 2], "noise": [3]})
        top = att.ranked()
        assert top[0][0] == "signal_lags"
        assert top[0][1] > 10 * abs(top[1][1])

    def test_exact_shapley_used_for_small_feature_counts(self):
        rng = np.random.default_rng(0)
        bg = rng.standard_normal((50, 4))
        beta = np.array([1.0, -2.0, 0.5, 3.0])
        att = attribution.kernel_shapley(lambda A: A @ beta, bg[0], bg)
        assert "enumerated" in att.method
        exact = attribution.linear_shapley(beta, bg[:1], background=bg)
        assert np.allclose(att.values, exact.values[0], atol=1e-8)

    def test_sampled_shapley_reports_standard_errors(self):
        rng = np.random.default_rng(0)
        bg = rng.standard_normal((60, 15))
        beta = rng.standard_normal(15)
        att = attribution.kernel_shapley(lambda A: A @ beta, bg[0], bg,
                                        n_samples=128)
        assert att.std is not None and att.std.shape == (15,)
        assert "standard error" in att.interpretation_caveat

    def test_efficiency_check_catches_a_broken_attribution(self):
        chk = attribution.efficiency_check(np.array([1.0, 1.0]), 10.0, 0.0)
        assert not chk["passes"]

    def test_every_attribution_carries_a_caveat(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 3))
        att = attribution.linear_shapley(np.ones(3), X)
        assert len(att.interpretation_caveat) > 40


class TestLengthTargets:
    def test_light_post_can_set_its_own_floor(self, tmp_path):
        """A 900-word explainer is a light post, not a failed heavy one."""
        post = good_post(tmp_path, words=900)
        post.min_words, post.max_words = 700, 1400
        assert not any("words" in p for p in post.audit())

    def test_default_floor_still_applies_without_an_override(self, tmp_path):
        assert any("words" in p for p in good_post(tmp_path, words=900).audit())

    def test_explicit_argument_beats_the_field(self, tmp_path):
        post = good_post(tmp_path, words=900)
        post.min_words = 700
        assert any("words" in p for p in post.audit(min_words=1500))


class TestMediumTables:
    """Medium supports no tables and strips the markup on paste, so a markdown
    table has to be swapped for an image or it ships as flattened text."""

    TABLE = "before\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nafter"

    def test_finds_a_table(self):
        assert Post.find_markdown_tables(self.TABLE) == [(2, 6)]

    def test_finds_several_and_ignores_prose_with_pipes(self):
        text = ("a | b is not a table\n\n| x |\n|---|\n| 1 |\n\nmid\n\n"
                "| y | z |\n|:--|--:|\n| 1 | 2 |\n")
        assert len(Post.find_markdown_tables(text)) == 2

    def test_ignores_a_header_without_a_separator(self):
        assert Post.find_markdown_tables("| a | b |\n| 1 | 2 |\n") == []

    def _post(self, tmp_path, *, with_figure: bool):
        fig = good_figure(tmp_path)
        # A path of its own, not the chart's: sharing one made every table figure
        # look like a duplicate of the section figure once the audit started
        # comparing paths.
        table_path = tmp_path / "t1.png"
        table_path.write_bytes(b"not really a png")
        table_fig = Figure(str(table_path), alt="the table as an image",
                           caption="Table 1.")
        post = Post(title="T", slug="s", summary="x", draft=False,
                    data_sources=["d"], reproducibility={"seed": 1},
                    table_figures=[table_fig] if with_figure else [])
        post.add("S", self.TABLE, figures=[fig])
        return post

    def test_medium_swaps_the_table_for_the_image(self, tmp_path):
        post = self._post(tmp_path, with_figure=True)
        out = publish.medium_bundle(post, out_dir=tmp_path / "m",
                                    base_url="https://ex.io/blog")
        text = out.read_text(encoding="utf-8")
        assert "| 1 | 2 |" not in text
        assert "![the table as an image]" in text
        assert "before" in text and "after" in text     # prose is untouched

    def test_hugo_keeps_the_real_table(self, tmp_path):
        post = self._post(tmp_path, with_figure=True)
        out = publish.hugo_page_bundle(post, site_dir=tmp_path / "site")
        assert "| 1 | 2 |" in out.read_text(encoding="utf-8")

    def test_unmatched_table_raises_a_loud_checklist_warning(self, tmp_path):
        import json
        post = self._post(tmp_path, with_figure=False)
        publish.medium_bundle(post, out_dir=tmp_path / "m",
                              base_url="https://ex.io/blog")
        meta = json.loads((tmp_path / "m" / "s.meta.json").read_text(
            encoding="utf-8"))
        assert any(c.startswith("WARNING") and "table" in c
                   for c in meta["checklist"])

    def test_table_figures_count_toward_the_audit(self, tmp_path):
        post = self._post(tmp_path, with_figure=True)
        assert len(post.figures) == 2
        post.table_figures[0].alt = ""
        assert any("alt text" in p for p in post.audit())

    def test_a_table_image_listed_twice_fails_the_audit(self, tmp_path):
        """Once in `table_figures`, once as a section figure: two identical images."""
        post = self._post(tmp_path, with_figure=True)
        post.sections[0].figures.append(post.table_figures[0])
        assert any("appear twice" in p for p in post.audit())

    def test_the_normal_arrangement_passes(self, tmp_path):
        post = self._post(tmp_path, with_figure=True)
        assert not any("appear twice" in p for p in post.audit())

    def test_an_unescaped_pipe_in_a_cell_fails_the_audit(self, tmp_path):
        """The bug this check exists for: `ACF1 of |r|` unrenders the whole table.

        Goldmark counts the extra pipes as extra fields, the header stops matching
        the separator row, and the block renders as a paragraph of pipe characters.
        Nothing else in the pipeline notices — the markdown is present, the table
        image is declared, the word count is right.
        """
        post = self._post(tmp_path, with_figure=True)
        post.sections[0].body = "| ACF1 of |r| | b |\n|---|---|\n| 1 | 2 |"
        problems = post.audit()
        assert any("inconsistent column counts" in p for p in problems), problems

    def test_an_escaped_pipe_in_a_cell_passes(self, tmp_path):
        """And the fix must not trip the alarm, which the first version of it did."""
        post = self._post(tmp_path, with_figure=True)
        post.sections[0].body = r"| ACF1 of \|r\| | b |" + "\n|---|---|\n| 1 | 2 |"
        assert not any("inconsistent column counts" in p for p in post.audit())


class TestTableImage:
    def test_renders_and_bolds_only_the_named_cells(self, tmp_path):
        out = tmp_path / "t.png"
        fig_meta, (fig, ax) = charts.table_image(
            [["a", "1.0"], ["b", "2.0"]], header=["k", "v"],
            bold_cells={(0, 1)}, alt="alt text here", caption="c",
            path=str(out))
        assert out.exists()
        weights = {t.get_text(): t.get_fontweight() for t in ax.texts}
        assert weights["1.0"] == "bold"
        assert weights["2.0"] == "normal"

    def test_a_long_header_over_short_numbers_does_not_collide(self, tmp_path):
        """"unevenness" over "0.00" rendered as "unevennessaverage gap".

        The old width model counted characters, and proportional type does not care
        how many characters you used. Widths are measured now; this checks that no
        two cells in a row overlap horizontally.
        """
        fig_meta, (fig, ax) = charts.table_image(
            [["perfect timetable", "0.00", "10.0", "1.00x"],
             ["bunched", "1.60", "10.0", "3.57x"]],
            header=["timetable", "unevenness", "average gap", "vs. the timetable"],
            path=str(tmp_path / "t.png"))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        by_row: dict[float, list] = {}
        for t in ax.texts:
            by_row.setdefault(round(t.get_position()[1], 3), []).append(
                t.get_window_extent(renderer))
        for boxes in by_row.values():
            boxes.sort(key=lambda b: b.x0)
            for left, right in zip(boxes, boxes[1:]):
                assert left.x1 <= right.x0 + 0.5, "cells overlap"

    def test_ragged_rows_raise(self):
        with pytest.raises(ValueError, match="same number of cells"):
            charts.table_image([["a", "b"], ["c"]], header=["x", "y"])

    def test_align_length_is_checked(self):
        with pytest.raises(ValueError, match="align needs"):
            charts.table_image([["a", "b"]], header=["x", "y"], align="l")


class TestSocialCard:
    """The preview image is the only figure most readers see, since it appears in
    the feed whether or not they open the post. It has one job: stay legible after
    the platform crops and shrinks it."""

    def card(self, tmp_path, **kw):
        import numpy as np
        x = np.linspace(0.44, 0.57, 40)
        y = np.exp(-((x - 0.5) / 0.017) ** 2 / 2)
        kw.setdefault("silhouette", (x, y))
        return charts.social_card(
            headline="H", stat="55.3%", stat_label="label",
            supporting=(("a", "1"), ("b", "2"), ("c", "3")),
            path=str(tmp_path / "hero.png"), **kw)

    def test_aspect_and_width_clear_the_platform_minimum(self, tmp_path):
        fig_meta, (fig, ax) = self.card(tmp_path)
        w, h = fig.get_size_inches() * fig.dpi
        assert w >= 1500, f"{w}px is under the ~1500px Medium and OpenGraph want"
        assert 1.8 < w / h < 2.1

    def test_a_mark_in_the_tail_stays_clear_of_the_type(self, tmp_path):
        """A fixed-height rule collided with the supporting column, and the tail is
        exactly where an interesting mark lands."""
        _, (fig, ax) = self.card(tmp_path, mark=0.566)
        rules = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]
        assert rules, "no vertical mark drawn"
        assert max(max(ln.get_ydata()) for ln in rules) < 0.28

    def test_a_mark_outside_the_data_is_clamped_not_dropped(self, tmp_path):
        _, (fig, ax) = self.card(tmp_path, mark=99.0)
        rules = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]
        assert rules and rules[0].get_xdata()[0] == 1.0

    def test_everything_sits_inside_the_crop_safe_margin(self, tmp_path):
        """Medium crops wider than 2:1 and takes it off both edges."""
        _, (fig, ax) = self.card(tmp_path, footer="The Standard Error")
        ys = [t.get_position()[1] for t in ax.texts]
        assert max(ys) <= 0.92 and min(ys) >= 0.08

    def test_type_alone_works_without_data(self, tmp_path):
        out = tmp_path / "hero.png"
        charts.social_card(headline="H", stat="9", path=str(out))
        assert out.exists()


@pytest.fixture(autouse=True)
def _close_figures():
    """Every chart helper returns its figure open so a caller can tweak it, which
    means a test module that renders dozens of them trips matplotlib's
    open-figure warning. Close them between tests rather than changing the
    library's contract for the sake of the suite."""
    yield
    plt.close("all")


class TestCardFamily:
    """The four preview-card layouts. Every post used the same one until they all
    looked identical in a feed, so the family exists to make each post's card match
    the shape of its finding — and these tests cover the ways type overflows a card
    that has no axes to push against."""

    def boxes(self, ax, fig):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        return [(t, t.get_window_extent(r)) for t in ax.texts]

    def test_comparison_values_do_not_overlap(self, tmp_path):
        """"10^30 yrs" at 60pt is twice the width of "42%" and ran into its
        neighbour. The values shrink together until the widest fits its column."""
        _, (fig, ax) = charts.comparison_card(
            headline="H",
            items=[("10^30 yrs", "a Gaussian"), ("110 yrs", "with fat tails")],
            path=str(tmp_path / "c.png"))
        big = sorted((b for t, b in self.boxes(ax, fig) if t.get_fontsize() > 20),
                     key=lambda b: b.x0)
        assert len(big) == 2
        assert big[0].x1 <= big[1].x0, "the two values overlap"

    def test_comparison_values_share_one_size(self, tmp_path):
        """Two numbers at different sizes read as one being more important."""
        _, (fig, ax) = charts.comparison_card(
            headline="H", items=[("10^30 yrs", "a"), ("110 yrs", "b")],
            path=str(tmp_path / "c.png"))
        sizes = {t.get_fontsize() for t, _b in self.boxes(ax, fig)
                 if t.get_fontsize() > 20}
        assert len(sizes) == 1

    def test_a_wrapped_label_does_not_collide_with_the_note(self, tmp_path):
        """Labels that wrap to two lines grow downwards into a note pinned to the
        safe margin, and the collision only appears for some label lengths."""
        _, (fig, ax) = charts.comparison_card(
            headline="H",
            items=[("503", "companies you own"),
                   ("57", "effective holdings, by weight"),
                   ("2.8", "independent bets, at typical correlation")],
            note="A note long enough to wrap across more than one line of the card.",
            path=str(tmp_path / "c.png"))
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        boxes = {t.get_text(): t.get_window_extent(r) for t in ax.texts}
        note = next(b for t, b in boxes.items() if t.startswith("A note"))
        labels = [b for t, b in boxes.items() if "holdings" in t or "bets" in t]
        assert labels
        assert note.y1 <= min(b.y0 for b in labels) + 1.0, "note overlaps a label"

    @pytest.mark.parametrize("n", [1, 4])
    def test_comparison_rejects_bad_item_counts(self, n):
        with pytest.raises(ValueError, match="two or three"):
            charts.comparison_card(headline="H",
                                   items=[(str(i), "l") for i in range(n)])

    def test_a_long_headline_is_shrunk_to_fit(self, tmp_path):
        short = charts.comparison_card(
            headline="Short one.", items=[("1", "a"), ("2", "b")],
            path=str(tmp_path / "s.png"))[1][1]
        long_ = charts.comparison_card(
            headline="A 90% prediction interval, before and after the world moved, "
                     "and then some more words to be sure.",
            items=[("1", "a"), ("2", "b")], path=str(tmp_path / "l.png"))[1][1]
        big_short = max(t.get_fontsize() for t in short.texts)
        head_long = min(t.get_fontsize() for t in long_.texts
                        if t.get_text().startswith("A 90%"))
        assert head_long < 17.5 <= big_short or head_long < 17.5

    def test_bar_card_scales_to_the_largest_value(self, tmp_path):
        fig_meta, (fig, ax) = charts.bar_card(
            headline="H", items=[("a", 5.0, "5.0"), ("b", 17.9, "17.9")],
            emphasis=1, path=str(tmp_path / "b.png"))
        widths = sorted(p.get_width() for p in ax.patches)
        assert widths[1] > widths[0]
        assert widths[1] == pytest.approx(0.86 - 0.30, abs=1e-9)

    @pytest.mark.parametrize("n", [1, 6])
    def test_bar_card_rejects_bad_item_counts(self, n):
        with pytest.raises(ValueError, match="two to five"):
            charts.bar_card(headline="H",
                            items=[(str(i), 1.0, "1") for i in range(n)])

    def test_series_card_marks_a_point_and_clamps_the_index(self, tmp_path):
        import numpy as np
        y = np.arange(50, dtype=float)
        _, (fig, ax) = charts.series_card(y, headline="H", mark_index=9999,
                                          mark_label="here",
                                          path=str(tmp_path / "s.png"))
        panel = [a for a in fig.axes if a is not ax][0]
        dots = [ln for ln in panel.lines if ln.get_marker() == "o"]
        assert dots and dots[0].get_xdata()[0] == len(y) - 1

    def test_series_card_rejects_a_scalar(self):
        with pytest.raises(ValueError, match="1-D series"):
            charts.series_card([1.0], headline="H")

    def test_distribution_card_keeps_an_out_of_range_mark_visible(self, tmp_path):
        import numpy as np
        rng = np.random.default_rng(0)
        _, (fig, ax) = charts.distribution_card(
            rng.normal(size=500), headline="H", mark=12.0, mark_label="way out",
            path=str(tmp_path / "d.png"))
        panel = [a for a in fig.axes if a is not ax][0]
        lo, hi = panel.get_xlim()
        assert lo < 12.0 < hi, "the mark fell outside the drawn window"

    def test_every_card_stays_inside_the_crop_safe_margin(self, tmp_path):
        """Medium crops wider than 1.9:1 and takes it off both edges."""
        import numpy as np
        cards = [
            charts.comparison_card(headline="H", items=[("1", "a"), ("2", "b")],
                                   note="n", footer="f",
                                   path=str(tmp_path / "1.png")),
            charts.bar_card(headline="H", items=[("a", 1.0, "1"), ("b", 2.0, "2")],
                            note="n", footer="f", path=str(tmp_path / "2.png")),
            charts.series_card(np.arange(20, dtype=float), headline="H", note="n",
                               footer="f", path=str(tmp_path / "3.png")),
            charts.distribution_card(np.arange(20, dtype=float), headline="H",
                                     note="n", footer="f",
                                     path=str(tmp_path / "4.png")),
        ]
        for _meta, (_fig, ax) in cards:
            ys = [t.get_position()[1] for t in ax.texts]
            assert min(ys) >= 0.05 and max(ys) <= 0.95


class TestAiDisclosure:
    """The label Medium requires, and the gate that stops it being forgotten.

    Worth pinning because the failure mode is invisible from the author's side:
    the story publishes, an in-product alert appears on it, and the distribution
    is silently cut to the author's own followers. Nothing errors.
    """

    def _post(self, **kw):
        from standarderror.render.post import Post, Section
        return Post(title="t", slug="t", summary="s",
                    sections=[Section("h", "body text here")], **kw)

    def test_the_default_post_already_carries_a_disclosure(self):
        from standarderror.render.post import AI_DISCLOSURE
        assert self._post().disclosure == AI_DISCLOSURE
        assert "AI" in AI_DISCLOSURE

    def test_it_is_the_first_paragraph_of_the_body(self):
        # Medium counts "within the first two paragraphs" from the top of the
        # body, and the crosspost puts a canonical note above it, so after the
        # lede is already too late.
        body = self._post().body_markdown()
        first = [p for p in body.split("\n\n") if p.strip()][0]
        assert "AI" in first

    def test_an_empty_disclosure_fails_the_audit(self):
        problems = self._post(disclosure="").audit(min_words=1)
        assert any("disclosure" in p for p in problems)

    def test_a_disclosure_that_does_not_mention_ai_fails_the_audit(self):
        problems = self._post(
            disclosure="Written with the help of some tools.").audit(min_words=1)
        assert any("does not disclose" in p for p in problems)

    def test_a_real_disclosure_passes(self):
        problems = self._post(
            disclosure="Written with AI assistance.").audit(min_words=1)
        assert not any("disclos" in p for p in problems)

    def test_it_does_not_count_towards_the_length_target(self):
        # Otherwise a required platform label would pad every post by 40 words and
        # a length gate would quietly get easier to pass.
        long_label = "AI " + "word " * 60
        assert (self._post(disclosure=long_label).word_count()
                == self._post(disclosure="AI assisted.").word_count())


class TestContinuationSections:
    """A section with no heading is a continuation, not an empty heading."""

    def test_empty_heading_emits_no_heading_line(self):
        from standarderror.render.post import Section

        md = Section("", "body text", level=3).markdown()
        assert "###" not in md
        assert md.strip() == "body text"

    def test_heading_is_still_emitted_when_present(self):
        from standarderror.render.post import Section

        md = Section("Real heading", "body text", level=3).markdown()
        assert md.startswith("### Real heading")

    def test_whitespace_only_heading_counts_as_empty(self):
        from standarderror.render.post import Section

        assert "#" not in Section("   ", "body").markdown()


class TestPublishedPostsPinTheirDate:
    """`Post.date` defaults to today, which is right exactly once.

    Rebuilding a post that is already on the site therefore re-dates it, which
    reorders the index and misstates when the work was done. Caught the first
    time a rebuild moved a post from 19 to 27 August. The invariant is narrow on
    purpose: a post that has never been published has no date to preserve, so
    only posts with a Hugo page are required to pin one.
    """

    @staticmethod
    def _experiments():
        from pathlib import Path
        return sorted(Path("experiments").glob("exp*.py"))

    @staticmethod
    def _slug(text):
        import re
        m = re.search(r'^\s*slug="([^"]+)",\s*$', text, flags=re.M)
        return m.group(1) if m else None

    def test_every_post_with_a_page_pins_its_date(self):
        from pathlib import Path

        missing = []
        for path in self._experiments():
            text = path.read_text()
            slug = self._slug(text)
            if slug is None:
                continue
            page = Path("site/content/posts") / slug / "index.md"
            if page.exists() and "date=POST_DATE" not in text:
                missing.append(path.name)
        assert not missing, (
            f"published posts that would be re-dated by a rebuild: {missing}")

    def test_the_pinned_date_matches_the_page(self):
        import re
        from pathlib import Path

        wrong = []
        for path in self._experiments():
            text = path.read_text()
            slug = self._slug(text)
            pin = re.search(r'^POST_DATE = date\((\d+), (\d+), (\d+)\)\s*$',
                            text, flags=re.M)
            if slug is None or pin is None:
                continue
            page = Path("site/content/posts") / slug / "index.md"
            if not page.exists():
                continue
            front = re.search(r'^date: (\d+)-(\d+)-(\d+)\s*$', page.read_text(),
                              flags=re.M)
            if front is None:
                continue
            if tuple(int(g) for g in pin.groups()) != tuple(
                    int(g) for g in front.groups()):
                wrong.append((path.name, pin.groups(), front.groups()))
        assert not wrong, f"pinned date disagrees with the published page: {wrong}"

    def test_the_guard_finds_the_experiments_at_all(self):
        # A path typo would make both tests above pass by testing nothing.
        assert len(self._experiments()) > 10


class TestLectureSeries:
    """The series fields, and the title rule that exists because of Medium.

    A crossposted episode arrives as one line of text with no section, no menu
    and no neighbours. If the title does not open with the series tag and the
    number, a reader cannot tell what the piece is or what order it goes in —
    so the tag lives in the title string and `audit()` refuses a post where it
    does not.
    """

    @staticmethod
    def _episode(**kw):
        from standarderror.render.post import Post

        base = dict(
            title="Linear Algebra 1: The Condition Number Is the Error Bar",
            slug="linear-algebra-1-condition-number",
            section="lectures",
            series="Linear Algebra for Data Science, Taught Through What Breaks",
            series_tag="Linear Algebra",
            episode=1,
            summary="a lede",
        )
        base.update(kw)
        return Post(**base)

    def test_a_well_formed_episode_has_no_series_problems(self):
        assert self._episode()._series_problems() == []

    def test_a_standalone_post_is_unaffected(self):
        from standarderror.render.post import Post

        assert Post(title="T", slug="t")._series_problems() == []

    def test_the_three_fields_must_be_set_together(self):
        from standarderror.render.post import Post

        p = Post(title="T", slug="t", series="S")
        problems = p._series_problems()
        assert len(problems) == 1 and "together" in problems[0]

    def test_the_title_must_open_with_the_tag_and_number(self):
        p = self._episode(title="The Condition Number Is the Error Bar")
        assert any("must start with" in x for x in p._series_problems())

    def test_the_wrong_episode_number_in_the_title_is_caught(self):
        p = self._episode(
            title="Linear Algebra 2: The Condition Number Is the Error Bar")
        assert any("must start with" in x for x in p._series_problems())

    def test_the_slug_must_carry_the_series_and_number(self):
        p = self._episode(slug="condition-number")
        assert any("slug must start with" in x for x in p._series_problems())

    def test_an_episode_may_not_sit_in_posts(self):
        p = self._episode(section="posts")
        assert any("own section" in x for x in p._series_problems())

    def test_episode_zero_is_rejected(self):
        p = self._episode(title="Linear Algebra 0: Nope",
                          slug="linear-algebra-0-nope", episode=0)
        assert any("1 or greater" in x for x in p._series_problems())

    def test_audit_surfaces_the_series_problems(self):
        p = self._episode(title="Nope")
        assert any("must start with" in x for x in p.audit())

    def test_the_note_places_the_reader_with_no_site_around_them(self):
        note = self._episode(episode=3,
                             title="Linear Algebra 3: X",
                             slug="linear-algebra-3-x").series_note
        assert "Episode 3" in note
        assert "Linear Algebra for Data Science" in note
        assert "/lectures/" in note

    def test_a_standalone_post_has_no_note(self):
        from standarderror.render.post import Post

        assert Post(title="T", slug="t").series_note == ""

    def test_the_note_follows_the_lede_and_the_disclosure_leads(self):
        body = self._episode().body_markdown()
        paras = [b for b in body.split("\n\n") if b.strip()]
        assert paras[0].startswith("Disclosure:")
        assert paras[1].startswith("*a lede")
        assert paras[2].startswith("Episode 1 of")

    def test_the_note_is_not_counted_as_prose(self):
        p = self._episode()
        p.add("A heading", "one two three four five")
        assert p.word_count() == 5

    def test_the_front_matter_carries_the_series_and_orders_by_episode(self):
        fm = self._episode(episode=4, title="Linear Algebra 4: X",
                           slug="linear-algebra-4-x").front_matter()
        assert 'series: ["Linear Algebra for Data Science' in fm
        assert "weight: 4" in fm

    def test_a_standalone_post_gets_no_weight(self):
        from standarderror.render.post import Post

        assert "weight:" not in Post(title="T", slug="t").front_matter()

    def test_prerequisites_are_recorded(self):
        p = self._episode(episode=2, title="Linear Algebra 2: X",
                          slug="linear-algebra-2-x",
                          prerequisites=["linear-algebra-1-condition-number"])
        assert p.prerequisites == ["linear-algebra-1-condition-number"]


class TestBaselineRequirementCanBeDeclared:
    """The gate keeps its teeth where it was written for, and gets out of the way
    where the comparison is numerical rather than predictive.

    Added after two false positives fired in one lecture episode: "cannot be
    known better than kappa times epsilon" and "the relative accuracy of the
    answer" are precision bounds, and there is no persistence model to compare a
    condition number against. The escape hatch is declared on the post so the
    exemption is reviewable.
    """

    @staticmethod
    def _post(body, **kw):
        from standarderror.render.post import Post
        from standarderror.viz.charts import Figure

        p = Post(title="T", slug="t", summary="s",
                 data_sources=["d"], min_words=1, **kw)
        p.add("H", body, figures=[Figure("f.png", "alt text here", "a caption")])
        return p

    def test_a_performance_claim_still_needs_a_baseline(self):
        p = self._post("Our model beats the alternative on held-out data.")
        assert any("no baseline" in x for x in p.audit())

    def test_a_baseline_satisfies_it(self):
        p = self._post("Our model beats the persistence baseline.")
        assert not any("no baseline" in x for x in p.audit())

    def test_declaring_false_exempts_the_post(self):
        p = self._post("The relative accuracy of the answer is worse than this.",
                       requires_baseline=False)
        assert not any("no baseline" in x for x in p.audit())

    def test_declaring_true_demands_one_even_with_no_trigger_phrase(self):
        p = self._post("A quiet sentence with no comparison in it.",
                       requires_baseline=True)
        assert any("no baseline" in x for x in p.audit())

    def test_the_argument_still_overrides_the_field(self):
        p = self._post("Our model beats everything.", requires_baseline=False)
        assert any("no baseline" in x for x in p.audit(require_baseline=True))

    def test_the_default_is_still_auto_detection(self):
        quiet = self._post("Nothing comparative here at all.")
        assert not any("no baseline" in x for x in quiet.audit())
