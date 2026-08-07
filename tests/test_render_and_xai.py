"""Tests for the publication gate, the chart guard rails, and attribution."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantpost.render import Post, publish
from quantpost.viz import charts, theme
from quantpost.viz.charts import Figure
from quantpost.xai import attribution


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
        post.code_url = "https://github.com/YOURNAME/quantpost"
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
        table_fig = Figure(str(tmp_path / "f1.png"), alt="the table as an image",
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
        _, (fig, ax) = self.card(tmp_path, footer="quantpost")
        ys = [t.get_position()[1] for t in ax.texts]
        assert max(ys) <= 0.92 and min(ys) >= 0.08

    def test_type_alone_works_without_data(self, tmp_path):
        out = tmp_path / "hero.png"
        charts.social_card(headline="H", stat="9", path=str(out))
        assert out.exists()
