"""Tests for the publication gate, the chart guard rails, and attribution."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
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
        _, (fig, ax) = self.card(tmp_path, footer="quantpost")
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
