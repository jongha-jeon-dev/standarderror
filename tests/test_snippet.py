"""Tests for build-time code execution.

The module's whole value is that it fails loudly, so most of these assert on
failures: code that raises, output that no longer says what the prose claims, and
a `show` override that has drifted from what actually ran. A version of this
module that silently swallowed any of the three would still render a post that
looked right.
"""

from __future__ import annotations

import pytest

from standarderror.render.snippet import Session, Snippet, SnippetError


class TestRunning:
    def test_it_captures_what_the_code_printed(self):
        s = Session()
        snip = s.run("print(2 + 2)")
        assert snip.code == "print(2 + 2)"
        assert snip.output == "4"

    def test_the_namespace_carries_across_snippets(self):
        s = Session()
        s.run("x = 41")
        assert s.run("print(x + 1)").output == "42"

    def test_indented_source_is_dedented_before_it_is_shown(self):
        # Snippets are written inside a function in the experiment script, so
        # every one of them arrives with eight spaces of indent that would both
        # break `exec` and look wrong in the post.
        s = Session()
        snip = s.run("""
            total = sum(range(4))
            print(total)
        """)
        assert snip.output == "6"
        assert snip.code.startswith("total =")

    def test_initial_names_are_available(self):
        s = Session(initial={"seed": 7})
        assert s.run("print(seed)").output == "7"

    def test_a_snippet_that_prints_nothing_renders_without_an_output_block(self):
        s = Session()
        snip = s.run("y = 1")
        assert snip.output == ""
        assert "```text" not in snip.markdown()

    def test_snippets_accumulate_in_order(self):
        s = Session()
        s.run("print('a')")
        s.run("print('b')")
        assert [x.output for x in s.snippets] == ["a", "b"]


class TestFailing:
    def test_code_that_raises_stops_the_build(self):
        s = Session()
        with pytest.raises(SnippetError) as e:
            s.run("1 / 0")
        assert "ZeroDivisionError" in str(e.value)
        assert "--- source ---" in str(e.value)

    def test_a_failed_snippet_is_not_recorded(self):
        s = Session()
        with pytest.raises(SnippetError):
            s.run("undefined_name")
        assert s.snippets == []

    def test_output_that_lost_the_promised_number_stops_the_build(self):
        # The point of `expect`: the prose says 0.83, the code changes, the block
        # now prints 0.71, and nothing else in the pipeline would notice.
        s = Session()
        s.run("print('auc 0.83')", expect="0.83")
        with pytest.raises(SnippetError) as e:
            s.run("print('auc 0.71')", expect="0.83")
        assert "0.83" in str(e.value)

    def test_every_missing_expectation_is_reported_at_once(self):
        s = Session()
        with pytest.raises(SnippetError) as e:
            s.run("print('one')", expect=["one", "two", "three"])
        assert "two" in str(e.value) and "three" in str(e.value)

    def test_a_shown_source_that_did_not_run_is_rejected(self):
        s = Session()
        with pytest.raises(SnippetError):
            s.run("x = 1\nprint(x)", show="x = 2\nprint(x)")

    def test_a_shown_source_that_drops_setup_lines_is_allowed(self):
        # The legitimate use: hide the import and the seed, show the three lines
        # that carry the idea. Every shown line really did run.
        s = Session()
        snip = s.run("import math\nprint(round(math.pi, 2))",
                     show="print(round(math.pi, 2))")
        assert snip.code == "print(round(math.pi, 2))"
        assert snip.output == "3.14"

    def test_reading_a_name_that_no_snippet_defined_raises(self):
        with pytest.raises(SnippetError):
            Session().value("nope")


class TestRendering:
    def test_markdown_is_a_code_fence_then_a_text_fence(self):
        md = Snippet("print(1)", "1").markdown()
        assert md.splitlines()[0] == "```python"
        assert "```text" in md
        assert md.count("```") == 4

    def test_the_language_is_carried_into_the_fence(self):
        assert Snippet("SELECT 1", "1", "sql").markdown().startswith("```sql")

    def test_prose_word_count_excludes_the_whole_block(self):
        # `Post.word_count` strips fenced blocks, and a code-forward post lives or
        # dies on that being true for the output fence as well as the code one.
        from standarderror.render.post import Post, Section
        s = Session()
        snip = s.run("print('alpha beta gamma delta')")
        p = Post(title="t", slug="t",
                 sections=[Section("h", "one two three\n\n" + snip.markdown())])
        assert p.word_count() == 3

    def test_a_value_can_be_read_back_for_the_prose(self):
        s = Session()
        s.run("auc = 0.834")
        assert s.value("auc") == 0.834


class TestNotionBlocks:
    """A code-forward post is the first thing to break the naive blank-line split."""

    def test_a_blank_line_inside_a_fence_does_not_cut_the_block_in_two(self):
        from standarderror.render.post import Post, Section
        from standarderror.render.publish import _to_notion_blocks
        body = "```python\nimport numpy as np\n\nprint(np.arange(3).sum())\n```"
        p = Post(title="t", slug="t", sections=[Section("h", body)])
        kinds = [b["type"] for b in _to_notion_blocks(p)]
        assert kinds == ["heading_2", "code"]

    def test_the_output_fence_becomes_a_plain_text_code_block(self):
        # Notion rejects the whole request on an unknown language name, and
        # "text" is not one it knows.
        from standarderror.render.post import Post, Section
        from standarderror.render.publish import _to_notion_blocks
        s = Session()
        snip = s.run("print('4')")
        p = Post(title="t", slug="t", sections=[Section("h", snip.markdown())])
        langs = [b["code"]["language"] for b in _to_notion_blocks(p)
                 if b["type"] == "code"]
        assert langs == ["python", "plain text"]

    def test_prose_on_either_side_of_a_block_survives(self):
        from standarderror.render.post import Post, Section
        from standarderror.render.publish import _to_notion_blocks
        body = "before\n\n```python\nx = 1\n\ny = 2\n```\n\nafter"
        p = Post(title="t", slug="t", sections=[Section("h", body)])
        kinds = [b["type"] for b in _to_notion_blocks(p)]
        assert kinds == ["heading_2", "paragraph", "code", "paragraph"]
