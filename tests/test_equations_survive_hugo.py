r"""Equations have to reach the maths renderer unchanged.

Hugo's markdown renderer runs before KaTeX, and it treats a backslash before
ASCII punctuation as a markdown escape. So `\;` reaches the page as `;`, `\\,`
as `,`, `\\!` as `!`, and `\\|` as `|` -- and a bare `|` then lets a following
`_F` be read as emphasis. Separately, a *wrapped* equation whose continuation
line begins `+ ` becomes a bullet list in the middle of the maths.

Six of the eight published lecture pages had at least one equation damaged this
way, two of them live, and the markdown source looked perfectly correct in every
case. It was only visible in the built HTML. These tests check the built form.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from standarderror.render.post import Post

REPO = Path(__file__).resolve().parents[1]
LECTURES = sorted((REPO / "site" / "content" / "lectures").glob("*/index.md"))

#: A backslash before any of these is a markdown escape, not a LaTeX macro.
FRAGILE = re.compile(r"\\[;,!|\[\]{}_#$%&~^]")
#: What SmartyPants rewrites. It runs before KaTeX and does not know it is
#: looking at maths, so an apostrophe -- the natural way to write a
#: derivative -- comes out as a curly quote KaTeX will not read as a prime.
#: Found in episode 2 of the second series, by diffing a Hugo build again;
#: nothing already in this file would have caught it.
SMARTENED = (chr(39), chr(34), "...", "--")
#: A line markdown would read as a list item or a heading.
LINE_INITIAL = re.compile(r"^\s*([-+*>#]|\d+\.)\s")


def equations(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"\$\$(.+?)\$\$", text, re.S)]


class TestTheHardener:
    def test_it_removes_every_fragile_escape(self):
        got = Post._safe_equation(
            r"\sin \theta \;\le\; \frac{2^{3/2} \, \lVert E \rVert}{g} \!")
        assert not FRAGILE.search(got), got
        assert r"\le" in got and r"\lVert" in got

    def test_it_pairs_a_norm_delimiter_into_macros(self):
        got = Post._safe_equation(r"\min_k \| Y - B \|_F \;=\; \sqrt{x}")
        assert r"\lVert Y - B \rVert_F" in got
        assert "|" not in got

    def test_it_joins_a_wrapped_equation_onto_one_line(self):
        """The continuation line here begins `+ `, which markdown reads as a
        bullet -- so the fix is to remove line starts, not to escape the plus."""
        got = Post._safe_equation("a - \\sqrt{b},\n  + \\sqrt{c}")
        assert "\n" not in got
        assert got == r"a - \sqrt{b}, + \sqrt{c}"

    def test_a_bare_thin_space_before_a_macro_is_handled(self):
        r"""`\;\right]` has no space after the `\;`, which an earlier version of
        the substitution table missed."""
        got = Post._safe_equation(r"\left[ a, b \;\right]")
        assert not FRAGILE.search(got), got

    def test_it_leaves_a_clean_equation_alone(self):
        clean = r"\lVert x \rVert \le \frac{a}{b} \quad \mathrm{tr}(H) = p"
        assert Post._safe_equation(clean) == clean

    def test_it_does_not_touch_code_fences(self):
        body = "text\n\n```python\nx = a \\, b  # $$not an equation$$\n```\n"
        assert Post._harden_equations(body) == body


@pytest.mark.parametrize("page", LECTURES, ids=lambda p: p.parent.name)
class TestEveryPublishedLecture:
    def test_no_equation_carries_a_fragile_escape(self, page):
        for eq in equations(page.read_text(encoding="utf-8")):
            assert not FRAGILE.search(eq), (
                f"{page.parent.name}: {FRAGILE.search(eq).group(0)!r} in "
                f"{' '.join(eq.split())[:110]}")

    def test_no_equation_spans_more_than_one_line(self, page):
        """A wrapped equation is one continuation line away from becoming a
        bullet list, and the wrapping buys nothing on a rendered page."""
        for eq in equations(page.read_text(encoding="utf-8")):
            body = eq.strip("\n")
            assert "\n" not in body, (
                f"{page.parent.name}: wrapped equation "
                f"{' '.join(body.split())[:110]}")

    def test_no_line_in_the_body_starts_a_list_by_accident(self, page):
        """Outside code fences and real lists, a line beginning `+ ` or `* ` is
        almost always a wrapped line that markdown is about to reinterpret."""
        text = page.read_text(encoding="utf-8")
        fence = False
        for n, line in enumerate(text.split("\n"), 1):
            if line.lstrip().startswith("```"):
                fence = not fence
                continue
            if fence or line.startswith("- ") or line.startswith("#"):
                continue
            m = LINE_INITIAL.match(line)
            if m and m.group(1) in {"+", "*"}:
                raise AssertionError(
                    f"{page.parent.name}:{n} starts with {m.group(1)!r}: {line[:80]}")


class TestSmartyPantsRunsOnTheMathsToo:
    """The failure that got past every test above it."""

    def test_the_hardener_turns_apostrophes_into_primes(self):
        eq = "f" + chr(39) + "(x) + f" + chr(39) * 2 + "(x)"
        got = Post._safe_equation(eq)
        assert chr(39) not in got, got
        assert got.count(chr(92) + "prime") == 3, got

    def test_a_triple_prime_is_consumed_before_a_single_one(self):
        got = Post._safe_equation("f" + chr(39) * 3 + "(x)")
        assert got.count(chr(92) + "prime") == 3, got
        assert chr(39) not in got

    def test_it_replaces_an_ellipsis_and_a_dash_run(self):
        got = Post._safe_equation("a + b + ... + z")
        assert "..." not in got and (chr(92) + "dots") in got, got
        assert "--" not in Post._safe_equation("a -- b")

    def test_a_hardened_equation_has_nothing_left_to_smarten(self):
        eq = (chr(92) + "frac{f(x+h) - f(x)}{h} = f" + chr(39)
              + "(x) + O(h^2)")
        got = Post._safe_equation(eq)
        for bad in SMARTENED:
            assert bad not in got, (bad, got)


@pytest.mark.parametrize("page", LECTURES, ids=lambda p: p.parent.name)
class TestNoPublishedEquationIsSmartenable:
    def test_it(self, page):
        for eq in equations(page.read_text(encoding="utf-8")):
            for bad in SMARTENED:
                assert bad not in eq, (
                    f"{page.parent.name}: {bad!r} in "
                    f"{chr(32).join(eq.split())[:110]}")
