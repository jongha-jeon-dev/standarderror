"""Code blocks that are executed at build time, with their real output captured.

A post that shows code and shows what the code printed has two ways to be wrong,
and both are invisible to the reader. The code can be a lightly-edited version of
what was actually run, and the printed numbers can be from an earlier run that a
later change moved. Neither breaks anything at publication time; both are found by
whoever tries to reproduce the post, if anyone ever does.

The fix is to make the post's snippets the only copy. `Session.run` executes the
source it is about to print, in a namespace shared across the post's snippets so
they read as one script, captures stdout, and raises if the code fails. The
experiment script imports the post from the same module that ran it, so a snippet
that no longer runs stops the build instead of shipping.

`expect` closes the second gap. Passing a substring of the output that the post's
prose quotes turns a drift in that number into a build failure rather than a
sentence that no longer matches the block above it.

    s = Session()
    s.run("import numpy as np; print(np.arange(3).sum())", expect="3")

What this deliberately does not do is sandbox anything. The code is the author's
own and runs with the author's imports; the point is reproducibility, not safety.
"""

from __future__ import annotations

import io
import textwrap
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass

__all__ = ["Snippet", "Session", "SnippetError"]


class SnippetError(RuntimeError):
    """A snippet failed to run, or its output no longer contains what was promised."""


@dataclass(frozen=True)
class Snippet:
    """Source and the output it actually produced, ready to render."""

    code: str
    output: str
    language: str = "python"

    def markdown(self) -> str:
        """Two fenced blocks: the code, then what it printed.

        Kept as separate fences rather than one block with `>>>` prompts because a
        reader who copies the block should get something that runs, and because
        the output half is not Python and should not be highlighted as if it were.
        """
        out = [f"```{self.language}", self.code, "```"]
        if self.output:
            out += ["", "```text", self.output, "```"]
        return "\n".join(out)

    @property
    def lines(self) -> int:
        return len(self.code.splitlines())


class Session:
    """One namespace for a post's snippets, executed in order.

    Sharing the namespace is what lets a post define a helper in one block and use
    it three blocks later without re-printing it, which is how the code in a
    readable article is actually laid out. It also means the order of `run` calls
    is load-bearing: a block that depends on an earlier one fails if the earlier
    one is deleted, which is the intended behaviour.
    """

    def __init__(self, *, initial: dict | None = None):
        self.ns: dict = dict(initial or {})
        self.snippets: list[Snippet] = []

    def run(self, code: str, *, expect: str | list[str] | None = None,
            language: str = "python", show: str | None = None) -> Snippet:
        """Execute `code`, capture stdout, and return the pair as a `Snippet`.

        `show` prints a different source than the one executed, for the one case
        where that is honest: a block whose runnable form needs an import or a seed
        that would be noise in the article. Use it sparingly — the whole guarantee
        of this module is that what is printed is what ran, and `show` is the hole
        in it, so the shown source must be a subset of the executed source rather
        than a paraphrase of it.
        """
        src = textwrap.dedent(code).strip("\n")
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compile(src, "<snippet>", "exec"), self.ns)
        except Exception as exc:                      # noqa: BLE001 - re-raised below
            raise SnippetError(
                f"snippet {len(self.snippets) + 1} failed: {type(exc).__name__}: {exc}\n"
                f"--- source ---\n{src}\n--- traceback ---\n{traceback.format_exc()}"
            ) from exc

        output = buf.getvalue().rstrip("\n")
        wanted = [expect] if isinstance(expect, str) else list(expect or [])
        missing = [w for w in wanted if w not in output]
        if missing:
            raise SnippetError(
                f"snippet {len(self.snippets) + 1} ran but its output no longer "
                f"contains {missing!r}.\n--- source ---\n{src}\n"
                f"--- output ---\n{output}")

        shown = textwrap.dedent(show).strip("\n") if show is not None else src
        if show is not None:
            for line in shown.splitlines():
                if line.strip() and line.strip() not in src:
                    raise SnippetError(
                        "`show` must be a subset of the code that ran; this line is "
                        f"not in it: {line.strip()!r}")
        snip = Snippet(shown, output, language)
        self.snippets.append(snip)
        return snip

    def value(self, name: str):
        """Read a name out of the shared namespace, for prose that quotes a result."""
        if name not in self.ns:
            raise SnippetError(f"{name!r} was never defined by a snippet")
        return self.ns[name]
