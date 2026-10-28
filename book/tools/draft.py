"""Assemble first drafts of the 22 chapters from the 70 section files.

For chapter C with episodes [e1, e2, ...] it writes
  book/chapters/chCC.md        # title, an intro placeholder, then each episode
                               # as a numbered section (## heading, its body
                               # with headings demoted, its closing sections
                               # removed), cross-references remapped
  book/merge/chCC.closing.md   # the removed closing sections, per section,
                               # for merging into chapter-level ones
Run once; the drafts are then edited by hand.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structure import CHAPTERS, WHERE  # noqa: E402

BOOK = Path(__file__).resolve().parents[1]
CLOSING = ("In practice", "Limits of the evidence", "Key points", "Exercises",
           "Further reading")
CODE = re.compile(r"(```.*?```)", re.S)
LIST = re.compile(r"\b(Chapters?)\s+(\d+(?:(?:,\s*(?:and\s+)?|\s+and\s+|\s+to\s+|\s*[–-]\s*|\s+or\s+)\d+)*)")


def sec(ep):
    c, s = WHERE[ep]
    return f"{c}.{s}"


def remap(text, here):
    def one(m):
        body = m.group(2)
        nums = [int(x) for x in re.findall(r"\d+", body)]
        seps = re.findall(r"\d+(\s*(?:,\s*(?:and\s+)?|\s+and\s+|\s+to\s+|[–-]|\s+or\s+)\s*)", body)
        eps = []
        for i, n in enumerate(nums):
            if i and seps[i - 1].strip() in ("to", "–", "-"):
                eps.extend(range(eps[-1] + 1, n + 1))
            else:
                eps.append(n)
        if any(e < 1 or e > 70 for e in eps):
            return m.group(0)
        refs = []
        for e in eps:
            r = sec(e)
            if r not in refs:
                refs.append(r)
        joiner = " or " if " or " in body else " and "
        if len(refs) == 1:
            return f"Section {refs[0]}"
        return "Sections " + ", ".join(refs[:-1]) + joiner + refs[-1]

    out = []
    for i, part in enumerate(CODE.split(text)):
        if i % 2:
            out.append(part)
            continue
        part = LIST.sub(one, part)
        part = re.sub(r"\bthis chapter\b", "this section", part)
        part = re.sub(r"\bThis chapter\b", "This section", part)
        out.append(part)
    return "".join(out)


def split_closing(body):
    pieces = re.split(r"^(## .+)$", body, flags=re.M)
    main, closing = [pieces[0]], []
    for h, t in zip(pieces[1::2], pieces[2::2]):
        name = h[3:].strip()
        (closing if name in CLOSING else main).append(f"{h}\n{t}" if name in CLOSING or True else "")
    keep = [pieces[0]]
    for h, t in zip(pieces[1::2], pieces[2::2]):
        if h[3:].strip() not in CLOSING:
            keep.append(f"{h}\n{t}")
    clos = [f"{h}\n{t}" for h, t in zip(pieces[1::2], pieces[2::2])
            if h[3:].strip() in CLOSING]
    return "".join(keep), "".join(clos)


def main():
    if any((BOOK / "chapters").glob("ch*.md")) and "--force" not in sys.argv:
        sys.exit("chapters already exist; drafts would overwrite edited chapters "
                 "(pass --force to do it anyway)")
    (BOOK / "chapters").mkdir(exist_ok=True)
    (BOOK / "merge").mkdir(exist_ok=True)
    for c, (title, eps) in CHAPTERS.items():
        draft = [f"# {title}\n\n<!-- INTRO: write the chapter introduction here -->\n"]
        closing = [f"# Closing material for Chapter {c}: {title}\n"]
        for k, ep in enumerate(eps, 1):
            s = (BOOK / f"sections/s{ep:02d}.md").read_text()
            comment = re.match(r"(<!--.*?-->)", s).group(1)
            stitle = re.search(r"^# (.+)$", s, re.M).group(1)
            body = s.split(f"# {stitle}", 1)[1]
            body, clos = split_closing(body)
            body = re.sub(r"^(#{2,5}) ", lambda m: "#" + m.group(1) + " ", body, flags=re.M)
            draft.append(f"## {stitle}\n\n{comment}\n{remap(body, ep).rstrip()}\n")
            closing.append(f"\n# From Section {c}.{k} (episode {ep}): {stitle}\n\n"
                           f"{remap(clos, ep)}")
        (BOOK / f"chapters/ch{c:02d}.md").write_text("\n".join(draft))
        (BOOK / f"merge/ch{c:02d}.closing.md").write_text("".join(closing))
    print("drafts written")


if __name__ == "__main__":
    main()
