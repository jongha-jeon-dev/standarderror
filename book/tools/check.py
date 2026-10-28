"""Check a rewritten chapter against its source episode.

Usage: python book/tools/check.py [NN ...]   (default: every chapter written)

For each chapter it reports
  * numbers in the prose that do not appear in the source (allowing the same
    value rounded to fewer decimals, or the same value as a percentage or a
    proportion); integers below 13 and chapter/part references are skipped;
  * source code blocks missing from the chapter, or out of order, and code
    blocks in the chapter that are not in the source;
  * source figures (other than the hero) the chapter does not use.
Exit status 1 if anything was flagged.
"""
import re
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\w.\-])-?\d[\d,]*(?:\.\d+)?|(?<![\w.])\.\d+")
CODE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
REF = re.compile(r"\b(?:Chapters?|Parts?|Figure|Figures|Table|Appendix|Exercise|"
                 r"Exercises|Section|step|steps|Step|Steps)\s+[\dA-Z]+(?:[.\-–]\d+)?"
                 r"(?:(?:,\s*|\s+and\s+|\s+to\s+|\s*[–-]\s*|\s+or\s+)\d+(?:\.\d+)?)*")


def nums(text):
    text = text.replace("\u2212", "-")
    out = []
    for m in NUM.finditer(text):
        s = m.group(0).replace(",", "")
        try:
            out.append((m.group(0), float(s), m.start()))
        except ValueError:
            pass
    return out


def decimals(tok):
    return len(tok.split(".")[1]) if "." in tok else 0


def allowed(tok, x, src_vals):
    d = decimals(tok.replace(",", ""))
    for v in src_vals:
        for w in (v, v * 100, v / 100):
            if abs(round(w, d) - x) < 10 ** (-d - 3) + 1e-12:
                return True
    return False


def check(n):
    src = (BOOK / f"src/ch{n:02d}.md").read_text()
    src += (BOOK / f"src/ch{n:02d}.data.md").read_text()
    ch = (BOOK / f"chapters/ch{n:02d}.md").read_text()
    issues = []
    sc = [c.strip() for c in CODE.findall(src)]
    cc = [c.strip() for c in CODE.findall(ch)]
    for c in sc:
        if c not in cc:
            issues.append("missing code block: " + c.splitlines()[0][:70])
    order = [cc.index(c) for c in sc if c in cc]
    if order != sorted(order):
        issues.append("code blocks out of order")
    for c in cc:
        if c not in sc:
            issues.append("code block not in source: " + c.splitlines()[0][:70])
    figs = set(re.findall(r"(ml\d+-f\d+[\w-]*\.png)", src))
    for f in sorted(figs):
        if f not in ch:
            issues.append("figure not used: " + f)
    src_vals = {v for _, v, _ in nums(CODE.sub(lambda m: m.group(0), src))}
    prose = CODE.sub("", ch)
    prose = re.sub(r"\]\([^)]*\)", "]", prose)          # image paths
    prose = re.sub(r"\{[^}]*\}", "", prose)              # attributes
    prose = re.sub(r"<!--.*?-->", "", prose, flags=re.S)
    prose = REF.sub("", prose)
    for tok, x, pos in nums(prose):
        if x == int(x) and abs(x) < 13 and "." not in tok:
            continue
        if not allowed(tok, x, src_vals):
            ctx = prose[max(0, pos - 50): pos + 30].replace("\n", " ")
            issues.append(f"number {tok!r} not in source: ...{ctx}...")
    return issues


def main():
    ns = [int(a) for a in sys.argv[1:]] or sorted(
        int(p.stem[2:]) for p in (BOOK / "chapters").glob("ch[0-9][0-9].md"))
    bad = 0
    for n in ns:
        iss = check(n)
        print(f"ch{n:02d}: {'ok' if not iss else str(len(iss)) + ' issue(s)'}")
        for i in iss:
            print("   ", i)
        bad += bool(iss)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
