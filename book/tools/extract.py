"""Extract each ML episode's rendered page into a book source file.

book/src/chNN.md   the episode body: title, summary, sections, code, figures
                   (figure paths bare file names), without the disclosure,
                   the series line, the Next section and the Data block.
book/src/chNN.data.md   the episode's Data and Reproducibility block.
book/figures/chNN/      the episode's figures, copied.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LECT = ROOT / "site/content/lectures"
BOOK = ROOT / "book"


def episodes():
    out = {}
    for d in LECT.glob("ml-*"):
        t = (d / "index.md").read_text()
        m = re.search(r'^title: "ML (\d+): (.*)"$', t, re.M)
        if m:
            out[int(m.group(1))] = (d, m.group(2).replace('\\"', '"'), t)
    return dict(sorted(out.items()))


def split(text):
    body = text.split("\n---\n", 1)[1] if text.startswith("---") else text
    body = re.sub(r"^Disclosure:.*\n+", "", body, flags=re.M)
    body = re.sub(r"^Episode \d+ of \*Machine Learning.*\n+", "", body, flags=re.M)
    main, _, data = body.partition("\n---\n\n### Data")
    main = re.sub(r"\n## Next\n.*", "\n", main, flags=re.S)
    return main.strip() + "\n", ("### Data" + data).strip() + "\n"


def main():
    eps = episodes()
    assert list(eps) == list(range(1, 71)), list(eps)
    for n, (d, title, text) in eps.items():
        main, data = split(text)
        hero = next(d.glob("*-hero.png")).name
        head = (f"<!-- source: {d.name}; hero: {hero} -->\n\n"
                f"# {title}\n\n")
        (BOOK / f"src/ch{n:02d}.md").write_text(head + main)
        (BOOK / f"src/ch{n:02d}.data.md").write_text(data)
        fd = BOOK / f"figures/ch{n:02d}"
        fd.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*.png"):
            shutil.copy2(p, fd / p.name)
    print(len(eps), "episodes extracted")


if __name__ == "__main__":
    main()
