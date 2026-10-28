"""Build the book: PDF (pandoc + XeLaTeX) and EPUB (pandoc).

    python book/tools/build.py [pdf] [epub]      (default: both)

Inputs: book/front/preface.md, book/front/parts.md, book/chapters/chNN.md,
book/figures/chNN/*.png, book/src/chNN.data.md and the series index (for the
appendices). Outputs: book/out/machine-learning-what-breaks.{pdf,epub}.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "book"
OUT = BOOK / "out"
NAME = "machine-learning-what-breaks"

TITLE = "Machine Learning, Taught Through What Breaks"
SUBTITLE = "Seventy sentences from the curriculum, measured"
AUTHOR = "Jongha Jeon"
YEAR = "2026"

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI",
         "XII", "XIII", "XIV"]


def parts():
    text = (BOOK / "front/parts.md").read_text()
    out = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        head, _, body = block.partition("\n")
        num, title = [s.strip() for s in head.split("|")]
        out.append((num, title, body.strip()))
    assert [p[0] for p in out] == ROMAN
    return out


def chapter(n: int, target: str) -> str:
    s = (BOOK / f"chapters/ch{n:02d}.md").read_text()
    hero = re.search(r"hero: (\S+?\.png)", s).group(1)
    s = re.sub(r"<!--.*?-->\s*", "", s, count=1, flags=re.S)
    s = s.replace("# episode 4's task", "# Chapter 4's task")
    s = _typeset_prose(s)
    figdir = f"figures/ch{n:02d}"
    s = re.sub(r"\]\((ml\d+-[\w-]+\.png)\)", lambda m: f"]({figdir}/{m.group(1)})", s)
    # Hero: an unnumbered summary picture right under the chapter title.
    lines = s.split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("# "))
    if target == "pdf":
        hero_md = ("\n```{=latex}\n\\begin{center}\\includegraphics[width=\\linewidth]"
                   f"{{{figdir}/{hero}}}\\end{{center}}\n```\n")
    else:
        hero_md = f'\n<p class="hero"><img src="{figdir}/{hero}" alt="Chapter {n} at a glance"/></p>\n'
    lines.insert(i + 1, hero_md)
    if target == "epub":                    # number chapters only, as in the PDF
        fence = False
        for k, ln in enumerate(lines):
            if ln.startswith("```"):
                fence = not fence
            elif not fence and re.match(r"#{2,6} ", ln) and not ln.rstrip().endswith("}"):
                lines[k] = ln.rstrip() + " {-}"
    return "\n".join(lines)


def _typeset_prose(s: str) -> str:
    """Outside code: '+-' as a plus-minus sign, a hyphen before a number as a
    minus sign."""
    parts = re.split(r"(```.*?```|`[^`\n]*`|\$\$.*?\$\$|\$[^$\n]*\$|\]\([^)]*\)|\{[^}]*\})", s, flags=re.S)
    for i in range(0, len(parts), 2):
        t = parts[i].replace("+-", "\u00b1")
        t = re.sub(r"(^|[\s(\[])-(\d)", "\\1\u2212\\2", t, flags=re.M)
        parts[i] = t
    return "".join(parts)


# ----------------------------------------------------------------- appendices

def appendix_practice() -> str:
    """Every chapter's 'In practice' section, grouped by part."""
    out = ["# Practice, Chapter by Chapter\n",
           "This appendix collects the *In practice* section of every chapter, "
           "grouped by part, as a reference to use on a project. Each list is "
           "the chapter's own; the chapter gives the evidence behind it.\n"]
    for k, (num, title, _) in enumerate(parts()):
        out.append(f"## Part {num}: {title}\n")
        for n in range(5 * k + 1, min(5 * k + 6, 71)):
            s = (BOOK / f"chapters/ch{n:02d}.md").read_text()
            t = re.search(r"^# (.+)$", s, re.M).group(1)
            m = re.search(r"^## In practice\n(.*?)(?=^## )", s, re.M | re.S)
            if not m:
                continue
            body = re.sub(r"^### ", "#### ", m.group(1).strip(), flags=re.M)
            out.append(f"### Chapter {n}: {t}\n\n{body}\n")
    return "\n".join(out)


def appendix_verdicts() -> str:
    idx = (ROOT / "site/content/lectures/_index.md").read_text()
    idx = idx.split("## Machine Learning, Taught Through What Breaks", 1)[1]
    sentences = {}
    for m in re.finditer(r"^\| (\d+) \| [^|]+ \| (.*) \|$", idx, re.M):
        n = int(m.group(1))
        sm = re.match(r"\*(.+?)\*", m.group(2))
        if sm:
            sentences[n] = sm.group(1)
    src = (ROOT / "experiments/ml170_verdicts.py").read_text()
    failed = set(map(int, re.search(r"FAILED = \{([^}]*)\}", src).group(1).replace("\n", " ").split(",")))
    held = set(map(int, re.search(r"HELD = \{([^}]*)\}", src).group(1).split(",")))
    assert len(sentences) == 69 and len(failed) == 38 and held == {3}
    out = ["# The Sixty-Nine Sentences\n",
           "Each chapter from 1 to 69 tested one sentence. This appendix lists "
           "them with the verdict Chapter 70 assigns: *failed* if the sentence "
           "was false in the setting the chapter measured, *conditions* if it "
           "held in some measured settings and not others, *held* if it held "
           "in every setting measured. The classification is the author's "
           "reading of each chapter's conclusion.\n"]
    for k, (num, title, _) in enumerate(parts()):
        rng = range(5 * k + 1, min(5 * k + 6, 70))
        if not len(rng):
            continue
        out.append("```{=latex}\n\\needspace{10\\baselineskip}\n```\n")
        out.append(f"## Part {num}: {title}\n")
        out.append("| Ch. | The sentence | Verdict |\n|----:|:" + "-" * 70 + "|:-----------|")
        for n in rng:
            v = "failed" if n in failed else "held" if n in held else "conditions"
            out.append(f"| {n} | {sentences[n][0].upper() + sentences[n][1:]} | {v} |")
        out.append("")
    return "\n".join(out)


def appendix_data() -> str:
    out = ["# Data and Code\n",
           "All code is in the repository at "
           "<https://github.com/jongha-jeon-dev/standarderror>. Each chapter "
           "is built by one module in `experiments/` (Chapter N by "
           "`ml1NN_*.py`, Chapter 70 by `ml170_verdicts.py`) from the "
           "machinery in `standarderror/ml/`, which is tested in "
           "`tests/test_ml.py`. The data are bundled with scikit-learn or "
           "simulated, so every result can be rebuilt offline. Below, each "
           "chapter's data and machinery.\n"]
    for n in range(1, 71):
        s = (BOOK / f"src/ch{n:02d}.data.md").read_text()
        data = s.split("### Data", 1)[1].split("### Reproducibility", 1)[0].strip()
        data = re.sub(r"^- Where this stops:.*$", "", data, flags=re.M).strip()
        data = data.replace("of the school-maths series", "trained in this repository")
        data = data.replace("The 69 earlier episodes of this series", "The 69 earlier chapters")
        data = data.replace("by the rule stated in the post", "by the rule stated in Chapter 70")
        data = data.replace("episode", "chapter").replace("Episode", "Chapter")
        t = re.search(r"^# (.+)$", (BOOK / f"chapters/ch{n:02d}.md").read_text(), re.M).group(1)
        out.append(f"### Chapter {n}: {t}\n\n{data}\n")
    return "\n".join(out)


# ---------------------------------------------------------------- assembly

def assemble(target: str) -> str:
    pre = (BOOK / "front/preface.md").read_text()
    md = []
    if target == "pdf":
        md.append("```{=latex}\n\\frontmatter\n" + TITLEPAGE + "\\tableofcontents\n```\n")
        md.append(pre)
        md.append("```{=latex}\n\\mainmatter\n```\n")
    else:
        md.append(pre)
    for k, (num, title, intro) in enumerate(parts()):
        if target == "pdf":
            md.append(f"```{{=latex}}\n\\part{{{title}}}\n```\n\n{intro}\n")
        else:
            md.append(f"# Part {num}: {title} {{.unnumbered .part}}\n\n{intro}\n")
        for n in range(5 * k + 1, min(5 * k + 6, 71)):
            md.append(chapter(n, target))
    apps = [appendix_practice(), appendix_verdicts(), appendix_data()]
    if target == "pdf":
        md.append("```{=latex}\n\\appendix\n\\sloppy\n```\n")
        md.extend(apps)
    else:
        for letter, a in zip("ABC", apps):
            a = re.sub(r"^# (.+)$", rf"# Appendix {letter}: \1 {{.unnumbered}}", a,
                       count=1, flags=re.M)
            md.append(a)
    return "\n\n".join(md)


TITLEPAGE = r"""\begin{titlepage}
\centering
\vspace*{4.5cm}
{\fontsize{26}{32}\selectfont\bfseries Machine Learning,\\[0.25em] Taught Through What Breaks\par}
\vspace{1.2cm}
{\large\itshape Seventy sentences from the curriculum, measured\par}
\vfill
{\Large Jongha Jeon\par}
\vspace{0.6em}
{\normalsize 2026\par}
\vspace*{1.5cm}
\end{titlepage}
\thispagestyle{empty}
\vspace*{\fill}
{\small\noindent Copyright \copyright{} 2026 Jongha Jeon.\par\medskip
\noindent Code, data notes and the build that produced every number in this book:
\url{https://github.com/jongha-jeon-dev/standarderror}.\par\medskip
\noindent The handwritten digits and breast cancer data are from the UCI Machine Learning
Repository (CC BY 4.0), as bundled with scikit-learn; the diabetes data are from
Efron, Hastie, Johnstone and Tibshirani (2004), as bundled with scikit-learn.\par}
\clearpage
"""

LATEX_HEADER = r"""
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\},fontsize=\footnotesize}
\fvset{fontsize=\footnotesize}
\usepackage{caption}
\usepackage{needspace}
\captionsetup{font=small,labelfont=bf,margin=0.5em}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE]{\small\nouppercase{\leftmark}}
\fancyhead[RO]{\small\nouppercase{\rightmark}}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0pt}
\makeatletter
\def\@endpart{\vspace{2.5em}}
\makeatother
\setcounter{tocdepth}{0}
\usepackage{newunicodechar}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{√}{\ensuremath{\surd}}
\newunicodechar{₁}{\ensuremath{_1}}
\newunicodechar{₂}{\ensuremath{_2}}
\newunicodechar{₀}{\ensuremath{_0}}
\graphicspath{{pdffig/}}
\setlength{\emergencystretch}{5em}
\raggedbottom
"""


def build_pdf():
    OUT.mkdir(exist_ok=True)
    _small_figures("pdffig", 1600)
    md = OUT / f"{NAME}.pdf.md"
    md.write_text(assemble("pdf"))
    hdr = OUT / "header.tex"
    hdr.write_text(LATEX_HEADER)
    cmd = ["pandoc", str(md), "-o", str(OUT / f"{NAME}.tex"), "--standalone",
           "--from", "markdown",
           "--top-level-division=chapter", "--number-sections",
           "--resource-path", str(BOOK), "--include-in-header", str(hdr),
           "-V", "documentclass=book", "-V", "classoption=openany",
           "-V", "papersize=b5", "-V", "fontsize=10pt",
           "-V", "geometry:top=2cm,bottom=2.2cm,inner=2.2cm,outer=1.8cm",
           "-V", "fontfamily=fontspec", "-V", "mainfont=TeX Gyre Pagella", "-V", "monofont=DejaVu Sans Mono",
           "-V", "monofontoptions=Scale=0.86", "-V", "mathfont=Latin Modern Math",
           "-V", "secnumdepth=0", "-V", "colorlinks=true", "-V", "linkcolor=black",
           "-V", "urlcolor=blue!50!black", "-V", "toccolor=black",
           "-V", f"title-meta={TITLE}", "-V", f"author-meta={AUTHOR}",
           "--highlight-style", "tango"]
    subprocess.run(cmd, check=True, cwd=BOOK)
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                            f"{NAME}.tex"], cwd=OUT, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:])
            sys.exit("xelatex failed")
    log = (OUT / f"{NAME}.log").read_text(errors="ignore")
    missing = sorted(set(re.findall(r"Missing character: There is no (\S+)", log)))
    print("PDF built; missing glyphs:", missing or "none")


EPUB_CSS = """
body { font-family: serif; line-height: 1.45; }
pre, code { font-family: monospace; font-size: 0.82em; }
pre { white-space: pre-wrap; background: #f6f6f4; padding: 0.5em; }
p.hero img, figure img { max-width: 100%; }
figcaption { font-size: 0.9em; font-style: italic; }
table { border-collapse: collapse; font-size: 0.9em; }
td, th { border-bottom: 1px solid #ccc; padding: 0.2em 0.4em; vertical-align: top; }
"""


def build_epub():
    OUT.mkdir(exist_ok=True)
    md = OUT / f"{NAME}.epub.md"
    md.write_text(assemble("epub"))
    css = OUT / "epub.css"
    css.write_text(EPUB_CSS)
    cmd = ["pandoc", str(md), "-o", str(OUT / f"{NAME}.epub"),
           "--from", "markdown-raw_tex", "--number-sections", "--toc", "--toc-depth=1",
           "--split-level=1", "--mathml", "--resource-path", str(BOOK / "out/epubfig"),
           "--css", str(css), "-M", f"title={TITLE}", "-M", f"subtitle={SUBTITLE}",
           "-M", f"author={AUTHOR}", "-M", f"date={YEAR}", "-M", "lang=en-GB"]
    _small_figures("epubfig", 1100)
    subprocess.run(cmd, check=True, cwd=BOOK)
    print("EPUB built")


def _small_figures(sub: str = "epubfig", width: int = 1100):
    """Figures quantised to 128 colours (and at most `width` px wide) under
    book/out/<sub>/, to keep the PDF and EPUB small."""
    from PIL import Image
    for p in (BOOK / "figures").rglob("*.png"):
        q = OUT / sub / p.relative_to(BOOK)
        if q.exists() and q.stat().st_mtime >= p.stat().st_mtime:
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(p).convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        im = im.quantize(colors=128, method=Image.Quantize.MEDIANCUT,
                         dither=Image.Dither.NONE)
        im.save(q, optimize=True)


if __name__ == "__main__":
    sys.path.insert(0, str(BOOK / "tools"))
    import extract  # sources and figures
    extract.main()
    which = sys.argv[1:] or ["pdf", "epub"]
    if "pdf" in which:
        build_pdf()
    if "epub" in which:
        build_epub()
