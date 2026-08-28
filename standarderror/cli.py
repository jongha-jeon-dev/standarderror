"""Command line: `standarderror <command>`.

    standarderror doctor                     # what works, what needs a key
    standarderror sources                    # list adapters and curated series
    standarderror fetch fred ust_10y vix --start 2015-01-01 --out data.csv
    standarderror run exp001_lorenz_esn      # run an experiment -> post + figures
    standarderror audit build/*.manifest.json
    standarderror publish exp001_lorenz_esn --medium
    standarderror run lec001_condition_number --gist   # code blocks -> one gist, for Medium
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from . import environment
from .config import SETTINGS


def cmd_doctor(_args) -> int:
    print("standarderror doctor\n" + "-" * 52)
    for k, v in environment().items():
        print(f"  {k:<14} {v}")
    print("\ncredentials")
    checks = [
        ("FRED_API_KEY", SETTINGS.fred_api_key,
         "optional — fred.get() works without it (keyless CSV)"),
        ("ECOS_API_KEY", None if SETTINGS.ecos_api_key == "sample"
         else SETTINGS.ecos_api_key,
         "using the 'sample' key: max 10 rows per call"),
        ("NOTION_TOKEN", SETTINGS.notion_token, "only needed for Notion output"),
    ]
    for name, val, note in checks:
        mark = "set" if val else "not set"
        print(f"  {name:<18} {mark:<9} {note}")
    print("\npaths")
    for name, p in (("cache", SETTINGS.cache_dir), ("build", SETTINGS.build_dir),
                    ("site", SETTINGS.site_dir)):
        print(f"  {name:<8} {p}  {'ok' if Path(p).exists() else 'missing'}")
    print(f"\nsite base URL   {SETTINGS.site_base_url}")
    if "example" in SETTINGS.site_base_url:
        print("  ! set SITE_BASE_URL in .env before crossposting to Medium")
    return 0


def cmd_sources(_args) -> int:
    from .sources import bis, ecb, ecos, fred
    print("fred    (no key needed for get(); key unlocks vintages)")
    for k, v in fred.CURATED.items():
        print(f"          {k:<22} {v}")
    print("\necb     (no key)")
    for k, v in ecb.CURATED.items():
        print(f"          {k:<22} {v}")
    print("\necos    (key in URL path; 'sample' works for 10 rows)")
    for k, v in ecos.CURATED.items():
        print(f"          {k:<22} {v.stat_code}/{v.item_code} [{v.cycle}] {v.label}")
    print("\nbis     (no key) dataflows: WS_CREDIT_GAP, WS_SPP, WS_DPP, ...")
    print(f"          api  {bis.API}")
    print(f"          bulk {bis.BULK}/{{FLOW}}_csv_flat.zip")
    print("\nhmda    (no key) aggregations first; CSV streams unbounded")
    print("market  stooq requires accept_terms=True; yfinance optional")
    print("local   Freddie Mac SF loan-level: manual download, licence applies")
    return 0


def cmd_fetch(args) -> int:
    mod = importlib.import_module(f".sources.{args.source}", package="standarderror")
    df = mod.get(args.series, start=args.start, end=args.end)
    print(df.tail(min(args.tail, len(df))).to_string())
    print(f"\n{len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
    from .sources import citations, licence_warnings
    for c in citations(df):
        print(f"source: {c}")
    for w in licence_warnings(df):
        print(f"WARNING: {w}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out)
        print(f"written: {args.out}")
    return 0


def cmd_run(args) -> int:
    name = args.experiment.removesuffix(".py")
    sys.path.insert(0, str(SETTINGS.repo_root / "experiments"))
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError:
        avail = sorted(p.stem for p in
                       (SETTINGS.repo_root / "experiments").glob("*.py"))
        print(f"no experiment {name!r}. available: {', '.join(avail)}",
              file=sys.stderr)
        return 2
    if not hasattr(mod, "build"):
        print(f"{name} must define build() -> Post", file=sys.stderr)
        return 2
    post = mod.build()
    if args.live:
        post.draft = False
    from .render import publish
    problems = post.audit()
    manifest = publish.write_manifest(post)
    state = "DRAFT" if post.draft else "LIVE"
    print(f"\n{post.title}\n  {post.word_count()} words, "
          f"{len(post.figures)} figures, {state}\n  manifest: {manifest}")
    if problems:
        print("\naudit FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("  audit: clean")
    if args.publish:
        page = publish.hugo_page_bundle(post, section=post.section)
        print(f"  hugo: {page}")
        if post.draft:
            print("  (draft: true — commit it, preview with `make serve`, then "
                  "re-run with --live when you are happy)")
        if args.medium:
            if post.draft:
                # The Medium import fetches images from the live site. Building
                # the bundle now would hand you absolute URLs that 404.
                print("  medium: SKIPPED — the post is still a draft, so its "
                      "image URLs are not live yet. Re-run with --live, push, "
                      "wait for Pages, then --medium.")
            else:
                print(f"  medium: {publish.medium_bundle(post)}")
    if args.gist:
        from .render import gist
        d = gist.gist_bundle(post, gist_url=args.gist_url)
        n = len(sorted(d.glob("*.py"))) or len([p for p in d.iterdir()
                                                if p.name != "PASTE.md"])
        print(f"  gist: {d}  ({n} file(s) + PASTE.md)")
    return 0


def cmd_audit(args) -> int:
    bad = 0
    for pattern in args.manifests:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            issues = data.get("audit", [])
            status = "clean" if not issues else f"{len(issues)} issue(s)"
            print(f"{data.get('slug', path)}: {data.get('word_count')} words, "
                  f"{len(data.get('figures', []))} figures — {status}")
            for i in issues:
                print(f"    - {i}")
            bad += bool(issues)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="standarderror",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="environment and credential check").set_defaults(
        func=cmd_doctor)
    sub.add_parser("sources", help="list adapters and curated series").set_defaults(
        func=cmd_sources)

    f = sub.add_parser("fetch", help="fetch a series and print/save it")
    f.add_argument("source", choices=["fred", "ecb", "ecos", "bis"])
    f.add_argument("series", nargs="+")
    f.add_argument("--start"); f.add_argument("--end")
    f.add_argument("--out"); f.add_argument("--tail", type=int, default=10)
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="run an experiment and audit its post")
    r.add_argument("experiment")
    r.add_argument("--publish", action="store_true",
                   help="write the Hugo page bundle")
    r.add_argument("--live", action="store_true",
                   help="publish for real (draft: false). Omit to stay a draft.")
    r.add_argument("--medium", action="store_true",
                   help="also write the Medium crosspost bundle; requires --live")
    r.add_argument("--gist", action="store_true",
                   help="split the code blocks into files for one GitHub gist, "
                        "plus the Medium paste order")
    r.add_argument("--gist-url", default=None,
                   help="the gist's URL, written into the paste order instead of "
                        "the placeholder (use after you have created it)")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("audit", help="re-check written manifests")
    a.add_argument("manifests", nargs="+")
    a.set_defaults(func=cmd_audit)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
