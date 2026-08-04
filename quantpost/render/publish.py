"""Writers: Hugo site, Medium crosspost bundle, Notion page.

**Publish to your own site first, then crosspost.** Medium's importer sets
`rel="canonical"` back to the original URL, so search engines credit your domain
and you are not renting your archive from a platform that has changed its
paywall rules repeatedly. `medium_bundle` therefore refuses to run without a
canonical URL — that is not pedantry, it is the entire reason to own the site.

Image handling differs by target and this is the usual source of broken posts:

* Hugo page bundle -> images sit beside `index.md`, referenced relatively.
* Medium -> the importer needs **absolute** URLs, so images must already be live
  on your site. Publish Hugo first; the bundle references those URLs.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from ..config import SETTINGS
from .post import Post

# ---------------------------------------------------------------- Hugo

def hugo_page_bundle(post: Post, *, site_dir: Path | None = None,
                     section: str = "posts") -> Path:
    """Write `content/{section}/{slug}/index.md` plus co-located images."""
    site = Path(site_dir or SETTINGS.site_dir)
    bundle = site / "content" / section / post.slug
    bundle.mkdir(parents=True, exist_ok=True)

    for fig in post.figures:
        src = Path(fig.path)
        if src.exists():
            shutil.copy2(src, bundle / src.name)

    (bundle / "index.md").write_text(post.hugo_markdown(image_base=""),
                                     encoding="utf-8")
    return bundle / "index.md"


# ---------------------------------------------------------------- Medium

def medium_bundle(post: Post, *, out_dir: Path | None = None,
                  base_url: str | None = None,
                  section: str = "posts") -> Path:
    """Markdown with absolute image URLs and a canonical note, ready to paste
    into Medium's import-a-story flow."""
    if post.draft:
        raise ValueError(
            "this post is still a draft. Medium's importer fetches the images "
            "from your live site, so publish there first (--live, push, wait for "
            "Pages) and then build the crosspost.")
    base = (base_url or SETTINGS.site_base_url).rstrip("/")
    canonical = post.canonical_url or f"{base}/{section}/{post.slug}/"
    if not canonical.startswith("http"):
        raise ValueError(
            "medium_bundle needs an absolute canonical URL. Publish to your own "
            "site first so the crosspost points home.")

    out = Path(out_dir or (SETTINGS.build_dir / "medium"))
    out.mkdir(parents=True, exist_ok=True)
    image_base = f"{canonical.rstrip('/')}"

    body = post.body_markdown(image_base=image_base)
    header = (f"# {post.title}\n\n"
              + (f"### {post.subtitle}\n\n" if post.subtitle else "")
              + f"> Originally published at [{base}]({canonical}).\n\n")
    path = out / f"{post.slug}.md"
    path.write_text(header + body, encoding="utf-8")

    (out / f"{post.slug}.meta.json").write_text(json.dumps({
        "title": post.title,
        "subtitle": post.subtitle,
        "canonical_url": canonical,
        "tags": post.tags[:5],       # Medium caps at 5
        "images": [f"{image_base}/{Path(f.path).name}" for f in post.figures],
        "checklist": [
            "Publish the Hugo page first so the image URLs above resolve.",
            "Import via https://medium.com/p/import — it sets rel=canonical.",
            "Medium keeps at most 5 tags.",
            "Check LaTeX: Medium has no math rendering. Convert display "
            "equations to images or restate them in prose.",
            "Code blocks import as plain text; re-add language hints or embed a "
            "GitHub gist.",
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- Notion

def notion_page(post: Post, *, database_id: str | None = None,
                token: str | None = None, dry_run: bool = False) -> dict:
    """Create a Notion page from the post. Blocks are chunked to the 100-per-
    request API limit; images must be publicly reachable URLs."""
    import requests

    token = token or SETTINGS.notion_token
    database_id = database_id or SETTINGS.notion_database_id
    blocks = _to_notion_blocks(post)
    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            "Name": {"title": [{"text": {"content": post.title}}]},
        },
        "children": blocks[:100],
    }
    if dry_run:
        return {"dry_run": True, "n_blocks": len(blocks), "payload": payload}
    if not token or not database_id:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATABASE_ID must be set")
    headers = {"Authorization": f"Bearer {token}",
               "Notion-Version": "2022-06-28",
               "Content-Type": "application/json"}
    r = requests.post("https://api.notion.com/v1/pages", headers=headers,
                      json=payload, timeout=SETTINGS.timeout)
    r.raise_for_status()
    page = r.json()
    for i in range(100, len(blocks), 100):
        rr = requests.patch(
            f"https://api.notion.com/v1/blocks/{page['id']}/children",
            headers=headers, json={"children": blocks[i:i + 100]},
            timeout=SETTINGS.timeout)
        rr.raise_for_status()
    return page


def _rich(text: str) -> list[dict]:
    # Notion caps a single rich_text item at 2000 characters.
    return [{"type": "text", "text": {"content": text[i:i + 2000]}}
            for i in range(0, max(len(text), 1), 2000)]


def _to_notion_blocks(post: Post) -> list[dict]:
    blocks: list[dict] = []
    if post.summary:
        blocks.append({"object": "block", "type": "callout",
                       "callout": {"rich_text": _rich(post.summary)}})
    for s in post.sections:
        key = f"heading_{min(max(s.level, 1), 3)}"
        blocks.append({"object": "block", "type": key,
                       key: {"rich_text": _rich(s.heading)}})
        for para in [p for p in s.body.split("\n\n") if p.strip()]:
            if para.strip().startswith("```"):
                code = para.strip().strip("`")
                lang, _, rest = code.partition("\n")
                blocks.append({"object": "block", "type": "code",
                               "code": {"rich_text": _rich(rest),
                                        "language": (lang or "python").strip()}})
            else:
                blocks.append({"object": "block", "type": "paragraph",
                               "paragraph": {"rich_text": _rich(para.strip())}})
        for f in s.figures:
            url = f.path if f.path.startswith("http") else None
            if url:
                blocks.append({"object": "block", "type": "image",
                               "image": {"type": "external",
                                         "external": {"url": url},
                                         "caption": _rich(f.caption or f.alt)}})
            else:
                blocks.append({"object": "block", "type": "paragraph",
                               "paragraph": {"rich_text": _rich(
                                   f"[figure: {Path(f.path).name} — {f.alt}]")}})
    return blocks


# ---------------------------------------------------------------- manifest

def write_manifest(post: Post, *, out_dir: Path | None = None) -> Path:
    """A machine-readable record of what was published, for the site index and
    for diffing a re-run against a published post."""
    out = Path(out_dir or SETTINGS.build_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{post.slug}.manifest.json"
    path.write_text(json.dumps({
        "title": post.title, "slug": post.slug, "date": post.date.isoformat(),
        "draft": post.draft,
        "tags": post.tags, "word_count": post.word_count(),
        "figures": [asdict(f) for f in post.figures],
        "data_sources": post.data_sources,
        "licence_warnings": post.licence_warnings,
        "reproducibility": post.reproducibility,
        "audit": post.audit(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
