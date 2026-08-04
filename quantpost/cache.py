"""On-disk HTTP cache with a provenance manifest.

Two reasons this exists and is not optional:

1. Public data APIs are rate-limited and occasionally down. A post that took
   40 fetches to draft should not need 40 more to re-render.
2. **Provenance.** Every cached payload records the exact URL, the UTC fetch
   timestamp and a sha256 of the bytes. When a chart in a published post is
   questioned six months later, `provenance.jsonl` says precisely which bytes
   produced it. Public macro series get revised; without this you cannot tell
   "my code changed" from "the vintage changed".
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from .config import SETTINGS


@dataclass(frozen=True)
class Provenance:
    url: str
    sha256: str
    fetched_at_utc: str
    n_bytes: int
    source: str
    note: str = ""


def _key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def _manifest_path() -> Path:
    return SETTINGS.cache_dir / "provenance.jsonl"


def record_provenance(p: Provenance) -> None:
    with _manifest_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def read_provenance() -> list[Provenance]:
    path = _manifest_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Provenance(**json.loads(line)))
    return out


def fetch(
    url: str,
    *,
    source: str,
    headers: dict[str, str] | None = None,
    ttl_hours: float | None = None,
    force: bool = False,
    note: str = "",
    session: requests.Session | None = None,
) -> bytes:
    """GET `url` with retry + backoff, caching bytes on disk.

    Raises `requests.HTTPError` on a non-2xx that survives retries. 429 and 5xx
    are retried with exponential backoff; 4xx other than 429 fail fast, because
    a bad series ID will never succeed and retrying only burns your quota.
    """
    SETTINGS.ensure_dirs()
    ttl = SETTINGS.cache_ttl_hours if ttl_hours is None else ttl_hours
    blob = SETTINGS.cache_dir / f"{source}-{_key(url)}.bin"

    if blob.exists() and not force:
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            blob.stat().st_mtime, tz=timezone.utc)
        if ttl < 0 or age < timedelta(hours=ttl):
            return blob.read_bytes()

    sess = session or requests.Session()
    hdrs = {"User-Agent": SETTINGS.user_agent}
    hdrs.update(headers or {})

    last_exc: Exception | None = None
    for attempt in range(SETTINGS.max_retries):
        try:
            resp = sess.get(url, headers=hdrs, timeout=SETTINGS.timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(
                    f"{resp.status_code} for {url}", response=resp)
            resp.raise_for_status()
            data = resp.content
            blob.write_bytes(data)
            record_provenance(Provenance(
                url=url,
                sha256=hashlib.sha256(data).hexdigest(),
                fetched_at_utc=datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
                n_bytes=len(data),
                source=source,
                note=note,
            ))
            return data
        except requests.HTTPError as exc:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)
            last_exc = exc
            retryable = status is None or status == 429 or status >= 500
            if not retryable:
                raise
            if attempt < SETTINGS.max_retries - 1:
                time.sleep(2.0 ** attempt)
        except requests.RequestException as exc:  # timeouts, DNS, conn reset
            last_exc = exc
            if attempt < SETTINGS.max_retries - 1:
                time.sleep(2.0 ** attempt)

    # Stale cache beats no data at all — but say so loudly.
    if blob.exists():
        import warnings
        warnings.warn(
            f"{source}: fetch failed ({last_exc}); serving STALE cache for {url}",
            RuntimeWarning, stacklevel=2)
        return blob.read_bytes()
    raise RuntimeError(f"{source}: fetch failed for {url}") from last_exc
