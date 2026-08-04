"""Central configuration: paths, API keys, HTTP defaults.

Keys are read from the environment (or a local .env) so nothing secret ever
lands in the repo. See .env.example.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency on python-dotenv)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


REPO_ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # --- filesystem ---
    repo_root: Path = REPO_ROOT
    cache_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("QUANTPOST_CACHE", REPO_ROOT / ".cache")))
    build_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("QUANTPOST_BUILD", REPO_ROOT / "build")))
    site_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("QUANTPOST_SITE", REPO_ROOT / "site")))

    # --- credentials (all optional; adapters degrade or raise clearly) ---
    fred_api_key: str | None = field(
        default_factory=lambda: os.environ.get("FRED_API_KEY") or None)
    ecos_api_key: str = field(
        default_factory=lambda: os.environ.get("ECOS_API_KEY", "sample"))
    notion_token: str | None = field(
        default_factory=lambda: os.environ.get("NOTION_TOKEN") or None)
    notion_database_id: str | None = field(
        default_factory=lambda: os.environ.get("NOTION_DATABASE_ID") or None)

    # --- site identity (used for canonical URLs on Medium crossposts) ---
    site_base_url: str = field(default_factory=lambda: os.environ.get(
        "SITE_BASE_URL", "https://jonghajeon.github.io/quantpost"))
    author: str = field(default_factory=lambda: os.environ.get(
        "POST_AUTHOR", "Jongha Jeon"))
    code_repo_url: str = field(default_factory=lambda: os.environ.get(
        "CODE_REPO_URL", "https://github.com/jonghajeon/quantpost"))

    # --- HTTP ---
    user_agent: str = field(default_factory=lambda: os.environ.get(
        "QUANTPOST_UA",
        "quantpost/0.1 (research; contact via repo issues)"))
    timeout: float = 30.0
    max_retries: int = 4
    cache_ttl_hours: float = float(os.environ.get("QUANTPOST_TTL_HOURS", "12"))

    # --- reproducibility ---
    seed: int = int(os.environ.get("QUANTPOST_SEED", "20260804"))

    def ensure_dirs(self) -> None:
        for p in (self.cache_dir, self.build_dir):
            p.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
SETTINGS.ensure_dirs()
