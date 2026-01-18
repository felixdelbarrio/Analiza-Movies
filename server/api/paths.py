# BASE_DIR + resolve_path + paths globales
from __future__ import annotations

import os
from pathlib import Path


# server/api/paths.py -> repo_root = parents[2] (server/api/*)
BASE_DIR = Path(__file__).resolve().parents[2]


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


def resolve_path(env_name: str, candidates: list[Path]) -> Path:
    raw = (os.getenv(env_name) or "").strip().strip('"').strip("'")
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        return p

    found = _first_existing(candidates)
    if found is None:
        # devolvemos el primer candidato (aunque no exista) para que el error sea explícito
        return candidates[0]
    return found


OMDB_CACHE_PATH = resolve_path(
    "OMDB_CACHE_PATH",
    [
        BASE_DIR / "data" / "omdb_cache.json",
        BASE_DIR / "omdb_cache.json",
    ],
)

WIKI_CACHE_PATH = resolve_path(
    "WIKI_CACHE_PATH",
    [
        BASE_DIR / "data" / "wiki_cache.json",
        BASE_DIR / "wiki_cache.json",
    ],
)

REPORT_ALL_PATH = resolve_path(
    "REPORT_ALL_PATH",
    [
        BASE_DIR / "reports" / "report_all.csv",
        BASE_DIR / "report_all.csv",
    ],
)

REPORT_FILTERED_PATH = resolve_path(
    "REPORT_FILTERED_PATH",
    [
        BASE_DIR / "reports" / "report_filtered.csv",
        BASE_DIR / "report_filtered.csv",
    ],
)

METADATA_FIX_PATH = resolve_path(
    "METADATA_FIX_PATH",
    [
        BASE_DIR / "reports" / "metadata_fix.csv",
        BASE_DIR / "metadata_fix.csv",
    ],
)
