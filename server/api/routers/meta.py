# /meta/files
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from server.api.paths import (
    METADATA_FIX_PATH,
    OMDB_CACHE_PATH,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
    WIKI_CACHE_PATH,
)

router = APIRouter()


def _hash_file_quick(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@router.get("/meta/files")
def meta_files(
    include_sha256: bool = Query(False, description="Si true, calcula sha256 (costoso en ficheros grandes)"),
) -> dict[str, Any]:
    paths = {
        "omdb_cache": OMDB_CACHE_PATH,
        "wiki_cache": WIKI_CACHE_PATH,
        "report_all": REPORT_ALL_PATH,
        "report_filtered": REPORT_FILTERED_PATH,
        "metadata_fix": METADATA_FIX_PATH,
    }

    out: dict[str, Any] = {}
    for k, p in paths.items():
        try:
            exists = p.exists()
            stat = p.stat() if exists else None
            out[k] = {
                "path": str(p),
                "exists": bool(exists),
                "size": int(stat.st_size) if stat else None,
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat() if stat else None,
                "sha256": _hash_file_quick(p) if (exists and include_sha256) else None,
            }
        except Exception as exc:
            out[k] = {"path": str(p), "exists": False, "error": repr(exc)}
    return out