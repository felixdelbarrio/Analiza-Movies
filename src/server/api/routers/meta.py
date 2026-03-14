# /meta/files
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from server.api.paths import (
    get_artifact_paths,
)

router = APIRouter()
logger = logging.getLogger(__name__)


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
    include_sha256: bool = Query(
        False, description="Si true, calcula sha256 (costoso en ficheros grandes)"
    ),
    profile_id: str | None = Query(None, description="Perfil de origen"),
) -> dict[str, Any]:
    paths_bundle = get_artifact_paths(profile_id)
    paths = {
        "omdb_cache": paths_bundle.omdb_cache_path,
        "wiki_cache": paths_bundle.wiki_cache_path,
        "report_all": paths_bundle.report_all_path,
        "report_filtered": paths_bundle.report_filtered_path,
        "metadata_fix": paths_bundle.metadata_fix_path,
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
                "mtime": (
                    datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                    if stat
                    else None
                ),
                "sha256": _hash_file_quick(p) if (exists and include_sha256) else None,
            }
        except Exception:
            logger.exception("No se pudo inspeccionar el fichero %s", p)
            out[k] = {
                "path": str(p),
                "exists": False,
                "error": "inspection_failed",
            }
    return out
