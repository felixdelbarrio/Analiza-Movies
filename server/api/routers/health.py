from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response

from server.api.caching.file_cache import FileCache
from server.api.deps import get_file_cache
from server.api.paths import (
    METADATA_FIX_PATH,
    OMDB_CACHE_PATH,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
    WIKI_CACHE_PATH,
)
from server.api.services import metrics
from server.api.services.omdb import load_payload as load_omdb
from server.api.services.wiki import load_payload as load_wiki

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@router.get("/ready")
def ready(cache: FileCache = Depends(get_file_cache)) -> dict[str, Any]:
    """
    Readiness:
    - comprueba existencia y lectura de ficheros críticos.
    - report_filtered es opcional (tu API puede devolver 204).
    """
    paths = {
        "omdb_cache": OMDB_CACHE_PATH,
        "wiki_cache": WIKI_CACHE_PATH,
        "report_all": REPORT_ALL_PATH,
        "report_filtered": REPORT_FILTERED_PATH,
        "metadata_fix": METADATA_FIX_PATH,
    }

    issues: dict[str, str] = {}
    for k, p in paths.items():
        try:
            if not p.exists():
                if k == "report_filtered":
                    continue
                issues[k] = f"missing: {p}"
                continue

            if p.suffix.lower() == ".json":
                if p == OMDB_CACHE_PATH:
                    _ = load_omdb(cache)
                elif p == WIKI_CACHE_PATH:
                    _ = load_wiki(cache)
                else:
                    _ = cache.load_json(p)
            elif p.suffix.lower() == ".csv":
                _ = cache.load_csv(p, text_columns=["poster_url", "trailer_url", "omdb_json"])
        except Exception as exc:
            issues[k] = f"unreadable: {p} ({exc!r})"

    if issues:
        raise HTTPException(status_code=503, detail={"ready": False, "issues": issues})

    return {"ready": True, "ts": datetime.now(timezone.utc).isoformat()}


@router.get("/metrics")
def metrics_endpoint() -> Response:
    body = metrics.render_prometheus()
    return Response(content=body, media_type="text/plain; version=0.0.4")