from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from server.api.caching.file_cache import FileCache
from server.api.deps import get_file_cache
from server.api.paths import get_artifact_paths
from server.api.services import metrics
from server.api.services.omdb import load_payload as load_omdb
from server.api.services.wiki import load_payload as load_wiki

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@router.get("/ready")
def ready(
    profile_id: str | None = Query(None, description="Perfil de origen"),
    cache: FileCache = Depends(get_file_cache),
) -> dict[str, Any]:
    """
    Readiness:
    - comprueba existencia y lectura de ficheros críticos.
    - report_filtered es opcional (tu API puede devolver 204).
    """
    paths_bundle = get_artifact_paths(profile_id)
    paths = {
        "omdb_cache": paths_bundle.omdb_cache_path,
        "wiki_cache": paths_bundle.wiki_cache_path,
        "report_all": paths_bundle.report_all_path,
        "report_filtered": paths_bundle.report_filtered_path,
        "metadata_fix": paths_bundle.metadata_fix_path,
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
                if p == paths_bundle.omdb_cache_path:
                    _ = load_omdb(cache, profile_id=profile_id)
                elif p == paths_bundle.wiki_cache_path:
                    _ = load_wiki(cache, profile_id=profile_id)
                else:
                    _ = cache.load_json(p)
            elif p.suffix.lower() == ".csv":
                _ = cache.load_csv(
                    p, text_columns=["poster_url", "trailer_url", "omdb_json"]
                )
        except Exception as exc:
            issues[k] = f"unreadable: {p} ({exc!r})"

    if issues:
        raise HTTPException(status_code=503, detail={"ready": False, "issues": issues})

    return {"ready": True, "ts": datetime.now(timezone.utc).isoformat()}


@router.get("/metrics")
def metrics_endpoint() -> Response:
    body = metrics.render_prometheus()
    return Response(content=body, media_type="text/plain; version=0.0.4")
