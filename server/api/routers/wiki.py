from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from server.api.caching.file_cache import FileCache
from server.api.caching.http_cache import maybe_not_modified, stat_or_none
from server.api.deps import get_file_cache
from server.api.paths import WIKI_CACHE_PATH
from server.api.services.wiki import load_payload

router = APIRouter()


def _payload(cache: FileCache) -> dict[str, Any]:
    try:
        return load_payload(cache)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/cache/wiki/records")
def wiki_records(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None),
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    st = stat_or_none(WIKI_CACHE_PATH)
    if maybe_not_modified(request=request, response=response, stat=st):
        return Response(status_code=304, headers=dict(response.headers))

    payload = _payload(cache)
    records = payload.get("records")
    if not isinstance(records, dict):
        raise HTTPException(status_code=500, detail="wiki_cache.json: falta 'records' dict")

    rids = sorted(records.keys())
    if status:
        wanted = status.strip().lower()
        rids = [rid for rid in rids if str((records.get(rid) or {}).get("status", "")).lower() == wanted]

    total = len(rids)
    page_rids = rids[offset : offset + limit]
    items = [{"rid": rid, **(records.get(rid) or {})} for rid in page_rids]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/cache/wiki/by-imdb/{imdb_id}")
def wiki_by_imdb(
    imdb_id: str,
    request: Request,
    response: Response,
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    st = stat_or_none(WIKI_CACHE_PATH)
    if maybe_not_modified(request=request, response=response, stat=st):
        return Response(status_code=304, headers=dict(response.headers))

    payload = _payload(cache)
    records = payload.get("records") or {}
    index_imdb = payload.get("index_imdb") or payload.get("index") or {}

    rid = index_imdb.get(imdb_id) or index_imdb.get(f"imdb:{imdb_id}")
    if not rid:
        raise HTTPException(status_code=404, detail=f"imdb_id no encontrado: {imdb_id}")

    rec = records.get(str(rid))
    if not rec:
        raise HTTPException(status_code=404, detail=f"rid no encontrado en records: {rid}")

    return {"rid": str(rid), **rec}


@router.get("/cache/wiki/record/{rid}")
def wiki_by_rid(
    rid: str,
    request: Request,
    response: Response,
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    st = stat_or_none(WIKI_CACHE_PATH)
    if maybe_not_modified(request=request, response=response, stat=st):
        return Response(status_code=304, headers=dict(response.headers))

    payload = _payload(cache)
    records = payload.get("records") or {}
    rec = records.get(str(rid))
    if not rec:
        raise HTTPException(status_code=404, detail=f"rid no encontrado: {rid}")
    return {"rid": str(rid), **rec}