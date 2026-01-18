from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from server.api.caching.file_cache import FileCache
from server.api.caching.http_cache import maybe_not_modified, stat_or_none
from server.api.deps import get_file_cache
from server.api.paths import OMDB_CACHE_PATH
from server.api.services.omdb import load_payload

router = APIRouter()


def _payload(cache: FileCache) -> dict[str, Any]:
    try:
        return load_payload(cache)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/cache/omdb/records")
def omdb_records(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    status: str | None = Query(
        None, description="Filtra por record.status (ok/not_found/error)"
    ),
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    st = stat_or_none(OMDB_CACHE_PATH)
    if maybe_not_modified(request=request, response=response, stat=st):
        return Response(status_code=304, headers=dict(response.headers))

    payload = _payload(cache)
    records = payload.get("records")
    if not isinstance(records, dict):
        raise HTTPException(
            status_code=500, detail="omdb_cache.json: falta 'records' dict"
        )

    rids = sorted(records.keys())
    if status:
        wanted = status.strip().lower()
        rids = [
            rid
            for rid in rids
            if str((records.get(rid) or {}).get("status", "")).lower() == wanted
        ]

    total = len(rids)
    page_rids = rids[offset : offset + limit]
    items = [{"rid": rid, **(records.get(rid) or {})} for rid in page_rids]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/cache/omdb/by-imdb/{imdb_id}")
def omdb_by_imdb(
    imdb_id: str,
    request: Request,
    response: Response,
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    st = stat_or_none(OMDB_CACHE_PATH)
    if maybe_not_modified(request=request, response=response, stat=st):
        return Response(status_code=304, headers=dict(response.headers))

    payload = _payload(cache)
    records = payload.get("records") or {}
    index_imdb = payload.get("index_imdb") or {}

    rid = index_imdb.get(imdb_id)
    if not rid:
        raise HTTPException(
            status_code=404, detail=f"imdb_id no encontrado en index_imdb: {imdb_id}"
        )

    rec = records.get(str(rid))
    if not rec:
        raise HTTPException(
            status_code=404, detail=f"rid no encontrado en records: {rid}"
        )

    return {"rid": str(rid), **rec}


@router.get("/cache/omdb/by-title-year")
def omdb_by_title_year(
    request: Request,
    response: Response,
    title: str = Query(..., min_length=1),
    year: str | None = Query(None, description="Año (p.ej. 1999)"),
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    st = stat_or_none(OMDB_CACHE_PATH)
    if maybe_not_modified(request=request, response=response, stat=st):
        return Response(status_code=304, headers=dict(response.headers))

    payload = _payload(cache)
    records = payload.get("records") or {}
    index_ty = payload.get("index_ty") or {}

    t = title.strip().lower()
    y = (year or "").strip()
    key = f"{t}|{y}" if y else t

    rid = index_ty.get(key)
    if not rid:
        alt = t if y else f"{t}|"
        rid = index_ty.get(alt)

    if not rid:
        raise HTTPException(status_code=404, detail=f"No encontrado en index_ty: {key}")

    rec = records.get(str(rid))
    if not rec:
        raise HTTPException(
            status_code=404, detail=f"rid no encontrado en records: {rid}"
        )

    return {"rid": str(rid), **rec}


@router.get("/cache/omdb/record/{rid}")
def omdb_by_rid(
    rid: str,
    request: Request,
    response: Response,
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    st = stat_or_none(OMDB_CACHE_PATH)
    if maybe_not_modified(request=request, response=response, stat=st):
        return Response(status_code=304, headers=dict(response.headers))

    payload = _payload(cache)
    records = payload.get("records") or {}
    rec = records.get(str(rid))
    if not rec:
        raise HTTPException(status_code=404, detail=f"rid no encontrado: {rid}")
    return {"rid": str(rid), **rec}
