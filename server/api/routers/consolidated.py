from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from server.api.caching.file_cache import FileCache
from server.api.caching.http_cache import maybe_not_modified, stat_or_none
from server.api.deps import get_file_cache
from server.api.paths import OMDB_CACHE_PATH, WIKI_CACHE_PATH
from server.api.services.consolidated import consolidate

# IMPORTANTE:
# - FastAPI espera que este símbolo exista para poder hacer:
#   `from server.api.routers.consolidated import router`
# - Debe estar a nivel de módulo (no dentro de funciones/condicionales).
router = APIRouter()


@router.get("/cache/consolidated/by-imdb/{imdb_id}")
def consolidated_by_imdb(
    imdb_id: str,
    request: Request,
    response: Response,
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    # Consolidated depende de dos ficheros: OMDb + Wiki.
    # Si el cliente manda If-None-Match/If-Modified-Since, devolvemos 304 cuando aplique.
    if maybe_not_modified(request=request, response=response, stat=stat_or_none(OMDB_CACHE_PATH)):
        return Response(status_code=304, headers=dict(response.headers))
    if maybe_not_modified(request=request, response=response, stat=stat_or_none(WIKI_CACHE_PATH)):
        return Response(status_code=304, headers=dict(response.headers))

    data = consolidate(cache=cache, imdb_id=imdb_id, title=None, year=None)

    if not data["sources"]["omdb"]["rid"] and not data["sources"]["wiki"]["rid"]:
        raise HTTPException(status_code=404, detail=f"No encontrado en caches para imdb_id={imdb_id}")

    return data


@router.get("/cache/consolidated/by-title-year")
def consolidated_by_title_year(
    request: Request,
    response: Response,
    title: str = Query(..., min_length=1),
    year: str | None = Query(None, description="Año (p.ej. 1999)"),
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    if maybe_not_modified(request=request, response=response, stat=stat_or_none(OMDB_CACHE_PATH)):
        return Response(status_code=304, headers=dict(response.headers))
    if maybe_not_modified(request=request, response=response, stat=stat_or_none(WIKI_CACHE_PATH)):
        return Response(status_code=304, headers=dict(response.headers))

    data = consolidate(cache=cache, imdb_id=None, title=title, year=year)

    if not data["sources"]["omdb"]["rid"] and not data["sources"]["wiki"]["rid"]:
        key = f"{title}|{year or ''}"
        raise HTTPException(status_code=404, detail=f"No encontrado en caches para {key}")

    return data


@router.get("/cache/consolidated/records")
def consolidated_records(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    status_omdb: str | None = Query(None, description="Filtra por status en omdb_cache (ok/not_found/error/...)"),
    status_wiki: str | None = Query(None, description="Filtra por status en wiki_cache (ok/no_qid/not_film/...)"),
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    if maybe_not_modified(request=request, response=response, stat=stat_or_none(OMDB_CACHE_PATH)):
        return Response(status_code=304, headers=dict(response.headers))
    if maybe_not_modified(request=request, response=response, stat=stat_or_none(WIKI_CACHE_PATH)):
        return Response(status_code=304, headers=dict(response.headers))

    omdb_payload = cache.load_json(OMDB_CACHE_PATH)
    wiki_payload = cache.load_json(WIKI_CACHE_PATH)

    omdb_records = omdb_payload.get("records") if isinstance(omdb_payload, dict) else {}
    wiki_records = wiki_payload.get("records") if isinstance(wiki_payload, dict) else {}

    imdbs: set[str] = set()

    for rec in omdb_records.values() if isinstance(omdb_records, dict) else []:
        if not isinstance(rec, dict):
            continue
        if status_omdb and str(rec.get("status", "")).lower() != status_omdb.strip().lower():
            continue
        v = rec.get("imdbID")
        if isinstance(v, str) and v.strip():
            imdbs.add(v.strip().lower())

    for rec in wiki_records.values() if isinstance(wiki_records, dict) else []:
        if not isinstance(rec, dict):
            continue
        if status_wiki and str(rec.get("status", "")).lower() != status_wiki.strip().lower():
            continue
        v = rec.get("imdbID")
        if isinstance(v, str) and v.strip():
            imdbs.add(v.strip().lower())

    imdb_list = sorted(imdbs)
    total = len(imdb_list)

    page = imdb_list[offset : offset + limit]
    items = [consolidate(cache=cache, imdb_id=imdb_id, title=None, year=None) for imdb_id in page]

    return {"items": items, "total": total, "limit": limit, "offset": offset}