from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from server.api.caching.file_cache import FileCache
from server.api.caching.http_cache import maybe_not_modified, stat_or_none
from server.api.deps import get_file_cache
from server.api.paths import METADATA_FIX_PATH, REPORT_ALL_PATH, REPORT_FILTERED_PATH
from server.api.services.reports import df_to_page, prepare_search_blob

router = APIRouter()
_TEXT_COLUMNS = ["poster_url", "trailer_url", "omdb_json"]


@router.get("/reports/all")
def reports_all(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    query: str | None = Query(None, description="Búsqueda simple (title/file/imdb)"),
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    st = stat_or_none(REPORT_ALL_PATH)
    if maybe_not_modified(request=request, response=response, stat=st):
        return Response(status_code=304, headers=dict(response.headers))

    try:
        df = cache.load_csv(REPORT_ALL_PATH, text_columns=_TEXT_COLUMNS)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No encontrado: {REPORT_ALL_PATH}")

    if "__search_blob" not in df.columns:
        prepare_search_blob(df)

    return df_to_page(df, offset=offset, limit=limit, query=query)


@router.get("/reports/filtered")
def reports_filtered(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    query: str | None = Query(None),
    empty_as_204: bool = Query(True, description="Si no existe, devuelve 204"),
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    if not REPORT_FILTERED_PATH.exists():
        if empty_as_204:
            return Response(status_code=204)
        raise HTTPException(
            status_code=404, detail=f"No encontrado: {REPORT_FILTERED_PATH}"
        )

    st = stat_or_none(REPORT_FILTERED_PATH)
    if maybe_not_modified(request=request, response=response, stat=st):
        return Response(status_code=304, headers=dict(response.headers))

    try:
        df = cache.load_csv(REPORT_FILTERED_PATH, text_columns=_TEXT_COLUMNS)
    except Exception:
        raise HTTPException(status_code=500, detail="Error leyendo report_filtered")

    if "__search_blob" not in df.columns:
        prepare_search_blob(df)

    return df_to_page(df, offset=offset, limit=limit, query=query)


@router.get("/reports/metadata-fix")
def metadata_fix(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    query: str | None = Query(None),
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    st = stat_or_none(METADATA_FIX_PATH)
    if maybe_not_modified(request=request, response=response, stat=st):
        return Response(status_code=304, headers=dict(response.headers))

    try:
        df = cache.load_csv(METADATA_FIX_PATH, text_columns=_TEXT_COLUMNS)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"No encontrado: {METADATA_FIX_PATH}"
        )

    if "__search_blob" not in df.columns:
        prepare_search_blob(df)

    return df_to_page(df, offset=offset, limit=limit, query=query)
