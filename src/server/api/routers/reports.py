from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from server.api.caching.file_cache import FileCache
from server.api.caching.http_cache import maybe_not_modified, stat_or_none
from server.api.deps import get_file_cache
from server.api.paths import (
    get_metadata_fix_path,
    get_report_all_path,
    get_report_filtered_path,
)
from server.api.services.reports import df_to_page, prepare_search_blob

router = APIRouter()
_TEXT_COLUMNS = ["poster_url", "trailer_url", "omdb_json"]


def _build_report_page(
    *,
    request: Request,
    response: Response,
    limit: int,
    offset: int,
    query: str | None,
    profile_id: str | None,
    cache: FileCache,
    path_getter: Any,
    missing_as_204: bool = False,
    read_error_detail: str | None = None,
) -> Any:
    path = path_getter(profile_id)
    if missing_as_204 and not path.exists():
        return Response(status_code=204)

    st = stat_or_none(path)
    if maybe_not_modified(request=request, response=response, stat=st):
        return Response(status_code=304, headers=dict(response.headers))

    try:
        df = cache.load_csv(path, text_columns=_TEXT_COLUMNS)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Recurso no encontrado")
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=read_error_detail or f"Error leyendo {path.name}",
        )

    if "__search_blob" not in df.columns:
        prepare_search_blob(df)

    return df_to_page(df, offset=offset, limit=limit, query=query)


@router.get("/reports/all")
def reports_all(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    query: str | None = Query(None, description="Búsqueda simple (title/file/imdb)"),
    profile_id: str | None = Query(None, description="Perfil de origen"),
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    return _build_report_page(
        request=request,
        response=response,
        limit=limit,
        offset=offset,
        query=query,
        profile_id=profile_id,
        cache=cache,
        path_getter=get_report_all_path,
    )


@router.get("/reports/filtered")
def reports_filtered(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    query: str | None = Query(None),
    empty_as_204: bool = Query(True, description="Si no existe, devuelve 204"),
    profile_id: str | None = Query(None, description="Perfil de origen"),
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    return _build_report_page(
        request=request,
        response=response,
        limit=limit,
        offset=offset,
        query=query,
        profile_id=profile_id,
        cache=cache,
        path_getter=get_report_filtered_path,
        missing_as_204=empty_as_204,
        read_error_detail="Error leyendo report_filtered",
    )


@router.get("/reports/metadata-fix")
def metadata_fix(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    query: str | None = Query(None),
    profile_id: str | None = Query(None, description="Perfil de origen"),
    cache: FileCache = Depends(get_file_cache),
) -> Any:
    return _build_report_page(
        request=request,
        response=response,
        limit=limit,
        offset=offset,
        query=query,
        profile_id=profile_id,
        cache=cache,
        path_getter=get_metadata_fix_path,
    )
