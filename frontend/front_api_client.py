from __future__ import annotations

# =============================================================================
# frontend/front_api_client.py
#
# Cliente HTTP minimalista (stdlib-only) para consumir la API pública (FastAPI)
# desde el front (Streamlit).
#
# - Sin dependencias externas (no requests/httpx).
# - Tipado estricto.
# - Pensado para paginación de endpoints:
#     /reports/all
#     /reports/filtered
#     /reports/metadata-fix
# =============================================================================

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Final

import pandas as pd


class ApiClientError(Exception):
    pass


@dataclass(frozen=True)
class _ApiPage:
    items: list[dict[str, object | None]]
    total: int
    limit: int
    offset: int


_API_MAX_PAGE_SIZE: Final[int] = 2000


def _build_url(base_url: str, path: str, params: dict[str, str | int] | None) -> str:
    base = base_url.rstrip("/")
    p = path if path.startswith("/") else f"/{path}"
    url = f"{base}{p}"
    if not params:
        return url
    return f"{url}?{urllib.parse.urlencode(params)}"


def _request_json(url: str, *, timeout_s: float) -> tuple[int, object | None]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = int(getattr(resp, "status", 200))
            if status == 204:
                return status, None
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        if status == 204:
            return status, None
        try:
            raw = exc.read()
        except Exception:
            raw = b""
        detail = raw.decode("utf-8", errors="replace").strip()
        raise ApiClientError(f"HTTP {status} en {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ApiClientError(f"Error de conexión en {url}: {exc!r}") from exc

    if not raw:
        return status, None

    try:
        return status, json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiClientError(f"Respuesta no-JSON en {url}: {exc}") from exc


def _parse_page(payload: object) -> _ApiPage:
    if not isinstance(payload, dict):
        raise ApiClientError("Payload inesperado (no dict).")

    items_obj = payload.get("items")
    total_obj = payload.get("total")
    limit_obj = payload.get("limit")
    offset_obj = payload.get("offset")

    if not isinstance(items_obj, list):
        raise ApiClientError("Payload inesperado: 'items' no es list.")
    items: list[dict[str, object | None]] = []
    for it in items_obj:
        if isinstance(it, dict):
            items.append(it)
        else:
            raise ApiClientError("Payload inesperado: item no es dict.")

    if not isinstance(total_obj, int):
        raise ApiClientError("Payload inesperado: 'total' no es int.")
    if not isinstance(limit_obj, int):
        raise ApiClientError("Payload inesperado: 'limit' no es int.")
    if not isinstance(offset_obj, int):
        raise ApiClientError("Payload inesperado: 'offset' no es int.")

    return _ApiPage(items=items, total=total_obj, limit=limit_obj, offset=offset_obj)


def _fetch_df_paginated(
    *,
    base_url: str,
    endpoint: str,
    timeout_s: float,
    page_size: int,
    query: str | None = None,
    empty_as_none: bool = False,
) -> pd.DataFrame | None:
    limit = max(1, min(int(page_size), _API_MAX_PAGE_SIZE))
    offset = 0
    all_items: list[dict[str, object | None]] = []
    total: int | None = None

    while True:
        params: dict[str, str | int] = {"limit": limit, "offset": offset}
        if query:
            params["query"] = query

        url = _build_url(base_url, endpoint, params)
        status, payload = _request_json(url, timeout_s=timeout_s)

        if status == 204:
            return None if empty_as_none else pd.DataFrame()

        page = _parse_page(payload)
        if total is None:
            total = page.total

        all_items.extend(page.items)

        offset += page.limit
        if total is not None and offset >= total:
            break

        # Evita bucles infinitos si el servidor devuelve algo incoherente
        if page.limit <= 0:
            raise ApiClientError("Paginación inválida: limit <= 0.")

    df = pd.DataFrame(all_items)
    if df.empty and total and total > 0:
        # Caso raro: total > 0 pero items vacíos (servidor inconsistente)
        raise ApiClientError("API devolvió total>0 pero sin items.")
    return df


def fetch_report_all_df(
    *, base_url: str, timeout_s: float, page_size: int, query: str | None = None
) -> pd.DataFrame:
    df = _fetch_df_paginated(
        base_url=base_url,
        endpoint="/reports/all",
        timeout_s=timeout_s,
        page_size=page_size,
        query=query,
        empty_as_none=False,
    )
    if df is None:
        return pd.DataFrame()
    return df


def fetch_report_filtered_df(
    *, base_url: str, timeout_s: float, page_size: int, query: str | None = None
) -> pd.DataFrame | None:
    return _fetch_df_paginated(
        base_url=base_url,
        endpoint="/reports/filtered",
        timeout_s=timeout_s,
        page_size=page_size,
        query=query,
        empty_as_none=True,
    )


def fetch_metadata_fix_df(
    *, base_url: str, timeout_s: float, page_size: int, query: str | None = None
) -> pd.DataFrame:
    df = _fetch_df_paginated(
        base_url=base_url,
        endpoint="/reports/metadata-fix",
        timeout_s=timeout_s,
        page_size=page_size,
        query=query,
        empty_as_none=False,
    )
    if df is None:
        return pd.DataFrame()
    return df
