# ETag/Last-Modified/304 helpers
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Final

from fastapi import Request, Response

_ETAG_PREFIX: Final[str] = 'W/"'


def stat_or_none(path: Path):
    try:
        return path.stat()
    except Exception:
        return None


def _etag_from_stat(st) -> str:
    return f'{_ETAG_PREFIX}{st.st_mtime_ns:x}-{st.st_size:x}"'


def _http_date_from_ts(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


def maybe_not_modified(*, request: Request, response: Response, stat) -> bool:
    """
    Aplica ETag/Last-Modified y decide si devolver 304.

    Devuelve True si el caller debería cortar y responder 304.
    """
    if stat is None:
        return False

    etag = _etag_from_stat(stat)
    last_modified = _http_date_from_ts(stat.st_mtime)

    response.headers["ETag"] = etag
    response.headers["Last-Modified"] = last_modified

    inm = (request.headers.get("if-none-match") or "").strip()
    if inm and inm == etag:
        response.status_code = 304
        return True

    ims = (request.headers.get("if-modified-since") or "").strip()
    if ims:
        try:
            ims_dt = parsedate_to_datetime(ims)
            if ims_dt.tzinfo is None:
                ims_dt = ims_dt.replace(tzinfo=timezone.utc)
            server_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if server_dt <= ims_dt:
                response.status_code = 304
                return True
        except Exception:
            pass

    return False
