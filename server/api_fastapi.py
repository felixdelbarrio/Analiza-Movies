from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response


# ============================================================
# Configuración de paths (con fallbacks)
# ============================================================

# server/api_fastapi.py  -> repo_root = parents[1]
BASE_DIR = Path(__file__).resolve().parents[1]


def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


def resolve_path(env_name: str, candidates: List[Path]) -> Path:
    raw = (os.getenv(env_name) or "").strip().strip('"').strip("'")
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        return p
    found = _first_existing(candidates)
    if found is None:
        # devolvemos el primer candidato (aunque no exista) para que el error sea explícito
        return candidates[0]
    return found


OMDB_CACHE_PATH = resolve_path(
    "OMDB_CACHE_PATH",
    [
        BASE_DIR / "data" / "omdb_cache.json",
        BASE_DIR / "omdb_cache.json",
    ],
)

WIKI_CACHE_PATH = resolve_path(
    "WIKI_CACHE_PATH",
    [
        BASE_DIR / "data" / "wiki_cache.json",
        BASE_DIR / "wiki_cache.json",
    ],
)

REPORT_ALL_PATH = resolve_path(
    "REPORT_ALL_PATH",
    [
        BASE_DIR / "reports" / "report_all.csv",
        BASE_DIR / "report_all.csv",
    ],
)

REPORT_FILTERED_PATH = resolve_path(
    "REPORT_FILTERED_PATH",
    [
        BASE_DIR / "reports" / "report_filtered.csv",
        BASE_DIR / "report_filtered.csv",
    ],
)

METADATA_FIX_PATH = resolve_path(
    "METADATA_FIX_PATH",
    [
        BASE_DIR / "reports" / "metadata_fix.csv",
        BASE_DIR / "metadata_fix.csv",
    ],
)


# ============================================================
# Cache de ficheros (mtime + carga en memoria)
# ============================================================


@dataclass
class CachedFile:
    mtime_ns: int
    data: Any


_json_cache: Dict[str, CachedFile] = {}
_csv_cache: Dict[str, CachedFile] = {}


def _mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def load_json_cached(path: Path) -> Any:
    key = str(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    m = _mtime_ns(path)
    cached = _json_cache.get(key)
    if cached and cached.mtime_ns == m:
        return cached.data

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    _json_cache[key] = CachedFile(mtime_ns=m, data=data)
    return data


TEXT_COLUMNS = ["poster_url", "trailer_url", "omdb_json"]


def load_csv_cached(path: Path) -> pd.DataFrame:
    key = str(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    m = _mtime_ns(path)
    cached = _csv_cache.get(key)
    if cached and cached.mtime_ns == m:
        return cached.data

    dtype_map: Dict[str, Any] = {c: "string" for c in TEXT_COLUMNS}
    df = pd.read_csv(path, dtype=dtype_map, encoding="utf-8")

    # Normaliza text columns si existen
    for col in TEXT_COLUMNS:
        if col in df.columns:
            try:
                df[col] = df[col].astype("string")
            except Exception:
                df[col] = df[col].astype(str)

    _csv_cache[key] = CachedFile(mtime_ns=m, data=df)
    return df


def _hash_file_quick(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _paginate(items: List[Any], offset: int, limit: int) -> List[Any]:
    if offset < 0:
        offset = 0
    if limit <= 0:
        limit = 1
    return items[offset : offset + limit]


def _df_to_page(df: pd.DataFrame, *, offset: int, limit: int, query: Optional[str]) -> Dict[str, Any]:
    view = df

    if query:
        q = query.strip().lower()
        if q:
            candidate_cols = [c for c in ["title", "name", "file", "path", "imdb_id", "imdbID"] if c in view.columns]
            if candidate_cols:
                mask = None
                for c in candidate_cols:
                    s = view[c].astype("string").fillna("").str.lower().str.contains(q)
                    mask = s if mask is None else (mask | s)
                if mask is not None:
                    view = view[mask]

    total = int(len(view))
    page = view.iloc[offset : offset + limit]

    # Convertimos NaN/NA a None para JSON
    items = page.where(pd.notnull(page), None).to_dict(orient="records")
    return {"items": items, "total": total, "limit": limit, "offset": offset}


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(title="Analiza Movies Public API", version="1.0.0")

# CORS configurable
cors_raw = (os.getenv("CORS_ORIGINS") or "*").strip()
allow_origins = ["*"] if cors_raw == "*" else [o.strip() for o in cors_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/meta/files")
def meta_files() -> Dict[str, Any]:
    paths = {
        "omdb_cache": OMDB_CACHE_PATH,
        "wiki_cache": WIKI_CACHE_PATH,
        "report_all": REPORT_ALL_PATH,
        "report_filtered": REPORT_FILTERED_PATH,
        "metadata_fix": METADATA_FIX_PATH,
    }

    out: Dict[str, Any] = {}
    for k, p in paths.items():
        try:
            exists = p.exists()
            stat = p.stat() if exists else None
            out[k] = {
                "path": str(p),
                "exists": bool(exists),
                "size": int(stat.st_size) if stat else None,
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat() if stat else None,
                "sha256": _hash_file_quick(p) if exists else None,
            }
        except Exception as exc:
            out[k] = {"path": str(p), "exists": False, "error": repr(exc)}

    return out


# -----------------------
# CSV endpoints
# -----------------------


@app.get("/reports/all")
def reports_all(
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    query: Optional[str] = Query(None, description="Búsqueda simple (title/file/imdb)"),
) -> Dict[str, Any]:
    try:
        df = load_csv_cached(REPORT_ALL_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No encontrado: {REPORT_ALL_PATH}")

    return _df_to_page(df, offset=offset, limit=limit, query=query)


@app.get("/reports/filtered")
def reports_filtered(
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    query: Optional[str] = Query(None),
    empty_as_204: bool = Query(True, description="Si no existe, devuelve 204"),
) -> Any:
    if not REPORT_FILTERED_PATH.exists():
        if empty_as_204:
            # 204 debe ir sin body para máxima compatibilidad
            return Response(status_code=204)
        raise HTTPException(status_code=404, detail=f"No encontrado: {REPORT_FILTERED_PATH}")

    try:
        df = load_csv_cached(REPORT_FILTERED_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error leyendo report_filtered: {exc!r}")

    return _df_to_page(df, offset=offset, limit=limit, query=query)


@app.get("/reports/metadata-fix")
def metadata_fix(
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    query: Optional[str] = Query(None),
) -> Dict[str, Any]:
    try:
        df = load_csv_cached(METADATA_FIX_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No encontrado: {METADATA_FIX_PATH}")

    return _df_to_page(df, offset=offset, limit=limit, query=query)


# -----------------------
# Cache OMDb endpoints
# -----------------------


def _omdb_payload() -> Dict[str, Any]:
    data = load_json_cached(OMDB_CACHE_PATH)
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="omdb_cache.json no es un objeto JSON")
    return data


@app.get("/cache/omdb/records")
def omdb_records(
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="Filtra por record.status (ok/not_found/error)"),
) -> Dict[str, Any]:
    payload = _omdb_payload()
    records = payload.get("records")
    if not isinstance(records, dict):
        raise HTTPException(status_code=500, detail="omdb_cache.json: falta 'records' dict")

    rids = sorted(records.keys())

    if status:
        wanted = status.strip().lower()
        filtered: List[str] = []
        for rid in rids:
            rec = records.get(rid) or {}
            if str(rec.get("status", "")).lower() == wanted:
                filtered.append(rid)
        rids = filtered

    total = len(rids)
    page_rids = _paginate(rids, offset, limit)
    items = [{"rid": rid, **(records.get(rid) or {})} for rid in page_rids]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@app.get("/cache/omdb/by-imdb/{imdb_id}")
def omdb_by_imdb(imdb_id: str) -> Dict[str, Any]:
    payload = _omdb_payload()
    records = payload.get("records") or {}
    index_imdb = payload.get("index_imdb") or {}

    rid = index_imdb.get(imdb_id)
    if not rid:
        raise HTTPException(status_code=404, detail=f"imdb_id no encontrado en index_imdb: {imdb_id}")

    rec = records.get(str(rid))
    if not rec:
        raise HTTPException(status_code=404, detail=f"rid no encontrado en records: {rid}")

    return {"rid": str(rid), **rec}


@app.get("/cache/omdb/by-title-year")
def omdb_by_title_year(
    title: str = Query(..., min_length=1),
    year: Optional[str] = Query(None, description="Año (p.ej. 1999)"),
) -> Dict[str, Any]:
    payload = _omdb_payload()
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
        raise HTTPException(status_code=404, detail=f"rid no encontrado en records: {rid}")

    return {"rid": str(rid), **rec}


@app.get("/cache/omdb/record/{rid}")
def omdb_by_rid(rid: str) -> Dict[str, Any]:
    payload = _omdb_payload()
    records = payload.get("records") or {}
    rec = records.get(str(rid))
    if not rec:
        raise HTTPException(status_code=404, detail=f"rid no encontrado: {rid}")
    return {"rid": str(rid), **rec}


# -----------------------
# Cache Wiki endpoints
# -----------------------


def _wiki_payload() -> Dict[str, Any]:
    data = load_json_cached(WIKI_CACHE_PATH)
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="wiki_cache.json no es un objeto JSON")
    return data


@app.get("/cache/wiki/records")
def wiki_records(
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
) -> Dict[str, Any]:
    payload = _wiki_payload()
    records = payload.get("records")
    if not isinstance(records, dict):
        raise HTTPException(status_code=500, detail="wiki_cache.json: falta 'records' dict")

    rids = sorted(records.keys())

    if status:
        wanted = status.strip().lower()
        filtered: List[str] = []
        for rid in rids:
            rec = records.get(rid) or {}
            if str(rec.get("status", "")).lower() == wanted:
                filtered.append(rid)
        rids = filtered

    total = len(rids)
    page_rids = _paginate(rids, offset, limit)
    items = [{"rid": rid, **(records.get(rid) or {})} for rid in page_rids]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@app.get("/cache/wiki/by-imdb/{imdb_id}")
def wiki_by_imdb(imdb_id: str) -> Dict[str, Any]:
    payload = _wiki_payload()
    records = payload.get("records") or {}
    index_imdb = payload.get("index_imdb") or payload.get("index") or {}

    rid = index_imdb.get(imdb_id) or index_imdb.get(f"imdb:{imdb_id}")
    if not rid:
        raise HTTPException(status_code=404, detail=f"imdb_id no encontrado: {imdb_id}")

    rec = records.get(str(rid))
    if not rec:
        raise HTTPException(status_code=404, detail=f"rid no encontrado en records: {rid}")

    return {"rid": str(rid), **rec}


@app.get("/cache/wiki/record/{rid}")
def wiki_by_rid(rid: str) -> Dict[str, Any]:
    payload = _wiki_payload()
    records = payload.get("records") or {}
    rec = records.get(str(rid))
    if not rec:
        raise HTTPException(status_code=404, detail=f"rid no encontrado: {rid}")
    return {"rid": str(rid), **rec}


# ============================================================
# Entrypoint para integración con setup.py (console_scripts)
# ============================================================

def main() -> None:
    """
    Runner para poder ejecutar:
      analiza-api
    (configurable por env vars)
    """
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "1") == "1"

    uvicorn.run("server.api_fastapi:app", host=host, port=port, reload=reload)