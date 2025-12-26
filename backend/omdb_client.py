from __future__ import annotations

"""
backend/omdb_client.py (schema v4)

Cliente OMDb + caché persistente indexada + TTL + bounded in-memory hot-cache
+ compaction en disco + limitador global + métricas + resumen final de métricas.

🧠 Principios
-------------
1) Fail-safe:
   - Fallos de red, rate-limit, o API key inválida NO deben romper el run.
   - Tras N fallos consecutivos (configurable) OMDb se desactiva para el run
     y el sistema sigue “cache-only”.

2) Cache-first:
   - Siempre intentamos resolver por caché (persistente + hot-cache).
   - Incluye negative caching persistente: "Movie not found!" se guarda con TTL propio.

3) ThreadPool safe:
   - requests.Session compartida (pooling) + Retry.
   - Semaphore global para limitar concurrencia.
   - Throttle global (mínimo intervalo entre llamadas) para no saturar OMDb.

4) Logs coherentes:
   - Usamos backend/logger.py:
       - logger.debug_ctx("OMDB", ...) para diagnóstico (gated por DEBUG_MODE).
       - logger.warning(..., always=True) para avisos importantes.
       - logger.progress(...) para salida limpia en SILENT+DEBUG (si procede).
   - Este módulo NO reimplementa SILENT/DEBUG: respeta el facade.

Configuración (backend/config.py)
--------------------------------
Este módulo consume variables definidas en config.py (sin try/except).

Core:
- OMDB_API_KEY
- OMDB_BASE_URL
- OMDB_HTTP_TIMEOUT_SECONDS
- OMDB_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT
- OMDB_HTTP_RETRY_TOTAL
- OMDB_HTTP_RETRY_BACKOFF_FACTOR
- OMDB_HTTP_USER_AGENT
- OMDB_DISABLE_AFTER_N_FAILURES

Rate limiting:
- OMDB_HTTP_MAX_CONCURRENCY
- OMDB_HTTP_MIN_INTERVAL_SECONDS
- OMDB_RATE_LIMIT_MAX_RETRIES
- OMDB_RATE_LIMIT_WAIT_SECONDS

Cache v4 (paths/TTL/flush/caps):
- OMDB_CACHE_PATH
- OMDB_CACHE_TTL_OK_SECONDS
- OMDB_CACHE_TTL_NOT_FOUND_SECONDS
- OMDB_CACHE_TTL_EMPTY_RATINGS_SECONDS
- OMDB_CACHE_FLUSH_MAX_DIRTY_WRITES
- OMDB_CACHE_FLUSH_MAX_SECONDS
- ANALIZA_OMDB_CACHE_MAX_RECORDS
- ANALIZA_OMDB_CACHE_MAX_INDEX_IMDB
- ANALIZA_OMDB_CACHE_MAX_INDEX_TY
- ANALIZA_OMDB_HOT_CACHE_MAX

✅ Métricas logging
------------------
Resumen final controlado por config.py:

- OMDB_METRICS_ENABLED
- OMDB_METRICS_TOP_N
- OMDB_METRICS_LOG_ON_SILENT_DEBUG
- OMDB_METRICS_LOG_EVEN_IF_ZERO

Compatibilidad (API pública estable)
------------------------------------
- omdb_request
- omdb_query_with_cache
- patch_cached_omdb_record
- iter_cached_omdb_records
- get_omdb_metrics_snapshot / reset_omdb_metrics
- search_omdb_by_imdb_id / search_omdb_by_title_and_year / search_omdb_with_candidates
- flush_omdb_cache()
- log_omdb_metrics_summary()

Notas de esquema (v4)
---------------------
- Sin migración automática: schema mismatch => cache vacío (fail-safe).
- Records únicos:
    records[rid] -> OmdbCacheItem
    index_imdb[imdb_id] -> rid
    index_ty["<norm_title>|<year_str>"] -> rid
- rid estable:
    - "imdb:<id>" si existe imdb
    - else "ty:<norm_title>|<year_str>"
"""

import atexit
import json
import os
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, Literal, TypedDict

import requests  # type: ignore[import-not-found]
from requests import Response  # type: ignore[import-not-found]
from requests.adapters import HTTPAdapter  # type: ignore[import-not-found]
from requests.exceptions import RequestException  # type: ignore[import-not-found]
from urllib3.util.retry import Retry  # type: ignore[import-not-found]

from backend import logger as logger
from backend.config_omdb import (
    ANALIZA_OMDB_CACHE_MAX_INDEX_IMDB,
    ANALIZA_OMDB_CACHE_MAX_INDEX_TY,
    ANALIZA_OMDB_CACHE_MAX_RECORDS,
    ANALIZA_OMDB_HOT_CACHE_MAX,
    OMDB_API_KEY,
    OMDB_BASE_URL,
    OMDB_CACHE_FLUSH_MAX_DIRTY_WRITES,
    OMDB_CACHE_FLUSH_MAX_SECONDS,
    OMDB_CACHE_PATH,
    OMDB_CACHE_TTL_EMPTY_RATINGS_SECONDS,
    OMDB_CACHE_TTL_NOT_FOUND_SECONDS,
    OMDB_CACHE_TTL_OK_SECONDS,
    OMDB_DISABLE_AFTER_N_FAILURES,
    OMDB_HTTP_MAX_CONCURRENCY,
    OMDB_HTTP_MIN_INTERVAL_SECONDS,
    OMDB_HTTP_RETRY_BACKOFF_FACTOR,
    OMDB_HTTP_RETRY_TOTAL,
    OMDB_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT,
    OMDB_HTTP_TIMEOUT_SECONDS,
    OMDB_HTTP_USER_AGENT,
    OMDB_METRICS_ENABLED,
    OMDB_METRICS_LOG_EVEN_IF_ZERO,
    OMDB_METRICS_LOG_ON_SILENT_DEBUG,
    OMDB_METRICS_TOP_N,
    OMDB_RATE_LIMIT_MAX_RETRIES,
    OMDB_RATE_LIMIT_WAIT_SECONDS,
)
from backend.movie_input import normalize_title_for_lookup

# ============================================================
#                       CACHE v4 (records + indexes + TTL)
# ============================================================

CacheStatus = Literal["ok", "not_found", "empty_ratings"]


class OmdbCacheItem(TypedDict):
    """
    Record del cache (schema v4).

    Title/Year:
      - Title: normalize_title_for_lookup(title) (canon del proyecto)
      - Year: str(year) o "" si desconocido

    imdbID:
      - normalizado a lower (si existe)

    omdb:
      - payload OMDb + campos internos opcionales (__prov, __wiki)

    fetched_at / ttl_s:
      - epoch seconds + TTL (segundos)

    status:
      - ok | not_found | empty_ratings
    """

    Title: str
    Year: str
    imdbID: str | None
    omdb: dict[str, object]
    fetched_at: int
    ttl_s: int
    status: CacheStatus


class OmdbCacheFile(TypedDict):
    """
    Estructura del fichero cache (schema v4).

    records:
      - dict[rid, OmdbCacheItem]
        rid estable:
          - "imdb:<id>" si existe imdb
          - else "ty:<norm_title>|<year_str>"

    index_imdb:
      - dict[imdb_id_norm, rid]

    index_ty:
      - dict["<norm_title>|<year_str>", rid]
    """

    schema: int
    records: dict[str, OmdbCacheItem]
    index_imdb: dict[str, str]
    index_ty: dict[str, str]


_SCHEMA_VERSION: Final[int] = 4
_CACHE_PATH: Final[Path] = Path(OMDB_CACHE_PATH)

_CACHE: OmdbCacheFile | None = None
_CACHE_LOCK = threading.RLock()  # reentrante (helpers encadenados)

# Flush batching
_FLUSH_MAX_DIRTY_WRITES: Final[int] = max(1, int(OMDB_CACHE_FLUSH_MAX_DIRTY_WRITES))
_FLUSH_MAX_SECONDS: Final[float] = max(0.1, float(OMDB_CACHE_FLUSH_MAX_SECONDS))

_CACHE_DIRTY: bool = False
_CACHE_DIRTY_WRITES: int = 0
_CACHE_LAST_FLUSH_TS: float = 0.0

# Caps para compaction
_COMPACT_MAX_RECORDS: Final[int] = int(ANALIZA_OMDB_CACHE_MAX_RECORDS)
_COMPACT_MAX_INDEX_IMDB: Final[int] = int(ANALIZA_OMDB_CACHE_MAX_INDEX_IMDB)
_COMPACT_MAX_INDEX_TY: Final[int] = int(ANALIZA_OMDB_CACHE_MAX_INDEX_TY)

# Bounded in-memory hot-cache (acelerador intra-run)
_HOT_CACHE_MAX: Final[int] = int(ANALIZA_OMDB_HOT_CACHE_MAX)
_HOT_CACHE_LOCK = threading.Lock()
_HOT_CACHE: dict[str, tuple[object, float]] = {}

# Sentinel explícito para negative cache intra-run (no confundir con persistente "not_found")
_HOT_MISS: Final[object] = object()


def _now_epoch() -> int:
    """Epoch seconds (para TTL persistente)."""
    return int(time.time())


def _ty_key(norm_title: str, norm_year: str) -> str:
    """Key estable para index_ty: "<norm_title>|<year_str>"."""
    return f"{norm_title}|{norm_year}"


def _cache_key_for_imdb(imdb_id: str) -> str:
    """Key estable hot-cache para búsquedas por imdb."""
    return f"imdb:{imdb_id.lower()}"


def _cache_key_for_title_year(norm_title: str, norm_year: str) -> str:
    """Key estable hot-cache para búsquedas por title+year."""
    return f"ty:{_ty_key(norm_title, norm_year)}"


def _rid_for_record(*, imdb_norm: str | None, norm_title: str, norm_year: str) -> str:
    """
    RID estable v4:
      - imdb:<id> si hay imdb
      - else ty:<title>|<year>
    """
    if imdb_norm:
        return f"imdb:{imdb_norm}"
    return f"ty:{_ty_key(norm_title, norm_year)}"


def _is_expired_item(item: OmdbCacheItem, now_epoch: int) -> bool:
    """
    Expiración TTL (persistente).
    Si faltan campos, consideramos expirado (fail-safe).
    """
    fetched_at = int(item.get("fetched_at", 0))
    ttl_s = int(item.get("ttl_s", 0))
    if fetched_at <= 0 or ttl_s <= 0:
        return True
    return (now_epoch - fetched_at) > ttl_s


# ============================================================
#                  MÉTRICAS (ThreadPool safe)
# ============================================================

_METRICS_LOCK = threading.Lock()
_METRICS: dict[str, int] = {
    "cache_hits": 0,
    "cache_misses": 0,
    "cache_store_writes": 0,
    "cache_patch_writes": 0,
    "cache_flush_writes": 0,
    "http_requests": 0,
    "http_failures": 0,
    "throttle_sleeps": 0,
    "rate_limit_hits": 0,
    "rate_limit_sleeps": 0,
    "disabled_switches": 0,
    "candidate_search_calls": 0,
    "cache_expired_hits": 0,
    "hot_cache_hits": 0,
    "hot_cache_misses": 0,
    "hot_cache_evictions": 0,
}


def _m_inc(key: str, delta: int = 1) -> None:
    """Incrementa contador de métrica de forma thread-safe."""
    with _METRICS_LOCK:
        _METRICS[key] = int(_METRICS.get(key, 0)) + int(delta)


def get_omdb_metrics_snapshot() -> dict[str, int]:
    """Snapshot thread-safe de métricas (telemetría local)."""
    with _METRICS_LOCK:
        return dict(_METRICS)


def reset_omdb_metrics() -> None:
    """Resetea métricas a 0."""
    with _METRICS_LOCK:
        for k in list(_METRICS.keys()):
            _METRICS[k] = 0


# ============================================================
# MÉTRICAS: LOG AL FINAL DEL RUN (estilo Plex)
# ============================================================

def _metrics_any_nonzero(snapshot: Mapping[str, int]) -> bool:
    """True si alguna métrica tiene valor != 0."""
    try:
        return any(int(v) != 0 for v in snapshot.values())
    except (ValueError, TypeError):
        return True


def _format_metrics_top(snapshot: Mapping[str, int], top_n: int) -> list[tuple[str, int]]:
    """
    Devuelve top-N por valor desc, estable por key si empatan.
    - Filtra ceros para que el resumen sea útil y compacto.
    """
    items: list[tuple[str, int]] = []
    for k, v in snapshot.items():
        try:
            iv = int(v)
        except (ValueError, TypeError):
            continue
        if iv == 0:
            continue
        items.append((str(k), iv))

    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items[: max(1, int(top_n))]


def log_omdb_metrics_summary(*, force: bool = False) -> None:
    """
    Log de métricas al final del run (o manual).

    - Respeta SILENT/DEBUG mediante backend/logger.py.
    - No hace I/O de disco: solo imprime contadores.
    """
    if not OMDB_METRICS_ENABLED and not force:
        return

    snap = get_omdb_metrics_snapshot()

    if (not OMDB_METRICS_LOG_EVEN_IF_ZERO) and (not force):
        if not _metrics_any_nonzero(snap):
            return

    silent = logger.is_silent_mode()
    debug = logger.is_debug_mode()

    if silent and (not debug) and (not force):
        return

    if silent and debug and (not OMDB_METRICS_LOG_ON_SILENT_DEBUG) and (not force):
        return

    try:
        top_n = int(OMDB_METRICS_TOP_N)
    except (ValueError, TypeError):
        top_n = 12
    top_n = max(1, min(100, top_n))

    top = _format_metrics_top(snap, top_n=top_n)

    header = "[OMDB][METRICS] summary"
    lines: list[str] = [header]

    if not top:
        lines.append("  (all zeros)")
    else:
        max_k = max((len(k) for k, _ in top), default=10)
        for k, v in top:
            lines.append(f"  {k.ljust(max_k)} : {v}")

    if silent and debug:
        for ln in lines:
            logger.progress(ln)
    else:
        for ln in lines:
            logger.info(ln)


# ============================================================
# LOGGING (centralizado en backend/logger.py)
# ============================================================

def _dbg(msg: object) -> None:
    """Diagnóstico contextual (solo si DEBUG_MODE=True)."""
    logger.debug_ctx("OMDB", msg)


def _warn_always(msg: object) -> None:
    """Aviso importante (visible incluso en SILENT_MODE)."""
    logger.warning(str(msg), always=True)


# ============================================================
# HTTP session + retry (lazy-init thread-safe)
# ============================================================

_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()


def _cap_int_runtime(value: int, *, min_v: int, max_v: int) -> int:
    """Cap defensivo (por si config/env cambia)."""
    if value < min_v:
        return min_v
    if value > max_v:
        return max_v
    return value


def _cap_float_runtime(value: float, *, min_v: float, max_v: float | None = None) -> float:
    """Cap defensivo (por si config/env cambia)."""
    if value < min_v:
        return min_v
    if max_v is not None and value > max_v:
        return max_v
    return value


def _get_session() -> requests.Session:
    """
    Singleton requests.Session con retries y pooling ajustado al nivel de concurrencia.

    Motivo:
    - Si el pool de conexiones es menor que la concurrencia efectiva (ThreadPool),
      el tráfico se serializa internamente y se pierde rendimiento.

    Política:
    - pool_connections / pool_maxsize se ajustan en torno a OMDB_HTTP_MAX_CONCURRENCY
      (cap defensivo para no crear pools gigantes).
    - Retry de urllib3 gestiona 429/5xx best-effort (no es garantía de éxito).
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION

        retry_total = _cap_int_runtime(int(OMDB_HTTP_RETRY_TOTAL), min_v=0, max_v=10)
        backoff = _cap_float_runtime(float(OMDB_HTTP_RETRY_BACKOFF_FACTOR), min_v=0.0, max_v=10.0)

        # Pool sizing: alineado con el límite de concurrencia del módulo.
        # Cap defensivo: si alguien pone 1000 en config, no queremos un pool gigante.
        pool_size = _cap_int_runtime(int(OMDB_HTTP_MAX_CONCURRENCY), min_v=1, max_v=64)

        session = requests.Session()

        retries = Retry(
            total=retry_total,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=pool_size,
            pool_maxsize=pool_size,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        ua = str(OMDB_HTTP_USER_AGENT).strip() or "Analiza-Movies/1.0 (local)"
        session.headers.update(
            {
                "User-Agent": ua,
                "Accept": "application/json,text/plain,*/*",
            }
        )

        _SESSION = session
        return session


# ============================================================
# LIMITADOR GLOBAL OMDb (ThreadPool safe)
# ============================================================

# Concurrencia máxima de requests OMDb (cap defensivo por seguridad)
_MAX_CONCURRENCY_RT: Final[int] = _cap_int_runtime(int(OMDB_HTTP_MAX_CONCURRENCY), min_v=1, max_v=256)
_OMDB_HTTP_SEMAPHORE = threading.Semaphore(_MAX_CONCURRENCY_RT)

# Throttle global (intervalo mínimo entre requests)
_OMDB_HTTP_THROTTLE_LOCK = threading.Lock()
_OMDB_HTTP_LAST_REQUEST_TS: float = 0.0


# ============================================================
# FLAGS / FAIL-SAFE
# ============================================================

OMDB_DISABLED: bool = False
OMDB_DISABLED_NOTICE_SHOWN: bool = False
OMDB_RATE_LIMIT_NOTICE_SHOWN: bool = False
OMDB_INVALID_KEY_NOTICE_SHOWN: bool = False

_OMDB_CONSECUTIVE_FAILURES: int = 0
_OMDB_FAILURES_LOCK = threading.Lock()

_DISABLE_AFTER_N_FAILURES: Final[int] = _cap_int_runtime(int(OMDB_DISABLE_AFTER_N_FAILURES), min_v=1, max_v=50)


def _mark_omdb_failure() -> None:
    """
    Fail-safe:
    - Tras N fallos consecutivos, deshabilita OMDb para el resto del run.
    - Evita runs rotos por caídas / rate limit / problemas de red.

    Importante:
    - N cuenta “fallos de transporte/protocolo”, no “Movie not found!”.
    - Si OMDb se deshabilita, seguimos resolviendo por caché (persistente/hot).
    """
    global OMDB_DISABLED, OMDB_DISABLED_NOTICE_SHOWN, _OMDB_CONSECUTIVE_FAILURES

    with _OMDB_FAILURES_LOCK:
        _OMDB_CONSECUTIVE_FAILURES += 1
        fails = _OMDB_CONSECUTIVE_FAILURES

    if fails < _DISABLE_AFTER_N_FAILURES:
        return

    if not OMDB_DISABLED:
        OMDB_DISABLED = True
        _m_inc("disabled_switches", 1)

    if not OMDB_DISABLED_NOTICE_SHOWN:
        _warn_always(
            "ERROR: OMDb desactivado para esta ejecución tras fallos consecutivos. "
            "A partir de ahora se usará únicamente la caché local."
        )
        OMDB_DISABLED_NOTICE_SHOWN = True


def _mark_omdb_success() -> None:
    """Resetea contador de fallos consecutivos tras una respuesta válida."""
    global _OMDB_CONSECUTIVE_FAILURES
    with _OMDB_FAILURES_LOCK:
        _OMDB_CONSECUTIVE_FAILURES = 0


def _omdb_http_get(*, base_url: str, params: Mapping[str, str], timeout_seconds: float) -> Response | None:
    """
    GET a OMDb con:
    - semaphore (concurrencia)
    - throttle (intervalo global)
    - métricas
    - manejo defensivo

    Devuelve:
      - Response en caso de éxito
      - None si OMDb está deshabilitado, no se adquirió el semaphore,
        o falló la request.
    """
    global _OMDB_HTTP_LAST_REQUEST_TS

    if OMDB_DISABLED:
        return None

    acquire_timeout = _cap_float_runtime(float(OMDB_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT), min_v=0.1, max_v=120.0)
    acquired = _OMDB_HTTP_SEMAPHORE.acquire(timeout=acquire_timeout)
    if not acquired:
        _dbg(f"Semaphore acquire timeout ({acquire_timeout:.1f}s). Skipping OMDb request.")
        return None

    try:
        min_interval = max(0.0, float(OMDB_HTTP_MIN_INTERVAL_SECONDS))
        if min_interval > 0.0:
            with _OMDB_HTTP_THROTTLE_LOCK:
                now = time.monotonic()
                wait_s = (_OMDB_HTTP_LAST_REQUEST_TS + min_interval) - now
                if wait_s > 0.0:
                    _m_inc("throttle_sleeps", 1)
                    _dbg(f"Throttle: sleeping {wait_s:.3f}s")
                    time.sleep(wait_s)
                _OMDB_HTTP_LAST_REQUEST_TS = time.monotonic()

        _m_inc("http_requests", 1)
        session = _get_session()

        t = _cap_float_runtime(float(timeout_seconds), min_v=0.5, max_v=120.0)
        return session.get(base_url, params=dict(params), timeout=t)

    except RequestException as exc:
        _m_inc("http_failures", 1)
        _dbg(f"HTTP error calling OMDb: {exc!r}")
        return None

    finally:
        _OMDB_HTTP_SEMAPHORE.release()


# ============================================================
# AUX: safe parsing / ratings
# ============================================================

def _safe_int(value: object) -> int | None:
    """Cast defensivo a int (None/ValueError/TypeError => None)."""
    try:
        if value is None:
            return None
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_float(value: object) -> float | None:
    """Cast defensivo a float (None/ValueError/TypeError => None)."""
    try:
        if value is None:
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_imdb_id(value: object) -> str | None:
    """
    Normaliza imdbID: strip + lower.
    En el proyecto tratamos IDs en minúscula para keys estables.
    """
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v.lower() if v else None


def normalize_imdb_votes(votes: object) -> int | None:
    """Convierte imdbVotes ("12,345") a int."""
    if not votes or votes == "N/A":
        return None
    if isinstance(votes, (int, float)):
        return int(votes)
    s = str(votes).strip().replace(",", "")
    return _safe_int(s)


def parse_rt_score_from_omdb(omdb_data: Mapping[str, object]) -> int | None:
    """Extrae Rotten Tomatoes (%)."""
    ratings_obj = omdb_data.get("Ratings") or []
    if not isinstance(ratings_obj, list):
        return None
    for r in ratings_obj:
        if not isinstance(r, Mapping):
            continue
        if r.get("Source") != "Rotten Tomatoes":
            continue
        val = r.get("Value")
        if not isinstance(val, str) or not val.endswith("%"):
            continue
        try:
            return int(val[:-1])
        except ValueError:
            return None
    return None


def parse_imdb_rating_from_omdb(omdb_data: Mapping[str, object]) -> float | None:
    """Extrae imdbRating (float)."""
    raw = omdb_data.get("imdbRating")
    if not raw or raw == "N/A":
        return None
    return _safe_float(raw)


def extract_year_from_omdb(omdb_data: Mapping[str, object]) -> int | None:
    """Extrae el año principal ("1994" o "1994–1996")."""
    raw = omdb_data.get("Year")
    if not raw or raw == "N/A":
        return None
    text = str(raw).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def extract_ratings_from_omdb(data: Mapping[str, object] | None) -> tuple[float | None, int | None, int | None]:
    """Devuelve (imdb_rating, imdb_votes, rt_score)."""
    if not data:
        return None, None, None
    imdb_rating = parse_imdb_rating_from_omdb(data)
    imdb_votes = normalize_imdb_votes(data.get("imdbVotes"))
    rt_score = parse_rt_score_from_omdb(data)
    return imdb_rating, imdb_votes, rt_score


def is_omdb_data_empty_for_ratings(data: Mapping[str, object] | None) -> bool:
    """
    Señal “empty ratings”:
    OMDb devuelve ficha, pero sin métricas útiles.
    Sirve para usar TTL intermedio y permitir refresh futuro.
    """
    if not data:
        return True
    imdb_rating = parse_imdb_rating_from_omdb(data)
    imdb_votes = normalize_imdb_votes(data.get("imdbVotes"))
    rt_score = parse_rt_score_from_omdb(data)
    return imdb_rating is None and imdb_votes is None and rt_score is None


def _extract_imdb_id_from_omdb_record(data: Mapping[str, object] | None) -> str | None:
    """Extrae imdbID de una respuesta OMDb con Response=True."""
    if not isinstance(data, Mapping):
        return None
    if data.get("Response") != "True":
        return None
    return _safe_imdb_id(data.get("imdbID"))


def _norm_year_str(year: int | None) -> str:
    """Normaliza year a str; year desconocido => ""."""
    return str(year) if year is not None else ""


def _is_movie_not_found(data: Mapping[str, object]) -> bool:
    """Detecta el not_found “estándar” de OMDb."""
    return data.get("Response") == "False" and data.get("Error") == "Movie not found!"


def _is_invalid_api_key(data: Mapping[str, object]) -> bool:
    """Detecta respuesta de API key inválida."""
    err = data.get("Error")
    return data.get("Response") == "False" and isinstance(err, str) and "api key" in err.lower()


def _pick_ttl_and_status(omdb_data: Mapping[str, object]) -> tuple[int, CacheStatus]:
    """
    TTL/status:
    - not_found: negative caching persistente (TTL propio)
    - empty_ratings: TTL intermedio
    - ok: TTL largo
    """
    if _is_movie_not_found(omdb_data):
        return max(60, int(OMDB_CACHE_TTL_NOT_FOUND_SECONDS)), "not_found"

    if is_omdb_data_empty_for_ratings(omdb_data):
        return max(60, int(OMDB_CACHE_TTL_EMPTY_RATINGS_SECONDS)), "empty_ratings"

    return max(60, int(OMDB_CACHE_TTL_OK_SECONDS)), "ok"


# ============================================================
# WIKI MINIMAL (sanitize)
# ============================================================

def _sanitize_wiki_lookup(v: object) -> dict[str, object] | None:
    """
    Sanitiza wiki_lookup:
    - evita crecer el cache con keys inesperadas
    """
    if not isinstance(v, Mapping):
        return None

    out: dict[str, object] = {}

    via = v.get("via")
    if isinstance(via, str) and via.strip():
        out["via"] = via.strip()

    imdb_id = v.get("imdb_id")
    if isinstance(imdb_id, str) and imdb_id.strip():
        out["imdb_id"] = imdb_id.strip()

    title = v.get("title")
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()

    year = v.get("year")
    if isinstance(year, int):
        out["year"] = year
    elif isinstance(year, str) and year.strip().isdigit():
        out["year"] = int(year.strip())

    return out or None


def _sanitize_wiki_block(v: object) -> dict[str, object] | None:
    """
    Sanitiza __wiki minimal:
      - imdb_id, wikidata_id, wikipedia_title, source_language, wiki_lookup
    """
    if not isinstance(v, Mapping):
        return None

    out: dict[str, object] = {}

    imdb_id = v.get("imdb_id")
    if isinstance(imdb_id, str) and imdb_id.strip():
        out["imdb_id"] = imdb_id.strip()

    wikidata_id = v.get("wikidata_id")
    if isinstance(wikidata_id, str) and wikidata_id.strip():
        out["wikidata_id"] = wikidata_id.strip()

    wikipedia_title = v.get("wikipedia_title")
    if isinstance(wikipedia_title, str) and wikipedia_title.strip():
        out["wikipedia_title"] = wikipedia_title.strip()

    source_language = v.get("source_language")
    if isinstance(source_language, str) and source_language.strip():
        out["source_language"] = source_language.strip()

    wiki_lookup = _sanitize_wiki_lookup(v.get("wiki_lookup"))
    if wiki_lookup is not None:
        out["wiki_lookup"] = wiki_lookup

    return out or None


# ============================================================
# PROVENANCE
# ============================================================

def _build_default_provenance(*, imdb_norm: str | None, norm_title: str, year: int | None) -> dict[str, object]:
    """
    Provenance base para trazabilidad:
    - lookup_key (cómo intentamos resolver)
    - had_imdb_hint
    """
    if imdb_norm:
        lookup_key = f"imdb_id:{imdb_norm}"
    else:
        lookup_key = f"title_year:{norm_title}|{year}" if year is not None else f"title:{norm_title}"
    return {"lookup_key": lookup_key, "had_imdb_hint": bool(imdb_norm)}


def _merge_provenance(existing: Mapping[str, object] | None, incoming: Mapping[str, object] | None) -> dict[str, object]:
    """Merge shallow: incoming pisa existing."""
    out: dict[str, object] = {}
    if isinstance(existing, Mapping):
        out.update(dict(existing))
    if isinstance(incoming, Mapping):
        out.update(dict(incoming))
    return out


def _attach_provenance(omdb_data: dict[str, object], prov: Mapping[str, object] | None) -> dict[str, object]:
    """Adjunta/mergea __prov dentro del payload OMDb."""
    existing = omdb_data.get("__prov")
    merged = _merge_provenance(existing if isinstance(existing, Mapping) else None, prov)
    if merged:
        omdb_data["__prov"] = merged
    return omdb_data


def _merge_dict_shallow(dst: dict[str, object], patch: Mapping[str, object]) -> dict[str, object]:
    """
    Merge shallow con reglas especiales:
    - "__prov": merge (no replace)
    - "__wiki": sanitize (si inválido, elimina)
    - resto: asigna tal cual
    """
    for k, v in patch.items():
        if k == "__prov" and isinstance(v, Mapping):
            existing = dst.get("__prov")
            dst["__prov"] = _merge_provenance(existing if isinstance(existing, Mapping) else None, v)
            continue

        if k == "__wiki":
            sanitized = _sanitize_wiki_block(v)
            if sanitized is None:
                dst.pop("__wiki", None)
            else:
                dst["__wiki"] = sanitized
            continue

        dst[k] = v  # type: ignore[assignment]

    return dst


# ============================================================
# HOT CACHE (bounded in-memory)
# ============================================================

def _hot_get(key: str) -> object | None:
    """
    Devuelve:
      - OmdbCacheItem si HIT
      - _HOT_MISS si MISS cacheado
      - None si no existe en hot cache
    """
    if _HOT_CACHE_MAX <= 0:
        return None
    with _HOT_CACHE_LOCK:
        v = _HOT_CACHE.get(key)
        if v is None:
            _m_inc("hot_cache_misses", 1)
            return None
        obj, _ts = v
        _HOT_CACHE[key] = (obj, time.monotonic())
        _m_inc("hot_cache_hits", 1)
        return obj


def _hot_put(key: str, obj: object) -> None:
    """
    Inserta/actualiza en hot cache (bounded).
    - Si excede el máximo, evict del más viejo (LRU aproximado).
    """
    if _HOT_CACHE_MAX <= 0:
        return
    with _HOT_CACHE_LOCK:
        _HOT_CACHE[key] = (obj, time.monotonic())
        if len(_HOT_CACHE) <= _HOT_CACHE_MAX:
            return

        oldest_k: str | None = None
        oldest_ts = float("inf")
        for k, (_obj, ts) in _HOT_CACHE.items():
            if ts < oldest_ts:
                oldest_ts = ts
                oldest_k = k

        if oldest_k is not None:
            _HOT_CACHE.pop(oldest_k, None)
            _m_inc("hot_cache_evictions", 1)


# ============================================================
# LOAD/SAVE CACHE (atómico + thread-safe) + compaction
# ============================================================

def _empty_cache() -> OmdbCacheFile:
    """Crea estructura vacía del cache (schema v4)."""
    return {"schema": _SCHEMA_VERSION, "records": {}, "index_imdb": {}, "index_ty": {}}


def _maybe_quarantine_corrupt_cache() -> None:
    """
    En DEBUG: renombra el cache corrupto para inspección.
    En normal: se recrea silenciosamente (fail-safe).
    """
    if not logger.is_debug_mode():
        return
    try:
        if not _CACHE_PATH.exists():
            return
        ts = int(time.time())
        bad_path = _CACHE_PATH.with_name(f"{_CACHE_PATH.name}.corrupt.{ts}")
        os.replace(str(_CACHE_PATH), str(bad_path))
        _dbg(f"Quarantined corrupt cache file -> {bad_path.name}")
    except OSError:
        return


def _load_cache_unlocked() -> OmdbCacheFile:
    """
    Carga cache (singleton) validando estructura.
    Política v4: si schema mismatch => recrear (no migración).

    Además:
    - Saneado ligero de índices (evita rid colgantes).
    - Compaction best-effort en memoria (sin forzar I/O).
    """
    global _CACHE, _CACHE_LAST_FLUSH_TS, _CACHE_DIRTY, _CACHE_DIRTY_WRITES
    if _CACHE is not None:
        return _CACHE

    _CACHE_LAST_FLUSH_TS = time.monotonic()
    _CACHE_DIRTY = False
    _CACHE_DIRTY_WRITES = 0

    if not _CACHE_PATH.exists():
        _CACHE = _empty_cache()
        return _CACHE

    try:
        raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _maybe_quarantine_corrupt_cache()
        _CACHE = _empty_cache()
        return _CACHE

    if not isinstance(raw, Mapping) or raw.get("schema") != _SCHEMA_VERSION:
        found_schema = raw.get("schema") if isinstance(raw, Mapping) else None
        _dbg(f"cache schema mismatch -> recreate (found={found_schema!r})")
        _CACHE = _empty_cache()
        return _CACHE

    records_obj = raw.get("records")
    index_imdb_obj = raw.get("index_imdb")
    index_ty_obj = raw.get("index_ty")

    if not isinstance(records_obj, Mapping) or not isinstance(index_imdb_obj, Mapping) or not isinstance(index_ty_obj, Mapping):
        _CACHE = _empty_cache()
        return _CACHE

    records: dict[str, OmdbCacheItem] = {}
    for rid, v in records_obj.items():
        if not isinstance(rid, str) or not isinstance(v, Mapping):
            continue

        title = v.get("Title")
        year = v.get("Year")
        imdb_id = v.get("imdbID")
        omdb = v.get("omdb")
        fetched_at = v.get("fetched_at")
        ttl_s = v.get("ttl_s")
        status = v.get("status")

        if not isinstance(title, str) or not isinstance(year, str):
            continue
        if imdb_id is not None and not isinstance(imdb_id, str):
            continue
        if not isinstance(omdb, Mapping):
            continue
        if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
            continue
        if status not in ("ok", "not_found", "empty_ratings"):
            continue

        imdb_norm = imdb_id.lower() if isinstance(imdb_id, str) and imdb_id.strip() else None

        records[rid] = {
            "Title": title,
            "Year": year,
            "imdbID": imdb_norm,
            "omdb": dict(omdb),
            "fetched_at": fetched_at,
            "ttl_s": ttl_s,
            "status": status,
        }

    index_imdb: dict[str, str] = {}
    for k, v in index_imdb_obj.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip():
            index_imdb[k.strip().lower()] = v

    index_ty: dict[str, str] = {}
    for k, v in index_ty_obj.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip():
            index_ty[k.strip()] = v

    # Saneado: elimina entradas de índice que apuntan a rids inexistentes.
    index_imdb = {k: rid for k, rid in index_imdb.items() if rid in records}
    index_ty = {k: rid for k, rid in index_ty.items() if rid in records}

    _CACHE = {"schema": _SCHEMA_VERSION, "records": records, "index_imdb": index_imdb, "index_ty": index_ty}

    try:
        _compact_cache_unlocked(_CACHE, force=False)
    except Exception:
        pass

    return _CACHE


def _save_cache_file_atomic(cache: OmdbCacheFile) -> None:
    """
    Escritura atómica:
    - temp file en el mismo directorio
    - fsync
    - replace

    Nota:
    - Best-effort: si fsync falla en FS remotos, seguimos.
    """
    dirpath = _CACHE_PATH.parent
    dirpath.mkdir(parents=True, exist_ok=True)

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(dirpath)) as tf:
            json.dump(cache, tf, ensure_ascii=False, indent=2)
            tf.flush()
            try:
                os.fsync(tf.fileno())
            except OSError:
                pass
            temp_name = tf.name

        os.replace(temp_name, str(_CACHE_PATH))
    finally:
        if temp_name and os.path.exists(temp_name) and temp_name != str(_CACHE_PATH):
            try:
                os.remove(temp_name)
            except OSError:
                pass


def _mark_dirty_unlocked() -> None:
    """Marca cache como dirty para write-back batching."""
    global _CACHE_DIRTY, _CACHE_DIRTY_WRITES
    _CACHE_DIRTY = True
    _CACHE_DIRTY_WRITES += 1


def _compact_cache_unlocked(cache: OmdbCacheFile, *, force: bool) -> None:
    """
    Compaction/GC de cache (schema v4).

    Hace:
      - remove expirados
      - cap records (más recientes primero)
      - rebuild índices coherentes
      - cap índices (por seguridad)
    """
    try:
        now_epoch = _now_epoch()

        records_in = cache.get("records")
        if not isinstance(records_in, Mapping):
            cache["records"] = {}
            records_in = cache["records"]

        records: dict[str, OmdbCacheItem] = {}
        for rid, it in records_in.items():
            if not isinstance(rid, str) or not isinstance(it, Mapping):
                continue
            fetched_at = it.get("fetched_at")
            ttl_s = it.get("ttl_s")
            if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
                continue
            if _is_expired_item(it, now_epoch):  # type: ignore[arg-type]
                continue

            d = dict(it)
            imdb_norm = d.get("imdbID")
            if isinstance(imdb_norm, str):
                imdb_norm = imdb_norm.strip().lower()
                d["imdbID"] = imdb_norm or None
            else:
                d["imdbID"] = None
            records[rid] = d  # type: ignore[assignment]

        if _COMPACT_MAX_RECORDS > 0 and len(records) > _COMPACT_MAX_RECORDS:
            ranked = sorted(records.items(), key=lambda kv: int(kv[1].get("fetched_at", 0)), reverse=True)
            records = dict(ranked[:_COMPACT_MAX_RECORDS])

        cache["records"] = records

        index_imdb: dict[str, str] = {}
        index_ty: dict[str, str] = {}
        for rid, it in records.items():
            title = it.get("Title")
            year = it.get("Year")
            if isinstance(title, str) and isinstance(year, str):
                index_ty[_ty_key(title, year)] = rid
            imdb = it.get("imdbID")
            if isinstance(imdb, str) and imdb.strip():
                index_imdb[imdb.strip().lower()] = rid

        if _COMPACT_MAX_INDEX_IMDB > 0 and len(index_imdb) > _COMPACT_MAX_INDEX_IMDB:
            keep = sorted(index_imdb.keys())[:_COMPACT_MAX_INDEX_IMDB]
            index_imdb = {k: index_imdb[k] for k in keep}

        if _COMPACT_MAX_INDEX_TY > 0 and len(index_ty) > _COMPACT_MAX_INDEX_TY:
            keep2 = sorted(index_ty.keys())[:_COMPACT_MAX_INDEX_TY]
            index_ty = {k: index_ty[k] for k in keep2}

        cache["index_imdb"] = index_imdb
        cache["index_ty"] = index_ty

        _dbg(
            "cache compacted"
            + (" (force)" if force else "")
            + f" | records={len(records)} idx_imdb={len(index_imdb)} idx_ty={len(index_ty)}"
        )
    except Exception as exc:
        _dbg(f"cache compaction failed: {exc!r}")


def _maybe_flush_unlocked(force: bool) -> None:
    """
    Flush del cache si:
    - force=True
    - o dirty_writes >= umbral
    - o han pasado X segundos desde el último flush

    Antes de escribir:
    - compaction/GC (reduce tamaño y elimina expirados)
    """
    global _CACHE_DIRTY, _CACHE_DIRTY_WRITES, _CACHE_LAST_FLUSH_TS, _CACHE
    if _CACHE is None:
        return
    if not _CACHE_DIRTY and not force:
        return

    now = time.monotonic()
    should_flush = force
    if not should_flush:
        if _CACHE_DIRTY_WRITES >= _FLUSH_MAX_DIRTY_WRITES:
            should_flush = True
        elif (now - _CACHE_LAST_FLUSH_TS) >= _FLUSH_MAX_SECONDS:
            should_flush = True

    if not should_flush:
        return

    _compact_cache_unlocked(_CACHE, force=force)
    _save_cache_file_atomic(_CACHE)

    _CACHE_DIRTY = False
    _CACHE_DIRTY_WRITES = 0
    _CACHE_LAST_FLUSH_TS = now
    _m_inc("cache_flush_writes", 1)


def flush_omdb_cache() -> None:
    """
    API pública: flush explícito del cache OMDb (si está dirty).
    Orquestadores lo llaman vía flush_external_caches().
    """
    with _CACHE_LOCK:
        _maybe_flush_unlocked(force=True)


def _flush_cache_on_exit() -> None:
    """
    Asegura persistencia al salir (best-effort) y emite resumen de métricas.

    Orden recomendado:
      1) flush del cache
      2) resumen de métricas
    """
    try:
        with _CACHE_LOCK:
            _maybe_flush_unlocked(force=True)
    except Exception:
        pass

    try:
        log_omdb_metrics_summary()
    except Exception:
        pass


atexit.register(_flush_cache_on_exit)


# ============================================================
# CACHE LOOKUP POLICY (schema v4 + hot cache)
# ============================================================

def _get_cached_item_unlocked(*, norm_title: str, norm_year: str, imdb_id_hint: str | None) -> OmdbCacheItem | None:
    """
    Lookup en cache persistente.

    Orden:
      1) imdb (más estable)
      2) title+year
    """
    cache = _load_cache_unlocked()
    records = cache["records"]
    idx_imdb = cache["index_imdb"]
    idx_ty = cache["index_ty"]
    now_epoch = _now_epoch()

    if imdb_id_hint:
        rid = idx_imdb.get(imdb_id_hint)
        if isinstance(rid, str):
            it = records.get(rid)
            if it is not None:
                if _is_expired_item(it, now_epoch):
                    _m_inc("cache_expired_hits", 1)
                    return None
                return it

    rid2 = idx_ty.get(_ty_key(norm_title, norm_year))
    if isinstance(rid2, str):
        it2 = records.get(rid2)
        if it2 is not None:
            if _is_expired_item(it2, now_epoch):
                _m_inc("cache_expired_hits", 1)
                return None
            return it2

    return None


def _get_cached_item(*, norm_title: str, norm_year: str, imdb_id_hint: str | None) -> OmdbCacheItem | None:
    """
    Lookup combinado:
      - hot-cache (bounded in-memory)
      - persistente (con lock)
    """
    imdb_key = _cache_key_for_imdb(imdb_id_hint) if imdb_id_hint else None
    ty_key = _cache_key_for_title_year(norm_title, norm_year) if norm_title or norm_year else None

    if imdb_key:
        h = _hot_get(imdb_key)
        if h is _HOT_MISS:
            return None
        if isinstance(h, dict):
            return h  # type: ignore[return-value]

    if ty_key:
        h2 = _hot_get(ty_key)
        if h2 is _HOT_MISS:
            return None
        if isinstance(h2, dict):
            return h2  # type: ignore[return-value]

    with _CACHE_LOCK:
        it = _get_cached_item_unlocked(norm_title=norm_title, norm_year=norm_year, imdb_id_hint=imdb_id_hint)

    if it is not None:
        if imdb_key and isinstance(it.get("imdbID"), str):
            _hot_put(imdb_key, it)
        if ty_key:
            _hot_put(ty_key, it)
        return it

    if imdb_key:
        _hot_put(imdb_key, _HOT_MISS)
    if ty_key:
        _hot_put(ty_key, _HOT_MISS)
    return None


def _cache_store_item(*, norm_title: str, norm_year: str, imdb_id: str | None, omdb_data: dict[str, object]) -> OmdbCacheItem:
    """
    Inserta/actualiza un record en cache v4:
      - records[rid] = item
      - index_ty y index_imdb apuntan a rid
      - TTL/status según payload
      - batching del flush
      - write-through al hot-cache
    """
    now_epoch = _now_epoch()
    ttl_s, status = _pick_ttl_and_status(omdb_data)

    imdb_norm = imdb_id.lower() if isinstance(imdb_id, str) and imdb_id.strip() else None

    norm_title_final = norm_title or normalize_title_for_lookup(str(omdb_data.get("Title") or "")) or ""
    y_extracted = extract_year_from_omdb(omdb_data)
    norm_year_final = norm_year or (str(y_extracted) if y_extracted is not None else "")

    item: OmdbCacheItem = {
        "Title": norm_title_final,
        "Year": norm_year_final,
        "imdbID": imdb_norm,
        "omdb": dict(omdb_data),
        "fetched_at": now_epoch,
        "ttl_s": int(ttl_s),
        "status": status,
    }

    rid = _rid_for_record(imdb_norm=imdb_norm, norm_title=norm_title_final, norm_year=norm_year_final)

    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        cache["records"][rid] = item
        cache["index_ty"][_ty_key(norm_title_final, norm_year_final)] = rid
        if imdb_norm:
            cache["index_imdb"][imdb_norm] = rid

        _m_inc("cache_store_writes", 1)
        _mark_dirty_unlocked()
        _maybe_flush_unlocked(force=False)

    _hot_put(_cache_key_for_title_year(norm_title_final, norm_year_final), item)
    if imdb_norm:
        _hot_put(_cache_key_for_imdb(imdb_norm), item)

    return item


def iter_cached_omdb_records() -> Iterable[dict[str, object]]:
    """
    Itera payloads OMDb (sin metadatos de cache).
    En schema v4 ya no hay duplicados por diseño.
    """
    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        records_copy = list(cache["records"].values())

    for it in records_copy:
        yield dict(it["omdb"])


# ============================================================
# PATCH / WRITE-BACK AL CACHE
# ============================================================

def patch_cached_omdb_record(*, norm_title: str, norm_year: str, imdb_id: str | None, patch: Mapping[str, object]) -> bool:
    """
    Aplica un patch shallow al payload OMDb cacheado.
    """
    imdb_norm = _safe_imdb_id(imdb_id) if imdb_id else None
    ty_k = _ty_key(norm_title, norm_year)

    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        records = cache["records"]

        rid: str | None = None
        if imdb_norm:
            rid = cache["index_imdb"].get(imdb_norm)
        if not rid:
            rid = cache["index_ty"].get(ty_k)

        if not rid or rid not in records:
            return False

        target = records[rid]
        omdb_obj = target.get("omdb")
        if not isinstance(omdb_obj, dict):
            omdb_obj = dict(omdb_obj) if isinstance(omdb_obj, Mapping) else {}
            target["omdb"] = omdb_obj

        _merge_dict_shallow(omdb_obj, patch)

        ttl_s, status = _pick_ttl_and_status(omdb_obj)
        target["ttl_s"] = int(ttl_s)
        target["status"] = status
        target["fetched_at"] = _now_epoch()

        imdb_now = _safe_imdb_id(target.get("imdbID")) if target.get("imdbID") else None
        title_now = target.get("Title") if isinstance(target.get("Title"), str) else norm_title
        year_now = target.get("Year") if isinstance(target.get("Year"), str) else norm_year

        new_rid = _rid_for_record(imdb_norm=imdb_now, norm_title=title_now, norm_year=year_now)
        if new_rid != rid:
            records[new_rid] = target
            records.pop(rid, None)
            rid = new_rid

        cache["index_ty"][_ty_key(title_now, year_now)] = rid
        if imdb_now:
            cache["index_imdb"][imdb_now] = rid

        _m_inc("cache_patch_writes", 1)
        _mark_dirty_unlocked()
        _maybe_flush_unlocked(force=False)

    _hot_put(_cache_key_for_title_year(title_now, year_now), target)
    if imdb_norm:
        _hot_put(_cache_key_for_imdb(imdb_norm), target)
    if imdb_now:
        _hot_put(_cache_key_for_imdb(imdb_now), target)

    return True


# ============================================================
# PETICIONES OMDb (red)
# ============================================================

def omdb_request(params: Mapping[str, object]) -> dict[str, object] | None:
    """
    Llama a OMDb (red) con:
    - API key presente
    - disabled switch
    - parseo defensivo JSON
    - invalid key -> desactiva OMDb para el resto del run

    Devuelve:
    - dict (aunque Response=False) o None si no se pudo llamar.
    """
    global OMDB_INVALID_KEY_NOTICE_SHOWN, OMDB_DISABLED, OMDB_DISABLED_NOTICE_SHOWN

    if OMDB_API_KEY is None or not str(OMDB_API_KEY).strip():
        if not OMDB_DISABLED:
            OMDB_DISABLED = True
            _m_inc("disabled_switches", 1)
        if not OMDB_DISABLED_NOTICE_SHOWN:
            _warn_always("ERROR: OMDB_API_KEY no configurada. OMDb queda desactivado; se usará solo caché.")
            OMDB_DISABLED_NOTICE_SHOWN = True
        return None

    if OMDB_DISABLED:
        return None

    base_url = str(OMDB_BASE_URL).strip() or "https://www.omdbapi.com/"
    req_params: dict[str, str] = {str(k): str(v) for k, v in params.items()}
    req_params["apikey"] = str(OMDB_API_KEY)

    timeout_s = _cap_float_runtime(float(OMDB_HTTP_TIMEOUT_SECONDS), min_v=0.5, max_v=120.0)

    resp = _omdb_http_get(base_url=base_url, params=req_params, timeout_seconds=timeout_s)
    if resp is None:
        _mark_omdb_failure()
        return None

    if resp.status_code != 200:
        _m_inc("http_failures", 1)
        _dbg(f"OMDb status != 200: {resp.status_code}")
        _mark_omdb_failure()
        return None

    try:
        data_obj = resp.json()
    except (ValueError, json.JSONDecodeError) as exc:
        _m_inc("http_failures", 1)
        _dbg(f"OMDb invalid JSON: {exc!r}")
        _mark_omdb_failure()
        return None

    if not isinstance(data_obj, dict):
        _m_inc("http_failures", 1)
        _dbg("OMDb returned JSON not dict.")
        _mark_omdb_failure()
        return None

    if _is_invalid_api_key(data_obj):
        if not OMDB_INVALID_KEY_NOTICE_SHOWN:
            _warn_always("ERROR: OMDb respondió 'Invalid API key!'. Revisa OMDB_API_KEY.")
            OMDB_INVALID_KEY_NOTICE_SHOWN = True
        if not OMDB_DISABLED:
            OMDB_DISABLED = True
            _m_inc("disabled_switches", 1)
        _mark_omdb_failure()
        return data_obj

    _mark_omdb_success()
    return data_obj


def _is_rate_limit_response(data: Mapping[str, object]) -> bool:
    """OMDb free-tier suele devolver {"Response":"False","Error":"Request limit reached!"}."""
    return data.get("Error") == "Request limit reached!"


def _request_with_rate_limit(params: Mapping[str, object]) -> dict[str, object] | None:
    """
    Wrapper con espera en rate limit:
    - reintenta hasta OMDB_RATE_LIMIT_MAX_RETRIES
    - si rate limit -> sleep OMDB_RATE_LIMIT_WAIT_SECONDS
    """
    global OMDB_RATE_LIMIT_NOTICE_SHOWN

    retries_local = 0
    max_retries = max(0, int(OMDB_RATE_LIMIT_MAX_RETRIES))
    wait_s = max(0.0, float(OMDB_RATE_LIMIT_WAIT_SECONDS))

    while retries_local <= max_retries:
        data = omdb_request(params)
        if data is None:
            return None

        if _is_rate_limit_response(data):
            _m_inc("rate_limit_hits", 1)
            _m_inc("rate_limit_sleeps", 1)

            if not OMDB_RATE_LIMIT_NOTICE_SHOWN:
                _warn_always(
                    "AVISO: límite de llamadas gratuitas de OMDb alcanzado. "
                    f"Esperando {wait_s:g} segundos antes de continuar..."
                )
                OMDB_RATE_LIMIT_NOTICE_SHOWN = True

            if wait_s > 0.0:
                time.sleep(wait_s)
            retries_local += 1
            continue

        return data

    return None


# ============================================================
# CANDIDATE SEARCH (fallback cuando t= falla)
# ============================================================

def _search_candidates_imdb_id(*, title_for_search: str, year: int | None) -> str | None:
    """
    Fallback:
    - endpoint s=
    - rankea por similitud de título y cercanía de año
    - devuelve imdbID más probable
    """
    _m_inc("candidate_search_calls", 1)

    title_q = title_for_search.strip()
    if not title_q or OMDB_DISABLED:
        return None
    if OMDB_API_KEY is None or not str(OMDB_API_KEY).strip():
        return None

    base_url = str(OMDB_BASE_URL).strip() or "https://www.omdbapi.com/"
    params_s: dict[str, str] = {"apikey": str(OMDB_API_KEY), "s": title_q, "type": "movie"}

    timeout_s = _cap_float_runtime(float(OMDB_HTTP_TIMEOUT_SECONDS), min_v=0.5, max_v=120.0)
    resp = _omdb_http_get(base_url=base_url, params=params_s, timeout_seconds=timeout_s)
    if resp is None:
        return None

    try:
        data_s = resp.json() if resp.status_code == 200 else None
    except (ValueError, json.JSONDecodeError):
        data_s = None

    if not isinstance(data_s, dict) or data_s.get("Response") != "True":
        return None

    results_obj = data_s.get("Search") or []
    if not isinstance(results_obj, list):
        return None

    ptit = normalize_title_for_lookup(title_q)

    def score_candidate(cand: Mapping[str, object]) -> float:
        """
        Heurística simple:
        - título exacto normalizado: +2
        - substring: +1
        - año exacto: +2
        - año +/-1: +1
        """
        score = 0.0

        ct_raw = cand.get("Title")
        ct = normalize_title_for_lookup(ct_raw) if isinstance(ct_raw, str) else ""
        if ptit and ct:
            if ptit == ct:
                score += 2.0
            elif ct in ptit or ptit in ct:
                score += 1.0

        cand_year: int | None = None
        cy = cand.get("Year")
        if isinstance(cy, str) and cy != "N/A":
            try:
                cand_year = int(cy[:4])
            except (ValueError, TypeError):
                cand_year = None

        if year is not None and cand_year is not None:
            if year == cand_year:
                score += 2.0
            elif abs(year - cand_year) <= 1:
                score += 1.0

        return score

    best_imdb: str | None = None
    best_score = float("-inf")

    for item in results_obj:
        if not isinstance(item, Mapping):
            continue
        s = score_candidate(item)
        if s > best_score:
            imdb_raw = _safe_imdb_id(item.get("imdbID"))
            if imdb_raw:
                best_score = s
                best_imdb = imdb_raw

    if best_imdb:
        _dbg(f"Candidate search best_imdb={best_imdb} score={best_score:.2f}")

    return best_imdb


# ============================================================
# API PRINCIPAL: query + cache + fallback
# ============================================================

def omdb_query_with_cache(
    *,
    title: str | None,
    year: int | None,
    imdb_id: str | None,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """
    Consulta OMDb con caché persistente + hot-cache.

    Resolución:
    1) Cache por imdb_id (si existe)
    2) Cache por title+year
    3) Red:
       A) imdb_id -> i=
       B) title (+year) -> t= y y=
          - si not_found y year dado -> reintenta sin year
          - si sigue not_found -> s= para candidato -> i=

    Negative caching persistente:
    - Guardamos Response=False Movie not found! en cache con TTL específico.
    """
    imdb_norm = _safe_imdb_id(imdb_id) if imdb_id else None
    year_str = _norm_year_str(year)

    # Normalizamos para cache (clave estable) pero mantenemos el título “humano” para t=
    norm_title = normalize_title_for_lookup(title or "")
    title_query = (title or "").strip()

    if not norm_title and imdb_norm is None:
        return None

    prov_base = _build_default_provenance(imdb_norm=imdb_norm, norm_title=norm_title, year=year)
    prov_in = _merge_provenance(prov_base, provenance)

    cached = _get_cached_item(norm_title=norm_title, norm_year=year_str, imdb_id_hint=imdb_norm)
    if cached is not None:
        _m_inc("cache_hits", 1)
        out = dict(cached["omdb"])
        _attach_provenance(out, prov_in)
        return out

    if OMDB_DISABLED:
        return None

    _m_inc("cache_misses", 1)

    # ------------------------------------------------------------------
    # A) imdb_id -> i=
    # ------------------------------------------------------------------
    if imdb_norm is not None:
        data_main = _request_with_rate_limit({"i": imdb_norm, "type": "movie", "plot": "short"})
        if isinstance(data_main, dict):
            imdb_from_resp = _extract_imdb_id_from_omdb_record(data_main)
            imdb_final = imdb_from_resp or imdb_norm

            if not norm_title:
                t_raw = data_main.get("Title")
                if isinstance(t_raw, str):
                    norm_title = normalize_title_for_lookup(t_raw) or norm_title
            if not year_str:
                y_resp = extract_year_from_omdb(data_main)
                if y_resp is not None:
                    year_str = str(y_resp)

            prov_final = dict(prov_in)
            prov_final["resolved_via"] = "i"
            _attach_provenance(data_main, prov_final)

            _cache_store_item(norm_title=norm_title, norm_year=year_str, imdb_id=imdb_final, omdb_data=dict(data_main))
            return dict(data_main)

        return None

    # ------------------------------------------------------------------
    # B) title (+year) -> t=
    # ------------------------------------------------------------------
    title_for_omdb = title_query or norm_title
    params_t: dict[str, object] = {"t": title_for_omdb, "type": "movie", "plot": "short"}
    if year is not None:
        params_t["y"] = str(year)

    data_t = _request_with_rate_limit(params_t)
    if not isinstance(data_t, dict):
        return None

    used_no_year = False

    if year is not None and _is_movie_not_found(data_t):
        params_no_year = dict(params_t)
        params_no_year.pop("y", None)
        data_no_year = _request_with_rate_limit(params_no_year)
        if isinstance(data_no_year, dict):
            data_t = data_no_year
            used_no_year = True

    if _is_movie_not_found(data_t):
        imdb_best = _search_candidates_imdb_id(title_for_search=title_for_omdb, year=year)
        if imdb_best:
            data_full = _request_with_rate_limit({"i": imdb_best, "type": "movie", "plot": "short"})
            if isinstance(data_full, dict):
                imdb_from_full = _extract_imdb_id_from_omdb_record(data_full)
                imdb_final2 = imdb_from_full or imdb_best

                y_resp2 = extract_year_from_omdb(data_full)
                year_str2 = year_str or (str(y_resp2) if y_resp2 is not None else "")

                prov_final2 = dict(prov_in)
                prov_final2["resolved_via"] = "s_i"
                _attach_provenance(data_full, prov_final2)

                _cache_store_item(norm_title=norm_title, norm_year=year_str2, imdb_id=imdb_final2, omdb_data=dict(data_full))
                return dict(data_full)

    imdb_final3 = _extract_imdb_id_from_omdb_record(data_t)
    y_resp3 = extract_year_from_omdb(data_t)
    year_str3 = year_str or (str(y_resp3) if y_resp3 is not None else "")

    prov_final3 = dict(prov_in)
    prov_final3["resolved_via"] = "t_y" if (year is not None and not used_no_year) else "t"
    _attach_provenance(data_t, prov_final3)

    _cache_store_item(norm_title=norm_title, norm_year=year_str3, imdb_id=imdb_final3, omdb_data=dict(data_t))
    return dict(data_t)


# ============================================================
# FUNCIONES PÚBLICAS (helpers)
# ============================================================

def search_omdb_by_imdb_id(imdb_id: str) -> dict[str, object] | None:
    """Helper: búsqueda por imdbID."""
    imdb_norm = _safe_imdb_id(imdb_id)
    if not imdb_norm:
        return None
    return omdb_query_with_cache(title=None, year=None, imdb_id=imdb_norm)


def search_omdb_by_title_and_year(title: str, year: int | None) -> dict[str, object] | None:
    """Helper: búsqueda por título y (opcional) año."""
    if not title.strip():
        return None
    return omdb_query_with_cache(title=title, year=year, imdb_id=None)


def search_omdb_with_candidates(plex_title: str, plex_year: int | None) -> dict[str, object] | None:
    """
    Compat helper:
    - intenta title+year
    - si no, s= para imdb candidato -> i=
    """
    title_raw = plex_title.strip()
    if not title_raw:
        return None

    data = search_omdb_by_title_and_year(title_raw, plex_year)
    if data and data.get("Response") == "True":
        return data

    imdb_best = _search_candidates_imdb_id(title_for_search=title_raw, year=plex_year)
    if not imdb_best:
        return None

    return search_omdb_by_imdb_id(imdb_best)