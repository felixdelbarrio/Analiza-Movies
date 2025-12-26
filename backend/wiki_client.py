from __future__ import annotations

"""
backend/wiki_client.py (schema v6)

Cliente “best-effort” para enriquecer títulos con Wikipedia/Wikidata con caché persistente,
sin wrappers OO (no hay WikiClient / get_wiki_client).

✅ Objetivos (v6)
----------------
1) API estable y simple:
   - get_wiki(title, year, imdb_id) -> WikiCacheItem | None
   - get_wiki_for_input(movie_input, title, year, imdb_id) -> WikiCacheItem | None
   - flush_wiki_cache() -> None   (para flush_external_caches() en el orquestador)

2) Negative caching REAL (persistente + TTL):
   - Si NO encontramos QID o el QID NO parece film -> guardamos item negativo (status != "ok")
   - Evita repetir SPARQL / Wikipedia calls caros en futuros runs.

3) Rendimiento / robustez:
   - Schema v6 optimizado: records + indexes (sin duplicación imdb:* y ty:* apuntando al mismo dict)
       - records[rid] -> WikiCacheItem (1 record por película)
       - index_imdb[imdb_id] -> rid
       - index_ty["<norm_title>|<norm_year>"] -> rid
   - TTL por estado:
       - ok: TTL largo
       - negativos: TTL más corto
       - imdb->qid negativos: TTL separado y conservador
       - disambiguation: TTL corto (para no repetir búsquedas ambiguas)
   - Write-back batching: flush por umbral de escrituras sucias o por tiempo.
   - Escritura atómica del JSON (temp + replace).
   - Compaction/GC antes de escribir:
       - elimina expirados
       - caps (records / imdb_qid / is_film / entities)
       - rebuild indexes coherentes
       - prune entities a QIDs realmente referenciados

4) Política de schema:
   - NO migración automática.
   - schema mismatch => cache vacío.

5) Logs alineados con backend/logger.py:
   - NO prints.
   - Debug contextual SOLO con logger.debug_ctx("WIKI", ...).
   - Info SOLO si NO silent y aporta valor.
   - Error siempre visible con logger.error(..., always=True).

✅ Métricas (igual que OMDb / otros módulos):
--------------------------------------------
- Contadores thread-safe.
- log_wiki_metrics_summary() con policy:
  - SILENT_MODE=True y DEBUG_MODE=False -> no imprime (salvo force=True).
  - SILENT_MODE=True y DEBUG_MODE=True  -> imprime vía progress si WIKI_METRICS_LOG_ON_SILENT_DEBUG=True.
  - SILENT_MODE=False                  -> imprime vía info.
  - Si todo está a 0 y WIKI_METRICS_LOG_EVEN_IF_ZERO=False -> no imprime (salvo force=True).

Cambios aplicados (alineados con lo propuesto)
----------------------------------------------
- Pooling HTTP: HTTPAdapter con pool_connections/pool_maxsize (cap defensivo) y pool_block=True para backpressure.
- ✅ Concurrencia HTTP real (WIKI_HTTP_MAX_CONCURRENCY):
  - Semáforo global (BoundedSemaphore) compartido por TODAS las llamadas HTTP del módulo.
  - Wrapper _http_get(...) para centralizar acquire/release y mantener “best-effort”.
- ✅ Circuit breaker suave (Wiki/WDQS):
  - Si detectamos racha de fallos (RequestException / 5xx / 429), “abrimos” el breaker un rato.
  - Mientras está abierto, evitamos nuevos requests (fallo rápido) para proteger el run y al proveedor.
  - Se cierra automáticamente tras el cooldown. Se resetea en éxito.
- ✅ Negative caching explícito para disambiguation:
  - Si Wikipedia REST devuelve type=disambiguation, guardamos un item negativo status="disambiguation"
    con TTL corto, evitando repetir búsquedas ambiguas en runs futuros.
  - No “congelamos” fallos de red como negativos.

Notas
-----
- “Best-effort”: puede devolver None si hay fallos transitorios de red o breaker abierto.
  (No cacheamos fallos de red como negativos para no “congelar” errores.)
- ThreadPool safe: locks globales alrededor de IO y mutaciones del cache.
- SPARQL throttle global con lock (evita tormentas de requests concurrentes).
"""

import atexit
import json
import os
import re
import tempfile
import threading
import time
import unicodedata
from collections.abc import Iterable, Mapping
from json import JSONDecodeError
from pathlib import Path
from typing import Final, Literal, Protocol
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

from backend import logger
from backend.config import (
    ANALIZA_WIKI_CACHE_MAX_ENTITIES,
    ANALIZA_WIKI_CACHE_MAX_IMDB_QID,
    ANALIZA_WIKI_CACHE_MAX_IS_FILM,
    ANALIZA_WIKI_CACHE_MAX_RECORDS,
    ANALIZA_WIKI_DEBUG,
    WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES,
    WIKI_CACHE_FLUSH_MAX_SECONDS,
    WIKI_CACHE_PATH,
    WIKI_CACHE_TTL_NEGATIVE_SECONDS,
    WIKI_CACHE_TTL_OK_SECONDS,
    WIKI_FALLBACK_LANGUAGE,
    WIKI_HTTP_MAX_CONCURRENCY,
    WIKI_HTTP_RETRY_BACKOFF_FACTOR,
    WIKI_HTTP_RETRY_TOTAL,
    WIKI_HTTP_TIMEOUT_SECONDS,
    WIKI_HTTP_USER_AGENT,
    WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS,
    WIKI_IS_FILM_TTL_SECONDS,
    WIKI_LANGUAGE,
    WIKI_METRICS_ENABLED,
    WIKI_METRICS_LOG_EVEN_IF_ZERO,
    WIKI_METRICS_LOG_ON_SILENT_DEBUG,
    WIKI_METRICS_TOP_N,
    WIKI_SPARQL_MIN_INTERVAL_SECONDS,
    WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS,
    WIKI_SPARQL_TIMEOUT_READ_SECONDS,
    WIKI_WDQS_URL,
    WIKI_WIKIDATA_API_BASE_URL,
    WIKI_WIKIDATA_ENTITY_BASE_URL,
    WIKI_WIKIPEDIA_API_BASE_URL,
    WIKI_WIKIPEDIA_REST_BASE_URL,
)
from backend.movie_input import normalize_title_for_lookup


# ============================================================================
# Tipos: dict-like (compatibilidad con el pipeline)
# ============================================================================

class WikidataEntity(dict):
    """Entidad etiquetada (qid -> label/description/type)."""


class WikiBlock(dict):
    """Bloque Wikipedia (REST summary) normalizado."""


class WikidataBlock(dict):
    """Bloque Wikidata: qid + listas de QIDs."""


WikiItemStatus = Literal["ok", "no_qid", "not_film", "imdb_no_qid", "disambiguation"]


class WikiCacheItem(dict):
    """
    Entrada principal cacheable (records[rid]).

    Campos esperados:
      - Title: str (normalize_title_for_lookup)
      - Year: str ("" si no hay año)
      - imdbID: str | None (lowercase)
      - wiki: WikiBlock
      - wikidata: WikidataBlock
      - fetched_at: int (epoch seconds)
      - ttl_s: int (seconds)
      - status: ok | no_qid | not_film | imdb_no_qid | disambiguation
    """


class ImdbQidCacheEntry(dict):
    """Cache imdbID -> QID (P345), con negative caching (qid=None)."""


class IsFilmCacheEntry(dict):
    """Cache QID -> is_film (heurística rápida, persistente)."""


class WikiCacheFile(dict):
    """
    Schema v6:
      - schema, language, fallback_language
      - records, index_imdb, index_ty, entities, imdb_qid, is_film
    """


# ============================================================================
# Protocol para idioma por item (evita acoplamiento fuerte)
# ============================================================================

class MovieInputLangProto(Protocol):
    def plex_library_language(self) -> str | None: ...
    def is_spanish_context(self) -> bool: ...
    def is_english_context(self) -> bool: ...
    def is_italian_context(self) -> bool: ...
    def is_french_context(self) -> bool: ...
    def is_japanese_context(self) -> bool: ...
    def is_korean_context(self) -> bool: ...
    def is_chinese_context(self) -> bool: ...


# ============================================================================
# Utils runtime caps (evitan valores absurdos aunque config ya los cape)
# ============================================================================

def _cap_int_runtime(value: int, *, min_v: int, max_v: int) -> int:
    if value < min_v:
        return min_v
    if value > max_v:
        return max_v
    return value


def _cap_float_runtime(value: float, *, min_v: float, max_v: float | None = None) -> float:
    if value < min_v:
        return min_v
    if max_v is not None and value > max_v:
        return max_v
    return value


# ============================================================================
# Constantes / estado interno
# ============================================================================

_SCHEMA_VERSION: Final[int] = 6
_CACHE_PATH: Final[Path] = Path(WIKI_CACHE_PATH)

_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()

_CACHE: WikiCacheFile | None = None
_CACHE_LOCK = threading.RLock()

_CACHE_DIRTY: bool = False
_CACHE_DIRTY_WRITES: int = 0
_CACHE_LAST_FLUSH_TS: float = 0.0

# Throttle SPARQL: usar monotonic para intervalos.
_LAST_SPARQL_MONO: float = 0.0
_SPARQL_THROTTLE_LOCK = threading.Lock()
_SPARQL_MIN_INTERVAL_S: Final[float] = max(0.0, float(WIKI_SPARQL_MIN_INTERVAL_SECONDS))

_TTL_OK_S: Final[int] = int(WIKI_CACHE_TTL_OK_SECONDS)
_TTL_NEGATIVE_S: Final[int] = int(WIKI_CACHE_TTL_NEGATIVE_SECONDS)
_TTL_IMDB_QID_NEGATIVE_S: Final[int] = int(WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS)
_TTL_IS_FILM_S: Final[int] = int(WIKI_IS_FILM_TTL_SECONDS)

# TTL específico para disambiguation: corto y defensivo (no bloquea demasiado).
# - mínimo 5 min
# - máximo 6h o _TTL_NEGATIVE_S (lo más pequeño)
_TTL_DISAMBIGUATION_S: Final[int] = max(300, min(int(_TTL_NEGATIVE_S), 6 * 3600))

_FLUSH_MAX_DIRTY_WRITES: Final[int] = max(1, int(WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES))
_FLUSH_MAX_SECONDS: Final[float] = max(0.1, float(WIKI_CACHE_FLUSH_MAX_SECONDS))

_COMPACT_MAX_RECORDS: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_RECORDS)
_COMPACT_MAX_IMDB_QID: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_IMDB_QID)
_COMPACT_MAX_IS_FILM: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_IS_FILM)
_COMPACT_MAX_ENTITIES: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_ENTITIES)

_WIKI_DEBUG_EXTRA: Final[bool] = bool(ANALIZA_WIKI_DEBUG)

_FILM_INSTANCE_ALLOWLIST: Final[set[str]] = {
    "Q11424",   # film
    "Q24862",   # feature film
    "Q202866",  # animated film
    "Q226730",  # television film
    "Q93204",   # short film
}

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Centralizados en config.py
_HTTP_TIMEOUT: Final[float] = max(0.5, float(WIKI_HTTP_TIMEOUT_SECONDS))
_HTTP_TIMEOUT_SPARQL: Final[tuple[float, float]] = (
    max(0.5, float(WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS)),
    max(1.0, float(WIKI_SPARQL_TIMEOUT_READ_SECONDS)),
)

# ----------------------------------------------------------------------------
# ✅ Concurrencia HTTP real: semáforo global (backpressure a nivel de módulo)
# ----------------------------------------------------------------------------
_HTTP_POOL_MAXSIZE: Final[int] = _cap_int_runtime(int(WIKI_HTTP_MAX_CONCURRENCY), min_v=1, max_v=128)
_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT: Final[float] = _cap_float_runtime(float(_HTTP_TIMEOUT) * 3.0, min_v=0.2, max_v=120.0)
_HTTP_SEM = threading.BoundedSemaphore(_HTTP_POOL_MAXSIZE)

# ----------------------------------------------------------------------------
# ✅ Circuit breaker suave (Wiki / WDQS)
# ----------------------------------------------------------------------------
# Política:
# - “Wiki” agrupa: Wikipedia REST/API + Wikidata API/entity
# - “WDQS” es SPARQL (WIKI_WDQS_URL)
# Umbrales defensivos (sin tocar config.py):
_CB_FAIL_THRESHOLD: Final[int] = 5
_CB_COOLDOWN_S: Final[float] = _cap_float_runtime(float(_HTTP_TIMEOUT) * 10.0, min_v=5.0, max_v=300.0)

_CB_LOCK = threading.Lock()
_CB_STATE: dict[str, dict[str, float | int]] = {
    "wiki": {"failures": 0, "open_until_mono": 0.0},
    "wdqs": {"failures": 0, "open_until_mono": 0.0},
}


# ============================================================================
# MÉTRICAS (ThreadPool safe)
# ============================================================================

_METRICS_LOCK = threading.Lock()
_METRICS: dict[str, int] = {
    "cache_hits": 0,
    "cache_misses": 0,
    "cache_store_writes": 0,
    "cache_flush_writes": 0,
    "cache_compactions": 0,
    "cache_expired_hits": 0,
    "wikipedia_summary_calls": 0,
    "wikipedia_search_calls": 0,
    "wikipedia_failures": 0,
    "wikidata_entity_calls": 0,
    "wikidata_labels_calls": 0,
    "wikidata_failures": 0,
    "sparql_calls": 0,
    "sparql_failures": 0,
    "sparql_throttle_sleeps": 0,
    "items_ok": 0,
    "items_negative": 0,
    "items_negative_imdb_no_qid": 0,
    "items_negative_no_qid": 0,
    "items_negative_not_film": 0,
    "items_negative_disambiguation": 0,
    "path_imdb": 0,
    "path_title_search": 0,
    # backpressure / semáforo
    "http_semaphore_timeouts": 0,
    # circuit breaker
    "cb_open_skips": 0,
    "cb_trips": 0,
    "cb_failures": 0,
    "cb_successes": 0,
}


def _m_inc(key: str, delta: int = 1) -> None:
    with _METRICS_LOCK:
        _METRICS[key] = int(_METRICS.get(key, 0)) + int(delta)


def get_wiki_metrics_snapshot() -> dict[str, int]:
    with _METRICS_LOCK:
        return dict(_METRICS)


def reset_wiki_metrics() -> None:
    with _METRICS_LOCK:
        for k in list(_METRICS.keys()):
            _METRICS[k] = 0


def _metrics_any_nonzero(snapshot: Mapping[str, int]) -> bool:
    try:
        return any(int(v) != 0 for v in snapshot.values())
    except (ValueError, TypeError):
        return True


def _format_metrics_top(snapshot: Mapping[str, int], top_n: int) -> list[tuple[str, int]]:
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


def log_wiki_metrics_summary(*, force: bool = False) -> None:
    """
    Log de métricas estilo “otros módulos”.
    Ver docstring al inicio del fichero para la policy.
    """
    if not WIKI_METRICS_ENABLED and not force:
        return

    snap = get_wiki_metrics_snapshot()

    if (not WIKI_METRICS_LOG_EVEN_IF_ZERO) and (not force):
        if not _metrics_any_nonzero(snap):
            return

    silent = logger.is_silent_mode()
    debug = logger.is_debug_mode()

    if silent and (not debug) and (not force):
        return
    if silent and debug and (not WIKI_METRICS_LOG_ON_SILENT_DEBUG) and (not force):
        return

    try:
        top_n = int(WIKI_METRICS_TOP_N)
    except (ValueError, TypeError):
        top_n = 12
    top_n = max(1, min(100, top_n))

    top = _format_metrics_top(snap, top_n=top_n)

    lines: list[str] = ["[WIKI][METRICS] summary"]
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


# ============================================================================
# Logging helpers (respetan backend/logger.py; sin política duplicada)
# ============================================================================

def _dbg(msg: object) -> None:
    logger.debug_ctx("WIKI", msg)


def _info(msg: str) -> None:
    if logger.is_silent_mode():
        return
    logger.info(msg)


def _err(msg: str) -> None:
    logger.error(msg, always=True)


# ============================================================================
# Circuit breaker helpers
# ============================================================================

def _cb_is_open(name: str) -> bool:
    with _CB_LOCK:
        st = _CB_STATE.get(name)
        if not st:
            return False
        open_until = float(st.get("open_until_mono", 0.0) or 0.0)
        return time.monotonic() < open_until


def _cb_on_success(name: str) -> None:
    with _CB_LOCK:
        st = _CB_STATE.get(name)
        if not st:
            return
        st["failures"] = 0
        st["open_until_mono"] = 0.0
    _m_inc("cb_successes", 1)


def _cb_on_failure(name: str, *, reason: str) -> None:
    trip = False
    open_until = 0.0
    failures = 0
    with _CB_LOCK:
        st = _CB_STATE.get(name)
        if not st:
            return
        failures = int(st.get("failures", 0) or 0) + 1
        st["failures"] = failures
        if failures >= _CB_FAIL_THRESHOLD:
            trip = True
            open_until = time.monotonic() + float(_CB_COOLDOWN_S)
            st["open_until_mono"] = open_until
            st["failures"] = 0  # reset tras trip (cooldown)
    _m_inc("cb_failures", 1)
    if trip:
        _m_inc("cb_trips", 1)
        _dbg(f"CB TRIP [{name}] cooldown={_CB_COOLDOWN_S:.1f}s reason={reason} failures={failures}")
    else:
        _dbg(f"CB FAIL [{name}] reason={reason} failures={failures}/{_CB_FAIL_THRESHOLD}")


# ============================================================================
# Session / HTTP
# ============================================================================

def _get_session() -> requests.Session:
    """
    requests.Session singleton con Retry y pooling.

    Pooling (requests/urllib3):
    - pool_maxsize/pool_connections dimensionado por WIKI_HTTP_MAX_CONCURRENCY
    - pool_block=True introduce backpressure

    Concurrencia real:
    - Además del pool, usamos un semáforo global (ver _http_get) para limitar
      simultaneidad total del módulo.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION

        session = requests.Session()

        ua = str(WIKI_HTTP_USER_AGENT).strip() or "Analiza-Movies/1.0 (local)"
        session.headers.update(
            {
                "User-Agent": ua,
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": f"{WIKI_LANGUAGE},{WIKI_FALLBACK_LANGUAGE};q=0.8,en;q=0.6,es;q=0.5",
            }
        )

        retry_total = _cap_int_runtime(int(WIKI_HTTP_RETRY_TOTAL), min_v=0, max_v=10)
        backoff = _cap_float_runtime(float(WIKI_HTTP_RETRY_BACKOFF_FACTOR), min_v=0.0, max_v=10.0)

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
            pool_connections=_HTTP_POOL_MAXSIZE,
            pool_maxsize=_HTTP_POOL_MAXSIZE,
            pool_block=True,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        _SESSION = session
        return session


def _http_get(
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: float | tuple[float, float] | None = None,
    cb_name: str = "wiki",
) -> requests.Response:
    """
    Wrapper único para GET:
    - Semáforo global (WIKI_HTTP_MAX_CONCURRENCY)
    - Circuit breaker (fallo rápido si está abierto)
    """
    if _cb_is_open(cb_name):
        _m_inc("cb_open_skips", 1)
        raise RequestException(f"WIKI circuit breaker open: {cb_name}")

    acquired = _HTTP_SEM.acquire(timeout=_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT)
    if not acquired:
        _m_inc("http_semaphore_timeouts", 1)
        raise RequestException("WIKI HTTP semaphore acquire timeout")

    try:
        resp = _get_session().get(url, params=params, timeout=timeout)
        # Consideramos 2xx/3xx como éxito; 429 y 5xx como fallo para breaker.
        if resp.status_code == 429 or resp.status_code >= 500:
            _cb_on_failure(cb_name, reason=f"status={resp.status_code}")
        else:
            _cb_on_success(cb_name)
        return resp
    except RequestException as exc:
        _cb_on_failure(cb_name, reason=f"exc={type(exc).__name__}")
        raise
    finally:
        try:
            _HTTP_SEM.release()
        except ValueError:
            _dbg("HTTP semaphore release ValueError (ignored)")


# ============================================================================
# Helpers generales (defensivos)
# ============================================================================

def _now_epoch() -> int:
    return int(time.time())


def _safe_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v or None


def _safe_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None
    return None


def _normalize_lang_code(lang: str) -> str:
    l = lang.strip().lower().replace("_", "-")
    if not l:
        return ""
    iso_map: dict[str, str] = {
        "spa": "es",
        "eng": "en",
        "ita": "it",
        "fra": "fr",
        "fre": "fr",
        "jpn": "ja",
        "jp": "ja",
        "kor": "ko",
        "zho": "zh",
        "chi": "zh",
    }
    base = l.split("-", 1)[0]
    return iso_map.get(base, base)


def _norm_imdb(imdb_id: str | None) -> str | None:
    if not isinstance(imdb_id, str):
        return None
    v = imdb_id.strip().lower()
    return v or None


def _ty_key(norm_title: str, norm_year: str) -> str:
    return f"{norm_title}|{norm_year}"


def _is_expired(fetched_at: int, ttl_s: int, now_epoch: int) -> bool:
    if fetched_at <= 0 or ttl_s <= 0:
        return True
    return (now_epoch - fetched_at) > ttl_s


def _mark_dirty_unlocked() -> None:
    global _CACHE_DIRTY, _CACHE_DIRTY_WRITES
    _CACHE_DIRTY = True
    _CACHE_DIRTY_WRITES += 1


# ============================================================================
# Cache IO (atomic write + load/validate)
# ============================================================================

def _save_cache_file_atomic(cache: WikiCacheFile) -> None:
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


def _maybe_flush_unlocked(force: bool) -> None:
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

    try:
        _compact_cache_unlocked(_CACHE, force=force)
        _save_cache_file_atomic(_CACHE)
        _CACHE_DIRTY = False
        _CACHE_DIRTY_WRITES = 0
        _CACHE_LAST_FLUSH_TS = now
        _m_inc("cache_flush_writes", 1)
    except Exception as exc:
        _dbg(f"cache flush failed: {exc!r}")


def flush_wiki_cache() -> None:
    try:
        with _CACHE_LOCK:
            _maybe_flush_unlocked(force=True)
    except Exception as exc:
        _dbg(f"flush_wiki_cache failed: {exc!r}")


def _flush_cache_on_exit() -> None:
    try:
        with _CACHE_LOCK:
            _maybe_flush_unlocked(force=True)
    except Exception:
        pass
    try:
        log_wiki_metrics_summary()
    except Exception:
        pass


atexit.register(_flush_cache_on_exit)


def _empty_cache() -> WikiCacheFile:
    return {
        "schema": _SCHEMA_VERSION,
        "language": str(WIKI_LANGUAGE),
        "fallback_language": str(WIKI_FALLBACK_LANGUAGE),
        "records": {},
        "index_imdb": {},
        "index_ty": {},
        "entities": {},
        "imdb_qid": {},
        "is_film": {},
    }


def _maybe_quarantine_corrupt_cache() -> None:
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


def _load_cache_unlocked() -> WikiCacheFile:
    global _CACHE, _CACHE_DIRTY, _CACHE_DIRTY_WRITES, _CACHE_LAST_FLUSH_TS
    if _CACHE is not None:
        return _CACHE

    _CACHE_DIRTY = False
    _CACHE_DIRTY_WRITES = 0
    _CACHE_LAST_FLUSH_TS = time.monotonic()

    if not _CACHE_PATH.exists():
        _CACHE = _empty_cache()
        return _CACHE

    try:
        raw_obj = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError) as exc:
        _dbg(f"cache read failed ({exc!r}); recreating")
        _maybe_quarantine_corrupt_cache()
        _CACHE = _empty_cache()
        return _CACHE

    if not isinstance(raw_obj, Mapping) or raw_obj.get("schema") != _SCHEMA_VERSION:
        found_schema = raw_obj.get("schema") if isinstance(raw_obj, Mapping) else None
        _dbg(f"cache schema mismatch -> recreate (found={found_schema!r})")
        _CACHE = _empty_cache()
        return _CACHE

    records_obj = raw_obj.get("records")
    index_imdb_obj = raw_obj.get("index_imdb")
    index_ty_obj = raw_obj.get("index_ty")
    entities_obj = raw_obj.get("entities")
    imdb_qid_obj = raw_obj.get("imdb_qid")
    is_film_obj = raw_obj.get("is_film")

    if (
        not isinstance(records_obj, Mapping)
        or not isinstance(index_imdb_obj, Mapping)
        or not isinstance(index_ty_obj, Mapping)
        or not isinstance(entities_obj, Mapping)
        or not isinstance(imdb_qid_obj, Mapping)
        or not isinstance(is_film_obj, Mapping)
    ):
        _CACHE = _empty_cache()
        return _CACHE

    records: dict[str, WikiCacheItem] = {}
    for rid, v in records_obj.items():
        if not isinstance(rid, str) or not isinstance(v, Mapping):
            continue

        title = v.get("Title")
        year = v.get("Year")
        imdb_id = v.get("imdbID")
        wiki = v.get("wiki")
        wikidata = v.get("wikidata")
        fetched_at = v.get("fetched_at")
        ttl_s = v.get("ttl_s")
        status = v.get("status")

        if not isinstance(title, str) or not isinstance(year, str):
            continue
        if imdb_id is not None and not isinstance(imdb_id, str):
            continue
        if not isinstance(wiki, Mapping) or not isinstance(wikidata, Mapping):
            continue
        if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
            continue
        if status not in ("ok", "no_qid", "not_film", "imdb_no_qid", "disambiguation"):
            continue

        records[rid] = WikiCacheItem(
            Title=title,
            Year=year,
            imdbID=_norm_imdb(imdb_id),
            wiki=dict(wiki),
            wikidata=dict(wikidata),
            fetched_at=fetched_at,
            ttl_s=ttl_s,
            status=status,
        )

    index_imdb: dict[str, str] = {}
    for k, v in index_imdb_obj.items():
        imdb = _norm_imdb(k if isinstance(k, str) else None)
        if not imdb or not isinstance(v, str):
            continue
        index_imdb[imdb] = v

    index_ty: dict[str, str] = {}
    for k, v in index_ty_obj.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip():
            index_ty[k] = v

    entities: dict[str, WikidataEntity] = {}
    for k, v in entities_obj.items():
        qid = _safe_str(k)
        if not qid or not isinstance(v, Mapping):
            continue
        entities[qid] = WikidataEntity(dict(v))

    imdb_qid: dict[str, ImdbQidCacheEntry] = {}
    for k, v in imdb_qid_obj.items():
        imdb = _norm_imdb(k if isinstance(k, str) else None)
        if not imdb or not isinstance(v, Mapping):
            continue
        fetched_at = v.get("fetched_at")
        ttl_s = v.get("ttl_s")
        if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
            continue
        qid_val = v.get("qid")
        qid = _safe_str(qid_val) if qid_val is not None else None
        imdb_qid[imdb] = ImdbQidCacheEntry(qid=qid, fetched_at=fetched_at, ttl_s=ttl_s)

    is_film: dict[str, IsFilmCacheEntry] = {}
    for k, v in is_film_obj.items():
        qid = _safe_str(k)
        if not qid or not isinstance(v, Mapping):
            continue
        fetched_at = v.get("fetched_at")
        ttl_s = v.get("ttl_s")
        is_film_val = v.get("is_film")
        if not isinstance(fetched_at, int) or not isinstance(ttl_s, int) or not isinstance(is_film_val, bool):
            continue
        is_film[qid] = IsFilmCacheEntry(is_film=is_film_val, fetched_at=fetched_at, ttl_s=ttl_s)

    index_imdb = {k: rid for k, rid in index_imdb.items() if rid in records}
    index_ty = {k: rid for k, rid in index_ty.items() if rid in records}

    _CACHE = WikiCacheFile(
        schema=_SCHEMA_VERSION,
        language=str(raw_obj.get("language") or WIKI_LANGUAGE),
        fallback_language=str(raw_obj.get("fallback_language") or WIKI_FALLBACK_LANGUAGE),
        records=records,
        index_imdb=index_imdb,
        index_ty=index_ty,
        entities=entities,
        imdb_qid=imdb_qid,
        is_film=is_film,
    )

    try:
        _compact_cache_unlocked(_CACHE, force=False)
    except Exception:
        pass

    return _CACHE


# ============================================================================
# Compaction / GC (schema v6)
# ============================================================================

def _compact_cache_unlocked(cache: WikiCacheFile, *, force: bool) -> None:
    try:
        now_epoch = _now_epoch()

        records_obj = cache.get("records")
        if not isinstance(records_obj, Mapping):
            cache["records"] = {}
            records_obj = cache["records"]

        records: dict[str, WikiCacheItem] = {}
        for rid, it in records_obj.items():
            if not isinstance(rid, str) or not isinstance(it, Mapping):
                continue
            fetched_at = it.get("fetched_at")
            ttl_s = it.get("ttl_s")
            if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
                continue
            if _is_expired(fetched_at, ttl_s, now_epoch):
                continue

            d = dict(it)
            imdb_norm = _norm_imdb(d.get("imdbID") if isinstance(d.get("imdbID"), str) else None)
            d["imdbID"] = imdb_norm
            records[rid] = WikiCacheItem(d)

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

            imdb = _norm_imdb(it.get("imdbID") if isinstance(it.get("imdbID"), str) else None)
            if imdb:
                index_imdb[imdb] = rid

        cache["index_imdb"] = index_imdb
        cache["index_ty"] = index_ty

        imdb_qid_obj = cache.get("imdb_qid")
        imdb_qid: dict[str, ImdbQidCacheEntry] = {}
        if isinstance(imdb_qid_obj, Mapping):
            for k, v in imdb_qid_obj.items():
                imdb = _norm_imdb(k if isinstance(k, str) else None)
                if not imdb or not isinstance(v, Mapping):
                    continue
                fetched_at = v.get("fetched_at")
                ttl_s = v.get("ttl_s")
                if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
                    continue
                if _is_expired(fetched_at, ttl_s, now_epoch):
                    continue
                imdb_qid[imdb] = ImdbQidCacheEntry(dict(v))

        if _COMPACT_MAX_IMDB_QID > 0 and len(imdb_qid) > _COMPACT_MAX_IMDB_QID:
            ranked = sorted(imdb_qid.items(), key=lambda kv: int(kv[1].get("fetched_at", 0)), reverse=True)
            imdb_qid = dict(ranked[:_COMPACT_MAX_IMDB_QID])
        cache["imdb_qid"] = imdb_qid

        is_film_obj = cache.get("is_film")
        is_film: dict[str, IsFilmCacheEntry] = {}
        if isinstance(is_film_obj, Mapping):
            for k, v in is_film_obj.items():
                qid = _safe_str(k)
                if not qid or not isinstance(v, Mapping):
                    continue
                fetched_at = v.get("fetched_at")
                ttl_s = v.get("ttl_s")
                if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
                    continue
                if _is_expired(fetched_at, ttl_s, now_epoch):
                    continue
                is_film[qid] = IsFilmCacheEntry(dict(v))

        if _COMPACT_MAX_IS_FILM > 0 and len(is_film) > _COMPACT_MAX_IS_FILM:
            ranked = sorted(is_film.items(), key=lambda kv: int(kv[1].get("fetched_at", 0)), reverse=True)
            is_film = dict(ranked[:_COMPACT_MAX_IS_FILM])
        cache["is_film"] = is_film

        referenced_qids: set[str] = set()
        for it in records.values():
            wikidata_block = it.get("wikidata")
            if isinstance(wikidata_block, Mapping):
                qid = wikidata_block.get("qid")
                if isinstance(qid, str) and qid.strip():
                    referenced_qids.add(qid.strip())
                for prop in ("directors", "countries", "genres"):
                    qids = wikidata_block.get(prop)
                    if isinstance(qids, list):
                        for q in qids:
                            if isinstance(q, str) and q.strip():
                                referenced_qids.add(q.strip())

            wiki_block = it.get("wiki")
            if isinstance(wiki_block, Mapping):
                wb = wiki_block.get("wikibase_item")
                if isinstance(wb, str) and wb.strip():
                    referenced_qids.add(wb.strip())

        entities_obj = cache.get("entities")
        entities: dict[str, WikidataEntity] = {}
        if isinstance(entities_obj, Mapping):
            for qid, ent in entities_obj.items():
                qs = _safe_str(qid)
                if not qs or qs not in referenced_qids:
                    continue
                if isinstance(ent, Mapping):
                    entities[qs] = WikidataEntity(dict(ent))

        if _COMPACT_MAX_ENTITIES > 0 and len(entities) > _COMPACT_MAX_ENTITIES:
            keep_keys = sorted(entities.keys())[:_COMPACT_MAX_ENTITIES]
            entities = {k: entities[k] for k in keep_keys}
        cache["entities"] = entities

        _m_inc("cache_compactions", 1)

        if _WIKI_DEBUG_EXTRA:
            _dbg(
                "cache compacted"
                + (" (force)" if force else "")
                + f" | records={len(records)} idx_imdb={len(index_imdb)} idx_ty={len(index_ty)} "
                  f"entities={len(entities)} imdb_qid={len(imdb_qid)} is_film={len(is_film)}"
            )
        else:
            _dbg(f"cache compacted{' (force)' if force else ''} | records={len(records)} entities={len(entities)}")

    except Exception as exc:
        _dbg(f"cache compaction failed: {exc!r}")


# ============================================================================
# Idiomas por item
# ============================================================================

def _detect_language_chain_from_input(movie_input: MovieInputLangProto | None) -> list[str]:
    chain: list[str] = []

    if movie_input is not None:
        try:
            lang = movie_input.plex_library_language()
        except Exception:
            lang = None

        if lang:
            base = _normalize_lang_code(lang)
            if base:
                chain.append(base)

        if not chain:
            try:
                if movie_input.is_spanish_context():
                    chain.append("es")
                elif movie_input.is_italian_context():
                    chain.append("it")
                elif movie_input.is_french_context():
                    chain.append("fr")
                elif movie_input.is_japanese_context():
                    chain.append("ja")
                elif movie_input.is_korean_context():
                    chain.append("ko")
                elif movie_input.is_chinese_context():
                    chain.append("zh")
                elif movie_input.is_english_context():
                    chain.append("en")
            except Exception:
                pass

    for cfg_lang in (_normalize_lang_code(WIKI_LANGUAGE), _normalize_lang_code(WIKI_FALLBACK_LANGUAGE)):
        if cfg_lang and cfg_lang not in chain:
            chain.append(cfg_lang)

    if "en" not in chain:
        chain.append("en")

    out: list[str] = []
    seen: set[str] = set()
    for l in chain:
        if l and l not in seen:
            seen.add(l)
            out.append(l)
    return out


def _best_wikipedia_languages_for_item(movie_input: MovieInputLangProto | None) -> tuple[str, str]:
    chain = _detect_language_chain_from_input(movie_input)
    primary = chain[0] if chain else (_normalize_lang_code(WIKI_LANGUAGE) or "en")

    fallback = ""
    for l in chain[1:]:
        if l != primary:
            fallback = l
            break

    if not fallback:
        fallback = _normalize_lang_code(WIKI_FALLBACK_LANGUAGE) or ("en" if primary != "en" else "")

    if fallback == primary:
        fallback = "en" if primary != "en" else ""

    return primary, fallback


# ============================================================================
# Canonicalización para ranking de búsquedas
# ============================================================================

def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _canon_cmp(text: str) -> str:
    base = _strip_accents(text).lower()
    tokens = _WORD_RE.findall(base)
    return " ".join(tokens)


# ============================================================================
# Wikipedia REST + Search
# ============================================================================

class _WikiDisambiguationError(Exception):
    def __init__(self, *, title: str, language: str) -> None:
        super().__init__(f"Wikipedia disambiguation: {language}:{title}")
        self.title = title
        self.language = language


def _fetch_wikipedia_summary_by_title(title: str, language: str) -> Mapping[str, object] | None:
    _m_inc("wikipedia_summary_calls", 1)

    safe_title = quote(title.replace(" ", "_"), safe="")
    base = str(WIKI_WIKIPEDIA_REST_BASE_URL).rstrip("/")
    url = f"{base.format(lang=language)}/page/summary/{safe_title}"

    _dbg(f"wikipedia.summary -> lang={language} title={title!r}")

    try:
        resp = _http_get(url, timeout=_HTTP_TIMEOUT, cb_name="wiki")
    except RequestException as exc:
        _m_inc("wikipedia_failures", 1)
        _dbg(f"wikipedia.summary EXC: {exc!r}")
        return None

    if resp.status_code != 200:
        _m_inc("wikipedia_failures", 1)
        _dbg(f"wikipedia.summary <- status={resp.status_code}")
        return None

    try:
        data = resp.json()
    except (ValueError, JSONDecodeError) as exc:
        _m_inc("wikipedia_failures", 1)
        _dbg(f"wikipedia.summary JSON EXC: {exc!r}")
        return None

    if not isinstance(data, Mapping):
        _m_inc("wikipedia_failures", 1)
        return None

    if _safe_str(data.get("type")) == "disambiguation":
        _dbg("wikipedia.summary -> disambiguation")
        raise _WikiDisambiguationError(title=title, language=language)

    return data


def _wikipedia_search(*, query: str, language: str, limit: int = 8) -> list[dict[str, object]]:
    _m_inc("wikipedia_search_calls", 1)

    params: dict[str, str] = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json",
        "utf8": "1",
    }
    url = str(WIKI_WIKIPEDIA_API_BASE_URL).format(lang=language)

    _dbg(f"wikipedia.search -> lang={language} q={query!r}")

    try:
        resp = _http_get(url, params=params, timeout=_HTTP_TIMEOUT, cb_name="wiki")
    except RequestException as exc:
        _m_inc("wikipedia_failures", 1)
        _dbg(f"wikipedia.search EXC: {exc!r}")
        return []

    if resp.status_code != 200:
        _m_inc("wikipedia_failures", 1)
        _dbg(f"wikipedia.search <- status={resp.status_code}")
        return []

    try:
        payload = resp.json()
    except (ValueError, JSONDecodeError) as exc:
        _m_inc("wikipedia_failures", 1)
        _dbg(f"wikipedia.search JSON EXC: {exc!r}")
        return []

    if not isinstance(payload, Mapping):
        return []

    q = payload.get("query")
    if not isinstance(q, Mapping):
        return []

    search_obj = q.get("search")
    if not isinstance(search_obj, list):
        return []

    out: list[dict[str, object]] = []
    for it in search_obj:
        if isinstance(it, Mapping):
            out.append(dict(it))
    return out


def _score_search_hit(*, hit_title: str, hit_snippet: str, wanted_title: str, year: int | None) -> float:
    score = 0.0
    want = _canon_cmp(wanted_title)
    got = _canon_cmp(hit_title)

    if want == got:
        score += 10.0
    elif want and got and (want in got or got in want):
        score += 6.0
    else:
        wt = set(want.split())
        gt = set(got.split())
        if wt and gt:
            score += min(5.0, len(wt & gt) * 0.8)

    sn = _canon_cmp(hit_snippet)
    if "pelicula" in sn or "película" in sn or "film" in sn or "movie" in sn or "largometraje" in sn:
        score += 2.0

    if year is not None:
        y = str(year)
        if y in hit_title:
            score += 2.0
        if y in hit_snippet:
            score += 1.0

    if "desambiguacion" in sn or "(desambiguacion" in _canon_cmp(hit_title):
        score -= 4.0

    return score


def _rank_wikipedia_candidates(*, lookup_title: str, year: int | None, language: str) -> list[str]:
    clean_title = " ".join(lookup_title.strip().split())

    queries: list[str] = []
    if year is not None:
        queries.append(f"{clean_title} {year} film")
        queries.append(f"{clean_title} {year} movie")
        if language == "es":
            queries.append(f"{clean_title} {year} película")

    queries.append(f"{clean_title} film")
    queries.append(f"{clean_title} movie")
    if language == "es":
        queries.append(f"{clean_title} película")
    queries.append(clean_title)

    scored: dict[str, float] = {}
    for q in queries:
        for hit in _wikipedia_search(query=q, language=language, limit=10):
            ht = _safe_str(hit.get("title"))
            if not ht:
                continue
            snippet_raw = hit.get("snippet")
            hs = str(snippet_raw) if snippet_raw is not None else ""
            s = _score_search_hit(hit_title=ht, hit_snippet=hs, wanted_title=clean_title, year=year)
            prev = scored.get(ht)
            if prev is None or s > prev:
                scored[ht] = s

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return [t for (t, s) in ranked if s >= 4.0]


def _choose_wikipedia_summary_candidates(
    *,
    title_for_lookup: str,
    year: int | None,
    primary_language: str,
    fallback_language: str,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for t in _rank_wikipedia_candidates(lookup_title=title_for_lookup, year=year, language=primary_language)[:8]:
        out.append((t, primary_language))

    if fallback_language and fallback_language != primary_language:
        for t in _rank_wikipedia_candidates(lookup_title=title_for_lookup, year=year, language=fallback_language)[:8]:
            out.append((t, fallback_language))

    return out


# ============================================================================
# Wikidata entity + labels
# ============================================================================

def _fetch_wikidata_entity_json(qid: str) -> Mapping[str, object] | None:
    _m_inc("wikidata_entity_calls", 1)

    base = str(WIKI_WIKIDATA_ENTITY_BASE_URL).rstrip("/")
    url = f"{base}/{qid}.json"
    _dbg(f"wikidata.entity -> qid={qid}")

    try:
        resp = _http_get(url, timeout=_HTTP_TIMEOUT, cb_name="wiki")
    except RequestException as exc:
        _m_inc("wikidata_failures", 1)
        _dbg(f"wikidata.entity EXC: {exc!r}")
        return None

    if resp.status_code != 200:
        _m_inc("wikidata_failures", 1)
        _dbg(f"wikidata.entity <- status={resp.status_code}")
        return None

    try:
        data = resp.json()
    except (ValueError, JSONDecodeError) as exc:
        _m_inc("wikidata_failures", 1)
        _dbg(f"wikidata.entity JSON EXC: {exc!r}")
        return None

    if not isinstance(data, Mapping):
        _m_inc("wikidata_failures", 1)
        return None

    entities = data.get("entities")
    if not isinstance(entities, Mapping):
        _m_inc("wikidata_failures", 1)
        return None

    entity = entities.get(qid)
    if not isinstance(entity, Mapping):
        _m_inc("wikidata_failures", 1)
        return None

    return entity


def _extract_qids_from_claims(entity: Mapping[str, object], property_id: str) -> list[str]:
    claims_obj = entity.get("claims")
    if not isinstance(claims_obj, Mapping):
        return []

    prop_claims = claims_obj.get(property_id)
    if not isinstance(prop_claims, list):
        return []

    qids: list[str] = []
    for claim in prop_claims:
        if not isinstance(claim, Mapping):
            continue
        mainsnak = claim.get("mainsnak")
        if not isinstance(mainsnak, Mapping):
            continue
        datavalue = mainsnak.get("datavalue")
        if not isinstance(datavalue, Mapping):
            continue
        value = datavalue.get("value")
        if not isinstance(value, Mapping):
            continue
        qid = _safe_str(value.get("id"))
        if qid and qid not in qids:
            qids.append(qid)
    return qids


def _chunked(values: list[str], size: int) -> Iterable[list[str]]:
    step = max(1, size)
    for i in range(0, len(values), step):
        yield values[i : i + step]


def _fetch_wikidata_labels(qids: list[str], language: str, fallback_language: str) -> dict[str, WikidataEntity]:
    if not qids:
        return {}

    _m_inc("wikidata_labels_calls", 1)

    out: dict[str, WikidataEntity] = {}
    languages = f"{language}|{fallback_language}" if fallback_language and fallback_language != language else language

    for batch in _chunked(qids, 50):
        ids = "|".join(batch)
        params: dict[str, str] = {
            "action": "wbgetentities",
            "ids": ids,
            "props": "labels|descriptions",
            "languages": languages,
            "format": "json",
        }

        try:
            resp = _http_get(str(WIKI_WIKIDATA_API_BASE_URL), params=params, timeout=_HTTP_TIMEOUT, cb_name="wiki")
        except RequestException as exc:
            _m_inc("wikidata_failures", 1)
            _dbg(f"wikidata.labels EXC: {exc!r}")
            continue

        if resp.status_code != 200:
            _m_inc("wikidata_failures", 1)
            _dbg(f"wikidata.labels <- status={resp.status_code}")
            continue

        try:
            payload = resp.json()
        except (ValueError, JSONDecodeError) as exc:
            _m_inc("wikidata_failures", 1)
            _dbg(f"wikidata.labels JSON EXC: {exc!r}")
            continue

        if not isinstance(payload, Mapping):
            continue

        entities_obj = payload.get("entities")
        if not isinstance(entities_obj, Mapping):
            continue

        for qid in batch:
            ent = entities_obj.get(qid)
            if not isinstance(ent, Mapping):
                continue

            label = qid
            labels_obj = ent.get("labels")
            if isinstance(labels_obj, Mapping):
                lp = labels_obj.get(language)
                lf = labels_obj.get(fallback_language) if fallback_language else None
                if isinstance(lp, Mapping):
                    label = str(lp.get("value") or qid)
                elif isinstance(lf, Mapping):
                    label = str(lf.get("value") or qid)

            desc: str | None = None
            descriptions_obj = ent.get("descriptions")
            if isinstance(descriptions_obj, Mapping):
                dp = descriptions_obj.get(language)
                df = descriptions_obj.get(fallback_language) if fallback_language else None
                if isinstance(dp, Mapping):
                    desc = _safe_str(dp.get("value"))
                elif isinstance(df, Mapping):
                    desc = _safe_str(df.get("value"))

            out[qid] = WikidataEntity(label=label, description=desc)

    return out


# ============================================================================
# SPARQL (con throttle global)
# ============================================================================

def _sparql_throttle() -> None:
    global _LAST_SPARQL_MONO
    if _SPARQL_MIN_INTERVAL_S <= 0.0:
        return

    with _SPARQL_THROTTLE_LOCK:
        now = time.monotonic()
        delta = now - _LAST_SPARQL_MONO
        if delta < _SPARQL_MIN_INTERVAL_S:
            sleep_s = _SPARQL_MIN_INTERVAL_S - delta
            _m_inc("sparql_throttle_sleeps", 1)
            time.sleep(sleep_s)
        _LAST_SPARQL_MONO = time.monotonic()


def _wikidata_sparql(query: str) -> Mapping[str, object] | None:
    _m_inc("sparql_calls", 1)

    params = {"format": "json", "query": query}

    _sparql_throttle()
    _dbg(f"wikidata.sparql -> len={len(query)}")

    try:
        resp = _http_get(str(WIKI_WDQS_URL), params=params, timeout=_HTTP_TIMEOUT_SPARQL, cb_name="wdqs")
    except RequestException as exc:
        _m_inc("sparql_failures", 1)
        _dbg(f"wikidata.sparql EXC: {exc!r}")
        return None

    if resp.status_code != 200:
        _m_inc("sparql_failures", 1)
        _dbg(f"wikidata.sparql <- status={resp.status_code}")
        return None

    try:
        data = resp.json()
    except (ValueError, JSONDecodeError) as exc:
        _m_inc("sparql_failures", 1)
        _dbg(f"wikidata.sparql JSON EXC: {exc!r}")
        return None

    return data if isinstance(data, Mapping) else None


# ============================================================================
# “is film” heurística (cacheada)
# ============================================================================

def _looks_like_film_from_wikipedia(wiki_raw: Mapping[str, object], language: str) -> bool:
    desc = str(wiki_raw.get("description") or "").strip().lower()
    if not desc:
        return False

    hints = (
        "película",
        "largometraje",
        "film",
        "movie",
        "feature film",
        "animated film",
        "television film",
        "motion picture",
        "映画",
        "영화",
        "电影",
        "電影",
    )
    if any(h.lower() in desc for h in hints):
        return True

    if language == "ja" and "映画" in desc:
        return True
    if language == "ko" and "영화" in desc:
        return True
    if language == "zh" and ("电影" in desc or "電影" in desc):
        return True

    return False


def _is_film_without_sparql(
    *,
    wd_entity: Mapping[str, object],
    wiki_raw: Mapping[str, object] | None,
    wiki_lang: str,
) -> bool:
    p31 = set(_extract_qids_from_claims(wd_entity, "P31"))
    if p31 & _FILM_INSTANCE_ALLOWLIST:
        return True

    if wiki_raw is not None and _looks_like_film_from_wikipedia(wiki_raw, wiki_lang):
        return True

    return False


def _is_film_cached(
    *,
    cache: WikiCacheFile,
    qid: str,
    wd_entity: Mapping[str, object],
    wiki_raw: Mapping[str, object] | None,
    wiki_lang: str,
) -> bool:
    now_epoch = _now_epoch()

    cached = cache["is_film"].get(qid)
    if cached is not None and not _is_expired(int(cached["fetched_at"]), int(cached["ttl_s"]), now_epoch):
        return bool(cached.get("is_film") is True)

    ok = _is_film_without_sparql(wd_entity=wd_entity, wiki_raw=wiki_raw, wiki_lang=wiki_lang)
    cache["is_film"][qid] = IsFilmCacheEntry(is_film=bool(ok), fetched_at=now_epoch, ttl_s=int(_TTL_IS_FILM_S))
    _mark_dirty_unlocked()
    _maybe_flush_unlocked(force=False)
    return ok


# ============================================================================
# imdbID -> QID (P345) + negative caching
# ============================================================================

def _fetch_qid_by_imdb_id(cache: WikiCacheFile, imdb_id: str) -> str | None:
    imdb_norm = _norm_imdb(imdb_id)
    if not imdb_norm:
        return None

    now_epoch = _now_epoch()
    cached = cache["imdb_qid"].get(imdb_norm)
    if cached is not None and not _is_expired(int(cached["fetched_at"]), int(cached["ttl_s"]), now_epoch):
        return cached.get("qid")

    query = f"""
SELECT ?item WHERE {{
  ?item wdt:P345 "{imdb_norm}" .
}}
LIMIT 2
""".strip()

    data = _wikidata_sparql(query)
    if not data:
        return None

    results = data.get("results")
    if not isinstance(results, Mapping):
        return None
    bindings = results.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        cache["imdb_qid"][imdb_norm] = ImdbQidCacheEntry(
            qid=None,
            fetched_at=now_epoch,
            ttl_s=int(_TTL_IMDB_QID_NEGATIVE_S),
        )
        _mark_dirty_unlocked()
        _maybe_flush_unlocked(force=False)
        return None

    first = bindings[0]
    if not isinstance(first, Mapping):
        return None
    item = first.get("item")
    if not isinstance(item, Mapping):
        return None
    val = _safe_str(item.get("value"))
    if not val:
        return None

    m = re.search(r"/entity/(Q\d+)$", val)
    qid = m.group(1) if m else None
    if not qid:
        return None

    cache["imdb_qid"][imdb_norm] = ImdbQidCacheEntry(qid=qid, fetched_at=now_epoch, ttl_s=int(_TTL_OK_S))
    _mark_dirty_unlocked()
    _maybe_flush_unlocked(force=False)
    return qid


def _extract_sitelink_title(entity: Mapping[str, object], language: str) -> str | None:
    sitelinks = entity.get("sitelinks")
    if not isinstance(sitelinks, Mapping):
        return None
    sl = sitelinks.get(f"{language}wiki")
    if not isinstance(sl, Mapping):
        return None
    return _safe_str(sl.get("title"))


def _pick_best_sitelink_title(entity: Mapping[str, object], languages: list[str]) -> tuple[str | None, str]:
    for lang in languages:
        title = _extract_sitelink_title(entity, lang)
        if title:
            return title, lang
    return None, (languages[0] if languages else (_normalize_lang_code(WIKI_LANGUAGE) or "en"))


# ============================================================================
# Cache lookup/store (schema v6)
# ============================================================================

def _get_cached_item(
    *,
    cache: WikiCacheFile,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
) -> WikiCacheItem | None:
    now_epoch = _now_epoch()

    records = cache["records"]
    idx_imdb = cache["index_imdb"]
    idx_ty = cache["index_ty"]

    if imdb_id:
        rid = idx_imdb.get(imdb_id)
        if isinstance(rid, str):
            it = records.get(rid)
            if it is not None:
                if _is_expired(int(it["fetched_at"]), int(it["ttl_s"]), now_epoch):
                    _m_inc("cache_expired_hits", 1)
                    return None
                return it

    rid2 = idx_ty.get(_ty_key(norm_title, norm_year))
    if isinstance(rid2, str):
        it2 = records.get(rid2)
        if it2 is not None:
            if _is_expired(int(it2["fetched_at"]), int(it2["ttl_s"]), now_epoch):
                _m_inc("cache_expired_hits", 1)
                return None
            return it2

    return None


def _rid_for_item(item: WikiCacheItem) -> str:
    imdb = _norm_imdb(item.get("imdbID") if isinstance(item.get("imdbID"), str) else None)
    if imdb:
        return f"imdb:{imdb}"

    title = item.get("Title")
    year = item.get("Year")
    t = title if isinstance(title, str) else ""
    y = year if isinstance(year, str) else ""
    return f"ty:{t}|{y}"


def _store_item_unlocked(cache: WikiCacheFile, item: WikiCacheItem) -> None:
    rid = _rid_for_item(item)
    cache["records"][rid] = item

    title = item.get("Title")
    year = item.get("Year")
    if isinstance(title, str) and isinstance(year, str):
        cache["index_ty"][_ty_key(title, year)] = rid

    imdb = _norm_imdb(item.get("imdbID") if isinstance(item.get("imdbID"), str) else None)
    if imdb:
        cache["index_imdb"][imdb] = rid

    _m_inc("cache_store_writes", 1)
    _mark_dirty_unlocked()
    _maybe_flush_unlocked(force=False)


# ============================================================================
# Builders
# ============================================================================

def _build_negative_item(
    *,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
    status: WikiItemStatus,
    wikibase_item: str | None,
    primary_language: str,
    fallback_language: str,
) -> WikiCacheItem:
    now_epoch = _now_epoch()

    wiki_block: WikiBlock = WikiBlock(
        language=primary_language,
        fallback_language=fallback_language,
        source_language="",
        wikipedia_title=None,
        wikipedia_pageid=None,
        wikibase_item=wikibase_item,
        summary="",
        description="",
        urls={},
    )

    wikidata_block: WikidataBlock = WikidataBlock()
    if wikibase_item:
        wikidata_block["qid"] = wikibase_item

    _m_inc("items_negative", 1)
    if status == "imdb_no_qid":
        _m_inc("items_negative_imdb_no_qid", 1)
        ttl = int(_TTL_NEGATIVE_S)
    elif status == "no_qid":
        _m_inc("items_negative_no_qid", 1)
        ttl = int(_TTL_NEGATIVE_S)
    elif status == "not_film":
        _m_inc("items_negative_not_film", 1)
        ttl = int(_TTL_NEGATIVE_S)
    elif status == "disambiguation":
        _m_inc("items_negative_disambiguation", 1)
        ttl = int(_TTL_DISAMBIGUATION_S)
    else:
        ttl = int(_TTL_NEGATIVE_S)

    return WikiCacheItem(
        Title=norm_title,
        Year=norm_year,
        imdbID=imdb_id,
        wiki=wiki_block,
        wikidata=wikidata_block,
        fetched_at=now_epoch,
        ttl_s=ttl,
        status=status,
    )


def _build_ok_item_and_merge_entities(
    *,
    cache: WikiCacheFile,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
    wiki_raw: Mapping[str, object],
    source_language: str,
    wikibase_item: str,
    wd_entity: Mapping[str, object],
    primary_language: str,
    fallback_language: str,
) -> WikiCacheItem:
    titles_obj = wiki_raw.get("titles")
    wikipedia_title: str | None = None
    if isinstance(titles_obj, Mapping):
        wikipedia_title = _safe_str(titles_obj.get("normalized")) or _safe_str(titles_obj.get("canonical"))

    wiki_block: WikiBlock = WikiBlock(
        language=primary_language,
        fallback_language=fallback_language,
        source_language=source_language,
        wikipedia_title=wikipedia_title,
        wikipedia_pageid=_safe_int(wiki_raw.get("pageid")),
        wikibase_item=wikibase_item,
        summary=str(wiki_raw.get("extract") or ""),
        description=str(wiki_raw.get("description") or ""),
        urls=dict(wiki_raw.get("content_urls")) if isinstance(wiki_raw.get("content_urls"), Mapping) else {},
    )

    if "originalimage" in wiki_raw or "thumbnail" in wiki_raw:
        images: dict[str, object] = {}
        original = wiki_raw.get("originalimage")
        thumb = wiki_raw.get("thumbnail")
        if isinstance(original, Mapping):
            images["original"] = dict(original)
        if isinstance(thumb, Mapping):
            images["thumbnail"] = dict(thumb)
        wiki_block["images"] = images

    wikidata_block: WikidataBlock = WikidataBlock(qid=wikibase_item)

    directors = _extract_qids_from_claims(wd_entity, "P57")
    countries = _extract_qids_from_claims(wd_entity, "P495")
    genres = _extract_qids_from_claims(wd_entity, "P136")

    if directors:
        wikidata_block["directors"] = directors
    if countries:
        wikidata_block["countries"] = countries
    if genres:
        wikidata_block["genres"] = genres

    to_label = list({*directors, *countries, *genres})
    labeled = _fetch_wikidata_labels(to_label, primary_language, fallback_language)

    for qid, ent in labeled.items():
        etype: str | None = None
        if qid in directors:
            etype = "person"
        elif qid in countries:
            etype = "country"
        elif qid in genres:
            etype = "genre"
        merged = WikidataEntity(ent)
        if etype:
            merged["type"] = etype
        cache["entities"][qid] = merged

    _m_inc("items_ok", 1)

    item = WikiCacheItem(
        Title=norm_title,
        Year=norm_year,
        imdbID=imdb_id,
        wiki=wiki_block,
        wikidata=wikidata_block,
        fetched_at=_now_epoch(),
        ttl_s=int(_TTL_OK_S),
        status="ok",
    )

    if not logger.is_silent_mode():
        year_label = norm_year if norm_year else "?"
        _info(f"[WIKI] cached ({source_language}): {norm_title} ({year_label})")

    return item


# ============================================================================
# Core único
# ============================================================================

def _get_wiki_impl(
    *,
    title: str,
    year: int | None,
    imdb_id: str | None,
    movie_input: MovieInputLangProto | None,
    primary: str,
    fallback: str,
) -> WikiCacheItem | None:
    lookup_title = normalize_title_for_lookup(title)
    if not lookup_title:
        _dbg(f"empty lookup_title from title={title!r}")
        return None

    norm_title = lookup_title
    norm_year = str(year) if year is not None else ""
    imdb_norm = _norm_imdb(imdb_id)

    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        cached = _get_cached_item(cache=cache, norm_title=norm_title, norm_year=norm_year, imdb_id=imdb_norm)
        if cached is not None:
            _m_inc("cache_hits", 1)
            _dbg("cache HIT")
            return cached

    _m_inc("cache_misses", 1)

    # --------------------------------------------------------
    # Path A: imdb -> qid -> entity -> sitelink -> wikipedia summary
    # --------------------------------------------------------
    if imdb_norm:
        _m_inc("path_imdb", 1)

        with _CACHE_LOCK:
            cache = _load_cache_unlocked()
            qid = _fetch_qid_by_imdb_id(cache, imdb_norm)

        if not qid:
            neg = _build_negative_item(
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=imdb_norm,
                status="imdb_no_qid",
                wikibase_item=None,
                primary_language=primary,
                fallback_language=fallback,
            )
            with _CACHE_LOCK:
                cache = _load_cache_unlocked()
                _store_item_unlocked(cache, neg)
            return neg

        wd_entity = _fetch_wikidata_entity_json(qid)
        if wd_entity is None:
            return None

        chain = _detect_language_chain_from_input(movie_input)
        sl_title, sl_lang = _pick_best_sitelink_title(wd_entity, chain)

        wiki_raw: Mapping[str, object] | None = None
        try:
            wiki_raw = _fetch_wikipedia_summary_by_title(sl_title, sl_lang) if sl_title else None
        except _WikiDisambiguationError:
            # Negative caching explícito (TTL corto) para evitar repetir ambigüedades.
            dis = _build_negative_item(
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=imdb_norm,
                status="disambiguation",
                wikibase_item=qid,
                primary_language=primary,
                fallback_language=fallback,
            )
            with _CACHE_LOCK:
                cache = _load_cache_unlocked()
                _store_item_unlocked(cache, dis)
            return dis

        with _CACHE_LOCK:
            cache = _load_cache_unlocked()

            if not _is_film_cached(cache=cache, qid=qid, wd_entity=wd_entity, wiki_raw=wiki_raw, wiki_lang=sl_lang):
                neg_nf = _build_negative_item(
                    norm_title=norm_title,
                    norm_year=norm_year,
                    imdb_id=imdb_norm,
                    status="not_film",
                    wikibase_item=qid,
                    primary_language=primary,
                    fallback_language=fallback,
                )
                _store_item_unlocked(cache, neg_nf)
                return neg_nf

            if wiki_raw is None:
                return None

            ok_item = _build_ok_item_and_merge_entities(
                cache=cache,
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=imdb_norm,
                wiki_raw=wiki_raw,
                source_language=sl_lang,
                wikibase_item=qid,
                wd_entity=wd_entity,
                primary_language=primary,
                fallback_language=fallback,
            )
            _store_item_unlocked(cache, ok_item)
            return ok_item

    # --------------------------------------------------------
    # Path B: title/year -> wikipedia search -> summary -> wikibase_item -> entity
    # --------------------------------------------------------
    _m_inc("path_title_search", 1)

    candidates = _choose_wikipedia_summary_candidates(
        title_for_lookup=lookup_title,
        year=year,
        primary_language=primary,
        fallback_language=fallback,
    )
    _dbg(f"candidates={len(candidates)} primary={primary} fallback={fallback}")

    for cand_title, cand_lang in candidates:
        try:
            wiki_raw = _fetch_wikipedia_summary_by_title(cand_title, cand_lang)
        except _WikiDisambiguationError:
            dis2 = _build_negative_item(
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=None,
                status="disambiguation",
                wikibase_item=None,
                primary_language=primary,
                fallback_language=fallback,
            )
            with _CACHE_LOCK:
                cache = _load_cache_unlocked()
                _store_item_unlocked(cache, dis2)
            return dis2

        if wiki_raw is None:
            continue

        wikibase_item = _safe_str(wiki_raw.get("wikibase_item"))
        if not wikibase_item:
            continue

        wd_entity = _fetch_wikidata_entity_json(wikibase_item)
        if wd_entity is None:
            continue

        with _CACHE_LOCK:
            cache = _load_cache_unlocked()

            if not _is_film_cached(
                cache=cache,
                qid=wikibase_item,
                wd_entity=wd_entity,
                wiki_raw=wiki_raw,
                wiki_lang=cand_lang,
            ):
                neg_nf2 = _build_negative_item(
                    norm_title=norm_title,
                    norm_year=norm_year,
                    imdb_id=None,
                    status="not_film",
                    wikibase_item=wikibase_item,
                    primary_language=primary,
                    fallback_language=fallback,
                )
                _store_item_unlocked(cache, neg_nf2)
                continue

            ok_item2 = _build_ok_item_and_merge_entities(
                cache=cache,
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=None,
                wiki_raw=wiki_raw,
                source_language=cand_lang,
                wikibase_item=wikibase_item,
                wd_entity=wd_entity,
                primary_language=primary,
                fallback_language=fallback,
            )
            _store_item_unlocked(cache, ok_item2)
            return ok_item2

    neg = _build_negative_item(
        norm_title=norm_title,
        norm_year=norm_year,
        imdb_id=None,
        status="no_qid",
        wikibase_item=None,
        primary_language=primary,
        fallback_language=fallback,
    )
    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        _store_item_unlocked(cache, neg)
    return neg


# ============================================================================
# API pública (sin wrappers OO)
# ============================================================================

def get_wiki(*, title: str, year: int | None, imdb_id: str | None) -> WikiCacheItem | None:
    primary = _normalize_lang_code(WIKI_LANGUAGE) or "en"
    fallback = _normalize_lang_code(WIKI_FALLBACK_LANGUAGE) or ("en" if primary != "en" else "")
    return _get_wiki_impl(title=title, year=year, imdb_id=imdb_id, movie_input=None, primary=primary, fallback=fallback)


def get_wiki_for_input(
    *,
    movie_input: MovieInputLangProto,
    title: str,
    year: int | None,
    imdb_id: str | None,
) -> WikiCacheItem | None:
    primary, fallback = _best_wikipedia_languages_for_item(movie_input)
    return _get_wiki_impl(
        title=title,
        year=year,
        imdb_id=imdb_id,
        movie_input=movie_input,
        primary=primary,
        fallback=fallback,
    )