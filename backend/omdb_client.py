"""
backend/omdb_client.py (schema v4)

Cliente OMDb + caché persistente indexada + TTL + bounded in-memory hot-cache
+ compaction en disco + limitador global + métricas + resumen final de métricas.

Objetivo de este archivo (además de la funcionalidad):
- ✅ Evitar falsos "unreachable" de Pyright/Pylance por stubs rotos (NoReturn)
  en logging/time.sleep u otros wrappers.
- ✅ Evitar "no-redef" por imports repetidos.

──────────────────────────────────────────────────────────────────────────────
CAMBIOS APLICADOS EN ESTA REVISION
──────────────────────────────────────────────────────────────────────────────
1) HOT CACHE: validación de “shape” antes de devolver dicts desde hot-cache:
   - añade _is_cache_item_dict() y usa cast(OmdbCacheItem, ...)
   - evita devolver dicts arbitrarios como si fueran OmdbCacheItem.

2) WIKI sanitize compat:
   - _sanitize_wiki_lookup acepta imdbID como fallback legacy si falta imdb_id.
   - _sanitize_wiki_block acepta imdbID como fallback legacy si falta imdb_id.  ✅

3) Compaction: coherencia con nota anti-unreachable:
   - reescribe el bucle para evitar `continue` en guard clauses.

4) Métricas sleep:
   - añade helper _m_add_ms() para *_ms_total (robustez y claridad).

5) HOT CACHE: evitar compartir dict mutable del cache persistente:
   - al devolver item desde disco/mem-cache, se guarda una copia superficial en hot-cache ✅
"""

from __future__ import annotations

import atexit
import json
import os
import random
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Final, Literal, Protocol, TypedDict, cast, runtime_checkable

import requests  # type: ignore[import-untyped]
from requests import Response  # type: ignore[import-not-found]
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from requests.exceptions import RequestException  # type: ignore[import-untyped]
from urllib3.util.retry import Retry  # type: ignore[import-not-found]

from backend import logger as logger
from backend.config_omdb import (
    ANALIZA_OMDB_CACHE_MAX_INDEX_IMDB,
    ANALIZA_OMDB_CACHE_MAX_INDEX_TY,
    ANALIZA_OMDB_CACHE_MAX_RECORDS,
    ANALIZA_OMDB_HOT_CACHE_MAX,
    OMDB_API_KEY,
    OMDB_API_KEYS,
    OMDB_BASE_URL,
    OMDB_CACHE_COMPACT_EVERY_N_FLUSHES,
    OMDB_CACHE_FLUSH_MAX_DIRTY_WRITES,
    OMDB_CACHE_FLUSH_MAX_SECONDS,
    OMDB_CACHE_JSON_INDENT,
    OMDB_CACHE_JSON_PRETTY,
    OMDB_CACHE_PATH,
    OMDB_CACHE_TTL_EMPTY_RATINGS_SECONDS,
    OMDB_CACHE_TTL_NOT_FOUND_SECONDS,
    OMDB_CACHE_TTL_OK_SECONDS,
    OMDB_CIRCUIT_BREAKER_OPEN_SECONDS,
    OMDB_CIRCUIT_BREAKER_THRESHOLD,
    OMDB_DISABLE_AFTER_N_FAILURES,
    OMDB_HOT_MISS_TTL_SECONDS,
    OMDB_HTTP_MAX_CONCURRENCY,
    OMDB_HTTP_MIN_INTERVAL_SECONDS,
    OMDB_HTTP_RETRY_BACKOFF_FACTOR,
    OMDB_HTTP_RETRY_TOTAL,
    OMDB_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT,
    OMDB_HTTP_TIMEOUT_SECONDS,
    OMDB_HTTP_USER_AGENT,
    OMDB_JITTER_RATIO,
    OMDB_KEYS_STATE_PATH,
    OMDB_KEY_EXHAUSTED_TTL_SECONDS,
    OMDB_KEY_ROTATION_ENABLED,
    OMDB_KEY_ROTATION_MAX_TRIES,
    OMDB_METRICS_ENABLED,
    OMDB_METRICS_LOG_EVEN_IF_ZERO,
    OMDB_METRICS_LOG_ON_SILENT_DEBUG,
    OMDB_METRICS_TOP_N,
    OMDB_RATE_LIMIT_MAX_RETRIES,
    OMDB_RATE_LIMIT_WAIT_SECONDS,
    OMDB_SINGLEFLIGHT_WAIT_SECONDS,
)
from backend.title_utils import (
    generate_sequel_title_variants,
    normalize_title_for_lookup,
)

# ============================================================
# Optional: RunMetrics (best-effort) - tipado con Protocol
# ============================================================


@runtime_checkable
class _RunMetricsProto(Protocol):
    def incr(self, key: str, delta: int = 1) -> None: ...
    def observe_ms(self, key: str, ms: int) -> None: ...


_RUN_METRICS: _RunMetricsProto | None = None
try:
    from backend.run_metrics import METRICS as _RUN_METRICS  # type: ignore[assignment]
except Exception:
    _RUN_METRICS = None


def _rm_incr(key: str, delta: int = 1) -> None:
    """Best-effort: incrementa métrica global del run si existe. (anti-pyright-unreachable)"""
    m_any: Any = _RUN_METRICS  # rompe propagación de NoReturn desde stubs externos
    if m_any is None:
        return
    fn_obj: object = getattr(m_any, "incr", None)
    if not callable(fn_obj):
        return
    try:
        cast(Callable[..., object], fn_obj)(str(key), int(delta))
    except Exception:
        pass


def _rm_observe_ms(key: str, ms: int) -> None:
    """Best-effort: observa latencias (ms) en métrica global del run si existe. (anti-pyright-unreachable)"""
    m_any: Any = _RUN_METRICS
    if m_any is None:
        return
    fn_obj: object = getattr(m_any, "observe_ms", None)
    if not callable(fn_obj):
        return
    try:
        cast(Callable[..., object], fn_obj)(str(key), int(ms))
    except Exception:
        pass


# ============================================================
# SAFE CAST HELPERS (para Pylance: int(object) / float(object))
# ============================================================


def _int_or_none(value: object | None) -> int | None:
    """
    Convierte a int solo si el tipo es seguro (int/float/str numérico).
    Devuelve None si no se puede.
    """
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            s = value.strip()
            if not s or s.upper() == "N/A":
                return None
            s = s.replace(",", "")
            return int(float(s))
        return None
    except (ValueError, TypeError):
        return None


def _int_default(value: object | None, default: int = 0) -> int:
    out = _int_or_none(value)
    return out if out is not None else int(default)


def _float_or_none(value: object | None) -> float | None:
    """
    Convierte a float solo si el tipo es seguro (int/float/str numérico).
    Devuelve None si no se puede.
    """
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if not s or s.upper() == "N/A":
                return None
            s = s.replace(",", "")
            return float(s)
        return None
    except (ValueError, TypeError):
        return None


# ============================================================
#                       CACHE v4 (records + indexes + TTL)
# ============================================================

CacheStatus = Literal["ok", "not_found", "empty_ratings"]


class OmdbCacheItem(TypedDict):
    Title: str
    Year: str
    imdbID: str | None
    omdb: dict[str, object]
    fetched_at: int
    ttl_s: int
    status: CacheStatus


class OmdbCacheFile(TypedDict):
    schema: int
    records: dict[str, OmdbCacheItem]
    index_imdb: dict[str, str]
    index_ty: dict[str, str]


_SCHEMA_VERSION: Final[int] = 4
_CACHE_PATH: Final[Path] = Path(OMDB_CACHE_PATH)

_CACHE: OmdbCacheFile | None = None
_CACHE_LOCK = threading.RLock()

# Flush batching
_FLUSH_MAX_DIRTY_WRITES: Final[int] = max(1, int(OMDB_CACHE_FLUSH_MAX_DIRTY_WRITES))
_FLUSH_MAX_SECONDS: Final[float] = max(0.1, float(OMDB_CACHE_FLUSH_MAX_SECONDS))

_CACHE_DIRTY: bool = False
_CACHE_DIRTY_WRITES: int = 0
_CACHE_LAST_FLUSH_TS: float = 0.0
_FLUSH_COUNT: int = 0  # para compaction amortizada si se configura

# Caps para compaction
_COMPACT_MAX_RECORDS: Final[int] = int(ANALIZA_OMDB_CACHE_MAX_RECORDS)
_COMPACT_MAX_INDEX_IMDB: Final[int] = int(ANALIZA_OMDB_CACHE_MAX_INDEX_IMDB)
_COMPACT_MAX_INDEX_TY: Final[int] = int(ANALIZA_OMDB_CACHE_MAX_INDEX_TY)

# JSON cache output (tunable)
_CACHE_JSON_PRETTY: Final[bool] = bool(OMDB_CACHE_JSON_PRETTY)
_CACHE_JSON_INDENT: Final[int] = max(0, int(OMDB_CACHE_JSON_INDENT))
_CACHE_COMPACT_EVERY_N_FLUSHES: Final[int] = max(
    0, int(OMDB_CACHE_COMPACT_EVERY_N_FLUSHES)
)

# Bounded in-memory hot-cache (acelerador intra-run)
_HOT_CACHE_MAX: Final[int] = int(ANALIZA_OMDB_HOT_CACHE_MAX)
_HOT_CACHE_LOCK = threading.Lock()

# value: (obj, last_touch_monotonic, expires_at_monotonic_or_0)
# - expires_at=0 => no expira (LRU normal)
_HOT_CACHE: dict[str, tuple[object, float, float]] = {}

# Sentinel explícito para negative cache intra-run
_HOT_MISS: Final[object] = object()
_HOT_MISS_TTL_S: Final[float] = max(0.0, float(OMDB_HOT_MISS_TTL_SECONDS))

# ============================================================
# SINGLE-FLIGHT (evita thundering herd por clave de lookup)
# ============================================================

_SINGLEFLIGHT_LOCK = threading.Lock()
_SINGLEFLIGHT_PRIMARY_EVENTS: dict[str, threading.Event] = {}
_SINGLEFLIGHT_KEY_TO_PRIMARY: dict[str, str] = {}
_SINGLEFLIGHT_PRIMARY_TO_KEYS: dict[str, set[str]] = {}
_SINGLEFLIGHT_WAIT_S: Final[float] = max(0.05, float(OMDB_SINGLEFLIGHT_WAIT_SECONDS))


def _singleflight_enter_multi(keys: list[str]) -> tuple[bool, threading.Event, str]:
    keys2 = [k for k in keys if isinstance(k, str) and k.strip()]
    if not keys2:
        ev = threading.Event()
        ev.set()
        return True, ev, ""

    with _SINGLEFLIGHT_LOCK:
        for k in keys2:
            pk = _SINGLEFLIGHT_KEY_TO_PRIMARY.get(k)
            if pk:
                ev2 = _SINGLEFLIGHT_PRIMARY_EVENTS.get(pk)
                if ev2 is not None:
                    return False, ev2, pk

        primary = keys2[0]
        ev = threading.Event()
        _SINGLEFLIGHT_PRIMARY_EVENTS[primary] = ev
        _SINGLEFLIGHT_PRIMARY_TO_KEYS[primary] = set(keys2)
        for k in keys2:
            _SINGLEFLIGHT_KEY_TO_PRIMARY[k] = primary
        return True, ev, primary


def _singleflight_leave(primary_key: str) -> None:
    if not primary_key:
        return

    with _SINGLEFLIGHT_LOCK:
        ev = _SINGLEFLIGHT_PRIMARY_EVENTS.pop(primary_key, None)
        keys = _SINGLEFLIGHT_PRIMARY_TO_KEYS.pop(primary_key, set())
        for k in keys:
            if _SINGLEFLIGHT_KEY_TO_PRIMARY.get(k) == primary_key:
                _SINGLEFLIGHT_KEY_TO_PRIMARY.pop(k, None)

    if ev is not None:
        ev.set()


# ============================================================
# TIME + keys
# ============================================================


def _now_epoch() -> int:
    return int(time.time())


def _ty_key(norm_title: str, norm_year: str) -> str:
    return f"{norm_title}|{norm_year}"


def _cache_key_for_imdb(imdb_id: str) -> str:
    return f"imdb:{imdb_id.lower()}"


def _cache_key_for_title_year(norm_title: str, norm_year: str) -> str:
    return f"ty:{_ty_key(norm_title, norm_year)}"


def _cache_key_for_title_only(norm_title: str) -> str:
    return f"t:{norm_title}"


def _rid_for_record(*, imdb_norm: str | None, norm_title: str, norm_year: str) -> str:
    if imdb_norm:
        return f"imdb:{imdb_norm}"
    return f"ty:{_ty_key(norm_title, norm_year)}"


def _is_expired_item(item: OmdbCacheItem, now_epoch: int) -> bool:
    fetched_at = _int_default(item.get("fetched_at"), 0)
    ttl_s = _int_default(item.get("ttl_s"), 0)
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
    "http_semaphore_timeouts": 0,
    "throttle_sleeps": 0,
    "rate_limit_hits": 0,
    "rate_limit_sleeps": 0,
    "disabled_switches": 0,
    "candidate_search_calls": 0,
    "cache_expired_hits": 0,
    "hot_cache_hits": 0,
    "hot_cache_misses": 0,
    "hot_cache_evictions": 0,
    "singleflight_waits": 0,
    "singleflight_wait_ms_total": 0,
    "cb_open_hits": 0,
    "cb_opens": 0,
    "cb_half_open_probes": 0,
    "cb_half_open_rejects": 0,
    "http_latency_ms_total": 0,
    "http_latency_ms_max": 0,
    "semaphore_wait_ms_total": 0,
    "throttle_sleep_ms_total": 0,
    "rate_limit_sleep_ms_total": 0,
    "jitter_sleeps": 0,
}


def _m_inc(key: str, delta: int = 1) -> None:
    with _METRICS_LOCK:
        _METRICS[key] = int(_METRICS.get(key, 0)) + int(delta)
    _rm_incr(f"omdb.{key}", int(delta))


def _m_add_ms(key: str, ms: int) -> None:
    # helper para contadores *_ms_total (evita ambigüedad con _m_inc)
    try:
        ims = int(ms)
    except (ValueError, TypeError):
        return
    if ims <= 0:
        return
    _m_inc(str(key), ims)


def _m_max(key: str, value: int) -> None:
    with _METRICS_LOCK:
        cur = int(_METRICS.get(key, 0))
        if value > cur:
            _METRICS[key] = value


def get_omdb_metrics_snapshot() -> dict[str, int]:
    with _METRICS_LOCK:
        return dict(_METRICS)


def reset_omdb_metrics() -> None:
    with _METRICS_LOCK:
        for k in list(_METRICS.keys()):
            _METRICS[k] = 0


# ============================================================
# LOGGING SAFE WRAPPERS (anti-pyright-unreachable)
# ============================================================

_LOGGER: Any = logger


def _log_call(fn_name: str, *args: object, **kwargs: object) -> None:
    fn_obj: object = getattr(_LOGGER, fn_name, None)
    if not callable(fn_obj):
        return
    try:
        cast(Callable[..., object], fn_obj)(*args, **kwargs)
    except Exception:
        pass


def _dbg(msg: object) -> None:
    _log_call("debug_ctx", "OMDB", msg)


def _warn_always(msg: object) -> None:
    _log_call("warning", str(msg), always=True)


def _info(msg: object) -> None:
    _log_call("info", str(msg))


def _progress(msg: object) -> None:
    _log_call("progress", str(msg))


# ============================================================
# MÉTRICAS: LOG AL FINAL DEL RUN
# ============================================================


def _metrics_any_nonzero(snapshot: Mapping[str, int]) -> bool:
    try:
        return any(int(v) != 0 for v in snapshot.values())
    except (ValueError, TypeError):
        return True


def _format_metrics_top(
    snapshot: Mapping[str, int], top_n: int
) -> list[tuple[str, int]]:
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

    lines: list[str] = ["[OMDB][METRICS] summary"]
    if not top:
        lines.append("  (all zeros)")
    else:
        max_k = max((len(k) for k, _ in top), default=10)
        for k, v in top:
            lines.append(f"  {k.ljust(max_k)} : {v}")

    if silent and debug:
        for ln in lines:
            _progress(ln)
    else:
        for ln in lines:
            _info(ln)


# ============================================================
# JITTER / SLEEP helpers
# ============================================================

_JITTER_RATIO: Final[float] = max(0.0, min(0.5, float(OMDB_JITTER_RATIO)))


def _safe_sleep(seconds: float) -> None:
    """
    FIX pyright unreachable:
    - algunos stubs marcan time.sleep como NoReturn (bug / stub roto).
    - lo llamamos vía getattr sobre Any para evitar propagación.
    """
    t_any: Any = time
    fn_obj: object = getattr(t_any, "sleep", None)
    if not callable(fn_obj):
        return
    try:
        cast(Callable[..., object], fn_obj)(float(seconds))
    except Exception:
        pass


def _sleep_with_jitter(
    base_seconds: float, *, reason_metric_ms_total: str | None = None
) -> float:
    s = max(0.0, float(base_seconds))
    if s <= 0.0:
        return 0.0

    if _JITTER_RATIO > 0.0:
        factor = 1.0 + random.uniform(-_JITTER_RATIO, _JITTER_RATIO)
        s = max(0.0, s * factor)
        _m_inc("jitter_sleeps", 1)

    if reason_metric_ms_total:
        _m_add_ms(str(reason_metric_ms_total), int(s * 1000.0))

    _safe_sleep(s)
    return s


# ============================================================
# HTTP session + retry (lazy-init thread-safe)
# ============================================================

_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()


def _cap_int_runtime(value: int, *, min_v: int, max_v: int) -> int:
    if value < min_v:
        return min_v
    if value > max_v:
        return max_v
    return value


def _cap_float_runtime(
    value: float, *, min_v: float, max_v: float | None = None
) -> float:
    if value < min_v:
        return min_v
    if max_v is not None and value > max_v:
        return max_v
    return value


def _get_session() -> requests.Session:
    """
    FIX pyright unreachable:
    - usa variable local para evitar narrowing agresivo dentro del lock.
    """
    global _SESSION

    session = _SESSION
    if session is not None:
        return session

    with _SESSION_LOCK:
        session = _SESSION
        if session is not None:
            return session

        retry_total = _cap_int_runtime(int(OMDB_HTTP_RETRY_TOTAL), min_v=0, max_v=10)
        backoff = _cap_float_runtime(
            float(OMDB_HTTP_RETRY_BACKOFF_FACTOR), min_v=0.0, max_v=10.0
        )
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
            pool_block=True,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        ua = str(OMDB_HTTP_USER_AGENT).strip() or "Analiza-Movies/1.0 (local)"
        session.headers.update(
            {"User-Agent": ua, "Accept": "application/json,text/plain,*/*"}
        )

        _SESSION = session
        return session


# ============================================================
# CIRCUIT BREAKER (CLOSED/OPEN/HALF-OPEN)
# ============================================================

_CB_LOCK = threading.Lock()
_CB_FAILURES: int = 0
_CB_OPEN_UNTIL: float = 0.0
_CB_HALF_OPEN_PROBE_IN_FLIGHT: bool = False

_CB_THRESHOLD: Final[int] = max(1, int(OMDB_CIRCUIT_BREAKER_THRESHOLD))
_CB_OPEN_SECONDS: Final[float] = max(0.5, float(OMDB_CIRCUIT_BREAKER_OPEN_SECONDS))


def _cb_state(now: float) -> Literal["closed", "open", "half_open"]:
    with _CB_LOCK:
        if now < _CB_OPEN_UNTIL:
            return "open"
        if _CB_FAILURES >= _CB_THRESHOLD and _CB_OPEN_UNTIL > 0.0:
            return "half_open"
        return "closed"


def _cb_try_enter() -> bool:
    global _CB_HALF_OPEN_PROBE_IN_FLIGHT

    now = time.monotonic()
    st = _cb_state(now)
    if st == "closed":
        return True
    if st == "open":
        _m_inc("cb_open_hits", 1)
        return False

    with _CB_LOCK:
        if _CB_HALF_OPEN_PROBE_IN_FLIGHT:
            _m_inc("cb_half_open_rejects", 1)
            return False
        _CB_HALF_OPEN_PROBE_IN_FLIGHT = True
        _m_inc("cb_half_open_probes", 1)
        return True


def _cb_record_success() -> None:
    global _CB_FAILURES, _CB_OPEN_UNTIL, _CB_HALF_OPEN_PROBE_IN_FLIGHT
    with _CB_LOCK:
        _CB_FAILURES = 0
        _CB_OPEN_UNTIL = 0.0
        _CB_HALF_OPEN_PROBE_IN_FLIGHT = False


def _cb_record_failure() -> None:
    global _CB_FAILURES, _CB_OPEN_UNTIL, _CB_HALF_OPEN_PROBE_IN_FLIGHT

    now = time.monotonic()
    with _CB_LOCK:
        _CB_FAILURES += 1

        if _CB_HALF_OPEN_PROBE_IN_FLIGHT:
            _CB_HALF_OPEN_PROBE_IN_FLIGHT = False
            _CB_OPEN_UNTIL = now + float(_CB_OPEN_SECONDS)
            _m_inc("cb_opens", 1)
            return

        if _CB_FAILURES >= _CB_THRESHOLD:
            new_until = now + float(_CB_OPEN_SECONDS)
            if new_until > _CB_OPEN_UNTIL:
                _CB_OPEN_UNTIL = new_until
            _m_inc("cb_opens", 1)


# ============================================================
# LIMITADOR GLOBAL OMDb (ThreadPool safe)
# ============================================================

_MAX_CONCURRENCY_RT: Final[int] = _cap_int_runtime(
    int(OMDB_HTTP_MAX_CONCURRENCY), min_v=1, max_v=256
)
_OMDB_HTTP_SEMAPHORE = threading.Semaphore(_MAX_CONCURRENCY_RT)

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

_DISABLE_AFTER_N_FAILURES: Final[int] = _cap_int_runtime(
    int(OMDB_DISABLE_AFTER_N_FAILURES), min_v=1, max_v=50
)


def _mark_omdb_failure() -> None:
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
    global _OMDB_CONSECUTIVE_FAILURES
    with _OMDB_FAILURES_LOCK:
        _OMDB_CONSECUTIVE_FAILURES = 0


def _omdb_http_get(
    *, base_url: str, params: Mapping[str, str], timeout_seconds: float
) -> Response | None:
    global _OMDB_HTTP_LAST_REQUEST_TS

    if OMDB_DISABLED:
        return None

    if not _cb_try_enter():
        return None

    acquire_timeout = _cap_float_runtime(
        float(OMDB_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT), min_v=0.1, max_v=120.0
    )

    t0_sem = time.monotonic()
    acquired = _OMDB_HTTP_SEMAPHORE.acquire(timeout=acquire_timeout)
    sem_wait_ms = int((time.monotonic() - t0_sem) * 1000.0)
    if sem_wait_ms > 0:
        _m_add_ms("semaphore_wait_ms_total", sem_wait_ms)
        _rm_observe_ms("omdb.http.semaphore_wait_ms", sem_wait_ms)

    if not acquired:
        _m_inc("http_semaphore_timeouts", 1)
        _dbg(
            f"Semaphore acquire timeout ({acquire_timeout:.1f}s). Skipping OMDb request."
        )
        _cb_record_failure()
        return None

    try:
        min_interval = max(0.0, float(OMDB_HTTP_MIN_INTERVAL_SECONDS))
        if min_interval > 0.0:
            with _OMDB_HTTP_THROTTLE_LOCK:
                now = time.monotonic()
                wait_s = (_OMDB_HTTP_LAST_REQUEST_TS + min_interval) - now
                if wait_s > 0.0:
                    _m_inc("throttle_sleeps", 1)
                    _dbg(f"Throttle: sleeping {wait_s:.3f}s (pre-jitter)")
                    slept = _sleep_with_jitter(
                        wait_s, reason_metric_ms_total="throttle_sleep_ms_total"
                    )
                    _rm_observe_ms("omdb.http.throttle_sleep_ms", int(slept * 1000.0))
                _OMDB_HTTP_LAST_REQUEST_TS = time.monotonic()

        session = _get_session()
        t_timeout = _cap_float_runtime(float(timeout_seconds), min_v=0.5, max_v=120.0)

        t0 = time.monotonic()
        _m_inc("http_requests", 1)
        resp = session.get(base_url, params=dict(params), timeout=t_timeout)
        dt_ms = int((time.monotonic() - t0) * 1000.0)
        if dt_ms >= 0:
            _m_add_ms("http_latency_ms_total", dt_ms)
            _m_max("http_latency_ms_max", dt_ms)
            _rm_observe_ms("omdb.http.latency_ms", dt_ms)

        return resp

    except RequestException as exc:
        _m_inc("http_failures", 1)
        _dbg(f"HTTP error calling OMDb: {exc!r}")
        _cb_record_failure()
        return None

    finally:
        _OMDB_HTTP_SEMAPHORE.release()


# ============================================================
# OMDb API KEY ROTATION (multi-key)
# ============================================================

_KEYS_STATE_LOCK = threading.RLock()
_KEYS_STATE_SCHEMA: Final[int] = 1
_KEYS_STATE_PATH: Final[Path] = Path(OMDB_KEYS_STATE_PATH)

_KEY_CURSOR: int = 0  # round-robin en memoria


def _configured_keys() -> list[str]:
    keys = [
        k.strip() for k in (OMDB_API_KEYS or []) if isinstance(k, str) and k.strip()
    ]
    if keys:
        return keys
    if OMDB_API_KEY is None:
        return []
    k = str(OMDB_API_KEY).strip()
    return [k] if k else []


def _normalize_key_state_record(rec: Mapping[str, object]) -> dict[str, object]:
    """
    Normaliza un registro de key-state:
    - exhausted_until -> float
    - invalid -> bool
    """
    out: dict[str, object] = {}

    # exhausted_until
    until_f = _float_or_none(rec.get("exhausted_until"))
    out["exhausted_until"] = float(until_f) if until_f is not None else 0.0

    # invalid
    out["invalid"] = bool(rec.get("invalid", False))

    return out


def _load_keys_state() -> dict[str, dict[str, object]]:
    try:
        if not _KEYS_STATE_PATH.exists():
            return {}

        raw = _KEYS_STATE_PATH.read_text(encoding="utf-8")
        obj = json.loads(raw)

        if not isinstance(obj, dict):
            return {}
        if obj.get("schema") != _KEYS_STATE_SCHEMA:
            return {}

        ks = obj.get("keys")
        if not isinstance(ks, dict):
            return {}

        out: dict[str, dict[str, object]] = {}

        for k, v in ks.items():
            if not isinstance(k, str) or not k.strip():
                continue
            if not isinstance(v, Mapping):
                continue

            out[k] = _normalize_key_state_record(v)

        return out

    except Exception:
        return {}


def _save_keys_state(state: dict[str, dict[str, object]]) -> None:
    tmp_path: str | None = None
    try:
        _KEYS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": _KEYS_STATE_SCHEMA, "keys": state}
        data = json.dumps(payload, ensure_ascii=False, indent=2)

        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            encoding="utf-8",
            dir=str(_KEYS_STATE_PATH.parent),
        ) as f:
            f.write(data)
            tmp_path = f.name

        if tmp_path:
            os.replace(tmp_path, _KEYS_STATE_PATH)

    except Exception:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _get_key_state_locked() -> dict[str, dict[str, object]]:
    # caller must hold _KEYS_STATE_LOCK
    if not hasattr(_get_key_state_locked, "_cache"):
        setattr(_get_key_state_locked, "_cache", _load_keys_state())
    st = cast(dict[str, dict[str, object]], getattr(_get_key_state_locked, "_cache"))
    return st


def _is_key_available(api_key: str, now: float) -> bool:
    with _KEYS_STATE_LOCK:
        st = _get_key_state_locked()
        rec = st.get(api_key)
        if not isinstance(rec, Mapping):
            return True

        if bool(rec.get("invalid")):
            return False

        until_any = rec.get("exhausted_until", 0.0)
        until_f = _float_or_none(until_any)
        return (until_f is None) or (float(until_f) <= now)


def _mark_key_exhausted(api_key: str) -> None:
    ttl = max(60.0, float(OMDB_KEY_EXHAUSTED_TTL_SECONDS))
    until = time.time() + ttl
    with _KEYS_STATE_LOCK:
        st = _get_key_state_locked()

        prev = st.get(api_key)
        rec: dict[str, object] = dict(prev) if isinstance(prev, Mapping) else {}

        rec["exhausted_until"] = float(until)
        rec["invalid"] = bool(rec.get("invalid", False))

        st[api_key] = rec
        _save_keys_state(st)


def _mark_key_invalid(api_key: str) -> None:
    with _KEYS_STATE_LOCK:
        st = _get_key_state_locked()

        prev = st.get(api_key)
        rec: dict[str, object] = dict(prev) if isinstance(prev, Mapping) else {}

        rec["invalid"] = True
        rec["exhausted_until"] = float(time.time() + 60.0 * 60.0 * 24.0 * 365.0)

        st[api_key] = rec
        _save_keys_state(st)


def _pick_next_key(keys: list[str]) -> str | None:
    global _KEY_CURSOR
    if not keys:
        return None
    now = time.time()
    start = _KEY_CURSOR % len(keys)
    for i in range(len(keys)):
        idx = (start + i) % len(keys)
        k = keys[idx]
        if _is_key_available(k, now):
            _KEY_CURSOR = (idx + 1) % len(keys)
            return k
    return None


def _rotation_tries_cap(keys_count: int) -> int:
    mx = int(OMDB_KEY_ROTATION_MAX_TRIES)
    if mx <= 0:
        return keys_count
    return min(mx, keys_count)


# ============================================================
# AUX: safe parsing / ratings
# ============================================================


def _safe_int(value: object | None) -> int | None:
    return _int_or_none(value)


def _safe_float(value: object | None) -> float | None:
    return _float_or_none(value)


def _safe_imdb_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v.lower() if v else None


def normalize_imdb_votes(votes: object) -> int | None:
    if not votes or votes == "N/A":
        return None
    if isinstance(votes, (int, float)):
        return int(votes)
    s = str(votes).strip().replace(",", "")
    return _safe_int(s)


def parse_rt_score_from_omdb(omdb_data: Mapping[str, object]) -> int | None:
    ratings_obj = omdb_data.get("Ratings") or []
    if not isinstance(ratings_obj, list):
        return None
    for r in ratings_obj:
        if isinstance(r, Mapping):
            if r.get("Source") == "Rotten Tomatoes":
                val = r.get("Value")
                if isinstance(val, str) and val.endswith("%"):
                    try:
                        return int(val[:-1])
                    except ValueError:
                        return None
    return None


def parse_imdb_rating_from_omdb(omdb_data: Mapping[str, object]) -> float | None:
    raw = omdb_data.get("imdbRating")
    if not raw or raw == "N/A":
        return None
    return _safe_float(raw)


def extract_year_from_omdb(omdb_data: Mapping[str, object]) -> int | None:
    raw = omdb_data.get("Year")
    if not raw or raw == "N/A":
        return None
    text = str(raw).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def parse_metacritic_score_from_omdb(omdb_data: Mapping[str, object]) -> int | None:
    # 1) Campo directo "Metascore": "83"
    raw = omdb_data.get("Metascore")
    if isinstance(raw, str):
        s = raw.strip()
        if s and s.upper() != "N/A" and s.isdigit():
            try:
                v = int(s)
                return v if 0 <= v <= 100 else None
            except Exception:
                pass

    # 2) Fallback en Ratings: {"Source":"Metacritic","Value":"83/100"}
    ratings_obj = omdb_data.get("Ratings")
    if isinstance(ratings_obj, list):
        for r in ratings_obj:
            if isinstance(r, Mapping) and r.get("Source") == "Metacritic":
                val = r.get("Value")
                if isinstance(val, str):
                    s2 = val.strip()
                    m = re.match(r"^(\d{1,3})\s*/\s*100$", s2)
                    if m:
                        try:
                            v2 = int(m.group(1))
                            return v2 if 0 <= v2 <= 100 else None
                        except Exception:
                            return None
    return None


def extract_ratings_from_omdb(
    data: Mapping[str, object] | None,
) -> tuple[float | None, int | None, int | None, int | None]:
    if not data:
        return None, None, None, None
    imdb_rating = parse_imdb_rating_from_omdb(data)
    imdb_votes = normalize_imdb_votes(data.get("imdbVotes"))
    rt_score = parse_rt_score_from_omdb(data)
    metacritic_score = parse_metacritic_score_from_omdb(data)
    return imdb_rating, imdb_votes, rt_score, metacritic_score


def is_omdb_data_empty_for_ratings(data: Mapping[str, object] | None) -> bool:
    if not data:
        return True
    imdb_rating = parse_imdb_rating_from_omdb(data)
    imdb_votes = normalize_imdb_votes(data.get("imdbVotes"))
    rt_score = parse_rt_score_from_omdb(data)
    metacritic_score = parse_metacritic_score_from_omdb(data)
    return (
        imdb_rating is None
        and imdb_votes is None
        and rt_score is None
        and metacritic_score is None
    )


def _extract_imdb_id_from_omdb_record(data: Mapping[str, object] | None) -> str | None:
    if not isinstance(data, Mapping):
        return None
    if data.get("Response") != "True":
        return None
    return _safe_imdb_id(data.get("imdbID"))


def _norm_year_str(year: int | None) -> str:
    return str(year) if year is not None else ""


def _is_movie_not_found(data: Mapping[str, object]) -> bool:
    return data.get("Response") == "False" and data.get("Error") == "Movie not found!"


def _is_invalid_api_key(data: Mapping[str, object]) -> bool:
    err = data.get("Error")
    return (
        data.get("Response") == "False"
        and isinstance(err, str)
        and "api key" in err.lower()
    )


def _is_success_response(data: Mapping[str, object]) -> bool:
    return data.get("Response") == "True"


def _pick_ttl_and_status(omdb_data: Mapping[str, object]) -> tuple[int, CacheStatus]:
    if omdb_data.get("Response") == "False":
        return max(60, int(OMDB_CACHE_TTL_NOT_FOUND_SECONDS)), "not_found"
    if is_omdb_data_empty_for_ratings(omdb_data):
        return max(60, int(OMDB_CACHE_TTL_EMPTY_RATINGS_SECONDS)), "empty_ratings"
    return max(60, int(OMDB_CACHE_TTL_OK_SECONDS)), "ok"


# ============================================================
# WIKI MINIMAL (sanitize)
# ============================================================


def _sanitize_wiki_lookup(v: object) -> dict[str, object] | None:
    if not isinstance(v, Mapping):
        return None
    out: dict[str, object] = {}
    via = v.get("via")
    if isinstance(via, str) and via.strip():
        out["via"] = via.strip()

    imdb_id = v.get("imdb_id")
    if not (isinstance(imdb_id, str) and imdb_id.strip()):
        imdb_id = v.get("imdbID")  # compat legacy
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
    if not isinstance(v, Mapping):
        return None
    out: dict[str, object] = {}

    imdb_id = v.get("imdb_id")
    if not (isinstance(imdb_id, str) and imdb_id.strip()):
        imdb_id = v.get("imdbID")  # compat legacy ✅
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
    fetched_at = v.get("fetched_at")
    if (
        isinstance(fetched_at, int)
        and fetched_at > 0
        and not isinstance(fetched_at, bool)
    ):
        out["fetched_at"] = fetched_at
    ttl_s = v.get("ttl_s")
    if isinstance(ttl_s, int) and ttl_s > 0 and not isinstance(ttl_s, bool):
        out["ttl_s"] = ttl_s
    wiki_lookup = _sanitize_wiki_lookup(v.get("wiki_lookup"))
    if wiki_lookup is not None:
        out["wiki_lookup"] = wiki_lookup
    return out or None


# ============================================================
# PROVENANCE
# ============================================================


def _build_default_provenance(
    *, imdb_norm: str | None, norm_title: str, year: int | None
) -> dict[str, object]:
    if imdb_norm:
        lookup_key = f"imdb_id:{imdb_norm}"
    else:
        lookup_key = (
            f"title_year:{norm_title}|{year}"
            if year is not None
            else f"title:{norm_title}"
        )
    return {"lookup_key": lookup_key, "had_imdb_hint": bool(imdb_norm)}


def _merge_provenance(
    existing: Mapping[str, object] | None, incoming: Mapping[str, object] | None
) -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(existing, Mapping):
        out.update(dict(existing))
    if isinstance(incoming, Mapping):
        out.update(dict(incoming))
    return out


def _attach_provenance(
    omdb_data: dict[str, object], prov: Mapping[str, object] | None
) -> dict[str, object]:
    existing = omdb_data.get("__prov")
    merged = _merge_provenance(
        existing if isinstance(existing, Mapping) else None, prov
    )
    if merged:
        omdb_data["__prov"] = merged
    return omdb_data


def _merge_dict_shallow(
    dst: dict[str, object], patch: Mapping[str, object]
) -> dict[str, object]:
    for k, v in patch.items():
        if k == "__prov" and isinstance(v, Mapping):
            existing = dst.get("__prov")
            dst["__prov"] = _merge_provenance(
                existing if isinstance(existing, Mapping) else None, v
            )
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
# HOT CACHE (bounded in-memory) + miss TTL
# ============================================================


def _hot_get(key: str) -> object | None:
    if _HOT_CACHE_MAX <= 0:
        return None
    now = time.monotonic()
    with _HOT_CACHE_LOCK:
        v = _HOT_CACHE.get(key)
        if v is None:
            _m_inc("hot_cache_misses", 1)
            return None

        obj, _last, expires_at = v
        if expires_at > 0.0 and now >= expires_at:
            _HOT_CACHE.pop(key, None)
            _m_inc("hot_cache_misses", 1)
            return None

        _HOT_CACHE[key] = (obj, now, expires_at)
        _m_inc("hot_cache_hits", 1)
        return obj


def _hot_put(key: str, obj: object, *, ttl_s: float | None = None) -> None:
    if _HOT_CACHE_MAX <= 0:
        return
    now = time.monotonic()
    expires_at = 0.0
    if ttl_s is not None:
        expires_at = now + max(0.0, float(ttl_s))
    with _HOT_CACHE_LOCK:
        _HOT_CACHE[key] = (obj, now, expires_at)
        if len(_HOT_CACHE) <= _HOT_CACHE_MAX:
            return

        oldest_k: str | None = None
        oldest_ts = float("inf")
        for k, (_obj, ts, _exp) in _HOT_CACHE.items():
            if ts < oldest_ts:
                oldest_ts = ts
                oldest_k = k

        if oldest_k is not None:
            _HOT_CACHE.pop(oldest_k, None)
            _m_inc("hot_cache_evictions", 1)


def _is_cache_item_dict(obj: object) -> bool:
    """
    Valida el “shape” mínimo de OmdbCacheItem (para hot-cache).
    Evita devolver dicts arbitrarios como si fueran OmdbCacheItem.

    Nota: bool es subclass de int, así que lo excluimos explícitamente.
    """
    if not isinstance(obj, Mapping):
        return False
    try:
        status = obj.get("status")

        fetched_at = obj.get("fetched_at")
        ttl_s = obj.get("ttl_s")

        fetched_ok = isinstance(fetched_at, int) and not isinstance(fetched_at, bool)
        ttl_ok = isinstance(ttl_s, int) and not isinstance(ttl_s, bool)

        return (
            isinstance(obj.get("omdb"), Mapping)
            and isinstance(obj.get("Title"), str)
            and isinstance(obj.get("Year"), str)
            and fetched_ok
            and ttl_ok
            and status in ("ok", "not_found", "empty_ratings")
        )
    except Exception:
        return False


# ============================================================
# LOAD/SAVE CACHE (atómico + thread-safe) + compaction
# ============================================================


def _empty_cache() -> OmdbCacheFile:
    return {"schema": _SCHEMA_VERSION, "records": {}, "index_imdb": {}, "index_ty": {}}


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


def _load_cache_unlocked() -> OmdbCacheFile:
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

    if (
        not isinstance(records_obj, Mapping)
        or not isinstance(index_imdb_obj, Mapping)
        or not isinstance(index_ty_obj, Mapping)
    ):
        _CACHE = _empty_cache()
        return _CACHE

    records: dict[str, OmdbCacheItem] = {}
    for rid, v in records_obj.items():
        if isinstance(rid, str) and isinstance(v, Mapping):
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

            imdb_norm = (
                imdb_id.lower()
                if isinstance(imdb_id, str) and imdb_id.strip()
                else None
            )
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

    index_imdb = {k: rid for k, rid in index_imdb.items() if rid in records}
    index_ty = {k: rid for k, rid in index_ty.items() if rid in records}

    _CACHE = {
        "schema": _SCHEMA_VERSION,
        "records": records,
        "index_imdb": index_imdb,
        "index_ty": index_ty,
    }

    try:
        _compact_cache_unlocked(_CACHE, force=False)
    except Exception:
        pass

    return _CACHE


def _save_cache_file_atomic(cache: OmdbCacheFile) -> None:
    dirpath = _CACHE_PATH.parent
    dirpath.mkdir(parents=True, exist_ok=True)

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=str(dirpath)
        ) as tf:
            if _CACHE_JSON_PRETTY:
                json.dump(cache, tf, ensure_ascii=False, indent=_CACHE_JSON_INDENT)
            else:
                json.dump(cache, tf, ensure_ascii=False, separators=(",", ":"))
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
    global _CACHE_DIRTY, _CACHE_DIRTY_WRITES
    _CACHE_DIRTY = True
    _CACHE_DIRTY_WRITES += 1


def _compact_cache_unlocked(cache: OmdbCacheFile, *, force: bool) -> None:
    """
    Compaction: purga expirados, normaliza imdbID, reconstruye índices y recorta tamaños.

    Nota: evitamos `continue` en guard clauses (reduce falsos "unreachable" con stubs NoReturn).
    """
    try:
        now_epoch = _now_epoch()
        records_in: dict[str, OmdbCacheItem] = cache["records"]

        records: dict[str, OmdbCacheItem] = {}
        for rid, it in records_in.items():
            fetched_at = it.get("fetched_at")
            ttl_s = it.get("ttl_s")

            ok_types = isinstance(fetched_at, int) and isinstance(ttl_s, int)
            keep = ok_types and (not _is_expired_item(it, now_epoch))
            if keep:
                d = dict(it)
                imdb_norm = d.get("imdbID")
                if isinstance(imdb_norm, str):
                    imdb_norm2 = imdb_norm.strip().lower()
                    d["imdbID"] = imdb_norm2 or None
                else:
                    d["imdbID"] = None

                records[rid] = d  # type: ignore[assignment]

        if _COMPACT_MAX_RECORDS > 0 and len(records) > _COMPACT_MAX_RECORDS:
            ranked = sorted(
                records.items(),
                key=lambda kv: _int_default(kv[1].get("fetched_at"), 0),
                reverse=True,
            )
            records = dict(ranked[:_COMPACT_MAX_RECORDS])

        cache["records"] = records

        index_imdb: dict[str, str] = {}
        index_ty: dict[str, str] = {}
        for rid, it in records.items():
            title = it.get("Title")
            year = it.get("Year")

            # ✅ FIX: solo index_ty si title+year no están vacíos
            if (
                isinstance(title, str)
                and isinstance(year, str)
                and title.strip()
                and year.strip()
            ):
                index_ty[_ty_key(title, year)] = rid

            imdb = it.get("imdbID")
            if isinstance(imdb, str) and imdb.strip():
                index_imdb[imdb.strip().lower()] = rid

        if _COMPACT_MAX_INDEX_IMDB > 0 and len(index_imdb) > _COMPACT_MAX_INDEX_IMDB:
            keep_keys = sorted(index_imdb.keys())[:_COMPACT_MAX_INDEX_IMDB]
            index_imdb = {k: index_imdb[k] for k in keep_keys}

        if _COMPACT_MAX_INDEX_TY > 0 and len(index_ty) > _COMPACT_MAX_INDEX_TY:
            keep_keys2 = sorted(index_ty.keys())[:_COMPACT_MAX_INDEX_TY]
            index_ty = {k: index_ty[k] for k in keep_keys2}

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
    global _CACHE_DIRTY, _CACHE_DIRTY_WRITES, _CACHE_LAST_FLUSH_TS, _CACHE, _FLUSH_COUNT
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

    _FLUSH_COUNT += 1

    do_full_compact = True
    if not force and _CACHE_COMPACT_EVERY_N_FLUSHES > 0:
        do_full_compact = (_FLUSH_COUNT % _CACHE_COMPACT_EVERY_N_FLUSHES) == 0

    _compact_cache_unlocked(_CACHE, force=do_full_compact or force)
    _save_cache_file_atomic(_CACHE)

    _CACHE_DIRTY = False
    _CACHE_DIRTY_WRITES = 0
    _CACHE_LAST_FLUSH_TS = now
    _m_inc("cache_flush_writes", 1)


def flush_omdb_cache() -> None:
    with _CACHE_LOCK:
        _maybe_flush_unlocked(force=True)


def _flush_cache_on_exit() -> None:
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

_FALLBACK_IMDB_TITLE_SCORE_MIN: Final[float] = 0.5
_FALLBACK_IMDB_INDEX_MAX_TOKENS_PER_TITLE: Final[int] = 6
_FALLBACK_IMDB_INDEX_MAX_TOKENS_PER_QUERY: Final[int] = 2

_FALLBACK_IMDB_INDEX: dict[str, set[str]] = {}
_FALLBACK_IMDB_INDEX_RECORDS: int = 0


def _invalidate_fallback_imdb_index_unlocked() -> None:
    global _FALLBACK_IMDB_INDEX, _FALLBACK_IMDB_INDEX_RECORDS
    _FALLBACK_IMDB_INDEX = {}
    _FALLBACK_IMDB_INDEX_RECORDS = -1


def _ensure_fallback_imdb_index_unlocked(
    cache: OmdbCacheFile,
) -> dict[str, set[str]] | None:
    global _FALLBACK_IMDB_INDEX, _FALLBACK_IMDB_INDEX_RECORDS

    records_obj: object = cache.get("records")
    idx_imdb_obj: object = cache.get("index_imdb")
    if not isinstance(records_obj, Mapping) or not isinstance(idx_imdb_obj, Mapping):
        return None

    records: dict[str, object] = cast(dict[str, object], records_obj)
    idx_imdb: Mapping[str, object] = cast(Mapping[str, object], idx_imdb_obj)

    rec_count = len(records)
    if _FALLBACK_IMDB_INDEX and _FALLBACK_IMDB_INDEX_RECORDS == rec_count:
        return _FALLBACK_IMDB_INDEX

    index: dict[str, set[str]] = {}
    for rid in idx_imdb.values():
        if not isinstance(rid, str):
            continue
        raw = records.get(rid)
        if not isinstance(raw, Mapping):
            continue
        title_raw = raw.get("Title")
        if not isinstance(title_raw, str) or not title_raw.strip():
            continue
        title_norm = normalize_title_for_lookup(title_raw)
        if not title_norm:
            continue

        tokens = sorted(_tokenize_norm_title(title_norm), key=len, reverse=True)
        if _FALLBACK_IMDB_INDEX_MAX_TOKENS_PER_TITLE > 0:
            tokens = tokens[:_FALLBACK_IMDB_INDEX_MAX_TOKENS_PER_TITLE]

        for tok in tokens:
            bucket = index.get(tok)
            if bucket is None:
                bucket = set()
                index[tok] = bucket
            bucket.add(rid)

    _FALLBACK_IMDB_INDEX = index
    _FALLBACK_IMDB_INDEX_RECORDS = rec_count
    return _FALLBACK_IMDB_INDEX


def _score_title_match(norm_title: str, cached_title: str) -> float:
    if not norm_title or not cached_title:
        return 0.0
    if norm_title == cached_title:
        return 1.0
    if norm_title in cached_title or cached_title in norm_title:
        return 0.9
    ntoks = _tokenize_norm_title(norm_title)
    ctoks = _tokenize_norm_title(cached_title)
    if not ntoks or not ctoks:
        return 0.0
    inter = len(ntoks & ctoks)
    union = len(ntoks | ctoks)
    return (inter / union) if union > 0 else 0.0


def _find_cached_imdb_fallback_unlocked(
    *, cache: OmdbCacheFile, norm_title: str, norm_year: str
) -> OmdbCacheItem | None:
    if not norm_title:
        return None

    records_obj: object = cache.get("records")
    idx_imdb_obj: object = cache.get("index_imdb")
    if not isinstance(records_obj, Mapping) or not isinstance(idx_imdb_obj, Mapping):
        return None

    records: dict[str, object] = cast(dict[str, object], records_obj)
    idx_imdb: Mapping[str, object] = cast(Mapping[str, object], idx_imdb_obj)

    candidate_rids: Iterable[str] | None = None
    index = _ensure_fallback_imdb_index_unlocked(cache)
    if index is not None:
        tokens = sorted(_tokenize_norm_title(norm_title), key=len, reverse=True)
        if tokens and _FALLBACK_IMDB_INDEX_MAX_TOKENS_PER_QUERY > 0:
            tokens = tokens[:_FALLBACK_IMDB_INDEX_MAX_TOKENS_PER_QUERY]
        if tokens:
            sets: list[set[str]] = []
            for tok in tokens:
                bucket = index.get(tok)
                if bucket:
                    sets.append(bucket)
            if sets:
                merged = set(sets[0])
                for s in sets[1:]:
                    merged &= s
                if not merged and tokens:
                    merged = set()
                    for tok in tokens:
                        bucket = index.get(tok)
                        if bucket:
                            merged.update(bucket)
                if merged:
                    candidate_rids = merged

    now_epoch = _now_epoch()
    best: OmdbCacheItem | None = None
    best_score = 0.0

    for rid in candidate_rids or idx_imdb.values():
        if not isinstance(rid, str):
            continue
        raw = records.get(rid)
        if not isinstance(raw, Mapping):
            continue
        item = cast(OmdbCacheItem, raw)
        if _is_expired_item(item, now_epoch):
            continue
        status = item.get("status")
        if status not in ("ok", "empty_ratings"):
            continue
        title_raw = item.get("Title")
        if not isinstance(title_raw, str) or not title_raw.strip():
            continue
        year_raw = item.get("Year")
        if norm_year and isinstance(year_raw, str) and year_raw.strip():
            if year_raw.strip() != norm_year:
                continue

        title_norm = normalize_title_for_lookup(title_raw) or title_raw.strip()
        score = _score_title_match(norm_title, title_norm)
        if score > best_score:
            best_score = score
            best = item

    if best is None or best_score < _FALLBACK_IMDB_TITLE_SCORE_MIN:
        return None

    return best

def _get_cached_item_unlocked(
    *, norm_title: str, norm_year: str, imdb_id_hint: str | None
) -> OmdbCacheItem | None:
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

    # ✅ FIX: solo intentamos TY si ambos están presentes
    if norm_title and norm_year:
        rid2 = idx_ty.get(_ty_key(norm_title, norm_year))
        if isinstance(rid2, str):
            it2 = records.get(rid2)
            if it2 is not None:
                if _is_expired_item(it2, now_epoch):
                    _m_inc("cache_expired_hits", 1)
                    return None
                if imdb_id_hint is None and it2.get("status") == "not_found":
                    alt = _find_cached_imdb_fallback_unlocked(
                        cache=cache, norm_title=norm_title, norm_year=norm_year
                    )
                    if alt is not None:
                        _m_inc("cache_fallback_imdb_hits", 1)
                        return alt
                return it2

    return None


def _get_cached_item(
    *, norm_title: str, norm_year: str, imdb_id_hint: str | None
) -> OmdbCacheItem | None:
    imdb_key = _cache_key_for_imdb(imdb_id_hint) if imdb_id_hint else None
    ty_key = (
        _cache_key_for_title_year(norm_title, norm_year)
        if (norm_title and norm_year)
        else None
    )

    if imdb_key:
        h = _hot_get(imdb_key)
        if h is _HOT_MISS:
            return None
        if _is_cache_item_dict(h):
            return cast(OmdbCacheItem, h)

    if ty_key:
        h2 = _hot_get(ty_key)
        if h2 is _HOT_MISS:
            return None
        if _is_cache_item_dict(h2):
            return cast(OmdbCacheItem, h2)

    with _CACHE_LOCK:
        it = _get_cached_item_unlocked(
            norm_title=norm_title, norm_year=norm_year, imdb_id_hint=imdb_id_hint
        )

    if it is not None:
        it_copy: OmdbCacheItem = cast(
            OmdbCacheItem, dict(it)
        )  # ✅ evita aliasing con records

        if imdb_key and isinstance(it_copy.get("imdbID"), str):
            _hot_put(imdb_key, it_copy)
        if (
            ty_key
            and isinstance(it_copy.get("Title"), str)
            and isinstance(it_copy.get("Year"), str)
        ):
            _hot_put(ty_key, it_copy)

        return it_copy

    miss_ttl = _HOT_MISS_TTL_S if _HOT_MISS_TTL_S > 0.0 else None
    if imdb_key:
        _hot_put(imdb_key, _HOT_MISS, ttl_s=miss_ttl)
    if ty_key:
        _hot_put(ty_key, _HOT_MISS, ttl_s=miss_ttl)

    return None


def _cache_store_item(
    *,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
    omdb_data: dict[str, object],
) -> OmdbCacheItem:
    now_epoch = _now_epoch()
    ttl_s, status = _pick_ttl_and_status(omdb_data)

    imdb_norm = (
        imdb_id.lower() if isinstance(imdb_id, str) and imdb_id.strip() else None
    )

    # Norm final (puede venir vacío si Response=False o payload raro)
    norm_title_final = (
        norm_title
        or normalize_title_for_lookup(str(omdb_data.get("Title") or ""))
        or ""
    )
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

    rid = _rid_for_record(
        imdb_norm=imdb_norm, norm_title=norm_title_final, norm_year=norm_year_final
    )

    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        cache["records"][rid] = item

        # ✅ FIX: solo index_ty si title+year no están vacíos (evita key "|")
        if norm_title_final and norm_year_final:
            cache["index_ty"][_ty_key(norm_title_final, norm_year_final)] = rid

        if imdb_norm:
            cache["index_imdb"][imdb_norm] = rid

        _m_inc("cache_store_writes", 1)
        _invalidate_fallback_imdb_index_unlocked()
        _mark_dirty_unlocked()
        _maybe_flush_unlocked(force=False)

    item_copy: OmdbCacheItem = cast(OmdbCacheItem, dict(item))

    # ✅ FIX: hot-cache ty/t solo si hay title/year/title
    if norm_title_final and norm_year_final:
        _hot_put(
            _cache_key_for_title_year(norm_title_final, norm_year_final), item_copy
        )
    if norm_title_final:
        _hot_put(_cache_key_for_title_only(norm_title_final), item_copy)
    if imdb_norm:
        _hot_put(_cache_key_for_imdb(imdb_norm), item_copy)

    return item


def iter_cached_omdb_records() -> Iterable[dict[str, object]]:
    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        records_copy = list(cache["records"].values())

    for it in records_copy:
        yield dict(it["omdb"])


# ============================================================
# PATCH / WRITE-BACK AL CACHE
# ============================================================


def patch_cached_omdb_record(
    *, norm_title: str, norm_year: str, imdb_id: str | None, patch: Mapping[str, object]
) -> bool:
    imdb_norm = _safe_imdb_id(imdb_id) if imdb_id else None
    ty_k = _ty_key(norm_title, norm_year)

    invalidate_prev_imdb: str | None = None
    invalidate_prev_ty: tuple[str, str] | None = None

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

        # Estado previo (para limpiar índices y hot-cache)
        prev_imdb_key = (
            _safe_imdb_id(target.get("imdbID")) if target.get("imdbID") else None
        )
        prev_title = target.get("Title") if isinstance(target.get("Title"), str) else ""
        prev_year = target.get("Year") if isinstance(target.get("Year"), str) else ""
        prev_ty_key = (
            _ty_key(prev_title, prev_year)
            if (prev_title.strip() and prev_year.strip())
            else ""
        )

        omdb_obj = target["omdb"]
        _merge_dict_shallow(omdb_obj, patch)

        ttl_s, status = _pick_ttl_and_status(omdb_obj)
        target["ttl_s"] = int(ttl_s)
        target["status"] = status
        target["fetched_at"] = _now_epoch()

        imdb_now = _safe_imdb_id(target.get("imdbID")) if target.get("imdbID") else None
        title_now = (
            target.get("Title") if isinstance(target.get("Title"), str) else norm_title
        )
        year_now = (
            target.get("Year") if isinstance(target.get("Year"), str) else norm_year
        )
        new_ty_key = (
            _ty_key(title_now, year_now)
            if (title_now.strip() and year_now.strip())
            else ""
        )

        old_rid = rid

        # Decide new_rid, evitando ty:|
        new_rid = old_rid
        if imdb_now:
            new_rid = _rid_for_record(
                imdb_norm=imdb_now, norm_title=title_now, norm_year=year_now
            )
        elif title_now.strip() and year_now.strip():
            new_rid = _rid_for_record(
                imdb_norm=None, norm_title=title_now, norm_year=year_now
            )

        # Re-key si aplica
        if new_rid != old_rid:
            records[new_rid] = target
            records.pop(old_rid, None)
            rid = new_rid

        # Limpieza de índices viejos (aunque no cambie rid)
        if (
            prev_imdb_key
            and cache["index_imdb"].get(prev_imdb_key) == old_rid
            and prev_imdb_key != imdb_now
        ):
            cache["index_imdb"].pop(prev_imdb_key, None)
            invalidate_prev_imdb = prev_imdb_key  # invalidar hot-cache fuera del lock

        if (
            prev_ty_key
            and cache["index_ty"].get(prev_ty_key) == old_rid
            and prev_ty_key != new_ty_key
        ):
            cache["index_ty"].pop(prev_ty_key, None)
            invalidate_prev_ty = (
                prev_title.strip(),
                prev_year.strip(),
            )  # invalidar hot-cache fuera del lock

        # Insertar índices actuales (una sola vez)
        if new_ty_key:
            cache["index_ty"][new_ty_key] = rid
        if imdb_now:
            cache["index_imdb"][imdb_now] = rid

        _m_inc("cache_patch_writes", 1)
        _invalidate_fallback_imdb_index_unlocked()
        _mark_dirty_unlocked()
        _maybe_flush_unlocked(force=False)

    target_copy: OmdbCacheItem = cast(OmdbCacheItem, dict(target))

    # Hot-cache: write-through nuevas claves
    if new_ty_key:
        _hot_put(_cache_key_for_title_year(title_now, year_now), target_copy)
    if isinstance(title_now, str) and title_now.strip():
        _hot_put(_cache_key_for_title_only(title_now), target_copy)
    if imdb_norm:
        _hot_put(_cache_key_for_imdb(imdb_norm), target_copy)
    if imdb_now:
        _hot_put(_cache_key_for_imdb(imdb_now), target_copy)

    # Hot-cache: invalidaciones fuera del lock (orden de locks consistente)
    miss_ttl = _HOT_MISS_TTL_S if _HOT_MISS_TTL_S > 0.0 else 5.0

    if invalidate_prev_imdb:
        _hot_put(_cache_key_for_imdb(invalidate_prev_imdb), _HOT_MISS, ttl_s=miss_ttl)

    if invalidate_prev_ty:
        pt, py = invalidate_prev_ty
        if pt and py:
            _hot_put(_cache_key_for_title_year(pt, py), _HOT_MISS, ttl_s=miss_ttl)

    return True


# ============================================================
# PETICIONES OMDb (red)
# ============================================================


def omdb_request(
    params: Mapping[str, object], *, api_key: str | None = None
) -> dict[str, object] | None:
    global OMDB_INVALID_KEY_NOTICE_SHOWN, OMDB_DISABLED, OMDB_DISABLED_NOTICE_SHOWN

    if OMDB_DISABLED:
        return None

    keys = _configured_keys()
    if not keys:
        if not OMDB_DISABLED:
            OMDB_DISABLED = True
            _m_inc("disabled_switches", 1)
        if not OMDB_DISABLED_NOTICE_SHOWN:
            _warn_always(
                "ERROR: OMDb sin API key configurada. OMDb queda desactivado; se usará solo caché."
            )
            OMDB_DISABLED_NOTICE_SHOWN = True
        return None

    use_key = (api_key or (keys[0] if keys else "")).strip()
    if not use_key:
        return None

    base_url = str(OMDB_BASE_URL).strip() or "https://www.omdbapi.com/"
    req_params: dict[str, str] = {str(k): str(v) for k, v in params.items()}
    req_params["apikey"] = use_key

    timeout_s = _cap_float_runtime(
        float(OMDB_HTTP_TIMEOUT_SECONDS), min_v=0.5, max_v=120.0
    )

    resp = _omdb_http_get(
        base_url=base_url, params=req_params, timeout_seconds=timeout_s
    )
    if resp is None:
        _mark_omdb_failure()
        return None

    if resp.status_code != 200:
        _m_inc("http_failures", 1)
        _dbg(f"OMDb status != 200: {resp.status_code}")
        _cb_record_failure()
        _mark_omdb_failure()
        return None

    try:
        data_obj = resp.json()
    except (ValueError, json.JSONDecodeError) as exc:
        _m_inc("http_failures", 1)
        _dbg(f"OMDb invalid JSON: {exc!r}")
        _cb_record_failure()
        _mark_omdb_failure()
        return None

    if not isinstance(data_obj, dict):
        _m_inc("http_failures", 1)
        _dbg("OMDb returned JSON not dict.")
        _cb_record_failure()
        _mark_omdb_failure()
        return None

    if _is_invalid_api_key(data_obj):
        _mark_key_invalid(use_key)
        if not OMDB_INVALID_KEY_NOTICE_SHOWN:
            _warn_always(
                "ERROR: OMDb respondió 'Invalid API key!'. Revisa OMDB_API_KEY/OMDB_API_KEYS."
            )
            OMDB_INVALID_KEY_NOTICE_SHOWN = True

        if not bool(OMDB_KEY_ROTATION_ENABLED) or _pick_next_key(keys) is None:
            if not OMDB_DISABLED:
                OMDB_DISABLED = True
                _m_inc("disabled_switches", 1)
            _cb_record_failure()
            _mark_omdb_failure()

        return data_obj

    _cb_record_success()
    _mark_omdb_success()
    return data_obj


def _is_rate_limit_response(data: Mapping[str, object]) -> bool:
    return data.get("Error") == "Request limit reached!"


def _request_with_rate_limit(params: Mapping[str, object]) -> dict[str, object] | None:
    global OMDB_RATE_LIMIT_NOTICE_SHOWN, OMDB_DISABLED, OMDB_DISABLED_NOTICE_SHOWN

    if OMDB_DISABLED:
        return None

    keys = _configured_keys()
    if not keys:
        return None

    rotation_enabled = bool(OMDB_KEY_ROTATION_ENABLED) and len(keys) > 1
    tries_cap = _rotation_tries_cap(len(keys)) if rotation_enabled else 1

    max_retries = max(0, int(OMDB_RATE_LIMIT_MAX_RETRIES))
    wait_s = max(0.0, float(OMDB_RATE_LIMIT_WAIT_SECONDS))

    last_data: dict[str, object] | None = None

    for _ in range(max(1, tries_cap)):
        api_key = _pick_next_key(keys) if rotation_enabled else keys[0]
        if api_key is None:
            break

        retries_local = 0
        while retries_local <= max_retries:
            data = omdb_request(params, api_key=api_key)
            if data is None:
                return None

            last_data = data

            if _is_rate_limit_response(data):
                _m_inc("rate_limit_hits", 1)
                _mark_key_exhausted(api_key)

                if not OMDB_RATE_LIMIT_NOTICE_SHOWN:
                    _warn_always(
                        "AVISO: límite de llamadas gratuitas de OMDb alcanzado para una API key. "
                        "Probando con la siguiente key disponible..."
                    )
                    OMDB_RATE_LIMIT_NOTICE_SHOWN = True

                if wait_s > 0.0:
                    _m_inc("rate_limit_sleeps", 1)
                    slept = _sleep_with_jitter(
                        wait_s, reason_metric_ms_total="rate_limit_sleep_ms_total"
                    )
                    _rm_observe_ms("omdb.http.rate_limit_sleep_ms", int(slept * 1000.0))

                break

            return data

    if not OMDB_DISABLED:
        OMDB_DISABLED = True
        _m_inc("disabled_switches", 1)
    if not OMDB_DISABLED_NOTICE_SHOWN:
        _warn_always(
            "AVISO: todas las OMDb API keys configuradas están agotadas o inválidas. "
            "OMDb queda desactivado para esta ejecución; se usará solo caché."
        )
        OMDB_DISABLED_NOTICE_SHOWN = True

    return last_data


# ============================================================
# CANDIDATE SEARCH (fallback cuando t= falla)
# ============================================================


def _tokenize_norm_title(s: str) -> set[str]:
    out: set[str] = set()
    for tok in s.replace("-", " ").replace("_", " ").split():
        t = tok.strip().lower()
        if len(t) >= 2:
            out.add(t)
    return out


def _search_candidates_imdb_id(
    *, title_for_search: str, year: int | None
) -> str | None:
    _m_inc("candidate_search_calls", 1)

    title_q = title_for_search.strip()
    if not title_q or OMDB_DISABLED:
        return None
    if not _configured_keys():
        return None

    data_s = _request_with_rate_limit({"s": title_q, "type": "movie"})
    if not isinstance(data_s, dict) or data_s.get("Response") != "True":
        return None

    results_obj = data_s.get("Search") or []
    if not isinstance(results_obj, list):
        return None

    ptit = normalize_title_for_lookup(title_q)
    ptoks = _tokenize_norm_title(ptit)

    def score_candidate(cand: Mapping[str, object]) -> float:
        score = 0.0

        ctype = cand.get("Type")
        if isinstance(ctype, str) and ctype and ctype.lower() != "movie":
            score -= 5.0

        ct_raw = cand.get("Title")
        ct = normalize_title_for_lookup(ct_raw) if isinstance(ct_raw, str) else ""
        if ptit and ct:
            if ptit == ct:
                score += 3.0
            elif ct in ptit or ptit in ct:
                score += 1.5

        if ptoks and ct:
            ctoks = _tokenize_norm_title(ct)
            if ctoks:
                inter = len(ptoks & ctoks)
                union = len(ptoks | ctoks)
                if union > 0:
                    score += 2.0 * (inter / union)

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
            else:
                d = abs(year - cand_year)
                if d == 1:
                    score += 0.75
                else:
                    score -= min(2.0, float(d) * 0.25)

        return score

    best_imdb: str | None = None
    best_score = float("-inf")

    for item in results_obj:
        if isinstance(item, Mapping):
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


def _singleflight_keys_for_request(
    *, norm_title: str, year_str: str, imdb_norm: str | None
) -> list[str]:
    keys: list[str] = []
    if imdb_norm:
        keys.append(f"imdb:{imdb_norm}")
    if norm_title and year_str:
        keys.append(f"ty:{_ty_key(norm_title, year_str)}")
    if norm_title:
        keys.append(_cache_key_for_title_only(norm_title))

    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _is_cacheable_omdb_payload(data: Mapping[str, object]) -> bool:
    if _is_success_response(data):
        return True
    return _is_movie_not_found(data)


def omdb_query_with_cache(
    *,
    title: str | None,
    year: int | None,
    imdb_id: str | None,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    imdb_norm = _safe_imdb_id(imdb_id) if imdb_id else None
    year_str = _norm_year_str(year)

    norm_title = normalize_title_for_lookup(title or "")
    title_query = (title or "").strip()

    if not norm_title and imdb_norm is None:
        return None

    prov_base = _build_default_provenance(
        imdb_norm=imdb_norm, norm_title=norm_title, year=year
    )
    prov_in = _merge_provenance(prov_base, provenance)

    cached = _get_cached_item(
        norm_title=norm_title, norm_year=year_str, imdb_id_hint=imdb_norm
    )
    if cached is not None:
        _m_inc("cache_hits", 1)
        out = dict(cached["omdb"])
        _attach_provenance(out, prov_in)
        return out

    if OMDB_DISABLED:
        return None

    _m_inc("cache_misses", 1)

    keys = _singleflight_keys_for_request(
        norm_title=norm_title, year_str=year_str, imdb_norm=imdb_norm
    )
    is_leader, ev, primary = _singleflight_enter_multi(keys)
    if not is_leader:
        _m_inc("singleflight_waits", 1)
        t0w = time.monotonic()
        ev.wait(timeout=float(_SINGLEFLIGHT_WAIT_S))
        waited_ms = int((time.monotonic() - t0w) * 1000.0)
        if waited_ms > 0:
            _m_add_ms("singleflight_wait_ms_total", waited_ms)
            _rm_observe_ms("omdb.singleflight.wait_ms", waited_ms)

        cached2 = _get_cached_item(
            norm_title=norm_title, norm_year=year_str, imdb_id_hint=imdb_norm
        )
        if cached2 is not None:
            _m_inc("cache_hits", 1)
            out2 = dict(cached2["omdb"])
            _attach_provenance(out2, prov_in)
            return out2

    try:
        if imdb_norm is not None:
            data_main = _request_with_rate_limit(
                {"i": imdb_norm, "type": "movie", "plot": "short"}
            )
            if not isinstance(data_main, dict):
                return None

            if not _is_cacheable_omdb_payload(data_main):
                return None

            # ✅ FIX CRÍTICO: si Response=True y el imdbID no coincide, RECHAZAR (no contaminar)
            imdb_from_resp = _extract_imdb_id_from_omdb_record(data_main)
            if (
                _is_success_response(data_main)
                and imdb_from_resp
                and imdb_from_resp != imdb_norm
            ):
                _warn_always(
                    "OMDB: IMDb mismatch on i= lookup (rejecting payload) | "
                    f"hint={imdb_norm} resp={imdb_from_resp} title={data_main.get('Title')!r} year={data_main.get('Year')!r}"
                )
                # Importante: NO cachear este payload
                return None

            imdb_final = imdb_from_resp or imdb_norm

            # Completar title/year para indexado si hace falta
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

            _cache_store_item(
                norm_title=norm_title,
                norm_year=year_str,
                imdb_id=imdb_final,
                omdb_data=dict(data_main),
            )
            return dict(data_main)

        title_for_omdb = title_query or norm_title
        params_t: dict[str, object] = {
            "t": title_for_omdb,
            "type": "movie",
            "plot": "short",
        }
        if year is not None:
            params_t["y"] = str(year)

        def _try_title_variants() -> dict[str, object] | None:
            variants: list[str] = []
            seen: set[str] = set()
            for src in (title_for_omdb, norm_title):
                if not src:
                    continue
                for variant in generate_sequel_title_variants(src):
                    v = variant.strip()
                    if not v:
                        continue
                    key = v.lower()
                    if key == title_for_omdb.lower():
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    variants.append(v)

            for variant in variants:
                params_var: dict[str, object] = {
                    "t": variant,
                    "type": "movie",
                    "plot": "short",
                }
                if year is not None:
                    params_var["y"] = str(year)

                data_var = _request_with_rate_limit(params_var)
                if not isinstance(data_var, dict):
                    continue

                used_no_year_var = False
                if year is not None and _is_movie_not_found(data_var):
                    params_no_year = dict(params_var)
                    params_no_year.pop("y", None)
                    data_no_year = _request_with_rate_limit(params_no_year)
                    if isinstance(data_no_year, dict):
                        data_var = data_no_year
                        used_no_year_var = True

                if _is_movie_not_found(data_var):
                    continue

                if not _is_cacheable_omdb_payload(data_var):
                    continue

                imdb_final = _extract_imdb_id_from_omdb_record(data_var)
                y_resp = extract_year_from_omdb(data_var)
                year_str_var = year_str or (str(y_resp) if y_resp is not None else "")

                prov_final = dict(prov_in)
                prov_final["resolved_via"] = (
                    "t_var_y"
                    if (year is not None and not used_no_year_var)
                    else "t_var"
                )
                _attach_provenance(data_var, prov_final)

                norm_title_var = norm_title or normalize_title_for_lookup(variant)
                _cache_store_item(
                    norm_title=norm_title_var,
                    norm_year=year_str_var,
                    imdb_id=imdb_final,
                    omdb_data=dict(data_var),
                )
                return dict(data_var)

            return None

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
            data_variant = _try_title_variants()
            if data_variant is not None:
                return data_variant

            imdb_best = _search_candidates_imdb_id(
                title_for_search=title_for_omdb, year=year
            )
            if imdb_best:
                data_full = _request_with_rate_limit(
                    {"i": imdb_best, "type": "movie", "plot": "short"}
                )
                if isinstance(data_full, dict):
                    if not _is_cacheable_omdb_payload(data_full):
                        return None

                    imdb_from_full = _extract_imdb_id_from_omdb_record(data_full)
                    imdb_final2 = imdb_from_full or imdb_best

                    y_resp2 = extract_year_from_omdb(data_full)
                    year_str2 = year_str or (
                        str(y_resp2) if y_resp2 is not None else ""
                    )

                    prov_final2 = dict(prov_in)
                    prov_final2["resolved_via"] = "s_i"
                    _attach_provenance(data_full, prov_final2)

                    _cache_store_item(
                        norm_title=norm_title,
                        norm_year=year_str2,
                        imdb_id=imdb_final2,
                        omdb_data=dict(data_full),
                    )
                    return dict(data_full)

        if not _is_cacheable_omdb_payload(data_t):
            return None

        imdb_final3 = _extract_imdb_id_from_omdb_record(data_t)
        y_resp3 = extract_year_from_omdb(data_t)
        year_str3 = year_str or (str(y_resp3) if y_resp3 is not None else "")

        prov_final3 = dict(prov_in)
        prov_final3["resolved_via"] = (
            "t_y" if (year is not None and not used_no_year) else "t"
        )
        _attach_provenance(data_t, prov_final3)

        _cache_store_item(
            norm_title=norm_title,
            norm_year=year_str3,
            imdb_id=imdb_final3,
            omdb_data=dict(data_t),
        )
        return dict(data_t)

    finally:
        if is_leader:
            _singleflight_leave(primary)


# ============================================================
# FUNCIONES PÚBLICAS (helpers)
# ============================================================


def search_omdb_by_imdb_id(imdb_id: str) -> dict[str, object] | None:
    imdb_norm = _safe_imdb_id(imdb_id)
    if not imdb_norm:
        return None
    return omdb_query_with_cache(title=None, year=None, imdb_id=imdb_norm)


def search_omdb_by_title_and_year(
    title: str, year: int | None
) -> dict[str, object] | None:
    if not title.strip():
        return None
    return omdb_query_with_cache(title=title, year=year, imdb_id=None)


def search_omdb_with_candidates(
    plex_title: str, plex_year: int | None
) -> dict[str, object] | None:
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


# ============================================================
# API PÚBLICA / COMPAT
# ============================================================

omdb_query = omdb_query_with_cache

__all__ = [
    "OMDB_DISABLED",
    "omdb_request",
    "omdb_query_with_cache",
    "omdb_query",
    "patch_cached_omdb_record",
    "iter_cached_omdb_records",
    "flush_omdb_cache",
    "extract_ratings_from_omdb",
    "is_omdb_data_empty_for_ratings",
    "parse_imdb_rating_from_omdb",
    "parse_rt_score_from_omdb",
    "normalize_imdb_votes",
    "extract_year_from_omdb",
    "get_omdb_metrics_snapshot",
    "reset_omdb_metrics",
    "log_omdb_metrics_summary",
    "search_omdb_by_imdb_id",
    "search_omdb_by_title_and_year",
    "search_omdb_with_candidates",
]
