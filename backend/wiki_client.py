"""
backend/wiki_client.py (schema v6) - refactor (client-only) - OPTIMIZED

Cliente best-effort para enriquecer títulos con Wikipedia/Wikidata con caché persistente.

✅ API pública (estable):
  - get_wiki(title, year, imdb_id) -> WikiCacheItem | None
  - get_wiki_for_input(movie_input, title, year, imdb_id) -> WikiCacheItem | None
  - flush_wiki_cache() -> None
  - métricas: get_wiki_metrics_snapshot / reset_wiki_metrics / log_wiki_metrics_summary

✅ Diseño (resumen):
  - Caché persistente en JSON (schema v6) con escritura atómica.
  - Índices por imdbID y por (Title|Year).
  - Single-flight (por request-key) para evitar duplicación concurrente.
  - SWR (stale-while-revalidate): ok expirado puede servirse durante grace window y
    se agenda refresh en background.
  - Circuit breaker suave para:
      * "wiki" (Wikipedia REST/API + Wikidata API)
      * "wdqs" (Wikidata Query Service / SPARQL)
  - Throttle global para SPARQL.
  - JSON cache configurable via config_wiki.py.

🚫 NO hace:
  - Orquestación de run/batch, progreso, pools, etc. (eso va en analiza_wiki.py)

Política de logs (alineado con backend/logger.py):
  - Debug contextual: logger.debug_ctx("WIKI", ...)
  - Info sólo si NO silent (y opcionalmente deshabilitable con WIKI_CLIENT_DISABLE_INFO_LOGS).
  - Errores: logger.error(..., always=True) (pero evitamos “spam”; este cliente es best-effort)

OPTIMIZACIONES CLAVE (vs. versión anterior):
  1) Menor contención de locks:
     - Nunca se mantiene _CACHE_LOCK durante red.
     - Lecturas/escrituras de caché agrupadas y con helpers pequeños.
  2) Flush “cheap”:
     - _maybe_flush_unlocked hace return rápido si no toca flush.
     - Se evita compaction agresiva salvo cuando toca flush.
  3) Ranking de búsqueda más eficiente:
     - Dedupe de queries, scoring mejorado, y menos llamadas redundantes.
  4) Single-flight real en force_refresh:
     - Si no eres líder, esperas, re-reads cache y sales.
  5) Robustez:
     - Quarantine de cache corrupto en debug.
     - Circuit breaker y métricas coherentes.
     - Normalización consistente de imdb/lang.

NOTA:
  - Este archivo está pensado para pegarse ENTERO (para evitar truncados).

FIX (typing / Pyright):
  - Evita kwargs en dict-subclasses y reconstruye dicts con cast + loops.
"""

from __future__ import annotations

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
from typing import Any, Final, Literal, Protocol, cast
from urllib.parse import quote

import requests  # type: ignore[import-untyped]
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from requests.exceptions import RequestException  # type: ignore[import-untyped]
from urllib3.util.retry import Retry

from backend import logger
from backend.config_wiki import (
    ANALIZA_WIKI_CACHE_MAX_ENTITIES,
    ANALIZA_WIKI_CACHE_MAX_IMDB_QID,
    ANALIZA_WIKI_CACHE_MAX_IS_FILM,
    ANALIZA_WIKI_CACHE_MAX_RECORDS,
    ANALIZA_WIKI_DEBUG,
    WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES,
    WIKI_CACHE_FLUSH_MAX_SECONDS,
    WIKI_CACHE_JSON_INDENT,
    WIKI_CACHE_JSON_PRETTY,
    WIKI_CACHE_PATH,
    WIKI_CACHE_SWR_OK_GRACE_SECONDS,
    WIKI_CACHE_TTL_NEGATIVE_SECONDS,
    WIKI_CACHE_TTL_OK_SECONDS,
    WIKI_CB_FAILURE_THRESHOLD,
    WIKI_CB_OPEN_SECONDS,
    WIKI_DISAMBIGUATION_NEGATIVE_TTL_SECONDS,
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
    WIKI_SEARCH_CANDIDATES_MAX_ENTRIES,
    WIKI_SEARCH_CANDIDATES_TTL_SECONDS,
    WIKI_SINGLEFLIGHT_WAIT_SECONDS,
    WIKI_SPARQL_MIN_INTERVAL_SECONDS,
    WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS,
    WIKI_SPARQL_TIMEOUT_READ_SECONDS,
    WIKI_WDQS_CB_FAILURE_THRESHOLD,
    WIKI_WDQS_CB_OPEN_SECONDS,
    WIKI_WDQS_URL,
    WIKI_WIKIDATA_API_BASE_URL,
    WIKI_WIKIDATA_ENTITY_BASE_URL,
    WIKI_WIKIPEDIA_API_BASE_URL,
    WIKI_WIKIPEDIA_REST_BASE_URL,
)
from backend.title_utils import (
    generate_sequel_title_variants,
    normalize_title_for_lookup,
)

# --------------------------------------------------------------------------------------
# Opcional: knob general en config.py (NO obligatorio).
# Si no existe, no pasa nada.
# --------------------------------------------------------------------------------------
try:
    from backend.config import WIKI_CLIENT_DISABLE_INFO_LOGS  # type: ignore
except Exception:  # pragma: no cover
    WIKI_CLIENT_DISABLE_INFO_LOGS = False  # type: ignore


# =============================================================================
# Tipos dict-like (schema v6)
# =============================================================================


class WikidataEntity(dict):
    """Entidad etiquetada (qid -> label/description/type)."""


class WikiBlock(dict):
    """Bloque Wikipedia (REST summary) normalizado."""


class WikidataBlock(dict):
    """Bloque Wikidata: qid + listas de QIDs."""


WikiItemStatus = Literal["ok", "no_qid", "not_film", "imdb_no_qid", "disambiguation"]


class WikiCacheItem(dict):
    """
    Entrada principal cacheable.
      - Title, Year, imdbID
      - wiki, wikidata
      - fetched_at, ttl_s, status
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
      - search_cache (opcional)
    """


# =============================================================================
# Protocol idioma por item (para get_wiki_for_input)
# =============================================================================


class MovieInputLangProto(Protocol):
    def plex_library_language(self) -> str | None: ...
    def is_spanish_context(self) -> bool: ...
    def is_english_context(self) -> bool: ...
    def is_italian_context(self) -> bool: ...
    def is_french_context(self) -> bool: ...
    def is_japanese_context(self) -> bool: ...
    def is_korean_context(self) -> bool: ...
    def is_chinese_context(self) -> bool: ...


# =============================================================================
# Utils defensivos
# =============================================================================


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


# =============================================================================
# Logging centralizado (client-only)
# =============================================================================


def _log_debug(msg: object) -> None:
    """
    Wrapper defensivo para evitar que Pyright propague NoReturn desde stubs del logger.
    """
    try:
        dbg = cast(
            Any, logger.debug_ctx
        )  # evita inferencias NoReturn / overloads raros
        dbg("WIKI", msg)
    except Exception:
        # best-effort: el cliente no debe romper por logging
        return


def _log_info(msg: str) -> None:
    """
    Info solo si NO silent y NO deshabilitado.
    Blindado contra stubs NoReturn del logger.
    """
    # Evitamos returns tempranos por si algún type-checker “constant-fold” raro
    silent = bool(logger.is_silent_mode())
    disabled = bool(WIKI_CLIENT_DISABLE_INFO_LOGS)

    if (not silent) and (not disabled):
        try:
            inf = cast(Any, logger.info)  # evita inferencias NoReturn / overloads raros
            inf(msg)
        except Exception:
            return


def _log_error(msg: str, *, always: bool = True) -> None:
    """
    Evitar “spam”: úsalo solo para condiciones realmente anómalas (cliente best-effort).
    Blindado contra stubs NoReturn del logger.
    """
    try:
        err = cast(Any, logger.error)  # evita inferencias NoReturn / overloads raros
        err(msg, always=always)
    except Exception:
        return


# =============================================================================
# Constantes / estado interno
# =============================================================================

_SCHEMA_VERSION: Final[int] = 6
_CACHE_PATH: Final[Path] = Path(WIKI_CACHE_PATH)

_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()

_CACHE: WikiCacheFile | None = None
_CACHE_LOCK = threading.RLock()

_CACHE_DIRTY: bool = False
_CACHE_DIRTY_WRITES: int = 0
_CACHE_LAST_FLUSH_TS: float = 0.0

# SPARQL throttle
_LAST_SPARQL_MONO: float = 0.0
_SPARQL_THROTTLE_LOCK = threading.Lock()
_SPARQL_MIN_INTERVAL_S: Final[float] = max(0.0, float(WIKI_SPARQL_MIN_INTERVAL_SECONDS))

# TTLs
_TTL_OK_S: Final[int] = int(WIKI_CACHE_TTL_OK_SECONDS)
_TTL_NEGATIVE_S: Final[int] = int(WIKI_CACHE_TTL_NEGATIVE_SECONDS)
_TTL_DISAMBIG_S: Final[int] = int(WIKI_DISAMBIGUATION_NEGATIVE_TTL_SECONDS)
_TTL_IMDB_QID_NEGATIVE_S: Final[int] = int(WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS)
_TTL_IS_FILM_S: Final[int] = int(WIKI_IS_FILM_TTL_SECONDS)

_SWR_OK_GRACE_S: Final[int] = max(0, int(WIKI_CACHE_SWR_OK_GRACE_SECONDS))
_SINGLEFLIGHT_WAIT_S: Final[float] = max(0.05, float(WIKI_SINGLEFLIGHT_WAIT_SECONDS))

_SEARCH_CAND_TTL_S: Final[int] = max(60, int(WIKI_SEARCH_CANDIDATES_TTL_SECONDS))
_SEARCH_CAND_MAX: Final[int] = max(0, int(WIKI_SEARCH_CANDIDATES_MAX_ENTRIES))

_FLUSH_MAX_DIRTY_WRITES: Final[int] = max(1, int(WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES))
_FLUSH_MAX_SECONDS: Final[float] = max(0.1, float(WIKI_CACHE_FLUSH_MAX_SECONDS))

# Compaction caps
_COMPACT_MAX_RECORDS: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_RECORDS)
_COMPACT_MAX_IMDB_QID: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_IMDB_QID)
_COMPACT_MAX_IS_FILM: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_IS_FILM)
_COMPACT_MAX_ENTITIES: Final[int] = int(ANALIZA_WIKI_CACHE_MAX_ENTITIES)

_WIKI_DEBUG_EXTRA: Final[bool] = bool(ANALIZA_WIKI_DEBUG)

# Wikidata P31 heurística
_FILM_INSTANCE_ALLOWLIST: Final[set[str]] = {
    "Q11424",  # film
    "Q24862",  # feature film
    "Q202866",  # animated film
    "Q226730",  # television film
    "Q93204",  # short film
}
_FILM_INSTANCE_DENYLIST: Final[set[str]] = {
    "Q5398426",  # television series
    "Q7725634",  # literary work
    "Q571",  # book
    "Q8261",  # novel
    "Q25379",  # episode
}

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_WIKI_TITLE_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(r"\s*\([^)]*\)\s*$")

# HTTP
_HTTP_TIMEOUT: Final[float] = max(0.5, float(WIKI_HTTP_TIMEOUT_SECONDS))
_HTTP_TIMEOUT_SPARQL: Final[tuple[float, float]] = (
    max(0.5, float(WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS)),
    max(1.0, float(WIKI_SPARQL_TIMEOUT_READ_SECONDS)),
)

_HTTP_POOL_MAXSIZE: Final[int] = _cap_int_runtime(
    int(WIKI_HTTP_MAX_CONCURRENCY), min_v=1, max_v=128
)
_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT: Final[float] = _cap_float_runtime(
    float(_HTTP_TIMEOUT) * 3.0, min_v=0.2, max_v=120.0
)
_HTTP_SEM = threading.BoundedSemaphore(_HTTP_POOL_MAXSIZE)


# =============================================================================
# Helpers básicos
# =============================================================================


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
    norm = lang.strip().lower().replace("_", "-")
    if not norm:
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
    base = norm.split("-", 1)[0]
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


def _request_key_for_singleflight(
    *, imdb_norm: str | None, norm_title: str, norm_year: str
) -> str:
    if imdb_norm:
        return f"imdb:{imdb_norm}"
    return f"ty:{norm_title}|{norm_year}"


def _extract_year_from_wd_entity(wd_entity: Mapping[str, Any]) -> str:
    """
    Intenta extraer el año (YYYY) desde Wikidata:
      - P577 (publication date)
    Devuelve "" si no puede.
    """
    try:
        claims_obj = wd_entity.get("claims")
        if not isinstance(claims_obj, Mapping):
            return ""

        p577 = claims_obj.get("P577")
        if not isinstance(p577, list) or not p577:
            return ""

        for claim in p577:
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

            # Formato típico: {"time":"+1999-10-15T00:00:00Z", ...}
            t = value.get("time")
            if isinstance(t, str) and len(t) >= 5:
                # "+YYYY-..."
                m = re.match(r"^[+-]?(\d{4})-", t)
                if m:
                    return m.group(1)

        return ""
    except Exception:
        return ""


# =============================================================================
# Typed constructors (fix dict-subclass typing)
# =============================================================================


def _mk_cache_file(d: Mapping[str, object]) -> WikiCacheFile:
    return cast(WikiCacheFile, dict(d))


def _mk_cache_item(d: Mapping[str, object]) -> WikiCacheItem:
    return cast(WikiCacheItem, dict(d))


def _mk_wiki_block(d: Mapping[str, object]) -> WikiBlock:
    return cast(WikiBlock, dict(d))


def _mk_wikidata_block(d: Mapping[str, object]) -> WikidataBlock:
    return cast(WikidataBlock, dict(d))


def _mk_entity(d: Mapping[str, object]) -> WikidataEntity:
    return cast(WikidataEntity, dict(d))


def _mk_imdb_qid_entry(d: Mapping[str, object]) -> ImdbQidCacheEntry:
    return cast(ImdbQidCacheEntry, dict(d))


def _mk_is_film_entry(d: Mapping[str, object]) -> IsFilmCacheEntry:
    return cast(IsFilmCacheEntry, dict(d))


# =============================================================================
# Circuit breaker suave
# =============================================================================


class _CircuitBreakerOpen(RequestException):
    """Excepción interna: el circuito está abierto (no cuenta como failure)."""


_CB_LOCK = threading.Lock()
_CB_STATE: dict[str, dict[str, Any]] = {
    "wiki": {"failure_count": 0, "open_until_mono": 0.0},
    "wdqs": {"failure_count": 0, "open_until_mono": 0.0},
}
_CB_THRESH: Final[dict[str, int]] = {
    "wiki": max(1, int(WIKI_CB_FAILURE_THRESHOLD)),
    "wdqs": max(1, int(WIKI_WDQS_CB_FAILURE_THRESHOLD)),
}
_CB_OPEN_S: Final[dict[str, float]] = {
    "wiki": max(0.1, float(WIKI_CB_OPEN_SECONDS)),
    "wdqs": max(0.1, float(WIKI_WDQS_CB_OPEN_SECONDS)),
}


def _cb_is_open(key: str) -> bool:
    now = time.monotonic()
    with _CB_LOCK:
        st = _CB_STATE.get(key)
        if not st:
            return False
        open_until = float(st.get("open_until_mono") or 0.0)
        return now < open_until


def _cb_short_circuit(key: str) -> None:
    if key == "wiki":
        _m_inc("cb_short_circuits_wiki", 1)
    elif key == "wdqs":
        _m_inc("cb_short_circuits_wdqs", 1)
    raise _CircuitBreakerOpen(f"WIKI circuit breaker OPEN ({key})")


def _cb_record_success(key: str) -> None:
    with _CB_LOCK:
        st = _CB_STATE.get(key)
        if not st:
            return
        st["failure_count"] = 0
        st["open_until_mono"] = 0.0


def _cb_record_failure(key: str) -> None:
    thr = _CB_THRESH.get(key, 5)
    open_s = _CB_OPEN_S.get(key, 20.0)
    now = time.monotonic()

    with _CB_LOCK:
        st = _CB_STATE.get(key)
        if not st:
            return
        fc = int(st.get("failure_count") or 0) + 1
        st["failure_count"] = fc

        if fc >= thr:
            open_until = float(st.get("open_until_mono") or 0.0)
            new_until = now + open_s
            if new_until > open_until:
                st["open_until_mono"] = new_until
                if key == "wiki":
                    _m_inc("cb_open_events_wiki", 1)
                elif key == "wdqs":
                    _m_inc("cb_open_events_wdqs", 1)
                _log_debug(f"CB OPEN ({key}) for {open_s:.1f}s (failures={fc}/{thr})")


# =============================================================================
# Single-flight + SWR refresh scheduler
# =============================================================================

_SINGLEFLIGHT_LOCK = threading.Lock()
_SINGLEFLIGHT_EVENTS: dict[str, threading.Event] = {}

_REFRESH_IN_FLIGHT_LOCK = threading.Lock()
_REFRESH_IN_FLIGHT: set[str] = set()


def _singleflight_enter(key: str) -> tuple[bool, threading.Event]:
    with _SINGLEFLIGHT_LOCK:
        ev = _SINGLEFLIGHT_EVENTS.get(key)
        if ev is not None:
            return False, ev
        ev = threading.Event()
        _SINGLEFLIGHT_EVENTS[key] = ev
        return True, ev


def _singleflight_leave(key: str) -> None:
    with _SINGLEFLIGHT_LOCK:
        ev = _SINGLEFLIGHT_EVENTS.pop(key, None)
    if ev is not None:
        ev.set()


# =============================================================================
# Métricas
# =============================================================================

_METRICS_LOCK = threading.Lock()
_METRICS: dict[str, int] = {
    "cache_hits": 0,
    "cache_misses": 0,
    "cache_store_writes": 0,
    "cache_flush_writes": 0,
    "cache_compactions": 0,
    "cache_expired_hits": 0,
    "cache_stale_served": 0,
    "cache_title_mismatch_purged": 0,
    "singleflight_waits": 0,
    "singleflight_wait_timeouts": 0,
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
    "http_semaphore_timeouts": 0,
    "cb_short_circuits_wiki": 0,
    "cb_short_circuits_wdqs": 0,
    "cb_open_events_wiki": 0,
    "cb_open_events_wdqs": 0,
    "items_ok_with_directors": 0,
    "items_ok_with_genres": 0,
    "items_ok_with_countries": 0,
    "items_ok_with_images": 0,
    "items_ok_with_summary": 0,
    "items_ok_with_description": 0,
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


def log_wiki_metrics_summary(*, force: bool = False) -> None:
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


# =============================================================================
# HTTP session / GET
# =============================================================================


def _get_session() -> requests.Session:
    """
    Crea una requests.Session con pool/retry.

    Nota typing:
    - Patrón “local -> lock -> local -> create” evita falsas ramas unreachable en type-checkers.
    """
    global _SESSION

    s1 = _SESSION
    if s1 is not None:
        return s1

    with _SESSION_LOCK:
        s2 = _SESSION
        if s2 is not None:
            return s2

        new_session = requests.Session()

        ua = str(WIKI_HTTP_USER_AGENT).strip() or "Analiza-Movies/1.0 (local)"
        new_session.headers.update(
            {
                "User-Agent": ua,
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": f"{WIKI_LANGUAGE},{WIKI_FALLBACK_LANGUAGE};q=0.8,en;q=0.6,es;q=0.5",
            }
        )

        retry_total = _cap_int_runtime(int(WIKI_HTTP_RETRY_TOTAL), min_v=0, max_v=10)
        backoff = _cap_float_runtime(
            float(WIKI_HTTP_RETRY_BACKOFF_FACTOR), min_v=0.0, max_v=10.0
        )

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
        new_session.mount("https://", adapter)
        new_session.mount("http://", adapter)

        _SESSION = new_session
        return new_session


def _close_session_on_exit() -> None:
    global _SESSION
    with _SESSION_LOCK:
        s = _SESSION
        _SESSION = None
    if s is not None:
        try:
            s.close()
        except Exception:
            pass


def _http_get(
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: float | tuple[float, float] | None = None,
    breaker: Literal["wiki", "wdqs"] = "wiki",
) -> requests.Response:
    """
    GET con:
      - circuito (antes de salir a red)
      - semáforo de concurrencia para evitar saturación de pool/endpoints
    """
    if _cb_is_open(breaker):
        _cb_short_circuit(breaker)

    acquired = _HTTP_SEM.acquire(timeout=_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT)
    if not acquired:
        _m_inc("http_semaphore_timeouts", 1)
        raise RequestException("WIKI HTTP semaphore acquire timeout")

    try:
        return _get_session().get(url, params=params, timeout=timeout)
    finally:
        try:
            _HTTP_SEM.release()
        except ValueError:
            _log_debug("HTTP semaphore release ValueError (ignored)")


# =============================================================================
# Cache IO (schema v6)
# =============================================================================


def _mark_dirty_unlocked() -> None:
    global _CACHE_DIRTY, _CACHE_DIRTY_WRITES
    _CACHE_DIRTY = True
    _CACHE_DIRTY_WRITES += 1


def _save_cache_file_atomic(cache: WikiCacheFile) -> None:
    """
    Escritura atómica:
      1) dump a tempfile en el mismo directorio
      2) fsync (best-effort)
      3) os.replace
    """
    dirpath = _CACHE_PATH.parent
    dirpath.mkdir(parents=True, exist_ok=True)

    dump_kwargs: dict[str, Any] = {"ensure_ascii": False}
    if bool(WIKI_CACHE_JSON_PRETTY):
        dump_kwargs["indent"] = max(0, min(8, int(WIKI_CACHE_JSON_INDENT)))
    else:
        dump_kwargs["separators"] = (",", ":")

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=str(dirpath)
        ) as tf:
            json.dump(cache, tf, **dump_kwargs)
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


def _empty_cache() -> WikiCacheFile:
    return _mk_cache_file(
        {
            "schema": _SCHEMA_VERSION,
            "language": str(WIKI_LANGUAGE),
            "fallback_language": str(WIKI_FALLBACK_LANGUAGE),
            "records": {},
            "index_imdb": {},
            "index_ty": {},
            "entities": {},
            "imdb_qid": {},
            "is_film": {},
            "search_cache": {},
        }
    )


def _maybe_quarantine_corrupt_cache() -> None:
    # Solo en debug para no sorprender en producción.
    if not logger.is_debug_mode():
        return
    try:
        if not _CACHE_PATH.exists():
            return
        ts = int(time.time())
        bad_path = _CACHE_PATH.with_name(f"{_CACHE_PATH.name}.corrupt.{ts}")
        os.replace(str(_CACHE_PATH), str(bad_path))
        _log_debug(f"Quarantined corrupt cache file -> {bad_path.name}")
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
        _log_debug(f"cache read failed ({exc!r}); recreating")
        _maybe_quarantine_corrupt_cache()
        _CACHE = _empty_cache()
        return _CACHE

    if not isinstance(raw_obj, Mapping) or raw_obj.get("schema") != _SCHEMA_VERSION:
        found_schema = raw_obj.get("schema") if isinstance(raw_obj, Mapping) else None
        _log_debug(f"cache schema mismatch -> recreate (found={found_schema!r})")
        _CACHE = _empty_cache()
        return _CACHE

    records_obj = raw_obj.get("records")
    index_imdb_obj = raw_obj.get("index_imdb")
    index_ty_obj = raw_obj.get("index_ty")
    entities_obj = raw_obj.get("entities")
    imdb_qid_obj = raw_obj.get("imdb_qid")
    is_film_obj = raw_obj.get("is_film")
    search_cache_obj = raw_obj.get("search_cache")

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

    # records
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

        records[rid] = _mk_cache_item(
            {
                "Title": title,
                "Year": year,
                "imdbID": _norm_imdb(imdb_id),
                "wiki": dict(wiki),
                "wikidata": dict(wikidata),
                "fetched_at": fetched_at,
                "ttl_s": ttl_s,
                "status": status,
            }
        )

    # indices
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

    # entities
    entities: dict[str, WikidataEntity] = {}
    for k, v in entities_obj.items():
        qid = _safe_str(k)
        if not qid or not isinstance(v, Mapping):
            continue
        entities[qid] = _mk_entity(v)

    # imdb_qid
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
        imdb_qid[imdb] = _mk_imdb_qid_entry(
            {"qid": qid, "fetched_at": fetched_at, "ttl_s": ttl_s}
        )

    # is_film
    is_film: dict[str, IsFilmCacheEntry] = {}
    for k, v in is_film_obj.items():
        qid = _safe_str(k)
        if not qid or not isinstance(v, Mapping):
            continue
        fetched_at = v.get("fetched_at")
        ttl_s = v.get("ttl_s")
        is_film_val = v.get("is_film")
        if (
            not isinstance(fetched_at, int)
            or not isinstance(ttl_s, int)
            or not isinstance(is_film_val, bool)
        ):
            continue
        is_film[qid] = _mk_is_film_entry(
            {"is_film": is_film_val, "fetched_at": fetched_at, "ttl_s": ttl_s}
        )

    # search_cache
    search_cache: dict[str, dict[str, Any]] = {}
    if isinstance(search_cache_obj, Mapping):
        for k, v in search_cache_obj.items():
            if not isinstance(k, str) or not isinstance(v, Mapping):
                continue
            fa = v.get("fetched_at")
            ttl = v.get("ttl_s")
            titles = v.get("titles")
            if (
                not isinstance(fa, int)
                or not isinstance(ttl, int)
                or not isinstance(titles, list)
            ):
                continue
            search_cache[k] = {
                "fetched_at": int(fa),
                "ttl_s": int(ttl),
                "titles": [t for t in titles if isinstance(t, str)],
            }

    # Limpieza de índices huérfanos
    index_imdb = {k: rid for k, rid in index_imdb.items() if rid in records}
    index_ty = {k: rid for k, rid in index_ty.items() if rid in records}

    _CACHE = _mk_cache_file(
        {
            "schema": _SCHEMA_VERSION,
            "language": str(raw_obj.get("language") or WIKI_LANGUAGE),
            "fallback_language": str(
                raw_obj.get("fallback_language") or WIKI_FALLBACK_LANGUAGE
            ),
            "records": records,
            "index_imdb": index_imdb,
            "index_ty": index_ty,
            "entities": entities,
            "imdb_qid": imdb_qid,
            "is_film": is_film,
            "search_cache": search_cache,
        }
    )

    # Best-effort compaction (rápida y defensiva)
    try:
        _compact_cache_unlocked(_CACHE, force=False)
    except Exception:
        pass

    return _CACHE


def _maybe_flush_unlocked(force: bool) -> None:
    """
    Flush condicionado (barato):
    - Si no está dirty y no force: return.
    - Si dirty, sólo flush si se supera umbral de writes o ventana temporal.
    """
    global _CACHE_DIRTY, _CACHE_DIRTY_WRITES, _CACHE_LAST_FLUSH_TS, _CACHE
    if _CACHE is None:
        return
    if not _CACHE_DIRTY and not force:
        return

    now = time.monotonic()
    if not force:
        if (
            _CACHE_DIRTY_WRITES < _FLUSH_MAX_DIRTY_WRITES
            and (now - _CACHE_LAST_FLUSH_TS) < _FLUSH_MAX_SECONDS
        ):
            return

    try:
        _compact_cache_unlocked(_CACHE, force=force)
        _save_cache_file_atomic(_CACHE)
        _CACHE_DIRTY = False
        _CACHE_DIRTY_WRITES = 0
        _CACHE_LAST_FLUSH_TS = now
        _m_inc("cache_flush_writes", 1)
    except Exception as exc:
        _log_debug(f"cache flush failed: {exc!r}")


def flush_wiki_cache() -> None:
    try:
        with _CACHE_LOCK:
            _maybe_flush_unlocked(force=True)
    except Exception as exc:
        _log_debug(f"flush_wiki_cache failed: {exc!r}")


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
    try:
        _close_session_on_exit()
    except Exception:
        pass


atexit.register(_flush_cache_on_exit)


# =============================================================================
# Compaction / GC (SWR-aware)
# =============================================================================
def _compact_cache_unlocked(cache: WikiCacheFile, *, force: bool) -> None:
    try:
        now_epoch = _now_epoch()

        records_obj = cache.get("records")
        if not isinstance(records_obj, Mapping):
            cache["records"] = {}
            records_obj = cast(Mapping[str, object], cache["records"])

        records: dict[str, WikiCacheItem] = {}
        for rid, it in records_obj.items():
            if not isinstance(rid, str) or not isinstance(it, Mapping):
                continue
            fetched_at = it.get("fetched_at")
            ttl_s = it.get("ttl_s")
            if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
                continue

            expired = _is_expired(fetched_at, ttl_s, now_epoch)
            if expired:
                status = it.get("status")
                if status == "ok" and _SWR_OK_GRACE_S > 0:
                    age = now_epoch - int(fetched_at)
                    if age > (int(ttl_s) + int(_SWR_OK_GRACE_S)):
                        continue
                else:
                    continue

            d = dict(it)
            d["imdbID"] = _norm_imdb(
                d.get("imdbID") if isinstance(d.get("imdbID"), str) else None
            )
            records[rid] = _mk_cache_item(d)

        if _COMPACT_MAX_RECORDS > 0 and len(records) > _COMPACT_MAX_RECORDS:
            ranked_rec = sorted(
                records.items(),
                key=lambda kv: int(kv[1].get("fetched_at", 0) or 0),
                reverse=True,
            )
            trimmed_records: dict[str, WikiCacheItem] = {}
            for rec_rid, rec_item in ranked_rec[:_COMPACT_MAX_RECORDS]:
                trimmed_records[rec_rid] = rec_item
            records = trimmed_records

        cache["records"] = records

        index_imdb: dict[str, str] = {}
        index_ty: dict[str, str] = {}
        for rec_rid, rec_item in records.items():
            title = rec_item.get("Title")
            year = rec_item.get("Year")

            if (
                isinstance(title, str)
                and isinstance(year, str)
                and title.strip()
                and year.strip()
            ):
                index_ty[_ty_key(title, year)] = rec_rid

            imdb = _norm_imdb(
                rec_item.get("imdbID")
                if isinstance(rec_item.get("imdbID"), str)
                else None
            )
            if imdb:
                index_imdb[imdb] = rec_rid

        cache["index_imdb"] = index_imdb
        cache["index_ty"] = index_ty

        imdb_qid_obj = cache.get("imdb_qid")
        imdb_qid: dict[str, ImdbQidCacheEntry] = {}
        if isinstance(imdb_qid_obj, Mapping):
            for imdb_key_raw, qid_entry_raw in imdb_qid_obj.items():
                imdb_key = _norm_imdb(
                    imdb_key_raw if isinstance(imdb_key_raw, str) else None
                )
                if not imdb_key or not isinstance(qid_entry_raw, Mapping):
                    continue
                fetched_at = qid_entry_raw.get("fetched_at")
                ttl_s = qid_entry_raw.get("ttl_s")
                if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
                    continue
                if _is_expired(fetched_at, ttl_s, now_epoch):
                    continue
                imdb_qid[imdb_key] = _mk_imdb_qid_entry(qid_entry_raw)

        if _COMPACT_MAX_IMDB_QID > 0 and len(imdb_qid) > _COMPACT_MAX_IMDB_QID:
            ranked_qid = sorted(
                imdb_qid.items(),
                key=lambda kv: int(kv[1].get("fetched_at", 0) or 0),
                reverse=True,
            )
            trimmed_qid: dict[str, ImdbQidCacheEntry] = {}
            for imdb_key, qid_entry in ranked_qid[:_COMPACT_MAX_IMDB_QID]:
                trimmed_qid[imdb_key] = qid_entry
            imdb_qid = trimmed_qid

        cache["imdb_qid"] = imdb_qid

        is_film_obj = cache.get("is_film")
        is_film: dict[str, IsFilmCacheEntry] = {}
        if isinstance(is_film_obj, Mapping):
            for qid_key_raw, is_entry_raw in is_film_obj.items():
                qid_key = _safe_str(qid_key_raw)
                if not qid_key or not isinstance(is_entry_raw, Mapping):
                    continue
                fetched_at = is_entry_raw.get("fetched_at")
                ttl_s = is_entry_raw.get("ttl_s")
                if not isinstance(fetched_at, int) or not isinstance(ttl_s, int):
                    continue
                if _is_expired(fetched_at, ttl_s, now_epoch):
                    continue
                is_film[qid_key] = _mk_is_film_entry(is_entry_raw)

        if _COMPACT_MAX_IS_FILM > 0 and len(is_film) > _COMPACT_MAX_IS_FILM:
            ranked_is = sorted(
                is_film.items(),
                key=lambda kv: int(kv[1].get("fetched_at", 0) or 0),
                reverse=True,
            )
            trimmed_is: dict[str, IsFilmCacheEntry] = {}
            for qid_key, is_entry in ranked_is[:_COMPACT_MAX_IS_FILM]:
                trimmed_is[qid_key] = is_entry
            is_film = trimmed_is

        cache["is_film"] = is_film

        referenced_qids: set[str] = set()
        for rec_item in records.values():
            wikidata_block = rec_item.get("wikidata")
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

            wiki_block = rec_item.get("wiki")
            if isinstance(wiki_block, Mapping):
                wb = wiki_block.get("wikibase_item")
                if isinstance(wb, str) and wb.strip():
                    referenced_qids.add(wb.strip())

        entities_obj = cache.get("entities")
        entities: dict[str, WikidataEntity] = {}
        if isinstance(entities_obj, Mapping):
            for ent_qid_raw, ent_raw in entities_obj.items():
                ent_qid = _safe_str(ent_qid_raw)
                if not ent_qid or ent_qid not in referenced_qids:
                    continue
                if isinstance(ent_raw, Mapping):
                    entities[ent_qid] = _mk_entity(ent_raw)

        if _COMPACT_MAX_ENTITIES > 0 and len(entities) > _COMPACT_MAX_ENTITIES:
            keep_keys = sorted(entities.keys())[:_COMPACT_MAX_ENTITIES]
            trimmed_ent: dict[str, WikidataEntity] = {}
            for ent_qid in keep_keys:
                trimmed_ent[ent_qid] = entities[ent_qid]
            entities = trimmed_ent

        cache["entities"] = entities

        sc_obj = cache.get("search_cache")
        sc: dict[str, dict[str, Any]] = {}
        if isinstance(sc_obj, Mapping):
            for sc_key_raw, sc_entry_raw in sc_obj.items():
                if not isinstance(sc_key_raw, str) or not isinstance(
                    sc_entry_raw, Mapping
                ):
                    continue
                fa = sc_entry_raw.get("fetched_at")
                ttl = sc_entry_raw.get("ttl_s")
                titles = sc_entry_raw.get("titles")
                if (
                    not isinstance(fa, int)
                    or not isinstance(ttl, int)
                    or not isinstance(titles, list)
                ):
                    continue
                if _is_expired(int(fa), int(ttl), now_epoch):
                    continue
                sc[sc_key_raw] = {
                    "fetched_at": int(fa),
                    "ttl_s": int(ttl),
                    "titles": [t for t in titles if isinstance(t, str)],
                }

        if _SEARCH_CAND_MAX > 0 and len(sc) > _SEARCH_CAND_MAX:
            ranked_sc = sorted(
                sc.items(),
                key=lambda kv: int(kv[1].get("fetched_at", 0) or 0),
                reverse=True,
            )
            trimmed_sc: dict[str, dict[str, Any]] = {}
            for sc_key, sc_entry in ranked_sc[:_SEARCH_CAND_MAX]:
                trimmed_sc[sc_key] = sc_entry
            sc = trimmed_sc

        cache["search_cache"] = sc

        _m_inc("cache_compactions", 1)

        if _WIKI_DEBUG_EXTRA:
            _log_debug(
                "cache compacted"
                + (" (force)" if force else "")
                + f" | records={len(records)} idx_imdb={len(index_imdb)} idx_ty={len(index_ty)} "
                f"entities={len(entities)} imdb_qid={len(imdb_qid)} is_film={len(is_film)} search_cache={len(sc)}"
            )
        else:
            _log_debug(
                f"cache compacted{' (force)' if force else ''} | records={len(records)} entities={len(entities)} search_cache={len(sc)}"
            )

    except Exception as exc:
        _log_debug(f"cache compaction failed: {exc!r}")


# =============================================================================
# Idiomas por item
# =============================================================================


def _detect_language_chain_from_input(
    movie_input: MovieInputLangProto | None,
) -> list[str]:
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

    for cfg_lang in (
        _normalize_lang_code(WIKI_LANGUAGE),
        _normalize_lang_code(WIKI_FALLBACK_LANGUAGE),
    ):
        if cfg_lang and cfg_lang not in chain:
            chain.append(cfg_lang)

    if "en" not in chain:
        chain.append("en")

    out: list[str] = []
    seen: set[str] = set()
    for lang_code in chain:
        if lang_code and lang_code not in seen:
            seen.add(lang_code)
            out.append(lang_code)
    return out


def _best_wikipedia_languages_for_item(
    movie_input: MovieInputLangProto | None,
) -> tuple[str, str]:
    chain = _detect_language_chain_from_input(movie_input)
    primary = chain[0] if chain else (_normalize_lang_code(WIKI_LANGUAGE) or "en")

    fallback = ""
    for lang_code in chain[1:]:
        if lang_code != primary:
            fallback = lang_code
            break

    if not fallback:
        fallback = _normalize_lang_code(WIKI_FALLBACK_LANGUAGE) or (
            "en" if primary != "en" else ""
        )

    if fallback == primary:
        fallback = "en" if primary != "en" else ""

    return primary, fallback


# =============================================================================
# Canonicalización para ranking de búsquedas
# =============================================================================


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _canon_cmp(text: str) -> str:
    base = _strip_accents(text).lower()
    tokens = _WORD_RE.findall(base)
    return " ".join(tokens)


def _strip_wikipedia_title_suffix(title: str) -> str:
    out = title.strip()
    if not out:
        return ""
    out = _WIKI_TITLE_SUFFIX_RE.sub("", out).strip()
    return out


def _titles_share_significant_tokens(lookup_title: str, wiki_title: str) -> bool:
    lt = _canon_cmp(lookup_title)
    wt = _canon_cmp(_strip_wikipedia_title_suffix(wiki_title))
    if not lt or not wt:
        return True
    if len(lt.replace(" ", "")) <= 3:
        return True
    if lt == wt or lt in wt or wt in lt:
        return True

    ltoks = {t for t in lt.split() if len(t) >= 3}
    wtoks = {t for t in wt.split() if len(t) >= 3}
    if not ltoks or not wtoks:
        return True
    return bool(ltoks & wtoks)


# =============================================================================
# Wikipedia REST + Search
# =============================================================================

_DISAMBIG_SENTINEL: Final[dict[str, Any]] = {"__wiki_disambiguation__": True}


def _is_disambiguation_payload(payload: Mapping[str, Any] | None) -> bool:
    if payload is None:
        return False
    return bool(payload.get("__wiki_disambiguation__") is True)


def _fetch_wikipedia_summary_by_title(
    title: str, language: str
) -> Mapping[str, Any] | None:
    _m_inc("wikipedia_summary_calls", 1)

    safe_title = quote(title.replace(" ", "_"), safe="")
    base = str(WIKI_WIKIPEDIA_REST_BASE_URL).rstrip("/")
    url = f"{base.format(lang=language)}/page/summary/{safe_title}"

    _log_debug(f"wikipedia.summary -> lang={language} title={title!r}")

    try:
        resp = _http_get(url, timeout=_HTTP_TIMEOUT, breaker="wiki")
    except _CircuitBreakerOpen:
        return None
    except RequestException as exc:
        _m_inc("wikipedia_failures", 1)
        _cb_record_failure("wiki")
        _log_debug(f"wikipedia.summary EXC: {exc!r}")
        return None

    if resp.status_code != 200:
        _m_inc("wikipedia_failures", 1)
        _cb_record_failure("wiki")
        _log_debug(f"wikipedia.summary <- status={resp.status_code}")
        return None

    try:
        data = resp.json()
    except (ValueError, JSONDecodeError) as exc:
        _m_inc("wikipedia_failures", 1)
        _cb_record_failure("wiki")
        _log_debug(f"wikipedia.summary JSON EXC: {exc!r}")
        return None

    if not isinstance(data, Mapping):
        _m_inc("wikipedia_failures", 1)
        _cb_record_failure("wiki")
        return None

    if _safe_str(data.get("type")) == "disambiguation":
        _log_debug("wikipedia.summary -> disambiguation")
        _cb_record_success("wiki")
        return _DISAMBIG_SENTINEL

    _cb_record_success("wiki")
    return data


def _wikipedia_search(
    *, query: str, language: str, limit: int = 8
) -> list[dict[str, Any]]:
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

    _log_debug(f"wikipedia.search -> lang={language} q={query!r}")

    try:
        resp = _http_get(url, params=params, timeout=_HTTP_TIMEOUT, breaker="wiki")
    except _CircuitBreakerOpen:
        return []
    except RequestException as exc:
        _m_inc("wikipedia_failures", 1)
        _cb_record_failure("wiki")
        _log_debug(f"wikipedia.search EXC: {exc!r}")
        return []

    if resp.status_code != 200:
        _m_inc("wikipedia_failures", 1)
        _cb_record_failure("wiki")
        _log_debug(f"wikipedia.search <- status={resp.status_code}")
        return []

    try:
        payload = resp.json()
    except (ValueError, JSONDecodeError) as exc:
        _m_inc("wikipedia_failures", 1)
        _cb_record_failure("wiki")
        _log_debug(f"wikipedia.search JSON EXC: {exc!r}")
        return []

    if not isinstance(payload, Mapping):
        _cb_record_failure("wiki")
        return []

    _cb_record_success("wiki")

    q = payload.get("query")
    if not isinstance(q, Mapping):
        return []

    search_obj = q.get("search")
    if not isinstance(search_obj, list):
        return []

    out: list[dict[str, Any]] = []
    for it in search_obj:
        if isinstance(it, Mapping):
            out.append(dict(it))
    return out


def _score_search_hit(
    *, hit_title: str, hit_snippet: str, wanted_title: str, year: int | None
) -> float:
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
    if any(
        tok in sn for tok in ("pelicula", "película", "film", "movie", "largometraje")
    ):
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


def _rank_wikipedia_candidates(
    *, lookup_title: str, year: int | None, language: str
) -> list[str]:
    clean_title = " ".join(lookup_title.strip().split())
    year_s = str(year) if year is not None else ""
    cache_key = f"{language}|{_canon_cmp(clean_title)}|{year_s}"

    now_epoch = _now_epoch()

    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        sc_obj = cache.get("search_cache")
        if isinstance(sc_obj, Mapping):
            entry = sc_obj.get(cache_key)
            if isinstance(entry, Mapping):
                fa = entry.get("fetched_at")
                ttl = entry.get("ttl_s")
                titles = entry.get("titles")
                if (
                    isinstance(fa, int)
                    and isinstance(ttl, int)
                    and isinstance(titles, list)
                ):
                    if not _is_expired(int(fa), int(ttl), now_epoch):
                        return [t for t in titles if isinstance(t, str)]

    def _queries_for_title(title_variant: str) -> list[str]:
        token = "película" if language == "es" else "film"
        queries: list[str] = []
        if year is not None:
            queries.append(f"{title_variant} {year} {token}")
        queries.append(f"{title_variant} {token}")
        queries.append(title_variant)

        seen_q: set[str] = set()
        deduped_queries: list[str] = []
        for q in queries:
            qq = q.strip()
            if qq and qq not in seen_q:
                seen_q.add(qq)
                deduped_queries.append(qq)
        return deduped_queries

    def _run_queries(title_variant: str, wanted_title: str) -> dict[str, float]:
        scored: dict[str, float] = {}
        for q in _queries_for_title(title_variant):
            for hit in _wikipedia_search(query=q, language=language, limit=10):
                ht = _safe_str(hit.get("title"))
                if not ht:
                    continue
                snippet_raw = hit.get("snippet")
                hs = str(snippet_raw) if snippet_raw is not None else ""
                s = _score_search_hit(
                    hit_title=ht, hit_snippet=hs, wanted_title=wanted_title, year=year
                )
                prev = scored.get(ht)
                if prev is None or s > prev:
                    scored[ht] = s
        return scored

    scored = _run_queries(clean_title, clean_title)
    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    out_titles = [t for (t, s) in ranked if s >= 4.0]

    if not out_titles:
        variants = generate_sequel_title_variants(clean_title)
        if variants:
            scored = _run_queries(variants[0], variants[0])
            ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
            out_titles = [t for (t, s) in ranked if s >= 4.0]

    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        sc = cache.get("search_cache")
        if not isinstance(sc, dict):
            cache["search_cache"] = {}
            sc = cache["search_cache"]
        sc[cache_key] = {
            "titles": out_titles[:32],
            "fetched_at": now_epoch,
            "ttl_s": int(_SEARCH_CAND_TTL_S),
        }
        _mark_dirty_unlocked()

    return out_titles


# =============================================================================
# Wikidata entity + labels
# =============================================================================


def _fetch_wikidata_entity_json(qid: str) -> Mapping[str, Any] | None:
    _m_inc("wikidata_entity_calls", 1)

    base = str(WIKI_WIKIDATA_ENTITY_BASE_URL).rstrip("/")
    url = f"{base}/{qid}.json"
    _log_debug(f"wikidata.entity -> qid={qid}")

    try:
        resp = _http_get(url, timeout=_HTTP_TIMEOUT, breaker="wiki")
    except _CircuitBreakerOpen:
        return None
    except RequestException as exc:
        _m_inc("wikidata_failures", 1)
        _cb_record_failure("wiki")
        _log_debug(f"wikidata.entity EXC: {exc!r}")
        return None

    if resp.status_code != 200:
        _m_inc("wikidata_failures", 1)
        _cb_record_failure("wiki")
        _log_debug(f"wikidata.entity <- status={resp.status_code}")
        return None

    try:
        data = resp.json()
    except (ValueError, JSONDecodeError) as exc:
        _m_inc("wikidata_failures", 1)
        _cb_record_failure("wiki")
        _log_debug(f"wikidata.entity JSON EXC: {exc!r}")
        return None

    if not isinstance(data, Mapping):
        _m_inc("wikidata_failures", 1)
        _cb_record_failure("wiki")
        return None

    entities = data.get("entities")
    if not isinstance(entities, Mapping):
        _m_inc("wikidata_failures", 1)
        _cb_record_failure("wiki")
        return None

    entity = entities.get(qid)
    if not isinstance(entity, Mapping):
        _m_inc("wikidata_failures", 1)
        _cb_record_failure("wiki")
        return None

    _cb_record_success("wiki")
    return entity


def _extract_qids_from_claims(entity: Mapping[str, Any], property_id: str) -> list[str]:
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


_IMDB_ID_RE = re.compile(r"^tt\\d{4,10}$", re.IGNORECASE)


def _extract_imdb_id_from_claims(entity: Mapping[str, Any]) -> str | None:
    claims_obj = entity.get("claims")
    if not isinstance(claims_obj, Mapping):
        return None

    prop_claims = claims_obj.get("P345")
    if not isinstance(prop_claims, list):
        return None

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
        if isinstance(value, str):
            imdb_raw = value.strip()
            if imdb_raw and _IMDB_ID_RE.match(imdb_raw):
                return _norm_imdb(imdb_raw)
        elif value is not None:
            imdb_raw = str(value).strip()
            if imdb_raw and _IMDB_ID_RE.match(imdb_raw):
                return _norm_imdb(imdb_raw)
    return None


def _chunked(values: list[str], size: int) -> Iterable[list[str]]:
    step = max(1, size)
    for i in range(0, len(values), step):
        yield values[i : i + step]


def _fetch_wikidata_labels(
    qids: list[str], language: str, fallback_language: str
) -> dict[str, WikidataEntity]:
    if not qids:
        return {}

    _m_inc("wikidata_labels_calls", 1)

    out: dict[str, WikidataEntity] = {}
    languages = (
        f"{language}|{fallback_language}"
        if fallback_language and fallback_language != language
        else language
    )

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
            resp = _http_get(
                str(WIKI_WIKIDATA_API_BASE_URL),
                params=params,
                timeout=_HTTP_TIMEOUT,
                breaker="wiki",
            )
        except _CircuitBreakerOpen:
            # Circuit abierto: no cuenta como failure y saltamos batch
            continue
        except RequestException as exc:
            _m_inc("wikidata_failures", 1)
            _cb_record_failure("wiki")
            _log_debug(f"wikidata.labels EXC: {exc!r}")
            continue

        if resp.status_code != 200:
            _m_inc("wikidata_failures", 1)
            _cb_record_failure("wiki")
            _log_debug(f"wikidata.labels <- status={resp.status_code}")
            continue

        try:
            payload = resp.json()
        except (ValueError, JSONDecodeError) as exc:
            _m_inc("wikidata_failures", 1)
            _cb_record_failure("wiki")
            _log_debug(f"wikidata.labels JSON EXC: {exc!r}")
            continue

        if not isinstance(payload, Mapping):
            _cb_record_failure("wiki")
            continue

        entities_obj = payload.get("entities")
        if not isinstance(entities_obj, Mapping):
            _cb_record_failure("wiki")
            continue

        _cb_record_success("wiki")

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
                df = (
                    descriptions_obj.get(fallback_language)
                    if fallback_language
                    else None
                )
                if isinstance(dp, Mapping):
                    desc = _safe_str(dp.get("value"))
                elif isinstance(df, Mapping):
                    desc = _safe_str(df.get("value"))

            out[qid] = _mk_entity({"label": label, "description": desc})

    return out


# =============================================================================
# SPARQL (con throttle global)
# =============================================================================


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


def _wikidata_sparql(query: str) -> Mapping[str, Any] | None:
    _m_inc("sparql_calls", 1)

    if _cb_is_open("wdqs"):
        _cb_short_circuit("wdqs")

    params = {"format": "json", "query": query}

    _sparql_throttle()
    _log_debug(f"wikidata.sparql -> len={len(query)}")

    try:
        resp = _http_get(
            str(WIKI_WDQS_URL),
            params=params,
            timeout=_HTTP_TIMEOUT_SPARQL,
            breaker="wdqs",
        )
    except _CircuitBreakerOpen:
        return None
    except RequestException as exc:
        _m_inc("sparql_failures", 1)
        _cb_record_failure("wdqs")
        _log_debug(f"wikidata.sparql EXC: {exc!r}")
        return None

    if resp.status_code != 200:
        _m_inc("sparql_failures", 1)
        _cb_record_failure("wdqs")
        _log_debug(f"wikidata.sparql <- status={resp.status_code}")
        return None

    try:
        data = resp.json()
    except (ValueError, JSONDecodeError) as exc:
        _m_inc("sparql_failures", 1)
        _cb_record_failure("wdqs")
        _log_debug(f"wikidata.sparql JSON EXC: {exc!r}")
        return None

    _cb_record_success("wdqs")
    return data if isinstance(data, Mapping) else None


# =============================================================================
# is_film (cacheado)
# =============================================================================


def _looks_like_film_from_wikipedia(wiki_raw: Mapping[str, Any], language: str) -> bool:
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


def _has_claim(entity: Mapping[str, Any], pid: str) -> bool:
    claims_obj = entity.get("claims")
    if not isinstance(claims_obj, Mapping):
        return False
    v = claims_obj.get(pid)
    return isinstance(v, list) and len(v) > 0


def _is_film_without_sparql(
    *,
    wd_entity: Mapping[str, Any],
    wiki_raw: Mapping[str, Any] | None,
    wiki_lang: str,
) -> bool:
    p31 = set(_extract_qids_from_claims(wd_entity, "P31"))

    if p31 & _FILM_INSTANCE_DENYLIST:
        return False
    if p31 & _FILM_INSTANCE_ALLOWLIST:
        return True

    has_director = _has_claim(wd_entity, "P57")
    has_cast = _has_claim(wd_entity, "P161")
    has_genre = _has_claim(wd_entity, "P136")
    has_date = _has_claim(wd_entity, "P577")
    has_duration = _has_claim(wd_entity, "P2047")

    if has_director and (has_cast or has_genre) and (has_date or has_duration):
        return True

    if (
        wiki_raw is not None
        and (not _is_disambiguation_payload(wiki_raw))
        and _looks_like_film_from_wikipedia(wiki_raw, wiki_lang)
    ):
        return True

    return False


def _is_film_cached_value(cache: WikiCacheFile, qid: str) -> bool | None:
    now_epoch = _now_epoch()
    is_film_map = cache.get("is_film")
    if not isinstance(is_film_map, Mapping):
        return None
    cached = is_film_map.get(qid)
    if cached is None:
        return None
    try:
        if not isinstance(cached, Mapping):
            return None
        if _is_expired(
            int(cached.get("fetched_at", 0) or 0),
            int(cached.get("ttl_s", 0) or 0),
            now_epoch,
        ):
            return None
        return bool(cached.get("is_film") is True)
    except Exception:
        return None


def _is_film_cached(
    *,
    cache: WikiCacheFile,
    qid: str,
    wd_entity: Mapping[str, Any],
    wiki_raw: Mapping[str, Any] | None,
    wiki_lang: str,
) -> bool:
    now_epoch = _now_epoch()

    is_film_map = cache.get("is_film")
    if not isinstance(is_film_map, dict):
        cache["is_film"] = {}
        is_film_map = cache["is_film"]

    cached = is_film_map.get(qid)
    if isinstance(cached, Mapping) and not _is_expired(
        int(cached.get("fetched_at", 0) or 0),
        int(cached.get("ttl_s", 0) or 0),
        now_epoch,
    ):
        return bool(cached.get("is_film") is True)

    ok = _is_film_without_sparql(
        wd_entity=wd_entity, wiki_raw=wiki_raw, wiki_lang=wiki_lang
    )
    is_film_map[qid] = _mk_is_film_entry(
        {"is_film": bool(ok), "fetched_at": now_epoch, "ttl_s": int(_TTL_IS_FILM_S)}
    )
    _mark_dirty_unlocked()
    # NO flush aquí (evita dependencia de lock)
    return ok


# =============================================================================
# imdbID -> QID (P345)
# =============================================================================


def _imdb_qid_cached_unlocked(
    cache: WikiCacheFile, imdb_norm: str
) -> tuple[bool, str | None]:
    """
    Devuelve (hit, qid).
    - hit=False si no existe o está expirado.
    - qid puede ser None (negative caching).
    """
    now_epoch = _now_epoch()
    imdb_qid_map = cache.get("imdb_qid")
    if not isinstance(imdb_qid_map, Mapping):
        return False, None

    cached = imdb_qid_map.get(imdb_norm)
    if cached is None:
        return False, None
    try:
        if not isinstance(cached, Mapping):
            return False, None
        if _is_expired(
            int(cached.get("fetched_at", 0) or 0),
            int(cached.get("ttl_s", 0) or 0),
            now_epoch,
        ):
            return False, None
        qid_obj = cached.get("qid")
        qid = _safe_str(qid_obj) if qid_obj is not None else None
        return True, qid
    except Exception:
        return False, None


def _imdb_qid_store_unlocked(
    cache: WikiCacheFile, imdb_norm: str, qid: str | None, *, ttl_s: int
) -> None:
    imdb_qid_map = cache.get("imdb_qid")
    if not isinstance(imdb_qid_map, dict):
        cache["imdb_qid"] = {}
        imdb_qid_map = cache["imdb_qid"]

    imdb_qid_map[imdb_norm] = _mk_imdb_qid_entry(
        {"qid": qid, "fetched_at": _now_epoch(), "ttl_s": int(ttl_s)}
    )
    _mark_dirty_unlocked()


def _resolve_qid_by_imdb_id(imdb_norm: str) -> str | None:
    if not imdb_norm:
        return None

    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        hit, qid_cached = _imdb_qid_cached_unlocked(cache, imdb_norm)
        if hit:
            return qid_cached

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

    qid: str | None = None
    if isinstance(bindings, list) and bindings:
        first = bindings[0]
        if isinstance(first, Mapping):
            item = first.get("item")
            if isinstance(item, Mapping):
                val = _safe_str(item.get("value"))
                if val:
                    m = re.search(r"/entity/(Q\d+)$", val)
                    qid = m.group(1) if m else None

    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        if qid:
            _imdb_qid_store_unlocked(cache, imdb_norm, qid, ttl_s=int(_TTL_OK_S))
        else:
            _imdb_qid_store_unlocked(
                cache, imdb_norm, None, ttl_s=int(_TTL_IMDB_QID_NEGATIVE_S)
            )

    return qid


def _extract_sitelink_title(entity: Mapping[str, Any], language: str) -> str | None:
    sitelinks = entity.get("sitelinks")
    if not isinstance(sitelinks, Mapping):
        return None
    sl = sitelinks.get(f"{language}wiki")
    if not isinstance(sl, Mapping):
        return None
    return _safe_str(sl.get("title"))


def _pick_best_sitelink_title(
    entity: Mapping[str, Any], languages: list[str]
) -> tuple[str | None, str]:
    for lang in languages:
        title = _extract_sitelink_title(entity, lang)
        if title:
            return title, lang
    return None, (
        languages[0] if languages else (_normalize_lang_code(WIKI_LANGUAGE) or "en")
    )


# =============================================================================
# Cache lookup/store + SWR
# =============================================================================


def _should_purge_title_mismatch(norm_title: str, item: WikiCacheItem) -> bool:
    if not norm_title:
        return False
    status = item.get("status")
    if not isinstance(status, str) or status != "ok":
        return False
    wiki_block = item.get("wiki")
    if not isinstance(wiki_block, Mapping):
        return False
    wiki_title = wiki_block.get("wikipedia_title")
    if not isinstance(wiki_title, str) or not wiki_title.strip():
        return False
    return not _titles_share_significant_tokens(norm_title, wiki_title)


def _purge_cached_record_unlocked(
    *, cache: WikiCacheFile, rid: str, item: WikiCacheItem
) -> None:
    records = cache.get("records")
    if isinstance(records, dict):
        records.pop(rid, None)

    idx_imdb = cache.get("index_imdb")
    if isinstance(idx_imdb, dict):
        imdb = _norm_imdb(
            item.get("imdbID") if isinstance(item.get("imdbID"), str) else None
        )
        if imdb and idx_imdb.get(imdb) == rid:
            idx_imdb.pop(imdb, None)
        for k, v in list(idx_imdb.items()):
            if v == rid:
                idx_imdb.pop(k, None)

    idx_ty = cache.get("index_ty")
    if isinstance(idx_ty, dict):
        title = item.get("Title")
        year = item.get("Year")
        if (
            isinstance(title, str)
            and isinstance(year, str)
            and idx_ty.get(_ty_key(title, year)) == rid
        ):
            idx_ty.pop(_ty_key(title, year), None)
        for k, v in list(idx_ty.items()):
            if v == rid:
                idx_ty.pop(k, None)

    _m_inc("cache_title_mismatch_purged", 1)
    _mark_dirty_unlocked()


def _get_cached_item(
    *,
    cache: WikiCacheFile,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
) -> tuple[WikiCacheItem | None, bool]:
    """
    Devuelve (item, served_stale_ok).
    - served_stale_ok=True sólo si status=ok y está en ventana SWR grace.

    Opción B:
      - Si norm_year está vacío, permitimos un lookup por título SOLO si
        existe exactamente 1 record con Title == norm_title (evita colisiones).
    """
    now_epoch = _now_epoch()

    records_obj = cache.get("records")
    idx_imdb_obj = cache.get("index_imdb")
    idx_ty_obj = cache.get("index_ty")
    if (
        not isinstance(records_obj, dict)
        or not isinstance(idx_imdb_obj, dict)
        or not isinstance(idx_ty_obj, dict)
    ):
        return None, False

    # ⚠️ Importante para Pyright:
    # - no tipar records como dict[str, WikiCacheItem] aquí, porque entonces `it` es siempre Mapping
    #   y los guards defensivos se vuelven "unreachable".
    records: dict[str, object] = cast(dict[str, object], records_obj)
    idx_imdb: dict[str, str] = cast(dict[str, str], idx_imdb_obj)
    idx_ty: dict[str, str] = cast(dict[str, str], idx_ty_obj)

    def _pick(rid: str) -> tuple[WikiCacheItem | None, bool]:
        raw = records.get(rid)
        if not isinstance(raw, Mapping):
            return None, False
        it = cast(WikiCacheItem, raw)

        fetched_at = int(it.get("fetched_at", 0) or 0)
        ttl_s = int(it.get("ttl_s", 0) or 0)

        if not _is_expired(fetched_at, ttl_s, now_epoch):
            return it, False

        _m_inc("cache_expired_hits", 1)

        if it.get("status") == "ok" and _SWR_OK_GRACE_S > 0:
            age = now_epoch - fetched_at
            if age <= (ttl_s + int(_SWR_OK_GRACE_S)):
                _m_inc("cache_stale_served", 1)
                return it, True

        return None, False

    def _should_bypass_title_cache_for_imdb(item: WikiCacheItem) -> bool:
        if not imdb_id:
            return False
        status = item.get("status")
        if not isinstance(status, str) or status == "ok":
            return False
        imdb_raw = item.get("imdbID")
        imdb_cached = _norm_imdb(imdb_raw) if isinstance(imdb_raw, str) else None
        return (not imdb_cached) or (imdb_cached != imdb_id)

    # 1) IMDb index
    if imdb_id:
        rid = idx_imdb.get(imdb_id)
        if isinstance(rid, str):
            it, stale = _pick(rid)
            if it is not None:
                return it, stale

    # 2) Title+Year index (solo si ambos no vacíos)
    if norm_title and norm_year:
        rid2 = idx_ty.get(_ty_key(norm_title, norm_year))
        if isinstance(rid2, str):
            it2, stale2 = _pick(rid2)
            if it2 is not None and not _should_bypass_title_cache_for_imdb(it2):
                if _should_purge_title_mismatch(norm_title, it2):
                    wiki_title = None
                    wiki_block = it2.get("wiki")
                    if isinstance(wiki_block, Mapping):
                        wiki_title = wiki_block.get("wikipedia_title")
                    _purge_cached_record_unlocked(cache=cache, rid=rid2, item=it2)
                    _log_debug(
                        "[cache] purged mismatched title/year entry | "
                        f"lookup={norm_title!r} wiki_title={wiki_title!r}"
                    )
                    return None, False
                return it2, stale2
        return None, False

    # 3) Opción B: si year vacío, usar SOLO si hay match único por Title
    if norm_title and not norm_year:
        found_rid: str | None = None
        for rid, raw in records.items():
            if not isinstance(raw, Mapping):
                continue

            t = raw.get("Title")
            if isinstance(t, str) and t == norm_title:
                if found_rid is not None:
                    # Ambiguo: más de uno
                    return None, False
                found_rid = rid

        if found_rid:
            it3, stale3 = _pick(found_rid)
            if it3 is not None and not _should_bypass_title_cache_for_imdb(it3):
                if _should_purge_title_mismatch(norm_title, it3):
                    wiki_title = None
                    wiki_block = it3.get("wiki")
                    if isinstance(wiki_block, Mapping):
                        wiki_title = wiki_block.get("wikipedia_title")
                    _purge_cached_record_unlocked(cache=cache, rid=found_rid, item=it3)
                    _log_debug(
                        "[cache] purged mismatched title-only entry | "
                        f"lookup={norm_title!r} wiki_title={wiki_title!r}"
                    )
                    return None, False
                return it3, stale3

    return None, False


def _rid_for_item(item: WikiCacheItem) -> str:
    imdb = _norm_imdb(
        item.get("imdbID") if isinstance(item.get("imdbID"), str) else None
    )
    if imdb:
        return f"imdb:{imdb}"
    title = item.get("Title")
    year = item.get("Year")
    t = title if isinstance(title, str) else ""
    y = year if isinstance(year, str) else ""
    return f"ty:{t}|{y}"


def _store_item_unlocked(cache: WikiCacheFile, item: WikiCacheItem) -> None:
    """
    FIX: Solo index_ty si Title y Year NO están vacíos.
    Evita la key colisionable "title|".
    """
    if not isinstance(cache.get("records"), dict):
        cache["records"] = {}
    if not isinstance(cache.get("index_ty"), dict):
        cache["index_ty"] = {}
    if not isinstance(cache.get("index_imdb"), dict):
        cache["index_imdb"] = {}

    records = cast(dict[str, WikiCacheItem], cache["records"])
    idx_ty = cast(dict[str, str], cache["index_ty"])
    idx_imdb = cast(dict[str, str], cache["index_imdb"])

    rid = _rid_for_item(item)
    records[rid] = item

    title = item.get("Title")
    year = item.get("Year")
    if (
        isinstance(title, str)
        and isinstance(year, str)
        and title.strip()
        and year.strip()
    ):
        idx_ty[_ty_key(title, year)] = rid

    imdb = _norm_imdb(
        item.get("imdbID") if isinstance(item.get("imdbID"), str) else None
    )
    if imdb:
        idx_imdb[imdb] = rid

    _m_inc("cache_store_writes", 1)
    _mark_dirty_unlocked()


# =============================================================================
# Builders
# =============================================================================


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

    wiki_block = _mk_wiki_block(
        {
            "language": primary_language,
            "fallback_language": fallback_language,
            "source_language": "",
            "wikipedia_title": None,
            "wikipedia_pageid": None,
            "wikibase_item": wikibase_item,
            "summary": "",
            "description": "",
            "urls": {},
        }
    )

    wikidata_block = _mk_wikidata_block({})
    if wikibase_item:
        wikidata_block["qid"] = wikibase_item

    _m_inc("items_negative", 1)
    ttl = int(_TTL_NEGATIVE_S)

    if status == "imdb_no_qid":
        _m_inc("items_negative_imdb_no_qid", 1)
    elif status == "no_qid":
        _m_inc("items_negative_no_qid", 1)
    elif status == "not_film":
        _m_inc("items_negative_not_film", 1)
    elif status == "disambiguation":
        _m_inc("items_negative_disambiguation", 1)
        ttl = int(_TTL_DISAMBIG_S)

    return _mk_cache_item(
        {
            "Title": norm_title,
            "Year": norm_year,
            "imdbID": imdb_id,
            "wiki": wiki_block,
            "wikidata": wikidata_block,
            "fetched_at": now_epoch,
            "ttl_s": ttl,
            "status": status,
        }
    )


def _build_ok_item_and_merge_entities(
    *,
    cache: WikiCacheFile,
    norm_title: str,
    norm_year: str,
    imdb_id: str | None,
    wiki_raw: Mapping[str, Any],
    source_language: str,
    wikibase_item: str,
    wd_entity: Mapping[str, Any],
    primary_language: str,
    fallback_language: str,
) -> WikiCacheItem:
    titles_obj = wiki_raw.get("titles")
    wikipedia_title: str | None = None
    if isinstance(titles_obj, Mapping):
        wikipedia_title = _safe_str(titles_obj.get("normalized")) or _safe_str(
            titles_obj.get("canonical")
        )

    summary_text = str(wiki_raw.get("extract") or "")
    description_text = str(wiki_raw.get("description") or "")

    urls_content: dict[str, Any] = {}
    content_urls_obj = wiki_raw.get("content_urls")
    if isinstance(content_urls_obj, Mapping):
        for k, v in content_urls_obj.items():
            if isinstance(k, str):
                urls_content[k] = v

    wiki_block = _mk_wiki_block(
        {
            "language": primary_language,
            "fallback_language": fallback_language,
            "source_language": source_language,
            "wikipedia_title": wikipedia_title,
            "wikipedia_pageid": _safe_int(wiki_raw.get("pageid")),
            "wikibase_item": wikibase_item,
            "summary": summary_text,
            "description": description_text,
            "urls": urls_content,
        }
    )

    has_images = False
    if "originalimage" in wiki_raw or "thumbnail" in wiki_raw:
        images: dict[str, Any] = {}
        original = wiki_raw.get("originalimage")
        thumb = wiki_raw.get("thumbnail")
        if isinstance(original, Mapping):
            images["original"] = dict(original)
            has_images = True
        if isinstance(thumb, Mapping):
            images["thumbnail"] = dict(thumb)
            has_images = True
        if images:
            wiki_block["images"] = images

    wikidata_block = _mk_wikidata_block({"qid": wikibase_item})

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

    if not isinstance(cache.get("entities"), dict):
        cache["entities"] = {}
    entities_map = cast(dict[str, WikidataEntity], cache["entities"])

    for qid, ent in labeled.items():
        etype: str | None = None
        if qid in directors:
            etype = "person"
        elif qid in countries:
            etype = "country"
        elif qid in genres:
            etype = "genre"
        merged = _mk_entity(ent)
        if etype:
            merged["type"] = etype
        entities_map[qid] = merged

    _m_inc("items_ok", 1)
    if directors:
        _m_inc("items_ok_with_directors", 1)
    if genres:
        _m_inc("items_ok_with_genres", 1)
    if countries:
        _m_inc("items_ok_with_countries", 1)
    if has_images:
        _m_inc("items_ok_with_images", 1)
    if summary_text.strip():
        _m_inc("items_ok_with_summary", 1)
    if description_text.strip():
        _m_inc("items_ok_with_description", 1)

    item = _mk_cache_item(
        {
            "Title": norm_title,
            "Year": norm_year,
            "imdbID": imdb_id,
            "wiki": wiki_block,
            "wikidata": wikidata_block,
            "fetched_at": _now_epoch(),
            "ttl_s": int(_TTL_OK_S),
            "status": "ok",
        }
    )

    year_label = norm_year if norm_year else "?"
    _log_info(f"[WIKI] cached ({source_language}): {norm_title} ({year_label})")

    return item


# =============================================================================
# Title/year path
# =============================================================================


def _try_title_year_path(
    *,
    lookup_title: str,
    norm_title: str,
    norm_year: str,
    year: int | None,
    imdb_id_for_store: str | None,
    primary: str,
    fallback: str,
) -> WikiCacheItem | None:
    """
    Path title/year:
      - Genera candidatos via wikipedia.search (cacheado)
      - Prueba wikipedia REST summary por candidato/lang
      - Si obtiene wikibase_item, valida is_film (cacheable)
      - En ok: enriquece labels (directors/countries/genres) y guarda en cache.
      - En negativo: guarda item negativo (no_qid / disambiguation / not_film).

    FIX:
      - Si norm_year está vacío, intenta extraer year desde Wikidata (P577).
      - Usa year_final al construir OK y negativos not_film.
    """
    disambig_seen = False
    any_non_disambig_summary = False

    def _try_candidates(cands: list[str], cand_lang: str) -> WikiCacheItem | None:
        nonlocal disambig_seen, any_non_disambig_summary

        for cand_title in cands:
            if not _titles_share_significant_tokens(lookup_title, cand_title):
                continue
            wiki_raw = _fetch_wikipedia_summary_by_title(cand_title, cand_lang)
            if wiki_raw is None:
                continue

            if _is_disambiguation_payload(wiki_raw):
                disambig_seen = True
                continue

            any_non_disambig_summary = True

            wikibase_item = _safe_str(wiki_raw.get("wikibase_item"))
            if not wikibase_item:
                continue

            # fast path: is_film cached value check (NO network)
            with _CACHE_LOCK:
                cache = _load_cache_unlocked()
                cached_is_film = _is_film_cached_value(cache, wikibase_item)

            if cached_is_film is False:
                # Si ya sabemos que NO es film, cacheamos "not_film" y seguimos con otros candidatos.
                # year_final: si no venía year, intentaremos rellenarlo al tener Wikibase item.
                year_final_sc = norm_year
                if not year_final_sc:
                    # no hacemos red aquí; solo con wd_entity. Pero aún no lo tenemos.
                    # Guardamos con norm_year (vacío) para evitar coste extra.
                    pass

                with _CACHE_LOCK:
                    cache = _load_cache_unlocked()
                    neg_nf_sc = _build_negative_item(
                        norm_title=norm_title,
                        norm_year=year_final_sc,
                        imdb_id=imdb_id_for_store,
                        status="not_film",
                        wikibase_item=wikibase_item,
                        primary_language=primary,
                        fallback_language=fallback,
                    )
                    _store_item_unlocked(cache, neg_nf_sc)
                continue

            wd_entity = _fetch_wikidata_entity_json(wikibase_item)
            if wd_entity is None:
                continue

            imdb_from_wd = _extract_imdb_id_from_claims(wd_entity)
            imdb_for_store = imdb_id_for_store or imdb_from_wd

            # FIX: year_final desde Wikidata si no venía year
            year_final = norm_year
            if not year_final:
                year_from_wd = _extract_year_from_wd_entity(wd_entity)
                if year_from_wd:
                    year_final = year_from_wd

            with _CACHE_LOCK:
                cache = _load_cache_unlocked()

                # valida is_film (cacheable)
                if not _is_film_cached(
                    cache=cache,
                    qid=wikibase_item,
                    wd_entity=wd_entity,
                    wiki_raw=wiki_raw,
                    wiki_lang=cand_lang,
                ):
                    neg_nf = _build_negative_item(
                        norm_title=norm_title,
                        norm_year=year_final,  # <-- FIX: usar year_final
                        imdb_id=imdb_for_store,
                        status="not_film",
                        wikibase_item=wikibase_item,
                        primary_language=primary,
                        fallback_language=fallback,
                    )
                    _store_item_unlocked(cache, neg_nf)
                    continue

                # OK: construye item + entities labels
                ok = _build_ok_item_and_merge_entities(
                    cache=cache,
                    norm_title=norm_title,
                    norm_year=year_final,  # <-- FIX: usar year_final
                    imdb_id=imdb_for_store,
                    wiki_raw=wiki_raw,
                    source_language=cand_lang,
                    wikibase_item=wikibase_item,
                    wd_entity=wd_entity,
                    primary_language=primary,
                    fallback_language=fallback,
                )
                _store_item_unlocked(cache, ok)
                return ok

        return None

    primary_candidates = _rank_wikipedia_candidates(
        lookup_title=lookup_title,
        year=year,
        language=primary,
    )[:8]
    _log_debug(
        f"[title/year] candidates={len(primary_candidates)} primary={primary} fallback={fallback}"
    )

    ok_primary = _try_candidates(primary_candidates, primary)
    if ok_primary is not None:
        return ok_primary

    if fallback and fallback != primary:
        fallback_candidates = _rank_wikipedia_candidates(
            lookup_title=lookup_title,
            year=year,
            language=fallback,
        )[:8]
        _log_debug(
            f"[title/year] fallback candidates={len(fallback_candidates)} fallback={fallback}"
        )
        ok_fallback = _try_candidates(fallback_candidates, fallback)
        if ok_fallback is not None:
            return ok_fallback

    # si TODOS los summaries fueron disambiguation (y ninguno no-disambig), guardamos status=disambiguation
    final_status: WikiItemStatus = "no_qid"
    if disambig_seen and not any_non_disambig_summary:
        final_status = "disambiguation"

    neg = _build_negative_item(
        norm_title=norm_title,
        norm_year=norm_year,
        imdb_id=imdb_id_for_store,
        status=final_status,
        wikibase_item=None,
        primary_language=primary,
        fallback_language=fallback,
    )
    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        _store_item_unlocked(cache, neg)
    return neg


# =============================================================================
# SWR refresh scheduler
# =============================================================================


def _schedule_refresh_if_needed(
    *,
    key: str,
    title: str,
    year: int | None,
    imdb_id: str | None,
    movie_input: MovieInputLangProto | None,
    primary: str,
    fallback: str,
) -> None:
    # Evitar tormenta de threads: solo 1 refresh por key.
    with _REFRESH_IN_FLIGHT_LOCK:
        if key in _REFRESH_IN_FLIGHT:
            return
        _REFRESH_IN_FLIGHT.add(key)

    # Si todo está claramente caído (ambos circuitos abiertos), no merece la pena spawnear.
    # (No es una condición de error; SWR ya sirvió stale y reintentará en futuras lecturas.)
    try:
        if _cb_is_open("wiki") and _cb_is_open("wdqs"):
            with _REFRESH_IN_FLIGHT_LOCK:
                _REFRESH_IN_FLIGHT.discard(key)
            return
    except Exception:
        # best-effort
        pass

    def _worker() -> None:
        try:
            _get_wiki_impl(
                title=title,
                year=year,
                imdb_id=imdb_id,
                movie_input=movie_input,
                primary=primary,
                fallback=fallback,
                force_refresh=True,
            )
        except Exception as exc:
            _log_debug(f"SWR refresh failed: {exc!r}")
        finally:
            with _REFRESH_IN_FLIGHT_LOCK:
                _REFRESH_IN_FLIGHT.discard(key)

    try:
        threading.Thread(
            target=_worker, name=f"wiki-refresh:{key}", daemon=True
        ).start()
    except Exception as exc:
        _log_debug(f"refresh thread spawn failed: {exc!r}")
        with _REFRESH_IN_FLIGHT_LOCK:
            _REFRESH_IN_FLIGHT.discard(key)


# =============================================================================
# Core (single-flight + SWR)
# =============================================================================


def _get_wiki_impl(
    *,
    title: str,
    year: int | None,
    imdb_id: str | None,
    movie_input: MovieInputLangProto | None,
    primary: str,
    fallback: str,
    force_refresh: bool = False,
) -> WikiCacheItem | None:
    raw_title = title.strip()
    lookup_title = normalize_title_for_lookup(raw_title)

    # Fallback defensivo si el normalizador deja el título vacío
    if not lookup_title:
        if raw_title:
            _log_debug(
                f"normalize_title_for_lookup produced empty; fallback to raw_title={raw_title!r}"
            )
            lookup_title = raw_title
        else:
            _log_debug(f"empty lookup_title from title={title!r}")
            return None

    norm_title = lookup_title
    norm_year = str(year) if year is not None else ""
    imdb_norm = _norm_imdb(imdb_id)

    sf_key = _request_key_for_singleflight(
        imdb_norm=imdb_norm,
        norm_title=norm_title,
        norm_year=norm_year,
    )

    # ------------------------------------------------------------------
    # Cache read (sin refresh)
    # ------------------------------------------------------------------
    if not force_refresh:
        with _CACHE_LOCK:
            cache = _load_cache_unlocked()
            cached, served_stale_ok = _get_cached_item(
                cache=cache,
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=imdb_norm,
            )
            if cached is not None:
                _m_inc("cache_hits", 1)
                _log_debug("cache HIT" + (" (stale-ok)" if served_stale_ok else ""))
                if served_stale_ok:
                    _schedule_refresh_if_needed(
                        key=sf_key,
                        title=title,
                        year=year,
                        imdb_id=imdb_id,
                        movie_input=movie_input,
                        primary=primary,
                        fallback=fallback,
                    )
                return cached

    _m_inc("cache_misses", 1)

    # ------------------------------------------------------------------
    # Single-flight
    # ------------------------------------------------------------------
    is_leader, ev = _singleflight_enter(sf_key)

    if not is_leader:
        _m_inc("singleflight_waits", 1)
        ev.wait(timeout=float(_SINGLEFLIGHT_WAIT_S))
        if not ev.is_set():
            _m_inc("singleflight_wait_timeouts", 1)

        with _CACHE_LOCK:
            cache = _load_cache_unlocked()
            cached2, served_stale_ok2 = _get_cached_item(
                cache=cache,
                norm_title=norm_title,
                norm_year=norm_year,
                imdb_id=imdb_norm,
            )
            if cached2 is not None:
                _m_inc("cache_hits", 1)
                _log_debug("cache HIT (after single-flight wait)")
                if served_stale_ok2:
                    _schedule_refresh_if_needed(
                        key=sf_key,
                        title=title,
                        year=year,
                        imdb_id=imdb_id,
                        movie_input=movie_input,
                        primary=primary,
                        fallback=fallback,
                    )
                return cached2

        _log_debug("single-flight leader produced no cache; continuing best-effort")

    def _leader_finalize() -> None:
        try:
            with _CACHE_LOCK:
                _maybe_flush_unlocked(force=False)
        except Exception:
            pass

    try:
        # ==============================================================
        # Path 1: IMDb → QID
        # ==============================================================
        if imdb_norm:
            _m_inc("path_imdb", 1)

            qid = _resolve_qid_by_imdb_id(imdb_norm)
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

            # fast-path is_film cache
            with _CACHE_LOCK:
                cache = _load_cache_unlocked()
                cached_is_film = _is_film_cached_value(cache, qid)

            if cached_is_film is False:
                # (year_final no se puede extraer aquí sin wd_entity; usamos norm_year)
                neg_nf = _build_negative_item(
                    norm_title=norm_title,
                    norm_year=norm_year,
                    imdb_id=imdb_norm,
                    status="not_film",
                    wikibase_item=qid,
                    primary_language=primary,
                    fallback_language=fallback,
                )
                with _CACHE_LOCK:
                    cache = _load_cache_unlocked()
                    _store_item_unlocked(cache, neg_nf)
                return neg_nf

            wd_entity = _fetch_wikidata_entity_json(qid)
            if wd_entity is None:
                return None

            # ✅ FIX: si year no venía, intenta derivarlo desde P577 y úsalo para OK / not_film
            year_final = norm_year
            if not year_final:
                year_from_wd = _extract_year_from_wd_entity(wd_entity)
                if year_from_wd:
                    year_final = year_from_wd

            chain = _detect_language_chain_from_input(movie_input)
            sl_title, sl_lang = _pick_best_sitelink_title(wd_entity, chain)
            wiki_raw = (
                _fetch_wikipedia_summary_by_title(sl_title, sl_lang)
                if sl_title
                else None
            )

            if wiki_raw is None or _is_disambiguation_payload(wiki_raw):
                _log_debug(
                    "[imdb->qid] missing/ambiguous summary; trying title/year fallback"
                )
                _m_inc("path_title_search", 1)
                return _try_title_year_path(
                    lookup_title=lookup_title,
                    norm_title=norm_title,
                    norm_year=norm_year,
                    year=year,
                    imdb_id_for_store=imdb_norm,
                    primary=primary,
                    fallback=fallback,
                )

            with _CACHE_LOCK:
                cache = _load_cache_unlocked()

                if not _is_film_cached(
                    cache=cache,
                    qid=qid,
                    wd_entity=wd_entity,
                    wiki_raw=wiki_raw,
                    wiki_lang=sl_lang,
                ):
                    # ✅ FIX: usar year_final
                    neg_nf = _build_negative_item(
                        norm_title=norm_title,
                        norm_year=year_final,
                        imdb_id=imdb_norm,
                        status="not_film",
                        wikibase_item=qid,
                        primary_language=primary,
                        fallback_language=fallback,
                    )
                    _store_item_unlocked(cache, neg_nf)
                    return neg_nf

                # ✅ FIX: usar year_final
                ok = _build_ok_item_and_merge_entities(
                    cache=cache,
                    norm_title=norm_title,
                    norm_year=year_final,
                    imdb_id=imdb_norm,
                    wiki_raw=wiki_raw,
                    source_language=sl_lang,
                    wikibase_item=qid,
                    wd_entity=wd_entity,
                    primary_language=primary,
                    fallback_language=fallback,
                )
                _store_item_unlocked(cache, ok)
                return ok

        # ==============================================================
        # Path 2: Title / Year
        # ==============================================================
        _m_inc("path_title_search", 1)
        return _try_title_year_path(
            lookup_title=lookup_title,
            norm_title=norm_title,
            norm_year=norm_year,
            year=year,
            imdb_id_for_store=None,
            primary=primary,
            fallback=fallback,
        )

    finally:
        if is_leader:
            try:
                _leader_finalize()
            finally:
                _singleflight_leave(sf_key)


# =============================================================================
# API pública
# =============================================================================


def get_wiki(
    *,
    title: str,
    year: int | None,
    imdb_id: str | None,
    force_refresh: bool = False,
) -> WikiCacheItem | None:
    primary = _normalize_lang_code(WIKI_LANGUAGE) or "en"
    fallback = _normalize_lang_code(WIKI_FALLBACK_LANGUAGE) or (
        "en" if primary != "en" else ""
    )
    return _get_wiki_impl(
        title=title,
        year=year,
        imdb_id=imdb_id,
        movie_input=None,
        primary=primary,
        fallback=fallback,
        force_refresh=bool(force_refresh),
    )


def get_wiki_for_input(
    *,
    movie_input: MovieInputLangProto,
    title: str,
    year: int | None,
    imdb_id: str | None,
    force_refresh: bool = False,
) -> WikiCacheItem | None:
    primary, fallback = _best_wikipedia_languages_for_item(movie_input)
    return _get_wiki_impl(
        title=title,
        year=year,
        imdb_id=imdb_id,
        movie_input=movie_input,
        primary=primary,
        fallback=fallback,
        force_refresh=bool(force_refresh),
    )


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "get_wiki",
    "get_wiki_for_input",
    "flush_wiki_cache",
    "get_wiki_metrics_snapshot",
    "reset_wiki_metrics",
    "log_wiki_metrics_summary",
    "WikiCacheItem",
    "WikiItemStatus",
]
