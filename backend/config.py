from __future__ import annotations

"""
backend/config.py

Carga y normaliza la configuración del proyecto desde variables de entorno (.env).

🎯 Principios
-------------
1) "Config as data":
   - Este módulo SOLO parsea, valida y expone constantes.
   - No ejecuta lógica de negocio.

2) Robusto ante entornos “sucios”:
   - Si una env var viene mal (p.ej. "abc" donde se esperaba int), no rompe.
   - Emite warning always=True (visible incluso en modo SILENT).

3) Logging coherente con backend/logger.py:
   - warnings always=True cuando una env var es inválida / cap.
   - dump de config solo si DEBUG_MODE y NO SILENT_MODE (para no spamear).

⚠️ Nota técnica importante
--------------------------
Este archivo se importa desde casi todo el proyecto, así que:
- Evitamos dependencias circulares y efectos secundarios caros.
- Evitamos efectos secundarios innecesarios.
- Definimos PATHS base (BASE_DIR/DATA_DIR) muy pronto para que estén disponibles.

OMDb / Wiki / Collection / Plex
-------------------------------
Centraliza parámetros de clientes y orquestadores:
- TTLs, throttles, batching, flush
- Paths
- Caps (compaction / in-memory LRU)
- Flags de comportamiento

✅ Mejoras
----------
- Plex Metrics: parse + caps.
- OMDb HTTP tuning: timeouts/retries/user-agent.
- Wiki HTTP tuning: timeouts/retries/user-agent + endpoints.
- Wiki Metrics: parse + caps, consumido por backend/wiki_client.py.

✅ NUEVO (orquestador Plex)
--------------------------
Añade knobs del orquestador Plex (analiza_plex.py), centralizables:

1) PLEX_PROGRESS_EVERY_N_MOVIES
2) PLEX_MAX_WORKERS_CAP
3) PLEX_MAX_INFLIGHT_FACTOR
4) PLEX_LIBRARY_LANGUAGE_DEFAULT
5) PLEX_LIBRARY_LANGUAGE_BY_NAME
6) PLEX_RUN_METRICS_ENABLED

✅ NUEVO (core de análisis)
--------------------------
Centraliza knobs del core (backend/analyze_input_core.py):

7) ANALYZE_TRACE_REASON_MAX_CHARS
8) ANALYZE_CORE_METRICS_ENABLED

✅ NUEVO (collection_analysis.py)
--------------------------------
Centraliza knobs del orquestador por item:

9)  COLLECTION_OMDB_JSON_MODE
10) COLLECTION_LAZY_WIKI_ALLOW_TITLE_YEAR_FALLBACK
11) COLLECTION_LAZY_WIKI_FORCE_OMDB_POST_CORE
12) COLLECTION_TRACE_ALSO_DEBUG_CTX

✅ NUEVO (movie_input.py)
------------------------
Centraliza knobs "de política" (no de algoritmo) para:
- normalización de títulos para lookup
- heurística de idioma de contexto

Estas flags NO deberían cambiar resultados “core” de scoring,
pero sí pueden modular:
- si el lookup limpia más/menos ruido
- el umbral de function words para inferir idioma
- evitar declarar inglés cuando hay escritura CJK

✅ NUEVO (logger.py)
-------------------
Persistencia opcional del log de ejecución a fichero (además de consola),
con nombre generado por ejecución (timestamp + PID) controlado aquí.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# En producción suele ser deseable NO sobre-escribir env vars ya definidas.
# Si quieres que .env siempre gane, cambia a override=True.
load_dotenv(override=False)

from backend import logger as _logger  # noqa: E402


# ============================================================
# Paths base (DEBEN definirse pronto)
# ============================================================

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "data"


# ============================================================
# Helpers: parseo defensivo de env vars
# ============================================================

_TRUE_SET: Final[set[str]] = {"1", "true", "t", "yes", "y", "on"}
_FALSE_SET: Final[set[str]] = {"0", "false", "f", "no", "n", "off"}


def _clean_env_raw(v: object | None) -> str | None:
    """
    Normaliza un valor de env “raw”:
    - None -> None
    - str  -> strip() y elimina comillas exteriores típicas.

    Esto evita sorpresas comunes en .env (p.ej. valores con comillas).
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    # Toleramos casos como '"reports"' o "'reports'"
    if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        s = s[1:-1].strip()
    return s or None


def _get_env_str(name: str, default: str | None = None) -> str | None:
    """
    Lee un string desde env de forma tolerante.
    - Si está ausente/ vacío => default
    - Si viene con comillas externas típicas => las elimina (_clean_env_raw)
    """
    v = _clean_env_raw(os.getenv(name))
    if v is None:
        return default
    return v


def _get_env_int(name: str, default: int) -> int:
    """
    Lee un int desde env de forma tolerante.
    - Si no parsea => warning(always=True) y default
    """
    v = _clean_env_raw(os.getenv(name))
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        _logger.warning(
            f"Invalid int for {name!r}: {v!r}, using default {default}",
            always=True,
        )
        return default


def _get_env_float(name: str, default: float) -> float:
    """
    Lee un float desde env de forma tolerante.
    - Si no parsea => warning(always=True) y default
    """
    v = _clean_env_raw(os.getenv(name))
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        _logger.warning(
            f"Invalid float for {name!r}: {v!r}, using default {default}",
            always=True,
        )
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    """
    Parseo tolerante:
      - True:  1/true/t/yes/y/on
      - False: 0/false/f/no/n/off

    Si viene algo distinto => warning(always=True) y default.
    """
    v = _clean_env_raw(os.getenv(name))
    if v is None:
        return default

    s = v.strip().lower()
    if s in _TRUE_SET:
        return True
    if s in _FALSE_SET:
        return False

    _logger.warning(
        f"Invalid bool for {name!r}: {v!r}, using default {default}",
        always=True,
    )
    return default


def _get_env_enum_str(
    name: str,
    *,
    default: str,
    allowed: set[str],
    normalize: bool = True,
) -> str:
    """
    Lee un string tipo "enum" desde env con validación (best-effort).

    Casos:
    - env ausente => default
    - env inválida => warning(always=True) + default
    """
    raw = _get_env_str(name, None)
    if raw is None:
        return default

    s = raw.strip()
    if normalize:
        s = s.lower()

    if s in allowed:
        return s

    _logger.warning(
        f"Invalid value for {name!r}: {raw!r}. Allowed={sorted(allowed)}. Using default {default!r}.",
        always=True,
    )
    return default


def _cap_int(name: str, value: int, *, min_v: int, max_v: int) -> int:
    """
    Asegura que value esté en [min_v, max_v].

    Política:
    - Nunca rompe.
    - Emite warning always=True si fuerza/capea.
    """
    if value < min_v:
        _logger.warning(f"{name} < {min_v}; forcing to {min_v}", always=True)
        return min_v
    if value > max_v:
        _logger.warning(f"{name} too high; capping to {max_v}", always=True)
        return max_v
    return value


def _cap_float_min(name: str, value: float, *, min_v: float) -> float:
    """Asegura value >= min_v."""
    if value < min_v:
        _logger.warning(f"{name} < {min_v}; forcing to {min_v}", always=True)
        return min_v
    return value


def _log_config_debug(label: str, value: object) -> None:
    """
    Dump de config solo si DEBUG_MODE=True y SILENT_MODE=False.
    """
    if not DEBUG_MODE or SILENT_MODE:
        return
    try:
        _logger.info(f"{label}: {value}")
    except Exception:
        print(f"{label}: {value}")


def _parse_env_kv_map(raw: str) -> dict[str, str]:
    """
    Parsea un mapping str->str desde env.

    Formatos aceptados (best-effort):
    1) JSON (recomendado):
        {"Movies":"es","Kids":"en"}
    2) Pares separados por coma:
        Movies:es,Kids:en
    """
    out: dict[str, str] = {}
    cleaned = raw.strip().strip('"').strip("'").strip()
    if not cleaned:
        return out

    # 1) Intento JSON primero (más explícito y sin ambigüedades)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    ks = str(k).strip()
                    vs = str(v).strip()
                    if ks and vs:
                        out[ks] = vs
            else:
                _logger.warning(
                    f"Invalid dict for env map: expected JSON object, got {type(obj).__name__}",
                    always=True,
                )
            return out
        except Exception as exc:
            _logger.warning(
                f"Invalid JSON for env map; falling back to 'k:v' parsing. err={exc!r}",
                always=True,
            )

    # 2) Fallback: "k:v,k:v"
    for part in cleaned.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            _logger.warning(f"Invalid map chunk (missing ':') ignored: {chunk!r}", always=True)
            continue
        k, v = chunk.split(":", 1)
        ks = k.strip()
        vs = v.strip()
        if not ks or not vs:
            _logger.warning(f"Invalid map chunk (empty key/value) ignored: {chunk!r}", always=True)
            continue
        out[ks] = vs

    return out


# ============================================================
# MODO DE EJECUCIÓN
# ============================================================

DEBUG_MODE: bool = _get_env_bool("DEBUG_MODE", False)
SILENT_MODE: bool = _get_env_bool("SILENT_MODE", False)

HTTP_DEBUG: bool = _get_env_bool("HTTP_DEBUG", False)
LOG_LEVEL: str | None = _get_env_str("LOG_LEVEL", None)


# ============================================================
# Reports (paths) — temprano, pero ya tenemos BASE_DIR
# ============================================================

_REPORTS_DIR_RAW: Final[str] = _get_env_str("REPORTS_DIR", "reports") or "reports"
_REPORTS_DIR_PATH_CANDIDATE = Path(_REPORTS_DIR_RAW)
REPORTS_DIR_PATH: Final[Path] = (
    _REPORTS_DIR_PATH_CANDIDATE
    if _REPORTS_DIR_PATH_CANDIDATE.is_absolute()
    else (BASE_DIR / _REPORTS_DIR_PATH_CANDIDATE)
)


# ============================================================
# PLEX
# ============================================================

BASEURL: str | None = _get_env_str("BASEURL", None)
PLEX_PORT: int = _cap_int("PLEX_PORT", _get_env_int("PLEX_PORT", 32400), min_v=1, max_v=65535)
PLEX_TOKEN: str | None = _get_env_str("PLEX_TOKEN", None)

_raw_exclude_plex: str = _get_env_str("EXCLUDE_PLEX_LIBRARIES", "") or ""
EXCLUDE_PLEX_LIBRARIES: list[str] = [x.strip() for x in _raw_exclude_plex.split(",") if x.strip()]

PLEX_ANALYZE_WORKERS: int = _cap_int(
    "PLEX_ANALYZE_WORKERS",
    _get_env_int("PLEX_ANALYZE_WORKERS", 8),
    min_v=1,
    max_v=64,
)

PLEX_PROGRESS_EVERY_N_MOVIES: int = _cap_int(
    "PLEX_PROGRESS_EVERY_N_MOVIES",
    _get_env_int("PLEX_PROGRESS_EVERY_N_MOVIES", 100),
    min_v=1,
    max_v=10_000,
)

PLEX_MAX_WORKERS_CAP: int = _cap_int(
    "PLEX_MAX_WORKERS_CAP",
    _get_env_int("PLEX_MAX_WORKERS_CAP", 64),
    min_v=1,
    max_v=512,
)

PLEX_MAX_INFLIGHT_FACTOR: int = _cap_int(
    "PLEX_MAX_INFLIGHT_FACTOR",
    _get_env_int("PLEX_MAX_INFLIGHT_FACTOR", 4),
    min_v=1,
    max_v=50,
)

PLEX_LIBRARY_LANGUAGE_DEFAULT: str = _get_env_str("PLEX_LIBRARY_LANGUAGE_DEFAULT", "es") or "es"

_PLEX_LIBRARY_LANGUAGE_BY_NAME_RAW: str = _get_env_str("PLEX_LIBRARY_LANGUAGE_BY_NAME", "") or ""
PLEX_LIBRARY_LANGUAGE_BY_NAME: dict[str, str] = _parse_env_kv_map(_PLEX_LIBRARY_LANGUAGE_BY_NAME_RAW)

PLEX_RUN_METRICS_ENABLED: bool = _get_env_bool("PLEX_RUN_METRICS_ENABLED", True)

PLEX_METRICS_ENABLED: bool = _get_env_bool("PLEX_METRICS_ENABLED", True)
PLEX_METRICS_TOP_N: int = _cap_int(
    "PLEX_METRICS_TOP_N",
    _get_env_int("PLEX_METRICS_TOP_N", 5),
    min_v=1,
    max_v=50,
)
PLEX_METRICS_LOG_ON_SILENT_DEBUG: bool = _get_env_bool("PLEX_METRICS_LOG_ON_SILENT_DEBUG", True)
PLEX_METRICS_LOG_EVEN_IF_ZERO: bool = _get_env_bool("PLEX_METRICS_LOG_EVEN_IF_ZERO", False)


# ============================================================
# MOVIE_INPUT (normalización + heurística idioma)
# ============================================================

MOVIE_INPUT_LOOKUP_STRIP_ACCENTS: bool = _get_env_bool("MOVIE_INPUT_LOOKUP_STRIP_ACCENTS", True)
MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE: bool = _get_env_bool("MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE", True)
MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP: bool = _get_env_bool("MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP", True)

MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS: int = _cap_int(
    "MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS",
    _get_env_int("MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS", 2),
    min_v=0,
    max_v=10,
)

MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK: bool = _get_env_bool(
    "MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK",
    True,
)


# ============================================================
# COLLECTION (orquestación por item): caches in-memory / trazas
# ============================================================

COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS: int = _cap_int(
    "COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS",
    _get_env_int("COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS", 1200),
    min_v=0,
    max_v=500_000,
)

COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS: int = _cap_int(
    "COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS",
    _get_env_int("COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS", 2400),
    min_v=0,
    max_v=1_000_000,
)

COLLECTION_TRACE_LINE_MAX_CHARS: int = _cap_int(
    "COLLECTION_TRACE_LINE_MAX_CHARS",
    _get_env_int("COLLECTION_TRACE_LINE_MAX_CHARS", 220),
    min_v=60,
    max_v=5000,
)

COLLECTION_ENABLE_LAZY_WIKI: bool = _get_env_bool("COLLECTION_ENABLE_LAZY_WIKI", True)

COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE: bool = _get_env_bool(
    "COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE",
    True,
)

_COLLECTION_OMDB_JSON_MODE_ALLOWED: Final[set[str]] = {"auto", "never", "always"}
COLLECTION_OMDB_JSON_MODE: str = _get_env_enum_str(
    "COLLECTION_OMDB_JSON_MODE",
    default="auto",
    allowed=_COLLECTION_OMDB_JSON_MODE_ALLOWED,
    normalize=True,
)

COLLECTION_LAZY_WIKI_ALLOW_TITLE_YEAR_FALLBACK: bool = _get_env_bool(
    "COLLECTION_LAZY_WIKI_ALLOW_TITLE_YEAR_FALLBACK",
    False,
)

COLLECTION_LAZY_WIKI_FORCE_OMDB_POST_CORE: bool = _get_env_bool(
    "COLLECTION_LAZY_WIKI_FORCE_OMDB_POST_CORE",
    True,
)

COLLECTION_TRACE_ALSO_DEBUG_CTX: bool = _get_env_bool(
    "COLLECTION_TRACE_ALSO_DEBUG_CTX",
    True,
)


# ============================================================
# CORE (backend/analyze_input_core.py)
# ============================================================

ANALYZE_TRACE_REASON_MAX_CHARS: int = _cap_int(
    "ANALYZE_TRACE_REASON_MAX_CHARS",
    _get_env_int("ANALYZE_TRACE_REASON_MAX_CHARS", 140),
    min_v=40,
    max_v=5000,
)

ANALYZE_CORE_METRICS_ENABLED: bool = _get_env_bool("ANALYZE_CORE_METRICS_ENABLED", True)


# ============================================================
# OMDb (API + throttling + TTLs + flush)
# ============================================================

OMDB_API_KEY: str | None = _get_env_str("OMDB_API_KEY", None)

OMDB_RATE_LIMIT_WAIT_SECONDS: int = _cap_int(
    "OMDB_RATE_LIMIT_WAIT_SECONDS",
    _get_env_int("OMDB_RATE_LIMIT_WAIT_SECONDS", 5),
    min_v=0,
    max_v=60 * 60,
)
OMDB_RATE_LIMIT_MAX_RETRIES: int = _cap_int(
    "OMDB_RATE_LIMIT_MAX_RETRIES",
    _get_env_int("OMDB_RATE_LIMIT_MAX_RETRIES", 1),
    min_v=0,
    max_v=20,
)

OMDB_HTTP_MAX_CONCURRENCY: int = _cap_int(
    "OMDB_HTTP_MAX_CONCURRENCY",
    _get_env_int("OMDB_HTTP_MAX_CONCURRENCY", 2),
    min_v=1,
    max_v=64,
)

OMDB_HTTP_MIN_INTERVAL_SECONDS: float = _cap_float_min(
    "OMDB_HTTP_MIN_INTERVAL_SECONDS",
    _get_env_float("OMDB_HTTP_MIN_INTERVAL_SECONDS", 0.10),
    min_v=0.0,
)

OMDB_CACHE_TTL_OK_SECONDS: int = _cap_int(
    "OMDB_CACHE_TTL_OK_SECONDS",
    _get_env_int("OMDB_CACHE_TTL_OK_SECONDS", 60 * 60 * 24 * 60),
    min_v=60,
    max_v=60 * 60 * 24 * 365 * 5,
)

OMDB_CACHE_TTL_NOT_FOUND_SECONDS: int = _cap_int(
    "OMDB_CACHE_TTL_NOT_FOUND_SECONDS",
    _get_env_int("OMDB_CACHE_TTL_NOT_FOUND_SECONDS", 60 * 60 * 24 * 7),
    min_v=60,
    max_v=60 * 60 * 24 * 365,
)

OMDB_CACHE_TTL_EMPTY_RATINGS_SECONDS: int = _cap_int(
    "OMDB_CACHE_TTL_EMPTY_RATINGS_SECONDS",
    _get_env_int("OMDB_CACHE_TTL_EMPTY_RATINGS_SECONDS", 60 * 60 * 24 * 3),
    min_v=60,
    max_v=60 * 60 * 24 * 365,
)

OMDB_CACHE_FLUSH_MAX_DIRTY_WRITES: int = _cap_int(
    "OMDB_CACHE_FLUSH_MAX_DIRTY_WRITES",
    _get_env_int("OMDB_CACHE_FLUSH_MAX_DIRTY_WRITES", 25),
    min_v=1,
    max_v=10_000,
)

OMDB_CACHE_FLUSH_MAX_SECONDS: float = _cap_float_min(
    "OMDB_CACHE_FLUSH_MAX_SECONDS",
    _get_env_float("OMDB_CACHE_FLUSH_MAX_SECONDS", 8.0),
    min_v=0.1,
)

OMDB_CACHE_PATH: Final[Path] = DATA_DIR / "omdb_cache.json"


# ============================================================
# OMDb (schema v4 internals: compaction + hot-cache)
# ============================================================

ANALIZA_OMDB_CACHE_MAX_RECORDS: int = _cap_int(
    "ANALIZA_OMDB_CACHE_MAX_RECORDS",
    _get_env_int("ANALIZA_OMDB_CACHE_MAX_RECORDS", 50_000),
    min_v=1_000,
    max_v=5_000_000,
)

ANALIZA_OMDB_CACHE_MAX_INDEX_IMDB: int = _cap_int(
    "ANALIZA_OMDB_CACHE_MAX_INDEX_IMDB",
    _get_env_int("ANALIZA_OMDB_CACHE_MAX_INDEX_IMDB", 70_000),
    min_v=1_000,
    max_v=10_000_000,
)

ANALIZA_OMDB_CACHE_MAX_INDEX_TY: int = _cap_int(
    "ANALIZA_OMDB_CACHE_MAX_INDEX_TY",
    _get_env_int("ANALIZA_OMDB_CACHE_MAX_INDEX_TY", 70_000),
    min_v=1_000,
    max_v=10_000_000,
)

ANALIZA_OMDB_HOT_CACHE_MAX: int = _cap_int(
    "ANALIZA_OMDB_HOT_CACHE_MAX",
    _get_env_int("ANALIZA_OMDB_HOT_CACHE_MAX", 2048),
    min_v=0,
    max_v=500_000,
)


# ============================================================
# OMDb (HTTP client tuning)
# ============================================================

OMDB_BASE_URL: str = _get_env_str("OMDB_BASE_URL", "https://www.omdbapi.com/") or "https://www.omdbapi.com/"

OMDB_HTTP_TIMEOUT_SECONDS: float = _cap_float_min(
    "OMDB_HTTP_TIMEOUT_SECONDS",
    _get_env_float("OMDB_HTTP_TIMEOUT_SECONDS", 10.0),
    min_v=0.5,
)

OMDB_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT: float = _cap_float_min(
    "OMDB_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT",
    _get_env_float("OMDB_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT", 30.0),
    min_v=0.1,
)

OMDB_HTTP_RETRY_TOTAL: int = _cap_int(
    "OMDB_HTTP_RETRY_TOTAL",
    _get_env_int("OMDB_HTTP_RETRY_TOTAL", 3),
    min_v=0,
    max_v=10,
)

OMDB_HTTP_RETRY_BACKOFF_FACTOR: float = _cap_float_min(
    "OMDB_HTTP_RETRY_BACKOFF_FACTOR",
    _get_env_float("OMDB_HTTP_RETRY_BACKOFF_FACTOR", 0.5),
    min_v=0.0,
)

OMDB_DISABLE_AFTER_N_FAILURES: int = _cap_int(
    "OMDB_DISABLE_AFTER_N_FAILURES",
    _get_env_int("OMDB_DISABLE_AFTER_N_FAILURES", 3),
    min_v=1,
    max_v=50,
)

OMDB_HTTP_USER_AGENT: str = _get_env_str("OMDB_HTTP_USER_AGENT", "Analiza-Movies/1.0 (local)") or "Analiza-Movies/1.0 (local)"


# ============================================================
# OMDB METRICS (resumen al final del run)
# ============================================================

OMDB_METRICS_ENABLED: bool = _get_env_bool("OMDB_METRICS_ENABLED", True)

OMDB_METRICS_TOP_N: int = _cap_int(
    "OMDB_METRICS_TOP_N",
    _get_env_int("OMDB_METRICS_TOP_N", 12),
    min_v=1,
    max_v=100,
)

OMDB_METRICS_LOG_ON_SILENT_DEBUG: bool = _get_env_bool("OMDB_METRICS_LOG_ON_SILENT_DEBUG", True)
OMDB_METRICS_LOG_EVEN_IF_ZERO: bool = _get_env_bool("OMDB_METRICS_LOG_EVEN_IF_ZERO", False)


# ============================================================
# WIKI (API + throttling + TTLs + flush + compaction caps)
# ============================================================

WIKI_LANGUAGE: str = _get_env_str("WIKI_LANGUAGE", "es") or "es"
WIKI_FALLBACK_LANGUAGE: str = _get_env_str("WIKI_FALLBACK_LANGUAGE", "en") or "en"
WIKI_DEBUG: bool = _get_env_bool("WIKI_DEBUG", False)

WIKI_SPARQL_MIN_INTERVAL_SECONDS: float = _cap_float_min(
    "WIKI_SPARQL_MIN_INTERVAL_SECONDS",
    _get_env_float("WIKI_SPARQL_MIN_INTERVAL_SECONDS", 0.20),
    min_v=0.0,
)

WIKI_CACHE_TTL_OK_SECONDS: int = _cap_int(
    "WIKI_CACHE_TTL_OK_SECONDS",
    _get_env_int("WIKI_CACHE_TTL_OK_SECONDS", 60 * 60 * 24 * 120),
    min_v=60,
    max_v=60 * 60 * 24 * 365 * 5,
)

WIKI_CACHE_TTL_NEGATIVE_SECONDS: int = _cap_int(
    "WIKI_CACHE_TTL_NEGATIVE_SECONDS",
    _get_env_int("WIKI_CACHE_TTL_NEGATIVE_SECONDS", 60 * 60 * 24 * 7),
    min_v=60,
    max_v=60 * 60 * 24 * 365,
)

WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS: int = _cap_int(
    "WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS",
    _get_env_int("WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS", 60 * 60 * 24 * 7),
    min_v=60,
    max_v=60 * 60 * 24 * 365,
)

WIKI_IS_FILM_TTL_SECONDS: int = _cap_int(
    "WIKI_IS_FILM_TTL_SECONDS",
    _get_env_int("WIKI_IS_FILM_TTL_SECONDS", 60 * 60 * 24 * 180),
    min_v=60,
    max_v=60 * 60 * 24 * 365 * 10,
)

WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES: int = _cap_int(
    "WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES",
    _get_env_int("WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES", 30),
    min_v=1,
    max_v=10_000,
)

WIKI_CACHE_FLUSH_MAX_SECONDS: float = _cap_float_min(
    "WIKI_CACHE_FLUSH_MAX_SECONDS",
    _get_env_float("WIKI_CACHE_FLUSH_MAX_SECONDS", 8.0),
    min_v=0.1,
)

WIKI_CACHE_PATH: Final[Path] = DATA_DIR / "wiki_cache.json"

ANALIZA_WIKI_CACHE_MAX_RECORDS: int = _cap_int(
    "ANALIZA_WIKI_CACHE_MAX_RECORDS",
    _get_env_int("ANALIZA_WIKI_CACHE_MAX_RECORDS", 25_000),
    min_v=1_000,
    max_v=5_000_000,
)

ANALIZA_WIKI_CACHE_MAX_IMDB_QID: int = _cap_int(
    "ANALIZA_WIKI_CACHE_MAX_IMDB_QID",
    _get_env_int("ANALIZA_WIKI_CACHE_MAX_IMDB_QID", 40_000),
    min_v=1_000,
    max_v=10_000_000,
)

ANALIZA_WIKI_CACHE_MAX_IS_FILM: int = _cap_int(
    "ANALIZA_WIKI_CACHE_MAX_IS_FILM",
    _get_env_int("ANALIZA_WIKI_CACHE_MAX_IS_FILM", 40_000),
    min_v=1_000,
    max_v=10_000_000,
)

ANALIZA_WIKI_CACHE_MAX_ENTITIES: int = _cap_int(
    "ANALIZA_WIKI_CACHE_MAX_ENTITIES",
    _get_env_int("ANALIZA_WIKI_CACHE_MAX_ENTITIES", 120_000),
    min_v=5_000,
    max_v=50_000_000,
)

ANALIZA_WIKI_DEBUG: bool = _get_env_bool("ANALIZA_WIKI_DEBUG", False)


# ============================================================
# WIKI (HTTP client tuning + endpoints)
# ============================================================

WIKI_HTTP_USER_AGENT: str = _get_env_str("WIKI_HTTP_USER_AGENT", "Analiza-Movies/1.0 (local)") or "Analiza-Movies/1.0 (local)"

WIKI_HTTP_TIMEOUT_SECONDS: float = _cap_float_min(
    "WIKI_HTTP_TIMEOUT_SECONDS",
    _get_env_float("WIKI_HTTP_TIMEOUT_SECONDS", 10.0),
    min_v=0.5,
)

WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS: float = _cap_float_min(
    "WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS",
    _get_env_float("WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS", 5.0),
    min_v=0.5,
)

WIKI_SPARQL_TIMEOUT_READ_SECONDS: float = _cap_float_min(
    "WIKI_SPARQL_TIMEOUT_READ_SECONDS",
    _get_env_float("WIKI_SPARQL_TIMEOUT_READ_SECONDS", 45.0),
    min_v=1.0,
)

WIKI_HTTP_RETRY_TOTAL: int = _cap_int(
    "WIKI_HTTP_RETRY_TOTAL",
    _get_env_int("WIKI_HTTP_RETRY_TOTAL", 3),
    min_v=0,
    max_v=10,
)

WIKI_HTTP_RETRY_BACKOFF_FACTOR: float = _cap_float_min(
    "WIKI_HTTP_RETRY_BACKOFF_FACTOR",
    _get_env_float("WIKI_HTTP_RETRY_BACKOFF_FACTOR", 0.8),
    min_v=0.0,
)

WIKI_WIKIPEDIA_REST_BASE_URL: str = _get_env_str(
    "WIKI_WIKIPEDIA_REST_BASE_URL",
    "https://{lang}.wikipedia.org/api/rest_v1",
) or "https://{lang}.wikipedia.org/api/rest_v1"

WIKI_WIKIPEDIA_API_BASE_URL: str = _get_env_str(
    "WIKI_WIKIPEDIA_API_BASE_URL",
    "https://{lang}.wikipedia.org/w/api.php",
) or "https://{lang}.wikipedia.org/w/api.php"

WIKI_WIKIDATA_API_BASE_URL: str = _get_env_str(
    "WIKI_WIKIDATA_API_BASE_URL",
    "https://www.wikidata.org/w/api.php",
) or "https://www.wikidata.org/w/api.php"

WIKI_WIKIDATA_ENTITY_BASE_URL: str = _get_env_str(
    "WIKI_WIKIDATA_ENTITY_BASE_URL",
    "https://www.wikidata.org/wiki/Special:EntityData",
) or "https://www.wikidata.org/wiki/Special:EntityData"

WIKI_WDQS_URL: str = _get_env_str(
    "WIKI_WDQS_URL",
    "https://query.wikidata.org/sparql",
) or "https://query.wikidata.org/sparql"


# ============================================================
# WIKI METRICS
# ============================================================

WIKI_METRICS_ENABLED: bool = _get_env_bool("WIKI_METRICS_ENABLED", True)

WIKI_METRICS_TOP_N: int = _cap_int(
    "WIKI_METRICS_TOP_N",
    _get_env_int("WIKI_METRICS_TOP_N", 12),
    min_v=1,
    max_v=100,
)

WIKI_METRICS_LOG_ON_SILENT_DEBUG: bool = _get_env_bool("WIKI_METRICS_LOG_ON_SILENT_DEBUG", True)
WIKI_METRICS_LOG_EVEN_IF_ZERO: bool = _get_env_bool("WIKI_METRICS_LOG_EVEN_IF_ZERO", False)


# ============================================================
# DLNA
# ============================================================

DLNA_SCAN_WORKERS: int = _cap_int(
    "DLNA_SCAN_WORKERS",
    _get_env_int("DLNA_SCAN_WORKERS", 2),
    min_v=1,
    max_v=8,
)

DLNA_BROWSE_MAX_RETRIES: int = _cap_int(
    "DLNA_BROWSE_MAX_RETRIES",
    _get_env_int("DLNA_BROWSE_MAX_RETRIES", 2),
    min_v=0,
    max_v=10,
)

DLNA_CB_FAILURE_THRESHOLD: int = _cap_int(
    "DLNA_CB_FAILURE_THRESHOLD",
    _get_env_int("DLNA_CB_FAILURE_THRESHOLD", 5),
    min_v=1,
    max_v=50,
)

DLNA_CB_OPEN_SECONDS: float = _cap_float_min(
    "DLNA_CB_OPEN_SECONDS",
    _get_env_float("DLNA_CB_OPEN_SECONDS", 20.0),
    min_v=0.1,
)


# ============================================================
# Scoring bayesiano / heurística
# ============================================================

BAYES_GLOBAL_MEAN_DEFAULT: float = _get_env_float("BAYES_GLOBAL_MEAN_DEFAULT", 6.5)
BAYES_DELETE_MAX_SCORE: float = _get_env_float("BAYES_DELETE_MAX_SCORE", 5.6)

BAYES_MIN_TITLES_FOR_GLOBAL_MEAN: int = _cap_int(
    "BAYES_MIN_TITLES_FOR_GLOBAL_MEAN",
    _get_env_int("BAYES_MIN_TITLES_FOR_GLOBAL_MEAN", 200),
    min_v=0,
    max_v=1_000_000,
)

RATING_MIN_TITLES_FOR_AUTO: int = _cap_int(
    "RATING_MIN_TITLES_FOR_AUTO",
    _get_env_int("RATING_MIN_TITLES_FOR_AUTO", 300),
    min_v=0,
    max_v=1_000_000,
)


def _parse_votes_by_year(raw: str) -> list[tuple[int, int]]:
    """
    Parsea IMDB_VOTES_BY_YEAR con formato:
        "1980:500,2000:2000,2010:5000,9999:10000"
    """
    if not raw:
        return []

    cleaned = raw.strip().strip('"').strip("'")
    table: list[tuple[int, int]] = []

    for part in cleaned.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        try:
            year_limit_str, votes_min_str = chunk.split(":")
            year_limit = int(year_limit_str.strip())
            votes_min = int(votes_min_str.strip())
            table.append((year_limit, votes_min))
        except Exception:
            if DEBUG_MODE and not SILENT_MODE:
                _logger.warning(f"Invalid IMDB_VOTES_BY_YEAR chunk ignored: {chunk!r}", always=True)
            continue

    return sorted(table, key=lambda x: x[0])


_IMDB_VOTES_BY_YEAR_RAW: str = _get_env_str(
    "IMDB_VOTES_BY_YEAR",
    "1980:500,2000:2000,2010:5000,9999:10000",
) or "1980:500,2000:2000,2010:5000,9999:10000"

IMDB_VOTES_BY_YEAR: list[tuple[int, int]] = _parse_votes_by_year(_IMDB_VOTES_BY_YEAR_RAW)

IMDB_KEEP_MIN_RATING: float = _get_env_float("IMDB_KEEP_MIN_RATING", 5.7)
IMDB_DELETE_MAX_RATING: float = _get_env_float("IMDB_DELETE_MAX_RATING", 5.5)

IMDB_KEEP_MIN_VOTES: int = _cap_int(
    "IMDB_KEEP_MIN_VOTES",
    _get_env_int("IMDB_KEEP_MIN_VOTES", 30000),
    min_v=0,
    max_v=2_000_000_000,
)


def get_votes_threshold_for_year(year: int | None) -> int:
    """
    Devuelve el umbral mínimo de votos según tablas por año.
    Fallback a último tramo si year es None o inválido.
    """
    if not IMDB_VOTES_BY_YEAR:
        return IMDB_KEEP_MIN_VOTES

    try:
        y = int(year) if year is not None else None
    except (TypeError, ValueError):
        y = None

    if y is None:
        return IMDB_VOTES_BY_YEAR[-1][1]

    for year_limit, votes_min in IMDB_VOTES_BY_YEAR:
        if y <= year_limit:
            return votes_min

    return IMDB_VOTES_BY_YEAR[-1][1]


# ============================================================
# Misidentificación / títulos sospechosos
# ============================================================

IMDB_RATING_LOW_THRESHOLD: float = _get_env_float("IMDB_RATING_LOW_THRESHOLD", 3.0)

RT_RATING_LOW_THRESHOLD: int = _cap_int(
    "RT_RATING_LOW_THRESHOLD",
    _get_env_int("RT_RATING_LOW_THRESHOLD", 20),
    min_v=0,
    max_v=100,
)

IMDB_MIN_VOTES_FOR_KNOWN: int = _cap_int(
    "IMDB_MIN_VOTES_FOR_KNOWN",
    _get_env_int("IMDB_MIN_VOTES_FOR_KNOWN", 100),
    min_v=0,
    max_v=2_000_000_000,
)


# ============================================================
# Rotten Tomatoes
# ============================================================

RT_KEEP_MIN_SCORE: int = _cap_int(
    "RT_KEEP_MIN_SCORE",
    _get_env_int("RT_KEEP_MIN_SCORE", 55),
    min_v=0,
    max_v=100,
)

IMDB_KEEP_MIN_RATING_WITH_RT: float = _get_env_float("IMDB_KEEP_MIN_RATING_WITH_RT", 6.0)

RT_DELETE_MAX_SCORE: int = _cap_int(
    "RT_DELETE_MAX_SCORE",
    _get_env_int("RT_DELETE_MAX_SCORE", 50),
    min_v=0,
    max_v=100,
)


# ============================================================
# Metacritic
# ============================================================

METACRITIC_KEEP_MIN_SCORE: int = _cap_int(
    "METACRITIC_KEEP_MIN_SCORE",
    _get_env_int("METACRITIC_KEEP_MIN_SCORE", 70),
    min_v=0,
    max_v=100,
)

METACRITIC_DELETE_MAX_SCORE: int = _cap_int(
    "METACRITIC_DELETE_MAX_SCORE",
    _get_env_int("METACRITIC_DELETE_MAX_SCORE", 40),
    min_v=0,
    max_v=100,
)


# ============================================================
# Percentiles automáticos
# ============================================================

AUTO_KEEP_RATING_PERCENTILE: float = _get_env_float("AUTO_KEEP_RATING_PERCENTILE", 0.90)
AUTO_DELETE_RATING_PERCENTILE: float = _get_env_float("AUTO_DELETE_RATING_PERCENTILE", 0.10)


# ============================================================
# Metadata fix
# ============================================================

METADATA_DRY_RUN: bool = _get_env_bool("METADATA_DRY_RUN", True)
METADATA_APPLY_CHANGES: bool = _get_env_bool("METADATA_APPLY_CHANGES", False)


# ============================================================
# Dashboard / borrado seguro
# ============================================================

DELETE_DRY_RUN: bool = _get_env_bool("DELETE_DRY_RUN", True)
DELETE_REQUIRE_CONFIRM: bool = _get_env_bool("DELETE_REQUIRE_CONFIRM", True)


# ============================================================
# Reports (paths)
# ============================================================

REPORT_ALL_FILENAME: Final[str] = "report_all.csv"
REPORT_FILTERED_FILENAME: Final[str] = "report_filtered.csv"
METADATA_FIX_FILENAME: Final[str] = "metadata_fix.csv"

REPORT_ALL_PATH: Final[str] = str(REPORTS_DIR_PATH / REPORT_ALL_FILENAME)
REPORT_FILTERED_PATH: Final[str] = str(REPORTS_DIR_PATH / REPORT_FILTERED_FILENAME)
METADATA_FIX_PATH: Final[str] = str(REPORTS_DIR_PATH / METADATA_FIX_FILENAME)


# ============================================================
# LOGGER (persistencia opcional a fichero por ejecución)
# ============================================================
#
# IMPORTANTE:
# - backend/logger.py NO debe decidir el nombre. Solo "consume" LOGGER_FILE_PATH.
# - Aquí generamos un path único por ejecución: <dir>/<prefix>_<timestamp>[_pid].log
# - Si LOGGER_FILE_PATH viene explícito en env, lo respetamos (para casos especiales).

LOGGER_FILE_ENABLED: bool = _get_env_bool("LOGGER_FILE_ENABLED", False)

_LOGGER_FILE_DIR_RAW: Final[str] = _get_env_str("LOGGER_FILE_DIR", "logs") or "logs"
_LOGGER_FILE_DIR_CANDIDATE = Path(_LOGGER_FILE_DIR_RAW)
LOGGER_FILE_DIR: Final[Path] = (
    _LOGGER_FILE_DIR_CANDIDATE
    if _LOGGER_FILE_DIR_CANDIDATE.is_absolute()
    else (BASE_DIR / _LOGGER_FILE_DIR_CANDIDATE)
)

LOGGER_FILE_PREFIX: Final[str] = _get_env_str("LOGGER_FILE_PREFIX", "run") or "run"
LOGGER_FILE_TIMESTAMP_FORMAT: Final[str] = _get_env_str(
    "LOGGER_FILE_TIMESTAMP_FORMAT",
    "%Y-%m-%d_%H-%M-%S",
) or "%Y-%m-%d_%H-%M-%S"
LOGGER_FILE_INCLUDE_PID: bool = _get_env_bool("LOGGER_FILE_INCLUDE_PID", True)

# Permite fijar un path exacto si se desea (override).
# Si es relativo => BASE_DIR/<path>.
_LOGGER_FILE_PATH_EXPLICIT_RAW: str | None = _get_env_str("LOGGER_FILE_PATH", None)


def _sanitize_filename_component(s: str) -> str:
    """
    Sanitiza una parte de nombre de fichero para evitar caracteres problemáticos.
    Mantiene letras/dígitos/._-@ y sustituye el resto por "_".
    """
    out_chars: list[str] = []
    for ch in (s or ""):
        if ch.isalnum() or ch in ("-", "_", ".", "@"):
            out_chars.append(ch)
        else:
            out_chars.append("_")
    cleaned = "".join(out_chars).strip("._-")
    return cleaned or "run"


def _build_logger_file_path() -> Path | None:
    """
    Calcula el Path final (o None) de LOGGER_FILE_PATH.

    Reglas:
    - LOGGER_FILE_ENABLED=False -> None
    - LOGGER_FILE_PATH explícito en env -> se respeta
    - Si no, generamos por ejecución: <dir>/<prefix>_<timestamp>[_pid].log
    """
    try:
        if not LOGGER_FILE_ENABLED:
            return None

        if isinstance(_LOGGER_FILE_PATH_EXPLICIT_RAW, str) and _LOGGER_FILE_PATH_EXPLICIT_RAW.strip():
            p = Path(_LOGGER_FILE_PATH_EXPLICIT_RAW.strip())
            return p if p.is_absolute() else (BASE_DIR / p)

        ts = datetime.now().strftime(LOGGER_FILE_TIMESTAMP_FORMAT)
        ts = _sanitize_filename_component(ts)
        prefix = _sanitize_filename_component(LOGGER_FILE_PREFIX)
        pid_part = f"_{os.getpid()}" if LOGGER_FILE_INCLUDE_PID else ""
        filename = f"{prefix}_{ts}{pid_part}.log"
        return LOGGER_FILE_DIR / filename

    except Exception as exc:
        _logger.warning(f"LOGGER_FILE_PATH build failed: {exc!r}", always=True)
        return None


LOGGER_FILE_PATH: Path | None = _build_logger_file_path()


# ============================================================
# DEBUG dump (solo si procede)
# ============================================================

_log_config_debug("DEBUG_MODE", DEBUG_MODE)
_log_config_debug("SILENT_MODE", SILENT_MODE)
_log_config_debug("LOG_LEVEL", LOG_LEVEL)
_log_config_debug("HTTP_DEBUG", HTTP_DEBUG)

_log_config_debug("BASEURL", BASEURL)
_log_config_debug("PLEX_PORT", PLEX_PORT)
_log_config_debug("PLEX_TOKEN", "****" if PLEX_TOKEN else None)
_log_config_debug("EXCLUDE_PLEX_LIBRARIES", EXCLUDE_PLEX_LIBRARIES)
_log_config_debug("PLEX_ANALYZE_WORKERS", PLEX_ANALYZE_WORKERS)

_log_config_debug("PLEX_PROGRESS_EVERY_N_MOVIES", PLEX_PROGRESS_EVERY_N_MOVIES)
_log_config_debug("PLEX_MAX_WORKERS_CAP", PLEX_MAX_WORKERS_CAP)
_log_config_debug("PLEX_MAX_INFLIGHT_FACTOR", PLEX_MAX_INFLIGHT_FACTOR)
_log_config_debug("PLEX_LIBRARY_LANGUAGE_DEFAULT", PLEX_LIBRARY_LANGUAGE_DEFAULT)
_log_config_debug("PLEX_LIBRARY_LANGUAGE_BY_NAME", PLEX_LIBRARY_LANGUAGE_BY_NAME)
_log_config_debug("PLEX_RUN_METRICS_ENABLED", PLEX_RUN_METRICS_ENABLED)

_log_config_debug("MOVIE_INPUT_LOOKUP_STRIP_ACCENTS", MOVIE_INPUT_LOOKUP_STRIP_ACCENTS)
_log_config_debug("MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE", MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE)
_log_config_debug("MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP", MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP)
_log_config_debug("MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS", MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS)
_log_config_debug("MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK", MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK)

_log_config_debug("ANALYZE_TRACE_REASON_MAX_CHARS", ANALYZE_TRACE_REASON_MAX_CHARS)
_log_config_debug("ANALYZE_CORE_METRICS_ENABLED", ANALYZE_CORE_METRICS_ENABLED)

_log_config_debug("PLEX_METRICS_ENABLED", PLEX_METRICS_ENABLED)
_log_config_debug("PLEX_METRICS_TOP_N", PLEX_METRICS_TOP_N)
_log_config_debug("PLEX_METRICS_LOG_ON_SILENT_DEBUG", PLEX_METRICS_LOG_ON_SILENT_DEBUG)
_log_config_debug("PLEX_METRICS_LOG_EVEN_IF_ZERO", PLEX_METRICS_LOG_EVEN_IF_ZERO)

_log_config_debug("COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS", COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS)
_log_config_debug("COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS", COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS)
_log_config_debug("COLLECTION_TRACE_LINE_MAX_CHARS", COLLECTION_TRACE_LINE_MAX_CHARS)
_log_config_debug("COLLECTION_ENABLE_LAZY_WIKI", COLLECTION_ENABLE_LAZY_WIKI)
_log_config_debug("COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE", COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE)

_log_config_debug("COLLECTION_OMDB_JSON_MODE", COLLECTION_OMDB_JSON_MODE)
_log_config_debug("COLLECTION_LAZY_WIKI_ALLOW_TITLE_YEAR_FALLBACK", COLLECTION_LAZY_WIKI_ALLOW_TITLE_YEAR_FALLBACK)
_log_config_debug("COLLECTION_LAZY_WIKI_FORCE_OMDB_POST_CORE", COLLECTION_LAZY_WIKI_FORCE_OMDB_POST_CORE)
_log_config_debug("COLLECTION_TRACE_ALSO_DEBUG_CTX", COLLECTION_TRACE_ALSO_DEBUG_CTX)

_log_config_debug("OMDB_API_KEY", "****" if OMDB_API_KEY else None)
_log_config_debug("OMDB_HTTP_MAX_CONCURRENCY", OMDB_HTTP_MAX_CONCURRENCY)
_log_config_debug("OMDB_HTTP_MIN_INTERVAL_SECONDS", OMDB_HTTP_MIN_INTERVAL_SECONDS)
_log_config_debug("OMDB_RATE_LIMIT_WAIT_SECONDS", OMDB_RATE_LIMIT_WAIT_SECONDS)
_log_config_debug("OMDB_RATE_LIMIT_MAX_RETRIES", OMDB_RATE_LIMIT_MAX_RETRIES)

_log_config_debug("OMDB_CACHE_TTL_OK_SECONDS", OMDB_CACHE_TTL_OK_SECONDS)
_log_config_debug("OMDB_CACHE_TTL_NOT_FOUND_SECONDS", OMDB_CACHE_TTL_NOT_FOUND_SECONDS)
_log_config_debug("OMDB_CACHE_TTL_EMPTY_RATINGS_SECONDS", OMDB_CACHE_TTL_EMPTY_RATINGS_SECONDS)
_log_config_debug("OMDB_CACHE_FLUSH_MAX_DIRTY_WRITES", OMDB_CACHE_FLUSH_MAX_DIRTY_WRITES)
_log_config_debug("OMDB_CACHE_FLUSH_MAX_SECONDS", OMDB_CACHE_FLUSH_MAX_SECONDS)

_log_config_debug("ANALIZA_OMDB_CACHE_MAX_RECORDS", ANALIZA_OMDB_CACHE_MAX_RECORDS)
_log_config_debug("ANALIZA_OMDB_CACHE_MAX_INDEX_IMDB", ANALIZA_OMDB_CACHE_MAX_INDEX_IMDB)
_log_config_debug("ANALIZA_OMDB_CACHE_MAX_INDEX_TY", ANALIZA_OMDB_CACHE_MAX_INDEX_TY)
_log_config_debug("ANALIZA_OMDB_HOT_CACHE_MAX", ANALIZA_OMDB_HOT_CACHE_MAX)

_log_config_debug("OMDB_BASE_URL", OMDB_BASE_URL)
_log_config_debug("OMDB_HTTP_TIMEOUT_SECONDS", OMDB_HTTP_TIMEOUT_SECONDS)
_log_config_debug("OMDB_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT", OMDB_HTTP_SEMAPHORE_ACQUIRE_TIMEOUT)
_log_config_debug("OMDB_HTTP_RETRY_TOTAL", OMDB_HTTP_RETRY_TOTAL)
_log_config_debug("OMDB_HTTP_RETRY_BACKOFF_FACTOR", OMDB_HTTP_RETRY_BACKOFF_FACTOR)
_log_config_debug("OMDB_DISABLE_AFTER_N_FAILURES", OMDB_DISABLE_AFTER_N_FAILURES)
_log_config_debug("OMDB_HTTP_USER_AGENT", OMDB_HTTP_USER_AGENT)

_log_config_debug("OMDB_METRICS_ENABLED", OMDB_METRICS_ENABLED)
_log_config_debug("OMDB_METRICS_TOP_N", OMDB_METRICS_TOP_N)
_log_config_debug("OMDB_METRICS_LOG_ON_SILENT_DEBUG", OMDB_METRICS_LOG_ON_SILENT_DEBUG)
_log_config_debug("OMDB_METRICS_LOG_EVEN_IF_ZERO", OMDB_METRICS_LOG_EVEN_IF_ZERO)

_log_config_debug("DLNA_SCAN_WORKERS", DLNA_SCAN_WORKERS)
_log_config_debug("DLNA_BROWSE_MAX_RETRIES", DLNA_BROWSE_MAX_RETRIES)
_log_config_debug("DLNA_CB_FAILURE_THRESHOLD", DLNA_CB_FAILURE_THRESHOLD)
_log_config_debug("DLNA_CB_OPEN_SECONDS", DLNA_CB_OPEN_SECONDS)

_log_config_debug("WIKI_LANGUAGE", WIKI_LANGUAGE)
_log_config_debug("WIKI_FALLBACK_LANGUAGE", WIKI_FALLBACK_LANGUAGE)
_log_config_debug("WIKI_DEBUG", WIKI_DEBUG)
_log_config_debug("WIKI_SPARQL_MIN_INTERVAL_SECONDS", WIKI_SPARQL_MIN_INTERVAL_SECONDS)
_log_config_debug("WIKI_CACHE_TTL_OK_SECONDS", WIKI_CACHE_TTL_OK_SECONDS)
_log_config_debug("WIKI_CACHE_TTL_NEGATIVE_SECONDS", WIKI_CACHE_TTL_NEGATIVE_SECONDS)
_log_config_debug("WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS", WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS)
_log_config_debug("WIKI_IS_FILM_TTL_SECONDS", WIKI_IS_FILM_TTL_SECONDS)
_log_config_debug("WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES", WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES)
_log_config_debug("WIKI_CACHE_FLUSH_MAX_SECONDS", WIKI_CACHE_FLUSH_MAX_SECONDS)

_log_config_debug("ANALIZA_WIKI_CACHE_MAX_RECORDS", ANALIZA_WIKI_CACHE_MAX_RECORDS)
_log_config_debug("ANALIZA_WIKI_CACHE_MAX_IMDB_QID", ANALIZA_WIKI_CACHE_MAX_IMDB_QID)
_log_config_debug("ANALIZA_WIKI_CACHE_MAX_IS_FILM", ANALIZA_WIKI_CACHE_MAX_IS_FILM)
_log_config_debug("ANALIZA_WIKI_CACHE_MAX_ENTITIES", ANALIZA_WIKI_CACHE_MAX_ENTITIES)
_log_config_debug("ANALIZA_WIKI_DEBUG", ANALIZA_WIKI_DEBUG)

_log_config_debug("WIKI_HTTP_USER_AGENT", WIKI_HTTP_USER_AGENT)
_log_config_debug("WIKI_HTTP_TIMEOUT_SECONDS", WIKI_HTTP_TIMEOUT_SECONDS)
_log_config_debug("WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS", WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS)
_log_config_debug("WIKI_SPARQL_TIMEOUT_READ_SECONDS", WIKI_SPARQL_TIMEOUT_READ_SECONDS)
_log_config_debug("WIKI_HTTP_RETRY_TOTAL", WIKI_HTTP_RETRY_TOTAL)
_log_config_debug("WIKI_HTTP_RETRY_BACKOFF_FACTOR", WIKI_HTTP_RETRY_BACKOFF_FACTOR)
_log_config_debug("WIKI_WIKIPEDIA_REST_BASE_URL", WIKI_WIKIPEDIA_REST_BASE_URL)
_log_config_debug("WIKI_WIKIPEDIA_API_BASE_URL", WIKI_WIKIPEDIA_API_BASE_URL)
_log_config_debug("WIKI_WIKIDATA_API_BASE_URL", WIKI_WIKIDATA_API_BASE_URL)
_log_config_debug("WIKI_WIKIDATA_ENTITY_BASE_URL", WIKI_WIKIDATA_ENTITY_BASE_URL)
_log_config_debug("WIKI_WDQS_URL", WIKI_WDQS_URL)

_log_config_debug("WIKI_METRICS_ENABLED", WIKI_METRICS_ENABLED)
_log_config_debug("WIKI_METRICS_TOP_N", WIKI_METRICS_TOP_N)
_log_config_debug("WIKI_METRICS_LOG_ON_SILENT_DEBUG", WIKI_METRICS_LOG_ON_SILENT_DEBUG)
_log_config_debug("WIKI_METRICS_LOG_EVEN_IF_ZERO", WIKI_METRICS_LOG_EVEN_IF_ZERO)

_log_config_debug("REPORTS_DIR", str(REPORTS_DIR_PATH))
_log_config_debug("REPORT_ALL_PATH", REPORT_ALL_PATH)
_log_config_debug("REPORT_FILTERED_PATH", REPORT_FILTERED_PATH)
_log_config_debug("METADATA_FIX_PATH", METADATA_FIX_PATH)

_log_config_debug("DATA_DIR", str(DATA_DIR))
_log_config_debug("OMDB_CACHE_PATH", str(OMDB_CACHE_PATH))
_log_config_debug("WIKI_CACHE_PATH", str(WIKI_CACHE_PATH))

_log_config_debug("METADATA_DRY_RUN", METADATA_DRY_RUN)
_log_config_debug("METADATA_APPLY_CHANGES", METADATA_APPLY_CHANGES)

_log_config_debug("IMDB_VOTES_BY_YEAR", IMDB_VOTES_BY_YEAR)

_log_config_debug("LOGGER_FILE_ENABLED", LOGGER_FILE_ENABLED)
_log_config_debug("LOGGER_FILE_DIR", str(LOGGER_FILE_DIR))
_log_config_debug("LOGGER_FILE_PREFIX", LOGGER_FILE_PREFIX)
_log_config_debug("LOGGER_FILE_TIMESTAMP_FORMAT", LOGGER_FILE_TIMESTAMP_FORMAT)
_log_config_debug("LOGGER_FILE_INCLUDE_PID", LOGGER_FILE_INCLUDE_PID)
_log_config_debug("LOGGER_FILE_PATH", str(LOGGER_FILE_PATH) if LOGGER_FILE_PATH else None)