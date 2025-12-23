from __future__ import annotations

"""
backend/config.py

Carga y normaliza la configuración del proyecto desde variables de entorno (.env).

🎯 Principios
-------------
1) "Config as data":
   - Solo parsea, valida y expone constantes.
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
- Definimos PATHS base (BASE_DIR/DATA_DIR) muy pronto para que estén disponibles.

OMDb / Wiki / Collection
------------------------
Centraliza parámetros de los clientes y orquestadores:
- TTLs, throttles, batching, flush
- Paths
- Caps (compaction / in-memory LRU)
"""

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()

from backend import logger as _logger  # noqa: E402


# ============================================================
# Paths base (DEBEN definirse pronto)
# ============================================================

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "data"

# Directorio por defecto para reports (string en env, Path internamente)
_REPORTS_DIR_RAW: Final[str] = os.getenv("REPORTS_DIR", "reports")
REPORTS_DIR_PATH: Final[Path] = Path(_REPORTS_DIR_RAW)


# ============================================================
# Helpers: parseo defensivo de env vars
# ============================================================

_TRUE_SET: Final[set[str]] = {"1", "true", "t", "yes", "y", "on"}
_FALSE_SET: Final[set[str]] = {"0", "false", "f", "no", "n", "off"}


def _get_env_str(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return str(v)


def _get_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
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
    v = os.getenv(name)
    if v is None or v == "":
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
    """
    v = os.getenv(name)
    if v is None or v == "":
        return default

    s = str(v).strip().lower()
    if s in _TRUE_SET:
        return True
    if s in _FALSE_SET:
        return False

    _logger.warning(
        f"Invalid bool for {name!r}: {v!r}, using default {default}",
        always=True,
    )
    return default


def _cap_int(name: str, value: int, *, min_v: int, max_v: int) -> int:
    """Asegura que value esté en [min_v, max_v]."""
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
    """Dump de config solo si DEBUG_MODE=True y SILENT_MODE=False."""
    if not DEBUG_MODE or SILENT_MODE:
        return
    try:
        _logger.info(f"{label}: {value}")
    except Exception:
        print(f"{label}: {value}")


# ============================================================
# MODO DE EJECUCIÓN
# ============================================================

DEBUG_MODE: bool = _get_env_bool("DEBUG_MODE", False)
SILENT_MODE: bool = _get_env_bool("SILENT_MODE", False)

HTTP_DEBUG: bool = _get_env_bool("HTTP_DEBUG", False)

LOG_LEVEL: str | None = _get_env_str("LOG_LEVEL", None)


# ============================================================
# PLEX
# ============================================================

BASEURL: str | None = _get_env_str("BASEURL", None)

PLEX_PORT: int = _cap_int("PLEX_PORT", _get_env_int("PLEX_PORT", 32400), min_v=1, max_v=65535)
PLEX_TOKEN: str | None = _get_env_str("PLEX_TOKEN", None)

_raw_exclude_plex: str = _get_env_str("EXCLUDE_PLEX_LIBRARIES", "") or ""
EXCLUDE_PLEX_LIBRARIES: list[str] = [x.strip() for x in _raw_exclude_plex.split(",") if x.strip()]


# ============================================================
# PERFORMANCE (ThreadPool)
# ============================================================

PLEX_ANALYZE_WORKERS: int = _cap_int(
    "PLEX_ANALYZE_WORKERS",
    _get_env_int("PLEX_ANALYZE_WORKERS", 8),
    min_v=1,
    max_v=64,
)


# ============================================================
# COLLECTION (orquestación por item): caches in-memory / trazas
# ============================================================

# LRU caches intra-run en backend/collection_analysis.py.
# - OMDb local cache (payloads potencialmente grandes)
# - Wiki local cache (solo minimal block, pequeño)
#
# Ajusta esto si:
# - Runs enormes consumen RAM (baja los valores)
# - Repetición alta de títulos entre librerías / colecciones (sube valores)
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

# Longitud máxima por línea de traza en logs por item.
COLLECTION_TRACE_LINE_MAX_CHARS: int = _cap_int(
    "COLLECTION_TRACE_LINE_MAX_CHARS",
    _get_env_int("COLLECTION_TRACE_LINE_MAX_CHARS", 220),
    min_v=60,
    max_v=5000,
)

# Permite desactivar el Lazy Wiki post-core sin tocar código.
COLLECTION_ENABLE_LAZY_WIKI: bool = _get_env_bool("COLLECTION_ENABLE_LAZY_WIKI", True)

# Permite desactivar persistencia de __wiki minimal en omdb_cache.json.
# (Útil si quieres runs 100% sin write-back.)
COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE: bool = _get_env_bool(
    "COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE",
    True,
)


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

# ✅ caps de compaction para wiki_client.py (schema v6)
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

# ✅ flag extra para debug (complementa DEBUG_MODE sin forzarlo)
ANALIZA_WIKI_DEBUG: bool = _get_env_bool("ANALIZA_WIKI_DEBUG", False)


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

_log_config_debug("COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS", COLLECTION_OMDB_LOCAL_CACHE_MAX_ITEMS)
_log_config_debug("COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS", COLLECTION_WIKI_LOCAL_CACHE_MAX_ITEMS)
_log_config_debug("COLLECTION_TRACE_LINE_MAX_CHARS", COLLECTION_TRACE_LINE_MAX_CHARS)
_log_config_debug("COLLECTION_ENABLE_LAZY_WIKI", COLLECTION_ENABLE_LAZY_WIKI)
_log_config_debug("COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE", COLLECTION_PERSIST_MINIMAL_WIKI_IN_OMDB_CACHE)

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