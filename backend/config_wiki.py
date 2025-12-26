from __future__ import annotations

from typing import Final
from pathlib import Path

from backend.config_base import (
    DATA_DIR,
    _cap_float_min,
    _cap_int,
    _get_env_bool,
    _get_env_float,
    _get_env_int,
    _get_env_str,
)

# ============================================================
# WIKI (API + throttling + TTLs + flush + compaction caps)
# ============================================================

WIKI_LANGUAGE: str = _get_env_str("WIKI_LANGUAGE", "es") or "es"
WIKI_FALLBACK_LANGUAGE: str = _get_env_str("WIKI_FALLBACK_LANGUAGE", "en") or "en"
WIKI_DEBUG: bool = _get_env_bool("WIKI_DEBUG", False)

WIKI_SPARQL_MIN_INTERVAL_SECONDS: float = _cap_float_min("WIKI_SPARQL_MIN_INTERVAL_SECONDS", _get_env_float("WIKI_SPARQL_MIN_INTERVAL_SECONDS", 0.20), min_v=0.0)

WIKI_CACHE_TTL_OK_SECONDS: int = _cap_int("WIKI_CACHE_TTL_OK_SECONDS", _get_env_int("WIKI_CACHE_TTL_OK_SECONDS", 60 * 60 * 24 * 120), min_v=60, max_v=60 * 60 * 24 * 365 * 5)
WIKI_CACHE_TTL_NEGATIVE_SECONDS: int = _cap_int("WIKI_CACHE_TTL_NEGATIVE_SECONDS", _get_env_int("WIKI_CACHE_TTL_NEGATIVE_SECONDS", 60 * 60 * 24 * 7), min_v=60, max_v=60 * 60 * 24 * 365)

WIKI_DISAMBIGUATION_NEGATIVE_TTL_SECONDS: int = _cap_int(
    "WIKI_DISAMBIGUATION_NEGATIVE_TTL_SECONDS",
    _get_env_int("WIKI_DISAMBIGUATION_NEGATIVE_TTL_SECONDS", 60 * 60 * 24 * 3),
    min_v=60,
    max_v=60 * 60 * 24 * 365,
)

WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS: int = _cap_int("WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS", _get_env_int("WIKI_IMDB_QID_NEGATIVE_TTL_SECONDS", 60 * 60 * 24 * 7), min_v=60, max_v=60 * 60 * 24 * 365)
WIKI_IS_FILM_TTL_SECONDS: int = _cap_int("WIKI_IS_FILM_TTL_SECONDS", _get_env_int("WIKI_IS_FILM_TTL_SECONDS", 60 * 60 * 24 * 180), min_v=60, max_v=60 * 60 * 24 * 365 * 10)

WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES: int = _cap_int("WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES", _get_env_int("WIKI_CACHE_FLUSH_MAX_DIRTY_WRITES", 30), min_v=1, max_v=10_000)
WIKI_CACHE_FLUSH_MAX_SECONDS: float = _cap_float_min("WIKI_CACHE_FLUSH_MAX_SECONDS", _get_env_float("WIKI_CACHE_FLUSH_MAX_SECONDS", 8.0), min_v=0.1)

WIKI_CACHE_PATH: Final[Path] = DATA_DIR / "wiki_cache.json"

ANALIZA_WIKI_CACHE_MAX_RECORDS: int = _cap_int("ANALIZA_WIKI_CACHE_MAX_RECORDS", _get_env_int("ANALIZA_WIKI_CACHE_MAX_RECORDS", 25_000), min_v=1_000, max_v=5_000_000)
ANALIZA_WIKI_CACHE_MAX_IMDB_QID: int = _cap_int("ANALIZA_WIKI_CACHE_MAX_IMDB_QID", _get_env_int("ANALIZA_WIKI_CACHE_MAX_IMDB_QID", 40_000), min_v=1_000, max_v=10_000_000)
ANALIZA_WIKI_CACHE_MAX_IS_FILM: int = _cap_int("ANALIZA_WIKI_CACHE_MAX_IS_FILM", _get_env_int("ANALIZA_WIKI_CACHE_MAX_IS_FILM", 40_000), min_v=1_000, max_v=10_000_000)
ANALIZA_WIKI_CACHE_MAX_ENTITIES: int = _cap_int("ANALIZA_WIKI_CACHE_MAX_ENTITIES", _get_env_int("ANALIZA_WIKI_CACHE_MAX_ENTITIES", 120_000), min_v=5_000, max_v=50_000_000)

ANALIZA_WIKI_DEBUG: bool = _get_env_bool("ANALIZA_WIKI_DEBUG", False)


# ============================================================
# WIKI (HTTP client tuning + endpoints)
# ============================================================

WIKI_HTTP_MAX_CONCURRENCY: int = _cap_int("WIKI_HTTP_MAX_CONCURRENCY", _get_env_int("WIKI_HTTP_MAX_CONCURRENCY", 4), min_v=1, max_v=64)
WIKI_HTTP_USER_AGENT: str = _get_env_str("WIKI_HTTP_USER_AGENT", "Analiza-Movies/1.0 (local)") or "Analiza-Movies/1.0 (local)"

WIKI_HTTP_TIMEOUT_SECONDS: float = _cap_float_min("WIKI_HTTP_TIMEOUT_SECONDS", _get_env_float("WIKI_HTTP_TIMEOUT_SECONDS", 10.0), min_v=0.5)
WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS: float = _cap_float_min("WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS", _get_env_float("WIKI_SPARQL_TIMEOUT_CONNECT_SECONDS", 5.0), min_v=0.5)
WIKI_SPARQL_TIMEOUT_READ_SECONDS: float = _cap_float_min("WIKI_SPARQL_TIMEOUT_READ_SECONDS", _get_env_float("WIKI_SPARQL_TIMEOUT_READ_SECONDS", 45.0), min_v=1.0)

WIKI_HTTP_RETRY_TOTAL: int = _cap_int("WIKI_HTTP_RETRY_TOTAL", _get_env_int("WIKI_HTTP_RETRY_TOTAL", 3), min_v=0, max_v=10)
WIKI_HTTP_RETRY_BACKOFF_FACTOR: float = _cap_float_min("WIKI_HTTP_RETRY_BACKOFF_FACTOR", _get_env_float("WIKI_HTTP_RETRY_BACKOFF_FACTOR", 0.8), min_v=0.0)

WIKI_WIKIPEDIA_REST_BASE_URL: str = _get_env_str("WIKI_WIKIPEDIA_REST_BASE_URL", "https://{lang}.wikipedia.org/api/rest_v1") or "https://{lang}.wikipedia.org/api/rest_v1"
WIKI_WIKIPEDIA_API_BASE_URL: str = _get_env_str("WIKI_WIKIPEDIA_API_BASE_URL", "https://{lang}.wikipedia.org/w/api.php") or "https://{lang}.wikipedia.org/w/api.php"
WIKI_WIKIDATA_API_BASE_URL: str = _get_env_str("WIKI_WIKIDATA_API_BASE_URL", "https://www.wikidata.org/w/api.php") or "https://www.wikidata.org/w/api.php"
WIKI_WIKIDATA_ENTITY_BASE_URL: str = _get_env_str("WIKI_WIKIDATA_ENTITY_BASE_URL", "https://www.wikidata.org/wiki/Special:EntityData") or "https://www.wikidata.org/wiki/Special:EntityData"
WIKI_WDQS_URL: str = _get_env_str("WIKI_WDQS_URL", "https://query.wikidata.org/sparql") or "https://query.wikidata.org/sparql"


# ============================================================
# WIKI METRICS
# ============================================================

WIKI_METRICS_ENABLED: bool = _get_env_bool("WIKI_METRICS_ENABLED", True)
WIKI_METRICS_TOP_N: int = _cap_int("WIKI_METRICS_TOP_N", _get_env_int("WIKI_METRICS_TOP_N", 12), min_v=1, max_v=100)
WIKI_METRICS_LOG_ON_SILENT_DEBUG: bool = _get_env_bool("WIKI_METRICS_LOG_ON_SILENT_DEBUG", True)
WIKI_METRICS_LOG_EVEN_IF_ZERO: bool = _get_env_bool("WIKI_METRICS_LOG_EVEN_IF_ZERO", False)


# ============================================================
# WIKI circuit breaker suave (compat + nombres nuevos)
# ============================================================

WIKI_CB_FAIL_THRESHOLD: int = _cap_int("WIKI_CB_FAIL_THRESHOLD", _get_env_int("WIKI_CB_FAIL_THRESHOLD", 5), min_v=1, max_v=50)

WIKI_CB_COOLDOWN_SECONDS: float = _cap_float_min(
    "WIKI_CB_COOLDOWN_SECONDS",
    _get_env_float("WIKI_CB_COOLDOWN_SECONDS", max(5.0, float(WIKI_HTTP_TIMEOUT_SECONDS) * 10.0)),
    min_v=0.1,
)

WIKI_CB_FAILURE_THRESHOLD: int = _cap_int(
    "WIKI_CB_FAILURE_THRESHOLD",
    _get_env_int("WIKI_CB_FAILURE_THRESHOLD", WIKI_CB_FAIL_THRESHOLD),
    min_v=1,
    max_v=50,
)

WIKI_CB_OPEN_SECONDS: float = _cap_float_min(
    "WIKI_CB_OPEN_SECONDS",
    _get_env_float("WIKI_CB_OPEN_SECONDS", float(WIKI_CB_COOLDOWN_SECONDS)),
    min_v=0.1,
)

WIKI_WDQS_CB_FAILURE_THRESHOLD: int = _cap_int(
    "WIKI_WDQS_CB_FAILURE_THRESHOLD",
    _get_env_int("WIKI_WDQS_CB_FAILURE_THRESHOLD", int(WIKI_CB_FAILURE_THRESHOLD)),
    min_v=1,
    max_v=50,
)

WIKI_WDQS_CB_OPEN_SECONDS: float = _cap_float_min(
    "WIKI_WDQS_CB_OPEN_SECONDS",
    _get_env_float("WIKI_WDQS_CB_OPEN_SECONDS", float(WIKI_CB_OPEN_SECONDS)),
    min_v=0.1,
)

# ============================================================
# ✅ NUEVO: SWR (stale-while-revalidate) para items OK
# ============================================================

WIKI_CACHE_SWR_OK_GRACE_SECONDS: int = _cap_int(
    "WIKI_CACHE_SWR_OK_GRACE_SECONDS",
    _get_env_int("WIKI_CACHE_SWR_OK_GRACE_SECONDS", 60 * 60 * 24 * 7),  # 7 días de “gracia”
    min_v=0,
    max_v=60 * 60 * 24 * 365 * 5,
)

# ============================================================
# ✅ NUEVO: cache de candidatos de búsqueda Wikipedia
# ============================================================

WIKI_SEARCH_CANDIDATES_MAX_ENTRIES: int = _cap_int(
    "WIKI_SEARCH_CANDIDATES_MAX_ENTRIES",
    _get_env_int("WIKI_SEARCH_CANDIDATES_MAX_ENTRIES", 20_000),
    min_v=0,
    max_v=5_000_000,
)

WIKI_SEARCH_CANDIDATES_TTL_SECONDS: int = _cap_int(
    "WIKI_SEARCH_CANDIDATES_TTL_SECONDS",
    _get_env_int("WIKI_SEARCH_CANDIDATES_TTL_SECONDS", 60 * 60 * 24 * 14),  # 14 días
    min_v=60,
    max_v=60 * 60 * 24 * 365 * 2,
)

# ============================================================
# ✅ NUEVO: single-flight (espera a que otro hilo complete)
# ============================================================

WIKI_SINGLEFLIGHT_WAIT_SECONDS: float = _cap_float_min(
    "WIKI_SINGLEFLIGHT_WAIT_SECONDS",
    _get_env_float("WIKI_SINGLEFLIGHT_WAIT_SECONDS", 1.5),
    min_v=0.0,
)