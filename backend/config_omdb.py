from __future__ import annotations

from pathlib import Path
from typing import Final

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

# Hot-cache negative TTL (misses) para evitar "miss pegado" infinito intra-run.
OMDB_HOT_MISS_TTL_SECONDS: float = _cap_float_min(
    "OMDB_HOT_MISS_TTL_SECONDS",
    _get_env_float("OMDB_HOT_MISS_TTL_SECONDS", 30.0),
    min_v=0.0,
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

# Jitter aplicado a sleeps (throttle/rate-limit) para desincronizar threads (0..0.5 recomendado).
OMDB_JITTER_RATIO: float = _cap_float_min(
    "OMDB_JITTER_RATIO",
    _get_env_float("OMDB_JITTER_RATIO", 0.10),
    min_v=0.0,
)
# Nota: cap superior (0.5) se aplica en runtime en omdb_client.py por simplicidad.

# ============================================================
# OMDb (single-flight + circuit breaker)  ✅ NUEVO
# ============================================================

# Single-flight: cuánto espera un "follower" a que el líder resuelva y escriba caché.
OMDB_SINGLEFLIGHT_WAIT_SECONDS: float = _cap_float_min(
    "OMDB_SINGLEFLIGHT_WAIT_SECONDS",
    _get_env_float("OMDB_SINGLEFLIGHT_WAIT_SECONDS", 1.25),
    min_v=0.05,
)

# Circuit breaker:
# - abre tras N fallos "duros" (red/protocolo) y evita hacer llamadas durante OPEN_SECONDS
OMDB_CIRCUIT_BREAKER_THRESHOLD: int = _cap_int(
    "OMDB_CIRCUIT_BREAKER_THRESHOLD",
    _get_env_int("OMDB_CIRCUIT_BREAKER_THRESHOLD", 5),
    min_v=1,
    max_v=50,
)
OMDB_CIRCUIT_BREAKER_OPEN_SECONDS: float = _cap_float_min(
    "OMDB_CIRCUIT_BREAKER_OPEN_SECONDS",
    _get_env_float("OMDB_CIRCUIT_BREAKER_OPEN_SECONDS", 20.0),
    min_v=0.5,
)

# ============================================================
# OMDb (cache JSON output + compaction amortizada) ✅ NUEVO
# ============================================================

# Escritura del JSON:
# - pretty=True => indent configurable (útil para debug/manual inspect)
# - pretty=False => compact (separators) para ahorrar disco/IO
OMDB_CACHE_JSON_PRETTY: bool = _get_env_bool("OMDB_CACHE_JSON_PRETTY", False)

OMDB_CACHE_JSON_INDENT: int = _cap_int(
    "OMDB_CACHE_JSON_INDENT",
    _get_env_int("OMDB_CACHE_JSON_INDENT", 2),
    min_v=0,
    max_v=8,
)

# Compaction amortizada:
# - 0 => comportamiento previo (compaction siempre que flush)
# - N>0 => compaction "full" cada N flushes (reduce CPU si el cache es grande)
OMDB_CACHE_COMPACT_EVERY_N_FLUSHES: int = _cap_int(
    "OMDB_CACHE_COMPACT_EVERY_N_FLUSHES",
    _get_env_int("OMDB_CACHE_COMPACT_EVERY_N_FLUSHES", 0),
    min_v=0,
    max_v=10_000,
)

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