from __future__ import annotations

"""
backend/config.py

Carga y normaliza la configuración del proyecto desde variables de entorno (.env).

Objetivos
---------
1) Centralizar toda la configuración en un único módulo importable.
2) Parsear de forma defensiva: si un valor es inválido → fallback a default + warning visible.
3) Evitar dependencias circulares: el logger se importa tras load_dotenv y usa sys.modules
   para leer flags (SILENT_MODE/DEBUG_MODE/LOG_LEVEL) si existen.
4) Mantener compatibilidad histórica: nombres/paths usados por otros módulos.

Filosofía de logs (alineada con logger.py)
-----------------------------------------
- Los warnings por parseo inválido deben ser visibles siempre (always=True).
- La impresión detallada del “dump” de config se hace SOLO si DEBUG_MODE=True y
  SILENT_MODE=False (no spamear en silent).
"""

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# 1) Cargar .env lo antes posible (antes de leer os.getenv).
load_dotenv()

# 2) Importar logger después de load_dotenv (evita que lea flags antes de tiempo).
from backend import logger as _logger  # noqa: E402


# ============================================================
# Helpers: parseo defensivo de env vars
# ============================================================

def _get_env_int(name: str, default: int) -> int:
    """
    Lee un int desde env.
    Si no existe -> default.
    Si existe pero es inválido -> warning visible + default.
    """
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
    """
    Lee un float desde env.
    Si no existe -> default.
    Si existe pero es inválido -> warning visible + default.
    """
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
    Lee un boolean desde env.
    Acepta: 1/true/yes/on como True (case-insensitive).
    """
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _cap_int(name: str, value: int, *, min_v: int, max_v: int) -> int:
    """
    Aplica caps defensivos a ints configurables.
    """
    if value < min_v:
        _logger.warning(f"{name} < {min_v}; forcing to {min_v}", always=True)
        return min_v
    if value > max_v:
        _logger.warning(f"{name} too high; capping to {max_v}", always=True)
        return max_v
    return value


def _cap_float_min(name: str, value: float, *, min_v: float) -> float:
    """
    Aplica mínimo defensivo a floats.
    """
    if value < min_v:
        _logger.warning(f"{name} < {min_v}; forcing to {min_v}", always=True)
        return min_v
    return value


# ============================================================
# MODO DE EJECUCIÓN (flags usados por logger y por el resto)
# ============================================================
# IMPORTANTE:
# - Definidos aquí una sola vez (tu archivo los tenía duplicados).
# - El logger consulta backend.config vía sys.modules, así que estos nombres
#   deben existir y ser estables.

DEBUG_MODE: bool = _get_env_bool("DEBUG_MODE", False)
SILENT_MODE: bool = _get_env_bool("SILENT_MODE", False)

# Opcional: controla ruido HTTP de librerías externas (urllib3/requests/plexapi).
HTTP_DEBUG: bool = _get_env_bool("HTTP_DEBUG", False)

# Opcional: nivel explícito del logger (si tu logger.py lo soporta).
# LOG_LEVEL="DEBUG"|"INFO"|"WARNING"|"ERROR"|"CRITICAL"
LOG_LEVEL: str | None = os.getenv("LOG_LEVEL")


def _log_config_debug(label: str, value: object) -> None:
    """
    Registra configuración solo en DEBUG_MODE y solo si NO estamos en SILENT_MODE.
    (El progreso/summary lo hacen los orquestadores; aquí evitamos ruido.)
    """
    if not DEBUG_MODE or SILENT_MODE:
        return
    try:
        _logger.info(f"{label}: {value}")
    except Exception:
        print(f"{label}: {value}")


# ============================================================
# PLEX
# ============================================================

# BASEURL debe contener SOLO el esquema + host (sin puerto), ej:
# BASEURL=http://192.168.1.10
BASEURL: str | None = os.getenv("BASEURL")

# Puerto para Plex
PLEX_PORT: int = _cap_int("PLEX_PORT", _get_env_int("PLEX_PORT", 32400), min_v=1, max_v=65535)

# Token para Plex
PLEX_TOKEN: str | None = os.getenv("PLEX_TOKEN")

# Bibliotecas de Plex a excluir (nombres separados por comas)
_raw_exclude_plex: str = os.getenv("EXCLUDE_PLEX_LIBRARIES", "")
EXCLUDE_PLEX_LIBRARIES: list[str] = [x.strip() for x in _raw_exclude_plex.split(",") if x.strip()]


# ============================================================
# PERFORMANCE (ThreadPool)
# ============================================================

# Analiza películas en paralelo (I/O bound: Plex + OMDb + Wiki).
# Puedes sobrescribirlo desde .env: PLEX_ANALYZE_WORKERS=8
PLEX_ANALYZE_WORKERS: int = _cap_int(
    "PLEX_ANALYZE_WORKERS",
    _get_env_int("PLEX_ANALYZE_WORKERS", 8),
    min_v=1,
    max_v=64,
)


# ============================================================
# OMDb
# ============================================================

OMDB_API_KEY: str | None = os.getenv("OMDB_API_KEY")

# Rate limit OMDb (cuando OMDb devuelve "Request limit reached!")
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

# Si TRUE: rehace llamadas OMDb cuando el registro en cache está incompleto para ratings
OMDB_RETRY_EMPTY_CACHE: bool = _get_env_bool("OMDB_RETRY_EMPTY_CACHE", False)

# Limitador global (usado por omdb_client)
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


# ============================================================
# WIKI
# ============================================================

WIKI_LANGUAGE: str = os.getenv("WIKI_LANGUAGE", "es")
WIKI_FALLBACK_LANGUAGE: str = os.getenv("WIKI_FALLBACK_LANGUAGE", "en")
WIKI_DEBUG: bool = _get_env_bool("WIKI_DEBUG", False)


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


# ============================================================
# IMDB votos mínimos por año (m dinámico en Bayes)
# ============================================================

def _parse_votes_by_year(raw: str) -> list[tuple[int, int]]:
    """
    Formato: "1980:500,2000:2000,2010:5000,9999:10000"

    - Si algún chunk es inválido se ignora (defensivo).
    - Resultado se devuelve ordenado por year_limit.
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
            continue

    return sorted(table, key=lambda x: x[0])


_IMDB_VOTES_BY_YEAR_RAW: str = os.getenv(
    "IMDB_VOTES_BY_YEAR",
    "1980:500,2000:2000,2010:5000,9999:10000",
)

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
    Devuelve el umbral mínimo de votos para un año dado según IMDB_VOTES_BY_YEAR.
    Si no hay tabla, usa IMDB_KEEP_MIN_VOTES.
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
# Metacritic (refuerzo secundario)
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
# Paths / caches / reports
# ============================================================

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "data"

OMDB_CACHE_PATH: Final[Path] = DATA_DIR / "omdb_cache.json"
WIKI_CACHE_PATH: Final[Path] = DATA_DIR / "wiki_cache.json"

REPORTS_DIR: Final[str] = os.getenv("REPORTS_DIR", "reports")

REPORT_ALL_FILENAME: Final[str] = "report_all.csv"
REPORT_FILTERED_FILENAME: Final[str] = "report_filtered.csv"
METADATA_FIX_FILENAME: Final[str] = "metadata_fix.csv"

REPORT_ALL_PATH: Final[str] = os.path.join(REPORTS_DIR, REPORT_ALL_FILENAME)
REPORT_FILTERED_PATH: Final[str] = os.path.join(REPORTS_DIR, REPORT_FILTERED_FILENAME)
METADATA_FIX_PATH: Final[str] = os.path.join(REPORTS_DIR, METADATA_FIX_FILENAME)


# ============================================================
# Dashboard / borrado seguro
# ============================================================

DELETE_DRY_RUN: bool = _get_env_bool("DELETE_DRY_RUN", True)
DELETE_REQUIRE_CONFIRM: bool = _get_env_bool("DELETE_REQUIRE_CONFIRM", True)


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

_log_config_debug("OMDB_API_KEY", "****" if OMDB_API_KEY else None)
_log_config_debug("OMDB_HTTP_MAX_CONCURRENCY", OMDB_HTTP_MAX_CONCURRENCY)
_log_config_debug("OMDB_HTTP_MIN_INTERVAL_SECONDS", OMDB_HTTP_MIN_INTERVAL_SECONDS)
_log_config_debug("OMDB_RATE_LIMIT_WAIT_SECONDS", OMDB_RATE_LIMIT_WAIT_SECONDS)
_log_config_debug("OMDB_RATE_LIMIT_MAX_RETRIES", OMDB_RATE_LIMIT_MAX_RETRIES)
_log_config_debug("OMDB_RETRY_EMPTY_CACHE", OMDB_RETRY_EMPTY_CACHE)

_log_config_debug("WIKI_LANGUAGE", WIKI_LANGUAGE)
_log_config_debug("WIKI_FALLBACK_LANGUAGE", WIKI_FALLBACK_LANGUAGE)
_log_config_debug("WIKI_DEBUG", WIKI_DEBUG)

_log_config_debug("REPORTS_DIR", REPORTS_DIR)
_log_config_debug("REPORT_ALL_PATH", REPORT_ALL_PATH)
_log_config_debug("REPORT_FILTERED_PATH", REPORT_FILTERED_PATH)
_log_config_debug("METADATA_FIX_PATH", METADATA_FIX_PATH)

_log_config_debug("DATA_DIR", str(DATA_DIR))
_log_config_debug("OMDB_CACHE_PATH", str(OMDB_CACHE_PATH))
_log_config_debug("WIKI_CACHE_PATH", str(WIKI_CACHE_PATH))

_log_config_debug("METADATA_DRY_RUN", METADATA_DRY_RUN)
_log_config_debug("METADATA_APPLY_CHANGES", METADATA_APPLY_CHANGES)

_log_config_debug("IMDB_VOTES_BY_YEAR", IMDB_VOTES_BY_YEAR)