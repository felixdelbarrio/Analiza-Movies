from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# Carga de variables de entorno desde .env
load_dotenv()

from backend import logger as _logger  # noqa: E402  (se importa tras load_dotenv)


def _get_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        _logger.warning(f"Invalid int for {name!r}: {v!r}, using default {default}")
        return default


def _get_env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        _logger.warning(f"Invalid float for {name!r}: {v!r}, using default {default}")
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# ----------------------------------------------------
# Conexión a servidor multimedia / Plex / OMDb
# ----------------------------------------------------
# BASEURL debe contener SOLO el esquema + host (sin puerto), ej:
#   BASEURL=http://192.168.1.10
BASEURL: str | None = os.getenv("BASEURL")

# Puertos independientes para Plex y dlna
PLEX_PORT: int = _get_env_int("PLEX_PORT", 32400)
dlna_PORT: int = _get_env_int("dlna_PORT", 8200)

PLEX_TOKEN: str | None = os.getenv("PLEX_TOKEN")
OMDB_API_KEY: str | None = os.getenv("OMDB_API_KEY")

# Prefijo de ficheros de salida (report_all, report_filtered, etc.)
OUTPUT_PREFIX: str = os.getenv("OUTPUT_PREFIX", "report")

# Bibliotecas a excluir (Plex y dlna) — separadas
_raw_exclude_plex: str = os.getenv("EXCLUDE_PLEX_LIBRARIES", "")
EXCLUDE_PLEX_LIBRARIES: list[str] = [
    x.strip() for x in _raw_exclude_plex.split(",") if x.strip()
]

_raw_exclude_dlna: str = os.getenv("EXCLUDE_DLNA_LIBRARIES", "")
EXCLUDE_DLNA_LIBRARIES: list[str] = [
    x.strip() for x in _raw_exclude_dlna.split(",") if x.strip()
]

# ----------------------------------------------------
# Umbrales de decisión para KEEP/DELETE
# ----------------------------------------------------
#
# IMPORTANTE:
# - El scoring principal es bayesiano (BAYES_*).
# - Estos valores actúan como refuerzo adicional y/o fallback
#   cuando no hay datos suficientes en omdb_cache.
# ----------------------------------------------------

# ---- Rotten Tomatoes (positivo / negativo) ----
RT_KEEP_MIN_SCORE: int = _get_env_int("RT_KEEP_MIN_SCORE", 55)
RT_DELETE_MAX_SCORE: int = _get_env_int("RT_DELETE_MAX_SCORE", 50)

# ---- IMDb con RT (refuerzo positivo) ----
IMDB_KEEP_MIN_RATING_WITH_RT: float = _get_env_float(
    "IMDB_KEEP_MIN_RATING_WITH_RT",
    6.0,
)

# ---- Fallbacks de rating (cuando no hay suficientes datos para auto-umbrales) ----
IMDB_KEEP_MIN_RATING: float = _get_env_float(
    "IMDB_KEEP_MIN_RATING",
    5.7,
)
IMDB_DELETE_MAX_RATING: float = _get_env_float(
    "IMDB_DELETE_MAX_RATING",
    5.5,
)

# ---- Votos mínimos globales (fallback) ----
IMDB_KEEP_MIN_VOTES: int = _get_env_int("IMDB_KEEP_MIN_VOTES", 30_000)

# ---- UNKNOWN por falta de información ----
IMDB_MIN_VOTES_FOR_KNOWN: int = _get_env_int("IMDB_MIN_VOTES_FOR_KNOWN", 100)

# ---- LOW thresholds (para misidentificación / sospechosos) ----
IMDB_RATING_LOW_THRESHOLD: float = _get_env_float("IMDB_RATING_LOW_THRESHOLD", 3.0)
RT_RATING_LOW_THRESHOLD: int = _get_env_int("RT_RATING_LOW_THRESHOLD", 20)

# ----------------------------------------------------
# Auto-umbrales de rating desde omdb_cache
# ----------------------------------------------------
AUTO_KEEP_RATING_PERCENTILE: float = _get_env_float(
    "AUTO_KEEP_RATING_PERCENTILE",
    0.70,
)
AUTO_DELETE_RATING_PERCENTILE: float = _get_env_float(
    "AUTO_DELETE_RATING_PERCENTILE",
    0.01,
)
RATING_MIN_TITLES_FOR_AUTO: int = _get_env_int(
    "RATING_MIN_TITLES_FOR_AUTO",
    300,
)

# ----------------------------------------------------
# Votos mínimos según antigüedad (para m en Bayes)
# ----------------------------------------------------
_IMDB_VOTES_BY_YEAR_RAW: str = os.getenv(
    "IMDB_VOTES_BY_YEAR",
    "1980:500,2000:2000,2010:5000,9999:10000",
)

# ---- Metacritic (crítica especializada, 0-100) ----
METACRITIC_KEEP_MIN_SCORE: int = _get_env_int("METACRITIC_KEEP_MIN_SCORE", 70)
METACRITIC_DELETE_MAX_SCORE: int = _get_env_int("METACRITIC_DELETE_MAX_SCORE", 40)


def _parse_votes_by_year(raw: str) -> list[tuple[int, int]]:
    """
    Parsea cadenas tipo:
      1980:500,2000:2000,2010:5000,9999:10000

    Admitiendo también comillas alrededor:
      "1980:500,2000:2000,2010:5000,9999:10000"
      '1980:500,2000:2000,2010:5000,9999:10000'
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
            # Silenciosamente ignoramos trozos mal formados
            continue

    return sorted(table, key=lambda x: x[0])


IMDB_VOTES_BY_YEAR: list[tuple[int, int]] = _parse_votes_by_year(
    _IMDB_VOTES_BY_YEAR_RAW,
)


def get_votes_threshold_for_year(year: int | None) -> int:
    """
    Devuelve el número mínimo de votos exigidos para una película
    según su año, usando la tabla IMDB_VOTES_BY_YEAR.

    Reglas:
      - Si la tabla está vacía: devolvemos IMDB_KEEP_MIN_VOTES (fallback).
      - Si year es None o no es convertible a int: usamos el tramo más exigente
        (último de IMDB_VOTES_BY_YEAR).
      - Recorre IMDB_VOTES_BY_YEAR y devuelve el primer votes_min
        cuyo year_limit sea >= year.
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


# ----------------------------------------------------
# Parámetros para el scoring bayesiano global
# ----------------------------------------------------
BAYES_GLOBAL_MEAN_DEFAULT: float = _get_env_float("BAYES_GLOBAL_MEAN_DEFAULT", 6.0)
BAYES_DELETE_MAX_SCORE: float = _get_env_float("BAYES_DELETE_MAX_SCORE", 4.9)
BAYES_MIN_TITLES_FOR_GLOBAL_MEAN: int = _get_env_int(
    "BAYES_MIN_TITLES_FOR_GLOBAL_MEAN",
    200,
)

# ----------------------------------------------------
# Rate limit OMDb
# ----------------------------------------------------
OMDB_RATE_LIMIT_WAIT_SECONDS: int = _get_env_int(
    "OMDB_RATE_LIMIT_WAIT_SECONDS",
    60,
)
OMDB_RATE_LIMIT_MAX_RETRIES: int = _get_env_int(
    "OMDB_RATE_LIMIT_MAX_RETRIES",
    1,
)

# ----------------------------------------------------
# Parámetros extra para corrección de metadata
# ----------------------------------------------------
METADATA_OUTPUT_PREFIX: str = os.getenv("METADATA_OUTPUT_PREFIX", "metadata_fix")
METADATA_MIN_RATING_FOR_OK: float = _get_env_float(
    "METADATA_MIN_RATING_FOR_OK",
    6.0,
)
METADATA_MIN_VOTES_FOR_OK: int = _get_env_int(
    "METADATA_MIN_VOTES_FOR_OK",
    2000,
)

METADATA_DRY_RUN: bool = _get_env_bool("METADATA_DRY_RUN", True)
METADATA_APPLY_CHANGES: bool = _get_env_bool("METADATA_APPLY_CHANGES", False)

OMDB_RETRY_EMPTY_CACHE: bool = _get_env_bool("OMDB_RETRY_EMPTY_CACHE", False)

SILENT_MODE: bool = _get_env_bool("SILENT_MODE", False)


# ----------------------------------------------------
# Logs de depuración de configuración
# ----------------------------------------------------
def _log_config_debug(label: str, value: object) -> None:
    """Registra configuración en el logger central respetando `SILENT_MODE`."""
    try:
        _logger.info(f"{label}: {value}")
    except Exception:
        if not SILENT_MODE:
            print(f"{label}: {value}")


_log_config_debug("DEBUG BASEURL", BASEURL)
_log_config_debug("DEBUG PLEX_PORT", PLEX_PORT)
_log_config_debug("DEBUG dlna_PORT", dlna_PORT)
_log_config_debug("DEBUG TOKEN", "****" if PLEX_TOKEN else None)
_log_config_debug("DEBUG EXCLUDE_PLEX_LIBRARIES", EXCLUDE_PLEX_LIBRARIES)
_log_config_debug("DEBUG EXCLUDE_DLNA_LIBRARIES", EXCLUDE_DLNA_LIBRARIES)
_log_config_debug("DEBUG METADATA_DRY_RUN", METADATA_DRY_RUN)
_log_config_debug("DEBUG METADATA_APPLY_CHANGES", METADATA_APPLY_CHANGES)
_log_config_debug("DEBUG OMDB_RETRY_EMPTY_CACHE", OMDB_RETRY_EMPTY_CACHE)
_log_config_debug("DEBUG SILENT_MODE", SILENT_MODE)
_log_config_debug("DEBUG IMDB_VOTES_BY_YEAR", IMDB_VOTES_BY_YEAR)