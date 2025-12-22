from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv

# Carga de variables de entorno desde .env
load_dotenv()

from backend import logger as _logger  # noqa: E402  (se importa tras load_dotenv)

############################################################
# 🔎 Funciones auxiliares de parseo para votas minimos por año
############################################################
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
            continue

    return sorted(table, key=lambda x: x[0])


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


############################################################
# 🔎 Funciones auxiliares para gestión del entorno
############################################################
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
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


############################################################
# 🔎  BUSQUEDAS EN PLEX
############################################################
# BASEURL debe contener SOLO el esquema + host (sin puerto), ej:
# BASEURL=http://192.168.1.10
BASEURL: str | None = os.getenv("BASEURL")

# Puerto para Plex
PLEX_PORT: int = _get_env_int("PLEX_PORT", 32400)

# Token para Plex
PLEX_TOKEN: str | None = os.getenv("PLEX_TOKEN")

# Bibliotecas de Plex a excluir (nombres separados por comas)
_raw_exclude_plex: str = os.getenv("EXCLUDE_PLEX_LIBRARIES", "")
EXCLUDE_PLEX_LIBRARIES: list[str] = [
    x.strip() for x in _raw_exclude_plex.split(",") if x.strip()
]

############################################################
# ⚡ PERFORMANCE (ThreadPool / Paralelismo controlado)
############################################################
# Analiza películas en paralelo (I/O bound: Plex + OMDb + Wiki).
#
# - PLEX_ANALYZE_WORKERS controla el tamaño del ThreadPool para analiza_plex.py.
# - Ajuste recomendado:
#     * 4-8 para NAS / redes lentas / OMDb muy limitado
#     * 8-16 para PC razonable + red estable
# - Límite defensivo para evitar configuraciones peligrosas:
#     * mínimo 1
#     * máximo 64
#
# Puedes sobrescribirlo desde .env:
#   PLEX_ANALYZE_WORKERS=8
_PLEX_ANALYZE_WORKERS_RAW: int = _get_env_int("PLEX_ANALYZE_WORKERS", 8)
if _PLEX_ANALYZE_WORKERS_RAW < 1:
    _logger.warning(
        "PLEX_ANALYZE_WORKERS < 1; forcing to 1",
        always=True,
    )
    PLEX_ANALYZE_WORKERS: int = 1
elif _PLEX_ANALYZE_WORKERS_RAW > 64:
    _logger.warning(
        "PLEX_ANALYZE_WORKERS too high; capping to 64",
        always=True,
    )
    PLEX_ANALYZE_WORKERS = 64
else:
    PLEX_ANALYZE_WORKERS = _PLEX_ANALYZE_WORKERS_RAW

############################################################
# 🔎  BUSQUEDAS EN OMDB
############################################################
OMDB_API_KEY: str | None = os.getenv("OMDB_API_KEY")
# Control de rate limit OMDb
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# OMDB_RATE_LIMIT_WAIT_SECONDS=5
OMDB_RATE_LIMIT_WAIT_SECONDS: int = _get_env_int("OMDB_RATE_LIMIT_WAIT_SECONDS", 5)

# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# OMDB_RATE_LIMIT_MAX_RETRIES=1
OMDB_RATE_LIMIT_MAX_RETRIES: int = _get_env_int("OMDB_RATE_LIMIT_MAX_RETRIES", 1)

# Si TRUE: rehace llamadas OMDb cuando el registro en cache está incompleto
# OMDB_RETRY_EMPTY_CACHE=false
OMDB_RETRY_EMPTY_CACHE: bool = _get_env_bool("OMDB_RETRY_EMPTY_CACHE", False)

############################################################
# ⚡ OMDB + THREADPOOL: LIMITADOR GLOBAL (suaviza picos)
############################################################
# Cuando activamos paralelismo (ThreadPool), varios hilos pueden intentar llamar a OMDb
# a la vez. Esto puede provocar:
#   - picos de tráfico
#   - más probabilidad de rate-limit / 429
#
# Para evitarlo, se definen 2 parámetros globales (usados por omdb_client):
#
# 1) OMDB_HTTP_MAX_CONCURRENCY:
#    - Nº máximo de llamadas HTTP a OMDb en paralelo (semaforo global).
#    - Default: 2 (conservador y suele funcionar bien incluso con planes limitados).
#
# 2) OMDB_HTTP_MIN_INTERVAL_SECONDS:
#    - Intervalo mínimo global entre llamadas HTTP.
#    - Default: 0.10s (suaviza bursts sin penalizar demasiado).
#
# Puedes sobrescribirlos desde .env:
#   OMDB_HTTP_MAX_CONCURRENCY=2
#   OMDB_HTTP_MIN_INTERVAL_SECONDS=0.10
_OMDB_HTTP_MAX_CONCURRENCY_RAW: int = _get_env_int("OMDB_HTTP_MAX_CONCURRENCY", 2)
if _OMDB_HTTP_MAX_CONCURRENCY_RAW < 1:
    _logger.warning(
        "OMDB_HTTP_MAX_CONCURRENCY < 1; forcing to 1",
        always=True,
    )
    OMDB_HTTP_MAX_CONCURRENCY: int = 1
elif _OMDB_HTTP_MAX_CONCURRENCY_RAW > 64:
    _logger.warning(
        "OMDB_HTTP_MAX_CONCURRENCY too high; capping to 64",
        always=True,
    )
    OMDB_HTTP_MAX_CONCURRENCY = 64
else:
    OMDB_HTTP_MAX_CONCURRENCY = _OMDB_HTTP_MAX_CONCURRENCY_RAW

_OMDB_HTTP_MIN_INTERVAL_SECONDS_RAW: float = _get_env_float(
    "OMDB_HTTP_MIN_INTERVAL_SECONDS",
    0.10,
)
if _OMDB_HTTP_MIN_INTERVAL_SECONDS_RAW < 0.0:
    _logger.warning(
        "OMDB_HTTP_MIN_INTERVAL_SECONDS < 0; forcing to 0.0",
        always=True,
    )
    OMDB_HTTP_MIN_INTERVAL_SECONDS: float = 0.0
else:
    OMDB_HTTP_MIN_INTERVAL_SECONDS = _OMDB_HTTP_MIN_INTERVAL_SECONDS_RAW

############################################################
# 🔎  BUSQUEDAS EN WIKI
############################################################
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# WIKI_LANGUAGE=es
WIKI_LANGUAGE: str = os.getenv("WIKI_LANGUAGE", "es")
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# WIKI_FALLBACK_LANGUAGE=en
WIKI_FALLBACK_LANGUAGE: str = os.getenv("WIKI_FALLBACK_LANGUAGE", "en")
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# WIKI_DEBUG=false
WIKI_DEBUG: bool = _get_env_bool("WIKI_DEBUG", False)

############################################################
# 🎬 1) Scoring bayesiano (regla principal)
############################################################

# Media global usada si la cache es insuficiente
# BAYES_GLOBAL_MEAN_DEFAULT=6.5
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
BAYES_GLOBAL_MEAN_DEFAULT: float = _get_env_float("BAYES_GLOBAL_MEAN_DEFAULT", 6.5)

# Score bayesiano por debajo de este valor → DELETE
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# BAYES_DELETE_MAX_SCORE=5.6
BAYES_DELETE_MAX_SCORE: float = _get_env_float("BAYES_DELETE_MAX_SCORE", 5.6)

# Nº mínimo de títulos para calcular media global real
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# BAYES_MIN_TITLES_FOR_GLOBAL_MEAN=200
BAYES_MIN_TITLES_FOR_GLOBAL_MEAN: int = _get_env_int(
    "BAYES_MIN_TITLES_FOR_GLOBAL_MEAN",
    200,
)

# Nº mínimo de títulos para activar percentiles automáticos
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# RATING_MIN_TITLES_FOR_AUTO=300
RATING_MIN_TITLES_FOR_AUTO: int = _get_env_int("RATING_MIN_TITLES_FOR_AUTO", 300)

############################################################
# 2) IMDB Votos mínimos según antigüedad (m dinámico en Bayes)
############################################################

# Formato: año_límite:votos-mínimos
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# IMDB_VOTES_BY_YEAR=1980:500,2000:2000,2010:5000,9999:10000
_IMDB_VOTES_BY_YEAR_RAW: str = os.getenv(
    "IMDB_VOTES_BY_YEAR",
    "1980:500,2000:2000,2010:5000,9999:10000",
)

IMDB_VOTES_BY_YEAR: list[tuple[int, int]] = _parse_votes_by_year(_IMDB_VOTES_BY_YEAR_RAW)

IMDB_KEEP_MIN_RATING: float = _get_env_float("IMDB_KEEP_MIN_RATING", 5.7)
IMDB_DELETE_MAX_RATING: float = _get_env_float("IMDB_DELETE_MAX_RATING", 5.5)
IMDB_KEEP_MIN_VOTES: int = _get_env_int("IMDB_KEEP_MIN_VOTES", 30000)

############################################################
# 3) Misidentificación / títulos sospechosos
############################################################

# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# IMDB_RATING_LOW_THRESHOLD=3.0
IMDB_RATING_LOW_THRESHOLD: float = _get_env_float("IMDB_RATING_LOW_THRESHOLD", 3.0)

# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# RT_RATING_LOW_THRESHOLD=20
RT_RATING_LOW_THRESHOLD: int = _get_env_int("RT_RATING_LOW_THRESHOLD", 20)

############################################################
# 4) UNKNOWN por falta de votos
############################################################

# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# IMDB_MIN_VOTES_FOR_KNOWN=100
IMDB_MIN_VOTES_FOR_KNOWN: int = _get_env_int("IMDB_MIN_VOTES_FOR_KNOWN", 100)

############################################################
# 5) 🧠 ROTTEN TOMATOES : PARÁMETROS DE CLASIFICACIÓN (el público manda)
############################################################

# 1) Rotten Tomatoes como señal POSITIVA
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# RT_KEEP_MIN_SCORE=55
RT_KEEP_MIN_SCORE: int = _get_env_int("RT_KEEP_MIN_SCORE", 55)

# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# IMDB_KEEP_MIN_RATING_WITH_RT=6.0
IMDB_KEEP_MIN_RATING_WITH_RT: float = _get_env_float(
    "IMDB_KEEP_MIN_RATING_WITH_RT",
    6.0,
)
# 2) Rotten Tomatoes como señal NEGATIVA
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# RT_DELETE_MAX_SCORE=50
RT_DELETE_MAX_SCORE: int = _get_env_int("RT_DELETE_MAX_SCORE", 50)

############################################################
# 6) 🧠 METACRITIC : (refuerzo secundario)
############################################################
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# METACRITIC_KEEP_MIN_SCORE=70
METACRITIC_KEEP_MIN_SCORE: int = _get_env_int("METACRITIC_KEEP_MIN_SCORE", 70)
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# METACRITIC_DELETE_MAX_SCORE=40
METACRITIC_DELETE_MAX_SCORE: int = _get_env_int("METACRITIC_DELETE_MAX_SCORE", 40)

############################################################
# 7) Percentiles automáticos (auto-ajuste por catálogo)
############################################################

# Percentil KEEP → top 90% mejores películas
# AUTO_KEEP_RATING_PERCENTILE=0.90
AUTO_KEEP_RATING_PERCENTILE: float = _get_env_float("AUTO_KEEP_RATING_PERCENTILE", 0.90)

# Percentil DELETE → peor 10 % del catálogo
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# AUTO_DELETE_RATING_PERCENTILE=0.10
AUTO_DELETE_RATING_PERCENTILE: float = _get_env_float(
    "AUTO_DELETE_RATING_PERCENTILE",
    0.10,
)

############################################################
# 🛠 CORRECCIÓN DE METADATA
############################################################
METADATA_DRY_RUN: bool = _get_env_bool("METADATA_DRY_RUN", True)

METADATA_APPLY_CHANGES: bool = _get_env_bool("METADATA_APPLY_CHANGES", False)

############################################################
# ⚙️ PARÁMETROS para CACHES
############################################################
# Nota: Algunas partes del proyecto (p.ej. omdb_client) importan DATA_DIR.
# Debe existir para mantener compatibilidad.
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "data"

# Cachés (si se usan en módulos concretos)
OMDB_CACHE_PATH: Final[Path] = DATA_DIR / "omdb_cache.json"
WIKI_CACHE_PATH: Final[Path] = DATA_DIR / "wiki_cache.json"

############################################################
# ⚙️ PARÁMETROS para INFORMES
############################################################
REPORTS_DIR: Final[str] = os.getenv("REPORTS_DIR", "reports")

REPORT_ALL_FILENAME: Final[str] = "report_all.csv"
REPORT_FILTERED_FILENAME: Final[str] = "report_filtered.csv"
METADATA_FIX_FILENAME: Final[str] = "metadata_fix.csv"

REPORT_ALL_PATH: Final[str] = os.path.join(REPORTS_DIR, REPORT_ALL_FILENAME)
REPORT_FILTERED_PATH: Final[str] = os.path.join(REPORTS_DIR, REPORT_FILTERED_FILENAME)
METADATA_FIX_PATH: Final[str] = os.path.join(REPORTS_DIR, METADATA_FIX_FILENAME)

############################################################
# ⚙️ MODO DE EJECUCIÓN
############################################################
# IMPORTANTE: se definen aquí para que el logger pueda consultarlos
# durante la carga de config (por ejemplo en warnings de parseo).
DEBUG_MODE: bool = _get_env_bool("DEBUG_MODE", False)
SILENT_MODE: bool = _get_env_bool("SILENT_MODE", False)

############################################################
# VARIABLES PARA EL DASHBOARD
# 🧹 BORRADO SEGURO
############################################################
# NO borra archivos, solo muestra candidatos
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# DELETE_DRY_RUN=true
DELETE_DRY_RUN = _get_env_bool("DELETE_DRY_RUN", True)

# Obliga a confirmar antes de borrar
# Podemos sacar esta variable a .evn para permitir cambio en configuración por usuario
# DELETE_REQUIRE_CONFIRM=true
DELETE_REQUIRE_CONFIRM = _get_env_bool("DELETE_REQUIRE_CONFIRM", True)

############################################################
# 🔎  LOGGER
############################################################
def _log_config_debug(label: str, value: object) -> None:
    """Registra configuración solo en DEBUG_MODE (y respetando SILENT_MODE)."""
    if not DEBUG_MODE or SILENT_MODE:
        return

    try:
        _logger.info(f"{label}: {value}")
    except Exception:
        print(f"{label}: {value}")


_log_config_debug("DEBUG DEBUG_MODE", DEBUG_MODE)
_log_config_debug("DEBUG BASEURL", BASEURL)
_log_config_debug("DEBUG PLEX_PORT", PLEX_PORT)
_log_config_debug("DEBUG TOKEN", "****" if PLEX_TOKEN else None)
_log_config_debug("DEBUG EXCLUDE_PLEX_LIBRARIES", EXCLUDE_PLEX_LIBRARIES)
_log_config_debug("DEBUG PLEX_ANALYZE_WORKERS", PLEX_ANALYZE_WORKERS)
_log_config_debug("DEBUG OMDB_HTTP_MAX_CONCURRENCY", OMDB_HTTP_MAX_CONCURRENCY)
_log_config_debug("DEBUG OMDB_HTTP_MIN_INTERVAL_SECONDS", OMDB_HTTP_MIN_INTERVAL_SECONDS)
_log_config_debug("DEBUG REPORTS_DIR", REPORTS_DIR)
_log_config_debug("DEBUG REPORT_ALL_PATH", REPORT_ALL_PATH)
_log_config_debug("DEBUG REPORT_FILTERED_PATH", REPORT_FILTERED_PATH)
_log_config_debug("DEBUG METADATA_FIX_PATH", METADATA_FIX_PATH)
_log_config_debug("DEBUG DATA_DIR", str(DATA_DIR))
_log_config_debug("DEBUG OMDB_CACHE_PATH", str(OMDB_CACHE_PATH))
_log_config_debug("DEBUG WIKI_CACHE_PATH", str(WIKI_CACHE_PATH))
_log_config_debug("DEBUG METADATA_DRY_RUN", METADATA_DRY_RUN)
_log_config_debug("DEBUG METADATA_APPLY_CHANGES", METADATA_APPLY_CHANGES)
_log_config_debug("DEBUG OMDB_RETRY_EMPTY_CACHE", OMDB_RETRY_EMPTY_CACHE)
_log_config_debug("DEBUG SILENT_MODE", SILENT_MODE)
_log_config_debug("DEBUG IMDB_VOTES_BY_YEAR", IMDB_VOTES_BY_YEAR)