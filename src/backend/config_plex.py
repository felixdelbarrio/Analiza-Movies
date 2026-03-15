from __future__ import annotations

from backend.config_base import (
    _cap_int,
    _get_env_bool,
    _get_env_int,
    _get_env_str,
    _parse_env_kv_map,
)

# ============================================================
# PLEX
# ============================================================

BASEURL: str | None = _get_env_str("BASEURL", None)
PLEX_PORT: int = _cap_int(
    "PLEX_PORT", _get_env_int("PLEX_PORT", 32400), min_v=1, max_v=65535
)
PLEX_TOKEN: str | None = _get_env_str("PLEX_TOKEN", None)

_raw_exclude_plex: str = _get_env_str("EXCLUDE_PLEX_LIBRARIES", "") or ""
EXCLUDE_PLEX_LIBRARIES: list[str] = [
    x.strip() for x in _raw_exclude_plex.split(",") if x.strip()
]

PLEX_ANALYZE_WORKERS: int = _cap_int(
    "PLEX_ANALYZE_WORKERS", _get_env_int("PLEX_ANALYZE_WORKERS", 8), min_v=1, max_v=64
)
PLEX_PROGRESS_EVERY_N_MOVIES: int = _cap_int(
    "PLEX_PROGRESS_EVERY_N_MOVIES",
    _get_env_int("PLEX_PROGRESS_EVERY_N_MOVIES", 100),
    min_v=1,
    max_v=10_000,
)

PLEX_MAX_WORKERS_CAP: int = _cap_int(
    "PLEX_MAX_WORKERS_CAP", _get_env_int("PLEX_MAX_WORKERS_CAP", 64), min_v=1, max_v=512
)
PLEX_MAX_INFLIGHT_FACTOR: int = _cap_int(
    "PLEX_MAX_INFLIGHT_FACTOR",
    _get_env_int("PLEX_MAX_INFLIGHT_FACTOR", 4),
    min_v=1,
    max_v=50,
)

PLEX_LIBRARY_LANGUAGE_DEFAULT: str = (
    _get_env_str("PLEX_LIBRARY_LANGUAGE_DEFAULT", "es") or "es"
)
_PLEX_LIBRARY_LANGUAGE_BY_NAME_RAW: str = (
    _get_env_str("PLEX_LIBRARY_LANGUAGE_BY_NAME", "") or ""
)
PLEX_LIBRARY_LANGUAGE_BY_NAME: dict[str, str] = _parse_env_kv_map(
    _PLEX_LIBRARY_LANGUAGE_BY_NAME_RAW
)

PLEX_RUN_METRICS_ENABLED: bool = _get_env_bool("PLEX_RUN_METRICS_ENABLED", True)

PLEX_METRICS_ENABLED: bool = _get_env_bool("PLEX_METRICS_ENABLED", True)
PLEX_METRICS_TOP_N: int = _cap_int(
    "PLEX_METRICS_TOP_N", _get_env_int("PLEX_METRICS_TOP_N", 5), min_v=1, max_v=50
)
PLEX_METRICS_LOG_ON_SILENT_DEBUG: bool = _get_env_bool(
    "PLEX_METRICS_LOG_ON_SILENT_DEBUG", True
)
PLEX_METRICS_LOG_EVEN_IF_ZERO: bool = _get_env_bool(
    "PLEX_METRICS_LOG_EVEN_IF_ZERO", False
)


# ============================================================
# MOVIE_INPUT (normalización + heurística idioma)
# ============================================================

MOVIE_INPUT_LOOKUP_STRIP_ACCENTS: bool = _get_env_bool(
    "MOVIE_INPUT_LOOKUP_STRIP_ACCENTS", True
)
MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE: bool = _get_env_bool(
    "MOVIE_INPUT_LOOKUP_REMOVE_BRACKETED_NOISE", True
)
MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP: bool = _get_env_bool(
    "MOVIE_INPUT_LOOKUP_REMOVE_TRAILING_DASH_GROUP", True
)

MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS: int = _cap_int(
    "MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS",
    _get_env_int("MOVIE_INPUT_LANG_FUNCTION_WORD_MIN_HITS", 2),
    min_v=0,
    max_v=10,
)

MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK: bool = _get_env_bool(
    "MOVIE_INPUT_LANG_SKIP_ENGLISH_IF_CJK", True
)
