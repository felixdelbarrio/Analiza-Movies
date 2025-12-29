from __future__ import annotations

"""
stats.py

Utilidades estadísticas para el proyecto (umbrales “auto”, percentiles y medias)
basadas en el cache de OMDb.

Qué hace este módulo
- Calcula una media global de IMDb (C) para el score bayesiano.
- Construye distribuciones de ratings IMDb (listas ordenadas) para percentiles.
- Deriva umbrales automáticos KEEP / DELETE:
    * Global (todos los títulos del cache)
    * NO_RT (solo títulos sin Rotten Tomatoes) para casos donde falta RT

Filosofía
- “Compute once”: todo se cachea en memoria por ejecución.
- Logging no intrusivo: usa logger central (respeta SILENT_MODE).
- Robusto ante datos corruptos o incompletos (omdb_cache heterogéneo).

Notas de diseño
- Percentil: se usa un percentil “índice discreto” sobre lista ordenada (sin interpolar),
  suficiente para umbrales operativos y estable entre ejecuciones.
- Si no hay suficientes títulos para auto-umbrales, se hace fallback a constantes
  del config (IMDB_KEEP_MIN_RATING / IMDB_DELETE_MAX_RATING).
"""

import threading

import pandas as pd

from backend import logger as _logger
from backend.config_scoring import (
    AUTO_DELETE_RATING_PERCENTILE,
    AUTO_KEEP_RATING_PERCENTILE,
    BAYES_GLOBAL_MEAN_DEFAULT,
    BAYES_MIN_TITLES_FOR_GLOBAL_MEAN,
    IMDB_DELETE_MAX_RATING,
    IMDB_KEEP_MIN_RATING,
    RATING_MIN_TITLES_FOR_AUTO,
)
from backend.omdb_client import (
    iter_cached_omdb_records,
    parse_imdb_rating_from_omdb,
    parse_rt_score_from_omdb,
)

# ============================================================================
# Logging controlado por SILENT_MODE (delegado al logger central)
# ============================================================================


def _log_stats(msg: object) -> None:
    """
    Logging “best-effort” para stats:
    - No debe romper el flujo aunque el logger falle por cualquier motivo.
    - Si el logger falla, NO spameamos stdout en silent (best-effort real).
    """
    try:
        _logger.info(str(msg))
    except Exception:
        try:
            if hasattr(_logger, "is_silent_mode") and _logger.is_silent_mode():
                return
        except Exception:
            return
        try:
            print(str(msg))
        except Exception:
            return


# ============================================================================
# Cache en memoria (se calcula una vez por ejecución)
# ============================================================================

_STATS_LOCK = threading.Lock()

# Media global IMDb (para bayes)
_GLOBAL_IMDB_MEAN_FROM_CACHE: float | None = None
_GLOBAL_IMDB_MEAN_SOURCE: str | None = None
_GLOBAL_IMDB_MEAN_COUNT: int | None = None

# Distribución de ratings (ordenada) - todos los títulos
_RATINGS_LIST: list[float] | None = None
_RATINGS_COUNT: int = 0

# Distribución de ratings (ordenada) - solo títulos sin RT
_RATINGS_NO_RT_LIST: list[float] | None = None
_RATINGS_NO_RT_COUNT: int = 0

# Auto-umbrales de rating (global)
_AUTO_KEEP_RATING_THRESHOLD: float | None = None
_AUTO_DELETE_RATING_THRESHOLD: float | None = None

# Auto-umbrales de rating (sin RT)
_AUTO_KEEP_RATING_THRESHOLD_NO_RT: float | None = None
_AUTO_DELETE_RATING_THRESHOLD_NO_RT: float | None = None


# ============================================================================
# Media global desde cache OMDb (para bayes)
# ============================================================================


def _compute_global_imdb_mean_from_cache_raw() -> tuple[float | None, int]:
    """
    Recorre el cache OMDb y calcula la media de imdbRating para entradas con rating válido.

    Returns:
      (mean_or_none, n_valid)

    - mean_or_none: None si no hay ratings válidos.
    - n_valid: número de ratings válidos encontrados (aunque no llegue al mínimo).
    """
    ratings: list[float] = []

    for data in iter_cached_omdb_records():
        if not isinstance(data, dict):
            continue
        r = parse_imdb_rating_from_omdb(data)
        if r is None:
            continue
        try:
            ratings.append(float(r))
        except Exception:
            continue

    if not ratings:
        _log_stats("INFO [stats] omdb_cache sin ratings IMDb válidos.")
        return None, 0

    return (sum(ratings) / len(ratings), len(ratings))


def get_global_imdb_mean_from_cache() -> float:
    """
    Devuelve la media global IMDb usada como C en el score bayesiano.

    Estrategia:
    - Si el cache aporta suficientes títulos (>= BAYES_MIN_TITLES_FOR_GLOBAL_MEAN),
      se usa la media calculada.
    - Si no, se usa BAYES_GLOBAL_MEAN_DEFAULT (fallback conservador).
    """
    global _GLOBAL_IMDB_MEAN_FROM_CACHE, _GLOBAL_IMDB_MEAN_SOURCE, _GLOBAL_IMDB_MEAN_COUNT

    if _GLOBAL_IMDB_MEAN_FROM_CACHE is not None:
        return _GLOBAL_IMDB_MEAN_FROM_CACHE

    with _STATS_LOCK:
        if _GLOBAL_IMDB_MEAN_FROM_CACHE is not None:
            return _GLOBAL_IMDB_MEAN_FROM_CACHE

        mean_cache, count = _compute_global_imdb_mean_from_cache_raw()

        if mean_cache is not None and count >= BAYES_MIN_TITLES_FOR_GLOBAL_MEAN:
            _GLOBAL_IMDB_MEAN_FROM_CACHE = float(mean_cache)
            _GLOBAL_IMDB_MEAN_SOURCE = f"omdb_cache (n={count})"
            _GLOBAL_IMDB_MEAN_COUNT = count
            _log_stats(
                "INFO [stats] Media global IMDb desde omdb_cache = "
                f"{_GLOBAL_IMDB_MEAN_FROM_CACHE:.3f} (n={count})"
            )
        else:
            reason = (
                "sin ratings válidos"
                if mean_cache is None
                else f"{count} < BAYES_MIN_TITLES_FOR_GLOBAL_MEAN={BAYES_MIN_TITLES_FOR_GLOBAL_MEAN}"
            )
            _GLOBAL_IMDB_MEAN_FROM_CACHE = float(BAYES_GLOBAL_MEAN_DEFAULT)
            _GLOBAL_IMDB_MEAN_SOURCE = f"default {BAYES_GLOBAL_MEAN_DEFAULT}"
            _GLOBAL_IMDB_MEAN_COUNT = count
            _log_stats(
                f"INFO [stats] Usando BAYES_GLOBAL_MEAN_DEFAULT={BAYES_GLOBAL_MEAN_DEFAULT} porque {reason}"
            )

        return _GLOBAL_IMDB_MEAN_FROM_CACHE


def get_global_imdb_mean_info() -> tuple[float, str, int]:
    """
    Devuelve:
      (mean, source, n_valid)

    - mean: media usada como C
    - source: texto “omdb_cache (n=...)” o “default ...”
    - n_valid: número de ratings válidos detectados en el cache
    """
    mean = get_global_imdb_mean_from_cache()
    source = _GLOBAL_IMDB_MEAN_SOURCE or "unknown"
    count = _GLOBAL_IMDB_MEAN_COUNT or 0
    return mean, source, count


# ============================================================================
# Media IMDb a partir del DataFrame (dashboard/resumen)
# ============================================================================


def compute_global_imdb_mean_from_df(df_all: pd.DataFrame) -> float | None:
    """
    Calcula la media de imdb_rating en el DataFrame del report.

    Devuelve:
      - float: si existe al menos un rating válido numérico
      - None: si falta la columna o no hay datos válidos
    """
    if "imdb_rating" not in df_all.columns:
        return None

    ratings = pd.to_numeric(df_all["imdb_rating"], errors="coerce").dropna()
    if ratings.empty:
        return None

    return float(ratings.mean())


# ============================================================================
# Distribuciones de ratings para percentiles (cache OMDb)
# ============================================================================


def _load_imdb_ratings_list_from_cache() -> tuple[list[float], int]:
    """
    Construye (una vez) la lista ordenada de ratings IMDb del cache (todos los títulos).

    Returns:
      (sorted_ratings, n)
    """
    global _RATINGS_LIST, _RATINGS_COUNT

    if _RATINGS_LIST is not None:
        return _RATINGS_LIST, _RATINGS_COUNT

    with _STATS_LOCK:
        if _RATINGS_LIST is not None:
            return _RATINGS_LIST, _RATINGS_COUNT

        ratings: list[float] = []
        for data in iter_cached_omdb_records():
            if not isinstance(data, dict):
                continue
            r = parse_imdb_rating_from_omdb(data)
            if r is None:
                continue
            try:
                ratings.append(float(r))
            except Exception:
                continue

        ratings.sort()
        _RATINGS_LIST = ratings
        _RATINGS_COUNT = len(ratings)

        if _RATINGS_COUNT == 0:
            _log_stats("INFO [stats] omdb_cache sin ratings válidos para auto-umbrales.")

        return _RATINGS_LIST, _RATINGS_COUNT


def _load_imdb_ratings_list_no_rt_from_cache() -> tuple[list[float], int]:
    """
    Igual que _load_imdb_ratings_list_from_cache, pero SOLO para títulos cuyo
    registro OMDb NO tiene Rotten Tomatoes (rt_score is None).

    Returns:
      (sorted_ratings_no_rt, n_no_rt)
    """
    global _RATINGS_NO_RT_LIST, _RATINGS_NO_RT_COUNT

    if _RATINGS_NO_RT_LIST is not None:
        return _RATINGS_NO_RT_LIST, _RATINGS_NO_RT_COUNT

    with _STATS_LOCK:
        if _RATINGS_NO_RT_LIST is not None:
            return _RATINGS_NO_RT_LIST, _RATINGS_NO_RT_COUNT

        ratings: list[float] = []
        for data in iter_cached_omdb_records():
            if not isinstance(data, dict):
                continue

            r = parse_imdb_rating_from_omdb(data)
            if r is None:
                continue

            rt = parse_rt_score_from_omdb(data)
            if rt is not None:
                continue

            try:
                ratings.append(float(r))
            except Exception:
                continue

        ratings.sort()
        _RATINGS_NO_RT_LIST = ratings
        _RATINGS_NO_RT_COUNT = len(ratings)

        if _RATINGS_NO_RT_COUNT == 0:
            _log_stats("INFO [stats] omdb_cache sin títulos válidos para auto-umbrales NO_RT.")

        return _RATINGS_NO_RT_LIST, _RATINGS_NO_RT_COUNT


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    """
    Percentil discreto sobre lista ORDENADA de floats.

    - p se clampa a [0, 1]
    - no interpola: usa idx = int(p * (n - 1))
    """
    if not sorted_vals:
        return None

    p = max(0.0, min(1.0, float(p)))
    n = len(sorted_vals)

    if p <= 0.0:
        return sorted_vals[0]
    if p >= 1.0:
        return sorted_vals[-1]

    idx = int(p * (n - 1))
    return sorted_vals[idx]


# ============================================================================
# Auto-umbrales globales (todos los títulos)
# ============================================================================


def get_auto_keep_rating_threshold() -> float:
    global _AUTO_KEEP_RATING_THRESHOLD

    if _AUTO_KEEP_RATING_THRESHOLD is not None:
        return _AUTO_KEEP_RATING_THRESHOLD

    ratings, n = _load_imdb_ratings_list_from_cache()

    if n >= RATING_MIN_TITLES_FOR_AUTO:
        val = _percentile(ratings, AUTO_KEEP_RATING_PERCENTILE)
        if val is not None:
            _AUTO_KEEP_RATING_THRESHOLD = float(val)
            _log_stats(
                "INFO [stats] IMDB_KEEP_MIN_RATING auto-ajustada (global): "
                f"{val:.3f} (p={AUTO_KEEP_RATING_PERCENTILE}, n={n})"
            )
            return _AUTO_KEEP_RATING_THRESHOLD

    _AUTO_KEEP_RATING_THRESHOLD = float(IMDB_KEEP_MIN_RATING)
    _log_stats(
        f"INFO [stats] Fallback IMDB_KEEP_MIN_RATING={IMDB_KEEP_MIN_RATING} "
        f"(n={n} < RATING_MIN_TITLES_FOR_AUTO={RATING_MIN_TITLES_FOR_AUTO})"
    )
    return _AUTO_KEEP_RATING_THRESHOLD


def get_auto_delete_rating_threshold() -> float:
    global _AUTO_DELETE_RATING_THRESHOLD

    if _AUTO_DELETE_RATING_THRESHOLD is not None:
        return _AUTO_DELETE_RATING_THRESHOLD

    ratings, n = _load_imdb_ratings_list_from_cache()

    if n >= RATING_MIN_TITLES_FOR_AUTO:
        val = _percentile(ratings, AUTO_DELETE_RATING_PERCENTILE)
        if val is not None:
            _AUTO_DELETE_RATING_THRESHOLD = float(val)
            _log_stats(
                "INFO [stats] IMDB_DELETE_MAX_RATING auto-ajustada (global): "
                f"{val:.3f} (p={AUTO_DELETE_RATING_PERCENTILE}, n={n})"
            )
            return _AUTO_DELETE_RATING_THRESHOLD

    _AUTO_DELETE_RATING_THRESHOLD = float(IMDB_DELETE_MAX_RATING)
    _log_stats(
        f"INFO [stats] Fallback IMDB_DELETE_MAX_RATING={IMDB_DELETE_MAX_RATING} "
        f"(n={n} < RATING_MIN_TITLES_FOR_AUTO={RATING_MIN_TITLES_FOR_AUTO})"
    )
    return _AUTO_DELETE_RATING_THRESHOLD


# ============================================================================
# Auto-umbrales específicos para títulos SIN RT
# ============================================================================


def get_auto_keep_rating_threshold_no_rt() -> float:
    global _AUTO_KEEP_RATING_THRESHOLD_NO_RT

    if _AUTO_KEEP_RATING_THRESHOLD_NO_RT is not None:
        return _AUTO_KEEP_RATING_THRESHOLD_NO_RT

    ratings, n = _load_imdb_ratings_list_no_rt_from_cache()

    if n >= RATING_MIN_TITLES_FOR_AUTO:
        val = _percentile(ratings, AUTO_KEEP_RATING_PERCENTILE)
        if val is not None:
            _AUTO_KEEP_RATING_THRESHOLD_NO_RT = float(val)
            _log_stats(
                "INFO [stats] IMDB_KEEP_MIN_RATING auto-ajustada (SIN_RT): "
                f"{val:.3f} (p={AUTO_KEEP_RATING_PERCENTILE}, n={n})"
            )
            return _AUTO_KEEP_RATING_THRESHOLD_NO_RT

    _AUTO_KEEP_RATING_THRESHOLD_NO_RT = float(get_auto_keep_rating_threshold())
    _log_stats(
        "INFO [stats] Fallback IMDB_KEEP_MIN_RATING_NO_RT usando umbral global "
        f"{_AUTO_KEEP_RATING_THRESHOLD_NO_RT:.3f} "
        f"(n_NO_RT={n} < RATING_MIN_TITLES_FOR_AUTO={RATING_MIN_TITLES_FOR_AUTO})"
    )
    return _AUTO_KEEP_RATING_THRESHOLD_NO_RT


def get_auto_delete_rating_threshold_no_rt() -> float:
    global _AUTO_DELETE_RATING_THRESHOLD_NO_RT

    if _AUTO_DELETE_RATING_THRESHOLD_NO_RT is not None:
        return _AUTO_DELETE_RATING_THRESHOLD_NO_RT

    ratings, n = _load_imdb_ratings_list_no_rt_from_cache()

    if n >= RATING_MIN_TITLES_FOR_AUTO:
        val = _percentile(ratings, AUTO_DELETE_RATING_PERCENTILE)
        if val is not None:
            _AUTO_DELETE_RATING_THRESHOLD_NO_RT = float(val)
            _log_stats(
                "INFO [stats] IMDB_DELETE_MAX_RATING auto-ajustada (SIN_RT): "
                f"{val:.3f} (p={AUTO_DELETE_RATING_PERCENTILE}, n={n})"
            )
            return _AUTO_DELETE_RATING_THRESHOLD_NO_RT

    _AUTO_DELETE_RATING_THRESHOLD_NO_RT = float(get_auto_delete_rating_threshold())
    _log_stats(
        "INFO [stats] Fallback IMDB_DELETE_MAX_RATING_NO_RT usando umbral global "
        f"{_AUTO_DELETE_RATING_THRESHOLD_NO_RT:.3f} "
        f"(n_NO_RT={n} < RATING_MIN_TITLES_FOR_AUTO={RATING_MIN_TITLES_FOR_AUTO})"
    )
    return _AUTO_DELETE_RATING_THRESHOLD_NO_RT