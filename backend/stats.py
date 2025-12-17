from __future__ import annotations

from typing import Final

import pandas as pd

from backend.config import (
    BAYES_GLOBAL_MEAN_DEFAULT,
    BAYES_MIN_TITLES_FOR_GLOBAL_MEAN,
    AUTO_KEEP_RATING_PERCENTILE,
    AUTO_DELETE_RATING_PERCENTILE,
    RATING_MIN_TITLES_FOR_AUTO,
    IMDB_KEEP_MIN_RATING,
    IMDB_DELETE_MAX_RATING,
)
from backend.omdb_client import (
    omdb_cache,
    parse_imdb_rating_from_omdb,
    parse_rt_score_from_omdb,
)
from backend import logger as _logger

# -------------------------------------------------------------------
# Logging controlado por SILENT_MODE
# -------------------------------------------------------------------


def _log_stats(msg: object) -> None:
    """Logea vía el logger central (respeta SILENT_MODE internamente)."""
    try:
        _logger.info(str(msg))
    except Exception:
        # Fallback ligero
        print(msg)


# -------------------------------------------------------------------
# Cache en memoria (solo se calculan una vez por ejecución)
# -------------------------------------------------------------------

# Media global IMDb (para bayes)
_GLOBAL_IMDB_MEAN_FROM_CACHE: float | None = None
_GLOBAL_IMDB_MEAN_SOURCE: str | None = None
_GLOBAL_IMDB_MEAN_COUNT: int | None = None

# Distribución de ratings (ordenada)
_RATINGS_LIST: list[float] | None = None
_RATINGS_COUNT: int = 0

# Distribución de ratings SOLO para títulos SIN RT
_RATINGS_NO_RT_LIST: list[float] | None = None
_RATINGS_NO_RT_COUNT: int = 0

# Auto-umbrales de rating KEEP / DELETE
_AUTO_KEEP_RATING_THRESHOLD: float | None = None
_AUTO_DELETE_RATING_THRESHOLD: float | None = None

# Auto-umbrales de rating KEEP / DELETE para pelis SIN RT
_AUTO_KEEP_RATING_THRESHOLD_NO_RT: float | None = None
_AUTO_DELETE_RATING_THRESHOLD_NO_RT: float | None = None


# -------------------------------------------------------------------
# Media global desde omdb_cache (para bayes)
# -------------------------------------------------------------------
def _compute_global_imdb_mean_from_cache_raw() -> tuple[float | None, int]:
    """
    Recorre omdb_cache y calcula la media de imdbRating para todas las
    entradas que tengan un rating válido. Devuelve (media, n_validos).
    """
    if not isinstance(omdb_cache, dict) or not omdb_cache:
        _log_stats("INFO [stats] omdb_cache vacío o no dict; sin ratings IMDb.")
        return None, 0

    ratings: list[float] = []
    for data in omdb_cache.values():
        if not isinstance(data, dict):
            continue
        r = parse_imdb_rating_from_omdb(data)
        if r is not None:
            ratings.append(float(r))

    if not ratings:
        _log_stats("INFO [stats] omdb_cache sin ratings IMDb válidos.")
        return None, 0

    return sum(ratings) / len(ratings), len(ratings)


def get_global_imdb_mean_from_cache() -> float:
    """
    Devuelve la media global IMDb que se usa como C en el score bayesiano.
    """
    global _GLOBAL_IMDB_MEAN_FROM_CACHE, _GLOBAL_IMDB_MEAN_SOURCE, _GLOBAL_IMDB_MEAN_COUNT

    if _GLOBAL_IMDB_MEAN_FROM_CACHE is not None:
        return _GLOBAL_IMDB_MEAN_FROM_CACHE

    mean_cache, count = _compute_global_imdb_mean_from_cache_raw()

    if mean_cache is not None and count >= BAYES_MIN_TITLES_FOR_GLOBAL_MEAN:
        _GLOBAL_IMDB_MEAN_FROM_CACHE = float(mean_cache)
        _GLOBAL_IMDB_MEAN_SOURCE = f"omdb_cache (n={count})"
        _GLOBAL_IMDB_MEAN_COUNT = count
        _log_stats(
            f"INFO [stats] Media global IMDb desde omdb_cache = "
            f"{_GLOBAL_IMDB_MEAN_FROM_CACHE:.3f} (n={count})"
        )
    else:
        reason = (
            "sin ratings válidos"
            if mean_cache is None
            else f"{count} < BAYES_MIN_TITLES_FOR_GLOBAL_MEAN={BAYES_MIN_TITLES_FOR_GLOBAL_MEAN}"
        )
        _GLOBAL_IMDB_MEAN_FROM_CACHE = BAYES_GLOBAL_MEAN_DEFAULT
        _GLOBAL_IMDB_MEAN_SOURCE = f"default {BAYES_GLOBAL_MEAN_DEFAULT}"
        _GLOBAL_IMDB_MEAN_COUNT = count
        _log_stats(
            f"INFO [stats] Usando BAYES_GLOBAL_MEAN_DEFAULT={BAYES_GLOBAL_MEAN_DEFAULT} porque {reason}"
        )

    return _GLOBAL_IMDB_MEAN_FROM_CACHE


def get_global_imdb_mean_info() -> tuple[float, str, int]:
    """
    Devuelve (media, fuente, n_validos) de la media global IMDb usada como C.
    """
    mean = get_global_imdb_mean_from_cache()
    source = _GLOBAL_IMDB_MEAN_SOURCE or "unknown"
    count = _GLOBAL_IMDB_MEAN_COUNT or 0
    return mean, source, count


# -------------------------------------------------------------------
# Media IMDb a partir del DataFrame (para el dashboard/resumen)
# -------------------------------------------------------------------
def compute_global_imdb_mean_from_df(df_all: pd.DataFrame) -> float | None:
    """
    Calcula la media IMDb sobre el DataFrame completo (columna imdb_rating).

    Devuelve:
      - float si hay al menos un rating válido.
      - None si no hay columna o todos son NaN / no numéricos.
    """
    if "imdb_rating" not in df_all.columns:
        return None

    ratings = pd.to_numeric(df_all["imdb_rating"], errors="coerce").dropna()
    if ratings.empty:
        return None

    return float(ratings.mean())


# -------------------------------------------------------------------
# Ratings list (ordenada) para percentiles (TODOS los títulos)
# -------------------------------------------------------------------
def _load_imdb_ratings_list_from_cache() -> tuple[list[float], int]:
    global _RATINGS_LIST, _RATINGS_COUNT

    if _RATINGS_LIST is not None:
        return _RATINGS_LIST, _RATINGS_COUNT

    ratings: list[float] = []
    if isinstance(omdb_cache, dict):
        for data in omdb_cache.values():
            if not isinstance(data, dict):
                continue
            r = parse_imdb_rating_from_omdb(data)
            if r is not None:
                ratings.append(float(r))

    ratings.sort()
    _RATINGS_LIST = ratings
    _RATINGS_COUNT = len(ratings)

    if _RATINGS_COUNT == 0:
        _log_stats("INFO [stats] omdb_cache sin ratings válidos para auto-umbrales.")

    return _RATINGS_LIST, _RATINGS_COUNT


# -------------------------------------------------------------------
# Ratings list SOLO de títulos SIN RT
# -------------------------------------------------------------------
def _load_imdb_ratings_list_no_rt_from_cache() -> tuple[list[float], int]:
    """
    Igual que _load_imdb_ratings_list_from_cache, pero SOLO para
    títulos cuya entrada de OMDb NO tiene Rotten Tomatoes (rt_score is None).
    """
    global _RATINGS_NO_RT_LIST, _RATINGS_NO_RT_COUNT

    if _RATINGS_NO_RT_LIST is not None:
        return _RATINGS_NO_RT_LIST, _RATINGS_NO_RT_COUNT

    ratings: list[float] = []
    if isinstance(omdb_cache, dict):
        for data in omdb_cache.values():
            if not isinstance(data, dict):
                continue
            r = parse_imdb_rating_from_omdb(data)
            if r is None:
                continue
            rt = parse_rt_score_from_omdb(data)
            if rt is not None:
                # Solo queremos las que NO tienen RT
                continue
            ratings.append(float(r))

    ratings.sort()
    _RATINGS_NO_RT_LIST = ratings
    _RATINGS_NO_RT_COUNT = len(ratings)

    if _RATINGS_NO_RT_COUNT == 0:
        _log_stats(
            "INFO [stats] omdb_cache sin títulos válidos para auto-umbrales NO_RT."
        )

    return _RATINGS_NO_RT_LIST, _RATINGS_NO_RT_COUNT


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    """
    Percentil sencillo sobre lista ORDENADA de floats.
    """
    if not sorted_vals:
        return None

    p = max(0.0, min(1.0, p))
    n = len(sorted_vals)

    if p == 0:
        return sorted_vals[0]
    if p == 1:
        return sorted_vals[-1]

    idx = int(p * (n - 1))
    return sorted_vals[idx]


# -------------------------------------------------------------------
# Auto-umbral KEEP (todos los títulos)
# -------------------------------------------------------------------
def get_auto_keep_rating_threshold() -> float:
    """
    Devuelve el umbral de rating para KEEP (auto-ajustado o fallback fijo).
    """
    global _AUTO_KEEP_RATING_THRESHOLD

    if _AUTO_KEEP_RATING_THRESHOLD is not None:
        return _AUTO_KEEP_RATING_THRESHOLD

    ratings, n = _load_imdb_ratings_list_from_cache()

    if n >= RATING_MIN_TITLES_FOR_AUTO:
        val = _percentile(ratings, AUTO_KEEP_RATING_PERCENTILE)
        if val is not None:
            _AUTO_KEEP_RATING_THRESHOLD = float(val)
            _log_stats(
                f"INFO [stats] IMDB_KEEP_MIN_RATING auto-ajustada (global): "
                f"{val:.3f} (p={AUTO_KEEP_RATING_PERCENTILE}, n={n})"
            )
            return _AUTO_KEEP_RATING_THRESHOLD

    # Fallback
    _AUTO_KEEP_RATING_THRESHOLD = IMDB_KEEP_MIN_RATING
    _log_stats(
        f"INFO [stats] Fallback IMDB_KEEP_MIN_RATING={IMDB_KEEP_MIN_RATING} "
        f"(n={n} < RATING_MIN_TITLES_FOR_AUTO={RATING_MIN_TITLES_FOR_AUTO})"
    )
    return _AUTO_KEEP_RATING_THRESHOLD


# -------------------------------------------------------------------
# Auto-umbral DELETE (todos los títulos)
# -------------------------------------------------------------------
def get_auto_delete_rating_threshold() -> float:
    """
    Devuelve el umbral de rating para DELETE (auto-ajustado o fallback fijo).
    """
    global _AUTO_DELETE_RATING_THRESHOLD

    if _AUTO_DELETE_RATING_THRESHOLD is not None:
        return _AUTO_DELETE_RATING_THRESHOLD

    ratings, n = _load_imdb_ratings_list_from_cache()

    if n >= RATING_MIN_TITLES_FOR_AUTO:
        val = _percentile(ratings, AUTO_DELETE_RATING_PERCENTILE)
        if val is not None:
            _AUTO_DELETE_RATING_THRESHOLD = float(val)
            _log_stats(
                f"INFO [stats] IMDB_DELETE_MAX_RATING auto-ajustada (global): "
                f"{val:.3f} (p={AUTO_DELETE_RATING_PERCENTILE}, n={n})"
            )
            return _AUTO_DELETE_RATING_THRESHOLD

    # Fallback
    _AUTO_DELETE_RATING_THRESHOLD = IMDB_DELETE_MAX_RATING
    _log_stats(
        f"INFO [stats] Fallback IMDB_DELETE_MAX_RATING={IMDB_DELETE_MAX_RATING} "
        f"(n={n} < RATING_MIN_TITLES_FOR_AUTO={RATING_MIN_TITLES_FOR_AUTO})"
    )
    return _AUTO_DELETE_RATING_THRESHOLD


# -------------------------------------------------------------------
# Auto-umbral KEEP para pelis SIN RT
# -------------------------------------------------------------------
def get_auto_keep_rating_threshold_no_rt() -> float:
    """
    Umbral KEEP específico para películas que NO tienen Rotten Tomatoes.
    Usa la distribución de ratings solo de títulos sin RT.
    """
    global _AUTO_KEEP_RATING_THRESHOLD_NO_RT

    if _AUTO_KEEP_RATING_THRESHOLD_NO_RT is not None:
        return _AUTO_KEEP_RATING_THRESHOLD_NO_RT

    ratings, n = _load_imdb_ratings_list_no_rt_from_cache()

    if n >= RATING_MIN_TITLES_FOR_AUTO:
        val = _percentile(ratings, AUTO_KEEP_RATING_PERCENTILE)
        if val is not None:
            _AUTO_KEEP_RATING_THRESHOLD_NO_RT = float(val)
            _log_stats(
                f"INFO [stats] IMDB_KEEP_MIN_RATING auto-ajustada (SIN_RT): "
                f"{val:.3f} (p={AUTO_KEEP_RATING_PERCENTILE}, n={n})"
            )
            return _AUTO_KEEP_RATING_THRESHOLD_NO_RT

    # Fallback → reutilizamos el umbral global
    _AUTO_KEEP_RATING_THRESHOLD_NO_RT = get_auto_keep_rating_threshold()
    _log_stats(
        "INFO [stats] Fallback IMDB_KEEP_MIN_RATING_NO_RT usando umbral global "
        f"{_AUTO_KEEP_RATING_THRESHOLD_NO_RT:.3f} (n_NO_RT={n} < RATING_MIN_TITLES_FOR_AUTO={RATING_MIN_TITLES_FOR_AUTO})"
    )
    return _AUTO_KEEP_RATING_THRESHOLD_NO_RT


# -------------------------------------------------------------------
# Auto-umbral DELETE para pelis SIN RT
# -------------------------------------------------------------------
def get_auto_delete_rating_threshold_no_rt() -> float:
    """
    Umbral DELETE específico para películas que NO tienen Rotten Tomatoes.
    Usa la distribución de ratings solo de títulos sin RT.
    """
    global _AUTO_DELETE_RATING_THRESHOLD_NO_RT

    if _AUTO_DELETE_RATING_THRESHOLD_NO_RT is not None:
        return _AUTO_DELETE_RATING_THRESHOLD_NO_RT

    ratings, n = _load_imdb_ratings_list_no_rt_from_cache()

    if n >= RATING_MIN_TITLES_FOR_AUTO:
        val = _percentile(ratings, AUTO_DELETE_RATING_PERCENTILE)
        if val is not None:
            _AUTO_DELETE_RATING_THRESHOLD_NO_RT = float(val)
            _log_stats(
                f"INFO [stats] IMDB_DELETE_MAX_RATING auto-ajustada (SIN_RT): "
                f"{val:.3f} (p={AUTO_DELETE_RATING_PERCENTILE}, n={n})"
            )
            return _AUTO_DELETE_RATING_THRESHOLD_NO_RT

    # Fallback → reutilizamos el umbral global
    _AUTO_DELETE_RATING_THRESHOLD_NO_RT = get_auto_delete_rating_threshold()
    _log_stats(
        "INFO [stats] Fallback IMDB_DELETE_MAX_RATING_NO_RT usando umbral global "
        f"{_AUTO_DELETE_RATING_THRESHOLD_NO_RT:.3f} (n_NO_RT={n} < RATING_MIN_TITLES_FOR_AUTO={RATING_MIN_TITLES_FOR_AUTO})"
    )
    return _AUTO_DELETE_RATING_THRESHOLD_NO_RT