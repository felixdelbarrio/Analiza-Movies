from __future__ import annotations

"""
frontend/config_front_scoring.py

Configs/constantes usadas por el FRONT para cálculos y visualizaciones de stats.

Requisitos:
- NO importar backend.
- Leer SOLO desde .env.front (cargado por config_front_base).
- Si falta una variable => usar defaults del código.
"""

from typing import Final

from frontend.config_front_base import _get_env_float, _get_env_int

# Defaults "sensatos" (ajústalos si quieres que coincidan 1:1 con backend)
BAYES_GLOBAL_MEAN_DEFAULT: Final[float] = _get_env_float("FRONT_BAYES_GLOBAL_MEAN_DEFAULT", 6.8)
BAYES_MIN_TITLES_FOR_GLOBAL_MEAN: Final[int] = _get_env_int("FRONT_BAYES_MIN_TITLES_FOR_GLOBAL_MEAN", 250)

RATING_MIN_TITLES_FOR_AUTO: Final[int] = _get_env_int("FRONT_RATING_MIN_TITLES_FOR_AUTO", 200)

AUTO_KEEP_RATING_PERCENTILE: Final[float] = _get_env_float("FRONT_AUTO_KEEP_RATING_PERCENTILE", 0.70)
AUTO_DELETE_RATING_PERCENTILE: Final[float] = _get_env_float("FRONT_AUTO_DELETE_RATING_PERCENTILE", 0.20)

IMDB_KEEP_MIN_RATING: Final[float] = _get_env_float("FRONT_IMDB_KEEP_MIN_RATING", 7.0)
IMDB_DELETE_MAX_RATING: Final[float] = _get_env_float("FRONT_IMDB_DELETE_MAX_RATING", 5.8)