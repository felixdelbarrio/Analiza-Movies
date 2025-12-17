from __future__ import annotations

from typing import Final

import pandas as pd

from backend import logger as _logger
from backend.stats import (
    compute_global_imdb_mean_from_df,
    get_global_imdb_mean_from_cache,
)


FILE_SIZE_COL: Final[str] = "file_size_gb"
DECISION_COL: Final[str] = "decision"


def _sum_size(df: pd.DataFrame, mask: pd.Series | None = None) -> float | None:
    """Suma la columna `file_size_gb` filtrada por mask si existe.

    Devuelve:
      - float con la suma en GB.
      - None si la columna no existe o si hay un error al sumar.
    """
    if FILE_SIZE_COL not in df.columns:
        return None

    series = df[FILE_SIZE_COL]
    if mask is not None:
        # Nos aseguramos de alinear índices; si falla, log y devolvemos None
        try:
            series = series.loc[mask]
        except Exception as exc:  # pragma: no cover (defensivo)
            _logger.warning(
                f"Error aplicando máscara en '{FILE_SIZE_COL}': {exc}. "
                "Devolviendo None."
            )
            return None

    try:
        total = float(series.sum(skipna=True))
    except Exception:  # pragma: no cover (defensivo)
        _logger.warning("Error sumando 'file_size_gb', devolviendo None")
        return None

    return total


def compute_summary(df_all: pd.DataFrame) -> dict[str, object]:
    """Calcula métricas resumen a partir del DataFrame completo.

    Acepta un DataFrame (posiblemente vacío). No lanza si faltan columnas: en
    esos casos devuelve conteos igual a 0 y tamaños en None.
    """
    if not isinstance(df_all, pd.DataFrame):
        raise TypeError("df_all debe ser un pandas.DataFrame")

    # Total de películas
    total_count = int(len(df_all))

    # Tamaño total en GB (si existe la columna file_size_gb)
    total_size = _sum_size(df_all)

    # Medias IMDb (aunque falte 'decision', estas métricas siguen siendo útiles)
    imdb_mean_df = compute_global_imdb_mean_from_df(df_all)
    imdb_mean_cache = get_global_imdb_mean_from_cache()

    # Si falta la columna 'decision', lo registramos y devolvemos ceros coherentes
    if DECISION_COL not in df_all.columns:
        _logger.warning(
            "Columna 'decision' no encontrada en df_all; "
            "devolviendo resumen con conteos 0."
        )
        return {
            "total_count": total_count,
            "total_size_gb": total_size,
            "keep_count": 0,
            "keep_size_gb": None,
            "dm_count": 0,
            "dm_size_gb": None,
            "delete_count": 0,
            "delete_size_gb": None,
            "maybe_count": 0,
            "maybe_size_gb": None,
            "imdb_mean_df": imdb_mean_df,
            "imdb_mean_cache": imdb_mean_cache,
        }

    # Masks por decisión
    decisions = df_all[DECISION_COL]
    keep_mask = decisions == "KEEP"
    del_mask = decisions == "DELETE"
    maybe_mask = decisions == "MAYBE"
    dm_mask = del_mask | maybe_mask

    keep_count = int(keep_mask.sum())
    delete_count = int(del_mask.sum())
    maybe_count = int(maybe_mask.sum())
    dm_count = int(dm_mask.sum())

    keep_size = _sum_size(df_all, keep_mask)
    delete_size = _sum_size(df_all, del_mask)
    maybe_size = _sum_size(df_all, maybe_mask)
    dm_size = _sum_size(df_all, dm_mask)

    return {
        "total_count": total_count,
        "total_size_gb": total_size,
        "keep_count": keep_count,
        "keep_size_gb": keep_size,
        "dm_count": dm_count,
        "dm_size_gb": dm_size,
        "delete_count": delete_count,
        "delete_size_gb": delete_size,
        "maybe_count": maybe_count,
        "maybe_size_gb": maybe_size,
        "imdb_mean_df": imdb_mean_df,
        "imdb_mean_cache": imdb_mean_cache,
    }