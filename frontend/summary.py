from __future__ import annotations

"""
summary.py

Cálculo de métricas resumen para el dashboard / CLI a partir del DataFrame del
reporte completo.

Objetivo
- Producir un dict con conteos por decisión (KEEP / DELETE / MAYBE) y tamaños
  agregados (GB) cuando existe la columna de tamaño.
- Ser tolerante a DataFrames incompletos: si faltan columnas, no se rompe;
  devuelve valores coherentes (conteos a 0, tamaños None).

Convenciones
- FILE_SIZE_COL = "file_size_gb" (derivada en frontend.data_utils.add_derived_columns)
- DECISION_COL  = "decision" (generada por scoring/analysis)

Logging
- Usa el logger central. Los warnings se emiten solo cuando ayudan a diagnosticar
  que faltan columnas esperadas (p.ej. decision o file_size_gb).
"""

from typing import Final

import pandas as pd

from backend import logger as _logger
from backend.stats import compute_global_imdb_mean_from_df, get_global_imdb_mean_from_cache

FILE_SIZE_COL: Final[str] = "file_size_gb"
DECISION_COL: Final[str] = "decision"


def _sum_size(df: pd.DataFrame, mask: pd.Series | None = None) -> float | None:
    """
    Suma la columna FILE_SIZE_COL (GB). Si se pasa mask, suma solo esas filas.

    Returns:
      - float: suma en GB
      - None: si no existe la columna, la máscara no alinea, o el cálculo falla

    Nota:
    - No fuerza tipos: asume que add_derived_columns ya generó una columna numérica.
      Aun así, este helper es defensivo y devuelve None si algo sale mal.
    """
    if FILE_SIZE_COL not in df.columns:
        return None

    series = df[FILE_SIZE_COL]

    if mask is not None:
        # Alinear índices de forma segura.
        try:
            series = series.loc[mask]
        except Exception as exc:  # pragma: no cover (defensivo)
            _logger.warning(
                f"Error aplicando máscara en '{FILE_SIZE_COL}': {exc}. Devolviendo None."
            )
            return None

    try:
        return float(series.sum(skipna=True))
    except Exception:  # pragma: no cover (defensivo)
        _logger.warning(f"Error sumando '{FILE_SIZE_COL}', devolviendo None")
        return None


def compute_summary(df_all: pd.DataFrame) -> dict[str, object]:
    """
    Calcula métricas resumen a partir del DataFrame completo.

    Este resumen se usa típicamente en el dashboard (KPIs) y en salidas de
    terminal (resumen final).

    Entrada:
      - df_all: DataFrame del CSV completo (puede ser vacío)

    Salida (keys principales):
      - total_count, total_size_gb
      - keep_count, keep_size_gb
      - delete_count, delete_size_gb
      - maybe_count, maybe_size_gb
      - dm_count (DELETE+MAYBE), dm_size_gb
      - imdb_mean_df, imdb_mean_cache

    Robustez:
    - Si falta DECISION_COL, se devuelven conteos 0 y tamaños None para los grupos.
    - Si falta FILE_SIZE_COL, tamaños serán None pero conteos siguen siendo válidos.
    """
    if not isinstance(df_all, pd.DataFrame):
        raise TypeError("df_all debe ser un pandas.DataFrame")

    total_count = int(len(df_all))
    total_size = _sum_size(df_all)

    # Medias IMDb (independientes de 'decision')
    imdb_mean_df = compute_global_imdb_mean_from_df(df_all)
    imdb_mean_cache = get_global_imdb_mean_from_cache()

    # Si falta la columna 'decision', devolvemos un resumen coherente.
    if DECISION_COL not in df_all.columns:
        _logger.warning(
            "Columna 'decision' no encontrada en df_all; devolviendo resumen con conteos 0 por decisión."
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

    decisions = df_all[DECISION_COL]
    keep_mask = decisions == "KEEP"
    del_mask = decisions == "DELETE"
    maybe_mask = decisions == "MAYBE"
    dm_mask = del_mask | maybe_mask  # DELETE+MAYBE

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