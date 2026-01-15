"""
frontend/summary.py

Cálculo de métricas resumen para el dashboard a partir del DataFrame del
reporte completo (report_all.csv).

Fuentes:
- SOLO DataFrame (normalmente cargado desde reports/report_all.csv)

Objetivo:
- Conteos por decisión (KEEP / DELETE / MAYBE) y tamaños agregados (GB)
- Ser tolerante a DataFrames incompletos: si faltan columnas, no se rompe;
  devuelve valores coherentes (conteos a 0, tamaños None).

Convenciones:
- FILE_SIZE_COL = "file_size_gb" (derivada en frontend.data_utils.add_derived_columns)
- DECISION_COL  = "decision" (generada por scoring/analysis)
"""

from __future__ import annotations

from typing import Final, TypedDict

import pandas as pd

from frontend.front_logger import log_warning
from frontend.front_stats import compute_global_imdb_mean_from_df

FILE_SIZE_COL: Final[str] = "file_size_gb"
DECISION_COL: Final[str] = "decision"


class Summary(TypedDict):
    total_count: int
    total_size_gb: float | None
    keep_count: int
    keep_size_gb: float | None
    dm_count: int
    dm_size_gb: float | None
    delete_count: int
    delete_size_gb: float | None
    maybe_count: int
    maybe_size_gb: float | None
    imdb_mean_df: float | None


def _sum_size(df: pd.DataFrame, mask: pd.Series | None = None) -> float | None:
    """
    Suma la columna FILE_SIZE_COL (GB). Si se pasa mask, suma solo esas filas.

    Returns:
      - float: suma en GB
      - None: si no existe la columna, la máscara no alinea, o el cálculo falla
    """
    if FILE_SIZE_COL not in df.columns:
        return None

    series = df[FILE_SIZE_COL]

    if mask is not None:
        try:
            series = series.loc[mask]
        except Exception as exc:  # defensivo
            log_warning(
                f"[summary] Error aplicando máscara en '{FILE_SIZE_COL}': {exc!r}. Devolviendo None."
            )
            return None

    try:
        return float(series.sum(skipna=True))
    except Exception as exc:  # defensivo
        log_warning(
            f"[summary] Error sumando '{FILE_SIZE_COL}': {exc!r}. Devolviendo None."
        )
        return None


def compute_summary(df_all: pd.DataFrame) -> Summary:
    """
    Calcula métricas resumen a partir del DataFrame completo.

    Salida (keys principales):
      - total_count, total_size_gb
      - keep_count, keep_size_gb
      - delete_count, delete_size_gb
      - maybe_count, maybe_size_gb
      - dm_count (DELETE+MAYBE), dm_size_gb
      - imdb_mean_df
    """
    if not isinstance(df_all, pd.DataFrame):
        raise TypeError("df_all debe ser un pandas.DataFrame")

    total_count = int(len(df_all))
    total_size = _sum_size(df_all)

    imdb_mean_df = compute_global_imdb_mean_from_df(df_all)

    if DECISION_COL not in df_all.columns:
        log_warning(
            "[summary] Columna 'decision' no encontrada; devolviendo conteos 0 por decisión."
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
        }

    decisions = df_all[DECISION_COL]
    keep_mask = decisions == "KEEP"
    del_mask = decisions == "DELETE"
    maybe_mask = decisions == "MAYBE"
    dm_mask = del_mask | maybe_mask

    keep_count = int(keep_mask.sum())
    delete_count = int(del_mask.sum())
    maybe_count = int(maybe_mask.sum())
    dm_count = int(dm_mask.sum())

    return {
        "total_count": total_count,
        "total_size_gb": total_size,
        "keep_count": keep_count,
        "keep_size_gb": _sum_size(df_all, keep_mask),
        "dm_count": dm_count,
        "dm_size_gb": _sum_size(df_all, dm_mask),
        "delete_count": delete_count,
        "delete_size_gb": _sum_size(df_all, del_mask),
        "maybe_count": maybe_count,
        "maybe_size_gb": _sum_size(df_all, maybe_mask),
        "imdb_mean_df": imdb_mean_df,
    }
