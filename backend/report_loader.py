from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from backend import logger as _logger
from frontend.data_utils import add_derived_columns

TEXT_COLUMNS: Final[list[str]] = ["poster_url", "trailer_url", "omdb_json"]


def _clean_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas no necesarias para el dashboard.

    Actualmente solo se elimina 'thumb' si existe.
    """
    cols = [c for c in df.columns if c != "thumb"]
    return df[cols]


def _cast_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que ciertas columnas se traten como texto (str)."""
    df = df.copy()
    for col in TEXT_COLUMNS:
        if col in df.columns:
            # Forzar tipo string, evitando None/NaN problemáticos en el frontend
            df[col] = df[col].astype(str)
    return df


def load_reports(
    all_csv_path: str,
    filtered_csv_path: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Carga y prepara los DataFrames usados por el dashboard.

    Parámetros:
      - all_csv_path: ruta al CSV completo (requerido)
      - filtered_csv_path: ruta al CSV filtrado (opcional)

    Returns:
      - df_all, df_filtered (df_filtered puede ser None)
    """
    all_path = Path(all_csv_path)
    if not all_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV completo: {all_csv_path}")

    dtype_map: dict[str, type[str]] = {c: str for c in TEXT_COLUMNS}

    try:
        # Leer forzando texto en las columnas conocidas para evitar conversiones extrañas
        df_all = pd.read_csv(all_path, dtype=dtype_map, encoding="utf-8")
    except Exception as exc:
        _logger.error(f"Error leyendo {all_path}: {exc}")
        raise

    df_all = _cast_text_columns(df_all)
    df_all = add_derived_columns(df_all)
    df_all = _clean_base_dataframe(df_all)

    df_filtered: pd.DataFrame | None = None
    if filtered_csv_path:
        filtered_path = Path(filtered_csv_path)
        if filtered_path.exists():
            try:
                df_filtered = pd.read_csv(
                    filtered_path,
                    dtype=dtype_map,
                    encoding="utf-8",
                )
                df_filtered = _clean_base_dataframe(df_filtered)
            except Exception as exc:
                _logger.error(f"Error leyendo {filtered_path}: {exc}")
                # No lanzamos para permitir que el dashboard use df_all sólo
                df_filtered = None

    return df_all, df_filtered