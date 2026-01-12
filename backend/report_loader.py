"""
backend/report_loader.py

Carga y prepara los CSVs generados por el pipeline (Plex/DLNA) para el dashboard.

Objetivo
- Leer report_all.csv (obligatorio) y report_filtered.csv (opcional).
- Normalizar tipos para evitar sorpresas (NaN/None, floats indeseados, etc.).
- Añadir columnas derivadas usadas por el frontend (add_derived_columns).
- Devolver DataFrames “seguros” para el render del dashboard.

Principios
- El dashboard debe funcionar aunque el CSV filtrado no exista (filtered_csv=SKIPPED).
- Evitar inferencias raras en campos textuales (URLs, JSON, etc.).
- Si el CSV filtrado falla, seguimos con df_all (no rompemos la app).

Notas
- `TEXT_COLUMNS` se fuerza a texto (string) porque:
  * URLs: a veces vienen como NaN y el frontend espera str.
  * omdb_json: puede ser muy largo y no queremos inferencia a NaN/float.
- `_clean_base_dataframe` elimina columnas que el dashboard no necesita (thumb).

Typing (Pylance/Pyright)
- Pandas runtime acepta dtype=dict[col, dtype].
- Algunos stubs que usa Pylance no modelan ese overload y dan error en `dtype=...`.
  El ignore se aplica SOLO a la línea del argumento `dtype=...`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, TYPE_CHECKING

import pandas as pd

from backend import logger as _logger
from frontend.data_utils import add_derived_columns

if TYPE_CHECKING:
    from pandas._typing import DtypeArg
else:
    DtypeArg = object  # type: ignore[assignment]

TEXT_COLUMNS: Final[tuple[str, ...]] = ("poster_url", "trailer_url", "omdb_json")


def _clean_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas no necesarias para el dashboard (p.ej. 'thumb')."""
    if "thumb" not in df.columns:
        return df
    cols = [c for c in df.columns if c != "thumb"]
    return df.loc[:, cols]


def _cast_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que ciertas columnas se traten como texto (string dtype de pandas)."""
    out = df.copy()
    for col in TEXT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out


def _dtype_map_for_text_columns() -> dict[str, "DtypeArg"]:
    """dtype-map para columnas sensibles (URLs/JSON)."""
    return {c: "string" for c in TEXT_COLUMNS}


def _read_csv_safe(path: Path, *, dtype_map: dict[str, "DtypeArg"]) -> pd.DataFrame:
    """
    Lee un CSV con defaults “seguros” para este proyecto.

    Nota:
    - Pylance puede fallar aquí por stubs (no por pandas real).
    """
    try:
        return pd.read_csv(
            path,
            dtype=dtype_map,  # type: ignore[arg-type]
            encoding="utf-8",
        )
    except Exception as exc:
        _logger.error(f"Error leyendo {path}: {exc}", always=True)
        raise


def load_reports(
    all_csv_path: str,
    filtered_csv_path: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Carga y prepara los DataFrames usados por el dashboard."""
    all_path = Path(all_csv_path)
    if not all_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV completo: {all_csv_path}")

    dtype_map = _dtype_map_for_text_columns()

    df_all = _read_csv_safe(all_path, dtype_map=dtype_map)
    df_all = _cast_text_columns(df_all)
    df_all = add_derived_columns(df_all)
    df_all = _clean_base_dataframe(df_all)

    df_filtered: pd.DataFrame | None = None
    if filtered_csv_path:
        filtered_path = Path(filtered_csv_path)
        if filtered_path.exists():
            try:
                df_filtered = _read_csv_safe(filtered_path, dtype_map=dtype_map)
                df_filtered = _cast_text_columns(df_filtered)
                # df_filtered = add_derived_columns(df_filtered)  # si tu UI lo necesita también aquí
                df_filtered = _clean_base_dataframe(df_filtered)
            except Exception as exc:
                _logger.error(f"Error leyendo {filtered_path}: {exc}", always=True)
                df_filtered = None

    return df_all, df_filtered


__all__ = ["load_reports"]
