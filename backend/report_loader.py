from __future__ import annotations

"""
report_loader.py

Carga y prepara los CSVs generados por el pipeline (Plex/DLNA) para el dashboard.

Objetivo
- Leer report_all.csv (obligatorio) y report_filtered.csv (opcional).
- Normalizar tipos para evitar sorpresas (NaN/None, floats indeseados, etc.).
- Añadir columnas derivadas usadas por el frontend (add_derived_columns).
- Devolver DataFrames “seguros” para el render del dashboard.

Principios (alineado con el trabajo previo)
- El dashboard debe funcionar aunque el CSV filtrado no exista (porque fue SKIPPED).
- Evitar que pandas infiera tipos raros en campos textuales (URLs, JSON, etc.).
- Mantener el loader tolerante a errores del CSV filtrado: si falla, seguimos con df_all.

Notas
- `TEXT_COLUMNS` se fuerza a texto (string) porque:
  * URLs: a veces vienen como NaN y el frontend espera str.
  * omdb_json: puede ser muy largo y no queremos inferencia a NaN/float.
- `_clean_base_dataframe` elimina columnas que el dashboard no necesita (thumb).
"""

from pathlib import Path
from typing import Final

import pandas as pd

from backend import logger as _logger
from frontend.data_utils import add_derived_columns

# Columnas que el frontend trata como texto (no numéricas) aunque puedan contener NaN.
TEXT_COLUMNS: Final[list[str]] = ["poster_url", "trailer_url", "omdb_json"]


def _clean_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas no necesarias para el dashboard.

    Actualmente:
    - 'thumb' se elimina si existe (mantenemos 'poster_url' como fuente principal).

    Importante:
    - Se devuelve una vista/selección del DF (sin copiar) para ser barato.
    """
    cols = [c for c in df.columns if c != "thumb"]
    return df[cols]


def _cast_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que ciertas columnas se traten como texto (string).

    Decisión:
    - Usamos `astype("string")` (dtype nativo de pandas) para manejar NA de forma consistente.
    - Convertimos a string solo columnas presentes en el DF (robusto a versiones de CSV).

    Nota:
    - Si el CSV trae valores mixtos, `string` evita que pandas convierta a float
      o a objetos con NaN raros.
    """
    df = df.copy()
    for col in TEXT_COLUMNS:
        if col in df.columns:
            try:
                df[col] = df[col].astype("string")
            except Exception:
                # Fallback ultra defensivo: evitamos romper el dashboard por una columna problemática.
                df[col] = df[col].astype(str)
    return df


def _read_csv_safe(path: Path, *, dtype_map: dict[str, object]) -> pd.DataFrame:
    """
    Lee un CSV con defaults “seguros” para este proyecto.

    - dtype: forzamos texto en columnas sensibles (URLs/JSON).
    - encoding: utf-8
    - keep_default_na: True (default) para que pandas reconozca NaN, pero luego
      normalizamos tipos en _cast_text_columns.
    """
    try:
        return pd.read_csv(path, dtype=dtype_map, encoding="utf-8")
    except Exception as exc:
        _logger.error(f"Error leyendo {path}: {exc}")
        raise


def load_reports(all_csv_path: str, filtered_csv_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Carga y prepara los DataFrames usados por el dashboard.

    Parámetros:
    - all_csv_path: ruta a report_all.csv (requerido).
    - filtered_csv_path: ruta a report_filtered.csv (opcional). Puede:
        * ser None
        * no existir (filtered_csv=SKIPPED)
        * existir pero estar corrupto -> se ignora y se devuelve None

    Returns:
    - (df_all, df_filtered)
      df_filtered puede ser None si no hay CSV filtrado o si falló su lectura.
    """
    all_path = Path(all_csv_path)
    if not all_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV completo: {all_csv_path}")

    # Forzar texto en columnas sensibles (URLs/JSON)
    dtype_map: dict[str, object] = {c: "string" for c in TEXT_COLUMNS}

    # --- report_all.csv (obligatorio) ---
    df_all = _read_csv_safe(all_path, dtype_map=dtype_map)
    df_all = _cast_text_columns(df_all)
    df_all = add_derived_columns(df_all)
    df_all = _clean_base_dataframe(df_all)

    # --- report_filtered.csv (opcional) ---
    df_filtered: pd.DataFrame | None = None
    if filtered_csv_path:
        filtered_path = Path(filtered_csv_path)

        # Si no existe, es un caso normal (filtered_csv=SKIPPED).
        if filtered_path.exists():
            try:
                df_filtered = _read_csv_safe(filtered_path, dtype_map=dtype_map)
                df_filtered = _cast_text_columns(df_filtered)
                # IMPORTANTE: el filtrado no necesita columnas derivadas si el dashboard ya las calcula desde df_all,
                # pero si tu UI usa derivadas en ambas tablas, puedes habilitarlo:
                # df_filtered = add_derived_columns(df_filtered)
                df_filtered = _clean_base_dataframe(df_filtered)
            except Exception as exc:
                _logger.error(f"Error leyendo {filtered_path}: {exc}")
                # No lanzamos para permitir que el dashboard funcione con df_all solamente.
                df_filtered = None

    return df_all, df_filtered