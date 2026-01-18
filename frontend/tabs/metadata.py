"""
metadata_tab.py

Pestaña 6 del dashboard (Streamlit): Corrección de metadata (sugerencias).

Este módulo carga y muestra el CSV de sugerencias generado por el backend
(p.ej. `metadata_fix.csv` / `metadata_suggestions.csv`) y permite:

- Filtrar por biblioteca (`library`) y por tipo de acción sugerida (`action`).
- Visualizar el resultado filtrado en una tabla.
- Descargar el CSV filtrado.

Diseño:
- Lectura defensiva del CSV (errores → DataFrame vacío + mensaje).
- Evitar dependencias del backend.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from frontend.components import aggrid_readonly

DEFAULT_EXPORT_NAME = "metadata_suggestions_filtered.csv"


def _load_metadata_csv(path: str) -> pd.DataFrame:
    """
    Carga defensiva del CSV. Si falla, devuelve DataFrame vacío y muestra error en UI.
    """
    try:
        return pd.read_csv(path, dtype="string", encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        st.error(f"Error leyendo CSV de sugerencias: {exc}")
        return pd.DataFrame()


def _render_df_body(df_meta: pd.DataFrame) -> None:
    st.write(
        "Este CSV contiene sugerencias de posibles errores de metadata en Plex.\n\n"
        "- Puedes filtrar por biblioteca y acción sugerida.\n"
        "- Puedes descargar el resultado filtrado como CSV."
    )

    # -------------------------
    # Filtros
    # -------------------------
    col_f1, col_f2 = st.columns(2)

    if "library" in df_meta.columns:
        with col_f1:
            series = df_meta["library"].astype("string").fillna("").map(str.strip)
            series = series.mask(series == "", pd.NA).dropna()
            libraries = series.unique().tolist()
            libraries.sort()
            library_filter = st.multiselect(
                "Biblioteca (library)",
                libraries,
                key="library_filter_metadata",
            )
    else:
        library_filter = []

    if "action" in df_meta.columns:
        with col_f2:
            series = df_meta["action"].astype("string").fillna("").map(str.strip)
            series = series.mask(series == "", pd.NA).dropna()
            actions = series.unique().tolist()
            actions.sort()
            action_filter = st.multiselect(
                "Acción sugerida",
                actions,
                key="action_filter_metadata",
            )
    else:
        action_filter = []

    df_view = df_meta

    if library_filter and "library" in df_view.columns:
        df_view = df_view[df_view["library"].isin(library_filter)]

    if action_filter and "action" in df_view.columns:
        df_view = df_view[df_view["action"].isin(action_filter)]

    st.write(f"Filas: **{len(df_view)}**")

    aggrid_readonly(df_view, key_suffix="metadata", height=420)

    # -------------------------
    # Exportación
    # -------------------------
    csv_export = df_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Descargar CSV filtrado",
        data=csv_export,
        file_name=DEFAULT_EXPORT_NAME,
        mime="text/csv",
    )


def render_df(df_meta: pd.DataFrame) -> None:
    """
    Renderiza la pestaña usando un DataFrame ya cargado (modo API).

    Args:
        df_meta: DataFrame con sugerencias de metadata.
    """
    st.write("### Corrección de metadata (sugerencias)")

    if df_meta.empty:
        st.info(
            "El CSV de sugerencias de metadata está vacío o no se pudo leer correctamente."
        )
        return

    _render_df_body(df_meta)


def render(metadata_sugg_csv: str) -> None:
    """
    Pestaña 6: Corrección de metadata (sugerencias).

    Args:
        metadata_sugg_csv: ruta al CSV de sugerencias de metadata.
    """
    st.write("### Corrección de metadata (sugerencias)")

    if not metadata_sugg_csv:
        st.info("No se ha especificado ruta para el CSV de sugerencias de metadata.")
        return

    if not os.path.exists(metadata_sugg_csv):
        st.info(
            f"No se encontró el CSV de sugerencias de metadata: `{metadata_sugg_csv}`"
        )
        return

    if not os.path.isfile(metadata_sugg_csv):
        st.warning(f"La ruta indicada no es un fichero: `{metadata_sugg_csv}`")
        return

    df_meta = _load_metadata_csv(metadata_sugg_csv)

    if df_meta.empty:
        st.info(
            "El CSV de sugerencias de metadata está vacío o no se pudo leer correctamente."
        )
        return

    _render_df_body(df_meta)
