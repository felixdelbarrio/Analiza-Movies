from __future__ import annotations

import os
from typing import Final

import pandas as pd
import streamlit as st

# Nombre por defecto del fichero exportado
DEFAULT_EXPORT_NAME: Final[str] = "metadata_suggestions_filtered.csv"


@st.cache_data(show_spinner=False)
def _load_metadata_csv(path: str) -> pd.DataFrame:
    """Carga el CSV de sugerencias de metadata de forma cacheada y defensiva."""
    try:
        dtype_hint: dict[str, str] = {
            "library": "string",
            "action": "string",
        }
        return pd.read_csv(path, dtype=dtype_hint, encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        st.error(f"Error leyendo CSV de sugerencias: {exc}")
        return pd.DataFrame()


def render(metadata_sugg_csv: str) -> None:
    """Pestaña 6: Corrección de metadata (sugerencias)."""
    st.write("### Corrección de metadata (sugerencias)")

    if not metadata_sugg_csv:
        st.info("No se ha especificado ruta para el CSV de sugerencias de metadata.")
        return

    if not os.path.exists(metadata_sugg_csv):
        st.info(f"No se encontró el CSV de sugerencias de metadata: `{metadata_sugg_csv}`")
        return

    if not os.path.isfile(metadata_sugg_csv):
        st.warning(f"La ruta indicada no es un fichero: `{metadata_sugg_csv}`")
        return

    df_meta = _load_metadata_csv(metadata_sugg_csv)

    if df_meta.empty:
        st.info("El CSV de sugerencias de metadata está vacío o no se pudo leer correctamente.")
        return

    st.write(
        "Este CSV contiene sugerencias de posibles errores de metadata en Plex.\n\n"
        "- Puedes filtrar por biblioteca y acción sugerida.\n"
        "- Puedes descargar el resultado filtrado como CSV."
    )

    # -------------------------
    # Filtros
    # -------------------------
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        if "library" in df_meta.columns:
            libs = (
                df_meta["library"]
                .dropna()
                .astype(str)
                .map(str.strip)
                .replace({"": None})
                .dropna()
                .unique()
                .tolist()
            )
            libs.sort()
            lib_filter = st.multiselect("Biblioteca", libs, key="lib_filter_metadata")
        else:
            lib_filter = []
            st.warning("El CSV no tiene columna 'library'.")

    with col_f2:
        if "action" in df_meta.columns:
            actions = (
                df_meta["action"]
                .dropna()
                .astype(str)
                .map(str.strip)
                .replace({"": None})
                .dropna()
                .unique()
                .tolist()
            )
            actions.sort()
            action_filter = st.multiselect("Acción sugerida", actions, key="action_filter_metadata")
        else:
            action_filter = []
            st.info("El CSV no incluye columna 'action'.")

    df_view = df_meta.copy()

    # --------------------------------------------
    # Aplicar filtros
    # --------------------------------------------
    if lib_filter and "library" in df_view.columns:
        df_view = df_view[df_view["library"].isin(lib_filter)]

    if action_filter and "action" in df_view.columns:
        df_view = df_view[df_view["action"].isin(action_filter)]

    st.write(f"Filas después de filtrar: **{len(df_view)}**")

    if df_view.empty:
        st.info("No hay filas que coincidan con los filtros seleccionados.")
        return

    # -------------------------
    # Tabla Streamlit corregida (SÍ: width="stretch", NO: use_container_width)
    # -------------------------
    st.dataframe(
        df_view,
        width="stretch",
        height=400,
    )

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