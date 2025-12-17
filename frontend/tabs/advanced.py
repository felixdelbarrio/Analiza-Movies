from __future__ import annotations

from typing import Sequence

import pandas as pd
import streamlit as st

from frontend.components import aggrid_with_row_click, render_detail_card


def _safe_unique_sorted(df: pd.DataFrame, col: str) -> list[str]:
    """Valores únicos no vacíos/NaN de una columna, ordenados alfabéticamente."""
    if col not in df.columns:
        return []
    return (
        df[col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .replace({"": None})
        .dropna()
        .unique()
        .tolist()
    )


def _ensure_numeric_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Devuelve una serie numérica segura; si no existe la columna, rellena con 0."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def render(df_all: pd.DataFrame) -> None:
    """Pestaña 3: Búsqueda avanzada."""
    st.write("### Búsqueda avanzada")

    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        st.info("No hay datos para búsqueda avanzada.")
        return

    df_view = df_all.copy()

    # ----------------------------------------------------------------
    # Filtros superiores
    # ----------------------------------------------------------------
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    # Biblioteca
    with col_f1:
        libraries = sorted(_safe_unique_sorted(df_view, "library"))
        lib_filter: Sequence[str] = st.multiselect(
            "Biblioteca",
            libraries,
            key="lib_filter_advanced",
        )

    # Decisión
    with col_f2:
        decisions = ["DELETE", "MAYBE", "KEEP", "UNKNOWN"]
        dec_filter: Sequence[str] = st.multiselect(
            "Decisión",
            decisions,
            default=decisions,
        )

    # IMDb rating mínimo
    with col_f3:
        min_imdb: float = st.slider("IMDb mínimo", 0.0, 10.0, 0.0, 0.1)

    # IMDb votos mínimos
    with col_f4:
        min_votes: int = st.slider("IMDb votos mínimos", 0, 200_000, 0, 1_000)

    # ----------------------------------------------------------------
    # Aplicar filtros
    # ----------------------------------------------------------------
    if lib_filter and "library" in df_view.columns:
        df_view = df_view[df_view["library"].isin(lib_filter)]

    if dec_filter and "decision" in df_view.columns:
        df_view = df_view[df_view["decision"].isin(dec_filter)]

    imdb_series = _ensure_numeric_column(df_view, "imdb_rating")
    votes_series = _ensure_numeric_column(df_view, "imdb_votes")

    df_view = df_view[
        (imdb_series >= float(min_imdb)) & (votes_series >= int(min_votes))
    ]

    st.write(f"Resultados: {len(df_view)} película(s)")

    if df_view.empty:
        st.info("No hay resultados que coincidan con los filtros actuales.")
        return

    # ----------------------------------------------------------------
    # Grid + detalle
    # ----------------------------------------------------------------
    col_grid, col_detail = st.columns([2, 1])

    with col_grid:
        selected_row = aggrid_with_row_click(df_view, "advanced")

    with col_detail:
        render_detail_card(selected_row, button_key_prefix="advanced")