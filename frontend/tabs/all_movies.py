"""
all_movies.py

Pestaña “Todas las películas” (Streamlit).

Responsabilidad:
- Mostrar el listado completo (df_all) en una tabla (AgGrid).
- Ordenar de forma “útil” cuando existan columnas relevantes.
- Mostrar el panel lateral de detalle al seleccionar una fila.

Principios:
- No mutar df_all: trabajar con una copia.
- Ser tolerante a columnas ausentes.
- Delegar UI compleja (grid + detalle) a frontend.components.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import streamlit as st

from frontend.components import aggrid_with_row_click, render_detail_card


TITLE_TEXT: Final[str] = "### Todas las películas"


def _sort_all_movies_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena el DataFrame por título (asc) si existe la columna.
    """
    if "title" not in df.columns:
        return df

    try:
        if df["title"].is_monotonic_increasing:
            return df
    except Exception:
        pass

    try:
        return df.sort_values(
            by=["title"],
            ascending=[True],
            na_position="last",
            ignore_index=True,
        )
    except Exception:
        # Degradación segura: si algún dtype raro rompe sort_values, devolvemos sin ordenar.
        return df


def render(df_all: pd.DataFrame) -> None:
    """
    Renderiza la pestaña 1: Todas las películas.

    Args:
        df_all: DataFrame completo (puede estar vacío).
    """
    st.write(TITLE_TEXT)

    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        st.info("No hay películas para mostrar.")
        return

    df_view = _sort_all_movies_view(df_all)

    col_grid, col_detail = st.columns([2, 1])

    with col_grid:
        selected_row = aggrid_with_row_click(
            df_view,
            "all",
            visible_order=[
                "title",
                "year",
                "library",
                "file_size_gb",
                "metacritic_score",
                "imdb_rating",
                "imdb_votes",
                "rt_score",
            ],
            auto_select_first=True,
        )

    with col_detail:
        render_detail_card(selected_row, button_key_prefix="all")
