from __future__ import annotations

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

from typing import Final

import pandas as pd
import streamlit as st

from frontend.components import aggrid_with_row_click, render_detail_card


TITLE_TEXT: Final[str] = "### Todas las películas"


def _sort_all_movies_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena el DataFrame de forma estable y útil si existen columnas candidatas.

    Orden propuesto:
    - decision (asc): para agrupar por acción sugerida (DELETE/MAYBE primero).
    - imdb_rating (desc): mejor rating arriba dentro del grupo.
    - imdb_votes (desc): más votos arriba.
    - year (desc): más reciente arriba.

    Nota:
    - El orden real de 'decision' dependerá de cómo compare strings; esto es suficiente
      para una ordenación básica. Si quieres un orden semántico estricto
      (DELETE, MAYBE, KEEP, UNKNOWN) podemos mapear a un ordinal.
    """
    df_view = df.copy()

    sort_candidates = ["decision", "imdb_rating", "imdb_votes", "year"]
    sort_cols = [c for c in sort_candidates if c in df_view.columns]

    if not sort_cols:
        return df_view

    ascending = [True] + [False] * (len(sort_cols) - 1)

    try:
        return df_view.sort_values(
            by=sort_cols,
            ascending=ascending,
            ignore_index=True,
        )
    except Exception:
        # Degradación segura: si algún dtype raro rompe sort_values, devolvemos sin ordenar.
        return df_view


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
        selected_row = aggrid_with_row_click(df_view, "all")

    with col_detail:
        render_detail_card(selected_row, button_key_prefix="all")