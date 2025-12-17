from __future__ import annotations

from typing import Final

import pandas as pd
import streamlit as st

from frontend.components import aggrid_with_row_click, render_detail_card


TITLE_TEXT: Final[str] = "### Todas las películas"


def render(df_all: pd.DataFrame) -> None:
    """Pestaña 1: Todas las películas."""
    st.write(TITLE_TEXT)

    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        st.info("No hay películas para mostrar.")
        return

    df_view = df_all.copy()

    # Ordenar de manera útil si existen las columnas
    sort_candidates = ["decision", "imdb_rating", "imdb_votes", "year"]
    sort_cols = [c for c in sort_candidates if c in df_view.columns]

    if sort_cols:
        # decisión Asc → ORDER: DELETE, MAYBE, KEEP, UNKNOWN
        # resto Desc → mejor rating / más votos arriba
        ascending = [True] + [False] * (len(sort_cols) - 1)
        df_view = df_view.sort_values(
            by=sort_cols,
            ascending=ascending,
            ignore_index=True,
        )

    col_grid, col_detail = st.columns([2, 1])

    with col_grid:
        selected_row = aggrid_with_row_click(df_view, "all")

    with col_detail:
        render_detail_card(selected_row, button_key_prefix="all")