from __future__ import annotations

from typing import Final

import pandas as pd
import streamlit as st

from frontend.components import aggrid_with_row_click, render_detail_card


TITLE_TEXT: Final[str] = "### Candidatas a borrar (DELETE / MAYBE)"


def render(df_all: pd.DataFrame, df_filtered: pd.DataFrame | None) -> None:
    """Pestaña 2: Candidatas a borrar (DELETE / MAYBE)."""
    st.write(TITLE_TEXT)

    # Nada que mostrar
    if not isinstance(df_filtered, pd.DataFrame) or df_filtered.empty:
        st.info("No hay CSV filtrado o está vacío.")
        return

    # Trabajamos con copia para no modificar el DataFrame original
    df_view = df_filtered.copy()

    # Aseguramos que solo contiene DELETE / MAYBE por si el CSV tuviera más cosas
    if "decision" in df_view.columns:
        df_view = df_view[df_view["decision"].isin(["DELETE", "MAYBE"])].copy()

    if df_view.empty:
        st.info("No hay películas marcadas como DELETE o MAYBE.")
        return

    # Ordenar por algo útil si existen las columnas
    sort_cols: list[str] = []
    for col in ("decision", "imdb_rating", "imdb_votes", "year", "file_size"):
        if col in df_view.columns:
            sort_cols.append(col)

    if sort_cols:
        # decisión asc, resto desc (borrados duros + peor rating + más tamaño arriba)
        ascending = [True] + [False] * (len(sort_cols) - 1)
        df_view = df_view.sort_values(by=sort_cols, ascending=ascending, ignore_index=True)

    # Pequeño resumen rápido
    total = len(df_view)
    delete_count = int((df_view["decision"] == "DELETE").sum()) if "decision" in df_view.columns else 0
    maybe_count = int((df_view["decision"] == "MAYBE").sum()) if "decision" in df_view.columns else 0
    st.caption(f"{total} candidata(s): DELETE={delete_count}, MAYBE={maybe_count}")

    col_grid, col_detail = st.columns([2, 1])

    with col_grid:
        selected_row = aggrid_with_row_click(df_view, "filtered")

    with col_detail:
        # usamos un prefix distinto para no chocar con la pestaña "all"
        render_detail_card(selected_row, button_key_prefix="candidates")