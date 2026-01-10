"""
candidates.py

Pestaña “Candidatas a borrar (DELETE / MAYBE)” (Streamlit).

Responsabilidad:
- Mostrar el CSV/DF filtrado (df_filtered) con las candidatas a borrar.
- Re-validar que solo contiene decisiones DELETE/MAYBE por robustez.
- Ordenar de forma útil (peores primero) si hay columnas disponibles.
- Mostrar un resumen corto del conteo de DELETE/MAYBE.
- Renderizar tabla (AgGrid) + panel de detalle.

Principios:
- No mutar df_filtered: trabajar con copias.
- Tolerancia a columnas ausentes (decision, imdb_rating, etc.).
- UI delegada a frontend.components.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import streamlit as st

from frontend.components import aggrid_with_row_click, render_detail_card

TITLE_TEXT: Final[str] = "### Candidatas a borrar (DELETE / MAYBE)"


def _filter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que el DataFrame contenga solo filas con decision DELETE/MAYBE si existe la columna.

    Si no existe la columna 'decision', se devuelve el DF tal cual.
    """
    df_view = df.copy()
    if "decision" in df_view.columns:
        df_view = df_view[df_view["decision"].isin(["DELETE", "MAYBE"])].copy()
    return df_view


def _sort_candidates_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena el DataFrame por columnas típicamente relevantes, si existen:

    - decision (asc): DELETE antes que MAYBE (orden alfabético funciona: DELETE < MAYBE).
    - imdb_rating (desc)
    - imdb_votes (desc)
    - year (desc)
    - file_size (desc)

    Nota:
    - Mantengo el esquema original del fichero que me pasaste (decision asc, resto desc).
      Si prefieres “más borrable arriba” (rating asc), lo ajusto.
    """
    df_view = df.copy()

    sort_cols: list[str] = []
    for col in ("decision", "imdb_rating", "imdb_votes", "year", "file_size"):
        if col in df_view.columns:
            sort_cols.append(col)

    if not sort_cols:
        return df_view

    ascending = [True] + [False] * (len(sort_cols) - 1)

    try:
        return df_view.sort_values(by=sort_cols, ascending=ascending, ignore_index=True)
    except Exception:
        return df_view


def _render_quick_caption(df: pd.DataFrame) -> None:
    """
    Muestra un resumen corto: total, DELETE, MAYBE.

    Si falta 'decision', muestra solo total.
    """
    total = int(len(df))
    if "decision" not in df.columns:
        st.caption(f"{total} candidata(s)")
        return

    delete_count = int((df["decision"] == "DELETE").sum())
    maybe_count = int((df["decision"] == "MAYBE").sum())
    st.caption(f"{total} candidata(s): DELETE={delete_count}, MAYBE={maybe_count}")


def render(df_all: pd.DataFrame, df_filtered: pd.DataFrame | None) -> None:
    """
    Renderiza la pestaña 2: Candidatas a borrar (DELETE/MAYBE).

    Args:
        df_all: DataFrame completo (no se usa aquí, pero se mantiene por compatibilidad de firma).
        df_filtered: DataFrame filtrado (puede ser None o vacío).
    """
    st.write(TITLE_TEXT)

    # Pyright-friendly: separar el narrowing en dos pasos.
    if df_filtered is None:
        st.info("No hay CSV filtrado o está vacío.")
        return

    if df_filtered.empty:
        st.info("No hay CSV filtrado o está vacío.")
        return

    df_view = _filter_candidates(df_filtered)

    if df_view.empty:
        st.info("No hay películas marcadas como DELETE o MAYBE.")
        return

    df_view = _sort_candidates_view(df_view)

    _render_quick_caption(df_view)

    col_grid, col_detail = st.columns([2, 1])

    with col_grid:
        selected_row = aggrid_with_row_click(df_view, "filtered")

    with col_detail:
        # prefix distinto para evitar colisiones con pestañas (all/advanced/etc.)
        render_detail_card(selected_row, button_key_prefix="candidates")
