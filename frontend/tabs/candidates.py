"""
candidates.py

Pestaña “Duplicadas por IMDb ID” (Streamlit).

Responsabilidad:
- Detectar películas duplicadas por imdb_id (mismo ID en una o varias bibliotecas).
- Mostrar una vista filtrada con los duplicados.
- Ordenar y resumir el resultado.
- Renderizar tabla (AgGrid) + panel de detalle.

Principios:
- No mutar df_all: trabajar con copias.
- Tolerancia a columnas ausentes (imdb_id, library, year, etc.).
- UI delegada a frontend.components.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import streamlit as st

from frontend.components import aggrid_with_row_click, render_detail_card

TITLE_TEXT: Final[str] = "### Duplicadas por IMDb ID"


def _normalize_imdb_id(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s in {"nan", "none"}:
        return None
    return s


def _filter_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve solo filas con imdb_id duplicado (mismo ID repetido).
    """
    if "imdb_id" not in df.columns:
        return df.iloc[0:0].copy()

    df_view = df.copy()
    imdb_norm = df_view["imdb_id"].apply(_normalize_imdb_id)
    df_view["_imdb_norm"] = imdb_norm
    df_view = df_view[df_view["_imdb_norm"].notna()].copy()

    if df_view.empty:
        return df_view

    counts = df_view["_imdb_norm"].value_counts()
    dup_ids = counts[counts > 1]
    if dup_ids.empty:
        return df_view.iloc[0:0].copy()

    df_view = df_view[df_view["_imdb_norm"].isin(dup_ids.index)].copy()
    df_view["dup_count"] = df_view["_imdb_norm"].map(dup_ids).fillna(0).astype(int)
    df_view = df_view.drop(columns=["_imdb_norm"])
    return df_view


def _sort_duplicates_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena por duplicados primero y luego por campos útiles si existen.
    """
    df_view = df.copy()

    sort_cols: list[str] = []
    for col in ("dup_count", "imdb_id", "library", "title", "year"):
        if col in df_view.columns:
            sort_cols.append(col)

    if not sort_cols:
        return df_view

    ascending = [False] + [True] * (len(sort_cols) - 1)

    try:
        return df_view.sort_values(by=sort_cols, ascending=ascending, ignore_index=True)
    except Exception:
        return df_view


def _render_quick_caption(df: pd.DataFrame) -> None:
    """
    Muestra un resumen corto: duplicados (filas) y grupos (imdb_id).
    """
    total = int(len(df))
    if "imdb_id" not in df.columns:
        st.caption(f"{total} duplicada(s)")
        return
    unique_ids = df["imdb_id"].dropna().nunique()
    st.caption(f"{total} duplicada(s) en {unique_ids} título(s) por IMDb ID")


def render(df_all: pd.DataFrame, df_filtered: pd.DataFrame | None) -> None:
    """
    Renderiza la pestaña 2: Duplicadas por IMDb ID.

    Args:
        df_all: DataFrame completo (fuente principal).
        df_filtered: DataFrame filtrado (ignorado; se mantiene por compatibilidad de firma).
    """
    st.write(TITLE_TEXT)

    if "imdb_id" not in df_all.columns:
        st.info("No hay columna imdb_id para detectar duplicados.")
        return

    df_view = _filter_duplicates(df_all)

    if df_view.empty:
        st.info("No hay duplicados por IMDb ID.")
        return

    df_view = _sort_duplicates_view(df_view)

    _render_quick_caption(df_view)

    col_grid, col_detail = st.columns([2, 1])

    with col_grid:
        selected_row = aggrid_with_row_click(df_view, "filtered")

    with col_detail:
        # prefix distinto para evitar colisiones con pestañas (all/advanced/etc.)
        render_detail_card(selected_row, button_key_prefix="candidates")
