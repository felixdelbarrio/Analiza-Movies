"""
advanced.py

Pestaña “Búsqueda avanzada” (Streamlit).

Responsabilidad:
- Mostrar filtros (biblioteca, decisión, umbrales de IMDb rating y votos).
- Aplicar filtros sobre df_all y mostrar resultados.
- Renderizar grid (AgGrid) y tarjeta de detalle (panel lateral).

Principios:
- No mutar df_all: trabajar sobre una copia.
- Ser tolerante a columnas ausentes: degradar de forma segura (sin crash).
- Mantener este módulo centrado en UI/filtrado; la lógica de render se delega a
  frontend.components (grid + detalle).
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
import streamlit as st

from frontend.components import aggrid_with_row_click, render_detail_card

_DECISION_LABELS: dict[str, str] = {
    "DELETE": "🟥 DELETE",
    "MAYBE": "🟨 MAYBE",
    "KEEP": "🟩 KEEP",
    "UNKNOWN": "⬜ UNKNOWN",
}


# ============================================================================
# Helpers
# ============================================================================


def _safe_unique_sorted(df: pd.DataFrame, col: str) -> list[str]:
    """
    Devuelve valores únicos no vacíos/NaN de una columna, ordenados alfabéticamente.

    Nota mypy:
    - pandas .unique().tolist() suele estar tipado como Any/list[Any] en stubs,
      así que construimos explícitamente list[str].
    """
    if col not in df.columns:
        return []

    raw: list[Any] = (
        df[col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .replace({"": None})
        .dropna()
        .unique()
        .tolist()
    )

    out: list[str] = []
    for v in raw:
        s = str(v).strip()
        if s:
            out.append(s)

    out.sort()
    return out


def _ensure_numeric_column(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Devuelve una serie numérica segura:
    - Si la columna no existe: serie float64 rellena con 0.0.
    - Si existe: convierte con errors='coerce' y rellena NaN con 0.0.
    """
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")

    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


# ============================================================================
# Render
# ============================================================================


def render(df_all: pd.DataFrame) -> None:
    st.write("### Búsqueda avanzada")

    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        st.info("No hay datos para búsqueda avanzada.")
        return

    df_view = df_all.copy()

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        libraries = _safe_unique_sorted(df_view, "library")
        lib_filter: Sequence[str] = st.multiselect(
            "Biblioteca",
            libraries,
            key="lib_filter_advanced",
        )

    with col_f2:
        decisions = ["DELETE", "MAYBE", "KEEP", "UNKNOWN"]
        dec_filter: Sequence[str] = st.multiselect(
            "Decisión",
            decisions,
            default=decisions,
            format_func=lambda v: _DECISION_LABELS.get(v, v),
            key="dec_filter_advanced",
        )

    with col_f3:
        min_imdb: float = st.slider("IMDb mínimo", 0.0, 10.0, 0.0, 0.1, key="min_imdb_advanced")

    with col_f4:
        min_votes: int = st.slider("IMDb votos mínimos", 0, 200_000, 0, 1_000, key="min_votes_advanced")

    if lib_filter and "library" in df_view.columns:
        df_view = df_view[df_view["library"].isin(lib_filter)]

    if dec_filter and "decision" in df_view.columns:
        df_view = df_view[df_view["decision"].isin(dec_filter)]

    imdb_series = _ensure_numeric_column(df_view, "imdb_rating")
    votes_series = _ensure_numeric_column(df_view, "imdb_votes")

    df_view = df_view[(imdb_series >= float(min_imdb)) & (votes_series >= int(min_votes))]

    st.write(f"Resultados: {len(df_view)} película(s)")

    if df_view.empty:
        st.info("No hay resultados que coincidan con los filtros actuales.")
        return

    col_grid, col_detail = st.columns([2, 1])

    with col_grid:
        selected_row = aggrid_with_row_click(
            df_view,
            "advanced",
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
        render_detail_card(selected_row, button_key_prefix="advanced")
