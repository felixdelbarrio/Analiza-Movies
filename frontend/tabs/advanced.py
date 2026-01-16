"""
advanced.py

Pestaña “Búsqueda avanzada” (Streamlit).

Responsabilidad:
- Mostrar filtros avanzados por columnas (biblioteca, decisión, métricas, etc.).
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


def _numeric_bounds(df: pd.DataFrame, col: str) -> tuple[float, float] | None:
    if col not in df.columns:
        return None

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None

    return float(series.min()), float(series.max())


def _apply_range_filter(
    df: pd.DataFrame,
    col: str,
    selected: tuple[float, float] | tuple[int, int] | None,
    default: tuple[float, float] | tuple[int, int] | None,
) -> pd.DataFrame:
    if selected is None or default is None:
        return df
    if col not in df.columns:
        return df
    if selected == default:
        return df

    series = pd.to_numeric(df[col], errors="coerce")
    return df[series.between(selected[0], selected[1], inclusive="both")]


# ============================================================================
# Render
# ============================================================================


def render(df_all: pd.DataFrame) -> None:
    st.write("### Búsqueda avanzada")

    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        st.info("No hay datos para búsqueda avanzada.")
        return

    df_view = df_all.copy()

    title_query = ""
    year_range: tuple[int, int] | None = None
    year_default: tuple[int, int] | None = None
    imdb_range: tuple[float, float] | None = None
    imdb_default: tuple[float, float] | None = None
    votes_range: tuple[int, int] | None = None
    votes_default: tuple[int, int] | None = None
    metacritic_range: tuple[float, float] | None = None
    metacritic_default: tuple[float, float] | None = None
    rt_range: tuple[float, float] | None = None
    rt_default: tuple[float, float] | None = None
    size_range: tuple[float, float] | None = None
    size_default: tuple[float, float] | None = None

    col_f1, col_f2, col_f3 = st.columns([1, 1, 1.4])

    with col_f1:
        libraries = _safe_unique_sorted(df_all, "library")
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
            format_func=lambda v: _DECISION_LABELS.get(v, v),
            key="dec_filter_advanced",
        )

    with col_f3:
        title_query = st.text_input("Título contiene", key="title_filter_advanced")

    with st.expander("Filtros numéricos", expanded=False):
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            year_bounds = _numeric_bounds(df_all, "year")
            if year_bounds is not None:
                min_year = int(year_bounds[0])
                max_year = int(year_bounds[1])
                year_default = (min_year, max_year)
                if min_year == max_year:
                    st.caption(f"Año: {min_year}")
                    year_range = year_default
                else:
                    year_range = st.slider(
                        "Año",
                        min_year,
                        max_year,
                        year_default,
                        step=1,
                        key="year_range_advanced",
                    )

            imdb_bounds = _numeric_bounds(df_all, "imdb_rating")
            if imdb_bounds is not None:
                min_imdb, max_imdb = imdb_bounds
                imdb_default = (min_imdb, max_imdb)
                if min_imdb == max_imdb:
                    st.caption(f"IMDb: {min_imdb:.1f}")
                    imdb_range = imdb_default
                else:
                    imdb_range = st.slider(
                        "IMDb",
                        min_imdb,
                        max_imdb,
                        imdb_default,
                        step=0.1,
                        format="%.1f",
                        key="imdb_range_advanced",
                    )

            votes_bounds = _numeric_bounds(df_all, "imdb_votes")
            if votes_bounds is not None:
                min_votes = int(votes_bounds[0])
                max_votes = int(votes_bounds[1])
                votes_default = (min_votes, max_votes)
                step = max(1, int(round((max_votes - min_votes) / 100)))
                if min_votes == max_votes:
                    st.caption(f"IMDb votos: {min_votes}")
                    votes_range = votes_default
                else:
                    votes_range = st.slider(
                        "IMDb votos",
                        min_votes,
                        max_votes,
                        votes_default,
                        step=step,
                        format="%d",
                        key="votes_range_advanced",
                    )

        with col_s2:
            metacritic_bounds = _numeric_bounds(df_all, "metacritic_score")
            if metacritic_bounds is not None:
                min_meta, max_meta = metacritic_bounds
                metacritic_default = (min_meta, max_meta)
                if min_meta == max_meta:
                    st.caption(f"Metacritic: {min_meta:.0f}")
                    metacritic_range = metacritic_default
                else:
                    metacritic_range = st.slider(
                        "Metacritic",
                        min_meta,
                        max_meta,
                        metacritic_default,
                        step=1.0,
                        format="%.0f",
                        key="metacritic_range_advanced",
                    )

            rt_bounds = _numeric_bounds(df_all, "rt_score")
            if rt_bounds is not None:
                min_rt, max_rt = rt_bounds
                rt_default = (min_rt, max_rt)
                if min_rt == max_rt:
                    st.caption(f"RT: {min_rt:.0f}")
                    rt_range = rt_default
                else:
                    rt_range = st.slider(
                        "RT",
                        min_rt,
                        max_rt,
                        rt_default,
                        step=1.0,
                        format="%.0f",
                        key="rt_range_advanced",
                    )

            size_bounds = _numeric_bounds(df_all, "file_size_gb")
            if size_bounds is not None:
                min_size, max_size = size_bounds
                size_default = (min_size, max_size)
                if min_size == max_size:
                    st.caption(f"Tamaño (GB): {min_size:.2f}")
                    size_range = size_default
                else:
                    size_range = st.slider(
                        "Tamaño (GB)",
                        min_size,
                        max_size,
                        size_default,
                        step=0.1,
                        format="%.1f",
                        key="size_range_advanced",
                    )

    if lib_filter and "library" in df_view.columns:
        df_view = df_view[df_view["library"].isin(lib_filter)]

    if dec_filter and "decision" in df_view.columns:
        df_view = df_view[df_view["decision"].isin(dec_filter)]

    title_query = title_query.strip()
    if title_query and "title" in df_view.columns:
        title_series = df_view["title"].fillna("").astype(str)
        df_view = df_view[
            title_series.str.contains(
                title_query, case=False, regex=False, na=False
            )
        ]

    df_view = _apply_range_filter(df_view, "year", year_range, year_default)
    df_view = _apply_range_filter(df_view, "imdb_rating", imdb_range, imdb_default)
    df_view = _apply_range_filter(df_view, "imdb_votes", votes_range, votes_default)
    df_view = _apply_range_filter(
        df_view, "metacritic_score", metacritic_range, metacritic_default
    )
    df_view = _apply_range_filter(df_view, "rt_score", rt_range, rt_default)
    df_view = _apply_range_filter(df_view, "file_size_gb", size_range, size_default)

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
