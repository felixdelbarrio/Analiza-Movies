"""
all_movies.py

Pestaña “Todas” (Streamlit).

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
from pandas.api.types import is_numeric_dtype

from frontend.components import (
    aggrid_with_row_click,
    render_decision_chip_styles,
    render_detail_card,
)
from frontend.config_front_charts import get_show_numeric_filters
from frontend.data_utils import dataframe_signature

_DECISION_LABELS: dict[str, str] = {
    "DELETE": "DELETE",
    "MAYBE": "MAYBE",
    "KEEP": "KEEP",
    "UNKNOWN": "UNKNOWN",
}


def _decision_label(value: object) -> str:
    label = _DECISION_LABELS.get(value) if isinstance(value, str) else None
    if isinstance(label, str):
        return label
    return str(value)


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

    series = df[col].dropna().astype(str).map(str.strip)
    series = series.mask(series == "", pd.NA).dropna()
    raw: list[Any] = series.unique().tolist()

    out: list[str] = []
    for v in raw:
        s = str(v).strip()
        if s:
            out.append(s)

    out.sort()
    return out


_FILTER_BOUNDS_COLS: tuple[str, ...] = (
    "year",
    "imdb_rating",
    "imdb_votes",
    "metacritic_score",
    "rt_score",
    "file_size_gb",
)


def _build_filter_profile(df: pd.DataFrame) -> dict[str, object]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in _FILTER_BOUNDS_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        if not is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        series = series.dropna()
        if series.empty:
            continue
        bounds[col] = (float(series.min()), float(series.max()))
    return {"libraries": _safe_unique_sorted(df, "library"), "bounds": bounds}


def _get_filter_profile(df: pd.DataFrame) -> dict[str, object]:
    signature = dataframe_signature(df)
    cache = st.session_state.get("all_movies_filter_profile")
    if isinstance(cache, dict) and cache.get("sig") == signature:
        cached = cache.get("value")
        if isinstance(cached, dict):
            return cached
    profile = _build_filter_profile(df)
    st.session_state["all_movies_filter_profile"] = {"sig": signature, "value": profile}
    return profile


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
    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        st.info("No hay datos para mostrar.")
        return

    df_view = df_all

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

    profile = _get_filter_profile(df_all)
    libraries = profile.get("libraries")
    bounds = profile.get("bounds")
    if not isinstance(libraries, list):
        libraries = _safe_unique_sorted(df_all, "library")
    if not isinstance(bounds, dict):
        bounds = {}

    with col_f1:
        lib_filter: Sequence[str] = st.multiselect(
            "Biblioteca",
            libraries,
            key="lib_filter_all_movies",
        )

    with col_f2:
        decisions = ["DELETE", "MAYBE", "KEEP", "UNKNOWN"]
        dec_filter: Sequence[str] = st.multiselect(
            "Decisión",
            decisions,
            format_func=_decision_label,
            key="dec_filter_all_movies",
        )
        colorize = bool(st.session_state.get("grid_colorize_rows", True))
        render_decision_chip_styles(
            "decision-chips-all-movies",
            enabled=colorize,
            selected_values=list(dec_filter),
        )

    with col_f3:
        title_query = st.text_input("Título contiene", key="title_filter_all_movies")

    if get_show_numeric_filters():
        with st.expander("Filtros numéricos", expanded=False):
            col_s1, col_s2 = st.columns(2)

            with col_s1:
                year_bounds = bounds.get("year")
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
                            key="year_range_all_movies",
                        )

                imdb_bounds = bounds.get("imdb_rating")
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
                            key="imdb_range_all_movies",
                        )

                votes_bounds = bounds.get("imdb_votes")
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
                            key="votes_range_all_movies",
                        )

            with col_s2:
                metacritic_bounds = bounds.get("metacritic_score")
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
                            key="metacritic_range_all_movies",
                        )

                rt_bounds = bounds.get("rt_score")
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
                            key="rt_range_all_movies",
                        )

                size_bounds = bounds.get("file_size_gb")
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
                            key="size_range_all_movies",
                        )

    if lib_filter and "library" in df_view.columns:
        df_view = df_view[df_view["library"].isin(lib_filter)]

    if dec_filter and "decision" in df_view.columns:
        df_view = df_view[df_view["decision"].isin(dec_filter)]

    title_query = title_query.strip()
    if title_query and "title" in df_view.columns:
        title_series = df_view["title"].fillna("").astype(str)
        df_view = df_view[
            title_series.str.contains(title_query, case=False, regex=False, na=False)
        ]

    df_view = _apply_range_filter(df_view, "year", year_range, year_default)
    df_view = _apply_range_filter(df_view, "imdb_rating", imdb_range, imdb_default)
    df_view = _apply_range_filter(df_view, "imdb_votes", votes_range, votes_default)
    df_view = _apply_range_filter(
        df_view, "metacritic_score", metacritic_range, metacritic_default
    )
    df_view = _apply_range_filter(df_view, "rt_score", rt_range, rt_default)
    df_view = _apply_range_filter(df_view, "file_size_gb", size_range, size_default)

    if "file_size_gb" in df_view.columns:
        size_series = pd.to_numeric(df_view["file_size_gb"], errors="coerce")
        df_view = (
            df_view.assign(_sort_size=size_series)
            .sort_values("_sort_size", ascending=False, na_position="last")
            .drop(columns=["_sort_size"])
        )
    initial_sort_model: list[dict[str, Any]] | None = None
    if "file_size_gb" in df_view.columns:
        initial_sort_model = [{"colId": "file_size_gb", "sort": "desc"}]

    if df_view.empty:
        st.info("No hay resultados que coincidan con los filtros actuales.")
        return

    def _results_caption(count: int, _total: int, _has_search: bool) -> str:
        return f"Resultados: {count} película(s)"

    col_grid, col_detail = st.columns([2, 1])

    with col_grid:
        selected_row = aggrid_with_row_click(
            df_view,
            "all_movies",
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
            initial_sort_model=initial_sort_model,
            auto_select_first=True,
            toolbar_caption_builder=_results_caption,
        )

    with col_detail:
        render_detail_card(selected_row)
