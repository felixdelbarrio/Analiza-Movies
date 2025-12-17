from __future__ import annotations

from typing import Final, Iterable

import altair as alt
import pandas as pd
import streamlit as st

from frontend.data_utils import (
    explode_genres_from_omdb_json,
    build_word_counts,
    decision_color,
    safe_json_loads_single,
)

VIEW_OPTIONS: Final[list[str]] = [
    "Distribución por decisión",
    "Rating IMDb por decisión",
    "Ratings IMDb vs RT",
    "Distribución por década",
    "Distribución por biblioteca",
    "Distribución por género (OMDb)",
    "Espacio ocupado por biblioteca/decisión",
    "Boxplot IMDb por biblioteca",
    "Ranking de directores",
    "Palabras más frecuentes en títulos DELETE/MAYBE",
    "Distribución por scoring_rule",
]


def _requires_columns(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    """
    Comprueba que `df` contiene todas las columnas indicadas.

    Devuelve:
      - True  si todas las columnas están presentes.
      - False si falta alguna (y muestra un mensaje informativo en Streamlit).
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.info(f"Faltan columna(s) requerida(s): {', '.join(missing)}.")
        return False
    return True


def _chart(
    chart: alt.Chart | alt.LayerChart | alt.HConcatChart | alt.VConcatChart,
) -> None:
    """Wrapper para mostrar gráficos siempre a ancho completo."""
    # Reemplaza use_container_width=True (deprecado) por width="stretch"
    st.altair_chart(chart, width="stretch")


def render(df_all: pd.DataFrame) -> None:
    """Pestaña 5: Gráficos."""
    st.write("### Gráficos")

    if df_all.empty:
        st.info("No hay datos para mostrar gráficos.")
        return

    df_g = df_all.copy()

    view = st.selectbox("Vista", VIEW_OPTIONS)

    # 1) Distribución por decisión
    if view == "Distribución por decisión":
        if not _requires_columns(df_g, ["decision", "title"]):
            return

        agg = (
            df_g.groupby("decision", dropna=False)["title"]
            .count()
            .reset_index()
            .rename(columns={"title": "count"})
        )

        if agg.empty:
            st.info("No hay datos para la distribución por decisión.")
            return

        chart = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("decision:N", title="Decisión"),
                y=alt.Y("count:Q", title="Número de películas"),
                color=decision_color("decision"),
                tooltip=["decision", "count"],
            )
        )
        _chart(chart)

    # 2) Rating IMDb por decisión
    elif view == "Rating IMDb por decisión":
        if not _requires_columns(df_g, ["imdb_rating", "decision"]):
            return

        data = df_g.dropna(subset=["imdb_rating"])
        if data.empty:
            st.info("No hay ratings IMDb válidos para mostrar.")
            return

        chart = (
            alt.Chart(data)
            .mark_boxplot()
            .encode(
                x=alt.X("decision:N", title="Decisión"),
                y=alt.Y("imdb_rating:Q", title="IMDb rating"),
                color=decision_color("decision"),
                tooltip=["decision"],
            )
        )
        _chart(chart)

    # 3) Ratings IMDb vs RT
    elif view == "Ratings IMDb vs RT":
        if not _requires_columns(df_g, ["imdb_rating", "rt_score", "decision"]):
            return

        data = df_g.dropna(subset=["imdb_rating", "rt_score"])
        if data.empty:
            st.info("No hay suficientes datos de IMDb y RT para mostrar.")
            return

        chart = (
            alt.Chart(data)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("imdb_rating:Q", title="IMDb rating"),
                y=alt.Y("rt_score:Q", title="RT score (%)"),
                color=decision_color("decision"),
                tooltip=[
                    "title",
                    "year",
                    "library",
                    "imdb_rating",
                    "rt_score",
                    "imdb_votes",
                    "decision",
                ],
            )
        )
        _chart(chart)

    # 4) Distribución por década
    elif view == "Distribución por década":
        if not _requires_columns(df_g, ["decade_label", "decision", "title"]):
            return

        data = df_g.dropna(subset=["decade_label"])
        if data.empty:
            st.info("No hay información de década disponible.")
            return

        agg = (
            data.groupby(["decade_label", "decision"], dropna=False)["title"]
            .count()
            .reset_index()
            .rename(columns={"title": "count"})
        )

        chart = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("decade_label:N", title="Década"),
                y=alt.Y("count:Q", title="Número de películas"),
                color=decision_color("decision"),
                tooltip=["decade_label", "decision", "count"],
            )
        )
        _chart(chart)

    # 5) Distribución por biblioteca
    elif view == "Distribución por biblioteca":
        if not _requires_columns(df_g, ["library", "decision", "title"]):
            return

        agg = (
            df_g.groupby(["library", "decision"], dropna=False)["title"]
            .count()
            .reset_index()
            .rename(columns={"title": "count"})
        )

        if agg.empty:
            st.info("No hay datos de biblioteca para mostrar.")
            return

        chart = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("library:N", title="Biblioteca"),
                y=alt.Y(
                    "count:Q",
                    title="Número de películas",
                    stack="normalize",
                ),
                color=decision_color("decision"),
                tooltip=["library", "decision", "count"],
            )
        )
        _chart(chart)

    # 6) Distribución por género (OMDb)
    elif view == "Distribución por género (OMDb)":
        df_gen = explode_genres_from_omdb_json(df_g)

        if df_gen.empty:
            st.info("No hay datos de género en omdb_json.")
            return

        agg = (
            df_gen.groupby(["genre", "decision"], dropna=False)["title"]
            .count()
            .reset_index()
            .rename(columns={"title": "count"})
        )

        top_n = st.slider("Top N géneros", 5, 50, 20)
        top_genres = (
            agg.groupby("genre")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        agg = agg[agg["genre"].isin(top_genres)]

        if agg.empty:
            st.info("No hay datos suficientes para los géneros seleccionados.")
            return

        chart = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("genre:N", title="Género"),
                y=alt.Y(
                    "count:Q",
                    title="Número de películas",
                    stack="normalize",
                ),
                color=decision_color("decision"),
                tooltip=["genre", "decision", "count"],
            )
        )
        _chart(chart)

    # 7) Espacio ocupado por biblioteca/decisión
    elif view == "Espacio ocupado por biblioteca/decisión":
        if not _requires_columns(df_g, ["file_size_gb", "library", "decision"]):
            return

        agg = (
            df_g.groupby(["library", "decision"], dropna=False)["file_size_gb"]
            .sum()
            .reset_index()
        )

        if agg.empty:
            st.info("No hay datos de tamaño de archivos.")
            return

        chart_space = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("library:N", title="Biblioteca"),
                y=alt.Y(
                    "file_size_gb:Q",
                    title="Tamaño (GB)",
                    stack="normalize",
                ),
                color=decision_color("decision"),
                tooltip=[
                    "library",
                    "decision",
                    alt.Tooltip("file_size_gb:Q", format=".2f"),
                ],
            )
        )
        _chart(chart_space)

        total_space = agg["file_size_gb"].sum()
        space_delete = agg.loc[agg["decision"] == "DELETE", "file_size_gb"].sum()
        space_maybe = agg.loc[agg["decision"] == "MAYBE", "file_size_gb"].sum()

        st.markdown(
            f"- Espacio total: **{total_space:.2f} GB**\n"
            f"- DELETE: **{space_delete:.2f} GB**\n"
            f"- MAYBE: **{space_maybe:.2f} GB**"
        )

    # 8) Boxplot IMDb por biblioteca
    elif view == "Boxplot IMDb por biblioteca":
        if not _requires_columns(df_g, ["imdb_rating", "library"]):
            return

        data = df_g.dropna(subset=["imdb_rating", "library"])
        if data.empty:
            st.info("No hay datos suficientes de IMDb/library.")
            return

        chart_box = (
            alt.Chart(data)
            .mark_boxplot()
            .encode(
                x=alt.X("library:N", title="Biblioteca"),
                y=alt.Y("imdb_rating:Q", title="IMDb rating"),
                tooltip=["library"],
            )
        )
        _chart(chart_box)

    # 9) Ranking de directores
    elif view == "Ranking de directores":
        if "omdb_json" not in df_g.columns:
            st.info("No existe información OMDb JSON (omdb_json).")
            return

        df_dir = df_g.copy()

        def extract_directors(raw):
            d = safe_json_loads_single(raw)
            if not isinstance(d, dict):
                return []
            val = d.get("Director")
            if not val:
                return []
            return [x.strip() for x in str(val).split(",") if x.strip()]

        df_dir["director_list"] = df_dir["omdb_json"].apply(extract_directors)
        df_dir = df_dir.explode("director_list")
        df_dir = df_dir[
            df_dir["director_list"].notna() & (df_dir["director_list"] != "")
        ]

        if df_dir.empty:
            st.info("No se encontraron directores en omdb_json.")
            return

        min_movies = st.slider("Mínimo nº de películas por director", 1, 10, 3)

        agg = (
            df_dir.groupby("director_list", dropna=False)
            .agg(
                imdb_mean=("imdb_rating", "mean"),
                count=("title", "count"),
            )
            .reset_index()
        )
        agg = agg[agg["count"] >= min_movies].sort_values(
            "imdb_mean", ascending=False
        )

        if agg.empty:
            st.info("No hay directores que cumplan el mínimo de películas.")
            return

        top_n = st.slider("Top N directores por IMDb medio", 5, 50, 20)
        agg_top = agg.head(top_n)

        chart = (
            alt.Chart(agg_top)
            .mark_bar()
            .encode(
                x=alt.X("director_list:N", title="Director"),
                y=alt.Y("imdb_mean:Q", title="IMDb medio"),
                tooltip=["director_list", "imdb_mean", "count"],
            )
        )
        _chart(chart)

    # 10) Palabras más frecuentes en títulos DELETE/MAYBE
    elif view == "Palabras más frecuentes en títulos DELETE/MAYBE":
        df_words = build_word_counts(df_g, ["DELETE", "MAYBE"])

        if df_words.empty:
            st.info("No hay datos suficientes para el análisis de palabras.")
            return

        top_n = st.slider("Top N palabras", 5, 50, 20)
        df_top = df_words.head(top_n)

        chart = (
            alt.Chart(df_top)
            .mark_bar()
            .encode(
                x=alt.X("word:N", title="Palabra"),
                y=alt.Y("count:Q", title="Frecuencia"),
                color=decision_color("decision"),
                tooltip=["word", "decision", "count"],
            )
        )
        _chart(chart)

    # 11) Distribución por scoring_rule
    elif view == "Distribución por scoring_rule":
        if not _requires_columns(df_g, ["scoring_rule", "decision", "title"]):
            return

        agg = (
            df_g.groupby(["scoring_rule", "decision"], dropna=False)["title"]
            .count()
            .reset_index()
            .rename(columns={"title": "count"})
        )

        if agg.empty:
            st.info("No hay datos de scoring_rule.")
            return

        chart = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("scoring_rule:N", title="Regla de scoring"),
                y=alt.Y("count:Q", title="Número de películas"),
                color=decision_color("decision"),
                tooltip=["scoring_rule", "decision", "count"],
            )
        )
        _chart(chart)