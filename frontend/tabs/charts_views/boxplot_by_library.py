"""Boxplot IMDb por biblioteca view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.tabs.charts_shared import (
    AltChart,
    AltSelection,
    _all_movies_link,
    _caption_bullets,
    _chart,
    _chart_accents,
    _boxplot_scale,
    _movie_tooltips,
    _requires_columns,
    _theme_tokens,
    _token,
)


def render(
    df_g: pd.DataFrame,
    *,
    lib_sel: AltSelection,
    imdb_ref: float,
    top_n_libs: int | None,
    min_n_libs: int,
    boxplot_horizontal: bool,
    compare_libs: list[str],
    show_insights: bool,
) -> AltChart | None:
    if not _requires_columns(df_g, ["imdb_rating", "library"]):
        return None

    data = df_g.dropna(subset=["imdb_rating", "library"])
    if data.empty:
        st.info("No hay datos suficientes de IMDb/biblioteca. Revisa filtros.")
        return None

    stats = (
        data.groupby("library", dropna=False)["imdb_rating"]
        .agg(imdb_median="median", count="count")
        .reset_index()
    )
    if stats.empty:
        st.info("No hay estadisticas para bibliotecas. Revisa filtros.")
        return None

    tokens = _theme_tokens()
    accents = _chart_accents()
    ref_color = _token(tokens, "text_3", "#666666")
    mean_color = accents["accent_soft"]
    median_color = _token(tokens, "text_1", "#ffffff")
    accent_color = accents["accent"]
    boxplot_scale = _boxplot_scale(tokens)

    global_mean = float(data["imdb_rating"].mean())
    num_libs = int(stats["library"].nunique())
    max_libs = max(1, min(50, num_libs))
    top_n = top_n_libs if top_n_libs is not None else min(20, max_libs)
    min_n = min_n_libs
    horizontal = boxplot_horizontal

    if top_n < num_libs:
        top_libs = (
            stats.sort_values("count", ascending=False).head(top_n)["library"].tolist()
        )
        data = data[data["library"].isin(top_libs)]
        stats = stats[stats["library"].isin(top_libs)]
    stats = stats[stats["count"] >= min_n]
    data = data[data["library"].isin(stats["library"])]
    if data.empty:
        st.info("No hay datos tras aplicar filtros. Revisa filtros.")
        return None

    data = data.merge(stats, on="library", how="left")
    order = stats.sort_values("imdb_median", ascending=False)["library"].tolist()
    top3 = stats.head(3)
    bottom3 = stats.tail(3)
    boxplot_insights = [
        "Top medianas: "
        + ", ".join(
            [f"{row.library} ({row.imdb_median:.1f})" for row in top3.itertuples()]
        ),
        "Bottom medianas: "
        + ", ".join(
            [f"{row.library} ({row.imdb_median:.1f})" for row in bottom3.itertuples()]
        ),
    ]
    if len(compare_libs) == 2:
        comp = stats[stats["library"].isin(compare_libs)].copy()
        if len(comp) == 2:
            delta = comp.iloc[0]["imdb_median"] - comp.iloc[1]["imdb_median"]
            boxplot_insights.append(
                f"Comparacion: {compare_libs[0]} vs {compare_libs[1]} | "
                f"Delta mediana IMDb: {delta:.2f}"
            )
            comp_chart = (
                alt.Chart(comp)
                .mark_bar()
                .encode(
                    x=alt.X("library:N", title="Biblioteca"),
                    y=alt.Y("imdb_median:Q", title="Mediana IMDb"),
                    tooltip=[
                        alt.Tooltip("library:N", title="Biblioteca"),
                        alt.Tooltip(
                            "imdb_median:Q", title="Mediana IMDb", format=".1f"
                        ),
                        alt.Tooltip("count:Q", title="Peliculas", format=".0f"),
                    ],
                )
            )
            comp_chart = _chart(comp_chart)
    if not top3.empty:
        top_lib = str(top3.iloc[0]["library"])
        link_top = _all_movies_link("Ver en Todas", libraries=[top_lib])
        if link_top:
            boxplot_insights.append(f"Biblioteca top: {top_lib}. {link_top}")
    if not bottom3.empty:
        bottom_lib = str(bottom3.iloc[0]["library"])
        link_bottom = _all_movies_link("Ver en Todas", libraries=[bottom_lib])
        if link_bottom:
            boxplot_insights.append(f"Biblioteca bottom: {bottom_lib}. {link_bottom}")
    if show_insights:
        _caption_bullets(boxplot_insights)

    tooltip_common = [
        alt.Tooltip("library:N", title="Biblioteca"),
        alt.Tooltip("imdb_median:Q", title="Mediana IMDb", format=".1f"),
        alt.Tooltip("count:Q", title="Peliculas", format=".0f"),
    ]
    ref_line = (
        alt.Chart(pd.DataFrame({"imdb_rating": [imdb_ref]}))
        .mark_rule(color=ref_color, strokeDash=[4, 4])
        .encode(y=alt.Y("imdb_rating:Q"))
    )
    mean_line = (
        alt.Chart(pd.DataFrame({"imdb_rating": [global_mean]}))
        .mark_rule(color=mean_color, strokeDash=[2, 2])
        .encode(y=alt.Y("imdb_rating:Q"))
    )
    chart_box = (
        alt.Chart(data)
        .mark_boxplot(size=40)
        .encode(
            x=(
                alt.X(
                    "library:N",
                    title="Biblioteca",
                    sort=order,
                    axis=alt.Axis(labelAngle=90, labelLimit=140),
                )
                if not horizontal
                else alt.X("imdb_rating:Q", title="IMDb rating (0-10)")
            ),
            y=(
                alt.Y("imdb_rating:Q", title="IMDb rating (0-10)")
                if not horizontal
                else alt.Y(
                    "library:N",
                    title="Biblioteca",
                    sort=order,
                )
            ),
            color=alt.Color("imdb_median:Q", scale=boxplot_scale, legend=None),
            tooltip=tooltip_common,
            opacity=alt.condition(lib_sel, alt.value(1), alt.value(0.2)),
        )
    )
    chart_median = (
        alt.Chart(stats)
        .mark_tick(color=median_color, thickness=2)
        .encode(
            x=(
                alt.X("library:N", sort=order)
                if not horizontal
                else alt.X("imdb_median:Q")
            ),
            y=(
                alt.Y("imdb_median:Q")
                if not horizontal
                else alt.Y("library:N", sort=order)
            ),
            tooltip=tooltip_common,
        )
    )
    chart_strip = (
        alt.Chart(data)
        .transform_calculate(imdb_jitter="datum.imdb_rating + (random() - 0.5) * 0.2")
        .mark_circle(size=18, opacity=0.25)
        .encode(
            x=(
                alt.X(
                    "library:N",
                    sort=order,
                    axis=alt.Axis(labelAngle=90, labelLimit=140),
                )
                if not horizontal
                else alt.X("imdb_jitter:Q")
            ),
            y=(
                alt.Y("imdb_jitter:Q")
                if not horizontal
                else alt.Y("library:N", sort=order)
            ),
            color=alt.value(accent_color),
            tooltip=_movie_tooltips(data),
            opacity=alt.condition(lib_sel, alt.value(1), alt.value(0.1)),
        )
        .add_params(lib_sel)
    )
    chart = chart_strip + chart_box + chart_median + ref_line + mean_line
    chart = _chart(chart)
    return chart
