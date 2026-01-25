"""Valor por GB (rating por tamano) view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.tabs.charts_shared import (
    AltChart,
    AltSelection,
    FONT_BODY,
    _all_movies_link,
    _caption_bullets,
    _chart,
    _decision_color,
    _format_pct,
    _requires_columns,
)


def _value_insights(
    data: pd.DataFrame, *, threshold: float, top_row: pd.Series
) -> list[str]:
    lines: list[str] = []
    lines.append("Valor = IMDb / GB. Mas alto = mejor.")
    title = str(top_row.get("title", ""))
    value = float(top_row.get("value_per_gb", 0.0))
    size_gb = float(top_row.get("file_size_gb", 0.0))
    imdb = float(top_row.get("imdb_rating", 0.0))
    link = _all_movies_link("Ver en Todas", title=title)
    lines.append(
        f"Peor valor: {title} ({value:.2f} IMDb/GB, {size_gb:.1f} GB, IMDb {imdb:.1f})."
        + (f" {link}" if link else "")
    )
    low_count = int((data["value_per_gb"] <= threshold).sum())
    total = int(len(data))
    if total > 0:
        lines.append(
            f"Bajo umbral (p25): {low_count} titulos ({_format_pct(low_count / total)})."
        )
    return lines


def _build_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Titulo"] = df["title"].astype(str)
    if "library" in df.columns:
        out["Biblioteca"] = df["library"].astype(str)
    if "year" in df.columns:
        out["Ano"] = pd.to_numeric(df["year"], errors="coerce").round(0)
    if "decision" in df.columns:
        out["Decision"] = df["decision"].astype(str)
    out["GB"] = pd.to_numeric(df["file_size_gb"], errors="coerce").round(2)
    out["IMDb"] = pd.to_numeric(df["imdb_rating"], errors="coerce").round(1)
    out["Valor/GB"] = pd.to_numeric(df["value_per_gb"], errors="coerce").round(2)
    out = out.dropna(subset=["GB", "IMDb", "Valor/GB"])
    return out


def render(
    df_g: pd.DataFrame,
    *,
    dec_sel: AltSelection,
    show_insights: bool,
    top_n: int = 15,
) -> AltChart | None:
    if not _requires_columns(
        df_g, ["file_size_gb", "imdb_rating", "title", "decision"]
    ):
        return None

    data = df_g.copy()
    data["file_size_gb"] = pd.to_numeric(data["file_size_gb"], errors="coerce")
    data["imdb_rating"] = pd.to_numeric(data["imdb_rating"], errors="coerce")
    data = data.dropna(subset=["file_size_gb", "imdb_rating"])
    data = data[data["file_size_gb"] > 0]

    if data.empty:
        st.info("No hay datos con tamano y rating para valor por GB.")
        return None

    data["value_per_gb"] = data["imdb_rating"] / data["file_size_gb"]
    data = data.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["value_per_gb"]
    )
    if data.empty:
        st.info("No hay valores validos para calcular valor por GB.")
        return None

    top = data.sort_values("value_per_gb", ascending=True).head(top_n).copy()
    threshold = float(data["value_per_gb"].quantile(0.25))

    if show_insights:
        insights = _value_insights(data, threshold=threshold, top_row=top.iloc[0])
        _caption_bullets(insights)

    bar = (
        alt.Chart(top)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("value_per_gb:Q", title="Valor por GB (IMDb/GB)"),
            y=alt.Y(
                "title:N",
                sort=alt.SortField("value_per_gb", order="ascending"),
                title=None,
                axis=alt.Axis(labelLimit=240),
            ),
            color=_decision_color(),
            tooltip=[
                alt.Tooltip("title:N", title="Titulo"),
                alt.Tooltip("value_per_gb:Q", title="Valor/GB", format=".2f"),
                alt.Tooltip("file_size_gb:Q", title="Tamano (GB)", format=".2f"),
                alt.Tooltip("imdb_rating:Q", title="IMDb", format=".1f"),
            ],
            opacity=alt.condition(dec_sel, alt.value(0.95), alt.value(0.25)),
        )
        .add_params(dec_sel)
    )
    value_labels = (
        alt.Chart(top)
        .mark_text(align="left", dx=6, font=FONT_BODY, fontSize=11)
        .encode(
            x=alt.X("value_per_gb:Q"),
            y=alt.Y("title:N", sort=alt.SortField("value_per_gb", order="ascending")),
            text=alt.Text("value_per_gb:Q", format=".2f"),
        )
    )
    chart = (bar + value_labels).properties(height=420)

    tab_chart, tab_list = st.tabs(["Ranking", "Listado de revision"])
    chart_rendered: AltChart | None = None
    with tab_chart:
        chart_rendered = _chart(chart)
    with tab_list:
        worst = data.sort_values("value_per_gb", ascending=True).head(max(top_n, 20))
        table = _build_table(worst)
        if table.empty:
            st.info("No hay filas para mostrar en el listado de revision.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)

    return chart_rendered
