"""Mapa de desperdicio (tamano vs rating) view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.tabs.charts_shared import (
    AltChart,
    AltSelection,
    FONT_BODY,
    _caption_bullets,
    _chart,
    _decision_color,
    _format_pct,
    _movie_tooltips,
    _requires_columns,
    _theme_tokens,
    _token,
)


def _waste_insights(
    data: pd.DataFrame, *, imdb_ref: float, size_threshold: float
) -> list[str]:
    lines: list[str] = []
    if data.empty:
        return lines

    lines.append(
        f"Zona roja = tamano >= {size_threshold:.1f} GB y IMDb < {imdb_ref:.1f}."
    )
    red_mask = (data["file_size_gb"] >= size_threshold) & (
        data["imdb_rating"] < imdb_ref
    )
    red_count = int(red_mask.sum())
    total = int(len(data))
    if total > 0:
        red_share = red_count / total
        red_size = float(data.loc[red_mask, "file_size_gb"].sum())
        lines.append(
            f"En zona roja: {red_count} titulos ({_format_pct(red_share)}) | "
            f"{red_size:.1f} GB."
        )

    if "waste_score" in data.columns and not data.empty:
        top = data.sort_values("waste_score", ascending=False).iloc[0]
        title = str(top.get("title", ""))
        size_gb = float(top.get("file_size_gb", 0.0))
        imdb = float(top.get("imdb_rating", 0.0))
        lines.append(
            f"Mayor desperdicio: {title} ({size_gb:.1f} GB, IMDb {imdb:.1f})."
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
    out["Desperdicio"] = pd.to_numeric(df["waste_score"], errors="coerce").round(2)
    out = out.dropna(subset=["GB", "IMDb"])
    return out


def render(
    df_g: pd.DataFrame,
    *,
    dec_sel: AltSelection,
    imdb_ref: float,
    show_insights: bool,
    top_n: int = 20,
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
        st.info("No hay datos con tamano y rating para el mapa de desperdicio.")
        return None

    size_values = data["file_size_gb"].dropna()
    if size_values.empty:
        st.info("No hay tamanos validos para el mapa de desperdicio.")
        return None

    size_threshold = (
        float(size_values.quantile(0.75))
        if len(size_values) >= 4
        else float(size_values.median())
    )
    size_threshold = max(size_threshold, float(size_values.min()))
    size_max = float(size_values.max())

    data["waste_score"] = (
        data["file_size_gb"] * (imdb_ref - data["imdb_rating"]).clip(lower=0)
    )

    if show_insights:
        insights = _waste_insights(
            data, imdb_ref=imdb_ref, size_threshold=size_threshold
        )
        _caption_bullets(insights)

    tokens = _theme_tokens()
    warn_color = _token(tokens, "decision_delete", "#e55b5b")
    ref_color = _token(tokens, "text_3", "#5b6270")
    zone = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x0": [size_threshold],
                    "x1": [size_max * 1.02],
                    "y0": [0.0],
                    "y1": [imdb_ref],
                }
            )
        )
        .mark_rect(opacity=0.08, color=warn_color, stroke=warn_color)
        .encode(x="x0:Q", x2="x1:Q", y="y0:Q", y2="y1:Q")
    )
    ref_line = (
        alt.Chart(pd.DataFrame({"imdb": [imdb_ref]}))
        .mark_rule(color=ref_color, strokeDash=[4, 4])
        .encode(y="imdb:Q")
    )
    size_line = (
        alt.Chart(pd.DataFrame({"size": [size_threshold]}))
        .mark_rule(color=ref_color, strokeDash=[4, 4])
        .encode(x="size:Q")
    )

    tooltip = _movie_tooltips(data)
    tooltip.append(alt.Tooltip("waste_score:Q", title="Desperdicio", format=".2f"))

    scatter = (
        alt.Chart(data)
        .mark_circle(opacity=0.85, stroke="#ffffff", strokeWidth=0.6)
        .encode(
            x=alt.X("file_size_gb:Q", title="Tamano (GB)"),
            y=alt.Y(
                "imdb_rating:Q",
                title="IMDb rating (0-10)",
                scale=alt.Scale(domain=[0, 10]),
            ),
            size=alt.Size(
                "file_size_gb:Q",
                scale=alt.Scale(range=[50, 900]),
                legend=None,
            ),
            color=_decision_color(),
            tooltip=tooltip,
            opacity=alt.condition(dec_sel, alt.value(0.9), alt.value(0.2)),
        )
        .add_params(dec_sel)
    )

    chart = zone + ref_line + size_line + scatter
    chart = chart.properties(height=420)

    tab_chart, tab_list = st.tabs(["Mapa", "Listado de revision"])
    chart_rendered: AltChart | None = None
    with tab_chart:
        chart_rendered = _chart(chart)
    with tab_list:
        red_mask = (data["file_size_gb"] >= size_threshold) & (
            data["imdb_rating"] < imdb_ref
        )
        focus = data[red_mask].copy()
        if focus.empty:
            focus = data.sort_values("waste_score", ascending=False).copy()
            st.caption("Sin zona roja clara; se muestran los mayores desperdicios.")
        focus = focus.sort_values("waste_score", ascending=False).head(top_n)
        table = _build_table(focus)
        if table.empty:
            st.info("No hay filas para mostrar en el listado de revision.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)

    return chart_rendered
