"""Rating IMDb por decision view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.tabs.charts_shared import (
    AltChart,
    AltSelection,
    DECISION_ORDER,
    _caption_bullets,
    _chart,
    _decision_color,
    _format_num,
    _format_pct,
    _requires_columns,
    _theme_tokens,
    _token,
)


def _imdb_decision_insights(data: pd.DataFrame, imdb_ref: float) -> list[str]:
    lines: list[str] = []
    if data.empty:
        return lines

    data = data.copy()
    data["imdb_rating"] = pd.to_numeric(data["imdb_rating"], errors="coerce")
    data = data.dropna(subset=["imdb_rating", "decision"])
    if data.empty:
        return lines

    stats = (
        data.groupby("decision", dropna=False)["imdb_rating"]
        .agg(count="count", mean="mean", median="median")
        .reset_index()
    )
    above_ref = (
        data.assign(above_ref=data["imdb_rating"] >= imdb_ref)
        .groupby("decision", dropna=False)["above_ref"]
        .sum()
        .rename("above_ref")
        .reset_index()
    )
    stats = stats.merge(above_ref, on="decision", how="left")
    stats["above_ref"] = stats["above_ref"].fillna(0)
    stats["above_share"] = stats["above_ref"] / stats["count"]

    stats = stats.sort_values("median", ascending=False)
    best = stats.iloc[0]
    worst = stats.iloc[-1]
    best_decision = str(best["decision"])
    worst_decision = str(worst["decision"])
    best_median = best["median"]
    worst_median = worst["median"]
    best_count = int(best["count"])
    worst_count = int(worst["count"])

    if len(stats) > 1:
        lines.append(
            "Mejor mediana: "
            f"{best_decision} ({_format_num(best_median)}, {best_count} titulos)"
            " | "
            "Peor mediana: "
            f"{worst_decision} ({_format_num(worst_median)}, {worst_count} titulos)"
        )
        if pd.notna(best_median) and pd.notna(worst_median):
            delta = float(best_median - worst_median)
            lines.append(f"Brecha de mediana: {delta:.1f} puntos IMDb")
    else:
        lines.append(
            f"Mediana IMDb: {best_decision} ({_format_num(best_median)}, {best_count} titulos)"
        )

    top_above = stats.sort_values("above_share", ascending=False).iloc[0]
    line_parts = [
        f"Sobre umbral IMDb >= {imdb_ref:.1f}: "
        f"{top_above['decision']} {_format_pct(top_above['above_share'])} "
        f"({int(top_above['above_ref'])})"
    ]
    if len(stats) > 1:
        bottom_above = stats.sort_values("above_share", ascending=True).iloc[0]
        if bottom_above["decision"] != top_above["decision"]:
            line_parts.append(
                f"Menor: {bottom_above['decision']} "
                f"{_format_pct(bottom_above['above_share'])} "
                f"({int(bottom_above['above_ref'])})"
            )
    lines.append(" | ".join(line_parts))

    return lines


def render(
    df_g: pd.DataFrame,
    *,
    dec_sel: AltSelection,
    imdb_ref: float,
    show_insights: bool,
) -> AltChart | None:
    if not _requires_columns(df_g, ["imdb_rating", "decision"]):
        return None

    data = df_g.dropna(subset=["imdb_rating"])
    if data.empty:
        st.info("No hay ratings IMDb validos para mostrar. Revisa filtros.")
        return None

    if show_insights:
        insights = _imdb_decision_insights(data, imdb_ref)
        _caption_bullets(insights)

    tokens = _theme_tokens()
    ref_color = _token(tokens, "text_3", "#666666")
    bin_spec = alt.Bin(step=0.5, extent=[0, 10])
    base = (
        alt.Chart(data)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("imdb_rating:Q", bin=bin_spec, title="IMDb rating (0-10)"),
            y=alt.Y("count():Q", title="Numero de peliculas"),
            color=_decision_color(),
            tooltip=[
                alt.Tooltip("decision:N", title="Decision"),
                alt.Tooltip("imdb_rating:Q", bin=bin_spec, title="IMDb (bin)"),
                alt.Tooltip("count():Q", title="Peliculas", format=".0f"),
            ],
            opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.2)),
        )
        .add_params(dec_sel)
    )
    ref_line = (
        alt.Chart(data)
        .mark_rule(color=ref_color, strokeDash=[4, 4])
        .encode(x=alt.datum(imdb_ref))
    )
    chart = (base + ref_line).facet(
        column=alt.Column("decision:N", title="Decision", sort=DECISION_ORDER)
    )
    chart = chart.resolve_scale(y="independent")
    chart = _chart(chart)
    return chart
