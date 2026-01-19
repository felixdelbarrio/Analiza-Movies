"""Ratings IMDb vs RT view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.tabs.charts_data import _mark_imdb_outliers
from frontend.tabs.charts_shared import (
    AltChart,
    AltSelection,
    _caption_bullets,
    _chart,
    _corr_strength,
    _decision_color,
    _format_pct,
    _movie_tooltips,
    _requires_columns,
    _theme_tokens,
    _token,
)


def _imdb_rt_insights(
    data: pd.DataFrame,
    imdb_ref: float,
    rt_ref: float,
) -> list[str]:
    lines: list[str] = []
    if data.empty:
        return lines

    imdb_raw = pd.to_numeric(data["imdb_rating"], errors="coerce")
    rt_raw = pd.to_numeric(data["rt_score"], errors="coerce")
    mask = imdb_raw.notna() & rt_raw.notna()
    total = int(mask.sum())
    if total <= 0:
        return lines

    imdb_scaled = imdb_raw[mask] * 10
    rt = rt_raw[mask]
    gap = imdb_scaled - rt

    within_10 = int((gap.abs() <= 10).sum())
    big_gap = int((gap.abs() >= 20).sum())

    line_parts: list[str] = []
    if total > 1:
        corr = imdb_scaled.corr(rt)
        if pd.notna(corr):
            strength = _corr_strength(corr)
            line_parts.append(f"Relacion IMDb vs RT: r={corr:.2f} ({strength})")
    line_parts.append(
        f"Alineacion: {_format_pct(within_10 / total)} ({within_10}) dentro de +/-10 pts"
    )
    if line_parts:
        lines.append(" | ".join(line_parts))

    median_gap = float(gap.median())
    if median_gap >= 0:
        gap_label = f"Brecha mediana: IMDb esta +{median_gap:.1f} pts sobre RT"
    else:
        gap_label = f"Brecha mediana: RT esta +{abs(median_gap):.1f} pts sobre IMDb"
    lines.append(
        f"{gap_label} | Discrepancias fuertes: "
        f"{_format_pct(big_gap / total)} ({big_gap}) >= 20 pts"
    )

    consensus_high = int(((imdb_raw >= imdb_ref) & (rt_raw >= rt_ref) & mask).sum())
    consensus_low = int(((imdb_raw < imdb_ref) & (rt_raw < rt_ref) & mask).sum())
    lines.append(
        "Consenso en umbrales: "
        f"{_format_pct(consensus_high / total)} ({consensus_high}) por encima de ambos "
        f"(IMDb >= {imdb_ref:.1f}, RT >= {rt_ref:.0f}) | "
        f"{_format_pct(consensus_low / total)} ({consensus_low}) por debajo de ambos"
    )

    return lines


def render(
    df_g: pd.DataFrame,
    *,
    dec_sel: AltSelection,
    imdb_ref: float,
    rt_ref: float,
    outlier_low: float,
    outlier_high: float,
    show_insights: bool,
) -> AltChart | None:
    if not _requires_columns(df_g, ["imdb_rating", "rt_score", "decision"]):
        return None

    data = df_g.dropna(subset=["imdb_rating", "rt_score"])
    if data.empty:
        st.info("No hay suficientes datos de IMDb y RT. Revisa filtros.")
        return None

    if show_insights:
        insights = _imdb_rt_insights(data, imdb_ref, rt_ref)
        _caption_bullets(insights)

    data = _mark_imdb_outliers(data, low=outlier_low, high=outlier_high)
    tokens = _theme_tokens()
    ref_color = _token(tokens, "text_3", "#666666")

    base = (
        alt.Chart(data)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("imdb_rating:Q", title="IMDb rating (0-10)"),
            y=alt.Y("rt_score:Q", title="RT score (%)"),
            color=_decision_color(),
            tooltip=_movie_tooltips(data),
            opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.2)),
        )
        .add_params(dec_sel)
    )
    outliers = (
        alt.Chart(data[data["imdb_outlier"].notna()])
        .mark_point(size=120, filled=True)
        .encode(
            x=alt.X("imdb_rating:Q"),
            y=alt.Y("rt_score:Q"),
            color=_decision_color(),
            shape=alt.Shape(
                "imdb_outlier:N",
                scale=alt.Scale(
                    domain=["Alta", "Baja"],
                    range=["triangle-up", "triangle-down"],
                ),
                legend=alt.Legend(orient="right", title="Outlier IMDb"),
            ),
            tooltip=_movie_tooltips(data),
            opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.2)),
        )
        .add_params(dec_sel)
    )
    ref_imdb = (
        alt.Chart(pd.DataFrame({"imdb_rating": [imdb_ref]}))
        .mark_rule(color=ref_color, strokeDash=[4, 4])
        .encode(x=alt.X("imdb_rating:Q"))
    )
    ref_rt = (
        alt.Chart(pd.DataFrame({"rt_score": [rt_ref]}))
        .mark_rule(color=ref_color, strokeDash=[4, 4])
        .encode(y=alt.Y("rt_score:Q"))
    )
    chart = base + outliers + ref_imdb + ref_rt
    chart = _chart(chart)
    return chart
