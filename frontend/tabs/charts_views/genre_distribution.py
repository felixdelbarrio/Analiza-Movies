"""Distribución por género (OMDb) view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.tabs.charts_data import _genres_agg
from frontend.tabs.charts_shared import (
    AltChart,
    AltSelection,
    _all_movies_link,
    _caption_bullets,
    _chart,
    _decision_color,
    _format_pct,
    _weighted_revision,
)


def _genre_distribution_stats(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()

    data = agg.copy()
    data["decision"] = data["decision"].fillna("UNKNOWN")
    data["count"] = pd.to_numeric(data["count"], errors="coerce").fillna(0)

    totals = data.groupby("genre", dropna=False)["count"].sum()
    if totals.empty:
        return pd.DataFrame()

    stats = pd.DataFrame({"total_count": totals})
    stats["delete_count"] = (
        data[data["decision"] == "DELETE"].groupby("genre", dropna=False)["count"].sum()
    )
    stats["maybe_count"] = (
        data[data["decision"] == "MAYBE"].groupby("genre", dropna=False)["count"].sum()
    )
    stats["prune_count"] = (
        data[data["decision"].isin(["DELETE", "MAYBE"])]
        .groupby("genre", dropna=False)["count"]
        .sum()
    )
    stats["keep_count"] = (
        data[data["decision"] == "KEEP"].groupby("genre", dropna=False)["count"].sum()
    )
    stats["unknown_count"] = (
        data[data["decision"] == "UNKNOWN"]
        .groupby("genre", dropna=False)["count"]
        .sum()
    )
    stats = stats.fillna(0)
    stats = stats[stats["total_count"] > 0]
    if stats.empty:
        return stats

    stats["prune_score"] = _weighted_revision(
        stats["delete_count"], stats["maybe_count"]
    )
    total_score = stats["prune_score"] + stats["keep_count"] + stats["unknown_count"]
    stats["prune_share"] = stats["prune_score"] / total_score
    stats["keep_share"] = stats["keep_count"] / total_score
    stats["unknown_share"] = stats["unknown_count"] / total_score
    stats = stats.reset_index()
    return stats


def _genre_distribution_insights(stats: pd.DataFrame) -> list[str]:
    if stats.empty:
        return []

    top_prune_share = stats.sort_values("prune_share", ascending=False).iloc[0]
    top_keep_share = stats.sort_values("keep_share", ascending=False).iloc[0]
    top_total = stats.sort_values("total_count", ascending=False).iloc[0]

    total_prune = float(stats["prune_score"].sum())
    top3_prune = float(
        stats.sort_values("prune_score", ascending=False).head(3)["prune_score"].sum()
    )
    total_unknown = float(stats["unknown_count"].sum())

    lines: list[str] = []
    link_prune = _all_movies_link(
        "Ver en Todas",
        genres=[str(top_prune_share.genre)],
        decisions=["DELETE", "MAYBE"],
    )
    link_keep = _all_movies_link(
        "Ver en Todas",
        genres=[str(top_keep_share.genre)],
        decisions=["KEEP"],
    )
    lines.append(
        "Mayor % en revision: "
        f"{top_prune_share.genre} ({_format_pct(top_prune_share.prune_share)} | "
        f"{int(top_prune_share.prune_count)} titulos)"
        + (f" {link_prune}" if link_prune else "")
        + " | "
        + "Mayor % KEEP: "
        f"{top_keep_share.genre} ({_format_pct(top_keep_share.keep_share)} | "
        f"{int(top_keep_share.keep_count)} titulos)"
        + (f" {link_keep}" if link_keep else "")
    )

    line_parts: list[str] = []
    if total_prune > 0:
        line_parts.append(
            "Revision concentrada: top 3 = "
            f"{_format_pct(top3_prune / total_prune)} ({int(top3_prune)} puntos)"
        )
    else:
        line_parts.append("Sin titulos en revision")

    if total_unknown > 0:
        top_unknown = stats.sort_values("unknown_share", ascending=False).iloc[0]
        link_unknown = _all_movies_link(
            "Ver en Todas",
            genres=[str(top_unknown.genre)],
            decisions=["UNKNOWN"],
        )
        line_parts.append(
            "Mayor UNKNOWN: "
            f"{top_unknown.genre} ({_format_pct(top_unknown.unknown_share)} | "
            f"{int(top_unknown.unknown_count)} titulos)"
            + (f" {link_unknown}" if link_unknown else "")
        )
    else:
        link_total = _all_movies_link(
            "Ver en Todas",
            genres=[str(top_total.genre)],
        )
        line_parts.append(
            f"Mayor volumen total: {top_total.genre} ({int(top_total.total_count)} titulos)"
            + (f" {link_total}" if link_total else "")
        )
    lines.append(" | ".join(line_parts))

    return lines


def render(
    df_g: pd.DataFrame,
    *,
    dec_sel: AltSelection,
    top_n_genres: int,
    show_insights: bool,
) -> AltChart | None:
    agg = _genres_agg(df_g)

    if agg.empty:
        st.info("No hay datos de genero. Revisa filtros o datos de OMDb.")
        return None

    top_genres = (
        agg.groupby("genre")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_genres)
        .index
    )
    agg = agg[agg["genre"].isin(top_genres)]

    if agg.empty:
        st.info("No hay datos suficientes para los generos seleccionados.")
        return None

    stats = _genre_distribution_stats(agg)
    order = None
    if not stats.empty:
        order = stats.sort_values(
            ["prune_share", "keep_share"], ascending=[False, True]
        )["genre"].tolist()

    if st.session_state.get("charts_view") != "Dashboard":
        st.markdown("**Distribución por género (OMDb)**")

    if show_insights and not stats.empty:
        insights = _genre_distribution_insights(stats)
        _caption_bullets(insights)

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("genre:N", title="Género", sort=order),
            y=alt.Y("count:Q", title="Número de películas", stack="normalize"),
            color=_decision_color(),
            tooltip=["genre", "decision", "count"],
            opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.2)),
        )
        .add_params(dec_sel)
    )
    chart = _chart(chart)
    return chart
