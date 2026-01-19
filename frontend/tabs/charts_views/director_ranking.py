"""Ranking de directores view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.tabs.charts_data import _director_decision_stats
from frontend.tabs.charts_shared import (
    AltChart,
    AltSelection,
    _caption_bullets,
    _chart,
    _chart_accents,
    _decision_color,
    _format_pct,
    _weighted_revision,
)


def _director_ranking_stats(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()

    data = agg.copy()
    data["decision"] = data["decision"].fillna("UNKNOWN")
    data["count"] = pd.to_numeric(data["count"], errors="coerce").fillna(0)

    totals = data.groupby("director_list", dropna=False)["count"].sum()
    if totals.empty:
        return pd.DataFrame()

    stats = pd.DataFrame({"total_count": totals})
    stats["delete_count"] = (
        data[data["decision"] == "DELETE"]
        .groupby("director_list", dropna=False)["count"]
        .sum()
    )
    stats["maybe_count"] = (
        data[data["decision"] == "MAYBE"]
        .groupby("director_list", dropna=False)["count"]
        .sum()
    )
    stats["prune_count"] = (
        data[data["decision"].isin(["DELETE", "MAYBE"])]
        .groupby("director_list", dropna=False)["count"]
        .sum()
    )
    stats["keep_count"] = (
        data[data["decision"] == "KEEP"]
        .groupby("director_list", dropna=False)["count"]
        .sum()
    )
    stats["unknown_count"] = (
        data[data["decision"] == "UNKNOWN"]
        .groupby("director_list", dropna=False)["count"]
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
    stats["balance"] = stats["prune_share"] - stats["keep_share"]

    imdb_mean = data.drop_duplicates("director_list")[
        ["director_list", "imdb_mean"]
    ].set_index("director_list")
    stats = stats.join(imdb_mean, how="left")
    stats = stats.reset_index()
    return stats


def _director_ranking_insights(stats: pd.DataFrame) -> list[str]:
    if stats.empty:
        return []

    worst = stats.sort_values(
        ["prune_score", "imdb_mean"], ascending=[False, True]
    ).iloc[0]
    best_score = stats.sort_values("imdb_mean", ascending=False).iloc[0]
    top_volume = stats.sort_values("total_count", ascending=False).iloc[0]

    total_prune = float(stats["prune_score"].sum())
    top3_prune = float(
        stats.sort_values("prune_score", ascending=False).head(3)["prune_score"].sum()
    )

    lines: list[str] = []
    lines.append(
        "Mas titulos en revision: "
        f"{worst.director_list} (DELETE {int(worst.delete_count)}, "
        f"MAYBE {int(worst.maybe_count)} | "
        f"{_format_pct(worst.prune_share)})"
        " | "
        "Mejor scoring: "
        f"{best_score.director_list} (IMDb {best_score.imdb_mean:.1f})"
    )

    line_parts = [
        "Mayor volumen: "
        f"{top_volume.director_list} ({int(top_volume.total_count)} titulos, "
        f"{_format_pct(top_volume.prune_share)} en revision)"
    ]
    if pd.notna(worst.imdb_mean) and pd.notna(best_score.imdb_mean):
        delta = float(best_score.imdb_mean - worst.imdb_mean)
        line_parts.append(f"Brecha de score: {delta:.1f} IMDb")
    if total_prune > 0:
        line_parts.append(
            "Revision concentrada: top 3 = "
            f"{_format_pct(top3_prune / total_prune)} ({int(top3_prune)} puntos)"
        )
    else:
        line_parts.append("Sin titulos en revision")
    lines.append(" | ".join(line_parts))

    return lines


def render(
    df_g: pd.DataFrame,
    *,
    dec_sel: AltSelection,
    min_movies_directors: int,
    top_n_directors: int,
    show_insights: bool,
) -> AltChart | None:
    agg = _director_decision_stats(df_g)
    if agg.empty:
        st.info("No se encontraron directores. Revisa filtros o datos de OMDb.")
        return None

    stats = _director_ranking_stats(agg)
    if stats.empty:
        st.info("No hay datos suficientes para el ranking de directores.")
        return None

    stats = stats[stats["total_count"] >= min_movies_directors]

    if stats.empty:
        st.info("No hay directores que cumplan el minimo. Revisa filtros.")
        return None

    if st.session_state.get("charts_view") != "Dashboard":
        st.markdown("**Ranking de directores (peores peliculas)**")

    if show_insights:
        insights = _director_ranking_insights(stats)
        _caption_bullets(insights)

    ordered_stats = stats.sort_values(
        ["prune_score", "imdb_mean"], ascending=[False, True]
    )
    selected = ordered_stats.head(min(top_n_directors, len(ordered_stats))).copy()
    order = selected["director_list"].tolist()

    best_scoring = selected.sort_values("imdb_mean", ascending=False).iloc[0][
        "director_list"
    ]
    if order and order[-1] != best_scoring and order[0] != best_scoring:
        order = [name for name in order if name != best_scoring] + [best_scoring]

    agg_top = agg[agg["director_list"].isin(order)]

    chart_bars = (
        alt.Chart(agg_top)
        .mark_bar()
        .encode(
            x=alt.X("director_list:N", title="Director", sort=order),
            y=alt.Y("count:Q", title="Peliculas"),
            color=_decision_color(),
            tooltip=[
                alt.Tooltip("director_list:N", title="Director"),
                alt.Tooltip("decision:N", title="Decision"),
                alt.Tooltip("count:Q", title="Peliculas", format=".0f"),
                alt.Tooltip("imdb_mean:Q", title="IMDb medio", format=".1f"),
                alt.Tooltip("count_total:Q", title="Total", format=".0f"),
            ],
            opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.2)),
        )
        .add_params(dec_sel)
    )
    imdb_layer_data = stats[stats["director_list"].isin(order)].copy()
    imdb_layer_data = imdb_layer_data[pd.notna(imdb_layer_data["imdb_mean"])]
    accent_color = _chart_accents()["accent"]
    imdb_axis = alt.Axis(
        orient="right",
        title="IMDb medio",
        titleColor=accent_color,
        labelColor=accent_color,
        tickColor=accent_color,
        domainColor=accent_color,
    )
    imdb_line = (
        alt.Chart(imdb_layer_data)
        .mark_line(color=accent_color, strokeWidth=2)
        .encode(
            x=alt.X("director_list:N", sort=order),
            y=alt.Y(
                "imdb_mean:Q",
                axis=imdb_axis,
                scale=alt.Scale(domain=[0, 10]),
            ),
            tooltip=[
                alt.Tooltip("director_list:N", title="Director"),
                alt.Tooltip("imdb_mean:Q", title="IMDb medio", format=".1f"),
                alt.Tooltip("total_count:Q", title="Total", format=".0f"),
                alt.Tooltip("prune_count:Q", title="DELETE+MAYBE", format=".0f"),
                alt.Tooltip("keep_count:Q", title="KEEP", format=".0f"),
            ],
        )
    )
    chart = (chart_bars + imdb_line).resolve_scale(y="independent")
    chart = _chart(chart)
    return chart
