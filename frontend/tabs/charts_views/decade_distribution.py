"""Distribución por década view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.tabs.charts_shared import (
    AltChart,
    AltSelection,
    _caption_bullets,
    _chart,
    _decision_color,
    _format_pct,
    _requires_columns,
    _weighted_revision,
)


def _decade_distribution_insights(agg: pd.DataFrame) -> list[str]:
    if agg.empty:
        return []

    data = agg.copy()
    data["decision"] = data["decision"].fillna("UNKNOWN")
    data["count"] = pd.to_numeric(data["count"], errors="coerce").fillna(0)

    totals = data.groupby("decade_label", dropna=False)["count"].sum()
    if totals.empty:
        return []

    stats = pd.DataFrame({"total_count": totals})
    stats["delete_count"] = (
        data[data["decision"] == "DELETE"]
        .groupby("decade_label", dropna=False)["count"]
        .sum()
    )
    stats["maybe_count"] = (
        data[data["decision"] == "MAYBE"]
        .groupby("decade_label", dropna=False)["count"]
        .sum()
    )
    stats["prune_count"] = (
        data[data["decision"].isin(["DELETE", "MAYBE"])]
        .groupby("decade_label", dropna=False)["count"]
        .sum()
    )
    stats["keep_count"] = (
        data[data["decision"] == "KEEP"]
        .groupby("decade_label", dropna=False)["count"]
        .sum()
    )
    stats["unknown_count"] = (
        data[data["decision"] == "UNKNOWN"]
        .groupby("decade_label", dropna=False)["count"]
        .sum()
    )
    stats = stats.fillna(0)
    stats = stats[stats["total_count"] > 0]
    if stats.empty:
        return []

    stats["prune_score"] = _weighted_revision(
        stats["delete_count"], stats["maybe_count"]
    )
    total_score = stats["prune_score"] + stats["keep_count"] + stats["unknown_count"]
    stats["prune_share"] = stats["prune_score"] / total_score
    stats["keep_share"] = stats["keep_count"] / total_score
    stats["unknown_share"] = stats["unknown_count"] / total_score
    stats = stats.reset_index()

    top_prune_share = stats.sort_values("prune_share", ascending=False).iloc[0]
    top_keep_share = stats.sort_values("keep_share", ascending=False).iloc[0]
    top_total = stats.sort_values("total_count", ascending=False).iloc[0]

    total_prune = float(stats["prune_score"].sum())
    top3_prune = float(
        stats.sort_values("prune_score", ascending=False).head(3)["prune_score"].sum()
    )
    total_unknown = float(stats["unknown_count"].sum())

    lines: list[str] = []
    lines.append(
        "Mayor % en revision: "
        f"{top_prune_share.decade_label} ({_format_pct(top_prune_share.prune_share)} | "
        f"{int(top_prune_share.prune_count)} titulos)"
        " | "
        "Mayor % KEEP: "
        f"{top_keep_share.decade_label} ({_format_pct(top_keep_share.keep_share)} | "
        f"{int(top_keep_share.keep_count)} titulos)"
    )

    line_parts = [
        "Decada con mas volumen: "
        f"{top_total.decade_label} ({int(top_total.total_count)} titulos)"
    ]
    if total_prune > 0:
        line_parts.append(
            "Revision concentrada: top 3 = "
            f"{_format_pct(top3_prune / total_prune)} ({int(top3_prune)} puntos)"
        )
    else:
        line_parts.append("Sin titulos en revision")

    if total_unknown > 0:
        top_unknown = stats.sort_values("unknown_share", ascending=False).iloc[0]
        line_parts.append(
            "Mayor UNKNOWN: "
            f"{top_unknown.decade_label} "
            f"({_format_pct(top_unknown.unknown_share)} | "
            f"{int(top_unknown.unknown_count)} titulos)"
        )
    lines.append(" | ".join(line_parts))

    return lines


def render(
    df_g: pd.DataFrame,
    *,
    dec_sel: AltSelection,
    show_insights: bool,
) -> AltChart | None:
    if not _requires_columns(df_g, ["decade_label", "decision", "title"]):
        return None

    data = df_g.dropna(subset=["decade_label"])
    if data.empty:
        st.info("No hay informacion de decada disponible. Revisa filtros.")
        return None

    agg = (
        data.groupby(["decade_label", "decision"], dropna=False)["title"]
        .count()
        .reset_index()
        .rename(columns={"title": "count"})
    )

    if show_insights:
        insights = _decade_distribution_insights(agg)
        _caption_bullets(insights)

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("decade_label:N", title="Década"),
            y=alt.Y("count:Q", title="Número de películas"),
            color=_decision_color(),
            tooltip=["decade_label", "decision", "count"],
            opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.2)),
        )
        .add_params(dec_sel)
    )
    chart = _chart(chart)
    return chart
