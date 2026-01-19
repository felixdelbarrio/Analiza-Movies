"""Espacio ocupado por biblioteca/decisión view."""

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


def _space_by_library_stats(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()

    data = agg.copy()
    data["decision"] = data["decision"].fillna("UNKNOWN")
    data["file_size_gb"] = pd.to_numeric(data["file_size_gb"], errors="coerce").fillna(
        0
    )

    totals = data.groupby("library", dropna=False)["file_size_gb"].sum()
    if totals.empty:
        return pd.DataFrame()

    stats = pd.DataFrame({"total_gb": totals})
    stats["delete_gb"] = (
        data[data["decision"] == "DELETE"]
        .groupby("library", dropna=False)["file_size_gb"]
        .sum()
    )
    stats["maybe_gb"] = (
        data[data["decision"] == "MAYBE"]
        .groupby("library", dropna=False)["file_size_gb"]
        .sum()
    )
    stats["prune_gb"] = (
        data[data["decision"].isin(["DELETE", "MAYBE"])]
        .groupby("library", dropna=False)["file_size_gb"]
        .sum()
    )
    stats["keep_gb"] = (
        data[data["decision"] == "KEEP"]
        .groupby("library", dropna=False)["file_size_gb"]
        .sum()
    )
    stats["unknown_gb"] = (
        data[data["decision"] == "UNKNOWN"]
        .groupby("library", dropna=False)["file_size_gb"]
        .sum()
    )
    stats = stats.fillna(0)
    stats = stats[stats["total_gb"] > 0]
    if stats.empty:
        return stats

    stats["prune_score_gb"] = _weighted_revision(stats["delete_gb"], stats["maybe_gb"])
    total_score = stats["prune_score_gb"] + stats["keep_gb"] + stats["unknown_gb"]
    stats["prune_share"] = stats["prune_score_gb"] / total_score
    stats["keep_share"] = stats["keep_gb"] / total_score
    stats["unknown_share"] = stats["unknown_gb"] / total_score
    stats = stats.reset_index()
    return stats


def _space_by_library_insights(stats: pd.DataFrame) -> list[str]:
    if stats.empty:
        return []

    top_prune_share = stats.sort_values("prune_share", ascending=False).iloc[0]
    top_keep_share = stats.sort_values("keep_share", ascending=False).iloc[0]
    top_total = stats.sort_values("total_gb", ascending=False).iloc[0]

    total_prune = float(stats["prune_score_gb"].sum())
    top3_prune = float(
        stats.sort_values("prune_score_gb", ascending=False)
        .head(3)["prune_score_gb"]
        .sum()
    )

    lines: list[str] = []
    lines.append(
        "Mayor % en revision: "
        f"{top_prune_share.library} ({_format_pct(top_prune_share.prune_share)} | "
        f"{top_prune_share.prune_score_gb:.1f} GB ponderados)"
        " | "
        "Mayor % KEEP: "
        f"{top_keep_share.library} ({_format_pct(top_keep_share.keep_share)} | "
        f"{top_keep_share.keep_gb:.1f} GB)"
    )

    line_parts = [
        f"Biblioteca mas grande: {top_total.library} ({top_total.total_gb:.1f} GB)"
    ]
    if total_prune > 0:
        line_parts.append(
            "Revision concentrada: top 3 = "
            f"{_format_pct(top3_prune / total_prune)} ({top3_prune:.1f} GB ponderados)"
        )
    else:
        line_parts.append("Sin espacio en revision")
    lines.append(" | ".join(line_parts))

    return lines


def render(
    df_g: pd.DataFrame,
    *,
    lib_sel: AltSelection,
    dec_sel: AltSelection,
    show_insights: bool,
) -> AltChart | None:
    if not _requires_columns(df_g, ["file_size_gb", "library", "decision"]):
        return None

    agg = (
        df_g.groupby(["library", "decision"], dropna=False)["file_size_gb"]
        .sum()
        .reset_index()
    )

    if agg.empty:
        st.info("No hay datos de tamano de archivos. Revisa filtros.")
        return None

    agg["decision"] = agg["decision"].fillna("UNKNOWN")
    agg["file_size_gb"] = pd.to_numeric(agg["file_size_gb"], errors="coerce").fillna(0)
    stats = _space_by_library_stats(agg)
    order = None
    if not stats.empty:
        order = stats.sort_values(
            ["prune_share", "keep_share"], ascending=[False, True]
        )["library"].tolist()

    if st.session_state.get("charts_view") != "Dashboard":
        st.markdown("**Espacio ocupado por biblioteca/decisión**")

    if show_insights and not stats.empty:
        insights = _space_by_library_insights(stats)
        _caption_bullets(insights)

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("library:N", title="Biblioteca", sort=order),
            y=alt.Y("file_size_gb:Q", title="Tamano (GB)", stack="normalize"),
            color=_decision_color(),
            tooltip=[
                "library",
                "decision",
                alt.Tooltip("file_size_gb:Q", title="Tamano (GB)", format=".2f"),
            ],
            opacity=alt.condition(lib_sel & dec_sel, alt.value(1), alt.value(0.2)),
        )
        .add_params(lib_sel, dec_sel)
    )
    chart = _chart(chart)
    return chart
