"""Distribución por decisión view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.tabs.charts_shared import (
    AltChart,
    AltSelection,
    DECISION_ORDER,
    FONT_BODY,
    _all_movies_link,
    _caption_bullets,
    _chart,
    _decision_color,
    _format_pct,
    _requires_columns,
    _theme_tokens,
    _token,
)


def _decision_distribution_insights(
    agg: pd.DataFrame,
    df_g: pd.DataFrame,
) -> list[str]:
    lines: list[str] = []
    if agg.empty or "decision" not in agg.columns or "count" not in agg.columns:
        return lines

    total = int(agg["count"].sum())
    if total <= 0:
        return lines

    counts = {key: 0 for key in DECISION_ORDER}
    for _, row in agg.iterrows():
        decision = str(row.get("decision"))
        if decision in counts:
            counts[decision] += int(row.get("count", 0))

    prune_total = counts["DELETE"] + counts["MAYBE"]
    keep_total = counts["KEEP"]
    unknown_total = counts["UNKNOWN"]

    lines.append("Anillo exterior: títulos. Anillo interior: espacio (GB).")

    primary_parts: list[str] = []
    if prune_total:
        link = _all_movies_link("Ver en Todas", decisions=["DELETE", "MAYBE"])
        primary_parts.append(
            f"En revisión: {prune_total} títulos ({_format_pct(prune_total / total)})"
            + (f" {link}" if link else "")
        )
    else:
        primary_parts.append("En revisión: 0")
    if keep_total:
        primary_parts.append(
            f"KEEP: {keep_total} títulos ({_format_pct(keep_total / total)})"
        )
    if unknown_total:
        primary_parts.append(
            f"UNKNOWN: {unknown_total} títulos ({_format_pct(unknown_total / total)})"
        )
    if primary_parts:
        lines.append(" | ".join(primary_parts))

    secondary_parts: list[str] = []
    if "file_size_gb" in df_g.columns:
        sizes = pd.to_numeric(df_g["file_size_gb"], errors="coerce")
        total_size = float(sizes.fillna(0).sum())
        if total_size > 0:
            mask_prune = df_g["decision"].isin(["DELETE", "MAYBE"])
            prune_size = float(sizes[mask_prune].fillna(0).sum())
            secondary_parts.append(
                "Espacio en revision: "
                f"{prune_size:.1f} GB ({_format_pct(prune_size / total_size)})"
            )
            if prune_total > 0:
                avg_prune = prune_size / prune_total
                secondary_parts.append(f"Tamaño medio en revision: {avg_prune:.1f} GB")
    else:
        if prune_total > 0:
            ratio = max(1, int(round(total / prune_total)))
            secondary_parts.append(f"1 de cada {ratio} titulos esta en revision")
        else:
            secondary_parts.append("Sin titulos en revision actualmente")

    if secondary_parts:
        lines.append(" | ".join(secondary_parts[:2]))
        if len(secondary_parts) > 2:
            lines.append(secondary_parts[2])

    return lines


def render(
    df_g: pd.DataFrame,
    *,
    dec_sel: AltSelection,
    show_insights: bool,
) -> AltChart | None:
    if not _requires_columns(df_g, ["decision", "title"]):
        return None

    agg = (
        df_g.groupby("decision", dropna=False)["title"]
        .count()
        .reset_index()
        .rename(columns={"title": "count"})
    )
    agg["decision"] = pd.Categorical(
        agg["decision"], categories=DECISION_ORDER, ordered=True
    )
    agg = agg.sort_values("decision")
    agg["decision_rank"] = (
        agg["decision"]
        .astype("string")
        .map({key: idx for idx, key in enumerate(DECISION_ORDER)})
        .fillna(len(DECISION_ORDER))
    )
    total_count = float(agg["count"].sum())
    agg["count_share"] = agg["count"] / total_count if total_count else 0.0

    if agg.empty:
        st.info("No hay datos para la distribucion por decision. Revisa filtros.")
        return None

    if show_insights:
        insights = _decision_distribution_insights(agg, df_g)
        _caption_bullets(insights)

    size_share = pd.Series(0.0, index=agg.index, dtype=float)
    size_gb = pd.Series(0.0, index=agg.index, dtype=float)
    if "file_size_gb" in df_g.columns:
        size_agg = (
            df_g.groupby("decision", dropna=False)["file_size_gb"].sum().reset_index()
        )
        agg = agg.merge(size_agg, on="decision", how="left")
        agg["file_size_gb"] = pd.to_numeric(
            agg["file_size_gb"], errors="coerce"
        ).fillna(0)
        total_size = float(agg["file_size_gb"].sum())
        size_gb = agg["file_size_gb"]
        if total_size > 0:
            size_share = agg["file_size_gb"] / total_size
        else:
            size_share = pd.Series(0.0, index=agg.index, dtype=float)
    agg["size_gb"] = size_gb
    agg["size_share"] = size_share

    ring_titles = agg.copy()
    ring_titles["metric"] = "Titulos"
    ring_titles["share"] = ring_titles["count_share"]

    ring_space = agg.copy()
    ring_space["metric"] = "Espacio"
    ring_space["share"] = ring_space["size_share"]

    has_size = bool("file_size_gb" in df_g.columns and float(agg["size_gb"].sum()) > 0)

    tokens = _theme_tokens()
    border = _token(tokens, "panel_border", "#1f2532")
    text_strong = _token(tokens, "text_1", "#f1f5f9")

    tooltip = [
        alt.Tooltip("decision:N", title="Decision"),
        alt.Tooltip("metric:N", title="Anillo"),
        alt.Tooltip("share:Q", title="Proporción", format=".1%"),
        alt.Tooltip("count:Q", title="Titulos", format=".0f"),
        alt.Tooltip("size_gb:Q", title="Tamano (GB)", format=".2f"),
    ]

    base = alt.Chart(ring_titles).encode(
        theta=alt.Theta("share:Q", stack=True),
        color=_decision_color(),
        order=alt.Order("decision_rank:Q"),
        tooltip=tooltip,
    )
    outer_ring = (
        base.mark_arc(
            innerRadius=120,
            outerRadius=175,
            cornerRadius=5,
            padAngle=0.01,
            stroke=border,
            strokeWidth=1.2,
        )
        .encode(opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.25)))
        .add_params(dec_sel)
    )

    inner_ring = alt.Chart(ring_space).encode(
        theta=alt.Theta("share:Q", stack=True),
        color=_decision_color(),
        order=alt.Order("decision_rank:Q"),
        tooltip=tooltip,
    )
    inner_ring = (
        inner_ring.mark_arc(
            innerRadius=70,
            outerRadius=110,
            cornerRadius=4,
            padAngle=0.01,
            stroke=border,
            strokeWidth=1.0,
        )
        .encode(opacity=alt.condition(dec_sel, alt.value(0.9), alt.value(0.2)))
        .add_params(dec_sel)
    )

    label_data = ring_titles[ring_titles["share"] >= 0.06].copy()
    label_data["label"] = (
        label_data["decision"]
        + " "
        + (label_data["share"] * 100).round(1).astype(str)
        + "%"
    )
    label_points = label_data.copy()
    label_points["label_radius"] = (
        label_points["decision"]
        .map({"KEEP": 170, "DELETE": 182, "MAYBE": 182, "UNKNOWN": 182})
        .fillna(182)
    )
    labels = (
        alt.Chart(label_points)
        .mark_text(fontWeight="bold", font=FONT_BODY, fontSize=11)
        .encode(
            theta=alt.Theta("share:Q", stack=True),
            radius=alt.Radius("label_radius:Q"),
            text="label:N",
            color=alt.value(text_strong),
        )
    )

    chart = outer_ring + labels
    if has_size:
        chart = inner_ring + outer_ring + labels
    chart = chart.properties(
        height=420,
        padding={"top": 32, "left": 12, "right": 12, "bottom": 36},
    )
    chart = _chart(chart)
    return chart
