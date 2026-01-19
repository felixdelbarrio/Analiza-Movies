"""Distribución por decisión view."""

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
    _format_pct,
    _requires_columns,
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

    primary_parts: list[str] = []
    if prune_total:
        primary_parts.append(
            f"DELETE + MAYBE: {_format_pct(prune_total / total)} ({prune_total})"
        )
    else:
        primary_parts.append("DELETE + MAYBE: 0")
    if keep_total:
        primary_parts.append(f"KEEP: {_format_pct(keep_total / total)} ({keep_total})")
    if unknown_total:
        primary_parts.append(
            f"UNKNOWN: {_format_pct(unknown_total / total)} ({unknown_total})"
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
                secondary_parts.append(f"Tamaño medio: {avg_prune:.1f} GB")
            size_by_dec = (
                df_g.groupby("decision", dropna=False)["file_size_gb"].sum().to_dict()
            )
            count_share = {
                key: (counts[key] / total if total else 0.0) for key in DECISION_ORDER
            }
            size_share = {
                key: (float(size_by_dec.get(key, 0.0)) / total_size)
                for key in DECISION_ORDER
            }
            deltas = {key: size_share[key] - count_share[key] for key in DECISION_ORDER}
            main_delta = max(deltas.items(), key=lambda item: abs(item[1]))
            secondary_parts.append(
                "Brecha tamaño vs títulos: "
                f"{main_delta[0]} {main_delta[1] * 100:+.1f} pp"
            )
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

    metric_points = pd.concat(
        [
            pd.DataFrame(
                {
                    "decision": agg["decision"].astype("string"),
                    "metric": "Titulos",
                    "value": agg["count_share"],
                    "count": agg["count"],
                    "size_gb": agg["size_gb"],
                }
            ),
            pd.DataFrame(
                {
                    "decision": agg["decision"].astype("string"),
                    "metric": "Espacio",
                    "value": agg["size_share"],
                    "count": agg["count"],
                    "size_gb": agg["size_gb"],
                }
            ),
        ],
        ignore_index=True,
    )

    metric_points["metric"] = pd.Categorical(
        metric_points["metric"], categories=["Titulos", "Espacio"], ordered=True
    )
    axis_pct = alt.Axis(title="Proporción", format=".0%")
    base = alt.Chart(metric_points).encode(
        x=alt.X("metric:N", title="Métrica", sort=["Titulos", "Espacio"]),
        y=alt.Y(
            "value:Q",
            title="Proporción",
            axis=axis_pct,
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=_decision_color(),
        detail="decision:N",
    )
    lines = (
        base.mark_line(strokeWidth=4)
        .encode(opacity=alt.condition(dec_sel, alt.value(0.85), alt.value(0.2)))
        .add_params(dec_sel)
    )
    points = (
        base.mark_point(filled=True, size=120)
        .encode(
            opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.3)),
            tooltip=[
                alt.Tooltip("decision:N", title="Decision"),
                alt.Tooltip("metric:N", title="Métrica"),
                alt.Tooltip("value:Q", title="Proporción", format=".1%"),
                alt.Tooltip("count:Q", title="Peliculas", format=".0f"),
                alt.Tooltip("size_gb:Q", title="Tamano (GB)", format=".2f"),
            ],
        )
        .add_params(dec_sel)
    )
    label_offsets = {
        "DELETE": 0.018,
        "MAYBE": -0.018,
        "KEEP": 0.02,
        "UNKNOWN": -0.02,
    }
    label_points = metric_points[metric_points["metric"] == "Espacio"].copy()
    label_points["label_share"] = (
        label_points["value"] + label_points["decision"].map(label_offsets).fillna(0)
    ).clip(lower=0.0, upper=1.0)
    label_points["label"] = (
        label_points["decision"]
        + " "
        + (label_points["value"] * 100).round(1).astype(str)
        + "%"
    )
    labels = (
        alt.Chart(label_points)
        .mark_text(align="left", dx=8, fontWeight="bold")
        .encode(
            x=alt.X("metric:N", sort=["Titulos", "Espacio"]),
            y=alt.Y("label_share:Q"),
            text="label:N",
            color=_decision_color(),
        )
    )
    chart = lines + points + labels
    chart = _chart(chart)
    return chart
