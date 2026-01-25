"""Palabras mas frecuentes en titulos DELETE/MAYBE view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.tabs.charts_data import _word_counts
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


def _word_ranking_stats(df_words: pd.DataFrame) -> pd.DataFrame:
    if df_words.empty:
        return pd.DataFrame()

    pivot = (
        df_words.pivot_table(
            index="word",
            columns="decision",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .copy()
    )
    if pivot.empty:
        return pd.DataFrame()

    delete_series = (
        pivot["DELETE"]
        if "DELETE" in pivot.columns
        else pd.Series(0, index=pivot.index)
    )
    maybe_series = (
        pivot["MAYBE"] if "MAYBE" in pivot.columns else pd.Series(0, index=pivot.index)
    )
    pivot["delete_count"] = pd.to_numeric(delete_series, errors="coerce").fillna(0)
    pivot["maybe_count"] = pd.to_numeric(maybe_series, errors="coerce").fillna(0)
    pivot["total_prune"] = pivot["delete_count"] + pivot["maybe_count"]
    pivot = pivot[pivot["total_prune"] > 0]
    if pivot.empty:
        return pd.DataFrame()

    pivot["score"] = _weighted_revision(pivot["delete_count"], pivot["maybe_count"])
    pivot["delete_share"] = (
        _weighted_revision(pivot["delete_count"], 0) / pivot["score"]
    )
    pivot["maybe_share"] = _weighted_revision(0, pivot["maybe_count"]) / pivot["score"]
    return pivot


def _word_ranking_insights(stats: pd.DataFrame) -> list[str]:
    if stats.empty:
        return []

    top_score = stats.sort_values("score", ascending=False).iloc[0]
    top_delete_ratio = stats.sort_values("delete_share", ascending=False).iloc[0]
    top_maybe_ratio = stats.sort_values("maybe_share", ascending=False).iloc[0]

    total_score = float(stats["score"].sum())
    top3_score = float(
        stats.sort_values("score", ascending=False).head(3)["score"].sum()
    )

    lines: list[str] = []
    link_score = _all_movies_link(
        "Ver en Todas",
        title=str(top_score.word),
        decisions=["DELETE", "MAYBE"],
    )
    link_delete = _all_movies_link(
        "Ver en Todas",
        title=str(top_delete_ratio.word),
        decisions=["DELETE"],
    )
    lines.append(
        "Palabra mas critica: "
        f"{top_score.word} (DELETE {int(top_score.delete_count)}, "
        f"MAYBE {int(top_score.maybe_count)})"
        + (f" {link_score}" if link_score else "")
        + " | "
        + "Revision mas decisiva: "
        f"{top_delete_ratio.word} ({_format_pct(top_delete_ratio.delete_share)} DELETE)"
        + (f" {link_delete}" if link_delete else "")
    )
    link_maybe = _all_movies_link(
        "Ver en Todas",
        title=str(top_maybe_ratio.word),
        decisions=["MAYBE"],
    )
    lines.append(
        "Mayor indecision: "
        f"{top_maybe_ratio.word} ({_format_pct(top_maybe_ratio.maybe_share)} MAYBE)"
        + (f" {link_maybe}" if link_maybe else "")
        + " | "
        f"Concentracion: top 3 = {_format_pct(top3_score / total_score)} del peso"
    )

    return lines


def render(
    df_g: pd.DataFrame,
    *,
    dec_sel: AltSelection,
    top_n_words: int,
    show_insights: bool,
) -> AltChart | None:
    df_words = _word_counts(df_g, ("DELETE", "MAYBE"))

    if df_words.empty:
        st.info(
            "No hay datos suficientes para el analisis de palabras. Revisa filtros."
        )
        return None

    stats = _word_ranking_stats(df_words)
    if stats.empty:
        st.info("No hay datos suficientes para el ranking de palabras.")
        return None

    stats = stats.sort_values(
        ["score", "maybe_count", "delete_count"], ascending=[False, False, False]
    )
    top_n = min(top_n_words, len(stats))
    top_words = stats.head(top_n)["word"].tolist()
    df_top = df_words[df_words["word"].isin(top_words)]

    if show_insights:
        insights = _word_ranking_insights(stats.head(top_n))
        _caption_bullets(insights)

    chart = (
        alt.Chart(df_top)
        .mark_bar()
        .encode(
            x=alt.X("word:N", title="Palabra", sort=top_words),
            y=alt.Y("count:Q", title="Frecuencia"),
            color=_decision_color(),
            tooltip=["word", "decision", "count"],
            opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.2)),
        )
        .add_params(dec_sel)
    )
    chart = _chart(chart)
    return chart
