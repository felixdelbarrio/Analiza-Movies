"""
charts.py

Pestaña “Gráficos” (Streamlit + Altair).

Responsabilidad:
- Ofrecer distintas vistas de gráficos sobre el DataFrame completo (df_all).
- Validar la presencia de columnas requeridas por cada vista.
- Mantener la UI resiliente: si faltan columnas o no hay datos, mostrar mensajes
  informativos en lugar de lanzar excepciones.

Notas de implementación:
- Se usa st.altair_chart(chart, width="stretch") para evitar use_container_width (deprecado).
- Los gráficos intentan ser “auto-explicativos” con tooltips.
- Para datos derivados (géneros/directores) se parsea omdb_json de forma segura.

Dependencias:
- frontend.data_utils: helpers de parsing y agregación (géneros, word counts, color).
"""

from __future__ import annotations

from typing import Any, Callable, Final, Iterable, TypeVar, cast

import altair as alt
import pandas as pd
import streamlit as st

from frontend.components import render_decision_chip_styles
from frontend.data_utils import (
    build_word_counts,
    decision_color,
    directors_from_omdb_json_or_cache,
    explode_genres_from_omdb_json,
)
from frontend.config_front_theme import get_front_theme, normalize_theme_key
from frontend.config_front_charts import (
    get_dashboard_views,
    get_show_chart_thresholds,
    get_show_numeric_filters,
)

VIEW_OPTIONS: Final[list[str]] = [
    "Dashboard",
    "Ratings IMDb vs Metacritic",
    "Ratings IMDb vs RT",
    "Distribución por decisión",
    "Espacio ocupado por biblioteca/decisión",
    "Distribución por década",
    "Distribución por género (OMDb)",
    "Ranking de directores",
    "Palabras más frecuentes en títulos DELETE/MAYBE",
    "Boxplot IMDb por biblioteca",
    "Rating IMDb por decisión",
]

DECISION_ORDER: Final[list[str]] = ["DELETE", "MAYBE", "KEEP", "UNKNOWN"]

IMDB_OUTLIER_HIGH: Final[float] = 8.5
IMDB_OUTLIER_LOW: Final[float] = 5.0
IMDB_REFERENCE: Final[float] = 7.0
RT_REFERENCE: Final[float] = 60.0
METACRITIC_REFERENCE: Final[float] = 60.0
DELETE_WEIGHT: Final[float] = 3.0
MAYBE_WEIGHT: Final[float] = 1.0

_F = TypeVar("_F", bound=Callable[..., Any])
AltChart = Any
AltSelection = Any
FONT_BODY: Final[str] = "Manrope"
FONT_DISPLAY: Final[str] = "Libre Baskerville"

_DECISION_PALETTES: Final[dict[str, dict[str, str]]] = {
    "noir": {
        "DELETE": "#e55b5b",
        "KEEP": "#56b37a",
        "MAYBE": "#e1b75b",
        "UNKNOWN": "#9da3ad",
    },
    "ivory": {
        "DELETE": "#b3473f",
        "KEEP": "#3f7f5a",
        "MAYBE": "#b0883a",
        "UNKNOWN": "#8f7f70",
    },
    "sapphire": {
        "DELETE": "#e46b78",
        "KEEP": "#4fb08a",
        "MAYBE": "#d7b365",
        "UNKNOWN": "#9aa7bd",
    },
    "verdant": {
        "DELETE": "#e06a63",
        "KEEP": "#4da97a",
        "MAYBE": "#d0b05c",
        "UNKNOWN": "#9aa79f",
    },
    "bordeaux": {
        "DELETE": "#e24b5f",
        "KEEP": "#2f9d6d",
        "MAYBE": "#d79a2b",
        "UNKNOWN": "#7d8faa",
    },
}
_CHART_ACCENTS: Final[dict[str, dict[str, str]]] = {
    "noir": {"accent": "#8dd2ff", "accent_soft": "#5aa7d9"},
    "ivory": {"accent": "#c9894c", "accent_soft": "#a56a3b"},
    "sapphire": {"accent": "#8bbcff", "accent_soft": "#5b84d8"},
    "verdant": {"accent": "#88d6b3", "accent_soft": "#4fa47b"},
    "bordeaux": {"accent": "#2d6fc7", "accent_soft": "#8fb7f1"},
}
_BOXPLOT_GRADIENTS: Final[dict[str, tuple[str, str, str]]] = {
    "noir": ("#1b2635", "#3a6ea8", "#8dd2ff"),
    "ivory": ("#ead8c5", "#c9894c", "#7d4421"),
    "sapphire": ("#1a2740", "#406fd1", "#9dd0ff"),
    "verdant": ("#1a2621", "#3f8b6a", "#b0e0cc"),
    "bordeaux": ("#eef4ff", "#8fb7f1", "#2d6fc7"),
}


def _cache_data_decorator() -> Callable[[_F], _F]:
    cache_fn = getattr(st, "cache_data", None)
    if callable(cache_fn):
        return cast(Callable[[_F], _F], cache_fn(show_spinner=False))
    cache_fn = getattr(st, "cache", None)
    if callable(cache_fn):
        return cast(Callable[[_F], _F], cache_fn)
    return cast(Callable[[_F], _F], lambda f: f)


def _current_theme_key() -> str:
    raw = st.session_state.get("front_theme")
    fallback = get_front_theme()
    return normalize_theme_key(raw if isinstance(raw, str) else fallback)


def _theme_tokens() -> dict[str, str]:
    raw = st.session_state.get("front_theme_tokens")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        out[str(k)] = str(v)
    return out


def _token(tokens: dict[str, str], key: str, fallback: str) -> str:
    return tokens.get(key, fallback)


def _decision_palette() -> dict[str, str]:
    theme_key = _current_theme_key()
    fallback = _DECISION_PALETTES.get(theme_key, _DECISION_PALETTES["noir"])
    tokens = _theme_tokens()
    if tokens:
        return {
            "DELETE": tokens.get("decision_delete", fallback["DELETE"]),
            "KEEP": tokens.get("decision_keep", fallback["KEEP"]),
            "MAYBE": tokens.get("decision_maybe", fallback["MAYBE"]),
            "UNKNOWN": tokens.get("decision_unknown", fallback["UNKNOWN"]),
        }
    return fallback


def _decision_color(field: str = "decision") -> alt.Color:
    return decision_color(field, palette=_decision_palette())


def _boxplot_scale(tokens: dict[str, str]) -> alt.Scale:
    theme_key = _current_theme_key()
    gradient = _BOXPLOT_GRADIENTS.get(theme_key)
    if gradient:
        return alt.Scale(range=list(gradient))
    start = _token(tokens, "metric_bg", "#111722")
    end = _token(tokens, "text_2", "#d1d5db")
    return alt.Scale(range=[start, end])


def _chart_accents() -> dict[str, str]:
    theme_key = _current_theme_key()
    return _CHART_ACCENTS.get(theme_key, _CHART_ACCENTS["noir"])


def _apply_chart_theme(chart: AltChart) -> AltChart:
    tokens = _theme_tokens()
    if not tokens:
        return chart

    bg = _token(tokens, "card_bg", "#11161f")
    border = _token(tokens, "panel_border", "#1f2532")
    text = _token(tokens, "text_2", "#d1d5db")
    text_strong = _token(tokens, "text_1", "#f1f5f9")
    grid = _token(tokens, "divider", "#242a35")

    return (
        chart.properties(background=bg)
        .configure_view(fill=bg, stroke=border, strokeWidth=1)
        .configure_axis(
            labelColor=text,
            titleColor=text_strong,
            gridColor=grid,
            gridOpacity=0.45,
            domainColor=grid,
            tickColor=grid,
            labelFont=FONT_BODY,
            titleFont=FONT_BODY,
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_legend(
            orient="right",
            labelColor=text,
            titleColor=text_strong,
            labelFont=FONT_BODY,
            titleFont=FONT_BODY,
            labelFontSize=11,
            titleFontSize=12,
            symbolSize=80,
            columns=1,
        )
        .configure_title(
            color=text_strong,
            font=FONT_DISPLAY,
            fontSize=16,
            anchor="start",
        )
        .configure_header(
            labelColor=text,
            titleColor=text_strong,
            labelFont=FONT_BODY,
            titleFont=FONT_BODY,
        )
    )


@_cache_data_decorator()
def _genres_agg(df: pd.DataFrame) -> pd.DataFrame:
    df_gen = explode_genres_from_omdb_json(df)
    if (
        df_gen.empty
        or "title" not in df_gen.columns
        or "decision" not in df_gen.columns
    ):
        return pd.DataFrame(columns=["genre", "decision", "count"])

    return (
        df_gen.groupby(["genre", "decision"], dropna=False)["title"]
        .count()
        .reset_index()
        .rename(columns={"title": "count"})
    )


@_cache_data_decorator()
def _director_stats(df: pd.DataFrame) -> pd.DataFrame:
    if not {"imdb_rating", "title"}.issubset(df.columns):
        return pd.DataFrame(columns=["director_list", "imdb_mean", "count"])
    if "omdb_json" not in df.columns and "imdb_id" not in df.columns:
        return pd.DataFrame(columns=["director_list", "imdb_mean", "count"])

    cols: list[str] = ["imdb_rating", "title"]
    if "omdb_json" in df.columns:
        cols.append("omdb_json")
    if "imdb_id" in df.columns:
        cols.append("imdb_id")

    df_dir = df.loc[:, cols].copy()
    if "omdb_json" in df_dir.columns:
        omdb_vals = df_dir["omdb_json"]
    else:
        omdb_vals = pd.Series([None] * len(df_dir), index=df_dir.index)
    if "imdb_id" in df_dir.columns:
        imdb_vals = df_dir["imdb_id"]
    else:
        imdb_vals = pd.Series([None] * len(df_dir), index=df_dir.index)
    df_dir["director_list"] = [
        directors_from_omdb_json_or_cache(omdb_raw, imdb_id)
        for omdb_raw, imdb_id in zip(omdb_vals, imdb_vals)
    ]
    df_dir = df_dir.explode("director_list", ignore_index=True)
    df_dir = df_dir[df_dir["director_list"].notna() & (df_dir["director_list"] != "")]

    if df_dir.empty:
        return pd.DataFrame(columns=["director_list", "imdb_mean", "count"])

    return (
        df_dir.groupby("director_list", dropna=False)
        .agg(
            imdb_mean=("imdb_rating", "mean"),
            count=("title", "count"),
        )
        .reset_index()
    )


@_cache_data_decorator()
def _director_decision_stats(df: pd.DataFrame) -> pd.DataFrame:
    if not {"imdb_rating", "title", "decision"}.issubset(df.columns):
        return pd.DataFrame(
            columns=["director_list", "decision", "count", "imdb_mean", "count_total"]
        )
    if "omdb_json" not in df.columns and "imdb_id" not in df.columns:
        return pd.DataFrame(
            columns=["director_list", "decision", "count", "imdb_mean", "count_total"]
        )

    cols: list[str] = ["imdb_rating", "title", "decision"]
    if "omdb_json" in df.columns:
        cols.append("omdb_json")
    if "imdb_id" in df.columns:
        cols.append("imdb_id")

    df_dir = df.loc[:, cols].copy()
    if "omdb_json" in df_dir.columns:
        omdb_vals = df_dir["omdb_json"]
    else:
        omdb_vals = pd.Series([None] * len(df_dir), index=df_dir.index)
    if "imdb_id" in df_dir.columns:
        imdb_vals = df_dir["imdb_id"]
    else:
        imdb_vals = pd.Series([None] * len(df_dir), index=df_dir.index)
    df_dir["director_list"] = [
        directors_from_omdb_json_or_cache(omdb_raw, imdb_id)
        for omdb_raw, imdb_id in zip(omdb_vals, imdb_vals)
    ]
    df_dir = df_dir.explode("director_list", ignore_index=True)
    df_dir = df_dir[df_dir["director_list"].notna() & (df_dir["director_list"] != "")]

    if df_dir.empty:
        return pd.DataFrame(
            columns=["director_list", "decision", "count", "imdb_mean", "count_total"]
        )

    stats_mean = (
        df_dir.groupby("director_list", dropna=False)
        .agg(imdb_mean=("imdb_rating", "mean"), count_total=("title", "count"))
        .reset_index()
    )
    counts = (
        df_dir.groupby(["director_list", "decision"], dropna=False)["title"]
        .count()
        .reset_index()
        .rename(columns={"title": "count"})
    )
    out = counts.merge(stats_mean, on="director_list", how="left")
    return out


@_cache_data_decorator()
def _word_counts(df: pd.DataFrame, decisions: tuple[str, ...]) -> pd.DataFrame:
    return build_word_counts(df, decisions)


def _requires_columns(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    """
    Comprueba que `df` contiene todas las columnas indicadas.

    Returns:
      - True  si todas las columnas están presentes.
      - False si falta alguna (y muestra un mensaje informativo en Streamlit).
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.info(
            f"Faltan columna(s) requerida(s): {', '.join(missing)}. "
            "Revisa la fuente de datos."
        )
        return False
    return True


def _chart(chart: AltChart) -> AltChart:
    """
    Wrapper para mostrar gráficos siempre a ancho completo.
    """
    chart = _apply_chart_theme(chart)
    st.altair_chart(chart, width="stretch")
    return chart


def _caption_bullets(lines: list[str]) -> None:
    if not lines:
        return
    for line in lines:
        st.caption(f"- {line}")


def _chart_png_bytes(chart: AltChart) -> bytes | None:
    try:
        import io

        buf = io.BytesIO()
        chart.save(buf, format="png")
        return buf.getvalue()
    except Exception:
        return None


def _chart_svg_bytes(chart: AltChart) -> bytes | None:
    try:
        import io

        buf = io.BytesIO()
        chart.save(buf, format="svg")
        return buf.getvalue()
    except Exception:
        return None


def _ordered_options(values: Iterable[object], order: list[str]) -> list[str]:
    unique = {str(v) for v in values if v is not None and str(v).strip() != ""}
    ordered: list[str] = []
    for item in order:
        if item in unique:
            ordered.append(item)
            unique.discard(item)
    ordered.extend(sorted(unique))
    return ordered


def _movie_tooltips(df: pd.DataFrame) -> list[alt.Tooltip]:
    out: list[alt.Tooltip] = []
    if "title" in df.columns:
        out.append(alt.Tooltip("title:N", title="Titulo"))
    if "year" in df.columns:
        out.append(alt.Tooltip("year:Q", title="Ano", format=".0f"))
    if "library" in df.columns:
        out.append(alt.Tooltip("library:N", title="Biblioteca"))
    if "decision" in df.columns:
        out.append(alt.Tooltip("decision:N", title="Decision"))
    if "imdb_rating" in df.columns:
        out.append(alt.Tooltip("imdb_rating:Q", title="IMDb", format=".1f"))
    if "rt_score" in df.columns:
        out.append(alt.Tooltip("rt_score:Q", title="RT (%)", format=".0f"))
    if "metacritic_score" in df.columns:
        out.append(alt.Tooltip("metacritic_score:Q", title="Metacritic", format=".0f"))
    if "imdb_votes" in df.columns:
        out.append(alt.Tooltip("imdb_votes:Q", title="Votos IMDb", format=",.0f"))
    if "file_size_gb" in df.columns:
        out.append(alt.Tooltip("file_size_gb:Q", title="Tamano (GB)", format=".2f"))
    return out


def _mark_imdb_outliers(
    data: pd.DataFrame,
    *,
    low: float = IMDB_OUTLIER_LOW,
    high: float = IMDB_OUTLIER_HIGH,
) -> pd.DataFrame:
    out = data.copy()
    out["imdb_outlier"] = pd.NA
    out.loc[out["imdb_rating"] >= high, "imdb_outlier"] = "Alta"
    out.loc[out["imdb_rating"] <= low, "imdb_outlier"] = "Baja"
    return out


def _render_view(
    view: str,
    df_g: pd.DataFrame,
    *,
    lib_sel: AltSelection,
    dec_sel: AltSelection,
    imdb_ref: float,
    rt_ref: float,
    meta_ref: float,
    outlier_low: float,
    outlier_high: float,
    top_n_genres: int,
    min_movies_directors: int,
    top_n_directors: int,
    top_n_words: int,
    top_n_libs: int | None,
    min_n_libs: int,
    boxplot_horizontal: bool,
    compare_libs: list[str],
    show_insights: bool = True,
) -> AltChart | None:
    chart_export: AltChart | None = None
    tokens = _theme_tokens()
    accents = _chart_accents()
    ref_color = _token(tokens, "text_3", "#666666")
    mean_color = accents["accent_soft"]
    median_color = _token(tokens, "text_1", "#ffffff")
    accent_color = accents["accent"]
    boxplot_scale = _boxplot_scale(tokens)
    # 1) Distribución por decisión
    if view == "Distribución por decisión":
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
                df_g.groupby("decision", dropna=False)["file_size_gb"]
                .sum()
                .reset_index()
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
            label_points["value"]
            + label_points["decision"].map(label_offsets).fillna(0)
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
        chart_export = chart

    # 2) Rating IMDb por decisión
    elif view == "Rating IMDb por decisión":
        if not _requires_columns(df_g, ["imdb_rating", "decision"]):
            return None

        data = df_g.dropna(subset=["imdb_rating"])
        if data.empty:
            st.info("No hay ratings IMDb validos para mostrar. Revisa filtros.")
            return None

        if show_insights:
            insights = _imdb_decision_insights(data, imdb_ref)
            _caption_bullets(insights)

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
        chart_export = chart

    # 3) Ratings IMDb vs RT
    elif view == "Ratings IMDb vs RT":
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
        chart_export = chart

    # 4) Ratings IMDb vs Metacritic
    elif view == "Ratings IMDb vs Metacritic":
        if not _requires_columns(df_g, ["imdb_rating", "metacritic_score", "decision"]):
            return None

        data = df_g.dropna(subset=["imdb_rating", "metacritic_score"])
        if data.empty:
            st.info("No hay suficientes datos de IMDb y Metacritic. Revisa filtros.")
            return None

        if show_insights:
            insights = _imdb_metacritic_insights(data, imdb_ref, meta_ref)
            _caption_bullets(insights)

        data = _mark_imdb_outliers(data, low=outlier_low, high=outlier_high)
        base = (
            alt.Chart(data)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("imdb_rating:Q", title="IMDb rating (0-10)"),
                y=alt.Y(
                    "metacritic_score:Q",
                    title="Metacritic score (0-100)",
                ),
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
                y=alt.Y("metacritic_score:Q"),
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
        ref_meta = (
            alt.Chart(pd.DataFrame({"metacritic_score": [meta_ref]}))
            .mark_rule(color=ref_color, strokeDash=[4, 4])
            .encode(y=alt.Y("metacritic_score:Q"))
        )
        chart = base + outliers + ref_imdb + ref_meta
        chart = _chart(chart)
        chart_export = chart

    # 5) Distribución por década
    elif view == "Distribución por década":
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
        chart_export = chart

    # 6) Distribución por género (OMDb)
    elif view == "Distribución por género (OMDb)":
        agg = _genres_agg(df_g)

        if agg.empty:
            st.info("No hay datos de genero. Revisa filtros o datos de OMDb.")
            return None

        top_n = top_n_genres
        top_genres = (
            agg.groupby("genre")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
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
            st.markdown(f"**{view}**")

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
        chart_export = chart

    # 7) Espacio ocupado por biblioteca/decisión
    elif view == "Espacio ocupado por biblioteca/decisión":
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
        agg["file_size_gb"] = pd.to_numeric(
            agg["file_size_gb"], errors="coerce"
        ).fillna(0)
        stats = _space_by_library_stats(agg)
        order = None
        if not stats.empty:
            order = stats.sort_values(
                ["prune_share", "keep_share"], ascending=[False, True]
            )["library"].tolist()

        if st.session_state.get("charts_view") != "Dashboard":
            st.markdown(f"**{view}**")

        if show_insights and not stats.empty:
            insights = _space_by_library_insights(stats)
            _caption_bullets(insights)

        chart_space = (
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
        chart_space = _chart(chart_space)
        chart_export = chart_space

    # 8) Boxplot IMDb por biblioteca
    elif view == "Boxplot IMDb por biblioteca":
        if not _requires_columns(df_g, ["imdb_rating", "library"]):
            return None

        data = df_g.dropna(subset=["imdb_rating", "library"])
        if data.empty:
            st.info("No hay datos suficientes de IMDb/biblioteca. Revisa filtros.")
            return None

        stats = (
            data.groupby("library", dropna=False)["imdb_rating"]
            .agg(imdb_median="median", count="count")
            .reset_index()
        )
        if stats.empty:
            st.info("No hay estadisticas para bibliotecas. Revisa filtros.")
            return None

        global_mean = float(data["imdb_rating"].mean())
        num_libs = int(stats["library"].nunique())
        max_libs = max(1, min(50, num_libs))
        top_n = top_n_libs if top_n_libs is not None else min(20, max_libs)
        min_n = min_n_libs
        horizontal = boxplot_horizontal

        if top_n < num_libs:
            top_libs = (
                stats.sort_values("count", ascending=False)
                .head(top_n)["library"]
                .tolist()
            )
            data = data[data["library"].isin(top_libs)]
            stats = stats[stats["library"].isin(top_libs)]
        stats = stats[stats["count"] >= min_n]
        data = data[data["library"].isin(stats["library"])]
        if data.empty:
            st.info("No hay datos tras aplicar filtros. Revisa filtros.")
            return None

        data = data.merge(stats, on="library", how="left")
        order = stats.sort_values("imdb_median", ascending=False)["library"].tolist()
        top3 = stats.head(3)
        bottom3 = stats.tail(3)
        boxplot_insights = [
            "Top medianas: "
            + ", ".join(
                [f"{row.library} ({row.imdb_median:.1f})" for row in top3.itertuples()]
            ),
            "Bottom medianas: "
            + ", ".join(
                [
                    f"{row.library} ({row.imdb_median:.1f})"
                    for row in bottom3.itertuples()
                ]
            ),
        ]
        if len(compare_libs) == 2:
            comp = stats[stats["library"].isin(compare_libs)].copy()
            if len(comp) == 2:
                delta = comp.iloc[0]["imdb_median"] - comp.iloc[1]["imdb_median"]
                boxplot_insights.append(
                    f"Comparacion: {compare_libs[0]} vs {compare_libs[1]} | "
                    f"Delta mediana IMDb: {delta:.2f}"
                )
                comp_chart = (
                    alt.Chart(comp)
                    .mark_bar()
                    .encode(
                        x=alt.X("library:N", title="Biblioteca"),
                        y=alt.Y("imdb_median:Q", title="Mediana IMDb"),
                        tooltip=[
                            alt.Tooltip("library:N", title="Biblioteca"),
                            alt.Tooltip(
                                "imdb_median:Q", title="Mediana IMDb", format=".1f"
                            ),
                            alt.Tooltip("count:Q", title="Peliculas", format=".0f"),
                        ],
                    )
                )
                comp_chart = _chart(comp_chart)
        if show_insights:
            _caption_bullets(boxplot_insights)

        tooltip_common = [
            alt.Tooltip("library:N", title="Biblioteca"),
            alt.Tooltip("imdb_median:Q", title="Mediana IMDb", format=".1f"),
            alt.Tooltip("count:Q", title="Peliculas", format=".0f"),
        ]
        ref_line = (
            alt.Chart(pd.DataFrame({"imdb_rating": [imdb_ref]}))
            .mark_rule(color=ref_color, strokeDash=[4, 4])
            .encode(y=alt.Y("imdb_rating:Q"))
        )
        mean_line = (
            alt.Chart(pd.DataFrame({"imdb_rating": [global_mean]}))
            .mark_rule(color=mean_color, strokeDash=[2, 2])
            .encode(y=alt.Y("imdb_rating:Q"))
        )
        color_scale = boxplot_scale
        chart_box = (
            alt.Chart(data)
            .mark_boxplot(size=40)
            .encode(
                x=(
                    alt.X(
                        "library:N",
                        title="Biblioteca",
                        sort=order,
                        axis=alt.Axis(labelAngle=-45, labelLimit=140),
                    )
                    if not horizontal
                    else alt.X("imdb_rating:Q", title="IMDb rating (0-10)")
                ),
                y=(
                    alt.Y("imdb_rating:Q", title="IMDb rating (0-10)")
                    if not horizontal
                    else alt.Y(
                        "library:N",
                        title="Biblioteca",
                        sort=order,
                    )
                ),
                color=alt.Color("imdb_median:Q", scale=color_scale, legend=None),
                tooltip=tooltip_common,
                opacity=alt.condition(lib_sel, alt.value(1), alt.value(0.2)),
            )
        )
        chart_median = (
            alt.Chart(stats)
            .mark_tick(color=median_color, thickness=2)
            .encode(
                x=(
                    alt.X("library:N", sort=order)
                    if not horizontal
                    else alt.X("imdb_median:Q")
                ),
                y=(
                    alt.Y("imdb_median:Q")
                    if not horizontal
                    else alt.Y("library:N", sort=order)
                ),
                tooltip=tooltip_common,
            )
        )
        chart_strip = (
            alt.Chart(data)
            .transform_calculate(
                imdb_jitter="datum.imdb_rating + (random() - 0.5) * 0.2"
            )
            .mark_circle(size=18, opacity=0.25)
            .encode(
                x=(
                    alt.X(
                        "library:N",
                        sort=order,
                        axis=alt.Axis(labelAngle=-45, labelLimit=140),
                    )
                    if not horizontal
                    else alt.X("imdb_jitter:Q")
                ),
                y=(
                    alt.Y("imdb_jitter:Q")
                    if not horizontal
                    else alt.Y("library:N", sort=order)
                ),
                color=alt.value(accent_color),
                tooltip=_movie_tooltips(data),
                opacity=alt.condition(lib_sel, alt.value(1), alt.value(0.1)),
            )
            .add_params(lib_sel)
        )
        chart = chart_strip + chart_box + chart_median + ref_line + mean_line
        chart = _chart(chart)
        chart_export = chart

    # 9) Ranking de directores
    elif view == "Ranking de directores":
        agg = _director_decision_stats(df_g)
        if agg.empty:
            st.info("No se encontraron directores. Revisa filtros o datos de OMDb.")
            return None

        stats = _director_ranking_stats(agg)
        if stats.empty:
            st.info("No hay datos suficientes para el ranking de directores.")
            return None

        min_movies = min_movies_directors
        top_n = top_n_directors
        stats = stats[stats["total_count"] >= min_movies]

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
        selected = ordered_stats.head(min(top_n, len(ordered_stats))).copy()
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
        chart_export = chart

    # 10) Palabras más frecuentes en títulos DELETE/MAYBE
    elif view == "Palabras más frecuentes en títulos DELETE/MAYBE":
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
        chart_export = chart

    return chart_export


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_num(value: float | int | None, fmt: str = ".1f") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return format(float(value), fmt)


def _weighted_revision(delete_value: Any, maybe_value: Any) -> Any:
    return delete_value * DELETE_WEIGHT + maybe_value * MAYBE_WEIGHT


def _corr_strength(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 0.7:
        return "alta"
    if abs_value >= 0.4:
        return "moderada"
    if abs_value >= 0.2:
        return "baja"
    return "muy baja"


def _imdb_metacritic_insights(
    data: pd.DataFrame,
    imdb_ref: float,
    meta_ref: float,
) -> list[str]:
    lines: list[str] = []
    if data.empty:
        return lines

    imdb_raw = pd.to_numeric(data["imdb_rating"], errors="coerce")
    meta_raw = pd.to_numeric(data["metacritic_score"], errors="coerce")
    mask = imdb_raw.notna() & meta_raw.notna()
    total = int(mask.sum())
    if total <= 0:
        return lines

    imdb_scaled = imdb_raw[mask] * 10
    meta = meta_raw[mask]
    gap = imdb_scaled - meta

    within_10 = int((gap.abs() <= 10).sum())
    big_gap = int((gap.abs() >= 20).sum())

    line_parts: list[str] = []
    if total > 1:
        corr = imdb_scaled.corr(meta)
        if pd.notna(corr):
            strength = _corr_strength(corr)
            line_parts.append(f"Relacion IMDb vs Metacritic: r={corr:.2f} ({strength})")
    line_parts.append(
        f"Alineacion: {_format_pct(within_10 / total)} ({within_10}) dentro de +/-10 pts"
    )
    if line_parts:
        lines.append(" | ".join(line_parts))

    median_gap = float(gap.median())
    if median_gap >= 0:
        gap_label = f"Brecha mediana: IMDb esta +{median_gap:.1f} pts sobre Metacritic"
    else:
        gap_label = (
            f"Brecha mediana: Metacritic esta +{abs(median_gap):.1f} pts sobre IMDb"
        )
    lines.append(
        f"{gap_label} | Discrepancias fuertes: "
        f"{_format_pct(big_gap / total)} ({big_gap}) >= 20 pts"
    )

    consensus_high = int(((imdb_raw >= imdb_ref) & (meta_raw >= meta_ref) & mask).sum())
    consensus_low = int(((imdb_raw < imdb_ref) & (meta_raw < meta_ref) & mask).sum())
    lines.append(
        "Consenso en umbrales: "
        f"{_format_pct(consensus_high / total)} ({consensus_high}) por encima de ambos "
        f"(IMDb >= {imdb_ref:.1f}, Metacritic >= {meta_ref:.0f}) | "
        f"{_format_pct(consensus_low / total)} ({consensus_low}) por debajo de ambos"
    )

    return lines


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
    lines.append(
        "Mayor % en revision: "
        f"{top_prune_share.genre} ({_format_pct(top_prune_share.prune_share)} | "
        f"{int(top_prune_share.prune_count)} titulos)"
        " | "
        "Mayor % KEEP: "
        f"{top_keep_share.genre} ({_format_pct(top_keep_share.keep_share)} | "
        f"{int(top_keep_share.keep_count)} titulos)"
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
        line_parts.append(
            "Mayor UNKNOWN: "
            f"{top_unknown.genre} ({_format_pct(top_unknown.unknown_share)} | "
            f"{int(top_unknown.unknown_count)} titulos)"
        )
    else:
        line_parts.append(
            f"Mayor volumen total: {top_total.genre} ({int(top_total.total_count)} titulos)"
        )
    lines.append(" | ".join(line_parts))

    return lines


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
    lines.append(
        "Palabra mas critica: "
        f"{top_score.word} (DELETE {int(top_score.delete_count)}, "
        f"MAYBE {int(top_score.maybe_count)})"
        " | "
        "Revision mas decisiva: "
        f"{top_delete_ratio.word} ({_format_pct(top_delete_ratio.delete_share)} DELETE)"
    )
    lines.append(
        "Mayor indecision: "
        f"{top_maybe_ratio.word} ({_format_pct(top_maybe_ratio.maybe_share)} MAYBE)"
        " | "
        f"Concentracion: top 3 = {_format_pct(top3_score / total_score)} del peso"
    )

    return lines


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
                "Espacio en revisión: "
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
            secondary_parts.append(f"1 de cada {ratio} títulos está en revisión")
        else:
            secondary_parts.append("Sin títulos en revisión actualmente")

    if secondary_parts:
        lines.append(" | ".join(secondary_parts[:2]))
        if len(secondary_parts) > 2:
            lines.append(secondary_parts[2])

    return lines


def _auto_insights(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    if "decision" in df.columns:
        counts = df["decision"].value_counts(dropna=False)
        if not counts.empty:
            top_dec = counts.index[0]
            insights.append(f"Decision con mas titulos: {top_dec} ({counts.iloc[0]}).")
    if "library" in df.columns and "imdb_rating" in df.columns:
        stats = (
            df.dropna(subset=["library", "imdb_rating"])
            .groupby("library")["imdb_rating"]
            .median()
            .sort_values(ascending=False)
        )
        if not stats.empty:
            top_lib = stats.index[0]
            bot_lib = stats.index[-1]
            insights.append(
                f"Biblioteca con mejor mediana IMDb: {top_lib} ({stats.iloc[0]:.1f})."
            )
            if top_lib != bot_lib:
                insights.append(
                    f"Biblioteca con peor mediana IMDb: {bot_lib} ({stats.iloc[-1]:.1f})."
                )
    if "imdb_rating" in df.columns:
        mean_imdb = pd.to_numeric(df["imdb_rating"], errors="coerce").mean()
        if pd.notna(mean_imdb):
            insights.append(f"IMDb medio global: {mean_imdb:.2f}.")
    return insights[:3]


def render(df_all: pd.DataFrame) -> None:
    """
    Renderiza la pestaña 5: Gráficos.

    Args:
        df_all: DataFrame completo (puede venir vacío).
    """
    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        st.info("No hay datos para mostrar graficos. Revisa la fuente de datos.")
        return

    df_g = df_all
    chart_export: (
        alt.Chart | alt.LayerChart | alt.HConcatChart | alt.VConcatChart | None
    ) = None

    lib_sel = alt.selection_point(
        fields=["library"],
        name="Library",
        empty=cast(Any, "all"),
        on="mouseover",
        clear="mouseout",
    )
    dec_sel = alt.selection_point(
        fields=["decision"],
        name="Decision",
        empty=cast(Any, "all"),
        on="mouseover",
        clear="mouseout",
    )

    if "charts_view" not in st.session_state:
        st.session_state["charts_view"] = VIEW_OPTIONS[0]
    if st.session_state.get("charts_view") not in VIEW_OPTIONS:
        st.session_state["charts_view"] = VIEW_OPTIONS[0]

    view = st.selectbox("Vista", VIEW_OPTIONS, key="charts_view")
    show_exec = view == "Dashboard"

    selected_libs: list[str] | None = None
    selected_decisions: list[str] | None = None
    top_n_genres: int = 20
    min_movies_directors: int = 3
    top_n_directors: int = 20
    top_n_words: int = 20
    top_n_libs: int | None = None
    min_n_libs: int = 5
    boxplot_horizontal: bool = False
    year_range: tuple[int, int] | None = None
    imdb_ref = IMDB_REFERENCE
    rt_ref = RT_REFERENCE
    meta_ref = METACRITIC_REFERENCE
    outlier_low = IMDB_OUTLIER_LOW
    outlier_high = IMDB_OUTLIER_HIGH
    compare_libs: list[str] = []
    export_csv = True

    show_numeric_filters = get_show_numeric_filters()
    show_chart_thresholds = get_show_chart_thresholds()

    with st.expander("Filtros", expanded=False):
        st.caption("Si los filtros estan vacios, se muestran todos los datos.")
        filters_cols = st.columns(2)
        with filters_cols[0]:
            if "library" in df_g.columns:
                libraries = _ordered_options(df_g["library"], [])
                if libraries:
                    selected_libs = st.multiselect(
                        "Biblioteca",
                        libraries,
                        default=[],
                        key="charts_filter_library",
                    )
        with filters_cols[1]:
            if "decision" in df_g.columns:
                decisions = _ordered_options(
                    df_g["decision"],
                    DECISION_ORDER,
                )
                if decisions:
                    selected_decisions = st.multiselect(
                        "Decision",
                        decisions,
                        default=[],
                        key="charts_filter_decision",
                    )
                    colorize = bool(st.session_state.get("grid_colorize_rows", True))
                    render_decision_chip_styles(
                        "decision-chips-charts",
                        enabled=colorize,
                        selected_values=(
                            list(selected_decisions) if selected_decisions else []
                        ),
                    )
        if show_numeric_filters:
            has_view_filters = view in {
                "Distribución por género (OMDb)",
                "Ranking de directores",
                "Palabras más frecuentes en títulos DELETE/MAYBE",
                "Boxplot IMDb por biblioteca",
            }
            if has_view_filters:
                st.markdown("**Filtros numericos**")
            if view == "Distribución por género (OMDb)":
                top_n_genres = st.slider(
                    "Top N generos", 5, 50, 20, key="charts_top_n_genres"
                )
            elif view == "Ranking de directores":
                min_movies_directors = st.slider(
                    "Minimo de peliculas por director",
                    1,
                    10,
                    3,
                    key="charts_min_movies_directors",
                )
                top_n_directors = st.slider(
                    "Directores a mostrar (peor -> mejor)",
                    5,
                    50,
                    20,
                    key="charts_top_n_directors",
                )
            elif view == "Palabras más frecuentes en títulos DELETE/MAYBE":
                top_n_words = st.slider(
                    "Top N palabras (peor -> mejor)",
                    5,
                    50,
                    20,
                    key="charts_top_n_words",
                )
            elif view == "Boxplot IMDb por biblioteca":
                num_libs = (
                    int(df_g["library"].nunique()) if "library" in df_g.columns else 1
                )
                max_libs = max(1, min(50, num_libs))
                default_top = min(20, max_libs)
                top_n_libs = st.slider(
                    "Top N bibliotecas por volumen",
                    1,
                    max_libs,
                    default_top,
                    key="charts_top_n_libs",
                )
                min_n_libs = st.slider(
                    "Minimo de peliculas por biblioteca",
                    1,
                    200,
                    5,
                    key="charts_min_n_libs",
                )
                boxplot_horizontal = st.checkbox(
                    "Vista horizontal", value=False, key="charts_boxplot_horizontal"
                )
                if "library" in df_g.columns:
                    compare_libs = st.multiselect(
                        "Comparar 2 bibliotecas",
                        _ordered_options(df_g["library"], []),
                        default=[],
                        max_selections=2,
                        key="charts_compare_libs",
                    )

        if show_chart_thresholds:
            st.markdown("**Umbrales**")
            imdb_ref = st.slider(
                "Referencia IMDb", 1.0, 9.0, IMDB_REFERENCE, 0.1, key="charts_imdb_ref"
            )
            outlier_low = st.slider(
                "IMDb outlier bajo",
                0.0,
                9.0,
                IMDB_OUTLIER_LOW,
                0.1,
                key="charts_imdb_outlier_low",
            )
            outlier_high = st.slider(
                "IMDb outlier alto",
                1.0,
                10.0,
                IMDB_OUTLIER_HIGH,
                0.1,
                key="charts_imdb_outlier_high",
            )
            rt_ref = st.slider(
                "Referencia RT (%)", 0, 100, int(RT_REFERENCE), key="charts_rt_ref"
            )
            meta_ref = st.slider(
                "Referencia Metacritic",
                0,
                100,
                int(METACRITIC_REFERENCE),
                key="charts_meta_ref",
            )

        rating_views = {
            "Ratings IMDb vs RT",
            "Ratings IMDb vs Metacritic",
            "Rating IMDb por decisión",
            "Boxplot IMDb por biblioteca",
        }
        if show_numeric_filters and view in rating_views and "year" in df_g.columns:
            year_vals = pd.to_numeric(df_g["year"], errors="coerce").dropna()
            if not year_vals.empty:
                min_year = int(year_vals.min())
                max_year = int(year_vals.max())
                year_range = st.slider(
                    "Rango de años",
                    min_year,
                    max_year,
                    (min_year, max_year),
                    key="charts_year_range",
                )

    if selected_libs == []:
        selected_libs = None
    if selected_decisions == []:
        selected_decisions = None

    if selected_libs is not None:
        df_g = df_g[df_g["library"].isin(selected_libs)]
    if selected_decisions is not None:
        df_g = df_g[df_g["decision"].isin(selected_decisions)]
    if year_range is not None and "year" in df_g.columns:
        year_num = pd.to_numeric(df_g["year"], errors="coerce")
        df_g = df_g[year_num.between(year_range[0], year_range[1], inclusive="both")]

    if df_g.empty:
        st.info("No hay datos tras aplicar filtros. Revisa filtros.")
        return

    if show_exec:
        available_exec = [v for v in VIEW_OPTIONS if v != "Dashboard"]
        exec_views = get_dashboard_views(available_exec)
        exec_views = exec_views[:3]
        exec_cols = st.columns(len(exec_views))
        for exec_view, col in zip(exec_views, exec_cols, strict=False):
            with col:
                st.markdown(f"**{exec_view}**")
                _render_view(
                    exec_view,
                    df_g,
                    lib_sel=lib_sel,
                    dec_sel=dec_sel,
                    imdb_ref=imdb_ref,
                    rt_ref=rt_ref,
                    meta_ref=meta_ref,
                    outlier_low=outlier_low,
                    outlier_high=outlier_high,
                    top_n_genres=top_n_genres,
                    min_movies_directors=min_movies_directors,
                    top_n_directors=top_n_directors,
                    top_n_words=top_n_words,
                    top_n_libs=top_n_libs,
                    min_n_libs=min_n_libs,
                    boxplot_horizontal=boxplot_horizontal,
                    compare_libs=compare_libs,
                    show_insights=True,
                )
        insights = _auto_insights(df_g)
        if insights:
            st.markdown("**Insights automaticos**")
            for item in insights:
                st.write(f"- {item}")
        return

    chart_export = _render_view(
        view,
        df_g,
        lib_sel=lib_sel,
        dec_sel=dec_sel,
        imdb_ref=imdb_ref,
        rt_ref=rt_ref,
        meta_ref=meta_ref,
        outlier_low=outlier_low,
        outlier_high=outlier_high,
        top_n_genres=top_n_genres,
        min_movies_directors=min_movies_directors,
        top_n_directors=top_n_directors,
        top_n_words=top_n_words,
        top_n_libs=top_n_libs,
        min_n_libs=min_n_libs,
        boxplot_horizontal=boxplot_horizontal,
        compare_libs=compare_libs,
    )

    if chart_export is not None:
        st.markdown("**Descargas**")
        if export_csv:
            csv_data = df_g.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar CSV filtrado",
                csv_data,
                file_name="charts_filtered.csv",
                mime="text/csv",
            )
        svg_bytes = _chart_svg_bytes(chart_export)
        if svg_bytes:
            st.download_button(
                "Descargar SVG del grafico",
                svg_bytes,
                file_name="chart.svg",
                mime="image/svg+xml",
            )
        png_bytes = _chart_png_bytes(chart_export)
        if png_bytes:
            st.download_button(
                "Descargar PNG del grafico",
                png_bytes,
                file_name="chart.png",
                mime="image/png",
            )
        if not svg_bytes and not png_bytes:
            st.info("Export no disponible: requiere dependencias extra de Altair.")
