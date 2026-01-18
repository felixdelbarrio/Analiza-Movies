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

from frontend.data_utils import (
    build_word_counts,
    decision_color,
    directors_from_omdb_json_or_cache,
    explode_genres_from_omdb_json,
)
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

IMDB_OUTLIER_HIGH: Final[float] = 8.5
IMDB_OUTLIER_LOW: Final[float] = 5.0
IMDB_REFERENCE: Final[float] = 7.0
RT_REFERENCE: Final[float] = 60.0
METACRITIC_REFERENCE: Final[float] = 60.0

_F = TypeVar("_F", bound=Callable[..., Any])


def _cache_data_decorator() -> Callable[[_F], _F]:
    cache_fn = getattr(st, "cache_data", None)
    if callable(cache_fn):
        return cast(Callable[[_F], _F], cache_fn(show_spinner=False))
    cache_fn = getattr(st, "cache", None)
    if callable(cache_fn):
        return cast(Callable[[_F], _F], cache_fn)
    return cast(Callable[[_F], _F], lambda f: f)


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


def _chart(
    chart: alt.Chart | alt.LayerChart | alt.HConcatChart | alt.VConcatChart,
) -> None:
    """
    Wrapper para mostrar gráficos siempre a ancho completo.
    """
    chart = chart.configure_legend(
        orient="right",
        titleFontSize=12,
        labelFontSize=11,
        symbolSize=80,
        columns=1,
    )
    st.altair_chart(chart, width="stretch")


def _chart_png_bytes(
    chart: alt.Chart | alt.LayerChart | alt.HConcatChart | alt.VConcatChart,
) -> bytes | None:
    try:
        import io

        buf = io.BytesIO()
        chart.save(buf, format="png")
        return buf.getvalue()
    except Exception:
        return None


def _chart_svg_bytes(
    chart: alt.Chart | alt.LayerChart | alt.HConcatChart | alt.VConcatChart,
) -> bytes | None:
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
    lib_sel: alt.Selection,
    dec_sel: alt.Selection,
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
) -> alt.Chart | alt.LayerChart | alt.HConcatChart | alt.VConcatChart | None:
    chart_export: (
        alt.Chart | alt.LayerChart | alt.HConcatChart | alt.VConcatChart | None
    ) = None
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

        if agg.empty:
            st.info("No hay datos para la distribucion por decision. Revisa filtros.")
            return None

        chart = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("decision:N", title="Decisión"),
                y=alt.Y("count:Q", title="Número de películas"),
                color=decision_color("decision"),
                tooltip=["decision", "count"],
                opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.2)),
            )
            .add_params(dec_sel)
        )
        _chart(chart)
        chart_export = chart

    # 2) Rating IMDb por decisión
    elif view == "Rating IMDb por decisión":
        if not _requires_columns(df_g, ["imdb_rating", "decision"]):
            return None

        data = df_g.dropna(subset=["imdb_rating"])
        if data.empty:
            st.info("No hay ratings IMDb validos para mostrar. Revisa filtros.")
            return None

        bin_spec = alt.Bin(step=0.5, extent=[0, 10])
        base = (
            alt.Chart(data)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("imdb_rating:Q", bin=bin_spec, title="IMDb rating (0-10)"),
                y=alt.Y("count():Q", title="Numero de peliculas"),
                color=decision_color("decision"),
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
            .mark_rule(color="#666", strokeDash=[4, 4])
            .encode(x=alt.datum(imdb_ref))
        )
        chart = (base + ref_line).facet(
            column=alt.Column("decision:N", title="Decision")
        )
        _chart(chart.resolve_scale(y="independent"))
        chart_export = chart.resolve_scale(y="independent")

    # 3) Ratings IMDb vs RT
    elif view == "Ratings IMDb vs RT":
        if not _requires_columns(df_g, ["imdb_rating", "rt_score", "decision"]):
            return None

        data = df_g.dropna(subset=["imdb_rating", "rt_score"])
        if data.empty:
            st.info("No hay suficientes datos de IMDb y RT. Revisa filtros.")
            return None

        data = _mark_imdb_outliers(data, low=outlier_low, high=outlier_high)
        base = (
            alt.Chart(data)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("imdb_rating:Q", title="IMDb rating (0-10)"),
                y=alt.Y("rt_score:Q", title="RT score (%)"),
                color=decision_color("decision"),
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
                color=decision_color("decision"),
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
            .mark_rule(color="#666", strokeDash=[4, 4])
            .encode(x=alt.X("imdb_rating:Q"))
        )
        ref_rt = (
            alt.Chart(pd.DataFrame({"rt_score": [rt_ref]}))
            .mark_rule(color="#666", strokeDash=[4, 4])
            .encode(y=alt.Y("rt_score:Q"))
        )
        chart = base + outliers + ref_imdb + ref_rt
        _chart(chart)
        chart_export = chart

    # 4) Ratings IMDb vs Metacritic
    elif view == "Ratings IMDb vs Metacritic":
        if not _requires_columns(df_g, ["imdb_rating", "metacritic_score", "decision"]):
            return None

        data = df_g.dropna(subset=["imdb_rating", "metacritic_score"])
        if data.empty:
            st.info("No hay suficientes datos de IMDb y Metacritic. Revisa filtros.")
            return None

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
                color=decision_color("decision"),
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
                color=decision_color("decision"),
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
            .mark_rule(color="#666", strokeDash=[4, 4])
            .encode(x=alt.X("imdb_rating:Q"))
        )
        ref_meta = (
            alt.Chart(pd.DataFrame({"metacritic_score": [meta_ref]}))
            .mark_rule(color="#666", strokeDash=[4, 4])
            .encode(y=alt.Y("metacritic_score:Q"))
        )
        chart = base + outliers + ref_imdb + ref_meta
        _chart(chart)
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

        chart = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("decade_label:N", title="Década"),
                y=alt.Y("count:Q", title="Número de películas"),
                color=decision_color("decision"),
                tooltip=["decade_label", "decision", "count"],
                opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.2)),
            )
            .add_params(dec_sel)
        )
        _chart(chart)
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

        chart = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("genre:N", title="Género"),
                y=alt.Y("count:Q", title="Número de películas", stack="normalize"),
                color=decision_color("decision"),
                tooltip=["genre", "decision", "count"],
                opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.2)),
            )
            .add_params(dec_sel)
        )
        _chart(chart)
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

        chart_space = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("library:N", title="Biblioteca"),
                y=alt.Y("file_size_gb:Q", title="Tamano (GB)", stack="normalize"),
                color=decision_color("decision"),
                tooltip=[
                    "library",
                    "decision",
                    alt.Tooltip("file_size_gb:Q", title="Tamano (GB)", format=".2f"),
                ],
                opacity=alt.condition(lib_sel & dec_sel, alt.value(1), alt.value(0.2)),
            )
            .add_params(lib_sel, dec_sel)
        )
        _chart(chart_space)
        chart_export = chart_space

        total_space = float(agg["file_size_gb"].sum())
        space_delete = float(agg.loc[agg["decision"] == "DELETE", "file_size_gb"].sum())
        space_maybe = float(agg.loc[agg["decision"] == "MAYBE", "file_size_gb"].sum())

        st.markdown(
            f"- Espacio total: **{total_space:.2f} GB**\n"
            f"- DELETE: **{space_delete:.2f} GB**\n"
            f"- MAYBE: **{space_maybe:.2f} GB**"
        )

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
        st.caption(
            "Top medianas: "
            + ", ".join(
                [f"{row.library} ({row.imdb_median:.1f})" for row in top3.itertuples()]
            )
            + " | "
            + "Bottom medianas: "
            + ", ".join(
                [
                    f"{row.library} ({row.imdb_median:.1f})"
                    for row in bottom3.itertuples()
                ]
            )
        )
        if len(compare_libs) == 2:
            comp = stats[stats["library"].isin(compare_libs)].copy()
            if len(comp) == 2:
                delta = comp.iloc[0]["imdb_median"] - comp.iloc[1]["imdb_median"]
                st.caption(
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
                _chart(comp_chart)

        tooltip_common = [
            alt.Tooltip("library:N", title="Biblioteca"),
            alt.Tooltip("imdb_median:Q", title="Mediana IMDb", format=".1f"),
            alt.Tooltip("count:Q", title="Peliculas", format=".0f"),
        ]
        ref_line = (
            alt.Chart(pd.DataFrame({"imdb_rating": [imdb_ref]}))
            .mark_rule(color="#666", strokeDash=[4, 4])
            .encode(y=alt.Y("imdb_rating:Q"))
        )
        mean_line = (
            alt.Chart(pd.DataFrame({"imdb_rating": [global_mean]}))
            .mark_rule(color="#9bd4ff", strokeDash=[2, 2])
            .encode(y=alt.Y("imdb_rating:Q"))
        )
        color_scale = alt.Scale(scheme="blues")
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
            .mark_tick(color="#ffffff", thickness=2)
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
                color=alt.value("#9bd4ff"),
                tooltip=_movie_tooltips(data),
                opacity=alt.condition(lib_sel, alt.value(1), alt.value(0.1)),
            )
            .add_params(lib_sel)
        )
        chart = chart_strip + chart_box + chart_median + ref_line + mean_line
        _chart(chart)
        chart_export = chart

    # 9) Ranking de directores
    elif view == "Ranking de directores":
        agg = _director_decision_stats(df_g)
        if agg.empty:
            st.info("No se encontraron directores. Revisa filtros o datos de OMDb.")
            return None

        min_movies = min_movies_directors
        top_n = top_n_directors
        agg = agg[agg["count_total"] >= min_movies]

        if agg.empty:
            st.info("No hay directores que cumplan el minimo. Revisa filtros.")
            return None

        order = (
            agg.drop_duplicates("director_list")
            .sort_values("imdb_mean", ascending=False)["director_list"]
            .tolist()
        )
        top_directors = order[:top_n]
        agg_top = agg[agg["director_list"].isin(top_directors)]

        chart = (
            alt.Chart(agg_top)
            .mark_bar()
            .encode(
                x=alt.X("director_list:N", title="Director", sort=top_directors),
                y=alt.Y("count:Q", title="Peliculas"),
                color=decision_color("decision"),
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
        _chart(chart)
        chart_export = chart

    # 10) Palabras más frecuentes en títulos DELETE/MAYBE
    elif view == "Palabras más frecuentes en títulos DELETE/MAYBE":
        df_words = _word_counts(df_g, ("DELETE", "MAYBE"))

        if df_words.empty:
            st.info(
                "No hay datos suficientes para el analisis de palabras. Revisa filtros."
            )
            return None

        top_n = top_n_words
        df_top = df_words.head(top_n)

        chart = (
            alt.Chart(df_top)
            .mark_bar()
            .encode(
                x=alt.X("word:N", title="Palabra"),
                y=alt.Y("count:Q", title="Frecuencia"),
                color=decision_color("decision"),
                tooltip=["word", "decision", "count"],
                opacity=alt.condition(dec_sel, alt.value(1), alt.value(0.2)),
            )
            .add_params(dec_sel)
        )
        _chart(chart)
        chart_export = chart

    return chart_export


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
    st.write("### Gráficos")

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
                    ["KEEP", "MAYBE", "DELETE", "UNKNOWN"],
                )
                if decisions:
                    selected_decisions = st.multiselect(
                        "Decision",
                        decisions,
                        default=[],
                        key="charts_filter_decision",
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
                    "Top N directores por IMDb medio",
                    5,
                    50,
                    20,
                    key="charts_top_n_directors",
                )
            elif view == "Palabras más frecuentes en títulos DELETE/MAYBE":
                top_n_words = st.slider(
                    "Top N palabras", 5, 50, 20, key="charts_top_n_words"
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
        st.subheader("Dashboard")
        available_exec = [v for v in VIEW_OPTIONS if v != "Dashboard"]
        exec_views = get_dashboard_views(available_exec)
        exec_views = exec_views[:3]
        exec_cols = st.columns(len(exec_views))
        for exec_view, col in zip(exec_views, exec_cols, strict=False):
            with col:
                st.caption(exec_view)
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
