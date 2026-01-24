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
"""

from __future__ import annotations

from typing import Any, Final, cast

import altair as alt
import pandas as pd
import streamlit as st

from frontend.components import render_decision_chip_styles
from frontend.config_front_charts import (
    get_dashboard_views,
    get_show_chart_thresholds,
    get_show_numeric_filters,
)
from frontend.tabs.charts_data import _mark_imdb_outliers
from frontend.tabs.charts_shared import (
    AltChart,
    AltSelection,
    DECISION_ORDER,
    IMDB_OUTLIER_HIGH,
    IMDB_OUTLIER_LOW,
    IMDB_REFERENCE,
    METACRITIC_REFERENCE,
    RT_REFERENCE,
    _chart_png_bytes,
    _chart_svg_bytes,
    _movie_tooltips,
    _ordered_options,
)
from frontend.tabs.charts_views import (
    boxplot_by_library,
    decade_distribution,
    decision_distribution,
    director_ranking,
    genre_distribution,
    imdb_by_decision,
    imdb_vs_metacritic,
    imdb_vs_rt,
    space_by_library,
    word_ranking,
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

_REEXPORTED: tuple[object, ...] = (_mark_imdb_outliers, _movie_tooltips)


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
    if view == "Distribución por decisión":
        return decision_distribution.render(
            df_g, dec_sel=dec_sel, show_insights=show_insights
        )
    if view == "Rating IMDb por decisión":
        return imdb_by_decision.render(
            df_g, dec_sel=dec_sel, imdb_ref=imdb_ref, show_insights=show_insights
        )
    if view == "Ratings IMDb vs RT":
        return imdb_vs_rt.render(
            df_g,
            dec_sel=dec_sel,
            imdb_ref=imdb_ref,
            rt_ref=rt_ref,
            outlier_low=outlier_low,
            outlier_high=outlier_high,
            show_insights=show_insights,
        )
    if view == "Ratings IMDb vs Metacritic":
        return imdb_vs_metacritic.render(
            df_g,
            dec_sel=dec_sel,
            imdb_ref=imdb_ref,
            meta_ref=meta_ref,
            outlier_low=outlier_low,
            outlier_high=outlier_high,
            show_insights=show_insights,
        )
    if view == "Distribución por década":
        return decade_distribution.render(
            df_g, dec_sel=dec_sel, show_insights=show_insights
        )
    if view == "Distribución por género (OMDb)":
        return genre_distribution.render(
            df_g,
            dec_sel=dec_sel,
            top_n_genres=top_n_genres,
            show_insights=show_insights,
        )
    if view == "Espacio ocupado por biblioteca/decisión":
        return space_by_library.render(
            df_g,
            lib_sel=lib_sel,
            dec_sel=dec_sel,
            show_insights=show_insights,
        )
    if view == "Boxplot IMDb por biblioteca":
        return boxplot_by_library.render(
            df_g,
            lib_sel=lib_sel,
            imdb_ref=imdb_ref,
            top_n_libs=top_n_libs,
            min_n_libs=min_n_libs,
            boxplot_horizontal=boxplot_horizontal,
            compare_libs=compare_libs,
            show_insights=show_insights,
        )
    if view == "Ranking de directores":
        return director_ranking.render(
            df_g,
            dec_sel=dec_sel,
            min_movies_directors=min_movies_directors,
            top_n_directors=top_n_directors,
            show_insights=show_insights,
        )
    if view == "Palabras más frecuentes en títulos DELETE/MAYBE":
        return word_ranking.render(
            df_g,
            dec_sel=dec_sel,
            top_n_words=top_n_words,
            show_insights=show_insights,
        )
    return None


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


def _apply_dashboard_card_styles() -> None:
    st.markdown(
        """
<style>
div[data-testid="stVerticalBlock"]:has(.mc-chart-card-anchor) {
  background: var(--mc-card-bg);
  border: 1px solid var(--mc-card-border);
  border-radius: 18px;
  padding: 0.85rem 0.95rem 0.95rem;
  box-shadow: var(--mc-card-shadow);
  position: relative;
  overflow: hidden;
  margin-bottom: 0.85rem;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-card-anchor)::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(130deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0) 55%);
  pointer-events: none;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-card-anchor) > div {
  position: relative;
  z-index: 1;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-card-anchor) .mc-chart-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.35rem;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-card-anchor) .mc-chart-card-title {
  display: flex;
  align-items: center;
  gap: 0.55rem;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-card-anchor) .mc-chart-card-badge {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.58rem;
  color: var(--mc-text-3);
  padding: 0.2rem 0.5rem;
  border-radius: 999px;
  background: var(--mc-tag-bg);
  border: 1px solid var(--mc-tag-border);
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-card-anchor) .mc-chart-card-heading {
  margin: 0;
  font-size: 1.05rem;
  color: var(--mc-text-1);
  font-family: var(--mc-font-display);
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-card-anchor) .mc-chart-card-index {
  font-size: 0.7rem;
  color: var(--mc-text-3);
  border: 1px solid var(--mc-panel-border);
  border-radius: 999px;
  padding: 0.1rem 0.45rem;
  background: var(--mc-panel-bg);
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-card-anchor) [data-testid="stCaption"] {
  color: var(--mc-text-3);
  margin-bottom: 0.35rem;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-card-anchor) [data-testid="stVegaLiteChart"] {
  margin-top: 0.8rem;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-downloads-anchor) {
  background: var(--mc-panel-bg);
  border: 1px solid var(--mc-panel-border);
  border-radius: 14px;
  padding: 0.6rem 0.75rem 0.75rem;
  margin-top: 0.9rem;
  text-align: left !important;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-downloads-anchor) [data-testid="stMarkdownContainer"],
div[data-testid="stVerticalBlock"]:has(.mc-chart-downloads-anchor) ~ div[data-testid="stVerticalBlock"] [data-testid="stMarkdownContainer"] {
  text-align: left !important;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-downloads-anchor) .mc-chart-downloads-title {
  margin: 0 0 0.55rem 0;
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--mc-text-3);
  text-align: left !important;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-downloads-anchor) div[data-testid="stHorizontalBlock"],
div[data-testid="stVerticalBlock"]:has(.mc-chart-downloads-anchor) ~ div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] {
  display: flex;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  justify-content: flex-start !important;
  align-items: center;
  gap: 0.6rem !important;
  overflow-x: auto;
  padding-bottom: 0.25rem;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-downloads-anchor) div[data-testid="column"],
div[data-testid="stVerticalBlock"]:has(.mc-chart-downloads-anchor) ~ div[data-testid="stVerticalBlock"] div[data-testid="column"] {
  flex: 0 0 auto !important;
  width: max-content !important;
  min-width: 0 !important;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-downloads-anchor) div[data-testid="column"] > div,
div[data-testid="stVerticalBlock"]:has(.mc-chart-downloads-anchor) ~ div[data-testid="stVerticalBlock"] div[data-testid="column"] > div {
  width: max-content !important;
}
div[data-testid="stVerticalBlock"]:has(.mc-chart-downloads-anchor) [data-testid="stDownloadButton"] button {
  width: auto !important;
  white-space: nowrap !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_dashboard_card_header(title: str, *, badge: str, index: int | None) -> None:
    index_html = (
        f'<span class="mc-chart-card-index">{index:02d}</span>' if index else ""
    )
    st.markdown(
        f"""
<div class="mc-chart-card-header">
  <div class="mc-chart-card-title">
    <span class="mc-chart-card-badge">{badge}</span>
    <h3 class="mc-chart-card-heading">{title}</h3>
  </div>
  {index_html}
</div>
""",
        unsafe_allow_html=True,
    )


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

    render_kwargs = dict(
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

    if show_exec:
        _apply_dashboard_card_styles()
        available_exec = [v for v in VIEW_OPTIONS if v != "Dashboard"]
        exec_views = get_dashboard_views(available_exec)
        exec_views = exec_views[:3]
        exec_cols = st.columns(len(exec_views))
        for index, (exec_view, col) in enumerate(
            zip(exec_views, exec_cols, strict=False), start=1
        ):
            with col:
                with st.container():
                    st.markdown(
                        '<div class="mc-chart-card-anchor"></div>',
                        unsafe_allow_html=True,
                    )
                    _render_dashboard_card_header(
                        exec_view, badge="Dashboard", index=index
                    )
                    _render_view(exec_view, df_g, show_insights=True, **render_kwargs)
        insights = _auto_insights(df_g)
        if insights:
            with st.container():
                st.markdown(
                    '<div class="mc-chart-card-anchor"></div>',
                    unsafe_allow_html=True,
                )
                _render_dashboard_card_header(
                    "Insights automaticos", badge="Insights", index=None
                )
                st.markdown("\n".join(f"- {item}" for item in insights))
        return

    _apply_dashboard_card_styles()
    with st.container():
        st.markdown(
            '<div class="mc-chart-card-anchor"></div>',
            unsafe_allow_html=True,
        )
        _render_dashboard_card_header(view, badge="Vista", index=None)
        chart_export = _render_view(
            view,
            df_g,
            **render_kwargs,
        )

        if chart_export is not None:
            download_items: list[tuple[str, bytes, str, str]] = []
            if export_csv:
                download_items.append(
                    (
                        "Descargar CSV filtrado",
                        df_g.to_csv(index=False).encode("utf-8"),
                        "charts_filtered.csv",
                        "text/csv",
                    )
                )
            svg_bytes = _chart_svg_bytes(chart_export)
            if svg_bytes:
                download_items.append(
                    (
                        "Descargar SVG del grafico",
                        svg_bytes,
                        "chart.svg",
                        "image/svg+xml",
                    )
                )
            png_bytes = _chart_png_bytes(chart_export)
            if png_bytes:
                download_items.append(
                    (
                        "Descargar PNG del grafico",
                        png_bytes,
                        "chart.png",
                        "image/png",
                    )
                )
            if download_items:
                with st.container():
                    st.markdown(
                        '<div class="mc-chart-downloads-anchor"></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<p class="mc-chart-downloads-title">Descargas</p>',
                        unsafe_allow_html=True,
                    )
                    download_cols = st.columns(len(download_items))
                    for (label, data, filename, mime), col in zip(
                        download_items, download_cols, strict=False
                    ):
                        with col:
                            st.download_button(
                                label,
                                data,
                                file_name=filename,
                                mime=mime,
                            )
                    if not svg_bytes and not png_bytes:
                        st.caption(
                            "Export PNG/SVG no disponible: requiere dependencias extra de Altair."
                        )
