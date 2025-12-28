from __future__ import annotations

# =============================================================================
# frontend/dashboard.py
#
# Dashboard principal (Streamlit) 100% desacoplado del backend.
#
# PRINCIPIOS:
# - 0 imports de backend.*
# - Config autocontenida en frontend/config_front_*.py (lee .env.front)
# - Modo API opcional con fallback a disco
# - Import robusto de tabs (no depende de frontend.tabs.__init__)
# - Tab "Resumen": si no existe módulo UI, render inline (evita crash)
# =============================================================================

import importlib
import inspect
import sys
import warnings
from pathlib import Path
from typing import Any, Final, Iterable

import pandas as pd
import streamlit as st

# =============================================================================
# 1) Fix de import path (solo para Streamlit)
# =============================================================================

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =============================================================================
# 2) Imports FRONT-ONLY (sin backend)
# =============================================================================

from frontend.config_front_artifacts import (
    METADATA_FIX_PATH,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
)
from frontend.config_front_base import (
    DELETE_DRY_RUN,
    DELETE_REQUIRE_CONFIRM,
    FRONT_API_BASE_URL,
    FRONT_API_PAGE_SIZE,
    FRONT_API_TIMEOUT_S,
    FRONT_DEBUG,
    FRONT_USE_API,
)
from frontend.data_utils import add_derived_columns, format_count_size
from frontend.front_api_client import (
    ApiClientError,
    fetch_metadata_fix_df,
    fetch_report_all_df,
    fetch_report_filtered_df,
)
from frontend.front_stats import compute_global_imdb_mean_from_df
from frontend.summary import compute_summary

# =============================================================================
# 3) Opciones globales
# =============================================================================

warnings.filterwarnings("ignore", category=UserWarning)

TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "title",
    "file",
    "imdb_id",
    "imdbID",
    "path",
    "library",
)


class _CsvCacheEntry:
    def __init__(self, *, mtime_ns: int, data: pd.DataFrame) -> None:
        self.mtime_ns = mtime_ns
        self.data = data


_csv_cache: dict[str, _CsvCacheEntry] = {}


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


def _read_csv_or_raise(path: Path, *, label: str) -> pd.DataFrame:
    """
    Lee un CSV obligatorio (si no existe, error).
    Cachea por mtime para evitar re-lecturas innecesarias.
    """
    key = f"{label}:{path}"
    if not path.exists():
        raise FileNotFoundError(str(path))

    mtime = _mtime_ns(path)
    cached = _csv_cache.get(key)
    if cached is not None and cached.mtime_ns == mtime:
        return cached.data

    dtype_map: dict[str, Any] = {c: "string" for c in TEXT_COLUMNS}
    df = pd.read_csv(path, dtype=dtype_map, encoding="utf-8")

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    _csv_cache[key] = _CsvCacheEntry(mtime_ns=mtime, data=df)
    return df


def _read_csv_or_none(path: Path) -> pd.DataFrame | None:
    """
    Lee un CSV opcional (si no existe, devuelve None).
    """
    if not path.exists():
        return None
    try:
        return _read_csv_or_raise(path, label=path.name)
    except Exception:
        return None


def _debug_banner(*, df_all: pd.DataFrame, df_filtered: pd.DataFrame | None) -> None:
    """
    Instrumentación en UI solo si FRONT_DEBUG=True.
    """
    if not FRONT_DEBUG:
        return

    f_rows = 0 if df_filtered is None else len(df_filtered)
    st.caption(
        "DEBUG | "
        f"all={len(df_all)} filtered={f_rows} | "
        f"REPORT_ALL={REPORT_ALL_PATH} | REPORT_FILTERED={REPORT_FILTERED_PATH} | META_FIX={METADATA_FIX_PATH}"
    )


def _hide_streamlit_chrome() -> None:
    """
    Oculta elementos de “chrome” de Streamlit (cabecera, toolbar) y ajusta paddings.
    """
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        footer {
            display: none !important;
            visibility: hidden !important;
            height: 0px !important;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.0rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_modal_state() -> None:
    """
    Inicializa claves de session_state usadas por el modal.
    """
    st.session_state.setdefault("modal_open", False)
    st.session_state.setdefault("modal_row", None)


def _import_tabs_module(module_name: str) -> Any | None:
    """
    Importa un módulo de pestaña sin depender de frontend.tabs.__init__.

    Returns:
      - módulo si existe
      - None si no existe
    """
    try:
        return importlib.import_module(f"frontend.tabs.{module_name}")
    except ModuleNotFoundError:
        return None


def _call_candidates_render(module: Any, *, df_all: pd.DataFrame, df_filtered: pd.DataFrame | None) -> None:
    """
    Llama a candidates.render() de forma compatible con firmas:
      - render(df_all)
      - render(df_all, df_filtered)
    """
    render_fn = getattr(module, "render", None)
    if render_fn is None:
        st.error("El módulo candidates no expone render().")
        return

    sig = inspect.signature(render_fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(params) >= 2:
        render_fn(df_all, df_filtered)
    else:
        render_fn(df_all)


def _requires_columns(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.info(f"Faltan columna(s) requerida(s): {', '.join(missing)}.")
        return False
    return True


def _render_summary_tab_inline(df_all: pd.DataFrame) -> None:
    """
    Render inline para la pestaña "Resumen" cuando no existe frontend.tabs.summary.
    Evita crash y da un resumen útil sin inventar dependencias.
    """
    st.write("### Resumen")

    if df_all.empty:
        st.info("No hay datos para mostrar.")
        return

    # Tabla por decisión
    if _requires_columns(df_all, ["decision", "title"]):
        agg = (
            df_all.groupby("decision", dropna=False)["title"]
            .count()
            .reset_index()
            .rename(columns={"title": "count"})
            .sort_values("count", ascending=False, ignore_index=True)
        )
        st.write("**Conteo por decisión**")
        st.dataframe(agg, width="stretch", height=220)

    # Tabla por biblioteca (si existe)
    if _requires_columns(df_all, ["library", "title"]):
        agg_lib = (
            df_all.groupby("library", dropna=False)["title"]
            .count()
            .reset_index()
            .rename(columns={"title": "count"})
            .sort_values("count", ascending=False, ignore_index=True)
        )
        st.write("**Conteo por biblioteca**")
        st.dataframe(agg_lib, width="stretch", height=260)

    # Top deletes (si existe imdb_rating)
    if "decision" in df_all.columns and "imdb_rating" in df_all.columns and "title" in df_all.columns:
        df_del = df_all[df_all["decision"] == "DELETE"].copy()
        if not df_del.empty:
            df_del["imdb_rating"] = pd.to_numeric(df_del["imdb_rating"], errors="coerce")
            df_del = df_del.sort_values("imdb_rating", ascending=True, ignore_index=True).head(25)
            st.write("**Top 25 DELETE por peor IMDb (si disponible)**")
            show_cols = [c for c in ["title", "year", "library", "imdb_rating", "imdb_votes", "file_size_gb"] if c in df_del.columns]
            st.dataframe(df_del[show_cols], width="stretch", height=320)


# =============================================================================
# 4) Streamlit settings
# =============================================================================

st.set_page_config(page_title="Movies Cleaner — Dashboard", layout="wide")

# =============================================================================
# 5) UI base
# =============================================================================

_hide_streamlit_chrome()
_init_modal_state()

if not st.session_state.get("modal_open"):
    st.title("🎬 Movies Cleaner — Dashboard")

# =============================================================================
# 6) Modal (front-only)
# =============================================================================

from frontend.components import render_modal

render_modal()
if st.session_state.get("modal_open"):
    st.stop()

# =============================================================================
# 7) Carga de datos (API + fallback)
# =============================================================================

if FRONT_USE_API:
    try:
        df_all = fetch_report_all_df(
            base_url=FRONT_API_BASE_URL,
            timeout_s=FRONT_API_TIMEOUT_S,
            page_size=FRONT_API_PAGE_SIZE,
        )
        if df_all.empty:
            raise ApiClientError("API devolvió report_all vacío.")

        df_filtered = fetch_report_filtered_df(
            base_url=FRONT_API_BASE_URL,
            timeout_s=FRONT_API_TIMEOUT_S,
            page_size=FRONT_API_PAGE_SIZE,
        )

        if FRONT_DEBUG:
            st.caption("DEBUG | Datos cargados desde API (fallback a disco disponible).")
    except ApiClientError as exc:
        if FRONT_DEBUG:
            st.caption(f"DEBUG | Falló carga desde API ({exc}); usando ficheros locales.")
        df_all = _read_csv_or_raise(REPORT_ALL_PATH, label="report_all.csv")
        df_filtered = _read_csv_or_none(REPORT_FILTERED_PATH)
else:
    df_all = _read_csv_or_raise(REPORT_ALL_PATH, label="report_all.csv")
    df_filtered = _read_csv_or_none(REPORT_FILTERED_PATH)

df_all = add_derived_columns(df_all)

_debug_banner(df_all=df_all, df_filtered=df_filtered)

# =============================================================================
# 8) Resumen general (KPIs)
# =============================================================================

st.subheader("Resumen general")

summary = compute_summary(df_all)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("TOTAL", format_count_size(int(summary["total_count"]), summary.get("total_size_gb")))
col2.metric("KEEP", format_count_size(int(summary["keep_count"]), summary.get("keep_size_gb")))
col3.metric("DELETE", format_count_size(int(summary.get("delete_count", 0)), summary.get("delete_size_gb")))
col4.metric("MAYBE", format_count_size(int(summary.get("maybe_count", 0)), summary.get("maybe_size_gb")))

imdb_mean_df = compute_global_imdb_mean_from_df(df_all)
if imdb_mean_df is not None and not pd.isna(imdb_mean_df):
    col5.metric("IMDb medio (analizado)", f"{imdb_mean_df:.2f}")
else:
    col5.metric("IMDb medio (analizado)", "N/A")

st.markdown("---")

# =============================================================================
# 9) Pestañas (tabs)
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Candidatos", "Resumen", "Advanced", "Delete", "Charts", "Metadata"],
)

with tab1:
    candidates_mod = _import_tabs_module("candidates")
    if candidates_mod is None:
        st.error("No existe el módulo frontend.tabs.candidates")
    else:
        _call_candidates_render(candidates_mod, df_all=df_all, df_filtered=df_filtered)

with tab2:
    # Si existe una tab summary real, úsala. Si no, render inline.
    summary_tab_mod = _import_tabs_module("summary")
    render_fn = getattr(summary_tab_mod, "render", None) if summary_tab_mod is not None else None
    if callable(render_fn):
        render_fn(df_all)
    else:
        _render_summary_tab_inline(df_all)

with tab3:
    advanced_mod = _import_tabs_module("advanced")
    if advanced_mod is None or not callable(getattr(advanced_mod, "render", None)):
        st.error("No existe frontend.tabs.advanced.render(df_all)")
    else:
        advanced_mod.render(df_all)

with tab4:
    delete_mod = _import_tabs_module("delete")
    if delete_mod is None or not callable(getattr(delete_mod, "render", None)):
        st.error("No existe frontend.tabs.delete.render(df_filtered, ...)")
    else:
        delete_mod.render(df_filtered, DELETE_DRY_RUN, DELETE_REQUIRE_CONFIRM)

with tab5:
    charts_mod = _import_tabs_module("charts")
    if charts_mod is None or not callable(getattr(charts_mod, "render", None)):
        st.error("No existe frontend.tabs.charts.render(df_all)")
    else:
        charts_mod.render(df_all)

with tab6:
    metadata_mod = _import_tabs_module("metadata")
    if metadata_mod is None:
        st.error("No existe frontend.tabs.metadata")
    else:
        if FRONT_USE_API:
            try:
                df_meta_fix = fetch_metadata_fix_df(
                    base_url=FRONT_API_BASE_URL,
                    timeout_s=FRONT_API_TIMEOUT_S,
                    page_size=FRONT_API_PAGE_SIZE,
                )
                render_df = getattr(metadata_mod, "render_df", None)
                if callable(render_df):
                    render_df(df_meta_fix)
                else:
                    render = getattr(metadata_mod, "render", None)
                    if callable(render):
                        render(METADATA_FIX_PATH)
                    else:
                        st.error("metadata: no existe render_df(df) ni render(path)")
            except ApiClientError as exc:
                if FRONT_DEBUG:
                    st.caption(f"DEBUG | Falló metadata-fix desde API ({exc}); usando fichero local.")
                render = getattr(metadata_mod, "render", None)
                if callable(render):
                    render(METADATA_FIX_PATH)
                else:
                    st.error("metadata: no existe render(path)")
        else:
            render = getattr(metadata_mod, "render", None)
            if callable(render):
                render(METADATA_FIX_PATH)
            else:
                st.error("metadata: no existe render(path)")