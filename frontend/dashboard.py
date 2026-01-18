from __future__ import annotations

# =============================================================================
# frontend/dashboard.py
#
# Dashboard principal (Streamlit) 100% desacoplado del backend.
#
# PRINCIPIOS:
# - 0 imports de backend.*
# - Config autocontenida en frontend/config_front_*.py (lee .env.front)
# - FRONT_MODE = "api" | "disk" (sin fallback)
# - Import robusto de tabs (no depende de frontend.tabs.__init__)
# - NO existe tab "Resumen" (como en el dashboard antiguo). El resumen es el KPI superior.
#
# Nota Pyright/Pylance:
# - Algunos stubs hacen que Pyright "demuestre" cosas demasiado fuertes y marque
#   ramas como unreachable. Para guard-clauses opcionales, usamos un cast a Any
#   para impedir que Pyright las elimine como "imposibles".
# =============================================================================

import importlib
import inspect
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Final, TypeVar, cast

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

from frontend.config_front_artifacts import (  # noqa: E402
    METADATA_FIX_PATH,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
)
from frontend.config_front_base import (  # noqa: E402
    DELETE_DRY_RUN,
    DELETE_REQUIRE_CONFIRM,
    FRONT_API_BASE_URL,
    FRONT_API_CACHE_TTL_S,
    FRONT_API_PAGE_SIZE,
    FRONT_API_TIMEOUT_S,
    FRONT_DEBUG,
    FRONT_MODE,
    FRONT_GRID_COLORIZE,
    save_front_grid_colorize,
)
from frontend.data_utils import add_derived_columns, format_count_size  # noqa: E402
from frontend.front_api_client import (  # noqa: E402
    ApiClientError,
    fetch_metadata_fix_df,
    fetch_report_all_df,
    fetch_report_filtered_df,
)
from frontend.summary import compute_summary  # noqa: E402
from frontend.components import render_modal  # noqa: E402
from frontend.config_front_charts import (  # noqa: E402
    get_dashboard_views,
    get_show_chart_thresholds,
    get_show_numeric_filters,
    save_dashboard_views,
    save_show_chart_thresholds,
    save_show_numeric_filters,
)
from frontend.tabs.charts import VIEW_OPTIONS  # noqa: E402

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


_F = TypeVar("_F", bound=Callable[..., Any])


def _cache_data_decorator(*, ttl_s: int | None = None) -> Callable[[_F], _F]:
    cache_fn = getattr(st, "cache_data", None)
    if callable(cache_fn):
        kwargs: dict[str, Any] = {"show_spinner": False}
        if ttl_s is not None:
            kwargs["ttl"] = ttl_s
        return cast(Callable[[_F], _F], cache_fn(**kwargs))
    cache_fn = getattr(st, "cache", None)
    if callable(cache_fn):
        return cast(Callable[[_F], _F], cache_fn)
    return cast(Callable[[_F], _F], lambda f: f)


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


@_cache_data_decorator()
def _read_report_all_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    df = pd.read_csv(path_str, dtype="string", encoding="utf-8")
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return add_derived_columns(df)


@_cache_data_decorator()
def _read_report_filtered_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    df = pd.read_csv(path_str, dtype="string", encoding="utf-8")
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def _read_csv_or_raise(path: Path, *, label: str) -> pd.DataFrame:
    """
    Lee un CSV obligatorio (si no existe, error).
    Cachea por mtime para evitar re-lecturas innecesarias.

    En modo DISK, si falla, se debe ver en UI y parar el flujo.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    mtime = _mtime_ns(path)
    if label == "report_all.csv":
        return cast(pd.DataFrame, _read_report_all_cached(str(path), mtime))
    return cast(pd.DataFrame, _read_report_filtered_cached(str(path), mtime))


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


_API_CACHE_TTL_S: int | None = (
    FRONT_API_CACHE_TTL_S if FRONT_API_CACHE_TTL_S > 0 else None
)


@_cache_data_decorator(ttl_s=_API_CACHE_TTL_S)
def _fetch_report_all_cached(
    base_url: str,
    timeout_s: float,
    page_size: int,
    query: str | None = None,
) -> pd.DataFrame:
    df = fetch_report_all_df(
        base_url=base_url,
        timeout_s=timeout_s,
        page_size=page_size,
        query=query,
    )
    return add_derived_columns(df)


@_cache_data_decorator(ttl_s=_API_CACHE_TTL_S)
def _fetch_report_filtered_cached(
    base_url: str,
    timeout_s: float,
    page_size: int,
    query: str | None = None,
) -> pd.DataFrame | None:
    return fetch_report_filtered_df(
        base_url=base_url,
        timeout_s=timeout_s,
        page_size=page_size,
        query=query,
    )


def _debug_banner(*, df_all: pd.DataFrame, df_filtered: pd.DataFrame | None) -> None:
    """
    Instrumentación en UI solo si FRONT_DEBUG=True.
    """
    if not FRONT_DEBUG:
        return

    f_rows = 0 if df_filtered is None else len(df_filtered)
    st.caption(
        "DEBUG | "
        f"mode={FRONT_MODE} all={len(df_all)} filtered={f_rows} | "
        f"REPORT_ALL={REPORT_ALL_PATH} | REPORT_FILTERED={REPORT_FILTERED_PATH} | META_FIX={METADATA_FIX_PATH}"
    )


def _bool_switch(label: str, *, key: str, value: bool, help: str | None = None) -> bool:
    toggle_fn = getattr(st, "toggle", None)
    if callable(toggle_fn):
        return bool(toggle_fn(label, value=value, key=key, help=help))
    return bool(st.checkbox(label, value=value, key=key, help=help))


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
    """
    try:
        return importlib.import_module(f"frontend.tabs.{module_name}")
    except ModuleNotFoundError:
        return None


def _call_candidates_render(
    module: Any, *, df_all: pd.DataFrame, df_filtered: pd.DataFrame | None
) -> None:
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
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(params) >= 2:
        render_fn(df_all, df_filtered)
    else:
        render_fn(df_all)


def _ui_error(msg: str) -> None:
    """
    Wrapper anti-stubs: llama a st.error() vía getattr/Any.
    """
    st_any: Any = st
    fn = getattr(st_any, "error", None)
    if callable(fn):
        try:
            fn(msg)
        except Exception:
            pass


# =============================================================================
# 4) Streamlit settings
# =============================================================================

st.set_page_config(page_title="Movies Cleaner — Dashboard", layout="wide")

# =============================================================================
# 5) UI base
# =============================================================================

_hide_streamlit_chrome()
_init_modal_state()

PENDING_DETAIL_KEY = "pending_open_detail_params"


def _get_query_params() -> dict[str, list[str]]:
    qp = getattr(st, "query_params", None)
    if qp is not None:
        out: dict[str, list[str]] = {}
        for k, v in qp.items():
            if isinstance(v, (list, tuple)):
                out[k] = [str(item) for item in v]
            else:
                out[k] = [str(v)]
        return out
    return cast(dict[str, list[str]], st.experimental_get_query_params())


def _clear_query_params() -> None:
    qp = getattr(st, "query_params", None)
    if qp is not None:
        qp.clear()
        return
    st.experimental_set_query_params()


def _first_param(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    if not values:
        return None
    value = str(values[0]).strip()
    if not value:
        return None
    try:
        from urllib.parse import unquote

        value = unquote(value)
    except Exception:
        pass
    return value or None


def _find_row_for_modal(
    df: pd.DataFrame, params: dict[str, list[str]]
) -> dict[str, Any] | None:
    imdb_id = _first_param(params, "imdb_id") or _first_param(params, "imdbID")
    if imdb_id:
        for col in ("imdb_id", "imdbID"):
            if col in df.columns:
                series = df[col].fillna("").astype(str)
                match = df[series.str.casefold() == imdb_id.casefold()]
                if not match.empty:
                    return dict(match.iloc[0])

    guid = _first_param(params, "guid")
    if guid and "guid" in df.columns:
        series = df["guid"].fillna("").astype(str)
        match = df[series.str.casefold() == guid.casefold()]
        if not match.empty:
            return dict(match.iloc[0])

    title = _first_param(params, "title")
    if title and "title" in df.columns:
        title_series = df["title"].fillna("").astype(str)
        mask = title_series.str.casefold() == title.casefold()
        year = _first_param(params, "year")
        if year and "year" in df.columns:
            try:
                year_num = float(year)
            except ValueError:
                year_num = None
            if year_num is not None:
                year_series = pd.to_numeric(df["year"], errors="coerce")
                mask = mask & (year_series == year_num)
        match = df[mask]
        if not match.empty:
            return dict(match.iloc[0])

    return None


params = _get_query_params()
if params.get("open_detail"):
    st.session_state[PENDING_DETAIL_KEY] = params
    _clear_query_params()

# =============================================================================
# 7) Carga de datos (SIN FALLBACK): FRONT_MODE = api | disk
# =============================================================================

df_all: pd.DataFrame | None = None
df_filtered: pd.DataFrame | None = None

if FRONT_MODE == "api":
    try:
        df_all = _fetch_report_all_cached(
            base_url=FRONT_API_BASE_URL,
            timeout_s=FRONT_API_TIMEOUT_S,
            page_size=FRONT_API_PAGE_SIZE,
        )
        if df_all is None or df_all.empty:
            raise ApiClientError("API devolvió report_all vacío.")

        df_filtered = _fetch_report_filtered_cached(
            base_url=FRONT_API_BASE_URL,
            timeout_s=FRONT_API_TIMEOUT_S,
            page_size=FRONT_API_PAGE_SIZE,
        )

        if FRONT_DEBUG:
            st.caption("DEBUG | Datos cargados exclusivamente desde API.")
    except ApiClientError as exc:
        _ui_error(f"Error cargando datos desde API (FRONT_MODE=api): {exc}")
        st.stop()

elif FRONT_MODE == "disk":
    try:
        df_all = _read_csv_or_raise(REPORT_ALL_PATH, label="report_all.csv")
        df_filtered = _read_csv_or_none(REPORT_FILTERED_PATH)

        if FRONT_DEBUG:
            st.caption("DEBUG | Datos cargados exclusivamente desde disco.")
    except Exception as exc:
        _ui_error(f"Error cargando datos desde disco (FRONT_MODE=disk): {exc!r}")
        st.stop()

else:
    _ui_error(f"FRONT_MODE desconocido: {FRONT_MODE!r}")
    st.stop()

# Guard clause (evita "unreachable" usando Any para que Pyright no lo colapse)
df_all_any: Any = df_all
if df_all_any is None:
    _ui_error("No se pudo cargar report_all (df_all=None).")
    raise RuntimeError("df_all is None")
df_all = cast(pd.DataFrame, df_all_any)

_debug_banner(df_all=df_all, df_filtered=df_filtered)

pending_params = st.session_state.pop(PENDING_DETAIL_KEY, None)
if pending_params:
    modal_row = _find_row_for_modal(df_all, pending_params)
    if modal_row is None:
        modal_row = st.session_state.get("detail_row_last")
    if modal_row is not None:
        st.session_state["modal_row"] = modal_row
        st.session_state["modal_open"] = True

render_modal()
if st.session_state.get("modal_open"):
    st.stop()


def _render_config_container(body: Callable[[], None]) -> bool:
    dialog: Any = getattr(st, "dialog", None)
    if callable(dialog):
        dialog_fn = cast(
            Callable[[str], Callable[[Callable[[], None]], Callable[[], None]]],
            dialog,
        )

        @dialog_fn("Configuracion")
        def _dlg() -> None:
            body()

        _dlg()
        return True
    with st.expander("Configuracion", expanded=True):
        body()
    return False


if "grid_colorize_rows" not in st.session_state:
    st.session_state["grid_colorize_rows"] = FRONT_GRID_COLORIZE

CONFIG_COLORIZE_KEY = "config_colorize"
CONFIG_SHOW_NUMERIC_KEY = "config_show_numeric_filters"
CONFIG_SHOW_THRESHOLDS_KEY = "config_show_chart_thresholds"
CONFIG_VIEWS_KEY = "config_dashboard_views"
CONFIG_EDITING_KEY = "config_editing"
CONFIG_RESET_KEY = "config_reset_pending"
CONFIG_DEFAULTS_KEY = "config_defaults"

if st.session_state.get("config_open"):
    available_dashboard = [v for v in VIEW_OPTIONS if v != "Dashboard"]

    def _load_config_defaults() -> dict[str, object]:
        return {
            CONFIG_COLORIZE_KEY: bool(
                st.session_state.get("grid_colorize_rows", FRONT_GRID_COLORIZE)
            ),
            CONFIG_SHOW_NUMERIC_KEY: get_show_numeric_filters(),
            CONFIG_SHOW_THRESHOLDS_KEY: get_show_chart_thresholds(),
            CONFIG_VIEWS_KEY: get_dashboard_views(available_dashboard),
        }

    if st.session_state.get(CONFIG_RESET_KEY, False) or not st.session_state.get(
        CONFIG_EDITING_KEY, False
    ):
        defaults = _load_config_defaults()
        st.session_state[CONFIG_DEFAULTS_KEY] = defaults
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)
        st.session_state[CONFIG_EDITING_KEY] = True
        st.session_state[CONFIG_RESET_KEY] = False

    def _config_body() -> None:
        st.subheader("Preferencias")
        colorize = st.checkbox(
            "Señalética de color en tablas",
            key=CONFIG_COLORIZE_KEY,
        )
        show_numeric_filters = st.checkbox(
            "Mostrar filtros numéricos",
            key=CONFIG_SHOW_NUMERIC_KEY,
        )
        show_chart_thresholds = st.checkbox(
            "Mostrar umbrales en gráficos",
            key=CONFIG_SHOW_THRESHOLDS_KEY,
        )
        exec_views = st.multiselect(
            "Graficos dashboard (max 3)",
            available_dashboard,
            key=CONFIG_VIEWS_KEY,
        )
        save_cols = st.columns(2)
        with save_cols[0]:
            if st.button("Guardar"):
                if len(exec_views) > 3:
                    st.error("Selecciona un maximo de 3 graficos.")
                    return
                save_front_grid_colorize(colorize)
                save_dashboard_views(exec_views)
                save_show_numeric_filters(show_numeric_filters)
                save_show_chart_thresholds(show_chart_thresholds)
                st.session_state["grid_colorize_rows"] = colorize
                st.session_state["config_open"] = False
                st.session_state[CONFIG_EDITING_KEY] = False
                st.session_state[CONFIG_RESET_KEY] = False
                st.session_state.pop(CONFIG_DEFAULTS_KEY, None)
                st.success("Configuracion guardada.")
                st.rerun()
        with save_cols[1]:
            if st.button("Cancelar"):
                st.session_state["config_open"] = False
                st.session_state[CONFIG_EDITING_KEY] = False
                st.session_state[CONFIG_RESET_KEY] = True
                st.session_state.pop(CONFIG_DEFAULTS_KEY, None)

    used_dialog = _render_config_container(_config_body)
    if used_dialog and st.session_state.get("config_open"):
        st.session_state["config_open"] = False
        st.session_state[CONFIG_EDITING_KEY] = False
        st.session_state[CONFIG_RESET_KEY] = True
        st.session_state.pop(CONFIG_DEFAULTS_KEY, None)

# =============================================================================
# 8) Resumen general (KPIs)
# =============================================================================

with st.container():
    st.markdown(
        """
<div id="summary-card-anchor"></div>
<style>
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) {
  background: linear-gradient(180deg, #121826 0%, #0f141d 100%);
  border: 1px solid #1f2532;
  border-radius: 18px;
  padding: 24px 26px 22px;
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) h1 {
  margin: 0;
  font-size: 2.1rem;
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) h3 {
  margin: 0 0 0.9rem 0;
}
div[data-testid="stVerticalBlock"]:has(#config-top-anchor) {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  height: 100%;
  padding-top: 0.25rem;
}
div[data-testid="stVerticalBlock"]:has(#config-top-anchor) button[data-testid="stBaseButton-secondary"],
div[data-testid="stVerticalBlock"]:has(#config-top-anchor) button[data-testid="baseButton-secondary"] {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  min-width: 0 !important;
  height: auto !important;
}
div[data-testid="stVerticalBlock"]:has(#config-top-anchor) button[data-testid="stBaseButton-secondary"] span,
div[data-testid="stVerticalBlock"]:has(#config-top-anchor) button[data-testid="baseButton-secondary"] span {
  color: inherit !important;
  font-size: 0.95rem;
  font-weight: 700;
  padding: 0.35rem 0.75rem;
  border-radius: 0.6rem;
  background: #171b24;
  border: 1px solid #262c38;
}
div[data-testid="stVerticalBlock"]:has(#config-top-anchor) button[data-testid="stBaseButton-secondary"]:hover span,
div[data-testid="stVerticalBlock"]:has(#config-top-anchor) button[data-testid="baseButton-secondary"]:hover span {
  color: #f3f4f6 !important;
  background: #202635;
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) [data-testid="stMetric"] {
  background: #111722;
  border: 1px solid #202737;
  border-radius: 12px;
  padding: 0.65rem 0.75rem;
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) [data-testid="stMetricLabel"] {
  color: #9ca3af;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  font-size: 0.72rem;
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) [data-testid="stMetricValue"] {
  font-size: 1.6rem;
}
</style>
""",
        unsafe_allow_html=True,
    )
    top_cols = st.columns([10, 2], gap="small")
    with top_cols[0]:
        st.title("🎬 Movies Cleaner — Dashboard")
    with top_cols[1]:
        st.markdown('<div id="config-top-anchor"></div>', unsafe_allow_html=True)
        if st.button("Configuracion", key="config_link_top", type="secondary"):
            st.session_state["config_open"] = True
    st.subheader("Resumen general")

    summary = compute_summary(df_all)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Películas", format_count_size(summary["total_count"], summary["total_size_gb"])
    )
    col2.metric(
        "KEEP", format_count_size(summary["keep_count"], summary["keep_size_gb"])
    )
    col3.metric(
        "DELETE", format_count_size(summary["delete_count"], summary["delete_size_gb"])
    )
    col4.metric(
        "MAYBE", format_count_size(summary["maybe_count"], summary["maybe_size_gb"])
    )

    imdb_mean_df = summary.get("imdb_mean_df")
    if imdb_mean_df is not None and not pd.isna(imdb_mean_df):
        col5.metric("IMDb medio (analizado)", f"{imdb_mean_df:.2f}")
    else:
        col5.metric("IMDb medio (analizado)", "N/A")

st.markdown('<div style="height: 0.8rem;"></div>', unsafe_allow_html=True)

# Consistencia visual: tags de multiselect con fondo neutro (evita rojo fijo).
st.markdown(
    """
<style>
div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
  background-color: #2b2f36 !important;
  border: 1px solid #3a404a !important;
}
div[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
  color: #e5e7eb !important;
}
.stDataFrame div[data-testid="stDataFrameToolbar"],
div[data-testid="stDataFrameToolbar"] {
  opacity: 1 !important;
  visibility: visible !important;
  pointer-events: auto !important;
}
div[data-testid="stDataFrameToolbar"] button,
div[data-testid="stDataFrameToolbar"] [data-testid="stDataFrameToolbarButton"] {
  opacity: 1 !important;
}
.ag-theme-alpine .ag-cell,
.ag-theme-alpine-dark .ag-cell {
  display: flex;
  align-items: center;
}
.ag-theme-alpine .ag-checkbox-input-wrapper,
.ag-theme-alpine-dark .ag-checkbox-input-wrapper {
  margin: 0 auto;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# 9) Pestañas (tabs)
# =============================================================================

st.markdown(
    """
<div id="tabs-row-anchor"></div>
<style>
div[data-testid="stVerticalBlock"]:has(#tabs-row-anchor) {
  margin-top: 0.2rem;
  background: #0f141d;
  border: 1px solid #1f2532;
  border-radius: 14px;
  padding: 0.35rem 0.6rem 0.1rem;
}
div[data-testid="stVerticalBlock"]:has(#tabs-row-anchor) div[data-testid="stTabs"] {
  padding-top: 0.2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📚 Todas",
        "📊 Gráficos",
        "⚠️ Candidatas",
        "🔁 Duplicadas",
        "🧠 Metadata",
    ]
)

with tab1:
    all_movies_mod = _import_tabs_module("all_movies")
    render_fn = (
        getattr(all_movies_mod, "render", None) if all_movies_mod is not None else None
    )
    if not callable(render_fn):
        _ui_error("No existe frontend.tabs.all_movies.render(df_all)")
    else:
        render_fn(df_all)

with tab2:
    charts_mod = _import_tabs_module("charts")
    render_fn = getattr(charts_mod, "render", None) if charts_mod is not None else None
    if not callable(render_fn):
        _ui_error("No existe frontend.tabs.charts.render(df_all)")
    else:
        render_fn(df_all)

with tab3:
    delete_mod = _import_tabs_module("delete")
    render_fn = getattr(delete_mod, "render", None) if delete_mod is not None else None
    if not callable(render_fn):
        _ui_error("No existe frontend.tabs.delete.render(df_filtered, ...)")
    else:
        render_fn(df_filtered, DELETE_DRY_RUN, DELETE_REQUIRE_CONFIRM)

with tab4:
    candidates_mod = _import_tabs_module("candidates")
    if candidates_mod is None:
        _ui_error("No existe el módulo frontend.tabs.candidates")
    else:
        _call_candidates_render(candidates_mod, df_all=df_all, df_filtered=df_filtered)

with tab5:
    metadata_mod = _import_tabs_module("metadata")
    if metadata_mod is None:
        _ui_error("No existe frontend.tabs.metadata")
    else:
        if FRONT_MODE == "api":
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
                        _ui_error("metadata: no existe render_df(df) ni render(path)")
            except ApiClientError as exc:
                _ui_error(f"Error cargando metadata desde API (FRONT_MODE=api): {exc}")
        else:
            render = getattr(metadata_mod, "render", None)
            if callable(render):
                render(METADATA_FIX_PATH)
            else:
                _ui_error("metadata: no existe render(path)")
