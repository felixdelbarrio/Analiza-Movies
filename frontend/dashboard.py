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
from typing import Any, Final, cast

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
    FRONT_API_PAGE_SIZE,
    FRONT_API_TIMEOUT_S,
    FRONT_DEBUG,
    FRONT_MODE,
)
from frontend.data_utils import add_derived_columns, format_count_size  # noqa: E402
from frontend.front_api_client import (  # noqa: E402
    ApiClientError,
    fetch_metadata_fix_df,
    fetch_report_all_df,
    fetch_report_filtered_df,
)
from frontend.front_stats import compute_global_imdb_mean_from_df  # noqa: E402
from frontend.summary import compute_summary  # noqa: E402
from frontend.components import render_modal  # noqa: E402

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

    En modo DISK, si falla, se debe ver en UI y parar el flujo.
    """
    key = f"{label}:{path}"
    if not path.exists():
        raise FileNotFoundError(str(path))

    mtime = _mtime_ns(path)
    cached = _csv_cache.get(key)
    if cached is not None and cached.mtime_ns == mtime:
        return cached.data

    # Nota: en runtime pandas acepta dtype=dict por columna, pero los stubs que tienes
    # no lo permiten. Como todas las columnas objetivo son "string", usamos dtype global.
    df = pd.read_csv(path, dtype="string", encoding="utf-8")

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

if not st.session_state.get("modal_open"):
    st.title("🎬 Movies Cleaner — Dashboard")

# =============================================================================
# 6) Modal (front-only)
# =============================================================================

render_modal()
if st.session_state.get("modal_open"):
    st.stop()

# =============================================================================
# 7) Carga de datos (SIN FALLBACK): FRONT_MODE = api | disk
# =============================================================================

df_all: pd.DataFrame | None = None
df_filtered: pd.DataFrame | None = None

if FRONT_MODE == "api":
    try:
        df_all = fetch_report_all_df(
            base_url=FRONT_API_BASE_URL,
            timeout_s=FRONT_API_TIMEOUT_S,
            page_size=FRONT_API_PAGE_SIZE,
        )
        if df_all is None or df_all.empty:
            raise ApiClientError("API devolvió report_all vacío.")

        df_filtered = fetch_report_filtered_df(
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

df_all = add_derived_columns(df_all)

_debug_banner(df_all=df_all, df_filtered=df_filtered)

# =============================================================================
# 8) Resumen general (KPIs)
# =============================================================================

st.subheader("Resumen general")

summary = compute_summary(df_all)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Películas", format_count_size(summary["total_count"], summary["total_size_gb"]))
col2.metric("KEEP", format_count_size(summary["keep_count"], summary["keep_size_gb"]))
col3.metric("DELETE", format_count_size(summary["delete_count"], summary["delete_size_gb"]))
col4.metric("MAYBE", format_count_size(summary["maybe_count"], summary["maybe_size_gb"]))

imdb_mean_df = compute_global_imdb_mean_from_df(df_all)
if imdb_mean_df is not None and not pd.isna(imdb_mean_df):
    col5.metric("IMDb medio (analizado)", f"{imdb_mean_df:.2f}")
else:
    col5.metric("IMDb medio (analizado)", "N/A")

_bool_switch(
    "Señalética de color en tablas",
    key="grid_colorize_rows",
    value=bool(st.session_state.get("grid_colorize_rows", True)),
    help="Activa colores por decisión (DELETE/KEEP/MAYBE/UNKNOWN) en el texto de cada fila.",
)

st.markdown("---")

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📚 Todas",
        "📊 Gráficos",
        "🔎 Búsqueda avanzada",
        "⚠️ Candidatas",
        "🔁 Duplicadas",
        "🧠 Metadata",
    ]
)

with tab1:
    all_movies_mod = _import_tabs_module("all_movies")
    render_fn = getattr(all_movies_mod, "render", None) if all_movies_mod is not None else None
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
    advanced_mod = _import_tabs_module("advanced")
    render_fn = getattr(advanced_mod, "render", None) if advanced_mod is not None else None
    if not callable(render_fn):
        _ui_error("No existe frontend.tabs.advanced.render(df_all)")
    else:
        render_fn(df_all)

with tab4:
    delete_mod = _import_tabs_module("delete")
    render_fn = getattr(delete_mod, "render", None) if delete_mod is not None else None
    if not callable(render_fn):
        _ui_error("No existe frontend.tabs.delete.render(df_filtered, ...)")
    else:
        render_fn(df_filtered, DELETE_DRY_RUN, DELETE_REQUIRE_CONFIRM)

with tab5:
    candidates_mod = _import_tabs_module("candidates")
    if candidates_mod is None:
        _ui_error("No existe el módulo frontend.tabs.candidates")
    else:
        _call_candidates_render(candidates_mod, df_all=df_all, df_filtered=df_filtered)

with tab6:
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
