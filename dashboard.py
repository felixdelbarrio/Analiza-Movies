from __future__ import annotations

# dashboard.py
#
# Dashboard principal (Streamlit) de Movies Cleaner.
#
# Responsabilidades:
# - Configurar Streamlit (layout, “chrome” oculto, estado modal).
# - Cargar reportes (CSV completo + CSV filtrado).
# - Calcular y renderizar resumen global.
# - Renderizar pestañas (All / Candidates / Advanced / Delete / Charts / Metadata).
# - Gestionar vista modal de detalle (overlay) que corta el flujo normal.
#
# Filosofía de logging:
# - Respetar SILENT_MODE: minimizar output.
# - Si SILENT_MODE=True y DEBUG_MODE=True: mostrar instrumentación útil (paths, shapes, thresholds).

import os
import warnings

import pandas as pd
import streamlit as st

from backend.config import (
    DEBUG_MODE,
    SILENT_MODE,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
    DELETE_DRY_RUN,
    DELETE_REQUIRE_CONFIRM,
    METADATA_FIX_PATH,
)
from backend.report_loader import load_reports
from backend.stats import (
    get_auto_delete_rating_threshold,
    get_auto_keep_rating_threshold,
    get_global_imdb_mean_info,
)
from frontend.summary import compute_summary
from frontend.data_utils import format_count_size


# ============================================================
# Helpers
# ============================================================


def _hide_streamlit_chrome() -> None:
    """Esconde cabecera de Streamlit y ajusta padding superior."""
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"],
        .stAppHeader,
        div[class*="stAppHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stCommandBar"] {
            display: none !important;
        }
        .block-container {
            padding-top: 0.5rem !important;
        }
        h1, h2, h3 {
            margin-top: 0.2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_modal_state() -> None:
    """Inicializa claves de estado global relacionadas con el modal."""
    st.session_state.setdefault("modal_open", False)
    st.session_state.setdefault("modal_row", None)


def _log_effective_thresholds_once() -> None:
    """
    Registra (1 vez por sesión) umbrales efectivos y media global para Bayes.

    - En SILENT_MODE: no logea nada (pero marca como 'ya logueado').
    - En DEBUG_MODE: además deja info adicional útil en el logger.
    """
    if st.session_state.get("thresholds_logged"):
        return

    st.session_state["thresholds_logged"] = True

    if SILENT_MODE:
        return

    # Lazy import del logger para evitar efectos colaterales innecesarios
    from backend import logger as _logger

    eff_keep = get_auto_keep_rating_threshold()
    eff_delete = get_auto_delete_rating_threshold()
    bayes_mean, bayes_source, bayes_n = get_global_imdb_mean_info()

    _logger.info(
        f"[DASH] thresholds: keep={eff_keep:.2f} delete={eff_delete:.2f} | "
        f"bayes_mean={bayes_mean:.3f} ({bayes_source}, n={bayes_n})"
    )

    if DEBUG_MODE:
        _logger.info(
            f"[DASH][DEBUG] REPORT_ALL_PATH={REPORT_ALL_PATH!r} "
            f"REPORT_FILTERED_PATH={REPORT_FILTERED_PATH!r} "
            f"METADATA_FIX_PATH={METADATA_FIX_PATH!r} "
            f"DELETE_DRY_RUN={DELETE_DRY_RUN} DELETE_REQUIRE_CONFIRM={DELETE_REQUIRE_CONFIRM}"
        )


def _debug_banner(*, df_all: pd.DataFrame, df_filtered: pd.DataFrame | None) -> None:
    """
    Muestra un pequeño bloque de depuración EN UI, solo si:
      - SILENT_MODE=True y DEBUG_MODE=True
    """
    if not (SILENT_MODE and DEBUG_MODE):
        return

    f_rows = 0 if df_filtered is None else len(df_filtered)
    st.caption(
        "DEBUG (silent): "
        f"all={len(df_all)} rows, filtered={f_rows} rows | "
        f"paths: all='{REPORT_ALL_PATH}', filtered='{REPORT_FILTERED_PATH}'"
    )


# ============================================================
# Warnings: silenciar SettingWithCopyWarning (st_aggrid/pandas)
# ============================================================

warnings.filterwarnings(
    "ignore",
    message=".*A value is trying to be set on a copy of a slice from a DataFrame.*",
    category=pd.errors.SettingWithCopyWarning,
)
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)


# ============================================================
# App setup
# ============================================================

st.set_page_config(page_title="Movies Cleaner", layout="wide")
_hide_streamlit_chrome()
_init_modal_state()

# Título (no mostramos si estamos en modo modal)
if not st.session_state.get("modal_open"):
    st.title("🎬 Movies Cleaner — Dashboard")

# Render modal (lazy import para evitar ciclos frontend.tabs -> backend -> frontend.components)
from frontend.components import render_modal  # frontend-only dependency

render_modal()
if st.session_state.get("modal_open"):
    st.stop()


# ============================================================
# Carga de datos
# ============================================================

if not os.path.exists(REPORT_ALL_PATH):
    st.error(
        "No se encuentra el CSV completo. "
        f"Ejecuta analiza-plex o analiza-dlna primero. (Ruta esperada: {REPORT_ALL_PATH})"
    )
    st.stop()

df_all, df_filtered = load_reports(REPORT_ALL_PATH, REPORT_FILTERED_PATH)

# Instrumentación UI si SILENT_MODE + DEBUG_MODE
_debug_banner(df_all=df_all, df_filtered=df_filtered)

# Log de umbrales efectivos (solo una vez, respetando SILENT_MODE)
_log_effective_thresholds_once()


# ============================================================
# Resumen general
# ============================================================

st.subheader("Resumen general")

summary = compute_summary(df_all)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Películas", format_count_size(summary["total_count"], summary["total_size_gb"]))
col2.metric("KEEP", format_count_size(summary["keep_count"], summary["keep_size_gb"]))
col3.metric("DELETE", format_count_size(summary.get("delete_count", 0), summary.get("delete_size_gb")))
col4.metric("MAYBE", format_count_size(summary.get("maybe_count", 0), summary.get("maybe_size_gb")))

imdb_mean_df = summary.get("imdb_mean_df")
imdb_mean_cache = summary.get("imdb_mean_cache")

if imdb_mean_df is not None and not pd.isna(imdb_mean_df):
    col5.metric("IMDb medio (analizado)", f"{imdb_mean_df:.2f}")
else:
    col5.metric("IMDb medio (analizado)", "N/A")

if imdb_mean_cache is not None and not pd.isna(imdb_mean_cache):
    st.caption(f"IMDb medio global (omdb_cache / bayes): **{imdb_mean_cache:.2f}**")

st.markdown("---")


# ============================================================
# Pestañas
# ============================================================
# Anti-circular-import:
# - Importamos las tabs de forma lazy dentro de cada bloque, para evitar que
#   una tab importe backend que a su vez importe frontend, etc.
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📚 Todas",
        "⚠️ Candidatas",
        "🔎 Búsqueda avanzada",
        "🧹 Borrado",
        "📊 Gráficos",
        "🧠 Metadata",
    ]
)

with tab1:
    from frontend.tabs import all_movies

    all_movies.render(df_all)

with tab2:
    from frontend.tabs import candidates

    candidates.render(df_all, df_filtered)

with tab3:
    from frontend.tabs import advanced

    advanced.render(df_all)

with tab4:
    # Importante: delete tab ya hace lazy import del backend.delete_logic internamente
    from frontend.tabs import delete

    delete.render(df_filtered, DELETE_DRY_RUN, DELETE_REQUIRE_CONFIRM)

with tab5:
    from frontend.tabs import charts

    charts.render(df_all)

with tab6:
    from frontend.tabs import metadata

    metadata.render(METADATA_FIX_PATH)