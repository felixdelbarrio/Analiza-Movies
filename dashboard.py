from __future__ import annotations

import os
import warnings

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from backend import logger as _logger
from backend.report_loader import load_reports
from backend.summary import compute_summary
from backend.stats import (
    get_auto_keep_rating_threshold,
    get_auto_delete_rating_threshold,
    get_global_imdb_mean_info,
)
from backend.config import (
    IMDB_KEEP_MIN_VOTES,
    IMDB_KEEP_MIN_RATING,
    IMDB_DELETE_MAX_RATING,
    IMDB_KEEP_MIN_RATING_WITH_RT,
    BAYES_GLOBAL_MEAN_DEFAULT,
    BAYES_DELETE_MAX_SCORE,
    BAYES_MIN_TITLES_FOR_GLOBAL_MEAN,
    AUTO_KEEP_RATING_PERCENTILE,
    AUTO_DELETE_RATING_PERCENTILE,
    RATING_MIN_TITLES_FOR_AUTO,
    IMDB_RATING_LOW_THRESHOLD,
    RT_RATING_LOW_THRESHOLD,
    SILENT_MODE,
)
from frontend.components import render_modal
from frontend.data_utils import format_count_size
from frontend.tabs import advanced, all_movies, candidates, charts, delete, metadata


# ============================================================
# Helpers
# ============================================================


def _env_bool(name: str, default: bool) -> bool:
    """Lee un booleano de ENV de forma robusta."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() == "true"


def _init_modal_state() -> None:
    """Inicializa claves de estado global relacionadas con el modal."""
    if "modal_open" not in st.session_state:
        st.session_state["modal_open"] = False
    if "modal_row" not in st.session_state:
        st.session_state["modal_row"] = None


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


def _log_effective_thresholds_once() -> None:
    """
    Escribe en la TERMINAL / logger los umbrales efectivos de scoring una sola vez.

    - Usa el logger central (`backend.logger`).
    - Respeta `SILENT_MODE`: si está activo no se logea nada.
    """
    if st.session_state.get("thresholds_logged"):
        return

    # Si el modo silencioso está activo, marcamos como logueado y salimos.
    if SILENT_MODE:
        st.session_state["thresholds_logged"] = True
        return

    eff_keep = get_auto_keep_rating_threshold()
    eff_delete = get_auto_delete_rating_threshold()
    bayes_mean, bayes_source, bayes_n = get_global_imdb_mean_info()

    _logger.info("================ UMBRALES DE SCORING EFECTIVOS ================")
    _logger.info(
        f"IMDB_KEEP_MIN_VOTES (fallback / votos por año): {IMDB_KEEP_MIN_VOTES}"
    )
    _logger.info(f"IMDB_KEEP_MIN_RATING (fallback): {IMDB_KEEP_MIN_RATING}")
    _logger.info(f"IMDB_DELETE_MAX_RATING (fallback): {IMDB_DELETE_MAX_RATING}")
    _logger.info(f"IMDB_KEEP_MIN_RATING_WITH_RT: {IMDB_KEEP_MIN_RATING_WITH_RT}")
    _logger.info(
        "AUTO_KEEP_RATING_PERCENTILE = "
        f"{AUTO_KEEP_RATING_PERCENTILE} "
        f"(RATING_MIN_TITLES_FOR_AUTO = {RATING_MIN_TITLES_FOR_AUTO})"
    )
    _logger.info(
        "AUTO_DELETE_RATING_PERCENTILE = "
        f"{AUTO_DELETE_RATING_PERCENTILE} "
        f"(RATING_MIN_TITLES_FOR_AUTO = {RATING_MIN_TITLES_FOR_AUTO})"
    )
    _logger.info(f"→ Umbral KEEP efectivo (auto/fallback)   = {eff_keep:.3f}")
    _logger.info(f"→ Umbral DELETE efectivo (auto/fallback) = {eff_delete:.3f}")
    _logger.info(
        "BAYES_GLOBAL_MEAN_DEFAULT = "
        f"{BAYES_GLOBAL_MEAN_DEFAULT} "
        f"(mín. títulos para media caché = {BAYES_MIN_TITLES_FOR_GLOBAL_MEAN})"
    )
    _logger.info(
        "Media global IMDb usada como C = "
        f"{bayes_mean:.3f} "
        f"(fuente = {bayes_source}, n = {bayes_n})"
    )
    _logger.info(f"BAYES_DELETE_MAX_SCORE = {BAYES_DELETE_MAX_SCORE}")
    _logger.info(f"IMDB_RATING_LOW_THRESHOLD = {IMDB_RATING_LOW_THRESHOLD}")
    _logger.info(f"RT_RATING_LOW_THRESHOLD   = {RT_RATING_LOW_THRESHOLD}")
    _logger.info("===============================================================")

    st.session_state["thresholds_logged"] = True


# ============================================================
# Configuración inicial
# ============================================================

# Silenciar SettingWithCopyWarning (st_aggrid/pandas)
warnings.filterwarnings(
    "ignore",
    message=".*A value is trying to be set on a copy of a slice from a DataFrame.*",
    category=pd.errors.SettingWithCopyWarning,
)
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)

# Carga de .env
load_dotenv()

# Parámetros de entorno
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "report")
DELETE_DRY_RUN = _env_bool("DELETE_DRY_RUN", True)
DELETE_REQUIRE_CONFIRM = _env_bool("DELETE_REQUIRE_CONFIRM", True)

ALL_CSV = f"{OUTPUT_PREFIX}_all.csv"
FILTERED_CSV = f"{OUTPUT_PREFIX}_filtered.csv"

METADATA_OUTPUT_PREFIX = os.getenv("METADATA_OUTPUT_PREFIX", "metadata_fix")
METADATA_SUGG_CSV = f"{METADATA_OUTPUT_PREFIX}_suggestions.csv"

# Página principal
st.set_page_config(page_title="Plex Movies Cleaner", layout="wide")
_hide_streamlit_chrome()
_init_modal_state()

# Título (no mostramos si estamos en modo modal)
if not st.session_state.get("modal_open"):
    st.title("🎬 Plex Movies Cleaner — Dashboard")

# Vista modal de detalle (si está activa, corta el flujo normal)
render_modal()
if st.session_state.get("modal_open"):
    st.stop()

# ============================================================
# Carga de datos
# ============================================================

if not os.path.exists(ALL_CSV):
    st.error("No se encuentra report_all.csv. Ejecuta analiza_plex.py primero.")
    st.stop()

df_all, df_filtered = load_reports(ALL_CSV, FILTERED_CSV)

# ============================================================
# Log de umbrales efectivos (solo una vez, respetando SILENT_MODE)
# ============================================================

_log_effective_thresholds_once()

# ============================================================
# Resumen general
# ============================================================

st.subheader("Resumen general")

summary = compute_summary(df_all)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Películas",
    format_count_size(summary["total_count"], summary["total_size_gb"]),
)
col2.metric(
    "KEEP",
    format_count_size(summary["keep_count"], summary["keep_size_gb"]),
)
col3.metric(
    "DELETE",
    format_count_size(
        summary.get("delete_count", 0),
        summary.get("delete_size_gb"),
    ),
)
col4.metric(
    "MAYBE",
    format_count_size(
        summary.get("maybe_count", 0),
        summary.get("maybe_size_gb"),
    ),
)

imdb_mean_df = summary.get("imdb_mean_df")
imdb_mean_cache = summary.get("imdb_mean_cache")

if imdb_mean_df is not None and not pd.isna(imdb_mean_df):
    col5.metric("IMDb medio (analizado)", f"{imdb_mean_df:.2f}")
else:
    col5.metric("IMDb medio (analizado)", "N/A")

if imdb_mean_cache is not None and not pd.isna(imdb_mean_cache):
    st.caption(
        f"IMDb medio global (omdb_cache / bayes): **{imdb_mean_cache:.2f}**"
    )

st.markdown("---")

# ============================================================
# Pestañas
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
    all_movies.render(df_all)

with tab2:
    candidates.render(df_all, df_filtered)

with tab3:
    advanced.render(df_all)

with tab4:
    delete.render(df_filtered, DELETE_DRY_RUN, DELETE_REQUIRE_CONFIRM)

with tab5:
    charts.render(df_all)

with tab6:
    metadata.render(METADATA_SUGG_CSV)