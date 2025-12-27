from __future__ import annotations

# =============================================================================
# frontend/dashboard.py
#
# Dashboard principal (Streamlit) 100% desacoplado del backend.
#
# IMPORTANTE SOBRE STREAMLIT:
# - Streamlit renderiza automáticamente el "docstring de módulo" (triple comillas
#   al inicio del archivo) como Markdown en la UI.
# - Por eso, NO ponemos docstring de módulo. En su lugar:
#     * Usamos comentarios con '#'
#     * Usamos docstrings dentro de funciones (esto NO se renderiza en la UI).
#
# FUENTES DE DATOS (solo ficheros en disco):
# - reports/: report_all.csv, report_filtered.csv, metadata_fix.csv
# - data/: omdb_cache.json, wiki_cache.json (opcional en UI)
#
# PRINCIPIOS DE DISEÑO:
# - 0 imports de backend.*
# - Config autocontenida en frontend/config_front_*.py (lee .env.front)
# - “Logging” mínimo: solo en UI si FRONT_DEBUG=True (st.caption)
# =============================================================================

import sys
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

# =============================================================================
# 1) Fix de import path (solo para ejecución directa con Streamlit)
#
# Caso típico:
#   streamlit run frontend/dashboard.py
#
# En ese modo, Python NO siempre incluye la raíz del proyecto en sys.path.
# Para permitir imports tipo: "from frontend.xxx import yyy"
# forzamos a que el PROJECT_ROOT esté en sys.path.
# =============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =============================================================================
# 2) Imports FRONT-ONLY (sin backend)
# =============================================================================

from frontend.config_front_base import FRONT_DEBUG
from frontend.config_front_artifacts import (
    METADATA_FIX_PATH,
    REPORT_ALL_PATH,
    REPORT_FILTERED_PATH,
)
from frontend.front_stats import compute_global_imdb_mean_from_df
from frontend.summary import compute_summary
from frontend.data_utils import format_count_size

# =============================================================================
# 3) Flags de acciones (borrado)
#
# - Si tu tab delete depende de estas flags, mantenlas aquí por ahora.
# - Recomendación: moverlas a un módulo:
#     frontend/config_front_actions.py
#   leyendo .env.front (con defaults).
# =============================================================================

DELETE_DRY_RUN: bool = True
DELETE_REQUIRE_CONFIRM: bool = True

# =============================================================================
# 4) Helpers UI (front-only)
# =============================================================================


def _hide_streamlit_chrome() -> None:
    """
    Oculta elementos de “chrome” de Streamlit (cabecera, toolbar) y ajusta paddings.

    Nota:
    - Esto es puramente estético.
    - Es seguro incluso si Streamlit cambia la estructura: en ese caso solo dejará
      de ocultar, pero no rompe la app.
    """
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
    """
    Inicializa claves de session_state usadas por el modal.

    Convención:
    - modal_open: bool
    - modal_row : contenido asociado al detalle (dict/serie/row, según tu implementación)
    """
    st.session_state.setdefault("modal_open", False)
    st.session_state.setdefault("modal_row", None)


def _debug_banner(*, df_all: pd.DataFrame, df_filtered: pd.DataFrame | None) -> None:
    """
    Instrumentación en UI solo si FRONT_DEBUG=True.

    Útil para diagnosticar:
    - número de filas cargadas
    - rutas efectivas de los artifacts (reports/ y metadata/)
    """
    if not FRONT_DEBUG:
        return

    f_rows = 0 if df_filtered is None else len(df_filtered)
    st.caption(
        "DEBUG | "
        f"all={len(df_all)} filtered={f_rows} | "
        f"all='{REPORT_ALL_PATH}' filtered='{REPORT_FILTERED_PATH}' metadata='{METADATA_FIX_PATH}'"
    )


def _read_csv_or_raise(path: Path, *, label: str) -> pd.DataFrame:
    """
    Lee un CSV obligatorio de forma defensiva y corta la ejecución si falla.

    Reglas:
    - Si el fichero no existe: st.error + st.stop
    - Si el CSV no se puede leer: st.error + st.stop
    - Si está vacío: st.error + st.stop
    """
    if not path.exists():
        st.error(f"No se encuentra {label}: {path}")
        st.stop()

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        st.error(f"No se pudo leer {label}: {path}\n\n{exc!r}")
        st.stop()

    if df.empty:
        st.error(f"{label} está vacío: {path}")
        st.stop()

    return df


def _read_csv_or_none(path: Path) -> pd.DataFrame | None:
    """
    Lee un CSV opcional.

    - Si no existe: devuelve None (normal)
    - Si existe pero no se puede leer: devuelve None y, si FRONT_DEBUG=True, lo muestra en UI.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if isinstance(df, pd.DataFrame) else None
    except Exception as exc:
        if FRONT_DEBUG:
            st.caption(f"DEBUG: read_csv failed for {path}: {exc!r}")
        return None


# =============================================================================
# 5) Silenciar warnings de Pandas típicos en UI (SettingWithCopyWarning)
# =============================================================================

warnings.filterwarnings(
    "ignore",
    message=".*A value is trying to be set on a copy of a slice from a DataFrame.*",
    category=pd.errors.SettingWithCopyWarning,
)
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)

# =============================================================================
# 6) Setup Streamlit (layout + estado)
# =============================================================================

st.set_page_config(page_title="Movies Cleaner", layout="wide")
_hide_streamlit_chrome()
_init_modal_state()

# Título (evitamos mostrarlo si estamos en modo modal para no “duplicar” UI)
if not st.session_state.get("modal_open"):
    st.title("🎬 Movies Cleaner — Dashboard")

# =============================================================================
# 7) Render modal (front-only)
#
# Import lazy para evitar ciclos:
# - componentes pueden importar tabs
# - tabs pueden importar componentes
# =============================================================================

from frontend.components import render_modal

render_modal()
if st.session_state.get("modal_open"):
    # Si el modal está abierto, paramos el flujo para “bloquear” la pantalla detrás.
    st.stop()

# =============================================================================
# 8) Carga de datos (front-only)
#
# report_all.csv es imprescindible.
# report_filtered.csv es opcional (depende de que el backend lo genere).
# =============================================================================

df_all = _read_csv_or_raise(REPORT_ALL_PATH, label="report_all.csv")
df_filtered = _read_csv_or_none(REPORT_FILTERED_PATH)

_debug_banner(df_all=df_all, df_filtered=df_filtered)

# =============================================================================
# 9) Resumen general (KPIs)
# =============================================================================

st.subheader("Resumen general")

# compute_summary debe ser robusto: si faltan columnas, no debe romper.
summary = compute_summary(df_all)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Películas", format_count_size(summary["total_count"], summary["total_size_gb"]))
col2.metric("KEEP", format_count_size(summary["keep_count"], summary["keep_size_gb"]))
col3.metric("DELETE", format_count_size(summary.get("delete_count", 0), summary.get("delete_size_gb")))
col4.metric("MAYBE", format_count_size(summary.get("maybe_count", 0), summary.get("maybe_size_gb")))

# Media IMDb calculada DIRECTAMENTE desde el df ya cargado (no re-lee fichero)
imdb_mean_df = compute_global_imdb_mean_from_df(df_all)
if imdb_mean_df is not None and not pd.isna(imdb_mean_df):
    col5.metric("IMDb medio (analizado)", f"{imdb_mean_df:.2f}")
else:
    col5.metric("IMDb medio (analizado)", "N/A")

st.markdown("---")

# =============================================================================
# 10) Pestañas (tabs)
#
# Notas de arquitectura:
# - Importamos las tabs dentro de cada bloque para evitar import cycles.
# - Cada tab debe seguir el principio: 0 imports de backend.
# =============================================================================

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
    from frontend.tabs import delete

    # Si df_filtered es None (no existe report_filtered.csv), la tab debe gestionarlo.
    delete.render(df_filtered, DELETE_DRY_RUN, DELETE_REQUIRE_CONFIRM)

with tab5:
    from frontend.tabs import charts

    charts.render(df_all)

with tab6:
    from frontend.tabs import metadata

    metadata.render(METADATA_FIX_PATH)