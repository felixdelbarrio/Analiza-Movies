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
    get_front_grid_colorize,
    save_front_grid_colorize,
)
from frontend.config_front_theme import (  # noqa: E402
    DEFAULT_THEME_KEY,
    get_front_theme,
    normalize_theme_key,
    save_front_theme,
)
from frontend.data_utils import (  # noqa: E402
    add_derived_columns,
    attach_data_version,
    format_count_size,
)
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

THEME_STATE_KEY: Final[str] = "front_theme"
THEMES: Final[dict[str, dict[str, object]]] = {
    "noir": {
        "label": "Noir Onyx",
        "tagline": "Oscuro cinematografico con contraste premium.",
        "tokens": {
            "app_bg": "#0b0f14",
            "app_decor": "radial-gradient(1100px circle at 8% -12%, rgba(58, 76, 120, 0.35), transparent 55%), radial-gradient(900px circle at 92% -18%, rgba(142, 114, 74, 0.22), transparent 52%)",
            "text_1": "#f1f5f9",
            "text_2": "#d1d5db",
            "text_3": "#9ca3af",
            "panel_bg": "linear-gradient(180deg, #121826 0%, #0f141d 100%)",
            "panel_border": "#1f2532",
            "panel_shadow": "0 18px 40px rgba(0, 0, 0, 0.35)",
            "button_bg": "#171b24",
            "button_border": "#262c38",
            "button_hover_bg": "#202635",
            "button_text": "#f3f4f6",
            "metric_bg": "#111722",
            "metric_border": "#202737",
            "tag_bg": "#2b2f36",
            "tag_border": "#3a404a",
            "tag_text": "#e5e7eb",
            "tabs_bg": "#0f141d",
            "tabs_border": "#1f2532",
            "card_bg": "#11161f",
            "card_border": "#1f2430",
            "card_shadow": "0 12px 30px rgba(0, 0, 0, 0.25)",
            "image_shadow": "0 8px 24px rgba(0, 0, 0, 0.35)",
            "pill_bg": "#171b24",
            "pill_border": "#262c38",
            "action_bg": "#171b24",
            "action_border": "#262c38",
            "action_text": "#e5e7eb",
            "action_hover_bg": "#202635",
            "divider": "#242a35",
            "summary_bg": "linear-gradient(180deg, rgba(22, 27, 38, 0.9), rgba(15, 20, 29, 0.95))",
            "summary_border": "#242b38",
            "summary_text": "#dbe2ea",
            "decision_delete": "#e55b5b",
            "decision_keep": "#56b37a",
            "decision_maybe": "#e1b75b",
            "decision_unknown": "#9da3ad",
        },
    },
    "ivory": {
        "label": "Ivory Atelier",
        "tagline": "Marfil editorial con luz suave y calidez de estudio.",
        "tokens": {
            "app_bg": "#f4efe7",
            "app_decor": "radial-gradient(1200px circle at 10% -18%, rgba(208, 183, 146, 0.45), transparent 58%), radial-gradient(900px circle at 95% -12%, rgba(171, 140, 98, 0.25), transparent 50%)",
            "text_1": "#2b2621",
            "text_2": "#4c4338",
            "text_3": "#8a7a68",
            "panel_bg": "linear-gradient(180deg, #ffffff 0%, #f1e7d9 100%)",
            "panel_border": "#dccbb7",
            "panel_shadow": "0 16px 34px rgba(86, 60, 32, 0.18)",
            "button_bg": "#f6efe4",
            "button_border": "#d9c7b2",
            "button_hover_bg": "#efe3d2",
            "button_text": "#2a2722",
            "metric_bg": "#fbf7f2",
            "metric_border": "#e1d2bf",
            "tag_bg": "#efe6da",
            "tag_border": "#d7c6b4",
            "tag_text": "#362f27",
            "tabs_bg": "#f6efe4",
            "tabs_border": "#dccbb7",
            "card_bg": "#fdfaf6",
            "card_border": "#e1d4c3",
            "card_shadow": "0 12px 30px rgba(108, 84, 55, 0.18)",
            "image_shadow": "0 10px 28px rgba(110, 84, 54, 0.18)",
            "pill_bg": "#f3eadf",
            "pill_border": "#dac9b6",
            "action_bg": "#f3eadf",
            "action_border": "#d7c6b4",
            "action_text": "#2d2720",
            "action_hover_bg": "#e8d8c5",
            "divider": "#e0d2c2",
            "summary_bg": "linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(240, 226, 210, 0.92))",
            "summary_border": "#decdb8",
            "summary_text": "#3d3329",
            "decision_delete": "#b3473f",
            "decision_keep": "#3f7f5a",
            "decision_maybe": "#b0883a",
            "decision_unknown": "#8f7f70",
        },
    },
    "sapphire": {
        "label": "Sapphire Executive",
        "tagline": "Azul ejecutivo con brillo frio y precision.",
        "tokens": {
            "app_bg": "#0b1220",
            "app_decor": "radial-gradient(1200px circle at 8% -18%, rgba(72, 118, 186, 0.35), transparent 56%), radial-gradient(900px circle at 92% -14%, rgba(60, 98, 146, 0.22), transparent 52%)",
            "text_1": "#eef3ff",
            "text_2": "#c6d2e6",
            "text_3": "#8fa3c4",
            "panel_bg": "linear-gradient(180deg, #111d32 0%, #0e1526 100%)",
            "panel_border": "#22304a",
            "panel_shadow": "0 18px 40px rgba(4, 12, 32, 0.45)",
            "button_bg": "#162033",
            "button_border": "#2a3a57",
            "button_hover_bg": "#1d2a43",
            "button_text": "#e6eefc",
            "metric_bg": "#111a2c",
            "metric_border": "#24324d",
            "tag_bg": "#1d2535",
            "tag_border": "#2c3a55",
            "tag_text": "#e6eefc",
            "tabs_bg": "#0f1726",
            "tabs_border": "#22304a",
            "card_bg": "#101a2b",
            "card_border": "#21314a",
            "card_shadow": "0 12px 30px rgba(3, 11, 28, 0.35)",
            "image_shadow": "0 8px 24px rgba(2, 8, 20, 0.45)",
            "pill_bg": "#162033",
            "pill_border": "#2a3a57",
            "action_bg": "#162033",
            "action_border": "#2a3a57",
            "action_text": "#e6eefc",
            "action_hover_bg": "#1d2a43",
            "divider": "#28344a",
            "summary_bg": "linear-gradient(180deg, rgba(20, 30, 50, 0.92), rgba(14, 21, 38, 0.94))",
            "summary_border": "#273755",
            "summary_text": "#e6edf8",
            "decision_delete": "#e46b78",
            "decision_keep": "#4fb08a",
            "decision_maybe": "#d7b365",
            "decision_unknown": "#9aa7bd",
        },
    },
    "verdant": {
        "label": "Verdant Club",
        "tagline": "Verde sobrio con aire de lounge privado.",
        "tokens": {
            "app_bg": "#0c1412",
            "app_decor": "radial-gradient(1100px circle at 8% -16%, rgba(74, 120, 104, 0.3), transparent 55%), radial-gradient(900px circle at 92% -14%, rgba(64, 104, 90, 0.22), transparent 52%)",
            "text_1": "#eef6f1",
            "text_2": "#c9d7d1",
            "text_3": "#93a39b",
            "panel_bg": "linear-gradient(180deg, #12201c 0%, #0f1714 100%)",
            "panel_border": "#23322d",
            "panel_shadow": "0 18px 40px rgba(4, 14, 10, 0.5)",
            "button_bg": "#17221e",
            "button_border": "#2a3b34",
            "button_hover_bg": "#1f2b26",
            "button_text": "#e3efe8",
            "metric_bg": "#121b17",
            "metric_border": "#26342e",
            "tag_bg": "#1b2621",
            "tag_border": "#2b3a33",
            "tag_text": "#e3efe8",
            "tabs_bg": "#111b18",
            "tabs_border": "#23322d",
            "card_bg": "#111b17",
            "card_border": "#22312b",
            "card_shadow": "0 12px 30px rgba(4, 12, 9, 0.35)",
            "image_shadow": "0 8px 24px rgba(6, 16, 12, 0.45)",
            "pill_bg": "#17221e",
            "pill_border": "#2a3b34",
            "action_bg": "#17221e",
            "action_border": "#2a3b34",
            "action_text": "#e3efe8",
            "action_hover_bg": "#1f2b26",
            "divider": "#293630",
            "summary_bg": "linear-gradient(180deg, rgba(20, 30, 26, 0.92), rgba(15, 23, 20, 0.95))",
            "summary_border": "#2a3a33",
            "summary_text": "#e1ebe6",
            "decision_delete": "#e06a63",
            "decision_keep": "#4da97a",
            "decision_maybe": "#d0b05c",
            "decision_unknown": "#9aa79f",
        },
    },
    "bordeaux": {
        "label": "Azure Atelier",
        "tagline": "Blanco porcelana y azules satinados con calma de galeria.",
        "tokens": {
            "app_bg": "#e6eef9",
            "app_decor": "radial-gradient(1200px circle at 12% -18%, rgba(77, 129, 199, 0.28), transparent 58%), radial-gradient(900px circle at 88% -10%, rgba(164, 196, 232, 0.42), transparent 54%)",
            "text_1": "#0e1b2e",
            "text_2": "#2b3f60",
            "text_3": "#5e738f",
            "panel_bg": "linear-gradient(180deg, #f9fcff 0%, #e4eefb 100%)",
            "panel_border": "#c2d3eb",
            "panel_shadow": "0 18px 40px rgba(33, 63, 104, 0.16)",
            "button_bg": "#e1ecfa",
            "button_border": "#b9cde6",
            "button_hover_bg": "#d2e2f6",
            "button_text": "#11233a",
            "metric_bg": "#e7f0fb",
            "metric_border": "#bfd2ea",
            "tag_bg": "#d9e7f9",
            "tag_border": "#b7cbe6",
            "tag_text": "#1d2f48",
            "tabs_bg": "#dbe8f8",
            "tabs_border": "#c0d3eb",
            "card_bg": "#f2f7ff",
            "card_border": "#c5d8ef",
            "card_shadow": "0 14px 32px rgba(28, 54, 92, 0.14)",
            "image_shadow": "0 12px 26px rgba(22, 40, 75, 0.18)",
            "pill_bg": "#e0ebf9",
            "pill_border": "#b9cde6",
            "action_bg": "#e0ebf9",
            "action_border": "#b9cde6",
            "action_text": "#1d2f48",
            "action_hover_bg": "#d2e2f6",
            "divider": "#c4d6ee",
            "summary_bg": "linear-gradient(180deg, rgba(248, 252, 255, 0.98), rgba(219, 233, 250, 0.98))",
            "summary_border": "#c4d6ee",
            "summary_text": "#1d2f48",
            "decision_delete": "#e24b5f",
            "decision_keep": "#2f9d6d",
            "decision_maybe": "#d79a2b",
            "decision_unknown": "#7d8faa",
        },
    },
}
THEME_ORDER: Final[tuple[str, ...]] = (
    "noir",
    "ivory",
    "sapphire",
    "verdant",
    "bordeaux",
)


def _resolve_theme_key(raw: str | None) -> str:
    if not isinstance(raw, str):
        return DEFAULT_THEME_KEY
    key = normalize_theme_key(raw)
    if key in THEMES:
        return key
    return DEFAULT_THEME_KEY


def _theme_label(theme_key: str) -> str:
    theme = THEMES.get(theme_key, THEMES[DEFAULT_THEME_KEY])
    return cast(str, theme.get("label", theme_key))


def _theme_tagline(theme_key: str) -> str:
    theme = THEMES.get(theme_key, THEMES[DEFAULT_THEME_KEY])
    return cast(str, theme.get("tagline", ""))


def _theme_tokens(theme_key: str) -> dict[str, str]:
    theme = THEMES.get(theme_key, THEMES[DEFAULT_THEME_KEY])
    tokens = theme.get("tokens", {})
    return cast(dict[str, str], tokens)


def _apply_theme(theme_key: str) -> None:
    tokens = _theme_tokens(theme_key)
    st.session_state["front_theme_tokens"] = dict(tokens)
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Manrope:wght@400;500;600;700&display=swap');
:root {{
  --mc-font-body: "Manrope", "Segoe UI", sans-serif;
  --mc-font-display: "Libre Baskerville", "Times New Roman", serif;
  --mc-app-bg: {tokens["app_bg"]};
  --mc-app-decor: {tokens["app_decor"]};
  --mc-text-1: {tokens["text_1"]};
  --mc-text-2: {tokens["text_2"]};
  --mc-text-3: {tokens["text_3"]};
  --mc-panel-bg: {tokens["panel_bg"]};
  --mc-panel-border: {tokens["panel_border"]};
  --mc-panel-shadow: {tokens["panel_shadow"]};
  --mc-button-bg: {tokens["button_bg"]};
  --mc-button-border: {tokens["button_border"]};
  --mc-button-hover-bg: {tokens["button_hover_bg"]};
  --mc-button-text: {tokens["button_text"]};
  --mc-metric-bg: {tokens["metric_bg"]};
  --mc-metric-border: {tokens["metric_border"]};
  --mc-tag-bg: {tokens["tag_bg"]};
  --mc-tag-border: {tokens["tag_border"]};
  --mc-tag-text: {tokens["tag_text"]};
  --mc-tabs-bg: {tokens["tabs_bg"]};
  --mc-tabs-border: {tokens["tabs_border"]};
  --mc-card-bg: {tokens["card_bg"]};
  --mc-card-border: {tokens["card_border"]};
  --mc-card-shadow: {tokens["card_shadow"]};
  --mc-image-shadow: {tokens["image_shadow"]};
  --mc-pill-bg: {tokens["pill_bg"]};
  --mc-pill-border: {tokens["pill_border"]};
  --mc-action-bg: {tokens["action_bg"]};
  --mc-action-border: {tokens["action_border"]};
  --mc-action-text: {tokens["action_text"]};
  --mc-action-hover-bg: {tokens["action_hover_bg"]};
  --mc-divider: {tokens["divider"]};
  --mc-summary-bg: {tokens["summary_bg"]};
  --mc-summary-border: {tokens["summary_border"]};
  --mc-summary-text: {tokens["summary_text"]};
  --mc-input-bg: {tokens["button_bg"]};
  --mc-input-border: {tokens["button_border"]};
  --mc-input-hover: {tokens["button_hover_bg"]};
  --mc-input-focus: {tokens["panel_border"]};
  --mc-input-text: {tokens["text_1"]};
  --mc-input-placeholder: {tokens["text_3"]};
  --mc-decision-delete: {tokens["decision_delete"]};
  --mc-decision-keep: {tokens["decision_keep"]};
  --mc-decision-maybe: {tokens["decision_maybe"]};
  --mc-decision-unknown: {tokens["decision_unknown"]};
  --mc-grid-bg: {tokens["card_bg"]};
  --mc-grid-header-bg: {tokens["button_bg"]};
  --mc-grid-header-text: {tokens["text_1"]};
  --mc-grid-text: {tokens["text_2"]};
  --mc-grid-border: {tokens["panel_border"]};
  --mc-grid-row-alt: {tokens["metric_bg"]};
  --mc-grid-row-hover: {tokens["tag_bg"]};
  --mc-grid-selected: {tokens["button_hover_bg"]};
  --gdg-bg-cell: var(--mc-grid-bg);
  --gdg-bg-cell-medium: var(--mc-grid-row-alt);
  --gdg-bg-cell-hover: var(--mc-grid-row-hover);
  --gdg-bg-cell-selected: var(--mc-grid-selected);
  --gdg-bg-header: var(--mc-grid-header-bg);
  --gdg-bg-header-hovered: var(--mc-grid-row-hover);
  --gdg-bg-header-selected: var(--mc-grid-selected);
  --gdg-bg-header-selected-hover: var(--mc-grid-selected);
  --gdg-bg-header-has-focus: var(--mc-grid-header-bg);
  --gdg-text-dark: var(--mc-grid-text);
  --gdg-text-medium: var(--mc-grid-text);
  --gdg-text-light: var(--mc-text-3);
  --gdg-border-color: var(--mc-grid-border);
  --gdg-horizontal-border-color: var(--mc-grid-border);
  --gdg-vertical-border-color: var(--mc-grid-border);
  --gdg-drilldown: var(--mc-grid-selected);
}}
body,
.stApp,
div[data-testid="stAppViewContainer"] {{
  background-color: var(--mc-app-bg) !important;
  background-image: var(--mc-app-decor);
  background-repeat: no-repeat;
  background-attachment: fixed;
  color: var(--mc-text-1);
  font-family: var(--mc-font-body);
}}
div[data-testid="stAppViewContainer"] h1,
div[data-testid="stAppViewContainer"] h2,
div[data-testid="stAppViewContainer"] h3,
div[data-testid="stAppViewContainer"] h4,
div[data-testid="stAppViewContainer"] h5,
div[data-testid="stAppViewContainer"] h6 {{
  color: var(--mc-text-1);
  font-family: var(--mc-font-display);
  letter-spacing: 0.01em;
}}
div[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
div[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] span {{
  color: var(--mc-text-2);
}}
div[data-testid="stAppViewContainer"] [data-testid="stMetricValue"] {{
  color: var(--mc-text-1);
}}
div[data-testid="stAppViewContainer"] [data-testid="stMetricLabel"] {{
  color: var(--mc-text-3);
}}
div[data-testid="stAppViewContainer"] a {{
  color: var(--mc-text-1);
}}
div[data-testid="stAppViewContainer"] a:hover {{
  color: var(--mc-text-2);
}}
div[data-testid="stAppViewContainer"] section.main,
div[data-testid="stAppViewContainer"] div.block-container {{
  background: transparent !important;
}}
div[data-testid="stDialog"],
div[role="dialog"] {{
  color: var(--mc-text-1);
}}
div[data-testid="stDialog"] > div,
div[data-testid="stDialog"] [data-testid="stDialogContent"],
div[role="dialog"] > div {{
  background: var(--mc-card-bg);
  border: 1px solid var(--mc-card-border);
  box-shadow: var(--mc-card-shadow);
}}
div[data-testid="stDialog"] h1,
div[data-testid="stDialog"] h2,
div[data-testid="stDialog"] h3,
div[data-testid="stDialog"] h4 {{
  color: var(--mc-text-1);
}}
div[data-testid="stDialog"] [data-testid="stWidgetLabel"] > div,
div[data-testid="stDialog"] [data-testid="stWidgetLabel"] span,
div[data-testid="stDialog"] [data-testid="stMarkdownContainer"] p {{
  color: var(--mc-text-1) !important;
}}
div[data-testid="stDialog"] [data-testid="stCheckbox"] label,
div[data-testid="stDialog"] [data-testid="stCheckbox"] span {{
  color: var(--mc-text-1) !important;
}}
div[data-testid="stExpander"] > details {{
  background: transparent !important;
}}
div[data-testid="stExpander"] > details > summary {{
  background: var(--mc-panel-bg) !important;
  border: 1px solid var(--mc-panel-border) !important;
  border-radius: 10px;
  color: var(--mc-text-1) !important;
  padding: 0.35rem 0.75rem !important;
}}
div[data-testid="stExpander"] > details[open] > summary {{
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
}}
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
  background: var(--mc-card-bg) !important;
  border: 1px solid var(--mc-panel-border) !important;
  border-top: 0 !important;
  border-radius: 0 0 10px 10px;
}}
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSlider"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stRadio"] label {{
  color: var(--mc-text-1) !important;
  font-weight: 600;
}}
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] > div {{
  background-color: var(--mc-input-bg) !important;
  border: 1px solid var(--mc-input-border) !important;
  box-shadow: none !important;
  min-height: 44px;
}}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {{
  border-color: var(--mc-input-focus) !important;
  box-shadow: 0 0 0 1px var(--mc-input-focus) !important;
}}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {{
  color: var(--mc-input-text) !important;
  caret-color: var(--mc-input-text);
  height: 44px;
  line-height: 44px;
}}
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {{
  color: var(--mc-input-placeholder) !important;
}}
div[data-baseweb="select"] div[role="combobox"],
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {{
  color: var(--mc-input-text) !important;
}}
div[data-baseweb="select"] div[role="combobox"] {{
  min-height: 44px;
  display: flex;
  align-items: center;
}}
div[data-baseweb="select"] {{
  color: var(--mc-input-text) !important;
}}
div[data-baseweb="select"] * {{
  color: var(--mc-input-text) !important;
}}
div[data-baseweb="select"] [class*="placeholder"] {{
  color: var(--mc-input-placeholder) !important;
}}
div[data-baseweb="select"] [class*="singleValue"] {{
  color: var(--mc-input-text) !important;
}}
div[data-baseweb="select"] svg {{
  color: var(--mc-text-2) !important;
  fill: var(--mc-text-2) !important;
}}
div[data-baseweb="select"] [aria-label="Clear value"] svg,
div[data-baseweb="select"] [aria-label="Clear"] svg,
div[data-baseweb="select"] [title="Clear"] svg {{
  color: var(--mc-text-2) !important;
  fill: var(--mc-text-2) !important;
}}
div[data-baseweb="popover"] div[role="listbox"],
div[data-baseweb="menu"],
div[role="listbox"] {{
  background-color: var(--mc-card-bg) !important;
  background: var(--mc-card-bg) !important;
  border: 1px solid var(--mc-panel-border) !important;
  box-shadow: var(--mc-card-shadow);
}}
div[data-baseweb="popover"] [data-baseweb="menu"],
div[data-baseweb="popover"] [data-baseweb="menu"] > ul,
div[data-baseweb="popover"] [data-baseweb="menu"] ul[role="listbox"],
div[data-baseweb="popover"] [role="listbox"],
div[data-baseweb="popover"] ul[role="listbox"] {{
  background-color: var(--mc-card-bg) !important;
  background: var(--mc-card-bg) !important;
  color: var(--mc-text-1) !important;
}}
div[data-baseweb="popover"] [data-baseweb="menu"] li,
div[data-baseweb="popover"] li[role="option"] {{
  background-color: var(--mc-card-bg) !important;
  color: var(--mc-text-1) !important;
}}
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] > div > div {{
  background-color: var(--mc-card-bg) !important;
  background: var(--mc-card-bg) !important;
}}
ul[role="listbox"] {{
  background-color: var(--mc-card-bg) !important;
  background: var(--mc-card-bg) !important;
  border: 1px solid var(--mc-panel-border) !important;
}}
div[data-baseweb="popover"] div[role="option"],
div[data-baseweb="menu"] div[role="option"],
div[role="listbox"] div[role="option"] {{
  color: var(--mc-text-1) !important;
  background-color: var(--mc-card-bg) !important;
  background: var(--mc-card-bg) !important;
}}
ul[role="listbox"] li[role="option"] {{
  color: var(--mc-text-1) !important;
  background-color: var(--mc-card-bg) !important;
  background: var(--mc-card-bg) !important;
}}
div[data-baseweb="popover"] div[role="option"][aria-selected="true"],
div[data-baseweb="popover"] div[role="option"]:hover,
div[data-baseweb="menu"] div[role="option"][aria-selected="true"],
div[data-baseweb="menu"] div[role="option"]:hover,
div[role="listbox"] div[role="option"][aria-selected="true"],
div[role="listbox"] div[role="option"]:hover {{
  background-color: var(--mc-input-hover) !important;
  background: var(--mc-input-hover) !important;
}}
ul[role="listbox"] li[role="option"][aria-selected="true"],
ul[role="listbox"] li[role="option"]:hover {{
  background-color: var(--mc-input-hover) !important;
  background: var(--mc-input-hover) !important;
}}
div[data-testid="stCheckbox"] span[role="checkbox"],
div[data-testid="stCheckbox"] span[aria-checked] {{
  background-color: var(--mc-input-bg) !important;
  border: 1px solid var(--mc-input-border) !important;
}}
div[data-testid="stCheckbox"] svg {{
  color: var(--mc-text-1) !important;
  fill: var(--mc-text-1) !important;
}}
div[data-baseweb="checkbox"] [role="checkbox"],
div[data-baseweb="checkbox"] [aria-checked] {{
  background-color: var(--mc-input-bg) !important;
  border: 1px solid var(--mc-input-border) !important;
}}
div[data-baseweb="checkbox"] svg {{
  color: var(--mc-text-1) !important;
  fill: var(--mc-text-1) !important;
}}
div[data-baseweb="checkbox"] label,
div[data-baseweb="checkbox"] span {{
  color: var(--mc-text-1) !important;
}}
div[data-testid="stDialog"] header,
div[data-testid="stDialog"] header h1,
div[data-testid="stDialog"] header h2,
div[data-testid="stDialog"] header h3 {{
  color: var(--mc-text-1) !important;
}}
button[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"],
button[data-testid="baseButton-secondary"],
button[data-testid="stBaseButton-secondary"] {{
  background: var(--mc-button-bg) !important;
  border: 1px solid var(--mc-button-border) !important;
  color: var(--mc-button-text) !important;
  box-shadow: none !important;
}}
button[data-testid="baseButton-primary"] span,
button[data-testid="stBaseButton-primary"] span,
button[data-testid="baseButton-secondary"] span,
button[data-testid="stBaseButton-secondary"] span {{
  color: var(--mc-button-text) !important;
}}
button[data-testid="baseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {{
  background: var(--mc-button-hover-bg) !important;
}}
.ag-theme-alpine,
.ag-theme-alpine-dark,
.ag-theme-streamlit {{
  --ag-background-color: var(--mc-grid-bg);
  --ag-header-background-color: var(--mc-grid-header-bg);
  --ag-header-foreground-color: var(--mc-grid-header-text);
  --ag-foreground-color: var(--mc-grid-text);
  --ag-odd-row-background-color: var(--mc-grid-row-alt);
  --ag-row-hover-color: var(--mc-grid-row-hover);
  --ag-selected-row-background-color: var(--mc-grid-selected);
  --ag-border-color: var(--mc-grid-border);
  --ag-row-border-color: var(--mc-divider);
  --ag-font-family: var(--mc-font-body);
}}
.ag-theme-alpine,
.ag-theme-alpine-dark,
.ag-theme-streamlit {{
  background-color: var(--mc-grid-bg) !important;
  color: var(--mc-grid-text) !important;
}}
.ag-theme-alpine .ag-root-wrapper,
.ag-theme-alpine-dark .ag-root-wrapper,
.ag-theme-streamlit .ag-root-wrapper,
.ag-theme-alpine .ag-root-wrapper-body,
.ag-theme-alpine-dark .ag-root-wrapper-body,
.ag-theme-streamlit .ag-root-wrapper-body,
.ag-theme-alpine .ag-root,
.ag-theme-alpine-dark .ag-root,
.ag-theme-streamlit .ag-root,
.ag-theme-alpine .ag-body-viewport,
.ag-theme-alpine-dark .ag-body-viewport,
.ag-theme-streamlit .ag-body-viewport,
.ag-theme-alpine .ag-center-cols-viewport,
.ag-theme-alpine-dark .ag-center-cols-viewport,
.ag-theme-streamlit .ag-center-cols-viewport,
.ag-theme-alpine .ag-body-horizontal-scroll-viewport,
.ag-theme-alpine-dark .ag-body-horizontal-scroll-viewport,
.ag-theme-streamlit .ag-body-horizontal-scroll-viewport,
.ag-theme-alpine .ag-body-vertical-scroll-viewport,
.ag-theme-alpine-dark .ag-body-vertical-scroll-viewport,
.ag-theme-streamlit .ag-body-vertical-scroll-viewport,
.ag-theme-alpine .ag-center-cols-clipper,
.ag-theme-alpine-dark .ag-center-cols-clipper,
.ag-theme-streamlit .ag-center-cols-clipper,
.ag-theme-alpine .ag-center-cols-container,
.ag-theme-alpine-dark .ag-center-cols-container,
.ag-theme-streamlit .ag-center-cols-container {{
  background-color: var(--mc-grid-bg) !important;
}}
.ag-theme-alpine .ag-header,
.ag-theme-alpine-dark .ag-header,
.ag-theme-streamlit .ag-header {{
  background-color: var(--mc-grid-header-bg) !important;
  border-bottom: 1px solid var(--mc-grid-border) !important;
}}
.ag-theme-alpine .ag-header-cell,
.ag-theme-alpine-dark .ag-header-cell,
.ag-theme-streamlit .ag-header-cell,
.ag-theme-alpine .ag-header-group-cell,
.ag-theme-alpine-dark .ag-header-group-cell,
.ag-theme-streamlit .ag-header-group-cell {{
  background-color: var(--mc-grid-header-bg) !important;
  color: var(--mc-grid-header-text) !important;
  border-color: var(--mc-grid-border) !important;
}}
.ag-theme-alpine .ag-cell,
.ag-theme-alpine-dark .ag-cell,
.ag-theme-streamlit .ag-cell {{
  color: inherit !important;
  border-color: var(--mc-grid-border) !important;
}}
.ag-theme-alpine .ag-row,
.ag-theme-alpine-dark .ag-row,
.ag-theme-streamlit .ag-row {{
  border-color: var(--mc-grid-border) !important;
}}
.ag-theme-alpine .ag-row-odd,
.ag-theme-alpine-dark .ag-row-odd,
.ag-theme-streamlit .ag-row-odd {{
  background-color: var(--mc-grid-row-alt) !important;
}}
.ag-theme-alpine .ag-row-hover,
.ag-theme-alpine-dark .ag-row-hover,
.ag-theme-streamlit .ag-row-hover {{
  background-color: var(--mc-grid-row-hover) !important;
}}
.ag-theme-alpine .ag-row-selected,
.ag-theme-alpine-dark .ag-row-selected,
.ag-theme-streamlit .ag-row-selected {{
  background-color: var(--mc-grid-selected) !important;
}}
.ag-theme-alpine .ag-icon,
.ag-theme-alpine-dark .ag-icon,
.ag-theme-streamlit .ag-icon {{
  color: var(--mc-grid-header-text) !important;
  fill: var(--mc-grid-header-text) !important;
}}
.ag-theme-alpine .ag-paging-panel,
.ag-theme-alpine-dark .ag-paging-panel,
.ag-theme-streamlit .ag-paging-panel {{
  color: var(--mc-grid-text) !important;
}}
div[data-testid="stDataFrame"] {{
  background-color: var(--mc-grid-bg) !important;
  border: 1px solid var(--mc-grid-border) !important;
  border-radius: 12px;
  overflow: hidden;
  --gdg-bg-cell: var(--mc-grid-bg);
  --gdg-bg-cell-medium: var(--mc-grid-row-alt);
  --gdg-bg-cell-hover: var(--mc-grid-row-hover);
  --gdg-bg-cell-selected: var(--mc-grid-selected);
  --gdg-bg-header: var(--mc-grid-header-bg);
  --gdg-bg-header-hovered: var(--mc-grid-row-hover);
  --gdg-bg-header-selected: var(--mc-grid-selected);
  --gdg-bg-header-selected-hover: var(--mc-grid-selected);
  --gdg-text-dark: var(--mc-grid-text);
  --gdg-text-medium: var(--mc-grid-text);
  --gdg-text-light: var(--mc-text-3);
  --gdg-border-color: var(--mc-grid-border);
  --gdg-horizontal-border-color: var(--mc-grid-border);
  --gdg-vertical-border-color: var(--mc-grid-border);
  --gdg-drilldown: var(--mc-grid-selected);
}}
glide-data-grid,
div[data-testid="stDataFrame"] glide-data-grid,
div[data-testid="stDataFrame"] .glide-data-grid {{
  background-color: var(--mc-grid-bg) !important;
  color: var(--mc-grid-text) !important;
}}
div[data-testid="stDataFrame"] canvas {{
  background-color: var(--mc-grid-bg) !important;
}}
div[data-testid="stDataFrame"] [role="grid"],
div[data-testid="stDataFrame"] [role="table"] {{
  background-color: var(--mc-grid-bg) !important;
  color: var(--mc-grid-text) !important;
}}
div[data-testid="stDataFrame"] [role="columnheader"],
div[data-testid="stDataFrame"] [role="rowheader"] {{
  background-color: var(--mc-grid-header-bg) !important;
  color: var(--mc-grid-header-text) !important;
  border-color: var(--mc-grid-border) !important;
}}
div[data-testid="stDataFrame"] [role="row"] {{
  background-color: var(--mc-grid-bg) !important;
}}
div[data-testid="stDataFrame"] [role="row"]:nth-child(even) {{
  background-color: var(--mc-grid-row-alt) !important;
}}
div[data-testid="stDataFrame"] [role="gridcell"] {{
  color: var(--mc-grid-text) !important;
  border-color: var(--mc-grid-border) !important;
}}
.theme-preview {{
  display: flex;
  gap: 0.35rem;
  margin-top: 0.2rem;
}}
.theme-chip {{
  width: 18px;
  height: 18px;
  border-radius: 6px;
  border: 1px solid var(--mc-panel-border);
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
}}
</style>
""",
        unsafe_allow_html=True,
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
    return add_derived_columns(df)


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
    df = fetch_report_filtered_df(
        base_url=base_url,
        timeout_s=timeout_s,
        page_size=page_size,
        query=query,
    )
    if df is None:
        return None
    return add_derived_columns(df)


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


def _attach_data_versions(
    *,
    df_all: pd.DataFrame,
    df_filtered: pd.DataFrame | None,
) -> None:
    """
    Añade un data_version estable para cacheos ligeros en frontend.
    """
    if FRONT_MODE == "disk":
        all_hint = _mtime_ns(REPORT_ALL_PATH)
        attach_data_version(df_all, source="disk", hint=all_hint)
        if df_filtered is not None:
            filt_hint = _mtime_ns(REPORT_FILTERED_PATH)
            attach_data_version(df_filtered, source="disk_filtered", hint=filt_hint)
        return

    api_hint = f"{FRONT_API_BASE_URL}:{FRONT_API_PAGE_SIZE}"
    attach_data_version(df_all, source="api", hint=api_hint)
    if df_filtered is not None:
        attach_data_version(df_filtered, source="api_filtered", hint=api_hint)


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
if THEME_STATE_KEY not in st.session_state:
    st.session_state[THEME_STATE_KEY] = _resolve_theme_key(get_front_theme())
_apply_theme(_resolve_theme_key(st.session_state.get(THEME_STATE_KEY)))

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

_attach_data_versions(df_all=df_all, df_filtered=df_filtered)
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


def _render_config_container(body: Callable[[], None]) -> None:
    with st.expander("Configuracion", expanded=True):
        body()


if "grid_colorize_rows" not in st.session_state:
    st.session_state["grid_colorize_rows"] = get_front_grid_colorize()
colorize_rows = bool(st.session_state.get("grid_colorize_rows", True))

CONFIG_COLORIZE_KEY = "config_colorize"
CONFIG_THEME_KEY = "config_theme"
CONFIG_SHOW_NUMERIC_KEY = "config_show_numeric_filters"
CONFIG_SHOW_THRESHOLDS_KEY = "config_show_chart_thresholds"
CONFIG_VIEWS_KEY = "config_dashboard_views"
CONFIG_EDITING_KEY = "config_editing"
CONFIG_RESET_KEY = "config_reset_pending"
CONFIG_DEFAULTS_KEY = "config_defaults"
config_clicked = False

# =============================================================================
# 8) Resumen general (KPIs)
# =============================================================================

with st.container():
    st.markdown(
        """
<div id="summary-card-anchor"></div>
<style>
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) {
  background: var(--mc-panel-bg);
  border: 1px solid var(--mc-panel-border);
  border-radius: 18px;
  padding: 10px 18px 20px;
  box-shadow: var(--mc-panel-shadow);
  margin-bottom: -0.2rem !important;
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) h1 {
  margin: 0;
  line-height: 1.1;
  font-size: 1.9rem;
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
  color: var(--mc-button-text) !important;
  font-size: 0.95rem;
  font-weight: 700;
  padding: 0.35rem 0.75rem;
  border-radius: 0.6rem;
  background: var(--mc-button-bg);
  border: 1px solid var(--mc-button-border);
}
div[data-testid="stVerticalBlock"]:has(#config-top-anchor) button[data-testid="stBaseButton-secondary"]:hover span,
div[data-testid="stVerticalBlock"]:has(#config-top-anchor) button[data-testid="baseButton-secondary"]:hover span {
  color: var(--mc-button-text) !important;
  background: var(--mc-button-hover-bg);
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) .summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.6rem;
  margin-top: 0.3rem;
  margin-bottom: 0;
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) .summary-card {
  background: var(--mc-metric-bg);
  border: 1px solid var(--mc-metric-border);
  border-radius: 12px;
  padding: 0.55rem 0.7rem;
  min-height: 68px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 0.2rem;
  box-sizing: border-box;
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) .summary-label {
  color: var(--mc-text-3);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-size: 0.64rem;
  line-height: 1.1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) .summary-value {
  font-size: 1.4rem;
  color: var(--mc-text-1);
  line-height: 1.1;
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) .summary-grid--colorize .summary-card--keep .summary-label,
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) .summary-grid--colorize .summary-card--keep .summary-value {
  color: var(--mc-decision-keep);
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) .summary-grid--colorize .summary-card--delete .summary-label,
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) .summary-grid--colorize .summary-card--delete .summary-value {
  color: var(--mc-decision-delete);
}
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) .summary-grid--colorize .summary-card--maybe .summary-label,
div[data-testid="stVerticalBlock"]:has(#summary-card-anchor) .summary-grid--colorize .summary-card--maybe .summary-value {
  color: var(--mc-decision-maybe);
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
        config_clicked = st.button(
            "Configuracion",
            key="config_link_top",
            type="secondary",
        )
    summary = compute_summary(df_all)

    imdb_mean_df = summary.get("imdb_mean_df")
    imdb_value = (
        f"{imdb_mean_df:.2f}"
        if imdb_mean_df is not None and not pd.isna(imdb_mean_df)
        else "N/A"
    )
    summary_cards = [
        (
            "Películas",
            format_count_size(summary["total_count"], summary["total_size_gb"]),
            "neutral",
        ),
        (
            "KEEP",
            format_count_size(summary["keep_count"], summary["keep_size_gb"]),
            "keep",
        ),
        (
            "DELETE",
            format_count_size(summary["delete_count"], summary["delete_size_gb"]),
            "delete",
        ),
        (
            "MAYBE",
            format_count_size(summary["maybe_count"], summary["maybe_size_gb"]),
            "maybe",
        ),
        ("IMDb medio (analizado)", imdb_value, "neutral"),
    ]
    grid_class = (
        "summary-grid summary-grid--colorize" if colorize_rows else "summary-grid"
    )
    cards_html = "\n".join(
        f'<div class="summary-card summary-card--{variant}">'
        f'<div class="summary-label">{label}</div>'
        f'<div class="summary-value">{value}</div>'
        "</div>"
        for label, value, variant in summary_cards
    )
    st.markdown(
        f'<div class="{grid_class}">{cards_html}</div>',
        unsafe_allow_html=True,
    )

available_dashboard = [v for v in VIEW_OPTIONS if v != "Dashboard"]


def _load_config_defaults() -> dict[str, object]:
    return {
        CONFIG_COLORIZE_KEY: bool(
            st.session_state.get("grid_colorize_rows", get_front_grid_colorize())
        ),
        CONFIG_THEME_KEY: _resolve_theme_key(
            st.session_state.get(THEME_STATE_KEY, get_front_theme())
        ),
        CONFIG_SHOW_NUMERIC_KEY: get_show_numeric_filters(),
        CONFIG_SHOW_THRESHOLDS_KEY: get_show_chart_thresholds(),
        CONFIG_VIEWS_KEY: get_dashboard_views(available_dashboard),
    }


def _ensure_config_defaults() -> None:
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
    theme_key = st.selectbox(
        "Tema de interfaz",
        list(THEME_ORDER),
        key=CONFIG_THEME_KEY,
        format_func=_theme_label,
    )
    theme_tagline = _theme_tagline(theme_key)
    if theme_tagline:
        st.caption(theme_tagline)
    preview_tokens = _theme_tokens(theme_key)
    st.markdown(
        f"""
<div class="theme-preview">
  <span class="theme-chip" style="background: {preview_tokens["app_bg"]}; border: 1px solid {preview_tokens["panel_border"]};"></span>
  <span class="theme-chip" style="background: {preview_tokens["panel_bg"]}; border: 1px solid {preview_tokens["panel_border"]};"></span>
  <span class="theme-chip" style="background: {preview_tokens["button_bg"]}; border: 1px solid {preview_tokens["button_border"]};"></span>
  <span class="theme-chip" style="background: {preview_tokens["summary_bg"]}; border: 1px solid {preview_tokens["summary_border"]};"></span>
  <span class="theme-chip" style="background: {preview_tokens["decision_keep"]}; border: 1px solid {preview_tokens["panel_border"]};"></span>
  <span class="theme-chip" style="background: {preview_tokens["decision_maybe"]}; border: 1px solid {preview_tokens["panel_border"]};"></span>
  <span class="theme-chip" style="background: {preview_tokens["decision_delete"]}; border: 1px solid {preview_tokens["panel_border"]};"></span>
  <span class="theme-chip" style="background: {preview_tokens["decision_unknown"]}; border: 1px solid {preview_tokens["panel_border"]};"></span>
</div>
""",
        unsafe_allow_html=True,
    )
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
            resolved_theme = _resolve_theme_key(theme_key)
            save_front_theme(resolved_theme)
            save_front_grid_colorize(colorize)
            save_dashboard_views(exec_views)
            save_show_numeric_filters(show_numeric_filters)
            save_show_chart_thresholds(show_chart_thresholds)
            st.session_state[THEME_STATE_KEY] = resolved_theme
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


dialog_fn: Any = getattr(st, "dialog", None)
_config_dialog_fn: Callable[[], None] | None = None
if callable(dialog_fn):
    dialog_fn = cast(
        Callable[[str], Callable[[Callable[[], None]], Callable[[], None]]], dialog_fn
    )

    @dialog_fn("Configuracion")
    def _config_dialog_impl() -> None:
        _config_body()

    _config_dialog_fn = _config_dialog_impl


def _open_config() -> None:
    _ensure_config_defaults()
    if _config_dialog_fn is not None:
        _config_dialog_fn()
    else:
        _render_config_container(_config_body)


if config_clicked:
    if _config_dialog_fn is not None:
        _open_config()
    else:
        st.session_state["config_open"] = True

if _config_dialog_fn is None and st.session_state.get("config_open"):
    _open_config()

# Auto-apply summary semaforo colors when enabled.
# Consistencia visual: tags de multiselect con fondo neutro (evita rojo fijo).
st.markdown(
    """
<style>
div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
  background-color: var(--mc-tag-bg) !important;
  border: 1px solid var(--mc-tag-border) !important;
}
div[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
  color: var(--mc-tag-text) !important;
}
div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
div[data-testid="stMultiSelect"] [data-baseweb="tag"] button {
  color: var(--mc-tag-text) !important;
  fill: var(--mc-tag-text) !important;
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
  margin-top: 0 !important;
  padding-top: 0 !important;
  margin-top: 0 !important;
  margin-bottom: 0.05rem;
  background: var(--mc-tabs-bg);
  border: 1px solid var(--mc-tabs-border);
  border-radius: 14px;
  padding: 0.08rem 0.35rem 0;
}
div[data-testid="stVerticalBlock"]:has(#tabs-row-anchor) div[data-testid="stTabs"] {
  padding-top: 0;
}
div[data-testid="stVerticalBlock"]:has(#tabs-row-anchor) [data-testid="stTabList"] {
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
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
