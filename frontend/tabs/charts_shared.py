"""
Shared helpers and constants for charts rendering.
"""

from __future__ import annotations

from typing import Any, Callable, Final, Iterable, TypeVar, cast

import altair as alt
import pandas as pd
import streamlit as st

from frontend.config_front_theme import get_front_theme, normalize_theme_key
from frontend.data_utils import dataframe_signature, decision_color

_F = TypeVar("_F", bound=Callable[..., Any])
AltChart = Any
AltSelection = Any

DECISION_ORDER: Final[list[str]] = ["DELETE", "MAYBE", "KEEP", "UNKNOWN"]

IMDB_OUTLIER_HIGH: Final[float] = 8.5
IMDB_OUTLIER_LOW: Final[float] = 5.0
IMDB_REFERENCE: Final[float] = 7.0
RT_REFERENCE: Final[float] = 60.0
METACRITIC_REFERENCE: Final[float] = 60.0
DELETE_WEIGHT: Final[float] = 3.0
MAYBE_WEIGHT: Final[float] = 1.0

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
        return cast(
            Callable[[_F], _F],
            cache_fn(
                show_spinner=False,
                hash_funcs={pd.DataFrame: dataframe_signature},
            ),
        )
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
            labelColor=text,
            titleColor=text_strong,
            labelFont=FONT_BODY,
            titleFont=FONT_BODY,
            labelFontSize=11,
            titleFontSize=12,
        )
    )


def _chart(chart: AltChart) -> AltChart:
    """
    Wrapper para mostrar graficos siempre a ancho completo.
    """
    chart = _apply_chart_theme(chart)
    chart = chart.properties(padding={"top": 14, "left": 8, "right": 8, "bottom": 8})
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
        out.append(alt.Tooltip("rt_score:Q", title="RT", format=".0f"))
    if "metacritic_score" in df.columns:
        out.append(alt.Tooltip("metacritic_score:Q", title="Metacritic", format=".0f"))
    if "imdb_votes" in df.columns:
        out.append(alt.Tooltip("imdb_votes:Q", title="Votos", format=","))
    if "file_size_gb" in df.columns:
        out.append(alt.Tooltip("file_size_gb:Q", title="Tamano (GB)", format=".2f"))
    return out


def _requires_columns(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    """
    Comprueba que `df` contiene todas las columnas indicadas.

    Returns:
      - True  si todas las columnas estan presentes.
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
