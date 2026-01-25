"""
all_movies.py

Pestaña “Todas” (Streamlit).

Responsabilidad:
- Mostrar filtros avanzados por columnas (biblioteca, decisión, métricas, etc.).
- Aplicar filtros sobre df_all y mostrar resultados.
- Renderizar grid (AgGrid) y tarjeta de detalle (panel lateral).

Principios:
- No mutar df_all: trabajar sobre una copia.
- Ser tolerante a columnas ausentes: degradar de forma segura (sin crash).
- Mantener este módulo centrado en UI/filtrado; la lógica de render se delega a
  frontend.components (grid + detalle).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast
from urllib.parse import quote
import html

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

from frontend.components import (
    aggrid_with_row_click,
    render_decision_chip_styles,
    render_detail_card,
    render_grid_toolbar,
)
from frontend.config_front_charts import get_show_numeric_filters
from frontend.data_utils import (
    dataframe_signature,
    directors_from_omdb_json_or_cache,
    explode_genres_from_omdb_json,
    safe_json_loads_single,
)

_DECISION_LABELS: dict[str, str] = {
    "DELETE": "DELETE",
    "MAYBE": "MAYBE",
    "KEEP": "KEEP",
    "UNKNOWN": "UNKNOWN",
}


def _decision_label(value: object) -> str:
    label = _DECISION_LABELS.get(value) if isinstance(value, str) else None
    if isinstance(label, str):
        return label
    return str(value)


# ============================================================================
# Helpers
# ============================================================================


def _safe_unique_sorted(df: pd.DataFrame, col: str) -> list[str]:
    """
    Devuelve valores únicos no vacíos/NaN de una columna, ordenados alfabéticamente.

    Nota mypy:
    - pandas .unique().tolist() suele estar tipado como Any/list[Any] en stubs,
      así que construimos explícitamente list[str].
    """
    if col not in df.columns:
        return []

    series = df[col].dropna().astype(str).map(str.strip)
    series = series.mask(series == "", pd.NA).dropna()
    raw: list[Any] = series.unique().tolist()

    out: list[str] = []
    for v in raw:
        s = str(v).strip()
        if s:
            out.append(s)

    out.sort()
    return out


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(cast(Any, value)))
    except Exception:
        return False


def _safe_text(value: object, *, fallback: str = "—") -> str:
    if _is_missing_value(value):
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    return html.escape(text)


def _format_int(value: object, *, fallback: str = "—") -> str:
    if _is_missing_value(value):
        return fallback
    try:
        value_str = str(value)
        return f"{int(float(value_str)):,}"
    except (TypeError, ValueError):
        return fallback


def _format_float(value: object, *, digits: int = 1, fallback: str = "—") -> str:
    if _is_missing_value(value):
        return fallback
    try:
        value_str = str(value)
        return f"{float(value_str):.{digits}f}"
    except (TypeError, ValueError):
        return fallback


def _format_year(value: object) -> str:
    if _is_missing_value(value):
        return ""
    try:
        value_str = str(value).strip()
        year_int = int(float(value_str))
        if year_int <= 0:
            return ""
        return str(year_int)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text


def _poster_from_row(row: Mapping[str, Any]) -> str | None:
    poster = row.get("poster_url")
    if isinstance(poster, str) and poster.strip() and poster.strip().upper() != "N/A":
        return poster.strip()
    omdb_raw = row.get("omdb_json")
    parsed = safe_json_loads_single(omdb_raw)
    if isinstance(parsed, Mapping):
        poster = parsed.get("Poster")
        if (
            isinstance(poster, str)
            and poster.strip()
            and poster.strip().upper() != "N/A"
        ):
            return poster.strip()
    return None


def _build_detail_query_for_row(row: Mapping[str, Any]) -> str | None:
    params: list[str] = ["open_detail=1"]

    def _add(key: str, param: str | None = None) -> None:
        value = row.get(key)
        if _is_missing_value(value):
            return
        value_str = str(value).strip()
        if not value_str:
            return
        params.append(f"{param or key}={quote(value_str)}")

    _add("imdb_id", "imdb_id")
    _add("guid", "guid")
    _add("title", "title")
    _add("year", "year")

    if len(params) == 1:
        return None
    return "?" + "&".join(params)


def _render_cards_styles() -> None:
    anchor_id = "all-movies-cards"
    st.markdown(
        f"""
<div id="{anchor_id}" style="display:none;"></div>
<style>
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) {{
  padding-top: 0.25rem;
}}
.mc-all-cards {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 1.15rem;
}}
.mc-all-card {{
  background: var(--mc-card-bg);
  border: 1px solid var(--mc-card-border);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: var(--mc-card-shadow);
  display: flex;
  flex-direction: column;
  min-height: 100%;
}}
.mc-all-card-link {{
  display: block;
  text-decoration: none;
  color: inherit;
  cursor: pointer;
}}
.mc-all-card-link:hover .mc-all-card {{
  transform: translateY(-2px);
  box-shadow: 0 14px 30px rgba(0, 0, 0, 0.35);
}}
.mc-all-card-link:focus-visible .mc-all-card {{
  outline: 2px solid var(--mc-input-focus);
  outline-offset: 2px;
}}
.mc-all-card-poster {{
  aspect-ratio: 2 / 3;
  background: var(--mc-panel-bg);
  border-bottom: 1px solid var(--mc-panel-border);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--mc-text-3);
  font-size: 1.6rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.mc-all-card-poster img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}}
.mc-all-card-body {{
  padding: 0.7rem 0.85rem 0.9rem;
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}}
.mc-all-card-title {{
  color: var(--mc-text-1);
  font-family: var(--mc-font-display);
  font-weight: 700;
  font-size: 0.98rem;
  line-height: 1.2;
}}
.mc-all-card-subtitle {{
  color: var(--mc-text-3);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}
.mc-all-card-library {{
  display: inline-flex;
  width: fit-content;
  padding: 0.25rem 0.55rem;
  border-radius: 999px;
  background: var(--mc-pill-bg);
  border: 1px solid var(--mc-pill-border);
  color: var(--mc-text-2);
  font-size: 0.7rem;
}}
.mc-all-card-stats {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.35rem 0.55rem;
}}
.mc-all-card-stat {{
  background: var(--mc-metric-bg);
  border: 1px solid var(--mc-metric-border);
  border-radius: 10px;
  padding: 0.3rem 0.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  font-size: 0.75rem;
  color: var(--mc-text-2);
}}
.mc-all-card-stat span {{
  color: var(--mc-text-3);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}
.mc-all-card-stat strong {{
  color: var(--mc-text-1);
  font-weight: 600;
}}
.mc-all-card-stat.full {{
  grid-column: span 2;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_view_toggle() -> str:
    anchor_id = "all-movies-view-toggle"
    st.markdown(
        f"""
<div id="{anchor_id}" style="display:none;"></div>
<style>
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] > div {{
  flex-direction: row !important;
  gap: 0.25rem;
  background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
  border: 1px solid var(--mc-panel-border);
  border-radius: 999px;
  padding: 0.3rem;
  width: fit-content;
  box-shadow: 0 12px 28px rgba(0,0,0,0.25);
  backdrop-filter: blur(8px);
  height: 42px;
  align-items: center;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) label[data-testid="stWidgetLabel"] {{
  display: none !important;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] > label {{
  display: none !important;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] div[role="radiogroup"] label {{
  margin: 0 !important;
  padding: 0.35rem 0.9rem !important;
  border-radius: 999px !important;
  border: 1px solid transparent !important;
  display: inline-flex !important;
  align-items: center !important;
  gap: 0.55rem !important;
  cursor: pointer;
  color: var(--mc-text-2) !important;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  transition: all 160ms ease;
  white-space: nowrap;
  height: 34px;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] {{
  margin: 0 !important;
  display: flex;
  align-items: center;
  height: 42px;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] > div {{
  margin: 0 !important;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] div[role="radiogroup"] label > div {{
  display: none !important;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] div[role="radiogroup"] label:hover {{
  background: var(--mc-button-hover-bg);
  border-color: var(--mc-button-border);
  transform: translateY(-1px);
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
  background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
  border-color: var(--mc-button-border);
  color: var(--mc-button-text) !important;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.06);
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] input {{
  display: none !important;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] div[role="radiogroup"] label span {{
  font-size: 0.75rem;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] div[role="radiogroup"] label::before {{
  content: "";
  width: 18px;
  height: 18px;
  display: inline-block;
  background: var(--mc-text-2);
  -webkit-mask-repeat: no-repeat;
  mask-repeat: no-repeat;
  -webkit-mask-position: center;
  mask-position: center;
  -webkit-mask-size: contain;
  mask-size: contain;
  transition: background 160ms ease;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] div[role="radiogroup"] label:nth-of-type(1)::before {{
  -webkit-mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><rect x='4' y='5' width='16' height='2' rx='1'/><rect x='4' y='11' width='16' height='2' rx='1'/><rect x='4' y='17' width='16' height='2' rx='1'/></svg>");
  mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><rect x='4' y='5' width='16' height='2' rx='1'/><rect x='4' y='11' width='16' height='2' rx='1'/><rect x='4' y='17' width='16' height='2' rx='1'/></svg>");
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] div[role="radiogroup"] label:nth-of-type(2)::before {{
  -webkit-mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><rect x='4' y='4' width='7' height='7' rx='1.5'/><rect x='13' y='4' width='7' height='7' rx='1.5'/><rect x='4' y='13' width='7' height='7' rx='1.5'/><rect x='13' y='13' width='7' height='7' rx='1.5'/></svg>");
  mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><rect x='4' y='4' width='7' height='7' rx='1.5'/><rect x='13' y='4' width='7' height='7' rx='1.5'/><rect x='4' y='13' width='7' height='7' rx='1.5'/><rect x='13' y='13' width='7' height='7' rx='1.5'/></svg>");
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked)::before {{
  background: var(--mc-button-text);
}}
</style>
""",
        unsafe_allow_html=True,
    )
    return cast(
        str,
        st.radio(
        "Vista",
        ["Tabla", "Carátulas"],
        horizontal=True,
        key="all_movies_view_mode",
        label_visibility="collapsed",
        format_func=lambda value: "Lista" if value == "Tabla" else "Mosaico",
        ),
    )


def _render_card_view(df_view: pd.DataFrame) -> None:
    _render_cards_styles()
    cards: list[str] = []
    for _, row in df_view.iterrows():
        row_map = cast(Mapping[str, Any], row)
        title_raw = row_map.get("title")
        title_raw_str = str(title_raw).strip() if title_raw not in (None, "") else ""
        if not title_raw_str:
            title_raw_str = "Sin título"
        title = html.escape(title_raw_str)
        year = _format_year(row_map.get("year"))
        header = f"{title} ({year})" if year else title
        header_text = f"{title_raw_str} ({year})" if year else title_raw_str
        library = _safe_text(row_map.get("library"), fallback="Sin biblioteca")
        poster_url = _poster_from_row(row_map)
        if poster_url:
            poster_html = f'<img src="{html.escape(poster_url)}" alt="{header}"/>'
        else:
            poster_html = "📷"

        size_gb = _format_float(row_map.get("file_size_gb"), digits=1)
        meta = _format_int(row_map.get("metacritic_score"))
        imdb = _format_float(row_map.get("imdb_rating"), digits=1)
        votes = _format_int(row_map.get("imdb_votes"))
        rt = _format_int(row_map.get("rt_score"))
        detail_query = _build_detail_query_for_row(row_map)

        card_html = f"""
<div class="mc-all-card">
  <div class="mc-all-card-poster">{poster_html}</div>
  <div class="mc-all-card-body">
    <div class="mc-all-card-title">{header}</div>
    <div class="mc-all-card-library">{library}</div>
    <div class="mc-all-card-stats">
      <div class="mc-all-card-stat"><span>GB</span><strong>{size_gb}</strong></div>
      <div class="mc-all-card-stat"><span>IMDb</span><strong>{imdb}</strong></div>
      <div class="mc-all-card-stat"><span>Meta</span><strong>{meta}</strong></div>
      <div class="mc-all-card-stat"><span>RT</span><strong>{rt}</strong></div>
      <div class="mc-all-card-stat full"><span>Votes</span><strong>{votes}</strong></div>
    </div>
  </div>
</div>
"""
        if detail_query:
            card_html = (
                f'<a class="mc-all-card-link" href="{detail_query}" '
                f'aria-label="Ver detalle de {html.escape(header_text)}">{card_html}</a>'
            )
        cards.append(card_html)
    st.markdown(
        f'<div class="mc-all-cards">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


_FILTER_BOUNDS_COLS: tuple[str, ...] = (
    "year",
    "imdb_rating",
    "imdb_votes",
    "metacritic_score",
    "rt_score",
    "file_size_gb",
)


def _build_filter_profile(df: pd.DataFrame) -> dict[str, object]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in _FILTER_BOUNDS_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        if not is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        series = series.dropna()
        if series.empty:
            continue
        bounds[col] = (float(series.min()), float(series.max()))
    return {"libraries": _safe_unique_sorted(df, "library"), "bounds": bounds}


def _get_filter_profile(df: pd.DataFrame) -> dict[str, object]:
    signature = dataframe_signature(df)
    cache = st.session_state.get("all_movies_filter_profile")
    if isinstance(cache, dict) and cache.get("sig") == signature:
        cached = cache.get("value")
        if isinstance(cached, dict):
            return cached
    profile = _build_filter_profile(df)
    st.session_state["all_movies_filter_profile"] = {"sig": signature, "value": profile}
    return profile


def _build_omdb_filter_profile(df: pd.DataFrame) -> dict[str, object]:
    if "omdb_json" not in df.columns and "imdb_id" not in df.columns:
        return {
            "genres": [],
            "directors": [],
            "genre_map": {},
            "director_map": {},
        }

    cols: list[str] = []
    if "omdb_json" in df.columns:
        cols.append("omdb_json")
    if "imdb_id" in df.columns:
        cols.append("imdb_id")
    df_base = df.loc[:, cols].copy()
    df_base["_row_id"] = df.index

    genre_map: dict[Any, list[str]] = {}
    genres_set: set[str] = set()
    try:
        df_gen = explode_genres_from_omdb_json(df_base)
        if "genre" in df_gen.columns:
            df_gen = df_gen[df_gen["genre"].notna() & (df_gen["genre"] != "")]
        if not df_gen.empty:
            grouped = df_gen.groupby("_row_id", dropna=False)["genre"].apply(list)
            genre_map = {
                idx: [g for g in genres if str(g).strip()]
                for idx, genres in grouped.items()
            }
            genres_set.update(
                str(g).strip()
                for g in df_gen["genre"].dropna().tolist()
                if str(g).strip()
            )
    except Exception:
        genre_map = {}
        genres_set = set()

    director_map: dict[Any, list[str]] = {}
    directors_set: set[str] = set()
    omdb_vals = (
        df_base["omdb_json"]
        if "omdb_json" in df_base.columns
        else pd.Series([None] * len(df_base), index=df_base.index)
    )
    imdb_vals = (
        df_base["imdb_id"]
        if "imdb_id" in df_base.columns
        else pd.Series([None] * len(df_base), index=df_base.index)
    )
    for row_id, omdb_raw, imdb_id in zip(df_base["_row_id"], omdb_vals, imdb_vals):
        directors = directors_from_omdb_json_or_cache(omdb_raw, imdb_id)
        if directors:
            director_map[row_id] = [d for d in directors if str(d).strip()]
            directors_set.update(str(d).strip() for d in directors if str(d).strip())

    return {
        "genres": sorted(genres_set),
        "directors": sorted(directors_set),
        "genre_map": genre_map,
        "director_map": director_map,
    }


def _get_omdb_filter_profile(df: pd.DataFrame) -> dict[str, object]:
    signature = dataframe_signature(df)
    cache = st.session_state.get("all_movies_omdb_filter_profile")
    if isinstance(cache, dict) and cache.get("sig") == signature:
        cached = cache.get("value")
        if isinstance(cached, dict):
            return cached
    profile = _build_omdb_filter_profile(df)
    st.session_state["all_movies_omdb_filter_profile"] = {
        "sig": signature,
        "value": profile,
    }
    return profile


def _apply_range_filter(
    df: pd.DataFrame,
    col: str,
    selected: tuple[float, float] | tuple[int, int] | None,
    default: tuple[float, float] | tuple[int, int] | None,
) -> pd.DataFrame:
    if selected is None or default is None:
        return df
    if col not in df.columns:
        return df
    if selected == default:
        return df

    series = pd.to_numeric(df[col], errors="coerce")
    return df[series.between(selected[0], selected[1], inclusive="both")]


def _apply_pending_filters(
    *,
    profile: dict[str, object],
) -> dict[str, tuple[float, float] | tuple[int, int]]:
    payload = st.session_state.pop("pending_all_movies_filters", None)
    if not isinstance(payload, dict):
        return {}

    pending_ranges: dict[str, tuple[float, float] | tuple[int, int]] = {}
    bounds = profile.get("bounds")
    libraries = profile.get("libraries")
    if not isinstance(bounds, dict):
        bounds = {}
    if not isinstance(libraries, list):
        libraries = []

    def _set_range(
        *,
        key: str,
        col: str,
        cast_int: bool = False,
        min_key: str,
        max_key: str,
    ) -> None:
        if col not in bounds:
            return
        raw_min = payload.get(min_key)
        raw_max = payload.get(max_key)
        if raw_min is None and raw_max is None:
            return
        try:
            min_b, max_b = bounds[col]
        except Exception:
            return
        lo = float(min_b)
        hi = float(max_b)
        try:
            if raw_min is not None:
                lo = max(lo, float(raw_min))
            if raw_max is not None:
                hi = min(hi, float(raw_max))
        except Exception:
            return
        if lo > hi:
            lo, hi = hi, lo
        if cast_int:
            lo_i = int(round(lo))
            hi_i = int(round(hi))
            pending_ranges[col] = (lo_i, hi_i)
            st.session_state[key] = (lo_i, hi_i)
        else:
            pending_ranges[col] = (lo, hi)
            st.session_state[key] = (lo, hi)

    libraries_payload = payload.get("libraries")
    if isinstance(libraries_payload, list):
        wanted = [str(item) for item in libraries_payload if str(item).strip()]
        if libraries:
            wanted = [lib for lib in wanted if lib in libraries]
        if wanted:
            st.session_state["lib_filter_all_movies"] = wanted

    decisions_payload = payload.get("decisions")
    if isinstance(decisions_payload, list):
        decisions = [str(item).strip().upper() for item in decisions_payload if item]
        allowed = {"DELETE", "MAYBE", "KEEP", "UNKNOWN"}
        decisions = [d for d in decisions if d in allowed]
        if decisions:
            st.session_state["dec_filter_all_movies"] = decisions

    title_payload = payload.get("title")
    if isinstance(title_payload, str) and title_payload.strip():
        st.session_state["title_filter_all_movies"] = title_payload.strip()

    genres_payload = payload.get("genres")
    if isinstance(genres_payload, list):
        wanted = [str(item) for item in genres_payload if str(item).strip()]
        if wanted:
            st.session_state["genre_filter_all_movies"] = wanted

    directors_payload = payload.get("directors")
    if isinstance(directors_payload, list):
        wanted = [str(item) for item in directors_payload if str(item).strip()]
        if wanted:
            st.session_state["director_filter_all_movies"] = wanted

    _set_range(
        key="year_range_all_movies",
        col="year",
        cast_int=True,
        min_key="year_min",
        max_key="year_max",
    )
    _set_range(
        key="imdb_range_all_movies",
        col="imdb_rating",
        cast_int=False,
        min_key="imdb_min",
        max_key="imdb_max",
    )
    _set_range(
        key="votes_range_all_movies",
        col="imdb_votes",
        cast_int=True,
        min_key="votes_min",
        max_key="votes_max",
    )
    _set_range(
        key="metacritic_range_all_movies",
        col="metacritic_score",
        cast_int=False,
        min_key="meta_min",
        max_key="meta_max",
    )
    _set_range(
        key="rt_range_all_movies",
        col="rt_score",
        cast_int=False,
        min_key="rt_min",
        max_key="rt_max",
    )
    _set_range(
        key="size_range_all_movies",
        col="file_size_gb",
        cast_int=False,
        min_key="size_min",
        max_key="size_max",
    )

    return pending_ranges


# ============================================================================
# Render
# ============================================================================


def render(df_all: pd.DataFrame) -> None:
    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        st.info("No hay datos para mostrar.")
        return

    df_view = df_all

    title_query = ""
    year_range: tuple[int, int] | None = None
    year_default: tuple[int, int] | None = None
    imdb_range: tuple[float, float] | None = None
    imdb_default: tuple[float, float] | None = None
    votes_range: tuple[int, int] | None = None
    votes_default: tuple[int, int] | None = None
    metacritic_range: tuple[float, float] | None = None
    metacritic_default: tuple[float, float] | None = None
    rt_range: tuple[float, float] | None = None
    rt_default: tuple[float, float] | None = None
    size_range: tuple[float, float] | None = None
    size_default: tuple[float, float] | None = None

    col_f1, col_f2, col_f3 = st.columns([1, 1, 1.4])

    profile = _get_filter_profile(df_all)
    libraries = profile.get("libraries")
    bounds = profile.get("bounds")
    if not isinstance(libraries, list):
        libraries = _safe_unique_sorted(df_all, "library")
    if not isinstance(bounds, dict):
        bounds = {}

    pending_ranges = _apply_pending_filters(profile=profile)
    show_numeric_filters = get_show_numeric_filters()
    omdb_profile = _get_omdb_filter_profile(df_all)
    genres = omdb_profile.get("genres", [])
    directors = omdb_profile.get("directors", [])
    genre_map = omdb_profile.get("genre_map", {})
    director_map = omdb_profile.get("director_map", {})
    if isinstance(genres, list) and genres:
        current_genres = st.session_state.get("genre_filter_all_movies")
        if isinstance(current_genres, list):
            st.session_state["genre_filter_all_movies"] = [
                g for g in current_genres if g in genres
            ]
    if isinstance(directors, list) and directors:
        current_directors = st.session_state.get("director_filter_all_movies")
        if isinstance(current_directors, list):
            st.session_state["director_filter_all_movies"] = [
                d for d in current_directors if d in directors
            ]

    with col_f1:
        lib_filter: Sequence[str] = st.multiselect(
            "Biblioteca",
            libraries,
            key="lib_filter_all_movies",
        )

    with col_f2:
        decisions = ["DELETE", "MAYBE", "KEEP", "UNKNOWN"]
        dec_filter: Sequence[str] = st.multiselect(
            "Decisión",
            decisions,
            format_func=_decision_label,
            key="dec_filter_all_movies",
        )
        colorize = bool(st.session_state.get("grid_colorize_rows", True))
        render_decision_chip_styles(
            "decision-chips-all-movies",
            enabled=colorize,
            selected_values=list(dec_filter),
        )

    with col_f3:
        title_query = st.text_input("Título contiene", key="title_filter_all_movies")

    if show_numeric_filters:
        with st.expander("Filtros numéricos", expanded=False):
            col_s1, col_s2 = st.columns(2)

            with col_s1:
                year_bounds = bounds.get("year")
                if year_bounds is not None:
                    min_year = int(year_bounds[0])
                    max_year = int(year_bounds[1])
                    year_default = (min_year, max_year)
                    if min_year == max_year:
                        st.caption(f"Año: {min_year}")
                        year_range = year_default
                    else:
                        year_range = st.slider(
                            "Año",
                            min_year,
                            max_year,
                            year_default,
                            step=1,
                            key="year_range_all_movies",
                        )

                imdb_bounds = bounds.get("imdb_rating")
                if imdb_bounds is not None:
                    min_imdb, max_imdb = imdb_bounds
                    imdb_default = (min_imdb, max_imdb)
                    if min_imdb == max_imdb:
                        st.caption(f"IMDb: {min_imdb:.1f}")
                        imdb_range = imdb_default
                    else:
                        imdb_range = st.slider(
                            "IMDb",
                            min_imdb,
                            max_imdb,
                            imdb_default,
                            step=0.1,
                            format="%.1f",
                            key="imdb_range_all_movies",
                        )

                votes_bounds = bounds.get("imdb_votes")
                if votes_bounds is not None:
                    min_votes = int(votes_bounds[0])
                    max_votes = int(votes_bounds[1])
                    votes_default = (min_votes, max_votes)
                    step = max(1, int(round((max_votes - min_votes) / 100)))
                    if min_votes == max_votes:
                        st.caption(f"IMDb votos: {min_votes}")
                        votes_range = votes_default
                    else:
                        votes_range = st.slider(
                            "IMDb votos",
                            min_votes,
                            max_votes,
                            votes_default,
                            step=step,
                            format="%d",
                            key="votes_range_all_movies",
                        )

            with col_s2:
                metacritic_bounds = bounds.get("metacritic_score")
                if metacritic_bounds is not None:
                    min_meta, max_meta = metacritic_bounds
                    metacritic_default = (min_meta, max_meta)
                    if min_meta == max_meta:
                        st.caption(f"Metacritic: {min_meta:.0f}")
                        metacritic_range = metacritic_default
                    else:
                        metacritic_range = st.slider(
                            "Metacritic",
                            min_meta,
                            max_meta,
                            metacritic_default,
                            step=1.0,
                            format="%.0f",
                            key="metacritic_range_all_movies",
                        )

                rt_bounds = bounds.get("rt_score")
                if rt_bounds is not None:
                    min_rt, max_rt = rt_bounds
                    rt_default = (min_rt, max_rt)
                    if min_rt == max_rt:
                        st.caption(f"RT: {min_rt:.0f}")
                        rt_range = rt_default
                    else:
                        rt_range = st.slider(
                            "RT",
                            min_rt,
                            max_rt,
                            rt_default,
                            step=1.0,
                            format="%.0f",
                            key="rt_range_all_movies",
                        )

                size_bounds = bounds.get("file_size_gb")
                if size_bounds is not None:
                    min_size, max_size = size_bounds
                    size_default = (min_size, max_size)
                    if min_size == max_size:
                        st.caption(f"Tamaño (GB): {min_size:.2f}")
                        size_range = size_default
                    else:
                        size_range = st.slider(
                            "Tamaño (GB)",
                            min_size,
                            max_size,
                            size_default,
                            step=0.1,
                            format="%.1f",
                            key="size_range_all_movies",
                        )

    omdb_expand = bool(
        st.session_state.get("genre_filter_all_movies")
        or st.session_state.get("director_filter_all_movies")
    )
    if isinstance(genres, list) or isinstance(directors, list):
        if genres or directors:
            with st.expander("Filtros OMDb", expanded=omdb_expand):
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    if isinstance(genres, list) and genres:
                        st.multiselect(
                            "Género",
                            genres,
                            key="genre_filter_all_movies",
                        )
                with col_g2:
                    if isinstance(directors, list) and directors:
                        st.multiselect(
                            "Director",
                            directors,
                            key="director_filter_all_movies",
                        )

    if pending_ranges:
        if year_range is None:
            year_range = cast(tuple[int, int] | None, pending_ranges.get("year"))
        if imdb_range is None:
            imdb_range = cast(
                tuple[float, float] | None, pending_ranges.get("imdb_rating")
            )
        if votes_range is None:
            votes_range = cast(tuple[int, int] | None, pending_ranges.get("imdb_votes"))
        if metacritic_range is None:
            metacritic_range = cast(
                tuple[float, float] | None, pending_ranges.get("metacritic_score")
            )
        if rt_range is None:
            rt_range = cast(tuple[float, float] | None, pending_ranges.get("rt_score"))
        if size_range is None:
            size_range = cast(
                tuple[float, float] | None, pending_ranges.get("file_size_gb")
            )
        if not show_numeric_filters:
            if year_default is None and bounds.get("year") is not None:
                min_year, max_year = bounds["year"]
                year_default = (int(min_year), int(max_year))
            if imdb_default is None and bounds.get("imdb_rating") is not None:
                min_imdb, max_imdb = bounds["imdb_rating"]
                imdb_default = (float(min_imdb), float(max_imdb))
            if votes_default is None and bounds.get("imdb_votes") is not None:
                min_votes, max_votes = bounds["imdb_votes"]
                votes_default = (int(min_votes), int(max_votes))
            if (
                metacritic_default is None
                and bounds.get("metacritic_score") is not None
            ):
                min_meta, max_meta = bounds["metacritic_score"]
                metacritic_default = (float(min_meta), float(max_meta))
            if rt_default is None and bounds.get("rt_score") is not None:
                min_rt, max_rt = bounds["rt_score"]
                rt_default = (float(min_rt), float(max_rt))
            if size_default is None and bounds.get("file_size_gb") is not None:
                min_size, max_size = bounds["file_size_gb"]
                size_default = (float(min_size), float(max_size))

    if lib_filter and "library" in df_view.columns:
        df_view = df_view[df_view["library"].isin(lib_filter)]

    if dec_filter and "decision" in df_view.columns:
        df_view = df_view[df_view["decision"].isin(dec_filter)]

    title_query = title_query.strip()
    if title_query and "title" in df_view.columns:
        title_series = df_view["title"].fillna("").astype(str)
        df_view = df_view[
            title_series.str.contains(title_query, case=False, regex=False, na=False)
        ]

    genre_filter = st.session_state.get("genre_filter_all_movies")
    if isinstance(genre_filter, (list, tuple)) and genre_filter:
        genre_set = {str(g) for g in genre_filter if str(g).strip()}
        if genre_set and isinstance(genre_map, dict):
            idx_series = df_view.index.to_series()
            mask = idx_series.map(
                lambda idx: bool(
                    genre_set.intersection(
                        set(genre_map.get(idx, [])) if genre_map.get(idx) else set()
                    )
                )
            )
            df_view = df_view[mask.values]

    director_filter = st.session_state.get("director_filter_all_movies")
    if isinstance(director_filter, (list, tuple)) and director_filter:
        director_set = {str(d) for d in director_filter if str(d).strip()}
        if director_set and isinstance(director_map, dict):
            idx_series = df_view.index.to_series()
            mask = idx_series.map(
                lambda idx: bool(
                    director_set.intersection(
                        set(director_map.get(idx, []))
                        if director_map.get(idx)
                        else set()
                    )
                )
            )
            df_view = df_view[mask.values]

    df_view = _apply_range_filter(df_view, "year", year_range, year_default)
    df_view = _apply_range_filter(df_view, "imdb_rating", imdb_range, imdb_default)
    df_view = _apply_range_filter(df_view, "imdb_votes", votes_range, votes_default)
    df_view = _apply_range_filter(
        df_view, "metacritic_score", metacritic_range, metacritic_default
    )
    df_view = _apply_range_filter(df_view, "rt_score", rt_range, rt_default)
    df_view = _apply_range_filter(df_view, "file_size_gb", size_range, size_default)

    if "file_size_gb" in df_view.columns:
        size_series = pd.to_numeric(df_view["file_size_gb"], errors="coerce")
        df_view = (
            df_view.assign(_sort_size=size_series)
            .sort_values("_sort_size", ascending=False, na_position="last")
            .drop(columns=["_sort_size"])
        )
    initial_sort_model: list[dict[str, Any]] | None = None
    if "file_size_gb" in df_view.columns:
        initial_sort_model = [{"colId": "file_size_gb", "sort": "desc"}]

    if df_view.empty:
        st.info("No hay resultados que coincidan con los filtros actuales.")
        return

    def _results_caption(count: int, _total: int, _has_search: bool) -> str:
        return f"Resultados: {count} película(s)"

    def _render_toggle_inline() -> None:
        _render_view_toggle()

    df_view, search_query, grid_height, download_slot = render_grid_toolbar(
        df_view,
        key_suffix="all_movies",
        download_filename="all_movies.csv",
        caption_builder=_results_caption,
        return_download_slot=True,
        right_extras=_render_toggle_inline,
        inline_search=True,
    )
    view_mode = str(st.session_state.get("all_movies_view_mode", "Tabla"))

    if df_view.empty:
        if search_query.strip():
            st.info("No hay filas que coincidan con la búsqueda.")
        else:
            st.info("No hay resultados que coincidan con los filtros actuales.")
        return

    if view_mode == "Carátulas":
        if download_slot is not None:
            csv_export = df_view.to_csv(index=False).encode("utf-8")
            download_slot.download_button(
                "⬇️",
                data=csv_export,
                file_name="all_movies.csv",
                mime="text/csv",
                key="grid_download_all_movies",
                help="Descargar CSV",
            )
        _render_card_view(df_view)
        return

    col_grid, col_detail = st.columns([2, 1])

    with col_grid:
        selected_row = aggrid_with_row_click(
            df_view,
            "all_movies",
            visible_order=[
                "title",
                "year",
                "library",
                "file_size_gb",
                "metacritic_score",
                "imdb_rating",
                "imdb_votes",
                "rt_score",
            ],
            initial_sort_model=initial_sort_model,
            auto_select_first=True,
            toolbar_caption_builder=_results_caption,
            show_toolbar=False,
            toolbar_state=(df_view, search_query, grid_height, download_slot),
        )

    with col_detail:
        render_detail_card(selected_row)
