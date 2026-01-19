"""
components.py

Componentes UI del dashboard (Streamlit + st_aggrid).

Objetivos
- Mostrar una tabla (AgGrid) optimizada para selección de una sola fila.
- Renderizar una “ficha” de detalle estilo Plex a partir de la fila seleccionada.
- Soportar un “modal” (detalle ampliado) usando st.session_state.

Principios
- Robustez frente a datos “raros” (NaN, None, tipos inesperados, Series/dict/DataFrame).
- Parseo de JSON (omdb_json) bajo demanda (solo para la fila en detalle).
- Keys únicas en widgets para evitar colisiones entre pestañas/vistas.

Notas de compatibilidad (st_aggrid)
- Se usa update_on=["selectionChanged"] (en lugar de GridUpdateMode.*).
- Se aplica autoSizeStrategy recomendado por st_aggrid.

Nota importante (Pyright/Pylance)
- Evitamos falsos positivos "Statement is unreachable" causados por stubs rotos
  (p.ej. pandas/streamlit) que marcan funciones como NoReturn o devuelven tipos demasiado
  estrechos. Para ello:
  - Evitamos ramas que Pyright infiere como imposibles (p.ej., DataFrame.to_dict(records)
    tipado como list[dict[...]]: no metemos un `else` “imposible”).
  - Llamamos a pd.isna mediante wrapper con Any para romper propagación de stubs raros.
"""

from __future__ import annotations

import html
import json
import re
from urllib.parse import quote
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from typing import Any, Callable, Literal, Optional, Protocol, TypeVar, cast

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from st_aggrid import GridOptionsBuilder, JsCode

from frontend.config_front_artifacts import OMDB_CACHE_PATH, WIKI_CACHE_PATH
from frontend.config_front_theme import (
    get_front_theme,
    is_dark_theme,
    normalize_theme_key,
)
from frontend.data_utils import safe_json_loads_single

RowDict = dict[str, Any]
GridCaptionBuilder = Callable[[int, int, bool], str | None]

_DECISION_ROW_STYLE = JsCode(
    """
function(params) {
  if (!params || !params.data) {
    return {};
  }
  const raw = params.data.decision;
  if (!raw) {
    return {};
  }
  const d = String(raw).toUpperCase();
  if (d === "DELETE") return { color: "var(--mc-decision-delete)" };
  if (d === "KEEP") return { color: "var(--mc-decision-keep)" };
  if (d === "MAYBE") return { color: "var(--mc-decision-maybe)" };
  if (d === "UNKNOWN") return { color: "var(--mc-decision-unknown)" };
  return {};
}
"""
)


class AgGridCallable(Protocol):
    def __call__(
        self,
        data: pd.DataFrame,
        *,
        gridOptions: Mapping[str, Any],
        update_on: Sequence[str],
        enable_enterprise_modules: bool,
        theme: str | None = None,
        custom_css: Mapping[str, Mapping[str, str]] | None = None,
        height: int,
        allow_unsafe_jscode: bool | None = None,
        key: str,
    ) -> Mapping[str, Any]: ...


# ============================================================================
# Compatibilidad rerun (Streamlit)
# ============================================================================


def _rerun() -> None:
    """
    Compatibilidad Streamlit:
    - Streamlit nuevo: st.rerun()
    - Streamlit antiguo: st.experimental_rerun()
    """
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return

    exp_rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(exp_rerun_fn):
        exp_rerun_fn()


def _columns_with_gap(
    spec: Sequence[int], *, gap: Literal["small", "medium", "large"]
) -> Sequence[Any]:
    try:
        return cast(Sequence[Any], st.columns(spec, gap=gap))
    except TypeError:
        return cast(Sequence[Any], st.columns(spec))


def _current_theme_key() -> str:
    raw = st.session_state.get("front_theme")
    fallback = get_front_theme()
    return normalize_theme_key(raw if isinstance(raw, str) else fallback)


def _aggrid_theme() -> str:
    key = _current_theme_key()
    return "alpine-dark" if is_dark_theme(key) else "alpine"


def _theme_tokens() -> dict[str, str]:
    raw = st.session_state.get("front_theme_tokens")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        out[str(k)] = str(v)
    return out


def _aggrid_custom_css() -> dict[str, dict[str, str]]:
    tokens = _theme_tokens()
    if not tokens:
        return {}

    def _token(key: str, fallback: str) -> str:
        return tokens.get(key, fallback)

    bg = _token("card_bg", "#11161f")
    header_bg = _token("button_bg", "#171b24")
    border = _token("panel_border", "#1f2532")
    text = _token("text_2", "#d1d5db")
    text_strong = _token("text_1", "#f1f5f9")
    row_alt = _token("metric_bg", "#111722")
    hover = _token("tag_bg", "#2b2f36")
    selected = _token("button_hover_bg", "#202635")
    dec_delete = _token("decision_delete", "#e53935")
    dec_keep = _token("decision_keep", "#43a047")
    dec_maybe = _token("decision_maybe", "#fbc02d")
    dec_unknown = _token("decision_unknown", "#9e9e9e")

    return {
        ".ag-root-wrapper": {
            "background-color": f"{bg} !important",
            "border": f"1px solid {border} !important",
            "border-radius": "12px",
            "color": text,
            "--mc-decision-delete": dec_delete,
            "--mc-decision-keep": dec_keep,
            "--mc-decision-maybe": dec_maybe,
            "--mc-decision-unknown": dec_unknown,
        },
        ".ag-body": {
            "background-color": f"{bg} !important",
        },
        ".ag-viewport": {
            "background-color": f"{bg} !important",
        },
        ".ag-body-viewport": {
            "background-color": f"{bg} !important",
        },
        ".ag-center-cols-viewport": {
            "background-color": f"{bg} !important",
        },
        ".ag-center-cols-container": {
            "background-color": f"{bg} !important",
        },
        ".ag-body-horizontal-scroll-viewport": {
            "background-color": f"{bg} !important",
        },
        ".ag-body-vertical-scroll-viewport": {
            "background-color": f"{bg} !important",
        },
        ".ag-header": {
            "background-color": f"{header_bg} !important",
            "border-bottom": f"1px solid {border} !important",
            "color": f"{text_strong} !important",
        },
        ".ag-header-cell, .ag-header-group-cell": {
            "background-color": f"{header_bg} !important",
            "color": f"{text_strong} !important",
            "border-color": f"{border} !important",
        },
        ".ag-row": {
            "background-color": f"{bg} !important",
            "color": text,
        },
        ".ag-row-odd": {
            "background-color": f"{row_alt} !important",
        },
        ".ag-row-hover": {
            "background-color": f"{hover} !important",
        },
        ".ag-row-selected": {
            "background-color": f"{selected} !important",
        },
        ".ag-cell": {
            "border-color": f"{border} !important",
            "color": "inherit !important",
        },
        ".ag-icon": {
            "color": f"{text_strong} !important",
            "fill": f"{text_strong} !important",
        },
        ".ag-paging-panel": {
            "color": f"{text} !important",
        },
    }


# Apply decision colors to multiselect chips inside a scoped container.
def render_decision_chip_styles(
    anchor_id: str,
    *,
    enabled: bool = True,
    selected_values: Sequence[str] | None = None,
) -> None:
    st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
    selected_payload = json.dumps([str(v).upper() for v in selected_values or []])
    components.html(
        f"""
<script>
(() => {{
  const win = window.parent;
  if (!win || !win.document) return;
  const doc = win.document;
  const enabled = {str(enabled).lower()};
  const selectedValues = {selected_payload};
  const anchor = doc.getElementById("{anchor_id}");
  if (!anchor) return;
  const root = anchor.closest('[data-testid="stVerticalBlock"]') || anchor.parentElement;
  if (!root) return;
  const widget = anchor.closest('div[data-testid="stMultiSelect"]') ||
    (anchor.closest('div[data-testid="column"]') ||
      anchor.closest('div[data-testid="stVerticalBlock"]'))?.querySelector('div[data-testid="stMultiSelect"]');
  const scope = widget || root;
  win.__mcDecisionTagObservers = win.__mcDecisionTagObservers || {{}};
  const existing = win.__mcDecisionTagObservers["{anchor_id}"];
  const palette = () => {{
    const styles = win.getComputedStyle(doc.documentElement);
    return {{
      DELETE: styles.getPropertyValue("--mc-decision-delete").trim() || "#e53935",
      KEEP: styles.getPropertyValue("--mc-decision-keep").trim() || "#43a047",
      MAYBE: styles.getPropertyValue("--mc-decision-maybe").trim() || "#fbc02d",
      UNKNOWN: styles.getPropertyValue("--mc-decision-unknown").trim() || "#9e9e9e",
    }};
  }};
  const parseColor = (value) => {{
    const v = (value || "").trim();
    if (!v) return null;
    if (v.startsWith("#")) {{
      let hex = v.slice(1);
      if (hex.length === 3) {{
        hex = hex.split("").map(ch => ch + ch).join("");
      }}
      if (hex.length === 6) {{
        const r = parseInt(hex.slice(0, 2), 16);
        const g = parseInt(hex.slice(2, 4), 16);
        const b = parseInt(hex.slice(4, 6), 16);
        return [r, g, b];
      }}
      return null;
    }}
    const match = v.match(/rgba?\\(([^)]+)\\)/i);
    if (match) {{
      const parts = match[1].split(",").map(p => parseFloat(p.trim()));
      if (parts.length >= 3 && parts.every(n => Number.isFinite(n))) {{
        return [parts[0], parts[1], parts[2]];
      }}
    }}
    return null;
  }};
  const textColorFor = (color) => {{
    const rgb = parseColor(color);
    if (!rgb) return "#0b1220";
    const [r, g, b] = rgb.map(v => v / 255);
    const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return luminance > 0.6 ? "#0b1220" : "#f8fafc";
  }};
  const keyFromLabel = (label) => {{
    if (!label) return null;
    const match = label.match(/\\b(DELETE|KEEP|MAYBE|UNKNOWN)\\b/);
    return match ? match[1] : null;
  }};
  const clearTagStyles = (tag) => {{
    tag.style.removeProperty("background-color");
    tag.style.removeProperty("border-color");
    tag.style.removeProperty("color");
    tag.querySelectorAll("span, svg, button").forEach((el) => {{
      el.style.removeProperty("color");
      el.style.removeProperty("fill");
    }});
  }};
  const applyTagStyles = (tag, color, textColor) => {{
    tag.style.setProperty("background-color", color, "important");
    tag.style.setProperty("border-color", color, "important");
    tag.style.setProperty("color", textColor, "important");
    tag.querySelectorAll("span, svg, button").forEach((el) => {{
      el.style.setProperty("color", textColor, "important");
      el.style.setProperty("fill", textColor, "important");
    }});
  }};
  const optionTarget = (option) => {{
    return option.querySelector('[data-baseweb="menu-item"]') || option;
  }};
  const findListbox = () => {{
    const combobox = scope.querySelector('[role="combobox"]');
    if (!combobox) return null;
    const listboxId =
      combobox.getAttribute("aria-controls") || combobox.getAttribute("aria-owns");
    if (!listboxId) return null;
    return doc.getElementById(listboxId);
  }};
  const ensureOptionMarker = (target, color) => {{
    let marker = target.querySelector('[data-mc-decision-marker="true"]');
    if (!marker) {{
      marker = doc.createElement("span");
      marker.setAttribute("data-mc-decision-marker", "true");
      marker.setAttribute("aria-hidden", "true");
      marker.style.display = "inline-block";
      marker.style.width = "8px";
      marker.style.height = "8px";
      marker.style.borderRadius = "2px";
      marker.style.marginRight = "8px";
      marker.style.flex = "0 0 auto";
      target.insertBefore(marker, target.firstChild);
    }}
    marker.style.backgroundColor = color;
    return marker;
  }};
  const removeOptionMarker = (target) => {{
    const marker = target.querySelector('[data-mc-decision-marker="true"]');
    if (marker && marker.parentElement) {{
      marker.parentElement.removeChild(marker);
    }}
  }};
  const clearOptionStyles = (option) => {{
    const target = optionTarget(option);
    target.style.removeProperty("background-image");
    target.style.removeProperty("background-repeat");
    target.style.removeProperty("background-size");
    target.style.removeProperty("background-position");
    target.style.removeProperty("padding-left");
    target.style.removeProperty("color");
    option.style.removeProperty("color");
    removeOptionMarker(target);
  }};
  const applyOptionStyles = (option, color) => {{
    const target = optionTarget(option);
    ensureOptionMarker(target, color);
    target.style.setProperty("display", "flex", "important");
    target.style.setProperty("align-items", "center", "important");
    target.style.setProperty("color", "var(--mc-text-1)", "important");
    option.style.setProperty("color", "var(--mc-text-1)", "important");
  }};
  const collectOptions = () => {{
    const listbox = findListbox();
    if (listbox) {{
      return Array.from(listbox.querySelectorAll('[role="option"]'));
    }}
    const options = new Set();
    doc.querySelectorAll('[role="option"]').forEach((node) => options.add(node));
    doc
      .querySelectorAll('[data-baseweb="menu"] li, [data-baseweb="menu"] [role="option"]')
      .forEach((node) => options.add(node));
    doc
      .querySelectorAll('[data-baseweb="popover"] [role="option"]')
      .forEach((node) => options.add(node));
    return Array.from(options);
  }};
  const apply = () => {{
    const colors = palette();
    const tags = scope.querySelectorAll('[data-baseweb="tag"]');
    const tagKeys =
      selectedValues.length && selectedValues.length === tags.length
        ? selectedValues
        : null;
    Array.from(tags).forEach((tag, idx) => {{
      const label = (tag.textContent || tag.innerText || "").toUpperCase();
      const key = tagKeys ? tagKeys[idx] : keyFromLabel(label);
      if (!key) return;
      if (!enabled) {{
        clearTagStyles(tag);
        return;
      }}
      const color = colors[key] || colors.UNKNOWN;
      const textColor = textColorFor(color);
      applyTagStyles(tag, color, textColor);
    }});
    collectOptions().forEach((option) => {{
      const label = (option.textContent || option.innerText || "").toUpperCase();
      const key = keyFromLabel(label);
      if (!key) return;
      if (!enabled) {{
        clearOptionStyles(option);
        return;
      }}
      const color = colors[key] || colors.UNKNOWN;
      applyOptionStyles(option, color);
    }});
  }};
  const scheduleApply = () => {{
    let tries = 0;
    const tick = () => {{
      apply();
      tries += 1;
      if (tries < 10) {{
        win.setTimeout(tick, 80);
      }}
    }};
    tick();
  }};
  if (existing && existing.root && typeof existing.root.disconnect === "function") {{
    existing.root.disconnect();
  }}
  if (existing && existing.body && typeof existing.body.disconnect === "function") {{
    existing.body.disconnect();
  }}
  if (existing && existing.listener) {{
    doc.removeEventListener("click", existing.listener, true);
    doc.removeEventListener("keydown", existing.listener, true);
    doc.removeEventListener("focusin", existing.listener, true);
  }}
  const rootObserver = new MutationObserver(apply);
  rootObserver.observe(root, {{ childList: true, subtree: true }});
  const bodyObserver = new MutationObserver(apply);
  if (doc.body) {{
    bodyObserver.observe(doc.body, {{ childList: true, subtree: true }});
  }}
  const listener = () => scheduleApply();
  doc.addEventListener("click", listener, true);
  doc.addEventListener("keydown", listener, true);
  doc.addEventListener("focusin", listener, true);
  win.__mcDecisionTagObservers["{anchor_id}"] = {{
    root: rootObserver,
    body: bodyObserver,
    listener: listener,
  }};
  scheduleApply();
}})();
</script>
""",
        height=0,
        width=0,
    )


# ============================================================================
# SAFE wrappers (anti pyright unreachable por stubs)
# ============================================================================


def _pd_isna(value: Any) -> bool:
    """
    Wrapper para pd.isna() evitando que stubs rotos propaguen NoReturn/unreachable.
    """
    pd_any: Any = pd
    fn_obj: object = getattr(pd_any, "isna", None)
    if not callable(fn_obj):
        return False
    try:
        out = cast(Any, fn_obj)(value)
        return bool(out)
    except Exception:
        return False


def _safe_year_suffix(year: Any) -> str:
    """
    Devuelve " (YYYY)" si year es usable; "" si no.
    """
    if year is None:
        return ""
    if _pd_isna(year):
        return ""
    try:
        return f" ({int(float(year))})"
    except Exception:
        return ""


# ============================================================================
# Normalización de filas
# ============================================================================


def _to_str_key_dict(src: Mapping[Hashable, Any]) -> RowDict:
    out: RowDict = {}
    for k, v in src.items():
        out[str(k)] = v
    return out


def _normalize_selected_rows(selected_raw: Any) -> list[RowDict]:
    """
    Normaliza el objeto devuelto por AgGrid a una lista de dict[str, Any].

    Devuelve siempre una lista; si no hay selección, lista vacía.
    """
    if selected_raw is None:
        return []

    # DataFrame -> list[dict]
    if isinstance(selected_raw, pd.DataFrame):
        # En los stubs típicos, esto sale como list[dict[str, Any]]
        records = selected_raw.to_dict(orient="records")
        out_df: list[RowDict] = []
        for rec in records:
            # FIX pyright unreachable: NO usar else aquí (pyright infiere rec como dict).
            # Aun así, mantenemos robustez runtime usando Any + try/except,
            # y hacemos el append UNA SOLA VEZ fuera del try/except.
            rec_any: Any = rec
            row_out_df: RowDict
            try:
                row_out_df = _to_str_key_dict(cast(Mapping[Hashable, Any], rec_any))
            except Exception:
                row_out_df = {"value": rec_any}
            out_df.append(row_out_df)
        return out_df

    # Series -> dict (en pandas real siempre Mapping)
    if isinstance(selected_raw, pd.Series):
        d_map = selected_raw.to_dict()
        return [_to_str_key_dict(cast(Mapping[Hashable, Any], d_map))]

    # Mapping -> una sola fila
    if isinstance(selected_raw, Mapping):
        return [_to_str_key_dict(cast(Mapping[Hashable, Any], selected_raw))]

    # list/tuple -> lista de filas
    if isinstance(selected_raw, (list, tuple)):
        out_list: list[RowDict] = []
        for item in selected_raw:
            if isinstance(item, pd.Series):
                out_list.append(
                    _to_str_key_dict(cast(Mapping[Hashable, Any], item.to_dict()))
                )
                continue
            if isinstance(item, Mapping):
                out_list.append(_to_str_key_dict(cast(Mapping[Hashable, Any], item)))
                continue

            tmp_obj: object
            try:
                tmp_obj = dict(item)  # type: ignore[arg-type]
            except Exception:
                tmp_obj = item

            if isinstance(tmp_obj, Mapping):
                out_list.append(_to_str_key_dict(cast(Mapping[Hashable, Any], tmp_obj)))
            else:
                out_list.append({"value": tmp_obj})

        return out_list

    # iterable genérico (no str/bytes)
    if isinstance(selected_raw, Iterable) and not isinstance(
        selected_raw, (str, bytes)
    ):
        out_it: list[RowDict] = []
        for x in selected_raw:
            if isinstance(x, pd.Series):
                out_it.append(
                    _to_str_key_dict(cast(Mapping[Hashable, Any], x.to_dict()))
                )
                continue
            if isinstance(x, Mapping):
                out_it.append(_to_str_key_dict(cast(Mapping[Hashable, Any], x)))
                continue

            tmp_obj2: object
            try:
                tmp_obj2 = dict(x)  # type: ignore[arg-type]
            except Exception:
                tmp_obj2 = x

            if isinstance(tmp_obj2, Mapping):
                out_it.append(_to_str_key_dict(cast(Mapping[Hashable, Any], tmp_obj2)))
            else:
                out_it.append({"value": tmp_obj2})

        return out_it

    return [{"value": selected_raw}]


def _normalize_row_to_dict(row: Any) -> Optional[RowDict]:
    """
    Convierte distintas formas de fila (Series, dict, etc.) a dict[str, Any].
    """
    if row is None:
        return None

    if isinstance(row, pd.Series):
        d_map = row.to_dict()
        return _to_str_key_dict(cast(Mapping[Hashable, Any], d_map))

    if isinstance(row, Mapping):
        return _to_str_key_dict(cast(Mapping[Hashable, Any], row))

    d2: object | None
    try:
        d2 = dict(row)  # type: ignore[arg-type]
    except Exception:
        d2 = None

    if isinstance(d2, Mapping):
        return _to_str_key_dict(cast(Mapping[Hashable, Any], d2))

    return None


def _value_contains_query(value: Any, query_lc: str) -> bool:
    if value is None:
        return False
    try:
        return query_lc in str(value).lower()
    except Exception:
        return False


def _json_contains_terms(value: Any, terms_lc: Sequence[str]) -> bool:
    if not terms_lc:
        return False
    pending = set(terms_lc)
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, Mapping):
            for k, v in current.items():
                if k is not None:
                    k_lc = str(k).lower()
                    pending = {t for t in pending if t not in k_lc}
                    if not pending:
                        return True
                stack.append(v)
            continue
        if isinstance(current, (list, tuple, set)):
            stack.extend(list(current))
            continue
        try:
            current_lc = str(current).lower()
        except Exception:
            continue
        pending = {t for t in pending if t not in current_lc}
        if not pending:
            return True
    return False


def _row_matches_detail_json(
    row: Mapping[str, Any] | pd.Series[Any], terms_lc: Sequence[str]
) -> bool:
    row_dict = _normalize_row_to_dict(row)
    if row_dict is None:
        return False

    raw_json = row_dict.get("omdb_json")
    parsed = safe_json_loads_single(raw_json)
    if parsed is not None and _json_contains_terms(parsed, terms_lc):
        return True
    if parsed is None and raw_json is not None:
        try:
            raw_lc = str(raw_json).lower()
        except Exception:
            raw_lc = ""
        if raw_lc and all(term in raw_lc for term in terms_lc):
            return True

    imdb_id = row_dict.get("imdb_id")
    if imdb_id is None or _pd_isna(imdb_id):
        imdb_id = row_dict.get("imdbID")
        if imdb_id is not None and _pd_isna(imdb_id):
            imdb_id = None
    title = row_dict.get("title")
    if title is not None and _pd_isna(title):
        title = None
    year = row_dict.get("year")
    if year is not None and _pd_isna(year):
        year = None

    omdb_dict = _get_omdb_record(imdb_id=imdb_id, title=title, year=year)
    if omdb_dict is not None and _json_contains_terms(omdb_dict, terms_lc):
        return True

    wiki_summary, _ = _get_wiki_summary(imdb_id=imdb_id, title=title, year=year)
    if wiki_summary and all(term in wiki_summary.lower() for term in terms_lc):
        return True

    return False


def _df_signature_for_search(df: pd.DataFrame) -> int:
    try:
        idx_hash = pd.util.hash_pandas_object(df.index, index=False).sum()
    except Exception:
        idx_hash = len(df)
    cols_sig = hash(tuple(df.columns))
    return hash((cols_sig, int(idx_hash)))


def _get_detail_search_cache() -> dict[tuple[str, int], list[Any]]:
    cache = st.session_state.get("grid_detail_search_cache")
    if not isinstance(cache, dict):
        cache = {}
        st.session_state["grid_detail_search_cache"] = cache
    return cache


def _apply_text_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    query = query.strip()
    if not query:
        return df

    terms = [t for t in query.split() if t]
    if not terms:
        return df

    df_str = df.astype("string", copy=False)
    mask_any = pd.Series(True, index=df.index)
    for term in terms:
        mask = df_str.apply(
            lambda col, t=term: col.str.contains(t, case=False, regex=False, na=False)
        )
        mask_any = mask_any & mask.any(axis=1)
    terms_lc = [t.lower() for t in terms]

    if not mask_any.all():
        needs_detail = (
            "omdb_json" in df.columns
            or "imdb_id" in df.columns
            or "imdbID" in df.columns
            or "title" in df.columns
        )
        if needs_detail:
            signature = _df_signature_for_search(df)
            cache = _get_detail_search_cache()
            cache_key = (" ".join(terms_lc), signature)
            cached_indices = cache.get(cache_key)
            if cached_indices is None:
                matched_indices: list[Any] = []
                for idx, row in df.loc[~mask_any].iterrows():
                    if _row_matches_detail_json(row, terms_lc):
                        matched_indices.append(idx)
                cache[cache_key] = matched_indices
                cached_indices = matched_indices
            if cached_indices:
                detail_mask = df.index.isin(cached_indices)
                mask_any = mask_any | detail_mask
    return cast(pd.DataFrame, df.loc[mask_any])


def render_grid_toolbar(
    df: pd.DataFrame,
    *,
    key_suffix: str,
    download_filename: str,
    search_label: str = "Busqueda avanzada",
    search_placeholder: str = "Busqueda avanzada",
    search_help: str = (
        "Simple: Actor | Doble: Actor 2006 | Multiple: Actor 2006 Director Genero"
    ),
    show_search: bool = True,
    caption_builder: GridCaptionBuilder | None = None,
) -> tuple[pd.DataFrame, str, int]:
    def _default_caption(count: int, total: int, has_search: bool) -> str:
        if has_search:
            return f"Filas: {count} / {total}"
        return f"Filas: {total}"

    total_rows = len(df)

    safe_suffix = re.sub(r"[^a-zA-Z0-9_-]", "-", key_suffix)
    anchor_id = f"grid-toolbar-{safe_suffix}"
    st.markdown(
        f"""
<div id="{anchor_id}" style="display:none;"></div>
<style>
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) button[data-testid="stBaseButton-secondary"],
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) button[data-testid="baseButton-secondary"] {{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  min-width: 0 !important;
  height: auto !important;
  border-radius: 0 !important;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) button[data-testid="stBaseButton-secondary"] > div,
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) button[data-testid="baseButton-secondary"] > div {{
  padding: 0 !important;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) button[data-testid="stBaseButton-secondary"] span,
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) button[data-testid="baseButton-secondary"] span {{
  font-size: 1.1rem;
  line-height: 1;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) button[data-testid="stBaseButton-secondary"]:hover,
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) button[data-testid="baseButton-secondary"]:hover {{
  filter: brightness(1.1);
  background: transparent !important;
}}
div[data-testid="stVerticalBlock"]:has(#{anchor_id}) .grid-caption {{
  text-align: left;
  font-size: 0.8rem;
  opacity: 0.7;
  margin-top: 0.1rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )

    search_query = ""
    df_view = df

    search_key = f"grid_search_{key_suffix}"
    search_open_key = f"grid_search_open_{key_suffix}"
    if show_search and search_open_key not in st.session_state:
        st.session_state[search_open_key] = False

    col_left, col_right = st.columns([10, 1])

    with col_left:
        left_slot = st.empty()

    with col_right:
        col_search: Any = None
        if show_search:
            col_search, col_download = _columns_with_gap([1, 1], gap="small")
        else:
            col_download = _columns_with_gap([1], gap="small")[0]

        if show_search and col_search is not None:
            with col_search:
                if st.button(
                    "🔍",
                    help=search_label,
                    key=f"grid_search_btn_{key_suffix}",
                    type="secondary",
                ):
                    is_open = bool(st.session_state.get(search_open_key, False))
                    st.session_state[search_open_key] = not is_open
                    if is_open:
                        st.session_state[search_key] = ""

            search_query = str(st.session_state.get(search_key, "") or "")
            df_view = _apply_text_search(df, search_query)

        with col_download:
            csv_export = df_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️",
                data=csv_export,
                file_name=download_filename,
                mime="text/csv",
                key=f"grid_download_{key_suffix}",
                help="Descargar CSV",
            )

    if show_search and bool(st.session_state.get(search_open_key, False)):
        st.text_input(
            search_label,
            key=search_key,
            placeholder=search_placeholder,
            label_visibility="collapsed",
            help=search_help,
        )

    if caption_builder is None:
        caption_builder = _default_caption

    caption = caption_builder(len(df_view), total_rows, bool(search_query))
    if caption:
        left_slot.markdown(
            f'<div class="grid-caption">{caption}</div>',
            unsafe_allow_html=True,
        )

    grid_height = 520

    return df_view, search_query, grid_height


# ============================================================================
# Tabla principal con selección de fila
# ============================================================================


def aggrid_with_row_click(
    df: pd.DataFrame,
    key_suffix: str,
    *,
    visible_order: Sequence[str] | None = None,
    auto_select_first: bool = False,
    show_toolbar: bool = True,
    toolbar_caption_builder: GridCaptionBuilder | None = None,
) -> Optional[dict[str, Any]]:
    """
    Muestra un AgGrid con selección de una sola fila.
    """
    if df.empty:
        st.info("No hay datos para mostrar.")
        return None

    default_order = [
        "title",
        "year",
        "library",
        "imdb_rating",
        "imdb_votes",
        "rt_score",
        "plex_rating",
        "decision",
        "reason",
    ]
    desired_order = list(visible_order) if visible_order else default_order
    visible_cols = [c for c in desired_order if c in df.columns]
    ordered_cols = visible_cols + [c for c in df.columns if c not in visible_cols]
    df_view = df[ordered_cols]

    search_query = ""
    grid_height = 520
    if show_toolbar:
        df_view, search_query, grid_height = render_grid_toolbar(
            df_view,
            key_suffix=key_suffix,
            download_filename=f"{key_suffix}_table.csv",
            caption_builder=toolbar_caption_builder,
        )
        if df_view.empty:
            if search_query.strip():
                st.info("No hay filas que coincidan con la búsqueda.")
            else:
                st.info("No hay datos para mostrar.")
            return None

    gb = GridOptionsBuilder.from_dataframe(df_view)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_grid_options(
        domLayout="normal",
        suppressRowTransform=True,
        wrapHeaderText=True,
        autoHeaderHeight=True,
        defaultColDef={
            "cellStyle": {
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "textAlign": "center",
            }
        },
    )

    colorize_rows = bool(st.session_state.get("grid_colorize_rows", True))
    if colorize_rows:
        gb.configure_grid_options(getRowStyle=_DECISION_ROW_STYLE)

    select_first_js = ""
    if auto_select_first:
        select_first_js = """
    params.api.forEachNode(function(node, index) {
      if (index === 0) {
        node.setSelected(true);
      }
    });
"""

    auto_size_js = JsCode(
        f"""
function(params) {{
  if (!params || !params.api || !params.columnApi) {{
    return;
  }}
  const cols = ['year','library','file_size_gb','imdb_rating','imdb_votes','rt_score'];
  setTimeout(function() {{
    params.columnApi.autoSizeColumns(cols, true);
    params.api.sizeColumnsToFit();
{select_first_js}
  }}, 0);
}}
"""
    )
    fit_js = JsCode(
        """
function(params) {
  if (!params || !params.api) {
    return;
  }
  setTimeout(function() {
    params.api.sizeColumnsToFit();
  }, 0);
}
"""
    )
    gb.configure_grid_options(
        onFirstDataRendered=auto_size_js, onGridSizeChanged=fit_js
    )

    header_names: dict[str, str] = {
        "title": "Title",
        "year": "Year",
        "library": "Library",
        "file_size_gb": "Size",
        "imdb_rating": "IMDb",
        "imdb_votes": "Votes",
        "metacritic_score": "Metacritic",
        "rt_score": "RT",
        "reason": "Reason",
        "file": "File",
    }
    auto_size_cols = {
        "year",
        "library",
        "file_size_gb",
        "imdb_rating",
        "imdb_votes",
        "rt_score",
    }
    wrap_cols = {"title", "reason", "file"}

    for col in df_view.columns:
        col_def: dict[str, Any] = {}
        header = header_names.get(col)
        if header:
            col_def["headerName"] = header
        if col == "file_size_gb":
            col_def["valueFormatter"] = "value != null ? value.toFixed(2) + ' GB' : ''"
        if col == "imdb_votes":
            col_def["valueFormatter"] = (
                "value != null ? Math.round(Number(value)).toLocaleString() : ''"
            )
        if col in auto_size_cols:
            col_def.update({"minWidth": 70, "suppressSizeToFit": True})
        if col == "year":
            col_def.update({"minWidth": 60, "maxWidth": 70, "suppressSizeToFit": True})
        if col in {"file_size_gb", "imdb_rating", "imdb_votes", "rt_score"}:
            col_def.update({"minWidth": 60, "maxWidth": 80, "suppressSizeToFit": True})
        if col == "metacritic_score":
            col_def.update({"minWidth": 70, "maxWidth": 90, "suppressSizeToFit": True})
        if col == "library":
            col_def.update(
                {"minWidth": 120, "maxWidth": 240, "suppressSizeToFit": True}
            )
        if col in wrap_cols:
            flex = 2 if col == "title" else 3
            col_def.update(
                {
                    "wrapText": True,
                    "autoHeight": True,
                    "cellStyle": {
                        "whiteSpace": "normal",
                        "lineHeight": "1.2",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "textAlign": "center",
                    },
                    "minWidth": 220,
                    "flex": flex,
                }
            )
        if col == "title":
            col_def["cellStyle"] = {
                "whiteSpace": "normal",
                "lineHeight": "1.2",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "flex-start",
                "textAlign": "left",
            }
        if col == "year":
            col_def["cellStyle"] = {"justifyContent": "flex-end", "textAlign": "right"}
        if col == "library":
            col_def["cellStyle"] = {"justifyContent": "flex-start", "textAlign": "left"}
        if col == "file_size_gb":
            col_def["cellStyle"] = {"justifyContent": "flex-end", "textAlign": "right"}
        if col == "imdb_rating":
            col_def["cellStyle"] = {"justifyContent": "flex-end", "textAlign": "right"}
        if col == "imdb_votes":
            col_def["cellStyle"] = {"justifyContent": "flex-end", "textAlign": "right"}
        if col == "rt_score":
            col_def["cellStyle"] = {"justifyContent": "flex-end", "textAlign": "right"}
        if col_def:
            gb.configure_column(col, **col_def)

    for col in df_view.columns:
        if col not in visible_cols:
            gb.configure_column(col, hide=True)

    grid_options = gb.build()

    import st_aggrid as st_aggrid_mod

    aggrid_fn = cast(AgGridCallable, getattr(st_aggrid_mod, "AgGrid"))

    theme_key = _current_theme_key()
    grid_response = aggrid_fn(
        df_view.copy(),
        gridOptions=grid_options,
        update_on=["selectionChanged"],
        enable_enterprise_modules=False,
        theme=_aggrid_theme(),
        custom_css=_aggrid_custom_css(),
        height=grid_height,
        allow_unsafe_jscode=True,
        key=f"aggrid_{key_suffix}_{int(colorize_rows)}_{theme_key}",
    )

    selected_raw = grid_response.get("selected_rows")
    selected_rows = _normalize_selected_rows(selected_raw)

    if not selected_rows:
        if auto_select_first:
            first_row = _normalize_row_to_dict(df_view.iloc[0])
            if first_row is not None:
                return dict(first_row)
        return None

    return dict(selected_rows[0])


# ============================================================================
# Tabla de solo lectura (AgGrid) para vistas simples
# ============================================================================


def aggrid_readonly(df: pd.DataFrame, *, key_suffix: str, height: int = 420) -> None:
    """
    Muestra un AgGrid solo-lectura con estilos del tema activo.
    """
    if df.empty:
        st.info("No hay datos para mostrar.")
        return

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_grid_options(
        domLayout="normal",
        suppressRowTransform=True,
        enableCellTextSelection=True,
        ensureDomOrder=True,
        defaultColDef={"resizable": True, "sortable": True},
    )

    auto_size_js = JsCode(
        """
function(params) {
  if (!params || !params.api || !params.columnApi) {
    return;
  }
  setTimeout(function() {
    params.api.sizeColumnsToFit();
  }, 0);
}
"""
    )
    fit_js = JsCode(
        """
function(params) {
  if (!params || !params.api) {
    return;
  }
  setTimeout(function() {
    params.api.sizeColumnsToFit();
  }, 0);
}
"""
    )
    gb.configure_grid_options(
        onFirstDataRendered=auto_size_js, onGridSizeChanged=fit_js
    )

    grid_options = gb.build()

    import st_aggrid as st_aggrid_mod

    aggrid_fn = cast(AgGridCallable, getattr(st_aggrid_mod, "AgGrid"))

    theme_key = _current_theme_key()
    aggrid_fn(
        df.copy(),
        gridOptions=grid_options,
        update_on=["selectionChanged"],
        enable_enterprise_modules=False,
        theme=_aggrid_theme(),
        custom_css=_aggrid_custom_css(),
        height=height,
        allow_unsafe_jscode=True,
        key=f"aggrid_readonly_{key_suffix}_{theme_key}",
    )


# ============================================================================
# Detalle de una película (panel tipo ficha)
# ============================================================================


def _get_from_omdb_or_row(
    row: Mapping[str, Any], omdb_dict: Mapping[str, Any] | None, key: str
) -> Any:
    """
    Devuelve primero row[key] y, si no existe/no es usable, omdb_dict[key].
    """
    if key in row and row.get(key) not in (None, ""):
        return row.get(key)
    if omdb_dict and isinstance(omdb_dict, Mapping):
        return omdb_dict.get(key)
    return None


def _safe_number_to_str(v: Any) -> str:
    """Convierte números/valores a string seguro para UI."""
    try:
        if v is None or (isinstance(v, float) and _pd_isna(v)):
            return "N/A"
        return str(v)
    except Exception:
        return "N/A"


def _safe_votes(v: Any) -> str:
    """Formatea votos con separador de miles; tolera strings tipo '12,345'."""
    try:
        if v is None or (isinstance(v, float) and _pd_isna(v)):
            return "N/A"
        if isinstance(v, str):
            v2 = v.replace(",", "")
            return f"{int(float(v2)):,}"
        return f"{int(float(v)):,}"
    except Exception:
        return "N/A"


def _safe_metacritic(v: Any) -> str:
    try:
        if v is None or (isinstance(v, float) and _pd_isna(v)):
            return "N/A"
        if isinstance(v, str):
            s = v.strip()
            if not s or s.upper() == "N/A":
                return "N/A"
            if "/" in s:
                s = s.split("/", 1)[0].strip()
            return str(int(float(s)))
        return str(int(float(v)))
    except Exception:
        return "N/A"


def _is_nonempty_str(value: Any) -> bool:
    """True si value es string “usable” (no '', 'nan', 'none')."""
    if value is None:
        return False
    if _pd_isna(value):
        return False
    s = str(value).strip()
    if not s:
        return False
    lower = s.lower()
    return lower not in ("nan", "none", "<na>")


def _build_plex_url(guid: Any) -> str | None:
    if guid in (None, ""):
        return None
    try:
        guid_str = str(guid).strip()
    except Exception:
        return None
    if not guid_str:
        return None
    return guid_str


def _build_imdb_url(imdb_id: Any) -> str | None:
    if imdb_id in (None, ""):
        return None
    return f"https://www.imdb.com/title/{imdb_id}"


def _build_detail_query(
    *, imdb_id: Any, guid: Any, title: Any, year: Any
) -> str | None:
    params: list[str] = ["open_detail=1"]
    if _is_nonempty_str(imdb_id) and not _pd_isna(imdb_id):
        params.append(f"imdb_id={quote(str(imdb_id))}")
    if _is_nonempty_str(guid) and not _pd_isna(guid):
        params.append(f"guid={quote(str(guid))}")
    if _is_nonempty_str(title):
        params.append(f"title={quote(str(title))}")
    if year not in (None, "") and not _pd_isna(year):
        params.append(f"year={quote(str(year))}")
    return "?" + "&".join(params)


_F = TypeVar("_F", bound=Callable[..., Any])


def _cache_data_decorator() -> Callable[[_F], _F]:
    cache_fn = getattr(st, "cache_data", None)
    if callable(cache_fn):
        return cast(Callable[[_F], _F], cache_fn(show_spinner=False))
    cache_fn = getattr(st, "cache", None)
    if callable(cache_fn):
        return cast(Callable[[_F], _F], cache_fn)
    return cast(Callable[[_F], _F], lambda f: f)


@_cache_data_decorator()
def _load_wiki_cache_json(path: str, mtime: float) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


@_cache_data_decorator()
def _load_omdb_cache_json(path: str, mtime: float) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _get_wiki_cache() -> Mapping[str, Any] | None:
    try:
        if not WIKI_CACHE_PATH.exists():
            return None
        mtime = WIKI_CACHE_PATH.stat().st_mtime
    except Exception:
        return None
    return cast(
        Mapping[str, Any] | None, _load_wiki_cache_json(str(WIKI_CACHE_PATH), mtime)
    )


def _get_omdb_cache() -> Mapping[str, Any] | None:
    try:
        if not OMDB_CACHE_PATH.exists():
            return None
        mtime = OMDB_CACHE_PATH.stat().st_mtime
    except Exception:
        return None
    return cast(
        Mapping[str, Any] | None, _load_omdb_cache_json(str(OMDB_CACHE_PATH), mtime)
    )


_WIKI_TITLE_CLEAN_RE = re.compile(r"[^0-9A-Za-záéíóúÁÉÍÓÚñÑüÜ]+")
_OMDB_TITLE_CLEAN_RE = re.compile(r"[^0-9A-Za-záéíóúÁÉÍÓÚñÑüÜ]+")


def _normalize_wiki_title(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    s = _WIKI_TITLE_CLEAN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _normalize_omdb_title(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    s = _OMDB_TITLE_CLEAN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _normalize_wiki_year(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not _pd_isna(value):
        return str(int(value))
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit() and len(s) == 4:
        return s
    return None


def _normalize_wiki_imdb(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v or None


def _normalize_omdb_imdb(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v or None


def _is_spanish_lang(lang: Any) -> bool:
    if not isinstance(lang, str):
        return False
    return lang.strip().lower().startswith("es")


def _get_wiki_summary(
    *,
    imdb_id: Any,
    title: Any,
    year: Any,
) -> tuple[str | None, str | None]:
    cache = _get_wiki_cache()
    if not isinstance(cache, Mapping):
        return None, None

    records = cache.get("records")
    idx_imdb = cache.get("index_imdb")
    idx_ty = cache.get("index_ty")

    if not isinstance(records, Mapping):
        return None, None

    def _item_from_rid(rid: object) -> Mapping[str, Any] | None:
        if not isinstance(rid, str):
            return None
        raw = records.get(rid)
        return raw if isinstance(raw, Mapping) else None

    imdb_norm = _normalize_wiki_imdb(imdb_id)
    if imdb_norm and isinstance(idx_imdb, Mapping):
        item = _item_from_rid(idx_imdb.get(imdb_norm))
        if item:
            return _extract_summary_from_item(item)

    title_norm = _normalize_wiki_title(title)
    year_norm = _normalize_wiki_year(year)
    if title_norm and year_norm and isinstance(idx_ty, Mapping):
        key = f"{title_norm}|{year_norm}"
        item = _item_from_rid(idx_ty.get(key))
        if item:
            return _extract_summary_from_item(item)

    return None, None


def _get_omdb_record(
    *,
    imdb_id: Any,
    title: Any,
    year: Any,
) -> Mapping[str, Any] | None:
    cache = _get_omdb_cache()
    if not isinstance(cache, Mapping):
        return None

    records = cache.get("records")
    idx_imdb = cache.get("index_imdb")
    idx_ty = cache.get("index_ty")

    if not isinstance(records, Mapping):
        return None

    def _item_from_rid(rid: object) -> Mapping[str, Any] | None:
        if not isinstance(rid, str):
            return None
        raw = records.get(rid)
        return raw if isinstance(raw, Mapping) else None

    imdb_norm = _normalize_omdb_imdb(imdb_id)
    if imdb_norm and isinstance(idx_imdb, Mapping):
        item = _item_from_rid(idx_imdb.get(imdb_norm))
        if item:
            return _extract_omdb_payload(item)

    title_norm = _normalize_omdb_title(title)
    year_norm = _normalize_wiki_year(year)
    if title_norm and year_norm and isinstance(idx_ty, Mapping):
        key = f"{title_norm}|{year_norm}"
        item = _item_from_rid(idx_ty.get(key))
        if item:
            return _extract_omdb_payload(item)

    return None


def _extract_omdb_payload(item: Mapping[str, Any]) -> Mapping[str, Any] | None:
    status = item.get("status")
    if status not in (None, "ok"):
        return None
    payload = item.get("omdb")
    return payload if isinstance(payload, Mapping) else None


def _extract_summary_from_item(
    item: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    status = item.get("status")
    if status not in (None, "ok"):
        return None, None

    wiki_block = item.get("wiki")
    if not isinstance(wiki_block, Mapping):
        return None, None

    summary = wiki_block.get("summary")
    if not _is_nonempty_str(summary):
        return None, None

    lang = wiki_block.get("source_language") or wiki_block.get("language")
    return str(summary), str(lang) if isinstance(lang, str) and lang.strip() else None


def render_detail_card(
    row: dict[str, Any] | pd.Series | Mapping[str, Any] | None,
) -> None:
    if row is None:
        st.info("Haz click en una fila para ver su detalle.")
        return

    normalized = _normalize_row_to_dict(row)
    if normalized is None:
        st.warning("Detalle no disponible: fila con formato inesperado.")
        return

    row_dict: RowDict = normalized

    omdb_dict: Mapping[str, Any] | None = None
    if "omdb_json" in row_dict:
        try:
            parsed = safe_json_loads_single(row_dict.get("omdb_json"))
            if isinstance(parsed, Mapping):
                omdb_dict = parsed
        except Exception:
            omdb_dict = None

    title = row_dict.get("title", "¿Sin título?")
    year = row_dict.get("year")
    library = row_dict.get("library")
    decision = row_dict.get("decision")
    reason = row_dict.get("reason")
    imdb_rating = row_dict.get("imdb_rating")
    imdb_votes = row_dict.get("imdb_votes")
    rt_score = row_dict.get("rt_score")

    poster_url = row_dict.get("poster_url")
    file_path = row_dict.get("file")
    file_size = row_dict.get("file_size")
    trailer_url = row_dict.get("trailer_url")
    plex_guid = row_dict.get("guid")
    imdb_id = row_dict.get("imdb_id")

    if omdb_dict is None:
        omdb_dict = _get_omdb_record(imdb_id=imdb_id, title=title, year=year)
    omdb_imdb_id = omdb_dict.get("imdbID") if isinstance(omdb_dict, Mapping) else None

    rated = _get_from_omdb_or_row(row_dict, omdb_dict, "Rated")
    released = _get_from_omdb_or_row(row_dict, omdb_dict, "Released")
    runtime = _get_from_omdb_or_row(row_dict, omdb_dict, "Runtime")
    genre = _get_from_omdb_or_row(row_dict, omdb_dict, "Genre")
    director = _get_from_omdb_or_row(row_dict, omdb_dict, "Director")
    writer = _get_from_omdb_or_row(row_dict, omdb_dict, "Writer")
    actors = _get_from_omdb_or_row(row_dict, omdb_dict, "Actors")
    language = _get_from_omdb_or_row(row_dict, omdb_dict, "Language")
    country = _get_from_omdb_or_row(row_dict, omdb_dict, "Country")
    awards = _get_from_omdb_or_row(row_dict, omdb_dict, "Awards")
    plot = _get_from_omdb_or_row(row_dict, omdb_dict, "Plot")
    metacritic = row_dict.get("metacritic_score")
    if metacritic is None and isinstance(omdb_dict, Mapping):
        metacritic = omdb_dict.get("Metascore")

    wiki_summary, wiki_lang = _get_wiki_summary(
        imdb_id=imdb_id or omdb_imdb_id,
        title=title,
        year=year,
    )
    summary_text: str | None = None
    summary_label = "Resumen"
    if wiki_summary and _is_spanish_lang(wiki_lang):
        summary_text = wiki_summary
        summary_label = "Resumen (Wiki ES)"
    elif _is_nonempty_str(plot):
        summary_text = str(plot)
        summary_label = "Resumen (OMDb)"
    elif wiki_summary:
        summary_text = wiki_summary
        summary_label = f"Resumen (Wiki {wiki_lang})" if wiki_lang else "Resumen (Wiki)"

    is_modal = bool(st.session_state.get("modal_open"))
    poster_width = 240 if is_modal else 260
    detail_anchor_id = "detail-card-modal" if is_modal else "detail-card-main"
    left_anchor_id = f"{detail_anchor_id}-left"
    metrics_anchor_id = f"{detail_anchor_id}-metrics"
    action_group_anchor_id = f"{detail_anchor_id}-action-group"
    imdb_url = _build_imdb_url(imdb_id)
    plex_url = _build_plex_url(plex_guid)

    with st.container():
        st.markdown(
            f'<div id="{detail_anchor_id}" style="display:none;"></div>',
            unsafe_allow_html=True,
        )
        extra_action_group_css = ""
        if not is_modal:
            extra_action_group_css = f"""
div[data-testid="stVerticalBlock"]:has(#{action_group_anchor_id}) {{
  background: var(--mc-panel-bg);
  border: 1px solid var(--mc-panel-border);
  border-radius: 14px;
  padding: 0.7rem 0.85rem;
  margin-top: 0.75rem;
}}
div[data-testid="stVerticalBlock"]:has(#{action_group_anchor_id}) [data-testid="stMetric"] {{
  background: var(--mc-metric-bg);
  border: 1px solid var(--mc-metric-border);
  border-radius: 12px;
  padding: 10px 12px;
  text-align: center;
}}
div[data-testid="stVerticalBlock"]:has(#{action_group_anchor_id}) [data-testid="stMetricLabel"] {{
  color: var(--mc-text-3);
  letter-spacing: 0.04em;
  font-size: 0.72rem;
  text-transform: uppercase;
}}
div[data-testid="stVerticalBlock"]:has(#{action_group_anchor_id}) [data-testid="stMetricValue"] {{
  font-size: 1.6rem;
}}
"""

        st.markdown(
            f"""
<style>
div[data-testid="stHorizontalBlock"]:has(#{left_anchor_id}) {{
  align-items: stretch;
}}
div[data-testid="stVerticalBlock"]:has(#{detail_anchor_id}) {{
  background: var(--mc-card-bg);
  border: 1px solid var(--mc-card-border);
  border-radius: 16px;
  padding: 18px 20px 22px;
  box-shadow: var(--mc-card-shadow);
}}
div[data-testid="stVerticalBlock"]:has(#{detail_anchor_id}) img {{
  border-radius: 12px;
  box-shadow: var(--mc-image-shadow);
}}
div[data-testid="stVerticalBlock"]:has(#{detail_anchor_id}) h4.detail-title {{
  margin: 0 0 0.5rem 0;
  display: inline-flex;
  align-items: center;
  padding: 0.45rem 0.85rem;
  border-radius: 0.65rem;
  background: var(--mc-pill-bg);
  border: 1px solid var(--mc-pill-border);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
}}
div[data-testid="stVerticalBlock"]:has(#{detail_anchor_id}) h4.detail-title a {{
  color: inherit;
  text-decoration: none;
}}
div[data-testid="stVerticalBlock"]:has(#{detail_anchor_id}) h4.detail-title a:hover {{
  filter: brightness(1.08);
}}
div[data-testid="stVerticalBlock"]:has(#{detail_anchor_id}) a.detail-action {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  padding: 0.45rem 0.7rem;
  border-radius: 0.6rem;
  background: var(--mc-action-bg);
  border: 1px solid var(--mc-action-border);
  color: var(--mc-action-text) !important;
  text-decoration: none !important;
  font-weight: 600;
  width: 100%;
}}
div[data-testid="stVerticalBlock"]:has(#{detail_anchor_id}) a.detail-action:hover {{
  background: var(--mc-action-hover-bg);
}}
div[data-testid="stVerticalBlock"]:has(#{detail_anchor_id}) hr {{
  border: 0;
  height: 1px;
  background: var(--mc-divider);
  margin: 1rem 0;
}}
div[data-testid="stVerticalBlock"]:has(#{detail_anchor_id}) p {{
  color: var(--mc-text-2);
  line-height: 1.55;
  max-width: 62ch;
  margin-left: auto;
  margin-right: auto;
}}
div[data-testid="stVerticalBlock"]:has(#detail-card-modal) p {{
  max-width: none;
  margin-left: 0;
  margin-right: 0;
}}
div[data-testid="stVerticalBlock"]:has(#detail-card-modal) .summary-block {{
  background: var(--mc-summary-bg);
  border: 1px solid var(--mc-summary-border);
  border-radius: 14px;
  padding: 18px 20px;
  width: 100%;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
}}
div[data-testid="stVerticalBlock"]:has(#detail-card-modal) .summary-block h4 {{
  margin: 0 0 0.65rem 0;
}}
div[data-testid="stVerticalBlock"]:has(#detail-card-modal) .summary-block p {{
  margin: 0;
  line-height: 1.65;
  font-size: 0.98rem;
  color: var(--mc-summary-text);
}}
div[data-testid="stVerticalBlock"]:has(#detail-card-modal) div[data-testid="stMetric"] {{
  background: var(--mc-metric-bg);
  border: 1px solid var(--mc-metric-border);
  border-radius: 12px;
  padding: 10px 12px;
  text-align: center;
}}
div[data-testid="stVerticalBlock"]:has(#{detail_anchor_id}) [data-testid="stMetricLabel"] {{
  color: var(--mc-text-3);
  letter-spacing: 0.04em;
  font-size: 0.75rem;
  text-transform: uppercase;
}}
div[data-testid="stVerticalBlock"]:has(#{detail_anchor_id}) [data-testid="stMetricValue"] {{
  font-size: 1.8rem;
}}
{extra_action_group_css}
</style>
""",
            unsafe_allow_html=True,
        )

        col_left, col_right = (
            _columns_with_gap([1, 4], gap="small")
            if is_modal
            else _columns_with_gap([1, 2], gap="small")
        )

        with col_left:
            st.markdown(f'<div id="{left_anchor_id}"></div>', unsafe_allow_html=True)
            if _is_nonempty_str(poster_url):
                st.image(str(poster_url), width=poster_width)
            else:
                st.write("📷 Sin póster")
            if is_modal and (imdb_url or plex_url):
                if imdb_url:
                    st.markdown(
                        f'<a class="detail-action" href="{imdb_url}">🎬 Ver en IMDb</a>',
                        unsafe_allow_html=True,
                    )
                if plex_url:
                    st.markdown(
                        f'<a class="detail-action" href="{plex_url}">📺 Ver en Plex</a>',
                        unsafe_allow_html=True,
                    )

        with col_right:
            header = str(title) + _safe_year_suffix(year)
            safe_header = html.escape(header)
            st.session_state["detail_row_last"] = row_dict
            if is_modal:
                st.markdown(
                    f'<h4 class="detail-title">{safe_header}</h4>',
                    unsafe_allow_html=True,
                )
            else:
                detail_query = _build_detail_query(
                    imdb_id=imdb_id,
                    guid=plex_guid,
                    title=title,
                    year=year,
                )
                if detail_query is None:
                    st.markdown(
                        f'<h4 class="detail-title">{safe_header}</h4>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<h4 class="detail-title"><a href="{detail_query}">{safe_header}</a></h4>',
                        unsafe_allow_html=True,
                    )
            if library:
                st.write(f"**Biblioteca:** {library}")
            if year:
                st.write(f"**Año:** {year}")
            if actors:
                st.write(f"**Actores:** {actors}")
            if is_modal:
                with st.container():
                    st.markdown(
                        f'<div id="{metrics_anchor_id}"></div>',
                        unsafe_allow_html=True,
                    )
                    m1, m2, m3 = st.columns(3)
                    m1.metric("IMDb", _safe_number_to_str(imdb_rating))
                    m1.caption(f"Votos IMDb: {_safe_votes(imdb_votes)}")

                    rt_str = _safe_number_to_str(rt_score)
                    m2.metric("RT", f"{rt_str}%" if rt_str != "N/A" else "N/A")

                    m3.metric("Metacritic", _safe_metacritic(metacritic))

        if not is_modal and (
            imdb_url or plex_url or imdb_rating or rt_score or metacritic
        ):
            with st.container():
                st.markdown(
                    f'<div id="{action_group_anchor_id}"></div>',
                    unsafe_allow_html=True,
                )
                group_cols = st.columns([1, 2], gap="small")
                with group_cols[0]:
                    if imdb_url:
                        st.markdown(
                            f'<a class="detail-action" href="{imdb_url}">🎬 Ver en IMDb</a>',
                            unsafe_allow_html=True,
                        )
                    if plex_url:
                        st.markdown(
                            f'<a class="detail-action" href="{plex_url}">📺 Ver en Plex</a>',
                            unsafe_allow_html=True,
                        )
                with group_cols[1]:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("IMDb", _safe_number_to_str(imdb_rating))
                    rt_str = _safe_number_to_str(rt_score)
                    m2.metric("RT", f"{rt_str}%" if rt_str != "N/A" else "N/A")
                    m3.metric("Metacritic", _safe_metacritic(metacritic))

        if _is_nonempty_str(summary_text):
            st.markdown("---")
            if is_modal:
                safe_label = html.escape(summary_label)
                safe_text = html.escape(str(summary_text)).replace("\n", "<br/>")
                st.markdown(
                    f'<div class="summary-block"><h4>{safe_label}</h4>'
                    f"<p>{safe_text}</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.write(f"#### {summary_label}")
                st.write(str(summary_text))

        if is_modal:
            st.markdown("---")
            st.write("#### Información OMDb")

            cols_basic = st.columns(4)
            with cols_basic[0]:
                if rated:
                    st.write(f"**Rated:** {rated}")
            with cols_basic[1]:
                if released:
                    st.write(f"**Estreno:** {released}")
            with cols_basic[2]:
                if runtime:
                    st.write(f"**Duración:** {runtime}")
            with cols_basic[3]:
                if genre:
                    st.write(f"**Género:** {genre}")

            st.write("")
            cols_credits = st.columns(3)
            with cols_credits[0]:
                if director:
                    st.write(f"**Director:** {director}")
            with cols_credits[1]:
                if writer:
                    st.write(f"**Guion:** {writer}")
            with cols_credits[2]:
                if actors:
                    st.write(f"**Reparto:** {actors}")

            st.write("")
            cols_prod = st.columns(3)
            with cols_prod[0]:
                if language:
                    st.write(f"**Idioma(s):** {language}")
            with cols_prod[1]:
                if country:
                    st.write(f"**País:** {country}")
            with cols_prod[2]:
                if awards:
                    st.write(f"**Premios:** {awards}")

            if _is_nonempty_str(plot) and not _is_nonempty_str(summary_text):
                st.markdown("---")
                st.write("#### Sinopsis")
                st.write(str(plot))

            st.markdown("---")
            st.write("#### Archivo")
            if file_path:
                st.code(str(file_path), language="bash")

            if file_size is not None and not (
                isinstance(file_size, float) and _pd_isna(file_size)
            ):
                try:
                    gb = float(file_size) / (1024**3)
                    st.write(f"**Tamaño:** {gb:.2f} GB")
                except Exception:
                    st.write(f"**Tamaño:** {file_size}")

            if decision:
                st.write(f"**Decisión:** `{decision}` — {reason}")

            if _is_nonempty_str(trailer_url):
                st.markdown("#### 🎞 Tráiler")
                st.video(str(trailer_url))

            with st.expander("Ver JSON completo"):
                full_row: MutableMapping[str, Any] = dict(row_dict)
                if omdb_dict is not None:
                    full_row["_omdb_parsed"] = dict(omdb_dict)
                st.json(full_row)


# ============================================================================
# Modal de detalle ampliado
# ============================================================================


def render_modal() -> None:
    """
    Vista de detalle ampliado usando el estado global de Streamlit.
    """
    if not st.session_state.get("modal_open"):
        return

    row = st.session_state.get("modal_row")
    if row is None:
        return

    c1, c2 = _columns_with_gap([10, 1], gap="small")
    with c1:
        st.markdown("### 🔍 Detalle ampliado")
    with c2:
        if st.button("✖", key="close_modal"):
            st.session_state["modal_open"] = False
            st.session_state["modal_row"] = None
            _rerun()

    render_detail_card(row)
