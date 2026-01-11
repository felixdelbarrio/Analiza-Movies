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

import json
import os
import re
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from typing import Any, Literal, Optional, Protocol, cast

import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder, JsCode

from frontend.config_front_artifacts import OMDB_CACHE_PATH, WIKI_CACHE_PATH
from frontend.data_utils import safe_json_loads_single

RowDict = dict[str, Any]

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
  if (d === "DELETE") return { color: "#e53935" };
  if (d === "KEEP") return { color: "#43a047" };
  if (d === "MAYBE") return { color: "#fbc02d" };
  if (d === "UNKNOWN") return { color: "#9e9e9e" };
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


def _columns_with_gap(spec: Sequence[int], *, gap: Literal["small", "medium", "large"]) -> Sequence[Any]:
    try:
        return st.columns(spec, gap=gap)
    except TypeError:
        return st.columns(spec)


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
                out_list.append(_to_str_key_dict(cast(Mapping[Hashable, Any], item.to_dict())))
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
    if isinstance(selected_raw, Iterable) and not isinstance(selected_raw, (str, bytes)):
        out_it: list[RowDict] = []
        for x in selected_raw:
            if isinstance(x, pd.Series):
                out_it.append(_to_str_key_dict(cast(Mapping[Hashable, Any], x.to_dict())))
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


# ============================================================================
# Tabla principal con selección de fila
# ============================================================================


def aggrid_with_row_click(
    df: pd.DataFrame,
    key_suffix: str,
    *,
    visible_order: Sequence[str] | None = None,
    auto_select_first: bool = False,
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
    df = df[ordered_cols]

    gb = GridOptionsBuilder.from_dataframe(df)
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

    if bool(st.session_state.get("grid_colorize_rows", True)):
        gb.configure_grid_options(getRowStyle=_DECISION_ROW_STYLE)

    resize_js = JsCode(
        """
function(params) {
  if (!params || !params.api || !params.columnApi) {
    return;
  }
  const cols = ['year','library','file_size_gb','imdb_rating','imdb_votes','rt_score'];
  setTimeout(function() {
    params.columnApi.autoSizeColumns(cols, true);
    params.api.sizeColumnsToFit();
  }, 0);
}
"""
    )
    gb.configure_grid_options(onGridReady=resize_js, onGridSizeChanged=resize_js)

    if auto_select_first:
        gb.configure_grid_options(
            onFirstDataRendered=JsCode(
                """
function(params) {
  if (!params || !params.api) {
    return;
  }
  params.api.forEachNode(function(node, index) {
    if (index === 0) {
      node.setSelected(true);
    }
  });
}
"""
            )
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
    auto_size_cols = {"year", "library", "file_size_gb", "imdb_rating", "imdb_votes", "rt_score"}
    wrap_cols = {"title", "reason", "file"}

    for col in df.columns:
        col_def: dict[str, Any] = {}
        header = header_names.get(col)
        if header:
            col_def["headerName"] = header
        if col == "file_size_gb":
            col_def["valueFormatter"] = "value != null ? value.toFixed(2) + ' GB' : ''"
        if col == "imdb_votes":
            col_def["valueFormatter"] = "value != null ? Math.round(Number(value)).toLocaleString() : ''"
        if col in auto_size_cols:
            col_def.update({"minWidth": 70, "suppressSizeToFit": True})
        if col == "year":
            col_def.update({"minWidth": 60, "maxWidth": 70, "suppressSizeToFit": True})
        if col in {"file_size_gb", "imdb_rating", "imdb_votes", "rt_score"}:
            col_def.update({"minWidth": 60, "maxWidth": 80, "suppressSizeToFit": True})
        if col == "metacritic_score":
            col_def.update({"minWidth": 70, "maxWidth": 90, "suppressSizeToFit": True})
        if col == "library":
            col_def.update({"minWidth": 120, "maxWidth": 240, "suppressSizeToFit": True})
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

    for col in df.columns:
        if col not in visible_cols:
            gb.configure_column(col, hide=True)

    grid_options = gb.build()

    import st_aggrid as st_aggrid_mod

    aggrid_fn = cast(AgGridCallable, getattr(st_aggrid_mod, "AgGrid"))

    colorize_rows = bool(st.session_state.get("grid_colorize_rows", True))
    grid_response = aggrid_fn(
        df.copy(),
        gridOptions=grid_options,
        update_on=["selectionChanged"],
        enable_enterprise_modules=False,
        height=520,
        allow_unsafe_jscode=True,
        key=f"aggrid_{key_suffix}_{int(colorize_rows)}",
    )

    selected_raw = grid_response.get("selected_rows")
    selected_rows = _normalize_selected_rows(selected_raw)

    if not selected_rows:
        return None

    return dict(selected_rows[0])


# ============================================================================
# Detalle de una película (panel tipo ficha)
# ============================================================================


def _get_from_omdb_or_row(row: Mapping[str, Any], omdb_dict: Mapping[str, Any] | None, key: str) -> Any:
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
    s = str(value).strip()
    if not s:
        return False
    lower = s.lower()
    return lower not in ("nan", "none")


def _build_plex_url(rating_key: Any) -> str | None:
    plex_base = os.getenv("PLEX_WEB_BASEURL") or os.getenv("PLEX_BASEURL") or os.getenv("BASEURL") or ""
    if not plex_base:
        return None
    if rating_key in (None, ""):
        return None
    return f"{plex_base}/web/index.html#!/server/library/metadata/{rating_key}"


def _build_imdb_url(imdb_id: Any) -> str | None:
    if imdb_id in (None, ""):
        return None
    return f"https://www.imdb.com/title/{imdb_id}"


def _cache_data_decorator() -> Any:
    cache_fn = getattr(st, "cache_data", None)
    if callable(cache_fn):
        return cache_fn(show_spinner=False)
    cache_fn = getattr(st, "cache", None)
    if callable(cache_fn):
        return cache_fn
    return lambda f: f


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
    return cast(Mapping[str, Any] | None, _load_wiki_cache_json(str(WIKI_CACHE_PATH), mtime))


def _get_omdb_cache() -> Mapping[str, Any] | None:
    try:
        if not OMDB_CACHE_PATH.exists():
            return None
        mtime = OMDB_CACHE_PATH.stat().st_mtime
    except Exception:
        return None
    return cast(Mapping[str, Any] | None, _load_omdb_cache_json(str(OMDB_CACHE_PATH), mtime))


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


def _extract_summary_from_item(item: Mapping[str, Any]) -> tuple[str | None, str | None]:
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
    show_modal_button: bool = True,
    button_key_prefix: Optional[str] = None,
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
    rating_key = row_dict.get("rating_key")
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

    if show_modal_button:
        col_left, col_right = _columns_with_gap([1, 2], gap="small")
        poster_width = 260
    else:
        col_left, col_right = _columns_with_gap([1, 4], gap="small")
        poster_width = 240

    with col_left:
        if _is_nonempty_str(poster_url):
            st.image(str(poster_url), width=poster_width)
        else:
            st.write("📷 Sin póster")

        imdb_url = _build_imdb_url(imdb_id)
        if imdb_url:
            st.markdown(f"[🎬 Ver en IMDb]({imdb_url})")

        plex_url = _build_plex_url(rating_key)
        if plex_url:
            st.markdown(f"[📺 Ver en Plex Web]({plex_url})")

        if show_modal_button:
            key_suffix = button_key_prefix or "default"
            button_key = f"open_modal_{key_suffix}"
            if st.button("🪟 Abrir en ventana", key=button_key):
                st.session_state["modal_row"] = row_dict
                st.session_state["modal_open"] = True
                _rerun()

    with col_right:
        header = str(title) + _safe_year_suffix(year)

        st.markdown(f"### {header}")
        if library:
            st.write(f"**Biblioteca:** {library}")
        if year:
            st.write(f"**Año:** {year}")
        if actors:
            st.write(f"**Actores:** {actors}")

        m1, m2, m3 = st.columns(3)
        m1.metric("IMDb", _safe_number_to_str(imdb_rating))
        if not show_modal_button:
            m1.caption(f"Votos IMDb: {_safe_votes(imdb_votes)}")

        rt_str = _safe_number_to_str(rt_score)
        m2.metric("RT", f"{rt_str}%" if rt_str != "N/A" else "N/A")

        m3.metric("Metacritic", _safe_metacritic(metacritic))

        if _is_nonempty_str(summary_text):
            st.markdown("---")
            st.write(f"#### {summary_label}")
            st.write(str(summary_text))

        if not show_modal_button:
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

            if file_size is not None and not (isinstance(file_size, float) and _pd_isna(file_size)):
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

    render_detail_card(row, show_modal_button=False)
