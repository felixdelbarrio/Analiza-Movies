from __future__ import annotations

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

Logging
- Este módulo no hace logging agresivo: en UI se prefieren st.info/st.warning.
- Si necesitas instrumentación, añade logs en capas “backend”; aquí se mantiene simple.
"""

import os
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from typing import Any, Dict, Optional, Protocol, cast

import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder

from frontend.data_utils import safe_json_loads_single


RowDict = dict[str, Any]


class AgGridCallable(Protocol):
    def __call__(
        self,
        data: pd.DataFrame,
        *,
        gridOptions: Mapping[str, Any],
        update_on: Sequence[str],
        enable_enterprise_modules: bool,
        height: int,
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

    st_aggrid puede devolver:
    - None
    - list[dict] (típico)
    - pd.DataFrame (dependiendo de configuración/versión)
    - dict (caso raro)
    - otros iterables

    Devuelve siempre una lista; si no hay selección, lista vacía.
    """
    if selected_raw is None:
        return []

    if isinstance(selected_raw, pd.DataFrame):
        records = selected_raw.to_dict(orient="records")
        out_df: list[RowDict] = []
        for r in records:
            if isinstance(r, Mapping):
                out_df.append(_to_str_key_dict(cast(Mapping[Hashable, Any], r)))
            else:
                out_df.append({"value": r})
        return out_df

    if isinstance(selected_raw, pd.Series):
        return [_to_str_key_dict(cast(Mapping[Hashable, Any], selected_raw.to_dict()))]

    if isinstance(selected_raw, Mapping):
        return [_to_str_key_dict(cast(Mapping[Hashable, Any], selected_raw))]

    if isinstance(selected_raw, (list, tuple)):
        out_list: list[RowDict] = []
        for item in selected_raw:
            if isinstance(item, pd.Series):
                out_list.append(_to_str_key_dict(cast(Mapping[Hashable, Any], item.to_dict())))
                continue
            if isinstance(item, Mapping):
                out_list.append(_to_str_key_dict(cast(Mapping[Hashable, Any], item)))
                continue
            try:
                tmp = dict(item)  # type: ignore[arg-type]
            except Exception:
                out_list.append({"value": item})
            else:
                if isinstance(tmp, Mapping):
                    out_list.append(_to_str_key_dict(cast(Mapping[Hashable, Any], tmp)))
                else:
                    out_list.append({"value": tmp})
        return out_list

    if isinstance(selected_raw, Iterable) and not isinstance(selected_raw, (str, bytes)):
        out_it: list[RowDict] = []
        for x in selected_raw:
            if isinstance(x, pd.Series):
                out_it.append(_to_str_key_dict(cast(Mapping[Hashable, Any], x.to_dict())))
                continue
            if isinstance(x, Mapping):
                out_it.append(_to_str_key_dict(cast(Mapping[Hashable, Any], x)))
                continue
            try:
                tmp = dict(x)  # type: ignore[arg-type]
            except Exception:
                out_it.append({"value": x})
            else:
                if isinstance(tmp, Mapping):
                    out_it.append(_to_str_key_dict(cast(Mapping[Hashable, Any], tmp)))
                else:
                    out_it.append({"value": tmp})
        return out_it

    return [{"value": selected_raw}]


def _normalize_row_to_dict(row: Any) -> Optional[RowDict]:
    """
    Convierte distintas formas de fila (Series, dict, etc.) a dict[str, Any].

    Returns:
        dict o None si es imposible convertir.
    """
    if row is None:
        return None

    if isinstance(row, pd.Series):
        d = row.to_dict()
        if isinstance(d, Mapping):
            return _to_str_key_dict(cast(Mapping[Hashable, Any], d))
        return {"value": d}

    if isinstance(row, Mapping):
        return _to_str_key_dict(cast(Mapping[Hashable, Any], row))

    try:
        d2 = dict(row)  # type: ignore[arg-type]
    except Exception:
        return None

    if isinstance(d2, Mapping):
        return _to_str_key_dict(cast(Mapping[Hashable, Any], d2))

    return None


# ============================================================================
# Tabla principal con selección de fila
# ============================================================================


def aggrid_with_row_click(df: pd.DataFrame, key_suffix: str) -> Optional[Dict[str, Any]]:
    """
    Muestra un AgGrid con selección de una sola fila.

    Args:
        df: DataFrame a renderizar. Si está vacío, muestra st.info y devuelve None.
        key_suffix: sufijo para key del componente AgGrid (evita colisiones).

    Returns:
        Un dict “normal” con los valores de la fila seleccionada o None.
    """
    if df.empty:
        st.info("No hay datos para mostrar.")
        return None

    desired_order = [
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
    visible_cols = [c for c in desired_order if c in df.columns]
    ordered_cols = visible_cols + [c for c in df.columns if c not in visible_cols]
    df = df[ordered_cols]

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_grid_options(domLayout="normal")

    for col in df.columns:
        if col not in visible_cols:
            gb.configure_column(col, hide=True)

    grid_options = gb.build()
    grid_options["autoSizeStrategy"] = {"type": "fitGridWidth"}

    import st_aggrid as st_aggrid_mod

    aggrid_fn = cast(AgGridCallable, getattr(st_aggrid_mod, "AgGrid"))

    grid_response = aggrid_fn(
        df,
        gridOptions=grid_options,
        update_on=["selectionChanged"],
        enable_enterprise_modules=False,
        height=520,
        key=f"aggrid_{key_suffix}",
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

    Nota:
    - Considera vacío: None y "".
    """
    if key in row and row.get(key) not in (None, ""):
        return row.get(key)
    if omdb_dict and isinstance(omdb_dict, Mapping):
        return omdb_dict.get(key)
    return None


def _safe_number_to_str(v: Any) -> str:
    """Convierte números/valores a string seguro para UI."""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "N/A"
        return str(v)
    except Exception:
        return "N/A"


def _safe_votes(v: Any) -> str:
    """Formatea votos con separador de miles; tolera strings tipo '12,345'."""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "N/A"
        if isinstance(v, str):
            v2 = v.replace(",", "")
            return f"{int(float(v2)):,}"
        return f"{int(float(v)):,}"
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


def render_detail_card(
    row: Dict[str, Any] | pd.Series | Mapping[str, Any] | None,
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

    col_left, col_right = st.columns([1, 2])

    with col_left:
        if _is_nonempty_str(poster_url):
            st.image(str(poster_url), width=280)
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
        header = str(title)
        try:
            if year is not None and not pd.isna(year):
                header += f" ({int(float(year))})"
        except Exception:
            pass

        st.markdown(f"### {header}")
        if library:
            st.write(f"**Biblioteca:** {library}")

        st.write(f"**Decisión:** `{decision}` — {reason}")

        m1, m2, m3 = st.columns(3)
        m1.metric("IMDb", _safe_number_to_str(imdb_rating))

        rt_str = _safe_number_to_str(rt_score)
        m2.metric("RT", f"{rt_str}%" if rt_str != "N/A" else "N/A")

        m3.metric("Votos", _safe_votes(imdb_votes))

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

        if _is_nonempty_str(plot):
            st.markdown("---")
            st.write("#### Sinopsis")
            st.write(str(plot))

        st.markdown("---")
        st.write("#### Archivo")
        if file_path:
            st.code(str(file_path), language="bash")

        if file_size is not None and not (isinstance(file_size, float) and pd.isna(file_size)):
            try:
                gb = float(file_size) / (1024**3)
                st.write(f"**Tamaño:** {gb:.2f} GB")
            except Exception:
                st.write(f"**Tamaño:** {file_size}")

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

    Requisitos de estado:
      - st.session_state["modal_open"] = True/False
      - st.session_state["modal_row"]  = dict de la fila seleccionada
    """
    if not st.session_state.get("modal_open"):
        return

    row = st.session_state.get("modal_row")
    if row is None:
        return

    c1, c2 = st.columns([10, 1])
    with c1:
        st.markdown("### 🔍 Detalle ampliado")
    with c2:
        if st.button("✖", key="close_modal"):
            st.session_state["modal_open"] = False
            st.session_state["modal_row"] = None
            _rerun()

    render_detail_card(row, show_modal_button=False)