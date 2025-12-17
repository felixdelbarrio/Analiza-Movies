from __future__ import annotations

import os
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder  # GridUpdateMode eliminado

from frontend.data_utils import safe_json_loads_single


# -------------------------------------------------------------------
# Tabla principal con selección de fila
# -------------------------------------------------------------------


def _normalize_selected_rows(selected_raw: Any) -> list[Mapping[str, Any]]:
    """Normaliza el objeto devuelto por AgGrid a una lista de mappings."""
    if selected_raw is None:
        return []

    # Caso DataFrame → lista de dicts
    if isinstance(selected_raw, pd.DataFrame):
        return selected_raw.to_dict(orient="records")

    # AgGrid normalmente devuelve list[dict]
    if isinstance(selected_raw, (list, tuple)):
        # Aseguramos que todos los elementos sean mappings o algo convertible luego
        return list(selected_raw)

    # Raro: un solo dict
    if isinstance(selected_raw, Mapping):
        return [selected_raw]

    # Intentar tratarlo como iterable genérico
    try:
        if hasattr(selected_raw, "__iter__") and not isinstance(
            selected_raw,
            (str, bytes),
        ):
            return list(selected_raw)
    except Exception:
        pass

    # Último recurso: devolverlo envuelto en lista para intentar dict() después
    return [selected_raw]


def aggrid_with_row_click(df: pd.DataFrame, key_suffix: str) -> Optional[Dict[str, Any]]:
    """
    Muestra un AgGrid con selección de una sola fila.
    Devuelve un dict con los valores de la fila seleccionada o None.
    """
    if df.empty:
        st.info("No hay datos para mostrar.")
        return None

    # Orden sugerido de columnas visibles
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

    # Ocultar columnas no prioritarias
    for col in df.columns:
        if col not in visible_cols:
            gb.configure_column(col, hide=True)

    grid_options = gb.build()
    # Nuevo autosize recomendado por st_aggrid:
    grid_options["autoSizeStrategy"] = {"type": "fitGridWidth"}

    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        # Reemplazo de GridUpdateMode.SELECTION_CHANGED
        update_on=["selectionChanged"],
        enable_enterprise_modules=False,
        height=520,
        key=f"aggrid_{key_suffix}",
    )

    selected_raw = grid_response.get("selected_rows")
    selected_rows = _normalize_selected_rows(selected_raw)

    if not selected_rows:
        return None

    first = selected_rows[0]

    if isinstance(first, pd.Series):
        return first.to_dict()
    if isinstance(first, Mapping):
        # Nos aseguramos de devolver un dict mutable “normal”
        return dict(first)

    # Último recurso: intentar dict() o envolverlo
    try:
        return dict(first)  # type: ignore[arg-type]
    except Exception:
        return {"value": first}


# -------------------------------------------------------------------
# Detalle de una película (panel tipo ficha)
# -------------------------------------------------------------------


def _normalize_row_to_dict(row: Any) -> Optional[Dict[str, Any]]:
    """Convierte distintas formas de fila (Series, dict, etc.) a dict."""
    if row is None:
        return None

    if isinstance(row, pd.Series):
        return row.to_dict()

    if isinstance(row, Mapping):
        return dict(row)

    try:
        return dict(row)  # type: ignore[arg-type]
    except Exception:
        # Demasiado raro para convertir; el caller decidirá qué hacer
        return None


def _get_from_omdb_or_row(
    row: Mapping[str, Any],
    omdb_dict: Mapping[str, Any] | None,
    key: str,
) -> Any:
    """Devuelve primero row[key] y, si no, omdb_dict[key]."""
    if key in row and row.get(key) not in (None, ""):
        return row.get(key)
    if omdb_dict and isinstance(omdb_dict, Mapping):
        return omdb_dict.get(key)
    return None


def _safe_number_to_str(v: Any) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "N/A"
        return str(v)
    except Exception:
        return "N/A"


def _safe_votes(v: Any) -> str:
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
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    lower = s.lower()
    return lower not in ("nan", "none")


def render_detail_card(
    row: Dict[str, Any] | pd.Series | Mapping[str, Any] | None,
    show_modal_button: bool = True,
    button_key_prefix: Optional[str] = None,
) -> None:
    """
    Panel lateral / ficha de detalle tipo Plex.
    Parseamos omdb_json SOLO para esta fila (si existe).

    button_key_prefix se usa para dar un key único al botón "Abrir en ventana"
    por pestaña (all / candidates / advanced, etc.) y evitar colisiones.
    """
    if row is None:
        st.info("Haz click en una fila para ver su detalle.")
        return

    normalized = _normalize_row_to_dict(row)
    if normalized is None:
        st.warning("Detalle no disponible: fila con formato inesperado.")
        return

    row = normalized  # a partir de aquí asumimos dict[str, Any]

    omdb_dict: Mapping[str, Any] | None = None
    if "omdb_json" in row:
        try:
            parsed = safe_json_loads_single(row.get("omdb_json"))
            if isinstance(parsed, Mapping):
                omdb_dict = parsed
        except Exception:
            omdb_dict = None

    title = row.get("title", "¿Sin título?")
    year = row.get("year")
    library = row.get("library")
    decision = row.get("decision")
    reason = row.get("reason")
    imdb_rating = row.get("imdb_rating")
    imdb_votes = row.get("imdb_votes")
    rt_score = row.get("rt_score")
    plex_rating = row.get("plex_rating")  # ahora mismo no se muestra, pero lo dejamos

    poster_url = row.get("poster_url")
    file_path = row.get("file")
    file_size = row.get("file_size")
    trailer_url = row.get("trailer_url")
    rating_key = row.get("rating_key")
    imdb_id = row.get("imdb_id")

    rated = _get_from_omdb_or_row(row, omdb_dict, "Rated")
    released = _get_from_omdb_or_row(row, omdb_dict, "Released")
    runtime = _get_from_omdb_or_row(row, omdb_dict, "Runtime")
    genre = _get_from_omdb_or_row(row, omdb_dict, "Genre")
    director = _get_from_omdb_or_row(row, omdb_dict, "Director")
    writer = _get_from_omdb_or_row(row, omdb_dict, "Writer")
    actors = _get_from_omdb_or_row(row, omdb_dict, "Actors")
    language = _get_from_omdb_or_row(row, omdb_dict, "Language")
    country = _get_from_omdb_or_row(row, omdb_dict, "Country")
    awards = _get_from_omdb_or_row(row, omdb_dict, "Awards")
    plot = _get_from_omdb_or_row(row, omdb_dict, "Plot")

    col_left, col_right = st.columns([1, 2])

    # POSTER + enlaces
    with col_left:
        if _is_nonempty_str(poster_url):
            st.image(poster_url, width=280)
        else:
            st.write("📷 Sin póster")

        if imdb_id:
            imdb_url = f"https://www.imdb.com/title/{imdb_id}"
            st.markdown(f"[🎬 Ver en IMDb]({imdb_url})")

        # Leer base de Plex desde varias posibles vars de entorno para mayor compatibilidad
        plex_base = (
            os.getenv("PLEX_WEB_BASEURL")
            or os.getenv("PLEX_BASEURL")
            or os.getenv("BASEURL")
            or ""
        )
        if plex_base and rating_key:
            plex_url = f"{plex_base}/web/index.html#!/server/library/metadata/{rating_key}"
            st.markdown(f"[📺 Ver en Plex Web]({plex_url})")

        if show_modal_button:
            key_suffix = button_key_prefix or "default"
            button_key = f"open_modal_{key_suffix}"
            if st.button("🪟 Abrir en ventana", key=button_key):
                st.session_state["modal_row"] = row
                st.session_state["modal_open"] = True
                st.experimental_rerun()

    # DETALLE
    with col_right:
        header = str(title)
        try:
            if year is not None and not pd.isna(year):
                header += f" ({int(float(year))})"
        except Exception:
            # ignorar año inválido
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

        if file_size is not None and not (
            isinstance(file_size, float) and pd.isna(file_size)
        ):
            try:
                gb = float(file_size) / (1024**3)
                st.write(f"**Tamaño:** {gb:.2f} GB")
            except Exception:
                st.write(f"**Tamaño:** {file_size}")

        if _is_nonempty_str(trailer_url):
            st.markdown("#### 🎞 Tráiler")
            st.video(trailer_url)

    with st.expander("Ver JSON completo"):
        try:
            full_row: MutableMapping[str, Any] = dict(row)
        except Exception:
            full_row = {"value": str(row)}
        if omdb_dict is not None:
            full_row["_omdb_parsed"] = dict(omdb_dict)
        st.json(full_row)


# -------------------------------------------------------------------
# Modal de detalle ampliado
# -------------------------------------------------------------------


def render_modal() -> None:
    """Vista de detalle ampliado usando el estado global de Streamlit."""
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
            st.experimental_rerun()

    render_detail_card(row, show_modal_button=False)