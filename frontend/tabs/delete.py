from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder  # GridUpdateMode eliminado

from backend.delete_logic import delete_files_from_rows


def _normalize_selected_rows(selected_raw: Any) -> list[dict[str, Any]]:
    """
    Normaliza lo que devuelve AgGrid a una lista de dicts.

    Acepta:
      - None → []
      - list/tuple de dicts/Series/objetos mapeables
      - DataFrame
      - dict (una sola fila)
      - otros iterables → list(...)
    """
    if selected_raw is None:
        return []

    # DataFrame -> registros
    if isinstance(selected_raw, pd.DataFrame):
        return selected_raw.to_dict(orient="records")

    # Lista o tupla
    if isinstance(selected_raw, (list, tuple)):
        rows: list[dict[str, Any]] = []
        for item in selected_raw:
            if isinstance(item, pd.Series):
                rows.append(item.to_dict())
            elif isinstance(item, dict):
                rows.append(item)
            else:
                try:
                    rows.append(dict(item))  # type: ignore[arg-type]
                except Exception:
                    rows.append({"value": item})
        return rows

    # dict -> una sola fila
    if isinstance(selected_raw, dict):
        return [selected_raw]

    # Otros iterables (no str/bytes)
    if isinstance(selected_raw, Iterable) and not isinstance(
        selected_raw,
        (str, bytes),
    ):
        out: list[dict[str, Any]] = []
        for x in selected_raw:
            try:
                out.append(dict(x))  # type: ignore[arg-type]
            except Exception:
                out.append({"value": x})
        return out

    # Fallback: envolver en una fila genérica
    return [{"value": selected_raw}]


def _compute_total_size_gb(rows: list[dict[str, Any]]) -> float | None:
    """Calcula el tamaño total en GB de las filas seleccionadas, si hay `file_size` en bytes."""
    if not rows:
        return None

    total_bytes = 0.0
    for r in rows:
        raw = r.get("file_size")
        if raw is None:
            continue
        try:
            val = float(raw)
        except Exception:
            continue
        if val >= 0:
            total_bytes += val

    if total_bytes <= 0:
        return None

    return total_bytes / (1024**3)


def render(
    df_filtered: pd.DataFrame | None,
    delete_dry_run: bool,
    delete_require_confirm: bool,
) -> None:
    """Pestaña 4: Borrado controlado de archivos."""
    st.write("### Borrado controlado de archivos")

    if df_filtered is None or df_filtered.empty:
        st.info("No hay CSV filtrado. Ejecuta primero el análisis.")
        return

    st.warning(
        "⚠️ Cuidado: aquí puedes borrar archivos físicamente.\n\n"
        f"- DELETE_DRY_RUN = `{delete_dry_run}`\n"
        f"- DELETE_REQUIRE_CONFIRM = `{delete_require_confirm}`"
    )

    df_view = df_filtered.copy()

    # ----------------------------
    # Filtros básicos
    # ----------------------------
    st.write("Filtra las películas que quieras borrar y selecciónalas en la tabla:")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        if "library" in df_view.columns:
            libs = (
                df_view["library"]
                .dropna()
                .astype(str)
                .map(str.strip)
                .replace({"": None})
                .dropna()
                .unique()
                .tolist()
            )
            libs.sort()
        else:
            libs = []

        lib_filter = st.multiselect(
            "Biblioteca",
            libs,
            key="lib_filter_delete",
        )

    with col_f2:
        if "decision" in df_view.columns:
            dec_filter = st.multiselect(
                "Decisión",
                ["DELETE", "MAYBE"],
                default=["DELETE", "MAYBE"],
            )
        else:
            dec_filter = []

    if lib_filter and "library" in df_view.columns:
        df_view = df_view[df_view["library"].isin(lib_filter)]
    if dec_filter and "decision" in df_view.columns:
        df_view = df_view[df_view["decision"].isin(dec_filter)]

    if df_view.empty:
        st.info("No hay filas que coincidan con los filtros actuales.")
        return

    # ----------------------------
    # Tabla interactiva (AgGrid)
    # ----------------------------
    gb = GridOptionsBuilder.from_dataframe(df_view)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_grid_options(domLayout="normal")
    grid_options = gb.build()

    # Sustituimos fit_columns_on_grid_load por autoSizeStrategy
    grid_options["autoSizeStrategy"] = {"type": "fitGridWidth"}

    grid_response = AgGrid(
        df_view,
        gridOptions=grid_options,
        # Reemplazo de GridUpdateMode.SELECTION_CHANGED
        update_on=["selectionChanged"],
        enable_enterprise_modules=False,
        height=500,
        key="aggrid_delete",
    )

    selected_rows_raw = grid_response.get("selected_rows")
    selected_rows = _normalize_selected_rows(selected_rows_raw)

    num_selected = len(selected_rows)
    st.write(f"Películas seleccionadas: **{num_selected}**")

    total_gb = _compute_total_size_gb(selected_rows)
    if total_gb is not None:
        st.write(f"Tamaño total de los archivos seleccionados: **{total_gb:.2f} GB**")

    # ----------------------------
    # Botón de borrado
    # ----------------------------
    if num_selected == 0:
        return

    if delete_require_confirm:
        confirm = st.checkbox(
            "Confirmo que quiero borrar físicamente los archivos seleccionados.",
            key="delete_confirm_checkbox",
        )
    else:
        confirm = True

    if st.button("🗑️ Ejecutar borrado", type="primary"):
        if not confirm:
            st.warning("Marca la casilla de confirmación antes de borrar.")
            return

        df_sel = pd.DataFrame(selected_rows)
        ok, err, logs = delete_files_from_rows(df_sel, delete_dry_run)

        if delete_dry_run:
            st.success(
                f"DRY RUN completado. Se habrían borrado {ok} archivo(s), "
                f"{err} error(es)."
            )
        else:
            st.success(f"Borrado completado. OK={ok}, errores={err}")

        st.text_area(
            "Log de borrado",
            value="\n".join(str(l) for l in logs),
            height=220,
        )