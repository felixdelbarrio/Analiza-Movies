# -*- coding: utf-8 -*-
"""
delete_tab.py

Pestana 4 del dashboard (Streamlit): borrado controlado de archivos a partir del
CSV filtrado (DELETE/MAYBE).

Importante (anti-circular-import):
- NO importamos delete_files_from_rows a nivel de modulo.
- Lo importamos dentro de render() (lazy import) para evitar ciclos backend<->frontend.

Notas st_aggrid:
- update_on=["selectionChanged"] (en lugar de GridUpdateMode)
- autoSizeStrategy = {"type": "fitGridWidth"}

Nota importante (Pyright/Pylance)
- Evitamos falsos positivos "Statement is unreachable" por stubs rotos (NoReturn)
  o tipos demasiado estrechos.
- Estrategia:
  - Normalizamos filas con UNA funcion (_row_from_any).
  - Evitamos que Pyright marque ramas como imposibles:
    * No usamos pd.isna() directo (wrapper _pd_isna).
    * Evitamos ramas/returns que Pyright infiere como inalcanzables.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Callable, Protocol, cast

import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder, JsCode

from frontend.components import render_grid_toolbar
from frontend.data_utils import add_derived_columns

Row = Mapping[str, Any]
Rows = Sequence[Row]
DeleteFilesFromRowsFn = Callable[[pd.DataFrame, bool], tuple[int, int, Sequence[str]]]

HIDDEN_COLUMNS: set[str] = {"plex_ta"}
_DECISION_LABELS: dict[str, str] = {
    "DELETE": "🟥 DELETE",
    "MAYBE": "🟨 MAYBE",
    "KEEP": "🟩 KEEP",
    "UNKNOWN": "⬜ UNKNOWN",
}
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
# Helpers seleccion / tamanos
# ============================================================================


def _pd_isna(value: Any) -> bool:
    """
    Wrapper para pd.isna() evitando que stubs rotos propaguen NoReturn/unreachable.
    """
    pd_any: Any = pd
    fn = getattr(pd_any, "isna", None)
    if not callable(fn):
        return False
    try:
        return bool(fn(value))
    except Exception:
        return False


def _to_str_key_dict(src: Mapping[Hashable, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in src.items():
        out[str(k)] = v
    return out


def _row_from_any(value: Any) -> dict[str, Any]:
    """
    Convierte cualquier cosa razonable a dict[str, Any] de forma defensiva.

    Anti-pyright-unreachable:
    - Evita ramas "else" que Pyright marca como imposibles.
    - No hace suposiciones agresivas sobre tipos.
    """
    # pandas Series -> dict
    if isinstance(value, pd.Series):
        try:
            value = value.to_dict()
        except Exception:
            return {"value": value}

    # Mapping directo
    if isinstance(value, Mapping):
        return _to_str_key_dict(cast(Mapping[Hashable, Any], value))

    # Iterable de pares (k, v) -> dict (si se puede)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        v_any: Any = value  # rompe narrowing agresivo
        try:
            as_dict_any: Any = dict(v_any)  # type: ignore[arg-type]
        except Exception:
            return {"value": value}

        if isinstance(as_dict_any, Mapping):
            return _to_str_key_dict(cast(Mapping[Hashable, Any], as_dict_any))

        return {"value": as_dict_any}

    return {"value": value}


def _normalize_selected_rows(selected_raw: Any) -> list[dict[str, Any]]:
    if selected_raw is None:
        return []

    if isinstance(selected_raw, pd.DataFrame):
        records = selected_raw.to_dict(orient="records")
        out_df: list[dict[str, Any]] = []
        for rec in records:
            out_df.append(_row_from_any(rec))
        return out_df

    if isinstance(selected_raw, pd.Series):
        return [_row_from_any(selected_raw)]

    if isinstance(selected_raw, Mapping):
        return [_row_from_any(selected_raw)]

    if isinstance(selected_raw, (list, tuple)):
        out_list: list[dict[str, Any]] = []
        for item in selected_raw:
            out_list.append(_row_from_any(item))
        return out_list

    if isinstance(selected_raw, Iterable) and not isinstance(
        selected_raw, (str, bytes)
    ):
        out_it: list[dict[str, Any]] = []
        for x in selected_raw:
            out_it.append(_row_from_any(x))
        return out_it

    return [_row_from_any(selected_raw)]


def _safe_float(x: Any) -> float | None:
    """
    Convierte a float de forma defensiva (acepta '1,234', NaN, etc.).
    Anti-pyright-unreachable:
    - No usa pd.isna() directo (usa _pd_isna).
    - Evita asignaciones/ramas que Pyright estrecha de forma rara.
    """
    if x is None:
        return None

    # NaN típico (float/np.float)
    if isinstance(x, float) and _pd_isna(x):
        return None

    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    try:
        return float(x)
    except Exception:
        return None


def _compute_total_size_gb(rows: Rows) -> float | None:
    if not rows:
        return None

    total_bytes = 0.0
    any_size = False

    for r in rows:
        val = _safe_float(r.get("file_size"))
        if val is None:
            continue
        if val >= 0:
            total_bytes += val
            any_size = True

    if not any_size or total_bytes <= 0:
        return None

    return total_bytes / (1024**3)


def _count_existing_files(rows: Rows) -> int:
    from pathlib import Path

    n = 0
    for r in rows:
        raw = r.get("file")
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        try:
            p = Path(s).expanduser().resolve()
        except Exception:
            continue
        try:
            if p.exists() and p.is_file():
                n += 1
        except Exception:
            continue
    return n


def _truncate_logs(lines: Sequence[str], max_lines: int = 400) -> list[str]:
    if len(lines) <= max_lines:
        return list(lines)
    head = list(lines[: max_lines // 2])
    tail = list(lines[-max_lines // 2 :])
    return (
        head
        + [f"... ({len(lines) - len(head) - len(tail)} lineas omitidas) ..."]
        + tail
    )


# ============================================================================
# Render
# ============================================================================


def render(
    df_filtered: pd.DataFrame | None,
    delete_dry_run: bool,
    delete_require_confirm: bool,
) -> None:
    import frontend.delete_logic as delete_logic

    delete_files_from_rows_fn = cast(
        DeleteFilesFromRowsFn,
        getattr(delete_logic, "delete_files_from_rows"),
    )

    import st_aggrid as st_aggrid_mod

    aggrid_fn = cast(AgGridCallable, getattr(st_aggrid_mod, "AgGrid"))

    st.write("### Candidatas (borrado controlado)")

    if df_filtered is None or df_filtered.empty:
        st.info("No hay CSV filtrado. Ejecuta primero el analisis.")
        return

    st.warning(
        "Cuidado: aqui puedes borrar archivos fisicamente.\n\n"
        f"- DELETE_DRY_RUN = `{delete_dry_run}`\n"
        f"- DELETE_REQUIRE_CONFIRM = `{delete_require_confirm}`"
    )

    df_view = df_filtered

    st.write("Filtra las peliculas que quieras borrar y seleccionalas en la tabla:")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        if "library" in df_view.columns:
            series = df_view["library"].dropna().astype(str).map(str.strip)
            series = series.mask(series == "", pd.NA).dropna()
            libs = series.unique().tolist()
            libs.sort()
        else:
            libs = []

        lib_filter = st.multiselect("Biblioteca", libs, key="lib_filter_delete")

    with col_f2:
        if "decision" in df_view.columns:
            dec_filter = st.multiselect(
                "Decision",
                ["DELETE", "MAYBE"],
                default=["DELETE", "MAYBE"],
                format_func=lambda v: _DECISION_LABELS.get(v, v),
                key="dec_filter_delete",
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

    df_view = add_derived_columns(df_view)

    desired_order = [
        "title",
        "year",
        "library",
        "file_size_gb",
        "imdb_rating",
        "metacritic_score",
        "rt_score",
        "reason",
        "file",
    ]
    visible_cols = [c for c in desired_order if c in df_view.columns]
    ordered_cols = visible_cols + [c for c in df_view.columns if c not in visible_cols]
    df_view = df_view[ordered_cols]

    df_view, search_query, grid_height = render_grid_toolbar(
        df_view,
        key_suffix="delete",
        download_filename="delete_table.csv",
        show_search=False,
    )
    if df_view.empty:
        if search_query.strip():
            st.info("No hay filas que coincidan con la búsqueda.")
        else:
            st.info("No hay filas para mostrar.")
        return

    gb = GridOptionsBuilder.from_dataframe(df_view)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    if bool(st.session_state.get("grid_colorize_rows", True)):
        gb.configure_grid_options(
            domLayout="normal",
            getRowStyle=_DECISION_ROW_STYLE,
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
    else:
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
        if col == "library":
            col_def["cellStyle"] = {"justifyContent": "flex-start", "textAlign": "left"}
        if col == "file":
            col_def["cellStyle"] = {"justifyContent": "flex-start", "textAlign": "left"}
        if col in {"year", "file_size_gb", "imdb_rating", "imdb_votes", "rt_score"}:
            col_def["cellStyle"] = {"justifyContent": "flex-end", "textAlign": "right"}
        if col == "reason":
            col_def["cellStyle"] = {"justifyContent": "flex-start", "textAlign": "left"}
        if col_def:
            gb.configure_column(col, **col_def)
    auto_size_js = JsCode(
        """
function(params) {
  if (!params || !params.api || !params.columnApi) {
    return;
  }
  const cols = ['year','file_size_gb','imdb_rating','imdb_votes','rt_score'];
  setTimeout(function() {
    params.columnApi.autoSizeColumns(cols, true);
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
    gb.configure_grid_options(enableCellTextSelection=True, ensureDomOrder=True)
    for col in df_view.columns:
        if col not in visible_cols:
            gb.configure_column(col, hide=True)
    for col in HIDDEN_COLUMNS:
        if col in df_view.columns:
            gb.configure_column(col, hide=True)
    grid_options = gb.build()

    colorize_rows = bool(st.session_state.get("grid_colorize_rows", True))
    grid_response = aggrid_fn(
        df_view.copy(),
        gridOptions=grid_options,
        update_on=["selectionChanged"],
        enable_enterprise_modules=False,
        height=grid_height,
        allow_unsafe_jscode=True,
        key=f"aggrid_delete_{int(colorize_rows)}",
    )

    selected_rows = _normalize_selected_rows(grid_response.get("selected_rows"))

    num_selected = len(selected_rows)
    st.write(f"Peliculas seleccionadas: **{num_selected}**")

    if num_selected > 0:
        existing = _count_existing_files(selected_rows)
        st.caption(
            f"Ficheros reales detectados en disco (de la seleccion): **{existing}**"
        )

    total_gb = _compute_total_size_gb(selected_rows)
    if total_gb is not None:
        st.write(f"Tamano total estimado (segun `file_size`): **{total_gb:.2f} GB**")

    if num_selected == 0:
        return

    confirm = True
    if delete_require_confirm:
        confirm = st.checkbox(
            "Confirmo que quiero borrar fisicamente los archivos seleccionados.",
            key="delete_confirm_checkbox",
        )

    if st.button("Ejecutar borrado", type="primary", key="btn_delete_exec"):
        if not confirm:
            st.warning("Marca la casilla de confirmacion antes de borrar.")
            return

        df_sel = pd.DataFrame(selected_rows)
        ok, err, logs = delete_files_from_rows_fn(df_sel, delete_dry_run)

        if delete_dry_run:
            st.success(
                f"DRY RUN completado. Se habrian borrado {ok} archivo(s), {err} error(es)."
            )
        else:
            st.success(f"Borrado completado. OK={ok}, errores={err}")

        logs_show = _truncate_logs([str(log_line) for log_line in logs])
        st.text_area("Log de borrado", value="\n".join(logs_show), height=260)
