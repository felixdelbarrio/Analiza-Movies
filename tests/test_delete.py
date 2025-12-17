from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import pandas as pd
import pytest

# -------------------------------------------------------------------
# Localizar proyecto y rutas de frontend/tabs de forma robusta
# -------------------------------------------------------------------

THIS_FILE = pathlib.Path(__file__).resolve()

PROJECT_ROOT: pathlib.Path | None = None
for parent in THIS_FILE.parents:
    candidate = parent / "frontend"
    if candidate.exists() and candidate.is_dir():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    pytest.skip(
        "No se ha encontrado un directorio 'frontend' en ningún padre del test; "
        "se omiten tests de delete tab.",
        allow_module_level=True,
    )

FRONTEND_DIR = PROJECT_ROOT / "frontend"
TABS_DIR = FRONTEND_DIR / "tabs"

DELETE_PATH = TABS_DIR / "delete.py"

if not DELETE_PATH.exists():
    pytest.skip(
        f"frontend/tabs/delete.py no encontrado en {DELETE_PATH}, "
        "se omiten tests de delete tab.",
        allow_module_level=True,
    )

# -------------------------------------------------------------------
# Crear paquetes sintéticos `frontend` y `frontend.tabs` si es necesario
# -------------------------------------------------------------------
import types

frontend_pkg = sys.modules.get("frontend")
if frontend_pkg is None:
    frontend_pkg = types.ModuleType("frontend")
    frontend_pkg.__path__ = [str(FRONTEND_DIR)]
    sys.modules["frontend"] = frontend_pkg

tabs_pkg = sys.modules.get("frontend.tabs")
if tabs_pkg is None:
    tabs_pkg = types.ModuleType("frontend.tabs")
    tabs_pkg.__path__ = [str(TABS_DIR)]
    sys.modules["frontend.tabs"] = tabs_pkg

# -------------------------------------------------------------------
# Cargar frontend.tabs.delete
# -------------------------------------------------------------------
spec_del = importlib.util.spec_from_file_location(
    "frontend.tabs.delete",
    DELETE_PATH,
)
if spec_del is None or spec_del.loader is None:
    raise ImportError(f"No se pudo crear spec para {DELETE_PATH}")

delete_tab = importlib.util.module_from_spec(spec_del)
sys.modules["frontend.tabs.delete"] = delete_tab
spec_del.loader.exec_module(delete_tab)  # type: ignore[assignment]


# -------------------------------------------------------------------
# Tests helpers: _normalize_selected_rows
# -------------------------------------------------------------------


def test_normalize_selected_rows_none() -> None:
    f = delete_tab._normalize_selected_rows  # type: ignore[attr-defined]
    assert f(None) == []


def test_normalize_selected_rows_dataframe() -> None:
    f = delete_tab._normalize_selected_rows  # type: ignore[attr-defined]
    df = pd.DataFrame([{"a": 1}, {"a": 2}])
    out = f(df)
    assert out == df.to_dict(orient="records")


def test_normalize_selected_rows_list_mixed() -> None:
    f = delete_tab._normalize_selected_rows  # type: ignore[attr-defined]

    s = pd.Series({"x": 1})
    lst = [
        {"a": 1},
        s,
        [("k", "v")],  # convertible a dict
        123,  # no convertible → {"value": 123}
    ]

    out = f(lst)
    assert out[0] == {"a": 1}
    assert out[1] == s.to_dict()
    assert out[2] == {"k": "v"}
    assert out[3] == {"value": 123}


def test_normalize_selected_rows_single_dict() -> None:
    f = delete_tab._normalize_selected_rows  # type: ignore[attr-defined]
    d = {"a": 1}
    out = f(d)
    assert out == [d]


def test_normalize_selected_rows_other_iterable() -> None:
    f = delete_tab._normalize_selected_rows  # type: ignore[attr-defined]

    class MyRow:
        def __iter__(self):
            return iter([("foo", 1), ("bar", 2)])

    iterable = [MyRow(), "no-dict"]
    out = f(iterable)
    assert out[0] == {"foo": 1, "bar": 2}
    assert out[1] == {"value": "no-dict"}


def test_normalize_selected_rows_fallback_scalar() -> None:
    f = delete_tab._normalize_selected_rows  # type: ignore[attr-defined]
    out = f(999)
    assert out == [{"value": 999}]


# -------------------------------------------------------------------
# Tests helper: _compute_total_size_gb
# -------------------------------------------------------------------


def test_compute_total_size_gb_empty() -> None:
    f = delete_tab._compute_total_size_gb  # type: ignore[attr-defined]
    assert f([]) is None


def test_compute_total_size_gb_valid_rows() -> None:
    f = delete_tab._compute_total_size_gb  # type: ignore[attr-defined]
    rows = [
        {"file_size": 1024**3},       # 1 GB
        {"file_size": 2 * 1024**3},   # 2 GB
        {"file_size": None},
    ]
    total = f(rows)
    assert total == pytest.approx(3.0, rel=1e-6)


def test_compute_total_size_gb_invalid_and_negative() -> None:
    f = delete_tab._compute_total_size_gb  # type: ignore[attr-defined]
    rows = [
        {"file_size": "bad"},
        {"file_size": -100},
        {"file_size": None},
    ]
    assert f(rows) is None


# -------------------------------------------------------------------
# Fixture para mockear streamlit
# -------------------------------------------------------------------


@pytest.fixture
def fake_streamlit(monkeypatch):
    """Parchea las funciones principales de streamlit usadas en delete.render."""
    import streamlit as st

    # st.write / info / warning / success / text_area
    monkeypatch.setattr(st, "write", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "info", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "warning", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "success", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "text_area", lambda *a, **k: None, raising=False)

    # columns devuelve N objetos dummy con context manager
    class DummyCol:
        def __enter__(self, *a, **k):
            return self

        def __exit__(self, *a, **k):
            return False

        def write(self, *a, **k):
            return None

        def info(self, *a, **k):
            return None

    def fake_columns(spec):
        if isinstance(spec, int):
            n = spec
        else:
            try:
                n = len(spec)
            except TypeError:
                n = 1
        return [DummyCol() for _ in range(n)]

    monkeypatch.setattr(st, "columns", fake_columns, raising=False)

    # multiselect: devolver siempre todas las opciones por defecto
    def fake_multiselect(label, options, *args, **kwargs):
        return list(options)

    monkeypatch.setattr(st, "multiselect", fake_multiselect, raising=False)

    # checkbox: por defecto False (tests lo pueden parchear)
    monkeypatch.setattr(st, "checkbox", lambda *a, **k: False, raising=False)

    # button: por defecto False (tests lo pueden parchear)
    monkeypatch.setattr(st, "button", lambda *a, **k: False, raising=False)

    return st


# -------------------------------------------------------------------
# Tests de render()
# -------------------------------------------------------------------


def test_render_no_filtered_df(fake_streamlit, monkeypatch) -> None:
    """Si df_filtered es None o vacío, debe mostrar info y no invocar AgGrid ni delete_files_from_rows."""
    import streamlit as st

    info_msgs: list[str] = []

    def fake_info(msg, *a, **k):
        info_msgs.append(str(msg))

    monkeypatch.setattr(st, "info", fake_info, raising=False)

    def fake_AgGrid(*a, **k):
        raise AssertionError("AgGrid no debe llamarse cuando df_filtered es None/vacío")

    monkeypatch.setattr(delete_tab, "AgGrid", fake_AgGrid, raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        delete_tab,
        "delete_files_from_rows",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("delete_files_from_rows no debe llamarse"),
        ),
        raising=False,
    )  # type: ignore[attr-defined]

    delete_tab.render(None, True, True)  # type: ignore[attr-defined]

    df_empty = pd.DataFrame([])
    delete_tab.render(df_empty, True, True)  # type: ignore[attr-defined]

    assert any("No hay CSV filtrado" in m for m in info_msgs)


def test_render_filters_to_empty(fake_streamlit, monkeypatch) -> None:
    """Si los filtros dejan el DataFrame vacío, no debe llamarse AgGrid."""
    import streamlit as st

    # multiselect de biblioteca devuelve valor que no existe
    def fake_multiselect(label, options, *a, **k):
        if label == "Biblioteca":
            return ["NO_EXISTE"]
        return []

    monkeypatch.setattr(st, "multiselect", fake_multiselect, raising=False)

    info_msgs: list[str] = []

    def fake_info(msg, *a, **k):
        info_msgs.append(str(msg))

    monkeypatch.setattr(st, "info", fake_info, raising=False)

    def fake_AgGrid(*a, **k):
        raise AssertionError("AgGrid no debe llamarse si df_view queda vacío")

    monkeypatch.setattr(delete_tab, "AgGrid", fake_AgGrid, raising=False)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        delete_tab,
        "delete_files_from_rows",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("delete_files_from_rows no debe llamarse"),
        ),
        raising=False,
    )  # type: ignore[attr-defined]

    df = pd.DataFrame(
        [
            {"title": "A", "library": "Lib1", "decision": "DELETE", "file_size": 100},
            {"title": "B", "library": "Lib2", "decision": "MAYBE", "file_size": 200},
        ]
    )

    delete_tab.render(df, True, True)  # type: ignore[attr-defined]

    assert any("No hay filas que coincidan" in m for m in info_msgs)


def test_render_with_rows_no_selection(fake_streamlit, monkeypatch) -> None:
    """Si AgGrid devuelve selección vacía, no debe invocarse delete_files_from_rows."""
    # Fake GridOptionsBuilder
    class DummyGB:
        def __init__(self, df):
            self.df = df

        @classmethod
        def from_dataframe(cls, df):
            return cls(df)

        def configure_selection(self, *a, **k):
            return None

        def configure_grid_options(self, *a, **k):
            return None

        def build(self):
            return {"dummy": True}

    monkeypatch.setattr(
        delete_tab,
        "GridOptionsBuilder",
        DummyGB,
        raising=False,
    )  # type: ignore[attr-defined]

    def fake_AgGrid(df, **kwargs):
        return {"selected_rows": []}

    monkeypatch.setattr(delete_tab, "AgGrid", fake_AgGrid, raising=False)  # type: ignore[attr-defined]

    def fake_delete(*a, **k):
        raise AssertionError("delete_files_from_rows no debe llamarse sin selección")

    monkeypatch.setattr(delete_tab, "delete_files_from_rows", fake_delete, raising=False)  # type: ignore[attr-defined]

    df = pd.DataFrame(
        [
            {"title": "A", "library": "Lib1", "decision": "DELETE", "file_size": 100},
        ]
    )

    delete_tab.render(df, True, True)  # type: ignore[attr-defined]


def test_render_delete_with_confirm_not_checked(fake_streamlit, monkeypatch) -> None:
    """Con confirmación requerida y checkbox sin marcar no debe borrar; muestra warning."""
    import streamlit as st

    class DummyGB:
        def __init__(self, df):
            self.df = df

        @classmethod
        def from_dataframe(cls, df):
            return cls(df)

        def configure_selection(self, *a, **k):
            return None

        def configure_grid_options(self, *a, **k):
            return None

        def build(self):
            return {}

    monkeypatch.setattr(delete_tab, "GridOptionsBuilder", DummyGB, raising=False)  # type: ignore[attr-defined]

    def fake_AgGrid(df, **kwargs):
        return {
            "selected_rows": [
                {"title": "A", "file_size": 1024**3},
            ]
        }

    monkeypatch.setattr(delete_tab, "AgGrid", fake_AgGrid, raising=False)  # type: ignore[attr-defined]

    # checkbox -> False (no confirma)
    monkeypatch.setattr(st, "checkbox", lambda *a, **k: False, raising=False)

    # button -> True (usuario pulsa borrar)
    monkeypatch.setattr(st, "button", lambda *a, **k: True, raising=False)

    warnings_msgs: list[str] = []

    def fake_warning(msg, *a, **k):
        warnings_msgs.append(str(msg))

    monkeypatch.setattr(st, "warning", fake_warning, raising=False)

    def fake_delete(*a, **k):
        raise AssertionError("delete_files_from_rows no debe llamarse si no hay confirmación")

    monkeypatch.setattr(delete_tab, "delete_files_from_rows", fake_delete, raising=False)  # type: ignore[attr-defined]

    df = pd.DataFrame(
        [
            {"title": "A", "library": "Lib1", "decision": "DELETE", "file_size": 1024**3},
        ]
    )

    delete_tab.render(df, delete_dry_run=True, delete_require_confirm=True)  # type: ignore[attr-defined]

    assert any("Marca la casilla de confirmación" in m for m in warnings_msgs)


def test_render_delete_dry_run_with_confirm_ok(fake_streamlit, monkeypatch) -> None:
    """Con confirmación y DRY RUN, debe llamar a delete_files_from_rows y mostrar mensaje de DRY RUN completado."""
    import streamlit as st

    class DummyGB:
        def __init__(self, df):
            self.df = df

        @classmethod
        def from_dataframe(cls, df):
            return cls(df)

        def configure_selection(self, *a, **k):
            return None

        def configure_grid_options(self, *a, **k):
            return None

        def build(self):
            return {}

    monkeypatch.setattr(delete_tab, "GridOptionsBuilder", DummyGB, raising=False)  # type: ignore[attr-defined]

    def fake_AgGrid(df, **kwargs):
        return {
            "selected_rows": [
                {"title": "A", "file_size": 1024**3},
                {"title": "B", "file_size": 2 * 1024**3},
            ]
        }

    monkeypatch.setattr(delete_tab, "AgGrid", fake_AgGrid, raising=False)  # type: ignore[attr-defined]

    # checkbox -> True (confirma)
    monkeypatch.setattr(st, "checkbox", lambda *a, **k: True, raising=False)

    # button -> True
    monkeypatch.setattr(st, "button", lambda *a, **k: True, raising=False)

    calls = {"called": False, "args": None}

    def fake_delete(df_sel, dry_run):
        calls["called"] = True
        calls["args"] = (df_sel, dry_run)
        return 2, 0, ["log1", "log2"]

    monkeypatch.setattr(delete_tab, "delete_files_from_rows", fake_delete, raising=False)  # type: ignore[attr-defined]

    success_msgs: list[str] = []

    def fake_success(msg, *a, **k):
        success_msgs.append(str(msg))

    monkeypatch.setattr(st, "success", fake_success, raising=False)

    df = pd.DataFrame(
        [
            {"title": "A", "library": "Lib1", "decision": "DELETE", "file_size": 1024**3},
            {"title": "B", "library": "Lib1", "decision": "MAYBE", "file_size": 2 * 1024**3},
        ]
    )

    delete_tab.render(df, delete_dry_run=True, delete_require_confirm=True)  # type: ignore[attr-defined]

    assert calls["called"] is True
    df_sel, dry_run_flag = calls["args"]
    assert isinstance(df_sel, pd.DataFrame)
    assert len(df_sel) == 2
    assert dry_run_flag is True
    assert any("DRY RUN completado" in m for m in success_msgs)


def test_render_delete_without_confirm_flag(fake_streamlit, monkeypatch) -> None:
    """Si delete_require_confirm=False, no se usa checkbox y se borra directamente al pulsar el botón."""
    import streamlit as st

    class DummyGB:
        def __init__(self, df):
            self.df = df

        @classmethod
        def from_dataframe(cls, df):
            return cls(df)

        def configure_selection(self, *a, **k):
            return None

        def configure_grid_options(self, *a, **k):
            return None

        def build(self):
            return {}

    monkeypatch.setattr(delete_tab, "GridOptionsBuilder", DummyGB, raising=False)  # type: ignore[attr-defined]

    def fake_AgGrid(df, **kwargs):
        return {"selected_rows": [{"title": "A", "file_size": 100}]}

    monkeypatch.setattr(delete_tab, "AgGrid", fake_AgGrid, raising=False)  # type: ignore[attr-defined]

    # button -> True
    monkeypatch.setattr(st, "button", lambda *a, **k: True, raising=False)

    called = {"count": 0}

    def fake_delete(df_sel, dry_run):
        called["count"] += 1
        return 1, 0, ["ok"]

    monkeypatch.setattr(delete_tab, "delete_files_from_rows", fake_delete, raising=False)  # type: ignore[attr-defined]

    df = pd.DataFrame(
        [
            {"title": "A", "library": "Lib1", "decision": "DELETE", "file_size": 100},
        ]
    )

    delete_tab.render(df, delete_dry_run=False, delete_require_confirm=False)  # type: ignore[attr-defined]

    assert called["count"] == 1