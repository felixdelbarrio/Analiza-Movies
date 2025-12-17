from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Mapping

import pandas as pd
import pytest

# -------------------------------------------------------------------
# Carga dinámica de frontend/data_utils.py y frontend/components.py
# -------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
DATA_UTILS_PATH = FRONTEND_DIR / "data_utils.py"
COMPONENTS_PATH = FRONTEND_DIR / "components.py"

# Aseguramos un paquete sintético "frontend"
import types

frontend_pkg = types.ModuleType("frontend")
frontend_pkg.__path__ = [str(FRONTEND_DIR)]
sys.modules.setdefault("frontend", frontend_pkg)

# Cargamos frontend.data_utils primero (lo necesita components)
spec_du = importlib.util.spec_from_file_location("frontend.data_utils", DATA_UTILS_PATH)
if spec_du is None or spec_du.loader is None:
    raise ImportError(f"No se pudo crear spec para {DATA_UTILS_PATH}")
data_utils_mod = importlib.util.module_from_spec(spec_du)
sys.modules["frontend.data_utils"] = data_utils_mod
spec_du.loader.exec_module(data_utils_mod)  # type: ignore[assignment]

# Ahora cargamos frontend.components
spec_comp = importlib.util.spec_from_file_location(
    "frontend.components",
    COMPONENTS_PATH,
)
if spec_comp is None or spec_comp.loader is None:
    raise ImportError(f"No se pudo crear spec para {COMPONENTS_PATH}")

components = importlib.util.module_from_spec(spec_comp)
sys.modules["frontend.components"] = components
spec_comp.loader.exec_module(components)  # type: ignore[assignment]


# -------------------------------------------------------------------
# Helpers puros: _normalize_selected_rows
# -------------------------------------------------------------------


def test_normalize_selected_rows_none() -> None:
    f = components._normalize_selected_rows  # type: ignore[attr-defined]
    assert f(None) == []


def test_normalize_selected_rows_dataframe() -> None:
    f = components._normalize_selected_rows  # type: ignore[attr-defined]
    df = pd.DataFrame([{"a": 1}, {"a": 2}])
    out = f(df)
    assert isinstance(out, list)
    assert isinstance(out[0], Mapping)
    assert out[0]["a"] == 1
    assert out[1]["a"] == 2


def test_normalize_selected_rows_list_of_dicts() -> None:
    f = components._normalize_selected_rows  # type: ignore[attr-defined]
    data = [{"x": 1}, {"x": 2}]
    out = f(data)
    assert isinstance(out, list)
    assert out == data
    assert out is not data  # no exigimos misma instancia


def test_normalize_selected_rows_tuple_of_series() -> None:
    f = components._normalize_selected_rows  # type: ignore[attr-defined]
    s1 = pd.Series({"x": 1})
    s2 = pd.Series({"x": 2})
    out = f((s1, s2))
    assert isinstance(out, list)
    assert out[0] is s1
    assert out[1] is s2


def test_normalize_selected_rows_single_mapping() -> None:
    f = components._normalize_selected_rows  # type: ignore[attr-defined]
    d = {"a": 123}
    out = f(d)
    assert isinstance(out, list)
    assert out[0] is d


def test_normalize_selected_rows_generic_iterable() -> None:
    f = components._normalize_selected_rows  # type: ignore[attr-defined]

    class CustomIter:
        def __iter__(self):
            yield {"a": 1}
            yield {"a": 2}

    ci = CustomIter()
    out = f(ci)
    assert isinstance(out, list)
    assert out[0]["a"] == 1
    assert out[1]["a"] == 2


def test_normalize_selected_rows_fallback_non_iterable() -> None:
    f = components._normalize_selected_rows  # type: ignore[attr-defined]

    class NotIterable:
        pass

    obj = NotIterable()
    out = f(obj)
    assert isinstance(out, list)
    assert out[0] is obj


# -------------------------------------------------------------------
# Helpers puros: _normalize_row_to_dict
# -------------------------------------------------------------------


def test_normalize_row_to_dict_none() -> None:
    f = components._normalize_row_to_dict  # type: ignore[attr-defined]
    assert f(None) is None


def test_normalize_row_to_dict_series() -> None:
    f = components._normalize_row_to_dict  # type: ignore[attr-defined]
    s = pd.Series({"a": 1, "b": 2})
    out = f(s)
    assert isinstance(out, dict)
    assert out["a"] == 1
    assert out["b"] == 2


def test_normalize_row_to_dict_mapping() -> None:
    f = components._normalize_row_to_dict  # type: ignore[attr-defined]
    m = {"x": 10}
    out = f(m)
    assert out == {"x": 10}
    assert out is not m  # copia


def test_normalize_row_to_dict_iterable_pairs() -> None:
    f = components._normalize_row_to_dict  # type: ignore[attr-defined]
    tpl = [("k1", 1), ("k2", 2)]
    out = f(tpl)
    assert out == {"k1": 1, "k2": 2}


def test_normalize_row_to_dict_invalid_returns_none() -> None:
    f = components._normalize_row_to_dict  # type: ignore[attr-defined]

    class Bad:
        def __iter__(self):
            raise RuntimeError("no iteration")

    out = f(Bad())
    assert out is None


# -------------------------------------------------------------------
# Helpers puros: _get_from_omdb_or_row
# -------------------------------------------------------------------


def test_get_from_omdb_or_row_prefers_row_value() -> None:
    f = components._get_from_omdb_or_row  # type: ignore[attr-defined]
    row = {"Rated": "R"}
    omdb = {"Rated": "PG-13"}
    assert f(row, omdb, "Rated") == "R"


def test_get_from_omdb_or_row_falls_back_to_omdb() -> None:
    f = components._get_from_omdb_or_row  # type: ignore[attr-defined]
    row = {}
    omdb = {"Rated": "PG-13"}
    assert f(row, omdb, "Rated") == "PG-13"


def test_get_from_omdb_or_row_missing_everywhere() -> None:
    f = components._get_from_omdb_or_row  # type: ignore[attr-defined]
    row = {}
    omdb = {}
    assert f(row, omdb, "Rated") is None


# -------------------------------------------------------------------
# Helpers puros: _safe_number_to_str
# -------------------------------------------------------------------


def test_safe_number_to_str_basic() -> None:
    f = components._safe_number_to_str  # type: ignore[attr-defined]
    assert f(5) == "5"
    assert f(3.14) == "3.14"
    assert f("7.8") == "7.8"


def test_safe_number_to_str_nan_and_none() -> None:
    f = components._safe_number_to_str  # type: ignore[attr-defined]
    assert f(None) == "N/A"
    assert f(float("nan")) == "N/A"


def test_safe_number_to_str_weird_object() -> None:
    f = components._safe_number_to_str  # type: ignore[attr-defined]

    class Broken:
        def __str__(self):
            raise RuntimeError("no str")

    assert f(Broken()) == "N/A"


# -------------------------------------------------------------------
# Helpers puros: _safe_votes
# -------------------------------------------------------------------


def test_safe_votes_with_int_and_float() -> None:
    f = components._safe_votes  # type: ignore[attr-defined]
    assert f(1234) == "1,234"
    assert f(5678.9) == "5,678"


def test_safe_votes_with_string_with_commas() -> None:
    f = components._safe_votes  # type: ignore[attr-defined]
    assert f("1,000") == "1,000"
    assert f("2500") == "2,500"


def test_safe_votes_with_none_and_nan() -> None:
    f = components._safe_votes  # type: ignore[attr-defined]
    assert f(None) == "N/A"
    assert f(float("nan")) == "N/A"


def test_safe_votes_with_invalid_string() -> None:
    f = components._safe_votes  # type: ignore[attr-defined]
    assert f("not-a-number") == "N/A"


# -------------------------------------------------------------------
# Helpers puros: _is_nonempty_str
# -------------------------------------------------------------------


def test_is_nonempty_str_basic() -> None:
    f = components._is_nonempty_str  # type: ignore[attr-defined]
    assert f("Hello")
    assert f("  hi  ")
    assert f(123)  # se convierte a "123"


def test_is_nonempty_str_empty_and_none() -> None:
    f = components._is_nonempty_str  # type: ignore[attr-defined]
    assert not f(None)
    assert not f("")
    assert not f("   ")


def test_is_nonempty_str_nan_and_none_strings() -> None:
    f = components._is_nonempty_str  # type: ignore[attr-defined]
    assert not f("nan")
    assert not f("NaN")
    assert not f("NONE")
    assert not f("none")


# -------------------------------------------------------------------
# aggrid_with_row_click (con AgGrid y streamlit parcheados)
# -------------------------------------------------------------------


class DummyStInfoCalls:
    def __init__(self) -> None:
        self.called = False
        self.args = None

    def __call__(self, *args, **kwargs) -> None:
        self.called = True
        self.args = (args, kwargs)


def test_aggrid_with_row_click_empty_df(monkeypatch) -> None:
    import streamlit as st

    info_tracker = DummyStInfoCalls()
    monkeypatch.setattr(st, "info", info_tracker)

    # AgGrid no debería ni llamarse porque el df está vacío
    def fake_AgGrid(*args, **kwargs):
        raise AssertionError("AgGrid no debe ser llamado con df vacío")

    monkeypatch.setattr(components, "AgGrid", fake_AgGrid)

    df = pd.DataFrame([])
    result = components.aggrid_with_row_click(df, "x")  # type: ignore[attr-defined]
    assert result is None
    assert info_tracker.called


def test_aggrid_with_row_click_with_selection_dict(monkeypatch) -> None:
    import streamlit as st

    monkeypatch.setattr(st, "info", lambda *a, **k: None, raising=False)

    def fake_AgGrid(df, gridOptions=None, **kwargs):
        return {
            "selected_rows": [
                {"title": "Movie A", "year": 2000, "library": "Lib1"},
            ]
        }

    monkeypatch.setattr(components, "AgGrid", fake_AgGrid)

    df = pd.DataFrame(
        [
            {"title": "Movie A", "year": 2000, "library": "Lib1"},
            {"title": "Movie B", "year": 2001, "library": "Lib2"},
        ]
    )

    row = components.aggrid_with_row_click(df, "test")  # type: ignore[attr-defined]
    assert isinstance(row, dict)
    assert row["title"] == "Movie A"
    assert row["year"] == 2000
    assert row["library"] == "Lib1"


def test_aggrid_with_row_click_selected_rows_dataframe(monkeypatch) -> None:
    import streamlit as st

    monkeypatch.setattr(st, "info", lambda *a, **k: None, raising=False)

    def fake_AgGrid(df, gridOptions=None, **kwargs):
        return {
            "selected_rows": pd.DataFrame(
                [{"title": "X", "year": 1999, "library": "L"}]
            )
        }

    monkeypatch.setattr(components, "AgGrid", fake_AgGrid)

    df = pd.DataFrame([{"title": "X", "year": 1999, "library": "L"}])
    row = components.aggrid_with_row_click(df, "dfsel")  # type: ignore[attr-defined]
    assert isinstance(row, dict)
    assert row["title"] == "X"


# -------------------------------------------------------------------
# render_detail_card y render_modal (smoke tests con Streamlit mockeado)
# -------------------------------------------------------------------


@pytest.fixture
def fake_streamlit(monkeypatch):
    """Parchea las funciones de streamlit usadas en render_detail_card/render_modal."""
    import streamlit as st

    # info / warning / write / markdown / image / metric / code / video / json
    monkeypatch.setattr(st, "info", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "warning", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "write", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "image", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "metric", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "code", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "video", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "json", lambda *a, **k: None, raising=False)

    # columns devuelve N objetos dummy con context manager y método metric
    class DummyCol:
        def __enter__(self, *a, **k):
            return self

        def __exit__(self, *a, **k):
            return False

        # Para llamadas tipo m1.metric(...)
        def metric(self, *args, **kwargs):
            return None

    def fake_columns(spec):
        # Streamlit acepta tanto un int como una lista de anchos
        if isinstance(spec, int):
            n = spec
        else:
            try:
                n = len(spec)
            except TypeError:
                n = 1
        return [DummyCol() for _ in range(n)]

    monkeypatch.setattr(st, "columns", fake_columns, raising=False)

    # button
    monkeypatch.setattr(st, "button", lambda *a, **k: False, raising=False)

    # expander
    class DummyExpander:
        def __enter__(self, *a, **k):
            return self

        def __exit__(self, *a, **k):
            return False

    monkeypatch.setattr(st, "expander", lambda *a, **k: DummyExpander(), raising=False)

    # session_state
    if not hasattr(st, "session_state"):
        st.session_state = {}
    else:
        st.session_state.clear()

    # experimental_rerun
    monkeypatch.setattr(st, "experimental_rerun", lambda: None, raising=False)

    return st


def test_render_detail_card_none(fake_streamlit) -> None:
    # Solo comprobamos que no lanza excepción
    components.render_detail_card(None)  # type: ignore[attr-defined]


def test_render_detail_card_minimal_row(fake_streamlit) -> None:
    row = {
        "title": "My Movie",
        "year": 2020,
        "library": "Lib",
        "decision": "DELETE",
        "reason": "Test reason",
        "imdb_rating": 7.5,
        "imdb_votes": 1000,
        "rt_score": 80,
        "file": "/tmp/file.mkv",
        "file_size": 1024**3,
    }
    components.render_detail_card(row, show_modal_button=False)  # type: ignore[attr-defined]


def test_render_detail_card_with_omdb_json(fake_streamlit) -> None:
    row = {
        "title": "Movie OMDb",
        "year": 2010,
        "library": "Lib2",
        "decision": "MAYBE",
        "reason": "Reason",
        "omdb_json": '{"Rated": "PG-13", "Runtime": "120 min", "Genre": "Action"}',
    }
    components.render_detail_card(row, show_modal_button=False)  # type: ignore[attr-defined]


def test_render_modal_states(fake_streamlit, monkeypatch) -> None:
    import streamlit as st

    # 1) modal_open = False → no hace nada
    st.session_state["modal_open"] = False
    components.render_modal()  # type: ignore[attr-defined]

    # 2) modal_open = True sin modal_row → no hace nada relevante
    st.session_state["modal_open"] = True
    st.session_state["modal_row"] = None
    components.render_modal()  # type: ignore[attr-defined]

    # 3) modal_open = True con fila → debe llamar a render_detail_card
    called = {"count": 0}

    def fake_render_detail_card(row, show_modal_button=True, button_key_prefix=None):
        called["count"] += 1

    monkeypatch.setattr(
        components,
        "render_detail_card",
        fake_render_detail_card,
    )

    st.session_state["modal_row"] = {"title": "X"}
    components.render_modal()  # type: ignore[attr-defined]
    assert called["count"] == 1