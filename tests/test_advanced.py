from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import pandas as pd
import pytest

# -------------------------------------------------------------------
# Localizar proyecto y rutas de frontend de forma robusta
# -------------------------------------------------------------------

THIS_FILE = pathlib.Path(__file__).resolve()

# Buscamos hacia arriba hasta encontrar un directorio que contenga "frontend"
PROJECT_ROOT: pathlib.Path | None = None
for parent in THIS_FILE.parents:
    candidate = parent / "frontend"
    if candidate.exists() and candidate.is_dir():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    pytest.skip(
        "No se ha encontrado un directorio 'frontend' en ningún padre del test; "
        "se omiten tests de advanced.",
        allow_module_level=True,
    )

FRONTEND_DIR = PROJECT_ROOT / "frontend"
TABS_DIR = FRONTEND_DIR / "tabs"

ADVANCED_PATH = TABS_DIR / "advanced.py"
COMPONENTS_PATH = FRONTEND_DIR / "components.py"

# Si no existe frontend/tabs/advanced.py, saltamos todo el módulo de tests
if not ADVANCED_PATH.exists():
    pytest.skip(
        f"frontend/tabs/advanced.py no encontrado en {ADVANCED_PATH}, "
        "se omiten tests de advanced.",
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
# Cargar primero frontend.components (porque advanced lo importa)
# -------------------------------------------------------------------
spec_comp = importlib.util.spec_from_file_location(
    "frontend.components",
    COMPONENTS_PATH,
)
if spec_comp is None or spec_comp.loader is None:
    raise ImportError(f"No se pudo crear spec para {COMPONENTS_PATH}")

components_mod = importlib.util.module_from_spec(spec_comp)
sys.modules["frontend.components"] = components_mod
spec_comp.loader.exec_module(components_mod)  # type: ignore[assignment]

# -------------------------------------------------------------------
# Cargar ahora frontend.tabs.advanced
# -------------------------------------------------------------------
spec_adv = importlib.util.spec_from_file_location(
    "frontend.tabs.advanced",
    ADVANCED_PATH,
)
if spec_adv is None or spec_adv.loader is None:
    raise ImportError(f"No se pudo crear spec para {ADVANCED_PATH}")

advanced = importlib.util.module_from_spec(spec_adv)
sys.modules["frontend.tabs.advanced"] = advanced
spec_adv.loader.exec_module(advanced)  # type: ignore[assignment]


# -------------------------------------------------------------------
# Tests de helpers: _safe_unique_sorted
# -------------------------------------------------------------------


def test_safe_unique_sorted_column_missing() -> None:
    f = advanced._safe_unique_sorted  # type: ignore[attr-defined]
    df = pd.DataFrame({"other": [1, 2, 3]})
    assert f(df, "library") == []


def test_safe_unique_sorted_basic_and_cleanup() -> None:
    f = advanced._safe_unique_sorted  # type: ignore[attr-defined]
    df = pd.DataFrame(
        {
            "library": [
                " Movies ",
                "Series",
                "Movies",
                "",
                "   ",
                None,
                "Documentales",
                "movies",  # string distinto
            ]
        }
    )
    result = f(df, "library")
    # No deben aparecer cadenas vacías ni espacios
    assert "" not in result
    assert "   " not in result
    # Valores únicos limpios
    assert set(result) == {"Movies", "Series", "Documentales", "movies"}


# -------------------------------------------------------------------
# Tests de helpers: _ensure_numeric_column
# -------------------------------------------------------------------


def test_ensure_numeric_column_missing_column() -> None:
    f = advanced._ensure_numeric_column  # type: ignore[attr-defined]
    df = pd.DataFrame({"a": [1, 2, 3]})
    s = f(df, "no_col")
    assert isinstance(s, pd.Series)
    assert list(s.index) == list(df.index)
    assert (s == 0.0).all()


def test_ensure_numeric_column_existing_mixed_values() -> None:
    f = advanced._ensure_numeric_column  # type: ignore[attr-defined]
    df = pd.DataFrame(
        {
            "imdb_rating": [7.5, "8.0", None, "bad", 0],
        }
    )
    s = f(df, "imdb_rating")
    assert list(s) == [7.5, 8.0, 0.0, 0.0, 0.0]
    assert s.dtype == "float64"


# -------------------------------------------------------------------
# Fixture para mockear streamlit en tests de render
# -------------------------------------------------------------------


@pytest.fixture
def fake_streamlit(monkeypatch):
    """Parchea las funciones principales de streamlit usadas en advanced.render."""
    import streamlit as st

    # st.write / st.info
    monkeypatch.setattr(st, "write", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "info", lambda *a, **k: None, raising=False)

    # columns devuelve N objetos dummy con context manager y los métodos usados
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

    # multiselect: si hay default, devolver default; si no, lista vacía
    def fake_multiselect(label, options, *args, **kwargs):
        if "default" in kwargs:
            return kwargs["default"]
        return []

    monkeypatch.setattr(st, "multiselect", fake_multiselect, raising=False)

    # slider: devolver su valor por defecto (3er/4º arg)
    def fake_slider(label, min_val, max_val, value, *args, **kwargs):
        return value

    monkeypatch.setattr(st, "slider", fake_slider, raising=False)

    return st


# -------------------------------------------------------------------
# Tests de render()
# -------------------------------------------------------------------


def test_render_no_data(fake_streamlit, monkeypatch) -> None:
    """Si df_all no es DataFrame o está vacío, debe mostrar info y no fallar."""
    # aggrid_with_row_click no debe llamarse
    def fake_aggrid(*a, **k):
        raise AssertionError("aggrid_with_row_click no debe llamarse con df vacío")

    monkeypatch.setattr(
        advanced,
        "aggrid_with_row_click",
        fake_aggrid,
        raising=False,
    )

    # Caso no-DataFrame
    advanced.render([])  # type: ignore[arg-type,attr-defined]

    # Caso DataFrame vacío
    df_empty = pd.DataFrame([])
    advanced.render(df_empty)  # type: ignore[attr-defined]


def test_render_happy_path(fake_streamlit, monkeypatch) -> None:
    """Camino normal: hay datos, se aplican filtros por defecto y se llama a render_detail_card."""
    selected = {"title": "Movie A", "year": 2000, "library": "Lib1"}

    def fake_aggrid(df, key_suffix: str):
        assert isinstance(df, pd.DataFrame)
        return selected

    monkeypatch.setattr(
        advanced,
        "aggrid_with_row_click",
        fake_aggrid,
        raising=False,
    )

    calls = {"count": 0, "last_row": None}

    def fake_render_detail_card(row, button_key_prefix=None):
        calls["count"] += 1
        calls["last_row"] = row

    monkeypatch.setattr(
        advanced,
        "render_detail_card",
        fake_render_detail_card,
        raising=False,
    )

    df_all = pd.DataFrame(
        [
            {
                "title": "Movie A",
                "year": 2000,
                "library": "Lib1",
                "decision": "DELETE",
                "imdb_rating": 7.5,
                "imdb_votes": 1500,
            },
            {
                "title": "Movie B",
                "year": 2010,
                "library": "Lib2",
                "decision": "KEEP",
                "imdb_rating": 8.0,
                "imdb_votes": 5000,
            },
        ]
    )

    advanced.render(df_all)  # type: ignore[attr-defined]

    assert calls["count"] == 1
    assert calls["last_row"] == selected


def test_render_filters_to_empty(fake_streamlit, monkeypatch) -> None:
    """Si los filtros dejan el DataFrame vacío, no debe llamar a aggrid_with_row_click."""
    import streamlit as st

    # Fuerza filtro de decisión que deje todo vacío
    def custom_multiselect(label, options, *args, **kwargs):
        if label == "Decisión":
            return ["KEEP"]  # ningún row DELETE/MAYBE/UNKNOWN en nuestro df
        if "default" in kwargs:
            return kwargs["default"]
        return []

    monkeypatch.setattr(st, "multiselect", custom_multiselect, raising=False)

    def fake_aggrid(*a, **k):
        raise AssertionError("aggrid_with_row_click no debe llamarse si no hay resultados")

    monkeypatch.setattr(
        advanced,
        "aggrid_with_row_click",
        fake_aggrid,
        raising=False,
    )

    monkeypatch.setattr(
        advanced,
        "render_detail_card",
        lambda *a, **k: None,
        raising=False,
    )

    df_all = pd.DataFrame(
        [
            {
                "title": "Movie A",
                "year": 2000,
                "library": "Lib1",
                "decision": "DELETE",
                "imdb_rating": 7.5,
                "imdb_votes": 1500,
            },
            {
                "title": "Movie B",
                "year": 2010,
                "library": "Lib2",
                "decision": "DELETE",
                "imdb_rating": 6.0,
                "imdb_votes": 200,
            },
        ]
    )

    advanced.render(df_all)  # type: ignore[attr-defined]