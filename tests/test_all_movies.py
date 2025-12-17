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
        "se omiten tests de all_movies.",
        allow_module_level=True,
    )

FRONTEND_DIR = PROJECT_ROOT / "frontend"
TABS_DIR = FRONTEND_DIR / "tabs"

ALL_MOVIES_PATH = TABS_DIR / "all_movies.py"
COMPONENTS_PATH = FRONTEND_DIR / "components.py"

# Si no existe frontend/tabs/all_movies.py, saltamos todo el módulo de tests
if not ALL_MOVIES_PATH.exists():
    pytest.skip(
        f"frontend/tabs/all_movies.py no encontrado en {ALL_MOVIES_PATH}, "
        "se omiten tests de all_movies.",
        allow_module_level=True,
    )

# -------------------------------------------------------------------
# Crear paquete sintético `frontend` y `frontend.tabs` si es necesario
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
# Cargar primero frontend.components (porque all_movies lo importa)
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
# Cargar ahora frontend.tabs.all_movies
# -------------------------------------------------------------------
spec_all = importlib.util.spec_from_file_location(
    "frontend.tabs.all_movies",
    ALL_MOVIES_PATH,
)
if spec_all is None or spec_all.loader is None:
    raise ImportError(f"No se pudo crear spec para {ALL_MOVIES_PATH}")

all_movies = importlib.util.module_from_spec(spec_all)
sys.modules["frontend.tabs.all_movies"] = all_movies
spec_all.loader.exec_module(all_movies)  # type: ignore[assignment]


# -------------------------------------------------------------------
# Fixture para mockear streamlit
# -------------------------------------------------------------------


@pytest.fixture
def fake_streamlit(monkeypatch):
    """Parchea las funciones principales de streamlit usadas en all_movies.render."""
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

    return st


# -------------------------------------------------------------------
# Tests de render()
# -------------------------------------------------------------------


def test_render_no_data(fake_streamlit, monkeypatch) -> None:
    """Si df_all no es DataFrame o está vacío, debe mostrar info y no llamar a aggrid/render_detail_card."""
    # aggrid_with_row_click no debe llamarse
    def fake_aggrid(*a, **k):
        raise AssertionError("aggrid_with_row_click no debe llamarse con df vacío")

    def fake_detail(*a, **k):
        raise AssertionError("render_detail_card no debe llamarse con df vacío")

    monkeypatch.setattr(
        all_movies,
        "aggrid_with_row_click",
        fake_aggrid,
        raising=False,
    )
    monkeypatch.setattr(
        all_movies,
        "render_detail_card",
        fake_detail,
        raising=False,
    )

    # Caso no-DataFrame
    all_movies.render([])  # type: ignore[arg-type]

    # Caso DataFrame vacío
    df_empty = pd.DataFrame([])
    all_movies.render(df_empty)


def test_render_happy_path(fake_streamlit, monkeypatch) -> None:
    """Camino normal: hay datos, se llama a aggrid_with_row_click y a render_detail_card con la selección."""
    selected = {"title": "Movie A", "year": 2000, "library": "Lib1"}

    calls: dict[str, Any] = {
        "aggrid": 0,
        "detail": 0,
        "detail_row": None,
        "detail_prefix": None,
    }

    def fake_aggrid(df, key_suffix: str):
        calls["aggrid"] += 1
        assert key_suffix == "all"
        assert isinstance(df, pd.DataFrame)
        # El DataFrame debe contener nuestras columnas originales (aunque ordenadas)
        assert {"title", "year", "library"}.issubset(df.columns)
        return selected

    def fake_render_detail_card(row, button_key_prefix=None):
        calls["detail"] += 1
        calls["detail_row"] = row
        calls["detail_prefix"] = button_key_prefix

    monkeypatch.setattr(
        all_movies,
        "aggrid_with_row_click",
        fake_aggrid,
        raising=False,
    )
    monkeypatch.setattr(
        all_movies,
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

    all_movies.render(df_all)

    assert calls["aggrid"] == 1
    assert calls["detail"] == 1
    assert calls["detail_row"] == selected
    # Debe usar el prefix "all" para la ficha de detalle
    assert calls["detail_prefix"] == "all"


def test_render_missing_sort_columns(fake_streamlit, monkeypatch) -> None:
    """Si faltan las columnas de ordenación, debe seguir funcionando y llamar a aggrid/render_detail_card."""
    selected = {"title": "Only Movie"}

    calls = {"aggrid": 0, "detail": 0}

    def fake_aggrid(df, key_suffix: str):
        calls["aggrid"] += 1
        assert key_suffix == "all"
        return selected

    def fake_render_detail_card(row, button_key_prefix=None):
        calls["detail"] += 1
        assert button_key_prefix == "all"

    monkeypatch.setattr(
        all_movies,
        "aggrid_with_row_click",
        fake_aggrid,
        raising=False,
    )
    monkeypatch.setattr(
        all_movies,
        "render_detail_card",
        fake_render_detail_card,
        raising=False,
    )

    # DataFrame sin decision/imdb_rating/imdb_votes/year
    df_all = pd.DataFrame(
        [
            {"title": "Only Movie", "library": "LibX"},
        ]
    )

    all_movies.render(df_all)

    assert calls["aggrid"] == 1
    assert calls["detail"] == 1