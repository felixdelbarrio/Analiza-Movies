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
        "se omiten tests de candidates.",
        allow_module_level=True,
    )

FRONTEND_DIR = PROJECT_ROOT / "frontend"
TABS_DIR = FRONTEND_DIR / "tabs"

CANDIDATES_PATH = TABS_DIR / "candidates.py"
COMPONENTS_PATH = FRONTEND_DIR / "components.py"

# Si no existe frontend/tabs/candidates.py, saltamos todo el módulo de tests
if not CANDIDATES_PATH.exists():
    pytest.skip(
        f"frontend/tabs/candidates.py no encontrado en {CANDIDATES_PATH}, "
        "se omiten tests de candidates.",
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
# Cargar primero frontend.components (porque candidates lo importa)
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
# Cargar ahora frontend.tabs.candidates
# -------------------------------------------------------------------
spec_cand = importlib.util.spec_from_file_location(
    "frontend.tabs.candidates",
    CANDIDATES_PATH,
)
if spec_cand is None or spec_cand.loader is None:
    raise ImportError(f"No se pudo crear spec para {CANDIDATES_PATH}")

candidates = importlib.util.module_from_spec(spec_cand)
sys.modules["frontend.tabs.candidates"] = candidates
spec_cand.loader.exec_module(candidates)  # type: ignore[assignment]


# -------------------------------------------------------------------
# Fixture para mockear streamlit
# -------------------------------------------------------------------


@pytest.fixture
def fake_streamlit(monkeypatch):
    """Parchea las funciones principales de streamlit usadas en candidates.render."""
    import streamlit as st

    # st.write / st.info / st.caption
    monkeypatch.setattr(st, "write", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "info", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None, raising=False)

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


def test_render_no_filtered(fake_streamlit, monkeypatch) -> None:
    """Si df_filtered no es DataFrame o está vacío, debe mostrar info y NO llamar a aggrid/render_detail."""
    # aggrid_with_row_click y render_detail_card NO deben llamarse
    def fake_aggrid(*a, **k):
        raise AssertionError("aggrid_with_row_click no debe llamarse cuando df_filtered es None/vacío")

    def fake_detail(*a, **k):
        raise AssertionError("render_detail_card no debe llamarse cuando df_filtered es None/vacío")

    monkeypatch.setattr(
        candidates,
        "aggrid_with_row_click",
        fake_aggrid,
        raising=False,
    )
    monkeypatch.setattr(
        candidates,
        "render_detail_card",
        fake_detail,
        raising=False,
    )

    df_all = pd.DataFrame([])

    # Caso df_filtered = None
    candidates.render(df_all, None)  # type: ignore[arg-type,attr-defined]

    # Caso df_filtered vacío
    df_empty = pd.DataFrame([])
    candidates.render(df_all, df_empty)  # type: ignore[attr-defined]


def test_render_filters_to_empty_delete_maybe(fake_streamlit, monkeypatch) -> None:
    """Si tras filtrar por DELETE/MAYBE no queda nada, debe mostrar info y no llamar a aggrid/render_detail."""
    def fake_aggrid(*a, **k):
        raise AssertionError("aggrid_with_row_click no debe llamarse si no hay DELETE/MAYBE")

    def fake_detail(*a, **k):
        raise AssertionError("render_detail_card no debe llamarse si no hay DELETE/MAYBE")

    monkeypatch.setattr(
        candidates,
        "aggrid_with_row_click",
        fake_aggrid,
        raising=False,
    )
    monkeypatch.setattr(
        candidates,
        "render_detail_card",
        fake_detail,
        raising=False,
    )

    df_all = pd.DataFrame([])
    # Solo KEEP → tras filtrar DELETE/MAYBE debe quedar vacío
    df_filtered = pd.DataFrame(
        [
            {
                "title": "Movie KEEP",
                "year": 2000,
                "library": "Lib1",
                "decision": "KEEP",
                "imdb_rating": 7.5,
                "imdb_votes": 1500,
                "file_size": 1_000,
            },
        ]
    )

    candidates.render(df_all, df_filtered)  # type: ignore[attr-defined]


def test_render_happy_path_and_calls(fake_streamlit, monkeypatch) -> None:
    """Camino normal: hay DELETE/MAYBE, se filtra y ordena, y se llama a aggrid y render_detail_card."""
    selected = {"title": "Bad Movie", "year": 1999, "library": "LibA"}

    calls = {
        "aggrid": 0,
        "detail": 0,
        "detail_row": None,
        "detail_prefix": None,
        "aggrid_df": None,
    }

    def fake_aggrid(df: pd.DataFrame, key_suffix: str) -> dict[str, Any]:
        calls["aggrid"] += 1
        calls["aggrid_df"] = df
        # Debe usar key_suffix "filtered"
        assert key_suffix == "filtered"
        return selected

    def fake_render_detail_card(row, button_key_prefix=None):
        calls["detail"] += 1
        calls["detail_row"] = row
        calls["detail_prefix"] = button_key_prefix

    monkeypatch.setattr(
        candidates,
        "aggrid_with_row_click",
        fake_aggrid,
        raising=False,
    )
    monkeypatch.setattr(
        candidates,
        "render_detail_card",
        fake_render_detail_card,
        raising=False,
    )

    df_all = pd.DataFrame([])  # no se usa dentro, pero la función lo acepta

    # Mezcla de decisiones
    df_filtered = pd.DataFrame(
        [
            {
                "title": "Bad Movie",
                "year": 1999,
                "library": "LibA",
                "decision": "DELETE",
                "imdb_rating": 3.0,
                "imdb_votes": 100,
                "file_size": 5_000,
            },
            {
                "title": "Maybe Movie",
                "year": 2005,
                "library": "LibB",
                "decision": "MAYBE",
                "imdb_rating": 5.0,
                "imdb_votes": 200,
                "file_size": 10_000,
            },
            {
                "title": "Keep Movie",
                "year": 2010,
                "library": "LibC",
                "decision": "KEEP",
                "imdb_rating": 8.0,
                "imdb_votes": 5_000,
                "file_size": 20_000,
            },
        ]
    )

    candidates.render(df_all, df_filtered)  # type: ignore[attr-defined]

    # Se han hecho las llamadas esperadas
    assert calls["aggrid"] == 1
    assert calls["detail"] == 1
    assert calls["detail_row"] == selected
    # Debe usar un prefix distinto a "all": "candidates"
    assert calls["detail_prefix"] == "candidates"

    # Verificamos que el DataFrame pasado a aggrid solo contiene DELETE/MAYBE
    df_passed = calls["aggrid_df"]
    assert isinstance(df_passed, pd.DataFrame)
    assert set(df_passed["decision"].unique()) <= {"DELETE", "MAYBE"}
    # KEEP no debe estar
    assert "KEEP" not in set(df_passed["decision"].unique())


def test_render_sorting_applied_when_columns_exist(fake_streamlit, monkeypatch) -> None:
    """Comprueba que se aplica la ordenación cuando existen las columnas de sort."""
    # No nos importa el valor devuelto, solo el orden del DataFrame que llega a aggrid
    captured_titles = []

    def fake_aggrid(df: pd.DataFrame, key_suffix: str):
        nonlocal captured_titles
        captured_titles = df["title"].tolist()
        return {"title": df.iloc[0]["title"]}

    monkeypatch.setattr(
        candidates,
        "aggrid_with_row_click",
        fake_aggrid,
        raising=False,
    )
    monkeypatch.setattr(
        candidates,
        "render_detail_card",
        lambda *a, **k: None,
        raising=False,
    )

    df_all = pd.DataFrame([])

    # Dos DELETE y un MAYBE con distintos ratings/votos/tamaño para forzar orden
    df_filtered = pd.DataFrame(
        [
            {
                "title": "Delete Low Rating",
                "year": 2000,
                "library": "Lib1",
                "decision": "DELETE",
                "imdb_rating": 4.0,
                "imdb_votes": 100,
                "file_size": 1_000,
            },
            {
                "title": "Delete High Rating",
                "year": 2001,
                "library": "Lib1",
                "decision": "DELETE",
                "imdb_rating": 7.0,
                "imdb_votes": 500,
                "file_size": 2_000,
            },
            {
                "title": "Maybe Mid",
                "year": 1999,
                "library": "Lib1",
                "decision": "MAYBE",
                "imdb_rating": 6.0,
                "imdb_votes": 300,
                "file_size": 3_000,
            },
        ]
    )

    candidates.render(df_all, df_filtered)  # type: ignore[attr-defined]

    # decisions asc (DELETE, DELETE, MAYBE), dentro de DELETE rating desc
    # Esperamos primero "Delete High Rating"
    assert captured_titles[0] == "Delete High Rating"
    # Y el último debería ser el MAYBE
    assert captured_titles[-1] == "Maybe Mid"