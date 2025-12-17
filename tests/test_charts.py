from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import altair as alt
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
        "se omiten tests de charts.",
        allow_module_level=True,
    )

FRONTEND_DIR = PROJECT_ROOT / "frontend"
TABS_DIR = FRONTEND_DIR / "tabs"

CHARTS_PATH = TABS_DIR / "charts.py"
DATA_UTILS_PATH = FRONTEND_DIR / "data_utils.py"

# Si no existe frontend/tabs/charts.py, saltamos todo el módulo de tests
if not CHARTS_PATH.exists():
    pytest.skip(
        f"frontend/tabs/charts.py no encontrado en {CHARTS_PATH}, "
        "se omiten tests de charts.",
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
# Cargar primero frontend.data_utils (porque charts lo importa)
# -------------------------------------------------------------------
spec_data = importlib.util.spec_from_file_location(
    "frontend.data_utils",
    DATA_UTILS_PATH,
)
if spec_data is None or spec_data.loader is None:
    raise ImportError(f"No se pudo crear spec para {DATA_UTILS_PATH}")

data_utils_mod = importlib.util.module_from_spec(spec_data)
sys.modules["frontend.data_utils"] = data_utils_mod
spec_data.loader.exec_module(data_utils_mod)  # type: ignore[assignment]

# -------------------------------------------------------------------
# Cargar ahora frontend.tabs.charts
# -------------------------------------------------------------------
spec_charts = importlib.util.spec_from_file_location(
    "frontend.tabs.charts",
    CHARTS_PATH,
)
if spec_charts is None or spec_charts.loader is None:
    raise ImportError(f"No se pudo crear spec para {CHARTS_PATH}")

charts = importlib.util.module_from_spec(spec_charts)
sys.modules["frontend.tabs.charts"] = charts
spec_charts.loader.exec_module(charts)  # type: ignore[assignment]


# -------------------------------------------------------------------
# Fixture para mockear streamlit
# -------------------------------------------------------------------


@pytest.fixture
def fake_streamlit(monkeypatch):
    """Parchea las funciones principales de streamlit usadas en charts.render."""
    import streamlit as st

    # write / info / markdown → no-op
    monkeypatch.setattr(st, "write", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "info", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None, raising=False)

    # selectbox → devolver siempre la primera opción por defecto
    def fake_selectbox(label, options, *args, **kwargs):
        if isinstance(options, (list, tuple)) and options:
            return options[0]
        return None

    monkeypatch.setattr(st, "selectbox", fake_selectbox, raising=False)

    # slider → devolver el valor por defecto
    def fake_slider(label, min_val, max_val, value, *args, **kwargs):
        return value

    monkeypatch.setattr(st, "slider", fake_slider, raising=False)

    # altair_chart → no-op por defecto
    monkeypatch.setattr(
        st,
        "altair_chart",
        lambda *a, **k: None,
        raising=False,
    )

    return st


# -------------------------------------------------------------------
# Tests de helpers: _requires_columns
# -------------------------------------------------------------------


def test_requires_columns_missing(fake_streamlit, monkeypatch) -> None:
    """_requires_columns debe devolver False y llamar a st.info si faltan columnas."""
    import streamlit as st

    info_calls: list[str] = []

    def fake_info(msg, *a, **k):
        info_calls.append(str(msg))

    monkeypatch.setattr(st, "info", fake_info, raising=False)

    f = charts._requires_columns  # type: ignore[attr-defined]
    df = pd.DataFrame({"a": [1, 2, 3]})

    ok = f(df, ["a", "b"])
    assert ok is False
    assert info_calls
    assert "Faltan columna(s) requerida(s)" in info_calls[0]


def test_requires_columns_ok(fake_streamlit, monkeypatch) -> None:
    """_requires_columns debe devolver True si todas las columnas existen."""
    import streamlit as st

    info_calls: list[str] = []

    def fake_info(msg, *a, **k):
        info_calls.append(str(msg))

    monkeypatch.setattr(st, "info", fake_info, raising=False)

    f = charts._requires_columns  # type: ignore[attr-defined]
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    ok = f(df, ["a", "b"])
    assert ok is True
    assert info_calls == []


# -------------------------------------------------------------------
# Tests de helper: _chart
# -------------------------------------------------------------------


def test_chart_calls_altair_chart_with_width_stretch(fake_streamlit, monkeypatch) -> None:
    """_chart debe llamar a st.altair_chart con width='stretch'."""
    import streamlit as st

    called = {"count": 0, "kwargs": None}

    def fake_altair_chart(chart_obj, **kwargs):
        called["count"] += 1
        called["kwargs"] = kwargs

    monkeypatch.setattr(st, "altair_chart", fake_altair_chart, raising=False)

    dummy_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    ch = alt.Chart(dummy_df).mark_bar().encode(x="x:Q", y="y:Q")

    charts._chart(ch)  # type: ignore[attr-defined]

    assert called["count"] == 1
    assert isinstance(called["kwargs"], dict)
    assert called["kwargs"].get("width") == "stretch"


# -------------------------------------------------------------------
# Tests de render(): rutas principales
# -------------------------------------------------------------------


def test_render_empty_df_shows_info(fake_streamlit, monkeypatch) -> None:
    """Si df_all está vacío, render debe mostrar info y no fallar."""
    import streamlit as st

    info_calls: list[str] = []

    def fake_info(msg, *a, **k):
        info_calls.append(str(msg))

    monkeypatch.setattr(st, "info", fake_info, raising=False)

    df_empty = pd.DataFrame([])
    charts.render(df_empty)  # type: ignore[attr-defined]

    assert info_calls


def test_render_distribution_by_decision_happy_path(fake_streamlit, monkeypatch) -> None:
    """Vista 'Distribución por decisión' con datos válidos debe llamar a _chart."""
    import streamlit as st

    def selectbox(label, options, *a, **k):
        return "Distribución por decisión"

    monkeypatch.setattr(st, "selectbox", selectbox, raising=False)

    chart_calls = {"count": 0}

    def fake_chart(ch):
        chart_calls["count"] += 1

    monkeypatch.setattr(charts, "_chart", fake_chart, raising=False)  # type: ignore[attr-defined]

    df = pd.DataFrame(
        [
            {"title": "A", "decision": "DELETE"},
            {"title": "B", "decision": "KEEP"},
        ]
    )

    charts.render(df)  # type: ignore[attr-defined]

    assert chart_calls["count"] == 1


def test_render_distribution_by_decision_no_data_after_group(fake_streamlit, monkeypatch) -> None:
    """Si tras agrupar no hay datos, no debe llamar a _chart."""
    import streamlit as st

    def selectbox(label, options, *a, **k):
        return "Distribución por decisión"

    monkeypatch.setattr(st, "selectbox", selectbox, raising=False)

    chart_calls = {"count": 0}

    def fake_chart(ch):
        chart_calls["count"] += 1

    monkeypatch.setattr(charts, "_chart", fake_chart, raising=False)  # type: ignore[attr-defined]

    df = pd.DataFrame({"title": [], "decision": []})

    charts.render(df)  # type: ignore[attr-defined]

    assert chart_calls["count"] == 0


def test_render_genres_view_uses_explode_and_chart(fake_streamlit, monkeypatch) -> None:
    """Vista 'Distribución por género (OMDb)' debe llamar a explode_genres_from_omdb_json y luego a _chart si hay datos."""
    import streamlit as st

    def selectbox(label, options, *a, **k):
        return "Distribución por género (OMDb)"

    monkeypatch.setattr(st, "selectbox", selectbox, raising=False)

    explode_calls = {"count": 0}

    def fake_explode(df):
        explode_calls["count"] += 1
        return pd.DataFrame(
            [
                {"genre": "Action", "decision": "DELETE", "title": "X"},
                {"genre": "Drama", "decision": "KEEP", "title": "Y"},
            ]
        )

    monkeypatch.setattr(
        charts,
        "explode_genres_from_omdb_json",
        fake_explode,
        raising=False,
    )  # type: ignore[attr-defined]

    chart_calls = {"count": 0}

    def fake_chart(ch):
        chart_calls["count"] += 1

    monkeypatch.setattr(charts, "_chart", fake_chart, raising=False)  # type: ignore[attr-defined]

    df_all = pd.DataFrame(
        [
            {"omdb_json": '{"Genre": "Action, Drama"}', "title": "X", "decision": "DELETE"},
        ]
    )

    charts.render(df_all)  # type: ignore[attr-defined]

    assert explode_calls["count"] == 1
    assert chart_calls["count"] == 1


def test_render_genres_view_empty_after_explode(fake_streamlit, monkeypatch) -> None:
    """Si explode_genres_from_omdb_json devuelve vacío, no debe llamar a _chart."""
    import streamlit as st

    def selectbox(label, options, *a, **k):
        return "Distribución por género (OMDb)"

    monkeypatch.setattr(st, "selectbox", selectbox, raising=False)

    def fake_explode(df):
        return pd.DataFrame(columns=["genre", "decision", "title"])

    monkeypatch.setattr(
        charts,
        "explode_genres_from_omdb_json",
        fake_explode,
        raising=False,
    )  # type: ignore[attr-defined]

    chart_calls = {"count": 0}

    def fake_chart(ch):
        chart_calls["count"] += 1

    monkeypatch.setattr(charts, "_chart", fake_chart, raising=False)  # type: ignore[attr-defined]

    df_all = pd.DataFrame(
        [
            {"omdb_json": "{}", "title": "X", "decision": "DELETE"},
        ]
    )

    charts.render(df_all)  # type: ignore[attr-defined]

    assert chart_calls["count"] == 0


def test_render_directors_ranking(fake_streamlit, monkeypatch) -> None:
    """Vista 'Ranking de directores' con datos válidos debe llamar a _chart."""
    import streamlit as st

    # Forzamos la vista
    def selectbox(label, options, *a, **k):
        return "Ranking de directores"

    monkeypatch.setattr(st, "selectbox", selectbox, raising=False)

    # 👈 IMPORTANTE: bajar el umbral de mínimo nº de pelis a 1
    def fake_slider(label, min_val, max_val, value, *args, **kwargs):
        # Tanto para "mínimo nº de películas" como para "Top N" nos vale devolver 1 o el default
        if "Mínimo nº de películas" in label:
            return 1
        return value

    monkeypatch.setattr(st, "slider", fake_slider, raising=False)

    chart_calls = {"count": 0}

    def fake_chart(ch):
        chart_calls["count"] += 1

    monkeypatch.setattr(charts, "_chart", fake_chart, raising=False)  # type: ignore[attr-defined]

    # Dos películas con mismo director (suficiente si min_movies=1)
    df_all = pd.DataFrame(
        [
            {
                "title": "Movie 1",
                "imdb_rating": 7.0,
                "omdb_json": '{"Director": "John Doe"}',
            },
            {
                "title": "Movie 2",
                "imdb_rating": 8.0,
                "omdb_json": '{"Director": "John Doe"}',
            },
        ]
    )

    charts.render(df_all)  # type: ignore[attr-defined]

    assert chart_calls["count"] == 1


def test_render_word_counts_view(fake_streamlit, monkeypatch) -> None:
    """Vista 'Palabras más frecuentes en títulos DELETE/MAYBE' debe usar build_word_counts y _chart."""
    import streamlit as st

    def selectbox(label, options, *a, **k):
        return "Palabras más frecuentes en títulos DELETE/MAYBE"

    monkeypatch.setattr(st, "selectbox", selectbox, raising=False)

    build_calls = {"count": 0}

    def fake_build(df, decisions):
        build_calls["count"] += 1
        return pd.DataFrame(
            [
                {"word": "bad", "decision": "DELETE", "count": 10},
                {"word": "boring", "decision": "MAYBE", "count": 5},
            ]
        )

    monkeypatch.setattr(
        charts,
        "build_word_counts",
        fake_build,
        raising=False,
    )  # type: ignore[attr-defined]

    chart_calls = {"count": 0}

    def fake_chart(ch):
        chart_calls["count"] += 1

    monkeypatch.setattr(charts, "_chart", fake_chart, raising=False)  # type: ignore[attr-defined]

    df_all = pd.DataFrame(
        [
            {"title": "Bad Movie", "decision": "DELETE"},
            {"title": "Maybe Boring Movie", "decision": "MAYBE"},
        ]
    )

    charts.render(df_all)  # type: ignore[attr-defined]

    assert build_calls["count"] == 1
    assert chart_calls["count"] == 1


def test_render_scoring_rule_distribution(fake_streamlit, monkeypatch) -> None:
    """Vista 'Distribución por scoring_rule' debe agrupar y llamar a _chart."""
    import streamlit as st

    def selectbox(label, options, *a, **k):
        return "Distribución por scoring_rule"

    monkeypatch.setattr(st, "selectbox", selectbox, raising=False)

    chart_calls = {"count": 0}

    def fake_chart(ch):
        chart_calls["count"] += 1

    monkeypatch.setattr(charts, "_chart", fake_chart, raising=False)  # type: ignore[attr-defined]

    df_all = pd.DataFrame(
        [
            {
                "title": "Movie A",
                "decision": "DELETE",
                "scoring_rule": "AUTO_LOW",
            },
            {
                "title": "Movie B",
                "decision": "KEEP",
                "scoring_rule": "AUTO_HIGH",
            },
        ]
    )

    charts.render(df_all)  # type: ignore[attr-defined]

    assert chart_calls["count"] == 1